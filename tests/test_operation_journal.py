"""EXP-014: the crash matrix, on a fake backend (arch §14, FW-REQ-008B).

One property carries all of this:

    **an unknown outcome never becomes a known one without evidence.**

The fault injection points FW-REQ-008B's acceptance criteria name are before
dispatch, after dispatch but before acknowledgement, after acknowledgement but
before checkpoint, and during compensation. Each has a test below, and each one
asks the same question: after the crash, what does the durable record say, and
can a reader tell what may have happened?

Everything here runs against a fake backend and a temporary journal. Nothing in
this slice enables a write.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass

import pytest

from fastworkflow.command_contract import (
    CommandEffectContract,
    ContractRegistry,
    ReconciliationDeclaration,
    binding_digest,
    logical_call_key,
)
from fastworkflow.operation_journal import (
    IdempotencyConflict,
    JournalError,
    JournalScope,
    JournalUnavailable,
    OperationJournal,
    OwnershipLost,
    ReconciliationRequired,
)
from fastworkflow.strict_write_gate import (
    AdapterReceipt,
    ExternalOperationRunner,
    ReconciliationDetermination,
    StrategyRegistry,
    StrictWriteGate,
    WriteDenied,
)


@pytest.fixture
def journal_path(tmp_path):
    return str(tmp_path / "journal" / "operations.sqlite3")


@pytest.fixture
def journal(journal_path):
    j = OperationJournal(journal_path)
    yield j
    j.close()


@pytest.fixture
def scope():
    return JournalScope(tenant="tenant-a", workflow="ido")


RECOVERABLE = ReconciliationDeclaration(
    strategy_id="tag-lookup",
    strategy_version="1",
    authoritative_source="umbrella",
    consistency_window_seconds=300,
    receipt_is_recoverable=True,
)

WRITE_CONTRACT = CommandEffectContract(
    definition_id="Resource/add_tag",
    effect_kind="write",
    effect_key="tag.add",
    idempotency="backend_deduplicated",
    reconciliation=RECOVERABLE,
)


def make_runner(journal, *, contract=WRITE_CONTRACT, mode="enabled"):
    registry = ContractRegistry()
    registry.register(contract)
    gate = StrictWriteGate(
        contracts=registry,
        journal=journal,
        enabled_commands={contract.definition_id: mode},
    )
    return ExternalOperationRunner(journal, gate)


def dispatch(runner, journal, scope, adapter, **kwargs):
    return runner.dispatch(
        scope,
        definition_id="Resource/add_tag",
        target={"uid": "ident-1"},
        parameters={"tag": "review"},
        adapter=adapter,
        turn_key="turn-1",
        step_index=3,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Storage configuration — verified, not assumed
# ----------------------------------------------------------------------


def test_the_durability_configuration_is_verified_on_open(journal, journal_path):
    """Arch §14.1. A database that came up in `synchronous=NORMAL` looks
    identical and has lost the one guarantee the module is for."""
    conn = sqlite3.connect(journal_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()
    assert journal.owner_epoch >= 1


def test_the_journal_file_is_private(journal, journal_path):
    mode = os.stat(journal_path).st_mode & 0o777
    assert mode == 0o600


# ----------------------------------------------------------------------
# Row 1-3: before dispatch
# ----------------------------------------------------------------------


def test_a_new_operation_starts_not_dispatched(journal, scope):
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    assert operation.outcome == "not-dispatched"
    assert not operation.is_protected


def test_a_replayed_logical_call_joins_the_existing_operation(journal, scope):
    """`UNIQUE(scope, logical_call_key)` as a safety property.

    The replay finds the operation that already exists rather than minting a
    second one — which is the difference between one effect and two.
    """
    first = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    second = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    assert first.operation_id == second.operation_id


def test_the_same_key_with_a_different_binding_is_a_conflict(journal, scope):
    """Not a retry: the caller reused an identity for a different call."""
    journal.create_or_get_operation(scope, "lck-1", "bind-1", definition_id="Resource/add_tag")
    with pytest.raises(IdempotencyConflict):
        journal.create_or_get_operation(
            scope, "lck-1", "bind-DIFFERENT", definition_id="Resource/add_tag"
        )


def test_a_crash_before_dispatch_leaves_nothing_to_reconcile(journal, scope):
    """Fault injection point 1. Nothing was attempted, and the record says so."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    # A restart: new epoch, same file.
    journal.close()
    restarted = OperationJournal(journal._path)
    try:
        recovered = restarted.get_operation(scope, operation.operation_id)
        assert recovered.outcome == "not-dispatched"
        assert not recovered.is_protected
        assert restarted.health(scope)["protected"] == 0
    finally:
        restarted.close()


# ----------------------------------------------------------------------
# Row 4-6: after dispatch, before acknowledgement
# ----------------------------------------------------------------------


def test_the_unknown_premark_is_durable_before_the_adapter_runs(journal, scope):
    """The load-bearing ordering, asserted from inside the adapter.

    If this ever inverts, a crash between dispatch and acknowledgement leaves
    NO evidence that anything was attempted, and the next run dispatches again.
    """
    runner = make_runner(journal)
    seen = {}

    def adapter(permit):
        seen["outcome"] = journal.get_operation(scope, permit.operation_id).outcome
        return AdapterReceipt(receipt_id="r1", outcome="succeeded", recoverable=True)

    dispatch(runner, journal, scope, adapter)
    assert seen["outcome"] == "outcome-unknown"


def test_a_crash_after_dispatch_leaves_outcome_unknown(journal, scope):
    """Fault injection point 2, and the state that says "we may have applied this"."""
    runner = make_runner(journal)

    def crashing_adapter(permit):
        raise ConnectionResetError("the response never arrived")

    with pytest.raises(ConnectionResetError):
        dispatch(runner, journal, scope, crashing_adapter, correlation="task-1")

    operations = journal._conn.execute("SELECT * FROM operations").fetchall()
    assert len(operations) == 1
    assert operations[0]["outcome"] == "outcome-unknown"
    # And the gate is up, so nothing else dispatches on this correlation.
    assert journal.gate_is_set(scope, "task-1")


def test_an_unknown_operation_is_not_retried_without_evidence(journal, scope):
    """FW-REQ-008B clause 1 and the retry matrix (§14.5)."""
    contract = WRITE_CONTRACT.model_copy(update={"idempotency": "unknown"})
    runner = make_runner(journal, contract=contract)

    def crashing_adapter(permit):
        raise ConnectionResetError("lost")

    with pytest.raises(ConnectionResetError):
        dispatch(runner, journal, scope, crashing_adapter, correlation="task-1")

    calls = []

    def second_attempt(permit):
        calls.append(1)
        return AdapterReceipt(receipt_id="r2", outcome="succeeded", recoverable=True)

    # Refused at the gate rather than deeper in the runner — the correlation's
    # reconciliation gate went up when the first attempt was lost, and the gate
    # is the cheapest point that can say no. Either refusal is correct; what
    # matters is that the backend is not called again.
    with pytest.raises((ReconciliationRequired, WriteDenied)):
        dispatch(runner, journal, scope, second_attempt, correlation="task-1")
    assert calls == [], "the backend must not be called again"

    # And with no correlation to gate on, the runner itself refuses, for the
    # same reason: the operation is unknown and the contract does not declare
    # backend deduplication.
    with pytest.raises(ReconciliationRequired):
        dispatch(runner, journal, scope, second_attempt)
    assert calls == []


def test_a_same_id_retry_is_allowed_only_under_declared_deduplication(journal, scope):
    """The one exception the matrix allows, and it is the contract's to make."""
    runner = make_runner(journal)  # idempotency == backend_deduplicated

    def crashing_adapter(permit):
        raise ConnectionResetError("lost")

    with pytest.raises(ConnectionResetError):
        dispatch(runner, journal, scope, crashing_adapter)

    def succeeding(permit):
        return AdapterReceipt(receipt_id="r2", outcome="succeeded", recoverable=True)

    record = dispatch(runner, journal, scope, succeeding)
    assert record.outcome == "succeeded"


def test_a_partially_applied_operation_is_never_retried(journal, scope):
    """Arch §14.5: never. Reconciliation may create a linked residual operation."""
    runner = make_runner(journal)

    def partial(permit):
        return AdapterReceipt(
            receipt_id="r1", outcome="partially-applied", recoverable=True,
            applied_subset=["a"],
        )

    record = dispatch(runner, journal, scope, partial, correlation="task-1")
    assert record.outcome == "partially-applied"

    calls = []

    def counted(permit):
        calls.append(1)
        return partial(permit)

    # Gated at the correlation, and refused by the runner even without one:
    # `backend_deduplicated` buys a retry of an UNKNOWN outcome, never of a
    # partial one (arch §14.5 — "never").
    with pytest.raises((ReconciliationRequired, WriteDenied)):
        dispatch(runner, journal, scope, counted, correlation="task-1")
    with pytest.raises(ReconciliationRequired):
        dispatch(runner, journal, scope, counted)
    assert calls == []


# ----------------------------------------------------------------------
# Row 7-9: after acknowledgement, before checkpoint
# ----------------------------------------------------------------------


def test_an_unrecoverable_success_receipt_does_not_certify_success(journal, scope):
    """Arch §14.6: a UUID that exists only in a lost POST response is not evidence.

    The adapter said "succeeded". It cannot support the claim, so the record
    stays unknown and reconciliation owns it — which is the difference between
    a receipt and a promise.
    """
    runner = make_runner(journal)

    def optimistic(permit):
        return AdapterReceipt(receipt_id="r1", outcome="succeeded", recoverable=False)

    record = dispatch(runner, journal, scope, optimistic, correlation="task-1")
    assert record.outcome == "outcome-unknown"
    assert journal.gate_is_set(scope, "task-1")


def test_a_succeeded_operation_is_immutable(journal, scope):
    """A late result is recorded and cannot overwrite (§14.4 last row)."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    permit = journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    journal.transition(scope, permit, "succeeded", reason="receipt", receipt_ref="r1")

    with pytest.raises(JournalError):
        journal.transition(scope, permit, "failed-before-effect", reason="late")
    assert journal.get_operation(scope, operation.operation_id).outcome == "succeeded"
    # The refusal is itself recorded, so the late result is not simply lost.
    assert any("cas-lost" in event["reason"] for event in journal.events(operation.operation_id))


def test_a_publication_failure_after_backend_success_leaves_unknown(journal, scope):
    """Arch §14.3 last paragraph, stated as a test.

    "If outcome publication fails after backend success, durable state remains
    unknown and the reconciliation gate stays set. The backend call is not
    repeated."
    """
    runner = make_runner(journal)
    calls = []

    def adapter(permit):
        calls.append(1)
        return AdapterReceipt(receipt_id="r1", outcome="succeeded", recoverable=True)

    # Publication fails: simulated by the CAS losing, which is what a concurrent
    # transition or a lost commit looks like from here.
    original_transition = journal.transition

    def failing_transition(*args, **kwargs):
        raise JournalError("commit lost")

    journal.transition = failing_transition
    try:
        with pytest.raises(JournalError):
            dispatch(runner, journal, scope, adapter, correlation="task-1")
    finally:
        journal.transition = original_transition

    assert calls == [1]
    stored = journal._conn.execute("SELECT outcome FROM operations").fetchone()[0]
    assert stored == "outcome-unknown"


# ----------------------------------------------------------------------
# Row 10-12: ownership, epochs and restart
# ----------------------------------------------------------------------


def test_a_restart_increments_the_epoch(journal, scope):
    first_epoch = journal.owner_epoch
    journal.close()
    restarted = OperationJournal(journal._path)
    try:
        assert restarted.owner_epoch == first_epoch + 1
    finally:
        restarted.close()


def test_a_permit_from_a_previous_epoch_cannot_transition(journal, scope):
    """Arch §14.1: restart treats every live prior permit as outcome-unknown."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    permit = journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    journal.close()

    restarted = OperationJournal(journal._path)
    try:
        with pytest.raises(OwnershipLost):
            restarted.transition(scope, permit, "succeeded", reason="late-from-old-epoch")
        assert (
            restarted.get_operation(scope, operation.operation_id).outcome
            == "outcome-unknown"
        )
    finally:
        restarted.close()


def test_an_attempt_under_a_stale_epoch_is_refused(journal, scope):
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    with pytest.raises(OwnershipLost):
        journal.begin_attempt(
            scope, operation.operation_id, operation.record_version,
            journal.owner_epoch - 1, deadline_monotonic=time.monotonic() + 30,
        )


def test_a_scope_cannot_read_another_scopes_operation(journal, scope):
    """Arch §14.8: an operation ID alone never grants access."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    other = JournalScope(tenant="tenant-b", workflow="ido")
    with pytest.raises(JournalError):
        journal.get_operation(other, operation.operation_id)


# ----------------------------------------------------------------------
# Reconciliation and compensation
# ----------------------------------------------------------------------


@dataclass
class _LookupStrategy:
    strategy_id: str = "tag-lookup"
    strategy_version: str = "1"
    allowed_source_outcomes: frozenset = frozenset({"outcome-unknown", "partially-applied"})
    answer: str = "failed-before-effect"

    def reconcile(self, scope, operation, deadline):
        return ReconciliationDetermination(
            determination=self.answer,
            # Produced by the strategy's rules; the caller cannot supply it.
            retry_eligible=self.answer == "failed-before-effect",
            rule_id="no-tag-present",
            evidence_ref="umbrella:query-1",
        )


def test_reconciliation_resolves_an_unknown_outcome_with_evidence(journal, scope):
    runner = make_runner(journal)
    runner.strategies.register(_LookupStrategy())

    def crashing(permit):
        raise ConnectionResetError("lost")

    with pytest.raises(ConnectionResetError):
        dispatch(runner, journal, scope, crashing, correlation="task-1")
    operation_id = journal._conn.execute("SELECT operation_id FROM operations").fetchone()[0]

    record = runner.reconcile(scope, operation_id, strategy_id="tag-lookup")
    assert record.outcome == "failed-before-effect"
    events = journal._conn.execute(
        "SELECT * FROM reconciliation_events WHERE operation_id = ?", (operation_id,)
    ).fetchall()
    assert events[0]["retry_eligible"] == 1
    assert events[0]["rule_id"] == "no-tag-present"


def test_an_unregistered_strategy_cannot_reconcile(journal, scope):
    """A strategy is versioned code, not a flag a caller can name into existence."""
    runner = make_runner(journal)
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    with pytest.raises(JournalError):
        runner.reconcile(scope, operation.operation_id, strategy_id="wishful-thinking")


def test_a_strategy_cannot_reconcile_an_outcome_it_does_not_accept(journal, scope):
    runner = make_runner(journal)
    runner.strategies.register(_LookupStrategy())
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )  # not-dispatched
    with pytest.raises(JournalError):
        runner.reconcile(scope, operation.operation_id, strategy_id="tag-lookup")


def test_compensation_is_a_new_linked_operation_that_erases_nothing(journal, scope):
    """Arch §14.7. Fault injection point 4.

    The link is durable before the compensation can be dispatched, so a crash
    after the compensating effect resolves through the already-durable link.
    """
    original = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    permit = journal.begin_attempt(
        scope, original.operation_id, original.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    journal.transition(scope, permit, "partially-applied", reason="partial", receipt_ref="r1")

    compensation = journal.create_compensation(
        scope, original.operation_id,
        logical_call_key="lck-1-comp", binding_digest="bind-comp",
        definition_id="Resource/remove_tag",
    )
    assert compensation.outcome == "not-dispatched"
    assert compensation.retention_class == "compensation"
    link = journal._conn.execute(
        "SELECT * FROM compensation_links WHERE original_operation_id = ?",
        (original.operation_id,),
    ).fetchone()
    assert link["compensation_operation_id"] == compensation.operation_id
    # The original is untouched.
    assert journal.get_operation(scope, original.operation_id).outcome == "partially-applied"


# ----------------------------------------------------------------------
# The gate, retention and capacity
# ----------------------------------------------------------------------


def test_a_gate_cannot_be_cleared_while_it_still_blocks(journal, scope):
    """Clearing it while an operation is unknown is the forbidden conversion,
    performed by omission."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    permit = journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    journal.set_gate(scope, "task-1", [operation.operation_id])

    with pytest.raises(ReconciliationRequired):
        journal.clear_gate(scope, "task-1")

    journal.transition(scope, permit, "succeeded", reason="receipt", receipt_ref="r1")
    journal.clear_gate(scope, "task-1")
    assert not journal.gate_is_set(scope, "task-1")


def test_writes_are_denied_by_default(journal, scope):
    """Per-command enablement, default-deny (plan §6)."""
    runner = make_runner(journal, mode="disabled")
    with pytest.raises(WriteDenied) as caught:
        dispatch(runner, journal, scope, lambda permit: None)
    assert caught.value.failure.code == "write-disabled"


def test_a_contract_without_reconciliation_cannot_dispatch(journal, scope):
    """An unknown outcome with no way to resolve it is a permanent unknown."""
    contract = WRITE_CONTRACT.model_copy(update={"reconciliation": None})
    runner = make_runner(journal, contract=contract)
    with pytest.raises(WriteDenied) as caught:
        dispatch(runner, journal, scope, lambda permit: None)
    assert caught.value.failure.code == "contract-insufficient"


def test_capacity_pressure_disables_dispatch_rather_than_deleting_a_record(journal, scope):
    """Arch §14.8. The unresolved records are the reason to stop, so deleting
    them to make room is backwards."""
    registry = ContractRegistry()
    registry.register(WRITE_CONTRACT)
    gate = StrictWriteGate(
        contracts=registry, journal=journal,
        enabled_commands={"Resource/add_tag": "enabled"},
    )
    runner = ExternalOperationRunner(journal, gate)

    # One unresolved operation, and a cap of one.
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    assert journal.health(scope, max_protected=1)["dispatch_enabled"] is False

    original_health = journal.health
    journal.health = lambda s, **kw: original_health(s, max_protected=1)
    try:
        with pytest.raises(WriteDenied) as caught:
            dispatch(runner, journal, scope, lambda permit: None)
        assert caught.value.failure.code == "journal-capacity"
    finally:
        journal.health = original_health

    # And the protected record is still there.
    assert journal.get_operation(scope, operation.operation_id).outcome == "outcome-unknown"


def test_compaction_writes_a_tombstone_and_blocks_recreation(journal, scope):
    """Arch §14.8: a tombstone stops a replayed old request from recreating a
    deleted operation under a new ID."""
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    permit = journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    journal.transition(scope, permit, "succeeded", reason="receipt", receipt_ref="r1")

    assert journal.compact(scope, older_than_seconds=-1, tombstone_seconds=3600) == 1
    with pytest.raises(IdempotencyConflict):
        journal.create_or_get_operation(
            scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
        )


def test_compaction_never_touches_a_protected_record(journal, scope):
    operation = journal.create_or_get_operation(
        scope, "lck-1", "bind-1", definition_id="Resource/add_tag"
    )
    journal.begin_attempt(
        scope, operation.operation_id, operation.record_version, journal.owner_epoch,
        deadline_monotonic=time.monotonic() + 30,
    )
    assert journal.compact(scope, older_than_seconds=-1, tombstone_seconds=3600) == 0
    assert journal.get_operation(scope, operation.operation_id).outcome == "outcome-unknown"


# ----------------------------------------------------------------------
# Logical-call key derivation
# ----------------------------------------------------------------------


def test_the_same_call_at_the_same_step_derives_the_same_key():
    kwargs = dict(
        scope_digest="s", turn_key="t1", step_index=3,
        definition_id="Resource/add_tag", effect_key="tag.add", target={"uid": "x"},
    )
    assert logical_call_key(**kwargs) == logical_call_key(**kwargs)
    assert logical_call_key(**{**kwargs, "step_index": 4}) != logical_call_key(**kwargs)
    assert logical_call_key(**{**kwargs, "effect_key": "tag.remove"}) != logical_call_key(**kwargs)


def test_the_binding_digest_covers_the_parameters_the_key_does_not():
    """Which is what catches a caller reusing a key for a different call."""
    base = dict(
        definition_id="Resource/add_tag", effect_key="tag.add",
        target={"uid": "x"}, security_scope="s",
    )
    assert binding_digest(**base, parameters={"tag": "a"}) != binding_digest(
        **base, parameters={"tag": "b"}
    )
