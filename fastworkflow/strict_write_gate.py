"""The only path an external write may take (arch §14.3-14.6, FW-REQ-008B).

    CommandDispatcher -> ExternalOperationRunner -> OperationJournal
                                                 -> BackendAdapter
                                                 -> OperationJournal
                      -> ExecutionRecorder

WEC never calls the journal or the backend directly. The runner below is the
middle of that chain, and the ordering inside it is the safety argument:

1. the operation is created or joined, so a replay finds the existing one;
2. the journal durably pre-marks `outcome-unknown` and issues a single-use
   permit, so a crash anywhere after this leaves a state that says "we may have
   applied this";
3. the adapter is invoked **exactly once**;
4. the outcome is committed under compare-and-set.

Step 2 before step 3 is the whole thing. Reversed — dispatch, then record — a
crash between them leaves no evidence that anything was attempted, and the next
run dispatches again.

**Writes are disabled.** `StrictWriteGate` denies by default and this slice
ships in shadow against a fake backend; G1W is a separate per-command gate
(implementation plan §6). What exists here is the machinery that gate will
need, exercised by the crash matrix, not a route to production writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Protocol

from fastworkflow.command_contract import (
    CommandEffectContract,
    ContractRegistry,
    binding_digest as compute_binding_digest,
    logical_call_key as compute_logical_call_key,
)
from fastworkflow.operation_journal import (
    DispatchPermit,
    JournalError,
    JournalScope,
    OperationJournal,
    OperationRecord,
    ReconciliationRequired,
    SideEffectOutcome,
)
from fastworkflow.typed_failure import ControlSignal, TypedFailure
from fastworkflow.utils.logging import logger

# Per-command write enablement (implementation plan §6). A command is absent
# from this set until its own G1W package is approved — backend-owner dedup
# attestation, receipts, the crash matrix, and a deployment scope.
WriteMode = Literal["disabled", "shadow", "enabled"]


class WriteDenied(ControlSignal):
    """A write was refused before dispatch.

    A `ControlSignal`, so it halts the turn rather than becoming an observation
    the model reads as ordinary data and reasons around (arch §8.4).
    """


@dataclass(frozen=True)
class AdapterReceipt:
    """What an adapter came back with (arch §14.6).

    `recoverable` is the field that matters and the one easiest to claim
    falsely: it means the backend can retrieve or deduplicate this receipt by
    operation ID or idempotency key **after the response was lost**. A UUID that
    exists only in a lost POST response is not recoverable evidence, and an
    adapter that says otherwise is the one thing this record cannot check.
    """

    receipt_id: str
    outcome: SideEffectOutcome
    recoverable: bool = False
    applied_subset: Optional[Any] = None
    backend_version: Optional[str] = None
    evidence_digest: Optional[str] = None


@dataclass(frozen=True)
class ReconciliationDetermination:
    """What a strategy concluded, and whether its rules allow a retry."""

    determination: SideEffectOutcome
    retry_eligible: bool
    rule_id: str
    evidence_ref: Optional[str] = None


class ReconciliationStrategy(Protocol):
    """Versioned code, not a free-form flag (arch §14.6)."""

    strategy_id: str
    strategy_version: str
    allowed_source_outcomes: frozenset

    def reconcile(
        self, scope: JournalScope, operation: OperationRecord, deadline: float
    ) -> ReconciliationDetermination: ...


class StrategyRegistry:
    """Registered strategies by id. Version is part of the identity."""

    def __init__(self) -> None:
        self._by_id: dict[str, ReconciliationStrategy] = {}

    def register(self, strategy: ReconciliationStrategy) -> None:
        self._by_id[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> Optional[ReconciliationStrategy]:
        return self._by_id.get(strategy_id)

    def __len__(self) -> int:
        return len(self._by_id)


@dataclass
class StrictWriteGate:
    """Whether this command may dispatch a write at all (FW-REQ-019C, plan §6).

    Default-deny, per command, and every clause has to hold: the deployment
    enabled *this* command, the contract carries what a write needs, and the
    journal is healthy enough to record one. A failure of any of them is a
    refusal, never a downgrade to "proceed without recording".
    """

    contracts: ContractRegistry
    journal: OperationJournal
    enabled_commands: Mapping[str, WriteMode] = field(default_factory=dict)

    def mode(self, definition_id: str) -> WriteMode:
        return self.enabled_commands.get(definition_id, "disabled")

    def check(
        self, scope: JournalScope, definition_id: str, *, correlation: Optional[str] = None
    ) -> CommandEffectContract:
        """Raise `WriteDenied` unless this command may dispatch. Returns its contract."""
        contract = self.contracts.get(definition_id)
        mode = self.mode(definition_id)
        if mode == "disabled":
            raise WriteDenied(
                TypedFailure(
                    disposition="permanent",
                    code="write-disabled",
                    detail=(
                        f"{definition_id} is not G1W-enabled. Per-command write "
                        "enablement is its own approval (plan §6)."
                    ),
                )
            )
        if refusal := contract.refusal_reason():
            raise WriteDenied(
                TypedFailure(
                    disposition="permanent", code="contract-insufficient", detail=refusal
                )
            )
        health = self.journal.health(scope)
        if not health["dispatch_enabled"]:
            # Capacity pressure disables NEW dispatch before deleting a
            # protected record (arch §14.8) — the unresolved records are the
            # reason to stop, so deleting them to make room is backwards.
            raise WriteDenied(
                TypedFailure(
                    disposition="permanent",
                    code="journal-capacity",
                    detail=(
                        f"{health['protected']} unresolved operation(s) of "
                        f"{health['max_protected']} allowed; no new dispatch until "
                        "they are reconciled"
                    ),
                )
            )
        if correlation and self.journal.gate_is_set(scope, correlation):
            raise WriteDenied(
                TypedFailure(
                    disposition="outcome-unknown",
                    code="reconciliation-required",
                    detail=f"reconciliation gate is set for {correlation!r}",
                )
            )
        return contract


class ExternalOperationRunner:
    """The dispatch invariant (arch §14.3), in one place.

    Being one place is the point: G1W conformance requires every possible
    external write inside a command to go through the instrumented adapter API,
    because the dispatcher cannot infer or make safe an arbitrary hidden write
    performed inside Python code.
    """

    def __init__(
        self,
        journal: OperationJournal,
        gate: StrictWriteGate,
        *,
        strategies: Optional[StrategyRegistry] = None,
    ):
        self.journal = journal
        self.gate = gate
        self.strategies = strategies or StrategyRegistry()

    def dispatch(
        self,
        scope: JournalScope,
        *,
        definition_id: str,
        target: Any,
        parameters: Any,
        adapter: Callable[[DispatchPermit], AdapterReceipt],
        turn_key: Optional[str] = None,
        task_id: Optional[str] = None,
        step_index: Optional[int] = None,
        correlation: Optional[str] = None,
        deadline_seconds: float = 60.0,
    ) -> OperationRecord:
        """Run one write through the invariant. The adapter is called at most once."""
        contract = self.gate.check(scope, definition_id, correlation=correlation)

        effect_key = contract.effect_key or definition_id
        logical_key = compute_logical_call_key(
            scope_digest=scope.digest,
            turn_key=turn_key,
            step_index=step_index,
            definition_id=definition_id,
            effect_key=effect_key,
            target=target,
        )
        binding = compute_binding_digest(
            definition_id=definition_id,
            effect_key=effect_key,
            target=target,
            parameters=parameters,
            security_scope=scope.digest,
        )

        operation = self.journal.create_or_get_operation(
            scope, logical_key, binding,
            definition_id=definition_id,
            contract_version=contract.contract_version,
            turn_key=turn_key, task_id=task_id, step_index=step_index,
        )

        # Step 5: an existing operation is JOINED, and what that means depends
        # entirely on where it got to.
        if operation.outcome == "succeeded":
            # Never dispatch again; replay the stored result (arch §14.5).
            logger.info(
                "Write %s joined an already-succeeded operation %s; not dispatching",
                definition_id, operation.operation_id,
            )
            return operation
        if operation.outcome in ("outcome-unknown", "partially-applied"):
            # Automatic retry is prohibited unless same-ID deduplication is
            # guaranteed or reconciliation proves absence. A partially-applied
            # operation is never retried at all.
            if (
                operation.outcome == "outcome-unknown"
                and contract.idempotency == "backend_deduplicated"
            ):
                logger.warning(
                    "Retrying %s under the same operation id %s: the contract "
                    "declares backend deduplication",
                    definition_id, operation.operation_id,
                )
            else:
                if correlation:
                    self.journal.set_gate(scope, correlation, [operation.operation_id])
                raise ReconciliationRequired(
                    f"operation {operation.operation_id} is {operation.outcome!r}; "
                    "it cannot be retried without reconciliation evidence"
                )

        # Step 6: the durable pre-mark and the permit, atomically.
        permit = self.journal.begin_attempt(
            scope,
            operation.operation_id,
            operation.record_version,
            self.journal.owner_epoch,
            deadline_monotonic=time.monotonic() + deadline_seconds,
            gate_correlation=correlation,
        )

        # Step 7: exactly one adapter invocation. Everything below this line
        # runs with `outcome-unknown` already durable, which is what makes any
        # failure here recoverable rather than invisible.
        try:
            receipt = adapter(permit)
        except BaseException as exc:
            # The call may have landed. Nothing here knows, and guessing is the
            # conversion FW-REQ-008B forbids — so the state stays unknown and
            # the gate goes up.
            logger.error(
                "Adapter for %s raised after dispatch was permitted: %s",
                definition_id, exc,
            )
            if correlation:
                self.journal.set_gate(scope, correlation, [operation.operation_id])
            raise

        if receipt.outcome == "succeeded" and not receipt.recoverable:
            # An unrecoverable success receipt cannot be re-derived after a
            # response loss, so it cannot support the claim it is making. The
            # outcome stays unknown and reconciliation owns it.
            logger.warning(
                "Adapter for %s reported success with a non-recoverable receipt; "
                "recording outcome-unknown instead",
                definition_id,
            )
            if correlation:
                self.journal.set_gate(scope, correlation, [operation.operation_id])
            return self.journal.transition(
                scope, permit, "outcome-unknown",
                reason="unrecoverable-success-receipt",
                receipt_ref=receipt.receipt_id,
                evidence_digest=receipt.evidence_digest,
            )

        # Steps 9-10: commit under CAS. A loser records a diagnostic event and
        # changes nothing.
        record = self.journal.transition(
            scope, permit, receipt.outcome,
            reason=f"adapter:{receipt.receipt_id}",
            receipt_ref=receipt.receipt_id,
            evidence_digest=receipt.evidence_digest,
        )
        if record.outcome in ("outcome-unknown", "partially-applied") and correlation:
            self.journal.set_gate(scope, correlation, [record.operation_id])
        return record

    def reconcile(
        self,
        scope: JournalScope,
        operation_id: str,
        *,
        strategy_id: str,
        deadline_seconds: float = 120.0,
    ) -> OperationRecord:
        """Resolve an unresolved operation through a registered strategy.

        The determination and the retry eligibility both come from the
        strategy's own rules. A caller cannot supply either, which is what stops
        "we think it probably worked" from becoming a recorded success.
        """
        operation = self.journal.get_operation(scope, operation_id)
        strategy = self.strategies.get(strategy_id)
        if strategy is None:
            raise JournalError(
                f"no registered reconciliation strategy {strategy_id!r}; a strategy "
                "is versioned code, not a flag"
            )
        if operation.outcome not in strategy.allowed_source_outcomes:
            raise JournalError(
                f"strategy {strategy_id} does not accept source outcome "
                f"{operation.outcome!r}"
            )
        determination = strategy.reconcile(
            scope, operation, time.monotonic() + deadline_seconds
        )
        return self.journal.record_reconciliation(
            scope, operation_id,
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.strategy_version,
            determination=determination.determination,
            retry_eligible=determination.retry_eligible,
            rule_id=determination.rule_id,
            evidence_ref=determination.evidence_ref,
        )
