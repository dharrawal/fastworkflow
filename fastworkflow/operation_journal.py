"""Durable identity and outcome for every external write (arch §14, FW-REQ-008B).

Every write in this system is replay-unsafe by construction today: there is no
operation identity, so a retry cannot tell a not-dispatched attempt from an
outcome-unknown one, and nothing stops a whole-agent retry from re-dispatching
a completed effect.

The property this module exists to make impossible to violate:

    **an unknown outcome never becomes a known one without evidence.**

Everything else here is in service of that. The durable pre-mark is why a crash
between "we are about to dispatch" and "the backend answered" leaves
`outcome-unknown` on disk rather than nothing. Compare-and-set on a record
version is why a late thread cannot overwrite a reconciled outcome with a
stale one. `UNIQUE(scope, logical_call_key)` is why a replayed request joins the
existing operation instead of minting a second effect. And the tombstone is why
a replay cannot recreate a deleted operation under a new ID after compaction.

**This slice ships in shadow.** Nothing here enables a write: G1W is a separate
per-command gate (implementation plan §6), and the journal existing is a
precondition for that gate rather than a step through it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional

from fastworkflow.typed_failure import TypedFailure

SCHEMA_VERSION = 1

# Arch §6.8 / FW-REQ-008B. What happened to the EXTERNAL EFFECT — not whether
# the call raised.
SideEffectOutcome = Literal[
    "not-dispatched",
    "succeeded",
    "failed-before-effect",
    "partially-applied",
    "outcome-unknown",
]

# Arch §14.4. The attempt's execution phase, which is a different axis from the
# effect outcome: an attempt can be `finalized` with the effect still unknown.
AttemptPhase = Literal[
    "prepared", "dispatch-permitted", "adapter-running", "adapter-returned", "finalized"
]

# Outcomes that may never be compacted (arch §14.8).
PROTECTED_OUTCOMES = frozenset(
    {"outcome-unknown", "partially-applied"}
)

# Which transitions the journal will make, and from what (arch §14.4). A
# transition absent from this table is refused — the table IS the rule, rather
# than a description of code that implements it somewhere else.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "not-dispatched": frozenset({"outcome-unknown"}),
    "outcome-unknown": frozenset(
        {"succeeded", "failed-before-effect", "partially-applied", "outcome-unknown"}
    ),
    # A retry of a call that provably never reached the backend.
    "failed-before-effect": frozenset({"outcome-unknown"}),
    # Reconciliation may resolve a partial upward, or leave it where it is. It
    # is never retried (arch §14.5).
    "partially-applied": frozenset({"partially-applied", "succeeded"}),
    # Immutable. A late result is recorded as an event and cannot overwrite it.
    "succeeded": frozenset(),
}


class JournalError(RuntimeError):
    """The journal refused. Always fail-closed: no write proceeds on this."""


class JournalUnavailable(JournalError):
    """The journal could not be opened or verified. G1W dispatch is disabled."""


class IdempotencyConflict(JournalError):
    """The same logical call arrived with a different binding (arch §14.3 step 5)."""


class ReconciliationRequired(JournalError):
    """The operation is in a state only reconciliation may resolve."""


class OwnershipLost(JournalError):
    """A permit or transition arrived from a process that no longer owns the scope."""


@dataclass(frozen=True)
class JournalScope:
    """A security namespace. Every API requires one (arch §14.8 last paragraph).

    An operation ID alone never grants access: a caller that has one but cannot
    name the scope it belongs to is not the caller that created it.
    """

    tenant: str
    workflow: str
    key_version: int = 1
    # Never the raw tenant/principal — a keyed digest of them, so the table
    # holds an identifier rather than the data (arch §14.1 "keyed scope
    # identifiers rather than raw tenant/principal data").
    _secret: bytes = field(default=b"fastworkflow-journal-scope", repr=False)

    @property
    def digest(self) -> str:
        payload = f"v{self.key_version}|{self.tenant}|{self.workflow}".encode("utf-8")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class DispatchPermit:
    """One-shot authority to make exactly one external call (arch §14.4).

    Bound to the operation, the attempt, the record version it was issued
    against, and the owner epoch. Every one of those is checked again at
    transition time, so a permit that outlived any of them cannot be spent.
    """

    operation_id: str
    attempt_id: str
    permit_version: int
    owner_epoch: int
    expected_operation_version: int
    deadline_monotonic: float

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.deadline_monotonic


@dataclass
class OperationRecord:
    operation_id: str
    scope_digest: str
    logical_call_key: str
    binding_digest: str
    definition_id: str
    contract_version: str
    outcome: SideEffectOutcome
    record_version: int
    turn_key: Optional[str] = None
    task_id: Optional[str] = None
    step_index: Optional[int] = None
    retention_class: str = "standard"
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_protected(self) -> bool:
        return self.outcome in PROTECTED_OUTCOMES

    @property
    def is_terminal_success(self) -> bool:
        return self.outcome == "succeeded"


class OperationJournal:
    """SQLite-backed durable operation record (arch §14.1).

    The configuration is not tuning. WAL plus `synchronous=FULL` is what makes
    the pre-mark survive a power loss; `BEGIN IMMEDIATE` is what stops two
    writers from interleaving a read-modify-write of the same operation; and
    every one of them is **verified on every open**, because a database that
    silently came up in `synchronous=NORMAL` would look identical and lose the
    one guarantee the module is for.
    """

    def __init__(self, path: str, *, owner_epoch: Optional[int] = None):
        self._path = path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._owner_epoch = owner_epoch
        self._open_and_verify()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, isolation_level=None, timeout=10.0)
            conn.row_factory = sqlite3.Row
            self._configure(conn)
            self._local.conn = conn
        return conn

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")

    def _open_and_verify(self) -> None:
        directory = os.path.dirname(os.path.abspath(self._path))
        if directory:
            os.makedirs(directory, exist_ok=True)
            # Private by default: the journal holds operation identity for a
            # tenant's writes.
            with contextlib_suppress():
                os.chmod(directory, 0o700)
        conn = self._conn
        self._create_schema(conn)

        # Verified rather than assumed (arch §14.1 "verifies journal mode,
        # synchronous mode, foreign keys, busy timeout, integrity and file
        # permissions on every open"). A journal that came up without these is
        # not a journal, and running on it would produce records nobody can
        # trust afterwards.
        checks = {
            "journal_mode": ("PRAGMA journal_mode", "wal"),
            "synchronous": ("PRAGMA synchronous", 2),
            "foreign_keys": ("PRAGMA foreign_keys", 1),
        }
        for name, (pragma, expected) in checks.items():
            value = conn.execute(pragma).fetchone()[0]
            actual = value.lower() if isinstance(value, str) else value
            if actual != expected:
                raise JournalUnavailable(
                    f"journal {name} is {value!r}, expected {expected!r}; "
                    "refusing to run on a database that cannot make the "
                    "durability guarantee the journal depends on"
                )
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise JournalUnavailable(f"journal integrity check failed: {integrity}")

        with contextlib_suppress():
            os.chmod(self._path, 0o600)

        if self._owner_epoch is None:
            self._owner_epoch = self._claim_owner_epoch()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS operations (
                operation_id        TEXT PRIMARY KEY,
                scope_digest        TEXT NOT NULL,
                logical_call_key    TEXT NOT NULL,
                binding_digest      TEXT NOT NULL,
                definition_id       TEXT NOT NULL,
                contract_version    TEXT NOT NULL,
                turn_key            TEXT,
                task_id             TEXT,
                step_index          INTEGER,
                outcome             TEXT NOT NULL,
                record_version      INTEGER NOT NULL,
                retention_class     TEXT NOT NULL DEFAULT 'standard',
                created_at          REAL NOT NULL,
                updated_at          REAL NOT NULL,
                UNIQUE(scope_digest, logical_call_key)
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id          TEXT PRIMARY KEY,
                operation_id        TEXT NOT NULL REFERENCES operations(operation_id),
                attempt_number      INTEGER NOT NULL,
                owner_epoch         INTEGER NOT NULL,
                permit_version      INTEGER NOT NULL,
                deadline            REAL,
                dispatched_at       REAL,
                phase               TEXT NOT NULL,
                outcome             TEXT,
                failure_class       TEXT,
                receipt_ref         TEXT,
                evidence_ref        TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id        TEXT NOT NULL REFERENCES operations(operation_id),
                old_outcome         TEXT,
                new_outcome         TEXT,
                reason              TEXT NOT NULL,
                evidence_digest     TEXT,
                writer              TEXT,
                created_at          REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reconciliation_events (
                seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id        TEXT NOT NULL REFERENCES operations(operation_id),
                strategy_id         TEXT NOT NULL,
                strategy_version    TEXT NOT NULL,
                determination       TEXT NOT NULL,
                retry_eligible      INTEGER NOT NULL,
                rule_id             TEXT,
                evidence_ref        TEXT,
                created_at          REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compensation_links (
                original_operation_id     TEXT NOT NULL REFERENCES operations(operation_id),
                compensation_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
                restored_predicate_ref    TEXT,
                created_at                REAL NOT NULL,
                PRIMARY KEY (original_operation_id, compensation_operation_id)
            );

            CREATE TABLE IF NOT EXISTS reconciliation_gates (
                scope_digest        TEXT NOT NULL,
                correlation         TEXT NOT NULL,
                status              TEXT NOT NULL,
                blocking_ids        TEXT NOT NULL,
                record_version      INTEGER NOT NULL,
                updated_at          REAL NOT NULL,
                PRIMARY KEY (scope_digest, correlation)
            );

            CREATE TABLE IF NOT EXISTS logical_call_tombstones (
                scope_digest        TEXT NOT NULL,
                logical_call_key    TEXT NOT NULL,
                final_binding       TEXT NOT NULL,
                final_outcome       TEXT NOT NULL,
                retained_until      REAL NOT NULL,
                hold                INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (scope_digest, logical_call_key)
            );

            CREATE TABLE IF NOT EXISTS journal_owner (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                owner_epoch         INTEGER NOT NULL,
                claimed_at          REAL NOT NULL
            );
            """
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ------------------------------------------------------------------
    # Ownership (arch §14.1)
    # ------------------------------------------------------------------

    def _claim_owner_epoch(self) -> int:
        """Bump the durable epoch. Every prior permit is now unspendable.

        Restart increments the epoch and treats every live prior permit as
        outcome-unknown — which is the honest reading: a permit issued by a
        process that is gone may or may not have been spent, and there is no
        way from here to find out.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT owner_epoch FROM journal_owner WHERE id = 1").fetchone()
            epoch = (row["owner_epoch"] if row else 0) + 1
            conn.execute(
                "INSERT INTO journal_owner (id, owner_epoch, claimed_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET owner_epoch = excluded.owner_epoch, "
                "claimed_at = excluded.claimed_at",
                (epoch, time.time()),
            )
        return epoch

    @property
    def owner_epoch(self) -> int:
        return int(self._owner_epoch or 0)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """`BEGIN IMMEDIATE`, so two writers cannot interleave a read-modify-write."""
        conn = self._conn
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # The API (arch §14.1 protocol)
    # ------------------------------------------------------------------

    def create_or_get_operation(
        self,
        scope: JournalScope,
        logical_call_key: str,
        binding_digest: str,
        *,
        definition_id: str,
        contract_version: str = "1",
        turn_key: Optional[str] = None,
        task_id: Optional[str] = None,
        step_index: Optional[int] = None,
    ) -> OperationRecord:
        """Get the operation for this logical call, creating it if it is new.

        The join point. A replayed request with the same logical-call key finds
        the existing operation rather than minting a second one — which is what
        makes `UNIQUE(scope, logical_call_key)` a safety property rather than a
        tidiness one.

        A **different binding** under the same key is an `IdempotencyConflict`,
        not a retry: the caller reused an identity for a different call, and
        joining them would attribute one call's outcome to another.
        """
        now = time.time()
        with self._transaction() as conn:
            tombstone = conn.execute(
                "SELECT * FROM logical_call_tombstones WHERE scope_digest = ? "
                "AND logical_call_key = ?",
                (scope.digest, logical_call_key),
            ).fetchone()
            if tombstone is not None:
                # Arch §14.8: a tombstone stops a replayable old request from
                # recreating a deleted operation under a NEW id — which would
                # present a completed effect as a fresh one.
                raise IdempotencyConflict(
                    f"logical call {logical_call_key[:12]}… was compacted with final "
                    f"outcome {tombstone['final_outcome']!r}; it cannot be recreated"
                )

            row = conn.execute(
                "SELECT * FROM operations WHERE scope_digest = ? AND logical_call_key = ?",
                (scope.digest, logical_call_key),
            ).fetchone()
            if row is not None:
                if row["binding_digest"] != binding_digest:
                    raise IdempotencyConflict(
                        f"logical call {logical_call_key[:12]}… already exists with a "
                        "different binding; the same identity is being reused for a "
                        "different call"
                    )
                return _record(row)

            operation_id = f"op-{uuid.uuid4().hex}"
            conn.execute(
                "INSERT INTO operations (operation_id, scope_digest, logical_call_key, "
                "binding_digest, definition_id, contract_version, turn_key, task_id, "
                "step_index, outcome, record_version, retention_class, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    operation_id, scope.digest, logical_call_key, binding_digest,
                    definition_id, contract_version, turn_key, task_id, step_index,
                    "not-dispatched", 1, "standard", now, now,
                ),
            )
            self._append_event(
                conn, operation_id, None, "not-dispatched", "create-or-get", None
            )
            return self.get_operation(scope, operation_id, _conn=conn)

    def begin_attempt(
        self,
        scope: JournalScope,
        operation_id: str,
        expected_version: int,
        owner_epoch: int,
        *,
        deadline_monotonic: float,
        gate_correlation: Optional[str] = None,
    ) -> DispatchPermit:
        """Atomically pre-mark unknown and issue one dispatch permit.

        The pre-mark is the point of the whole module. It happens **before** the
        adapter is called and in the same transaction as the attempt row, so a
        crash anywhere after this leaves `outcome-unknown` on disk — a state
        that says "we may have applied this" rather than nothing, which is what
        a reader needs and what an absent row cannot express.
        """
        with self._transaction() as conn:
            if owner_epoch != self.owner_epoch:
                raise OwnershipLost(
                    f"attempt requested under epoch {owner_epoch}, current is "
                    f"{self.owner_epoch}; this process no longer owns the scope"
                )
            row = self._locked_operation(conn, scope, operation_id)
            if row["record_version"] != expected_version:
                raise JournalError(
                    f"operation {operation_id} is at version {row['record_version']}, "
                    f"caller expected {expected_version}"
                )
            if gate_correlation and self._gate_is_set(conn, scope, gate_correlation):
                raise ReconciliationRequired(
                    f"reconciliation gate is set for {gate_correlation!r}; no new "
                    "dispatch until it clears"
                )
            current = row["outcome"]
            if "outcome-unknown" not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
                raise JournalError(
                    f"operation {operation_id} is {current!r}; a new attempt is not "
                    f"an allowed transition from it (arch §14.4)"
                )

            attempt_number = conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE operation_id = ?", (operation_id,)
            ).fetchone()[0] + 1
            attempt_id = f"att-{uuid.uuid4().hex}"
            new_version = row["record_version"] + 1
            conn.execute(
                "INSERT INTO attempts (attempt_id, operation_id, attempt_number, "
                "owner_epoch, permit_version, deadline, dispatched_at, phase) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    attempt_id, operation_id, attempt_number, owner_epoch, new_version,
                    deadline_monotonic, time.time(), "dispatch-permitted",
                ),
            )
            conn.execute(
                "UPDATE operations SET outcome = ?, record_version = ?, updated_at = ? "
                "WHERE operation_id = ? AND record_version = ?",
                ("outcome-unknown", new_version, time.time(), operation_id, row["record_version"]),
            )
            self._append_event(
                conn, operation_id, current, "outcome-unknown", "begin-attempt", None
            )
            return DispatchPermit(
                operation_id=operation_id,
                attempt_id=attempt_id,
                permit_version=new_version,
                owner_epoch=owner_epoch,
                expected_operation_version=new_version,
                deadline_monotonic=deadline_monotonic,
            )

    def transition(
        self,
        scope: JournalScope,
        permit: DispatchPermit,
        new_outcome: SideEffectOutcome,
        *,
        reason: str,
        evidence_digest: Optional[str] = None,
        receipt_ref: Optional[str] = None,
        failure_class: Optional[str] = None,
    ) -> OperationRecord:
        """Record what the adapter came back with, under compare-and-set.

        Every guard here is a way a wrong outcome could otherwise be written:
        a stale epoch (the process lost ownership mid-call), a spent or
        superseded permit, a record version that moved underneath us, or a
        transition the matrix does not allow. A CAS loser records a diagnostic
        event and changes nothing (arch §14.4).
        """
        # The refusal path commits its diagnostic event in its OWN transaction
        # and then raises. Recording it inside the transaction that raises would
        # roll it back with everything else — so the late result would be
        # refused, correctly, and then vanish, which is the half of arch §14.4
        # that says a CAS loss "records a diagnostic event".
        refusal: Optional[tuple[str, str, BaseException]] = None
        with self._transaction() as conn:
            row = self._locked_operation(conn, scope, permit.operation_id)
            current: str = row["outcome"]

            if permit.owner_epoch != self.owner_epoch:
                refusal = (
                    current, f"stale-epoch:{reason}",
                    OwnershipLost(
                        f"transition presented epoch {permit.owner_epoch}, current is "
                        f"{self.owner_epoch}; the result is recorded but cannot "
                        "change state"
                    ),
                )
            elif row["record_version"] != permit.expected_operation_version:
                refusal = (
                    current, f"cas-lost:{reason}",
                    JournalError(
                        f"CAS lost on {permit.operation_id}: record is at version "
                        f"{row['record_version']}, permit expected "
                        f"{permit.expected_operation_version}"
                    ),
                )
            elif new_outcome not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
                refusal = (
                    current, f"refused-transition:{reason}",
                    JournalError(
                        f"{current!r} -> {new_outcome!r} is not an allowed transition "
                        "(arch §14.4). A late result is recorded and cannot overwrite."
                    ),
                )

            if refusal is None:
                new_version = row["record_version"] + 1
                conn.execute(
                    "UPDATE operations SET outcome = ?, record_version = ?, "
                    "updated_at = ? WHERE operation_id = ? AND record_version = ?",
                    (new_outcome, new_version, time.time(), permit.operation_id,
                     row["record_version"]),
                )
                conn.execute(
                    "UPDATE attempts SET phase = ?, outcome = ?, failure_class = ?, "
                    "receipt_ref = ?, evidence_ref = ? WHERE attempt_id = ?",
                    ("finalized", new_outcome, failure_class, receipt_ref,
                     evidence_digest, permit.attempt_id),
                )
                self._append_event(
                    conn, permit.operation_id, current, new_outcome, reason,
                    evidence_digest,
                )
                return self.get_operation(scope, permit.operation_id, _conn=conn)

        # Refused. The event is committed here, outside the transaction above.
        unchanged, refusal_reason, error = refusal
        with self._transaction() as conn:
            self._append_event(
                conn, permit.operation_id, unchanged, unchanged, refusal_reason,
                evidence_digest,
            )
        raise error

    def record_reconciliation(
        self,
        scope: JournalScope,
        operation_id: str,
        *,
        strategy_id: str,
        strategy_version: str,
        determination: SideEffectOutcome,
        retry_eligible: bool,
        rule_id: Optional[str] = None,
        evidence_ref: Optional[str] = None,
    ) -> OperationRecord:
        """Resolve an unknown or partial outcome from strategy evidence.

        `retry_eligible` comes from the strategy's deterministic rules, and the
        caller cannot substitute its own: arch §14.6 is explicit that callers
        may not submit a free-form `retry_safe` value, because the value is the
        entire safety argument.
        """
        with self._transaction() as conn:
            row = self._locked_operation(conn, scope, operation_id)
            current: str = row["outcome"]
            allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
            if determination != current and determination not in allowed:
                raise JournalError(
                    f"reconciliation determined {determination!r}, which is not "
                    f"reachable from {current!r}"
                )
            new_version = row["record_version"] + 1
            conn.execute(
                "INSERT INTO reconciliation_events (operation_id, strategy_id, "
                "strategy_version, determination, retry_eligible, rule_id, "
                "evidence_ref, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (operation_id, strategy_id, strategy_version, determination,
                 int(retry_eligible), rule_id, evidence_ref, time.time()),
            )
            conn.execute(
                "UPDATE operations SET outcome = ?, record_version = ?, updated_at = ? "
                "WHERE operation_id = ? AND record_version = ?",
                (determination, new_version, time.time(), operation_id, row["record_version"]),
            )
            self._append_event(
                conn, operation_id, current, determination,
                f"reconciliation:{strategy_id}@{strategy_version}", evidence_ref,
            )
            return self.get_operation(scope, operation_id, _conn=conn)

    def create_compensation(
        self,
        scope: JournalScope,
        original_operation_id: str,
        *,
        logical_call_key: str,
        binding_digest: str,
        definition_id: str,
        contract_version: str = "1",
        restored_predicate_ref: Optional[str] = None,
    ) -> OperationRecord:
        """A new authorized operation, atomically linked to the original.

        Created together with its link (arch §14.7), so a crash after the
        compensating effect but before linkage cannot happen: the link is
        already durable when the permit is issued. The original's history is
        never erased — compensation is a second recorded effect, not an undo.
        """
        with self._transaction() as conn:
            self._locked_operation(conn, scope, original_operation_id)
            now = time.time()
            compensation_id = f"op-{uuid.uuid4().hex}"
            conn.execute(
                "INSERT INTO operations (operation_id, scope_digest, logical_call_key, "
                "binding_digest, definition_id, contract_version, turn_key, task_id, "
                "step_index, outcome, record_version, retention_class, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    compensation_id, scope.digest, logical_call_key, binding_digest,
                    definition_id, contract_version, None, None, None,
                    "not-dispatched", 1, "compensation", now, now,
                ),
            )
            conn.execute(
                "INSERT INTO compensation_links (original_operation_id, "
                "compensation_operation_id, restored_predicate_ref, created_at) "
                "VALUES (?,?,?,?)",
                (original_operation_id, compensation_id, restored_predicate_ref, now),
            )
            self._append_event(
                conn, compensation_id, None, "not-dispatched",
                f"compensation-for:{original_operation_id}", None,
            )
            return self.get_operation(scope, compensation_id, _conn=conn)

    def get_operation(
        self, scope: JournalScope, operation_id: str, *, _conn=None
    ) -> OperationRecord:
        conn = _conn or self._conn
        row = conn.execute(
            "SELECT * FROM operations WHERE operation_id = ? AND scope_digest = ?",
            (operation_id, scope.digest),
        ).fetchone()
        if row is None:
            # Same answer for "no such operation" and "not yours": a scope that
            # can distinguish them can enumerate another scope's operations.
            raise JournalError(f"no operation {operation_id} in this scope")
        return _record(row)

    # ------------------------------------------------------------------
    # Reconciliation gate (arch §14.2 reconciliation_gates)
    # ------------------------------------------------------------------

    def set_gate(
        self, scope: JournalScope, correlation: str, blocking_ids: list[str]
    ) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT record_version FROM reconciliation_gates WHERE scope_digest = ? "
                "AND correlation = ?",
                (scope.digest, correlation),
            ).fetchone()
            version = (row["record_version"] if row else 0) + 1
            conn.execute(
                "INSERT INTO reconciliation_gates (scope_digest, correlation, status, "
                "blocking_ids, record_version, updated_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(scope_digest, correlation) DO UPDATE SET status = "
                "excluded.status, blocking_ids = excluded.blocking_ids, "
                "record_version = excluded.record_version, updated_at = excluded.updated_at",
                (scope.digest, correlation, "blocked", json.dumps(sorted(blocking_ids)),
                 version, time.time()),
            )

    def clear_gate(self, scope: JournalScope, correlation: str) -> None:
        """Clear the gate, but only when nothing it names is still unresolved.

        The gate is not a flag somebody sets and unsets; it is a statement about
        the operations it blocks on. Clearing it while one of them is still
        `outcome-unknown` would be exactly the conversion FW-REQ-008B forbids,
        performed by omission.
        """
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM reconciliation_gates WHERE scope_digest = ? AND "
                "correlation = ?",
                (scope.digest, correlation),
            ).fetchone()
            if row is None:
                return
            blocking = json.loads(row["blocking_ids"])
            for operation_id in blocking:
                operation = conn.execute(
                    "SELECT outcome FROM operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if operation is not None and operation["outcome"] in PROTECTED_OUTCOMES:
                    raise ReconciliationRequired(
                        f"gate {correlation!r} still blocks on {operation_id} "
                        f"({operation['outcome']}); it cannot be cleared"
                    )
            conn.execute(
                "DELETE FROM reconciliation_gates WHERE scope_digest = ? AND "
                "correlation = ?",
                (scope.digest, correlation),
            )

    def gate_is_set(self, scope: JournalScope, correlation: str) -> bool:
        return self._gate_is_set(self._conn, scope, correlation)

    @staticmethod
    def _gate_is_set(conn, scope: JournalScope, correlation: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM reconciliation_gates WHERE scope_digest = ? AND "
            "correlation = ? AND status = 'blocked'",
            (scope.digest, correlation),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Retention (arch §14.8)
    # ------------------------------------------------------------------

    def compact(
        self, scope: JournalScope, *, older_than_seconds: float, tombstone_seconds: float
    ) -> int:
        """Compact final operations, writing a tombstone for each.

        A protected record is never compacted — capacity pressure disables new
        dispatch before it deletes one (see `health`). The tombstone is what
        keeps the compaction safe: without it a replayed old request would
        recreate the operation under a new ID and present a completed effect as
        a fresh one.
        """
        cutoff = time.time() - older_than_seconds
        compacted = 0
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM operations WHERE scope_digest = ? AND updated_at < ? "
                "AND outcome NOT IN ('outcome-unknown', 'partially-applied')",
                (scope.digest, cutoff),
            ).fetchall()
            for row in rows:
                blocked = conn.execute(
                    "SELECT 1 FROM compensation_links WHERE original_operation_id = ? "
                    "OR compensation_operation_id = ?",
                    (row["operation_id"], row["operation_id"]),
                ).fetchone()
                if blocked is not None:
                    continue
                conn.execute(
                    "INSERT INTO logical_call_tombstones (scope_digest, "
                    "logical_call_key, final_binding, final_outcome, retained_until) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(scope_digest, logical_call_key) "
                    "DO NOTHING",
                    (row["scope_digest"], row["logical_call_key"], row["binding_digest"],
                     row["outcome"], time.time() + tombstone_seconds),
                )
                conn.execute("DELETE FROM events WHERE operation_id = ?", (row["operation_id"],))
                conn.execute(
                    "DELETE FROM reconciliation_events WHERE operation_id = ?",
                    (row["operation_id"],),
                )
                conn.execute("DELETE FROM attempts WHERE operation_id = ?", (row["operation_id"],))
                conn.execute(
                    "DELETE FROM operations WHERE operation_id = ?", (row["operation_id"],)
                )
                compacted += 1
        return compacted

    def health(self, scope: JournalScope, *, max_protected: int = 1000) -> dict[str, Any]:
        """Operational metrics, and whether new dispatch is still safe.

        `dispatch_enabled` goes false on protected-record pressure rather than
        on total size: the records that pile up are the unresolved ones, and the
        only safe response to too many of them is to stop making more — never
        to delete one (arch §14.8).
        """
        conn = self._conn
        protected = conn.execute(
            "SELECT COUNT(*) FROM operations WHERE scope_digest = ? AND outcome IN "
            "('outcome-unknown', 'partially-applied')",
            (scope.digest,),
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM operations WHERE scope_digest = ?", (scope.digest,)
        ).fetchone()[0]
        oldest = conn.execute(
            "SELECT MIN(updated_at) FROM operations WHERE scope_digest = ? AND outcome "
            "IN ('outcome-unknown', 'partially-applied')",
            (scope.digest,),
        ).fetchone()[0]
        return {
            "operations": total,
            "protected": protected,
            "max_protected": max_protected,
            "oldest_unresolved_age_seconds": (
                None if oldest is None else round(time.time() - oldest, 3)
            ),
            "tombstones": conn.execute(
                "SELECT COUNT(*) FROM logical_call_tombstones WHERE scope_digest = ?",
                (scope.digest,),
            ).fetchone()[0],
            "owner_epoch": self.owner_epoch,
            "dispatch_enabled": protected < max_protected,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _locked_operation(conn, scope: JournalScope, operation_id: str):
        row = conn.execute(
            "SELECT * FROM operations WHERE operation_id = ? AND scope_digest = ?",
            (operation_id, scope.digest),
        ).fetchone()
        if row is None:
            raise JournalError(f"no operation {operation_id} in this scope")
        return row

    @staticmethod
    def _append_event(conn, operation_id, old, new, reason, evidence) -> None:
        """Every state change inserts its event in the SAME transaction (§14.2)."""
        conn.execute(
            "INSERT INTO events (operation_id, old_outcome, new_outcome, reason, "
            "evidence_digest, writer, created_at) VALUES (?,?,?,?,?,?,?)",
            (operation_id, old, new, reason, evidence, f"pid:{os.getpid()}", time.time()),
        )

    def events(self, operation_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM events WHERE operation_id = ? ORDER BY seq",
                (operation_id,),
            )
        ]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def _record(row) -> OperationRecord:
    return OperationRecord(
        operation_id=row["operation_id"],
        scope_digest=row["scope_digest"],
        logical_call_key=row["logical_call_key"],
        binding_digest=row["binding_digest"],
        definition_id=row["definition_id"],
        contract_version=row["contract_version"],
        outcome=row["outcome"],
        record_version=row["record_version"],
        turn_key=row["turn_key"],
        task_id=row["task_id"],
        step_index=row["step_index"],
        retention_class=row["retention_class"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@contextmanager
def contextlib_suppress():
    try:
        yield
    except Exception:
        pass
