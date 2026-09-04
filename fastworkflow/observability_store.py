"""SQLite observability store + background-writer TraceSink (Phase 2).

Implements the "black box" of the observability design
(docs/fastworkflow_observability_studio_design.md §3.2): one
``observability.sqlite3`` per workflow under the state root, holding
conversations, turn records, OTel-shaped spans, offloaded artifacts, train
runs, and a writer-health diagnostics row.

Structure:

- ``ObservabilityStore`` — schema + synchronous operations (id minting,
  upserts, reads, prune, forget-channel). Writes use short-lived
  ``BEGIN IMMEDIATE`` transactions on per-call connections (house precedent:
  ``kvstore.py``; the chatbot's read layer uses per-request connections so
  checkpointing never starves [R12]).
- ``SQLiteTraceSink`` — the TraceSink implementation: two queues ([R13]: a
  small turn-record/label queue with a bounded-timeout put — the only case a
  turn record may drop in v1 — and a droppable span queue bounded by
  ``FW_OBS_QUEUE_MAX``), drained by one daemon writer thread with batched
  transactions; ``close()`` (sentinel + bounded join) is wired to atexit and
  entry-point exit paths [R7]. Writer errors/drops land in the
  ``diagnostics`` table and are surfaced by the chatbot UI [R13].
- ``get_observability_sink()`` — process-wide factory honoring
  ``FW_OBSERVABILITY`` ([R4]: fastWorkflow's own entry points default it ON;
  library embedders opt in), one sink (= one writer thread) per DB path.

Durability class (Phase A, [R14]): everything is best-effort; a write failure
never fails a turn. Multi-process writers are supported on local filesystems
only (WAL constraint — the state root must not be NFS).
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.utils.logging import logger

SCHEMA_VERSION = 1
CAPTURE_PROFILE_VAR = "FW_OBS_CAPTURE_PROFILE"

TERMINAL_TURN_STATUSES = frozenset({"completed", "failed", "cancelled", "abandoned"})

# Defaults per design §5.
_DEFAULT_DB_MAX_BYTES = 1_073_741_824
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_INLINE_ARTIFACT_BYTES = 262_144
_DEFAULT_QUEUE_MAX = 10_000

# Turn-record queue: small and separate [R13]. The bounded-timeout put is the
# only case a turn record may drop in v1.
_RECORD_QUEUE_MAX = 256
_RECORD_PUT_TIMEOUT_S = 2.0
_RECORD_BUSY_MAX_RETRIES = 5

# Sync-first turn-record writes (Phase 7 §2.4, rulings I1/I6/C8/C9).
_DEFAULT_SYNC_WRITE_TIMEOUT_S = 5
_DEFAULT_SYNC_BREAKER_COOLDOWN_S = 60
# Terminal records that fell back to the queue and have not been confirmed
# durable ride this ring until a retry lands them. Bounded: it is a memory
# holder on a path that only runs when the DB is already unhealthy, and the
# window the history trim defers by is bounded with it (ruling I1/I2).
_PENDING_RETRY_MAX = 64

_PRUNE_BATCH_ROWS = 5_000
_PRUNE_MAX_BATCHES = 20
_pruning_lock = threading.Lock()
_pruning_suppression_depth = 0

# Additive feature markers deliberately do not bump SCHEMA_VERSION. Readers use
# these markers to avoid querying tables/columns that older snapshots lack.
FEATURE_DISTILLATION_V1 = "distillation_v1"
FEATURE_EXPERIMENTS_V1 = "experiments_v1"
FEATURE_EXPERIMENT_LIFECYCLE_V1 = "experiment_lifecycle_v1"

CAPTURE_POLICY_VERSION = "1"
CAPTURE_REGIME_DIAGNOSTIC = "observability_capture_regime"


@dataclass(frozen=True)
class WriterHealthDelta:
    """Change in writer-health counters across one measured interval."""

    records_dropped: int = 0
    spans_dropped: int = 0
    write_errors: int = 0
    refused_terminal_writes: int = 0
    busy_retries: int = 0
    sync_fallbacks: int = 0
    incomparable: bool = False

    @property
    def evidence_valid(self) -> bool:
        return not self.incomparable and self.records_dropped == 0

    def problems(self) -> tuple[str, ...]:
        problems: list[str] = []
        if self.incomparable:
            problems.append("writer health could not be compared")
        if self.records_dropped:
            problems.append(f"{self.records_dropped} turn record(s) were dropped")
        if self.spans_dropped:
            problems.append(f"{self.spans_dropped} span(s) were dropped")
        if self.write_errors:
            problems.append(f"{self.write_errors} writer error(s) occurred")
        return tuple(problems)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {
            "records_dropped": self.records_dropped,
            "spans_dropped": self.spans_dropped,
            "write_errors": self.write_errors,
            "refused_terminal_writes": self.refused_terminal_writes,
            "busy_retries": self.busy_retries,
            "sync_fallbacks": self.sync_fallbacks,
            "incomparable": self.incomparable,
            "evidence_valid": self.evidence_valid,
            "problems": list(self.problems()),
        }


def observability_config() -> dict[str, str]:
    return {
        "FW_OBSERVABILITY": _env("FW_OBSERVABILITY", "1"),
        CAPTURE_PROFILE_VAR: _env(CAPTURE_PROFILE_VAR, "debug"),
    }


def health_delta(
    before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]
) -> WriterHealthDelta:
    fields = (
        "records_dropped",
        "spans_dropped",
        "write_errors",
        "refused_terminal_writes",
        "busy_retries",
        "sync_fallbacks",
    )
    values = {
        field: max(0, int((after or {}).get(field) or 0) - int((before or {}).get(field) or 0))
        for field in fields
    }
    return WriterHealthDelta(**values, incomparable=before is None or after is None)


@contextlib.contextmanager
def suppress_pruning():
    global _pruning_suppression_depth
    with _pruning_lock:
        _pruning_suppression_depth += 1
    try:
        yield
    finally:
        with _pruning_lock:
            _pruning_suppression_depth -= 1


def pruning_suppressed() -> bool:
    with _pruning_lock:
        return _pruning_suppression_depth > 0


class IncompatibleObservabilityDB(RuntimeError):
    """The DB was written by a newer fastWorkflow; readers refuse it [R11]."""


class ExperimentNotFound(KeyError):
    """An experiment write matched no row.

    Raised rather than passed over: `clear_conversations` is an HTTP-triggered
    whole-DB erase that can land while a harness is running, and a silent no-op
    there leaves turns labelled against a container that no longer exists
    (`[XR15]`).
    """

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(f"no experiment {experiment_id!r} in this database")


class ExperimentIsClosed(ValueError):
    """An attempt was written to an experiment that is no longer running.

    `complete` and `invalid` are terminal: their attempt rows are the evidence a
    reported score rests on, and a second run under the same id would overwrite
    them in place.
    """

    def __init__(self, experiment_id: str, status: str) -> None:
        self.experiment_id = experiment_id
        self.status = status
        super().__init__(
            f"experiment {experiment_id!r} is {status!r}, not running; its "
            "attempts are closed. Start a new experiment rather than rewriting "
            "the record a score was reported from."
        )


class AttemptValueConflict(ValueError):
    """A terminal attempt value was rewritten to a different value."""


class CaptureRegimeChanged(ValueError):
    """An experiment was re-created under a different capture profile/policy."""

    def __init__(self, experiment_id: str, stored: str, incoming: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(
            f"experiment {experiment_id!r} was captured under {stored} and is "
            f"now being written under {incoming}. The two halves would not be "
            "measuring the same columns; record the second half as its own "
            "experiment."
        )


class HypothesisIsWriteOnce(ValueError):
    """A stored hypothesis was rewritten to a different value (`[XR12]`).

    One mutable description (`notes`) beside one immutable one is what makes the
    immutable one mean anything: a pre-registered prediction that can be revised
    after the outcome is not a pre-registration.
    """

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(
            f"experiment {experiment_id!r} already has a hypothesis; it is "
            "write-once by design. Record the revision in `notes` instead."
        )


def _env(name: str, default: str) -> str:
    """FW_* knob: process env first, then the workflow env file, then default."""
    value = os.environ.get(name)
    if value is None or value == "":
        value = fastworkflow._env_vars.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(values: list[Any], size: int = 400) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _iso_to_ms(value: Optional[str]) -> int:
    """ISO timestamp → ms epoch (legacy conversation-record convention)."""
    if not value:
        return 0
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ----------------------------------------------------------------------
# Redaction [R20]
# ----------------------------------------------------------------------

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")

# Known credential shapes, scrubbed independently of the environment.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"),
]

_REDACTED = "[REDACTED]"


class Redactor:
    """Sink-boundary scrub of credential shapes and loaded secret env values.

    Collects the VALUES of every ``*_API_KEY``/``*_TOKEN``-style variable from
    the process environment and the loaded fastworkflow env files, and removes
    them (plus well-known credential shapes) from any text persisted.
    """

    def __init__(self) -> None:
        values: set[str] = set()
        sources: list[dict] = [dict(os.environ)]
        env_vars = getattr(fastworkflow, "_env_vars", None)
        if isinstance(env_vars, dict):
            sources.append(env_vars)
        for source in sources:
            for key, value in source.items():
                if not isinstance(value, str) or len(value) < 8:
                    continue
                upper = str(key).upper()
                # Infix match: the house convention is LITELLM_API_KEY_<ROLE>,
                # so the secret marker is not necessarily the suffix.
                if any(marker in upper for marker in _SECRET_ENV_SUFFIXES):
                    values.add(value)
        # Longest first so partial overlaps cannot resurrect a suffix.
        self._values = sorted(values, key=len, reverse=True)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for value in self._values:
            if value in text:
                text = text.replace(value, _REDACTED)
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(_REDACTED, text)
        return text


# ----------------------------------------------------------------------
# Turn-record serialization (size policy [R10], envelopes, traceback gate)
# ----------------------------------------------------------------------


def _sanitize_json_value(value: Any) -> Any:
    """Coerce a dumped value into JSON-safe form; non-serializable values
    become placeholder envelopes rather than failing the record."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return {
        "__fw_unserializable__": type(value).__name__,
        "repr": repr(value)[:1024],
    }


def serialize_turn_result(turn_result: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a TurnResult into (turn_row, artifact_rows) at emission time.

    - ``record_json`` holds the full internal TurnResult (post-envelope,
      pre-redaction — the sink redacts the serialized text) [R10].
    - Any artifact value over ``FW_OBS_INLINE_ARTIFACT_BYTES`` is replaced in
      place by a ref envelope; the artifacts table is the only value holder.
    - ``traceback`` artifacts persist only under FW_OBS_CAPTURE_TRACEBACKS=1
      [R20].

    Runs in the caller thread so the row snapshots the turn as emitted (the
    accumulator's CommandOutput objects mutate on resume).
    """
    turn_output = turn_result.turn_output
    inline_limit = _env_int("FW_OBS_INLINE_ARTIFACT_BYTES", _DEFAULT_INLINE_ARTIFACT_BYTES)
    capture_tracebacks = _env("FW_OBS_CAPTURE_TRACEBACKS", "0") == "1"

    try:
        record = turn_result.model_dump(mode="python")
    except Exception:
        record = {"turn_output": {"turn_key": turn_output.turn_key}}
    record = _sanitize_json_value(record)
    # computed_field `success` is included by model_dump; make sure it is
    # present even on the fallback path.
    record.setdefault("turn_output", {}).setdefault("success", turn_output.success)

    turn_key = turn_output.turn_key
    channel_id = turn_result.channel_id or ""
    artifact_rows: list[dict[str, Any]] = []

    for command_output in record.get("turn_output", {}).get("command_outputs", []):
        response = command_output.get("command_response") or {}
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for key in list(artifacts.keys()):
            if key == "traceback" and not capture_tracebacks:
                artifacts[key] = "[suppressed; set FW_OBS_CAPTURE_TRACEBACKS=1]"
                continue
            value_json = json.dumps(artifacts[key], ensure_ascii=False)
            size = len(value_json.encode("utf-8"))
            if size <= inline_limit:
                continue
            artifact_id = uuid.uuid4().hex
            sha256 = hashlib.sha256(value_json.encode("utf-8")).hexdigest()
            content_type = (
                "text/plain" if isinstance(artifacts[key], str) else "application/json"
            )
            artifact_rows.append(
                {
                    "artifact_id": artifact_id,
                    "turn_key": turn_key,
                    "channel_id": channel_id,
                    "span_id": None,
                    "key": key,
                    "content_type": content_type,
                    "size_bytes": size,
                    "sha256": sha256,
                    "inline_value": value_json.encode("utf-8"),
                    "error": None,
                }
            )
            # Envelope shape per final spec [A10] / this design [R10].
            artifacts[key] = {
                "__fw_artifact_ref__": artifact_id,
                "size": size,
                "content_type": content_type,
                "content_encoding": None,
                "error": None,
            }

    turn_row = {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": turn_result.conversation_id,
        "ordinal": turn_result.ordinal,
        "user_message": turn_result.user_message or "",
        "refined_user_message": turn_result.refined_user_message,
        "entry_workflow_name": turn_result.entry_workflow_name or "",
        "entry_context": turn_result.entry_context or "",
        "status": turn_output.status.value,
        "success": 1 if turn_output.success else 0,
        "failure_reason": turn_output.failure_reason,
        "answer": turn_output.answer or "",
        # Stamped by WEC._build_turn_result only when the turn appended a
        # conversation-history entry, so these are exactly the rows the
        # _USABLE_TURN_FILTER admits as conversation memory.
        "conversation_summary": getattr(turn_result, "conversation_summary", None),
        "conversation_traces": getattr(turn_result, "conversation_traces", None),
        "started_at": (
            turn_result.started_at.isoformat() if turn_result.started_at else None
        ),
        "completed_at": (
            turn_result.completed_at.isoformat() if turn_result.completed_at else None
        ),
        "suspended_ms": int(turn_result.suspended_ms or 0),
        "continuation_of": turn_result.continuation_of,
        # The experiment container's labels (`fix-bn1` `[XR17]`). Bound on the
        # WEC before the turn and copied off the TurnResult here, so they take
        # the same path as channel_id rather than being stitched on by a later
        # query. NULL on every ordinary turn. `upsert_turn_row` derives its
        # column list from this dict, so these three keys are also what writes
        # them -- and a key here with no matching column raises on `_sync_write`
        # and trips the sync breaker, which is why the DDL and this projection
        # must ship together.
        "experiment_id": turn_result.experiment_id,
        "task_id": turn_result.task_id,
        "attempt": (
            None if turn_result.attempt is None else int(turn_result.attempt)
        ),
        "record_version": 1,
        "record_json": json.dumps(record, ensure_ascii=False),
    }
    return turn_row, artifact_rows


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------

_SCHEMA_STATEMENTS = [
    # experiment_id/task_id/attempt are the experiment container's labels
    # (`[XR4]`). They are here AND in the guarded ALTER block in _ensure_schema:
    # on a fresh DB this literal is what creates them (PRAGMA table_info returns
    # nothing, so the ALTER guard is False), on an existing DB the ALTER is.
    # NULL means "not part of an experiment", so no backfill is needed.
    """CREATE TABLE IF NOT EXISTS conversations (
        channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
        topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
        started_at TEXT, last_turn_at TEXT, updated_at TEXT,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        PRIMARY KEY (channel_id, conversation_id))""",
    """CREATE TABLE IF NOT EXISTS conversation_counters (
        channel_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS turns (
        turn_key TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
        user_message TEXT NOT NULL, refined_user_message TEXT,
        entry_workflow_name TEXT, entry_context TEXT,
        status TEXT NOT NULL, success INTEGER NOT NULL,
        failure_reason TEXT, answer TEXT,
        conversation_summary TEXT, conversation_traces TEXT,
        started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
        continuation_of TEXT, record_version INTEGER NOT NULL,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        record_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
        parent_span_id TEXT, name TEXT NOT NULL,
        kind TEXT NOT NULL,
        channel_id TEXT,
        command_name TEXT, context TEXT,
        start_ns INTEGER NOT NULL, end_ns INTEGER,
        status TEXT NOT NULL, attributes TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL,
        channel_id TEXT,
        span_id TEXT, key TEXT NOT NULL, content_type TEXT,
        size_bytes INTEGER, sha256 TEXT,
        inline_value BLOB, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS train_runs (
        run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
        completed_at TEXT, metrics_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS diagnostics (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        hypothesis TEXT,
        notes TEXT,
        arm TEXT,
        baseline_experiment_id TEXT,
        status TEXT NOT NULL,
        invalid_reason TEXT,
        invalid_detail TEXT,
        declared_tasks INTEGER NOT NULL,
        declared_attempts INTEGER NOT NULL,
        workflow_name TEXT,
        capture_profile TEXT NOT NULL,
        capture_policy_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS experiment_attempts (
        experiment_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        channel_id TEXT NOT NULL,
        conversation_id INTEGER,
        outcome TEXT,
        outcome_source TEXT,
        reward REAL,
        restarts INTEGER NOT NULL DEFAULT 0,
        started_at TEXT NOT NULL,
        execution_status TEXT,
        execution_finished_at TEXT,
        finished_at TEXT,
        detail_json TEXT,
        source_attempt_json TEXT,
        PRIMARY KEY (experiment_id, task_id, attempt))""",
    """CREATE TABLE IF NOT EXISTS experiment_evidence_runs (
        experiment_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        evidence_run_id TEXT NOT NULL,
        valid INTEGER NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        record_json TEXT NOT NULL,
        PRIMARY KEY (experiment_id, seq))""",
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(channel_id, conversation_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_key)",
    "CREATE INDEX IF NOT EXISTS idx_turns_experiment ON turns(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_experiment_attempt ON conversations(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_baseline ON experiments(baseline_experiment_id) WHERE baseline_experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_attempts_channel ON experiment_attempts(channel_id)",
]


class ObservabilityStore:
    """Schema owner + synchronous operations on one observability DB.

    Thread/process-safe by construction: every method opens its own
    short-lived WAL connection (timeout=30, ``BEGIN IMMEDIATE`` for writes).
    """

    def __init__(self, db_path: str, *, migrate: bool = True) -> None:
        self.db_path = db_path
        if migrate:
            self._ensure_schema()
        self._features = self._load_features()

    @staticmethod
    def open_for_annotation(db_path: str) -> "ObservabilityStore":
        """Open an existing DB read-write without creating or migrating it."""
        return ObservabilityStore(db_path, migrate=False)

    def _store_redactor(self) -> Redactor:
        redactor = getattr(self, "_redactor", None)
        if redactor is None:
            redactor = Redactor()
            self._redactor = redactor
        return redactor

    # -- connections ----------------------------------------------------

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            try:
                os.chmod(parent, 0o700)  # [R4]
            except OSError:
                pass
        fresh = not os.path.exists(self.db_path)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            if fresh:
                # auto_vacuum must be set at creation, before any table [R12].
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build reads up to "
                    f"v{SCHEMA_VERSION}. Refusing to open a newer DB [R11]."
                )
            # Existing databases need the experiment labels before the indexes
            # below are created. Each column is guarded separately so a
            # partially interrupted migration self-heals.
            for table in ("turns", "conversations"):
                cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if not cols:
                    continue
                for column, declaration in (
                    ("experiment_id", "TEXT"),
                    ("task_id", "TEXT"),
                    ("attempt", "INTEGER"),
                ):
                    if column not in cols:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                        )
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            # Pre-release column migration (schema v1 was never shipped, but
            # dev DBs created by earlier work-in-progress builds exist):
            # CREATE IF NOT EXISTS cannot add columns to an existing table.
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "updated_at" not in existing_cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT")
            attempt_cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(experiment_attempts)"
                ).fetchall()
            }
            for column, declaration in (
                ("execution_status", "TEXT"),
                ("execution_finished_at", "TEXT"),
                ("source_attempt_json", "TEXT"),
            ):
                if attempt_cols and column not in attempt_cols:
                    conn.execute(
                        f"ALTER TABLE experiment_attempts ADD COLUMN "
                        f"{column} {declaration}"
                    )
            if attempt_cols and "execution_finished_at" not in attempt_cols:
                # The old compatibility operation wrote execution and outcome
                # together. Preserve that meaning when opening a pre-split DB.
                conn.execute(
                    """UPDATE experiment_attempts
                          SET execution_status=CASE
                                  WHEN finished_at IS NULL THEN NULL
                                  ELSE 'completed' END,
                              execution_finished_at=finished_at
                        WHERE execution_finished_at IS NULL"""
                )
            if found < SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            # Write probe: every statement above is a no-op on an existing
            # schema, so an unwritable DB would otherwise open "successfully"
            # and fail on every later write. Fail here instead, so the factory
            # degrades to no-sink at open time.
            conn.execute(
                """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                ("schema_opened", json.dumps({"schema_version": SCHEMA_VERSION}), _utcnow_iso()),
            )
            self._merge_schema_features(
                conn,
                [FEATURE_EXPERIMENTS_V1, FEATURE_EXPERIMENT_LIFECYCLE_V1],
            )
            conn.execute(
                """INSERT INTO diagnostics (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                (
                    CAPTURE_REGIME_DIAGNOSTIC,
                    json.dumps(
                        {
                            "capture_profile": _env(
                                CAPTURE_PROFILE_VAR, "debug"
                            ),
                            "capture_policy_version": CAPTURE_POLICY_VERSION,
                        }
                    ),
                    _utcnow_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self.db_path, 0o600)  # [R4]
            wal = f"{self.db_path}-wal"
            if os.path.exists(wal):
                os.chmod(wal, 0o600)
        except OSError:
            pass

    @staticmethod
    def _merge_schema_features(
        conn: sqlite3.Connection, features: list[str]
    ) -> None:
        """Merge feature markers without dropping markers from other builds."""
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key='schema_features'"
        ).fetchone()
        known: list[str] = []
        if row is not None:
            try:
                loaded = json.loads(row[0])
                if isinstance(loaded, list):
                    known = [str(name) for name in loaded]
            except (ValueError, TypeError):
                known = []
        merged = sorted(set(known) | set(features))
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            ("schema_features", json.dumps(merged), _utcnow_iso()),
        )

    def _load_features(self) -> frozenset[str]:
        """Read feature markers, falling back to additive schema detection."""
        conn = None
        try:
            conn = self._connect(timeout=5.0)
            row = conn.execute(
                "SELECT value FROM diagnostics WHERE key='schema_features'"
            ).fetchone()
            if row is not None:
                loaded = json.loads(row[0])
                if isinstance(loaded, list):
                    return frozenset(str(name) for name in loaded)
            detected: set[str] = set()
            turn_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "experiment_id" in turn_cols:
                detected.add(FEATURE_EXPERIMENTS_V1)
            attempt_cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(experiment_attempts)"
                ).fetchall()
            }
            if {
                "execution_status",
                "execution_finished_at",
                "source_attempt_json",
            } <= attempt_cols:
                detected.add(FEATURE_EXPERIMENT_LIFECYCLE_V1)
            span_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(spans)").fetchall()
            }
            if "distillation_pass" in span_cols:
                detected.add(FEATURE_DISTILLATION_V1)
            return frozenset(detected)
        except Exception:
            return frozenset()
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def has_feature(self, name: str) -> bool:
        return name in self._features

    def capture_regime(self) -> Optional[tuple[str, str]]:
        """Return the regime installed by the process that owns this store."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM diagnostics WHERE key=?",
                    (CAPTURE_REGIME_DIAGNOSTIC,),
                ).fetchone()
            if row is None:
                return None
            value = json.loads(row["value"])
            return (
                str(value["capture_profile"]),
                str(value["capture_policy_version"]),
            )
        except (KeyError, TypeError, ValueError, sqlite3.Error):
            return None

    # -- identity [R1] ---------------------------------------------------

    def mint_conversation_id(
        self,
        channel_id: str,
        legacy_floor: int = 0,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> int:
        """Atomically reserve the next conversation id for a channel.

        The observability DB is the sole id-minting authority; dual-write
        consumers (the legacy conversation store) consume the same id so the
        stores cannot diverge on identity.

        Minting is a per-channel monotonic counter (never MAX-derived), so
        forget-channel/prune can never cause id reuse; the counter is seeded
        at first mint from ``max(existing rows, legacy_floor)`` — callers
        crossing the Phase-7 cutover pass the legacy store's
        ``last_conversation_id`` as ``legacy_floor`` so ids never alias
        against pre-cutover conversations (review ruling C2).

        Uses a SHORT busy timeout (ruling C9's principle): minting runs
        synchronously in request paths — FastAPI's event loop included — so a
        contended DB must fail fast (callers degrade to the legacy reserve
        path) rather than stall every channel for the writer timeout.
        """
        with self._connect(
            timeout=float(_env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", 5))
        ) as conn:
            conn.execute("BEGIN IMMEDIATE")
            counter = conn.execute(
                "SELECT next_id FROM conversation_counters WHERE channel_id=?",
                (channel_id,),
            ).fetchone()
            max_row = conn.execute(
                "SELECT COALESCE(MAX(conversation_id), 0) FROM conversations WHERE channel_id=?",
                (channel_id,),
            ).fetchone()
            floor = max(int(max_row[0]), int(legacy_floor or 0))
            next_id = int(counter["next_id"]) if counter is not None else 1
            new_id = max(next_id, floor + 1)
            conn.execute(
                """INSERT INTO conversation_counters (channel_id, next_id) VALUES (?, ?)
                   ON CONFLICT(channel_id) DO UPDATE SET
                     next_id=MAX(conversation_counters.next_id, excluded.next_id)""",
                (channel_id, new_id + 1),
            )
            now = _utcnow_iso()
            # The experiment labels ride the mint because this is where the
            # conversation row is created, and an attempt IS a conversation
            # (`[XR4]`). Scrub-only, on the same terms as the turn path, so the
            # two copies stay byte-identical and joinable (`[XR7]`).
            redactor = self._store_redactor()
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at, updated_at,
                    experiment_id, task_id, attempt)
                   VALUES (?, ?, NULL, NULL, 'open', 1, ?, NULL, ?, ?, ?, ?)""",
                (
                    channel_id,
                    new_id,
                    now,
                    now,
                    experiment_id,
                    redactor.redact(task_id),
                    None if attempt is None else int(attempt),
                ),
            )
            conn.commit()
        return new_id

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> str:
        """Upsert a conversation's topic/summary ([R15]; labels are mutable).

        A None topic or summary preserves the stored value, so the blank-topic
        policy — a failed generation never clobbers a good title — carries
        over from the legacy store. Topic uniquification runs inside the same
        transaction as the write (review ruling I9: no TOCTOU across the async
        label path; Python-side casefold, never SQLite's ASCII-only lower()).

        Returns the topic actually STORED — collision-suffixed where one was
        written, or the preserved existing title on a blank generation. A
        caller that reports or logs the label must use this rather than its own
        candidate, which is the contract the legacy store's
        ``update_conversation_topic_summary`` established (ruling I9).
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            stored = self.apply_label_txn(
                conn, channel_id, conversation_id, topic, summary
            )
            conn.commit()
        return stored

    def apply_label_txn(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> str:
        """The single label-write enforcement point (caller owns the txn).

        Returns the stored topic (see ``record_conversation_label``).
        """
        if topic is not None:
            topic = self._unique_topic_in_txn(
                conn, channel_id, topic, exclude_conversation_id=conversation_id
            )
            if not topic:
                # Blank stays the "no title yet" sentinel — never stored as a
                # title (legacy blank-topic policy).
                topic = None
        now = _utcnow_iso()
        conn.execute(
            """INSERT INTO conversations
               (channel_id, conversation_id, topic, summary, status,
                next_ordinal, started_at, last_turn_at, updated_at)
               VALUES (?, ?, ?, ?, 'open', 1, ?, NULL, ?)
               ON CONFLICT(channel_id, conversation_id) DO UPDATE SET
                 topic=COALESCE(excluded.topic, conversations.topic),
                 summary=COALESCE(excluded.summary, conversations.summary),
                 updated_at=excluded.updated_at""",
            (channel_id, conversation_id, topic, summary, now, now),
        )
        if topic is not None:
            return topic
        row = conn.execute(
            "SELECT topic FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        return (row["topic"] or "") if row is not None else ""

    @staticmethod
    def _topic_norm(value: str) -> str:
        # Python casefolding — SQLite lower() is ASCII-only (ruling I9).
        return value.casefold().strip()

    def _unique_topic_in_txn(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        candidate_topic: str,
        exclude_conversation_id: Optional[int] = None,
    ) -> str:
        """Legacy-faithful uniquification: case/whitespace-insensitive
        collision suffixing, blank exemption decided before the scan,
        self-exclusion, each suffixed candidate renormalized."""
        if not self._topic_norm(candidate_topic):
            return ""
        rows = conn.execute(
            "SELECT conversation_id, topic FROM conversations "
            "WHERE channel_id=? AND topic IS NOT NULL",
            (channel_id,),
        ).fetchall()
        existing = {
            self._topic_norm(row["topic"])
            for row in rows
            if row["conversation_id"] != exclude_conversation_id and row["topic"]
        }
        final_topic = candidate_topic
        collision_count = 0
        while self._topic_norm(final_topic) in existing:
            collision_count += 1
            final_topic = f"{candidate_topic} {collision_count}"
        return final_topic

    # -- writes (used by the writer thread; also callable directly) ------

    def upsert_span_rows(self, conn: sqlite3.Connection, spans: list[tracing.Span], redactor: Redactor) -> None:
        for span in spans:
            attributes = redactor.redact(
                json.dumps(_sanitize_json_value(span.attributes), ensure_ascii=False)
            )
            conn.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, kind, channel_id,
                    command_name, context, start_ns, end_ns, status, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
                     end_ns=COALESCE(excluded.end_ns, spans.end_ns),
                     status=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                 THEN excluded.status ELSE spans.status END,
                     attributes=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                     THEN excluded.attributes ELSE spans.attributes END,
                     command_name=COALESCE(excluded.command_name, spans.command_name),
                     context=COALESCE(excluded.context, spans.context)""",
                (
                    span.span_id,
                    span.trace_id,
                    span.parent_span_id,
                    span.name,
                    span.kind,
                    span.channel_id,
                    span.command_name,
                    span.context,
                    span.start_ns,
                    span.end_ns,
                    span.status,
                    attributes,
                ),
            )

    def upsert_turn_row(
        self,
        conn: sqlite3.Connection,
        turn_row: dict[str, Any],
        artifact_rows: list[dict[str, Any]],
        redactor: Redactor,
    ) -> bool:
        """Apply the [R2] lifecycle: INSERT at first emission; one guarded
        status transition to a terminal status; write-once for rows already
        terminal (identical-content retries claim idempotent success).

        Returns False when a conflicting write against a terminal row was
        refused (counted by the caller).
        """
        turn_row = dict(turn_row)
        # failure_reason is included because it can embed exception/provider
        # text (e.g. a LiteLLM AuthenticationError body) — the [R20] scenario.
        # task_id is SCRUB-ONLY and not policed (`[XR6]`/`[XR7]`): policing it
        # would withhold nothing (the plaintext rides into record_json above,
        # which `_apply_capture_policy` never walks) while breaking every
        # equality lookup the experiment read layer is built on. It must be
        # scrubbed on BOTH label routes -- here and in mint_conversation_id --
        # and in the container tables, or the copies stop being joinable.
        #
        # `experiment_id` is deliberately NOT in this list. It is a machine-minted
        # opaque id (`exp-<32 hex>`, `[XR1]`) and the join key of every score, and
        # it is stored raw in `experiments`/`experiment_attempts`/
        # `experiment_evidence_runs`. Scrubbing it here and not there is what
        # makes a join silently return nothing -- the same class of defect the
        # scrub-on-both-routes rule above exists to prevent. Every other
        # machine-minted join key in this file (turn_key, trace_id, run_id,
        # artifact_id) is likewise stored as-is.
        for text_col in (
            "user_message",
            "refined_user_message",
            "answer",
            "failure_reason",
            "conversation_summary",
            "conversation_traces",
            "task_id",
            "record_json",
        ):
            if turn_row.get(text_col):
                turn_row[text_col] = redactor.redact(turn_row[text_col])

        existing = conn.execute(
            "SELECT status, record_json FROM turns WHERE turn_key=?",
            (turn_row["turn_key"],),
        ).fetchone()

        if existing is not None and existing["status"] in TERMINAL_TURN_STATUSES:
            if (
                existing["status"] == turn_row["status"]
                and existing["record_json"] == turn_row["record_json"]
            ):
                return True  # idempotent retry
            if turn_row["status"] not in TERMINAL_TURN_STATUSES:
                # A late-arriving pre-terminal emission (e.g. the queued
                # awaiting_user record draining after the terminal sync write)
                # is expected ordering noise, not a violation — ignore it
                # without counting (ruling C8).
                return True
            logger.warning(
                f"Refusing write to terminal turn row {turn_row['turn_key']} "
                f"(stored {existing['status']}, incoming {turn_row['status']}) [R2]"
            )
            return False

        # Ordinal assignment on first insert of a conversation-bound turn.
        if (
            existing is None
            and turn_row.get("conversation_id") is not None
            and turn_row.get("ordinal") is None
        ):
            turn_row["ordinal"] = self._assign_ordinal(
                conn,
                turn_row["channel_id"],
                turn_row["conversation_id"],
                experiment_id=turn_row.get("experiment_id"),
                task_id=turn_row.get("task_id"),
                attempt=turn_row.get("attempt"),
            )

        columns = list(turn_row.keys())
        placeholders = ", ".join("?" for _ in columns)
        update_cols = [c for c in columns if c != "turn_key"]
        if existing is not None:
            # Keep the ordinal assigned at first insert.
            update_cols = [c for c in update_cols if c != "ordinal"]
        assignments = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        conn.execute(
            f"INSERT INTO turns ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(turn_key) DO UPDATE SET {assignments}",
            [turn_row[c] for c in columns],
        )

        if turn_row.get("conversation_id") is not None:
            now = _utcnow_iso()
            conn.execute(
                """UPDATE conversations SET last_turn_at=?, updated_at=?
                   WHERE channel_id=? AND conversation_id=?""",
                (now, now, turn_row["channel_id"], turn_row["conversation_id"]),
            )

        for artifact in artifact_rows:
            inline_value = artifact.get("inline_value")
            if isinstance(inline_value, (bytes, bytearray)):
                redacted = redactor.redact(
                    bytes(inline_value).decode("utf-8", errors="replace")
                )
                inline_value = redacted.encode("utf-8")
            conn.execute(
                """INSERT INTO artifacts
                   (artifact_id, turn_key, channel_id, span_id, key, content_type,
                    size_bytes, sha256, inline_value, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(artifact_id) DO NOTHING""",
                (
                    artifact["artifact_id"],
                    artifact["turn_key"],
                    artifact.get("channel_id"),
                    artifact.get("span_id"),
                    artifact["key"],
                    artifact.get("content_type"),
                    artifact.get("size_bytes"),
                    artifact.get("sha256"),
                    inline_value,
                    artifact.get("error"),
                ),
            )
        return True

    def _assign_ordinal(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        conversation_id: int,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> int:
        row = conn.execute(
            "SELECT next_ordinal FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        if row is None:
            # Conversation row not minted here (e.g. restored session) —
            # create it so ordinals stay dense from 1.
            # The labels are copied off the turn row being inserted: this row
            # was not minted here (restored session, or a turn whose conversation
            # predates the experiment binding), so the turn is the only carrier.
            #
            # Scrubbed HERE rather than trusting the caller: three routes reach
            # this insert (`upsert_turn_row`'s text loop, which has scrubbed;
            # `reserve_turn_ordinal` from the sink's degraded queue path, which
            # has not; and a direct call), and a value scrubbed on one route and
            # not another is what makes the turns/conversations join silently
            # return nothing. The scrub is idempotent, so doing it again is free.
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at,
                    experiment_id, task_id, attempt)
                   VALUES (?, ?, NULL, NULL, 'open', 2, ?, NULL, ?, ?, ?)""",
                (
                    channel_id,
                    conversation_id,
                    _utcnow_iso(),
                    experiment_id,
                    self._store_redactor().redact(task_id),
                    None if attempt is None else int(attempt),
                ),
            )
            return 1
        ordinal = int(row["next_ordinal"] or 1)
        conn.execute(
            "UPDATE conversations SET next_ordinal=? WHERE channel_id=? AND conversation_id=?",
            (ordinal + 1, channel_id, conversation_id),
        )
        return ordinal

    def reserve_turn_ordinal(
        self,
        channel_id: str,
        conversation_id: int,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        """Reserve a turn ordinal in a tiny standalone transaction.

        Used by the sync-first emit's degraded fallback so ordinals stay
        chronological even when the row itself is queued (ruling I6).
        Returns None when the reservation itself cannot be made.
        """
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                # The labels ride along because `_assign_ordinal` CREATES the
                # conversations row when it is missing: reserving without them
                # would mint an unlabelled attempt conversation on the degraded
                # path, and the UNIQUE index would then refuse the labelled one.
                ordinal = self._assign_ordinal(
                    conn,
                    channel_id,
                    conversation_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    attempt=attempt,
                )
                conn.commit()
            return ordinal
        except Exception:
            return None

    # -- consolidation reads (Phase 7; "usable rows" filter per ruling I4) --
    #
    # A turns row exists for every logical turn — cancelled turns, abandoned
    # suspensions, and turns whose history never grew carry a NULL
    # conversation_summary. Conversation-memory consumers must therefore see
    # only rows that correspond to a real conversation-history entry:
    _USABLE_TURN_FILTER = (
        "status IN ('completed','failed') AND conversation_summary IS NOT NULL"
    )

    def count_usable_turns(self, channel_id: str, conversation_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM turns WHERE channel_id=? AND conversation_id=? "
                f"AND {self._USABLE_TURN_FILTER}",
                (channel_id, conversation_id),
            ).fetchone()
            return int(row[0])

    def get_memory_window(
        self, channel_id: str, conversation_id: int, max_turns: int
    ) -> list[dict[str, Any]]:
        """The newest ``max_turns`` usable turns as canonical 3-key memory
        dicts (oldest-first), feedback joined in — the gate-1 [R3] read that
        replaces the legacy ``get_conversation_window``."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT t.conversation_summary, t.conversation_traces, f.feedback_json
                    FROM turns t LEFT JOIN feedback f ON f.turn_key = t.turn_key
                    WHERE t.channel_id=? AND t.conversation_id=?
                      AND {self._USABLE_TURN_FILTER}
                    ORDER BY t.ordinal DESC, t.turn_key DESC LIMIT ?""",
                (channel_id, conversation_id, max_turns),
            ).fetchall()
        window = []
        for row in reversed(rows):
            feedback = None
            if row["feedback_json"]:
                try:
                    feedback = json.loads(row["feedback_json"])
                except ValueError:
                    feedback = row["feedback_json"]
            window.append(
                {
                    "conversation summary": row["conversation_summary"],
                    "conversation_traces": row["conversation_traces"],
                    "feedback": feedback,
                }
            )
        return window

    def conversation_summaries(
        self, channel_id: str, conversation_id: int
    ) -> list[dict[str, Any]]:
        """Each usable turn's summary, in order (labeling input)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT conversation_summary FROM turns "
                f"WHERE channel_id=? AND conversation_id=? AND {self._USABLE_TURN_FILTER} "
                f"ORDER BY ordinal, turn_key",
                (channel_id, conversation_id),
            ).fetchall()
            return [{"conversation summary": r["conversation_summary"]} for r in rows]

    def conversation_label_state(
        self, channel_id: str, conversation_id: int
    ) -> tuple[str, int]:
        """(stored topic or '', usable turn count) — the lazy-label trigger's
        one read (legacy ``get_conversation_label_state`` parity)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT topic FROM conversations WHERE channel_id=? AND conversation_id=?",
                (channel_id, conversation_id),
            ).fetchone()
        return (
            (row["topic"] or "") if row is not None else "",
            self.count_usable_turns(channel_id, conversation_id),
        )

    def newest_conversation_ids(self, channel_id: str, limit: int = 2) -> list[int]:
        """Newest conversation ids for a channel (restore + step-back)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id FROM conversations WHERE channel_id=? "
                "ORDER BY conversation_id DESC LIMIT ?",
                (channel_id, limit),
            ).fetchall()
            return [int(r[0]) for r in rows]

    def get_last_completed_turn_key(
        self, channel_id: str, conversation_id: int
    ) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT turn_key FROM turns WHERE channel_id=? AND conversation_id=? "
                f"AND {self._USABLE_TURN_FILTER} ORDER BY ordinal DESC, turn_key DESC LIMIT 1",
                (channel_id, conversation_id),
            ).fetchone()
            return row["turn_key"] if row is not None else None

    def list_conversation_summaries(
        self, channel_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """/conversations projection (ruling C7): only conversations with at
        least one usable turn (no reserved-but-empty phantoms), NULLs
        projected to '', timestamps as ms epoch, ordered by updated_at desc."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT c.conversation_id, c.topic, c.summary, c.started_at,
                           COALESCE(c.updated_at, c.last_turn_at, c.started_at) AS updated_at
                    FROM conversations c
                    WHERE c.channel_id=? AND EXISTS (
                        SELECT 1 FROM turns t
                        WHERE t.channel_id=c.channel_id
                          AND t.conversation_id=c.conversation_id
                          AND {self._USABLE_TURN_FILTER})
                    ORDER BY updated_at DESC LIMIT ?""",
                (channel_id, limit),
            ).fetchall()
        return [
            {
                "conversation_id": int(r["conversation_id"]),
                "topic": r["topic"] or "",
                "summary": r["summary"] or "",
                "created_at": _iso_to_ms(r["started_at"]),
                "updated_at": _iso_to_ms(r["updated_at"]),
            }
            for r in rows
        ]

    def dump_all_conversations(self, channel_id: str) -> list[dict[str, Any]]:
        """Admin-dump reconstruction of the hydrated legacy shape (ruling C7):
        one object per conversation with 3-key turns (+feedback) inlined."""
        dumped = []
        for conv in self.list_conversation_summaries(channel_id, limit=1_000_000):
            conv_id = conv["conversation_id"]
            dumped.append(
                {
                    "channel_id": channel_id,
                    "conversation_id": conv_id,
                    "topic": conv["topic"],
                    "summary": conv["summary"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"],
                    "turns": self.get_memory_window(channel_id, conv_id, 1_000_000),
                }
            )
        return dumped

    def upsert_feedback(self, turn_key: str, feedback_json: str) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO feedback (turn_key, feedback_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(turn_key) DO UPDATE SET
                     feedback_json=excluded.feedback_json, updated_at=excluded.updated_at""",
                (turn_key, feedback_json, _utcnow_iso()),
            )
            conn.commit()

    def record_train_run(
        self,
        run_id: str,
        workflow_fingerprint: Optional[str],
        started_at: Optional[str],
        completed_at: Optional[str],
        metrics: dict[str, Any],
    ) -> None:
        """Persist one training run's metrics at publication time (Phase 6)."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO train_runs
                   (run_id, workflow_fingerprint, started_at, completed_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     workflow_fingerprint=excluded.workflow_fingerprint,
                     started_at=excluded.started_at,
                     completed_at=excluded.completed_at,
                     metrics_json=excluded.metrics_json""",
                (
                    run_id,
                    workflow_fingerprint,
                    started_at,
                    completed_at,
                    json.dumps(_sanitize_json_value(metrics), ensure_ascii=False),
                ),
            )
            conn.commit()

    def set_diagnostic(self, conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (key, json.dumps(value, ensure_ascii=False), _utcnow_iso()),
        )

    # -- reads (GET /turns, run_chatbot) ---------------------------------

    def get_turn(self, turn_key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM turns WHERE turn_key=?", (turn_key,)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns", (trace_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_conversations(
        self, channel_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM conversations"
        params: list[Any] = []
        if channel_id is not None:
            query += " WHERE channel_id=?"
            params.append(channel_id)
        query += " ORDER BY COALESCE(last_turn_at, started_at) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def list_turns(
        self,
        channel_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        status: Optional[str] = None,
        success: Optional[bool] = None,
        command_name: Optional[str] = None,
        context: Optional[str] = None,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Turn rows, newest first, without record_json (fetch one turn for that).

        The experiment filters extend this route rather than getting a parallel
        implementation (`[XR9]`); they ride `idx_turns_experiment`.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if channel_id is not None:
            clauses.append("channel_id=?")
            params.append(channel_id)
        if conversation_id is not None:
            clauses.append("conversation_id=?")
            params.append(conversation_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if success is not None:
            clauses.append("success=?")
            params.append(1 if success else 0)
        if context is not None:
            # Substring match (the debug UI's semantics), parameterized and
            # LIKE-escaped; SQLite LIKE is ASCII-case-insensitive, matching
            # the previous client-side filter.
            escaped = (
                context.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            clauses.append("entry_context LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        if command_name is not None:
            clauses.append(
                "turn_key IN (SELECT trace_id FROM spans WHERE command_name=?)"
            )
            params.append(command_name)
        # [DR29]: a DB written before the experiment columns existed must
        # degrade, not raise. The base turn list is the whole point of the debug
        # UI, and a viewer opened on a post-mortem snapshot never migrates it
        # ([R12]), so an unguarded projection would 500 the main view forever
        # with "internal error: OperationalError" and no actionable reason.
        labelled = self.has_feature(FEATURE_EXPERIMENTS_V1)
        if labelled:
            if experiment_id is not None:
                clauses.append("experiment_id=?")
                params.append(experiment_id)
            if task_id is not None:
                clauses.append("task_id=?")
                params.append(task_id)
            if attempt is not None:
                clauses.append("attempt=?")
                params.append(int(attempt))
        elif experiment_id is not None or task_id is not None or attempt is not None:
            # An experiment filter against a DB that records no experiments
            # matches nothing. Returning [] is the honest answer; silently
            # ignoring the filter and returning every turn would be worse than
            # raising.
            return []
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT turn_key, channel_id, conversation_id, ordinal, user_message, "
            "entry_workflow_name, entry_context, status, success, failure_reason, "
            "answer, started_at, completed_at, suspended_ms"
            + (", experiment_id, task_id, attempt " if labelled else " ")
            + f"FROM turns{where} ORDER BY turn_key DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def list_channels(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT channel_id FROM turns ORDER BY channel_id"
            ).fetchall()
            return [r[0] for r in rows]

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def list_train_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM train_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def writer_health(self) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, updated_at FROM diagnostics WHERE key='writer_health'"
            ).fetchone()
            if row is None:
                return None
            health = json.loads(row["value"])
            health["updated_at"] = row["updated_at"]
            return health

    # -- the experiment container (`fix-bn1`, experiment_container_design.md) --
    #
    # CAPTURE POLICY, decided here rather than by omission (`[XR6]`): every text
    # column of `experiments`, `experiment_attempts` and
    # `experiment_evidence_runs` is SCRUB-ONLY -- `redactor.redact(value)` with
    # no `policy.apply` call, the `spans.channel_id` code shape.
    #
    # The precedent is `set_diagnostic` plus `_POLICY_EXEMPT_TURN_COLUMNS`, not
    # `spans.channel_id`'s erasure argument. These rows are not evidence ABOUT a
    # tenant; they are the record of whether the evidence may be used at all --
    # an `EvidenceRun`'s valid/problems, an attempt's outcome, a pre-registered
    # hypothesis. Withholding them reduces nothing a tenant would care about and
    # makes the bundle uninterpretable under exactly the profile an
    # evidence-grade run uses, since `opaque-payload` and `user-text` both map to
    # `omit` there. The claim that makes this safe is a DATAFLOW claim and is
    # tested: no code path exists by which workflow, model or user content
    # reaches these tables, except `task_id`, which the caller supplies from its
    # own task-set file. The residual risk -- an operator pasting a credential
    # into `notes`, an exception repr inside `record_json.problems` -- is exactly
    # what the scrub catches, which is why this is scrub-only and not untouched.
    #
    # No `POLICY_PATH_EXPERIMENT_*` constants are declared: a constant never
    # passed to `policy.apply` is inert, and the one genuinely scrub-only column
    # in this file, `spans.channel_id`, deliberately has none either.

    _EXPERIMENT_STATUSES = frozenset(
        {
            "running",
            "capture_complete",
            "awaiting_evaluation",
            "complete",
            "invalid",
        }
    )
    _EXECUTION_STATUSES = frozenset(
        {"completed", "failed", "cancelled", "abandoned"}
    )
    _ATTEMPT_OUTCOMES = frozenset({"pass", "fail", "error", "incomplete"})
    _INVALID_REASONS = frozenset(
        {
            "attempt_shortfall",
            "evidence_run_invalid",
            "turns_erased",
            "never_completed",
            "operator",
        }
    )

    def _scrub(self, value: Any) -> Any:
        """Credential-scrub one experiment-surface value. Falsy passes through."""
        return self._store_redactor().redact(value)

    def create_experiment(
        self,
        experiment_id: str,
        label: str,
        *,
        declared_tasks: int,
        declared_attempts: int,
        hypothesis: Optional[str] = None,
        arm: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        capture_profile: Optional[str] = None,
        capture_policy_version: Optional[str] = None,
    ) -> None:
        """Pre-register an experiment. Written BEFORE any task runs.

        `declared_tasks` and `declared_attempts` are required and positive: they
        are the denominator every score is computed against (`[XR14]`), and a
        score computed over surviving rows instead is the exact failure
        `EvidenceRun` exists to prevent one layer down.

        Re-creating an existing experiment is how a resume re-attaches. The
        `DO UPDATE` set deliberately excludes `hypothesis`, `status`,
        `invalid_reason` and `invalid_detail`: a resume must not be able to
        launder a rewritten prediction or an `invalid` verdict back to
        `running` (`[XR12]`).
        """
        if not experiment_id or not label:
            raise ValueError("experiment_id and label are required")
        declared_tasks = int(declared_tasks)
        declared_attempts = int(declared_attempts)
        if declared_tasks <= 0 or declared_attempts <= 0:
            raise ValueError(
                "declared_tasks and declared_attempts must both be positive: "
                "they are the denominator, and a score over an undeclared "
                "denominator is computed over whatever survived"
            )
        capture_profile = capture_profile or _env("FW_OBS_CAPTURE_PROFILE", "debug")
        capture_policy_version = capture_policy_version or "1"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # A re-create (the resume path) under a DIFFERENT capture regime is
            # refused rather than silently keeping the first one. The stored
            # profile is what `compare_experiments` gates on, so a run whose
            # second half was captured under another policy would compare as if
            # both halves matched -- and the column would say so.
            existing = conn.execute(
                """SELECT capture_profile, capture_policy_version, status
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if existing is not None and (
                existing["capture_profile"] != capture_profile
                or existing["capture_policy_version"] != capture_policy_version
            ):
                conn.rollback()
                raise CaptureRegimeChanged(
                    experiment_id,
                    f"{existing['capture_profile']}/"
                    f"{existing['capture_policy_version']}",
                    f"{capture_profile}/{capture_policy_version}",
                )
            conn.execute(
                """INSERT INTO experiments
                   (experiment_id, label, hypothesis, notes, arm,
                    baseline_experiment_id, status, invalid_reason,
                    invalid_detail, declared_tasks, declared_attempts,
                    workflow_name, capture_profile, capture_policy_version,
                    created_at, completed_at)
                   VALUES (?, ?, ?, NULL, ?, ?, 'running', NULL, NULL,
                           ?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(experiment_id) DO UPDATE SET
                     label=excluded.label,
                     arm=excluded.arm,
                     baseline_experiment_id=excluded.baseline_experiment_id,
                     -- The denominator is rewritable only while the experiment
                     -- is still running. Once it is complete or invalid, its
                     -- score has been computed against the declaration, and
                     -- changing the declaration afterwards silently restates
                     -- every number already reported from it -- the same
                     -- after-the-fact rewrite `hypothesis` is write-once to
                     -- prevent, one field over.
                     declared_tasks=CASE WHEN experiments.status='running'
                       THEN excluded.declared_tasks ELSE experiments.declared_tasks END,
                     declared_attempts=CASE WHEN experiments.status='running'
                       THEN excluded.declared_attempts ELSE experiments.declared_attempts END,
                     workflow_name=excluded.workflow_name""",
                (
                    experiment_id,
                    self._scrub(label),
                    self._scrub(hypothesis),
                    self._scrub(arm),
                    baseline_experiment_id,
                    declared_tasks,
                    declared_attempts,
                    self._scrub(workflow_name),
                    capture_profile,
                    capture_policy_version,
                    _utcnow_iso(),
                ),
            )
            conn.commit()

    def set_experiment_hypothesis(self, experiment_id: str, hypothesis: str) -> None:
        """Write-once (`[XR12]`), enforced here and nowhere else.

        The single enforcement point, the `apply_label_txn` shape. A UI-only
        guard would be a guard against honest mistakes, and the failure this
        must prevent -- rewriting a prediction after seeing the outcome -- is
        not an honest mistake. `non-NULL -> different` and `non-NULL -> NULL`
        are both refused; an identical rewrite is an idempotent success, the
        `upsert_turn_row` precedent.
        """
        scrubbed = self._scrub(hypothesis)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT hypothesis FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ExperimentNotFound(experiment_id)
            stored = row["hypothesis"]
            if stored is not None and stored != scrubbed:
                raise HypothesisIsWriteOnce(experiment_id)
            conn.execute(
                "UPDATE experiments SET hypothesis=? WHERE experiment_id=?",
                (scrubbed, experiment_id),
            )
            conn.commit()

    def update_experiment_notes(self, experiment_id: str, notes: Optional[str]) -> None:
        """Freely editable, by design and by contrast with `hypothesis`."""
        self._update_experiment(
            "UPDATE experiments SET notes=? WHERE experiment_id=?",
            (self._scrub(notes), experiment_id),
            experiment_id,
        )

    def _update_experiment(
        self, sql: str, params: tuple, experiment_id: str
    ) -> None:
        """Run an experiment UPDATE, raising when it matches no row.

        A 0-row update means the container is gone -- `clear_conversations` is
        an HTTP-triggered whole-DB erase and can land mid-run. Failing the
        harness loudly beats accumulating turns labelled against a container
        that no longer exists (`[XR15]`).
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(sql, params)
            if cursor.rowcount == 0:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            conn.commit()

    def record_evidence_segment(
        self,
        experiment_id: str,
        seq: int,
        evidence_run_id: str,
        record: dict[str, Any],
    ) -> None:
        """Record one `evidence_run()` segment (`[XR1]`).

        One row per segment rather than an appended JSON array, because
        appending to a column is a read-modify-write and `[XR20]` forbids that
        on any column a capture policy might act on. Here each segment is an
        independent INSERT and `valid` is a queryable column.

        `record` is the WHOLE `EvidenceRun.as_record()`, not its `observability`
        sub-dict: the sub-dict alone carries neither the run id, nor `valid`,
        nor `problems`, nor the archive digest.
        """
        payload = json.dumps(_sanitize_json_value(record), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone() is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            conn.execute(
                """INSERT INTO experiment_evidence_runs
                   (experiment_id, seq, evidence_run_id, valid, started_at,
                    completed_at, record_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(experiment_id, seq) DO UPDATE SET
                     evidence_run_id=excluded.evidence_run_id,
                     -- Monotone in invalidity, like `status <> 'invalid'` one
                     -- table over: once a segment has reported that evidence
                     -- was lost, re-writing that seq must not be able to erase
                     -- the report. Every other invalidity in this container is
                     -- terminal or write-once, and a rewritable one is a
                     -- verdict that can be revised after seeing the outcome.
                     valid=CASE WHEN experiment_evidence_runs.valid = 0
                                THEN 0 ELSE excluded.valid END,
                     started_at=excluded.started_at,
                     completed_at=excluded.completed_at,
                     record_json=excluded.record_json""",
                (
                    experiment_id,
                    int(seq),
                    evidence_run_id,
                    1 if record.get("valid") else 0,
                    record.get("started_at"),
                    record.get("completed_at"),
                    self._scrub(payload),
                ),
            )
            conn.commit()

    def start_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        channel_id: str,
        conversation_id: Optional[int] = None,
        source_attempt_key: Optional[dict[str, Any]] = None,
    ) -> None:
        """Open an attempt row before its first turn.

        The row's existence is not the execution completion marker --
        `execution_finished_at` is (`[XR13]`). An attempt that crashed halfway
        has rows and an open marker, which is what makes it visible to the
        resume selector and fatal to a `complete` verdict.
        """
        if not task_id:
            raise ValueError("task_id is required")
        attempt = int(attempt)
        if attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        source_attempt_json = (
            None
            if source_attempt_key is None
            else self._scrub(
                json.dumps(
                    {
                        "version": 1,
                        "value": _sanitize_json_value(source_attempt_key),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] != "running":
                # A closed experiment's verdicts are not rewritable. Without
                # this, re-invoking a driver script that pins its experiment_id
                # would silently overwrite all 45 stored outcomes and the
                # evidence segment of a `complete` run whose numbers had already
                # been quoted -- and `run()` has no guard of its own, unlike
                # `resume()`. Enforced here, where the `[XR12]` invariants live.
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, row["status"])
            existing = conn.execute(
                """SELECT channel_id, source_attempt_json
                     FROM experiment_attempts
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, self._scrub(task_id), attempt),
            ).fetchone()
            if existing is not None and (
                existing["channel_id"] != self._scrub(channel_id)
                or existing["source_attempt_json"] != source_attempt_json
            ):
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} was already "
                    "started with different identity metadata"
                )
            conn.execute(
                """INSERT INTO experiment_attempts
                   (experiment_id, task_id, attempt, channel_id, conversation_id,
                    outcome, outcome_source, reward, restarts, started_at,
                    execution_status, execution_finished_at, finished_at,
                    detail_json, source_attempt_json)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?,
                           NULL, NULL, NULL, NULL, ?)
                   ON CONFLICT(experiment_id, task_id, attempt) DO UPDATE SET
                     channel_id=excluded.channel_id,
                     conversation_id=COALESCE(excluded.conversation_id,
                                              experiment_attempts.conversation_id)""",
                (
                    experiment_id,
                    self._scrub(task_id),
                    attempt,
                    self._scrub(channel_id),
                    conversation_id,
                    _utcnow_iso(),
                    source_attempt_json,
                ),
            )
            conn.commit()

    def terminalize_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        execution_status: str,
        conversation_id: Optional[int] = None,
    ) -> None:
        """Record execution terminality independently from evaluation."""
        if execution_status not in self._EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status {execution_status!r} is not one of "
                f"{sorted(self._EXECUTION_STATUSES)}"
            )
        task_id = self._scrub(task_id)
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT a.execution_status, a.conversation_id,
                          e.status AS experiment_status
                     FROM experiment_attempts a
                     JOIN experiments e ON e.experiment_id=a.experiment_id
                    WHERE a.experiment_id=? AND a.task_id=? AND a.attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_status"] is not None:
                same_conversation = (
                    conversation_id is None
                    or row["conversation_id"] is None
                    or int(row["conversation_id"]) == int(conversation_id)
                )
                if row["execution_status"] == execution_status and same_conversation:
                    conn.rollback()
                    return
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} already has "
                    f"execution terminal value {row['execution_status']!r}"
                )
            if row["experiment_status"] != "running":
                conn.rollback()
                raise ExperimentIsClosed(
                    experiment_id, row["experiment_status"]
                )
            conn.execute(
                """UPDATE experiment_attempts
                      SET execution_status=?, execution_finished_at=?,
                          conversation_id=COALESCE(?, conversation_id)
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (
                    execution_status,
                    _utcnow_iso(),
                    conversation_id,
                    experiment_id,
                    task_id,
                    attempt,
                ),
            )
            conn.commit()

    def record_attempt_outcome(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a grade after execution, idempotently and write-once."""
        if outcome not in self._ATTEMPT_OUTCOMES:
            raise ValueError(
                f"outcome {outcome!r} is not one of {sorted(self._ATTEMPT_OUTCOMES)}"
            )
        if not outcome_source:
            raise ValueError(
                "outcome_source is required: an unattributed verdict cannot be "
                "told apart from a fallback"
            )
        task_id = self._scrub(task_id)
        source = self._scrub(outcome_source)
        reward_value = None if reward is None else float(reward)
        detail_json = (
            None
            if detail is None
            else self._scrub(
                json.dumps(
                    _sanitize_json_value(detail),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT a.execution_finished_at, a.outcome, a.outcome_source,
                          a.reward, a.detail_json,
                          e.status AS experiment_status
                     FROM experiment_attempts a
                     JOIN experiments e ON e.experiment_id=a.experiment_id
                    WHERE a.experiment_id=? AND a.task_id=? AND a.attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_finished_at"] is None:
                conn.rollback()
                raise ValueError(
                    f"attempt {experiment_id}/{task_id}/{attempt} execution "
                    "is still open"
                )
            incoming = (outcome, source, reward_value, detail_json)
            stored = (
                row["outcome"],
                row["outcome_source"],
                row["reward"],
                row["detail_json"],
            )
            if row["outcome"] is not None:
                if stored == incoming:
                    conn.rollback()
                    return
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} already has "
                    f"outcome {row['outcome']!r}; conflicting grades are refused"
                )
            if row["experiment_status"] not in {
                "running",
                "awaiting_evaluation",
            }:
                conn.rollback()
                raise ExperimentIsClosed(
                    experiment_id, row["experiment_status"]
                )
            conn.execute(
                """UPDATE experiment_attempts
                      SET outcome=?, outcome_source=?, reward=?, finished_at=?,
                          detail_json=?
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (
                    outcome,
                    source,
                    reward_value,
                    _utcnow_iso(),
                    detail_json,
                    experiment_id,
                    task_id,
                    attempt,
                ),
            )
            conn.commit()

    def finish_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
        conversation_id: Optional[int] = None,
    ) -> None:
        """Record an attempt's verdict (`[XR13]`).

        The verdict is WRITTEN, never derived from turn columns at read time.
        `outcome_source` names who decided -- a benchmark's reward function, a
        contract evaluator, an operator, or the literal `derived` for the
        turn-status fallback. Recording the source is what keeps a fallback from
        masquerading as a measurement.
        """
        self.terminalize_attempt(
            experiment_id,
            task_id,
            attempt,
            execution_status="completed",
            conversation_id=conversation_id,
        )
        self.record_attempt_outcome(
            experiment_id,
            task_id,
            attempt,
            outcome=outcome,
            outcome_source=outcome_source,
            reward=reward,
            detail=detail,
        )

    def restart_attempt(self, experiment_id: str, task_id: str, attempt: int) -> int:
        """Clear a crashed attempt so it can be re-run under the same labels.

        Deletes that attempt's conversations and turns in ONE transaction and
        bumps `restarts`. The deletion is deliberate (`[XR18]`): the abandoned
        partial trajectory is evidence of nothing, `idx_conv_experiment_attempt`
        is UNIQUE so a second conversation under the same three labels is
        refused outright, and leaving the rows would pin the attempt's derived
        diagnostic to 0 forever. `restarts` is what makes a task that keeps
        crashing visible rather than silently retried.

        Returns the number of turn rows deleted.
        """
        task_id = self._scrub(task_id)
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT execution_finished_at FROM experiment_attempts
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_finished_at"] is not None:
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} execution is "
                    "terminal and cannot be restarted"
                )
            turn_keys = [
                r[0]
                for r in conn.execute(
                    """SELECT turn_key FROM turns
                        WHERE experiment_id=? AND task_id=? AND attempt=?""",
                    (experiment_id, task_id, attempt),
                ).fetchall()
            ]
            if turn_keys:
                for chunk in _chunked(turn_keys):
                    marks = ", ".join("?" for _ in chunk)
                    conn.execute(
                        f"DELETE FROM feedback WHERE turn_key IN ({marks})", chunk
                    )
                    conn.execute(
                        f"DELETE FROM artifacts WHERE turn_key IN ({marks})", chunk
                    )
                    conn.execute(
                        f"DELETE FROM spans WHERE trace_id IN ({marks})", chunk
                    )
            deleted = conn.execute(
                """DELETE FROM turns
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            ).rowcount
            conn.execute(
                """DELETE FROM conversations
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            )
            conn.execute(
                """UPDATE experiment_attempts
                      SET restarts=restarts+1, outcome=NULL, outcome_source=NULL,
                          reward=NULL, execution_status=NULL,
                          execution_finished_at=NULL, finished_at=NULL,
                          detail_json=NULL,
                          conversation_id=NULL, started_at=?
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (_utcnow_iso(), experiment_id, task_id, attempt),
            )
            conn.commit()
        return deleted

    def complete_experiment(
        self,
        experiment_id: str,
        *,
        force_invalid: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> str:
        """Close an in-process experiment or defer it pending evaluation."""
        return self._complete_experiment(
            experiment_id,
            force_invalid=force_invalid,
            detail=detail,
            external_capture=False,
        )

    def complete_external_capture(self, experiment_id: str) -> str:
        """Close external execution without making its live store reportable."""
        return self._complete_experiment(
            experiment_id,
            force_invalid=None,
            detail=None,
            external_capture=True,
        )

    def _complete_experiment(
        self,
        experiment_id: str,
        *,
        force_invalid: Optional[str],
        detail: Optional[str],
        external_capture: bool,
    ) -> str:
        """Close an experiment. The STORE decides `complete` (`[XR14]`).

        The caller may request completion or force `invalid`; it may not assert
        completeness. `set_experiment_hypothesis` makes the strictly weaker
        pre-registration invariant store-enforced for exactly this reason, and a
        headline score rests on this one.

        `complete` requires all three: every declared (task, attempt) pair
        finished with an outcome, no outcome of `incomplete`, and no evidence
        segment marked invalid. Anything else is `invalid` with a closed reason
        code naming which check failed.

        `invalid` is TERMINAL: the UPDATE carries `AND status <> 'invalid'`, so
        neither a resume nor a later completion can clear a verdict recorded by
        `forget_channel` or by a failed evidence run.

        Returns the status actually stored.
        """
        if force_invalid is not None and force_invalid not in self._INVALID_REASONS:
            raise ValueError(
                f"invalid_reason {force_invalid!r} is not one of "
                f"{sorted(self._INVALID_REASONS)}"
            )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT status, declared_tasks, declared_attempts
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] == "invalid":
                conn.rollback()
                return "invalid"
            if (
                row["status"] in {"complete", "capture_complete"}
                and force_invalid is None
            ):
                conn.rollback()
                return str(row["status"])

            reason: Optional[str] = force_invalid
            status: Optional[str] = None
            if reason is None:
                expected = int(row["declared_tasks"]) * int(row["declared_attempts"])
                # The SHAPE must match the declaration, not merely the count.
                # Counting finished rows against `expected` alone lets a row
                # outside the declared set pay for a declared pair that never
                # ran: a resume whose task list gained two tasks and lost one
                # reaches `finished == expected` with a declared task missing,
                # and `experiment_scores` then divides more scored attempts than
                # the denominator and reports pass@1 = 1.33 as a headline number.
                # So: every row finished, exactly as many rows as declared, and
                # exactly as many distinct tasks as declared.
                counts = conn.execute(
                    """SELECT
                         COUNT(*) AS rows_total,
                         COUNT(DISTINCT task_id) AS tasks_total,
                         SUM(CASE WHEN execution_finished_at IS NOT NULL
                                  THEN 1 ELSE 0 END) AS executed,
                         SUM(CASE WHEN finished_at IS NOT NULL AND outcome IS NOT NULL
                                  THEN 1 ELSE 0 END) AS evaluated,
                         SUM(CASE WHEN outcome='incomplete' THEN 1 ELSE 0 END)
                              AS incomplete
                       FROM experiment_attempts WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchone()
                rows_total = int(counts["rows_total"] or 0)
                tasks_total = int(counts["tasks_total"] or 0)
                executed = int(counts["executed"] or 0)
                evaluated = int(counts["evaluated"] or 0)
                incomplete = int(counts["incomplete"] or 0)
                declared_tasks = int(row["declared_tasks"])
                bad_segments = conn.execute(
                    """SELECT COUNT(*) FROM experiment_evidence_runs
                        WHERE experiment_id=? AND valid=0""",
                    (experiment_id,),
                ).fetchone()[0]
                if (
                    executed != expected
                    or rows_total != expected
                    or tasks_total != declared_tasks
                ):
                    reason = "attempt_shortfall"
                    detail = (
                        f"{executed} finished and {rows_total} recorded of "
                        f"{expected} declared attempts across {tasks_total} of "
                        f"{declared_tasks} declared tasks; {incomplete} incomplete"
                    )
                elif incomplete:
                    reason = "attempt_shortfall"
                    detail = (
                        f"{evaluated} evaluated of {expected} declared attempts; "
                        f"{incomplete} incomplete"
                    )
                elif bad_segments:
                    reason = "evidence_run_invalid"
                    detail = f"{bad_segments} evidence segment(s) reported invalid"
                elif evaluated != expected:
                    status = "awaiting_evaluation"
                elif external_capture:
                    status = "capture_complete"
                else:
                    status = "complete"

            if reason is not None:
                status = "invalid"
            elif status is None:
                status = "complete"
            detail_text = self._scrub(detail) if reason is not None else None
            cursor = conn.execute(
                """UPDATE experiments
                      SET status=?, completed_at=?, invalid_reason=?,
                          invalid_detail=CASE
                              WHEN ? IS NULL THEN invalid_detail
                              WHEN invalid_detail IS NULL THEN ?
                              ELSE invalid_detail || char(10) || ? END
                    WHERE experiment_id=? AND status <> 'invalid'""",
                (
                    status,
                    None if status == "awaiting_evaluation" else _utcnow_iso(),
                    reason,
                    # Bound to None on the `complete` branch: `invalid_detail`
                    # is the explanation of an invalid verdict, and a detail
                    # string sitting on a complete experiment reads as one.
                    detail_text,
                    detail_text,
                    detail_text,
                    experiment_id,
                ),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return "invalid"
            conn.commit()
        return status

    def invalidate_experiments_in_txn(
        self,
        conn: sqlite3.Connection,
        experiment_ids: Iterable[str],
        reason: str,
        detail: Optional[str] = None,
    ) -> int:
        """Mark experiments invalid inside the caller's transaction.

        Used by `forget_channel`, which must not DELETE an experiment (44 of its
        45 attempts may live in other channels) but must never leave one
        scoreable after its turns are gone. `invalid_detail` is append-only so a
        second cause does not erase the first (`[XR15]`).
        """
        ids = [e for e in dict.fromkeys(experiment_ids) if e]
        if not ids:
            return 0
        scrubbed = self._scrub(detail)
        touched = 0
        for chunk in _chunked(ids):
            marks = ", ".join("?" for _ in chunk)
            touched += conn.execute(
                f"""UPDATE experiments
                       SET status='invalid', invalid_reason=?,
                           completed_at=COALESCE(completed_at, ?),
                           invalid_detail=CASE
                               WHEN ? IS NULL THEN invalid_detail
                               WHEN invalid_detail IS NULL THEN ?
                               ELSE invalid_detail || char(10) || ? END
                     WHERE experiment_id IN ({marks})""",
                [reason, _utcnow_iso(), scrubbed, scrubbed, scrubbed, *chunk],
            ).rowcount
        return touched

    # -- experiment reads ------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """One experiment plus its evidence segments, or None."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                return None
            experiment = dict(row)
            segments = []
            for seg in conn.execute(
                """SELECT * FROM experiment_evidence_runs
                    WHERE experiment_id=? ORDER BY seq""",
                (experiment_id,),
            ).fetchall():
                segment = dict(seg)
                try:
                    segment["record"] = json.loads(segment.pop("record_json"))
                except (ValueError, KeyError):
                    segment["record"] = None
                segments.append(segment)
        experiment["evidence_runs"] = segments
        return experiment

    def list_experiments(
        self,
        status: Optional[str] = None,
        arm: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Experiments newest first, each with its observed attempt counts."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("e.status=?")
            params.append(status)
        if arm is not None:
            clauses.append("e.arm=?")
            params.append(arm)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT e.experiment_id, e.label, e.status, e.arm, "
            "e.baseline_experiment_id, e.declared_tasks, e.declared_attempts, "
            "e.invalid_reason, e.workflow_name, e.capture_profile, "
            "e.created_at, e.completed_at, "
            "(SELECT COUNT(*) FROM experiment_attempts a "
            "  WHERE a.experiment_id=e.experiment_id) AS attempts_started, "
            "(SELECT COUNT(*) FROM experiment_attempts a "
            "  WHERE a.experiment_id=e.experiment_id AND a.finished_at IS NOT NULL "
            "    AND a.outcome IS NOT NULL) AS attempts_finished "
            f"FROM experiments e{where} ORDER BY e.created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def experiment_attempt_rows(
        self, experiment_id: str, task_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Attempt rows, ordered by task then attempt."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return []
        clauses = ["experiment_id=?"]
        params: list[Any] = [experiment_id]
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(task_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM experiment_attempts
                     WHERE {' AND '.join(clauses)}
                     ORDER BY task_id, attempt""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def experiment_tasks(self, experiment_id: str) -> list[dict[str, Any]]:
        """One row per task: its attempts' outcomes, and whether all passed."""
        attempts = self.experiment_attempt_rows(experiment_id)
        by_task: dict[str, dict[str, Any]] = {}
        for row in attempts:
            task = by_task.setdefault(
                row["task_id"],
                {"task_id": row["task_id"], "attempts": [], "outcomes": []},
            )
            task["attempts"].append(row)
            task["outcomes"].append(row["outcome"])
        for task in by_task.values():
            outcomes = task["outcomes"]
            task["passed_all"] = bool(outcomes) and all(o == "pass" for o in outcomes)
            task["passed_any"] = any(o == "pass" for o in outcomes)
        return [by_task[k] for k in sorted(by_task)]

    def experiment_scores(self, experiment_id: str) -> dict[str, Any]:
        """pass@1 / pass^k over an experiment (`[XR13]`, `[XR14]`).

        Both are computed against the DECLARED denominator, never against
        surviving rows: a run that lost 12 of 45 attempts must not score 33/33
        and look perfect.

        A headline number is returned ONLY for `status='complete'`. A running or
        invalid experiment gets its per-task detail and its status in place of a
        score -- a provisional number in a UI becomes a quoted number in a
        document.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ExperimentNotFound(experiment_id)
        tasks = self.experiment_tasks(experiment_id)
        declared_tasks = int(experiment["declared_tasks"])
        declared_attempts = int(experiment["declared_attempts"])
        expected = declared_tasks * declared_attempts
        scored = [
            row
            for task in tasks
            for row in task["attempts"]
            if row["finished_at"] is not None and row["outcome"] is not None
        ]
        result: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": experiment["status"],
            "invalid_reason": experiment["invalid_reason"],
            "declared_tasks": declared_tasks,
            "declared_attempts": declared_attempts,
            "expected_attempts": expected,
            "scored_attempts": len(scored),
            "tasks": tasks,
            "outcome_sources": sorted(
                {row["outcome_source"] for row in scored if row["outcome_source"]}
            ),
            "pass_at_1": None,
            "pass_at_k": None,
            "reportable": False,
        }
        if experiment["status"] != "complete":
            result["reason_not_reportable"] = (
                f"experiment status is {experiment['status']!r}; a score is only "
                "reportable for a complete experiment"
            )
            return result
        if len(scored) != expected or len(tasks) != declared_tasks:
            # Unreachable while `complete_experiment` is the only way to reach
            # `complete`, and kept anyway: this function divides by the DECLARED
            # denominator, so a set of rows that does not match the declaration
            # produces a ratio above 1.0 rather than an error. A score that can
            # exceed 1.0 is worse than no score.
            result["reportable"] = False
            result["reason_not_reportable"] = (
                f"{len(scored)} scored attempts across {len(tasks)} tasks do not "
                f"match the declared {expected} across {declared_tasks}; the "
                "experiment is marked complete but its rows do not support a score"
            )
            return result
        passed = sum(1 for row in scored if row["outcome"] == "pass")
        # pass^k is over DECLARED tasks: a task with no attempt row at all is a
        # task that did not pass every attempt, and dividing by the tasks that
        # happen to be present is the denominator error this guards against.
        all_passed = sum(1 for task in tasks if task["passed_all"])
        result["pass_at_1"] = passed / expected
        result["pass_at_k"] = all_passed / declared_tasks
        result["reportable"] = True
        return result

    def compare_experiments(
        self, experiment_id: str, baseline_experiment_id: str
    ) -> dict[str, Any]:
        """Treatment vs baseline, per task (`[XR19]`).

        Reports flip counts and sample size; it does NOT claim significance. A
        query layer that emits a p-value is a query layer that will be quoted as
        if it had run the protocol.

        Refuses unless both are complete, both declare the same shape, their
        task-id SETS are equal, and they were captured under the same profile.
        Cardinality is not comparability: two 15x3 runs over disjoint task sets
        would otherwise report "0 regressions" while sharing no task.
        """
        treatment = self.get_experiment(experiment_id)
        baseline = self.get_experiment(baseline_experiment_id)
        if treatment is None:
            raise ExperimentNotFound(experiment_id)
        if baseline is None:
            raise ExperimentNotFound(baseline_experiment_id)
        problems: list[str] = []
        for side, exp in (("treatment", treatment), ("baseline", baseline)):
            if exp["status"] != "complete":
                problems.append(
                    f"{side} {exp['experiment_id']} is {exp['status']!r}, not complete"
                )
        if (treatment["declared_tasks"], treatment["declared_attempts"]) != (
            baseline["declared_tasks"],
            baseline["declared_attempts"],
        ):
            problems.append(
                f"declared shapes differ: treatment "
                f"{treatment['declared_tasks']}x{treatment['declared_attempts']} "
                f"vs baseline {baseline['declared_tasks']}x"
                f"{baseline['declared_attempts']}"
            )
        if treatment["capture_profile"] != baseline["capture_profile"] or (
            treatment["capture_policy_version"] != baseline["capture_policy_version"]
        ):
            problems.append(
                f"capture regimes differ: treatment "
                f"{treatment['capture_profile']}/"
                f"{treatment['capture_policy_version']} vs baseline "
                f"{baseline['capture_profile']}/"
                f"{baseline['capture_policy_version']}; the two arms are not "
                "measuring the same columns"
            )
        t_tasks = {t["task_id"]: t for t in self.experiment_tasks(experiment_id)}
        b_tasks = {
            t["task_id"]: t for t in self.experiment_tasks(baseline_experiment_id)
        }
        only_treatment = sorted(set(t_tasks) - set(b_tasks))
        only_baseline = sorted(set(b_tasks) - set(t_tasks))
        if only_treatment or only_baseline:
            problems.append(
                f"task sets differ: {len(only_treatment)} only in treatment, "
                f"{len(only_baseline)} only in baseline"
            )
        if problems:
            return {
                "comparable": False,
                "problems": problems,
                "only_in_treatment": only_treatment,
                "only_in_baseline": only_baseline,
            }
        improved, regressed, unchanged = [], [], []
        expected_flips = 0.0
        k = int(treatment["declared_attempts"])
        for task_id in sorted(t_tasks):
            t_pass = t_tasks[task_id]["passed_all"]
            b_pass = b_tasks[task_id]["passed_all"]
            if t_pass and not b_pass:
                improved.append(task_id)
            elif b_pass and not t_pass:
                regressed.append(task_id)
            else:
                unchanged.append(task_id)
            expected_flips += self._expected_flip_probability(
                t_tasks[task_id], b_tasks[task_id], k
            )
        return {
            "comparable": True,
            "problems": [],
            "treatment": self.experiment_scores(experiment_id),
            "baseline": self.experiment_scores(baseline_experiment_id),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "tasks_compared": len(t_tasks),
            "attempts_per_task": k,
            "expected_flips_if_nothing_changed": round(expected_flips, 3),
            "observed_flips": len(improved) + len(regressed),
        }

    @staticmethod
    def _expected_flip_probability(
        treatment_task: dict[str, Any], baseline_task: dict[str, Any], k: int
    ) -> float:
        """How often this task's pass^k verdict would flip if NOTHING changed.

        The question `fix-bn1.7` asks -- "how many flips are attributable to
        variance rather than the change" -- has an answer that does not require
        claiming significance, and this is it. Pool both arms' attempts for one
        task to estimate a single per-attempt pass rate p, then a flip in either
        direction has probability 2 * p^k * (1 - p^k) under the hypothesis that
        the arms are identical. Summed over tasks, that is the number of flips a
        pair of arms that differ in nothing would be expected to produce.

        **What this is not.** It is not a p-value and it is not a test. It is an
        expectation under one crude null, offered so that "3 tasks flipped"
        stops reading as "3 tasks improved" when the expected number is 2.6. The
        statistical protocol lives outside this file, deliberately: a query layer
        that emits a significance verdict is a query layer that will be quoted as
        if it had run one.

        A task with no attempts contributes 0: nothing that was never run can
        flip.
        """
        outcomes = [
            o
            for o in (treatment_task["outcomes"] + baseline_task["outcomes"])
            if o is not None
        ]
        if not outcomes or k <= 0:
            return 0.0
        p = sum(1 for o in outcomes if o == "pass") / len(outcomes)
        p_all = p ** k
        return 2.0 * p_all * (1.0 - p_all)

    def experiment_labels_for_turn(
        self, turn_key: str
    ) -> Optional[dict[str, Any]]:
        """Return experiment labels attached to a turn, if any."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return None
        with self._connect() as conn:
            row = conn.execute(
                """SELECT t.experiment_id, t.task_id, t.attempt, e.label, e.status
                     FROM turns t LEFT JOIN experiments e
                       ON e.experiment_id = t.experiment_id
                    WHERE t.turn_key=? AND t.experiment_id IS NOT NULL""",
                (turn_key,),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_distillation_runs(
        self, *, experiment_id: Optional[str] = None, **_: Any
    ) -> list[dict[str, Any]]:
        """Compatibility read seam; distillation is not shipped by this port."""
        return []

    # -- maintenance [R12] and erasure [R21] -----------------------------

    def db_size_bytes(self) -> int:
        """DB file size including the -wal sidecar [R12]."""
        total = 0
        for path in (self.db_path, f"{self.db_path}-wal"):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    def archive_to(self, destination: str) -> dict[str, Any]:
        """Create a consistent, read-only SQLite archive with a digest."""
        target = Path(destination)
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing evidence archive: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("VACUUM INTO ?", (str(target),))
        digest = hashlib.sha256()
        with open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size_bytes = target.stat().st_size
        target.chmod(0o444)
        return {
            "path": str(target),
            "size_bytes": size_bytes,
            "sha256": digest.hexdigest(),
            "schema_version": SCHEMA_VERSION,
            "read_only": True,
        }

    def prune(
        self,
        retention_days: Optional[int] = None,
        max_bytes: Optional[int] = None,
        include_conversationless_turns: bool = False,
    ) -> dict[str, int]:
        """Bounded prune of spans/artifacts beyond the retention horizon, plus
        oldest-first eviction while over the size cap. Conversations and turn
        records are exempt (config §5 / [R16]). Runs incremental_vacuum.

        ``include_conversationless_turns`` (operator opt-in, ruling C10) also
        deletes conversation-less turn records (e.g. per-invocation CLI
        channels) older than the horizon, with their feedback — otherwise no
        retention knob ever reaches them.
        """
        if pruning_suppressed():
            return {"suppressed": 1}
        if retention_days is None:
            retention_days = _env_int("FW_OBS_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
        if max_bytes is None:
            max_bytes = _env_int("FW_OBS_DB_MAX_BYTES", _DEFAULT_DB_MAX_BYTES)

        horizon_ns = int(
            (time.time() - retention_days * 86_400) * 1_000_000_000
        )
        horizon_key = datetime.fromtimestamp(
            max(0.0, time.time() - retention_days * 86_400), tz=timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        deleted = {"spans": 0, "artifacts": 0}

        with self._connect() as conn:
            for _ in range(_PRUNE_MAX_BATCHES):
                conn.execute("BEGIN IMMEDIATE")
                spans_cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans WHERE start_ns < ? LIMIT ?)",
                    (horizon_ns, _PRUNE_BATCH_ROWS),
                )
                deleted["spans"] += spans_cur.rowcount
                artifacts_cur = conn.execute(
                    "DELETE FROM artifacts WHERE artifact_id IN "
                    "(SELECT artifact_id FROM artifacts WHERE turn_key < ? LIMIT ?)",
                    (horizon_key, _PRUNE_BATCH_ROWS),
                )
                deleted["artifacts"] += artifacts_cur.rowcount
                conn.commit()
                if (
                    spans_cur.rowcount < _PRUNE_BATCH_ROWS
                    and artifacts_cur.rowcount < _PRUNE_BATCH_ROWS
                ):
                    break

            if include_conversationless_turns:
                deleted["conversationless_turns"] = 0
                for _ in range(_PRUNE_MAX_BATCHES):
                    conn.execute("BEGIN IMMEDIATE")
                    keys = [
                        r[0]
                        for r in conn.execute(
                            "SELECT turn_key FROM turns WHERE conversation_id IS NULL "
                            "AND turn_key < ? LIMIT ?",
                            (horizon_key, _PRUNE_BATCH_ROWS),
                        ).fetchall()
                    ]
                    for key in keys:
                        conn.execute("DELETE FROM feedback WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM spans WHERE trace_id=?", (key,))
                        conn.execute("DELETE FROM artifacts WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM turns WHERE turn_key=?", (key,))
                    conn.commit()
                    deleted["conversationless_turns"] += len(keys)
                    if len(keys) < _PRUNE_BATCH_ROWS:
                        break

            # Size-cap eviction, oldest spans first (turn keys sort by time).
            for _ in range(_PRUNE_MAX_BATCHES):
                if self.db_size_bytes() <= max_bytes:
                    break
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans ORDER BY start_ns LIMIT ?)",
                    (_PRUNE_BATCH_ROWS,),
                )
                conn.commit()
                if cur.rowcount == 0:
                    break
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted

    def forget_channel(self, channel_id: str) -> dict[str, int]:
        """First-class erasure [R21]: delete a channel across all tables, then
        checkpoint-truncate the WAL and reclaim pages."""
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            touched_experiments = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT experiment_id FROM experiment_attempts "
                    "WHERE channel_id=?",
                    (channel_id,),
                ).fetchall()
            ]
            deleted["feedback"] = conn.execute(
                "DELETE FROM feedback WHERE turn_key IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id,),
            ).rowcount
            deleted["spans"] = conn.execute(
                "DELETE FROM spans WHERE channel_id=? OR trace_id IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id, channel_id),
            ).rowcount
            deleted["artifacts"] = conn.execute(
                "DELETE FROM artifacts WHERE channel_id=? OR turn_key IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id, channel_id),
            ).rowcount
            deleted["turns"] = conn.execute(
                "DELETE FROM turns WHERE channel_id=?", (channel_id,)
            ).rowcount
            deleted["conversations"] = conn.execute(
                "DELETE FROM conversations WHERE channel_id=?", (channel_id,)
            ).rowcount
            # The experiment container `[XR15]`. An experiment is NOT the
            # channel's to delete -- 44 of its 45 attempts may live in other
            # channels -- but it must never stay scoreable once its turns are
            # gone, because after this its denominator is unreconstructable.
            # So: delete this channel's attempt rows, and mark every experiment
            # they belonged to terminally invalid. Ids collected BEFORE the
            # deletes, per [DR44].
            deleted["experiment_attempts"] = conn.execute(
                "DELETE FROM experiment_attempts WHERE channel_id=?", (channel_id,)
            ).rowcount
            invalidated = self.invalidate_experiments_in_txn(
                conn,
                touched_experiments,
                "turns_erased",
                f"turns erased by forget_channel for channel {channel_id!r}",
            )
            if invalidated:
                deleted["experiments_invalidated"] = invalidated
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted

    def clear_conversations(self) -> dict[str, int]:
        """Delete every recorded conversation and its turn-level observability.

        Training runs, writer diagnostics, and monotonic conversation counters
        survive. Keeping counters prevents a clear operation from reusing a
        conversation identity that may still be referenced outside this DB.
        """
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in (
                "experiment_evidence_runs",
                "experiment_attempts",
                "experiments",
            ):
                deleted[table] = conn.execute(f"DELETE FROM {table}").rowcount
            for table in ("feedback", "spans", "artifacts", "turns", "conversations"):
                deleted[table] = conn.execute(f"DELETE FROM {table}").rowcount
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted


class ReadOnlyObservabilityStore(ObservabilityStore):
    """Read-only view of an existing observability DB (the chatbot's debug
    layer). Never creates, migrates, or writes the file — the viewer must be
    able to open a post-mortem snapshot it does not own, and inspecting a DB
    must not mutate it. Construction raises when the file is absent/unopenable
    (``sqlite3.OperationalError``) or written by a newer build
    (``IncompatibleObservabilityDB`` [R11]); callers degrade gracefully.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        conn = self._connect()
        try:
            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build reads up to "
                    f"v{SCHEMA_VERSION}. Refusing to open a newer DB [R11]."
                )
        finally:
            conn.close()
        self._features = self._load_features()

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn


# ----------------------------------------------------------------------
# The sink: two queues + one daemon writer thread [R7][R8][R13]
# ----------------------------------------------------------------------


class SQLiteTraceSink:
    """TraceSink writing to an ObservabilityStore via a background thread.

    Never raises to callers. Turn records/labels ride a small dedicated queue
    (bounded-timeout put, then drop-with-log — the only case a turn record may
    drop in v1); spans ride a droppable queue bounded by FW_OBS_QUEUE_MAX
    (drop-and-count) [R13].
    """

    def __init__(self, db_path: str) -> None:
        self.store = ObservabilityStore(db_path)
        try:
            self._db_ino = os.stat(db_path).st_ino
        except OSError:
            self._db_ino = None
        self._redactor = Redactor()
        self._record_queue: queue.Queue = queue.Queue(maxsize=_RECORD_QUEUE_MAX)
        self._span_queue: queue.Queue = queue.Queue(
            maxsize=_env_int("FW_OBS_QUEUE_MAX", _DEFAULT_QUEUE_MAX)
        )
        self._closed = False
        self._stop = threading.Event()
        self._health = {
            "spans_dropped": 0,
            "records_dropped": 0,
            "write_errors": 0,
            "busy_retries": 0,
            "refused_terminal_writes": 0,
            "sync_writes": 0,
            "sync_fallbacks": 0,
            "sync_write_ms_max": 0,
            "pending_retry_depth": 0,
            "sync_breaker_open": False,
            "last_error": None,
        }
        self._health_dirty = False
        self._health_lock = threading.Lock()
        # Sync-first write state (§2.4). The breaker deadline is a monotonic
        # timestamp; the ring holds terminal rows the sync path could not land.
        self._sync_lock = threading.Lock()
        self._sync_breaker_until = 0.0
        self._pending: "dict[str, tuple]" = {}
        self._writer = threading.Thread(
            target=self._writer_loop, name="fw-obs-writer", daemon=True
        )
        self._writer.start()
        # Opportunistic bounded prune at sink startup [R12].
        try:
            self.store.prune()
        except Exception as exc:
            logger.warning(f"Observability startup prune failed: {exc!r}")

    # -- TraceSink protocol ---------------------------------------------

    def emit_span(self, span: tracing.Span) -> None:
        if self._closed:
            return
        try:
            snapshot = tracing.Span(
                span_id=span.span_id,
                trace_id=span.trace_id,
                name=span.name,
                kind=span.kind,
                parent_span_id=span.parent_span_id,
                channel_id=span.channel_id,
                command_name=span.command_name,
                context=span.context,
                start_ns=span.start_ns,
                end_ns=span.end_ns,
                status=span.status,
                attributes=dict(span.attributes),
            )
            self._span_queue.put_nowait(("span", snapshot))
        except queue.Full:
            self._count("spans_dropped")
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    def emit_turn_record(self, record: Any) -> bool:
        """Write the turn record, synchronously by default. Returns "stored".

        Sync-first (§2.4 as amended by rulings I6/C8): EVERY turn-record
        emission — awaiting_user and terminal alike — takes the same path, so
        one logical turn can never be split across the sync and queued paths
        and arrive out of order. The queue is only the degraded fallback.

        The return value is the ack ruling I1 requires. The observability DB is
        the conversation record now, so a caller that drops turns out of its
        in-memory history has to know whether they were actually persisted:
        False means "queued, not yet durable" and the caller must defer its
        trim. Never raises; a caller that cannot use the ack can ignore it.
        """
        if self._closed:
            return False
        try:
            turn_row, artifact_rows = serialize_turn_result(record)
        except Exception as exc:
            self._count("write_errors", error=f"serialize: {exc!r}")
            return False

        if self._sync_available() and self._sync_write(turn_row, artifact_rows):
            self._forget_pending(turn_row["turn_key"])
            return True

        self._count("sync_fallbacks")
        self._queue_turn_row(turn_row, artifact_rows)
        return False

    def _sync_available(self) -> bool:
        with self._sync_lock:
            return time.monotonic() >= self._sync_breaker_until

    def _sync_write(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> bool:
        """One short BEGIN IMMEDIATE on the caller thread. Never raises.

        Its own connection with a SHORT busy timeout (ruling C9): the default
        30 s would put a wedged DB in front of a user's turn for half a minute.
        On failure the breaker opens so a broken disk degrades to Phase-A
        queued behaviour instead of taxing every subsequent turn.
        """
        started = time.monotonic()
        conn = None
        try:
            conn = self.store._connect(
                timeout=float(
                    _env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", _DEFAULT_SYNC_WRITE_TIMEOUT_S)
                )
            )
            conn.execute("BEGIN IMMEDIATE")
            accepted = self.store.upsert_turn_row(
                conn, turn_row, artifact_rows, self._redactor
            )
            conn.commit()
        except Exception as exc:
            if conn is not None:
                self._rollback(conn)
            self._trip_sync_breaker(exc)
            return False
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        if not accepted:
            self._count("refused_terminal_writes")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        with self._health_lock:
            self._health["sync_writes"] = int(self._health["sync_writes"]) + 1
            if elapsed_ms > int(self._health["sync_write_ms_max"] or 0):
                self._health["sync_write_ms_max"] = elapsed_ms
            self._health_dirty = True
        # A refusal means a terminal row is already there: the turn IS durable,
        # which is what the ack promises. Only a failed write is not.
        return True

    def _trip_sync_breaker(self, exc: Exception) -> None:
        cooldown = _env_int(
            "FW_OBS_SYNC_BREAKER_COOLDOWN_S", _DEFAULT_SYNC_BREAKER_COOLDOWN_S
        )
        with self._sync_lock:
            self._sync_breaker_until = time.monotonic() + cooldown
        with self._health_lock:
            self._health["sync_breaker_open"] = True
            self._health_dirty = True
        self._count("write_errors", error=f"sync write: {exc!r}")

    def _queue_turn_row(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> None:
        """Degraded path: reserve the ordinal, enqueue, and remember terminals.

        The ordinal is reserved synchronously in its own tiny transaction
        (ruling I6) so a record that rides the queue still sorts where it
        happened — otherwise a turn written while the DB was briefly wedged
        would land after turns that came later.
        """
        if (
            turn_row.get("conversation_id") is not None
            and turn_row.get("ordinal") is None
        ):
            turn_row["ordinal"] = self.store.reserve_turn_ordinal(
                turn_row["channel_id"],
                turn_row["conversation_id"],
                experiment_id=turn_row.get("experiment_id"),
                task_id=turn_row.get("task_id"),
                attempt=turn_row.get("attempt"),
            )
        if turn_row["status"] in TERMINAL_TURN_STATUSES:
            self._remember_pending(turn_row, artifact_rows)
        try:
            self._record_queue.put(
                ("turn", turn_row, artifact_rows, 0), timeout=_RECORD_PUT_TIMEOUT_S
            )
        except queue.Full:
            self._count("records_dropped")
            logger.warning(
                f"Observability turn-record queue full; DROPPED record for "
                f"{turn_row.get('turn_key')} [R13]"
            )
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    def _remember_pending(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> None:
        """Hold a terminal row for retry until a write of it is confirmed."""
        with self._sync_lock:
            self._pending[turn_row["turn_key"]] = (turn_row, artifact_rows)
            while len(self._pending) > _PENDING_RETRY_MAX:
                # Oldest first: dict preserves insertion order, and the oldest
                # entry is the one whose turn has been unrecorded longest.
                oldest = next(iter(self._pending))
                del self._pending[oldest]
                self._count("records_dropped")
                logger.warning(
                    f"Observability pending-retry ring full; giving up on "
                    f"turn record {oldest} [R13]"
                )
            depth = len(self._pending)
        with self._health_lock:
            self._health["pending_retry_depth"] = depth
            self._health_dirty = True

    def _forget_pending(self, turn_key: str) -> None:
        with self._sync_lock:
            if self._pending.pop(turn_key, None) is None:
                return
            depth = len(self._pending)
        with self._health_lock:
            self._health["pending_retry_depth"] = depth
            self._health_dirty = True

    def pending_retry_depth(self) -> int:
        """Terminal records still awaiting a confirmed write (tests, health)."""
        with self._sync_lock:
            return len(self._pending)

    def health_snapshot(self) -> dict[str, Any]:
        with self._health_lock:
            return dict(self._health)

    def persist_health(self) -> None:
        conn = self.store._connect()
        try:
            self._maybe_write_health(conn, force=True)
        finally:
            conn.close()

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None:
        if self._closed:
            return
        try:
            self._record_queue.put(
                ("label", channel_id, conversation_id, topic, summary, 0),
                timeout=_RECORD_PUT_TIMEOUT_S,
            )
        except queue.Full:
            self._count("records_dropped")
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    # -- lifecycle -------------------------------------------------------

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything enqueued so far is written (tests, close)."""
        done = threading.Event()
        try:
            self._record_queue.put(("flush", done), timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        """Stop signal + bounded join + final drain and commit [R7]. Idempotent.

        Emissions racing with close are dropped (the sink is closed); the
        writer drains everything already enqueued before exiting, so the last
        turn of a session is never lost.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._writer.join(timeout)
        if self._writer.is_alive():
            logger.warning("Observability writer did not stop within timeout")

    # -- internals -------------------------------------------------------

    def _count(self, key: str, error: Optional[str] = None) -> None:
        with self._health_lock:
            self._health[key] = int(self._health.get(key) or 0) + 1
            if error is not None:
                self._health["last_error"] = error[:500]
            self._health_dirty = True

    def _writer_loop(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self.store._connect()
            while not self._stop.is_set():
                item = self._next_item()
                if item is None:
                    self._heartbeat(conn)
                    continue
                self._apply_batch(conn, [item] + self._drain_pending())
            # Final drain: everything enqueued before close() is written.
            while items := self._drain_pending():
                self._apply_batch(conn, items)
            # Then the retry ring, which holds terminal rows the queue may have
            # dropped — the last thing standing between a wedged-then-recovered
            # DB and a permanently missing turn.
            self._retry_pending(conn)
        except Exception as exc:  # writer must never crash the process
            self._count("write_errors", error=repr(exc))
            logger.warning(f"Observability writer loop error: {exc!r}")
        finally:
            if conn is not None:
                try:
                    self._maybe_write_health(conn, force=True)
                    conn.commit()
                except Exception:
                    pass
                conn.close()

    def _heartbeat(self, conn: sqlite3.Connection) -> None:
        """Idle-tick work: flush health, retry the pending ring, re-arm the breaker.

        All three are deliberately off the turn path — this runs on the writer
        thread between drains, so a wedged DB costs a background retry rather
        than a user's latency.
        """
        self._retry_pending(conn)
        self._maybe_rearm_sync_breaker()
        self._maybe_write_health(conn)

    def _retry_pending(self, conn: sqlite3.Connection) -> None:
        """Re-write terminal rows the sync path could not land (ruling I1).

        The upsert is idempotent on turn_key, so a row the queue already
        delivered is claimed as an idempotent retry rather than refused.
        """
        with self._sync_lock:
            if not self._pending:
                return
            items = list(self._pending.items())
        landed = []
        for turn_key, (turn_row, artifact_rows) in items:
            try:
                conn.execute("BEGIN IMMEDIATE")
                self.store.upsert_turn_row(
                    conn, turn_row, artifact_rows, self._redactor
                )
                conn.commit()
            except Exception as exc:
                self._rollback(conn)
                self._count("write_errors", error=f"pending retry: {exc!r}")
                break  # still unhealthy; leave the rest for the next tick
            landed.append(turn_key)
        for turn_key in landed:
            self._forget_pending(turn_key)

    def _maybe_rearm_sync_breaker(self) -> None:
        """Close the breaker only after a write probe succeeds (ruling C9).

        The cooldown elapsing proves nothing about the DB, and re-arming blind
        would put the next user turn back in front of the same wedged file.
        The probe is a diagnostics upsert on the sync path's own connection —
        the same write shape, at the same busy timeout, off the turn path.
        """
        with self._sync_lock:
            if self._sync_breaker_until == 0.0:
                return
            if time.monotonic() < self._sync_breaker_until:
                return
        conn = None
        try:
            conn = self.store._connect(
                timeout=float(
                    _env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", _DEFAULT_SYNC_WRITE_TIMEOUT_S)
                )
            )
            conn.execute("BEGIN IMMEDIATE")
            self.store.set_diagnostic(
                conn, "sync_breaker_probe", {"at": _utcnow_iso()}
            )
            conn.commit()
        except Exception:
            # Still wedged: hold the breaker open for another cooldown rather
            # than probing on every idle tick.
            with self._sync_lock:
                self._sync_breaker_until = time.monotonic() + _env_int(
                    "FW_OBS_SYNC_BREAKER_COOLDOWN_S", _DEFAULT_SYNC_BREAKER_COOLDOWN_S
                )
            return
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        with self._sync_lock:
            self._sync_breaker_until = 0.0
        with self._health_lock:
            self._health["sync_breaker_open"] = False
            self._health_dirty = True
        logger.info("Observability sync-write breaker re-armed after a successful probe")

    def _next_item(self) -> Any:
        """One item, records first; None on idle timeout (health heartbeat)."""
        try:
            return self._record_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._span_queue.get(timeout=0.25)
        except queue.Empty:
            return None

    def _drain_pending(self, limit: int = 512) -> list:
        items = []
        for _ in range(limit):
            try:
                items.append(self._record_queue.get_nowait())
                continue
            except queue.Empty:
                pass
            try:
                items.append(self._span_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _apply_batch(self, conn: sqlite3.Connection, items: list) -> None:
        flush_events: list[threading.Event] = []
        spans: list[tracing.Span] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            for item in items:
                kind = item[0]
                if kind == "span":
                    spans.append(item[1])
                elif kind == "turn":
                    self._apply_turn(conn, item)
                elif kind == "label":
                    self._apply_label(conn, item)
                elif kind == "flush":
                    flush_events.append(item[1])
            if spans:
                self.store.upsert_span_rows(conn, spans, self._redactor)
            self._maybe_write_health(conn, in_txn=True)
            conn.commit()
        except sqlite3.OperationalError as exc:
            # SQLITE_BUSY under multi-process contention [R8].
            self._rollback(conn)
            self._count("busy_retries", error=repr(exc))
            self._requeue_records(items)
        except Exception as exc:
            self._rollback(conn)
            self._count("write_errors", error=repr(exc))
        finally:
            for event in flush_events:
                event.set()

    def _apply_turn(self, conn: sqlite3.Connection, item: tuple) -> None:
        _, turn_row, artifact_rows, _retries = item
        accepted = self.store.upsert_turn_row(
            conn, turn_row, artifact_rows, self._redactor
        )
        if not accepted:
            self._count("refused_terminal_writes")
        # The row landed, so the retry ring no longer owes anyone this turn.
        # Cleared inside the batch txn rather than after the commit: a commit
        # failure rolls the batch back and requeues it, and the ring entry is
        # re-added by that path if it is still needed.
        self._forget_pending(turn_row["turn_key"])

    def _apply_label(self, conn: sqlite3.Connection, item: tuple) -> None:
        _, channel_id, conversation_id, topic, summary, _retries = item
        # Labels are persisted text too — same [R20] sink-boundary scrub as
        # turn rows and span attributes.
        topic = self._redactor.redact(topic) if topic else topic
        summary = self._redactor.redact(summary) if summary else summary
        # Single enforcement point: uniquification inside the writer's own
        # transaction (ruling I9).
        self.store.apply_label_txn(conn, channel_id, conversation_id, topic, summary)

    def _requeue_records(self, items: list) -> None:
        """Bounded retry for turn records/labels on SQLITE_BUSY; spans drop [R8]."""
        for item in items:
            kind = item[0]
            if kind == "span":
                self._count("spans_dropped")
                continue
            if kind == "flush":
                item[1].set()
                continue
            retries = item[-1]
            if retries >= _RECORD_BUSY_MAX_RETRIES:
                self._count("records_dropped")
                continue
            retried = item[:-1] + (retries + 1,)
            try:
                self._record_queue.put_nowait(retried)
            except queue.Full:
                self._count("records_dropped")

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.rollback()
        except Exception:
            pass

    def _maybe_write_health(
        self, conn: sqlite3.Connection, force: bool = False, in_txn: bool = False
    ) -> None:
        with self._health_lock:
            if not (self._health_dirty or force):
                return
            snapshot = dict(self._health)
            self._health_dirty = False
        try:
            if not in_txn:
                conn.execute("BEGIN IMMEDIATE")
            self.store.set_diagnostic(conn, "writer_health", snapshot)
            if not in_txn:
                conn.commit()
        except Exception:
            self._rollback(conn)
            with self._health_lock:
                self._health_dirty = True


# ----------------------------------------------------------------------
# Factory [R4]
# ----------------------------------------------------------------------

_sinks_lock = threading.Lock()
_sinks: dict[str, SQLiteTraceSink] = {}


def observability_enabled(default_on: bool) -> bool:
    """FW_OBSERVABILITY master switch. fastWorkflow's own entry points pass
    default_on=True; library embedders get the sink only with FW_OBSERVABILITY=1."""
    value = _env("FW_OBSERVABILITY", "1" if default_on else "0")
    return value not in ("0", "false", "False", "no", "off")


def get_observability_sink(
    workflow_path: str, *, entry_point: bool = True
) -> Optional[SQLiteTraceSink]:
    """The process-wide sink for a workflow's observability DB, or None when
    disabled. One sink (one writer thread) per DB path; closed atexit [R7].
    Never raises — a store that cannot open degrades to no sink plus a warning.
    """
    if not observability_enabled(default_on=entry_point):
        return None
    try:
        db_path = state_paths.observability_db(workflow_path)
        with _sinks_lock:
            sink = _sinks.get(db_path)
            if sink is not None and not sink._closed and _sink_is_stale(sink, db_path):
                # The DB file was deleted/replaced under the cached sink (its
                # writer would silently write into the old inode). Recycle.
                try:
                    sink.close(timeout=2.0)
                except Exception:
                    pass
                sink = None
            if sink is None or sink._closed:
                sink = SQLiteTraceSink(db_path)
                _sinks[db_path] = sink
            return sink
    except Exception as exc:
        logger.warning(f"Observability sink unavailable for {workflow_path}: {exc!r}")
        return None


def existing_observability_sink(
    workflow_path: str,
) -> Optional[SQLiteTraceSink]:
    """Return this process's live sink without constructing one."""
    db_path = state_paths.observability_db(workflow_path)
    with _sinks_lock:
        sink = _sinks.get(db_path)
        if sink is None or sink._closed or _sink_is_stale(sink, db_path):
            return None
        return sink


def _sink_is_stale(sink: SQLiteTraceSink, db_path: str) -> bool:
    try:
        return os.stat(db_path).st_ino != sink._db_ino
    except OSError:
        return True  # file gone


def close_all_sinks() -> None:
    with _sinks_lock:
        sinks = list(_sinks.values())
        _sinks.clear()
    for sink in sinks:
        try:
            sink.close()
        except Exception:
            pass


atexit.register(close_all_sinks)
