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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

import fastworkflow
from fastworkflow import capture_policy as capture_policy_module
from fastworkflow import runtime_manifest, state_paths, tracing
from fastworkflow.utils.logging import logger

SCHEMA_VERSION = 1

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

# Which drop counters carry an affected-turn-key list, and where it lives in the
# health dict. Only drops get one: a write error is about the DB, not about a turn.
_DROP_TURN_KEY_FIELDS: dict[str, str] = {
    "spans_dropped": "spans_dropped_turn_keys",
    "records_dropped": "records_dropped_turn_keys",
}

# Enough to name the affected turns of a run that lost a little evidence, small
# enough that a run losing everything cannot grow the list without bound. Past the
# cap the run is invalid anyway and the exact list has stopped being actionable.
_DROP_TURN_KEY_MAX = 256

# Counters an evidence run compares before and after (§12.4). Turn-record drops
# invalidate a run outright; the rest are reported.
_HEALTH_DELTA_COUNTERS: tuple[str, ...] = (
    "records_dropped",
    "spans_dropped",
    "write_errors",
    "refused_terminal_writes",
    "busy_retries",
    "sync_fallbacks",
)


class IncompatibleObservabilityDB(RuntimeError):
    """The DB was written by a newer fastWorkflow; readers refuse it [R11]."""


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


# Which capture profile this deployment records under (arch §12.0 delta 3).
# Defaults to `debug`, which is byte-for-byte today's behavior: EXP-003 is a
# Phase 0 instrumentation slice, so installing the policy must change nothing
# until a deployment asks it to.
CAPTURE_PROFILE_VAR = "FW_OBS_CAPTURE_PROFILE"
_DEFAULT_CAPTURE_PROFILE = "debug"

# Profiles are immutable and cheap to share, and capture runs on every command of
# every turn, so they are built once per name rather than per turn (FW-NFR-005
# overhead is an EXP-003 stop condition).
_CAPTURE_POLICY_CACHE: dict[str, "capture_policy_module.CapturePolicy"] = {}


# Retention pruning must not run while an evaluation is recording (§12.4: "pruning
# shall not run mid-evaluation") — the prune horizon is 30 days by default, but a
# size-capped prune evicts oldest-first regardless of age, so a long or
# high-volume run can delete its own early spans.
#
# Two mechanisms, because a run is not always one process: the chatbot spawns a
# server, so an in-process flag cannot reach the writer that actually prunes. The
# env var propagates to children; the counter serves a same-process harness.
SUPPRESS_PRUNE_VAR = "FW_OBS_SUPPRESS_PRUNE"
_prune_suppression_lock = threading.Lock()
_prune_suppression_depth = 0


def pruning_suppressed() -> bool:
    """Whether retention pruning is currently withheld."""
    if _env(SUPPRESS_PRUNE_VAR, "0") not in ("0", "false", "False", "no", "off"):
        return True
    with _prune_suppression_lock:
        return _prune_suppression_depth > 0


# Every FW_OBS_* knob that changes what is captured or retained, paired with its
# default. Enumerated rather than discovered by scanning os.environ for the prefix,
# because provenance must record the value **in effect** — including the defaults
# nobody set, which a scan cannot see. A run whose provenance omits
# FW_OBS_RETENTION_DAYS because it was unset is a run nobody can reproduce.
#
# FW_OBS_MAX_ATTR_BYTES lives in tracing.py and reads os.environ directly rather
# than through _env, so a value set only in a workflow env file does NOT take
# effect there. It is listed here with the resolution tracing actually performs,
# so provenance records the truth rather than the intent.
_OBS_CONFIG_VARS: tuple[tuple[str, str], ...] = (
    ("FW_OBSERVABILITY", "1"),
    (CAPTURE_PROFILE_VAR, _DEFAULT_CAPTURE_PROFILE),
    ("FW_OBS_RETENTION_DAYS", str(_DEFAULT_RETENTION_DAYS)),
    ("FW_OBS_DB_MAX_BYTES", str(_DEFAULT_DB_MAX_BYTES)),
    ("FW_OBS_INLINE_ARTIFACT_BYTES", str(_DEFAULT_INLINE_ARTIFACT_BYTES)),
    ("FW_OBS_CAPTURE_TRACEBACKS", "0"),
    ("FW_OBS_QUEUE_MAX", str(_DEFAULT_QUEUE_MAX)),
    ("FW_OBS_SYNC_WRITE_TIMEOUT_S", str(_DEFAULT_SYNC_WRITE_TIMEOUT_S)),
    ("FW_OBS_SYNC_BREAKER_COOLDOWN_S", str(_DEFAULT_SYNC_BREAKER_COOLDOWN_S)),
    (SUPPRESS_PRUNE_VAR, "0"),
)


def observability_config() -> dict[str, str]:
    """The FW_OBS_* values in effect, defaults included (§12.4)."""
    config = {name: _env(name, default) for name, default in _OBS_CONFIG_VARS}
    config["FW_OBS_MAX_ATTR_BYTES"] = str(
        os.environ.get("FW_OBS_MAX_ATTR_BYTES") or tracing._DEFAULT_MAX_ATTR_BYTES
    )
    return config


@contextlib.contextmanager
def suppress_pruning():
    """Withhold retention pruning for the duration of the block.

    Re-entrant by counting rather than by a boolean, so two nested evidence runs
    cannot have the inner one's exit re-enable pruning under the outer one.
    """
    global _prune_suppression_depth
    with _prune_suppression_lock:
        _prune_suppression_depth += 1
    try:
        yield
    finally:
        with _prune_suppression_lock:
            _prune_suppression_depth -= 1


def resolve_capture_policy() -> "capture_policy_module.CapturePolicy":
    """The policy this process captures under.

    Raises `CaptureProfileError` on an unrecognized profile name rather than
    falling back — see `capture_policy.policy_for_profile`. The sink resolves this
    in its constructor so a misconfigured deployment fails at startup instead of
    discovering months later that it recorded tenant data verbatim.
    """
    name = _env(CAPTURE_PROFILE_VAR, _DEFAULT_CAPTURE_PROFILE)
    policy = _CAPTURE_POLICY_CACHE.get(name)
    if policy is None:
        policy = capture_policy_module.policy_for_profile(name)
        _CAPTURE_POLICY_CACHE[name] = policy
    return policy


# Turn columns the capture policy deliberately does NOT touch.
#
# These two are not evidence, they are operational state: `get_memory_window` and
# `_USABLE_TURN_FILTER` read exactly `conversation_summary` and
# `conversation_traces` to rebuild the agent's conversation memory, and the filter
# requires the summary to be non-NULL. Withholding them would not reduce what a
# bundle exposes — it would make the agent forget, which is a behavior change and
# therefore outside a Phase 0 slice.
#
# PII in conversation memory is a real gap; it is fix-cj4's. It needs a redaction
# that leaves memory usable, which is a different problem from withholding
# evidence, and solving it by omission here would silently degrade every
# evidence-profile run's agent.
_POLICY_EXEMPT_TURN_COLUMNS = frozenset({"conversation_summary", "conversation_traces"})

# Turn columns that are pure evidence — nothing operational reads them — paired
# with what they actually contain. `failure_reason` is `opaque-payload` rather
# than text because it can embed a provider error body (the [R20] scenario), so
# nobody can say what is in it.
_POLICED_TURN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_message", "user-text"),
    ("refined_user_message", "user-text"),
    ("answer", "user-text"),
    ("failure_reason", "opaque-payload"),
)

# ----------------------------------------------------------------------
# The write paths that do NOT ride the TurnResult pipeline (fix-ajv.9)
# ----------------------------------------------------------------------
#
# `serialize_turn_result` is where the capture policy meets a turn, and
# `upsert_turn_row` is where the credential scrub meets one. Five persisted
# surfaces reach SQLite without passing through either: conversation labels,
# feedback, train-run metrics, writer diagnostics, and the SCALAR columns beside
# a span's (already scrubbed) `attributes` JSON. FW-REQ-002 clause 3 requires
# every captured field to have a declared policy, so each of the five is decided
# here rather than by omission — including the three that are deliberately
# scrub-only, whose reasons are recorded at their write sites.
#
# Policy paths are named constants because a deployment re-admitting one of these
# under the evidence profile has to spell the path exactly (see
# `CapturePolicy.policy_for`), and a path that only exists as a literal inside a
# method is a path nobody can find in order to spell it.
POLICY_PATH_SPAN_NAME = "span.name"
POLICY_PATH_SPAN_COMMAND_NAME = "span.command_name"
POLICY_PATH_SPAN_CONTEXT = "span.context"
POLICY_PATH_CONVERSATION_TOPIC = "conversation.topic"
POLICY_PATH_CONVERSATION_SUMMARY = "conversation.summary"
POLICY_PATH_TRAIN_METRICS = "train_run.metrics_json"


def _protected_text(
    value: Any,
    *,
    redactor: Redactor,
    policy: "capture_policy_module.CapturePolicy",
    field_path: str,
    classification: str,
) -> Any:
    """Credential-scrub a persisted string, then apply the capture policy to it.

    **Scrub first, policy second**, which is the opposite order from
    `_POLICED_TURN_COLUMNS` (there the policy runs in `serialize_turn_result` and
    the scrub runs later, in `upsert_turn_row`). Two reasons it has to be this way
    on these paths:

    * A conversation label can arrive by either of two routes —
      `SQLiteTraceSink._apply_label`, which scrubs before calling
      `apply_label_txn`, or `ObservabilityStore.record_conversation_label`, which
      does not. Scrubbing first makes both produce `policy(scrub(text))`, because
      the scrub is idempotent. Policing first would give the same label two
      different digests depending on which route wrote it, and a digest that
      depends on plumbing is not a digest anyone can compare.
    * The badge left behind carries a digest of what it replaced. Digesting the
      unscrubbed text would make the badge a confirmation oracle for a guessed
      credential, which is a strange thing for a redaction record to be.

    Returns TEXT, always: an envelope is serialized here because every caller
    binds the result to a TEXT column and sqlite3 cannot bind a mapping. Same
    reasoning as `_policed_column`, which does it for the turn row.
    """
    if not value:
        return value
    scrubbed = redactor.redact(value)
    captured = policy.apply(field_path, scrubbed, classification=classification)
    if capture_policy_module.is_capture_envelope(captured):
        return json.dumps(captured, ensure_ascii=False)
    return captured


class WriterHealthDelta(BaseModel):
    """What the store lost between two health snapshots (§12.4).

    The asymmetry is the point. A dropped **turn record** invalidates an evidence
    run: turn records are the evidence, and one missing turn means the run's
    numerator and denominator disagree in a way no analysis can repair. A dropped
    **span** does not invalidate the run — spans are best-effort by design — but it
    must be reported with the turns it affected, so a reader knows which turns have
    incomplete detail rather than assuming all of them are whole.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    records_dropped: int = 0
    spans_dropped: int = 0
    write_errors: int = 0
    refused_terminal_writes: int = 0
    busy_retries: int = 0
    sync_fallbacks: int = 0
    records_dropped_turn_keys: tuple[str, ...] = ()
    spans_dropped_turn_keys: tuple[str, ...] = ()
    # Non-zero means the affected-turn lists were capped and are incomplete.
    dropped_turn_keys_elided: int = 0
    # True when either snapshot was unavailable. Distinct from "no drops": nothing
    # was compared, so nothing may be claimed.
    incomparable: bool = False

    @property
    def lost_turn_records(self) -> bool:
        return self.records_dropped > 0

    @property
    def evidence_valid(self) -> bool:
        """Whether a run over this interval may be reported as evidence.

        False when a turn record was dropped, and False when the interval could
        not be compared at all — an unknown is not a pass. Dropped spans leave
        this True; they are reported through `problems()`.
        """
        return not self.incomparable and not self.lost_turn_records

    def problems(self) -> tuple[str, ...]:
        """Every reason this interval is imperfect, worst first.

        Returns all of them rather than the first, so an operator sees the whole
        picture in one pass.
        """
        found: list[str] = []
        if self.incomparable:
            found.append(
                "writer health could not be compared (a snapshot was missing); "
                "evidence validity is unknown, which is not the same as valid"
            )
        if self.records_dropped:
            affected = ", ".join(self.records_dropped_turn_keys) or "unknown turns"
            found.append(
                f"{self.records_dropped} turn record(s) DROPPED, affecting: "
                f"{affected}. The run is not valid evidence (§12.4)."
            )
        if self.spans_dropped:
            affected = ", ".join(self.spans_dropped_turn_keys) or "unknown turns"
            found.append(
                f"{self.spans_dropped} span(s) dropped, affecting: {affected}. "
                "These turns have incomplete detail; the run remains valid."
            )
        if self.dropped_turn_keys_elided:
            found.append(
                f"{self.dropped_turn_keys_elided} further affected turn key(s) were "
                "not recorded (list capped); the lists above are incomplete"
            )
        if self.refused_terminal_writes:
            found.append(
                f"{self.refused_terminal_writes} write(s) to an already-terminal "
                "turn row were refused"
            )
        if self.write_errors:
            found.append(f"{self.write_errors} write error(s)")
        return tuple(found)


def health_delta(
    before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]
) -> WriterHealthDelta:
    """Compare two `health_snapshot()` results.

    Counters are cumulative and monotonic, so the delta is a subtraction; the
    affected-turn-key lists are set differences, which is what makes the result
    specific to this run rather than to the DB's whole history.

    A missing snapshot yields `incomparable=True` rather than a zero delta,
    because "we could not tell" and "nothing was dropped" are the two answers an
    evidence gate must never confuse.
    """
    if before is None or after is None:
        return WriterHealthDelta(incomparable=True)

    counters = {
        name: max(0, int(after.get(name) or 0) - int(before.get(name) or 0))
        for name in _HEALTH_DELTA_COUNTERS
    }
    new_keys: dict[str, tuple[str, ...]] = {}
    for counter, field in _DROP_TURN_KEY_FIELDS.items():
        seen_before = set(before.get(field) or ())
        new_keys[field] = tuple(
            key for key in (after.get(field) or ()) if key not in seen_before
        )
    return WriterHealthDelta(
        **counters,
        records_dropped_turn_keys=new_keys["records_dropped_turn_keys"],
        spans_dropped_turn_keys=new_keys["spans_dropped_turn_keys"],
        dropped_turn_keys_elided=max(
            0,
            int(after.get("dropped_turn_keys_elided") or 0)
            - int(before.get("dropped_turn_keys_elided") or 0),
        ),
    )


def _capture_classify_for_turn(turn_result: Any) -> Optional[Any]:
    """Resolve ``RuntimeMetadata.capture_classification`` for one turn, if any.

    The workflow folderpath is carried on ``TurnResult.metadata`` because the
    sink has no other durable link to the manifest registered at startup.
    Unregistered or absent metadata yields None, which is the evidence
    profile's default-deny input.
    """
    metadata = getattr(turn_result, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return None
    folderpath = metadata.get("workflow_folderpath")
    if not folderpath:
        return None
    runtime = runtime_manifest.get_runtime_metadata(folderpath)
    if runtime is None:
        return None
    return runtime.capture_classification


def _policy_classification(
    classify: Optional[Any], command_name: str, field_name: str
) -> Optional[str]:
    """The workflow's declared classification for one parameter, or None.

    None is the default-deny input, so a resolver that raises must be treated as
    "unclassified" rather than allowed to lose the whole turn record: under the
    evidence profile that omits the field, which is the conservative direction.
    """
    if classify is None:
        return None
    try:
        return classify(command_name, field_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"capture classification resolver failed: {exc!r}")
        return None


def _apply_capture_policy(
    record: dict[str, Any],
    policy: "capture_policy_module.CapturePolicy",
    classify: Optional[Any] = None,
) -> None:
    """Apply the field policy to one dumped TurnResult, in place.

    Runs on the `model_dump()` copy, never on the live accumulator objects, so
    what the caller and the user see is untouched — the policy governs what is
    *persisted*, which is the whole reason it is applied here rather than at the
    sink's string boundary.

    Ordering matters: this runs BEFORE the artifact-offload pass. A withheld
    artifact collapses to a small envelope and is therefore never offloaded, so
    its bytes never reach the `artifacts` table. Applying the policy afterwards
    would redact the record while leaving the raw value in `inline_value`.
    """
    for command_output in record.get("turn_output", {}).get("command_outputs", []):
        command_name = command_output.get("command_name") or "unknown"
        # A FAILED command's response and artifacts are diagnostic content —
        # an exception repr, a message, a traceback — not the command's normal
        # output, so they must not inherit the policy written for its happy
        # path. `CapturePolicy.apply` returns a value WHOLE when a declared
        # policy is not gated for this sink, so a perfectly reasonable
        # `command.X.response` rule (X's normal response is benign, keep it)
        # would also release X's failure text once fix-ajv.16 started naming
        # failed commands. A separate segment makes releasing error text
        # something a deployment has to say, rather than something it inherits.
        # fix-ajv.18.
        #
        # ask_user is excluded deliberately [A7]: `success=False` on an
        # ask_user entry means the question is still unanswered, not that
        # anything failed, and its response is the user's ANSWER — ordinary
        # user text that belongs on the ordinary path.
        #
        # Read via the structural marker with the name as fallback, mirroring
        # `CommandOutput.is_ask_user` — this walks the model_dump()ed dict, so
        # it cannot call the property. `ask_user_entry` absent means a record
        # written before that field existed (fix-ajv.17); True/False are
        # authoritative, and False is what a failed command NAMED `ask_user`
        # carries, which is the whole point of not testing the name here.
        response_dict = command_output.get("command_response") or {}
        marker = command_output.get("ask_user_entry")
        is_ask_user = marker if marker is not None else command_name == "ask_user"
        is_failure = response_dict.get("success") is False and not is_ask_user
        # PARAMETERS DELIBERATELY STAY ON THE ORDINARY PATH, and this asymmetry
        # is the point rather than an oversight. A failure's parameters are the
        # SAME values the success path carries, so a rule written to gate them
        # must keep applying; moving them under `.error.` would stop that rule
        # matching and fall through to the profile default — which under
        # `debug` returns the value whole. Separating them would un-gate the
        # one field group the success policy is right about.
        outcome_prefix = (
            f"command.{command_name}.error" if is_failure else f"command.{command_name}"
        )
        parameters = command_output.get("command_parameters")
        if isinstance(parameters, dict):
            for field_name in list(parameters):
                parameters[field_name] = policy.apply(
                    f"command.{command_name}.parameters.{field_name}",
                    parameters[field_name],
                    classification=_policy_classification(
                        classify, command_name, field_name
                    ),
                )
        elif parameters is not None:
            # The ask_user role inversion [A10]: for an `ask_user` entry
            # `command_parameters` is the agent's *question* as a str, not a
            # parameter mapping — and the response below is the user's *answer*.
            command_output["command_parameters"] = policy.apply(
                f"command.{command_name}.parameters",
                parameters,
                classification="user-text",
            )

        response = command_output.get("command_response") or {}
        if response.get("response"):
            response["response"] = policy.apply(
                f"{outcome_prefix}.response",
                response["response"],
                classification="user-text",
            )
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for key in list(artifacts):
            value = artifacts[key]
            # An artifact ref envelope is a pointer, not content: the value has
            # already been moved out, and digesting a pointer loses the join
            # without protecting anything.
            if isinstance(value, dict) and "__fw_artifact_ref__" in value:
                continue
            artifacts[key] = policy.apply(
                f"{outcome_prefix}.artifacts.{key}",
                value,
                classification=_policy_classification(classify, command_name, key),
            )


def _policed_column(
    policy: "capture_policy_module.CapturePolicy",
    column: str,
    classification: str,
    value: Any,
) -> Any:
    """A turn text column after policy, still bindable as TEXT.

    An envelope is serialized rather than returned as a dict: these are TEXT
    columns and sqlite3 cannot bind a mapping, and `conversation_summary`'s
    non-NULL contract shows how much the read side depends on their shape.
    """
    if not value:
        return value
    captured = policy.apply(f"turn.{column}", value, classification=classification)
    if capture_policy_module.is_capture_envelope(captured):
        return json.dumps(captured, ensure_ascii=False)
    return captured


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def serialize_turn_result(
    turn_result: Any,
    *,
    policy: "Optional[capture_policy_module.CapturePolicy]" = None,
    classify: Optional[Any] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a TurnResult into (turn_row, artifact_rows) at emission time.

    - ``record_json`` holds the full internal TurnResult (post-envelope,
      post-capture-policy, pre-credential-redaction — the sink redacts the
      serialized text) [R10].
    - Any artifact value over ``FW_OBS_INLINE_ARTIFACT_BYTES`` is replaced in
      place by a ref envelope; the artifacts table is the only value holder.
    - ``traceback`` artifacts persist only under FW_OBS_CAPTURE_TRACEBACKS=1
      [R20].

    Runs in the caller thread so the row snapshots the turn as emitted (the
    accumulator's CommandOutput objects mutate on resume).

    The capture policy (arch §6.6) runs here rather than at the sink's string
    boundary because it is per-field and `Redactor` operates on already-serialized
    JSON: by the time the text exists, the field structure the policy classifies
    is gone. The two compose — the policy decides what is captured, the redactor
    still scrubs credential shapes out of whatever survives. `policy=None`
    resolves `FW_OBS_CAPTURE_PROFILE`, which defaults to the verbatim `debug`
    profile, so this is a no-op unless a deployment opts in.
    """
    turn_output = turn_result.turn_output
    inline_limit = _env_int("FW_OBS_INLINE_ARTIFACT_BYTES", _DEFAULT_INLINE_ARTIFACT_BYTES)
    capture_tracebacks = _env("FW_OBS_CAPTURE_TRACEBACKS", "0") == "1"
    policy = policy or resolve_capture_policy()

    try:
        record = turn_result.model_dump(mode="python")
    except Exception:
        record = {"turn_output": {"turn_key": turn_output.turn_key}}
    record = _sanitize_json_value(record)
    # computed_field `success` is included by model_dump; make sure it is
    # present even on the fallback path.
    record.setdefault("turn_output", {}).setdefault("success", turn_output.success)

    _apply_capture_policy(record, policy, classify)

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
        "record_version": 1,
        "record_json": json.dumps(record, ensure_ascii=False),
    }
    for column, classification in _POLICED_TURN_COLUMNS:
        turn_row[column] = _policed_column(
            policy, column, classification, turn_row[column]
        )
    return turn_row, artifact_rows


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------

_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS conversations (
        channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
        topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
        started_at TEXT, last_turn_at TEXT, updated_at TEXT,
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
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(channel_id, conversation_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_key)",
]


class ObservabilityStore:
    """Schema owner + synchronous operations on one observability DB.

    Thread/process-safe by construction: every method opens its own
    short-lived WAL connection (timeout=30, ``BEGIN IMMEDIATE`` for writes).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    # -- protection for the non-TurnResult write paths (fix-ajv.9) -------
    #
    # Built lazily and cached on the instance rather than in __init__, for three
    # reasons that each rule out the alternatives:
    #
    # * `Redactor()` walks the whole environment plus every loaded fastworkflow
    #   env file. A store that only ever reads — every chatbot request opens one —
    #   must not pay for that, and `ReadOnlyObservabilityStore` deliberately does
    #   not run this __init__ at all, so an attribute set there would not exist.
    # * A process-wide singleton would freeze the secret list at whichever import
    #   happened first, which is exactly the ordering a test (or a late
    #   `fastworkflow.init`) changes underneath it.
    # * Per call would be worse still: `set_diagnostic` runs on the writer's
    #   health heartbeat, so construction cost there is on a repeating path.

    def _store_redactor(self) -> Redactor:
        redactor = getattr(self, "_redactor", None)
        if redactor is None:
            redactor = Redactor()
            self._redactor = redactor
        return redactor

    def _store_capture_policy(self) -> "capture_policy_module.CapturePolicy":
        """This store's capture profile, resolved once.

        Resolved here as well as on the sink because the sync label path
        (`record_conversation_label`) reaches SQLite without a sink in sight —
        that is the whole of item 5 in fix-ajv.9.
        """
        policy = getattr(self, "_capture_policy", None)
        if policy is None:
            policy = resolve_capture_policy()
            self._capture_policy = policy
        return policy

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

    # -- identity [R1] ---------------------------------------------------

    def mint_conversation_id(self, channel_id: str, legacy_floor: int = 0) -> int:
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
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at, updated_at)
                   VALUES (?, ?, NULL, NULL, 'open', 1, ?, NULL, ?)""",
                (channel_id, new_id, now, now),
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
        # fix-ajv.9 item 5, the one live gap: BOTH layers, applied here because
        # this is the single label-write enforcement point and the production
        # route to it — run_fastapi_mcp/utils.ensure_topic_and_summary calling
        # `record_conversation_label` — is the SYNC one, which never touches
        # `SQLiteTraceSink._apply_label` and so never met the credential scrub
        # either. Protecting the enforcement point rather than the two callers is
        # what makes it impossible to add a third route that skips this.
        #
        # `user-text` and not `controlled-vocabulary`: a topic and a summary are
        # LLM output generated FROM a real user's conversation, so their content
        # is whatever the conversation was about — an order number, a name, an
        # address. Under `evidence` they become badges; the UI degrades to
        # "a 34-byte user-text title was here", which is §12.0 delta 3's
        # requirement and is why this is not simply an omission.
        #
        # AFTER uniquification, not before. `_unique_topic_in_txn` compares
        # casefolded titles and appends " 1", " 2" on collision: policing first
        # would append that suffix outside the envelope's closing brace and leave
        # a column holding text that no longer parses as JSON. The cost is that
        # collision suffixing stops distinguishing anything under `evidence`,
        # where two identical titles digest identically — acceptable, because
        # uniquification exists so a human can pick a conversation out of a list
        # by its title, and under `evidence` every title in that list is a badge.
        redactor = self._store_redactor()
        policy = self._store_capture_policy()
        topic = _protected_text(
            topic,
            redactor=redactor,
            policy=policy,
            field_path=POLICY_PATH_CONVERSATION_TOPIC,
            classification="user-text",
        )
        summary = _protected_text(
            summary,
            redactor=redactor,
            policy=policy,
            field_path=POLICY_PATH_CONVERSATION_SUMMARY,
            classification="user-text",
        )
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
        # fix-ajv.9 item 4: the scalar columns beside `attributes`. The
        # attributes JSON has been scrubbed since [R20]; the four scalars written
        # next to it never were, and one of them can carry entity content.
        #
        # `context` is `workflow.current_command_context_displayname`, which calls
        # a workflow-supplied `get_displayname(instance)` hook — the bundled
        # simple_workflow_template returns the work item's absolute path from it.
        # So it is not a type name, it is a label about a specific instance:
        # `user-text`, and withheld under `evidence`.
        #
        # `name` and `command_name` are closed vocabularies — the span taxonomy in
        # tracing.py and the workflow's own command set — so they are declared
        # rather than withheld, which is what FW-REQ-002 clause 3 asks for. The
        # `controlled-vocabulary` default bounds them at 256 bytes and passes
        # anything shorter through untouched, so this is inert for every real
        # command name while still refusing to let an unbounded value in.
        #
        # `channel_id` is SCRUB-ONLY, and deliberately so. It is an identifier, so
        # the evidence default would digest it, and a digest still joins — but
        # `forget_channel` erases a channel with `DELETE FROM spans WHERE
        # channel_id=?`, so digesting this column would silently narrow
        # first-class erasure [R21] to whatever the `trace_id IN (...)` fallback
        # happens to still cover. Reducing exposure by weakening erasure is not a
        # trade this slice gets to make; a joinable pseudonym applied to
        # turns/artifacts/spans at once, with `forget_channel` taught to match it,
        # is the real fix and is follow-up work.
        policy = self._store_capture_policy()
        for span in spans:
            attributes = redactor.redact(
                json.dumps(_sanitize_json_value(span.attributes), ensure_ascii=False)
            )
            span_name = _protected_text(
                span.name,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_NAME,
                classification="controlled-vocabulary",
            )
            command_name = _protected_text(
                span.command_name,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_COMMAND_NAME,
                classification="controlled-vocabulary",
            )
            context = _protected_text(
                span.context,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_CONTEXT,
                classification="user-text",
            )
            # `Redactor.redact` returns a falsy input unchanged, so a None
            # channel_id stays None rather than becoming "".
            channel_id = redactor.redact(span.channel_id)
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
                    span_name,
                    span.kind,
                    channel_id,
                    command_name,
                    context,
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
        for text_col in (
            "user_message",
            "refined_user_message",
            "answer",
            "failure_reason",
            "conversation_summary",
            "conversation_traces",
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
                conn, turn_row["channel_id"], turn_row["conversation_id"]
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
        self, conn: sqlite3.Connection, channel_id: str, conversation_id: int
    ) -> int:
        row = conn.execute(
            "SELECT next_ordinal FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        if row is None:
            # Conversation row not minted here (e.g. restored session) —
            # create it so ordinals stay dense from 1.
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at)
                   VALUES (?, ?, NULL, NULL, 'open', 2, ?, NULL)""",
                (channel_id, conversation_id, _utcnow_iso()),
            )
            return 1
        ordinal = int(row["next_ordinal"] or 1)
        conn.execute(
            "UPDATE conversations SET next_ordinal=? WHERE channel_id=? AND conversation_id=?",
            (ordinal + 1, channel_id, conversation_id),
        )
        return ordinal

    def reserve_turn_ordinal(self, channel_id: str, conversation_id: int) -> Optional[int]:
        """Reserve a turn ordinal in a tiny standalone transaction.

        Used by the sync-first emit's degraded fallback so ordinals stay
        chronological even when the row itself is queued (ruling I6).
        Returns None when the reservation itself cannot be made.
        """
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                ordinal = self._assign_ordinal(conn, channel_id, conversation_id)
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
        """Upsert a turn's feedback. Credential-scrubbed, NOT policy-withheld.

        fix-ajv.9 item 1, and the one of the five where the two layers disagree.
        The scrub applies for the same reason it applies everywhere: it is
        unconditional, and `nl_feedback` is free text a user typed, which is a
        place a pasted token lands. Scrubbing serialized JSON cannot corrupt it —
        every credential pattern is confined to characters that cannot appear
        unescaped inside a JSON string, so a replacement can never cross a
        delimiter (pinned by test).
        WHY NO CAPTURE POLICY: this column is read by `get_memory_window`,
        which passes the parsed value straight into `dspy.History` through
        `conversation_history_io.restore_history_from_turns` — it is the agent's
        memory of being corrected, not evidence about the agent. Under `evidence`
        a withheld value would still parse, so the agent would silently receive a
        badge dict where its feedback used to be and behave differently. That is a
        behavior change, not a reduction in exposure, and Phase 0 does not make
        those; it is the same call `_POLICY_EXEMPT_TURN_COLUMNS` records for
        `conversation_summary` and `conversation_traces`, and it belongs with
        fix-cj4's conversation-memory redaction, which has to leave memory usable.
        """
        feedback_json = self._store_redactor().redact(feedback_json)
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
        """Persist one training run's metrics at publication time (Phase 6).

        fix-ajv.9 item 2: BOTH layers, classified `opaque-payload`.

        Not `controlled-vocabulary`, which is what a dict of thresholds and F1
        scores looks like from the outside. `collect_train_metrics` assembles this
        by reading whatever JSON is sitting in `___command_info`, and one of those
        files carries free text: `heldout_evaluation.EscalationScore.failures`
        records the verbatim `utterance` of every case that failed, and
        `metrics_persistence` copies the whole `escalation` block through. Those
        utterances are synthetic today, but "nobody can enumerate what is in
        here" is the definition of `opaque-payload`, and default-deny exists for
        precisely the field whose contents grow when someone edits a file
        elsewhere.

        An evidence deployment that has reviewed its metrics and wants them in the
        bundle re-admits them by name — a `CaptureFieldPolicy` on
        `POLICY_PATH_TRAIN_METRICS` with `redact_before_trace=False`. The other
        four columns of the row (run_id, fingerprint, timestamps) are unpoliced,
        so a bundle always knows a training run happened and which sources it was
        built from, even when the metrics themselves are a badge.
        """
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
                    _protected_text(
                        json.dumps(_sanitize_json_value(metrics), ensure_ascii=False),
                        redactor=self._store_redactor(),
                        policy=self._store_capture_policy(),
                        field_path=POLICY_PATH_TRAIN_METRICS,
                        classification="opaque-payload",
                    ),
                ),
            )
            conn.commit()

    def set_diagnostic(self, conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
        """Upsert one diagnostics row. Credential-scrubbed, NOT policy-withheld.

        fix-ajv.9 item 3. The scrub earns its place here more than anywhere else
        on this list: `writer_health.last_error` is `repr(exc)`, and the [R20]
        scenario that motivated the redactor in the first place is a LiteLLM
        `AuthenticationError` whose body echoes the key.

        WHY NO CAPTURE POLICY: this table is not a record of the workload, it is
        the record of whether the record can be trusted. `health_delta` and
        `evidence_run` read `writer_health` to decide whether a run may be
        reported as evidence at all, and `WriterHealthDelta.problems()` names the
        affected turn keys so a partly-damaged run can be salvaged instead of
        discarded. Withholding it under the `evidence` profile would blind the
        evidence gate — under the one profile that exists to make the gate
        meaningful — and would digest the very turn keys an operator needs in
        order to go and look at those turns.
        """
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (
                key,
                self._store_redactor().redact(
                    json.dumps(value, ensure_ascii=False)
                ),
                _utcnow_iso(),
            ),
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
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Turn rows, newest first, without record_json (fetch one turn for that)."""
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
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT turn_key, channel_id, conversation_id, ordinal, user_message, "
            "entry_workflow_name, entry_context, status, success, failure_reason, "
            "answer, started_at, completed_at, suspended_ms "
            f"FROM turns{where} ORDER BY turn_key DESC LIMIT ? OFFSET ?"
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
        """Copy the DB to `destination` as a single consistent file.

        Uses ``VACUUM INTO`` rather than copying the file, and that is not a
        micro-optimization. The store runs in WAL mode, so at any instant the
        newest committed rows may live in the ``-wal`` sidecar and not in the main
        file: ``shutil.copy`` of the DB alone silently produces an archive missing
        the end of the run — precisely the turns an evaluation cares most about.
        ``VACUUM INTO`` takes a read transaction and writes one compacted,
        checkpointed, WAL-free file, which is also what "immutable bundle" wants.

        Returns the archive's path, byte size and SHA-256. The digest is what makes
        the bundle checkable later: an archive nobody can verify is a copy, not
        evidence.

        The file is left mode 0444, which is what makes "immutable" structural
        rather than aspirational. Without it, merely *reading* the archive breaks
        it: `ObservabilityStore.__init__` runs `_ensure_schema` and switches the DB
        to WAL, so opening an archive to inspect it rewrites its header, spawns a
        `-wal` sidecar, and changes the sha256 recorded here — the digest would
        then fail to verify and nobody could tell tampering from a colleague
        having looked. Read archives with `ReadOnlyObservabilityStore`, which this
        permission now enforces instead of merely recommending.

        Refuses an existing destination. Overwriting an archive is not a recovery
        from a mistake, it is the destruction of the previous run's evidence.
        """
        target = Path(destination)
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing evidence archive: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            # Parameterized: the path is a value here, not identifier syntax.
            conn.execute("VACUUM INTO ?", (str(target),))

        # Digest before chmod, so what is recorded is what was written.
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

        Returns ``{"suppressed": 1}`` and deletes nothing while an evidence run
        holds pruning off (§12.4). That marker matters: an all-zero result would be
        indistinguishable from "there was nothing to prune", so a caller checking
        whether retention ran could not tell the difference.
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
        # Resolved once, here, so an unrecognized FW_OBS_CAPTURE_PROFILE fails
        # when the sink is built rather than on every turn — and so a deployment
        # that asked for `evidence` cannot end up running verbatim.
        self._capture_policy = resolve_capture_policy()
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
            # Which turns lost evidence, per §12.4. Bounded; see _count.
            "spans_dropped_turn_keys": [],
            "records_dropped_turn_keys": [],
            "dropped_turn_keys_elided": 0,
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
            # A span's trace_id IS the logical turn key (tracing.py), so the
            # affected turn is known here without extra plumbing.
            self._count("spans_dropped", turn_key=span.trace_id)
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
            turn_row, artifact_rows = serialize_turn_result(
                record,
                policy=self._capture_policy,
                classify=_capture_classify_for_turn(record),
            )
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
                turn_row["channel_id"], turn_row["conversation_id"]
            )
        if turn_row["status"] in TERMINAL_TURN_STATUSES:
            self._remember_pending(turn_row, artifact_rows)
        try:
            self._record_queue.put(
                ("turn", turn_row, artifact_rows, 0), timeout=_RECORD_PUT_TIMEOUT_S
            )
        except queue.Full:
            self._count("records_dropped", turn_key=turn_row.get("turn_key"))
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
                self._count("records_dropped", turn_key=oldest)
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

    def _count(
        self, key: str, error: Optional[str] = None, turn_key: Optional[str] = None
    ) -> None:
        """Bump one health counter, and remember which turn a drop belonged to.

        `turn_key` exists because §12.4 requires dropped spans to be "reported
        with the affected turn keys": a count alone tells an evaluation that it
        lost evidence but not which turns are now incomplete, which is the only
        fact that lets a run be salvaged rather than discarded. Bounded, because a
        pathological run must not turn a drop counter into a memory leak — the
        elided count preserves the honesty of the set when the cap is hit.
        """
        with self._health_lock:
            self._health[key] = int(self._health.get(key) or 0) + 1
            if error is not None:
                self._health["last_error"] = error[:500]
            if turn_key and key in _DROP_TURN_KEY_FIELDS:
                affected = self._health[_DROP_TURN_KEY_FIELDS[key]]
                if turn_key not in affected:
                    if len(affected) < _DROP_TURN_KEY_MAX:
                        affected.append(turn_key)
                    else:
                        self._health["dropped_turn_keys_elided"] = (
                            int(self._health.get("dropped_turn_keys_elided") or 0) + 1
                        )
            self._health_dirty = True

    def health_snapshot(self) -> dict[str, Any]:
        """A copy of the live counters, without waiting for the writer to persist.

        The `diagnostics` row lags by up to a heartbeat, so an evidence run that
        read only the row could snapshot a drop that had already happened as
        though it had not. Lists are copied so a caller holding two snapshots
        cannot find they are the same object.
        """
        with self._health_lock:
            snapshot = dict(self._health)
        for field in _DROP_TURN_KEY_FIELDS.values():
            snapshot[field] = list(snapshot.get(field) or ())
        return snapshot

    def persist_health(self) -> None:
        """Force the counters into the `diagnostics` row.

        Called at the end of an evidence run so the archived DB carries the same
        verdict the in-process delta reported; without it the archive can say
        "healthy" about a run that dropped records after the last heartbeat.
        """
        try:
            with self.store._connect() as conn:
                self._maybe_write_health(conn, force=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not persist writer health: {exc!r}")

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
                self._count("spans_dropped", turn_key=getattr(item[1], "trace_id", None))
                continue
            if kind == "flush":
                item[1].set()
                continue
            # A "turn" item carries its row at [1]; a "label" item has no turn.
            turn_key = item[1].get("turn_key") if kind == "turn" else None
            retries = item[-1]
            if retries >= _RECORD_BUSY_MAX_RETRIES:
                self._count("records_dropped", turn_key=turn_key)
                continue
            retried = item[:-1] + (retries + 1,)
            try:
                self._record_queue.put_nowait(retried)
            except queue.Full:
                self._count("records_dropped", turn_key=turn_key)

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


def existing_observability_sink(workflow_path: str) -> Optional[SQLiteTraceSink]:
    """The live sink for this workflow's DB **if this process already has one**.

    Deliberately does NOT construct. `get_observability_sink` creating a sink on
    demand is right for a writer and wrong for an observer: a harness driving a
    separate server process has no sink of its own, and asking for one starts a
    second writer thread against a database another process is writing AND runs
    `SQLiteTraceSink.__init__`'s opportunistic prune — so an evidence gate could
    prune the very evidence it was opening to protect. fix-ajv.13.

    Never raises; a path that cannot be resolved reads as "no sink".
    """
    try:
        db_path = state_paths.observability_db(workflow_path)
    except Exception:
        return None
    with _sinks_lock:
        sink = _sinks.get(db_path)
    if sink is None or sink._closed:
        return None
    return sink


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
