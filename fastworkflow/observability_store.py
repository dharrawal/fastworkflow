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
from typing import Any, Iterable, Optional

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
# Distillation retention (distillation design §10, [DR24][DR43][DR52]).
_DEFAULT_DISTILL_NEGATIVE_PIN_DAYS = 90
# §12 rule 4: a verdict note is an annotation, not a document.
_VERDICT_NOTE_MAX_BYTES = 4096
# How many per-trace barrier marks to hold before sweeping the applied ones.
_TRACE_MARK_CEILING = 1024
# Whole traces per size-cap eviction batch [DR27]. Small, because a batch is
# one transaction and a trace can be large; the loop runs up to
# _PRUNE_MAX_BATCHES times.
_EVICT_TRACES_PER_BATCH = 32
_DEFAULT_DISTILL_PIN_MAX_FRACTION = 0.5
# Per-span bytes the attribute length does not account for (row header, the
# id/name columns, and the index entries the span carries). Used only to size
# the pinned set against the cap; it is an estimate and is labelled one.
_PINNED_ROW_OVERHEAD_BYTES = 256
# Feature marker written into diagnostics instead of a SCHEMA_VERSION bump
# ([DR28]): the version gate is fail-closed and coarse, so a bump would make
# every v3.2.0 build refuse whole DBs over tables it never queries.
FEATURE_DISTILLATION_V1 = "distillation_v1"
# The experiment container (`fix-bn1`, experiment_container_design.md `[XR5]`).
# Same mechanism, same reason: three additive tables and six additive columns
# are not a compatibility break, and `user_version` stopped being a usable
# signal at v3.2.0.
FEATURE_EXPERIMENTS_V1 = "experiments_v1"


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
def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _chunked(values: list[Any], size: int = 500) -> Any:
    """Slice an id list into SQLite-parameter-sized IN() chunks."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _in_placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _merge_nonzero(counts: dict[str, int], added: dict[str, int]) -> None:
    """Merge sweep counters into a prune result, dropping the zeroes.

    The historical result is exactly ``{"spans", "artifacts"}``; a workflow
    that never distills must keep seeing that, so a distillation counter shows
    up only when it actually did something.
    """
    for key, value in added.items():
        if value:
            counts[key] = counts.get(key, 0) + value


# `fix-sb8.18`: a `pinned_span_count` the WRITER resolves. `[DR43]` wants the
# trace's live span count as the pin is taken, and the producer used to read it
# with its own sqlite3 connection — a synchronous DB read on the turn thread at
# the completion of every pinned run, stacked on top of the barrier waits. The
# writer thread already holds a connection inside the batch transaction that
# writes the row, so the count is a free join there and costs the user nothing.
COUNT_LIVE_SPANS = "@fw.count-live-spans"


class OrphanedCitation(Exception):
    """A citation whose divergence row never landed (`fix-sb8.16`).

    Raised by `upsert_distillation_row` so the writer counts the suppression
    instead of writing a citation that points at nothing — §15's provenance
    recipes join through `distillation_insight_citations` and would read an
    orphan as evidence an insight was drawn from a divergence that no longer
    exists.
    """

_LIVE_SPAN_COUNT_SQL = """
SELECT COUNT(*) FROM spans WHERE trace_id IN (
    SELECT trace_id FROM distillation_passes WHERE run_id = ?
    UNION SELECT ?
)
"""


def _loads_or_none(value: Any) -> Any:
    """Parse a stored JSON column for a JSON response, or None.

    The `/api/spans` `attributes` precedent: the server parses stored blobs
    server-side so the SPA never has to `JSON.parse` a field that may be
    NULL or malformed.
    """
    if not value:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


# §15's provenance recipes, verbatim and `[DR54]`-corrected. They live here as
# constants rather than inline strings because `fix-sb8.12` ships them as
# documentation and the documented text has to be the executed text.

_SUPPORT_RUNS_SQL = """
WITH cited AS (
  SELECT DISTINCT d.command_key, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at,
       d.divergence_id, d.kind, d.command_name, d.material
FROM distillation_runs r
JOIN distillation_divergences d ON d.run_id = r.run_id
JOIN cited ON cited.command_key = d.command_key AND cited.kind = d.kind
WHERE r.comparable = 1
  AND r.replay_of IS NULL      -- a replay tests an insight; it is not support for it
ORDER BY r.started_at DESC
"""

_CONTRADICT_RUNS_SQL = """
WITH cited AS (
  SELECT DISTINCT d.command_key, d.command_name, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at, cited.command_name
FROM distillation_runs r
CROSS JOIN cited            -- deliberate: every cited (command, kind) against every run
WHERE cited.command_name IS NOT NULL   -- run-level divergences have no command [DR54]
  AND r.comparable = 1
  AND r.replay_of IS NULL
  AND EXISTS (SELECT 1 FROM spans s
               WHERE s.trace_id = r.turn_key            -- [DR1]: the invariant holds
                 AND s.distillation_pass = 'student'
                 AND s.name = 'fw.command.execute'
                 AND s.command_name = cited.command_name)
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id
                     AND d2.command_key = cited.command_key
                     AND d2.kind = cited.kind)
ORDER BY r.started_at DESC
"""

_CONTRADICT_RUN_LEVEL_SQL = """
WITH cited AS (
  SELECT DISTINCT r0.entry_context, r0.workflow_name
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  JOIN distillation_runs r0 ON r0.run_id = d.run_id
  WHERE i.insight_id = :insight_id AND d.level = 'run'
)
SELECT r.run_id, r.turn_key, r.started_at, r.user_message
FROM distillation_runs r
JOIN cited ON cited.entry_context IS r.entry_context
          AND cited.workflow_name IS r.workflow_name
WHERE r.comparable = 1 AND r.replay_of IS NULL
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id AND d2.level = 'run')
ORDER BY r.started_at DESC
"""


_WEEKLY_RATE_SQL = """
SELECT strftime('%Y-W%W', r.started_at) AS week,
       COUNT(*)                                                       AS runs,
       SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END)           AS diverged,
       SUM(r.material_divergences)                                    AS material,
       ROUND(1.0 * SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END)
                 / COUNT(*), 3)                                       AS rate
FROM distillation_runs r
WHERE r.comparable = 1 AND r.replay_of IS NULL{scope}
GROUP BY week ORDER BY week
"""

_BY_COMMAND_SQL = """
SELECT d.command_name, COUNT(*) AS n,
       GROUP_CONCAT(DISTINCT r.run_id) AS run_ids
FROM distillation_divergences d
JOIN distillation_runs r ON r.run_id = d.run_id
WHERE d.kind = 'missing-in-student' AND d.material = 1
  AND r.comparable = 1 AND r.replay_of IS NULL{scope}
GROUP BY d.command_name ORDER BY n DESC
"""

_BY_KIND_SQL = """
SELECT d.level, d.kind, COUNT(*) AS n,
       SUM(CASE WHEN d.material = 1 THEN 1 ELSE 0 END) AS material,
       COUNT(DISTINCT d.run_id) AS runs
FROM distillation_divergences d
JOIN distillation_runs r ON r.run_id = d.run_id
WHERE r.comparable = 1 AND r.replay_of IS NULL{scope}
GROUP BY d.level, d.kind ORDER BY n DESC
"""

# The promotion view. Correlated subqueries rather than a join chain so an
# insight with no corroboration appears with a zero instead of vanishing, and
# `material_support_runs` counts RUNS rather than divergence rows — both
# corrections forced by `[DR54]`'s fixture.
_PROMOTION_SQL = """
SELECT i.insight_id, i.kind, i.text, i.run_id, i.created_at,
       (SELECT v.verdict FROM distillation_verdicts v
         WHERE v.insight_id = i.insight_id AND v.superseded = 0
         ORDER BY v.created_at DESC LIMIT 1)                     AS verdict,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL   -- run-level citations key on nothing
           AND r.comparable = 1
           AND r.replay_of IS NULL           -- a replay is a test, not support [DR54]
           AND r.isolation_verified = 1      -- promotion is a causal claim [DR48]
           AND r.evidence_pruned = 0)                            AS support_runs,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL AND sup.material = 1
           AND r.comparable = 1 AND r.replay_of IS NULL
           AND r.isolation_verified = 1 AND r.evidence_pruned = 0) AS material_support_runs
FROM distillation_insights i
ORDER BY support_runs DESC
"""

_COST_SQL = """
SELECT p.role,
       COUNT(*) AS passes, SUM(p.tokens) AS tokens,
       ROUND(SUM(p.cost_usd), 4) AS cost_usd,
       ROUND(AVG(p.wall_ms)) AS avg_ms,
       SUM(p.cache_hits) AS cache_hits, SUM(p.cache_misses) AS cache_misses
FROM distillation_passes p
JOIN distillation_runs r ON r.run_id = p.run_id
WHERE r.comparable = 1 AND r.cache_asymmetric = 0 AND p.role IN ('teacher','student')
GROUP BY p.role
"""


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
    for column, classification in _POLICED_TURN_COLUMNS:
        turn_row[column] = _policed_column(
            policy, column, classification, turn_row[column]
        )
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
        status TEXT NOT NULL, attributes TEXT NOT NULL,
        distillation_pass TEXT)""",
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
    # Distillation records (distillation design §9). No inline REFERENCES:
    # this file declares none and does not enable PRAGMA foreign_keys, so the
    # joins are by convention and the delete order in forget_channel is what
    # keeps them consistent [DR22][DR44].
    """CREATE TABLE IF NOT EXISTS distillation_runs (
        run_id TEXT PRIMARY KEY,
        turn_key TEXT NOT NULL,
        channel_id TEXT, conversation_id INTEGER,
        user_message TEXT NOT NULL,
        workflow_name TEXT, entry_context TEXT,
        comparable INTEGER NOT NULL,
        comparable_reason TEXT,
        isolation_verified INTEGER,
        fingerprint_teacher TEXT, fingerprint_student TEXT,
        restore_ok_pre_student INTEGER,
        restore_ok_post_compare INTEGER,
        cache_asymmetric INTEGER NOT NULL DEFAULT 0,
        left_steps INTEGER, right_steps INTEGER,
        planning_diverged INTEGER NOT NULL DEFAULT 0,
        exec_diverged INTEGER NOT NULL DEFAULT 0,
        material_divergences INTEGER NOT NULL DEFAULT 0,
        planning_insights INTEGER NOT NULL DEFAULT 0,
        execution_insights INTEGER NOT NULL DEFAULT 0,
        extractor_empty INTEGER NOT NULL DEFAULT 0,
        extractor_model TEXT,
        insight_set_json TEXT,
        replay_of TEXT,
        replay_trace_id TEXT,
        pinned INTEGER NOT NULL DEFAULT 0,
        pinned_at TEXT,
        pinned_span_count INTEGER,
        turn_fields_from TEXT,
        evidence_pruned INTEGER NOT NULL DEFAULT 0,
        started_at TEXT, completed_at TEXT,
        run_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS distillation_passes (
        run_id TEXT NOT NULL,
        pass_label TEXT NOT NULL,
        role TEXT NOT NULL,
        seq INTEGER NOT NULL,
        trace_id TEXT NOT NULL,
        agent_model TEXT, planner_model TEXT, model_params_json TEXT,
        entry_fingerprint TEXT, exit_fingerprint TEXT,
        first_span_id TEXT, last_span_id TEXT,
        wall_ms INTEGER, tokens INTEGER, cost_usd REAL,
        cache_hits INTEGER, cache_misses INTEGER,
        entry_prompt_fingerprint TEXT, exit_prompt_fingerprint TEXT,
        history_bound INTEGER,
        summary_hash TEXT,
        spans_dropped_delta INTEGER,
        entry_inputs_json TEXT,
        PRIMARY KEY (run_id, pass_label))""",
    """CREATE TABLE IF NOT EXISTS distillation_divergences (
        divergence_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        level TEXT NOT NULL,
        left_pass TEXT NOT NULL,
        right_pass TEXT NOT NULL,
        align_index INTEGER NOT NULL,
        kind TEXT NOT NULL,
        material INTEGER,
        replayable INTEGER NOT NULL DEFAULT 1,
        command_key TEXT, command_name TEXT, context TEXT,
        left_step_key TEXT, right_step_key TEXT,
        left_span_id TEXT, right_span_id TEXT,
        param_diff_json TEXT,
        detail_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS distillation_insights (
        insight_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        text_hash TEXT NOT NULL,
        extractor_span_id TEXT,
        insight_file TEXT,
        file_entry_number INTEGER,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS distillation_insight_citations (
        insight_id TEXT NOT NULL,
        divergence_id TEXT NOT NULL,
        PRIMARY KEY (insight_id, divergence_id))""",
    """CREATE TABLE IF NOT EXISTS distillation_verdicts (
        verdict_id TEXT PRIMARY KEY,
        insight_id TEXT NOT NULL,
        verdict TEXT NOT NULL,
        note TEXT,
        actor TEXT NOT NULL,
        replay_run_id TEXT,
        superseded INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL)""",
    # ------------------------------------------------------------------
    # The experiment container (`fix-bn1`, docs/experiment_container_design.md).
    # Additive, no REFERENCES (this file declares none and does not enable
    # PRAGMA foreign_keys), so the joins are by convention and the delete order
    # in forget_channel/clear_conversations is what keeps them consistent
    # [DR22][DR44].
    # ------------------------------------------------------------------
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
    # The unit of scoring `[XR13]`. An attempt's verdict is WRITTEN here, never
    # derived from turn columns at read time; `finished_at IS NULL` is the
    # completion marker the resume selector and the denominator check both read.
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
        finished_at TEXT,
        detail_json TEXT,
        PRIMARY KEY (experiment_id, task_id, attempt))""",
    # One row per evidence-run SEGMENT `[XR1]`. A child table rather than a JSON
    # array on `experiments`, because appending a segment to a policed column is
    # a read-modify-write and `[XR20]` forbids that; here each segment is an
    # independent INSERT and `valid` is a queryable column.
    """CREATE TABLE IF NOT EXISTS experiment_evidence_runs (
        experiment_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        evidence_run_id TEXT NOT NULL,
        valid INTEGER NOT NULL,
        started_at TEXT, completed_at TEXT,
        record_json TEXT NOT NULL,
        PRIMARY KEY (experiment_id, seq))""",
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(channel_id, conversation_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_key)",
    # Partial, following the idx_spans_command precedent: distillation costs
    # nothing on the spans that are not distillation.
    "CREATE INDEX IF NOT EXISTS idx_spans_trace_pass ON spans(trace_id, distillation_pass) WHERE distillation_pass IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_distill_runs_turn ON distillation_runs(turn_key)",
    "CREATE INDEX IF NOT EXISTS idx_distill_runs_channel ON distillation_runs(channel_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_distill_runs_replay ON distillation_runs(replay_of) WHERE replay_of IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_distill_runs_pinned ON distillation_runs(run_id) WHERE pinned = 1",
    "CREATE INDEX IF NOT EXISTS idx_distill_passes_run ON distillation_passes(run_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_distill_passes_trace ON distillation_passes(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_distill_div_run ON distillation_divergences(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_distill_div_kind ON distillation_divergences(kind, command_name)",
    "CREATE INDEX IF NOT EXISTS idx_distill_insights_run ON distillation_insights(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_distill_insights_hash ON distillation_insights(text_hash)",
    "CREATE INDEX IF NOT EXISTS idx_distill_citations_div ON distillation_insight_citations(divergence_id)",
    "CREATE INDEX IF NOT EXISTS idx_distill_verdicts_insight ON distillation_verdicts(insight_id, created_at)",
    # Partial, following idx_spans_command: an ordinary chatbot turn is not in
    # an experiment and must cost nothing.
    "CREATE INDEX IF NOT EXISTS idx_turns_experiment ON turns(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    # UNIQUE: this is what makes "one conversation per (experiment, task,
    # attempt)" an invariant rather than a hope, and what forces a resume to be
    # explicit about the crashed attempt's rows instead of quietly minting a
    # second conversation under the same three labels `[XR3]`.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_experiment_attempt ON conversations(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_baseline ON experiments(baseline_experiment_id) WHERE baseline_experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_attempts_channel ON experiment_attempts(channel_id)",
]


# Distillation record kinds -> (table, primary key columns, writable columns)
# [DR46]. The column tuples mirror the §9 DDL above; a payload key outside
# them is dropped by upsert_distillation_row rather than raising.
_DISTILL_RECORD_TABLES: dict[str, tuple[str, tuple[str, ...], frozenset[str]]] = {
    "run": (
        "distillation_runs",
        ("run_id",),
        frozenset(
            """run_id turn_key channel_id conversation_id user_message
            workflow_name entry_context comparable comparable_reason
            isolation_verified fingerprint_teacher fingerprint_student
            restore_ok_pre_student restore_ok_post_compare cache_asymmetric
            left_steps right_steps planning_diverged exec_diverged
            material_divergences planning_insights execution_insights
            extractor_empty extractor_model insight_set_json replay_of
            replay_trace_id pinned pinned_at pinned_span_count turn_fields_from
            evidence_pruned started_at completed_at run_json""".split()
        ),
    ),
    "pass": (
        "distillation_passes",
        ("run_id", "pass_label"),
        frozenset(
            """run_id pass_label role seq trace_id agent_model planner_model
            model_params_json entry_fingerprint exit_fingerprint first_span_id
            last_span_id wall_ms tokens cost_usd cache_hits cache_misses
            entry_prompt_fingerprint exit_prompt_fingerprint history_bound
            summary_hash spans_dropped_delta entry_inputs_json""".split()
        ),
    ),
    "divergence": (
        "distillation_divergences",
        ("divergence_id",),
        frozenset(
            """divergence_id run_id level left_pass right_pass align_index kind
            material replayable command_key command_name context left_step_key
            right_step_key left_span_id right_span_id param_diff_json
            detail_json""".split()
        ),
    ),
    "insight": (
        "distillation_insights",
        ("insight_id",),
        frozenset(
            """insight_id run_id kind text text_hash extractor_span_id
            insight_file file_entry_number created_at""".split()
        ),
    ),
    "citation": (
        "distillation_insight_citations",
        ("insight_id", "divergence_id"),
        frozenset({"insight_id", "divergence_id"}),
    ),
}

# The pinned trace set [DR25]. `pinned` lives on distillation_runs, never on
# distillation_passes: a per-pass pin can retain the student trace of an
# accepted insight while deleting the teacher trace it cites, which looks like
# data rather than a bug. At run granularity a pin is atomic by construction.
_PINNED_TRACES_SQL = (
    "SELECT p.trace_id FROM distillation_passes p "
    "JOIN distillation_runs r ON r.run_id = p.run_id WHERE r.pinned = 1"
)

# The tables the distillation records live in, newest-dependency first — the
# order forget_channel/clear_conversations delete in [DR44].
# The experiment container's tables, newest-dependency first -- the order
# clear_conversations deletes in `[XR15]`. `prune()` deliberately does NOT touch
# them: turns and conversations are already exempt from retention [R16], and
# pruning a container out from under attempts that still exist would produce
# exactly the orphan shape [DR44] prevents.
_EXPERIMENT_TABLES = (
    "experiment_evidence_runs",
    "experiment_attempts",
    "experiments",
)

_DISTILL_TABLES = (
    "distillation_verdicts",
    "distillation_insight_citations",
    "distillation_insights",
    "distillation_divergences",
    "distillation_passes",
    "distillation_runs",
)


class ObservabilityStore:
    """Schema owner + synchronous operations on one observability DB.

    Thread/process-safe by construction: every method opens its own
    short-lived WAL connection (timeout=30, ``BEGIN IMMEDIATE`` for writes).
    """

    def __init__(self, db_path: str, migrate: bool = True) -> None:
        self.db_path = db_path
        self._features: frozenset[str] = frozenset()
        if migrate:
            self._ensure_schema()
        self._features = self._load_features()

    @staticmethod
    def open_for_annotation(db_path: str) -> "ObservabilityStore":
        """A writable handle that will NOT migrate the file `[DR53]`.

        `_ensure_schema` is not a no-op on an existing DB: it runs every
        `_SCHEMA_STATEMENTS` entry and the `ALTER TABLE` block, stamps
        `PRAGMA user_version`, writes a `diagnostics` probe row and chmods the
        file and its parent. So constructing an ordinary store to append one
        verdict would mutate the post-mortem snapshot the viewer's read-only
        contract exists to protect — and would silently create the six
        distillation tables in a pre-distillation DB, which is the exact state
        `[DR29]` promises to degrade on rather than migrate.

        This handle opens read-write, detects features from what is already
        there, and touches nothing else. The caller is responsible for
        refusing when the feature marker is absent.

        Deliberately not a `classmethod`: `ReadOnlyObservabilityStore` inherits
        it, and "give me a writable handle" resolved through the read-only
        subclass is a contradiction — its `__init__` does not even take the
        flag. Naming the base class here makes that unreachable.
        """
        return ObservabilityStore(db_path, migrate=False)

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
            # BEFORE the statements, not after them like the conversations
            # migration below: idx_spans_trace_pass names distillation_pass,
            # so on an existing DB the CREATE INDEX fails with "no such
            # column" unless the ALTER has already run. On a fresh DB the
            # table does not exist yet, PRAGMA table_info returns nothing, and
            # the column arrives with the CREATE TABLE.
            #
            # Additive, and deliberately WITHOUT a version bump ([DR28]): the
            # premise of the migration comment below ("schema v1 was never
            # shipped") expired at v3.2.0, so two released builds now disagree
            # about the spans shape at the same user_version. The replacement
            # guarantee is the schema_features marker written further down,
            # which readers feature-detect instead of version-gating.
            span_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(spans)").fetchall()
            }
            if span_cols and "distillation_pass" not in span_cols:
                conn.execute("ALTER TABLE spans ADD COLUMN distillation_pass TEXT")
            # The experiment container's labels `[XR5]`, same shape and same
            # placement, and for the same reason: idx_turns_experiment and
            # idx_conv_experiment_attempt name these columns, so on an existing
            # DB the CREATE INDEX in the loop below fails "no such column"
            # unless the ALTER has already run. The `<table>_cols and` guard is
            # load-bearing in BOTH directions -- without it the ALTER runs
            # before the CREATE TABLE on a fresh DB and fails "no such table";
            # with it and without the columns in the CREATE TABLE literal, a
            # fresh DB never gets them at all.
            # Per COLUMN, not per table. DDL in Python's sqlite3 autocommits, so
            # the three ALTERs are three transactions: an interruption after the
            # first leaves a table with `experiment_id` and without the other
            # two, and a guard keyed on `experiment_id` alone would then skip
            # the block forever and never repair it. Guarding each column makes
            # the migration self-healing from any partial state.
            for table in ("turns", "conversations"):
                cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if not cols:
                    continue  # fresh DB: the CREATE TABLE literal supplies them
                for column, decl in (
                    ("experiment_id", "TEXT"),
                    ("task_id", "TEXT"),
                    ("attempt", "INTEGER"),
                ):
                    if column not in cols:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
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
            self._merge_schema_features(
                conn, [FEATURE_DISTILLATION_V1, FEATURE_EXPERIMENTS_V1]
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
    def _merge_schema_features(conn: sqlite3.Connection, features: list[str]) -> None:
        """Add *features* to diagnostics['schema_features'], merged not
        overwritten — another build's markers are not ours to drop [DR28]."""
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
        if merged == sorted(set(known)):
            return
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            ("schema_features", json.dumps(merged, ensure_ascii=False), _utcnow_iso()),
        )

    def _load_features(self) -> frozenset[str]:
        """The DB's feature set, read once at construction.

        The diagnostics marker is authoritative; the PRAGMA fallback covers a
        DB migrated by a build that added the column before the marker existed,
        and a read-only handle on a DB it may not migrate ([DR28]/[DR29]).
        """
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
            span_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(spans)").fetchall()
            }
            if "distillation_pass" in span_cols:
                detected.add(FEATURE_DISTILLATION_V1)
            turn_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "experiment_id" in turn_cols:
                detected.add(FEATURE_EXPERIMENTS_V1)
            return frozenset(detected)
        except Exception:
            # An unreadable/absent marker means "assume nothing"; every caller
            # of has_feature degrades rather than raising [DR29].
            return frozenset()
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def has_feature(self, name: str) -> bool:
        """Runtime feature detection, in place of a SCHEMA_VERSION bump [DR28].

        ``has_feature("distillation_v1")`` is False on a DB written by a
        pre-distillation build and never migrated (the read-only viewer's
        post-mortem-snapshot case), so a projection naming
        ``spans.distillation_pass`` or a distillation table can be skipped
        instead of raising ``no such column`` [DR29].
        """
        return name in self._features

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
                    command_name, context, start_ns, end_ns, status, attributes,
                    distillation_pass)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
                     end_ns=COALESCE(excluded.end_ns, spans.end_ns),
                     status=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                 THEN excluded.status ELSE spans.status END,
                     attributes=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                     THEN excluded.attributes ELSE spans.attributes END,
                     command_name=COALESCE(excluded.command_name, spans.command_name),
                     context=COALESCE(excluded.context, spans.context)""",
                # distillation_pass is deliberately NOT in the DO UPDATE set:
                # the label is a fact about where the span was opened, so
                # write-once at open is the correct semantics and a close
                # emitted outside the pass cannot relabel it.
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
                    span.distillation_pass,
                ),
            )

    def upsert_distillation_row(
        self,
        conn: sqlite3.Connection,
        kind: str,
        payload: dict[str, Any],
        redactor: Redactor,
    ) -> None:
        """Write one distillation row ([DR46]); upsert on the table's PK.

        ``kind`` selects the table (``run`` | ``pass`` | ``divergence`` |
        ``insight`` | ``citation``); ``payload`` is a flat column->value map.
        Unknown columns are dropped rather than raising, so a producer built
        against a later column set degrades to a partial row on an older
        build instead of failing a turn. A run row is upsertable so it can be
        written at start (``comparable``, ``user_message``) and completed
        later; verdicts are not written here — that route is one of [DR46]'s
        two off-turn-thread exemptions.

        ``pinned_span_count = COUNT_LIVE_SPANS`` asks the writer to resolve the
        column from the table rather than carrying a value the producer read
        itself (`fix-sb8.18`).

        A ``citation`` whose divergence row is not in the table raises
        `OrphanedCitation` rather than writing an orphan (`fix-sb8.16`).
        """
        spec = _DISTILL_RECORD_TABLES.get(kind)
        if spec is None:
            raise ValueError(f"unknown distillation record kind {kind!r}")
        table, key_cols, columns = spec
        if payload.get("pinned_span_count") == COUNT_LIVE_SPANS:
            # Resolved here, on the writer's connection, rather than by the
            # producer on the turn thread (`fix-sb8.18`). The pass rows are
            # enqueued ahead of this row and the record queue is FIFO, so they
            # are already applied; `turn_key` is unioned in so a run whose pass
            # rows were lost still counts its own trace. Spans in THIS batch
            # are applied after the records, which keeps the count a slight
            # under-count — the same conservative direction as before, so
            # `distillation_evidence_shortfall` can never invent a loss.
            payload = dict(payload)
            payload["pinned_span_count"] = conn.execute(
                _LIVE_SPAN_COUNT_SQL,
                (payload.get("run_id"), payload.get("turn_key")),
            ).fetchone()[0]
        row = {
            name: (redactor.redact(value) if isinstance(value, str) else value)
            for name, value in payload.items()
            if name in columns
        }
        missing = [name for name in key_cols if row.get(name) in (None, "")]
        if missing:
            raise ValueError(f"{table} record is missing key column(s) {missing}")
        names = list(row)
        placeholders = ", ".join("?" for _ in names)
        updatable = [name for name in names if name not in key_cols]
        if updatable:
            assignment = ", ".join(f"{name}=excluded.{name}" for name in updatable)
        else:
            # A pure key row (citations): the second write is a no-op, not a
            # constraint violation.
            assignment = None
        conflict = ", ".join(key_cols)
        if kind == "citation":
            # `fix-sb8.16`: a citation is only written once the divergence row
            # it names is IN THE TABLE. Divergence records are enqueued ahead
            # of the citations drawn from them and the record queue is FIFO, so
            # by the time this runs the row is either present or it was lost —
            # and a citation written anyway is an orphan that §15's provenance
            # recipes read as real evidence. The guard lives here, on the
            # writer's connection inside the batch transaction, so the ordering
            # costs the turn thread neither a barrier nor a read.
            written = conn.execute(
                f"INSERT INTO {table} ({', '.join(names)}) "
                f"SELECT {placeholders} WHERE EXISTS (SELECT 1 FROM "
                "distillation_divergences WHERE divergence_id=?) "
                f"ON CONFLICT({conflict}) DO NOTHING",
                tuple(row[name] for name in names) + (row["divergence_id"],),
            ).rowcount
            if not written:
                # Distinguishable from the ON CONFLICT no-op by asking the
                # table, so a re-emitted citation is not reported as a loss.
                orphan = conn.execute(
                    f"SELECT 1 FROM {table} WHERE insight_id=? AND divergence_id=?",
                    (row["insight_id"], row["divergence_id"]),
                ).fetchone()
                if orphan is None:
                    raise OrphanedCitation(
                        f"divergence {row['divergence_id']!r} is not in the "
                        f"table; citation from insight {row['insight_id']!r} "
                        "suppressed"
                    )
            return
        sql = (
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders}) "
            + (
                f"ON CONFLICT({conflict}) DO UPDATE SET {assignment}"
                if assignment
                else f"ON CONFLICT({conflict}) DO NOTHING"
            )
        )
        conn.execute(sql, tuple(row[name] for name in names))

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

    def get_spans(
        self, trace_id: str, distillation_pass: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Every span of a trace, or only one pass's ([DR1], [DR6], [DR7]).

        `distillation_pass` is what turns the shared `trace_id == turn_key`
        into two independently viewable waterfalls: the passes share one trace
        and are told apart by the column, so filtering here is the whole of
        "neither pass interleaves the other". `'none'` selects the spans
        belonging to no pass (the turn wrapper and anything outside both), which
        is what the UI's third tab needs. On a pre-distillation DB the column
        does not exist, so the filter is dropped rather than raising [DR29].
        """
        query = "SELECT * FROM spans WHERE trace_id=?"
        params: list[Any] = [trace_id]
        if distillation_pass is not None and self.has_feature(FEATURE_DISTILLATION_V1):
            if distillation_pass == "none":
                query += " AND distillation_pass IS NULL"
            else:
                query += " AND distillation_pass=?"
                params.append(distillation_pass)
        query += " ORDER BY start_ns"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
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

    # -- distillation read layer (`fix-sb8.6`, §12.1 [DR55]) -------------
    #
    # Every method degrades to an empty result on a DB written by a
    # pre-distillation build [DR29], so the viewer opening a post-mortem
    # snapshot gets an empty view rather than "no such table".

    _RUN_LIST_COLUMNS = (
        "run_id, turn_key, channel_id, conversation_id, user_message, "
        "workflow_name, entry_context, started_at, completed_at, comparable, "
        "comparable_reason, isolation_verified, cache_asymmetric, "
        "planning_diverged, exec_diverged, material_divergences, "
        "planning_insights, execution_insights, extractor_empty, replay_of, "
        "pinned, evidence_pruned"
    )

    def _distillation_ready(self) -> bool:
        return self.has_feature(FEATURE_DISTILLATION_V1)

    def list_distillation_runs(
        self,
        channel_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
        comparable: Optional[bool] = None,
        diverged: Optional[bool] = None,
        include_replays: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """The run list, newest first (§12.1 row 1).

        Replays are excluded by default for the same reason every §15 recipe
        carries ``replay_of IS NULL``: a replay is a TEST of an insight, not
        independent evidence for it, and listing the two together invites
        exactly the double-count `[DR54]` caught in the promotion query.
        """
        if not self._distillation_ready():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if channel_id is not None:
            clauses.append("channel_id=?")
            params.append(channel_id)
        if conversation_id is not None:
            clauses.append("conversation_id=?")
            params.append(conversation_id)
        if comparable is not None:
            clauses.append("comparable=?")
            params.append(1 if comparable else 0)
        if diverged is not None:
            clauses.append(
                "(planning_diverged=1 OR exec_diverged=1)"
                if diverged
                else "(planning_diverged=0 AND exec_diverged=0)"
            )
        if not include_replays:
            clauses.append("replay_of IS NULL")
        if experiment_id is not None:
            # The experiment -> distillation cross-link (`[XR9]`). A distillation
            # sweep run INSIDE an experiment must be reachable from both
            # directions without a second "run" concept in the URL space, and
            # extending this shipped route is what keeps `run` distillation's
            # noun. Degrades to no rows on a DB with no experiment columns.
            if not self.has_feature(FEATURE_EXPERIMENTS_V1):
                return []
            clauses.append(
                "turn_key IN (SELECT turn_key FROM turns WHERE experiment_id=?)"
            )
            params.append(experiment_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT {self._RUN_LIST_COLUMNS} FROM distillation_runs{where} "
            "ORDER BY COALESCE(started_at, turn_key) DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def experiment_labels_for_turn(self, turn_key: str) -> Optional[dict[str, Any]]:
        """The experiment an existing turn belongs to, or None (`[XR9]`).

        The distillation -> experiment direction of the cross-link: one extra
        statement on a detail route, the shape `[DR55]` used for `retention`.
        """
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

    def get_distillation_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """One run plus its passes in execution order (§12.1 row 2).

        ``run_json`` is parsed into ``record``, mirroring how
        ``/api/turn/<k>`` treats ``record_json`` — the SPA reads one shape for
        both, and a malformed blob degrades to None rather than a 500.
        """
        if not self._distillation_ready():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM distillation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            run = dict(row)
            passes = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM distillation_passes WHERE run_id=? ORDER BY seq",
                    (run_id,),
                ).fetchall()
            ]
        try:
            run["record"] = json.loads(run.pop("run_json"))
        except (ValueError, TypeError, KeyError):
            run["record"] = None
        for pass_row in passes:
            pass_row["model_params"] = _loads_or_none(pass_row.get("model_params_json"))
            pass_row["entry_inputs"] = _loads_or_none(pass_row.get("entry_inputs_json"))
        return {"run": run, "passes": passes}

    def list_distillation_divergences(
        self,
        run_id: str,
        kind: Optional[str] = None,
        material: Optional[str] = None,
        level: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """A run's aligned divergence rows, in alignment order (§12.1 row 3).

        ``material`` takes ``'1'``, ``'0'`` or ``'null'`` because the column is
        three-valued by design ([DR20]: NULL means the run was not comparable,
        so materiality was never computed) and a bool cannot say that.
        """
        if not self._distillation_ready():
            return []
        clauses = ["run_id=?"]
        params: list[Any] = [run_id]
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if level is not None:
            clauses.append("level=?")
            params.append(level)
        if material is not None:
            if material == "null":
                clauses.append("material IS NULL")
            else:
                clauses.append("material=?")
                params.append(1 if material in ("1", "true", "True") else 0)
        query = (
            "SELECT * FROM distillation_divergences WHERE "
            f"{' AND '.join(clauses)} ORDER BY level, align_index"
        )
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        for row in rows:
            row["param_diff"] = _loads_or_none(row.get("param_diff_json"))
            row["detail"] = _loads_or_none(row.get("detail_json"))
        return rows

    def distillation_insights(
        self,
        run_id: Optional[str] = None,
        insight_id: Optional[str] = None,
        text_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """§13.2's provenance closure, in whichever direction was asked for.

        Exactly one selector, and all three resolve to the same shape:
        insights, the citations that bind them to divergence rows, and every
        verdict recorded against them. ``insight_id`` additionally returns the
        §15 support and contradiction run lists, which is the reverse
        direction — from a rule back to the turns that argue for and against
        it — and the whole point of acceptance criterion 7.
        """
        empty: dict[str, Any] = {"insights": [], "citations": [], "verdicts": []}
        if not self._distillation_ready():
            return empty
        if insight_id is not None:
            where, params = "i.insight_id=?", [insight_id]
        elif run_id is not None:
            where, params = "i.run_id=?", [run_id]
        elif text_hash is not None:
            where, params = "i.text_hash=?", [text_hash]
        else:
            return empty
        with self._connect() as conn:
            insights = [
                dict(r)
                for r in conn.execute(
                    f"SELECT i.* FROM distillation_insights i WHERE {where} "
                    "ORDER BY i.created_at, i.insight_id",
                    params,
                ).fetchall()
            ]
            if not insights:
                return empty
            ids = [row["insight_id"] for row in insights]
            citations: list[dict[str, Any]] = []
            verdicts: list[dict[str, Any]] = []
            for chunk in _chunked(ids):
                marks = _in_placeholders(len(chunk))
                citations.extend(
                    dict(r)
                    for r in conn.execute(
                        "SELECT c.insight_id, c.divergence_id, d.run_id, d.level, "
                        "d.kind, d.material, d.command_name, d.align_index, "
                        "d.left_span_id, d.right_span_id "
                        "FROM distillation_insight_citations c "
                        "LEFT JOIN distillation_divergences d "
                        "  ON d.divergence_id = c.divergence_id "
                        f"WHERE c.insight_id IN ({marks}) "
                        "ORDER BY c.insight_id, d.align_index",
                        chunk,
                    ).fetchall()
                )
                verdicts.extend(
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM distillation_verdicts WHERE insight_id IN "
                        f"({marks}) ORDER BY created_at DESC",
                        chunk,
                    ).fetchall()
                )
            result: dict[str, Any] = {
                "insights": insights,
                "citations": citations,
                "verdicts": verdicts,
            }
            if insight_id is not None:
                result["runs"] = {
                    "support": [
                        dict(r)
                        for r in conn.execute(
                            _SUPPORT_RUNS_SQL, {"insight_id": insight_id}
                        ).fetchall()
                    ],
                    "contradict": [
                        dict(r)
                        for r in conn.execute(
                            _CONTRADICT_RUNS_SQL, {"insight_id": insight_id}
                        ).fetchall()
                    ],
                    "contradict_run_level": [
                        dict(r)
                        for r in conn.execute(
                            _CONTRADICT_RUN_LEVEL_SQL, {"insight_id": insight_id}
                        ).fetchall()
                    ],
                }
        return result

    # -- adjudication verdicts (`fix-sb8.9`, §12 [DR30]) -----------------

    VERDICTS = (
        "supported",
        "not-supported-by-cited-evidence",
        "overfit-to-single-turn",
        "duplicate-of-existing",
        "contradicted-by-other-turns",
    )

    def insert_verdict(
        self,
        insight_id: str,
        verdict: str,
        actor: str,
        note: Optional[str] = None,
        replay_run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append one adjudication verdict, superseding the previous one.

        §12's normative constraints, all of them enforced here rather than at
        the route so the CLI, the replay path and HTTP cannot diverge:

        1. Touches ONLY `distillation_verdicts`. It cannot alter, delete or
           rewrite a span, turn, artifact, run, pass, divergence, insight or
           citation row — which is the property `[R12]` actually protects.
        2. Append-only WITH SUPERSEDE, in one transaction: the history of
           judgements is itself evidence. An insight accepted, then rejected
           after a replay, is precisely the signal `fix-sb8.11` exists to
           produce, and an in-place UPDATE would erase it.
        3. `verdict` is validated against the closed §9 enum and `actor`
           against `human` / `agent:<name>` / `replay`; `note` is capped at
           4 KiB.

        The consequential unpin of a rejected run is NOT written here: §12
        rule 1 forbids it, and `prune()`'s sweep derives it ([DR52]).

        Raises `ValueError` on a rejected input or an unknown insight.
        """
        if verdict not in self.VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}")
        if not (
            actor == "human"
            or actor == "replay"
            or (actor.startswith("agent:") and len(actor) > len("agent:"))
        ):
            raise ValueError(
                f"actor must be 'human', 'replay' or 'agent:<name>', not {actor!r}"
            )
        if note is not None and len(note.encode("utf-8")) > _VERDICT_NOTE_MAX_BYTES:
            raise ValueError("note exceeds 4 KiB")
        if not self.has_feature(FEATURE_DISTILLATION_V1):
            raise ValueError("this database predates distillation recording")
        row = {
            "verdict_id": f"vd-{uuid.uuid4().hex[:12]}",
            "insight_id": insight_id,
            "verdict": verdict,
            "note": note,
            "actor": actor,
            "replay_run_id": replay_run_id,
            "superseded": 0,
            "created_at": _utcnow_iso(),
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            known = conn.execute(
                "SELECT 1 FROM distillation_insights WHERE insight_id=?",
                (insight_id,),
            ).fetchone()
            if known is None:
                conn.rollback()
                raise ValueError(f"unknown insight {insight_id!r}")
            conn.execute(
                "UPDATE distillation_verdicts SET superseded=1 "
                "WHERE insight_id=? AND superseded=0",
                (insight_id,),
            )
            conn.execute(
                "INSERT INTO distillation_verdicts (verdict_id, insight_id, "
                "verdict, note, actor, replay_run_id, superseded, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["verdict_id"],
                    row["insight_id"],
                    row["verdict"],
                    row["note"],
                    row["actor"],
                    row["replay_run_id"],
                    0,
                    row["created_at"],
                ),
            )
            conn.commit()
        return row

    def distillation_retention_status(self, run_id: str) -> dict[str, Any]:
        """Why this run's evidence is still here, or why it is not (`fix-sb8.13`).

        §10.3's pin classes are computed, never stored as a label: `pinned` is
        one bit and the reason it is set is spread across divergence flags,
        insight rows and verdicts. A pin nobody can explain is a pin nobody
        trusts, and the operator's actual question — "this trace is three
        months old, why do I still have it, and when does it go?" — has no
        answer anywhere in the schema. This composes one.

        Returns the pin class, the release condition, and the `[DR43]`
        shortfall so a run whose evidence an older build already pruned says
        so rather than rendering an empty diff.
        """
        if not self._distillation_ready():
            return {"run_id": run_id, "known": False}
        negative_pin_days = _env_int(
            "FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", _DEFAULT_DISTILL_NEGATIVE_PIN_DAYS
        )
        retention_days = _env_int("FW_OBS_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pinned, pinned_at, started_at, planning_diverged, "
                "exec_diverged, comparable, comparable_reason, evidence_pruned "
                "FROM distillation_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                return {"run_id": run_id, "known": False}
            insights = conn.execute(
                "SELECT COUNT(*) FROM distillation_insights WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            # The live verdict per insight, which is what decides whether the
            # rejected-only arm will release this run on its next sweep.
            live_verdicts = [
                r[0]
                for r in conn.execute(
                    "SELECT COALESCE((SELECT v.verdict FROM distillation_verdicts v "
                    "  WHERE v.insight_id = i.insight_id AND v.superseded = 0 "
                    "  ORDER BY v.created_at DESC LIMIT 1), 'unadjudicated') "
                    "FROM distillation_insights i WHERE i.run_id=?",
                    (run_id,),
                ).fetchall()
            ]
        diverged = bool(row["planning_diverged"] or row["exec_diverged"])
        open_verdicts = [
            verdict
            for verdict in live_verdicts
            if verdict in ("unadjudicated", "supported")
        ]
        if not row["comparable"]:
            pin_class, release = "non-comparable", "not pinned: " + (
                row["comparable_reason"] or "the passes are not comparable"
            )
        elif insights and open_verdicts:
            pin_class = "produced-an-insight"
            release = (
                "pinned until every insight on this run carries a rejecting "
                "verdict; adjudication has not concluded"
            )
        elif insights:
            pin_class = "rejected-only"
            release = (
                "every insight on this run is rejected; the next prune "
                "releases the pin and the evidence goes at the "
                f"{retention_days}-day horizon"
            )
        elif not diverged:
            pin_class = "no-divergence"
            release = (
                f"pinned as the contradiction set for {negative_pin_days} days "
                "from the pin, then released"
            )
        else:
            pin_class = "diverged-no-insight"
            release = (
                "not pinned: nothing cites this run's evidence and it is not "
                "the contradiction pool"
            )
        expected_pin = pin_class in ("produced-an-insight", "no-divergence")
        if expected_pin and not row["pinned"]:
            # The class says "keep this" and the bit says otherwise: the row
            # was written by a build without §10.3's pin predicate, so the
            # evidence is on the ordinary horizon and nobody was told. Worth
            # saying out loud — it is the same class of silent loss `[DR43]`'s
            # `pinned_span_count` exists to catch after the fact.
            release = (
                f"NOT PINNED despite falling in the {pin_class} class — this "
                "run was recorded by a build without the retention pin, so "
                f"its evidence goes at the {retention_days}-day horizon"
            )
        status = {
            "run_id": run_id,
            "known": True,
            "pinned": bool(row["pinned"]),
            "pinned_at": row["pinned_at"],
            "pin_class": pin_class,
            "pin_expected": expected_pin,
            "release": release,
            "insights": int(insights),
            "open_verdicts": len(open_verdicts),
            "negative_pin_days": negative_pin_days,
            "retention_days": retention_days,
        }
        shortfall = self.distillation_evidence_shortfall(run_id)
        if shortfall is not None:
            status["evidence"] = shortfall
        return status

    def distillation_corpus(
        self, channel_id: Optional[str] = None
    ) -> dict[str, Any]:
        """§15's aggregates: one turn is an anecdote (`fix-sb8.10`).

        Five views, each a §15 recipe executed verbatim so the UI and a
        scripting agent read the same numbers:

        * `weekly` — divergence rate over time, the signal for whether the
          student improves as insights accumulate.
        * `by_command` — material `missing-in-student` divergences by command.
        * `by_kind` — the same denominator broken out by taxonomy kind.
        * `promotion` — per-insight support and contradiction counts. This is
          the ONE view carrying `isolation_verified = 1` ([DR48]), so until
          `fix-35m.3` lands it is legitimately empty; `promotion_blocked` says
          so rather than leaving a reader to read zero rows as "no support".
        * `cost` — teacher vs student tokens, cost and latency, excluding
          cache-asymmetric runs because that is precisely the confound that
          makes the cost columns incomparable ([DR16]).
        """
        empty = {
            "weekly": [],
            "by_command": [],
            "by_kind": [],
            "promotion": [],
            "promotion_blocked": True,
            "cost": [],
        }
        if not self._distillation_ready():
            return empty
        scope = " AND r.channel_id = :channel" if channel_id else ""
        params = {"channel": channel_id} if channel_id else {}
        with self._connect() as conn:
            weekly = [
                dict(r)
                for r in conn.execute(
                    _WEEKLY_RATE_SQL.replace("{scope}", scope), params
                ).fetchall()
            ]
            by_command = [
                dict(r)
                for r in conn.execute(
                    _BY_COMMAND_SQL.replace("{scope}", scope), params
                ).fetchall()
            ]
            by_kind = [
                dict(r)
                for r in conn.execute(
                    _BY_KIND_SQL.replace("{scope}", scope), params
                ).fetchall()
            ]
            promotion = [
                dict(r) for r in conn.execute(_PROMOTION_SQL).fetchall()
            ]
            cost = [dict(r) for r in conn.execute(_COST_SQL).fetchall()]
            isolation_verified = conn.execute(
                "SELECT COUNT(*) FROM distillation_runs WHERE isolation_verified = 1"
            ).fetchone()[0]
        return {
            "weekly": weekly,
            "by_command": by_command,
            "by_kind": by_kind,
            "promotion": promotion,
            # Not "there is no support" — "the causal claim promotion rests on
            # is not yet checkable", which is a different thing to render.
            "promotion_blocked": not isolation_verified,
            "cost": cost,
        }

    def export_distillation_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """One run as a self-contained JSON document (`fix-sb8.12`).

        Built entirely out of STORED ROWS, never out of a live object: the
        `Redactor` scrubs secrets at the sink boundary ([R20]), so an export
        assembled from memory would route around redaction and put credentials
        in a file whose whole purpose is to be handed to an extraction agent.

        Carries both passes' spans, so an agent working offline from the file
        can reach the evidence a divergence row cites without the DB.
        """
        if not self._distillation_ready():
            return None
        detail = self.get_distillation_run(run_id)
        if detail is None:
            return None
        run = detail["run"]
        provenance = self.distillation_insights(run_id=run_id)
        traces = sorted(
            {p["trace_id"] for p in detail["passes"] if p.get("trace_id")}
            | ({run["turn_key"]} if run.get("turn_key") else set())
        )
        spans: dict[str, list[dict[str, Any]]] = {}
        for trace_id in traces:
            for span in self.get_spans(trace_id):
                span["attributes"] = _loads_or_none(span.get("attributes"))
                spans.setdefault(trace_id, []).append(span)
        return {
            "export_version": 1,
            "exported_at": _utcnow_iso(),
            "run": run,
            "passes": detail["passes"],
            "divergences": self.list_distillation_divergences(run_id),
            "insights": provenance["insights"],
            "citations": provenance["citations"],
            "verdicts": provenance["verdicts"],
            "retention": self.distillation_retention_status(run_id),
            "spans": spans,
        }

    def distillation_run_for_turn(self, turn_key: str) -> Optional[str]:
        """The run id recorded against a user-visible turn, if any.

        What lets the turn list mark a distillation turn without a second
        round trip per row.
        """
        if not self._distillation_ready():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM distillation_runs WHERE turn_key=? "
                "AND replay_of IS NULL ORDER BY started_at LIMIT 1",
                (turn_key,),
            ).fetchone()
            return row[0] if row is not None else None

    def distillation_turn_markers(
        self, turn_keys: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Run markers for a page of turn rows, keyed by turn_key.

        One statement for the whole page rather than one per row: the turn list
        renders 100 rows by default and a per-row query would be 100 opens of
        the read-only handle.
        """
        if not self._distillation_ready() or not turn_keys:
            return {}
        markers: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            for chunk in _chunked(list(turn_keys)):
                marks = _in_placeholders(len(chunk))
                for row in conn.execute(
                    "SELECT run_id, turn_key, comparable, comparable_reason, "
                    "planning_diverged, exec_diverged, material_divergences, "
                    "planning_insights, execution_insights, extractor_empty, "
                    "pinned, evidence_pruned FROM distillation_runs "
                    f"WHERE replay_of IS NULL AND turn_key IN ({marks})",
                    chunk,
                ).fetchall():
                    markers[row["turn_key"]] = dict(row)
        return markers

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

    _EXPERIMENT_STATUSES = frozenset({"running", "complete", "invalid"})
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
        if capture_profile is None or capture_policy_version is None:
            policy = self._store_capture_policy()
            capture_profile = capture_profile or policy.profile
            capture_policy_version = (
                capture_policy_version or policy.policy_version
            )
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
    ) -> None:
        """Open an attempt row before its first turn.

        The row's existence is not the completion marker -- `finished_at` is
        (`[XR13]`). An attempt that crashed halfway has rows and an open marker,
        which is what makes it visible to the resume selector and fatal to a
        `complete` verdict.
        """
        if not task_id:
            raise ValueError("task_id is required")
        attempt = int(attempt)
        if attempt <= 0:
            raise ValueError("attempt must be a positive integer")
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
            conn.execute(
                """INSERT INTO experiment_attempts
                   (experiment_id, task_id, attempt, channel_id, conversation_id,
                    outcome, outcome_source, reward, restarts, started_at,
                    finished_at, detail_json)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?, NULL, NULL)
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
        if outcome not in self._ATTEMPT_OUTCOMES:
            raise ValueError(
                f"outcome {outcome!r} is not one of {sorted(self._ATTEMPT_OUTCOMES)}"
            )
        if not outcome_source:
            raise ValueError(
                "outcome_source is required: an unattributed verdict cannot be "
                "told apart from a fallback"
            )
        self._update_experiment(
            """UPDATE experiment_attempts
                  SET outcome=?, outcome_source=?, reward=?, finished_at=?,
                      detail_json=?,
                      conversation_id=COALESCE(?, conversation_id)
                WHERE experiment_id=? AND task_id=? AND attempt=?""",
            (
                outcome,
                self._scrub(outcome_source),
                None if reward is None else float(reward),
                _utcnow_iso(),
                None
                if detail is None
                else self._scrub(
                    json.dumps(_sanitize_json_value(detail), ensure_ascii=False)
                ),
                conversation_id,
                experiment_id,
                self._scrub(task_id),
                int(attempt),
            ),
            experiment_id,
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
                """SELECT 1 FROM experiment_attempts
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
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
                self._delete_derived_trace_spans(conn, turn_keys)
            # The distillation closure of those turns, before the turns
            # themselves ([DR44]: nothing cascades, and the id sets must be
            # collected before the parent rows go). Without this, a restarted
            # attempt leaves distillation_runs rows pointing at turn_keys that
            # no longer exist, which read as real evidence.
            if turn_keys:
                run_ids: list[str] = []
                for chunk in _chunked(turn_keys):
                    marks = ", ".join("?" for _ in chunk)
                    run_ids.extend(
                        r[0]
                        for r in conn.execute(
                            f"SELECT run_id FROM distillation_runs "
                            f"WHERE turn_key IN ({marks})",
                            chunk,
                        ).fetchall()
                    )
                if run_ids:
                    self._delete_distillation_runs(conn, run_ids)
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
                          reward=NULL, finished_at=NULL, detail_json=NULL,
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

            reason: Optional[str] = force_invalid
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
                         SUM(CASE WHEN finished_at IS NOT NULL AND outcome IS NOT NULL
                                  THEN 1 ELSE 0 END) AS finished,
                         SUM(CASE WHEN outcome='incomplete' THEN 1 ELSE 0 END)
                              AS incomplete
                       FROM experiment_attempts WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchone()
                rows_total = int(counts["rows_total"] or 0)
                tasks_total = int(counts["tasks_total"] or 0)
                finished = int(counts["finished"] or 0)
                incomplete = int(counts["incomplete"] or 0)
                declared_tasks = int(row["declared_tasks"])
                bad_segments = conn.execute(
                    """SELECT COUNT(*) FROM experiment_evidence_runs
                        WHERE experiment_id=? AND valid=0""",
                    (experiment_id,),
                ).fetchone()[0]
                if (
                    finished != expected
                    or rows_total != expected
                    or tasks_total != declared_tasks
                    or incomplete
                ):
                    reason = "attempt_shortfall"
                    detail = (
                        f"{finished} finished and {rows_total} recorded of "
                        f"{expected} declared attempts across {tasks_total} of "
                        f"{declared_tasks} declared tasks; {incomplete} incomplete"
                    )
                elif bad_segments:
                    reason = "evidence_run_invalid"
                    detail = f"{bad_segments} evidence segment(s) reported invalid"

            status = "invalid" if reason is not None else "complete"
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
                    _utcnow_iso(),
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
        negative_pin_days: Optional[int] = None,
        pin_max_fraction: Optional[float] = None,
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
        Distillation (design §10): a pinned run's traces are exempt from both
        arms, the predicate riding INSIDE each victim-selection subquery
        ([DR52] — on the outer DELETE, the ``rowcount == 0`` break below turns
        one all-pinned batch into "stop evicting altogether"); the six
        distillation tables are themselves under retention, an unpinned run
        past the horizon losing its divergences and its ``entry_inputs_json``
        while its conclusions survive marked ``evidence_pruned``; and the
        bounded pin-release sweep runs first, so a run released this pass is
        prunable this pass ([DR52]). The distillation counters join the result
        dict only when non-zero, so the historical two-key result is what a
        workflow that never distills still gets.
        """
        if pruning_suppressed():
            return {"suppressed": 1}
        if retention_days is None:
            retention_days = _env_int("FW_OBS_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
        if max_bytes is None:
            max_bytes = _env_int("FW_OBS_DB_MAX_BYTES", _DEFAULT_DB_MAX_BYTES)
        if negative_pin_days is None:
            negative_pin_days = _env_int(
                "FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", _DEFAULT_DISTILL_NEGATIVE_PIN_DAYS
            )
        if pin_max_fraction is None:
            pin_max_fraction = _env_float(
                "FW_OBS_DISTILL_PIN_MAX_FRACTION", _DEFAULT_DISTILL_PIN_MAX_FRACTION
            )

        horizon_ns = int(
            (time.time() - retention_days * 86_400) * 1_000_000_000
        )
        horizon_dt = datetime.fromtimestamp(
            max(0.0, time.time() - retention_days * 86_400), tz=timezone.utc
        )
        horizon_key = horizon_dt.strftime("%Y%m%dT%H%M%S")
        horizon_iso = horizon_dt.isoformat()
        deleted = {"spans": 0, "artifacts": 0}

        with self._connect() as conn:
            _merge_nonzero(
                deleted, self._release_distillation_pins(conn, negative_pin_days)
            )
            for _ in range(_PRUNE_MAX_BATCHES):
                conn.execute("BEGIN IMMEDIATE")
                spans_cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans WHERE start_ns < ? "
                    f"AND trace_id NOT IN ({_PINNED_TRACES_SQL}) LIMIT ?)",
                    (horizon_ns, _PRUNE_BATCH_ROWS),
                )
                deleted["spans"] += spans_cur.rowcount
                artifacts_cur = conn.execute(
                    "DELETE FROM artifacts WHERE artifact_id IN "
                    "(SELECT artifact_id FROM artifacts WHERE turn_key < ? "
                    f"AND turn_key NOT IN ({_PINNED_TRACES_SQL}) LIMIT ?)",
                    (horizon_key, _PRUNE_BATCH_ROWS),
                )
                deleted["artifacts"] += artifacts_cur.rowcount
                conn.commit()
                if (
                    spans_cur.rowcount < _PRUNE_BATCH_ROWS
                    and artifacts_cur.rowcount < _PRUNE_BATCH_ROWS
                ):
                    break

            _merge_nonzero(
                deleted,
                self._prune_distillation_evidence(conn, horizon_key, horizon_iso),
            )

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
                    run_ids = []
                    # A replay trace goes with the turn it replays (§3.5 row 4):
                    # its trace_id is '<turn_key>~replay.<n>'. Hoisted out of
                    # the per-key loop for the same reason as in
                    # forget_channel: the per-key LIKE is a full scan of
                    # `spans`, and this batch holds up to _PRUNE_BATCH_ROWS keys.
                    self._delete_derived_trace_spans(conn, keys)
                    for key in keys:
                        conn.execute("DELETE FROM feedback WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM spans WHERE trace_id=?", (key,))
                        conn.execute("DELETE FROM artifacts WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM turns WHERE turn_key=?", (key,))
                        run_ids.extend(
                            r[0]
                            for r in conn.execute(
                                "SELECT run_id FROM distillation_runs WHERE turn_key=?",
                                (key,),
                            ).fetchall()
                        )
                    # The whole turn is being erased on operator request, so
                    # its distillation closure goes with it rather than
                    # surviving as a run row pointing at a turn that is gone.
                    # No pin exemption here: this arm is opt-in and total.
                    self._delete_distillation_runs(conn, run_ids)
                    conn.commit()
                    deleted["conversationless_turns"] += len(keys)
                    if len(keys) < _PRUNE_BATCH_ROWS:
                        break

            # Size-cap eviction, oldest first and TRACE-ATOMICALLY [DR27].
            _merge_nonzero(deleted, self._evict_oldest_traces(conn, max_bytes))

            _merge_nonzero(
                deleted, self._enforce_pin_ceiling(conn, max_bytes, pin_max_fraction)
            )

            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted


    # -- distillation retention [DR25][DR43][DR52] -----------------------

    def pin_distillation_run(self, run_id: str, pinned: bool = True) -> bool:
        """Pin (or release) one run's evidence against retention.

        Pinning records ``pinned_at`` and the live span count at that moment
        ([DR43]): the pin only binds builds that carry the prune predicate, so
        a later shortfall against ``pinned_span_count`` is how a loss caused
        by an older binary is detected rather than discovered. Returns False
        when the run row does not exist.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if pinned:
                span_count = conn.execute(
                    "SELECT COUNT(*) FROM spans WHERE trace_id IN "
                    "(SELECT trace_id FROM distillation_passes WHERE run_id=?)",
                    (run_id,),
                ).fetchone()[0]
                cur = conn.execute(
                    "UPDATE distillation_runs SET pinned=1, pinned_at=?, "
                    "pinned_span_count=? WHERE run_id=?",
                    (_utcnow_iso(), int(span_count), run_id),
                )
            else:
                # pinned_at/pinned_span_count survive a release: they are the
                # record of what was protected and when.
                cur = conn.execute(
                    "UPDATE distillation_runs SET pinned=0 WHERE run_id=?", (run_id,)
                )
            conn.commit()
            return cur.rowcount > 0

    def distillation_evidence_shortfall(self, run_id: str) -> Optional[dict[str, Any]]:
        """Live span count vs the count recorded at pin time [DR43].

        ``incomplete`` is what the run header renders as "evidence incomplete
        — N of M spans are gone; a build without the retention pin may have
        pruned them", and what excludes the run from the promotion view.
        Returns None when the run does not exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pinned, pinned_at, pinned_span_count, evidence_pruned "
                "FROM distillation_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            live = conn.execute(
                "SELECT COUNT(*) FROM spans WHERE trace_id IN "
                "(SELECT trace_id FROM distillation_passes WHERE run_id=?)",
                (run_id,),
            ).fetchone()[0]
        recorded = row["pinned_span_count"]
        missing = 0
        if recorded is not None and int(recorded) > int(live):
            missing = int(recorded) - int(live)
        return {
            "run_id": run_id,
            "pinned": bool(row["pinned"]),
            "pinned_at": row["pinned_at"],
            "pinned_span_count": recorded,
            "live_span_count": int(live),
            "missing_span_count": missing,
            "evidence_pruned": bool(row["evidence_pruned"]),
            "incomplete": bool(missing) or bool(row["evidence_pruned"]),
        }

    def _release_distillation_pins(
        self, conn: sqlite3.Connection, negative_pin_days: int
    ) -> dict[str, int]:
        """The bounded pin-release sweep [DR52], run before both prune arms.

        Two classes lose their pin here, and both are computed rather than
        written by anyone: a no-divergence run that produced NO insight and is
        older than FW_OBS_DISTILL_NEGATIVE_PIN_DAYS (the contradiction set is
        valuable, but not forever), and a run whose every insight's newest
        non-superseded verdict is a rejection — the verdict route writes only
        ``distillation_verdicts`` (§12 rule 1), so the unpin has to be a
        consequence the pruner derives.

        The two arms partition on "has an insight", not on "diverged": a run
        can agree and still carry an insight, and that row is pinned by the
        insight class (`fix-sb8.17`).
        """
        cutoff = datetime.fromtimestamp(
            max(0.0, time.time() - negative_pin_days * 86_400), tz=timezone.utc
        ).isoformat()
        released = 0
        conn.execute("BEGIN IMMEDIATE")
        # The no-divergence class is time-limited, but "no divergence" and
        # "produced no insight" are not the same predicate (`fix-sb8.17`).
        # §10.3 pins an insight-producing run FOREVER, so a run that agreed and
        # still yielded an insight belongs to the insight class and is released
        # only by the rejected-only arm below. Nothing produces such a row today
        # — extraction is gated on divergence — but `fix-sb8.11`'s replay is
        # precisely an agreeing run that says something about an insight, and
        # the failure mode is silent evidence loss for the most interesting
        # insight there is.
        released += conn.execute(
            "UPDATE distillation_runs SET pinned=0 WHERE run_id IN "
            "(SELECT run_id FROM distillation_runs r WHERE r.pinned=1 "
            " AND r.planning_diverged=0 AND r.exec_diverged=0 "
            " AND NOT EXISTS (SELECT 1 FROM distillation_insights i "
            "                  WHERE i.run_id = r.run_id) "
            " AND COALESCE(r.pinned_at, r.started_at) IS NOT NULL "
            " AND COALESCE(r.pinned_at, r.started_at) < ? LIMIT ?)",
            (cutoff, _PRUNE_BATCH_ROWS),
        ).rowcount
        # Rejected-only runs: every insight adjudicated, none supported and
        # none left unadjudicated. A run with no insights is not in scope here
        # — that is the no-divergence class above. Divergence is deliberately
        # not part of this predicate, so an agreeing run carrying an insight
        # is released here and only here.
        released += conn.execute(
            """UPDATE distillation_runs SET pinned=0 WHERE run_id IN (
                 SELECT r.run_id FROM distillation_runs r
                  WHERE r.pinned=1
                    AND EXISTS (SELECT 1 FROM distillation_insights i
                                 WHERE i.run_id = r.run_id)
                    AND NOT EXISTS (
                        SELECT 1 FROM distillation_insights i
                         WHERE i.run_id = r.run_id
                           AND COALESCE((SELECT v.verdict FROM distillation_verdicts v
                                          WHERE v.insight_id = i.insight_id
                                            AND v.superseded = 0
                                          ORDER BY v.created_at DESC LIMIT 1),
                                        'unadjudicated')
                               IN ('unadjudicated', 'supported'))
                  LIMIT ?)""",
            (_PRUNE_BATCH_ROWS,),
        ).rowcount
        conn.commit()
        return {"distillation_pins_released": released}

    def _prune_distillation_evidence(
        self, conn: sqlite3.Connection, horizon_key: str, horizon_iso: str
    ) -> dict[str, int]:
        """The six tables under retention [DR52].

        Past the horizon an UNPINNED run loses the bulk — its divergence rows
        and its passes' ``entry_inputs_json`` — while the conclusions (the run
        row and its insights) survive, small, marked ``evidence_pruned`` so
        the UI can say "the trace behind this is gone" instead of rendering an
        empty diff.
        """
        deleted = {"distillation_divergences": 0, "distillation_evidence_pruned": 0}
        for _ in range(_PRUNE_MAX_BATCHES):
            conn.execute("BEGIN IMMEDIATE")
            run_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT run_id FROM distillation_runs "
                    " WHERE pinned=0 AND evidence_pruned=0 "
                    "   AND ((started_at IS NOT NULL AND started_at < ?) "
                    "        OR (started_at IS NULL AND turn_key < ?)) LIMIT ?",
                    (horizon_iso, horizon_key, _PRUNE_BATCH_ROWS),
                ).fetchall()
            ]
            for chunk in _chunked(run_ids):
                marks = _in_placeholders(len(chunk))
                # Citations first: a divergence deleted out from under one
                # leaves a citation pointing at nothing, which §15's recipes
                # would read as real provenance.
                conn.execute(
                    "DELETE FROM distillation_insight_citations WHERE divergence_id IN "
                    "(SELECT divergence_id FROM distillation_divergences "
                    f"WHERE run_id IN ({marks}))",
                    chunk,
                )
                deleted["distillation_divergences"] += conn.execute(
                    f"DELETE FROM distillation_divergences WHERE run_id IN ({marks})",
                    chunk,
                ).rowcount
                conn.execute(
                    "UPDATE distillation_passes SET entry_inputs_json=NULL "
                    f"WHERE run_id IN ({marks})",
                    chunk,
                )
                deleted["distillation_evidence_pruned"] += conn.execute(
                    f"UPDATE distillation_runs SET evidence_pruned=1 "
                    f"WHERE run_id IN ({marks})",
                    chunk,
                ).rowcount
            conn.commit()
            if len(run_ids) < _PRUNE_BATCH_ROWS:
                break
        return deleted

    @staticmethod
    def _read_diagnostic(conn: sqlite3.Connection, key: str) -> Optional[dict[str, Any]]:
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            loaded = json.loads(row[0])
        except (ValueError, TypeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def _pinned_span_bytes(self, conn: sqlite3.Connection) -> int:
        """Estimated bytes held by pinned traces: attribute bytes (97% of a
        trace's cost, design §10.1) plus a flat per-row allowance for the row
        header and index entries LENGTH(attributes) cannot see."""
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(attributes)), 0) FROM spans "
            f"WHERE trace_id IN ({_PINNED_TRACES_SQL})"
        ).fetchone()
        return int(row[1]) + int(row[0]) * _PINNED_ROW_OVERHEAD_BYTES

    def _evict_oldest_traces(
        self, conn: sqlite3.Connection, max_bytes: int
    ) -> dict[str, int]:
        """Size-cap eviction, oldest trace first and whole traces only `[DR27]`.

        This arm used to delete `LIMIT 5000` spans in `start_ns` order with no
        trace awareness, which cuts across trace boundaries by construction:
        the surviving half of a trace renders as a waterfall with silently
        missing rows — a turn that reads as though the agent skipped steps it
        actually took. That is worse than losing the trace outright, because
        nothing about it looks lossy. Whole traces, and an eviction marker so
        the loss is a fact somebody can find rather than an inference from a
        gap.

        The pin predicate stays INSIDE the victim selection `[DR52]`: on the
        outer `DELETE`, a batch whose oldest rows are all pinned deletes
        nothing, rowcount is 0, and the loop abandons eviction with the DB over
        its cap and evictable spans still present — no error and no marker.
        Pinned traces are evicted only by `_enforce_pin_ceiling`, and only
        once the pinned set has itself outgrown its share of the cap.
        """
        evicted = 0
        for _ in range(_PRUNE_MAX_BATCHES):
            if self.db_size_bytes() <= max_bytes:
                break
            conn.execute("BEGIN IMMEDIATE")
            traces = [
                r[0]
                for r in conn.execute(
                    "SELECT trace_id FROM spans "
                    f"WHERE trace_id NOT IN ({_PINNED_TRACES_SQL}) "
                    "GROUP BY trace_id ORDER BY MIN(start_ns) LIMIT ?",
                    (_EVICT_TRACES_PER_BATCH,),
                ).fetchall()
            ]
            if not traces:
                conn.commit()
                break
            marks = _in_placeholders(len(traces))
            # A distillation run whose evidence goes must say so, or its UI
            # renders an empty diff instead of "the trace behind this is gone".
            conn.execute(
                "UPDATE distillation_runs SET evidence_pruned=1 WHERE run_id IN "
                f"(SELECT run_id FROM distillation_passes WHERE trace_id IN ({marks}))",
                traces,
            )
            conn.execute(f"DELETE FROM spans WHERE trace_id IN ({marks})", traces)
            evicted += len(traces)
            marker = self._read_diagnostic(conn, "span_evictions") or {}
            self.set_diagnostic(
                conn,
                "span_evictions",
                {
                    "at": _utcnow_iso(),
                    "reason": "size-cap",
                    "traces_evicted": len(traces),
                    "total_traces_evicted": int(marker.get("total_traces_evicted") or 0)
                    + len(traces),
                },
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"traces_evicted": evicted}

    def _enforce_pin_ceiling(
        self, conn: sqlite3.Connection, max_bytes: int, pin_max_fraction: float
    ) -> dict[str, int]:
        """The pinned set's ceiling [DR52]: the cap wins over the pin, loudly.

        Two of the five retention classes pin indefinitely, so a pinned set
        left unbounded reaches the cap on its own and the cap silently stops
        holding. When eviction has finished and the DB is still over cap, this
        records ``distill_pin_over_cap``; if the pinned set alone is over
        FW_OBS_DISTILL_PIN_MAX_FRACTION of the cap it then evicts pinned
        traces oldest-first and TRACE-ATOMICALLY — a half-deleted trace
        renders as a waterfall with silently missing rows [DR27] — marking
        each evicted run ``evidence_pruned`` and writing the eviction marker.
        """
        size = self.db_size_bytes()
        if size <= max_bytes:
            return {}
        pinned_bytes = self._pinned_span_bytes(conn)
        ceiling = int(max_bytes * pin_max_fraction)
        conn.execute("BEGIN IMMEDIATE")
        self.set_diagnostic(
            conn,
            "distill_pin_over_cap",
            {
                "at": _utcnow_iso(),
                "db_size_bytes": size,
                "max_bytes": max_bytes,
                "over_cap_bytes": size - max_bytes,
                "pinned_bytes": pinned_bytes,
                "pin_max_fraction": pin_max_fraction,
                "ceiling_bytes": ceiling,
                "evicting_pinned": pinned_bytes > ceiling,
            },
        )
        conn.commit()
        logger.warning(
            f"Observability DB is {size} bytes over a {max_bytes}-byte cap with "
            f"~{pinned_bytes} bytes of pinned distillation evidence "
            f"(ceiling {ceiling}) [DR52]"
        )
        if pinned_bytes <= ceiling:
            return {"pinned_traces_evicted": 0}

        evicted = 0
        for _ in range(_PRUNE_MAX_BATCHES):
            if self.db_size_bytes() <= max_bytes:
                break
            if self._pinned_span_bytes(conn) <= ceiling:
                break
            conn.execute("BEGIN IMMEDIATE")
            traces = [
                r[0]
                for r in conn.execute(
                    "SELECT trace_id FROM spans "
                    f"WHERE trace_id IN ({_PINNED_TRACES_SQL}) "
                    "GROUP BY trace_id ORDER BY MIN(start_ns) LIMIT 32"
                ).fetchall()
            ]
            if not traces:
                conn.commit()
                break
            marks = _in_placeholders(len(traces))
            conn.execute(
                f"UPDATE distillation_runs SET evidence_pruned=1 WHERE run_id IN "
                f"(SELECT run_id FROM distillation_passes WHERE trace_id IN ({marks}))",
                traces,
            )
            conn.execute(
                f"DELETE FROM spans WHERE trace_id IN ({marks})", traces
            )
            evicted += len(traces)
            marker = self._read_diagnostic(conn, "span_evictions") or {}
            self.set_diagnostic(
                conn,
                "span_evictions",
                {
                    "at": _utcnow_iso(),
                    "reason": "pinned-set-over-ceiling",
                    "traces_evicted": len(traces),
                    "total_traces_evicted": int(marker.get("total_traces_evicted") or 0)
                    + len(traces),
                },
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"pinned_traces_evicted": evicted}

    def forget_channel(self, channel_id: str) -> dict[str, int]:
        """First-class erasure [R21]: delete a channel across all tables, then
        checkpoint-truncate the WAL and reclaim pages."""
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # [DR44]: the id sets are collected BEFORE anything is deleted.
            # PRAGMA foreign_keys is off and the schema declares no
            # REFERENCES, so nothing cascades and the wrong order leaves
            # orphan divergences that read as real evidence.
            turn_keys = [
                r[0]
                for r in conn.execute(
                    "SELECT turn_key FROM turns WHERE channel_id=?", (channel_id,)
                ).fetchall()
            ]
            run_rows = conn.execute(
                "SELECT run_id, turn_key FROM distillation_runs "
                "WHERE channel_id=? OR turn_key IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id, channel_id),
            ).fetchall()
            run_ids = [r[0] for r in run_rows]
            turn_keys.extend(r[1] for r in run_rows if r[1])
            touched_experiments = [
                r[0]
                for r in conn.execute(
                    """SELECT DISTINCT experiment_id FROM turns
                        WHERE channel_id=? AND experiment_id IS NOT NULL
                       UNION
                       SELECT DISTINCT experiment_id FROM experiment_attempts
                        WHERE channel_id=? AND experiment_id IS NOT NULL""",
                    (channel_id, channel_id),
                ).fetchall()
            ]
            deleted.update(self._delete_distillation_runs(conn, run_ids))
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
            # Replay traces are '<turn_key>~replay.<n>' [DR41]: the channel_id
            # arm above catches the ones that carry a channel, this catches
            # the rest [DR44]. Set-based on purpose — see
            # _delete_derived_trace_spans: one statement per turn key made
            # [R21] erasure quadratic in channel size.
            deleted["spans"] += self._delete_derived_trace_spans(conn, turn_keys)
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

    @staticmethod
    def _delete_derived_trace_spans(
        conn: sqlite3.Connection, turn_keys: Iterable[str]
    ) -> int:
        """Delete the spans of every trace derived from one of *turn_keys*.

        Derived trace ids are ``'<turn_key>~<suffix>'`` — today only
        ``~replay.<n>`` (§3.5 row 4, `[DR41]`) — and `[DR44]` requires erasure
        to reach them.

        Deliberately ONE statement rather than one ``trace_id LIKE key || '~%'``
        per key: a LIKE with an ESCAPE clause cannot use ``idx_spans_trace``, so
        the per-key shape was a full scan of `spans` per turn of the channel,
        all of it inside the caller's ``BEGIN IMMEDIATE`` — quadratic in channel
        size on the ordinary-turn erasure path (measured here at 0.039s for 500
        turns and 3.06s for 4000: 8x the turns for 78x the work; the regression
        auditor measured 0.08s -> 9.26s on a larger seed). The keys go into a
        temp table so the count of statements against `spans` stays constant
        however large the channel is.

        Turn keys are minted by `mint_turn_key` (`turn.py:91`) as
        ``YYYYMMDDTHHMMSSZ-<hex>`` and never contain '~', so the text before the
        first '~' of a derived trace id is exactly the turn key it derives from.
        """
        keys = sorted({key for key in turn_keys if key})
        if not keys:
            return 0
        conn.execute("DROP TABLE IF EXISTS temp._forget_turn_keys")
        conn.execute(
            "CREATE TEMP TABLE _forget_turn_keys (turn_key TEXT PRIMARY KEY)"
        )
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO temp._forget_turn_keys(turn_key) VALUES (?)",
                [(key,) for key in keys],
            )
            return conn.execute(
                "DELETE FROM spans WHERE instr(trace_id, '~') > 0 AND "
                "substr(trace_id, 1, instr(trace_id, '~') - 1) IN "
                "(SELECT turn_key FROM temp._forget_turn_keys)"
            ).rowcount
        finally:
            conn.execute("DROP TABLE IF EXISTS temp._forget_turn_keys")

    def _delete_distillation_runs(
        self, conn: sqlite3.Connection, run_ids: list[str]
    ) -> dict[str, int]:
        """Delete the distillation closure of *run_ids*, children first [DR44].

        Called inside the caller's transaction with the parent rows still
        present; verdicts hang off insight ids, so those are collected before
        the insights they belong to are deleted.
        """
        deleted = {table: 0 for table in _DISTILL_TABLES}
        if not run_ids:
            return deleted
        insight_ids: list[str] = []
        for chunk in _chunked(run_ids):
            insight_ids.extend(
                r[0]
                for r in conn.execute(
                    "SELECT insight_id FROM distillation_insights WHERE run_id IN "
                    f"({_in_placeholders(len(chunk))})",
                    chunk,
                ).fetchall()
            )
        for chunk in _chunked(insight_ids):
            marks = _in_placeholders(len(chunk))
            deleted["distillation_verdicts"] += conn.execute(
                f"DELETE FROM distillation_verdicts WHERE insight_id IN ({marks})",
                chunk,
            ).rowcount
            deleted["distillation_insight_citations"] += conn.execute(
                "DELETE FROM distillation_insight_citations WHERE insight_id IN "
                f"({marks})",
                chunk,
            ).rowcount
        for chunk in _chunked(run_ids):
            marks = _in_placeholders(len(chunk))
            # Citations also hang off divergence ids: an insight from another
            # run may cite a divergence of this one.
            deleted["distillation_insight_citations"] += conn.execute(
                "DELETE FROM distillation_insight_citations WHERE divergence_id IN "
                "(SELECT divergence_id FROM distillation_divergences "
                f"WHERE run_id IN ({marks}))",
                chunk,
            ).rowcount
            deleted["distillation_insights"] += conn.execute(
                f"DELETE FROM distillation_insights WHERE run_id IN ({marks})", chunk
            ).rowcount
            deleted["distillation_divergences"] += conn.execute(
                f"DELETE FROM distillation_divergences WHERE run_id IN ({marks})",
                chunk,
            ).rowcount
            deleted["distillation_passes"] += conn.execute(
                f"DELETE FROM distillation_passes WHERE run_id IN ({marks})", chunk
            ).rowcount
            deleted["distillation_runs"] += conn.execute(
                f"DELETE FROM distillation_runs WHERE run_id IN ({marks})", chunk
            ).rowcount
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
            # All six distillation tables included [DR44]: they hold verbatim
            # user content (user_message, entry_inputs_json, param_diff_json,
            # detail_json, insight text) and the parent design's erasure
            # wording is "across all tables".
            # The three experiment tables are included for the same reason the
            # six distillation tables are: this is "clear all conversations", and
            # a container row surviving it would hold operator prose and a score
            # for turns that no longer exist `[XR15]`. Newest-dependency first.
            for table in (
                "feedback",
                "spans",
                "artifacts",
                "turns",
                "conversations",
            ) + _DISTILL_TABLES + _EXPERIMENT_TABLES:
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
        self._features: frozenset[str] = frozenset()
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
        # This handle can never migrate, so feature detection is the only
        # thing standing between a pre-distillation snapshot and a bare
        # "no such column" 500 [DR29].
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


class _FlushTicket:
    """One outstanding `SQLiteTraceSink.flush()` barrier `[DR49]`.

    A bare `threading.Event` could only say that the writer *reached* the
    barrier, which is what made §7.5's read barrier a liveness barrier rather
    than a durability one: `_apply_batch` set the event in its `finally`, so a
    caller was told the flush had succeeded even when the batch had just been
    rolled back and its spans discarded. The ticket carries the outcome too.

    It also carries the barrier's WATERMARK: the enqueue sequence numbers the
    writer has to reach for this barrier's work to be done. Settling on that
    rather than on global queue quiescence is what keeps the barrier from
    being starved (`fix-sb8.15`) — one sink is shared by every channel on a
    workflow DB, so "both queues are empty" is a condition another channel's
    steady span emission can prevent forever.
    """

    __slots__ = ("event", "ok", "span_mark", "record_mark")

    def __init__(self, span_mark: int = 0, record_mark: int = 0) -> None:
        self.event = threading.Event()
        self.ok = False
        self.span_mark = span_mark
        self.record_mark = record_mark


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
        # [DR49] durability-barrier state. `_write_failures` counts batches
        # that rolled back; `_pending_flushes` holds barriers whose batch
        # committed but whose watermark the writer has not reached yet. Both
        # are touched only on the writer thread.
        self._write_failures = 0
        self._failures_reported = 0
        self._pending_flushes: list[_FlushTicket] = []
        # Barrier watermarks (`fix-sb8.15`). `_span_enq`/`_record_enq` count
        # everything ever accepted onto each queue and are bumped by producer
        # threads under `_seq_lock`; `_span_done`/`_record_done` count what the
        # writer has dequeued and are touched only on the writer thread. Each
        # queue is FIFO and drained in order, so `done >= mark` is exactly
        # "the first `mark` items enqueued on that queue have been applied".
        #
        # `_trace_last_seq` is what makes a barrier SCOPED: for each trace it
        # holds `_span_enq` as of that trace's most recent span, so a
        # `flush(trace_id=...)` caller waits for its own trace's spans and not
        # for a backlog another channel piled on afterwards. Entries at or
        # below `_span_done` are already satisfied and are dropped when the
        # map grows — a missing entry reads as 0, which is the same answer.
        self._seq_lock = threading.Lock()
        self._span_enq = 0
        self._record_enq = 0
        self._span_done = 0
        self._record_done = 0
        self._trace_last_seq: dict[str, int] = {}
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
                distillation_pass=span.distillation_pass,
            )
            self._span_queue.put_nowait(("span", snapshot))
            self._note_enqueued("span", snapshot.trace_id)
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
            self._note_enqueued("record")
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
            self._note_enqueued("record")
        except queue.Full:
            self._count("records_dropped")
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    def emit_distillation_record(self, kind: str, payload: dict[str, Any]) -> None:
        """Queue one distillation row on the record queue [DR46].

        Deliberately the record queue and not a direct store write: it
        inherits the existing writer thread, busy-retry, breaker and
        writer_health counters, so a lock contention or OperationalError can
        never surface on the turn thread ([R14]). Records are small and rare
        (a handful per distillation run) next to spans, so the small bounded
        queue is the right one.
        """
        if self._closed:
            return
        if kind not in _DISTILL_RECORD_TABLES:
            self._count("write_errors", error=f"distillation kind: {kind!r}")
            return
        try:
            # Copied: the producer owns its dict and may keep mutating it
            # while this one waits on the writer thread.
            self._record_queue.put(
                ("distill", kind, dict(payload), 0), timeout=_RECORD_PUT_TIMEOUT_S
            )
            self._note_enqueued("record")
        except queue.Full:
            self._count("records_dropped")
            logger.warning(
                f"Observability record queue full; DROPPED distillation "
                f"{kind} record [R13]"
            )
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    # -- lifecycle -------------------------------------------------------

    def flush(
        self, timeout: float = 10.0, trace_id: Optional[str] = None
    ) -> bool:
        """Durability barrier: block until everything enqueued so far is
        COMMITTED, and report whether it was `[DR49]`.

        Returns ``True`` only when both queues drained to empty and no batch
        has been discarded since the previous barrier settled. Returns
        ``False`` when:

        * a batch rolled back — ``sqlite3.OperationalError`` (SQLITE_BUSY under
          multi-process contention `[R8]`, whose recovery drops the batch's
          spans outright) or any other write error — and no later barrier has
          reported that loss yet;
        * the barrier could not be enqueued (record queue full);
        * the barrier did not complete within *timeout* (a wedged or stopped
          writer).

        A malformed individual distillation record is NOT a barrier failure:
        `_apply_distillation` counts it in ``writer_health['write_errors']`` and
        keeps the rest of its batch, so the batch is not discarded. Detecting
        that loss is `records_dropped` / `write_errors` monitoring, not this.

        The bool is load-bearing, not a courtesy: §7.5's read barrier is the
        only thing standing between "the aligner holds these `Span` objects in
        memory" and "the table holds the rows a divergence record is about to
        cite". A ``False`` return means spans or records enqueued before the
        call may not be in the table, so the caller must record the run
        ``comparable = 0`` / ``evidence-incomplete`` rather than treat what it
        holds in memory as persisted evidence.

        Committing the one batch the sentinel happens to land in is NOT enough
        for that meaning. ``_next_item`` takes the record queue first, so the
        sentinel can be dequeued ahead of spans that were enqueued before it;
        completing the barrier there would let the caller cite spans still
        sitting in a 10,000-slot queue. The barrier therefore waits on a
        WATERMARK — enqueue sequence numbers taken as the ticket is created —
        rather than on the queues being empty.

        *trace_id* SCOPES the barrier, and is what a caller with a trace to
        cite should pass. Waiting for empty was the original spelling and it is
        starvable (`fix-sb8.15`): `get_observability_sink` caches one sink per
        workflow DB and shares it across channels, so any other channel
        emitting spans steadily keeps both-queues-empty from ever being true
        and burns the whole timeout. An unscoped watermark narrows that window
        but does not close it — a backlog another channel piles up between the
        caller's last span and its `flush()` call is still enqueued "before"
        the ticket. Scoped to a trace, the barrier waits for the caller's own
        spans and nothing else: `_trace_last_seq[trace_id]` is the sequence of
        that trace's most recent span, and the span queue is FIFO, so
        `_span_done` reaching it means every span of that trace has been
        applied. Records are not scoped — they are rare and small, and a
        distillation run's own rows ride that queue.

        Passing no *trace_id* keeps the original whole-sink meaning, which is
        what `close`-time and test callers want.
        """
        with self._seq_lock:
            span_mark = (
                self._trace_last_seq.get(trace_id, 0)
                if trace_id is not None
                else self._span_enq
            )
            ticket = _FlushTicket(span_mark, self._record_enq)
        try:
            self._record_queue.put(("flush", ticket), timeout=timeout)
        except queue.Full:
            return False
        self._note_enqueued("record")
        if not ticket.event.wait(timeout):
            return False
        return ticket.ok

    def _note_enqueued(self, which: str, trace_id: Optional[str] = None) -> None:
        """Count one accepted enqueue for the `fix-sb8.15` barrier watermark."""
        with self._seq_lock:
            if which == "span":
                self._span_enq += 1
                if trace_id:
                    self._trace_last_seq[trace_id] = self._span_enq
                    if len(self._trace_last_seq) > _TRACE_MARK_CEILING:
                        self._trim_trace_marks()
            else:
                self._record_enq += 1

    def _trim_trace_marks(self) -> None:
        """Drop `_trace_last_seq` entries the writer has already passed.

        Called from the one place the map grows, because a session that never
        takes a barrier would otherwise accumulate one int per trace for the
        life of the process — a slow leak in the ordinary path, paid for by a
        feature only distillation uses.

        Dropping is safe: a trace whose last span has been applied cannot hold
        a barrier open, so a later `flush(trace_id=...)` for it reads the
        missing entry as 0 and settles immediately — the same answer the
        retained mark would give. `_span_done` is the writer thread's, read
        here without the lock; an under-read only keeps an entry one sweep
        longer. Caller holds `_seq_lock`.
        """
        done = self._span_done
        self._trace_last_seq = {
            trace: seq for trace, seq in self._trace_last_seq.items() if seq > done
        }

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
                    # Idle: both queues are empty, so every parked [DR49]
                    # barrier's watermark has been reached.
                    self._settle_flushes([])
                    self._heartbeat(conn)
                    continue
                self._apply_batch(conn, [item] + self._drain_pending())
                self._settle_flushes([])
            # Final drain: everything enqueued before close() is written.
            while items := self._drain_pending():
                self._apply_batch(conn, items)
            self._settle_flushes([])
            # Then the retry ring, which holds terminal rows the queue may have
            # dropped — the last thing standing between a wedged-then-recovered
            # DB and a permanently missing turn.
            self._retry_pending(conn)
        except Exception as exc:  # writer must never crash the process
            self._count("write_errors", error=repr(exc))
            logger.warning(f"Observability writer loop error: {exc!r}")
        finally:
            self._abandon_flushes()
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
        """One item, records first; None on idle timeout (health heartbeat).

        Every dequeue advances the matching `fix-sb8.15` watermark. Advancing
        at dequeue rather than after the commit is safe because
        `_settle_flushes` is only ever called once a batch has been fully
        applied — committed, or rolled back and force-settled.
        """
        try:
            item = self._record_queue.get_nowait()
            self._record_done += 1
            return item
        except queue.Empty:
            pass
        try:
            item = self._span_queue.get(timeout=0.25)
            self._span_done += 1
            return item
        except queue.Empty:
            return None

    def _drain_pending(self, limit: int = 512) -> list:
        items = []
        for _ in range(limit):
            try:
                items.append(self._record_queue.get_nowait())
                self._record_done += 1
                continue
            except queue.Empty:
                pass
            try:
                items.append(self._span_queue.get_nowait())
                self._span_done += 1
            except queue.Empty:
                break
        return items

    def _apply_batch(self, conn: sqlite3.Connection, items: list) -> None:
        tickets: list[_FlushTicket] = []
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
                elif kind == "distill":
                    self._apply_distillation(conn, item)
                elif kind == "flush":
                    tickets.append(item[1])
            if spans:
                self.store.upsert_span_rows(conn, spans, self._redactor)
            self._maybe_write_health(conn, in_txn=True)
            conn.commit()
        except sqlite3.OperationalError as exc:
            # SQLITE_BUSY under multi-process contention [R8]. The batch is
            # gone — _requeue_records drops its spans outright — so any barrier
            # covering it has failed [DR49].
            self._rollback(conn)
            self._count("busy_retries", error=repr(exc))
            self._write_failures += 1
            self._settle_flushes(tickets, force=True)
            self._requeue_records(items)
            return
        except Exception as exc:
            # Nothing is requeued on this arm at all, so the loss is total.
            self._rollback(conn)
            self._count("write_errors", error=repr(exc))
            self._write_failures += 1
            self._settle_flushes(tickets, force=True)
            return
        # Committed. A barrier is still not satisfied until everything
        # enqueued before its ticket has been applied, so a ticket whose
        # watermark this batch did not reach is parked rather than completed.
        self._settle_flushes(tickets)

    def _settle_flushes(
        self, tickets: list[_FlushTicket], *, force: bool = False
    ) -> None:
        """Complete the [DR49] barriers whose work is finished.

        A barrier is satisfied once every item enqueued BEFORE its ticket has
        been dequeued and applied — `_span_done`/`_record_done` have reached
        the ticket's watermark — AND no batch has been discarded since the
        previous barrier settled. `force` completes every outstanding ticket
        now, because a batch just rolled back and those barriers can no longer
        be met.

        The watermark test replaced a both-queues-empty test, which was
        starvable by any other producer on the shared sink (`fix-sb8.15`).
        Tickets whose watermark is not yet reached stay parked and are
        re-examined after every batch and on every idle tick, so a ticket is
        never left waiting on an event that has already passed.

        The unreported-failure watermark, rather than a per-ticket baseline, is
        what makes the guarantee hold in the case the auditor's sequence does
        not cover: a batch can fail *before* `flush()` is called and still have
        carried spans the caller is about to cite, because it was that
        caller's own earlier `emit_span`. Comparing against a baseline taken at
        call time would report that batch as a satisfied barrier. Charging
        every loss to the next barrier costs at most one conservative ``False``
        per loss, and no loss goes unreported.
        """
        pending = self._pending_flushes
        pending.extend(tickets)
        if not pending:
            return
        if force:
            ready, parked = list(pending), []
        else:
            span_done, record_done = self._span_done, self._record_done
            ready, parked = [], []
            for ticket in pending:
                target = (
                    ready
                    if ticket.span_mark <= span_done
                    and ticket.record_mark <= record_done
                    else parked
                )
                target.append(ticket)
            if not ready:
                return
        self._pending_flushes = parked
        ok = self._write_failures == self._failures_reported
        self._failures_reported = self._write_failures
        for ticket in ready:
            ticket.ok = ok
            ticket.event.set()

    def _abandon_flushes(self) -> None:
        """Release still-waiting barriers when the writer thread is going away.

        Without this a `flush()` caller would block for its whole timeout on a
        writer that has already exited. It reports failure, which is the truth:
        the queues were not drained.
        """
        for ticket in self._pending_flushes:
            ticket.ok = False
            ticket.event.set()
        self._pending_flushes.clear()

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

    def _apply_distillation(self, conn: sqlite3.Connection, item: tuple) -> None:
        _, record_kind, payload, _retries = item
        try:
            self.store.upsert_distillation_row(
                conn, record_kind, payload, self._redactor
            )
        except sqlite3.OperationalError:
            # SQLITE_BUSY and friends belong to the batch handler, which
            # rolls back and requeues the whole batch [R8].
            raise
        except OrphanedCitation as exc:
            # Not malformed and not this batch's fault: the divergence row it
            # names was lost earlier. Counted as a write error so the run is
            # completed `comparable = 0` / `evidence-incomplete` rather than
            # silently losing one edge of §13.2's provenance chain.
            self._count("write_errors", error=f"distillation citation: {exc}")
            logger.warning(f"Observability: {exc} [fix-sb8.16]")
        except Exception as exc:
            # A malformed record is this record's problem: counting it here
            # keeps it from rolling back the rest of a good batch.
            self._count("write_errors", error=f"distillation {record_kind}: {exc!r}")

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
                # Barrier tickets are settled by _settle_flushes on the failure
                # arm that called us; requeueing one would report a rolled-back
                # batch as a satisfied barrier [DR49].
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
                self._note_enqueued("record")
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
