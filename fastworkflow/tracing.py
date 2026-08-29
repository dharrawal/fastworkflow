"""OTel-shaped trace emission at the agent↔workflow boundary.

Implements the flight-recorder half of the observability design
(docs/fastworkflow_observability_studio_design.md §3.1): a ``TraceSink``
protocol with a no-op default, the v1 span taxonomy, and safe emission
helpers that NEVER raise to the caller — a broken sink degrades to a log
line, not a failed turn.

Spans are OTel-*aligned* records, not wire-conformant OTel (decision D4):
``trace_id`` is the logical turn_key, span ids are opaque strings, and the
translation to real OTel ids is an external script's contract ([R26]).

Sink discovery is duck-typed off the host object (WorkflowExecutionContext,
or ChatSession delegating to its core) via ``trace_sink`` /
``current_turn_key`` / ``trace_span_stack`` — deliberately NOT the
transport-queue contract, so queue-less embedders still trace ([R28]).

This module is stdlib-only by design: it is imported by core runtime
modules and must never pull torch/dspy/transformers.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Host propagation for deep emission sites (v2 spans, D3 as amended)
# ----------------------------------------------------------------------
#
# The NLU internals (intent detection, parameter extraction) run several call
# frames below CommandExecutor.invoke_command and have no reference to the
# WEC/ChatSession that owns the trace sink. Rather than threading a host
# parameter through the whole pipeline, invoke_command binds the host into a
# ContextVar for the duration of the call; deep sites read it back with
# ``current_host()``. The pipeline is synchronous on the calling thread, so
# the binding is race-free per turn — and it honors [R28]: the sink is still
# reached via the WEC, never via a transport queue.

_current_host: contextvars.ContextVar = contextvars.ContextVar(
    "fw_trace_host", default=None
)


def current_host() -> Any:
    """The trace host bound by the nearest enclosing ``host_scope`` (or None)."""
    return _current_host.get()


@contextlib.contextmanager
def host_scope(host: Any) -> Iterator[None]:
    """Bind *host* as the current trace host for the enclosed call stack."""
    token = _current_host.set(host)
    try:
        yield
    finally:
        _current_host.reset(token)

# ----------------------------------------------------------------------
# Span-name taxonomy
# ----------------------------------------------------------------------

# v1 — emitted at the agent↔workflow boundary (decision D3).
SPAN_TURN = "fw.turn"
SPAN_PLANNER_PLAN = "fw.planner.plan"
SPAN_PLANNER_REPLAN = "fw.planner.replan"
SPAN_AGENT_TOOL_CALL = "fw.agent.tool_call"
SPAN_COMMAND_EXECUTE = "fw.command.execute"
SPAN_ASK_USER = "fw.ask_user"

V1_SPAN_NAMES = frozenset(
    {
        SPAN_TURN,
        SPAN_PLANNER_PLAN,
        SPAN_PLANNER_REPLAN,
        SPAN_AGENT_TOOL_CALL,
        SPAN_COMMAND_EXECUTE,
        SPAN_ASK_USER,
    }
)

# The agent loop's own structure. Without these two, a turn's shape has to be
# INFERRED by a reader — the planner and the ReAct calls are flat siblings of
# fw.turn, and "which tool call belongs to which reasoning step" is only
# recoverable from DSPy module names plus timestamps. They make the two levels
# a developer actually thinks in — the executor ran, and it took these steps —
# structural facts instead of a heuristic.
#
# fw.agent.execute is the executor as a phase, sibling to fw.planner.plan. It
# is NOT fw.command.execute: that one is a single command inside a tool call,
# this one is the whole loop.
SPAN_AGENT_EXECUTE = "fw.agent.execute"
SPAN_AGENT_STEP = "fw.agent.step"

AGENT_LOOP_SPAN_NAMES = frozenset({SPAN_AGENT_EXECUTE, SPAN_AGENT_STEP})

# Originally reserved-for-v2 so the schema needed no migration when the deeper
# emitters landed. They have (D3 amendment, 2026-08-26): fw.nlu.* emit inside
# the CME pipeline and fw.llm.call at the DSPy callback level; only fw.train.*
# is still reserved. The set name is historical and kept for stability.
SPAN_NLU_INTENT = "fw.nlu.intent"
SPAN_NLU_PARAM_EXTRACTION = "fw.nlu.param_extraction"
SPAN_LLM_CALL = "fw.llm.call"
SPAN_TRAIN_PREFIX = "fw.train."

RESERVED_V2_SPAN_NAMES = frozenset(
    {SPAN_NLU_INTENT, SPAN_NLU_PARAM_EXTRACTION, SPAN_LLM_CALL, SPAN_TRAIN_PREFIX}
)

# Distillation structure spans (distillation design §8, [DR21]). Deliberately
# NOT under fw.train.*: that prefix means the training pipeline, distillation
# is a run-time surface, and SPAN_TRAIN_PREFIX is stored above as a *prefix*
# rather than a name, so set membership against it is already ambiguous.
# All five are KIND_INTERNAL — they are structure; the model invocations
# inside them stay fw.llm.call spans, now parented by the enclosing pass.
SPAN_DISTILL_RUN = "fw.distill.run"
SPAN_DISTILL_PASS = "fw.distill.pass"
SPAN_DISTILL_COMPARE = "fw.distill.compare"
SPAN_DISTILL_EXTRACT = "fw.distill.extract"
SPAN_DISTILL_REPLAY = "fw.distill.replay"

DISTILL_SPAN_NAMES = frozenset(
    {
        SPAN_DISTILL_RUN,
        SPAN_DISTILL_PASS,
        SPAN_DISTILL_COMPARE,
        SPAN_DISTILL_EXTRACT,
        SPAN_DISTILL_REPLAY,
    }
)

# Span kinds (spans.kind column): internal | llm | human_wait | tool.
KIND_INTERNAL = "internal"
KIND_LLM = "llm"
KIND_HUMAN_WAIT = "human_wait"
KIND_TOOL = "tool"

# Span statuses. "open" means started and not yet ended; a store treats a
# re-emission of the same span_id as an idempotent upsert ([R2][R6]).
STATUS_OPEN = "open"
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_AWAITING_USER = "awaiting_user"

_DEFAULT_MAX_ATTR_BYTES = 16384


@dataclass
class Span:
    """One OTel-shaped span record (spans table shape, design §3.2)."""

    span_id: str
    trace_id: str  # = turn_key, or <turn_key>~replay.<n> for a replay [DR1]
    name: str
    kind: str = KIND_INTERNAL
    parent_span_id: Optional[str] = None
    channel_id: Optional[str] = None
    command_name: Optional[str] = None
    context: Optional[str] = None
    start_ns: int = 0
    end_ns: Optional[int] = None
    status: str = STATUS_OPEN
    attributes: dict[str, Any] = field(default_factory=dict)
    # The distillation pass this span was emitted inside, or None ([DR23]): a
    # real column rather than an attribute, so it is indexable, outside the
    # 16 KiB attribute cap, and outside the Redactor's blind substring pass.
    distillation_pass: Optional[str] = None


# ----------------------------------------------------------------------
# Sink protocol
# ----------------------------------------------------------------------


@runtime_checkable
class TraceSink(Protocol):
    """Trace/record sink (design §3.1). Implementations must never raise to
    the caller; the emission helpers below additionally guard every call."""

    def emit_span(self, span: Span) -> None: ...

    def emit_turn_record(self, record: Any) -> None:
        """Receive the internal TurnResult at turn finalize (typed ``Any`` to
        keep this module import-light)."""
        ...

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None: ...  # [R15]

    def emit_distillation_record(self, kind: str, payload: dict[str, Any]) -> None:
        """Persist one distillation row ([DR46]).

        ``kind`` is one of ``run`` | ``pass`` | ``divergence`` | ``insight`` |
        ``citation`` and selects the table; ``payload`` is a flat column->value
        mapping whose keys are validated against that table's columns by the
        implementation. Upsert semantics on the table's primary key, so a run
        row may be written at start and completed later.

        Verdicts are NOT written through here: they are a viewer-side HTTP
        route, one of the two deliberate off-turn-thread exemptions [DR46].
        """
        ...


class NoOpTraceSink:
    """Default sink: tracing structurally present, nothing recorded."""

    def emit_span(self, span: Span) -> None:
        pass

    def emit_turn_record(self, record: Any) -> None:
        pass

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None:
        pass

    def emit_distillation_record(self, kind: str, payload: dict[str, Any]) -> None:
        pass


# ----------------------------------------------------------------------
# Span identity
# ----------------------------------------------------------------------


def deterministic_span_id(
    turn_key: str,
    span_name: str,
    attempt: int = 0,
    pass_label: Optional[str] = None,
) -> str:
    """Deterministic span id for spans that must close in a different process
    than the one that opened them (fw.turn, fw.ask_user) — [R6].

    ``pass_label`` is folded into the digest **only when non-None** ([DR11]):
    the two-pass shapes mint the same (turn_key, span_name, attempt) triple
    twice, so without it the student's span would upsert over the teacher's.
    Leaving it None keeps the digest byte-identical for every pre-distillation
    caller, root_span_id included.
    """
    seed = f"{turn_key}|{span_name}|{attempt}"
    if pass_label is not None:
        seed = f"{seed}|{pass_label}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return digest[:32]


def root_span_id(turn_key: str) -> str:
    """The fw.turn root span id for a logical turn."""
    return deterministic_span_id(turn_key, SPAN_TURN, 0)


def distill_pass_span_id(turn_key: str, pass_label: str) -> str:
    """The fw.distill.pass span id for one pass of a distilled turn ([DR51]).

    Deterministic rather than a uuid4 because _close_ask_user_span rebuilds its
    Span from pure functions of the turn key and can therefore only parent onto
    something it can *compute*, and parent_span_id is not in the span upsert's
    DO UPDATE set — the open has to be right the first time. The label alone
    carries the uniqueness, so the attempt ordinal stays 0: a caller holding
    only the label (which is all the ask_user close has) must be able to
    reproduce this id.
    """
    return deterministic_span_id(
        turn_key, SPAN_DISTILL_PASS, 0, pass_label=pass_label
    )


def _max_attr_bytes() -> int:
    try:
        return int(os.environ.get("FW_OBS_MAX_ATTR_BYTES", "") or _DEFAULT_MAX_ATTR_BYTES)
    except ValueError:
        return _DEFAULT_MAX_ATTR_BYTES


def cap_attr_value(value: Any) -> Any:
    """Cap one attribute value at FW_OBS_MAX_ATTR_BYTES.

    Truncation is lossy-and-counted ([R10]): an over-limit string becomes an
    envelope carrying ``truncated: True``, the original byte length, and the
    sha256 of the original — never a silent prefix.
    """
    if not isinstance(value, str):
        return value
    limit = _max_attr_bytes()
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return {
        "truncated": True,
        "original_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "value": raw[:limit].decode("utf-8", errors="ignore"),
    }


def _capped(attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not attributes:
        return {}
    return {key: cap_attr_value(value) for key, value in attributes.items()}


# ----------------------------------------------------------------------
# Duck-typed host access (WEC directly, or ChatSession via _core)
# ----------------------------------------------------------------------


def _resolve(host: Any, attr: str) -> Any:
    value = getattr(host, attr, None)
    if value is not None:
        return value
    core = getattr(host, "_core", None)
    if core is not None:
        return getattr(core, attr, None)
    return None


def get_sink(host: Any) -> Optional[TraceSink]:
    sink = _resolve(host, "trace_sink")
    return sink if sink is not None and not isinstance(sink, NoOpTraceSink) else None


def get_turn_key(host: Any) -> Optional[str]:
    return _resolve(host, "current_turn_key")


def get_channel_id(host: Any) -> Optional[str]:
    return _resolve(host, "observability_channel_id")


def get_replay_trace_id(host: Any) -> Optional[str]:
    """The derived replay trace id bound on *host*, or None ([DR41]).

    A SECOND attribute beside `current_turn_key`, never a substitute for it:
    the deterministic span ids `root_span_id(turn_key)` and
    `deterministic_span_id(turn_key, 'fw.ask_user', attempt)` are pure
    functions of the turn key, so re-running under the original key would
    regenerate the SAME span ids and the upsert's `DO UPDATE` would rewrite
    the very evidence a pin exists to protect ([DR4]).
    """
    return _resolve(host, "current_replay_trace_id")


def get_distillation_pass(host: Any) -> Optional[str]:
    """The distillation pass currently executing on *host*, or None ([DR3])."""
    return _resolve(host, "current_distillation_pass")


def _get_stack(host: Any) -> Optional[list]:
    return _resolve(host, "trace_span_stack")


# ----------------------------------------------------------------------
# Emission helpers — never raise to the caller
# ----------------------------------------------------------------------


def _emit(sink: TraceSink, span: Span) -> None:
    try:
        sink.emit_span(span)
    except Exception as exc:  # a broken sink must never fail a turn
        logger.warning(f"TraceSink.emit_span failed (span {span.name}): {exc!r}")


def datetime_to_ns(value: Any) -> Optional[int]:
    """Epoch nanoseconds for a datetime, or None when absent/unparseable."""
    try:
        return int(value.timestamp() * 1_000_000_000) if value is not None else None
    except Exception:
        return None


def start_span(
    host: Any,
    name: str,
    *,
    kind: str = KIND_INTERNAL,
    attributes: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    emit_open: bool = False,
    use_stack: bool = True,
) -> Optional[Span]:
    """Open a span under the host's current turn; returns None when there is
    no active sink or no open turn. Never raises.

    Short-lived spans are emitted once, at ``end_span``; pass
    ``emit_open=True`` for long-lived spans (fw.turn, fw.ask_user) whose open
    event must be visible before — and closable after — a suspension ([R6]).
    ``use_stack=False`` keeps a span off the parenting stack (fw.turn and
    fw.ask_user: children reach the root via its deterministic id, which
    survives suspension where the in-memory stack does not).
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        # [DR41]: a replay's derived id wins where one is bound; otherwise the
        # turn key, which is every ordinary span. Declining requires NEITHER.
        trace_id = get_replay_trace_id(host) or get_turn_key(host)
        if not trace_id:
            return None

        stack = _get_stack(host)
        if parent_span_id is None:
            if stack:
                parent_span_id = stack[-1].span_id
            elif name != SPAN_TURN:
                parent_span_id = root_span_id(trace_id)

        span = Span(
            span_id=span_id or uuid.uuid4().hex,
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            channel_id=get_channel_id(host),
            command_name=command_name,
            context=context,
            start_ns=time.time_ns(),
            status=STATUS_OPEN,
            attributes=_capped(attributes),
            distillation_pass=get_distillation_pass(host),
        )

        if use_stack and stack is not None:
            stack.append(span)

        if emit_open:
            _emit(sink, span)
        return span
    except Exception as exc:
        logger.warning(f"start_span({name}) failed: {exc!r}")
        return None


def end_span(
    host: Any,
    span: Optional[Span],
    *,
    status: str = STATUS_OK,
    attributes: Optional[dict[str, Any]] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    close: bool = True,
) -> None:
    """Finish (or, with ``close=False``, update-in-place) a span and emit it.
    Never raises. A None span (start_span declined) is a silent no-op."""
    if span is None:
        return
    try:
        if close:
            span.end_ns = time.time_ns()
        span.status = status
        if attributes:
            span.attributes.update(_capped(attributes))
        if command_name is not None:
            span.command_name = command_name
        if context is not None:
            span.context = context

        stack = _get_stack(host)
        if stack is not None and span in stack:
            stack.remove(span)

        sink = get_sink(host)
        if sink is not None:
            _emit(sink, span)
    except Exception as exc:
        logger.warning(f"end_span({span.name}) failed: {exc!r}")


DISTILLATION_RECORD_KINDS = frozenset(
    {"run", "pass", "divergence", "insight", "citation"}
)


def emit_distillation_record(host: Any, kind: str, payload: dict[str, Any]) -> bool:
    """Hand one distillation row to the host's sink ([DR46]). Never raises.

    Returns True when a sink accepted it. The sink queues the write on its own
    writer thread, so this is the only path a live turn may use — a direct
    ``ObservabilityStore._connect()`` on the turn thread would put lock
    contention in front of ``_execute_message``, which [R14] forbids.
    """
    try:
        if kind not in DISTILLATION_RECORD_KINDS:
            logger.warning(f"emit_distillation_record: unknown kind {kind!r}")
            return False
        sink = get_sink(host)
        if sink is None:
            return False
        emit = getattr(sink, "emit_distillation_record", None)
        if emit is None:  # a foreign sink predating [DR46]
            return False
        emit(kind, payload)
        return True
    except Exception as exc:
        logger.warning(f"TraceSink.emit_distillation_record({kind}) failed: {exc!r}")
        return False


def emit_turn_record(host: Any, record: Any) -> Optional[bool]:
    """Hand the finalized internal TurnResult to the sink. Never raises.

    Returns the sink's "stored" ack (Phase 7 ruling I1), or None when there is
    no sink at all. The three values are distinct on purpose: True and False
    both mean a durable store exists and say whether the row reached it, while
    None means nothing is recording turns, so a caller gating on durability has
    nothing to wait for and must not defer forever.
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        return bool(sink.emit_turn_record(record))
    except Exception as exc:
        logger.warning(f"TraceSink.emit_turn_record failed: {exc!r}")
        return False
