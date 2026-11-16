"""The ``fw.compose`` span: one trace record per compose call (fix-weih.28).

The composer (``fastworkflow.run_chatbot.composer``) renders and verifies; this module is the transport side that
records what a call did, so a composed answer can be inspected beside its turn in the trace viewer. It lives next to
the server routes rather than in the composer because the composer must not import core tracing: rendering stays in
``run_chatbot``, and the server, which already owns the observability sink, decides what reaches the store.

A compose request arrives after its turn finished, on its own HTTP request, so there is no live trace host. The span
is emitted through a minimal duck-typed host (``tracing`` discovers a sink, a turn key and a span stack by
attribute) and filed under the turn the page named, or under its own ``compose-<id>`` trace when the page named
none or named a turn the caller's channel does not own.

Attributes are counts, short reason codes and labels. No user text, no answer text, no result values and no raw
model output: those stay out of the store (the raw output is DEBUG-log only, in the composer).
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastworkflow import tracing
from fastworkflow.utils.logging import logger

UNATTACHED_TRACE_PREFIX = "compose-"


class _ComposeHost:
    """Just enough of a trace host for ``tracing.start_span``: a sink, a trace id, a channel and an empty stack."""

    def __init__(self, sink: Any, trace_id: str, channel_id: Optional[str]) -> None:
        self.trace_sink = sink
        self.current_turn_key = trace_id
        self.observability_channel_id = channel_id
        self.trace_span_stack: list = []


def _compose_attributes(
    kind: str, outcome: str, results: int, duration_ms: int, model: str, attached: bool, diag: Any
) -> dict[str, Any]:
    """The span's attributes. A literal dict on purpose: the span contract test reads the emitted keys from it."""
    second = getattr(diag, "second", None)
    return {
        "compose_kind": kind,
        "outcome": outcome,
        "results": results,
        "model": model,
        "duration_ms": duration_ms,
        "turn_attached": attached,
        "digest_chars": getattr(diag, "digest_chars", None),
        "answer_chars": getattr(diag, "answer_chars", None),
        "excerpt_chars": getattr(diag, "excerpt_chars", None),
        "collapsed": getattr(diag, "collapsed", None),
        "sections_kept": getattr(diag, "sections_kept", None),
        "sections_proposed": getattr(diag, "sections_proposed", None),
        "blocks_kept": getattr(diag, "blocks_kept", None),
        "blocks_proposed": getattr(diag, "blocks_proposed", None),
        "lead": getattr(diag, "lead", None),
        "drops": dict(diag.drops) if diag is not None else None,
        "retry": getattr(diag, "retry", None),
        "sections_unnested": getattr(diag, "sections_unnested", None),
        "refs_repaired": getattr(diag, "refs_repaired", None),
        "sections_merged": getattr(diag, "sections_merged", None),
        "truncated": getattr(diag, "truncated", None),
        "token_limit": getattr(diag, "token_limit", None),
        "second_pass": None if second is None else {
            "drops": dict(second.drops),
            "sections_kept": second.sections_kept,
            "sections_proposed": second.sections_proposed,
            "blocks_kept": second.blocks_kept,
            "blocks_proposed": second.blocks_proposed,
        },
    }


def emit_compose_span(
    sink: Any,
    *,
    kind: str,
    channel_id: Optional[str],
    turn_key: Optional[str],
    started_ns: int,
    outcome: str,
    results: int,
    duration_ms: int,
    model: str,
    diag: Any,
) -> Optional[str]:
    """Write one ``fw.compose`` span; the trace id it was filed under, or None when nothing was written.

    ``turn_key`` is a trace id the caller already authorised (or None). Never raises: tracing must not fail or
    slow a compose response, so a failure is one log line.
    """
    try:
        if sink is None:
            return None
        attached = bool(turn_key)
        trace_id = turn_key or f"{UNATTACHED_TRACE_PREFIX}{uuid.uuid4().hex}"
        host = _ComposeHost(sink, trace_id, channel_id)
        span = tracing.start_span(host, tracing.SPAN_COMPOSE, kind=tracing.KIND_INTERNAL)
        if span is None:
            return None
        span.start_ns = started_ns
        if not attached:
            span.parent_span_id = None   # there is no turn root to hang under
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK if outcome in ("ok", "not_needed") else tracing.STATUS_ERROR,
            attributes=_compose_attributes(kind, outcome, results, duration_ms, model, attached, diag),
        )
        return trace_id
    except Exception as exc:  # noqa: BLE001 - tracing never fails the response
        logger.warning(f"compose span not recorded: {exc!r}")
        return None
