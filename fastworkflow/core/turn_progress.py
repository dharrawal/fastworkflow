"""Stage progress of one turn: which model-backed stage is running (fix-kq49).

A turn that carries a multi-part request does several model-backed stages before the first
sub-task review question: the split, the referents, the match of each sub-task to a command, the
draft of each sub-task's arguments. ``emit`` announces the START of one of them on the session's
command trace queue (the channel the live ``trace`` frames already travel on), as a
``ProgressEvent``. Consumers that do not know the type drop it: ``collect_trace_events`` and the
CLI skip it, and the streaming endpoint sends it as a ``progress`` frame, which a client that
does not know the frame type ignores.

The payload is the stage key, a short generic label and counts. Never user text, model text or a
workflow's own names.

Emission is best-effort and costs one context-variable read when nothing listens. The listener is
bound per turn in a ``ContextVar`` (``bound``), so two sessions that share a workflow never see
each other's stages. A worker thread that is not started inside the turn's context sees no
listener; run it under ``contextvars.copy_context().run`` to keep its events.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from fastworkflow.utils.logging import logger

STAGE_SPLIT = "split"
STAGE_REFERENTS = "referents"
STAGE_MATCH = "match"
STAGE_DRAFT = "draft"
STAGE_REVIEW = "review"

_LABELS = {
    STAGE_SPLIT: "Splitting the request",
    STAGE_REFERENTS: "Working out what each sub-task refers to",
    STAGE_MATCH: "Matching sub-task",
    STAGE_DRAFT: "Drafting sub-task",
    STAGE_REVIEW: "Preparing the sub-task review",
}


@dataclass
class ProgressEvent:
    """The start of one stage of the turn. ``index`` and ``total`` are set when the stage runs once per sub-task."""
    stage: str
    label: str
    timestamp_ms: int
    index: Optional[int] = None
    total: Optional[int] = None

    def to_dict(self) -> dict:
        data = {"stage": self.stage, "label": self.label, "timestamp_ms": self.timestamp_ms}
        if self.index is not None:
            data["index"] = self.index
        if self.total is not None:
            data["total"] = self.total
        return data


_listener: contextvars.ContextVar[Optional[Callable[[ProgressEvent], None]]] = contextvars.ContextVar(
    "fastworkflow_turn_progress_listener", default=None)


def label_for(stage: str, index: Optional[int] = None, total: Optional[int] = None) -> str:
    base = _LABELS.get(stage, stage)
    if index is not None and total is not None and total > 1:
        return f"{base} {index} of {total}"
    return base


@contextlib.contextmanager
def bound(context) -> Iterator[None]:
    """Listen for this turn's stages on ``context``'s command trace queue, if it has one."""
    queue = getattr(context, "command_trace_queue", None)
    token = _listener.set(queue.put if queue is not None else None)
    try:
        yield
    finally:
        _listener.reset(token)


def emit(stage: str, index: Optional[int] = None, total: Optional[int] = None) -> None:
    """Announce that ``stage`` starts. Never raises; does nothing when no listener is bound."""
    listener = _listener.get()
    if listener is None:
        return
    try:
        listener(ProgressEvent(stage, label_for(stage, index, total), int(time.time() * 1000), index, total))
    except Exception as exc:  # a listener must never break the turn
        logger.debug(f"turn progress event '{stage}' dropped: {exc!r}")
