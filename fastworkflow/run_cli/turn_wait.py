"""The terminal's wait for one turn: read the trace queue to the sentinel, stoppable by Ctrl-C or a deadline.

Kept apart from ``run_cli/__main__.py`` (which needs prompt_toolkit and rich) so a test can drive the wait with a real
``ChatSession`` and no terminal. The wait is the loop ``fastworkflow run_cli`` always had: one blocking read of the
trace queue per 0.1 s tick. The deadline is one clock comparison per tick, and there is no watcher thread, so a
turn that is not interrupted and has no deadline costs what it did before.
"""
from __future__ import annotations

import queue
import time
from enum import Enum
from typing import Callable, Optional


class WaitOutcome(str, Enum):
    DONE = "done"            # the turn ended (its sentinel was read); its output is on the output queue
    INTERRUPTED = "interrupted"  # Ctrl-C: the turn was abandoned (its output is dropped)
    TIMED_OUT = "timed_out"  # the deadline passed: the turn was abandoned


#: What both messages say, because it is the guarantee (see ChatSession.abandon_turn): the wait stopped, the work did not.
ABANDONED_NOTE = ("stopped waiting, but the turn is still finishing in the background and its commands may still "
                  "take effect; its output will be discarded. Your next message runs after it.")


def interrupted_message() -> str:
    return f"Interrupted: {ABANDONED_NOTE}"


def timed_out_message(seconds: float) -> str:
    return f"Turn passed the {seconds:g}s limit (--timeout_seconds): {ABANDONED_NOTE}"


def wait_for_turn(
    chat_session,
    on_event: Callable[[object], None],
    timeout_seconds: Optional[float] = None,
    on_tick: Optional[Callable[[int], None]] = None,
    tick: float = 0.1,
) -> WaitOutcome:
    """Read the trace queue until the turn's sentinel, handing each event to ``on_event``.

    ``on_tick(n)`` is called after each empty tick (the spinner's elapsed seconds). Returns DONE at the sentinel.
    On Ctrl-C, or when ``timeout_seconds`` (None: no limit) has passed, the turn is abandoned
    (``ChatSession.abandon_turn``) and INTERRUPTED / TIMED_OUT returned; when the turn had in fact just ended,
    the wait continues to its sentinel and returns DONE. The clock starts at the call, so it includes the time
    an earlier abandoned turn still takes to finish.
    """
    trace_queue = chat_session.command_trace_queue
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    ticks = 0
    while True:
        try:
            if deadline is not None and time.monotonic() >= deadline:
                deadline = None  # asked once: if the turn had just ended, read it out
                if chat_session.abandon_turn():
                    return WaitOutcome.TIMED_OUT
            try:
                event = trace_queue.get(timeout=tick)
            except queue.Empty:
                ticks += 1
                if on_tick is not None:
                    on_tick(ticks)
                continue
            if event is None:
                return WaitOutcome.DONE
            on_event(event)
        except KeyboardInterrupt:
            if chat_session.abandon_turn():
                return WaitOutcome.INTERRUPTED
