"""The turn announces which model-backed stage is running (fix-kq49).

Integration tests on the annotated library copy with the real agent and the real review gate (the fixtures of
``test_round1_gate``). The one scripted seam is the planner's split, as in the gate tests. Nothing here mocks the
listener path: events travel the context's real command trace queue and the real streaming drain.
"""
from __future__ import annotations

import asyncio
import queue
import threading

import pytest

from fastworkflow import turn_progress
from fastworkflow.run_fastapi_mcp.utils import (
    _deliver_trace_or_progress, collect_trace_events, run_process_message_with_trace_stream)
from fastworkflow.turn import TurnStatus
from fastworkflow.turn_progress import ProgressEvent
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _drain(q: queue.Queue) -> list:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            return events


def test_the_stages_of_a_turn_that_opens_a_review_are_announced_in_order(gated, monkeypatch):
    events_queue: queue.Queue = queue.Queue()
    gated.ctx.set_transport_queues(command_trace_queue=events_queue)
    _split_into(monkeypatch, [EMMA, JOKE])

    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")

    assert out.status is TurnStatus.AWAITING_USER
    progress = [event for event in _drain(events_queue) if isinstance(event, ProgressEvent)]
    stages = [(event.stage, event.index, event.total) for event in progress]
    assert stages[0] == ("split", None, None)
    assert stages[1:3] == [("match", 1, 2), ("match", 2, 2)]
    assert stages[3:5] == [("draft", 1, 2), ("draft", 2, 2)]
    assert stages[-1] == ("review", None, None)
    assert progress[1].label == "Matching sub-task 1 of 2"
    assert progress[3].label == "Drafting sub-task 1 of 2"
    # Counts and generic labels only: no part of the user's request is in a payload.
    for event in progress:
        text = repr(event.to_dict())
        assert "Emma" not in text and "joke" not in text
    stamps = [event.timestamp_ms for event in progress]
    assert stamps == sorted(stamps)


def test_a_turn_with_no_queue_announces_nothing_and_still_runs(gated, monkeypatch):
    assert gated.ctx.command_trace_queue is None
    _split_into(monkeypatch, [EMMA, JOKE])
    assert gated.ctx.process_turn(f"{EMMA} and {JOKE}").status is TurnStatus.AWAITING_USER


def test_a_listener_that_raises_does_not_break_the_turn(gated, monkeypatch):
    class Broken(queue.Queue):
        """A queue whose progress listener fails; command traces still land, so only the new path is under test."""
        def put(self, event, *args, **kwargs):
            if isinstance(event, ProgressEvent):
                raise RuntimeError("the listener is gone")
            return super().put(event, *args, **kwargs)

    gated.ctx.set_transport_queues(command_trace_queue=Broken())
    _split_into(monkeypatch, [EMMA, JOKE])
    with turn_progress.bound(gated.ctx):
        turn_progress.emit("split")              # raises inside the listener, swallowed by emit
    assert gated.ctx.process_turn(f"{EMMA} and {JOKE}").status is TurnStatus.AWAITING_USER


def test_emission_from_a_worker_thread_with_the_turns_context(gated):
    events_queue: queue.Queue = queue.Queue()
    gated.ctx.set_transport_queues(command_trace_queue=events_queue)
    import contextvars

    with turn_progress.bound(gated.ctx):
        copied = contextvars.copy_context()
        threads = [threading.Thread(target=copied.copy().run, args=(turn_progress.emit, "draft", number, 3))
                   for number in (1, 2, 3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert sorted(event.index for event in _drain(events_queue)) == [1, 2, 3]
    # A thread started outside the turn's context sees no listener and emits nothing.
    turn_progress.emit("draft", 1, 3)
    assert events_queue.empty()


def test_the_stream_delivers_progress_to_on_progress_and_never_to_on_trace(gated, monkeypatch):

    events_queue: queue.Queue = queue.Queue()
    gated.ctx.set_transport_queues(command_trace_queue=events_queue)
    _split_into(monkeypatch, [EMMA, JOKE])
    traces, progress = [], []

    class Runtime:
        execution_context = gated.ctx
        channel_id = "kq49"

    class Sessions:
        def persist(self, *args, **kwargs):
            return None

    async def go(on_progress):
        import fastworkflow.run_fastapi_mcp.utils as utils
        monkeypatch.setattr(utils, "persist_pending_after_turn", lambda *a, **k: None)
        return await run_process_message_with_trace_stream(
            Runtime(), f"{EMMA} and {JOKE}", 120, Sessions(), traces.append, on_progress=on_progress)

    out = asyncio.run(go(progress.append))
    assert out.status is TurnStatus.AWAITING_USER
    assert [frame["stage"] for frame in progress][0] == "split"
    assert progress[-1]["stage"] == "review"
    assert all("direction" in trace for trace in traces)            # only command traces reach on_trace


def test_progress_is_dropped_by_the_consumers_that_do_not_know_it(gated):
    events_queue: queue.Queue = queue.Queue()
    gated.ctx.set_transport_queues(command_trace_queue=events_queue)
    with turn_progress.bound(gated.ctx):
        turn_progress.emit("split")
    assert len(events_queue.queue) == 1

    class Runtime:
        execution_context = gated.ctx

    assert collect_trace_events(Runtime()) == []                   # the polling/non-stream path
    emitted = []
    asyncio.run(_deliver_trace_or_progress(ProgressEvent("split", "x", 0), emitted.append, None, None))
    assert emitted == []                                           # no on_progress: silently dropped

    async def failing(_data):
        raise RuntimeError("socket closed")

    asyncio.run(_deliver_trace_or_progress(ProgressEvent("split", "x", 0), emitted.append, failing, None))
    assert emitted == []                                           # a failing on_progress is swallowed
