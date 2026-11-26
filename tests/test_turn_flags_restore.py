"""A session saved in the middle of a turn restores the turn's flags (bd fix-fpyb.3).

The sub-task review state, the turn's stops (``fw_not_run``) and the unattended mark are per-turn state: a turn
resumed in another process must behave as the live one would. Real components throughout: the annotated library copy,
its real agent and tools, and the scripted agent steps of the round-1 tests.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow.observability import tracing
from fastworkflow import waiting as waiting_module
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - fixtures
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_planning_command_phases import RecordingSink  # noqa: F401
from tests.test_round1_gate import EMMA, JOKE, _marker, _open_review, _split_into, gated  # noqa: F401
from tests.test_round1_handoff import _planner_stub, _scripted_context  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def _saved(ctx) -> dict:
    return json.loads(json.dumps(ctx.serialize_state(channel_id="turn-flags")))


def _suspended_in_an_accepted_review(annotated_library, monkeypatch):
    """A reviewed turn: the review was accepted and the first goal's agent then asked the user a question."""
    _planner_stub(monkeypatch, [])
    _split_into(monkeypatch, [EMMA, JOKE])
    first = _scripted_context(annotated_library, monkeypatch)
    first.ctx.process_turn(f"{EMMA} and {JOKE}")
    _open_review(type("S", (), {"store": __import__("fastworkflow.approval_binding", fromlist=["x"]).store_for(first.wf)}))
    first.script.steps = [("finish", {}), ("ask_user", {"clarification_request": "Which copy?"})]
    asked = first.ctx.process_turn("accept")
    assert asked.status is TurnStatus.AWAITING_USER
    assert first.ctx._turn_subtask_review == "reviewed"
    return first


def test_the_review_state_of_a_turn_survives_a_restore(annotated_library, monkeypatch):
    first = _suspended_in_an_accepted_review(annotated_library, monkeypatch)
    state = _saved(first.ctx)

    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    second.script.steps = [("finish", {})] * 3
    out = second.ctx.process_turn("the second one")
    assert out.status is TurnStatus.COMPLETED
    # Span contract 63: the review's outcome is on the turn's fw.review span, which the first process wrote before it
    # saved; the resumed segment reopens only the stage that was open (the run). The turn's tree, as a store holding
    # both processes' emissions reads it, says what the live process would have said.
    turn_ids = {span.trace_id for sink in (first.sink, second.sink) for span in sink.spans
                if span.name == tracing.SPAN_TURN}
    assert len(turn_ids) == 1                       # one turn, continued in the second process
    both = type("Both", (), {"sink": type("Sink", (), {"spans": [*first.sink.spans, *second.sink.spans]})()})()
    assert _marker(both) == "reviewed"
    assert second.ctx._turn_subtask_review == "reviewed"


def test_a_refused_typed_call_of_a_reviewed_turn_is_not_asked_after_a_restore(annotated_library, monkeypatch):
    first = _suspended_in_an_accepted_review(annotated_library, monkeypatch)
    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(_saved(first.ctx))
    # The guard of _open_parameter_loop: a turn resolved before its review is not asked again while it runs.
    assert second.ctx._turn_subtask_review not in ("not_applicable", "fast_path")


def test_the_stops_of_a_turn_survive_a_restore(annotated_library, monkeypatch):
    first = _suspended_in_an_accepted_review(annotated_library, monkeypatch)
    stop = {"kind": "not_run_here", "command": None, "phase": None, "open_step": None, "done": [], "facts": {},
            "remaining": [], "suggested_plan": "", "means": "from_scratch", "sub_task": 2, "reason": "cancelled",
            "request": JOKE}
    first.ctx._turn_not_run.append(dict(stop))
    state = _saved(first.ctx)

    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    second.script.steps = [("finish", {})] * 3
    out = second.ctx.process_turn("the second one")
    carried = [entry
               for output in out.command_outputs
               for entry in (output.command_response.artifacts.get("fw_not_run") or [])]
    assert [entry["sub_task"] for entry in carried] == [2]


def test_an_unattended_turn_stays_unattended_after_a_restore(annotated_library, monkeypatch):
    first = _scripted_context(annotated_library, monkeypatch)
    first.ctx._begin_turn("tell Ann a joke", unattended=True)
    state = _saved(first.ctx)

    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    assert second.ctx._turn_unattended is True
    second.ctx._enable_all_tools()
    assert "ask_user" in second.ctx._workflow_tool_agent.disabled_tools    # nobody is there to answer
    assert waiting_module.can_review(second.ctx) is False


def test_a_blob_without_the_keys_restores_the_defaults(annotated_library, monkeypatch):
    first = _scripted_context(annotated_library, monkeypatch)
    first.ctx._begin_turn("tell Ann a joke")
    state = _saved(first.ctx)
    for key in ("subtask_review", "not_run", "unattended"):
        state["turn"].pop(key, None)
    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    assert second.ctx._turn_subtask_review == "not_applicable"
    assert second.ctx._turn_not_run == [] and second.ctx._turn_unattended is False
