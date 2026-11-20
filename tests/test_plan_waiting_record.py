"""The plan path's questions are the turn's ONE waiting record (bd fix-fpyb.8, step 5 of ``docs/pending_state_design.md``).

The which-command question, a bound plan's values question at the gate, and the values question a running part puts are
``WorkflowExecutionContext._waiting`` records of kind ``intent`` and ``parameters`` (owner ``plan``; ``after`` is
``resolve_next`` at the gate and ``resume_part`` while a part runs). ``PlanResolution`` holds the resolution's progress
only. Each test drives a real session to the question, checks the record, saves and restores the session, and then
answers it (by ``cancel`` and by ``/cancel`` where the reply is a cancel). Real contexts on the real test workflows.
"""
from __future__ import annotations

import dataclasses
import json

import dspy
import pytest

from fastworkflow import parameter_clarification as loops
from fastworkflow import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow._workflows.plan_metadata_extraction.resolution import PlanResolution
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import build_query_with_next_steps as REAL_BUILD
from fastworkflow.workflow_agent import load_planning_command_map
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT
from tests.conftest import script_matcher
from tests.test_approval_binding import _finish, annotated_library, todo  # noqa: F401 - fixtures
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers
    A_NOTE, ABOUT_DUNE, ABOUT_EMMA, BOOK_PLAN, HOST, MEMBER, NOTE, NOTE_PLAN, _as_a_new_process, _loop, _reply, _TextPlanner,
    no_agent)
from tests.test_round1_gate import JOKE, gated  # noqa: F401 - fixture
from tests.test_step_helpers import library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.test_waiting_kinds import _last, _no_agent_joke_decisions

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

#: What the resolution held while a question was out, before step 5: none of it is a field of it now.
_REMOVED = {"pending", "asking", "choice", "loop", "at_gate"}


def _holds_no_question(ctx) -> None:
    resolution = ctx._plan_resolution
    assert resolution is None or not _REMOVED & {each.name for each in dataclasses.fields(resolution)}
    assert not _REMOVED & set(PlanResolution().to_state())
    assert not _REMOVED & set(ctx.serialize_state(channel_id="plan-record").get("plan_resolution") or {})


def _restored(ctx, channel: str = "plan-record") -> dict:
    """Save the session, drop what is live and bring back only what was saved; returns the saved blob."""
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id=channel)))
    _as_a_new_process(ctx, blob)
    return blob


# ---- which command -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cancel", ["cancel", "/cancel"])
def test_a_which_command_question_is_a_record_that_survives_a_restore_and_a_cancel_ends_it(no_agent, monkeypatch, cancel):
    _no_agent_joke_decisions(no_agent, monkeypatch)
    no_agent.parts = [JOKE]
    ctx = no_agent.ctx
    asked = ctx.process_turn(JOKE, approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER and _last(asked)["question_kind"] == "plan_choice"

    held = ctx._waiting
    assert (held.kind, held.owner, held.question_kind) == ("intent", "plan", "plan_choice")
    assert held.payload["part"] == 0 and held.payload["record"]["origin"] == 0 and "after" not in held.payload
    assert held.question == asked.answer and ctx._resolution_pending and ctx.waiting is held
    _holds_no_question(ctx)

    blob = _restored(ctx)

    assert blob["waiting"]["kind"] == "intent" and blob["waiting"]["owner"] == "plan"
    assert ctx._waiting is not None and ctx._waiting.payload["record"] == held.payload["record"]
    assert ctx._resolution_pending and ctx._pending_prompt_kind(ctx._turn_outputs[-1])[0] == "plan_choice"

    done = ctx.process_turn(cancel, approval_surface=HOST)

    assert done.status is TurnStatus.COMPLETED
    (stop,) = [entry for entry in _last(done)[NOT_RUN_ARTIFACT] if entry["command"] is None]
    assert stop["reason"] == "it was declined when asked which planning command should run it"
    assert ctx._waiting is None and ctx.waiting is None and not ctx._resolution_pending


# ---- a bound plan's values, at the gate --------------------------------------------------------------------------------

@pytest.mark.parametrize("cancel", ["cancel", "/cancel"])
def test_a_values_question_at_the_gate_is_a_record_that_survives_a_restore_and_a_cancel_drops_the_part(no_agent, cancel):
    ctx = no_agent.ctx
    asked = ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER and _last(asked)["question_kind"] == "planning_parameter"

    held = ctx._waiting
    assert (held.kind, held.owner, held.question_kind) == ("parameters", "plan", "planning_parameter")
    assert held.payload["after"] == loops.AFTER_RESOLVE_NEXT and held.payload["part"] == 1
    assert held.question == asked.answer and held.payload["loop"]["command"] == NOTE_PLAN
    assert ctx._plan_resolution.index == 1
    _holds_no_question(ctx)

    blob = _restored(ctx)

    assert blob["waiting"]["payload"]["after"] == "resolve_next" and blob["plan_resolution"]["index"] == 1
    assert _loop(ctx) == loops.ParameterLoop.from_state(held.payload["loop"])
    assert ctx._resolution_pending and ctx._pending_prompt_kind(ctx._turn_outputs[-1])[0] == "planning_parameter"

    review = ctx.process_turn(cancel, approval_surface=HOST)

    assert review.status is TurnStatus.AWAITING_USER and _last(review)["question_kind"] == "review"
    assert [item["goal"] for item in _last(review)["review"]["items"]] == [ABOUT_EMMA, ABOUT_DUNE]
    # The review is the turn's record now (it points at the approval store's record; step 8).
    assert ctx.waiting is ctx._waiting and ctx._waiting.kind == "review" and ctx._plan_resolution is None


def test_the_reply_to_a_restored_values_question_completes_the_part_as_it_did_before(no_agent):
    ctx = no_agent.ctx
    ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    _restored(ctx)

    review = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")

    assert review.status is TurnStatus.AWAITING_USER and _last(review)["question_kind"] == "review"
    assert ctx._waiting.kind == "review" and ctx._plan_resolution is None


def test_a_values_question_asked_again_is_put_back_as_the_one_record(no_agent):
    ctx = no_agent.ctx
    ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)

    again = _reply(no_agent, f"member: {MEMBER}")        # progress: asked again for the text alone

    assert again.status is TurnStatus.AWAITING_USER and _last(again)["question_kind"] == "planning_parameter"
    held = ctx._waiting
    assert (held.kind, held.payload["after"], held.payload["loop"]["open"]) == ("parameters", "resolve_next", ["text"])
    assert held.question == again.answer and ctx._plan_resolution.index == 1


# ---- the values question a running part puts ---------------------------------------------------------------------------

def _refused_book_part(todo, monkeypatch):
    """With an agent model the typed planning call of the one part is refused for its missing title: the question is
    put while the part runs (the setup of ``tests/test_waiting_kinds.py``)."""
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    request = "who wrote that book"
    command = load_planning_command_map(todo.wf.folderpath)[BOOK_PLAN]
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {p.name: None for p in command.params}, request)

    def refuse(planning_command, request_text, *args, **kwargs):
        raise refused.value

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda *args, **kwargs: None)
    script_matcher(monkeypatch, lambda text: {command.name: 0.9, "none": 0.0})
    monkeypatch.setattr(workflow_agent, "typed_planning_call", refuse)
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: [user_query])
    planner = _TextPlanner([], adapter=dspy.JSONAdapter())
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda *args, **kwargs: REAL_BUILD(*args, **{**kwargs, "planner_lm": planner}))
    return request


def test_a_value_question_asked_while_a_part_runs_is_the_same_kind_resumed_in_place(todo, monkeypatch):
    request = _refused_book_part(todo, monkeypatch)
    ctx = todo.ctx

    first = ctx.process_turn(request)

    assert first.status is TurnStatus.AWAITING_USER and _last(first)["question_kind"] == "planning_parameter"
    held = ctx._waiting
    assert (held.kind, held.owner, held.question_kind) == ("parameters", "plan", "planning_parameter")
    assert held.payload["after"] == loops.AFTER_RESUME_PART and held.payload["part"] == 0
    assert held.question == first.answer and ctx._resolution_pending
    # No resolution exists for this question (the turn's one part was not resolved before a review).
    assert ctx._plan_resolution is None
    assert ctx._part_loop is not None and ctx._part_loop.asks == 1

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="part-record")))

    assert blob["waiting"]["payload"]["after"] == "resume_part" and blob["plan_resolution"] is None
    assert blob["part_loop"]["asks"] == 1
    restored = loops.ParameterLoop.from_state(blob["waiting"]["payload"]["loop"])
    assert restored.command == BOOK_PLAN and restored.origin == 0
    _as_a_new_process(ctx, blob)
    assert ctx._resolution_pending and ctx._part_question_out and ctx._part_loop.asks == 1

    todo.script.steps = _finish()
    done = ctx.process_turn("title: Dune")

    # The reply resumed the part in place: the record left the slot, the loop is kept for the asking rule's count. The
    # typed call is mocked to refuse the title again, so (step 6, consequence b) the question is asked a second time.
    assert done.status is TurnStatus.AWAITING_USER
    assert ctx._waiting is not None and ctx._part_loop.asks == 2
