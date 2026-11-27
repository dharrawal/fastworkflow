"""A "/" message sent while a question is out is the reply to it, with the leading "/" removed (bd fix-fpyb.7, step 2 of
``docs/pending_state_design.md``, owner decision 2026-11-19).

One test per pending kind, on a real session driven to the question as ``tests/test_waiting_kinds.py`` drives it. Each
sends the kind's own cancel reply (or, for an agent question, a plain answer) with a leading "/", and checks two things:
the message was READ as that reply (the question ended or moved on exactly as the plain reply does), and it did not start
a command of its own: ``_process_message`` (the deterministic path) is never entered by a plan, agent, review, approval
or decision question, and on the CME's own questions the "assistant command" flag is never set, so nothing is appended
to the suspended turn beside the question. ``/approvals`` and ``/cancel_pending`` keep their meaning (last tests).
"""
from __future__ import annotations

import pytest

import fastworkflow
from fastworkflow.review import approval_binding as ab
from fastworkflow import workflow_agent
from fastworkflow.core.turn import TurnStatus
from fastworkflow.engine.workflow_execution_context import (
    AWAITING_ANSWER_REFUSAL_MESSAGE, NOT_RUN_ARTIFACT, UsePlanError, WorkflowExecutionContext)
from tests.test_approval_binding import _ask, _finish, annotated_library, todo  # noqa: F401 - fixtures
from tests.test_no_agent_decisions import Conditional
from tests.test_no_agent_mode import _plan
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401 - the CME on plain_required_workflow
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers
    A_NOTE, ABOUT_DUNE, ABOUT_EMMA, HOST, no_agent)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated, remote_decides_emma_and_joke  # noqa: F401
from tests.test_step_helpers import _build_planning_command, library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.test_waiting_kinds import _last, _no_agent_joke_decisions, no_param_model  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture
def deterministic_runs(monkeypatch):
    """The messages the deterministic path (``_process_message``) was entered with, the real one still running."""
    seen: list[str] = []
    real = WorkflowExecutionContext._process_message

    def spy(self, message):
        seen.append(message)
        return real(self, message)

    monkeypatch.setattr(WorkflowExecutionContext, "_process_message", spy)
    return seen


def test_a_slash_message_answers_an_agent_question(todo, deterministic_runs, monkeypatch):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    assert todo.ctx.process_turn("find a book").status is TurnStatus.AWAITING_USER
    outputs = len(todo.ctx._turn_outputs)
    replies: list[str] = []
    real = workflow_agent._post_ask_user_response
    monkeypatch.setattr(workflow_agent, "_post_ask_user_response",
                        lambda request, answer, ctx: replies.append(answer) or real(request, answer, ctx))
    todo.script.steps = _finish()

    done = todo.ctx.process_turn("/the red one")

    assert replies == ["the red one"] and deterministic_runs == []
    assert done.status is TurnStatus.COMPLETED and todo.ctx.waiting is None
    assert not any(output.command_name == "the" for output in todo.ctx._turn_outputs[outputs:])


def test_a_slash_message_answers_a_plans_which_command_question(no_agent, monkeypatch, deterministic_runs):
    _no_agent_joke_decisions(no_agent, monkeypatch)
    no_agent.parts = [JOKE]
    ctx = no_agent.ctx
    assert _last(ctx.process_turn(JOKE, approval_surface=HOST))["question_kind"] == "plan_choice"

    done = ctx.process_turn("/cancel", approval_surface=HOST)

    assert deterministic_runs == [] and done.status is TurnStatus.COMPLETED
    (stop,) = [entry for entry in _last(done)[NOT_RUN_ARTIFACT] if entry["command"] is None]
    assert stop["reason"] == "it was declined when asked which planning command should run it"
    assert ctx.waiting is None


def test_a_slash_message_answers_a_plans_value_question_at_the_gate(no_agent, deterministic_runs):
    ctx = no_agent.ctx
    asked = ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    assert _last(asked)["question_kind"] == "planning_parameter"

    review = ctx.process_turn("/cancel", approval_surface=HOST)

    assert deterministic_runs == []
    assert _last(review)["question_kind"] == "review"
    assert [item["goal"] for item in _last(review)["review"]["items"]] == [ABOUT_EMMA, ABOUT_DUNE]


def test_a_slash_message_answers_a_sub_task_review(gated, no_param_model, monkeypatch, deterministic_runs, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    assert gated.ctx.process_turn(f"{EMMA} and {JOKE}").status is TurnStatus.AWAITING_USER
    _open_review(gated)

    done = gated.ctx.process_turn("/no")

    assert deterministic_runs == []
    assert done.status is TurnStatus.COMPLETED and done.answer.startswith("You declined the sub-task review")
    assert gated.ctx.waiting is None and ab.active_reviews(gated.wf) == []


def test_a_slash_message_answers_an_action_approval(todo, deterministic_runs):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    todo.script.steps = _finish()

    done = todo.ctx.process_turn("/no")

    assert deterministic_runs == [] and done.status is TurnStatus.COMPLETED
    assert todo.ctx.waiting is None and ab.active_approvals(todo.wf) == []
    assert todo.chores.notes == todo.base, "a declined approval never runs the command"


def test_a_slash_message_answers_a_decision_step(library_env, no_agent_model, no_planner_key, monkeypatch,
                                                 deterministic_runs):
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    assert _last(library_env.ctx.process_turn("look up the book Dune"))["question_kind"] == "phase_step"

    done = library_env.ctx.process_turn("/cancel")

    assert deterministic_runs == [] and done.status is TurnStatus.COMPLETED
    (stop,) = _last(done)[NOT_RUN_ARTIFACT]
    assert stop["open_step"]["reason"].startswith("declined")
    assert library_env.ctx.waiting is None


def test_a_slash_message_answers_the_open_commands_value_question(session, deterministic_runs):
    ctx = session(False)
    assert _last(ctx.process_turn(PARTIAL)).get("parameter_request") is not None

    done = ctx.process_turn("/abort")

    assert "aborted" in done.answer and ctx.waiting is None and not ctx.has_open_command()
    assert not ctx.assistant_round


def test_a_slash_message_answers_the_cmes_which_command_question(session, deterministic_runs):
    ctx = session(False)
    ctx.process_turn(PARTIAL)
    assert _last(ctx.process_turn("you misunderstood")).get("plan_choice") is not None

    done = ctx.process_turn("/abort")

    assert "aborted" in done.answer and ctx.waiting is None and not ctx.has_open_command()
    assert not ctx.assistant_round


def test_with_nothing_out_a_slash_message_is_still_a_deterministic_command(session):
    ctx = session(False)
    ctx.process_turn("/" + PARTIAL)
    # The round was started by the "/" message: its question belongs to the assistant path.
    assert ctx.waiting is not None and ctx.waiting.kind == "parameters"
    assert ctx.assistant_round and ctx.waiting.assistant


def test_the_management_messages_keep_their_meaning_while_a_question_is_out(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    todo.ctx.process_turn("find a book")

    todo.ctx.process_turn("/approvals")

    # Answered as a management message: the question is untouched and nothing reached the agent.
    assert todo.ctx.waiting is not None and todo.ctx.waiting.kind == "agent"
    assert todo.ctx.cancel_pending() is True


def test_process_action_and_use_plan_share_one_refusal_while_a_question_is_out(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    todo.ctx.process_turn("find a book")

    refused = todo.ctx.process_action(fastworkflow.Action(command_name="list_books", command="list_books"))
    assert refused.command_response.response == AWAITING_ANSWER_REFUSAL_MESSAGE and not refused.command_response.success
    with pytest.raises(UsePlanError) as error:
        todo.ctx.check_use_plan("anything")
    assert str(error.value) == AWAITING_ANSWER_REFUSAL_MESSAGE
