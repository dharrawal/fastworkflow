"""One test per pending kind: the derived ``waiting`` view and what ``//cancel`` does (bd fix-fpyb.8 step 1, fix-fpyb.6).

``WorkflowExecutionContext.waiting`` is a read-only view derived from today's flags and records
(``fastworkflow/waiting.py``). For each kind of question a session can be waiting on, a real session is driven to that
question and two things are checked: (a) the view's kind and the pointer to the record that holds the question, and (b)
that ``ChatSession.cancel_pending_prompt``'s lookup (``_pending_cancel_reply``) names the kind's cancel reply and that
sending it ends the question the way typing that reply does. Real contexts on the real test workflows (the library
copies and tests/plain_required_workflow), the model seams scripted the way the files named below script them; no mocks.

* ``agent`` (an ``ask_user`` suspension): reached with the scripted agent of tests/test_approval_binding.py. A
  ChatSession is Topology A: its ``ask_user`` BLOCKS on the queue, so it never holds a suspended one, and the blocked
  read is ended by the queue's own cancel (``test_cancel_ends_a_pending_question_through_the_engines_cancelled_command``
  in tests/test_cli_turn_control.py, a real threaded session). Here the lookup says "not a reply" and the Topology B
  embedder's ``cancel_pending`` ends it.
* a part's value question asked while it runs, with an agent model (``at_gate`` False): reached as
  tests/test_parameter_loop.py reaches it, with the typed planning call refused. The question is never put to a CLI
  session (it has a blocking queue), and with an agent model the reply is today's text for the part's own call, not a
  cancel (design 3.6, step 6 changes that): the test pins today's behaviour.
"""
from __future__ import annotations

import json

import dspy
import pytest

from fastworkflow.review import approval_binding as ab
from fastworkflow import intent_loop, no_agent_mode, workflow_agent
from fastworkflow.parameters import parameter_clarification
from fastworkflow.planning import planning_commands as pc
from fastworkflow import waiting as waiting_module
from fastworkflow.chat_session import ChatSession
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import build_query_with_next_steps as REAL_BUILD
from fastworkflow.workflow_agent import load_planning_command_map
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT, WorkflowExecutionContext
from tests.conftest import script_matcher
from tests.test_approval_binding import _ask, _finish, annotated_library, todo  # noqa: F401 - fixtures
from tests.test_no_agent_decisions import Conditional
from tests.test_no_agent_mode import _plan
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401 - the CME on plain_required_workflow
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers
    A_NOTE, ABOUT_DUNE, ABOUT_EMMA, BOOK_PLAN, HOST, NOTE_PLAN, _TextPlanner, no_agent)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated, remote_decides_emma_and_joke  # noqa: F401
from tests.test_step_helpers import _build_planning_command, library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _chat_over(ctx) -> ChatSession:
    """A real ChatSession whose core is *ctx*: ``_pending_cancel_reply`` is a lookup on the core's ``waiting``."""
    chat = ChatSession.__new__(ChatSession)
    chat._core = ctx
    return chat


def _cancel_reply(ctx) -> tuple[str, str]:
    reply = _chat_over(ctx)._pending_cancel_reply()
    assert reply is not None, f"nothing to cancel: {ctx.waiting}"
    return reply


def _last(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def _no_agent_joke_decisions(env, monkeypatch):
    """The remote decision model says ``none`` of what is not about the library (tests/test_plan_choice_question.py):
    the part is ad hoc and asks which planning command is meant."""
    plans = {ABOUT_EMMA: BOOK_PLAN, ABOUT_DUNE: BOOK_PLAN, A_NOTE: NOTE_PLAN}

    def match(text):
        name = plans.get(text)
        return {name: 0.9, "none": 0.0} if name else {"none": 1.0}

    env.matched = script_matcher(monkeypatch, match)


# ======================================================================================================================
# The table and the record
# ======================================================================================================================

def test_every_kind_has_a_table_row_and_the_record_survives_json():
    assert set(waiting_module.TABLE) == set(waiting_module.KINDS)
    # The client-visible values are the ones that were there before the table.
    assert {kind: spec.question_kind for kind, spec in waiting_module.TABLE.items()} == {
        "intent": "plan_choice", "parameters": "planning_parameter", "review": "review", "approval": "approval",
        "decision": "phase_step", "agent": "agent"}
    record = waiting_module.Waiting("parameters", owner="cme", question="q?", payload={"record": {"asks": 1}})
    assert waiting_module.Waiting.from_state(record.to_state()) == record
    assert record.cancel_text == "abort" and waiting_module.Waiting("parameters").cancel_text == "cancel"
    assert waiting_module.Waiting.from_state({"kind": "nonsense"}) is None and waiting_module.Waiting.from_state(None) is None


def test_nothing_pending_is_no_record(gated):
    assert gated.ctx.waiting is None
    assert _chat_over(gated.ctx)._pending_cancel_reply() is None


# ======================================================================================================================
# agent
# ======================================================================================================================

def test_an_agent_question_is_the_agent_kind_and_is_ended_by_the_embedders_cancel(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    out = todo.ctx.process_turn("find a book")
    assert out.status is TurnStatus.AWAITING_USER
    assert _last(out)["question_kind"] == "agent"

    waiting = todo.ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question) == ("agent", "plan", "Which book do you mean?")
    assert waiting.question_kind == "agent" and waiting.cancel_text is None
    # Not a reply: a ChatSession's blocked read is ended by the queue (tests/test_cli_turn_control.py).
    assert _chat_over(todo.ctx)._pending_cancel_reply() is None

    assert todo.ctx.cancel_pending() is True
    assert todo.ctx.waiting is None and not todo.ctx.awaiting_user


# ======================================================================================================================
# the plan's own questions
# ======================================================================================================================

def test_a_plans_which_command_question_is_cancelled_by_cancel(no_agent, monkeypatch):
    _no_agent_joke_decisions(no_agent, monkeypatch)
    no_agent.parts = [JOKE]
    ctx = no_agent.ctx
    asked = ctx.process_turn(JOKE, approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER and _last(asked)["question_kind"] == "plan_choice"

    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question_kind) == ("intent", "plan", "plan_choice")
    assert waiting is ctx._waiting and waiting.payload["record"]["origin"] == 0 and waiting.payload["part"] == 0
    assert waiting.question == waiting.payload["record"]["question"] != ""
    assert _cancel_reply(ctx) == ("clarification", "cancel")

    done = ctx.process_turn("cancel", approval_surface=HOST)

    assert done.status is TurnStatus.COMPLETED
    (stop,) = [entry for entry in _last(done)[NOT_RUN_ARTIFACT] if entry["command"] is None]
    assert stop["reason"] == "it was declined when asked which planning command should run it"
    assert ctx.waiting is None and not ctx.has_pending_state()


def test_a_plans_value_question_at_the_gate_is_cancelled_by_cancel(no_agent):
    ctx = no_agent.ctx
    asked = ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER and _last(asked)["question_kind"] == "planning_parameter"

    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question_kind) == ("parameters", "plan", "planning_parameter")
    assert waiting is ctx._waiting and waiting.payload["after"] == "resolve_next"
    assert waiting.payload["part"] == ctx._plan_resolution.index
    assert _cancel_reply(ctx) == ("clarification", "cancel")

    review = ctx.process_turn("cancel", approval_surface=HOST)

    # Cancelling drops that part only; the review of the other two is now the question (tests/test_parameter_loop.py).
    assert review.status is TurnStatus.AWAITING_USER and _last(review)["question_kind"] == "review"
    assert [item["goal"] for item in _last(review)["review"]["items"]] == [ABOUT_EMMA, ABOUT_DUNE]
    assert ctx.waiting.kind == "review"


def test_a_parts_value_question_asked_while_it_runs_is_the_parameters_kind(todo, monkeypatch):
    """With an agent model a part's value question is the same kind as any other: ``cancel`` is its cancel reply and ends
    the part (step 6, consequence b; it used to be read as the part's text and handed to the agent)."""
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
    ctx = todo.ctx

    first = ctx.process_turn(request)

    assert first.status is TurnStatus.AWAITING_USER and _last(first)["question_kind"] == "planning_parameter"
    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question_kind) == ("parameters", "plan", "planning_parameter")
    assert waiting is ctx._waiting and waiting.payload["after"] == "resume_part"
    assert _cancel_reply(ctx) == ("clarification", "cancel")
    # The reply the lookup names cancels the part: it is not run, the turn completes naming it as not run, and no agent
    # step was taken for it.
    todo.script.steps = _finish()
    done = ctx.process_turn("cancel")
    assert done.status is TurnStatus.COMPLETED and ctx.waiting is None
    assert "declined" in done.answer


# ======================================================================================================================
# review and approval (the approval store)
# ======================================================================================================================

@pytest.fixture
def no_param_model(monkeypatch):
    """The review tests' guard (tests/test_cancel_pending_review.py), asked for by name so it does not apply to every test
    here: the parameter-extraction model is never reached."""
    from fastworkflow.parameters.signatures import InputForParamExtraction

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"the parameter-extraction model was called for {self.command!r}")

    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", refuse)
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def test_a_sub_task_review_is_cancelled_by_no(gated, no_param_model, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    asked = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert asked.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)

    waiting = gated.ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question_kind) == ("review", "plan", "review")
    assert waiting.payload == {"record_id": record.id, "suspended_agent": False} or waiting.payload["record_id"] == record.id
    assert _cancel_reply(gated.ctx) == ("review", "no")

    done = gated.ctx.process_turn("no")

    assert done.status is TurnStatus.COMPLETED and done.answer.startswith("You declined the sub-task review")
    assert gated.ctx.waiting is None and ab.active_reviews(gated.wf) == []


def test_an_action_approval_is_cancelled_by_no(todo):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER and _last(out)["question_kind"] == "approval"
    record = todo.store.active()

    waiting = todo.ctx.waiting
    assert (waiting.kind, waiting.question_kind) == ("approval", "approval")
    assert waiting.payload["record_id"] == record.id and waiting.payload["suspended_agent"] is True
    assert _cancel_reply(todo.ctx) == ("review", "no")

    todo.script.steps = _finish()
    done = todo.ctx.process_turn("no")

    assert done.status is TurnStatus.COMPLETED
    assert todo.ctx.waiting is None and ab.active_approvals(todo.wf) == []
    assert todo.chores.notes == todo.base, "a declined approval never runs the command"


# ======================================================================================================================
# decision step
# ======================================================================================================================

def test_a_decision_step_is_cancelled_by_cancel(library_env, no_agent_model, no_planner_key, monkeypatch):
    # The turn is not the sub-task review gate's subject (the fixture of tests/test_turn_result_capture.py, inline).
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    out = library_env.ctx.process_turn("look up the book Dune")
    assert out.status is TurnStatus.AWAITING_USER and _last(out)["question_kind"] == "phase_step"

    ctx = library_env.ctx
    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner, waiting.question_kind) == ("decision", "plan", "phase_step")
    assert waiting is ctx._waiting and waiting.payload["decision"]["kind"] == "conditional"
    assert ctx._phase_run is not None and "decision" not in ctx._phase_run     # the record holds it, the phase does not
    assert _cancel_reply(ctx) == ("clarification", "cancel")

    done = ctx.process_turn("cancel")

    # The reply `cancel` declines: the command ends with its stop, not an error (tests/test_no_agent_decisions.py).
    assert done.status is TurnStatus.COMPLETED
    (stop,) = _last(done)[NOT_RUN_ARTIFACT]
    assert stop["open_step"]["reason"].startswith("declined") and stop["phase"] == "find"
    assert ctx.waiting is None and ctx._phase_run is None


# ======================================================================================================================
# the CME (deterministic path)
# ======================================================================================================================

def test_the_open_commands_value_question_is_cancelled_by_abort(session):
    ctx = session(False)
    first = ctx.process_turn(PARTIAL)
    assert _last(first).get("parameter_request") is not None

    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner) == ("parameters", "cme")
    assert waiting.payload["loop"] == parameter_clarification.pending(ctx._cme_workflow.context).to_state()
    assert waiting is ctx._waiting and waiting.payload["after"] == "run_command"
    assert _cancel_reply(ctx) == ("clarification", "abort")

    done = ctx.process_turn("abort")

    assert "aborted" in done.answer
    assert ctx.waiting is None and not ctx.has_open_command()


def test_the_which_command_question_of_the_cme_is_cancelled_by_abort(session):
    ctx = session(False)
    ctx.process_turn(PARTIAL)
    out = ctx.process_turn("you misunderstood")
    assert _last(out).get("plan_choice") is not None

    waiting = ctx.waiting
    assert (waiting.kind, waiting.owner) == ("intent", "cme")
    assert waiting.payload["record"] is intent_loop.pending_record(ctx._cme_workflow.context) and waiting is ctx._waiting
    assert _cancel_reply(ctx) == ("clarification", "abort")

    done = ctx.process_turn("abort")

    assert "aborted" in done.answer
    assert ctx.waiting is None and not ctx.has_open_command()


# ======================================================================================================================
# the one-question rule
# ======================================================================================================================

def _approval_beside_the_open_command(session):
    """An approval of the approval store raised while the CME's command is open (the case that was two questions at once)."""
    ctx = session(False)
    ctx.process_turn(PARTIAL)
    assert ctx.waiting.kind == "parameters" and ctx.waiting.owner == "cme"
    import time

    from fastworkflow.review.review_engine import AWAITING, ReviewRecord

    record = ReviewRecord(
        id="approval-1", token_hash="", command="add_note", annotation=None, raw_command="add_note", binding={},
        fingerprint="", rendering="Run add_note?", rendering_hash="", required_identity="", created_turn=0, turns_seen=0,
        ttl_turns=2, ttl_seconds=600, created_at=time.time(), state=AWAITING, origin="agent", suspended_agent=True)
    ab.store_for(ctx._app_workflow).add(record)
    ctx._put_approval_record(record)         # what ``approval_binding.gate`` does when it asks
    return ctx, record


def test_two_questions_out_at_once_cannot_exist_a_new_question_replaces_the_one_out(session):
    """The turn's one slot holds one record: opening a question closes the one that was open (the strict/non-strict
    switch and the check that two were out are gone with the second source of a question)."""
    ctx, record = _approval_beside_the_open_command(session)
    assert ctx.waiting is ctx._waiting and ctx.waiting.kind == "approval" and ctx.waiting.payload["record_id"] == record.id
    assert not ctx.has_open_command()
    assert _chat_over(ctx)._pending_cancel_reply() == ("review", "no")
    assert not hasattr(waiting_module, "derive") and not hasattr(waiting_module, "STRICT")


def test_a_question_that_rides_the_agents_suspension_is_not_a_flag_that_can_be_set(session):
    ctx, _record = _approval_beside_the_open_command(session)
    assert ctx._awaiting_user is True                 # derived from the record: an approval that suspended the agent
    with pytest.raises(AttributeError):
        ctx._awaiting_user = False
    assert ctx.awaiting_user is True


# ======================================================================================================================
# the review and the decision step are records (step 8): the removed flags are gone, the record is saved and restored
# ======================================================================================================================

def test_the_removed_review_flags_are_gone_from_the_frame_the_state_and_the_context():
    from fastworkflow.planning.plan_frame import PlanFrame
    from fastworkflow.session.session_state_store import SCHEMA_VERSION

    frame = PlanFrame.turn()
    assert not hasattr(frame, "subtask_review_open") and not hasattr(frame, "framework_entry")
    assert not hasattr(WorkflowExecutionContext, "_review_slot")
    assert SCHEMA_VERSION == 8


def test_a_sub_task_review_is_the_turns_record_pointing_at_the_stores_record(gated, no_param_model, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    store_record = _open_review(gated)
    ctx = gated.ctx

    held = ctx._waiting
    assert (held.kind, held.owner) == ("review", "plan") and ctx.waiting is held
    assert held.payload == {"record_id": store_record.id, "slot": None, "gate": True, "suspended_agent": True}
    assert held.question == store_record.rendering
    assert ctx._gate_review_out() and ctx._subtask_review_mark() == "open"     # the span's value is derived from the record
    assert ab.store_for(gated.wf).get(held.payload["record_id"]) is store_record    # the store keeps the record

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="review-record")))
    assert blob["waiting"]["kind"] == "review" and blob["waiting"]["payload"]["record_id"] == store_record.id
    assert "subtask_review_open" not in blob and "review_slot" not in blob and "decision" not in (blob["skill_phase_run"] or {})

    ctx._waiting = None                                    # as a new process: only the blob holds the question
    ctx.apply_serialized_state(blob)
    assert ctx.waiting.kind == "review" and ctx._gate_review_out() and ctx.waiting.payload["record_id"] == store_record.id
    assert _cancel_reply(ctx) == ("review", "no")

    done = ctx.process_turn("no")
    assert done.status is TurnStatus.COMPLETED and ctx._waiting is None and ctx._subtask_review_mark() == "declined"


def test_a_stand_alone_framework_review_is_a_record_that_is_not_the_gate_and_ends_with_the_call(todo):
    todo.wf.record = todo.library
    awaiting = todo.ctx.run_framework_planning_command(
        "review_subtasks", {"subtasks": "book a flight; tell Ann"}, "review the sub-tasks")
    assert awaiting.command_response.artifacts.get("awaiting_user") is True
    held = todo.ctx._waiting
    assert held.kind == "review" and held.payload["gate"] is False and held.payload["record_id"] == todo.store.active().id
    assert not todo.ctx._gate_review_out() and todo.ctx._subtask_review_mark() == "not_applicable"

    todo.script.steps = _finish()
    done = todo.ctx.process_turn("accept")
    assert done.status is TurnStatus.COMPLETED and todo.ctx._waiting is None and todo.ctx.waiting is None


def test_the_review_slot_question_is_saved_and_restored_inside_the_review_record(gated, no_param_model, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    _open_review(gated)
    ctx = gated.ctx
    cme = ctx._plan_cme.context
    waiting_module.port_of(cme).put(waiting_module.Waiting(
        waiting_module.KIND_INTENT, owner=waiting_module.OWNER_CME, payload={"record": intent_loop.open_question(
            "x", [{"name": "A/b", "description": "", "closest": True}])}))
    assert ctx._waiting.payload["slot"] is waiting_module.port_of(cme).get() and ctx.waiting.kind == "review"

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="slot")))
    assert blob["waiting"]["payload"]["slot"]["kind"] == "intent"
    ctx._waiting = None
    ctx._clear_cme_continuation(ctx._plan_cme)
    ctx.apply_serialized_state(blob)
    restored = waiting_module.port_of(ctx._plan_cme.context).get()
    assert restored is ctx._waiting.payload["slot"] and restored.kind == "intent"
    assert intent_loop.waiting_for(ctx._plan_cme.context) == intent_loop.WAITING_INTENT

    ctx._reset_agent_suspension()                           # the suspension ends: the review and its slot go together
    assert ctx._waiting is None and waiting_module.port_of(ctx._plan_cme.context).get() is None


def test_a_paused_decision_is_the_turns_record_and_phase_run_does_not_hold_it(library_env, no_agent_model, no_planner_key,
                                                                              monkeypatch):
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune")
    ctx = library_env.ctx
    held = ctx._waiting
    assert (held.kind, held.owner) == ("decision", "plan") and held.question == first.answer
    assert "decision" not in (ctx._phase_run or {}) and "decision" not in (ctx._turn_frame().phase_run or {})

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="decision-record"), default=str))
    assert blob["waiting"]["kind"] == "decision" and "decision" not in (blob["skill_phase_run"] or {})
    ctx._waiting = None
    ctx.apply_serialized_state(blob)
    assert ctx.waiting.kind == "decision" and ctx._waiting.payload["decision"]["asked"] == 1
    assert _cancel_reply(ctx) == ("clarification", "cancel")


def test_a_cancel_pending_drops_the_decision_record_with_the_suspension(library_env, no_agent_model, no_planner_key,
                                                                         monkeypatch):
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    assert library_env.ctx.cancel_pending() is True
    assert library_env.ctx._waiting is None and library_env.ctx.waiting is None


# ======================================================================================================================
# the agent's question and an approval are records (step 9)
# ======================================================================================================================

def test_the_agents_own_question_is_the_turns_agent_record_and_the_flags_it_replaced_are_gone(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    out = todo.ctx.process_turn("find a book")
    assert out.status is TurnStatus.AWAITING_USER
    ctx = todo.ctx

    held = ctx._waiting
    assert (held.kind, held.owner, held.question) == ("agent", "plan", "Which book do you mean?")
    assert held.payload == {"message": "find a book"} and ctx._suspended_message() == "find a book"
    assert ctx._awaiting_user is True and ctx.awaiting_user is True
    for gone in ("_suspended_user_message", "_pending_clarification_request"):
        assert not hasattr(ctx, gone)
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="agent-record")))
    assert blob["waiting"]["payload"] == {"message": "find a book"}
    assert not {"awaiting_user", "suspended_user_message", "pending_clarification_request",
                "assistant_command"} & set(blob)

    todo.script.steps = _finish()
    done = ctx.process_turn("the blue one")
    assert done.status is TurnStatus.COMPLETED and ctx._waiting is None and not ctx._awaiting_user


def test_an_approval_the_agent_asked_is_a_record_pointing_at_the_store_and_ends_with_the_stores_record(todo):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    record = todo.store.active()
    held = todo.ctx._waiting
    assert (held.kind, held.question) == ("approval", record.rendering)
    assert held.payload == {"record_id": record.id, "suspended_agent": True}
    assert todo.ctx._awaiting_user is True and todo.store.get(held.payload["record_id"]) is record   # the store keeps it

    # The store ends it by itself (here: the time to live ran out): the turn's record ends with it, in one place.
    from fastworkflow.review import review_engine

    assert review_engine.expire_if_due(todo.ctx, todo.store, record, surface=todo.store.surface, now=record.expires_at + 1)
    assert record.state == "expired" and todo.ctx._waiting is None and not todo.ctx._awaiting_user


def test_a_decline_decided_on_the_store_ends_the_record_too(todo):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    result = ab.resolve_reply(todo.ctx, todo.wf, "no")
    assert result.done and record.state == "declined" and todo.ctx._waiting is None


def test_an_approval_typed_by_the_user_is_not_a_suspension_until_a_reply_is_not_understood(todo, annotated_library):
    from fastworkflow import Action
    from fastworkflow.command_executor import CommandExecutor
    from tests.test_approval_binding import MANDATORY

    out = CommandExecutor.perform_action(todo.wf, Action(
        command_name=MANDATORY, command="add_note <text>Ann</text>", parameters={"text": "Ann"}))
    assert out.command_response.artifacts.get("awaiting_user") is True
    held = todo.ctx._waiting                       # put by the gate through the store's owner, whatever the host
    assert held.kind == "approval" and held.payload["suspended_agent"] is False and not todo.ctx._awaiting_user

    unclear = todo.ctx.process_turn("maybe")
    assert unclear.status is TurnStatus.AWAITING_USER and todo.ctx._awaiting_user    # asked again: the next message continues
    assert todo.ctx._waiting.payload[waiting_module.ASKED_AGAIN] is True

    todo.ctx.process_turn("yes")
    assert todo.ctx._waiting is None and not todo.ctx._awaiting_user
