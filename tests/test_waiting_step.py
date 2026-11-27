"""The one transition function and the saved, parked and ended forms of every kind (bd fix-fpyb.8, step 9 of
``docs/pending_state_design.md``).

``waiting.step(record, event, facts)`` decides what a message, a cancel and a restore do to the question that is out; the
table of section 3.4 is checked here as a table, over every kind and every event, with no session. The rest drives real
sessions (the fixtures of ``tests/test_waiting_kinds.py``) to each kind and checks the saved blob (no key of the removed
flags, the record is the question), the restore, and the parking of a conversation switched away from while a question is
out: one thing is parked, the record, with the store's record it points at.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow import approval_binding as ab
from fastworkflow import waiting as wm
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import _ask, _finish, annotated_library, todo  # noqa: F401 - fixtures
from tests.test_no_agent_decisions import Conditional
from tests.test_no_agent_mode import _plan
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401 - the CME on plain_required_workflow
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers
    A_NOTE, ABOUT_DUNE, ABOUT_EMMA, HOST, no_agent)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated, remote_decides_emma_and_joke  # noqa: F401
from tests.test_step_helpers import _build_planning_command, library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.test_waiting_kinds import _no_agent_joke_decisions, no_param_model  # noqa: F401
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

REMOVED_KEYS = ("awaiting_user", "suspended_user_message", "pending_clarification_request", "assistant_command")


# ======================================================================================================================
# the table, with no session
# ======================================================================================================================

def _record(kind, owner="plan", **payload):
    return wm.Waiting(kind, owner=owner, question="q?", payload=payload)


RECORDS = {
    "none": None,
    "plan_intent": _record(wm.KIND_INTENT),
    "plan_parameters": _record(wm.KIND_PARAMETERS, after="resolve_next"),
    "cme_intent": _record(wm.KIND_INTENT, owner="cme"),
    "cme_parameters": _record(wm.KIND_PARAMETERS, owner="cme", after="run_command"),
    "review": _record(wm.KIND_REVIEW, record_id="r", slot=None, gate=True, suspended_agent=True),
    "approval_suspended": _record(wm.KIND_APPROVAL, record_id="a", suspended_agent=True),
    "approval_direct": _record(wm.KIND_APPROVAL, record_id="a", suspended_agent=False),
    "decision": _record(wm.KIND_DECISION, decision={}),
    "agent": _record(wm.KIND_AGENT),
}

#: ``MESSAGE "yes"`` on the agent path: (action, begin, reader). The CME's questions are read on the deterministic path
#: because their round is a "/" round (``assistant_round``), which is how a real session holds one.
MESSAGE = {
    "none": (wm.ACT_START_AGENT, True, ""),
    "plan_intent": (wm.ACT_RESUME_AGENT, False, wm.READER_RESOLUTION),
    "plan_parameters": (wm.ACT_RESUME_AGENT, False, wm.READER_RESOLUTION),
    "cme_intent": (wm.ACT_DETERMINISTIC, True, wm.READER_CME),
    "cme_parameters": (wm.ACT_DETERMINISTIC, True, wm.READER_CME),
    "review": (wm.ACT_RESUME_AGENT, False, wm.READER_APPROVAL),
    "approval_suspended": (wm.ACT_RESUME_AGENT, False, wm.READER_APPROVAL),
    "approval_direct": (wm.ACT_APPROVAL, True, wm.READER_APPROVAL),
    "decision": (wm.ACT_RESUME_AGENT, False, wm.READER_DECISION),
    "agent": (wm.ACT_RESUME_AGENT, False, wm.READER_AGENT),
}

#: ``CANCEL``: (action, text, label).
CANCEL = {
    "none": (wm.ACT_NOTHING, "", ""),
    "plan_intent": (wm.ACT_REPLY, "cancel", "clarification"),
    "plan_parameters": (wm.ACT_REPLY, "cancel", "clarification"),
    "cme_intent": (wm.ACT_REPLY, "abort", "clarification"),
    "cme_parameters": (wm.ACT_REPLY, "abort", "clarification"),
    "review": (wm.ACT_REPLY, "no", "review"),
    "approval_suspended": (wm.ACT_REPLY, "no", "review"),
    "approval_direct": (wm.ACT_REPLY, "no", "review"),
    "decision": (wm.ACT_REPLY, "cancel", "clarification"),
    "agent": (wm.ACT_NOTHING, "", "question"),
}


def _facts(name, **more):
    return wm.Facts(agent_path=True, assistant_round=name.startswith("cme"), **more)


@pytest.mark.parametrize("name", RECORDS)
def test_a_message_is_decided_by_one_call_for_every_kind(name):
    record = RECORDS[name]
    action, begin, reader = MESSAGE[name]
    move = wm.step(record, wm.Event(wm.MESSAGE, "yes"), _facts(name))
    assert (move.action, move.begin, move.reader, move.text, move.assistant) == (action, begin, reader, "yes", False)


@pytest.mark.parametrize("name", RECORDS)
def test_a_slash_message_is_the_reply_with_its_slash_removed_and_with_nothing_out_a_deterministic_command(name):
    record = RECORDS[name]
    move = wm.step(record, wm.Event(wm.MESSAGE, "//yes"), _facts(name))
    if record is None:
        # Nothing out: the "/" message opens the assistant round and the deterministic path takes it, text unchanged.
        assert (move.action, move.begin, move.assistant, move.text) == (wm.ACT_DETERMINISTIC, True, True, "//yes")
    else:
        assert (move.text, move.assistant) == ("yes", False)
        assert move.action == MESSAGE[name][0]


@pytest.mark.parametrize("name", RECORDS)
def test_a_cancel_is_the_kinds_reply_from_the_table(name):
    action, text, label = CANCEL[name]
    move = wm.step(RECORDS[name], wm.Event(wm.CANCEL))
    assert (move.action, move.text, move.label) == (action, text, label if RECORDS[name] is not None else "")


@pytest.mark.parametrize("name", RECORDS)
def test_a_restore_puts_the_record_back_as_saved(name):
    record = RECORDS[name]
    move = wm.step(None, wm.Event(wm.RESTORE, state=None if record is None else json.loads(json.dumps(record.to_state()))))
    assert move.action == wm.ACT_RESTORE and move.record == record


def test_the_users_management_messages_keep_their_text_and_meaning():
    facts = _facts("none", manage=True)
    for text in ("/approvals", "/revoke_approval add_note"):
        move = wm.step(None, wm.Event(wm.MESSAGE, text), facts)
        assert (move.action, move.text, move.assistant, move.begin) == (wm.ACT_MANAGE, text, False, True)
    # A review or an approval that suspended the agent answers its own reply: the message is not management.
    for name in ("review", "approval_suspended"):
        move = wm.step(RECORDS[name], wm.Event(wm.MESSAGE, "/approvals"), _facts(name, manage=True))
        assert (move.action, move.text) == (wm.ACT_RESUME_AGENT, "/approvals")
    # An approval raised outside the agent's loop does not: management is answered beside it.
    assert wm.step(RECORDS["approval_direct"], wm.Event(wm.MESSAGE, "/approvals"),
                   _facts("approval_direct", manage=True)).action == wm.ACT_MANAGE


def test_cancel_pending_reaches_the_approval_store_whatever_suspended_the_agent():
    for name in ("review", "approval_suspended", "approval_direct"):
        move = wm.step(RECORDS[name], wm.Event(wm.MESSAGE, "/cancel_pending"), _facts(name, cancel_pending=True))
        assert (move.action, move.text) == (wm.ACT_APPROVAL, "/cancel_pending")


def test_which_kinds_ride_on_the_agents_suspension():
    assert {name: (rec is not None and rec.rides_suspension) for name, rec in RECORDS.items()} == {
        "none": False, "plan_intent": True, "plan_parameters": True, "cme_intent": False, "cme_parameters": False,
        "review": True, "approval_suspended": True, "approval_direct": False, "decision": True, "agent": True}
    asked = wm.Waiting(wm.KIND_APPROVAL, payload={"record_id": "a", "suspended_agent": False, wm.ASKED_AGAIN: True})
    assert asked.rides_suspension


# ======================================================================================================================
# every kind on a real session: saved, restored, parked
# ======================================================================================================================

def _reach_agent(request):
    todo = request.getfixturevalue("todo")
    request.getfixturevalue("annotated_library")
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    assert todo.ctx.process_turn("find a book").status is TurnStatus.AWAITING_USER
    return todo.ctx


def _reach_approval(request):
    todo = request.getfixturevalue("todo")
    request.getfixturevalue("annotated_library")
    assert _ask(todo, "add_note <text>Ann</text>", steps=_finish()).status is TurnStatus.AWAITING_USER
    return todo.ctx


def _reach_review(request):
    gated = request.getfixturevalue("gated")
    request.getfixturevalue("no_param_model")
    request.getfixturevalue("remote_decides_emma_and_joke")   # the remote model binds EMMA, leaves JOKE ad hoc
    _split_into(request.getfixturevalue("monkeypatch"), [EMMA, JOKE])
    assert gated.ctx.process_turn(f"{EMMA} and {JOKE}").status is TurnStatus.AWAITING_USER
    _open_review(gated)
    return gated.ctx


def _reach_plan_question(request):
    env = request.getfixturevalue("no_agent")
    monkeypatch = request.getfixturevalue("monkeypatch")
    _no_agent_joke_decisions(env, monkeypatch)
    env.parts = [JOKE]
    assert env.ctx.process_turn(JOKE, approval_surface=HOST).status is TurnStatus.AWAITING_USER
    return env.ctx


def _reach_decision(request):
    env = request.getfixturevalue("library_env")
    request.getfixturevalue("no_agent_model")
    request.getfixturevalue("no_planner_key")
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, env)
    assert env.ctx.process_turn("look up the book Dune").status is TurnStatus.AWAITING_USER
    return env.ctx


def _reach_cme(request):
    ctx = request.getfixturevalue("session")(False)
    ctx.process_turn(PARTIAL)
    return ctx


REACH = {"agent": _reach_agent, "approval": _reach_approval, "review": _reach_review, "plan_question": _reach_plan_question,
         "decision": _reach_decision, "cme": _reach_cme}
KIND_OF = {"agent": "agent", "approval": "approval", "review": "review", "plan_question": "intent",
           "decision": "decision", "cme": "parameters"}


@pytest.mark.parametrize("name", REACH)
def test_a_saved_blob_holds_the_record_and_none_of_the_removed_flags_and_restores_it(request, name):
    ctx = REACH[name](request)
    before = ctx._waiting
    assert before.kind == KIND_OF[name] and ctx.waiting is before
    asleep = before.rides_suspension

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id=f"step-{name}")))

    assert blob["schema_version"] == 8 and blob["waiting"]["kind"] == KIND_OF[name]
    assert not [key for key in REMOVED_KEYS if key in blob]
    assert blob["waiting"]["assistant"] is ctx.assistant_round
    ctx._waiting = None
    ctx.apply_serialized_state(blob)
    assert ctx._waiting == before and ctx._awaiting_user is asleep
    assert ctx.assistant_round is before.assistant


@pytest.mark.parametrize("name", ["agent", "review", "plan_question", "decision", "cme"])
def test_one_thing_is_parked_when_a_conversation_is_switched_away_from_and_it_comes_back_whole(request, name):
    ctx = REACH[name](request)
    before = ctx._waiting
    store = ab.store_for(ctx._app_workflow)
    held = store.active()
    ctx.park_pending_for_new_conversation()          # filed under the conversation it belongs to ("none" when it has none)

    assert ctx._waiting is None and not ctx._awaiting_user and ctx.has_parked_pending()
    (key,) = ctx._parked_pending
    (snapshot,) = ctx._parked_pending[key]
    assert "reviews" not in snapshot, "the store's record is parked inside the turn's record, not beside it"
    if name == "review":
        assert snapshot["waiting"]["payload"][wm.STORE_RECORD_KEY]["id"] == held.id
        assert store.active() is None and store.get(held.id) is None
    else:
        assert wm.STORE_RECORD_KEY not in snapshot["waiting"].get("payload", {})

    ctx._restore_parked_pending(None if key == "none" else int(key))     # what binding the conversation again does

    assert not ctx.has_parked_pending() and ctx._waiting == before
    assert wm.STORE_RECORD_KEY not in ctx._waiting.payload
    if name == "review":
        assert store.active().id == held.id


def test_an_approval_is_cancelled_not_parked_and_its_record_ends_with_the_stores(request):
    ctx = _reach_approval(request)
    store = ab.store_for(ctx._app_workflow)
    held = store.active()
    ctx.bind_observability_identity(conversation_id=1)
    ctx.park_pending_for_new_conversation()
    assert held.state == "invalidated" and ctx._waiting is None and not ctx.has_parked_pending()
