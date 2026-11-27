"""A user-typed "/" command that is held for approval must not leave the session on the assistant path (fix-weih.15).

``CommandExecutor.invoke_command`` holds a "/" command that needs approval and returns the held output. The
``is_assistant_mode_command`` flag that ``_route_message`` set for the "/" message used to stay set after the
approval was answered, so ``_should_run_agent_for_message`` refused the agent for every later plain message and a later
"/" command was read as natural language. The reply to a pending approval is routed by ``approval_binding.active_record``
before the flag is read, so the flag is not needed while the approval is open; it is cleared when the reply resolves it.

Real WorkflowExecutionContext, real approval store, the annotated library copy, a scripted agent. Any model call fails.
"""
from __future__ import annotations

import pytest

from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _added, _finish, annotated_library, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")



def _flag(todo) -> bool:
    return todo.ctx.assistant_round


def _held(todo, text):
    out = todo.ctx.process_turn(f"/add_note text={text}")
    last = out.command_outputs[-1].command_response.artifacts
    assert out.status is TurnStatus.COMPLETED and last["awaiting_user"] is True
    assert todo.store.active() is not None
    return out


def _plain_message_reaches_the_agent(todo):
    """A plain message in an agent session runs the scripted agent (its one step is consumed)."""
    todo.script.steps = _finish()
    out = todo.ctx.process_turn("do it")
    assert todo.script.steps == [], "the agent never asked for its scripted step: it did not run"
    assert "Ambiguous intent" not in out.answer
    assert not any("Ambiguous intent" in o.command_response.response for o in out.command_outputs)
    return out


REPLIES = [
    pytest.param("yes", ["Ann"], id="yes"),
    # bd fix-qjne.11: "always" is no longer an answer to a "/" add_note approval (REQUIRED_ALWAYS asks every
    # time); the test of that reply is test_always_is_not_an_answer_to_a_required_always_approval below.
    pytest.param("no", [], id="no"),
]


@pytest.mark.parametrize("reply, notes", REPLIES)
def test_agent_session_is_reachable_after_a_slash_approval_is_answered(
        todo, annotated_library, grant_root, reply, notes):
    _held(todo, "Ann")
    assert _flag(todo), "while the approval is open the flag stays (the reply is routed by the open record)"
    todo.script.steps = _finish()
    todo.ctx.process_turn(reply)
    assert _added(todo.chores, todo.base) == notes
    assert todo.store.active() is None

    _plain_message_reaches_the_agent(todo)                   # (a) the agent runs, not the wildcard path
    assert _added(todo.chores, todo.base) == notes

    out = todo.ctx.process_turn("/add_note text=Cy")         # "/" commands still work
    assert out.command_outputs[-1].command_response.artifacts["awaiting_user"] is True
    assert _added(todo.chores, todo.base) == notes
    assert todo.store.active() is not None
    # (e) last: the plain message and the "/" command above are the behaviour; the flag is its cause.
    assert not _flag(todo) or todo.store.active() is not None


@pytest.mark.parametrize("assistant", [False, True], ids=["agent", "assistant"])
def test_always_is_not_an_answer_to_a_required_always_approval(todo, annotated_library, grant_root, assistant):
    """The reply the REPLIES list used to carry: it asks again, records nothing, runs nothing; "no" then ends it."""
    if assistant:
        todo.ctx._run_as_agent = False
        todo.ctx._cme_workflow.context.pop("run_as_agent", None)
        todo.wf.context.pop("run_as_agent", None)
    _held(todo, "Ann")
    todo.script.steps = []
    out = todo.ctx.process_turn("always")
    assert "asks every time" in out.answer
    assert todo.store.active() is not None and _added(todo.chores, todo.base) == []
    todo.ctx.process_turn("no")
    assert todo.store.active() is None and _added(todo.chores, todo.base) == []
    again = todo.ctx.process_turn("/add_note text=Cy")                 # no grant: asked again
    assert again.command_outputs[-1].command_response.artifacts["awaiting_user"] is True


def test_a_reply_that_is_not_yes_always_or_no_asks_again_and_keeps_the_approval(
        todo, annotated_library, grant_root):
    """(b) A non-matching reply is not a new message: the open record is asked again (ActionReplyPolicy REPROMPT), so
    the approval stays open and the flag stays until it is finally answered."""
    _held(todo, "Ann")
    out = todo.ctx.process_turn("maybe")
    assert todo.store.active() is not None
    assert _added(todo.chores, todo.base) == []
    assert todo.script.steps == []
    assert "Ambiguous intent" not in out.answer
    assert _flag(todo)
    todo.ctx.process_turn("no")
    assert todo.store.active() is None
    _plain_message_reaches_the_agent(todo)
    assert not _flag(todo)


def test_a_failed_slash_command_keeps_the_flag_for_the_pending_parameter_round(session):
    """(d) The case that must not change: a "/" command missing its parameters leaves the flag set, so the next bare
    message (``abort``) stays on the assistant path in an agent session (the round trips are pinned in
    tests/test_parameter_form_reply_contract.py)."""
    ctx = session(True)
    out = ctx.process_turn(PARTIAL)
    assert "parameter_request" in out.command_outputs[-1].command_response.artifacts
    assert ctx.assistant_round
    ctx.process_turn("abort")
    assert ctx.agent_invoked == []


@pytest.mark.parametrize("reply, notes", REPLIES)
def test_assistant_session_is_unchanged(todo, annotated_library, grant_root, reply, notes):
    """(c) Without an agent the same sequence resolves the approval and plain messages stay deterministic. The flag
    is not read for routing there, so clearing it changes nothing a user can see."""
    todo.ctx._run_as_agent = False
    todo.ctx._cme_workflow.context.pop("run_as_agent", None)
    todo.wf.context.pop("run_as_agent", None)
    _held(todo, "Ann")
    todo.script.steps = []
    todo.ctx.process_turn(reply)
    assert _added(todo.chores, todo.base) == notes
    assert todo.store.active() is None
    assert not _flag(todo)
    todo.ctx.process_turn("do it")                           # never reaches the agent: nothing scripted
    assert todo.script.steps == []
    todo.ctx.process_turn("abort")                           # leave the ambiguity round "do it" opened
    out = todo.ctx.process_turn("/add_note text=Cy")
    assert out.command_outputs[-1].command_response.artifacts["awaiting_user"] is True


# ----------------------------------------------------------------------
# A pending "/" approval (or parameter round) dropped without a reply
# ----------------------------------------------------------------------

DROPS = [
    pytest.param(lambda ctx: ctx.bind_observability_identity(conversation_id=902), id="conversation-bound"),
    pytest.param(lambda ctx: ctx.park_pending_for_new_conversation(), id="new-conversation-button"),
    pytest.param(lambda ctx: ctx.cancel_pending_approvals(), id="cancel-pending-approvals"),
]


@pytest.mark.parametrize("drop", DROPS)
def test_a_dropped_slash_approval_does_not_strand_the_session_on_the_assistant_path(
        todo, annotated_library, grant_root, drop):
    todo.ctx.bind_observability_identity(conversation_id=901)
    _held(todo, "Ann")
    drop(todo.ctx)
    assert todo.store.active() is None
    _plain_message_reaches_the_agent(todo)
    assert _added(todo.chores, todo.base) == []
    assert not _flag(todo)


@pytest.mark.parametrize("drop", DROPS[:2])
def test_a_new_conversation_does_not_strand_a_session_with_a_pending_parameter_round(session, drop):
    """No approval at all: the new conversation resets the NLU stage, so the "/" flag has no meaning in it."""
    ctx = session(True)
    ctx.bind_observability_identity(conversation_id=901)
    out = ctx.process_turn(PARTIAL)
    assert "parameter_request" in out.command_outputs[-1].command_response.artifacts
    drop(ctx)
    with pytest.raises(AssertionError, match="the agent was invoked"):
        ctx.process_turn("do it")
    assert ctx.agent_invoked, "the plain message did not reach the agent"
