"""The three words the chat approval card sends are the words the server accepts (fix-weih.12).

The card (``257-chat-approval.js``) sends exactly ``yes``, ``always`` or ``no`` as ordinary chat text.
``review_engine.ActionReplyPolicy`` reads a yes with the one yes reader (``no_agent_mode.is_yes``: yes, true, ok,
agreed, accept or go as the whole reply, case and a trailing ``.`` or ``!`` ignored) and ``always`` and ``no`` exactly, so
a near miss such as ``"y"`` or ``"yes please"`` does not approve. The buttons send the lowercase words.
Proven for both waiting shapes: an agent-originated approval (turn status AWAITING_USER) and a user-typed "/"
approval (turn status COMPLETED with ``artifacts.awaiting_user``). Real WorkflowExecutionContext, real approval
store, the real agent with scripted steps. Any model call, or an agent step nobody scripted, fails the test.
"""
from __future__ import annotations

import pytest

from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _added, _ask, _finish, annotated_library, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

APPROVE, ALWAYS, DECLINE = "yes", "always", "no"   # the card's TM_APPROVAL_REPLIES
NEAR_MISSES = ["y", "run it", "yes please", " yes please ", "No thanks", "always please"]
OTHER_YESES = ["Yes", "YES", "yes.", "ok", "agreed", "Go!"]   # the one yes reader's words: each approves


def _last(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def _agent_approval(todo, text="Ann"):
    out = _ask(todo, f"add_note <text>{text}</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    assert _last(out)["question_kind"] == "approval" and _last(out)["approval"]["origin"] == "agent"
    return out


def _user_approval(todo, text="Ann"):
    out = todo.ctx.process_turn(f"/add_note text={text}")
    assert out.status is TurnStatus.COMPLETED
    assert _last(out)["awaiting_user"] is True
    assert _last(out)["approval"]["origin"] == "user"
    return out


SHAPES = [pytest.param(_agent_approval, id="agent"), pytest.param(_user_approval, id="user-slash")]


@pytest.mark.parametrize("make", SHAPES)
def test_yes_runs_the_command_once(todo, annotated_library, grant_root, make):
    make(todo)
    assert _added(todo.chores, todo.base) == []
    todo.script.steps = _finish()
    todo.ctx.process_turn(APPROVE)
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert todo.store.active() is None
    # A second yes has nothing pending and cannot run it again.
    todo.script.steps = _finish()
    todo.ctx.process_turn(APPROVE)
    assert _added(todo.chores, todo.base) == ["Ann"]


@pytest.mark.parametrize("make", SHAPES)
def test_no_does_not_run_and_leaves_nothing_pending(todo, annotated_library, grant_root, make):
    make(todo)
    record = todo.store.active()
    todo.script.steps = _finish()
    todo.ctx.process_turn(DECLINE)
    assert _added(todo.chores, todo.base) == []
    assert record.state == "declined"
    assert todo.store.active() is None
    assert todo.ctx.awaiting_user is False


def test_always_runs_it_and_records_a_grant_so_the_next_call_needs_no_prompt(todo, annotated_library, grant_root):
    # bd fix-qjne.11: the standing approval is a delegatable command's (show_notes); add_note is REQUIRED_ALWAYS.
    first = _ask(todo, "show_notes", steps=_finish())
    assert first.status is TurnStatus.AWAITING_USER
    assert _last(first)["approval"]["can_always"] is True      # the card shows "Always allow"
    todo.script.steps = _finish()
    todo.ctx.process_turn(ALWAYS)
    assert todo.store.active() is None
    # The grant stands: the same command needs no prompt next time.
    out = _ask(todo, "show_notes", steps=_finish())
    assert out.status is TurnStatus.COMPLETED
    assert not _last(out).get("awaiting_user")
    assert "Record/show_notes" in todo.ctx.process_turn("/approvals").answer


@pytest.mark.parametrize("make", SHAPES)
def test_always_is_not_offered_or_accepted_for_a_required_always_command(todo, annotated_library, grant_root, make):
    first = make(todo)
    assert _last(first)["approval"]["can_always"] is False     # the card hides "Always allow"
    todo.script.steps = []
    try:
        todo.ctx.process_turn(ALWAYS)
    except AssertionError as err:   # the scripted agent was asked for a step: still not approved
        assert "asked for a step" in str(err)
    assert _added(todo.chores, todo.base) == []                # not read as yes
    assert todo.store.active() is not None                     # the question is still open
    assert "Record/add_note" not in todo.ctx.process_turn("/approvals").answer   # nothing recorded


@pytest.mark.parametrize("make", SHAPES)
@pytest.mark.parametrize("word", NEAR_MISSES)
def test_a_near_miss_does_not_approve(todo, annotated_library, grant_root, make, word):
    """Why the buttons send exact lowercase words: anything else is not an approval and costs a round."""
    make(todo)
    todo.script.steps = []          # reaching the agent for a step would fail loudly
    try:
        todo.ctx.process_turn(word)
    except AssertionError as err:   # the scripted agent was asked for a step: still not approved
        assert "asked for a step" in str(err)
    assert _added(todo.chores, todo.base) == []
    assert todo.store.active() is not None


@pytest.mark.parametrize("make", SHAPES)
@pytest.mark.parametrize("word", OTHER_YESES)
def test_every_yes_of_the_one_reader_approves(todo, annotated_library, grant_root, make, word):
    """The approval question reads a yes with the same reader as every other yes (owner decision 2026-11-23)."""
    make(todo)
    todo.script.steps = _finish()
    todo.ctx.process_turn(word)
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert todo.store.active() is None


@pytest.mark.parametrize("make", SHAPES)
@pytest.mark.parametrize("word", ["No", "no.", "NO!"])
def test_no_is_read_as_a_yes_is_whole_reply_case_and_end_mark_ignored(todo, annotated_library, grant_root, make, word):
    make(todo)
    record = todo.store.active()
    todo.script.steps = _finish()
    todo.ctx.process_turn(word)
    assert _added(todo.chores, todo.base) == [] and record.state == "declined"
