"""A reply to the CME's which-command question is read as the reply while the question is out (nothing-matched
question and the listing after "you misunderstood" included), in an agent session as in an assistant one: the "/"
round ends when the question ends, never while its record is out. Real ``WorkflowExecutionContext`` on a copy of
``tests/plain_required_workflow`` (the fixtures of ``tests/test_cme_waiting_record.py``)."""
from __future__ import annotations

import pytest

from fastworkflow import intent_loop
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _open_which_command(ctx, how):
    if how == "misunderstood":
        ctx.process_turn(PARTIAL)
        out = ctx.process_turn("you misunderstood")
    else:
        out = ctx.process_turn("/zzqx wibble blorp")
    assert ctx._waiting is not None and ctx._waiting.kind == "intent", out.answer
    return out


@pytest.mark.parametrize("how", ["misunderstood", "nothing_matched"])
@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
@pytest.mark.parametrize("prefix", ["", "/"], ids=["bare_reply", "slash_reply"])
def test_the_reply_to_the_which_command_question_is_read_as_the_reply(session, how, run_as_agent, prefix):
    ctx = session(run_as_agent)
    session.llm.allowed = True          # the picked command's extraction finds nothing: its parameter question opens
    _open_which_command(ctx, how)
    assert ctx.assistant_round and ctx._waiting.assistant      # the round is kept while the question is out

    out = ctx.process_turn(prefix + "record_values")
    assert ctx.agent_invoked == []
    assert "plan_choice" not in (out.command_outputs[-1].command_response.artifacts or {})
    # picked: the command's parameter question is out and the round still holds
    assert intent_loop.waiting_for(ctx._cme_workflow.context) == intent_loop.WAITING_PARAMETERS
    assert ctx.assistant_round


@pytest.mark.parametrize("how", ["misunderstood", "nothing_matched"])
@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
def test_the_round_ends_when_the_question_is_cancelled(session, how, run_as_agent):
    ctx = session(run_as_agent)
    _open_which_command(ctx, how)
    out = ctx.process_turn("cancel")
    assert ctx.agent_invoked == []
    assert ctx._waiting is None and not ctx.assistant_round
    assert "aborted" in out.command_outputs[-1].command_response.response


@pytest.mark.parametrize("how", ["misunderstood", "nothing_matched"])
def test_cancel_pending_ends_the_round(session, how):
    ctx = session(True)
    _open_which_command(ctx, how)
    assert ctx.cancel_pending() is True
    assert ctx._waiting is None and not ctx.assistant_round


@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
def test_two_unreadable_replies_end_the_question_and_the_round(session, run_as_agent):
    ctx = session(run_as_agent)
    _open_which_command(ctx, "nothing_matched")
    ctx.process_turn("hmm unsure")
    assert ctx._waiting is not None and ctx.assistant_round
    ctx.process_turn("really unsure")
    assert ctx.agent_invoked == []
    assert ctx._waiting is None and not ctx.assistant_round


@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
def test_the_slash_parameter_question_carries_the_kind_of_the_record(session, run_as_agent):
    """The `/` path's parameter question is tagged ``question_kind`` like the plan path's (``waiting.TABLE``), beside its
    ``parameter_request``; the turn stays a completed one (the chat page draws its form from a completed turn)."""
    from fastworkflow import waiting

    ctx = session(run_as_agent)
    out = ctx.process_turn(PARTIAL)
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts["question_kind"] == waiting.TABLE[waiting.KIND_PARAMETERS].question_kind == "planning_parameter"
    assert artifacts["parameter_request"]["command"].endswith("record_values")
    assert out.status.name == "COMPLETED"
