"""A '/' parameter round keeps its assistant routing when it is parked or rebuilt from saved state (fix-icce).

A failed '/' command leaves a parameter round pending and relies on ``is_assistant_mode_command`` so the next bare reply
stays on the deterministic path in an agent session. The flag is saved with the round (serialized state and parked
snapshot) and restored with it. Real WorkflowExecutionContext on a copy of tests/plain_required_workflow; the agent
entry points and the LLM extractor fail loudly if reached.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow import NLUPipelineStage
from tests.test_parameter_form_reply_contract import PARTIAL, _last, _ran, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FLAG = "is_assistant_mode_command"
CONV_A, CONV_B = 901, 902


def _saved(ctx) -> dict:
    return json.loads(json.dumps(ctx.serialize_state(channel_id="icce")))


def _open_round(ctx):
    out = ctx.process_turn(PARTIAL)
    assert "parameter_request" in _last(out).artifacts
    assert FLAG in ctx._cme_workflow.context


def test_returning_to_a_parked_round_keeps_the_deterministic_path(session):
    ctx = session(True)
    ctx.bind_observability_identity(conversation_id=CONV_A)
    _open_round(ctx)
    ctx.bind_observability_identity(conversation_id=CONV_B)
    assert FLAG not in ctx._cme_workflow.context            # nothing leaks into the other conversation
    ctx.bind_observability_identity(conversation_id=CONV_A)
    out = ctx.process_turn("abort")
    assert ctx.agent_invoked == []
    assert not _ran(out)


def test_a_round_parked_in_a_does_not_route_b_to_the_assistant(session):
    ctx = session(True)
    ctx.bind_observability_identity(conversation_id=CONV_A)
    _open_round(ctx)
    ctx.bind_observability_identity(conversation_id=CONV_B)
    with pytest.raises(AssertionError, match="the agent was invoked"):
        ctx.process_turn("do it")
    assert ctx.agent_invoked


def test_a_rebuilt_context_keeps_the_deterministic_path_mid_round(session):
    ctx = session(True)
    _open_round(ctx)
    state = _saved(ctx)
    assert state["nlu_stage"] == NLUPipelineStage.PARAMETER_EXTRACTION.value
    rebuilt = session(True)
    rebuilt.apply_serialized_state(state)
    out = rebuilt.process_turn("abort")
    assert rebuilt.agent_invoked == []
    assert not _ran(out)


def test_a_state_saved_without_the_key_loads_as_before(session):
    ctx = session(True)
    _open_round(ctx)
    state = _saved(ctx)
    state.pop("assistant_command", None)
    rebuilt = session(True)
    rebuilt.apply_serialized_state(state)                   # loads; the flag is simply not restored
    assert FLAG not in rebuilt._cme_workflow.context


def test_a_session_with_no_round_saves_no_flag(session):
    ctx = session(True)
    assert not _saved(ctx).get("assistant_command")
    rebuilt = session(True)
    rebuilt.apply_serialized_state(_saved(ctx))
    assert FLAG not in rebuilt._cme_workflow.context


def test_a_finished_round_leaves_nothing_to_restore_so_the_agent_stays_reachable(session):
    ctx = session(True)
    _open_round(ctx)
    ctx.process_turn("abort")
    state = _saved(ctx)
    assert state["assistant_command"] is False
    rebuilt = session(True)
    rebuilt.apply_serialized_state(state)
    with pytest.raises(AssertionError, match="the agent was invoked"):
        rebuilt.process_turn("do it")
    assert rebuilt.agent_invoked
