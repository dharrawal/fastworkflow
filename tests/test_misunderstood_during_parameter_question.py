"""``you misunderstood`` as the reply to a pending parameter question, through the public turn entry.

Real WorkflowExecutionContext on a copy of tests/plain_required_workflow, human deterministic session; the LLM
extractor and the agent entry points fail loudly. The reply ends the parameter question and the user gets the
which-command question, as a normal turn."""
from __future__ import annotations

import pytest

from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.mark.parametrize("reply", ["you misunderstood", "/you misunderstood"])
def test_misunderstood_reply_to_a_parameter_question_gives_the_which_command_question(session, reply):
    ctx = session(False)
    first = ctx.process_turn(PARTIAL)
    assert first.command_outputs[-1].command_response.artifacts.get("parameter_request") is not None
    out = ctx.process_turn(reply)
    response = out.command_outputs[-1].command_response
    assert (response.artifacts or {}).get("plan_choice") is not None, response
    assert ctx.agent_invoked == [] and session.llm.calls == 0
