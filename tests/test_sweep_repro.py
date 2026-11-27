"""Reproductions for four open issues, one test each (issue verification sweep, 2026-11-20).

A test marked ``xfail(strict=True)`` asserts the CORRECT behaviour and fails today, so it reproduces its issue; when the
issue is fixed the strict xfail turns the run red and the mark is to be removed. An unmarked test passes today: it is
evidence the issue is already fixed. No model call: the real components with the model seams scripted the way the
files imported from do.
"""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow import intent_loop
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.parameters import check
from fastworkflow.review.review_proposals import MEANS_FROM_SCRATCH
from fastworkflow.turn import TurnStatus
from tests.conftest import script_matcher
from tests.test_agent_command_not_text import ctx, todo_workflow_path  # noqa: F401 - an agent-mode context over the todo workflow
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401 - the CME on plain_required_workflow
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.test_unreviewed_none_veto import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    BY_TITLE, EMMA, JOKE, _finish, _split_into, annotated_library, gated, no_draft_extraction, no_live_model,
    no_model_call_for_parameters, todo)

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


class _Find(BaseModel):
    query: str = Field(description="what to look for, for example Ann",
                       json_schema_extra={"copied_from_command_text": True})
    kind: Optional[str] = None
    start: Optional[int] = None


def _validated(value: str, text: str, command_name: str = "find_application"):
    """The execution command's validation step on the agent's text (``InputForParamExtraction.validate_parameters``):
    since bd fix-qoyz the written-in-the-request check is made only on a field marked ``copied_from_command_text``, by
    the one guard (``planning_commands.option_command`` -> ``check_plan``), which reads the text with a leading
    own-command word stripped (``fill.without_own_command_word``). ``(is_valid, message, refused field names)``."""
    from fastworkflow.parameters.signatures import InputForParamExtraction

    step = InputForParamExtraction(command=text, input_for_param_extraction_class=None)
    is_valid, message, _suggestions, fields = step.validate_parameters(None, command_name, _Find(query=value))
    return is_valid, message, fields


def test_a_value_only_the_command_name_spells_is_not_grounded_in_the_agents_text():
    """fix-2juj: 'Application' is spelled only inside the command word 'find_application' the agent's text begins with,
    so it is not written in the request: a marked field holding it is refused and asked for, on the real path."""
    is_valid, message, fields = _validated("Application", "find_application SCOM")
    assert not is_valid and fields == ["query"]
    assert "the query 'Application' is not in your request" in message


@pytest.mark.parametrize("value, text, command_name", [
    ("SCOM", "find_application SCOM", "find_application"),
    ("Application", "find_application Application", "find_application"),     # also written after the word
    ("SCOM", "Ctx/find_application SCOM", "Ctx/find_application"),
    ("Application", "find application Application", "find_application"),     # the user's own words are not the command token
])
def test_a_value_the_request_writes_still_passes_the_grounding_guard(value, text, command_name):
    is_valid, message, fields = _validated(value, text, command_name)
    assert is_valid and fields == [], message


def test_cancel_pending_ends_a_question_the_deterministic_path_holds(session):  # noqa: F811
    context = session(False)
    first = context.process_turn(PARTIAL)
    assert context.waiting is not None and context.waiting.owner == "cme"
    assert first.status is TurnStatus.AWAITING_USER or context.waiting is not None
    assert context.cancel_pending() is True
    assert context.waiting is None
    assert not context.has_open_command()


def test_cancel_pending_ends_the_deterministic_which_command_question(session):  # noqa: F811
    from tests.test_cme_waiting_record import _which_command_question

    context = _which_command_question(session)
    assert context.waiting.kind == "intent" and context.waiting.owner == "cme"
    assert context.cancel_pending() is True
    assert context.waiting is None
    # nothing is out any more: a second cancel has nothing to drop, and the next message is a request
    assert context.cancel_pending() is False
    context.process_turn(PARTIAL)
    assert context.waiting is not None and context.waiting.kind == "parameters"


def test_an_unreviewed_draft_binds_only_a_confident_match(gated, monkeypatch):  # noqa: F811
    """fix-6lpc: a remote 'none' below the old 0.9 veto bar, with a weak pick, leaves the part to the agent."""
    script_matcher(monkeypatch, {EMMA: {BY_TITLE: 0.15, "none": 0.8}})
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert gated.ctx._turn_subtask_plan.items[0].means.kind == MEANS_FROM_SCRATCH


def test_blank_text_reaches_no_intent_detection(ctx):  # noqa: F811
    """fix-wfxt: blank text is no intent decision and no executor dispatch (the predict check raised on it)."""
    workflow = ctx.get_active_workflow()
    for text in ("", "   ", "\t\n"):
        assert intent_loop.resolve(workflow, text).outcome == intent_loop.NONE
        output = CommandExecutor.invoke_command(ctx, text)
        assert "hit the <Enter> key" in output.command_response.response
