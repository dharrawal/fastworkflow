"""The CME holds ONE pending question at a time (bd fix-fpyb.4): a which-command question (``intent_loop``) or a
parameter question (``parameter_loop``), never both. Real WorkflowExecutionContext on a copy of
tests/plain_required_workflow, deterministic path; the LLM extractor and the agent entry points fail loudly."""
from __future__ import annotations

import json

import pytest

from fastworkflow.intent import intent_loop
from fastworkflow.parameters import parameter_clarification
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _open(ctx) -> set:
    """What the CME's context says is pending, read through the helpers (the record itself is the turn's)."""
    context = ctx._cme_workflow.context
    return {name for name, held in (("intent", intent_loop.pending_record(context)),
                                    ("parameters", parameter_clarification.pending(context))) if held is not None}


def _say(ctx, text):
    """The reply as the CME's router reads it (the executor's own continuation is not what is under test)."""
    from fastworkflow._workflows.command_metadata_extraction._commands.wildcard import ResponseGenerator

    return ResponseGenerator()(ctx._cme_workflow, text, None)


def test_you_misunderstood_during_a_parameter_question_leaves_one_record(session):
    ctx = session(False)
    ctx.process_turn(PARTIAL)
    assert _open(ctx) == {"parameters"}
    _say(ctx, "you misunderstood")
    assert len(_open(ctx)) == 1, _open(ctx)
