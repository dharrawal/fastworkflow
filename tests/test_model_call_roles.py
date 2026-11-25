"""Every model call's span says whose model it is (span contract 63, owner decision 2026-11-25).

The ``fw.llm.call`` span's ``role`` is the caller's own model setting (``LLM_PLANNER``, ``JEV``, ...), named where the
call is made (``dspy_utils.role_context``), so it is there whichever LM object the caller holds: the LMs here are
scripted ``DummyLM`` doubles that ``get_lm`` never tagged. Real components: the real splitter, the one-job gate, the
review and its free-text edit reader on the library copy.
"""
from __future__ import annotations

import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import remote_outage, tracing, workflow_agent
from fastworkflow.turn import TurnStatus
from tests import span_tree
from tests.test_approval_binding import (  # noqa: F401 - the real agent on a library copy, and the guards
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _open_review, gated, no_draft_extraction  # noqa: F401
from tests.test_split_one_job_gate import REAL_SPLIT, _splitter
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "recorded_review_router")


@pytest.fixture
def gate_on(monkeypatch):
    """The one-job gate on (``tests/conftest.py`` turns it off), its remote model a scripted double with no answer: the
    gate's call is made and fails, so the splitter runs."""
    remote_outage.reset()
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        name: value for name, value in fastworkflow._env_vars.items()
        if name != workflow_agent.SPLIT_GATE_THRESHOLD_SETTING})
    monkeypatch.delenv(workflow_agent.SPLIT_GATE_THRESHOLD_SETTING, raising=False)
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent, "_choose_referent", lambda *args: None)
    yield
    remote_outage.reset()


def test_every_model_call_of_a_split_gated_reviewed_turn_names_its_role(gated, gate_on, monkeypatch,
                                                                         no_draft_extraction):
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", REAL_SPLIT)
    gated.ctx._current_planner_lm = _splitter([EMMA, JOKE])
    asked = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert asked.status is TurnStatus.AWAITING_USER
    # each part's which-command question (the head is unsure, and the remote double gives no answer), then the review
    for _ in range(2):
        assert "Which one is meant?" in asked.answer
        asked = gated.ctx.process_turn("1")
    _open_review(gated)
    # a free-text reply: the review's edit reader reads it with the planner's model
    reader = DummyLM([{"edits": []}] * 4)
    gated.ctx._current_planner_lm = reader
    gated.ctx.process_turn("use book_by_title for the second part")

    assert len(reader.history) == 1
    calls = span_tree.named(gated.sink, tracing.SPAN_LLM_CALL)
    by_parent = {span.span_id: span.name for span in span_tree.spans(gated.sink)}
    where = {by_parent.get(call.parent_span_id) for call in calls}
    # the gate's call, the splitter's and the edit reader's were all made
    assert {tracing.SPAN_SPLIT_GATE, tracing.SPAN_SPLIT_SPLITTER, tracing.SPAN_REVIEW_EDIT} <= where, where
    assert [(by_parent.get(call.parent_span_id), call.attributes.get("role")) for call in calls
            if not call.attributes.get("role")] == []
    roles = {by_parent.get(call.parent_span_id): call.attributes["role"] for call in calls}
    assert roles[tracing.SPAN_SPLIT_GATE] == "JEV" and roles[tracing.SPAN_SPLIT_SPLITTER] == "LLM_PLANNER"
    assert roles[tracing.SPAN_REVIEW_EDIT] == "LLM_PLANNER"
