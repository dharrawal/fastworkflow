"""The turn span records the split of a turn whose request the model was asked to split (bd fix-hfqv; recording only).

``fw.turn`` ``subtask_split`` = {request, raw, merged, attempts, changed_by}: the request as the splitter saw it, the model's
sub-tasks before any code rule, the list the turn runs with, and the code steps that changed it. Real components throughout,
no mocks: the planner's model is the scripted ``DummyLM`` the neighbouring split tests use, so the real merge rules run.
"""
from __future__ import annotations

import json

import pytest
from dspy.utils import DummyLM

from fastworkflow import split_merge, tracing, workflow_agent
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_plan_reply_policy import _wire_plan, planner, script_param_extraction  # noqa: F401
from tests.test_round1_gate import _split_into, gated, no_draft_extraction  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

#: The real split: the ``todo`` fixture the gate tests build on replaces it with "the request whole".
REAL_SPLIT = workflow_agent.split_into_subtasks

#: No planner key: rule 3's joins are then the lexical rule's, with no model call.
pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "no_planner_key")

LIBRARY = "fastworkflow/examples/library"
REQUEST = "Find the book Dune, then list its author and its year, and tell Ann a joke."
RAW = ["Find the book Dune.", "List the author of Dune.", "List the year of Dune.", "Tell Ann a joke."]
MERGED = ["Find the book Dune. List the author of Dune. List the year of Dune.", "Tell Ann a joke."]
INDEPENDENT_REQUEST = "Find the book Dune. Find the book Emma."
INDEPENDENT = ["Find the book Dune.", "Find the book Emma."]
#: The library example has no _workflow.py: the built-in prompt (span contract v61).
PROMPT = {"prompt_digest": workflow_agent.split_prompt(LIBRARY)["digest"], "prompt_source": "built_in"}


def _lm(items: list[str]) -> DummyLM:
    text = "\n".join(f"{n}. {item}" for n, item in enumerate(items, 1))
    return DummyLM([{"reasoning": "r", "subtasks": text}])


def _turn_span(todo):
    return [span for span in todo.sink.spans if span.name == tracing.SPAN_TURN][-1]


def test_the_recorder_holds_request_raw_merged_and_the_rule_that_changed_the_list():
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(REQUEST, planner_lm=_lm(RAW), workflow_folderpath=LIBRARY)
    assert out == MERGED
    assert record == {"request": REQUEST, "raw": RAW, "merged": MERGED, "attempts": 1, "changed_by": ["group"],
                      **PROMPT}


def test_a_split_no_rule_changes_has_no_changed_by():
    with workflow_agent.recording_split() as record:
        workflow_agent.split_into_subtasks(INDEPENDENT_REQUEST, planner_lm=_lm(INDEPENDENT), workflow_folderpath=LIBRARY)
    assert record["raw"] == record["merged"] == INDEPENDENT
    assert "changed_by" not in record


def test_a_single_part_split_is_recorded():
    request = "Tell Ann a joke."
    with workflow_agent.recording_split() as record:
        workflow_agent.split_into_subtasks(request, planner_lm=_lm([request]), workflow_folderpath=LIBRARY)
    assert record == {"request": request, "raw": [request], "merged": [request], "attempts": 1, **PROMPT}


def test_nothing_is_recorded_without_a_recorder_or_when_no_split_was_asked():
    assert workflow_agent.split_into_subtasks(REQUEST, planner_lm=_lm(RAW), workflow_folderpath=LIBRARY) == MERGED
    with workflow_agent.recording_split() as record:
        pass
    assert record == {}


def test_merge_subtasks_reports_its_rules_without_changing_its_result():
    identifiers = workflow_agent.identifier_patterns(LIBRARY)
    applied: list[str] = []
    with_report = split_merge.merge_subtasks(REQUEST, RAW, identifiers, applied=applied)
    assert with_report == split_merge.merge_subtasks(REQUEST, RAW, identifiers)
    assert applied == ["group"]
    unchanged: list[str] = []
    assert split_merge.merge_subtasks(INDEPENDENT_REQUEST, INDEPENDENT, identifiers, applied=unchanged) == INDEPENDENT
    assert unchanged == []


def test_a_split_turn_carries_subtask_split_on_its_span(gated, monkeypatch, no_draft_extraction):
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", REAL_SPLIT)
    gated.ctx._current_planner_lm = _lm(RAW)
    gated.script.steps = _finish() * 4
    out = gated.ctx.process_turn(REQUEST, unattended=True)
    assert out.status is TurnStatus.COMPLETED
    recorded = _turn_span(gated).attributes["subtask_split"]
    assert recorded["request"] == REQUEST
    assert recorded["raw"] == RAW and recorded["merged"] == MERGED
    assert recorded["changed_by"] == ["group"] and recorded["attempts"] == 1
    json.dumps(recorded)


def test_a_split_turn_the_rules_leave_alone_has_equal_lists_and_no_changed_by(gated, monkeypatch, no_draft_extraction):
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", REAL_SPLIT)
    gated.ctx._current_planner_lm = _lm(INDEPENDENT)
    gated.script.steps = _finish() * 4
    gated.ctx.process_turn(INDEPENDENT_REQUEST, unattended=True)
    recorded = _turn_span(gated).attributes["subtask_split"]
    assert recorded["raw"] == recorded["merged"] == INDEPENDENT
    assert "changed_by" not in recorded


def test_a_turn_that_was_not_split_has_no_subtask_split(gated, monkeypatch, no_draft_extraction):
    _split_into(monkeypatch, ["Tell Ann a joke."])
    gated.script.steps = _finish() * 2
    gated.ctx.process_turn("Tell Ann a joke.", unattended=True)
    assert _turn_span(gated).attributes.get("subtask_split") is None


def test_the_attribute_is_part_of_the_turn_contract():
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_TURN]
    assert "subtask_split" in contract.attributes and contract.version == 9
    assert tracing.split_record_attribute(None) is None and tracing.split_record_attribute({}) is None
    long_request = "x" * (tracing.MAX_ATTR_BYTES + 10)
    capped = tracing.split_record_attribute(
        {"request": long_request, "raw": [long_request], "merged": ["a"], "attempts": 1})
    assert capped["request"]["truncated"] is True and capped["raw"][0]["truncated"] is True


def test_a_split_whose_every_answer_stops_at_the_cap_is_recorded_as_failed(caplog):
    """The request runs as one part, and the turn says so: ``failed`` on the record (and so on the span) and a log line."""
    from tests.test_text_planner_and_split import _ECHOED_THEN_CUT, _ScriptedReplies

    lm = _ScriptedReplies([_ECHOED_THEN_CUT, _ECHOED_THEN_CUT])
    with caplog.at_level("WARNING"), workflow_agent.recording_split() as record:
        assert workflow_agent.split_into_subtasks("the request", planner_lm=lm) == ["the request"]
    assert record["failed"] == workflow_agent.SPLIT_FAILED_CUT_OFF and record["merged"] == ["the request"]
    assert tracing.split_record_attribute(record)["failed"] == "cut_off"
    assert any("the split failed" in line for line in caplog.messages)


def test_a_split_that_answers_has_no_failed_key():
    with workflow_agent.recording_split() as record:
        workflow_agent.split_into_subtasks(INDEPENDENT_REQUEST, planner_lm=_lm(INDEPENDENT), workflow_folderpath=LIBRARY)
    assert "failed" not in record and "failed" not in tracing.split_record_attribute(record)
