"""A part bound to an EXECUTION command that lacks a required value is asked for it on the plan path (no agent model),
through the one parameter clarification loop a planning command's missing value is asked through, as the "/" path asks.

Library example, no agent model, the planner's split, the matcher and the extractor scripted (``tests/test_parameter_loop.py``)."""
from __future__ import annotations

import pytest

from fastworkflow.utils.signatures import InputForParamExtraction
from tests.conftest import script_matcher
from tests.test_parameter_loop import HOST, no_agent, _offline, BOOK_PLAN  # noqa: F401
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_approval_binding import annotated_library, grant_root, todo, no_live_model, no_model_call_for_parameters  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FIND_BOOK, AUTHOR = "Library/find_book", "Library/author_books"
ASK = "find a book"


def _nothing_extracted(monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", lambda self, cls, name, folder: cls.model_construct())


def _artifacts(out):
    return out.command_outputs[-1].command_response.artifacts


def test_a_picked_execution_command_with_a_value_missing_is_asked_for_it(no_agent, monkeypatch):
    _nothing_extracted(monkeypatch)
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0})
    no_agent.parts = [JOKE]
    asked = no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    assert _artifacts(asked)["question_kind"] == "plan_choice"

    picked = no_agent.ctx.process_turn(FIND_BOOK, approval_surface=HOST)
    artifacts = _artifacts(picked)
    assert artifacts.get("question_kind") == "planning_parameter", picked.answer
    assert artifacts["parameter_request"]["command"] == FIND_BOOK and artifacts["parameter_request"]["missing"] == ["query"]
    assert "not run" not in picked.answer.lower()
    held = no_agent.ctx._waiting
    assert held.kind == "parameters" and held.owner == "plan"

    ran = no_agent.ctx.process_turn("query: Dune", approval_surface=HOST)
    assert ran.status.name == "COMPLETED", ran.answer
    assert "Dune" in ran.answer
    assert no_agent.ctx._waiting is None
    # the user's own answer settled it: no review of the turn
    assert _artifacts(ran).get("question_kind") != "review"


def test_cancelling_the_value_question_of_a_picked_execution_command_ends_the_part(no_agent, monkeypatch):
    _nothing_extracted(monkeypatch)
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0})
    no_agent.parts = [JOKE]
    no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    no_agent.ctx.process_turn(FIND_BOOK, approval_surface=HOST)
    out = no_agent.ctx.process_turn("cancel", approval_surface=HOST)
    entry = _artifacts(out)["fw_not_run"][0]
    assert entry["command"] == FIND_BOOK and "declined" in entry["reason"]


def test_an_execution_command_named_by_the_user_with_a_value_missing_is_asked_for_it(no_agent, monkeypatch):
    _nothing_extracted(monkeypatch)
    script_matcher(monkeypatch, lambda t: {"none": 0.0})
    no_agent.parts = ["find_book"]
    out = no_agent.ctx.process_turn("find_book", approval_surface=HOST)
    artifacts = _artifacts(out)
    assert artifacts.get("question_kind") == "planning_parameter", out.answer
    assert artifacts["parameter_request"]["missing"] == ["query"]
    ran = no_agent.ctx.process_turn("Dune", approval_surface=HOST)
    assert ran.status.name == "COMPLETED" and "Dune" in ran.answer, ran.answer


def test_an_execution_command_the_local_model_is_sure_of_with_a_value_missing_is_asked_for_it(no_agent, monkeypatch):
    """The fast path (one part, no review) asks as well: it is the same part, the same loop."""
    from fastworkflow._workflows.plan_metadata_extraction import resolution
    from tests.test_round1_gate import FIND

    _nothing_extracted(monkeypatch)
    script_matcher(monkeypatch, lambda t: {"none": 0.0})
    no_agent.parts = [FIND]
    intent = resolution.part_intent(no_agent.wf, FIND)
    assert resolution.is_fast_path(intent, no_agent.wf) and intent.kind == resolution.KIND_EXECUTION
    out = no_agent.ctx.process_turn(FIND, approval_surface=HOST)
    artifacts = _artifacts(out)
    assert artifacts.get("question_kind") == "planning_parameter", out.answer
    assert artifacts["parameter_request"]["command"] == intent.command and artifacts["parameter_request"]["missing"]
    ran = no_agent.ctx.process_turn("query: Ann Lee", approval_surface=HOST)
    assert ran.status.name == "COMPLETED", ran.answer
    assert _artifacts(ran).get("question_kind") != "review"
