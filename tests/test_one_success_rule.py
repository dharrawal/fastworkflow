"""One success rule on both paths: a turn that ran to its end is COMPLETED, and whether a part did what was asked is
said by that part's own entry (its command output's ``success``, the ``fw_not_run`` entry). FAILED is for a turn that
could not be completed (agent exhaustion, an error of the framework itself).

Library example, no agent model, the matcher and the extractor scripted (``tests/test_parameter_loop.py``)."""
from __future__ import annotations

import sys

import pytest

from fastworkflow.turn import TurnStatus
from fastworkflow.utils.signatures import InputForParamExtraction
from tests.conftest import script_matcher
from tests.test_parameter_loop import HOST, no_agent, _offline, BOOK_PLAN  # noqa: F401
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_approval_binding import annotated_library, grant_root, todo, no_live_model, no_model_call_for_parameters  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FIND_BOOK, AUTHOR = "Library/find_book", "Library/author_books"


def _artifacts(out):
    return out.command_outputs[-1].command_response.artifacts


def _response_generator(module_suffix: str):
    modules = [m for n, m in sys.modules.items() if n.endswith(module_suffix) and "annotated" in n]
    assert modules, module_suffix
    return modules[-1].ResponseGenerator


def test_a_slash_command_whose_code_raises_is_a_completed_turn_with_a_failed_output(no_agent, monkeypatch, caplog):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(query="Dune"))

    def boom(self, workflow, command, input):
        raise RuntimeError("the shelf fell over")

    monkeypatch.setattr(_response_generator("Library.find_book"), "__call__", boom)
    out = no_agent.ctx.process_turn("/find_book Dune", approval_surface=HOST)    # does not raise
    assert out.status == TurnStatus.COMPLETED and out.failure_reason is None
    last = out.command_outputs[-1]
    assert last.command_response.success is False and out.success is False
    assert last.command_name == FIND_BOOK
    assert out.answer == f"Execution error in {FIND_BOOK}: RuntimeError('the shelf fell over')"
    assert "Traceback" not in out.answer
    assert last.command_response.artifacts["error_type"] == "RuntimeError"
    assert last.command_response.artifacts["error_message"] == "the shelf fell over"
    # the traceback is carried as data, in the shape of a raising planning command's `__result__` failure
    # (`planning_phases.run_call_result`: the artifact `traceback`, a string cut at 4000 characters)
    trace = last.command_response.artifacts["traceback"]
    assert isinstance(trace, str) and len(trace) <= 4000
    assert trace.startswith("Traceback") and "RuntimeError: the shelf fell over" in trace
    assert "Traceback" in caplog.text                                    # the traceback is in the log
    assert no_agent.ctx._waiting is None and not no_agent.ctx.assistant_round
    # the session goes on: the next message is an ordinary turn
    again = no_agent.ctx.process_turn("/find_book Dune", approval_surface=HOST)
    assert again.status == TurnStatus.COMPLETED


def test_the_same_part_on_the_plan_path_is_also_a_completed_turn_with_a_failed_output(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(query="Dune"))
    monkeypatch.setattr(_response_generator("Library.find_book"), "__call__",
                        lambda self, workflow, command, input: (_ for _ in ()).throw(RuntimeError("boom")))
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.95, "none": 0.0})
    no_agent.parts = ["find a book called Dune"]
    out = no_agent.ctx.process_turn("find a book called Dune", approval_surface=HOST)
    assert out.status == TurnStatus.COMPLETED


def test_both_which_command_questions_on_the_slash_path_are_not_successes(no_agent, monkeypatch):
    """Nothing matched and ambiguous are the same state, a question that is out: ``success`` False on both, the question
    as data (``question_kind`` ``plan_choice``), as the plan path's question is (AWAITING_USER, ``success`` False)."""
    seen = {}
    for label, distribution in {"nothing": {"none": 1.0},
                                "ambiguous": {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0}}.items():
        script_matcher(monkeypatch, lambda t, d=distribution: d)
        no_agent.parts = [JOKE]
        out = no_agent.ctx.process_turn("/" + JOKE, approval_surface=HOST)
        last = out.command_outputs[-1]
        seen[label] = (out.status, last.command_response.success, _artifacts(out).get("question_kind"))
        no_agent.ctx.process_turn("/cancel", approval_surface=HOST)
    assert seen["nothing"] == seen["ambiguous"] == (TurnStatus.COMPLETED, False, "plan_choice")
