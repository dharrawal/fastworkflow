"""fix-cwah: a run that ran out of steps while paging a read says so. Agent mode on the annotated library copy, the
real agent with a scripted model (the approval tests' ``todo`` context), the library's pages shrunk to one row."""
from __future__ import annotations

import sys

import pytest

from fastworkflow import no_agent_mode
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

READ = "find_book query=a start={}"


@pytest.fixture
def paged(todo, monkeypatch):
    # The commands of the annotated copy import their own copy of the library module: shrink the pages there.
    for module in [m for name, m in list(sys.modules.items()) if name.endswith("application.library")]:
        monkeypatch.setattr(module, "PAGE_SIZE", 1)
    monkeypatch.setattr("fastworkflow.workflow_agent.run_progress_check", lambda *a, **k: "DONE SUMMARY")
    return todo


def _pages(todo, count):
    return [("execute_workflow_query", {"command": READ.format(start)}) for start in range(count)]


def test_a_read_that_ran_out_of_steps_states_the_rows_read_of_the_total(paged):
    paged.ctx._workflow_tool_agent.max_iters = 1  # every run reads one page and is out of steps
    paged.script.steps = _pages(paged, 3)
    result = paged.ctx.process_turn("find every book with an a in it")

    total = len(paged.library.find_books("a"))
    assert total > 3
    answer = result.answer
    assert f"3 rows read of {total} the command reported" in answer
    assert no_agent_mode.partial_read_sentence("step limit") in answer
    assert answer.count("rows read of") == 1


def test_a_read_that_finished_within_the_steps_adds_nothing(paged):
    paged.script.steps = [*_pages(paged, 1), ("finish", {})]
    result = paged.ctx.process_turn("find a book with an a")

    assert "rows read of" not in result.answer
    assert "partial read" not in result.answer


def test_the_page_cap_and_the_step_limit_share_one_sentence():
    entry = no_agent_mode.stop_entry(
        command="X/y", phase="p", open_step={"number": 1, "text": "t", "kind": "page_cap", "reason": "capped"},
        done=["p: 1. list things"], facts={}, remaining=[], suggested_plan="", means="planning")
    assert no_agent_mode.partial_read_sentence("page cap") in no_agent_mode.ran_statement(entry)
    assert "step limit" in no_agent_mode.partial_read_sentence("step limit")
