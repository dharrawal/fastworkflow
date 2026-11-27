"""fix-cwah: a run that ran out of steps while paging a read says so. Agent mode on the annotated library copy, the
real agent with a scripted model (the approval tests' ``todo`` context), the library's pages shrunk to one row."""
from __future__ import annotations

import sys

import pytest

from fastworkflow.core import no_agent_mode
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_step_helpers import library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401

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



# ---------------------------------------------------------------- a phase of a planning command out of steps

def test_a_planning_command_phase_out_of_steps_states_the_rows_read_of_the_total(library_env, monkeypatch):
    """A phase whose agent ran out of steps while it paged a read ends the answer with the same partial-read note."""
    from fastworkflow.planning import planning_phases
    from fastworkflow.planning import run
    from tests.test_no_agent_mode import _plan
    from tests.test_step_helpers import _build_planning_command

    lib = library_env.wf.root
    monkeypatch.setattr(sys.modules[type(lib).__module__], "PAGE_SIZE", 1)
    monkeypatch.setattr(planning_phases, "PHASE_MAX_STEPS", 2)
    monkeypatch.setattr(planning_phases, "CODE_MAX_PAGES", 1)
    library_env.runner.name_find_book_next_pages = True

    class ReadByHand:
        def find(self, input):
            return [run("find_book", query="a", all_pages=True)]

        def report(self, input):
            return ["say what was found"]

    ctx = library_env.ctx
    build_agent = ctx._build_react_agent

    def small_agent():
        agent = build_agent()
        agent.max_iters = 2     # the phase's executor runs out of steps (its own cap), not the phase's budget
        return agent

    monkeypatch.setattr(ctx, "_build_react_agent", small_agent)
    model = library_env.model
    model.run("find_book query=a start=1")
    model.run("find_book query=a start=2")
    # Two parts, so the turn runs through its sub-task loop; the second part is a different command-free request.
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *args, **kwargs: ["look up the book Dune", "look up the book Emma"])
    for _ in range(2):
        model.run("find_book query=a start=1")
        model.run("find_book query=a start=2")
    library_env.planning_command = _build_planning_command(ReadByHand)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    total = len(lib.find_books("a"))
    assert total > 2
    assert f"3 rows read of {total} the command reported" in result.answer
    assert no_agent_mode.partial_read_sentence("step limit") in result.answer


# ---------------------------------------------------------------- an unsplit single part with no plan

def test_an_unsplit_single_part_out_of_steps_states_the_rows_read_of_the_total(paged, monkeypatch):
    """The turn's one part runs through ``_run_turn_subtasks``' own branch; its answer carries the same note once."""
    from fastworkflow.engine.workflow_execution_context import WorkflowExecutionContext

    seen = []
    original = WorkflowExecutionContext._run_turn_subtasks

    def watch(self, subtasks, *args, **kwargs):
        seen.append((list(subtasks), self._turn_subtask_plan))
        return original(self, subtasks, *args, **kwargs)

    monkeypatch.setattr(WorkflowExecutionContext, "_run_turn_subtasks", watch)
    paged.ctx._workflow_tool_agent.max_iters = 1
    paged.script.steps = _pages(paged, 3)
    result = paged.ctx.process_turn("find every book with an a in it")

    assert len(seen) == 1 and len(seen[0][0]) == 1 and seen[0][1] is None
    total = len(paged.library.find_books("a"))
    assert f"3 rows read of {total} the command reported" in result.answer
    assert result.answer.count("rows read of") == 1
