"""No-agent mode (fix-9m0p): a planning command whose steps all ran by code, with no ``__result__`` text, still
answers with a code-written statement of what ran and the facts it recorded, then the notice that the wording needs a
model. Real components, on the library copies the other no-agent suites use."""
from __future__ import annotations

import sys

from fastworkflow import no_agent_mode
from fastworkflow.planning import run
from fastworkflow.turn import TurnStatus
from tests.test_no_agent_mode import (  # noqa: F401 - the autouse no-agent fixture, helpers and fixtures
    NoResult, _no_agent, _plan, _stops)
from tests.test_step_helpers import (  # noqa: F401
    FoundBooks, OpenedBook, _build_planning_command, library_env)
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401


def test_an_all_code_command_with_no_result_text_states_what_ran(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(NoResult)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    answer = result.answer
    # What ran, from the phase's own step lines.
    assert "What ran" in answer and stop["done"][0] in answer
    # The facts the phase recorded, in plain markdown.
    assert "find" in answer and "book_uids" in answer
    for uid in stop["facts"]["find"]["book_uids"]:
        assert str(uid) in answer
    # The notice stays, after the statement.
    assert "its answer needs a model to write it" in answer
    assert answer.index("What ran") < answer.index("needs a model")
    assert library_env.model.asked == []


def test_a_stop_at_an_open_step_states_what_ran_before_its_notice():
    entry = no_agent_mode.stop_entry(
        command="X/y", phase="p", open_step={"number": 2, "text": "t", "kind": "prose", "reason": "why"},
        done=["p: 1. a"], facts={"p": {"n": 1}}, remaining=[], suggested_plan="", means="planning")
    text = no_agent_mode.explanation(entry)
    assert "What ran" in text and "p: 1. a" in text and "n: 1" in text
    assert text.index("What ran") < text.index("then it stopped at step 2 (why)")


def test_a_stop_with_nothing_done_keeps_its_short_text():
    entry = no_agent_mode.stop_entry(
        command="X/y", phase="p", open_step={"number": 1, "text": "t", "kind": "prose", "reason": "why"},
        done=[], facts={}, remaining=[], suggested_plan="", means="planning")
    text = no_agent_mode.explanation(entry)
    assert "What ran" not in text and text.startswith("X/y ran as far as code can run it")


def test_a_page_cap_stop_says_the_read_is_partial():
    entry = no_agent_mode.stop_entry(
        command="X/y", phase="p", open_step={"number": 1, "text": "t", "kind": "page_cap", "reason": "capped"},
        done=["p: 1. list things"], facts={"p": {"total": 40}}, remaining=[], suggested_plan="", means="planning")
    text = no_agent_mode.explanation(entry)
    assert "What ran" in text and "total: 40" in text
    assert "page cap" in text and "partial read" in text


# ---------------------------------------------------------------- the commands' own result text, and a capped read

class ReadWriteRead:
    """A read, a read that opens a record, then a write: the write's own response is the only sign it was done."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title)]

    def book(self, input, find) -> OpenedBook:
        return [run("open_book", named=input.title)]

    def notes(self, book):
        return [run("add_note", text="the cover is torn")]


def test_the_answer_carries_each_steps_own_result_text(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(ReadWriteRead)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert len(stop["done"]) == 3
    # Additive key (the book already held one note): one result text per step line that ran, as the command's response was recorded.
    assert set(stop["results"]) == set(stop["done"])
    write_line = stop["done"][-1]
    assert "add_note" in write_line and '"total":2' in stop["results"][write_line].replace(" ", "")
    # The answer prints it under its step line, so a reader can tell the change was made.
    lines = result.answer.splitlines()
    at = next(i for i, line in enumerate(lines) if write_line in line)
    assert '"total"' in lines[at + 1] and lines[at + 1].lstrip().startswith(">")
    assert library_env.model.asked == []


def _stop_with_results(results: dict[str, str]) -> dict:
    return no_agent_mode.stop_entry(
        command="X/y", phase="p", open_step={"number": None, "text": "", "kind": "answer", "reason": "r"},
        done=list(results), facts={}, remaining=[], suggested_plan="", means="planning", results=results)


def test_a_steps_result_text_of_a_few_thousand_characters_is_shown_in_full():
    # Decision A (2026-11-23): the statement is the only answer the user gets, so what a step read is shown. The
    # longest step result a stopped plan's statement cut in the attended benchmark was about 3,250 characters.
    long_text = "\n".join(f"row {i} of the report, with the account and the rights it holds" for i in range(50))
    assert 3000 < len(long_text) < no_agent_mode.RESULT_TEXT_CAP
    text = no_agent_mode.ran_statement(_stop_with_results({"p: 1. report": long_text}))
    assert "row 0 of the report" in text and "row 49 of the report" in text
    assert "not shown" not in text


def test_a_result_text_over_the_step_limit_is_cut_at_a_line_and_says_how_much_and_where_the_rest_is():
    long_text = "\n".join(f"row {i:05d} of a very long report" for i in range(2000))
    assert len(long_text) > no_agent_mode.RESULT_TEXT_CAP
    text = no_agent_mode.ran_statement(_stop_with_results({"p: 1. report": long_text}))
    shown = [line for line in text.splitlines() if line.lstrip().startswith("> row")]
    # Cut at a line boundary: every row shown is whole.
    assert shown and all(line.rstrip().endswith("of a very long report") or "not shown" in line for line in shown)
    assert "row 00000" in text and "row 01999" not in text
    hidden = len(long_text) - sum(len(line.split("> ", 1)[1].split(" [...")[0]) + 1 for line in shown) + 1
    assert f"[... {hidden} more characters not shown" in text
    assert "fw_not_run" in text and "results" in text
    assert len(text) < no_agent_mode.RESULT_TEXT_CAP + 1000


def test_the_statement_bounds_the_result_text_of_all_steps_together():
    step = "\n".join(f"line {i:04d} of a step result" for i in range(200))
    assert no_agent_mode.RESULT_TEXT_CAP < len(step)
    results = {f"p: {n}. read {n}": step for n in range(1, 11)}
    text = no_agent_mode.ran_statement(_stop_with_results(results))
    # Every step line is still listed, each result says how much of it is not shown, and the whole stays bounded.
    for line in results:
        assert line in text
    assert text.count("not shown") == len(results)
    assert len(text) < no_agent_mode.RESULTS_TOTAL_CAP + 1000
    # A step past the total gets no rows, only the note of how much it read.
    assert f"[... {len(step)} more characters not shown" in text


def test_a_page_cap_stop_states_the_rows_read_and_the_total(library_env, monkeypatch):
    from fastworkflow import planning_phases

    lib = library_env.wf.root
    book_cls = type(lib.books["b1"])
    monkeypatch.setattr(sys.modules[type(lib).__module__], "PAGE_SIZE", 1)
    monkeypatch.setattr(planning_phases, "CODE_MAX_PAGES", 2)
    lib.books["b5"] = book_cls(lib, "b5", "Heretics of Dune", "Frank Herbert", 1, "Central")
    library_env.runner.name_find_book_next_pages = True

    class AllPages:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query="Frank Herbert", all_pages=True)]

    library_env.planning_command = _build_planning_command(AllPages)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "page_cap"
    assert stop["facts"] == {}  # nothing is filled until a phase ends: the rows are in the data, not the facts
    read = stop["partial_read"]
    assert read["rows_read"] == 2 and read["total"] == 3 and read["pages"] == 2
    assert read["rows"] == ["b1", "b2"]
    answer = result.answer
    assert "2 rows read of 3 the command reported" in answer
    assert "b1" in answer and "b2" in answer and "b5" not in answer
    assert "partial read" in answer
    assert library_env.model.asked == []
