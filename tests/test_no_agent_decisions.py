"""No-agent mode, second part (fix-rn9p): a DECISION step of a planning command pauses the turn like ask_user and resumes.

A conditional step (``when(...)``) is answered yes or no; a command step that leaves out one plain required value takes
that value. Everything else a model or the client has to carry out (a prose step, a failed step) still ends the command
as a completed turn with the structured stop (``tests/test_no_agent_mode.py``). Real components throughout, on the
library copies the phase tests use; the agent LM is never built (the autouse fixture of ``test_no_agent_mode``).
"""
from __future__ import annotations

import json

import pytest
from pydantic import Field

import fastworkflow
from fastworkflow import no_agent_mode
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.planning import phase, run, when
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import (
    NOT_RUN_ARTIFACT, PHASE_STEP_ARTIFACT, TURN_PARTS_ARTIFACT, WorkflowExecutionContext)
from tests.test_nested_planning_commands import (  # noqa: F401 - fixtures
    BOOK_BY_TITLE, BOOK_LOANS, _BOOK_BY_TITLE_SOURCE, _nested_state)
from tests.test_no_agent_mode import (  # noqa: F401 - the autouse no-agent fixture, helpers and fixtures
    CodeOnly, NoResult, OpenStep, PLAN_TEXT, _no_agent, _plan, _sent, _stops)
from tests.test_step_helpers import (  # noqa: F401 - fixtures and helpers
    FoundBooks, OpenedBook, _build_planning_command, library_env)
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401 - shared fixtures

COMMAND = "Library/book_lookup"
#: The agent LM seam as the code has it; the autouse fixture of ``test_no_agent_mode`` replaces it with one that fails.
_REAL_AGENT_CONTEXT = WorkflowExecutionContext._agent_dspy_context


class Conditional:
    """A phase whose second step is conditional and whose inner step code can run: find_book by the title."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title),
                when("the request also wants the author's other books", run("find_book", query="Persuasion"))]

    def __result__(self, input, find):
        return f"{input.title}: {len(find.book_uids)} found"


class TwoDecisions:
    """One phase with two conditional steps, then a result."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title),
                when("the title is a series", run("find_book", query="Dune")),
                when("the author also wrote others", run("find_book", query="Persuasion"))]

    def __result__(self, input, find):
        return f"{input.title}: {len(find.book_uids)} found"


class DecisionThenProse:
    """A conditional step, then a step of prose: the prose step is the agent's (free work), so the command stops there."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title),
                when("the request also wants the author's other books", run("find_book", query="Persuasion")),
                "work out which of them is the oldest"]

    def __result__(self, input, find):
        return f"{input.title}: {len(find.book_uids)} found"


class NamedThenDecision:
    """The phase opens a book (code), then asks whether to add a note: a write the step names."""

    def book(self, input) -> OpenedBook:
        return [run("find_book", query=input.title), run("open_book", named=input.title),
                when("the reader wants the note added", run("add_note", text="asked about"))]

    def __result__(self, input, book):
        return f"{book.title} handled"


class MissingValue:
    """A command step that gives no value for the one required field of add_note."""

    def book(self, input) -> OpenedBook:
        return [run("find_book", query=input.title), run("open_book", named=input.title), "add_note"]

    def __result__(self, input, book):
        return f"{book.title} noted"


class ThreePhases:
    """A decision in the first phase, then a phase of one step and a phase that fans out over the first one's books."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title),
                when("the request also wants the author's other books", run("find_book", query="Persuasion"))]

    def book(self, input, find) -> OpenedBook:
        return [run("open_book", named=input.title)]

    @phase(over="find.book_uids")
    def each(self, uid) -> OpenedBook:
        return [run("open_book", book_uid=uid)]

    def __result__(self, input, find, book, each):
        return f"{input.title}: {len(find.book_uids)} found"


def _pending(result):
    """The question (the turn's answer) and the artifacts of the pending prompt's entry (the last output, an ask_user entry)."""
    entry = result.command_outputs[-1]
    assert entry.is_ask_user and entry.command_response.success is False
    return result.answer, entry.command_response.artifacts


def _find_runs(env):
    return [command for command in _sent(env) if command.startswith("find_book")]


# ---------------------------------------------------------------- suspend

def test_a_conditional_step_suspends_with_a_question_and_the_pending_prompt_data(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.AWAITING_USER
    text, artifacts = _pending(result)
    assert artifacts["question_kind"] == "phase_step"
    data = artifacts[PHASE_STEP_ARTIFACT]
    assert data["command"] == COMMAND and data["phase"] == "find"
    assert data["step"]["number"] == 2 and data["step"]["kind"] == "conditional"
    assert data["step"]["text"].startswith("only when the request also wants the author's other books")
    assert data["decision"] == {"kind": "conditional", "condition": "the request also wants the author's other books",
                                "run_if_yes": "find_book query=Persuasion", "replies": ["yes", "no"]}
    assert data["decline"] == "cancel" and data["asked"] == 1
    assert len(data["done"]) == 1 and "find_book query=Dune" in data["done"][0]
    assert data["remaining"] == [] and list(data["facts"]) == []
    # The question is plain and self-sufficient: what ran, what is asked, how to answer, how to decline.
    assert text == library_env.ctx.waiting.question
    assert "ran 1 step" in text and "reached step 2" in text
    assert "Reply yes to run 'find_book query=Persuasion' now, or no to skip it" in text
    assert "Reply cancel to stop here without deciding." in text
    assert _find_runs(library_env) == ["find_book query=Dune"]
    assert library_env.model.asked == []
    # No stop yet: the command is not over.
    assert NOT_RUN_ARTIFACT not in artifacts


def test_yes_runs_the_inner_step_itself_and_the_command_completes_with_its_result(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn("yes")

    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Persuasion"]
    # The facts are the last run's: the one the yes made (Persuasion is one book, Dune two).
    assert result.answer == no_agent_mode.ran_note("Dune: 1 found", COMMAND)
    assert _stops(result) is None
    assert library_env.ctx._awaiting_user is False and library_env.ctx._phase_run is None
    # The yes branch's run is a step of the phase: its facts hold what the inner step read.
    outputs = [o for o in result.command_outputs if o.command_name == "Library/find_book"]
    assert len(outputs) == 2
    assert library_env.model.asked == []


def test_no_skips_the_inner_step_and_the_command_completes(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn("No.")

    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env) == ["find_book query=Dune"]
    assert "Dune: 2 found" in result.answer
    assert _stops(result) is None
    assert library_env.model.asked == []


@pytest.mark.parametrize("reply", ["y", "Yes", "TRUE", "yes."])
def test_the_accepted_yes_replies(reply, library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn(reply)
    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env)[-1] == "find_book query=Persuasion"


@pytest.mark.parametrize("reply", ["n", "false", "No!"])
def test_the_accepted_no_replies(reply, library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn(reply)
    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env) == ["find_book query=Dune"]


# ---------------------------------------------------------------- an unclear reply, a decline

def test_an_unclear_reply_asks_again_once_then_the_second_one_ends_with_the_stop(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")

    again = library_env.ctx.process_turn("go ahead and run it")
    assert again.status == TurnStatus.AWAITING_USER
    text, artifacts = _pending(again)
    assert text.startswith("I could not read that reply as yes or no. ")
    assert "Reply yes to run" in text
    assert artifacts["question_kind"] == "phase_step" and artifacts[PHASE_STEP_ARTIFACT]["asked"] == 2

    ended = library_env.ctx.process_turn("maybe")
    assert ended.status == TurnStatus.COMPLETED
    (stop,) = _stops(ended)
    assert stop["open_step"]["kind"] == "conditional" and stop["open_step"]["number"] == 2
    assert "twice" in stop["open_step"]["reason"]
    assert stop["command"] == COMMAND and len(stop["done"]) == 1
    assert _find_runs(library_env) == ["find_book query=Dune"]
    assert library_env.ctx._phase_run is None and library_env.ctx._awaiting_user is False


def test_a_decline_ends_the_command_with_the_stop_at_that_step_not_an_error(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn("cancel")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "conditional" and stop["open_step"]["number"] == 2
    assert stop["open_step"]["reason"].startswith("declined")
    assert stop["command"] == COMMAND and stop["phase"] == "find"
    assert stop["suggested_plan"].startswith("1. only when the request also wants")
    assert result.command_outputs[-1].success is not False
    assert _find_runs(library_env) == ["find_book query=Dune"]
    assert library_env.ctx._awaiting_user is False


# ---------------------------------------------------------------- two decisions in a phase; decision then prose

def test_two_decision_steps_in_one_phase_suspend_one_after_the_other(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(TwoDecisions)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    assert _pending(first)[1][PHASE_STEP_ARTIFACT]["step"]["number"] == 2

    second = library_env.ctx.process_turn("yes")
    assert second.status == TurnStatus.AWAITING_USER
    text, artifacts = _pending(second)
    data = artifacts[PHASE_STEP_ARTIFACT]
    assert data["step"]["number"] == 3 and data["decision"]["condition"] == "the author also wrote others"
    # What ran includes the step the first yes ran.
    assert len(data["done"]) == 2 and "find_book query=Dune" in data["done"][1]
    assert "ran 2 steps" in text
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Dune"]

    done = library_env.ctx.process_turn("no")
    assert done.status == TurnStatus.COMPLETED
    assert _stops(done) is None and "found" in done.answer
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Dune"]


def test_a_decision_step_followed_by_a_prose_step_stops_after_the_resume_and_names_what_ran(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(DecisionThenProse)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER

    result = library_env.ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["number"] == 3 and stop["open_step"]["kind"] == "prose"
    assert stop["open_step"]["text"] == "work out which of them is the oldest"
    # What ran after the resume is in the stop: the step the yes ran, after the one before the pause.
    assert len(stop["done"]) == 2
    assert "find_book query=Dune" in stop["done"][0] and "find_book query=Persuasion" in stop["done"][1]
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Persuasion"]
    assert library_env.model.asked == []


# ---------------------------------------------------------------- a named step is a decision when the yes can run

def test_a_conditional_step_that_opens_a_row_by_name_runs_after_a_yes(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(OpenStep)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    data = _pending(first)[1][PHASE_STEP_ARTIFACT]
    assert data["phase"] == "book" and data["decision"]["run_if_yes"] == "open_book with the uid of the Dune match"
    assert list(data["facts"]) == ["find"] and len(data["facts"]["find"]["book_uids"]) == 2

    result = library_env.ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    assert "open_book book_uid=b1" in _sent(library_env)
    assert "Dune opened" in result.answer and _stops(result) is None


# ---------------------------------------------------------------- the write-refusal guard and approvals on the inner step

def test_the_inner_step_that_writes_runs_only_after_a_yes_and_through_the_same_guard(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(NamedThenDecision)
    _plan(monkeypatch, library_env)
    book = library_env.wf.root.books["b1"]
    before = list(book.notes)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    assert list(book.notes) == before, "nothing is written before the client says yes"

    result = library_env.ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    ran = [o for o in result.command_outputs if (o.command_name or "").endswith("add_note")]
    assert len(ran) == 1
    # The step is the phase's own, so the write guard (a step names the command) lets it run, as for any code step.
    assert ran[0].success is True, ran[0].command_response.response
    assert list(book.notes) == [*before, "asked about"]


def test_a_no_never_runs_the_writing_inner_step(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(NamedThenDecision)
    _plan(monkeypatch, library_env)
    book = library_env.wf.root.books["b1"]
    before = list(book.notes)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn("no")
    assert result.status == TurnStatus.COMPLETED
    assert list(book.notes) == before
    assert not any((o.command_name or "").endswith("add_note") for o in result.command_outputs)


# ---------------------------------------------------------------- a missing plain value

def test_a_missing_required_value_asks_for_it_and_runs_the_step_with_the_reply(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(MissingValue)
    _plan(monkeypatch, library_env)
    book = library_env.wf.root.books["b1"]
    before = list(book.notes)
    first = library_env.ctx.process_turn("look up the book Dune")

    assert first.status == TurnStatus.AWAITING_USER
    text, artifacts = _pending(first)
    assert artifacts["question_kind"] == "phase_step"
    data = artifacts[PHASE_STEP_ARTIFACT]
    assert data["step"] == {"number": 3, "text": "add_note", "kind": "missing_value"}
    assert data["decision"]["kind"] == "missing_value" and data["decision"]["field"] == "text"
    assert data["decision"]["type"] == "str" and data["decision"]["description"] == "the note to add"
    assert data["decision"]["reply"] == "the value of the field"
    assert "needs a value for text (str: the note to add)" in text
    assert "Reply with that value alone" in text and "Reply cancel to stop here" in text
    assert len(data["done"]) == 2

    result = library_env.ctx.process_turn("the cover is torn")
    assert result.status == TurnStatus.COMPLETED
    assert "add_note text=the cover is torn" in _sent(library_env)
    assert list(book.notes) == [*before, "the cover is torn"]
    assert "Dune noted" in result.answer and _stops(result) is None


def test_a_missing_value_that_is_declined_ends_with_the_stop_at_that_step(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(MissingValue)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    result = library_env.ctx.process_turn("cancel")
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "missing_value" and stop["open_step"]["reason"].startswith("declined")
    assert not any(command.startswith("add_note") for command in _sent(library_env))


def test_an_empty_missing_value_reply_asks_again_then_stops(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(MissingValue)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    again = library_env.ctx.process_turn("   ")
    assert again.status == TurnStatus.AWAITING_USER
    assert _pending(again)[0].startswith("I need a value for text (str). ")
    ended = library_env.ctx.process_turn("  ")
    (stop,) = _stops(ended)
    assert stop["open_step"]["kind"] == "missing_value" and "twice" in stop["open_step"]["reason"]


def test_a_missing_value_reply_written_as_field_value_gives_the_value_not_the_label(library_env, monkeypatch):
    """The step's reply goes through the reader every parameter question uses (``parameter_clarification.read_reply``,
    fix-yu3l.4): the question form's ``<field>: <value>`` line is the value, as the value alone is."""
    library_env.planning_command = _build_planning_command(MissingValue)
    _plan(monkeypatch, library_env)
    book = library_env.wf.root.books["b1"]
    before = list(book.notes)
    library_env.ctx.process_turn("look up the book Dune")

    result = library_env.ctx.process_turn("text: the cover is torn")
    assert result.status == TurnStatus.COMPLETED
    assert "add_note text=the cover is torn" in _sent(library_env)
    assert list(book.notes) == [*before, "the cover is torn"]

# ---------------------------------------------------------------- everything else stays free work

def test_a_prose_step_is_free_work_and_still_ends_with_the_stop(library_env, monkeypatch):
    class Prose:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title), "work out which of them is the oldest"]

        def __result__(self, input, find):
            return "x"

    library_env.planning_command = _build_planning_command(Prose)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")
    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "prose"


def test_a_conditional_step_whose_inner_step_code_cannot_run_is_free_work(library_env, monkeypatch):
    class Unrunnable:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title), when("a book matched", "tell the reader about it")]

        def __result__(self, input, find):
            return "x"

    library_env.planning_command = _build_planning_command(Unrunnable)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")
    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "conditional"


def test_an_unattended_turn_cannot_be_asked_so_a_decision_step_ends_with_the_stop(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune", unattended=True)
    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["open_step"]["kind"] == "conditional" and stop["open_step"]["number"] == 2
    assert stop["open_step"]["reason"].startswith("a conditional step")
    assert library_env.ctx._awaiting_user is False


# ---------------------------------------------------------------- a "/" command while the question is pending

def test_a_slash_message_sent_while_the_question_is_pending_is_the_reply(library_env, monkeypatch):
    """Changed (bd fix-fpyb.7, owner decision 2026-11-19): a "/"-prefixed message while any question is out is read as the
    reply to it, with the "/" removed. It never reaches the deterministic path; here it cannot be read as yes or no, so
    the question is asked once more, and a later plain reply still answers it."""
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    handled: list[str] = []

    def deterministic(cls, session, command):
        handled.append(command)
        session.end_round()   # as the real invoke_command ends
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="done"))

    runner = CommandExecutor.__dict__["invoke_command"]
    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(deterministic))
    again = library_env.ctx.process_turn("/find_book query=Messiah")
    monkeypatch.setattr(CommandExecutor, "invoke_command", runner)

    assert library_env.ctx._awaiting_user is True
    # The "/" message was the reply (it could not be read as yes or no: asked once more), not a command of its own.
    assert handled == []
    assert again.answer.startswith("I could not read that reply as yes or no. ")
    assert library_env.ctx._waiting.payload["decision"]["asked"] == 2
    assert _find_runs(library_env) == ["find_book query=Dune"]
    result = library_env.ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env)[-1] == "find_book query=Persuasion"


def test_cancel_pending_clears_the_paused_command_as_it_clears_any_suspension(library_env, monkeypatch):
    """POST /cancel_pending is the existing abandon path: it drops the suspension with no stop (the reply ``cancel``
    is what ends the command with its structured stop)."""
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    assert library_env.ctx.cancel_pending() is True
    assert library_env.ctx._awaiting_user is False and library_env.ctx._phase_run is None


# ---------------------------------------------------------------- a multi-part turn

def test_a_decision_inside_part_one_resumes_into_the_remaining_parts(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env, matches={"book": library_env.planning_command}, ad_hoc_plan=PLAN_TEXT)
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *args, **kwargs: ["look up the book Dune", "plan a trip to Mars"])
    first = library_env.ctx.process_turn("look up the book Dune and plan a trip to Mars")
    assert first.status == TurnStatus.AWAITING_USER
    assert _pending(first)[1]["question_kind"] == "phase_step"

    result = library_env.ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["sub_task"] == 2 and stop["means"] == "from_scratch"
    assert result.answer.startswith("### 1. look up the book Dune\n")
    assert "Dune: 1 found" in result.answer and "### 2. plan a trip to Mars" in result.answer
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Persuasion"]
    assert TURN_PARTS_ARTIFACT in result.command_outputs[-1].command_response.artifacts


# ---------------------------------------------------------------- state survives a restart

def test_a_paused_decision_survives_a_save_and_restore_of_the_context(library_env, monkeypatch):
    """The pause is the phase's state plus the awaiting flag and a minimal agent stash, all serialised as an ask_user
    suspension is; the restored phases are rebuilt from the planning command (the same seam the ask_user restore uses)."""
    from fastworkflow import planning_commands

    command = _build_planning_command(Conditional)
    library_env.planning_command = command
    _plan(monkeypatch, library_env)
    monkeypatch.setattr(planning_commands, "planning_command", lambda folder, name: command)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    ctx = library_env.ctx
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="decision-restore"), default=str))

    # As a new process: nothing live is left but what the blob holds.
    ctx._phase_run = ctx._phase_live = None
    ctx._waiting = None
    ctx._workflow_tool_agent.clear_suspension()
    ctx.apply_serialized_state(blob)

    assert ctx._awaiting_user is True and ctx._waiting.payload["decision"]["kind"] == "conditional"
    assert ctx.waiting.question == first.answer
    result = ctx.process_turn("yes")
    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Persuasion"]


# ---------------------------------------------------------------- nested

_NESTED_YES = _BOOK_BY_TITLE_SOURCE.replace(
    "from fastworkflow.planning import run\n", "from fastworkflow.planning import run, when\n").replace(
    'return [run("open_book", named=input.title)]',
    'return [when("a book matched", run("open_book", named=input.title))]')


def test_a_decision_in_a_nested_command_suspends_the_turn_and_resumes_into_the_nested_command_then_the_parent(
        tmp_path, initialized_fastworkflow, monkeypatch):
    from tests.test_nested_planning_commands import BOOK_BY_TITLE_FILE
    from tests.test_step_helpers import _real_planning_command

    state = _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                          command_texts={BOOK_BY_TITLE_FILE: _NESTED_YES})
    try:
        state.planning_command = _real_planning_command(BOOK_LOANS, "_commands/Library/book_loans.py").command
        state.args = {"title": "Dune"}
        first = state.ctx.process_turn("who has Dune on loan")
        assert first.status == TurnStatus.AWAITING_USER
        data = first.command_outputs[-1].command_response.artifacts[PHASE_STEP_ARTIFACT]
        assert data["command"] == BOOK_BY_TITLE and data["decision"]["kind"] == "conditional"
        assert len(state.ctx._frame_stack()) == 2

        result = state.ctx.process_turn("yes")
    finally:
        state.ctx.pop_active_workflow()

    assert result.status == TurnStatus.COMPLETED
    assert len(state.ctx._frame_stack()) == 1
    assert _stops(result) is None
    assert state.model.asked == [] and state.nested.asked == []


# ---------------------------------------------------------------- with the agent model set nothing changes

def test_with_the_agent_model_set_the_conditional_step_is_the_agents_and_no_question_is_put(library_env, monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/model")
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", _REAL_AGENT_CONTEXT)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    library_env.model.run("find_book query=Persuasion")
    library_env.model.finish()
    result = library_env.ctx.process_turn("look up the book Dune")

    assert library_env.ctx.no_agent_mode is False
    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked, "the agent was asked for the conditional step"
    assert _find_runs(library_env) == ["find_book query=Dune", "find_book query=Persuasion"]
    artifacts = [o.command_response.artifacts for o in result.command_outputs]
    assert not any(NOT_RUN_ARTIFACT in a or PHASE_STEP_ARTIFACT in a or "question_kind" in a for a in artifacts)


# ---------------------------------------------------------------- the readers, on their own

@pytest.mark.parametrize("reply, expected", [
    ("yes", True), ("Y", True), (" true ", True), ("yes!", True), ("no", False), ("N", False), ("False.", False),
    ("yes please", None), ("go", None), ("no changes", None), ("ok", None), ("", None), ("maybe", None)])
def test_the_yes_no_reader_is_strict(reply, expected):
    assert no_agent_mode.read_yes_no(reply) is expected


@pytest.mark.parametrize("reply, expected", [
    ("cancel", True), ("Abort.", True), ("stop", True), ("never mind", True), ("no", False), ("cancel the second", False)])
def test_the_decline_reader_is_whole_reply_only(reply, expected):
    assert no_agent_mode.declines(reply) is expected


def test_a_value_is_written_into_a_step_before_its_note():
    from fastworkflow import planning_phases

    assert planning_phases.step_with_value("add_note", "text", "the cover is torn") == "add_note text=the cover is torn"
    assert planning_phases.step_with_value("set_due loan_uid=l1 (for the loan)", "days", "5") == \
        "set_due loan_uid=l1 days=5 (for the loan)"
    assert planning_phases.step_with_value("add_note", "text", "a (b); c") == 'add_note text="a (b); c"'


def test_only_a_single_plain_required_value_is_a_decision():
    from typing import Optional

    from pydantic import BaseModel
    from fastworkflow import planning_phases
    from fastworkflow.observation_offloading import finish_check

    class One(BaseModel):
        text: str = Field(description="the note")

    class Two(BaseModel):
        loan_uid: str = Field(description="uid")
        days: int = Field(description="days")

    class Listy(BaseModel):
        names: list[str] = Field(description="names")

    class Found(BaseModel):
        book_uid: str = Field(description="uid", json_schema_extra={"from_command_output": "find_book.book_uids"})

    class Typed(BaseModel):
        days: Optional[int] = Field(default=None)
        limit: int = Field(description="how many")

    classes = {"add_note": One, "set_due": Two, "list_names": Listy, "open_found": Found, "typed": Typed}
    commands = list(classes)

    def field(text):
        step = finish_check.parse_plan(f"1. {text}", commands, [])[0]
        return planning_phases.missing_value_field(step, lambda name: [classes[name]])

    assert field("add_note") == {"field": "text", "type": "str", "description": "the note"}
    assert field("typed") == {"field": "limit", "type": "int", "description": "how many"}
    assert field("set_due loan_uid=l1") == {"field": "days", "type": "int", "description": "days"}
    assert field("set_due") is None        # two values are missing
    assert field("list_names") is None     # not a plain value
    assert field("open_found") is None     # a value that must be found from a listing


# ---------------------------------------------------------------- an unverified host-surface client

def test_an_unverified_host_client_answers_the_question_no_identity_is_needed(library_env, monkeypatch):
    """The question is an ordinary suspended question, not an approval record: ``approval_binding`` reads a reply only
    when an approval record is active, so a client with no verified identity answers it like an ask_user question."""
    from fastworkflow.approval_binding import Surface

    host = Surface(kind="host", verified=False, uid=None)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune", approval_surface=host)
    assert first.status == TurnStatus.AWAITING_USER
    assert _pending(first)[1]["question_kind"] == "phase_step"

    result = library_env.ctx.process_turn("yes", approval_surface=host)
    assert result.status == TurnStatus.COMPLETED
    assert _find_runs(library_env)[-1] == "find_book query=Persuasion"



# ---------------------------------------------------------------- later phases and the continuation of a stop

LATER_PLAN = ("1. only when the request also wants the author's other books, find_book query=Persuasion\n"
              "2. (phase book) open_book with the uid of the Dune match\n"
              "3. (phase each, for each uid) open_book book_uid=<uid>\n"
              "4. " + no_agent_mode.ANSWER_LINE)


def test_the_pending_data_and_the_question_name_the_later_phases(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(ThreePhases)
    _plan(monkeypatch, library_env)
    text, artifacts = _pending(library_env.ctx.process_turn("look up the book Dune"))
    data = artifacts[PHASE_STEP_ARTIFACT]
    # The remaining steps of THIS phase are none; the work left is in the later phases, which the data names.
    assert data["remaining"] == [] and data["later_phases"] == ["book", "each"]
    assert "carries on with its remaining steps and its later phases (book, each)." in text


def test_a_command_with_no_later_phase_says_remaining_steps_only(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    text, artifacts = _pending(library_env.ctx.process_turn("look up the book Dune"))
    assert artifacts[PHASE_STEP_ARTIFACT]["later_phases"] == []
    assert "carries on with its remaining steps." in text and "later phases" not in text


def test_a_declined_stop_carries_the_later_phases_steps_in_its_suggested_plan(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(ThreePhases)
    _plan(monkeypatch, library_env)
    library_env.ctx.process_turn("look up the book Dune")
    (stop,) = _stops(library_env.ctx.process_turn("cancel"))
    assert stop["later_phases"] == ["book", "each"]
    assert stop["suggested_plan"] == LATER_PLAN


def test_a_first_part_stop_carries_the_same_continuation(library_env, monkeypatch):
    """An unattended turn cannot be asked: the stop comes at once, with the same continuation."""
    library_env.planning_command = _build_planning_command(ThreePhases)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune", unattended=True)
    (stop,) = _stops(result)
    assert stop["later_phases"] == ["book", "each"] and stop["suggested_plan"] == LATER_PLAN
