"""fix-fixv: a command that carries ``@writes`` runs inside a planning command's phase only when a
step of that phase names it.

Each case runs the library's real planning commands on a copy of the library, through the real phase
runner and the executor's own dispatch (``real_dispatch``: only the extraction hop is scripted, as the
other library tests script it); the executor's model is scripted (``Model``), the accepted LLM boundary.
"""
from __future__ import annotations

import json
import os

import pytest
from pydantic import Field

import fastworkflow
from fastworkflow.planning import command_preview
from fastworkflow.core import command_writes
from fastworkflow import workflow_agent
from fastworkflow.observability import tracing
from fastworkflow.core.command_directory import CommandLoadError
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.planning import phase, run, when
from fastworkflow.core.turn import TurnStatus
from fastworkflow.workflow_agent import _what_can_i_do
from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_step_helpers import (
    LIBRARY,
    _library_workflow,
    FoundBooks,
    OpenedBook,
    _build_planning_command,
    _make_library_runner_env,
    _real_planning_command,
)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

AUTHOR_BOOKS = "Library/author_books"
NOTE_MEMBER = "Library/note_member"
WRITES_FILES = (
    "_commands/Library/author_books.py",
    "_commands/Library/note_member.py",
    "_commands/_shared/__init__.py", "_commands/_shared/wording.py",
)


@pytest.fixture
def writes_env(tmp_path, initialized_fastworkflow, monkeypatch):
    state = _make_library_runner_env(
        tmp_path, initialized_fastworkflow, monkeypatch,
        extra_commands=WRITES_FILES, real_dispatch=True)
    yield state
    state.ctx.pop_active_workflow()


def _library(state):
    return state.wf.root


def _notes(state) -> dict[str, list[str]]:
    library = _library(state)
    return {**{uid: list(book.notes) for uid, book in library.books.items()},
            **{uid: list(member.notes) for uid, member in library.members.items()}}


def _outputs(state, name: str):
    return [o for o in state.ctx._turn_outputs if (o.command_name or "").split("/")[-1] == name]


def test_the_library_declares_add_note_as_its_one_write_and_note_member_writes_through_its_steps(
        initialized_fastworkflow):
    # add_note carries @writes; note_member's steps run add_note, so it writes too; author_books
    # (find_book, open_book) reaches no write: it is read-only.
    RoutingRegistry.clear_registry()
    writes = command_writes.write_commands(LIBRARY)
    assert writes == {"Record/add_note", NOTE_MEMBER}
    assert AUTHOR_BOOKS not in writes


def test_a_read_only_planning_command_refuses_a_note_its_phase_does_not_name(writes_env):
    # The recorded failure (fix-48np.1, L2): inside author_books, the phase agent added "the cover is
    # torn" to a book. Here the agent finishes the find phase, then, on a book its book phase opened,
    # adds a note no step names. Without the declaration the note lands on the book (Persuasion, b4).
    # fix-s70h: a result-driven last phase whose every step code ran takes no model step, so the real
    # author_books never asks the agent; this read-only copy of it adds an open step ("when ...") that
    # code cannot run, which is the agent's, so the agent is asked and attempts the write there.
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.author)]

        @phase(over="find.book_uids", max_items=5, noun="books")
        def book(self, uid) -> OpenedBook:
            return [run("open_book", book_uid=uid),
                    when("the member asked to annotate the book", run("open_book", book_uid=uid))]

        def __result__(self, input, book):
            return "the books were opened"

    before = _notes(writes_env)
    writes_env.model.finish()
    writes_env.model.run('add_note text="the spine is cracked"')
    writes_env.model.finish()
    writes_env.model.finish()
    writes_env.planning_command = _build_planning_command(
        Plan, {"author": (str, Field(description="the author's name"))})
    writes_env.args = {"author": "Austen"}

    result = writes_env.ctx.process_turn("show every Austen book in the library")

    assert result.status == TurnStatus.COMPLETED
    assert _notes(writes_env) == before            # no record changed
    [refused] = _outputs(writes_env, "add_note")
    assert refused.success is False
    assert "changes the application's data" in refused.command_response.response
    # A refused write never ran: it is marked, and the action log (what distillation and the policy
    # read as executed) holds the phase's real steps but not the refused write.
    assert refused.command_response.artifacts.get("write_refused") is True
    logged = json.dumps(writes_env.ctx._action_log)
    assert "open_book" in logged, "the action log records the steps that ran"
    assert "add_note" not in logged and "spine is cracked" not in logged
    assert writes_env.model.asked, "the agent was asked for a step, and tried the write"
    # the books were still opened, by the phase's own steps
    assert [o.command_response.output.book_uid for o in _outputs(writes_env, "open_book")] == ["b3", "b4"]


def test_a_planning_command_whose_step_names_the_write_runs_it(writes_env):
    # note_member's phase names add_note: the guard lets it run, once, on the member's record.
    writes_env.planning_command = _real_planning_command(NOTE_MEMBER, "_commands/Library/note_member.py").command
    writes_env.args = {"member": "Raj Patel", "text": "he paid his fine"}

    result = writes_env.ctx.process_turn("note on Raj Patel's record that he paid his fine")

    assert result.status == TurnStatus.COMPLETED
    assert _library(writes_env).members["m2"].notes == ["prefers email reminders", "he paid his fine"]
    [added] = _outputs(writes_env, "add_note")
    assert added.success is True


def test_the_listing_text_of_a_read_only_phase_has_no_add_note(writes_env):
    # The listing itself, read with a phase's steps in place: add_note is gone from the Book listing.
    library = _library(writes_env)
    writes_env.wf.record = library.books["b3"]
    assert "add_note" in _what_can_i_do(writes_env.ctx)
    writes_env.ctx._phase_run = {"plan": "1. open_book book_uid=b3"}
    try:
        assert "add_note" not in _what_can_i_do(writes_env.ctx)
        assert "show_notes" in _what_can_i_do(writes_env.ctx)
        writes_env.ctx._phase_run = {"plan": "1. add_note text=hi"}
        assert "add_note" in _what_can_i_do(writes_env.ctx)
    finally:
        writes_env.ctx._phase_run = None


def test_the_retired_writes_attribute_fails_the_load_naming_the_decorator(tmp_path):
    RoutingRegistry.clear_registry()
    path = os.path.join(LIBRARY, "_commands", "Record", "add_note.py")
    text = open(path, encoding="utf-8").read().replace(
        "@fastworkflow.writes\nclass ResponseGenerator:\n", "class ResponseGenerator:\n    writes = True\n", 1)
    assert "writes = True" in text
    folder = _library_workflow(tmp_path, command_texts={"_commands/Record/add_note.py": text})
    # that command does not load and is listed; the workflow loads (bd fix-ftdl.2)
    problems = [p for p in command_preview.load_problems(folder) if p.startswith("Record/add_note: ")]
    assert len(problems) == 1 and "writes = True is removed: write @writes" in problems[0], problems
    with pytest.raises(CommandLoadError, match=r"writes = True is removed: write @writes"):
        RoutingRegistry.get_definition(folder).get_command_class(
            "Record/add_note", fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
    RoutingRegistry.clear_registry()


def test_a_planning_command_may_not_declare_writes(tmp_path):
    RoutingRegistry.clear_registry()
    path = os.path.join(LIBRARY, "_commands", "Library", "note_member.py")
    text = open(path, encoding="utf-8").read().replace(
        "class PlanGenerator:\n", "@fastworkflow.writes\nclass PlanGenerator:\n", 1)
    text = "import fastworkflow\n" + text
    assert "@fastworkflow.writes" in text
    folder = _library_workflow(tmp_path, command_texts={"_commands/Library/note_member.py": text},
                               extra_commands=WRITES_FILES)
    # that planning command does not load and is listed; the workflow loads (bd fix-ftdl.2)
    problems = [p for p in command_preview.load_problems(folder) if p.startswith("Library/note_member: ")]
    assert len(problems) == 1 and "a planning command cannot carry @writes" in problems[0], problems
    assert "Library/note_member" not in workflow_agent.load_planning_command_map(folder)
    RoutingRegistry.clear_registry()


def test_a_write_the_phase_names_once_runs_once_whatever_the_agent_repeats(writes_env):
    # fix-ddis: note_member's one step names add_note once, and code runs it. The phase agent, asked for the
    # open step that follows, calls add_note again: the phase's own count is spent, so it is refused unrun.
    writes_env.planning_command = _real_planning_command(NOTE_MEMBER, "_commands/Library/note_member.py").command
    writes_env.args = {"member": "Raj Patel", "text": "he paid his fine"}
    writes_env.model.run('add_note text="he paid his fine again"')
    writes_env.model.finish()
    writes_env.model.finish()

    result = writes_env.ctx.process_turn("note on Raj Patel's record that he paid his fine")

    assert result.status == TurnStatus.COMPLETED
    assert _library(writes_env).members["m2"].notes == ["prefers email reminders", "he paid his fine"]
    outputs = _outputs(writes_env, "add_note")
    assert [o.success for o in outputs] == [True, False]
    assert outputs[1].command_response.artifacts.get("write_refused") is True
    assert "already ran" in outputs[1].command_response.response


def test_a_failed_write_does_not_use_up_the_phases_count(writes_env):
    # Only a successful run counts: the refusal state is per phase run and a failed attempt can be retried.
    writes_env.ctx._phase_run = {"plan": "1. add_note text=hi"}
    try:
        assert command_writes.phase_write_refusal(writes_env.ctx, writes_env.wf.folderpath, "Record/add_note") is None
        assert command_writes.phase_write_refusal(writes_env.ctx, writes_env.wf.folderpath, "Record/add_note") is None
        command_writes.record_phase_write(writes_env.ctx, writes_env.wf.folderpath, "Record/add_note")
        refusal = command_writes.phase_write_refusal(writes_env.ctx, writes_env.wf.folderpath, "Record/add_note")
        assert refusal is not None and "already ran" in refusal
        writes_env.ctx._phase_run = {"plan": "1. add_note text=hi\n2. add_note text=ho"}
        assert command_writes.phase_write_refusal(writes_env.ctx, writes_env.wf.folderpath, "Record/add_note") is None
    finally:
        writes_env.ctx._phase_run = None


def _note_each_book_plan(item_steps):
    """A fan-out planning command over the author's books; *item_steps(uid)* are one book's steps."""
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.author)]

        @phase(over="find.book_uids", max_items=5, noun="books")
        def book(self, uid) -> OpenedBook:
            return item_steps(uid)

        def __result__(self, input, book):
            return "the books were handled"

    return _build_planning_command(Plan, {"author": (str, Field(description="the author's name"))})


def test_a_fan_out_phase_writes_once_per_item_and_refuses_a_repeat_within_an_item(writes_env):
    # fix-ddis: the count belongs to one item's phase run (the plan is set again for each item), so the
    # step that names add_note once writes once for EACH book; the agent's repeat inside a book is refused.
    writes_env.planning_command = _note_each_book_plan(
        lambda uid: [run("open_book", book_uid=uid), run("add_note", text="checked"),
                     when("the member asked for more on the book", run("open_book", book_uid=uid))])
    writes_env.args = {"author": "Austen"}
    before = _notes(writes_env)
    for _ in range(2):                                  # each of the two books: the agent repeats, then finishes
        writes_env.model.run('add_note text="checked again"')
        writes_env.model.finish()

    result = writes_env.ctx.process_turn("note every Austen book as checked")

    assert result.status == TurnStatus.COMPLETED
    after = _notes(writes_env)
    assert after["b3"] == before["b3"] + ["checked"] and after["b4"] == before["b4"] + ["checked"]
    assert {uid: notes for uid, notes in after.items() if uid not in ("b3", "b4")} == {
        uid: notes for uid, notes in before.items() if uid not in ("b3", "b4")}
    assert [o.success for o in _outputs(writes_env, "add_note")] == [True, False, True, False]
    for refused in _outputs(writes_env, "add_note")[1::2]:
        assert refused.command_response.artifacts.get("write_refused") is True
        assert "already ran" in refused.command_response.response


@pytest.fixture
def flaky_writes_env(tmp_path, initialized_fastworkflow, monkeypatch):
    # The library's own add_note, with one real way to fail: it refuses a note that says "bad".
    path = os.path.join(LIBRARY, "_commands", "Record", "add_note.py")
    source = open(path, encoding="utf-8").read()
    guarded = source.replace(
        "        record = workflow.record\n",
        "        if 'bad' in input.text:\n            raise fastworkflow.Refused('that note is refused')\n"
        "        record = workflow.record\n")
    assert guarded != source
    state = _make_library_runner_env(
        tmp_path, initialized_fastworkflow, monkeypatch,
        extra_commands=WRITES_FILES, real_dispatch=True,
        command_texts={"_commands/Record/add_note.py": guarded})
    yield state
    state.ctx.pop_active_workflow()


def test_a_write_that_fails_does_not_use_up_the_phases_count(flaky_writes_env):
    # fix-ddis: the one step that names add_note is the agent's (a conditional). Its first call really fails
    # (the command refuses it), which must not spend the count: the retry succeeds, and only then is a third
    # call refused as a repeat.
    env = flaky_writes_env
    env.planning_command = _note_each_book_plan(
        lambda uid: [run("open_book", book_uid=uid),
                     when("the member asked to annotate the book", run("add_note", text="annotation"))])
    env.args = {"author": "Austen"}
    before = _notes(env)
    env.model.run('add_note text="a bad note"')
    env.model.run('add_note text="a good note"')
    env.model.run('add_note text="a third note"')
    env.model.finish()
    env.model.finish()

    result = env.ctx.process_turn("annotate every Austen book")

    assert result.status == TurnStatus.COMPLETED
    outputs = _outputs(env, "add_note")
    assert [o.success for o in outputs] == [False, True, False]
    assert outputs[0].command_response.artifacts.get("write_refused") is not True    # it ran, and failed
    assert "refused" in outputs[0].command_response.response
    assert outputs[2].command_response.artifacts.get("write_refused") is True        # the repeat, unrun
    assert "already ran" in outputs[2].command_response.response
    after = _notes(env)
    assert after["b3"] == before["b3"] + ["a good note"]
    assert after["b4"] == before["b4"]


# ----------------------------------------------------------------------
# bd fix-ftdl.7.6: the policy layer's writing commands are the @writes marks (one list)
# ----------------------------------------------------------------------

_TABLE_HEAD = (
    "from fastworkflow.policy.policy_decision import ContractFacts, DecisionTable\n"
    "TABLE = DecisionTable(version='writes-test/1', rows=())\n"
)


def _with_policy_table(tmp_path, body: str) -> str:
    folder = _library_workflow(tmp_path, extra_commands=WRITES_FILES)
    os.makedirs(os.path.join(folder, "_policy"), exist_ok=True)
    with open(os.path.join(folder, "_policy", "table.py"), "w", encoding="utf-8") as handle:
        handle.write(_TABLE_HEAD + body)
    return folder


def test_the_policy_reads_the_writes_marks(tmp_path):
    from fastworkflow.policy.policy_loader import load_policy

    RoutingRegistry.clear_registry()
    folder = _with_policy_table(tmp_path, "FACTS = ContractFacts(failure_pattern='Error')\n")
    _point, facts = load_policy(folder)
    # By qualified name and by leaf name (the policy rows read the name the agent called).
    assert facts.is_write("Record/add_note") and facts.is_write("add_note")
    assert facts.is_write("note_member")                 # writes through its steps
    assert not facts.is_write("find_book") and not facts.is_write("author_books")
    assert facts.reports_failure("Error: no such member")   # the table's other facts are kept
    RoutingRegistry.clear_registry()


def test_a_policy_table_that_lists_its_own_writing_commands_fails_the_load(tmp_path):
    from fastworkflow.policy.policy_decision import PolicyError
    from fastworkflow.policy.policy_loader import load_policy

    RoutingRegistry.clear_registry()
    folder = _with_policy_table(tmp_path, "FACTS = ContractFacts(write_commands=frozenset({'add_note'}))\n")
    with pytest.raises(PolicyError, match=r"FACTS\.write_commands is removed: mark each command that changes "
                                          r"data with @fastworkflow\.writes"):
        load_policy(folder)
    RoutingRegistry.clear_registry()
