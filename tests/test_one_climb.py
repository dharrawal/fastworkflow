"""One climb, used by every entry path (bd fix-ftdl.3, fix-qjne.29, fix-84bk item 1).

The rule: a command runs on the record of the scope that owns it, whichever way it was reached. A message already
does this (``intent_loop.choose``); these tests pin it for the direct entry paths that go through
``CommandExecutor.perform_action`` (a direct action over HTTP, an MCP tool call, ``fastworkflow command run-code``, the
approval re-run), and pin the one live chain every lookup reads (``context_navigation.live_chain``): identity for the
root and for a repeated record, and one error for a context class on the path with no ``get_parent``.

Real workflows only, copied to ``tmp_path`` (the library example and ``tests/todo_list_workflow``); no model is called.
"""
from __future__ import annotations

import os
import shutil
import uuid

import pytest

import fastworkflow
from fastworkflow import context_navigation, intent_detector
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.planning import run_code
from fastworkflow.workflow_agent import planning_scopes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
TODO = os.path.join(ROOT, "tests", "todo_list_workflow")

#: A global command that says which record it was handed (written into the copy only).
WHERE_AM_I = '''"""where_am_i: name the record this command runs on."""
import fastworkflow


class Signature:
    """Name the record this command runs on."""

    plain_utterances = ["which record do you run on", "where does this run", "name the record you act on"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        return type(workflow.record).__name__
'''

#: A Library command that moves the workflow into a Book without declaring it (written into the copy only).
SNEAK_INTO_BOOK = '''"""sneak_into_book: open b1 without declaring the move."""
import fastworkflow


class Signature:
    """Open the first book without saying so."""

    plain_utterances = ["sneak into the first book", "quietly open b1", "slip into book one"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        library = workflow.record
        workflow.record = library.books["b1"]
        return "in"
'''


def _copy(source: str, destination) -> str:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    return str(destination)


@pytest.fixture
def library(tmp_path) -> str:
    folder = _copy(LIBRARY, tmp_path / "library")
    with open(os.path.join(folder, "_commands", "where_am_i.py"), "w") as handle:
        handle.write(WHERE_AM_I)
    with open(os.path.join(folder, "_commands", "Library", "sneak_into_book.py"), "w") as handle:
        handle.write(SNEAK_INTO_BOOK)
    return folder


@pytest.fixture
def todo(tmp_path) -> str:
    return _copy(TODO, tmp_path / "todo_list_workflow")


def _started(folder: str, name: str):
    if fastworkflow.RoutingRegistry is None:
        fastworkflow.init({})
    workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"one-climb-{name}-{uuid.uuid4().hex}")
    CommandExecutor.perform_action(workflow, fastworkflow.workflow_startup_action(folder))
    return workflow


def _act(workflow, command_name: str, **parameters):
    return CommandExecutor.perform_action(
        workflow, fastworkflow.Action(command_name=command_name, command=command_name, parameters=parameters))


# ------------------------------------------------------------------ a direct action binds the owner's record

def test_a_direct_action_of_an_enclosing_command_runs_on_the_enclosing_record(library):
    """Standing on a Member, Library/count_overdue runs on the Library; the workflow stays on the Member."""
    workflow = _started(library, "enclosing")
    _act(workflow, "Library/open_member", member_uid="m1")
    member = workflow._current_command_context
    assert type(member).__name__ == "Member"

    output = _act(workflow, "Library/count_overdue")

    assert output.command_response.success, output.command_response.response
    assert workflow._current_command_context is member          # the current position did not move
    assert output.context == "Library"                          # filed under the record it ran on, as a message is
    assert output.ran_in_context_name == "Library"


def test_a_direct_action_of_a_global_command_runs_on_the_root(library):
    """After opening a book, a global command is handed the root, not the book."""
    workflow = _started(library, "global")
    _act(workflow, "Library/open_book", book_uid="b1")
    assert type(workflow._current_command_context).__name__ == "Book"

    output = _act(workflow, "where_am_i")

    assert output.command_response.response == "Library"
    assert type(workflow._current_command_context).__name__ == "Book"


def test_a_direct_action_on_the_current_record_is_filed_as_before(library):
    workflow = _started(library, "current")
    _act(workflow, "Library/open_book", book_uid="b1")

    output = _act(workflow, "Book/show_book")

    assert output.command_response.success, output.command_response.response
    assert output.context == "Book: Dune (b1)"
    assert output.ran_in_context_name is None


def test_an_mcp_tool_call_climbs_like_a_message(library):
    workflow = _started(library, "mcp")
    _act(workflow, "Library/open_member", member_uid="m1")

    result = CommandExecutor.perform_mcp_tool_call(
        workflow, fastworkflow.MCPToolCall(name="Library/count_overdue", arguments={}))

    assert not result.isError, result.content[0].text


def test_a_direct_action_that_moves_without_declaring_it_fails_and_the_record_is_put_back(library):
    """A command may move the session only onto a context it declares (load-check sweep, batch 2)."""
    workflow = _started(library, "move")
    before = workflow._current_command_context

    output = _act(workflow, "Library/sneak_into_book")

    assert not output.command_response.success
    assert ('sneak_into_book moved the workflow onto a Book record but declares no @descends_to: a command may move '
            'the session only onto a context it declares. Declare @descends_to("Book")'
            in output.command_response.response)
    assert workflow._current_command_context is before


# ------------------------------------------------------------------ run-code climbs (fix-84bk item 1)

def test_run_code_runs_an_enclosing_command_from_a_member(library):
    result = run_code(library, "count_overdue", start_with=["open_member member_uid=m1"])
    assert "context: Library" in result.text


def test_run_code_runs_the_fan_out_example_over_every_item(library):
    """The once-per-item page's own check: author_books opens each book from the Library, the second item included."""
    result = run_code(library, "author_books", {"author": "Jane Austen"})
    assert result.text.count("Phase book (item") == 2
    assert result.open_step is None


def test_run_code_runs_an_enclosing_planning_command_from_a_member(library):
    """A Library planning command is on a Member's lookup path, so run-code runs it from there."""
    result = run_code(library, "author_books", {"author": "Jane Austen"}, start_with=["open_member member_uid=m1"])
    assert result.text.count("Phase book (item") == 2


# ------------------------------------------------------------------ workflow.record outside a command

def test_workflow_record_outside_a_command_is_the_current_position(library):
    workflow = _started(library, "getter")
    library_record = workflow.root
    assert workflow.record is library_record
    book = library_record.books["b1"]
    workflow.record = book
    assert workflow.record is book
    assert workflow.is_record_root is False


# ------------------------------------------------------------------ the one live chain

def _nested_todo(todo: str):
    from tests.todo_list_workflow.application.todo_list import TodoList
    from tests.todo_list_workflow.application.todo_manager import TodoListManager

    if fastworkflow.RoutingRegistry is None:
        fastworkflow.init({})
    workflow = fastworkflow.Workflow.create(todo, workflow_id_str=f"one-climb-todo-{uuid.uuid4().hex}")
    manager = TodoListManager(file_path=os.path.join(todo, "lists.json"), load_from_file=False)
    outer = TodoList(1, "house")
    outer.parent = manager
    inner = TodoList(2, "kitchen")
    inner.parent = outer
    workflow.root = manager
    workflow.record = inner
    return workflow, manager, outer, inner


def test_the_live_chain_of_a_nested_record_of_the_same_class_reaches_the_root(todo):
    workflow, manager, outer, inner = _nested_todo(todo)

    chain = list(context_navigation.live_chain(workflow))

    assert [record for record, _ in chain] == [inner, outer, manager]
    assert [name for _, name in chain] == ["TodoList", "TodoList", "TodoListManager"]
    assert context_navigation.live_context_chain(workflow) == ("TodoList", "TodoListManager", "*")
    assert [scope.record for scope in intent_detector.scopes_for(workflow)] == [inner, outer, manager]
    assert [s.context_name for s in context_navigation.exact_name_scopes(workflow)][-2:] == ["TodoListManager", "*"]
    assert "TodoListManager" in planning_scopes(workflow)


def test_the_root_is_found_by_identity(todo):
    workflow, manager, outer, inner = _nested_todo(todo)
    assert list(context_navigation.live_chain(workflow, manager)) == [(manager, "TodoListManager")]
    assert len(list(context_navigation.live_chain(workflow, inner))) == 3


def test_a_context_class_with_no_get_parent_is_one_error_naming_the_context(library):
    """Every walker raises the same error, naming the context and what to add; none stops silently."""
    path = os.path.join(library, "_commands", "Member", "_Member.py")
    with open(path) as handle:
        source = handle.read()
    with open(path, "w") as handle:
        # a helper's name starts with _: a public unknown member of Context is a load error of its own
        handle.write(source.replace("def get_parent(", "def _not_get_parent("))
    workflow = _started(library, "no-parent")
    member = workflow.root.members["m1"]
    workflow.record = member

    expected = r"Member.*has no get_parent.*_commands/Member/_Member\.py"
    for walker in (lambda: list(context_navigation.live_chain(workflow)),
                   lambda: context_navigation.live_context_chain(workflow),
                   lambda: context_navigation.exact_name_scopes(workflow),
                   lambda: planning_scopes(workflow),
                   lambda: intent_detector.scopes_for(workflow)):
        with pytest.raises(AttributeError, match=expected):
            walker()


# ------------------------------------------------------------------ a move is checked (the load-check sweep, item 8)

#: A Library command that moves into a Book and then ends as its *ending* says (written into the copy only).
_MOVE_THEN = '''"""{name}: open b1, then {ending}."""
import fastworkflow
from fastworkflow import descends_to
from fastworkflow.workflow_execution_context import CommandCancelledError


class Signature:
    """Open the first book, then {ending}."""

    plain_utterances = ["open the first book and {ending}", "go into b1 then {ending}", "b1, then {ending}"]


class _Shelf:
    pass


{decorator}
class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        library = workflow.record
        workflow.record = {target}
        {body}
'''


def _write_mover(folder: str, name: str, ending: str, body: str, target: str = 'library.books["b1"]',
                 decorator: str = "") -> None:
    with open(os.path.join(folder, "_commands", "Library", f"{name}.py"), "w") as handle:
        handle.write(_MOVE_THEN.format(name=name, ending=ending, body=body, target=target, decorator=decorator))


def test_a_command_that_moves_and_then_raises_leaves_the_record_where_it_was(library):
    _write_mover(library, "move_then_crash", "crash", 'raise RuntimeError("backend down")')
    workflow = _started(library, "move-crash")
    before = workflow._current_command_context
    with pytest.raises(RuntimeError, match="backend down"):
        _act(workflow, "Library/move_then_crash")
    assert workflow._current_command_context is before


def test_a_command_that_moves_and_then_refuses_leaves_the_record_where_it_was(library):
    _write_mover(library, "move_then_refuse", "refuse", 'raise fastworkflow.Refused("no such shelf")')
    workflow = _started(library, "move-refuse")
    before = workflow._current_command_context

    output = _act(workflow, "Library/move_then_refuse")

    assert not output.command_response.success and output.command_response.response == "no such shelf"
    assert workflow._current_command_context is before


def test_a_suspended_command_keeps_its_move(library):
    """An ask_user suspension is no failure: the command resumes where it stood."""
    _write_mover(library, "move_then_ask", "ask", 'raise CommandCancelledError("asked")')
    workflow = _started(library, "move-ask")
    with pytest.raises(BaseException, match="asked"):
        _act(workflow, "Library/move_then_ask")
    assert type(workflow._current_command_context).__name__ == "Book"


def test_a_move_onto_a_record_whose_class_is_no_context_fails_the_command(library):
    _write_mover(library, "move_onto_shelf", "stop", 'return "on a shelf"', target="_Shelf()")
    workflow = _started(library, "move-shelf")
    before = workflow._current_command_context

    output = _act(workflow, "Library/move_onto_shelf")

    assert not output.command_response.success
    assert ("move_onto_shelf moved the workflow onto a _Shelf record, but _Shelf is no context of the workflow"
            in output.command_response.response)
    assert "_commands/_Shelf/" in output.command_response.response
    assert workflow._current_command_context is before


def test_a_move_that_disagrees_with_the_declared_target_names_the_declared_one(library):
    _write_mover(library, "open_member_wrongly", "stop", 'return "in"', decorator='@descends_to("Member")')
    workflow = _started(library, "move-declared")
    before = workflow._current_command_context

    output = _act(workflow, "Library/open_member_wrongly")

    assert not output.command_response.success
    assert ('open_member_wrongly moved the workflow onto a Book record but declares @descends_to("Member"), and Book '
            'is not Member nor a context that inherits from it' in output.command_response.response)
    assert workflow._current_command_context is before
