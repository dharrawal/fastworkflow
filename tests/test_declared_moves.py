"""Declared moves and the enclosing record: one rule each (load-check sweep, batch 2).

Moves: a command may move the session only onto a context it declares with ``@descends_to``; a move onto any other
context fails the command and the record is put back (``command_executor._check_move``). "Onto the declared context"
means the record's context is the declared one or one that inherits from it (``inherits`` on its ``class Context``). A
command that declares a move and does not make it has not moved; setting ``workflow.root`` is not a record move; a move
onto another record of the command's own context is a move and is declared like any other.

The enclosing record: with a root, a record that is not the root whose context has no ``get_parent`` (no
``_<Context>.py`` or no method) is ``MissingGetParentError`` at the climb, and a command that moves onto one fails.

Real workflows copied to ``tmp_path`` (``simple_workflow_template``, ``messaging_app_3``, ``messaging_app_4``,
``tests/todo_list_workflow``, the library); no model is called.
"""
from __future__ import annotations

import os
import shutil
import uuid

import pytest

import fastworkflow
from fastworkflow.planning import command_preview
from fastworkflow.core import context_navigation
from fastworkflow.core.command_directory import CommandLoadError
from fastworkflow.engine.command_executor import CommandExecutor, call_response_generator
from fastworkflow.core.command_routing import ModuleType
from fastworkflow.core.workflow import MissingGetParentError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "fastworkflow", "examples")
SIMPLE = os.path.join(EXAMPLES, "simple_workflow_template")
MESSAGING_3 = os.path.join(EXAMPLES, "messaging_app_3")
MESSAGING_4 = os.path.join(EXAMPLES, "messaging_app_4")
LIBRARY = os.path.join(EXAMPLES, "library")
TODO = os.path.join(ROOT, "tests", "todo_list_workflow")

DECORATOR = '@fastworkflow.descends_to("{}")\n'


def _copy(source: str, tmp_path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    return str(destination)


def _started(folder: str):
    if fastworkflow.RoutingRegistry is None:
        fastworkflow.init({})
    workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"declared-moves-{uuid.uuid4().hex}")
    if os.path.isfile(os.path.join(folder, "startup_action.json")):
        output = CommandExecutor.perform_action(workflow, fastworkflow.workflow_startup_action(folder))
        assert output.command_response.success, output.command_response.response
    return workflow


def _act(workflow, command_name: str, **parameters):
    return CommandExecutor.perform_action(
        workflow, fastworkflow.Action(command_name=command_name, command=command_name, parameters=parameters))


def _edit(path: str, old: str, new: str) -> None:
    with open(path) as handle:
        text = handle.read()
    assert old in text, path
    with open(path, "w") as handle:
        handle.write(text.replace(old, new, 1))


def _epic_with_a_story(folder: str):
    workflow = _started(folder)
    epic = workflow.root
    output = _act(workflow, "WorkItem/add_child_workitem", workitem_type="Story")
    assert output.command_response.success, output.command_response.response
    assert workflow._current_command_context is epic
    return workflow, epic, output.command_response.output.new_child_path


# ------------------------------------------------------------------ the move rule

def test_the_startup_command_sets_the_root_and_is_no_move(tmp_path):
    """``workflow.root = ...`` is not ``workflow.record = ...``: the startup command declares nothing and succeeds."""
    folder = _copy(MESSAGING_4, tmp_path, "messaging_app_4")
    workflow = _started(folder)          # asserts the startup command's success
    assert type(workflow.root).__name__ == "ChatRoom"
    assert workflow._current_command_context is workflow.root


def test_a_move_onto_another_record_of_the_commands_own_context_is_declared_and_made(tmp_path):
    workflow, epic, story_path = _epic_with_a_story(_copy(SIMPLE, tmp_path, "simple_workflow_template"))

    output = _act(workflow, "WorkItem/go_to_workitem", path=story_path)

    assert output.command_response.success, output.command_response.response
    story = workflow._current_command_context
    assert story is not epic and type(story).__name__ == "WorkItem"


def test_an_undeclared_move_onto_another_record_of_the_same_context_fails(tmp_path):
    folder = _copy(SIMPLE, tmp_path, "simple_undeclared")
    _edit(os.path.join(folder, "_commands", "WorkItem", "go_to_workitem.py"), DECORATOR.format("WorkItem"), "")
    workflow, epic, story_path = _epic_with_a_story(folder)

    output = _act(workflow, "WorkItem/go_to_workitem", path=story_path)

    assert not output.command_response.success
    assert output.command_response.response == (
        'go_to_workitem moved the workflow onto a WorkItem record but declares no @descends_to: a '
        'command may move the session only onto a context it declares. Declare @descends_to("WorkItem") on its '
        'ResponseGenerator. The move was undone')
    assert workflow._current_command_context is epic


def test_a_move_onto_a_context_that_inherits_from_the_declared_one_is_made(tmp_path):
    """``@descends_to("User")`` covers a PremiumUser: PremiumUser's base is User in the inheritance file."""
    from fastworkflow.examples.messaging_app_4.application.user import PremiumUser, User

    workflow = _started(_copy(MESSAGING_4, tmp_path, "messaging_app_4"))
    room = workflow.root
    room.add_user(User(room, "ann"))
    room.add_user(PremiumUser(room, "bob"))

    output = _act(workflow, "ChatRoom/set_current_user", user_name="bob")

    assert output.command_response.success, output.command_response.response
    assert type(workflow._current_command_context).__name__ == "PremiumUser"


def test_a_command_that_declares_a_move_and_does_not_make_it_has_not_moved(tmp_path):
    workflow = _started(_copy(MESSAGING_4, tmp_path, "messaging_app_4"))

    output = _act(workflow, "ChatRoom/set_current_user", user_name="nobody")

    assert output.command_response.success, output.command_response.response
    assert output.command_response.output.user_found is False
    assert workflow._current_command_context is workflow.root


def test_a_command_that_opens_a_record_and_puts_its_own_back_has_not_moved(tmp_path):
    folder = _copy(LIBRARY, tmp_path, "library")
    with open(os.path.join(folder, "_commands", "Library", "peek_into_book.py"), "w") as handle:
        handle.write('''"""peek_into_book: open b1, read its title, come back."""
import fastworkflow


class Signature:
    """Read the first book's title from the library."""

    plain_utterances = ["peek into the first book", "what is b1 called", "glance at book one"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        library = workflow.record
        book = library.books["b1"]
        workflow.record = book
        workflow.record = library
        return book.title
''')
    workflow = _started(folder)

    output = _act(workflow, "Library/peek_into_book")

    assert output.command_response.success, output.command_response.response
    assert workflow._current_command_context is workflow.root


class _UndeclaredOpener:
    """A nested generator, run with no command name, that moves without declaring it."""

    def __call__(self, workflow, command, command_parameters):
        workflow.record = workflow.record.users[0]
        return "moved"


def test_a_nested_call_reads_the_declaration_from_the_generator_class_that_ran(tmp_path):
    """A command that runs another command's generator calls it with no command name: the class carries the rule."""
    from fastworkflow.examples.messaging_app_4.application.user import User

    folder = _copy(MESSAGING_4, tmp_path, "messaging_app_4")
    workflow = _started(folder)
    room = workflow.root
    room.add_user(User(room, "ann"))
    routing = fastworkflow.RoutingRegistry.get_definition(folder)
    generator = routing.get_command_class("ChatRoom/set_current_user", ModuleType.RESPONSE_GENERATION_INFERENCE)
    parameters = routing.get_command_class("ChatRoom/set_current_user", ModuleType.COMMAND_PARAMETERS_CLASS)

    workflow._command_context_for_response_generation = room      # as a dispatch binds it
    declared = call_response_generator(generator(), workflow, "", parameters.model_construct(user_name="ann"))
    assert declared.command_response.success, declared.command_response.response
    assert type(workflow._current_command_context).__name__ == "User"

    workflow.record = room
    undeclared = call_response_generator(_UndeclaredOpener(), workflow, "", None)
    assert not undeclared.command_response.success
    assert undeclared.command_response.response.startswith(
        "test_declared_moves moved the workflow onto a User record but declares no @descends_to")
    assert workflow._current_command_context is room


def test_a_move_within_its_own_context_is_not_an_entry_command(tmp_path):
    """Declaring ``@descends_to("WorkItem")`` on a WorkItem command with no required input does not hide it from the
    remote model: it enters no other context, so it is no edge and no entry-only command."""
    folder = _copy(SIMPLE, tmp_path, "simple_entry")
    _started(folder)
    assert context_navigation.collect_descend_edges(folder) == ()
    assert context_navigation.context_entry_commands(folder) == frozenset()


# ------------------------------------------------------------------ the enclosing record

def _todo_without_item_callback(tmp_path) -> str:
    folder = _copy(TODO, tmp_path, "todo_no_item_callback")
    os.remove(os.path.join(folder, "_commands", "TodoItem", "_TodoItem.py"))
    return folder


def test_with_a_root_a_move_onto_a_record_whose_context_has_no_get_parent_fails(tmp_path):
    """A context with no get_parent declares no parents (bd fix-qjne.36), so a command that descends to it from another
    context is stopped when the workflow loads, naming the parents to declare; acting it raises the load error and the
    record stays."""
    folder = _todo_without_item_callback(tmp_path)
    problems = [str(problem) for problem in command_preview.load_problems(folder)]
    line = next(problem for problem in problems if problem.startswith("TodoList/add_child_todoitem: "))
    assert "TodoItem declares no parents" in line and 'Add "TodoList" to the parents of TodoItem' in line

    workflow = _started(folder)
    output = _act(workflow, "TodoListManager/create_todo_list", description="house")
    assert output.command_response.success, output.command_response.response
    todo_list = workflow._current_command_context

    with pytest.raises(CommandLoadError, match="TodoItem declares no parents"):
        _act(workflow, "TodoList/add_child_todoitem", description="milk", assign_to="me", is_complete=False)

    assert workflow._current_command_context is todo_list


def test_with_a_root_the_climb_from_such_a_record_is_one_named_error(tmp_path):
    from tests.todo_list_workflow.application.todo_item import TodoItem

    workflow = _started(_todo_without_item_callback(tmp_path))
    _act(workflow, "TodoListManager/create_todo_list", description="house")
    item = TodoItem(1, "milk")
    item.parent = workflow._current_command_context
    workflow.record = item

    with pytest.raises(MissingGetParentError, match=r"the context TodoItem has no get_parent \(there is no "
                                                    r"_commands/TodoItem/_TodoItem\.py\)"):
        context_navigation.live_context_chain(workflow)


def test_with_no_root_no_callback_file_means_the_global_commands(tmp_path):
    folder = _copy(MESSAGING_3, tmp_path, "messaging_app_3")
    os.remove(os.path.join(folder, "_commands", "User", "_User.py"))
    workflow = _started(folder)
    assert workflow.root is None

    output = _act(workflow, "initialize_user", name="ann", is_premium_user=False)

    assert output.command_response.success, output.command_response.response
    assert type(workflow._current_command_context).__name__ == "User"
    assert context_navigation.live_context_chain(workflow) == ("User", "*")


def test_a_move_that_cannot_be_checked_fails_the_command(tmp_path):
    """Fail closed: when the workflow's contexts cannot be read after a command moved, the move is not taken on trust."""
    folder = _copy(LIBRARY, tmp_path, "library_unreadable")
    with open(os.path.join(folder, "_commands", "Library", "open_and_break.py"), "w") as handle:
        handle.write('''"""open_and_break: spoil the inheritance file, then open b1."""
import os

import fastworkflow


class Signature:
    """Open the first book after spoiling the workflow's inheritance file."""

    plain_utterances = ["open and break", "spoil then open b1", "break the file and open book one"]


@fastworkflow.descends_to("Book")
class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        with open(os.path.join(workflow.folderpath, "_commands", "context_inheritance_model.json"), "w") as handle:
            handle.write("{")
        library = workflow.record
        workflow.record = library.books["b1"]
        return "in"
''')
    workflow = _started(folder)

    output = _act(workflow, "Library/open_and_break")

    assert not output.command_response.success
    assert output.command_response.response.startswith(
        "open_and_break moved the workflow onto a Book record, but the move could not be checked: the workflow's "
        "contexts could not be read (CommandContextModelValidationError")
    assert output.command_response.response.endswith("The move was undone")
    assert workflow._current_command_context is workflow.root
