"""Navigation is declared once, on the command: ``@descends_to`` (design 3.15, Phase 10 step 2).

Each case runs on a COPY of a real workflow (the library example, or ``tests/todo_list_workflow``).
Nothing is mocked: the routing definition, the context classes and the command classes are the
copy's own. The one stand-in is the CME hop (``script_command_extraction``, from
``tests/test_step_helpers``): the copy has no trained intent heads, so the command name and its
``name=value`` arguments are handed over as a successful extraction would, and
``CommandExecutor._invoke_command_impl`` runs the rest for real.
"""
from __future__ import annotations

import os
import re
import shutil
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.engine.command_executor import CommandExecutor
from fastworkflow.core.context_navigation import DescendEdge, collect_descend_edges
from tests.test_step_helpers import script_command_extraction
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
)  # the registry is cleared per test: command classes are cached by name alone

LIBRARY = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")
TODO = os.path.join(os.path.dirname(__file__), "todo_list_workflow")

_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)

#: The library's navigation edges, recorded before Phase 10 step 2 (the library declared
#: ``descends_to`` on open_book and open_member and no ``enter_command`` anywhere).
LIBRARY_EDGES_BEFORE = (
    DescendEdge("Library", "open_book", "Book", ("book_uid",)),
    DescendEdge("Library", "open_member", "Member", ("member_uid",)),
)

UNDECLARED_FAILURE = (
    'open_book moved the workflow onto a Book record but declares no @descends_to: a command may move the session '
    'only onto a context it declares. Declare @descends_to("Book") on its ResponseGenerator. The move was undone'
)

#: The todo workflow's declared moves (load-check sweep, batch 2: every command that moves declares it). A move within
#: a context (add_child_todolist: a TodoList opens another TodoList) is declared but is no edge.
TODO_EDGES = (
    DescendEdge("TodoList", "add_child_todoitem", "TodoItem", ("description", "assign_to", "is_complete")),
    DescendEdge("TodoList", "get_child_by_id", "TodoItem", ("todoitem_id",)),
    DescendEdge("TodoListManager", "create_todo_list", "TodoList", ("description",)),
    DescendEdge("TodoListManager", "get_todo_list", "TodoList", ("id",)),
)


def _copy(tmp_path, source: str, name: str):
    folder = tmp_path / name
    shutil.copytree(source, folder, ignore=_COPY_IGNORE, symlinks=True)
    return folder


def _edit(path, marker: str, insertion: str) -> None:
    text = path.read_text()
    assert marker in text, path
    path.write_text(text.replace(marker, insertion, 1))


def _declare_descends_to(folder, command_file: str, context: str) -> None:
    _edit(folder / "_commands" / command_file, "class ResponseGenerator:\n",
          f'@fastworkflow.descends_to("{context}")\nclass ResponseGenerator:\n')


def _undeclare_all(folder) -> None:
    for path in (folder / "_commands").rglob("*.py"):
        text = path.read_text()
        if "@fastworkflow.descends_to(" in text:
            path.write_text(re.sub(r'@fastworkflow\.descends_to\("\w+"\)\n', "", text))


def _undeclare_open_book(folder) -> None:
    path = folder / "_commands" / "Library" / "open_book.py"
    text = path.read_text()
    assert '@fastworkflow.descends_to("Book")\n' in text, path
    path.write_text(text.replace('@fastworkflow.descends_to("Book")\n', "", 1))


def _application(folder):
    return fastworkflow.Workflow.create(workflow_folderpath=str(folder),
                                        workflow_id_str=f"descends-{folder.name}")


def _cme_for(app):
    return fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )


def _session(app):
    """The three members _invoke_command_impl reads off a session: the active workflow, the CME
    workflow and its context. The workflows are real."""
    return SimpleNamespace(get_active_workflow=lambda: app, cme_workflow=_cme_for(app))


# ---------------------------------------------------------------------------
# A retired declaration is a load error
# ---------------------------------------------------------------------------

def test_enter_command_on_a_context_class_fails_to_load_naming_descends_to(
        tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_enter_command")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")

    with pytest.raises(ValueError) as raised:
        fastworkflow.RoutingRegistry.get_definition(str(folder))

    assert str(raised.value) == (
        '_commands/TodoList/_TodoList.py: Context.enter_command is no longer read; '
        'declare @descends_to("TodoList") on get_todo_list\'s ResponseGenerator'
    )


def test_enter_command_fails_again_on_every_read_because_nothing_is_cached(
        tmp_path, initialized_fastworkflow):
    """Workflow.create is lazy; the routing read is where the workflow loads, and a failed
    load is not cached, so a second read fails the same way."""
    folder = _copy(tmp_path, TODO, "todo_enter_command_twice")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")
    _application(folder)  # creating the workflow does not read its routing

    for _ in range(2):
        with pytest.raises(ValueError, match=re.escape('declare @descends_to("TodoList") on get_todo_list')):
            fastworkflow.RoutingRegistry.get_definition(str(folder))


def test_enter_commands_list_fails_to_load_naming_each_command(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_enter_commands")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_commands = ['get_todo_list <todo_list_id>', 'add_child_todolist x']\n")

    with pytest.raises(ValueError) as raised:
        fastworkflow.RoutingRegistry.get_definition(str(folder))

    message = str(raised.value)
    assert "Context.enter_commands is no longer read" in message
    assert 'declare @descends_to("TodoList") on get_todo_list\'s ResponseGenerator' in message
    assert 'declare @descends_to("TodoList") on add_child_todolist\'s ResponseGenerator' in message


def test_enter_command_stays_a_load_error_when_the_definition_was_persisted(
        tmp_path, initialized_fastworkflow):
    """A routing definition saved before the retirement is not trusted to skip the check."""
    folder = _copy(tmp_path, TODO, "todo_enter_command_persisted")
    fastworkflow.RoutingRegistry.get_definition(str(folder))  # saves routing_definition.json
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")
    fastworkflow.RoutingRegistry.clear_registry()

    with pytest.raises(ValueError, match="Context.enter_command is no longer read"):
        fastworkflow.RoutingRegistry.get_definition(str(folder))


# ---------------------------------------------------------------------------
# @descends_to is the edge
# ---------------------------------------------------------------------------

def test_descends_to_on_a_command_is_the_collected_edge(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_descends")
    _undeclare_all(folder)
    _declare_descends_to(folder, "TodoListManager/get_todo_list.py", "TodoList")

    edges = collect_descend_edges(str(folder))

    assert edges == (
        DescendEdge("TodoListManager", "get_todo_list", "TodoList", ("id",)),
    )


def test_the_todo_workflows_declared_moves_are_its_edges(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_declared")

    assert set(collect_descend_edges(str(folder))) == set(TODO_EDGES)


def test_a_command_without_descends_to_collects_no_edge(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_no_descends")
    _undeclare_all(folder)

    assert collect_descend_edges(str(folder)) == ()


def test_library_navigation_edges_are_unchanged_by_the_migration(tmp_path, initialized_fastworkflow):
    """Before and after: the library never declared enter_command, so its @descends_to edges
    (recorded before this change) are the same set."""
    folder = _copy(tmp_path, LIBRARY, "library_edges")

    assert set(collect_descend_edges(str(folder))) == set(LIBRARY_EDGES_BEFORE)


# ---------------------------------------------------------------------------
# A move the command does not declare fails the command, every time
# ---------------------------------------------------------------------------

def test_an_undeclared_move_fails_the_command_every_time_and_the_record_is_put_back(
        tmp_path, initialized_fastworkflow, monkeypatch):
    folder = _copy(tmp_path, LIBRARY, "library_undeclared")
    _undeclare_open_book(folder)
    app = _application(folder)
    app.root = _library_root()
    app.record = app.root
    session = _session(app)
    script_command_extraction(monkeypatch, app)

    for uid in ("b1", "b2"):
        output = CommandExecutor._invoke_command_impl(session, f"open_book book_uid={uid}")
        assert not output.command_response.success
        assert output.command_response.response == UNDECLARED_FAILURE
        assert app._current_command_context is app.root


def test_a_declared_move_is_made(tmp_path, initialized_fastworkflow, monkeypatch):
    folder = _copy(tmp_path, LIBRARY, "library_declared")
    app = _application(folder)
    app.root = _library_root()
    app.record = app.root
    session = _session(app)
    script_command_extraction(monkeypatch, app)

    output = CommandExecutor._invoke_command_impl(session, "open_book book_uid=b1")

    assert output.command_response.success, output.command_response.response
    assert app._current_command_context_name == "Book"


def _library_root():
    from fastworkflow.examples.library.application.library import Library

    return Library()
