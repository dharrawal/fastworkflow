"""A root object's context class needs no ``get_parent`` (bd fix-qjne.8).

``Workflow.get_parent`` answers the root itself without asking its context class, so every reader
that climbs (``go_up``, the lookup path, the live context chain, the intent walk) stops at the root
before a context class's ``get_parent`` could be called. A callback file of a ROOT context that
still writes ``return None`` is ceremony, and the shipped workflows no longer carry it.

The behaviour tests run on a COPY of ``tests/todo_list_workflow`` whose root callback class has the
method stripped, whatever the shipped file says; the last test reads the shipped files.
"""
import ast
import os
import re
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import context_navigation
from fastworkflow.workflow_agent import _explicit_agent_command
from tests.test_lookup_exact_names import GLOBAL_PROBE, _RefusingRouter, _run
from tests.todo_list_workflow.application.todo_item import TodoItem
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

REPO = Path(__file__).resolve().parent.parent
TODO_SOURCE = REPO / "tests" / "todo_list_workflow"

_GET_PARENT = re.compile(
    r"    @classmethod\n    def get_parent\(cls, record: \w+\)[^\n]*:\n        return None\n\n?")


def _has_get_parent(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(isinstance(node, ast.FunctionDef) and node.name == "get_parent"
               for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) for node in cls.body)


@pytest.fixture
def rootless(tmp_path):
    """A todo workflow copy whose root context class has no ``get_parent``: item -> list -> manager (root)."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        TODO_SOURCE, workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    (workflow_path / "_commands" / "global_probe.py").write_text(GLOBAL_PROBE)
    callback = workflow_path / "_commands" / "TodoListManager" / "_TodoListManager.py"
    callback.write_text(_GET_PARENT.sub("", callback.read_text()))
    assert not _has_get_parent(callback)

    app = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path), workflow_id_str=f"rootless-{tmp_path.name}")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id, workflow_context={"app_workflow": app})
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    item = TodoItem(1, "milk")
    item.parent = todo_list
    app.root = manager
    app.record = item
    return app, cme, item, todo_list, manager


def test_the_root_has_no_get_parent_and_the_framework_still_climbs_to_it(rootless, setup_test_environment):
    app, _, item, todo_list, manager = rootless
    context_class = fastworkflow.CommandContextModel.load(app.folderpath).get_context_class(
        "TodoListManager", fastworkflow.ModuleType.CONTEXT_CLASS)
    assert context_class is not None and not hasattr(context_class, "get_parent")

    assert app.get_parent(item) is todo_list
    assert app.get_parent(todo_list) is manager
    assert app.get_parent(manager) is manager
    assert context_navigation.live_context_chain(app) == ("TodoItem", "TodoList", "TodoListManager", "*")
    app.record = manager
    assert context_navigation.live_context_chain(app) == ("TodoListManager", "*")


def test_go_up_climbs_to_the_root_and_stops_there(rootless, setup_test_environment):
    from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.go_up import (
        ResponseGenerator)
    app, _, _, _, manager = rootless
    wrapper = fastworkflow.Workflow.create(workflow_folderpath="/tmp", workflow_id_str=f"wrap-{app.id}")
    wrapper.context = {"app_workflow": app}

    ResponseGenerator()(wrapper, "IntentDetection/go_up", None)
    ResponseGenerator()(wrapper, "IntentDetection/go_up", None)
    assert app._current_command_context is manager
    assert app._is_current_command_context_root

    text = ResponseGenerator()(wrapper, "IntentDetection/go_up", None)
    assert text.startswith("Already at the top-level 'global' context.")
    assert app._current_command_context is manager


def test_the_lookup_path_from_a_nested_record_ends_at_the_root_then_global(rootless, setup_test_environment):
    app, _, _, _, _ = rootless
    scopes = [scope.context_name for scope in context_navigation.exact_name_scopes(app)]
    assert scopes[-1] == "*"
    assert scopes.index("TodoList") < scopes.index("TodoListManager") < scopes.index("*")


def test_a_command_of_the_root_by_exact_name_runs_from_a_nested_record(rootless, monkeypatch, setup_test_environment):
    app, cme, item, _, manager = rootless
    monkeypatch.setattr(fastworkflow.model_pipeline_training, "CommandRouter", _RefusingRouter)
    assert _explicit_agent_command("list_todo_lists", app) == "list_todo_lists"

    run = _run(app, cme, monkeypatch, "list_todo_lists")

    assert run.command_name == "TodoListManager/list_todo_lists"
    assert app._command_context_for_response_generation is manager
    assert run.ran_on_context_name == "TodoListManager"
    assert app._current_command_context is item


def test_a_root_context_class_with_a_get_parent_still_loads(tmp_path, setup_test_environment):
    """Nothing forbids the method: a root whose class still has it loads and routes as before."""
    workflow_path = tmp_path / "todo_with"
    shutil.copytree(
        TODO_SOURCE, workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    callback = workflow_path / "_commands" / "TodoListManager" / "_TodoListManager.py"
    if not _has_get_parent(callback):
        callback.write_text(callback.read_text().replace(
            "class Context:\n",
            "class Context:\n    @classmethod\n    def get_parent(cls, record):\n        return None\n\n", 1))
    assert _has_get_parent(callback)
    app = fastworkflow.Workflow.create(workflow_folderpath=str(workflow_path), workflow_id_str=f"with-{tmp_path.name}")
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    app.root = manager
    assert app.get_parent(manager) is manager


SHIPPED_ROOT_CALLBACKS = [
    "fastworkflow/examples/library/_commands/Library/_Library.py",
    "fastworkflow/examples/messaging_app_2/_commands/User/_User.py",
    "fastworkflow/examples/messaging_app_4/_commands/ChatRoom/_ChatRoom.py",
    "tests/todo_list_workflow/_commands/TodoListManager/_TodoListManager.py",
]


#: messaging_app_3 never sets a root (its startup assigns ``workflow.record``), so its User and PremiumUser are not
#: roots: they keep ``get_parent`` returning None, which sends the climb straight to the global commands (bd fix-ftdl.8.1).
ROOTLESS_CALLBACKS = [
    "fastworkflow/examples/messaging_app_3/_commands/User/_User.py",
    "fastworkflow/examples/messaging_app_3/_commands/PremiumUser/_PremiumUser.py",
]


@pytest.mark.parametrize("relative", ROOTLESS_CALLBACKS)
def test_a_context_of_a_workflow_with_no_root_keeps_its_get_parent(relative):
    assert _has_get_parent(REPO / relative), f"{relative}: not a root, so the climb asks it for the enclosing record"


@pytest.mark.parametrize("relative", SHIPPED_ROOT_CALLBACKS)
def test_a_shipped_root_context_class_writes_no_get_parent(relative):
    assert not _has_get_parent(REPO / relative), f"{relative}: a root needs no get_parent"


MESSAGING_APP_3 = REPO / "fastworkflow" / "examples" / "messaging_app_3"


def test_a_context_with_no_get_parent_in_a_workflow_with_no_root_goes_to_the_global_commands(
        tmp_path, setup_test_environment):
    """No root: a context class with no ``get_parent`` means "no enclosing record", as ``return None`` does, so the
    first climb from its record reaches the global commands instead of raising (load-check sweep, item 1)."""
    from fastworkflow.examples.messaging_app_3.application.user import User

    workflow_path = tmp_path / "messaging_app_3"
    shutil.copytree(
        MESSAGING_APP_3, workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    callback = workflow_path / "_commands" / "User" / "_User.py"
    callback.write_text(_GET_PARENT.sub("", callback.read_text()))
    assert not _has_get_parent(callback)
    app = fastworkflow.Workflow.create(workflow_folderpath=str(workflow_path), workflow_id_str=f"m3-{tmp_path.name}")
    assert app.root is None
    user = User("Jack")
    app.record = user

    assert app.get_parent(user) is None
    assert context_navigation.live_context_chain(app) == ("User", "*")


def test_a_context_with_no_get_parent_under_a_root_still_raises_on_the_climb(tmp_path, setup_test_environment):
    """With a root, a record that is not the root needs its context's answer: the error stays."""
    workflow_path = tmp_path / "todo_no_item_parent"
    shutil.copytree(
        TODO_SOURCE, workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    callback = workflow_path / "_commands" / "TodoItem" / "_TodoItem.py"
    callback.write_text(callback.read_text().replace(
        "    @classmethod\n    def get_parent(cls, record: TodoItem) -> Optional[TodoList]:\n        return record.parent",
        "    pass"))
    assert not _has_get_parent(callback)
    app = fastworkflow.Workflow.create(workflow_folderpath=str(workflow_path), workflow_id_str=f"rooted-{tmp_path.name}")
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    item = TodoItem(1, "milk")
    item.parent = todo_list
    app.root = manager
    with pytest.raises(AttributeError, match="has no get_parent"):
        app.get_parent(item)
