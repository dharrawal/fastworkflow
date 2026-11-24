"""Six author mistakes that used to fail silently are explicit (bd fix-qjne.16).

Each mistake is built on a TEMP COPY of the library example's commands (``tests.test_field_sources._build_workflow``).
The convention is the one of ``tests/test_producer_declarations.py``: a bad command file is ONE ``<command>: <reason>``
line of ``command_preview.load_problems``, that command is unavailable, and the rest of the workflow loads. The last
class of tests is the run-time warning of a record whose enclosing record cannot be found.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import ModuleType, command_preview, workflow_agent
from fastworkflow import planning_commands as pc
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.workflow import MissingGetParentError
from tests.test_field_sources import ROOT, _build_workflow


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()
    workflow_agent.split_settings.cache_clear()


def _write(folder: str, relative: str, text: str) -> Path:
    path = Path(folder) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _ids(folder: str, command: str):
    return RoutingRegistry.get_definition(folder).get_command_class(
        command, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)


# ------------------------------------------------------------------ 1. a Signature with no generator class

_SIGNATURE_ONLY = '''import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Say hello."""

    class Input(BaseModel):
        name: str = Field(description="who to greet")

    class Output(BaseModel):
        text: str = Field(description="the greeting")

    plain_utterances = ["say hello to Ann"]


class {generator}:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        return Signature.Output(text="hello")
'''


@pytest.mark.parametrize("misspelled", ["ResponseGenerater", "responsegenerator", "Response_Generator", "Generator"])
def test_a_signature_without_a_response_or_plan_generator_is_a_load_problem_and_not_available(tmp_path, misspelled):
    folder = _build_workflow(tmp_path, "wf_misspelled",
                             extra_commands={"Library/greet": _SIGNATURE_ONLY.replace("{generator}", misspelled)})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].startswith("Library/greet: "), problems
    assert "greet.py" in problems[0] and "has a Signature but no ResponseGenerator or PlanGenerator" in problems[0]
    assert "the class name must be exactly ResponseGenerator" in problems[0]
    with pytest.raises(Exception, match="has a Signature but no ResponseGenerator or PlanGenerator"):
        _ids(folder, "Library/greet")
    assert _ids(folder, "Library/find_book") is not None      # the rest of the workflow loads


def test_a_signature_with_no_generator_at_all_is_the_same_problem(tmp_path):
    source = _SIGNATURE_ONLY.split("class {generator}")[0]
    folder = _build_workflow(tmp_path, "wf_missing", extra_commands={"Library/greet": source})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and "has a Signature but no ResponseGenerator or PlanGenerator" in problems[0], problems


def test_the_rule_leaves_real_commands_and_non_commands_alone(tmp_path):
    """A generator with no Signature (a startup command), a helper whose name starts with an underscore and a planning
    command are all fine. A file that imports its ResponseGenerator is not (the load-check sweep: a generator is a
    class defined in the command file; it was accepted before)."""
    folder = _build_workflow(tmp_path, "wf_fine", extra_commands={
        "Library/_helper": _SIGNATURE_ONLY.split("class {generator}")[0],                       # a helper: skipped
        "Library/greet": _SIGNATURE_ONLY.replace("{generator}", "ResponseGenerator"),           # the exact name
        "Library/greet_again": ("from .greet import ResponseGenerator, Signature  # noqa: F401\n"),   # imported
        "no_signature": ("import fastworkflow\n\n\nclass ResponseGenerator:\n"
                         "    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> None:\n"
                         "        return None\n"),
    })
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].startswith("Library/greet_again: "), problems
    assert "ResponseGenerator is imported or assigned here, not defined" in problems[0]
    assert (Path(folder) / "_commands" / "startup.py").is_file()


# ------------------------------------------------------------------ 2. a context class not named Context

def test_a_context_callback_file_without_a_class_named_context_is_a_load_problem(tmp_path):
    folder = _build_workflow(tmp_path, "wf_ctx")
    path = Path(folder) / "_commands" / "Book" / "_Book.py"
    path.write_text(path.read_text().replace("class Context:", "class Contxt:"))
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1, problems
    assert problems[0].startswith("Book: ") and "_Book.py" in problems[0]
    assert "is the context class file of Book but defines no class Context" in problems[0]


def test_a_context_folder_with_no_callback_file_stays_fine(tmp_path):
    folder = _build_workflow(tmp_path, "wf_ctx_none")
    (Path(folder) / "_commands" / "Record" / "_Record.py").unlink()
    assert command_preview.load_problems(folder) == []


# ------------------------------------------------------------------ 3. _workflow.py in the wrong place

SUBJECT_HINTS = 'class Context:\n    subject_hints = "a book"\n'


def test_a_workflow_settings_file_below_the_commands_folder_is_a_load_problem(tmp_path):
    folder = _build_workflow(tmp_path, "wf_split_place")
    _write(folder, "_commands/Book/_workflow.py", SUBJECT_HINTS)
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1, problems
    assert "Book/_workflow.py" in problems[0]
    assert "the workflow settings are read only from _commands/_workflow.py" in problems[0]
    assert workflow_agent.split_settings(folder) == workflow_agent.SplitSettings()      # it is ignored
    assert _ids(folder, "Library/find_book") is not None                              # the rest loads


def test_a_split_file_below_the_commands_folder_is_the_removed_spelling(tmp_path):
    folder = _build_workflow(tmp_path, "wf_split_below")
    _write(folder, "_commands/Book/_split.py", 'subject_hints = "a book"\n')
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide, problems
    assert problems[0].startswith("_commands/Book/_split.py: _split.py is removed: move its settings")
    assert "class Context in _commands/_workflow.py" in problems[0]


def test_the_workflow_settings_file_in_the_commands_folder_is_fine(tmp_path):
    folder = _build_workflow(tmp_path, "wf_split_ok")
    _write(folder, "_commands/_workflow.py", SUBJECT_HINTS)
    assert command_preview.load_problems(folder) == []
    assert workflow_agent.split_settings(folder).subject_hints == "a book"


# ------------------------------------------------------------------ 4. an identifier pattern anchored to the whole text

def _split_with(folder: str, pattern_source: str) -> list[str]:
    _write(folder, "_commands/_workflow.py",
           f"import re\n\n\nclass Context:\n    identifier_patterns = [{pattern_source}]\n")
    return command_preview.load_problems(folder)


@pytest.mark.parametrize("pattern_source", [
    're.compile(r"^B\\d+")', 're.compile(r"B\\d+$")', 're.compile(r"^B\\d+$")',
    're.compile(r"\\AB\\d+")', 're.compile(r"B\\d+\\Z")', 're.compile(r"\\A\\d{4}\\Z")',
])
def test_an_anchored_identifier_pattern_is_a_load_problem(tmp_path, pattern_source):
    folder = _build_workflow(tmp_path, "wf_anchored")
    problems = _split_with(folder, pattern_source)
    assert len(problems) == 1, problems
    assert "_workflow.py" in problems[0] and "identifier_patterns are searched inside the message" in problems[0]
    assert "remove the ^ and $ anchors, use \\b for word boundaries" in problems[0]
    assert "d+" in problems[0] or "d{4}" in problems[0]                 # it names the pattern


@pytest.mark.parametrize("pattern_source", [
    're.compile(r"\\bB\\d+\\b")',
    're.compile(r"B\\d+\\$")',                       # an escaped dollar is a literal dollar
    're.compile(r"\\$\\d+")',
    're.compile(r"B\\d+\\\\$")',                     # an escaped backslash, then a real $: anchored (checked below)
    're.compile(r"^B\\d+$", re.MULTILINE)',          # a line boundary inside a message
    're.compile(r"(?m)^B\\d+$")',
    're.compile(r"[$^]B\\d+")',
])
def test_an_identifier_pattern_that_can_match_inside_a_message_is_fine(tmp_path, pattern_source):
    folder = _build_workflow(tmp_path, "wf_unanchored")
    problems = _split_with(folder, pattern_source)
    if pattern_source == 're.compile(r"B\\d+\\\\$")':
        assert len(problems) == 1                     # the backslash is escaped, the $ is an anchor
    else:
        assert problems == []


# ------------------------------------------------------------------ 5. a phase annotated with a command's Input alone

_INPUT_PHASE = '''from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook


class Signature:
    """Find a book."""

    class Input(BaseModel):
        query: str = Field(description="words")

    plain_utterances = ["find a book"]


class PlanGenerator:
    def find(self, input) -> {annotation}:
        return [run("find_book", query=input.query)]
'''


@pytest.mark.parametrize("annotation", ["FindBook.Input", "FindBook.Input | FindBook.Output", "FindBook.Output | FindBook.Input"])
def test_a_phase_handing_on_a_commands_input_is_a_load_problem_and_the_command_does_not_load(tmp_path, annotation):
    folder = _build_workflow(tmp_path, "wf_input_phase", extra_planning_commands={
        "Library/inspect_book": _INPUT_PHASE.replace("{annotation}", annotation)})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].startswith("Library/inspect_book: "), problems
    assert "inspect_book.py" in problems[0] and "FindBook.Input" in problems[0] or "find_book's Input" in problems[0]
    assert "a phase hands on a command's Output, a union of Outputs, or a facts model" in problems[0]
    assert "class Facts(BaseModel): asked: FindBook.Input" in problems[0]
    assert "Library/inspect_book" not in workflow_agent.load_planning_command_map(folder)


def test_a_commands_input_as_a_field_of_a_facts_model_still_loads_and_runs(tmp_path):
    source = '''from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook


class Signature:
    """Find a book."""

    class Input(BaseModel):
        query: str = Field(description="words")

    plain_utterances = ["find a book"]


class Facts(BaseModel):
    asked: FindBook.Input
    found: FindBook.Output


class PlanGenerator:
    def find(self, input) -> Facts:
        return [run("find_book", query=input.query)]

    def __result__(self, find):
        return f"asked {find.asked.query}: {find.found.total} found"
'''
    folder = _build_workflow(tmp_path, "wf_input_field", extra_planning_commands={"Library/inspect_book": source})
    assert command_preview.load_problems(folder) == []
    assert "Library/inspect_book" in workflow_agent.load_planning_command_map(folder)


# ------------------------------------------------------------------ 6. no enclosing record: one rule keyed on the root
# (load-check sweep, batch 2): with a root, a record that is not the root whose context has no _<Context>.py or no
# get_parent raises MissingGetParentError; ``return None`` is the explicit "go to the global commands" and is silent.
# The warning that once reported it is gone; these tests also check that nothing is logged in its place.

_WARNING = "is not the workflow's root and has no enclosing record"


@pytest.fixture
def warnings_logged(monkeypatch):
    wf_module = sys.modules["fastworkflow.workflow"]
    from fastworkflow import context_navigation
    seen: list[str] = []
    monkeypatch.setattr(wf_module.logger, "warning", lambda msg, *a, **k: seen.append(str(msg)))
    monkeypatch.setattr(wf_module, "_WARNED_NO_ENCLOSING_RECORD", set(), raising=False)
    context_navigation_logger = getattr(context_navigation, "logger", None)
    if context_navigation_logger is not None:
        monkeypatch.setattr(context_navigation_logger, "warning", lambda msg, *a, **k: seen.append(str(msg)))
    return seen


def _todo_app(tmp_path, *, edit_item_class=None, edit_list_class=None, root=True):
    """item -> list -> manager (root) on a temp copy of the todo workflow, the classes optionally rewritten."""
    from tests.todo_list_workflow.application.todo_item import TodoItem
    from tests.todo_list_workflow.application.todo_list import TodoList
    from tests.todo_list_workflow.application.todo_manager import TodoListManager

    source = Path(ROOT) / "tests" / "todo_list_workflow"
    folder = tmp_path / f"todo_{abs(hash((edit_item_class, edit_list_class, root)))}"
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    for relative, text in (("TodoItem/_TodoItem.py", edit_item_class), ("TodoList/_TodoList.py", edit_list_class)):
        if text is not None:
            (folder / "_commands" / relative).write_text(text)
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"warn-{tmp_path.name}-{folder.name}")
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    item = TodoItem(1, "milk")
    item.parent = todo_list
    if root:
        app.root = manager
    app.record = item
    return app, item, todo_list, manager


_NO_GET_PARENT = "class Context:\n    pass\n"
_NONE_PARENT = "class Context:\n    @classmethod\n    def get_parent(cls, record):\n        return None\n"


def _lookup_twice(app):
    from fastworkflow import context_navigation
    from fastworkflow.intent_detector import scopes_for

    for _ in range(2):
        # a class with no get_parent at all raises AttributeError from every walk (bd fix-ftdl.3: one climb)
        for walk in (lambda: context_navigation.live_context_chain(app),
                     lambda: list(context_navigation.exact_name_scopes(app)), lambda: scopes_for(app)):
            with contextlib.suppress(AttributeError):
                walk()


@pytest.mark.parametrize("edit", [_NO_GET_PARENT, _NONE_PARENT], ids=["no_get_parent", "get_parent_returns_none"])
def test_a_nested_record_with_no_enclosing_record_logs_no_warning(tmp_path, warnings_logged, edit,
                                                                setup_test_environment):
    """No get_parent is the error that names the context and what to add (the lookup stops); ``return None`` is the
    author's explicit "straight to the global commands" and is silent. Neither logs a warning."""
    app, item, *_ = _todo_app(tmp_path, edit_item_class=edit)
    _lookup_twice(app)
    assert [w for w in warnings_logged if _WARNING in w] == [], warnings_logged
    if edit == _NONE_PARENT:
        assert app.get_parent(item) is None
    else:
        with pytest.raises(MissingGetParentError):
            app.get_parent(item)


def test_a_context_with_no_callback_class_raises_too(tmp_path, warnings_logged, setup_test_environment):
    """No ``_TodoItem.py`` under a root is the same error as no ``get_parent``, naming the file to create."""
    app, item, *_ = _todo_app(tmp_path)
    os.remove(Path(app.folderpath) / "_commands" / "TodoItem" / "_TodoItem.py")
    RoutingRegistry.clear_registry()
    for _ in range(2):
        with pytest.raises(MissingGetParentError, match=r"the context TodoItem has no get_parent \(there is no "
                                                        r"_commands/TodoItem/_TodoItem\.py\).*get_parent\(cls, record\)"):
            app.get_parent(item)
    assert [w for w in warnings_logged if _WARNING in w] == [], warnings_logged


def test_a_missing_get_parent_raises_one_error_naming_the_context(tmp_path, warnings_logged, setup_test_environment):
    """Every time, with the context, its file and what to add; the error is the message, so no warning besides it."""
    app, item, *_ = _todo_app(tmp_path, edit_item_class=_NO_GET_PARENT)
    for _ in range(2):
        with pytest.raises(AttributeError, match=r"the context TodoItem has no get_parent.*"
                                                 r"_commands/TodoItem/_TodoItem\.py"):
            app.get_parent(item)
    assert [w for w in warnings_logged if _WARNING in w] == []


def test_the_warning_is_silent_when_get_parent_gives_the_enclosing_record(tmp_path, warnings_logged,
                                                                        setup_test_environment):
    app, item, todo_list, manager = _todo_app(tmp_path)
    _lookup_twice(app)
    assert app.get_parent(item) is todo_list and app.get_parent(todo_list) is manager
    assert [w for w in warnings_logged if _WARNING in w] == []


def test_the_warning_is_silent_for_the_root_itself(tmp_path, warnings_logged, setup_test_environment):
    app, _, _, manager = _todo_app(tmp_path)
    app.record = manager
    _lookup_twice(app)
    assert app.get_parent(manager) is manager
    assert [w for w in warnings_logged if _WARNING in w] == []


def test_the_warning_is_silent_for_a_workflow_with_no_root(tmp_path, warnings_logged, setup_test_environment):
    app, item, *_ = _todo_app(tmp_path, edit_item_class=_NONE_PARENT, root=False)
    assert app.root is None
    _lookup_twice(app)
    assert [w for w in warnings_logged if _WARNING in w] == []


def test_the_warning_does_not_change_routing(tmp_path, warnings_logged, setup_test_environment):
    from fastworkflow import context_navigation
    app, *_ = _todo_app(tmp_path, edit_item_class=_NONE_PARENT)
    assert context_navigation.live_context_chain(app) == ("TodoItem", "*")
