"""A context's class declares what it inherits and what contains its records (bd fix-qjne.36).

``inherits = [...]`` and ``parents = [...]`` on the ``class Context`` of ``_commands/<Context>/_<Context>.py`` replace
``_commands/context_inheritance_model.json``, which is a removed spelling. Every case builds a small workflow of its
own in a temp folder; nothing calls a model.

The merged model the declarations resolve to is compared with ``_OLD_LOADER``: what the loader that read the JSON file
returned for the same workflow written with the equivalent file (computed with that loader before it was replaced).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.planning import command_preview
from fastworkflow import workflow_agent
from fastworkflow.planning import planning_commands as pc
from fastworkflow.core.command_context_model import CommandContextModel, CommandContextModelValidationError
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.migrate_context_declarations import migrate
from fastworkflow.core.workflow import MissingGetParentError
from tests.test_one_rule_for_author_mistakes import _exec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(ROOT / "fastworkflow" / "examples" / "fastworkflow.env")))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()
    workflow_agent.split_settings.cache_clear()
    yield
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()


# Records: a record's context is its class name.
class Shelf:
    pass


class Box:
    def __init__(self, parent=None):
        self.parent = parent


class Lid(Box):
    pass


_GET_PARENT = '''
    @classmethod
    def get_parent(cls, record):
        return record.parent
'''


def _context(*lines: str, get_parent: bool = False) -> str:
    body = "".join(f"    {line}\n" for line in lines) or ("" if get_parent else "    pass\n")
    return "class Context:\n" + body + (_GET_PARENT if get_parent else "")


def _workflow(tmp_path: Path, contexts: dict[str, str | None], commands: dict[str, str] | None = None,
              name: str = "wf") -> str:
    """A workflow with a ``Shelf/look`` command, one folder per context, its class file when the source is given,
    and *commands* (``"Ctx/name"`` or ``"name"``: source)."""
    folder = tmp_path / name
    commands_dir = folder / "_commands"
    commands_dir.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    for context, source in contexts.items():
        (commands_dir / context).mkdir()
        if source is not None:
            (commands_dir / context / f"_{context}.py").write_text(source)
    for command, source in {"Shelf/look": _exec(), **(commands or {})}.items():
        (commands_dir / f"{command}.py").write_text(source)
    return str(folder)


def _problems(folder: str) -> list[str]:
    return [str(problem) for problem in command_preview.load_problems(folder)]


def _model(folder: str) -> dict:
    model = CommandContextModel.load(folder)
    return {name: model.commands(name) for name in sorted(model._command_contexts)}


# ------------------------------------------------------------------ the declarations resolve as the file did

#: The old loader's merged model for the workflow of ``test_declarations_resolve_to_the_model_the_file_gave``,
#: written with ``{"Box": {"base": ["Shelf"]}, "Lid": {"base": ["Box", "*"]}, "Bin": {"base": ["Box"]}}``.
_OLD_LOADER = {
    "commands": {
        "*": ["greet", "wildcard"],
        "Bin": ["Box/open", "Shelf/look"],
        "Box": ["Box/open", "Shelf/look"],
        "IntentDetection": ["IntentDetection/go_up", "IntentDetection/reset_context", "IntentDetection/what_can_i_do",
                            "IntentDetection/what_is_current_context"],
        "Lid": ["Box/open", "Lid/look", "greet", "wildcard"],
        "Shelf": ["Shelf/look"],
    },
    "bases": {"*": [], "Bin": ["Box", "Shelf"], "Box": ["Shelf"], "IntentDetection": [],
              "Lid": ["Box", "Shelf", "*"], "Shelf": []},
    "static_bases": {"Bin": ["Box", "Shelf"], "Box": ["Shelf"], "Lid": ["Box", "Shelf", "*"]},
}

_RESOLVED = {"Shelf": _context(), "Box": _context('inherits = ["Shelf"]'),
             "Lid": _context('inherits = ["Box", "*"]'), "Bin": _context('inherits = ["Box"]')}
_RESOLVED_COMMANDS = {"Box/open": _exec(), "Lid/look": _exec(), "greet": _exec()}


def test_declarations_resolve_to_the_model_the_file_gave(tmp_path):
    folder = _workflow(tmp_path, _RESOLVED, _RESOLVED_COMMANDS)
    model = CommandContextModel.load(folder)
    assert {"commands": _model(folder),
            "bases": {name: model.ordered_base_contexts(name) for name in sorted(model._command_contexts)},
            "static_bases": {k: list(v) for k, v in CommandContextModel.load_base_contexts(folder).items()},
            } == _OLD_LOADER
    assert CommandContextModel.declared_bases(folder) == {"Box": ["Shelf"], "Lid": ["Box", "*"], "Bin": ["Box"]}
    assert _problems(folder) == []


def test_the_migration_helper_writes_the_same_model(tmp_path):
    """The helper turns the equivalent file into the declarations: the same merged model, the file deleted."""
    folder = _workflow(tmp_path, {"Shelf": None, "Box": "class Context:\n    \"\"\"A box.\"\"\"\n",
                                  "Lid": _context(), "Bin": None}, _RESOLVED_COMMANDS)
    path = Path(folder) / "_commands" / "context_inheritance_model.json"
    path.write_text(json.dumps({"Box": {"base": ["Shelf"]}, "Lid": {"base": ["Box", "*"]},
                                "Bin": {"base": ["Box"]}, "Shelf": {"base": []}}))
    report = migrate(folder)
    assert not path.exists() and report.created == ["Bin"]
    assert report.written == {"Bin": ["Box"], "Box": ["Shelf"], "Lid": ["Box", "*"]}
    assert _model(folder) == _OLD_LOADER["commands"]
    assert _problems(folder) == []


def test_the_migration_helper_lists_the_contexts_that_need_parents(tmp_path):
    folder = _workflow(tmp_path, {"Shelf": None, "Box": _context(get_parent=True)})
    (Path(folder) / "_commands" / "context_inheritance_model.json").write_text("{}")
    assert migrate(folder).needs_parents == {"Box": "not annotated"}


# ------------------------------------------------------------------ the removed file


def test_the_inheritance_file_is_a_removed_spelling(tmp_path):
    folder = _workflow(tmp_path, {"Shelf": None})
    (Path(folder) / "_commands" / "context_inheritance_model.json").write_text("{}")
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide, problems
    assert problems[0].startswith("_commands/context_inheritance_model.json: ")
    assert "is removed and not read: declare each context's bases as inherits = " in problems[0]
    assert "fastworkflow migrate-contexts" in problems[0]
    with pytest.raises(CommandContextModelValidationError, match="is removed and not read"):
        CommandContextModel.load(folder)


# ------------------------------------------------------------------ the load checks of inherits and parents


@pytest.mark.parametrize("contexts, expected", [
    ({"Box": _context('inherits = ["Shelv"]')}, "inherits names 'Shelv', which is no context; did you mean 'Shelf'?"),
    ({"Box": _context('inherits = ["Box"]')}, "inherits names 'Box', the context itself"),
    ({"Box": _context('inherits = ["Shelf", "Shelf"]')}, "inherits names 'Shelf' twice"),
    ({"Box": _context("inherits = []")}, "inherits = [] declares nothing: remove it"),
    ({"Box": _context('inherits = "Shelf"')}, "inherits must be a list of context names written as string literals"),
    ({"Box": _context('inherits = ["Lid"]'), "Lid": _context('inherits = ["Box"]')}, "Inheritance cycle detected"),
    ({"Box": _context('parents = ["Shelv"]', get_parent=True)},
     "parents names 'Shelv', which is no context; did you mean 'Shelf'?"),
    ({"Box": _context('parents = ["*"]', get_parent=True)}, "parents names '*', which is no context"),
    ({"Box": _context('parents = ["Shelf", "Shelf"]', get_parent=True)}, "parents names 'Shelf' twice"),
    ({"Box": None}, "the context folder 'Box' holds no command file and inherits none"),
])
def test_each_mistake_in_the_declarations_is_one_workflow_wide_line_naming_the_file(tmp_path, contexts, expected):
    folder = _workflow(tmp_path, {"Shelf": None, **contexts})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide, problems
    assert expected in problems[0], problems
    assert "_commands/Box" in problems[0], problems


@pytest.mark.parametrize("box, expected", [
    (_context('parents = ["Shelf"]'), "parents is declared but there is no get_parent"),
    (_context(get_parent=True), "get_parent is written but no parents is declared"),
    (_context("occupiable = False", 'parents = ["Shelf"]', get_parent=True),
     "parents is declared on a context declared occupiable = False"),
])
def test_parents_go_with_get_parent(tmp_path, box, expected):
    folder = _workflow(tmp_path, {"Shelf": None, "Box": box}, {"Box/open": _exec()})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide, problems
    assert problems[0].startswith("_commands/Box/_Box.py: ") and expected in problems[0], problems


def test_records_nested_in_their_own_kind_and_several_top_level_contexts_load(tmp_path):
    """A box can sit in a box or on a shelf; Shelf and Bin are both top level (no parents, no get_parent)."""
    folder = _workflow(tmp_path, {"Shelf": None, "Bin": None,
                                  "Box": _context('parents = ["Box", "Shelf"]', get_parent=True)},
                       {"Box/open": _exec(), "Bin/empty": _exec()})
    assert _problems(folder) == []
    assert CommandContextModel.declared_parents(folder) == {"Box": ["Box", "Shelf"]}


# ------------------------------------------------------------------ @descends_to is checked against parents


@pytest.mark.parametrize("box, entered_from, ok", [
    (_context('parents = ["Shelf"]', get_parent=True), "Shelf", True),
    (_context('parents = ["Bin"]', get_parent=True), "Shelf", False),
    (None, "Shelf", False),                                        # Box is top level: it sits in nothing
    (_context('parents = ["Shelf"]', get_parent=True), "Lid", True),  # Lid inherits Shelf
])
def test_a_descends_to_target_must_list_the_commands_context_among_its_parents(tmp_path, box, entered_from, ok):
    contexts = {"Shelf": None, "Bin": None, "Box": box, "Lid": _context('inherits = ["Shelf"]')}
    commands = {"Bin/empty": _exec(), "Box/open": _exec(),
                f"{entered_from}/open_box": _exec(decorators='@descends_to("Box")')}
    problems = _problems(_workflow(tmp_path, contexts, commands))
    if ok:
        assert problems == []
        return
    assert len(problems) == 1 and problems[0].startswith(f"{entered_from}/open_box: "), problems
    assert (f'@descends_to("Box") enters a context whose records do not sit in a {entered_from}' in problems[0]
            and f'Add "{entered_from}" to the parents of Box' in problems[0]), problems


def test_a_global_command_may_enter_any_context(tmp_path):
    """A global command runs on the root, whatever its context, so its target is not checked."""
    folder = _workflow(tmp_path, {"Shelf": None, "Box": None},
                       {"Box/open": _exec(), "open_box": _exec(decorators='@descends_to("Box")')})
    assert _problems(folder) == []


# ------------------------------------------------------------------ the climb checks what get_parent returns


def test_get_parent_must_return_a_record_of_a_declared_parent(tmp_path):
    folder = _workflow(tmp_path, {"Shelf": None, "Lid": _context('inherits = ["Box"]'),
                                  "Box": _context('parents = ["Shelf", "Box"]', get_parent=True)},
                       {"Box/open": _exec()})
    assert _problems(folder) == []
    workflow = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str="ctx-decl-climb")
    shelf = Shelf()
    workflow.root = shelf
    assert workflow.get_parent(Box(parent=shelf)) is shelf
    outer = Box(parent=shelf)
    assert workflow.get_parent(Box(parent=outer)) is outer                # a box in a box
    lid = Lid(parent=shelf)
    assert workflow.get_parent(Box(parent=lid)) is lid                   # a Lid is a Box
    assert workflow.get_parent(Box(parent=None)) is None                  # None: the global commands
    with pytest.raises(MissingGetParentError) as error:
        workflow.get_parent(Box(parent=object()))
    assert str(error.value).startswith(
        "the get_parent of context Box returned a object record, but Box declares parents = ['Shelf', 'Box']")
    assert "add 'object' to parents" in str(error.value)
