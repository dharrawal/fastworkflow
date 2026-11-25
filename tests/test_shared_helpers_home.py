"""A workflow's helper modules have one home, the package ``_commands/_shared/`` (bd fix-qjne.33).

* Command files import it the way they imported an underscore helper before: ``from ._shared.x import Y`` from a root
  command, ``from .._shared.x import Y`` from a context folder. One module object serves both.
* The framework never reads ``_shared/`` as commands, contexts, previews or training inputs.
* Any other ``.py`` file whose name starts with ``_``, directly in ``_commands/`` or in a context folder (other than
  ``_workflow.py`` and the folder's own ``_<Context>.py``), and any other underscore folder of ``_commands/``, is one
  workflow-wide load problem naming ``_commands/_shared/``; a misspelt settings or context class file gets a
  did-you-mean. ``__init__.py`` and ``__pycache__`` are allowed everywhere.

Each case runs on a TEMP COPY of the library example, whose ``Library`` commands import ``_shared/wording.py``. No
model is called.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import command_preview, workflow_agent
from fastworkflow import planning_commands as pc
from fastworkflow.command_context_model import context_folders
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.planning import run_code

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")

#: An execution command that words a number with the shared helper; {dots} is "." at the root, ".." in a context.
COPIES_COMMAND = '''import fastworkflow
from pydantic import BaseModel, Field

from {dots}_shared.wording import copies


class Signature:
    """Say a number of copies in words."""

    class Input(BaseModel):
        number: int = Field(description="how many copies")

    class Output(BaseModel):
        text: str = Field(description="the copies in words")

    plain_utterances = ["how many copies is 3 in words"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(text=copies(input.number))
'''

#: A file that would be a planning command anywhere else in _commands/.
PLANNING_LOOKALIKE = '''class Signature:
    """Not a command: it lives in the helper package."""


class PlanGenerator:
    def steps(self):
        return ["say hello"]
'''


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()
    workflow_agent.split_settings.cache_clear()
    yield
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()


@pytest.fixture
def library(tmp_path) -> Path:
    folder = tmp_path / "library"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    return folder


def _problems(folder: Path) -> list:
    return command_preview.load_problems(str(folder))


def _one_workflow_wide_problem(folder: Path, label: str) -> str:
    problems = _problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide and problems[0].startswith(f"{label}: "), problems
    return str(problems[0])


def test_the_shipped_library_keeps_its_helper_in_shared_and_loads(library):
    commands = Path(LIBRARY) / "_commands"
    assert (commands / "_shared" / "__init__.py").is_file() and (commands / "_shared" / "wording.py").is_file()
    assert not list(commands.glob("**/_wording.py"))
    assert _problems(library) == []      # its copy: a load writes the routing cache beside the commands


def test_a_root_command_and_a_context_command_import_the_one_shared_module(library):
    commands = library / "_commands"
    (commands / "copies_in_words.py").write_text(COPIES_COMMAND.format(dots="."), encoding="utf-8")
    (commands / "Library" / "copies_here.py").write_text(COPIES_COMMAND.format(dots=".."), encoding="utf-8")
    assert _problems(library) == []

    assert run_code(str(library), "copies_in_words", {"number": 3}).output.text == "3 copies"
    assert run_code(str(library), "Library/copies_here", {"number": 1}).output.text == "1 copy"
    # a context's planning command that imports it the same way (the library's book_by_title) runs too
    assert "Dune" in run_code(str(library), "book_by_title", {"title": "Dune"}).text

    wording = os.path.realpath(commands / "_shared" / "wording.py")
    shared = [name for name, module in sys.modules.items()
              if name.endswith("._commands._shared.wording")
              and os.path.realpath(getattr(module, "__file__", "") or "") == wording]
    assert len(shared) == 1, shared      # one module object, whoever imports it


def test_shared_is_no_context_no_command_no_preview_and_no_training_input(library):
    from fastworkflow.train.selective_training import compute_planning_fingerprints

    (library / "_commands" / "_shared" / "lookalike.py").write_text(PLANNING_LOOKALIKE, encoding="utf-8")
    assert _problems(library) == []
    definition = RoutingRegistry.get_definition(str(library))
    assert "_shared" not in definition.contexts
    assert "_shared" not in context_folders(library)
    directory = definition.command_directory
    assert not [name for name in (*directory.map_command_2_metadata, *directory.map_context_2_metadata)
                if "shared" in name or "lookalike" in name or "wording" in name]
    assert not [name for name, _ in workflow_agent._planning_command_files(str(library / "_commands"))
                if "shared" in name or "lookalike" in name]
    assert not [name for name in command_preview.render_previews(str(library)) if "lookalike" in name]
    assert not [name for name in compute_planning_fingerprints(str(library)) if "lookalike" in name]


def test_a_stray_helper_in_commands_is_one_workflow_wide_problem_naming_shared(library):
    (library / "_commands" / "_helper.py").write_text("X = 1\n", encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/_helper.py")
    assert "a file whose name starts with an underscore is no longer ignored" in problem
    assert ("A helper module belongs in the package _commands/_shared/: move it to _commands/_shared/helper.py and "
            "import it as from ._shared.helper import ...") in problem
    assert "Did you mean" not in problem


def test_a_stray_helper_in_a_context_folder_is_one_workflow_wide_problem_naming_shared(library):
    (library / "_commands" / "Library" / "_helper.py").write_text("X = 1\n", encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/Library/_helper.py")
    assert "move it to _commands/_shared/helper.py and import it as from .._shared.helper import ..." in problem


def test_a_misspelt_settings_file_gets_a_did_you_mean(library):
    (library / "_commands" / "_worklfow.py").write_text("class Context:\n    split_examples = []\n",
                                                        encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/_worklfow.py")
    assert "Did you mean _commands/_workflow.py?" in problem


def test_a_misspelt_context_class_file_gets_a_did_you_mean(library):
    (library / "_commands" / "Book" / "_Bok.py").write_text("X = 1\n", encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/Book/_Bok.py")
    assert "Did you mean _commands/Book/_Book.py?" in problem
    # one at the root, named like a context's file, is pointed at that context's folder
    (library / "_commands" / "Book" / "_Bok.py").unlink()
    (library / "_commands" / "_Member.py").write_text("X = 1\n", encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/_Member.py")
    assert "Did you mean _commands/Member/_Member.py?" in problem


def test_another_underscore_folder_of_commands_is_one_workflow_wide_problem(library):
    (library / "_commands" / "_old").mkdir()
    (library / "_commands" / "_old" / "util.py").write_text("X = 1\n", encoding="utf-8")
    problem = _one_workflow_wide_problem(library, "_commands/_old/")
    assert "helper modules belong in the package _commands/_shared/" in problem


def test_init_files_and_pycache_are_allowed_everywhere(library):
    commands = library / "_commands"
    for folder in (commands, commands / "Library", commands / "_shared"):
        (folder / "__init__.py").touch()
        (folder / "__pycache__").mkdir(exist_ok=True)
        (folder / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"")
    assert _problems(library) == []
