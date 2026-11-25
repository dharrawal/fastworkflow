"""``fastworkflow build`` writes no context_inheritance_model.json and never edits a hand-written ``_<Context>.py``
(bd fix-qjne.37.1). Runs the build's own generation step on a copy of tests/example_workflow's application."""
import argparse
import shutil
from pathlib import Path

import pytest

from fastworkflow.build.__main__ import run_command_generation
from fastworkflow.command_context_model import CommandContextModel

APPLICATION = Path(__file__).resolve().parents[1] / "example_workflow" / "application"


def _build(workflow: Path):
    if not (workflow / "application").exists():
        shutil.copytree(APPLICATION, workflow / "application", ignore=shutil.ignore_patterns("__pycache__"))
    (workflow / "_commands").mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(app_dir=str(workflow / "application"), workflow_folderpath=str(workflow),
                              overwrite=False, stub_commands=None, no_startup=True)
    return run_command_generation(args)


@pytest.mark.integration
def test_a_built_workflow_has_no_json_and_its_declarations_resolve_as_the_json_did(tmp_path, capsys):
    workflow = tmp_path / "wf"
    classes, model = _build(workflow)
    out = capsys.readouterr().out

    assert not list(workflow.rglob("*.json"))
    # The JSON build wrote before: every class with its analysed bases
    old_json = {name: {"base": [b for b in info.bases if b in classes]} for name, info in classes.items()}
    assert model == old_json
    assert 'TodoList: inherits = ["TodoItem"]  (_commands/TodoList/_TodoList.py: written, a new file)' in out

    # The command set of every context, through the declarations, is the one the old JSON gave: the context's own
    # commands and those of every context it inherits, transitively
    loaded = CommandContextModel.load(workflow)
    own = {name: {c.split("/")[-1] for c in loaded._command_contexts.get(name, {}).get("/", [])} for name in classes}

    def through_json(name):
        names = set(own[name])
        for base in old_json[name]["base"]:
            names |= through_json(base)
        return names

    assert CommandContextModel.declared_bases(workflow) == {"TodoList": ["TodoItem"]}
    for name in classes:
        assert {c.split("/")[-1] for c in loaded.commands(name)} == through_json(name), name


@pytest.mark.integration
def test_a_second_build_leaves_every_existing_context_file_byte_identical(tmp_path, capsys):
    workflow = tmp_path / "wf"
    _build(workflow)
    hand = workflow / "_commands" / "TodoList" / "_TodoList.py"
    hand.write_bytes(b'# my own\nclass Context:\n    inherits = ["TodoItem", "TodoListManager"]\n')
    files = {path: path.read_bytes() for path in (workflow / "_commands").glob("*/_*.py")}
    capsys.readouterr()

    _build(workflow)
    out = capsys.readouterr().out

    assert {path: path.read_bytes() for path in (workflow / "_commands").glob("*/_*.py")} == files
    assert ('TodoList: inherits = ["TodoItem"]  (_commands/TodoList/_TodoList.py declares '
            'inherits = ["TodoItem", "TodoListManager"]: differs, not changed)') in out
    assert not list(workflow.rglob("*.json"))
