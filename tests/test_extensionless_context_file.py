"""The context class file is ``_<Folder>.py`` only (fix-apjz.5).

An extensionless ``_<Folder>`` used to be accepted as a second spelling. It shipped, so a workflow that has one
fails to load with a problem that names the replacement. A copy of the library example, never trained.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_preview import load_problems

LIBRARY = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "library"


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def _copy(tmp_path, name):
    target = tmp_path / name
    shutil.copytree(LIBRARY, target, ignore=shutil.ignore_patterns("___command_info", "___convo_info", "__pycache__"))
    return target


def test_the_library_example_loads_with_its_py_context_files(tmp_path):
    folder = _copy(tmp_path, "ok")
    assert CommandDirectory.load(str(folder)).get_context_metadata("Book").context_module_path.endswith("_Book.py")
    assert load_problems(str(folder)) == []


def test_an_extensionless_context_file_fails_the_load_and_names_the_py_replacement(tmp_path):
    folder = _copy(tmp_path, "extensionless")
    (folder / "_commands" / "Book" / "_Book.py").rename(folder / "_commands" / "Book" / "_Book")

    with pytest.raises(ValueError, match=r"_Book.*_Book\.py"):
        CommandDirectory.load(str(folder))
    problems = load_problems(str(folder))
    assert problems and any("_Book" in line and "_Book.py" in line for line in problems), problems
