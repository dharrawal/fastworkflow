"""``fastworkflow build`` derives each context's ``inherits`` and never writes context_inheritance_model.json
(bd fix-qjne.37.1).

These tests replaced the ones of the retired ``InheritanceBlockRegenerator``, which merged the derived bases into the
JSON file: each keeps its intent (bases derived from the classes, a hand-written declaration survives, a second run
changes nothing, the result is said) against the declarations on ``class Context``.
"""
from pathlib import Path

import pytest

from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.build.context_model_generator import (
    declare_inherits, derive_inherits, existing_context_files, generate_context_model)
from fastworkflow.core.command_context_model import REMOVED_INHERITANCE_FILE


@pytest.fixture
def mock_classes():
    return {
        "TodoList": ClassInfo(name="TodoList", module_path="app/models/todo_list.py", bases=["BaseModel"]),
        "TodoItem": ClassInfo(name="TodoItem", module_path="app/models/todo_item.py", bases=["BaseModel"]),
        "User": ClassInfo(name="User", module_path="app/models/user.py", bases=["BaseModel", "TodoList", "object"]),
        "BaseModel": ClassInfo(name="BaseModel", module_path="app/models/base.py", bases=[]),
    }


def _workflow(tmp_path, folders=("Book", "Member", "Record")) -> Path:
    for folder in folders:
        (tmp_path / "_commands" / folder).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_bases_are_derived_from_the_analysed_classes_only(mock_classes):
    """A base that is not an analysed class (``object``) is no context."""
    assert derive_inherits(mock_classes) == {"BaseModel": [], "TodoItem": ["BaseModel"],
                                             "TodoList": ["BaseModel"], "User": ["BaseModel", "TodoList"]}


def test_a_new_context_file_gets_the_derived_line_and_no_json_is_written(tmp_path, capsys, mock_classes):
    folder = _workflow(tmp_path, folders=mock_classes)
    model = generate_context_model(mock_classes, str(folder))
    assert model["User"] == {"base": ["BaseModel", "TodoList"]} and model["BaseModel"] == {"base": []}
    assert (folder / "_commands/User/_User.py").read_text() == (
        'class Context:\n    inherits = ["BaseModel", "TodoList"]\n')
    assert not (folder / "_commands/BaseModel/_BaseModel.py").exists()   # nothing to declare
    assert not list(folder.rglob("*.json"))
    out = capsys.readouterr().out
    assert ('User: inherits = ["BaseModel", "TodoList"]  (_commands/User/_User.py: written, a new file)' in out)
    assert "BaseModel: no inherits" in out


def test_a_hand_written_declaration_is_never_edited_and_the_difference_is_said(tmp_path, capsys):
    """Book's Python class has no Record base, but the author declared one: build keeps the file byte for byte and
    prints the line it derives beside what the file declares."""
    folder = _workflow(tmp_path)
    book = folder / "_commands/Book/_Book.py"
    book.write_bytes(b'"""Book."""\nclass Context:\n    inherits = ["Record"]   # by hand\n')
    before = book.read_bytes()
    classes = {"Book": ClassInfo("Book", "book.py", bases=["Item"]), "Item": ClassInfo("Item", "item.py")}
    report = declare_inherits(folder, derive_inherits(classes))
    assert book.read_bytes() == before
    assert report.differs() == {"Book": (["Item"], ["Record"])}
    assert ('Book: inherits = ["Item"]  (_commands/Book/_Book.py declares inherits = ["Record"]: differs, '
            'not changed)') in report.lines()


def test_a_hand_written_file_with_no_declaration_is_not_edited_either(tmp_path):
    folder = _workflow(tmp_path)
    member = folder / "_commands/Member/_Member.py"
    member.write_text("class Context:\n    @classmethod\n    def get_parent(cls, record):\n        return None\n")
    before = member.read_bytes()
    report = declare_inherits(folder, {"Member": ["Record"], "Record": []})
    assert member.read_bytes() == before
    assert report.differs() == {"Member": (["Record"], None)}
    assert report.needs_parents == {"Member": "not annotated"}
    assert any(line.startswith("Member: has get_parent, write parents = [...]") for line in report.lines())


def test_with_no_classes_nothing_is_declared(tmp_path):
    folder = _workflow(tmp_path)
    report = declare_inherits(folder, derive_inherits({}))
    assert report.lines() == [] and not list((folder / "_commands").rglob("_*.py"))


def test_a_second_build_changes_nothing(tmp_path, mock_classes):
    """The file the first build created is hand-written for the second: it is read, not rewritten."""
    folder = _workflow(tmp_path, folders=mock_classes)
    generate_context_model(mock_classes, str(folder))
    files = {path: path.read_bytes() for path in existing_context_files(folder)}
    report = declare_inherits(folder, derive_inherits(mock_classes))
    assert report.written == [] and report.differs() == {}
    assert {path: path.read_bytes() for path in existing_context_files(folder)} == files
    assert 'User: inherits = ["BaseModel", "TodoList"]  (_commands/User/_User.py: declared)' in report.lines()


def test_a_leftover_inheritance_file_is_named_not_read_or_deleted(tmp_path):
    folder = _workflow(tmp_path)
    leftover = folder / "_commands" / REMOVED_INHERITANCE_FILE
    leftover.write_text('{"Book": {"base": ["Record"]}}')
    report = declare_inherits(folder, {"Book": [], "Record": []})
    assert leftover.read_text() == '{"Book": {"base": ["Record"]}}'
    assert report.lines()[-1].startswith(f"_commands/{REMOVED_INHERITANCE_FILE} is removed and not read")
