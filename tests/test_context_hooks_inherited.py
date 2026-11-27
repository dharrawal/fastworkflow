"""A context class may inherit its serialization hooks from a plain base class in a helper module.

A temp copy of the bundled ``library`` example has its hooks moved, unchanged in behaviour, onto one base class
in ``_commands/_shared/checkpoint.py`` (the helper package, which the scanner never reads); each ``Context`` of the
copy then inherits it. The same flow is driven through ``project_command_contexts`` and
``restore_command_contexts`` on the untouched copy and on the inheriting copy, and the two must agree. Nothing here
calls a model.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.session import serialization_hooks
from fastworkflow.session.serialization_hooks import ProjectionOutcome
from fastworkflow.session.state_serialization import validate_state
from tests.test_workflow_serialization_hooks import (  # noqa: F401  (fixtures and helpers)
    EXAMPLES, initialized_fastworkflow, make_workflow)

LIBRARY = EXAMPLES / "library"

HELPER = '''\
from typing import Any, Optional

import fastworkflow

from ...application.library import Library


class CheckpointHooks:
    """The four hooks, written once. A context class inherits them: ``class Context(CheckpointHooks)``."""

    state_version = 1

    @classmethod
    def get_state(cls, record) -> dict[str, Any]:
        library = record if type(record).__name__ == "Library" else record.library
        return {"library": library.to_state_dict(), "uid": None if record is library else record.uid}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow, *, state_version: int):
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        library = Library.from_state_dict(state["library"])
        return library if state["uid"] is None else _find(library, state["uid"])

    @classmethod
    def get_locator(cls, record) -> str:
        kind = "book" if type(record).__name__ == "Book" else "member"
        return f"{kind}:{record.uid}"

    @classmethod
    def find_by_locator(cls, anchor, locator: str) -> Optional[Any]:
        kind, _, uid = locator.partition(":")
        return (anchor.books if kind == "book" else anchor.members).get(uid)


def _find(library, uid):
    return library.books.get(uid) or library.members.get(uid)
'''

LIBRARY_CLASS = '''\
from .._shared.checkpoint import CheckpointHooks
from ...application.library import Library


class Context(CheckpointHooks):
    @classmethod
    def get_displayname(cls, record: Library) -> str:
        return "Library"
'''

RECORD_CLASS = '''\
from .._shared.checkpoint import CheckpointHooks


class Context(CheckpointHooks):
    inherits = ["Record"]
    parents = ["Library"]

    @classmethod
    def get_parent(cls, record):
        return record.library

    @classmethod
    def get_displayname(cls, record) -> str:
        return f"{type(record).__name__}: {record.uid}"
@@EXTRA@@'''

MEMBER_OWN_STATE = '''
    @classmethod
    def get_state(cls, record):
        state = super().get_state(record)
        state["written_by"] = "Member"
        return state
'''


def _copy(tmp_path: Path, name: str, *, inherit: bool) -> Path:
    folder = tmp_path / name / "library"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___convo_info", "__pycache__"))
    if inherit:
        commands = folder / "_commands"
        (commands / "_shared").mkdir(exist_ok=True)   # the library keeps its own helper there
        (commands / "_shared" / "checkpoint.py").write_text(HELPER)
        (commands / "Library" / "_Library.py").write_text(LIBRARY_CLASS)
        (commands / "Book" / "_Book.py").write_text(RECORD_CLASS.replace("@@EXTRA@@", ""))
        (commands / "Member" / "_Member.py").write_text(RECORD_CLASS.replace("@@EXTRA@@", MEMBER_OWN_STATE))
    return folder


@pytest.fixture
def plain(tmp_path):
    return _copy(tmp_path, "plain", inherit=False)


@pytest.fixture
def inheriting(tmp_path):
    return _copy(tmp_path, "inheriting", inherit=True)


def _library():
    from fastworkflow.examples.library.application.library import Library
    library = Library()
    library.books["b1"].notes.append("added in the session")
    library.members["m2"].notes.append("moved to the Riverside branch")
    return library


def _project(folder: Path, *, anchor: str):
    """Project a workflow standing on the library root or on a record (``books:b3``), or with no root at all."""
    workflow, library = make_workflow(folder), _library()
    if anchor == "root":
        workflow.root = library
        workflow.record = library
    elif anchor == "book_in_root":
        workflow.root = library
        workflow.record = library.books["b3"]
    else:  # a record with no root context: the anchor is the current record
        kind, uid = anchor.split(":")
        workflow.record = getattr(library, kind)[uid]
    projection = serialization_hooks.project_command_contexts(workflow)
    assert projection.outcome is ProjectionOutcome.SERIALIZABLE, projection.reason
    record = projection.as_record()
    validate_state({"context": record})
    return projection, record


def _restore(folder: Path, record):
    restored = make_workflow(folder)
    assert serialization_hooks.restore_command_contexts(restored, record)
    return restored


def _library_dict(state: dict) -> dict:
    return state["library"] if "library" in state else state


# (a) it loads, and nothing is reported as lacking a hook

def test_a_the_inheriting_workflow_loads_and_no_context_lacks_a_hook(initialized_fastworkflow, inheriting, plain):
    from fastworkflow.planning import command_preview
    from fastworkflow.run_fastapi_mcp import checkpoint

    assert command_preview.load_problems(str(inheriting)) == []
    assert checkpoint.contexts_without_state_hook(str(inheriting)) == []
    assert checkpoint.startup_pinning_warning(str(inheriting)) is None
    assert checkpoint.contexts_without_state_hook(str(plain)) == []


def test_a_the_check_is_real_a_context_with_no_hooks_and_no_base_is_listed(initialized_fastworkflow, inheriting):
    from fastworkflow.run_fastapi_mcp import checkpoint

    (inheriting / "_commands" / "Member" / "_Member.py").write_text("class Context:\n    pass\n")
    assert checkpoint.contexts_without_state_hook(str(inheriting)) == ["Member"]


# (b) the round trip agrees with the hooks written on each class

@pytest.mark.parametrize("anchor", ["root", "book_in_root", "books:b4", "members:m3"])
def test_b_round_trip_matches_the_class_by_class_hooks(initialized_fastworkflow, plain, inheriting, anchor):
    expected_projection, expected_record = _project(plain, anchor=anchor)
    projection, record = _project(inheriting, anchor=anchor)

    for field in ("context_name", "anchor_is_root", "current_locator", "response_locator", "state_version"):
        assert getattr(projection, field) == getattr(expected_projection, field), field
    assert _library_dict(record["state"]) == _library_dict(expected_record["state"])

    expected = _restore(plain, expected_record)
    restored = _restore(inheriting, record)

    def facts(workflow):
        anchor_object = workflow.root if workflow.root is not None else workflow.record
        node = workflow._current_command_context
        return (type(anchor_object).__name__, type(node).__name__, getattr(node, "uid", None),
                workflow.root is not None, node is not anchor_object)

    assert facts(restored) == facts(expected)
    library = restored.root if restored.root is not None else restored._current_command_context.library
    expected_library = (expected.root if expected.root is not None
                        else expected._current_command_context.library)
    assert library.to_state_dict() == expected_library.to_state_dict()
    node = restored._current_command_context
    if restored.root is not None and node is not restored.root:
        assert node is restored.root.books[node.uid], "the current record is a node of the restored root"
    if restored.root is None:
        assert node.library.books.get(node.uid, node) is node or node.library.members[node.uid] is node


# (c) the helper is neither a command nor a context

def test_c_the_helper_module_is_not_registered(initialized_fastworkflow, inheriting, plain):
    from fastworkflow.command_directory import get_cached_command_directory

    for folder in (plain, inheriting):
        directory = get_cached_command_directory(str(folder))
        assert not [n for n in directory.map_command_2_metadata if "checkpoint" in n.lower()]
        assert not [n for n in directory.map_context_2_metadata if "checkpoint" in n.lower()]
        model = fastworkflow.CommandContextModel.load(str(folder))
        assert not [n for n in model._command_contexts if "checkpoint" in n.lower()]
    assert (inheriting / "_commands" / "_shared" / "checkpoint.py").is_file()


# (d) a class that overrides one inherited hook gets its own

def test_d_an_override_wins_and_calls_the_inherited_hook_through_super(initialized_fastworkflow, inheriting):
    context_model = fastworkflow.CommandContextModel.load(str(inheriting))
    kind = fastworkflow.ModuleType.CONTEXT_CLASS
    book, member, library = (context_model.get_context_class(n, kind) for n in ("Book", "Member", "Library"))
    assert "get_state" in vars(member) and "get_state" not in vars(book)
    assert book.get_state.__func__ is library.get_state.__func__, "Book and Library share the one inherited hook"
    assert member.get_state.__func__ is not book.get_state.__func__

    _, record = _project(inheriting, anchor="members:m3")
    assert record["state"]["written_by"] == "Member"
    _, record = _project(inheriting, anchor="books:b4")
    assert "written_by" not in record["state"]
    restored = _restore(inheriting, record)
    assert type(restored._current_command_context).__name__ == "Book"


# (e) the import that works from a callback file

def test_e_the_relative_import_of_the_helper_resolves_from_a_callback_file(initialized_fastworkflow, inheriting):
    context_model = fastworkflow.CommandContextModel.load(str(inheriting))
    library = context_model.get_context_class("Library", fastworkflow.ModuleType.CONTEXT_CLASS)
    base = library.__mro__[1]
    assert base.__name__ == "CheckpointHooks"
    assert base.__module__.endswith("._commands._shared.checkpoint")
    # one base class for every context, not one copy per file
    book = context_model.get_context_class("Book", fastworkflow.ModuleType.CONTEXT_CLASS)
    assert book.__mro__[1] is base


# the other members the framework reads off the class

def test_the_other_class_members_are_read_with_getattr_so_they_inherit(initialized_fastworkflow, inheriting):
    from fastworkflow.context_identity import declared_instance_label
    from fastworkflow.entry_declarations import declared_occupiable

    commands = inheriting / "_commands"
    (commands / "_shared" / "checkpoint.py").write_text(HELPER.replace(
        "    state_version = 1\n",
        '    state_version = 1\n    instance_label_attr = "uid"\n    occupiable = False\n', 1))
    # a base that also declares the shared get_parent and get_displayname
    (commands / "_shared" / "checkpoint.py").write_text((commands / "_shared" / "checkpoint.py").read_text().replace(
        "    @classmethod\n    def get_state",
        "    @classmethod\n    def get_parent(cls, record):\n        return record.library\n\n"
        "    @classmethod\n    def get_displayname(cls, record) -> str:\n        return 'inherited'\n\n"
        "    @classmethod\n    def get_state", 1))
    (commands / "Book" / "_Book.py").write_text("from .._shared.checkpoint import CheckpointHooks\n\n\n"
                                                "class Context(CheckpointHooks):\n    inherits = [\"Record\"]\n")
    # the base now declares occupiable = False for Member too, which then declares no parents (bd fix-qjne.36)
    member = commands / "Member" / "_Member.py"
    member.write_text(member.read_text().replace('    parents = ["Library"]\n', ""))
    workflow, library = make_workflow(inheriting), _library()
    workflow.root = library
    book = library.books["b3"]
    context_class = fastworkflow.CommandContextModel.load(str(inheriting)).get_context_class(
        "Book", fastworkflow.ModuleType.CONTEXT_CLASS)
    assert "get_parent" not in vars(context_class)
    assert declared_instance_label(context_class, book) == "b3"
    assert declared_occupiable(str(inheriting), "Book") is False
    assert workflow.get_parent(book) is library
    assert workflow._displayname_of(book) == "inherited"
