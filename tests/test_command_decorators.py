"""Command class declarations are decorators (design idioms, spec section 4).

``@user_approval_for_execution(UserApproval.X)``, ``@writes`` and ``@descends_to("<Context>")``
replace the class attributes ``approval``, ``writes`` and ``descends_to`` / ``descend_parameter``. Each decorator
stores its value on a private attribute with a mark only the decorator sets. A retired attribute (naming the
decorator to write) and a private attribute written by hand fail the load of THAT command: it is listed by
``command_preview.load_problems`` and does not load, and the rest of the workflow loads (bd fix-ftdl.2).

Each load case runs on a COPY of the library example, edited in the copy. Nothing is mocked: the routing
definition, the command classes and the saved ``command_directory.json`` are the copy's own.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.core import command_declarations, command_writes
from fastworkflow.agent import workflow_agent
from fastworkflow.planning import command_preview
from fastworkflow.planning import planning_commands as pc
from fastworkflow.core.approval_annotations import command_approval
from fastworkflow.core.command_declarations import UserApproval
from fastworkflow.core.command_directory import CommandLoadError
from fastworkflow.core.command_routing import RoutingRegistry
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)


def _copy(tmp_path, name: str) -> Path:
    folder = tmp_path / name
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    RoutingRegistry.clear_registry()
    return folder


def _edit(folder: Path, command_file: str, old: str, new: str) -> None:
    path = folder / "_commands" / command_file
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new, 1))


def _load(folder: Path):
    RoutingRegistry.clear_registry()
    return RoutingRegistry.get_definition(str(folder))


def _fails_load(folder: Path, command: str) -> str:
    """The one load problem of *command* (the workflow itself loads): the command does not load and is listed."""
    _load(folder)
    pc.resolve_field_sources.cache_clear()
    problems = command_preview.load_problems(str(folder))
    mine = [p for p in problems if p.startswith(f"{command}: ")]
    assert len(mine) == 1, problems
    metadata = RoutingRegistry.get_definition(str(folder)).command_directory.map_command_2_metadata[command]
    if metadata.is_planning:
        assert command not in workflow_agent.load_planning_command_map(str(folder))
    else:
        with pytest.raises(CommandLoadError):
            RoutingRegistry.get_definition(str(folder)).get_command_class(
                command, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
    assert RoutingRegistry.get_definition(str(folder)).get_command_class(
        "Library/find_book", fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS) is not None
    return mine[0]


def _command_directory(folder: Path) -> dict:
    return json.loads((folder / "___command_info" / "command_directory.json").read_text())


# ---------------------------------------------------------------------------
# The decorators on a class
# ---------------------------------------------------------------------------

def test_user_approval_has_the_two_values_the_gate_reads():
    assert [member.name for member in UserApproval] == ["REQUIRED_ALWAYS", "REQUIRED_UNTIL_DELEGATED"]
    assert UserApproval.REQUIRED_ALWAYS.value == "required_always"
    assert UserApproval.REQUIRED_UNTIL_DELEGATED.value == "required_until_delegated"


@pytest.mark.parametrize("argument", ["mandatory", "delegatable", None, True, 1, UserApproval])
def test_a_non_member_argument_fails_when_the_decorator_is_applied(argument):
    with pytest.raises(TypeError, match="takes a UserApproval member"):
        fastworkflow.user_approval_for_execution(argument)


def test_the_approval_decorator_stores_its_member_on_a_private_attribute():
    @fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_UNTIL_DELEGATED)
    class ResponseGenerator:
        pass

    assert ResponseGenerator.__fw_approval__ is UserApproval.REQUIRED_UNTIL_DELEGATED
    assert not hasattr(ResponseGenerator, "approval")
    assert command_declarations.declared_approval(ResponseGenerator) is UserApproval.REQUIRED_UNTIL_DELEGATED


def test_a_class_without_the_decorator_declares_no_approval():
    class ResponseGenerator:
        pass

    assert command_declarations.declared_approval(ResponseGenerator) is None


def test_writes_marks_an_execution_command_and_is_the_only_form():
    @fastworkflow.writes
    class ResponseGenerator:
        pass

    assert ResponseGenerator.__fw_writes__ is True
    assert command_declarations.declares_writes(ResponseGenerator)
    assert fastworkflow.writes is command_declarations.writes


def test_writes_on_something_that_is_not_a_class_fails():
    with pytest.raises(TypeError, match="decorates a class"):
        fastworkflow.writes(42)


def test_descends_to_stores_the_context():
    @fastworkflow.descends_to("Member")
    class Moves:
        pass

    @fastworkflow.descends_to("  Book ")
    class Stripped:
        pass

    assert Moves.__fw_descends_to__ == "Member"
    assert Stripped.__fw_descends_to__ == "Book"
    assert command_declarations.declared_descend(Moves) == "Member"
    assert command_declarations.declared_descend(Stripped) == "Book"
    assert command_declarations.declared_descend(object) is None


@pytest.mark.parametrize("context", ["", "   ", None, 3])
def test_descends_to_needs_a_context_name(context):
    with pytest.raises(TypeError, match="takes the name of a context"):
        fastworkflow.descends_to(context)


def test_the_package_exports_the_four_names():
    assert fastworkflow.UserApproval is UserApproval
    assert fastworkflow.user_approval_for_execution is command_declarations.user_approval_for_execution
    assert fastworkflow.descends_to is command_declarations.descends_to
    assert fastworkflow.writes is command_declarations.writes


def test_the_planning_module_exports_the_approval_and_navigation_decorators():
    from fastworkflow import planning

    assert planning.UserApproval is UserApproval
    assert planning.user_approval_for_execution is command_declarations.user_approval_for_execution
    assert planning.descends_to is command_declarations.descends_to


# ---------------------------------------------------------------------------
# The decorators on the library, loaded
# ---------------------------------------------------------------------------

def test_an_approval_decorator_is_saved_as_its_value_in_command_directory_json(tmp_path):
    folder = _copy(tmp_path, "approval_saved")
    _edit(folder, "Library/open_book.py", "class ResponseGenerator:\n",
          "@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_ALWAYS)\n"
          "class ResponseGenerator:\n")
    _edit(folder, "Library/open_member.py", "class ResponseGenerator:\n",
          "@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_UNTIL_DELEGATED)\n"
          "class ResponseGenerator:\n")

    _load(folder)

    metadata = _command_directory(folder)["map_command_2_metadata"]
    assert metadata["Library/open_book"]["approval"] == "required_always"
    assert metadata["Library/open_member"]["approval"] == "required_until_delegated"
    assert command_approval(str(folder), "Library/open_book") == "required_always"


def test_an_undecorated_command_has_no_approval_key_in_command_directory_json(tmp_path):
    folder = _copy(tmp_path, "approval_absent")
    _load(folder)

    metadata = _command_directory(folder)["map_command_2_metadata"]
    assert "approval" not in metadata["Library/find_book"]
    assert command_approval(str(folder), "Library/find_book") is None


def test_removing_the_decorator_clears_the_saved_approval(tmp_path):
    folder = _copy(tmp_path, "approval_cleared")
    _edit(folder, "Library/open_book.py", "class ResponseGenerator:\n",
          "@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_ALWAYS)\n"
          "class ResponseGenerator:\n")
    _load(folder)
    assert _command_directory(folder)["map_command_2_metadata"]["Library/open_book"]["approval"] == "required_always"

    _edit(folder, "Library/open_book.py",
          "@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_ALWAYS)\n", "")
    _load(folder)

    assert "approval" not in _command_directory(folder)["map_command_2_metadata"]["Library/open_book"]


def test_writes_on_an_execution_command_is_read_as_a_write(tmp_path):
    folder = _copy(tmp_path, "writes_read")
    _load(folder)

    assert command_writes.write_commands(str(folder)) >= {"Record/add_note"}


def test_writes_on_a_planning_command_fails_the_load(tmp_path):
    folder = _copy(tmp_path, "writes_on_plan")
    _edit(folder, "Library/note_member.py", "class PlanGenerator:\n",
          "import fastworkflow\n\n\n@fastworkflow.writes\nclass PlanGenerator:\n")

    assert "cannot carry @writes" in _fails_load(folder, "Library/note_member")


def test_a_retired_descends_to_attribute_fails_the_load_naming_the_decorator(tmp_path):
    folder = _copy(tmp_path, "retired_descends")
    _edit(folder, "Library/open_book.py",
          '@fastworkflow.descends_to("Book")\nclass ResponseGenerator:\n',
          'class ResponseGenerator:\n    descends_to = "Book"\n')

    assert 'descends_to = "Book" is removed: write @descends_to("Book")' in _fails_load(folder, "Library/open_book")


def test_a_retired_descend_parameter_fails_the_load_saying_the_move_is_unconditional(tmp_path):
    folder = _copy(tmp_path, "retired_gate")
    _edit(folder, "Library/open_book.py",
          '@fastworkflow.descends_to("Book")\nclass ResponseGenerator:\n',
          'class ResponseGenerator:\n    descends_to = "Book"\n    descend_parameter = "book_uid"\n')

    text = _fails_load(folder, "Library/open_book")
    assert 'descends_to = "Book" is removed: write @descends_to("Book")' in text
    assert 'descend_parameter = "book_uid" is removed' in text
    assert "unconditional" in text


@pytest.mark.parametrize("spelling", ['when_parameter="book_uid"', 'when_parameter=""', "when_parameter=3"])
def test_descends_to_with_when_parameter_fails_the_load_of_that_command(tmp_path, spelling):
    folder = _copy(tmp_path, "when_parameter_removed")
    _edit(folder, "Library/open_book.py", '@fastworkflow.descends_to("Book")',
          f'@fastworkflow.descends_to("Book", {spelling})')

    text = _fails_load(folder, "Library/open_book")
    assert "open_book" in text
    assert "when_parameter is removed" in text
    assert "@descends_to always declares an unconditional move" in text
    assert "assigns workflow.record only when it moves" in text


def test_a_retired_writes_attribute_fails_the_load(tmp_path):
    folder = _copy(tmp_path, "retired_writes")
    _edit(folder, "Record/add_note.py", "@fastworkflow.writes\nclass ResponseGenerator:\n",
          "class ResponseGenerator:\n    writes = True\n")

    assert "writes = True is removed: write @writes" in _fails_load(folder, "Record/add_note")


def test_a_hand_written_approval_attribute_fails_the_load(tmp_path):
    folder = _copy(tmp_path, "hand_approval")
    _edit(folder, "Library/open_book.py", "class ResponseGenerator:\n",
          "class ResponseGenerator:\n    __fw_approval__ = fastworkflow.UserApproval.REQUIRED_ALWAYS\n")

    text = _fails_load(folder, "Library/open_book")
    assert "__fw_approval__ is written by hand" in text
    assert "@user_approval_for_execution" in text


def test_a_hand_written_writes_attribute_fails_the_load(tmp_path):
    folder = _copy(tmp_path, "hand_writes")
    _edit(folder, "Record/add_note.py", "@fastworkflow.writes\nclass ResponseGenerator:\n",
          "class ResponseGenerator:\n    __fw_writes__ = True\n")

    text = _fails_load(folder, "Record/add_note")
    assert "__fw_writes__ is written by hand" in text
    assert "@writes" in text


def test_a_hand_written_descends_to_attribute_fails_the_load(tmp_path):
    folder = _copy(tmp_path, "hand_descends")
    _edit(folder, "Library/open_book.py", '@fastworkflow.descends_to("Book")\nclass ResponseGenerator:\n',
          'class ResponseGenerator:\n    __fw_descends_to__ = ("Book", None)\n')

    text = _fails_load(folder, "Library/open_book")
    assert "__fw_descends_to__ is written by hand" in text
    assert "@descends_to" in text
