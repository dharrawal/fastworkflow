"""The ``approval`` annotation: declared on a command class, validated at load, exposed (Phase 10 step 10.13).

``@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS | REQUIRED_UNTIL_DELEGATED)`` on a command's
``ResponseGenerator`` is read the way ``@descends_to`` is (on a planning command's ``PlanGenerator`` it is a load error). Absent means agent discretion. This file covers the
declaration only: nothing is gated here (enforcement is fix-3rsq.10.14).

Each case runs on a COPY of the library example, annotated in the copy. Nothing is mocked: the routing
definition, the command classes, the planning commands and the listing builders are the copy's own.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.core import approval_annotations as aa
from fastworkflow.core.approval_annotations import (
    REQUIRED_ALWAYS, REQUIRED_UNTIL_DELEGATED, command_approval)
from fastworkflow.planning import command_preview
from fastworkflow.core.command_directory import CommandLoadError
from fastworkflow.core.command_metadata_api import CommandMetadataAPI
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.examples.library.application.library import Library
from fastworkflow.run_listing import command_shortlist_text
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"

_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)

MARKER_MANDATORY = "(needs your approval before it runs)"
MARKER_DELEGATABLE = "(may need your approval)"


def _copy(tmp_path, name: str) -> Path:
    folder = tmp_path / name
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    return folder


def _annotate(folder: Path, command_file: str, member: str, *, on: str = "ResponseGenerator") -> Path:
    """Decorate the command's ``ResponseGenerator`` (or ``PlanGenerator``) in the copy with
    ``@user_approval_for_execution(UserApproval.<member>)``."""
    return _decorate(folder, command_file,
                     f"@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.{member})", on=on)


def _decorate(folder: Path, command_file: str, decorator_line: str, *, on: str = "ResponseGenerator") -> Path:
    """Put *decorator_line* on the line above the class *on* in the copy."""
    path = folder / "_commands" / command_file
    text = path.read_text()
    marker = f"class {on}:\n"
    assert text.count(marker) == 1, path
    path.write_text(text.replace(marker, f"{decorator_line}\n{marker}", 1))
    return path


def _in_class(folder: Path, command_file: str, body_line: str, *, on: str = "ResponseGenerator") -> Path:
    """Write *body_line* as a class attribute of the class *on* in the copy (a retired spelling)."""
    path = folder / "_commands" / command_file
    text = path.read_text()
    marker = f"class {on}:\n"
    assert text.count(marker) == 1, path
    path.write_text(text.replace(marker, f"class {on}:\n    {body_line}\n", 1))
    return path


def _cme() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def _listing_app(folder: Path):
    """The copy as an app workflow, entered at the Library root (the shortlist's own context)."""
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"approval-{folder.name}")
    library = Library()
    app.root = library
    app.record = library
    return app


# ---------------------------------------------------------------------------
# Accepted values
# ---------------------------------------------------------------------------

def test_both_allowed_values_load_and_are_read_by_the_accessor(tmp_path):
    folder = _copy(tmp_path, "accepted")
    _annotate(folder, "Library/open_book.py", "REQUIRED_ALWAYS")
    _annotate(folder, "Library/open_member.py", "REQUIRED_UNTIL_DELEGATED")

    assert command_approval(str(folder), "Library/open_book") == REQUIRED_ALWAYS
    assert command_approval(str(folder), "Library/open_member") == REQUIRED_UNTIL_DELEGATED


def test_an_unannotated_command_has_no_approval(tmp_path):
    folder = _copy(tmp_path, "unannotated")
    assert command_approval(str(folder), "Library/find_book") is None


def test_an_unknown_command_has_no_approval(tmp_path):
    folder = _copy(tmp_path, "unknown_name")
    assert command_approval(str(folder), "Library/no_such_command") is None


def test_the_value_is_the_snapshot_in_command_directory_json(tmp_path):
    folder = _copy(tmp_path, "snapshot")
    _annotate(folder, "Library/open_book.py", "REQUIRED_ALWAYS")
    RoutingRegistry.get_definition(str(folder))

    snapshot = json.loads((folder / "___command_info" / "command_directory.json").read_text())
    assert snapshot["map_command_2_metadata"]["Library/open_book"]["approval"] == "required_always"


# ---------------------------------------------------------------------------
# Invalid values are load errors
# ---------------------------------------------------------------------------

def _open_book_problem(folder: Path) -> str:
    """The one load problem of open_book: that command does not load, the workflow does (bd fix-ftdl.2)."""
    RoutingRegistry.clear_registry()
    problems = [p for p in command_preview.load_problems(str(folder)) if p.startswith("Library/open_book: ")]
    assert len(problems) == 1, problems
    return problems[0]


def test_a_retired_string_spelling_is_a_load_error_naming_the_decorator(tmp_path):
    folder = _copy(tmp_path, "retired_string")
    _in_class(folder, "Library/open_book.py", 'approval = "mandatory"')

    message = _open_book_problem(folder)
    assert "Library/open_book.py" in message
    assert 'approval = "mandatory" is removed: write @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)' in message
    with pytest.raises(CommandLoadError):
        RoutingRegistry.get_definition(str(folder)).get_command_class(
            "Library/open_book", fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)


def test_a_retired_non_string_spelling_is_a_load_error(tmp_path):
    folder = _copy(tmp_path, "retired_non_string")
    _in_class(folder, "Library/open_book.py", "approval = True")

    assert "approval = True is removed" in _open_book_problem(folder)


def test_a_decorator_argument_that_is_not_a_member_fails_at_import(tmp_path):
    folder = _copy(tmp_path, "non_member")
    _decorate(folder, "Library/open_book.py",
              '@fastworkflow.user_approval_for_execution("sometimes")')

    # the file does not import: that command (and each command file that imports it) does not load
    assert "takes a UserApproval member" in _open_book_problem(folder)
    definition = RoutingRegistry.get_definition(str(folder))
    assert "Library/open_book" not in definition.command_directory.map_command_2_metadata
    assert definition.get_command_class(
        "Library/find_book", fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS) is not None


def test_the_accessor_surfaces_a_load_error_rather_than_hiding_it(tmp_path):
    folder = _copy(tmp_path, "accessor_error")
    _in_class(folder, "Library/open_book.py", 'approval = "maybe"')

    assert 'approval = "maybe" is removed' in _open_book_problem(folder)
    assert command_approval(str(folder), "Library/open_book") is None


# ---------------------------------------------------------------------------
# Planning commands
# ---------------------------------------------------------------------------

def test_a_planning_command_needs_what_the_commands_its_steps_run_need(tmp_path):
    # bd fix-ftdl.7.6: one inheritance rule for both marks. A planning command's approval is derived from the
    # commands its steps run, as command_writes.write_commands derives its writes (it replaced the one-level helper
    # planning_steps_needing_approval). member_report's steps run find_member, open_member, list_member_loans and
    # show_notes; only open_member is annotated.
    folder = _copy(tmp_path, "planning_steps")
    _annotate(folder, "Library/open_member.py", "REQUIRED_UNTIL_DELEGATED")

    assert command_approval(str(folder), "Library/member_report") == REQUIRED_UNTIL_DELEGATED
    assert aa.approval_marker(str(folder), "Library/member_report") == " " + MARKER_DELEGATABLE
    assert command_approval(str(folder), "Library/author_books") is None     # its steps run no annotated command


def test_the_strongest_approval_a_step_reaches_wins(tmp_path):
    folder = _copy(tmp_path, "planning_strongest")
    _annotate(folder, "Library/open_member.py", "REQUIRED_UNTIL_DELEGATED")
    _annotate(folder, "Record/show_notes.py", "REQUIRED_ALWAYS")
    assert command_approval(str(folder), "Library/member_report") == REQUIRED_ALWAYS


def test_a_planning_command_inherits_through_a_planning_command_its_steps_run(tmp_path):
    folder = _copy(tmp_path, "planning_nested")
    _annotate(folder, "Record/add_note.py", "REQUIRED_ALWAYS")
    (folder / "_commands" / "Library" / "relay_note.py").write_text(
        '''from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Pass a note on to a member's record."""

    plain_utterances = ["pass a note on to Ann Lee", "relay to Raj Patel that he called"]

    class Input(BaseModel):
        member: str = Field(description="the member's name")
        text: str = Field(description="what the note says")


class PlanGenerator:
    def relay(self, input):
        return [run("note_member", member=input.member, text=input.text)]
''')
    assert command_approval(str(folder), "Library/note_member") == REQUIRED_ALWAYS
    assert command_approval(str(folder), "Library/relay_note") == REQUIRED_ALWAYS


def test_the_annotation_on_a_planning_command_is_a_load_error(tmp_path):
    # bd fix-ftdl.7.6: it used to load and only add a listing marker, while @writes there is a load error. Both
    # marks now follow one rule: the mark goes on the execution command a step runs.
    folder = _copy(tmp_path, "planning_self")
    _annotate(folder, "Library/member_report.py", "REQUIRED_ALWAYS", on="PlanGenerator")

    from fastworkflow.planning.command_preview import load_problems

    [problem] = [p for p in load_problems(str(folder)) if "member_report" in p]
    assert "Library/member_report.py" in problem
    assert ("a planning command cannot carry @user_approval_for_execution. Its steps are asked about when they "
            "run a command that carries it: put the decorator on that execution command") in problem


# ---------------------------------------------------------------------------
# Listing markers
# ---------------------------------------------------------------------------

def test_the_planner_and_executor_listing_mark_an_annotated_command(tmp_path):
    folder = _copy(tmp_path, "listing_markers")
    _annotate(folder, "Library/open_book.py", "REQUIRED_ALWAYS")
    _annotate(folder, "Library/open_member.py", "REQUIRED_UNTIL_DELEGATED")
    folder_str = str(folder)

    full = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), active_context_name="Library")
    assert "- open_book " + MARKER_MANDATORY in full
    assert "- find_book " + MARKER_MANDATORY not in full
    assert "- find_book " + MARKER_DELEGATABLE not in full

    callable_here = CommandMetadataAPI.format_callable_here_command(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), owning_context_name="Library",
        qualified_command_name="Library/open_member", active_context_name="Library")
    assert callable_here.splitlines()[0] == "- open_member " + MARKER_DELEGATABLE

    deferred = CommandMetadataAPI.format_deferred_planner_command(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), owning_context_name="Library",
        qualified_command_name="Library/open_member", active_context_name="*")
    assert deferred.splitlines()[0] == "- open_member " + MARKER_DELEGATABLE


def test_the_single_command_listing_marks_the_name_line(tmp_path):
    folder = _copy(tmp_path, "single_marker")
    _annotate(folder, "Library/open_book.py", "REQUIRED_ALWAYS")

    text = CommandMetadataAPI.get_command_display_text_for_command(
        subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library",
        qualified_command_name="Library/open_book")
    assert text.splitlines()[0] == "- open_book " + MARKER_MANDATORY


def test_the_run_listing_shortlist_marks_an_annotated_command(tmp_path):
    folder = _copy(tmp_path, "shortlist_marker")
    _annotate(folder, "Library/open_member.py", "REQUIRED_UNTIL_DELEGATED")
    app = _listing_app(folder)

    text = command_shortlist_text(app, ["Library/open_member", "Library/find_member"])
    assert "- open_member " + MARKER_DELEGATABLE in text
    assert "- find_member " + MARKER_DELEGATABLE not in text
    assert "- find_member (" not in text


def test_marker_text_is_the_only_difference_an_annotation_makes(tmp_path):
    """The annotated listing equals the plain one once the marker is taken out."""
    plain_folder = _copy(tmp_path, "plain_side")
    annotated_folder = _copy(tmp_path, "annotated_side")
    _annotate(annotated_folder, "Library/open_book.py", "REQUIRED_ALWAYS")

    def listing(folder):
        # Command classes are cached by name for the process, so each copy is read on a fresh registry.
        RoutingRegistry.clear_registry()
        return CommandMetadataAPI.get_command_display_text(
            subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library")

    annotated = listing(annotated_folder)
    plain = listing(plain_folder)
    assert MARKER_MANDATORY in annotated and MARKER_MANDATORY not in plain
    assert annotated.replace(" " + MARKER_MANDATORY, "") == plain


# ---------------------------------------------------------------------------
# Unannotated workflows are unchanged
# ---------------------------------------------------------------------------

def test_an_unannotated_workflow_writes_no_approval_key_and_no_marker(tmp_path):
    folder = _copy(tmp_path, "byte_identical")
    RoutingRegistry.get_definition(str(folder))

    snapshot = (folder / "___command_info" / "command_directory.json").read_text()
    assert '"approval"' not in snapshot

    listing = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library")
    assert "approval" not in listing
    assert MARKER_MANDATORY not in listing and MARKER_DELEGATABLE not in listing


def test_the_marker_is_empty_for_an_unannotated_command(tmp_path):
    folder = _copy(tmp_path, "marker_empty")
    assert aa.approval_marker(str(folder), "Library/find_book") == ""
    assert aa.approval_marker(str(folder), "Library/no_such_command") == ""
