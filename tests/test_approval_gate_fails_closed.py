"""The approval gate fails CLOSED (bd fix-ftdl.8.17): a command call whose approval mark the gate cannot read is not run
silently.

Two of the three fail-open cases of the issue are closed here, in ``approval_binding.gate``:

* a name that matches several commands (a leaf found in more than one context) when one of them carries an approval mark:
  the call is refused with a message naming the commands, and runs when called by its qualified name;
* routing that cannot be read: the call is refused with a message, never run unchecked.

The third, the process switch ``FW_APPROVAL_BINDING`` (``off`` / ``shadow``), is a deliberate deployer switch with its
own tests (``tests/test_approval_binding.py``) and is unchanged.

Real library copy with a second ``add_note`` (in the Member context) so the leaf ``add_note`` matches two commands, both
carrying ``REQUIRED_ALWAYS``; the real gate, store and routing. No model is called.
"""
from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

from fastworkflow import approval_binding as ab
from tests.test_approval_binding import (  # noqa: F401 - the real agent on a library copy, and the guards
    _COPY_IGNORE, LIBRARY, _annotate, no_live_model, no_model_call_for_parameters, todo)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture
def annotated_library(tmp_path):
    folder = tmp_path / "library_two_add_notes"
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    os.symlink(LIBRARY / "___command_info" / "versions", folder / "___command_info" / "versions")
    shutil.copy(folder / "_commands" / "Record" / "add_note.py", folder / "_commands" / "Member" / "add_note.py")
    _annotate(folder, "Record/add_note.py", "REQUIRED_ALWAYS")
    _annotate(folder, "Member/add_note.py", "REQUIRED_ALWAYS")
    _annotate(folder, "Record/show_notes.py", "REQUIRED_UNTIL_DELEGATED")
    return folder


def _refused(output) -> str:
    assert output is not None, "the gate let the call run"
    assert output.command_response.success is False
    return output.command_response.response


def test_a_name_that_matches_several_marked_commands_is_refused_not_run(todo):
    text = _refused(ab.gate(todo.ctx, todo.wf, "add_note", None, "add_note"))
    assert "Member/add_note" in text and "Record/add_note" in text and "was not run" in text, text
    assert todo.store.records == []


def test_the_qualified_name_is_still_gated_as_before(todo):
    with ab.origin_scope(ab.Origin.AGENT_DIRECT):
        held = ab.gate(todo.ctx, todo.wf, "Record/add_note", None, "add_note")
    assert held is not None and "Reply yes to approve" in held.command_response.response


def test_a_name_that_matches_no_command_has_no_mark_and_is_not_refused_here(todo):
    assert ab.gate(todo.ctx, todo.wf, "no_such_command", None, "no_such_command") is None


def test_routing_that_cannot_be_read_refuses_the_call(tmp_path):
    workflow = SimpleNamespace(folderpath=str(tmp_path / "no_workflow_here"))
    text = _refused(ab.gate(None, workflow, "add_note", None, "add_note"))
    assert "add_note was not run" in text and "approval" in text, text
