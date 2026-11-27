"""No-agent mode: a yes whose inner step is an approval-annotated WRITE (fix-rn9p).

The yes makes the server run the conditional step's inner command itself, as a step of the phase. That is not an
approval and not a bypass of one: the command goes through the same approval gate as any code step. On a host surface
whose identity is not verified, a command that needs the end user's approval is refused and nothing is written; with
a verified identity the client gets the approval prompt (and its token) and the write waits for it. Real components on the
annotated library copy of ``tests/test_approval_binding.py``; no model is called (the agent LM must never be built).
"""
from __future__ import annotations

import pytest

from fastworkflow.review import approval_binding as ab
from fastworkflow.review.approval_binding import Surface
from fastworkflow.planning import run, when
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - fixtures and helpers
    _added, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_no_agent_mode import _no_agent, _plan  # noqa: F401 - the autouse no-agent fixture and the planner seam
from tests.test_step_helpers import FoundBooks, _build_planning_command  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture


class NoteIfWanted:
    """One phase whose only step is conditional, and whose inner step is the annotated write ``add_note``."""

    def note(self, input) -> FoundBooks:
        return [when("the reader wants the note added", run("add_note", text="asked about"))]

    def __result__(self, input, note):
        return "handled"


def _prepare(todo, monkeypatch):
    todo.planning_command = _build_planning_command(NoteIfWanted)
    todo.args = {"title": "Dune"}
    _plan(monkeypatch, todo)


def test_a_yes_on_an_unverified_host_does_not_bypass_the_approval_gate(todo, monkeypatch):
    """The question needs no identity; the write it leads to still does. The unverified host's yes runs the inner step
    through the gate, which refuses it: nothing is written, no approval is recorded, and the command ends with its
    stop (``failed``) whose reason carries the refusal."""
    _prepare(todo, monkeypatch)
    host = Surface("host", verified=False, uid="alice")
    first = todo.ctx.process_turn("note the book", approval_surface=host)
    assert first.status == TurnStatus.AWAITING_USER
    assert first.command_outputs[-1].command_response.artifacts["question_kind"] == "phase_step"
    assert _added(todo.chores, todo.base) == []

    second = todo.ctx.process_turn("yes", approval_surface=host)
    assert second.status == TurnStatus.COMPLETED
    refused = [o for o in second.command_outputs if (o.command_name or "").endswith("add_note")]
    assert len(refused) == 1 and refused[0].success is False
    assert ab.MSG_IDENTITY[:40] in refused[0].command_response.response
    (stop,) = second.command_outputs[-1].command_response.artifacts["fw_not_run"]
    assert stop["open_step"]["kind"] == "failed" and ab.MSG_IDENTITY[:40] in stop["open_step"]["reason"]
    assert _added(todo.chores, todo.base) == [] and todo.store.active() is None


def test_a_yes_on_a_verified_host_asks_for_the_approval_and_writes_nothing_yet(todo, monkeypatch):
    """With a verified identity the yes reaches the approval prompt of the annotated command: the turn waits on it (the
    ordinary approval record, carried in the ``approval`` artifact) and the write has not run."""
    _prepare(todo, monkeypatch)
    host = Surface("host", verified=True, uid="alice")
    assert todo.ctx.process_turn("note the book", approval_surface=host).status == TurnStatus.AWAITING_USER

    second = todo.ctx.process_turn("yes", approval_surface=host)
    assert second.status == TurnStatus.AWAITING_USER
    assert second.answer.startswith("The agent wants to run Record/add_note with text=asked about.")
    assert "approval" in second.command_outputs[-1].command_response.artifacts
    record = todo.store.active()
    assert record is not None and record.command == "Record/add_note" and record.state == "awaiting"
    assert _added(todo.chores, todo.base) == []
