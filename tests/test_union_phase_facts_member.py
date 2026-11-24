"""A phase annotated with a union of Outputs hands on the member that ran (bd fix-ftdl.4.1).

Both members have every field defaulted, so a read-back that tries the members in order would accept any
run's saved values as the first member. The saved facts name the member that filled them, and a later phase,
``__result__`` and a restored session all read it back by that name. No model is called: this uses the
phase-runner harness of ``test_planning_command_phases`` (scripted executor, real runner).
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from fastworkflow.planning_commands import call_phases
from fastworkflow.planning_phases import typed_phase_outputs
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_planning_command_phases import (  # noqa: F401 - fixtures and helpers
    SEEN, SCRIPT, FILL_COMMANDS, NoteFacts, _phased_command, _run, command_output, no_review_gate, phased)
from tests.test_turn_result_capture import initialized_fastworkflow, todo_workflow_path  # noqa: F401

ByBook = command_output("find_book", book_uids=(list[str], Field(default_factory=list)))
ByLoan = command_output("list_loans", loan_uids=(list[str], Field(default_factory=list)))


class UnionPlan:
    def find(self, input) -> ByBook | ByLoan:
        return ["list_loans"]

    def notes(self, find) -> NoteFacts:
        SEEN["notes"] = {"find": find}
        return ["count_notes"]

    def __result__(self, input, find, notes=None):
        SEEN["report"] = {"find": find}
        return "done."


def _script(phased):
    phased.command = _phased_command(PlanGenerator=UnionPlan)
    phased.script["find"] = dict(runs=[_run("list_loans", {"loan_uids": ["l1", "l2"]})], record="Library",
                                 answer="found")
    phased.script["notes"] = dict(runs=[_run("count_notes", {"total": 3})], record="Library", answer="3")


def test_a_later_phase_and_result_get_the_member_that_ran(phased):
    _script(phased)
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.status == TurnStatus.COMPLETED
    for place in ("notes", "report"):
        found = SEEN[place]["find"]
        assert type(found) is ByLoan, place
        assert found.loan_uids == ["l1", "l2"]


def test_a_restored_session_reads_the_member_that_ran(phased):
    _script(phased)
    phased.start(suspend_in="notes")
    first = phased.ctx.process_turn("show the book Dune")
    assert first.status == TurnStatus.AWAITING_USER

    blob = json.loads(json.dumps(phased.ctx.serialize_state(channel_id="union-facts")))
    restored = WorkflowExecutionContext(run_as_agent=True)
    restored.apply_serialized_state(blob)

    phases = call_phases(phased.command, restored._phase_run["args"])
    found = typed_phase_outputs(restored._phase_run["outputs"], phases)["find"]
    assert type(found) is ByLoan and found.loan_uids == ["l1", "l2"]

