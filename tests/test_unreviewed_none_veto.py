"""The veto of an unreviewed draft in agent mode (fix-6lpc).

When no one can answer the sub-task review (an unattended turn, a host surface), the draft runs as drafted. A planning
command the draft picked for a sub-task is not bound when Jev's "none" is at or above
``workflow_agent.PLANNING_NONE_VETO_THRESHOLD`` (0.9): the sub-task is ``from_scratch`` and the agent plans it. A reviewed
draft (a human sees it), an exact command name and a silent or unsure Jev are not touched. Driven as
tests/test_round1_gate.py is (real library copy, trained heads, scripted agent model); Jev's match is the one scripted seam.
"""
from __future__ import annotations

import pytest

from fastworkflow import review_proposals
from fastworkflow.review_proposals import MEANS_FROM_SCRATCH, MEANS_PLANNING
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def _jev(monkeypatch, none: float | None):
    """Jev's planning-command match, scripted: *none* is its probability for "none" on every text (None: silent)."""
    def match(workflow, text, exclude=None):
        return "", None, ({} if none is None else {"none": none})

    monkeypatch.setattr(review_proposals, "planning_match", match)


def _first_means(plan):
    return plan.items[0].means


def test_the_bar_is_above_the_match_bar():
    from fastworkflow import workflow_agent
    assert workflow_agent.PLANNING_NONE_VETO_THRESHOLD == 0.9 > workflow_agent.PLANNING_COMMAND_MATCH_THRESHOLD


def test_unreviewed_and_a_confident_none_leaves_the_sub_task_to_the_agent(gated, monkeypatch):
    _jev(monkeypatch, 0.95)
    _split_into(monkeypatch, [EMMA, JOKE])
    asked: list[str] = []
    original = gated.ctx._call_agent_for_query

    def spy(query, *args, **kwargs):
        asked.append(query)
        return original(query, *args, **kwargs)

    monkeypatch.setattr(gated.ctx, "_call_agent_for_query", spy)
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert [r for r in gated.store.records if r.command == "review_subtasks"] == []
    assert _first_means(gated.ctx._turn_subtask_plan).kind == MEANS_FROM_SCRATCH
    assert any(EMMA in query for query in asked)            # the agent was called with the sub-task
    assert gated.script.steps == []


def test_unreviewed_and_a_none_below_the_veto_bar_binds_as_before(gated, monkeypatch):
    _jev(monkeypatch, 0.7)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert out.status is TurnStatus.COMPLETED
    means = _first_means(gated.ctx._turn_subtask_plan)
    assert (means.kind, means.name) == (MEANS_PLANNING, "Library/book_by_title")


def test_a_host_surface_turn_is_vetoed_too(gated, monkeypatch):
    from fastworkflow.approval_binding import Surface
    _jev(monkeypatch, 0.95)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=Surface(kind="host", verified=True, uid="user-1"))
    assert _first_means(gated.ctx._turn_subtask_plan).kind == MEANS_FROM_SCRATCH


def test_a_reviewed_draft_is_shown_unchanged_whatever_jev_says(gated, monkeypatch):
    _jev(monkeypatch, 0.95)
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert out.status is TurnStatus.AWAITING_USER
    assert _open_review(gated).binding["items"][0]["means"] == {"kind": "planning", "name": "Library/book_by_title"}


def test_an_exact_command_name_is_never_vetoed(gated, monkeypatch):
    _jev(monkeypatch, 0.95)
    plan = review_proposals.draft_round1_proposal(gated.wf, ["count_books"], none_veto=True)
    assert (plan.items[0].means.kind, plan.items[0].means.name) == (MEANS_PLANNING, "Library/count_books")


def test_a_silent_jev_changes_nothing(gated, monkeypatch):
    _jev(monkeypatch, None)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert _first_means(gated.ctx._turn_subtask_plan).name == "Library/book_by_title"


def test_the_veto_is_off_unless_asked_for(gated, monkeypatch):
    _jev(monkeypatch, 0.95)
    assert _first_means(review_proposals.draft_round1_proposal(gated.wf, [EMMA, JOKE])).name == "Library/book_by_title"
    assert _first_means(review_proposals.draft_round1_proposal(gated.wf, [EMMA, JOKE], none_veto=True)
                        ).kind == MEANS_FROM_SCRATCH
