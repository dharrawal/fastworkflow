"""An unreviewed draft in agent mode binds only what the one decision matched (bd fix-o1lg; was the veto of fix-6lpc).

When no one can answer the sub-task review (an unattended turn, a host surface), the draft runs as drafted. The veto that
dropped a planning pick when Jev's "none" reached 0.9 was deleted with the score merge it corrected: the part's means is
now ``resolution.part_intent`` (exact name, then the local model, then the remote model), and a draft nobody reviews
binds a MATCH only. A remote "none" or an ambiguity leaves the part to the agent; a remote match and an exact name are
bound. Driven as tests/test_round1_gate.py is (real library copy, trained heads, scripted agent model); the remote model's
answer is the one scripted seam (``script_matcher``).
"""
from __future__ import annotations

import pytest

from fastworkflow.review import review_proposals
from fastworkflow.review.review_proposals import MEANS_FROM_SCRATCH, MEANS_PLANNING
from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

BY_TITLE = "Library/book_by_title"


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def _jev(monkeypatch, scores: dict | None):
    """The remote decision model's answer for EMMA (None: no answer, so the local model's stands)."""
    return script_matcher(monkeypatch, {} if scores is None else {EMMA: scores})


NONE = {"none": 0.95}
MATCH = {BY_TITLE: 0.93, "none": 0.04}
AMBIGUOUS = {BY_TITLE: 0.45, "Library/author_books": 0.4, "none": 0.1}


def _first_means(plan):
    return plan.items[0].means


def test_unreviewed_and_a_remote_none_leaves_the_sub_task_to_the_agent(gated, monkeypatch):
    _jev(monkeypatch, NONE)
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


def test_unreviewed_and_a_remote_match_binds(gated, monkeypatch):
    _jev(monkeypatch, MATCH)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert out.status is TurnStatus.COMPLETED
    means = _first_means(gated.ctx._turn_subtask_plan)
    assert (means.kind, means.name) == (MEANS_PLANNING, BY_TITLE)


@pytest.mark.parametrize("scores", [AMBIGUOUS, None], ids=["remote_ambiguous", "remote_silent"])
def test_unreviewed_and_no_match_leaves_the_sub_task_to_the_agent(gated, monkeypatch, scores):
    """An ambiguity (the remote model unsure, or silent so the local model's unsure answer stands) is a pick nobody
    would judge: the agent plans the part (it bound the best-ranked plan before bd fix-o1lg)."""
    _jev(monkeypatch, scores)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert _first_means(gated.ctx._turn_subtask_plan).kind == MEANS_FROM_SCRATCH


def test_a_host_surface_turn_binds_only_a_match_too(gated, monkeypatch):
    from fastworkflow.review.approval_binding import Surface
    _jev(monkeypatch, NONE)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=Surface(kind="host", verified=True, uid="user-1"))
    assert _first_means(gated.ctx._turn_subtask_plan).kind == MEANS_FROM_SCRATCH


def test_an_attended_turn_asks_about_an_ambiguity_before_the_review_and_shows_a_remote_none_as_from_scratch(
        gated, monkeypatch):
    """Rewritten for bd fix-o1lg S4/S5 (the reviewed draft bound the ambiguity's best plan and named the other): the
    ambiguous part is asked about first; the picked plan is bound and the part the remote model said none fits is
    from scratch in the review."""
    _jev(monkeypatch, AMBIGUOUS)
    _split_into(monkeypatch, [EMMA, JOKE])
    asked = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert asked.status is TurnStatus.AWAITING_USER
    artifacts = asked.command_outputs[-1].command_response.artifacts
    assert artifacts["question_kind"] == "plan_choice"
    assert [each["name"] for each in artifacts["plan_choice"]["choices"]][:2] == [BY_TITLE, "Library/author_books"]
    out = gated.ctx.process_turn(BY_TITLE)
    if out.command_outputs[-1].command_response.artifacts.get("question_kind") == "plan_choice":
        out = gated.ctx.process_turn("none")                  # the joke is asked about too when it is ambiguous
    items = _open_review(gated).binding["items"]
    assert items[0]["means"] == {"kind": "planning", "name": BY_TITLE} and "alternatives" not in items[0]
    assert items[1]["means"]["kind"] == "from_scratch"


def test_an_exact_command_name_is_bound_whatever_the_remote_model_would_say(gated, monkeypatch):
    asked = _jev(monkeypatch, NONE)
    plan = review_proposals.draft_round1_proposal(gated.wf, ["count_books"])
    assert (plan.items[0].means.kind, plan.items[0].means.name) == (MEANS_PLANNING, "Library/count_books")
    assert asked == []
