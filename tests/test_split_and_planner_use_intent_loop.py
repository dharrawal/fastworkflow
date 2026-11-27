"""The split's over-cap reduction and the planner's plan template decide "which plan does this text mean" with the
intent loop (``resolution.part_intent``, memoised for the turn), not with a planning-command match of their own
(bd fix-zvme). The remote decision model is scripted (``script_matcher``); the library example is the workflow."""
from __future__ import annotations

import pytest

from fastworkflow.core import turn_cache
from fastworkflow import workflow_agent
from fastworkflow._workflows.plan_metadata_extraction import resolution
from tests.conftest import no_local_model, script_matcher
from tests.test_approval_binding import annotated_library, todo  # noqa: F401
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

PLAN = "Library/book_by_title"
ASKS_FOR_PLAN = "who wrote Emma and how many copies do we have"
ASKS_FOR_NOTHING = "tell Ann a joke about cats"


@pytest.fixture
def remote(gated, monkeypatch):  # noqa: F811
    """No local model: the scripted remote model is the whole decision. Returns (workflow, texts the model was asked)."""
    no_local_model(monkeypatch)
    asked = script_matcher(monkeypatch, {ASKS_FOR_PLAN: {PLAN: 0.97, "none": 0.03},
                                         ASKS_FOR_NOTHING: {"none": 0.99, PLAN: 0.01}})
    turn_cache.begin_turn()
    return gated.wf, asked


def test_the_old_planning_command_match_is_gone():
    for name in ("match_planning_command", "_planning_command_matcher", "planner_planning_command_for",
                 "plan_template_for"):
        assert not hasattr(workflow_agent, name), name


def test_a_part_labelled_by_the_split_is_not_decided_again_by_the_resolution(remote):
    workflow, asked = remote
    assert workflow_agent.plan_label(workflow, ASKS_FOR_PLAN) == PLAN
    assert asked == [ASKS_FOR_PLAN]
    intent = resolution.part_intent(workflow, ASKS_FOR_PLAN)      # the resolution that follows: the turn's memo
    assert (intent.outcome, intent.command) == (resolution.MATCH, PLAN)
    assert asked == [ASKS_FOR_PLAN], "the same text was decided a second time"


def test_the_label_is_the_resolutions_own_answer(remote):
    workflow, _ = remote
    for text in (ASKS_FOR_PLAN, ASKS_FOR_NOTHING):
        intent = resolution.part_intent(workflow, text)
        expected = intent.command if intent.outcome == resolution.MATCH and intent.kind == resolution.KIND_PLANNING else None
        assert workflow_agent.plan_label(workflow, text) == expected


def test_only_a_planning_match_is_a_label_and_nothing_is_ever_asked(remote):
    workflow, _ = remote
    assert workflow_agent.plan_label(workflow, ASKS_FOR_NOTHING) is None
    # an execution command that is named is a match, but not a plan
    execution = resolution.part_intent(workflow, "find_member")
    assert execution.kind != resolution.KIND_PLANNING and workflow_agent.plan_label(workflow, "find_member") is None


def test_one_remote_call_per_labelled_part_and_none_for_a_repeated_text(remote):
    workflow, asked = remote
    texts = [ASKS_FOR_PLAN, ASKS_FOR_NOTHING, ASKS_FOR_PLAN]
    labels = [workflow_agent.plan_label(workflow, text) for text in texts]
    assert labels == [PLAN, None, PLAN]
    assert asked == [ASKS_FOR_PLAN, ASKS_FOR_NOTHING]


@pytest.mark.usefixtures("no_planner_key")
def test_a_split_over_the_cap_asks_the_remote_model_once_per_part_and_the_resolution_asks_no_more(gated, monkeypatch):  # noqa: F811
    no_local_model(monkeypatch)
    names = ["Alice Smith", "Bob Jones", "Carol White", "Dave Brown", "Erin Black", "Frank Green", "Grace Hall",
             "Henry Ford", "Irene Adler", "Jack Frost", "Kate Moss", "Liam Neal"][:workflow_agent.MAX_SUBTASKS + 2]
    parts = [f"Check the loans of {name}." for name in names]
    asked = script_matcher(monkeypatch, {text: {PLAN: 0.97, "none": 0.03} for text in parts})
    turn_cache.begin_turn()
    reduction = workflow_agent._reduced_to_cap(" ".join(parts), parts, gated.wf.folderpath, "", gated.wf)
    assert reduction is not None and len(reduction.items) <= workflow_agent.MAX_SUBTASKS
    assert sorted(asked) == sorted(parts), "one remote call per labelled part, no more"
    before = list(asked)
    for text in parts:
        assert resolution.part_intent(gated.wf, text).command == PLAN
    assert asked == before, "the resolution decided a labelled part a second time"


def test_without_the_app_workflow_nothing_is_labelled(gated, monkeypatch):  # noqa: F811
    no_local_model(monkeypatch)
    asked = script_matcher(monkeypatch, {})
    parts = [f"part {n} of the request about book {n}" for n in range(workflow_agent.MAX_SUBTASKS + 2)]
    reduction = workflow_agent._reduced_to_cap(" ".join(parts), parts, gated.wf.folderpath, "")
    assert reduction is not None and asked == []
