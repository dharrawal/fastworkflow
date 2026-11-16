"""Switching an ambiguous step's plan re-drafts its inputs (bd fix-1ma8.1).

The companion of ``tests/test_plan_match_ambiguity.py`` with drafting ON (the shared parameter extractor, scripted as
``tests/test_drafted_subtask_inputs.py`` scripts it): the draft carries the alternatives, a switch is the means edit,
``SubTaskProposal.revised`` drafts the inputs again for the new plan, and what the user set in the same reply is kept.
Also the no-agent draft (``confident_only``), which keeps the alternatives only for a pick that is still the winner.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow import review_proposals, run_listing
from fastworkflow.review_engine import ProposalReplyPolicy, Verdict, apply_edits
from fastworkflow.review_proposals import Means, SubTaskInput
from tests.test_approval_binding import annotated_library  # noqa: F401
from tests.test_drafted_subtask_inputs import EMMA, JOKE, _framework, library, scripted  # noqa: F401

BY_TITLE = "Library/book_by_title"
AUTHOR = "Library/author_books"
LOANS = "Library/book_loans"


@pytest.fixture
def no_heads(monkeypatch):
    monkeypatch.setattr(run_listing, "head_answers", lambda workflow, sub_task, command_info_root=None: {})


def _jev(monkeypatch, by_text, library=None):
    """Jev's match, scripted: the distribution of each text, and (given the workflow) the planning command it picks
    when its best option is at or above the match bar, as ``planning_match`` returns it."""
    from fastworkflow import workflow_agent

    def match(workflow, text, exclude=None):
        distribution = dict(by_text.get(text, {}))
        best = max((name for name in distribution if name != "none"), key=distribution.get, default=None)
        command = None
        if library is not None and best is not None and distribution[best] >= workflow_agent.PLANNING_COMMAND_MATCH_THRESHOLD:
            command = workflow_agent.load_planning_command_map(library.folderpath).get(best)
        return "", command, distribution

    monkeypatch.setattr(review_proposals, "planning_match", match)


def _host(library):
    """What ``SubTaskProposal.revised`` reads from the session: the app workflow and the planner's LM."""
    return SimpleNamespace(_app_workflow=library, _current_planner_lm=None)


def test_a_switch_re_drafts_the_inputs_for_the_new_plan_and_swaps_the_alternatives(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.55, AUTHOR: 0.45, "none": 0.0}, JOKE: {"none": 0.95}})
    lm = scripted({"reasoning": "the title", "title": "Emma"}, {"reasoning": "the author", "author": "Emma"})

    draft = review_proposals.draft_round1_proposal(library, [EMMA, JOKE])

    first = draft.items[0]
    assert first.means == Means(kind="planning", name=BY_TITLE)
    assert first.inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]
    assert [each.name for each in first.alternatives] == [AUTHOR]
    assert draft.items[1].alternatives == []

    decision = ProposalReplyPolicy().decide(draft, "step 1: use author_books")
    assert decision.verdict is Verdict.EDIT
    revised = apply_edits(draft, list(decision.edits)).revised(_host(library), draft)

    switched = revised.items[0]
    assert switched.means == Means(kind="planning", name=AUTHOR)
    assert switched.inputs == [SubTaskInput(name="author", value="Emma", drafted=True)]       # drafted again, for author_books
    assert switched.drafted_means == AUTHOR
    assert [each.name for each in switched.alternatives] == [BY_TITLE]
    assert switched.alternatives[0].description.startswith("Show who wrote a book")
    assert len(lm.prompts) == 2 and "author" in lm.prompts[1]
    assert revised.items[1] == draft.items[1]                                                 # the other part is untouched


def test_what_the_user_set_in_the_same_reply_is_kept_over_the_new_draft(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.55, AUTHOR: 0.45, "none": 0.0}})
    scripted({"reasoning": "the title", "title": "Emma"}, {"reasoning": "the author", "author": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    reply = "change 1: means=planning:Library/author_books; change 1: author=Emma Woodhouse"
    edits = ProposalReplyPolicy().decide(draft, reply).edits
    revised = apply_edits(draft, list(edits)).revised(_host(library), draft)
    assert revised.items[0].means.name == AUTHOR
    assert {each.name: each.value for each in revised.items[0].inputs}["author"] == "Emma Woodhouse"


def test_a_means_the_user_types_that_is_not_an_alternative_ends_the_alternatives(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.55, AUTHOR: 0.45, "none": 0.0}})
    scripted({"reasoning": "the title", "title": "Emma"}, {"reasoning": "the title", "title": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    edits = ProposalReplyPolicy().decide(draft, f"change 1: means=planning:{LOANS}").edits
    revised = apply_edits(draft, list(edits)).revised(_host(library), draft)
    assert revised.items[0].means.name == LOANS and revised.items[0].alternatives == []


def test_the_alternatives_follow_a_switch_even_with_drafting_off(library, no_heads, monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_DRAFT_ARGUMENTS", "off")
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.55, AUTHOR: 0.45, LOANS: 0.25, "none": 0.0}})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    assert [each.name for each in draft.items[0].alternatives] == [AUTHOR, LOANS] and draft.items[0].inputs == []
    edits = ProposalReplyPolicy().decide(draft, f"use {LOANS}").edits
    revised = apply_edits(draft, list(edits)).revised(_host(library), draft)
    assert revised.items[0].means.name == LOANS
    assert [each.name for each in revised.items[0].alternatives] == [BY_TITLE, AUTHOR]


def test_a_no_agent_draft_keeps_the_alternatives_of_a_confident_winner(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.55, AUTHOR: 0.45, "none": 0.0}}, library)       # Jev's pick is the winner (>= 0.5)
    scripted({"reasoning": "the title", "title": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA], confident_only=True)
    assert draft.items[0].means == Means(kind="planning", name=BY_TITLE)
    assert [each.name for each in draft.items[0].alternatives] == [AUTHOR]
    assert review_proposals.plan_ambiguity_record(draft) == [
        {"subtask": 1, "means": f"planning:{BY_TITLE}", "alternatives": [AUTHOR]}]


def test_a_no_agent_draft_that_is_not_confident_is_ad_hoc_and_names_no_alternatives(library, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.45, AUTHOR: 0.4, "none": 0.1}}, library)         # below the 0.5 match bar
    draft = review_proposals.draft_round1_proposal(library, [EMMA], confident_only=True)
    assert draft.items[0].means.kind == "from_scratch" and draft.items[0].alternatives == []


def test_an_unreviewed_draft_vetoed_by_jev_has_no_alternatives(library, scripted, no_heads, monkeypatch):
    # the planning pick is dropped to from_scratch by Jev's confident "none" (fix-6lpc), so its alternatives go with it
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.05, AUTHOR: 0.04, "none": 0.91, LOANS: 0.0}}, library)
    draft = review_proposals.draft_round1_proposal(library, [EMMA], none_veto=True)
    assert draft.items[0].means.kind == "from_scratch" and draft.items[0].alternatives == []
