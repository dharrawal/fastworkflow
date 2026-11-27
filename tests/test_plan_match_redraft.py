"""Changing a step's plan in the review re-drafts its inputs (bd fix-1ma8.1).

The companion of ``tests/test_plan_match_ambiguity.py`` with drafting ON (the shared parameter extractor, scripted as
``tests/test_drafted_subtask_inputs.py`` scripts it): a means edit makes ``SubTaskProposal.revised`` draft the inputs
again for the new plan, and what the user set in the same reply is kept. The draft binds only a match (bd fix-o1lg):
since S4/S5 an ambiguity is asked about before the review, and the review's plan alternatives are gone (bd fix-gwr7).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.review import review_proposals
from fastworkflow.review.review_engine import ProposalReplyPolicy, Verdict, apply_edits
from fastworkflow.review.review_proposals import Means, SubTaskInput
from tests.test_approval_binding import annotated_library  # noqa: F401
from tests.test_drafted_subtask_inputs import EMMA, JOKE, _framework, library, scripted  # noqa: F401
from tests.conftest import no_local_model, script_matcher

BY_TITLE = "Library/book_by_title"
AUTHOR = "Library/author_books"
LOANS = "Library/book_loans"


@pytest.fixture
def no_heads(monkeypatch):
    no_local_model(monkeypatch)


def _jev(monkeypatch, by_text, library=None):
    """The remote decision model's answer, scripted for each text (``script_matcher``)."""
    script_matcher(monkeypatch, by_text)


#: No plan reaches the accept bar (0.9): an ambiguity, best first.
EMMA_AMBIGUOUS = {BY_TITLE: 0.45, AUTHOR: 0.4, "none": 0.0}


def _host(library):
    """What ``SubTaskProposal.revised`` reads from the session: the app workflow and the planner's LM."""
    return SimpleNamespace(_app_workflow=library, _current_planner_lm=None)


def test_what_the_user_set_in_the_same_reply_is_kept_over_the_new_draft(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: EMMA_AMBIGUOUS})
    scripted({"reasoning": "the title", "title": "Emma"}, {"reasoning": "the author", "author": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    reply = "change 1: means=planning:Library/author_books; change 1: author=Emma Woodhouse"
    # The policy is given the workflow, as the session gives it: the reply's input names are read from its plans.
    edits = ProposalReplyPolicy(app_workflow=library).decide(draft, reply).edits
    revised = apply_edits(draft, list(edits)).revised(_host(library), draft)
    assert revised.items[0].means.name == AUTHOR
    assert {each.name: each.value for each in revised.items[0].inputs}["author"] == "Emma Woodhouse"


def test_a_means_the_user_types_is_applied_and_the_inputs_are_re_drafted(library, scripted, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: EMMA_AMBIGUOUS})
    scripted({"reasoning": "the title", "title": "Emma"}, {"reasoning": "the title", "title": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    edits = ProposalReplyPolicy().decide(draft, f"change 1: means=planning:{LOANS}").edits
    revised = apply_edits(draft, list(edits)).revised(_host(library), draft)
    assert revised.items[0].means.name == LOANS


def test_a_no_agent_draft_binds_a_match(library, scripted, no_heads, monkeypatch):
    """A match is a decision, so it has no alternatives (bd fix-o1lg; a margin rule named the runner-up before)."""
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.93, AUTHOR: 0.07, "none": 0.0}})                  # a match (>= 0.9)
    scripted({"reasoning": "the title", "title": "Emma"})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    assert draft.items[0].means == Means(kind="planning", name=BY_TITLE)


def test_a_no_agent_draft_that_is_not_confident_is_ad_hoc(library, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.45, AUTHOR: 0.4, "none": 0.1}})                   # below the 0.5 match bar
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    assert draft.items[0].means.kind == "from_scratch"


def test_a_part_the_remote_model_says_nothing_fits_is_ad_hoc(library, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.05, AUTHOR: 0.04, "none": 0.91, LOANS: 0.0}})
    draft = review_proposals.draft_round1_proposal(library, [EMMA])
    assert draft.items[0].means.kind == "from_scratch"
