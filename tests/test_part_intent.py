"""``resolution.part_intent``: the one decision of which command or plan a part gets (bd fix-o1lg, steps S2 and S3).

Real components: the annotated library copy with its trained local intent models (the review gate's ``gated``
fixture). The remote decision model's answer is the one scripted seam (``script_matcher``, tests/conftest.py), so no
test here calls a model.
"""
from __future__ import annotations

import pytest

from fastworkflow import no_agent_mode, turn_cache
from fastworkflow._workflows.plan_metadata_extraction import resolution
from tests.conftest import script_matcher
from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401
from tests.test_round1_gate import FIND, JOKE, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

BY_TITLE = "Library/book_by_title"
AUTHOR = "Library/author_books"
LOANS = "Library/book_loans"


@pytest.fixture(autouse=True)
def a_turn():
    """Each test is one turn: the decision is memoised in the turn's memo."""
    turn_cache.begin_turn()


def test_an_exact_command_name_is_decided_with_no_model(gated, monkeypatch):
    asked = script_matcher(monkeypatch, lambda text: {"none": 1.0})
    intent = resolution.part_intent(gated.wf, "count_books")
    assert (intent.outcome, intent.command, intent.kind, intent.decided_by) == (
        resolution.MATCH, "Library/count_books", resolution.KIND_PLANNING, resolution.BY_EXACT_NAME)
    assert resolution.is_fast_path(intent) and asked == []


def test_a_confident_local_plain_command_wins_and_the_remote_model_is_not_asked(gated, monkeypatch):
    """The failure the decision fixes: a middling remote plan match no longer overrides a confident local answer."""
    asked = script_matcher(monkeypatch, {FIND: {BY_TITLE: 0.93, "none": 0.1}})
    intent = resolution.part_intent(gated.wf, FIND)
    assert (intent.outcome, intent.command, intent.kind, intent.decided_by) == (
        resolution.MATCH, "Library/find_member", resolution.KIND_EXECUTION, resolution.BY_CLASSIFIER)
    assert resolution.is_fast_path(intent) and asked == []


def test_an_unsure_local_model_hands_the_part_to_the_remote_model_whose_match_decides(gated, monkeypatch):
    asked = script_matcher(monkeypatch, {JOKE: {BY_TITLE: 0.93, AUTHOR: 0.05, "none": 0.02}})
    intent = resolution.part_intent(gated.wf, JOKE)
    assert (intent.outcome, intent.command, intent.kind, intent.decided_by) == (
        resolution.MATCH, BY_TITLE, resolution.KIND_PLANNING, resolution.BY_MATCHER)
    assert not resolution.is_fast_path(intent)                       # a remote match is drafted, as before
    assert asked == [JOKE]


def test_an_ambiguity_ranks_its_candidates_and_its_plans_order_the_closest(gated, monkeypatch):
    script_matcher(monkeypatch, {JOKE: {AUTHOR: 0.45, BY_TITLE: 0.3, LOANS: 0.25, "none": 0.0}})
    intent = resolution.part_intent(gated.wf, JOKE)
    assert intent.outcome == resolution.AMBIGUOUS and intent.decided_by == resolution.BY_QUESTION
    assert list(intent.ambiguous) == [AUTHOR, BY_TITLE, LOANS]
    assert [each.name for each in intent.candidates[:3]] == [AUTHOR, BY_TITLE, LOANS]
    assert resolution.draft_choice(intent) == (None, None)            # asked before the review, never guessed
    closest = [plan["name"] for plan in no_agent_mode.available_plans(gated.wf, intent.planning_scores())
               if plan["closest"]]
    assert closest == [AUTHOR, BY_TITLE, LOANS]


def test_a_remote_none_is_an_ad_hoc_part(gated, monkeypatch):
    script_matcher(monkeypatch, {JOKE: {"none": 0.95}})
    intent = resolution.part_intent(gated.wf, JOKE)
    assert intent.outcome == resolution.NONE and intent.command is None
    assert resolution.draft_choice(intent) == (None, None)


def test_a_failed_remote_call_leaves_the_local_models_unsure_answer_as_an_ambiguity_or_none(gated, monkeypatch):
    script_matcher(monkeypatch, {})                                  # every call fails
    intent = resolution.part_intent(gated.wf, JOKE)
    assert intent.outcome in (resolution.AMBIGUOUS, resolution.NONE)
    assert intent.decided_by != resolution.BY_MATCHER


def test_the_decision_is_made_once_per_part_and_turn(gated, monkeypatch):
    asked = script_matcher(monkeypatch, {JOKE: {BY_TITLE: 0.93, "none": 0.1}})
    first = resolution.part_intent(gated.wf, JOKE)
    assert resolution.part_intent(gated.wf, JOKE) == first
    assert resolution.part_intent(gated.wf, JOKE, exclude={LOANS}).command == BY_TITLE
    assert asked == [JOKE]
    turn_cache.begin_turn()                                          # a new turn decides again
    resolution.part_intent(gated.wf, JOKE)
    assert asked == [JOKE, JOKE]


def test_an_excluded_match_becomes_an_ambiguity_among_the_rest_or_none(gated, monkeypatch):
    script_matcher(monkeypatch, {JOKE: {BY_TITLE: 0.93, AUTHOR: 0.05, "none": 0.02}})
    rest = resolution.part_intent(gated.wf, JOKE, exclude={BY_TITLE})
    assert rest.outcome == resolution.AMBIGUOUS and BY_TITLE not in rest.ambiguous and rest.ambiguous
    assert all(each.name != BY_TITLE for each in rest.candidates)
    everything = {each.name for each in resolution.part_intent(gated.wf, JOKE).candidates}
    assert resolution.part_intent(gated.wf, JOKE, exclude=everything).outcome == resolution.NONE


def test_a_pinned_plan_is_the_answer_and_nothing_is_detected(gated, monkeypatch):
    asked = script_matcher(monkeypatch, lambda text: pytest.fail("a pinned plan must not be detected"))
    intent = resolution.part_intent(gated.wf, JOKE, pinned=BY_TITLE)
    assert (intent.outcome, intent.command, intent.kind, intent.decided_by) == (
        resolution.MATCH, BY_TITLE, resolution.KIND_PLANNING, resolution.BY_PINNED)
    assert asked == []


def test_a_blank_part_matches_nothing(gated, monkeypatch):
    asked = script_matcher(monkeypatch, lambda text: pytest.fail("a blank part must not be detected"))
    assert resolution.part_intent(gated.wf, "   ").outcome == resolution.NONE and asked == []


# ---- a confident local answer that names a PLANNING command is not a decision on its own (fix-o1lg.2) -------------
#
# The local model can be carried to a multi-step plan by a keyword. Its confident planning answer is checked by the
# remote decision model when the key is set, and is an ambiguity (asked) when it is not; it never takes the fast path.

WRITER = "who's the writer of Emma and how many do we own"           # the local model is confident: book_by_title


@pytest.mark.usefixtures("no_planner_key")
def test_with_no_key_a_local_plan_pick_is_an_ambiguity_never_the_fast_path(gated):
    intent = resolution.part_intent(gated.wf, WRITER)
    assert intent.outcome == resolution.AMBIGUOUS and intent.decided_by == resolution.BY_QUESTION
    assert intent.ambiguous[0] == BY_TITLE and len(intent.ambiguous) <= resolution.MAX_ALTERNATIVES
    assert not resolution.is_fast_path(intent)


def test_the_remote_model_checks_a_local_plan_pick_and_its_match_stands(gated, monkeypatch):
    asked = script_matcher(monkeypatch, {WRITER: {AUTHOR: 0.93, BY_TITLE: 0.04, "none": 0.03}})
    intent = resolution.part_intent(gated.wf, WRITER)
    assert (intent.outcome, intent.command, intent.kind, intent.decided_by) == (
        resolution.MATCH, AUTHOR, resolution.KIND_PLANNING, resolution.BY_MATCHER)
    assert not resolution.is_fast_path(intent) and asked == [WRITER]
    # The local answer is kept among the candidates, right after the decided command.
    assert [each.name for each in intent.candidates[:2]] == [AUTHOR, BY_TITLE]


def test_a_remote_none_on_a_local_plan_pick_keeps_the_local_plan_as_the_top_candidate(gated, monkeypatch):
    script_matcher(monkeypatch, {WRITER: {"none": 0.9}})
    intent = resolution.part_intent(gated.wf, WRITER)
    assert intent.outcome == resolution.NONE and intent.decided_by == resolution.BY_MATCHER
    assert intent.candidates[0].name == BY_TITLE
    assert next(iter(intent.planning_scores())) == BY_TITLE           # the closest plan an ad hoc part lists


def test_a_remote_ambiguity_on_a_local_plan_pick_asks_with_the_local_plan_first(gated, monkeypatch):
    script_matcher(monkeypatch, {WRITER: {AUTHOR: 0.4, LOANS: 0.3, "none": 0.1}})
    intent = resolution.part_intent(gated.wf, WRITER)
    assert intent.outcome == resolution.AMBIGUOUS and intent.ambiguous[0] == BY_TITLE
    assert set(intent.ambiguous[1:]) <= {AUTHOR, LOANS} and AUTHOR in intent.ambiguous


def test_a_failed_remote_check_of_a_local_plan_pick_is_an_ambiguity(gated, monkeypatch):
    script_matcher(monkeypatch, {})                                  # the call fails
    intent = resolution.part_intent(gated.wf, WRITER)
    assert intent.outcome == resolution.AMBIGUOUS and intent.ambiguous[0] == BY_TITLE
    assert not resolution.is_fast_path(intent)


def test_a_confident_local_execution_command_still_decides_alone(gated, monkeypatch):
    asked = script_matcher(monkeypatch, lambda text: pytest.fail("the remote model must not be asked"))
    intent = resolution.part_intent(gated.wf, FIND)
    assert intent.kind == resolution.KIND_EXECUTION and resolution.is_fast_path(intent) and asked == []


# ---- commands that only enter a context are not offered to the remote model (fix-o1lg.2) ---------------------------

BRANCHES = "Library/list_branches"


def test_the_remote_model_is_never_offered_a_command_that_only_enters_a_context(gated, monkeypatch):
    """A request that names a workspace by its subject goes to the command or plan that serves it, not to the opener:
    the remote link's options leave the entry-only commands out, and an answer naming one cannot win."""
    from fastworkflow import intent_detector, intent_loop

    monkeypatch.setattr(intent_loop, "_entry_only", lambda workflow: frozenset({BRANCHES}))
    offered: list = []
    monkeypatch.setattr(intent_detector, "matcher_available", lambda: True)

    def answer(folder, scope_path, utterance, exclude=frozenset()):
        offered.append({c.name for _, group in intent_detector._matcher_groups(folder, scope_path, exclude)
                        for c in group})
        return {"scores": {BRANCHES: 0.9, BY_TITLE: 0.93, "none": 0.0}, "model": "scripted", "latency_ms": 0.0,
                "calls": 1}

    monkeypatch.setattr(intent_detector, "_matcher_distribution", answer)
    intent = resolution.part_intent(gated.wf, JOKE)
    assert offered and all(BRANCHES not in names for names in offered)
    assert all(BY_TITLE in names for names in offered)
    assert (intent.outcome, intent.command) == (resolution.MATCH, BY_TITLE)
    # An exact name still wins.
    assert resolution.part_intent(gated.wf, "list_branches").command == BRANCHES


def test_an_entry_only_command_is_one_that_always_descends_and_takes_no_required_input(tmp_path):
    """The marker: ``@descends_to`` with no ``when_parameter``, on a command with no required input. ``open_book`` and
    ``open_member`` descend but need a record id, so they are not entry-only."""
    import os
    import shutil

    import fastworkflow
    from fastworkflow import context_navigation

    source = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")
    folder = tmp_path / "library"
    shutil.copytree(source, folder, symlinks=True, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    assert context_navigation.context_entry_commands(str(folder)) == frozenset()
    path = folder / "_commands" / "Library" / "list_branches.py"
    text = path.read_text()
    assert "class ResponseGenerator:\n" in text
    path.write_text(text.replace("class ResponseGenerator:\n",
                                 '@fastworkflow.descends_to("Book")\nclass ResponseGenerator:\n', 1))
    fastworkflow.RoutingRegistry.clear_registry()
    assert context_navigation.context_entry_commands(str(folder)) == frozenset({BRANCHES})
