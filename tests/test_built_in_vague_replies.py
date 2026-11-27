"""bd fix-1hs8: a head with no foreign command is seeded with the built-in vague replies.

Pure tests (nothing trains, no model loads): seed selection on private copies of real workflows,
the training signature, and the review reader's handling of a wildcard answer that names no command.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.review import review_engine
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    BUILT_IN_VAGUE_REPLIES,
    BUILT_IN_VAGUE_SOURCE,
    not_here_samples_for_context,
    not_here_seed_rows,
    select_escalation_rows,
    trained_command_labels,
)
from fastworkflow.train import selective_training
from tests.test_not_here_class import MESSAGING, _private_copy

PLAN_WORKFLOW = fastworkflow.get_internal_workflow_path("plan_metadata_extraction")

# The owner's held-out vague replies and the vague strings the reply tests use. The built-in list
# must not contain or repeat them, or the held-out measurement would measure memorisation.
HELD_OUT_VAGUE = {
    "i guess, maybe", "what do you think?", "whatever works", "hmm", "can you do something about it",
    "not sure what you mean", "it depends", "i don't know", "ehh kind of", "tweak it a bit",
    "could be better", "let me think", "huh?", "that one maybe", "something feels off", "improve it",
    "is that right?", "possibly", "do the usual", "asdf", "make it better somehow", "hmm not sure",
    "unclear one", "unclear two",
}


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def plan_copy(tmp_path):
    path = _private_copy(tmp_path, PLAN_WORKFLOW, "plan_copy")
    yield path
    RoutingRegistry.clear_registry()


@pytest.fixture
def messaging_copy(tmp_path):
    path = _private_copy(tmp_path, MESSAGING, "messaging_copy")
    yield path
    RoutingRegistry.clear_registry()


def test_the_built_in_list_is_small_diverse_and_clear_of_the_held_out_replies():
    assert 8 <= len(BUILT_IN_VAGUE_REPLIES) <= 9
    lowered = [text.lower() for text in BUILT_IN_VAGUE_REPLIES]
    assert len(set(lowered)) == len(lowered)
    assert not set(lowered) & HELD_OUT_VAGUE
    corpus = os.path.join(PLAN_WORKFLOW, "heldout_reply_corpus.jsonl")
    with open(corpus, encoding="utf-8") as handle:
        texts = {json.loads(line)["text"].strip().lower() for line in handle if line.strip()}
    assert not set(lowered) & texts


def test_a_single_context_workflow_without_foreign_commands_gets_the_built_in_class(plan_copy):
    crd = RoutingRegistry.get_definition(plan_copy)
    workflow = fastworkflow.Workflow.create(plan_copy, workflow_id_str="vague-seed-single")
    ctx = "ProposalReview"
    selection = select_escalation_rows(ctx, crd, workflow, context_utterances=set())
    assert selection.included
    assert selection.rows == sorted(BUILT_IN_VAGUE_REPLIES)
    assert selection.not_here_samples == {BUILT_IN_VAGUE_SOURCE: list(BUILT_IN_VAGUE_REPLIES)}
    assert not_here_samples_for_context(plan_copy, crd, ctx) == selection.not_here_samples


def test_a_built_in_row_that_is_one_of_the_heads_own_utterances_is_not_taken_twice(plan_copy):
    crd = RoutingRegistry.get_definition(plan_copy)
    workflow = fastworkflow.Workflow.create(plan_copy, workflow_id_str="vague-seed-own")
    ctx = "ProposalReview"
    own = {BUILT_IN_VAGUE_REPLIES[0]}
    selection = select_escalation_rows(ctx, crd, workflow, context_utterances=own)
    assert BUILT_IN_VAGUE_REPLIES[0] not in selection.rows


def test_heads_that_already_have_foreign_seeds_are_untouched(messaging_copy):
    crd = RoutingRegistry.get_definition(messaging_copy)
    workflow = fastworkflow.Workflow.create(messaging_copy, workflow_id_str="vague-seed-multi")
    checked = 0
    for ctx in sorted(crd.contexts):
        picks, _available = not_here_seed_rows(
            messaging_copy, crd, trained_command_labels(ctx, crd), 1, context_name=ctx)
        foreign_rows = {row for rows in picks.values() for row in rows}
        if not foreign_rows:
            continue
        checked += 1
        selection = select_escalation_rows(ctx, crd, workflow, context_utterances=set())
        assert BUILT_IN_VAGUE_SOURCE not in picks
        assert selection.rows == sorted(foreign_rows)
        assert BUILT_IN_VAGUE_SOURCE not in (selection.not_here_samples or {})
        assert not set(selection.rows) & set(BUILT_IN_VAGUE_REPLIES)
    assert checked


def test_the_internal_cme_arm_builds_no_class_and_no_built_in_rows():
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    crd = RoutingRegistry.get_definition(cme)
    picks, _available = not_here_seed_rows(cme, crd, set(), 1)
    assert BUILT_IN_VAGUE_SOURCE not in picks
    assert not_here_samples_for_context(cme, crd, sorted(crd.contexts)[0]) is None


def test_the_signature_records_the_built_in_source_so_editing_the_list_retrains(plan_copy):
    signature, _unresolved = selective_training.compute_training_signature(
        plan_copy, ["ProposalReview"])
    head = signature.contexts["ProposalReview"]
    assert head.wildcard_sources == [BUILT_IN_VAGUE_SOURCE]
    assert head.expects_wildcard_label
    assert head.not_here_samples == {BUILT_IN_VAGUE_SOURCE: list(BUILT_IN_VAGUE_REPLIES)}
    assert BUILT_IN_VAGUE_SOURCE in signature.command_fingerprints


def test_a_multi_context_signature_does_not_mention_the_built_in_source(messaging_copy):
    crd = RoutingRegistry.get_definition(messaging_copy)
    signature, _unresolved = selective_training.compute_training_signature(
        messaging_copy, sorted(crd.contexts))
    assert BUILT_IN_VAGUE_SOURCE not in signature.command_fingerprints
    assert all(BUILT_IN_VAGUE_SOURCE not in c.wildcard_sources for c in signature.contexts.values())


# ---------------------------------------------------------------- the reader

@pytest.mark.parametrize("routed,name,expected", [
    (True, "ProposalReview/decline", True),
    (True, None, False),     # "nope": routed flag set, no command named
    (True, "", False),
    (False, "ProposalReview/decline", False),
    (False, None, False),
])
def test_a_routed_flag_without_a_command_name_is_unrouted(routed, name, expected):
    out = SimpleNamespace(routed=routed, command_name=name)
    assert review_engine._routed_to_command(out) is expected


# ---------------------------------------------------------------- the reader, against the head

from tests.plan_head import HEAD_DIR as _HEAD_DIR, head_trained  # noqa: E402
from tests.test_plan_reply_policy import (  # noqa: E402,F401  (fixtures the reader tests use)
    _PlannerLM,
    _open_review_two_step,
    annotated_library,
    gated,
    initialized_fastworkflow as _reader_init,
    no_draft_extraction,
    no_live_model,
    script_param_extraction,
    todo,
)


def _head_has_wildcard_class() -> bool:
    if not head_trained():
        return False
    from fastworkflow.model_pipeline_training import CommandRouter

    router = CommandRouter(_HEAD_DIR)
    return any(
        str(router.predict_with_details(text)[1].get("top_label")) == "wildcard"
        for text in ("what do you think?", "asdf", "something feels off", "tweak it a bit")
    )


def _reply(gated, monkeypatch, text):
    from fastworkflow.review.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx._current_planner_lm = _PlannerLM([{"edits": []}] * 30)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    return record, review


class _ReaderCases:
    """What the review does with a vague or non-decision reply. The bodies are shared by two classes: one feeds the
    reader the head's RECORDED answers (reader behaviour, runs everywhere), the other the real trained head (the head
    asks again, which needs the head)."""

    def test_the_review_record_is_the_root_so_a_wildcard_answer_has_somewhere_to_walk(self, gated):
        workflow = gated.ctx._plan_workflow
        assert workflow.root is not None
        assert workflow._current_command_context is workflow.root

    @pytest.mark.parametrize("text", ["nope", "what do you think?", "I don't know", "asdf"])
    def test_no_reply_raises_and_a_non_decision_never_edits(self, gated, monkeypatch, text):
        """"nope" used to raise ValueError (routed flag, no command name); the vague ones must ask."""
        record, review = _reply(gated, monkeypatch, text)
        assert record.state not in (review_engine.REVISED,)

    @pytest.mark.parametrize("text", ["what do you think?", "I don't know", "something feels off"])
    def test_a_vague_reply_asks_again_and_never_accepts_or_declines(self, gated, monkeypatch, text):
        record, review = _reply(gated, monkeypatch, text)
        assert not review.done
        assert record.state not in (review_engine.CONFIRMED, review_engine.DECLINED,
                                    review_engine.REVISED)
        assert review.text == review_engine.SUB_TASK_UNREADABLE_REPLY


@pytest.mark.usefixtures("_reader_init", "no_live_model", "no_draft_extraction", "recorded_review_router")
class TestReaderOnRecordedHeadAnswers(_ReaderCases):
    """Reader behaviour: the head's answer for each reply is the recording (tests/recorded_router.py). No head needed."""


@pytest.mark.requires_plan_head
@pytest.mark.usefixtures("_reader_init", "no_live_model", "no_draft_extraction")
class TestReaderWithTrainedHead(_ReaderCases):
    """Head behaviour: the same replies asked of the trained head, which must route the vague ones to a wildcard
    answer the reader turns into a question. A head with no wildcard class cannot, so it is reported, not skipped
    silently, when heads are required."""

    @pytest.fixture(autouse=True)
    def _wildcard_class(self, request):
        # An absent head is the marker's to report (tests/conftest.py); this is about a head that is there.
        if head_trained() and request.function.__name__ == "test_a_vague_reply_asks_again_and_never_accepts_or_declines" \
                and not _head_has_wildcard_class():
            from tests import plan_head

            message = "this head has no wildcard class (retrain it with the built-in vague replies)"
            if plan_head.heads_required():
                pytest.fail(f"{plan_head.REQUIRE_HEADS_ENV} is set and {message}", pytrace=False)
            pytest.skip(message)
