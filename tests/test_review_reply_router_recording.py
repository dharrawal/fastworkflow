"""The recorded answer of the consultation head (tests/recorded_router.py): that it is sound, and that the head still gives it.

The reader tests (tests/test_consultation_reply_policy.py, tests/test_built_in_vague_replies.py,
tests/test_round1_handoff.py) feed the reader the head's answer for each reply from
``tests/fixtures/review_reply_router_answers.json``, so they need no trained head. The drift test below is the one
that needs it: it asks the real head every recorded reply and fails, naming the replies, when a retrain changed a
reader-relevant answer.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow.model_pipeline_training import CommandRouter
from tests import consultation_head as head
from tests import recorded_router
from tests.recorded_router import RecordedRouter, UnrecordedUtterance, installed, load_recording
from tests.test_consultation_reply_policy import (  # noqa: F401  (the fixtures the reader tests use)
    _open_review_two_step, annotated_library, gated, initialized_fastworkflow, no_draft_extraction, no_live_model,
    script_param_extraction, todo)


def test_the_recording_names_its_head_and_holds_answers_as_predict_with_details_returns_them():
    recording = load_recording()
    assert recording["metadata"]["head_version_id"] and recording["metadata"]["recorded_at"]
    assert recording["answers"]
    for reply, answer in recording["answers"].items():
        assert isinstance(reply, str) and reply
        assert answer["labels"] and all(isinstance(label, str) for label in answer["labels"])
        details = answer["details"]
        assert {"confident", "top_label", "topk_labels", "confidence"} <= set(details), reply
        assert answer["labels"] == ([details["top_label"]] if details["confident"] else details["topk_labels"]), reply
        json.dumps(answer)


def test_a_reply_with_no_recording_fails_loudly_instead_of_being_guessed():
    router = RecordedRouter.from_recording({"answers": {}})
    with pytest.raises(UnrecordedUtterance, match="no recorded head answer"):
        router.predict_with_details("a reply nobody recorded")
    assert router.unrecorded == ["a reply nobody recorded"]


def test_the_router_cache_entry_is_put_back_whatever_was_there():
    cache = CommandRouter._instances_cache
    key = recorded_router.ROUTER_PATH
    for prior in (None, object.__new__(RecordedRouter)):
        if prior is None:
            cache.pop(key, None)
        else:
            cache[key] = prior
        with installed(RecordedRouter.from_recording({"answers": {}})) as router:
            assert cache[key] is router
        assert cache.get(key) is prior
    cache.pop(key, None)


@pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction")
def test_the_reader_asks_the_recorded_router_at_the_key_production_looks_up(gated, monkeypatch, recorded_review_router):
    """If the recording were installed under another key the reader would reach for a real head; prove it does not."""
    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx.process_turn("look up step 1")
    assert "look up step 1" in recorded_review_router.asked
    assert record.state == "revised"


@pytest.mark.requires_consultation_head
def test_recorded_router_matches_the_head():
    """Drift: the current head answers every recorded reply as it was recorded (label and ambiguity flag)."""
    recording = load_recording()
    real = CommandRouter(head.HEAD_DIR)
    drifted = []
    for reply, recorded in recording["answers"].items():
        _labels, details = real.predict_with_details(reply)
        was = (recorded["details"]["top_label"], bool(recorded["details"]["confident"]))
        now = (str(details["top_label"]), bool(details["confident"]))
        if was != now:
            drifted.append(f"{reply!r}: recorded (top_label, confident)={was}, head now {now}")
    assert not drifted, (
        f"the head (current version {head.head_version_id()}) no longer answers {len(drifted)} of "
        f"{len(recording['answers'])} recorded replies as version {recording['metadata']['head_version_id']} did, "
        "so the reader tests run on stale routing. Re-record with `pytest --record-review-router "
        "tests/test_consultation_reply_policy.py tests/test_built_in_vague_replies.py tests/test_round1_handoff.py` "
        "and review the diff of tests/fixtures/review_reply_router_answers.json.\n" + "\n".join(drifted)
    )
