from __future__ import annotations

import json
import sqlite3

import pytest

from fastworkflow.review.sidecar import (
    REVIEW_DATABASE_NAME,
    ReviewAuthorizationError,
    ReviewSidecar,
    ReviewValidationError,
    validate_review_assignment,
)


def _assignment() -> dict:
    return {
        "id": "assignment-1",
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": ["adjudicator"],
        "blinded": True,
        "rows": [
            {
                "id": "row-1",
                "turn_ref": {
                    "store_id": "store-a",
                    "logical_turn_key": "turn-1",
                },
            }
        ],
        "questions": [
            {
                "id": "verdict",
                "prompt": "Choose one outcome.",
                "type": "single-select",
                "vocabulary": ["left", "right"],
            },
            {
                "id": "flags",
                "prompt": "Choose any applicable values.",
                "type": "multi-select",
                "vocabulary": ["alpha", "beta"],
            },
            {
                "id": "note",
                "prompt": "Add a short note.",
                "type": "bounded-note",
                "max_length": 20,
            },
        ],
    }


def _sidecar(tmp_path) -> ReviewSidecar:
    return ReviewSidecar(
        tmp_path / REVIEW_DATABASE_NAME,
        canonical_manifest_digest="a" * 64,
        workspace_identity="workspace-1",
        evidence_store_digests={"store-a": "b" * 64},
    )


def test_two_raters_keep_distinct_answers_and_append_revisions(tmp_path):
    sidecar = _sidecar(tmp_path)
    created = sidecar.create_assignment(_assignment())
    first_token = created["rater_capabilities"]["rater-a"]
    second_token = created["rater_capabilities"]["rater-b"]

    sidecar.capture_answer(first_token, "row-1", "verdict", "left")
    sidecar.capture_answer(second_token, "row-1", "verdict", "right")
    sidecar.capture_answer(first_token, "row-1", "verdict", "right")

    current = sidecar.current_answers("assignment-1")
    assert {
        (row["rater_slot_id"], row["answer"], row["revision"]) for row in current
    } == {("rater-a", "right", 2), ("rater-b", "right", 1)}
    assert [row["answer"] for row in sidecar.answer_revisions("assignment-1")] == [
        "left",
        "right",
        "right",
    ]

    with sqlite3.connect(sidecar.db_path) as conn:
        stored_hashes = {
            row[0]
            for row in conn.execute("SELECT capability_hash FROM review_rater_slots")
        }
        assert first_token not in stored_hashes
        assert second_token not in stored_hashes
        assert all(len(value) == 64 for value in stored_hashes)


def test_capability_derives_slot_and_separates_adjudicator_role(tmp_path):
    sidecar = _sidecar(tmp_path)
    created = sidecar.create_assignment(_assignment())
    rater_token = created["rater_capabilities"]["rater-a"]
    adjudicator_token = created["adjudicator_capabilities"]["adjudicator"]

    with pytest.raises(ReviewAuthorizationError, match="adjudicator"):
        sidecar.capture_adjudication(rater_token, "row-1", "verdict", "left")
    with pytest.raises(ReviewAuthorizationError, match="rater"):
        sidecar.capture_answer(adjudicator_token, "row-1", "verdict", "left")

    captured = sidecar.capture_adjudication(
        adjudicator_token, "row-1", "verdict", "right"
    )
    assert captured["adjudicator_slot_id"] == "adjudicator"


def test_assignment_progress_returns_only_capability_latest_answers(tmp_path):
    sidecar = _sidecar(tmp_path)
    created = sidecar.create_assignment(_assignment())
    first_token = created["rater_capabilities"]["rater-a"]
    second_token = created["rater_capabilities"]["rater-b"]

    sidecar.capture_answer(first_token, "row-1", "verdict", "left")
    sidecar.capture_answer(first_token, "row-1", "verdict", "right")
    sidecar.capture_answer(second_token, "row-1", "verdict", "left")

    progress = sidecar.assignment_progress("assignment-1", first_token)

    assert progress["assignment"]["blinded"] is True
    assert progress["rater_slot_id"] == "rater-a"
    assert progress["answered_count"] == 1
    assert progress["total_answers"] == 3
    assert progress["current_answers"][0]["answer"] == "right"
    assert progress["current_answers"][0]["revision"] == 2
    assert {
        answer["rater_slot_id"] for answer in progress["current_answers"]
    } == {"rater-a"}


def test_assignment_validation_reports_reasons_and_refuses_bad_answers(tmp_path):
    invalid = _assignment()
    invalid["rows"] = [{"id": "row-1", "turn_ref": {"store_id": "store-a"}}]
    with pytest.raises(ReviewValidationError, match="logical_turn_key"):
        validate_review_assignment(invalid)

    invalid = _assignment()
    invalid["questions"][2].pop("max_length")
    with pytest.raises(ReviewValidationError, match="max_length"):
        validate_review_assignment(invalid)

    sidecar = _sidecar(tmp_path)
    token = sidecar.create_assignment(_assignment())["rater_capabilities"]["rater-a"]
    with pytest.raises(ReviewValidationError, match="outside"):
        sidecar.capture_answer(token, "row-1", "verdict", "undeclared")
    with pytest.raises(ReviewValidationError, match="max_length"):
        sidecar.capture_answer(token, "row-1", "note", "x" * 21)
    assert sidecar.answer_revisions("assignment-1") == []


def test_manifest_factory_binds_canonical_manifest_and_store_digests(tmp_path):
    manifest = {
        "schema": "fastworkflow-observability-workspace/1",
        "workspace_id": "workspace-1",
        "label": "Workspace",
        "stores": [
            {
                "store_id": "store-a",
                "path": "archive.sqlite3",
                "mode": "sealed",
                "sha256": "b" * 64,
            }
        ],
        "experiments": [],
        "projected_attempts": [],
    }
    manifest_path = tmp_path / "workspace.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    sidecar = ReviewSidecar.from_workspace_manifest(manifest_path)
    sidecar.create_assignment(_assignment())

    assert sidecar.db_path == str(tmp_path / REVIEW_DATABASE_NAME)
    with sqlite3.connect(sidecar.db_path) as conn:
        binding = conn.execute("""SELECT canonical_manifest_digest, workspace_identity,
                      evidence_store_digests_json
               FROM review_assignments""").fetchone()
    assert len(binding[0]) == 64
    assert binding[1] == "workspace-1"
    assert json.loads(binding[2]) == {"store-a": "b" * 64}


def test_closed_vocabulary_is_assignment_data_not_framework_vocabulary():
    normalized = validate_review_assignment(_assignment())

    assert normalized["questions"][0]["vocabulary"] == ["left", "right"]
    assert normalized["questions"][1]["vocabulary"] == ["alpha", "beta"]
