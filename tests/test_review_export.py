from __future__ import annotations

from fastworkflow.review.sidecar import REVIEW_DATABASE_NAME, ReviewSidecar


def _assignment() -> dict:
    return {
        "id": "assignment/export",
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": [],
        "blinded": True,
        "rows": [
            {"id": "row-1", "turn_ref": {"turn_key": "turn-1"}},
            {"id": "row-2", "turn_ref": {"turn_key": "turn-2"}},
        ],
        "questions": [
            {
                "id": "choice",
                "prompt": "Choose one.",
                "type": "single-select",
                "vocabulary": ["alpha", "beta"],
            },
            {
                "id": "note",
                "prompt": "Record a note.",
                "type": "bounded-note",
                "max_length": 40,
            },
        ],
    }


def test_export_is_assignment_shaped_latest_and_verbatim(tmp_path):
    sidecar = ReviewSidecar(
        tmp_path / REVIEW_DATABASE_NAME,
        canonical_manifest_digest="a" * 64,
        workspace_identity="workspace-1",
        evidence_store_digests={"store-a": "b" * 64},
    )
    capabilities = sidecar.create_assignment(_assignment())["rater_capabilities"]

    coordinates = [
        ("rater-a", "row-1", "choice", "alpha"),
        ("rater-a", "row-1", "note", "first"),
        ("rater-a", "row-2", "choice", "beta"),
        ("rater-a", "row-2", "note", "second"),
        ("rater-b", "row-1", "choice", "beta"),
        ("rater-b", "row-1", "note", "third"),
        ("rater-b", "row-2", "choice", "alpha"),
    ]
    for rater, row, question, answer in coordinates:
        sidecar.capture_answer(capabilities[rater], row, question, answer)
    sidecar.capture_answer(capabilities["rater-a"], "row-1", "choice", "beta")

    exported = sidecar.export_assignment("assignment/export")

    assert exported["assignment_id"] == "assignment/export"
    assert [row["turn_ref"] for row in exported["rows"]] == [
        {"turn_key": "turn-1"},
        {"turn_key": "turn-2"},
    ]
    assert exported["rows"][0]["rater_answers"][0]["answers"][0] == {
        "question_id": "choice",
        "revision": 2,
        "answer": "beta",
    }
    assert exported["unanswered"] == [
        {
            "row_id": "row-2",
            "rater_slot_id": "rater-b",
            "question_id": "note",
        }
    ]
    serialized_keys = {
        key
        for row in exported["rows"]
        for rater in row["rater_answers"]
        for answer in rater["answers"]
        for key in answer
    } | set(exported)
    assert serialized_keys.isdisjoint(
        {"kappa", "agreement", "score", "pass", "fail", "interpretation"}
    )
