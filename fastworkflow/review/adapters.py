"""Lossless JSON adapters for versioned formal-review contracts.

The adapters only rename and regroup sidecar fields. They deliberately do not
interpret answers, calculate scores, or import application-domain packages.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

IDO_RATING_SCHEMA = "ido-rating-v1"
EXP028_ANSWER_RATING_SCHEMA = "exp028-answer-rating-v1"

_IDO_FIELDS = (
    "primary",
    "contributing_causes",
    "coverage",
    "declines",
    "note",
)
_EXP028_FIELDS = (
    "presented",
    "not_presented",
    "overclaimed",
    "not_decidable",
)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_contract_questions(
    export: dict[str, Any], fields: tuple[str, ...]
) -> None:
    question_ids = {
        question.get("id")
        for question in export.get("questions", [])
        if isinstance(question, dict)
    }
    missing = [field for field in fields if field not in question_ids]
    if missing:
        raise ValueError(
            "sidecar export is missing contract question ids: " + ", ".join(missing)
        )


def _revision_payload(revision: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in revision.items()
        if key
        not in {
            "assignment_id",
            "row_id",
            "rater_slot_id",
            "adjudicator_slot_id",
            "question_id",
        }
    }


def _group_revisions(
    revisions: list[dict[str, Any]],
    *,
    row_id: str,
    slot_key: str,
    slot_id: str,
    fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {field: [] for field in fields}
    for revision in revisions:
        if revision.get("row_id") != row_id or revision.get(slot_key) != slot_id:
            continue
        question_id = revision.get("question_id")
        if question_id in grouped:
            grouped[question_id].append(_revision_payload(revision))
    return grouped


def _sidecar_export_to_contract(
    sidecar_export: dict[str, Any],
    *,
    schema: str,
    fields: tuple[str, ...],
    human_key: str,
    adjudicator_key: str,
) -> dict[str, Any]:
    export = _require_mapping(sidecar_export, "sidecar export")
    _require_contract_questions(export, fields)
    answer_revisions = list(export.get("answer_revisions", []))
    adjudication_revisions = list(export.get("adjudication_revisions", []))
    rater_slots = list(export.get("rater_slots", []))
    adjudicator_slots = list(export.get("adjudicator_slots", []))
    rows = []
    for row in export.get("rows", []):
        row_id = row["id"]
        human_reviews = []
        for rater_slot_id in rater_slots:
            human_reviews.append(
                {
                    "rater": rater_slot_id,
                    **_group_revisions(
                        answer_revisions,
                        row_id=row_id,
                        slot_key="rater_slot_id",
                        slot_id=rater_slot_id,
                        fields=fields,
                    ),
                }
            )
        adjudications = []
        for adjudicator_slot_id in adjudicator_slots:
            adjudications.append(
                {
                    "adjudicator": adjudicator_slot_id,
                    "provenance": {"rater_slots": deepcopy(rater_slots)},
                    **_group_revisions(
                        adjudication_revisions,
                        row_id=row_id,
                        slot_key="adjudicator_slot_id",
                        slot_id=adjudicator_slot_id,
                        fields=fields,
                    ),
                }
            )
        rows.append(
            {
                "row_id": row_id,
                "turn_ref": deepcopy(row["turn_ref"]),
                human_key: human_reviews,
                adjudicator_key: adjudications,
            }
        )
    return {
        "schema": schema,
        "assignment_id": export["assignment_id"],
        "blinded": export["blinded"],
        "rater_slots": rater_slots,
        "adjudicator_slots": adjudicator_slots,
        "questions": deepcopy(export["questions"]),
        "rows": rows,
        "unanswered": deepcopy(export.get("unanswered", [])),
    }


def _flatten_contract_revisions(
    rows: list[dict[str, Any]],
    *,
    collection_key: str,
    name_key: str,
    slot_key: str,
    fields: tuple[str, ...],
    assignment_id: str,
) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        for reviewer in row.get(collection_key, []):
            slot_id = reviewer[name_key]
            for question_id in fields:
                for revision in reviewer.get(question_id, []):
                    flattened.append(
                        {
                            "assignment_id": assignment_id,
                            "row_id": row["row_id"],
                            slot_key: slot_id,
                            "question_id": question_id,
                            **deepcopy(revision),
                        }
                    )
    flattened.sort(
        key=lambda value: (
            value["row_id"],
            value[slot_key],
            value["question_id"],
            value["revision"],
        )
    )
    return flattened


def _latest_answers(
    revisions: list[dict[str, Any]],
    *,
    row_id: str,
    slot_key: str,
    slot_id: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = {}
    for revision in revisions:
        if revision["row_id"] == row_id and revision[slot_key] == slot_id:
            question_id = revision["question_id"]
            if (
                question_id not in latest
                or revision["revision"] > latest[question_id]["revision"]
            ):
                latest[question_id] = revision
    return [
        {
            "question_id": question["id"],
            "revision": latest[question["id"]]["revision"],
            "answer": deepcopy(latest[question["id"]]["answer"]),
        }
        for question in questions
        if question["id"] in latest
    ]


def _contract_to_sidecar_export(
    document: dict[str, Any],
    *,
    schema: str,
    fields: tuple[str, ...],
    human_key: str,
    adjudicator_key: str,
) -> dict[str, Any]:
    contract = _require_mapping(document, schema)
    if contract.get("schema") != schema:
        raise ValueError(f"expected schema {schema!r}")
    rows = list(contract.get("rows", []))
    assignment_id = contract["assignment_id"]
    questions = deepcopy(contract["questions"])
    answer_revisions = _flatten_contract_revisions(
        rows,
        collection_key=human_key,
        name_key="rater",
        slot_key="rater_slot_id",
        fields=fields,
        assignment_id=assignment_id,
    )
    adjudication_revisions = _flatten_contract_revisions(
        rows,
        collection_key=adjudicator_key,
        name_key="adjudicator",
        slot_key="adjudicator_slot_id",
        fields=fields,
        assignment_id=assignment_id,
    )
    sidecar_rows = []
    for row in rows:
        sidecar_rows.append(
            {
                "id": row["row_id"],
                "turn_ref": deepcopy(row["turn_ref"]),
                "rater_answers": [
                    {
                        "rater_slot_id": slot_id,
                        "answers": _latest_answers(
                            answer_revisions,
                            row_id=row["row_id"],
                            slot_key="rater_slot_id",
                            slot_id=slot_id,
                            questions=questions,
                        ),
                    }
                    for slot_id in contract["rater_slots"]
                ],
                "adjudicator_answers": [
                    {
                        "adjudicator_slot_id": slot_id,
                        "answers": _latest_answers(
                            adjudication_revisions,
                            row_id=row["row_id"],
                            slot_key="adjudicator_slot_id",
                            slot_id=slot_id,
                            questions=questions,
                        ),
                    }
                    for slot_id in contract["adjudicator_slots"]
                ],
            }
        )
    return {
        "assignment_id": assignment_id,
        "rater_slots": deepcopy(contract["rater_slots"]),
        "adjudicator_slots": deepcopy(contract["adjudicator_slots"]),
        "blinded": contract["blinded"],
        "questions": questions,
        "rows": sidecar_rows,
        "unanswered": deepcopy(contract.get("unanswered", [])),
        "answer_revisions": answer_revisions,
        "adjudication_revisions": adjudication_revisions,
    }


def sidecar_export_to_ido_rating(sidecar_export: dict[str, Any]) -> dict[str, Any]:
    """Convert a sidecar export to the ``ido-rating-v1`` JSON contract."""
    return _sidecar_export_to_contract(
        sidecar_export,
        schema=IDO_RATING_SCHEMA,
        fields=_IDO_FIELDS,
        human_key="ratings",
        adjudicator_key="adjudications",
    )


def ido_rating_to_sidecar_export(document: dict[str, Any]) -> dict[str, Any]:
    """Convert an ``ido-rating-v1`` JSON document to a sidecar export."""
    return _contract_to_sidecar_export(
        document,
        schema=IDO_RATING_SCHEMA,
        fields=_IDO_FIELDS,
        human_key="ratings",
        adjudicator_key="adjudications",
    )


def sidecar_export_to_exp028_answer_rating(
    sidecar_export: dict[str, Any],
) -> dict[str, Any]:
    """Convert a sidecar export to ``exp028-answer-rating-v1`` JSON."""
    return _sidecar_export_to_contract(
        sidecar_export,
        schema=EXP028_ANSWER_RATING_SCHEMA,
        fields=_EXP028_FIELDS,
        human_key="human_attestations",
        adjudicator_key="adjudicator_attestations",
    )


def exp028_answer_rating_to_sidecar_export(
    document: dict[str, Any],
) -> dict[str, Any]:
    """Convert ``exp028-answer-rating-v1`` JSON to a sidecar export."""
    return _contract_to_sidecar_export(
        document,
        schema=EXP028_ANSWER_RATING_SCHEMA,
        fields=_EXP028_FIELDS,
        human_key="human_attestations",
        adjudicator_key="adjudicator_attestations",
    )
