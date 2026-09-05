"""Domain-neutral, capability-gated formal review sidecar.

Formal review is intentionally separate from ``ObservabilityStore.feedback``:
feedback is mutable agent memory, while this database records independent
rater slots, immutable answer revisions, and separately attributed
adjudications.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REVIEW_DATABASE_NAME = "observability-reviews.sqlite3"
REVIEW_SCHEMA = "fastworkflow-observability-review/1"
_QUESTION_TYPES = frozenset({"single-select", "multi-select", "bounded-note"})


class ReviewValidationError(ValueError):
    """A review assignment, rubric, binding, or captured answer is invalid."""


class ReviewAuthorizationError(PermissionError):
    """A capability is invalid or is not authorized for the requested action."""


class ReviewNotFoundError(KeyError):
    """A referenced assignment, row, or question does not exist."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError(f"document must be JSON-native: {exc}") from exc


def canonical_document_digest(value: Any) -> str:
    """Return SHA-256 over the canonical JSON encoding of a document."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def review_database_path(workspace_manifest_path: str | Path) -> Path:
    """Return the fixed review database path beside a workspace manifest."""
    return Path(workspace_manifest_path).resolve().parent / REVIEW_DATABASE_NAME


def _normalize_slots(raw: Any, field: str, *, minimum: int) -> list[str]:
    if not isinstance(raw, list) or len(raw) < minimum:
        raise ReviewValidationError(
            f"{field} must be an array containing at least {minimum} slot(s)"
        )
    slots: list[str] = []
    for index, value in enumerate(raw):
        if isinstance(value, dict):
            value = value.get("id", value.get("slot_id"))
        slots.append(_required_text(value, f"{field}[{index}].id"))
    if len(set(slots)) != len(slots):
        raise ReviewValidationError(f"{field} contains duplicate slot ids")
    return slots


def _normalize_turn_ref(row: dict[str, Any], field: str) -> dict[str, str]:
    raw = row.get("turn_ref")
    if raw is None:
        raw = {
            key: row[key]
            for key in ("store_id", "logical_turn_key", "turn_key")
            if key in row
        }
    if not isinstance(raw, dict):
        raise ReviewValidationError(f"{field}.turn_ref must be an object")
    has_workspace_field = "store_id" in raw or "logical_turn_key" in raw
    if has_workspace_field:
        if raw.get("turn_key") is not None:
            raise ReviewValidationError(
                f"{field}.turn_ref must use either workspace keys or turn_key"
            )
        return {
            "store_id": _required_text(
                raw.get("store_id"), f"{field}.turn_ref.store_id"
            ),
            "logical_turn_key": _required_text(
                raw.get("logical_turn_key"),
                f"{field}.turn_ref.logical_turn_key",
            ),
        }
    return {
        "turn_key": _required_text(raw.get("turn_key"), f"{field}.turn_ref.turn_key")
    }


def _normalize_question(raw: Any, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ReviewValidationError(f"{field} must be an object")
    question_id = _required_text(raw.get("id", raw.get("question_id")), f"{field}.id")
    prompt = _required_text(raw.get("prompt"), f"{field}.prompt")
    question_type = _required_text(raw.get("type"), f"{field}.type")
    if question_type not in _QUESTION_TYPES:
        raise ReviewValidationError(
            f"{field}.type must be one of {sorted(_QUESTION_TYPES)}"
        )
    normalized: dict[str, Any] = {
        "id": question_id,
        "prompt": prompt,
        "type": question_type,
    }
    vocabulary = raw.get("vocabulary", raw.get("options"))
    if question_type in {"single-select", "multi-select"}:
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or not all(isinstance(value, str) and value for value in vocabulary)
        ):
            raise ReviewValidationError(
                f"{field}.vocabulary must be a non-empty array of strings"
            )
        if len(set(vocabulary)) != len(vocabulary):
            raise ReviewValidationError(f"{field}.vocabulary contains duplicate values")
        normalized["vocabulary"] = list(vocabulary)
        if "max_length" in raw or "max_chars" in raw:
            raise ReviewValidationError(
                f"{field} cannot set a note bound for a closed-vocabulary question"
            )
    else:
        if vocabulary is not None:
            raise ReviewValidationError(
                f"{field} cannot declare vocabulary for a bounded note"
            )
        maximum = raw.get("max_length", raw.get("max_chars"))
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ReviewValidationError(
                f"{field}.max_length must be a positive integer"
            )
        normalized["max_length"] = maximum
    return normalized


def validate_review_assignment(document: Any) -> dict[str, Any]:
    """Validate and normalize a domain-neutral review assignment document."""
    if not isinstance(document, dict):
        raise ReviewValidationError("assignment must be a JSON object")
    assignment_id = _required_text(
        document.get("id", document.get("assignment_id")), "assignment.id"
    )
    blinded = document.get("blinded", document.get("blinding"))
    if not isinstance(blinded, bool):
        raise ReviewValidationError("assignment.blinded must be a boolean")
    rater_slots = _normalize_slots(
        document.get("rater_slots"), "assignment.rater_slots", minimum=2
    )
    adjudicator_slots = _normalize_slots(
        document.get("adjudicator_slots", []),
        "assignment.adjudicator_slots",
        minimum=0,
    )
    if set(rater_slots) & set(adjudicator_slots):
        raise ReviewValidationError("rater and adjudicator slot ids must be distinct")

    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ReviewValidationError("assignment.rows must be a non-empty array")
    rows = []
    row_ids: set[str] = set()
    for index, raw in enumerate(raw_rows):
        field = f"assignment.rows[{index}]"
        if not isinstance(raw, dict):
            raise ReviewValidationError(f"{field} must be an object")
        row_id = _required_text(raw.get("id", raw.get("row_id")), f"{field}.id")
        if row_id in row_ids:
            raise ReviewValidationError(f"duplicate row id {row_id!r}")
        row_ids.add(row_id)
        rows.append({"id": row_id, "turn_ref": _normalize_turn_ref(raw, field)})

    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ReviewValidationError("assignment.questions must be a non-empty array")
    questions = []
    question_ids: set[str] = set()
    for index, raw in enumerate(raw_questions):
        question = _normalize_question(raw, f"assignment.questions[{index}]")
        if question["id"] in question_ids:
            raise ReviewValidationError(f"duplicate question id {question['id']!r}")
        question_ids.add(question["id"])
        questions.append(question)

    return {
        "schema": REVIEW_SCHEMA,
        "id": assignment_id,
        "rater_slots": rater_slots,
        "adjudicator_slots": adjudicator_slots,
        "blinded": blinded,
        "rows": rows,
        "questions": questions,
    }


def _validate_digest(value: Any, field: str) -> str:
    digest = _required_text(value, field).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ReviewValidationError(f"{field} must be 64 hexadecimal digits")
    return digest


def _validate_binding(
    canonical_manifest_digest: Any,
    workspace_identity: Any,
    evidence_store_digests: Any,
) -> tuple[str, str, dict[str, str]]:
    manifest_digest = _validate_digest(
        canonical_manifest_digest, "canonical_manifest_digest"
    )
    identity = _required_text(workspace_identity, "workspace_identity")
    if not isinstance(evidence_store_digests, dict) or not evidence_store_digests:
        raise ReviewValidationError("evidence_store_digests must be a non-empty object")
    digests = {
        _required_text(store_id, "evidence_store_digests key"): _validate_digest(
            digest, f"evidence_store_digests[{store_id!r}]"
        )
        for store_id, digest in evidence_store_digests.items()
    }
    return manifest_digest, identity, digests


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS review_assignments (
        assignment_id TEXT PRIMARY KEY,
        schema TEXT NOT NULL,
        canonical_manifest_digest TEXT NOT NULL,
        workspace_identity TEXT NOT NULL,
        evidence_store_digests_json TEXT NOT NULL,
        rubric_json TEXT NOT NULL,
        blinded INTEGER NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS review_rows (
        assignment_id TEXT NOT NULL,
        row_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        store_id TEXT,
        logical_turn_key TEXT,
        turn_key TEXT,
        PRIMARY KEY (assignment_id, row_id),
        FOREIGN KEY (assignment_id) REFERENCES review_assignments(assignment_id),
        CHECK (
          (store_id IS NOT NULL AND logical_turn_key IS NOT NULL AND turn_key IS NULL)
          OR
          (store_id IS NULL AND logical_turn_key IS NULL AND turn_key IS NOT NULL)))""",
    """CREATE TABLE IF NOT EXISTS review_questions (
        assignment_id TEXT NOT NULL,
        question_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        prompt TEXT NOT NULL,
        question_type TEXT NOT NULL,
        vocabulary_json TEXT,
        note_max_length INTEGER,
        PRIMARY KEY (assignment_id, question_id),
        FOREIGN KEY (assignment_id) REFERENCES review_assignments(assignment_id))""",
    """CREATE TABLE IF NOT EXISTS review_rater_slots (
        assignment_id TEXT NOT NULL,
        rater_slot_id TEXT NOT NULL,
        role TEXT NOT NULL,
        capability_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        PRIMARY KEY (assignment_id, rater_slot_id),
        FOREIGN KEY (assignment_id) REFERENCES review_assignments(assignment_id))""",
    """CREATE TABLE IF NOT EXISTS review_answer_revisions (
        assignment_id TEXT NOT NULL,
        row_id TEXT NOT NULL,
        rater_slot_id TEXT NOT NULL,
        question_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        answer_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (
          assignment_id, row_id, rater_slot_id, question_id, revision),
        FOREIGN KEY (assignment_id, row_id)
          REFERENCES review_rows(assignment_id, row_id),
        FOREIGN KEY (assignment_id, question_id)
          REFERENCES review_questions(assignment_id, question_id),
        FOREIGN KEY (assignment_id, rater_slot_id)
          REFERENCES review_rater_slots(assignment_id, rater_slot_id))""",
    """CREATE TABLE IF NOT EXISTS review_adjudications (
        assignment_id TEXT NOT NULL,
        row_id TEXT NOT NULL,
        adjudicator_slot_id TEXT NOT NULL,
        question_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        answer_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (
          assignment_id, row_id, adjudicator_slot_id, question_id, revision),
        FOREIGN KEY (assignment_id, row_id)
          REFERENCES review_rows(assignment_id, row_id),
        FOREIGN KEY (assignment_id, question_id)
          REFERENCES review_questions(assignment_id, question_id),
        FOREIGN KEY (assignment_id, adjudicator_slot_id)
          REFERENCES review_rater_slots(assignment_id, rater_slot_id))""",
    """CREATE VIEW IF NOT EXISTS current_review_answers AS
       SELECT value.*
       FROM review_answer_revisions value
       WHERE value.revision = (
         SELECT MAX(candidate.revision)
         FROM review_answer_revisions candidate
         WHERE candidate.assignment_id=value.assignment_id
           AND candidate.row_id=value.row_id
           AND candidate.rater_slot_id=value.rater_slot_id
           AND candidate.question_id=value.question_id)""",
    """CREATE VIEW IF NOT EXISTS current_review_adjudications AS
       SELECT value.*
       FROM review_adjudications value
       WHERE value.revision = (
         SELECT MAX(candidate.revision)
         FROM review_adjudications candidate
         WHERE candidate.assignment_id=value.assignment_id
           AND candidate.row_id=value.row_id
           AND candidate.adjudicator_slot_id=value.adjudicator_slot_id
           AND candidate.question_id=value.question_id)""",
    """CREATE TRIGGER IF NOT EXISTS review_answers_no_update
       BEFORE UPDATE ON review_answer_revisions
       BEGIN SELECT RAISE(ABORT, 'review answer revisions are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS review_answers_no_delete
       BEFORE DELETE ON review_answer_revisions
       BEGIN SELECT RAISE(ABORT, 'review answer revisions are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS review_adjudications_no_update
       BEFORE UPDATE ON review_adjudications
       BEGIN SELECT RAISE(ABORT, 'review adjudications are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS review_adjudications_no_delete
       BEFORE DELETE ON review_adjudications
       BEGIN SELECT RAISE(ABORT, 'review adjudications are immutable'); END""",
)


class ReviewSidecar:
    """SQLite owner for one manifest-bound formal review database."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        canonical_manifest_digest: Optional[str] = None,
        workspace_identity: Optional[str] = None,
        evidence_store_digests: Optional[dict[str, str]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self._binding = (
            canonical_manifest_digest,
            workspace_identity,
            evidence_store_digests,
        )
        self._ensure_schema()

    @classmethod
    def from_workspace_manifest(
        cls, workspace_manifest_path: str | Path
    ) -> "ReviewSidecar":
        path = Path(workspace_manifest_path).resolve(strict=True)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReviewValidationError(
                f"workspace manifest is not valid JSON: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ReviewValidationError("workspace manifest must be an object")
        stores = manifest.get("stores")
        if not isinstance(stores, list) or not stores:
            raise ReviewValidationError(
                "workspace manifest stores must be a non-empty array"
            )
        digests: dict[str, str] = {}
        for index, store in enumerate(stores):
            if not isinstance(store, dict):
                raise ReviewValidationError(
                    f"workspace manifest stores[{index}] must be an object"
                )
            store_id = _required_text(
                store.get("store_id"),
                f"workspace manifest stores[{index}].store_id",
            )
            digests[store_id] = _validate_digest(
                store.get("sha256"),
                f"workspace manifest stores[{index}].sha256",
            )
        return cls(
            review_database_path(path),
            canonical_manifest_digest=canonical_document_digest(manifest),
            workspace_identity=_required_text(
                manifest.get("workspace_id"), "workspace manifest workspace_id"
            ),
            evidence_store_digests=digests,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.commit()
        Path(self.db_path).chmod(0o600)

    @staticmethod
    def _capability_hash(capability: str) -> str:
        return hashlib.sha256(capability.encode("utf-8")).hexdigest()

    def create_assignment(
        self,
        document: Any,
        *,
        canonical_manifest_digest: Optional[str] = None,
        workspace_identity: Optional[str] = None,
        evidence_store_digests: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Validate, bind, and persist an assignment; return capabilities once."""
        assignment = validate_review_assignment(document)
        configured = self._binding
        binding = _validate_binding(
            canonical_manifest_digest or configured[0],
            workspace_identity or configured[1],
            evidence_store_digests or configured[2],
        )
        now = datetime.now(timezone.utc).isoformat()
        capabilities = {
            slot_id: secrets.token_urlsafe(32)
            for slot_id in (assignment["rater_slots"] + assignment["adjudicator_slots"])
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO review_assignments
                   (assignment_id, schema, canonical_manifest_digest,
                    workspace_identity, evidence_store_digests_json, rubric_json,
                    blinded, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    assignment["id"],
                    REVIEW_SCHEMA,
                    binding[0],
                    binding[1],
                    _canonical_json(binding[2]),
                    _canonical_json(assignment),
                    1 if assignment["blinded"] else 0,
                    now,
                ),
            )
            for ordinal, row in enumerate(assignment["rows"], start=1):
                turn_ref = row["turn_ref"]
                conn.execute(
                    """INSERT INTO review_rows
                       (assignment_id, row_id, ordinal, store_id,
                        logical_turn_key, turn_key)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        assignment["id"],
                        row["id"],
                        ordinal,
                        turn_ref.get("store_id"),
                        turn_ref.get("logical_turn_key"),
                        turn_ref.get("turn_key"),
                    ),
                )
            for ordinal, question in enumerate(assignment["questions"], start=1):
                conn.execute(
                    """INSERT INTO review_questions
                       (assignment_id, question_id, ordinal, prompt,
                        question_type, vocabulary_json, note_max_length)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        assignment["id"],
                        question["id"],
                        ordinal,
                        question["prompt"],
                        question["type"],
                        (
                            _canonical_json(question["vocabulary"])
                            if "vocabulary" in question
                            else None
                        ),
                        question.get("max_length"),
                    ),
                )
            for role, slot_ids in (
                ("rater", assignment["rater_slots"]),
                ("adjudicator", assignment["adjudicator_slots"]),
            ):
                for slot_id in slot_ids:
                    conn.execute(
                        """INSERT INTO review_rater_slots
                           (assignment_id, rater_slot_id, role,
                            capability_hash, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            assignment["id"],
                            slot_id,
                            role,
                            self._capability_hash(capabilities[slot_id]),
                            now,
                        ),
                    )
            conn.commit()
        return {
            "assignment_id": assignment["id"],
            "rater_capabilities": {
                slot_id: capabilities[slot_id] for slot_id in assignment["rater_slots"]
            },
            "adjudicator_capabilities": {
                slot_id: capabilities[slot_id]
                for slot_id in assignment["adjudicator_slots"]
            },
        }

    def get_assignment(self, assignment_id: str) -> dict[str, Any]:
        """Return a persisted assignment without reissuing its capabilities."""
        assignment_id = _required_text(assignment_id, "assignment_id")
        with self._connect() as conn:
            row = conn.execute(
                """SELECT rubric_json FROM review_assignments
                   WHERE assignment_id=?""",
                (assignment_id,),
            ).fetchone()
        if row is None:
            raise ReviewNotFoundError(f"unknown assignment_id {assignment_id!r}")
        return dict(json.loads(row["rubric_json"]))

    def export_assignment(self, assignment_id: str) -> dict[str, Any]:
        """Return latest rater answers and explicit unanswered coordinates."""
        assignment = self.get_assignment(assignment_id)
        latest = self.current_answers(assignment["id"])
        answers_by_coordinate = {
            (
                answer["row_id"],
                answer["rater_slot_id"],
                answer["question_id"],
            ): answer
            for answer in latest
        }
        unanswered: list[dict[str, str]] = []
        rows = []
        for row in assignment["rows"]:
            rater_answers = []
            for rater_slot_id in assignment["rater_slots"]:
                answers = []
                for question in assignment["questions"]:
                    question_id = question["id"]
                    answer = answers_by_coordinate.get(
                        (row["id"], rater_slot_id, question_id)
                    )
                    if answer is None:
                        unanswered.append(
                            {
                                "row_id": row["id"],
                                "rater_slot_id": rater_slot_id,
                                "question_id": question_id,
                            }
                        )
                        continue
                    answers.append(
                        {
                            "question_id": question_id,
                            "revision": answer["revision"],
                            "answer": answer["answer"],
                        }
                    )
                rater_answers.append(
                    {
                        "rater_slot_id": rater_slot_id,
                        "answers": answers,
                    }
                )
            rows.append(
                {
                    "id": row["id"],
                    "turn_ref": row["turn_ref"],
                    "rater_answers": rater_answers,
                }
            )
        return {
            "assignment_id": assignment["id"],
            "rater_slots": assignment["rater_slots"],
            "questions": assignment["questions"],
            "rows": rows,
            "unanswered": unanswered,
        }

    def _slot_for_capability(
        self, conn: sqlite3.Connection, capability: str, role: str
    ) -> sqlite3.Row:
        if not isinstance(capability, str) or not capability:
            raise ReviewAuthorizationError("a capability is required")
        capability_hash = self._capability_hash(capability)
        row = conn.execute(
            """SELECT assignment_id, rater_slot_id, role, capability_hash
               FROM review_rater_slots WHERE capability_hash=?""",
            (capability_hash,),
        ).fetchone()
        if (
            row is None
            or row["role"] != role
            or not hmac.compare_digest(row["capability_hash"], capability_hash)
        ):
            raise ReviewAuthorizationError(
                f"capability is not authorized for role {role!r}"
            )
        return row

    @staticmethod
    def _question(
        conn: sqlite3.Connection, assignment_id: str, question_id: str
    ) -> sqlite3.Row:
        row = conn.execute(
            """SELECT * FROM review_questions
               WHERE assignment_id=? AND question_id=?""",
            (assignment_id, question_id),
        ).fetchone()
        if row is None:
            raise ReviewNotFoundError(
                f"unknown question_id {question_id!r} for assignment "
                f"{assignment_id!r}"
            )
        return row

    @staticmethod
    def _validate_answer(question: sqlite3.Row, answer: Any) -> str:
        question_type = question["question_type"]
        vocabulary = (
            json.loads(question["vocabulary_json"])
            if question["vocabulary_json"] is not None
            else None
        )
        if question_type == "single-select":
            if not isinstance(answer, str) or answer not in vocabulary:
                raise ReviewValidationError(
                    f"answer for {question['question_id']!r} is outside its "
                    "declared vocabulary"
                )
        elif question_type == "multi-select":
            if (
                not isinstance(answer, list)
                or not all(isinstance(value, str) for value in answer)
                or len(set(answer)) != len(answer)
                or any(value not in vocabulary for value in answer)
            ):
                raise ReviewValidationError(
                    f"answer for {question['question_id']!r} must contain only "
                    "unique values from its declared vocabulary"
                )
        elif question_type == "bounded-note":
            if not isinstance(answer, str):
                raise ReviewValidationError(
                    f"answer for {question['question_id']!r} must be a string"
                )
            if len(answer) > int(question["note_max_length"]):
                raise ReviewValidationError(
                    f"answer for {question['question_id']!r} exceeds max_length "
                    f"{question['note_max_length']}"
                )
        else:
            raise ReviewValidationError(
                f"unsupported stored question type {question_type!r}"
            )
        return _canonical_json(answer)

    @staticmethod
    def _require_row(conn: sqlite3.Connection, assignment_id: str, row_id: str) -> None:
        exists = conn.execute(
            """SELECT 1 FROM review_rows
               WHERE assignment_id=? AND row_id=?""",
            (assignment_id, row_id),
        ).fetchone()
        if exists is None:
            raise ReviewNotFoundError(
                f"unknown row_id {row_id!r} for assignment {assignment_id!r}"
            )

    def _capture(
        self,
        *,
        capability: str,
        role: str,
        table: str,
        slot_column: str,
        row_id: str,
        question_id: str,
        answer: Any,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            slot = self._slot_for_capability(conn, capability, role)
            assignment_id = str(slot["assignment_id"])
            slot_id = str(slot["rater_slot_id"])
            self._require_row(conn, assignment_id, row_id)
            question = self._question(conn, assignment_id, question_id)
            answer_json = self._validate_answer(question, answer)
            revision = int(
                conn.execute(
                    f"""SELECT COALESCE(MAX(revision), 0) + 1 FROM {table}
                        WHERE assignment_id=? AND row_id=?
                          AND {slot_column}=? AND question_id=?""",
                    (assignment_id, row_id, slot_id, question_id),
                ).fetchone()[0]
            )
            conn.execute(
                f"""INSERT INTO {table}
                    (assignment_id, row_id, {slot_column}, question_id,
                     revision, answer_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    assignment_id,
                    row_id,
                    slot_id,
                    question_id,
                    revision,
                    answer_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        return {
            "assignment_id": assignment_id,
            "row_id": row_id,
            slot_column: slot_id,
            "question_id": question_id,
            "revision": revision,
            "answer": json.loads(answer_json),
        }

    def capture_answer(
        self, capability: str, row_id: str, question_id: str, answer: Any
    ) -> dict[str, Any]:
        """Append an answer revision, deriving rater identity from capability."""
        return self._capture(
            capability=capability,
            role="rater",
            table="review_answer_revisions",
            slot_column="rater_slot_id",
            row_id=row_id,
            question_id=question_id,
            answer=answer,
        )

    def capture_adjudication(
        self, capability: str, row_id: str, question_id: str, answer: Any
    ) -> dict[str, Any]:
        """Append a separately attributed adjudication revision."""
        return self._capture(
            capability=capability,
            role="adjudicator",
            table="review_adjudications",
            slot_column="adjudicator_slot_id",
            row_id=row_id,
            question_id=question_id,
            answer=answer,
        )

    def current_answers(
        self, assignment_id: str, *, row_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM current_review_answers WHERE assignment_id=?"
        params: list[Any] = [assignment_id]
        if row_id is not None:
            query += " AND row_id=?"
            params.append(row_id)
        query += " ORDER BY row_id, rater_slot_id, question_id"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        for row in rows:
            row["answer"] = json.loads(row.pop("answer_json"))
        return rows

    def answer_revisions(
        self, assignment_id: str, *, row_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM review_answer_revisions WHERE assignment_id=?"
        params: list[Any] = [assignment_id]
        if row_id is not None:
            query += " AND row_id=?"
            params.append(row_id)
        query += " ORDER BY row_id, rater_slot_id, question_id, revision"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        for row in rows:
            row["answer"] = json.loads(row.pop("answer_json"))
        return rows
