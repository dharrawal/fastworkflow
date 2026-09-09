"""Pre-run review records, separate from execution evidence and benchmark payloads.

Reads never create a database. Revisions and decisions are append-only; writes
compare the revision the browser actually displayed inside a SQLite transaction.
Approval records a local operator's declared name, not an authenticated identity.
It does not authorize a runner or prove that execution used this configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "fastworkflow-experiment-setup/1"


class SetupConflict(ValueError):
    """The reviewed revision is no longer current."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _json_native(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_native(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _json_native(item)
        return
    raise ValueError("setup must contain only JSON-native finite values")


def validate_setup(spec):
    _json_native(spec)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    _text(spec.get("experiment_id"), "experiment_id")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", str(spec.get("experiment_id", ""))
    ):
        raise ValueError("experiment_id must be a simple identifier (up to 120 characters)")
    # Same optional prose the studio stores on the experiment: a string, empty
    # if the author wrote nothing. `control`/`change`/`operator_policy` were
    # a parallel frame the UI never collected and are not carried into the run.
    if not isinstance(spec.get("description", ""), str):
        raise ValueError("description must be text")
    for name in ("configuration", "model_routes", "budgets"):
        if not isinstance(spec.get(name), dict) or not spec[name]:
            raise ValueError(f"{name} must be a non-empty object")
    cost = spec.get("estimated_cost")
    if not isinstance(cost, dict):
        raise ValueError("estimated_cost must be an object")
    for name in ("currency", "basis"):
        _text(cost.get(name), f"estimated_cost.{name}")
    for name in ("low", "high"):
        if type(cost.get(name)) not in (int, float) or cost[name] < 0:
            raise ValueError(f"estimated_cost.{name} must be a non-negative number")
    if cost["high"] < cost["low"]:
        raise ValueError("estimated_cost.high must be at least low")
    if type(spec.get("repetitions")) is not int or spec["repetitions"] < 1:
        raise ValueError("repetitions must be a positive integer")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be an object")
        for name in ("task_id", "description"):
            _text(task.get(name), f"task.{name}")
        if task["task_id"] in ids:
            raise ValueError("duplicate task_id")
        ids.add(task["task_id"])
        if task.get("split") not in ("tuning", "held_out"):
            raise ValueError("task.split must be tuning or held_out")
        if "input" not in task:
            raise ValueError("task.input is required")
        outcomes = task.get("expected_outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError("task.expected_outcomes must be a non-empty list")
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                raise ValueError("each expected outcome must be an object")
            for name in ("description", "evidence"):
                _text(outcome.get(name), f"outcome.{name}")
    return spec


def setup_digest(spec):
    validate_setup(spec)
    encoded = json.dumps(
        spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class ExperimentSetups:
    def __init__(self, workflow_path):
        self.path = Path(workflow_path) / "experiment_setups" / "reviews.sqlite3"

    @contextmanager
    def _connect(self, *, write=False):
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=15)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS revisions (
                    experiment_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    digest TEXT NOT NULL, spec TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, revision));
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, digest TEXT NOT NULL,
                    decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    comment TEXT NOT NULL, created_at TEXT NOT NULL);
            """)
        else:
            conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except BaseException:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _read(self, conn, experiment_id):
        rows = conn.execute(
            "SELECT * FROM revisions WHERE experiment_id=? ORDER BY revision DESC",
            (experiment_id,),
        ).fetchall()
        if not rows:
            raise KeyError(experiment_id)
        revisions = []
        for row in rows:
            item = dict(row)
            item["spec"] = json.loads(item["spec"])
            if setup_digest(item["spec"]) != item["digest"]:
                raise ValueError("stored setup digest mismatch")
            revisions.append(item)
        current = revisions[0]
        decisions = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM decisions WHERE experiment_id=? ORDER BY decision_id DESC",
                (experiment_id,),
            )
        ]
        latest = next(
            (d for d in decisions if d["revision"] == current["revision"]), None
        )
        return dict(
            current,
            status=latest["decision"] if latest else "needs_review",
            review=latest,
            history=revisions,
            decisions=decisions,
        )

    def get(self, experiment_id):
        if not self.path.exists():
            raise KeyError(experiment_id)
        with self._connect() as conn:
            return self._read(conn, experiment_id)

    def list(self):
        if not self.path.exists():
            return []
        with self._connect() as conn:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT experiment_id FROM revisions ORDER BY experiment_id"
                )
            ]
            return [self._read(conn, experiment_id) for experiment_id in ids]

    def save(self, spec, expected_revision):
        digest = setup_digest(spec)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        with self._connect(write=True) as conn:
            row = conn.execute(
                "SELECT MAX(revision) FROM revisions WHERE experiment_id=?",
                (spec["experiment_id"],),
            ).fetchone()
            current = row[0] or 0
            if current != expected_revision:
                raise SetupConflict("setup changed; reload before saving")
            conn.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?)",
                (
                    spec["experiment_id"],
                    current + 1,
                    digest,
                    json.dumps(spec, allow_nan=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return self._read(conn, spec["experiment_id"])

    def decide(self, experiment_id, revision, digest, decision, reviewer, comment=""):
        if type(revision) is not int or revision < 1:
            raise ValueError("revision must be a positive integer")
        if decision not in ("approved", "changes_requested"):
            raise ValueError("decision must be approved or changes_requested")
        _text(reviewer, "reviewer")
        if not isinstance(comment, str):
            raise ValueError("comment must be text")
        if decision == "changes_requested":
            _text(comment, "reason for changes")
        if not self.path.exists():
            raise KeyError(experiment_id)
        with self._connect(write=True) as conn:
            current = self._read(conn, experiment_id)
            if current["revision"] != revision or current["digest"] != digest:
                raise SetupConflict(
                    "setup changed; reload and review the current revision"
                )
            conn.execute(
                "INSERT INTO decisions (experiment_id,revision,digest,decision,reviewer,comment,created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    revision,
                    digest,
                    decision,
                    reviewer.strip(),
                    comment,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return self._read(conn, experiment_id)

    def approved(self, experiment_id, revision, digest):
        """Driver handoff: validate an exact, currently approved revision.

        Runners must separately bind these inputs/configuration to execution.
        This is a review receipt, never permission to invoke a model or backend.
        """
        current = self.get(experiment_id)
        if (
            current["revision"] != revision
            or current["digest"] != digest
            or current["status"] != "approved"
        ):
            raise SetupConflict("this exact setup revision is not currently approved")
        return {
            key: current[key]
            for key in ("experiment_id", "revision", "digest", "spec", "review")
        }
