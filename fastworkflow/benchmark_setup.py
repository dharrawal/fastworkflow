"""Simple benchmark authoring and experiment identities for harness handoff.

Benchmark versions are immutable catalog files. Experiment registrations can be
created before an execution store exists. A controller binds a registration to
its actual evidence store when it declares the attempts, never at UI creation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastworkflow.benchmark_catalog import (
    BenchmarkManifestError,
    benchmarks_root,
    list_versions,
    load_version,
    write_version,
    _safe_segment,
)


class BenchmarkSetupConflict(ValueError):
    pass


class ExperimentDeleted(BenchmarkSetupConflict):
    """A deleted registration must never fall back to an unregistered run."""



@contextmanager
def _lock(workflow_path):
    # The project targets Unix; flock coordinates HTTP threads and harness processes.
    import fcntl

    root = benchmarks_root(workflow_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_benchmark(workflow_path, body):
    """Generate benchmark/version/new task identities; retain existing task IDs."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    title, description = body.get("title"), body.get("description", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    raw = body.get("tasks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("add at least one task")
    benchmark_id = body.get("benchmark_id") or f"benchmark-{uuid.uuid4().hex}"
    _safe_segment(benchmark_id, "benchmark_id")
    with _lock(workflow_path):
        versions = list_versions(workflow_path, benchmark_id)
        latest = versions[-1] if versions else None
        if body.get("expected_version") != latest:
            raise BenchmarkSetupConflict("benchmark changed; reopen before saving")
        prior = load_version(workflow_path, benchmark_id, latest) if latest else None
        known = {t["task_id"]: t for t in prior["tasks"]} if prior else {}
        tasks, seen = [], set()
        for item in raw:
            if not isinstance(item, dict) or not isinstance(
                item.get("prompt", ""), str
            ):
                raise ValueError("each task prompt must be text (or omitted)")
            task_id = item.get("task_id")
            if task_id is not None and task_id not in known:
                raise ValueError("new task IDs are assigned automatically")
            task_id = task_id or f"task_{uuid.uuid4().hex}"
            if task_id in seen:
                raise ValueError("duplicate task ID")
            seen.add(task_id)
            old = known.get(task_id, {})
            tasks.append(
                {
                    "task_id": task_id,
                    "prompt": item.get("prompt", ""),
                    "description": old.get("description", ""),
                    "payload": old.get("payload", {}),
                }
            )
        numbers = [int(v[1:]) for v in versions if re.fullmatch(r"v\d+", v)]
        version = f"v{max(numbers, default=0) + 1}"
        return write_version(
            workflow_path,
            {
                "benchmark_id": benchmark_id,
                "version": version,
                "title": title.strip(),
                "description": description,
                "tasks": tasks,
            },
        )


def _registration_path(workflow_path, experiment_id):
    _safe_segment(experiment_id, "experiment_id")
    return benchmarks_root(workflow_path) / ".experiments" / f"{experiment_id}.json"


def create_experiment(workflow_path, benchmark_id, version, description=""):
    """Mint an experiment identity; the description is the author's, optional.

    Creation stays one click: the description is free text the author may fill
    in later through `update_experiment_description`, for as long as the
    registration has not been handed to a runner. Copying the benchmark title
    in as a default would put a description on every experiment nobody wrote.
    """
    manifest = load_version(workflow_path, benchmark_id, version)
    if not isinstance(description, str):
        raise ValueError("description must be text")
    experiment_id = f"exp-{uuid.uuid4().hex}"
    record = {
        "experiment_id": experiment_id,
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "benchmark_digest_sha256": manifest["digest_sha256"],
        "description": description.strip(),
        "task_ids": [task["task_id"] for task in manifest["tasks"]],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "store": None,
    }
    with _lock(workflow_path):
        _atomic_json(_registration_path(workflow_path, experiment_id), record)
    return record


def update_experiment_description(workflow_path, experiment_id, description):
    """Edit the description while the registration is still the only record.

    Refused once a runner has bound the registration: from that point the
    description the run declared lives in its evidence store, and a
    registration edited afterwards would disagree with a store nothing here
    can write.
    """
    if not isinstance(description, str):
        raise ValueError("description must be text")
    with _lock(workflow_path):
        record = load_experiment(workflow_path, experiment_id)
        if record.get("store") is not None:
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner; its description "
                "is now part of the recorded run."
            )
        record["description"] = description.strip()
        _atomic_json(_registration_path(workflow_path, experiment_id), record)
        return record


def load_experiment(workflow_path, experiment_id):
    path = _registration_path(workflow_path, experiment_id)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        if (path.parent / ".deleted" / path.name).is_file():
            raise ExperimentDeleted("This experiment was deleted. Create a new experiment.")
        raise KeyError(experiment_id)
    if record.get("experiment_id") != experiment_id:
        raise ValueError("experiment registration identity mismatch")
    return record


def registered_experiments(workflow_path, benchmark_id):
    root = benchmarks_root(workflow_path) / ".experiments"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            rows.append(load_experiment(workflow_path, path.stem))
        except (KeyError, ExperimentDeleted):
            continue  # A deletion can complete between enumeration and read.
    return [row for row in rows if row["benchmark_id"] == benchmark_id]


def experiment_manifest(workflow_path, experiment_id):
    record = load_experiment(workflow_path, experiment_id)
    manifest = load_version(
        workflow_path, record["benchmark_id"], record["benchmark_version"]
    )
    if manifest["digest_sha256"] != record["benchmark_digest_sha256"]:
        raise BenchmarkSetupConflict(
            "benchmark contents no longer match this experiment's pinned version"
        )
    return record, manifest


def bind_experiment(workflow_path, experiment_id, db_path, store_id):
    with _lock(workflow_path):
        record = load_experiment(workflow_path, experiment_id)
        target = {"db_path": os.path.abspath(db_path), "store_id": store_id}
        if record.get("store") not in (None, target):
            raise BenchmarkSetupConflict(
                "experiment is already bound to another evidence store"
            )
        record["store"] = target
        _atomic_json(_registration_path(workflow_path, experiment_id), record)


def delete_empty_experiment(workflow_path, experiment_id):
    """Remove an unused registration, serialized against a runner's store binding.

    A tombstone prevents delayed runners from treating a deleted ID as a new,
    unregistered experiment. No evidence database or benchmark version is touched.
    """
    with _lock(workflow_path):
        record = load_experiment(workflow_path, experiment_id)
        if record.get("store") is not None:
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner and cannot be deleted."
            )
        path = _registration_path(workflow_path, experiment_id)
        deleted = path.parent / ".deleted" / path.name
        deleted.parent.mkdir(exist_ok=True)
        os.replace(path, deleted)
        return record
