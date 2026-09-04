"""Post-drain external experiment evidence sealing."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat

import pytest

from fastworkflow import observability_store as obs
from fastworkflow import state_paths
from fastworkflow.experiment import ExperimentController, experiment_store_readiness


@pytest.fixture
def installed_db(tmp_path, monkeypatch):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    path = str(tmp_path / "observability.sqlite3")
    store = obs.ObservabilityStore(path)
    identity = store.store_identity()
    assert identity is not None
    return path, identity


def _controller(installed_db, *, external=True):
    path, identity = installed_db
    return ExperimentController(
        path,
        identity,
        migrate=False,
        external=external,
    )


def _create(controller, *, required_segments=1):
    controller.create_experiment(
        "exp-1",
        "external",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[("task-1", 1, "native-1")],
        required_evidence_segments=required_segments,
    )


def _finish(controller):
    controller.start_attempt(
        "exp-1",
        "task-1",
        1,
        "channel-1",
        source_key="native-1",
    )
    controller.finish_attempt(
        "exp-1",
        "task-1",
        1,
        outcome="pass",
        outcome_source="external-grader",
    )


def _segment(*, valid=True):
    return {
        "valid": valid,
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": "2026-09-04T00:01:00+00:00",
        "problems": [] if valid else ["writer records dropped"],
    }


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_external_shortfall_is_retryable_but_invalid_evidence_is_terminal(
    installed_db,
):
    controller = _controller(installed_db)
    _create(controller)

    assert controller.complete_experiment("exp-1") == "running"
    _finish(controller)
    assert controller.complete_experiment("exp-1") == "running"

    controller.record_evidence_segment("exp-1", 1, "run-1", _segment(valid=False))
    assert controller.complete_experiment("exp-1") == "invalid"
    with pytest.raises(obs.ExperimentIsClosed):
        controller.record_evidence_segment(
            "exp-1", 2, "run-2", _segment(valid=True)
        )


def test_capture_complete_needs_archive_digest_before_reportable(
    installed_db, tmp_path
):
    controller = _controller(installed_db)
    _create(controller)
    _finish(controller)
    controller.record_evidence_segment("exp-1", 1, "run-1", _segment())

    assert controller.complete_experiment("exp-1") == "capture_complete"
    assert controller.store.experiment_scores("exp-1")["reportable"] is False
    with pytest.raises(obs.ExperimentIsClosed):
        controller.record_evidence_segment(
            "exp-1", 2, "run-2", _segment(valid=True)
        )

    archive_path = tmp_path / "workspace.sqlite3"
    archive = controller.seal_workspace_evidence(
        "exp-1", str(archive_path)
    )
    assert archive["experiment_status"] == "complete"
    assert archive["sealed"] is True
    assert archive["store_identity"] == installed_db[1]
    experiment = controller.store.get_experiment("exp-1")
    assert experiment["workspace_archive_sha256"] == archive["sha256"]
    assert experiment["workspace_store_identity"] == installed_db[1]
    assert controller.store.experiment_scores("exp-1")["reportable"] is True


def test_drain_blocks_certification_until_owned_writer_stops(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    sink = obs.get_observability_sink(str(workflow))
    assert sink is not None
    path = state_paths.observability_db(str(workflow))
    identity = experiment_store_readiness(path)["store_id"]
    controller = ExperimentController(path, identity, migrate=False, external=True)
    _create(controller, required_segments=0)
    _finish(controller)

    with pytest.raises(obs.WriterStillOpen):
        controller.complete_experiment("exp-1")
    health = controller.drain_before_certify()
    assert sink._writer.is_alive() is False
    assert health["pending_retry_depth"] == 0
    assert controller.complete_experiment("exp-1") == "capture_complete"


def test_archive_includes_committed_wal_without_mutating_source(
    installed_db, tmp_path
):
    path, identity = installed_db
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE wal_evidence (value TEXT NOT NULL)")
    writer.execute("INSERT INTO wal_evidence VALUES ('committed-in-wal')")
    writer.commit()
    assert os.path.exists(f"{path}-wal")
    before = {
        path: _sha256(path),
        f"{path}-wal": _sha256(f"{path}-wal"),
    }

    archive_path = tmp_path / "sealed.sqlite3"
    archive = obs.ObservabilityStore(path, migrate=False).archive_to(
        str(archive_path)
    )
    after = {
        path: _sha256(path),
        f"{path}-wal": _sha256(f"{path}-wal"),
    }
    writer.close()

    assert before == after
    assert archive["source_bytes_verified_unchanged"] is True
    assert archive["sidecar_free"] is True
    assert archive["store_identity"] == identity
    assert archive["sha256"] == _sha256(archive_path)
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o444
    assert not os.path.exists(f"{archive_path}-wal")
    assert not os.path.exists(f"{archive_path}-shm")
    archive_uri = archive_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as conn:
        assert conn.execute("SELECT value FROM wal_evidence").fetchone()[0] == (
            "committed-in-wal"
        )


def test_source_change_aborts_seal_and_removes_destination(
    installed_db, tmp_path, monkeypatch
):
    path, _ = installed_db
    store = obs.ObservabilityStore(path, migrate=False)
    original = store._file_digest
    source_calls = 0

    def changed_after_snapshot(candidate):
        nonlocal source_calls
        result = original(candidate)
        if candidate == os.path.abspath(path):
            source_calls += 1
            if source_calls >= 2 and result is not None:
                result = dict(result)
                result["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(store, "_file_digest", changed_after_snapshot)
    destination = tmp_path / "must-not-exist.sqlite3"
    with pytest.raises(obs.SourceChangedDuringArchive):
        store.archive_to(str(destination))
    assert not destination.exists()


def test_archive_open_never_migrates_source(installed_db, tmp_path, monkeypatch):
    path, _ = installed_db
    store = obs.ObservabilityStore(path, migrate=False)

    def forbidden(self):
        raise AssertionError("archive migrated its source")

    monkeypatch.setattr(obs.ObservabilityStore, "_ensure_schema", forbidden)
    archive = store.archive_to(str(tmp_path / "sealed.sqlite3"))
    assert archive["sealed"] is True
