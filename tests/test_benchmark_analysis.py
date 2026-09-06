"""Focused tests for opaque benchmark and experiment analysis storage."""

from __future__ import annotations

import json
import sqlite3

import pytest

from fastworkflow import observability_store as obs
from fastworkflow.benchmark_catalog import (
    BenchmarkManifestError,
    benchmarks_root,
    load_analysis,
    load_version,
    write_analysis,
    write_version,
)


def _sample_spec(
    *,
    benchmark_id: str = "smoke",
    version: str = "v1",
    task_id: str = "case-01",
) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "version": version,
        "description": "what this corpus claims to test",
        "tasks": [
            {
                "task_id": task_id,
                "description": "human/agent one-liner",
                "payload": {"nested": {"keep": "keys"}},
            }
        ],
    }


@pytest.fixture
def store(tmp_path, monkeypatch) -> obs.ObservabilityStore:
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    path = str(tmp_path / "observability.sqlite3")
    return obs.ObservabilityStore(path)


def _seed_complete(store: obs.ObservabilityStore, experiment_id: str = "exp-1") -> None:
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=1,
        declared_attempts=1,
    )
    store.start_attempt(experiment_id, "t0", 1, "chan:t0:1")
    store.finish_attempt(
        experiment_id, "t0", 1, outcome="pass", outcome_source="test"
    )
    store.complete_experiment(experiment_id)


class TestBenchmarkAnalysis:
    def test_write_load_analysis_round_trip(self, tmp_path):
        workflow = tmp_path / "workflow"
        workflow.mkdir()
        write_version(workflow, _sample_spec())
        payload = {
            "conclusions": ["task-a regressed on latency"],
            "nested": {"z": 1, "a": 2},
        }

        written = write_analysis(workflow, "smoke", payload)
        loaded = load_analysis(workflow, "smoke")

        assert written == payload
        assert loaded == payload
        on_disk = json.loads(
            (benchmarks_root(workflow) / "smoke" / "analysis.json").read_text(
                encoding="utf-8"
            )
        )
        assert on_disk == payload

    def test_version_digest_unchanged_after_analysis_write(self, tmp_path):
        workflow = tmp_path / "workflow"
        workflow.mkdir()
        before = write_version(workflow, _sample_spec())
        version_path = benchmarks_root(workflow) / "smoke" / "v1.json"
        version_bytes_before = version_path.read_bytes()

        write_analysis(workflow, "smoke", {"notes": "post-hoc review"})

        after = load_version(workflow, "smoke", "v1")
        assert after["digest_sha256"] == before["digest_sha256"]
        assert version_path.read_bytes() == version_bytes_before

    def test_load_analysis_returns_none_when_missing(self, tmp_path):
        workflow = tmp_path / "workflow"
        workflow.mkdir()
        write_version(workflow, _sample_spec())

        assert load_analysis(workflow, "smoke") is None

    def test_analysis_path_traversal_refused(self, tmp_path):
        workflow = tmp_path / "workflow"
        workflow.mkdir()

        with pytest.raises(BenchmarkManifestError, match="single path segment"):
            write_analysis(workflow, "../escape", {"x": 1})

    def test_analysis_tuple_value_refused(self, tmp_path):
        workflow = tmp_path / "workflow"
        workflow.mkdir()
        write_version(workflow, _sample_spec())

        with pytest.raises(
            BenchmarkManifestError,
            match=r"payload\.bindings must be a JSON-native value",
        ):
            write_analysis(workflow, "smoke", {"bindings": (1, 2)})


class TestExperimentAnalysis:
    def test_analysis_and_notes_do_not_overwrite_each_other(self, store):
        store.create_experiment(
            "exp-1", "label", declared_tasks=1, declared_attempts=1
        )
        store.update_experiment_notes("exp-1", "review notes")
        store.update_experiment_analysis(
            "exp-1", {"findings": ["latency spike on t0"]}
        )

        experiment = store.get_experiment("exp-1")
        assert experiment["notes"] == "review notes"
        assert json.loads(experiment["analysis_json"]) == {
            "findings": ["latency spike on t0"],
        }

        store.update_experiment_notes("exp-1", "revised notes")
        store.update_experiment_analysis("exp-1", {"findings": ["updated"]})

        experiment = store.get_experiment("exp-1")
        assert experiment["notes"] == "revised notes"
        assert json.loads(experiment["analysis_json"]) == {"findings": ["updated"]}

    def test_clearing_analysis_leaves_notes(self, store):
        store.create_experiment(
            "exp-1", "label", declared_tasks=1, declared_attempts=1
        )
        store.update_experiment_notes("exp-1", "keep me")
        store.update_experiment_analysis("exp-1", {"x": 1})

        store.update_experiment_analysis("exp-1", None)

        experiment = store.get_experiment("exp-1")
        assert experiment["notes"] == "keep me"
        assert experiment["analysis_json"] is None

    def test_experiment_scores_ignore_analysis(self, store):
        _seed_complete(store)
        store.update_experiment_analysis(
            "exp-1",
            {"conclusions": ["would change scores if counted"]},
        )

        score = store.experiment_scores("exp-1")

        assert score["reportable"] is True
        assert score["pass_at_1"] == pytest.approx(1.0)
        assert "analysis_json" not in score
        assert "analysis" not in score

    def test_read_only_store_refuses_analysis_update(self, tmp_path, monkeypatch):
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
        db_path = str(tmp_path / "observability.sqlite3")
        live = obs.ObservabilityStore(db_path)
        live.create_experiment(
            "exp-1", "label", declared_tasks=1, declared_attempts=1
        )

        readonly = obs.ReadOnlyObservabilityStore(db_path)
        with pytest.raises(sqlite3.OperationalError):
            readonly.update_experiment_analysis("exp-1", {"x": 1})

    def test_non_object_analysis_refused(self, store):
        store.create_experiment(
            "exp-1", "label", declared_tasks=1, declared_attempts=1
        )
        with pytest.raises(ValueError, match="JSON object"):
            store.update_experiment_analysis("exp-1", ["not", "an", "object"])
