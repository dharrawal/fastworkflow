"""Focused tests for experiment benchmark pins."""

from __future__ import annotations

import pytest

from fastworkflow import observability_store as obs
from fastworkflow.benchmark_catalog import write_version
from fastworkflow.experiment import (
    BenchmarkPinDigestMismatch,
    ExperimentController,
    ExperimentHarness,
    ExperimentTask,
    experiment_store_readiness,
)


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    path = str(tmp_path / "observability.sqlite3")
    obs.ObservabilityStore(path)
    return path


def _controller(db_path: str) -> ExperimentController:
    identity = experiment_store_readiness(db_path)["store_id"]
    return ExperimentController(
        db_path, identity, migrate=False, external=False
    )


def _declarations(tasks: int = 1, attempts: int = 1):
    return [
        (f"t{index}", attempt, f"channel:t{index}:{attempt}")
        for index in range(tasks)
        for attempt in range(1, attempts + 1)
    ]


def _sample_pin(*, benchmark_id: str = "smoke", version: str = "v1") -> dict[str, str]:
    return {
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "benchmark_digest_sha256": "deadbeef",
    }


def _seed_complete(store: obs.ObservabilityStore, experiment_id: str) -> None:
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=2,
        declared_attempts=2,
    )
    for task_index in range(2):
        task_id = f"t{task_index}"
        for attempt in (1, 2):
            store.start_attempt(experiment_id, task_id, attempt, f"c:{task_id}:{attempt}")
            store.finish_attempt(
                experiment_id,
                task_id,
                attempt,
                outcome="pass",
                outcome_source="test",
            )
    assert store.complete_experiment(experiment_id) == "complete"


def test_create_with_pin_returns_it_from_get_experiment(db_path):
    controller = _controller(db_path)
    pin = _sample_pin()

    controller.create_experiment(
        "exp-1",
        "label",
        declared_tasks=1,
        declared_attempts=1,
        declarations=_declarations(),
        **pin,
    )

    experiment = controller.store.get_experiment("exp-1")
    assert experiment["benchmark_id"] == pin["benchmark_id"]
    assert experiment["benchmark_version"] == pin["benchmark_version"]
    assert experiment["benchmark_digest_sha256"] == pin["benchmark_digest_sha256"]


def test_list_experiments_returns_benchmark_pin(db_path):
    store = obs.ObservabilityStore(db_path)
    pin = _sample_pin()
    store.create_experiment(
        "exp-1",
        "label",
        declared_tasks=1,
        declared_attempts=1,
        **pin,
    )

    listed = store.list_experiments()
    assert len(listed) == 1
    assert listed[0]["benchmark_id"] == pin["benchmark_id"]
    assert listed[0]["benchmark_version"] == pin["benchmark_version"]
    assert listed[0]["benchmark_digest_sha256"] == pin["benchmark_digest_sha256"]


def test_create_without_pin_still_works(db_path):
    controller = _controller(db_path)

    controller.create_experiment(
        "exp-1",
        "label",
        declared_tasks=1,
        declared_attempts=1,
        declarations=_declarations(),
    )

    experiment = controller.store.get_experiment("exp-1")
    assert experiment["benchmark_id"] is None
    assert experiment["benchmark_version"] is None
    assert experiment["benchmark_digest_sha256"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"benchmark_id": "smoke"},
        {"benchmark_version": "v1"},
        {"benchmark_digest_sha256": "abc"},
        {"benchmark_id": "smoke", "benchmark_version": "v1"},
    ],
)
def test_partial_pin_refused(db_path, kwargs):
    controller = _controller(db_path)

    with pytest.raises(obs.PartialBenchmarkPin):
        controller.create_experiment(
            "exp-1",
            "label",
            declared_tasks=1,
            declared_attempts=1,
            declarations=_declarations(),
            **kwargs,
        )


def test_recreate_with_different_pin_refused(db_path):
    controller = _controller(db_path)
    first = _sample_pin()

    controller.create_experiment(
        "exp-1",
        "label",
        declared_tasks=1,
        declared_attempts=1,
        declarations=_declarations(),
        **first,
    )

    second = _sample_pin(benchmark_id="other", version="v2")
    with pytest.raises(obs.BenchmarkPinIsWriteOnce):
        controller.create_experiment(
            "exp-1",
            "label-2",
            declared_tasks=1,
            declared_attempts=1,
            declarations=_declarations(),
            **second,
        )

    experiment = controller.store.get_experiment("exp-1")
    assert experiment["benchmark_id"] == first["benchmark_id"]
    assert experiment["benchmark_version"] == first["benchmark_version"]
    assert experiment["benchmark_digest_sha256"] == first["benchmark_digest_sha256"]


def test_recreate_unpinned_running_experiment_with_pin_stores_it(db_path):
    """The resume path must not silently drop a pin the first create lacked.

    The write-once guard only fires when a pin is already stored, so an
    upsert that never assigns the pin columns would leave the row unpinned
    with no error.
    """
    store = obs.ObservabilityStore(db_path)
    store.create_experiment("exp-1", "label", declared_tasks=1, declared_attempts=1)
    assert store.get_experiment("exp-1")["benchmark_id"] is None

    pin = _sample_pin()
    store.create_experiment(
        "exp-1", "label", declared_tasks=1, declared_attempts=1, **pin
    )

    experiment = store.get_experiment("exp-1")
    assert experiment["benchmark_id"] == pin["benchmark_id"]
    assert experiment["benchmark_version"] == pin["benchmark_version"]
    assert experiment["benchmark_digest_sha256"] == pin["benchmark_digest_sha256"]

    # Once taken, the pin is write-once like any other.
    with pytest.raises(obs.BenchmarkPinIsWriteOnce):
        store.create_experiment(
            "exp-1",
            "label",
            declared_tasks=1,
            declared_attempts=1,
            **_sample_pin(benchmark_id="other", version="v2"),
        )
    experiment = store.get_experiment("exp-1")
    assert experiment["benchmark_id"] == pin["benchmark_id"]
    assert experiment["benchmark_version"] == pin["benchmark_version"]
    assert experiment["benchmark_digest_sha256"] == pin["benchmark_digest_sha256"]


def test_recreate_without_pin_keeps_stored_pin(db_path):
    store = obs.ObservabilityStore(db_path)
    pin = _sample_pin()
    store.create_experiment(
        "exp-1", "label", declared_tasks=1, declared_attempts=1, **pin
    )

    store.create_experiment("exp-1", "label-2", declared_tasks=1, declared_attempts=1)

    experiment = store.get_experiment("exp-1")
    assert experiment["description"] == "label-2"
    assert experiment["benchmark_id"] == pin["benchmark_id"]
    assert experiment["benchmark_version"] == pin["benchmark_version"]
    assert experiment["benchmark_digest_sha256"] == pin["benchmark_digest_sha256"]


def test_compare_mismatched_digest_is_not_comparable(db_path):
    store = obs.ObservabilityStore(db_path)
    _seed_complete(store, "exp-a")
    _seed_complete(store, "exp-b")

    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE experiments
                  SET benchmark_id='smoke',
                      benchmark_version='v1',
                      benchmark_digest_sha256=?
                WHERE experiment_id='exp-a'""",
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",),
        )
        conn.execute(
            """UPDATE experiments
                  SET benchmark_id='smoke',
                      benchmark_version='v1',
                      benchmark_digest_sha256=?
                WHERE experiment_id='exp-b'""",
            ("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",),
        )
        conn.commit()

    result = store.compare_experiments("exp-b", "exp-a")
    assert result["comparable"] is False
    assert any("benchmark pins differ" in problem for problem in result["problems"])
    assert any("aaaaaaaa" in problem for problem in result["problems"])
    assert any("bbbbbbbb" in problem for problem in result["problems"])


def test_compare_both_unpinned_uses_existing_shape_rules(db_path):
    store = obs.ObservabilityStore(db_path)
    _seed_complete(store, "exp-a")
    _seed_complete(store, "exp-b")

    result = store.compare_experiments("exp-b", "exp-a")
    assert result["comparable"] is True
    assert result["problems"] == []


def test_controller_verifies_digest_when_workflow_folder_available(tmp_path, db_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    written = write_version(
        workflow,
        {
            "benchmark_id": "smoke",
            "version": "v1",
            "description": "catalog",
            "tasks": [
                {
                    "task_id": "case-01",
                    "description": "one",
                    "payload": {},
                }
            ],
        },
    )
    controller = _controller(db_path)

    controller.create_experiment(
        "exp-1",
        "label",
        declared_tasks=1,
        declared_attempts=1,
        declarations=_declarations(),
        benchmark_id="smoke",
        benchmark_version="v1",
        benchmark_digest_sha256=written["digest_sha256"],
        workflow_folderpath=str(workflow),
    )

    experiment = controller.store.get_experiment("exp-1")
    assert experiment["benchmark_digest_sha256"] == written["digest_sha256"]


def test_controller_refuses_digest_mismatch_when_workflow_folder_available(
    tmp_path, db_path
):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(
        workflow,
        {
            "benchmark_id": "smoke",
            "version": "v1",
            "description": "catalog",
            "tasks": [
                {
                    "task_id": "case-01",
                    "description": "one",
                    "payload": {},
                }
            ],
        },
    )
    controller = _controller(db_path)

    with pytest.raises(BenchmarkPinDigestMismatch):
        controller.create_experiment(
            "exp-1",
            "label",
            declared_tasks=1,
            declared_attempts=1,
            declarations=_declarations(),
            benchmark_id="smoke",
            benchmark_version="v1",
            benchmark_digest_sha256="0" * 64,
            workflow_folderpath=str(workflow),
        )


def test_harness_run_forwards_benchmark_pin_to_create_experiment(
    tmp_path, monkeypatch
):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    written = write_version(
        workflow,
        {
            "benchmark_id": "smoke",
            "version": "v1",
            "description": "catalog",
            "tasks": [
                {
                    "task_id": "case-01",
                    "description": "one",
                    "payload": {},
                }
            ],
        },
    )
    harness = ExperimentHarness(
        str(workflow),
        description="label",
        experiment_id="exp-harness",
        benchmark_id="smoke",
        benchmark_version="v1",
        benchmark_digest_sha256=written["digest_sha256"],
    )
    captured: dict[str, object] = {}

    def fake_create_experiment(*args, **kwargs):
        captured["kwargs"] = kwargs
        harness._controller.store.create_experiment(
            harness.experiment_id,
            harness.description,
            declared_tasks=kwargs["declared_tasks"],
            declared_attempts=kwargs["declared_attempts"],
            benchmark_id=kwargs.get("benchmark_id"),
            benchmark_version=kwargs.get("benchmark_version"),
            benchmark_digest_sha256=kwargs.get("benchmark_digest_sha256"),
        )
        harness._controller.store.declare_experiment_attempts(
            harness.experiment_id, kwargs["declarations"]
        )

    monkeypatch.setattr(harness._controller, "create_experiment", fake_create_experiment)
    monkeypatch.setattr(harness, "_execute", lambda pairs, grader, seq: {"ok": True})

    harness.run([ExperimentTask(task_id="t0", messages=["hi"])], attempts=1)

    kwargs = captured["kwargs"]
    assert kwargs["benchmark_id"] == "smoke"
    assert kwargs["benchmark_version"] == "v1"
    assert kwargs["benchmark_digest_sha256"] == written["digest_sha256"]
    assert kwargs["workflow_folderpath"] == str(workflow)
