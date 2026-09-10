"""Focused tests for workflow-local benchmark catalogs."""

from __future__ import annotations

import json
from enum import Enum

import pytest

from fastworkflow.benchmark.catalog import (
    SCHEMA,
    BenchmarkAlreadyExistsError,
    BenchmarkManifestError,
    benchmarks_root,
    list_benchmarks,
    list_versions,
    load_version,
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
                "payload": {
                    "nested": {"keep": "keys"},
                    "order": [3, 2, 1],
                },
            }
        ],
    }


def test_write_load_round_trip_preserves_payload_keys(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()

    written = write_version(workflow, spec)
    loaded = load_version(workflow, "smoke", "v1")

    assert written["digest_sha256"] == loaded["digest_sha256"]
    assert loaded["schema"] == SCHEMA
    assert loaded["benchmark_id"] == "smoke"
    assert loaded["version"] == "v1"
    assert loaded["tasks"][0]["payload"] == spec["tasks"][0]["payload"]

    on_disk = json.loads(
        (benchmarks_root(workflow) / "smoke" / "v1.json").read_text(encoding="utf-8")
    )
    assert on_disk["tasks"][0]["payload"] == spec["tasks"][0]["payload"]


def test_digest_stability_on_repeated_load(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    first = load_version(workflow, "smoke", "v1")
    second = load_version(workflow, "smoke", "v1")

    assert first["digest_sha256"] == second["digest_sha256"]


def test_digest_uses_actual_file_bytes_when_not_canonical(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    version_path = benchmarks_root(workflow) / "smoke" / "v1.json"
    non_canonical = version_path.read_text(encoding="utf-8").replace(
        '"description": "what this corpus claims to test"',
        '"description":"what this corpus claims to test"',
    )
    version_path.write_text(non_canonical, encoding="utf-8")

    loaded = load_version(workflow, "smoke", "v1")
    assert loaded["description"] == "what this corpus claims to test"
    assert loaded["digest_sha256"] != write_version(
        tmp_path / "other-workflow", _sample_spec()
    )["digest_sha256"]


def test_duplicate_task_id_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"] = [
        {
            "task_id": "case-01",
            "description": "first",
            "payload": {},
        },
        {
            "task_id": "case-01",
            "description": "duplicate",
            "payload": {},
        },
    ]

    with pytest.raises(BenchmarkManifestError, match="duplicate task_id"):
        write_version(workflow, spec)


def test_overwrite_existing_version_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    changed = _sample_spec()
    changed["description"] = "changed description"
    with pytest.raises(BenchmarkAlreadyExistsError, match="immutable"):
        write_version(workflow, changed)


def test_multiple_benchmarks_in_one_workflow(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec(benchmark_id="alpha", version="v1"))
    write_version(
        workflow,
        _sample_spec(
            benchmark_id="beta",
            version="v1",
            task_id="beta-case",
        ),
    )
    write_version(
        workflow,
        _sample_spec(
            benchmark_id="alpha",
            version="v2",
            task_id="alpha-case-2",
        ),
    )

    assert list_benchmarks(workflow) == ["alpha", "beta"]
    assert list_versions(workflow, "alpha") == ["v1", "v2"]
    assert list_versions(workflow, "beta") == ["v1"]


def test_analysis_json_is_not_listed_as_version(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    catalog_dir = benchmarks_root(workflow) / "smoke"
    catalog_dir.joinpath("analysis.json").write_text(
        json.dumps({"notes": "opaque analysis"}),
        encoding="utf-8",
    )

    assert list_versions(workflow, "smoke") == ["v1"]


def test_reserved_analysis_version_refused(tmp_path):
    """`analysis.json` is the mutable sidecar; a version named `analysis`
    would be hidden by list_versions and overwritten by the next
    write_analysis, breaking version immutability."""
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    with pytest.raises(BenchmarkManifestError, match="reserved"):
        write_version(workflow, _sample_spec(version="analysis"))
    with pytest.raises(BenchmarkManifestError, match="reserved"):
        write_version(workflow, _sample_spec(version="Analysis"))
    assert not (benchmarks_root(workflow) / "smoke" / "analysis.json").exists()
    assert list_versions(workflow, "smoke") == ["v1"]

    with pytest.raises(BenchmarkManifestError, match="reserved"):
        load_version(workflow, "smoke", "analysis")


def test_list_versions_sorts_naturally(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    for version in ("v10", "v2", "v1"):
        write_version(
            workflow,
            _sample_spec(version=version, task_id=f"case-{version}"),
        )

    assert list_versions(workflow, "smoke") == ["v1", "v2", "v10"]


def test_path_traversal_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(benchmark_id="../escape"))

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(version="../v1"))


def test_slash_in_benchmark_id_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(benchmark_id="foo/bar"))

    assert not (benchmarks_root(workflow) / "foo").exists()


def test_folder_benchmark_id_mismatch_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec(benchmark_id="smoke", version="v1"))

    wrong_dir = benchmarks_root(workflow) / "other"
    wrong_dir.mkdir()
    canonical = json.dumps(
        {
            "schema": SCHEMA,
            "benchmark_id": "smoke",
            "version": "v1",
            "description": "mismatch",
            "tasks": [
                {
                    "task_id": "case-01",
                    "description": "one",
                    "payload": {},
                }
            ],
        },
        sort_keys=True,
        indent=2,
    ) + "\n"
    wrong_dir.joinpath("v1.json").write_text(canonical, encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="does not match expected"):
        load_version(workflow, "other", "v1")


def test_empty_task_list_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"] = []

    with pytest.raises(BenchmarkManifestError, match="non-empty array"):
        write_version(workflow, spec)


def test_payload_tuple_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"bindings": (3, 2, 1)}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.bindings must be a JSON-native value",
    ):
        write_version(workflow, spec)


def test_payload_enum_refused(tmp_path):
    class Flavor(Enum):
        VANILLA = "vanilla"

    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"flavor": Flavor.VANILLA}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.flavor must be a JSON-native value",
    ):
        write_version(workflow, spec)


def test_payload_non_string_key_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {1: "not-a-string-key"}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload keys must be strings",
    ):
        write_version(workflow, spec)


def test_payload_non_finite_float_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"score": float("nan")}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.score must be a finite number",
    ):
        write_version(workflow, spec)


def test_payload_nested_json_native_list_accepted(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    assert spec["tasks"][0]["payload"]["order"] == [3, 2, 1]

    written = write_version(workflow, spec)

    assert written["tasks"][0]["payload"]["order"] == [3, 2, 1]
    assert written["tasks"][0]["payload"]["nested"] == {"keep": "keys"}
