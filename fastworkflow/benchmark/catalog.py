"""Workflow-local versioned benchmark catalogs.

Benchmark corpora live under ``<workflow>/benchmarks/<benchmark_id>/`` as
immutable version files. Each file is the source of truth; nothing is stored in
``observability.sqlite3``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

SCHEMA = "fastworkflow-benchmark/1"
_ANALYSIS_FILENAME = "analysis.json"
_ANALYSIS_STEM = PurePath(_ANALYSIS_FILENAME).stem
_NATURAL_SORT_CHUNKS = re.compile(r"(\d+)")


class BenchmarkCatalogError(RuntimeError):
    """Base class for benchmark catalog failures."""


class BenchmarkManifestError(BenchmarkCatalogError, ValueError):
    """A benchmark version file does not satisfy the v1 contract."""


class BenchmarkAlreadyExistsError(BenchmarkCatalogError):
    """Refused to overwrite an existing immutable version file."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_segment(value: Any, field: str) -> str:
    raw = _required_text(value, field)
    normalized = PurePath(raw.replace("\\", "/"))
    if (
        Path(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ".." in normalized.parts
        or len(normalized.parts) != 1
    ):
        raise BenchmarkManifestError(
            f"{field} must be a single path segment and must not contain '/' or '..'"
        )
    return raw


def _safe_version(value: Any) -> str:
    """A version is a single path segment that is not the analysis sidecar."""
    version = _safe_segment(value, "version")
    if version.lower() == _ANALYSIS_STEM:
        raise BenchmarkManifestError(
            f"version {_ANALYSIS_STEM!r} is reserved for the analysis sidecar"
        )
    return version


def _natural_sort_key(text: str) -> list[object]:
    """Sort ``v1, v2, v10`` numerically rather than lexically."""
    return [
        (0, int(chunk)) if chunk.isdigit() else (1, chunk)
        for chunk in _NATURAL_SORT_CHUNKS.split(text)
        if chunk
    ]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: dict[str, Any]) -> bytes:
    text = json.dumps(data, sort_keys=True, indent=2) + "\n"
    return text.encode("utf-8")


def _require_json_native(value: Any, field: str) -> None:
    """Refuse Python values that JSON would coerce or cannot represent faithfully."""

    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise BenchmarkManifestError(f"{field} keys must be strings")
            _require_json_native(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_native(item, f"{field}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkManifestError(f"{field} must be a finite number")
        return
    raise BenchmarkManifestError(
        f"{field} must be a JSON-native value "
        "(object, array, string, number, boolean, or null)"
    )


def _validate_tasks(raw: Any, field: str = "tasks") -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise BenchmarkManifestError(f"{field} must be an array")
    if not raw:
        raise BenchmarkManifestError(f"{field} must be a non-empty array")
    seen: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkManifestError(f"{item_field} must be an object")
        task_id = _required_text(item.get("task_id"), f"{item_field}.task_id")
        if task_id in seen:
            raise BenchmarkManifestError(f"duplicate task_id {task_id!r}")
        seen.add(task_id)
        description = item.get("description", "")
        if not isinstance(description, str):
            raise BenchmarkManifestError(f"{item_field}.description must be text")
        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise BenchmarkManifestError(f"{item_field}.payload must be a JSON object")
        _require_json_native(payload, f"{item_field}.payload")
        task = {"task_id": task_id, "description": description, "payload": payload}
        if "prompt" in item:
            if not isinstance(item["prompt"], str):
                raise BenchmarkManifestError(f"{item_field}.prompt must be text")
            task["prompt"] = item["prompt"]
        tasks.append(task)
    return tasks


def _validate_manifest(
    data: Any,
    *,
    expected_benchmark_id: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BenchmarkManifestError("benchmark manifest must be a JSON object")
    schema = data.get("schema")
    if schema != SCHEMA:
        raise BenchmarkManifestError(
            f"schema must be {SCHEMA!r}, found {schema!r}"
        )
    benchmark_id = _required_text(data.get("benchmark_id"), "benchmark_id")
    version = _required_text(data.get("version"), "version")
    if expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
        raise BenchmarkManifestError(
            f"benchmark_id {benchmark_id!r} does not match expected "
            f"{expected_benchmark_id!r}"
        )
    if expected_version is not None and version != expected_version:
        raise BenchmarkManifestError(
            f"version {version!r} does not match expected {expected_version!r}"
        )
    description = data.get("description", "")
    if not isinstance(description, str):
        raise BenchmarkManifestError("description must be text")
    tasks = _validate_tasks(data.get("tasks"))
    result = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "version": version,
        "description": description,
        "tasks": tasks,
    }
    if "title" in data:
        result["title"] = _required_text(data["title"], "title")
    return result


def _analysis_path(
    workflow_folderpath: str | Path,
    benchmark_id: str,
) -> Path:
    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    root = benchmarks_root(workflow_folderpath).resolve()
    catalog_dir = (root / safe_benchmark_id).resolve()
    try:
        catalog_dir.relative_to(root)
    except ValueError as exc:
        raise BenchmarkManifestError(
            "benchmark_id resolves outside the benchmarks root"
        ) from exc
    analysis_path = (catalog_dir / _ANALYSIS_FILENAME).resolve()
    try:
        analysis_path.relative_to(catalog_dir)
    except ValueError as exc:
        raise BenchmarkManifestError(
            "analysis path resolves outside the benchmark catalog directory"
        ) from exc
    if analysis_path.name != _ANALYSIS_FILENAME:
        raise BenchmarkManifestError(
            f"analysis filename must be {_ANALYSIS_FILENAME!r}, "
            f"found {analysis_path.name!r}"
        )
    return analysis_path


def _version_path(
    workflow_folderpath: str | Path,
    benchmark_id: str,
    version: str,
) -> Path:
    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    safe_version = _safe_version(version)
    root = benchmarks_root(workflow_folderpath).resolve()
    catalog_dir = (root / safe_benchmark_id).resolve()
    try:
        catalog_dir.relative_to(root)
    except ValueError as exc:
        raise BenchmarkManifestError(
            "benchmark_id resolves outside the benchmarks root"
        ) from exc
    expected_name = f"{safe_version}.json"
    version_path = (catalog_dir / expected_name).resolve()
    try:
        version_path.relative_to(catalog_dir)
    except ValueError as exc:
        raise BenchmarkManifestError(
            "version resolves outside the benchmark catalog directory"
        ) from exc
    if version_path.name != expected_name:
        raise BenchmarkManifestError(
            f"version filename must be {expected_name!r}, found {version_path.name!r}"
        )
    return version_path


def benchmarks_root(workflow_folderpath: str | Path) -> Path:
    """Return ``<workflow>/benchmarks``."""

    return Path(workflow_folderpath) / "benchmarks"


def list_benchmarks(workflow_folderpath: str | Path) -> list[str]:
    """Return benchmark directory names under the workflow's ``benchmarks/`` root."""

    root = benchmarks_root(workflow_folderpath)
    if not root.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            names.append(entry.name)
    return names


def list_versions(workflow_folderpath: str | Path, benchmark_id: str) -> list[str]:
    """Return version ids for one benchmark, excluding ``analysis.json``."""

    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    catalog_dir = benchmarks_root(workflow_folderpath) / safe_benchmark_id
    if not catalog_dir.is_dir():
        return []
    versions: list[str] = []
    for entry in sorted(catalog_dir.iterdir()):
        if not entry.is_file() or entry.suffix != ".json":
            continue
        if entry.name == _ANALYSIS_FILENAME:
            continue
        versions.append(entry.stem)
    versions.sort(key=_natural_sort_key)
    return versions


def load_version(
    workflow_folderpath: str | Path,
    benchmark_id: str,
    version: str,
) -> dict[str, Any]:
    """Load and validate one immutable benchmark version."""

    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    safe_version = _safe_version(version)
    path = _version_path(workflow_folderpath, safe_benchmark_id, safe_version)
    if not path.is_file():
        raise BenchmarkManifestError(f"benchmark version file not found: {path.name}")
    file_bytes = path.read_bytes()
    digest = _sha256_bytes(file_bytes)
    try:
        raw = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(
            f"benchmark version file is not valid JSON: {exc}"
        ) from exc
    manifest = _validate_manifest(
        raw,
        expected_benchmark_id=safe_benchmark_id,
        expected_version=safe_version,
    )
    if path.parent.name != manifest["benchmark_id"]:
        raise BenchmarkManifestError(
            f"folder name {path.parent.name!r} does not match benchmark_id "
            f"{manifest['benchmark_id']!r}"
        )
    expected_filename = f"{manifest['version']}.json"
    if path.name != expected_filename:
        raise BenchmarkManifestError(
            f"filename {path.name!r} does not match version "
            f"{manifest['version']!r}"
        )
    result = dict(manifest)
    result["digest_sha256"] = digest
    return result


def load_analysis(
    workflow_folderpath: str | Path,
    benchmark_id: str,
) -> Any:
    """Load mutable benchmark analysis, or None when no file exists."""

    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    path = _analysis_path(workflow_folderpath, safe_benchmark_id)
    if not path.is_file():
        return None
    file_bytes = path.read_bytes()
    try:
        raw = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(
            f"analysis file is not valid JSON: {exc}"
        ) from exc
    _require_json_native(raw, "analysis")
    return raw


def write_analysis(
    workflow_folderpath: str | Path,
    benchmark_id: str,
    payload: Any,
) -> Any:
    """Write or overwrite mutable benchmark analysis in canonical JSON form."""

    _require_json_native(payload, "payload")
    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    path = _analysis_path(workflow_folderpath, safe_benchmark_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload))
    return payload


def write_version(workflow_folderpath: str | Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Create a new immutable benchmark version file in canonical JSON form."""

    if not isinstance(spec, dict):
        raise BenchmarkManifestError("spec must be a JSON object")
    benchmark_id = _safe_segment(spec.get("benchmark_id"), "benchmark_id")
    version = _safe_version(spec.get("version"))
    manifest = _validate_manifest(
        {
            "schema": SCHEMA,
            "benchmark_id": benchmark_id,
            "version": version,
            "description": spec.get("description"),
            "tasks": spec.get("tasks"),
            **({"title": spec["title"]} if "title" in spec else {}),
        },
        expected_benchmark_id=benchmark_id,
        expected_version=version,
    )
    path = _version_path(workflow_folderpath, benchmark_id, version)
    if path.exists():
        raise BenchmarkAlreadyExistsError(
            f"benchmark version already exists and is immutable: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    file_bytes = _canonical_json(manifest)
    try:
        with path.open("xb") as stream:
            stream.write(file_bytes)
    except FileExistsError as exc:
        raise BenchmarkAlreadyExistsError(f"benchmark version already exists: {path}") from exc
    result = dict(manifest)
    result["digest_sha256"] = _sha256_bytes(file_bytes)
    return result
