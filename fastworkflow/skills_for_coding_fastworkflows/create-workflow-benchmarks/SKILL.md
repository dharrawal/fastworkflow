---
name: create-workflow-benchmarks
description: >-
  Create versioned workflow test benchmarks under benchmarks/, pin them on experiment rows,
  and run measured experiments via drivers — without storing corpora in observability.sqlite3
  or extending the native schema with messages[]. Covers layout, fastworkflow-benchmark/1,
  benchmark_catalog API, immutability, driver-owned opaque payload, IDO HTTP vs in-process
  harness, and where to write analysis. Use when adding a benchmark corpus, snapshotting cases
  for an experiment, or wiring a driver to load and execute pinned tasks.
---

# Creating workflow benchmarks

Versioned test benchmarks are **workflow-local files**. An experiment **pins** a corpus digest;
it never copies the corpus into `observability.sqlite3`.

This skill is separate from `build-task-benchmarks` (multi-turn conversation design). Use that
skill to *design* tasks; use this one to *publish* them as immutable catalog versions and run
experiments against a pin.

## Three objects

| Object | What it is | Where |
|---|---|---|
| **Benchmark version** | Corpus: id, description, version, tasks | `<workflow>/benchmarks/<benchmark_id>/vN.json` |
| **Experiment** | One measured run, optionally pinned | Evidence SQLite (experiment row + pin columns) |
| **Study / workspace** | Several experiments or store segments | Manifest / IDO collection |

## Layout

Use `benchmarks/` (not `tests/` — pytest owns `tests/`). Folder name equals `benchmark_id`;
filename equals version. One immutable JSON file per version. Sibling `analysis.json` is mutable
and **not** part of the version digest.

```text
<workflow>/
  benchmarks/
    smoke/
      v1.json
      v2.json
      analysis.json
    ido-exp028-gate4/
      v4.json
      analysis.json
```

## Native schema (`fastworkflow-benchmark/1`)

FastWorkflow validates only the envelope. `payload` is an opaque JSON object — drivers define
its keys.

```json
{
  "schema": "fastworkflow-benchmark/1",
  "benchmark_id": "smoke",
  "version": "v2",
  "description": "what this corpus claims to test",
  "tasks": [
    {
      "task_id": "case-01",
      "description": "human/agent one-liner",
      "payload": {}
    }
  ]
}
```

Rules:

- `task_id` unique within a version.
- `payload` must be a JSON object; FastWorkflow does **not** schema-check its keys.
  Values must be JSON-native (object/array/string/number/boolean/null), not Python
  tuples or enums.
- **Do not** add optional `messages[]` (or any other native utterance field) “for convenience.”
  Put utterances, protocol refs, Tau snapshots, or harness steps in `payload` from the driver.

## Python API (`fastworkflow.benchmark_catalog`)

```python
from fastworkflow.benchmark_catalog import (
    benchmarks_root,
    list_benchmarks,
    list_versions,
    load_version,
    write_version,
)

# Create (immutable — refuses overwrite)
written = write_version(workflow_folderpath, {
    "benchmark_id": "smoke",
    "version": "v2",
    "description": "what this corpus claims to test",
    "tasks": [
        {
            "task_id": "case-01",
            "description": "one-liner for humans/agents",
            "payload": {"driver_key": "driver-owned"},
        }
    ],
})
digest = written["digest_sha256"]

# Read / enumerate
load_version(workflow_folderpath, "smoke", "v2")
list_benchmarks(workflow_folderpath)
list_versions(workflow_folderpath, "smoke")
```

Versions are **immutable**. To change a corpus, write a **new** version file (`v3.json`), never
edit an existing `vN.json`. `write_version` raises if the file already exists.

## Pin on the experiment row

When creating an experiment (via `ExperimentController`, IDO adapter, or equivalent), copy onto
the experiment row:

- `benchmark_id`
- `version`
- SHA-256 digest of the version file bytes

Two experiments are comparable only when that pin matches (unless an explicit id-only version
diff is requested). Capture-policy mismatch rules still apply.

The corpus stays on disk under `<workflow>/benchmarks/` — **never** in SQLite.

## Drivers own `payload`

Running the corpus is the **driver's** job. FastWorkflow core, `CommandExecutor`, and WEC must
**not** parse `payload`. Execution still sees turns and traces, not benchmark rows.

| Driver | How it runs tasks | Payload |
|---|---|---|
| **IDO measured collections** | HTTP `POST /invoke_agent` after register/bootstrap — **not** `ExperimentHarness` | Driver snapshots frozen-protocol cases into `payload` before pin |
| **In-process harness** | `ExperimentHarness` (one valid driver) | May store utterances or step lists in `payload` if that driver reads them |

Do not assume every workflow uses the same payload shape. Tau Bench corpora belong to Tau's own
workflows (retail, airline, …), not the IDO workflow. FastWorkflow only needs pin + opaque
payload so another workflow *could* import an external snapshot.

## Authoring paths

**Coding agents** write the same files humans create in live `run_chatbot`:

- Benchmarks surface (when present): list benchmarks, create benchmark, add new version, show
  `task_id` / `description`, pretty-print `payload` as opaque JSON — **not** a task player.
- Creating **experiments** remains a driver/API action (no New Experiment form in this slice).

**Workspace / read-only mode** displays the pin and benchmark metadata; it does **not** rewrite
corpus files under `benchmarks/`.

## Where to write analysis

Analysis is **interpretation**, not evidence. FastWorkflow never parses either blob for ranking.

| Location | Scope | Must not contain |
|---|---|---|
| Experiment analysis JSON | One run's conclusions | `hypothesis`, `notes`, scores, traces |
| `<benchmark_id>/analysis.json` | Cross-experiment conclusions (best run, dimension winners) | Anything that should change a version digest |

Editing `analysis.json` must **not** change any `vN.json` digest. Sealed archives get analysis
as a sidecar; do not rewrite sealed evidence.

## Workflow checklist

```
- [ ] Corpus under <workflow>/benchmarks/<benchmark_id>/vN.json (folder name = benchmark_id)
- [ ] schema is fastworkflow-benchmark/1; task_ids unique; payload is opaque object
- [ ] New version = new file; never overwrite vN.json
- [ ] Experiment row pins benchmark_id + version + digest_sha256
- [ ] No corpus rows in observability.sqlite3
- [ ] No native messages[] in the schema; utterances live in payload via the driver
- [ ] IDO collections use HTTP /invoke_agent, not ExperimentHarness, for measured runs
- [ ] Run analysis → experiment analysis JSON; cross-run synthesis → analysis.json
- [ ] Workspace mode does not mutate benchmark files
```

## Don'ts

- Do not store benchmark tasks in `observability.sqlite3`.
- Do not add `messages[]` (or similar) to the native benchmark schema.
- Do not parse `payload` in FastWorkflow core, WEC, or `CommandExecutor`.
- Do not use `ExperimentHarness` for IDO measured collections.
- Do not treat Tau Bench as the IDO-workflow corpus.
- Do not put scores, traces, or hypothesis text in analysis fields.
- Do not edit an existing version file to “fix” a corpus — write `v(N+1).json`.

## Related

- Designing multi-turn conversation content and scoring axes: `build-task-benchmarks`.
- Architecture §21 in `docs/experiment-observability-architecture-design.md`.
- Loader tests: `tests/test_benchmark_catalog.py`.
