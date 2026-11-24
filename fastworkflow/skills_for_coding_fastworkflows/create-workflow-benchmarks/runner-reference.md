# Runner handoff and optional pre-run review

Use this reference when connecting an application's runner to UI-authored benchmarks. Source
contracts are `fastworkflow/benchmark/setup.py`, `benchmark/catalog.py`, `experiment/runner.py`, and
`experiment/setup.py` in the installed framework checkout.

## Resolve the frozen input

```python
from fastworkflow.benchmark.setup import experiment_manifest

registration, manifest = experiment_manifest(workflow_folderpath, experiment_id)
# Verifies the registered version's byte digest, not whatever version is newest.
```

Preserve `task_id` exactly. Resolve fixture and operator policy from the driver's own contract.
A prompt is optional metadata; if neither it nor driver data supplies runnable input, report an
incomplete setup before execution. Never invent the missing task or omit it from the denominator.

## In-process example

This example explicitly adopts a single-prompt driver convention. It runs models/backend commands
when `run()` is reached; use only after runtime configuration, fixtures and execution authorization
are in place. It is not a dry-run snippet.

```python
from fastworkflow.experiment.runner import ExperimentHarness, ExperimentTask
from fastworkflow.benchmark.setup import experiment_manifest

registration, manifest = experiment_manifest(workflow_folderpath, experiment_id)
tasks = []
for task in manifest["tasks"]:
    prompt = task.get("prompt", "")
    if not prompt.strip():
        raise ValueError(f"Driver input required for task {task['task_id']}")
    tasks.append(ExperimentTask(task_id=task["task_id"], messages=[prompt]))

harness = ExperimentHarness.from_benchmark_experiment(
    workflow_folderpath, experiment_id,
    max_workers=1,
)
# grade_attempt is the application's outcome checker (see Grader in experiment/runner.py).
result = harness.run(tasks, grader=grade_attempt)
```

`from_benchmark_experiment` takes the registration's `description` and `runs_per_task` as
defaults (pass `description=`, or `attempts=` to `run`, to override them). `run` refuses a task
set that is not exactly the pinned version's tasks.

`ExperimentTask.messages` is an in-process harness input, not a benchmark manifest field. The
messages are sent in order, each as its own turn; a message sent while a question is out is the
reply to it. `unattended=True` runs the turns with no ask_user tool. The experiment's conformance
check reads `invalid` when a run's first message differs from its task's saved prompt (a task without
a prompt is not checked; a registration with the user-simulator flag is not checked). The
harness accepts one experiment per instance. Constructing it opens the live evidence store even
before model execution; do not instantiate it to inspect archives.

A grader receives `AttemptRun` and returns `(outcome, outcome_source, reward, detail)`;
returning `None` uses the framework's derived fallback. That fallback only measures whether
commands reported failure, and can mark a recovered command error as a failed attempt. Supply
a real application grader for task-success claims, with its identity and evidence in the result.
A task grader must check the expected application outcome. Runtime completion and all commands
succeeding do not prove that the user received the right answer. An attempt whose last turn
is `awaiting_user` is incomplete; do not count it as a verified success or erase it to make a run
finish. The last command output's `artifacts["question_kind"]` names the question (`agent`,
`plan_choice`, `planning_parameter`, `review`, `approval`, `phase_step`). To end a question inside
an attempt, reply to it (`cancel`, or `no` to a review or approval); `cancel_pending` records no
turn, so the attempt would still read `awaiting_user`.
For interactive tasks, define who answers, which inputs are valid, the question/turn budget and
a timeout. Record Q/A and responder identity in the driver's evidence. A simulated responder is
not a human rater, and a saved review decision does not implement this responder.

## External / HTTP collectors

Use `ExperimentController(workflow_folderpath, expected_store_identity)`, with the store identity
the server reports as `experiment_store_readiness.store_id` on `GET /probes/readyz`. It records into
the workflow's one live DB, `state_paths.observability_db(workflow_folderpath)`, which the server
and the UI resolve too. Do not give the run its own `FASTWORKFLOW_STATE_ROOT` or
`FASTWORKFLOW_WORKFLOW_ID`: a different location is a different DB, so a registered experiment
raises `ExperimentNotRegisteredHere` (naming the path and both variables) and an unregistered one
logs a WARNING. The server must be started first (it installs the store; the controller never
migrates it).

1. `controller.create_experiment(experiment_id, description, declared_tasks=..., declared_attempts=...,
   declarations=..., registered=True)` with the UI's `experiment_id`. `declarations` holds one
   `(task_id, attempt_number, source_key)` tuple per attempt; `declared_tasks` is the number of
   unique tasks and `declared_attempts` the repetitions per task. The pin and the user-simulator
   flag come from the registration.
2. Per attempt, `register_attempt(experiment_id, task_id, attempt, source_key, channel_id)` returns
   an `AttemptBootstrap`. Send `POST /initialize` with that `channel_id` and
   `experiment_bootstrap = {registration_id, secret, store_id}`; the server claims the attempt,
   binds the session and stamps its runtime snapshot. Then send the attempt's turns.
3. `finish_attempt(...)`, which marks the attempt finished and records its outcome in one call.
4. After stopping the server, `drain_before_certify()`, then `complete_experiment(experiment_id)`.

Inserting experiment labels after sending HTTP turns loses attempt provenance; channel IDs,
claims and attempt fencing are not interchangeable with UI labels.

For a server-backed run, examine the bound attempt's `runtime_snapshot` to see what actually ran.
It holds the workflow's content fingerprint, the trained model version, whether observability
was on and pruning suppressed, and the server's pid. A null snapshot means it was not recorded,
not that the intended settings were used. `/probes/readyz?runtime=true` describes the current server; it is not historical
proof of a prior attempt's configuration. For measured external runs, pruning suppression must be
set in the server environment before startup (`FW_OBS_SUPPRESS_PRUNE=1`).

## Optional reviewable experiment setup

When a project uses pre-run review, save a spec via
`ExperimentSetups(workflow_folderpath).save(spec, expected_revision=0)` for a new setup
(`fastworkflow.experiment.setup`). The spec fields are:

| Fields | Content |
|---|---|
| `schema` | `"fastworkflow-experiment-setup/1"` |
| `experiment_id` | Nonempty simple identifier (up to 120 characters); same name as the studio experiment |
| `description` | Optional free-text string; empty if the author wrote nothing |
| `configuration`, `model_routes`, `budgets` | Nonempty JSON objects describing the intended execution |
| `estimated_cost` | `currency`, `basis`, and nonnegative `low`/`high` with `high >= low` |
| `repetitions` | Positive integer |
| `tasks` | Nonempty list with unique `task_id`, nonempty `description`, `split` (`tuning` or `held_out`), `input`, and nonempty `expected_outcomes` |
| Each expected outcome | Nonempty `description` and `evidence` strings |

Use benchmark task IDs in these setup tasks. The setup is a review record separate from the
benchmark catalog; it does not automatically resolve benchmark pins or drive the runner. Keep
input/configuration consistent explicitly.

Saving edits requires the revision read and creates a new revision needing review. Decisions
(`decide(experiment_id, revision, digest, decision, reviewer, comment)`, `approved` or
`changes_requested`, the latter with a comment) name an exact revision and digest. Use
`approved(experiment_id, revision, digest)` to retrieve an exact current review receipt for a driver
that supports this check. The reviewer name is self-declared; a receipt is not authenticated
execution authority and does not prove the runner used the settings. Do not record the owner's
approval on their behalf or treat a review receipt as permission for additional paid calls.

## Local UI API equivalents

The browser supplies the local session authentication; use that same authenticated session for
programmatic UI requests. URL-encode IDs. These are `run_chatbot` APIs, not `/invoke_agent` APIs:

| Action | Method and route |
|---|---|
| Save simple benchmark | `POST /api/benchmark-setup` with the `save_benchmark` body |
| Write a version with explicit IDs | `POST /api/benchmarks/<benchmark_id>/versions` with the `write_version` spec |
| Register experiment | `POST /api/benchmarks/<benchmark_id>/experiments` with `{"version":"v1"}`; optional `description`, `runs_per_task`, `user_simulator` |
| Read registration and pinned tasks | `GET /api/benchmark-experiments/<experiment_id>` |
| Read version | `GET /api/benchmarks/<benchmark_id>/versions/<version>` |
| Read/write experiment notes | `GET /api/experiment/<experiment_id>` / `PATCH /api/experiment/<experiment_id>` with `{"notes": "..."}` |

Execution reads/writes go to the workflow's one live database, so a registered experiment's
evidence is visible only once it is recorded there. Do not confuse a registration with a recorded experiment:
a registration can exist before there are attempts or conversations to show.
