# Overhead benchmark for the observability substrate (bead fix-49m.2)

Measures the runtime cost of the observability substrate being ported onto
`feat/experiment-observability-3.3` (change 1 = `7b1f413`, change 4 =
`5b1e85e` on `fix/training-gates-singleton-and-provenance`), by running the
same scripted agent turn at three points:

| point            | checkout                                             | label suggestion   |
|------------------|------------------------------------------------------|--------------------|
| fork (v3.2.0)    | `9904df5`                                            | `fork-9904df5`     |
| change 4         | `5b1e85e`                                            | `change4-5b1e85e`  |
| end of port      | `feat/experiment-observability-3.3` after the port   | `port-3.3-<sha>`   |

Report only; no switch is added on the result.

**Running the measurement needs the owner's explicit go.** Everything below
is "how", not "do it now". The harness's own unit tests and `--help` are safe
to run at any time.

## What one run does

1. Puts the chosen checkout first on `sys.path` (the venv's `fastworkflow.pth`
   points at `/home/drawal/rl/fastworkflow`, so this is how a worktree gets
   selected) and refuses to continue if `fastworkflow` imports from anywhere
   else.
2. `fastworkflow.init()` with the checkout's example env file plus every
   `LLM_*` role pointed at a stub model name, a fresh temporary
   `FASTWORKFLOW_STATE_ROOT`, and `FW_OBSERVABILITY=1` (or `0` with
   `--observability off`). `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are
   set so the intent-detection models can only load from local artifacts.
3. Routes every LM lookup to the stub (`fastworkflow.utils.dspy_utils.get_lm`,
   `fastworkflow.conversation_labeling.get_lm`, `dspy.configure(lm=...)`) and
   installs a tripwire that makes any real `dspy.LM`/`litellm` call raise and
   be counted (`tripwire_trips` in the report must be 0).
4. Builds `ChatSession(run_as_agent=True)`, binds the SQLite observability
   sink and mints a conversation id exactly as `fastworkflow run` does, then
   `start_workflow(<fixture>, keep_alive=True)`.
5. One warm-up turn (2 commands; lazy agent init, model warm-up, first sink
   writes), then widens the ReAct `max_iters` so the script — not the
   iteration cap — ends each turn.
6. Enqueues N identical user messages on `user_message_queue` and, for each,
   waits for the CLI's turn-complete sentinel on `command_trace_queue` and
   collects the `CommandOutput`. Each turn: planner call, K ReAct steps each
   issuing `execute_workflow_query`, one `finish` step, the extract call and
   the turn-summary call (K=48 gives 48 commands and 52 model calls per turn).
7. `sink.flush()`, then reads the DB size (main + `-wal` + `-shm`) and row
   counts per table, span-name and span-status histograms, and — when the
   schema exposes durations — `fw.command.execute` span durations as a
   cross-check.

## Metrics in the JSON report

* `metrics.turn_wall_ms` — median/p90/mean/min/max of per-turn wall time.
  Turn i is measured sentinel-to-sentinel, so it includes the previous turn's
  post-sentinel finalize (turn record + root span close). The last turn's
  finalize lands in `total_wall_s` / `flush_s` instead.
* `metrics.command_gap_ms` — per-command latency: the gap between the stub
  returning a tool-selecting ReAct step and the runtime asking for the next
  step. That is exactly the command's dispatch + execution + span/recorder
  bookkeeping + trajectory formatting, with zero stub time inside it.
* `metrics.turn_wall_per_command_ms` — turn wall / commands, for reference.
* `metrics.stub_lm_time_per_turn_ms` — time spent inside the stub (should be
  small and identical across checkouts; subtract if you want "runtime only").
* `metrics.process_cpu_s`, `process_cpu_per_turn_s` — `time.process_time()`
  over the measured section, all threads (the sink's writer thread included).
* `metrics.peak_rss_bytes` (VmHWM), `rss_before_bytes`, `rss_after_bytes`,
  `ru_maxrss_bytes`.
* `observability_db_bytes` after flush and `observability_db_bytes_after_close`.
* `observability_rows.tables` (every table), `span_names`, `spans_by_status`,
  `command_execute_span_ms` (best effort).
* `stub` — per-turn model calls by kind, commands issued, anomalies,
  `unparsed_prompts` (must be 0: a non-zero value means a prompt format the
  stub did not recognise and answered with plain text).
* `guards` — which optional steps applied on this checkout (`runtime_manifest`
  absent at the fork, applied from 5b1e85e; `react_max_iters_original`;
  `trace_sink_bound`; `conversation_id_minted`; `is_workflow_trained`).
* `notes` — anything that deviated (commands per turn not equal to K, a turn
  with `success=False`, sink missing, ...). Read these before comparing.
* `summary` — the one-line summary also printed to stdout.

## Trained artifacts (one-time per worktree)

The fixture is `tests/hello_world_workflow` (one command,
`add_two_numbers <first_num>..</first_num> <second_num>..</second_num>`; the
XML-tagged parameters are parsed by regex in agent mode, so no LLM parameter
extraction is involved). Its model artifacts (`___command_info/`, ~275 MB) and
the internal CME workflow's artifacts (~550 MB) are gitignored, so a fresh
worktree is **untrained** and the driver refuses to start. Training needs paid
model calls and is out of scope; copy the already-trained artifacts from the
main worktree instead (symlinks inside are relative; `cp -a` keeps them):

```bash
SRC=/home/drawal/rl/fastworkflow
for WT in /tmp/fw-bench-fork /tmp/fw-bench-change4 /home/drawal/rl/fastworkflow-observability; do
  cp -a "$SRC/tests/hello_world_workflow/___command_info" "$WT/tests/hello_world_workflow/"
  cp -a "$SRC/fastworkflow/_workflows/command_metadata_extraction/___command_info" \
        "$WT/fastworkflow/_workflows/command_metadata_extraction/"
done
```

Caveat to keep in mind when reading results: those artifacts were produced by
the current main-worktree code. `CommandRouter` at 9904df5 reads the same
files (`global/{tinymodel.pth,largemodel.pth,label_encoder.pkl,threshold.json,
tiny_ambiguous_threshold.json,large_ambiguous_threshold.json}`), so they load,
but if a checkout rejects them the failure shows up in the warm-up turn
(`notes` / a non-zero exit), not silently. The `___command_info/*.json`
routing files are regenerated by the runtime when stale.

`ls -d <WT>/fastworkflow/_workflows/command_metadata_extraction/___command_info`
must exist in every worktree; the driver only warns about it
(`guards.cme_command_info_present`).

## Exact commands

Interpreter for everything: `/home/drawal/rl/fastworkflow/.venv/bin/python`
(dspy 3.3.0). `BENCH` below is this folder.

```bash
PY=/home/drawal/rl/fastworkflow/.venv/bin/python
BENCH=/home/drawal/rl/fastworkflow-observability/benchmarks/overhead
OUT=$BENCH/results            # or anywhere else; results are gitignored-free JSON
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
```

### 0. Harness self-test (safe; no session, no model)

```bash
cd "$BENCH" && $PY -m pytest -q -p no:cacheprovider test_overhead_harness.py
$PY "$BENCH/run_overhead.py" --help
```

### 1. Worktrees (owner creates; read-only git for agents)

```bash
cd /home/drawal/rl/fastworkflow
git worktree add /tmp/fw-bench-fork    9904df5
git worktree add /tmp/fw-bench-change4 5b1e85e
# the port is measured in place: /home/drawal/rl/fastworkflow-observability
```

Then copy the trained artifacts into each (section above).

### 2. Runs (OWNER GO REQUIRED)

Same `--turns` and `--commands-per-turn` at every point; one process per run.

```bash
# fork, v3.2.0
$PY "$BENCH/run_overhead.py" --checkout /tmp/fw-bench-fork \
    --commit fork-9904df5 --turns 10 --commands-per-turn 48 \
    --output "$OUT/fork-9904df5.json"

# change 4
$PY "$BENCH/run_overhead.py" --checkout /tmp/fw-bench-change4 \
    --commit change4-5b1e85e --turns 10 --commands-per-turn 48 \
    --output "$OUT/change4-5b1e85e.json"

# end of port (after fix-49m.3/.4/.5 land on the 3.3 branch)
$PY "$BENCH/run_overhead.py" --checkout /home/drawal/rl/fastworkflow-observability \
    --commit "port-3.3-$(git -C /home/drawal/rl/fastworkflow-observability rev-parse --short HEAD)" \
    --turns 10 --commands-per-turn 48 \
    --output "$OUT/port-3.3.json"
```

Useful variants:

* Observability off at the same checkout (isolates the sink + substrate cost
  from everything else): add `--observability off` and a different `--output`.
* A capture-profile knob on 5b1e85e+/the port: `--env FW_OBS_CAPTURE_PROFILE=<value>`
  (`--env` entries go into the init dict and `os.environ`; names only are
  recorded in the report, never values).
* Repeats: run the same command 3x with `-1`, `-2`, `-3` suffixed outputs;
  the driver is deliberately one-run-per-process (CommandRouter and sink
  caches are process-wide).
* Keep the state root for inspection: `--keep-state` (path is in the report).
* A different fixture: `--workflow <trained folder> --script commands.json`
  where `commands.json` is a JSON list of command templates cycled to fill a
  turn (`{i}`, `{j}`, `{n}` are substituted).

### 3. Compare

```bash
$PY "$BENCH/compare.py" "$OUT/fork-9904df5.json" "$OUT/change4-5b1e85e.json" "$OUT/port-3.3.json"
```

First file is the baseline; deltas are printed per column.

## Entry points the harness depends on (all present at 9904df5)

Verified with `git show 9904df5:<path>` on 2026-09-06:

| entry point | evidence at 9904df5 |
|---|---|
| `fastworkflow.init(env_vars)` | `fastworkflow/__init__.py:224` |
| `fastworkflow.ChatSession(run_as_agent=...)` | `fastworkflow/chat_session.py:100-103` |
| `ChatSession.start_workflow(path, keep_alive=True)` | `chat_session.py:152` |
| `ChatSession.user_message_queue`, `.command_output_queue`, `.command_trace_queue` (None sentinel per turn) | `chat_session.py:297,301,305`; sentinel `workflow_execution_context.py:1417` |
| `ChatSession.workflow_tool_agent` (`max_iters`, `iteration_counter` attrs) | `chat_session.py:267`; `utils/react.py:71-72` |
| `fastworkflow.observability_store.get_observability_sink(workflow_path)` and `sink.flush()`, `sink.close()`, `sink.store.mint_conversation_id(...)` | `observability_store.py:1814,1530,1539,1275+`; used identically by `run/__main__.py:191-216` |
| `ChatSession._core.set_trace_sink(...)`, `.bind_observability_identity(conversation_id=...)`, `.observability_channel_id` | private, but this is exactly what the CLI does (`run/__main__.py:203-215`); guarded with `set_trace_sink` on ChatSession preferred if it ever appears |
| `fastworkflow.model_pipeline_training.is_workflow_trained(path)` | `model_pipeline_training.py:746` (guarded) |
| `fastworkflow.state_paths.observability_db(path)` | `state_paths.py:134` (guarded, falls back to a glob under the state root) |
| `fastworkflow.utils.dspy_utils.get_lm` (the single LM factory) | `utils/dspy_utils.py:9`; every runtime caller goes through it (`workflow_execution_context.py:1474,1957`, `workflow_agent.py:644`, `utils/signatures.py:290`, `distillation.py`, `conversation_labeling.py:17` binds the name) |
| `dspy.utils.DummyLM` | dspy 3.3.0; same idiom as `tests/test_dspy_observability.py` |

Import-guarded (absent at the fork, mirrored from the CLI when present):
`fastworkflow.runtime_manifest.{check_startup_conformance, deployment_env,
register_runtime_metadata}` (added in 5b1e85e's `run/__main__.py`).

The harness never imports `capture_policy`, `execution_recorder`,
`decision_signals`, `provenance`, `evidence_run` or any other ported module;
it measures whatever the runtime does on its own.

## How the stub LM works

`stub_lm.ScriptedStubLM` subclasses `dspy.utils.DummyLM`. On every call it
parses the output-field list DSPy's adapter appended to the last user message
(`[[ ## field ## ]]` markers with optional "must be formatted as a valid
Python <type>" hints for ChatAdapter and fastWorkflow's
`CommandsSystemPreludeAdapter`; the "order of fields" list for JSONAdapter) and
fabricates a value per field, then formats the reply with the adapter active
in `dspy.settings` so it parses on the way back. Recognised signatures:

* ReAct step (`next_tool_name` present): pops the next command from the turn
  script and answers `execute_workflow_query {"command": ...}`; when the
  script is exhausted it answers `finish {}`.
* extract (`final_answer`), planner (`next_steps`), turn summary
  (`conversation_summary`), `reasoning`: canned strings.
* anything else: a parse-safe default from the type hint (`True`, `0`, `[]`,
  `{}`, first `Literal` choice, or a string).

A turn opens on the first planner/ReAct call after the previous turn's
`finish` and records every call's timestamps; the gaps between consecutive
ReAct calls are the per-command latencies. Nothing in the stub depends on
fastWorkflow, so it behaves identically at all three points.

## Known limitations (things not verifiable without a real run)

* The end-to-end path (agent init -> planner -> ReAct -> `CommandExecutor` ->
  intent detection with the copied artifacts -> sink) has only been reasoned
  through against `git show 9904df5:...`; the first warm-up turn is where a
  mismatch would surface. The harness exits non-zero and writes `notes` if it
  does.
* Spans ride a droppable, bounded queue (`FW_OBS_QUEUE_MAX`); under a stub
  that returns instantly the writer may fall behind and drop spans, which would
  show as fewer `spans` rows than expected. If that happens, re-run with
  `--env FW_OBS_QUEUE_MAX=<large>` at every point and say so in the report.
* Per-turn wall time excludes the final turn's post-sentinel finalize (see
  Metrics); with `--turns 10` the median/p90 are unaffected.
* The ReAct `max_iters` widening relies on the public `max_iters` attribute of
  `fastWorkflowReAct`; if a later checkout caps commands per turn elsewhere
  (e.g. a turn budget), `notes` reports "commands per turn != K" and the
  comparison must use a K every point can honour.
