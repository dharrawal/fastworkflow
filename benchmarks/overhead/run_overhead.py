#!/usr/bin/env python
"""Overhead benchmark driver for fastWorkflow (bead fix-49m.2).

Runs N identical scripted agent turns in-process through the public
``fastworkflow.ChatSession`` transport (the same queues ``fastworkflow run``
uses), against a stub LM (no network) and a fresh ``FASTWORKFLOW_STATE_ROOT``,
and reports wall time per turn and per command, process CPU time, peak RSS,
observability DB size after flush, and per-table row counts.

Designed to run unchanged at the three comparison points:
  * 9904df5 (v3.2.0 fork)   * 5b1e85e (change 4)   * feat/experiment-observability-3.3 after the port
It therefore depends only on entry points present at 9904df5 and import-guards
everything added later (see README.md, "Entry points").

Usage (from any directory; ``--checkout`` selects which fastworkflow tree is
imported by putting it first on sys.path — the venv's ``fastworkflow.pth``
points at /home/drawal/rl/fastworkflow and would otherwise win):

    /home/drawal/rl/fastworkflow/.venv/bin/python run_overhead.py \
        --checkout /tmp/fw-bench-fork --commit fork-9904df5 \
        --turns 10 --commands-per-turn 48 --output results/fork.json

DO NOT run the measurement without the owner's go (bead fix-49m.2).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import resource
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
STUB_MODEL_NAME = "stub/overhead-benchmark"

# Env-file keys the runtime reads through get_lm(); all routed to the stub.
LLM_ROLE_VARS = (
    "LLM_AGENT", "LLM_PLANNER", "LLM_PARAM_EXTRACTION", "LLM_CONVERSATION_STORE",
    "LLM_SYNDATA_GEN", "LLM_DISTILLATION", "LLM_COMMAND_METADATA_GEN",
)

DEFAULT_SCRIPT_HELLO_WORLD = [
    # tests/hello_world_workflow: one command, two float parameters. XML-tagged
    # parameters are parsed by regex in agent mode (no LLM extraction).
    "add_two_numbers <first_num>{i}</first_num> <second_num>{j}</second_num>",
]


# ---------------------------------------------------------------------------
# Small helpers (pure; unit-tested)
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (pct in [0, 100]); None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = max(1, int(round(pct / 100.0 * len(ordered) + 0.5)))
    return ordered[min(rank, len(ordered)) - 1]


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def ms(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 1000.0, 3)


def build_script(template: list[str], count: int) -> list[str]:
    """Expand ``template`` cyclically to ``count`` commands, filling {i}/{j}."""
    out = []
    for n in range(count):
        line = template[n % len(template)]
        out.append(line.format(i=n + 1, j=n + 2, n=n))
    return out


def read_proc_status() -> dict[str, int]:
    """VmRSS / VmHWM in bytes from /proc/self/status (Linux); {} elsewhere."""
    result: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    result[key] = int(rest.strip().split()[0]) * 1024
    except OSError:
        pass
    return result


def ru_maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return rss * 1024 if platform.system() == "Linux" else rss


def db_size_bytes(db_path: str) -> dict[str, int]:
    sizes = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = db_path + suffix
        if os.path.exists(p):
            sizes[suffix or "main"] = os.path.getsize(p)
    sizes["total"] = sum(v for k, v in sizes.items() if k != "total")
    return sizes


def db_row_counts(db_path: str) -> dict[str, Any]:
    """Row count per table plus a span-name histogram, read-only, schema-agnostic."""
    out: dict[str, Any] = {"tables": {}, "span_names": {}, "spans_by_status": {}}
    if not os.path.exists(db_path):
        out["error"] = "db file missing"
        return out
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        out["error"] = repr(exc)
        return out
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        for name in names:
            try:
                out["tables"][name] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except sqlite3.Error as exc:
                out["tables"][name] = f"error: {exc}"
        if "spans" in names:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(spans)")}
            if "name" in cols:
                out["span_names"] = {
                    r[0]: r[1] for r in conn.execute(
                        "SELECT name, COUNT(*) FROM spans GROUP BY name ORDER BY name")
                }
            if "status" in cols:
                out["spans_by_status"] = {
                    r[0]: r[1] for r in conn.execute(
                        "SELECT status, COUNT(*) FROM spans GROUP BY status ORDER BY status")
                }
            # Best-effort per-command durations from the store itself, as a
            # cross-check on the stub-observed gaps. Column names vary by
            # version; try the common shapes and give up quietly.
            if "name" in cols:
                dur_expr = None
                if "duration_ms" in cols:
                    dur_expr = "duration_ms"
                elif {"started_at_ms", "ended_at_ms"} <= cols:
                    dur_expr = "(ended_at_ms - started_at_ms)"
                elif {"start_ms", "end_ms"} <= cols:
                    dur_expr = "(end_ms - start_ms)"
                if dur_expr:
                    rows = conn.execute(
                        f"SELECT {dur_expr} FROM spans WHERE name = 'fw.command.execute' AND {dur_expr} IS NOT NULL"
                    ).fetchall()
                    vals = [float(r[0]) for r in rows]
                    out["command_execute_span_ms"] = stats(vals)
    finally:
        conn.close()
    return out


def git_head(checkout: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key, cmd in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("describe", ["git", "describe", "--always", "--dirty"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            info[key] = subprocess.run(
                cmd, cwd=checkout, check=True, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception as exc:  # git missing / not a repo — informational only
            info[key] = f"unavailable ({type(exc).__name__})"
    return info


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkout", default=os.getcwd(),
                   help="fastWorkflow checkout to benchmark (inserted at sys.path[0]). Default: cwd")
    p.add_argument("--commit", required=True,
                   help="label for this checkout in the results (e.g. fork-9904df5, change4-5b1e85e, port-3.3)")
    p.add_argument("--workflow", default=None,
                   help="trained workflow folder. Default: <checkout>/tests/hello_world_workflow")
    p.add_argument("--env-file", default=None,
                   help="fastworkflow env file for message templates. Default: <checkout>/fastworkflow/examples/fastworkflow.env")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env-file entries (repeatable), e.g. --env FW_OBS_CAPTURE_PROFILE=full")
    p.add_argument("--turns", type=int, default=10, help="measured turns (default 10)")
    p.add_argument("--commands-per-turn", type=int, default=48,
                   help="scripted commands per turn (default 48 -> ~52 model calls/turn)")
    p.add_argument("--warmup-turns", type=int, default=1, help="unmeasured warm-up turns (default 1)")
    p.add_argument("--warmup-commands", type=int, default=2, help="commands per warm-up turn (default 2)")
    p.add_argument("--script", default=None,
                   help="JSON file: list of command templates cycled to fill a turn ({i},{j},{n} substituted)")
    p.add_argument("--observability", choices=("on", "off"), default="on",
                   help="FW_OBSERVABILITY switch for this run (default on)")
    p.add_argument("--state-root", default=None,
                   help="FASTWORKFLOW_STATE_ROOT for this run. Default: fresh temp dir")
    p.add_argument("--keep-state", action="store_true", help="do not delete the temp state root")
    p.add_argument("--output", default=None, help="write the JSON report here (default: stdout only)")
    p.add_argument("--turn-timeout", type=float, default=600.0, help="seconds to wait for one turn")
    p.add_argument("--no-tripwire", action="store_true",
                   help="do not patch litellm/dspy.LM to fail on real calls (default: patched)")
    p.add_argument("--skip-manifest-conformance", action="store_true",
                   help="do not mirror the CLI's runtime-manifest conformance step (5b1e85e+)")
    return p.parse_args(argv)


def build_env_vars(args: argparse.Namespace, state_root: str, checkout: str) -> tuple[dict[str, str], list[str]]:
    notes: list[str] = []
    env_vars: dict[str, str] = {}
    env_file = args.env_file or os.path.join(checkout, "fastworkflow", "examples", "fastworkflow.env")
    if os.path.isfile(env_file):
        try:
            from dotenv import dotenv_values
            env_vars.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
            notes.append(f"env file loaded: {env_file}")
        except Exception as exc:
            notes.append(f"env file not loaded ({exc!r}): {env_file}")
    else:
        notes.append(f"env file absent: {env_file}")
    for var in LLM_ROLE_VARS:
        env_vars[var] = STUB_MODEL_NAME
    env_vars["FASTWORKFLOW_STATE_ROOT"] = state_root
    env_vars["FW_OBSERVABILITY"] = "1" if args.observability == "on" else "0"
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        env_vars[k] = v
    return env_vars, notes


def drain_startup(chat_session, timeout: float = 0.5) -> None:
    """Mirror the CLI: consume any startup output and trace events."""
    try:
        chat_session.command_output_queue.get(timeout=timeout)
    except queue.Empty:
        pass
    trace_q = getattr(chat_session, "command_trace_queue", None)
    if trace_q is None:
        return
    while True:
        try:
            evt = trace_q.get_nowait()
        except queue.Empty:
            break
        if evt is None:
            break


def wait_for_turn(chat_session, timeout: float) -> dict[str, Any]:
    """Block until the worker signals turn completion (None sentinel on the
    trace queue, as the CLI does), then collect the queued CommandOutput."""
    trace_q = chat_session.command_trace_queue
    deadline = time.perf_counter() + timeout
    events = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"turn did not complete within {timeout}s")
        try:
            evt = trace_q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if evt is None:
            break
        events += 1
    t_done = time.perf_counter()
    info: dict[str, Any] = {"trace_events": events, "t_done": t_done}
    try:
        out = chat_session.command_output_queue.get(timeout=5.0)
        resp = getattr(getattr(out, "command_response", None), "response", None)
        info["success"] = bool(getattr(out, "success", True))
        info["response_chars"] = len(resp) if isinstance(resp, str) else None
        info["awaiting_user"] = bool(
            getattr(getattr(out, "command_response", None), "artifacts", {}).get("awaiting_user", False)
        )
    except queue.Empty:
        info["success"] = None
        info["output_missing"] = True
    return info


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    checkout = os.path.abspath(args.checkout)
    if not os.path.isdir(os.path.join(checkout, "fastworkflow")):
        raise SystemExit(f"--checkout {checkout} has no fastworkflow/ package directory")
    workflow = os.path.abspath(args.workflow or os.path.join(checkout, "tests", "hello_world_workflow"))
    if not os.path.isdir(workflow):
        raise SystemExit(f"workflow folder not found: {workflow}")

    temp_root = None
    if args.state_root:
        state_root = os.path.abspath(args.state_root)
        os.makedirs(state_root, exist_ok=True)
    else:
        temp_root = tempfile.mkdtemp(prefix="fw-overhead-state-")
        state_root = temp_root

    # Offline guards BEFORE anything imports transformers/huggingface.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["FASTWORKFLOW_STATE_ROOT"] = state_root
    os.environ["FW_OBSERVABILITY"] = "1" if args.observability == "on" else "0"
    for item in args.env:
        if "=" in item:
            k, v = item.split("=", 1)
            os.environ[k] = v

    # Select the checkout under test.
    sys.path.insert(0, checkout)
    sys.path.insert(0, str(HERE))
    import fastworkflow  # noqa: E402
    fw_file = os.path.abspath(fastworkflow.__file__)
    if not fw_file.startswith(checkout + os.sep):
        raise SystemExit(
            f"imported fastworkflow from {fw_file}, not from --checkout {checkout}; "
            "another install is shadowing the checkout"
        )
    import dspy  # noqa: E402
    from stub_lm import NetworkTripwire, ScriptedStubLM, install_stub  # noqa: E402

    report: dict[str, Any] = {
        "commit_label": args.commit,
        "checkout": checkout,
        "git": git_head(checkout),
        "fastworkflow_file": fw_file,
        "workflow": workflow,
        "python": sys.version.split()[0],
        "dspy_version": getattr(dspy, "__version__", "unknown"),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "state_root": state_root,
        "observability": args.observability,
        "config": {
            "turns": args.turns,
            "commands_per_turn": args.commands_per_turn,
            "warmup_turns": args.warmup_turns,
            "warmup_commands": args.warmup_commands,
            "env_overrides": list(args.env),
            "tripwire": not args.no_tripwire,
        },
        "notes": [],
        "guards": {},
    }
    notes = report["notes"]

    env_vars, env_notes = build_env_vars(args, state_root, checkout)
    notes.extend(env_notes)
    report["env_keys"] = sorted(env_vars.keys())  # names only; never values

    t_init0 = time.perf_counter()
    fastworkflow.init(env_vars=env_vars)
    report["init_s"] = time.perf_counter() - t_init0

    # Mirror the CLI's runtime-manifest conformance step (present from 5b1e85e;
    # absent at the fork). A workflow without a manifest passes with every
    # feature off — this only matters so the port's execution path matches
    # `fastworkflow run`.
    report["guards"]["runtime_manifest"] = "skipped"
    if not args.skip_manifest_conformance:
        try:
            from fastworkflow.runtime_manifest import (  # type: ignore
                check_startup_conformance, deployment_env, register_runtime_metadata,
            )
        except Exception:
            report["guards"]["runtime_manifest"] = "module absent (pre-5b1e85e)"
        else:
            try:
                register_runtime_metadata(
                    workflow,
                    check_startup_conformance(
                        workflow, env=deployment_env(getattr(fastworkflow, "_env_vars", env_vars))
                    ),
                )
                report["guards"]["runtime_manifest"] = "applied"
            except Exception as exc:
                report["guards"]["runtime_manifest"] = f"failed: {exc!r}"
                notes.append(f"runtime manifest conformance failed: {exc!r}")

    # Fail fast on an untrained fixture, as the CLI does.
    try:
        from fastworkflow.model_pipeline_training import is_workflow_trained
        trained, missing = is_workflow_trained(workflow)
    except Exception as exc:
        trained, missing = True, [f"check unavailable: {exc!r}"]
        report["guards"]["is_workflow_trained"] = "unavailable"
    else:
        report["guards"]["is_workflow_trained"] = trained
    if not trained:
        raise SystemExit(
            f"workflow is not trained (missing contexts: {missing}). Copy the trained\n"
            f"artifacts into this checkout first — see README.md 'Trained artifacts'."
        )
    cme_info = os.path.join(checkout, "fastworkflow", "_workflows", "command_metadata_extraction", "___command_info")
    report["guards"]["cme_command_info_present"] = os.path.isdir(cme_info)
    if not os.path.isdir(cme_info):
        notes.append("internal CME workflow has no ___command_info; copy it too (README 'Trained artifacts')")

    # Stub LM + tripwire.
    template = DEFAULT_SCRIPT_HELLO_WORLD
    if args.script:
        with open(args.script, "r", encoding="utf-8") as fh:
            template = json.load(fh)
        if not isinstance(template, list) or not all(isinstance(x, str) for x in template):
            raise SystemExit("--script must be a JSON list of strings")
    measured_script = build_script(template, args.commands_per_turn)
    warmup_script = build_script(template, args.warmup_commands)
    report["config"]["script_template"] = template

    stub = ScriptedStubLM(warmup_script if args.warmup_turns else measured_script)
    report["patch_points"] = install_stub(stub)
    tripwire = None if args.no_tripwire else NetworkTripwire().install()

    # Session wiring, mirroring fastworkflow/run/__main__.py.
    t_sess0 = time.perf_counter()
    chat_session = fastworkflow.ChatSession(run_as_agent=True)
    sink = None
    core = getattr(chat_session, "_core", None)
    if args.observability == "on":
        try:
            from fastworkflow.observability.store import get_observability_sink
            sink = get_observability_sink(workflow)
        except Exception as exc:
            notes.append(f"get_observability_sink unavailable: {exc!r}")
        if sink is not None:
            setter = getattr(chat_session, "set_trace_sink", None) or getattr(core, "set_trace_sink", None)
            if setter is None:
                notes.append("no set_trace_sink on ChatSession/_core; sink NOT bound")
            else:
                setter(sink)
                report["guards"]["trace_sink_bound"] = True
            try:
                core.bind_observability_identity(
                    conversation_id=sink.store.mint_conversation_id(core.observability_channel_id)
                )
                report["guards"]["conversation_id_minted"] = True
            except Exception as exc:
                report["guards"]["conversation_id_minted"] = f"failed: {exc!r}"
        else:
            notes.append("observability sink is None although --observability on (FW_OBSERVABILITY env?)")
    chat_session.start_workflow(workflow, keep_alive=True)
    drain_startup(chat_session)
    report["session_start_s"] = time.perf_counter() - t_sess0

    db_path = None
    try:
        from fastworkflow import state_paths
        db_path = state_paths.observability_db(workflow)
    except Exception:
        for cand in Path(state_root).rglob("observability.sqlite3"):
            db_path = str(cand)
            break
    report["observability_db"] = db_path

    message = "Run the scripted commands."

    def run_turn(timeout: float) -> dict[str, Any]:
        chat_session.user_message_queue.put(message)
        return wait_for_turn(chat_session, timeout)

    # Warm-up: lazy agent init, first CommandRouter use, first sink writes.
    warmups: list[dict[str, Any]] = []
    for _ in range(args.warmup_turns):
        t0 = time.perf_counter()
        info = run_turn(args.turn_timeout)
        info["wall_s"] = info.pop("t_done") - t0
        warmups.append(info)
    report["warmup"] = warmups
    stub.close_turn()

    # Widen the ReAct iteration budget so the script, not max_iters, ends the
    # loop (the agent's counter is not reset between turns at 9904df5).
    agent = getattr(chat_session, "workflow_tool_agent", None)
    if agent is not None and hasattr(agent, "max_iters"):
        report["guards"]["react_max_iters_original"] = getattr(agent, "max_iters")
        agent.max_iters = (args.turns + args.warmup_turns + 1) * (args.commands_per_turn + 2) + 100
        report["guards"]["react_max_iters_set"] = agent.max_iters
        if hasattr(agent, "iteration_counter"):
            agent.iteration_counter = 0
    else:
        report["guards"]["react_max_iters_original"] = "agent or max_iters attribute not found"
        notes.append("could not widen ReAct max_iters; commands per turn may be capped by the default")

    stub.set_script(measured_script)
    stub_turns_before = len(stub.turns)

    # ---- measured section -------------------------------------------------
    rss_before = read_proc_status()
    cpu0 = time.process_time()
    t_start = time.perf_counter()
    turns: list[dict[str, Any]] = []
    error: Optional[str] = None
    prev_done = t_start
    try:
        for i in range(args.turns):
            chat_session.user_message_queue.put(message)
        for i in range(args.turns):
            info = wait_for_turn(chat_session, args.turn_timeout)
            t_done = info.pop("t_done")
            info["index"] = i
            info["wall_s"] = t_done - prev_done
            prev_done = t_done
            turns.append(info)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    t_turns_done = time.perf_counter()

    flush_s = None
    if sink is not None:
        t_f0 = time.perf_counter()
        try:
            flushed = sink.flush()
        except Exception as exc:
            flushed = f"error: {exc!r}"
        flush_s = time.perf_counter() - t_f0
        report["guards"]["sink_flush_result"] = flushed
    t_end = time.perf_counter()
    cpu1 = time.process_time()
    rss_after = read_proc_status()
    stub.close_turn()

    # ---- results ----------------------------------------------------------
    stub_summary = stub.summary()
    # Stub turns are opened by the LM traffic; everything before the measured
    # section (warm-up) is sliced off by count, not by assumption.
    stub_turns = stub_summary["turns"][stub_turns_before:]
    turn_walls = [t["wall_s"] for t in turns]
    command_gaps = [g for st in stub_turns for g in st["command_gaps_s"]]
    commands_issued = [st["commands_issued"] for st in stub_turns]
    model_calls = [st["model_calls"] for st in stub_turns]
    stub_time = [st["stub_time_s"] for st in stub_turns]
    per_turn_cmd_wall = [
        (t["wall_s"] / c) for t, c in zip(turns, commands_issued) if c
    ]

    report["error"] = error
    report["turns"] = turns
    report["stub"] = stub_summary
    report["tripwire_trips"] = tripwire.trips if tripwire else None
    report["metrics"] = {
        "turn_wall_ms": {k: ms(v) if k != "n" else v for k, v in stats(turn_walls).items()},
        "command_gap_ms": {k: ms(v) if k != "n" else v for k, v in stats(command_gaps).items()},
        "turn_wall_per_command_ms": {k: ms(v) if k != "n" else v for k, v in stats(per_turn_cmd_wall).items()},
        "stub_lm_time_per_turn_ms": {k: ms(v) if k != "n" else v for k, v in stats(stub_time).items()},
        "commands_per_turn": stats([float(c) for c in commands_issued]),
        "model_calls_per_turn": stats([float(c) for c in model_calls]),
        "total_wall_s": t_end - t_start,
        "turns_wall_s": t_turns_done - t_start,
        "flush_s": flush_s,
        "process_cpu_s": cpu1 - cpu0,
        "process_cpu_per_turn_s": (cpu1 - cpu0) / len(turns) if turns else None,
        "rss_before_bytes": rss_before.get("VmRSS"),
        "rss_after_bytes": rss_after.get("VmRSS"),
        "peak_rss_bytes": rss_after.get("VmHWM") or ru_maxrss_bytes(),
        "ru_maxrss_bytes": ru_maxrss_bytes(),
    }
    if db_path:
        report["observability_db_bytes"] = db_size_bytes(db_path)
        report["observability_rows"] = db_row_counts(db_path)
    else:
        report["observability_db_bytes"] = None
        report["observability_rows"] = None

    close_s = None
    if sink is not None:
        t_c0 = time.perf_counter()
        try:
            sink.close()
        except Exception as exc:
            notes.append(f"sink.close failed: {exc!r}")
        close_s = time.perf_counter() - t_c0
        if db_path:
            report["observability_db_bytes_after_close"] = db_size_bytes(db_path)
    report["metrics"]["close_s"] = close_s
    if tripwire:
        tripwire.uninstall()

    expected = args.commands_per_turn
    if any(c != expected for c in commands_issued):
        notes.append(f"commands per turn != {expected}: {commands_issued} (budget/cap on this checkout?)")
    if any(t.get("success") is False for t in turns):
        notes.append("one or more turns returned success=False")
    if error:
        notes.append(f"measurement aborted: {error}")

    summary = one_line(report)
    report["summary"] = summary
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    print(summary)

    if temp_root and not args.keep_state:
        shutil.rmtree(temp_root, ignore_errors=True)
    return 1 if error else 0


def one_line(report: dict[str, Any]) -> str:
    m = report["metrics"]
    rows = report.get("observability_rows") or {}
    tables = rows.get("tables", {}) if isinstance(rows, dict) else {}
    dbb = report.get("observability_db_bytes") or {}

    def fmt(v, unit="", scale=1.0, nd=1):
        return "n/a" if v is None else f"{v / scale:.{nd}f}{unit}"

    return (
        f"[{report['commit_label']}] obs={report['observability']} "
        f"turns={m['turn_wall_ms']['n']} cmds/turn={fmt(m['commands_per_turn']['median'], nd=0)} "
        f"calls/turn={fmt(m['model_calls_per_turn']['median'], nd=0)} | "
        f"turn p50={fmt(m['turn_wall_ms']['median'], 'ms')} p90={fmt(m['turn_wall_ms']['p90'], 'ms')} | "
        f"cmd p50={fmt(m['command_gap_ms']['median'], 'ms', nd=2)} p90={fmt(m['command_gap_ms']['p90'], 'ms', nd=2)} | "
        f"cpu={fmt(m['process_cpu_s'], 's', nd=2)} peak_rss={fmt(m['peak_rss_bytes'], 'MB', 1024 * 1024)} | "
        f"db={fmt(dbb.get('total'), 'KB', 1024)} spans={tables.get('spans', 'n/a')} turns_rows={tables.get('turns', 'n/a')} | "
        f"tripwire={report.get('tripwire_trips')} err={'yes' if report.get('error') else 'no'}"
    )


if __name__ == "__main__":
    sys.exit(main())
