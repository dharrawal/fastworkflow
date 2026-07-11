#!/usr/bin/env python
"""trace_turn.py — read a turn's command traces from the places fastworkflow
actually records them. READ-ONLY: parses and pretty-prints, never writes.

fastworkflow records what the agent did in three layers (all verified against
v2.22.2 sources):

1. action.jsonl (CLI debug mirror, cwd of the `fastworkflow run` process).
   Written only when WorkflowExecutionContext is constructed with
   mirror_action_log_to_file=True — ChatSession does this (chat_session.py:118),
   so it exists for CLI runs only. DELETED AND REWRITTEN AT EVERY TURN START
   (workflow_execution_context.py:711-713): it always holds just the LAST turn.
   Two record shapes (both appended by workflow_agent.py):
     tool call : {"command", "command_name", "parameters", "response"}
     ask_user  : {"agent_query", "user_response"}

2. The in-memory action log: WorkflowExecutionContext.append_action_log()
   accumulates the same records per turn (cleared at turn start); the agent's
   final-answer synthesis reads it. Grab it in-process via ctx.action_log.

3. Live CommandTraceEvent queue (chat_session.command_trace_queue /
   WorkflowExecutionContext.command_trace_queue): AGENT_TO_WORKFLOW and
   WORKFLOW_TO_AGENT events emitted around every tool call
   (workflow_agent.py:130-139, 209-218), terminated by a None sentinel per
   turn. The CLI drains it to render the dim yellow/green "Agent >"/"Workflow >"
   lines; the FastAPI turns engine drains it DESTRUCTIVELY into the response's
   "traces" field (run_fastapi_mcp/utils.py:collect_trace_events) — read it
   from the JSON body of /invoke_agent | /invoke_assistant | /initialize.

Usage:
    # Pretty-print an action.jsonl (defaults to ./action.jsonl)
    trace_turn.py actions [path/to/action.jsonl]

    # Pretty-print the traces field of a saved server turn-response body
    #   e.g.: curl -s -X POST .../invoke_agent -H "Authorization: Bearer $T" \
    #              -H 'Content-Type: application/json' \
    #              -d '{"user_query": "..."}' > /tmp/turn.json
    trace_turn.py response /tmp/turn.json

    # Print the capture recipes (where to hook, in code) and exit
    trace_turn.py explain
"""

import argparse
import json
import sys
from pathlib import Path

EXPLAIN = """\
How to capture a turn's trace, per topology:

CLI (Topology A):
  * Run `fastworkflow run <wf> <env> <passwords>`; live traces render
    automatically while the spinner runs. After the turn, `action.jsonl` in the
    CWD holds the last turn's records — copy it aside BEFORE the next turn.

In-process (tests / experiment harnesses, Topology B):
  ctx = fastworkflow.WorkflowExecutionContext(...)   # or via ChatSession
  out = ctx.process_turn(message)                    # TurnOutput
  out.turn_key                # developer handle for the logical turn
  out.status, out.success     # lifecycle vs all-commands-succeeded (orthogonal)
  out.command_outputs         # per-command CommandOutput provenance
  ctx.action_log              # list[dict], this turn's tool-call/ask_user records
  # Live events instead: inject a queue via
  #   ctx.set_transport_queues(command_trace_queue=queue.Queue())
  # then drain it until the None sentinel.

FastAPI server:
  POST /invoke_agent (or /invoke_assistant, /initialize) and read "traces",
  "turn_key", "status", "success", "command_outputs" from the JSON body.
  Caveat (v2.22.2): trace collection is a destructive queue drain — a retried
  request that rejoins an in-flight turn can find "traces" already consumed
  (fix-85g Step 2 will add a replay buffer). /invoke_agent_stream streams
  NDJSON/SSE events instead and is NOT integrated with the turns registry.
"""


def print_action_record(i: int, rec: dict) -> None:
    if "agent_query" in rec:
        print(f"[{i}] ask_user")
        print(f"    agent asked : {rec.get('agent_query')}")
        print(f"    user replied: {rec.get('user_response')}")
        return
    print(f"[{i}] tool call")
    print(f"    agent sent      : {rec.get('command')}")
    print(f"    resolved command: {rec.get('command_name')}")
    print(f"    parameters      : {rec.get('parameters')}")
    resp = str(rec.get("response", ""))
    if len(resp) > 400:
        resp = resp[:400] + f"... [{len(resp)} chars]"
    print(f"    response        : {resp}")


def cmd_actions(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        print(
            f"No {p} found. action.jsonl only exists in the CWD of a CLI "
            "(`fastworkflow run`) process, only in agent mode, and only for the "
            "most recent turn (deleted at each turn start).",
            file=sys.stderr,
        )
        return 1
    print(f"{p} — records for the LAST turn only:\n")
    bad = 0
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            bad += 1
            print(f"[{i}] UNPARSEABLE line ({e}): {line[:120]}")
            continue
        print_action_record(i, rec)
    if bad:
        print(f"\nWARNING: {bad} unparseable line(s) — file may have been "
              "written to concurrently or truncated.")
    return 0


def cmd_response(path: str) -> int:
    p = Path(path)
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Cannot read {p} as JSON: {e}", file=sys.stderr)
        return 1
    for key in ("turn_key", "exec_state", "status", "success", "failure_reason"):
        if key in body:
            print(f"{key:15s}: {body[key]}")
    answer = body.get("answer")
    if answer:
        print(f"{'answer':15s}: {str(answer)[:300]}")
    traces = body.get("traces") or []
    print(f"\ntraces ({len(traces)}):")
    for i, t in enumerate(traces):
        direction = t.get("direction", "?")
        if direction == "agent_to_workflow":
            print(f"[{i}] Agent -> Workflow: {t.get('raw_command')}")
        else:
            ok = t.get("success")
            mark = "OK" if ok else ("FAIL" if ok is False else "?")
            print(
                f"[{i}] Workflow -> Agent [{mark}]: {t.get('command_name')}, "
                f"{t.get('parameters')}"
            )
            resp = str(t.get("response_text") or "")
            print(f"      {resp[:300]}")
    if not traces:
        print(
            "  (empty — deterministic non-agent turn, traces already drained by an\n"
            "   earlier poll of the same turn_key, or the turn deferred with 202)"
        )
    outs = body.get("command_outputs") or []
    if outs:
        print(f"\ncommand_outputs ({len(outs)}):")
        for i, o in enumerate(outs):
            print(f"[{i}] {o.get('command_name') or '(unresolved)'} "
                  f"params={o.get('command_parameters')!r} "
                  f"duration_ms={o.get('duration_ms')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("actions", help="pretty-print an action.jsonl")
    a.add_argument("path", nargs="?", default="action.jsonl")
    r = sub.add_parser("response", help="pretty-print a saved server turn-response JSON body")
    r.add_argument("path")
    sub.add_parser("explain", help="print the capture recipes and exit")
    args = ap.parse_args()

    if args.cmd == "actions":
        return cmd_actions(args.path)
    if args.cmd == "response":
        return cmd_response(args.path)
    print(EXPLAIN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
