#!/usr/bin/env python
"""inspect_command_info.py — read-only dump of a workflow's ___command_info/.

Reports, for one workflow directory:
  * command_directory.json: every command, its plain/template utterance counts,
    whether it is parameterized, and whether its <command>_param_labeled.json
    (DSPy few-shot examples) exists and how many valid/rejected examples it has
  * routing_definition.json: contexts and the commands routed in each
  * per-context model artifacts: tinymodel.pth/, largemodel.pth/,
    label_encoder.pkl, threshold.json + tiny/large ambiguous thresholds
    (present/missing, plus the trained threshold values)
  * fingerprint freshness: stamped source_fingerprint vs one recomputed over
    the live _commands tree (see check_cache_freshness.py for the full story)

Usage:
    .venv/bin/python .claude/skills/fastworkflow-diagnostics-and-tooling/scripts/inspect_command_info.py <workflow_dir> [--json]

READ-ONLY guarantee: this script never writes. It deliberately avoids
CommandDirectory.get_commandinfo_folderpath() (which mkdirs ___command_info)
and builds paths by hand. The only fastworkflow import is
compute_commands_source_fingerprint, which is pure (stat + sha256).
Note: `import fastworkflow` takes ~5-10s (it pulls in dspy/torch); pass
--no-fingerprint to skip it and stay stdlib-only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_FILES = [
    "tinymodel.pth",
    "largemodel.pth",
    "label_encoder.pkl",
    "threshold.json",
    "tiny_ambiguous_threshold.json",
    "large_ambiguous_threshold.json",
]
GLOBAL_CONTEXT_FOLDER = "global"  # the '*' context maps to this folder


def internal_cme_contexts() -> set:
    """Context names of the internal command_metadata_extraction workflow.

    These are trained inside fastworkflow's bundled CME workflow, not per app
    workflow. Discovered from the installed package without importing it
    (find_spec on a top-level package does not execute it); falls back to the
    names verified at v2.22.2.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("fastworkflow")
        pkg_dir = Path(list(spec.submodule_search_locations)[0])
        cme_cmds = pkg_dir / "_workflows" / "command_metadata_extraction" / "_commands"
        found = {p.name for p in cme_cmds.iterdir() if p.is_dir() and not p.name.startswith("__")}
        if found:
            return found
    except Exception:
        pass
    return {"IntentDetection", "ErrorCorrection"}


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


def read_threshold(path: Path):
    data = load_json(path)
    if isinstance(data, dict) and "confidence_threshold" in data:
        return data["confidence_threshold"]
    return None


def inspect(workflow_dir: str, with_fingerprint: bool = True) -> dict:
    wf = Path(workflow_dir).resolve()
    info_dir = wf / "___command_info"
    report: dict = {"workflow": str(wf), "command_info_dir": str(info_dir)}

    if not wf.is_dir():
        report["error"] = "workflow directory does not exist"
        return report
    if not info_dir.is_dir():
        report["error"] = (
            "___command_info/ missing — workflow was never built/trained "
            "(run `fastworkflow train <workflow> <env> <passwords>`)"
        )
        return report

    # ---- command_directory.json --------------------------------------
    cmd_dir = load_json(info_dir / "command_directory.json")
    commands = {}
    if isinstance(cmd_dir, dict) and "__error__" not in cmd_dir:
        utt_meta = cmd_dir.get("map_command_2_utterance_metadata", {})
        cmd_meta = cmd_dir.get("map_command_2_metadata", {})
        for name in sorted(set(utt_meta) | set(cmd_meta)):
            um = utt_meta.get(name, {})
            cm = cmd_meta.get(name, {})
            parameterized = bool(cm.get("command_parameters_class"))
            # param_labeled files: ___command_info/<command_name>_param_labeled.json
            # (command names may contain '/', producing nested dirs — see
            # train/__main__.py:127, which joins the raw command_name)
            labeled_path = info_dir / f"{name}_param_labeled.json"
            labeled = load_json(labeled_path)
            labeled_info = None
            if isinstance(labeled, dict) and "__error__" not in labeled:
                labeled_info = {
                    "valid_examples": len(labeled.get("valid_examples", [])),
                    "rejected_examples": len(labeled.get("rejected_examples", [])),
                }
            commands[name] = {
                "plain_utterances": len(um.get("plain_utterances", []) or []),
                "template_utterances": len(um.get("template_utterances", []) or []),
                "parameterized": parameterized,
                "param_labeled_json": labeled_info
                if labeled_info is not None
                else ("MISSING" if parameterized else "n/a"),
            }
        report["command_directory"] = {
            "core_command_names": cmd_dir.get("core_command_names", []),
            "source_fingerprint": cmd_dir.get("source_fingerprint"),
            "commands": commands,
        }
    else:
        report["command_directory"] = cmd_dir or "MISSING"

    # ---- routing_definition.json -------------------------------------
    routing = load_json(info_dir / "routing_definition.json")
    contexts = {}
    if isinstance(routing, dict) and "__error__" not in routing:
        contexts = routing.get("contexts", {})
        report["routing_definition"] = {
            "contexts": {k: sorted(v) for k, v in contexts.items()},
            "source_fingerprint": routing.get("source_fingerprint"),
        }
    else:
        report["routing_definition"] = routing or "MISSING"

    # ---- per-context model artifacts ----------------------------------
    # Trained model folders are the subdirs of ___command_info that contain
    # threshold.json ('global' is the '*' context). Also cross-check against
    # the contexts declared in routing_definition.json.
    models = {}
    declared = {
        (GLOBAL_CONTEXT_FOLDER if c == "*" else c) for c in contexts
    }
    on_disk = {p.name for p in info_dir.iterdir() if p.is_dir()}
    for ctx in sorted(declared | on_disk):
        ctx_dir = info_dir / ctx
        entry = {"declared_in_routing": ctx in declared, "folder_exists": ctx_dir.is_dir()}
        if ctx_dir.is_dir():
            present, missing = [], []
            for fname in MODEL_FILES:
                (present if (ctx_dir / fname).exists() else missing).append(fname)
            entry["present"] = present
            entry["missing"] = missing
            entry["thresholds"] = {
                "confidence_threshold": read_threshold(ctx_dir / "threshold.json"),
                "tiny_ambiguous": read_threshold(ctx_dir / "tiny_ambiguous_threshold.json"),
                "large_ambiguous": read_threshold(ctx_dir / "large_ambiguous_threshold.json"),
            }
        models[ctx] = entry
    report["context_models"] = models

    # A context is "trained" when threshold.json exists (mirrors
    # fastworkflow.model_pipeline_training.is_workflow_trained). Internal
    # command_metadata_extraction (CME) contexts are trained inside the CME
    # workflow, never per app workflow, so their absence here is BY DESIGN
    # (ErrorCorrection additionally has no model at all: its commands are
    # matched only by exact/fuzzy utterance match).
    internal = internal_cme_contexts()
    untrained = [
        c
        for c, e in models.items()
        if e.get("declared_in_routing")
        and c not in internal
        and "threshold.json" not in (e.get("present") or [])
    ]
    report["untrained_declared_contexts"] = untrained

    # ---- fingerprint freshness ----------------------------------------
    if with_fingerprint:
        try:
            from fastworkflow.command_directory import (
                compute_commands_source_fingerprint,
            )

            live = compute_commands_source_fingerprint(str(wf))
            stamped_cd = (
                report["command_directory"].get("source_fingerprint")
                if isinstance(report["command_directory"], dict)
                else None
            )
            stamped_rd = (
                report["routing_definition"].get("source_fingerprint")
                if isinstance(report["routing_definition"], dict)
                else None
            )
            report["fingerprint"] = {
                "live": live,
                "command_directory_json": "FRESH" if stamped_cd == live else "STALE",
                "routing_definition_json": "FRESH" if stamped_rd == live else "STALE",
            }
        except Exception as e:  # keep the inspection usable without the venv
            report["fingerprint"] = f"unavailable ({type(e).__name__}: {e})"

    return report


def print_human(report: dict) -> None:
    print(f"Workflow: {report['workflow']}")
    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return

    cd = report.get("command_directory")
    if isinstance(cd, dict) and "commands" in cd:
        print(f"\nCommands ({len(cd['commands'])}):")
        print(f"  {'command':45s} {'plain':>5s} {'tmpl':>5s} {'params':>6s}  param_labeled")
        for name, c in cd["commands"].items():
            pl = c["param_labeled_json"]
            pl_str = (
                f"{pl['valid_examples']} valid / {pl['rejected_examples']} rejected"
                if isinstance(pl, dict)
                else pl
            )
            print(
                f"  {name:45s} {c['plain_utterances']:5d} {c['template_utterances']:5d} "
                f"{'yes' if c['parameterized'] else 'no':>6s}  {pl_str}"
            )
        print(f"  core commands: {', '.join(cd['core_command_names'])}")
    else:
        print(f"\ncommand_directory.json: {cd}")

    rd = report.get("routing_definition")
    if isinstance(rd, dict) and "contexts" in rd:
        print("\nRouting contexts:")
        for ctx, cmds in rd["contexts"].items():
            print(f"  {ctx}: {len(cmds)} commands")
    else:
        print(f"\nrouting_definition.json: {rd}")

    print("\nModel artifacts per context ('global' == the '*' context):")
    for ctx, e in report["context_models"].items():
        if not e.get("folder_exists"):
            print(f"  {ctx}: NO MODEL FOLDER (declared_in_routing={e['declared_in_routing']})")
            continue
        missing = e.get("missing") or []
        status = "complete" if not missing else f"MISSING: {', '.join(missing)}"
        th = e.get("thresholds", {})
        print(
            f"  {ctx}: {status} | thresholds: conf={th.get('confidence_threshold')} "
            f"tiny_amb={th.get('tiny_ambiguous')} large_amb={th.get('large_ambiguous')}"
        )
    if report.get("untrained_declared_contexts"):
        print(
            "\n  UNTRAINED contexts (declared but no threshold.json): "
            + ", ".join(report["untrained_declared_contexts"])
        )

    fp = report.get("fingerprint")
    if isinstance(fp, dict):
        print(
            f"\nFingerprint: command_directory.json={fp['command_directory_json']} "
            f"routing_definition.json={fp['routing_definition_json']}"
        )
        print(f"  live fingerprint: {fp['live']}")
    elif fp:
        print(f"\nFingerprint: {fp}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow_dir", help="path to the workflow folder (the one containing _commands/)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--no-fingerprint",
        action="store_true",
        help="skip the fingerprint check (avoids the slow `import fastworkflow`)",
    )
    args = ap.parse_args()

    report = inspect(args.workflow_dir, with_fingerprint=not args.no_fingerprint)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print_human(report)
    return 1 if "error" in report else 0


if __name__ == "__main__":
    sys.exit(main())
