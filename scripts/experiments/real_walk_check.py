#!/usr/bin/env python3
"""Run the product's REAL detection walk on the cases ``remote_link_worth.py local`` wrote, and record what it did.

Why: ``remote_link_worth.py analyse`` scores configurations with an offline re-implementation of the walk
(``walk``/``evaluate``). This script drives the real code on a trained copy so the two can be compared
(``real_walk_compare.py``).

Per case it records, with NO remote model call (``JEV_API_KEY`` is dropped from the env dict handed to
``fastworkflow.init`` AND from ``os.environ``; the run stops if ``intent_detector.matcher_available()`` is true):

* ``per_scope``: the real ``ClassifierEngine`` answer (``intent_detector.engine_for``, the product's constructor, so the
  reachable-label hand-on is the product's) of every scope the case can reach, with the head's numbers;
* ``detect_local``: ``intent_detector.detect(workflow, utterance)`` with its default engines (the walk the simulation
  claims to re-implement), the scopes it asked in order and the answers' kinds;
* ``decide_local``: ``intent_loop.resolve(workflow, utterance, core=False)`` (what ``part_intent`` runs for a turn):
  phase 1 the local-only ``detect`` (engines without the hand-on), phase 2 the chain walk ``_walk`` (engines with the
  hand-on; with no key the chain is the local model alone) and the global scope asked last, then the planning-command
  second opinion (a confident local answer that names a planning command is an ambiguity without a key);
* with ``--answers``: ``decide_chain``, the same decision with a key "present" and the matcher's call replaced
  by the stored distribution of that case (``intent_detector._matcher_distribution``; the engine classes, the
  decision thresholds and the walk are the product's). A case with no stored answer answers as a failed call does.

The workflow object is a stand-in with the attributes the walk reads (``folderpath``, ``root``, ``get_parent``,
``_current_command_context``), whose records are instances of classes named like the contexts, chained as the
case's ``path`` says; ``fastworkflow.Workflow.get_command_context_name`` is the product's own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GLOBAL = "*"


class _StandIn:
    """What ``intent_detector.walk``/``detect`` and ``intent_loop.resolve`` read from a workflow."""

    def __init__(self, folder: str, chain: list, root):
        self.folderpath = folder
        self.root = root
        self._current_command_context = chain[0] if chain else None
        self._chain = chain

    @property
    def _current_command_context_name(self) -> str:
        import fastworkflow

        return fastworkflow.Workflow.get_command_context_name(self._current_command_context)

    def get_parent(self, record=None):
        if record is self.root or record is None:
            return self.root
        return getattr(record, "_parent", None)


_CLASSES: dict = {}


def _record(name: str):
    if name not in _CLASSES:
        _CLASSES[name] = type(name, (), {})
    return _CLASSES[name]()


def _stand_in(folder: str, path: list[str], root_mode: str) -> _StandIn:
    """Records for the contexts of *path* (the global scope left out), nearest first. ``chain-end``: the last
    record is the workflow's root. ``dummy``: the root is a record of its own and the top of the chain has no
    parent (its ``get_parent`` returns None: back to global)."""
    names = [n for n in path if n != GLOBAL]
    chain = [_record(n) for n in names]
    for child, parent in zip(chain, chain[1:]):
        child._parent = parent
    if chain:
        chain[-1]._parent = None
    if root_mode == "chain-end":
        root = chain[-1] if chain else _record("Root")
    else:
        root = _record("Root")
    return _StandIn(folder, chain, root)


def _numbers(answer) -> dict:
    d = answer.details or {}
    return {"kind": answer.kind, "labels": list(answer.labels), "confident": bool(answer.confident),
            "p_in": d.get("p_in"), "p_out": d.get("p_out"), "confidence": d.get("confidence"),
            "topk": list(zip(d.get("topk_labels", ()), d.get("topk_scores", ()))),
            "outside": list(zip(d.get("outside_labels", ())[:3], d.get("outside_scores", ())[:3])),
            "handed_on": list(d.get("handed_on", ())), "engine": answer.engine}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True, help="cases_*.json written by remote_link_worth.py local")
    ap.add_argument("--env", required=True)
    ap.add_argument("--passwords", required=True)
    ap.add_argument("--root", choices=("chain-end", "dummy"), required=True,
                    help="chain-end: the last record of the path is the workflow root (a root context); dummy: the root "
                         "is a record apart and the top of the path has no parent")
    ap.add_argument("--answers", help="answers.jsonl of stored remote distributions: also run the chain replay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from dotenv import dotenv_values

    env = {**dotenv_values(args.env), **dotenv_values(args.passwords)}
    env.pop("JEV_API_KEY", None)
    env["FW_LM_CACHE"] = "0"
    import fastworkflow
    from fastworkflow import intent_detector as idt
    from fastworkflow import intent_loop
    from fastworkflow import workflow_agent
    from dspy.utils import DummyLM

    # Importing the product loads a dotenv file into os.environ (a key may come back), so the drop is done AFTER the
    # imports, and init is given an env dict without the key.
    os.environ.pop("JEV_API_KEY", None)
    os.environ.pop("FW_REQUIRE_HEADS", None)
    fastworkflow.init(env_vars=env)

    if idt.matcher_available():
        print("STOP: the remote decision model is available (JEV_API_KEY effective)", file=sys.stderr)
        return 2
    data = json.load(open(args.cases, encoding="utf-8"))
    header, cases = data["header"], data["cases"]
    folder = header["folder"]
    entry_only = frozenset(header["entry_only"])
    stored: dict = {}
    if args.answers:
        for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r and r["workflow"] == header["name"]:
                    stored[(r["context"], r["utterance"])] = r
    if args.limit:
        cases = cases[: args.limit]

    # -- instrumentation: which scope was asked, in which phase, with which answer
    log: list = []
    state = {"phase": 1, "walk_detection": None, "current": None, "replay": False, "missing": 0, "pathmis": 0,
             "excl_mis": 0}
    real_engine_for = idt.engine_for
    real_walk = intent_loop._walk

    class _Proxy:
        def __init__(self, context, inner):
            self.context, self.inner = context, inner
            self.name = getattr(inner, "name", "")

        def decide(self, utterance):
            answer = self.inner.decide(utterance)
            log.append({"phase": state["phase"], "scope": self.context, "kind": answer.kind,
                        "labels": list(answer.labels)[:3], "engine": answer.engine})
            return answer

    def engine_for(wf_folder, context_name, *a, **k):
        engine = real_engine_for(wf_folder, context_name, *a, **k)
        return None if engine is None else _Proxy(context_name, engine)

    def walk_marked(workflow, text, engine_of, path, global_last, *rest, **named):
        state["phase"] = 2
        detection = real_walk(workflow, text, engine_of, path, global_last, *rest, **named)
        state["walk_detection"] = detection
        return detection

    def matcher_distribution(wf_folder, scope_path, utterance, exclude=frozenset()):
        case = state["current"]
        row = stored.get((case["context"], utterance))
        if row is None:
            state["missing"] += 1
            return {"scores": {}, "model": None, "latency_ms": 0.0, "calls": 0, "error": "no stored answer"}
        if list(scope_path) != list(row["scope_path"]):
            state["pathmis"] += 1
        if sorted(exclude) != sorted(row.get("exclude", [])):
            state["excl_mis"] += 1
        return {"scores": row["scores"], "model": row.get("model"), "latency_ms": 0.0, "calls": 1}

    idt.engine_for = engine_for
    intent_loop._walk = walk_marked
    real_jev = workflow_agent._jev_lm
    real_dist = idt._matcher_distribution

    def run_decide(stand, utterance):
        log.clear()
        state.update(phase=1, walk_detection=None)
        intent = intent_loop.resolve(stand, utterance, core=False)
        det = state["walk_detection"]
        phase2 = [e for e in log if e["phase"] == 2]
        decider = (phase2 or log or [None])[-1]
        last_kind = det.answers[-1].kind if det is not None and det.answers else None
        return {"outcome": intent.outcome, "command": intent.command, "ckind": intent.kind,
                "decided_by": intent.decided_by, "ambiguous": list(intent.ambiguous or ()),
                "candidates": [c.name for c in intent.candidates][:5],
                "asked": [(e["phase"], e["scope"], e["kind"]) for e in log],
                "deciding_scope": decider["scope"] if decider else None,
                "walk_ran": det is not None, "walk_last_kind": last_kind,
                "walk_outcome": det.outcome if det is not None else None,
                "bare": last_kind == idt.BARE_VALUE}

    rows = []
    try:
        for index, case in enumerate(cases):
            state["current"] = case
            path = list(case["path"])
            stand = _stand_in(folder, path, args.root)
            real_path = idt.walk_path(stand)
            utt = case["utterance"]
            # per-scope real engine answers (the product's engine_for; no remote model)
            per_scope = {}
            scope_list = list(dict.fromkeys([case["context"], *path, GLOBAL] if case["context"] != GLOBAL else [GLOBAL]))
            memo: dict = {}
            for scope in scope_list:
                engine = real_engine_for(folder, scope, scope_path=real_path, memo=memo)
                if engine is not None:
                    per_scope[scope] = _numbers(engine.decide(utt))
            # intent_detector.detect, default engines
            log.clear()
            state["phase"] = 1
            detection = idt.detect(stand, utt)
            detect_local = {"outcome": detection.outcome, "command": detection.command,
                            "candidates": list(detection.candidates),
                            "scope": detection.scope.context_name if detection.scope else None,
                            "asked": [(e["scope"], e["kind"]) for e in log],
                            "answer_kinds": [a.kind for a in detection.answers],
                            "bare": bool(detection.answers) and detection.answers[-1].kind == idt.BARE_VALUE}
            decide_local = run_decide(stand, utt)
            row = {"id": case["id"], "set": case["set"], "kind": case["kind"], "context": case["context"],
                   "utterance": utt, "expected": case["expected"], "path": path, "real_path": list(real_path),
                   "per_scope": per_scope, "detect_local": detect_local, "decide_local": decide_local}
            if args.answers:
                workflow_agent._jev_lm = lambda: DummyLM([])
                idt._matcher_distribution = matcher_distribution
                state["missing"] = 0
                try:
                    if not idt.matcher_available():
                        print("STOP: replay could not make the matcher available", file=sys.stderr)
                        return 2
                    row["decide_chain"] = run_decide(stand, utt)
                    row["decide_chain"]["missing_stored"] = state["missing"]
                    row["decide_chain"]["has_stored"] = (case["context"], utt) in stored
                finally:
                    workflow_agent._jev_lm = real_jev
                    idt._matcher_distribution = real_dist
            rows.append(row)
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(cases)}", flush=True)
    finally:
        idt.engine_for = real_engine_for
        intent_loop._walk = real_walk
        workflow_agent._jev_lm = real_jev
        idt._matcher_distribution = real_dist
    Path(args.out).write_text(json.dumps({"header": {k: header[k] for k in ("name", "family", "folder", "core", "entry_only")},
                                          "replay_path_mismatches": state["pathmis"],
                                          "replay_exclude_mismatches": state["excl_mis"], "rows": rows}, indent=0),
                              encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} cases; replay scope-path mismatches {state['pathmis']}, exclude mismatches {state['excl_mis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
