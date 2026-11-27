#!/usr/bin/env python3
"""Does the remote decision model (the chain's second link) earn its place behind the local intent model?

Four steps, the paid one apart from the free ones:

    local    train (optionally) a temp copy of a workflow offline, then record, for every benchmark case, the local
             model's answer at every scope the walk asks (``intent_detector.ClassifierEngine``, the product's
             reachable-label hand-on). Lists the cases the local model does NOT answer with one command or a
             bare value at the commit threshold (those are the cases the chain would ask the remote model about).
             No model call: a cold utterance cache stops the run, ``JEV_API_KEY`` is removed from the process.
    ask      ONE remote call per listed case, the way the product asks it: ``MatcherEngine(folder, context,
             scope_path, exclude=<commands that only enter a context>).distribution(utterance)`` is the call
             ``intent_loop.resolve(core=False)`` makes through ``engine_for`` (same ``_ask_matcher`` underneath).
             Every raw answer (all option scores) is appended to a JSONL file as it arrives; a rerun skips stored
             cases; a hard budget counts the attempts; it stops on a credit/auth error or 5 failures in a row.
    verify   free: the offline re-implementation of ``MatcherEngine.decide`` used by ``analyse`` is compared with the
             product class on every stored answer (and, with the local answers replayed, ``ChainEngine``).
    analyse  free: from the stored answers, score configurations (local alone; the chain at chosen thresholds; a
             threshold pair picked on development by a fixed rule; an agreement gate) by simulating the product's
             walk. See ``--help`` of each step.

The simulated walk (``intent_loop.resolve(core=False)`` and ``intent_detector.detect``):
phase 1 is the local-only walk (global scope first when it has a head, then the case's context, then its
enclosing contexts); a confident non-core command anywhere decides, with no remote call. Otherwise phase 2, the chain
walk: each scope answers with its local model when that is confident (one command or a bare value), else with the
remote model's answer for that scope read from the one stored distribution over the whole path (accept / decline
thresholds as configured; a failed call leaves the local answer standing).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GLOBAL = "*"
UNSURE = 0.2          # intent_detector.MATCHER_UNSURE_THRESHOLD
MAX_CANDIDATES = 3    # intent_detector.MATCHER_MAX_CANDIDATES
NONE = "none"
FATAL_MARKERS = ("402", "401", "403", "credit", "unauthorized", "authentication", "payment", "insufficient",
                 "quota", "billing", "api key", "api_key", "forbidden")


# ============================================================================ local


def _train_copy(workflow: str, copy_root: str) -> str:
    """Train a temp copy of *workflow* with the product trainer, embedding layout, no model call."""
    import fastworkflow
    import single_intent_model as runner
    from fastworkflow import model_pipeline_training as heads
    from fastworkflow.train import determinism, utterance_cache
    from fastworkflow.train import intent_model_layout as layout
    from fastworkflow.train import workflow_model as wm
    import embedding_head as E

    class A:
        encoder = layout.DEFAULT_ENCODER
        pooling = None
        commit_threshold = None

    env = E._environment(workflow, A)
    env.pop("JEV_API_KEY", None)
    fastworkflow.init(env_vars=env)
    copy = runner.copy_workflow(workflow, copy_root)
    cache = wm._ReadOnlyUtteranceCache(copy, mode=utterance_cache.MODE_REUSE)
    handle = fastworkflow.Workflow.create(copy, workflow_id_str="remote_worth_" + os.path.basename(copy))
    previous = (determinism.get_provenance_recorder(), utterance_cache.get_utterance_cache())
    try:
        determinism.set_provenance_recorder(determinism.ProvenanceRecorder(copy))
        utterance_cache.set_utterance_cache(cache)
        with wm.no_model_calls():
            heads.train(handle, intent_model_layout=layout.EMBEDDING)
    finally:
        determinism.set_provenance_recorder(previous[0])
        utterance_cache.set_utterance_cache(previous[1])
        handle.close()
    return copy


def _entry_only(folder: str) -> list[str]:
    from fastworkflow.core import context_navigation

    try:
        return sorted(context_navigation.context_entry_commands(folder))
    except Exception:  # noqa: BLE001 - as intent_loop._entry_only: unreadable declaration leaves nothing out
        return []


def cmd_local(args) -> int:
    from dotenv import dotenv_values

    import fastworkflow

    workflow = os.path.abspath(args.workflow)
    if args.train:
        copy = _train_copy(workflow, args.copy_root)
    else:
        copy = os.path.abspath(args.trained_copy)
    env = {**dotenv_values(args.env or os.path.join(workflow, "fastworkflow.env")),
           **dotenv_values(args.passwords or os.path.join(workflow, "fastworkflow.passwords.env"))}
    env.pop("JEV_API_KEY", None)           # the local step never asks a remote model
    os.environ.pop("JEV_API_KEY", None)
    env["FW_LM_CACHE"] = "0"
    fastworkflow.init(env_vars=env)
    from fastworkflow.intent import intent_detector as idt
    from fastworkflow import model_pipeline_training as heads
    from fastworkflow import workflow_agent
    from fastworkflow.intent.nlu_labels import PARAMETER_VALUE_LABEL, is_escalation, label_of
    from fastworkflow.train import artifact_versioning
    from fastworkflow.train import heldout_evaluation as he

    info = os.path.join(copy, "___command_info")
    scopes_path = os.path.join(info, "workflow", "scopes.json")
    scopes = json.load(open(scopes_path, encoding="utf-8"))
    summary = {"encoder": scopes.get("encoder"), "commit_threshold": scopes.get("threshold"),
               "schema_version": scopes.get("schema_version"), "decision": scopes.get("decision"),
               "layout": scopes.get("layout")}
    print("scopes.json:", json.dumps(summary))
    if args.expect_threshold is not None:
        assert abs(float(scopes["threshold"]) - args.expect_threshold) < 1e-9, "unexpected commit threshold"
        assert scopes.get("schema_version") == 2, "unexpected schema version"
        assert args.expect_encoder in json.dumps(scopes.get("encoder")), "unexpected encoder"

    contexts = {}
    for name in sorted(os.listdir(info)):
        folder = os.path.join(info, name)
        if os.path.isfile(os.path.join(folder, "threshold.json")):
            contexts[idt.GLOBAL_CONTEXT if name == "global" else name] = folder
    routers = {context: heads.CommandRouter(folder) for context, folder in contexts.items()}
    assert artifact_versioning.context_folder_name(GLOBAL) == "global"

    paths = {}
    if args.paths_file:
        paths = {k: list(v) for k, v in json.load(open(args.paths_file, encoding="utf-8")).items()}
    core = sorted(workflow_agent.core_command_names())
    entry_only = _entry_only(copy)

    def path_of(context: str) -> list[str]:
        path = list(paths.get(context, [context]))
        if context not in path:
            path.insert(0, context)
        if context != GLOBAL and GLOBAL in contexts and GLOBAL not in path:
            path.append(GLOBAL)
        return path

    reach_cache: dict = {}

    def reachable(path):
        key = tuple(path)
        if key not in reach_cache:
            reach_cache[key] = idt.reachable_labels(copy, key)
        return reach_cache[key]

    def commands_of(labels):
        return [l for l in labels if not is_escalation(l) and label_of(l) != PARAMETER_VALUE_LABEL]

    out_cases = []
    own_by_scope: dict[str, list[str]] = {}
    for set_name, files in (item.split("=", 1) for item in args.sets):
        set_cases = [case for file in files.split(",") for case in he.load_benchmark_file(file)]
        for index, case in enumerate(set_cases):
            path = path_of(case.context)
            scope_list = list(dict.fromkeys([case.context, *path, GLOBAL] if case.context != GLOBAL else [GLOBAL]))
            local = {}
            for scope in scope_list:
                if scope not in routers:
                    continue
                engine = idt.ClassifierEngine(routers[scope], reachable=lambda p=tuple(path): reachable(p))
                answer = engine.decide(case.utterance)
                labels = [he.normalize_label(str(l)) for l in answer.labels]
                local[scope] = {"kind": answer.kind, "labels": labels, "commands": commands_of(labels),
                                "confident": bool(idt.head_is_confident(answer)),
                                "scores": {str(k): float(v) for k, v in answer.scores.items()},
                                "handed_on": list(answer.details.get("handed_on", ()))}
            for scope in path:
                if scope not in own_by_scope:
                    try:
                        own_by_scope[scope] = [c.name for c in idt.matcher_candidates(copy, scope)
                                               if c.name not in set(entry_only)]
                    except ValueError:
                        own_by_scope[scope] = []
            expected = (he.normalize_label(case.expected_label or "") if case.kind == "routing"
                        else he.normalize_label(case.expected_ancestor_command or ""))
            out_cases.append({
                "id": f"{args.name}|{set_name}|{index}", "workflow": args.name, "set": set_name, "kind": case.kind,
                "context": case.context, "utterance": case.utterance, "expected": expected, "path": path,
                "global_first": case.context != GLOBAL and GLOBAL in contexts,
                "local": local, "listed": not local.get(case.context, {}).get("confident", False),
            })
    header = {"name": args.name, "family": args.family, "folder": copy, "scopes_json": summary,
              "core": core, "entry_only": entry_only, "own_by_scope": own_by_scope,
              "contexts_with_a_head": sorted(contexts)}
    Path(args.out).write_text(json.dumps({"header": header, "cases": out_cases}, indent=1), encoding="utf-8")
    listed = [c for c in out_cases if c["listed"]]
    print(f"wrote {args.out}: {len(out_cases)} cases")
    for set_name in sorted({c["set"] for c in out_cases}):
        rows = [c for c in out_cases if c["set"] == set_name]
        lst = [c for c in rows if c["listed"]]
        bare = sum(1 for c in rows if not c["listed"] and c["local"][c["context"]]["kind"] == idt.BARE_VALUE)
        wild = sum(1 for c in lst if c["local"].get(c["context"], {}).get("labels") == ["wildcard"])
        print(f"  [{args.name}] set {set_name}: {len(rows)} cases ({sum(c['kind']=='routing' for c in rows)} routing, "
              f"{sum(c['kind']=='escalation' for c in rows)} escalation); listed (not answered locally): {len(lst)} "
              f"({sum(c['kind']=='routing' for c in lst)} routing, {sum(c['kind']=='escalation' for c in lst)} "
              f"escalation; {wild} of them the vague-reply class alone); bare-value answers (not listed): {bare}")
    print(f"  [{args.name}] LISTED TOTAL: {len(listed)}")
    return 0


# ============================================================================ ask


def cmd_ask(args) -> int:
    from dotenv import dotenv_values

    env = {**dotenv_values(args.env), **dotenv_values(args.passwords)}
    if not env.get("JEV_API_KEY"):
        print("JEV_API_KEY is not set in the env files given.", file=sys.stderr)
        return 1
    env["FW_LM_CACHE"] = "0"
    import fastworkflow

    fastworkflow.init(env_vars=env)
    from fastworkflow.intent import intent_detector as idt

    data = json.load(open(args.cases, encoding="utf-8"))
    header, cases = data["header"], data["cases"]
    store = Path(args.answers)
    counter_path = Path(args.counter)
    done: set[str] = set()
    spent_before = 0
    if store.exists():
        for line in store.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(row["id"])
    counter = json.loads(counter_path.read_text()) if counter_path.exists() else {"attempts": 0, "failed": 0}
    spent_before = counter["attempts"]
    todo = [c for c in cases if c["listed"] and c["id"] not in done]
    if args.set:
        todo = [c for c in todo if c["set"] in args.set]
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(done)} stored, {len(todo)} to ask now; attempts counted so far {spent_before} (budget {args.budget})")
    if spent_before + len(todo) > args.budget:
        print(f"STOP: {spent_before} + {len(todo)} would exceed the budget of {args.budget}.", file=sys.stderr)
        return 2

    lock = threading.Lock()
    state = {"stop": None, "fail_run": 0}
    original = idt._ask_matcher

    def counted(utterance, groups):
        with lock:
            if counter["attempts"] >= args.budget:
                raise RuntimeError("budget exhausted")      # surfaces as a failed call; never reached by the check above
            counter["attempts"] += 1
            counter_path.write_text(json.dumps(counter), encoding="utf-8")
        return original(utterance, groups)

    idt._ask_matcher = counted
    entry_only = frozenset(header["entry_only"])
    folder = header["folder"]

    def one(case):
        if state["stop"]:
            return None
        engine = idt.MatcherEngine(folder, case["context"], tuple(case["path"]), fallback=False, exclude=entry_only)
        answered = engine.distribution(case["utterance"])
        scores = {str(k): float(v) for k, v in answered.get("scores", {}).items()}
        row = {"id": case["id"], "workflow": case["workflow"], "set": case["set"], "kind": case["kind"],
               "context": case["context"], "utterance": case["utterance"], "expected": case["expected"],
               "scope_path": list(engine.scope_path), "exclude": sorted(entry_only), "scores": scores,
               "options": len(scores), "choice": max(scores, key=scores.get) if scores else None,
               "model": answered.get("model"), "latency_ms": answered.get("latency_ms"),
               "calls": answered.get("calls"), "per_scope": bool(answered.get("per_scope")),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if "error" in answered:
            row["error"] = answered["error"]
        with lock:
            with store.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if "error" in row:
                counter["failed"] += 1
                state["fail_run"] += 1
                counter_path.write_text(json.dumps(counter), encoding="utf-8")
                low = row["error"].lower()
                if any(m in low for m in FATAL_MARKERS):
                    state["stop"] = f"payment/credit/auth-looking error: {row['error'][:200]}"
                elif state["fail_run"] >= 5:
                    state["stop"] = f"5 consecutive failures: {row['error'][:200]}"
            else:
                state["fail_run"] = 0
        return row

    n = 0
    with futures.ThreadPoolExecutor(max_workers=min(4, args.workers)) as pool:
        for row in pool.map(one, todo):
            if row is None:
                continue
            n += 1
            if args.verbose or n <= 3:
                print(json.dumps(row)[:1500])
            elif n % 25 == 0:
                print(f"  {n}/{len(todo)} asked; attempts {counter['attempts']}; failed {counter['failed']}", flush=True)
    print(f"asked {n}; attempts total {counter['attempts']}; failed total {counter['failed']}")
    if state["stop"]:
        print("STOPPED:", state["stop"], file=sys.stderr)
        return 3
    return 0


# ============================================================================ the matcher's decision, offline


def remote_answer(scope: str, scores: dict, own: list[str], accept: float, decline: float) -> dict:
    """``MatcherEngine.decide`` for *scope* from a stored distribution, with the two commit numbers apart."""
    own_scores = {name: float(scores.get(name, 0.0)) for name in own}
    ranked = sorted(own, key=lambda name: -own_scores[name])
    none_score = float(scores.get(NONE, 0.0))
    elsewhere = max((s for o, s in scores.items() if o != NONE and o not in own_scores), default=0.0)
    best = own_scores[ranked[0]] if ranked else 0.0
    kept = [n for n in ranked if own_scores[n] > UNSURE][:MAX_CANDIDATES]
    if ranked and best >= accept:
        return {"kind": "match", "labels": [ranked[0]], "commands": [ranked[0]], "by": "remote"}
    if none_score >= decline or elsewhere >= decline or not kept:
        return {"kind": "not_mine", "labels": ["wildcard"], "commands": [], "by": "remote"}
    return {"kind": "ambiguous", "labels": kept, "commands": kept, "by": "remote"}


# ============================================================================ verify


def cmd_verify(args) -> int:
    from dotenv import dotenv_values

    env = {**dotenv_values(args.env), **dotenv_values(args.passwords)}
    env["FW_LM_CACHE"] = "0"
    import fastworkflow

    fastworkflow.init(env_vars=env)
    from fastworkflow.intent import intent_detector as idt

    rows = {}
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    data = json.load(open(args.cases, encoding="utf-8"))
    header = data["header"]
    entry_only = frozenset(header["entry_only"])
    mism = checked = 0
    chain_mism = 0
    real = idt._matcher_distribution
    for case in data["cases"]:
        r = rows.get(case["id"])
        if r is None or "error" in r:
            continue
        idt._matcher_distribution = lambda f, p, u, exclude=frozenset(), _r=r: {
            "scores": _r["scores"], "model": _r["model"], "latency_ms": 0.0, "calls": 1}
        engine = idt.MatcherEngine(header["folder"], case["context"], tuple(case["path"]), exclude=entry_only)
        a = engine.decide(case["utterance"])
        mine = remote_answer(case["context"], r["scores"], header["own_by_scope"][case["context"]],
                              idt.remote_accept_threshold(), idt.remote_decline_threshold())
        checked += 1
        if (a.kind, list(a.labels)) != (mine["kind"], mine["labels"]):
            mism += 1
            print("MISMATCH", case["id"], a.kind, a.labels, mine)

        class Replay:
            def predict_with_details(self, utterance, _c=case):
                return list(_c["local"][_c["context"]]["labels"]), {}

        chain = idt.ChainEngine(idt.ClassifierEngine(Replay()), engine).decide(case["utterance"])
        loc = case["local"][case["context"]]
        expect = (loc["kind"], loc["labels"]) if loc["confident"] else (mine["kind"], mine["labels"])
        if (chain.kind, list(chain.labels)) != expect:
            chain_mism += 1
            print("CHAIN MISMATCH", case["id"], chain.kind, chain.labels, expect)
    idt._matcher_distribution = real
    print(f"verified {checked} stored answers: matcher decide mismatches {mism}, chain mismatches {chain_mism}")
    return 0 if not mism and not chain_mism else 1


# ============================================================================ analyse


class Data:
    def __init__(self, case_files, answers_path):
        self.cases = []
        self.header = {}
        for item in case_files:
            data = json.load(open(item, encoding="utf-8"))
            h = data["header"]
            self.header[h["name"]] = h
            for c in data["cases"]:
                c["family"] = h["family"]
                self.cases.append(c)
        self.answers = {}
        self.failed = []
        for line in Path(answers_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" in r:
                    self.failed.append(r)
                else:
                    self.answers[r["id"]] = r

    def core(self, case):
        return set(self.header[case["workflow"]]["core"])

    def own(self, case, scope):
        return self.header[case["workflow"]]["own_by_scope"].get(scope, [])


def local_answer(case, scope):
    loc = case["local"].get(scope)
    if loc is None:
        return {"kind": "not_mine", "labels": [], "commands": [], "by": "none"}
    return {"kind": loc["kind"], "labels": loc["labels"], "commands": loc["commands"], "by": "local",
            "confident": loc["confident"]}


def walk(data: Data, case, ans_of):
    """``intent_detector.walk`` + ``detect``'s outcome: ('match', cmd, by) | ('ambiguous', cands) | ('bare',) | ('none',)."""
    core = data.core(case)
    context = case["context"]
    scopes = [s for s in case["path"] if s != GLOBAL] if context != GLOBAL else [GLOBAL]
    if case["global_first"]:
        g = ans_of(GLOBAL)
        if g["kind"] == "match" and g["labels"][0] not in core:
            return ("match", g["labels"][0], g["by"])
    ans = ans_of(scopes[0])
    is_core = ans["kind"] == "match" and ans["labels"][0] in core
    if ans["kind"] != "ambiguous" and not is_core:
        i = 1
        while ans["kind"] not in ("match", "bare_value") and i < len(scopes):
            ans = ans_of(scopes[i])
            i += 1
    if ans["kind"] == "match":
        return ("match", ans["labels"][0], ans["by"])
    if ans["kind"] == "ambiguous":
        return ("ambiguous", [l for l in ans["commands"]])
    if ans["kind"] == "bare_value":
        return ("bare",)
    return ("none",)


def is_expected(case, command: str) -> bool:
    expected = case["expected"]
    if case["kind"] == "routing":
        return command == expected
    return command == expected or command.split("/")[-1] == expected.split("/")[-1]


def make_cfg(kind, accept=0.5, decline=0.5):
    return {"kind": kind, "accept": accept, "decline": decline}


def evaluate(data: Data, case, cfg):
    """The outcome of one case under *cfg* (None = local alone). Returns (class, by, detail)."""
    local_of = lambda scope: local_answer(case, scope)  # noqa: E731
    det = walk(data, case, local_of)
    if cfg is not None and case["listed"] and not (det[0] == "match" and det[1] not in data.core(case)):
        dist = data.answers.get(case["id"])

        def chain_of(scope):
            loc = local_answer(case, scope)
            if loc.get("confident"):
                return loc
            if dist is None:
                return loc                                  # the call failed or was not made: the local answer stands
            own = data.own(case, scope)
            r = remote_answer(scope, dist["scores"], own, cfg["accept"], cfg["decline"])
            if cfg["kind"] == "gate" and r["kind"] == "match":
                top = loc["commands"][0] if loc["commands"] else None
                if r["labels"][0] != top:
                    cands = list(dict.fromkeys([*loc["commands"], r["labels"][0]]))
                    return {"kind": "ambiguous", "labels": cands, "commands": cands, "by": "remote_gate"}
            return r

        det = walk(data, case, chain_of)
    if det[0] == "match":
        cls = "right" if is_expected(case, det[1]) else "wrong"
        return cls, det[2], det
    if det[0] == "bare":
        return "wrong", "local", det
    if det[0] == "ambiguous":
        ok = any(is_expected(case, c) for c in det[1])
        return ("q_right" if ok else "q_none"), "question", det
    return "q_none", "question", det


def tally(data: Data, cases, cfg) -> dict:
    out = {}
    for name in ("routing", "escalation", "pooled"):
        out[name] = {"n": 0, "right": 0, "wrong": 0, "q_right": 0, "q_none": 0, "added_wrong": 0,
                     "wrong_remote": 0, "wrong_remote_right_in_local": 0}
    for case in cases:
        cls, by, det = evaluate(data, case, cfg)
        for name in (case["kind"], "pooled"):
            row = out[name]
            row["n"] += 1
            row[cls] += 1
            if cls == "wrong" and str(by).startswith("remote"):
                row["added_wrong"] += 1
                row["wrong_remote"] += 1
                local_cands = case["local"].get(case["context"], {}).get("commands", [])
                if any(is_expected(case, c) for c in local_cands):
                    row["wrong_remote_right_in_local"] += 1
    return out


def pct(a, n):
    return f"{100.0 * a / n:5.1f}%" if n else "  n/a"


def fmt_row(label, t) -> str:
    cols = []
    for name in ("routing", "escalation", "pooled"):
        r = t[name]
        cols.append(f"{name[:4]} n={r['n']:<4} R {r['right']:<4}({pct(r['right'], r['n'])}) W {r['wrong']:<3}({pct(r['wrong'], r['n'])}) "
                    f"Q+ {r['q_right']:<4} Q- {r['q_none']:<4}")
    return f"{label:<26}" + " | ".join(cols)


GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def pick(grid_rows, n_all: int):
    """The fixed rule: among points whose added wrong-confident answers are at most 1.0% of all cases, the one with
    the most right answers (ties: higher accept, then higher decline); none qualify: the fewest wrong."""
    ok = [g for g in grid_rows if g["tally"]["pooled"]["added_wrong"] <= 0.01 * n_all]
    if ok:
        return max(ok, key=lambda g: (g["tally"]["pooled"]["right"], g["accept"], g["decline"])), True
    return min(grid_rows, key=lambda g: (g["tally"]["pooled"]["wrong"], -g["tally"]["pooled"]["right"],
                                         -g["accept"], -g["decline"])), False


def add_tallies(a, b):
    out = {}
    for name in a:
        out[name] = {k: a[name][k] + b[name][k] for k in a[name]}
    return out


def agreement(data: Data, cases) -> dict:
    out = {"agree": [0, 0], "disagree_remote": [0, 0], "disagree_local_top": [0, 0], "no_local_top": [0, 0],
           "remote_none": [0, 0], "cases": 0}
    for case in cases:
        d = data.answers.get(case["id"])
        if d is None or not case["listed"]:
            continue
        out["cases"] += 1
        choice = d["choice"]
        loc = case["local"].get(case["context"], {})
        top = loc["commands"][0] if loc.get("commands") else None
        right = choice is not None and choice != NONE and is_expected(case, choice)
        if choice == NONE:
            out["remote_none"][0] += 1
            out["remote_none"][1] += 0
            continue
        if top is None:
            out["no_local_top"][0] += 1
            out["no_local_top"][1] += right
        elif choice == top:
            out["agree"][0] += 1
            out["agree"][1] += right
        else:
            out["disagree_remote"][0] += 1
            out["disagree_remote"][1] += right
            out["disagree_local_top"][0] += 1
            out["disagree_local_top"][1] += is_expected(case, top)
    return out


def cmd_analyse(args) -> int:
    data = Data(args.cases, args.answers)
    result: dict = {}
    families = sorted({c["family"] for c in data.cases})
    dev = {f: [c for c in data.cases if c["family"] == f and c["set"] == "dev"] for f in families}
    conf = {f: [c for c in data.cases if c["family"] == f and c["set"] == "conf"] for f in families}
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("LEGEND: R right, W wrong confident (a single wrong answer acted on, bare value counted), Q+ question with the right "
        "command among the candidates, Q- question without it.")
    say(f"stored answers {len(data.answers)}, failed calls {len(data.failed)}")
    for f in families:
        say(f"family {f}: dev {len(dev[f])} cases ({sum(c['listed'] for c in dev[f])} listed), "
            f"conf {len(conf[f])} ({sum(c['listed'] for c in conf[f])} listed)")

    grids, picked = {}, {}
    for f in families:
        rows = []
        for acc in GRID:
            for dec in GRID:
                rows.append({"accept": acc, "decline": dec, "tally": tally(data, dev[f], make_cfg("chain", acc, dec))})
        grids[f] = rows
        base = tally(data, dev[f], None)
        say(f"\n=== development grid, family {f}  (N={len(dev[f])}; added-wrong limit {0.01 * len(dev[f]):.1f}); "
            f"pooled routing+escalation: right / wrong / question; added wrong = wrong answers the remote link gave")
        say(f"local alone (A): right {base['pooled']['right']}  wrong {base['pooled']['wrong']}  "
            f"question {base['pooled']['q_right'] + base['pooled']['q_none']}")
        say("accept decline |  right  wrong question (Q+/Q-) | added_wrong")
        for g in rows:
            p = g["tally"]["pooled"]
            say(f"  {g['accept']:.1f}   {g['decline']:.1f}   | {p['right']:5d} {p['wrong']:5d} {p['q_right'] + p['q_none']:5d} "
                f"({p['q_right']}/{p['q_none']}) | {p['added_wrong']}")
        best, qualified = pick(rows, len(dev[f]))
        picked[f] = (best["accept"], best["decline"])
        say(f"PICK {f}: accept {best['accept']}, decline {best['decline']} "
            f"({'qualified' if qualified else 'NO point qualified: fewest wrong'})")
    joint_rows = []
    if len(families) > 1:
        for i, (acc, dec) in enumerate((a, d) for a in GRID for d in GRID):
            t = None
            for f in families:
                row = next(g for g in grids[f] if g["accept"] == acc and g["decline"] == dec)
                t = row["tally"] if t is None else add_tallies(t, row["tally"])
            joint_rows.append({"accept": acc, "decline": dec, "tally": t})
        n_all = sum(len(dev[f]) for f in families)
        best, qualified = pick(joint_rows, n_all)
        picked["joint"] = (best["accept"], best["decline"])
        say(f"\nJOINT PICK (one pair for both families, same rule on the pooled development sets, N={n_all}): "
            f"accept {best['accept']}, decline {best['decline']} ({'qualified' if qualified else 'none qualified'})")

    configs = [("A local alone", None), ("B 0.5/0.5", make_cfg("chain", 0.5, 0.5)),
               ("C 0.6/0.8", make_cfg("chain", 0.6, 0.8))]
    for f in families:
        configs_f = [*configs, (f"D {picked[f][0]}/{picked[f][1]} (own family)", make_cfg("chain", *picked[f]))]
        if "joint" in picked:
            configs_f.append((f"D-joint {picked['joint'][0]}/{picked['joint'][1]}", make_cfg("chain", *picked["joint"])))
        configs_f.append(("E agreement gate 0.5/0.5", make_cfg("gate", 0.5, 0.5)))
        for set_name, group in (("DEVELOPMENT", dev[f]), ("CONFIRMATION", conf[f])):
            say(f"\n=== {set_name}, family {f} (N={len(group)}); E is defined before looking at confirmation; "
                f"D is picked on development only")
            for label, cfg in configs_f:
                t = tally(data, group, cfg)
                say(fmt_row(label, t))
                result.setdefault(f, {}).setdefault(set_name, {})[label] = t
            for label, cfg in configs_f[1:]:
                t = result[f][set_name][label]["pooled"]
                say(f"    {label}: remote wrong answers {t['wrong_remote']}, of them the right command was among "
                    f"the local candidates {t['wrong_remote_right_in_local']}")
        for set_name, group in (("development", dev[f]), ("confirmation", conf[f])):
            ag = agreement(data, group)
            say(f"\n--- remote vs local top, family {f}, {set_name} (listed cases with a stored answer: {ag['cases']}; "
                f"remote's own top option; escalation cases count right when the ancestor command is chosen)")
            say(f"  agree (same command): n={ag['agree'][0]} right {ag['agree'][1]} ({pct(ag['agree'][1], ag['agree'][0])})")
            say(f"  disagree: n={ag['disagree_remote'][0]} remote right {ag['disagree_remote'][1]} "
                f"({pct(ag['disagree_remote'][1], ag['disagree_remote'][0])}); local top right {ag['disagree_local_top'][1]} "
                f"({pct(ag['disagree_local_top'][1], ag['disagree_local_top'][0])})")
            say(f"  no local command candidate (vague-reply class alone): n={ag['no_local_top'][0]} remote right "
                f"{ag['no_local_top'][1]} ({pct(ag['no_local_top'][1], ag['no_local_top'][0])}); remote said none: "
                f"{ag['remote_none'][0]}")
            result[f].setdefault("agreement", {})[set_name] = ag
    lat = {}
    for f in families:
        ls = [r["latency_ms"] for r in data.answers.values() if data.header[r["workflow"]]["family"] == f]
        fl = [r for r in data.failed if data.header[r["workflow"]]["family"] == f]
        if ls:
            ls.sort()
            p95 = ls[min(len(ls) - 1, math.ceil(0.95 * len(ls)) - 1)]
            lat[f] = {"n": len(ls), "median_ms": statistics.median(ls), "p95_ms": p95, "failed": len(fl)}
            say(f"\nlatency, family {f}: n={len(ls)} median {statistics.median(ls):.0f} ms, p95 {p95:.0f} ms; "
                f"failed calls {len(fl)}")
    per_scope = sum(1 for r in data.answers.values() if r.get("per_scope"))
    say(f"answers asked per scope (more than 64 options): {per_scope}")
    result["grids"] = grids
    result["picked"] = picked
    result["latency"] = lat
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


# ============================================================================ main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("local", help="record the local model's answers and list the cases it does not answer")
    s.add_argument("--workflow", required=True, help="the workflow folder (env files are read from it by default)")
    s.add_argument("--name", required=True)
    s.add_argument("--family", required=True, help="the group the workflow belongs to in the analysis")
    s.add_argument("--train", action="store_true", help="train a temp copy offline under --copy-root")
    s.add_argument("--copy-root")
    s.add_argument("--trained-copy", help="an already trained copy (without --train)")
    s.add_argument("--sets", nargs="+", required=True, help="NAME=file1.json[,file2.json...] ... (names: dev, conf)")
    s.add_argument("--paths-file", help="JSON {context: [context, enclosing contexts...]}")
    s.add_argument("--env")
    s.add_argument("--passwords")
    s.add_argument("--expect-threshold", type=float)
    s.add_argument("--expect-encoder", default="intfloat/e5-base-v2")
    s.add_argument("--out", required=True)
    s = sub.add_parser("ask", help="one remote call per listed case (PAID)")
    s.add_argument("--cases", required=True)
    s.add_argument("--answers", required=True, help="JSONL of raw answers (appended; a rerun skips stored cases)")
    s.add_argument("--counter", required=True, help="durable JSON counter of attempts and failures")
    s.add_argument("--env", required=True)
    s.add_argument("--passwords", required=True)
    s.add_argument("--budget", type=int, required=True, help="hard cap on attempts, counted across runs by --counter")
    s.add_argument("--limit", type=int)
    s.add_argument("--set", nargs="*")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--verbose", action="store_true")
    s = sub.add_parser("verify", help="free: compare the offline decision with the product's on stored answers")
    s.add_argument("--cases", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--env", required=True)
    s.add_argument("--passwords", required=True)
    s = sub.add_parser("analyse", help="free: configurations A to E from the stored answers")
    s.add_argument("--cases", nargs="+", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"local": cmd_local, "ask": cmd_ask, "verify": cmd_verify, "analyse": cmd_analyse}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
