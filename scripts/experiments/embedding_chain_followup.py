#!/usr/bin/env python3
"""EXPLORATORY follow-up to ``docs/single_intent_model_preregistration_v5.md`` (bead fix-o5wi.4).

Written after the confirmation sets were scored once, so nothing here is a registered result. Three
measurements on the confirmation cases, with the same metrics as the v5 document:

* the local head alone, BASE (as shipped) and V4 alone (confirmed escalation, k = 10, no agreement gate);
* the old per-context classifier pair on the same cases (seeds rebuilt under a scratch folder, or a
  previous trained version, scored read-only);
* the whole detection chain: the local head decides when its answer is one command or a bare value, every
  other answer goes to the remote decision model over the scope path. ``intent_detector.ChainEngine``,
  ``ClassifierEngine`` and ``MatcherEngine`` are used as they are; only the local head's answer is replayed
  (a router that returns the answer ``heads`` recorded) and the remote model's answer is read from, or
  written to, a recording file, so a rerun asks nothing twice.

Steps (``heads`` and ``report`` need the framework's ``.venv``; ``old`` and ``chain`` run in the
environment of the workflow they score):

    heads   --data DIR [DIR ...] --out FILE [--cache DIR]
    old     --workflow PATH --cases FILE --out FILE [--name NAME] (--seeds-root DIR [--arm H3] --seeds 42 43 44 | --version-dir DIR)
    chain   --workflow PATH --name NAME --heads FILE --env FILE [--passwords FILE ...] --recording FILE --out FILE
            [--paths-file FILE] [--star]
    report  --heads FILE --chain FILE [FILE ...] --old FILE [FILE ...]
    heads --set dev ; chain [--arms BASE] [--thresholds M,D] [--grid]   (second follow-up: tuning the remote link's two thresholds)
    grid    --chain FILE [FILE ...]        development sweep and the pair the rule chooses
    errors  --chain FILE [FILE ...]        types and confidence of the remote model's wrong answers
    tuned   --chain FILE ... --tuned FILE ... [--label M/D]

``--star`` appends the global scope to every other context's scope path (what ``walk_path`` does when the
workflow has a trained global head). ``chain`` makes LIVE remote-model calls for every answer not in the
recording (the model cache is off); it stops when 5 calls in a row fail. A data directory is the one
``embedding_commit_rule.py dump`` writes (``rows.json``, ``cases_conf.json``); a directory whose name starts
with ``large`` is the "large" group, every other one is pooled into "examples".
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WILDCARD, PARAMETER_VALUE = "wildcard", "parameter_value"
V4 = dict(grouping="text", gate=0, confirm=10, wrong=0.01, esc=0.005)
ARMS = ("BASE", "V4")


def group_of(name: str) -> str:
    return "large" if name.startswith("large") else "examples"


# --------------------------------------------------------------------------- verdicts (shared)


def verdict(kind: str, expected: str | None, predicted: list[str]) -> dict:
    """The per-case flags of the v5 document (``embedding_commit_rule.verdicts``)."""
    lone, routing = len(predicted) == 1, kind == "routing"
    return {
        "kind": kind,
        "top1": routing and predicted == [expected],
        "in_list": routing and expected in predicted,
        "one_label": lone,
        "lone_wrong": routing and lone and predicted != [expected],
        "lone_right": routing and lone and predicted == [expected],
        "false_escalation": routing and lone and predicted == [WILDCARD],
        "escalated": kind == "escalation" and lone and predicted == [WILDCARD],
    }


def summarise(v: list[dict]) -> dict:
    routing = [x for x in v if x["kind"] == "routing"]
    esc = [x for x in v if x["kind"] == "escalation"]
    n = max(len(routing), 1)
    lone = sum(x["one_label"] for x in routing)
    return {
        "routing": len(routing), "escalation": len(esc),
        "top1": sum(x["top1"] for x in routing) / n,
        "in_list": sum(x["in_list"] for x in routing) / n,
        "one_label": lone / n,
        "wrong_lone": sum(x["lone_wrong"] for x in routing) / n,
        "false_esc": sum(x["false_escalation"] for x in routing) / n,
        "esc_recall": [sum(x["escalated"] for x in esc), len(esc)],
        "precision": (sum(x["lone_right"] for x in routing) / lone) if lone else None,
    }


def record(case, predicted: list[str]) -> dict:
    routing = case.kind == "routing"
    return {"workflow": None, "context": case.context, "kind": case.kind, "utterance": case.utterance,
            "expected": case.expected_label.strip() if routing else (case.expected_ancestor_command or ""),
            "predicted": predicted}


# --------------------------------------------------------------------------- heads


def _heads(args) -> int:
    import embedding_commit_rule as ecr

    out: dict = {arm: {} for arm in ARMS}
    for folder in args.data:
        data = ecr.Data(folder, args.cache)
        cases = ecr.load_cases(folder, args.set)
        feats = data.case_features(args.set, cases)
        for arm, config in (("BASE", ecr.BASE_CONFIG), ("V4", ecr.make(
                dict(grouping=V4["grouping"], gate=V4["gate"], confirm=V4["confirm"]), V4["wrong"], V4["esc"]))):
            cal = ecr.calibrate(data.records(config["grouping"]), config)
            rows = []
            for f in feats:
                row = record(f["case"], ecr.decide(f, config, cal["threshold"], cal["margin"]))
                row["workflow"] = data.name
                rows.append(row)
            out[arm][data.name] = rows
            print(f"{arm} {data.name}: threshold {cal['threshold']:.3f} margin {cal['margin']:.3f}, {len(rows)} cases")
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    return 0


# --------------------------------------------------------------------------- old pair


def _old(args) -> int:
    import fastworkflow
    from fastworkflow.intent.router import CommandRouter
    from fastworkflow.train import artifact_versioning
    from fastworkflow.train import heldout_evaluation as he

    fastworkflow.init({})
    cases = he.load_benchmark_file(args.cases)
    contexts = sorted({c.context for c in cases})
    if args.version_dir:
        roots = [args.version_dir]
        folder = lambda root, ctx: os.path.join(root, artifact_versioning.context_folder_name(ctx))  # noqa: E731
        labels = [os.path.basename(os.path.normpath(args.version_dir))]
    else:
        name = os.path.basename(os.path.normpath(args.workflow))
        roots = [os.path.join(args.seeds_root, name, args.arm, str(s), name) for s in args.seeds]
        folder = lambda root, ctx: os.path.join(  # noqa: E731
            root, "___command_info", artifact_versioning.context_folder_name(ctx))
        labels = [str(s) for s in args.seeds]
    out = {"versions": labels, "missing_contexts": {}, "seeds": []}
    for label, root in zip(labels, roots):
        routers, missing = {}, []
        for ctx in contexts:
            path = folder(root, ctx)
            if os.path.isdir(path):
                routers[ctx] = CommandRouter(path)
            else:
                missing.append(ctx)
        out["missing_contexts"][label] = {
            "contexts": len(missing), "cases": sum(c.context in missing for c in cases)}
        rows = []
        for case in cases:
            if case.context not in routers:
                continue
            predicted, _details = routers[case.context].predict_with_details(case.utterance)
            row = record(case, [he.normalize_label(str(x)) for x in predicted])
            row["workflow"] = args.name or os.path.basename(os.path.normpath(args.workflow))
            rows.append(row)
        out["seeds"].append(rows)
        print(f"old {label}: {len(rows)} cases scored, {len(missing)} contexts missing "
              f"({out['missing_contexts'][label]['cases']} cases)")
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    return 0


# --------------------------------------------------------------------------- chain


class ReplayRouter:
    """The local head's recorded answer behind ``CommandRouter.predict_with_details``."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.context = None

    def predict_with_details(self, utterance: str):
        return list(self.answers[(self.context, utterance)]), {}


def _scope_paths(args, workflow: str) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    if Path(workflow).name == "library":
        from score_matcher_engine import LIBRARY_PATHS

        paths.update({k: list(v) for k, v in LIBRARY_PATHS.items()})
    if args.paths_file:
        paths.update({k: list(v) for k, v in json.load(open(args.paths_file, encoding="utf-8")).items()})
    return paths


GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def parameterised_matcher(idt):
    """``MatcherEngine`` with its two commit numbers apart. In the product the accept and decline thresholds (default 0.9 each, ``remote_accept_threshold`` and
    ``remote_decline_threshold``) are read in ``decide``: a scope's best command at or above it is a match, and "none" or a command of another
    scope at or above it declines. This subclass is ``decide`` with those two reads given their own values;
    ``MATCHER_UNSURE_THRESHOLD`` (0.2), the candidate cap and everything else are the product's. At the product defaults
    it is checked equal to the product's class on every case of a run (``chain`` asserts it)."""

    class Parameterised(idt.MatcherEngine):
        match_threshold = idt.DEFAULT_REMOTE_ACCEPT_THRESHOLD
        decline_threshold = idt.DEFAULT_REMOTE_DECLINE_THRESHOLD

        def decide(self, utterance):
            answered = self.distribution(utterance)
            all_scores = answered["scores"]
            try:
                own = [c.name for c in idt.matcher_candidates(self.workflow_folderpath, self.context_name)
                       if c.name not in self.exclude]
            except ValueError:
                own = []
            scores = {name: float(all_scores.get(name, 0.0)) for name in own}
            ranked = sorted(own, key=lambda name: -scores[name])
            none_score = float(all_scores.get(idt.MATCHER_NONE, 0.0))
            elsewhere = max((s for o, s in all_scores.items()
                             if o != idt.MATCHER_NONE and o not in scores and o not in self.exclude), default=0.0)
            best = scores[ranked[0]] if ranked else 0.0
            kept = [n for n in ranked if scores[n] > idt.MATCHER_UNSURE_THRESHOLD][:idt.MATCHER_MAX_CANDIDATES]
            if "error" in answered:
                kind, labels = idt.NOT_MINE, [idt.WILDCARD_LABEL]
            elif ranked and best >= self.match_threshold:
                kind, labels = idt.MATCH, [ranked[0]]
            elif none_score >= self.decline_threshold or elsewhere >= self.decline_threshold or not kept:
                kind, labels = idt.NOT_MINE, [idt.WILDCARD_LABEL]
            else:
                kind, labels = idt.AMBIGUOUS, kept
            details = {"confident": kind == idt.MATCH, "confidence": best, "none_score": none_score,
                       "latency_ms": answered.get("latency_ms", 0.0), "calls": answered.get("calls", 0)}
            if "error" in answered:
                details["error"] = answered["error"]
            return idt.ScopeAnswer(kind=kind, labels=labels, scores={}, confident=kind == idt.MATCH,
                                   engine=self.name, fallback=self.fallback, details=details)

    return Parameterised


def _chain(args) -> int:
    from dotenv import dotenv_values

    env_vars: dict = dict(dotenv_values(args.env))
    for path in args.passwords or ():
        env_vars.update(dotenv_values(path))
    if not env_vars.get("JEV_API_KEY"):
        print("JEV_API_KEY is not set in the env files given.", file=sys.stderr)
        return 1
    env_vars["FW_LM_CACHE"] = "0"
    import fastworkflow

    fastworkflow.init(env_vars=env_vars)
    from fastworkflow.intent import intent_detector as idt
    from score_matcher_engine import _copy_without_artifacts

    heads = json.load(open(args.heads, encoding="utf-8"))
    paths = _scope_paths(args, args.workflow)
    recording_path = Path(args.recording)
    recording: dict = json.loads(recording_path.read_text(encoding="utf-8")) if recording_path.exists() else {}
    state = {"calls": 0, "live": 0, "failed_in_a_row": 0, "latency": None, "dirty": 0, "last": None,
             "allow_live": True}
    original = idt._matcher_distribution
    candidates_memo: dict = {}
    real_candidates = idt.matcher_candidates

    def candidates(folder, context):
        if (folder, context) not in candidates_memo:
            candidates_memo[(folder, context)] = real_candidates(folder, context)
        return candidates_memo[(folder, context)]

    idt.matcher_candidates = candidates

    def recorded(folder, scope_path, utterance, exclude=frozenset()):
        key = json.dumps([args.name, list(scope_path), utterance, sorted(exclude)])
        state["calls"] += 1
        answer = None
        if key not in recording:
            if not state["allow_live"]:
                raise SystemExit(f"no recorded answer for {utterance!r}; a grid pass never calls the model")
            answer = original(folder, scope_path, utterance, exclude) if exclude else original(
                folder, scope_path, utterance)
            state["live"] += 1
            if "error" in answer:
                state["failed_in_a_row"] += 1
                if state["failed_in_a_row"] >= 5:
                    raise SystemExit(f"stopping: 5 remote calls in a row failed ({answer['error'][:120]})")
                state["latency"], state["last"] = answer.get("latency_ms", 0.0), answer
                return answer          # a failure is never recorded, so a rerun asks again
            state["failed_in_a_row"] = 0
            recording[key] = answer
            state["dirty"] += 1
            if state["dirty"] >= 10:
                recording_path.write_text(json.dumps(recording), encoding="utf-8")
                state["dirty"] = 0
        result = copy.deepcopy(recording.get(key) or answer)
        state["latency"], state["last"] = result.get("latency_ms", 0.0), result
        return result

    idt._matcher_distribution = recorded
    Parameterised = parameterised_matcher(idt)
    match_thr, decline_thr = (float(x) for x in args.thresholds.split(","))
    arms = args.arms or list(ARMS)

    def own_names(folder, context):
        return {c.name for c in candidates(folder, context)}

    def run(folder, rows_in, mthr, dthr, *, detail):
        """One pass over a heads file's rows. The product class at the product defaults, the subclass elsewhere."""
        router = ReplayRouter({(r["context"], r["utterance"]): r["predicted"] for r in rows_in})
        rows_out = []
        for number, r in enumerate(rows_in, 1):
            router.context = r["context"]
            path = paths.get(r["context"], [r["context"]])
            if args.star and "*" not in path:
                path = [*path, "*"]
            if (mthr, dthr) == (idt.DEFAULT_REMOTE_ACCEPT_THRESHOLD, idt.DEFAULT_REMOTE_DECLINE_THRESHOLD):
                matcher = idt.MatcherEngine(folder, r["context"], path)
            else:
                matcher = Parameterised(folder, r["context"], path)
                matcher.match_threshold, matcher.decline_threshold = mthr, dthr
            engine = idt.ChainEngine(idt.ClassifierEngine(router), matcher)
            before = state["calls"]
            answer = engine.decide(r["utterance"])
            asked = state["calls"] > before
            kind = answer.kind
            predicted = ([WILDCARD] if kind == idt.NOT_MINE else [PARAMETER_VALUE] if kind == idt.BARE_VALUE
                         else list(answer.labels))
            decided_by = "matcher" if (asked and answer.engine == idt.ENGINE_MATCHER) else (
                "head_after_matcher_error" if asked else "head")
            row = {**r, "head_predicted": r["predicted"], "predicted": predicted, "answer_kind": kind,
                   "asked": asked, "decided_by": decided_by, "matcher_error": "matcher_error" in answer.details,
                   "latency_ms": state["latency"] if asked else None}
            if detail and asked and state["last"] is not None and "error" not in state["last"]:
                scores = state["last"]["scores"]
                own = own_names(folder, r["context"])
                own_scores = {n: float(scores.get(n, 0.0)) for n in own}
                other = {o: float(s) for o, s in scores.items() if o != idt.MATCHER_NONE and o not in own_scores}
                row["m"] = {"best_own": max(own_scores.values(), default=0.0),
                            "none": float(scores.get(idt.MATCHER_NONE, 0.0)),
                            "elsewhere": max(other.values(), default=0.0)}
            rows_out.append(row)
            if detail and number % 25 == 0:
                print(f"  {number}/{len(rows_in)} (live calls so far {state['live']})", flush=True)
        return rows_out

    out: dict = {}
    with tempfile.TemporaryDirectory(prefix="chain_followup_") as scratch:
        folder = str(_copy_without_artifacts(Path(args.workflow).resolve(), Path(scratch)))
        for arm in arms:
            rows_in = heads[arm][args.name]
            rows_out = run(folder, rows_in, match_thr, decline_thr, detail=True)
            out[arm] = rows_out
            print(f"{arm}: asked {sum(x['asked'] for x in rows_out)} of {len(rows_out)} "
                  f"(thresholds match {match_thr}, decline {decline_thr})")
        state["allow_live"] = False
        if args.grid:
            rows_in = heads[arms[0]][args.name]
            # equality check: the subclass at the product defaults against the product's class
            Parameterised_default = Parameterised
            router = ReplayRouter({(r["context"], r["utterance"]): r["predicted"] for r in rows_in})
            mismatches = 0
            for r in rows_in:
                router.context = r["context"]
                path = paths.get(r["context"], [r["context"]])
                if args.star and "*" not in path:
                    path = [*path, "*"]
                a = idt.ChainEngine(idt.ClassifierEngine(router), idt.MatcherEngine(folder, r["context"], path)).decide(r["utterance"])
                m = Parameterised_default(folder, r["context"], path)
                b = idt.ChainEngine(idt.ClassifierEngine(router), m).decide(r["utterance"])
                mismatches += (a.kind, list(a.labels)) != (b.kind, list(b.labels))
            assert mismatches == 0, f"the parameterised matcher differs from the product's on {mismatches} cases"
            print("parameterised matcher equals the product's matcher at the product defaults on every case")
            grid = []
            for mthr in GRID:
                for dthr in GRID:
                    rows_g = run(folder, rows_in, mthr, dthr, detail=False)
                    grid.append({"match": mthr, "decline": dthr, "counts": counts(rows_g)})
            out["_grid"] = grid
    recording_path.write_text(json.dumps(recording), encoding="utf-8")
    out["_calls"] = {"live_calls_this_run": state["live"], "recorded_answers": len(recording)}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"live remote calls this run: {state['live']}; recording holds {len(recording)} answers")
    return 0


def counts(rows: list[dict]) -> dict:
    """Additive counts of one pass (pooled across workflows by summing)."""
    routing = [r for r in rows if r["kind"] == "routing"]
    esc = [r for r in rows if r["kind"] == "escalation"]
    wrong = [r for r in routing if len(r["predicted"]) == 1 and r["predicted"] != [r["expected"]]]
    return {
        "cases": len(rows), "routing": len(routing), "escalation": len(esc),
        "top1": sum(r["predicted"] == [r["expected"]] for r in routing),
        "wrong_head": sum(r["decided_by"] != "matcher" for r in wrong),
        "wrong_remote": sum(r["decided_by"] == "matcher" for r in wrong),
        "false_esc": sum(r["predicted"] == [WILDCARD] for r in routing),
        "escalated": sum(r["predicted"] == [WILDCARD] for r in esc),
        "questions": sum(r["answer_kind"] == "ambiguous" for r in rows),
        "asked": sum(r["asked"] for r in rows),
    }


def pooled_grid(files: list[str], g: str) -> dict:
    """``{(match, decline): summed counts}`` over the chain files whose workflows are in group *g*."""
    out: dict = {}
    for path in files:
        d = json.load(open(path, encoding="utf-8"))
        arm_rows = next(v for k, v in d.items() if k in ARMS)
        if group_of(arm_rows[0]["workflow"]) != g:
            continue
        for cell in d["_grid"]:
            acc = out.setdefault((cell["match"], cell["decline"]), {})
            for k, v in cell["counts"].items():
                acc[k] = acc.get(k, 0) + v
    return out


def rates(c: dict) -> dict:
    n = max(c["routing"], 1)
    return {"top1": c["top1"] / n, "wrong": (c["wrong_head"] + c["wrong_remote"]) / n,
            "wrong_head": c["wrong_head"], "wrong_remote": c["wrong_remote"], "false_esc": c["false_esc"] / n,
            "esc": f"{c['escalated']}/{c['escalation']}", "questions": c["questions"] / max(c["cases"], 1)}


RULE = ("lowest mean over the two groups of wrong lone answers (share of routing cases) among the pairs whose "
        "questions to the user are at most 15% of all requests in BOTH groups; ties to the higher mean strict top-1")


def choose_pair(grids: dict[str, dict]) -> tuple:
    best = None
    for pair in grids["examples"]:
        r = [rates(grids[g][pair]) for g in ("examples", "large")]
        if max(x["questions"] for x in r) > 0.15:
            continue
        key = (sum(x["wrong"] for x in r) / 2, -sum(x["top1"] for x in r) / 2)
        if best is None or key < best[0]:
            best = (key, pair)
    return best[1]


def _grid_report(args) -> int:
    grids = {g: pooled_grid(args.chain, g) for g in ("examples", "large")}
    print("Development sweep, chain with head BASE (wrong lone = head+remote; questions are a share of all requests)")
    for g in grids:
        print(f"\n{g}: match threshold (rows) x decline threshold (columns)")
        for label, f in (("strict top-1", lambda r: f"{r['top1']:.1%}"),
                         ("wrong lone (head+remote)", lambda r: f"{r['wrong']:.1%} ({r['wrong_head']}+{r['wrong_remote']})"),
                         ("false esc / esc recall", lambda r: f"{r['false_esc']:.1%} / {r['esc']}"),
                         ("questions", lambda r: f"{r['questions']:.1%}")):
            print(f"\n{label}\n| match \\ decline | " + " | ".join(str(d) for d in GRID) + " |\n|---|" + "---|" * len(GRID))
            for m in GRID:
                print(f"| {m} | " + " | ".join(f(rates(grids[g][(m, d)])) for d in GRID) + " |")
    pair = choose_pair(grids)
    print(f"\nRule: {RULE}\nChosen (match, decline) = {pair}")
    return 0


def _errors_report(args) -> int:
    """Types of the remote model's wrong answers and its confidence (rows written with ``m`` at 0.5/0.5)."""
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]

    def bucket(x):
        for i, (lo, hi) in enumerate(bins):
            if lo <= x < hi:
                return i
        return 0 if x < 0.5 else 4

    for g in ("examples", "large"):
        rows = []
        for path in args.chain:
            d = json.load(open(path, encoding="utf-8"))
            rows += [r for r in d["BASE"] if group_of(r["workflow"]) == g]
        asked = [r for r in rows if r["decided_by"] == "matcher" and "m" in r]
        types: dict[str, int] = {}
        wrong_hist, right_hist = [0] * 5, [0] * 5
        for r in asked:
            m, kind, routing = r["m"], r["answer_kind"], r["kind"] == "routing"
            conf = m["best_own"] if kind == "match" else max(m["none"], m["elsewhere"])
            if kind == "ambiguous":
                if routing and len(r["predicted"]) == 1 and r["predicted"] != [r["expected"]]:
                    # one candidate left: the metric counts it as a lone answer, the chain asks the user
                    types["a question with one wrong candidate (a lone answer by the metric)"] = types.get(
                        "a question with one wrong candidate (a lone answer by the metric)", 0) + 1
                continue
            if routing:
                if kind == "match":
                    right = r["predicted"] == [r["expected"]]
                    t = None if right else "wrong command of the same scope"
                else:
                    right = False
                    t = ("a command of another scope on the path (wrong pass-up)" if m["elsewhere"] >= 0.5 and m["elsewhere"] >= m["none"]
                         else '"none" when a command was right' if m["none"] >= 0.5
                         else "declined with no command above 0.2 (wrong pass-up)")
            else:
                right = kind == "not_mine"
                t = None if right else "a command when it should have passed up"
            if right:
                right_hist[bucket(conf)] += 1
            else:
                types[t] = types.get(t, 0) + 1
                wrong_hist[bucket(conf)] += 1
        print(f"\n{g}: remote model decided {len(asked)} cases; lone answers right {sum(right_hist)}, wrong {sum(wrong_hist)}")
        for t, n in sorted(types.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d}  {t}")
        print("  confidence 0.5-0.6 / 0.6-0.7 / 0.7-0.8 / 0.8-0.9 / 0.9-1.0")
        print("  wrong answers:", wrong_hist)
        print("  right answers:", right_hist)
    return 0


def _tuned_report(args) -> int:
    for g in ("examples", "large"):
        for label, files in (("current 0.5/0.5", args.chain), (f"tuned {args.label}", args.tuned)):
            rows = []
            for path in files:
                d = json.load(open(path, encoding="utf-8"))
                rows += [r for r in d["BASE"] if group_of(r["workflow"]) == g]
            c = counts(rows)
            r = rates(c)
            print(f"| {g} | {label} | {c['cases']} | {r['top1']:.1%} | {r['wrong']:.1%} ({c['wrong_head']}+{c['wrong_remote']}) | "
                  f"{r['false_esc']:.1%} | {r['esc']} | {r['questions']:.1%} ({c['questions']}) | {c['asked']} |")
    return 0


# --------------------------------------------------------------------------- report


def fmt(s: dict) -> str:
    prec = "-" if s["precision"] is None else f"{s['precision']:.1%}"
    return (f"{s['top1']:.1%} | {s['in_list']:.1%} | {s['one_label']:.1%} | {s['wrong_lone']:.1%} | "
            f"{s['false_esc']:.1%} | {s['esc_recall'][0]}/{s['esc_recall'][1]} | {prec}")


HEADER = ("| Row | Cases (routing + escalation) | Strict top-1 | In-list | One label | Wrong lone | False esc. | "
          "Esc. recall | Lone precision |\n|---|---|---|---|---|---|---|---|---|")


def vs(rows: list[dict]) -> list[dict]:
    return [verdict(r["kind"], r["expected"] if r["kind"] == "routing" else None, r["predicted"]) for r in rows]


def majority(per_seed: list[list[dict]]) -> list[dict]:
    keys = [(r["context"], r["kind"], r["utterance"]) for r in per_seed[0]]
    for seed in per_seed:
        assert keys == [(r["context"], r["kind"], r["utterance"]) for r in seed]
    out = []
    for versions in zip(*per_seed):
        v = [verdict(r["kind"], r["expected"] if r["kind"] == "routing" else None, r["predicted"])
             for r in versions]
        merged = dict(v[0])
        for name in v[0]:
            if name != "kind":
                merged[name] = sum(bool(x[name]) for x in v) * 2 > len(v)
        out.append(merged)
    return out


def _pool(per_workflow: dict[str, list]) -> dict[str, list]:
    pooled: dict[str, list] = {}
    for name, value in per_workflow.items():
        pooled.setdefault(group_of(name), []).append(value)
    return pooled


def _report(args) -> int:
    heads = json.load(open(args.heads, encoding="utf-8"))
    chains = [json.load(open(p, encoding="utf-8")) for p in args.chain]
    olds = [json.load(open(p, encoding="utf-8")) for p in args.old]
    for g in ("examples", "large"):
        group_heads = {arm: [r for wf, rows in heads[arm].items() if group_of(wf) == g for r in rows] for arm in ARMS}
        if not group_heads["BASE"]:
            continue
        lines = [HEADER]
        old_by_wf = {}
        for old in olds:
            wf = old["seeds"][0][0]["workflow"] if old["seeds"][0] else None
            if wf and group_of(wf) == g:
                old_by_wf[wf] = old
        if old_by_wf:
            per_seed = [[r for old in old_by_wf.values() for r in old["seeds"][i]]
                        for i in range(len(next(iter(old_by_wf.values()))["seeds"]))]
            labels = next(iter(old_by_wf.values()))["versions"]
            if len(per_seed) > 1:
                merged = majority(per_seed)
                lines.append(f"| old pair, majority of seeds {'/'.join(labels)} | {len(merged)} | {fmt(summarise(merged))} |")
                for lab, rows in zip(labels, per_seed):
                    lines.append(f"| old pair, seed {lab} | {len(rows)} | {fmt(summarise(vs(rows)))} |")
            else:
                lines.append(f"| old pair, previous version | {len(per_seed[0])} | {fmt(summarise(vs(per_seed[0])))} |")
        for arm in ARMS:
            rows = group_heads[arm]
            lines.append(f"| head {arm} | {len(rows)} | {fmt(summarise(vs(rows)))} |")
        extra = []
        for arm in ARMS:
            members = {w for w in heads[arm] if group_of(w) == g}
            rows = [r for c in chains for r in c.get(arm, []) if r["workflow"] in members]
            if not rows:
                continue
            lines.append(f"| chain with {arm} | {len(rows)} | {fmt(summarise(vs(rows)))} |")
            routing = [r for r in rows if r["kind"] == "routing"]
            asked = [r for r in rows if r["asked"]]
            wrong = [r for r in routing if len(r["predicted"]) == 1 and r["predicted"] != [r["expected"]]]
            by_head = sum(r["decided_by"] != "matcher" for r in wrong)
            lat = [r["latency_ms"] for r in asked if r["latency_ms"] is not None]
            question = [r for r in rows if r["answer_kind"] == "ambiguous"]
            extra.append(
                f"chain {arm}: remote model asked on {len(asked)} of {len(rows)} cases ({len(asked) / len(rows):.1%}); "
                f"final wrong lone answers {len(wrong)}: head confidently wrong {by_head}, remote model wrong "
                f"{len(wrong) - by_head}; mean added latency {sum(lat) / max(len(lat), 1) / 1000:.2f} s per asked case; "
                f"ends at a question {len(question)} of {len(rows)} ({len(question) / len(rows):.1%}; "
                f"routing {sum(r['answer_kind'] == 'ambiguous' for r in routing)} of {len(routing)}); "
                f"remote call failures fell back to the head on {sum(r['matcher_error'] for r in rows)}")
        print(f"\n### {g}\n" + "\n".join(lines) + "\n\n" + "\n".join(f"* {e}" for e in extra))
    return 0


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("heads")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cache", default=None)
    s.add_argument("--set", default="conf", help="cases_<set>.json of each data directory (conf or dev)")
    s = sub.add_parser("old")
    s.add_argument("--workflow", required=True)
    s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name", default=None, help="the data directory's name (a name starting with large is the large group)")
    s.add_argument("--seeds-root")
    s.add_argument("--arm", default="H3")
    s.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    s.add_argument("--version-dir")
    s = sub.add_parser("chain")
    s.add_argument("--workflow", required=True)
    s.add_argument("--name", required=True, help="the data directory's name in the heads file")
    s.add_argument("--heads", required=True)
    s.add_argument("--env", required=True)
    s.add_argument("--passwords", nargs="*")
    s.add_argument("--recording", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--paths-file")
    s.add_argument("--star", action="store_true")
    s.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    s.add_argument("--thresholds", default="0.5,0.5", help="MATCH,DECLINE commit thresholds of the remote link")
    s.add_argument("--grid", action="store_true", help="also score the 5x5 grid of both thresholds from the recording")
    s = sub.add_parser("report")
    s.add_argument("--heads", required=True)
    s.add_argument("--chain", nargs="+", required=True)
    s.add_argument("--old", nargs="+", required=True)
    s = sub.add_parser("grid")
    s.add_argument("--chain", nargs="+", required=True, help="chain files written with --grid")
    s = sub.add_parser("errors")
    s.add_argument("--chain", nargs="+", required=True)
    s = sub.add_parser("tuned")
    s.add_argument("--chain", nargs="+", required=True, help="chain files at 0.5/0.5")
    s.add_argument("--tuned", nargs="+", required=True, help="chain files at the chosen thresholds")
    s.add_argument("--label", default="")
    args = p.parse_args(argv)
    return {"heads": _heads, "old": _old, "chain": _chain, "report": _report, "grid": _grid_report,
            "errors": _errors_report, "tuned": _tuned_report}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
