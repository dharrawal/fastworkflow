#!/usr/bin/env python3
"""Run the pre-registered "one intent model for the workflow" experiments.

An arm is named on the command line as ``NAME=KIND[:FOLDER]``:

* ``shipped``       the per-context heads in the workflow's own ``___command_info``.
* ``heads:FOLDER``  per-context heads rebuilt by ``heads`` (one set per seed).
* ``single:FOLDER`` one model pair for the workflow, read through the scope's mask.
* ``large:FOLDER``  the same artifacts with every decision answered by the large model.
* ``large-holdout:FOLDER``  as ``large``, with the threshold and margin that
  ``calibrate-holdout`` computed on the persona holdout.

Usage
-----
    single_intent_model.py heads  --workflow WF --seed 42 --out DIR
    single_intent_model.py single --workflow WF --seed 42 --arm M2 --out DIR
    single_intent_model.py calibrate-holdout --workflow WF --arm M2 --out DIR
    single_intent_model.py run    --workflow WF --benchmark BENCH.json --out DIR \
        [--no-train] [--tag NAME] [--arms H3=heads:H3 S=large:M2 ...]
    single_intent_model.py pool   --results A/results.json B/results.json --out pooled.json \
        --baseline H3 --conditions v2

``run`` trains the first round's M2 and C2 for every seed unless ``--no-train`` is given,
then scores the arms on the benchmark and writes ``results[_TAG].json`` and
``verdicts[_TAG].json``. Artifacts already in the output directory are never trained again,
so the same artifacts can be scored on another benchmark file. ``pool`` adds workflows up
and evaluates the pass conditions of the first (``v1``) or second (``v2``) pre-registration.

Nothing is written into the workflow folder: rows are rebuilt and heads are trained on a
copy of it (loading a workflow refreshes two files of its ``___command_info``), and the
shipped heads are only read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_OUT_ROOT = os.path.join(tempfile.gettempdir(), "single_intent_model")
SEEDS = (42, 43, 44)
#: The arms of the first pre-registration, scored when ``run`` is given no ``--arms``.
FIRST_ROUND_ARMS = ("H=shipped", "M2=single:M2", "M1=large:M2", "C2=single:C2")
ARM_KINDS = ("shipped", "heads", "single", "large", "large-holdout")
VERDICT_FIELDS = ("top1", "in_list", "one_label", "lone_wrong", "false_escalation", "escalated")
#: Not copied when the workflow is duplicated: model weights, chat history, bytecode.
COPY_IGNORED = ("versions", "___convo_info", "___workflow_contexts", "__pycache__")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """The 95% Wilson score interval for a proportion."""
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b/c discordant pairs (as in score_benchmark.py)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def rate(successes: int, total: int) -> dict:
    return {
        "n": total,
        "k": successes,
        "rate": successes / total if total else None,
        "wilson95": wilson(successes, total),
    }


def summarise(verdicts: list[dict]) -> dict:
    """Every benchmark rate of the pre-registration from one arm's per-case verdicts."""
    routing = [v for v in verdicts if v["kind"] == "routing"]
    escalation = [v for v in verdicts if v["kind"] == "escalation"]
    per_scope: dict[str, list[dict]] = defaultdict(list)
    for verdict in routing:
        per_scope[verdict["scope"]].append(verdict)
    return {
        "routing_top1": rate(sum(v["top1"] for v in routing), len(routing)),
        "routing_in_list": rate(sum(v["in_list"] for v in routing), len(routing)),
        "routing_one_label_share": rate(sum(v["one_label"] for v in routing), len(routing)),
        "false_escalation_on_routing": rate(
            sum(v["false_escalation"] for v in routing), len(routing)),
        "lone_wrong_of_routing": rate(sum(v["lone_wrong"] for v in routing), len(routing)),
        "lone_wrong_of_lone_labels": rate(
            sum(v["lone_wrong"] for v in routing), sum(v["one_label"] for v in routing)),
        "escalation_recall": rate(sum(v["escalated"] for v in escalation), len(escalation)),
        "per_scope_top1": {
            scope: rate(sum(v["top1"] for v in cases), len(cases))
            for scope, cases in sorted(per_scope.items())
        },
        "per_scope_in_list": {
            scope: rate(sum(v["in_list"] for v in cases), len(cases))
            for scope, cases in sorted(per_scope.items())
        },
    }


def majority(per_seed: list[list[dict]]) -> list[dict]:
    """One verdict per case from several seeds: each boolean is what most seeds said."""
    merged = []
    for versions in zip(*per_seed):
        verdict = {k: versions[0][k] for k in ("key", "scope", "kind", "utterance", "expected")}
        for name in VERDICT_FIELDS:
            verdict[name] = sum(bool(v[name]) for v in versions) * 2 > len(versions)
        merged.append(verdict)
    return merged


def seed_agreement(per_seed: list[list[dict]]) -> dict:
    """How often the seeds give the same verdict for a case.

    The verdict of a routing case is its top-1 result and of an escalation case whether it
    escalated; the two looser readings are reported beside it.
    """
    total = len(per_seed[0])
    same_verdict = same_in_list = same_labels = 0
    for versions in zip(*per_seed):
        primary = "top1" if versions[0]["kind"] == "routing" else "escalated"
        same_verdict += len({bool(v[primary]) for v in versions}) == 1
        same_in_list += len({(bool(v[primary]), bool(v["in_list"])) for v in versions}) == 1
        same_labels += len({tuple(v["predicted"]) for v in versions}) == 1
    return {
        "cases": total,
        "same_verdict": rate(same_verdict, total),
        "same_verdict_and_in_list": rate(same_in_list, total),
        "same_predicted_labels": rate(same_labels, total),
    }


def paired_against(baseline: list[dict], arm: list[dict]) -> dict:
    """Exact McNemar of an arm against the baseline, per verdict field, on the same cases."""
    by_key = {v["key"]: v for v in arm}
    result = {}
    for name, kind in (("top1", "routing"), ("in_list", "routing"), ("escalated", "escalation")):
        baseline_only = arm_only = 0
        for verdict in baseline:
            if verdict["kind"] != kind:
                continue
            other = by_key[verdict["key"]]
            baseline_only += bool(verdict[name]) and not other[name]
            arm_only += bool(other[name]) and not verdict[name]
        result[name] = {
            "baseline_right_arm_wrong": baseline_only,
            "arm_right_baseline_wrong": arm_only,
            "exact_mcnemar_p": exact_mcnemar_p(baseline_only, arm_only),
        }
    return result


def latency_summary(milliseconds: list[float]) -> dict:
    ordered = sorted(milliseconds)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
    }


# ---------------------------------------------------------------------------
# Arms and models
# ---------------------------------------------------------------------------


def parse_arm(spec: str) -> tuple[str, str, str | None]:
    """``NAME=KIND[:FOLDER]`` as ``(name, kind, folder)``."""
    name, _, rest = spec.partition("=")
    kind, _, folder = rest.partition(":")
    if not name or kind not in ARM_KINDS or (kind != "shipped" and not folder):
        raise SystemExit(f"arm {spec!r} is not NAME=KIND[:FOLDER] with KIND in {ARM_KINDS}")
    return name, kind, folder or None


def arm_root(kind: str, folder: str | None, seed: int | None, *, workflow: str, out: str) -> str:
    """Where one arm's artifacts for one seed live."""
    name = os.path.basename(os.path.normpath(workflow))
    if kind == "shipped":
        return workflow
    if kind == "heads":
        return os.path.join(out, name, folder, str(seed), name)
    return os.path.join(out, name, folder, str(seed))


def heads_folder(workflow: str, scope: str) -> str:
    from fastworkflow.train import artifact_versioning

    return os.path.join(workflow, "___command_info", artifact_versioning.context_folder_name(scope))


def build_router(kind: str, root: str, scope: str):
    if kind in ("shipped", "heads"):
        from fastworkflow.train.model_pipeline_training import CommandRouter

        return CommandRouter(heads_folder(root, scope))
    from fastworkflow.train.workflow_model import WorkflowModelRouter

    return WorkflowModelRouter(
        root, scope, large_only=kind.startswith("large"),
        calibration="holdout" if kind == "large-holdout" else "split")


def directory_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(os.path.realpath(path)):
        total += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return total


def arm_bytes(kind: str, root: str, scopes: list[str]) -> int:
    """Bytes a version of this arm must keep on disk."""
    from fastworkflow.train import workflow_model as wm

    if kind in ("shipped", "heads"):
        return sum(directory_bytes(heads_folder(root, scope)) for scope in scopes)
    total = directory_bytes(root)
    if kind.startswith("large"):
        # A large-only reader never opens the tiny model, so a version need not ship it.
        total -= directory_bytes(os.path.join(root, wm.TINY_MODEL_DIRNAME))
    return total


def resident_bytes() -> int:
    """Resident memory of this process now (peak, where the current value is unreadable)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# ---------------------------------------------------------------------------
# Steps of a run
# ---------------------------------------------------------------------------


def copy_workflow(workflow: str, destination_root: str) -> str:
    """Duplicate the workflow's code and utterance caches, without its model weights.

    The copy has no published version: the links and the pointer into ``versions`` are
    dropped with it, so a trainer run on the copy writes a fresh, unversioned layout.
    """
    destination = os.path.join(destination_root, os.path.basename(os.path.normpath(workflow)))
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    shutil.copytree(workflow, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*COPY_IGNORED))
    info = os.path.join(destination, "___command_info")
    for name in os.listdir(info) if os.path.isdir(info) else []:
        path = os.path.join(info, name)
        if os.path.islink(path) or name in ("current.json", ".publication.lock"):
            os.remove(path)
    return destination


def environment_for(workflow: str, env_file: str | None, passwords_file: str | None) -> dict:
    """The env values to rebuild rows under: the workflow's files, and the cache's model."""
    from dotenv import dotenv_values

    from fastworkflow.train.workflow_model import cached_generation_model

    examples = Path(__file__).resolve().parents[2] / "fastworkflow" / "examples"
    own = Path(workflow) / "fastworkflow.env"
    env_file = env_file or str(own if own.is_file() else examples / "fastworkflow.env")
    own_passwords = Path(workflow) / "fastworkflow.passwords.env"
    passwords_file = passwords_file or str(
        own_passwords if own_passwords.is_file() else examples / "fastworkflow.passwords.env")
    values = {**dotenv_values(env_file), **dotenv_values(passwords_file)}
    # The cache key holds the generating model. An env file edited since the heads were
    # trained would make every lookup a miss, so the cache's own model string is used.
    if cached := cached_generation_model(workflow):
        values["LLM_SYNDATA_GEN"] = cached
    return values


def predict_cases(router_for, cases) -> list[dict]:
    """One verdict per benchmark case, scored by the library scorers on cached predictions."""
    from fastworkflow.train import heldout_evaluation as he

    escalation_labels = {he.normalize_label(label) for label in he.DEFAULT_ESCALATION_LABELS}
    verdicts = []
    for case in cases:
        labels, details = router_for(case.context).predict_with_details(case.utterance)
        predicted = [he.normalize_label(str(label)) for label in labels]
        lone_escalation = len(predicted) == 1 and predicted[0] in escalation_labels
        expected = he.normalize_label(case.expected_label or "") if case.kind == "routing" else None
        verdicts.append({
            "key": f"{case.context}|{case.kind}|{he.normalize_utterance(case.utterance)}",
            "scope": case.context,
            "kind": case.kind,
            "utterance": case.utterance,
            "expected": expected if case.kind == "routing" else case.expected_ancestor_command,
            "predicted": predicted,
            "top1": case.kind == "routing" and predicted == [expected],
            "in_list": case.kind == "routing" and expected in predicted,
            "one_label": len(predicted) == 1,
            # A lone label that is not the expected one: the error nothing downstream can catch.
            "lone_wrong": case.kind == "routing" and len(predicted) == 1 and predicted != [expected],
            "false_escalation": case.kind == "routing" and lone_escalation,
            "escalated": case.kind == "escalation" and lone_escalation,
            "details": {
                name: details.get(name)
                for name in ("model_tier", "confidence", "decision", "p_in", "p_out")
                if name in details
            },
        })

    # The counts above must be the library scorers' counts: they define the metrics.
    by_key = {(v["scope"], v["utterance"]): v["predicted"] for v in verdicts}
    for scope in sorted({case.context for case in cases}):
        def recorded(utterance, _scope=scope):
            return by_key[(_scope, utterance)]
        routing = he.benchmark_cases_for_context(cases, scope, "routing")
        escalation = he.benchmark_cases_for_context(cases, scope, "escalation")
        mine = [v for v in verdicts if v["scope"] == scope]
        score = he.score_routing(routing, recorded)
        assert score.top1_correct == sum(v["top1"] for v in mine), scope
        assert score.in_list_correct == sum(v["in_list"] for v in mine), scope
        assert he.score_escalation(escalation, recorded).correct == sum(
            v["escalated"] for v in mine), scope
    return verdicts


def measure_latency(routers: dict, cases, repetitions: int) -> dict:
    """Milliseconds per decision for each arm, the arms taking turns on every case.

    Taking turns spreads whatever else the machine is doing evenly over the arms, which a
    pass per arm would not. The first pass over the cases is a warm-up and is not timed.
    """
    for arm_routers in routers.values():
        for case in cases:
            arm_routers[case.context].predict_with_details(case.utterance)
    timings: dict[str, list[float]] = {arm: [] for arm in routers}
    for _ in range(repetitions):
        for case in cases:
            for arm, arm_routers in routers.items():
                started = time.perf_counter()
                arm_routers[case.context].predict_with_details(case.utterance)
                timings[arm].append((time.perf_counter() - started) * 1000)
    return timings


def measure_memory(kind: str, root: str, scopes: list[str]) -> dict:
    """Resident memory of a fresh process that loads every scope of one arm."""
    command = [sys.executable, os.path.abspath(__file__), "memory", "--kind", kind,
               "--root", root, "--scopes", *scopes]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _memory(args: argparse.Namespace) -> int:
    import fastworkflow

    fastworkflow.init({})
    import torch  # imported before the first reading so it is not counted as model memory

    before = resident_bytes()
    for scope in args.scopes:
        router = build_router(args.kind, args.root, scope)
        router.predict_with_details("warm up")
        # Weights are memory-mapped and only become resident when read. One decision may be
        # answered by the tiny model alone, so every model the arm keeps loaded is run once.
        pipeline = router.modelpipeline
        pairs = [(pipeline.distil_model, pipeline.distil_tokenizer)]
        if not args.kind.startswith("large"):
            pairs.append((pipeline.tiny_model, pipeline.tiny_tokenizer))
        for model, tokenizer in pairs:
            with torch.no_grad():
                model(**tokenizer("warm up", return_tensors="pt").to(pipeline.device))
    after = resident_bytes()
    print(json.dumps({"before_bytes": before, "after_bytes": after,
                      "loaded_bytes": after - before}))
    return 0


def _heads(args: argparse.Namespace) -> int:
    """Rebuild the per-context heads at one seed, on a copy of the workflow."""
    from fastworkflow.train import workflow_model as wm

    workflow = os.path.abspath(args.workflow)
    name = os.path.basename(os.path.normpath(workflow))
    destination = os.path.join(args.out, name, args.arm, str(args.seed))
    done = os.path.join(destination, "heads.json")
    if os.path.isfile(done) and not args.retrain:
        print(f"{args.arm} seed {args.seed}: heads present, not retrained")
        return 0
    copy = copy_workflow(workflow, destination)
    started = time.monotonic()
    folders = wm.rebuild_heads(
        copy,
        seed=args.seed,
        env_vars=environment_for(workflow, args.env, args.passwords),
        not_here_samples=wm.recorded_not_here_samples(workflow),
        built_in_vague_class=not args.no_built_in_vague_class,
    )
    with open(done, "w", encoding="utf-8") as f:
        json.dump({"workflow_copy": copy, "contexts": folders, "seed": args.seed,
                   "built_in_vague_class": not args.no_built_in_vague_class,
                   "train_seconds": time.monotonic() - started}, f, indent=2)
    print(f"{args.arm} seed {args.seed}: heads in {copy} ({time.monotonic() - started:.0f} s)")
    return 0


def _single(args: argparse.Namespace) -> int:
    """Train the workflow's one model pair at one seed, under a named recipe folder."""
    from fastworkflow.train import workflow_model as wm

    workflow = os.path.abspath(args.workflow)
    name = os.path.basename(os.path.normpath(workflow))
    model_dir = os.path.join(args.out, name, args.arm, str(args.seed))
    if os.path.isfile(os.path.join(model_dir, wm.SCOPES_FILENAME)) and not args.retrain:
        print(f"{args.arm} seed {args.seed}: artifacts present, not retrained")
        return 0
    copy = copy_workflow(workflow, os.path.join(args.out, name, "_workflow_copy"))
    rows = wm.workflow_rows(copy, env_vars=environment_for(workflow, args.env, args.passwords),
                            wildcard_weight=args.wildcard_weight)
    payload = wm.train_workflow_model(rows, model_dir, seed=args.seed)
    print(f"{args.arm} seed {args.seed}: trained in {payload['train_seconds']:.0f} s into {model_dir}")
    return 0


def _calibrate_holdout(args: argparse.Namespace) -> int:
    """Recompute existing single-model artifacts' decision numbers on the persona holdout."""
    from fastworkflow.train import workflow_model as wm

    workflow = os.path.abspath(args.workflow)
    name = os.path.basename(os.path.normpath(workflow))
    copy = copy_workflow(workflow, os.path.join(args.out, name, "_workflow_copy"))
    rows = wm.workflow_rows(copy, env_vars=environment_for(workflow, args.env, args.passwords),
                            wildcard_weight=args.wildcard_weight)
    for seed in args.seeds:
        model_dir = os.path.join(args.out, name, args.arm, str(seed))
        payload = wm.calibrate_from_holdout(rows, model_dir)
        print(f"{name} {args.arm} seed {seed}: {payload['unseen_held_out_rows']} unseen held-out "
              f"rows, margin {payload['escalation_margin_large_only']:.4f}, workflow threshold "
              f"{payload['workflow_ambiguous_threshold']:.4f}")
        for scope, entry in payload["scopes"].items():
            print(f"  {scope:8} threshold {entry['large_ambiguous_threshold']:.4f} "
                  f"({entry['source'][:14]}) held out {entry['held_out_rows']}, unseen "
                  f"{entry['unseen_rows']}, top label right {entry['top_label_right']}")
    return 0


def train_first_round(args: argparse.Namespace, workflow: str, out: str) -> None:
    """Train the first round's M2 and C2 for every seed that is not already there."""
    from fastworkflow.train import workflow_model as wm

    name = os.path.basename(os.path.normpath(workflow))
    copy = copy_workflow(workflow, os.path.join(out, name, "_workflow_copy"))
    rows = wm.workflow_rows(copy, env_vars=environment_for(workflow, args.env, args.passwords))
    print(f"rows: {len(rows.rows)} over {len(rows.labels)} labels, scopes {sorted(rows.scopes)}")
    for finding in rows.findings:
        print(f"  row check: {finding}")
    for arm in ("M2", "C2"):
        for seed in args.seeds:
            model_dir = os.path.join(out, name, arm, str(seed))
            if os.path.isfile(os.path.join(model_dir, wm.SCOPES_FILENAME)) and not args.retrain:
                print(f"{arm} seed {seed}: artifacts present, not retrained")
                continue
            print(f"{arm} seed {seed}: training into {model_dir}")
            payload = wm.train_workflow_model(rows, model_dir, seed=seed, paired_context=(arm == "C2"))
            print(f"  trained in {payload['train_seconds']:.0f} s")


def training_facts(kind: str, root: str, scopes: list[str]) -> dict:
    """What an arm's artifacts record about their own training and calibration."""
    from fastworkflow.train import workflow_model as wm

    facts: dict = {}
    if kind in ("single", "large", "large-holdout"):
        with open(os.path.join(root, wm.SCOPES_FILENAME), "r", encoding="utf-8") as f:
            recorded = json.load(f)
        f1 = "in_distribution_f1_large_only" if kind != "single" else "in_distribution_f1"
        margin = "escalation_margin_large_only" if kind != "single" else "escalation_margin"
        if kind == "large-holdout":
            with open(os.path.join(root, wm.HOLDOUT_SCOPES_FILENAME), "r", encoding="utf-8") as f:
                holdout = json.load(f)
            return {
                "train_seconds": recorded.get("train_seconds"),
                "in_distribution_f1": {s: recorded["scopes"][s][f1] for s in scopes},
                "escalation_margin": holdout[margin],
                "thresholds": {s: {
                    "large_ambiguous_threshold": holdout["scopes"][s]["large_ambiguous_threshold"],
                    "source": holdout["scopes"][s]["source"],
                    "unseen_rows": holdout["scopes"][s]["unseen_rows"]} for s in scopes},
                "unseen_held_out_rows": holdout["unseen_held_out_rows"],
            }
        facts["train_seconds"] = recorded.get("train_seconds")
        facts["in_distribution_f1"] = {s: recorded["scopes"][s][f1] for s in scopes}
        facts["escalation_margin"] = recorded[margin]
        facts["thresholds"] = {
            s: {k: recorded["scopes"][s][k] for k in (
                "tier_threshold", "tiny_ambiguous_threshold", "large_ambiguous_threshold")}
            for s in scopes
        }
        return facts
    info = os.path.join(root, "___command_info")
    try:
        with open(os.path.join(info, "heldout_evaluation.json"), "r") as f:
            reports = {r["context"]: r for r in json.load(f).get("contexts", [])}
        facts["in_distribution_f1"] = {s: reports.get(s, {}).get("in_distribution_f1") for s in scopes}
    except (OSError, ValueError):
        pass
    for candidate in (os.path.join(os.path.dirname(root), "heads.json"),
                      os.path.join(os.path.realpath(os.path.join(info, "current")), "manifest.json")):
        if os.path.isfile(candidate):
            with open(candidate, "r") as f:
                recorded = json.load(f)
            facts["train_seconds"] = recorded.get("train_seconds", recorded.get("train_duration_seconds"))
            break
    return facts


def summarise_arm(per_seed: list[list[dict]]) -> tuple[dict, list[dict]]:
    """One arm's rates per seed and by majority verdict, and the majority verdicts."""
    merged = majority(per_seed)
    return {
        "majority": summarise(merged),
        "seeds": [summarise(verdicts) for verdicts in per_seed],
        "seed_agreement": seed_agreement(per_seed) if len(per_seed) > 1 else None,
    }, merged


def compare_with_baseline(arm_seeds: list[list[dict]], arm_majority: list[dict],
                          base_seeds: list[list[dict]], base_majority: list[dict]) -> dict:
    """Paired tests of an arm against the baseline: by majority verdict and seed by seed.

    Seed n of the arm meets seed n of the baseline when both have that many seeds; a
    baseline with one set of artifacts meets every seed of the arm.
    """
    return {
        "majority": paired_against(base_majority, arm_majority),
        "seeds": [
            paired_against(base_seeds[index] if len(base_seeds) == len(arm_seeds) else base_majority,
                           verdicts)
            for index, verdicts in enumerate(arm_seeds)
        ],
    }


def _run(args: argparse.Namespace) -> int:
    from fastworkflow.train import heldout_evaluation as he

    workflow = os.path.abspath(args.workflow)
    name = os.path.basename(os.path.normpath(workflow))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, name), exist_ok=True)
    arms = [parse_arm(spec) for spec in (args.arms or FIRST_ROUND_ARMS)]
    baseline = args.baseline or arms[0][0]
    if not args.no_train:
        train_first_round(args, workflow, out)
        if args.train_only:
            return 0
    else:
        import fastworkflow

        fastworkflow.init({})

    cases = he.load_benchmark_file(args.benchmark)
    scopes = sorted({case.context for case in cases})

    def roots_of(kind: str, folder: str | None) -> list[str]:
        seeds = [None] if kind == "shipped" else args.seeds
        return [arm_root(kind, folder, seed, workflow=workflow, out=out) for seed in seeds]

    def routers_of(kind: str, root: str) -> dict:
        return {scope: build_router(kind, root, scope) for scope in scopes}

    verdicts: dict[str, list[list[dict]]] = {}
    majorities: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    facts: dict[str, list[dict]] = {}
    for arm, kind, folder in arms:
        verdicts[arm] = [predict_cases(routers_of(kind, root).__getitem__, cases)
                         for root in roots_of(kind, folder)]
        summaries[arm], majorities[arm] = summarise_arm(verdicts[arm])
        facts[arm] = [training_facts(kind, root, scopes) for root in roots_of(kind, folder)]
    for arm in summaries:
        if arm != baseline:
            summaries[arm][f"mcnemar_vs_{baseline}"] = compare_with_baseline(
                verdicts[arm], majorities[arm], verdicts[baseline], majorities[baseline])

    operational: dict = {"bytes_on_disk": {
        arm: arm_bytes(kind, roots_of(kind, folder)[0], scopes) for arm, kind, folder in arms}}
    if not args.skip_operational:
        first = {arm: (kind, roots_of(kind, folder)[0]) for arm, kind, folder in arms}
        timings = measure_latency(
            {arm: routers_of(kind, root) for arm, (kind, root) in first.items()},
            cases, args.latency_repetitions)
        operational.update({
            "latency": {arm: latency_summary(values) for arm, values in timings.items()},
            "latency_ms_raw": timings,
            "latency_note": f"first seed of each arm; arms take turns per case, "
                            f"{args.latency_repetitions} timed passes after one warm-up pass",
            "resident_memory": {arm: measure_memory(kind, root, scopes)
                                for arm, (kind, root) in first.items()},
            "torch_threads": __import__("torch").get_num_threads(),
        })

    suffix = f"_{args.tag}" if args.tag else ""
    results = {
        "workflow": workflow,
        "benchmark": os.path.abspath(args.benchmark),
        "cases": {"routing": sum(c.kind == "routing" for c in cases),
                  "escalation": sum(c.kind == "escalation" for c in cases)},
        "scopes": scopes,
        "seeds": list(args.seeds),
        "baseline": baseline,
        "arm_specs": {arm: {"kind": kind, "folder": folder} for arm, kind, folder in arms},
        "arms": summaries,
        "training": facts,
        "operational": operational,
    }
    with open(os.path.join(out, name, f"results{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out, name, f"verdicts{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"workflow": workflow, "seeds": verdicts, "majority": majorities}, f, indent=2)
    print_table(name, summaries)
    print(f"wrote {os.path.join(out, name, f'results{suffix}.json')} and verdicts{suffix}.json")
    return 0


# ---------------------------------------------------------------------------
# Reporting and pooling
# ---------------------------------------------------------------------------


def _cell(entry: dict) -> str:
    if not entry or not entry.get("n"):
        return "      n/a      "
    low, high = entry["wilson95"]
    return f"{entry['rate']:6.1%} [{low:4.0%},{high:4.0%}]"


def print_table(title: str, arms: dict) -> None:
    print(f"\n== {title}")
    names = ("routing_top1", "routing_in_list", "routing_one_label_share",
             "lone_wrong_of_lone_labels", "escalation_recall", "false_escalation_on_routing")
    print(f"{'arm':4} " + " ".join(f"{name[:17]:>17}" for name in names))
    for arm, summary in arms.items():
        summary = summary.get("majority", summary)
        print(f"{arm:4} " + " ".join(f"{_cell(summary[name]):>17}" for name in names))


def print_seed_ranges(arms: dict) -> None:
    """Per-seed counts beside the majority table, since a majority hides the spread."""
    for arm, summary in arms.items():
        if len(summary["seeds"]) < 2:
            continue
        cells = []
        for name in ("routing_top1", "routing_in_list", "routing_one_label_share",
                     "lone_wrong_of_routing", "escalation_recall", "false_escalation_on_routing"):
            cells.append(f"{name[8:] if name.startswith('routing_') else name} "
                         + "/".join(str(seed[name]["k"]) for seed in summary["seeds"])
                         + f" of {summary['seeds'][0][name]['n']}")
        agreement = summary["seed_agreement"]["same_verdict"]
        print(f"  {arm:4} per seed: " + "; ".join(cells) + f"; seeds agree {agreement['rate']:.1%}")


def conditions_v1(pooled: dict, per_workflow: dict, baseline: str, candidate: str) -> list[dict]:
    """The seven pass conditions of the first pre-registration for one arm."""
    base, arm = pooled["arms"][baseline]["majority"], pooled["arms"][candidate]["majority"]
    paired = pooled["arms"][candidate][f"mcnemar_vs_{baseline}"]["majority"]
    conditions = []

    def add(number, text, passed, numbers):
        conditions.append({"condition": number, "text": text, "pass": bool(passed), **numbers})

    top1_drop = base["routing_top1"]["rate"] - arm["routing_top1"]["rate"]
    mcnemar = paired["top1"]
    baseline_better = (
        mcnemar["exact_mcnemar_p"] < 0.05
        and mcnemar["baseline_right_arm_wrong"] > mcnemar["arm_right_baseline_wrong"]
    )
    add(1, "pooled top-1 not worse than the baseline by more than 3 points, and McNemar does "
           "not show the baseline better at p < 0.05",
        top1_drop <= 0.03 and not baseline_better,
        {"baseline": base["routing_top1"]["rate"], "arm": arm["routing_top1"]["rate"], "mcnemar": mcnemar})
    in_list_drop = base["routing_in_list"]["rate"] - arm["routing_in_list"]["rate"]
    add(2, "pooled in-list not worse than the baseline by more than 2 points", in_list_drop <= 0.02,
        {"baseline": base["routing_in_list"]["rate"], "arm": arm["routing_in_list"]["rate"]})
    # A benchmark without escalation cases has no recall; only the false-escalation half applies.
    recall_drop = (base["escalation_recall"]["rate"] or 0.0) - (arm["escalation_recall"]["rate"] or 0.0)
    false_escalation = arm["false_escalation_on_routing"]["rate"]
    add(3, "escalation recall not worse than the baseline by more than one case in twenty, and "
           "false escalation on routing cases at most 3%",
        recall_drop <= 0.05 and false_escalation <= 0.03,
        {"baseline_recall": base["escalation_recall"]["rate"],
         "arm_recall": arm["escalation_recall"]["rate"], "arm_false_escalation": false_escalation})
    losses = scope_losses(base, arm, minimum_cases=20, fields=("per_scope_top1",))
    add(4, "no scope with at least 20 cases loses more than 10 points of top-1",
        all(v["per_scope_top1"]["loss"] <= 0.10 for v in losses.values()), {"scopes": losses})
    conditions.append(disk_condition(5, per_workflow, baseline, candidate))
    conditions.append(latency_condition(6, pooled, per_workflow, baseline, candidate))
    agreement = pooled["arms"][candidate]["seed_agreement"]["same_verdict"]
    add(7, "the three seeds agree on at least 90% of case verdicts", agreement["rate"] >= 0.90,
        {"agreement": agreement["rate"], "n": agreement["n"]})
    return conditions


def scope_losses(base: dict, arm: dict, *, minimum_cases: int, fields: tuple) -> dict:
    """Points each large enough scope loses against the baseline, per rate."""
    losses: dict = {}
    for scope, entry in base["per_scope_top1"].items():
        if entry["n"] < minimum_cases:
            continue
        losses[scope] = {"n": entry["n"]}
        for name in fields:
            losses[scope][name] = {
                "baseline": base[name][scope]["rate"], "arm": arm[name][scope]["rate"],
                "loss": base[name][scope]["rate"] - arm[name][scope]["rate"]}
    return losses


def disk_condition(number: int, per_workflow: dict, baseline: str, candidate: str) -> dict:
    most_contexts = max(per_workflow.values(), key=lambda r: len(r["scopes"]))
    disk = most_contexts["operational"]["bytes_on_disk"]
    return {"condition": number,
            "text": "disk per version at most 40% of the heads' on the workflow with most contexts",
            "pass": disk[candidate] <= 0.40 * disk[baseline],
            "workflow": most_contexts["workflow"], "baseline_bytes": disk[baseline],
            "arm_bytes": disk[candidate], "ratio": disk[candidate] / disk[baseline]}


def latency_condition(number: int, pooled: dict, per_workflow: dict, baseline: str, candidate: str) -> dict:
    latency = pooled.get("latency") or {}
    if baseline not in latency or candidate not in latency:
        return {"condition": number, "text": "latency p95 at most 120% of the baseline's",
                "pass": None, "note": "latency was not measured in these results"}
    return {"condition": number, "text": "latency p95 at most 120% of the baseline's",
            "pass": latency[candidate]["p95_ms"] <= 1.20 * latency[baseline]["p95_ms"],
            "baseline_p95_ms": latency[baseline]["p95_ms"], "arm_p95_ms": latency[candidate]["p95_ms"],
            "baseline_mean_ms": latency[baseline]["mean_ms"], "arm_mean_ms": latency[candidate]["mean_ms"]}


def conditions_v2(pooled: dict, per_workflow: dict, baseline: str, candidate: str) -> list[dict]:
    """The six pass conditions of the second pre-registration for one arm."""
    base_arm, arm_all = pooled["arms"][baseline], pooled["arms"][candidate]
    base, arm = base_arm["majority"], arm_all["majority"]
    paired = arm_all[f"mcnemar_vs_{baseline}"]
    conditions = []

    def add(number, text, passed, numbers):
        conditions.append({"condition": number, "text": text, "pass": bool(passed), **numbers})

    def baseline_better(test: dict) -> bool:
        return (test["exact_mcnemar_p"] < 0.05
                and test["baseline_right_arm_wrong"] > test["arm_right_baseline_wrong"])

    add(1, "strict top-1 by majority not below the baseline, and not below it by McNemar at "
           "p < 0.05 in any single seed",
        arm["routing_top1"]["rate"] >= base["routing_top1"]["rate"]
        and not any(baseline_better(seed["top1"]) for seed in paired["seeds"]),
        {"baseline": base["routing_top1"]["rate"], "arm": arm["routing_top1"]["rate"],
         "mcnemar_per_seed": [seed["top1"] for seed in paired["seeds"]]})
    add(2, "in-list by majority not below the baseline by more than 3 points, and McNemar does "
           "not show the baseline better at p < 0.05",
        base["routing_in_list"]["rate"] - arm["routing_in_list"]["rate"] <= 0.03
        and not baseline_better(paired["majority"]["in_list"]),
        {"baseline": base["routing_in_list"]["rate"], "arm": arm["routing_in_list"]["rate"],
         "mcnemar": paired["majority"]["in_list"]})
    false_escalation = [seed["false_escalation_on_routing"]["rate"] for seed in arm_all["seeds"]]
    add(3, "escalation recall by majority above the baseline, and false escalation on routing "
           "cases at most 3% in every seed",
        # A benchmark without escalation cases has no recall to compare.
        (arm["escalation_recall"]["rate"] is None
         or arm["escalation_recall"]["rate"] > base["escalation_recall"]["rate"])
        and max(false_escalation) <= 0.03,
        {"baseline_recall": base["escalation_recall"]["rate"],
         "arm_recall": arm["escalation_recall"]["rate"], "arm_false_escalation_per_seed": false_escalation})
    losses = scope_losses(base, arm, minimum_cases=30, fields=("per_scope_top1", "per_scope_in_list"))
    add(4, "no scope with at least 30 routing cases loses more than 10 points of in-list or top-1",
        all(v[name]["loss"] <= 0.10 for v in losses.values()
            for name in ("per_scope_top1", "per_scope_in_list")),
        {"scopes": losses})
    own, other = arm_all["seed_agreement"]["same_verdict"], base_arm["seed_agreement"]["same_verdict"]
    add(5, "seed agreement on case verdicts not more than 5 points below the baseline's own",
        own["rate"] >= other["rate"] - 0.05, {"baseline": other["rate"], "arm": own["rate"]})
    disk = disk_condition(6, per_workflow, baseline, candidate)
    latency = latency_condition(6, pooled, per_workflow, baseline, candidate)
    conditions.append({
        "condition": 6,
        "text": "disk at most 40% of the heads', and latency p95 at most 120% of theirs",
        "pass": None if latency["pass"] is None else bool(disk["pass"] and latency["pass"]),
        "disk": disk, "latency": latency})
    return conditions


def _pool(args: argparse.Namespace) -> int:
    per_workflow: dict = {}
    seeds: dict[str, list[list[dict]]] = {}
    timings: dict[str, list[float]] = defaultdict(list)
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        name = os.path.basename(os.path.normpath(result["workflow"]))
        per_workflow[name] = result
        verdict_path = os.path.join(
            os.path.dirname(path), os.path.basename(path).replace("results", "verdicts", 1))
        with open(verdict_path, "r", encoding="utf-8") as f:
            for arm, per_seed in json.load(f)["seeds"].items():
                target = seeds.setdefault(arm, [[] for _ in per_seed])
                for index, cases in enumerate(per_seed):
                    # A scope name is only unique inside its workflow, so it is qualified here.
                    target[index].extend(
                        {**case, "scope": f"{name}:{case['scope']}", "key": f"{name}|{case['key']}"}
                        for case in cases)
        for arm, values in (result["operational"].get("latency_ms_raw") or {}).items():
            timings[arm].extend(values)

    baseline = args.baseline or next(iter(per_workflow.values()))["baseline"]
    arms, majorities = {}, {}
    for arm, per_seed in seeds.items():
        arms[arm], majorities[arm] = summarise_arm(per_seed)
    for arm in arms:
        if arm != baseline:
            arms[arm][f"mcnemar_vs_{baseline}"] = compare_with_baseline(
                seeds[arm], majorities[arm], seeds[baseline], majorities[baseline])
    pooled = {
        "workflows": sorted(per_workflow),
        "baseline": baseline,
        "arms": arms,
        "latency": {arm: latency_summary(values) for arm, values in timings.items()},
    }
    # The third pre-registration keeps the second's conditions unchanged.
    judge = conditions_v2 if args.conditions in ("v2", "v3") else conditions_v1
    candidates = args.candidates or [
        arm for arm in arms if arm != baseline and len(seeds[arm]) > 1]
    pooled["pass_conditions"] = {
        arm: judge(pooled, per_workflow, baseline, arm) for arm in candidates}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2)

    for name, result in per_workflow.items():
        print_table(name, result["arms"])
        print_seed_ranges(result["arms"])
    print_table("pooled", arms)
    print_seed_ranges(arms)
    for arm, conditions in pooled["pass_conditions"].items():
        print(f"\npass conditions ({args.conditions}), {arm} against {baseline}:")
        for condition in conditions:
            verdict = {True: "PASS", False: "FAIL", None: "n/a "}[condition["pass"]]
            print(f"  {condition['condition']}. {verdict}  {condition['text']}")
    print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="Score arms on a benchmark (training the first round's first).")
    run.add_argument("--workflow", required=True)
    run.add_argument("--benchmark", required=True,
                     help="Benchmark file to score; any file, the artifacts are not retrained for it.")
    run.add_argument("--out", default=DEFAULT_OUT_ROOT,
                     help=f"Artifact and result root (default {DEFAULT_OUT_ROOT}).")
    run.add_argument("--tag", default="", help="Suffix of the result files, e.g. v2.")
    run.add_argument("--arms", nargs="+", default=None,
                     help="NAME=KIND[:FOLDER] ...; default the first round's arms.")
    run.add_argument("--baseline", default=None, help="Arm the others are paired with; default the first.")
    run.add_argument("--no-train", action="store_true", help="Only score artifacts that exist.")
    run.add_argument("--env", default=None, help="Env file; default the workflow's own, else the examples'.")
    run.add_argument("--passwords", default=None)
    run.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    run.add_argument("--retrain", action="store_true", help="Train even when artifacts are present.")
    run.add_argument("--train-only", action="store_true", help="Stop after training.")
    run.add_argument("--skip-operational", action="store_true", help="No latency or memory measurement.")
    run.add_argument("--latency-repetitions", type=int, default=3)
    run.set_defaults(func=_run)

    heads = sub.add_parser("heads", help="Rebuild the per-context heads at one seed.")
    heads.add_argument("--workflow", required=True)
    heads.add_argument("--out", default=DEFAULT_OUT_ROOT)
    heads.add_argument("--arm", default="H3", help="Folder name of this set of heads.")
    heads.add_argument("--seed", type=int, required=True)
    heads.add_argument("--env", default=None)
    heads.add_argument("--passwords", default=None)
    heads.add_argument("--retrain", action="store_true")
    heads.add_argument("--no-built-in-vague-class", action="store_true",
                       help="For a head trained before a head without foreign commands "
                            "got the built-in vague replies as its wildcard class.")
    heads.set_defaults(func=_heads)

    single = sub.add_parser("single", help="Train the workflow's one model pair at one seed.")
    single.add_argument("--workflow", required=True)
    single.add_argument("--out", default=DEFAULT_OUT_ROOT)
    single.add_argument("--arm", required=True, help="Folder name of this recipe.")
    single.add_argument("--seed", type=int, required=True)
    single.add_argument("--wildcard-weight", type=int, default=None,
                        help="Repeats of each built-in vague reply in the workflow wildcard "
                             "class; 0 leaves the class out; default the heads' weight.")
    single.add_argument("--env", default=None)
    single.add_argument("--passwords", default=None)
    single.add_argument("--retrain", action="store_true")
    single.set_defaults(func=_single)

    calibrate = sub.add_parser(
        "calibrate-holdout", help="Recompute thresholds and margin on the persona holdout.")
    calibrate.add_argument("--workflow", required=True)
    calibrate.add_argument("--out", default=DEFAULT_OUT_ROOT)
    calibrate.add_argument("--arm", required=True, help="Folder of the trained artifacts, e.g. M2.")
    calibrate.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    calibrate.add_argument("--wildcard-weight", type=int, default=None,
                           help="The weight the artifacts were trained with (default the heads').")
    calibrate.add_argument("--env", default=None)
    calibrate.add_argument("--passwords", default=None)
    calibrate.set_defaults(func=_calibrate_holdout)

    pool = sub.add_parser("pool", help="Add workflows up and evaluate the pass conditions.")
    pool.add_argument("--results", nargs="+", required=True)
    pool.add_argument("--out", required=True)
    pool.add_argument("--baseline", default=None, help="Default: the baseline the results name.")
    pool.add_argument("--candidates", nargs="+", default=None,
                      help="Arms to judge; default every seeded arm but the baseline.")
    pool.add_argument("--conditions", choices=("v1", "v2", "v3"), default="v1",
                      help="v3 is v2's conditions, for the third pre-registration.")
    pool.set_defaults(func=_pool)

    memory = sub.add_parser("memory", help="Internal: load one arm and print its resident memory.")
    memory.add_argument("--kind", required=True, choices=ARM_KINDS)
    memory.add_argument("--root", required=True)
    memory.add_argument("--scopes", nargs="+", required=True)
    memory.set_defaults(func=_memory)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
