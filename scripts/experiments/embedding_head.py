#!/usr/bin/env python3
"""Score the ``embedding`` intent model layout through the product trainer and router.

``product`` trains a temporary copy of each workflow with the product trainer
(`model_pipeline_training.train(..., intent_model_layout="embedding")`) and scores the benchmarks
through the product router (`CommandRouter` on the copy's context folders, behind
`intent_detector.ClassifierEngine`, which is what hands a request on with the commands of other scopes
among the candidates). It reports, per workflow and pooled:

* right outright: the one label the head answers is the expected command;
* wrong confident: the head answers one label and it is not the expected command (a bare value or the
  vague-reply class counts too);
* handed on, with the right command among the candidates / without it (routing requests the head did not
  answer with one label);
* for escalation requests (the command is in an ancestor scope): how often the expected ancestor command is
  among the candidates handed on, and how often the head answered one label anyway.

It also reports the encoder's load time and memory, the head's training time, the artifact's bytes and the
request latency on this machine (``--latency-threads 10 1`` measures each torch thread count).

The candidates are scored with every command of every other scope reachable. The runtime keeps only the
commands of scopes on the request's lookup path, which can only remove competitors, so the share of
escalation requests whose ancestor is among the candidates is a lower bound.

Usage
-----
    embedding_head.py product [--encoder ID] [--suffixes "" _v2 _v3] [--workflows PATH ...] [--out DIR]

``--suffixes`` picks the benchmark files ``tests/fixtures/intent_benchmarks/<workflow><suffix>.json`` (``""`` is
the first benchmark); several are pooled. ``--encoder`` is any encoder the layout can load (default
`intent_model_layout.DEFAULT_ENCODER`). ``--commit-threshold`` overrides ``INTENT_COMMIT_THRESHOLD``.

The pre-registered calibration arithmetic of ``docs/single_intent_model_preregistration_v4.md`` (the former
``pooled`` mode) needed the cross-fit calibration of a threshold and a pass-up margin, which were removed from the
product; it is in git history.

Nothing is written into a workflow folder: rows are rebuilt and heads are fitted on a copy. No
language model is called; a cold utterance cache stops the run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_WORKFLOWS = (
    str(REPO / "fastworkflow" / "examples" / "library"),
    str(REPO / "fastworkflow" / "examples" / "retail_workflow"),
)
DEFAULT_OUT_ROOT = os.path.join(tempfile.gettempdir(), "embedding_head")
BENCHMARKS = REPO / "tests" / "fixtures" / "intent_benchmarks"


def _benchmark_cases(workflow: str, suffixes: list[str]):
    from fastworkflow.train import heldout_evaluation as he

    name = os.path.basename(os.path.normpath(workflow))
    cases = []
    for suffix in suffixes:
        cases.extend(he.load_benchmark_file(str(BENCHMARKS / f"{name}{suffix}.json")))
    return cases


def _environment(workflow: str, args: argparse.Namespace) -> dict:
    import single_intent_model as runner
    from fastworkflow.train import intent_model_layout as layout

    env = runner.environment_for(workflow, None, None)
    env[layout.SETTING] = layout.EMBEDDING
    env[layout.ENCODER_SETTING] = args.encoder
    for setting, value in ((layout.POOLING_SETTING, args.pooling),
                           (layout.COMMIT_THRESHOLD_SETTING, args.commit_threshold)):
        env.pop(setting, None)
        if value is not None:
            env[setting] = str(value)
    return env


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def verdict(case, labels: list[str], details: dict) -> dict:
    """One case's outcome from what the detector's local-model link answered (`ClassifierEngine.decide`)."""
    from fastworkflow.train import heldout_evaluation as he

    predicted = [he.normalize_label(str(label)) for label in labels]
    one = len(predicted) == 1
    out = {"kind": case.kind, "scope": case.context, "utterance": case.utterance, "predicted": predicted,
           "one_label": one, "handed_on": list(details.get("handed_on", ()))}
    if case.kind == "routing":
        expected = he.normalize_label(case.expected_label or "")
        out["expected"] = expected
        out["right_outright"] = predicted == [expected]
        out["wrong_confident"] = one and predicted != [expected]
        out["handed_with_right"] = (not one) and expected in predicted
        out["handed_without_right"] = (not one) and expected not in predicted
    else:
        ancestor = he.normalize_label(case.expected_ancestor_command or "")
        out["expected"] = ancestor
        out["lone_answer"] = one
        out["ancestor_among_candidates"] = (not one) and any(
            label == ancestor or label.split("/")[-1] == ancestor.split("/")[-1] for label in predicted)
    return out


def summarise(verdicts: list[dict]) -> dict:
    """The pooled counts and shares of the metrics listed in the module docstring."""
    routing = [v for v in verdicts if v["kind"] == "routing"]
    escalation = [v for v in verdicts if v["kind"] == "escalation"]
    n, m = len(routing), len(escalation)
    count = lambda rows, field: sum(bool(v[field]) for v in rows)  # noqa: E731
    return {
        "routing_cases": n, "escalation_cases": m,
        "right_outright": _share(count(routing, "right_outright"), n),
        "wrong_confident": _share(count(routing, "wrong_confident"), n),
        "handed_on_with_right": _share(count(routing, "handed_with_right"), n),
        "handed_on_without_right": _share(count(routing, "handed_without_right"), n),
        "escalation_ancestor_among_candidates": [count(escalation, "ancestor_among_candidates"), m],
        "escalation_lone_answer": [count(escalation, "lone_answer"), m],
    }


def format_summary(name: str, summary: dict) -> str:
    found, cases = summary["escalation_ancestor_among_candidates"]
    lone, _ = summary["escalation_lone_answer"]
    return (
        f"{name}: right outright {summary['right_outright']:.1%}, wrong confident {summary['wrong_confident']:.1%}, "
        f"handed on with the right command {summary['handed_on_with_right']:.1%}, without it "
        f"{summary['handed_on_without_right']:.1%}; escalation: ancestor among the candidates {found}/{cases}, "
        f"answered with one label {lone}/{cases} ({summary['routing_cases']} routing, {cases} escalation cases)"
    )


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * share))]


def _resident_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _encoder_load(args: argparse.Namespace) -> dict:
    """Seconds and resident memory to load the encoder in this process (weights already on disk)."""
    from fastworkflow.train import embedding_head as eh
    from fastworkflow.train import intent_model_layout as layout

    before = _resident_mb()
    started = time.perf_counter()
    encoder = eh.SentenceEncoder.get(
        args.encoder, args.pooling, layout.default_prefix(args.encoder))
    encoder.embed(["warm up"])
    return {"encoder": args.encoder, "load_seconds": time.perf_counter() - started,
            "resident_mb_added": _resident_mb() - before, "resident_mb_total": _resident_mb()}


def _timings(routers: dict, cases, repetitions: int, threads: int) -> dict:
    """Milliseconds per request through the product router with *threads* torch (intra-op) threads, one
    request at a time. Each request is a text the encoder has not just embedded: its small memo is cleared
    before each call, so a number is one forward pass plus the head. The thread count is restored after."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for case in cases[:20]:
            routers[case.context].predict_with_details(case.utterance)
        timings: list[float] = []
        for _ in range(repetitions):
            for case in cases:
                router = routers[case.context]
                router.modelpipeline.encoder._memo.clear()
                started = time.perf_counter()
                router.predict_with_details(case.utterance)
                timings.append((time.perf_counter() - started) * 1000)
    finally:
        torch.set_num_threads(previous)
    return {"threads": threads, "requests": len(timings), "median_ms": statistics.median(timings),
            "p95_ms": _percentile(timings, 0.95)}


def _product(args: argparse.Namespace) -> int:
    import fastworkflow
    import single_intent_model as runner
    from fastworkflow.intent import intent_detector
    from fastworkflow import model_pipeline_training as heads
    from fastworkflow.train import artifact_versioning, determinism, embedding_head, utterance_cache
    from fastworkflow.train import intent_model_layout as layout
    from fastworkflow.train import workflow_model as wm
    from fastworkflow.train.selective_training import contexts_for_training

    suffix_tag = "+".join(s or "_v1" for s in args.suffixes)
    out = os.path.join(args.out, "product", args.encoder.replace("/", "__"))
    first_env = _environment(os.path.abspath(args.workflows[0]), args)
    fastworkflow.init(env_vars=first_env)
    load = _encoder_load(args)
    print(f"encoder {load['encoder']}: loaded in {load['load_seconds']:.1f} s, +{load['resident_mb_added']:.0f} MB "
          f"resident ({load['resident_mb_total']:.0f} MB total)")
    verdicts, per_workflow = [], {}
    for workflow in args.workflows:
        workflow = os.path.abspath(workflow)
        name = os.path.basename(os.path.normpath(workflow))
        fastworkflow.init(env_vars=_environment(workflow, args))
        copy = runner.copy_workflow(workflow, os.path.join(out, name))
        cache = wm._ReadOnlyUtteranceCache(copy, mode=utterance_cache.MODE_REUSE)
        handle = fastworkflow.Workflow.create(copy, workflow_id_str=f"embedding_head_{name}")
        previous = (determinism.get_provenance_recorder(), utterance_cache.get_utterance_cache())
        started = time.monotonic()
        try:
            determinism.set_provenance_recorder(determinism.ProvenanceRecorder(copy))
            utterance_cache.set_utterance_cache(cache)
            with wm.no_model_calls():
                heads.train(handle, intent_model_layout=layout.EMBEDDING)
            contexts = sorted(contexts_for_training(handle.folderpath))
        finally:
            determinism.set_provenance_recorder(previous[0])
            utterance_cache.set_utterance_cache(previous[1])
            handle.close()
        train_seconds = time.monotonic() - started

        info = os.path.join(copy, "___command_info")
        model_dir = os.path.join(info, layout.WORKFLOW_MODEL_DIRNAME)
        summary = embedding_head.read_summary(model_dir)
        with open(os.path.join(model_dir, layout.SCOPES_FILENAME), encoding="utf-8") as f:
            scopes_file = json.load(f)
        routers = {
            context: heads.CommandRouter(
                os.path.join(info, artifact_versioning.context_folder_name(context)))
            for context in contexts
        }
        # Every command of every scope is reachable (see the module docstring).
        everything = frozenset(label for label in scopes_file["labels"]
                               if label not in (embedding_head.PARAMETER_VALUE_LABEL, embedding_head.WILDCARD_LABEL))
        engines = {context: intent_detector.ClassifierEngine(router, reachable=everything)
                   for context, router in routers.items()}
        cases = _benchmark_cases(workflow, args.suffixes)
        scored = []
        for case in cases:
            answer = engines[case.context].decide(case.utterance)
            scored.append(verdict(case, answer.labels, answer.details))
        verdicts.extend(scored)
        per_workflow[name] = {
            "summary": summarise(scored),
            "threshold": summary["threshold"],
            "cross_fit": summary["cross_fit"],
            "labels": summary["labels"],
            "rows": summary["rows"],
            "description_rows": summary["description_rows"],
            "head_seconds": {key: summary[key] for key in (
                "embed_seconds", "fit_seconds", "cross_fit_seconds", "train_seconds")},
            "trainer_seconds_with_reports": train_seconds,
            "artifact_bytes": summary["bytes"],
            "context_files_bytes": sum(
                runner.directory_bytes(os.path.join(info, artifact_versioning.context_folder_name(c)))
                for c in contexts),
            "latency": [_timings(routers, cases, args.latency_repetitions, threads)
                        for threads in args.latency_threads],
        }
        print(format_summary(name, per_workflow[name]["summary"]))
        cross_fit = summary["cross_fit"]
        print(f"  commit threshold {summary['threshold']:.2f}; cross-fit {cross_fit['accuracy']:.1%} right of "
              f"{cross_fit['rows']} rows, {cross_fit['confused_pairs']} confused pairs, "
              f"{cross_fit['pairs_over_limit']} over 10%; {summary['rows']} rows "
              f"({summary['description_rows']} description and name rows)")
        print(f"  head: {summary['train_seconds']:.1f} s (embedding {summary['embed_seconds']:.1f} s, "
              f"fit {summary['fit_seconds']:.1f} s, cross-fit {summary['cross_fit_seconds']:.1f} s); "
              f"trainer with reports {train_seconds:.1f} s; artifact {summary['bytes']:,} bytes")
        for latency in per_workflow[name]["latency"]:
            print(f"  latency, {latency['threads']} thread(s): median {latency['median_ms']:.1f} ms, "
                  f"p95 {latency['p95_ms']:.1f} ms ({latency['requests']} requests)")
    pooled = summarise(verdicts)
    print(format_summary("pooled", pooled))
    _write(out, f"results{suffix_tag}.json", {
        "mode": "product", "encoder": args.encoder, "suffixes": args.suffixes, "encoder_load": load,
        "pooled": pooled, "workflows": per_workflow})
    _write(out, f"verdicts{suffix_tag}.json", verdicts)
    return 0


def _write(folder: str, name: str, payload) -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    from fastworkflow.train import intent_model_layout as layout

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("product",))
    parser.add_argument("--encoder", default=layout.DEFAULT_ENCODER)
    parser.add_argument("--pooling", choices=("cls", "mean"), default=None)
    parser.add_argument("--commit-threshold", type=float, default=None)
    parser.add_argument("--suffixes", nargs="+", default=["_v3"],
                        help='benchmark file suffixes, pooled: "" (the first), "_v2", "_v3", "_v4"')
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--out", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--latency-repetitions", type=int, default=2)
    parser.add_argument("--latency-threads", nargs="+", type=int, default=[10, 1])
    args = parser.parse_args(argv)
    return _product(args)


if __name__ == "__main__":
    raise SystemExit(main())
