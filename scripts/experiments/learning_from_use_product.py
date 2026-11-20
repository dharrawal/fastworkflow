#!/usr/bin/env python3
"""Learning from use, through the product code (bd fix-o5wi.6): the recording, the refit with its gate, the lower-only
calibration and the reload of a published head.

It trains a temporary COPY of each example workflow with the product trainer (`fastworkflow train`'s `train_workflow`, so
the copy has real versions), then plays the development benchmarks (1 to 3 pooled, routing requests, shuffled) as a
stream of requests with truth as the confirmation:

* each request is asked of the local head through the product router (`intent_detector.ClassifierEngine`, behind
  `intent_detector._command_router`, which picks up a newly published head);
* a confident right answer is recorded as an unconfirmed ``local`` row (it teaches nothing); an answer that is not
  confident is a which-command question the user answers with the truth (a confirmed ``user_choice`` row); a confident
  wrong answer is a review the user switches to the truth (a confirmed ``review_switch`` row whose ``rejected_label``
  is the wrong answer). Every row is written by `intent_learning.record`, the one writer the resolution path and the
  CME path call, with the evidence the engine's answer carries (`intent_learning.evidence_of`);
* after every ``--every`` new teaching rows the product refit runs (`head_refit.learn`: refit behind its gate, then the
  calibration), as ``INTENT_LEARNING_REFIT=background`` would.

The fourth benchmark (scored once, never played as a stream) is scored with the base head and with the head the stream
ended on, and the shuffles of the stream are repeated. No model is called: the encoder is local and the utterance
caches are read back. Nothing is written into a workflow folder; the copies live under ``--out``.

Usage: learning_from_use_product.py [--workflows PATH ...] [--shuffles 3] [--every 25] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

DEFAULT_WORKFLOWS = (str(REPO / "fastworkflow" / "examples" / "library"),
                     str(REPO / "fastworkflow" / "examples" / "retail_workflow"))
BENCHMARKS = REPO / "tests" / "fixtures" / "intent_benchmarks"


def _cases(name: str, suffixes) -> list:
    from fastworkflow.train import heldout_evaluation as he

    out = []
    for suffix in suffixes:
        out.extend(he.load_benchmark_file(str(BENCHMARKS / f"{name}{suffix}.json")))
    return out


def _environment(workflow: str) -> dict:
    import single_intent_model as runner
    from fastworkflow.train import intent_model_layout as layout

    env = runner.environment_for(workflow, None, None)
    env.pop("JEV_API_KEY", None)
    for setting in (layout.SETTING, layout.ENCODER_SETTING, layout.POOLING_SETTING, layout.COMMIT_THRESHOLD_SETTING):
        env.pop(setting, None)
    return env


def _train_copy(workflow: str, root: str) -> str:
    """A trained, versioned copy of *workflow* (no model call: the utterance and parameter-example caches are read)."""
    import single_intent_model as runner

    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.train import workflow_model
    from fastworkflow.train.__main__ import train_workflow
    from fastworkflow.train import __main__ as trainer

    copy = runner.copy_workflow(workflow, root)
    fastworkflow.init(env_vars=_environment(workflow))
    RoutingRegistry.clear_registry()
    # The parameter-example files are written for the parameter extractor, not for the intent head, and drawing them
    # would call a model when their cache key is stale: this run trains the intent head only.
    trainer._generate_dspy_examples_helper = lambda workflow: None
    with workflow_model.no_model_calls():
        train_workflow(copy)
    return copy


def _engines(workflow: str, contexts, everything):
    from fastworkflow import intent_detector

    return {context: intent_detector.ClassifierEngine(
        intent_detector._command_router(intent_detector.head_folder(workflow, context)), reachable=everything)
        for context in contexts}


def _score(workflow: str, cases, everything) -> dict:
    from embedding_head import summarise, verdict

    engines = _engines(workflow, {case.context for case in cases}, everything)
    scored = []
    for case in cases:
        answer = engines[case.context].decide(case.utterance)
        scored.append(verdict(case, answer.labels, answer.details))
    return summarise(scored)


def _play(workflow: str, stream, everything, every: int) -> dict:
    """The stream through the recording hook and the product refit. Returns what happened."""
    from fastworkflow import intent_detector
    from fastworkflow import intent_learning as learning
    from fastworkflow.train import head_refit

    intent_detector.HEAD_RELOAD_CHECK_SECONDS = 0.0
    counts = {"right_outright": 0, "wrong_confident": 0, "asked": 0}
    refits = []
    for case in stream:
        engine = intent_detector.ClassifierEngine(
            intent_detector._command_router(intent_detector.head_folder(workflow, case.context)),
            reachable=everything)
        answer = engine.decide(case.utterance)
        evidence = {**learning.evidence_of([answer]), "context": case.context}
        if len(answer.labels) == 1 and answer.labels[0] == case.expected_label:
            counts["right_outright"] += 1
            how, rejected = learning.HOW_LOCAL, None
        elif len(answer.labels) == 1:
            counts["wrong_confident"] += 1
            how, rejected = learning.HOW_REVIEW_SWITCH, answer.labels[0]
        else:
            counts["asked"] += 1
            how, rejected = learning.HOW_USER_CHOICE, None
        learning.record(workflow, text=case.utterance, context=case.context, command=case.expected_label, how=how,
                        evidence=evidence, rejected_label=rejected)
        if learning.new_teaching_count(workflow) >= every:
            result = head_refit.learn(workflow)
            refits.append(result)
    final = head_refit.learn(workflow)
    refits.append(final)
    return {"stream": counts, "refits": refits, "final": final}


def _run(args) -> int:
    import single_intent_model as runner

    import fastworkflow
    from fastworkflow import intent_detector
    from fastworkflow import intent_learning as learning
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.train import artifact_versioning as av

    out = os.path.join(args.out, "learning_from_use")
    report = {}
    for source in args.workflows:
        name = os.path.basename(os.path.normpath(source))
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)
        base = _train_copy(os.path.abspath(source), os.path.join(out, "base"))
        base_version = av.resolve_current_version(base)
        model_dir = av.version_dir(base, base_version) / "workflow"
        labels = json.loads((model_dir / "scopes.json").read_text())["labels"]
        everything = frozenset(label for label in labels if label not in ("parameter_value", "wildcard"))
        dev = [c for c in _cases(name, ("", "_v2", "_v3")) if c.kind == "routing"]
        confirm = _cases(name, ("_v4",))
        runs = []
        for seed in range(args.shuffles):
            work = os.path.join(out, f"run{seed}", name)
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(os.path.dirname(work), exist_ok=True)
            shutil.copytree(base, work, symlinks=True)
            fastworkflow.init(env_vars=_environment(os.path.abspath(source)))
            RoutingRegistry.clear_registry()
            stream = list(dev)
            random.Random(seed).shuffle(stream)
            intent_detector.HEAD_RELOAD_CHECK_SECONDS = 0.0
            before = _score(work, confirm, everything)
            played = _play(work, stream, everything, args.every)
            intent_detector.HEAD_RELOAD_CHECK_SECONDS = 0.0
            after = _score(work, confirm, everything)
            published = [r["refit"] for r in played["refits"] if r["refit"].published]
            final_version = av.resolve_current_version(work)
            summary = {
                "shuffle": seed, "stream": played["stream"], "teaching_rows": learning.stats(work)["teaching_rows"],
                "corrections": learning.stats(work)["corrections"], "refits_published": len(published),
                "refit_reasons": [r["refit"].reason for r in played["refits"] if not r["refit"].published],
                "refit_seconds": [round(r.seconds, 1) for r in published],
                "embedded_texts": sum(r.embedded_texts for r in published),
                "gate_last": published[-1].gate if published else None,
                "bytes_base_head": runner.directory_bytes(str(model_dir)),
                "bytes_final_head": runner.directory_bytes(str(av.version_dir(work, final_version) / "workflow")),
                "before": before, "after": after, "calibration": played["final"]["calibration"],
                "versions": [v.version_id for v in av.list_versions(work)],
            }
            runs.append(summary)
            print(f"{name} shuffle {seed}: {summary['teaching_rows']} teaching rows, {summary['refits_published']} "
                  f"refit(s); v4 right outright {before['right_outright']:.1%} -> {after['right_outright']:.1%}, "
                  f"wrong confident {before['wrong_confident']:.1%} -> {after['wrong_confident']:.1%}, handed on "
                  f"{before['handed_on_with_right'] + before['handed_on_without_right']:.1%} -> "
                  f"{after['handed_on_with_right'] + after['handed_on_without_right']:.1%}")
        report[name] = {"base_version": base_version, "dev_requests": len(dev), "v4_requests": before["routing_cases"],
                        "runs": runs}
        mean = lambda key, which: statistics.mean(r[which][key] for r in runs)  # noqa: E731
        print(f"{name}: mean over {len(runs)} shuffles, v4 right outright {mean('right_outright', 'before'):.1%} -> "
              f"{mean('right_outright', 'after'):.1%}; wrong confident {mean('wrong_confident', 'before'):.1%} -> "
              f"{mean('wrong_confident', 'after'):.1%}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"results written to {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--shuffles", type=int, default=3)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "learning_from_use_runs"))
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
