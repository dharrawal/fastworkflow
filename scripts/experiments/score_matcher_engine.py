#!/usr/bin/env python3
"""Score the model matcher engine on a fixed intent benchmark, the way the trained heads are scored.

``intent_detector.MatcherEngine`` is the fallback engine of a scope with no trained head. Before it may
route execution commands it is measured against the heads on the same held-out benchmark
(``docs/unified_command_pipeline_design.md``, sections 3 and 8). This script does that measurement:

* ``--mode chain`` scores the detection chain (``intent_detector.ChainEngine``, the only behaviour since the
  ``INTENT_DETECTION_ENGINE`` setting was removed): the workflow's shipped head of the case's context answers
  first (``head_is_confident``: one command or a bare value), the matcher's answer is taken otherwise, and a
  matcher call that fails leaves the head's answer standing; the heads alone and the matcher alone are scored
  beside it on the same answers, each with its latency (one remote call per case, shared by the arms);
* every benchmark case is asked ONCE, with the scope path the live walk would use from the case's context,
  and the answer for the case's own scope is the verdict;
* a routing case passes **top-1** when the scope answers ``match`` on the expected label, and **in-list**
  when the expected label is among the labels returned (a match, or the candidates of an ambiguity);
* an escalation case passes when the scope answers ``not_mine`` (the walk would climb);
* latency is recorded per model call (mean, p95), with the response cache off unless ``--cache`` is given;
* with ``--baseline`` (a verdict file of ``score_benchmark.py score``) the two are paired case by case and
  compared with the exact two-sided McNemar test, per axis.

The verdict file has the format ``score_benchmark.py`` writes (``key``, ``context``, ``kind``,
``utterance``, ``predicted``, ``expected``, ``top1``, ``in_list``, ``escalated``), plus the matcher's own
facts per case, so ``score_benchmark.py compare`` reads it too.

These are REAL model calls (one per case), so ``JEV_API_KEY`` must be in the env files given. Nothing is
written into the workflow folder: the workflow is copied without its trained artifacts and scored there.

Scope paths. The walk asks the current context, then the context of each enclosing record up to the root
record; it does not climb to the global scope (``*`` is only asked by "confident global first", which needs a
trained global head). ``--path Context=Context,Parent`` gives the path of one context; a context without one
is asked alone. The paths of the bundled library example are built in (``--workflow`` ending in ``library``).

Usage
-----
    score_matcher_engine.py --workflow fastworkflow/examples/library \\
        --benchmark tests/fixtures/intent_benchmarks/library.json \\
        --env fastworkflow/examples/library/fastworkflow.env \\
        --passwords fastworkflow/examples/library/fastworkflow.passwords.env \\
        --baseline baseline_library.json --out matcher_library.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

#: The scope paths of the bundled library example: a Book or a Member is a record inside the Library root
#: record; Record is the base both inherit from, scored as a record inside the Library as well.
LIBRARY_PATHS = {
    "*": ["*"],
    "Library": ["Library"],
    "Book": ["Book", "Library"],
    "Member": ["Member", "Library"],
    "Record": ["Record", "Library"],
}


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b/c discordant pairs (as ``score_benchmark.py`` computes it)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def _copy_without_artifacts(workflow: Path, parent: Path) -> Path:
    target = parent / workflow.name
    shutil.copytree(workflow, target, ignore=shutil.ignore_patterns(
        "___command_info", "___convo_info", "___workflow_contexts", "__pycache__"))
    return target


def _paths(args: argparse.Namespace) -> dict[str, list[str]]:
    paths = dict(LIBRARY_PATHS) if Path(args.workflow).name == "library" else {}
    for item in args.path or ():
        context, _, rest = item.partition("=")
        paths[context] = [name for name in rest.split(",") if name]
    return paths


ARMS = ("heads", "matcher", "chain")


def _axis(name: str, field: str, before: dict[str, dict], after: dict[str, dict],
          before_name: str, after_name: str) -> dict | None:
    """One paired comparison of a boolean verdict field: counts, the discordant pairs, exact McNemar."""
    pairs = [(before[key], after[key]) for key in after
             if key in before and field in after[key] and field in before[key]]
    if not pairs:
        return None
    fixed = sum(1 for b, a in pairs if not b[field] and a[field])
    broken = sum(1 for b, a in pairs if b[field] and not a[field])
    return {
        "axis": name, "n": len(pairs),
        before_name: sum(bool(b[field]) for b, _ in pairs), after_name: sum(bool(a[field]) for _, a in pairs),
        f"{after_name}_only": fixed, f"{before_name}_only": broken,
        "mcnemar_p": exact_mcnemar_p(broken, fixed),
    }


def _verdict(case, kind: str, labels: list[str], core, heldout_evaluation) -> dict:
    """The verdict fields of one answer (its kind and its labels, best first) for one benchmark case.

    ``top1`` is STRICT (``heldout_evaluation.score_routing``'s definition): the answer is one command, the
    expected one. ``first_ranked`` only asks that the expected label is ranked first, alone or not.
    ``false_escalation`` is a routing case answered "not here"."""
    predicted = [heldout_evaluation.normalize_label(str(label)) for label in labels]
    verdict: dict = {"predicted": predicted, "answer": kind}
    if case.kind == "routing":
        expected = heldout_evaluation.normalize_label(case.expected_label or "")
        listed = kind in ("match", "ambiguous")
        verdict.update(
            expected=expected,
            # The framework's core commands are not offered to the matcher (the typed-name layers match
            # them), so a case that expects one cannot pass there; it is counted, and reported apart.
            expected_core=expected in core,
            top1=kind == "match" and predicted[:1] == [expected],
            first_ranked=listed and predicted[:1] == [expected],
            in_list=listed and expected in predicted,
            false_escalation=kind == "not_mine",
        )
    else:
        verdict.update(expected_ancestor_command=case.expected_ancestor_command, escalated=kind == "not_mine")
    return verdict


def _summary(records: list[dict]) -> dict:
    routing = [r for r in records if r["kind"] == "routing"]
    escalation = [r for r in records if r["kind"] == "escalation"]
    latencies = [float(r["latency_ms"]) for r in records]
    called = [r for r in records if r.get("calls")]
    out = {
        "cases": len(records),
        "model_calls": sum(int(r.get("calls") or 0) for r in records),
        "cases_with_a_model_call": len(called),
        "model_call_share": round(len(called) / len(records), 4) if records else None,
        "errors": sum("error" in r for r in records),
        "routing": {"n": len(routing), "top1": sum(r["top1"] for r in routing),
                    "first_ranked": sum(r["first_ranked"] for r in routing),
                    "in_list": sum(r["in_list"] for r in routing),
                    "false_escalation": sum(r["false_escalation"] for r in routing)},
        "routing_without_core_cases": {
            "n": sum(not r["expected_core"] for r in routing),
            "top1": sum(r["top1"] for r in routing if not r["expected_core"]),
            "in_list": sum(r["in_list"] for r in routing if not r["expected_core"])},
        "escalation": {"n": len(escalation), "recall": sum(r["escalated"] for r in escalation)},
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "p50": round(_percentile(latencies, 0.50), 1) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
            "total_s": round(sum(latencies) / 1000.0, 1),
        },
    }
    by_context: dict[str, dict] = {}
    for r in records:
        row = by_context.setdefault(r["context"], {"routing_n": 0, "top1": 0, "in_list": 0,
                                                   "escalation_n": 0, "escalated": 0})
        if r["kind"] == "routing":
            row["routing_n"] += 1
            row["top1"] += r["top1"]
            row["in_list"] += r["in_list"]
        else:
            row["escalation_n"] += 1
            row["escalated"] += r["escalated"]
    out["by_context"] = by_context
    return out


def _print_summary(name: str, summary: dict) -> None:
    routing, escalation, latency = summary["routing"], summary["escalation"], summary["latency_ms"]
    print(f"  [{name}] model calls on {summary['cases_with_a_model_call']}/{summary['cases']} cases "
          f"({summary['model_call_share']:.1%}), {summary['errors']} failed")
    if routing["n"]:
        n = routing["n"]
        print(f"    routing    n={n}  strict top-1={routing['top1']}/{n} ({routing['top1'] / n:.1%})  "
              f"first-ranked={routing['first_ranked']}/{n} ({routing['first_ranked'] / n:.1%})  "
              f"in-list={routing['in_list']}/{n} ({routing['in_list'] / n:.1%})  "
              f"false escalation={routing['false_escalation']}/{n} ({routing['false_escalation'] / n:.1%})")
        without = summary["routing_without_core_cases"]
        if without["n"] and without["n"] != n:
            print(f"    routing, cases expecting a framework core command left out: n={without['n']}  "
                  f"strict top-1={without['top1']}/{without['n']} ({without['top1'] / without['n']:.1%})  "
                  f"in-list={without['in_list']}/{without['n']} ({without['in_list'] / without['n']:.1%})")
    if escalation["n"]:
        n = escalation["n"]
        print(f"    escalation n={n}  recall={escalation['recall']}/{n} ({escalation['recall'] / n:.1%})")
    print(f"    latency    mean={latency['mean']} ms  p95={latency['p95']} ms  max={latency['max']} ms  "
          f"total={latency['total_s']} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workflow", required=True, help="Workflow folder path.")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file.")
    parser.add_argument("--env", required=True, help="The workflow env file (as `fastworkflow run` takes it).")
    parser.add_argument("--passwords", required=True, help="The passwords env file (holds JEV_API_KEY).")
    parser.add_argument("--out", required=True, help="Where to write the per-case verdicts.")
    parser.add_argument("--mode", choices=("matcher", "chain"), default="matcher",
                        help="matcher: the matcher alone. chain: the workflow's shipped head of the case's "
                             "context first, the matcher only when the head is not confident; the heads alone "
                             "and the matcher alone are scored beside it, on the same answers.")
    parser.add_argument("--baseline", default=None, help="A score_benchmark.py verdict file to compare with.")
    parser.add_argument("--path", action="append", help="Context=Context,Parent,... (repeatable).")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N cases.")
    parser.add_argument("--cache", action="store_true",
                        help="Let answers come from the response cache (latency is then not a measurement).")
    args = parser.parse_args(argv)

    from dotenv import dotenv_values

    env_vars = {**dotenv_values(args.env), **dotenv_values(args.passwords)}
    if not env_vars.get("JEV_API_KEY"):
        print(f"JEV_API_KEY is not set in {args.env} or {args.passwords}: the matcher cannot be asked.",
              file=sys.stderr)
        return 1
    if not args.cache:
        env_vars["FW_LM_CACHE"] = "0"

    import time

    import fastworkflow

    fastworkflow.init(env_vars=env_vars)
    from fastworkflow import intent_detector, workflow_agent
    from fastworkflow.train import heldout_evaluation

    core = workflow_agent.core_command_names()
    cases = heldout_evaluation.load_benchmark_file(args.benchmark)[: args.limit]
    paths = _paths(args)
    workflow = str(Path(args.workflow).resolve())
    heads: dict = {}
    if args.mode == "chain":
        # The shipped heads, read where they are (nothing is written there), each asked once before timing
        # so that no case pays for loading a model.
        for context in sorted({case.context for case in cases}):
            head = intent_detector.engine_for(workflow, context, allow_matcher=False)
            if head is None:
                print(f"context {context!r} has no trained head under {workflow}: the chain needs one.", file=sys.stderr)
                return 1
            head.decide("warm up")
            heads[context] = head

    arms: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    inherited_wrong: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="matcher_score_") as scratch:
        folder = str(_copy_without_artifacts(Path(workflow), Path(scratch)))
        option_counts = {
            context: len({c.name for name in paths.get(context, [context])
                          for c in intent_detector.matcher_candidates(folder, name)}) + 1
            for context in sorted({case.context for case in cases})
        }
        for number, case in enumerate(cases, 1):
            path = paths.get(case.context, [case.context])
            base = {
                "key": f"{case.context}|{case.kind}|{heldout_evaluation.normalize_utterance(case.utterance)}",
                "context": case.context, "kind": case.kind, "utterance": case.utterance,
            }
            answer = intent_detector.MatcherEngine(folder, case.context, path).decide(case.utterance)
            details = answer.details
            matcher = {
                **base, **_verdict(case, answer.kind, answer.labels, core, heldout_evaluation),
                "scope_path": path, "top_label": details.get("top_label"), "confidence": details.get("confidence"),
                "none_score": details.get("none_score"), "latency_ms": details.get("latency_ms"),
                "calls": details.get("calls"),
            }
            if "error" in details:
                matcher["error"] = details["error"]
                matcher["escalated"] = False if case.kind == "escalation" else matcher.get("escalated")
            arms["matcher"].append(matcher)
            if args.mode == "chain":
                started = time.perf_counter()
                head_answer = heads[case.context].decide(case.utterance)
                head_ms = round((time.perf_counter() - started) * 1000.0, 1)
                confident = intent_detector.head_is_confident(head_answer)
                head = {
                    **base, **_verdict(case, head_answer.kind, head_answer.labels, core, heldout_evaluation),
                    "confident": confident, "confidence": head_answer.details.get("confidence"),
                    "latency_ms": head_ms, "calls": 0,
                }
                arms["heads"].append(head)
                if confident:
                    chain = {**head, "decided_by": "classifier"}
                    wrong = (not head["top1"]) if case.kind == "routing" else True
                    if wrong:                      # a confident head answer the chain takes over as it is
                        inherited_wrong.append({k: head[k] for k in (
                            "context", "kind", "utterance", "predicted", "confidence") if k in head}
                            | {"expected": head.get("expected") or f"(escalate to {case.expected_ancestor_command})"})
                elif "error" in matcher:
                    # ChainEngine: when the matcher's call fails, the head's answer stands.
                    chain = {**head, "decided_by": "classifier", "matcher_error": matcher["error"],
                             "latency_ms": round(head_ms + float(matcher["latency_ms"] or 0.0), 1)}
                else:
                    chain = {**matcher, "decided_by": "matcher", "head_predicted": head["predicted"],
                             "latency_ms": round(head_ms + float(matcher["latency_ms"] or 0.0), 1)}
                arms["chain"].append(chain)
            if number % 20 == 0:
                print(f"  {number}/{len(cases)} cases asked", file=sys.stderr, flush=True)

    scored = [arm for arm in ARMS if arms[arm]]
    summaries = {arm: _summary(arms[arm]) for arm in scored}
    comparisons = []
    by_key = {arm: {r["key"]: r for r in arms[arm]} for arm in scored}
    pairs = [("heads", "chain"), ("matcher", "chain"), ("heads", "matcher")] if args.mode == "chain" else []
    for before, after in pairs:
        for name, field in (("strict top-1", "top1"), ("first-ranked", "first_ranked"), ("in-list", "in_list"),
                            ("escalation recall", "escalated"), ("false escalation", "false_escalation")):
            axis = _axis(f"{after} vs {before}: {name}", field, by_key[before], by_key[after], before, after)
            if axis:
                comparisons.append(axis)
    main_arm = "chain" if args.mode == "chain" else "matcher"
    if args.baseline:
        baseline = {r["key"]: r for r in json.loads(Path(args.baseline).read_text(encoding="utf-8"))["cases"]}
        # A score_benchmark.py verdict file: its ``top1`` is first-ranked, so it is paired with first-ranked.
        for name, theirs, ours in (("first-ranked", "top1", "first_ranked"), ("in-list", "in_list", "in_list"),
                                   ("escalation recall", "escalated", "escalated")):
            renamed = {key: ({**r, ours: r[theirs]} if theirs in r else r) for key, r in baseline.items()}
            axis = _axis(f"{main_arm} vs baseline file: {name}", ours, renamed, by_key[main_arm],
                         "baseline", main_arm)
            if axis:
                comparisons.append(axis)

    summary = {**summaries[main_arm], "options_per_call": option_counts,
               "max_options_in_one_call": intent_detector.MATCHER_MAX_OPTIONS,
               "response_cache": bool(args.cache), "comparisons": comparisons}
    payload = {
        "workflow": str(args.workflow),
        "engine": main_arm,
        "benchmark": str(args.benchmark),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "cases": arms[main_arm],
    }
    if args.mode == "chain":
        payload["arms"] = {arm: {"summary": summaries[arm], "cases": arms[arm]} for arm in scored}
        payload["confident_head_answers_chain_inherits_wrong"] = inherited_wrong
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}: {len(cases)} case(s), mode {args.mode}")
    for arm in scored:
        _print_summary(arm, summaries[arm])
    print(f"  options per matcher call: {option_counts}")
    if args.mode == "chain":
        print(f"  confident head answers the chain inherits wrong: {len(inherited_wrong)}")
        for row in inherited_wrong:
            print(f"    [{row['context']}] {row['utterance']!r} -> {row['predicted']} (expected {row['expected']})")
    for axis in comparisons:
        names = [key for key in axis if key not in ("axis", "n", "mcnemar_p") and not key.endswith("_only")]
        print(f"  {axis['axis']:<42} n={axis['n']}  " + "  ".join(f"{name} {axis[name]}" for name in names)
              + "  discordant " + " / ".join(f"{key} {axis[key]}" for key in axis if key.endswith("_only"))
              + f"  exact McNemar p = {axis['mcnemar_p']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
