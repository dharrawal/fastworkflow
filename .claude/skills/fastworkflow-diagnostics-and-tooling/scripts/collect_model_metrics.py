#!/usr/bin/env python
"""collect_model_metrics.py — smoke-test a workflow's trained intent
classifiers on their stored seed utterances and report per-command accuracy,
ambiguity rate, and confusion pairs.

WHAT THIS MEASURES (be honest with yourself):
  The two-tier TinyBERT/DistilBERT pipeline is trained on seed
  (plain/template) utterances PLUS synthetic LLM-generated utterances that are
  NOT persisted anywhere after `fastworkflow train` finishes. Only the seeds
  survive, inside command_directory.json. So this script scores the router on
  a SUBSET OF ITS OWN TRAINING DATA — a health check, not generalization:
    * a healthy trained context scores near-perfect here; anything below ~0.9
      top-1 accuracy on seeds means broken/mismatched artifacts, label drift
      (commands changed since training), or a genuinely confused label pair
    * true held-out F1 is NOT cheaply computable offline; you would need to
      regenerate utterances via LLM_SYNDATA_GEN and hold them out. Gap
      documented, not papered over.

Per utterance it applies the SAME decision rule as runtime CommandRouter.predict
(fastworkflow/model_pipeline_training.py:321-334): predict with TinyBERT, fall
back to DistilBERT below threshold.json's confidence_threshold, then return a
single label only above the used model's ambiguous threshold, else top-k.

Usage:
    .venv/bin/python .claude/skills/fastworkflow-diagnostics-and-tooling/scripts/collect_model_metrics.py <workflow_dir> [--context CTX] [--json]

Requires a TRAINED workflow (___command_info/<ctx>/threshold.json etc.).
Loads models onto CPU/GPU read-only; expect ~10-60s. Never writes.
NEVER point experiment teardown at these model dirs (fix-0hb incident):
this script only reads them.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

GLOBAL_CONTEXT_FOLDER = "global"


def load_labeled_utterances(info_dir: Path, context_commands: list[str]) -> list[tuple[str, str]]:
    """(utterance, expected_command) pairs from the persisted seed utterances."""
    cmd_dir = json.loads((info_dir / "command_directory.json").read_text(encoding="utf-8"))
    utt_meta = cmd_dir.get("map_command_2_utterance_metadata", {})
    pairs = []
    for cmd in context_commands:
        um = utt_meta.get(cmd) or {}
        for utt in (um.get("plain_utterances") or []) + (um.get("template_utterances") or []):
            if isinstance(utt, str) and utt.strip():
                pairs.append((utt, cmd))
    return pairs


def evaluate_context(workflow_dir: Path, ctx_name: str, ctx_commands: list[str]) -> dict:
    # Import late: pulls in torch/transformers (slow).
    from fastworkflow.model_pipeline_training import (
        CommandRouter,
        predict_single_sentence,
    )

    info_dir = workflow_dir / "___command_info"
    ctx_folder = GLOBAL_CONTEXT_FOLDER if ctx_name == "*" else ctx_name
    model_dir = info_dir / ctx_folder
    if not (model_dir / "threshold.json").is_file():
        return {"context": ctx_name, "skipped": "no threshold.json (context untrained)"}

    router = CommandRouter(str(model_dir))
    pairs = load_labeled_utterances(info_dir, ctx_commands)
    if not pairs:
        return {"context": ctx_name, "skipped": "no stored seed utterances for its commands"}

    per_cmd: dict[str, Counter] = defaultdict(Counter)
    confusions: Counter = Counter()
    distil_used = 0
    confidences = []
    ambiguous_n = 0

    for utt, expected in pairs:
        r = predict_single_sentence(router.modelpipeline, utt, router.label_encoder_path)
        top1, conf, used_distil = r["label"], r["confidence"], r["used_distil"]
        topk = list(r["topk_labels"])
        distil_used += bool(used_distil)
        confidences.append(float(conf))

        # Same single-vs-ambiguous rule as CommandRouter.predict
        amb_threshold = (
            router.large_ambiguous_confidence_threshold
            if used_distil
            else router.tiny_ambiguous_confidence_threshold
        )
        is_single = conf > amb_threshold
        if not is_single:
            ambiguous_n += 1

        c = per_cmd[expected]
        c["n"] += 1
        if top1 == expected:
            c["top1_correct"] += 1
        else:
            confusions[(expected, top1)] += 1
        if is_single and top1 == expected:
            c["confident_correct"] += 1
        elif not is_single and expected in topk:
            c["ambiguous_but_in_topk"] += 1

    n = len(pairs)
    top1_total = sum(c["top1_correct"] for c in per_cmd.values())
    return {
        "context": ctx_name,
        "model_dir": str(model_dir),
        "n_utterances": n,
        "n_commands_with_utterances": len(per_cmd),
        "top1_accuracy": round(top1_total / n, 4),
        "ambiguous_rate": round(ambiguous_n / n, 4),
        "distil_fallback_rate": round(distil_used / n, 4),
        "mean_confidence": round(sum(confidences) / n, 4),
        "thresholds": {
            "confidence_threshold": router.confidence_threshold,
            "tiny_ambiguous": router.tiny_ambiguous_confidence_threshold,
            "large_ambiguous": router.large_ambiguous_confidence_threshold,
        },
        "per_command": {
            cmd: {
                "n": c["n"],
                "top1_correct": c["top1_correct"],
                "confident_correct": c["confident_correct"],
                "ambiguous_but_in_topk": c["ambiguous_but_in_topk"],
            }
            for cmd, c in sorted(per_cmd.items())
        },
        "confusion_pairs": [
            {"expected": e, "predicted": p, "count": cnt}
            for (e, p), cnt in confusions.most_common()
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow_dir", help="path to a TRAINED workflow folder")
    ap.add_argument("--context", help="evaluate only this routing context (e.g. '*' or 'User')")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    wf = Path(args.workflow_dir).resolve()
    info_dir = wf / "___command_info"
    routing_path = info_dir / "routing_definition.json"
    if not routing_path.is_file():
        print(f"ERROR: {routing_path} missing — build/train the workflow first", file=sys.stderr)
        return 1
    contexts = json.loads(routing_path.read_text(encoding="utf-8")).get("contexts", {})
    if args.context:
        if args.context not in contexts:
            print(f"ERROR: context {args.context!r} not in {sorted(contexts)}", file=sys.stderr)
            return 1
        contexts = {args.context: contexts[args.context]}

    results = [
        evaluate_context(wf, ctx, sorted(set(cmds)))
        for ctx, cmds in contexts.items()
    ]

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return 0

    print(f"Workflow: {wf}")
    print("Scores are on SEED utterances (training data) — health check, not held-out F1.\n")
    for r in results:
        if "skipped" in r:
            print(f"=== {r['context']}: SKIPPED — {r['skipped']}\n")
            continue
        print(f"=== {r['context']} ({r['n_utterances']} utterances, "
              f"{r['n_commands_with_utterances']} commands)")
        print(f"  top1_accuracy={r['top1_accuracy']}  ambiguous_rate={r['ambiguous_rate']}  "
              f"distil_fallback_rate={r['distil_fallback_rate']}  "
              f"mean_confidence={r['mean_confidence']}")
        th = r["thresholds"]
        print(f"  thresholds: conf={th['confidence_threshold']:.4f} "
              f"tiny_amb={th['tiny_ambiguous']:.4f} large_amb={th['large_ambiguous']:.4f}")
        for cmd, c in r["per_command"].items():
            print(f"    {cmd:45s} n={c['n']:<3d} top1={c['top1_correct']:<3d} "
                  f"confident={c['confident_correct']:<3d} amb_topk_hit={c['ambiguous_but_in_topk']}")
        if r["confusion_pairs"]:
            print("  confusions (expected -> predicted):")
            for cp in r["confusion_pairs"]:
                print(f"    {cp['expected']} -> {cp['predicted']}  x{cp['count']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
