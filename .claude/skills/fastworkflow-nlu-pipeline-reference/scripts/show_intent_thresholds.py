#!/usr/bin/env python3
"""Read-only inspector for a trained fastWorkflow workflow's intent-model artifacts.

Usage:
    python show_intent_thresholds.py <workflow_folderpath>

Prints, per command-context folder under <workflow>/___command_info/:
  - confidence_threshold        (threshold.json - tiny->distil escalation)
  - tiny_ambiguous_threshold    (tiny_ambiguous_threshold.json - single-label vs top-k)
  - large_ambiguous_threshold   (large_ambiguous_threshold.json)
  - label classes               (label_encoder.pkl, if sklearn is importable)
  - presence of model dirs      (tinymodel.pth/, largemodel.pth/)
  - stale files                 (legacy ambiguous_threshold.json is NOT read by CommandRouter)

Interpretation:
  - Missing threshold.json for a context == that context is untrained
    (is_workflow_trained() in fastworkflow/model_pipeline_training.py checks exactly this).
  - confidence_threshold is where TinyBERT hands off to DistilBERT.
  - *_ambiguous thresholds gate single-label return vs top-k clarification in
    CommandRouter.predict().

This script only reads files. It never writes or trains.
"""
import json
import os
import sys


def read_threshold(path):
    try:
        with open(path) as f:
            return json.load(f)["confidence_threshold"]
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError) as e:
        return f"<unreadable: {e}>"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    workflow = sys.argv[1]
    info_root = os.path.join(workflow, "___command_info")
    if not os.path.isdir(info_root):
        print(f"ERROR: {info_root} does not exist - workflow not built/trained.")
        sys.exit(2)

    context_dirs = sorted(
        d for d in os.listdir(info_root)
        if os.path.isdir(os.path.join(info_root, d)) and d != "__pycache__"
    )
    if not context_dirs:
        print(f"No context folders under {info_root} - workflow built but not trained.")
        sys.exit(0)

    for ctx in context_dirs:
        ctx_path = os.path.join(info_root, ctx)
        print(f"\n=== context: {ctx} ===")
        conf = read_threshold(os.path.join(ctx_path, "threshold.json"))
        tiny_amb = read_threshold(os.path.join(ctx_path, "tiny_ambiguous_threshold.json"))
        large_amb = read_threshold(os.path.join(ctx_path, "large_ambiguous_threshold.json"))
        if conf is None:
            print("  UNTRAINED: threshold.json missing (CommandRouter would crash here)")
            continue
        print(f"  confidence_threshold (tiny->distil escalation): {conf}")
        print(f"  tiny_ambiguous_threshold  (below => top-k):     {tiny_amb}")
        print(f"  large_ambiguous_threshold (below => top-k):     {large_amb}")

        for model_dir in ("tinymodel.pth", "largemodel.pth"):
            present = os.path.isdir(os.path.join(ctx_path, model_dir))
            print(f"  {model_dir}/: {'present' if present else 'MISSING'}")

        legacy = os.path.join(ctx_path, "ambiguous_threshold.json")
        if os.path.exists(legacy):
            print("  NOTE: legacy ambiguous_threshold.json present - dead file, not read by CommandRouter")

        le_path = os.path.join(ctx_path, "label_encoder.pkl")
        if os.path.exists(le_path):
            try:
                import pickle
                with open(le_path, "rb") as f:
                    le = pickle.load(f)
                classes = list(le.classes_)
                print(f"  labels ({len(classes)}): {classes}")
            except Exception as e:  # noqa: BLE001 - diagnostic tool, report and move on
                print(f"  label_encoder.pkl present but unloadable here: {e}")
        else:
            print("  label_encoder.pkl: MISSING")


if __name__ == "__main__":
    main()
