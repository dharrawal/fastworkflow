#!/usr/bin/env python3
"""Lint an experiment document against the Appendix G standard.

Appendix G of docs/tau2_retail_reliability_implementation_plan.md requires every
docs/experiments/EXX_<name>.md to contain, in order:
  (1) pre-registration (hypothesis, config diff, metrics, analysis plan, date,
      committed BEFORE the run)
  (2) baseline numbers with CIs
  (3) results with CIs + McNemar vs. predecessor
  (4) failure modes before/after (bucket table)
  (5) surprises / negative results
  (6) links to run logs + traces
  (7) bd issue ID
  (8) reproduction command line

This linter is heuristic (heading/keyword matching), read-only, and has zero
dependencies. Exit code 0 = all sections found; 1 = missing sections; 2 = usage
or file error. A pass here does NOT certify the science — only that the paper
trail has all eight parts. Rule: "if it isn't written here with a number and an
interval, it didn't happen."

Usage:
    python3 lint_experiment_doc.py docs/experiments/E3_invariants.md
"""

import re
import sys

# (section number, human name, list of regexes — ANY match counts, case-insensitive)
SECTIONS = [
    (1, "pre-registration", [r"pre-?registration", r"registered\s+before"]),
    (1, "hypothesis (inside pre-registration)", [r"hypothesis"]),
    (2, "baseline numbers with CIs", [r"baseline"]),
    (3, "results with CIs + McNemar", [r"\bresults?\b"]),
    (3, "McNemar paired test", [r"mcnemar"]),
    (3, "confidence intervals", [r"confidence interval", r"\bCI\b", r"\bCIs\b",
                                 r"clopper-?pearson", r"wilson"]),
    (4, "failure modes before/after", [r"failure modes?", r"before\s*/?\s*after",
                                       r"before\s*(→|->)\s*after"]),
    (5, "surprises / negative results", [r"surprise", r"negative results?"]),
    (6, "links to run logs + traces", [r"run logs?", r"traces?\b"]),
    (7, "bd issue ID", [r"\bbd[- ]?issue\b", r"\b(fix|fastworkflow)-[a-z0-9]{2,4}\b"]),
    (8, "reproduction command line", [r"reproduc", r"repro command"]),
]


def lint(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        return 2

    missing = []
    for num, name, patterns in SECTIONS:
        if not any(re.search(p, text, re.IGNORECASE) for p in patterns):
            missing.append((num, name))

    # Numbers-with-intervals sniff test: at least one "x% [a, b]"-ish or "x/15"-ish figure.
    has_numbers = bool(
        re.search(r"\d+\s*/\s*\d+", text) or re.search(r"\d+(\.\d+)?\s*%", text)
    )

    if missing:
        print(f"FAIL: {path} is missing {len(missing)} Appendix G element(s):")
        for num, name in missing:
            print(f"  - section ({num}): {name}")
    if not has_numbers:
        print("WARN: no proportions found (e.g. '12/15' or '80%'). "
              "\"If it isn't written here with a number and an interval, it didn't happen.\"")
    if not missing:
        print(f"OK: {path} contains all 8 Appendix G elements (heuristic check). "
              "Now verify the pre-registration commit predates the run.")
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(lint(sys.argv[1]))
