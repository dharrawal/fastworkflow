#!/usr/bin/env python3
"""Which scope gave each wrong (or right) confident answer of the REAL local-only walk (``real_walk_check.py`` output).

    real_walk_breakdown.py --real real_a.json [--real real_b.json ...] [--set dev|conf] [--listing out.txt]

For ``intent_detector.detect`` (the walk the simulation re-implements, ``detect_local``) it classifies the answering
scope of each case that ended in a command or a bare value:

    own          the case's own context answered
    global-first the global scope, asked first, answered (a confident application command)
    enclosing    an enclosing scope (not global) answered in the climb
    global-climb the global scope answered in the climb (after the own and enclosing scopes declined)
    bare         the walk ended on a bare value (counted as a wrong answer, as in ``remote_link_worth.py``)

and prints, per scope kind and outcome (right / wrong): counts, the deciding scope's top probability (``p_in``),
and, for wrong answers, whether the case's OWN head alone would have been confidently wrong, confidently right or
not confident (what a score of each case at its own context alone sees). ``--listing`` writes one line per wrong
answer: scope, probability, expected, given, name overlap flag. Escalation cases that come out right are tabulated by
answering scope too.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys

GENERIC = {"list", "show", "open", "find", "get", "by", "of", "uid", "record", "records", "the", "a", "an", "and",
           "for", "to", "in", "from", "all", "view"}


def tokens(name: str) -> set[str]:
    return {t for t in name.split("/")[-1].lower().split("_") if t and t not in GENERIC}


def is_expected(row, command: str) -> bool:
    if row["kind"] == "routing":
        return command == row["expected"]
    return command == row["expected"] or command.split("/")[-1] == row["expected"].split("/")[-1]


def answering(row) -> tuple[str, str | None, float | None]:
    """(scope kind, scope name, p_in of the deciding scope) of the case's real ``detect`` walk."""
    d = row["detect_local"]
    asked = d["asked"]
    ctx = row["context"]
    if d["outcome"] == "match" or d["bare"]:
        scope = asked[-1][0]
        first_global = ctx != "*" and asked and asked[0][0] == "*" and len(asked) == 1 and d["outcome"] == "match"
        # the global scope asked first answers alone: one entry; (a case at the global scope asks it once)
        if ctx == "*":
            kind = "own"
        elif first_global:
            kind = "global-first"
        elif scope == ctx:
            kind = "own"
        elif scope == "*":
            kind = "global-climb"
        else:
            kind = "enclosing"
        if d["bare"]:
            kind_bare = kind
            kind = "bare"
            _ = kind_bare
        ps = row["per_scope"].get(scope, {})
        return kind, scope, ps.get("confidence") if d["bare"] else ps.get("p_in")
    return "none", None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", action="append", required=True)
    ap.add_argument("--set", choices=("dev", "conf"), help="only this set")
    ap.add_argument("--listing")
    args = ap.parse_args()
    rows = []
    for path in args.real:
        rows += json.load(open(path, encoding="utf-8"))["rows"]
    if args.set:
        rows = [r for r in rows if r["set"] == args.set]
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    wrong_lines = []
    esc_right = collections.Counter()
    for r in rows:
        d = r["detect_local"]
        kind, scope, p = answering(r)
        if kind == "none":
            continue
        if d["bare"]:
            outcome = "wrong"
            given = "<bare value>"
        else:
            given = d["command"]
            outcome = "right" if is_expected(r, given) else "wrong"
        own = r["per_scope"].get(r["context"], {})
        if own.get("kind") == "match" and own.get("confident"):
            own_view = "own head confident " + ("right" if is_expected(r, own["labels"][0]) else "wrong")
        elif own.get("kind") == "bare_value":
            own_view = "own head says bare value"
        else:
            own_view = "own head not confident"
        table[(kind, outcome)]["n"].append(1)
        table[(kind, outcome)]["p"].append(p if p is not None else 0.0)
        table[(kind, outcome)][own_view].append(1)
        if r["kind"] == "escalation" and outcome == "right":
            esc_right[kind] += 1
        if outcome == "wrong":
            overlap = bool(tokens(given) & tokens(r["expected"])) if not d["bare"] else False
            planning_like = "/" not in (given or "")
            wrong_lines.append((kind, scope, p, r, given, overlap, planning_like, own_view))
    print(f"{'scope kind':<13}{'outcome':<7}{'n':>4}  {'p min/median/max':<22}  own-head view")
    for (kind, outcome), v in sorted(table.items()):
        ps = v["p"]
        views = {k: len(x) for k, x in v.items() if k not in ("n", "p")}
        print(f"{kind:<13}{outcome:<7}{len(v['n']):>4}  {min(ps):.2f} / {statistics.median(ps):.2f} / {max(ps):.2f}       {views}")
    wrong = [w for w in wrong_lines]
    print(f"\nwrong answers: {len(wrong)}; routing {sum(1 for w in wrong if w[3]['kind']=='routing')}, "
          f"escalation {sum(1 for w in wrong if w[3]['kind']=='escalation')}")
    by = collections.Counter((w[0], w[3]["kind"]) for w in wrong)
    for k, v in sorted(by.items()):
        print(f"   {k[0]:<13}{k[1]:<11}{v}")
    print(f"  of them: given command is a global-scope-style name (no '/'): {sum(1 for w in wrong if w[6])}; "
          f"name tokens overlap expected: {sum(1 for w in wrong if w[5])}")
    print("escalation cases RIGHT by answering scope:", dict(esc_right))
    if args.listing:
        with open(args.listing, "w", encoding="utf-8") as f:
            for kind, scope, p, r, given, overlap, planning_like, own_view in sorted(wrong_lines, key=lambda w: (w[0], -(w[2] or 0))):
                f.write(f"{r['set']:<5}{kind:<13}scope={scope:<24}p={p if p is None else round(p, 3)!s:<6}ctx={r['context']:<22}"
                        f"exp={r['expected']:<44}got={given:<40}overlap={overlap!s:<6}noslash={planning_like!s:<6}{own_view}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
