#!/usr/bin/env python3
"""Offline effect of walk rules on "local alone", recomputed from the REAL engines' answers (``real_walk_check.py``).

    real_walk_rules.py --real real_a.json=chain-end --real real_b.json=dummy [--family examples=lib,retail --family large=large]

The walk is ``intent_detector.detect``'s, re-implemented over the recorded per-scope answers (``per_scope``): the global
scope first (a confident application command wins), else the case's own scope; an ambiguity or a core command from it
ends the walk; otherwise the climb asks the enclosing scopes (and the global scope again when the application's
``get_parent`` ends in None, ``=dummy``; with ``=chain-end`` the root context is the last scope and the global scope is
asked only first), until a match or a bare value. With NO rule this must reproduce ``detect_local`` on every case (it
is checked; the script stops otherwise).

A rule says when an answer of the GLOBAL scope is accepted for a case whose own context is not the global scope. A global
answer a rule refuses is treated as the head's "not confident" answer for that scope: an ambiguity whose candidates are
the head's top inside the scope (approximation: the hand-on commands of a refused answer were not recorded, so a refused
answer can lose a right candidate; "Q+ may be understated for rules that refuse").

The rules are fixed in ``RULES`` below, defined from the development half's breakdown before any confirmation number was
computed (see the report); thresholds compare the global answer's ``p_in`` (the probability of its best command).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

GLOBAL = "*"
OWN_PLAUSIBLE = 0.10

RULES = {
    "R0 none (current)":                      lambda g, own: True,
    "R1 global-first p>=0.80":                lambda g, own: g["p_in"] >= 0.80,
    "R2 global-first p>=0.90":                lambda g, own: g["p_in"] >= 0.90,
    "R3 own head's best p_in<=0.10":          lambda g, own: (own or {}).get("p_in") is None or own["p_in"] <= OWN_PLAUSIBLE,
    "R4 R1 and R3":                           lambda g, own: g["p_in"] >= 0.80 and ((own or {}).get("p_in") is None or own["p_in"] <= OWN_PLAUSIBLE),
    "R5 global may not answer outside own":   lambda g, own: False,
}


def is_expected(row, command):
    if row["kind"] == "routing":
        return command == row["expected"]
    return command == row["expected"] or command.split("/")[-1] == row["expected"].split("/")[-1]


def offline_walk(row, core, climb_global, rule):
    ctx = row["context"]
    ps = row["per_scope"]
    path = [s for s in row["path"] if s != GLOBAL]
    gf = ctx != GLOBAL and row["detect_local"]["asked"][0][0] == GLOBAL

    def answer(scope):
        a = ps.get(scope)
        if a is None:
            return {"kind": "not_mine", "labels": []}
        if scope == GLOBAL and ctx != GLOBAL and a["kind"] == "match" and a["labels"][0] not in core \
                and not rule(a, ps.get(ctx)):
            cands = [l for l, _ in a["topk"]]
            return {"kind": "ambiguous", "labels": cands}
        return a

    if gf:
        g = answer(GLOBAL)
        if g["kind"] == "match" and g["labels"][0] not in core:
            return ("match", g["labels"][0])
    scopes = path if ctx != GLOBAL else [GLOBAL]
    if ctx != GLOBAL and climb_global:
        scopes = [*scopes, GLOBAL]
    a = answer(scopes[0])
    is_core = a["kind"] == "match" and a["labels"][0] in core
    if a["kind"] != "ambiguous" and not is_core:
        i = 1
        while a["kind"] not in ("match", "bare_value") and i < len(scopes):
            a = answer(scopes[i])
            i += 1
    if a["kind"] == "match":
        return ("match", a["labels"][0])
    if a["kind"] == "ambiguous":
        reserved = ("wildcard", "parameter_value")
        return ("ambiguous", tuple(l for l in a["labels"] if l not in reserved and "escalation" not in l))
    if a["kind"] == "bare_value":
        return ("bare",)
    return ("none",)


def real_sig(d):
    if d["outcome"] == "match":
        return ("match", d["command"])
    if d["outcome"] == "ambiguous":
        return ("ambiguous", tuple(d["candidates"]))
    return ("bare",) if d["bare"] else ("none",)


def klass(row, sig):
    if sig[0] == "match":
        return "right" if is_expected(row, sig[1]) else "wrong"
    if sig[0] == "bare":
        return "wrong"
    if sig[0] == "ambiguous":
        return "q_right" if any(is_expected(row, c) for c in sig[1]) else "q_none"
    return "q_none"


def tally(rows, rule, cores, climbs):
    out = {n: collections.Counter() for n in ("routing", "escalation", "pooled")}
    for r in rows:
        sig = offline_walk(r, cores[r["_file"]], climbs[r["_file"]], rule)
        c = klass(r, sig)
        for n in (r["kind"], "pooled"):
            out[n]["n"] += 1
            out[n][c] += 1
    return out


def fmt(label, t):
    cols = []
    for n in ("routing", "escalation", "pooled"):
        c = t[n]
        cols.append(f"{n[:4]} n={c['n']:<4} R {c['right']:<4} W {c['wrong']:<3} Q+ {c['q_right']:<4} Q- {c['q_none']:<4}")
    return f"{label:<38}" + " | ".join(cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", action="append", required=True, help="real.json=chain-end|dummy")
    ap.add_argument("--family", action="append", default=[], help="name=case_name,case_name")
    ap.add_argument("--set", choices=("dev", "conf"), help="only this set")
    args = ap.parse_args()
    rows, cores, climbs = [], {}, {}
    for item in args.real:
        path, mode = item.rsplit("=", 1)
        j = json.load(open(path, encoding="utf-8"))
        cores[path] = set(j["header"]["core"])
        climbs[path] = mode == "dummy"
        for r in j["rows"]:
            r["_file"], r["_name"] = path, j["header"]["name"]
            rows.append(r)
    fams = {}
    for item in args.family:
        n, v = item.split("=", 1)
        fams[n] = v.split(",")
    if not fams:
        fams = {"all": sorted({r["_name"] for r in rows})}
    # the walk with no rule must be the real detect
    bad = 0
    for r in rows:
        sig = offline_walk(r, cores[r["_file"]], climbs[r["_file"]], RULES["R0 none (current)"])
        if sig != real_sig(r["detect_local"]):
            bad += 1
            if bad <= 5:
                print("OFFLINE WALK != REAL detect:", r["id"], sig, real_sig(r["detect_local"]))
    print(f"offline walk with no rule vs real intent_detector.detect: {bad} mismatches of {len(rows)} cases")
    if bad:
        return 1
    for fam, names in fams.items():
        for set_name in ("dev", "conf"):
            if args.set and set_name != args.set:
                continue
            sub = [r for r in rows if r["_name"] in names and r["set"] == set_name]
            if not sub:
                continue
            print(f"\n=== {fam}, {set_name}: {len(sub)} cases")
            for label, rule in RULES.items():
                print(fmt(label, tally(sub, rule, cores, climbs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
