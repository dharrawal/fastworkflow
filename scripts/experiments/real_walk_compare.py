#!/usr/bin/env python3
"""Compare ``remote_link_worth.py``'s offline walk with the real walk recorded by ``real_walk_check.py``.

    real_walk_compare.py --answers answers.jsonl --pair cases_a.json=real_a.json [--pair ...] [--family examples=a,b]

Rows (per set and family; routing / escalation / pooled right R, wrong W, question with the right command among the
candidates Q+, question without it Q-; a bare value counts as a wrong answer, as in ``remote_link_worth.py``):

    sim A       the simulation, local alone                      (``remote_link_worth.evaluate(cfg=None)``)
    real A (d)  ``intent_detector.detect`` with its default engines  (``detect_local``)
    real A (p)  ``intent_loop.resolve``, no key                     (``decide_local``: the product's part decision)
    sim B       the simulation, chain 0.5/0.5, stored answers
    real B (p)  ``intent_loop.resolve`` with the stored answers replayed through the real matcher engine (``decide_chain``)

and the case-by-case mismatches between the simulation and each real run, by kind.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote_link_worth as R  # noqa: E402


def is_expected(case, command):
    return R.is_expected(case, command)


def sim_sig(det):
    if det[0] == "match":
        return ("match", det[1])
    if det[0] == "ambiguous":
        return ("ambiguous", tuple(det[1]))
    return (det[0],)


def real_detect_sig(d):
    if d["outcome"] == "match":
        return ("match", d["command"])
    if d["outcome"] == "ambiguous":
        return ("ambiguous", tuple(d["candidates"]))
    return ("bare",) if d["bare"] else ("none",)


def real_decide_sig(d):
    if d["outcome"] == "match":
        return ("match", d["command"])
    if d["outcome"] == "ambiguous":
        return ("ambiguous", tuple(d["ambiguous"]))
    return ("bare",) if d["bare"] else ("none",)


def klass(case, sig):
    if sig[0] == "match":
        return "right" if is_expected(case, sig[1]) else "wrong"
    if sig[0] == "bare":
        return "wrong"
    if sig[0] == "ambiguous":
        return "q_right" if any(is_expected(case, c) for c in sig[1]) else "q_none"
    return "q_none"


def tally(pairs):
    out = {n: collections.Counter() for n in ("routing", "escalation", "pooled")}
    for case, cls in pairs:
        for n in (case["kind"], "pooled"):
            out[n]["n"] += 1
            out[n][cls] += 1
    return out


def fmt(label, t):
    cols = []
    for n in ("routing", "escalation", "pooled"):
        c = t[n]
        cols.append(f"{n[:4]} n={c['n']:<4} R {c['right']:<4} W {c['wrong']:<3} Q+ {c['q_right']:<4} Q- {c['q_none']:<4}")
    return f"{label:<12}" + " | ".join(cols)


def kind_of_change(a, b):
    if a == b:
        return "same"
    if a[0] != b[0]:
        return f"{a[0]} -> {b[0]}"
    if a[0] == "match":
        return "match: other command"
    if a[0] == "ambiguous":
        return "ambiguous: other candidate list"
    return "same"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--pair", action="append", required=True, help="cases.json=real.json")
    ap.add_argument("--family", action="append", default=[], help="name=case_name,case_name (names in the cases headers)")
    ap.add_argument("--out", help="write the per-case comparison JSON here")
    ap.add_argument("--show", type=int, default=6, help="mismatched cases to print per kind")
    args = ap.parse_args()

    files = [p.split("=", 1)[0] for p in args.pair]
    data = R.Data(files, args.answers)
    real = {}
    for p in args.pair:
        j = json.load(open(p.split("=", 1)[1], encoding="utf-8"))
        for r in j["rows"]:
            real[r["id"]] = r
    fams: dict[str, list[str]] = {}
    for item in args.family:
        n, v = item.split("=", 1)
        fams[n] = v.split(",")
    if not fams:
        fams = {"all": sorted({c["workflow"] for c in data.cases})}
    cfg = R.make_cfg("plain", 0.5, 0.5)
    percase = []
    for case in data.cases:
        r = real[case["id"]]
        s_a = R.evaluate(data, case, None)
        s_b = R.evaluate(data, case, cfg)
        percase.append({
            "case": case, "real": r,
            "simA": (s_a[0], sim_sig(s_a[2])), "simB": (s_b[0], sim_sig(s_b[2])),
            "realAd": real_detect_sig(r["detect_local"]), "realAp": real_decide_sig(r["decide_local"]),
            "realBp": real_decide_sig(r["decide_chain"]) if "decide_chain" in r else None,
        })

    # -- local per-scope answers: stored (simulation input) vs the real engines'
    mism_local = collections.Counter()
    checked_local = 0
    for pc in percase:
        for scope, st in pc["case"]["local"].items():
            re_ = pc["real"]["per_scope"].get(scope)
            checked_local += 1
            if re_ is None or re_["kind"] != st["kind"] or re_["labels"] != st["labels"] or bool(re_["confident"]) != bool(st["confident"]):
                mism_local[scope == pc["case"]["context"]] += 1
    print(f"stored per-scope local answers compared with the real engines: {checked_local}, mismatches {sum(mism_local.values())}")
    print(f"real scope path != stored path: {sum(1 for pc in percase if pc['real']['real_path'] != [s for s in pc['case']['path']] and pc['real']['real_path'] != pc['case']['path'][:len(pc['real']['real_path'])])}"
          " (cases where the real walk path is not a prefix of the stored path)")

    for fam, names in fams.items():
        for set_name in ("dev", "conf"):
            rows = [pc for pc in percase if pc["case"]["workflow"] in names and pc["case"]["set"] == set_name]
            if not rows:
                continue
            print(f"\n=== family {fam}, set {set_name}: {len(rows)} cases")
            print(fmt("sim A", tally((pc["case"], pc["simA"][0]) for pc in rows)))
            print(fmt("real A (d)", tally((pc["case"], klass(pc["case"], pc["realAd"])) for pc in rows)))
            print(fmt("real A (p)", tally((pc["case"], klass(pc["case"], pc["realAp"])) for pc in rows)))
            print(fmt("sim B", tally((pc["case"], pc["simB"][0]) for pc in rows)))
            if all(pc["realBp"] for pc in rows):
                print(fmt("real B (p)", tally((pc["case"], klass(pc["case"], pc["realBp"])) for pc in rows)))
            for label, key_sim, key_real in (("sim A vs real A (d)", "simA", "realAd"), ("sim A vs real A (p)", "simA", "realAp"),
                                              ("sim B vs real B (p)", "simB", "realBp")):
                kinds = collections.Counter()
                examples = collections.defaultdict(list)
                for pc in rows:
                    if pc[key_real] is None:
                        continue
                    sim = pc[key_sim][1]
                    kind = kind_of_change(sim, pc[key_real])
                    c_sim, c_real = pc[key_sim][0], klass(pc["case"], pc[key_real])
                    cc = "" if c_sim == c_real else f" [class {c_sim} -> {c_real}]"
                    kinds[kind + cc] += 1
                    if kind != "same":
                        examples[kind + cc].append(pc)
                diff = sum(v for k, v in kinds.items() if not k.startswith("same"))
                print(f"  {label}: {diff} of {len(rows)} cases differ in outcome signature")
                for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
                    if not k.startswith("same"):
                        print(f"      {v:>4}  {k}")
                        for pc in examples[k][: args.show]:
                            rr = pc["real"]
                            print(f"            ctx={pc['case']['context']:<22} exp={pc['case']['expected']:<34} sim={pc[key_sim][1]} real={pc[key_real]} "
                                  f"asked_d={rr['detect_local']['asked']}")
    if args.out:
        Path(args.out).write_text(json.dumps([{k: v for k, v in pc.items() if k not in ("case", "real")} | {"id": pc["case"]["id"]}
                                               for pc in percase], default=list), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
