#!/usr/bin/env python3
"""Learning curve for "the local intent model learns from corrections made during use". No model call.

Question: if every request the local head did not answer right outright were added to its training rows
(with the command that was finally right), how much better would it get on requests it has NOT seen?

The local model is the ``embedding`` head: a frozen sentence encoder plus one logistic-regression head over
every label of the workflow, read through each context's mask. The fit, the cross-fitted calibration and the
decision are the product's own functions (``fastworkflow/train/embedding_head.py``: ``fit_head``,
``cross_fitted_probabilities``, ``calibration_records``, ``calibrate``, ``decide``); the encoder is the
product's ``SentenceEncoder``. Vectors are cached on disk, so a refit is only the head fit and the calibration.

Simulation (one run = one workflow, a "seen in use" set and a "test" set of benchmark cases):
  * the seen requests are taken in a shuffled order (three shuffles); the head answers each one; when it does
    not name the right command outright (wrong, unsure, or passing up wrongly; for an escalation request, not
    passing up) the request and the command that was finally right become a learned row;
  * learned rows are ADDED to the training rows and a background worker refits the head every ``--batch`` (25)
    corrections, calibration included; the stream goes on with the refitted head;
  * after 0, 25, 50, 100, 200 corrections and at the end ("all"), the head is scored on the TEST set.
Variants at "all": every seen request added, learned rows counted x1/x3/x5 (repeated rows), the label spread of
the learned rows, the runtime similarity cache (cosine >= 0.85 to a stored corrected request answers directly,
the since-removed ``fastworkflow/cache_matching.py``) alone and with the refit, and a share of learned rows given a wrong label
(another command of the same context).
Disjointness: the training rows are the workflow's generated utterances, the benchmarks were written apart from
them, and the seen/test cases come from different benchmark files (examples) or a hash split of one file
(large workflow). Test requests whose text equals a training row or a seen request are dropped and counted;
the share of test requests with a seen request at cosine >= 0.85 is reported.

Reproduce the example numbers (D from ``embedding_commit_rule.py dump``; see ``local_head_levers.py``)::

    B=tests/fixtures/intent_benchmarks
    python scripts/experiments/embedding_commit_rule.py dump --workflow fastworkflow/examples/library --out D/library
    python scripts/experiments/embedding_commit_rule.py dump --workflow fastworkflow/examples/retail_workflow --out D/retail_workflow
    python scripts/experiments/learn_from_corrections.py --out OUT \\
        --wf library=D/library --wf retail_workflow=D/retail_workflow \\
        --run "library v1->v3=library|seen=$B/library.json|test=$B/library_v3.json" \\
        --run "library v1+v2->v3=library|seen=$B/library.json,$B/library_v2.json|test=$B/library_v3.json" \\
        --run "retail v1->v3=retail_workflow|seen=$B/retail_workflow.json|test=$B/retail_workflow_v3.json" \\
        --run "retail v1+v2->v3=retail_workflow|seen=$B/retail_workflow.json,$B/retail_workflow_v2.json|test=$B/retail_workflow_v3.json"

A large workflow is passed the same way with its development cases split in half by ``split=FILE``
(sorted by SHA-256 of ``context\\nutterance``: the first half is "seen in use", the second "test")::

    --wf large=DIR --run "large=large|split=PATH/dev_half.json"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastworkflow.train import embedding_head as eh  # noqa: E402
import head_rule_as_experimented as legacy  # noqa: E402  - the calibration and decision the head had (fix-o5wi.9)

CHECKPOINTS = (0, 25, 50, 100, 200)
CACHE_COSINE = 0.85
# The product no longer calibrates a threshold or a pass-up margin (bd fix-o5wi.9), and its default encoder is no
# longer bge-small: these targets, the calibration and decision (`head_rule_as_experimented`) and the encoder
# (`legacy.LEGACY_ENCODER`) are the head as it was when `docs/learning_from_corrections.md` was measured.
WRONG_TARGET = 0.01
ESC_TARGET = 0.005
RESERVED = eh.RESERVED_LABELS


def norm_name(label: str) -> str:
    return label.split("/")[-1]


# --------------------------------------------------------------------------- workflow data


class Workflow:
    def __init__(self, name: str, folder: str, cache: str):
        raw = json.load(open(os.path.join(folder, "rows.json"), encoding="utf-8"))
        self.name = name
        self.rows = [(r[0], r[1]) for r in raw["rows"]]
        self.labels = sorted({l for _t, l in self.rows})
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.masks = {s: list(m) for s, m in raw["masks"].items()}
        self.idx = {s: eh.mask_indices(self.labels, m) for s, m in self.masks.items()}
        self.texts = [t for t, _l in self.rows]
        self.y = np.array([self.index[l] for _t, l in self.rows])
        self.cache = cache
        self.enc = None
        self.X = self.embed(self.texts)

    def embed(self, texts):
        texts = list(texts)
        key = hashlib.sha1("\n".join([legacy.LEGACY_ENCODER, *texts]).encode()).hexdigest()
        path = os.path.join(self.cache, f"emb_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        if self.enc is None:
            self.enc = legacy.legacy_encoder()
        mat = np.asarray(self.enc.embed(texts), dtype=np.float64)
        os.makedirs(self.cache, exist_ok=True)
        np.save(path, mat)
        return mat

    def resolve_ancestor(self, case) -> int | None:
        anc = case.get("expected_ancestor_command") or ""
        if anc in self.index:
            return self.index[anc]
        inside = {self.labels[i] for i in self.idx[case["context"]][0]}
        hits = [l for l in self.labels if norm_name(l) == norm_name(anc) and l not in inside and l not in RESERVED]
        return self.index[hits[0]] if len(hits) == 1 else None

    def keep(self, case) -> bool:
        """The dump's filter: the context exists and the expected command is in it (routing), or the ancestor
        exists and is not in the context (escalation)."""
        if case["context"] not in self.masks:
            return False
        inside = {norm_name(m) for m in self.masks[case["context"]]}
        if case["kind"] == "routing":
            return norm_name(case["expected_label"]) in inside
        anc = norm_name(case.get("expected_ancestor_command") or "")
        return anc not in inside and anc in {norm_name(l) for l in self.labels}


def load_cases(paths):
    out = []
    for p in paths:
        raw = json.load(open(p, encoding="utf-8"))
        out += raw["cases"] if isinstance(raw, dict) else raw
    return out


class Cases:
    """Cases with their embeddings and expected label ids (routing: the command; escalation: the ancestor)."""

    def __init__(self, W: Workflow, cases):
        self.cases = [c for c in cases if W.keep(c)]
        self.Q = W.embed([c["utterance"] for c in self.cases]) if self.cases else np.zeros((0, W.X.shape[1]))
        self.ctx = [c["context"] for c in self.cases]
        self.esc = np.array([c["kind"] == "escalation" for c in self.cases], dtype=bool)
        targets = [W.resolve_ancestor(c) if c["kind"] == "escalation" else W.index[c["expected_label"]]
                   for c in self.cases]
        self.target = np.array([-1 if t is None else t for t in targets], dtype=int)

    def subset(self, sel):
        o = object.__new__(Cases)
        o.cases = [self.cases[i] for i in sel]
        o.Q, o.ctx, o.esc, o.target = self.Q[sel], [self.ctx[i] for i in sel], self.esc[sel], self.target[sel]
        return o


# --------------------------------------------------------------------------- fit and score


class Head:
    def __init__(self, W: Workflow, extra_X, extra_texts, extra_y, copies: int = 1):
        t0 = time.perf_counter()
        n_extra = len(extra_y)
        X, texts, y = W.X, list(W.texts), W.y
        if n_extra:
            reps = max(copies, 1)
            X = np.vstack([X] + [np.asarray(extra_X)] * reps)
            texts = texts + list(extra_texts) * reps
            y = np.concatenate([y] + [np.asarray(extra_y)] * reps)
        rows = [(t, W.labels[i]) for t, i in zip(texts, y)]
        t1 = time.perf_counter()
        self.coef, self.intercept = eh.fit_head(X, y, len(W.labels))
        t2 = time.perf_counter()
        cf, _facts = eh.cross_fitted_probabilities(X, y, texts, len(W.labels))
        records = legacy.calibration_records(cf, rows, W.labels, W.masks)
        cal = legacy.calibrate(records, WRONG_TARGET, ESC_TARGET)
        t3 = time.perf_counter()
        self.threshold, self.margin = cal["threshold"], cal["margin"]
        self.fit_s, self.cal_s = t2 - t1, t3 - t2
        self.W = W

    def probs(self, Q):
        return eh.softmax_probabilities(Q, self.coef, self.intercept)

    def decisions(self, C: Cases, P=None):
        """Per case: (decision, label id of the single answer or -1)."""
        P = self.probs(C.Q) if P is None else P
        out = []
        for p, ctx in zip(P, C.ctx):
            inside, outside = self.W.idx[ctx]
            d, ret, _info = legacy.decide(p, self.W.labels, inside, outside, self.threshold, self.margin)
            out.append((d, self.W.index[ret[0]] if len(ret) == 1 else -1))
        return out


def right_outright(head: Head, C: Cases, P=None):
    """Boolean per case: the request is answered right outright (routing: the one command; escalation: passed up)."""
    ok = []
    for (d, lab), esc, t in zip(head.decisions(C, P), C.esc, C.target):
        ok.append((d == eh.DECISION_NOT_MINE) if esc else (d == eh.DECISION_MATCH and lab == t))
    return np.array(ok, dtype=bool)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


class Store:
    """Stored corrected requests: vectors, label ids, normalised texts. ``hits(C, level)`` is the runtime rule:
    a request within cosine ``level`` of a stored one is answered with that one's label; ``level`` "exact" matches
    the normalised text (case, punctuation and spacing dropped) instead."""

    def __init__(self, X, labels, texts):
        self.X, self.labels = np.asarray(X), np.asarray(labels, dtype=int)
        self.norm = {}
        for t, l in zip(texts, labels):
            self.norm[normalise(t)] = int(l)

    def hits(self, C: "Cases", level):
        hit = np.zeros(len(C.cases), dtype=bool)
        lab = np.full(len(C.cases), -1)
        if not len(self.labels) or not len(C.cases):
            return hit, lab
        if level == "exact":
            for i, c in enumerate(C.cases):
                k = normalise(c["utterance"])
                if k in self.norm:
                    hit[i], lab[i] = True, self.norm[k]
            return hit, lab
        sims = C.Q @ self.X.T
        best = sims.argmax(axis=1)
        return sims[np.arange(len(best)), best] >= level, self.labels[best]


def wrong_at(conf, correct, share):
    n = len(conf)
    if not n:
        return 0.0
    order = np.argsort(-np.asarray(conf), kind="stable")
    take = int(np.ceil(share * n))
    return float((~np.asarray(correct, dtype=bool)[order[:take]]).sum() / n)


def score(head: Head, C: Cases, cache=None) -> dict:
    """The test metrics. ``cache`` = (stored vectors, stored label ids): a hit at cosine >= 0.85 answers directly."""
    W = head.W
    P = head.probs(C.Q)
    dec = head.decisions(C, P)
    hit = np.zeros(len(C.cases), dtype=bool)
    hit_label = np.full(len(C.cases), -1)
    if cache is not None:
        hit, hit_label = cache[0].hits(C, cache[1])
    first, rank, conf, correct, right, wrong = [], [], [], [], [], []
    esc_ok = []
    for i, (p, ctx) in enumerate(zip(P, C.ctx)):
        d, lab = dec[i]
        if C.esc[i]:
            esc_ok.append(bool(hit[i] and hit_label[i] == C.target[i]) if hit[i] else d == eh.DECISION_NOT_MINE)
            continue
        inside = W.idx[ctx][0]
        inside = inside[[W.labels[j] not in RESERVED for j in inside]]
        order = inside[np.argsort(-p[inside], kind="stable")]
        t = C.target[i]
        first.append(order[0] == t)
        rank.append(int(np.flatnonzero(order == t)[0]) + 1 if t in order else 99)
        best_all = int(np.argmax(p))
        conf.append(float(p[best_all]))
        correct.append(best_all == t)
        if hit[i]:
            ok = hit_label[i] == t
            right.append(ok); wrong.append(not ok)
        else:
            ok = d == eh.DECISION_MATCH and lab == t
            right.append(ok)
            wrong.append((d in (eh.DECISION_MATCH, eh.DECISION_NOT_MINE, eh.DECISION_BARE_VALUE)) and not ok)
    n = max(len(first), 1)
    rank = np.array(rank)
    return {
        "n": len(first), "n_esc": len(esc_ok),
        "first": float(np.mean(first)) if first else None,
        "r3": float((rank <= 3).mean()) if first else None,
        "wrong50": wrong_at(conf, correct, 0.5), "wrong70": wrong_at(conf, correct, 0.7),
        "right": sum(right) / n, "wrong": sum(wrong) / n, "handed": 1 - (sum(right) + sum(wrong)) / n,
        "esc": float(np.mean(esc_ok)) if esc_ok else None,
        "threshold": head.threshold, "margin": head.margin,
        "cache_hits": float(hit[~C.esc].mean()) if (~C.esc).any() and cache is not None else None,
        "cache_precision": (float(np.mean(hit_label[hit & ~C.esc] == C.target[hit & ~C.esc]))
                            if cache is not None and (hit & ~C.esc).any() else None),
    }


# --------------------------------------------------------------------------- simulation


def stream_run(W: Workflow, seen: Cases, test: Cases, order, batch: int):
    """The online simulation. Returns (checkpoints {label: metrics}, corrected list [(case index, label id)])."""
    head = Head(W, [], [], [])
    points = {0: {**score(head, test), "corrections": 0}}
    learned = []  # (seen index, label id)
    pending = 0
    for i in order:
        if seen.target[i] < 0:
            continue
        if right_outright(head, seen.subset([i]))[0]:
            continue
        learned.append((int(i), int(seen.target[i])))
        pending += 1
        if pending == batch:
            head = refit(W, seen, learned)
            pending = 0
            k = len(learned)
            if k in CHECKPOINTS:
                points[k] = {**score(head, test), "corrections": k}
    if pending or not learned:
        head = refit(W, seen, learned)
    points["all"] = {**score(head, test), "corrections": len(learned)}
    return points, learned


def refit(W, seen, learned, labels=None, copies=1):
    idx = [i for i, _l in learned]
    ys = [l for _i, l in learned] if labels is None else labels
    return Head(W, seen.Q[idx], [seen.cases[i]["utterance"] for i in idx], ys, copies)


def poisoned(W, seen, learned, share, rng):
    labels = [l for _i, l in learned]
    n = len(labels)
    for k in rng.choice(n, size=int(round(share * n)), replace=False):
        ctx = seen.ctx[learned[k][0]]
        inside = [j for j in W.idx[ctx][0] if W.labels[j] not in RESERVED and j != labels[k]]
        if inside:
            labels[k] = int(rng.choice(inside))
    return labels


def variants(W, seen, test, learned, seed):
    out = {}
    base = refit(W, seen, learned)
    def store(L, labs=None):
        return Store(seen.Q[[i for i, _l in L]], labs or [l for _i, l in L], [seen.cases[i]["utterance"] for i, _l in L])
    out["refit x1 (main)"] = score(base, test)
    all_idx = [(int(i), int(seen.target[i])) for i in range(len(seen.cases)) if seen.target[i] >= 0]
    out["(a) every seen request added"] = {**score(refit(W, seen, all_idx), test), "rows": len(all_idx)}
    for c in (3, 5):
        out[f"(b) learned rows x{c}"] = score(refit(W, seen, learned, copies=c), test)
    zero = Head(W, [], [], [])
    out["(d) cache alone (no refit)"] = score(zero, test, cache=(store(learned), CACHE_COSINE))
    out["(d) cache + refit"] = score(base, test, cache=(store(learned), CACHE_COSINE))
    for share in (0.05, 0.10):
        rng = np.random.default_rng(1000 + seed)
        labs = poisoned(W, seen, learned, share, rng)
        h = refit(W, seen, learned, labels=labs)
        out[f"(e) {int(share*100)}% wrong labels"] = score(h, test)
        if share == 0.10:
            out["(e) 10% wrong + cache"] = score(h, test, cache=(store(learned, labs), CACHE_COSINE))
    # (c) label spread
    cnt = Counter(l for _i, l in learned)
    total = max(sum(cnt.values()), 1)
    top = cnt.most_common(3)
    rows_per = np.bincount(W.y, minlength=len(W.labels))
    out["_spread"] = {
        "learned": total, "labels_hit": len(cnt), "labels": len(W.labels),
        "top3_share": sum(c for _l, c in top) / total,
        "top1_share": top[0][1] / total if top else 0.0,
        "top3_rows_in_training": [int(rows_per[l]) for l, _c in top],
    }
    return out


# --------------------------------------------------------------------------- recurring requests

_QUOTED = re.compile(r"\"[^\"]+\"|(?<![\w'])'[^']{1,80}'(?![\w'])|“[^”]+”")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_ID = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{3,}\b")
_NUMBER = re.compile(r"(?<![\w])\d[\d,./:-]*(?![\w])")
_NAME = re.compile(r"(?<!^)(?<![.!?] )\b(?!I['\u2019](?:m|d|ll|ve)\b)[A-Z][\w'-]*(?:[ \t]+(?!I['\u2019](?:m|d|ll|ve)\b)[A-Z][\w'-]*)*")
KINDS = (("quoted", _QUOTED), ("email", _EMAIL), ("id", _ID), ("number", _NUMBER), ("name", _NAME))


def values_of(text: str):
    """[(kind, start, end)] of non-overlapping concrete values, earlier kinds win."""
    taken, out = [], []
    for kind, rx in KINDS:
        for m in rx.finditer(text):
            if m.start() == 0 and kind == "name":
                continue
            if m.group(0) in ("I", "I'm", "I'd", "I'll"):
                continue
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append((kind, m.start(), m.end()))
    return sorted(out, key=lambda v: v[1])


def h(*parts) -> int:
    return int(hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def swap_values(text: str, label: str, pools: dict):
    """The request with each concrete value replaced by another of the same kind taken from other requests of
    the same command (training rows and seen requests). None when a value has no alternative."""
    vals = values_of(text)
    if not vals:
        return None
    out, pos = [], 0
    for kind, s, e in vals:
        cur = text[s:e]
        alts = [v for v in pools.get((label, kind), ()) if v != cur]
        if not alts:
            return None
        out.append(text[pos:s]); out.append(alts[h(text, cur) % len(alts)]); pos = e
    out.append(text[pos:])
    return "".join(out)


def surface_variants(text: str):
    """Four light variations: case, punctuation, a courtesy word, one typo (deterministic)."""
    low = text.lower() if text != text.lower() else text[:1].upper() + text[1:]
    stripped = re.sub(r"[?!.,;]", "", text)
    punct = stripped if stripped != text else text + "?"
    polite = re.sub(r"\b[Pp]lease,?\s*", "", text)
    polite = polite if polite != text else "please " + text[:1].lower() + text[1:]
    words = text.split(" ")
    long_ = [i for i, w in enumerate(words) if len(re.sub(r"\W", "", w)) >= 5]
    typo = text
    if long_:
        i = long_[h(text, "typo") % len(long_)]
        w = words[i]
        j = 1 + h(text, "j") % max(len(w) - 3, 1)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        typo = " ".join(words)
    return {"case": low, "punctuation": punct, "courtesy": polite, "typo": typo}


def recurrence_sets(W, seen):
    """Routing cases built from the seen requests: verbatim, values swapped, light surface variation."""
    pools: dict = {}
    pool_src = [(t, W.labels[y]) for t, y in zip(W.texts, W.y)] + [(c["utterance"], c["expected_label"]) for c in seen.cases if c["kind"] == "routing"]
    for t, lab in pool_src:
        for kind, s, e in values_of(t):
            pools.setdefault((lab, kind), set()).add(t[s:e])
    pools = {k: sorted(v) for k, v in pools.items()}
    verb, swapped, surf, examples = [], [], [], []
    for c in seen.cases:
        if c["kind"] != "routing":
            continue
        base = {"context": c["context"], "kind": "routing", "expected_label": c["expected_label"]}
        verb.append({**base, "utterance": c["utterance"]})
        sw = swap_values(c["utterance"], c["expected_label"], pools)
        if sw and sw != c["utterance"]:
            swapped.append({**base, "utterance": sw})
            if len(examples) < 3 and h(c["utterance"], "ex") % 3 == 0:
                examples.append((c["utterance"], sw))
        for v in surface_variants(c["utterance"]).values():
            if v != c["utterance"]:
                surf.append({**base, "utterance": v})
    return {"verbatim": verb, "values swapped": swapped, "surface variation": surf}, examples


LEVELS = (0.85, 0.90, 0.95, "exact")


def lvl(x):
    return "exact text" if x == "exact" else f"cosine {x:.2f}"


def recurrence(W, seen, test, learned, base, refit_head, sets):
    """{set: {mechanism: metrics}} for the unseen test set and the recurring sets."""
    st = Store(seen.Q[[i for i, _l in learned]], [l for _i, l in learned], [seen.cases[i]["utterance"] for i, _l in learned])
    out = {}
    allsets = {"unseen phrasings": test}
    for name, cases in sets.items():
        allsets[name] = Cases(W, cases)
    for name, C in allsets.items():
        r = {"no learning": score(base, C), "refit alone": score(refit_head, C)}
        for lv in LEVELS:
            r[f"cache alone, {lvl(lv)}"] = score(base, C, cache=(st, lv))
        for lv in LEVELS:
            r[f"refit + cache, {lvl(lv)}"] = score(refit_head, C, cache=(st, lv))
        out[name] = r
    return out


def near_miss(W, seen, test, learned):
    """How often a stored request captures a request of a DIFFERENT command, per level. ``corrected`` stores
    the learned rows (the real mechanism); ``every seen`` stores every seen request (the stress case)."""
    out = {}
    routing = ~test.esc
    sel = np.flatnonzero(routing)
    Q, tgt = test.Q[sel], test.target[sel]
    for name, L in (("corrected", learned),
                    ("every seen", [(i, int(seen.target[i])) for i in range(len(seen.cases)) if seen.target[i] >= 0 and not seen.esc[i]])):
        idx = [i for i, _l in L]
        lab = np.array([l for _i, l in L])
        sims = Q @ seen.Q[idx].T
        best = sims.argmax(axis=1)
        top = sims[np.arange(len(best)), best]
        diff = lab[None, :] != tgt[:, None]
        other = np.where(diff, sims, -1).max(axis=1)
        for lv in (0.85, 0.90, 0.95):
            captured = (top >= lv) & (lab[best] != tgt)
            hit = top >= lv
            out[f"{name} @ {lv:.2f}"] = {
                "hit": float(hit.mean()), "captured_wrong": float(captured.mean()),
                "wrong_share_of_hits": float(captured.sum() / hit.sum()) if hit.any() else None,
                "exposed": float((other >= lv).mean())}
    return out


def timing(W, seen, learned, repeats=3):
    idx = [i for i, _l in learned]
    xs, ts, ys = seen.Q[idx], [seen.cases[i]["utterance"] for i in idx], [l for _i, l in learned]
    runs = [Head(W, xs, ts, ys) for _ in range(repeats)]
    return {"rows": len(W.rows) + len(ys),
            "fit_s": float(np.median([r.fit_s for r in runs])), "calibration_s": float(np.median([r.cal_s for r in runs])),
            "total_s": float(np.median([r.fit_s + r.cal_s for r in runs]))}


# --------------------------------------------------------------------------- reporting


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def pct(a):
    return "-" if a is None else f"{100*a[0]:.1f} ({100*a[1]:.0f}-{100*a[2]:.0f})"


METRICS = (("first", "first"), ("r3", "top3"), ("wrong50", "wrong@50"), ("wrong70", "wrong@70"),
           ("right", "right outright"), ("wrong", "wrong confident"), ("handed", "handed on"), ("esc", "esc recall"))


REC_METRICS = (("first", "first"), ("right", "right outright"), ("wrong", "wrong confident"), ("handed", "handed on"),
               ("cache_hits", "cache answers"))


def rec_table(title, rows):
    print(f"\n#### {title}\n")
    print("| " + " | ".join(["mechanism"] + [m[1] for m in REC_METRICS]) + " |")
    print("|" + "---|" * (len(REC_METRICS) + 1))
    for name, ms in rows:
        print("| " + " | ".join([name] + [pct(agg([m[k] for m in ms])) for k, _n in REC_METRICS]) + " |")


def table(title, rows):
    print(f"\n#### {title}\n")
    print("| " + " | ".join(["step"] + [m[1] for m in METRICS]) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for name, ms in rows:
        print("| " + " | ".join([name] + [pct(agg([m[k] for m in ms])) for k, _n in METRICS]) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wf", action="append", required=True, help="NAME=DIR holding rows.json")
    ap.add_argument("--run", action="append", required=True,
                    help="TITLE=WF|seen=F1,F2|test=F3  or  TITLE=WF|split=FILE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--show-examples", default="", help="comma-separated workflow names whose value-swap examples are printed")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "emb")
    wfs = {}
    for spec in a.wf:
        n, d = spec.split("=", 1)
        wfs[n] = Workflow(n, d, cache)
    results = {}
    for spec in a.run:
        title, rest = spec.split("=", 1)
        parts = rest.split("|")
        W = wfs[parts[0]]
        kv = dict(p.split("=", 1) for p in parts[1:])
        if "split" in kv:
            cases = [c for c in load_cases([kv["split"]]) if W.keep(c)]
            cases.sort(key=lambda c: hashlib.sha256(f"{c['context']}\n{c['utterance']}".encode()).hexdigest())
            half = len(cases) // 2
            seen_cases, test_cases = cases[:half], cases[half:]
        else:
            seen_cases = load_cases(kv["seen"].split(","))
            test_cases = load_cases(kv["test"].split(","))
        seen_texts = {c["utterance"].strip().lower() for c in seen_cases}
        train_texts = {t.strip().lower() for t in W.texts}
        dropped = [c for c in test_cases if c["utterance"].strip().lower() in seen_texts | train_texts]
        test_cases = [c for c in test_cases if c not in dropped]
        seen, test = Cases(W, seen_cases), Cases(W, test_cases)
        near = float(np.mean((test.Q @ seen.Q.T).max(axis=1) >= CACHE_COSINE)) if len(seen.cases) and len(test.cases) else 0.0
        near_train = float(np.mean((test.Q @ W.X.T).max(axis=1) >= CACHE_COSINE)) if len(test.cases) else 0.0
        info = {"seen": len(seen.cases), "test": len(test.cases), "test_routing": int((~test.esc).sum()),
                "test_escalation": int(test.esc.sum()), "dropped_duplicates": len(dropped),
                "test_with_seen_near_dup": near, "test_with_training_row_near_dup": near_train}
        print(f"\n### {title}\n\n{info}", flush=True)
        per_shuffle, var_all, spreads, rec_all, nm_all = [], [], [], [], []
        base = Head(W, [], [], [])
        rec_sets, examples = recurrence_sets(W, seen)
        if W.name in a.show_examples.split(","):
            print("\nvalue-swap examples (before -> after):")
            for b, af in examples:
                print(f"  {b!r}\n    -> {af!r}")
        learned0 = None
        for s in range(a.shuffles):
            order = np.random.default_rng(s).permutation(len(seen.cases))
            pts, learned = stream_run(W, seen, test, order, a.batch)
            per_shuffle.append(pts)
            v = variants(W, seen, test, learned, s)
            spreads.append(v.pop("_spread"))
            var_all.append(v)
            sets = dict(rec_sets)
            sets["verbatim, corrected requests only"] = [
                {"context": seen.ctx[i], "kind": "routing", "expected_label": W.labels[l], "utterance": seen.cases[i]["utterance"]}
                for i, l in learned if not seen.esc[i]]
            rec_all.append(recurrence(W, seen, test, learned, base, refit(W, seen, learned), sets))
            nm_all.append(near_miss(W, seen, test, learned))
            if learned0 is None:
                learned0 = learned
            print(f"  shuffle {s}: {len(learned)} corrections, first {pts[0]['first']:.3f} -> {pts['all']['first']:.3f}", flush=True)
        labels = [0, 25, 50, 100, 200, "all"]
        rows = []
        for lab in labels:
            ms = [p[lab] for p in per_shuffle if lab in p]
            if ms:
                n = agg([m["corrections"] for m in ms])
                rows.append((f"{lab} (n={len(ms)}{', mean ' + format(n[0], '.0f') if lab == 'all' else ''})", ms))
        table(f"{title}: learning curve on the test set (mean, min-max over shuffles; % of requests)", rows)
        vrows = [("0 corrections (baseline)", [p[0] for p in per_shuffle])]
        for name in var_all[0]:
            vrows.append((name, [v[name] for v in var_all]))
        table(f"{title}: variants at all corrections", vrows)
        print(f"\nrecurring sets: {({k: len(v) for k, v in rec_sets.items()})}")
        for setname in rec_all[0]:
            rows = [(m, [r[setname][m] for r in rec_all]) for m in rec_all[0][setname]]
            rec_table(f"{title}: {setname}", rows)
        print(f"\n#### {title}: near-miss capture by the cache (share of unseen routing requests)\n")
        print("| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |")
        print("|---|---|---|---|---|")
        for key in nm_all[0]:
            f = lambda k: pct(agg([n[key][k] for n in nm_all]))
            print(f"| {key} | {f('hit')} | {f('captured_wrong')} | {f('wrong_share_of_hits')} | {f('exposed')} |")
        hits = [v["(d) cache + refit"] for v in var_all]
        print(f"\ncache: hit rate {pct(agg([h['cache_hits'] for h in hits]))}, hit precision {pct(agg([h['cache_precision'] for h in hits]))}")
        print("spread of learned rows over labels:", json.dumps(spreads))
        tm = timing(W, seen, learned0)
        print("refit+recalibration (single thread, vectors cached):", tm)
        results[title] = {"info": info, "curve": per_shuffle, "variants": var_all, "spread": spreads, "timing": tm, "recurrence": rec_all, "near_miss": nm_all,
                          "baseline_threshold": per_shuffle[0][0]["threshold"]}
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
