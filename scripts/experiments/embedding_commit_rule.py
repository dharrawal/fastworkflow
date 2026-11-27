#!/usr/bin/env python3
"""Commit-rule experiment for the ``embedding`` intent head (``docs/single_intent_model_preregistration_v5.md``).

The head (a frozen sentence encoder and one logistic-regression head per workflow) answers with one
WRONG label too often. Four ways to cut that are scored here, all computed from the same embeddings,
with no model call and no change to the product:

* V1  stricter calibration targets (used when the remote decision link exists);
* V2  cross-fit folds grouped by producing persona (seeds by their own text) instead of by row text;
* V3  agreement gate: commit only when the head's top label and a nearest-neighbour vote over the
      training rows, restricted to the context's mask, name the same label;
* V4  confirmed escalation: a lone "not here" only when a nearest-neighbour vote over ALL labels
      also lands outside the mask.

Steps (each reads and writes plain files; nothing is written into a workflow folder):

    split    --cases FILE --out-prefix P        deterministic two-way split by a hash of (context, utterance)
    dump     --workflow PATH --out DIR --set NAME=FILE[,FILE...] ...
                                                 rebuild rows from the utterance caches (no model call)
    develop  --data DIR [DIR ...] --out DIR     score every variant on the development sets, apply the
                                                 selection rule, write the recipes
    confirm  --data DIR [DIR ...] --out DIR --recipes DIR
                                                 score BASE and the registered recipes on the confirmation
                                                 sets (refuses unless the pre-registration names the recipes)
    pairs    --data DIR [DIR ...] --out DIR     the most confused command pairs of the base head's
                                                 cross-fitted rows

A data directory holds ``rows.json`` (rows, labels, masks, row groups) and one benchmark file per
named set (``cases_<name>.json``). A directory whose name starts with ``large`` is reported as the
"large" group; every other directory is pooled into the "examples" group. ``dump`` of a workflow needs
that workflow's own Python environment (it imports the framework to read the caches); every other
step runs anywhere. Every workflow's head is calibrated on its own rows, as the product does.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

ENCODER = "BAAI/bge-small-en-v1.5"
PARAMETER_VALUE = "parameter_value"
WILDCARD = "wildcard"
RESERVED = (PARAMETER_VALUE, WILDCARD)
FOLDS = 5
HEAD_C, HEAD_ITER = 100.0, 3000
FLOOR = 0.5
TOP_K = 3
KS = (5, 10)
WRONG_GRID = (0.01, 0.005, 0.0025, 0.001)
ESC_GRID = (0.005, 0.0025, 0.001)
BASE_TARGETS = (0.01, 0.005)
COMMIT_LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8)
PREREG = REPO / "docs" / "single_intent_model_preregistration_v5.md"

# --------------------------------------------------------------------------- split


def split_key(context: str, utterance: str) -> int:
    """0 = development half, 1 = confirmation half: the low bit of the first byte of a SHA-256 of
    ``context + newline + utterance``."""
    digest = hashlib.sha256(f"{context}\n{utterance}".encode("utf-8")).digest()
    return digest[0] & 1


def _split(args) -> int:
    raw = json.load(open(args.cases, encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    halves = ([], [])
    for case in cases:
        halves[split_key(case["context"], case["utterance"])].append(case)
    for name, half in zip(("dev", "conf"), halves):
        path = f"{args.out_prefix}_{name}.json"
        json.dump({"schema_version": 1, "cases": half}, open(path, "w", encoding="utf-8"), indent=1)
        kinds = {k: sum(c["kind"] == k for c in half) for k in ("routing", "escalation")}
        print(f"{path}: {len(half)} cases {kinds}")
    return 0


# --------------------------------------------------------------------------- dump


def _dump(args) -> int:
    import single_intent_model as runner
    from fastworkflow.train import determinism
    from fastworkflow.train import heldout_evaluation as he
    from fastworkflow.train import workflow_model as wm

    workflow = os.path.abspath(args.workflow)
    os.makedirs(args.out, exist_ok=True)
    env = runner.environment_for(workflow, None, None)
    copy = runner.copy_workflow(workflow, os.path.join(args.out, "_copy"))
    rows = wm.workflow_rows(copy, env_vars=env)
    provenance = determinism.ProvenanceRecorder.load(copy)
    out_rows, personas = [], 0
    for text, label in rows.rows:
        group = None
        record = provenance.get(label)
        if record is not None:
            persona = record.utterance_personas.get(text)
            if persona and persona != "__seed__":
                group, personas = f"persona:{persona}", personas + 1
        out_rows.append([text, label, group])
    masks = {scope: list(entry.mask) for scope, entry in rows.scopes.items()}
    labels = rows.labels
    json.dump({"rows": out_rows, "labels": labels, "masks": masks, "findings": rows.findings[:5]},
              open(os.path.join(args.out, "rows.json"), "w", encoding="utf-8"))
    print(f"rows {len(out_rows)} ({personas} with a recorded persona), labels {len(labels)}, "
          f"contexts {len(masks)}")
    norm = he.normalize_label
    label_set = {norm(label) for label in labels}
    for spec in args.set:
        name, files = spec.split("=", 1)
        kept, dropped = [], 0
        for path in files.split(","):
            raw = json.load(open(path, encoding="utf-8"))
            for case in (raw["cases"] if isinstance(raw, dict) else raw):
                if case["context"] not in masks:
                    dropped += 1
                    continue
                inside = {norm(m) for m in masks[case["context"]]}
                if case["kind"] == "routing" and norm(case["expected_label"]) not in inside:
                    dropped += 1
                    continue
                if case["kind"] == "escalation":
                    ancestor = norm(case.get("expected_ancestor_command") or "")
                    if ancestor in inside or ancestor not in label_set:
                        dropped += 1
                        continue
                kept.append(case)
        json.dump({"schema_version": 1, "cases": kept},
                  open(os.path.join(args.out, f"cases_{name}.json"), "w", encoding="utf-8"), indent=1)
        print(f"set {name}: kept {len(kept)}, dropped {dropped} (context or command not in the workflow)")
    return 0


# --------------------------------------------------------------------------- encoder


_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from fastworkflow.train import embedding_head as eh
        from fastworkflow.train import intent_model_layout as layout

        _ENCODER = eh.SentenceEncoder.get(ENCODER, layout.default_pooling(ENCODER))
    return _ENCODER


def embed_texts(texts, cache):
    key = hashlib.sha1(("\n".join(texts) + ENCODER).encode("utf-8")).hexdigest()
    path = os.path.join(cache, f"emb_{key}.npy") if cache else None
    if path and os.path.exists(path):
        return np.load(path)
    vectors = np.asarray(get_encoder().embed(list(texts)))
    if path:
        os.makedirs(cache, exist_ok=True)
        np.save(path, vectors)
    return vectors


def knn_vote(sims: np.ndarray, index_labels: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Label index of the k-nearest vote for each row of *sims* (cosine), over index rows whose label
    is *allowed*. Plurality, ties by the larger summed similarity."""
    ok = allowed[index_labels]
    sub = sims[:, ok]
    labels = index_labels[ok]
    k = min(k, sub.shape[1])
    near = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    nsim = np.take_along_axis(sub, near, axis=1)
    nlab = labels[near]
    score = np.zeros((sims.shape[0], len(allowed)))
    rows = np.repeat(np.arange(sims.shape[0]), k)
    np.add.at(score, (rows, nlab.ravel()), 1.0 + nsim.ravel() / (10.0 * k))
    return np.argmax(score, axis=1)


# --------------------------------------------------------------------------- data


class Data:
    """One workflow: rows, masks, embeddings, the head fitted on every row, and cross-fits."""

    def __init__(self, folder: str, cache: str | None):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        raw = json.load(open(os.path.join(folder, "rows.json"), encoding="utf-8"))
        self.labels = list(raw["labels"])
        self.index_of = {label: i for i, label in enumerate(self.labels)}
        self.texts = [r[0] for r in raw["rows"]]
        self.targets = np.array([self.index_of[r[1]] for r in raw["rows"]])
        self.row_groups = [r[2] for r in raw["rows"]]
        self.masks = {scope: list(m) for scope, m in raw["masks"].items()}
        self.scopes = sorted(self.masks)
        self.scope_pos = {s: i for i, s in enumerate(self.scopes)}
        self.mask_bool = np.zeros((len(self.scopes), len(self.labels)), dtype=bool)
        for scope, mask in self.masks.items():
            for label in mask:
                if label in self.index_of:
                    self.mask_bool[self.scope_pos[scope], self.index_of[label]] = True
        self.reserved = np.array([label in RESERVED for label in self.labels])
        self.vectors = np.asarray(embed_texts(self.texts, cache), dtype=np.float64)
        self.coef, self.intercept = self._fit()
        # the nearest-neighbour index: one entry per distinct (text, label)
        seen, rows = set(), []
        for i, (t, l) in enumerate(zip(self.texts, self.targets)):
            if (t, int(l)) not in seen:
                seen.add((t, int(l)))
                rows.append(i)
        self.index_rows = np.array(rows)
        self._cross = {}
        self._cases = {}

    def _fit(self):
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            head = LogisticRegression(C=HEAD_C, max_iter=HEAD_ITER).fit(self.vectors, self.targets)
        if list(head.classes_) != list(range(len(self.labels))):
            raise ValueError("rows do not cover every label")
        return head.coef_.astype(np.float32), head.intercept_.astype(np.float32)

    def groups(self, grouping: str) -> np.ndarray:
        """Fold groups. 'text': rows with the same text. 'persona': a generated row with every row of
        the persona that produced it (a seed, a bare value or a vague reply stands alone by its text);
        rows that share a text are always in one group. A generated row's seed utterance is not
        recorded anywhere (the generator is given all the seeds at once), so the producing persona is
        the closest recoverable provenance."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for text, group in zip(self.texts, self.row_groups):
            a = find(f"t:{text}")
            if grouping == "persona" and group:
                parent[find(a)] = find(group)
        keys = [find(f"t:{t}") for t in self.texts]
        return np.unique(np.array(keys, dtype=object).astype(str), return_inverse=True)[1]

    def cross(self, grouping: str):
        if grouping in self._cross:
            return self._cross[grouping]
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from threadpoolctl import threadpool_limits

        groups = self.groups(grouping)
        folds = min(FOLDS, int(groups.max()) + 1)
        probabilities = np.zeros((len(self.targets), len(self.labels)))
        splits = list(GroupKFold(n_splits=folds).split(self.vectors, self.targets, groups))
        with threadpool_limits(limits=1):
            for train, test in splits:
                head = LogisticRegression(C=HEAD_C, max_iter=HEAD_ITER).fit(self.vectors[train], self.targets[train])
                probabilities[np.ix_(test, head.classes_)] = head.predict_proba(self.vectors[test])
        self._cross[grouping] = (probabilities, splits, int(groups.max()) + 1)
        return self._cross[grouping]

    def records(self, grouping: str) -> dict:
        """Calibration records (one per distinct text and context holding its label), as arrays.

        p_in, gap (p_out - p_in), right, best (0 command, 1 bare value, 2 wildcard), and for k in KS:
        agree<k> (a nearest-neighbour vote over the training side of the row's fold, restricted to
        the mask, names the head's top label inside the mask) and nnout<k> (the vote over all labels
        lands outside the mask)."""
        probabilities, splits, _ = self.cross(grouping)
        fold_of = np.zeros(len(self.targets), dtype=int)
        for f, (_train, test) in enumerate(splits):
            fold_of[test] = f
        seen, rows_used = set(), []
        for i, text in enumerate(self.texts):
            if self.labels[self.targets[i]] in RESERVED or text in seen:
                continue
            seen.add(text)
            rows_used.append(i)
        out = {k: [] for k in ("p_in", "gap", "right", "best", "agree5", "agree10", "nnout5", "nnout10")}
        index_set = set(self.index_rows.tolist())
        all_mask = np.ones(len(self.labels), dtype=bool)
        for f, (train, _test) in enumerate(splits):
            train_idx = np.array(sorted(set(train.tolist()) & index_set))
            test_rows = [i for i in rows_used if fold_of[i] == f]
            if not test_rows:
                continue
            sims = self.vectors[test_rows] @ self.vectors[train_idx].T
            train_lab = self.targets[train_idx]
            for scope_i in range(len(self.scopes)):
                mask = self.mask_bool[scope_i]
                sel = [j for j, i in enumerate(test_rows) if mask[self.targets[i]]]
                if not sel:
                    continue
                inside = np.flatnonzero(mask)
                outside = np.flatnonzero(~mask & ~self.reserved)
                chosen = [test_rows[j] for j in sel]
                P = probabilities[chosen]
                top = inside[np.argmax(P[:, inside], axis=1)]
                p_in = P[np.arange(len(sel)), top]
                p_out = P[:, outside].max(axis=1) if len(outside) else np.zeros(len(sel))
                best_idx = np.argmax(P, axis=1)
                out["p_in"] += p_in.tolist()
                out["gap"] += (p_out - p_in).tolist()
                out["right"] += (top == self.targets[chosen]).tolist()
                out["best"] += [1 if self.labels[b] == PARAMETER_VALUE else 2 if self.labels[b] == WILDCARD else 0
                                for b in best_idx]
                S = sims[sel]
                for k in KS:
                    out[f"agree{k}"] += (knn_vote(S, train_lab, mask, k) == top).tolist()
                    out[f"nnout{k}"] += (~mask[knn_vote(S, train_lab, all_mask, k)]).tolist()
        return {k: np.array(v) for k, v in out.items()}

    def case_features(self, name: str, cases) -> list[dict]:
        if name in self._cases:
            return self._cases[name]
        vec = np.asarray(get_encoder().embed([c.utterance for c in cases]), dtype=np.float64)
        logits = vec @ self.coef.T.astype(np.float64) + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        sims = vec @ self.vectors[self.index_rows].T
        index_lab = self.targets[self.index_rows]
        all_mask = np.ones(len(self.labels), dtype=bool)
        votes = {}
        for scope in sorted({c.context for c in cases}):
            sel = [i for i, c in enumerate(cases) if c.context == scope]
            mask = self.mask_bool[self.scope_pos[scope]]
            for k in KS:
                vin = knn_vote(sims[sel], index_lab, mask, k)
                vall = knn_vote(sims[sel], index_lab, all_mask, k)
                for j, i in enumerate(sel):
                    votes[(i, k)] = (int(vin[j]), int(vall[j]))
        feats = []
        for i, case in enumerate(cases):
            mask = self.mask_bool[self.scope_pos[case.context]]
            inside = np.flatnonzero(mask)
            outside = np.flatnonzero(~mask & ~self.reserved)
            row = P[i]
            order = inside[np.argsort(-row[inside], kind="stable")][:TOP_K]
            feats.append({
                "case": case, "best": self.labels[int(np.argmax(row))],
                "p_in": float(row[order[0]]) if len(order) else 0.0,
                "p_out": float(row[outside].max()) if len(outside) else 0.0,
                "top": [self.labels[j] for j in order],
                "agree": {k: bool(len(order)) and votes[(i, k)][0] == int(order[0]) for k in KS},
                "nnout": {k: not mask[votes[(i, k)][1]] for k in KS},
            })
        self._cases[name] = feats
        return feats


# --------------------------------------------------------------------------- recipe


def config_name(c: dict) -> str:
    parts = [f"targets {c['wrong']:g}/{c['esc']:g}"]
    if c["grouping"] != "text":
        parts.append("persona folds")
    if c["gate"]:
        parts.append(f"gate k={c['gate']}")
    if c["confirm"]:
        parts.append(f"confirm k={c['confirm']}")
    return ", ".join(parts)


def calibrate(rec: dict, c: dict) -> dict:
    """Threshold and margin of one recipe from cross-fitted records. Mirrors
    `head_rule_as_experimented.calibrate`, the product's `embedding_head.calibrate` as it was until fix-o5wi.9
    (floor 0.5 under the threshold); the agreement gate is counted in the
    wrong commits and the confirmation in the rows sent up."""
    count = len(rec["p_in"])
    gap, p_in, right, best = rec["gap"], rec["p_in"], rec["right"], rec["best"]
    confirm = rec[f"nnout{c['confirm']}"] if c["confirm"] else np.ones(count, dtype=bool)
    gaps = np.sort(gap[(gap > 0) & (best != 1) & confirm])[::-1]
    allowed = int(c["esc"] * count)
    margin = float(gaps[allowed] + 1e-9) if len(gaps) > allowed else 0.0
    ahead = gap <= 0
    gate = rec[f"agree{c['gate']}"] if c["gate"] else np.ones(count, dtype=bool)
    wrong_p = np.sort(p_in[ahead & ~right & gate])
    threshold = 1.0
    for cand in sorted({round(float(x), 3) for x in p_in[ahead]}):
        above = len(wrong_p) - int(np.searchsorted(wrong_p, cand, side="left"))
        if above / count <= c["wrong"]:
            threshold = float(cand)
            break
    return {"threshold": max(threshold, FLOOR), "margin": margin, "records": count,
            "accuracy": float((ahead & right).mean()) if count else None}


def decide(f: dict, c: dict, threshold: float, margin: float) -> list[str]:
    """Labels returned for one case (the rule of `embedding_head.decide`, plus V3 and V4)."""
    top, p_in, p_out, best = f["top"], f["p_in"], f["p_out"], f["best"]
    confirmed = f["nnout"][c["confirm"]] if c["confirm"] else True
    gate = f["agree"][c["gate"]] if c["gate"] else True
    if best == PARAMETER_VALUE:
        return [PARAMETER_VALUE]
    if best == WILDCARD:
        return [WILDCARD] if (confirmed or not top) else list(top)
    if not top:
        return [WILDCARD]
    if p_out > p_in and p_out - p_in >= margin:
        return [WILDCARD] if confirmed else list(top)
    if p_out <= p_in and p_in >= threshold and gate:
        return top[:1]
    return list(top)


def verdicts(feats, c, threshold, margin) -> list[dict]:
    out = []
    for f in feats:
        case = f["case"]
        pred = decide(f, c, threshold, margin)
        lone, routing = len(pred) == 1, case.kind == "routing"
        expected = case.expected_label.strip() if routing else None
        out.append({
            "kind": case.kind,
            "top1": routing and pred == [expected],
            "in_list": routing and expected in pred,
            "one_label": lone,
            "lone_wrong": routing and lone and pred != [expected],
            "lone_right": routing and lone and pred == [expected],
            "false_escalation": routing and lone and pred == [WILDCARD],
            "escalated": case.kind == "escalation" and lone and pred == [WILDCARD],
        })
    return out


def summarise(v: list[dict]) -> dict:
    routing = [x for x in v if x["kind"] == "routing"]
    esc = [x for x in v if x["kind"] == "escalation"]
    n = max(len(routing), 1)
    lone = sum(x["one_label"] for x in routing)
    return {
        "routing": len(routing), "escalation": len(esc),
        "top1": sum(x["top1"] for x in routing) / n,
        "in_list": sum(x["in_list"] for x in routing) / n,
        "one_label": lone / n,
        "wrong_lone": sum(x["lone_wrong"] for x in routing) / n,
        "false_esc": sum(x["false_escalation"] for x in routing) / n,
        "esc_recall": [sum(x["escalated"] for x in esc), len(esc)],
        "precision": (sum(x["lone_right"] for x in routing) / lone) if lone else None,
    }


# --------------------------------------------------------------------------- groups


def load_cases(folder: str, name: str):
    from fastworkflow.train import heldout_evaluation as he

    return he.load_benchmark_file(os.path.join(folder, f"cases_{name}.json"))


class Group:
    """A reporting group: one or more workflows whose verdicts are pooled."""

    def __init__(self, folders: list[str], cache: str | None):
        self.data = [Data(f, cache) for f in folders]
        self._cal = {}

    def calibration(self, c: dict) -> list[dict]:
        key = json.dumps(c, sort_keys=True)
        if key not in self._cal:
            self._cal[key] = [calibrate(d.records(c["grouping"]), c) for d in self.data]
        return self._cal[key]

    def score(self, c: dict, set_name: str, threshold=None) -> tuple[dict, list[dict]]:
        cal = self.calibration(c)
        allv = []
        for d, k in zip(self.data, cal):
            feats = d.case_features(set_name, load_cases(d.folder, set_name))
            allv += verdicts(feats, c, k["threshold"] if threshold is None else threshold, k["margin"])
        return summarise(allv), cal

    def sweep(self, c: dict, set_name: str) -> dict:
        """Wrong lone answers when the threshold is moved to reach a committed share (margin as calibrated)."""
        points = []
        for thr in np.round(np.arange(0.0, 1.0001, 0.005), 3):
            s, _ = self.score(c, set_name, float(thr))
            points.append((s["one_label"], s["wrong_lone"], float(thr)))
        out = {}
        for level in COMMIT_LEVELS:
            ok = [p for p in points if p[0] >= level - 1e-9]
            if not ok:
                out[level] = None
                continue
            best = min(ok, key=lambda p: (p[0], p[2]))
            out[level] = {"committed": best[0], "wrong_lone": best[1], "threshold": best[2]}
        return out


STRUCTURES = [dict(grouping=g, gate=a, confirm=b) for g in ("text", "persona") for a in (0, *KS) for b in (0, *KS)]


def make(structure: dict, wrong: float, esc: float) -> dict:
    return {**structure, "wrong": wrong, "esc": esc}


def violation(dev: dict, base: dict) -> float:
    """The selection rule, in one number that is zero exactly when every limit holds. Group by group:
    wrong lone answers at most 3.5% (examples) or at most 45% of BASE's (large workflow); strict top-1
    at most 8 points below BASE; in-list not below BASE; escalation recall not more than 2 points
    (examples) or 8% (large, relative) below BASE; false escalation at most 1.2%. Each excess is
    counted in points of a hundred."""
    v = 0.0
    for g, s in dev.items():
        b = base[g]
        limit = 0.035 if g == "examples" else 0.45 * b["wrong_lone"]
        v += max(0.0, s["wrong_lone"] - limit) * 100
        v += max(0.0, b["top1"] - 0.08 - s["top1"]) * 100
        v += max(0.0, b["in_list"] - s["in_list"]) * 100
        r = s["esc_recall"][0] / max(s["esc_recall"][1], 1)
        rb = b["esc_recall"][0] / max(b["esc_recall"][1], 1)
        v += max(0.0, (rb - 0.02 if g == "examples" else 0.92 * rb) - r) * 100
        v += max(0.0, s["false_esc"] - 0.012) * 100
    return v


def parts_of(c: dict) -> int:
    return (c["grouping"] != "text") + bool(c["gate"]) + bool(c["confirm"]) + ((c["wrong"], c["esc"]) != BASE_TARGETS)


def mean_top1(summary: dict) -> float:
    return float(np.mean([s["top1"] for s in summary.values()]))


def choose(results: list[dict], base: dict, remote: bool) -> dict:
    """The recipe: among configurations that satisfy every limit (only the base targets when there is
    no remote link), the highest mean strict top-1 over the groups, ties to fewer parts; when none
    does, the smallest violation."""
    pool = [r for r in results if remote or (r["config"]["wrong"], r["config"]["esc"]) == BASE_TARGETS]
    scored = [(violation(r["summary"], base), r) for r in pool]
    passing = [s for s in scored if s[0] == 0.0]
    if passing:
        best = max(passing, key=lambda s: (mean_top1(s[1]["summary"]), -parts_of(s[1]["config"])))
    else:
        best = min(scored, key=lambda s: (s[0], -mean_top1(s[1]["summary"]), parts_of(s[1]["config"])))
    return {"config": best[1]["config"], "satisfies_every_limit": best[0] == 0.0, "violation": best[0],
            "dev": best[1]["summary"]}


# --------------------------------------------------------------------------- reports


def fmt(s: dict) -> str:
    prec = "-" if s["precision"] is None else f"{s['precision']:.1%}"
    return (f"{s['top1']:.1%} | {s['in_list']:.1%} | {s['one_label']:.1%} | {s['wrong_lone']:.1%} | "
            f"{s['false_esc']:.1%} | {s['esc_recall'][0]}/{s['esc_recall'][1]} | {prec}")


HEADER = ("| Variant | Group | Strict top-1 | In-list | One label | Wrong lone | False esc. | Esc. recall | "
          "Lone precision |\n|---|---|---|---|---|---|---|---|---|")


def _groups(args) -> dict[str, Group]:
    examples = [d for d in args.data if not os.path.basename(os.path.normpath(d)).startswith("large")]
    large = [d for d in args.data if os.path.basename(os.path.normpath(d)).startswith("large")]
    groups = {}
    if examples:
        groups["examples"] = Group(examples, args.cache)
    if large:
        groups["large"] = Group(large, args.cache)
    return groups


BASE_CONFIG = make(dict(grouping="text", gate=0, confirm=0), *BASE_TARGETS)


def _crosscheck(groups) -> None:
    """BASE through this script's own functions must equal the head's rule as it was when this ran (``head_rule_as_experimented``)."""
    try:
        from fastworkflow.train import embedding_head as eh
        import head_rule_as_experimented as legacy   # the calibration the product had when this ran (fix-o5wi.9)
    except Exception as exc:  # noqa: BLE001 - the module may be mid-edit
        print(f"crosscheck skipped ({exc})")
        return
    for grp in groups.values():
        d = grp.data[0]
        labels_of_rows = [d.labels[t] for t in d.targets]
        eh_probs, _facts = eh.cross_fitted_probabilities(d.vectors, d.targets, d.texts, len(d.labels))
        assert np.allclose(d.cross("text")[0], eh_probs, atol=1e-6), "cross-fit differs from the product"
        ref = legacy.calibrate(legacy.calibration_records(eh_probs, list(zip(d.texts, labels_of_rows)), d.labels,
                                                          d.masks), *BASE_TARGETS, threshold_floor=FLOOR)
        mine = calibrate(d.records("text"), BASE_CONFIG)
        assert abs(ref["threshold"] - mine["threshold"]) < 1e-9 and abs(ref["margin"] - mine["margin"]) < 1e-6, (ref, mine)
        print(f"crosscheck ok: threshold {mine['threshold']:.3f} and margin {mine['margin']:.3f} "
              f"equal the rule as it was before fix-o5wi.9 ({len(d.texts)} rows)")


def _develop(args) -> int:
    groups = _groups(args)
    _crosscheck(groups)
    base = {g: grp.score(BASE_CONFIG, "dev")[0] for g, grp in groups.items()}
    results = []
    for structure in STRUCTURES:
        for wrong, esc in itertools.product(WRONG_GRID, ESC_GRID):
            c = make(structure, wrong, esc)
            results.append({"config": c, "summary": {g: grp.score(c, "dev")[0] for g, grp in groups.items()}})
    accuracy = {}
    for g, grp in groups.items():
        for grouping in ("text", "persona"):
            recs = [d.records(grouping) for d in grp.data]
            n = sum(len(r["p_in"]) for r in recs)
            accuracy[f"{g}/{grouping}"] = {
                "records": n, "right": sum(int((r["right"] & (r["gap"] <= 0)).sum()) for r in recs) / n}
    remote, no_remote = choose(results, base, True), choose(results, base, False)
    best_targets = {}
    for structure in STRUCTURES:
        pool = [r for r in results if all(r["config"][k] == v for k, v in structure.items())]
        pick = min(pool, key=lambda r: (violation(r["summary"], base), -mean_top1(r["summary"])))
        best_targets[f"{structure['grouping']}/{structure['gate']}/{structure['confirm']}"] = pick["config"]
    trade = {label: {g: grp.sweep(c, "dev") for g, grp in groups.items()}
             for label, c in (("BASE", BASE_CONFIG), ("remote recipe", remote["config"]),
                              ("no-remote recipe", no_remote["config"]))}
    report = {"base": base, "base_calibration": {g: grp.calibration(BASE_CONFIG) for g, grp in groups.items()},
              "fold_accuracy": accuracy, "recipe_remote": remote, "recipe_no_remote": no_remote,
              "best_targets_by_structure": best_targets, "tradeoff": trade, "all": results}
    os.makedirs(args.out, exist_ok=True)
    json.dump(report, open(os.path.join(args.out, "develop.json"), "w", encoding="utf-8"), indent=1, default=str)
    json.dump({"remote": remote["config"], "no_remote": no_remote["config"]},
              open(os.path.join(args.out, "recipes.json"), "w"), indent=1)
    _print_develop(report, groups)
    return 0


def _print_develop(report, groups) -> None:
    print("\nBASE calibration per workflow:")
    for g, cals in report["base_calibration"].items():
        for k in cals:
            print(f"  {g}: threshold {k['threshold']:.3f}, margin {k['margin']:.3f}, {k['records']} records")
    print("\nHow right the cross-fitted rows are (the best label inside the mask is the row's):")
    for k, v in report["fold_accuracy"].items():
        print(f"  {k}: {v['right']:.1%} of {v['records']}")
    lookup = {json.dumps(r["config"], sort_keys=True): r["summary"] for r in report["all"]}

    def get(structure, targets):
        return lookup[json.dumps(make(structure, *targets), sort_keys=True)]

    print("\n" + HEADER)
    named = [
        ("BASE", dict(grouping="text", gate=0, confirm=0), BASE_TARGETS),
        ("V1 best targets", dict(grouping="text", gate=0, confirm=0), None),
        ("V2 persona folds", dict(grouping="persona", gate=0, confirm=0), BASE_TARGETS),
        ("V3 gate k=5", dict(grouping="text", gate=5, confirm=0), BASE_TARGETS),
        ("V3 gate k=10", dict(grouping="text", gate=10, confirm=0), BASE_TARGETS),
        ("V4 confirm k=5", dict(grouping="text", gate=0, confirm=5), BASE_TARGETS),
        ("V4 confirm k=10", dict(grouping="text", gate=0, confirm=10), BASE_TARGETS),
        ("V2+V3 k=10", dict(grouping="persona", gate=10, confirm=0), BASE_TARGETS),
        ("V2+V4 k=10", dict(grouping="persona", gate=0, confirm=10), BASE_TARGETS),
        ("V3+V4 k=10", dict(grouping="text", gate=10, confirm=10), BASE_TARGETS),
        ("V2+V3+V4 k=10", dict(grouping="persona", gate=10, confirm=10), BASE_TARGETS),
    ]
    for label, structure, targets in named:
        if targets is None:
            c = report["best_targets_by_structure"][f"{structure['grouping']}/{structure['gate']}/{structure['confirm']}"]
            targets, label = (c["wrong"], c["esc"]), f"{label} {c['wrong']:g}/{c['esc']:g}"
        for g in groups:
            print(f"| {label} | {g} | {fmt(get(structure, targets)[g])} |")
    for key in ("recipe_remote", "recipe_no_remote"):
        r = report[key]
        print(f"\n{key}: {config_name(r['config'])} (every limit met: {r['satisfies_every_limit']}, violation {r['violation']:.2f})")
        for g in groups:
            print(f"| {key} | {g} | {fmt(r['dev'][g])} |")
    print("\nWrong lone answers (committed share) when the threshold is moved to commit on about 40..80% of routing cases:")
    for label, per in report["tradeoff"].items():
        for g, sweep in per.items():
            cells = []
            for level in COMMIT_LEVELS:
                p = sweep.get(level) if level in sweep else sweep.get(str(level))
                cells.append("n/a" if p is None else f"{p['wrong_lone']:.1%} ({p['committed']:.0%})")
            print(f"| {label} | {g} | " + " | ".join(cells) + " |")


# --------------------------------------------------------------------------- confirm


def _confirm(args) -> int:
    recipes = json.load(open(os.path.join(args.recipes, "recipes.json")))
    doc = PREREG.read_text(encoding="utf-8") if PREREG.exists() else ""
    for label, c in recipes.items():
        if config_name(c) not in doc:
            print(f"refusing: the pre-registration does not name the recipe {config_name(c)!r}")
            return 2
    groups = _groups(args)
    arms = {"BASE": BASE_CONFIG, **{f"recipe ({label})": c for label, c in recipes.items()}}
    out = {}
    for arm, c in arms.items():
        out[arm] = {"config": c}
        for g, grp in groups.items():
            s, cal = grp.score(c, "conf")
            out[arm][g] = {"summary": s, "calibration": cal}
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "confirm.json"), "w"), indent=1, default=str)
    print(HEADER)
    for arm, v in out.items():
        for g in groups:
            print(f"| {arm} | {g} | {fmt(v[g]['summary'])} |")
    return 0


# --------------------------------------------------------------------------- pairs


def _pairs(args) -> int:
    for folder in args.data:
        d = Data(folder, args.cache)
        probabilities = d.cross("text")[0]
        seen, rows = set(), []
        for i, text in enumerate(d.texts):
            if d.labels[d.targets[i]] in RESERVED or text in seen:
                continue
            seen.add(text)
            rows.append(i)
        per_label, confusion = {}, {}
        for i in rows:
            t = int(d.targets[i])
            per_label[t] = per_label.get(t, 0) + 1
            guess = int(np.argmax(probabilities[i]))
            if guess != t and d.labels[guess] not in RESERVED:
                confusion[(t, guess)] = confusion.get((t, guess), 0) + 1
        pairs = {}
        for (a, b), n in confusion.items():
            pairs.setdefault(tuple(sorted((a, b))), {})[(a, b)] = n
        table = []
        for (a, b), both in pairs.items():
            ab, ba = both.get((a, b), 0), both.get((b, a), 0)
            table.append({"a": d.labels[a], "b": d.labels[b], "a_as_b": ab, "a_rows": per_label[a],
                          "a_share": ab / per_label[a], "b_as_a": ba, "b_rows": per_label[b],
                          "b_share": ba / per_label[b], "total": ab + ba,
                          "max_share": max(ab / per_label[a], ba / per_label[b])})
        table.sort(key=lambda r: (-r["total"], r["a"], r["b"]))
        over = sum(r["max_share"] > 0.10 for r in table)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"confused_pairs_{d.name}.json")
        json.dump({"workflow": d.name, "pairs_over_10_percent": over, "confused_pairs": len(table),
                   "top10": table[:10], "all": table}, open(path, "w"), indent=1)
        print(f"{d.name}: {len(table)} confused pairs, {over} with more than 10% of a command's rows "
              f"given to the other; wrote {path}")
        if not d.name.startswith("large"):
            for r in table[:10]:
                print(f"  {r['a']} -> {r['b']}: {r['a_as_b']}/{r['a_rows']} ({r['a_share']:.1%}); "
                      f"{r['b']} -> {r['a']}: {r['b_as_a']}/{r['b_rows']} ({r['b_share']:.1%})")
    return 0


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("split")
    s.add_argument("--cases", required=True)
    s.add_argument("--out-prefix", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--workflow", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--set", action="append", default=[], help="NAME=FILE[,FILE...]")
    for name in ("develop", "confirm", "pairs"):
        s = sub.add_parser(name)
        s.add_argument("--data", nargs="+", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--cache", default=None)
        if name == "confirm":
            s.add_argument("--recipes", required=True, help="the folder develop wrote recipes.json to")
    args = p.parse_args(argv)
    return {"split": _split, "dump": _dump, "develop": _develop, "confirm": _confirm, "pairs": _pairs}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
