#!/usr/bin/env python3
"""Lever sweep for the LOCAL intent model (the ``embedding`` head), with no model call of any kind.

Where does a frozen sentence encoder plus one linear head stand on a large workflow, and which cheap
change helps? Everything here reads rows from the utterance caches (rows are never generated) and
scores the development benchmarks only. Nothing is written into a workflow folder or into the product.

Steps (each reads and writes plain files):

    meta      --workflow COPY --rows ROWS.json --out FILE
                  per label: seed utterances, docstring, planning flag, the plain commands a planning
                  command's source names. Needs the workflow's own Python environment (imports the framework).
    diagnose  ...   threshold-free diagnosis of the shipped head: ranking, misses by kind, by training rows,
                  by value or no value, and against the cross-fit's most confused pairs.
    sweep     ...   run named variants (encoder, head, data shape, scope, fine-tuning, combinations) and write
                  one JSON per variant under OUT/results.
    latency   ...   median request time per encoder at 10 threads and at 1 thread.
    tables    ...   print the markdown tables from OUT/results.

Common arguments: ``--wf NAME=DIR`` for each workflow (DIR holds ``rows.json`` and ``cases_dev.json`` as
``embedding_commit_rule.py dump`` writes them, and ``meta.json`` from the ``meta`` step), ``--group
GROUP=NAME,NAME`` for the reporting groups (pooled), ``--pairs NAME=FILE`` for the cross-fit's confused pairs
(``embedding_commit_rule.py pairs``), ``--out DIR`` for scratch (embedding cache, results).

Example numbers (library + retail, development sets v1-v3) are reproduced from the repo with::

    python scripts/experiments/embedding_commit_rule.py dump --workflow fastworkflow/examples/library --out D/library \\
        --set dev=tests/fixtures/intent_benchmarks/library.json,..._v2.json,..._v3.json      (same for retail_workflow)
    python scripts/experiments/local_head_levers.py meta --workflow D/library/_copy/library --rows D/library/rows.json --out D/library/meta.json
    python scripts/experiments/local_head_levers.py sweep --wf library=D/library --wf retail_workflow=D/retail_workflow \\
        --group examples=library,retail_workflow --out OUT --variants all

The large workflow's folders are passed the same way as a third ``--wf`` and a ``--group large=NAME``.

Definitions (all threshold-free):
  first-ranked   the best label INSIDE the context's mask is the expected one.
  recall@k       the expected label is among the k best inside the mask (ties count against the variant).
  pass-up        for escalation cases: the best label overall is not inside the mask (any other command, or the
                 workflow ``wildcard`` label); "ancestor" is the best label overall being the expected
                 ancestor command exactly.
  wrong@N        requests are ranked by the head's own confidence (the score of its best label overall); the top
                 N% of routing requests are "committed"; the answer is that best label overall, so a request
                 passed up to another context's command or to ``wildcard`` is a wrong lone answer. Reported as a
                 share of ALL routing requests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PARAMETER_VALUE = "parameter_value"
WILDCARD = "wildcard"
RESERVED = (PARAMETER_VALUE, WILDCARD)
TAU = 0.05  # temperature of the similarity weighting (kNN votes, prototype softmax)
MAXLEN = 128

# encoder id -> (pooling, text prefix). The documented recipe of each model.
#   bge: CLS, no prefix (the shipped setting; the "query instruction" is for retrieval, not for this).
#   sentence-transformers all-*: attention-masked mean, no prefix.   gte: mean, no prefix.
#   e5: mean, "query: " prefix on every text (the documented prefix for symmetric tasks).
ENCODERS = {
    "BAAI/bge-small-en-v1.5": ("cls", ""),
    "BAAI/bge-base-en-v1.5": ("cls", ""),
    "sentence-transformers/all-MiniLM-L12-v2": ("mean", ""),
    "sentence-transformers/all-mpnet-base-v2": ("mean", ""),
    "thenlper/gte-small": ("mean", ""),
    "thenlper/gte-base": ("mean", ""),
    "intfloat/e5-small-v2": ("mean", "query: "),
    "intfloat/e5-base-v2": ("mean", "query: "),
}
SHIPPED = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------------------------------- encoder


class Encoder:
    """A frozen (or supplied) transformer read the way ``embedding_head.SentenceEncoder`` reads one."""

    def __init__(self, model_id: str, cache: str | None, model=None, tag: str = ""):
        self.model_id, self.cache, self.tag = model_id, cache, tag
        self.pooling, self.prefix = ENCODERS[model_id]
        self._model = model
        self._tok = None

    def _load(self):
        if self._tok is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            if self._model is None:
                self._model = AutoModel.from_pretrained(self.model_id).eval()

    def tokens(self, texts):
        self._load()
        return self._tok([self.prefix + t for t in texts], padding=True, truncation=True,
                         max_length=MAXLEN, return_tensors="pt")

    def pool(self, out, enc):
        import torch

        hidden = out.last_hidden_state
        if self.pooling == "cls":
            vec = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(vec, dim=-1)

    def embed(self, texts, batch: int = 64, use_cache: bool = True) -> np.ndarray:
        texts = list(texts)
        key = hashlib.sha1("\n".join([self.model_id, self.tag, self.prefix, self.pooling, *texts]).encode()).hexdigest()
        path = os.path.join(self.cache, f"emb_{key}.npy") if (self.cache and use_cache) else None
        if path and os.path.exists(path):
            return np.load(path)
        self._load()
        import torch

        # sort by length so the padding stays small; the result is put back in order
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), 0))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                idx = order[start:start + batch]
                enc = self.tokens([texts[i] for i in idx])
                chunks.append(self.pool(self._model(**enc), enc).float().numpy())
        mat = np.concatenate(chunks) if chunks else out
        result = np.zeros_like(mat)
        result[order] = mat
        if path:
            os.makedirs(self.cache, exist_ok=True)
            np.save(path, result)
        return result


_ENC_CACHE: dict = {}


def encoder_for(model_id: str, cache: str | None) -> Encoder:
    if model_id not in _ENC_CACHE:
        _ENC_CACHE.clear()  # one model in memory at a time
        _ENC_CACHE[model_id] = Encoder(model_id, cache)
    return _ENC_CACHE[model_id]


# --------------------------------------------------------------------------- value masking

PLACEHOLDER = "[VALUE]"
#: (threshold, margin) of the shipped head per workflow, from the product's own calibration on its rows
#: (cross-checked equal to the product's in the earlier commit-rule run). Passed in by --shipped-op.
SHIPPED_OP: dict = {}
CALIBRATE: set = set()  # variants that also get the cross-fit calibrations (per workflow and per context)
_QUOTED = re.compile(r"\"[^\"]+\"|(?<![\w'])'[^']{1,80}'(?![\w'])|“[^”]+”")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_ID = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{3,}\b")
_NUMBER = re.compile(r"(?<![\w])\d[\d,./:-]*(?![\w])")
_CAP = re.compile(r"[A-Z][\w'-]*")


_NAME_RUN = re.compile(r"\b[A-Z][\w'-]*(?:[ \t]+[A-Z][\w'-]*)+")


def _mask_names(text: str) -> str:
    """Runs of two or more capitalised words become the placeholder; the first word of the text is a
    sentence start and never counts toward a run."""
    def sub(m):
        run = m.group(0)
        if m.start() == 0:
            head, _, rest = run.partition(" ")
            return run if " " not in rest and not rest.strip() or len(rest.split()) < 2 else head + " " + PLACEHOLDER
        return PLACEHOLDER
    return _NAME_RUN.sub(sub, text)


def mask_values(text: str) -> str:
    """Concrete values -> one placeholder: quoted strings, emails, ids (letters and digits mixed), numbers,
    capitalised multi-word names."""
    text = _QUOTED.sub(PLACEHOLDER, text)
    text = _EMAIL.sub(PLACEHOLDER, text)
    text = _ID.sub(PLACEHOLDER, text)
    text = _NUMBER.sub(PLACEHOLDER, text)
    text = _mask_names(text)
    return re.sub(r"(?:\[VALUE\][\s,]*){2,}", PLACEHOLDER + " ", text).strip()


def carries_value(text: str) -> bool:
    """The broad test used by the diagnosis: a quoted string, an email, an id or number, or any capitalised
    word that does not open the text."""
    if _QUOTED.search(text) or _EMAIL.search(text) or _ID.search(text) or _NUMBER.search(text):
        return True
    words = text.split()
    return any(_CAP.fullmatch(w.strip(".,;:!?\"'")) and w.strip(".,;:!?\"'") not in ("I", "I'm", "I'd", "I'll") for w in words[1:])


# --------------------------------------------------------------------------- data


def norm_name(label: str) -> str:
    return label.split("/")[-1]


def owner_of(label: str) -> str:
    return label.split("/")[0] if "/" in label else ""


def humanise(label: str) -> str:
    return norm_name(label).replace("_", " ")


def first_sentence(doc: str) -> str:
    doc = " ".join((doc or "").split())
    return re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0] if doc else ""


class Workflow:
    """One workflow: rows, masks, held-out development cases, per-label metadata."""

    def __init__(self, name: str, folder: str):
        self.name, self.folder = name, folder
        raw = json.load(open(os.path.join(folder, "rows.json"), encoding="utf-8"))
        self.labels = list(raw["labels"])
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.texts = [r[0] for r in raw["rows"]]
        self.y = np.array([self.index[r[1]] for r in raw["rows"]])
        self.masks = {s: list(m) for s, m in raw["masks"].items()}
        self.scopes = sorted(self.masks)
        self.mask_bool = {s: np.zeros(len(self.labels), dtype=bool) for s in self.scopes}
        for s, m in self.masks.items():
            for l in m:
                if l in self.index:
                    self.mask_bool[s][self.index[l]] = True
        self.reserved = np.array([l in RESERVED for l in self.labels])
        meta_path = os.path.join(folder, "meta.json")
        self.meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        cases = json.load(open(os.path.join(folder, "cases_dev.json"), encoding="utf-8"))["cases"]
        self.cases = cases
        self.rows_per_label = np.bincount(self.y, minlength=len(self.labels))
        seeds = {l: set(m.get("seeds", [])) for l, m in self.meta.items()}
        self.is_seed = np.array([t in seeds.get(self.labels[l], ()) for t, l in zip(self.texts, self.y)])
        suffixes: dict[str, list[int]] = {}
        for i, l in enumerate(self.labels):
            if not self.reserved[i]:
                suffixes.setdefault(norm_name(l), []).append(i)
        self.shared = {s: ids for s, ids in suffixes.items() if len(ids) > 1}

    def kind(self, label: str) -> str:
        """The diagnosis kind of a label: planning, navigation, or command."""
        m = self.meta.get(label, {})
        name = norm_name(label)
        if m.get("planning"):
            return "planning"
        if owner_of(label) == "IntentDetection" or re.match(r"(open|go|reset|what_can|what_is_current)", name):
            return "navigation"
        return "command"


# --------------------------------------------------------------------------- heads


def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = a - a.max(axis=axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=axis, keepdims=True)


class HeadState:
    """A fitted head over `n_labels` label slots (a slot with no training rows scores 0)."""

    def __init__(self, X, y, w, n_labels, spec):
        self.n_labels, self.spec = n_labels, spec
        self.kind = spec["head"]
        self.coef = self.intercept = self.classes = None
        if self.kind in ("lr", "avg", "lrproto"):
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=spec["C"], max_iter=3000)
            if np.allclose(w, 1.0):
                clf.fit(X, y)
            else:
                clf.fit(X, y, sample_weight=w)
            self.coef = clf.coef_.astype(np.float32)
            self.intercept = clf.intercept_.astype(np.float32)
            self.classes = clf.classes_
            if len(self.classes) == 2:  # sklearn folds two classes into one row
                self.coef = np.vstack([-self.coef[0], self.coef[0]]) / 2
                self.intercept = np.array([-self.intercept[0], self.intercept[0]]) / 2
        if self.kind in ("knn", "avg"):
            seen, keep = set(), []
            texts = spec["_texts"]
            for i in range(len(y)):
                key = (texts[i], int(y[i]))
                if key not in seen:
                    seen.add(key)
                    keep.append(i)
            self.Xi, self.yi, self.wi = X[keep].astype(np.float32), y[keep], w[keep]
        if self.kind in ("proto", "lrproto"):
            protos = np.zeros((n_labels, X.shape[1]), dtype=np.float32)
            for l in np.unique(y):
                m = y == l
                v = (X[m] * w[m, None]).sum(0)
                protos[l] = v / max(np.linalg.norm(v), 1e-9)
            self.protos = protos
            self.present = np.isin(np.arange(n_labels), np.unique(y))

    def _lr(self, Q):
        logits = Q @ self.coef.T.astype(np.float64) + self.intercept
        P = softmax(logits)
        out = np.zeros((len(Q), self.n_labels))
        out[:, self.classes] = P
        return out

    def _knn(self, Q):
        k = min(self.spec["k"], len(self.yi))
        sims = Q @ self.Xi.T
        near = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ns = np.take_along_axis(sims, near, axis=1)
        nl, nw = self.yi[near], self.wi[near]
        weight = np.exp((ns - ns.max(axis=1, keepdims=True)) / TAU) * nw
        share = np.zeros((len(Q), self.n_labels))
        best = np.zeros((len(Q), self.n_labels))
        rows = np.repeat(np.arange(len(Q)), k)
        np.add.at(share, (rows, nl.ravel()), weight.ravel())
        np.maximum.at(best, (rows, nl.ravel()), ns.ravel())
        share /= share.sum(axis=1, keepdims=True)
        return share * np.clip(best, 0, 1)

    def _proto(self, Q):
        sims = Q @ self.protos.T.astype(np.float64)
        sims[:, ~self.present] = -1e9
        return softmax(sims / TAU)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        if self.kind == "lr":
            return self._lr(Q)
        if self.kind == "knn":
            return self._knn(Q)
        if self.kind == "proto":
            return self._proto(Q)
        if self.kind == "avg":
            return (self._lr(Q) + self._knn(Q)) / 2
        if self.kind == "lrproto":
            return (self._lr(Q) + self._proto(Q)) / 2
        raise ValueError(self.kind)

    def nbytes(self) -> int:
        total = 0
        if self.coef is not None:
            total += self.coef.nbytes + self.intercept.nbytes
        if self.kind in ("knn", "avg"):
            total += self.Xi.nbytes + self.yi.astype(np.int32).nbytes
        if self.kind in ("proto", "lrproto"):
            total += self.protos.nbytes
        return total


def build_rows(W: Workflow, spec: dict):
    """Training texts, label ids, weights and (for the context-name scope) the context of each row."""
    texts = list(W.texts)
    y = list(W.y)
    w = [1.0] * len(texts)
    if spec.get("seed_weight"):
        w = [spec["seed_weight"] if s else 1.0 for s in W.is_seed]
    if spec.get("desc"):
        for l, label in enumerate(W.labels):
            if W.reserved[l]:
                continue
            texts.append(humanise(label)); y.append(l); w.append(1.0)
            d = first_sentence(W.meta.get(label, {}).get("doc", ""))
            if d:
                texts.append(d); y.append(l); w.append(1.0)
    if spec.get("mask"):
        texts = [mask_values(t) for t in texts]
    return texts, np.array(y), np.array(w)


def context_label(scope: str) -> str:
    return "global" if scope == "*" else re.sub(r"(?<=[a-z])(?=[A-Z])", " ", scope).lower()


class Model:
    """A variant fitted on one workflow: embeds, fits and answers a score matrix (cases x labels)."""

    def __init__(self, W: Workflow, spec: dict, enc: Encoder):
        self.W, self.spec, self.enc = W, spec, enc
        self.states: dict = {}
        texts, y, w = build_rows(W, spec)
        n_real = len(W.labels)
        # merge: labels that share a command name train as one label, mapped back at decision time
        self.group = np.arange(n_real)
        if spec.get("merge"):
            for ids in W.shared.values():
                for i in ids:
                    self.group[i] = ids[0]
        yg = self.group[y]
        scope = spec.get("scope", "workflow")
        if scope == "ctxprefix":
            T, Y, Wt = [], [], []
            for s in W.scopes:
                inside = W.mask_bool[s] | W.reserved
                for t, l, wt in zip(texts, yg, w):
                    orig_ok = inside[l] if not spec.get("merge") else any(inside[o] for o in np.flatnonzero(self.group == l))
                    if orig_ok:
                        T.append(f"{context_label(s)}: {t}"); Y.append(l); Wt.append(wt)
            texts, yg, w = T, np.array(Y), np.array(Wt)
        self.X = enc.embed(texts)
        self.rows_text, self.rows_y, self.rows_w = texts, yg, w
        spec = {**spec, "_texts": texts}
        self.spec = spec
        if scope == "percontext":
            for s in W.scopes:
                inside = W.mask_bool[s] | W.reserved
                if spec.get("merge"):
                    ok = np.array([any(inside[o] for o in np.flatnonzero(self.group == l)) for l in yg])
                else:
                    ok = inside[yg]
                Xs, ys, ws, ts = self.X[ok], yg[ok], w[ok], [t for t, k in zip(texts, ok) if k]
                if spec.get("foreign_seeds") and WILDCARD in W.index:
                    # the product's sparse "not here" class: one seed of every command outside the mask, as wildcard
                    seed_rows = [i for i in range(len(W.texts)) if W.is_seed[i]]
                    taken = set()
                    extra = []
                    for i in seed_rows:
                        l = int(W.y[i])
                        if not inside[l] and l not in taken:
                            taken.add(l)
                            extra.append(i)
                    if extra:
                        Xs = np.vstack([Xs, self.X[extra]])
                        ys = np.concatenate([ys, np.full(len(extra), W.index[WILDCARD])])
                        ws = np.concatenate([ws, np.ones(len(extra))])
                        ts = ts + [texts[i] + "\u0000foreign" for i in extra]
                sub_spec = {**spec, "_texts": ts}
                self.states[s] = HeadState(Xs, ys, ws, n_real, sub_spec)
        else:
            self.states[None] = HeadState(self.X, yg, w, n_real, spec)

    def case_texts(self, cases):
        out = []
        for c in cases:
            t = mask_values(c["utterance"]) if self.spec.get("mask") else c["utterance"]
            if self.spec.get("scope") == "ctxprefix":
                t = f"{context_label(c['context'])}: {t}"
            out.append(t)
        return out

    def scores(self, cases, Q=None) -> np.ndarray:
        if Q is None:
            Q = self.enc.embed(self.case_texts(cases))
        n_real = len(self.W.labels)
        S = np.zeros((len(cases), n_real))
        if self.spec.get("scope") == "percontext":
            ctxs = np.array([c["context"] for c in cases])
            for s, st in self.states.items():
                sel = np.flatnonzero(ctxs == s)
                if len(sel):
                    S[sel] = st.predict(Q[sel])
        else:
            S = self.states[None].predict(Q)
        if self.spec.get("merge"):  # a merged label's score goes to each member
            S = S[:, self.group]
        return S

    def nbytes(self) -> int:
        return sum(st.nbytes() for st in self.states.values())

    def head_ms(self, Q: np.ndarray, repeat: int = 40) -> float:
        """Median time of the head alone for one request (the first case's vector)."""
        q, ctx = Q[:1], self.W.cases[0]["context"]
        st = self.states.get(ctx) or self.states.get(None)
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            st.predict(q)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))


# --------------------------------------------------------------------------- scoring


def evaluate(W: Workflow, S: np.ndarray, cases=None) -> dict:
    """Per-case arrays for one workflow (pooled later)."""
    cases = W.cases if cases is None else cases
    out = {k: [] for k in ("first", "rank", "conf", "correct", "conf_in", "correct_in",
                           "passup", "ancestor", "esc_best_wc", "ctx", "expected", "pred_in", "best_all", "text",
                           "r_pin", "r_pout", "r_best", "e_pin", "e_pout", "e_best", "e_ctx")}
    for c, row in zip(cases, S):
        mask = W.mask_bool[c["context"]]
        inside = np.flatnonzero(mask & ~W.reserved)
        best_all = int(np.argmax(row))
        outside = np.flatnonzero(~mask & ~W.reserved)
        pin = float(row[inside].max()) if len(inside) else 0.0
        pout = float(row[outside].max()) if len(outside) else 0.0
        bl = W.labels[best_all]
        bkind = 1 if bl == PARAMETER_VALUE else 2 if bl == WILDCARD else 0
        if c["kind"] == "routing":
            out["r_pin"].append(pin); out["r_pout"].append(pout); out["r_best"].append(bkind)
            e = W.index[c["expected_label"]]
            top = inside[int(np.argmax(row[inside]))]
            ranks = int((row[inside] >= row[e]).sum())  # ties count against the variant (the label itself is one)
            out["first"].append(top == e)
            out["rank"].append(ranks)
            out["conf"].append(float(row[best_all]))
            out["correct"].append(best_all == e)
            out["conf_in"].append(float(row[top]))
            out["correct_in"].append(top == e)
            out["ctx"].append(c["context"]); out["expected"].append(e); out["pred_in"].append(int(top))
            out["best_all"].append(best_all); out["text"].append(c["utterance"])
        else:
            out["e_pin"].append(pin); out["e_pout"].append(pout); out["e_best"].append(bkind); out["e_ctx"].append(c["context"])
            anc = c.get("expected_ancestor_command") or ""
            lab = W.labels[best_all]
            out["passup"].append(not (mask[best_all] and not W.reserved[best_all]))
            out["ancestor"].append(lab == anc or norm_name(lab) == norm_name(anc))
            out["esc_best_wc"].append(lab == WILDCARD)
    return out


def pool(parts: list[dict]) -> dict:
    return {k: sum((p[k] for p in parts), []) for k in parts[0]}


def committed_wrong(conf, correct, share: float) -> float:
    n = len(conf)
    if not n:
        return 0.0
    order = np.argsort(-np.asarray(conf), kind="stable")
    take = int(math.ceil(share * n))
    return float((~np.asarray(correct, dtype=bool)[order[:take]]).sum() / n)


def summarise(e: dict) -> dict:
    n = len(e["first"])
    rank = np.array(e["rank"])
    m = len(e["passup"])
    return {
        "routing": n, "escalation": m,
        "first": float(np.mean(e["first"])),
        "r3": float((rank <= 3).mean()), "r5": float((rank <= 5).mean()), "r10": float((rank <= 10).mean()),
        "passup": float(np.mean(e["passup"])) if m else None,
        "ancestor": float(np.mean(e["ancestor"])) if m else None,
        "wrong50": committed_wrong(e["conf"], e["correct"], 0.5),
        "wrong70": committed_wrong(e["conf"], e["correct"], 0.7),
        "wrong50_in": committed_wrong(e["conf_in"], e["correct_in"], 0.5),
        "wrong70_in": committed_wrong(e["conf_in"], e["correct_in"], 0.7),
    }


# --------------------------------------------------------------------------- calibration (per workflow, per context)

FOLDS = 5
MIN_CONTEXT_ROWS = 30  # distinct rows a context needs before it gets its own threshold and margin
FLOOR = 0.5


def _fold_groups(texts):
    return np.unique(np.array(texts, dtype=object).astype(str), return_inverse=True)[1]


def _crossfit(model: "Model", idx: np.ndarray):
    """Cross-fitted score rows (five folds, grouped by row text) for the training rows `idx` with the head
    the variant uses over those rows."""
    from sklearn.model_selection import GroupKFold

    texts = [model.rows_text[i] for i in idx]
    y, w, X = model.rows_y[idx], model.rows_w[idx], model.X[idx]
    groups = _fold_groups(texts)
    folds = min(FOLDS, int(groups.max()) + 1)
    P = np.zeros((len(idx), len(model.W.labels)))
    for train, test in GroupKFold(n_splits=folds).split(X, y, groups):
        st = HeadState(X[train], y[train], w[train], len(model.W.labels), {**model.spec, "_texts": [texts[i] for i in train]})
        P[test] = st.predict(X[test])
    return P


def context_records(model: "Model") -> dict:
    """{context: records of its own distinct rows}, from cross-fitted scores. Each distinct row text is counted
    in every context whose mask holds its label; reserved labels' rows are not counted. `rows` is the count."""
    W = model.W
    n_real = len(W.labels)
    first_of = {}
    for i, t in enumerate(model.rows_text):
        first_of.setdefault(t, i)
    distinct = np.array(sorted(first_of.values()))
    distinct = distinct[~W.reserved[model.rows_y[distinct]]]
    out = {}
    shared_P = None
    for s in W.scopes:
        mask = W.mask_bool[s]
        idx = distinct[mask[model.rows_y[distinct]]]
        if not len(idx):
            continue
        if model.spec.get("scope") == "percontext":
            inside_all = mask | W.reserved
            ctx_rows = np.flatnonzero(inside_all[model.rows_y])
            P_ctx = _crossfit(model, ctx_rows)
            where = {int(r): k for k, r in enumerate(ctx_rows)}
            P = P_ctx[[where[int(i)] for i in idx]]
        else:
            if shared_P is None:
                shared_P = _crossfit(model, np.arange(len(model.rows_text)))
            P = shared_P[idx]
        inside = np.flatnonzero(mask & ~W.reserved)
        outside = np.flatnonzero(~mask & ~W.reserved)
        top = inside[np.argmax(P[:, inside], axis=1)]
        p_in = P[np.arange(len(idx)), top]
        p_out = P[:, outside].max(axis=1) if len(outside) else np.zeros(len(idx))
        best = np.argmax(P, axis=1)
        out[s] = {"p_in": p_in, "gap": p_out - p_in, "right": top == model.rows_y[idx],
                  "best": np.array([1 if W.labels[b] == PARAMETER_VALUE else 2 if W.labels[b] == WILDCARD else 0 for b in best]),
                  "rows": len(idx)}
    return out


def calibrate(rec: dict, wrong: float, esc: float = 0.005) -> tuple[float, float]:
    """The product's rule until fix-o5wi.9 (``embedding_head.calibrate``; see ``head_rule_as_experimented.py``): the margin is the smallest at which at most `esc` of
    the rows are sent up, the threshold the lowest at which wrong lone answers are at most `wrong` of the
    rows, never below 0.5."""
    count = len(rec["p_in"])
    gap, p_in, right, best = rec["gap"], rec["p_in"], rec["right"], rec["best"]
    gaps = np.sort(gap[(gap > 0) & (best != 1)])[::-1]
    allowed = int(esc * count)
    margin = float(gaps[allowed] + 1e-9) if len(gaps) > allowed else 0.0
    ahead = gap <= 0
    wrong_p = np.sort(p_in[ahead & ~right])
    threshold = 1.0
    for cand in sorted({round(float(x), 3) for x in p_in[ahead]}):
        above = len(wrong_p) - int(np.searchsorted(wrong_p, cand, side="left"))
        if above / count <= wrong:
            threshold = float(cand)
            break
    return max(threshold, FLOOR), margin


def _concat(recs: list[dict]) -> dict:
    return {k: np.concatenate([r[k] for r in recs]) for k in ("p_in", "gap", "right", "best")}


def calibration_tables(records: dict, wrong: float, esc: float = 0.005):
    """(workflow-wide (threshold, margin), {context: (threshold, margin)}, contexts that fell back)."""
    wide = calibrate(_concat(list(records.values())), wrong, esc)
    per, fell = {"__default__": wide[0]}, {"__default__": wide[1]}
    pm = {}
    fallbacks = 0
    for s, rec in records.items():
        if rec["rows"] < MIN_CONTEXT_ROWS:
            per[s], pm[s] = wide
            fallbacks += 1
        else:
            per[s], pm[s] = calibrate(rec, wrong, esc)
    pm["__default__"] = wide[1]
    return wide, per, pm, fallbacks


# --------------------------------------------------------------------------- operating points (Part 3)

INF = float("inf")
TARGETS = (0.005, 0.01, 0.02, 0.03, 0.05)
#: right-outright shares at which variants are compared by their wrong confident answers
COMPARE_AT = {"examples": (0.5, 0.6), "large": (0.2, 0.3)}


def _per_case(v, ctxs):
    """A scalar stays a scalar; a {context: value} map (with "__default__") becomes one value per case."""
    if isinstance(v, dict):
        return np.array([v.get(c, v["__default__"]) for c in ctxs], dtype=float)
    return v


def op_counts(e: dict, thr, margin, never: bool = False) -> dict:
    """The product's decision rule until fix-o5wi.9 (``embedding_head.decide``; see ``head_rule_as_experimented.py``) on one workflow's (or a pool's) cases, counted.

    best label overall is a bare value: that one answer (wrong for a routing request); the workflow wildcard:
    "not here" alone (a pass-up); else with p_in the best inside the mask and p_out the best outside it, a
    pass-up when p_out - p_in >= margin, one command when p_in >= thr and p_in >= p_out, else the top three.
    ``thr`` and ``margin`` are numbers or {context: number} maps. ``never`` (or margin = inf) never commits a
    pass-up on the local model's word (a wildcard-best request is shown the top three instead)."""
    never = never or (not isinstance(margin, dict) and margin == INF)
    pin, pout, best = (np.array(e[k]) for k in ("r_pin", "r_pout", "r_best"))
    first, rank = np.array(e["first"]), np.array(e["rank"])
    rt, rm = _per_case(thr, e["ctx"]), _per_case(margin, e["ctx"])
    passup = np.zeros(len(pin), bool) if never else ((best == 2) | ((best == 0) & (pout > pin) & (pout - pin >= rm)))
    bare = best == 1
    one_cmd = (best == 0) & ~passup & (pout <= pin) & (pin >= rt)
    right = one_cmd & first
    lone = passup | bare | one_cmd
    wrong = lone & ~right
    asked = ~lone
    n = max(len(pin), 1)
    ep, eo, eb = (np.array(e[k]) for k in ("e_pin", "e_pout", "e_best"))
    ne = len(ep)
    if not ne:
        esc = None
    elif never:
        esc = 0
    else:
        em = _per_case(margin, e["e_ctx"])
        esc = int(((eb == 2) | ((eb == 0) & (eo > ep) & (eo - ep >= em))).sum())
    return {"n": n, "right": int(right.sum()), "wrong": int(wrong.sum()), "wrong_passup": int((passup & ~right).sum()),
            "asked_top3": int((asked & (rank <= 3)).sum()), "asked_other": int((asked & (rank > 3)).sum()),
            "esc": esc, "esc_n": ne, "_passup_ctx": [c for c, k in zip(e["ctx"], passup & ~right) if k]}


def op_row(c: dict, how: str, thr: float, margin: float) -> dict:
    n = c["n"]
    return {"how": how, "threshold": thr, "margin": None if margin == INF else margin,
            "right": c["right"] / n, "wrong": c["wrong"] / n, "wrong_passup": c["wrong_passup"] / n,
            "asked_top3": c["asked_top3"] / n, "asked_other": c["asked_other"] / n,
            "esc_recall": None if c["esc"] is None else c["esc"] / max(c["esc_n"], 1)}


def concentration(ctxs: list[str]) -> dict:
    """How the wrong pass-ups spread over contexts, with no names: sorted counts, and how many contexts hold
    half and 80% of them."""
    counts = sorted(Counter(ctxs).values(), reverse=True)
    total = sum(counts)
    out = {"total": total, "contexts_with_any": len(counts), "counts": counts}
    for share in (0.5, 0.8):
        run, k = 0, 0
        for c in counts:
            if run >= share * total:
                break
            run += c; k += 1
        out[f"contexts_for_{int(share * 100)}pct"] = k
    return out


KEEP_LEVELS = (0.9, 0.75, 0.5)  # rows that keep pass-ups answer at least this share of the never-pass-up row's right answers (first level that is reachable)


def choose_point(e: dict, target: float, allow_passup: bool = True):
    """The (threshold, margin) chosen on the cases themselves (in-sample).

    Never passing up alone (`allow_passup` False): the point answering the most requests right outright with
    all wrong confident answers within `target` of all routing requests (a bare value answered where a command
    was meant counts as wrong). Passing up kept: among points within the same budget (a wrong pass-up counts)
    that answer at least the first reachable KEEP_LEVELS share of that point's right answers, the one handing on the most escalation
    cases, then answering the most right."""
    pin = np.array(e["r_pin"])
    thr_grid = np.unique(np.concatenate([[0.0], np.round(pin, 6), [1.01]]))
    thr_grid = thr_grid[::max(1, len(thr_grid) // 400)]
    gaps = np.array(e["r_pout"]) - pin
    egaps = np.array(e["e_pout"]) - np.array(e["e_pin"])
    cand = np.unique(np.round(np.concatenate([gaps[gaps > 0], egaps[egaps > 0]]), 4))
    if len(cand) > 80:
        cand = np.unique(np.quantile(cand, np.linspace(0, 1, 80)))

    def best_of(margins, need_right, rank_esc):
        best = None
        for m in margins:
            for t in thr_grid:
                c = op_counts(e, float(t), m)
                if c["wrong"] / c["n"] > target + 1e-12 or c["right"] < need_right:
                    continue
                key = ((c["esc"] or 0, c["right"]) if rank_esc else (c["right"], 0)) + (-float(t),)
                if best is None or key > best[0]:
                    best = (key, float(t), m, c)
        return best

    base = best_of([INF], 0, False)
    if not allow_passup or base is None:
        return base if not allow_passup else None
    for keep in KEEP_LEVELS:
        found = best_of([0.0, *cand.tolist()], keep * base[3]["right"], True)
        if found:
            return (*found, keep)
    return None


def _sum_counts(cs: list[dict]) -> dict:
    tot = {k: sum(c[k] or 0 for c in cs) for k in cs[0] if not k.startswith("_")}
    tot["esc"] = tot["esc"] if tot["esc_n"] else None
    tot["_passup_ctx"] = [x for c in cs for x in c["_passup_ctx"]]
    return tot


def calibrated_rows(evs: list[dict], records: list[dict]) -> list[dict]:
    """Operating points from the product's cross-fit rule, hitting each wrong-answer target on the training
    rows and then scored on the development cases: one threshold and margin per workflow (what the product
    does), one pair per context (workflow-wide pair for a context with fewer than MIN_CONTEXT_ROWS rows), and
    the per-context thresholds with pass-up never committed alone. Out of sample: the cases were not used."""
    rows = []
    for tgt in TARGETS:
        fams = {"one pair per workflow": [], "one pair per context": [], "per-context thresholds, never pass up alone": []}
        notes = {k: [] for k in fams}
        for ev, rec in zip(evs, records):
            wide, per, pm, fb = calibration_tables(rec, tgt)
            fams["one pair per workflow"].append(op_counts(ev, wide[0], wide[1]))
            fams["one pair per context"].append(op_counts(ev, per, pm))
            fams["per-context thresholds, never pass up alone"].append(op_counts(ev, per, pm, never=True))
            own = [v for k, v in per.items() if k != "__default__" and rec[k]["rows"] >= MIN_CONTEXT_ROWS]
            notes["one pair per workflow"].append(f"{wide[0]:.2f}/{wide[1]:.2f}")
            notes["one pair per context"].append(f"thr {min(own):.2f}-{max(own):.2f}, {len(own)} own, {fb} fell back" if own else "all fell back")
            notes["per-context thresholds, never pass up alone"].append(notes["one pair per context"][-1])
        for fam, cs in fams.items():
            if fam.startswith("per-context") and tgt not in (0.01, 0.03):
                continue
            tot = _sum_counts(cs)
            r = op_row(tot, f"{fam}, target {100 * tgt:g}%", 0.0, INF if fam.endswith("alone") else 0.0)
            r["calib"] = "; ".join(notes[fam])
            r["passup_spread"] = concentration(tot["_passup_ctx"])
            rows.append(r)
    return rows


def frontier(e: dict) -> list[tuple[float, float]]:
    """(right share, wrong share) pairs reachable by any (threshold, margin): the efficient frontier."""
    pin = np.array(e["r_pin"])
    thr_grid = np.unique(np.round(pin, 6))
    gaps = np.array(e["r_pout"]) - pin
    cand = np.unique(np.round(gaps[gaps > 0], 4))
    if len(cand) > 60:
        cand = np.unique(np.quantile(cand, np.linspace(0, 1, 60)))
    pts = []
    for m in [0.0, *cand.tolist(), INF]:
        for t in thr_grid[::max(1, len(thr_grid) // 250)]:
            c = op_counts(e, float(t), m)
            pts.append((c["right"] / c["n"], c["wrong"] / c["n"]))
    return pts


def wrong_at_right(pts, share: float):
    """Least wrong-confident share among points answering at least `share` right outright; None if unreachable."""
    ok = [w for r, w in pts if r >= share - 1e-9]
    return min(ok) if ok else None


def operating_table(e: dict, parts: list[dict] | None, shipped_op: list[tuple[float, float]] | None) -> list[dict]:
    rows = []
    if shipped_op and parts:
        cs = [op_counts(p, t, m) for p, (t, m) in zip(parts, shipped_op)]
        tot = _sum_counts(cs)
        row = op_row(tot, "the shipped operating point", shipped_op[-1][0], shipped_op[-1][1])
        row["passup_spread"] = concentration(tot["_passup_ctx"])
        rows.append(row)
    for tgt in TARGETS:
        b = choose_point(e, tgt)
        if b:
            rows.append({**op_row(b[3], f"target {100 * tgt:g}% wrong, pass-ups kept (keeps {100 * b[4]:g}% of the right answers)", b[1], b[2])})
        else:
            rows.append({"how": f"target {100 * tgt:g}% wrong, pass-ups kept", "unreachable": True})
    for tgt in (0.01, 0.03):
        b = choose_point(e, tgt, allow_passup=False)
        if b:
            rows.append(op_row(b[3], f"never pass up alone, target {100 * tgt:g}%", b[1], b[2]))
    return rows


# --------------------------------------------------------------------------- variants


def base_spec(**kw) -> dict:
    spec = {"enc": SHIPPED, "head": "lr", "C": 100.0, "k": 10}
    spec.update(kw)
    return spec


def variant_table() -> dict:
    v: dict[str, dict] = {"shipped": base_spec()}
    short = lambda m: m.split("/")[-1]
    for m in ENCODERS:
        if m != SHIPPED:
            v[f"enc_{short(m)}"] = base_spec(enc=m)
    for c in (10, 1000):
        v[f"lr_C{c}"] = base_spec(C=float(c))
    for k in (5, 10):
        v[f"knn_k{k}"] = base_spec(head="knn", k=k)
    v["lr_knn5"] = base_spec(head="avg", k=5)
    v["lr_knn10"] = base_spec(head="avg", k=10)
    v["proto"] = base_spec(head="proto")
    v["lr_proto"] = base_spec(head="lrproto")
    v["data_desc"] = base_spec(desc=True)
    v["data_seed3"] = base_spec(seed_weight=3.0)
    v["data_mask"] = base_spec(mask=True)
    v["data_desc_seed3"] = base_spec(desc=True, seed_weight=3.0)
    v["data_all3"] = base_spec(desc=True, seed_weight=3.0, mask=True)
    v["scope_percontext"] = base_spec(scope="percontext")
    v["scope_percontext_fs"] = base_spec(scope="percontext", foreign_seeds=True)
    v["scope_ctxprefix"] = base_spec(scope="ctxprefix")
    v["scope_merge"] = base_spec(merge=True)
    v["finetune"] = base_spec(finetune=True)
    v["finetune_repeat"] = base_spec(finetune=True, repeat=True)
    base = "BAAI/bge-base-en-v1.5"
    v["finetune_base"] = base_spec(enc=base, finetune=True)
    v["finetune_base_repeat"] = base_spec(enc=base, finetune=True, repeat=True)
    v["finetune_last2"] = base_spec(finetune=True, train_layers=2)
    v["finetune_last1"] = base_spec(finetune=True, train_layers=1)
    v["finetune_base_last2"] = base_spec(enc=base, finetune=True, train_layers=2)
    v["project"] = base_spec(project=True)
    v["project_base"] = base_spec(enc=base, project=True)
    return v


def extra_variants(path: str | None) -> dict:
    """Combinations are listed in a JSON file ({name: spec}) so they can be chosen after the single levers."""
    if not path or not os.path.exists(path):
        return {}
    return {n: base_spec(**s) for n, s in json.load(open(path)).items()}


# --------------------------------------------------------------------------- fine-tuning


def finetune_encoder(W: Workflow, spec: dict, cache: str, seed: int = 0, epochs: int = 4, batch: int = 32):
    """Classification fine-tuning of the encoder on the workflow's training rows: a cosine classifier over every
    label, initialised at the class prototypes, trained with cross-entropy (scale 20). Returns the Encoder and
    the training seconds. Fixed seed, deterministic algorithms."""
    import torch
    from transformers import AutoModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    texts, y, _w = build_rows(W, spec)
    enc = Encoder(spec["enc"], None)
    enc._model = AutoModel.from_pretrained(spec["enc"])
    enc._load()
    model = enc._model
    base = enc.embed(texts, use_cache=False)
    protos = np.zeros((len(W.labels), base.shape[1]), dtype=np.float32)
    for l in np.unique(y):
        v = base[y == l].mean(0)
        protos[l] = v / max(np.linalg.norm(v), 1e-9)
    head = torch.nn.Parameter(torch.tensor(protos))
    layers = spec.get("train_layers")
    if layers:  # only the last `layers` transformer layers are trained (and would be stored per workflow)
        for prm in model.parameters():
            prm.requires_grad = False
        for prm in model.encoder.layer[-layers:].parameters():
            prm.requires_grad = True
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    model.train()
    opt = torch.optim.AdamW([{"params": trainable, "lr": 3e-5}, {"params": [head], "lr": 1e-3}], weight_decay=0.01)
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    yt = torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(texts), generator=gen).tolist()
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            tokens = enc.tokens([texts[i] for i in idx])
            vec = enc.pool(model(**tokens), tokens)
            logits = 20.0 * vec @ torch.nn.functional.normalize(head, dim=-1).T
            loss = torch.nn.functional.cross_entropy(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    seconds = time.perf_counter() - t0
    model.eval()
    torch.use_deterministic_algorithms(False)
    enc.tag = f"ft-{W.name}-{seed}-{time.time_ns()}"
    enc.train_info = {"device": str(next(model.parameters()).device),
                      "trained_bytes": 4 * sum(prm.numel() for prm in trainable),
                      "encoder_bytes": 4 * sum(prm.numel() for prm in model.parameters())}
    return enc, seconds


class ProjectedEncoder:
    """A frozen encoder followed by a small trained linear projection (d x d, started at the identity), the
    output re-normalised. Only the projection would be stored per workflow."""

    def __init__(self, base: Encoder, matrix: np.ndarray):
        self.base, self.matrix = base, matrix.astype(np.float32)
        self.model_id = base.model_id

    def embed(self, texts, use_cache: bool = True) -> np.ndarray:
        v = self.base.embed(texts, use_cache=use_cache) @ self.matrix.T
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def train_projection(W: Workflow, spec: dict, base: Encoder, seed: int = 0, epochs: int = 30, batch: int = 64):
    """A d x d projection on the frozen embeddings, trained like the fine-tuning (cosine classifier at the class
    prototypes, scale 20) with the same seed. Returns the ProjectedEncoder and the training seconds."""
    import torch

    torch.manual_seed(seed)
    texts, y, _w = build_rows(W, spec)
    X = torch.tensor(base.embed(texts))
    d = X.shape[1]
    P = torch.nn.Parameter(torch.eye(d))
    protos = torch.stack([torch.nn.functional.normalize(X[torch.tensor(y == l)].mean(0), dim=0) if (y == l).any() else torch.zeros(d)
                          for l in range(len(W.labels))])
    head = torch.nn.Parameter(protos.clone())
    opt = torch.optim.AdamW([{"params": [P], "lr": 1e-3}, {"params": [head], "lr": 1e-3}], weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    yt = torch.tensor(y)
    t0 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(len(texts), generator=gen)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            vec = torch.nn.functional.normalize(X[idx] @ P.T, dim=-1)
            loss = torch.nn.functional.cross_entropy(20.0 * vec @ torch.nn.functional.normalize(head, dim=-1).T, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc = ProjectedEncoder(base, P.detach().numpy())
    enc.train_info = {"device": "cpu", "trained_bytes": 4 * d * d, "encoder_bytes": 0}
    return enc, time.perf_counter() - t0


# --------------------------------------------------------------------------- run


def load_groups(args):
    wfs = {}
    for spec in args.wf:
        name, folder = spec.split("=", 1)
        wfs[name] = Workflow(name, folder)
    groups = {}
    for spec in args.group:
        name, members = spec.split("=", 1)
        groups[name] = [wfs[m] for m in members.split(",")]
    return wfs, groups


def run_variant(name: str, spec: dict, wfs: dict, groups: dict, out: str) -> dict:
    cache = os.path.join(out, "emb")
    result = {"spec": {k: v for k, v in spec.items() if not k.startswith("_")}, "groups": {}, "per_workflow": {}}
    parts = {g: [] for g in groups}
    shipped = SHIPPED_OP if name == "shipped" else {}
    records_by_wf: dict = {}
    evs_by_wf: dict = {}
    for wname, W in wfs.items():
        t0 = time.perf_counter()
        info = {}
        if spec.get("finetune"):
            enc, secs = finetune_encoder(W, spec, cache)
            info["train_seconds"] = secs
            info.update(enc.train_info)
        elif spec.get("project"):
            enc, secs = train_projection(W, spec, encoder_for(spec["enc"], cache))
            info["train_seconds"] = secs
            info.update(enc.train_info)
        else:
            enc = encoder_for(spec["enc"], cache)
        model = Model(W, spec, enc)
        S = model.scores(W.cases)
        if spec.get("repeat"):
            enc2, secs2 = finetune_encoder(W, spec, cache) if spec.get("finetune") else train_projection(W, spec, encoder_for(spec["enc"], cache))
            S2 = Model(W, spec, enc2).scores(W.cases)
            info["repeat_same_top"] = float(np.mean(S.argmax(1) == S2.argmax(1)))
            info["repeat_max_abs_diff"] = float(np.abs(S - S2).max())
            info["train_seconds_2"] = secs2
        ev = evaluate(W, S)
        evs_by_wf[wname] = ev
        if name in CALIBRATE:
            records_by_wf[wname] = context_records(model)
            wide, per, pm, fb = calibration_tables(records_by_wf[wname], 0.01)
            info["calibration_at_1pct"] = {"workflow_wide": [wide[0], wide[1]], "contexts_own": len(per) - 1 - fb, "contexts_fallback": fb}
            info["bytes_with_context_thresholds"] = model.nbytes() + 8 * len(W.scopes)
        Q = model.enc.embed(model.case_texts(W.cases))
        info.update({"bytes": model.nbytes(), "head_ms": model.head_ms(Q), "seconds": time.perf_counter() - t0})
        result["per_workflow"][wname] = {**summarise(ev), **info}
        for g, members in groups.items():
            if W in members:
                parts[g].append(ev)
        np.save(os.path.join(out, "scores", f"{name}__{wname}.npy"), S)
    for g, evs in parts.items():
        s = summarise(pool(evs))
        s["bytes"] = float(np.mean([result["per_workflow"][w.name]["bytes"] for w in groups[g]]))
        s["head_ms"] = float(np.mean([result["per_workflow"][w.name]["head_ms"] for w in groups[g]]))
        pooled = pool(evs)
        pts = frontier(pooled)
        s["wrong_at_right"] = {f"{r:g}": wrong_at_right(pts, r) for r in COMPARE_AT.get(g, (0.2, 0.5))}
        s["max_right"] = max(r for r, _ in pts)
        ship = [shipped[w.name] for w in groups[g]] if shipped and all(w.name in shipped for w in groups[g]) else None
        s["operating_points"] = operating_table(pooled, evs, ship)
        if name in CALIBRATE:
            s["calibrated_points"] = calibrated_rows([evs_by_wf[w.name] for w in groups[g]], [records_by_wf[w.name] for w in groups[g]])
        result["groups"][g] = s
    return result


def cmd_sweep(args):
    wfs, groups = load_groups(args)
    CALIBRATE.update((args.calibrate or "").split(","))
    for spec in args.shipped_op or []:
        name, tm = spec.split("=", 1)
        SHIPPED_OP[name] = tuple(float(x) for x in tm.split(","))
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "scores"), exist_ok=True)
    table = {**variant_table(), **extra_variants(args.combos)}
    names = list(table) if args.variants == "all" else args.variants.split(",")
    # group by encoder so one model is in memory at a time
    names.sort(key=lambda n: (table[n]["enc"], n))
    for n in names:
        path = os.path.join(args.out, "results", f"{n}.json")
        if os.path.exists(path) and not args.force:
            print("skip", n)
            continue
        t0 = time.perf_counter()
        res = run_variant(n, table[n], wfs, groups, args.out)
        json.dump(res, open(path, "w"), indent=1)
        print(f"{n}: " + "; ".join(f"{g} first {s['first']:.3f} r3 {s['r3']:.3f}" for g, s in res["groups"].items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


# --------------------------------------------------------------------------- latency


def cmd_latency(args):
    import torch
    from threadpoolctl import threadpool_limits

    wfs, _ = load_groups(args)
    W = next(iter(wfs.values()))
    requests = [c["utterance"] for c in W.cases[:30]]
    out = {}
    for m in ENCODERS:
        enc = Encoder(m, None)
        enc._load()
        row = {}
        for threads in (10, 1):
            torch.set_num_threads(threads)
            times = []
            with threadpool_limits(limits=threads):
                for _ in range(2):  # warm up
                    enc.embed(requests[:2], use_cache=False)
                for rep in range(3):
                    for r in requests:
                        t0 = time.perf_counter()
                        enc.embed([r], use_cache=False)
                        times.append((time.perf_counter() - t0) * 1000)
            row[f"ms_{threads}"] = float(np.median(times))
        out[m] = row
        print(m, row, flush=True)
    json.dump(out, open(os.path.join(args.out, "latency.json"), "w"), indent=1)
    return 0


# --------------------------------------------------------------------------- diagnosis


def cmd_diagnose(args):
    wfs, groups = load_groups(args)
    enc = encoder_for(SHIPPED, os.path.join(args.out, "emb"))
    pairs = {}
    for spec in args.pairs or []:
        n, p = spec.split("=", 1)
        pairs[n] = json.load(open(p))["top10"]
    report = {}
    for gname, members in groups.items():
        rows = []
        esc = {"n": 0, "passup": 0, "ancestor": 0, "wildcard": 0}
        for W in members:
            model = Model(W, base_spec(), enc)
            S = model.scores(W.cases)
            ev = evaluate(W, S)
            n_rows = W.rows_per_label
            conf_pairs = {frozenset((p["a"], p["b"])) for p in pairs.get(W.name, [])}
            for i in range(len(ev["first"])):
                e, p = ev["expected"][i], ev["pred_in"][i]
                le, lp = W.labels[e], W.labels[p]
                ke, kp = W.kind(le), W.kind(lp)
                kind = None
                if not ev["first"][i]:
                    if ke == "navigation" or kp == "navigation":
                        kind = "d_both_navigation" if ke == kp else "d_one_navigation"
                    elif (ke == "planning") != (kp == "planning"):
                        pl, pn = (le, lp) if ke == "planning" else (lp, le)
                        kind = "c" if pn in W.meta.get(pl, {}).get("built_from", []) else "c_other"
                    elif norm_name(le) == norm_name(lp):
                        kind = "b"
                    elif owner_of(le) == owner_of(lp):
                        kind = "a"
                    else:
                        kind = "e"
                rows.append({"same_name": norm_name(le) == norm_name(lp) and le != lp, "first": bool(ev["first"][i]), "rank": ev["rank"][i], "kind": kind,
                             "n_rows": int(n_rows[e]), "value": carries_value(ev["text"][i]),
                             "pair": frozenset((le, lp)) in conf_pairs, "wf": W.name,
                             "best_outside": not (W.mask_bool[ev["ctx"][i]][ev["best_all"][i]] and not W.reserved[ev["best_all"][i]]),
                             "ctx_same_owner": owner_of(le) == owner_of(lp)})
            for k in range(len(ev["passup"])):
                esc["n"] += 1; esc["passup"] += ev["passup"][k]; esc["ancestor"] += ev["ancestor"][k]; esc["wildcard"] += ev["esc_best_wc"][k]
        n = len(rows)
        miss = [r for r in rows if not r["first"]]
        kinds = {}
        for r in miss:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        bins = [(0, 15), (16, 25), (26, 30), (31, 10 ** 6)]
        by_rows = {f"{a}-{b if b < 10 ** 6 else 'up'}": [sum(r["first"] for r in rows if a <= r["n_rows"] <= b),
                                                      sum(1 for r in rows if a <= r["n_rows"] <= b)] for a, b in bins}
        by_value = {"value": [sum(r["first"] for r in rows if r["value"]), sum(1 for r in rows if r["value"])],
                    "no_value": [sum(r["first"] for r in rows if not r["value"]), sum(1 for r in rows if not r["value"])]}
        rank = np.array([r["rank"] for r in rows])
        report[gname] = {
            "routing": n, "first": float(np.mean([r["first"] for r in rows])),
            "r3": float((rank <= 3).mean()), "r5": float((rank <= 5).mean()), "r10": float((rank <= 10).mean()),
            "misses": len(miss), "miss_kinds": kinds,
            "miss_same_command_name_other_context": sum(r["same_name"] for r in miss),
            "miss_best_overall_outside_mask": sum(r["best_outside"] for r in miss),
            "miss_in_confused_pairs": sum(r["pair"] for r in miss),
            "by_expected_rows": by_rows, "by_value": by_value,
            "escalation": {"n": esc["n"], "best_overall_outside_mask": esc["passup"], "best_is_expected_ancestor": esc["ancestor"],
                           "best_is_wildcard": esc["wildcard"]},
        }
        print(json.dumps({gname: report[gname]}, indent=1))
    json.dump(report, open(os.path.join(args.out, "diagnosis.json"), "w"), indent=1)
    return 0


# --------------------------------------------------------------------------- tables


def cmd_tables(args):
    latency = json.load(open(os.path.join(args.out, "latency.json"))) if os.path.exists(os.path.join(args.out, "latency.json")) else {}
    results = {f[:-5]: json.load(open(os.path.join(args.out, "results", f))) for f in sorted(os.listdir(os.path.join(args.out, "results")))}
    order = ["shipped"] + [n for n in results if n != "shipped"]
    f = lambda x: "-" if x is None else f"{100 * x:.1f}"
    for g in sorted({g for r in results.values() for g in r["groups"]}):
        shares = list(COMPARE_AT.get(g, (0.2, 0.5)))
        print(f"\n### {g}: ranking (sorted by wrong confident answers at {100 * shares[0]:g}% right outright)\n")
        head = " | ".join(f"Wrong @{100 * r:g}% right" for r in shares)
        print(f"| Variant | {head} | Max right | First | Top 3 | Top 5 | Top 10 | Pass-up | Ancestor | Wrong@50 | Wrong@70 | Head ms | Head bytes |")
        print("|---|" + "---|" * (len(shares) + 11))
        def key(n):
            w = results[n]["groups"][g]["wrong_at_right"].get(f"{shares[0]:g}")
            return (w is None, w if w is not None else 0, n)
        for n in sorted(results, key=key):
            s = results[n]["groups"][g]
            w = " | ".join(f(s["wrong_at_right"].get(f"{r:g}")) for r in shares)
            print(f"| {n} | {w} | {f(s['max_right'])} | {f(s['first'])} | {f(s['r3'])} | {f(s['r5'])} | {f(s['r10'])} | {f(s['passup'])} | {f(s['ancestor'])} "
                  f"| {f(s['wrong50'])} | {f(s['wrong70'])} | {s['head_ms']:.2f} | {int(s['bytes'])} |")
    for n in (args.ops.split(",") if args.ops else []):
        for g, s in results[n]["groups"].items():
            print(f"\n### Operating points: {n}, {g} (in-sample on the development cases)\n")
            print("| Point | Right outright | Wrong confident | of it wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold | Margin |")
            print("|---|---|---|---|---|---|---|---|---|")
            for r in s["operating_points"]:
                if r.get("unreachable"):
                    print(f"| {r['how']} | not reachable: no pass-up setting stays in the budget and keeps half the right answers | | | | | | | |")
                    continue
                m = "never" if r["margin"] is None else f"{r['margin']:.3f}"
                print(f"| {r['how']} | {f(r['right'])} | {f(r['wrong'])} | {f(r['wrong_passup'])} | {f(r['asked_top3'])} | {f(r['asked_other'])} | {f(r['esc_recall'])} | {r['threshold']:.3f} | {m} |")
            if s.get("calibrated_points"):
                print(f"\n#### {n}, {g}: thresholds from the cross-fit rule, scored on the development cases (out of sample)\n")
                print("| Point | Right outright | Wrong confident | of it wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |")
                print("|---|---|---|---|---|---|---|---|")
                for r in s["calibrated_points"]:
                    print(f"| {r['how']} | {f(r['right'])} | {f(r['wrong'])} | {f(r['wrong_passup'])} | {f(r['asked_top3'])} | {f(r['asked_other'])} | {f(r['esc_recall'])} | {r['calib']} |")
    if latency:
        print("\n### Encoder request latency (ms, median)\n")
        for m, row in latency.items():
            print(f"| {m} | {row['ms_10']:.1f} | {row['ms_1']:.1f} |")
    return 0


# --------------------------------------------------------------------------- meta


def cmd_meta(args):
    """Per label: seeds, docstring, planning flag and the plain commands a planning command's source names."""
    import inspect

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import single_intent_model as runner
    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry, is_visible_planning

    env = runner.environment_for(args.workflow, None, None)
    fastworkflow.init(env_vars=dict(env))
    crd = RoutingRegistry.get_definition(args.workflow, load_cached=False)
    labels = json.load(open(args.rows))["labels"]
    planning = {n for n, m in crd.command_directory.map_command_2_metadata.items() if is_visible_planning(m)}
    res = {}
    for l in labels:
        seeds, doc = [], ""
        try:
            m = crd.command_directory.get_utterance_metadata(l)
            seeds = list(m.plain_utterances) if m else []
        except Exception:
            pass
        try:
            doc = inspect.getdoc(crd.get_command_class(l, fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)) or ""
        except Exception:
            pass
        res[l] = {"seeds": seeds, "doc": doc, "planning": l in planning, "built_from": []}
    for l, v in res.items():
        if not v["planning"]:
            continue
        try:
            path = crd.command_directory.map_command_2_metadata[l].response_generation_module_path
            path = path if os.path.isabs(path) else os.path.join(args.workflow, path)
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        v["built_from"] = [q for q in res if q != l and not res[q]["planning"]
                           and re.search(r"\b" + re.escape(norm_name(q)) + r"\b", text)]
    json.dump(res, open(args.out, "w"), indent=1)
    print(len(res), "labels,", sum(v["planning"] for v in res.values()), "planning")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name, fn in (("sweep", cmd_sweep), ("diagnose", cmd_diagnose), ("latency", cmd_latency), ("tables", cmd_tables)):
        p = sub.add_parser(name)
        p.add_argument("--wf", action="append", default=[])
        p.add_argument("--group", action="append", default=[])
        p.add_argument("--pairs", action="append")
        p.add_argument("--calibrate", default="shipped,scope_percontext", help="variants that get cross-fit calibrations")
        p.add_argument("--shipped-op", action="append", help="NAME=threshold,margin of the shipped head")
        p.add_argument("--out", required=True)
        p.add_argument("--variants", default="all")
        p.add_argument("--combos", default=None, help="JSON file {name: spec} of combinations")
        p.add_argument("--force", action="store_true")
        p.add_argument("--ops", default=None, help="tables: comma list of variants whose operating-point table is printed")
        p.set_defaults(fn=fn)
    p = sub.add_parser("meta")
    p.add_argument("--workflow", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_meta)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
