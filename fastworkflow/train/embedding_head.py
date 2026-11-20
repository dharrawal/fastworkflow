"""The ``embedding`` intent model layout: a frozen sentence encoder and one linear head per workflow.

The encoder is a pre-trained sentence encoder that is never trained here. A request is embedded
once (the encoder's pooling and text prefix, L2-normalised), and ONE multinomial logistic
regression over every label of the workflow (every command once, ``parameter_value`` and the
workflow-wide ``wildcard``) turns the vector into probabilities. A context is answered by reading
those probabilities through its mask, as the ``workflow`` layout does (`train.workflow_model`),
without renormalising inside the mask.

The decision rule is the one chosen in ``docs/local_intent_model_levers.md`` (Parts 2 and 3): the
head commits to one command at a fixed probability (`intent_model_layout.DEFAULT_COMMIT_THRESHOLD`)
and NEVER passes a request up on its own. When its best command lies outside the context's mask it
is not confident: it hands the request on, naming the best outside commands beside the context's own
(see `decide`), and the next link (the remote decision model, else the user) decides. In order a run uses:

* `SentenceEncoder` loads an encoder once per process and embeds texts.
* `training_rows` adds each command's description and humanised name as rows and weights the seeds.
* `fit_embedding_head` embeds the rows, fits the head, cross-fits it for the report and writes the artifacts.
* `cross_fit_report` is the training report's numbers: cross-fit accuracy and the most confused pairs.
  It does not set the threshold.
* `decide` is the decision in one context. Training-time scoring and the runtime both call it.
* `EmbeddingHeadRouter` is the runtime reader, with `CommandRouter`'s interface.
* `EmbeddingLayoutTrainer` is what `model_pipeline_training.train` hands its per-context rows to
  when a workflow is trained in this layout.

The artifacts, in ``___command_info/versions/<id>/workflow/``:

* ``head.npz``: ``coef`` (labels x dimensions) and ``intercept`` (labels), float32.
* ``scopes.json``: the label order, the encoder id, pooling and prefix, each context's mask, the
  commit threshold, and the cross-fit facts. ``schema_version`` 2 is this decision rule. A head
  trained before it (``schema_version`` 1, with a ``margin``) is read as it always was, with its own
  threshold and pass-up margin, until the workflow is retrained.

The encoder's weights are not stored: they are the published checkpoint the id names.

Inference is numpy and the encoder: scikit-learn fits the head and is not called at run time.
`model_pipeline_training` imports this module lazily (this module imports it).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np

from fastworkflow import model_pipeline_training as heads
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from fastworkflow.train import heldout_evaluation, intent_model_layout, workflow_model
from fastworkflow.train.intent_model_layout import HEAD_FILENAME, SCOPES_FILENAME
from fastworkflow.utils.logging import logger

#: 2 is the decision rule that never passes up alone (`decide`); 1 is the older rule with a calibrated
#: pass-up margin, still read for a head trained before it.
SCOPES_SCHEMA_VERSION = 2
MAX_LENGTH = 128
EMBED_BATCH_SIZE = 64
#: The head: `sklearn.linear_model.LogisticRegression(C=HEAD_C, max_iter=HEAD_MAX_ITER)`.
HEAD_C = 100.0
HEAD_MAX_ITER = 3000
#: Folds of the cross-fit the training report is computed on. It sets nothing.
CROSS_FIT_FOLDS = 5
#: A seed utterance (one the command's author wrote) weighs this much in the fit, against 1 for a generated row.
SEED_WEIGHT = 3.0
#: Candidates returned when the answer is not confident, as the other layouts do.
TOP_K = 3
#: How many commands outside the context's mask a hand-on may carry. The runtime keeps the best of them that
#: are reachable from where the request was asked (`intent_detector.ClassifierEngine`).
OUTSIDE_K = 10
#: A confused pair is reported as "exceeds" when this share of either command's rows go to the other.
PAIR_SHARE_LIMIT = 0.10
PAIRS_REPORTED = 10
MODEL_TIER = "embedding"
RESERVED_LABELS = (PARAMETER_VALUE_LABEL, WILDCARD_LABEL)

DECISION_MATCH = "match"
DECISION_AMBIGUOUS = "ambiguous"
DECISION_NOT_MINE = "not_mine"
DECISION_BARE_VALUE = "bare_value"

#: How many single-text vectors an encoder keeps. A walk asks several contexts about one request
#: and the cache layer asks about it too; they share one forward pass.
_VECTOR_MEMO_SIZE = 256


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------


class SentenceEncoder:
    """One frozen sentence encoder, loaded once per model id per process.

    `SentenceEncoder.get(model_id, pooling, prefix)` returns the shared reader for that triple; readers of
    one model id with different pooling or prefix share the loaded weights. Every workflow and context of
    the process that names the id uses the same object. *prefix* is put before every text (``query: `` for
    the e5 family).
    """

    _models: ClassVar[dict[str, tuple]] = {}
    _instances: ClassVar[dict[tuple[str, str, str], "SentenceEncoder"]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, model_id: str, pooling: str, prefix: str = ""):
        self.model_id = model_id
        self.pooling = pooling
        self.prefix = prefix
        self.tokenizer, self.model, self.device = self._load(model_id)
        self._memo: OrderedDict[str, np.ndarray] = OrderedDict()
        self._memo_lock = threading.Lock()
        # A fast tokenizer is not safe to call from two threads at once.
        self._tokenizer_lock = threading.Lock()

    @classmethod
    def _load(cls, model_id: str) -> tuple:
        if model_id not in cls._models:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(device)
            model.eval()
            cls._models[model_id] = (tokenizer, model, device)
        return cls._models[model_id]

    @classmethod
    def get(cls, model_id: str, pooling: Optional[str] = None,
            prefix: Optional[str] = None) -> "SentenceEncoder":
        pooling = pooling or intent_model_layout.default_pooling(model_id)
        if pooling not in (intent_model_layout.POOLING_CLS, intent_model_layout.POOLING_MEAN):
            raise ValueError(f"unknown pooling {pooling!r}")
        prefix = intent_model_layout.default_prefix(model_id) if prefix is None else prefix
        key = (model_id, pooling, prefix)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_id, pooling, prefix)
            return cls._instances[key]

    def embed(self, texts: Sequence[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        """One L2-normalised vector per text, ``(len(texts), dimensions)`` float32."""
        import torch

        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = [self.prefix + str(text) for text in texts[start:start + batch_size]]
                with self._tokenizer_lock:
                    encoded = self.tokenizer(
                        batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                        return_tensors="pt")
                encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                if self.pooling == intent_model_layout.POOLING_CLS:
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"][..., None].to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    @property
    def dimensions(self) -> int:
        return int(self.model.config.hidden_size)

    def vector(self, text: str) -> np.ndarray:
        """The vector of one text, remembered for the last few texts asked."""
        with self._memo_lock:
            cached = self._memo.get(text)
            if cached is not None:
                self._memo.move_to_end(text)
                return cached
        vector = self.embed([text])[0]
        with self._memo_lock:
            self._memo[text] = vector
            while len(self._memo) > _VECTOR_MEMO_SIZE:
                self._memo.popitem(last=False)
        return vector


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------


def softmax_probabilities(vectors: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """The head's probabilities for each vector: a matrix product and a softmax."""
    logits = np.asarray(vectors, dtype=np.float64) @ coef.T.astype(np.float64) + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _logistic_regression():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=HEAD_C, max_iter=HEAD_MAX_ITER)


def _one_thread():
    """Fit on one linear-algebra thread. The matrices are small (hundreds of rows by a few hundred
    dimensions), and on them a thread pool costs far more than it saves: the library example's head
    fitted in 10.7 s on this machine's default pool and in 0.14 s on one thread."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1)


def _fit_input(vectors: np.ndarray) -> np.ndarray:
    """The vectors as the head is fitted on them: float64. scikit-learn fits in the precision it is
    given, and on float32 the optimiser stops earlier at a slightly different head (the registered
    run fitted in float64; one benchmark answer of 264 differs between the two)."""
    return np.asarray(vectors, dtype=np.float64)


def _uniform(weights: Optional[np.ndarray]) -> bool:
    return weights is None or bool(np.allclose(weights, 1.0))


def fit_head(
    vectors: np.ndarray, targets: np.ndarray, label_count: int, weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(coef, intercept)`` of the head fitted on every row, each row counting *weights* times (default
    1). Every label must have a row."""
    with _one_thread():
        head = _logistic_regression()
        if _uniform(weights):
            head.fit(_fit_input(vectors), targets)
        else:
            head.fit(_fit_input(vectors), targets, sample_weight=np.asarray(weights, dtype=np.float64))
    if list(head.classes_) != list(range(label_count)):
        raise ValueError("the head was fitted on rows that do not cover every label")
    if head.coef_.shape[0] != label_count:
        raise ValueError("the head is not multinomial over every label")
    return head.coef_.astype(np.float32), head.intercept_.astype(np.float32)


def cross_fit_splits(vectors: np.ndarray, targets: np.ndarray, texts: Sequence[str]) -> tuple[int, list]:
    """``(folds, [(train positions, test positions), ...])`` of the cross-fit: ``CROSS_FIT_FOLDS`` folds grouped by
    row text, deterministic for the same rows. Fewer than two distinct texts give ``(folds, [])``. The training report
    and the refit (`train.head_refit`) cut the base rows the same way."""
    from sklearn.model_selection import GroupKFold

    groups = np.unique(np.asarray(list(texts), dtype=object).astype(str), return_inverse=True)[1]
    folds = min(CROSS_FIT_FOLDS, int(groups.max()) + 1 if len(groups) else 0)
    if folds < 2:
        return folds, []
    return folds, list(GroupKFold(n_splits=folds).split(vectors, targets, groups))


def cross_fitted_probabilities(
    vectors: np.ndarray, targets: np.ndarray, texts: Sequence[str], label_count: int,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Each row's probabilities from a head that was not fitted on its text.

    ``CROSS_FIT_FOLDS`` folds, grouped by row text: a repeated text (the built-in vague replies are
    repeated) is never on both sides of a fold. A label that is absent from a fold's training side
    gets probability 0 there. Returns the probabilities and what was done; with fewer than two
    distinct texts there is nothing to cross-fit and the in-sample probabilities are returned, and
    the facts say so. For the training report only: nothing is set from it.
    """
    vectors = _fit_input(vectors)
    weights = None if _uniform(weights) else np.asarray(weights, dtype=np.float64)
    probabilities = np.zeros((len(targets), label_count))
    folds, splits = cross_fit_splits(vectors, targets, texts)
    if folds >= 2:
        if all(len(set(targets[train].tolist())) >= 2 for train, _test in splits):
            with _one_thread():
                for train, test in splits:
                    head = _logistic_regression()
                    if weights is None:
                        head.fit(vectors[train], targets[train])
                    else:
                        head.fit(vectors[train], targets[train], sample_weight=weights[train])
                    probabilities[np.ix_(test, head.classes_)] = head.predict_proba(vectors[test])
            return probabilities, {"folds": folds, "grouped_by": "row text", "cross_fitted": True}
    coef, intercept = fit_head(vectors, targets, label_count, weights)
    return softmax_probabilities(vectors, coef, intercept), {
        "folds": 0, "grouped_by": "row text", "cross_fitted": False,
        "note": "too few distinct rows to cross-fit: the numbers are from the rows the head was fitted on",
    }


def mask_indices(labels: Sequence[str], mask: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, outside)`` label positions for a context with *mask*. The reserved labels are in
    neither: ``parameter_value`` and ``wildcard`` are read as the best label overall."""
    index_of = {label: index for index, label in enumerate(labels)}
    inside = [index_of[label] for label in mask if label in index_of]
    held = set(inside)
    outside = [index for index, label in enumerate(labels)
               if index not in held and label not in RESERVED_LABELS]
    return np.array(inside, dtype=int), np.array(outside, dtype=int)


def decide(
    probabilities: np.ndarray, labels: Sequence[str], inside: np.ndarray, outside: np.ndarray,
    threshold: float, margin: Optional[float] = None,
) -> tuple[str, list[str], dict]:
    """The decision in one context: ``(decision, labels returned, numbers)``.

    1. The best label overall is ``parameter_value``: a bare value. It is ``wildcard`` (the built-in
       vague replies' class): not here.
    2. ``p_in`` is the largest probability inside the mask, ``p_out`` the largest outside it
       (reserved labels not counted); nothing is renormalised.
    3. One command when the best command overall is inside the mask and ``p_in >= threshold``.
    4. Otherwise the head is NOT confident and hands the request on: the context's top three. It never
       answers "not here" because a command outside the mask is ahead. When one is, the numbers carry
       ``outside_labels`` / ``outside_scores`` (the best ``OUTSIDE_K`` commands outside the mask, best
       first), which the detector adds to the candidates it can reach, so that the next link, or the
       question, can pick the command where it lives.

    *margin* is the older rule's pass-up margin: a head trained before the rule above has one, and with
    a number here a lone "not here" is answered when ``p_out - p_in >= margin``. None is the rule above.
    """
    order = inside[np.argsort(-probabilities[inside], kind="stable")][:TOP_K] if len(inside) else inside
    topk_labels = [labels[index] for index in order]
    topk_scores = [float(probabilities[index]) for index in order]
    p_in = topk_scores[0] if topk_scores else 0.0
    p_out = float(probabilities[outside].max()) if len(outside) else 0.0
    best_index = int(np.argmax(probabilities))
    best = labels[best_index]
    outside_labels: list[str] = []
    outside_scores: list[float] = []
    if best == PARAMETER_VALUE_LABEL:
        decision, returned, confidence = DECISION_BARE_VALUE, [PARAMETER_VALUE_LABEL], float(probabilities[best_index])
    elif best == WILDCARD_LABEL:
        decision, returned, confidence = DECISION_NOT_MINE, [WILDCARD_LABEL], float(probabilities[best_index])
    elif not topk_labels or (margin is not None and p_out > p_in and p_out - p_in >= margin):
        decision, returned, confidence = DECISION_NOT_MINE, [WILDCARD_LABEL], p_out
    elif p_out <= p_in and p_in >= threshold:
        decision, returned, confidence = DECISION_MATCH, topk_labels[:1], p_in
    else:
        decision, returned, confidence = DECISION_AMBIGUOUS, list(topk_labels), p_in
        if margin is None and p_out > p_in:
            ranked = outside[np.argsort(-probabilities[outside], kind="stable")][:OUTSIDE_K]
            outside_labels = [labels[index] for index in ranked]
            outside_scores = [float(probabilities[index]) for index in ranked]
    return decision, returned, {
        "confidence": confidence,
        "top_label": returned[0],
        "topk_labels": topk_labels,
        "topk_scores": topk_scores,
        "p_in": p_in,
        "p_out": p_out,
        "outside_labels": outside_labels,
        "outside_scores": outside_scores,
    }


# ---------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------


@dataclass
class TrainingRows:
    """The rows the head is fitted on and what was added to the workflow's rows."""

    rows: list[tuple[str, str]]
    weights: np.ndarray
    #: How many leading rows are the workflow's own rows (the rest are description and name rows).
    base_count: int
    description_rows: int
    seed_rows: int


def first_sentence(doc: str) -> str:
    """The first sentence of a docstring on one line, "" when there is none."""
    doc = " ".join((doc or "").split())
    return re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0] if doc else ""


def humanise(label: str) -> str:
    """A command label as words: the leaf of the qualified name, underscores as spaces."""
    return label.split("/")[-1].replace("_", " ")


def _seed_texts(crd, label: str) -> set[str]:
    try:
        metadata = crd.command_directory.get_utterance_metadata(label)
    except Exception:  # noqa: BLE001 - a command without utterance metadata has no seeds
        return set()
    return set(metadata.plain_utterances) if metadata and metadata.plain_utterances else set()


def _command_doc(crd, label: str) -> str:
    import inspect

    import fastworkflow

    try:
        signature = crd.get_command_class(label, fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    except Exception:  # noqa: BLE001 - a command whose module does not load is described by its name
        return ""
    return inspect.getdoc(signature) or ""


def training_rows(rows: Sequence[tuple[str, str]], crd=None) -> TrainingRows:
    """The head's rows: the workflow's rows, then for each command a row of its humanised name and a row
    of the first sentence of its description (its ``Signature`` docstring), as in the experiment's
    "description rows + seeds x3" variant. A row whose text is one of its command's seed utterances
    weighs ``SEED_WEIGHT``. *crd* is the routing definition; without one nothing is added or weighted."""
    rows = list(rows)
    weights = [1.0] * len(rows)
    seeds = 0
    extra: list[tuple[str, str]] = []
    if crd is not None:
        seed_cache: dict[str, set[str]] = {}
        for position, (text, label) in enumerate(rows):
            if label in RESERVED_LABELS:
                continue
            if label not in seed_cache:
                seed_cache[label] = _seed_texts(crd, label)
            if text in seed_cache[label]:
                weights[position] = SEED_WEIGHT
                seeds += 1
        for label in sorted({label for _text, label in rows} - set(RESERVED_LABELS)):
            extra.append((humanise(label), label))
            description = first_sentence(_command_doc(crd, label))
            if description:
                extra.append((description, label))
    return TrainingRows(
        rows=rows + extra, weights=np.array(weights + [1.0] * len(extra)), base_count=len(rows),
        description_rows=len(extra), seed_rows=seeds)


# ---------------------------------------------------------------------------
# The training report's numbers (cross-fit; nothing is set from them)
# ---------------------------------------------------------------------------


def _distinct_command_rows(rows: Sequence[tuple[str, str]], base_count: int) -> list[int]:
    seen: set[str] = set()
    picked: list[int] = []
    for position in range(base_count):
        text, label = rows[position]
        if label in RESERVED_LABELS or text in seen:
            continue
        seen.add(text)
        picked.append(position)
    return picked


def confused_pairs(
    probabilities: np.ndarray, rows: Sequence[tuple[str, str]], labels: Sequence[str], positions: Sequence[int],
) -> list[dict]:
    """The command pairs whose rows the head gives to each other, most confused first. A row counts for
    its command when the best label overall is another command; a guess of ``parameter_value`` or
    ``wildcard`` is not a pair. Each pair lists both directions: ``a_as_b`` of ``a_rows`` and the reverse,
    with their shares, and ``max_share`` the larger share."""
    index_of = {label: index for index, label in enumerate(labels)}
    per_label: dict[int, int] = {}
    confusion: dict[tuple[int, int], int] = {}
    for position in positions:
        target = index_of[rows[position][1]]
        per_label[target] = per_label.get(target, 0) + 1
        guess = int(np.argmax(probabilities[position]))
        if guess != target and labels[guess] not in RESERVED_LABELS:
            confusion[(target, guess)] = confusion.get((target, guess), 0) + 1
    pairs: dict[tuple[int, int], dict] = {}
    for (a, b), count in confusion.items():
        pairs.setdefault((min(a, b), max(a, b)), {})[(a, b)] = count
    table = []
    for (a, b), both in pairs.items():
        a_as_b, b_as_a = both.get((a, b), 0), both.get((b, a), 0)
        a_rows, b_rows = per_label.get(a, 0), per_label.get(b, 0)
        table.append({
            "a": labels[a], "b": labels[b],
            "a_as_b": a_as_b, "a_rows": a_rows, "a_share": a_as_b / a_rows if a_rows else 0.0,
            "b_as_a": b_as_a, "b_rows": b_rows, "b_share": b_as_a / b_rows if b_rows else 0.0,
            "total": a_as_b + b_as_a,
        })
        table[-1]["max_share"] = max(table[-1]["a_share"], table[-1]["b_share"])
    table.sort(key=lambda row: (-row["total"], row["a"], row["b"]))
    return table


def cross_fit_report(
    probabilities: np.ndarray, rows: Sequence[tuple[str, str]], labels: Sequence[str],
    masks: Mapping[str, Sequence[str]], base_count: int, facts: Mapping,
) -> dict:
    """The cross-fit facts of one workflow for the training report.

    ``accuracy`` is the share of distinct command rows whose best label overall is the row's command;
    ``pairs`` the ten most confused pairs and ``pairs_over_limit`` how many pairs have more than
    ``PAIR_SHARE_LIMIT`` of either command's rows given to the other; ``scopes`` per context the rows whose
    command it holds and how many of them its best command is right on (what the context's held-out
    report calls ``in_distribution_f1``)."""
    positions = _distinct_command_rows(rows, base_count)
    index_of = {label: index for index, label in enumerate(labels)}
    right = sum(1 for position in positions
                if int(np.argmax(probabilities[position])) == index_of[rows[position][1]])
    table = confused_pairs(probabilities, rows, labels, positions)
    scopes: dict[str, dict] = {}
    for scope, mask in sorted(masks.items()):
        inside, _outside = mask_indices(labels, mask)
        held = {labels[index] for index in inside}
        entry = {"rows": 0, "right": 0}
        for position in positions:
            label = rows[position][1]
            if label not in held or not len(inside):
                continue
            entry["rows"] += 1
            entry["right"] += int(int(inside[np.argmax(probabilities[position][inside])]) == index_of[label])
        entry["accuracy"] = entry["right"] / entry["rows"] if entry["rows"] else None
        scopes[scope] = entry
    return {
        **facts,
        "rows": len(positions),
        "accuracy": right / len(positions) if positions else None,
        "confused_pairs": len(table),
        "pairs_over_limit": sum(1 for row in table if row["max_share"] > PAIR_SHARE_LIMIT),
        "pair_share_limit": PAIR_SHARE_LIMIT,
        "pairs": table[:PAIRS_REPORTED],
        "scopes": scopes,
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FittedEmbeddingHead:
    """What one fit wrote."""

    #: The content of ``scopes.json``.
    payload: dict
    model_dir: str


def fit_embedding_head(
    rows: Sequence[tuple[str, str]],
    masks: Mapping[str, Sequence[str]],
    out_dir: str,
    *,
    encoder: str = intent_model_layout.DEFAULT_ENCODER,
    pooling: Optional[str] = None,
    prefix: Optional[str] = None,
    commit_threshold: float = intent_model_layout.DEFAULT_COMMIT_THRESHOLD,
    weights: Optional[np.ndarray] = None,
    base_count: Optional[int] = None,
    vectors: Optional[np.ndarray] = None,
    crd=None,
) -> FittedEmbeddingHead:
    """Embed *rows* (``(utterance, label)``), fit the head on all of them, cross-fit it for the report,
    and write ``head.npz`` and ``scopes.json`` to *out_dir*.

    *masks* is each context's commands. The head commits at *commit_threshold*; the cross-fit sets nothing.
    When *crd* (the routing definition) is given the rows are first extended by `training_rows`: the
    commands' description and name rows, and the seeds weighted. Otherwise *rows* are fitted as given, with
    *weights* (default 1), the first *base_count* of them (default all) being the workflow's own rows.
    *vectors* are the rows' embeddings when the caller already has them (the experiment runner embeds once
    and fits several times).
    """
    started = time.monotonic()
    pooling = pooling or intent_model_layout.default_pooling(encoder)
    prefix = intent_model_layout.default_prefix(encoder) if prefix is None else prefix
    seed_rows = 0
    if crd is not None:
        built = training_rows(rows, crd)
        rows, weights, base_count, seed_rows = built.rows, built.weights, built.base_count, built.seed_rows
    base_count = len(rows) if base_count is None else base_count
    description_rows = len(rows) - base_count
    labels = sorted({label for _text, label in rows})
    index_of = {label: index for index, label in enumerate(labels)}
    texts = [text for text, _label in rows]
    targets = np.array([index_of[label] for _text, label in rows])

    embed_started = time.monotonic()
    if vectors is None:
        vectors = SentenceEncoder.get(encoder, pooling, prefix).embed(texts)
    embed_seconds = time.monotonic() - embed_started

    fit_started = time.monotonic()
    coef, intercept = fit_head(vectors, targets, len(labels), weights)
    fit_seconds = time.monotonic() - fit_started

    cross_fit_started = time.monotonic()
    cross_fitted, cross_fit_facts = cross_fitted_probabilities(vectors, targets, texts, len(labels), weights)
    cross_fit = cross_fit_report(cross_fitted, rows, labels, masks, base_count, cross_fit_facts)
    cross_fit_seconds = time.monotonic() - cross_fit_started

    payload = {
        "schema_version": SCOPES_SCHEMA_VERSION,
        "layout": intent_model_layout.EMBEDDING,
        "labels": labels,
        "encoder": {"model": encoder, "pooling": pooling, "prefix": prefix, "max_length": MAX_LENGTH,
                    "dimensions": int(vectors.shape[1]), "normalized": True},
        "head": {"kind": "logistic_regression", "C": HEAD_C, "max_iter": HEAD_MAX_ITER,
                 "file": HEAD_FILENAME},
        "rows": {"total": len(rows), "own": base_count, "description": description_rows,
                 "seeds_weighted": seed_rows, "seed_weight": SEED_WEIGHT,
                 "distinct_texts": len(set(texts))},
        "decision": {"rule": "never_passes_up", "commit_threshold": float(commit_threshold),
                     "outside_candidates": OUTSIDE_K},
        "threshold": float(commit_threshold),
        "cross_fit": cross_fit,
        "scopes": {
            scope: {
                "mask": [label for label in mask if label in index_of],
                "threshold": float(commit_threshold),
                "cross_fit": cross_fit["scopes"].get(scope, {"rows": 0, "right": 0, "accuracy": None}),
            }
            for scope, mask in sorted(masks.items())
        },
        "embed_seconds": embed_seconds,
        "fit_seconds": fit_seconds,
        "cross_fit_seconds": cross_fit_seconds,
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, HEAD_FILENAME), coef=coef, intercept=intercept)
    payload["rows"]["row_store_bytes"] = write_row_store(out_dir, rows, labels, vectors, weights, base_count)
    payload["train_seconds"] = time.monotonic() - started
    with open(os.path.join(out_dir, SCOPES_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return FittedEmbeddingHead(payload=payload, model_dir=out_dir)


def write_row_store(out_dir: str, rows: Sequence[tuple[str, str]], labels: Sequence[str], vectors: np.ndarray,
                    weights: Optional[np.ndarray], base_count: int) -> int:
    """Write ``rows.npz`` beside the head: the vectors the fit just computed, so a refit with learned rows
    (`train.head_refit`) never embeds the base rows again. ``vectors`` are float16 (the encoder's output is
    L2-normalised, and the head is fitted on float64 copies), ``targets`` index *labels*, ``weights`` are the fit's,
    ``kinds`` are 0 the workflow's own row, 1 an own row that is a seed utterance (weighted), 2 a name or description
    row, and ``texts`` the rows' texts. Returns the file's bytes."""
    weights = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=np.float32)
    index_of = {label: index for index, label in enumerate(labels)}
    kinds = np.array([2 if position >= base_count else int(weights[position] > 1.0)
                      for position in range(len(rows))], dtype=np.uint8)
    path = os.path.join(out_dir, intent_model_layout.ROWS_FILENAME)
    np.savez_compressed(
        path, vectors=np.asarray(vectors, dtype=np.float16), weights=np.asarray(weights, dtype=np.float32),
        targets=np.array([index_of[label] for _text, label in rows], dtype=np.int32), kinds=kinds,
        texts=np.array([text for text, _label in rows], dtype=str))
    return os.path.getsize(path)


def read_row_store(model_dir: str) -> Optional[dict]:
    """The rows `write_row_store` wrote (``vectors`` as float32, ``weights``, ``targets``, ``kinds``, ``texts``), or
    None when the head has no row store (a head trained before it cannot be refitted)."""
    path = os.path.join(model_dir, intent_model_layout.ROWS_FILENAME)
    if not os.path.isfile(path):
        return None
    with np.load(path) as stored:
        return {"vectors": np.asarray(stored["vectors"], dtype=np.float32),
                "weights": np.asarray(stored["weights"], dtype=np.float64),
                "targets": np.asarray(stored["targets"], dtype=np.int64),
                "kinds": np.asarray(stored["kinds"]), "texts": [str(text) for text in stored["texts"]]}


def read_summary(model_dir: str) -> Optional[dict]:
    """What a trained head's ``scopes.json`` says about itself, for the training output and the
    version manifest: the encoder, the commit threshold, the cross-fit facts and the times.
    None when *model_dir* holds no embedding head."""
    try:
        with open(os.path.join(model_dir, SCOPES_FILENAME), "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if payload.get("layout") != intent_model_layout.EMBEDDING:
        return None
    cross_fit = payload.get("cross_fit") or {}
    return {
        "encoder": payload["encoder"]["model"],
        "pooling": payload["encoder"]["pooling"],
        "prefix": payload["encoder"].get("prefix", ""),
        "labels": len(payload["labels"]),
        "rows": payload["rows"]["total"],
        "description_rows": payload["rows"].get("description", 0),
        "threshold": payload["threshold"],
        "cross_fit": {
            name: cross_fit.get(name)
            for name in ("rows", "accuracy", "folds", "cross_fitted", "confused_pairs", "pairs_over_limit")
        },
        "embed_seconds": payload.get("embed_seconds"),
        "fit_seconds": payload.get("fit_seconds"),
        "cross_fit_seconds": payload.get("cross_fit_seconds"),
        "train_seconds": payload.get("train_seconds"),
        # The head itself; the row store a refit reads (`write_row_store`) is counted apart.
        "bytes": sum(
            os.path.getsize(os.path.join(model_dir, name)) for name in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, name)) and name != intent_model_layout.ROWS_FILENAME),
        "row_store_bytes": payload["rows"].get("row_store_bytes", 0),
    }


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class EmbeddingHeadPipeline:
    """The loaded head of one workflow model directory, shared by its contexts.

    Holds the label order, the head's two arrays and the process's shared encoder.
    """

    _instances: ClassVar[dict[str, "EmbeddingHeadPipeline"]] = {}
    _build_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, SCOPES_FILENAME), "r", encoding="utf-8") as f:
            self.scopes_file = json.load(f)
        if self.scopes_file.get("layout") != intent_model_layout.EMBEDDING:
            raise ValueError(f"{model_dir} does not hold an embedding head")
        self.labels: list[str] = list(self.scopes_file["labels"])
        with np.load(os.path.join(model_dir, HEAD_FILENAME)) as head:
            self.coef = np.asarray(head["coef"], dtype=np.float64)
            self.intercept = np.asarray(head["intercept"], dtype=np.float64)
        if self.coef.shape[0] != len(self.labels):
            raise ValueError(f"{model_dir}: the head and the label order do not agree")
        # A head trained before the prefix was recorded was fitted on unprefixed text.
        self.encoder = SentenceEncoder.get(
            self.scopes_file["encoder"]["model"], self.scopes_file["encoder"]["pooling"],
            self.scopes_file["encoder"].get("prefix", ""))
        self.device = self.encoder.device

    @classmethod
    def for_directory(cls, model_dir: str) -> "EmbeddingHeadPipeline":
        # The real path: every context of a version shares one load and a new version is a new key.
        key = os.path.realpath(model_dir)
        with cls._build_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key)
            return cls._instances[key]

    def probabilities(self, text: str) -> np.ndarray:
        """The head's probability for every label of the workflow, for one request."""
        return softmax_probabilities(self.encoder.vector(text).reshape(1, -1), self.coef, self.intercept)[0]


class EmbeddingHeadRouter:
    """`CommandRouter`'s interface over the workflow's embedding head, for one context.

    ``predict`` returns one label when confident and the context's top three otherwise. "Not here" is
    the single label ``wildcard`` (the built-in vague replies' class) and a bare value the single label
    ``parameter_value``, exactly as the other layouts report them, so their consumers need no change.

    The router tells which rule a head was trained with from its ``scopes.json``: ``schema_version`` 2
    (a ``decision`` block, no ``margin``) never passes a request up on its own; a head without it (version 1,
    with a ``margin``) is read as it always was, with its pass-up margin, until the workflow is retrained.
    """

    def __init__(self, model_dir: str, context_name: str):
        self.model_dir = model_dir
        self.context_name = context_name
        self.modelpipeline = EmbeddingHeadPipeline.for_directory(model_dir)
        scopes_file = self.modelpipeline.scopes_file
        if context_name not in scopes_file["scopes"]:
            raise KeyError(
                f"Scope {context_name!r} is not in {os.path.join(model_dir, SCOPES_FILENAME)}; "
                f"it has {sorted(scopes_file['scopes'])}."
            )
        scope = scopes_file["scopes"][context_name]
        self._labels = self.modelpipeline.labels
        self._inside, self._outside = mask_indices(self._labels, scope["mask"])
        trained_threshold = float(scope.get("threshold", scopes_file["threshold"]))
        if int(scopes_file.get("schema_version", 1)) >= 2:
            self.threshold, self.threshold_source = intent_model_layout.threshold_in_force(model_dir, trained_threshold)
        else:
            self.threshold, self.threshold_source = trained_threshold, "trained"
        self.head_version = intent_model_layout.workflow_and_version_of(model_dir)[1]
        #: None for a head that never passes up alone; a number for a head trained with the older rule.
        self.escalation_margin: Optional[float] = (
            None if int(scopes_file.get("schema_version", 1)) >= 2
            else float(scope.get("margin", scopes_file.get("margin", 0.0))))

    def predict(self, command: str) -> list[str]:
        return self.predict_with_details(command)[0]

    def predict_with_details(self, command: str) -> tuple[list[str], dict]:
        """``predict`` plus the numbers behind the decision: the keys `CommandRouter` documents
        (``model_tier`` is "embedding"), and ``p_in``, ``p_out``, ``decision``, ``outside_labels`` and
        ``outside_scores`` (the commands outside the mask the head ranks ahead of its own best, when it
        hands a request on for that reason). ``confidence`` is ``p_in`` for a command or a list and the
        best label's own probability for a reserved label. ``topk_labels`` are the context's best
        candidates with their probabilities, not renormalised."""
        if not isinstance(command, str):
            raise ValueError("Input must be a string")
        if not command.strip():
            raise ValueError("Input text cannot be empty")
        decision, labels, numbers = decide(
            self.modelpipeline.probabilities(command), self._labels, self._inside, self._outside,
            self.threshold, self.escalation_margin)
        details = {
            "model_tier": MODEL_TIER,
            "confidence": numbers["confidence"],
            "ambiguous_threshold": self.threshold,
            "confident": decision != DECISION_AMBIGUOUS,
            "top_label": numbers["top_label"],
            "topk_labels": numbers["topk_labels"],
            "topk_scores": numbers["topk_scores"],
            "p_in": numbers["p_in"],
            "p_out": numbers["p_out"],
            "outside_labels": numbers["outside_labels"],
            "outside_scores": numbers["outside_scores"],
            "decision": decision,
            "head_version": self.head_version,
            "threshold_source": self.threshold_source,
        }
        if self.escalation_margin is not None:
            details["escalation_margin"] = self.escalation_margin
        return list(labels), details


# ---------------------------------------------------------------------------
# The trainer's side
# ---------------------------------------------------------------------------


class EmbeddingLayoutTrainer(workflow_model.LayoutTrainer):
    """Trains a workflow into the ``embedding`` layout from inside `train`.

    The per-context loop and the row assembly are `LayoutTrainer`'s: the rows are the rows the
    ``workflow`` layout fits, plus each command's description and name rows with the seeds weighted
    (`training_rows`). `finish` embeds them, fits the head on all of them, cross-fits it for the
    report, writes ``workflow/`` and each context's two small files, and returns the per-context
    held-out reports.
    """

    layout = intent_model_layout.EMBEDDING

    def finish(self) -> list:
        if not self.scopes:
            return []
        rows = self.rows()
        for finding in rows.findings:
            logger.info(f"embedding head rows: {finding}")
        settings = intent_model_layout.embedding_settings(self.workflow_folderpath)
        name = os.path.basename(os.path.normpath(self.workflow_folderpath))
        print(
            f"\n=== Fitting the embedding head for workflow: {name} "
            f"({len(rows.labels)} labels, {len(rows.rows)} rows, {len(self.scopes)} contexts; "
            f"encoder {settings['encoder']}, {settings['pooling']} pooling) ===\n"
        )
        fitted = fit_embedding_head(
            rows.rows, {scope: entry.mask for scope, entry in self.scopes.items()},
            self.model_dir(), crd=self.crd, **settings)
        trained_texts = {text for text, _label in rows.rows}

        context_dirs: dict[str, str] = {}
        for context_name in sorted(self.scopes):
            context_dir = os.path.dirname(
                heads.get_artifact_path(self.workflow_folderpath, context_name, "threshold.json"))
            intent_model_layout.write_context_files(context_dir, context_name, self.layout)
            context_dirs[context_name] = context_dir

        reports = [
            self._embedding_report(context_name, context_dirs[context_name], fitted, trained_texts)
            for context_name in sorted(self.scopes)
        ]
        payload = fitted.payload
        cross_fit = payload["cross_fit"]
        accuracy = cross_fit.get("accuracy")
        print(
            f"Embedding head written to {self.model_dir()}: embedding the rows took "
            f"{payload['embed_seconds']:.1f} s, fitting the head {payload['fit_seconds']:.1f} s, "
            f"the cross-fit for the report {payload['cross_fit_seconds']:.1f} s. Commit threshold "
            f"{payload['threshold']:.2f} ({cross_fit['rows']} cross-fitted rows"
            + (f", {accuracy:.1%} right" if accuracy is not None else "")
            + f"; {cross_fit['pairs_over_limit']} command pairs confused over "
              f"{cross_fit['pair_share_limit']:.0%})."
        )
        return reports

    def _embedding_report(self, context_name: str, context_dir: str,
                          fitted: FittedEmbeddingHead, trained_texts: set):
        payload = fitted.payload
        scope_numbers = payload["scopes"][context_name]
        scope_cross_fit = scope_numbers["cross_fit"]
        split = self._splits.get(context_name)
        notes = list(split.notes) if split is not None else []
        heldout = list(split.heldout) if split is not None else []
        # A command shared by two contexts is one label with the union of their rows, so a row
        # this context held out can have been fitted through the other.
        unseen = [record for record in heldout if record.utterance not in trained_texts]
        if len(unseen) != len(heldout):
            notes.append(
                f"{len(heldout) - len(unseen)} of {len(heldout)} held-out rows were left out "
                f"of the score: the workflow's one head was fitted on them through another "
                f"context that shares the command."
            )
        notes.append(
            "in_distribution_f1 is the share of this context's cross-fitted training rows "
            "whose best command is the right one (embedding layout).")
        report = heldout_evaluation.HeldoutReport(
            context=context_name,
            in_distribution_f1=scope_cross_fit.get("accuracy"),
            seed=self.seed,
            heldout_personas=list(self.scopes[context_name].heldout_personas),
            notes=notes,
            large_ambiguity={
                "failed_rows": scope_cross_fit["rows"] - scope_cross_fit["right"],
                "rule": "fixed_commit_threshold",
                "threshold": payload["threshold"],
                "has_wildcard_class": True,
                "at_floor_without_wildcard": False,
                "wildcard_seeded_from_built_in_vague_replies": True,
                "built_in_vague_weight": heads.BUILT_IN_VAGUE_WEIGHT,
                "intent_model_layout": intent_model_layout.EMBEDDING,
                "encoder": payload["encoder"]["model"],
                "cross_fit": {
                    name: payload["cross_fit"].get(name)
                    for name in ("rows", "accuracy", "confused_pairs", "pairs_over_limit")
                },
            },
        )
        try:
            has_benchmark = any(
                case.context == context_name and case.kind in {"routing", "escalation"}
                for case in self.benchmark_cases
            )
            predict_labels = None
            if unseen or has_benchmark:
                # The reader the runtime uses, found through the context folder as it will be.
                router = heads.CommandRouter(context_dir)

                def predict_labels(utterance: str, _router=router) -> list[str]:
                    return [str(label) for label in _router.predict(utterance)]

            heads._score_heldout_context(
                report, unseen, self.benchmark_cases, predict_labels,
                split_for_calibration=split if unseen else None)
        except Exception as exc:  # noqa: BLE001 - as in `train`: the head is written and usable
            report.notes.append(f"held-out evaluation failed: {exc}")
            logger.error(f"Held-out evaluation failed for context '{context_name}': {exc}")
        return report
