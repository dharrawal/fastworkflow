"""The embedding head's calibration and decision rule AS IT WAS when the 2026-11-17 experiments ran.

Not the product. bd fix-o5wi.9 removed this from ``fastworkflow/train/embedding_head.py`` (the product's commit
threshold is now a fixed default, it never answers "not here" for a command outside the mask, and it has no pass-up
margin), and that file was never committed, so no git revision has these functions. The experiment scripts
(``embedding_commit_rule.py``, ``embedding_chain_followup.py``, ``learn_from_corrections.py``) and the documents that
cite them (``docs/single_intent_model_preregistration_v4.md`` and ``_v5.md``, ``docs/learning_from_corrections.md``,
``docs/local_intent_model_levers.md``) describe THIS rule, so they import it from here instead of from the product.

What is reconstructed (the rule as ``AGENTS.md`` described it before fix-o5wi.9):

* ``calibration_records``: from the five-fold cross-fitted probabilities of the training rows (folds grouped by row
  text), each DISTINCT row text counted once in every context whose mask holds its label; the rows of reserved labels
  (``parameter_value``, ``wildcard``) are not counted. A record is ``(p_in, p_out - p_in, right, bare)``.
* ``calibrate``: the margin is the smallest at which at most ``esc_target`` of the records are sent up (the
  ``int(esc_target * n)``-th largest positive gap, plus 1e-9; 0.0 when there are not that many); the threshold is the
  lowest cross-fitted ``p_in`` (rounded to 3 places, among records with ``p_out <= p_in``) at which the wrong lone
  answers are at most ``wrong_target`` of all records, never below ``threshold_floor`` (0.5).
* ``decide``: best label overall ``parameter_value`` is a bare value; the workflow ``wildcard`` is "not here"; else one
  command when ``p_out <= p_in`` and ``p_in >= threshold``, the lone "not here" when ``p_out - p_in >= margin``, else
  the context's top three.

The encoder of those experiments is ``LEGACY_ENCODER`` (CLS pooling, no prefix), not the product's new default.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from fastworkflow.train import embedding_head as eh

LEGACY_ENCODER = "BAAI/bge-small-en-v1.5"
LEGACY_POOLING = "cls"
LEGACY_PREFIX = ""
THRESHOLD_FLOOR = 0.5

PARAMETER_VALUE_LABEL = eh.PARAMETER_VALUE_LABEL
WILDCARD_LABEL = eh.WILDCARD_LABEL
RESERVED_LABELS = eh.RESERVED_LABELS
DECISION_MATCH = eh.DECISION_MATCH
DECISION_AMBIGUOUS = eh.DECISION_AMBIGUOUS
DECISION_NOT_MINE = eh.DECISION_NOT_MINE
DECISION_BARE_VALUE = eh.DECISION_BARE_VALUE
TOP_K = eh.TOP_K


def legacy_encoder() -> "eh.SentenceEncoder":
    """The frozen encoder the experiments used (``BAAI/bge-small-en-v1.5``, CLS pooling, no prefix)."""
    return eh.SentenceEncoder.get(LEGACY_ENCODER, LEGACY_POOLING, LEGACY_PREFIX)


def decide(
    probabilities: np.ndarray, labels: Sequence[str], inside: np.ndarray, outside: np.ndarray,
    threshold: float, margin: Optional[float] = None,
) -> tuple[str, list[str], dict]:
    """``(decision, labels returned, numbers)`` in one context, by the rule before fix-o5wi.9."""
    margin = 0.0 if margin is None else margin   # the old head always had one; a head without calibration had 0.0
    order = inside[np.argsort(-probabilities[inside], kind="stable")][:TOP_K] if len(inside) else inside
    topk_labels = [labels[index] for index in order]
    topk_scores = [float(probabilities[index]) for index in order]
    p_in = topk_scores[0] if topk_scores else 0.0
    p_out = float(probabilities[outside].max()) if len(outside) else 0.0
    best_index = int(np.argmax(probabilities))
    best = labels[best_index]
    if best == PARAMETER_VALUE_LABEL:
        decision, returned, confidence = DECISION_BARE_VALUE, [PARAMETER_VALUE_LABEL], float(probabilities[best_index])
    elif best == WILDCARD_LABEL:
        decision, returned, confidence = DECISION_NOT_MINE, [WILDCARD_LABEL], float(probabilities[best_index])
    elif not topk_labels or (p_out > p_in and p_out - p_in >= margin):
        decision, returned, confidence = DECISION_NOT_MINE, [WILDCARD_LABEL], p_out
    elif p_out <= p_in and p_in >= threshold:
        decision, returned, confidence = DECISION_MATCH, topk_labels[:1], p_in
    else:
        decision, returned, confidence = DECISION_AMBIGUOUS, list(topk_labels), p_in
    return decision, returned, {
        "confidence": confidence, "top_label": returned[0], "topk_labels": topk_labels,
        "topk_scores": topk_scores, "p_in": p_in, "p_out": p_out,
    }


def calibration_records(
    probabilities: np.ndarray, rows: Sequence[tuple[str, str]], labels: Sequence[str], masks: dict[str, Sequence[str]],
) -> list[tuple[float, float, bool, bool]]:
    """One ``(p_in, p_out - p_in, right, bare)`` per distinct row text and context whose mask holds its label.

    *probabilities* are the rows' cross-fitted probabilities (``eh.cross_fitted_probabilities``), aligned with *rows*."""
    reserved = set(RESERVED_LABELS)
    scopes = {scope: eh.mask_indices(labels, mask) for scope, mask in masks.items()}
    held = {scope: set(mask) for scope, mask in masks.items()}
    seen: set[str] = set()
    records: list[tuple[float, float, bool, bool]] = []
    for index, (text, label) in enumerate(rows):
        if label in reserved or text in seen:
            continue
        seen.add(text)
        for scope, (inside, outside) in scopes.items():
            if label not in held[scope]:
                continue
            _decision, returned, numbers = decide(probabilities[index], labels, inside, outside, 0.0, 0.0)
            p_in, p_out = numbers["p_in"], numbers["p_out"]
            best = labels[int(np.argmax(probabilities[index]))]
            records.append((p_in, p_out - p_in, numbers["topk_labels"][:1] == [label], best == PARAMETER_VALUE_LABEL))
    return records


def calibrate(records: Sequence[tuple[float, float, bool, bool]], wrong_target: float, esc_target: float,
              threshold_floor: float = THRESHOLD_FLOOR) -> dict:
    """``{"threshold", "margin", "records", "accuracy"}`` from the cross-fitted records (see the module docstring)."""
    count = len(records)
    if not count:
        return {"threshold": 1.0, "margin": 0.0, "records": 0, "accuracy": None}
    gaps = sorted((gap for _p, gap, _r, bare in records if gap > 0 and not bare), reverse=True)
    allowed = int(esc_target * count)
    margin = float(gaps[allowed] + 1e-9) if len(gaps) > allowed else 0.0
    ahead = [(p_in, right) for p_in, gap, right, _bare in records if gap <= 0]
    wrong = sorted(p_in for p_in, right in ahead if not right)
    threshold = 1.0
    for cand in sorted({round(float(p_in), 3) for p_in, _right in ahead}):
        above = len(wrong) - int(np.searchsorted(wrong, cand, side="left"))
        if above / count <= wrong_target:
            threshold = float(cand)
            break
    accuracy = sum(1 for _p, gap, right, _b in records if gap <= 0 and right) / count
    return {"threshold": max(threshold, threshold_floor), "margin": margin, "records": count, "accuracy": float(accuracy)}
