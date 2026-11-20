"""Refit the embedding head with the rows a workflow learned from use (bd fix-o5wi.6).

A refit is the stored base rows (``rows.npz``, written by `embedding_head.fit_embedding_head`: the vectors the train
already computed) plus the learned rows (`intent_learning.teaching_rows`, weighted like a seed utterance), fitted with
the same ``LogisticRegression`` over the same labels and masks. The base rows are never embedded again; a learned text
is embedded once, by the head's own encoder, and its vector is kept in the learning store.

A refit is published only when it passes two gates, both measured by cross-fit on rows the fit did not see. Confirmed
real requests outweigh the synthetic base rows, and the owner's priority stays "few confident wrong answers", so each gate
has one small, named tolerance (bd fix-o5wi.6; measured on a large workflow, a literal gate rejected 21 of 26 refits that
were right outright more often on held-out learned rows):

* the base rows, cut into the folds the training report uses (`embedding_head.cross_fit_splits`), with the learned
  rows added to every training side: accuracy no lower than the same cross-fit without them, less
  `GATE_BASE_TOLERANCE` (1.5 points);
* the learned rows, five folds grouped by normalised text, each held out of a fit on the base rows and the other
  learned rows: right outright no lower, and wrong confident no higher, than the same rows held out of a fit on the
  base rows alone, both read through the context's mask at the threshold in force; except that wrong confident may be
  higher by `GATE_WRONG_TOLERANCE_ROWS` (one row) when right outright is higher by at least
  `GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS` (three rows). (The head being replaced is not the baseline: after an earlier
  refit it has seen some of these rows, so scoring it on them would be in-sample.)

The gate report says which rule let a refit through when a tolerance was used (`rule` on each gate). The base rows'
cross-fit WITHOUT the learned rows depends only on the stored base rows, so its totals are kept in the learning store,
keyed by a digest of those rows, and reused by the next refit (the refit side is always computed).

It is published as a NEW version through the existing atomic versioning (`artifact_versioning`), whose manifest says
``refit_of``, the learned rows and the gate numbers, and whose ``lineage`` lists the full train it descends from.
Retention keeps that full train. The built-in workflows and every head trained before the row store have none: a head
without ``rows.npz`` is not refittable until its next train.

This module makes no model call: the encoder is a local one.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fastworkflow import intent_learning as learning
from fastworkflow.train import artifact_versioning, embedding_head as eh, intent_model_layout as layout

#: The base rows' accuracy may fall by at most this much (a share, 1.5 points) when learned rows are added: they are
#: synthetic, and a confirmed real request outweighs a point of agreement with them (of 26 refit attempts on a large workflow the
#: 0.5 point gate rejected 21, and 19 of those 21 failed this condition; in 20 of the 21 the held-out learned rows
#: were right outright more often).
GATE_BASE_TOLERANCE = 0.015
#: Wrong confident on the held-out learned rows may be higher by this many rows ...
GATE_WRONG_TOLERANCE_ROWS = 1
#: ... only when right outright is higher by at least this many rows, so one more wrong answer must buy real gains.
GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS = 3
#: Layout of the cached base cross-fit totals in the learning store; bump when the computation changes.
BASE_CROSSFIT_CACHE_FORMAT = 1
BASE_CROSSFIT_META_KEY = "base_crossfit"
LEARNED_FOLDS = 5
LOCK_FILENAME = "intent_learning.lock"

try:  # pragma: no cover - platform
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass
class RefitReport:
    """What one refit did. ``published`` is False with ``reason`` saying why."""

    published: bool = False
    reason: str = ""
    base_version: Optional[str] = None
    new_version: Optional[str] = None
    learned_rows: int = 0
    gate: dict = field(default_factory=dict)
    seconds: float = 0.0
    embedded_texts: int = 0
    artifact_bytes: int = 0

    def summary_line(self) -> str:
        if self.published:
            return (f"refit published as {self.new_version} (of {self.base_version}): {self.learned_rows} learned "
                    f"rows, {self.embedded_texts} text(s) embedded, {self.seconds:.1f} s")
        return f"refit not published: {self.reason}"


@contextlib.contextmanager
def refit_lock(workflow_folderpath: str):
    """A file lock that keeps two refits of one workflow (two processes, or a background thread and a CLI run) from
    running at once. Yields whether it was taken; never blocks."""
    folder = os.path.join(workflow_folderpath, learning.CONVO_FOLDER)
    os.makedirs(folder, exist_ok=True)
    if fcntl is None:  # pragma: no cover
        yield True
        return
    fd = os.open(os.path.join(folder, LOCK_FILENAME), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _fingerprint(teaching) -> str:
    import hashlib

    text = json.dumps([[row.norm, row.context, row.command] for row in teaching], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _fit_probabilities(train_vectors, train_targets, train_weights, test_vectors, label_count) -> np.ndarray:
    """Probabilities of *test_vectors* from a head fitted on the training side; a label absent from it gets 0."""
    out = np.zeros((len(test_vectors), label_count))
    with eh._one_thread():
        head = eh._logistic_regression()
        head.fit(eh._fit_input(train_vectors), train_targets, sample_weight=np.asarray(train_weights, dtype=np.float64))
        out[:, head.classes_] = head.predict_proba(eh._fit_input(test_vectors))
    return out


def _context_decisions(probabilities, labels, scopes, contexts, threshold) -> list[Optional[tuple[str, str]]]:
    """``(decision, top label)`` of each row through its context's mask; None for a context the head does not have."""
    masks: dict[str, tuple] = {}
    answers = []
    for row, context in zip(probabilities, contexts):
        if context not in scopes:
            answers.append(None)
            continue
        if context not in masks:
            masks[context] = eh.mask_indices(labels, scopes[context]["mask"])
        inside, outside = masks[context]
        decision, returned, _numbers = eh.decide(row, labels, inside, outside, threshold)
        answers.append((decision, returned[0]))
    return answers


def _score(answers, expected) -> dict:
    scored = [(answer, want) for answer, want in zip(answers, expected) if answer is not None]
    n = len(scored)
    right = sum(1 for (decision, top), want in scored if decision == eh.DECISION_MATCH and top == want)
    wrong = sum(1 for (decision, top), want in scored if decision == eh.DECISION_MATCH and top != want)
    return {"rows": n, "right_outright": right / n if n else 0.0, "wrong_confident": wrong / n if n else 0.0,
            "right_rows": right, "wrong_rows": wrong}


def base_gate_verdict(right_without: int, right_with: int, counted: int) -> tuple[bool, str]:
    """Gate (a) on counts: ``(passed, rule)``. ``rule`` is "no loss" (also for a gain) or "base tolerance" when the
    accuracy fell by no more than `GATE_BASE_TOLERANCE`; "" when it failed. Integer rows, with a hair for the float."""
    lost = right_without - right_with
    if lost <= 0:
        return True, "no loss"
    if lost <= GATE_BASE_TOLERANCE * counted + 1e-9:
        return True, "base tolerance"
    return False, ""


def learned_gate_verdict(before: dict, after: dict) -> tuple[bool, str]:
    """Gate (b) on the two scores' row counts: ``(passed, rule)``. Right outright must not fall; wrong confident must
    not rise, except by `GATE_WRONG_TOLERANCE_ROWS` row(s) when right outright rose by at least
    `GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS` rows (rule "wrong confident tolerance"). "" when it failed."""
    right_gain = after["right_rows"] - before["right_rows"]
    wrong_gain = after["wrong_rows"] - before["wrong_rows"]
    if right_gain < 0:
        return False, ""
    if wrong_gain <= 0:
        return True, "strict"
    if wrong_gain <= GATE_WRONG_TOLERANCE_ROWS and right_gain >= GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS:
        return True, "wrong confident tolerance"
    return False, ""


def _base_digest(base, label_count: int) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(f"{BASE_CROSSFIT_CACHE_FORMAT}|{label_count}|{len(base['targets'])}".encode())
    for name in ("vectors", "targets", "weights", "kinds"):
        h.update(np.ascontiguousarray(base[name]).tobytes())
    h.update("\x00".join(base["texts"]).encode("utf-8"))
    return h.hexdigest()[:24]


def _base_gate(base, labels, learned_vectors, learned_targets, learned_weights, workflow_folderpath=None,
               store_baseline=False) -> dict:
    """Gate (a): the base rows' cross-fit accuracy with and without the learned rows in every training side. The
    "without" side depends only on the base rows, so its totals are read from the learning store when an earlier refit
    of the same rows kept them (`store_baseline` writes them after computing)."""
    rows = list(zip(base["texts"], [labels[target] for target in base["targets"]]))
    own = int(np.sum(base["kinds"] < 2))
    positions = eh._distinct_command_rows(rows, own)
    folds, splits = eh.cross_fit_splits(base["vectors"], base["targets"], base["texts"])
    if folds < 2 or not positions:
        return {"checked": False, "note": "too few base rows to cross-fit", "passed": True}
    wanted = set(positions)
    digest = _base_digest(base, len(labels))
    known = learning.get_meta(workflow_folderpath, BASE_CROSSFIT_META_KEY) if workflow_folderpath else None
    cached = known if known and known.get("digest") == digest else None
    right_without = right_with = counted = 0
    for train, test in splits:
        test = [position for position in test if position in wanted]
        if not test:
            continue
        vectors = base["vectors"][train]
        targets = base["targets"][train]
        weights = base["weights"][train]
        expected = base["targets"][test]
        if cached is None:
            without = _fit_probabilities(vectors, targets, weights, base["vectors"][test], len(labels))
            right_without += int(np.sum(np.argmax(without, axis=1) == expected))
        with_learned = _fit_probabilities(
            np.concatenate([vectors, learned_vectors]), np.concatenate([targets, learned_targets]),
            np.concatenate([weights, learned_weights]), base["vectors"][test], len(labels))
        right_with += int(np.sum(np.argmax(with_learned, axis=1) == expected))
        counted += len(test)
    if cached is not None:
        right_without = int(cached["right_without"])
        if int(cached["counted"]) != counted:  # the folds changed under the same digest: do not trust the cache
            return _base_gate(base, labels, learned_vectors, learned_targets, learned_weights, None, False)
    elif store_baseline and workflow_folderpath:
        learning.set_meta(workflow_folderpath, BASE_CROSSFIT_META_KEY,
                          {"digest": digest, "right_without": right_without, "counted": counted})
    before, after = right_without / counted, right_with / counted
    passed, rule = base_gate_verdict(right_without, right_with, counted)
    return {"checked": True, "rows": counted, "accuracy_before": before, "accuracy_after": after,
            "tolerance": GATE_BASE_TOLERANCE, "baseline_cached": cached is not None, "rule": rule, "passed": passed}


def _learned_gate(base, labels, scopes, teaching, learned_vectors, learned_targets, learned_weights, threshold) -> dict:
    """Gate (b): the learned rows by leave-out cross-fit. Each fold of learned rows (grouped by normalised text) is
    held out of two fits, one on the base rows alone (what the head knew before it learned from use) and one on the base
    rows and the other learned rows; both are read through the context's mask at the threshold in force. The second
    must be right outright no less often, and wrong confident no more often (one row more is allowed when right outright
    gained three: `learned_gate_verdict`). Both sides are held out, so a row an
    earlier refit already learned is never scored by a head that has seen it."""
    from sklearn.model_selection import GroupKFold

    contexts = [row.context for row in teaching]
    expected = [row.command for row in teaching]
    groups = np.unique(np.array([row.norm for row in teaching], dtype=str), return_inverse=True)[1]
    folds = min(LEARNED_FOLDS, int(groups.max()) + 1)
    without = np.zeros((len(teaching), len(labels)))
    with_learned = np.zeros((len(teaching), len(labels)))
    for train, test in GroupKFold(n_splits=folds).split(learned_vectors, learned_targets, groups):
        without[test] = _fit_probabilities(
            base["vectors"], base["targets"], base["weights"], learned_vectors[test], len(labels))
        with_learned[test] = _fit_probabilities(
            np.concatenate([base["vectors"], learned_vectors[train]]),
            np.concatenate([base["targets"], learned_targets[train]]),
            np.concatenate([base["weights"], learned_weights[train]]), learned_vectors[test], len(labels))
    before = _score(_context_decisions(without, labels, scopes, contexts, threshold), expected)
    after = _score(_context_decisions(with_learned, labels, scopes, contexts, threshold), expected)
    passed, rule = learned_gate_verdict(before, after)
    return {"checked": True, "rows": after["rows"], "folds": folds, "current": before, "refit": after,
            "wrong_tolerance_rows": GATE_WRONG_TOLERANCE_ROWS,
            "wrong_tolerance_needs_right_rows": GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS, "rule": rule, "passed": passed}


def _encoder_key(encoder: dict) -> str:
    return f"{encoder['model']}|{encoder['pooling']}|{encoder.get('prefix', '')}"


def _learned_vectors(workflow_folderpath: str, encoder: dict, teaching) -> tuple[np.ndarray, int]:
    """The vector of each teaching row, from the store when its text was embedded before, else embedded now (only
    those texts) and stored. Returns the vectors and how many texts were embedded."""
    key = _encoder_key(encoder)
    cached = learning.cached_vectors(workflow_folderpath, key, [row.norm for row in teaching])
    missing = [row for row in teaching if row.norm not in cached]
    if missing:
        vectors = eh.SentenceEncoder.get(encoder["model"], encoder["pooling"], encoder.get("prefix", "")).embed(
            [row.text for row in missing])
        fresh = {row.norm: vector for row, vector in zip(missing, vectors)}
        learning.store_vectors(workflow_folderpath, key, fresh)
        cached.update({norm: np.asarray(vector, dtype=np.float16).astype(np.float32) for norm, vector in fresh.items()})
    return np.stack([cached[row.norm] for row in teaching]), len(missing)


def _head_of_current(workflow_folderpath: str):
    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        return None, None, None
    model_dir = str(artifact_versioning.version_dir(workflow_folderpath, version) / layout.WORKFLOW_MODEL_DIRNAME)
    return version, model_dir, os.path.join(model_dir, layout.SCOPES_FILENAME)


def refit(workflow_folderpath: str, *, publish: bool = True, min_rows: int = learning.MIN_TEACHING_ROWS) -> RefitReport:
    """Refit the workflow's current head with its teaching rows; publish it when the gates pass (`publish=False`
    runs everything and writes nothing). The caller holds `refit_lock` when more than one process may refit."""
    started = time.monotonic()
    report = RefitReport()
    version, model_dir, scopes_path = _head_of_current(workflow_folderpath)
    report.base_version = version
    if version is None or not os.path.isfile(scopes_path):
        report.reason = "the workflow has no trained head"
        return report
    with open(scopes_path, encoding="utf-8") as f:
        scopes_file = json.load(f)
    if scopes_file.get("layout") != layout.EMBEDDING or int(scopes_file.get("schema_version", 1)) < 2:
        report.reason = (f"the current head is of the {scopes_file.get('layout')!r} layout, and only the "
                         f"{layout.EMBEDDING!r} layout is refitted (other layouts are unchanged)")
        return report
    base = eh.read_row_store(model_dir)
    if base is None:
        report.reason = (f"version {version} has no stored rows ({layout.ROWS_FILENAME}), so it cannot be refitted "
                         "until the workflow is trained again (`fastworkflow train`)")
        return report
    labels = list(scopes_file["labels"])
    index_of = {label: index for index, label in enumerate(labels)}
    all_rows = learning.rows(workflow_folderpath, confirmed_only=True)
    teaching = learning.teaching_rows(workflow_folderpath, labels=labels)
    teaching = [row for row in teaching if row.context in scopes_file["scopes"]]
    report.learned_rows = len(teaching)
    if len(teaching) < min_rows:
        report.reason = (f"{len(teaching)} teaching row(s), and at least {min_rows} distinct texts are needed to check "
                         "a refit")
        return report
    fingerprint = _fingerprint(teaching)
    manifest = artifact_versioning.read_manifest(workflow_folderpath, version)
    if manifest.get("learned_fingerprint") == fingerprint:
        report.reason = f"nothing new: version {version} was refitted on exactly these {len(teaching)} rows"
        return report

    learned_vectors, report.embedded_texts = _learned_vectors(workflow_folderpath, scopes_file["encoder"], teaching)
    learned_targets = np.array([index_of[row.command] for row in teaching], dtype=np.int64)
    learned_weights = np.full(len(teaching), eh.SEED_WEIGHT)
    threshold, _why = layout.threshold_in_force(model_dir, float(scopes_file["threshold"]))
    gate_a = _base_gate(base, labels, learned_vectors, learned_targets, learned_weights, workflow_folderpath,
                        store_baseline=publish)
    gate_b = _learned_gate(base, labels, scopes_file["scopes"], teaching, learned_vectors, learned_targets,
                           learned_weights, threshold)
    report.gate = {"base_rows": gate_a, "learned_rows": gate_b, "passed": bool(gate_a["passed"] and gate_b["passed"])}
    marker = {"row_id": max((row["id"] for row in all_rows), default=0)}
    if not report.gate["passed"]:
        failed = [name for name, gate in (("base rows", gate_a), ("learned rows", gate_b)) if not gate["passed"]]
        report.reason = f"the gate failed on the {' and '.join(failed)}: {_gate_text(report.gate)}"
        report.seconds = time.monotonic() - started
        if publish:
            # An attempt counts for the background trigger: the next one waits for INTENT_LEARNING_REFIT_EVERY new rows.
            learning.set_meta(workflow_folderpath, "refit_marker", {**marker, "version": version, "published": False})
        return report
    if not publish:
        report.reason = "dry run: the gate passed, nothing was written"
        report.seconds = time.monotonic() - started
        return report

    new_coef, new_intercept = eh.fit_head(
        np.concatenate([base["vectors"], learned_vectors]), np.concatenate([base["targets"], learned_targets]),
        len(labels), np.concatenate([base["weights"], learned_weights]))
    report.new_version = _publish(workflow_folderpath, version, model_dir, scopes_file, new_coef, new_intercept,
                                  report, fingerprint, len(base["targets"]))
    report.published = True
    report.seconds = time.monotonic() - started
    learning.set_meta(workflow_folderpath, "refit_marker", {**marker, "version": report.new_version, "published": True})
    return report


def _gate_text(gate: dict) -> str:
    a, b = gate["base_rows"], gate["learned_rows"]
    parts = []
    if a.get("checked"):
        parts.append(f"base rows {a['accuracy_before']:.1%} -> {a['accuracy_after']:.1%}"
                     + (f" ({a['rule']})" if a.get("rule") == "base tolerance" else ""))
    if b.get("checked"):
        parts.append(f"learned rows right outright {b['current']['right_outright']:.1%} -> "
                     f"{b['refit']['right_outright']:.1%}, wrong confident {b['current']['wrong_confident']:.1%} -> "
                     f"{b['refit']['wrong_confident']:.1%}"
                     + (f" ({b['rule']})" if b.get("rule") == "wrong confident tolerance" else ""))
    return "; ".join(parts)


def _publish(workflow_folderpath, base_version, base_model_dir, scopes_file, coef, intercept, report, fingerprint,
             base_rows) -> str:
    new_id = artifact_versioning.new_version_id()
    base_dir = artifact_versioning.version_dir(workflow_folderpath, base_version)
    new_dir = artifact_versioning.version_dir(workflow_folderpath, new_id)
    artifact_versioning.ensure_versions_root(workflow_folderpath)
    shutil.copytree(base_dir, new_dir, symlinks=True)
    model_dir = new_dir / layout.WORKFLOW_MODEL_DIRNAME
    payload = json.loads(json.dumps(scopes_file))
    payload["rows"]["learned"] = report.learned_rows
    payload["rows"]["total"] = base_rows + report.learned_rows
    payload["refit"] = {"of": base_version, "learned_rows": report.learned_rows, "gate": report.gate,
                        "learned_weight": eh.SEED_WEIGHT}
    np.savez(model_dir / layout.HEAD_FILENAME, coef=coef, intercept=intercept)
    with open(model_dir / layout.SCOPES_FILENAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    base_manifest = artifact_versioning.read_manifest(workflow_folderpath, base_version)
    lineage = [*learning.lineage_of(workflow_folderpath, base_version), new_id]
    summary = eh.read_summary(str(model_dir))
    report.artifact_bytes = (summary or {}).get("bytes", 0)
    artifact_versioning.write_manifest(
        workflow_folderpath, new_id,
        created_at=artifact_versioning._utc_now(),
        previous_version=base_version, refit_of=base_version, lineage=lineage, learned_rows=report.learned_rows,
        learned_fingerprint=fingerprint, refit_gate=report.gate,
        notes=f"refit of {base_version} with {report.learned_rows} learned rows (intent learning)",
        **({"intent_embedding_head": summary} if summary else {}),
        **({"seed": base_manifest["seed"]} if "seed" in base_manifest else {}))
    with artifact_versioning.publication_lock(workflow_folderpath):
        artifact_versioning.publish_version(workflow_folderpath, new_id)
        # The full train this lineage descends from is kept whatever the retention window says.
        artifact_versioning.retain_current_and_previous(workflow_folderpath, base_version, also_keep=[lineage[0]])
    return new_id


def learn(workflow_folderpath: str, *, publish: bool = True) -> dict:
    """What ``fastworkflow learn`` does: the store's statistics, the refit with its gate, and the calibration of the
    commit threshold for the (possibly new) current head lineage. Makes no model call."""
    result: dict = {"stats": learning.stats(workflow_folderpath)}
    with refit_lock(workflow_folderpath) as acquired:
        report = refit(workflow_folderpath, publish=publish) if acquired else RefitReport(
            reason="another refit of this workflow is running")
    result["refit"] = report
    version, model_dir, scopes_path = _head_of_current(workflow_folderpath)
    result["calibration"] = None
    if version and os.path.isfile(scopes_path):
        with open(scopes_path, encoding="utf-8") as f:
            scopes_file = json.load(f)
        if scopes_file.get("layout") == layout.EMBEDDING and int(scopes_file.get("schema_version", 1)) >= 2:
            lineage = learning.lineage_of(workflow_folderpath, version)
            decision = learning.calibrate(
                workflow_folderpath, base_threshold=float(scopes_file["threshold"]), lineage=lineage)
            if publish:
                learning.store_calibration(workflow_folderpath, decision, lineage)
            result["calibration"] = decision
    return result


def format_report(result: dict) -> str:
    """The short report ``fastworkflow learn`` prints."""
    stats, report, calibration = result["stats"], result["refit"], result["calibration"]
    lines = [f"Recorded requests: {stats['rows']} ({stats['confirmed']} confirmed, {stats['corrections']} "
             f"corrections); {stats['teaching_rows']} teach."]
    lines.append(report.summary_line() + (f" Gate: {_gate_text(report.gate)}." if report.gate else ""))
    if calibration is None:
        lines.append("Commit threshold: not calibrated (no embedding head).")
    elif calibration["threshold"] is None:
        best = max(calibration["evidence"], key=lambda e: e["band_rows"], default=None)
        lines.append(
            f"Commit threshold stays {calibration['base']:.2f}: {calibration['confirmed']} confirmed requests of "
            f"{calibration['needs']['confirmed']} needed"
            + (f"; the best candidate {best['candidate']:.2f} has {best['band_rows']} rows in its band of "
               f"{calibration['needs']['band']} needed, {best['band_wrong']} wrong "
               f"({best['wrong_share_of_band']:.1%} of the band, at most {calibration['needs']['max_wrong_share']:.0%})"
               if best else "") + ".")
    else:
        lines.append(f"Commit threshold lowered to {calibration['threshold']:.2f} from {calibration['base']:.2f} "
                     f"({calibration['confirmed']} confirmed requests).")
    return "\n".join(lines)
