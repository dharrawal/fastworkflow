"""One intent model for a whole workflow, read through a per-scope output mask.

The per-context heads train one classifier pair for every context, and a command inherited
by several contexts is learned several times. This module trains ONE pair over the
workflow's whole label set (every command once, plus ``parameter_value`` and one
workflow-wide ``wildcard``) and answers for a scope by masking the output to that scope's
commands. "Not here" needs no training rows of its own: it is "the best label lies outside
the mask".

Four pieces, in the order a run uses them:

* `workflow_rows` rebuilds the labelled rows from the utterance caches. It never calls a
  model and never writes into the workflow folder.
* `train_workflow_model` fits the two checkpoints and calibrates every threshold.
* `calibrate_scopes` is the calibration on its own (the thresholds of every scope and the
  workflow's escalation margin), so the rule can be re-run without refitting.
* `WorkflowModelRouter` is the runtime reader, with `CommandRouter`'s interface.
* `LayoutTrainer` is what `model_pipeline_training.train` hands its per-context rows to when a
  workflow is trained in the ``workflow`` artifact layout (``INTENT_MODEL_LAYOUT=workflow``,
  see `intent_model_layout`). It fits the large model alone and writes the layout's files.
  The experiment runner and the trainer go through the same row assembly, fit and
  calibration functions in this module, so what was measured is what ships.

The decision in a scope (design section 6 of ``docs/unified_command_pipeline_design.md``):

1. Softmax over all labels. ``p_in`` is the largest probability inside the scope's mask,
   ``p_out`` the largest outside it, not counting ``parameter_value``.
2. ``parameter_value`` or ``wildcard`` as the overall best label is returned as that label.
3. ``wildcard`` (not mine) when ``p_out > p_in + margin``. A foreign command with the same
   simple name as a local one is not counted in ``p_out``: nearest scope wins.
4. Otherwise the probabilities are renormalised inside the mask: one label above the
   scope's ambiguity threshold for the tier that answered, else the top three.
5. The tiny model answers when its masked confidence is at or above the scope's tier
   threshold, else the large model answers.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import fastworkflow
from fastworkflow.train import model_pipeline_training as heads
from fastworkflow.intent.nlu_labels import (
    PARAMETER_VALUE_LABEL,
    PARAMETER_VALUE_PLACEHOLDERS,
    WILDCARD_LABEL,
    label_of,
)
from fastworkflow.train import determinism, heldout_evaluation, intent_model_layout, utterance_cache
from fastworkflow.train.intent_model_layout import (
    LABEL_ENCODER_FILENAME,
    LARGE_MODEL_DIRNAME,
    SCOPES_FILENAME,
)
from fastworkflow.train.selective_training import contexts_for_training
from fastworkflow.utils.logging import logger

TINY_MODEL_DIRNAME = "tinymodel.pth"
SCOPES_SCHEMA_VERSION = 1
#: The tiers a model folder holds, as its ``scopes.json`` names them.
TIERS_TWO = ("tiny", "large")
TIERS_LARGE_ONLY = ("large",)

#: The name a paired-input model reads for the global context, whose own name is "*".
GLOBAL_SCOPE_DISPLAY_NAME = "global"

# The shipped hyperparameters of `model_pipeline_training.train`. They are restated here
# because that function keeps them as locals of one long loop; changing either copy alone
# makes the single model and the heads incomparable.
TINY_LEARNING_RATE = 1e-4
TINY_EPOCHS = 12
LARGE_LEARNING_RATE = 5e-5
LARGE_EPOCHS = 5
BATCH_SIZE = 10
MAX_LENGTH = 128
DEFAULT_TINY_CHECKPOINT = "google/bert_uncased_L-4_H-128_A-2"
DEFAULT_LARGE_CHECKPOINT = "distilbert-base-uncased"

#: Share of in-scope calibration rows the escalation margin may let escalate.
MAX_FALSE_ESCALATION_RATE = 0.02
#: Points of the tier-threshold sweep, as in `find_optimal_threshold`.
TIER_SWEEP_POINTS = 20
#: Weight of large-model usage in the sweep's score, as in `find_optimal_threshold`.
TIER_SWEEP_USAGE_WEIGHT = 0.15
#: Candidates returned when the answer is not confident, as the heads do.
TOP_K = 3

#: Calibration computed on the persona holdout instead of the in-distribution split.
HOLDOUT_SCOPES_FILENAME = "scopes_holdout.json"
CALIBRATION_SPLIT = "split"
CALIBRATION_HOLDOUT = "holdout"
#: Share of lone labels that must be right on the held-out rows at the ambiguity threshold.
LONE_LABEL_PRECISION = 0.90
#: Fewer held-out rows than this in a scope and the workflow-wide threshold is used.
MIN_HOLDOUT_ROWS_PER_SCOPE = 20

DECISION_MATCH = "match"
DECISION_AMBIGUOUS = "ambiguous"
DECISION_NOT_MINE = "not_mine"
DECISION_BARE_VALUE = "bare_value"


class UtteranceCacheMissError(RuntimeError):
    """A command's utterances are not in the cache, so the rows cannot be rebuilt offline."""


class ModelCallAttemptedError(RuntimeError):
    """Something tried to call a language model while rows were being rebuilt offline."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class ScopeRows:
    """One context's view of the workflow: its mask and the rows its head was trained on."""

    name: str
    #: The labels a decision in this scope may return as a command, sorted.
    mask: list[str]
    #: Every cached row of each command in the mask, before any holdout.
    cached_rows: dict[str, list[str]]
    #: The rows left after the head's whole-persona holdout: what its head trained on.
    trained_rows: dict[str, list[str]]
    heldout_personas: list[str] = field(default_factory=list)
    #: The commands declared by this context itself, not inherited and not core ones.
    own_commands: list[str] = field(default_factory=list)


@dataclass
class WorkflowRows:
    """The labelled rows of one workflow, per scope and as one global set."""

    workflow_folderpath: str
    scopes: dict[str, ScopeRows]
    #: ``(utterance, label)`` for the single model: every command once, the bare values,
    #: and the workflow-wide wildcard class.
    rows: list[tuple[str, str]]
    #: Plain-language findings of the consistency checks; empty when everything agreed.
    findings: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return sorted({label for _text, label in self.rows})

    def mask_of(self, scope: str) -> list[str]:
        return list(self.scopes[scope].mask)


class _ReadOnlyUtteranceCache(utterance_cache.UtteranceCache):
    """A cache that answers hits and refuses everything else.

    The stock cache treats a miss as "call the model and store the result". Rebuilding
    rows must do neither, so a miss stops the run and a write is an error.
    """

    def lookup(self, fingerprint, seed):
        entry = super().lookup(fingerprint, seed)
        if entry is None:
            raise UtteranceCacheMissError(
                f"No cached utterances for '{fingerprint.command_name}' (seed {seed}, "
                f"model {fingerprint.inputs.get('model')!r}) under {self.root}. The rows "
                f"can only be rebuilt from a cache written with the same generation "
                f"settings; check LLM_SYNDATA_GEN and the proxy base against the "
                f"'fingerprint_inputs' of the files there."
            )
        return entry

    def store(self, *args, **kwargs):
        raise ModelCallAttemptedError(
            "The read-only utterance cache was asked to store generated utterances."
        )


@contextlib.contextmanager
def no_model_calls() -> Iterator[None]:
    """Make every litellm completion raise for the duration of the block."""
    import litellm

    def _refuse(*_args, **_kwargs):
        raise ModelCallAttemptedError(
            "A language model call was attempted while rebuilding training rows offline."
        )

    saved = {name: getattr(litellm, name) for name in ("completion", "acompletion")}
    for name in saved:
        setattr(litellm, name, _refuse)
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(litellm, name, original)


def cached_generation_model(workflow_folderpath: str) -> Optional[str]:
    """The model string most of the workflow's cached utterances were generated with.

    The cache key includes the generating model, so rows can only be read back under the
    same ``LLM_SYNDATA_GEN``. An env file edited since the last training run would
    otherwise turn every lookup into a miss.
    """
    root = os.path.join(
        workflow_folderpath, determinism.COMMAND_INFO_FOLDERNAME, utterance_cache.CACHE_DIRNAME
    )
    models: Counter = Counter()
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                    model = json.load(f).get("fingerprint_inputs", {}).get("model")
            except (OSError, ValueError):
                continue
            if model:
                models[model] += 1
    return models.most_common(1)[0][0] if models else None


def _persona_split(
    scope_rows: Mapping[str, Sequence[str]], recorder: determinism.ProvenanceRecorder
) -> heldout_evaluation.PersonaSplit:
    """The whole-persona split of one scope's own rows, exactly as `train` draws it for a head.

    The split reads the recorder as it stands when this context is reached, which is why
    `workflow_rows` visits contexts in the trainer's order.
    """
    persona_by_utterance: dict[str, str] = {}
    for provenance in recorder.records.values():
        persona_by_utterance.update(provenance.utterance_personas)
    labeled = [
        heldout_evaluation.LabeledUtterance(
            utterance=utterance,
            label=label,
            persona=persona_by_utterance.get(utterance, heldout_evaluation.SEED_PERSONA_ID),
        )
        for label, utterances in scope_rows.items()
        for utterance in utterances
    ]
    split = heldout_evaluation.split_by_persona(
        labeled,
        seed=determinism.get_split_seed(),
        persona_batches=heads._recorded_persona_batches(recorder),
    )
    if split.heldout:
        return split
    # Nothing could be held out: every row trains, in the order it was given.
    return heldout_evaluation.PersonaSplit(
        train=labeled, heldout=[], train_personas=split.train_personas,
        heldout_personas=[], notes=split.notes)


def _persona_holdout(
    scope_rows: Mapping[str, Sequence[str]], recorder: determinism.ProvenanceRecorder
) -> tuple[dict[str, list[str]], list[str]]:
    """Remove the personas a head holds out, exactly as `train` does for that head.

    The heads never train on their held-out personas' rows, so the single model must not
    either, or it would be compared with more data than the baseline had.
    """
    split = _persona_split(scope_rows, recorder)
    trained: dict[str, list[str]] = {label: [] for label in scope_rows}
    for record in split.train:
        trained[record.label].append(record.utterance)
    return trained, list(split.heldout_personas)


def build_scope_rows(
    scope_name: str,
    by_command: Mapping[str, Sequence[str]],
    crd,
    recorder: Optional[determinism.ProvenanceRecorder],
    *,
    apply_persona_holdout: bool = True,
) -> tuple[Optional[ScopeRows], Optional[heldout_evaluation.PersonaSplit]]:
    """One context's rows for the single model, from the utterances of each of its commands.

    *by_command* is what `cache_context_command_utterances` returns for the context. The
    offline rebuild (`workflow_rows`) and the trainer (`LayoutTrainer`) both come through
    here, so a warm cache gives the two the same rows. Returns ``(None, None)`` for a context
    with no utterances, which the per-context trainer skips too; the split is None when no
    persona holdout was drawn (no recorder, or the caller asked for none).
    """
    cached = {
        command: list(utterances) for command, utterances in by_command.items() if utterances
    }
    if not cached:
        return None, None
    split: Optional[heldout_evaluation.PersonaSplit] = None
    if apply_persona_holdout and recorder is not None:
        split = _persona_split(cached, recorder)
        trained: dict[str, list[str]] = {label: [] for label in cached}
        for record in split.train:
            trained[record.label].append(record.utterance)
        heldout_personas = list(split.heldout_personas)
    else:
        trained, heldout_personas = {c: list(r) for c, r in cached.items()}, []
    scope = ScopeRows(
        name=scope_name,
        mask=sorted(cached),
        cached_rows=cached,
        trained_rows=trained,
        heldout_personas=heldout_personas,
        own_commands=sorted(heads.own_command_names(crd, scope_name) & set(cached)),
    )
    return scope, split


def assemble_rows(
    scopes: Mapping[str, ScopeRows], wildcard_weight: Optional[int] = None
) -> tuple[list[tuple[str, str]], list[str]]:
    """The single model's labelled rows from every scope's rows, and what was odd about them.

    Every command once with the union of the rows its scopes trained on, the bare values
    with their literals, and one workflow-wide ``wildcard`` class: each built-in vague reply
    repeated *wildcard_weight* times (the heads' weight when None; 0 leaves the class out).
    """
    findings: list[str] = []
    commands = sorted({command for scope in scopes.values() for command in scope.mask})
    rows: list[tuple[str, str]] = []
    for command in commands:
        cached_variants = {
            tuple(scope.cached_rows[command])
            for scope in scopes.values()
            if command in scope.cached_rows
        }
        if len(cached_variants) > 1:
            findings.append(f"{command}: its cached rows differ between scopes")
        holders = [
            scope.trained_rows[command]
            for scope in scopes.values()
            if command in scope.trained_rows
        ]
        union = _union_rows(holders)
        if any(Counter(rows_in_scope) != Counter(union) for rows_in_scope in holders):
            sizes = ", ".join(
                f"{scope.name}={len(scope.trained_rows[command])}"
                for scope in scopes.values()
                if command in scope.trained_rows
            )
            findings.append(
                f"{command}: scopes hold out different personas, so the union has "
                f"{len(union)} rows ({sizes})"
            )
        rows.extend((text, command) for text in union)

    command_texts = {text for text, _label in rows}
    rows.extend(
        (text, PARAMETER_VALUE_LABEL)
        for text in sorted(set(PARAMETER_VALUE_PLACEHOLDERS) - command_texts)
    )
    weight = heads.BUILT_IN_VAGUE_WEIGHT if wildcard_weight is None else int(wildcard_weight)
    rows.extend([(text, WILDCARD_LABEL) for text in heads.BUILT_IN_VAGUE_REPLIES] * weight)
    return rows, findings


def _union_rows(per_scope: Sequence[Sequence[str]]) -> list[str]:
    """One command's rows across scopes: each text as often as the scope that has it most.

    A repeated text inside one scope is two training rows and stays two; the same text
    seen in two scopes is one row, because the single model learns the command once.
    """
    most: Counter = Counter()
    order: list[str] = []
    for rows in per_scope:
        counts = Counter(rows)
        for text in rows:
            if text not in most:
                order.append(text)
            most[text] = max(most[text], counts[text])
    return [text for text in order for _ in range(most[text])]


def _provenance_findings(
    workflow_folderpath: str, scopes: Mapping[str, ScopeRows]
) -> list[str]:
    """Compare the rebuilt rows with what the shipped training run recorded."""
    findings: list[str] = []
    info = os.path.join(workflow_folderpath, determinism.COMMAND_INFO_FOLDERNAME)
    try:
        with open(os.path.join(info, determinism.PROVENANCE_FILENAME), "r") as f:
            recorded = json.load(f).get("context_training", {})
    except (OSError, ValueError) as error:
        return [f"training provenance not readable ({error}); row counts unchecked"]

    for scope_name, scope in sorted(scopes.items()):
        recorded_scope = recorded.get(scope_name)
        if recorded_scope is None:
            findings.append(f"scope {scope_name}: no recorded context training to compare")
            continue
        recorded_counts = {
            command: entry.get("row_count", 0)
            for command, entry in recorded_scope.items()
            if label_of(command) not in (WILDCARD_LABEL, PARAMETER_VALUE_LABEL)
            and entry.get("row_count", 0)
        }
        rebuilt_counts = {command: len(rows) for command, rows in scope.cached_rows.items()}
        for command in sorted(set(recorded_counts) | set(rebuilt_counts)):
            if recorded_counts.get(command) != rebuilt_counts.get(command):
                findings.append(
                    f"scope {scope_name}, {command}: {rebuilt_counts.get(command)} rows "
                    f"rebuilt, {recorded_counts.get(command)} recorded at training"
                )

    try:
        with open(os.path.join(info, heldout_evaluation.REPORT_FILENAME), "r") as f:
            reports = {r["context"]: r for r in json.load(f).get("contexts", [])}
    except (OSError, ValueError):
        reports = {}
    for scope_name, scope in sorted(scopes.items()):
        report = reports.get(scope_name)
        if report is None:
            continue
        if sorted(report.get("heldout_personas") or []) != sorted(scope.heldout_personas):
            findings.append(
                f"scope {scope_name}: rebuilt persona holdout {sorted(scope.heldout_personas)} "
                f"differs from the shipped head's {sorted(report.get('heldout_personas') or [])}"
            )
    return findings


def workflow_rows(
    workflow_folderpath: str,
    *,
    env_vars: Optional[Mapping[str, str]] = None,
    apply_persona_holdout: bool = True,
    wildcard_weight: Optional[int] = None,
) -> WorkflowRows:
    """Rebuild the workflow's labelled rows from its utterance caches, with no model call.

    ``wildcard_weight`` is how often each built-in vague reply is repeated in the
    workflow-wide ``wildcard`` class: the heads' weight when None, and 0 leaves the class
    out, so the model has no label of its own for "none of these".

    ``env_vars`` initialises fastworkflow first when given; the generating model and the
    proxy base in it must be the ones the caches were written with (see
    `cached_generation_model`). A cache miss raises `UtteranceCacheMissError`.

    Loading a workflow refreshes ``command_directory.json`` and ``routing_definition.json``
    in its ``___command_info``. Pass a scratch copy of the folder when the original must
    stay byte-identical; nothing else is written.
    """
    if env_vars is not None:
        fastworkflow.init(env_vars=dict(env_vars))

    previous_recorder = determinism.get_provenance_recorder()
    previous_cache = utterance_cache.get_utterance_cache()
    recorder = determinism.ProvenanceRecorder(workflow_folderpath)
    cache = _ReadOnlyUtteranceCache(workflow_folderpath, mode=utterance_cache.MODE_REUSE)
    workflow = fastworkflow.Workflow.create(
        workflow_folderpath, workflow_id_str=f"workflow_rows_{workflow_folderpath}"
    )
    scopes: dict[str, ScopeRows] = {}
    try:
        determinism.set_provenance_recorder(recorder)
        utterance_cache.set_utterance_cache(cache)
        crd = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath, load_cached=False)
        context_cache: dict = {}
        command_cache: dict[str, list[str]] = {}
        with no_model_calls():
            # Sorted, and with the two caches shared across contexts, as `train` does: the
            # persona holdout of a context reads the generation records made so far.
            for scope_name in sorted(contexts_for_training(workflow.folderpath)):
                by_command = heads.cache_context_command_utterances(
                    scope_name, crd, workflow, context_cache, command_cache
                )
                scope, _split = build_scope_rows(
                    scope_name, by_command, crd, recorder,
                    apply_persona_holdout=apply_persona_holdout)
                if scope is not None:
                    scopes[scope_name] = scope
    finally:
        determinism.set_provenance_recorder(previous_recorder)
        utterance_cache.set_utterance_cache(previous_cache)
        workflow.close()

    if cache.stats["miss"] or cache.stats["stored"]:
        raise UtteranceCacheMissError(
            f"The utterance cache reported {cache.stats['miss']} miss(es) and "
            f"{cache.stats['stored']} write(s) while rebuilding rows; expected none."
        )

    findings = _provenance_findings(workflow_folderpath, scopes)
    rows, row_findings = assemble_rows(scopes, wildcard_weight)
    findings.extend(row_findings)
    return WorkflowRows(
        workflow_folderpath=workflow_folderpath, scopes=scopes, rows=rows, findings=findings
    )


# ---------------------------------------------------------------------------
# The per-context heads, rebuilt
# ---------------------------------------------------------------------------


def recorded_not_here_samples(workflow_folderpath: str) -> dict[str, dict[str, list[str]]]:
    """The "not here" samples each shipped head recorded, ``{context: {command: sample}}``.

    Read from the training signature of the workflow's current version. A head rebuilt with
    these trains on the same foreign seed rows as the shipped one.
    """
    from fastworkflow.train import artifact_versioning, selective_training

    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        return {}
    path = (artifact_versioning.version_dir(workflow_folderpath, version)
            / selective_training.SIGNATURE_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        contexts = json.load(f).get("contexts", {})
    return {
        name: {command: list(sample) for command, sample in (entry.get("not_here_samples") or {}).items()}
        for name, entry in contexts.items()
    }


def rebuild_heads(
    workflow_folderpath: str,
    *,
    seed: int,
    env_vars: Optional[Mapping[str, str]] = None,
    not_here_samples: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
    built_in_vague_class: bool = True,
) -> dict[str, str]:
    """Train the per-context heads with the shipped trainer, offline, at one model seed.

    This calls `model_pipeline_training.train` itself rather than restating it, so the rows
    (own rows, "not here" seeds, bare values, the persona holdout), the fitting loop and
    every threshold rule are the shipped ones by construction. The utterance cache is
    read-only and model calls raise, as in `workflow_rows`.

    *workflow_folderpath* must be a scratch copy with no published version: the trainer
    then writes each head's six files to ``___command_info/<context>/``, where a plain
    `CommandRouter` loads them. ``built_in_vague_class=False`` rebuilds a head that was
    trained before a head without foreign commands got the built-in vague replies as its
    ``wildcard`` class. Returns ``{context: artifact folder}``.
    """
    from fastworkflow.train import artifact_versioning

    if env_vars is not None:
        fastworkflow.init(env_vars=dict(env_vars))
    if artifact_versioning.resolve_current_version(workflow_folderpath) is not None:
        raise ValueError(
            f"{workflow_folderpath} has a published version; rebuild heads on a scratch copy "
            f"without its versions folder so nothing published is overwritten."
        )

    previous_recorder = determinism.get_provenance_recorder()
    previous_cache = utterance_cache.get_utterance_cache()
    previous_seed = determinism.get_train_seed()
    vague_replies = heads.BUILT_IN_VAGUE_REPLIES
    cache = _ReadOnlyUtteranceCache(workflow_folderpath, mode=utterance_cache.MODE_REUSE)
    workflow = fastworkflow.Workflow.create(
        workflow_folderpath, workflow_id_str=f"rebuild_heads_{seed}_{workflow_folderpath}"
    )
    try:
        determinism.set_provenance_recorder(determinism.ProvenanceRecorder(workflow_folderpath))
        utterance_cache.set_utterance_cache(cache)
        determinism.set_train_seed(seed)
        if not built_in_vague_class:
            heads.BUILT_IN_VAGUE_REPLIES = ()
        with no_model_calls():
            heads.train(workflow, not_here_samples=not_here_samples)
        contexts = sorted(contexts_for_training(workflow.folderpath))
    finally:
        heads.BUILT_IN_VAGUE_REPLIES = vague_replies
        determinism.set_train_seed(previous_seed)
        determinism.set_provenance_recorder(previous_recorder)
        utterance_cache.set_utterance_cache(previous_cache)
        workflow.close()
    if cache.stats["miss"] or cache.stats["stored"]:
        raise UtteranceCacheMissError("The utterance cache was not fully warm for the heads.")
    info = os.path.join(workflow_folderpath, determinism.COMMAND_INFO_FOLDERNAME)
    return {
        name: os.path.join(info, artifact_versioning.context_folder_name(name))
        for name in contexts
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def scope_display_name(scope: str) -> str:
    """The text a paired-input model reads as its first segment for *scope*."""
    return GLOBAL_SCOPE_DISPLAY_NAME if scope == "*" else scope


def _encode(tokenizer, texts: Sequence[str], scope_names: Optional[Sequence[str]]):
    """Tokenise utterances, as a (scope name, utterance) pair when scope names are given."""
    arguments = (list(scope_names), list(texts)) if scope_names is not None else (list(texts),)
    return tokenizer(
        *arguments, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )


def _fit(model, tokenizer, examples, *, learning_rate: float, epochs: int, paired: bool) -> None:
    """Fine-tune *model* on ``(text, label index, scope name)`` examples.

    The loop of `model_pipeline_training.train`: AdamW, shuffled batches of ten, no
    schedule. That loop evaluates after every epoch and its evaluation leaves the model in
    evaluation mode, so only the first epoch trains with dropout. Kept here on purpose:
    dropout in every epoch would be a different training recipe from the heads'.
    """

    def collate(batch):
        encodings = _encode(
            tokenizer,
            [item[0] for item in batch],
            [item[2] for item in batch] if paired else None,
        )
        return encodings, torch.tensor([item[1] for item in batch], dtype=torch.long)

    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for encodings, labels in loader:
            encodings = {name: tensor.to(heads.device) for name, tensor in encodings.items()}
            optimizer.zero_grad()
            loss = model(**encodings, labels=labels.to(heads.device)).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            f"workflow model epoch {epoch + 1}/{epochs}: loss {total_loss / max(len(loader), 1):.4f}"
        )
        model.eval()


@torch.no_grad()
def label_probabilities(
    model, tokenizer, texts: Sequence[str], scope_name: Optional[str] = None
) -> np.ndarray:
    """Softmax over every label for each text; ``scope_name`` makes the input a pair."""
    model.eval()
    chunks = []
    for start in range(0, len(texts), 32):
        batch = list(texts[start:start + 32])
        encodings = _encode(tokenizer, batch, [scope_name] * len(batch) if scope_name else None)
        encodings = {name: tensor.to(heads.device) for name, tensor in encodings.items()}
        chunks.append(torch.softmax(model(**encodings).logits, dim=1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, model.config.num_labels))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def confidence_stats(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> dict:
    """The statistics `analyze_model_confidence` returns, from arrays instead of a loader.

    The shipped function runs a model over a DataLoader and takes the unmasked softmax. The
    threshold rules that consume its result only read these numbers, so building them from
    masked probabilities lets the rules themselves be reused unchanged.
    """
    correct = predictions == labels

    def summary(values: np.ndarray) -> dict:
        present = len(values) > 0
        return {
            "min": float(np.min(values)) if present else None,
            "max": float(np.max(values)) if present else None,
            "mean": float(np.mean(values)) if present else None,
            "median": float(np.median(values)) if present else None,
            "count": int(len(values)),
        }

    successful = summary(confidences[correct])
    successful["low_quantile"] = (
        float(np.quantile(confidences[correct], heads.CORRECT_CONFIDENCE_LOW_QUANTILE))
        if correct.any() else None
    )
    return {"failed": summary(confidences[~correct]), "successful": successful}


def _ndcg(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    """NDCG at *k* as `ModelPipeline.evaluate` reports it: the mean of per-batch means."""
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]
    gains = np.zeros(len(labels))
    for position in range(order.shape[1]):
        gains += (order[:, position] == labels) / np.log2(position + 2)
    batches = [gains[start:start + BATCH_SIZE] for start in range(0, len(gains), BATCH_SIZE)]
    return float(np.mean([batch.mean() for batch in batches])) if batches else 0.0


def sweep_tier_threshold(
    tiny: np.ndarray, large: np.ndarray, labels: np.ndarray, tiny_stats: dict
) -> dict:
    """The tier threshold by the shipped sweep, over masked probabilities.

    `find_optimal_threshold` needs a `ModelPipeline` and a DataLoader; this is the same
    search (twenty points between the tiny model's mean failed and mean successful
    confidence, scored by F1 x NDCG x a penalty on large-model usage) on arrays. With no
    failed or no successful row there is nothing to sweep and the sentinel -1 is returned,
    which leaves every decision to the tiny model, as the shipped function does.
    """
    low, high = tiny_stats["failed"]["mean"], tiny_stats["successful"]["mean"]
    tiny_confidence, tiny_prediction = tiny.max(axis=1), tiny.argmax(axis=1)
    large_prediction = large.argmax(axis=1)
    k = min(TOP_K, tiny.shape[1])

    def evaluate(threshold: float) -> dict:
        use_large = tiny_confidence < threshold
        predictions = np.where(use_large, large_prediction, tiny_prediction)
        blended = np.where(use_large[:, None], large, tiny)
        return {
            "threshold": float(threshold),
            "f1": float(f1_score(labels, predictions, average="weighted")),
            "ndcg": _ndcg(blended, labels, k),
            "large_usage": float(use_large.mean() * 100),
        }

    if not low or not high:
        result = evaluate(-1.0)
        result["swept"] = False
        return result
    results = [evaluate(threshold) for threshold in np.linspace(low, high, TIER_SWEEP_POINTS)]
    best = max(
        results,
        key=lambda r: r["f1"] * r["ndcg"] * (1 - TIER_SWEEP_USAGE_WEIGHT * (r["large_usage"] / 100)),
    )
    best["swept"] = True
    return best


def _outside_mask(labels: Sequence[str], mask: Sequence[str]) -> np.ndarray:
    """Which labels count toward ``p_out`` for a scope with *mask*.

    Not the mask itself, not ``parameter_value``, and not a foreign command that shares its
    simple name with a local one: the same wording under two contexts is the local command.
    """
    local = set(mask)
    local_simple_names = {name.rsplit("/", 1)[-1] for name in local}
    return np.array([
        label not in local
        and label != PARAMETER_VALUE_LABEL
        and (label == WILDCARD_LABEL or label.rsplit("/", 1)[-1] not in local_simple_names)
        for label in labels
    ])


def smallest_margin(gaps: Sequence[float], reserved_escalations: int, row_count: int) -> float:
    """The smallest margin at which at most 2% of in-scope rows escalate.

    ``gaps`` are ``p_out - p_in`` for the rows a margin can keep; ``reserved_escalations``
    counts the rows that escalate whatever the margin (their best label is the workflow
    wildcard). Never negative: a negative margin would escalate a row whose best label is
    local.
    """
    allowed = int(MAX_FALSE_ESCALATION_RATE * row_count) - reserved_escalations
    descending = sorted((float(gap) for gap in gaps), reverse=True)
    if not descending or allowed >= len(descending):
        return 0.0
    if allowed < 0:
        # The reserved escalations alone are over budget; the margin can only stop adding to them.
        return max(0.0, descending[0])
    return max(0.0, descending[allowed])


def calibrate_scopes(
    labels: Sequence[str],
    masks: Mapping[str, Sequence[str]],
    calibration_rows: Sequence[tuple[str, str]],
    probabilities,
    *,
    large_only: bool = False,
) -> dict:
    """Thresholds for every scope and the workflow's escalation margins.

    ``probabilities(tier, texts, scope)`` returns the full softmax of ``"tiny"`` or
    ``"large"`` for *texts* asked in *scope*. A scope is calibrated on the calibration rows
    whose label is in its mask, read through the mask and renormalised.

    ``large_only`` calibrates a model folder that has no tiny model: only the large model is
    asked, and only its numbers are written (the ambiguity threshold of each scope by the
    shipped large-tier rule, and the large-only escalation margin). Those numbers are
    computed exactly as the two-tier call computes them, so a large model reads the same
    under either.
    """
    labels = list(labels)
    index_of = {label: index for index, label in enumerate(labels)}
    wildcard_index = index_of.get(WILDCARD_LABEL)
    tiers = TIERS_LARGE_ONLY if large_only else TIERS_TWO
    modes = ("large_only",) if large_only else ("two_tier", "large_only")
    scopes: dict[str, dict] = {}
    gaps = {mode: [] for mode in modes}
    reserved = {mode: 0 for mode in modes}
    in_scope_rows = 0

    for scope, mask in sorted(masks.items()):
        mask = [label for label in mask if label in index_of]
        mask_index = np.array([index_of[label] for label in mask])
        position_in_mask = {label: position for position, label in enumerate(mask)}
        rows = [(text, label) for text, label in calibration_rows if label in position_in_mask]
        if not rows:
            raise heads.TrainingDataError(
                f"Scope {scope} has no calibration row, so its thresholds cannot be computed."
            )
        texts = [text for text, _label in rows]
        truth = np.array([position_in_mask[label] for _text, label in rows])
        full = {tier: probabilities(tier, texts, scope) for tier in tiers}
        masked = {
            tier: full[tier][:, mask_index] / full[tier][:, mask_index].sum(axis=1, keepdims=True)
            for tier in full
        }
        stats = {
            tier: confidence_stats(masked[tier].max(axis=1), masked[tier].argmax(axis=1), truth)
            for tier in masked
        }
        large_ambiguous, large_rule = heads.large_ambiguous_threshold_and_rule(stats["large"])
        outside = _outside_mask(labels, mask)
        answering = {"large_only": full["large"]}
        entry = {
            "display_name": scope_display_name(scope),
            "mask": mask,
        }
        how = {"calibration_rows": len(rows)}
        if not large_only:
            sweep = sweep_tier_threshold(masked["tiny"], masked["large"], truth, stats["tiny"])
            tier_threshold = sweep["threshold"]
            tiny_answers = masked["tiny"].max(axis=1) >= tier_threshold
            answering["two_tier"] = np.where(tiny_answers[:, None], full["tiny"], full["large"])
            entry["tier_threshold"] = tier_threshold
            entry["tiny_ambiguous_threshold"] = heads.separated_tiny_ambiguous_threshold(
                tier_threshold, stats["tiny"])
            how["tier_threshold"] = (
                "shipped sweep over masked, renormalised probabilities"
                if sweep["swept"] else
                "no sweep: the tiny model had no failed or no successful calibration row"
            )
            how["tier_sweep_result"] = sweep
            how["tiny_ambiguous_threshold"] = "separated_tiny_ambiguous_threshold"
        entry["large_ambiguous_threshold"] = large_ambiguous
        if not large_only:
            entry["in_distribution_f1"] = sweep["f1"]
        entry["in_distribution_f1_large_only"] = float(
            f1_score(truth, masked["large"].argmax(axis=1), average="weighted")
        )
        how["large_ambiguous_threshold"] = f"large_ambiguous_threshold_and_rule: {large_rule}"
        how["large_ambiguous_rule"] = large_rule
        if not large_only:
            how["tiny_stats"] = stats["tiny"]
        how["large_stats"] = stats["large"]
        entry["how"] = how

        for mode in modes:
            answer = answering[mode]
            is_wildcard = (
                answer.argmax(axis=1) == wildcard_index
                if wildcard_index is not None else np.zeros(len(rows), dtype=bool)
            )
            p_in = answer[:, mask_index].max(axis=1)
            p_out = answer[:, outside].max(axis=1) if outside.any() else np.zeros(len(rows))
            reserved[mode] += int(is_wildcard.sum())
            gaps[mode].extend((p_out - p_in)[~is_wildcard].tolist())
        in_scope_rows += len(rows)
        scopes[scope] = entry

    margins = {
        mode: smallest_margin(gaps[mode], reserved[mode], in_scope_rows) for mode in modes
    }
    payload: dict = {"scopes": scopes}
    if not large_only:
        payload["escalation_margin"] = margins["two_tier"]
    payload["escalation_margin_large_only"] = margins["large_only"]
    payload["escalation_margin_how"] = {
        "rule": (
            f"smallest margin >= 0 at which at most {MAX_FALSE_ESCALATION_RATE:.0%} of "
            f"in-scope calibration rows escalate, pooled over scopes"
        ),
        "in_scope_calibration_rows": in_scope_rows,
        "rows_whose_best_label_is_wildcard": dict(reserved),
        "rows_escalating_at_margin_zero": {
            mode: int(sum(gap > 0 for gap in gaps[mode])) for mode in modes
        },
    }
    return payload


def unseen_holdout_rows(rows: WorkflowRows, trained_texts: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Per scope, the rows of its held-out personas that the single model never trained on.

    A scope's head holds out whole personas, but the single model trains on the union of
    every scope's rows, so a row one scope holds out can reach the model through another
    scope that shares the command. Only rows whose text is in no training row are kept.
    """
    unseen: dict[str, list[tuple[str, str]]] = {}
    for name, scope in rows.scopes.items():
        kept: list[tuple[str, str]] = []
        for command in scope.mask:
            remaining = Counter(scope.cached_rows.get(command, [])) - Counter(
                scope.trained_rows.get(command, []))
            for text in scope.cached_rows.get(command, []):
                if remaining[text] > 0:
                    remaining[text] -= 1
                    if text not in trained_texts:
                        kept.append((text, command))
        unseen[name] = kept
    return unseen


def lowest_precise_threshold(confidences: np.ndarray, correct: np.ndarray) -> tuple[float, dict]:
    """The lowest ambiguity threshold at which a lone label is right at least 90% of the time.

    A lone label is given when the confidence is above the threshold. Candidates are the
    floor and every observed confidence above it. Never below the single-label floor; when
    no candidate reaches the precision with at least one lone label left, the ceiling.
    """
    floor, ceiling = heads.SINGLE_LABEL_RESOLUTION_FLOOR, heads.MAX_AMBIGUITY_THRESHOLD
    candidates = [floor] + sorted({float(c) for c in confidences if floor < c < ceiling})
    for threshold in candidates:
        lone = confidences > threshold
        if lone.any() and correct[lone].mean() >= LONE_LABEL_PRECISION:
            return threshold, {
                "rows": int(len(confidences)), "lone_labels": int(lone.sum()),
                "lone_labels_right": int(correct[lone].sum()), "reached_precision": True}
    lone = confidences > ceiling
    return ceiling, {
        "rows": int(len(confidences)), "lone_labels": int(lone.sum()),
        "lone_labels_right": int(correct[lone].sum()) if lone.any() else 0,
        "reached_precision": False}


def calibrate_from_holdout(rows: WorkflowRows, model_dir: str) -> dict:
    """Recompute a trained model's large-tier decision numbers on the persona holdout.

    Nothing is refitted and ``scopes.json`` is left as it is; the result is written beside
    it as ``scopes_holdout.json`` and read with ``calibration="holdout"``. *rows* must be
    the rows the model in *model_dir* was trained from (same workflow, same wildcard
    weight): the training side of its split is rebuilt here to know what it saw.

    Per scope the ambiguity threshold is `lowest_precise_threshold` on the scope's unseen
    held-out rows, read through the mask; a scope with fewer than
    ``MIN_HOLDOUT_ROWS_PER_SCOPE`` rows takes the value computed over every scope's rows.
    The escalation margin is `smallest_margin` over the same rows, pooled.
    """
    with open(os.path.join(model_dir, SCOPES_FILENAME), "r", encoding="utf-8") as f:
        recorded = json.load(f)
    if recorded.get("paired_context"):
        raise ValueError("Holdout calibration is defined for the unpaired model only.")
    labels = list(recorded["labels"])
    index_of = {label: index for index, label in enumerate(labels)}
    if sorted({label for _text, label in rows.rows}) != sorted(labels):
        raise ValueError(f"The rows given do not have the label set of the model in {model_dir}.")
    dataset = [(text, index_of[label]) for text, label in rows.rows]
    train_data, test_data = heads.split_training_data(dataset, lambda encoded: labels[encoded])
    if len(train_data) != recorded["rows"]["training"]:
        raise ValueError(
            f"The rows given split into {len(train_data)} training rows; the model in "
            f"{model_dir} was trained on {recorded['rows']['training']}.")
    trained_texts = {text for text, _encoded in train_data}
    union_texts = {text for text, _label in rows.rows}

    pipeline = WorkflowModelPipeline.for_directory(model_dir)
    unseen = unseen_holdout_rows(rows, trained_texts)
    wildcard_index = index_of.get(WILDCARD_LABEL)
    per_scope: dict[str, dict] = {}
    pooled_confidence, pooled_correct, gaps, reserved = [], [], [], 0
    for name, scope in sorted(rows.scopes.items()):
        mask = [label for label in recorded["scopes"][name]["mask"]]
        mask_index = np.array([index_of[label] for label in mask])
        held_out = unseen[name]
        full = label_probabilities(
            pipeline.distil_model, pipeline.distil_tokenizer, [text for text, _label in held_out])
        inside = full[:, mask_index]
        masked = inside / inside.sum(axis=1, keepdims=True) if len(held_out) else inside
        truth = np.array([mask.index(label) for _text, label in held_out], dtype=int)
        confidence = masked.max(axis=1) if len(held_out) else np.zeros(0)
        correct = (masked.argmax(axis=1) == truth) if len(held_out) else np.zeros(0, dtype=bool)
        if len(held_out):
            outside = _outside_mask(labels, mask)
            is_wildcard = (full.argmax(axis=1) == wildcard_index
                           if wildcard_index is not None else np.zeros(len(held_out), dtype=bool))
            p_out = full[:, outside].max(axis=1) if outside.any() else np.zeros(len(held_out))
            reserved += int(is_wildcard.sum())
            gaps.extend((p_out - inside.max(axis=1))[~is_wildcard].tolist())
        pooled_confidence.extend(confidence.tolist())
        pooled_correct.extend(correct.tolist())
        all_held_out = sum(
            (Counter(scope.cached_rows[c]) - Counter(scope.trained_rows.get(c, []))).total()
            for c in scope.mask)
        per_scope[name] = {
            "confidence": confidence, "correct": correct,
            "counts": {
                "held_out_rows": int(all_held_out),
                "unseen_rows": len(held_out),
                "dropped_trained_through_another_scope": int(all_held_out) - len(held_out),
                "unseen_rows_in_no_scope_s_rows": sum(text not in union_texts for text, _l in held_out),
                "top_label_right": int(correct.sum()),
            },
        }

    total = len(pooled_confidence)
    workflow_threshold, workflow_how = lowest_precise_threshold(
        np.array(pooled_confidence), np.array(pooled_correct, dtype=bool))
    scopes: dict[str, dict] = {}
    for name, entry in per_scope.items():
        enough = len(entry["confidence"]) >= MIN_HOLDOUT_ROWS_PER_SCOPE
        threshold, how = (
            lowest_precise_threshold(entry["confidence"], entry["correct"])
            if enough else (workflow_threshold, workflow_how))
        scopes[name] = {
            "large_ambiguous_threshold": threshold,
            "source": "scope" if enough else "workflow-wide (fewer than "
                      f"{MIN_HOLDOUT_ROWS_PER_SCOPE} unseen held-out rows)",
            "how": how,
            **entry["counts"],
        }
    payload = {
        "schema_version": SCOPES_SCHEMA_VERSION,
        "calibration": CALIBRATION_HOLDOUT,
        "rule": (
            f"per scope, the lowest threshold at which a lone label is right at least "
            f"{LONE_LABEL_PRECISION:.0%} of the time on its unseen held-out rows (large model, "
            f"masked and renormalised); floor {heads.SINGLE_LABEL_RESOLUTION_FLOOR}, ceiling "
            f"{heads.MAX_AMBIGUITY_THRESHOLD}; margin: smallest at which at most "
            f"{MAX_FALSE_ESCALATION_RATE:.0%} of the same rows escalate, pooled over scopes"),
        "unseen_held_out_rows": total,
        "workflow_ambiguous_threshold": workflow_threshold,
        "workflow_how": workflow_how,
        "escalation_margin_large_only": smallest_margin(gaps, reserved, total),
        "escalation_margin_how": {
            "rows_whose_best_label_is_wildcard": reserved,
            "rows_escalating_at_margin_zero": int(sum(gap > 0 for gap in gaps)),
        },
        "scopes": scopes,
    }
    with open(os.path.join(model_dir, HOLDOUT_SCOPES_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _training_examples(
    train_rows: Sequence[tuple[str, str]],
    masks: Mapping[str, Sequence[str]],
    index_of: Mapping[str, int],
    paired_context: bool,
) -> list[tuple[str, int, str]]:
    """``(text, label index, scope name)`` rows for one fit.

    For a paired-input model each row is repeated for every scope whose mask holds its
    label. The two reserved labels are in no mask and are valid everywhere (a bare value or
    a vague reply can be typed in any context), so they are repeated for every scope.
    """
    if not paired_context:
        return [(text, index_of[label], "") for text, label in train_rows]
    examples = []
    for text, label in train_rows:
        for scope, mask in sorted(masks.items()):
            if label in mask or label in (PARAMETER_VALUE_LABEL, WILDCARD_LABEL):
                examples.append((text, index_of[label], scope_display_name(scope)))
    return examples


@dataclass
class FittedWorkflowModel:
    """What one fit wrote and what it was fitted on."""

    #: The content of ``scopes.json``.
    payload: dict
    #: The training side of the in-distribution split, ``(utterance, label)``.
    train_rows: list[tuple[str, str]]
    #: The evaluation side: the rows the thresholds were computed on.
    calibration_rows: list[tuple[str, str]]


def fit_workflow_model(
    rows: WorkflowRows,
    out_dir: str,
    *,
    seed: int,
    paired_context: bool = False,
    large_only: bool = False,
) -> FittedWorkflowModel:
    """Fit the workflow's model, calibrate it, and write the artifacts to *out_dir*.

    Writes ``largemodel.pth/``, ``label_encoder.pkl`` and ``scopes.json``, and
    ``tinymodel.pth/`` unless ``large_only``. With ``paired_context`` the scope name is the
    first segment of a paired input, at training and at inference.

    ``large_only`` is the layout `fastworkflow train` ships (arm S of the second
    pre-registration): the tiny model is neither loaded, fitted nor saved, and
    ``scopes.json`` holds the large model's numbers alone. A large model fitted this way does
    not have the weights of one fitted after a tiny model at the same seed, because the tiny
    fit draws from the same random stream first; its rows, recipe and calibration are the same.

    One stratified in-distribution split (`split_training_data`, its fixed seed) is taken
    over the global rows. The model trains on its training side, and each scope is
    calibrated on the evaluation side read through its mask. A split per scope cannot be
    used as it is for the heads: the scopes share commands, so the evaluation rows of one
    scope would be training rows of another and no threshold would be measured on rows the
    model has not seen.
    """
    if large_only and paired_context:
        raise ValueError("The large-only model is the unpaired one; paired input is an experiment arm.")
    started = time.monotonic()
    determinism.seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    encoder = LabelEncoder()
    encoder.fit([label for _text, label in rows.rows])
    labels = [str(label) for label in encoder.classes_]
    index_of = {label: index for index, label in enumerate(labels)}
    masks = {name: scope.mask for name, scope in rows.scopes.items()}

    dataset = [(text, index_of[label]) for text, label in rows.rows]
    train_data, test_data = heads.split_training_data(dataset, lambda encoded: labels[encoded])
    train_rows = [(text, labels[encoded]) for text, encoded in train_data]
    calibration_rows = [(text, labels[encoded]) for text, encoded in test_data]
    examples = _training_examples(train_rows, masks, index_of, paired_context)

    tiny_name = fastworkflow.get_env_var("INTENT_DETECTION_TINY_MODEL", default=DEFAULT_TINY_CHECKPOINT)
    large_name = fastworkflow.get_env_var("INTENT_DETECTION_LARGE_MODEL", default=DEFAULT_LARGE_CHECKPOINT)
    tiers: dict = {}
    if not large_only:
        # Both checkpoints are loaded before either is fitted, as the heads' trainer does, so
        # a seed draws the same initial classifier weights and batch order in both trainers.
        tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_name)
        tiny_model = AutoModelForSequenceClassification.from_pretrained(
            tiny_name, num_labels=len(labels)).to(heads.device)
        tiers["tiny"] = (tiny_model, tiny_tokenizer)
    large_tokenizer = AutoTokenizer.from_pretrained(large_name)
    large_model = AutoModelForSequenceClassification.from_pretrained(
        large_name, num_labels=len(labels)).to(heads.device)
    tiers["large"] = (large_model, large_tokenizer)

    if not large_only:
        _fit(tiny_model, tiny_tokenizer, examples,
             learning_rate=TINY_LEARNING_RATE, epochs=TINY_EPOCHS, paired=paired_context)
        heads.save_model(tiny_model, tiny_tokenizer, os.path.join(out_dir, TINY_MODEL_DIRNAME))
    _fit(large_model, large_tokenizer, examples,
         learning_rate=LARGE_LEARNING_RATE, epochs=LARGE_EPOCHS, paired=paired_context)
    heads.save_model(large_model, large_tokenizer, os.path.join(out_dir, LARGE_MODEL_DIRNAME))
    with open(os.path.join(out_dir, LABEL_ENCODER_FILENAME), "wb") as f:
        pickle.dump(encoder, f)
    fit_seconds = time.monotonic() - started

    def probabilities(tier: str, texts: Sequence[str], scope: str) -> np.ndarray:
        model, tokenizer = tiers[tier]
        return label_probabilities(
            model, tokenizer, texts, scope_display_name(scope) if paired_context else None
        )

    payload = {
        "schema_version": SCOPES_SCHEMA_VERSION,
        "labels": labels,
        "tiers": list(TIERS_LARGE_ONLY if large_only else TIERS_TWO),
        "paired_context": bool(paired_context),
        "seed": int(seed),
        "base_checkpoints": (
            {"large": large_name} if large_only else {"tiny": tiny_name, "large": large_name}
        ),
        "rows": {
            "total": len(rows.rows),
            "training": len(train_rows),
            "calibration": len(calibration_rows),
            "training_examples_after_pairing": len(examples),
        },
        **calibrate_scopes(labels, masks, calibration_rows, probabilities, large_only=large_only),
    }
    payload["fit_seconds"] = fit_seconds
    payload["train_seconds"] = time.monotonic() - started
    with open(os.path.join(out_dir, SCOPES_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return FittedWorkflowModel(
        payload=payload, train_rows=train_rows, calibration_rows=calibration_rows)


def train_workflow_model(
    rows: WorkflowRows,
    out_dir: str,
    *,
    seed: int,
    paired_context: bool = False,
    large_only: bool = False,
) -> dict:
    """`fit_workflow_model`, returning the content of ``scopes.json`` alone."""
    return fit_workflow_model(
        rows, out_dir, seed=seed, paired_context=paired_context, large_only=large_only).payload


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class WorkflowModelPipeline:
    """The one loaded model pair of a workflow model directory, shared by its scopes.

    Named like `ModelPipeline`'s attributes (``distil_model``, ``distil_tokenizer``,
    ``device``) because the embedding cache reads the large model's encoder through them.
    The tiny model is loaded on first use, so a large-only reader never pays for it.
    """

    _instances: ClassVar[dict[str, "WorkflowModelPipeline"]] = {}
    _build_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = str(heads.device)
        with open(os.path.join(model_dir, SCOPES_FILENAME), "r", encoding="utf-8") as f:
            self.scopes_file = json.load(f)
        self.labels: list[str] = list(self.scopes_file["labels"])
        #: False for a folder trained large-only: there is no tiny model to load.
        self.has_tiny = "tiny" in (self.scopes_file.get("tiers") or TIERS_TWO)
        self.paired_context = bool(self.scopes_file.get("paired_context"))
        large_path = os.path.join(model_dir, LARGE_MODEL_DIRNAME)
        self.distil_tokenizer = AutoTokenizer.from_pretrained(large_path)
        self.distil_model = AutoModelForSequenceClassification.from_pretrained(
            large_path).to(heads.device)
        self.distil_model.eval()
        self._tiny = None

    @classmethod
    def for_directory(cls, model_dir: str) -> "WorkflowModelPipeline":
        # The real path: a context entry reaches the model through a link into its artifact
        # version, so every scope of a version shares one load and a new version is a new key.
        key = os.path.realpath(model_dir)
        with cls._build_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key)
            return cls._instances[key]

    def _tiny_pair(self):
        with self._build_lock:
            if self._tiny is None:
                tiny_path = os.path.join(self.model_dir, TINY_MODEL_DIRNAME)
                model = AutoModelForSequenceClassification.from_pretrained(
                    tiny_path).to(heads.device)
                model.eval()
                self._tiny = (model, AutoTokenizer.from_pretrained(tiny_path))
            return self._tiny

    @property
    def tiny_model(self):
        return self._tiny_pair()[0]

    @property
    def tiny_tokenizer(self):
        return self._tiny_pair()[1]

    def probabilities(self, tier: str, text: str, scope: str) -> np.ndarray:
        """The softmax over every label for one utterance asked in *scope*."""
        model, tokenizer = (
            self._tiny_pair() if tier == "tiny" else (self.distil_model, self.distil_tokenizer)
        )
        scope_name = scope_display_name(scope) if self.paired_context else None
        return label_probabilities(model, tokenizer, [text], scope_name)[0]


class WorkflowModelRouter:
    """`CommandRouter`'s interface over the workflow's one model, for one scope.

    ``predict`` returns one label when confident and the top three otherwise. "Not mine" is
    the single label ``wildcard`` and a bare value the single label ``parameter_value``,
    exactly as the per-context heads report them, so their consumers need no change.
    ``large_only`` answers every decision with the large model, from the same artifacts.
    ``calibration="holdout"`` reads the ambiguity threshold and the escalation margin that
    `calibrate_from_holdout` computed, instead of the ones of the in-distribution split.
    """

    def __init__(self, model_dir: str, context_name: str, *, large_only: bool = False,
                 calibration: str = CALIBRATION_SPLIT):
        if calibration == CALIBRATION_HOLDOUT and not large_only:
            raise ValueError("The holdout calibration holds large-model numbers only; "
                             "read it with large_only=True.")
        self.model_dir = model_dir
        self.context_name = context_name
        self.modelpipeline = WorkflowModelPipeline.for_directory(model_dir)
        # A folder trained without a tiny model can only be read large-only.
        large_only = large_only or not self.modelpipeline.has_tiny
        self.large_only = large_only
        self.calibration = calibration
        self.label_encoder_path = os.path.join(model_dir, LABEL_ENCODER_FILENAME)
        scopes_file = self.modelpipeline.scopes_file
        if context_name not in scopes_file["scopes"]:
            raise KeyError(
                f"Scope {context_name!r} is not in {os.path.join(model_dir, SCOPES_FILENAME)}; "
                f"it has {sorted(scopes_file['scopes'])}."
            )
        scope = scopes_file["scopes"][context_name]
        labels = self.modelpipeline.labels
        index_of = {label: index for index, label in enumerate(labels)}
        self._labels = labels
        self._mask_labels: list[str] = list(scope["mask"])
        self._mask_index = np.array([index_of[label] for label in self._mask_labels])
        self._outside = _outside_mask(labels, self._mask_labels)
        self.large_ambiguous_confidence_threshold = float(scope["large_ambiguous_threshold"])
        # A large-only folder records no tiny numbers. The tier threshold is then the value no
        # probability reaches (the one the context's threshold.json carries), and the tiny
        # ambiguity threshold, which nothing reads, mirrors the large one.
        self.confidence_threshold = float(
            scope.get("tier_threshold", intent_model_layout.WORKFLOW_LAYOUT_TIER_THRESHOLD))
        self.tiny_ambiguous_confidence_threshold = float(
            scope.get("tiny_ambiguous_threshold", self.large_ambiguous_confidence_threshold))
        self.escalation_margin = float(
            scopes_file["escalation_margin_large_only" if large_only else "escalation_margin"]
        )
        if calibration == CALIBRATION_HOLDOUT:
            # The same weights and mask; only the two decision numbers come from the other file.
            with open(os.path.join(model_dir, HOLDOUT_SCOPES_FILENAME), "r", encoding="utf-8") as f:
                holdout = json.load(f)
            self.large_ambiguous_confidence_threshold = float(
                holdout["scopes"][context_name]["large_ambiguous_threshold"])
            self.escalation_margin = float(holdout["escalation_margin_large_only"])

    def predict(self, command: str) -> list[str]:
        return self.predict_with_details(command)[0]

    def _masked(self, probabilities: np.ndarray) -> np.ndarray:
        inside = probabilities[self._mask_index]
        return inside / inside.sum()

    def predict_with_details(self, command: str) -> tuple[list[str], dict]:
        """``predict`` plus the numbers behind the decision.

        The keys `CommandRouter.predict_with_details` returns, and ``p_in``, ``p_out``,
        ``escalation_margin`` and ``decision``. ``confidence`` is the probability of
        ``top_label``: renormalised inside the mask for a command, unmasked for
        ``wildcard`` and ``parameter_value``. ``topk_labels`` are always the scope's own
        best candidates with their masked scores.
        """
        if not isinstance(command, str):
            raise ValueError("Input must be a string")
        if not command.strip():
            raise ValueError("Input text cannot be empty")

        tier = "large"
        probabilities = None
        if not self.large_only:
            tiny = self.modelpipeline.probabilities("tiny", command, self.context_name)
            if float(self._masked(tiny).max()) >= self.confidence_threshold:
                tier, probabilities = "tiny", tiny
        if probabilities is None:
            probabilities = self.modelpipeline.probabilities("large", command, self.context_name)

        masked = self._masked(probabilities)
        order = np.argsort(-masked, kind="stable")[:TOP_K]
        topk_labels = [self._mask_labels[position] for position in order]
        topk_scores = [float(masked[position]) for position in order]
        ambiguous_threshold = (
            self.tiny_ambiguous_confidence_threshold if tier == "tiny"
            else self.large_ambiguous_confidence_threshold
        )
        p_in = float(probabilities[self._mask_index].max())
        p_out = float(probabilities[self._outside].max()) if self._outside.any() else 0.0
        best = self._labels[int(probabilities.argmax())]

        if best == PARAMETER_VALUE_LABEL:
            decision, labels = DECISION_BARE_VALUE, [PARAMETER_VALUE_LABEL]
            confidence = float(probabilities.max())
        elif best == WILDCARD_LABEL:
            decision, labels = DECISION_NOT_MINE, [WILDCARD_LABEL]
            confidence = float(probabilities.max())
        elif p_out > p_in + self.escalation_margin:
            decision, labels, confidence = DECISION_NOT_MINE, [WILDCARD_LABEL], p_out
        elif topk_scores[0] > ambiguous_threshold:
            decision, labels, confidence = DECISION_MATCH, topk_labels[:1], topk_scores[0]
        else:
            decision, labels, confidence = DECISION_AMBIGUOUS, list(topk_labels), topk_scores[0]

        details = {
            "model_tier": tier,
            "confidence": confidence,
            "ambiguous_threshold": float(ambiguous_threshold),
            "confident": decision != DECISION_AMBIGUOUS,
            "top_label": labels[0],
            "topk_labels": topk_labels,
            "topk_scores": topk_scores,
            "p_in": p_in,
            "p_out": p_out,
            "escalation_margin": self.escalation_margin,
            "decision": decision,
        }
        return list(labels), details


# ---------------------------------------------------------------------------
# The trainer's side: the ``workflow`` artifact layout
# ---------------------------------------------------------------------------

#: Receives the rows `LayoutTrainer` assembled, when a caller asked to see them
#: (`observe_assembly`). None at every other time.
_ASSEMBLY_OBSERVER: Optional[list] = None


@contextlib.contextmanager
def observe_assembly() -> Iterator[list]:
    """Collect the `WorkflowRows` every `LayoutTrainer` assembles inside the block.

    For the check that the trainer's in-process rows are the rows `workflow_rows` rebuilds
    from the caches: run `model_pipeline_training.train(..., stop_before_fit=True,
    intent_model_layout="workflow")` inside the block and compare.
    """
    global _ASSEMBLY_OBSERVER
    previous, collected = _ASSEMBLY_OBSERVER, []
    _ASSEMBLY_OBSERVER = collected
    try:
        yield collected
    finally:
        _ASSEMBLY_OBSERVER = previous


def row_differences(left: WorkflowRows, right: WorkflowRows) -> list[str]:
    """How two row sets of one workflow differ, in plain words; empty when they are the same.

    Compares the single model's rows as a multiset (a repeated text is two rows) and every
    scope's mask, trained rows and held-out personas.
    """
    differences: list[str] = []
    if Counter(left.rows) != Counter(right.rows):
        only_left = Counter(left.rows) - Counter(right.rows)
        only_right = Counter(right.rows) - Counter(left.rows)
        differences.append(
            f"model rows differ: {sum(only_left.values())} only on the left, "
            f"{sum(only_right.values())} only on the right"
        )
    for name in sorted(set(left.scopes) | set(right.scopes)):
        if name not in left.scopes or name not in right.scopes:
            differences.append(f"scope {name} is on one side only")
            continue
        a, b = left.scopes[name], right.scopes[name]
        if a.mask != b.mask:
            differences.append(f"scope {name}: masks differ")
        if {c: Counter(r) for c, r in a.trained_rows.items()} != {
                c: Counter(r) for c, r in b.trained_rows.items()}:
            differences.append(f"scope {name}: trained rows differ")
        if sorted(a.heldout_personas) != sorted(b.heldout_personas):
            differences.append(f"scope {name}: held-out personas differ")
    return differences


class LayoutTrainer:
    """Trains a workflow into the ``workflow`` artifact layout from inside `train`.

    `model_pipeline_training.train` runs its usual per-context loop (label set, utterances
    from the cache or the generator, provenance) and hands each context's own rows to
    `add_context` instead of fitting a pair of models for it. `finish` then fits the one large
    model over the union, calibrates every scope, writes ``workflow/`` and each context's two
    small files, and returns the per-context held-out reports. The recipe is arm S of the
    second pre-registration (``docs/single_intent_model_preregistration_v2.md``).

    The per-context "not here" seed rows are not built or used: a scope declines a request
    because the model's best label lies outside its mask.

    `train.embedding_head.EmbeddingLayoutTrainer` is the same row assembly with a frozen encoder
    and a linear head in place of the fine-tuned model: it overrides ``layout`` and `finish`.
    """

    #: The artifact layout this trainer writes.
    layout: ClassVar[str] = intent_model_layout.WORKFLOW

    def __init__(self, workflow_folderpath: str, crd, *, seed: int,
                 benchmark_cases: Sequence[heldout_evaluation.BenchmarkCase] = ()):
        self.workflow_folderpath = workflow_folderpath
        self.crd = crd
        self.seed = seed
        self.benchmark_cases = list(benchmark_cases)
        self.scopes: dict[str, ScopeRows] = {}
        self._splits: dict[str, Optional[heldout_evaluation.PersonaSplit]] = {}

    def add_context(self, context_name: str, by_command: Mapping[str, Sequence[str]],
                    trained_labels: set[str]) -> None:
        """Take one context's utterances per command, as the per-context loop assembled them."""
        if context_name == intent_model_layout.WORKFLOW_MODEL_DIRNAME:
            raise heads.TrainingDataError(
                f"A context named {context_name!r} cannot be trained in the "
                f"'{self.layout}' layout: its artifact folder is where the "
                f"workflow's one model is written. Rename the context or train with "
                f"{intent_model_layout.SETTING}={intent_model_layout.PER_CONTEXT}."
            )
        recorder = determinism.get_provenance_recorder()
        scope, split = build_scope_rows(context_name, by_command, self.crd, recorder)
        if scope is None:
            return
        self.scopes[context_name] = scope
        self._splits[context_name] = split

        own_rows = sum(len(rows) for rows in scope.cached_rows.values())
        context_texts = {text for rows in scope.cached_rows.values() for text in rows}
        if recorder is not None:
            # Said per context because the training report lists every context's reserved
            # classes: there is no per-context "not here" class to count in this layout.
            recorder.record_context(
                context_name=context_name,
                command_name=WILDCARD_LABEL,
                status=determinism.ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
                row_count=0,
                reason=(
                    "no per-context 'not here' rows: the workflow's one model declines a "
                    "request when its best label lies outside this context's mask, and its "
                    "one wildcard class is the built-in vague replies"
                ),
                own_row_count=own_rows,
                raw_candidate_count=0,
                deduplicated_candidate_count=0,
                always_include_count=0,
                selected_budget=None,
                coverage_floor=0,
                coverage_floor_applied=False,
            )
        heads._record_parameter_value_context_training(
            context_name, sorted(set(PARAMETER_VALUE_PLACEHOLDERS) - context_texts), own_rows)
        if split is not None:
            heads._export_holdout(context_name, set(trained_labels), split)

    def rows(self) -> WorkflowRows:
        """The single model's rows from the contexts added so far."""
        rows, findings = assemble_rows(self.scopes)
        assembled = WorkflowRows(
            workflow_folderpath=self.workflow_folderpath, scopes=dict(self.scopes),
            rows=rows, findings=findings)
        if _ASSEMBLY_OBSERVER is not None:
            _ASSEMBLY_OBSERVER.append(assembled)
        return assembled

    def model_dir(self) -> str:
        """Where the one model is written: ``workflow/`` beside the context folders, inside the
        artifact version being assembled (or ``___command_info`` when there is none)."""
        any_context = sorted(self.scopes)[0]
        context_dir = os.path.dirname(
            heads.get_artifact_path(self.workflow_folderpath, any_context, "threshold.json"))
        return os.path.join(os.path.dirname(context_dir), intent_model_layout.WORKFLOW_MODEL_DIRNAME)

    def finish(self) -> list[heldout_evaluation.HeldoutReport]:
        """Fit, calibrate, write the layout, and score each context through the published reader."""
        if not self.scopes:
            return []
        rows = self.rows()
        for finding in rows.findings:
            logger.info(f"workflow model rows: {finding}")
        print(
            f"\n=== Training one model for workflow: "
            f"{os.path.basename(os.path.normpath(self.workflow_folderpath))} "
            f"({len(rows.labels)} labels, {len(rows.rows)} rows, {len(self.scopes)} contexts) ===\n"
        )
        fitted = fit_workflow_model(rows, self.model_dir(), seed=self.seed, large_only=True)
        trained_texts = {text for text, _label in fitted.train_rows}

        context_dirs: dict[str, str] = {}
        for context_name in sorted(self.scopes):
            context_dir = os.path.dirname(
                heads.get_artifact_path(self.workflow_folderpath, context_name, "threshold.json"))
            intent_model_layout.write_context_files(context_dir, context_name, self.layout)
            context_dirs[context_name] = context_dir

        reports = [
            self._report(context_name, context_dirs[context_name], fitted, trained_texts)
            for context_name in sorted(self.scopes)
        ]
        margin = fitted.payload["escalation_margin_large_only"]
        print(f"Workflow model written to {self.model_dir()} (escalation margin {margin:.4f})")
        return reports

    def _report(self, context_name: str, context_dir: str, fitted: FittedWorkflowModel,
                trained_texts: set[str]) -> heldout_evaluation.HeldoutReport:
        scope_numbers = fitted.payload["scopes"][context_name]
        large_stats = scope_numbers["how"]["large_stats"]
        threshold = scope_numbers["large_ambiguous_threshold"]
        split = self._splits.get(context_name)
        notes = list(split.notes) if split is not None else []
        heldout = list(split.heldout) if split is not None else []
        # A command shared by two contexts is one label with the union of their rows, so a row
        # this context held out can have been trained through the other. Scoring it would be
        # scoring a training row.
        unseen = [record for record in heldout if record.utterance not in trained_texts]
        if len(unseen) != len(heldout):
            notes.append(
                f"{len(heldout) - len(unseen)} of {len(heldout)} held-out rows were left out "
                f"of the score: the workflow's one model trained on them through another "
                f"context that shares the command."
            )
        report = heldout_evaluation.HeldoutReport(
            context=context_name,
            in_distribution_f1=scope_numbers["in_distribution_f1_large_only"],
            seed=self.seed,
            heldout_personas=list(self.scopes[context_name].heldout_personas),
            notes=notes,
            large_ambiguity={
                "failed_rows": heads.large_failed_row_count(large_stats),
                "rule": scope_numbers["how"]["large_ambiguous_rule"],
                "threshold": threshold,
                # The model has a wildcard class of its own, workflow-wide.
                "has_wildcard_class": True,
                "at_floor_without_wildcard": heads.at_floor_without_wildcard(threshold, True),
                "wildcard_seeded_from_built_in_vague_replies": True,
                "built_in_vague_weight": heads.BUILT_IN_VAGUE_WEIGHT,
                "intent_model_layout": intent_model_layout.WORKFLOW,
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
        except Exception as exc:  # noqa: BLE001 - as in `train`: the model is written and usable
            report.notes.append(f"held-out evaluation failed: {exc}")
            logger.error(f"Held-out evaluation failed for context '{context_name}': {exc}")
        return report
