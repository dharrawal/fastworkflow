import contextlib
import hashlib
import os
from typing import Callable, Iterable, Mapping, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import torch 
# from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import json
import os
from torch.utils.data import random_split
import fastworkflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import pickle
import threading
from pathlib import Path
from collections import Counter

from fastworkflow.core.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.train import heldout_evaluation
from fastworkflow.train.determinism import (
    ContextTrainingStatus,
    get_provenance_recorder,
)
from fastworkflow.train.artifact_versioning import PLANNING_HEADS_PRODUCTION, PLANNING_HEADS_RECORD_KEY
from fastworkflow.train.selective_training import contexts_for_training
from fastworkflow.utils.logging import logger
from fastworkflow.core.command_directory import is_visible_planning
from fastworkflow.intent.nlu_labels import (
    PARAMETER_VALUE_LABEL,
    PARAMETER_VALUE_PLACEHOLDERS,
    WILDCARD_LABEL,
    label_of,
)
from fastworkflow.intent.router import (
    CommandRouter,
    GLOBAL_CONTEXT_FOLDER,
    ModelPipeline,
    calculate_ndcg_at_k,
    device,
    predict_single_sentence,
)


# transformers 5.x emits a tqdm "Loading weights" progress bar on every
# from_pretrained() call. fastWorkflow loads one intent-detection model per
# context, so these bars are noise when running/training a workflow. Suppress them.
try:
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
except Exception:  # noqa: BLE001 - older transformers may not expose this
    pass

dataset=None
label_encoder=LabelEncoder()


class TrainingDataError(ValueError):
    """Raised before model fitting when a label cannot be trained and evaluated."""


def _name_labels(
    labels: Iterable[int],
    decode_label: Optional[Callable[[int], str]] = None,
) -> str:
    """Render labels as a plain comma-separated list, sorted for a stable message.

    Deliberately not a list repr. `decode_label` returns a numpy string, so a repr
    renders as `[np.str_('cmd')]`, and the report prints through a rich console that
    reads a bracketed run with no spaces as console markup and deletes it - which
    silently dropped the one detail these messages exist to carry.
    """
    names = sorted(
        str(decode_label(label)) if decode_label else str(label) for label in labels
    )
    return ", ".join(names)


def split_training_data(
    dataset: list[tuple[str, int]],
    decode_label: Optional[Callable[[int], str]] = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a deterministic, class-aware train/evaluation split.

    Every label that *can* appear on both sides does. A label with fewer rows than
    `heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL` cannot be stratified, so it trains
    and goes unmeasured rather than failing the run: that is already what the persona
    holdout does when its own split would starve a label, and aborting here instead
    spends the entire LLM and GPU budget only to publish nothing - over a label whose
    evaluation coverage the developer may never have wanted. Being unmeasured is
    reported, not fatal.

    Splitting still replaces the unstratified 25% split that could randomly put every
    row for a small class in the evaluation set, leaving the classifier unable to learn
    that command.

    ``decode_label`` turns the encoded label back into its command name for the
    warning and error messages. Without it those messages name labels the developer
    cannot map to commands without reverse-engineering the LabelEncoder, which is the
    state this argument exists to prevent.
    """
    label_counts = Counter(label for _, label in dataset)
    unmeasurable = {
        label
        for label, count in label_counts.items()
        if count < heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL
    }
    # Partition rather than filter, so the unmeasurable rows still reach the model. They
    # are appended in dataset order to keep the split reproducible.
    splittable = [row for row in dataset if row[1] not in unmeasurable]
    unmeasured_rows = [row for row in dataset if row[1] in unmeasurable]
    if unmeasurable:
        logger.warning(
            f"{len(unmeasurable)} intent label(s) have fewer than "
            f"{heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL} rows, so they train "
            f"without an evaluation row and are unmeasured on the routing axis: "
            f"{_name_labels(unmeasurable, decode_label)}"
        )

    label_counts = Counter(label for _, label in splittable)
    label_count = len(label_counts)
    if not label_count:
        # Nothing can be measured, so there is no evaluation set to tune a threshold
        # against and no score to publish. That is a different failure from one thin
        # label and stays fatal.
        raise TrainingDataError(
            f"No intent label has the "
            f"{heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL} rows needed to form an "
            f"evaluation set; every label is below the floor: "
            f"{_name_labels(unmeasurable, decode_label)}"
        )

    test_size = max((len(splittable) + 3) // 4, label_count)
    if len(splittable) - test_size < label_count:
        raise TrainingDataError(
            f"Cannot split {len(splittable)} rows across {label_count} labels while "
            "keeping at least one row per label in both training and evaluation sets."
        )

    train_data, test_data = train_test_split(
        splittable,
        test_size=test_size,
        random_state=42,
        stratify=[label for _, label in splittable],
    )
    return list(train_data) + unmeasured_rows, list(test_data)


def save_label_encoder(filepath):
    global label_encoder
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(filepath):
    """Rebind the trainer's module-level encoder. Not for inference - use
    ``get_label_encoder``, which hands back the encoder instead of sharing it."""
    global label_encoder
    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)


def find_optimal_confidence_threshold(model, test_loader, device, min_threshold=0.5129, max_top3_usage=0.3, step_size=0.01, k_val=3):
    """
    Find optimal confidence threshold above the escalation threshold while limiting top@3 usage.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: torch device
        min_threshold: Minimum threshold (escalation threshold)
        max_top3_usage: Maximum allowed top@3 usage (default 0.3 or 30%)
        step_size: Step size for threshold search
    """
    # Get confidence statistics
    stats, confidences, predictions, labels, failed_cases = analyze_model_confidence(
        model, test_loader, device
    )
    
    # Set search range starting from escalation threshold
    start_threshold = min_threshold
    end_threshold = min(stats['successful']['max'], 0.95)
    
    best_metrics = None
    optimal_threshold = None
    best_score = 0
    
    # Store results for all thresholds
    thresholds = []
    f1_scores = []
    top3_usages = []
    combined_scores = []
    
    def calculate_score(f1, top3_usage):
        """
        Scoring function that:
        1. Prioritizes F1 score
        2. Heavily penalizes exceeding max_top3_usage
        """
        if top3_usage > max_top3_usage:
            return f1 * (1 - 2 * (top3_usage - max_top3_usage))  # Strong penalty for exceeding limit
        return f1
    
    # Test different correct thresholds 
    model.eval()
    with torch.no_grad():
        for threshold in tqdm(np.arange(start_threshold, end_threshold, step_size), 
                            desc="Finding optimal threshold"):
            true_labels = []
            predicted_labels = []
            top3_count = 0
            total = 0
            correct_top1 = 0
            correct_top3 = 0
            
            for encodings, labels, _ in test_loader:
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                top_probs, top_preds = torch.topk(probs, k=k_val, dim=1) #TODO remove the hardcode k value set it based on len of y
                max_confidences = top_probs[:, 0]
                
                batch_size = labels.size(0)
                total += batch_size
                
                for i in range(batch_size):
                    if max_confidences[i] >= threshold:
                        # Use top-1 prediction
                        pred = top_preds[i, 0]
                        if pred == labels[i]:
                            correct_top1 += 1
                    else:
                        # Use top-3 predictions
                        top3_count += 1
                        if labels[i] in top_preds[i]:
                            pred = labels[i]
                            correct_top3 += 1
                        else:
                            pred = top_preds[i, 0]
                    
                    true_labels.append(labels[i].cpu().item())
                    predicted_labels.append(pred.cpu().item())
            
            # Calculate metrics
            f1 = f1_score(true_labels, predicted_labels, average='weighted')
            top3_usage = top3_count / total
            top1_accuracy = correct_top1 / (total - top3_count) if (total - top3_count) > 0 else 0
            top3_accuracy = correct_top3 / top3_count if top3_count > 0 else 0
            
            # Calculate combined score
            combined_score = calculate_score(f1, top3_usage)
            
            thresholds.append(threshold)
            f1_scores.append(f1)
            top3_usages.append(top3_usage)
            combined_scores.append(combined_score)
            
            # Update best threshold if current score is better and top3 usage is within limit
            if combined_score > best_score and top3_usage <= max_top3_usage:
                best_score = combined_score
                optimal_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_score': f1,
                    'top3_usage': top3_usage,
                    'top1_accuracy': top1_accuracy,
                    'top3_accuracy': top3_accuracy,
                    'combined_score': combined_score
                }
    return optimal_threshold, best_metrics


def analyze_model_confidence(model, test_loader, device, model_name=""):
    model.eval()
    failed_cases = []
    failed_confidences = []
    successful_confidences = []
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for encodings, labels, _ in tqdm(test_loader, desc=f"Analyzing {model_name} confidence"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidence = torch.max(probs, dim=1).values

            # Store all results
            all_confidences.extend(confidence.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Analyze correct and incorrect predictions
            correct_mask = (predictions == labels)
            incorrect_mask = ~correct_mask

            for idx in torch.where(incorrect_mask)[0]:
                failed_confidences.append(confidence[idx].item())
                failed_cases.append({
                    'true_label': labels[idx].item(),
                    'predicted_label': predictions[idx].item(),
                    'confidence': confidence[idx].item()
                })

            successful_confidences.extend(confidence[correct_mask].cpu().numpy())

    stats = {
        'failed': {
            'min': np.min(failed_confidences) if failed_confidences else None,
            'max': np.max(failed_confidences) if failed_confidences else None,
            'mean': np.mean(failed_confidences) if failed_confidences else None,
            'median': np.median(failed_confidences) if failed_confidences else None,
            'count': len(failed_confidences),
        },
        'successful': {
            'min': np.min(successful_confidences) if successful_confidences else None,
            'max': np.max(successful_confidences) if successful_confidences else None,
            'mean': np.mean(successful_confidences) if successful_confidences else None,
            'median': np.median(successful_confidences) if successful_confidences else None,
            'count': len(successful_confidences),
            # The low quantile of correct-row confidences: what the large tier's
            # ambiguity threshold is built from when there are too few failures.
            'low_quantile': (
                float(np.quantile(successful_confidences, CORRECT_CONFIDENCE_LOW_QUANTILE))
                if successful_confidences else None
            ),
        }
    }
    return stats, all_confidences, all_predictions, all_labels, failed_cases

# ---------------------------------------------------------------------------
# R3: tier threshold vs ambiguity threshold must not collapse.
#
# `ModelPipeline.predict` escalates to the large model when
# `confidence < confidence_threshold` (the TIER threshold), and
# `CommandRouter.predict_with_details` then reports a single confident label when
# `confidence > ambiguous_threshold`. If the tiny tier's ambiguity threshold is at
# or below its tier threshold, every prediction the tiny tier keeps is by
# construction above the ambiguity threshold, so the tier can never report an
# ambiguity: it either escalates or resolves outright, and a wrong resolution is
# silent. Measured on a published router 20260905T132341Z-a0605e this held in
# ALL 18 trained contexts (7 of them with the two files byte-identical), because
# `find_optimal_threshold` sweeps upward from `tiny_stats['failed']['mean']` while
# the writer used that same failed-mean as the ambiguity threshold.
#
# The fix gives the tiny tier a real ambiguity band by placing its ambiguity
# threshold strictly above the tier threshold, and refuses single-label resolution
# below an absolute floor at either tier.
#
# MARGIN — the preferred separation is the midpoint of the interval the tier sweep
# itself ran over, `(failed_mean + successful_mean) / 2`: the sweep's own endpoints
# are the model's measured "typically wrong" and "typically right" confidences, so
# the midpoint is the point the training data says stops being evidence for the top
# label and is not a constant invented from outside. `TIER_AMBIGUITY_MIN_SEPARATION`
# is the fallback when the sweep picked a tier threshold at or above that midpoint;
# 0.05 is far above float noise and narrow enough not to undo a deliberately high
# tier threshold.
#
# CEILING — `MAX_AMBIGUITY_THRESHOLD` exists so a context can still resolve
# a single label: an ambiguity threshold at 1.0 would make `confidence >
# ambiguous_threshold` unsatisfiable. It is a *resolvability* cap, not a second
# invariant, so it must never be allowed to pull the ambiguity threshold down to or
# below the tier threshold. `find_optimal_threshold` sweeps up to the tiny tier's
# measured successful mean, so a confidently separated context legitimately picks a
# tier threshold at or above 0.99; clamping to a flat 0.99 there collapsed the pair
# and aborted the whole workflow's training from inside `train()`'s per-context loop.
# Above that point the cap becomes the midpoint of the headroom that is actually
# left, `(tier + 1) / 2`: still strictly above the tier threshold, still strictly
# below certainty, and it halves the resolvable band rather than deleting it. The
# tier threshold itself is never rewritten to fit the cap, because that number is
# consumed as the escalation point (`ModelPipeline.predict_batch`: `need_distil =
# tiny_confidence < self.confidence_threshold`) and was chosen by the sweep to
# balance F1, NDCG and DistilBERT usage; the ambiguity threshold is consumed only by
# `CommandRouter.predict_with_details` to choose one label over a candidate list.
# Moving the cap changes how loudly a very confident tier reports ambiguity; moving
# the tier would silently re-route traffic between the two models.
#
# CERTAIN TIER — a tier threshold of exactly 1.0 is reachable too: float32 softmax
# rounds the top probability to 1.0 once the logit gap passes about 17, so a context
# whose correct held-out rows all saturate has a successful mean of exactly 1.0, and
# that is the top point of the sweep. At that tier the tiny model keeps only
# predictions whose confidence is exactly 1.0, where every other label has zero mass.
# No ambiguity threshold can sit above 1.0 and still be satisfiable, and none is
# needed, so the separation invariant is waived there and the flat cap is written.
#
# FLOOR — `SINGLE_LABEL_RESOLUTION_FLOOR = 0.5` is the point at which the model
# stops putting more posterior mass on the winning label than on everything else
# combined. Below it a "confident" single label is not supported by the model's own
# distribution, so the runtime must show candidates instead. It is applied to the
# large tier as well, because the rule is about single-label resolution, not about
# which tier produced it. The cost is deliberate: more ambiguity turns, traded for
# the silent misroutes this band makes loud. DistilBERT usage does not change,
# because escalation reads only the tier threshold, which none of this moves.
#
# LARGE TIER SOURCE — the large tier's ambiguity threshold is the mean confidence of
# the large model's MISCLASSIFIED test rows, floored and capped as above. That mean is
# a statistic of the model's errors, so it needs errors to exist. With none, or with
# a handful, it is undefined or noise, and the old code wrote the floor silently: a
# model that fit the in-distribution test split BETTER got a LOOSER threshold (a head
# retrained on a regenerated corpus went from 0.738 to exactly 0.5 this way). Below
# `LARGE_FAILED_ROWS_MIN` misclassified rows the threshold is therefore taken from the
# other side of the split: the `CORRECT_CONFIDENCE_LOW_QUANTILE` quantile of the
# correct rows' confidences, i.e. a confidence below which correct predictions are
# rare, so anything under it is shown as candidates. With `LARGE_FAILED_ROWS_MIN` or
# more misclassified rows the value is the failed mean exactly as before, so a head
# that calibrates normally does not move. Both paths are clamped to [floor, cap]; the
# rule that produced the value is named in `LARGE_RULE_*` and recorded in the
# per-context held-out report. The tiny tier is untouched.
# ---------------------------------------------------------------------------
TIER_AMBIGUITY_MIN_SEPARATION = 0.05
LARGE_FAILED_ROWS_MIN = 5
CORRECT_CONFIDENCE_LOW_QUANTILE = 0.10
LARGE_RULE_FAILED_MEAN = "failed_mean"
LARGE_RULE_CORRECT_QUANTILE = "correct_quantile"
LARGE_RULE_FLOOR_NO_STATISTICS = "floor_no_statistics"
SINGLE_LABEL_RESOLUTION_FLOOR = 0.5
MAX_AMBIGUITY_THRESHOLD = 0.99
CERTAIN_TIER_THRESHOLD = 1.0


def keeps_only_certain_predictions(tier_threshold) -> bool:
    """Whether *tier_threshold* leaves the tiny tier only confidence-1.0 predictions."""
    return tier_threshold is not None and float(tier_threshold) == CERTAIN_TIER_THRESHOLD


def resolvable_ambiguity_ceiling(tier_threshold) -> float:
    """The highest ambiguity threshold that still leaves a context able to resolve.

    `MAX_AMBIGUITY_THRESHOLD` normally, which is what every tier threshold below it
    has always been clamped to. For a tier threshold at or above that constant — which
    the sweep can legitimately pick for a well separated context — the flat cap would
    sit at or under the tier and collapse the pair, so the ceiling becomes the midpoint
    of the remaining headroom, `(tier + 1) / 2`. That is strictly above any tier below
    1.0 and strictly below 1.0, so both the separation invariant and resolvability
    survive. A tier of exactly 1.0 is what a float32-saturated sweep produces; it
    keeps the flat cap, because `(1 + 1) / 2` would be an unsatisfiable 1.0 and the
    only predictions that tier keeps are certain ones. A tier above 1.0 is not
    something any sweep over softmax probabilities can produce; the ceiling then falls
    at or below it and `write_ambiguity_thresholds` raises, which is the intended
    treatment of a genuinely collapsed pair.
    """
    if tier_threshold is None or keeps_only_certain_predictions(tier_threshold):
        return MAX_AMBIGUITY_THRESHOLD
    tier = float(tier_threshold)
    # `tier != tier` is the NaN test: a NaN tier keeps the flat cap it has always had
    # rather than propagating into the written threshold.
    if tier != tier or tier < 0 or tier < MAX_AMBIGUITY_THRESHOLD:
        return MAX_AMBIGUITY_THRESHOLD
    return (tier + 1.0) / 2.0


def separated_tiny_ambiguous_threshold(tier_threshold, tiny_stats) -> float:
    """The tiny tier's ambiguity threshold, guaranteed strictly above *tier_threshold*.

    Returns the largest of the sweep-interval midpoint, `tier_threshold +
    TIER_AMBIGUITY_MIN_SEPARATION` and `SINGLE_LABEL_RESOLUTION_FLOOR`, clamped below
    `resolvable_ambiguity_ceiling(tier_threshold)` so a context can still resolve
    something. A missing statistic (no failures, or no successes, in the held-out
    split) drops that term rather than the whole computation; the floor is always
    present, so the result is a usable threshold even when `find_optimal_threshold`
    returned its -1 sentinel.
    """
    failed_mean = tiny_stats['failed']['mean']
    successful_mean = tiny_stats['successful']['mean']

    candidates = [SINGLE_LABEL_RESOLUTION_FLOOR]
    if tier_threshold is not None and tier_threshold >= 0:
        candidates.append(float(tier_threshold) + TIER_AMBIGUITY_MIN_SEPARATION)
    if failed_mean is not None and successful_mean is not None:
        candidates.append((float(failed_mean) + float(successful_mean)) / 2.0)

    return min(max(candidates), resolvable_ambiguity_ceiling(tier_threshold))


def large_failed_row_count(large_stats) -> int:
    """Misclassified test rows behind *large_stats*.

    Stats written before `analyze_model_confidence` reported a count carry only the
    mean: a present mean is read as enough rows, an absent one as none, which is what
    those stats meant when the failed mean was the only input.
    """
    failed = large_stats['failed']
    count = failed.get('count')
    if count is not None:
        return int(count)
    return 0 if failed['mean'] is None else LARGE_FAILED_ROWS_MIN


def large_ambiguous_threshold_and_rule(large_stats) -> tuple:
    """The large tier's ambiguity threshold and the rule that produced it.

    See "LARGE TIER SOURCE" above the constants. Returns ``(threshold, rule)`` with
    the threshold always inside ``[SINGLE_LABEL_RESOLUTION_FLOOR,
    MAX_AMBIGUITY_THRESHOLD]``.
    """
    failed_mean = large_stats['failed']['mean']
    if failed_mean is not None and large_failed_row_count(large_stats) >= LARGE_FAILED_ROWS_MIN:
        measured, rule = float(failed_mean), LARGE_RULE_FAILED_MEAN
    else:
        quantile = large_stats['successful'].get('low_quantile')
        if quantile is None:
            measured, rule = 0.0, LARGE_RULE_FLOOR_NO_STATISTICS
        else:
            measured, rule = float(quantile), LARGE_RULE_CORRECT_QUANTILE
    return min(max(measured, SINGLE_LABEL_RESOLUTION_FLOOR), MAX_AMBIGUITY_THRESHOLD), rule


def floored_large_ambiguous_threshold(large_stats) -> float:
    """The large tier's ambiguity threshold, never below the single-label floor."""
    return large_ambiguous_threshold_and_rule(large_stats)[0]


def at_floor_without_wildcard(large_threshold, has_wildcard_class) -> bool:
    """True when a head known to have no wildcard class publishes its large threshold at the floor."""
    return has_wildcard_class is False and large_threshold == SINGLE_LABEL_RESOLUTION_FLOOR


def write_ambiguity_thresholds(
    context_artifacts_dir, tier_threshold, tiny_stats, large_stats,
    *, has_wildcard_class=None,
) -> tuple:
    """Write both ambiguity thresholds for one context and return ``(tiny, large)``.

    Takes the context's artifact directory rather than a workflow path so the
    invariant it establishes -- ``tiny_ambiguous > tier`` and both ambiguity
    thresholds at or above `SINGLE_LABEL_RESOLUTION_FLOOR` -- can be exercised over
    real files for every context without loading a model or a routing definition.
    Raises `ValueError` rather than publishing a collapsed pair: a collapsed pair
    makes "confident" structurally unfalsifiable at that tier, and because nothing
    downstream notices, it can sit unnoticed in a published artifact set. The guard
    is reserved for a pair no legitimate sweep could produce — a tier threshold
    above certainty, or a margin computation that stopped separating. A tier threshold the sweep
    can actually pick, including one at or above `MAX_AMBIGUITY_THRESHOLD`, is always
    published, because aborting `train()` for the whole workflow is a far worse answer
    to a well separated context than a narrow ambiguity band is. That includes a tier
    of exactly 1.0, which keeps only certain predictions and so needs no band.

    ``has_wildcard_class`` is None when the caller does not say (no check) and a bool
    when it does. A head with NO wildcard / "none of these" class (False) whose large
    threshold sits exactly at the floor gets one WARNING naming the context and what to
    do (see `at_floor_without_wildcard`): the threshold is then the only thing standing
    between a vague request and a confident single label, and a value pinned at the
    floor means calibration found nothing. The thresholds are still written, because
    raising would abort training for every later context of the workflow. `train()`
    passes it and records the same boolean in the held-out report. The warning is meant
    to become an error once a retrain shows the built-in heads do not trip it
    (bd fix-1hs8).
    """
    tiny_ambiguous_threshold = separated_tiny_ambiguous_threshold(tier_threshold, tiny_stats)
    large_ambiguous_threshold, large_rule = large_ambiguous_threshold_and_rule(large_stats)
    if large_rule == LARGE_RULE_CORRECT_QUANTILE:
        logger.info(
            f"Large ambiguity threshold for {context_artifacts_dir} comes from the "
            f"{CORRECT_CONFIDENCE_LOW_QUANTILE:.0%} quantile of correct-row confidences "
            f"({large_ambiguous_threshold:.4f}): only {large_failed_row_count(large_stats)} "
            f"misclassified test rows (< {LARGE_FAILED_ROWS_MIN}) to calibrate on."
        )

    if (
        tier_threshold is not None
        and not keeps_only_certain_predictions(tier_threshold)
        and tiny_ambiguous_threshold <= tier_threshold
    ):
        raise ValueError(
            f"tiny ambiguity threshold {tiny_ambiguous_threshold} does not sit above "
            f"tier threshold {tier_threshold}: the tiny tier could never report an "
            f"ambiguity in {context_artifacts_dir}"
        )
    if min(tiny_ambiguous_threshold, large_ambiguous_threshold) < SINGLE_LABEL_RESOLUTION_FLOOR:
        raise ValueError(
            f"ambiguity threshold below the single-label floor "
            f"{SINGLE_LABEL_RESOLUTION_FLOOR} in {context_artifacts_dir}"
        )

    if at_floor_without_wildcard(large_ambiguous_threshold, has_wildcard_class):
        logger.warning(
            f"large ambiguity threshold sits exactly at the single-label floor "
            f"{SINGLE_LABEL_RESOLUTION_FLOOR} (rule: {large_rule}) in "
            f"{context_artifacts_dir}, and this head has no wildcard / 'none of these' "
            f"class to catch vague requests. Add training utterances for this context "
            f"or give it a wildcard class, then retrain."
        )

    os.makedirs(context_artifacts_dir, exist_ok=True)
    for filename, value in (
        ("tiny_ambiguous_threshold.json", tiny_ambiguous_threshold),
        ("large_ambiguous_threshold.json", large_ambiguous_threshold),
    ):
        with open(os.path.join(context_artifacts_dir, filename), 'w') as f:
            json.dump({'confidence_threshold': value}, f)

    return tiny_ambiguous_threshold, large_ambiguous_threshold


def find_optimal_threshold(tiny_stats, test_loader, pipeline):
    # Generate threshold range based on confidence statistics
    min_threshold = tiny_stats['failed']['mean']
    max_threshold = tiny_stats['successful']['mean']
    if min_threshold and max_threshold:
        thresholds = np.linspace(min_threshold, max_threshold, 20)
    else:
        return (
            {            
                'threshold': -1,
                'f1': -1,
                'ndcg': -1,
                'distil_usage': -1
            }, [{            
                'threshold': -1,
                'f1': -1,
                'ndcg': -1,
                'distil_usage': -1
            }]
        )

    results = []
    for threshold in tqdm(thresholds, desc="Finding optimal threshold"):
        pipeline.confidence_threshold = threshold
        f1, ndcg, stats = pipeline.evaluate(test_loader)
        results.append({
            'threshold': threshold,
            'f1': f1,
            'ndcg': ndcg,
            'distil_usage': stats['distil_percentage']
        })
    
    # Find threshold with best balance of performance and efficiency
    alpha = 0.15
    best_result = max(results, key=lambda x: x['f1'] * x['ndcg'] * (1 - alpha * (x['distil_usage'] / 100)))

    return best_result, results


# Workflow folderpath (resolved) -> artifact version currently being written.
# Installed by the trainer for the duration of a run; empty at every other time.
_active_artifact_version: dict[str, str] = {}


def set_active_artifact_version(workflow_folderpath: str, version_id: Optional[str]) -> None:
    """Route `get_artifact_path` writes for *workflow_folderpath* into *version_id*.

    Pass ``None`` to clear. The trainer installs this for the duration of a run so a
    retrain assembles a NEW version instead of overwriting the live one in place, which
    would leave a failed run with neither the old artifacts nor a complete new set.
    Keeping it here rather than threading a parameter through means the six
    existing `get_artifact_path` call sites need no changes.
    """
    key = str(Path(workflow_folderpath).resolve())
    if version_id is None:
        _active_artifact_version.pop(key, None)
    else:
        _active_artifact_version[key] = version_id


def get_artifact_path(workflow_folderpath: str, context_name: str, filename: str) -> str:
    """Return the absolute path for a model/artefact for *context_name*.

    Under versioning the file lives at:
        <workflow>/___command_info/versions/<version>/<context_name>/<filename>
    which every legacy reader still reaches through the per-context compatibility entry
    at ``<workflow>/___command_info/<context_name>/``.

    When no version is active and none has been published (a workflow that has never been
    trained under the versioned layout) this falls back to the historical unversioned
    path, so a partially migrated tree keeps working.

    The directory is created if it does not yet exist. The special context name "*" is
    mapped to GLOBAL_CONTEXT_FOLDER.
    """
    from fastworkflow import RoutingRegistry
    from fastworkflow.train import artifact_versioning as av

    ctx_folder = context_name if context_name != "*" else GLOBAL_CONTEXT_FOLDER
    crd = RoutingRegistry.get_definition(workflow_folderpath)
    # Preserve the ___command_info mkdir side effect that callers have always relied on.
    info_root = Path(crd.command_directory.get_commandinfo_folderpath(workflow_folderpath))

    key = str(Path(workflow_folderpath).resolve())
    version_id = _active_artifact_version.get(key) or av.resolve_current_version(
        workflow_folderpath
    )
    if version_id is None:
        base_dir = info_root / ctx_folder
        base_dir.mkdir(parents=True, exist_ok=True)
        return str(base_dir / filename)

    return str(
        av.context_artifact_dir(workflow_folderpath, version_id, context_name) / filename
    )


# ---------------------------------------------------------------------

# After training loop is complete
def save_model(model,tokenizer, save_path):
    # Save the model
    model.save_pretrained(save_path)
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def evaluate_model(model, data_loader, device, k_val):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_ndcg = 0
    num_batches = 0

    with torch.no_grad():
        for encodings, labels, _ in tqdm(data_loader, desc="Evaluating"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Calculate test loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            ndcg = calculate_ndcg_at_k(outputs.logits, labels, k=k_val) # TODO this should be min of 3
            total_ndcg += ndcg
            num_batches += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches  # Average test loss
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_ndcg = total_ndcg / num_batches

    return f1, avg_ndcg, avg_loss


def _requires_utterances(cmd_dir: object, cmd_name: str) -> bool:
    """Return True if *cmd_name* declares a `Signature` (with or without an `Input`) and therefore
    needs utterance-based intent classification.

    This is determined after lazy-hydrating the command so the metadata is
    accurate.  A command with *no* `Signature` (such as a startup command, whatever its
    name) runs only as an action and is excluded from model training, as routing
    excludes it (`command_directory.declares_signature`).
    """
    # Ensure metadata is fully populated before inspection
    cmd_dir.ensure_command_hydrated(cmd_name)
    metadata = cmd_dir.get_command_metadata(cmd_name)
    return bool(metadata.input_for_param_extraction_class)


def _get_utterances(workflow: fastworkflow.Workflow,
                    workflow_folderpath: str, 
                    cmd_dir: object, cmd: str) -> list[str]:
    """Safely retrieve utterances for *cmd* when required.

    If the command does not declare a `Signature`, we return an empty list
    (no utterances needed) **without** calling `get_utterance_metadata`.
    """
    if not _requires_utterances(cmd_dir, cmd):
        return []

    um = cmd_dir.get_utterance_metadata(cmd)
    if not um:
        raise KeyError(
            f"Could not find utterance metadata for command '{cmd}'. "
            "It might be missing from the _commands directory."
        )

    func = um.get_generated_utterances_func(workflow_folderpath)
    return func(workflow, cmd) if func else []


def _get_cached_command_utterances(
    workflow: fastworkflow.Workflow,
    workflow_folderpath: str,
    cmd_dir: object,
    cmd: str,
    command_cache: dict[str, list[str]],
) -> list[str]:
    """Generate once per fully-qualified command, then reuse across contexts.

    A generated command function receives `(workflow, command_name)` and no context.
    Context inheritance changes where its rows are used, not how its rows are generated.
    Keying this cache by context used to redraw the same command for every context and
    made its last generation overwrite all earlier provenance.
    """
    if cmd not in command_cache:
        command_cache[cmd] = _get_utterances(
            workflow, workflow_folderpath, cmd_dir, cmd
        )
    return command_cache[cmd]


def _record_context_training(
    context_name: str,
    command_name: str,
    cmd_dir: object,
    utterances: list[str],
) -> None:
    """Record context row use or the trainer's exact skip reason."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    if not _requires_utterances(cmd_dir, command_name):
        recorder.record_context(
            context_name=context_name,
            command_name=command_name,
            status=ContextTrainingStatus.SKIPPED_NO_INPUT,
            reason="command has no Signature.Input",
        )
        return

    if not utterances:
        recorder.record_context(
            context_name=context_name,
            command_name=command_name,
            status=ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
            reason="utterance generator returned no rows",
        )
        return

    generation = recorder.records.get(command_name)
    fell_back = bool(generation and generation.fell_back)
    recorder.record_context(
        context_name=context_name,
        command_name=command_name,
        status=(
            ContextTrainingStatus.INCLUDED_FALLBACK
            if fell_back
            else ContextTrainingStatus.INCLUDED
        ),
        row_count=len(utterances),
        reason=generation.fallback_reason if fell_back and generation else None,
    )


def _record_wildcard_context_training(
    context_name: str,
    escalation_rows: Optional[list[str]],
    *,
    own_row_count: int,
    raw_candidate_count: int,
    deduplicated_candidate_count: int,
    always_include_rows: list[str],
    selected_budget: Optional[int],
    coverage_floor: int,
    sampled_commands: Optional[int] = None,
) -> None:
    """Record escalation-class rows and every denominator used to select them."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    included = escalation_rows is not None
    recorder.record_context(
        context_name=context_name,
        command_name=WILDCARD_LABEL,
        status=(
            ContextTrainingStatus.INCLUDED
            if included
            else ContextTrainingStatus.SKIPPED_NO_UTTERANCES
        ),
        row_count=len(escalation_rows or []),
        reason=(
            "reserved escalation class"
            if included
            else (
                "no escalation class (this context has no foreign commands to supply 'not here' "
                "rows, or it is the internal CME, which has no wildcard class)"
            )
        ),
        own_row_count=own_row_count,
        raw_candidate_count=raw_candidate_count,
        deduplicated_candidate_count=deduplicated_candidate_count,
        always_include_count=len(always_include_rows),
        selected_budget=selected_budget,
        coverage_floor=coverage_floor,
        coverage_floor_applied=(
            coverage_floor > own_row_count if included else False
        ),
        sampled_commands=sampled_commands,
    )


def _record_parameter_value_context_training(
    context_name: str,
    parameter_value_rows: list[str],
    own_row_count: int,
) -> None:
    """Record bare-value reserved rows without wildcard-only budget fields."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    recorder.record_context(
        context_name=context_name,
        command_name=PARAMETER_VALUE_LABEL,
        status=(
            ContextTrainingStatus.INCLUDED
            if parameter_value_rows
            else ContextTrainingStatus.SKIPPED_NO_UTTERANCES
        ),
        row_count=len(parameter_value_rows),
        reason="reserved bare-value class",
        own_row_count=own_row_count,
        raw_candidate_count=len(PARAMETER_VALUE_PLACEHOLDERS),
        deduplicated_candidate_count=len(parameter_value_rows),
        always_include_count=0,
    )


def own_command_names(crd: RoutingDefinition, ctx_name: str) -> set[str]:
    """A context's OWN commands: its ``"/"`` list in the context model, not inherited ones.

    Read from the model's private table because the scoped layout (Phase 9 step 3) needs the
    own list and ``crd.contexts`` only holds the inherited union. Unknown contexts give ``set()``.
    """
    entry = crd.context_model._command_contexts.get(ctx_name) or {}
    return set(entry.get("/") or [])


def trained_command_labels(ctx_name: str, crd: RoutingDefinition) -> set[str]:
    """Every command label a context trains: its own, base-resolved, plus the core ones.

    `crd.contexts` is already `context_model.commands(ctx) | core_command_names`
    (`command_routing.build`); the union is repeated here because that is the invariant
    this function names and callers must not have to know which half supplies the core
    commands. Shared by `train()` and by the escalation tests so the two cannot disagree
    about the label space, which is the disagreement AR5 was.

    EXPERIMENT-ONLY (Phase 9 step 3): under the ``scoped`` head layout the base is the
    context's own list and action-only commands (no ``Signature``) are left out.
    Those never had rows, so the flattened label set is unchanged in practice.
    """
    cmd_dir = crd.command_directory
    if active_experiment_head_scope() == SCOPED:
        candidates = {
            name for name in own_command_names(crd, ctx_name)
            if not _is_action_only(cmd_dir, name)
        }
    else:
        candidates = set(crd.contexts[ctx_name])
    labels: set[str] = set()
    for name in candidates | set(cmd_dir.core_command_names):
        try:
            if cmd_dir.get_command_metadata(name).is_planning:
                continue
        except KeyError:
            pass
        labels.add(name)
    labels |= planning_labels_in_layout(ctx_name, crd)
    return labels


def cache_context_command_utterances(
    ctx_name: str,
    crd: RoutingDefinition,
    workflow: fastworkflow.Workflow,
    cache: dict,
    command_cache: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Populate ``cache[ctx_name]`` for every label this context trains, and return them.

    One loop over every label, using the cache when it already holds the command and
    GENERATING when it does not. The returned mapping is restricted to the labels asked
    for, so an entry an earlier visit left behind for some other command can never become
    a label this context trains.
    """
    workflow_folderpath = workflow.folderpath
    cmd_dir = crd.command_directory
    map_cmd_2_uttlist = cache.setdefault(ctx_name, {})

    # `label_of` splits off the context qualifier, so the reserved label is recognised
    # whether or not a command name is qualified.
    wanted = sorted(
        cmd
        for cmd in trained_command_labels(ctx_name, crd)
        if label_of(cmd) != WILDCARD_LABEL
    )
    for cmd_name in wanted:
        if cmd_name in map_cmd_2_uttlist:
            print(f"Getting cached utterances for context: {ctx_name}, command: {cmd_name}\n")
        else:
            print(f"Generating utterances for context: {ctx_name}, command: {cmd_name} ...\n")
            map_cmd_2_uttlist[cmd_name] = _get_cached_command_utterances(
                workflow,
                workflow_folderpath,
                cmd_dir,
                cmd_name,
                command_cache,
            )
    return {cmd_name: map_cmd_2_uttlist[cmd_name] for cmd_name in wanted}


#: Hand-written seeds per foreign command in the sparse "not here" class (Phase 9 step 1,
#: design 2.15 and 3.9). A trainer constant, deliberately not a setting or an env var.
#: Phase 9 step 4 made the sparse_1 construction (one seed) the shipped one; step 1 measured
#: 8 as ``sparse_8``.
NOT_HERE_SEEDS = 1

#: Built-in vague / non-committal replies (bd fix-1hs8). A head whose "not here" class would
#: otherwise have NO rows (a single-context workflow with no core commands has no foreign
#: command to sample) is seeded from this list, so it still has a class for "none of these" and
#: a vague reply is not routed confidently to the nearest real command. Domain-neutral on
#: purpose: hedges, questions back, a contentless improvement request, filler, deferrals,
#: off-topic and garbage. Used as written, one row each, like every other foreign seed (no
#: utterance generation). Heads that already have foreign seeds never read it.
BUILT_IN_VAGUE_REPLIES: tuple[str, ...] = (
    "I'm still undecided",
    "why are you asking me that?",
    "give it a bit more oomph",
    "well, you see, um",
    "ask me again later",
    "sorry, what was the question again?",
    "blah blah qwerty",
    "the weather is lovely today",
    "we'll see how it goes",
)

#: Times each built-in row is repeated in the training rows. Nine rows against 34-41 rows for
#: each real command are out-voted (measured, bd fix-1hs8: 5 of 23 held-out vague replies were
#: still acted on), so the class is repeated to about the size of one real command (9 x 4 = 36).
#: Part of the training signature, so changing it retrains the head. Applies only to a head
#: seeded from the built-in list; foreign seeds are never repeated.
BUILT_IN_VAGUE_WEIGHT = 4

#: The pseudo foreign-command name the built-in rows are recorded under (selection, the head's
#: not-here samples, its training signature), so the version check sees them as one more source.
BUILT_IN_VAGUE_SOURCE = "built_in/vague_reply"

#: The shipped construction: what every real training run builds when no experiment
#: override is in force. ``resolve_reserved_class_arm`` returns it for ``None``.
SHIPPED_ARM = "sparse_1"

#: arm -> (construction, ratio, seeds per foreign command). Every arm is the sparse
#: ``"not_here"`` class: a few seeds of every foreign command, with no ancestor sampling and
#: no fill. ``None`` for seeds means ``NOT_HERE_SEEDS``, so ``sparse_1`` (the shipped arm)
#: follows the trainer constant, while ``sparse_8`` is pinned at 8 whatever the constant
#: says. The ratio column is unused and kept as ``None`` for the shape of the record.
RESERVED_CLASS_ARMS: dict[str, tuple[str, Optional[float], Optional[int]]] = {
    SHIPPED_ARM: ("not_here", None, None),
    "sparse_8": ("not_here", None, 8),
}

#: Arms that existed for the Phase 9 step 1 measurement and no longer do (Phase 9 step 5).
#: The ancestor construction they trained is gone, so naming one is refused with this text.
#: The recorded step 1 and step 3 results stay as history; they are not reproducible.
RETIRED_ARMS: dict[str, str] = {
    "control": (
        "the ancestor construction was retired in Phase 9 step 5; the sparse 'not here' "
        "class is the only escalation class (use sparse_1, the shipped arm)"
    ),
    "ratio_0_5": (
        "the ancestor construction was retired in Phase 9 step 5; the sparse 'not here' "
        "class is the only escalation class (use sparse_1, the shipped arm)"
    ),
}

#: The internal command_metadata_extraction workflow. It has no wildcard class at all: none of
#: its contexts has a parent, and the sparse class is an APP construction (design 2.15, 3.9).
#: Set only by ``arm_for_workflow``; it is not selectable as an experiment arm.
INTERNAL_CME_ARM = "internal_cme"

#: The construction label of each arm, as recorded in the not-here construction record and
#: the manifests. ``INTERNAL_CME_ARM`` is ``"none"``: no wildcard class, by design.
_INTERNAL_CONSTRUCTIONS: dict[str, str] = {INTERNAL_CME_ARM: "none"}

#: EXPERIMENT-ONLY (Phase 9 step 1). Never a public setting and never read from the
#: environment. The Phase 9 measurement script sets it through
#: ``experiment_reserved_class_arm``; tests pass ``reserved_class_arm=`` to
#: ``select_escalation_rows`` instead. ``None`` is the shipped construction (``SHIPPED_ARM``),
#: which every real training run uses. ``resolve_reserved_class_arm`` is the only reader.
_EXPERIMENT_RESERVED_CLASS_ARM: Optional[str] = None


def active_experiment_arm() -> Optional[str]:
    """The experiment override for this process, or None for a shipped run.

    Used for provenance and the holdout export only. The selection itself is read by
    ``resolve_reserved_class_arm``.
    """
    return _EXPERIMENT_RESERVED_CLASS_ARM


#: EXPERIMENT-ONLY. While a dict, ``train()`` fills it with each head's persona-holdout
#: split (see ``_export_holdout``); ``train_workflow`` sets it only under an active arm and
#: writes it to ``___command_info/not_here_holdout.json``. ``None`` means nothing is exported.
_HOLDOUT_EXPORT: Optional[dict] = None

HOLDOUT_EXPORT_FILENAME = "not_here_holdout.json"


def _export_holdout(ctx_name: str, own_labels: set[str], split) -> None:
    """Record one head's persona holdout for the not-here scorer (no effect unless exporting)."""
    export = _HOLDOUT_EXPORT
    if export is None:
        return
    export[ctx_name] = {
        "own_labels": sorted(own_labels),
        "own_heldout": [
            {"utterance": record.utterance, "label": record.label, "persona": record.persona}
            for record in split.heldout
            if record.label in own_labels
            and record.label not in (WILDCARD_LABEL, PARAMETER_VALUE_LABEL)
        ],
        # the not-here rows this head was TRAINED on, after the persona split removed the held-out ones
        "not_here_training": sorted({
            record.utterance for record in split.train if record.label == WILDCARD_LABEL
        }),
        # every row this head trains on, after the split: what the no-leak check reads
        "training_utterances": sorted({record.utterance for record in split.train}),
    }


def write_holdout_export(workflow_folderpath: str, arm: str, seed: int) -> str:
    """Write the exported holdouts of one arm run; returns the path."""
    folder = os.path.join(workflow_folderpath, "___command_info")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, HOLDOUT_EXPORT_FILENAME)
    heads: dict = {}
    # A selective run retrains only some heads. Keep the holdouts of the others from the
    # previous run under the same arm; a different arm starts a fresh export.
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("reserved_class_arm") == arm:
            heads.update(previous.get("heads", {}))
    heads.update(_HOLDOUT_EXPORT or {})
    document = {
        "schema": 1,
        "reserved_class_arm": arm,
        "seed": seed,
        "heads": dict(sorted(heads.items())),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def resolve_reserved_class_arm(explicit: Optional[str] = None) -> str:
    """The arm in force: the explicit argument, else the experiment override, else the shipped arm.

    With no override this is ``SHIPPED_ARM`` (sparse_1), the production construction. A retired
    arm (``RETIRED_ARMS``) is refused with the reason, not reported as an unknown name.
    """
    arm = explicit if explicit is not None else _EXPERIMENT_RESERVED_CLASS_ARM
    if arm is None:
        return SHIPPED_ARM
    if arm in RETIRED_ARMS:
        raise ValueError(f"reserved-class arm {arm!r} is retired: {RETIRED_ARMS[arm]}")
    if arm == INTERNAL_CME_ARM:  # already resolved by arm_for_workflow; never an experiment arm
        return arm
    if arm not in RESERVED_CLASS_ARMS:
        raise ValueError(
            f"unknown reserved-class arm {arm!r}; expected one of {sorted(RESERVED_CLASS_ARMS)}"
        )
    return arm


def arm_construction(arm: str) -> str:
    """The construction an arm builds: ``"not_here"`` (sparse) or ``"none"`` (internal CME).

    ``arm`` must be a resolved arm (from ``resolve_reserved_class_arm`` or ``arm_for_workflow``).
    """
    if arm in _INTERNAL_CONSTRUCTIONS:
        return _INTERNAL_CONSTRUCTIONS[arm]
    return RESERVED_CLASS_ARMS[arm][0]


def is_internal_cme_folder(folderpath: str) -> bool:
    """True when *folderpath* is the internal command_metadata_extraction workflow (or inside it).

    Compared by resolved path, the same test the trainer's CME check uses.
    """
    cme = os.path.realpath(fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    real = os.path.realpath(folderpath)
    return real == cme or real.startswith(cme + os.sep)


def is_plan_metadata_extraction_folder(folderpath: str) -> bool:
    """True when *folderpath* is the internal plan metadata extraction workflow (or inside it)."""
    plan = os.path.realpath(fastworkflow.get_internal_workflow_path("plan_metadata_extraction"))
    real = os.path.realpath(folderpath)
    return real == plan or real.startswith(plan + os.sep)


def arm_for_workflow(folderpath: Optional[str], explicit: Optional[str] = None) -> str:
    """The reserved-class arm a workflow trains under.

    The sparse "not here" class is an APP construction (design 2.15 and 3.9: app heads). The
    internal CME workflow has no wildcard class at all (``INTERNAL_CME_ARM``, construction
    ``"none"``). No experiment override reaches the CME. Every other workflow resolves as
    ``resolve_reserved_class_arm``.
    """
    if folderpath is not None and is_internal_cme_folder(folderpath):
        return INTERNAL_CME_ARM
    return resolve_reserved_class_arm(explicit)


def not_here_construction(explicit: Optional[str] = None, folderpath: Optional[str] = None) -> dict:
    """The construction record for the arm in force, as written into every trained artifact.

    The shipped app construction is ``{"construction": "sparse", "seeds_per_command": 1}``.
    The internal CME records ``{"construction": "none", "reason": ...}``: it has no wildcard
    class. A version without this record is refused at load (see
    ``artifact_versioning.construction_record_problem``).
    """
    arm = arm_for_workflow(folderpath, explicit)
    if arm_construction(arm) == "none":
        return {"construction": "none", "reason": "internal workflow: no wildcard class"}
    _, _, seeds = RESERVED_CLASS_ARMS[arm]
    record = {
        "construction": "sparse",
        "seeds_per_command": NOT_HERE_SEEDS if seeds is None else seeds,
    }
    # Only a workflow WITH planning commands records the head layout: for any other workflow the
    # record, and so the selective-training signature, is exactly what it was before.
    if folderpath is not None and planning_command_names_of(
            RoutingRegistry.get_definition(folderpath)):
        record[PLANNING_HEADS_RECORD_KEY] = active_planning_heads()
    return record


def construction_records(
    arm: Optional[str] = None,
    head_scope: Optional[str] = None,
    folderpath: Optional[str] = None,
) -> tuple[dict, dict]:
    """The records every training run writes: ``(manifest fields, provenance run metadata)``.

    The trainer writes the first into the version manifest and the second into the run's
    provenance, so both carry the same not-here construction.
    """
    from fastworkflow.train.artifact_versioning import NOT_HERE_RECORD_KEY
    from fastworkflow.train.determinism import get_train_seed

    not_here = not_here_construction(arm, folderpath)
    run_metadata = {
        "reserved_class_arm": arm_for_workflow(folderpath, arm),
        "train_seed": get_train_seed(),
        "not_here": not_here,
    }
    if head_scope is not None:
        run_metadata["head_scope"] = head_scope
    # The scorer refuses a trained copy whose layout is not the arm's (score_planning_match).
    run_metadata["planning_heads"] = active_planning_heads()
    return {NOT_HERE_RECORD_KEY: not_here}, run_metadata


@contextlib.contextmanager
def experiment_reserved_class_arm(arm: Optional[str]):
    """EXPERIMENT-ONLY: select a reserved-class arm for the duration of a ``with`` block."""
    global _EXPERIMENT_RESERVED_CLASS_ARM
    if arm == INTERNAL_CME_ARM:
        raise ValueError(f"{INTERNAL_CME_ARM!r} is the internal CME's arm and is not selectable")
    resolve_reserved_class_arm(arm)  # validates the name before anything changes
    previous = _EXPERIMENT_RESERVED_CLASS_ARM
    _EXPERIMENT_RESERVED_CLASS_ARM = arm
    try:
        yield
    finally:
        _EXPERIMENT_RESERVED_CLASS_ARM = previous


#: EXPERIMENT-ONLY (Phase 9 step 3). The two head layouts that step 3 compares. ``flattened``
#: is today: a head learns its own and inherited commands, and the runtime climbs the parent
#: chain on "not here". ``scoped`` (design 3.9): a head learns only its own "/" commands, plus
#: the core commands, ``parameter_value`` and ``wildcard``; the sparse not-here class covers
#: every other command, and the persona holdout is one split over the whole workflow.
FLATTENED = "flattened"
SCOPED = "scoped"
HEAD_SCOPES = (FLATTENED, SCOPED)

#: EXPERIMENT-ONLY. Never a public setting and never read from the environment. ``None`` is
#: today's layout, which every real training run uses. ``resolve_head_scope`` and
#: ``active_experiment_head_scope`` are the readers.
_EXPERIMENT_HEAD_SCOPE: Optional[str] = None

#: Written by ``train()`` under the scoped layout, once per run, into ``___command_info``.
LOOKUP_HOLDOUT_FILENAME = "lookup_holdout.json"


def active_experiment_head_scope() -> Optional[str]:
    """The experiment head layout for this process, or None for today's layout."""
    return _EXPERIMENT_HEAD_SCOPE


def _require_sparse_arm() -> None:
    """Any active head scope needs the sparse arm: its one global persona holdout is only
    defined alongside the sparse not-here class, which both layouts train with."""
    arm = _EXPERIMENT_RESERVED_CLASS_ARM
    if arm is None:
        raise ValueError(
            "a head scope needs a reserved-class arm: enter experiment_reserved_class_arm() "
            "around it (sparse_1 or sparse_8)"
        )
    if arm_construction(arm) != "not_here":
        raise ValueError(
            f"a head scope needs a sparse (not_here) arm, not {arm!r}"
        )


def resolve_head_scope(explicit: Optional[str] = None) -> str:
    """The head layout in force: the explicit argument, else the experiment override, else flattened."""
    scope = explicit if explicit is not None else _EXPERIMENT_HEAD_SCOPE
    if scope is None:
        return FLATTENED
    if scope not in HEAD_SCOPES:
        raise ValueError(f"unknown head scope {scope!r}; expected one of {list(HEAD_SCOPES)}")
    _require_sparse_arm()
    return scope


@contextlib.contextmanager
def experiment_head_scope(scope: Optional[str]):
    """EXPERIMENT-ONLY: select a head layout for the duration of a ``with`` block.

    ``None`` restores today's layout. Any other layout must be entered INSIDE
    ``experiment_reserved_class_arm`` with a sparse arm, and raises ValueError otherwise.
    """
    global _EXPERIMENT_HEAD_SCOPE
    if scope is not None and scope not in HEAD_SCOPES:
        raise ValueError(f"unknown head scope {scope!r}; expected one of {list(HEAD_SCOPES)}")
    previous = _EXPERIMENT_HEAD_SCOPE
    _EXPERIMENT_HEAD_SCOPE = scope
    try:
        if scope is not None:
            _require_sparse_arm()
        yield
    finally:
        _EXPERIMENT_HEAD_SCOPE = previous


#: The planning-command head layouts. ``LABELLED`` is PRODUCTION (Phase 10 step 1 decision,
#: fix-3rsq.10.12): a planning command is a label of every head whose lookup path holds its scope
#: (``planning_head_labels``). ``NOT_HERE`` is the EXPERIMENT BASELINE the step-1 arms compare
#: against: planning commands only in the sparse "not here" class. Intent detection makes no
#: planning/execution distinction, so no real run selects ``NOT_HERE``.
PLANNING_HEADS_LABELLED = PLANNING_HEADS_PRODUCTION
PLANNING_HEADS_NOT_HERE = "not_here"
PLANNING_HEADS_MODES = (PLANNING_HEADS_LABELLED, PLANNING_HEADS_NOT_HERE)

#: EXPERIMENT-ONLY. ``None`` means production (``PLANNING_HEADS_LABELLED``). Never read from the
#: environment. ``active_planning_heads`` and ``trained_command_labels`` are the readers.
_EXPERIMENT_PLANNING_HEADS: Optional[str] = None


def active_experiment_planning_heads() -> Optional[str]:
    """The experiment planning-head override for this process, or None for production."""
    return _EXPERIMENT_PLANNING_HEADS


def active_planning_heads() -> str:
    """The planning-head layout in force: the experiment override, else production (labelled)."""
    override = active_experiment_planning_heads()
    return PLANNING_HEADS_LABELLED if override is None else override


@contextlib.contextmanager
def experiment_planning_heads(mode: Optional[str]):
    """EXPERIMENT-ONLY: select a planning-head layout for the duration of a ``with`` block.

    ``None`` restores production (labelled). ``PLANNING_HEADS_NOT_HERE`` is the baseline arm.
    Any non-None mode must be entered INSIDE ``experiment_reserved_class_arm`` with a sparse arm,
    and raises ValueError otherwise.
    """
    global _EXPERIMENT_PLANNING_HEADS
    if mode is not None and mode not in PLANNING_HEADS_MODES:
        raise ValueError(f"unknown planning-head mode {mode!r}; expected one of {list(PLANNING_HEADS_MODES)}")
    previous = _EXPERIMENT_PLANNING_HEADS
    _EXPERIMENT_PLANNING_HEADS = mode
    try:
        if mode is not None:
            _require_sparse_arm()
        yield
    finally:
        _EXPERIMENT_PLANNING_HEADS = previous


def planning_head_labels(ctx_name: str, crd: RoutingDefinition) -> set[str]:
    """The planning commands head *ctx_name* labels under ``PLANNING_HEADS_LABELLED``.

    A planning command is a label of a head whose lookup path holds its scope: the head's own
    context and the contexts it inherits (``inherited_base_contexts``). A root planning command
    (no scope) belongs to the global head ``*`` only, as root execution commands do. Every
    planning command in scope is a label, whether or not it declares parameters: its seeds are
    its rows, as a command's seeds are an execution command's. Pure in its arguments apart from
    the metadata reads.
    """
    if ctx_name == "*":  # the global head holds the root scope only
        scopes = {""}
    else:
        scopes = {ctx_name} | set(crd.context_model.inherited_base_contexts(ctx_name))
    return {name for name in planning_command_names_of(crd) if name.rpartition("/")[0] in scopes}


def planning_labels_in_layout(ctx_name: str, crd: RoutingDefinition) -> set[str]:
    """The planning commands *ctx_name* labels in the layout in force: its labels when labelled, else none."""
    if active_planning_heads() == PLANNING_HEADS_LABELLED:
        return planning_head_labels(ctx_name, crd)
    return set()


def planning_command_names_of(crd: RoutingDefinition) -> list[str]:
    """Every planning command the routing definition's directory holds, sorted."""
    cmd_dir = crd.command_directory
    names = []
    for name in sorted(cmd_dir.get_commands()):
        try:
            if is_visible_planning(cmd_dir.get_command_metadata(name)):
                names.append(name)
        except KeyError:
            continue
    return names


def _is_action_only(cmd_dir: object, name: str) -> bool:
    """True for a command with no ``Signature`` (a startup command, whatever its name): never a label or a row."""
    return not _requires_utterances(cmd_dir, name)


def _stable_rank(context_name: str, command_name: str, seed: str) -> str:
    """A process-independent sort key for one seed under one head (never Python's salted hash)."""
    payload = "\x00".join((context_name, command_name, seed))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_not_here_sample(
    context_name: str, command_name: str, seeds: Sequence[str], count: int
) -> list[str]:
    """Up to *count* seeds of *command_name*, chosen for one head (design 2.50).

    The lowest stable hash of (head, command, seed) first. Chosen this way, not as the first
    seed, so a command's sample does not always fall on the same phrasing position, and the
    head name is part of the key, so two heads need not record the same sample. Deterministic
    in any process: it reads nothing but its arguments.
    """
    if count <= 0:
        return []
    ranked = sorted(
        set(seeds),
        key=lambda seed: _stable_rank(context_name, command_name, seed),
    )
    return ranked[:count]


def resolve_not_here_sample(
    context_name: str,
    command_name: str,
    seeds: Sequence[str],
    count: int,
    recorded: Optional[Sequence[str]],
) -> list[str]:
    """The sample a head uses for *command_name*: the recorded one while it still holds, else a new choice.

    The recorded sample is reused whenever every recorded seed text still exists among the
    command's seeds and it is no larger than *count*. If any recorded seed was edited away, the
    command's sample is chosen again (that is the only way a head's sample for a command moves
    without ``--regenerate-utterances``).
    """
    if recorded:
        present = set(seeds)
        if len(recorded) <= count and all(text in present for text in recorded):
            return list(recorded)
    return choose_not_here_sample(context_name, command_name, seeds, count)


def _is_planning_name(cmd_dir: object, name: str) -> bool:
    try:
        return bool(is_visible_planning(cmd_dir.get_command_metadata(name)))
    except KeyError:
        return False


def not_here_seed_rows(
    workflow_folderpath: str,
    crd: RoutingDefinition,
    head_labels: set[str],
    seeds_per_command: int,
    *,
    context_name: str = "",
    recorded: Optional[Mapping[str, Sequence[str]]] = None,
) -> tuple[dict[str, list[str]], int]:
    """``({foreign command: up to N seeds}, available foreign seed count)`` for one head.

    When no foreign command yields a seed, the built-in vague replies stand in under
    ``BUILT_IN_VAGUE_SOURCE`` (never for the internal CME, which has no class).

    A foreign command is every command of the workflow that the head does not label:
    execution commands from the routing definition (never the reserved wildcard, never
    a planning command), then planning commands, whose seeds are read exactly as their
    utterances are read today (``workflow_agent.planning_command_utterances``). Only
    hand-written seeds are used, so no LLM generation runs for this class.

    ``recorded`` maps a command to the sample this head recorded for it in its last train
    (design 2.50). Each command's sample is ``resolve_not_here_sample``: the recorded one while
    it still holds, else chosen again. ``None`` chooses every sample afresh.
    """
    from fastworkflow.agent.workflow_agent import planning_command_utterances

    cmd_dir = crd.command_directory
    execution: set[str] = set(cmd_dir.core_command_names)
    for names in crd.contexts.values():
        execution |= set(names)

    picks: dict[str, list[str]] = {}
    available = 0
    for name in sorted(execution - set(head_labels)):
        if label_of(name) == WILDCARD_LABEL or _is_planning_name(cmd_dir, name):
            continue
        seeds = list(_get_seed_utterances(cmd_dir, name))
        available += len(seeds)
        picks[name] = resolve_not_here_sample(
            context_name, name, seeds, seeds_per_command, (recorded or {}).get(name))

    for name, seeds in sorted(planning_command_utterances(workflow_folderpath).items()):
        if name in head_labels or name in picks or label_of(name) == WILDCARD_LABEL:
            continue
        seeds = list(seeds)
        available += len(seeds)
        picks[name] = resolve_not_here_sample(
            context_name, name, seeds, seeds_per_command, (recorded or {}).get(name))

    if not any(picks.values()) and not is_internal_cme_folder(workflow_folderpath):
        # No foreign command supplies a row (a single-context workflow without core commands):
        # the head would have no "not here" class. Seed it from the built-in vague replies.
        picks[BUILT_IN_VAGUE_SOURCE] = list(BUILT_IN_VAGUE_REPLIES)
        available += len(BUILT_IN_VAGUE_REPLIES)
    return picks, available


def not_here_samples_for_context(
    workflow_folderpath: str,
    crd: RoutingDefinition,
    ctx_name: str,
    reserved_class_arm: Optional[str] = None,
    recorded: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[dict[str, list[str]]]:
    """The not-here samples one head records: ``{foreign command: its sample}``, or None.

    None when the arm builds no not-here class (the internal CME). Commands with no seeds have
    no sample and are left out. The planner and the trainer both call this, so the signature
    records exactly what the trainer uses.
    """
    arm = arm_for_workflow(workflow_folderpath, reserved_class_arm)
    if arm_construction(arm) == "none":
        return None
    picks, _available = not_here_seed_rows(
        workflow_folderpath,
        crd,
        trained_command_labels(ctx_name, crd),
        _seeds_per_command(arm),
        context_name=ctx_name,
        recorded=recorded,
    )
    return {name: list(rows) for name, rows in sorted(picks.items()) if rows}


def _seeds_per_command(arm: str) -> int:
    _, _, seeds = RESERVED_CLASS_ARMS[arm]
    return NOT_HERE_SEEDS if seeds is None else seeds


def _get_seed_utterances(cmd_dir: object, cmd_name: str) -> list[str]:
    """A command's hand-written seeds (``Signature.plain_utterances``); none without an Input."""
    if not _requires_utterances(cmd_dir, cmd_name):
        return []
    metadata = cmd_dir.get_utterance_metadata(cmd_name)
    return list(metadata.plain_utterances) if metadata else []


def _not_here_selection(
    ctx_name: str,
    crd: RoutingDefinition,
    workflow: fastworkflow.Workflow,
    *,
    context_utterances: set[str],
    seeds_per_command: int,
    recorded: Optional[Mapping[str, Sequence[str]]] = None,
) -> "EscalationSelection":
    """The sparse "not here" class for one head (experiment arm; see RESERVED_CLASS_ARMS)."""
    picks, available = not_here_seed_rows(
        workflow.folderpath,
        crd,
        trained_command_labels(ctx_name, crd),
        seeds_per_command,
        context_name=ctx_name,
        recorded=recorded,
    )
    chosen = {row for rows in picks.values() for row in rows}
    rows = sorted(chosen - set(context_utterances))
    return EscalationSelection(
        rows=rows,
        included=bool(rows),
        raw_candidate_count=available,
        deduplicated_candidate_count=len(rows),
        always_include_rows=[],
        budget=None,
        coverage_floor=0,
        not_here_samples={name: list(rows_) for name, rows_ in sorted(picks.items()) if rows_},
    )


class EscalationSelection(NamedTuple):
    """One context's escalation-class rows and every denominator that chose them."""

    #: Rows labelled `WILDCARD_LABEL`, sorted. Empty when no class is emitted.
    rows: list[str]
    #: False when no class is emitted (the internal CME, which has no wildcard class), which
    #: is a different thing from a class that came out empty after selection.
    included: bool
    raw_candidate_count: int
    deduplicated_candidate_count: int
    always_include_rows: list[str]
    #: None when nothing was emitted, so no budget was computed.
    budget: Optional[int]
    coverage_floor: int
    #: The sparse class's samples for this head, ``{foreign command: its sample}`` (design 2.50),
    #: or None when no not-here class is built. These are what the head's signature records.
    not_here_samples: Optional[dict[str, list[str]]] = None


def select_escalation_rows(
    ctx_name: str,
    crd: RoutingDefinition,
    workflow: fastworkflow.Workflow,
    *,
    context_utterances: set[str],
    reserved_class_arm: Optional[str] = None,
    recorded_samples: Optional[Mapping[str, Sequence[str]]] = None,
) -> EscalationSelection:
    """Choose the escalation-class (``wildcard``) rows for one context, and report every denominator.

    ``reserved_class_arm`` is EXPERIMENT-ONLY (see ``RESERVED_CLASS_ARMS``). Left at
    ``None`` it resolves to the experiment override, then to ``SHIPPED_ARM`` (sparse_1),
    which is what every real training run builds. Every sparse arm samples seeds of the
    foreign commands (``not_here_seed_rows``), so no ancestor is read. The internal CME has
    no wildcard class and returns an empty, not-included selection.

    Extracted from `train()` so the whole escalation-row decision is one callable, and a test
    that drives it drives the shipped decision.
    """
    arm = arm_for_workflow(workflow.folderpath, reserved_class_arm)
    if arm_construction(arm) == "none":
        return EscalationSelection(
            rows=[],
            included=False,
            raw_candidate_count=0,
            deduplicated_candidate_count=0,
            always_include_rows=[],
            budget=None,
            coverage_floor=0,
        )
    return _not_here_selection(
        ctx_name,
        crd,
        workflow,
        context_utterances=context_utterances,
        seeds_per_command=_seeds_per_command(arm),
        recorded=recorded_samples,
    )


def _recorded_persona_batches(recorder) -> Optional[list[list[str]]]:
    """Reconstruct each command's generation batches from provenance (AR1).

    Batches are consecutive slices of the recorded ordered `persona_ids`, sized by
    `personas_per_batch` in the same record's `generator_config` -- both already
    written for reproducibility, so nothing new has to be recorded to align the
    holdout to them.

    Returns None when no record carries a batch size, which is the signal to
    `split_by_persona` that it must disclose an unaligned split rather than imply a
    clean one.
    """
    batches: list[list[str]] = []
    for provenance in recorder.records.values():
        persona_ids = list(provenance.persona_ids or ())
        size = (provenance.generator_config or {}).get("personas_per_batch")
        if not persona_ids or not isinstance(size, int) or size < 1:
            continue
        batches.extend(
            persona_ids[start:start + size]
            for start in range(0, len(persona_ids), size)
        )
    return batches or None


def lookup_split_command_names(crd: RoutingDefinition, contexts: Iterable[str]) -> list[str]:
    """The commands the scoped layout's ONE persona holdout is drawn over (sorted).

    Every command any context OWNS, once each, minus planning commands and action-only
    commands (no ``Signature``, such as a startup command): those are never held out. Inherited commands are not added
    separately because each is owned by exactly one context.
    """
    cmd_dir = crd.command_directory
    names: set[str] = set()
    for context_name in contexts:
        names |= own_command_names(crd, context_name)
    return sorted(
        name for name in names
        if label_of(name) != WILDCARD_LABEL
        and not _is_planning_name(cmd_dir, name)
        and not _is_action_only(cmd_dir, name)
    )


def lookup_holdout_split(
    labeled: Sequence[heldout_evaluation.LabeledUtterance],
    seed: int,
    persona_batches: Optional[list[list[str]]],
) -> heldout_evaluation.PersonaSplit:
    """The ONE persona split of a scoped run. Same ``split_by_persona`` and seed as step 1."""
    return heldout_evaluation.split_by_persona(
        list(labeled), seed=seed, persona_batches=persona_batches)


def scoped_head_split(
    global_split: heldout_evaluation.PersonaSplit,
    head_rows: Sequence[heldout_evaluation.LabeledUtterance],
    head_labels: set[str],
) -> heldout_evaluation.PersonaSplit:
    """One head's view of the global split.

    ``train``: this head's rows (own, not-here, wildcard, parameter_value) minus EVERY globally
    held-out utterance text. ``heldout``: the global held-out rows whose label this head
    trains, so the per-head scoring is unchanged. Pure: it reads nothing but its arguments,
    which is what makes "the same split whichever head asks" a checkable property.
    """
    held_texts = {record.utterance for record in global_split.heldout}
    train = [record for record in head_rows if record.utterance not in held_texts]
    heldout = [record for record in global_split.heldout if record.label in head_labels]
    return heldout_evaluation.PersonaSplit(
        train=train,
        heldout=heldout,
        train_personas=list(global_split.train_personas),
        heldout_personas=sorted({record.persona for record in heldout}),
        notes=list(global_split.notes),
    )


def write_lookup_holdout_export(
    workflow_folderpath: str,
    arm: str,
    seed: int,
    split: heldout_evaluation.PersonaSplit,
    *,
    head_scope: str,
    command_names: Sequence[str],
) -> str:
    """Write ``___command_info/lookup_holdout.json`` for one run under any head scope."""
    folder = os.path.join(workflow_folderpath, "___command_info")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, LOOKUP_HOLDOUT_FILENAME)
    notes = [
        f"one persona split over {len(command_names)} owned command(s); planning and "
        f"action-only commands (no Signature) are not held out",
        *split.notes,
    ]
    document = {
        "schema": 1,
        "head_scope": head_scope,
        "arm": arm,
        "seed": seed,
        "heldout": [
            {"utterance": record.utterance, "label": record.label, "persona": record.persona}
            for record in split.heldout
        ],
        "heldout_personas": sorted(split.heldout_personas),
        "notes": notes,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _compute_lookup_split(
    workflow: fastworkflow.Workflow,
    crd: RoutingDefinition,
    contexts: Iterable[str],
    command_cache: dict[str, list[str]],
    seed: int,
    head_scope: str,
) -> Optional[heldout_evaluation.PersonaSplit]:
    """Compute the run's global split BEFORE any head trains, under either head layout.

    Generation (or the utterance cache) is reached through ``_get_cached_command_utterances``,
    which fills ``command_cache`` and the provenance recorder for every owned command. Each
    head later reads the same cache, so no command is generated twice. Returns None when no
    provenance recorder is active, which is the same condition under which today's per-head
    split is skipped.
    """
    recorder = get_provenance_recorder()
    if recorder is None:
        print("Scoped layout: no provenance recorder, so no persona holdout is drawn")
        return None
    names = lookup_split_command_names(crd, contexts)
    rows: list[tuple[str, str]] = []
    for name in names:
        for utterance in _get_cached_command_utterances(
            workflow, workflow.folderpath, crd.command_directory, name, command_cache
        ):
            rows.append((utterance, name))
    persona_by_utterance: dict[str, str] = {}
    for provenance in recorder.records.values():
        persona_by_utterance.update(provenance.utterance_personas)
    labeled = [
        heldout_evaluation.LabeledUtterance(
            utterance=utterance,
            label=label,
            persona=persona_by_utterance.get(utterance, heldout_evaluation.SEED_PERSONA_ID),
        )
        for utterance, label in rows
    ]
    split = lookup_holdout_split(labeled, seed, _recorded_persona_batches(recorder))
    arm = active_experiment_arm()
    if arm is None:  # unreachable: resolve_head_scope refuses scoped without an arm
        raise ValueError("scoped layout reached the holdout without an arm")
    path = write_lookup_holdout_export(
        workflow.folderpath, arm, seed, split, head_scope=head_scope, command_names=names)
    print(
        f"Head scope {head_scope}: one persona holdout over {len(labeled)} rows of {len(names)} "
        f"command(s); {len(split.heldout)} held out -> {path}"
    )
    return split


def _score_heldout_context(
    report: heldout_evaluation.HeldoutReport,
    heldout_records: list[heldout_evaluation.LabeledUtterance],
    benchmark_cases: list[heldout_evaluation.BenchmarkCase],
    predict_labels: Optional[heldout_evaluation.PredictFn],
    split_for_calibration: Optional[heldout_evaluation.PersonaSplit] = None,
) -> None:
    """Score independent persona-holdout and fixed-benchmark populations."""
    if heldout_records:
        if predict_labels is None:
            raise ValueError("persona holdout scoring requires a predictor")
        # Escalation-class rows are appended to the training tuples before the
        # persona split is drawn, so they arrive here carrying real persona
        # attribution and used to be scored as routes. Split them onto their own
        # axis (D2) rather than dropping them, so the class is measured, not hidden.
        if split_for_calibration is not None:
            report.leak_calibration = heldout_evaluation.calibrate_leak(
                split_for_calibration)

        routable, escalation_class = heldout_evaluation.partition_by_routability(
            heldout_records)
        if routable:
            report.routing = heldout_evaluation.score_routing(
                routable, predict_labels)
        if escalation_class:
            report.holdout_escalation = heldout_evaluation.score_escalation(
                escalation_class, predict_labels)
        if not routable:
            report.notes.append(
                f"Persona holdout for this context is entirely escalation-class "
                f"({len(escalation_class)} rows); no held-out routing was scored."
            )
    else:
        report.notes.append(
            "Persona holdout unavailable; persona-held-out routing was not scored. "
            "Fixed benchmark routing and escalation remain independently measurable."
        )

    # `kind` is required. Omitting it raised a TypeError that the guard
    # below turned into a note on the report, so escalation silently
    # never scored while routing kept working -- the failure looked like
    # "this workflow has no escalation cases" (bd fix-588).
    escalation_cases = heldout_evaluation.benchmark_cases_for_context(
        benchmark_cases, report.context, "escalation")
    routing_cases = heldout_evaluation.benchmark_cases_for_context(
        benchmark_cases, report.context, "routing")
    if (escalation_cases or routing_cases) and predict_labels is None:
        raise ValueError("benchmark scoring requires a predictor")

    if escalation_cases:
        report.escalation = heldout_evaluation.score_escalation(
            escalation_cases, predict_labels)
    if routing_cases:
        report.benchmark_routing = heldout_evaluation.score_routing(
            routing_cases, predict_labels)
        print(
            f"Benchmark routing [{report.context}]: "
            f"{report.benchmark_routing.top1_correct}"
            f"/{report.benchmark_routing.total} top-1"
        )


def preflight_benchmark(
    workflow_folderpath: str,
    crd: Optional[RoutingDefinition] = None,
) -> list[heldout_evaluation.BenchmarkCase]:
    """Load and validate the workflow's benchmark file, raising on a seed leak.

    Returns the loaded cases (empty when the workflow ships no benchmark) so a caller
    that is about to score can reuse them.

    Runs off the command directory and routing definition alone — no workflow instance,
    no model, no LLM — which is what lets `train.__main__.train_workflow` call it before
    it spends anything. `train` calls it again as a second line of defence for direct
    callers that bypass the orchestrator.

    ------------------------------------------------------------------
    R1b: a benchmark that shares phrasings with the training seeds measures
    memorisation, so it is checked BEFORE any training happens. Failing fast is the
    point: discovering the leak after the run would mean every number it produced has
    to be thrown away, and the run costs LLM calls and GPU minutes.
    ------------------------------------------------------------------
    """
    benchmark_path = heldout_evaluation.default_benchmark_path(workflow_folderpath)
    if not os.path.isfile(benchmark_path):
        return []

    if crd is None:
        crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    cmd_dir = crd.command_directory

    benchmark_cases = heldout_evaluation.load_benchmark_file(benchmark_path)
    seed_utterances_by_command: dict[str, list[str]] = {}
    for command_key in cmd_dir.get_execution_commands():
        if metadata := cmd_dir.get_utterance_metadata(command_key):
            seed_utterances_by_command[command_key] = list(metadata.plain_utterances)
    # Deliberately NOT caught: an unnoticed leak silently turns the whole evaluation
    # into a memorisation score, which is the failure this disjointness check
    # exists to remove.
    heldout_evaluation.assert_benchmark_disjoint_from_seeds(
        benchmark_cases, seed_utterances_by_command)
    # A close-but-not-equal match is a judgement call, so it is reported, not enforced.
    for warning in heldout_evaluation.find_near_duplicate_benchmark_cases(
        benchmark_cases, seed_utterances_by_command
    ):
        logger.warning(warning)

    # A benchmark case can also be defective in a way that has nothing to do with
    # leakage: a routing case naming a label the context does not have can never
    # pass, and an "escalation" case whose command is not actually absent here and
    # present elsewhere in the workflow is not testing escalation at all. Either one drags the
    # reported score down for a reason that is not the model's fault, so both are
    # reported before the run rather than discovered as a mysteriously low number.
    core_command_names = set(cmd_dir.core_command_names)
    context_label_space = {
        context_name: set(command_list) | core_command_names
        for context_name, command_list in crd.contexts.items()
    }
    for problem in (
        heldout_evaluation.validate_routing_cases(
            benchmark_cases, context_label_space)
        + heldout_evaluation.validate_escalation_cases(
            benchmark_cases, context_label_space)
    ):
        logger.warning(f"Benchmark defect: {problem}")

    print(f"Benchmark: {len(benchmark_cases)} case(s) from {benchmark_path}")
    return benchmark_cases


def train(
    workflow: fastworkflow.Workflow,
    contexts_to_train: Optional[set[str]] = None,
    *,
    stop_before_fit: bool = False,
    not_here_samples: Optional[Mapping[str, Optional[Mapping[str, Sequence[str]]]]] = None,
    intent_model_layout: Optional[str] = None,
) -> dict[str, dict[str, list[str]]]:
    """Train intent-classification models **per command context**.

    ``intent_model_layout`` ``"embedding"`` fits ONE linear head over a frozen sentence encoder
    for the whole workflow instead (`train.embedding_head.EmbeddingLayoutTrainer`), and
    ``"workflow"`` fine-tunes ONE large model (`train.workflow_model.LayoutTrainer`): the
    per-context loop below still builds each context's label set, utterances and provenance,
    hands the context's own rows over, and fits nothing itself. No "not here" seed rows are
    built, so the returned samples are empty. ``None`` (and ``"per_context"``) is the
    per-context layout, whatever the default of the setting is. The setting is the
    caller's to read (`train.intent_model_layout.layout_for_workflow`); this function never
    reads it.

    ``contexts_to_train`` is the selective-retraining hook. ``None`` trains every
    context. Automatic planning passes a set when unchanged contexts can safely be
    carried forward. A set restricts the run
    to those contexts, and the CALLER then becomes responsible for carrying every other
    context's artifacts forward into the version being assembled
    (``train.selective_training.carry_forward_contexts``). Skipping a context here and
    forgetting that is not a slow workflow, it is an un-trained one: `publish_version`
    removes the compatibility entry of any context its version does not contain.

    ``stop_before_fit`` runs the complete per-context training-data assembly, including
    provenance recording, but skips model loading, fitting, evaluation, and artifact
    writes. It is a cheap support seam for checking trainer orchestration without
    downloading or fitting HuggingFace models.

    ``not_here_samples`` maps each context to the not-here samples its signature recorded
    (design 2.50). A head reuses a recorded sample while it still holds and chooses again
    otherwise; ``None`` chooses every sample afresh. Returns the samples each trained head
    used, ``{context: {foreign command: its sample}}``, which the caller checks against the
    signature it writes.
    """

    import time

    from fastworkflow.train.determinism import get_model_seed, get_split_seed, seed_everything

    # Seed before anything samples: utterance generation picks personas, torch initialises
    # classifier heads, and the train/test split shuffles. Seeding is necessary but NOT
    # sufficient for reproducibility -- the iteration order of the sets feeding the split
    # matters just as much, which is why the loops below sort (fix-9mo).
    # Model randomness (weight init, shuffling, dropout) follows --train-seed when one is set.
    # The persona split below does not: it uses get_split_seed(), so every seed and arm holds
    # out the same cases.
    seed = seed_everything(get_model_seed())
    split_seed = get_split_seed()
    print(f"Training seed: {seed} (persona split seed: {split_seed})")

    heldout_reports: list[heldout_evaluation.HeldoutReport] = []

    workflow_folderpath = workflow.folderpath
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath, load_cached=False)
    cmd_dir = crd.command_directory

    # Second line of defence. `train.__main__.train_workflow` runs the same preflight
    # before it spends anything on parameter-example generation; this call covers direct
    # callers of `train()` and re-reads the file in case it changed since.
    benchmark_cases = preflight_benchmark(workflow_folderpath, crd)

    context_utterance_cache = {}
    # Generation identity is the fully-qualified command. Context identity belongs to
    # the labelled-row use recorded below, not to the LLM draw or utterance cache key.
    command_utterance_cache: dict[str, list[str]] = {}

    # Get contexts specific to this workflow (not from command_metadata_extraction)
    # The set itself is computed by `selective_training.contexts_for_training` so that the
    # selective-retraining planner and this loop cannot disagree about which contexts
    # exist. A context the
    # planner never considered would be neither retrained nor carried forward, and that
    # presents as part of a workflow silently becoming untrained.
    context_set_for_training = contexts_for_training(workflow_folderpath)
    used_not_here_samples: dict[str, dict[str, list[str]]] = {}

    workflow_layout_trainer = None
    if intent_model_layout not in (None, "per_context"):
        if intent_model_layout not in ("workflow", "embedding"):
            raise ValueError(f"unknown intent model layout {intent_model_layout!r}")
        if contexts_to_train is not None or active_experiment_head_scope() is not None:
            raise ValueError(
                f"The '{intent_model_layout}' intent model layout trains one model for every "
                f"context: it cannot train a selection of contexts or an experiment head layout."
            )
        # Lazy: both modules import this one.
        if intent_model_layout == "embedding":
            from fastworkflow.train import embedding_head

            trainer_class = embedding_head.EmbeddingLayoutTrainer
        else:
            from fastworkflow.train import workflow_model

            trainer_class = workflow_model.LayoutTrainer
        workflow_layout_trainer = trainer_class(
            workflow_folderpath, crd, seed=seed, benchmark_cases=benchmark_cases)

    # EXPERIMENT-ONLY (Phase 9 step 3). Under ANY active head layout the persona holdout is ONE
    # split over every owned command of the workflow, drawn here before any head trains, so a
    # head trained first and a head trained last hold out the same utterances, and flattened and
    # scoped runs hold out the same cases. It is computed over ALL contexts, not only the ones this
    # call retrains, so a selective run holds out the same cases as a full one. The commands are
    # generated (or read from the cache) here, once.
    lookup_split: Optional[heldout_evaluation.PersonaSplit] = None
    scope = active_experiment_head_scope()
    if scope is not None:
        scope = resolve_head_scope()  # refuses any layout without a sparse arm, before generation
        lookup_split = _compute_lookup_split(
            workflow, crd, context_set_for_training, command_utterance_cache, split_seed, scope)

    if contexts_to_train is not None:
        requested = set(contexts_to_train)
        skipped = sorted(context_set_for_training - requested)
        context_set_for_training = context_set_for_training & requested
        if skipped:
            print(f"Selective training: skipping {len(skipped)} context(s) whose "
                  f"artifacts the caller will carry forward: {', '.join(skipped)}")

    # Only iterate through contexts defined in this specific workflow.
    # sorted(): context_set_for_training is a set, and a context visited first as an
    # ANCESTOR gets its cache populated from context_model.commands(), which excludes the
    # core commands. Iterating in hash order therefore decided, per interpreter start,
    # which contexts lost their core-command labels (AR5). Sorting makes the order stable;
    # the cache-fill below makes the outcome order-independent as well.
    #
    # Sorting alone was not enough, because a STABLE order is still an order: whether a
    # context's cache entry held core commands when a descendant read it depended on
    # their two names, so renaming one context moved another context's escalation
    # budget. `select_escalation_rows` filters core commands out of the escalation
    # population instead of relying on the fill (bd fix-4ej).
    for ctx_name in sorted(context_set_for_training):
        print(f"\n=== Training model for workflow_folderpath: {workflow_folderpath.split('/')[-1]} and context: {ctx_name} ===\n")
        
        # Build the label set for this context
        train_cmds: set[str] = trained_command_labels(ctx_name, crd)
        if not train_cmds:
            print(f"Skipping context {ctx_name} - no commands to train")
            continue  # nothing to train

        context_utterances = set()
        utterance_command_tuples: list[tuple[str, str]] = []
        map_cmd_2_uttlist = cache_context_command_utterances(
            ctx_name,
            crd,
            workflow,
            context_utterance_cache,
            command_utterance_cache,
        )
        for cmd_name, cmd_utterances in map_cmd_2_uttlist.items():
            _record_context_training(
                ctx_name, cmd_name, cmd_dir, cmd_utterances
            )
            utterance_command_tuples.extend(
                list(zip(cmd_utterances, [cmd_name] * len(cmd_utterances)))
            )
            context_utterances |= set(cmd_utterances)

        if not context_utterances:
            print(f"Skipping context {ctx_name} - no utterances available")
            continue  # skip empty

        own_rows = len(utterance_command_tuples)
        if workflow_layout_trainer is not None:
            # The context's own rows are all the one model takes from it; the persona holdout
            # and the reserved classes are the layout trainer's, and nothing is fitted here.
            workflow_layout_trainer.add_context(ctx_name, map_cmd_2_uttlist, train_cmds)
            continue
        escalation = select_escalation_rows(
            ctx_name,
            crd,
            workflow,
            context_utterances=context_utterances,
            recorded_samples=(
                None if not_here_samples is None
                else not_here_samples.get(ctx_name)
            ),
        )
        if escalation.not_here_samples is not None:
            used_not_here_samples[ctx_name] = escalation.not_here_samples
        if escalation.included:
            weight = (
                BUILT_IN_VAGUE_WEIGHT
                if set(escalation.not_here_samples or {}) == {BUILT_IN_VAGUE_SOURCE}
                else 1
            )
            utterance_command_tuples.extend(
                list(zip(escalation.rows, [WILDCARD_LABEL] * len(escalation.rows))) * weight
            )
        _record_wildcard_context_training(
            ctx_name,
            escalation.rows if escalation.included else None,
            own_row_count=own_rows,
            raw_candidate_count=escalation.raw_candidate_count,
            deduplicated_candidate_count=escalation.deduplicated_candidate_count,
            always_include_rows=escalation.always_include_rows,
            selected_budget=escalation.budget,
            coverage_floor=escalation.coverage_floor,
            sampled_commands=(
                None if escalation.not_here_samples is None
                else len(escalation.not_here_samples)
            ),
        )

        # PARAMETER_VALUE_LABEL is the PARAMETER_EXTRACTION stage's bare-value catcher and
        # is emitted in EVERY context: a user can type a bare value anywhere. These are the
        # seven literals that used to be trained into the wildcard class, which is what
        # taught the escalation classifier that "france" means "escalate to my parent".
        parameter_value_rows = sorted(set(PARAMETER_VALUE_PLACEHOLDERS) - context_utterances)
        if parameter_value_rows:
            utterance_command_tuples.extend(
                list(zip(parameter_value_rows,
                         [PARAMETER_VALUE_LABEL] * len(parameter_value_rows)))
            )
        _record_parameter_value_context_training(
            ctx_name, parameter_value_rows, own_rows
        )


        # ------------------------------------------------------------------
        # R1a: reserve WHOLE PERSONAS before training, for a real generalisation
        # measure. The class-aware split below still divides rows from the same
        # generated corpus, so utterances written by the same persona, from the same
        # seed utterance, can land on both sides of it. Scoring that is close to scoring
        # memorisation, which is why the F1 it produces has never been a safe basis
        # for "did this change help?". It is still computed and still used to
        # calibrate the ambiguity thresholds -- it is only its use as a QUALITY
        # metric that is unsound -- so it is reported under a name that says what it is.
        # ------------------------------------------------------------------
        heldout_records: list[heldout_evaluation.LabeledUtterance] = []
        heldout_personas: list[str] = []
        split_notes: list[str] = []
        # Initialised here rather than only inside the branch below: the scoring call
        # is far downstream and would raise NameError on the no-recorder path.
        heldout_split: Optional[heldout_evaluation.PersonaSplit] = None
        if (recorder := get_provenance_recorder()) is not None:
            # utterance text -> persona id, across every command generated this run.
            # Anything absent is a hand-written seed utterance, which split_by_persona
            # always keeps in training: a developer's declared input is not
            # generalisation data.
            persona_by_utterance: dict[str, str] = {}
            for provenance in recorder.records.values():
                persona_by_utterance.update(provenance.utterance_personas)

            labeled = [
                heldout_evaluation.LabeledUtterance(
                    utterance=utterance,
                    label=label,
                    persona=persona_by_utterance.get(
                        utterance, heldout_evaluation.SEED_PERSONA_ID),
                )
                for utterance, label in utterance_command_tuples
            ]
            if lookup_split is not None:
                # EXPERIMENT (Phase 9 step 3, scoped layout). The one global split, seen from this
                # head: every globally held-out text is removed from this head's rows, wildcard
                # and parameter_value rows included. Set here, not in the shared block below,
                # because a head with no held-out cases of its own must still drop the others'.
                split = scoped_head_split(lookup_split, labeled, train_cmds)
                utterance_command_tuples = [(record.utterance, record.label) for record in split.train]
            else:
                # Phase 9. The held-out cases must not depend on the reserved class, so this
                # head's persona split is taken over its OWN rows only (its own commands, the
                # same under every arm and under the shipped sparse_1 construction). The reserved
                # rows (not-here and bare values) are always trained, except any text the split
                # holds out. Production and the experiment arms take this one path.
                own_rows = [
                    record for record in labeled
                    if record.label in train_cmds
                    and record.label not in (WILDCARD_LABEL, PARAMETER_VALUE_LABEL)
                ]
                own_split = heldout_evaluation.split_by_persona(
                    own_rows,
                    seed=split_seed,
                    persona_batches=_recorded_persona_batches(recorder),
                )
                held_out_texts = {record.utterance for record in own_split.heldout}
                reserved_rows = [
                    record for record in labeled
                    if record.label in (WILDCARD_LABEL, PARAMETER_VALUE_LABEL)
                    and record.utterance not in held_out_texts
                ]
                split = heldout_evaluation.PersonaSplit(
                    train=[*own_split.train, *reserved_rows],
                    heldout=own_split.heldout,
                    train_personas=own_split.train_personas,
                    heldout_personas=own_split.heldout_personas,
                    notes=own_split.notes,
                )
            _export_holdout(ctx_name, train_cmds, split)
            if split.heldout:
                # Held-out personas are removed from training entirely. Without this
                # the score below would be measuring the training set.
                utterance_command_tuples = [
                    (record.utterance, record.label) for record in split.train
                ]
                heldout_records = list(split.heldout)
                heldout_split = split
                heldout_personas = list(split.heldout_personas)
                split_notes = list(split.notes)
                print(
                    f"Held out {len(heldout_records)} utterances from "
                    f"{len(heldout_personas)} personas for evaluation"
                )
            else:
                split_notes = list(split.notes)
                print(
                    "No held-out personas available for this context; "
                    "only the in-distribution score will be reported"
                )

        if stop_before_fit:
            print(
                f"Training data assembly complete for context {ctx_name}; "
                "stopping before model fitting"
            )
            continue

        print("Utterances generation complete! Beginning model pipeline training\n")

        # ==================================================================================
        # Original training procedure below, with only artefact paths changed to per-context
        # ==================================================================================

        # unpack the test data and train data
        X, y = zip(*utterance_command_tuples)
        num= len(set(y))
        k_val = 3 if num>2 else 2
        # Base models are configurable so downstream apps can swap them without
        # code changes. The defaults are transformers 5.x-compatible BERT/DistilBERT
        # checkpoints that ship a `model_type` and a loadable tokenizer.
        tiny_model_name = fastworkflow.get_env_var(
            "INTENT_DETECTION_TINY_MODEL", default="google/bert_uncased_L-4_H-128_A-2")
        large_model_name = fastworkflow.get_env_var(
            "INTENT_DETECTION_LARGE_MODEL", default="distilbert-base-uncased")

        model_name = tiny_model_name
        print(f"\nLoading {model_name}...")
        tiny_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tiny_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)

        model_name = large_model_name
        print(f"Loading {model_name}...")
        distil_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #large_model = AutoModel.from_pretrained(model_name).to(device)
        large_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)
        global label_encoder
        dataset = list(zip(X, y))
        #label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Now create the dataset with encoded labels
        dataset = list(zip(X, y_encoded))
        train_data, test_data = split_training_data(
            dataset, lambda encoded: label_encoder.inverse_transform([encoded])[0]
        )

        # ---------------------------------------------------------------
        # Collate fn that keeps raw *texts* so we can avoid decode→encode
        # later during evaluation / fallback inference.
        # ---------------------------------------------------------------
        def make_collate_fn(tok):
            def _fn(batch):
                texts = [item[0] for item in batch]
                labels_tensor = torch.tensor([item[1] for item in batch], dtype=torch.long)
                encodings = tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                return encodings, labels_tensor, texts
            return _fn

        train_loader = DataLoader(
            train_data,
            batch_size=10,
            shuffle=True,
            collate_fn=make_collate_fn(tiny_tokenizer)
        )

        test_loader = DataLoader(
            test_data,
            batch_size=10,
            shuffle=False,
            collate_fn=make_collate_fn(tiny_tokenizer)
        )

        # -----------------------------------------------------------------
        # Preserve the Tiny-BERT test loader before we overwrite *test_loader*
        # for Distil-BERT.  All Tiny-BERT analysis & threshold-tuning should
        # continue to use this cached version to avoid tokenizer mismatch.
        # -----------------------------------------------------------------
        tiny_test_loader = test_loader

        #batch_size = 64  # Increased batch size
        optimizer = AdamW(tiny_model.parameters(), lr=1e-4)  # Slightly higher learning rate
        num_epochs = 12
        
        from time import time
        print("Starting training...")
        tiny_model.train()
        best_ndcg = 0
        best_f1 = 0
        training_start_time = time()
        training_losses = []  # Store training loss for each epoch
        test_losses = []
        for epoch in range(num_epochs):
            epoch_start_time = time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")

            for batch_idx, (encodings, labels, _) in enumerate(progress_bar):
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = tiny_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            avg_train_loss = total_loss / len(train_loader)
            training_losses.append(avg_train_loss)  # Append training loss for the epoch

            # Evaluate after each epoch
            f1, ndcg, avg_test_loss = evaluate_model(tiny_model, test_loader, device, k_val)
            test_losses.append(avg_test_loss)
            epoch_time = time() - epoch_start_time
            print(f"Epoch {epoch + 1} Results:")
            print(f"F1 Score: {f1:.4f}")
            print(f"NDCG@3: {ndcg:.4f}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")

        # Save paths updated to use context-specific folders
        tiny_path = get_artifact_path(workflow_folderpath, ctx_name, "tinymodel.pth")
        save_model(tiny_model, tiny_tokenizer, tiny_path)
        total_training_time = time() - training_start_time


        train_loader = DataLoader(
            train_data,
            batch_size=10,
            shuffle=True,
            collate_fn=make_collate_fn(distil_tokenizer)
        )

        test_loader = DataLoader(
            test_data,
            batch_size=10,
            shuffle=False,
            collate_fn=make_collate_fn(distil_tokenizer)
        )

        optimizer = AdamW(large_model.parameters(), lr=5e-5)
        num_epochs = 5

        print("Started training distilBert...")
        large_model.train()
        best_ndcg = 0
        best_f1 = 0
        num_epochs=5
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")

            for batch_idx, (encodings, labels, _) in enumerate(progress_bar):
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = large_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            # Evaluate after each epoch
            f1, ndcg, avg_loss = evaluate_model(large_model, test_loader, device, k_val)
            print(f"Epoch {epoch + 1} Results:")
            print(f"F1 Score: {f1:.4f}")
            print(f"NDCG@3: {ndcg:.4f}")

        # Save paths updated to use context-specific folders
        large_path = get_artifact_path(workflow_folderpath, ctx_name, "largemodel.pth")
        save_model(large_model, distil_tokenizer, large_path)

        pipeline = ModelPipeline(
            tiny_model_path=tiny_path,  
            distil_model_path=large_path,
            confidence_threshold=0.65
        )
        
        # Save paths updated to use context-specific folders
        label_path = get_artifact_path(workflow_folderpath, ctx_name, "label_encoder.pkl")
        save_label_encoder(label_path)

        print("\nAnalyzing TinyBERT confidence patterns...")
        tiny_stats, tiny_confidences, tiny_predictions, tiny_labels, tiny_failed = analyze_model_confidence(tiny_model, tiny_test_loader, device, "TinyBERT")

        print("\nTinyBERT Confidence Statistics:")
        print("\nFalse Classifications:")
        if tiny_stats['failed']['min'] is not None:
            print(f"Minimum Confidence: {tiny_stats['failed']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if tiny_stats['failed']['max'] is not None:
            print(f"Maximum Confidence: {tiny_stats['failed']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if tiny_stats['failed']['mean'] is not None:
            print(f"Mean Confidence: {tiny_stats['failed']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if tiny_stats['failed']['median'] is not None:
            print(f"Median Confidence: {tiny_stats['failed']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nTrue Classifications:")
        if tiny_stats['successful']['min'] is not None:
            print(f"Minimum Confidence: {tiny_stats['successful']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if tiny_stats['successful']['max'] is not None:
            print(f"Maximum Confidence: {tiny_stats['successful']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if tiny_stats['successful']['mean'] is not None:
            print(f"Mean Confidence: {tiny_stats['successful']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if tiny_stats['successful']['median'] is not None:
            print(f"Median Confidence: {tiny_stats['successful']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nAnalyzing DistilBERT confidence patterns...")
        large_stats, large_confidences, large_predictions, large_labels, large_failed = analyze_model_confidence(large_model, tiny_test_loader, device, "DistilBERT")

        print("\nTinyBERT Confidence Statistics:")
        print("\nFalse Classifications:")
        if large_stats['failed']['min'] is not None:
            print(f"Minimum Confidence: {large_stats['failed']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if large_stats['failed']['max'] is not None:
            print(f"Maximum Confidence: {large_stats['failed']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if large_stats['failed']['mean'] is not None:
            print(f"Mean Confidence: {large_stats['failed']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if large_stats['failed']['median'] is not None:
            print(f"Median Confidence: {large_stats['failed']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nTrue Classifications:")
        if large_stats['successful']['min'] is not None:
            print(f"Minimum Confidence: {large_stats['successful']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if large_stats['successful']['max'] is not None:
            print(f"Maximum Confidence: {large_stats['successful']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if large_stats['successful']['mean'] is not None:
            print(f"Mean Confidence: {large_stats['successful']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if large_stats['successful']['median'] is not None:
            print(f"Median Confidence: {large_stats['successful']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nFinding optimal threshold...")
        best_result, all_results = find_optimal_threshold(tiny_stats, tiny_test_loader, pipeline)
        print("\nOptimal Threshold Results:")
        print(f"Threshold: {best_result['threshold']:.4f}")
        print(f"F1 Score: {best_result['f1']:.4f}")
        print(f"NDCG@3: {best_result['ndcg']:.4f}")
        print(f"DistilBERT Usage: {best_result['distil_usage']:.2f}%")

        pipeline.confidence_threshold = best_result['threshold']

        threshold = best_result['threshold']
        # Save paths updated to use context-specific folders
        threshold_path = get_artifact_path(workflow_folderpath, ctx_name, "threshold.json")
        with open(threshold_path, 'w') as f:
            json.dump({'confidence_threshold': threshold}, f)

        f1, ndcg, stats = pipeline.evaluate(tiny_test_loader)

        print("\nEvaluation Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"NDCG@3: {ndcg:.4f}")
        print("\nModel Usage Statistics:")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"DistilBERT Usage: {stats['distil_percentage']:.2f}%")
        print(f"TinyBERT Usage: {stats['tiny_percentage']:.2f}%")


        
        # R3: one writer for both ambiguity thresholds, so the separation invariant is
        # established where the files are produced rather than asserted afterwards.
        has_wildcard_class = bool(escalation.included and escalation.rows)
        tiny_ambiguous_threshold, large_ambiguous_threshold = write_ambiguity_thresholds(
            os.path.dirname(threshold_path), threshold, tiny_stats, large_stats,
            has_wildcard_class=has_wildcard_class,
        )
        print(
            f"Ambiguity thresholds (R3): tier={threshold:.4f} "
            f"tiny_ambiguous={tiny_ambiguous_threshold:.4f} "
            f"large_ambiguous={large_ambiguous_threshold:.4f}"
        )

    
        text = "list commands"
        result = predict_single_sentence(pipeline, text, label_path)
        print(f"Predicted label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Used DistilBERT: {'Yes' if result['used_distil'] else 'No'}")

        # ------------------------------------------------------------------
        # R1a/R1b: score the reserved personas through the REAL runtime path.
        # CommandRouter.predict is what intent detection actually calls, thresholds
        # and all, so this measures what a user would experience rather than what the
        # raw classifier head emits. Every artifact it needs was written above; the
        # model directory is taken from the threshold path so this keeps working
        # whether or not artifacts are being routed into a version.
        # ------------------------------------------------------------------
        report = heldout_evaluation.HeldoutReport(
            context=ctx_name,
            in_distribution_f1=f1,
            seed=seed,
            heldout_personas=heldout_personas,
            notes=split_notes,
            large_ambiguity={
                "failed_rows": large_failed_row_count(large_stats),
                "rule": large_ambiguous_threshold_and_rule(large_stats)[1],
                "threshold": large_ambiguous_threshold,
                "has_wildcard_class": has_wildcard_class,
                "at_floor_without_wildcard": at_floor_without_wildcard(
                    large_ambiguous_threshold, has_wildcard_class),
                "wildcard_seeded_from_built_in_vague_replies": bool(
                    escalation.not_here_samples
                    and BUILT_IN_VAGUE_SOURCE in escalation.not_here_samples),
                "built_in_vague_weight": (
                    BUILT_IN_VAGUE_WEIGHT
                    if escalation.not_here_samples
                    and BUILT_IN_VAGUE_SOURCE in escalation.not_here_samples else None),
            },
        )
        try:
            has_context_benchmark = any(
                case.context == ctx_name
                and case.kind in {"routing", "escalation"}
                for case in benchmark_cases
            )
            predict_labels = None
            if heldout_records or has_context_benchmark:
                router = CommandRouter(os.path.dirname(threshold_path))

                def predict_labels(utterance: str, _router=router) -> list[str]:
                    """Adapt `CommandRouter.predict` to the scorer's contract.

                    `predict` returns either a one-element list holding a numpy string or
                    the raw numpy top-k array. The scorer needs a plain `list[str]`: a
                    numpy array raises "truth value of an array is ambiguous" the moment
                    anything tests it for emptiness.
                    """
                    return [str(label) for label in _router.predict(utterance)]

            # `kind` is required. Omitting it raised a TypeError that the guard
            # below turned into a note on the report, so escalation silently
            # never scored while routing kept working -- the failure looked like
            # "this workflow has no escalation cases" (bd fix-588).
            _score_heldout_context(
                report, heldout_records, benchmark_cases, predict_labels,
                split_for_calibration=heldout_split)
        except Exception as exc:
            # A scoring failure must never destroy a completed training run: the
            # models are already on disk and usable. Record it and move on.
            report.notes.append(f"held-out evaluation failed: {exc}")
            logger.error(
                f"Held-out evaluation failed for context '{ctx_name}': {exc}")
        heldout_reports.append(report)

    # End of context loop

    if workflow_layout_trainer is not None:
        if stop_before_fit:
            workflow_layout_trainer.rows()  # the assembly, for a caller observing it
            return used_not_here_samples
        heldout_reports = workflow_layout_trainer.finish()

    if stop_before_fit:
        return used_not_here_samples

    if heldout_reports:
        print(heldout_evaluation.format_report(heldout_reports))
        _write_heldout_report(workflow_folderpath, heldout_reports)
    return used_not_here_samples


def _write_heldout_report(
    workflow_folderpath: str,
    heldout_reports: list[heldout_evaluation.HeldoutReport],
) -> Optional[str]:
    """Persist the held-out evaluation report, saying so loudly when it cannot.

    Returns the path written, or None on failure.

    Not fatal — the models are already on disk and usable, and a failed report write is
    not a reason to discard a completed run. Not silent either: a failure leaves
    `heldout_evaluation.json` stale or absent while the run reports success, and
    `selective_training.merge_heldout_evaluation` then merges this run's contexts into a
    baseline describing different artifacts, while `capture_heldout_evaluation` returns
    None on the next run. Both consequences are invisible from the training output, which
    is why the write cannot simply be suppressed.
    """
    try:
        report_path = heldout_evaluation.write_report(
            workflow_folderpath, heldout_reports)
    except OSError as exc:
        message = (
            f"Could not write the held-out evaluation report for "
            f"{workflow_folderpath}: {type(exc).__name__}: {exc}. The scores printed "
            f"above are NOT on disk; heldout_evaluation.json is stale or absent, so the "
            f"next selective run merges against the wrong baseline."
        )
        logger.error(message)
        print(f"WARNING: {message}", flush=True)
        return None
    print(f"Held-out evaluation report: {report_path}")
    return report_path