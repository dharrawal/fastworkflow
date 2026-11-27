"""Which artifact layout a workflow's intent models are trained into, and how to recognise it.

Three layouts exist:

* ``embedding`` (the default for an application workflow): a frozen sentence encoder that is never
  trained and ONE logistic-regression head over every command of the workflow, in the version's
  ``workflow/`` folder, read for a context through that context's output mask
  (`train.embedding_head`). Each context folder holds two small files only.
* ``workflow``: ONE fine-tuned large classifier for the whole workflow in the same folder, read
  through the same masks (`train.workflow_model`). Each context folder holds the same two files.
* ``per_context``: one tiny and one large fine-tuned classifier for every context, each in its
  own folder of the artifact version. Still trained and read when asked for by name; no longer
  what the framework's own workflows get (`FRAMEWORK_LAYOUT`).

``INTENT_MODEL_LAYOUT`` in the workflow env file chooses; `DEFAULT_LAYOUT` is what an unset
setting means. The setting is read when training starts and nowhere else. The runtime never reads
it: it reads whatever layout the published version has, which is recognised from the files in a
context folder. That is what lets a workflow be switched by retraining alone.

This module imports nothing heavy (no torch, no trainer), because the startup gates and the
artifact versioning code read it.
"""

from __future__ import annotations

import json
import os
from typing import Optional

SETTING = "INTENT_MODEL_LAYOUT"
PER_CONTEXT = "per_context"
WORKFLOW = "workflow"
EMBEDDING = "embedding"
LAYOUTS = (PER_CONTEXT, WORKFLOW, EMBEDDING)
#: THE default: the layout `fastworkflow train` writes for an application workflow whose env file
#: does not set ``INTENT_MODEL_LAYOUT``. Changing the default is changing this one name.
DEFAULT_LAYOUT = EMBEDDING
#: The layout of the framework's own workflows (the command metadata extraction workflow and the
#: plan metadata extraction workflow), whatever the setting or the default says. It is the embedding layout:
#: a per-context retrain, which any application's `fastworkflow train` triggers when the trainer's
#: source changes, did not reproduce the previous head (17 of 43 recorded review replies changed
#: their answer), while the embedding head is deterministic, trains in seconds and is 21 KB.
#: Their embedding settings are the defaults, not the application's (`embedding_settings`).
FRAMEWORK_LAYOUT = EMBEDDING
#: The layouts with one model for the workflow: a context folder holds a pointer to it.
ONE_MODEL_LAYOUTS = (WORKFLOW, EMBEDDING)

#: Manifest key (and provenance run-metadata key) naming a version's layout. Written for every
#: layout but ``per_context``, so a ``per_context`` manifest is the file it always was; a manifest
#: without the key is ``per_context``.
MANIFEST_KEY = "intent_model_layout"

#: The folder of an artifact version that holds the workflow's one model. Must match
#: `artifact_versioning.WORKFLOW_MODEL_DIRNAME` (that module names it too, to stay free of
#: imports; `tests/test_intent_model_layout.py` asserts the two agree).
WORKFLOW_MODEL_DIRNAME = "workflow"
LARGE_MODEL_DIRNAME = "largemodel.pth"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
SCOPES_FILENAME = "scopes.json"
#: The ``embedding`` layout's head: ``coef`` and ``intercept`` of the logistic regression.
HEAD_FILENAME = "head.npz"
#: The ``embedding`` layout's row store, beside the head: the vectors the fit computed, which a refit with learned rows
#: (`train.head_refit`) reads instead of embedding the base rows again. Not required: a head without it is complete and
#: is served as before, it just cannot be refitted until its next train.
ROWS_FILENAME = "rows.npz"
#: What the ``workflow/`` folder must hold in the ``workflow`` layout. ``largemodel.pth`` is a
#: directory (save_pretrained).
REQUIRED_WORKFLOW_MODEL_ARTIFACTS: tuple[str, ...] = (
    LARGE_MODEL_DIRNAME,
    LABEL_ENCODER_FILENAME,
    SCOPES_FILENAME,
)
#: What it must hold in the ``embedding`` layout. The encoder is not stored: it is never trained,
#: and ``scopes.json`` names it.
REQUIRED_EMBEDDING_MODEL_ARTIFACTS: tuple[str, ...] = (HEAD_FILENAME, SCOPES_FILENAME)
REQUIRED_MODEL_ARTIFACTS: dict[str, tuple[str, ...]] = {
    WORKFLOW: REQUIRED_WORKFLOW_MODEL_ARTIFACTS,
    EMBEDDING: REQUIRED_EMBEDDING_MODEL_ARTIFACTS,
}

THRESHOLD_FILENAME = "threshold.json"
#: The marker in a context folder that says "my model is the workflow's one model". Its
#: ``layout`` field says which kind of model that is (``workflow`` or ``embedding``).
WORKFLOW_SCOPE_FILENAME = "workflow_scope.json"
#: What each context folder must hold in a one-model layout.
REQUIRED_WORKFLOW_CONTEXT_ARTIFACTS: tuple[str, ...] = (
    THRESHOLD_FILENAME,
    WORKFLOW_SCOPE_FILENAME,
)

#: The ``confidence_threshold`` written to a context's ``threshold.json`` in a one-model
#: layout. That file is what every "is this context trained" gate looks for, so it is still
#: written. In the per-context layout its number is the confidence at or above which the tiny
#: model answers; here there is no tiny model, and no probability reaches 2, so read by the old
#: rule the number still says the true thing: the one model answers every time. The
#: thresholds that decide anything are in ``workflow/scopes.json``.
WORKFLOW_LAYOUT_TIER_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# The ``embedding`` layout's settings (workflow env file, read by `fastworkflow train` only)
# ---------------------------------------------------------------------------

ENCODER_SETTING = "INTENT_EMBEDDING_MODEL"
#: THE default encoder: frozen, never trained, one shared download (about 418 MB) and nothing of it
#: stored per workflow. Chosen in ``docs/local_intent_model_levers.md`` (Part 2).
DEFAULT_ENCODER = "intfloat/e5-base-v2"
#: The encoder of the framework's own workflows (plan, command metadata extraction), pinned: they follow the
#: new rules (no pass-up, the fixed commit threshold, description rows, seeds weighted) but NOT the new default
#: encoder. On a temp copy of the plan metadata extraction workflow with `intfloat/e5-base-v2` the held-out reply corpus
#: scored 30 of 36 top answers with `look_up` at 2 of 6, under the test bar's 0.50 per command, and a vague
#: reply ("hmm") answered `confirm` outright; with this encoder 31 of 36, `look_up` 3 of 6 and every vague
#: reply the wildcard class (bd fix-o5wi.9). They move to the default encoder when that gate passes.
FRAMEWORK_ENCODER = "BAAI/bge-small-en-v1.5"
POOLING_SETTING = "INTENT_EMBEDDING_POOLING"
POOLING_CLS = "cls"
POOLING_MEAN = "mean"
#: The probability the best command of a context must reach for the head to answer with that one
#: command. A conservative constant, not a calibration: the experiment's 1%-wrong point on the bundled
#: examples was 0.65 and a large workflow reached 1% at 0.50, so 0.65 is the safe side. Asking the
#: cross-fit of the training rows for it gave 5 to 8% wrong on real requests at a 0.5% target, so the
#: trainer no longer derives it from them.
DEFAULT_COMMIT_THRESHOLD = 0.65
COMMIT_THRESHOLD_SETTING = "INTENT_COMMIT_THRESHOLD"
#: Settings that existed and were removed with the pass-up margin and the cross-fit threshold. A value
#: left in an env file is ignored with one warning (`warn_if_removed_settings`).
REMOVED_SETTINGS = ("INTENT_WRONG_COMMIT_TARGET", "INTENT_FALSE_ESCALATION_TARGET")

_warned_values: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned_values:
        _warned_values.add(message)
        from fastworkflow.utils.logging import logger

        logger.warning(message)


def parse_layout(raw: object) -> str:
    """A layout name from a setting's text. Empty or missing is `DEFAULT_LAYOUT`; anything else
    that is not a layout logs one warning per distinct value and is the default too."""
    value = str(raw or "").strip().lower()
    if not value:
        return DEFAULT_LAYOUT
    if value in LAYOUTS:
        return value
    _warn_once(f"{SETTING}={raw!r} is not one of {', '.join(LAYOUTS)}: using {DEFAULT_LAYOUT!r}.")
    return DEFAULT_LAYOUT


def layout_setting() -> str:
    """What ``INTENT_MODEL_LAYOUT`` asks for, read from the workflow env like every setting."""
    import fastworkflow

    return parse_layout(fastworkflow.get_env_var(SETTING, default=DEFAULT_LAYOUT))


def default_pooling(encoder: str) -> str:
    """The pooling an encoder is read with when the setting does not say: the CLS vector for the
    ``BAAI/bge-*`` family, the attention-masked mean for everything else (which is what the
    ``sentence-transformers/*`` encoders are trained with)."""
    return POOLING_CLS if str(encoder).lower().startswith("baai/bge-") else POOLING_MEAN


def parse_pooling(raw: object, encoder: str) -> str:
    """``cls`` or ``mean``. Empty is the encoder's default; another value warns once and is that too."""
    value = str(raw or "").strip().lower()
    if value in (POOLING_CLS, POOLING_MEAN):
        return value
    if value:
        _warn_once(f"{POOLING_SETTING}={raw!r} is not {POOLING_CLS!r} or {POOLING_MEAN!r}: "
                   f"using {default_pooling(encoder)!r}.")
    return default_pooling(encoder)


def default_prefix(encoder: str) -> str:
    """The text put before every text the encoder reads: ``query: `` for the ``intfloat/e5-*`` family (its
    documented prefix for symmetric matching, used in the experiment), nothing for any other encoder."""
    return "query: " if str(encoder).lower().startswith("intfloat/e5-") else ""


def parse_threshold(raw: object, default: float, setting: str) -> float:
    """A commit threshold: a number above 0 and at most 1. Anything else warns once and is the default."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if 0.0 < value <= 1.0:
        return value
    _warn_once(f"{setting}={raw!r} is not a number above 0 and at most 1: using {default}.")
    return default


def calibrated_commit_threshold(workflow_folderpath: str) -> Optional[float]:
    """THE place a commit threshold calibrated on recorded real requests is read from (fix-o5wi.6): the value
    `fastworkflow learn` stored for the workflow's current head lineage (`intent_learning`), or None. It can only be
    lower than the default (the evidence is censored: see `intent_learning.calibrate`)."""
    from fastworkflow.intent import intent_learning

    return intent_learning.calibrated_threshold(workflow_folderpath)


def workflow_and_version_of(model_dir: str) -> tuple[Optional[str], Optional[str]]:
    """``(workflow folder, version id)`` of a head folder ``<workflow>/___command_info/versions/<id>/workflow``;
    ``(None, None)`` for a folder that is not laid out so (a test's scratch head)."""
    real = os.path.realpath(model_dir)
    version_folder = os.path.dirname(real)
    versions = os.path.dirname(version_folder)
    info = os.path.dirname(versions)
    if os.path.basename(versions) != "versions" or os.path.basename(info) != "___command_info":
        return None, None
    return os.path.dirname(info), os.path.basename(version_folder)


def threshold_in_force(model_dir: str, trained: float) -> tuple[float, str]:
    """``(the commit threshold a head is read with, why)``. The threshold the head was trained with, unless the
    workflow has a calibrated value for this head's lineage (`calibrated_commit_threshold`) and
    ``INTENT_COMMIT_THRESHOLD`` is not set: an explicit setting always wins. The reason is ``setting``, ``calibrated``
    or ``trained``."""
    import fastworkflow
    from fastworkflow.intent import intent_learning

    if str(fastworkflow.get_env_var(COMMIT_THRESHOLD_SETTING, default="") or "").strip():
        return trained, "setting"
    workflow, version = workflow_and_version_of(model_dir)
    calibrated = intent_learning.calibrated_threshold(workflow, version) if workflow else None
    if calibrated is not None and calibrated < trained:
        return float(calibrated), "calibrated"
    return trained, "trained"


def commit_threshold(workflow_folderpath: Optional[str] = None) -> float:
    """The probability at or above which the head answers with one command: the workflow env's
    ``INTENT_COMMIT_THRESHOLD`` when set, else a calibrated value when there is one
    (`calibrated_commit_threshold`), else `DEFAULT_COMMIT_THRESHOLD`. Written into the head at training."""
    import fastworkflow

    raw = fastworkflow.get_env_var(COMMIT_THRESHOLD_SETTING, default="")
    if str(raw or "").strip():
        return parse_threshold(raw, DEFAULT_COMMIT_THRESHOLD, COMMIT_THRESHOLD_SETTING)
    calibrated = calibrated_commit_threshold(workflow_folderpath) if workflow_folderpath else None
    return float(calibrated) if calibrated is not None else DEFAULT_COMMIT_THRESHOLD


def warn_if_removed_settings() -> None:
    """One warning per setting still present in the workflow env: the pass-up margin and the cross-fit
    threshold were removed, so these are ignored."""
    import fastworkflow

    for name in REMOVED_SETTINGS:
        try:
            raw = str(fastworkflow.get_env_var(name, default="") or "").strip()
        except Exception:  # noqa: BLE001 - an env that cannot be read has no stale setting to report
            continue
        if raw:
            _warn_once(
                f"{name}={raw!r} is ignored: the setting was removed. The local model no longer passes a "
                f"request up on its own and its commit threshold is {COMMIT_THRESHOLD_SETTING} "
                f"(default {DEFAULT_COMMIT_THRESHOLD}). Remove the line from the workflow env file.")


def embedding_settings(workflow_folderpath: Optional[str] = None) -> dict:
    """The ``embedding`` layout's settings as they stand in the workflow env: ``encoder``, ``pooling``,
    ``prefix`` and ``commit_threshold``. They decide what a run writes, so they are global inputs of the
    training signature.

    For the framework's own workflows (*workflow_folderpath* given and `is_framework_workflow`)
    they are `FRAMEWORK_ENCODER` and the default threshold whatever the env says: those workflows are shared by every application
    and are retrained from any application's ``fastworkflow train``, so one application's
    encoder choice must not change them."""
    import fastworkflow

    warn_if_removed_settings()
    if workflow_folderpath is not None and is_framework_workflow(workflow_folderpath):
        return {
            "encoder": FRAMEWORK_ENCODER,
            "pooling": default_pooling(FRAMEWORK_ENCODER),
            "prefix": default_prefix(FRAMEWORK_ENCODER),
            "commit_threshold": DEFAULT_COMMIT_THRESHOLD,
        }

    encoder = str(fastworkflow.get_env_var(ENCODER_SETTING, default=DEFAULT_ENCODER) or "").strip()
    encoder = encoder or DEFAULT_ENCODER
    return {
        "encoder": encoder,
        "pooling": parse_pooling(fastworkflow.get_env_var(POOLING_SETTING, default=""), encoder),
        "prefix": default_prefix(encoder),
        "commit_threshold": commit_threshold(workflow_folderpath),
    }


def is_framework_workflow(workflow_folderpath: str) -> bool:
    """Whether *workflow_folderpath* is one of the framework's own workflows: the command
    metadata extraction workflow or the plan metadata extraction workflow."""
    from fastworkflow.model_pipeline_training import (
        is_internal_cme_folder,
        is_plan_metadata_extraction_folder,
    )

    return bool(
        is_internal_cme_folder(workflow_folderpath)
        or is_plan_metadata_extraction_folder(workflow_folderpath)
    )


def layout_for_workflow(workflow_folderpath: str) -> str:
    """The layout `fastworkflow train` writes for *workflow_folderpath*.

    The setting, except for the framework's own workflows (the command metadata extraction
    workflow and the plan metadata extraction workflow). Those are
    trained with a construction of their own and are shared by every application, so an
    application's setting must not change them: they get `FRAMEWORK_LAYOUT`.
    """
    setting = layout_setting()
    if setting == FRAMEWORK_LAYOUT:
        return setting
    return FRAMEWORK_LAYOUT if is_framework_workflow(workflow_folderpath) else setting


def is_workflow_scope_folder(context_folder: str) -> bool:
    """Whether *context_folder* is a context of a one-model layout (``workflow`` or ``embedding``)."""
    return os.path.isfile(os.path.join(context_folder, WORKFLOW_SCOPE_FILENAME))


def read_scope_marker(context_folder: str) -> dict:
    """The content of a context folder's ``workflow_scope.json``."""
    with open(os.path.join(context_folder, WORKFLOW_SCOPE_FILENAME), "r", encoding="utf-8") as f:
        marker = json.load(f)
    if not isinstance(marker, dict) or marker.get("layout") not in ONE_MODEL_LAYOUTS:
        raise ValueError(
            f"{os.path.join(context_folder, WORKFLOW_SCOPE_FILENAME)} does not describe a "
            f"context of the {' or '.join(repr(name) for name in ONE_MODEL_LAYOUTS)} layout."
        )
    return marker


def context_layout(context_folder: str) -> str:
    """The layout a trained context folder is in, read from its files: the marker's ``layout``
    for a one-model layout, else ``per_context``."""
    if not is_workflow_scope_folder(context_folder):
        return PER_CONTEXT
    return read_scope_marker(_real_context_folder(context_folder))["layout"]


def _real_context_folder(context_folder: str) -> str:
    """The folder inside the artifact version that *context_folder* stands for.

    A context entry of ``___command_info`` is a symlink into the version, and ``realpath``
    follows it. Where symlinks are not available the entry is a folder of hard links with a
    marker naming the version folder it mirrors; the model sits beside that one, not beside
    the mirror.
    """
    from fastworkflow.train.artifact_versioning import COMPAT_MARKER_FILENAME

    real = os.path.realpath(context_folder)
    try:
        with open(os.path.join(real, COMPAT_MARKER_FILENAME), "r", encoding="utf-8") as f:
            source = json.load(f).get("source")
    except (OSError, ValueError, AttributeError):
        return real
    return os.path.realpath(source) if isinstance(source, str) and source else real


def workflow_model_dir(context_folder: str) -> str:
    """The real path of the workflow model folder a context of a one-model layout reads.

    Always a path inside one artifact version, so a newly published version is a different
    path and its model is loaded fresh.
    """
    real = _real_context_folder(context_folder)
    relative = read_scope_marker(real).get("model") or os.path.join("..", WORKFLOW_MODEL_DIRNAME)
    return os.path.realpath(os.path.join(real, relative))


def workflow_model_complete(model_dir: str, layout: str = WORKFLOW) -> bool:
    """Whether a ``workflow/`` folder holds the entries of its layout."""
    return all(
        os.path.exists(os.path.join(model_dir, name)) for name in REQUIRED_MODEL_ARTIFACTS[layout]
    )


def workflow_context_complete(context_folder: str) -> bool:
    """Whether a context of a one-model layout can be read: its two small files are there and
    the model they point to is complete for the layout the marker names."""
    if not all(
        os.path.exists(os.path.join(context_folder, name))
        for name in REQUIRED_WORKFLOW_CONTEXT_ARTIFACTS
    ):
        return False
    try:
        return workflow_model_complete(
            workflow_model_dir(context_folder), context_layout(context_folder))
    except (OSError, ValueError, KeyError):
        return False


def workflow_head_present(context_folder: str) -> bool:
    """Whether *context_folder* is a trained context of a one-model layout. The checks that ask
    "does this context have a trained head" accept this beside the per-context files."""
    return is_workflow_scope_folder(context_folder) and workflow_context_complete(context_folder)


def write_context_files(context_folder: str, context_name: str, layout: str = WORKFLOW) -> None:
    """Write the two small files of one context of a one-model layout (*layout*)."""
    if layout not in ONE_MODEL_LAYOUTS:
        raise ValueError(f"{layout!r} is not a layout with one model for the workflow")
    os.makedirs(context_folder, exist_ok=True)
    model = "large model" if layout == WORKFLOW else "embedding head"
    with open(os.path.join(context_folder, THRESHOLD_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "confidence_threshold": WORKFLOW_LAYOUT_TIER_THRESHOLD,
                "layout": layout,
                "note": (
                    f"Not a tuned number. This context is answered by the workflow's one "
                    f"{model} (see workflow_scope.json); its thresholds are in the "
                    f"model folder's scopes.json. No probability reaches this value, which "
                    f"is how the old reading of this file says the one model always answers."
                ),
            },
            f,
        )
    with open(os.path.join(context_folder, WORKFLOW_SCOPE_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "layout": layout,
                "model": f"../{WORKFLOW_MODEL_DIRNAME}",
                "context": context_name,
            },
            f,
        )


def version_layout(version_folder: str) -> Optional[str]:
    """The layout of the artifact version at *version_folder*, read from its folders: None when
    it holds no trained context at all."""
    try:
        entries = sorted(os.scandir(version_folder), key=lambda entry: entry.name)
    except OSError:
        return None
    found: Optional[str] = None
    for entry in entries:
        if not entry.is_dir():
            continue
        if is_workflow_scope_folder(entry.path):
            try:
                return context_layout(entry.path)
            except (OSError, ValueError):
                return WORKFLOW
        if os.path.exists(os.path.join(entry.path, THRESHOLD_FILENAME)):
            found = PER_CONTEXT
    return found
