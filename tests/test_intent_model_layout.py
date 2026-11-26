"""The intent model layouts (``INTENT_MODEL_LAYOUT``: ``embedding``, the default, ``workflow`` and
``per_context``): the setting, the artifact checks for every layout, the planner's rules, and the
runtime reader of the ``workflow`` layout. The embedding head itself (its decision, training rows,
fit and reader) is in ``tests/test_embedding_head.py``.

Nothing here trains. The folder checks run on hand-built version folders with placeholder files,
as ``tests/test_not_here_selective.py`` builds them. The planner runs on a copy of a bundled
example with a published placeholder version. The runtime reader is exercised on REAL weights
when a model trained by the single-model experiment is on this machine (it lives outside the
repository, so those tests skip with the path they looked for when it is absent); the weights
are linked into a temporary version folder, never copied.

The end-to-end run (`fastworkflow train` in the layout, then the runtime) is the slow test in
``tests/test_intent_model_layout_training.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import intent_detector
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from fastworkflow.train import artifact_versioning
from fastworkflow.train import intent_model_layout as layout
from fastworkflow.train import selective_training as st
from fastworkflow.train import utterance_cache

REPO = Path(__file__).resolve().parent.parent
MESSAGING = REPO / "fastworkflow" / "examples" / "messaging_app_4"
LIBRARY = REPO / "fastworkflow" / "examples" / "library"
SEED = 42

#: A library model trained by the single-model experiment (arm S of the second
#: pre-registration): tiny and large checkpoints, the label encoder and scopes.json. About
#: 270 MB, outside the repository; tests that need real weights skip when it is not there.
EXPERIMENT_MODEL = Path(os.environ.get(
    "FW_TEST_WORKFLOW_MODEL_DIR",
    "/tmp/claude-1000/-home-drawal-rl-fastworkflow/fdae6525-4d9b-48b0-b46d-789659918e51/"
    "scratchpad/single_model/library/M2/42",
))


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _with_setting(monkeypatch, value):
    env = {name: v for name, v in fastworkflow._env_vars.items() if name != layout.SETTING}
    if value is not None:
        env[layout.SETTING] = value
    monkeypatch.delenv(layout.SETTING, raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars", env)


# ---------------------------------------------------------------- the setting

def test_the_default_layout_is_one_constant_and_it_is_the_embedding_layout(monkeypatch):
    _with_setting(monkeypatch, None)
    assert layout.DEFAULT_LAYOUT == layout.EMBEDDING
    assert layout.layout_setting() == layout.DEFAULT_LAYOUT
    assert layout.parse_layout(None) == layout.DEFAULT_LAYOUT
    assert layout.LAYOUTS == (layout.PER_CONTEXT, layout.WORKFLOW, layout.EMBEDDING)
    # Moving the constant is all it takes to change the default.
    monkeypatch.setattr(layout, "DEFAULT_LAYOUT", layout.PER_CONTEXT)
    assert layout.layout_setting() == layout.PER_CONTEXT and layout.parse_layout("") == layout.PER_CONTEXT


@pytest.mark.parametrize("raw, expected", [
    ("workflow", layout.WORKFLOW),
    ("  Workflow ", layout.WORKFLOW),
    ("per_context", layout.PER_CONTEXT),
    ("PER_CONTEXT", layout.PER_CONTEXT),
    ("embedding", layout.EMBEDDING),
    (" Embedding", layout.EMBEDDING),
    ("", layout.DEFAULT_LAYOUT),
])
def test_the_setting_is_read_from_the_workflow_env(monkeypatch, raw, expected):
    _with_setting(monkeypatch, raw)
    assert layout.layout_setting() == expected


def test_an_invalid_value_warns_once_and_means_the_default(monkeypatch):
    from fastworkflow.utils.logging import logger

    warnings: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    monkeypatch.setattr(layout, "_warned_values", set())
    _with_setting(monkeypatch, "one_model")

    assert layout.layout_setting() == layout.DEFAULT_LAYOUT
    assert layout.layout_setting() == layout.DEFAULT_LAYOUT
    assert len(warnings) == 1
    assert "one_model" in warnings[0] and layout.SETTING in warnings[0]


def test_the_frameworks_own_workflows_are_trained_in_the_framework_layout(monkeypatch, tmp_path):
    """An application's setting does not move the shared internal workflows: they are always in
    `FRAMEWORK_LAYOUT`, which is the embedding layout (a per-context retrain of them did not
    reproduce the previous head)."""
    assert layout.FRAMEWORK_LAYOUT == layout.EMBEDDING
    internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    plan = fastworkflow.get_internal_workflow_path("plan_metadata_extraction")
    package = fastworkflow.get_fastworkflow_package_path()

    for setting, application in (("workflow", layout.WORKFLOW), ("per_context", layout.PER_CONTEXT),
                                 (None, layout.EMBEDDING), ("embedding", layout.EMBEDDING)):
        _with_setting(monkeypatch, setting)
        assert layout.layout_for_workflow(internal) == layout.FRAMEWORK_LAYOUT
        assert layout.layout_for_workflow(plan) == layout.FRAMEWORK_LAYOUT
        # the package that holds them is no workflow (fix-bl97.2): it is read as any other folder
        assert layout.layout_for_workflow(package) == application
        assert layout.layout_for_workflow(str(tmp_path)) == application


def test_the_frameworks_embedding_settings_ignore_the_applications(monkeypatch, tmp_path):
    """The framework's heads are retrained from any application's train: one application's encoder
    or threshold must not change them. An application's own settings are read as before."""
    monkeypatch.setitem(fastworkflow._env_vars, layout.ENCODER_SETTING, "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setitem(fastworkflow._env_vars, layout.COMMIT_THRESHOLD_SETTING, "0.8")
    framework = layout.embedding_settings(fastworkflow.get_internal_workflow_path("plan_metadata_extraction"))
    # The framework's workflows keep their pinned encoder: the default one failed their reply gate.
    assert framework == {"encoder": layout.FRAMEWORK_ENCODER, "pooling": "cls", "prefix": "", "commit_threshold": 0.65}
    assert layout.FRAMEWORK_ENCODER == "BAAI/bge-small-en-v1.5" != layout.DEFAULT_ENCODER
    application = layout.embedding_settings(str(tmp_path))
    assert application["encoder"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert application["prefix"] == "" and application["commit_threshold"] == 0.8
    assert layout.embedding_settings() == application


def test_the_embedding_settings_and_their_defaults(monkeypatch):
    _with_setting(monkeypatch, None)
    env = {name: value for name, value in fastworkflow._env_vars.items() if not name.startswith("INTENT_")}
    monkeypatch.setattr(fastworkflow, "_env_vars", env)
    for name in (layout.ENCODER_SETTING, layout.POOLING_SETTING, layout.COMMIT_THRESHOLD_SETTING,
                 *layout.REMOVED_SETTINGS):
        monkeypatch.delenv(name, raising=False)
    assert layout.embedding_settings() == {
        "encoder": "intfloat/e5-base-v2", "pooling": "mean", "prefix": "query: ", "commit_threshold": 0.65}
    assert layout.DEFAULT_COMMIT_THRESHOLD == 0.65 and layout.calibrated_commit_threshold("anywhere") is None

    # Pooling and the prefix follow the encoder family unless the setting says otherwise.
    assert layout.default_pooling("BAAI/bge-base-en-v1.5") == "cls"
    assert layout.default_pooling("sentence-transformers/all-MiniLM-L12-v2") == "mean"
    assert layout.default_pooling("someone/another-encoder") == "mean"
    assert layout.default_prefix("intfloat/e5-small-v2") == "query: "
    assert layout.default_prefix("BAAI/bge-base-en-v1.5") == ""
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        **env, layout.ENCODER_SETTING: "sentence-transformers/all-MiniLM-L12-v2",
        layout.COMMIT_THRESHOLD_SETTING: "0.8"})
    assert layout.embedding_settings() == {
        "encoder": "sentence-transformers/all-MiniLM-L12-v2", "pooling": "mean", "prefix": "",
        "commit_threshold": 0.8}
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        **env, layout.ENCODER_SETTING: "someone/another-encoder", layout.POOLING_SETTING: "CLS"})
    assert layout.embedding_settings()["pooling"] == "cls"


def test_a_calibrated_threshold_is_read_from_one_place_and_the_setting_wins(monkeypatch):
    """`calibrated_commit_threshold` is where a value calibrated on recorded real requests will be read
    (fix-o5wi.6). Nothing calibrates today, so it is None; a value there beats the default, and the
    setting beats both."""
    env = {name: value for name, value in fastworkflow._env_vars.items() if not name.startswith("INTENT_")}
    monkeypatch.setattr(fastworkflow, "_env_vars", env)
    monkeypatch.delenv(layout.COMMIT_THRESHOLD_SETTING, raising=False)
    assert layout.commit_threshold("some/workflow") == 0.65
    monkeypatch.setattr(layout, "calibrated_commit_threshold", lambda workflow: 0.55)
    assert layout.commit_threshold("some/workflow") == 0.55
    monkeypatch.setattr(fastworkflow, "_env_vars", {**env, layout.COMMIT_THRESHOLD_SETTING: "0.7"})
    assert layout.commit_threshold("some/workflow") == 0.7


def test_an_invalid_embedding_setting_warns_once_and_means_the_default(monkeypatch):
    from fastworkflow.utils.logging import logger

    warnings: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    monkeypatch.setattr(layout, "_warned_values", set())
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        **fastworkflow._env_vars, layout.POOLING_SETTING: "max", layout.COMMIT_THRESHOLD_SETTING: "1.5"})
    for _ in range(2):
        settings = layout.embedding_settings()
        assert settings["pooling"] == layout.default_pooling(settings["encoder"])
        assert settings["commit_threshold"] == 0.65
    assert len(warnings) == 2
    monkeypatch.setattr(fastworkflow, "_env_vars", {**fastworkflow._env_vars, layout.COMMIT_THRESHOLD_SETTING: "0"})
    assert layout.embedding_settings()["commit_threshold"] == 0.65


def test_the_removed_settings_are_ignored_with_one_warning(monkeypatch):
    """The pass-up margin and the cross-fit threshold were removed with their targets: a value left in
    an env file does nothing but warn, once per setting."""
    from fastworkflow.utils.logging import logger

    warnings: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    monkeypatch.setattr(layout, "_warned_values", set())
    clean = {name: value for name, value in fastworkflow._env_vars.items() if not name.startswith("INTENT_")}
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        **clean, "INTENT_WRONG_COMMIT_TARGET": "0.02", "INTENT_FALSE_ESCALATION_TARGET": "0.01"})
    for _ in range(2):
        assert layout.embedding_settings() == {
            "encoder": "intfloat/e5-base-v2", "pooling": "mean", "prefix": "query: ", "commit_threshold": 0.65}
    assert len(warnings) == 2
    assert all("is ignored: the setting was removed" in message for message in warnings)
    assert {name for name in layout.REMOVED_SETTINGS if any(name in message for message in warnings)} == set(
        layout.REMOVED_SETTINGS)
    # Without them there is nothing to say.
    warnings.clear()
    monkeypatch.setattr(layout, "_warned_values", set())
    monkeypatch.setattr(fastworkflow, "_env_vars", clean)
    layout.embedding_settings()
    assert warnings == []


def test_the_two_modules_name_the_layout_files_alike():
    """`artifact_versioning` restates two names to stay free of imports."""
    assert artifact_versioning.WORKFLOW_MODEL_DIRNAME == layout.WORKFLOW_MODEL_DIRNAME
    assert artifact_versioning.WORKFLOW_SCOPE_FILENAME == layout.WORKFLOW_SCOPE_FILENAME
    assert st.REQUIRED_WORKFLOW_MODEL_ARTIFACTS == ("largemodel.pth", "label_encoder.pkl", "scopes.json")
    assert st.REQUIRED_WORKFLOW_CONTEXT_ARTIFACTS == ("threshold.json", "workflow_scope.json")


# ---------------------------------------------------------------- hand-built version folders

def _per_context_version(workflow: Path, version: str, contexts) -> Path:
    vdir = artifact_versioning.version_dir(str(workflow), version)
    for context_name in contexts:
        folder = vdir / artifact_versioning.context_folder_name(context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()
    return vdir


def _workflow_layout_version(workflow: Path, version: str, contexts, layout_name: str = layout.WORKFLOW) -> Path:
    """A version in a one-model layout (``workflow`` or ``embedding``): real small files, placeholder
    model entries."""
    vdir = artifact_versioning.version_dir(str(workflow), version)
    model = vdir / layout.WORKFLOW_MODEL_DIRNAME
    model.mkdir(parents=True, exist_ok=True)
    for artifact in layout.REQUIRED_MODEL_ARTIFACTS[layout_name]:
        (model / artifact).touch()
    for context_name in contexts:
        layout.write_context_files(
            str(vdir / artifact_versioning.context_folder_name(context_name)), context_name, layout_name)
    return vdir


def test_an_embedding_layout_version_is_complete_with_one_head_and_two_small_files_per_context(tmp_path):
    vdir = _workflow_layout_version(tmp_path, "v1", ["*", "Book"], layout.EMBEDDING)

    assert sorted(os.listdir(vdir / layout.WORKFLOW_MODEL_DIRNAME)) == ["head.npz", "scopes.json"]
    assert st.context_artifacts_complete(str(tmp_path), "v1", "*")
    assert st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert layout.version_layout(str(vdir)) == layout.EMBEDDING
    assert layout.context_layout(str(vdir / "Book")) == layout.EMBEDDING
    assert sorted(os.listdir(vdir / "Book")) == ["threshold.json", "workflow_scope.json"]
    # The pointer is the workflow layout's, and its `layout` field says which kind of model it points to.
    assert json.loads((vdir / "global" / "workflow_scope.json").read_text()) == {
        "layout": "embedding", "model": "../workflow", "context": "*"}
    sentinel = json.loads((vdir / "Book" / "threshold.json").read_text())
    assert sentinel["confidence_threshold"] == layout.WORKFLOW_LAYOUT_TIER_THRESHOLD == 2.0
    assert sentinel["layout"] == "embedding"
    assert intent_detector.has_head(str(vdir / "Book"))
    assert not intent_detector.has_head(str(vdir / layout.WORKFLOW_MODEL_DIRNAME))

    # The fine-tuned model's entries do not make an embedding version complete, and the other way round.
    (vdir / layout.WORKFLOW_MODEL_DIRNAME / "head.npz").unlink()
    for artifact in layout.REQUIRED_WORKFLOW_MODEL_ARTIFACTS:
        (vdir / layout.WORKFLOW_MODEL_DIRNAME / artifact).touch()
    assert not st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert not intent_detector.has_head(str(vdir / "Book"))


def test_a_marker_that_names_no_known_layout_is_refused(tmp_path):
    folder = tmp_path / "Book"
    folder.mkdir()
    (folder / "workflow_scope.json").write_text(json.dumps({"layout": "something_else", "context": "Book"}))
    with pytest.raises(ValueError, match="does not describe"):
        layout.read_scope_marker(str(folder))
    with pytest.raises(ValueError, match="not a layout with one model"):
        layout.write_context_files(str(folder), "Book", layout.PER_CONTEXT)


def test_a_per_context_version_is_complete_with_its_six_files_per_context(tmp_path):
    vdir = _per_context_version(tmp_path, "v1", ["*", "Book"])

    assert st.context_artifacts_complete(str(tmp_path), "v1", "*")
    assert st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert layout.version_layout(str(vdir)) == layout.PER_CONTEXT

    (vdir / "Book" / "largemodel.pth").unlink()
    assert not st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert st.context_artifacts_complete(str(tmp_path), "v1", "*")


def test_a_workflow_layout_version_is_complete_with_one_model_and_two_small_files_per_context(tmp_path):
    vdir = _workflow_layout_version(tmp_path, "v1", ["*", "Book"])

    assert st.context_artifacts_complete(str(tmp_path), "v1", "*")
    assert st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert layout.version_layout(str(vdir)) == layout.WORKFLOW
    assert sorted(os.listdir(vdir / "Book")) == ["threshold.json", "workflow_scope.json"]
    assert json.loads((vdir / "global" / "workflow_scope.json").read_text()) == {
        "layout": "workflow", "model": "../workflow", "context": "*"}
    # The threshold file every "is this context trained" gate looks for carries the documented
    # value no probability reaches.
    threshold = json.loads((vdir / "Book" / "threshold.json").read_text())
    assert threshold["confidence_threshold"] == layout.WORKFLOW_LAYOUT_TIER_THRESHOLD > 1.0


@pytest.mark.parametrize("missing", ["largemodel.pth", "label_encoder.pkl", "scopes.json"])
def test_a_missing_model_entry_makes_every_context_of_the_layout_incomplete(tmp_path, missing):
    vdir = _workflow_layout_version(tmp_path, "v1", ["*", "Book"])
    (vdir / layout.WORKFLOW_MODEL_DIRNAME / missing).unlink()

    assert not st.context_artifacts_complete(str(tmp_path), "v1", "*")
    assert not st.context_artifacts_complete(str(tmp_path), "v1", "Book")


@pytest.mark.parametrize("missing", ["threshold.json", "workflow_scope.json"])
def test_a_missing_small_file_makes_that_context_incomplete(tmp_path, missing):
    vdir = _workflow_layout_version(tmp_path, "v1", ["*", "Book"])
    (vdir / "Book" / missing).unlink()

    assert not st.context_artifacts_complete(str(tmp_path), "v1", "Book")
    assert st.context_artifacts_complete(str(tmp_path), "v1", "*")


def test_the_model_folder_is_not_a_context_of_the_version(tmp_path):
    """Publication routes contexts; the model folder is reached through them and gets no entry."""
    _workflow_layout_version(tmp_path, "v1", ["*", "Book"])
    artifact_versioning.write_manifest(str(tmp_path), "v1", seed=SEED)

    assert artifact_versioning.version_context_names(str(tmp_path), "v1") == ["Book", "global"]

    artifact_versioning.publish_version(str(tmp_path), "v1")
    info = artifact_versioning.command_info_root(str(tmp_path))
    assert not (info / layout.WORKFLOW_MODEL_DIRNAME).exists()
    assert (info / "Book" / layout.WORKFLOW_SCOPE_FILENAME).is_file()
    # Through the published entry, the model folder is the one inside the version.
    assert layout.workflow_model_dir(str(info / "Book")) == os.path.realpath(
        artifact_versioning.version_dir(str(tmp_path), "v1") / layout.WORKFLOW_MODEL_DIRNAME)
    assert artifact_versioning.read_manifest(str(tmp_path), "v1")["contexts"] == ["Book", "global"]


def test_a_per_context_context_that_is_called_workflow_is_still_a_context(tmp_path):
    _per_context_version(tmp_path, "v1", ["*", "workflow"])

    assert artifact_versioning.version_context_names(str(tmp_path), "v1") == ["global", "workflow"]


def test_carrying_a_workflow_layout_context_forward_brings_the_model_with_it(tmp_path):
    _workflow_layout_version(tmp_path, "v1", ["*", "Book"])

    assert artifact_versioning.carry_forward_context(str(tmp_path), "v1", "v2", "Book")
    assert artifact_versioning.carry_forward_context(str(tmp_path), "v1", "v2", "*")

    assert st.context_artifacts_complete(str(tmp_path), "v2", "Book")
    assert st.context_artifacts_complete(str(tmp_path), "v2", "*")
    v2 = artifact_versioning.version_dir(str(tmp_path), "v2")
    assert sorted(os.listdir(v2)) == ["Book", "global", "workflow"]


def test_a_workflow_layout_context_is_not_carried_forward_without_its_model(tmp_path):
    vdir = _workflow_layout_version(tmp_path, "v1", ["Book"])
    shutil.rmtree(vdir / layout.WORKFLOW_MODEL_DIRNAME)

    assert not artifact_versioning.carry_forward_context(str(tmp_path), "v1", "v2", "Book")


def test_a_per_context_context_is_carried_forward_as_before(tmp_path):
    _per_context_version(tmp_path, "v1", ["Book"])

    assert artifact_versioning.carry_forward_context(str(tmp_path), "v1", "v2", "Book")
    v2 = artifact_versioning.version_dir(str(tmp_path), "v2")
    assert sorted(os.listdir(v2)) == ["Book"]
    assert st.context_artifacts_complete(str(tmp_path), "v2", "Book")


def test_a_context_of_either_layout_counts_as_a_trained_head(tmp_path):
    from fastworkflow.train.score_lookup import HEAD_MARKERS

    per_context = tmp_path / "per_context"
    per_context.mkdir()
    for marker in HEAD_MARKERS:
        (per_context / marker).touch()
    vdir = _workflow_layout_version(tmp_path, "v1", ["Book", "Member"])
    (vdir / "Member" / "workflow_scope.json").unlink()
    empty = tmp_path / "empty"
    empty.mkdir()

    assert intent_detector.has_head(str(per_context))
    assert intent_detector.has_head(str(vdir / "Book"))
    assert not intent_detector.has_head(str(vdir / "Member"))
    assert not intent_detector.has_head(str(empty))
    # The model folder itself is never a head.
    assert not intent_detector.has_head(str(vdir / layout.WORKFLOW_MODEL_DIRNAME))


# ---------------------------------------------------------------- the planner

def _copy(tmp_path: Path, source: Path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


def _seed_utterance_cache(workflow: str) -> None:
    """A placeholder cache entry per execution command, which the planner requires for carry-forward."""
    root = os.path.join(workflow, "___command_info", utterance_cache.CACHE_DIRNAME)
    os.makedirs(root, exist_ok=True)
    context_commands, _ = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    for commands in context_commands.values():
        for command in commands:
            with open(os.path.join(root, f"{utterance_cache.slugify(command)}.baseline.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"entries": {}}, handle)


def _publish_baseline(workflow: str, layout_name: str) -> str:
    """Publish a complete-looking version in *layout_name*. No model is trained."""
    RoutingRegistry.clear_registry()
    _seed_utterance_cache(workflow)
    contexts = sorted(st.contexts_for_training(workflow))
    version = artifact_versioning.new_version_id()
    if layout_name in layout.ONE_MODEL_LAYOUTS:
        _workflow_layout_version(Path(workflow), version, contexts, layout_name)
    else:
        _per_context_version(Path(workflow), version, contexts)
    signature, _unresolved = st.compute_training_signature(
        workflow, contexts, seed=SEED, intent_model_layout_name=layout_name)
    artifact_versioning.write_manifest(workflow, version, seed=SEED)
    st.save_training_signature(workflow, version, signature)
    artifact_versioning.publish_version(workflow, version)
    return version


def _plan(workflow: str, version: str, layout_name):
    RoutingRegistry.clear_registry()  # the registry caches metadata per process; an edit must be re-read
    plan, _signature = st.compute_training_plan(
        workflow, sorted(st.contexts_for_training(workflow)),
        changed_only=True, seed=SEED, carry_forward_from=version,
        intent_model_layout_name=layout_name)
    return plan


def _edit_one_seed(workflow: str) -> str:
    """Add a seed to one real execution command; return its qualified name."""
    context_commands, _u = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    known = set().union(*context_commands.values())
    commands_dir = Path(workflow) / "_commands"
    for path in sorted(commands_dir.rglob("*.py")):
        rel = path.relative_to(commands_dir)
        name = str(rel.with_suffix("")).replace(os.sep, "/")
        text = path.read_text(encoding="utf-8")
        if (any(part.startswith(("_", ".")) for part in rel.parts) or name not in known
                or "PlanGenerator" in text or "plain_utterances = [" not in text):
            continue
        path.write_text(text.replace(
            "plain_utterances = [", 'plain_utterances = [\n        "seed added by the layout test",', 1),
            encoding="utf-8")
        return name
    raise AssertionError("no execution command with a seed list was found to edit")


def test_an_unchanged_workflow_in_any_layout_plans_nothing(tmp_path):
    for layout_name in layout.LAYOUTS:
        workflow = _copy(tmp_path, MESSAGING, f"unchanged_{layout_name}")
        version = _publish_baseline(workflow, layout_name)

        plan = _plan(workflow, version, layout_name)

        assert plan.contexts_to_train == [], (layout_name, plan.reasons, plan.global_reasons)


def test_no_layout_given_is_the_per_context_layout(tmp_path):
    """A caller that predates the setting, and a signature saved before it existed, are per-context."""
    workflow = _copy(tmp_path, MESSAGING, "legacy_caller")
    version = _publish_baseline(workflow, layout.PER_CONTEXT)

    assert _plan(workflow, version, None).contexts_to_train == []
    saved = json.loads(st.signature_path(workflow, version).read_text())
    assert "intent_model_layout" not in saved and "workflow_model_source_digest" not in saved
    assert "embedding_head" not in saved


def test_the_workflow_layout_is_named_in_its_saved_signature(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "named")
    version = _publish_baseline(workflow, layout.WORKFLOW)

    saved = json.loads(st.signature_path(workflow, version).read_text())
    assert saved["intent_model_layout"] == "workflow"
    assert saved["workflow_model_source_digest"]
    assert "embedding_head" not in saved


def test_the_embedding_layout_records_its_encoder_and_targets_in_the_signature(tmp_path, monkeypatch):
    workflow = _copy(tmp_path, MESSAGING, "embedding_named")
    version = _publish_baseline(workflow, layout.EMBEDDING)

    saved = json.loads(st.signature_path(workflow, version).read_text())
    assert saved["intent_model_layout"] == "embedding" and saved["workflow_model_source_digest"]
    recorded = json.loads(saved["embedding_head"])
    assert recorded["encoder"] == layout.embedding_settings()["encoder"]
    assert recorded["pooling"] == layout.embedding_settings()["pooling"]
    assert recorded["prefix"] == layout.embedding_settings()["prefix"] == "query: "
    assert recorded["commit_threshold"] == 0.65 and "wrong_commit_target" not in recorded
    assert recorded["source_digest"]
    assert _plan(workflow, version, layout.EMBEDDING).contexts_to_train == []

    # The encoder and each target are global inputs: changing one refits the head.
    for name, value in ((layout.ENCODER_SETTING, "BAAI/bge-base-en-v1.5"),
                        (layout.COMMIT_THRESHOLD_SETTING, "0.7"),
                        (layout.POOLING_SETTING, "cls")):
        monkeypatch.setattr(fastworkflow, "_env_vars", {**fastworkflow._env_vars, name: value})
        plan = _plan(workflow, version, layout.EMBEDDING)
        assert plan.is_full_retrain, name
        assert any("embedding_head changed" in reason for reason in plan.global_reasons), (name, plan.global_reasons)
        monkeypatch.undo()


def test_one_changed_command_refits_the_head_in_the_embedding_layout(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "embedding_edit")
    version = _publish_baseline(workflow, layout.EMBEDDING)
    command = _edit_one_seed(workflow)

    plan = _plan(workflow, version, layout.EMBEDDING)

    assert plan.is_full_retrain
    assert plan.contexts_to_train == sorted(st.contexts_for_training(workflow))
    assert plan.contexts_carried_forward == [] and plan.dirty_commands == [command]
    assert any("INTENT_MODEL_LAYOUT=embedding: one model serves every context" in reason
               for reason in plan.global_reasons)


@pytest.mark.parametrize("before, after", [
    (layout.PER_CONTEXT, layout.WORKFLOW),
    (layout.WORKFLOW, layout.PER_CONTEXT),
    (layout.PER_CONTEXT, layout.EMBEDDING),
    (layout.EMBEDDING, layout.PER_CONTEXT),
    (layout.WORKFLOW, layout.EMBEDDING),
    (layout.EMBEDDING, layout.WORKFLOW),
])
def test_a_change_of_layout_is_a_full_retrain(tmp_path, before, after):
    workflow = _copy(tmp_path, MESSAGING, f"{before}_to_{after}")
    version = _publish_baseline(workflow, before)

    plan = _plan(workflow, version, after)

    assert plan.is_full_retrain
    assert plan.contexts_to_train == sorted(st.contexts_for_training(workflow))
    assert plan.contexts_carried_forward == []
    assert any("intent_model_layout changed" in reason and repr(after) in reason
               for reason in plan.global_reasons), plan.global_reasons


def test_one_changed_command_retrains_every_context_in_the_workflow_layout(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "workflow_edit")
    version = _publish_baseline(workflow, layout.WORKFLOW)
    command = _edit_one_seed(workflow)

    plan = _plan(workflow, version, layout.WORKFLOW)

    assert plan.is_full_retrain
    assert plan.contexts_to_train == sorted(st.contexts_for_training(workflow))
    assert plan.contexts_carried_forward == []
    assert plan.dirty_commands == [command]
    assert any("one model serves every context" in reason for reason in plan.global_reasons)


def test_the_same_change_in_the_per_context_layout_stays_selective(tmp_path):
    """The control for the test above: the layout, not the edit, is what widens the plan."""
    workflow = _copy(tmp_path, MESSAGING, "per_context_edit")
    version = _publish_baseline(workflow, layout.PER_CONTEXT)
    _edit_one_seed(workflow)

    plan = _plan(workflow, version, layout.PER_CONTEXT)

    assert not plan.is_full_retrain
    assert plan.contexts_to_train and plan.contexts_carried_forward


def test_a_workflow_layout_version_that_lost_its_model_is_retrained(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "lost_model")
    version = _publish_baseline(workflow, layout.WORKFLOW)
    shutil.rmtree(artifact_versioning.version_dir(workflow, version) / layout.WORKFLOW_MODEL_DIRNAME)

    plan = _plan(workflow, version, layout.WORKFLOW)

    assert plan.is_full_retrain
    assert plan.contexts_to_train == sorted(st.contexts_for_training(workflow))


def test_the_trainer_refuses_a_selection_of_contexts_in_the_workflow_layout(tmp_path):
    """There is one model: `train` cannot be asked for some of its contexts."""
    from fastworkflow import model_pipeline_training

    workflow_path = _copy(tmp_path, MESSAGING, "selection")
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str="layout-selection")
    try:
        for one_model_layout in layout.ONE_MODEL_LAYOUTS:
            with pytest.raises(ValueError, match="one model for every context"):
                model_pipeline_training.train(
                    workflow, contexts_to_train={"*"}, stop_before_fit=True,
                    intent_model_layout=one_model_layout)
        with pytest.raises(ValueError, match="unknown intent model layout"):
            model_pipeline_training.train(workflow, stop_before_fit=True, intent_model_layout="other")
    finally:
        workflow.close()


# ---------------------------------------------------------------- the router: dispatch

@pytest.fixture
def clean_router_cache():
    """`CommandRouter` remembers an instance per folder, also one whose loading failed."""
    from fastworkflow.model_pipeline_training import CommandRouter

    before = set(CommandRouter._instances_cache)
    yield
    for key in set(CommandRouter._instances_cache) - before:
        CommandRouter._instances_cache.pop(key, None)


def test_the_router_reads_the_layout_from_the_marker_in_the_folder(tmp_path, clean_router_cache):
    """With the marker, the workflow's model folder is what gets opened; without it, the
    context's own files. Both folders are empty of weights, so each fails on the first file of
    the layout it chose, which is what tells the two apart without loading a model."""
    from fastworkflow.model_pipeline_training import CommandRouter

    vdir = _workflow_layout_version(tmp_path, "v1", ["Book"])
    with pytest.raises(Exception) as workflow_error:
        CommandRouter(str(vdir / "Book"))
    assert "scopes.json" in str(workflow_error.value) or isinstance(
        workflow_error.value, (ValueError, json.JSONDecodeError))

    plain = tmp_path / "plain" / "Book"
    plain.mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as per_context_error:
        CommandRouter(str(plain))
    assert "threshold.json" in str(per_context_error.value)


# ---------------------------------------------------------------- the router: real weights

@pytest.fixture(scope="module")
def published_experiment_model(tmp_path_factory):
    """A published ``workflow``-layout version whose model folder is a LINK to weights the
    single-model experiment trained. Returns ``(workflow folder, scope names)``."""
    required = ["largemodel.pth", "label_encoder.pkl", "scopes.json"]
    if not all((EXPERIMENT_MODEL / name).exists() for name in required):
        pytest.skip(
            f"no trained workflow model at {EXPERIMENT_MODEL} (it is produced by "
            f"scripts/experiments/single_intent_model.py and lives outside the repository)")
    workflow = tmp_path_factory.mktemp("published_layout") / "library"
    vdir = artifact_versioning.version_dir(str(workflow), "v1")
    vdir.mkdir(parents=True)
    os.symlink(EXPERIMENT_MODEL, vdir / layout.WORKFLOW_MODEL_DIRNAME)
    scopes = sorted(json.loads((EXPERIMENT_MODEL / "scopes.json").read_text())["scopes"])
    for scope in scopes:
        layout.write_context_files(str(vdir / artifact_versioning.context_folder_name(scope)), scope)
    artifact_versioning.write_manifest(str(workflow), "v1", seed=SEED)
    artifact_versioning.publish_version(str(workflow), "v1")
    yield str(workflow), scopes
    from fastworkflow.model_pipeline_training import CommandRouter

    for key in [key for key in CommandRouter._instances_cache if key.startswith(str(workflow))]:
        CommandRouter._instances_cache.pop(key, None)


def _router(workflow: str, scope: str):
    from fastworkflow.model_pipeline_training import CommandRouter

    return CommandRouter(intent_detector.head_folder(workflow, scope))


DETAIL_KEYS = {"model_tier", "confidence", "ambiguous_threshold", "confident", "top_label",
               "topk_labels", "topk_scores", "p_in", "p_out", "escalation_margin"}


def test_every_scope_of_a_version_shares_one_loaded_model(published_experiment_model):
    from fastworkflow.model_pipeline_training import CommandRouter
    from fastworkflow.train import workflow_model

    workflow, scopes = published_experiment_model
    routers = {scope: _router(workflow, scope) for scope in scopes}

    assert all(isinstance(router, CommandRouter) for router in routers.values())
    assert len({id(router.modelpipeline) for router in routers.values()}) == 1
    pipeline = routers[scopes[0]].modelpipeline
    assert isinstance(pipeline, workflow_model.WorkflowModelPipeline)
    # Keyed by the real path of the model folder: the weights themselves, not the link to them.
    assert workflow_model.WorkflowModelPipeline._instances[os.path.realpath(EXPERIMENT_MODEL)] is pipeline
    assert hasattr(pipeline.distil_model, "distilbert") and pipeline.distil_tokenizer is not None
    assert pipeline.device in ("cpu", "cuda")
    # Same process-wide instance cache, keyed by the normalised path ('*' is 'global').
    star = os.path.join(workflow, "___command_info", "*")
    assert CommandRouter(star) is routers["*"]
    assert star.replace("*", "global") in CommandRouter._instances_cache
    assert routers["Book"].label_encoder_path == str(EXPERIMENT_MODEL / "label_encoder.pkl")


def test_a_command_of_the_scope_is_answered_with_the_documented_details(published_experiment_model):
    workflow, _scopes = published_experiment_model
    router = _router(workflow, "Book")

    labels, details = router.predict_with_details("show me this book")

    assert labels == ["Book/show_book"]
    assert router.predict("show me this book") == labels
    assert set(details) == DETAIL_KEYS
    assert details["model_tier"] == "large"
    assert details["confident"] is True and details["top_label"] == "Book/show_book"
    assert details["confidence"] > details["ambiguous_threshold"]
    assert len(details["topk_labels"]) == len(details["topk_scores"]) == 3
    assert all(label.startswith(("Book/", "IntentDetection/")) for label in details["topk_labels"])
    assert details["p_in"] > details["p_out"]
    assert details["escalation_margin"] >= 0.0
    json.dumps(details)  # the span attributes are written as JSON


def test_a_request_for_another_scope_is_declined_with_the_wildcard_label(published_experiment_model):
    """Escalation: a Library request asked of the Book scope is the single label `wildcard`,
    and the same request asked of the Library scope is a Library command."""
    workflow, _scopes = published_experiment_model

    labels, details = _router(workflow, "Book").predict_with_details("list all the branches")
    assert labels == [WILDCARD_LABEL]
    assert details["p_out"] > details["p_in"] + details["escalation_margin"]
    assert all(label != WILDCARD_LABEL for label in details["topk_labels"])

    library = _router(workflow, "Library").predict("list all the branches")
    assert "Library/list_branches" in library


def test_a_bare_value_is_the_parameter_value_label(published_experiment_model):
    workflow, _scopes = published_experiment_model

    assert _router(workflow, "Book").predict("france") == [PARAMETER_VALUE_LABEL]


def test_the_detector_contract_reads_the_layout_without_change(published_experiment_model, monkeypatch):
    """`has_head`, the engine of a scope and the contract's four kinds, over the published entry."""
    workflow, scopes = published_experiment_model

    for scope in scopes:
        assert intent_detector.has_head(intent_detector.head_folder(workflow, scope))
    engine = intent_detector.engine_for(workflow, "Book", allow_matcher=False)
    assert isinstance(engine, intent_detector.ClassifierEngine)
    # The folder holds only the published model (no _commands/), so the engine's filter of the commands that do not
    # load (RoutingDefinition.not_loading, which reads the workflow's routing) cannot run here: the model's answers are
    # read through the same engine with no workflow to filter against.
    engine = intent_detector.ClassifierEngine(engine.router)

    assert engine.decide("show me this book").kind == intent_detector.MATCH
    assert engine.decide("list all the branches").kind == intent_detector.NOT_MINE
    assert engine.decide("france").kind == intent_detector.BARE_VALUE


def test_warm_up_loads_every_context_of_the_layout(published_experiment_model):
    from fastworkflow.model_pipeline_training import warm_intent_heads

    workflow, scopes = published_experiment_model

    assert warm_intent_heads(workflow) == sorted(
        artifact_versioning.context_folder_name(scope) for scope in scopes)


def test_the_signal_version_still_names_the_artifact_version(published_experiment_model):
    from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
        _classifier_signal_version,
    )

    workflow, _scopes = published_experiment_model
    folder = os.path.join(workflow, "___command_info", "*")

    assert _classifier_signal_version(folder, "large") == "intent-classifier/v1/global/large"


def test_a_newly_published_version_is_read_from_its_own_model_folder(published_experiment_model, tmp_path):
    """The model is found through the real path of the version's folder, so two versions are two
    keys even when one context entry name serves both over time."""
    workflow, _scopes = published_experiment_model
    other = tmp_path / "other"
    vdir = _workflow_layout_version(other, "v2", ["Book"])

    first = layout.workflow_model_dir(os.path.join(workflow, "___command_info", "Book"))
    second = layout.workflow_model_dir(str(vdir / "Book"))

    assert first == os.path.realpath(EXPERIMENT_MODEL)
    assert second == os.path.realpath(vdir / layout.WORKFLOW_MODEL_DIRNAME)
    assert first != second


# ---------------------------------------------------------------- the rows the trainer assembles

def test_the_trainers_rows_are_the_rows_rebuilt_from_the_caches(tmp_path, monkeypatch):
    """The experiment measured rows rebuilt from the utterance caches; the trainer assembles its
    rows in process, inside `train`. On a warm cache the two are the same rows, so what the
    trainer ships is what was measured. No model is called and nothing is fitted."""
    from dotenv import dotenv_values

    from fastworkflow import model_pipeline_training
    from fastworkflow.train import determinism, workflow_model

    cache_root = LIBRARY / "___command_info" / utterance_cache.CACHE_DIRNAME
    if not cache_root.is_dir() or not any(cache_root.iterdir()):
        pytest.skip(f"the library example has no utterance cache at {cache_root} (it is a local, "
                    f"untracked training output), so its rows cannot be rebuilt offline")
    copy = tmp_path / "library"
    shutil.copytree(LIBRARY, copy, symlinks=True, ignore=shutil.ignore_patterns(
        "versions", "___convo_info", "___workflow_contexts", "__pycache__"))
    info = copy / "___command_info"
    for entry in list(info.iterdir()):
        if entry.is_symlink() or entry.name in ("current.json", ".publication.lock"):
            entry.unlink()
    RoutingRegistry.clear_registry()
    examples = REPO / "fastworkflow" / "examples"
    env = {**dotenv_values(examples / "fastworkflow.env"),
           **dotenv_values(examples / "fastworkflow.passwords.env")}
    # The cache key holds the generating model, so the cache's own model string is used.
    env["LLM_SYNDATA_GEN"] = workflow_model.cached_generation_model(str(copy))
    monkeypatch.setattr(fastworkflow, "_env_vars", env)

    try:
        rebuilt = workflow_model.workflow_rows(str(copy))
    except workflow_model.UtteranceCacheMissError as miss:
        pytest.skip(f"the library example's utterance cache is not warm for this checkout: {miss}")

    recorder = determinism.ProvenanceRecorder(str(copy))
    cache = utterance_cache.UtteranceCache(str(copy), mode=utterance_cache.MODE_REUSE)
    workflow = fastworkflow.Workflow.create(str(copy), workflow_id_str="layout-rows-in-process")
    determinism.set_provenance_recorder(recorder)
    utterance_cache.set_utterance_cache(cache)
    try:
        with workflow_model.no_model_calls(), workflow_model.observe_assembly() as assembled:
            used = model_pipeline_training.train(
                workflow, stop_before_fit=True, intent_model_layout="workflow")
    finally:
        determinism.set_provenance_recorder(None)
        utterance_cache.set_utterance_cache(None)
        workflow.close()

    assert used == {}  # no per-context "not here" samples are drawn in this layout
    assert cache.stats["miss"] == 0 and cache.stats["stored"] == 0
    assert len(assembled) == 1
    assert workflow_model.row_differences(assembled[0], rebuilt) == []
    # In the same order too: the in-distribution split is drawn over the list as it stands.
    assert assembled[0].rows == rebuilt.rows
    assert assembled[0].labels == rebuilt.labels
    assert WILDCARD_LABEL in rebuilt.labels and PARAMETER_VALUE_LABEL in rebuilt.labels
    wildcard_rows = [text for text, label in assembled[0].rows if label == WILDCARD_LABEL]
    assert len(wildcard_rows) == (
        len(model_pipeline_training.BUILT_IN_VAGUE_REPLIES) * model_pipeline_training.BUILT_IN_VAGUE_WEIGHT)
    # Nothing was written into a model folder: the run stopped before fitting.
    assert not (info / layout.WORKFLOW_MODEL_DIRNAME).exists()
