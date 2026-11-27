"""End to end: `fastworkflow train` in each intent model layout, then the runtime.

Real training runs on temporary COPIES of the bundled library example (five contexts, an
inheritance edge, nested records), so this file costs minutes. Nothing under
``fastworkflow/examples`` or ``tests`` is trained into or changed. Two tests:

* the default layout (``embedding``: a frozen sentence encoder and one linear head), trained with
  no layout setting at all, then refitted with another encoder;
* the ``workflow`` layout (one fine-tuned model), then back to ``per_context``, each pinned by name.

No model is called. The copy brings the example's own generated-utterance cache, so the
classifier's rows are read back, and its cached parameter examples are stored again under
today's cache key (see `_rekey_parameter_examples`). The run is wrapped in a guard that raises
on any model call, so a cold cache fails here instead of spending. The caches are local,
untracked training output: on a checkout without them the test skips and says so.

The runtime steps run with no planner key (the ``no_planner_key`` fixture and `_training_env`), so
the detection chain is the local model alone and an unconfident answer calls nothing.

What the ``workflow`` layout test checks, in one test because each step needs the one before it:

1. ``INTENT_MODEL_LAYOUT=workflow`` publishes one model folder and two small files per context.
2. Every "is it trained" gate says yes, and the CME path routes a typed request: a command of
   the current record, and a parent's command from a nested record (escalation).
3. A second train is a no-op.
4. Setting the layout to ``per_context`` retrains everything into the per-context layout.

The ``embedding`` layout test checks the same five things for the default layout, with a change of
encoder (a full refit) in place of step 5.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.intent import intent_detector
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow.engine.command_executor import CommandExecutor
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.train import artifact_versioning, param_example_cache
from fastworkflow.train import intent_model_layout as layout
from fastworkflow.train import selective_training as st
from fastworkflow.train import utterance_cache
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixture
    initialized_fastworkflow,
)

# A real `train_workflow`, twice. The markers are the ones the other real-training files carry
# (tests/test_train_modern_stack.py), so `-m "not slow"` and `-m "not requires_llm_key"` deselect
# this file with them. This one needs no key when the copied caches are warm (see the module
# docstring) and skips when they are not.
pytestmark = [pytest.mark.slow, pytest.mark.requires_llm_key]

REPO = Path(__file__).resolve().parent.parent
LIBRARY = REPO / "fastworkflow" / "examples" / "library"
CONTEXT_FOLDERS = ["Book", "Library", "Member", "Record", "global"]

#: Phrasings for commands of the Book the session stands on. Probed against the trained model:
#: the first one it names a Book command for with confidence is the one routed.
OWN_CANDIDATES = [
    "show the book",
    "show me this book",
    "show this book's details",
    "what are the details of this book",
    "list the loans of this book",
]

#: Phrasings for Library commands that take no values, typed while standing on a Book. The Book
#: scope has to decline them and the Library scope has to name the command.
PARENT_CANDIDATES = [
    "list all the branches",
    "which branches does the library have",
    "show me the library branches",
    "how many books are there in the library",
    "count the books in the library",
]
PARENT_COMMANDS = {"Library/list_branches", "Library/count_books"}


def _copy_library(destination: Path) -> str:
    """The example's code and caches, without its trained versions or the links into them."""
    shutil.copytree(LIBRARY, destination, symlinks=True, ignore=shutil.ignore_patterns(
        "versions", "___convo_info", "___workflow_contexts", "__pycache__"))
    info = destination / "___command_info"
    for entry in list(info.iterdir()):
        if entry.is_symlink() or entry.name in ("current.json", ".publication.lock"):
            entry.unlink()
    RoutingRegistry.clear_registry()
    return str(destination)


def _training_env(workflow_path: str, layout_name: str | None, extra: dict | None = None) -> dict:
    from fastworkflow.train import workflow_model

    examples = REPO / "fastworkflow" / "examples"
    env = {**dotenv_values(examples / "fastworkflow.env"),
           **dotenv_values(examples / "fastworkflow.passwords.env")}
    # The utterance cache key holds the generating model, so the cache's own model string is used.
    env["LLM_SYNDATA_GEN"] = workflow_model.cached_generation_model(workflow_path)
    # No planner key: the runtime steps assert the local model's answers, and with a key an
    # unconfident answer would be a live call to the remote decision model.
    env.pop("JEV_API_KEY", None)
    for name in (layout.SETTING, layout.ENCODER_SETTING, layout.POOLING_SETTING,
                 layout.COMMIT_THRESHOLD_SETTING, *layout.REMOVED_SETTINGS):
        env.pop(name, None)
    if layout_name is not None:
        env[layout.SETTING] = layout_name
    env.update(extra or {})
    return env


def _rekey_parameter_examples(workflow_path: str) -> None:
    """Store the copy's cached parameter examples under today's cache key.

    The key includes a digest of the prompt code, which has changed since the example's
    parameter examples were drawn, so every lookup would be a miss and a model call. The
    examples themselves are still what the trainer would write; only their key is renewed.
    Skips the test when a command has no cached examples to renew.
    """
    from fastworkflow.train import __main__ as trainer
    from fastworkflow.train.determinism import get_training_seed
    from fastworkflow.train.generate_param_examples import param_example_fingerprint

    RoutingRegistry.get_definition(workflow_path, load_cached=False)
    cache = param_example_cache.ParamExampleCache(workflow_path, mode=param_example_cache.MODE_REUSE)
    cached = {}
    for path in glob.glob(os.path.join(cache.root, "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cached[payload["command_name"]] = payload
    seed = get_training_seed()
    commands = trainer._get_commands_with_parameters(
        os.path.join(workflow_path, "___command_info", "command_directory.json"))
    for name, metadata in commands.items():
        fields = trainer.parameter_model_of(workflow_path, metadata)
        if fields is None:
            continue
        entry = (cached.get(name) or {}).get("entries", {}).get(str(seed))
        if not entry:
            pytest.skip(f"the library example has no cached parameter examples for {name!r} at "
                        f"seed {seed}, so training it would call a model")
        fingerprint = param_example_fingerprint(
            fields.model_fields, name, trainer.PARAM_EXAMPLE_COUNT,
            trainer.PARAM_VALIDATION_THRESHOLD)
        assert cache.store(fingerprint, seed, entry["valid_examples"], entry.get("rejected_examples"))
    RoutingRegistry.clear_registry()


def _train(workflow_path: str, layout_name: str | None, extra: dict | None = None) -> str:
    """One `train_workflow` under *layout_name* (None: no setting at all, so the default layout), with
    model calls refused. Returns the current version."""
    from fastworkflow.train import workflow_model
    from fastworkflow.train.__main__ import train_workflow

    fastworkflow.init(env_vars=_training_env(workflow_path, layout_name, extra))
    RoutingRegistry.clear_registry()
    with workflow_model.no_model_calls():
        train_workflow(workflow_path)
    return artifact_versioning.resolve_current_version(workflow_path)


def _forget_loaded_routers(workflow_path: str) -> None:
    """A process keeps the router of a context folder once loaded. After a retrain in the same
    process the folder name is the same and its content is not, so the test drops them, as a
    server restart would."""
    from fastworkflow.model_pipeline_training import CommandRouter

    for key in [key for key in CommandRouter._instances_cache if key.startswith(workflow_path)]:
        CommandRouter._instances_cache.pop(key, None)


def _session_on_a_book(workflow_path: str, name: str):
    """The trained copy with a real Library as its root and a real Book as the current record."""
    app = fastworkflow.Workflow.create(workflow_folderpath=workflow_path, workflow_id_str=name)
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    if app.root is None:
        from fastworkflow.examples.library.application.library import Library

        app.root = Library()
    app.record = app.root.books["b1"]
    return app, cme


def _confident(cme, candidates, head: str):
    """The candidates the trained model names one command for when asked in *head* (the local model alone)."""
    from fastworkflow.intent import intent_detector

    app = cme.context["app_workflow"]
    engine = intent_detector.engine_for(app.folderpath, head, allow_matcher=False)
    hits = []
    for phrasing in candidates:
        answer = engine.decide(phrasing)
        if answer.kind == intent_detector.MATCH and answer.confident and answer.labels[0] not in CORE_LABELS:
            hits.append((phrasing, answer.labels[0]))
    return hits


#: The framework's own commands are not a scope's answer for these probes.
CORE_LABELS = frozenset({"IntentDetection/go_up", "IntentDetection/reset_context", "IntentDetection/what_can_i_do",
                         "IntentDetection/what_is_current_context", "ErrorCorrection/abort",
                         "ErrorCorrection/you_misunderstood"})


def _spy_on_the_heads_asked(monkeypatch) -> list:
    """Record each head the decision asks, in order, and call through (a spy on ``intent_detector.engine_for``)."""
    from fastworkflow.intent import intent_detector

    asked: list = []
    original = intent_detector.engine_for

    def spy(workflow_folderpath, context_name, *args, **kwargs):
        engine = original(workflow_folderpath, context_name, *args, **kwargs)
        if engine is not None:
            decide = engine.decide

            def recording(utterance):
                asked.append(context_name)
                return decide(utterance)

            engine.decide = recording
        return engine

    monkeypatch.setattr(intent_detector, "engine_for", spy)
    return asked


def _route(cme, text: str):
    """Run the typed text through the CME walk, as a deterministic turn does."""
    return CommandExecutor.perform_action(
        cme, fastworkflow.Action(command_name="wildcard", command=text))


def test_train_and_run_in_the_workflow_layout_then_back(
        tmp_path, initialized_fastworkflow, monkeypatch, no_planner_key):
    from fastworkflow.model_pipeline_training import (
        CommandRouter,
        is_workflow_trained,
        warm_intent_heads,
    )
    from fastworkflow.train import workflow_model

    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets package not installed; intent-detection training is skipped.")
    cache_root = LIBRARY / "___command_info" / utterance_cache.CACHE_DIRNAME
    if not cache_root.is_dir() or not any(cache_root.glob("*.json")):
        pytest.skip(f"the library example has no generated-utterance cache at {cache_root} (local, "
                    f"untracked training output), so training a copy would call a model")

    workflow_path = _copy_library(tmp_path / "library")
    fastworkflow.init(env_vars=_training_env(workflow_path, layout.WORKFLOW))
    _rekey_parameter_examples(workflow_path)
    info = Path(workflow_path) / "___command_info"

    # ------------------------------------------------------------------ 1. train: the layout
    try:
        version = _train(workflow_path, layout.WORKFLOW)
    except workflow_model.ModelCallAttemptedError as attempted:
        pytest.skip(f"the copied caches are not warm for this checkout, and the run would have "
                    f"called a model: {attempted}")
    vdir = artifact_versioning.version_dir(workflow_path, version)

    assert sorted(entry.name for entry in vdir.iterdir() if entry.is_dir()) == sorted(
        [*CONTEXT_FOLDERS, layout.WORKFLOW_MODEL_DIRNAME])
    assert sorted(os.listdir(vdir / layout.WORKFLOW_MODEL_DIRNAME)) == [
        "label_encoder.pkl", "largemodel.pth", "scopes.json"]
    assert (vdir / layout.WORKFLOW_MODEL_DIRNAME / "largemodel.pth").is_dir()
    for folder in CONTEXT_FOLDERS:
        assert sorted(os.listdir(vdir / folder)) == ["threshold.json", "workflow_scope.json"], folder
        # The published entry is the usual link into the version; the model folder has none.
        assert os.path.realpath(info / folder) == os.path.realpath(vdir / folder)
        assert st.context_artifacts_complete(workflow_path, version, "*" if folder == "global" else folder)
    assert not (info / layout.WORKFLOW_MODEL_DIRNAME).exists()
    assert json.loads((vdir / "global" / "workflow_scope.json").read_text()) == {
        "layout": "workflow", "model": "../workflow", "context": "*"}

    manifest = artifact_versioning.read_manifest(workflow_path, version)
    assert manifest[layout.MANIFEST_KEY] == layout.WORKFLOW
    assert manifest["contexts"] == CONTEXT_FOLDERS
    assert sorted(manifest["contexts_retrained"]) == ["*", "Book", "Library", "Member", "Record"]
    signature = st.load_training_signature(workflow_path, version)
    assert signature.intent_model_layout == layout.WORKFLOW and signature.workflow_model_source_digest

    scopes = json.loads((vdir / layout.WORKFLOW_MODEL_DIRNAME / "scopes.json").read_text())
    assert scopes["tiers"] == ["large"]
    assert sorted(scopes["scopes"]) == ["*", "Book", "Library", "Member", "Record"]
    assert "wildcard" in scopes["labels"] and "parameter_value" in scopes["labels"]
    assert all("tier_threshold" not in scope for scope in scopes["scopes"].values())

    # The same reports as a per-context run, one entry per context.
    heldout = json.loads((info / "heldout_evaluation.json").read_text())
    assert sorted(entry["context"] for entry in heldout["contexts"]) == [
        "*", "Book", "Library", "Member", "Record"]
    assert all(entry["large_ambiguity"]["intent_model_layout"] == "workflow"
               for entry in heldout["contexts"])
    assert all(not any("held-out evaluation failed" in note for note in entry["notes"])
               for entry in heldout["contexts"])
    assert (info / "training_report.json").is_file()
    provenance = json.loads((info / "training_provenance.json").read_text())
    assert set(provenance["context_training"]) == {"*", "Book", "Library", "Member", "Record"}

    # ------------------------------------------------------------------ 2. the runtime
    assert is_workflow_trained(workflow_path) == (True, [])
    assert artifact_versioning.construction_record_problem(workflow_path) is None
    assert warm_intent_heads(workflow_path) == CONTEXT_FOLDERS
    routers = {folder: CommandRouter(str(info / folder)) for folder in CONTEXT_FOLDERS}
    assert len({id(router.modelpipeline) for router in routers.values()}) == 1
    assert intent_detection._classifier_signal_version(str(info / "*"), "large") == (
        f"intent-classifier/{version}/global/large")

    app, cme = _session_on_a_book(workflow_path, "layout-e2e")
    try:
        assert intent_detector.global_head_trained(app) is True

        own = [(phrasing, name) for phrasing, name in _confident(cme, OWN_CANDIDATES, "Book")
               if name.startswith("Book/")]
        assert own, "the Book scope named a Book command for none of the probed phrasings"
        phrasing, expected = own[0]
        out = _route(cme, phrasing)
        assert out.command_response.artifacts["command_name"] == expected
        assert "ran_on_context_name" not in out.command_response.artifacts

        # Escalation: typed on the Book, declined there, named by the Library scope and run on it.
        declined = [
            candidate for candidate in PARENT_CANDIDATES
            if [str(label) for label in routers["Book"].predict(candidate)] == ["wildcard"]
        ]
        parent = [(phrasing, name) for phrasing, name in _confident(cme, declined, "Library")
                  if name in PARENT_COMMANDS]
        assert parent, (f"no probed Library request was both declined by the Book scope "
                        f"({declined}) and named by the Library scope")
        asked = _spy_on_the_heads_asked(monkeypatch)
        phrasing, expected = parent[0]
        app.record = app.root.books["b1"]
        out = _route(cme, phrasing)
        assert out.command_response.artifacts["command_name"] == expected
        assert out.command_response.artifacts["ran_on_context_name"] == "Library"
        assert asked.index("Book") < asked.index("Library")
    finally:
        app.close()  # the root session; the CME workflow is its child and closes with it

    # ------------------------------------------------------------------ 4. a second train is a no-op
    assert _train(workflow_path, layout.WORKFLOW) == version
    assert [info.version_id for info in artifact_versioning.list_versions(workflow_path)] == [version]

    # ------------------------------------------------------------------ 5. to the per-context layout
    _forget_loaded_routers(workflow_path)
    per_context_version = _train(workflow_path, layout.PER_CONTEXT)

    assert per_context_version != version
    new_vdir = artifact_versioning.version_dir(workflow_path, per_context_version)
    assert sorted(entry.name for entry in new_vdir.iterdir() if entry.is_dir()) == CONTEXT_FOLDERS
    for folder in CONTEXT_FOLDERS:
        assert sorted(os.listdir(new_vdir / folder)) == sorted(st.REQUIRED_CONTEXT_ARTIFACTS), folder
    new_manifest = artifact_versioning.read_manifest(workflow_path, per_context_version)
    assert layout.MANIFEST_KEY not in new_manifest
    assert sorted(new_manifest["contexts_retrained"]) == ["*", "Book", "Library", "Member", "Record"]
    assert new_manifest["contexts_carried_forward"] == []
    assert "intent_model_layout" not in json.loads(
        st.signature_path(workflow_path, per_context_version).read_text())
    assert is_workflow_trained(workflow_path) == (True, [])
    # The workflow-layout version is the one recovery point kept.
    assert sorted(info.version_id for info in artifact_versioning.list_versions(workflow_path)) == sorted(
        [version, per_context_version])
    router = CommandRouter(str(info / "Book"))
    assert getattr(router, "_workflow_router", None) is None
    assert router.tiny_path and os.path.isdir(router.tiny_path)
    labels, details = router.predict_with_details("show me this book")
    assert details["model_tier"] in ("tiny", "large") and "p_in" not in details
    assert "Book/show_book" in [str(label) for label in labels]
    _forget_loaded_routers(workflow_path)


#: An encoder of another family (mean pooling) for the change-of-encoder step. Small, and read
#: with the pooling its family default gives.
OTHER_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def test_train_and_run_in_the_default_embedding_layout(
        tmp_path, initialized_fastworkflow, monkeypatch, no_planner_key, capsys):
    from fastworkflow.model_pipeline_training import (
        CommandRouter,
        is_workflow_trained,
        warm_intent_heads,
    )
    from fastworkflow.train import embedding_head, workflow_model

    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets package not installed; intent-detection training is skipped.")
    cache_root = LIBRARY / "___command_info" / utterance_cache.CACHE_DIRNAME
    if not cache_root.is_dir() or not any(cache_root.glob("*.json")):
        pytest.skip(f"the library example has no generated-utterance cache at {cache_root} (local, "
                    f"untracked training output), so training a copy would call a model")
    try:
        embedding_head.SentenceEncoder.get(layout.DEFAULT_ENCODER)
    except Exception as error:  # noqa: BLE001 - no network and no cached copy of the encoder
        pytest.skip(f"the default sentence encoder {layout.DEFAULT_ENCODER} cannot be loaded here "
                    f"({type(error).__name__}: {error})")

    workflow_path = _copy_library(tmp_path / "library")
    fastworkflow.init(env_vars=_training_env(workflow_path, None))
    _rekey_parameter_examples(workflow_path)
    info = Path(workflow_path) / "___command_info"

    # ------------------------------------------------------------------ 1. train with no layout setting
    assert layout.SETTING not in fastworkflow._env_vars and layout.SETTING not in os.environ
    capsys.readouterr()
    # A cold cache is a failure here, not a skip and not a model call: `no_model_calls` raises.
    version = _train(workflow_path, None)
    printed = capsys.readouterr().out
    vdir = artifact_versioning.version_dir(workflow_path, version)

    # `fastworkflow train` says which layout and encoder it used and how long the head took.
    assert f"Intent model layout: embedding (frozen encoder {layout.DEFAULT_ENCODER}, mean pooling" in printed
    assert f"Intent model: layout embedding, encoder {layout.DEFAULT_ENCODER}" in printed
    assert "The head took " in printed and " bytes." in printed

    assert sorted(entry.name for entry in vdir.iterdir() if entry.is_dir()) == sorted(
        [*CONTEXT_FOLDERS, layout.WORKFLOW_MODEL_DIRNAME])
    assert sorted(os.listdir(vdir / layout.WORKFLOW_MODEL_DIRNAME)) == ["head.npz", "rows.npz", "scopes.json"]
    for folder in CONTEXT_FOLDERS:
        assert sorted(os.listdir(vdir / folder)) == ["threshold.json", "workflow_scope.json"], folder
        assert os.path.realpath(info / folder) == os.path.realpath(vdir / folder)
        assert st.context_artifacts_complete(workflow_path, version, "*" if folder == "global" else folder)
    assert not (info / layout.WORKFLOW_MODEL_DIRNAME).exists()
    assert json.loads((vdir / "global" / "workflow_scope.json").read_text()) == {
        "layout": "embedding", "model": "../workflow", "context": "*"}

    scopes = json.loads((vdir / layout.WORKFLOW_MODEL_DIRNAME / "scopes.json").read_text())
    assert scopes["layout"] == "embedding" and scopes["encoder"]["model"] == layout.DEFAULT_ENCODER
    assert sorted(scopes["scopes"]) == ["*", "Book", "Library", "Member", "Record"]
    assert "wildcard" in scopes["labels"] and "parameter_value" in scopes["labels"]
    # The rows are the rows the workflow layout assembles from the same caches, all of them.
    rebuilt = workflow_model.workflow_rows(workflow_path)
    assert scopes["labels"] == rebuilt.labels
    assert {name: entry["mask"] for name, entry in scopes["scopes"].items()} == {
        name: rebuilt.mask_of(name) for name in rebuilt.scopes}
    # The rows are those rows plus each command's name and description rows; seeds are weighted.
    assert scopes["rows"]["own"] == len(rebuilt.rows) and scopes["rows"]["description"] > 0
    assert scopes["rows"]["total"] == scopes["rows"]["own"] + scopes["rows"]["description"]
    assert scopes["rows"]["seeds_weighted"] > 0 and scopes["rows"]["seed_weight"] == 3.0
    assert scopes["schema_version"] == 2 and scopes["encoder"]["prefix"] == "query: "
    assert scopes["threshold"] == layout.DEFAULT_COMMIT_THRESHOLD == 0.65
    assert "margin" not in scopes and "calibration" not in scopes
    cross_fit = scopes["cross_fit"]
    assert cross_fit["rows"] > 500 and cross_fit["accuracy"] > 0.9
    assert cross_fit["folds"] == 5 and cross_fit["cross_fitted"] is True
    assert cross_fit["pairs_over_limit"] <= cross_fit["confused_pairs"] and len(cross_fit["pairs"]) <= 10
    # Seconds, and tens of kilobytes.
    assert scopes["fit_seconds"] < 30 and scopes["train_seconds"] < 120
    assert embedding_head.read_summary(str(vdir / layout.WORKFLOW_MODEL_DIRNAME))["bytes"] < 500_000

    # The calibration facts are in the version manifest and in the held-out report.
    manifest = artifact_versioning.read_manifest(workflow_path, version)
    assert manifest[layout.MANIFEST_KEY] == layout.EMBEDDING
    assert manifest["contexts"] == CONTEXT_FOLDERS
    assert sorted(manifest["contexts_retrained"]) == ["*", "Book", "Library", "Member", "Record"]
    recorded = manifest["intent_embedding_head"]
    assert recorded["encoder"] == layout.DEFAULT_ENCODER and recorded["pooling"] == "mean"
    assert recorded["threshold"] == scopes["threshold"] and "margin" not in recorded
    assert recorded["cross_fit"]["rows"] == cross_fit["rows"]
    assert recorded["cross_fit"]["accuracy"] == cross_fit["accuracy"]
    assert recorded["fit_seconds"] == scopes["fit_seconds"] and recorded["bytes"] > 0
    signature = st.load_training_signature(workflow_path, version)
    assert signature.intent_model_layout == layout.EMBEDDING
    assert json.loads(signature.embedding_head)["encoder"] == layout.DEFAULT_ENCODER

    heldout = json.loads((info / "heldout_evaluation.json").read_text())
    assert sorted(entry["context"] for entry in heldout["contexts"]) == [
        "*", "Book", "Library", "Member", "Record"]
    for entry in heldout["contexts"]:
        facts = entry["large_ambiguity"]
        assert facts["intent_model_layout"] == "embedding" and facts["encoder"] == layout.DEFAULT_ENCODER
        assert facts["threshold"] == scopes["threshold"] and "escalation_margin" not in facts
        assert facts["rule"] == "fixed_commit_threshold"
        assert facts["cross_fit"]["rows"] == cross_fit["rows"]
        assert not any("held-out evaluation failed" in note for note in entry["notes"])
    assert (info / "training_report.json").is_file()
    report = json.loads((info / "training_report.json").read_text())["intent_model"]
    assert report["layout"] == "embedding" and report["encoder"] == layout.DEFAULT_ENCODER
    assert report["accuracy"] == cross_fit["accuracy"] and report["commit_threshold"] == 0.65
    assert report["pairs_over_limit"] == cross_fit["pairs_over_limit"] and len(report["pairs"]) <= 10
    assert "LOCAL INTENT MODEL" in (info / "training_report.txt").read_text()
    provenance = json.loads((info / "training_provenance.json").read_text())
    assert set(provenance["context_training"]) == {"*", "Book", "Library", "Member", "Record"}

    # ------------------------------------------------------------------ 2. the runtime
    assert is_workflow_trained(workflow_path) == (True, [])
    assert artifact_versioning.construction_record_problem(workflow_path) is None
    assert warm_intent_heads(workflow_path) == CONTEXT_FOLDERS
    routers = {folder: CommandRouter(str(info / folder)) for folder in CONTEXT_FOLDERS}
    assert len({id(router.modelpipeline) for router in routers.values()}) == 1
    assert all(isinstance(router._workflow_router, embedding_head.EmbeddingHeadRouter)
               for router in routers.values())
    assert intent_detection._classifier_signal_version(str(info / "*"), "embedding") == (
        f"intent-classifier/{version}/global/embedding")
    from fastworkflow.intent import intent_detector

    assert intent_detector.engine_choice(workflow_path, "Book") == "classifier"       # no planner key
    assert intent_detector.startup_report(workflow_path) == ([], [
        "Intent detection: the trained local model decides when it is confident; otherwise the user or the "
        "agent is asked (JEV_API_KEY is not set, so there is no remote decision model)."])

    app, cme = _session_on_a_book(workflow_path, "embedding-e2e")
    try:
        assert intent_detector.global_head_trained(app) is True

        own = [(phrasing, name) for phrasing, name in _confident(cme, OWN_CANDIDATES, "Book")
               if name.startswith("Book/")]
        assert own, "the Book scope named a Book command for none of the probed phrasings"
        phrasing, expected = own[0]
        out = _route(cme, phrasing)
        assert out.command_response.artifacts["command_name"] == expected
        assert "ran_on_context_name" not in out.command_response.artifacts

        # Escalation: the Book head never passes a request up on its own. Typed on the Book, it is not confident
        # and hands the request on with the Library's command among the candidates; with no remote decision model
        # the question offers that command, and choosing it runs it on the Library.
        handed = []
        for candidate in PARENT_CANDIDATES:
            labels, details = routers["Book"].predict_with_details(candidate)
            if len(labels) > 1 and details["outside_labels"][:1] == ["Library/list_branches"]:
                handed.append(candidate)
        assert handed, "the Book head handed on none of the probed Library requests with list_branches first"
        asked = _spy_on_the_heads_asked(monkeypatch)
        app.record = app.root.books["b1"]
        question = _route(cme, handed[0])
        assert "ambiguous" in question.command_response.response.lower()
        assert "list_branches" in question.command_response.response
        assert "command_name" not in question.command_response.artifacts        # nothing ran yet
        out = _route(cme, "list_branches")
        assert out.command_response.artifacts["command_name"] == "Library/list_branches"
        assert out.command_response.artifacts["ran_on_context_name"] == "Library"
        # The Book head was asked and handed the request on; the pick is read from the question, so no other head is asked.
        assert asked and set(asked) <= {"*", "Book"} and "Book" in asked
    finally:
        app.close()  # the root session; the CME workflow is its child and closes with it

    # ------------------------------------------------------------------ 4. a second train is a no-op
    assert _train(workflow_path, None) == version
    assert _train(workflow_path, layout.EMBEDDING) == version                 # naming the default changes nothing
    assert [info.version_id for info in artifact_versioning.list_versions(workflow_path)] == [version]

    # ------------------------------------------------------------------ 5. another encoder refits the head
    try:
        embedding_head.SentenceEncoder.get(OTHER_ENCODER)
    except Exception:  # noqa: BLE001 - this step needs a second encoder; the rest stands without it
        _forget_loaded_routers(workflow_path)
        return
    _forget_loaded_routers(workflow_path)
    other_version = _train(workflow_path, None, {layout.ENCODER_SETTING: OTHER_ENCODER})

    assert other_version != version
    other_vdir = artifact_versioning.version_dir(workflow_path, other_version)
    other_scopes = json.loads((other_vdir / layout.WORKFLOW_MODEL_DIRNAME / "scopes.json").read_text())
    assert other_scopes["encoder"]["model"] == OTHER_ENCODER and other_scopes["encoder"]["pooling"] == "mean"
    assert other_scopes["encoder"]["prefix"] == ""                              # no prefix outside the e5 family
    other_manifest = artifact_versioning.read_manifest(workflow_path, other_version)
    assert sorted(other_manifest["contexts_retrained"]) == ["*", "Book", "Library", "Member", "Record"]
    assert other_manifest["contexts_carried_forward"] == []
    assert other_manifest["intent_embedding_head"]["encoder"] == OTHER_ENCODER
    router = CommandRouter(str(info / "Book"))
    assert router.modelpipeline.encoder.model_id == OTHER_ENCODER
    labels, details = router.predict_with_details("show me this book")
    assert details["model_tier"] == "embedding" and labels
    _forget_loaded_routers(workflow_path)
