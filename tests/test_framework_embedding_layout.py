"""bd fix-o5wi.3: the framework's own workflows (plan, command_metadata_extraction) use the embedding head.

Two kinds of test. The scratch ones train a private COPY of each workflow from its utterance cache with no model call
(a cache miss raises, it never generates) and read the head through the product router. The in-place ones read the
head that `fastworkflow train` published (local and gitignored; they carry `requires_plan_head` like the
other head-quality tests).

A copy is only the framework's workflow when `get_internal_workflow_path` says so: `CommandDirectory` skips the core
commands for the plan metadata extraction workflow by its real path, so a copy trained without that carried the core commands
(`IntentDetection/reset_context`, ...) in the review head's mask and in its candidate lists.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    BUILT_IN_VAGUE_REPLIES,
    BUILT_IN_VAGUE_WEIGHT,
    CommandRouter,
    train,
)
from fastworkflow.train import determinism, intent_model_layout as layout, utterance_cache
from fastworkflow.train import workflow_model as wm

PLAN_WORKFLOW = fastworkflow.get_internal_workflow_path("plan_metadata_extraction")
CME = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
REVIEW_COMMANDS = {f"ProposalReview/{name}" for name in ("add", "confirm", "decline", "drop", "edit", "look_up")}
CME_COMMANDS = {f"IntentDetection/{name}" for name in ("go_up", "reset_context", "what_can_i_do", "what_is_current_context")}
ENV_TEMPLATE = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "fastworkflow.env")


def _train_copy(source: str, name: str, tmp_path, monkeypatch) -> str:
    """Train *source*'s copy in the embedding layout, from its utterance cache, as the framework's workflow."""
    if not os.path.isdir(os.path.join(source, "___command_info", "utterance_cache")):
        pytest.skip(f"{name} has no utterance cache on this machine; nothing can be rebuilt without a model call")
    copy = str(tmp_path / "_workflows" / name)
    shutil.copytree(source, copy, symlinks=True,
                    ignore=shutil.ignore_patterns("versions", "___convo_info", "___workflow_contexts", "__pycache__"))
    info = os.path.join(copy, "___command_info")
    for entry in os.listdir(info):
        if os.path.islink(os.path.join(info, entry)) or entry in ("current.json", ".publication.lock"):
            os.remove(os.path.join(info, entry))
    env = {key: value for key, value in dotenv_values(ENV_TEMPLATE).items() if value is not None}
    env["LLM_SYNDATA_GEN"] = wm.cached_generation_model(copy) or env.get("LLM_SYNDATA_GEN", "")
    env.pop(layout.SETTING, None)
    fastworkflow.init(env_vars=env)
    RoutingRegistry.clear_registry()
    real = fastworkflow.get_internal_workflow_path
    monkeypatch.setattr(fastworkflow, "get_internal_workflow_path", lambda n: copy if n == name else real(n))
    cache = wm._ReadOnlyUtteranceCache(copy, mode=utterance_cache.MODE_REUSE)
    handle = fastworkflow.Workflow.create(copy, workflow_id_str=f"framework-embedding-{name}")
    previous = (determinism.get_provenance_recorder(), utterance_cache.get_utterance_cache())
    try:
        determinism.set_provenance_recorder(determinism.ProvenanceRecorder(copy))
        utterance_cache.set_utterance_cache(cache)
        with wm.no_model_calls():
            train(handle, intent_model_layout=layout.EMBEDDING)
    finally:
        determinism.set_provenance_recorder(previous[0])
        utterance_cache.set_utterance_cache(previous[1])
        handle.close()
    assert not cache.stats["miss"] and not cache.stats["stored"], cache.stats
    return copy


def _scopes(model_root: str) -> dict:
    with open(os.path.join(model_root, "workflow", "scopes.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def isolate_registry():
    yield
    RoutingRegistry.clear_registry()


def test_the_framework_layout_is_embedding():
    assert layout.FRAMEWORK_LAYOUT == layout.EMBEDDING
    for path in (PLAN_WORKFLOW, CME, fastworkflow.get_fastworkflow_package_path()):
        assert layout.is_framework_workflow(path)


@pytest.mark.usefixtures("isolate_registry")
def test_the_review_head_has_its_own_commands_and_the_vague_class(tmp_path, monkeypatch):
    copy = _train_copy(PLAN_WORKFLOW, "plan_metadata_extraction", tmp_path, monkeypatch)
    info = os.path.join(copy, "___command_info")
    scopes = _scopes(info)
    # the label space the per-context head had: the six review commands, no core command
    assert set(scopes["scopes"]["ProposalReview"]["mask"]) == REVIEW_COMMANDS
    assert set(scopes["labels"]) == REVIEW_COMMANDS | {"parameter_value", "wildcard"}
    assert layout.context_layout(os.path.join(info, "ProposalReview")) == layout.EMBEDDING
    router = CommandRouter(os.path.join(info, "ProposalReview"))
    for reply in ("hmm", "not sure", "I'm still undecided", "let me think about it", "blah"):
        labels, _details = router.predict_with_details(reply)
        assert labels == ["wildcard"], (reply, labels)
    for reply in ("yes, go ahead", "that plan works", "I'll pass"):
        labels, _details = router.predict_with_details(reply)
        assert set(labels) <= REVIEW_COMMANDS and "wildcard" not in labels, (reply, labels)
    # the built-in rows are in the head, as often as the per-context heads repeat them
    assert scopes["rows"]["total"] >= len(BUILT_IN_VAGUE_REPLIES) * BUILT_IN_VAGUE_WEIGHT


@pytest.mark.usefixtures("isolate_registry")
def test_the_cme_heads_keep_the_core_command_label_space(tmp_path, monkeypatch):
    copy = _train_copy(CME, "command_metadata_extraction", tmp_path, monkeypatch)
    info = os.path.join(copy, "___command_info")
    scopes = _scopes(info)
    assert {name: set(entry["mask"]) for name, entry in scopes["scopes"].items()} == {
        "*": CME_COMMANDS, "IntentDetection": CME_COMMANDS}
    for folder in ("global", "IntentDetection"):
        router = CommandRouter(os.path.join(info, folder))
        for reply, expected in (("go up", "go_up"), ("reset context", "reset_context"),
                                ("what can i do", "what_can_i_do"), ("where am i", "what_is_current_context")):
            labels, _details = router.predict_with_details(reply)
            assert labels == [f"IntentDetection/{expected}"], (folder, reply, labels)


@pytest.mark.requires_plan_head
def test_the_published_framework_heads_are_embedding_heads_with_their_own_masks():
    for root, expected in ((PLAN_WORKFLOW, {"ProposalReview": REVIEW_COMMANDS}),
                           (CME, {"IntentDetection": CME_COMMANDS, "*": CME_COMMANDS})):
        current = os.path.join(root, "___command_info")
        for context, mask in expected.items():
            folder = os.path.join(current, "global" if context == "*" else context)
            assert layout.context_layout(folder) == layout.EMBEDDING, folder
            assert layout.workflow_head_present(folder), folder
            assert set(_scopes(os.path.dirname(os.path.realpath(folder)))["scopes"][context]["mask"]) == mask
