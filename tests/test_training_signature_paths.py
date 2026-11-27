"""The training signature does not depend on how the workflow path was spelled (bd fix-o5wi.6 follow-up).

A command's source path is recorded in the signature. Planning a workflow through a path relative to the current
directory used to bake that spelling in, so a later run from another directory saw a different signature (and, with the
file unreadable from there, a changed command). Real components: a temporary copy of the hello_world test workflow
(never trained, nothing under ``tests`` is written), planned from two directories through a relative path and through
the absolute one.
"""

from __future__ import annotations

import json
import os
import shutil

import fastworkflow
from fastworkflow.core.command_directory import CommandDirectory
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.train import selective_training as st
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

HELLO = os.path.join(os.path.dirname(__file__), "hello_world_workflow")


def _fresh_copy(tmp_path, name: str):
    """A new copy per spelling: the first plan persists ``command_directory.json``, which a second would read back."""
    workflow = tmp_path / name / "hello"
    shutil.copytree(HELLO, workflow, ignore=shutil.ignore_patterns("___command_info", "___convo_info", "__pycache__"))
    return workflow


def _signature(path: str) -> dict:
    # What `train_workflow` does first with the path it is given: build the command directory and persist it.
    CommandDirectory.load(path).save()
    RoutingRegistry._definitions.clear()          # a cached definition would hide the spelling of the first call
    signature, _unresolved = st.compute_training_signature(
        path, st.contexts_for_training(path), seed=42, intent_model_layout_name="embedding")
    return json.loads(signature.model_dump_json())


def test_the_same_workflow_has_one_signature_whatever_the_spelling_or_the_working_directory(
        initialized_fastworkflow, tmp_path, monkeypatch):  # noqa: F811
    absolute = _signature(str(_fresh_copy(tmp_path, "a")))
    paths = [fp["source_path"] for fp in absolute["command_fingerprints"].values() if fp.get("source_path")] \
        if "command_fingerprints" in absolute else []

    inside = _fresh_copy(tmp_path, "b")
    monkeypatch.chdir(inside.parent)
    from_parent = _signature("hello")
    inside = _fresh_copy(tmp_path, "c")
    monkeypatch.chdir(inside)
    from_inside = _signature(".")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    sibling = _fresh_copy(tmp_path, "d")
    monkeypatch.chdir(elsewhere)
    from_sibling = _signature(os.path.relpath(sibling, elsewhere))

    def normal(signature: dict, name: str) -> str:
        return json.dumps(signature, sort_keys=True).replace(str(tmp_path / name), "<root>")

    assert normal(from_parent, "b") == normal(absolute, "a")
    assert normal(from_inside, "c") == normal(absolute, "a")
    assert normal(from_sibling, "d") == normal(absolute, "a")
    assert all(os.path.isabs(path) for path in paths)


def test_the_fingerprints_do_not_depend_on_the_spelling_either(initialized_fastworkflow, tmp_path, monkeypatch):  # noqa: F811
    absolute_copy = _fresh_copy(tmp_path, "a")
    RoutingRegistry._definitions.clear()
    expected = st.compute_command_fingerprints(str(absolute_copy))
    relative_copy = _fresh_copy(tmp_path, "b")
    monkeypatch.chdir(relative_copy)
    CommandDirectory.load(".").save()
    RoutingRegistry._definitions.clear()
    found = st.compute_command_fingerprints(".")
    assert expected and all(fp.resolved and os.path.isabs(fp.source_path) for fp in found.values())
    assert {n: (f.source_sha256, f.seed_utterances_sha256) for n, f in found.items()} == \
        {n: (f.source_sha256, f.seed_utterances_sha256) for n, f in expected.items()}
