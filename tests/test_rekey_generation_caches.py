"""``scripts/rekey_generation_caches.py``: the one-off re-key of the generated-data caches after the generator model
left their keys (bd fix-ygxo). Runs on a private copy of messaging_app_4 with cache files written in the shape the
caches had before (the model among the fingerprint inputs and in the key). No model is called."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.train import generate_synthetic as gs
from fastworkflow.train import not_here_pilot as pilot
from fastworkflow.train import param_example_cache as pc
from fastworkflow.train import utterance_cache as uc

ROOT = Path(__file__).resolve().parent.parent
MESSAGING = ROOT / "fastworkflow" / "examples" / "messaging_app_4"
SCRIPT = ROOT / "scripts" / "rekey_generation_caches.py"
OLD_MODEL = "mistral/the-model-that-wrote-it"


def _script():
    spec = importlib.util.spec_from_file_location("rekey_generation_caches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({"LLM_SYNDATA_GEN": "mistral/today-s-model"})
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def workflow(tmp_path) -> str:
    destination = tmp_path / "messaging_copy"
    shutil.copytree(MESSAGING, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    yield str(destination)
    RoutingRegistry.clear_registry()


def _write_old(root: str, fingerprint, entries: dict, model: str = OLD_MODEL) -> str:
    """A cache file as written before the model left the key: the model among the inputs, the key hashed with it."""
    inputs = {**fingerprint.inputs, "model": model}
    key = uc.variant_key_of(inputs)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{uc.slugify(fingerprint.command_name)}.{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cache_format_version": 1, "command_name": fingerprint.command_name, "variant_key": key,
                   "fingerprint_inputs": inputs, "entries": entries}, f, indent=2, sort_keys=True)
    return path


def _old_caches(folder: str) -> dict[str, int]:
    """Old-shape files for every command the trainer would generate for, keyed as the trainer keys them today
    except for the model. Returns how many files of each cache were written."""
    from fastworkflow.train.__main__ import (
        PARAM_EXAMPLE_COUNT, PARAM_VALIDATION_THRESHOLD, _get_commands_with_parameters, parameter_model_of)
    from fastworkflow.model_pipeline_training import get_route_layer_filepath_model
    from fastworkflow.train.generate_param_examples import extract_field_details, param_example_fingerprint

    cmd_dir = pilot.prepare_registry(folder)
    seed = str(gs.get_training_seed())
    counts = [gs._resolve_generation_count(None, name) for name in gs.GENERATION_COUNT_DEFAULTS]
    utterance_root = os.path.join(folder, "___command_info", uc.CACHE_DIRNAME)
    written = {"utterance": 0, "param": 0}
    for name in pilot.utterance_command_names(cmd_dir):
        seeds = list(cmd_dir.get_utterance_metadata(name).plain_utterances)
        fingerprint = gs.utterance_fingerprint(seeds, name, *counts)
        _write_old(utterance_root, fingerprint, {seed: {"seed": int(seed), "generated_utterances": [f"{name} one"]}})
        written["utterance"] += 1
    param_root = os.path.join(folder, "___command_info", pc.CACHE_DIRNAME)
    json_path = get_route_layer_filepath_model(folder, "command_directory.json")
    for name, metadata in sorted(_get_commands_with_parameters(json_path).items()):
        fields = parameter_model_of(folder, metadata)
        if fields is None:
            continue
        fingerprint = param_example_fingerprint(
            fields.model_fields, name, PARAM_EXAMPLE_COUNT, PARAM_VALIDATION_THRESHOLD,
            field_details=extract_field_details(fields.model_fields))
        if fingerprint is None:
            continue
        _write_old(param_root, fingerprint,
                   {seed: {"seed": int(seed), "valid_examples": [{"fields": {"command": name}, "inputs": ["command"]}]}})
        written["param"] += 1
    return written


def test_a_rekeyed_cache_is_a_hit_for_every_unchanged_command(workflow):
    written = _old_caches(workflow)
    assert written["utterance"] and written["param"]
    before = pilot.cache_report(workflow)
    assert before["utterance_cache"]["hits"] == 0 and before["param_example_cache"]["hits"] == 0

    result = _script().rekey_workflow(workflow, dry_run=False)

    assert result["utterance_cache"]["files"] == written["utterance"]
    assert result["param_example_cache"]["files"] == written["param"]
    after = pilot.cache_report(workflow)
    assert after["utterance_cache"]["hits"] == after["utterance_cache"]["commands"] == written["utterance"]
    assert after["param_example_cache"]["hits"] == written["param"]


def test_a_dry_run_counts_and_writes_nothing_and_a_second_run_is_a_no_op(workflow):
    written = _old_caches(workflow)
    root = Path(workflow) / "___command_info"
    snapshot = {p: p.read_bytes() for p in root.rglob("*.json")}

    dry = _script().rekey_workflow(workflow, dry_run=True)
    assert dry["utterance_cache"]["files"] == dry["utterance_cache"]["entries"] == written["utterance"]
    assert {p: p.read_bytes() for p in root.rglob("*.json")} == snapshot

    _script().rekey_workflow(workflow, dry_run=False)
    again = _script().rekey_workflow(workflow, dry_run=False)
    assert again["utterance_cache"]["files"] == 0
    assert again["utterance_cache"]["already_current"] == written["utterance"]


def test_two_files_that_differed_only_in_the_model_become_one_with_both_seeds(workflow):
    cmd_dir = pilot.prepare_registry(workflow)
    name = pilot.utterance_command_names(cmd_dir)[0]
    seeds = list(cmd_dir.get_utterance_metadata(name).plain_utterances)
    fingerprint = gs.utterance_fingerprint(seeds, name, 4, 5, 1)
    root = os.path.join(workflow, "___command_info", uc.CACHE_DIRNAME)
    _write_old(root, fingerprint, {"7": {"seed": 7, "generated_utterances": ["older"]},
                                   "42": {"seed": 42, "generated_utterances": ["older 42"]}}, model="a/one")
    newer = _write_old(root, fingerprint, {"42": {"seed": 42, "generated_utterances": ["newer 42"]}}, model="b/two")
    os.utime(newer, (os.path.getmtime(newer) + 10,) * 2)

    counts = _script().rekey_cache_dir(root, dry_run=False)

    assert (counts["files"], counts["merged"]) == (2, 1)
    cache = uc.UtteranceCache(workflow)
    assert cache.lookup(fingerprint, 7).generated_utterances == ["older"]
    assert cache.lookup(fingerprint, 42).generated_utterances == ["newer 42"]   # the newer file's entry wins
    assert [p for p in os.listdir(root) if p.endswith(".json")] == [os.path.basename(cache.entry_path(fingerprint))]
