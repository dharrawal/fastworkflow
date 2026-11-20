"""The held-out plan-intent fixture (``tests/planning_test_sets/library_plan_intent_heldout.json``, bd fix-o1lg S0)
holds what its notes promise: enough requests per group, names that exist in the library example, and no
request that is a seed utterance of a library command or a case of the other library test sets.

No model is trained and no model is called: the seeds are read from a private copy's command files.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.planning_commands import command_utterances, resolve_field_sources

LIBRARY = os.path.join("fastworkflow", "examples", "library")
SETS = os.path.join("tests", "planning_test_sets")
FIXTURE = os.path.join(SETS, "library_plan_intent_heldout.json")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(FIXTURE) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def library_copy(tmp_path_factory) -> str:
    fastworkflow.init({})
    destination = tmp_path_factory.mktemp("plan_intent_fixture") / "library"
    shutil.copytree(LIBRARY, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    yield str(destination)
    RoutingRegistry.clear_registry()


def _own_planning(folder: str) -> set[str]:
    """The library's own planning commands (the framework's built-in ones have no file in its ``_commands``)."""
    names = resolve_field_sources(folder).planning_commands
    return {name for name in names if os.path.exists(os.path.join(folder, "_commands", f"{name}.py"))}


def _texts(fixture: dict) -> list[str]:
    return [case["utterance"] for case in fixture["cases"]] + [
        text for request in fixture["two_part"] for text in (request["request"], *(p["utterance"] for p in request["parts"]))]


def test_every_group_has_enough_requests(fixture, library_copy):
    planning = _own_planning(library_copy)
    per_plan = collections.Counter(case["expected"] for case in fixture["cases"] if case["group"] == "plan")
    assert set(per_plan) == planning
    assert min(per_plan.values()) >= 6
    groups = collections.Counter(case["group"] for case in fixture["cases"])
    assert groups["execution"] >= 20
    assert groups["none"] >= 10
    assert groups["resembles_plan"] >= 10
    assert len(fixture["two_part"]) >= 12
    assert all(len(request["parts"]) == 2 for request in fixture["two_part"])


def test_expected_names_are_real_commands_of_the_right_kind(fixture, library_copy):
    planning = _own_planning(library_copy)
    crd = RoutingRegistry.get_definition(library_copy)
    execution = {name for names in crd.contexts.values() for name in names} - planning
    parts = list(fixture["cases"]) + [p for request in fixture["two_part"] for p in request["parts"]]
    for part in parts:
        if part["expect"] == "plan":
            assert part["expected"] in planning, part
        elif part["expect"] == "execution":
            assert part["expected"] in execution, part
        else:
            assert part["expect"] == "none" and part["expected"] is None, part


def test_disjoint_from_seeds_and_the_other_library_sets(fixture, library_copy):
    seeds = {_norm(text) for texts in command_utterances(library_copy).values() for text in texts}
    assert seeds, "no seed utterance was read"
    others: set[str] = set()
    for name in ("library.json", "library_handwritten.json"):
        with open(os.path.join(SETS, name)) as handle:
            others |= {_norm(case["utterance"]) for case in json.load(handle)["cases"]}
    texts = [_norm(text) for text in _texts(fixture)]
    assert not (set(texts) & seeds)
    assert not (set(texts) & others)
    singles = [_norm(case["utterance"]) for case in fixture["cases"]]
    assert len(singles) == len(set(singles)), "a request is repeated"
