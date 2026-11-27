"""The split record names the splitter prompt that produced it (bd fix-qjne.26 stage 0; recording only).

``prompt_digest`` on the ``fw.split`` span (12 hex characters of the SHA-256 of the splitter signature's instructions and field
descriptions, and of the declared examples) and ``prompt_source`` (``built_in``, or ``built_in+workflow``
when the ``class Context`` of ``_commands/_workflow.py`` declares ``split_examples``; a replacement prompt, a ``Signature`` in that file, is a load
error, so the value ``workflow`` is never written), so later replays and rounds can tell which prompt produced a split. Real
components, no mocks: the planner's model is the scripted ``DummyLM`` the neighbouring split tests use.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from dspy.utils import DummyLM

from fastworkflow.observability import tracing
from fastworkflow import workflow_agent
from tests.test_approval_binding import (  # noqa: F401
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_plan_reply_policy import _wire_plan, planner, script_param_extraction  # noqa: F401
from tests.test_round1_gate import gated, no_draft_extraction  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.workflow_settings_file import write_workflow_file

REAL_SPLIT = workflow_agent.split_into_subtasks

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "no_planner_key")

LIBRARY = "fastworkflow/examples/library"
REQUEST = "Find the book Dune. Find the book Emma."
PARTS = ["Find the book Dune.", "Find the book Emma."]
OWN_EXAMPLES = '''
split_examples = [("Find the book Dune and list its loans.", ["Find the book Dune.", "List the loans of Dune."])]
'''
OWN_INSTRUCTIONS = '''
class Signature:
    """Split a library request into its independent jobs, one numbered line each."""
'''
OWN_FIELD_DESCRIPTION = '''
from pydantic import BaseModel, Field


class Signature:
    class Output(BaseModel):
        subtasks: str = Field(description="the jobs, one numbered line each")
'''
ONLY_PATTERNS = "import re\nidentifier_patterns = [re.compile(r'\\bM-[0-9]+')]\n"


def _lm(items: list[str]) -> DummyLM:
    text = "\n".join(f"{n}. {item}" for n, item in enumerate(items, 1))
    return DummyLM([{"reasoning": "r", "subtasks": text}])


def _workflow(folder: Path, split_file: str) -> str:
    write_workflow_file(folder / "_commands", split_file)
    return str(folder)


def _recorded(folder: str | None, conversation: str = "") -> dict:
    with workflow_agent.recording_split() as record:
        workflow_agent.split_into_subtasks(REQUEST, planner_lm=_lm(PARTS), workflow_folderpath=folder,
                                           conversation=conversation)
    return record


def test_a_workflow_without_a_split_file_uses_the_built_in_prompt():
    built_in = workflow_agent.split_prompt(None)
    assert built_in["source"] == workflow_agent.SPLIT_PROMPT_BUILT_IN == "built_in"
    assert re.fullmatch(r"[0-9a-f]{12}", built_in["digest"])
    assert workflow_agent.split_prompt(LIBRARY) == built_in
    record = _recorded(LIBRARY)
    assert (record["prompt_digest"], record["prompt_source"]) == (built_in["digest"], "built_in")


def test_a_split_file_with_its_own_examples_is_the_built_in_and_workflow_prompt(tmp_path):
    """A workflow changes the splitter's prompt with ``split_examples`` (the only way left): the source and digest say so."""
    folder = _workflow(tmp_path / "own", OWN_EXAMPLES)
    own = workflow_agent.split_prompt(folder)
    assert own["source"] == workflow_agent.SPLIT_PROMPT_BUILT_IN_AND_WORKFLOW == "built_in+workflow"
    assert own["digest"] != workflow_agent.split_prompt(None)["digest"]
    record = _recorded(folder)
    assert (record["prompt_digest"], record["prompt_source"]) == (own["digest"], "built_in+workflow")


@pytest.mark.parametrize("split_file", [OWN_INSTRUCTIONS, OWN_FIELD_DESCRIPTION, OWN_EXAMPLES + OWN_INSTRUCTIONS],
                         ids=["instructions", "field_description", "with_examples"])
def test_a_signature_in_the_split_file_fails_the_load_naming_split_examples(tmp_path, split_file):
    """The replacement prompt is retired: a ``Signature`` (instructions or a field description) is a load error."""
    from fastworkflow.planning.planning_commands import PlanningCommandFormatError

    folder = _workflow(tmp_path / "sig", split_file)
    with pytest.raises(PlanningCommandFormatError, match=r"_workflow\.py: a Signature .* is removed: write split_examples"):
        workflow_agent.split_prompt(folder)


def test_prompt_source_is_never_workflow(tmp_path):
    assert not hasattr(workflow_agent, "SPLIT_PROMPT_WORKFLOW")
    sources = {workflow_agent.split_prompt(None)["source"], workflow_agent.split_prompt(LIBRARY)["source"],
               workflow_agent.split_prompt(_workflow(tmp_path / "p", ONLY_PATTERNS))["source"],
               workflow_agent.split_prompt(_workflow(tmp_path / "e", OWN_EXAMPLES))["source"]}
    assert sources == {"built_in", "built_in+workflow"}


def test_a_split_file_with_only_identifier_patterns_keeps_the_built_in_prompt(tmp_path):
    folder = _workflow(tmp_path / "patterns", ONLY_PATTERNS)
    assert workflow_agent.split_prompt(folder) == workflow_agent.split_prompt(None)


def test_the_digest_is_of_the_prompt_text_only(tmp_path):
    """The same settings file in two workflows gives one digest, and a turn's earlier conversation is not part of it."""
    first = _workflow(tmp_path / "a", OWN_EXAMPLES)
    second = _workflow(tmp_path / "b", OWN_EXAMPLES)
    assert workflow_agent.split_prompt(first) == workflow_agent.split_prompt(second)
    assert _recorded(first, conversation="Earlier: a turn about Dune.")["prompt_digest"] == \
        workflow_agent.split_prompt(first)["digest"]


def test_the_span_attribute_carries_the_prompt_and_the_contract_says_so():
    assert tracing.SPAN_CONTRACT_VERSION >= 61
    assert {"prompt_digest", "prompt_source"} <= tracing.SPAN_CONTRACTS[tracing.SPAN_SPLIT].attributes
    attribute = tracing.split_stage_attributes(_recorded(LIBRARY))
    assert attribute["prompt_digest"] == workflow_agent.split_prompt(None)["digest"]
    assert attribute["prompt_source"] == "built_in"
    no_prompt = tracing.split_stage_attributes({"request": "r", "raw": [], "merged": ["r"], "attempts": 1})
    assert no_prompt["prompt_digest"] is None and no_prompt["prompt_source"] is None


def test_a_split_turn_carries_the_prompt_on_its_span(gated, monkeypatch, no_draft_extraction):
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", REAL_SPLIT)
    gated.ctx._current_planner_lm = _lm(PARTS)
    gated.script.steps = _finish() * 4
    gated.ctx.process_turn(REQUEST, unattended=True)
    recorded = [span for span in gated.sink.spans if span.name == tracing.SPAN_SPLIT][-1].attributes
    assert recorded["prompt_source"] == "built_in"
    assert recorded["prompt_digest"] == workflow_agent.split_prompt(None)["digest"]   # no shipped workflow has a _workflow.py
    json.dumps(recorded)
