"""Assistant mode binds a plain ``/command name=value`` form in code before the LLM (fix-weih.1).

Real workflow (``tests/plain_required_workflow``), real ``ParameterExtraction``. The only double
is the LLM extractor, patched at the seam a live model would answer.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.utils.signatures import InputForParamExtraction

FOLDER = str(Path(__file__).parent.joinpath("plain_required_workflow").resolve())
RECORD = "record_values"
FULL = f'{RECORD}: {RECORD} name=Bob count=3 ratio=1.5 ids=["a","b"]'


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def workflow():
    fastworkflow.RoutingRegistry.get_definition(FOLDER, load_cached=False)
    return fastworkflow.Workflow.create(FOLDER, workflow_id_str=f"kv_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def llm():
    """Patch the LLM extractor; ``calls`` counts runs. ``answer`` is what it returns."""
    state = {"calls": 0, "answer": None}

    def fake(self, cls, command_name, folder):
        state["calls"] += 1
        if state["answer"] is None:
            raise AssertionError("the LLM ran")
        return cls.model_construct(**state["answer"])

    with patch.object(InputForParamExtraction, "extract_parameters", fake):
        yield state


def _run(wf, text, cme_context, run_as_agent=False):
    if run_as_agent:
        c = wf.context
        c["run_as_agent"] = True
        wf.context = c
    diagnostics: dict = {}
    out = ParameterExtraction(
        SimpleNamespace(context=dict(cme_context)), wf, RECORD, text)._extract_impl(diagnostics)
    return out, diagnostics


def _assert_bound(out):
    assert out.parameters_are_valid is True
    p = out.cmd_parameters
    assert (p.name, p.count, p.ratio, p.ids) == ("Bob", 3, 1.5, ["a", "b"])


def test_slash_command_in_agent_session_binds_in_code(workflow, llm):
    out, diag = _run(workflow, FULL, {"is_assistant_mode_command": True}, run_as_agent=True)
    _assert_bound(out)
    assert llm["calls"] == 0
    assert diag["extraction_method"] == "xml_regex"
    assert diag["deterministic_form"] == "key_value"


def test_assistant_session_binds_in_code(workflow, llm):
    out, diag = _run(workflow, FULL, {})
    _assert_bound(out)
    assert llm["calls"] == 0
    assert diag["extraction_method"] == "xml_regex"
    assert diag["deterministic_form"] == "key_value"


def test_resent_full_command_with_leading_slash_binds_in_code(workflow, llm):
    out, _ = _run(workflow, f'{RECORD}: /{RECORD} name=Bob count=3 ratio=1.5 ids=["a","b"]', {})
    _assert_bound(out)
    assert llm["calls"] == 0


def test_partial_command_asks_for_the_missing_fields(workflow, llm):
    out, _ = _run(workflow, f"{RECORD}: {RECORD} name=Bob count=3", {})
    assert out.parameters_are_valid is False
    assert "ratio" in out.error_msg and "ids" in out.error_msg
    assert llm["calls"] == 0


def test_natural_language_falls_back_to_the_llm(workflow, llm):
    llm["answer"] = {"name": "Bob"}
    out, diag = _run(workflow, f"{RECORD}: /{RECORD} for Bob", {})
    assert llm["calls"] == 1
    assert diag["extraction_method"] == "llm"
    assert out.parameters_are_valid is False


def test_agent_mode_is_unchanged(workflow, llm):
    out, diag = _run(workflow, FULL, {}, run_as_agent=True)
    _assert_bound(out)
    assert llm["calls"] == 0
    assert diag["deterministic_form"] == "key_value"
