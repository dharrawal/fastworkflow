"""The parameter form's replies, bound by the real server side (fix-weih.5).

1. Reply contract: ``tests/fixtures/parameter_form_replies.json`` lists the exact text the chat form
   composes (``tmComposeCommandReply``, asserted in ``chatbot_parameter_form_dom.cjs``) and what the
   real ``ParameterExtraction`` binds from it. A case with ``"binds_in_code": false`` is a value the
   parser does not bind in code: it reaches the LLM extractor (so it costs a model call).
2. Round trips through the real ``WorkflowExecutionContext.process_turn``: Run, Cancel ("abort")
   in an assistant session and in an agent session. The LLM extractor and the agent entry points
   fail loudly if reached.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import fastworkflow
from fastworkflow import intent_loop
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

TESTS = Path(__file__).parent
FOLDER = str((TESTS / "plain_required_workflow").resolve())
CASES = json.loads((TESTS / "fixtures" / "parameter_form_replies.json").read_text(encoding="utf-8"))
RECORD = "record_values"


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class LlmSeam:
    """The LLM extractor: raises unless a case allows it, and counts calls."""

    def __init__(self, allowed: bool = False):
        self.allowed = allowed
        self.calls = 0

    def __call__(self, cls, command_name, folder):
        self.calls += 1
        if not self.allowed:
            raise AssertionError("the LLM ran")
        return cls.model_construct()


# ---------------------------------------------------------------------------
# 1. The reply contract
# ---------------------------------------------------------------------------


def test_the_fixture_covers_the_required_inputs():
    names = {case["name"] for case in CASES}
    for needle in ("plain", "parenthesis", "double_quote_and_parenthesis", "trailing_comma", "semicolon",
                   "multi_line", "None", "wrapped_in_quotes", "ids_with_a_space_and_a_comma", "negative"):
        assert any(needle in name for name in names), needle


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_reply_the_form_sends_binds_as_expected(case):
    RoutingRegistry.get_definition(FOLDER, load_cached=False)
    workflow = fastworkflow.Workflow.create(FOLDER, workflow_id_str=f"rc_{uuid.uuid4().hex[:8]}")
    in_code = case.get("binds_in_code", True)
    seam = LlmSeam(allowed=not in_code)
    with patch.object(InputForParamExtraction, "extract_parameters", lambda _self, *a: seam(*a)):
        out = ParameterExtraction(
            SimpleNamespace(context={"is_assistant_mode_command": True}), workflow, RECORD,
            f"{RECORD}: {case['reply']}")._extract_impl({})
    if not in_code:
        assert seam.calls == 1, "a value the parser does not bind reaches the LLM extractor"
        return
    assert seam.calls == 0
    assert out.parameters_are_valid is True, out.error_msg
    for field, value in case["expected"].items():
        assert getattr(out.cmd_parameters, field) == value, field


# ---------------------------------------------------------------------------
# 2. Round trips through WorkflowExecutionContext
# ---------------------------------------------------------------------------

PARTIAL = "/record_values name=Bob count=3"
RUN = '/record_values name=Bob count=3 ratio=1.5 ids=["a","b"]'
ZED = '/record_values name=Zed count=1 ratio=2.0 ids=["z"]'


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A factory ``make(run_as_agent)`` -> ctx on a copy of the workflow; the agent entry points raise."""
    folder = tmp_path / "plain_required_workflow"
    shutil.copytree(
        FOLDER, folder,
        ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__"))
    RoutingRegistry.get_definition(str(folder), load_cached=False)
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context",
                        lambda self: (SimpleNamespace(model="test-model"), None))
    seam = LlmSeam()
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", lambda _self, *a: seam(*a))

    def make(run_as_agent: bool):
        ctx = WorkflowExecutionContext(run_as_agent=run_as_agent)
        ctx.bind_app_workflow(fastworkflow.Workflow.create(
            str(folder), workflow_id_str=f"rt_{uuid.uuid4().hex[:8]}"))
        invoked = []

        def agent_entry(*args, **kwargs):
            invoked.append(args)
            raise AssertionError("the agent was invoked")

        monkeypatch.setattr(ctx, "_process_agent_message", agent_entry)
        monkeypatch.setattr(ctx, "_resume_agent_message", agent_entry)
        ctx.agent_invoked = invoked
        return ctx

    make.llm = seam
    return make


def _last(out):
    return out.command_outputs[-1].command_response


def _ran(out) -> list[str]:
    """The summaries of ``record_values`` runs in the turn (a run answers ``name=...; count=...``)."""
    return [o.command_response.response for o in out.command_outputs
            if o.command_response.success and o.command_response.response.startswith("name=")]


def _no_pending_round(ctx):
    context = ctx._cme_workflow.context
    assert intent_loop.waiting_for(context) is None


@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
def test_run_binds_the_composed_reply_and_the_turn_carries_no_request(session, run_as_agent):
    ctx = session(run_as_agent)
    first = ctx.process_turn(PARTIAL)
    request = _last(first).artifacts["parameter_request"]
    assert set(request["missing"]) == {"ratio", "ids"}
    assert request["values"] == {"name": "Bob", "count": 3}
    assert _ran(first) == []

    done = ctx.process_turn(RUN)
    assert _ran(done) == ["name=Bob; count=3; ratio=1.5; ids=['a', 'b']"]
    assert "parameter_request" not in _last(done).artifacts
    assert session.llm.calls == 0
    assert ctx.agent_invoked == []
    _no_pending_round(ctx)


@pytest.mark.parametrize("run_as_agent", [False, True], ids=["assistant_session", "agent_session"])
def test_cancel_abandons_the_pending_round_and_the_next_command_starts_clean(session, run_as_agent):
    ctx = session(run_as_agent)
    first = ctx.process_turn(PARTIAL)
    assert "parameter_request" in _last(first).artifacts

    cancelled = ctx.process_turn("abort")
    assert _last(cancelled).response == "command aborted\n"
    assert _ran(cancelled) == []
    assert "parameter_request" not in _last(cancelled).artifacts
    assert ctx.agent_invoked == [], "a bare 'abort' stays on the assistant path"
    _no_pending_round(ctx)

    fresh = ctx.process_turn(ZED)
    assert _ran(fresh) == ["name=Zed; count=1; ratio=2.0; ids=['z']"]
    assert "Bob" not in _last(fresh).response
    assert session.llm.calls == 0
    assert ctx.agent_invoked == []
