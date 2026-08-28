"""fw.nlu.intent spans carry decision_uncertainty under real traced flows.

Complements ``tests/test_nlu_decision_signals.py``: that file proves the
assembling functions and static no-read guards; this one proves the structured
record actually lands on spans emitted by the real resolution path against the
trained ``hello_world`` workflow — no mocks at the NLU boundary.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import tracing
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    ESCALATION_OUTCOMES,
)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.decision_signals import DecisionUncertainty
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

HELLO_WORLD = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "hello_world"
)


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    env = dotenv_values("fastworkflow/examples/fastworkflow.env")
    fastworkflow.init(dict(env))
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    """Real TraceSink implementation — the pluggable seam the design defines."""

    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def hello_ctx():
    if not Path(HELLO_WORLD, "___command_info").is_dir():
        pytest.skip("hello_world is not trained on this machine")
    sink = RecordingTraceSink()
    wf = fastworkflow.Workflow.create(
        HELLO_WORLD,
        workflow_id_str=f"decsig-{uuid.uuid4().hex}",
        workflow_context={"run_as_agent": True},
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(wf)
    ctx._begin_turn("decision signal capture test")
    ctx.push_active_workflow(wf)
    yield ctx, sink
    ctx.pop_active_workflow()


def _signal(uncertainty: DecisionUncertainty, kind: str):
    matching = [s for s in uncertainty.signals if s.kind == kind]
    assert matching, f"no '{kind}' signal in {uncertainty!r}"
    return matching[0]


def test_exact_prefix_intent_span_carries_decision_uncertainty(hello_ctx):
    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(
        ctx,
        "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
    )
    span = sink.named(tracing.SPAN_NLU_INTENT)[-1]

    assert "decision_uncertainty" in span.attributes
    recorded = span.attributes["decision_uncertainty"]
    assert recorded["decision_kind"] == "command-identity"
    assert recorded["signals_absent_reason"] == "deterministic-resolution"
    assert DecisionUncertainty(**recorded).candidate_count == 1


def test_fuzzy_prematch_intent_span_carries_distance_and_threshold(hello_ctx):
    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(ctx, "what can i do")

    span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
    assert span.attributes["matcher_layer"] == "fuzzy_prematch"
    assert "decision_uncertainty" in span.attributes

    uncertainty = DecisionUncertainty(**span.attributes["decision_uncertainty"])
    assert _signal(uncertainty, "fuzzy-score").value == span.attributes["fuzzy_distance"]
    assert span.attributes["fuzzy_threshold"] is not None


def test_classifier_intent_span_carries_confidence_signal(hello_ctx):
    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(
        ctx, "please compute the total of five plus three"
    )

    spans = [
        s
        for s in sink.named(tracing.SPAN_NLU_INTENT)
        if s.attributes.get("matcher_layer") == "classifier"
    ]
    assert spans, "classifier layer never ran"

    span = spans[0]
    uncertainty = DecisionUncertainty(**span.attributes["decision_uncertainty"])
    confidence = _signal(uncertainty, "classifier-confidence")
    assert confidence.value == span.attributes["classifier"]["confidence"]
    assert confidence.signal_version == span.attributes["classifier_signal_version"]


def test_every_intent_span_carries_escalation_outcome(hello_ctx):
    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(ctx, "what can i do")
    CommandExecutor.invoke_command(
        ctx, "add_two_numbers <first_num>1</first_num><second_num>2</second_num>"
    )

    spans = sink.named(tracing.SPAN_NLU_INTENT)
    assert spans
    for span in spans:
        assert span.attributes["escalation_outcome"] in ESCALATION_OUTCOMES
        assert "decision_uncertainty" in span.attributes


def test_param_extraction_span_carries_slot_binding_uncertainty(hello_ctx):
    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(
        ctx,
        "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
    )

    span = sink.named(tracing.SPAN_NLU_PARAM_EXTRACTION)[-1]
    assert "decision_uncertainty" in span.attributes
    recorded = DecisionUncertainty(**span.attributes["decision_uncertainty"])
    assert recorded.decision_kind == "slot-binding"
    assert recorded.signals
    assert recorded.signals[0].value == span.attributes["extraction_method"]
