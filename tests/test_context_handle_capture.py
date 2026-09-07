"""Context-before/after handles and consequence on executed commands.

Architecture §12.0 deltas 2 and 4, §6.6.1, §6.7, and FW-REQ-002's acceptance
criteria — specifically the two that name this behavior directly:

* "An authorized navigation command records distinct context-before and
  context-after handles."
* "A non-navigation command records identical context-before and context-after
  handles."

Before this, `fw.command.execute` captured context only *after* execution (the
span's `context` field, set at close from `command_output.context`) and there was
no context-before anywhere, so a record could not say whether a command had moved
the workflow at all.

Two things these tests are careful about, because both are ways the capture can
look right and mean nothing:

**The handles are type-only, and that is asserted rather than glossed.** §6.7's
concrete handle needs an instance key, and fastWorkflow has no framework-level
identity for a context instance — the current context is an arbitrary application
object whose only framework-visible identity is its class. A test that passed
because handles happened to differ would hide that; `test_the_handle_is_type_only`
pins the degradation so the day a real projector lands, it fails and gets updated.

**Unknown must not read as cheap.** An undeclared command's effect contract is
`unknown`, which §6.6.1 requires be treated as write-capable and floored at high
consequence. The failure mode is silent: `read_only` would produce a clean-looking
row that under-reports every command in every workflow without a manifest, which
is nearly all of them.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.capture_policy import HMAC_KEY_VAR
from fastworkflow.runtime_manifest import (
    CommandDeclaration,
    EffectContract,
    RuntimeManifest,
    clear_runtime_metadata,
    get_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

# Moves the workflow's command context from TodoListManager down to the created
# TodoList (create_todo_list.py line 55).
NAVIGATING_COMMAND = "TodoListManager/create_todo_list"

# Reads and returns; never touches current_command_context.
NON_NAVIGATING_COMMAND = "TodoListManager/list_todo_lists"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"handles-{uuid.uuid4().hex}",
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _last_tool_call(sink: RecordingTraceSink) -> tracing.Span:
    return sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1]


# ----------------------------------------------------------------------
# FW-REQ-002 acceptance criteria
# ----------------------------------------------------------------------


def test_a_navigation_command_records_distinct_handles(ctx, sink):
    """create_todo_list descends TodoListManager -> TodoList."""
    ctx.process_action_turn(_action(NAVIGATING_COMMAND, description="groceries"))

    span = _last_tool_call(sink)
    before = span.attributes[tracing.ATTR_CONTEXT_BEFORE]
    after = span.attributes[tracing.ATTR_CONTEXT_AFTER]

    assert before["context_type"] == "TodoListManager"
    assert after["context_type"] == "TodoList"
    assert before["handle_id"] != after["handle_id"]


def test_a_non_navigation_command_records_identical_handles(ctx, sink):
    """list_todo_lists reads and returns; the workflow does not move.

    Compared on `handle_id`, which is the handle's identity. `issued_at` differs
    by construction — the two handles are projected at different instants, and
    recording the same timestamp for both would be a lie about when the after
    handle was taken.
    """
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    span = _last_tool_call(sink)
    before = span.attributes[tracing.ATTR_CONTEXT_BEFORE]
    after = span.attributes[tracing.ATTR_CONTEXT_AFTER]

    assert before["handle_id"] == after["handle_id"]
    assert before["context_type"] == after["context_type"] == "TodoListManager"
    assert before["issued_at"] != after["issued_at"]


def test_the_prose_path_records_handles_too(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """FW-REQ-002 clause 5: capture semantics are shared across paths.

    The CME hop is stood in for because the test workflow ships no trained intent
    models; the dispatch it stands in for is the real `perform_action`, so the
    handle projection under test runs for real.
    """
    from fastworkflow.command_executor import CommandExecutor

    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"handles-prose-{uuid.uuid4().hex}"
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, wf, action):
        command_output = real_perform_action(
            workflow, _action(NAVIGATING_COMMAND, description="groceries")
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = NAVIGATING_COMMAND
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))

    try:
        context.process_turn("make me a grocery list")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert execute.attributes[tracing.ATTR_CONTEXT_BEFORE]["context_type"] == (
            "TodoListManager"
        )
        assert execute.attributes[tracing.ATTR_CONTEXT_AFTER]["context_type"] == (
            "TodoList"
        )
    finally:
        with suppress(Exception):
            context.close()


# ----------------------------------------------------------------------
# What the handle is, and what it is not (§6.7)
# ----------------------------------------------------------------------


def test_the_handle_is_type_only(ctx, sink):
    """§6.7 feature-off legacy behavior, pinned so it cannot be assumed away.

    A type-only handle says which context type was active and nothing about which
    instance, so it cannot contribute to G2A/G2B. This test fails the day a real
    projector lands, which is the point: the claim "we capture context identity"
    must not be able to become true by accident.
    """
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    handle = _last_tool_call(sink).attributes[tracing.ATTR_CONTEXT_BEFORE]
    assert handle["instance_fingerprint"] is None
    assert handle["hmac_key_version"] is None
    assert handle["projector_id"] == tracing.CONTEXT_PROJECTOR_ID


def test_an_hmac_key_alone_does_not_make_the_handle_concrete(
    ctx, sink, monkeypatch
):
    """Configuring the key is not the missing piece; an instance key is.

    Worth pinning because the opposite is the natural assumption — the key is the
    visible knob, so an operator who sets it would otherwise believe they had
    turned on concrete context evidence.
    """
    monkeypatch.setenv(HMAC_KEY_VAR, "deployment-secret")
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    handle = _last_tool_call(sink).attributes[tracing.ATTR_CONTEXT_AFTER]
    assert handle["instance_fingerprint"] is None


def test_no_display_label_is_captured(ctx, sink):
    """§6.7 admits a label only under an explicit allowlist, and there is none.

    The label is the one field on this contract that can carry entity content.
    """
    ctx.process_action_turn(_action(NAVIGATING_COMMAND, description="Bob's medication"))

    span = _last_tool_call(sink)
    for key in (tracing.ATTR_CONTEXT_BEFORE, tracing.ATTR_CONTEXT_AFTER):
        assert span.attributes[key]["display_label"] is None


def test_no_security_scope_is_claimed(ctx, sink):
    """§6.7's host-injected SecurityContext does not exist in fastWorkflow.

    Naming a tenant here would assert a scope no code enforces, so the handle
    says `unscoped` instead. FW-NFR-010 tenant scoping is the outstanding delta
    arch §12.4 already names.
    """
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    handle = _last_tool_call(sink).attributes[tracing.ATTR_CONTEXT_BEFORE]
    assert handle["security_scope_ref"] == tracing.UNSCOPED_SECURITY_SCOPE


# ----------------------------------------------------------------------
# Consequence (§6.6.1)
# ----------------------------------------------------------------------


def test_an_undeclared_command_is_unknown_write_capable_and_high(ctx, sink):
    """The todo workflow ships no manifest, so nothing declares its effects.

    `unknown` rather than `read_only` is the whole rule: an absent contract is a
    reason for more caution, not less (§7.3, §6.6.1). `read_only` here would
    silently under-report every command of every workflow without a manifest.
    """
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
    assert consequence["effect_kind"] == "unknown"
    assert consequence["consequence_class"] == "high"
    assert consequence["assessor_version"] == "default/1"


def test_reversibility_and_blast_radius_stay_unknown(ctx, sink):
    """Nothing in the manifest schema declares either, so neither is guessed."""
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
    assert consequence["reversibility"] == "unknown"
    assert consequence["blast_radius"] == "unknown"


def test_a_declared_effect_contract_reaches_the_span(ctx, sink, todo_workflow_path):
    """The registry is what makes a declaration reachable at execution time.

    `check_startup_conformance` returned a `RuntimeMetadata` that both entry
    points discarded, so before this a workflow that declared `read_only` and one
    that declared nothing produced identical records.
    """
    manifest = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={
            NON_NAVIGATING_COMMAND: CommandDeclaration(
                effect=EffectContract(kind="read_only")
            )
        },
    )
    register_runtime_metadata(
        todo_workflow_path, merge_and_gate(manifest, deployment_features={}, env={})
    )
    try:
        ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

        consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
        assert consequence["effect_kind"] == "read_only"
    finally:
        clear_runtime_metadata()


def test_an_unregistered_workflow_reads_as_unknown_not_read_only():
    """The fallback, at the helper rather than through a whole turn."""
    assert get_runtime_metadata("/no/such/workflow") is None
    consequence = tracing.consequence_assessment("/no/such/workflow", "anything")
    assert consequence["effect_kind"] == "unknown"


def test_the_registry_key_survives_an_unresolved_path(tmp_path):
    """A relative or symlinked path must find its own registration.

    `Workflow` resolves the folderpath it is given, so a table keyed on the raw
    CLI argument would miss — silently, reporting `unknown` for a workflow that
    declared its effects.
    """
    resolved = tmp_path / "wf"
    resolved.mkdir()
    unresolved = tmp_path / "." / "wf"

    metadata = merge_and_gate(None, deployment_features={}, env={})
    register_runtime_metadata(str(unresolved), metadata)
    try:
        assert get_runtime_metadata(str(resolved)) is metadata
    finally:
        clear_runtime_metadata()
