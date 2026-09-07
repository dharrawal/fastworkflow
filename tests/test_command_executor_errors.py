import uuid

import fastworkflow
import pytest

from fastworkflow import ModuleType, tracing
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError


# ---------------------------------------------------------------------------
# Helpers to monkeypatch registry and extractor
# ---------------------------------------------------------------------------


class FaultyRG:  # noqa: D401
    def __call__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("boom")


class DummyCRD:  # minimal stand-in for RoutingDefinition
    def get_command_class(self, name, module_type):  # noqa: D401
        # Only the response generator is faulty. Returning FaultyRG for
        # COMMAND_PARAMETERS_CLASS would make perform_action treat it as a
        # pydantic Input model and run validate_parameters on it.
        if name == "fail" and module_type == ModuleType.RESPONSE_GENERATION_INFERENCE:
            return FaultyRG
        return None


def _monkey_registry(monkeypatch):
    monkeypatch.setattr(
        fastworkflow.RoutingRegistry,
        "get_definition",
        lambda _: DummyCRD(),
    )


def test_perform_action_wraps_error(monkeypatch):
    fastworkflow.init({})

    _monkey_registry(monkeypatch)

    workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str="errs_pa",
    )

    action = fastworkflow.Action(command_name="fail", command="fail")

    with pytest.raises(RuntimeError):
        CommandExecutor.perform_action(workflow, action)


def test_invoke_command_wraps_error(monkeypatch):
    """A failing response generator propagates out of the NLU path too.

    The sibling above covers the direct-action entry point. This one covers
    invoke_command, which reaches the same response generator after intent
    detection has named the command — the two must not diverge on whether a
    command's own exception escapes or gets swallowed into a CommandOutput.

    Driven through a real WorkflowExecutionContext because that is what calls
    invoke_command in production (workflow_execution_context.py:1175 passes
    `self`). The parameter is annotated ChatSession, but only three members are
    touched — cme_workflow, get_active_workflow() and cme_workflow._context —
    and WEC supplies all three, so the annotation is looser than the contract.

    This test previously stubbed CommandExecutor._invoke_command_metadata_extraction_workflow
    and built its session with ChatSession(workflow_folderpath=..., workflow_id_str=...).
    Both are gone: the CME hop is now a perform_action call with the `wildcard`
    command, and ChatSession takes neither argument. It had been skipping on a
    hasattr() guard for the missing method, which meant it silently asserted
    nothing rather than failing when the implementation moved.
    """
    fastworkflow.init({})

    _monkey_registry(monkeypatch)

    from fastworkflow import CommandOutput, CommandResponse
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    # Stand in for intent detection. Everything below the CME hop is real: this
    # only names the command the way a successful extraction would, so the
    # response generator lookup and call are the code under test.
    #
    # perform_action is the seam because that is what the CME hop is now
    # (invoke_command calls it with command_name="wildcard"). It cannot be
    # reached through the stubbed registry instead — DummyCRD has no class for
    # "wildcard", so perform_action would raise ValueError over the RuntimeError
    # this test is about. Only the CME hop goes through perform_action;
    # invoke_command instantiates the resolved command's generator directly, so
    # stubbing it does not hide the call under test.
    def _stub_cme(workflow, action):
        return CommandOutput(
            command_response=CommandResponse(
                response="stub",
                artifacts={
                    "command_name": "fail",
                    "cmd_parameters": None,
                    "command": action.command,
                },
            ),
            success=True,
            command_handled=False,
        )

    monkeypatch.setattr(CommandExecutor, "perform_action", _stub_cme)

    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str="err_ivk",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="err_ivk")
    ctx.bind_app_workflow(app_workflow)

    # invoke_command resolves the target through get_active_workflow(), which on a
    # WEC reads only the context-local stack — there is no fallback to the bound
    # app workflow, deliberately. Production pushes it for the duration of the
    # turn (workflow_execution_context.py:780), so pushing it here is what makes
    # this the same call and not a contrived one.
    ctx.push_active_workflow(app_workflow)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            CommandExecutor.invoke_command(ctx, "fail")
    finally:
        ctx.pop_active_workflow()
        ctx.close()


# ---------------------------------------------------------------------------
# Helpers for the dispatch-status tests below (ported from 5b1e85e; the
# fix-ajv.16 routed-identity tests are NOT ported — out of scope for fix-49m.4)
# ---------------------------------------------------------------------------


class RecordingTraceSink:
    def __init__(self):
        self.spans = []
        self.turn_records = []

    def emit_span(self, span):
        self.spans.append(span)

    def emit_turn_record(self, record):
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, *args):
        pass

    def named(self, name):
        return [span for span in self.spans if span.name == name]


def _raising_registry(monkeypatch, exc_factory):
    """Registry stand-in whose only command, `fail`, raises from its generator."""

    class RaisingRG:
        def __call__(self, *args, **kwargs):
            raise exc_factory()

    class RaisingCRD:
        def get_command_class(self, name, module_type):
            if (
                name == "fail"
                and module_type == ModuleType.RESPONSE_GENERATION_INFERENCE
            ):
                return RaisingRG
            return None

    monkeypatch.setattr(
        fastworkflow.RoutingRegistry, "get_definition", lambda _: RaisingCRD()
    )


def _stub_cme_naming(command_name: str):
    """Stand in for intent detection: name the command the way extraction would."""
    from fastworkflow import CommandOutput, CommandResponse

    def _stub(workflow, action):
        return CommandOutput(
            command_response=CommandResponse(
                response="stub",
                artifacts={
                    "command_name": command_name,
                    "cmd_parameters": None,
                    "command": action.command,
                },
            ),
            success=True,
            command_handled=False,
        )

    return _stub


def _failing_ctx(monkeypatch, exc_factory):
    """A traced WEC with an open turn whose `fail` command raises."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    fastworkflow.init({})
    _raising_registry(monkeypatch, exc_factory)
    monkeypatch.setattr(
        CommandExecutor, "perform_action", _stub_cme_naming("fail")
    )

    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str=f"err_identity_{uuid.uuid4().hex}",
    )
    sink = RecordingTraceSink()
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(app_workflow)
    # A recorder is minted per turn, so the ledger assertions need one open.
    ctx._begin_turn("make it fail")
    ctx.push_active_workflow(app_workflow)
    return ctx, sink, app_workflow


def test_control_signal_is_neither_stamped_nor_recorded(monkeypatch):
    """Cancellation is not a failed dispatch: it produces no outcome to join."""
    from fastworkflow.workflow_execution_context import CommandCancelledError

    ctx, _sink, _wf = _failing_ctx(
        monkeypatch, lambda: CommandCancelledError("cancelled")
    )
    try:
        with pytest.raises(CommandCancelledError) as excinfo:
            CommandExecutor.invoke_command(ctx, "fail")
        records = ctx._execution_recorder.records()
    finally:
        ctx.pop_active_workflow()
        ctx.close()

    assert getattr(excinfo.value, "_fw_call_id", None) is None
    assert not records


# ---------------------------------------------------------------------------
# A control signal is not a failure, at either decision (fix-ajv.19)
# ---------------------------------------------------------------------------
#
# The error path makes two decisions about an in-flight exception: whether to
# record it as a dispatch, and what status to close its span with. When each
# was written with its own isinstance check they disagreed — AskUserSuspend was
# a control signal for the first and a FAILURE for the second — so an ordinary
# pause for input closed as STATUS_ERROR, and the chatbot waterfall draws any
# span with that status as a red ERROR node (index.html:1671). Both decisions
# now read the one predicate; these tests pin them together so a future edit to
# one has to face the other.


def _control_signals():
    from fastworkflow.utils.react import AskUserSuspend
    from fastworkflow.workflow_execution_context import CommandCancelledError

    return [
        pytest.param(lambda: CommandCancelledError("cancelled"), id="cancelled"),
        pytest.param(lambda: AskUserSuspend("which order?"), id="ask_user_suspend"),
    ]


@pytest.mark.parametrize("exc_factory", _control_signals())
def test_a_control_signal_never_closes_its_span_as_an_error(monkeypatch, exc_factory):
    ctx, sink, _wf = _failing_ctx(monkeypatch, exc_factory)
    try:
        with pytest.raises(BaseException):
            CommandExecutor.invoke_command(ctx, "fail")
    finally:
        ctx.pop_active_workflow()
        ctx.close()

    span = sink.named(tracing.SPAN_COMMAND_EXECUTE)[-1]
    assert span.status == tracing.STATUS_CANCELLED
    assert span.status != tracing.STATUS_ERROR


@pytest.mark.parametrize("exc_factory", _control_signals())
def test_the_two_decisions_agree_for_every_control_signal(monkeypatch, exc_factory):
    """Not recorded as a dispatch AND not an error span — the pair, not one of them."""
    ctx, sink, _wf = _failing_ctx(monkeypatch, exc_factory)
    try:
        with pytest.raises(BaseException) as excinfo:
            CommandExecutor.invoke_command(ctx, "fail")
        records = ctx._execution_recorder.records()
    finally:
        ctx.pop_active_workflow()
        ctx.close()

    span = sink.named(tracing.SPAN_COMMAND_EXECUTE)[-1]
    assert not records
    assert getattr(excinfo.value, "_fw_call_id", None) is None
    assert span.status != tracing.STATUS_ERROR


# ---------------------------------------------------------------------------
# One mapping for every dispatch site (fix-ajv.19), and no site may skip it
# (fix-ajv.21)
# ---------------------------------------------------------------------------


def test_the_status_mapping_is_a_single_shared_function():
    """Five sites used to carry their own isinstance check and disagreed.

    Pinned as a unit test on the helper so the intent survives even if a call
    site is rewritten: control signals are not failures, everything else is.
    """
    from fastworkflow.utils.react import AskUserSuspend
    from fastworkflow.workflow_execution_context import CommandCancelledError

    assert tracing.status_for_dispatch_exception(AskUserSuspend("q")) == (
        tracing.STATUS_CANCELLED
    )
    assert tracing.status_for_dispatch_exception(CommandCancelledError("c")) == (
        tracing.STATUS_CANCELLED
    )
    assert tracing.status_for_dispatch_exception(RuntimeError("boom")) == (
        tracing.STATUS_ERROR
    )
    assert tracing.is_control_signal(AskUserSuspend("q")) is True
    assert tracing.is_control_signal(RuntimeError("boom")) is False


def test_a_span_status_of_awaiting_user_is_never_produced_for_a_dispatch():
    """The taxonomy decision, pinned so it is not silently reverted.

    `awaiting_user` is a TURN-level state here — the store's non-terminal turn
    status, and the only thing the SPA tests for (it reads `turn.status`, never
    `span.status`). A span status describes that span's own outcome.
    """
    from fastworkflow.utils.react import AskUserSuspend

    assert tracing.status_for_dispatch_exception(AskUserSuspend("q")) != (
        tracing.STATUS_AWAITING_USER
    )


@pytest.mark.parametrize("exc_factory", _control_signals())
def test_the_agent_tool_span_is_closed_even_when_a_control_signal_escapes(
    monkeypatch, exc_factory
):
    """fix-ajv.21: AskUserSuspend used to unwind past this frame uncaught.

    The arm named only CommandCancelledError, so the other control signal — a
    BaseException by design — closed no span. The fw.agent.tool_call span opened
    for the tool leaked onto the parenting stack: every later span in the turn
    nested under it, and it never got an end time or a status.
    """
    from fastworkflow.workflow_agent import _execute_workflow_query

    ctx, sink, _wf = _failing_ctx(monkeypatch, exc_factory)
    try:
        with pytest.raises(BaseException):
            _execute_workflow_query("fail", ctx)
    finally:
        ctx.pop_active_workflow()
        ctx.close()

    tool_spans = sink.named(tracing.SPAN_AGENT_TOOL_CALL)
    assert tool_spans, "the agent tool span must be emitted at all"
    assert tool_spans[-1].status == tracing.STATUS_CANCELLED
    assert tool_spans[-1].status != tracing.STATUS_OPEN
