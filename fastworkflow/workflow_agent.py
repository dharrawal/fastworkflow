"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""

import time
import traceback
from datetime import datetime, timezone
from enum import Enum

import dspy

import fastworkflow
from fastworkflow import tracing
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_execution_context import CommandCancelledError
from fastworkflow.utils import dspy_utils
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.typed_failure import classify_exception
from fastworkflow.worker_health import unwrap_request
from fastworkflow.runtime_config import DEFAULT_REACT_MAX_ITERATIONS
from fastworkflow.utils.react import AskUserSuspend, fastWorkflowReAct
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter

# Where the command text a workflow is about to execute came from (FW-REQ-001
# clause 3). This used to be inferred from `iteration_counter <= 0` — a
# process-lifetime counter read as an origin signal, which is why a clarification
# had to reset it to -1 and why the second turn of a session inferred the wrong
# answer. Stated by the dispatcher instead of guessed from a budget.
CONTEXT_KEY_INVOCATION_ORIGIN = "invocation_origin"


class InvocationOrigin(str, Enum):
    """Who chose the command text being executed."""

    # A human typed it: the deterministic / assistant path, where the message
    # goes to the command executor without an agent choosing anything.
    USER = "user"
    # The ReAct agent selected it as a tool call. True of every command the
    # agent dispatches, including the first of a turn and the first after a
    # clarification answer — both of which the old heuristic called `user`.
    AGENT = "agent"


class WorkflowAgentSignature(dspy.Signature):
    """
    Carefully review the user request, then execute the next steps using available tools for building the final answer.
    Every user intent must be fully addressed before returning the final answer.
    """
    user_query = dspy.InputField(desc="The natural language user query.")
    final_answer = dspy.OutputField(desc="Comprehensive final answer with supporting evidence to demonstrate that every user intent has been fully addressed.")

def _append_action_record(chat_session_obj, record: dict) -> None:
    """Append to session-scoped action log (WEC or ChatSession delegating to core).

    Duck-typed like _append_turn_output; no-ops gracefully if neither exposes the
    action log. The cwd action.jsonl fallback was retired in Phase 7 [R25] — the
    in-process log is the only sink, and the observability DB is the post-mortem
    record.
    """
    if hasattr(chat_session_obj, "append_action_log"):
        chat_session_obj.append_action_log(record)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_action_log"):
        core.append_action_log(record)


def _append_turn_output(chat_session_obj, command_output) -> None:
    """Append a CommandOutput to the turn accumulator (WEC or ChatSession delegating to core).

    Duck-typed like _append_action_record; no-ops gracefully if the accumulator
    methods are not available yet.
    """
    if hasattr(chat_session_obj, "append_turn_output"):
        chat_session_obj.append_turn_output(command_output)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_turn_output"):
        core.append_turn_output(command_output)


def _append_ask_user_entry(chat_session_obj, question: str):
    """Append an unanswered ask_user entry to the turn accumulator (duck-typed).

    Only called on the Topology-A blocking path (_ask_user_tool). Topology-B
    suspension (AskUserSuspend) does NOT call this — the WEC core appends the
    unanswered entry itself at suspend time, so this split avoids double-appends.
    No-ops gracefully if the accumulator methods are not available yet.
    """
    if hasattr(chat_session_obj, "append_ask_user_entry"):
        return chat_session_obj.append_ask_user_entry(question)
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_ask_user_entry"):
        return core.append_ask_user_entry(question)
    return None


def _complete_ask_user_entry(chat_session_obj, answer: str) -> None:
    """Record the user's answer on the pending ask_user entry (duck-typed).

    No-ops gracefully if the accumulator methods are not available yet.
    """
    if hasattr(chat_session_obj, "complete_ask_user_entry"):
        chat_session_obj.complete_ask_user_entry(answer)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "complete_ask_user_entry"):
        core.complete_ask_user_entry(answer)


def _what_can_i_do(chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Returns a list of available commands, including their names and parameters.
    """
    current_workflow = chat_session_obj.get_active_workflow()
    return CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
    )

def _refresh_agent_available_commands(host) -> None:
    """Re-scope the active ReAct agent's ``available_commands`` to the CURRENT context.

    Invoked by the workflow's context-change observer (registered in WEC agent init), so it
    fires only on an actual context switch. This is the EXECUTOR's scoped view
    (_what_can_i_do); it never touches the planner's full map. No-ops if there is no active
    agent or it was invoked without available_commands.

    ``host`` is duck-typed for ChatSession and WorkflowExecutionContext: resolve the agent
    via the public ``workflow_tool_agent`` property first, then ``_workflow_tool_agent``.
    Resolve the workflow via ``get_active_workflow()`` with an ``app_workflow`` /
    ``_app_workflow`` fallback so a missing contextvar stack does not skip the refresh.
    """
    agent = getattr(host, "workflow_tool_agent", None)
    if agent is None:
        agent = getattr(host, "_workflow_tool_agent", None)
    inputs = getattr(agent, "inputs", None)
    if not (isinstance(inputs, dict) and "available_commands" in inputs):
        return

    current_workflow = None
    getter = getattr(host, "get_active_workflow", None)
    if callable(getter):
        current_workflow = getter()
    if current_workflow is None:
        current_workflow = getattr(host, "app_workflow", None)
    if current_workflow is None:
        current_workflow = getattr(host, "_app_workflow", None)
    if current_workflow is None:
        return

    inputs["available_commands"] = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        active_context_name=current_workflow.current_command_context_name,
    )


def _intent_misunderstood(
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Shows the full list of available command names so you can specify the command name you really meant
    Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
    """
    return _what_can_i_do(chat_session_obj = chat_session_obj)


def _resolve_or_escalate(result, chat_session_obj: fastworkflow.ChatSession, response_text: str) -> str:
    """
    Route intent-clarification predictor output: recurse on resolve, or escalate to outer ask_user.
    """
    clarified_cmd = getattr(result, "clarified_command", "") or ""
    if bool(getattr(result, "needs_human", False)) or not clarified_cmd:
        # Break recursion: reset CME clarification stage to INTENT_DETECTION.
        _execute_workflow_query("abort", chat_session_obj=chat_session_obj)
        question = getattr(result, "clarification_question", "") or response_text
        # Directive observation -> outer agent calls its own ask_user (blocks in A, suspends in B).
        return (
            "Intent clarification needs the user. "
            f"Use the ask_user tool to ask: {question}"
        )
    return _execute_workflow_query(clarified_cmd, chat_session_obj=chat_session_obj)


def _execute_workflow_query(command: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Executes the command and returns either a response, or a clarification request.
    Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
    Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
    Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
    """
    # Emit trace event before execution
    if chat_session_obj.command_trace_queue is not None:
        chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
        direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
        raw_command=command,
        command_name=None,
        parameters=None,
        response_text=None,
        success=None,
        timestamp_ms=int(time.time() * 1000),
        turn_key=tracing.get_turn_key(chat_session_obj),
        ))

    # fw.agent.tool_call span — deliberately OUTSIDE the trace-queue guard:
    # the sink is reached via the WEC/ChatSession core, not the transport-queue
    # contract, so queue-less embedders still trace [R28].
    span = tracing.start_span(
        chat_session_obj,
        tracing.SPAN_AGENT_TOOL_CALL,
        kind=tracing.KIND_TOOL,
        attributes={"raw_command": command},
    )

    # §12.1.1's shared capture for the agent-tool path (arch §12.0 deltas 1-2/4).
    # This was the one fw.agent.tool_call emitter it had not reached, which is why
    # the agent-tool, resumed-turn and distillation rows of the conformance matrix
    # could not claim P0 — and why a single contract version for this span name
    # would have described three different attribute sets.
    #
    # The projection is `tracing`'s, not a copy of the WorkflowExecutionContext
    # methods: the same record written three ways is three things to drift.
    # Resolving the workflow is gated on a span having opened, so with
    # observability off this seam costs a `start_span` that already declined. One
    # workflow reference serves both handles, matching the sibling sites — the
    # handle is re-projected after execution, so a context the command MOVED is
    # still recorded as a transition.
    workflow = tracing.active_workflow(chat_session_obj) if span is not None else None
    context_before = tracing.context_before(span, workflow)

    # Directly invoke the command without going through queues
    # This allows the agent to synchronously call workflow tools
    from fastworkflow.command_executor import CommandExecutor, _annotation
    started = datetime.now(timezone.utc)
    try:
        command_output = CommandExecutor.invoke_command(chat_session_obj, command)
    except BaseException as e:
        # BaseException, not CommandCancelledError. This arm used to name only
        # that one exception, and the comment beside it even observed that
        # AskUserSuspend "subclasses BaseException, so `except Exception` below
        # never catches it either" — and then nothing caught it here at all. So
        # an agent-mode ask_user unwound straight through this frame to the
        # react loop and the fw.agent.tool_call span opened above was NEVER
        # closed: it leaked onto the parenting stack, every later span in the
        # turn nested under it, and it never got an end time or a status.
        # fix-ajv.21.
        if tracing.is_control_signal(e):
            tracing.end_span(
                chat_session_obj,
                span,
                status=tracing.status_for_dispatch_exception(e),
            )
            raise
        if not isinstance(e, Exception):
            # Any other BaseException (KeyboardInterrupt, SystemExit): close the
            # span so it does not leak, but do not build a failure CommandOutput
            # for it — that is reserved for real command failures, and this
            # frame must not dress an interpreter-level signal as one.
            tracing.end_span(
                chat_session_obj,
                span,
                status=tracing.STATUS_ERROR,
                attributes={"error_type": type(e).__name__},
            )
            raise
        # Capture the failed tool call as a CommandOutput(success=False).
        # This block must never mask the original exception.
        try:
            try:
                error_message = str(e)[:1000]
            except Exception:
                error_message = repr(e)[:1000]
            # Routing has almost always completed by the time a command raises,
            # and command_executor stamps what it had resolved onto the
            # exception. These stay empty only when the failure really did
            # pre-empt routing, or when invoke_command was replaced wholesale
            # (tests/test_turn_result_capture.py does), hence the getattr
            # defaults rather than a declared attribute.
            #
            # CAPTURE-POLICY CONSEQUENCE, handled in _apply_capture_policy.
            # Naming the command moves this record's policy field paths from
            # command.unknown.* to command.<name>.*, and that ran the UNSAFE
            # way round: capture_policy short-circuits and returns a value
            # WHOLE when a declared policy is not gated for the sink, so a rule
            # written about a command's benign NORMAL response would also
            # release the failure text below — an exception repr, an error
            # message, a 4KB traceback. Fixed in fix-ajv.18 by deriving a
            # failed command's response/artifacts paths under
            # command.<name>.error.* instead, so releasing error text is
            # something a deployment declares rather than inherits. Parameters
            # stay on the ordinary path deliberately; see the comment there.
            # A blanket command.unknown.* rule no longer matches these
            # failures either — that is the same fix, seen from the other side.
            failure_output = fastworkflow.CommandOutput(
                command_name=_annotation(e, "_fw_command_name") or "",
                workflow_name=_annotation(e, "_fw_workflow_name") or "",
                context=_annotation(e, "_fw_context") or "",
                command_call_id=_annotation(e, "_fw_call_id"),
                # Explicitly False, not left to the name: this is the output
                # whose routed name could collide with `ask_user`. fix-ajv.17.
                ask_user_entry=False,
                command_response=
                    fastworkflow.CommandResponse(
                        response=f"Execution error: {e!r}"[:500],
                        success=False,
                        artifacts={
                            "error_type": type(e).__name__,
                            "error_message": error_message,
                            "traceback": traceback.format_exc()[:4000],
                        },
                    ),
                started_at=started,
                duration_ms=int(
                    (datetime.now(timezone.utc) - started).total_seconds() * 1000
                ),
            )
            _append_turn_output(chat_session_obj, failure_output)
        except Exception:
            pass  # capture failed — swallow, never mask the original exception
        # ido-cex.7: the success end_span below names the command and context;
        # this one did not, so the third producer of fw.agent.tool_call closed
        # an anonymous row for exactly the failures the failure_output above
        # has just finished naming. Read through _annotation, not getattr, for
        # the reason that helper exists — a hostile __getattr__ here would
        # replace the exception being reported with one about the reporting.
        tracing.end_span(
            chat_session_obj,
            span,
            status=tracing.STATUS_ERROR,
            command_name=_annotation(e, "_fw_command_name") or None,
            context=_annotation(e, "_fw_context") or None,
            attributes={"error_type": type(e).__name__},
        )
        raise

    command_output.started_at = started
    command_output.duration_ms = int(
        (datetime.now(timezone.utc) - started).total_seconds() * 1000
    )
    _append_turn_output(chat_session_obj, command_output)

    # Emit trace event after execution
    # Extract command name and parameters from command_output
    name = command_output.command_name
    params = command_output.command_parameters

    # Live path: typed Pydantic instance. After cold-rehydrate of turn outputs,
    # command_parameters is the dumped dict [A10]. Accept both.
    if params is None:
        params_dict = None
    elif hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = None

    # Extract response text
    response_text = ""
    if command_output.command_response.response:
        response_text = command_output.command_response.response
    else:
        response_text = "Command executed successfully but produced no output."

    tracing.end_span(
        chat_session_obj,
        span,
        status=tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR,
        command_name=name or None,
        context=command_output.context or None,
        attributes={
            "response_text": response_text,
            "success": bool(command_output.success),
            **tracing.capture_attributes(
                span, command_output, context_before, workflow
            ),
        },
    )

    if chat_session_obj.command_trace_queue is not None:
        chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
            direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
            raw_command=None,
            command_name=name,
            parameters=params_dict,
            response_text=response_text,
            success=bool(command_output.success),
            timestamp_ms=int(time.time() * 1000),
            turn_key=tracing.get_turn_key(chat_session_obj),
        ))

    # Append executed action to the session action log for external consumers
    # (agent mode only): distillation compares teacher/student passes off it and
    # final-answer synthesis summarizes it.
    record = {
        "command": command,
        "command_name": name,
        "parameters": params_dict,
        "response": response_text
    }
    _append_action_record(chat_session_obj, record)

    # Check workflow context to determine if we're in an error state that needs specialized handling
    cme_workflow = chat_session_obj.cme_workflow
    nlu_stage = cme_workflow.context.get("NLU_Pipeline_Stage")

    # Handle intent ambiguity clarification state with specialized agent.
    # The intent clarification agent is always present here: _execute_workflow_query
    # is only reachable as a tool of the workflow_tool_agent, and both agents are
    # initialized together in WorkflowExecutionContext._initialize_agent_functionality.
    if nlu_stage == fastworkflow.NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION:
        return _resolve_intent_ambiguity(
            chat_session_obj, cme_workflow, command, response_text
        )
    # Handle intent misunderstanding clarification state with specialized agent
    if nlu_stage == fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION:
        return _resolve_intent_misunderstanding(
            chat_session_obj, command, response_text
        )
    # Handle parameter extraction errors with abort
    if nlu_stage == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION:
        abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
        # Thread the active planning context so replanning uses the same planner LM
        # and insights as the current turn (critical for distillation: otherwise
        # replans silently fall back to LLM_PLANNER instead of the teacher/student LM).
        planning_insights = getattr(chat_session_obj, '_planning_insights', None)
        planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
        return build_query_with_next_steps(
            f'{response_text}\n{abort_confirmation}',
            chat_session_obj, with_agent_inputs_and_trajectory=True,
            planning_insights=planning_insights, planner_lm=planner_lm,
            trace_trigger="parameter_extraction_error",
        )

    # Clean up the origin marker after command execution: it describes one
    # dispatch, and a value left behind would describe the next one wrongly.
    workflow = chat_session_obj.get_active_workflow()
    if CONTEXT_KEY_INVOCATION_ORIGIN in workflow.context:
        del workflow.context[CONTEXT_KEY_INVOCATION_ORIGIN]

    return response_text


# TODO Rename this here and in `_execute_workflow_query`
def _resolve_intent_misunderstanding(chat_session_obj, command, response_text):
    intent_agent = chat_session_obj.intent_clarification_agent
    # Get the workflow agent's trajectory and inputs for context
    workflow_tool_agent = chat_session_obj.workflow_tool_agent
    agent_inputs = workflow_tool_agent.inputs if workflow_tool_agent else {}
    agent_trajectory = workflow_tool_agent.current_trajectory if workflow_tool_agent else {}

    # Inherit the ambient agent LM (set by the caller's dspy.context) rather than
    # hardcoding LLM_AGENT.
    result = intent_agent(
        original_command=command,
        error_message=response_text,
        agent_inputs=agent_inputs,
        agent_trajectory=agent_trajectory,
    )

    return _resolve_or_escalate(result, chat_session_obj, response_text)


# TODO Rename this here and in `_execute_workflow_query`
def _resolve_intent_ambiguity(chat_session_obj, cme_workflow, command, response_text):
    intent_agent = chat_session_obj.intent_clarification_agent
    # Use CommandsSystemPreludeAdapter specifically for workflow agent calls
    agent_adapter = CommandsSystemPreludeAdapter()

    # Get suggested commands from intent detection system
    from fastworkflow._workflows.command_metadata_extraction.intent_detection import CommandNamePrediction
    predictor = CommandNamePrediction(cme_workflow)
    suggested_commands = predictor._get_suggested_commands(predictor.path)

    suggested_commands = list(suggested_commands) if suggested_commands is not None else []

    # Get metadata for only the suggested commands
    current_workflow = chat_session_obj.get_active_workflow()
    suggested_commands_metadata = CommandMetadataAPI.get_suggested_commands_metadata(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
        suggested_command_names=suggested_commands
    )

    # Get the workflow agent's trajectory and inputs for context
    workflow_tool_agent = chat_session_obj.workflow_tool_agent
    agent_inputs = workflow_tool_agent.inputs if workflow_tool_agent else {}
    agent_trajectory = workflow_tool_agent.current_trajectory if workflow_tool_agent else {}

    with dspy.context(adapter=agent_adapter):
        result = intent_agent(
            original_command=command,
            error_message=response_text,
            agent_inputs=agent_inputs,
            agent_trajectory=agent_trajectory,
            available_commands=suggested_commands_metadata  # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter
        )
    return _resolve_or_escalate(result, chat_session_obj, response_text)


def _post_ask_user_response(
    clarification_request: str,
    user_response: str,
    chat_session_obj,
) -> str:
    """
    Steps 2-4 after the user answers an ask_user clarification (Topology-A parity).

    Appends the dialog to the session action log, sets raw_user_message, and replans.
    Returns the observation string for the ReAct loop.
    """
    _append_action_record(
        chat_session_obj,
        {
            "agent_query": clarification_request,
            "user_response": user_response,
        },
    )
    # Complete the pending unanswered ask_user entry with the user's answer.
    # Both topologies route resumes through here.
    _complete_ask_user_entry(chat_session_obj, user_response)

    workflow = chat_session_obj.get_active_workflow()
    if workflow:
        workflow.context["raw_user_message"] = user_response

    planning_insights = getattr(chat_session_obj, '_planning_insights', None)
    planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
    return build_query_with_next_steps(
        user_response,
        chat_session_obj,
        with_agent_inputs_and_trajectory=True,
        planning_insights=planning_insights,
        planner_lm=planner_lm,
        trace_trigger="ask_user_response",
    )


def _ask_user_tool(clarification_request: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    If the missing_information_guidance_tool does not help and only as the last resort, request clarification for missing information from the human user. 
    The clarification_request must be plain text without any formatting.
    Note that using the wrong command name can produce missing information errors. Double-check with the missing_information_guidance_tool to verify that the correct command name is being used 
    """
    user_queue = chat_session_obj.user_message_queue
    output_queue = chat_session_obj.command_output_queue
    if user_queue is None:
        raise CommandCancelledError("ask_user requires a user_message_queue (not available in this context)")

    active = chat_session_obj.get_active_workflow()
    workflow_name = active.folderpath.split('/')[-1] if active else ""
    command_output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response=clarification_request),
        workflow_name=workflow_name,
    )
    if output_queue is not None:
        output_queue.put(command_output)
        # Preserve the transport contract: the output must be visible before the
        # trace sentinel releases the CLI to read it. In v3.0 the queued payload
        # becomes a partial TurnOutput, but the pairing and ordering stay the same.
        trace_queue = chat_session_obj.command_trace_queue
        if trace_queue is not None:
            trace_queue.put(None)

    # Topology A: append the unanswered ask_user entry before blocking.
    # (Topology B appends it inside the WEC core at AskUserSuspend time.)
    _append_ask_user_entry(chat_session_obj, clarification_request)

    # Topology A (blocking): a persistent worker thread runs the agent and a human
    # is expected to answer, so we block indefinitely. (Topology B has no queue and
    # suspends via AskUserSuspend instead — see the ask_user tool closure.)
    #
    # Unwrapped because the answer may arrive as a request envelope (arch §13.3):
    # a clarification answer is a submission like any other. The envelope's
    # completion belongs to the TURN, which the worker loop delivers, so nothing
    # is completed here — completing it on receipt would tell the submitter the
    # turn was done while the agent was still running.
    user_query, _envelope = unwrap_request(user_queue.get())

    return _post_ask_user_response(
        clarification_request, user_query, chat_session_obj
    )

def initialize_workflow_tool_agent(chat_session: fastworkflow.ChatSession,
                                   max_iters: int = DEFAULT_REACT_MAX_ITERATIONS,
                                   execution_insights: str | None = None,
                                   on_step_complete=None):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.

    Args:
        chat_session: fastworkflow.ChatSession instance
        max_iters: Maximum iterations for the ReAct agent. Defaults to the
            deployment default, which is derived from measured per-item command
            cost (`runtime_config`, `ido-24b.4`) rather than being a second
            hand-written copy of 25. Production turns take their limit from
            `WorkflowExecutionContext._effective_react_max_iterations`, which
            applies the restrictive minimum of arch §6.0; this default only
            covers a caller that constructs an agent directly.
        execution_insights: Optional workflow-specific execution anti-patterns to
            append to the agent signature docstring (knowledge distillation).
        on_step_complete: Optional callback(step_idx, trajectory) -> bool for
            step-by-step interception (distillation). Return False to stop early.

    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    chat_session_obj = chat_session
    if not chat_session_obj:
        raise ValueError("chat session cannot be null")

    # Build the agent signature. When execution insights are supplied, append them
    # to WorkflowAgentSignature's instructions as anti-patterns to avoid; otherwise
    # use the module-level WorkflowAgentSignature unchanged.
    if execution_insights:
        enhanced_docstring = (
            f"{WorkflowAgentSignature.__doc__}\n\nCRITICAL ANTI-PATTERNS TO AVOID:\n{execution_insights}"
        )

        class AgentSignature(dspy.Signature):
            __doc__ = enhanced_docstring
            user_query = dspy.InputField(desc="The natural language user query.")
            final_answer = dspy.OutputField(desc="Comprehensive final answer with supporting evidence to demonstrate that every user intent has been fully addressed.")
    else:
        AgentSignature = WorkflowAgentSignature

    def what_can_i_do() -> str:
        """
        Returns a list of available commands, including their names and parameters
        """
        return _what_can_i_do(chat_session_obj=chat_session_obj)

    def intent_misunderstood() -> str:
        """
        Shows the full list of available command names so you can specify the command name you really meant
        Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
        """
        return _intent_misunderstood(chat_session_obj = chat_session_obj)

    def execute_workflow_query(command: str) -> str:
        """
        Takes just a single argument called 'command'.
        Executes the command and returns either a response, or a clarification request.
        Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
        Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
        Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
        """
        # The agent chose this command, so the origin is stated, not inferred.
        # `is_user_command` — the old flag, written from `iteration_counter <= 0`
        # — had no reader anywhere in fastWorkflow or in the IDO workflow; the
        # comment claiming validate_extracted_parameters consumed it was stale.
        workflow = chat_session_obj.get_active_workflow()
        if workflow:
            workflow.context[CONTEXT_KEY_INVOCATION_ORIGIN] = (
                InvocationOrigin.AGENT.value
            )
        
        # Executed ONCE (EXP-011, arch §8.4 phase 2). The blind two-attempt
        # loop this replaces re-dispatched the command on any exception, which
        # is a second effect for anything that had already reached the backend —
        # exactly the replay FW-REQ-008B clause 3 forbids, made with no
        # knowledge of whether the first attempt landed.
        #
        # Control signals (AskUserSuspend, CommandCancelledError, ControlSignal)
        # subclass BaseException and pass straight through: a suspension or a
        # reconciliation-required signal must never become an observation the
        # model can read as ordinary data.
        try:
            return _execute_workflow_query(command, chat_session_obj=chat_session_obj)
        except Exception as e:
            failure = classify_exception(e)
            logger.error(
                "Command %r failed (%s/%s): %s",
                command, failure.disposition, failure.code, failure.detail,
            )
            # The classification is the observation, so the trajectory carries
            # what kind of failure this was rather than a bare error string —
            # and the model is not told to "Terminate immediately", which was a
            # directive the runtime had no standing to issue.
            return failure.as_observation()

    def ask_user(clarification_request: str) -> str:
        """
        Only as the last resort, request clarification for missing information from the human user. 
        The clarification_request must be plain text without any formatting.
        Note that using the wrong command name can produce missing information errors. Double-check with the what_can_i_do tool to verify that the correct command name is being used 
        """
        # No budget reset here. Asking the user does not buy the turn a fresh
        # iteration budget (FW-REQ-001 clause 2, arch §6.4): the same
        # LogicalTurnBudget is serialized at suspension and restored on resume.
        if chat_session_obj.user_message_queue is not None:
            return _ask_user_tool(clarification_request, chat_session_obj=chat_session_obj)
        raise AskUserSuspend(clarification_request)

    tools = [
        what_can_i_do,
        execute_workflow_query,
        # missing_information_guidance,
        intent_misunderstood,
        ask_user,
    ]

    return fastWorkflowReAct(
        AgentSignature,
        tools=tools,
        max_iters=max_iters,
        on_step_complete=on_step_complete,
        decision_point=_policy_decision_point(),
        contract_facts=_contract_facts(chat_session_obj),
    )


def _contract_facts(chat_session_obj):
    """The declared facts a `proceed` row may rest on — computed, not asserted.

    `read_only_surface` is true when every command the manifest declares is
    either `read_only` or write-capable **and not G1W-enabled**. Architecture
    §7.3 makes an undeclared command `unknown` and §6.6.1 makes `unknown`
    write-capable, so a single undeclared command drops the surface — the
    conservative direction, and the one
    `consequence-distribution-degenerate` forces, since `effect_kind` is the
    only informative field the effect contract carries.

    **Why write-capable-but-disabled still counts as read-only.** FW-REQ-021
    clause 2 evaluates consequence for the candidate action *in its binding*,
    not for the command in the abstract. A write that cannot dispatch has no
    write consequence. In this workflow the declared writes are exactly the
    three G1W-track commands — `add_tag`, `remove_tag`, `apply_remediation` —
    and per-command write enablement is its own approval (plan §6), none of
    which has been given. Reading the surface as write-capable because those
    three exist would silence the policy on a surface where no write can happen.

    **The enablement list is read, never assumed.** `FW_G1W_ENABLED` is a
    comma-separated set of G1W-approved definition ids, empty by default. Enable
    one and the surface stops being read-only and this table goes silent, which
    is the conservative direction arriving on its own rather than by anyone
    remembering to arrange it.

    A missing manifest yields the default facts, which are `unknown` — so a
    workflow that ships no manifest gets caution rather than a free proceed.
    """
    from fastworkflow.policy_decision import ContractFacts

    from fastworkflow.runtime_manifest import load_manifest

    workflow = getattr(chat_session_obj, "app_workflow", None)
    folderpath = getattr(workflow, "folderpath", None)
    if not folderpath:
        return ContractFacts()
    try:
        manifest = load_manifest(folderpath)
    except Exception:  # noqa: BLE001 - absence is a legitimate state, see above
        return ContractFacts()
    if manifest is None or not getattr(manifest, "commands", None):
        return ContractFacts()
    import os

    enabled_writes = {
        name.strip()
        for name in os.environ.get("FW_G1W_ENABLED", "").split(",")
        if name.strip()
    }
    unguarded = sorted(
        name for name, declaration in manifest.commands.items()
        if declaration.effect_kind() != "read_only" and name in enabled_writes
    )
    undeclared = sorted(
        name for name, declaration in manifest.commands.items()
        if declaration.effect_kind() == "unknown"
    )
    read_only = not unguarded and not undeclared
    return ContractFacts(
        effect_kind="read_only" if read_only else "unknown",
        read_only_surface=read_only,
        authorization_scope=getattr(chat_session_obj, "authorization_scope",
                                    "unknown") or "unknown",
    )


def _policy_decision_point():
    """Build FW-REQ-017's decision point from the workflow's declared table.

    Two env vars, both defaulting to the clause-5 no-op, because a policy that
    turns itself on is not a feature flag. `FW_ASK_POLICY` is the mode; the
    table is imported from the module the workflow names in
    `FW_ASK_POLICY_TABLE` (`package.module:ATTR`).

    Import failure is a hard error rather than a silent fall back to OFF. A run
    configured to ENFORCE that quietly enforced nothing would be recorded as a
    treatment arm and measured as one, which is the confound the whole
    experiment exists to avoid.
    """
    import importlib
    import os

    from fastworkflow.policy_decision import (
        NO_OP_TABLE, PolicyDecisionPoint, PolicyMode)

    mode = PolicyMode(os.environ.get("FW_ASK_POLICY", PolicyMode.OFF.value))
    if mode is PolicyMode.OFF:
        return PolicyDecisionPoint(NO_OP_TABLE, PolicyMode.OFF)
    reference = os.environ.get("FW_ASK_POLICY_TABLE", "")
    if not reference:
        raise ValueError(
            "FW_ASK_POLICY=%s with no FW_ASK_POLICY_TABLE: the mechanism is "
            "framework-side but the table is workflow content (FW-REQ-017 "
            "clause 2), so there is nothing to enforce" % mode.value)
    module_name, _, attribute = reference.partition(":")
    table = getattr(importlib.import_module(module_name), attribute or "TABLE")
    return PolicyDecisionPoint(table, mode)


def build_query_with_next_steps(user_query: str,
    chat_session_obj: fastworkflow.ChatSession, with_agent_inputs_and_trajectory: bool = False,
    planning_insights: str | None = None, planner_lm = None,
    trace_trigger: str | None = None) -> str:
    """
    Generate a todo list.
    Return a string that combine the user query and todo list

    Args:
        user_query: The user's natural language query
        chat_session_obj: The active chat session
        with_agent_inputs_and_trajectory: Whether to include agent trajectory for replanning
        planning_insights: Optional workflow-specific planning insights to append to
            the planner signature docstring (knowledge distillation)
        planner_lm: Optional planner LM to use (if None, uses LLM_PLANNER from env)
        trace_trigger: What re-triggered planning mid-turn (e.g.
            "ask_user_response", "parameter_extraction_error"). None means the
            turn's initial plan; set, it marks the span fw.planner.replan.
    """
    base_docstring = """
    Carefully review the user_query and generate a next steps sequence based only on available commands.
    Walk the graph of commands based on the 'available_from' hints to build the most appropriate command sequence.

    IMPORTANT: 9 times out of 10 information can be found via available commands. However, when generating the plan:
    - If required information is missing and cannot be found via commands, explicitly specify in the plan that the user needs to be consulted
    - If confirmation is needed before proceeding, explicitly specify in the plan that user confirmation is required
    """
    if planning_insights:
        enhanced_docstring = f"{base_docstring}\n\nCRITICAL PATTERNS FOR THIS WORKFLOW:\n{planning_insights}"
    else:
        enhanced_docstring = base_docstring

    class TaskPlannerSignature(dspy.Signature):
        __doc__ = enhanced_docstring
        user_query: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    class TaskPlannerWithTrajectoryAndAgentInputsSignature(dspy.Signature):
        __doc__ = enhanced_docstring
        agent_inputs: dict = dspy.InputField()
        agent_trajectory: dict = dspy.InputField()
        user_response: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    current_workflow = chat_session_obj.get_active_workflow()
    available_commands = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
    )

    # Use provided planner_lm if available (distillation mode), else build from env
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    agent_adapter = CommandsSystemPreludeAdapter()

    # fw.planner.plan for the turn's initial plan, fw.planner.replan for
    # mid-turn re-planning (trace_trigger names what re-triggered it).
    span = tracing.start_span(
        chat_session_obj,
        tracing.SPAN_PLANNER_REPLAN if trace_trigger else tracing.SPAN_PLANNER_PLAN,
        kind=tracing.KIND_LLM,
        attributes={
            "model": getattr(planner_lm, "model", None),
            "replan_trigger": trace_trigger,
        },
    )
    try:
        with dspy.context(lm=planner_lm, adapter=agent_adapter):
            if with_agent_inputs_and_trajectory:
                workflow_tool_agent = chat_session_obj.workflow_tool_agent
                task_planner_func = dspy.ChainOfThought(TaskPlannerWithTrajectoryAndAgentInputsSignature)
                cleaned_agent_inputs = {k: v for k, v in workflow_tool_agent.inputs.items() if k != "available_commands"}
                prediction = task_planner_func(
                    agent_inputs = cleaned_agent_inputs,
                    agent_trajectory = workflow_tool_agent.current_trajectory,
                    user_response = user_query,
                    available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter
            else:
                task_planner_func = dspy.ChainOfThought(TaskPlannerSignature)
                prediction = task_planner_func(
                    user_query=user_query,
                    available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter
    except BaseException:
        tracing.end_span(chat_session_obj, span, status=tracing.STATUS_ERROR)
        raise
    tracing.end_span(
        chat_session_obj,
        span,
        attributes={"plan": prediction.next_steps or ""},
    )

    if not prediction.next_steps:
        return user_query

    generated_plan = prediction.next_steps.split()
    # Capture the generated plan for distillation when a capture list is present
    # on the session (set only during a DistillationSession planning pass).
    planning_capture = getattr(chat_session_obj, '_planning_steps_capture', None)
    if planning_capture is not None:
        from fastworkflow.distillation import PlanningStep
        planning_capture.append(PlanningStep(
            step_number=len(planning_capture),
            user_query=user_query,
            generated_plan=generated_plan,
            reasoning=getattr(prediction, 'reasoning', ''),
        ))

    steps_formatted = " ".join(generated_plan)
    user_query_and_next_steps = f"{user_query}\n\nExecute these next steps:\n{steps_formatted}"
    return (
        f'User Query:\n{user_query_and_next_steps}'
        if with_agent_inputs_and_trajectory else
        user_query_and_next_steps
    )
