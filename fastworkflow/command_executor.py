import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_interfaces import CommandExecutorInterface

from fastworkflow import Action, CommandOutput, ChatSession
from fastworkflow.execution_recorder import record_execution, recorder_for
from fastworkflow import ModuleType
from fastworkflow.utils.signatures import InputForParamExtraction
from pathlib import Path
from fastworkflow.command_routing import RoutingDefinition
from typing import Optional
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_resolution import (
    ExactResolution,
    index_for_workflow,
    simple_name_resolution_enforced,
)
from fastworkflow import result_handles
from fastworkflow.utils.logging import logger


# ------------------------------------------------------------------
# Module-level delegation configuration and exceptions
# ------------------------------------------------------------------

MAX_DELEGATION_DEPTH: int = 10  # Safety limit for delegation hops


class CommandNotFoundError(Exception):
    """Raised when a command cannot be resolved in any accessible context."""


def _annotate_exception(exc: BaseException, **fields) -> None:
    """Stamp routed identity onto an in-flight exception, first writer wins.

    An exception unwinds outward, so the innermost frame that knew the identity
    is the one that owns it; an enclosing dispatch must not overwrite what a
    nested one already stamped. Attribute setting is best-effort — an exception
    with __slots__ refuses it — because losing the annotation is survivable and
    masking the original exception is not.
    """
    for name, value in fields.items():
        # The READ is inside the guard too, not just the write. An exception
        # whose type defines a __getattr__ that raises something other than
        # AttributeError would otherwise propagate out of here and destroy the
        # command's real exception — the precise failure this function's
        # docstring says must never happen, reintroduced by the probe meant to
        # avoid it.
        try:
            if getattr(exc, name, None) is None:
                setattr(exc, name, value)
        except Exception:
            pass


def _annotation(exc: BaseException, name: str):
    """Read an annotation without letting a hostile __getattr__ escape.

    Same reason the write is guarded: this runs while an exception is already
    unwinding, and anything raised here replaces the failure the caller is
    trying to report with one about the reporting.
    """
    try:
        return getattr(exc, name, None)
    except Exception:
        return None


class CommandExecutor(CommandExecutorInterface):
    @classmethod
    def invoke_command(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        if not command:
            return CommandOutput(
                command_response=
                    fastworkflow.CommandResponse(
                        response="You just hit the <Enter> key. How about a command or some feedback instead?"
                    )
            )

        # One id per command execution, minted before the span so both sides
        # carry it (arch §12.0 delta 1). This is the join key between the
        # CommandOutput that lands in the turn's record_json and the span that
        # produced it; before it there was none.
        call_id = tracing.new_command_call_id()
        parent_call_id = tracing.current_call_id()

        # fw.command.execute boundary span (observability design §3.1, D3).
        # chat_session is duck-typed (WEC, or ChatSession delegating to its
        # core); with no sink or open turn the helpers no-op.
        span = tracing.start_span(
            chat_session,
            tracing.SPAN_COMMAND_EXECUTE,
            kind=tracing.KIND_TOOL,
            attributes={
                "raw_command": command,
                tracing.ATTR_COMMAND_CALL_ID: call_id,
                tracing.ATTR_PARENT_CALL_ID: parent_call_id,
            },
        )

        # Context BEFORE execution (arch §12.0 delta 2, FW-REQ-002). Gated on a
        # span having actually opened, for the same reason the attribute prep
        # below is: with tracing off this work must not run at all.
        context_before = (
            tracing.context_handle(cls._active_workflow(chat_session))
            if span is not None
            else None
        )

        # Bound before the try so the error path can still file this dispatch's
        # inner hops when call_scope itself is what raised.
        child_calls: list = []
        try:
            # Bind the trace host for the deep NLU emission sites
            # (fw.nlu.intent / fw.nlu.param_extraction) — they run several
            # frames down with no reference to the session ([R28]; D3 as
            # amended). call_scope additionally makes this dispatch the parent
            # of every perform_action hop underneath it.
            with tracing.host_scope(chat_session), tracing.call_scope(
                call_id
            ) as child_calls:
                command_output = cls._invoke_command_impl(chat_session, command)
        except BaseException as exc:
            # CommandCancelledError/AskUserSuspend are BaseException control
            # signals — close the span and always re-raise untouched. Which
            # exceptions those are lives in `tracing`, not here, so this site
            # and the four others cannot drift apart (fix-ajv.19); the local
            # imports that used to name them are gone with the isinstance.
            is_control_signal = tracing.is_control_signal(exc)
            tracing.end_span(
                chat_session,
                span,
                # One mapping, shared with every other dispatch site so they
                # cannot drift apart again. fix-ajv.19.
                status=tracing.status_for_dispatch_exception(exc),
                # Routing usually got far enough to bind these before the
                # generator raised; _invoke_command_impl stamps what it had.
                # Without them the error span was the only one in the taxonomy
                # that could not say which command it covered.
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={"error_type": type(exc).__name__},
            )
            if not is_control_signal:
                # A failure is still a dispatch that happened. Without the id,
                # the outcome the caller builds from `exc` cannot be joined to
                # this span or to the record below.
                _annotate_exception(exc, _fw_call_id=call_id)
                # Paired with the stamp rather than with the span: an outcome
                # carrying a command_call_id must have an ExecutionRecordRef to
                # join to (tests/test_dispatch_path_conformance.py). Control
                # signals yield no outcome, so they stay unrecorded as before.
                try:
                    record_execution(
                        recorder_for(chat_session),
                        command_call_id=call_id,
                        parent_call_id=parent_call_id,
                        span_id=span.span_id if span is not None else None,
                        child_calls=child_calls,
                    )
                except Exception:
                    # Every other observation site on this path is wrapped so
                    # it cannot mask the command's exception; this one was
                    # bare. Recording is best-effort, unwinding is not.
                    pass
            raise

        # Attribute prep stays inside the never-raise boundary and runs only
        # when a span was actually opened: a user-authored parameters model
        # whose model_dump() raises must not fail the turn, and with tracing
        # off this work must not run at all.
        params_dict = None
        if span is not None:
            try:
                params = command_output.command_parameters
                if hasattr(params, "model_dump"):
                    params_dict = params.model_dump()
                elif isinstance(params, dict):
                    params_dict = params
            except Exception:
                params_dict = None

        # Stamped unconditionally, not only when a span opened: the id is what
        # makes the outcome joinable, and a turn recorded with tracing off can
        # still be read back through a public API.
        command_output.command_call_id = call_id

        # Observation compaction (ido-mn1.6.1). Stores the full payload of a
        # command that OPTED IN, keyed by this dispatch's call id — which is why
        # it happens here and not at the agent seam: the call id is minted in
        # this frame, and `plan.py` already binds a captured handle to the
        # producing `command_call_id`, so one id serves both. A command that did
        # not opt in returns None from one Mapping lookup and nothing else
        # happens: no store write, no attribute, no change to `command_output`.
        # Unconditional on tracing, unlike the attribute prep above, because the
        # agent needs the payload whether or not anything is being recorded.
        #
        # The second half of the return is what storing this payload DISPLACED.
        # It rides this span rather than being logged and forgotten because a
        # composition step resolves cited handles at the end of a turn: when one
        # is gone, the only way to find out why is a record naming the dispatch
        # that dropped it.
        result_handle_id, evicted_handle_ids = (
            result_handles.store_from_command_output(chat_session, command_output)
        )

        context_after = None
        consequence = None
        if span is not None:
            workflow = cls._active_workflow(chat_session)
            context_after = tracing.context_handle(workflow)
            consequence = tracing.consequence_assessment(
                getattr(workflow, "folderpath", None),
                command_output.command_name or None,
            )

        tracing.end_span(
            chat_session,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "parameters": params_dict,
                "response_text": command_output.command_response.response or "",
                "success": bool(command_output.success),
                tracing.ATTR_CONTEXT_BEFORE: context_before,
                tracing.ATTR_CONTEXT_AFTER: context_after,
                tracing.ATTR_CONSEQUENCE: consequence,
                # The internal CME/core hops this dispatch made, each naming its
                # parent (arch §12.1 item 5). They have no spans of their own, so
                # this ledger is where their correlation lives. An empty list is
                # recorded rather than omitted: "this dispatch made no inner
                # calls" and "nothing captured them" are different facts, and an
                # absent key cannot tell them apart.
                tracing.ATTR_CHILD_CALLS: list(child_calls),
                # Spread, not a key with a None value: an off-path command must
                # produce a span byte-identical to the one it produced before
                # this feature existed, and `result_handle_id: null` is not that.
                **tracing.result_handle_attributes(
                    result_handle_id, evicted_handle_ids
                ),
            },
        )
        record_execution(
            recorder_for(chat_session),
            command_call_id=call_id,
            parent_call_id=parent_call_id,
            span_id=span.span_id if span is not None else None,
            child_calls=child_calls,
        )
        return command_output

    @staticmethod
    def _active_workflow(chat_session: 'fastworkflow.ChatSession'):
        """The workflow whose command context this dispatch acts on, or None.

        Duck-typed and never raising, like the rest of the tracing seam: this is
        called only to build capture attributes, and a host that cannot answer
        must degrade to an absent handle rather than fail the command.
        """
        try:
            return chat_session.get_active_workflow()
        except Exception:
            return None

    @classmethod
    def _invoke_command_impl(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        command_output = cls.perform_action(
            chat_session.cme_workflow, 
            Action(
                command_name = "wildcard",
                command = command)
        )

        if command_output.command_handled:       
            # important to clear the current command mode from the workflow context
            if "is_assistant_mode_command" in chat_session.cme_workflow._context:
                del chat_session.cme_workflow._context["is_assistant_mode_command"]
            return command_output
        elif not command_output.success:       
            return command_output

        command_name = command_output.command_response.artifacts["command_name"]
        input_obj = command_output.command_response.artifacts["cmd_parameters"]

        workflow = chat_session.get_active_workflow()
        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow.current_command_context_displayname

        # ido-cex.7: the seam starts HERE, not at the response-generator call.
        # command_name, workflow_name and context are bound three lines up, and
        # everything between there and the call could still raise — the routing
        # lookup, the missing-class check below, reading raw_user_message off a
        # workflow context. Those raises unwound with nothing on them, so a
        # failure this frame had already named arrived at workflow_agent as
        # CommandOutput(command_name=''), indistinguishable from one that really
        # did pre-empt routing.
        try:
            command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
                workflow.folderpath
            )

            response_generation_class = command_routing_definition.get_command_class(
                command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
            if not response_generation_class:
                # The message names the scope as well as the command: a command
                # that exists but is not reachable from here and one that does
                # not exist at all produce the same exception, and only the
                # context tells a reader which happened.
                raise ValueError(
                    f"Response generation class not found for command name "
                    f"'{command_name}' in context '{context}' of workflow "
                    f"'{workflow_name}'"
                )
            response_generation_object = response_generation_class()

            raw_user_message = command
            if "raw_user_message" in workflow.context:
                raw_user_message = workflow.context['raw_user_message']

            if command_parameters_class := (
                command_routing_definition.get_command_class(
                    command_name, ModuleType.COMMAND_PARAMETERS_CLASS
                )
            ):
                command_output = response_generation_object(workflow, raw_user_message, input_obj)
            else:
                command_output = response_generation_object(workflow, raw_user_message)
        except BaseException as exc:
            # Routing resolved this identity above; the assignments that would
            # publish it onto the CommandOutput sit below the raise, so on
            # failure it died with the frame and every caller had to report the
            # command as unnamed. Carry it out on the exception instead, and
            # re-raise unconditionally — this seam observes, it never handles.
            _annotate_exception(
                exc,
                _fw_command_name=command_name,
                _fw_workflow_name=workflow_name,
                _fw_context=context,
            )
            raise

        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_name = command_name
        command_output.command_parameters = input_obj or None

        # important to clear the current command mode from the workflow context
        if "is_assistant_mode_command" in chat_session.cme_workflow._context:
            del chat_session.cme_workflow._context["is_assistant_mode_command"]

        return command_output

    @classmethod
    def check_capability(
        cls,
        workflow: fastworkflow.Workflow,
        command_name: str,
        *,
        requested_context: Optional[str] = None,
    ) -> Optional["ExactResolution"]:
        """The same capability check the agent path uses (arch §10.4).

        A named direct action must not be able to reach a command the agent
        path would refuse: `perform_action` had no eligibility check at all, so
        anything with a response-generation class ran, in any context.

        Returns the failed resolution when the command is not callable here, and
        None when it is callable or when the check cannot be made (an unknown
        workflow model falls through to current behavior rather than failing a
        turn). `requested_context` is the MCP path's declared context; naming
        anything but the current effective context is `not-callable-here` in P0
        (arch §10.4), because the alternative — accepting it and executing
        somewhere else — is the silent cross-context dispatch FW-REQ-005 exists
        to stop.
        """
        current = workflow.current_command_context_name or "*"
        if requested_context and requested_context != current:
            return ExactResolution(
                token=command_name,
                failure="not-callable-here",
                detail=(
                    f"MCP tool call requested context '{requested_context}' but the "
                    f"current context is '{current}'; P0 dispatches only in the "
                    "current effective context"
                ),
            )

        index = index_for_workflow(workflow.folderpath)
        if index is None:
            return None
        resolution = index.resolve_exact(command_name, current)
        if resolution.resolved:
            return None
        # An identity nothing recognises is not this check's business: the
        # command may be a core or CME command the index does not model, and
        # refusing it here would be inventing a restriction.
        return None if resolution.is_unknown else resolution

    @classmethod
    def _capability_refusal(
        cls,
        workflow: fastworkflow.Workflow,
        command_name: str,
        resolution: "ExactResolution",
    ) -> Optional[fastworkflow.CommandOutput]:
        """Refuse, or record and continue, depending on the feature mode.

        `shadow` — the P0 default — records the would-be refusal on the span and
        lets the dispatch proceed, so the check is measurable before it changes
        an outcome (architecture §16's slice modes). `enforce` returns the typed
        failure. Nothing navigates and no backend call is made either way
        (FW-REQ-005 clauses 3 and 5).
        """
        detail = resolution.detail or f"'{command_name}' is not callable here"
        if not simple_name_resolution_enforced(workflow.folderpath):
            logger.warning(
                "Capability check (shadow) would refuse '%s': %s",
                command_name, detail,
            )
            return None
        response = fastworkflow.CommandResponse(response=detail, success=False)
        response.artifacts["typed_failure"] = resolution.failure
        response.artifacts["command_name"] = command_name
        return CommandOutput(
            command_name=command_name,
            command_response=response,
            workflow_name=workflow.folderpath.split("/")[-1],
        )

    @classmethod
    def perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        # Arch §10.4: the direct-action path goes through the same capability
        # check as everything else, so a named action cannot bypass current-
        # context eligibility.
        if refusal_source := cls.check_capability(workflow, action.command_name):
            if refusal := cls._capability_refusal(
                workflow, action.command_name, refusal_source
            ):
                return refusal

        workflow.command_context_for_response_generation = \
            workflow.current_command_context

        # One id per dispatch through this method (arch §12.0 delta 1). It is
        # the outermost id on the direct-action, startup-action and MCP paths,
        # and a child id on the internal CME hop that invoke_command makes — in
        # which case call_scope files it under the enclosing command call.
        call_id = tracing.new_command_call_id()

        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow.current_command_context_displayname

        # ido-cex.7. The same annotating seam _invoke_command_impl has, for the
        # same reason and with the same rule: from here down the identity is
        # bound, so no failure below may unwind anonymously. fix-ajv.16 closed
        # the NLU path and left this one — and this is the path the MCP tool
        # call, the startup action, the agent's direct actions and
        # invoke_command's own CME hop all take, so its raises were reaching
        # workflow_agent as CommandOutput(command_name='') exactly the way the
        # NLU path's used to.
        #
        # Six raises sit inside it: the missing response-generation class, a
        # parameters model that rejects action.parameters, the two "did not
        # return a CommandOutput" type checks, the validate_parameters refusal,
        # and the response generator's own exception.
        #
        # The CME hop dispatches Action('wildcard') against the metadata-
        # extraction workflow, so a failure during intent detection is now
        # named 'wildcard' in 'command_metadata_extraction' rather than left
        # blank. That is the truthful origin and it is what the blank name was
        # hiding: a turn that died BEFORE routing now says so, instead of being
        # indistinguishable from a routed command that lost its name.
        try:
            command_routing_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)

            response_generation_class = (
                command_routing_definition.get_command_class(
                    action.command_name,
                    ModuleType.RESPONSE_GENERATION_INFERENCE,
                )
            )
            if not response_generation_class:
                # Scope in the message, not only on the exception: the MCP
                # entry point below stringifies this and drops the annotation,
                # so the text is the only carrier left on that path.
                raise ValueError(
                    f"Response generation class not found for command name "
                    f"'{action.command_name}' in context '{context}' of workflow "
                    f"'{workflow_name}'"
                )

            response_generation_object = response_generation_class()

            command_parameters_class = (
                command_routing_definition.get_command_class(
                    action.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
                )
            )
            if not command_parameters_class:
                with tracing.call_scope(call_id, command_name=action.command_name):
                    command_output = response_generation_object(workflow, action.command)

                # Validate that response_generation_object returns a CommandOutput, not a string
                if not isinstance(command_output, CommandOutput):
                    raise TypeError(
                        f"Response generation object for command '{action.command_name}' "
                        f"in context '{context}' did not return a CommandOutput "
                        f"(got {type(command_output).__name__}). This indicates an "
                        f"implementation error in the response generator."
                    )

                # Set the additional attributes
                command_output.workflow_name = workflow_name
                command_output.context = context
                command_output.command_call_id = call_id
                return command_output

            # Always resolve the command's Signature class via create() so
            # validate_extracted_parameters (and db_lookup) run on the direct-action
            # path the same way they do on the NLU path. Validate even when
            # action.parameters is empty/falsy — context preconditions still apply.
            if action.parameters:
                input_obj = command_parameters_class(**action.parameters)
            else:
                input_obj = command_parameters_class()

            input_for_param_extraction = InputForParamExtraction.create(
                workflow, action.command_name, action.command
            )
            is_valid, error_msg, _, _ = input_for_param_extraction.validate_parameters(
                workflow, action.command_name, input_obj
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid action parameters for command '{action.command_name}' "
                    f"in context '{context}'\n{error_msg}"
                )

            with tracing.call_scope(call_id, command_name=action.command_name):
                command_output = response_generation_object(workflow, action.command, input_obj)

            # Validate that response_generation_object returns a CommandOutput, not a string
            if not isinstance(command_output, CommandOutput):
                raise TypeError(
                    f"Response generation object for command '{action.command_name}' "
                    f"in context '{context}' did not return a CommandOutput "
                    f"(got {type(command_output).__name__}). This indicates an "
                    f"implementation error in the response generator."
                )
        except BaseException as exc:
            # First writer wins, so a nested dispatch that already named itself
            # keeps its name and this outer frame does not overwrite it. Not
            # gated on tracing.is_control_signal: _invoke_command_impl stamps
            # control signals too, and two dispatch seams disagreeing about
            # which exceptions carry identity is the drift fix-ajv.19 removed
            # from the status mapping.
            _annotate_exception(
                exc,
                _fw_command_name=action.command_name,
                _fw_workflow_name=workflow_name,
                _fw_context=context,
            )
            raise

        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_call_id = call_id

        return command_output

    # MCP-compliant methods
    @classmethod
    def perform_mcp_tool_call(
        cls,
        workflow: fastworkflow.Workflow,
        tool_call: fastworkflow.MCPToolCall,
        command_context: str = '*'
    ) -> fastworkflow.MCPToolResult:
        """
        MCP-compliant tool execution method.
        
        Args:
            workflow: FastWorkflow workflow
            tool_call: MCP tool call request
            workitem_path: The context in which to execute the command. If None, it must be in the arguments.
            
        Returns:
            MCPToolResult: MCP-compliant result format
        """
        try:
            context = tool_call.arguments.get('workitem_path', command_context)
            if not context:
                # ido-cex.7: this arm is swallowed into an MCPToolResult below,
                # where the annotation cannot travel, so the tool name has to be
                # in the text or the caller gets a refusal that names nothing.
                raise ValueError(
                    f"Context ('workitem_path') must be provided for the MCP tool "
                    f"call '{tool_call.name}'."
                )

            # Arch §10.4: this used to read the requested context and then
            # ignore it — every MCP call ran in whatever context the workflow
            # happened to be in, whatever the caller asked for. In P0 the
            # requested context may only be the current effective context;
            # anything else is a typed not-callable-here rather than a silent
            # dispatch somewhere the caller did not ask for.
            if refusal_source := cls.check_capability(
                workflow, tool_call.name, requested_context=context
            ):
                if refusal := cls._capability_refusal(
                    workflow, tool_call.name, refusal_source
                ):
                    return refusal.to_mcp_result()

            # Convert MCP tool call to FastWorkflow Action using helper method
            action = fastworkflow.Action(
                command_name=tool_call.name,
                command=tool_call.arguments.get('command', ''),
                parameters=dict(tool_call.arguments.items()),
            )

            # Execute using existing perform_action method
            command_output = cls.perform_action(workflow, action)

            # Convert to MCP format
            return command_output.to_mcp_result()

        except Exception as e:
            # Return error in MCP format. ido-cex.7: the MCP result is a flat
            # string with no command_name field, and this handler is the end of
            # the line for the identity perform_action stamped on the exception
            # — so it is spelled into the text here. Without it every MCP
            # failure read "Error: <message>" with nothing saying which tool
            # produced it.
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(
                    type="text",
                    text=f"Error in MCP tool '{tool_call.name}': {str(e)}",
                )],
                isError=True
            )
