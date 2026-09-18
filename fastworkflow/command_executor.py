import fastworkflow
from fastworkflow import auto_navigation, tracing
from fastworkflow.command_interfaces import CommandExecutorInterface

from fastworkflow import Action, CommandOutput, ChatSession
from fastworkflow.observability.execution_recorder import record_execution, recorder_for
from fastworkflow import ModuleType
from fastworkflow.utils.signatures import InputForParamExtraction
from pathlib import Path
from fastworkflow.command_routing import RoutingDefinition
from typing import Any, Optional
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.auto_navigation import AUTO_NAVIGATION_ARTIFACT


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
        *,
        auto_navigation_step: Optional[dict] = None,
    ) -> fastworkflow.CommandOutput:
        """Run one command as an execute step.

        ``auto_navigation_step`` is set only on the two steps an auto-navigation
        dispatch composes (ido-8ps.9 part b): the declared entry command, then
        the command the agent originally sent. Both go through this method, so
        both get an ordinary ``fw.command.execute`` span, an ordinary execution
        record and an ordinary observation -- the A1/A2 conventions hold because
        nothing about the step path is special. The only difference is the four
        attributes naming the dispatch on the span.
        """
        claim_check = getattr(
            chat_session, "assert_experiment_claim_current", None
        )
        if claim_check is not None:
            claim_check()
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

        # ido-8ps.9 part b. Written with literal keys, like every other
        # attribute on this span, so the span-contract scan can recover them
        # statically -- a key it cannot read is a contract nobody checked. They
        # are absent on an ordinary step: "the agent typed this" and "the
        # framework composed this" must be different span shapes, not the same
        # shape with a False in it.
        navigation_attributes: dict = {}
        if auto_navigation_step:
            navigation_attributes = {
                "auto_navigated": True,
                "auto_navigation_rule": auto_navigation_step.get(
                    "auto_navigation_rule"),
                "entered_context": auto_navigation_step.get("entered_context"),
                "auto_navigation_step": auto_navigation_step.get(
                    "auto_navigation_step"),
            }

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
                **navigation_attributes,
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

        # The context this dispatch started in, by name. Cheap (a class name)
        # and read outside the span gate, because the registry below needs it
        # whether or not anything is being traced.
        context_name_before = cls._context_name(chat_session)

        # ...and the context OBJECT it started in (ido-8yb/F11). A class name
        # cannot tell "still in Account" from "in a different Account": an
        # inherited or root open-by-identifier command moves straight from one
        # instance to the next with the name unchanged. The reference is a local
        # of this dispatch and dies with it; it is held rather than an `id()`
        # kept precisely so `is` below cannot be fooled by a reused address.
        context_instance_before = cls._context_instance(chat_session)

        # ido-8ps.13: the context this command RAN IN, recorded against the
        # execute step's own O alias BEFORE the command can move the context.
        # Outside the span gate for the same reason as the line above: this is
        # runtime presentation, not capture, and a run with tracing off must
        # print the same observation.
        cls._remember_execute_context(chat_session)

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
                # that could not say which command it covered. fix-ajv.16 FW-3.
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={"error_type": type(exc).__name__},
            )
            if not is_control_signal:
                # A failure is still a dispatch that happened. Without the id,
                # the outcome the caller builds from `exc` cannot be joined to
                # this span or to the record below. fix-ajv.16 FW-1.
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

        # ido-8ps.9 part a: the turn-scoped registry rule 3 reads. Recorded
        # unconditionally, not only when a span opened -- the registry is
        # runtime behaviour, not capture, and a run with tracing off must
        # resolve the same handles. It is a lookup table from a handle the agent
        # can WRITE to the context instance it denotes; nothing scans it for
        # what happened recently.
        cls._remember_context_entry(
            chat_session, command_output, context_name_before,
            context_instance_before)

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

    @classmethod
    def _context_name(cls, chat_session: 'fastworkflow.ChatSession') -> Optional[str]:
        """The current command context's name, or None. Never raises."""
        try:
            workflow = cls._active_workflow(chat_session)
            return None if workflow is None else workflow.current_command_context_name
        except Exception:
            return None

    @classmethod
    def _context_instance(cls, chat_session: 'fastworkflow.ChatSession') -> Any:
        """The current command context OBJECT, or None. Never raises.

        The framework has no portable identity for a context instance
        (`tracing.context_handle` says so and degrades to a type-only handle),
        but WITHIN one dispatch the object itself is an identity, and it is the
        same one `Workflow.current_command_context`'s setter uses to decide
        whether the context changed at all. Nothing is captured from it: it is
        compared with `is` and dropped.
        """
        try:
            workflow = cls._active_workflow(chat_session)
            return None if workflow is None else workflow.current_command_context
        except Exception:
            return None

    @classmethod
    def _entry_contract(
        cls, chat_session: 'fastworkflow.ChatSession', context_name: str
    ) -> Optional[auto_navigation.EntryContract]:
        """The entry contract *context_name* declares, or None. Never raises.

        The context model's own answer to two questions this recorder has: does
        this command ENTER this context (ido-8yb/F11), and which of its
        parameters IDENTIFY the instance (ido-nx6/F31). One lookup, because two
        readings of one declaration is one reading plus a drift. None covers
        every way a context declines to be auto-entered, and a context rule 3
        could not dispatch to is a context this table has nothing to say about.
        """
        try:
            workflow = cls._active_workflow(chat_session)
            folderpath = getattr(workflow, "folderpath", None)
            return (
                auto_navigation.entry_contract_for(folderpath, context_name)
                if folderpath else None
            )
        except Exception:
            return None

    @classmethod
    def _remember_execute_context(
        cls, chat_session: 'fastworkflow.ChatSession'
    ) -> None:
        """File the context-at-execution for this step's alias line (ido-8ps.13).

        The context is taken here, before dispatch, and the rule this fixes is
        stated in ``docs/observation_search.md``: a command that MOVES the
        context is printed with the context it RAN IN. ``open_account_by_uid``
        therefore reads as the DirectoryExplorer command it is, and the
        ``list_permissions`` that follows it reads as the account's.

        Silent and best-effort from end to end: with no agent step in flight
        there is no ``O`` namespace to file under, and a failure to describe a
        context must never fail the command that ran in it.
        """
        try:
            from fastworkflow.context_identity import context_clause_for
            from fastworkflow.observation_offloading.state import record_context_clause
            from fastworkflow.result_handles import current_execute_alias, current_scope

            alias = current_execute_alias()
            if not alias:
                return
            record_context_clause(
                current_scope(), alias,
                context_clause_for(cls._active_workflow(chat_session)))
        except Exception:  # noqa: BLE001 - presentation must never fail a turn
            pass

    @classmethod
    def _remember_context_entry(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command_output: fastworkflow.CommandOutput,
        context_name_before: Optional[str],
        context_instance_before: Any = None,
    ) -> None:
        """Record a context this command entered, for rule 3 to look up later.

        Only a SUCCESSFUL command that actually moved the context is recorded,
        and only the three facts rule 3 rebuilds an entry from: the context, the
        parameter values that entered it, and the ``O`` alias of this step. It is
        never consulted except to answer "what does this handle denote".

        "Moved the context" is an INSTANCE transition, not a type change
        (ido-8yb/F11). It used to be read off the class name alone, so a valid
        inherited or root ``open_account_by_uid`` run from inside Account A --
        landing in Account B, both named Account -- recorded nothing, and B's
        printed ``O`` alias could not resolve afterwards even though the agent
        had entered B in this very turn.

        A move that keeps the class name is recorded only when the workflow
        DECLARES this command as that context's entry command. An ordinary
        command that happens to rebuild or swap the context object is not an
        entry and has no business in a table of "what does this handle denote";
        a move that also changes the class name keeps the older, looser rule, so
        nothing that used to be recorded stops being recorded.
        """
        try:
            if not command_output.success:
                return
            if cls._was_auto_navigated(command_output):
                # ido-91o (F14). This frame is the OUTER frame of a two-step
                # dispatch: `_auto_navigate` ran the declared entry command and
                # then the original command through `invoke_command`, and the
                # entry step recorded the entry itself -- with the ENTRY
                # command's name and the values that entered the context. This
                # frame sees the context moved (it started outside) and the
                # ORIGINAL command's name and parameters, so recording here
                # files a second entry for one entry, under the same `O` alias.
                # When the original command happens to carry the entry
                # contract's required parameter names with other values, rule 3
                # then sees two entries behind one handle and declines as
                # ambiguous -- for a handle this very dispatch produced.
                return
            context_name_after = cls._context_name(chat_session)
            if not context_name_after:
                return
            type_changed = context_name_after != context_name_before
            if not type_changed:
                # Cheapest question first: the overwhelming majority of commands
                # move nothing, and they must not pay for a context-model read.
                instance_after = cls._context_instance(chat_session)
                if instance_after is None or instance_after is context_instance_before:
                    return
            command_name = (command_output.command_name or "").split("/")[-1]
            contract = cls._entry_contract(chat_session, context_name_after)
            if not type_changed and (
                contract is None or contract.command_name != command_name
            ):
                return
            parameters = command_output.command_parameters
            if hasattr(parameters, "model_dump"):
                parameters = parameters.model_dump()
            if not isinstance(parameters, dict):
                parameters = {}
            from fastworkflow.result_handles import current_execute_alias, current_scope

            # The scope OBJECT, not only its id (ido-dhw): the durable copy of
            # this entry has to carry the channel and experiment it belongs to,
            # which is what the sidecar's erasure and retention read.
            scope = current_scope()
            auto_navigation.record_context_entry(
                scope.scope_id,
                scope=scope,
                context=context_name_after,
                command_name=command_name,
                parameters=parameters,
                alias=current_execute_alias(),
                # Which of those parameters IDENTIFY the instance, as the entry
                # contract states it. Without this the registry publishes every
                # value it was handed -- defaults included -- as a handle
                # (ido-nx6/F31).
                required_parameters=(
                    () if contract is None else contract.required_parameters),
            )
        except Exception:  # noqa: BLE001 - never fail a command over the registry
            pass

    @staticmethod
    def _was_auto_navigated(command_output: fastworkflow.CommandOutput) -> bool:
        """Did this output come back from a two-step dispatch (ido-91o/F14)?

        The marks are put on the artifacts of the FINAL output by
        ``_auto_navigate`` and nowhere else; the two inner steps carry them on
        their spans only, so the entry step still records its entry. Never
        raises: an output whose artifacts cannot be read is treated as an
        ordinary one, which is what it was before this check existed.
        """
        try:
            artifacts = command_output.command_response.artifacts or {}
            return bool(artifacts.get(auto_navigation.ATTR_AUTO_NAVIGATED))
        except Exception:  # noqa: BLE001
            return False

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
            # ido-8ps.9 part b: the CME decided this utterance names a command
            # owned by a context the walk cannot reach, and that the context
            # model says how to enter it without guessing. Running the plan is
            # this frame's job: the CME has the workflow but not the session,
            # and only the session can put a command through the ordinary step
            # path.
            plan = (command_output.command_response.artifacts or {}).get(
                AUTO_NAVIGATION_ARTIFACT)
            if plan:
                return cls._auto_navigate(chat_session, plan)
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

        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
            workflow.folderpath
        )

        response_generation_class = command_routing_definition.get_command_class(
            command_name,
            ModuleType.RESPONSE_GENERATION_INFERENCE,
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{command_name}' "
            )
        response_generation_object = response_generation_class()

        raw_user_message = command
        if "raw_user_message" in workflow.context:
            raw_user_message = workflow.context['raw_user_message']

        try:
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
            # fix-ajv.16 FW-2.
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
    def _auto_navigate(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        plan: dict,
    ) -> fastworkflow.CommandOutput:
        """[enter the owning context; run the original command], in that order.

        Both steps go through ``invoke_command``, so each is a real execute step
        with its own span, its own execution record and its own response text;
        the agent sees both, labelled, in the observation of the tool call it
        made. An entry step that fails, or that succeeds WITHOUT moving the
        context, STOPS the dispatch: the original command is not run in the
        context that had already declined it.

        ``auto_navigation.dispatching`` marks the two inner steps so that a
        foreign name declined INSIDE them cannot start a second dispatch: the
        rule composes two steps, not a search.
        """
        rule = plan.get("rule")
        entered = plan.get("entered_context")
        entry_utterance = str(plan.get("entry_utterance") or "")
        original_utterance = str(plan.get("original_utterance") or "")
        marks = {
            auto_navigation.ATTR_AUTO_NAVIGATED: True,
            auto_navigation.ATTR_AUTO_NAVIGATION_RULE: rule,
            auto_navigation.ATTR_ENTERED_CONTEXT: entered,
        }
        banner = (
            f"[auto-navigation rule {rule}] '{plan.get('command_name')}' is owned by "
            f"the {entered} context; entered it with '{entry_utterance}'."
        )
        # Deliberately NOT called `context_before`: that name belongs to the
        # observability context PROJECTION (`tracing.context_handle`, line 138),
        # and `tests/test_no_capture_control_flow.py` forbids any condition that
        # mentions it, because a captured value reaching control flow is EXP-003's
        # stop condition. This is a context class NAME, read for the dispatch's own
        # did-it-move check and never captured.
        context_at_dispatch = cls._context_name(chat_session)
        with auto_navigation.dispatching():
            entry_output = cls.invoke_command(
                chat_session, entry_utterance,
                auto_navigation_step={
                    **marks,
                    auto_navigation.ATTR_AUTO_NAVIGATION_STEP:
                        auto_navigation.STEP_ENTRY,
                },
            )
            entry_text = entry_output.command_response.response or ""
            # The declaration said this command enters that context. Offline,
            # the validator can only check that the command exists and could be
            # run from here -- whether it MOVES the context is a runtime fact,
            # and this is where it becomes one. A declared entry command with an
            # optional identifier (IDO's `open_finding <finding_uid>`, whose
            # parameter has a default) succeeds while listing rather than
            # entering; running the original command after that would be running
            # it in the context that already declined it. A session that
            # cannot name its context either way is not evidence of
            # anything, so the dispatch proceeds.
            entered_context_name = cls._context_name(chat_session)
            did_not_move = (
                context_at_dispatch is not None
                and entered_context_name is not None
                and entered_context_name == context_at_dispatch
            )
            if not entry_output.success or did_not_move:
                why = "did not enter" if entry_output.success else "could not enter"
                entry_output.command_response.response = (
                    f"[auto-navigation rule {rule}] '{entry_utterance}' {why} the "
                    f"{entered} context, so '{plan.get('command_name')}' was not "
                    f"run. Enter {entered} yourself, then run it.\n{entry_text}"
                )
                entry_output.command_response.artifacts["command_handled"] = True
                entry_output.command_response.success = False
                return entry_output

            final_output = cls.invoke_command(
                chat_session, original_utterance,
                auto_navigation_step={
                    **marks,
                    auto_navigation.ATTR_AUTO_NAVIGATION_STEP:
                        auto_navigation.STEP_ORIGINAL,
                },
            )
        final_text = final_output.command_response.response or ""
        final_output.command_response.response = (
            f"{banner}\n{entry_text}\n\n{final_text}".strip()
        )
        final_output.command_response.artifacts.update(marks)
        return final_output

    @classmethod
    def perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        workflow.command_context_for_response_generation = \
            workflow.current_command_context

        # One id per dispatch through this method (arch §12.0 delta 1). It is
        # the outermost id on the direct-action, startup-action and MCP paths,
        # and a child id on the internal CME hop that invoke_command makes — in
        # which case call_scope files it under the enclosing command call.
        call_id = tracing.new_command_call_id()

        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow.current_command_context_displayname
        
        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)

        response_generation_class = (
            command_routing_definition.get_command_class(
                action.command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{action.command_name}'"
            )

        response_generation_object = response_generation_class()

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            try:
                with tracing.call_scope(call_id, command_name=action.command_name):
                    command_output = response_generation_object(workflow, action.command)
            except BaseException as exc:
                # Same seam as `_invoke_command_impl`, for the entry point the
                # other one never reaches: perform_action is the direct-action,
                # startup-action and MCP path, and it is also the CME hop
                # invoke_command makes with command_name="wildcard". Routing is
                # already decided here — `action.command_name` names it, and
                # `workflow_name`/`context` were bound above — but the assignments
                # that publish them onto the CommandOutput sit below the raise, so
                # on failure they died with the frame. Carry them out on the
                # exception and re-raise unconditionally: this seam observes, it
                # never handles. First-writer-wins means a nested dispatch that
                # already stamped a more specific identity keeps it. fix-ajv.16
                # FW-2, extended to this path.
                _annotate_exception(
                    exc,
                    _fw_command_name=action.command_name,
                    _fw_workflow_name=workflow_name,
                    _fw_context=context,
                )
                raise
            
            # Validate that response_generation_object returns a CommandOutput, not a string
            if not isinstance(command_output, CommandOutput):
                raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")
                
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
                f"Invalid action parameters for command '{action.command_name}'\n{error_msg}"
            )

        try:
            with tracing.call_scope(call_id, command_name=action.command_name):
                command_output = response_generation_object(workflow, action.command, input_obj)
        except BaseException as exc:
            # Same seam as `_invoke_command_impl`, for the entry point the
            # other one never reaches: perform_action is the direct-action,
            # startup-action and MCP path, and it is also the CME hop
            # invoke_command makes with command_name="wildcard". Routing is
            # already decided here — `action.command_name` names it, and
            # `workflow_name`/`context` were bound above — but the assignments
            # that publish them onto the CommandOutput sit below the raise, so
            # on failure they died with the frame. Carry them out on the
            # exception and re-raise unconditionally: this seam observes, it
            # never handles. First-writer-wins means a nested dispatch that
            # already stamped a more specific identity keeps it. fix-ajv.16
            # FW-2, extended to this path.
            _annotate_exception(
                exc,
                _fw_command_name=action.command_name,
                _fw_workflow_name=workflow_name,
                _fw_context=context,
            )
            raise
        
        # Validate that response_generation_object returns a CommandOutput, not a string
        if not isinstance(command_output, CommandOutput):
            raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")
        
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
                raise ValueError("Context ('workitem_path') must be provided for an MCP tool call.")

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
            # Return error in MCP format
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
