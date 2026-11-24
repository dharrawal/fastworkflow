import contextlib
import inspect
import logging
from contextlib import nullcontext

import fastworkflow
from pydantic import BaseModel
from fastworkflow import tracing
from fastworkflow import approval_binding, command_writes, intent_loop, waiting
from fastworkflow.command_interfaces import CommandExecutorInterface
from fastworkflow.command_declarations import declared_descend

from fastworkflow import Action, CommandOutput, ChatSession
from fastworkflow.observability.execution_recorder import record_execution, recorder_for
from fastworkflow import ModuleType
from fastworkflow.utils.python_utils import legacy_module_names
from fastworkflow.utils.signatures import InputForParamExtraction, parameters_with_unfilled
from pathlib import Path
from fastworkflow.command_routing import RoutingDefinition
from typing import Optional
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.workflow import running_command


def _declared_output_model(response_generator_class) -> type | None:
    """The command's own ``Signature.Output`` class, or None when its module declares none.

    Read from the globals of the ``__call__`` that the class defines, so it does not depend on the
    module being registered in ``sys.modules``.
    """
    call = getattr(response_generator_class, "__call__", None)
    namespace = getattr(call, "__globals__", None) or {}
    signature = namespace.get("Signature")
    if signature is None:
        signature = getattr(inspect.getmodule(response_generator_class), "Signature", None)
    output = getattr(signature, "Output", None)
    return output if inspect.isclass(output) and issubclass(output, BaseModel) else None


def paging_line(output, command_parameters) -> str:
    """The paging line a framework-written response ends with when rows remain (design 3.14): the
    rows read, their total and the next page's paging value, e.g. ``Rows 1-20 of 57; next page:
    start=20.`` Empty when the Output is not a page, the command has no paging parameter, or no rows remain.
    """
    from fastworkflow.observation_offloading.facts import output_page, paging_parameters

    paging = paging_parameters(type(command_parameters)) if command_parameters is not None else ()
    page = output_page(output)
    if not paging or page is None or page[0] == 0:
        return ""
    rows, total = page
    given = command_parameters.model_dump(exclude_unset=True) if command_parameters is not None else {}
    name = next((n for n in paging if n in given), paging[0])
    start = getattr(command_parameters, name, None) or 0
    if not isinstance(start, int) or isinstance(start, bool):
        start = 0
    after = start + rows
    if after >= total:
        return ""
    return f"Rows {start + 1}-{after} of {total}; next page: {name}={after}."


def _declared_output_message(command_name: str, returned: str, returned_by: str | None = None) -> str:
    """The call error for a command that declares Signature.Output and returned something else."""
    verb = f"its {returned_by} returned" if returned_by else "returned"
    return (f"{command_name} declares an Output but {verb} {returned}; return the Output, "
            "or CommandResponse(response=..., output=<Output>)")


def command_response_of(result, *, command_name: str, output_model: type | None,
                        returned_by: str | None = None) -> fastworkflow.CommandResponse:
    """The one rule for what a command may return, as a ``CommandResponse`` (design 3.14).

    Applied to a ``ResponseGenerator.__call__`` (by ``frame_command_result``) and to a planning command's
    ``__result__`` (``returned_by="__result__"``, which only words the error). The short forms, and only these:

    - the command's own ``Signature.Output``, exactly that class: ``response`` is the Output as JSON and
      ``output`` the instance;
    - a ``str``: the text of the response;
    - a ``CommandResponse``: itself (its artifacts and next actions), with ``success`` left to the framework. A
      ``CommandResponse`` with ``success=False`` is a call error: a refusal is ``raise fastworkflow.Refused(message)``.

    A command that declares ``Signature.Output`` must return it, directly or as
    ``CommandResponse(response=..., output=<Output>)``, and the class is exact in both forms (a subclass instance is
    another model, as a different model is). A str, or a CommandResponse without that output, is a call error
    (fix-3rsq.declared-output-required). Anything else, a hand-built ``CommandOutput`` included, fails the call,
    naming the command and the type it returned. A ``TypeError`` is raised for each.
    """
    if isinstance(result, fastworkflow.CommandResponse):
        if result.success is False:
            raise TypeError(
                f"{command_name} returned a CommandResponse with success=False: success is the framework's "
                "to set. Refuse with raise fastworkflow.Refused(message)"
            )
        if output_model is not None and type(result.output) is not output_model:
            returned = "CommandResponse without output" if result.output is None else (
                f"CommandResponse(output={type(result.output).__qualname__})")
            raise TypeError(_declared_output_message(command_name, returned, returned_by))
        return result
    if isinstance(result, str):
        if output_model is not None:
            raise TypeError(_declared_output_message(command_name, "str", returned_by))
        return fastworkflow.CommandResponse(response=result)
    if output_model is not None and type(result) is output_model:
        return fastworkflow.CommandResponse(response=result.model_dump_json(), output=result)
    # the module as the workflow's name for it (a namespace digest written back to the bare basename)
    returned = legacy_module_names(f"{type(result).__module__}.{type(result).__qualname__}")
    if output_model is not None:
        raise TypeError(_declared_output_message(command_name, type(result).__qualname__, returned_by))
    declared = "it declares no Signature.Output" if returned_by else "its ResponseGenerator declares no Signature.Output"
    source = f"its {returned_by}" if returned_by else "its ResponseGenerator"
    raise TypeError(
        f"{command_name} returned {returned}, which is not a short form {source} may return "
        f"({declared}): return that Output, a str or a CommandResponse (artifacts or next actions), "
        "or raise fastworkflow.Refused(message)"
    )


def frame_command_result(result, *, command_name: str, output_model: type | None,
                         command_parameters=None) -> CommandOutput:
    """Wrap what a ``ResponseGenerator.__call__`` returned in a CommandOutput (design 3.14).

    What may be returned is judged by ``command_response_of``, the one rule ``__result__`` follows too. What is
    specific to an execution command is added here: a directly returned Output gets the paging line (``paging_line``)
    after its JSON.
    """
    response = command_response_of(result, command_name=command_name, output_model=output_model)
    if response is result or not isinstance(result, BaseModel):
        return CommandOutput(command_response=response)
    line = paging_line(result, command_parameters)
    if line:
        response.response = f"{response.response}\n{line}"
    return CommandOutput(command_response=response)


def _assistant_round(chat_session) -> bool:
    """The round was started by a "/" message (the CME's port says so)."""
    return waiting.port_of(chat_session.cme_workflow.context).assistant


def _end_round(chat_session) -> None:
    """The "/" round ends (the one place a command's run clears the mark)."""
    waiting.port_of(chat_session.cme_workflow.context).end_round()


#: The refusal of a command that writes while a part the user asked only to look up runs (bd fix-ftdl.4.3).
LOOK_UP_ONLY_REFUSAL = (
    "{command} was not run: it changes the application's data, and you asked for this part only to be looked up.")


def _look_up_only_refusal(chat_session, workflow_folderpath: str, command_name: str) -> Optional[str]:
    """The refusal for running *command_name* now, or None: a command that writes (``command_writes.write_commands``),
    called while the turn's part that runs is marked read-only (``SubTask.read_only``: the user answered the review
    with "look it up"). Code only, and only as complete as the ``@writes`` marks: a command that writes without the
    mark is not known to write."""
    host = command_writes._host(chat_session)
    running = getattr(host, "_running_part_item", None)
    item = running() if callable(running) else None
    if item is None or not getattr(item, "read_only", False):
        return None
    if command_name not in command_writes.write_commands(workflow_folderpath):
        return None
    return LOOK_UP_ONLY_REFUSAL.format(command=command_name)


def _pending_before_cme_reply(workflow) -> dict:
    """What the CME workflow was in the middle of, read before the router runs (an abort clears it): what it was
    waiting for (``intent_loop.waiting_for``), the command text being processed and the command whose parameters were
    being extracted. Empty when nothing."""
    try:
        from fastworkflow import intent_loop

        context = workflow.context
        pending = {"waiting_for": intent_loop.waiting_for(context),
                   "command": context.get("command"), "command_name": context.get("command_name")}
        return {key: value for key, value in pending.items() if value is not None}
    except Exception:  # noqa: BLE001
        return {}


def _cme_reply_artifacts(output, pending: dict | None) -> dict:
    """Artifacts of the CME replies that were text only: ``aborted`` (what ``abort`` ended, from *pending*, taken before
    it ran) and ``command_choices`` (the command names ``you_misunderstood`` offers, in the order it lists them);
    ``capabilities`` for ``what_can_i_do``. Read from the nested command's typed Output."""
    from fastworkflow.capability_answer import CapabilityListing

    if isinstance(output, CapabilityListing):
        return {"capabilities": output.model_dump(mode="json")}
    if getattr(output, "command_name", None) == "abort" and hasattr(output, "command"):
        return {"aborted": dict(pending or {})}
    if isinstance(getattr(output, "valid_command_names", None), list):
        return {"command_choices": list(output.valid_command_names)}
    return {}


def _no_agent_note(route, workflow) -> str:
    """One plain sentence on a request the deterministic path could not route (an ambiguous intent, or no command), when
    the session has no agent model (``no_agent_mode``): why it was not planned and what to do. Added here, not in the
    router's command file, so the internal workflow's training fingerprint (a hash of those files) does not move.
    Empty in agent mode, for an agent's own call, and for any reply that is not an unrouted request."""
    if workflow is None or route.handled or route.routed or route.command_name is not None \
            or route.parameter_request is not None or route.plan_choice:
        return ""    # a reply that offers candidate commands needs a pick, not an explanation of the mode
    try:
        context = workflow.context
        app_workflow = context.get("app_workflow")
        agentic = (not waiting.port_of(context).assistant
                   and app_workflow is not None and bool(app_workflow.context.get("run_as_agent")))
        if agentic or not context.get("no_agent_mode"):
            return ""
    except Exception:  # noqa: BLE001 - a note is never worth failing a reply
        return ""
    from fastworkflow.no_agent_mode import NOT_PLANNED_NOTE
    return f"\n{NOT_PLANNED_NOTE}"


def _wildcard_command_output(framed: CommandOutput, pending: dict | None = None, workflow=None) -> CommandOutput:
    """The turn's CommandOutput for the CME router, built from its typed routing result (fix-3rsq.12.3).

    The router (``_workflows/command_metadata_extraction/_commands/wildcard.py``) returns its
    ``Signature.Output``; the framework frames it here. ``success`` is ``routed``; a handled answer
    carries the artifact ``command_handled``; a routed answer carries the extraction the turn reads
    (``command``, ``command_name``, ``cmd_parameters`` and, when the command ran on an enclosing
    record, ``ran_on_context_name`` / ``ran_on_context_clause``). This is what the router returned
    before it returned its Output: the same response text, success and artifacts.
    """
    route = framed.command_response.output
    if route is None or not all(hasattr(route, name) for name in ("handled", "routed", "response")):
        raise TypeError(
            "wildcard must return its Signature.Output (the routing result), "
            f"not {type(route).__qualname__}"
        )
    artifacts: dict = {}
    if route.handled:
        artifacts["command_handled"] = True
    if route.command_name is not None:
        artifacts["command"] = route.command
        artifacts["command_name"] = route.command_name
        artifacts["cmd_parameters"] = route.cmd_parameters
    if route.ran_on_context_clause is not None:
        artifacts["ran_on_context_name"] = route.ran_on_context_name
        artifacts["ran_on_context_clause"] = route.ran_on_context_clause
    if route.reply_intent is not None:
        artifacts["reply_intent"] = route.reply_intent
    if route.parameter_request is not None:
        artifacts["parameter_request"] = route.parameter_request
        # Tagged as the same kind the waiting record of a parameters question has everywhere (``waiting.TABLE``).
        artifacts["question_kind"] = waiting.TABLE[waiting.KIND_PARAMETERS].question_kind
    if route.plan_choice:
        # The which-command question, in the one shape every path asks it (`intent_loop.prompt_artifacts`).
        artifacts["question_kind"] = "plan_choice"
        artifacts["plan_choice"] = route.plan_choice
    artifacts.update(_cme_reply_artifacts(route.output, pending))
    return CommandOutput(
        command_response=fastworkflow.CommandResponse(
            response=route.response + _no_agent_note(route, workflow),
            # A which-command question that is out is not a success: nothing ran and the user has still to answer (the
            # one rule of the plan path's question and of the ambiguous one: ``success`` False, the question as data).
            success=route.routed and not route.plan_choice,
            artifacts=artifacts,
            output=route.output,
        )
    )


def call_response_generator(response_generator, workflow, command, command_parameters, *, command_name: str = ""):
    """Call ``response_generator(workflow, command, command_parameters)`` and frame its result.

    ``command_parameters`` is the filled ``Input``, or None for a command with no ``Input``. A
    ``fastworkflow.Refused`` the call raises is its answer: a failed CommandOutput with the refusal's
    message, and the fields it named as the artifact ``refused_fields`` (``Refused.as_output``; the command ran, so
    nothing is asked again). Any other return is framed by ``frame_command_result``.
    """
    before = getattr(workflow, "_current_command_context", None)
    assignments = getattr(workflow, "_record_assignments", 0)
    try:
        with running_command():     # workflow.record is the invoked record while the command runs
            result = response_generator(workflow, command, command_parameters)
    except fastworkflow.Refused as refusal:
        _restore_record(workflow, before)
        return refusal.as_output()
    except Exception:
        # a command that fails leaves the record where it was; an ask_user suspension (a BaseException) resumes
        # where the command stood, so it keeps its move
        _restore_record(workflow, before)
        raise
    if (refused_move := _check_move(workflow, response_generator, command_name, before, assignments)) is not None:
        return refused_move
    framed = frame_command_result(
        result,
        command_name=command_name or type(response_generator).__qualname__,
        output_model=_declared_output_model(type(response_generator)),
        command_parameters=command_parameters,
    )
    if command_name.endswith(WHAT_CAN_I_DO_COMMAND) and isinstance(result, str):
        _attach_capability_listing(framed, workflow)
    return framed


#: The CME command whose listing is also data. Its command file is part of the internal workflow's training
#: fingerprint, so the data is attached here and not in the file (fix-szfq).
WHAT_CAN_I_DO_COMMAND = "IntentDetection/what_can_i_do"


def _attach_capability_listing(framed: CommandOutput, workflow) -> None:
    """Give a ``what_can_i_do`` answer its listing as data: ``CommandResponse.output`` (a ``CapabilityListing``) and the
    artifact ``capabilities``. The response text is untouched. Best effort: a listing that cannot be built is absent."""
    try:
        from fastworkflow.capability_answer import what_can_i_do_data

        agent_mode = False
        with contextlib.suppress(Exception):
            agent_mode = bool(fastworkflow.chat_session and fastworkflow.chat_session.run_as_agent)
        listing = what_can_i_do_data(workflow, for_agents=agent_mode)
        framed.command_response.output = listing
        framed.command_response.artifacts["capabilities"] = listing.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - never fails the command
        logging.getLogger(__name__).debug("capability listing not attached", exc_info=True)


logger = logging.getLogger(__name__)

def _restore_record(workflow, before) -> None:
    """Put *workflow* back on the record it stood on before a command that moved it failed."""
    if workflow is not None and getattr(workflow, "_current_command_context", before) is not before:
        workflow.record = before


def _check_move(workflow, response_generator, command_name: str, before,
                assignments: int) -> Optional[CommandOutput]:
    """The check of a command that moved the session, after it returned: the failed output, or None.

    A command moved the session when it assigned ``workflow.record`` (``Workflow._record_assignments`` changed while it
    ran) and the session no longer stands on the record it stood on before. Setting ``workflow.root`` (the startup
    command) is a different statement and is no move; a command that declares a move and does not make it has not
    moved. A move to no record (the global scope) is onto no context. A move onto a record fails the command, and the
    record is put back, when:
    - the record's class is no context of the workflow (no ``_commands/<ClassName>/`` folder, no entry of the
      inheritance file);
    - the command's ``@descends_to``, read from the generator class that ran, does not name the record's context or a
      context that context inherits from in ``context_inheritance_model.json`` (a PremiumUser is a User): one rule,
      "a command may move the session only onto a context it declares";
    - the workflow has a root, the record is not it, and its context has no ``get_parent``
      (``Workflow.missing_get_parent``), so the next climb could not go on.
    The framework's own navigation (go_up, reset_context: the core commands, known by name from the command metadata
    extraction workflow, ``_core_command_names``) is not an author's move and is not checked against a declaration.
    """
    if workflow is None or getattr(workflow, "_record_assignments", assignments) == assignments:
        return None
    after = getattr(workflow, "_current_command_context", before)
    if after is before or after is None:
        return None
    target = type(after).__name__
    short = command_name.split("/")[-1] or type(response_generator).__module__.rsplit(".", 1)[-1]
    try:
        folder = workflow.folderpath
        model = fastworkflow.CommandContextModel.load(folder)
        if not workflow.is_context(target, model):
            problem = (f"{short} moved the workflow onto a {target} record, but {target} is no context of the "
                       f"workflow: a context is the folder _commands/{target}/, named exactly as the record's class")
        elif command_name in _core_command_names():
            return None
        elif (declared := declared_descend(type(response_generator))) is None:
            problem = (f'{short} moved the workflow onto a {target} record but declares no @descends_to: a command may '
                       f'move the session only onto a context it declares. Declare @descends_to("{target}") on its '
                       f'ResponseGenerator')
        elif declared != target and declared not in model.inherited_base_contexts(target):
            problem = (f'{short} moved the workflow onto a {target} record but declares @descends_to("{declared}"), '
                       f'and {target} is not {declared} nor a context that inherits from it: a command may move the '
                       f'session only onto a context it declares')
        elif (missing := workflow.missing_get_parent(after)) is not None:
            problem = f"{short} moved the workflow onto a {target} record, but {missing}"
        else:
            return None
    except Exception as error:  # noqa: BLE001 - a move that cannot be checked is not taken on trust (fails closed)
        logger.warning("move check: the contexts of %r could not be read; %r fails", getattr(workflow, "folderpath", None),
                       command_name, exc_info=True)
        problem = (f"{short} moved the workflow onto a {target} record, but the move could not be checked: the "
                   f"workflow's contexts could not be read ({type(error).__name__})")
    _restore_record(workflow, before)
    return CommandOutput(command_response=fastworkflow.CommandResponse(
        response=f"{problem}. The move was undone", success=False))


def _owner_record(workflow, command_name: str):
    """The record a command reached by name runs on: the record of the nearest scope of the lookup path that owns it,
    the root for a global command, the current position for a framework command (``intent_loop.choose``, the decision a
    message's chosen command takes). A command no scope of the path has keeps the current position."""
    chosen = intent_loop.choose(workflow, command_name)
    return chosen.record if chosen is not None else workflow._current_command_context


def _ran_on(workflow, record) -> Optional[tuple[str, str]]:
    """``(context name, clause)`` of the record a command runs on when it is not the current position, else None: what
    the CME's walk reports for a message (``wildcard``), so a direct action is filed the same way."""
    if record is workflow._current_command_context:
        return None
    from fastworkflow.context_identity import identity_of
    from fastworkflow.observation_offloading.labels import context_clause

    name, label = identity_of(workflow, record)
    return name, (context_clause(name, label) if record is not None else "")


def _file_under(command_output: CommandOutput, ran_on: Optional[tuple[str, str]]) -> None:
    """File a run on an enclosing record (or the root, for a global command) under that record, as the message path
    does (``_invoke_command_impl``)."""
    if ran_on is None:
        return
    name, clause = ran_on
    command_output.context = name or "global"
    command_output.ran_in_context_name = name
    command_output.ran_in_context_clause = clause


def _core_command_names() -> frozenset[str]:
    # Lazy: workflow_agent imports this module.
    from fastworkflow.workflow_agent import core_command_names

    return core_command_names()


# ------------------------------------------------------------------
# Module-level delegation configuration and exceptions
# ------------------------------------------------------------------

MAX_DELEGATION_DEPTH: int = 10  # Safety limit for delegation hops


class CommandNotFoundError(Exception):
    """Raised when a command cannot be resolved in any accessible context."""


class AlreadyInContextError(CommandNotFoundError):
    """The command enters the context the caller is already in."""


def _command_name_token(text: str) -> str:
    parts = (text or "").strip().split(maxsplit=1)
    return parts[0].lstrip("/") if parts else ""


def _resolve_planning_command_name(workflow_folderpath: str, name: str) -> str | None:
    """Return the qualified planning command name *name* refers to, or None."""
    if not name:
        return None
    directory = CommandDirectory.load(workflow_folderpath)
    if name in directory.map_command_2_metadata:
        meta = directory.get_command_metadata(name)
        return name if meta.is_planning else None
    short = name.split("/")[-1].lower()
    matches = [
        qualified_name
        for qualified_name, meta in directory.map_command_2_metadata.items()
        if meta.is_planning and qualified_name.split("/")[-1].lower() == short
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _planning_command_refusal_output(
    qualified_name: str,
    *,
    workflow: fastworkflow.Workflow | None = None,
    message: str | None = None,
) -> CommandOutput:
    """The refusal for a planning command reached outside agent mode: the direct and MCP paths,
    and assistant-mode ``/`` commands that need the agent (``message`` then carries the reasons,
    design 3.4). In agent mode a planning command's steps run it
    (WorkflowExecutionContext.run_planning_command)."""
    if message is None:
        message = f"{qualified_name} is a planning command; it runs in agent mode."
    command_output = CommandOutput(
        command_response=fastworkflow.CommandResponse(response=message, success=False),
        command_name=qualified_name,
    )
    if workflow is not None:
        command_output.workflow_name = workflow.folderpath.split("/")[-1]
        command_output.context = workflow._current_command_context_displayname
    return command_output


def _assistant_planning_refusal(
    folderpath: str, qualified_name: str, workflow: fastworkflow.Workflow
) -> CommandOutput | None:
    """Assistant mode's answer for an exactly named planning command (fix-3rsq.10.8, design 3.4).

    None when the planning command is MECHANICAL: it then runs through the normal run path. A
    refusal (the 3.4 message with its reasons) when it NEEDS THE AGENT, or when it cannot be
    classified at all: a command the framework cannot read is refused rather than run.
    """
    from fastworkflow.planning_needs_agent import planning_command_needs_agent, refusal_message

    try:
        needs_agent, reasons = planning_command_needs_agent(folderpath, qualified_name)
    except ValueError as error:
        return _planning_command_refusal_output(
            qualified_name, workflow=workflow,
            message=f"{qualified_name} is a planning command; it cannot be checked for assistant "
                    f"mode: {error}.")
    if not needs_agent:
        return None
    return _planning_command_refusal_output(
        qualified_name, workflow=workflow, message=refusal_message(qualified_name, reasons))


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


def _policy_rewrite(chat_session, command_name: str, input_obj) -> Optional[str]:
    """The policy guard's rewrite for this command, or None to run it.

    The guard belongs to the agent serving the session right now, looked up at
    each call: distillation swaps agents mid-session, so a reference taken
    earlier could ask the wrong one. None whenever no policy is enabled.
    """
    from fastworkflow.policy_runtime import policy_of

    policy = policy_of(getattr(chat_session, "workflow_tool_agent", None))
    if policy is None:
        return None
    if input_obj is None:
        params = {}
    elif hasattr(input_obj, "model_dump"):
        try:
            params = input_obj.model_dump()
        except Exception:  # noqa: BLE001 - the guard is still consulted
            params = {}
    else:
        params = dict(input_obj) if isinstance(input_obj, dict) else {}
    return policy.guard(command_name, params)


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
        claim_check = getattr(
            chat_session, "assert_experiment_claim_current", None
        )
        if claim_check is not None:
            claim_check()
        # Whitespace-only text is the same empty command (fix-wfxt): it has no command for intent
        # detection to route, and predict_single_sentence raises on it.
        if not command or not command.strip():
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

        # Context type BEFORE execution (arch §12.0 delta 2). Gated on a span
        # having actually opened, for the same reason the attribute prep below
        # is: with tracing off this work must not run at all.
        context_before = (
            tracing.context_type(cls._active_workflow(chat_session))
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
                # that could not say which command it covered. fix-ajv.16 FW-3.
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={
                    "error_type": type(exc).__name__,
                    # A control signal is not a failure, so it records no outcome.
                    **({} if is_control_signal else {"success": False}),
                },
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

        context_after = None
        if span is not None:
            workflow = cls._active_workflow(chat_session)
            context_after = tracing.context_type(workflow)

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

    @staticmethod
    def _active_workflow(chat_session: 'fastworkflow.ChatSession'):
        """The workflow whose command context this dispatch acts on, or None.

        Duck-typed and never raising, like the rest of the tracing seam: this is
        called only to build capture attributes, and a host that cannot answer
        must degrade to an absent context type rather than fail the command.
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
        app_workflow = chat_session.get_active_workflow()
        # Assistant mode answers an exactly named planning command before any extraction: a
        # command that needs the agent is refused with its reasons. A mechanical one falls
        # through to the extraction hop below, and the post-extraction branch runs it. In agent
        # mode the name goes through the extraction hop and runs as a planning command.
        if app_workflow is not None and _assistant_round(chat_session):
            token = _command_name_token(command)
            if planning_name := _resolve_planning_command_name(
                app_workflow.folderpath, token
            ):
                # No agent model: the command runs to its first open step instead of being refused up front.
                refusal = (None if getattr(chat_session, "no_agent_mode", False) is True else
                           _assistant_planning_refusal(app_workflow.folderpath, planning_name, app_workflow))
                if refusal is not None:
                    return refusal

        command_output = cls.perform_action(
            chat_session.cme_workflow, 
            Action(
                command_name = "wildcard",
                command = command)
        )

        if command_output.command_handled:       
            # important to clear the current command mode from the workflow context. A handled output
            # that still holds the CME's question (the nothing-matched which-command question) keeps
            # the round: the next message is the reply to it, as for the other questions.
            if intent_loop.waiting_for(chat_session.cme_workflow.context) is None:
                _end_round(chat_session)
            return command_output
        elif not command_output.success:       
            return command_output

        command_name = command_output.command_response.artifacts["command_name"]
        input_obj = command_output.command_response.artifacts["cmd_parameters"]
        # Set by the CME walk when the command runs on an enclosing record (None otherwise).
        ran_on_name = command_output.command_response.artifacts.get("ran_on_context_name")
        ran_on_clause = command_output.command_response.artifacts.get("ran_on_context_clause")

        workflow = chat_session.get_active_workflow()
        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow._current_command_context_displayname

        # A command that writes is refused unrun while a part the user asked only to look up runs (bd fix-ftdl.4.3), and
        # inside a planning command's phase whose steps do not name it (fix-fixv, command_writes).
        refusal = (_look_up_only_refusal(chat_session, workflow.folderpath, command_name)
                   or command_writes.phase_write_refusal(chat_session, workflow.folderpath, command_name))
        if refusal is not None:
            _end_round(chat_session)
            # Marked so the agent loop neither logs it as an executed action nor lets the policy
            # remember it as run (workflow_agent._execute_workflow_query skips both, as for policy_blocked).
            refused = CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=refusal, success=False, artifacts={"write_refused": True}),
                command_name=command_name)
            refused.workflow_name = workflow_name
            refused.context = context
            return refused

        if planning_name := _resolve_planning_command_name(
            workflow.folderpath, command_name
        ):
            assistant = _assistant_round(chat_session)
            if assistant:
                refusal = (None if getattr(chat_session, "no_agent_mode", False) is True else
                           _assistant_planning_refusal(workflow.folderpath, planning_name, workflow))
                if refusal is not None:
                    _end_round(chat_session)
                    return refusal
            # Agent mode: the step's arguments were extracted above; the planning command
            # runs them as a nested frame. Its output is the step's observation. Assistant mode
            # runs a mechanical planning command the same way, with its steps gated as
            # agent-originated (ADR-EXP-025 section 5: a planning command's steps are the agent's
            # calls), and without suspension, since no agent is asking.
            try:
                origin = (approval_binding.origin_scope(approval_binding.Origin.AGENT_DIRECT)
                          if assistant else nullcontext())
                with origin:
                    planning_output = chat_session.run_planning_command(
                        planning_name, input_obj, command
                    )
            finally:
                if assistant:
                    _end_round(chat_session)
            planning_output.workflow_name = workflow.folderpath.split("/")[-1]
            planning_output.context = workflow._current_command_context_displayname
            planning_output.command_name = planning_name
            planning_output.command_parameters = input_obj or None
            return planning_output

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
        # Approval gate (ADR-EXP-025): the one choke point the agent's calls and planning-command
        # steps reach. Unannotated and user-originated calls pass straight through.
        held = approval_binding.gate(chat_session, workflow, command_name, input_obj, command)
        if held is not None:
            return held

        response_generation_object = response_generation_class()

        raw_user_message = command
        if "raw_user_message" in workflow.context:
            raw_user_message = workflow.context['raw_user_message']

        # The policy verifier's guard (fastworkflow.policy_runtime), consulted
        # here because only here does the command exist in canonical form: its
        # real name and validated parameters, whatever wording the agent used.
        # A block returns the rewrite as a failed output and nothing executes.
        # All policy logic lives with the guard; this is only the call point.
        if (rewrite := _policy_rewrite(chat_session, command_name, input_obj)) is not None:
            blocked = CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=rewrite,
                    success=False,
                    artifacts={"policy_blocked": True},
                )
            )
            blocked.workflow_name = workflow_name
            blocked.context = context
            blocked.command_name = command_name
            blocked.command_parameters = input_obj or None
            _end_round(chat_session)
            return blocked

        try:
            # command_parameters is the filled Input, or None for a command with no Input (3.14).
            command_parameters_class = command_routing_definition.get_command_class(
                command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
            command_output = call_response_generator(
                response_generation_object, workflow, raw_user_message,
                input_obj if command_parameters_class else None,
                command_name=command_name,
            )
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
                # The command's own code raised (not the framework): the "/" path reports it as the command's failed
                # output (``WorkflowExecutionContext._process_message``), not as a failed turn.
                _fw_raised_by_command=True,
            )
            raise


        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_name = command_name
        command_output.command_parameters = input_obj or None
        # A command the walk ran on an enclosing record is filed under that record.
        _file_under(command_output, (ran_on_name or "", ran_on_clause) if ran_on_clause is not None else None)

        if command_output.success:
            command_writes.record_phase_write(chat_session, workflow.folderpath, command_name)

        # important to clear the current command mode from the workflow context
        _end_round(chat_session)

        return command_output

    @classmethod
    def perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:
        # The command runs on the record of the scope that owns it, as a message's does (bd fix-ftdl.3); the current
        # position does not move. Its validators and its code both read that record as workflow.record.
        with running_command():
            return cls._perform_action(workflow, action)

    @classmethod
    def _perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        workflow._command_context_for_response_generation = _owner_record(workflow, action.command_name)
        ran_on = _ran_on(workflow, workflow._command_context_for_response_generation)

        # One id per dispatch through this method (arch §12.0 delta 1). It is
        # the outermost id on the direct-action, startup-action and MCP paths,
        # and a child id on the internal CME hop that invoke_command makes — in
        # which case call_scope files it under the enclosing command call.
        call_id = tracing.new_command_call_id()
        pending = _pending_before_cme_reply(workflow) if action.command_name == "wildcard" else None

        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow._current_command_context_displayname

        if action.command_name != "wildcard":
            if planning_name := _resolve_planning_command_name(
                workflow.folderpath, action.command_name
            ):
                return _planning_command_refusal_output(
                    planning_name,
                    workflow=workflow,
                )
        
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
            if action.command_name != "wildcard":
                held = approval_binding.gate(tracing.current_host(), workflow, action.command_name,
                                             None, action.command)
                if held is not None:
                    held.command_call_id = call_id
                    return held
            try:
                with tracing.call_scope(call_id, command_name=action.command_name):
                    command_output = call_response_generator(
                        response_generation_object, workflow, action.command, None,
                        command_name=action.command_name,
                    )
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
            
            if action.command_name == "wildcard":
                command_output = _wildcard_command_output(command_output, pending, workflow)

            # Validate that response_generation_object returns a CommandOutput, not a string
            if not isinstance(command_output, CommandOutput):
                raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")

            # Set the additional attributes
            command_output.workflow_name = workflow_name
            command_output.context = context
            command_output.command_name = action.command_name
            command_output.command_call_id = call_id
            _file_under(command_output, ran_on)
            return command_output

        # Always resolve the command's Signature class via create() so
        # the Input's model validators run on the direct-action
        # path the same way they do on the NLU path. Validate even when
        # action.parameters is empty/falsy — context preconditions still apply.
        # A required field the caller leaves out is asked for, as on the NLU path (fix-3rsq.plain-required).
        input_obj = parameters_with_unfilled(command_parameters_class, action.parameters or {})

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

        if action.command_name != "wildcard":
            held = approval_binding.gate(tracing.current_host(), workflow, action.command_name,
                                         input_obj, action.command)
            if held is not None:
                held.command_call_id = call_id
                return held

        try:
            with tracing.call_scope(call_id, command_name=action.command_name):
                command_output = call_response_generator(
                    response_generation_object, workflow, action.command, input_obj,
                    command_name=action.command_name,
                )
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
        
        if action.command_name == "wildcard":
            command_output = _wildcard_command_output(command_output, pending, workflow)

        # Validate that response_generation_object returns a CommandOutput, not a string
        if not isinstance(command_output, CommandOutput):
            raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")

        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_name = action.command_name
        command_output.command_parameters = input_obj or None
        command_output.command_call_id = call_id
        _file_under(command_output, ran_on)

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

            # Execute using existing perform_action method. An MCP tool call is agent-originated
            # (owner answer 3); the MCP caller cannot be suspended, so an ask returns its prompt.
            with approval_binding.origin_scope(approval_binding.Origin.AGENT_DIRECT):
                command_output = cls.perform_action(workflow, action)

            # Convert to MCP format
            return command_output.to_mcp_result()

        except Exception as e:
            # Return error in MCP format
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
