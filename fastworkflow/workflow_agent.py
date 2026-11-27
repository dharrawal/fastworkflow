"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""

import contextlib
import contextvars
import dataclasses
import difflib
import functools
import hashlib
import json
import os
import re
import threading
import time
import traceback
from typing import Any, Mapping, Optional
from datetime import datetime, timezone
from pathlib import Path

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from dspy.adapters.types.tool import Tool
from dspy.experimental import Choice, TypeSafe

import fastworkflow
from fastworkflow.observability import tracing
from fastworkflow import approval_binding
from fastworkflow import intent_loop
from fastworkflow import turn_progress
from fastworkflow.utils import subtask_pool
from fastworkflow import planning_commands
from fastworkflow.planning_commands import (
    PlanningCommandCommand, PlanningCommandFormatError, PlanningCommandInputError, typed_planning_call, refusal_question,
    refusal_names_an_input)
from fastworkflow.split_merge import (
    JoinedSubtask, Reduction, definite_descriptions, merge_subtasks, reduce_to_cap, request_subjects, subjects_in,
    unnamed_subjects, with_referent, without_repeated_clauses)
from fastworkflow.utils.logging import logger
from fastworkflow.policy.policy_runtime import policy_of
from fastworkflow.workflow_execution_context import CommandCancelledError
from fastworkflow.utils import dspy_utils
from fastworkflow.command_directory import (
    CommandDirectory, _command_kind_from_ast, _handle_command_field_source_errors, check_no_planning_commands_folder,
    get_cached_command_directory,
)
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_executor import AlreadyInContextError, CommandNotFoundError
from fastworkflow.context_navigation import (
    command_on_path, enters_current_context, lookup_scopes, render_already_in_context,
    scope_command_names, unavailable_command_message)
from fastworkflow.context_navigation import home_contexts as context_homes
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
from fastworkflow.utils.dspy_logger import watch_cut_off_replies
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.search import search_memory as search_observation
from fastworkflow.observation_offloading.labels import annotated_observation
from fastworkflow.observation_offloading.archive import command_run_record
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    current_execute_alias,
    note_command_run,
    note_folded_command,
    observability_db_path,
    scope_for_host,
)
from fastworkflow.context_identity import context_clause_for, context_identity

_FINAL_ANSWER_DESC = "The answer from the evidence gathered: one line for each part of the request, giving what was found with its names and uids, or saying plainly that it was not found or not reached. Never claim work the trajectory does not show, and do not say the task is complete."

class WorkflowAgentSignature(dspy.Signature):
    """
    Carefully review the user request, then execute the next steps using available tools for building the final answer.
    Address every part of the request before finishing.
    A command output offloaded to memory is not lost: every observation of this turn is normally restored in full
    when the final answer is written. If the answer's evidence limit is reached, the oldest observations are not
    restored and the answer names them. If you need a value from an offloaded observation to choose your next
    step, run its command again; never re-run commands just to collect rows for the final answer.
    """
    user_query = dspy.InputField(desc="The natural language user query.")
    final_answer = dspy.OutputField(desc=_FINAL_ANSWER_DESC)

def run_progress_check(chat_session_obj, trigger: str, *, done_only: bool = False) -> str:
    """What's done and what remains, from the agent's current view of its trajectory.

    done_only: return just the "what's done" summary, without the remaining steps.
    """
    return build_query_with_next_steps(
        "",
        chat_session_obj, with_agent_inputs_and_trajectory=True,
        planner_lm=getattr(chat_session_obj, '_current_planner_lm', None),
        trace_trigger=trigger,
        done_only=done_only,
    )


def _append_action_record(chat_session_obj, record: dict) -> None:
    """Append to session-scoped action log (WEC or ChatSession delegating to core).

    Duck-typed like _append_turn_output; no-ops gracefully if neither exposes the
    action log. The cwd action.jsonl fallback has been retired — the
    in-process log is the only sink, and the observability DB is the post-mortem
    record.
    """
    if hasattr(chat_session_obj, "append_action_log"):
        chat_session_obj.append_action_log(record)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_action_log"):
        core.append_action_log(record)


def _held_context(chat_session_obj, command_output, ran_in_clause, ran_in_name):
    """The (clause, name) a command's held run is recorded under.

    A planning command's child leaves the workflow where its last step ended,
    so its run names the context the workflow is in once the call returns;
    any other command ran, and is held, in the context it was called from.
    Never raises: a label hook must not stop the command.
    """
    # A run the parent walk executed on an enclosing record is filed under that record.
    ran_on_clause = getattr(command_output, "ran_in_context_clause", None)
    if ran_on_clause is not None:
        return ran_on_clause, getattr(command_output, "ran_in_context_name", None) or ""
    active = chat_session_obj.get_active_workflow()
    try:
        if active is None or planning_commands.planning_command(
                active.folderpath, command_output.command_name) is None:
            return ran_in_clause, ran_in_name
        return context_clause_for(active), context_identity(active)[0]
    except Exception:  # noqa: BLE001 - a label hook must not stop the command
        return ran_in_clause, ran_in_name


def _hold_command_run(chat_session_obj, command_output, context_clause,
                      context_name=None, alias=None) -> None:
    """Hold this command's run for the in-flight execute step's archive row.

    Its name, success, context, parameters and ``Signature.Output``
    (``CommandResponse.output``) are stored beside the step's text when the
    step is archived; the observation the agent reads is not touched. Outside
    an agent execute step there is no row to store it in, and nothing is held.
    *alias* names the step the run is held under; None reads the in-flight step.
    Never raises: this record must not be able to fail a command.
    """
    try:
        name = command_output.command_name
        if not name:
            return
        if alias is None:
            alias = current_execute_alias(getattr(chat_session_obj, "workflow_tool_agent", None))
        if alias is None:
            return
        note_command_run(scope_for_host(chat_session_obj), alias, command_run_record(
            command_name=name,
            success=bool(command_output.success),
            context_clause=context_clause,
            context_name=context_name,
            parameters=command_output.command_parameters,
            output=command_output.command_response.output,
        ))
    except Exception:  # noqa: BLE001
        logger.debug("could not hold the command run for its archive row", exc_info=True)


def command_run_response(command_output) -> str:
    """The response text a command run gives the agent: its response, or a line saying it produced none.

    Shared by the step that ran the command (_execute_workflow_query) and the step a
    nested run finishes for after a suspension (WorkflowExecutionContext).
    """
    if command_output.command_response.response:
        return command_output.command_response.response
    return "Command executed successfully but produced no output."


def _parameters_dict(params) -> dict | None:
    """A command's parameters as a dict: a typed Input is dumped, a dict is kept."""
    if params is None:
        return None
    if hasattr(params, "model_dump"):
        return params.model_dump()
    if isinstance(params, dict):
        return params
    return None


def action_record(command: str, command_output, response_text: str) -> dict:
    """The action-log record of one command run: the text the agent sent, the command it
    ran as, its parameters and the response it gave.

    Written by the step that ran the command, and by the step a nested run completes after
    a suspension, so the two records read alike.
    """
    return {
        "command": command,
        "command_name": command_output.command_name,
        "parameters": _parameters_dict(command_output.command_parameters),
        "response": response_text,
    }


def run_observation(alias: str | None, ran_in_clause: str, response: str, *,
                    folded_note: str = "", context_changed: bool,
                    now_clause: str | None) -> str:
    """The observation a command run gives the agent: its response, the note on arguments
    folded into its command, and the handle line naming the step it ran in (with where it
    left the agent, when the run moved it).

    Shared by the step that ran the command and by the step a nested run completes after a
    suspension, so both observations read alike.
    """
    text = response + folded_note
    if alias is None:
        return text
    return annotated_observation(alias, ran_in_clause, text, context_changed=context_changed,
                                 now_in=now_clause if context_changed else None)


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

    Inside a planning command's phase, the commands that write and that no step of the phase names are
    left out (fix-fixv, ``command_writes.unplanned_write_commands``).
    """
    workflow = chat_session_obj.get_active_workflow()
    return _executor_commands_text(workflow, _unplanned_writes(chat_session_obj, workflow))


def _unplanned_writes(host, workflow) -> frozenset[str]:
    """The write commands the running phase's steps do not name (none outside a phase)."""
    from fastworkflow import command_writes

    if workflow is None:
        return frozenset()
    return command_writes.unplanned_write_commands(host, workflow.folderpath)


def _executor_commands_text(current_workflow, exclude: frozenset[str] = frozenset()) -> str:
    """The active context's commands in full (minus *exclude*), then one line per other context."""
    commands = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow._current_command_context_name,
        exclude=exclude,
    )
    other_contexts = CommandMetadataAPI.get_other_contexts_text(
        subject_workflow_path=current_workflow.folderpath,
        active_context_name=current_workflow._current_command_context_name,
        navigation_workflow=current_workflow,
        exclude=exclude,
    )
    return f"{commands}\n\n{other_contexts}" if other_contexts else commands

def _refresh_agent_available_commands(host, exclude: frozenset[str] = frozenset()) -> None:
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

    inputs["available_commands"] = _executor_commands_text(
        current_workflow, exclude | _unplanned_writes(host, current_workflow))


_REPEAT_REASON = "it just returned the same result again"


class _RepeatGuard:
    """One executor run's repeat memory; lives in the agent's ``run_state``.

    ``pending`` is set by a repeat in the step now running and becomes
    ``blocked_command`` (name, reason, command text) at the end of that step, so it
    governs exactly the next step. Only that same command text is blocked: the
    command with other arguments (its next page) still runs.
    """

    def __init__(self) -> None:
        self.last_results: dict[tuple[str, str], str] = {}
        self.pending: tuple[str, str, str] | None = None
        self.blocked_command: tuple[str, str, str] | None = None


def _repeat_guard(chat_session_obj) -> _RepeatGuard | None:
    run_state = getattr(getattr(chat_session_obj, "workflow_tool_agent", None), "run_state", None)
    if run_state is None:
        return None
    return run_state.setdefault("repeat", _RepeatGuard())


def _set_blocked_command(chat_session_obj, guard: _RepeatGuard, block: tuple[str, str, str] | None) -> None:
    guard.blocked_command = block


def _end_repeat_step(chat_session_obj) -> None:
    """End-of-step hook: a repeat in the step that just ran blocks the NEXT step only."""
    guard = _repeat_guard(chat_session_obj)
    if guard is None:
        return
    upcoming, guard.pending = guard.pending, None
    if upcoming != guard.blocked_command:
        _set_blocked_command(chat_session_obj, guard, upcoming)


def _clear_blocked_command(chat_session_obj) -> None:
    """Drop the block now: ask_user suspends the run, so the end-of-step hook would not run."""
    guard = _repeat_guard(chat_session_obj)
    if guard is not None and guard.blocked_command is not None:
        _set_blocked_command(chat_session_obj, guard, None)


def _display_context_name(context: str) -> str:
    return "global" if context == "*" else context


def _command_name_matches(token: str, name: str) -> bool:
    """Whether an agent's command token names *name* (qualified or short form)."""
    return token.lower() == (name if "/" in token else name.split("/")[-1]).lower()


def core_command_names() -> frozenset[str]:
    """The framework's core command names, as the routing registry lists them (qualified
    by context): the IntentDetection and ErrorCorrection contexts of the internal
    command_metadata_extraction workflow.

    Read from the routing registry on every call rather than memoised: the registry
    can be cleared and rebuilt, and a stale answer would outlive it.
    """
    cme = fastworkflow.RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    return frozenset(name
                     for context in ("IntentDetection", "ErrorCorrection")
                     for name in cme.get_command_names(context))


#: Set while the framework itself runs a planning command (WorkflowExecutionContext.run_framework_planning_command):
#: the framework's plan steps may then name a framework execution command (review_proposal) by exact name. The
#: agent's own calls never run in this scope, so they are still refused.
_framework_call_var: contextvars.ContextVar[bool] = contextvars.ContextVar("fw_framework_call", default=False)


@contextlib.contextmanager
def framework_call_scope():
    token = _framework_call_var.set(True)
    try:
        yield
    finally:
        _framework_call_var.reset(token)


def _framework_execution_names(app) -> list[str]:
    return [name for name, meta in app.command_directory.map_command_2_metadata.items()
            if meta.framework and not meta.is_planning]


def _framework_call_name(command: str, workflow) -> str | None:
    """The framework execution command *command* names, when the framework itself is calling it; else None."""
    if not _framework_call_var.get() or workflow is None:
        return None
    token = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    app = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    hits = [name for name in _framework_execution_names(app) if name == token or name.split("/")[-1] == token]
    return hits[0] if len(hits) == 1 else None


_FRAMEWORK_PARAMETER = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


def _framework_action_parameters(command: str) -> dict:
    """``name=value`` (value quoted or not) pairs of a framework step's command text."""
    return {match.group(1): match.group(2) if match.group(2) is not None else match.group(3)
            for match in _FRAMEWORK_PARAMETER.finditer(command)}


def _explicit_agent_command(command: str, workflow) -> str:
    """Resolve the agent tool's command token against its current command surface.

    This tool takes command names, not natural-language intents. Never let a
    command name that is unavailable here fall through to fuzzy/cache/classifier
    substitution: name the contexts that do have it, and the commands that
    reach one of them from here, instead. A token that names no command in any
    context of the workflow is not a command name, so it goes to the assistant
    NLU unchanged. Parameter extraction still runs as before.
    """
    parts = command.strip().split(maxsplit=1)
    token = parts[0].lstrip("/") if parts else ""
    app = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    current_context = workflow._current_command_context_name
    # A planning command is available where it is runnable, by the same rule as an
    # execution command: a name outside this set is refused with the same message.
    available = (set(app.get_command_names(current_context))
                 | set(app.get_planning_command_names(current_context))
                 | core_command_names())
    if _framework_call_var.get():
        framework_hits = [name for name in _framework_execution_names(app)
                          if _command_name_matches(token, name)]
        if len(framework_hits) == 1:
            return framework_hits[0] + (" " + parts[1] if len(parts) > 1 else "")
    matches = [name for name in available if _command_name_matches(token, name)]
    if not matches and (on_path := command_on_path(workflow, token)) is not None:
        # 3.9, "Calls by exact name": a command of an enclosing record or of the global
        # scope runs here, bound to that record (the CME walk binds it). Nearest scope wins.
        scope, hits = on_path
        if len(hits) != 1:
            raise CommandNotFoundError(
                f"Command {token!r} is ambiguous in context "
                f"{_display_context_name(scope.context_name)!r}; it matches "
                f"{', '.join(repr(m) for m in sorted(hits))}. Use the qualified name.")
        name = hits[0]
        short = name.split("/")[-1]
        # The same collapse rule as the current scope, within the scope the name was found in.
        scope_names = scope_command_names(workflow.folderpath, scope.context_name)
        if sum(other.split("/")[-1].lower() == short.lower() for other in scope_names) != 1:
            raise CommandNotFoundError(f"Ambiguous command name {short!r} in context "
                                       f"{_display_context_name(scope.context_name)!r}")
        # A qualified token is kept qualified: a nearer scope may hold a command with the
        # same short name, and the CME walk must find the one the token names.
        named = name if "/" in token else short
        return named + (" " + parts[1] if len(parts) > 1 else "")
    if not matches:
        home_contexts = context_homes(workflow, token)
        if not home_contexts:
            return command
        if enters_current_context(workflow, token, current_context):
            raise AlreadyInContextError(
                render_already_in_context(token, current_context, workflow))
        raise CommandNotFoundError(unavailable_command_message(
            workflow, token, current_context, home_contexts))
    if len(matches) != 1:
        raise CommandNotFoundError(
            f"Command {token!r} is ambiguous in context "
            f"{_display_context_name(current_context)!r}; it matches "
            f"{', '.join(repr(m) for m in sorted(matches))}. Use the qualified name.")
    # The CME exact-prefix matcher consumes short names. Do not permit a
    # qualified token to collapse onto a different command with the same tail.
    short = matches[0].split("/")[-1]
    if sum(name.split("/")[-1].lower() == short.lower() for name in available) != 1:
        raise CommandNotFoundError(f"Ambiguous command name {short!r} in current context")
    return short + (" " + parts[1] if len(parts) > 1 else "")


def _commands_in(command: str, names: set[str]) -> list[str]:
    """The lines of *command* that start with a known command name (short, lowercased)."""
    found = []
    for line in (raw.strip() for raw in command.splitlines()):
        token = re.split(r"[\s<]", line, maxsplit=1)[0].lstrip("/")
        if line and token.split("/")[-1].lower() in names:
            found.append(line)
    return found


# Keys a model nests a command's parameters under, and keys it puts the command
# name under when ``command`` itself is missing (fix-5jss).
_NESTED_ARG_KEYS = ("args", "arguments", "params", "parameters", "kwargs")
_COMMAND_NAME_KEYS = ("command_name", "cmd", "action", "function", "tool")
_COMMAND_NAME_TOKEN = re.compile(r"[A-Za-z_][\w./-]*")
_COMMAND_TOOL_NAME = "execute_workflow_query"
_XML_TAG = re.compile(r"</?[A-Za-z_][\w.-]*>")


def _arg_text(value) -> str:
    """A tool argument as command text: strings as they are, everything else JSON."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _key_value_text(key: str, value: str) -> str:
    """``key=value``, quoted only where a parser would otherwise cut the value.

    The key=value parser strips a trailing ``,``/``;`` and a trailing ``(note)``.
    A code-run step's chain ends at the first ``(`` or ``;`` outside quotes
    (``finish_check.step_chain``), so a value that contains either is quoted, not
    only one that ends with them. A `` word=`` inside a value ends it whatever
    the quoting, so ``=`` (or a comma inside the value) is no reason to quote.
    """
    if "(" in value or ";" in value or value[-1] in ",":
        quote = '"' if '"' not in value else "'"
        if quote not in value:
            return f"{key}={quote}{value}{quote}"
    return f"{key}={value}"


def fold_tool_kwargs(command, extra: dict,
                     fields: frozenset[str] | None = None) -> tuple[str, list[str]]:
    """Fold keyword arguments the model passed beside ``command`` into the command text.

    The command tool takes one ``command`` string, but small models often pass a
    command's parameters as tool arguments of their own
    (``{"command": "find_item", "name": "Ann Lee"}``), and the tool's arg check
    refused every such call with a traceback the model then repeated (fix-5jss).
    Each extra argument is appended as `` key=value`` (as ``<key>value</key>``
    when the command text already uses XML tags), so the usual parsers bind it
    or the usual refusals say why not:

    - a dict under ``args``/``arguments``/``params``/``parameters``/``kwargs`` is
      unpacked into its keys; any other value there is a bare positional value;
    - with no command text, the name may come from ``command_name``, ``cmd``,
      ``action``, ``function`` or ``tool`` when that value is one token;
    - lists and dicts become JSON text; ``None`` and empty strings carry no value;
    - given the command's parameter *fields*, a lone argument whose key is none
      of them is sent as a bare value (``uid=ab12`` for a ``item_uid`` field
      becomes ``ab12``), for the positional parser and its guards.

    Returns the folded command, and one note per argument that carried no value
    so that nothing is dropped unrecorded.
    """
    text = command.strip() if isinstance(command, str) else _arg_text(command) if command else ""
    pairs: list[tuple[str | None, object]] = []
    for key, value in extra.items():
        if key in _NESTED_ARG_KEYS and isinstance(value, dict):
            pairs.extend(value.items())
        elif key in _NESTED_ARG_KEYS:
            pairs.append((None, value))
        else:
            pairs.append((key, value))
    if not text:
        for i, (key, value) in enumerate(pairs):
            if (key in _COMMAND_NAME_KEYS and isinstance(value, str)
                    and _COMMAND_NAME_TOKEN.fullmatch(value.strip())):
                text = value.strip()
                del pairs[i]
                break
    if fields is not None and len(pairs) == 1 and pairs[0][0] not in fields:
        pairs = [(None, pairs[0][1])]
    xml = bool(_XML_TAG.search(text))
    parts = [text] if text else []
    empty: list[str] = []
    for key, value in pairs:
        value_text = "" if value is None else _arg_text(value)
        if not value_text:
            empty.append(f"{key or 'args'} had no value")
        elif key is None:
            parts.append(value_text)
        elif xml:
            parts.append(f"<{key}>{value_text}</{key}>")
        else:
            parts.append(_key_value_text(key, value_text))
    return " ".join(parts), empty


def _parameter_fields(command: str, workflow) -> frozenset[str] | None:
    """The parameter field names of the command *command* starts with, if it is one
    command of the current context; None when that is not certain."""
    token = (command.split() or [""])[0].lstrip("/")
    if workflow is None or not _COMMAND_NAME_TOKEN.fullmatch(token):
        return None
    try:
        app = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        context = workflow._current_command_context_name
        names = [name for name in (*app.get_command_names(context),
                                   *app.get_planning_command_names(context))
                 if _command_name_matches(token, name)]
        if len(names) != 1:
            return None
        parameters = app.get_command_class(names[0], fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:  # noqa: BLE001 - a lookup aid, never a reason to fail the call
        return None
    return frozenset(parameters.model_fields) if parameters is not None else frozenset()


def _execute_workflow_query(command: str, chat_session_obj: fastworkflow.ChatSession,
                            *, internal: bool = False) -> str:
    """
    Executes the command and returns either a response, or a clarification request.
    Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
    Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
    Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state

    ``internal`` marks the framework's own calls (the abort that ends an error
    state): the repeat guard neither refuses nor records them, since a refused
    abort leaves the session stuck in a stale parameter round (fix-hia1.13).
    """
    # Refuse a bundle of several commands before anything runs: intent detection
    # would execute only the first one and silently drop the rest.
    active = chat_session_obj.get_active_workflow()
    if active is not None:
        app = fastworkflow.RoutingRegistry.get_definition(active.folderpath)
        known = {name.split("/")[-1].lower()
                 for names in app.contexts.values() for name in names}
        known |= {name.split("/")[-1].lower() for name in core_command_names()}
        known.discard("wildcard")
        commands = _commands_in(command, known)
        if len(commands) >= 2:
            return (f"You sent {len(commands)} commands in one call; nothing ran. "
                    f"Send one command per call: {commands[0]} first.")

    # A command blocked for this step (it just repeated its own result) is refused unrun.
    guard = _repeat_guard(chat_session_obj)
    token = (command.split() or [""])[0].lstrip("/").split("/")[-1].lower()
    if internal:
        guard = None
    if guard is not None and guard.blocked_command and " ".join(command.split()) == guard.blocked_command[2]:
        return (f"{guard.blocked_command[0]} with these arguments is unavailable for this step: "
                f"{guard.blocked_command[1]}. Use what it returned, or do something else.")
    ran_in = active._current_command_context_name if active is not None else None
    try:
        ran_in_clause = context_clause_for(active) if active is not None else ""
        ran_in_name = context_identity(active)[0]
    except Exception:  # noqa: BLE001 - a label hook must not stop the command
        ran_in_clause = ran_in_name = None

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

    # The dispatch is recorded on the step that made it (bd fix-qjne.34: the step's fw.step span, ``tracing.note``;
    # nothing when no step is open) — deliberately OUTSIDE the trace-queue guard: the sink is reached via the
    # WEC/ChatSession core, not the transport-queue contract, so queue-less embedders still trace [R28]. §12.1.1's
    # shared capture (arch §12.0 deltas 1-2/4) is `tracing`'s projection; the context type is re-read after execution,
    # so a context the command MOVED is still recorded as a transition.
    tracing.note(tracing.SPAN_STEP, raw_command=command)
    workflow = tracing.active_workflow(chat_session_obj)
    context_before = tracing.context_type(workflow)

    # Directly invoke the command without going through queues
    # This allows the agent to synchronously call workflow tools
    from fastworkflow.command_executor import CommandExecutor, _annotation
    started = datetime.now(timezone.utc)
    try:
        resolved_command = _explicit_agent_command(command, chat_session_obj.get_active_workflow())
        framework_name = _framework_call_name(resolved_command, chat_session_obj.get_active_workflow())
        if framework_name is not None:
            # A framework execution command (review_proposal) is not routed, so the intent path cannot name it:
            # the framework's own step calls it by exact name, through the executor.
            command_output = CommandExecutor.perform_action(
                chat_session_obj.get_active_workflow(),
                fastworkflow.Action(command_name=framework_name, command=resolved_command,
                                    parameters=_framework_action_parameters(resolved_command)))
        else:
            command_output = CommandExecutor.invoke_command(chat_session_obj, resolved_command)
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
            raise
        if not isinstance(e, Exception):
            # Any other BaseException (KeyboardInterrupt, SystemExit): do not build a failure CommandOutput
            # for it — that is reserved for real command failures, and this
            # frame must not dress an interpreter-level signal as one.
            raise
        if isinstance(e, CommandNotFoundError):
            # Routing guidance, not a command failure: no command ran, and the
            # caller hands the agent the message as its observation. Recording
            # it as a failed CommandOutput put a traceback artifact in the
            # turn's answer and marked the turn unsuccessful.
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
            # (tests/test_turn_result_capture.py does), hence the annotation
            # defaults rather than a declared attribute. fix-ajv.16 FW-2.
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
        tracing.note(tracing.SPAN_STEP, success=False)
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
    params_dict = _parameters_dict(params)

    # Extract response text
    response_text = command_run_response(command_output)

    # Same command text, same context, same response as this run's previous call: block it next step.
    if guard is not None:
        key = (ran_in, " ".join(command.split()))
        if guard.last_results.get(key) == response_text:
            guard.pending = ((name or token).split("/")[-1], _REPEAT_REASON, key[1])
        guard.last_results[key] = response_text

    tracing.note(tracing.SPAN_STEP, response_text=response_text, success=bool(command_output.success),
                 command_call_id=getattr(command_output, "command_call_id", None), context_before=context_before,
                 context_after=tracing.context_type(workflow))

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
    # A command the policy blocked, or a write refused inside a read-only planning command's phase
    # (command_writes), never ran, so it is not an executed action; what ran is also what the policy
    # remembers (fastworkflow.policy.policy_runtime).
    artifacts = getattr(command_output.command_response, "artifacts", None) or {}
    if not (artifacts.get("policy_blocked") or artifacts.get("write_refused")):
        _append_action_record(chat_session_obj, action_record(command, command_output, response_text))
        if not internal and (policy := policy_of(chat_session_obj.workflow_tool_agent)):
            policy.record_command(
                name or (command.split() or [""])[0],
                params_dict,
                response_text,
                bool(command_output.success),
            )
    held_clause, held_name = _held_context(chat_session_obj, command_output, ran_in_clause, ran_in_name)
    _hold_command_run(chat_session_obj, command_output, held_clause, held_name)
    if command_output.ran_in_context_clause is not None:
        # The observation the agent reads names the record the command ran on (read back
        # by execute_workflow_query, which holds the step's run state).
        run_state = getattr(getattr(chat_session_obj, "workflow_tool_agent", None), "run_state", None)
        if isinstance(run_state, dict):
            run_state["ran_on_clause"] = command_output.ran_in_context_clause

    # Check workflow context to determine if we're in an error state that needs specialized handling
    cme_workflow = chat_session_obj.cme_workflow
    waiting_for = intent_loop.waiting_for(cme_workflow.context)

    # Intent ambiguity / misunderstanding: the NLU's reply already lists the
    # candidate commands (ambiguity) or this context's commands (misunderstanding),
    # so the agent chooses from it. Abort first so its next command is routed
    # fresh rather than read as an answer to the clarification prompt.
    if waiting_for == intent_loop.WAITING_INTENT:
        abort_confirmation = _execute_workflow_query(
            'abort', chat_session_obj=chat_session_obj, internal=True)
        return f'{response_text}\n{abort_confirmation}'
    # Handle parameter extraction errors with abort
    if waiting_for == intent_loop.WAITING_PARAMETERS:
        abort_confirmation = _execute_workflow_query(
            'abort', chat_session_obj=chat_session_obj, internal=True)
        # Thread the active planner LM so replanning uses the same planner LM
        # as the current turn (critical for distillation: otherwise
        # replans silently fall back to LLM_PLANNER instead of the teacher/student LM).
        planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
        return build_query_with_next_steps(
            f'{response_text}\n{abort_confirmation}',
            chat_session_obj, with_agent_inputs_and_trajectory=True,
            planner_lm=planner_lm,
            trace_trigger="parameter_extraction_error",
        )

    # Clean up the context flag after command execution
    workflow = chat_session_obj.get_active_workflow()
    if "is_user_command" in workflow.context:
        del workflow.context["is_user_command"]

    return response_text


def _apply_reply_intent(agent, intent: str) -> None:
    """Take ask_user away after a reply that stops the asking; leave only finish after an abort.

    "wait" and "answer" change nothing: a hesitation keeps ask_user for the question asked again.
    """
    if intent == "stop_asking":
        agent.disabled_tools = agent.disabled_tools | {"ask_user"}
    elif intent == "abort":
        agent.disabled_tools = set(agent.tools) - {"finish"}


def _refusal_may_ask_attended(chat_session_obj) -> bool:
    """Whether a refusal may be put to the user as a question that suspends the turn:
    the turn can ask (``waiting.can_ask``: attended). How often it asks is the
    parameter loop's asking rule (``parameter_clarification``), not decided here.

    The callers add their own check that nothing ran yet in the turn (a re-run would repeat its
    side effects): a planning command call's refusal checks the whole turn, a fast-path call its own outputs.
    """
    from fastworkflow import waiting

    return waiting.can_ask(chat_session_obj)


def _planning_command_refusal_may_ask(chat_session_obj) -> bool:
    """Whether a refused planning command call may be put to the user, suspending the turn.

    Only when the turn can ask (attended), no
    run has happened yet in the turn (a re-run would repeat its side effects).
    Every other case keeps the text planner's fallback.
    """
    return (
        _refusal_may_ask_attended(chat_session_obj)
        and not getattr(chat_session_obj, "_turn_outputs", True)
    )


def planning_command_refusal_to_ask(chat_session_obj, command: PlanningCommandCommand,
                                    refusal: PlanningCommandInputError, request_text: str = "") -> str | None:
    """The question a refused planning command call puts to the user, or None when the text planner
    plans the request instead.

    Asked only when the turn may ask (``_planning_command_refusal_may_ask``) and the refusal names one
    of the command's inputs (``refusal_names_an_input``): a missing or guessed value is the user's to
    give. A refusal that names no input says the command does not fit the request, which the user
    cannot fix by answering, so it is not put to them.

    The execution context opens the parameter clarification loop for it (``_open_parameter_loop``: the one
    place that decides whether to ask and keeps the question's state, for every caller), with *request_text*
    as the text the call's values were read from. A session object that has no loop gets the same decision
    and the question's text only.
    """
    open_loop = getattr(chat_session_obj, "_open_parameter_loop", None)
    if open_loop is not None:
        return open_loop(command, refusal, request_text=request_text)
    if not _planning_command_refusal_may_ask(chat_session_obj):
        return None
    if not refusal_names_an_input(command, refusal):
        return None
    return refusal_question(command, refusal)


def _post_ask_user_response(
    clarification_request: str,
    user_response: str,
    chat_session_obj,
) -> str:
    """
    Steps 2-4 after the user answers an ask_user clarification (Topology-A parity).

    Appends the dialog to the session action log, sets raw_user_message, and replans.
    On a "wait" reply (the user is checking, not answering) it does not replan: it returns
    an instruction to ask the same question again, which the ReAct loop reads as the observation.
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
    if policy := policy_of(chat_session_obj.workflow_tool_agent):
        policy.observe_reply(clarification_request, user_response)

    workflow = chat_session_obj.get_active_workflow()
    if workflow:
        workflow.context["raw_user_message"] = user_response

    intent = classify_user_reply(user_response, clarification_request)
    _apply_reply_intent(chat_session_obj.workflow_tool_agent, intent)
    if intent == "wait":
        # A hesitation is not an answer: no replan, and the ReAct loop asks the same question again.
        return (
            "The user asked for a moment and has not answered yet. Reply briefly that you will wait, "
            f"and ask the same question again with ask_user: {clarification_request}"
        )
    planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
    return build_query_with_next_steps(
        user_response,
        chat_session_obj,
        with_agent_inputs_and_trajectory=True,
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
    user_query = user_queue.get()

    return _post_ask_user_response(
        clarification_request, user_query, chat_session_obj
    )

def initialize_workflow_tool_agent(chat_session: fastworkflow.ChatSession, max_iters: int = 25,
                                   on_step_complete=None):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.

    Args:
        chat_session: fastworkflow.ChatSession instance
        max_iters: Maximum iterations for the ReAct agent
        on_step_complete: Optional callback(step_idx, trajectory) -> bool for
            step-by-step interception (distillation). Return False to stop early.

    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    chat_session_obj = chat_session
    if not chat_session_obj:
        raise ValueError("chat session cannot be null")

    AgentSignature = WorkflowAgentSignature

    # **_ignored on the zero-argument tools: a small model often passes one
    # anyway, and that should not cost it a traceback for an observation.
    # Wrapped below with an empty args schema, so the model is shown none.
    def what_can_i_do(**_ignored) -> str:
        """
        Returns a list of available commands, including their names and parameters
        """
        return _what_can_i_do(chat_session_obj=chat_session_obj)

    def run_execute_workflow_query(command: str) -> str:
        # The agent's own call (and the planning-command steps it runs): agent-originated for the
        # approval gate (ADR-EXP-025 section 5). Suspendable, so an ask suspends this tool call.
        with approval_binding.origin_scope(approval_binding.Origin.AGENT):
            return _run_execute_workflow_query_unscoped(command)

    def _run_execute_workflow_query_unscoped(command: str) -> str:
        # Check if this command originated from user input (iteration_counter == 0)
        # Set flag in workflow context so the Input's model validators can access it
        is_user_command = chat_session_obj.workflow_tool_agent.iteration_counter <= 0 
        run_state = getattr(chat_session_obj.workflow_tool_agent, "run_state", None)
        by_code = isinstance(run_state, dict) and bool(run_state.get("by_code"))
        workflow = chat_session_obj.get_active_workflow()
        if workflow:
            workflow.context["is_user_command"] = is_user_command
            workflow.context["by_code"] = by_code

        try:
            # Retry logic for workflow execution
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    return _execute_workflow_query(command, chat_session_obj=chat_session_obj)
                except AlreadyInContextError as e:
                    return reenter_from_parent(command, e)
                except CommandNotFoundError as e:
                    # Deterministic and recoverable: the message says where to go.
                    return str(e)
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        message = f"Terminate immediately! Exception processing {command}: {str(e)}"
                        logger.critical(message)                    
                        return message
                    # Continue to next attempt
                    logger.warning(f"Attempt {attempt + 1} failed for command '{command}': {str(e)}")
        finally:
            active = chat_session_obj.get_active_workflow()
            for wf in (workflow, active):
                if wf is not None and "by_code" in wf.context:
                    del wf.context["by_code"]
            if isinstance(run_state, dict):
                run_state.pop("by_code", None)

    def reenter_from_parent(command: str, refusal: AlreadyInContextError) -> str:
        """Open another instance of the context the agent is in: go_up, then the command.

        A weak model told "go_up first" retries the same command in place many
        times over. Re-opening the SAME instance is refused as before, since
        going up and back in would only repeat it.
        """
        workflow = chat_session_obj.get_active_workflow()
        name, _, instance = context_clause_for(workflow).partition(" ")
        uid = instance.split(" ")[0]
        if uid and uid in command:
            return str(refusal)
        _execute_workflow_query("go_up", chat_session_obj=chat_session_obj)
        try:
            return (f"(Left the current {name} with go_up first.)\n"
                    + _execute_workflow_query(command, chat_session_obj=chat_session_obj))
        except CommandNotFoundError as again:
            return str(again)

    # ``command`` defaults and ``**extra`` collects anything else, so a call with
    # parameters as tool arguments is folded instead of refused (fix-5jss). The
    # Tool below still shows the model the one ``command`` argument.
    def execute_workflow_query(command: str = "", **extra) -> str:
        """
        Takes just a single argument called 'command'.
        Executes the command and returns either a response, or a clarification request.
        Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
        Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
        Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
        """
        folded_note = ""
        if extra or not isinstance(command, str):
            name = command if isinstance(command, str) else ""
            for key in _COMMAND_NAME_KEYS:
                name = name or (extra[key] if isinstance(extra.get(key), str) else "")
            command, empty = fold_tool_kwargs(
                command, extra, _parameter_fields(name, chat_session_obj.get_active_workflow()))
            # Read by the ReAct loop into the step span's `folded_command`.
            run_state = getattr(chat_session_obj.workflow_tool_agent, "run_state", None)
            if isinstance(run_state, dict):
                run_state["folded_command"] = command
            # Parameters with no command name: refused here, since intent
            # detection would guess a command from ``key=value`` text.
            if not _COMMAND_NAME_TOKEN.fullmatch((command.split() or [""])[0].lstrip("/")):
                return (f"No command name was given, so nothing ran. Put the command name "
                        f"first in the command string, then its parameters: "
                        f"<command_name> {command}".rstrip())
            # The step's archive row keeps this text as its command: the finish
            # check reads what ran, not the bare ``command`` the model sent.
            folded_alias = current_execute_alias(chat_session_obj.workflow_tool_agent)
            if folded_alias is not None:
                note_folded_command(scope_for_host(chat_session_obj), folded_alias, command)
            folded_note =(f"\n(Ran as: {command} -- parameters go inside the command string."
                           + (f" Not used: {'; '.join(empty)}." if empty else "") + ")")
        # A call that names no workflow command is refused before dispatch (fix-wfxt). Blank text
        # has no command for intent detection to route, and the tool's own name is not a command.
        named = (command.split() or [""])[0].lstrip("/")
        if not named:
            return ("No command was given, so nothing ran. Name a workflow command first, then its "
                    "parameters after it, e.g. <command_name> <param>value</param>.")
        if named.lower() == _COMMAND_TOOL_NAME:
            return (f"{_COMMAND_TOOL_NAME} is this tool, not a workflow command, so nothing ran. "
                    "Name a workflow command first, then its parameters after it.")
        # The context this command RUNS IN, taken before dispatch: a command that
        # moves the context is evidence about the context it ran in.
        alias = current_execute_alias(chat_session_obj.workflow_tool_agent)
        workflow = chat_session_obj.get_active_workflow()
        clause = context_clause_for(workflow)
        context_before = getattr(workflow, "_current_command_context", None)
        # What a nested run this command starts needs to finish this step after a suspension:
        # the command as the action log holds it, and the note on its folded arguments.
        run_state = getattr(chat_session_obj.workflow_tool_agent, "run_state", None)
        if isinstance(run_state, dict):
            run_state["execute_call"] = {"command": command, "folded_note": folded_note}
            run_state.pop("ran_on_clause", None)
        try:
            response = run_execute_workflow_query(command)
        finally:
            if isinstance(run_state, dict):
                run_state.pop("execute_call", None)
        if isinstance(run_state, dict):
            ran_on_clause = run_state.pop("ran_on_clause", None)
            if ran_on_clause is not None:
                clause = ran_on_clause
        workflow_after = chat_session_obj.get_active_workflow()
        context_changed = getattr(workflow_after, "_current_command_context", None) is not context_before
        # Name where the command left the agent: a weak model otherwise reads
        # the context the command ran in as the one it is now in.
        return run_observation(
            alias, clause, response, folded_note=folded_note, context_changed=context_changed,
            now_clause=context_clause_for(workflow_after) if context_changed else None)

    def ask_user(clarification_request: str) -> str:
        """
        Only as the last resort, request clarification for missing information from the human user. Ask at most once about the same thing.
        The clarification_request must be plain text without any formatting.
        Questions to the user must use domain words only; never include command names, context names, or record uids.
        When asking for a name that may contain 'and' or several words, ask the user to wrap the full name in single
        or double quotes.
        Note that using the wrong command name can produce missing information errors. Double-check with the what_can_i_do tool to verify that the correct command name is being used
        """
        _clear_blocked_command(chat_session_obj)
        # reset iteration counter, everytime we ask the user
        # reset to -1, because we are dual purposing (iteration_counter <= 0) to check
        # if command passed to execute_workflow_query() originated either:
        # externally or inside agent loop immediately after an ask_user()_call 
        chat_session_obj.workflow_tool_agent.iteration_counter = -1
        if chat_session_obj.user_message_queue is not None:
            return _ask_user_tool(clarification_request, chat_session_obj=chat_session_obj)
        raise AskUserSuspend(clarification_request)

    def search_memory(question: str, alias: str) -> str:
        """
        Answers a question about ONE earlier execute_workflow_query observation.
        alias is the O-number printed on that observation's first line ("Observation O8 (...)") or in its offload label, e.g. O8.
        Goal is to locate values needed to choose the next step: a uid, a count, one field, whether an item is present.
        Offloaded observations are restored for the final answer, so do not search just to collect rows to report.
        question must be clear, relevant and complete sentence
        """
        return search_observation(
            question, alias,
            scope=scope_for_host(chat_session_obj),
            selected_archive=archive_for_path(observability_db_path(chat_session_obj)),
        )

    tools = [
        Tool(what_can_i_do, args={}),
        Tool(execute_workflow_query, args={"command": {"type": "string"}}),
        ask_user,
        # search_memory disabled 2026-10-09: in live runs it retrieved nothing useful; kept for later.
        # search_memory,
    ]

    agent = build_tool_agent(
        chat_session_obj,
        AgentSignature,
        tools,
        max_iters=max_iters,
        on_step_complete=on_step_complete,
    )
    agent.on_step_end = lambda: _end_repeat_step(chat_session_obj)
    # The policy verifier (the workflow's _policy/table.py); nothing without one.
    from fastworkflow.policy.policy_loader import attach_policy

    attach_policy(agent, chat_session_obj)
    return agent


def _previous_turn_summaries_for_planner(chat_session_obj: fastworkflow.ChatSession) -> str:
    """Every completed turn summary for the initial planner (not mid-turn replans)."""
    history = getattr(chat_session_obj, "conversation_history", None)
    messages = getattr(history, "messages", None) if history is not None else None
    if not messages:
        return ""
    lines: list[str] = []
    for turn_number, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            continue
        summary = message.get("conversation summary") or message.get("conversation_summary")
        if summary:
            lines.append(f"Turn {turn_number}: {summary}")
    return "\n".join(lines)


#: More sub-tasks than this is a list too long to fan out: run the request whole.
MAX_SUBTASKS = 10

#: Output cap (tokens) of one splitter call; ``LLM_SPLIT_MAX_TOKENS`` overrides it. Without a cap
#: the splitter's reasoning can loop for 100k+ characters, past the LM timeout. A capped answer
#: that does not parse gets DSPy's JSON retry (capped too), whose reply is the split when complete. A reply cut off at
#: the cap is never used as the split, even when it parses: with no complete reply ``_split`` retries once, then leaves the
#: request whole (fix-rj86). Set from the largest stored splitter output that
#: did not loop: 17.6k characters, about 4,700 tokens at the fewest characters per token measured
#: on long splitter outputs (3.77; fix-ajir.1).
SPLIT_MAX_TOKENS = 4700


def split_max_tokens() -> int:
    """The splitter's output cap in tokens: ``LLM_SPLIT_MAX_TOKENS``, else ``SPLIT_MAX_TOKENS``."""
    return int(fastworkflow.get_env_var("LLM_SPLIT_MAX_TOKENS", int, default=SPLIT_MAX_TOKENS))


def _split_lm(planner_lm, temperature: float, retry: bool = False):
    """The planner LM as the splitter uses it: ``temperature``, output capped at ``split_max_tokens()``.

    The retry's other temperature is what makes it a new answer. On a planner whose temperature is
    left out (a model that takes only its default, ``dspy_utils.drops_rejected_params``) it would
    change nothing, so the retry skips the response cache instead (fix-z62t).
    """
    if retry and dspy_utils.drops_rejected_params(planner_lm):
        return planner_lm.copy(temperature=temperature, max_tokens=split_max_tokens(), cache=False)
    return planner_lm.copy(temperature=temperature, max_tokens=split_max_tokens())


#: Causes of a failed split already logged at WARNING, as ``(planner model, root error type)``: a misconfigured
#: planner fails every turn the same way, and is logged once per process rather than once per turn.
_SPLIT_FAILURES_WARNED: set[tuple[str, str]] = set()


def _log_split_failure(planner_lm, error: Exception) -> None:
    """Log a split that failed, with its cause: at WARNING the first time this cause is seen, else at DEBUG.

    The cause is the error at the root of the chain: DSPy wraps every provider error in its own type.
    """
    root, seen = error, {id(error)}
    while (inner := root.__cause__ or root.__context__) is not None and id(inner) not in seen:
        root = inner
        seen.add(id(root))
    model = str(getattr(planner_lm, "model", "") or "")
    cause = (model, f"{type(root).__module__}.{type(root).__qualname__}")
    message = f"could not split the request with {model or 'the planner'}; running it whole"
    if cause in _SPLIT_FAILURES_WARNED:
        logger.debug(f"{message} ({type(root).__name__}: {root})")
        return
    _SPLIT_FAILURES_WARNED.add(cause)
    logger.warning(f"{message}. Every split that fails this way runs whole; later ones are logged at DEBUG "
                   f"({type(root).__name__}: {root})", exc_info=error)


_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")
#: A numbered-list marker anywhere in a line ("2. ", "3) "), not part of a number such as "3.5".
_INLINE_MARKER = re.compile(r"(?<!\S)(\d{1,2})[.)]\s+")
#: Typographic hyphens a model writes into names it was asked to copy exactly.
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})
#: A DSPy field marker ("[[ ## completed ## ]]") a splitter reply carried into a sub-task's text
#: (the same pattern planning_phases.clean_answer removes from answers). No request holds one.
_FIELD_MARKER = re.compile(r"[ \t]*\[\[\s*##\s*[A-Za-z0-9_ ]*?\s*##\s*\]\]")


def _split_items(reply: str) -> list[str]:
    """The sub-tasks a splitter reply lists: one per numbered line, and one per item of a numbered list
    written inline on a line ("1. A 2. B 3. C", possibly after reasoning a model wrote in the same field).

    An inline list is the last run that starts at "1." and counts up without a gap; text before it is not a
    sub-task. A stray opening brace or bracket left by the reply's JSON is cut from the end of each item,
    and a DSPy field marker (``[[ ## completed ## ]]``) from anywhere in the reply.
    """
    items: list[str] = []
    reply = _FIELD_MARKER.sub("", reply)
    for line in reply.splitlines():
        markers = list(_INLINE_MARKER.finditer(line))
        parts: list[str] = []
        for index in range(len(markers) - 1, -1, -1):
            if markers[index].group(1) != "1":
                continue
            run = [markers[index]]
            for marker in markers[index + 1:]:
                if int(marker.group(1)) != len(run) + 1:
                    break
                run.append(marker)
            if len(run) > 1:
                ends = [marker.start() for marker in run[1:]] + [len(line)]
                parts = [line[marker.end():end] for marker, end in zip(run, ends)]
                break
        if not parts and (match := _NUMBERED_LINE.match(line)):
            parts = [match.group(1)]
        items.extend(re.sub(r"\s*[{\[]+\s*$", "", part.strip()).strip() for part in parts)
    return [item for item in items if item]


#: The built-in splitter's rules: ``SplitIntoSubtasks``'s instructions up to their examples. A workflow that declares
#: ``split_examples`` in ``_commands/_workflow.py`` gets these alone as the instructions, and its examples as demos
#: (``_split_signature``, ``split_demos``; bd fix-qjne.26 stage 3).
SPLIT_RULES = """\
Split the user's request into the separate requests it bundles together.
A sub-task is a request the user could have sent on its own, with its own answer. It is NOT a step: never break one request into the steps needed to carry it out.
- job_types lists kinds of job a request in this workflow may contain, and how each one splits. Use it only to decide how to split: every sub-task is made from the user's own words, never from job_types text, and a job type the user did not ask for is not a sub-task.
- Sentences that only give background (why the user needs it, what is happening to a group) or announce what follows are not sub-tasks.
- Copy the user's own words, and every name exactly as written. A name that contains "and" is one name: keep it whole. Change only what a sub-task needs to make sense on its own: replace words that point to another part of the request ("it", "both", "each of them", "the same treatment", "that group", he/she) with what they refer to, using names.
- When the same work is asked for several named items, make one sub-task per item. What is asked about one item stays together in that item's sub-task.
- Work asked for a group as a whole (every member of a group, a whole list) is one sub-task.
- An item together with the things it refers to is one sub-task, even when the request says it in two sentences.
- Never add, drop or reinterpret any work the user asked for.
- A request that asks for one thing is returned unchanged, as a single sub-task."""

#: The built-in splitter's generic examples, after ``SPLIT_RULES`` in ``SplitIntoSubtasks``'s instructions. A
#: workflow's ``split_examples`` replace them; without those the instructions are the two, as they always were.
SPLIT_GENERIC_EXAMPLES = """\
Example 1
user_query: The job on item Alpha and the job on item Beta both fail: check each item and the thing it feeds. Three people need a new account: Pat Doe, Lee Kim and Sam Roe. Go through the whole Group North list too, every member's record. The case 'Queue stalls at night' is still open, so work it and hand it to one of the owners it names. Max Ong is moving teams, so a new locker for Max Ong.
subtasks:
1. The job on item Alpha fails: check item Alpha and the thing it feeds.
2. The job on item Beta fails: check item Beta and the thing it feeds.
3. Pat Doe needs a new account.
4. Lee Kim needs a new account.
5. Sam Roe needs a new account.
6. Go through the whole Group North list, every member's record.
7. The case 'Queue stalls at night' is still open: work it and hand it to one of the owners it names.
8. Max Ong is moving teams: a new locker for Max Ong.

Example 2
user_query: Show me Pat Doe's record, its owner and the items attached to it.
subtasks:
1. Show me Pat Doe's record, its owner and the items attached to it.

Example 3
user_query: Find item Alpha and tell me who uses it, then list the cases it has.
subtasks:
1. Find item Alpha and tell me who uses it.
2. List the cases item Alpha has."""


class SplitIntoSubtasks(dspy.Signature):
    __doc__ = f"{SPLIT_RULES}\n\n{SPLIT_GENERIC_EXAMPLES}"

    job_types: str = dspy.InputField(desc="kinds of job a request may contain, and how each one splits")
    user_query: str = dspy.InputField()
    subtasks: str = dspy.OutputField(desc="the sub-tasks as a numbered list, one per line")


def _read_planning_command_file(name: str, path: str, workflow_folderpath: str):
    """The ``PlanningCommandFile`` of ``_commands/<name>.py`` (or ``_commands/<Context>/<name>.py``, where
    *name* is ``"<Context>/<stem>"``).

    The file is imported as a command file is (``python_utils.get_module`` under the
    workflow's package, so package-relative imports work: ``from ._shared.helpers import ...`` reads
    ``_commands/_shared/helpers.py``). Raises PlanningCommandFormatError when the file stem is not a Python
    identifier or the declaration is malformed, and the import's own error for a file that
    does not import.
    """
    from fastworkflow.utils import python_utils

    CommandDirectory.check_planning_command_path(name, path)
    module = python_utils.get_module(path, workflow_folderpath)
    return planning_commands.command_from_module(name, module, path)


def _load_planning_command_file(name: str, path: str, workflow_folderpath: str):
    """The ``PlanningCommandFile`` of ``_commands/<name>.py`` (``_read_planning_command_file``), or None when it
    cannot be read: a broken planning command file costs that planning command, not the workflow. The error is
    logged, and ``fastworkflow command preview`` reports the planning command as not loaded.
    """
    try:
        return _read_planning_command_file(name, path, workflow_folderpath)
    except PlanningCommandFormatError as error:
        logger.error(f"skipping planning command {name!r}: {error}")
    except Exception:  # noqa: BLE001 - workflow code
        logger.error(f"skipping planning command {name!r}: could not import {path}", exc_info=True)
    return None


def _is_planning_file(path: str) -> bool:
    """Whether the command file at *path* defines a top-level ``PlanGenerator``: read from the
    source (``command_directory._command_kind_from_ast``), never imported."""
    try:
        return _command_kind_from_ast(Path(path))[1]
    except (ValueError, SyntaxError):
        return False


def _planning_command_files(root: str) -> list[tuple[str, str]]:
    """``(name, path)`` of each planning command file of a workflow's ``_commands`` folder, by name.

    A planning command is a command file with a top-level ``PlanGenerator`` (as ``CommandDirectory``
    classifies it): a root one is ``_commands/<name>.py``, named by its file stem, and a context
    one is ``_commands/<Context>/<name>.py``, named ``"<Context>/<stem>"`` (the file stem is a
    Python identifier, snake_case; ``_read_planning_command_file`` refuses another). A ``.py`` file or subfolder
    whose name starts with ``_`` or ``.`` (such as ``__pycache__`` or the helper package ``_shared/``) is skipped at
    both levels (``command_directory.helper_file_problems`` reports any other underscore file); ``_workflow.py`` is
    the workflow's settings (``split_settings``). Execution commands and other files are not returned.
    """
    out = []
    for entry in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        if entry.startswith(("_", ".")):
            continue
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            for sub in sorted(os.listdir(path)):
                if sub.startswith(("_", ".")):
                    continue
                sub_path = os.path.join(path, sub)
                stem, ext = os.path.splitext(sub)
                if ext == ".py" and _is_planning_file(sub_path):
                    out.append((f"{entry}/{stem}", sub_path))
            continue
        stem, ext = os.path.splitext(entry)
        if ext == ".py" and _is_planning_file(path):
            out.append((stem, path))
    return out


#: Held while a workflow's planning commands are read, so threads that start together build them once.
_parsed_planning_commands_lock = threading.RLock()


def _parsed_planning_commands(workflow_folderpath: str) -> tuple[
        tuple[str, str, tuple[str, ...], "PlanningCommandCommand", tuple[str, ...]], ...]:
    """``(name, description, template steps, command, plain_utterances)`` per planning command;
    see ``load_planning_commands``. Built once per process (``_build_parsed_planning_commands``): a planning command file that
    cannot be read is skipped (``_load_planning_command_file``), so it is not read again each turn."""
    with _parsed_planning_commands_lock:
        return _build_parsed_planning_commands(workflow_folderpath)


@functools.lru_cache(maxsize=None)
def _build_parsed_planning_commands(workflow_folderpath: str) -> tuple[
        tuple[str, str, tuple[str, ...], "PlanningCommandCommand", tuple[str, ...]], ...]:
    check_no_planning_commands_folder(workflow_folderpath)
    root = os.path.join(workflow_folderpath, "_commands")
    identifiers = identifier_patterns(workflow_folderpath)
    field_resolution = planning_commands.resolve_field_sources(workflow_folderpath)
    try:   # read once for every planning command's step and phase-source checks (None: each check reads it again)
        signatures = command_signatures(workflow_folderpath)
    except Exception as error:  # noqa: BLE001 - the checks warn and skip, as they did before
        logger.warning(f"could not read the workflow's command signatures to check its planning commands: "
                       f"{type(error).__name__}: {error}")
        signatures = None
    logged_field_source_commands: set[str] = set()
    files = []   # (name, PlanningCommandFile)
    for name, path in _planning_command_files(root):
        loaded = _load_planning_command_file(name, path, workflow_folderpath)
        if loaded is not None:
            files.append((name, loaded))
    planning_command_names = {name for name, _ in files}
    for problem in field_resolution.errors:
        if not problem.command:
            logger.error(problem.format())
        elif (
            problem.command not in planning_command_names
            and problem.command not in logged_field_source_commands
        ):
            logged_field_source_commands.add(problem.command)
            _handle_command_field_source_errors(
                problem.command,
                field_resolution.errors_for(problem.command),
                workflow_folderpath=workflow_folderpath,
            )
    out = []
    for name, loaded in files:
        command = dataclasses.replace(loaded.command, identifiers=identifiers)
        # Every problem found in the file fails this command's load, a fixed-steps command's too (bd fix-ftdl.2).
        if load_errors := field_resolution.errors_for(name):
            logged_field_source_commands.add(name)
            logger.error(f"skipping planning command {name!r}: " + "; ".join(load_errors))
            continue
        # The steps of every planning command are checked, fixed steps included: a run() step names a command of
        # the workflow with its fields (a plain sentence names none and is run by the agent).
        step_problems = step_command_problems(command, workflow_folderpath, signatures)
        if step_problems:
            logger.error(f"skipping planning command {name!r}: " + "; ".join(step_problems))
            continue
        fixed = planning_commands.fixed_steps(command)
        out.append((name, loaded.description, loaded.steps if fixed else command.template, command,
                    loaded.plain_utterances))
    return tuple(out)


def command_signatures(workflow_folderpath: str) -> dict[str, Any]:
    """The ``Signature`` of each command a planning command's steps and phase sources may name: the routed
    execution commands (``workflow_signatures``) and the planning commands. A planning command is
    not in routing; its ``Signature`` comes from ``planning_commands.resolve_field_sources``, the same
    source ``planning_phases.parameter_classes`` reads a planning command's ``Input`` from (Phase 5)."""
    from fastworkflow.observation_offloading.facts import workflow_signatures

    signatures = dict(workflow_signatures(workflow_folderpath))
    resolution = planning_commands.resolve_field_sources(workflow_folderpath)
    for name in resolution.planning_commands:
        if name in resolution.signatures:
            signatures[name] = resolution.signatures[name]
    return signatures


def step_command_problems(command: PlanningCommandCommand, workflow_folderpath: str,
                          signatures: Optional[dict] = None) -> list[str]:
    """``planning_commands.check_step_commands`` of *command* against the workflow's command
    signatures (``command_signatures``, or *signatures* when the caller read them once): a step
    naming a command the workflow lacks, or a parameter its ``Signature.Input`` lacks. A load
    error (the planning command is skipped); empty, with a logged warning, when the signatures cannot be read
    (``fastworkflow command preview --check`` still checks)."""
    if signatures is None:
        try:
            signatures = command_signatures(workflow_folderpath)
        except Exception as error:  # noqa: BLE001 - checked again by the preview's --check
            logger.warning(f"planning command {command.name!r}: could not read the workflow's command signatures "
                           f"to check its steps: {type(error).__name__}: {error}")
            return []
    return planning_commands.check_step_commands(command, signatures, _not_loading(workflow_folderpath))


def _not_loading(workflow_folderpath: str) -> frozenset[str]:
    """The commands of the workflow that do not load (by qualified and by short name): a step naming one says so,
    pointing at the command's own problem, rather than calling it no command of the workflow."""
    try:
        resolution = planning_commands.resolve_field_sources(workflow_folderpath)
        names = {problem.command for problem in resolution.errors if problem.command}
        names |= set(resolution.import_errors)
        names |= set(get_cached_command_directory(workflow_folderpath).load_errors)
    except Exception:  # noqa: BLE001 - the plain message stands
        return frozenset()
    return frozenset(names | {name.rsplit("/", 1)[-1] for name in names})


def load_planning_commands(workflow_folderpath: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """``(name, description, steps)`` for each planning command of the workflow (``_commands/<name>.py``
    and ``_commands/<Context>/<name>.py``).

    A planning command is defined by the file ``_commands/<name>.py`` at the root or
    ``_commands/<Context>/<name>.py`` in a context subfolder (format:
    ``fastworkflow.planning_commands``; which files are planning commands: ``_planning_command_files``); its name is
    the file stem (or ``"<Context>/<stem>"`` in a context) and the ``Signature`` docstring
    its description. A
    planning command's template text is its steps (its ``steps()`` on placeholder inputs; for a planning command
    without parameters or phases, its fixed ``steps()``), and ``load_planning_command_map`` holds its
    parsed form. A planning command file that cannot be imported or is malformed fails that command only: it is not
    here, the other planning commands still load, and it is a load problem (``command_preview.load_problems``, which
    ``fastworkflow run_cli``, ``train``, the server and ``command preview`` report); a malformed workflow settings
    file raises PlanningCommandFormatError. ``_commands`` also holds the workflow's
    settings, ``_workflow.py`` (``split_settings``), and the helper package ``_shared/``.
    """
    return tuple((name, description, steps) for name, description, steps, *_ in _parsed_planning_commands(workflow_folderpath))


def load_planning_command_map(workflow_folderpath: str) -> dict[str, PlanningCommandCommand]:
    """The workflow's planning commands as commands, by name: each planning command's parsed form, which a call of it
    runs (a planning command with fixed steps has one with no parameters, ``planning_commands.fixed_steps``)."""
    return {name: command for name, _, _, command, _ in _parsed_planning_commands(workflow_folderpath)}


def planning_command_utterances(workflow_folderpath: str) -> dict[str, tuple[str, ...]]:
    """Each planning command's ``Signature.plain_utterances`` (empty when it declares none), by name."""
    return {name: utterances for name, _, _, _, utterances in _parsed_planning_commands(workflow_folderpath)}


def warm_planning_commands(workflow_folderpath: str) -> None:
    """Read the workflow's planning commands and its commands' utterances once, at startup, so the first
    request does not pay for them (``load_planning_commands``; ``planning_commands.command_utterances``,
    which ``typed_planning_call`` reaches through ``command_texts``). The planning command cache is keyed by
    the path string: pass the one the turn path uses, the app workflow's ``folderpath``.
    A workflow without planning commands has nothing to warm. Never raises: a failure is
    logged and the turn path meets it again."""
    if not _planning_command_files(os.path.join(workflow_folderpath, "_commands")):
        if _jev_lm() is None:   # the classifier checks of execution commands' fields need the key too (2.46)
            try:
                _warn_missing_jev_key(workflow_folderpath)
            except Exception as error:  # noqa: BLE001 - warm-up must not stop startup
                logger.debug(f"planning command warm-up: no classifier check listing for {workflow_folderpath}: {error}")
        return
    loaded = ()
    for step in (load_planning_commands, planning_commands.command_utterances):
        try:
            result = step(workflow_folderpath)
            if step is load_planning_commands:
                loaded = result
        except Exception as error:  # noqa: BLE001 - warm-up must not stop startup
            logger.warning(f"planning command warm-up: {step.__name__} failed for {workflow_folderpath}: "
                           f"{type(error).__name__}: {error}")
    if loaded and _jev_lm() is None:
        _warn_missing_jev_key(workflow_folderpath)


#: The workflow's own settings file, ``_commands/_workflow.py``: its ``class Context`` holds the workflow-level
#: request-reading settings (see ``split_settings``). The whole workflow is a context with this one settings file.
WORKFLOW_FILE = "_workflow.py"
#: The removed spelling of those settings: a ``_split.py`` file anywhere in ``_commands/`` fails the load.
REMOVED_SPLIT_FILE = "_split.py"
#: The workflow-level settings, the only public members ``class Context`` of ``_workflow.py`` may have.
WORKFLOW_SETTINGS = ("split_examples", "identifier_patterns", "subject_hints")


@dataclasses.dataclass(frozen=True)
class SplitExample:
    """One splitter example: the request (``user_query``, "" when the example states none) and the sub-tasks it
    splits into, in order."""

    user_query: str
    subtasks: tuple[str, ...]


#: The most examples a workflow's ``split_examples`` may declare. Each example is one demo of the splitter call, and
#: each demo repeats the workflow's job list (one line per planning command, about 7,000 characters in a large
#: workflow), so the splitter's input grows by about one job list per example: 12 keeps such a call near 100,000
#: characters, and is twice the six examples that did the work measured in bd fix-qjne.26.
MAX_SPLIT_EXAMPLES = 12


@dataclasses.dataclass(frozen=True)
class SplitSettings:
    """What a workflow's ``_commands/_workflow.py`` (its ``class Context``) sets for the split stage; the defaults
    when absent."""

    #: The workflow's own examples (``split_examples``), given to the splitter as demos in place of the generic
    #: examples of its instructions; () when it declares none.
    examples: tuple[SplitExample, ...] = ()
    #: The workflow's identifier formats (``identifier_patterns``).
    identifier_patterns: tuple[re.Pattern, ...] = ()
    #: What the workflow counts as a record, for the referent step (``resolve_referents``), or None.
    subject_hints: str | None = None


def split_settings_from_module(module, path: str) -> SplitSettings:
    """The ``SplitSettings`` the ``class Context`` of a workflow settings file (already imported) declares.

    The file must define ``class Context``; its public members (own or inherited) must be workflow settings
    (``WORKFLOW_SETTINGS``), each optional. Raises PlanningCommandFormatError for a malformed one (see
    ``split_settings``).
    """
    context = module.__dict__.get("Context")
    if not isinstance(context, type):
        stray = [name for name in WORKFLOW_SETTINGS if name in module.__dict__]
        raise PlanningCommandFormatError(
            f"{path}: defines no class Context: write the workflow's settings ({', '.join(WORKFLOW_SETTINGS)}) as "
            f"class attributes of class Context{f' (it sets {stray[0]} outside a class)' if stray else ''}")
    if stray := [name for name in WORKFLOW_SETTINGS if name in module.__dict__]:
        raise PlanningCommandFormatError(
            f"{path}: {stray[0]} is set outside class Context, where it is not read: move it into class Context")
    if hasattr(context, "split_guidance"):
        raise PlanningCommandFormatError(
            f"{path}: split_guidance is removed: write split_examples = [(\"<request>\", [\"<part 1>\", \"<part 2>\"]), "
            "...], the workflow's own examples of requests and the parts they split into")
    if hasattr(context, "Signature"):
        raise PlanningCommandFormatError(
            f"{path}: a Signature (a replacement splitter prompt) is removed: write split_examples = "
            "[(\"<request>\", [\"<part 1>\", \"<part 2>\"]), ...] for the workflow's own examples; the framework's "
            "splitting rules are not replaceable")
    from fastworkflow.command_context_model import CONTEXT_HOOKS

    for name in sorted(name for name in dir(context) if not name.startswith("_") and name not in WORKFLOW_SETTINGS):
        if name in CONTEXT_HOOKS:
            raise PlanningCommandFormatError(
                f"{path}: Context.{name} is a context hook, but the workflow is not a record context and has no "
                f"parent: class Context of {WORKFLOW_FILE} holds only the workflow's settings, "
                f"{', '.join(WORKFLOW_SETTINGS)}")
        close = difflib.get_close_matches(name, WORKFLOW_SETTINGS, n=1)
        raise PlanningCommandFormatError(
            f"{path}: Context.{name} is not a workflow setting{f'; did you mean {close[0]!r}?' if close else '.'} "
            f"The settings are {', '.join(WORKFLOW_SETTINGS)}; a helper starts with _")
    examples = _declared_split_examples(getattr(context, "split_examples", None), path)
    patterns = getattr(context, "identifier_patterns", ())
    if isinstance(patterns, (str, re.Pattern)) or not isinstance(patterns, (list, tuple)):
        raise PlanningCommandFormatError(f"{path}: identifier_patterns must be a list of compiled patterns (re.compile)")
    for pattern in patterns:
        if not isinstance(pattern, re.Pattern):
            raise PlanningCommandFormatError(f"{path}: identifier_patterns holds {pattern!r}; write re.compile({pattern!r})")
    hints = getattr(context, "subject_hints", None)
    if hints is not None and not isinstance(hints, str):
        raise PlanningCommandFormatError(f"{path}: subject_hints must be a str")
    return SplitSettings(examples=examples, identifier_patterns=tuple(patterns), subject_hints=hints)


def _declared_split_examples(declared, path: str) -> tuple[SplitExample, ...]:
    """The ``split_examples`` of a workflow settings file as ``SplitExample``s; () when it declares none. Raises
    PlanningCommandFormatError naming the file and the example for a malformed one (see ``split_settings``)."""
    if declared is None:
        return ()
    shape = 'a non-empty list of (request, parts) pairs, such as [("<request>", ["<part 1>", "<part 2>"])]'
    if not isinstance(declared, list) or not declared:
        raise PlanningCommandFormatError(f"{path}: split_examples must be {shape}")
    if len(declared) > MAX_SPLIT_EXAMPLES:
        raise PlanningCommandFormatError(
            f"{path}: split_examples holds {len(declared)} examples, more than {MAX_SPLIT_EXAMPLES}: each one repeats "
            "the workflow's job list in every splitter call; keep the ones that show this workflow's own rules")
    examples = []
    for number, item in enumerate(declared, start=1):
        where = f"{path}: split_examples example {number}"
        if not (isinstance(item, tuple) and len(item) == 2):
            raise PlanningCommandFormatError(f"{where} must be a (request, parts) pair; split_examples must be {shape}")
        request, parts = item
        if not (isinstance(request, str) and request.strip()):
            raise PlanningCommandFormatError(f"{where}: the request must be a non-empty str")
        if not (isinstance(parts, list) and parts and all(isinstance(part, str) and part.strip() for part in parts)):
            raise PlanningCommandFormatError(f"{where}: the parts must be a list of at least one non-empty str")
        if len(parts) > MAX_SUBTASKS:
            raise PlanningCommandFormatError(
                f"{where}: {len(parts)} parts, more than the {MAX_SUBTASKS} a split may have")
        for part in parts:
            if "\n" in part.strip() or _NUMBERED_LINE.match(part):
                raise PlanningCommandFormatError(
                    f"{where}: the part {part.strip()[:60]!r} must be one line without a list number (the framework "
                    "numbers the parts)")
        examples.append(SplitExample(request.strip(), tuple(part.strip() for part in parts)))
    return tuple(examples)


@functools.lru_cache(maxsize=None)
def split_settings(workflow_folderpath: str) -> SplitSettings:
    """The workflow's split settings, from the ``class Context`` of its optional ``_commands/_workflow.py``; the
    defaults without one.

    The whole workflow is a context with its own settings file, ``_commands/_workflow.py`` (beside the global
    commands; its leading ``_`` keeps it from being read as a command, a planning command or a training input). Its
    ``class Context`` holds the workflow-level request-reading settings as class attributes, each optional; any other
    public member fails the load (a context hook such as ``get_parent`` too: the workflow is not a record context and
    has no parent), and so does a file that defines no ``class Context``:

    - ``split_examples``: the workflow's own examples, as plain data: a non-empty list of at most
      ``MAX_SPLIT_EXAMPLES`` ``(request, parts)`` pairs, the request a non-empty str and the parts a list of one to
      ``MAX_SUBTASKS`` non-empty one-line str without a list number. They REPLACE the generic examples of the built-in
      instructions: the splitter reads ``SPLIT_RULES`` as its instructions and each example as a demo before the real
      request, with the same job list (``split_demos``; bd fix-qjne.26 stage 3). The splitter's rules are the
      framework's and are not replaceable: ``split_guidance`` (a str added to the instructions) and a ``Signature``
      (a replacement prompt) are removed spellings, and a ``Context`` that defines either fails the load naming
      ``split_examples``.
    - ``identifier_patterns``: a list of compiled regular expressions (``re.compile``), the
      workflow's identifier formats (``identifier_patterns``).
    - ``subject_hints``: a str telling the referent step what the workflow counts as a record
      (``resolve_referents``).

    ``_commands/_workflow.py``, where these settings were once written as module attributes, is a removed spelling: a
    workflow that has one fails the load, naming ``_workflow.py``; it is never read.

    The file is imported as a planning command file is (``python_utils.get_module`` under the workflow's
    package, so package-relative imports work). It is the whole workflow configuration, so a file
    that cannot be imported (an invalid pattern raises in ``re.compile``) is not skipped, as a
    planning command file would be, but fails the load as a malformed planning command file does: the error is logged
    and PlanningCommandFormatError raised naming the file.
    """
    from fastworkflow.utils import python_utils

    removed = os.path.join(workflow_folderpath, "_commands", REMOVED_SPLIT_FILE)
    if os.path.isfile(removed):
        raise PlanningCommandFormatError(_removed_split_file_message(removed))
    path = os.path.join(workflow_folderpath, "_commands", WORKFLOW_FILE)
    if not os.path.isfile(path):
        return SplitSettings()
    try:
        module = python_utils.get_module(path, workflow_folderpath)
    except Exception as exc:  # noqa: BLE001 - reported as the file's error
        logger.error(f"could not import the workflow settings {path}", exc_info=True)
        raise PlanningCommandFormatError(f"{path}: could not import the workflow settings: {exc}") from exc
    return split_settings_from_module(module, path)


def _removed_split_file_message(path: str) -> str:
    return (f"{path}: {REMOVED_SPLIT_FILE} is removed: move its settings (split_examples, identifier_patterns, "
            f"subject_hints) to class Context in _commands/{WORKFLOW_FILE}, as class attributes")


def split_file_problems(workflow_folderpath: str) -> list[str]:
    """The mistakes of the workflow settings that do not stop them loading, one ``<file>: <reason>`` each (bd fix-qjne.16).

    - A ``_workflow.py`` anywhere but directly in ``_commands/`` (``_commands/Book/_workflow.py``) is never read: the
      settings come only from ``_commands/_workflow.py``. A ``_split.py`` below ``_commands/`` is the removed spelling
      (the one directly in ``_commands/`` fails the load, ``split_settings``).
    - An ``identifier_patterns`` entry anchored to the whole text can never match inside a message, and the patterns
      are searched inside it: it starts with ``^`` or ``\\A``, or ends with an unescaped ``$`` or ``\\Z``, and was
      not compiled with ``re.MULTILINE`` (with it ``^`` and ``$`` match at a line boundary inside a message).
    """
    root = os.path.join(os.path.abspath(workflow_folderpath), "_commands")
    problems = []
    for folder, directories, files in os.walk(root):
        directories[:] = sorted(d for d in directories if d != "__pycache__")
        if os.path.abspath(folder) == root:
            continue
        for name in (WORKFLOW_FILE, REMOVED_SPLIT_FILE):
            if name in files:
                label = os.path.relpath(os.path.join(folder, name), os.path.dirname(root)).replace(os.sep, "/")
                problems.append(_removed_split_file_message(label) if name == REMOVED_SPLIT_FILE else
                                f"{label}: the workflow settings are read only from _commands/{WORKFLOW_FILE}; this "
                                f"file is ignored here: move its settings there")
    if os.path.isfile(os.path.join(root, WORKFLOW_FILE)):
        try:
            patterns = split_settings(os.path.abspath(workflow_folderpath)).identifier_patterns
        except Exception:  # noqa: BLE001 - a file that does not load is reported by the planning command load
            patterns = ()
        problems.extend(
            f"_commands/{WORKFLOW_FILE}: identifier_patterns holds {pattern.pattern!r}, anchored to the whole text, so "
            "it can never match inside a message: identifier_patterns are searched inside the message; remove the "
            "^ and $ anchors, use \\b for word boundaries"
            for pattern in patterns if _anchored_to_whole_text(pattern))
    return problems


def _anchored_to_whole_text(pattern: re.Pattern) -> bool:
    """Whether *pattern* can only match the whole text: see ``split_file_problems``. Conservative: only a leading
    ``^`` / ``\\A`` or a trailing unescaped ``$`` / ``\\Z`` of a pattern without ``re.MULTILINE`` counts."""
    source = pattern.pattern
    if not isinstance(source, str) or pattern.flags & re.MULTILINE:
        return False
    if source.startswith(("^", "\\A")):
        return True
    for end, escaped in (("$", False), ("Z", True)):
        if source.endswith(end):
            backslashes = len(source[:-1]) - len(source[:-1].rstrip("\\"))
            # an odd run of backslashes before the end escapes it (``\$`` is a dollar; ``\Z`` is the anchor)
            if (backslashes % 2 == 1) == escaped:
                return True
    return False


def identifier_patterns(workflow_folderpath: str) -> tuple[re.Pattern, ...]:
    """The workflow's identifier patterns (``identifier_patterns`` of its optional ``_commands/_workflow.py``).

    Each match in a request is a subject for the split merge (``split_merge.merge_subtasks``),
    like a quoted name: the framework knows no workflow's identifier formats.
    """
    return split_settings(workflow_folderpath).identifier_patterns


_EXAMPLE_REQUEST_LINE = re.compile(r"^\s*user_query\s*:\s*(.*?)\s*$", re.I)
_EXAMPLE_SUBTASKS_LINE = re.compile(r"^\s*subtasks\s*:\s*$", re.I)


def split_examples(workflow_folderpath: str | None, *, workflow_only: bool = False) -> tuple[SplitExample, ...]:
    """The examples the splitter is shown for the workflow, as data (bd fix-0iuo.23): the workflow's declared
    ``split_examples`` when it has them (they replace the generic ones, so with or without *workflow_only* they are
    the whole answer), else ``SplitIntoSubtasks``'s (also for None).

    An example of the instructions is a line ``subtasks:`` and the numbered lines after it (each read by the rule a
    splitter reply's numbered lines are read by), up to the first non-blank line that is not numbered; its request is
    the last ``user_query: ...`` line before it since the previous example. With *workflow_only*, only the workflow's
    own examples are returned (its ``split_examples``): () when its settings set none. A settings file that does
    not load raises as ``split_settings`` does."""
    if workflow_folderpath is not None and (settings := split_settings(workflow_folderpath)).examples:
        return settings.examples
    if workflow_only:
        return ()
    instructions = SplitIntoSubtasks.instructions
    examples: list[SplitExample] = []
    request, subtasks, inside = "", None, False
    for line in (instructions or "").splitlines():
        if _EXAMPLE_SUBTASKS_LINE.match(line):
            if subtasks is not None:
                examples.append(SplitExample(request, tuple(subtasks)))
                request = ""
            subtasks, inside = [], True
            continue
        if inside and (match := _NUMBERED_LINE.match(line)):
            subtasks.append(match.group(1))
            continue
        if line.strip():
            inside = False
        if match := _EXAMPLE_REQUEST_LINE.match(line):
            if subtasks is not None:
                examples.append(SplitExample(request, tuple(subtasks)))
                subtasks = None
            request = match.group(1)
    if subtasks is not None:
        examples.append(SplitExample(request, tuple(subtasks)))
    return tuple(examples)


@functools.lru_cache(maxsize=None)
def _split_signature(workflow_folderpath: str) -> type[dspy.Signature]:
    """``SplitIntoSubtasks``, with the workflow's ``_commands/_workflow.py`` applied: with its ``split_examples``, the
    built-in rules alone (``SPLIT_RULES``; the examples are demos, ``split_demos``), else unchanged."""
    if split_settings(workflow_folderpath).examples:
        return SplitIntoSubtasks.with_instructions(SPLIT_RULES)
    return SplitIntoSubtasks


def split_demos(workflow_folderpath: str | None, job_types: str) -> list[dspy.Example]:
    """The workflow's ``split_examples`` as the splitter's demos, in order (none without them): each carries the
    request as ``user_query``, *job_types* (the job list of the call, the same text the real request is given) and the
    parts as the numbered list a splitter reply is read as (``subtasks``). No reasoning text is given."""
    examples = split_settings(workflow_folderpath).examples if workflow_folderpath else ()
    return [dspy.Example(job_types=job_types, user_query=example.user_query,
                         subtasks="\n".join(f"{n}. {part}" for n, part in enumerate(example.subtasks, start=1)))
            .with_inputs("job_types", "user_query") for example in examples]


#: ``prompt_source`` on the ``fw.split`` span: the splitter's own instructions, or the built-in rules with a
#: workflow's ``split_examples`` as demos in place of the generic examples.
SPLIT_PROMPT_BUILT_IN = "built_in"
SPLIT_PROMPT_BUILT_IN_AND_WORKFLOW = "built_in+workflow"


def split_prompt(workflow_folderpath: str | None) -> dict[str, str]:
    """Which splitter prompt is in force for the workflow (bd fix-qjne.26 stage 0): ``{"digest", "source"}``.

    ``digest`` is the first 12 hex characters of the SHA-256 of the splitter signature's instructions and of each
    field's name and description, in order (the turn's optional ``conversation`` field is not part of it), then of
    each declared example's request and parts (``split_examples``; not the job list its demo carries, which is the
    call's), so two splits with the same digest were asked with the same prompt text. ``source`` is
    ``SPLIT_PROMPT_BUILT_IN_AND_WORKFLOW`` when the workflow's ``_commands/_workflow.py`` declares ``split_examples``
    (the built-in rules and its examples), else ``SPLIT_PROMPT_BUILT_IN`` (also for None). A workflow settings file that does not load raises as
    ``split_settings`` does."""
    signature = _split_signature(workflow_folderpath) if workflow_folderpath else SplitIntoSubtasks
    settings = split_settings(workflow_folderpath) if workflow_folderpath else SplitSettings()
    parts = [signature.instructions or ""]
    for name, field in signature.fields.items():
        parts.append(f"{name}: {(field.json_schema_extra or {}).get('desc', '')}")
    for example in settings.examples:
        parts.append(f"example: {example.user_query}")
        parts.extend(f"{n}. {part}" for n, part in enumerate(example.subtasks, start=1))
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    source = SPLIT_PROMPT_BUILT_IN_AND_WORKFLOW if settings.examples else SPLIT_PROMPT_BUILT_IN
    return {"digest": digest, "source": source}


def _job_types_for(workflow_folderpath: str | None) -> str:
    """One ``- name: description`` line per planning command of the workflow; "(none)" without planning commands."""
    planning_commands = load_planning_commands(workflow_folderpath) if workflow_folderpath else ()
    if not planning_commands:
        return "(none)"
    return "\n".join(f"- {name}: {description}" for name, description, _ in planning_commands)


def _planning_scope(name: str) -> str:
    """The context name of planning command ``name`` (``"Library"`` for ``"Library/dune_search"``),
    or ``""`` for a root planning command."""
    return name.rpartition("/")[0]


def planning_scopes(workflow) -> tuple[str, ...]:
    """The planning command scopes on ``workflow``'s lookup path, ending with ``""`` for the root.

    Planning commands are matched scope by scope along the lookup path:
    1. The active context.
    2. Its bases, in ``CommandContextModel`` order, transitively.
    3. Each live ancestor's context (``context_navigation.live_chain``, up to the root context
       object) and that ancestor's bases.
    4. The root planning commands (``""``).

    A scope already on the path is skipped. Does not depend on intent-classifier code.
    """
    if workflow is None:
        return ("",)
    from fastworkflow.context_navigation import live_chain

    return lookup_scopes(getattr(workflow, "folderpath", None), [name for _, name in live_chain(workflow)])


def planning_names_on_path(workflow) -> list[str]:
    """The planning commands the matcher could choose from for *workflow*'s current state, in lookup order: the
    commands of each scope of ``planning_scopes`` (nearest scope first, the root's last), each scope's by name.

    The one source of "the plans available here": the intent loop's decision offers exactly these, scope by scope, so a
    listing of them, and the check of a plan a client names, can never disagree with what the decision could have picked.
    """
    if workflow is None or not getattr(workflow, "folderpath", None):
        return []
    by_scope: dict[str, list[str]] = {}
    for name, _, _ in load_planning_commands(workflow.folderpath):
        by_scope.setdefault(_planning_scope(name), []).append(name)
    names: list[str] = []
    for scope in dict.fromkeys(planning_scopes(workflow)):
        names.extend(sorted(by_scope.get(scope, ())))
    return names


#: The planning command matcher shows at most this many of a planning command's ``plain_utterances`` (its first
#: ones) after its description, to bound the prompt.
MATCHER_EXAMPLES = 9


def _matcher_job_types(workflow_folderpath: str, examples: int = MATCHER_EXAMPLES,
                       scope: str | None = None) -> str:
    """The planning command matcher's job types: one ``- name: description`` line per planning command, followed by
    ``e.g. "...", "..."`` when the planning command declares ``plain_utterances`` (at most ``examples``).
    With ``scope`` given (``""`` for root, or a context name), only that scope's planning commands are listed.

    The commands and their texts come from the detector contract (``intent_detector.planning_candidates``,
    the planning candidates of each scope), and each line is ``intent_detector.candidate_line``: the text is
    what it always was, byte for byte (``tests/test_intent_matcher_engine.py`` compares it with the old builder)."""
    from fastworkflow import intent_detector

    lines = [intent_detector.candidate_line(candidate.name, candidate.description, candidate.examples, examples)
             for candidate in intent_detector.planning_candidates(workflow_folderpath)
             if scope is None or _planning_scope(candidate.name) == scope]
    return "\n".join(lines)


#: The bar of ``run_listing.planning_candidates``: a planning command is the confident one of a listing's gathering at
#: or above this probability. It decides no plan: "which command does this text mean" is the intent loop's alone.
PLANNING_COMMAND_MATCH_THRESHOLD = 0.5

#: The planner's instructions (the planner signatures use them as their docstring unchanged).
PLANNER_INSTRUCTIONS = """
    Carefully review the user_query and generate a next steps sequence based only on available commands.
    Walk the graph of commands based on the 'available_from' hints to build the most appropriate command sequence.
    Use names and values exactly as written in the user_query; never abbreviate or reword them.

    IMPORTANT: 9 times out of 10 information can be found via available commands. However, when generating the plan:
    - If required information is missing and cannot be found via commands, explicitly specify in the plan that the user needs to be consulted
    - If confirmation is needed before proceeding, explicitly specify in the plan that user confirmation is required
    """


def planner_commands_text(current_workflow) -> str:
    """The planner's listing: every context's commands, each planning command with its template."""
    return CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow._current_command_context_name,
        navigation_workflow=current_workflow,
    )


def planner_listing_text(current_workflow, sub_task: str,
                         planning_probabilities: Mapping[str, float] | None = None) -> str:
    """The planner's listing for *sub_task* (run_listing.run_listing_text). Production (no experiment arm, the default
    FULL): ``planner_commands_text``, byte for byte. The experiment selector ``THRESHOLD`` gives it while it is at most
    ``run_listing.FULL_LISTING_MAX_CHARS``, else the trained-head shortlist; ``COMMAND_SHORTLIST`` gives the shortlist.
    *planning_probabilities*: the turn's planning-command match distribution (None without JEV_API_KEY)."""
    from fastworkflow import run_listing

    arm = run_listing.effective_listing_arm(current_workflow)
    if arm == run_listing.FULL:
        return planner_commands_text(current_workflow)
    return run_listing.run_listing_text(current_workflow, sub_task, arm=arm,
                                        planning_probabilities=planning_probabilities)


def sub_task_commands_text(chat_session_obj, sub_task: str,
                           planning_probabilities: Mapping[str, float] | None = None) -> str:
    """The ``available_commands`` of a sub-task's run (run_listing.run_listing_text). Production (no experiment arm,
    the default FULL): ``_what_can_i_do``, byte for byte. The experiment selector ``THRESHOLD`` gives it while it is at
    most ``run_listing.FULL_LISTING_MAX_CHARS``, else the trained-head shortlist; ``COMMAND_SHORTLIST`` gives the
    shortlist. *planning_probabilities*: the turn's planning-command match distribution (None without JEV_API_KEY)."""
    from fastworkflow import run_listing

    arm = run_listing.effective_listing_arm(chat_session_obj.get_active_workflow())
    full = _what_can_i_do(chat_session_obj)
    if arm == run_listing.FULL:
        return full
    return run_listing.run_listing_text(chat_session_obj.get_active_workflow(), sub_task, arm=arm,
                                        planning_probabilities=planning_probabilities, full=full)


#: A section header of the planner's listing: the active context's, the one a command enters, or the
#: scope of a command callable here by its name (3.9, Catalogs: an enclosing context, or the global scope).
_LISTING_SECTION = re.compile(
    r"^Commands available (?:in the current context\b"
    r"|after entering the (?P<context>\S+) context\b"
    r"|here from (?:the (?P<here>\S+) context|(?P<scope>the global scope))\b)")
#: A command's own line in the planner's listing (its block starts here).
_LISTING_COMMAND = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*$")


def typed_planner_commands_text(current_workflow) -> str:
    """The typed planning call's listing (design 3.4): the planner's listing with each command's
    block cut to its name, the first sentence of its description and its parameter names.

    The section headers and the order of the commands are the planner listing's (so every command
    it names is named here, once per section), and each block's description and parameters are
    those of the context the section is for: a command listed in several contexts keeps what it
    does in each. A planning command's parameters are its own typed fields.
    """
    full = planner_commands_text(current_workflow)
    folderpath = current_workflow.folderpath
    active = current_workflow._current_command_context_name
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    crd = fastworkflow.RoutingRegistry.get_definition(folderpath)
    # (context, name) -> (description, parameter names), from every context's metadata; and the
    # first entry of each name, for a block whose section is not one of the contexts.
    by_context: dict[tuple[str, str], tuple[str, list[str]]] = {}
    by_name: dict[str, tuple[str, list[str]]] = {}
    for context in crd.contexts:
        info = CommandMetadataAPI.get_enhanced_command_info(
            subject_workflow_path=folderpath, cme_workflow_path=cme_path, active_context_name=context)
        for cmd in info.get("commands", []):
            name = cmd.get("name")
            if not name:
                continue
            doc = " ".join(str(cmd.get("doc_string") or "").split())
            params = [str(i.get("name")) for i in cmd.get("inputs", []) or [] if i.get("name")]
            by_context.setdefault((context, name), (doc, params))
            by_name.setdefault(name, (doc, params))
    for planning_command_name, planning_command in load_planning_command_map(folderpath).items():
        name = planning_command_name.rpartition("/")[2]
        params = [p.name for p in planning_command.params]
        for key, (doc, _) in list(by_context.items()):
            if key[1] == name:
                by_context[key] = (doc or planning_command.description, params)
        if name in by_name:
            doc, _ = by_name[name]
            by_name[name] = (doc or planning_command.description, params)
        else:
            by_name[name] = (planning_command.description, params)
    out: list[str] = []
    context = active
    for line in full.split("\n"):
        section = _LISTING_SECTION.match(line)
        if section:
            if section["scope"]:
                context = "*"
            else:
                context = section["context"] or section["here"] or active
            out.append(line)
            continue
        command = _LISTING_COMMAND.match(line)
        if command is None:
            continue
        name = command.group(1)
        doc, params = by_context.get((context, name)) or by_name.get(name, ("", []))
        out.append(f"- {name}")
        sentence = re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0] if doc else ""
        if sentence:
            out.append(f"  {sentence}")
        if params:
            out.append(f"  parameters: {', '.join(params)}")
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def _jev_lm():
    """The Jev (dspy.experimental TypeSafe) LM, or None without a JEV_API_KEY."""
    # default="" (not None): get_env_var warns about a variable that is missing and has no default,
    # and the missing key is the INFO line of 3.9, not a warning.
    key = fastworkflow.get_env_var("JEV_API_KEY", default="")
    if not key:
        return None
    lm = TypeSafe(api_key=key, cache=dspy_utils.lm_cache_enabled())
    lm.fastworkflow_role = "JEV"    # its model-call spans say whose call it is (``dspy_logger``), as a role's LM does
    return lm


@functools.lru_cache(maxsize=None)
def _warn_missing_jev_key(workflow_folderpath: str) -> None:
    """The INFO line of design 3.9, once per workflow folder per process, when a workflow has
    planning commands (``_commands``) and ``JEV_API_KEY`` is not set. Planning commands are ordinary
    intent labels, so intent detection still matches them; without the key, Jev's planning-match
    candidates are off, and so are the classifier checks (``classifier_confirms``) of the fields that name planning commands.
    """
    fields = planning_commands.verify_field_names(workflow_folderpath)
    if not _planning_command_files(os.path.join(workflow_folderpath, "_commands")):
        if fields:
            logger.info(f"{workflow_folderpath}: JEV_API_KEY is not set: the classifier checks of "
                        f"{', '.join(fields)} are inactive.")
        return
    checks_clause = f" and the classifier checks of {', '.join(fields)}" if fields else ""
    logger.info(
        f"{workflow_folderpath}: planning commands are matched by intent detection; without "
        f"JEV_API_KEY, Jev's planning-match candidates{checks_clause} are off."
    )


def _choice_value_and_probability(choice) -> tuple[str, float]:
    value = str(choice.value)
    return value, float((choice.probabilities or {}).get(value, choice.confidence))


def _template_text(steps: tuple[str, ...]) -> str:
    lines = "\n".join(f"- {step}" for step in steps)
    return (f"A job like this is usually done as:\n{lines}\n"
            "Adapt this to the request: drop steps it does not need and add steps it asks for.")


#: A reply's non-answer intent applies only at or above this Jev probability.
REPLY_INTENT_THRESHOLD = 0.5
REPLY_INTENTS = (
    ("answer", "The reply gives a value or direction for the work to go on, or asks a question back about the "
               "request (which one, what are the options): the agent answers it and asks again."),
    ("stop_asking", "The reply explicitly asks the agent to stop asking questions: 'stop asking', 'just do it', "
                    "'don't ask me again', 'never mind'. A question back or a hesitation is never this."),
    ("abort", "The reply asks to stop, cancel or abort the work."),
    ("wait", "The reply asks for a moment or says the user is checking, or hesitates ('hmm, not sure', 'let me "
             "think'); it gives no value and asks nothing."),
)

#: The explicit requests to stop being asked. stop_asking needs one of these in the reply as well as Jev's
#: choice (see classify_user_reply): a question back or a hesitation never stops the asking. The patterns are
#: general families (a verb of stopping + asking/questions/pestering; "no more questions"; "don't ask me
#: (about) that/this/it (again)"; "did not ask me about that again"), not the wording of any one test set.
_STOP_ASKING_REQUEST = re.compile(
    r"\b(?:stop\s+(?:(?:with\s+)?the\s+)?(?:asking|questions?)"
    r"|(?:quit|stop)\s+(?:with\s+the\s+)?(?:asking|questions?|pestering|bugging|hassling|badgering|nagging)"
    r"|no\s+(?:more|further)\s+questions?"
    r"|(?:don'?t|do\s+not|never)\s+ask(?:\s+me)?(?:\s+(?:about\s+)?(?:that|this|it))?"
    r"(?:\s+(?:again|anymore|any\s+more))?"
    r"|(?:did\s+not|didn'?t|would\s+not|wouldn'?t|won'?t)\s+ask\s+me(?:\s+about\s+(?:that|this|it))?"
    r"(?:\s+(?:again|anymore|any\s+more))?"
    r"|just\s+do\s+it|no\s+need\s+to\s+ask|never\s*mind)\b",
    re.IGNORECASE,
)


class ClassifyReply(dspy.Signature):
    """Decide what a user's reply to the agent's clarification request does.

    Three readings are easy to confuse. A question back ("which one do you mean?", "what are the
    options?") is "answer": the agent answers it and asks again, so it is not a request to stop asking.
    A hesitation ("hmm, not sure", "let me think") is "wait". Only an explicit request to stop being
    asked ("stop asking", "just do it", "don't ask me again", "never mind") is "stop_asking".
    """

    question: str = dspy.InputField(desc="the clarification request the agent asked the user")
    reply: str = dspy.InputField(desc="the user's reply")
    intent: Choice[REPLY_INTENTS] = dspy.OutputField(
        desc="What does the reply do? A question back is answer; a hesitation is wait; "
             "stop_asking only for an explicit request to stop being asked.")


def _reply_wait_enabled() -> bool:
    """False when FW_JEV_REPLY_WAIT is set to off, 0 or false (case-insensitive)."""
    value = str(fastworkflow.get_env_var("FW_JEV_REPLY_WAIT", default="on"))
    return value.strip().lower() not in {"off", "0", "false"}


def _classify_reply_predictor():
    """The predictor for ClassifyReply; with FW_JEV_REPLY_WAIT off, "wait" is not an offered option."""
    if _reply_wait_enabled():
        return dspy.Predict(ClassifyReply)
    options = tuple(option for option in REPLY_INTENTS if option[0] != "wait")
    return dspy.Predict(ClassifyReply.with_updated_fields("intent", type_=Choice[options]))


def classify_user_reply(reply: str, question: str) -> str:
    """The reply's intent: "abort", "stop_asking", "wait", or "answer" (the default).

    "wait" is a hesitation ("hold on", "one sec"): the user has not answered yet. Any non-answer
    intent needs Jev to be confident at REPLY_INTENT_THRESHOLD. With FW_JEV_REPLY_WAIT off, "wait"
    is not offered to Jev at all, so the result is one of the other three.

    "stop_asking" is also vetoed by a deterministic guard: it stands only when the reply holds an
    explicit request to stop being asked (_STOP_ASKING_REQUEST). A question back or a hesitation that
    Jev reads as stop_asking becomes "answer", so ask_user stays available for the turn. A stop phrase
    that also carries an answer value still stands as stop_asking: the value check that would have
    caught it was dropped (fix-jd9o.12, owner decision). Measured on a fresh set, that check vetoed 5
    of 10 real stops, while without it about 1 in 130 value replies that also say "stop asking" is
    taken as a stop. A stop only ends asking for that turn, so the trade favours the real stops.

    Never raises: without a Jev key, or on a failure, the reply is taken as an answer. That is the
    same rule on both paths: stop_asking needs Jev's reading and an explicit request, and without a
    key there is no reading, so the reply is an answer.
    """
    jev_lm = _jev_lm()
    if jev_lm is None:
        return "answer"
    try:
        predictor = _classify_reply_predictor()
        with dspy_utils.role_context("JEV", jev_lm):
            choice = predictor(question=question, reply=reply).intent
        value, probability = _choice_value_and_probability(choice)
        logger.info(f"reply intent for {reply!r}: {value} (confidence {probability:.2f})")
        if value == "stop_asking" and not _STOP_ASKING_REQUEST.search(reply):
            logger.info(f"reply intent {value!r} vetoed: the reply holds no explicit request to stop asking")
            return "answer"
        if value != "answer" and probability >= REPLY_INTENT_THRESHOLD:
            return value
        return "answer"
    except Exception as error:  # noqa: BLE001 - the reply is then taken as an answer; never fail the turn
        from fastworkflow import remote_outage

        remote_outage.report(error, "reply intent")
        return "answer"


def command_match_signature(options: tuple):
    """The signature of the model matcher engine's one call (``intent_detector.MatcherEngine``): which of
    *options* (``(name, description)`` pairs, ``none`` among them) a message asks for, given the commands
    listed by scope. Defined here, where ``Choice`` annotations are evaluated, not kept as text."""

    class MatchCommand(dspy.Signature):
        """Decide which command a message to a workflow asks for."""

        message: str = dspy.InputField(desc="the message sent to the workflow")
        commands: str = dspy.InputField(
            desc="the available commands grouped by scope, the scope the user is in first, one per line as "
                 "'- name: description', some followed by example messages for that command (e.g. ...)")
        command: Choice[options] = dspy.OutputField(
            desc="Which command does the message ask for? 'none' if none of them fits.")

    return MatchCommand


#: Phrases that point at other parts of the request instead of naming them; a sub-task with one is broken.
_UNRESOLVED_REFERENCE = re.compile(
    r"\b(?:each of them|both of them|all of them)\b", re.IGNORECASE)


#: A brace placeholder the model left in a sub-task ("{HIGHER_RISK_PERSON} carries more risk"): an identifier, or
#: words, in braces. "{}", "{ }" and JSON-ish text ('{"a": 1}', '{1}') do not match: they start with no letter.
_BRACE_PLACEHOLDER = re.compile(r"\{[A-Za-z][A-Za-z0-9_ ]*\}")

#: The same in square brackets ("[Person 1 name] starts on the roster") and a numbered label that opens a sub-task
#: ("Control 1: who is in it"); both are the model's stand-ins for a record, not text of the request.
_SQUARE_PLACEHOLDER = re.compile(r"\[[A-Za-z][A-Za-z0-9_ ]*\]")
_NUMBERED_OPENER = re.compile(r"^\s*[A-Za-z]+(?: [A-Za-z]+)? \d+\s*:")


def _invented_placeholders(text: str, user_query: str) -> list[str]:
    """The placeholders of ``text`` ({NAME}, [Person 1 name], an "X N:" opener) whose exact text is not in the
    user's request (a request may hold them)."""
    found = _BRACE_PLACEHOLDER.findall(text) + _SQUARE_PLACEHOLDER.findall(text)
    opener = _NUMBERED_OPENER.match(text)
    if opener:
        found.append(opener.group(0).strip())
    return [p for p in found if p not in user_query]


def _visibly_broken(subtasks: list[str], user_query: str = "") -> bool:
    """Two identical sub-tasks, one that still refers to the rest of the request ("each of them"), or one with
    a brace placeholder the request does not contain."""
    return (len(set(subtasks)) < len(subtasks)
            or any(_UNRESOLVED_REFERENCE.search(t) or _invented_placeholders(t, user_query) for t in subtasks))


def _whole_if_placeholder(user_query: str, subtasks: list[str], rules: list[str] | None = None) -> list[str]:
    """The request whole when a chosen sub-task still holds a placeholder the request does not (it would be
    shown, approved, and then fail downstream as a name)."""
    if any(_invented_placeholders(t, user_query) for t in subtasks):
        logger.warning("a sub-task still holds an unresolved placeholder; leaving the request whole")
        if rules is not None:
            rules.append("whole_if_placeholder")
        return [user_query]
    return subtasks


def _breakage(user_query: str, subtasks: list[str], workflow_folderpath: str | None) -> tuple[int, int]:
    """How broken a visibly broken split is, for choosing between two: (the request's records named by more
    than one sub-task, each extra naming once; the sub-tasks with a leftover reference).

    A record in two sub-tasks is a job split twice, or a summary sub-task repeating jobs the others hold
    ("Quarterly review of A, B and C: each of them ...", next to A's, B's and C's own), or a job the model
    invented for a record another sub-task already covers; the merge then runs those jobs twice.
    """
    references = sum(bool(_UNRESOLVED_REFERENCE.search(t)) for t in subtasks)
    try:
        identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()
        known = request_subjects(user_query, identifiers)
        in_request = subjects_in(user_query, known, identifiers)
        named = [s for t in subtasks for s in subjects_in(t, known, identifiers) if s in in_request]
    except Exception:  # noqa: BLE001 - only a tie-break; never fail the turn
        logger.warning("could not count the records the split names twice", exc_info=True)
        named = []
    return len(named) - len(set(named)), references


#: The splitter's ``conversation`` input (added only when the turn has earlier turns, so a first turn's prompt is
#: unchanged): the earlier turns' summaries, for words of the new request that point back to them.
SPLIT_CONVERSATION_DESC = ("earlier turns of the conversation, for resolving words of user_query that point back to "
                           "them ('its loans' after a turn about Dune). Background only: nothing in it is a "
                           "sub-task, and a sub-task takes from it only the name such a word refers to")


#: The turn's recorder of its split (bd fix-hfqv, recording only): ``recording_split`` sets it, ``split_into_subtasks``
#: fills it. A caller that did not set one records nothing, and a replaced ``split_into_subtasks`` fills nothing.
_split_recorder: contextvars.ContextVar[dict | None] = contextvars.ContextVar("fw_split_recorder", default=None)


@contextlib.contextmanager
def recording_split():
    """A dict that ``split_into_subtasks`` fills when the model was asked to split the request inside the block, with
    ``request``, ``raw`` (the model's sub-tasks of the attempt used, before any code rule), ``merged`` (the list the
    turn runs with), ``changed_by`` (the code steps that changed it, in order; absent when none) and ``attempts``
    (model calls made), ``failed`` when the split gave no answer (``SPLIT_FAILED_CUT_OFF``, ``SPLIT_FAILED_ERROR``),
    ``prompt_digest`` and ``prompt_source`` (``split_prompt``) when the splitter's prompt was built, and ``gate`` (the
    one-job gate's verdict, ``_one_job_gate``) when the gate was asked. A request the gate kept whole has ``raw`` empty,
    ``merged`` the request, ``attempts`` 0 and no prompt keys. It stays empty when no split was asked for."""
    record: dict = {}
    token = _split_recorder.set(record)
    try:
        yield record
    finally:
        _split_recorder.reset(token)


#: The one-job gate in front of the splitter (bd fix-qjne.26): the remote decision model is asked whether the request
#: is one job, and a confident "one" runs it as one part with no splitter call. ``FW_SPLIT_GATE_THRESHOLD`` (workflow env)
#: is the bar, read in one place (``split_gate_threshold``); ``off`` turns the gate off. Measured on 167 labelled requests
#: of one large workflow, three asks each: at 0.9, 83% of the single-part requests skipped the splitter and no
#: multi-part one did. The default is 0.85: on that workflow's benchmark (168 tasks, two arms) it skipped no multi-part
#: request (highest score on one 0.69), and the requests scoring between 0.85 and 0.9 were all single-part, including
#: the over-splits that cost lines. It was chosen on the tasks it was tested on, from one workflow: a workflow that sees
#: a multi-part request kept whole should raise it.
SPLIT_GATE_THRESHOLD_SETTING = "FW_SPLIT_GATE_THRESHOLD"
DEFAULT_SPLIT_GATE_THRESHOLD = 0.85
SPLIT_GATE_OFF = "off"

_ONE_OR_SEVERAL = (
    ("one", "The request asks for one job: one thing to look up or do, or several details about one named "
            "thing that one kind of job covers, or one action over several records handled together."),
    ("several", "The request asks for two or more separate jobs: different named things that each need their "
                "own work, or different kinds of work, done one after another."),
)


class OneOrSeveralJobs(dspy.Signature):
    """Decide whether a request sent to a workflow asks for one job or for several separate jobs."""

    request: str = dspy.InputField(desc="the request sent to the workflow")
    job_types: str = dspy.InputField(desc="kinds of job a request may contain, and how each one splits")
    jobs: Choice[_ONE_OR_SEVERAL] = dspy.OutputField(
        desc="Does the request ask for one job or for several separate jobs? "
             "Several details or records that one kind of job covers are one job.")


def split_gate_threshold() -> float | None:
    """The one-job gate's bar: ``FW_SPLIT_GATE_THRESHOLD``, a number above 0 and at most 1 (default 0.85; any other
    value warns once and is the default), or None when it is ``off``."""
    from fastworkflow.train.intent_model_layout import parse_threshold

    raw = fastworkflow.get_env_var(SPLIT_GATE_THRESHOLD_SETTING, default="")
    if str(raw if raw is not None else "").strip().lower() == SPLIT_GATE_OFF:
        return None
    return parse_threshold(raw, DEFAULT_SPLIT_GATE_THRESHOLD, SPLIT_GATE_THRESHOLD_SETTING)


def _one_job_gate(user_query: str, job_types: str) -> dict | None:
    """The one-job gate's verdict, ``{score, label, threshold, skipped}`` (``score`` is the probability of "one",
    ``skipped`` whether the splitter is skipped), or None when it was not asked: no remote decision model, the gate
    ``off``, or a failed call (reported through ``remote_outage``; the splitter then runs). The request and the same job
    list the splitter is given (``_job_types_for``) are the inputs."""
    jev_lm = _jev_lm()
    if jev_lm is None:
        return None
    threshold = split_gate_threshold()
    if threshold is None:
        return None
    # The gate's own span, in the split stage (bd fix-qjne.34): its model call is the span's child.
    host = tracing.current_host()
    span = tracing.start_span(host, tracing.SPAN_SPLIT_GATE, attributes={"threshold": threshold})
    try:
        with dspy_utils.role_context("JEV", jev_lm):
            choice = dspy.Predict(OneOrSeveralJobs)(request=user_query, job_types=job_types).jobs
        label, probability = _choice_value_and_probability(choice)
        score = probability if label == "one" else float((choice.probabilities or {}).get("one", 1.0 - probability))
    except Exception as error:  # noqa: BLE001 - the gate is an optimisation; the splitter runs
        from fastworkflow import remote_outage

        remote_outage.report(error, "split gate")
        tracing.end_span(host, span, status=tracing.STATUS_ERROR,
                         attributes={"score": None, "label": None, "skipped": False})
        return None
    skipped = label == "one" and score >= threshold
    tracing.end_span(host, span, attributes={"score": score, "label": label, "skipped": skipped})
    logger.info(f"split gate: {label} (p(one) {score:.2f}, bar {threshold}); "
                f"{'the request is one part, no split' if skipped else 'the splitter runs'}")
    return {"score": score, "label": label, "threshold": threshold, "skipped": skipped}


def split_into_subtasks(user_query: str, planner_lm=None,
                        workflow_folderpath: str | None = None, conversation: str = "", workflow=None) -> list[str]:
    """The request as independent sub-tasks, in order (``_split``); the split is also recorded for the turn's span when a
    ``recording_split`` block is open."""
    state: dict = {}
    result = _split(user_query, planner_lm, workflow_folderpath, conversation, workflow, state)
    record = _split_recorder.get()
    if record is not None and (state.get("attempts") or state.get("gate")):
        record.update(request=user_query, raw=list(state.get("raw", [])), merged=[str(t) for t in result],
                      attempts=state.get("attempts", 0))
        if state.get("gate"):
            record["gate"] = dict(state["gate"])
        if state.get("rules"):
            record["changed_by"] = list(state["rules"])
        if state.get("doubts"):
            record["doubts"] = list(state["doubts"])
        if state.get("failed"):
            record["failed"] = state["failed"]
        if state.get("prompt"):
            record["prompt_digest"] = state["prompt"]["digest"]
            record["prompt_source"] = state["prompt"]["source"]
    return result


#: ``failed`` on the ``fw.split`` span (and on the ``fw.split.splitter`` span of the attempt): why the split gave no answer and the request ran as one part. Every
#: answer stopped at the output cap (the retry without the reasoning field included), or the model call raised.
SPLIT_FAILED_CUT_OFF = "cut_off"
SPLIT_FAILED_ERROR = "error"


def _split(user_query: str, planner_lm, workflow_folderpath: str | None, conversation: str, workflow,
           state: dict) -> list[str]:
    """The request as independent sub-tasks, in order; ``[user_query]`` when it is not split.
    ``state`` is filled for the recording: ``attempts``, ``raw``, ``rules`` and ``prompt`` (``split_prompt``).

    ``user_query`` is the user's new request only: it is what the merge treats as the request (its ground truth,
    and the text it returns when everything merges back into one job). ``conversation`` (optional) is the
    earlier turns' summaries, given to the splitter as a separate input (``SPLIT_CONVERSATION_DESC``) so it can
    resolve references to them; the merge reads it only so that a name taken from it is not an invention, and
    never returns it (fix-bydz).

    One call on the planner LM, with no command catalog: the split is about the
    request, not about how to carry it out. It is shown the workflow's planning commands
    (``_commands``), as the kinds of job it may contain.
    Any failure, an empty answer, or more than MAX_SUBTASKS sub-tasks leaves the
    request whole. A reply cut off at ``split_max_tokens()`` is never parsed into a split, even
    when it parses (a field header written mid-reasoning, e.g. the output template
    echoed back, parses as "{subtasks}"). A complete reply is the answer whichever adapter produced it: the library's
    own fallback inside the same call (its JSON retry of a cut-off first reply) is used when its reply was complete.
    Only a call with no complete reply that parsed gets the one retry, without the free
    reasoning field (where the model loops), and a second cut-off reply leaves the
    request whole, recorded as a failed split (``state["failed"]``) and logged. A non-empty sub-tasks field with no numbered item (``it``) is
    visibly broken too and gets the same one retry. When both replies are visibly broken, the
    one that names fewer of the request's records in two sub-tasks is used (``_breakage``).
    The model's split then goes through ``split_merge.merge_subtasks`` (code only), which joins parts that are not independent jobs and returns the
    request whole when one job is left.

    ``workflow`` (the app workflow, optional): a split over the cap labels its parts with the plan the intent loop gives
    each (``plan_label``) so parts of one plan can be joined; without it nothing is labelled.
    """
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER")
    job_types = _job_types_for(workflow_folderpath)
    # The one-job gate: a confident "one" from the remote decision model is the request as one part, as a split the
    # model left whole returns it, and the splitter is not called (``state["attempts"]`` stays unset: 0 calls).
    gate = _one_job_gate(user_query, job_types)
    if gate is not None:
        state["gate"] = gate
        if gate["skipped"]:
            return [user_query]
    # Temperature 0 makes the first attempt less random, not repeatable: the same request can still split
    # differently from one run to the next (the provider is not deterministic at 0). A split that is
    # visibly broken or cut off at the cap is retried once at 0.7, which also misses the LM cache.
    # A split whose only flaw is a leftover reference ("each of them") is still
    # better than none, so it is kept when the retry does not fix it; when both
    # attempts are like that, the less broken one (``_breakage``; the first on a tie).
    # A model that refuses a temperature (Bedrock Claude Haiku 5.5 takes only 1) is called again
    # without it (``dspy_utils.call_dropping_rejected_params``); its retry skips the cache (``_split_lm``).
    usable: list[list[str]] = []
    raws: dict[tuple, list[str]] = {}
    rules: list[str] = state.setdefault("rules", [])
    cut_off = False
    for attempt, temperature in enumerate((0.0, 0.7)):
        state["attempts"] = attempt + 1
        replies: list = []
        # A reply that stopped at the output cap is the model looping in its free reasoning text: the one retry then
        # asks for the sub-tasks without the reasoning field (``dspy.Predict``), the same signature and instructions.
        reasoned = not cut_off

        def split(lm) -> str:
            with dspy_utils.role_context("LLM_PLANNER", lm, adapter=CommandsSystemPreludeAdapter()):
                signature = _split_signature(workflow_folderpath) if workflow_folderpath else SplitIntoSubtasks
                state["prompt"] = split_prompt(workflow_folderpath)
                inputs = {"job_types": job_types, "user_query": user_query}
                if conversation:
                    signature = signature.insert(
                        1, "conversation", dspy.InputField(desc=SPLIT_CONVERSATION_DESC), type_=str)
                    inputs["conversation"] = conversation
                module = dspy.ChainOfThought(signature) if reasoned else dspy.Predict(signature)
                demos = split_demos(workflow_folderpath, job_types)
                for predictor in module.predictors() if demos else ():
                    predictor.demos = demos
                return module(**inputs).subtasks or ""

        # One span per model attempt, in the split stage (bd fix-qjne.34): its model call is the span's child.
        host = tracing.current_host()
        splitter = tracing.start_span(host, tracing.SPAN_SPLIT_SPLITTER, attributes={"attempt": attempt + 1})
        try:
            with watch_cut_off_replies() as replies:
                text = dspy_utils.call_dropping_rejected_params(
                    _split_lm(planner_lm, temperature, retry=attempt > 0), split)
        except Exception as error:  # noqa: BLE001 - splitting is an optimisation; never fail the turn
            if any(replies):
                logger.warning(f"the split was cut off at {split_max_tokens()} tokens and did not parse")
                tracing.end_span(host, splitter, attributes={"raw": None, "failed": SPLIT_FAILED_CUT_OFF})
                cut_off = True
                continue
            tracing.end_span(host, splitter, status=tracing.STATUS_ERROR,
                             attributes={"raw": None, "failed": SPLIT_FAILED_ERROR})
            _log_split_failure(planner_lm, error)
            state["failed"] = SPLIT_FAILED_ERROR
            return [user_query]
        if replies and replies[-1]:
            # The reply that parsed stopped at the cap: whatever parsed is not the model's answer. The parsed reply is
            # always the call's last one, so a complete fallback reply (the library's JSON retry of a cut-off first
            # reply, inside the same call) is the answer and is used.
            logger.warning(f"the split was cut off at {split_max_tokens()} tokens; not using it")
            tracing.end_span(host, splitter, attributes={"raw": None, "failed": SPLIT_FAILED_CUT_OFF})
            cut_off = True
            continue
        cut_off = False
        subtasks = [without_repeated_clauses(item.translate(_HYPHENS)) for item in _split_items(text)]
        state["raw"] = list(subtasks)
        tracing.end_span(host, splitter, attributes={"raw": list(subtasks), "failed": None})
        # Disabled after benchmark pass 4 (fix-kdag never fired in 840 runs):
        # if not subtasks and text.strip():
        #     # A non-empty sub-tasks field with no numbered item ("it") is a visibly broken reply: retried once.
        #     logger.warning("the split listed no numbered sub-task; retrying once")
        #     continue
        if len(subtasks) > MAX_SUBTASKS:
            # The number of items is never a reason to run the request whole: the list is brought down to the cap
            # (copies dropped, then parts joined). Only a split that listed nothing, or a reduction that failed, is whole.
            reduction = _reduced_to_cap(user_query, subtasks, workflow_folderpath, conversation, workflow)
            if reduction is None:
                return [user_query]
            rules.append("reduced_to_cap")
            if reduction.merged:
                return _whole_if_placeholder(
                    user_query, _with_referents(user_query, reduction.items, workflow_folderpath), rules)
            subtasks = reduction.items
        if not subtasks:
            return [user_query]
        if not _visibly_broken(subtasks, user_query):
            return _whole_if_placeholder(
                user_query, _merged(user_query, subtasks, workflow_folderpath, conversation, rules, state.setdefault("doubts", [])), rules)
        # A sub-task the model listed twice is one job: the list without its copies is a candidate too.
        unique = list(dict.fromkeys(subtasks))
        if len(unique) >= 2:
            usable.append(unique)
            raws[tuple(unique)] = list(state["raw"])
    if not usable:
        if cut_off:
            # every answer stopped at the cap: the request runs as one part, and the turn says so (span and log)
            state["failed"] = SPLIT_FAILED_CUT_OFF
            logger.warning(f"the split failed: every answer stopped at {split_max_tokens()} tokens; the request runs "
                           "as one part")
        return [user_query]
    best = min(usable, key=lambda subtasks: _breakage(user_query, subtasks, workflow_folderpath))
    state["raw"] = raws[tuple(best)]
    state["doubts"] = []   # only the doubts of the split that is kept
    return _whole_if_placeholder(user_query, _merged(user_query, best, workflow_folderpath, conversation, rules, state.setdefault("doubts", [])), rules)


def _with_referents(user_query: str, items: list[str], workflow_folderpath: str | None) -> list[str]:
    """``resolve_referents`` on the sub-tasks that are not joined. A joined one (``JoinedSubtask``) keeps its text and parts."""
    if len(items) < 2:
        return items
    plain = [item for item in items if not isinstance(item, JoinedSubtask)]
    resolved = iter(resolve_referents(user_query, plain, workflow_folderpath))
    return [item if isinstance(item, JoinedSubtask) else next(resolved) for item in items]


def plan_label(workflow, text: str) -> str | None:
    """The planning command *text* means, or None: the intent loop's decision for the part (``resolution.part_intent``,
    memoised for the turn, so the resolution that follows does not decide the same text again). Only a match whose kind
    is planning counts; an ambiguity or no match is "no plan" here. Never asks a question."""
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    intent = resolution.part_intent(workflow, text)
    if intent.outcome == resolution.MATCH and intent.kind == resolution.KIND_PLANNING:
        return intent.command
    return None


def _reduced_to_cap(user_query: str, subtasks: list[str], workflow_folderpath: str | None,
                    conversation: str, workflow=None) -> Reduction | None:
    """``split_merge.reduce_to_cap`` on a split of more than MAX_SUBTASKS items, with Jev's checks when there is
    a key (near copies, same request) and, given the app *workflow*, the part's plan (``plan_label``); None if it fails.

    Jev calls: at most ``REDUCTION_JEV_CALLS`` here, plus at most two per remaining sub-task in ``resolve_referents``.
    """
    try:
        identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()

        def label(text: str) -> str | None:
            return plan_label(workflow, text)

        reduction = reduce_to_cap(
            user_query, subtasks, MAX_SUBTASKS, identifiers, context=conversation,
            same_job=_same_request_check(user_query),
            near_copy=_same_request_check(user_query, NEAR_COPY_THRESHOLD),
            label=label if workflow is not None else None)
        logger.info(f"the split listed {len(subtasks)} sub-tasks, over {MAX_SUBTASKS}: reduced to "
                    f"{len(reduction.items)} ({reduction.jev_calls} Jev calls)")
        return reduction
    except Exception:  # noqa: BLE001 - never fail the turn
        logger.warning("could not reduce a split over the cap", exc_info=True)
        return None


def _merged(user_query: str, subtasks: list[str], workflow_folderpath: str | None = None,
            conversation: str = "", rules: list[str] | None = None,
            doubts: list[dict] | None = None) -> list[str]:
    """``split_merge.merge_subtasks`` on the model's split; the split unchanged if the merge fails. *doubts* collects
    the rewrites the merge did not make because they would have dropped a name the part carried (``merge_subtasks``).

    With a Jev key, rule 3's joins (a sub-task naming no subject, or the same subjects as the one before it)
    are Jev's call (``_same_request_check``); without one, the lexical rule joins them as before.
    One sub-task is one job, returned as the request the user wrote: the model's rewrite of a single job can
    drop the words that qualify it (a leading record phrase), and nothing else restores them.
    """
    if sum(1 for item in subtasks if item and item.strip()) < 2:
        return [user_query]
    applied: list[str] = []
    # The merge's own span, in the split stage (bd fix-qjne.34): its model calls (rule 3's checks) are its children.
    host = tracing.current_host()
    span = tracing.start_span(host, tracing.SPAN_SPLIT_MERGE)
    try:
        identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()
        merged = merge_subtasks(user_query, subtasks, identifiers, same_job=_same_request_check(user_query),
                                context=conversation, applied=applied, doubts=doubts)
    except Exception:  # noqa: BLE001 - the merge is a clean-up; never fail the turn
        logger.warning("could not merge the split; keeping it as the model gave it", exc_info=True)
        tracing.end_span(host, span, status=tracing.STATUS_ERROR,
                         attributes={"merged": None, "rules": list(applied) or None, "doubts": None})
        return subtasks
    tracing.end_span(host, span, attributes={"merged": [str(text) for text in merged], "rules": list(applied) or None,
                                             "doubts": list(doubts) if doubts else None})
    if rules is not None:
        rules.extend(applied)
    return resolve_referents(user_query, merged, workflow_folderpath) if len(merged) > 1 else merged


#: Rule 3 of the merge joins two adjacent sub-tasks (no subject, or the same subjects) only when Jev says they are
#: one request at or above this probability; below it they stay two jobs. Calibrated on the fix-bydz diagnosis
#: requests (library example and planning tutorial, Nemotron splits): Jev gave independent jobs 0.00-0.01 and steps
#: of one job 0.27-0.99, so the bar is low and only a confident "separate" keeps two parts apart.
SAME_REQUEST_THRESHOLD = 0.2

#: A near copy (over the cap) is dropped only when Jev says it is one request with at least this probability.
NEAR_COPY_THRESHOLD = 0.9

_SAME_REQUEST_OPTIONS = (
    ("separate", "Two separate requests: each part asks for something of its own and gets its own answer, "
                 "and either could be asked alone."),
    ("one", "One request: the later part is a step of the earlier one; it continues it, or acts on or "
            "narrows what the earlier part finds (find a record, then open it)."),
)


class SameRequest(dspy.Signature):
    """Decide whether two parts cut from a user's request are separate requests or steps of one request."""

    request: str = dspy.InputField(desc="the user's whole request")
    earlier: str = dspy.InputField(desc="a part cut from the request")
    later: str = dspy.InputField(desc="the part cut from the request right after it")
    relation: Choice[_SAME_REQUEST_OPTIONS] = dspy.OutputField(
        desc="Are the two parts separate requests, each needing its own answer, or one request whose "
             "later part is a step of the earlier one?")


def _same_request_check(user_query: str, threshold: float = SAME_REQUEST_THRESHOLD):
    """The merge's rule 3 decision (``split_merge.SameJob``) on Jev, or None without a Jev key.

    ``(earlier, later) -> True`` when Jev says the two parts are one request with probability at least
    ``SAME_REQUEST_THRESHOLD``, False when it says otherwise, None (the lexical rule) when the call fails.
    """
    jev_lm = _jev_lm()
    if jev_lm is None:
        return None

    def same_job(earlier: str, later: str) -> bool | None:
        try:
            with dspy_utils.role_context("JEV", jev_lm):
                choice = dspy.Predict(SameRequest)(request=user_query, earlier=earlier, later=later).relation
            value, probability = _choice_value_and_probability(choice)
            one = probability if value == "one" else float((choice.probabilities or {}).get("one", 1 - probability))
        except Exception:  # noqa: BLE001 - the lexical rule decides; never fail the turn
            logger.warning("could not ask whether two sub-tasks are one request", exc_info=True)
            return None
        logger.info(f"same request? {earlier!r} / {later!r}: {value} (one: {one:.2f})")
        return one >= threshold

    return same_job


#: A sub-task's referent is written in only at or above this Jev probability.
REFERENT_THRESHOLD = 0.5


def resolve_referents(user_query: str, subtasks: list[str], workflow_folderpath: str | None) -> list[str]:
    """Each sub-task with the request's record it acts on but does not name written in (one Jev call each).

    A cut sub-task can lose the record its words point back to: "Tag the customer `x`" after "Ann Lee
    leaves", "Confirm whether Raj Patel is among its holders". For a sub-task that leaves some of
    the request's subjects unnamed, Jev chooses which of them, if any, it is about
    (``split_merge.unnamed_subjects``), told what the workflow counts as a record by its optional
    ``subject_hints`` in ``_commands/_workflow.py``; ``split_merge.with_referent`` writes the choice in. Without a Jev
    key, or on any failure, the sub-tasks are returned unchanged.

    When a record is chosen and the sub-task holds definite descriptions ("the customer", see
    ``split_merge.definite_descriptions``), a second Jev call chooses which of them, if any, stands
    for it, and the record is written in right after it ("Tag the customer Ann Lee `x`"). Code cannot
    tell "the customer `x`" (a person, `x` a tag put on her) from "the report 'x'" (`x` its title)
    without knowing the workflow's kinds of record; the model, given the hints, can.
    """
    jev_lm = _jev_lm()
    if jev_lm is None:
        return subtasks
    turn_progress.emit(turn_progress.STAGE_REFERENTS)
    identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()
    notes = (split_settings(workflow_folderpath).subject_hints if workflow_folderpath else None) or "(none)"

    def resolved(number: int, text: str) -> str:
        # One span per sub-task, in the split stage (bd fix-qjne.34): its model calls are the span's children.
        host = tracing.current_host()
        span = tracing.start_span(host, tracing.SPAN_SPLIT_REFERENT, attributes={"part": number})
        subject = description = None
        try:
            candidates = unnamed_subjects(user_query, text, identifiers)
            subject = _choose_referent(user_query, text, candidates, notes, jev_lm) if candidates else None
            descriptions = definite_descriptions(text) if subject else []
            description = _choose_description(user_query, text, subject, descriptions, notes, jev_lm) \
                if descriptions else None
            return with_referent(text, subject, description) if subject else text
        except Exception:  # noqa: BLE001 - resolving a referent is a clean-up; never fail the turn
            logger.warning("could not resolve the sub-task's referent", exc_info=True)
            return text
        finally:
            tracing.end_span(host, span, attributes={"subject": subject, "description": description})

    # Each sub-task is resolved from the request and its own original text, never from another's resolved text, so the
    # calls are independent: they wait on Jev together (subtask_pool), and the results keep the sub-tasks' order.
    return subtask_pool.map_in_order([functools.partial(resolved, number, text)
                                      for number, text in enumerate(subtasks, 1)])


def _choose_referent(user_query: str, text: str, candidates: list[str], notes: str, jev_lm) -> str | None:
    options = tuple((name, f"The sub-task acts on or asks about {name} without naming it.") for name in candidates) + (
        ("none", "The sub-task names every record it is about, or is about none of these."),)

    class ChooseReferent(dspy.Signature):
        """Decide which record of the request a sub-task is about without naming it."""

        request: str = dspy.InputField(desc="the user's whole request")
        subtask: str = dspy.InputField(desc="one sub-task cut from the request, run on its own")
        notes: str = dspy.InputField(desc="what this workflow counts as a record, and what it does not")
        record: Choice[options] = dspy.OutputField(
            desc="Which record named in the request, but not in the sub-task, is the sub-task about "
                 "(through words like 'it', 'its', 'that team', 'the customer')? 'none' if it names "
                 "every record it is about.")

    with dspy_utils.role_context("JEV", jev_lm):
        choice = dspy.Predict(ChooseReferent)(request=user_query, subtask=text, notes=notes).record
    value, probability = _choice_value_and_probability(choice)
    logger.info(f"referent for {text!r}: {value} (confidence {probability:.2f})")
    return value if value in candidates and probability >= REFERENT_THRESHOLD else None


def _choose_description(user_query: str, text: str, subject: str, descriptions: list[str], notes: str,
                        jev_lm) -> str | None:
    """Which of the sub-task's definite descriptions ("the customer") stands for ``subject``, if any."""
    options = tuple((phrase, f"'{phrase}' in the sub-task means {subject}.") for phrase in descriptions) + (
        ("none", f"None of these phrases means {subject}; each means something else."),)

    class ChooseDescription(dspy.Signature):
        """Decide which phrase of a sub-task stands for a record the sub-task does not name."""

        request: str = dspy.InputField(desc="the user's whole request")
        subtask: str = dspy.InputField(desc="one sub-task cut from the request, run on its own")
        record: str = dspy.InputField(desc="the record of the request the sub-task is about without naming it")
        notes: str = dspy.InputField(desc="what this workflow counts as a record, and what it does not")
        phrase: Choice[options] = dspy.OutputField(
            desc="Which 'the ...' phrase of the sub-task means the record (as 'the customer' can mean "
                 "a customer named earlier)? 'none' if each phrase means something else, such as a tag, a "
                 "request, another record or a record the work is to find.")

    with dspy_utils.role_context("JEV", jev_lm):
        choice = dspy.Predict(ChooseDescription)(request=user_query, subtask=text, record=subject,
                                                 notes=notes).phrase
    value, probability = _choice_value_and_probability(choice)
    logger.info(f"description of {subject!r} in {text!r}: {value} (confidence {probability:.2f})")
    return value if value in descriptions and probability >= REFERENT_THRESHOLD else None


def build_query_with_next_steps(user_query: str,
    chat_session_obj: fastworkflow.ChatSession, with_agent_inputs_and_trajectory: bool = False,
    planner_lm = None,
    trace_trigger: str | None = None,
    planner_user_query: str | None = None,
    plan_template: str | None = None,
    done_only: bool = False,
    planning_command: PlanningCommandCommand | None = None,
    planning_probabilities: Mapping[str, float] | None = None,
    plan_trigger: str | None = None) -> str:
    """
    Generate a todo list.
    Return a string that combine the user query and todo list

    Args:
        user_query: The user's natural language query
        chat_session_obj: The active chat session
        with_agent_inputs_and_trajectory: Whether to include agent trajectory for replanning
        planner_lm: Optional planner LM to use (if None, uses LLM_PLANNER from env)
        trace_trigger: What re-triggered planning mid-turn (e.g.
            "ask_user_response", "parameter_extraction_error"). None means the
            part's first plan; set, it is the fw.plan span's ``trigger``.
        planner_user_query: Current-turn text for the planner LLM only. When prior
            turns are supplied via ``previous_turn_summaries``, pass the raw user
            message here so ``user_query`` (often a refined string that already
            embeds recent summaries) is not duplicated in the planner prompt.
        plan_template: Optional planning command steps for the planner only, appended to the
            request in the planner prompt (not in the returned string). Used on
            the turn's initial plan, not on a replan.
        done_only: With a trajectory, return only the "what's done" summary.
        planning_command: The matched planning command's parsed form. On the turn's initial plan the
            planner then fills the planning command's typed parameters and code expands its steps
            (``planning_commands.typed_planning_call``; span ``plan_source`` "skill_call"; a planning command
            without parameters makes no planner call); Jev, when configured, checks the
            planning command's classifier-checked parameters against the request. The call's phases are then
            run by the execution context (``_phased_planning_command_call``). Any failure falls back to
            the text planner with ``plan_template``, except a refusal of the arguments
            (PlanningCommandInputError) on a turn that may ask (``_planning_command_refusal_may_ask``): its question
            is left on ``chat_session_obj._planning_command_refusal``, no plan is made, and the execution
            context suspends the turn on it. Ignored on a replan.
        plan_trigger: Why a first plan is made, for its span only (``refused``: a bound planning command was
            refused and the agent takes over; ``handoff``: a run that hit the step cap is continued).
    """
    current_workflow = chat_session_obj.get_active_workflow()
    base_docstring = PLANNER_INSTRUCTIONS
    # The plain-text planner.
    class TaskPlannerTextSignature(dspy.Signature):
        __doc__ = base_docstring
        user_query: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    class TaskPlannerTextWithTrajectoryAndAgentInputsSignature(dspy.Signature):
        __doc__ = base_docstring
        agent_inputs: dict = dspy.InputField()
        agent_trajectory: dict = dspy.InputField()
        user_response: str = dspy.InputField(desc="the user's latest response, empty for a progress check")
        whats_done: str = dspy.OutputField(desc="concise summary of what the agent_trajectory has accomplished so far, including key values found")
        next_steps: str = dspy.OutputField(desc="the remaining tasks of the original plan, revised for what is done, as a numbered list of short sentences separated by line breaks")

    class TaskPlannerTextWithPreviousTurnSummariesSignature(dspy.Signature):
        __doc__ = base_docstring
        user_query: str = dspy.InputField()
        previous_turn_summaries: str = dspy.InputField(
            desc="Summaries of every completed turn before this one, oldest first"
        )
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    available_commands = planner_listing_text(current_workflow, user_query,
                                              planning_probabilities=planning_probabilities)

    # Use provided planner_lm if available (distillation mode), else build from env
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER")
    agent_adapter = CommandsSystemPreludeAdapter()

    # One fw.plan per plan, a child of the part it plans for (bd fix-qjne.34): `trigger` is what planned again mid-turn
    # (trace_trigger), else the caller's reason for a first plan (*plan_trigger*: refused, handoff), else initial. The
    # steps that follow carry its id (``tracing.note_plan``).
    span = tracing.start_span(
        chat_session_obj,
        tracing.SPAN_PLAN,
        kind=tracing.KIND_LLM,
        attributes={
            "model": getattr(planner_lm, "model", None),
            "trigger": trace_trigger or plan_trigger or "initial",
        },
    )

    def with_template(text: str) -> str:
        return f"{text}\n\n{plan_template}" if plan_template else text

    initial_plan = trace_trigger is None and not with_agent_inputs_and_trajectory
    previous_turn_summaries = _previous_turn_summaries_for_planner(chat_session_obj) if initial_plan else ""
    request_text = (planner_user_query if previous_turn_summaries and planner_user_query is not None
                    else user_query)

    def plan_with():
        if with_agent_inputs_and_trajectory:
            workflow_tool_agent = chat_session_obj.workflow_tool_agent
            task_planner_func = dspy.ChainOfThought(TaskPlannerTextWithTrajectoryAndAgentInputsSignature)
            agent_inputs, agent_trajectory = workflow_tool_agent.planner_view()
            cleaned_agent_inputs = {k: v for k, v in agent_inputs.items() if k != "available_commands"}
            return task_planner_func(
                agent_inputs = cleaned_agent_inputs,
                agent_trajectory = agent_trajectory,
                user_response = user_query,
                available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter
        if previous_turn_summaries:
            task_planner_func = dspy.ChainOfThought(
                TaskPlannerTextWithPreviousTurnSummariesSignature
            )
            return task_planner_func(
                user_query=with_template(request_text),
                previous_turn_summaries=previous_turn_summaries,
                available_commands=available_commands,
            )
        task_planner_func = dspy.ChainOfThought(TaskPlannerTextSignature)
        return task_planner_func(
            user_query=with_template(user_query),
            available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter

    plan_text = ""
    plan_source = None
    planning_command_args = None
    planning_command_notes = None
    planning_command_not_in_request = None
    planning_command_fallback = None
    prediction = None
    # The question to put to the user when the planning command call is refused and the turn may ask (see
    # _planning_command_refusal_may_ask); None when the text planner takes over as before.
    refusal_ask = None
    if initial_plan:
        # A phased planning command call (planning_phases): the runner reads it after this plan.
        chat_session_obj._phased_planning_command_call = None
        # A refused planning command call the turn asks the user about (set below; read by the execution context).
        chat_session_obj._planning_command_refusal = None
    try:
        with dspy_utils.role_context("LLM_PLANNER", planner_lm, adapter=agent_adapter):
            if planning_command is not None and initial_plan:
                try:
                    # The typed planning call gets the trimmed listing (design 3.4); the text planner keeps the full one.
                    call = typed_planning_call(planning_command, request_text, typed_planner_commands_text(current_workflow),
                                           check_lm=_jev_lm(), workflow=current_workflow)
                    prediction, plan_text, planning_command_args = call.prediction, call.plan, call.args
                    planning_command_notes = list(call.notes) or None
                    planning_command_not_in_request = list(call.not_in_request) or None
                    plan_source = "skill_call"
                    chat_session_obj._phased_planning_command_call = (planning_command, call)
                    logger.info(f"planning command call {planning_command.name}{call.args}"
                                + (f" ({'; '.join(call.notes)})" if call.notes else ""))
                except PlanningCommandInputError as exc:  # asked of the user, or the text planner takes over
                    planning_command_fallback = f"{type(exc).__name__}: {exc}"[:500]
                    refusal_ask = planning_command_refusal_to_ask(chat_session_obj, planning_command, exc, request_text)
                    if refusal_ask is not None:
                        logger.info(f"planning command call {planning_command.name} refused, asking the user: {refusal_ask!r}")
                    else:
                        logger.info(f"planning command call {planning_command.name} failed, planning from its template: {planning_command_fallback}")
                except Exception as exc:  # noqa: BLE001 - the text planner with the template takes over
                    planning_command_fallback = f"{type(exc).__name__}: {exc}"[:500]
                    logger.info(f"planning command call {planning_command.name} failed, planning from its template: {planning_command_fallback}")
            if plan_source is None and refusal_ask is None:
                prediction = plan_with()
                plan_text = prediction.next_steps or ""
                plan_source = "text" if plan_text else "none"
    except BaseException:
        tracing.end_span(chat_session_obj, span, status=tracing.STATUS_ERROR)
        raise
    if refusal_ask is not None:
        # No plan for the refused call: the turn waits on the user, and the execution context
        # reads the question and suspends (_call_agent_for_query).
        chat_session_obj._planning_command_refusal = refusal_ask
        plan_source = "none"
    tracing.end_span(
        chat_session_obj,
        span,
        attributes={
            "plan": plan_text,
            "plan_source": plan_source,
            "command": planning_command.name if planning_command is not None and initial_plan else None,
            "command_args": planning_command_args,
            "command_notes": planning_command_notes,
            "command_fallback": planning_command_fallback,
            "command_refusal_asked": True if refusal_ask is not None else None,
            "argument_check_not_in_request": planning_command_not_in_request,
        },
    )
    tracing.note_plan(chat_session_obj, span)

    if not with_agent_inputs_and_trajectory:
        # The run's initial plan, with its line breaks, for the finish check.
        chat_session_obj._initial_plan_text = plan_text

    # A mid-turn replan with nothing remaining still reports what's done and says to finish.
    if not plan_text and not with_agent_inputs_and_trajectory:
        return user_query

    generated_plan = plan_text.split()
    # Capture the generated plan for distillation when a capture list is present
    # on the session (set only during a DistillationSession planning pass).
    planning_capture = getattr(chat_session_obj, '_planning_steps_capture', None)
    if planning_capture is not None:
        from fastworkflow.evaluation.distillation import PlanningStep
        planning_capture.append(PlanningStep(
            step_number=len(planning_capture),
            user_query=user_query,
            generated_plan=generated_plan,
            reasoning=getattr(prediction, 'reasoning', ''),
        ))

    steps_formatted = " ".join(generated_plan)
    if with_agent_inputs_and_trajectory:
        whats_done = getattr(prediction, "whats_done", "") or ""
        if done_only:
            return whats_done
        sections = [f"User Query:\n{user_query}"] if user_query else []
        sections.append(f"What's done:\n{whats_done}")
        sections.append(
            f"What remains (execute these next steps):\n{steps_formatted}"
            if steps_formatted else
            "Nothing left to do. Call finish now."
        )
        return "\n\n".join(sections)
    return f"{user_query}\n\nExecute these next steps:\n{steps_formatted}"
