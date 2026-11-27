"""The ``@writes`` declaration: a command that changes the application's data says so (fix-fixv).

An execution command's ``ResponseGenerator`` carries the decorator::

    @writes    # running it changes the application's data (adds a note, cancels an order)

Absent means the command only reads, or only moves the workflow between records. A planning command
cannot carry it (a load error): it writes when one of its steps (directly, or through a planning
command it calls) runs a command that writes. A planning command whose steps reach no write is
read-only.

What the framework does with it. Inside a planning command's phase, the agent runs the steps of that
phase, and nothing else should change the application. So in a phase, a command that writes runs only
when one of the phase's steps names it:

- the phase's command listing leaves out the commands that write and that no step of the phase names
  (``unplanned_write_commands``);
- a call of such a command inside the phase is refused unrun, with a message that says why
  (``phase_write_refusal``, called by ``CommandExecutor`` before the approval gate).

Outside a planning command's phase (the turn's agent, a plain goal, a command the user typed) nothing
changes. The decorator is defined in ``fastworkflow.command_declarations``, whose ``declaration_problems`` fails the
load of a planning command that carries it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import fastworkflow
from fastworkflow.command_declarations import WRITES_ATTRIBUTE, declares_writes  # noqa: F401 - re-exported

logger = logging.getLogger(__name__)

#: The refusal the agent reads when a phase calls a write command its steps do not name.
UNPLANNED_WRITE_REFUSAL = (
    "{command} was not run: it changes the application's data, and no step of this part of the job "
    "runs it. Do only the steps given; this part of the job writes nothing that its steps do not name.")


#: The refusal for a write the phase's steps name, once it has run as many times as they name it.
REPEATED_WRITE_REFUSAL = (
    "{command} was not run: it changes the application's data, and it already ran as many times in this "
    "part of the job as its steps name it. Do not repeat it; go on with the remaining steps.")


def reached_commands(workflow_folderpath: str) -> dict[str, frozenset[str]]:
    """For each planning command of *workflow_folderpath*, the qualified names of every command its steps run,
    directly or through the planning commands they call. The one inheritance rule of the command marks: a planning
    command writes (``write_commands``) and needs an approval (``approval_annotations.command_approval``) when a
    command it reaches does. Empty when the planning commands cannot be read."""
    try:
        from fastworkflow.planning import planning_commands

        callees = planning_commands.resolve_field_sources(workflow_folderpath).callees
    except Exception:  # noqa: BLE001 - planning commands unread: nothing is known to be reached
        logger.debug("planning commands unread for %r", workflow_folderpath, exc_info=True)
        return {}
    reached: dict[str, frozenset[str]] = {}
    for name in callees:
        seen: set[str] = set()
        stack = list(callees[name])
        while stack:   # a cycle is a load problem of its own; the walk visits each command once
            called = stack.pop()
            if called not in seen:
                seen.add(called)
                stack.extend(callees.get(called, ()))
        reached[name] = frozenset(seen)
    return reached


def write_commands(workflow_folderpath: str) -> frozenset[str]:
    """The qualified names of the commands of *workflow_folderpath* that write: each execution command
    that carries ``@writes``, and each planning command whose steps reach one (``reached_commands``).
    Raises what reading the workflow's routing raises (not initialised, does not load): "the commands could not be
    read" is not "no command writes", and every caller relies on the answer (the policy's write rules, the review's
    badges and explicit yes, the look-up-only and phase guards). A folder with no ``_commands/`` has no commands, so
    none of them writes: that is known, not unknown."""
    if not os.path.isdir(os.path.join(workflow_folderpath, "_commands")):
        return frozenset()
    routing = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    writes: set[str] = set()
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if metadata.is_planning:
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class declares nothing
            continue
        if declares_writes(command_class):
            writes.add(command_name)
    if not writes:
        return frozenset()
    return frozenset(writes | {name for name, reached in reached_commands(workflow_folderpath).items()
                               if reached & writes})


def _host(chat_session: Any) -> Any:
    """The WorkflowExecutionContext behind *chat_session* (a ChatSession holds one as its core)."""
    core = getattr(chat_session, "_core", None)
    return core if core is not None else chat_session


def _plain(name: str) -> str:
    return str(name or "").strip().split("/")[-1].lower()


def _phase_state(chat_session: Any) -> Optional[dict]:
    """The state of the planning-command phase now running, or None when none runs."""
    try:
        state = getattr(_host(chat_session), "_phase_run", None)
    except Exception:  # noqa: BLE001 - a host without frames runs no phase
        return None
    return state if isinstance(state, dict) and state.get("plan") else None


def _planned_step_counts(chat_session: Any) -> Optional[dict[str, int]]:
    """How many steps of the running phase name each command (plain name), or None when no phase runs
    or its steps cannot be read (no guard)."""
    state = _phase_state(chat_session)
    if state is None:
        return None
    from fastworkflow.planning import planning_phases

    try:
        names = _host(chat_session)._phase_command_names()
        return {name: len(steps) for name, steps in planning_phases.planned_steps(state["plan"], **names).items()}
    except Exception:  # noqa: BLE001 - steps that cannot be read name nothing for sure: no guard
        logger.debug("phase steps unread for the writes guard", exc_info=True)
        return None


def phase_planned_commands(chat_session: Any) -> Optional[frozenset[str]]:
    """The plain names of the commands the steps of the planning-command phase now running name, or
    None when no planning command's phase is running (the turn's own agent, a typed command)."""
    counts = _planned_step_counts(chat_session)
    return None if counts is None else frozenset(counts)


def record_phase_write(chat_session: Any, workflow_folderpath: str, command_name: str) -> None:
    """Count one successful run of the write *command_name* in the running phase (``state["writes_run"]``,
    reset with the phase's plan). Nothing outside a phase, or for a command that does not write."""
    state = _phase_state(chat_session)
    if state is None or command_name not in write_commands(workflow_folderpath):
        return
    runs = state.setdefault("writes_run", {})
    runs[_plain(command_name)] = runs.get(_plain(command_name), 0) + 1


def unplanned_write_commands(chat_session: Any, workflow_folderpath: str) -> frozenset[str]:
    """The commands that write and that no step of the running phase names, by qualified and by short
    name (the listings read either). Empty outside a planning command's phase."""
    planned = phase_planned_commands(chat_session)
    if planned is None:
        return frozenset()
    out: set[str] = set()
    for name in write_commands(workflow_folderpath):
        if _plain(name) not in planned:
            out.update({name, name.split("/")[-1]})
    return frozenset(out)


def phase_write_refusal(chat_session: Any, workflow_folderpath: str, command_name: str) -> Optional[str]:
    """The refusal for running *command_name* now, or None to run it: a command that writes, called
    inside a planning command's phase whose steps do not name it, or that has already run as many times
    as the steps name it."""
    counts = _planned_step_counts(chat_session)
    if counts is None or command_name not in write_commands(workflow_folderpath):
        return None
    plain = _plain(command_name)
    if plain not in counts:
        return UNPLANNED_WRITE_REFUSAL.format(command=command_name)
    done = (_phase_state(chat_session) or {}).get("writes_run", {}).get(plain, 0)
    return REPEATED_WRITE_REFUSAL.format(command=command_name) if done >= counts[plain] else None
