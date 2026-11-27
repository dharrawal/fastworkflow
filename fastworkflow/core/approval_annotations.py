"""The approval annotation: a decorator on a command class that marks a command as needing the user's say.

An execution command's ``ResponseGenerator`` may declare::

    @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)            # asked every run, never delegated
    @user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)   # the user may delegate (own command, or "always")

Absent means the agent decides, with no gate. A planning command cannot carry it (a load error, as for ``@writes``):
it needs what the commands its steps run need, directly or through the planning commands they call
(``command_writes.reached_commands``, the one inheritance rule of both marks), and each of those calls is asked about
when it runs. The decorator and the enum live in
``fastworkflow.core.command_declarations``. This module VALIDATES and EXPOSES the annotation: it is read from
the class, checked when the routing definition loads, stored on the command's metadata (so it lands in
``command_directory.json`` as the enum's value), and rendered as a short marker in the planner and
executor listings. Nothing here enforces anything; the approval gate in ``approval_binding`` reads it.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import fastworkflow
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.core.command_declarations import UserApproval, declared_approval

logger = logging.getLogger(__name__)

#: The stored values (``CommandMetadata.approval``, ``command_directory.json``): the enum's values.
REQUIRED_ALWAYS = UserApproval.REQUIRED_ALWAYS.value
REQUIRED_UNTIL_DELEGATED = UserApproval.REQUIRED_UNTIL_DELEGATED.value
APPROVAL_VALUES = tuple(member.value for member in UserApproval)

#: The short markers the listings append to a command's line. Unannotated commands get none.
APPROVAL_MARKERS = {
    REQUIRED_ALWAYS: " (needs your approval before it runs)",
    REQUIRED_UNTIL_DELEGATED: " (may need your approval)",
}


class ApprovalAnnotationError(ValueError):
    """A command's approval declaration is not a ``UserApproval`` member (the decorator prevents this)."""


def check_approval_annotations(workflow_folderpath: str, routing) -> None:
    """Snapshot each execution command's ``@user_approval_for_execution`` value onto its metadata.

    The decorator itself rejects a non-``UserApproval`` argument at import, so a value that is not an
    enum member here is an internal error and raises :class:`ApprovalAnnotationError`. Retired spellings
    and hand-written private attributes fail the command's load (``command_declarations.declaration_problems``).
    Core commands carry no annotation and are skipped. A command whose class cannot be read is left
    unannotated here (its load error is raised elsewhere). The metadata is the snapshot: when an
    annotation changes, the command directory is re-saved so ``command_directory.json`` carries it.
    """
    problems: list[str] = []
    changed = False
    core = set(routing.command_directory.core_command_names)
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if command_name in core or metadata.is_planning:
            # a planning command's approval is derived (``command_approval``); the decorator on it fails its load
            # (``command_declarations.declaration_problems``)
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class is reported by its own load path
            logger.debug("approval declaration unread for %r", command_name, exc_info=True)
            command_class = None
        declared = declared_approval(command_class)
        if declared is None and command_class is not None and "__fw_approval__" in vars(command_class):
            problems.append(f"{_source_label(metadata, workflow_folderpath)}: {command_name}'s approval "
                            "is not a UserApproval member")
            continue
        stored = declared.value if declared is not None else None
        if metadata.approval != stored:
            metadata.approval = stored
            changed = True
    if problems:
        raise ApprovalAnnotationError("\n".join(problems))
    if changed:
        routing.command_directory.save()


def _source_label(metadata, workflow_folderpath: str) -> str:
    path = metadata.response_generation_module_path or ""
    if not path:
        return workflow_folderpath
    return os.path.relpath(os.path.abspath(path), os.path.abspath(workflow_folderpath))


def command_approval(workflow_folderpath: str, qualified_name: str) -> Optional[str]:
    """The ``UserApproval`` value a command needs (``"required_always"``, ``"required_until_delegated"``),
    or None (agent discretion).

    An execution command needs what it declares. A planning command needs the strongest value among the commands
    its steps reach (``command_writes.reached_commands``): its own call is never gated, each step's call is. None
    also when the workflow or the command cannot be read. A load error in an annotation is raised.
    """
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
    except ApprovalAnnotationError:
        raise
    except Exception:  # noqa: BLE001 - no routing definition means no annotation
        logger.debug("routing definition unread for approval of %r", qualified_name, exc_info=True)
        return None
    directory = routing.command_directory.map_command_2_metadata
    metadata = directory.get(qualified_name)
    if metadata is None:
        return None
    if not metadata.is_planning:
        return metadata.approval
    from fastworkflow.core.command_writes import reached_commands

    values = {directory[name].approval for name in reached_commands(workflow_folderpath).get(qualified_name, ())
              if name in directory}
    return next((value for value in _STRONGEST_FIRST if value in values), None)


#: The approval values, strongest first: a planning command needs the strongest one its steps reach.
_STRONGEST_FIRST = (REQUIRED_ALWAYS, REQUIRED_UNTIL_DELEGATED)


def approval_marker(workflow_folderpath: str, qualified_name: str) -> str:
    """The listing marker for a command: ``""`` when it needs no approval, else its short marker."""
    value = command_approval(workflow_folderpath, qualified_name)
    return APPROVAL_MARKERS.get(value, "") if value else ""
