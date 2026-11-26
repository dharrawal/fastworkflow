"""Load a workflow's policy table.

A workflow's rules live in ``<workflow>/_policy/table.py``. A workflow that has
that file has its rules enforced; one without it runs unchanged. There is no
setting: a table is workflow content, like ``_commands``.

The module exposes ``TABLE`` (a ``DecisionTable``) and optionally ``FACTS``
(``ContractFacts``). The commands that change data are not a fact the table
declares: the loader reads them from the ``@writes`` marks (one list).

A table file that exists but cannot be loaded is a HARD ERROR at startup,
deliberately: rules the developer wrote must not be skipped without a word.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
from pathlib import Path
from typing import Optional

from fastworkflow.policy.policy_decision import (
    NO_OP_TABLE,
    ContractFacts,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyError,
    PolicyMode,
)
from fastworkflow.utils.logging import logger

POLICY_FOLDER = "_policy"
TABLE_FILE = "table.py"


def workflow_table_path(workflow_folderpath: Optional[str]) -> Optional[Path]:
    """``<workflow>/_policy/table.py`` when the workflow has one, else None."""
    if not workflow_folderpath:
        return None
    path = Path(workflow_folderpath) / POLICY_FOLDER / TABLE_FILE
    return path if path.is_file() else None


def load_table_module(reference: str, workflow_folderpath: Optional[str] = None):
    """(module, attribute name) for a ``path.py[:ATTR]`` or ``module[:ATTR]``.

    A relative path is tried against the current directory, then the workflow
    folder. Used for the workflow's own table and for a table given explicitly
    (``fastworkflow policy replay --table``).
    """
    module_ref, _, attribute = reference.strip().partition(":")
    attribute = attribute or "TABLE"
    if module_ref.endswith(".py"):
        path = Path(module_ref)
        if not path.is_absolute() and not path.is_file() and workflow_folderpath:
            path = Path(workflow_folderpath) / module_ref
        if not path.is_file():
            raise ValueError(
                f"Policy table {module_ref} does not exist"
                + (f" (also tried under {workflow_folderpath})" if workflow_folderpath else "")
            )
        spec = importlib.util.spec_from_file_location(f"fw_policy_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return module, attribute


def load_policy(
    workflow_folderpath: Optional[str] = None,
    *,
    reference: Optional[str] = None,
) -> tuple[PolicyDecisionPoint, ContractFacts]:
    """The decision point and the workflow's facts.

    Without ``reference``, the workflow's own ``_policy/table.py``; a workflow
    without one gets the no-op decision point. ``reference`` names another table
    (the replay audit, tests) and must exist.
    """
    if reference is None:
        path = workflow_table_path(workflow_folderpath)
        if path is None:
            return PolicyDecisionPoint(NO_OP_TABLE, PolicyMode.OFF), ContractFacts()
        reference = str(path)
    module, attribute = load_table_module(reference, workflow_folderpath)
    table = getattr(module, attribute, None)
    if not isinstance(table, DecisionTable):
        raise ValueError(
            f"Policy table {reference}: {attribute} is "
            f"{type(table).__name__}, not a DecisionTable"
        )
    facts = getattr(module, "FACTS", None) or ContractFacts()
    if not isinstance(facts, ContractFacts):
        raise ValueError(f"Policy table {reference}: FACTS is not ContractFacts")
    if facts.write_commands:
        raise PolicyError(
            f"Policy table {reference}: FACTS.write_commands is removed: mark each command that changes data "
            "with @fastworkflow.writes on its ResponseGenerator; the policy reads the marks "
            f"(listed here: {', '.join(sorted(facts.write_commands))})"
        )
    facts = with_workflow_writes(with_workflow_identifiers(facts, workflow_folderpath), workflow_folderpath)
    logger.info(
        f"Policy: table {table.version} ({len(table.rows)} rows) from {reference}"
    )
    return PolicyDecisionPoint(table, PolicyMode.ENFORCE), facts


def with_workflow_identifiers(
    facts: ContractFacts, workflow_folderpath: Optional[str]
) -> ContractFacts:
    """``facts`` with ``id_patterns`` taken from the workflow when it gives none.

    A workflow may already declare its identifier formats for the split merge
    (``identifier_patterns`` in ``_commands/_workflow.py``); a table need not repeat them.
    """
    if facts.id_patterns or not workflow_folderpath:
        return facts
    try:
        from fastworkflow.workflow_agent import identifier_patterns

        patterns = tuple(identifier_patterns(workflow_folderpath))   # compiled: their flags are kept
    except Exception as exc:  # noqa: BLE001 - the workflow settings file reports its own errors
        logger.warning(f"Policy: workflow identifier patterns unavailable: {exc}")
        return facts
    return dataclasses.replace(facts, id_patterns=patterns) if patterns else facts


def with_workflow_writes(
    facts: ContractFacts, workflow_folderpath: Optional[str]
) -> ContractFacts:
    """``facts`` with ``write_commands`` read from the workflow's ``@writes`` marks
    (``command_writes.write_commands``: the marked execution commands and the
    planning commands whose steps reach one), by qualified and by leaf name, as
    the agent may call either. A workflow whose commands cannot be read is a ``PolicyError``: with no
    commands known to write, the table's write rules would silently never fire."""
    if not workflow_folderpath:
        return facts
    from fastworkflow.command_writes import write_commands

    try:
        names = write_commands(workflow_folderpath)
    except Exception as exc:  # noqa: BLE001 - any failure to read the commands is the same error here
        raise PolicyError(
            f"Policy: the commands of {workflow_folderpath} could not be read ({type(exc).__name__}: {exc}), "
            "so the commands that change data are unknown and the table's write rules would never fire; "
            "load the policy after fastworkflow.init, on a workflow whose commands load"
        ) from exc
    return dataclasses.replace(
        facts, write_commands=frozenset(names) | {name.split("/")[-1] for name in names}
    )


def attach_policy(agent, host) -> None:
    """Give a ReAct agent its workflow's policy. A no-op without ``_policy/table.py``.

    ``host`` is the WorkflowExecutionContext the agent serves; the table is
    resolved against its app workflow's folder.
    """
    from fastworkflow.policy.policy_runtime import AgentPolicy

    workflow = getattr(host, "_app_workflow", None)
    point, facts = load_policy(getattr(workflow, "folderpath", None))
    if point.enabled:
        agent.policy = AgentPolicy(point, facts, host=host)
