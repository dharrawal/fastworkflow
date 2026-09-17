"""Oldest-first packed-trajectory compaction (Arm D)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping, Optional

from fastworkflow import context_budget
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.labels import (
    alias_line,
    estimated_tokens,
    is_offload_label,
    label_alias,
    offload_label,
    offload_saving_bytes,
    printed_alias,
    printed_context,
    replacement_saves_space,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import (
    archive,
    archived_digest,
    context_clause_of,
    default_scope,
    evict_hot_handles,
    hot_handle_max_bytes_from_env,
    hot_payload_bytes,
    mark_archived,
    mark_offloaded,
    record_event,
    remember_handle,
)

#: An execute observation is worth offloading when replacing it with its own
#: label frees at least this many UTF-8 bytes of trajectory (ido-986.14.6).
#: It replaces a 1,000-estimated-token floor (~4 KB) that asked how big the
#: observation was rather than how much residency the swap would buy: a 3 KB
#: listing page stayed resident for the whole turn while its label would have
#: cost ~400 B, and a 300 B fact could never be worth replacing at all because
#: its label is larger than it is. The label is the actual label for that step,
#: description and command text included, so the saving is the real one.
#: The value at the reference context window. The effective one is a fraction of
#: the model's window (``context_budget.OFFLOAD_MIN_SAVING``).
MIN_OFFLOAD_SAVING_BYTES = context_budget.REFERENCE_OFFLOAD_MIN_SAVING_BYTES
RECENT_OBSERVATIONS_PROTECTED = 5
#: The packed-trajectory target at the reference context window
#: (``context_budget.TRAJECTORY``).
PACKED_TARGET_BYTES = context_budget.REFERENCE_TRAJECTORY_MAX_BYTES
#: The tuning overrides. The budgets themselves come from the window.
TRAJECTORY_MAX_BYTES_ENV = context_budget.TRAJECTORY.override_env
MIN_OFFLOAD_SAVING_BYTES_ENV = context_budget.OFFLOAD_MIN_SAVING.override_env


#: The tool whose steps own the agent-visible ``O`` namespace. One name, so
#: the dispatch-side ledger and the printed alias agree on what counts.
EXECUTE_TOOL_NAME = "execute_workflow_query"

_STEP_KEY = re.compile(r"^(?:tool_name|observation)_(\d+)$")


def packed_target_bytes_from_env() -> int:
    """The packed-trajectory target for this run. See ``fastworkflow.context_budget``."""
    return context_budget.trajectory_max_bytes()


def min_offload_saving_bytes_from_env() -> int:
    """The minimum saving an offload must buy. ``0`` means "whenever the label is smaller"."""
    return context_budget.offload_min_saving_bytes()


def step_indexes(trajectory: Mapping[str, Any]) -> list[int]:
    """Every step index still present, ascending, gaps included.

    The base ReAct's context-window fallback pops the oldest step's keys, so the
    trajectory can start at step 3 or skip a step in the middle. Scanning the
    keys, rather than counting up from zero until the first miss, keeps
    compaction and replan skeletons working after such a truncation.
    """
    indexes: set[int] = set()
    for key in trajectory:
        match = _STEP_KEY.match(str(key))
        if match:
            indexes.add(int(match.group(1)))
    return sorted(indexes)


def execute_ordinals(
    trajectory: Mapping[str, Any], *, ordinal_offset: int = 0
) -> list[tuple[int, int]]:
    """``(step_index, ordinal)`` for every execute_workflow_query step present.

    ``ordinal_offset`` is the number of execute steps already truncated out of
    this trajectory, so the ``O{n}`` alias of a surviving step never shifts onto
    an alias an earlier, now-removed step already persisted under.

    This is the numbering RULE, not the authority. A live turn's authority is
    the agent's ledger (``StructuredContinuationReAct.execute_ordinal_by_step``),
    which is seeded by this function and then only ever extended; callers that
    hold an agent pass its pairs in as ``executes`` rather than recounting.
    """
    found: list[tuple[int, int]] = []
    ordinal = ordinal_offset
    for index in step_indexes(trajectory):
        if str(trajectory.get(f"tool_name_{index}") or "") == "execute_workflow_query":
            ordinal += 1
            found.append((index, ordinal))
    return found


def annotate_execute_observations(
    trajectory: dict[str, Any],
    *,
    ordinal_offset: int = 0,
    executes: Optional[list[tuple[int, int]]] = None,
    scope: Optional[RuntimeHandleScope] = None,
) -> list[dict[str, Any]]:
    """Print the canonical ``O{n}`` handle on every execute observation, in place.

    The agent only ever saw an alias on an offload label, so it guessed ReAct
    step numbers when it wanted to search a result that was still inline. This
    prints the same alias ``execute_ordinals`` assigns -- including
    ``ordinal_offset`` for execute steps truncated out of the trajectory -- on
    the observation itself, the moment the step completes.

    The line is presentation only. ``strip_alias_line`` recovers the exact
    command response for the archive and for the offload label's description,
    so stored text and its digest stay comparable with observations recorded
    before this existed.

    An alias, once printed, is never rewritten: a surviving step's ordinal
    cannot change, so a disagreement is a bug, not a renumbering. It is
    recorded as ``alias_conflict`` and the text is left exactly as it stands.

    ``ido-8ps.13``: the line also names the context the command RAN IN and, where
    the workflow declares one, that context's instance identity. The clause was
    captured at dispatch (``CommandExecutor._remember_execute_context``) and is
    read here rather than recomputed, because by now the current context may
    have moved -- a command that ENTERS a context is printed with the context it
    ran in, not with the one it entered. A step whose clause was never recorded
    (no dispatch of ours, an older recording, a capture that failed) prints the
    plain A1 line: the clause is presentation, and its absence is never guessed
    at.
    """
    if executes is None:
        executes = execute_ordinals(trajectory, ordinal_offset=ordinal_offset)
    selected_scope = scope or default_scope()
    annotated: list[dict[str, Any]] = []
    for step_index, ordinal in executes:
        key = f"observation_{step_index}"
        text = trajectory.get(key)
        if not isinstance(text, str) or is_offload_label(text):
            continue
        alias = f"O{ordinal}"
        shown = printed_alias(text)
        if shown == alias:
            continue
        if shown is not None:
            record_event(
                {
                    "kind": "alias_conflict",
                    "step_index": step_index,
                    "printed_alias": shown,
                    "expected_alias": alias,
                }
            )
            continue
        clause = context_clause_of(selected_scope, alias) or ""
        line = alias_line(alias, clause)
        trajectory[key] = line + text
        record_event(
            {
                "kind": "context_line",
                "scope_id": selected_scope.scope_id,
                "alias": alias,
                "step_index": step_index,
                "context": printed_context(line) or "",
                "context_recorded": context_clause_of(selected_scope, alias) is not None,
                "has_instance": bool(clause and " " in clause),
                "line_utf8_bytes": len(line.encode("utf-8")),
                "clause_utf8_bytes": (
                    len(line.encode("utf-8")) - len(alias_line(alias).encode("utf-8"))),
            }
        )
        annotated.append({"alias": alias, "step_index": step_index, "context": clause})
    return annotated


def archive_execute_observations(
    trajectory: Mapping[str, Any],
    *,
    executes: Optional[list[tuple[int, int]]] = None,
    ordinal_offset: int = 0,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
    hot_handle_max_bytes: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Persist every execute observation under its canonical alias, offload or not.

    Compaction only ever archived what it was about to replace, so an alias the
    agent could read inline resolved to nothing: ``search_memory`` answered
    "no matching offloaded handle" for a handle the run had just printed. Here
    the observation becomes durable as soon as the step completes, and the
    offload decision is purely a residency decision.

    The alias is the one actually printed on the observation when there is one
    (``annotate_execute_observations`` runs first), so a handle the agent can
    see is always the key it is stored under -- including the ``alias_conflict``
    case, where the printed alias stands and the recomputed one is not used.

    Stored text is the raw command response: ``strip_alias_line`` removes the
    presentation line, exactly as the offload path does, so the same alias
    written twice is the same bytes and the same digest. Writes are
    insert-or-nothing and skipped entirely once this process has written that
    digest for that alias, so revisiting a step costs nothing.

    This is an availability optimisation on the hot path of every agent step.
    A failure to persist must never lose the inline evidence or abort the turn:
    it records ``archive_refused`` and leaves the observation as it stands.
    """
    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    if hot_handle_max_bytes is None:
        hot_handle_max_bytes = hot_handle_max_bytes_from_env()
    if executes is None:
        executes = execute_ordinals(trajectory, ordinal_offset=ordinal_offset)
    archived: list[dict[str, Any]] = []
    for step_index, ordinal in executes:
        shown = trajectory.get(f"observation_{step_index}")
        if not isinstance(shown, str):
            continue
        if is_offload_label(shown):
            alias = label_alias(shown) or f"O{ordinal}"
            mark_offloaded(selected_scope, alias)
            continue
        alias = printed_alias(shown) or f"O{ordinal}"
        original = strip_alias_line(shown)
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if archived_digest(selected_scope, alias) == digest:
            continue
        args = trajectory.get(f"tool_args_{step_index}")
        command = ""
        if isinstance(args, Mapping):
            command = str(args.get("command") or "")
        try:
            store.persist(
                selected_scope,
                alias=alias,
                offload_order=ordinal,
                command_name=command,
                step_index=step_index,
                text=original,
                text_sha256=digest,
            )
        except Exception as error:  # noqa: BLE001
            record_event(
                {
                    "kind": "archive_refused",
                    "scope_id": selected_scope.scope_id,
                    "alias": alias,
                    "step_index": step_index,
                    "reason": "persistence_failed_original_retained",
                    "error": type(error).__name__,
                }
            )
            continue
        mark_archived(selected_scope, alias, text_sha256=digest, inline=True)
        remember_handle(
            selected_scope,
            {
                "alias": alias,
                "text": original,
                "text_sha256": digest,
                "command": command,
                "step_index": step_index,
                "offload_order": ordinal,
            },
        )
        evicted = evict_hot_handles(
            selected_scope, hot_handle_max_bytes=hot_handle_max_bytes
        )
        record = {
            "alias": alias,
            "step_index": step_index,
            "text_sha256": digest,
            "utf8_bytes": len(original.encode("utf-8")),
            "hot_evictions": evicted,
        }
        archived.append(record)
        record_event(
            {
                "kind": "observation_archived",
                "scope_id": selected_scope.scope_id,
                "hot_payload_bytes": hot_payload_bytes(selected_scope),
                **record,
            }
        )
    return archived


def _over_packed_target(
    text: str,
    *,
    packed_target_bytes: int,
    packed_target_tokens: Optional[int],
) -> bool:
    if packed_target_tokens is not None:
        return estimated_tokens(text) > packed_target_tokens
    return len(text.encode("utf-8")) > packed_target_bytes


def compact_trajectory(
    trajectory: dict[str, Any],
    *,
    min_offload_saving_bytes: Optional[int] = None,
    recent_observations_protected: int = RECENT_OBSERVATIONS_PROTECTED,
    packed_target_tokens: Optional[int] = None,
    packed_target_bytes: Optional[int] = None,
    hot_handle_max_bytes: Optional[int] = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
    ordinal_offset: int = 0,
    executes: Optional[list[tuple[int, int]]] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> list[dict[str, Any]]:
    """Mutate trajectory observations in place. Return offload decisions.

    An execute observation is eligible when replacing it with its own label
    frees at least ``min_offload_saving_bytes`` UTF-8 bytes
    (``MIN_OFFLOAD_SAVING_BYTES``, or ``FW_OFFLOAD_MIN_SAVING_BYTES``). The
    label is built first for exactly that reason: eligibility is a property of
    the swap, not of the observation. Everything around it is unchanged --
    oldest-first order, the five most recent execute observations protected,
    the packed target, and ``replacement_saves_space``.

    ``ordinal_offset`` counts execute steps the agent has truncated out of the
    trajectory (see ``execute_ordinals``); recency protection is measured over
    the steps still present.

    ``executes`` is the agent's own ``(step_index, ordinal)`` ledger when there
    is an agent (ido-7qd). Passing it, rather than recounting here, is what
    keeps the alias printed on an observation identical to the alias the
    command already declared and stamped under -- including after a cold
    resume, where this trajectory begins mid-turn. ``ordinal_offset`` is then
    only the seed for the fallback count, and recency protection is measured
    from the first ordinal actually present either way.
    """

    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    if min_offload_saving_bytes is None:
        min_offload_saving_bytes = min_offload_saving_bytes_from_env()
    if packed_target_tokens is None and packed_target_bytes is None:
        packed_target_bytes = packed_target_bytes_from_env()
    if hot_handle_max_bytes is None:
        hot_handle_max_bytes = hot_handle_max_bytes_from_env()
    if hot_handle_max_bytes < 0:
        raise ValueError("hot_handle_max_bytes cannot be negative")
    if executes is None:
        executes = execute_ordinals(trajectory, ordinal_offset=ordinal_offset)
    else:
        present = set(step_indexes(trajectory))
        executes = [pair for pair in executes if pair[0] in present]
    if not executes:
        return []
    # Print the handle before measuring: the packed target must be checked
    # against the trajectory the agent actually receives.
    annotate_execute_observations(
        trajectory, executes=executes, scope=selected_scope)
    # Then make every execute observation durable, whatever the offload
    # decision below turns out to be. Residency and availability are separate:
    # a handle the agent can read inline must resolve too.
    archive_execute_observations(
        trajectory,
        executes=executes,
        scope=selected_scope,
        selected_archive=store,
        hot_handle_max_bytes=hot_handle_max_bytes,
    )
    # Measured from the oldest ordinal still present, so an explicit ledger and
    # a recount protect exactly the same steps.
    protected_from = (executes[0][1] - 1) + max(
        1, len(executes) - recent_observations_protected + 1
    )
    packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
    decisions: list[dict[str, Any]] = []
    for step_index, ordinal in executes:
        key = f"observation_{step_index}"
        response = trajectory.get(key)
        if not isinstance(response, str):
            continue
        alias = f"O{ordinal}"
        # Every offload decision is taken on the exact command response, so the
        # printed handle cannot shift eligibility, savings or the stored digest.
        original = strip_alias_line(response)
        size = {
            "characters": len(original),
            "utf8_bytes": len(original.encode("utf-8")),
            "estimated_tokens": estimated_tokens(original),
        }
        recency_protected = ordinal >= protected_from
        already_label = is_offload_label(response)
        decision = {
            "alias": alias,
            "step_index": step_index,
            "action": "kept",
            "reason": "below_min_saving",
            "recency_protected": recency_protected,
            "response_size": size,
        }
        if already_label:
            decision["reason"] = "already_label"
            decisions.append(decision)
            continue
        if recency_protected:
            decision["reason"] = "recent_observation_protected"
            decisions.append(decision)
            continue
        # Eligibility is the saving, so the label has to exist before the
        # question can be asked. It is the label this step would really get --
        # same alias, same command text, same authored description -- never a
        # stand-in, or the measured saving would not be the one taken.
        args = trajectory.get(f"tool_args_{step_index}")
        command = ""
        if isinstance(args, Mapping):
            command = str(args.get("command") or "")
        label = offload_label(
            alias=alias,
            command_name=command or "execute_workflow_query",
            response=original,
            description=describe_output(command, original) if describe_output else "",
        )
        saving = offload_saving_bytes(original, label)
        decision["offload_saving_bytes"] = saving
        decision["label_size"] = {
            "characters": len(label),
            "utf8_bytes": len(label.encode("utf-8")),
        }
        if saving < min_offload_saving_bytes:
            decision["reason"] = "below_min_saving"
            decision["min_offload_saving_bytes"] = min_offload_saving_bytes
        elif not _over_packed_target(
            packed_text,
            packed_target_bytes=packed_target_bytes,
            packed_target_tokens=packed_target_tokens,
        ):
            decision["reason"] = "eligible_but_target_already_met"
        else:
            if not replacement_saves_space(original, label):
                decision["reason"] = "replacement_not_smaller"
                decisions.append(decision)
                continue
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
            packed_utf8_bytes_before = len(packed_text.encode("utf-8"))
            try:
                store.persist(
                    selected_scope,
                    alias=alias,
                    offload_order=ordinal,
                    command_name=command,
                    step_index=step_index,
                    text=original,
                    text_sha256=digest,
                )
            except Exception as error:  # noqa: BLE001
                decision["reason"] = "persistence_failed_original_retained"
                decision["persistence_error"] = type(error).__name__
                record_event(
                    {
                        "kind": "offload_refused",
                        "scope_id": selected_scope.scope_id,
                        "alias": alias,
                        "reason": decision["reason"],
                        "error": type(error).__name__,
                    }
                )
                decisions.append(decision)
                continue
            remember_handle(
                selected_scope,
                {
                    "alias": alias,
                    "text": original,
                    "text_sha256": digest,
                    "command": command,
                    "step_index": step_index,
                    "offload_order": ordinal,
                },
            )
            evicted = evict_hot_handles(
                selected_scope, hot_handle_max_bytes=hot_handle_max_bytes
            )
            trajectory[key] = label
            mark_offloaded(selected_scope, alias)
            decision["action"] = "offloaded"
            decision["reason"] = "oldest_eligible_until_target"
            decision["label"] = label
            decision["text_sha256"] = digest
            decision["persisted_before_label"] = True
            decision["hot_evictions"] = evicted
            decision["hot_payload_bytes"] = hot_payload_bytes(selected_scope)
            decision["packed_utf8_bytes_before"] = packed_utf8_bytes_before
            packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
            record_event(
                {
                    "kind": "offload",
                    "scope_id": selected_scope.scope_id,
                    "alias": alias,
                    "step_index": step_index,
                    "text_sha256": digest,
                    "estimated_tokens": size["estimated_tokens"],
                    "offload_saving_bytes": saving,
                    "packed_utf8_bytes_before": packed_utf8_bytes_before,
                    "packed_utf8_bytes_after": len(packed_text.encode("utf-8")),
                    "hot_payload_bytes": hot_payload_bytes(selected_scope),
                    "hot_evictions": evicted,
                }
            )
        decisions.append(decision)
    return decisions
