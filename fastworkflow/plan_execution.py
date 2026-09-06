"""Run-to-terminal plan execution for EXP-028 Arms B/C.

Arm B executes the compiled leaf order without composite orchestration. Arm C
uses the same public task coverage but keeps composite grouping metadata for
accounting and context reuse.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Union

from fastworkflow.binding_normalizers import (
    AliasCandidate,
    alias_resolver_handle_field,
    resolve_binding_alias,
)
from fastworkflow.plan import (
    LEAF_LEVELS,
    CompositeGroup,
    PlanExecutionMetadata,
    PlanNode,
    PlanRecord,
    render_account,
)
from fastworkflow.typed_failure import (
    CODE_EXTRACTION_TRUNCATED,
    CODE_PROVIDER_TIMEOUT,
    TypedFailure,
    extraction_truncated_failure,
    is_provider_timeout,
    provider_timeout_failure,
)


class PlanExecutionArm(str, Enum):
    A = "a"
    B = "b"
    C = "c"


class PlanExecutionOutcome(str, Enum):
    """One deterministic plan lifecycle outcome."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_USER = "needs-user"
    EXHAUSTED = "exhausted"
    BLOCKED = "blocked"
    FAILED = "failed"
    CENSORED = "censored"
    PROVIDER_TIMEOUT = "provider-timeout"


def plan_execution_arm_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> PlanExecutionArm:
    source = os.environ if env is None else env
    return PlanExecutionArm(
        source.get("FW_PLAN_EXECUTION_ARM", "a").lower()
    )


def plan_stress_mode_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Parse the explicit stress-mode switch without truthiness drift."""
    source = os.environ if env is None else env
    raw = source.get("FW_PLAN_STRESS_MODE", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        "FW_PLAN_STRESS_MODE must be one of "
        "0/1, false/true, no/yes, or off/on"
    )


@dataclass
class SafetyEnvelopeState:
    """Safety state for runaway/cycle censoring."""

    enabled: bool = True
    wall_time_limit_s: int = 1800
    started_at_epoch_s: float = 0.0
    last_fingerprint: Optional[str] = None
    repeated_fingerprint_count: int = 0
    censored: bool = False
    censored_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.wall_time_limit_s <= 0:
            raise ValueError("wall_time_limit_s must be positive")
        if self.started_at_epoch_s <= 0:
            self.started_at_epoch_s = time.time()

    def wall_time_expired(self, now_epoch_s: Optional[float] = None) -> bool:
        """Whether the arm-invariant emergency wall envelope has elapsed."""
        if not self.enabled:
            return False
        now = time.time() if now_epoch_s is None else now_epoch_s
        return now - self.started_at_epoch_s >= self.wall_time_limit_s

    def censor(self, reason: str) -> None:
        """Record an infrastructure censor without converting it to task failure."""
        self.censored = True
        self.censored_reason = reason

    def observe_fingerprint(self, fingerprint: str) -> bool:
        """Return True on the fourth identical consecutive no-progress action."""
        if fingerprint == self.last_fingerprint:
            self.repeated_fingerprint_count += 1
        else:
            self.last_fingerprint = fingerprint
            self.repeated_fingerprint_count = 0
        if self.enabled and self.repeated_fingerprint_count >= 3:
            self.censor("no-progress-cycle")
            return True
        return False

    def to_state(self) -> dict:
        return {
            "enabled": self.enabled,
            "wall_time_limit_s": self.wall_time_limit_s,
            "started_at_epoch_s": self.started_at_epoch_s,
            "last_fingerprint": self.last_fingerprint,
            "repeated_fingerprint_count": self.repeated_fingerprint_count,
            "censored": self.censored,
            "censored_reason": self.censored_reason,
        }

    @classmethod
    def from_state(cls, state: Optional[dict]) -> "SafetyEnvelopeState":
        if not isinstance(state, dict):
            return cls(enabled=False)
        return cls(
            enabled=bool(state.get("enabled", True)),
            wall_time_limit_s=int(state.get("wall_time_limit_s", 1800)),
            started_at_epoch_s=float(state.get("started_at_epoch_s", 0.0)),
            last_fingerprint=state.get("last_fingerprint"),
            repeated_fingerprint_count=int(state.get("repeated_fingerprint_count", 0)),
            censored=bool(state.get("censored", False)),
            censored_reason=state.get("censored_reason"),
        )


@dataclass
class PlanExecutionResult:
    answer: str
    outcome: PlanExecutionOutcome = PlanExecutionOutcome.PARTIAL
    exhausted: bool = False
    suspended: bool = False
    needs_user: bool = False
    clarification: Optional[str] = None
    censored: bool = False
    censored_reason: Optional[str] = None
    provider_timeout: bool = False
    # ido-mn1.6.10. Goal ids whose composition call the provider stopped at its
    # completion limit. Not `censored`: a truncated leaf keeps its answer and
    # the walk keeps going, because one leaf that outran its token limit says
    # nothing about the leaves after it. Carried as ids and not a boolean so a
    # reader can tell one truncated leaf in twelve from twelve in twelve —
    # arm C concatenates every leaf's answer, and which one was cut is the whole
    # question when the composed answer is short.
    extraction_truncated_goal_ids: tuple[str, ...] = ()
    extraction_truncation_failures: tuple[tuple[str, TypedFailure], ...] = ()
    failure: Optional[TypedFailure] = None
    successful_leaf_goal_ids: tuple[str, ...] = ()
    metadata: Optional[PlanExecutionMetadata] = None

    @property
    def extraction_truncated(self) -> bool:
        return bool(self.extraction_truncated_goal_ids)

    @property
    def extraction_failure(self) -> Optional[TypedFailure]:
        """Typed aggregate reason for one or more truncated leaf answers."""
        if not self.extraction_truncation_failures:
            return None
        goal_ids = tuple(
            goal_id for goal_id, _failure in self.extraction_truncation_failures
        )
        return TypedFailure(
            disposition="transient",
            code=CODE_EXTRACTION_TRUNCATED,
            detail=(
                "composition stopped at its completion limit for plan goal(s): "
                + ", ".join(goal_ids)
                + "; the partial answers remain execution evidence"
            ),
            completed_work=tuple(
                {
                    "goal_id": goal_id,
                    "failure": failure.to_state(),
                }
                for goal_id, failure in self.extraction_truncation_failures
            ),
        )


@dataclass(frozen=True)
class PlanExecutionBinding:
    """One authoritative input exposed to an isolated leaf execution."""

    value: Union[str, tuple[str, ...]]
    source: str
    kind: Optional[str]
    normalizer: Optional[str] = None
    resolver: Optional[str] = None
    command_call_id: Optional[str] = None
    source_spans: tuple[tuple[int, int], ...] = ()

    def to_state(self) -> dict[str, object]:
        return {
            "value": (
                list(self.value)
                if isinstance(self.value, tuple)
                else self.value
            ),
            "source": self.source,
            "kind": self.kind,
            "normalizer": self.normalizer,
            "resolver": self.resolver,
            "command_call_id": self.command_call_id,
            "source_spans": [list(span) for span in self.source_spans],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "PlanExecutionBinding":
        if not isinstance(state, Mapping):
            raise TypeError("plan execution binding state must be an object")
        raw_value = state.get("value")
        if raw_value is None:
            raise ValueError("plan execution binding state requires a value")
        value = (
            tuple(str(item) for item in raw_value)
            if isinstance(raw_value, (list, tuple))
            else str(raw_value)
        )
        return cls(
            value=value,
            source=str(state.get("source") or ""),
            kind=(
                str(state["kind"])
                if state.get("kind") is not None
                else None
            ),
            normalizer=(
                str(state["normalizer"])
                if state.get("normalizer") is not None
                else None
            ),
            resolver=(
                str(state["resolver"])
                if state.get("resolver") is not None
                else None
            ),
            command_call_id=(
                str(state["command_call_id"])
                if state.get("command_call_id") is not None
                else None
            ),
            source_spans=_restore_source_spans(state.get("source_spans")),
        )


def _restore_source_spans(value: object) -> tuple[tuple[int, int], ...]:
    spans = []
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("source_spans must be a list")
    for span in value:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise TypeError("each source span must be a two-item list")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise ValueError("source spans must be increasing non-negative ranges")
        spans.append((start, end))
    return tuple(spans)


def _restore_shared_bindings(
    value: object,
) -> tuple[tuple[str, Union[str, tuple[str, ...]]], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("shared_bindings must be a list")
    restored = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("each shared binding must be a two-item list")
        name = str(item[0])
        if not name or name in names:
            raise ValueError("shared binding names must be non-empty and unique")
        names.add(name)
        raw = item[1]
        binding_value: Union[str, tuple[str, ...]] = (
            tuple(str(entry) for entry in raw)
            if isinstance(raw, (list, tuple))
            else str(raw)
        )
        restored.append((name, binding_value))
    return tuple(restored)


@dataclass(frozen=True)
class PlanExecutionScope:
    """Private execution context passed to the workflow leaf chokepoint."""

    arm: PlanExecutionArm
    task_goal_id: Optional[str]
    composite_group_id: Optional[str] = None
    composite_skill: Optional[str] = None
    shared_context_id: Optional[str] = None
    shared_bindings: tuple[
        tuple[str, Union[str, tuple[str, ...]]],
        ...,
    ] = ()
    member_ordinal: Optional[int] = None
    member_count: Optional[int] = None
    # True only after this context's non-empty shared binding input was already
    # supplied to an earlier leaf; repeating an identifier alone is not reuse.
    context_reused: bool = False
    composite_group_path: tuple[str, ...] = ()
    composite_skill_path: tuple[str, ...] = ()
    shared_context_ids: tuple[str, ...] = ()
    task_bindings: tuple[tuple[str, PlanExecutionBinding], ...] = ()
    leaf_bindings: tuple[tuple[str, PlanExecutionBinding], ...] = ()
    producer_call_ids: tuple[str, ...] = ()
    navigation_context: Optional[str] = None

    def to_state(self) -> dict[str, object]:
        def bindings_state(bindings):
            return [
                {"name": name, "binding": binding.to_state()}
                for name, binding in bindings
            ]

        return {
            "arm": self.arm.value,
            "task_goal_id": self.task_goal_id,
            "composite_group_id": self.composite_group_id,
            "composite_skill": self.composite_skill,
            "shared_context_id": self.shared_context_id,
            "shared_bindings": [
                [
                    name,
                    list(value) if isinstance(value, tuple) else value,
                ]
                for name, value in self.shared_bindings
            ],
            "member_ordinal": self.member_ordinal,
            "member_count": self.member_count,
            "context_reused": self.context_reused,
            "composite_group_path": list(self.composite_group_path),
            "composite_skill_path": list(self.composite_skill_path),
            "shared_context_ids": list(self.shared_context_ids),
            "task_bindings": bindings_state(self.task_bindings),
            "leaf_bindings": bindings_state(self.leaf_bindings),
            "producer_call_ids": list(self.producer_call_ids),
            "navigation_context": self.navigation_context,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "PlanExecutionScope":
        if not isinstance(state, Mapping):
            raise TypeError("plan execution scope state must be an object")

        def restore_bindings(key: str):
            restored = []
            names: set[str] = set()
            for item in state.get(key) or ():
                if not isinstance(item, Mapping):
                    raise TypeError(f"{key} entries must be objects")
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{key} entries require a name")
                if name in names:
                    raise ValueError(f"{key} repeats binding {name!r}")
                names.add(name)
                restored.append(
                    (
                        name,
                        PlanExecutionBinding.from_state(item.get("binding")),
                    )
                )
            return tuple(restored)

        return cls(
            arm=PlanExecutionArm(str(state.get("arm"))),
            task_goal_id=(
                str(state["task_goal_id"])
                if state.get("task_goal_id") is not None
                else None
            ),
            composite_group_id=(
                str(state["composite_group_id"])
                if state.get("composite_group_id") is not None
                else None
            ),
            composite_skill=(
                str(state["composite_skill"])
                if state.get("composite_skill") is not None
                else None
            ),
            shared_context_id=(
                str(state["shared_context_id"])
                if state.get("shared_context_id") is not None
                else None
            ),
            shared_bindings=_restore_shared_bindings(
                state.get("shared_bindings")
            ),
            member_ordinal=(
                int(state["member_ordinal"])
                if state.get("member_ordinal") is not None
                else None
            ),
            member_count=(
                int(state["member_count"])
                if state.get("member_count") is not None
                else None
            ),
            context_reused=bool(state.get("context_reused")),
            composite_group_path=tuple(
                str(item) for item in (state.get("composite_group_path") or ())
            ),
            composite_skill_path=tuple(
                str(item) for item in (state.get("composite_skill_path") or ())
            ),
            shared_context_ids=tuple(
                str(item) for item in (state.get("shared_context_ids") or ())
            ),
            task_bindings=restore_bindings("task_bindings"),
            leaf_bindings=restore_bindings("leaf_bindings"),
            producer_call_ids=tuple(
                str(item) for item in (state.get("producer_call_ids") or ())
            ),
            navigation_context=(
                str(state["navigation_context"])
                if state.get("navigation_context") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class _ScheduledLeaf:
    leaf: PlanNode
    task_goal_id: Optional[str]
    group_path: tuple[CompositeGroup, ...] = ()
    member_ordinal: Optional[int] = None


def render_leaf_instruction(
    node: PlanNode,
    scope: PlanExecutionScope,
) -> str:
    """Render the private orchestration envelope used at the agent chokepoint."""
    executable_goal = node.executable_goal_text or node.goal_text
    task_envelope = _render_task_execution_envelope(scope)
    if (
        scope.arm is not PlanExecutionArm.C
        or scope.composite_group_id is None
    ):
        return f"{task_envelope}Goal: {executable_goal}"
    shared_bindings = json.dumps(
        dict(scope.shared_bindings),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Private composite orchestration: "
        f"{' > '.join(scope.composite_skill_path)} groups "
        f"{' > '.join(scope.composite_group_path)}, member "
        f"{scope.member_ordinal} of "
        f"{scope.member_count}, shared context {scope.shared_context_id}, "
        f"bindings {shared_bindings}, reused shared input "
        f"{str(scope.context_reused).lower()}. "
        "Preserve reusable navigation and context "
        "within this group; execute only the public task goal below and do not "
        f"add another task.\n{task_envelope}Goal: {executable_goal}"
    )


def _binding_payload(
    bindings: tuple[tuple[str, PlanExecutionBinding], ...],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "value": binding.value,
            "source": binding.source,
            "kind": binding.kind,
            "normalizer": binding.normalizer,
            "resolver": binding.resolver,
            "command_call_id": binding.command_call_id,
            "source_spans": binding.source_spans,
        }
        for name, binding in bindings
    }


def _render_task_execution_envelope(scope: PlanExecutionScope) -> str:
    if not (
        scope.task_bindings
        or scope.leaf_bindings
        or scope.producer_call_ids
        or scope.navigation_context
    ):
        return ""
    payload = json.dumps(
        {
            "task_goal_id": scope.task_goal_id,
            "task_bindings": _binding_payload(scope.task_bindings),
            "leaf_bindings": _binding_payload(scope.leaf_bindings),
            "navigation_context": scope.navigation_context,
            "producer_call_ids": scope.producer_call_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Private task execution envelope (authoritative inputs, not another "
        f"task): {payload}. Bound inputs are already supplied; do not ask the "
        "operator to repeat them. Preserve their typed provenance and producer "
        "call ids. Use the stated navigation context as the starting point, "
        "while still verifying the command is available before dispatch. "
        "A declared candidate resolver must return one deterministic handle; "
        "never guess an ambiguous result.\n"
    )


def alias_resolution_feedback(
    scope: PlanExecutionScope,
    command_output: Any,
) -> str:
    """Resolve declared aliases from aligned listing artifacts for one leaf."""
    response = getattr(command_output, "command_response", None)
    artifacts = getattr(response, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        return ""
    labels = artifacts.get("labels")
    if not isinstance(labels, (list, tuple)):
        return ""

    declared: dict[tuple[str, str], PlanExecutionBinding] = {}
    for name, binding in (*scope.task_bindings, *scope.leaf_bindings):
        if binding.resolver:
            declared[(name, binding.resolver)] = binding

    feedback: list[str] = []
    for (name, resolver_id), binding in sorted(declared.items()):
        handle_field = alias_resolver_handle_field(resolver_id)
        handles = artifacts.get(handle_field)
        if not isinstance(handles, (list, tuple)) or len(handles) != len(labels):
            continue
        aliases = artifacts.get("aliases")
        candidates = []
        for index, (handle, label) in enumerate(zip(handles, labels)):
            candidate_aliases: tuple[str, ...] = ()
            if isinstance(aliases, Mapping):
                raw_aliases = aliases.get(str(handle)) or ()
                candidate_aliases = (
                    (str(raw_aliases),)
                    if isinstance(raw_aliases, str)
                    else tuple(str(alias) for alias in raw_aliases)
                )
            elif isinstance(aliases, (list, tuple)) and index < len(aliases):
                raw_aliases = aliases[index]
                candidate_aliases = (
                    (str(raw_aliases),)
                    if isinstance(raw_aliases, str)
                    else tuple(str(alias) for alias in (raw_aliases or ()))
                )
            candidates.append(
                AliasCandidate(
                    handle=str(handle),
                    label=str(label),
                    aliases=candidate_aliases,
                )
            )
        query = (
            ", ".join(binding.value)
            if isinstance(binding.value, tuple)
            else binding.value
        )
        resolution = resolve_binding_alias(resolver_id, query, candidates)
        payload = {
            "binding_name": name,
            "binding_source": binding.source,
            "binding_kind": binding.kind,
            "input": query,
            "resolver_id": resolution.resolver_id,
            "status": resolution.status,
            "handle": resolution.handle,
            "label": resolution.label,
            "matched_by": resolution.matched_by,
            "candidate_handles": resolution.candidate_handles,
            "producer_call_id": getattr(command_output, "command_call_id", None),
        }
        instruction = (
            "Continue with the resolved handle and do not ask for this bound "
            "input again."
            if resolution.status == "resolved"
            else (
                "Do not guess. A clarification may name these concrete "
                "candidates because disambiguation is genuinely missing."
                if resolution.status == "ambiguous"
                else (
                    "Do not ask the operator to repeat the same alias. Use a "
                    "declared producer or return this typed resolver gap."
                )
            )
        )
        feedback.append(
            "Private deterministic binding resolution: "
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + f". {instruction}"
        )
    return "\n".join(feedback)


def execute_plan(
    plan: PlanRecord,
    *,
    execute_leaf: Callable[..., object],
    arm: PlanExecutionArm,
    safety: SafetyEnvelopeState,
    resumed_from_goal_id: Optional[str] = None,
    resumed_leaf_result: Optional[object] = None,
    resolve_delayed_bindings: Optional[Callable[[PlanRecord], bool]] = None,
    current_navigation_context: Optional[Callable[[], Optional[str]]] = None,
    on_progress: Optional[
        Callable[[PlanRecord, PlanNode, str], None]
    ] = None,
) -> PlanExecutionResult:
    """Execute executable leaves in deterministic order to one terminal state."""
    if arm is PlanExecutionArm.A:
        raise ValueError(
            "Arm A is the flat planner control and cannot execute a compiled plan"
        )
    if plan.execution is not None and plan.execution.arm != arm.value:
        raise ValueError(
            "cannot resume a plan under a different execution arm: "
            f"stored={plan.execution.arm}, requested={arm.value}"
        )
    if (resumed_from_goal_id is None) != (resumed_leaf_result is None):
        raise ValueError(
            "a resumed leaf requires both its goal id and its result"
        )
    if resumed_from_goal_id is not None:
        resumed_leaf = plan.node(resumed_from_goal_id)
        if (
            resumed_leaf is None
            or not resumed_leaf.is_leaf
            or resumed_leaf.status != "needs-user"
        ):
            raise ValueError(
                "a resumed result may be consumed only by the active "
                "needs-user leaf"
            )
    reconcile_plan_statuses(plan)

    scheduled_leaf_ids: list[str] = []
    scheduled_task_ids: list[str] = []
    applied_group_ids: list[str] = []
    grouped_task_ids: set[str] = set()
    grouped_leaf_ids: set[str] = set()

    def refresh_schedule() -> tuple[_ScheduledLeaf, ...]:
        refreshed = _execution_schedule(plan, arm)
        for scheduled in refreshed:
            if scheduled.leaf.goal_id not in scheduled_leaf_ids:
                scheduled_leaf_ids.append(scheduled.leaf.goal_id)
            if (
                scheduled.task_goal_id is not None
                and scheduled.task_goal_id not in scheduled_task_ids
            ):
                scheduled_task_ids.append(scheduled.task_goal_id)
            for group in scheduled.group_path:
                if group.group_id not in applied_group_ids:
                    applied_group_ids.append(group.group_id)
            if scheduled.group_path:
                grouped_leaf_ids.add(scheduled.leaf.goal_id)
                if scheduled.task_goal_id is not None:
                    grouped_task_ids.add(scheduled.task_goal_id)
        return refreshed

    schedule = refresh_schedule()
    executed_leaf_ids: list[str] = list(
        plan.execution.executed_leaf_goal_ids
        if plan.execution is not None
        else ()
    )
    context_reuse_count = (
        plan.execution.context_reuse_count
        if plan.execution is not None and plan.execution.arm == arm.value
        else 0
    )
    seen_context_inputs = _completed_context_inputs(plan)
    # ido-mn1.6.10: goal ids whose extraction the provider cut at its completion
    # limit, in execution order. A list rather than a set so the order a reader
    # sees is the order the walk took.
    extraction_truncated_ids: list[str] = list(
        plan.execution.extraction_truncated_goal_ids
        if plan.execution is not None
        else ()
    )
    extraction_truncation_failures: dict[str, TypedFailure] = dict(
        plan.execution.extraction_truncation_failures
        if plan.execution is not None
        else {}
    )
    terminal_failure: Optional[TypedFailure] = (
        plan.execution.terminal_failure
        if plan.execution is not None
        else None
    )

    def finish_metadata() -> PlanExecutionMetadata:
        schedule_leaf_ids = tuple(scheduled_leaf_ids)
        schedule_task_ids = tuple(scheduled_task_ids)
        metadata = PlanExecutionMetadata(
            arm=arm.value,
            packing_applied=(
                arm is PlanExecutionArm.C and bool(applied_group_ids)
            ),
            scheduled_leaf_goal_ids=schedule_leaf_ids,
            scheduled_task_goal_ids=schedule_task_ids,
            schedule_sha256=_schedule_sha256(
                schedule_leaf_ids,
                schedule_task_ids,
            ),
            composite_group_ids=tuple(applied_group_ids),
            composite_groups_applied=len(applied_group_ids),
            grouped_task_count=len(grouped_task_ids),
            grouped_leaf_count=len(grouped_leaf_ids),
            shared_binding_count=sum(
                len(group.shared_bindings)
                for group in plan.composite_groups
                if group.group_id in applied_group_ids
            ),
            context_reuse_count=context_reuse_count,
            executed_leaf_goal_ids=_ordered_unique(executed_leaf_ids),
            public_task_keys=plan.compiled_public_task_keys,
            extraction_truncated_goal_ids=tuple(extraction_truncated_ids),
            extraction_truncation_failures=dict(
                extraction_truncation_failures
            ),
            terminal_failure=terminal_failure,
        )
        plan.execution = metadata
        return metadata

    def result_for(
        outcome: PlanExecutionOutcome,
        *,
        suspended: bool = False,
        clarification: Optional[str] = None,
        censored_reason: Optional[str] = None,
        failure: Optional[TypedFailure] = None,
    ) -> PlanExecutionResult:
        # Read here rather than passed in at every `result_for` call site: a
        # truncation recorded on leaf 3 has to survive a plan that ends on leaf
        # 9 for any reason at all, and an argument would have to be threaded
        # through eight returns to say the same thing.
        _propagate_parent_statuses(plan)
        successful_leaf_ids = tuple(
            leaf.goal_id for leaf in plan.leaves if leaf.status == "done"
        )
        return PlanExecutionResult(
            answer=(
                clarification
                if suspended and clarification
                else _render_execution_account(plan, outcome)
            ),
            outcome=outcome,
            exhausted=outcome is PlanExecutionOutcome.EXHAUSTED,
            suspended=suspended,
            needs_user=(
                suspended or outcome is PlanExecutionOutcome.NEEDS_USER
            ),
            clarification=clarification,
            censored=outcome is PlanExecutionOutcome.CENSORED,
            censored_reason=censored_reason,
            provider_timeout=(
                outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
            ),
            extraction_truncated_goal_ids=tuple(extraction_truncated_ids),
            extraction_truncation_failures=tuple(
                (goal_id, extraction_truncation_failures[goal_id])
                for goal_id in extraction_truncated_ids
                if goal_id in extraction_truncation_failures
            ),
            failure=failure,
            successful_leaf_goal_ids=successful_leaf_ids,
            metadata=finish_metadata(),
        )

    def notify_progress(leaf: PlanNode, reason: str) -> None:
        finish_metadata()
        if on_progress is not None:
            on_progress(plan, leaf, reason)

    def resolve_bindings() -> bool:
        if resolve_delayed_bindings is None:
            return False
        return bool(resolve_delayed_bindings(plan))

    accepts_scope = _execute_leaf_accepts_scope(execute_leaf)
    pending_resumed_result = resumed_leaf_result
    resolve_bindings()

    while True:
        schedule = refresh_schedule()
        item = _next_runnable_leaf(
            plan,
            schedule,
            resumed_from_goal_id=(
                resumed_from_goal_id
                if pending_resumed_result is not None
                else None
            ),
        )
        if item is None:
            _mark_terminally_unreachable_leaves(plan)
            _propagate_parent_statuses(plan)
            outcome = (
                (
                    PlanExecutionOutcome.PROVIDER_TIMEOUT
                    if terminal_failure.code == CODE_PROVIDER_TIMEOUT
                    else PlanExecutionOutcome.FAILED
                )
                if terminal_failure is not None
                else _plan_outcome(plan)
            )
            return result_for(
                outcome,
                failure=terminal_failure,
            )

        leaf = item.leaf

        if safety.censored or safety.wall_time_expired():
            if not safety.censored:
                safety.censor("wall-time-cutoff")
            _mark_remaining_not_reached(plan, leaf.goal_id)
            leaf.status = "blocked"
            leaf.failure_reason = f"censored:{safety.censored_reason}"
            notify_progress(leaf, "censored")
            return result_for(
                PlanExecutionOutcome.CENSORED,
                censored_reason=safety.censored_reason,
            )

        shared_context_ids = tuple(
            f"{plan.plan_id}:{group.group_id}"
            for group in item.group_path
        )
        active_group = item.group_path[-1] if item.group_path else None
        shared_context_id = (
            shared_context_ids[-1] if shared_context_ids else None
        )
        active_shared_bindings = (
            tuple(sorted(active_group.shared_bindings.items()))
            if active_group is not None
            else ()
        )
        context_reused = (
            shared_context_id is not None
            and bool(active_shared_bindings)
            and seen_context_inputs.get(shared_context_id)
            == active_shared_bindings
        )
        if context_reused:
            context_reuse_count += 1
        if shared_context_id is not None and active_shared_bindings:
            seen_context_inputs[shared_context_id] = active_shared_bindings
        task_node = (
            plan.node(item.task_goal_id)
            if item.task_goal_id is not None
            else None
        )
        scope = PlanExecutionScope(
            arm=arm,
            task_goal_id=item.task_goal_id,
            composite_group_id=(
                active_group.group_id if active_group is not None else None
            ),
            composite_skill=(
                active_group.composite_skill if active_group is not None else None
            ),
            shared_context_id=shared_context_id,
            shared_bindings=active_shared_bindings,
            member_ordinal=item.member_ordinal,
            member_count=(
                len(active_group.member_goal_ids)
                if active_group is not None
                else None
            ),
            context_reused=context_reused,
            composite_group_path=tuple(
                group.group_id for group in item.group_path
            ),
            composite_skill_path=tuple(
                group.composite_skill for group in item.group_path
            ),
            shared_context_ids=shared_context_ids,
            task_bindings=_execution_bindings(plan, task_node),
            leaf_bindings=_execution_bindings(plan, leaf),
            producer_call_ids=_producer_call_ids(plan, leaf),
            navigation_context=(
                current_navigation_context()
                if current_navigation_context is not None
                else None
            ),
        )
        if (
            pending_resumed_result is not None
            and leaf.goal_id == resumed_from_goal_id
        ):
            result = pending_resumed_result
            pending_resumed_result = None
        else:
            try:
                result = (
                    execute_leaf(leaf, scope)
                    if accepts_scope
                    else execute_leaf(leaf)
                )
            except BaseException as exc:
                if not is_provider_timeout(exc):
                    raise
                if leaf.command_call_ids:
                    if leaf.goal_id not in executed_leaf_ids:
                        executed_leaf_ids.append(leaf.goal_id)
                    plan.budget_consumed += 1
                leaf.status = "blocked"
                leaf.failure_reason = "provider-timeout"
                failure = provider_timeout_failure(
                    exc,
                    completed_work=_completed_leaf_evidence(plan),
                )
                terminal_failure = failure
                notify_progress(leaf, "provider-timeout")
                return result_for(
                    PlanExecutionOutcome.PROVIDER_TIMEOUT,
                    failure=failure,
                )
        command_call_ids = tuple(
            str(command_call_id)
            for command_call_id in (
                getattr(result, "command_call_ids", ()) or ()
            )
            if command_call_id
        )
        if command_call_ids:
            leaf.command_call_ids = _ordered_unique(
                (*leaf.command_call_ids, *command_call_ids)
            )
        if leaf.goal_id not in executed_leaf_ids:
            executed_leaf_ids.append(leaf.goal_id)
        exhausted = bool(getattr(result, "exhausted", False))
        suspended = bool(getattr(result, "suspended", False))
        censored = bool(getattr(result, "censored", False))
        # ido-mn1.6.10, recorded BEFORE the censor/suspend/failure branches
        # because every one of them returns: a leaf that was truncated and then
        # blocked for an unrelated reason is still a leaf that was truncated,
        # and the two facts answer different questions.
        if getattr(result, "extraction_truncated", False):
            if leaf.goal_id not in extraction_truncated_ids:
                extraction_truncated_ids.append(leaf.goal_id)
            truncation_failure = getattr(result, "extraction_failure", None)
            if not isinstance(truncation_failure, TypedFailure):
                truncation_failure = extraction_truncated_failure(
                    completed_work=tuple(leaf.command_call_ids)
                )
            extraction_truncation_failures[leaf.goal_id] = truncation_failure
        failure = getattr(result, "failure", None)
        if censored:
            safety.censor(
                str(getattr(result, "censored_reason", "") or "unknown")
            )
            leaf.status = "blocked"
            leaf.failure_reason = f"censored:{safety.censored_reason}"
            _mark_remaining_not_reached(plan, leaf.goal_id)
            notify_progress(leaf, "censored")
            return result_for(
                PlanExecutionOutcome.CENSORED,
                censored_reason=safety.censored_reason,
            )
        if suspended:
            leaf.status = "needs-user"
            notify_progress(leaf, "needs-user")
            return result_for(
                PlanExecutionOutcome.NEEDS_USER,
                suspended=True,
                clarification=getattr(result, "clarification", None),
            )
        plan.budget_consumed += 1
        if isinstance(failure, TypedFailure):
            terminal_failure = failure
            leaf.status = "blocked"
            leaf.failure_reason = failure.code
            notify_progress(leaf, "failed")
            resolve_bindings()
            continue
        if not exhausted and not leaf.command_call_ids:
            leaf.status = "blocked"
            leaf.failure_reason = "leaf returned without command execution evidence"
            notify_progress(leaf, "blocked")
            resolve_bindings()
            continue
        if exhausted:
            leaf.status = "exhausted"
            leaf.failure_reason = "max_iters_exhausted"
            notify_progress(leaf, "exhausted")
            return result_for(PlanExecutionOutcome.EXHAUSTED)
        else:
            leaf.status = "done"
            leaf.failure_reason = None
            notify_progress(leaf, "done")
        resolve_bindings()


def _execution_bindings(
    plan: PlanRecord,
    node: Optional[PlanNode],
) -> tuple[tuple[str, PlanExecutionBinding], ...]:
    if node is None:
        return ()
    catalog = getattr(plan, "_catalog", None)
    skill = (
        catalog.get(node.skill or "")
        if catalog is not None and hasattr(catalog, "get")
        else None
    )
    projected: list[tuple[str, PlanExecutionBinding]] = []
    for name, binding in sorted(node.bindings.items()):
        if binding.value is None:
            continue
        slot = skill.slot(name) if skill is not None else None
        value = (
            tuple(str(item) for item in binding.value)
            if isinstance(binding.value, list)
            else str(binding.value)
        )
        projected.append(
            (
                name,
                PlanExecutionBinding(
                    value=value,
                    source=binding.source,
                    kind=binding.kind,
                    normalizer=binding.normalizer,
                    resolver=(
                        getattr(slot, "resolver", None)
                        if slot is not None
                        else None
                    ),
                    command_call_id=binding.command_call_id,
                    source_spans=tuple(
                        (span.start, span.end)
                        for span in binding.source_spans
                    ),
                ),
            )
        )
    return tuple(projected)


def _producer_call_ids(
    plan: PlanRecord,
    leaf: PlanNode,
) -> tuple[str, ...]:
    call_ids: list[str] = []
    current: Optional[PlanNode] = leaf
    ancestors: list[PlanNode] = []
    while current is not None:
        ancestors.append(current)
        current = (
            plan.node(current.parent_goal_id)
            if current.parent_goal_id is not None
            else None
        )
    for node in reversed(ancestors):
        for binding in node.bindings.values():
            if binding.command_call_id:
                call_ids.append(binding.command_call_id)

    prerequisite_ids: set[str] = set()
    pending = [
        prerequisite_id
        for node in ancestors
        for prerequisite_id in node.prerequisites
    ]
    while pending:
        prerequisite_id = pending.pop()
        if prerequisite_id in prerequisite_ids:
            continue
        prerequisite_ids.add(prerequisite_id)
        predecessor = plan.node(prerequisite_id)
        if predecessor is not None:
            pending.extend(predecessor.prerequisites)

    for prerequisite_id in (
        node.goal_id for node in plan.nodes if node.goal_id in prerequisite_ids
    ):
        for candidate in plan.nodes:
            if not _is_descendant(plan, candidate, prerequisite_id):
                continue
            call_ids.extend(candidate.command_call_ids)
    return _ordered_unique(call_ids)


def _next_runnable_leaf(
    plan: PlanRecord,
    schedule: tuple[_ScheduledLeaf, ...],
    *,
    resumed_from_goal_id: Optional[str],
) -> Optional[_ScheduledLeaf]:
    for item in schedule:
        leaf = item.leaf
        if (
            resumed_from_goal_id is not None
            and leaf.goal_id == resumed_from_goal_id
        ):
            return item
        if leaf.status != "not-reached":
            continue
        if _prerequisites_satisfied(plan, leaf):
            return item
    return None


def _mark_terminally_unreachable_leaves(plan: PlanRecord) -> None:
    """Close leaves blocked by a terminal prerequisite without inventing work."""
    terminal_non_success = {
        "blocked",
        "exhausted",
        "needs-user",
        "skipped",
    }
    for leaf in plan.leaves:
        if leaf.status != "not-reached" or not leaf.prerequisites:
            continue
        blockers = tuple(
            prerequisite
            for prerequisite in leaf.prerequisites
            if (
                (predecessor := plan.node(prerequisite)) is not None
                and predecessor.status in terminal_non_success
            )
        )
        if blockers:
            leaf.status = "skipped"
            leaf.failure_reason = (
                "prerequisite-not-complete:" + ",".join(blockers)
            )


def _propagate_parent_statuses(plan: PlanRecord) -> None:
    """Project descendant leaf outcomes onto every non-leaf plan node."""
    nodes = sorted(
        (node for node in plan.nodes if not node.is_leaf),
        key=lambda node: plan.depth_of(node.goal_id),
        reverse=True,
    )
    for node in nodes:
        children = plan.children(node.goal_id)
        if not children:
            continue
        statuses = tuple(child.status for child in children)
        if all(status == "done" for status in statuses):
            node.status = "done"
            node.failure_reason = None
        elif "needs-user" in statuses:
            node.status = "needs-user"
            node.failure_reason = "descendant-needs-user"
        elif "exhausted" in statuses:
            node.status = "exhausted"
            node.failure_reason = "descendant-exhausted"
        elif "blocked" in statuses:
            node.status = "blocked"
            node.failure_reason = "descendant-blocked"
        elif all(status == "skipped" for status in statuses):
            node.status = "skipped"
            node.failure_reason = "all-descendants-skipped"
        elif "skipped" in statuses:
            node.status = "blocked"
            node.failure_reason = "descendant-skipped"
        else:
            node.status = "not-reached"
            node.failure_reason = None


def reconcile_plan_statuses(plan: PlanRecord) -> None:
    """Make every parent status agree with its current leaf evidence."""
    _mark_terminally_unreachable_leaves(plan)
    _propagate_parent_statuses(plan)


def _plan_outcome(plan: PlanRecord) -> PlanExecutionOutcome:
    leaves = plan.leaves
    node_statuses = tuple(node.status for node in plan.nodes)
    if "needs-user" in node_statuses:
        return PlanExecutionOutcome.NEEDS_USER
    if "exhausted" in node_statuses:
        return PlanExecutionOutcome.EXHAUSTED
    if "blocked" in node_statuses:
        return PlanExecutionOutcome.BLOCKED
    if not leaves:
        return PlanExecutionOutcome.COMPLETED
    statuses = tuple(leaf.status for leaf in leaves)
    if all(status == "done" for status in statuses):
        return PlanExecutionOutcome.COMPLETED
    if "not-reached" in statuses or "skipped" in statuses:
        return PlanExecutionOutcome.PARTIAL
    return PlanExecutionOutcome.BLOCKED


def _completed_leaf_evidence(plan: PlanRecord) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "goal_id": leaf.goal_id,
            "status": leaf.status,
            "command_call_ids": leaf.command_call_ids,
        }
        for leaf in plan.leaves
        if leaf.status == "done" or leaf.command_call_ids
    )


def _render_execution_account(
    plan: PlanRecord,
    outcome: PlanExecutionOutcome,
) -> str:
    """Render an account for complete and incomplete plans alike."""
    leaves = plan.leaves
    done = tuple(leaf for leaf in leaves if leaf.status == "done")
    evidence_leaves = tuple(leaf for leaf in leaves if leaf.command_call_ids)
    command_calls = sum(len(leaf.command_call_ids) for leaf in evidence_leaves)
    parts = [
        f"Plan outcome: {outcome.value}.",
        f"{len(done)} of {len(leaves)} leaves done.",
        (
            "Execution evidence: "
            f"{len(evidence_leaves)} leaves carry {command_calls} command call "
            f"{'reference' if command_calls == 1 else 'references'}."
        ),
    ]
    needs_user_nodes = tuple(
        node for node in plan.nodes if node.status == "needs-user"
    )
    blocked_nodes = tuple(
        node for node in plan.nodes if node.status == "blocked"
    )
    if needs_user_nodes:
        parts.append(
            f"{len(needs_user_nodes)} plan "
            f"{'node needs' if len(needs_user_nodes) == 1 else 'nodes need'} "
            "additional input."
        )
    if blocked_nodes:
        parts.append(
            f"{len(blocked_nodes)} plan "
            f"{'node is' if len(blocked_nodes) == 1 else 'nodes are'} blocked."
        )
    incomplete_account = render_account(plan)
    repeated_count = f"{len(done)} of {len(leaves)} leaves done."
    if incomplete_account.startswith(repeated_count):
        incomplete_account = incomplete_account[len(repeated_count) :].strip()
    if incomplete_account:
        parts.append(incomplete_account)
    return " ".join(parts)


def _execution_schedule(
    plan: PlanRecord,
    arm: PlanExecutionArm,
) -> tuple[_ScheduledLeaf, ...]:
    leaves = tuple(
        node
        for node in plan.nodes
        if node.level in LEAF_LEVELS
        and node.executable
        and node.executable_goal_text
    )
    if arm is PlanExecutionArm.B or not plan.composite_groups:
        return tuple(
            _ScheduledLeaf(
                leaf=leaf,
                task_goal_id=_public_task_ancestor(plan, leaf),
            )
            for leaf in leaves
        )

    leaf_positions = {
        leaf.goal_id: position for position, leaf in enumerate(leaves)
    }
    assigned_leaf_ids: set[str] = set()
    units: list[tuple[int, str, tuple[_ScheduledLeaf, ...]]] = []
    root_groups = tuple(
        group
        for group in plan.composite_groups
        if group.parent_group_id is None
    )
    for group in root_groups:
        grouped: list[_ScheduledLeaf] = []
        for member_ordinal, member_goal_id in enumerate(
            group.member_goal_ids,
            start=1,
        ):
            for leaf in leaves:
                if leaf.goal_id in assigned_leaf_ids:
                    continue
                if not _is_descendant(plan, leaf, member_goal_id):
                    continue
                assigned_leaf_ids.add(leaf.goal_id)
                grouped.append(
                    _ScheduledLeaf(
                        leaf=leaf,
                        task_goal_id=member_goal_id,
                        group_path=_group_path_for_member(
                            plan,
                            group,
                            member_goal_id,
                        ),
                        member_ordinal=member_ordinal,
                    )
                )
        if grouped:
            units.append(
                (
                    min(leaf_positions[item.leaf.goal_id] for item in grouped),
                    group.group_id,
                    tuple(grouped),
                )
            )
    for leaf in leaves:
        if leaf.goal_id in assigned_leaf_ids:
            continue
        units.append(
            (
                leaf_positions[leaf.goal_id],
                f"leaf:{leaf.goal_id}",
                (
                    _ScheduledLeaf(
                        leaf=leaf,
                        task_goal_id=_public_task_ancestor(plan, leaf),
                    ),
                ),
            )
        )
    return tuple(
        item
        for _position, _unit_id, unit in sorted(
            units,
            key=lambda entry: (entry[0], entry[1]),
        )
        for item in unit
    )


def _public_task_ancestor(plan: PlanRecord, leaf: PlanNode) -> Optional[str]:
    node: Optional[PlanNode] = leaf
    while node is not None:
        if node.level == "task" and node.is_public:
            return node.goal_id
        node = (
            plan.node(node.parent_goal_id)
            if node.parent_goal_id is not None
            else None
        )
    return None


def _group_path_for_member(
    plan: PlanRecord,
    root_group: CompositeGroup,
    member_goal_id: str,
) -> tuple[CompositeGroup, ...]:
    path = [root_group]
    current = root_group
    while True:
        children = sorted(
            (
                group
                for group in plan.composite_groups
                if group.parent_group_id == current.group_id
                and member_goal_id in group.member_goal_ids
            ),
            key=lambda group: group.group_id,
        )
        if not children:
            return tuple(path)
        current = children[0]
        path.append(current)


def _is_descendant(
    plan: PlanRecord,
    node: PlanNode,
    ancestor_goal_id: str,
) -> bool:
    current: Optional[PlanNode] = node
    while current is not None:
        if current.goal_id == ancestor_goal_id:
            return True
        current = (
            plan.node(current.parent_goal_id)
            if current.parent_goal_id is not None
            else None
        )
    return False


def _ordered_unique(values) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _schedule_sha256(
    leaf_goal_ids: tuple[str, ...],
    task_goal_ids: tuple[str, ...],
) -> str:
    """Hash only execution order, never an arm label or grouping metadata."""
    payload = json.dumps(
        {
            "leaf_goal_ids": leaf_goal_ids,
            "task_goal_ids": task_goal_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _completed_context_inputs(
    plan: PlanRecord,
) -> dict[
    str,
    tuple[tuple[str, Union[str, tuple[str, ...]]], ...],
]:
    completed: dict[
        str,
        tuple[tuple[str, Union[str, tuple[str, ...]]], ...],
    ] = {}
    for group in plan.composite_groups:
        if group.parent_group_id is not None or not group.shared_bindings:
            continue
        for member_goal_id in group.member_goal_ids:
            if any(
                leaf.status == "done"
                and _is_descendant(plan, leaf, member_goal_id)
                for leaf in plan.leaves
            ):
                completed[f"{plan.plan_id}:{group.group_id}"] = tuple(
                    sorted(group.shared_bindings.items())
                )
                break
    return completed


def _execute_leaf_accepts_scope(execute_leaf: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(execute_leaf).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2 or any(
        parameter.kind is parameter.VAR_POSITIONAL
        for parameter in parameters
    )


def _prerequisites_satisfied(plan: PlanRecord, node: PlanNode) -> bool:
    for required in node.prerequisites:
        predecessor = plan.node(required)
        if predecessor is None:
            return False
        if predecessor.status != "done":
            return False
    return True


def _leaf_fingerprint(leaf: PlanNode) -> str:
    payload = f"{leaf.goal_id}|{leaf.goal_text}|{sorted(leaf.bindings)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mark_remaining_not_reached(plan: PlanRecord, active_goal_id: str) -> None:
    started = False
    for node in plan.nodes:
        if node.goal_id == active_goal_id:
            started = True
        if started and node.level in LEAF_LEVELS and node.status == "not-reached":
            node.status = "not-reached"
