"""The plan: a decomposition tree, an execution DAG, and a deterministic walk.

EXP-028 decisions 3, 5 and 6. The planner in force before this module was one
`dspy.ChainOfThought` whose only output was `next_steps: "task descriptions as a
numbered list of short sentences separated by line breaks"` — and then
`prediction.next_steps.split()` joined back with single spaces, so the line
breaks the signature asked for were destroyed before the plan reached the
executor. Requirements §4.5 says a prose list appended to a user message is not
a plan; FW-REQ-010 clause 1 says a goal shall not be reconstructed by parsing
prose. Nothing there could be.

**No model call occurs anywhere in this module, and that is a stop condition
rather than a preference.** P-01, FW-REQ-012, and §14.1's rejection of "a ReAct
self-tool as the recursive plan executor" all say the same thing: one model
phase selects (`workflow_agent.select_skills`), with one provider call per
application attempt and at most one validation-guided retry, and nothing below
it is a model call. The assertion is structural — `tests/test_plan.py` parses
this file and fails on an `import dspy` — because a test that only exercises the
happy path cannot see a model call added to a branch it does not reach.

**Two structures, and they are not the same structure** (§4.13, P-03).

* The *decomposition tree* is `parent_goal_id`: what expansion produced.
* The *execution DAG* is `prerequisites`: within one node's children the steps
  are sequential, and across sibling task nodes there are no edges at all —
  three leavers are independent.

v1 executes them in one total order and that is a scheduler decision, not the
schema (§19.0 invariant 2): the edges are persisted from the first version so
that widening later is a scheduling change rather than a migration.

**`completed` is not in the status vocabulary and its absence is load-bearing.**
P-07 and §19.0 invariant 3 reserve that word for the contract evaluator, and
FW-REQ-010's acceptance criterion — a goal cannot become complete without its
predicate evaluated against evidence — is not met by v1 and must not appear to
be. `done` means the leaf's loop finished inside its allowance. Whether the goal
holds is a question asked independently, by something else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from fastworkflow.binding_normalizers import normalize_binding_value
from fastworkflow.skill_catalog import MAX_DEPTH, Skill, SkillCatalog, Step
from fastworkflow.typed_failure import TypedFailure

#: The flag, read exactly the way `FW_ASK_POLICY` is (EXP-028 decision 6).
PLAN_MODE_ENV_VAR = "FW_PLAN_DECOMPOSITION"


class PlanMode(str, Enum):
    """Rollout state, per the plan's `off`/`shadow`/`enforce` convention.

    `shadow` selects, binds, expands, records the whole plan record and its
    spans — and executes the old way. That is worth having for EXP-026's
    reason: the decomposition's quality becomes measurable on the same runs as
    the control, so "would the selector have picked the right skills, over the
    right subjects" is answered before any behaviour depends on it.
    """

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class PlanConfigurationError(ValueError):
    """A non-off mode with no manifest-enabled catalogue to act on.

    A hard error rather than a silent fall back to off, for EXP-025a's reason
    unchanged: a run configured to ENFORCE that quietly enforced nothing is
    still recorded and measured as a treatment arm.
    """


#: Leaf status vocabulary (decision 5). Deliberately not `completed`.
#:
#: * `done` — the leaf's loop finished inside its allowance;
#: * `exhausted` — it spent its allowance (EXP-027's `TurnPartial` is the record);
#: * `blocked` — it could not proceed for a reason that is not the budget;
#: * `needs-user` — a required slot could not be sourced (FW-REQ-011 clause 6);
#: * `not-reached` — the plan stopped before it;
#: * `skipped` — a prerequisite decided it should not run.
LeafStatus = Literal[
    "done", "exhausted", "blocked", "needs-user", "not-reached", "skipped"
]

LEAF_STATUSES: tuple[str, ...] = (
    "done",
    "exhausted",
    "blocked",
    "needs-user",
    "not-reached",
    "skipped",
)

#: What a node is. `commands` is a command sequence — the other kind of leaf
#: (decision 3, "leaves are atomic-skill nodes and command sequences").
NodeLevel = Literal["task", "composite", "atomic", "commands"]

#: The levels that are leaves of the decomposition tree.
LEAF_LEVELS: frozenset[str] = frozenset({"atomic", "commands"})

#: Binding precedence (decision 2), recorded per binding with its source:
#: the explicit utterance; a handle captured by an already-executed leaf;
#: otherwise `needs-user`.
#:
#: Historical pre-Gate wording retained for context: "the explicit utterance;
#: a handle captured by an already-executed leaf; the skill's `on_repeat` rule;
#: otherwise `needs-user`." Gate 2 proved that treating `on_repeat` as a value
#: fabricates data; it is now retained separately as policy.
#:
#: `skill` is the fifth and is not a precedence tier: it is a value the skill
#: body itself supplies (`inspect-entity entity_type=identity`), which is not
#: sourced at bind time at all. Recording it as `utterance` would be a false
#: provenance claim in a record whose whole purpose is provenance.
#: The earlier implementation continued: "dropping it would leave
#: `inspect-entity`'s required `entity_type` unbound so the binder would reach
#: for `on_repeat` — a `browse_catalog` fallback — when the body already said
#: `identity`." The literal still binds; only the fallback-as-value claim was
#: removed.
BindingSource = Literal["utterance", "captured", "needs-user", "skill"]
BindingKind = Literal[
    "exact_text", "normalized_enum", "captured_handle", "skill_literal"
]

#: Public/private projection (FW-REQ-010B). Every task- and composite-level node
#: is public, its projection being its rendered `goal` with slots substituted.
#: Every atomic node, every navigation and handle-binding node, is private.
Visibility = Literal["public", "private"]
PlanRecordMode = Literal["off", "shadow", "enforce"]
EdgeProvenance = Literal[
    "data",
    "explicit-order",
    "composite",
    "stable-tiebreak",
    "composite-pack",
]


class _PlanModel(BaseModel):
    """Shared posture: no unknown keys, assignment validated.

    `extra="forbid"` for `decision_signals._Strict`'s reason — a typo'd field
    that parses is a field nobody notices is missing. Not frozen, unlike the
    turn-capture records: a plan record is live state for the length of a turn,
    and the executor moves a leaf from `not-reached` to `done`. What must not be
    editable after the fact is the *turn record*, and that is a copy.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PlanEdge(_PlanModel):
    """One execution-DAG edge and why it exists."""

    from_goal_id: str
    to_goal_id: str
    provenance: EdgeProvenance


class CompositeGroup(_PlanModel):
    """Private orchestration synthesized over existing public task nodes.

    A group is not a selected skill invocation and is never part of the public
    projection. Its members are the exact task nodes already compiled from the
    selector's task-first output. ``member_goal_ids`` is in the composite
    expansion order, which Arm C may schedule; Arm B ignores it.
    """

    group_id: str
    composite_skill: str
    parent_group_id: Optional[str] = None
    depth: int = Field(default=1, ge=1)
    member_goal_ids: tuple[str, ...]
    member_task_keys: tuple[str, ...]
    shared_bindings: dict[str, Union[str, tuple[str, ...]]] = Field(
        default_factory=dict
    )
    orchestration_edges: tuple[PlanEdge, ...] = ()
    signature_sha256: str

    @model_validator(mode="after")
    def _members_align(self) -> "CompositeGroup":
        if not self.member_goal_ids:
            raise ValueError("a composite group requires at least one member")
        if len(self.member_goal_ids) != len(self.member_task_keys):
            raise ValueError(
                "composite group member ids and task keys must have equal length"
            )
        if len(self.member_goal_ids) != len(set(self.member_goal_ids)):
            raise ValueError("a composite group cannot repeat a member")
        return self


class CompositePackingMetrics(_PlanModel):
    """Deterministic compiler counters for private composite packing."""

    candidate_count: int = Field(default=0, ge=0)
    selected_root_group_count: int = Field(default=0, ge=0)
    selected_recursive_group_count: int = Field(default=0, ge=0)
    packed_task_count: int = Field(default=0, ge=0)
    unpacked_task_count: int = Field(default=0, ge=0)
    orchestration_edge_count: int = Field(default=0, ge=0)
    shared_binding_count: int = Field(default=0, ge=0)
    packing_sha256: str = ""


class PlanExecutionMetadata(_PlanModel):
    """Observable private scheduling/accounting produced by one B/C execution."""

    arm: Literal["b", "c"]
    packing_applied: bool
    scheduled_leaf_goal_ids: tuple[str, ...] = ()
    scheduled_task_goal_ids: tuple[str, ...] = ()
    schedule_sha256: str
    composite_group_ids: tuple[str, ...] = ()
    composite_groups_applied: int = Field(default=0, ge=0)
    grouped_task_count: int = Field(default=0, ge=0)
    grouped_leaf_count: int = Field(default=0, ge=0)
    shared_binding_count: int = Field(default=0, ge=0)
    context_reuse_count: int = Field(default=0, ge=0)
    executed_leaf_goal_ids: tuple[str, ...] = ()
    public_task_keys: tuple[str, ...] = ()
    # Composition truncation is orthogonal to leaf command execution: a leaf
    # can carry durable command evidence and still have an answer the provider
    # stopped at its completion limit. Persist both the affected goals and
    # their typed reasons so a resumed aggregate cannot silently upgrade them.
    extraction_truncated_goal_ids: tuple[str, ...] = ()
    extraction_truncation_failures: dict[str, TypedFailure] = Field(
        default_factory=dict
    )
    # Provider/task terminal failures belong to the aggregate execution record,
    # not only to the process-local PlanExecutionResult that first observed one.
    terminal_failure: Optional[TypedFailure] = None


class SourceSpan(_PlanModel):
    """One exact ``[start, end)`` character span in the user utterance."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str

    @model_validator(mode="after")
    def _span_matches_text_length(self) -> "SourceSpan":
        if self.end <= self.start:
            raise ValueError("a source span end must be greater than its start")
        if self.end - self.start != len(self.text):
            raise ValueError("a source span length must equal its recorded text length")
        return self


class InvocationEvidence(_PlanModel):
    """Selector/caller evidence for one explicitly supplied invocation slot."""

    kind: BindingKind
    source_spans: tuple[SourceSpan, ...] = ()
    normalizer: Optional[str] = None
    command_call_id: Optional[str] = None

    @model_validator(mode="after")
    def _kind_matches_evidence(self) -> "InvocationEvidence":
        if self.kind in ("exact_text", "normalized_enum") and not self.source_spans:
            raise ValueError(f"{self.kind} evidence requires source spans")
        if self.kind not in ("exact_text", "normalized_enum") and self.source_spans:
            raise ValueError(f"{self.kind} evidence cannot carry source spans")
        if self.kind == "normalized_enum" and not self.normalizer:
            raise ValueError("normalized_enum evidence requires a normalizer id")
        if self.kind != "normalized_enum" and self.normalizer is not None:
            raise ValueError("only normalized_enum evidence may name a normalizer")
        if self.kind == "captured_handle" and not self.command_call_id:
            raise ValueError(
                "captured_handle evidence requires its producing command_call_id"
            )
        if self.kind != "captured_handle" and self.command_call_id is not None:
            raise ValueError(
                "command_call_id is only valid on captured_handle evidence"
            )
        return self


class Binding(_PlanModel):
    """One slot value, and where it came from.

    `command_call_id` is set only for a `captured` binding: it is the id of the
    command whose `CommandOutput.artifacts` produced the handle, which is what
    makes "the first account on the portrait" a traceable claim rather than an
    assertion.
    """

    value: Union[str, list[str], None] = None
    source: BindingSource
    kind: Optional[BindingKind] = None
    source_spans: tuple[SourceSpan, ...] = ()
    normalizer: Optional[str] = None
    command_call_id: Optional[str] = None
    on_repeat_policy: Optional[str] = None

    @model_validator(mode="after")
    def _source_matches_evidence(self) -> "Binding":
        if self.source == "needs-user" and self.value is not None:
            raise ValueError("a needs-user binding cannot carry a value")
        if self.source != "needs-user" and self.value is None:
            raise ValueError(f"a {self.source} binding must carry a value")
        if self.source == "captured" and not self.command_call_id:
            raise ValueError(
                "a captured binding requires its producing command_call_id"
            )
        if self.source != "captured" and self.command_call_id is not None:
            raise ValueError("command_call_id is only valid on a captured binding")
        expected_source = {
            "exact_text": "utterance",
            "normalized_enum": "utterance",
            "captured_handle": "captured",
            "skill_literal": "skill",
            None: "needs-user",
        }[self.kind]
        if self.source != expected_source:
            raise ValueError(
                f"binding kind {self.kind!r} requires source {expected_source!r}"
            )
        if self.kind in ("exact_text", "normalized_enum"):
            if not self.source_spans:
                raise ValueError(f"a {self.kind} binding requires source spans")
            expected_span_count = (
                len(self.value) if isinstance(self.value, list) else 1
            )
            if len(self.source_spans) != expected_span_count:
                raise ValueError(
                    f"a {self.kind} binding requires one source span per value"
                )
        elif self.source_spans:
            raise ValueError(f"a {self.kind} binding cannot carry source spans")
        if self.kind == "normalized_enum" and not self.normalizer:
            raise ValueError("a normalized_enum binding requires a normalizer id")
        if self.kind != "normalized_enum" and self.normalizer is not None:
            raise ValueError("only a normalized_enum binding may name a normalizer")
        if self.source == "needs-user" and not self.on_repeat_policy:
            # A missing policy is legal for optional or deliberately unbindable
            # slots. Required-slot policy is validated by the skill catalogue.
            pass
        if self.source != "needs-user" and self.on_repeat_policy is not None:
            raise ValueError("on_repeat_policy is only valid on needs-user bindings")
        return self

    @property
    def is_bound(self) -> bool:
        return self.value is not None


class PlanNode(_PlanModel):
    """One goal of the plan (decision 5).

    Not a command invocation: FW-REQ-010 clause 2 and §19.0 invariant 1 forbid
    that, which is what `goal_text` — rendered from the skill's `goal` template
    with its bindings substituted, template-derived and never paraphrased by a
    model (FW-REQ-010B clause 11) — exists to prevent.
    """

    goal_id: str
    parent_goal_id: Optional[str] = None
    prerequisites: tuple[str, ...] = ()
    level: NodeLevel
    #: None for a command sequence and for an ad-hoc fallback (FW-REQ-011
    #: clause 10's logging obligation: an ad-hoc step is recorded as a node
    #: with no skill, and nothing promotes it).
    skill: Optional[str] = None
    goal_text: str = ""
    executable_goal_text: Optional[str] = None
    executable: bool = False
    visibility: Visibility = "private"
    task_key: Optional[str] = None
    bindings: dict[str, Binding] = Field(default_factory=dict)
    status: LeafStatus = "not-reached"
    budget_limit: Optional[int] = Field(default=None, ge=0)
    budget_consumed: int = Field(default=0, ge=0)
    failure_reason: Optional[str] = None
    command_call_ids: tuple[str, ...] = ()
    prerequisite_provenance: dict[str, EdgeProvenance] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _leaf_status_has_evidence(self) -> "PlanNode":
        if self.is_leaf and self.status == "done" and not self.command_call_ids:
            raise ValueError("a done leaf requires at least one command_call_id")
        if self.budget_limit is not None and self.budget_consumed > self.budget_limit:
            raise ValueError("budget_consumed cannot exceed budget_limit")
        if self.executable and not self.is_leaf:
            raise ValueError("only leaf nodes may be executable")
        if self.executable and not self.executable_goal_text:
            raise ValueError("an executable leaf requires executable_goal_text")
        if (
            self.executable
            and self.executable_goal_text
            and _has_unresolved_placeholder(self.executable_goal_text)
        ):
            raise ValueError(
                "an executable leaf goal must be fully rendered (no placeholders)"
            )
        unknown_provenance = set(self.prerequisite_provenance) - set(self.prerequisites)
        if unknown_provenance:
            raise ValueError(
                "prerequisite provenance references unknown prerequisites: "
                + ", ".join(sorted(unknown_provenance))
            )
        return self

    @property
    def is_leaf(self) -> bool:
        return self.level in LEAF_LEVELS

    @property
    def is_public(self) -> bool:
        return self.visibility == "public"

    def bound_values(self) -> dict[str, Union[str, list[str]]]:
        return {
            name: binding.value
            for name, binding in self.bindings.items()
            if binding.value is not None
        }

    def unbound_slots(self) -> tuple[str, ...]:
        return tuple(
            name for name, binding in self.bindings.items() if binding.value is None
        )


class PlanRecord(_PlanModel):
    """The whole plan of one logical turn, as it is stored on the turn record.

    `skills_fingerprint` and `selection_model` are here rather than derivable
    because the treatment is partly workflow *content*: a ledger that cannot say
    which catalogue and which model produced a decomposition is a run, not an
    arm (FW-REQ-006 clause 9).
    """

    plan_id: str
    skills_fingerprint: Optional[str] = None
    selection_model: Optional[str] = None
    mode: PlanRecordMode = PlanMode.OFF.value
    nodes: tuple[PlanNode, ...] = ()
    edges: tuple[PlanEdge, ...] = ()
    requested_public_task_keys: tuple[str, ...] = ()
    compiled_public_task_keys: tuple[str, ...] = ()
    composite_groups: tuple[CompositeGroup, ...] = ()
    packing: CompositePackingMetrics = Field(default_factory=CompositePackingMetrics)
    execution: Optional[PlanExecutionMetadata] = None
    budget_limit: Optional[int] = Field(default=None, ge=0)
    budget_consumed: int = Field(default=0, ge=0)

    # The catalogue a record was expanded from, carried so `bind_captured` can
    # expand a delayed node without every caller threading it through. A
    # PrivateAttr rather than a field: it is not state, it is the loader's
    # object, and putting it on the model would put ten skill bodies into every
    # serialized turn record.
    _catalog: Any = PrivateAttr(default=None)
    _utterance: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _budget_is_consistent(self) -> "PlanRecord":
        if self.budget_limit is not None and self.budget_consumed > self.budget_limit:
            raise ValueError("budget_consumed cannot exceed budget_limit")
        return self

    # -- reading -----------------------------------------------------------
    def node(self, goal_id: str) -> Optional[PlanNode]:
        return next((n for n in self.nodes if n.goal_id == goal_id), None)

    def children(self, goal_id: str) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.parent_goal_id == goal_id)

    @property
    def roots(self) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.parent_goal_id is None)

    @property
    def leaves(self) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.is_leaf)

    @property
    def public_nodes(self) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.is_public)

    def depth_of(self, goal_id: str) -> int:
        """1 for a root; one more for each ancestor above it."""
        depth = 1
        node = self.node(goal_id)
        while node is not None and node.parent_goal_id is not None:
            depth += 1
            node = self.node(node.parent_goal_id)
        return depth

    @property
    def max_depth(self) -> int:
        return max((self.depth_of(n.goal_id) for n in self.nodes), default=0)

    def projection(self) -> tuple[str, ...]:
        """The public projection: rendered goals, in plan order.

        v1 records it and does not present it. Presenting it is a verification
        round-trip with the operator, and adding one to the treatment arm would
        change the thing being measured (decision 2).
        """
        return tuple(n.goal_text for n in self.public_nodes)

    # -- writing -----------------------------------------------------------
    def add_node(self, node: PlanNode) -> PlanNode:
        self.nodes = self.nodes + (node,)
        return node

    def add_edge(
        self, from_goal_id: str, to_goal_id: str, provenance: EdgeProvenance
    ) -> PlanEdge:
        edge = PlanEdge(
            from_goal_id=from_goal_id,
            to_goal_id=to_goal_id,
            provenance=provenance,
        )
        if edge not in self.edges:
            self.edges = self.edges + (edge,)
        return edge


# ----------------------------------------------------------------------
# The flag
# ----------------------------------------------------------------------


def plan_mode_from_env(env: Optional[Mapping[str, str]] = None) -> PlanMode:
    """Parse `FW_PLAN_DECOMPOSITION`, defaulting to OFF.

    An enum parse rather than a truthiness check, so `FW_PLAN_DECOMPOSITION=1`
    raises instead of quietly meaning something.
    """
    source = os.environ if env is None else env
    return PlanMode(source.get(PLAN_MODE_ENV_VAR, PlanMode.OFF.value))


def require_catalog(mode: PlanMode, catalog: Optional[SkillCatalog]) -> None:
    """Raise unless a non-off mode has a manifest-enabled catalogue to act on."""
    if mode is PlanMode.OFF:
        return
    if catalog is None or not catalog.is_enabled or not len(catalog):
        raise PlanConfigurationError(
            f"{PLAN_MODE_ENV_VAR}={mode.value} with no manifest-enabled skill "
            "catalogue. Enablement is by manifest, never by file presence "
            "(FW-REQ-006 clause 1): declare 'skills_v1' in the workflow "
            "manifest's features block. Falling back to off here would record "
            "and measure a run that enforced nothing as a treatment arm."
        )
    try:
        catalog_mode = PlanMode(catalog.mode)
    except ValueError as exc:
        raise PlanConfigurationError(
            f"skill catalogue has unknown mode {catalog.mode!r}"
        ) from exc
    permissiveness = {
        PlanMode.OFF: 0,
        PlanMode.SHADOW: 1,
        PlanMode.ENFORCE: 2,
    }
    if permissiveness[mode] > permissiveness[catalog_mode]:
        raise PlanConfigurationError(
            f"{PLAN_MODE_ENV_VAR}={mode.value} exceeds the manifest-enabled "
            f"skills_v1 mode {catalog_mode.value}; a deployment may be equally "
            "or more restrictive, never more permissive"
        )


# ----------------------------------------------------------------------
# Selection output
# ----------------------------------------------------------------------


class Invocation(_PlanModel):
    """One `(skill_name, {slot: value})` the selector chose.

    Several invocations of one skill is the normal case, not an edge: the
    corpus's leaver batches are three `leaver-offboarding-sweep` invocations
    with three `identity_query` bindings and three sibling goals, per
    FW-REQ-011 clause 4.
    """

    skill_name: str
    slots: dict[str, Union[str, list[str]]] = Field(default_factory=dict)
    provenance: dict[str, InvocationEvidence] = Field(default_factory=dict)


def _canonical_slot_value(
    value: Union[str, list[str], tuple[str, ...], None]
) -> str:
    if value is None:
        return "<unbound>"
    if isinstance(value, (list, tuple)):
        return "[" + "|".join(str(item) for item in value) + "]"
    return str(value)


def _canonical_value_task_key(
    skill_name: str,
    bindings: Mapping[str, Union[str, list[str], tuple[str, ...], None]],
) -> str:
    scalar_parts: list[str] = []
    list_parts: list[str] = []
    for slot_name in sorted(bindings):
        value = bindings[slot_name]
        rendered = _canonical_slot_value(value)
        if isinstance(value, (list, tuple)):
            list_parts.append(f"{slot_name}={rendered}")
        else:
            scalar_parts.append(f"{slot_name}={rendered}")
    subject = "|".join(scalar_parts) or "<none>"
    scope = "|".join(list_parts) or "<none>"
    return f"{skill_name}::{subject}::{scope}"


def _canonical_task_key(skill_name: str, bindings: Mapping[str, Binding]) -> str:
    return _canonical_value_task_key(
        skill_name,
        {slot_name: binding.value for slot_name, binding in bindings.items()},
    )


def _expected_public_keys(
    catalog: SkillCatalog,
    skill: Skill,
    bindings: Mapping[str, Binding],
) -> tuple[str, ...]:
    keys: list[str] = []
    if _visibility(skill.level) == "public":
        keys.append(_canonical_task_key(skill.name, bindings))
    if skill.level == "atomic":
        return tuple(keys)
    for step in skill.steps:
        if step.kind == "commands" or step.skill is None:
            continue
        child_skill = catalog.get(step.skill)
        if child_skill is None:
            continue
        if step.kind == "for_each":
            list_binding = bindings.get(step.list_slot or "")
            list_values = list_binding.value if list_binding is not None else None
            if isinstance(list_values, list):
                for position, item in enumerate(list_values):
                    child_bindings = _child_bindings(
                        SimpleNamespace(bindings=dict(bindings)),
                        child_skill,
                        step.arguments,
                        loop_variable=step.loop_variable,
                        loop_value=item,
                        loop_binding=list_binding,
                        loop_index=position,
                    )
                    keys.extend(_expected_public_keys(catalog, child_skill, child_bindings))
            elif list_values is not None:
                child_bindings = _child_bindings(
                    SimpleNamespace(bindings=dict(bindings)),
                    child_skill,
                    step.arguments,
                    loop_variable=step.loop_variable,
                )
                keys.extend(_expected_public_keys(catalog, child_skill, child_bindings))
            continue
        child_bindings = _child_bindings(
            SimpleNamespace(bindings=dict(bindings)),
            child_skill,
            step.arguments,
        )
        keys.extend(_expected_public_keys(catalog, child_skill, child_bindings))
    return tuple(keys)


def _requested_public_task_keys(
    catalog: SkillCatalog,
    invocations: Sequence[Invocation],
    utterance: str,
) -> tuple[str, ...]:
    keys: list[str] = []
    for invocation in invocations:
        skill = catalog.get(invocation.skill_name)
        if skill is None:
            continue
        _validate_invocation(skill, invocation, utterance)
        keys.extend(
            _expected_public_keys(
                catalog,
                skill,
                _invocation_bindings(skill, invocation, utterance),
            )
        )
    return tuple(keys)


@dataclass(frozen=True)
class _SlotExpression:
    kind: str
    slot_name: Optional[str] = None
    literal: Optional[str] = None


@dataclass(frozen=True)
class _CompositePattern:
    child_skill: str
    child_expressions: tuple[tuple[str, _SlotExpression], ...]
    iteration_expression: Optional[_SlotExpression]
    order: tuple[int, ...]
    group_path: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class _DerivedAssignment:
    slot_name: str
    value: Union[str, tuple[str, ...]]
    append: bool = False


@dataclass(frozen=True)
class _PatternMatch:
    node_position: int
    pattern_index: int
    assignments: tuple[_DerivedAssignment, ...]


@dataclass(frozen=True)
class _CompositeCandidate:
    composite_skill: str
    member_positions: tuple[int, ...]
    member_goal_ids: tuple[str, ...]
    member_task_keys: tuple[str, ...]
    bindings: tuple[tuple[str, Union[str, tuple[str, ...]]], ...]
    matches: tuple[_PatternMatch, ...]

    @property
    def stable_key(self) -> tuple[Any, ...]:
        return (
            self.composite_skill,
            self.member_positions,
            json.dumps(
                dict(self.bindings),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @property
    def order_inversions(self) -> int:
        return sum(
            left > right
            for index, left in enumerate(self.member_positions)
            for right in self.member_positions[index + 1 :]
        )


_UNSET = object()
_MIN_COMPOSITE_PACK_MEMBERS = 2


def _root_slot_expression(slot: Any) -> _SlotExpression:
    return _SlotExpression(
        kind="root-list" if slot.list else "root-scalar",
        slot_name=slot.name,
    )


def _expression_for_argument(
    raw: str,
    environment: Mapping[str, _SlotExpression],
    *,
    loop_variable: Optional[str] = None,
    loop_expression: Optional[_SlotExpression] = None,
) -> _SlotExpression:
    if not (raw.startswith("{") and raw.endswith("}")):
        return _SlotExpression(kind="literal", literal=raw)
    reference = raw[1:-1].strip()
    if loop_variable is not None and reference == loop_variable:
        return loop_expression or _SlotExpression(kind="unresolved")
    return environment.get(reference, _SlotExpression(kind="unresolved"))


def _child_expression_environment(
    parent_environment: Mapping[str, _SlotExpression],
    child_skill: Skill,
    arguments: Mapping[str, str],
    *,
    loop_variable: Optional[str] = None,
    loop_expression: Optional[_SlotExpression] = None,
) -> dict[str, _SlotExpression]:
    child_environment: dict[str, _SlotExpression] = {}
    for slot in child_skill.slots:
        if slot.name in arguments:
            child_environment[slot.name] = _expression_for_argument(
                arguments[slot.name],
                parent_environment,
                loop_variable=loop_variable,
                loop_expression=loop_expression,
            )
        elif slot.name in parent_environment:
            child_environment[slot.name] = parent_environment[slot.name]
        else:
            child_environment[slot.name] = _SlotExpression(kind="absent")
    return child_environment


def _list_item_expression(expression: _SlotExpression) -> _SlotExpression:
    if expression.kind == "root-list":
        return _SlotExpression(kind="root-item", slot_name=expression.slot_name)
    return _SlotExpression(kind="unresolved")


def _composite_has_exact_task_cover(
    catalog: SkillCatalog,
    composite: Skill,
) -> bool:
    """Whether inverse packing would omit no executable composite step."""
    if not composite.steps:
        return False
    for step in composite.steps:
        if step.kind == "guidance":
            continue
        if step.kind == "commands" or step.skill is None:
            return False
        child = catalog.get(step.skill)
        if child is None:
            return False
        if (
            child.level == "composite"
            and not _composite_has_exact_task_cover(catalog, child)
        ):
            return False
    return True


def _composite_patterns(
    catalog: SkillCatalog, composite: Skill
) -> tuple[_CompositePattern, ...]:
    if not _composite_has_exact_task_cover(catalog, composite):
        return ()
    patterns: list[_CompositePattern] = []
    root_environment = {
        slot.name: _root_slot_expression(slot) for slot in composite.slots
    }

    def walk(
        skill: Skill,
        environment: Mapping[str, _SlotExpression],
        order_prefix: tuple[int, ...],
        group_path: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> None:
        for step in skill.steps:
            if step.kind in ("commands", "guidance") or step.skill is None:
                continue
            child_skill = catalog.get(step.skill)
            if child_skill is None:
                continue
            order = order_prefix + (step.ordinal,)
            iteration_expression: Optional[_SlotExpression] = None
            loop_expression: Optional[_SlotExpression] = None
            if step.kind == "for_each":
                iteration_expression = environment.get(
                    step.list_slot or "",
                    _SlotExpression(kind="unresolved"),
                )
                loop_expression = _list_item_expression(iteration_expression)
            child_environment = _child_expression_environment(
                environment,
                child_skill,
                step.arguments,
                loop_variable=step.loop_variable,
                loop_expression=loop_expression,
            )
            if child_skill.level == "composite":
                walk(
                    child_skill,
                    child_environment,
                    order,
                    group_path + ((child_skill.name, order),),
                )
                continue
            if child_skill.level not in ("task", "atomic"):
                continue
            patterns.append(
                _CompositePattern(
                    child_skill=child_skill.name,
                    child_expressions=tuple(sorted(child_environment.items())),
                    iteration_expression=iteration_expression,
                    order=order,
                    group_path=group_path,
                )
            )

    walk(
        composite,
        root_environment,
        (),
        ((composite.name, ()),),
    )
    return tuple(sorted(patterns, key=lambda pattern: (pattern.order, pattern.child_skill)))


def _normalise_assignment_value(
    value: Union[str, list[str], tuple[str, ...]],
) -> Union[str, tuple[str, ...]]:
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _match_expression(
    expression: _SlotExpression,
    value: Union[str, list[str], None],
) -> Optional[_DerivedAssignment]:
    if expression.kind == "literal":
        return (
            _DerivedAssignment("", "")
            if value == expression.literal
            else None
        )
    if expression.kind in ("unresolved", "absent"):
        return _DerivedAssignment("", "") if value is None else None
    if value is None or expression.slot_name is None:
        return None
    if expression.kind == "root-item":
        if isinstance(value, list):
            return None
        return _DerivedAssignment(
            expression.slot_name,
            str(value),
            append=True,
        )
    if expression.kind == "root-list":
        if not isinstance(value, list):
            return None
        return _DerivedAssignment(
            expression.slot_name,
            tuple(str(item) for item in value),
        )
    if expression.kind == "root-scalar":
        if isinstance(value, list):
            return None
        return _DerivedAssignment(expression.slot_name, str(value))
    return None


def _pattern_match(
    node: PlanNode,
    child_skill: Skill,
    pattern: _CompositePattern,
    *,
    node_position: int,
    pattern_index: int,
) -> Optional[_PatternMatch]:
    if node.skill != pattern.child_skill:
        return None
    expressions = dict(pattern.child_expressions)
    assignments: list[_DerivedAssignment] = []
    actual_names = set(node.bindings)
    for slot in child_skill.slots:
        expression = expressions.get(slot.name, _SlotExpression(kind="absent"))
        if slot.name not in actual_names:
            if expression.kind == "absent" and not slot.required:
                continue
            return None
        assignment = _match_expression(
            expression,
            node.bindings[slot.name].value,
        )
        if assignment is None:
            return None
        if assignment.slot_name:
            assignments.append(assignment)
    if actual_names - {slot.name for slot in child_skill.slots}:
        return None
    return _PatternMatch(
        node_position=node_position,
        pattern_index=pattern_index,
        assignments=tuple(assignments),
    )


def _merge_assignments(
    current: Mapping[str, Union[str, tuple[str, ...]]],
    additions: Sequence[_DerivedAssignment],
) -> Optional[dict[str, Union[str, tuple[str, ...]]]]:
    merged = dict(current)
    for addition in additions:
        existing = merged.get(addition.slot_name, _UNSET)
        if addition.append:
            if existing is _UNSET:
                merged[addition.slot_name] = (str(addition.value),)
            elif isinstance(existing, tuple):
                merged[addition.slot_name] = existing + (str(addition.value),)
            else:
                return None
            continue
        value = _normalise_assignment_value(addition.value)
        if existing is not _UNSET and existing != value:
            return None
        merged[addition.slot_name] = value
    return merged


def _expression_value(
    expression: _SlotExpression,
    assignments: Mapping[str, Union[str, tuple[str, ...]]],
    iteration_items: Mapping[str, str],
) -> Any:
    if expression.kind == "literal":
        return expression.literal
    if expression.kind in ("unresolved", "absent"):
        return _UNSET
    if expression.slot_name is None:
        return _UNSET
    if expression.kind == "root-item":
        return iteration_items.get(expression.slot_name, _UNSET)
    return assignments.get(expression.slot_name, _UNSET)


def _generated_pattern_keys(
    pattern: _CompositePattern,
    child_skill: Skill,
    assignments: Mapping[str, Union[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    iterations: tuple[Mapping[str, str], ...]
    if pattern.iteration_expression is None:
        iterations = ({},)
    else:
        source = _expression_value(pattern.iteration_expression, assignments, {})
        if source is _UNSET or source is None:
            return ()
        if not isinstance(source, tuple):
            return ()
        source_slot = pattern.iteration_expression.slot_name
        if source_slot is None:
            return ()
        iterations = tuple({source_slot: item} for item in source)

    keys: list[str] = []
    expressions = dict(pattern.child_expressions)
    for iteration_items in iterations:
        values: dict[str, Union[str, tuple[str, ...], None]] = {}
        for slot in child_skill.slots:
            expression = expressions.get(slot.name, _SlotExpression(kind="absent"))
            value = _expression_value(expression, assignments, iteration_items)
            if value is _UNSET:
                if expression.kind == "unresolved" or slot.required:
                    values[slot.name] = None
                continue
            values[slot.name] = value
        keys.append(_canonical_value_task_key(child_skill.name, values))
    return tuple(keys)


def _required_composite_slots_bound(
    composite: Skill,
    assignments: Mapping[str, Union[str, tuple[str, ...]]],
) -> bool:
    for slot in composite.required_slots:
        value = assignments.get(slot.name)
        if value is None or value == ():
            return False
    return True


def _candidate_from_seed(
    catalog: SkillCatalog,
    composite: Skill,
    patterns: Sequence[_CompositePattern],
    eligible_nodes: Sequence[PlanNode],
    matches_by_node: Mapping[int, tuple[_PatternMatch, ...]],
    seed: _PatternMatch,
) -> Optional[_CompositeCandidate]:
    """Build one maximal exact candidate under the seed's scalar bindings."""
    assignments = _merge_assignments({}, seed.assignments)
    if assignments is None:
        return None
    chosen: dict[int, _PatternMatch] = {seed.node_position: seed}
    for node_position in range(len(eligible_nodes)):
        if node_position in chosen:
            continue
        for match in matches_by_node.get(node_position, ()):
            merged = _merge_assignments(assignments, match.assignments)
            if merged is None:
                continue
            chosen[node_position] = match
            assignments = merged
            break

    ordered_matches = tuple(
        sorted(
            chosen.values(),
            key=lambda match: (
                patterns[match.pattern_index].order,
                match.node_position,
                match.pattern_index,
            ),
        )
    )
    canonical_assignments: dict[str, Union[str, tuple[str, ...]]] = {}
    for match in ordered_matches:
        merged = _merge_assignments(canonical_assignments, match.assignments)
        if merged is None:
            return None
        canonical_assignments = merged
    if len(ordered_matches) < _MIN_COMPOSITE_PACK_MEMBERS:
        return None
    if not _required_composite_slots_bound(composite, canonical_assignments):
        return None

    generated_keys: list[str] = []
    for pattern in patterns:
        child_skill = catalog.get(pattern.child_skill)
        if child_skill is None:
            return None
        generated_keys.extend(
            _generated_pattern_keys(
                pattern,
                child_skill,
                canonical_assignments,
            )
        )
    member_nodes = tuple(
        eligible_nodes[match.node_position] for match in ordered_matches
    )
    member_keys = tuple(node.task_key or "" for node in member_nodes)
    if not all(member_keys) or tuple(generated_keys) != member_keys:
        return None
    return _CompositeCandidate(
        composite_skill=composite.name,
        member_positions=tuple(match.node_position for match in ordered_matches),
        member_goal_ids=tuple(node.goal_id for node in member_nodes),
        member_task_keys=member_keys,
        bindings=tuple(sorted(canonical_assignments.items())),
        matches=ordered_matches,
    )


def _has_composite_ancestor(record: PlanRecord, node: PlanNode) -> bool:
    ancestor = record.node(node.parent_goal_id) if node.parent_goal_id else None
    while ancestor is not None:
        if ancestor.level == "composite":
            return True
        ancestor = (
            record.node(ancestor.parent_goal_id)
            if ancestor.parent_goal_id is not None
            else None
        )
    return False


def _packing_eligible_nodes(record: PlanRecord) -> tuple[PlanNode, ...]:
    return tuple(
        node
        for node in record.nodes
        if node.level == "task"
        and node.parent_goal_id is None
        and node.is_public
        and node.task_key is not None
        and not _has_composite_ancestor(record, node)
    )


def _composite_candidates(
    record: PlanRecord,
    catalog: SkillCatalog,
) -> tuple[
    tuple[_CompositeCandidate, ...],
    dict[str, tuple[_CompositePattern, ...]],
    tuple[PlanNode, ...],
]:
    eligible_nodes = _packing_eligible_nodes(record)
    candidates: dict[tuple[Any, ...], _CompositeCandidate] = {}
    patterns_by_composite: dict[str, tuple[_CompositePattern, ...]] = {}
    for composite_name in catalog:
        composite = catalog[composite_name]
        if composite.level != "composite":
            continue
        patterns = _composite_patterns(catalog, composite)
        patterns_by_composite[composite.name] = patterns
        if not patterns:
            continue
        matches_by_node: dict[int, tuple[_PatternMatch, ...]] = {}
        for node_position, node in enumerate(eligible_nodes):
            matches: list[_PatternMatch] = []
            for pattern_index, pattern in enumerate(patterns):
                child_skill = catalog.get(pattern.child_skill)
                if child_skill is None:
                    continue
                match = _pattern_match(
                    node,
                    child_skill,
                    pattern,
                    node_position=node_position,
                    pattern_index=pattern_index,
                )
                if match is not None:
                    matches.append(match)
            if matches:
                matches_by_node[node_position] = tuple(
                    sorted(
                        matches,
                        key=lambda match: (
                            patterns[match.pattern_index].order,
                            match.pattern_index,
                        ),
                    )
                )
        seeds = tuple(
            match
            for node_position in sorted(matches_by_node)
            for match in matches_by_node[node_position]
        )
        for seed in seeds:
            candidate = _candidate_from_seed(
                catalog,
                composite,
                patterns,
                eligible_nodes,
                matches_by_node,
                seed,
            )
            if candidate is not None:
                candidates[candidate.stable_key] = candidate
    return (
        tuple(sorted(candidates.values(), key=lambda candidate: candidate.stable_key)),
        patterns_by_composite,
        eligible_nodes,
    )


def _packing_is_better(
    proposed: tuple[int, ...],
    incumbent: tuple[int, ...],
    candidates: Sequence[_CompositeCandidate],
) -> bool:
    def primary(selection: tuple[int, ...]) -> tuple[int, int, int, int]:
        sizes = [len(candidates[index].member_positions) for index in selection]
        return (
            sum(sizes),
            sum(size * size for size in sizes),
            -len(sizes),
            -sum(candidates[index].order_inversions for index in selection),
        )

    proposed_primary = primary(proposed)
    incumbent_primary = primary(incumbent)
    if proposed_primary != incumbent_primary:
        return proposed_primary > incumbent_primary
    proposed_key = tuple(candidates[index].stable_key for index in proposed)
    incumbent_key = tuple(candidates[index].stable_key for index in incumbent)
    return proposed_key < incumbent_key


def _select_non_overlapping_candidates(
    candidates: Sequence[_CompositeCandidate],
    task_count: int,
) -> tuple[_CompositeCandidate, ...]:
    """Choose the deterministic maximum-coverage non-overlapping packing.

    The objective is lexicographic: tasks covered, concentration into larger
    groups, fewer groups, fewer task-order inversions, then the canonical
    composite/member/binding key. The memoized exact set-packing search matters:
    largest-candidate-first can choose one three-task group over two compatible
    two-task groups and leave a task unpacked.
    """
    if not candidates or task_count == 0:
        return ()
    masks = tuple(
        sum(1 << position for position in candidate.member_positions)
        for candidate in candidates
    )
    by_position: dict[int, tuple[int, ...]] = {
        position: tuple(
            index
            for index, mask in enumerate(masks)
            if mask & (1 << position)
        )
        for position in range(task_count)
    }

    @lru_cache(maxsize=None)
    def choose(remaining_mask: int) -> tuple[int, ...]:
        if remaining_mask == 0:
            return ()
        first_bit = remaining_mask & -remaining_mask
        first_position = first_bit.bit_length() - 1
        best = choose(remaining_mask ^ first_bit)
        for candidate_index in by_position.get(first_position, ()):
            candidate_mask = masks[candidate_index]
            if candidate_mask & remaining_mask != candidate_mask:
                continue
            remainder = choose(remaining_mask ^ candidate_mask)
            proposed = tuple(sorted((candidate_index, *remainder)))
            if _packing_is_better(proposed, best, candidates):
                best = proposed
        return best

    selected = choose((1 << task_count) - 1)
    return tuple(candidates[index] for index in selected)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _group_edges(member_goal_ids: Sequence[str]) -> tuple[PlanEdge, ...]:
    return tuple(
        PlanEdge(
            from_goal_id=left,
            to_goal_id=right,
            provenance="composite-pack",
        )
        for left, right in zip(member_goal_ids, member_goal_ids[1:])
    )


def _composite_group_payload(
    *,
    group_id: str,
    composite_skill: str,
    parent_group_id: Optional[str],
    depth: int,
    member_goal_ids: Sequence[str],
    member_task_keys: Sequence[str],
    shared_bindings: Mapping[str, Union[str, tuple[str, ...]]],
    orchestration_edges: Sequence[PlanEdge],
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "composite_skill": composite_skill,
        "parent_group_id": parent_group_id,
        "depth": depth,
        "member_goal_ids": tuple(member_goal_ids),
        "member_task_keys": tuple(member_task_keys),
        "shared_bindings": dict(sorted(shared_bindings.items())),
        "orchestration_edges": [
            edge.model_dump(mode="json") for edge in orchestration_edges
        ],
    }


def _composite_group(
    *,
    group_id: str,
    composite_skill: str,
    parent_group_id: Optional[str],
    depth: int,
    member_nodes: Sequence[PlanNode],
    shared_bindings: Mapping[str, Union[str, tuple[str, ...]]],
) -> CompositeGroup:
    member_goal_ids = tuple(node.goal_id for node in member_nodes)
    member_task_keys = tuple(node.task_key or "" for node in member_nodes)
    edges = _group_edges(member_goal_ids)
    signature_payload = _composite_group_payload(
        group_id=group_id,
        composite_skill=composite_skill,
        parent_group_id=parent_group_id,
        depth=depth,
        member_goal_ids=member_goal_ids,
        member_task_keys=member_task_keys,
        shared_bindings=shared_bindings,
        orchestration_edges=edges,
    )
    return CompositeGroup(
        group_id=group_id,
        composite_skill=composite_skill,
        parent_group_id=parent_group_id,
        depth=depth,
        member_goal_ids=member_goal_ids,
        member_task_keys=member_task_keys,
        shared_bindings=dict(sorted(shared_bindings.items())),
        orchestration_edges=edges,
        signature_sha256=_canonical_sha256(signature_payload),
    )


def _groups_for_candidate(
    candidate: _CompositeCandidate,
    *,
    root_ordinal: int,
    eligible_nodes: Sequence[PlanNode],
    patterns: Sequence[_CompositePattern],
) -> tuple[CompositeGroup, ...]:
    root_group_id = f"pack-{root_ordinal}"
    bindings = dict(candidate.bindings)
    matches = candidate.matches
    root_nodes = tuple(
        eligible_nodes[match.node_position] for match in matches
    )
    groups: list[CompositeGroup] = [
        _composite_group(
            group_id=root_group_id,
            composite_skill=candidate.composite_skill,
            parent_group_id=None,
            depth=1,
            member_nodes=root_nodes,
            shared_bindings=bindings,
        )
    ]

    nested_members: dict[
        tuple[tuple[str, tuple[int, ...]], ...],
        list[PlanNode],
    ] = {}
    for match in matches:
        pattern = patterns[match.pattern_index]
        node = eligible_nodes[match.node_position]
        for depth in range(2, len(pattern.group_path) + 1):
            prefix = pattern.group_path[:depth]
            members = nested_members.setdefault(prefix, [])
            if node not in members:
                members.append(node)

    path_ids: dict[
        tuple[tuple[str, tuple[int, ...]], ...],
        str,
    ] = {((candidate.composite_skill, ()),): root_group_id}
    child_counts: Counter[str] = Counter()
    for path in sorted(nested_members, key=lambda item: (len(item), item)):
        parent_path = path[:-1]
        parent_group_id = path_ids.get(parent_path, root_group_id)
        child_counts[parent_group_id] += 1
        group_id = f"{parent_group_id}.{child_counts[parent_group_id]}"
        path_ids[path] = group_id
        groups.append(
            _composite_group(
                group_id=group_id,
                composite_skill=path[-1][0],
                parent_group_id=parent_group_id,
                depth=len(path),
                member_nodes=nested_members[path],
                shared_bindings=bindings,
            )
        )
    return tuple(groups)


def _synthesize_composite_packing(
    record: PlanRecord,
    catalog: SkillCatalog,
) -> None:
    candidates, patterns_by_composite, eligible_nodes = _composite_candidates(
        record,
        catalog,
    )
    selected = _select_non_overlapping_candidates(
        candidates,
        len(eligible_nodes),
    )
    selected = tuple(
        sorted(
            selected,
            key=lambda candidate: (
                min(candidate.member_positions),
                candidate.stable_key,
            ),
        )
    )
    groups: list[CompositeGroup] = []
    packed_positions: set[int] = set()
    for root_ordinal, candidate in enumerate(selected, start=1):
        packed_positions.update(candidate.member_positions)
        groups.extend(
            _groups_for_candidate(
                candidate,
                root_ordinal=root_ordinal,
                eligible_nodes=eligible_nodes,
                patterns=patterns_by_composite[candidate.composite_skill],
            )
        )
    group_tuple = tuple(groups)
    packing_payload = [
        group.model_dump(mode="json") for group in group_tuple
    ]
    record.composite_groups = group_tuple
    record.packing = CompositePackingMetrics(
        candidate_count=len(candidates),
        selected_root_group_count=len(selected),
        selected_recursive_group_count=max(0, len(group_tuple) - len(selected)),
        packed_task_count=len(packed_positions),
        unpacked_task_count=len(eligible_nodes) - len(packed_positions),
        orchestration_edge_count=sum(
            len(group.orchestration_edges) for group in group_tuple
        ),
        shared_binding_count=sum(
            len(group.shared_bindings)
            for group in group_tuple
            if group.parent_group_id is None
        ),
        packing_sha256=_canonical_sha256(packing_payload),
    )


# ----------------------------------------------------------------------
# Deterministic expansion
# ----------------------------------------------------------------------


def expand(
    catalog: SkillCatalog,
    invocations: Sequence[Invocation],
    utterance: str,
    *,
    plan_id: Optional[str] = None,
    mode: Union[PlanMode, str] = PlanMode.ENFORCE,
    selection_model: Optional[str] = None,
    budget_limit: Optional[int] = None,
) -> PlanRecord:
    """Walk the selected invocations into a plan. No model call, at any depth.

    Each invocation becomes a task- or composite-level goal node whose
    `goal_text` is the skill's `goal` template with its bindings substituted.
    Expansion then walks that skill's body: a step naming a child skill becomes
    a child node with slots bound from the parent; a `for each` step fans out
    one child per element of the bound list, in the order the utterance gave
    them; every other step is a command-sequence leaf. Atomic nodes are leaves.

    Depth is re-checked here and not only at load (FW-REQ-012 clauses 1 and 3):
    the catalogue bounds the `uses` graph, and it is expansion that turns that
    graph into a tree.
    """
    record = PlanRecord(
        plan_id=plan_id or uuid.uuid4().hex,
        skills_fingerprint=getattr(catalog, "fingerprint", None),
        selection_model=selection_model,
        mode=_record_mode(mode),
        budget_limit=budget_limit,
    )
    record._catalog = catalog  # noqa: SLF001 - see _catalog's declaration
    record._utterance = utterance

    requested_public_keys = _requested_public_task_keys(
        catalog, invocations, utterance
    )
    duplicate_requested = sorted(
        key
        for key, count in Counter(requested_public_keys).items()
        if count > 1
    )
    if duplicate_requested:
        raise PlanConfigurationError(
            "duplicate task coverage in requested public task keys: "
            + ", ".join(duplicate_requested)
        )
    record.requested_public_task_keys = tuple(requested_public_keys)

    previous_root_goal_id: Optional[str] = None
    for index, invocation in enumerate(invocations, start=1):
        skill = catalog.get(invocation.skill_name)
        if skill is None:
            # FW-REQ-011 clause 10: an ad-hoc fallback is recorded, as a node
            # with `skill: None`, and nothing promotes it. Growing the
            # catalogue from runs unreviewed is what §14.1 rejects.
            record.add_node(
                PlanNode(
                    goal_id=f"g{index}",
                    level="commands",
                    skill=None,
                    goal_text=invocation.skill_name,
                    visibility="private",
                    bindings=_literal_bindings(invocation.slots, utterance),
                    status="blocked",
                    failure_reason=(
                        f"selector named '{invocation.skill_name}', which is "
                        "not in the catalogue"
                    ),
                )
            )
            continue
        _validate_invocation(skill, invocation, utterance)
        bindings = _invocation_bindings(skill, invocation, utterance)
        node = record.add_node(
            PlanNode(
                goal_id=f"g{index}",
                level=_node_level(skill),
                skill=skill.name,
                goal_text="",
                executable_goal_text=None,
                executable=False,
                visibility=_visibility(skill.level),
                task_key=_canonical_task_key(skill.name, bindings),
                bindings=bindings,
                status="needs-user" if _needs_user(bindings) else "not-reached",
            )
        )
        node.goal_text = (
            _invocation_goal_text(skill, node.bindings)
            if skill.level == "atomic"
            else _render(skill.goal or skill.description, node.bindings)
        )
        _set_executable_state(node)
        if previous_root_goal_id is not None:
            record.add_edge(previous_root_goal_id, node.goal_id, "stable-tiebreak")
        previous_root_goal_id = node.goal_id
        if not _awaiting(node):
            _expand_children(record, node, catalog)
    compiled_public_keys = tuple(
        node.task_key for node in record.public_nodes if node.task_key is not None
    )
    record.compiled_public_task_keys = compiled_public_keys
    if Counter(requested_public_keys) != Counter(compiled_public_keys):
        raise PlanConfigurationError(
            "requested public task-key multiset does not match compiled public "
            "task-key multiset"
        )
    _synthesize_composite_packing(record, catalog)
    validate_compiled_plan(record)
    return record


def _expand_children(
    record: PlanRecord, node: PlanNode, catalog: SkillCatalog
) -> tuple[PlanNode, ...]:
    """Expand one node's body into its children. Idempotent by construction:
    the caller expands a node exactly once, when its slots are bound."""
    if node.level in LEAF_LEVELS:
        return ()
    skill = catalog.get(node.skill or "")
    if skill is None:
        return ()
    if record.depth_of(node.goal_id) >= MAX_DEPTH:
        # Re-checked at expansion, not only at load. A catalogue whose `uses`
        # graph is three deep can still be walked into a deeper tree by a
        # future step form; this is where that would be caught.
        raise PlanConfigurationError(
            f"expanding node {node.goal_id} ({node.skill}) at depth "
            f"{record.depth_of(node.goal_id)} would exceed maximum plan depth "
            f"{MAX_DEPTH}"
        )

    created: list[PlanNode] = []
    if not skill.steps:
        # A body with no numbered steps is one command sequence. It is not
        # nothing: `control-failure-investigation` is prose today, and dropping
        # it would silently plan a task node with no work under it.
        created.append(
            _add_child(
                record,
                node,
                ordinal=1,
                level="commands",
                skill=None,
                text=skill.body.strip(),
                bindings=node.bindings,
            )
        )
        _chain(record, created)
        return tuple(created)

    previous_step: tuple[PlanNode, ...] = ()
    for step in skill.steps:
        if step.kind == "guidance":
            if not previous_step:
                raise PlanConfigurationError(
                    f"skill {skill.name} step {step.ordinal} is guidance with "
                    "no preceding executable step"
                )
            guidance = _render(step.text, node.bindings)
            for prior in previous_step:
                prior.goal_text = f"{prior.goal_text}\n\n{guidance}"
                _set_executable_state(prior)
            continue
        current_step = tuple(_expand_step(record, node, skill, step, catalog))
        prerequisites = tuple(item.goal_id for item in previous_step)
        for child in current_step:
            child.prerequisites = prerequisites
            child.prerequisite_provenance = {
                goal_id: "explicit-order" for goal_id in prerequisites
            }
            for goal_id in prerequisites:
                record.add_edge(goal_id, child.goal_id, "explicit-order")
        created.extend(current_step)
        if current_step:
            previous_step = current_step
    return tuple(created)


def _expand_step(
    record: PlanRecord,
    node: PlanNode,
    skill: Skill,
    step: Step,
    catalog: SkillCatalog,
) -> list[PlanNode]:
    if step.kind == "commands" or step.skill is None:
        return [
            _add_child(
                record,
                node,
                ordinal=step.ordinal,
                level="commands",
                skill=None,
                text=step.text,
                bindings=node.bindings,
            )
        ]

    child_skill = catalog.get(step.skill)
    if child_skill is None:  # pragma: no cover - the loader rejects this
        return [
            _add_child(
                record,
                node,
                ordinal=step.ordinal,
                level="commands",
                skill=None,
                text=step.text,
                bindings={},
            )
        ]

    if step.kind == "for_each":
        return _fan_out(record, node, step, child_skill, catalog)

    bindings = _child_bindings(node, child_skill, step.arguments)
    child = _add_child(
        record,
        node,
        ordinal=step.ordinal,
        level=_node_level(child_skill),
        skill=child_skill.name,
        text=_render(step.text, node.bindings),
        bindings=bindings,
        goal_template=child_skill.goal,
        visibility=_visibility(child_skill.level),
    )
    if _step_has_data_dependency(step, node):
        record.add_edge(node.goal_id, child.goal_id, "data")
    if not _awaiting(child):
        _expand_children(record, child, catalog)
    return [child]


def _fan_out(
    record: PlanRecord,
    node: PlanNode,
    step: Step,
    child_skill: Skill,
    catalog: SkillCatalog,
) -> list[PlanNode]:
    """`for each {x} in {xs}: <child> <slot>={x}` — one child per element.

    The review amendment's step form. A `composite` authored for a recurring
    batch shape cannot name three children it does not know the names of; this
    is how it says the same thing the selector says with three sibling
    invocations. No model call is involved: the list is bound at selection like
    any other slot, and the order is the order the utterance gave.
    """
    binding = node.bindings.get(step.list_slot or "")
    if binding is None:
        return []
    values = binding.value if binding is not None else None
    if not isinstance(values, list):
        # The width of the fan-out is unknown until the list is bound, so the
        # plan records one node for the step and `bind_captured` widens it.
        child = _add_child(
            record,
            node,
            ordinal=step.ordinal,
            level=_node_level(child_skill),
            skill=child_skill.name,
            text=_render(step.text, node.bindings),
            bindings=_child_bindings(
                node, child_skill, step.arguments, loop_variable=step.loop_variable
            ),
            goal_template=child_skill.goal,
            visibility=_visibility(child_skill.level),
        )
        if _step_has_data_dependency(step, node):
            record.add_edge(node.goal_id, child.goal_id, "data")
        return [child]

    created: list[PlanNode] = []
    for position, element in enumerate(values):
        bindings = _child_bindings(
            node,
            child_skill,
            step.arguments,
            loop_variable=step.loop_variable,
            loop_value=element,
            loop_binding=binding,
            loop_index=position,
        )
        child = _add_child(
            record,
            node,
            ordinal=step.ordinal,
            level=_node_level(child_skill),
            skill=child_skill.name,
            text=_render(step.text, node.bindings),
            bindings=bindings,
            goal_template=child_skill.goal,
            visibility=_visibility(child_skill.level),
            suffix=f"{step.ordinal}.{position + 1}",
        )
        if _step_has_data_dependency(step, node):
            record.add_edge(node.goal_id, child.goal_id, "data")
        if not _awaiting(child):
            _expand_children(record, child, catalog)
        created.append(child)
    return created


def _add_child(
    record: PlanRecord,
    parent: PlanNode,
    *,
    ordinal: int,
    level: str,
    skill: Optional[str],
    text: str,
    bindings: Mapping[str, Binding],
    goal_template: Optional[str] = None,
    visibility: str = "private",
    suffix: Optional[str] = None,
) -> PlanNode:
    goal_id = f"{parent.goal_id}.{suffix or ordinal}"
    while record.node(goal_id) is not None:
        goal_id = f"{goal_id}'"
    node = PlanNode(
        goal_id=goal_id,
        parent_goal_id=parent.goal_id,
        level=level,  # type: ignore[arg-type]
        skill=skill,
        goal_text=_render(goal_template or text, bindings),
        visibility=visibility,  # type: ignore[arg-type]
        task_key=(
            _canonical_task_key(skill, bindings)
            if skill is not None and visibility == "public"
            else None
        ),
        bindings=dict(bindings),
        status="needs-user" if _needs_user(bindings) else "not-reached",
    )
    _set_executable_state(node)
    created = record.add_node(node)
    record.add_edge(parent.goal_id, created.goal_id, "composite")
    return created


def _chain(record: PlanRecord, nodes: Sequence[PlanNode]) -> None:
    """Sequential prerequisites within one node's children (decision 3).

    Across sibling task nodes there are no edges at all — three leavers are
    independent — which is why this is called per parent and never per plan.
    """
    for previous, node in zip(nodes, nodes[1:]):
        node.prerequisites = (previous.goal_id,)
        node.prerequisite_provenance = {previous.goal_id: "explicit-order"}
        record.add_edge(previous.goal_id, node.goal_id, "explicit-order")


def _set_executable_state(node: PlanNode) -> None:
    executable = (
        node.level in LEAF_LEVELS
        and not _awaiting(node)
        and not _has_unresolved_placeholder(node.goal_text)
    )
    node.executable_goal_text = node.goal_text if executable else None
    node.executable = executable


def _validate_composite_packing(record: PlanRecord, known: set[str]) -> None:
    groups = record.composite_groups
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise PlanConfigurationError("composite packing repeats a group_id")
    by_group_id = {group.group_id: group for group in groups}
    eligible_nodes = _packing_eligible_nodes(record)
    eligible_ids = {node.goal_id for node in eligible_nodes}
    root_member_ids: set[str] = set()
    packing_adjacency: dict[str, set[str]] = {
        goal_id: set() for goal_id in known
    }

    for group in groups:
        if group.parent_group_id is None:
            if group.depth != 1:
                raise PlanConfigurationError(
                    f"root composite group {group.group_id} must have depth 1"
                )
            if len(group.member_goal_ids) < _MIN_COMPOSITE_PACK_MEMBERS:
                raise PlanConfigurationError(
                    f"root composite group {group.group_id} has fewer than "
                    f"{_MIN_COMPOSITE_PACK_MEMBERS} task members"
                )
            overlap = root_member_ids & set(group.member_goal_ids)
            if overlap:
                raise PlanConfigurationError(
                    "selected composite groups overlap task nodes: "
                    + ", ".join(sorted(overlap))
                )
            root_member_ids.update(group.member_goal_ids)
        else:
            parent = by_group_id.get(group.parent_group_id)
            if parent is None:
                raise PlanConfigurationError(
                    f"composite group {group.group_id} names unknown parent "
                    f"{group.parent_group_id}"
                )
            if group.depth != parent.depth + 1:
                raise PlanConfigurationError(
                    f"composite group {group.group_id} depth does not follow "
                    f"parent {parent.group_id}"
                )
            if not set(group.member_goal_ids) <= set(parent.member_goal_ids):
                raise PlanConfigurationError(
                    f"composite group {group.group_id} contains a task outside "
                    f"parent {parent.group_id}"
                )

        if not set(group.member_goal_ids) <= eligible_ids:
            raise PlanConfigurationError(
                f"composite group {group.group_id} references an ineligible "
                "or unknown public task node"
            )
        expected_task_keys = tuple(
            (record.node(goal_id).task_key if record.node(goal_id) else None)
            for goal_id in group.member_goal_ids
        )
        if expected_task_keys != group.member_task_keys:
            raise PlanConfigurationError(
                f"composite group {group.group_id} task keys do not match "
                "its member nodes"
            )
        expected_edges = _group_edges(group.member_goal_ids)
        if group.orchestration_edges != expected_edges:
            raise PlanConfigurationError(
                f"composite group {group.group_id} orchestration edges are "
                "not its deterministic member chain"
            )
        expected_signature = _canonical_sha256(
            _composite_group_payload(
                group_id=group.group_id,
                composite_skill=group.composite_skill,
                parent_group_id=group.parent_group_id,
                depth=group.depth,
                member_goal_ids=group.member_goal_ids,
                member_task_keys=group.member_task_keys,
                shared_bindings=group.shared_bindings,
                orchestration_edges=group.orchestration_edges,
            )
        )
        if group.signature_sha256 != expected_signature:
            raise PlanConfigurationError(
                f"composite group {group.group_id} signature is not deterministic"
            )
        for edge in group.orchestration_edges:
            if edge.from_goal_id not in known or edge.to_goal_id not in known:
                raise PlanConfigurationError(
                    f"composite group edge {edge.from_goal_id}->{edge.to_goal_id} "
                    "references an unknown node"
                )
            packing_adjacency[edge.from_goal_id].add(edge.to_goal_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visiting:
            raise PlanConfigurationError(
                f"composite packing contains a cycle through {goal_id}"
            )
        if goal_id in visited:
            return
        visiting.add(goal_id)
        for child_id in packing_adjacency[goal_id]:
            visit(child_id)
        visiting.remove(goal_id)
        visited.add(goal_id)

    for goal_id in known:
        visit(goal_id)

    root_groups = tuple(
        group for group in groups if group.parent_group_id is None
    )
    expected_metrics = {
        "selected_root_group_count": len(root_groups),
        "selected_recursive_group_count": len(groups) - len(root_groups),
        "packed_task_count": len(root_member_ids),
        "unpacked_task_count": len(eligible_nodes) - len(root_member_ids),
        "orchestration_edge_count": sum(
            len(group.orchestration_edges) for group in groups
        ),
        "shared_binding_count": sum(
            len(group.shared_bindings) for group in root_groups
        ),
    }
    for field_name, expected in expected_metrics.items():
        if getattr(record.packing, field_name) != expected:
            raise PlanConfigurationError(
                f"composite packing metric {field_name} is "
                f"{getattr(record.packing, field_name)}, expected {expected}"
            )
    if record.packing.candidate_count < len(root_groups):
        raise PlanConfigurationError(
            "composite packing selected more root groups than candidates"
        )
    expected_packing_sha = _canonical_sha256(
        [group.model_dump(mode="json") for group in groups]
    )
    if (
        record.packing.packing_sha256
        and record.packing.packing_sha256 != expected_packing_sha
    ):
        raise PlanConfigurationError(
            "composite packing serialization digest does not match its groups"
        )

    execution = record.execution
    if execution is not None:
        unknown_execution_groups = (
            set(execution.composite_group_ids) - set(group_ids)
        )
        if unknown_execution_groups:
            raise PlanConfigurationError(
                "execution metadata references unknown composite groups: "
                + ", ".join(sorted(unknown_execution_groups))
            )
        if Counter(execution.public_task_keys) != Counter(
            record.compiled_public_task_keys
        ):
            raise PlanConfigurationError(
                "execution metadata public task keys drift from compiled coverage"
            )
        unknown_schedule_nodes = (
            set(execution.scheduled_leaf_goal_ids)
            | set(execution.scheduled_task_goal_ids)
            | set(execution.executed_leaf_goal_ids)
        ) - known
        if unknown_schedule_nodes:
            raise PlanConfigurationError(
                "execution metadata references unknown plan nodes: "
                + ", ".join(sorted(unknown_schedule_nodes))
            )


def validate_compiled_plan(record: PlanRecord) -> None:
    """Enforce coverage, overlap, DAG, and rendered-goal invariants."""
    node_ids = [node.goal_id for node in record.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise PlanConfigurationError("compiled plan repeats a goal_id")
    known = set(node_ids)

    requested = Counter(record.requested_public_task_keys)
    compiled = Counter(record.compiled_public_task_keys)
    if requested != compiled:
        raise PlanConfigurationError(
            "requested public task-key multiset does not match compiled public "
            "task-key multiset"
        )
    duplicated = sorted(key for key, count in compiled.items() if count > 1)
    if duplicated:
        raise PlanConfigurationError(
            "compiled public task coverage overlaps: " + ", ".join(duplicated)
        )

    adjacency: dict[str, set[str]] = {goal_id: set() for goal_id in known}
    edge_pairs = set()
    for edge in record.edges:
        if edge.from_goal_id not in known or edge.to_goal_id not in known:
            raise PlanConfigurationError(
                f"plan edge {edge.from_goal_id}->{edge.to_goal_id} "
                "references an unknown node"
            )
        if edge.from_goal_id == edge.to_goal_id:
            raise PlanConfigurationError(
                f"plan edge {edge.from_goal_id}->{edge.to_goal_id} is a self-cycle"
            )
        adjacency[edge.from_goal_id].add(edge.to_goal_id)
        edge_pairs.add((edge.from_goal_id, edge.to_goal_id))

    for node in record.nodes:
        if set(node.prerequisite_provenance) != set(node.prerequisites):
            raise PlanConfigurationError(
                f"node {node.goal_id} prerequisite provenance is incomplete"
            )
        for prerequisite in node.prerequisites:
            if prerequisite not in known:
                raise PlanConfigurationError(
                    f"node {node.goal_id} requires unknown node {prerequisite}"
                )
            if (prerequisite, node.goal_id) not in edge_pairs:
                raise PlanConfigurationError(
                    f"node {node.goal_id} prerequisite {prerequisite} "
                    "has no matching DAG edge"
                )

    _validate_composite_packing(record, known)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visiting:
            raise PlanConfigurationError(
                f"compiled execution DAG contains a cycle through {goal_id}"
            )
        if goal_id in visited:
            return
        visiting.add(goal_id)
        for child_id in adjacency[goal_id]:
            visit(child_id)
        visiting.remove(goal_id)
        visited.add(goal_id)

    for goal_id in node_ids:
        visit(goal_id)

    for node in record.nodes:
        if (
            (node.is_leaf or node.is_public)
            and not _awaiting(node)
            and _has_unresolved_placeholder(node.goal_text)
        ):
            raise PlanConfigurationError(
                f"node {node.goal_id} has bound inputs but an unresolved goal: "
                f"{node.goal_text!r}"
            )
        if node.is_leaf and not _awaiting(node) and not node.executable:
            raise PlanConfigurationError(
                f"leaf {node.goal_id} has bound inputs but no executable goal"
            )


def _has_unresolved_placeholder(text: str) -> bool:
    return re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", text) is not None


def _step_has_data_dependency(step: Step, parent: PlanNode) -> bool:
    for raw in step.arguments.values():
        if not (raw.startswith("{") and raw.endswith("}")):
            continue
        reference = raw[1:-1].strip()
        if reference == step.loop_variable:
            return True
        if reference in parent.bindings:
            return True
    return False


# ----------------------------------------------------------------------
# Binding
# ----------------------------------------------------------------------


def _invocation_bindings(
    skill: Skill, invocation: Invocation, utterance: str
) -> dict[str, Binding]:
    """Bind supplied values with typed evidence; leave repeat policy as policy."""
    supplied = _validate_invocation(skill, invocation, utterance)
    bindings: dict[str, Binding] = {}
    for slot in skill.slots:
        if slot.name in supplied:
            bindings[slot.name] = supplied[slot.name]
        elif slot.required:
            bindings[slot.name] = _fallback(slot)
    return bindings


def _validate_invocation(
    skill: Skill, invocation: Invocation, utterance: str
) -> dict[str, Binding]:
    declared = {slot.name: slot for slot in skill.slots}
    unknown_evidence = set(invocation.provenance) - set(invocation.slots)
    if unknown_evidence:
        names = ", ".join(sorted(unknown_evidence))
        raise PlanConfigurationError(
            f"invocation of '{skill.name}' has provenance for unbound slot(s): "
            f"{names}"
        )

    bindings: dict[str, Binding] = {}
    for name, value in invocation.slots.items():
        slot = declared.get(name)
        if slot is None:
            raise PlanConfigurationError(
                f"invocation of '{skill.name}' binds undeclared slot '{name}'"
            )
        is_list = isinstance(value, list)
        if slot.list != is_list:
            expected = "a list" if slot.list else "one string"
            raise PlanConfigurationError(
                f"invocation of '{skill.name}' slot '{name}' requires {expected}"
            )
        bindings[name] = _invocation_binding(
            skill, slot, value, invocation.provenance.get(name), utterance
        )
    return bindings


def _invocation_binding(
    skill: Skill,
    slot: Any,
    value: Union[str, list[str]],
    evidence: Optional[InvocationEvidence],
    utterance: str,
) -> Binding:
    values = value if isinstance(value, list) else [value]
    if any(not item for item in values):
        raise PlanConfigurationError(
            f"invocation of '{skill.name}' slot '{slot.name}' has an empty value"
        )

    if evidence is None:
        kind = slot.binding_kind
        if kind == "exact_text":
            spans = _find_exact_source_spans(
                skill.name, slot.name, values, utterance
            )
            evidence = InvocationEvidence(kind=kind, source_spans=spans)
        elif kind == "normalized_enum":
            try:
                spans = _find_exact_source_spans(
                    skill.name, slot.name, values, utterance
                )
            except PlanConfigurationError as exc:
                raise PlanConfigurationError(
                    f"invocation of '{skill.name}' normalized slot "
                    f"'{slot.name}' requires exact source span(s) and normalizer "
                    f"{slot.normalizer!r}"
                ) from exc
            evidence = InvocationEvidence(
                kind=kind,
                source_spans=spans,
                normalizer=slot.normalizer,
            )
        else:
            raise PlanConfigurationError(
                f"invocation of '{skill.name}' slot '{slot.name}' with "
                f"binding_kind {kind!r} requires explicit provenance"
            )

    if evidence.kind == "skill_literal":
        raise PlanConfigurationError(
            f"invocation of '{skill.name}' slot '{slot.name}' cannot claim "
            "skill_literal provenance; only a parsed skill body can supply it"
        )
    if evidence.kind != slot.binding_kind and evidence.kind != "captured_handle":
        raise PlanConfigurationError(
            f"invocation of '{skill.name}' slot '{slot.name}' declares "
            f"binding_kind {slot.binding_kind!r}, not {evidence.kind!r}"
        )

    _validate_source_spans(skill.name, slot.name, evidence.source_spans, utterance)
    if evidence.kind == "exact_text":
        for item, span in zip(values, evidence.source_spans):
            if span.text != item:
                raise PlanConfigurationError(
                    f"invocation of '{skill.name}' slot '{slot.name}' value "
                    f"{item!r} is not copied verbatim from source span "
                    f"[{span.start}, {span.end})"
                )
        source: BindingSource = "utterance"
    elif evidence.kind == "normalized_enum":
        if evidence.normalizer != slot.normalizer:
            raise PlanConfigurationError(
                f"invocation of '{skill.name}' slot '{slot.name}' must use "
                f"normalizer {slot.normalizer!r}, got {evidence.normalizer!r}"
            )
        for item, span in zip(values, evidence.source_spans):
            normalized = normalize_binding_value(evidence.normalizer or "", span.text)
            if normalized != item:
                raise PlanConfigurationError(
                    f"invocation of '{skill.name}' slot '{slot.name}' source "
                    f"{span.text!r} normalizes to {normalized!r}, not {item!r}"
                )
        source = "utterance"
    else:
        source = "captured"

    try:
        return Binding(
            value=value,
            source=source,
            kind=evidence.kind,
            source_spans=evidence.source_spans,
            normalizer=evidence.normalizer,
            command_call_id=evidence.command_call_id,
        )
    except ValueError as exc:
        raise PlanConfigurationError(
            f"invocation of '{skill.name}' slot '{slot.name}' has invalid "
            f"{evidence.kind} provenance: {exc}"
        ) from exc


def _find_exact_source_spans(
    skill_name: str,
    slot_name: str,
    values: Sequence[str],
    utterance: str,
) -> tuple[SourceSpan, ...]:
    spans: list[SourceSpan] = []
    cursor = 0
    for value in values:
        start = utterance.find(value, cursor)
        if start < 0:
            raise PlanConfigurationError(
                f"invocation of '{skill_name}' slot '{slot_name}' value "
                f"{value!r} is not copied verbatim from the utterance"
            )
        end = start + len(value)
        spans.append(SourceSpan(start=start, end=end, text=utterance[start:end]))
        cursor = end
    return tuple(spans)


def _validate_source_spans(
    skill_name: str,
    slot_name: str,
    spans: Sequence[SourceSpan],
    utterance: str,
) -> None:
    previous_end = -1
    for span in spans:
        if span.end > len(utterance) or utterance[span.start : span.end] != span.text:
            raise PlanConfigurationError(
                f"invocation of '{skill_name}' slot '{slot_name}' source span "
                f"[{span.start}, {span.end}) does not match the utterance"
            )
        if span.start < previous_end:
            raise PlanConfigurationError(
                f"invocation of '{skill_name}' slot '{slot_name}' source spans "
                "overlap or are out of order"
            )
        previous_end = span.end


def _child_bindings(
    parent: PlanNode,
    child_skill: Skill,
    arguments: Mapping[str, str],
    *,
    loop_variable: Optional[str] = None,
    loop_value: Optional[str] = None,
    loop_binding: Optional[Binding] = None,
    loop_index: Optional[int] = None,
) -> dict[str, Binding]:
    """Bind a child's slots from the parent's bindings — never from invention.

    `entity_type=identity` is a value the body supplies (`skill`);
    `query={identity_query}` resolves against the parent's bindings and carries
    the parent binding's source with it, so a value that reached the plan from
    the utterance is still recorded as having.
    """
    bindings: dict[str, Binding] = {}
    for slot_name, raw in arguments.items():
        if not (raw.startswith("{") and raw.endswith("}")):
            bindings[slot_name] = Binding(
                value=raw,
                source="skill",
                kind="skill_literal",
            )
            continue
        reference = raw[1:-1].strip()
        if loop_variable is not None and reference == loop_variable:
            if loop_value is None:
                child_slot = child_skill.slot(slot_name)
                bindings[slot_name] = Binding(
                    value=None,
                    source="needs-user",
                    on_repeat_policy=(
                        child_slot.on_repeat if child_slot is not None else None
                    ),
                )
            else:
                bindings[slot_name] = (
                    _binding_list_item(loop_binding, loop_value, loop_index)
                    if loop_binding is not None
                    else Binding(
                        value=loop_value,
                        source="skill",
                        kind="skill_literal",
                    )
                )
            continue
        inherited = parent.bindings.get(reference)
        if inherited is not None and inherited.value is not None:
            bindings[slot_name] = inherited.model_copy(deep=True)
            continue
        # An explicit placeholder says a prior step is expected to supply this
        # value. Keep it pending so captured evidence gets precedence over the
        # child's on_repeat fallback; applying that fallback now would make the
        # producing command's later artifact impossible to bind.
        child_slot = child_skill.slot(slot_name)
        bindings[slot_name] = Binding(
            value=None,
            source="needs-user",
            on_repeat_policy=(
                child_slot.on_repeat if child_slot is not None else None
            ),
        )

    for slot in child_skill.slots:
        if slot.name in bindings:
            continue
        inherited = parent.bindings.get(slot.name)
        if inherited is not None and inherited.value is not None:
            bindings[slot.name] = inherited.model_copy(deep=True)
        elif slot.required:
            bindings[slot.name] = _fallback(slot)
    return bindings


def _binding_list_item(
    binding: Binding, value: str, index: Optional[int] = None
) -> Binding:
    source_spans = binding.source_spans
    if isinstance(binding.value, list) and source_spans:
        if index is None:
            try:
                index = binding.value.index(value)
            except ValueError:
                index = -1
        source_spans = (
            (source_spans[index],)
            if index is not None and 0 <= index < len(source_spans)
            else source_spans
        )
    return Binding(
        value=value,
        source=binding.source,
        kind=binding.kind,
        source_spans=source_spans,
        normalizer=binding.normalizer,
        command_call_id=binding.command_call_id,
    )


def _fallback(slot) -> Binding:
    """Record a missing value and its repeat policy without using policy as data."""
    return Binding(
        value=None,
        source="needs-user",
        on_repeat_policy=getattr(slot, "on_repeat", None),
    )


def _literal_bindings(
    slots: Mapping[str, Any], utterance: str
) -> dict[str, Binding]:
    return {
        name: Binding(
            value=value,
            source="utterance",
            kind="exact_text",
            source_spans=_find_exact_source_spans(
                "<unknown-skill>",
                name,
                value if isinstance(value, list) else [value],
                utterance,
            ),
        )
        for name, value in slots.items()
    }


def _needs_user(bindings: Mapping[str, Binding]) -> bool:
    return any(
        binding.source == "needs-user" and binding.value is None
        for binding in bindings.values()
    )


def _awaiting(node: PlanNode) -> bool:
    """Whether this node's children wait on a value it does not have yet.

    Decision 3's delayed expansion: the node exists, and only slot *values* are
    late — never the set of children.
    """
    return _needs_user(node.bindings)


def bind_captured(
    record: PlanRecord,
    node: PlanNode,
    kind: str,
    artifacts: Iterable[Any],
    *,
    slot: Optional[str] = None,
    catalog: Optional[SkillCatalog] = None,
) -> Optional[Binding]:
    """Precedence 2: bind a slot from a handle an already-executed leaf produced.

    `artifacts` is the executed leaves' outputs in execution order — anything
    exposing `command_call_id` and `command_response.artifacts` (a
    `CommandOutput`), or a `(command_call_id, artifacts)` pair, or a bare dict.

    **The first uid of the requested kind, in execution order.** First, not
    "best": choosing among them is a decision, and a decision here is either a
    model call (barred by P-01) or a heuristic nobody measured.
    `leaver-offboarding-sweep`'s "pick one account uid" becomes "the first
    account on the portrait", which is `ido-j0c.11` defect 1's own
    recommendation.

    The manifest's per-command `capture` block is deliberately not consulted:
    it is `dict[field_name, DataClassification]` — the FW-REQ-002 classification
    of a command's *parameter* fields, used for redaction — and it says nothing
    about what a command produces.

    Returns the binding it made, or None when no artifact of that kind was
    found. Expands the node's children when the value it supplies is the one
    they were waiting on.
    """
    artifact_entries: Iterable[Any]
    if isinstance(artifacts, Mapping):
        artifact_entries = (artifacts,)
    else:
        artifact_entries = artifacts
    found = _first_handle(kind, artifact_entries)
    if found is None:
        return None
    value, command_call_id = found
    if not command_call_id:
        raise ValueError(f"captured '{kind}' has no producing command_call_id")
    if isinstance(value, list):
        value = value[0]

    target = slot or _target_slot(node, kind)
    if target is None:
        raise ValueError(
            f"node {node.goal_id} has no unbound slot to bind a '{kind}' to; "
            "pass slot= to say which"
        )

    catalog = catalog or getattr(record, "_catalog", None)
    node.bindings = {
        **node.bindings,
        target: Binding(
            value=value,
            source="captured",
            kind="captured_handle",
            command_call_id=command_call_id,
        ),
    }
    bound_skill = catalog.get(node.skill or "") if catalog is not None else None
    if bound_skill is not None:
        node.goal_text = (
            _invocation_goal_text(bound_skill, node.bindings)
            if bound_skill.level == "atomic"
            else _render(bound_skill.goal or bound_skill.description, node.bindings)
        )
        if node.visibility == "public":
            node.task_key = _canonical_task_key(bound_skill.name, node.bindings)
    if node.status == "needs-user" and not _needs_user(node.bindings):
        node.status = "not-reached"
    _set_executable_state(node)

    if (
        catalog is not None
        and not _awaiting(node)
        and not record.children(node.goal_id)
    ):
        _expand_children(record, node, catalog)
    return node.bindings[target]


def _widen(
    record: PlanRecord,
    node: PlanNode,
    slot: str,
    values: Sequence[str],
    command_call_id: Optional[str],
) -> list[PlanNode]:
    """A delayed fan-out arriving: elements 2..N become siblings of `node`.

    The following sibling's prerequisite moves to the last of them, so the
    execution DAG stays a chain within the parent's children rather than
    branching around the nodes that arrived late.

    Reserved for a future explicit multi-handle policy. Decision 3's v1 binder
    takes the first uid and deliberately does not call this helper.
    """
    created: list[PlanNode] = []
    previous = node
    for position, element in enumerate(values, start=2):
        clone = PlanNode(
            goal_id=f"{node.goal_id}.{position}",
            parent_goal_id=node.parent_goal_id,
            prerequisites=(previous.goal_id,),
            level=node.level,
            skill=node.skill,
            goal_text=node.goal_text,
            visibility=node.visibility,
            bindings={
                **node.bindings,
                slot: Binding(
                    value=element,
                    source="captured",
                    kind="captured_handle",
                    command_call_id=command_call_id,
                ),
            },
        )
        clone.goal_text = _render(node.goal_text, clone.bindings)
        record.add_node(clone)
        created.append(clone)
        previous = clone

    for other in record.nodes:
        if other is node or other in created:
            continue
        if node.goal_id in other.prerequisites:
            other.prerequisites = tuple(
                previous.goal_id if p == node.goal_id else p
                for p in other.prerequisites
            )
    return created


def _target_slot(node: PlanNode, kind: str) -> Optional[str]:
    unbound = node.unbound_slots()
    if kind in unbound:
        return kind
    if len(unbound) == 1:
        return unbound[0]
    return kind if kind in node.bindings else None


def _first_handle(
    kind: str, artifacts: Iterable[Any]
) -> Optional[tuple[Union[str, list[str]], Optional[str]]]:
    """The first value of `kind`, in execution order, with its producing call id.

    Two key shapes, because both are what commands emit: the scalar `kind`, and
    the plural listing key `kind + "s"` that `listing_payload` writes
    (`{"account_uids": [...], "labels": [...], "total": n}`). A list answers as
    a list; taking its head is `bind_captured`'s decision, not this one's.
    The decision-3 first-uid rule is authoritative at the caller: a plural
    listing contributes its first non-null item and never widens the plan.
    """
    for entry in artifacts or ():
        payload, command_call_id = _artifact_payload(entry)
        if not isinstance(payload, Mapping):
            continue
        for key in (kind, f"{kind}s"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, (list, tuple)):
                items = [str(item) for item in value if item is not None]
                if items:
                    return items, command_call_id
            elif value is not None:
                return str(value), command_call_id
    return None


def _artifact_payload(entry: Any) -> tuple[Any, Optional[str]]:
    response = getattr(entry, "command_response", None)
    if response is not None:
        return getattr(response, "artifacts", None), getattr(
            entry, "command_call_id", None
        )
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[1], entry[0]
    if isinstance(entry, Mapping) and "artifacts" in entry:
        return entry["artifacts"], entry.get("command_call_id")
    if isinstance(entry, Mapping):
        return (
            {key: value for key, value in entry.items() if key != "command_call_id"},
            entry.get("command_call_id"),
        )
    return entry, None


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _invocation_goal_text(skill: Skill, bindings: Mapping[str, Binding]) -> str:
    """Render a private atomic invocation without inventing a public predicate."""
    arguments: list[str] = []
    for slot in skill.slots:
        binding = bindings.get(slot.name)
        if binding is None or binding.value is None:
            continue
        value = (
            ", ".join(binding.value)
            if isinstance(binding.value, list)
            else str(binding.value)
        )
        arguments.append(f"{slot.name}={value}")
    return " ".join((skill.name, *arguments)).strip()


def _render(template: str, bindings: Mapping[str, Binding]) -> str:
    """Substitute `{slot}` from bindings. An unbound slot keeps its placeholder.

    Template-derived from structured state, never paraphrased by a model
    (FW-REQ-010B clause 11). An unbound placeholder is left standing because
    the alternative — dropping it, or filling it with a plausible noun — is the
    fabrication the whole binding record exists to make visible.
    """
    text = template or ""
    for name, binding in bindings.items():
        if binding.value is None:
            continue
        value = (
            ", ".join(binding.value)
            if isinstance(binding.value, list)
            else str(binding.value)
        )
        text = text.replace("{%s}" % name, value)
    return text


def _node_level(skill: Skill) -> NodeLevel:
    if skill.level == "task":
        return "task"
    if skill.level == "composite":
        return "composite"
    if skill.level == "atomic":
        return "atomic"
    raise PlanConfigurationError(
        f"skill {skill.name} has unsupported plan level {skill.level!r}"
    )


def _record_mode(mode: Union[PlanMode, str]) -> PlanRecordMode:
    value = mode.value if isinstance(mode, PlanMode) else str(mode)
    if value == "off":
        return "off"
    if value == "shadow":
        return "shadow"
    if value == "enforce":
        return "enforce"
    raise PlanConfigurationError(f"unknown plan mode {value!r}")


def _visibility(level: str) -> Visibility:
    """FW-REQ-010B: task and composite are public, everything else is private.

    Clause 3 is satisfied structurally — no mandatory outcome is represented
    only privately, because the atomics are always children of a public task
    node — and it produces clause 3's acceptance shape: a nine-leaf plan
    projects to a three-line public list.
    """
    return "public" if level in ("task", "composite") else "private"


def render_account(record: PlanRecord) -> str:
    """The honest account of decision 5, or `""` when every leaf is done.

    > *N of M leaves done. Not reached: `<goal text>`, `<goal text>`. Stopped
    > after K of L iterations.*

    **Every number comes from the plan record and the budget counters, none
    from a model.** That is EXP-027's one non-negotiable constraint, and it is
    stronger here than in EXP-027 because "which leaves" is a fact the plan
    holds rather than an item count nothing owns.
    """
    leaves = record.leaves
    if not leaves:
        return ""
    done = [leaf for leaf in leaves if leaf.status == "done"]
    if len(done) == len(leaves):
        return ""

    parts = [f"{len(done)} of {len(leaves)} leaves done."]
    not_reached = [leaf for leaf in leaves if leaf.status == "not-reached"]
    if not_reached:
        parts.append(
            "Not reached: " + ", ".join(leaf.goal_text for leaf in not_reached) + "."
        )
    if record.budget_limit:
        parts.append(
            f"Stopped after {record.budget_consumed} of {record.budget_limit} "
            "iterations."
        )
    return " ".join(parts)
