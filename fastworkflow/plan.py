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
self-tool as the recursive plan executor" all say the same thing: one model call
selects (`workflow_agent.select_skills`), and nothing below it is a model call.
The assertion is structural — `tests/test_plan.py` parses this file and fails on
an `import dspy` — because a test that only exercises the happy path cannot see
a model call added to a branch it does not reach.

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

import os
import uuid
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from fastworkflow.skill_catalog import MAX_DEPTH, Skill, SkillCatalog, Step

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
#: the explicit utterance; a handle captured by an already-executed leaf; the
#: skill's `on_repeat` rule; otherwise `needs-user`.
#:
#: `skill` is the fifth and is not a precedence tier: it is a value the skill
#: body itself supplies (`inspect-entity entity_type=identity`), which is not
#: sourced at bind time at all. Recording it as `utterance` would be a false
#: provenance claim in a record whose whole purpose is provenance, and dropping
#: it would leave `inspect-entity`'s required `entity_type` unbound so the
#: binder would reach for `on_repeat` — a `browse_catalog` fallback — when the
#: body already said `identity`.
BindingSource = Literal["utterance", "captured", "on_repeat", "needs-user", "skill"]

#: Public/private projection (FW-REQ-010B). Every task- and composite-level node
#: is public, its projection being its rendered `goal` with slots substituted.
#: Every atomic node, every navigation and handle-binding node, is private.
Visibility = Literal["public", "private"]
PlanRecordMode = Literal["off", "shadow", "enforce"]


class _PlanModel(BaseModel):
    """Shared posture: no unknown keys, assignment validated.

    `extra="forbid"` for `decision_signals._Strict`'s reason — a typo'd field
    that parses is a field nobody notices is missing. Not frozen, unlike the
    turn-capture records: a plan record is live state for the length of a turn,
    and the executor moves a leaf from `not-reached` to `done`. What must not be
    editable after the fact is the *turn record*, and that is a copy.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Binding(_PlanModel):
    """One slot value, and where it came from.

    `command_call_id` is set only for a `captured` binding: it is the id of the
    command whose `CommandOutput.artifacts` produced the handle, which is what
    makes "the first account on the portrait" a traceable claim rather than an
    assertion.
    """

    value: Union[str, list[str], None] = None
    source: BindingSource
    command_call_id: Optional[str] = None

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
    visibility: Visibility = "private"
    bindings: dict[str, Binding] = Field(default_factory=dict)
    status: LeafStatus = "not-reached"
    budget_limit: Optional[int] = Field(default=None, ge=0)
    budget_consumed: int = Field(default=0, ge=0)
    failure_reason: Optional[str] = None
    command_call_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _leaf_status_has_evidence(self) -> "PlanNode":
        if self.is_leaf and self.status == "done" and not self.command_call_ids:
            raise ValueError("a done leaf requires at least one command_call_id")
        if self.budget_limit is not None and self.budget_consumed > self.budget_limit:
            raise ValueError("budget_consumed cannot exceed budget_limit")
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
                    bindings=_literal_bindings(invocation.slots),
                    status="blocked",
                    failure_reason=(
                        f"selector named '{invocation.skill_name}', which is "
                        "not in the catalogue"
                    ),
                )
            )
            continue
        _validate_invocation(skill, invocation, utterance)
        bindings = _invocation_bindings(skill, invocation)
        node = record.add_node(
            PlanNode(
                goal_id=f"g{index}",
                level=_node_level(skill),
                skill=skill.name,
                goal_text="",
                visibility=_visibility(skill.level),
                bindings=bindings,
                status="needs-user" if _needs_user(bindings) else "not-reached",
            )
        )
        node.goal_text = _render(skill.goal or skill.description, node.bindings)
        if not _awaiting(node):
            _expand_children(record, node, catalog)
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
                bindings={},
            )
        )
        _chain(created)
        return tuple(created)

    previous_step: tuple[PlanNode, ...] = ()
    for step in skill.steps:
        current_step = tuple(_expand_step(record, node, skill, step, catalog))
        prerequisites = tuple(item.goal_id for item in previous_step)
        for child in current_step:
            child.prerequisites = prerequisites
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
                bindings={},
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
        text=step.text,
        bindings=bindings,
        goal_template=child_skill.goal,
        visibility=_visibility(child_skill.level),
    )
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
            text=step.text,
            bindings=_child_bindings(
                node, child_skill, step.arguments, loop_variable=step.loop_variable
            ),
            goal_template=child_skill.goal,
            visibility=_visibility(child_skill.level),
        )
        return [child]

    created: list[PlanNode] = []
    for position, element in enumerate(values, start=1):
        bindings = _child_bindings(
            node,
            child_skill,
            step.arguments,
            loop_variable=step.loop_variable,
            loop_value=element,
            loop_source=binding.source if binding else "utterance",
        )
        child = _add_child(
            record,
            node,
            ordinal=step.ordinal,
            level=_node_level(child_skill),
            skill=child_skill.name,
            text=step.text,
            bindings=bindings,
            goal_template=child_skill.goal,
            visibility=_visibility(child_skill.level),
            suffix=f"{step.ordinal}.{position}",
        )
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
        bindings=dict(bindings),
        status="needs-user" if _needs_user(bindings) else "not-reached",
    )
    return record.add_node(node)


def _chain(nodes: Sequence[PlanNode]) -> None:
    """Sequential prerequisites within one node's children (decision 3).

    Across sibling task nodes there are no edges at all — three leavers are
    independent — which is why this is called per parent and never per plan.
    """
    for previous, node in zip(nodes, nodes[1:]):
        node.prerequisites = (previous.goal_id,)


# ----------------------------------------------------------------------
# Binding
# ----------------------------------------------------------------------


def _invocation_bindings(skill: Skill, invocation: Invocation) -> dict[str, Binding]:
    """Precedence 1 and 3 at the top of the tree: the utterance, then `on_repeat`."""
    bindings: dict[str, Binding] = {}
    for slot in skill.slots:
        if slot.name in invocation.slots:
            bindings[slot.name] = Binding(
                value=invocation.slots[slot.name], source="utterance"
            )
        elif slot.required:
            bindings[slot.name] = _fallback(slot)
    for name, value in invocation.slots.items():
        # A slot the selector bound that the skill does not declare is still
        # recorded: dropping it would lose the only evidence that the selector
        # and the catalogue disagreed.
        bindings.setdefault(name, Binding(value=value, source="utterance"))
    return bindings


def _validate_invocation(skill: Skill, invocation: Invocation, utterance: str) -> None:
    declared = {slot.name: slot for slot in skill.slots}
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
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not item or item not in utterance:
                raise PlanConfigurationError(
                    f"invocation of '{skill.name}' slot '{name}' value "
                    f"{item!r} is not copied verbatim from the utterance"
                )


def _child_bindings(
    parent: PlanNode,
    child_skill: Skill,
    arguments: Mapping[str, str],
    *,
    loop_variable: Optional[str] = None,
    loop_value: Optional[str] = None,
    loop_source: str = "utterance",
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
            bindings[slot_name] = Binding(value=raw, source="skill")
            continue
        reference = raw[1:-1].strip()
        if loop_variable is not None and reference == loop_variable:
            if loop_value is None:
                bindings[slot_name] = Binding(value=None, source="needs-user")
            else:
                bindings[slot_name] = Binding(
                    value=loop_value, source=loop_source  # type: ignore[arg-type]
                )
            continue
        inherited = parent.bindings.get(reference)
        if inherited is not None and inherited.value is not None:
            bindings[slot_name] = Binding(
                value=inherited.value,
                source=inherited.source,
                command_call_id=inherited.command_call_id,
            )
            continue
        # An explicit placeholder says a prior step is expected to supply this
        # value. Keep it pending so captured evidence gets precedence over the
        # child's on_repeat fallback; applying that fallback now would make the
        # producing command's later artifact impossible to bind.
        bindings[slot_name] = Binding(value=None, source="needs-user")

    for slot in child_skill.slots:
        if slot.name in bindings:
            continue
        inherited = parent.bindings.get(slot.name)
        if inherited is not None and inherited.value is not None:
            bindings[slot.name] = Binding(
                value=inherited.value,
                source=inherited.source,
                command_call_id=inherited.command_call_id,
            )
        elif slot.required:
            bindings[slot.name] = _fallback(slot)
    return bindings


def _fallback(slot) -> Binding:
    """Precedence 3, then 4.

    `on_repeat` before asking is the deliberate choice (decision 2): ido's
    slots already carry a deterministic fallback for exactly this, and using it
    at bind time is the ask-policy's rows A and B applied one layer earlier,
    where the answer is deterministic rather than a rewrite of a question
    already formed. A required slot with no utterance value, no captured handle
    and no `on_repeat` is `needs-user` and never a fabricated value
    (FW-REQ-011 acceptance criterion 3).
    """
    if slot is not None and slot.on_repeat:
        return Binding(value=slot.on_repeat, source="on_repeat")
    return Binding(value=None, source="needs-user")


def _literal_bindings(slots: Mapping[str, Any]) -> dict[str, Binding]:
    return {
        name: Binding(value=value, source="utterance") for name, value in slots.items()
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
            command_call_id=command_call_id,
        ),
    }
    bound_skill = catalog.get(node.skill or "") if catalog is not None else None
    if bound_skill is not None:
        node.goal_text = _render(
            bound_skill.goal or bound_skill.description, node.bindings
        )
    if node.status == "needs-user" and not _needs_user(node.bindings):
        node.status = "not-reached"

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
                    value=element, source="captured", command_call_id=command_call_id
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
