"""EXP-028 deterministic plan expansion, binding, and accounting."""

from __future__ import annotations

import ast
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import LMError
from pydantic import ValidationError

from fastworkflow import CommandOutput, CommandResponse, workflow_agent
from fastworkflow.plan import (
    Binding,
    Invocation,
    InvocationEvidence,
    PlanConfigurationError,
    PlanEdge,
    PlanMode,
    PlanNode,
    PlanRecord,
    SourceSpan,
    bind_captured,
    expand,
    plan_mode_from_env,
    render_account,
    require_catalog,
)
from fastworkflow.runtime_manifest import load_manifest
from fastworkflow.skill_catalog import (
    Skill,
    SkillCatalog,
    Slot,
    load_skill_catalog,
    parse_steps,
)
from fastworkflow.workflow_agent import _plan_decomposition_point, select_skills

FIXTURE_WORKFLOW = Path(__file__).parent / "fixtures" / "skills_workflow"
PLAN_MODULE = Path(__file__).parents[1] / "fastworkflow" / "plan.py"


@pytest.fixture
def catalog() -> SkillCatalog:
    manifest = load_manifest(str(FIXTURE_WORKFLOW))
    assert manifest is not None
    return load_skill_catalog(str(FIXTURE_WORKFLOW), manifest)


def _skill(
    name: str,
    level: str,
    *,
    goal: str | None = None,
    slots: tuple[Slot, ...] = (),
    uses: tuple[str, ...] = (),
    body: str = "",
) -> Skill:
    return Skill(
        name=name,
        description=f"Exercise {name}.",
        level=level,
        goal=goal,
        slots=slots,
        uses=uses,
        body=body,
        path=f"/test/_skills/{name}/SKILL.md",
        content_hash=f"sha256:{name}",
        steps=parse_steps(body, uses=uses, path=name),
    )


def _catalog(*skills: Skill) -> SkillCatalog:
    return SkillCatalog(
        {skill.name: skill for skill in skills},
        fingerprint="sha256:test-catalogue",
        mode="enforce",
    )


def test_fixture_composite_expands_to_its_declared_leaves(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="offboarding-batch",
                slots={"identity_queries": ["Devon Morrison"]},
            )
        ],
        "Devon Morrison is leaving",
    )

    assert record.max_depth == 3
    assert [(node.level, node.skill) for node in record.leaves] == [
        ("atomic", "inspect-thing"),
        ("commands", None),
        ("commands", None),
    ]
    assert record.roots[0].skill == "offboarding-batch"
    assert record.roots[0].visibility == "public"


def test_atomic_directly_under_composite_is_private_with_public_ancestor():
    atomic = _skill(
        "inspect-subject",
        "atomic",
        slots=(
            Slot(
                name="query",
                required=True,
                on_repeat="known",
                description="Subject",
            ),
        ),
    )
    composite = _skill(
        "packet",
        "composite",
        goal="The {subject} packet is inspected.",
        slots=(Slot(name="subject", description="Subject"),),
        uses=("inspect-subject",),
        body="1. inspect-subject query={subject}",
    )

    record = expand(
        _catalog(composite, atomic),
        [Invocation(skill_name="packet", slots={"subject": "Casey"})],
        "Inspect Casey",
    )

    root = record.roots[0]
    child = record.children(root.goal_id)[0]
    assert root.visibility == "public"
    assert child.level == "atomic"
    assert child.visibility == "private"
    assert child.parent_goal_id == root.goal_id
    assert record.depth_of(child.goal_id) == 2


def test_for_each_fans_out_in_utterance_order(catalog):
    people = ["Devon Morrison", "Sean Lyons", "Jennifer Sellers"]

    record = expand(
        catalog,
        [
            Invocation(
                skill_name="offboarding-batch",
                slots={"identity_queries": people},
            )
        ],
        "Devon Morrison, Sean Lyons and Jennifer Sellers are leaving",
    )

    children = record.children(record.roots[0].goal_id)
    assert [child.skill for child in children] == ["leaver-sweep"] * 3
    assert [child.bindings["identity_query"].value for child in children] == people
    assert [child.goal_text.split("'s", 1)[0] for child in children] == people
    assert all(child.prerequisites == () for child in children)
    assert [
        child.bindings["identity_query"].source_spans[0].text
        for child in children
    ] == people


def test_absent_optional_list_slots_create_no_phantom_children():
    child = _skill(
        "handle-subject",
        "task",
        goal="The {subject} request is handled.",
        slots=(Slot(name="subject"),),
        body="1. `known`",
    )
    packet = _skill(
        "packet",
        "composite",
        goal="The requested subjects are handled.",
        slots=(
            Slot(name="requested", list=True),
            Slot(name="omitted", list=True),
        ),
        uses=("handle-subject",),
        body=(
            "1. for each {subject} in {requested}: "
            "handle-subject subject={subject}\n"
            "2. for each {subject} in {omitted}: "
            "handle-subject subject={subject}"
        ),
    )

    record = expand(
        _catalog(packet, child),
        [
            Invocation(
                skill_name="packet",
                slots={"requested": ["Casey"]},
            )
        ],
        "Handle Casey",
    )

    children = record.children(record.roots[0].goal_id)
    assert len(children) == 1
    assert children[0].bindings["subject"].value == "Casey"


def test_step_after_fan_out_depends_on_every_fan_out_sibling():
    worker = _skill(
        "handle-subject",
        "task",
        goal="The {subject} request is handled.",
        slots=(Slot(name="subject"),),
        body="1. `known`",
    )
    final = _skill(
        "inspect-summary",
        "atomic",
        slots=(Slot(name="query"),),
    )
    packet = _skill(
        "packet",
        "composite",
        goal="The packet is handled.",
        slots=(
            Slot(name="subjects", list=True),
            Slot(name="summary_subject"),
        ),
        uses=("handle-subject", "inspect-summary"),
        body=(
            "1. for each {subject} in {subjects}: "
            "handle-subject subject={subject}\n"
            "2. inspect-summary query={summary_subject}"
        ),
    )

    record = expand(
        _catalog(packet, worker, final),
        [
            Invocation(
                skill_name="packet",
                slots={
                    "subjects": ["Casey", "Devon"],
                    "summary_subject": "summary",
                },
            )
        ],
        "Handle Casey and Devon, then inspect summary",
    )

    children = record.children(record.roots[0].goal_id)
    fan_out = children[:2]
    following = children[2]
    assert all(child.prerequisites == () for child in fan_out)
    assert following.prerequisites == tuple(child.goal_id for child in fan_out)


def test_delayed_expansion_binds_first_captured_uid_and_producer(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="account-walk",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Walk Devon Morrison's first account",
    )
    root = record.roots[0]
    delayed = next(
        node
        for node in record.children(root.goal_id)
        if node.skill == "account-portrait"
    )
    assert delayed.status == "needs-user"
    assert record.children(delayed.goal_id) == ()

    output = CommandOutput(
        command_name="list_accounts",
        command_call_id="call-list-accounts",
        command_response=CommandResponse(
            response="two accounts",
            artifacts={
                "account_uids": ["acct-first", "acct-second"],
                "labels": ["First", "Second"],
            },
        ),
    )
    binding = bind_captured(
        record,
        delayed,
        "account_uid",
        [output],
    )

    assert binding == Binding(
        value="acct-first",
        source="captured",
        kind="captured_handle",
        command_call_id="call-list-accounts",
    )
    assert delayed.bindings["account_uid"] == binding
    assert delayed.status == "not-reached"
    assert delayed.goal_text == "Account acct-first has been opened and portrayed."
    assert [node.skill for node in record.children(delayed.goal_id)] == [
        "inspect-thing",
        None,
    ]
    assert (
        len(
            [
                node
                for node in record.children(root.goal_id)
                if node.skill == "account-portrait"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "artifacts",
    [
        {
            "command_call_id": "call-list-accounts",
            "artifacts": {"account_uids": ["acct-first", "acct-second"]},
        },
        {
            "command_call_id": "call-list-accounts",
            "account_uids": ["acct-first", "acct-second"],
        },
    ],
)
def test_bind_captured_accepts_a_mapping_directly(catalog, artifacts):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="account-walk",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Walk Devon Morrison's first account",
    )
    delayed = next(node for node in record.nodes if node.skill == "account-portrait")

    binding = bind_captured(
        record,
        delayed,
        "account_uid",
        artifacts,
    )

    assert binding == Binding(
        value="acct-first",
        source="captured",
        kind="captured_handle",
        command_call_id="call-list-accounts",
    )


def test_bind_captured_refuses_untraceable_artifacts(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="account-walk",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Walk Devon Morrison's first account",
    )
    delayed = next(node for node in record.nodes if node.skill == "account-portrait")

    with pytest.raises(ValueError, match="command_call_id"):
        bind_captured(
            record,
            delayed,
            "account_uid",
            {"account_uids": ["acct-first"]},
        )


def test_explicit_missing_child_placeholder_waits_for_capture_before_fallback():
    child = _skill(
        "inspect-subject",
        "atomic",
        slots=(
            Slot(
                name="query",
                required=True,
                on_repeat="find a default subject",
            ),
        ),
    )
    parent = _skill(
        "investigate",
        "task",
        goal="The investigation is presented.",
        uses=("inspect-subject",),
        body="1. inspect-subject query={entity_uid}",
    )

    record = expand(
        _catalog(parent, child),
        [Invocation(skill_name="investigate")],
        "Investigate the finding",
    )

    delayed = record.children(record.roots[0].goal_id)[0]
    assert delayed.bindings["query"] == Binding(
        value=None,
        source="needs-user",
        on_repeat_policy="find a default subject",
    )
    assert delayed.status == "needs-user"


def test_binding_source_utterance(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    assert record.roots[0].bindings["identity_query"] == Binding(
        value="Devon Morrison",
        source="utterance",
        kind="exact_text",
        source_spans=(SourceSpan(start=0, end=14, text="Devon Morrison"),),
    )


def test_body_literal_binding_records_its_skill_source(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    atomic = next(node for node in record.leaves if node.skill == "inspect-thing")
    assert atomic.bindings["entity_type"] == Binding(
        value="identity",
        source="skill",
        kind="skill_literal",
    )
    assert atomic.bindings["query"] == Binding(
        value="Devon Morrison",
        source="utterance",
        kind="exact_text",
        source_spans=(SourceSpan(start=0, end=14, text="Devon Morrison"),),
    )


def test_normalized_enum_binding_records_exact_source_span_and_normalizer():
    inspect = _skill(
        "inspect-subject",
        "atomic",
        slots=(
            Slot(
                name="entity_type",
                binding_kind="normalized_enum",
                normalizer="entity-type@1",
            ),
            Slot(name="query"),
        ),
    )
    utterance = "Inspect the right SCCM_Monitoring Specialist"
    start = utterance.index("right")
    record = expand(
        _catalog(inspect),
        [
            Invocation(
                skill_name="inspect-subject",
                slots={
                    "entity_type": "permission",
                    "query": "SCCM_Monitoring Specialist",
                },
                provenance={
                    "entity_type": InvocationEvidence(
                        kind="normalized_enum",
                        normalizer="entity-type@1",
                        source_spans=(
                            SourceSpan(
                                start=start,
                                end=start + len("right"),
                                text="right",
                            ),
                        ),
                    )
                },
            )
        ],
        utterance,
    )

    binding = record.roots[0].bindings["entity_type"]
    assert binding.kind == "normalized_enum"
    assert binding.normalizer == "entity-type@1"
    assert binding.source_spans == (
        SourceSpan(start=start, end=start + len("right"), text="right"),
    )


def test_normalized_enum_binding_rejects_a_span_that_normalizes_differently():
    inspect = _skill(
        "inspect-subject",
        "atomic",
        slots=(
            Slot(
                name="entity_type",
                binding_kind="normalized_enum",
                normalizer="entity-type@1",
            ),
        ),
        body="1. `inspect`",
    )
    utterance = "Inspect the account"
    start = utterance.index("account")

    with pytest.raises(PlanConfigurationError, match="normalizes to 'account'"):
        expand(
            _catalog(inspect),
            [
                Invocation(
                    skill_name="inspect-subject",
                    slots={"entity_type": "permission"},
                    provenance={
                        "entity_type": InvocationEvidence(
                            kind="normalized_enum",
                            normalizer="entity-type@1",
                            source_spans=(
                                SourceSpan(
                                    start=start,
                                    end=start + len("account"),
                                    text="account",
                                ),
                            ),
                        )
                    },
                )
            ],
            utterance,
        )


def test_captured_invocation_binding_requires_and_records_producer_call_id():
    investigate = _skill(
        "investigate",
        "task",
        goal="Control {control_code} is investigated.",
        slots=(Slot(name="control_code"),),
        body="1. `known`",
    )
    record = expand(
        _catalog(investigate),
        [
            Invocation(
                skill_name="investigate",
                slots={"control_code": "ctrl_derived"},
                provenance={
                    "control_code": InvocationEvidence(
                        kind="captured_handle",
                        command_call_id="call-list-findings",
                    )
                },
            )
        ],
        "Investigate the contractor control",
    )

    assert record.roots[0].bindings["control_code"] == Binding(
        value="ctrl_derived",
        source="captured",
        kind="captured_handle",
        command_call_id="call-list-findings",
    )
    with pytest.raises(ValidationError, match="producing command_call_id"):
        InvocationEvidence(kind="captured_handle")


def test_child_scope_renders_atomic_goal_and_merges_nonexecuting_guidance():
    atomic = _skill(
        "inspect-subject",
        "atomic",
        slots=(
            Slot(name="entity_type"),
            Slot(name="query"),
        ),
    )
    parent = _skill(
        "review-subject",
        "task",
        goal="{subject} is reviewed.",
        slots=(Slot(name="subject"),),
        uses=("inspect-subject",),
        body=(
            "1. inspect-subject entity_type=identity query={subject}\n"
            "2. [presentation] Present the completed review for {subject}."
        ),
    )
    record = expand(
        _catalog(parent, atomic),
        [Invocation(skill_name="review-subject", slots={"subject": "Casey"})],
        "Review Casey",
    )

    assert len(record.leaves) == 1
    leaf = record.leaves[0]
    assert leaf.executable
    assert leaf.executable_goal_text == leaf.goal_text
    assert "{subject}" not in leaf.goal_text
    assert "query=Casey" in leaf.goal_text
    assert "Present the completed review for Casey." in leaf.goal_text


def test_on_repeat_remains_policy_and_is_not_used_as_a_value(catalog):
    record = expand(
        catalog,
        [Invocation(skill_name="leaver-sweep")],
        "Someone is leaving",
    )

    binding = record.roots[0].bindings["identity_query"]
    assert binding.source == "needs-user"
    assert binding.value is None
    assert binding.on_repeat_policy == (
        "find_identity with query=*, then offer the named matches"
    )


def test_unbindable_required_slot_is_needs_user_and_never_invented():
    skill = _skill(
        "needs-input",
        "task",
        goal="The {subject} request is handled.",
        slots=(
            Slot(
                name="subject",
                required=True,
                on_repeat=None,
                description="Subject",
            ),
        ),
        body="1. `known`",
    )

    record = expand(
        _catalog(skill),
        [Invocation(skill_name="needs-input")],
        "Handle the request",
    )

    root = record.roots[0]
    assert root.status == "needs-user"
    assert root.bindings["subject"] == Binding(
        value=None,
        source="needs-user",
    )
    assert "{subject}" in root.goal_text
    assert record.children(root.goal_id) == ()


def test_prerequisites_are_sequential_within_a_node(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    children = record.children(record.roots[0].goal_id)
    assert children[0].prerequisites == ()
    assert children[1].prerequisites == (children[0].goal_id,)
    assert children[2].prerequisites == (children[1].goal_id,)


@pytest.mark.parametrize(
    ("invocation", "utterance", "message"),
    [
        (
            Invocation(
                skill_name="offboarding-batch",
                slots={"identity_queries": "Devon Morrison"},
            ),
            "Devon Morrison is leaving",
            "requires a list",
        ),
        (
            Invocation(
                skill_name="leaver-sweep",
                slots={"unknown": "Devon Morrison"},
            ),
            "Devon Morrison is leaving",
            "undeclared slot",
        ),
        (
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Invented Person"},
            ),
            "Devon Morrison is leaving",
            "verbatim",
        ),
    ],
)
def test_invocation_slot_shape_and_values_are_validated(
    catalog, invocation, utterance, message
):
    with pytest.raises(PlanConfigurationError, match=message):
        expand(catalog, [invocation], utterance)


def test_sibling_task_roots_have_no_cross_node_prerequisites(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            ),
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Sean Lyons"},
            ),
        ],
        "Devon Morrison and Sean Lyons are leaving",
    )

    first, second = record.roots
    assert first.prerequisites == ()
    assert second.prerequisites == ()
    first_subtree_ids = {
        node.goal_id
        for node in record.nodes
        if node.goal_id == first.goal_id or node.goal_id.startswith(f"{first.goal_id}.")
    }
    assert not first_subtree_ids.intersection(second.prerequisites)


def test_expansion_rechecks_the_depth_limit():
    leaf = _skill("leaf", "atomic")
    lower = _skill(
        "lower",
        "task",
        goal="The {subject} lower goal is satisfied.",
        slots=(Slot(name="subject"),),
        uses=("leaf",),
        body="1. leaf",
    )
    middle = _skill(
        "middle",
        "task",
        goal="The {subject} middle goal is satisfied.",
        slots=(Slot(name="subject"),),
        uses=("lower",),
        body="1. lower subject={subject}",
    )
    root = _skill(
        "root",
        "composite",
        goal="The {subject} root goal is satisfied.",
        slots=(Slot(name="subject"),),
        uses=("middle",),
        body="1. middle subject={subject}",
    )

    with pytest.raises(PlanConfigurationError, match="depth"):
        expand(
            _catalog(root, middle, lower, leaf),
            [Invocation(skill_name="root", slots={"subject": "Casey"})],
            "Handle Casey",
        )


def test_render_account_uses_only_record_counters_and_goal_text():
    record = PlanRecord(
        plan_id="plan",
        nodes=(
            PlanNode(
                goal_id="done",
                level="atomic",
                skill="done-skill",
                goal_text="done goal",
                status="done",
                command_call_ids=("call-done",),
            ),
            PlanNode(
                goal_id="second",
                level="commands",
                goal_text="second goal",
                status="not-reached",
            ),
            PlanNode(
                goal_id="third",
                level="atomic",
                skill="third-skill",
                goal_text="third goal",
                status="not-reached",
            ),
        ),
        budget_limit=12,
        budget_consumed=7,
    )

    assert render_account(record) == (
        "1 of 3 leaves done. Not reached: second goal, third goal. "
        "Stopped after 7 of 12 iterations."
    )


def test_render_account_is_empty_when_every_leaf_is_done():
    record = PlanRecord(
        plan_id="plan",
        nodes=(
            PlanNode(
                goal_id="first",
                level="atomic",
                skill="first",
                status="done",
                command_call_ids=("call-first",),
            ),
            PlanNode(
                goal_id="second",
                level="commands",
                status="done",
                command_call_ids=("call-second",),
            ),
        ),
        budget_limit=10,
        budget_consumed=6,
    )

    assert render_account(record) == ""


def test_completed_is_not_a_plan_node_status(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )
    assert all(node.status != "completed" for node in record.nodes)

    with pytest.raises(ValidationError):
        PlanNode(
            goal_id="invalid",
            level="commands",
            status="completed",
        )


@pytest.mark.parametrize(
    "binding",
    [
        {
            "value": "invented",
            "source": "needs-user",
        },
        {
            "value": "captured",
            "source": "captured",
        },
        {
            "value": None,
            "source": "utterance",
        },
        {
            "value": "uttered",
            "source": "utterance",
            "command_call_id": "call-id",
        },
    ],
)
def test_binding_source_and_evidence_must_be_consistent(binding):
    with pytest.raises(ValidationError):
        Binding(**binding)


def test_done_leaf_requires_a_command_call():
    with pytest.raises(ValidationError, match="command_call_id"):
        PlanNode(
            goal_id="leaf",
            level="commands",
            status="done",
        )


@pytest.mark.parametrize(
    "record",
    [
        {"plan_id": "plan", "mode": "sometimes"},
        {"plan_id": "plan", "budget_limit": -1},
        {"plan_id": "plan", "budget_limit": 2, "budget_consumed": 3},
    ],
)
def test_plan_record_rejects_invalid_modes_and_budgets(record):
    with pytest.raises(ValidationError):
        PlanRecord(**record)


def test_plan_module_never_imports_dspy():
    tree = ast.parse(PLAN_MODULE.read_text(encoding="utf-8"))

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(name == "dspy" or name.startswith("dspy.") for name in imports)


def test_select_skills_makes_one_typed_model_call_with_cards_only(catalog, monkeypatch):
    calls = []

    class FakeSelector:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                invocations=[
                    {
                        "skill_name": "leaver-sweep",
                        "bindings": [
                            {
                                "slot_name": "identity_query",
                                "value": "Devon Morrison",
                                "source_text": "Devon Morrison",
                            }
                        ],
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: FakeSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    selected = select_skills(
        "Devon Morrison is leaving",
        catalog,
        SimpleNamespace(model="selector-model"),
    )

    assert selected == [
        Invocation(
            skill_name="leaver-sweep",
            slots={"identity_query": "Devon Morrison"},
            provenance={
                "identity_query": InvocationEvidence(
                    kind="exact_text",
                    source_spans=(
                        SourceSpan(start=0, end=14, text="Devon Morrison"),
                    ),
                )
            },
        )
    ]
    assert selected.selection_model == "selector-model"
    assert len(calls) == 1
    assert "config" not in calls[0]
    cards = json.loads(calls[0]["catalogue"])
    assert cards
    assert all("body" not in card for card in cards)
    assert all(card["level"] == "task" for card in cards)


def test_select_skills_retries_compiler_validation_once(catalog, monkeypatch):
    calls = []

    class DuplicateThenValidSelector:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            invocation = {
                "skill_name": "leaver-sweep",
                "bindings": [
                    {
                        "slot_name": "identity_query",
                        "value": "Devon Morrison",
                        "source_text": "Devon Morrison",
                    }
                ],
            }
            return SimpleNamespace(
                invocations=(
                    [invocation, invocation]
                    if len(calls) == 1
                    else [invocation]
                )
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: DuplicateThenValidSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    selected = select_skills(
        "Devon Morrison is leaving",
        catalog,
        SimpleNamespace(model="selector-model"),
    )

    assert len(calls) == 2
    assert selected.validation_retry_used is True
    assert len(selected.validation_errors) == 1
    assert "repeats canonical invocation" in selected.validation_errors[0]
    assert "repeats canonical invocation" in calls[1]["validation_feedback"]


def test_select_skills_refuses_composite_names_under_task_first_contract(
    catalog, monkeypatch
):
    calls = []

    class CompositeSelector:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                invocations=[
                    {
                        "skill_name": "offboarding-batch",
                        "bindings": [
                            {
                                "slot_name": "identity_queries",
                                "value": "Devon Morrison",
                                "source_text": "Devon Morrison",
                            }
                        ],
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: CompositeSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(
        PlanConfigurationError, match="must name a task skill"
    ):
        select_skills(
            "Devon Morrison is leaving",
            catalog,
            SimpleNamespace(model="selector-model"),
        )

    assert len(calls) == 2
    assert all(
        card["level"] == "task"
        for card in json.loads(calls[0]["catalogue"])
    )
    assert "must name a task skill" in (
        calls[1]["validation_feedback"]
    )


def test_select_skills_does_not_response_retry_provider_failures(
    catalog, monkeypatch
):
    calls = []

    class FailingProviderSelector:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            raise LMError("provider authentication failed")

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: FailingProviderSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(LMError, match="provider authentication failed"):
        select_skills(
            "Devon Morrison is leaving",
            catalog,
            SimpleNamespace(model="selector-model"),
        )

    assert len(calls) == 1


def test_select_skills_rejects_prose_instead_of_parsing_it(catalog, monkeypatch):
    class ProseSelector:
        def __call__(self, **_kwargs):
            return SimpleNamespace(invocations="1. leaver-sweep Devon Morrison")

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: ProseSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(ValueError, match="JSON list"):
        select_skills(
            "Devon Morrison is leaving",
            catalog,
            SimpleNamespace(model="selector-model"),
        )


def test_select_skills_rejects_nonverbatim_slot_values(catalog, monkeypatch):
    class InventingSelector:
        def __call__(self, **_kwargs):
            return SimpleNamespace(
                invocations=[
                    {
                        "skill_name": "leaver-sweep",
                        "bindings": [
                            {
                                "slot_name": "identity_query",
                                "value": "Invented Person",
                                "source_text": "Devon Morrison",
                            }
                        ],
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: InventingSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(PlanConfigurationError, match="verbatim"):
        select_skills(
            "Devon Morrison is leaving",
            catalog,
            SimpleNamespace(model="selector-model"),
        )


def test_select_skills_carries_normalized_source_span_into_invocation(
    monkeypatch,
):
    inspect = _skill(
        "inspect-subject",
        "task",
        goal="{entity_type} is inspected.",
        slots=(
            Slot(
                name="entity_type",
                binding_kind="normalized_enum",
                normalizer="entity-type@1",
            ),
        ),
        body="1. `inspect`",
    )
    utterance = "Inspect the right"
    start = utterance.index("right")

    class NormalizingSelector:
        def __call__(self, **_kwargs):
            return SimpleNamespace(
                invocations=[
                    {
                        "skill_name": "inspect-subject",
                        "bindings": [
                            {
                                "slot_name": "entity_type",
                                "value": "permission",
                                "source_text": "right",
                            }
                        ],
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: NormalizingSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    selected = select_skills(
        utterance,
        _catalog(inspect),
        SimpleNamespace(model="selector-model"),
    )

    evidence = selected[0].provenance["entity_type"]
    assert evidence.kind == "normalized_enum"
    assert evidence.normalizer == "entity-type@1"
    assert evidence.source_spans == (
        SourceSpan(start=start, end=start + len("right"), text="right"),
    )


def test_plan_mode_parses_from_fw_plan_decomposition():
    assert plan_mode_from_env({}) is PlanMode.OFF
    assert plan_mode_from_env({"FW_PLAN_DECOMPOSITION": "shadow"}) is PlanMode.SHADOW
    assert plan_mode_from_env({"FW_PLAN_DECOMPOSITION": "enforce"}) is PlanMode.ENFORCE
    with pytest.raises(ValueError):
        plan_mode_from_env({"FW_PLAN_DECOMPOSITION": "yes"})


@pytest.mark.parametrize("mode", [PlanMode.SHADOW, PlanMode.ENFORCE])
def test_non_off_mode_requires_a_manifest_enabled_nonempty_catalogue(mode):
    with pytest.raises(PlanConfigurationError):
        require_catalog(mode, None)
    with pytest.raises(PlanConfigurationError):
        require_catalog(mode, SkillCatalog(mode="off"))
    with pytest.raises(PlanConfigurationError):
        require_catalog(mode, SkillCatalog(mode="enforce"))


def test_off_mode_does_not_require_a_catalogue():
    require_catalog(PlanMode.OFF, None)


def test_plan_mode_cannot_exceed_manifest_enabled_catalogue():
    skill = _skill("leaf", "atomic")

    with pytest.raises(PlanConfigurationError, match="more permissive"):
        require_catalog(
            PlanMode.ENFORCE,
            SkillCatalog({"leaf": skill}, mode="shadow"),
        )

    require_catalog(
        PlanMode.SHADOW,
        SkillCatalog({"leaf": skill}, mode="enforce"),
    )


def test_plan_decomposition_point_hard_errors_without_enabled_catalogue(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FW_PLAN_DECOMPOSITION", "enforce")
    session = SimpleNamespace(app_workflow=SimpleNamespace(folderpath=str(tmp_path)))

    with pytest.raises(PlanConfigurationError):
        _plan_decomposition_point(session)


def test_plan_decomposition_point_loads_dual_gated_catalogue(monkeypatch):
    monkeypatch.setenv("FW_PLAN_DECOMPOSITION", "shadow")
    monkeypatch.setenv(
        "FASTWORKFLOW_RUNTIME_FEATURES",
        "skills_v1=enforce",
    )
    session = SimpleNamespace(
        app_workflow=SimpleNamespace(folderpath=str(FIXTURE_WORKFLOW))
    )

    mode, loaded = _plan_decomposition_point(session)

    assert mode is PlanMode.SHADOW
    assert loaded.mode == "enforce"
    assert loaded.names == (
        "account-portrait",
        "account-walk",
        "inspect-thing",
        "leaver-sweep",
        "offboarding-batch",
    )


def test_public_nodes_carry_canonical_task_keys(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="offboarding-batch",
                slots={"identity_queries": ["Devon Morrison", "Sean Lyons"]},
            )
        ],
        "Devon Morrison and Sean Lyons are leaving",
    )
    public = [node for node in record.public_nodes]
    assert public
    assert all(node.task_key for node in public)
    assert tuple(node.task_key for node in public) == record.compiled_public_task_keys
    assert record.requested_public_task_keys == record.compiled_public_task_keys


def test_duplicate_requested_public_task_keys_are_refused():
    sweep = _skill(
        "leaver-sweep",
        "task",
        goal="The {identity_query} sweep is complete.",
        slots=(Slot(name="identity_query", required=True),),
        body="1. `inspect`",
    )
    with pytest.raises(PlanConfigurationError, match="duplicate task coverage"):
        expand(
            _catalog(sweep),
            [
                Invocation(
                    skill_name="leaver-sweep",
                    slots={"identity_query": "Devon Morrison"},
                ),
                Invocation(
                    skill_name="leaver-sweep",
                    slots={"identity_query": "Devon Morrison"},
                ),
            ],
            "Devon Morrison is leaving",
        )


def test_edges_record_provenance_classes(catalog):
    record = expand(
        catalog,
        [
            Invocation(
                skill_name="offboarding-batch",
                slots={"identity_queries": ["Devon Morrison"]},
            ),
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Sean Lyons"},
            ),
        ],
        "Devon Morrison and Sean Lyons are leaving",
    )
    kinds = {edge.provenance for edge in record.edges}
    assert {"composite", "explicit-order", "data", "stable-tiebreak"} <= kinds
    assert all(isinstance(edge, PlanEdge) for edge in record.edges)


def test_only_fully_rendered_leaves_are_marked_executable(catalog):
    child = _skill(
        "inspect-subject",
        "atomic",
        goal="Inspect {subject}.",
        slots=(Slot(name="subject", required=True),),
    )
    parent = _skill(
        "packet",
        "task",
        goal="Handle packet.",
        uses=("inspect-subject",),
        body="1. inspect-subject subject={missing_subject}\n2. `record completion`",
    )
    record = expand(_catalog(parent, child), [Invocation(skill_name="packet")], "Handle it")
    executable = [node for node in record.leaves if node.executable]
    non_executable = [node for node in record.leaves if not node.executable]
    assert executable
    assert non_executable
    assert all(node.executable_goal_text for node in executable)
    assert all(node.executable_goal_text is None for node in non_executable)


def test_task_first_compiler_synthesizes_private_composite_without_coverage_drift(
    catalog,
):
    invocations = [
        Invocation(
            skill_name="leaver-sweep",
            slots={"identity_query": "Devon Morrison"},
        ),
        Invocation(
            skill_name="leaver-sweep",
            slots={"identity_query": "Sean Lyons"},
        ),
    ]
    record = expand(
        catalog,
        invocations,
        "Devon Morrison and Sean Lyons are leaving",
        plan_id="packing-exact",
    )

    assert [node.skill for node in record.public_nodes] == [
        "leaver-sweep",
        "leaver-sweep",
    ]
    assert record.requested_public_task_keys == record.compiled_public_task_keys
    assert record.packing.selected_root_group_count == 1
    assert record.packing.packed_task_count == 2
    group = record.composite_groups[0]
    assert group.composite_skill == "offboarding-batch"
    assert group.member_goal_ids == ("g1", "g2")
    assert group.shared_bindings == {
        "identity_queries": ("Devon Morrison", "Sean Lyons")
    }
    assert all(node.level != "composite" for node in record.nodes)


def test_composite_packing_refuses_singletons_and_literal_near_matches():
    inspect = _skill(
        "inspect-subject",
        "task",
        goal="{query} is inspected as {entity_type}.",
        slots=(Slot(name="entity_type"), Slot(name="query")),
        body="1. `inspect`",
    )
    packet = _skill(
        "identity-packet",
        "composite",
        goal="The identities in {queries} are inspected.",
        slots=(Slot(name="queries", list=True),),
        uses=("inspect-subject",),
        body=(
            "1. for each {query} in {queries}: "
            "inspect-subject entity_type=identity query={query}"
        ),
    )
    catalog = _catalog(packet, inspect)

    singleton = expand(
        catalog,
        [
            Invocation(
                skill_name="inspect-subject",
                slots={"entity_type": "identity", "query": "Casey"},
            )
        ],
        "Inspect Casey as identity",
    )
    near_match = expand(
        catalog,
        [
            Invocation(
                skill_name="inspect-subject",
                slots={"entity_type": "account", "query": "Casey"},
            ),
            Invocation(
                skill_name="inspect-subject",
                slots={"entity_type": "account", "query": "Riley"},
            ),
        ],
        "Inspect Casey and Riley as account",
    )

    assert singleton.composite_groups == ()
    assert singleton.packing.unpacked_task_count == 1
    assert near_match.composite_groups == ()
    assert near_match.packing.candidate_count == 0
    assert near_match.requested_public_task_keys == near_match.compiled_public_task_keys


def test_composite_packing_refuses_partial_cover_with_private_extra_work():
    first = _skill("task-a", "task", goal="A is done.", body="1. `a`")
    second = _skill("task-b", "task", goal="B is done.", body="1. `b`")
    inspect = _skill("inspect", "atomic", goal="Inspection is done.")
    with_atomic = _skill(
        "packet-with-atomic",
        "composite",
        goal="Packet is done.",
        uses=("task-a", "task-b", "inspect"),
        body="1. task-a\n2. task-b\n3. inspect",
    )
    with_commands = _skill(
        "packet-with-commands",
        "composite",
        goal="Packet is done.",
        uses=("task-a", "task-b"),
        body="1. task-a\n2. task-b\n3. `summarize`",
    )
    invocations = [
        Invocation(skill_name="task-a"),
        Invocation(skill_name="task-b"),
    ]

    atomic_record = expand(
        _catalog(first, second, inspect, with_atomic),
        invocations,
        "Do A and B",
    )
    commands_record = expand(
        _catalog(first, second, with_commands),
        invocations,
        "Do A and B",
    )

    assert atomic_record.packing.candidate_count == 0
    assert atomic_record.composite_groups == ()
    assert commands_record.packing.candidate_count == 0
    assert commands_record.composite_groups == ()


def test_composite_packing_requires_exact_shared_scalar_bindings():
    triage = _skill(
        "triage",
        "task",
        goal="{permission} is triaged for {requester}.",
        slots=(Slot(name="permission"), Slot(name="requester")),
        body="1. `triage`",
    )
    packet = _skill(
        "request-packet",
        "composite",
        goal="{permissions} are triaged for {requester}.",
        slots=(
            Slot(name="permissions", list=True),
            Slot(name="requester"),
        ),
        uses=("triage",),
        body=(
            "1. for each {permission} in {permissions}: "
            "triage permission={permission} requester={requester}"
        ),
    )
    record = expand(
        _catalog(packet, triage),
        [
            Invocation(
                skill_name="triage",
                slots={"permission": "Admin", "requester": "Casey"},
            ),
            Invocation(
                skill_name="triage",
                slots={"permission": "Operator", "requester": "Casey"},
            ),
            Invocation(
                skill_name="triage",
                slots={"permission": "Reader", "requester": "Riley"},
            ),
        ],
        "Admin and Operator for Casey; Reader for Riley",
    )

    assert record.packing.packed_task_count == 2
    assert record.packing.unpacked_task_count == 1
    assert record.composite_groups[0].member_goal_ids == ("g1", "g2")
    assert record.composite_groups[0].shared_bindings == {
        "permissions": ("Admin", "Operator"),
        "requester": "Casey",
    }


def test_composite_set_packing_maximizes_total_non_overlapping_coverage():
    tasks = tuple(
        _skill(
            name,
            "task",
            goal=f"{name} is done.",
            body="1. `work`",
        )
        for name in ("task-a", "task-b", "task-c", "task-d")
    )
    wide = _skill(
        "wide",
        "composite",
        goal="Wide work is done.",
        uses=("task-a", "task-b", "task-c"),
        body="1. task-a\n2. task-b\n3. task-c",
    )
    left = _skill(
        "left",
        "composite",
        goal="Left work is done.",
        uses=("task-a", "task-b"),
        body="1. task-a\n2. task-b",
    )
    right = _skill(
        "right",
        "composite",
        goal="Right work is done.",
        uses=("task-c", "task-d"),
        body="1. task-c\n2. task-d",
    )
    record = expand(
        _catalog(*tasks, wide, left, right),
        [Invocation(skill_name=task.name) for task in tasks],
        "Do task a, task b, task c, and task d",
    )

    root_groups = [
        group for group in record.composite_groups if group.parent_group_id is None
    ]
    assert [group.composite_skill for group in root_groups] == ["left", "right"]
    assert record.packing.packed_task_count == 4
    assert record.packing.selected_root_group_count == 2


def test_composite_overlap_tie_breaks_by_stable_composite_name():
    first = _skill("task-a", "task", goal="A is done.", body="1. `a`")
    second = _skill("task-b", "task", goal="B is done.", body="1. `b`")
    alpha = _skill(
        "alpha-packet",
        "composite",
        goal="Alpha packet is done.",
        uses=("task-a", "task-b"),
        body="1. task-a\n2. task-b",
    )
    zeta = _skill(
        "zeta-packet",
        "composite",
        goal="Zeta packet is done.",
        uses=("task-a", "task-b"),
        body="1. task-a\n2. task-b",
    )

    record = expand(
        _catalog(first, second, alpha, zeta),
        [Invocation(skill_name="task-a"), Invocation(skill_name="task-b")],
        "Do A and B",
    )

    assert record.packing.candidate_count == 2
    assert record.packing.selected_root_group_count == 1
    assert record.composite_groups[0].composite_skill == "alpha-packet"


def test_recursive_composite_packing_records_private_group_hierarchy():
    first = _skill("task-a", "task", goal="A is done.", body="1. `a`")
    second = _skill("task-b", "task", goal="B is done.", body="1. `b`")
    inner = _skill(
        "z-inner",
        "composite",
        goal="Inner work is done.",
        uses=("task-a", "task-b"),
        body="1. task-a\n2. task-b",
    )
    outer = _skill(
        "a-outer",
        "composite",
        goal="Outer work is done.",
        uses=("z-inner",),
        body="1. z-inner",
    )

    record = expand(
        _catalog(first, second, inner, outer),
        [Invocation(skill_name="task-a"), Invocation(skill_name="task-b")],
        "Do A and B",
    )

    assert record.packing.selected_root_group_count == 1
    assert record.packing.selected_recursive_group_count == 1
    root, nested = record.composite_groups
    assert root.composite_skill == "a-outer"
    assert nested.composite_skill == "z-inner"
    assert nested.parent_group_id == root.group_id
    assert nested.member_goal_ids == root.member_goal_ids


def test_composite_packing_serialization_is_deterministic(catalog):
    invocations = [
        Invocation(
            skill_name="leaver-sweep",
            slots={"identity_query": "Devon Morrison"},
        ),
        Invocation(
            skill_name="leaver-sweep",
            slots={"identity_query": "Sean Lyons"},
        ),
    ]
    first = expand(
        catalog,
        invocations,
        "Devon Morrison and Sean Lyons are leaving",
        plan_id="deterministic-plan",
    )
    second = expand(
        catalog,
        invocations,
        "Devon Morrison and Sean Lyons are leaving",
        plan_id="deterministic-plan",
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert first.packing.packing_sha256 == second.packing.packing_sha256
