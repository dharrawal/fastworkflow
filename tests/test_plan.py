"""EXP-028 deterministic plan expansion, binding, and accounting."""

from __future__ import annotations

import ast
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fastworkflow import CommandOutput, CommandResponse, workflow_agent
from fastworkflow.plan import (
    Binding,
    Invocation,
    PlanConfigurationError,
    PlanMode,
    PlanNode,
    PlanRecord,
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
    )
    assert atomic.bindings["query"] == Binding(
        value="Devon Morrison",
        source="utterance",
    )


def test_binding_source_on_repeat(catalog):
    record = expand(
        catalog,
        [Invocation(skill_name="leaver-sweep")],
        "Someone is leaving",
    )

    binding = record.roots[0].bindings["identity_query"]
    assert binding.source == "on_repeat"
    assert binding.value == ("find_identity with query=*, then offer the named matches")


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
                        "slots": {"identity_query": "Devon Morrison"},
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "ChainOfThought",
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
        )
    ]
    assert selected.selection_model == "selector-model"
    assert len(calls) == 1
    cards = json.loads(calls[0]["catalogue"])
    assert cards
    assert all("body" not in card for card in cards)


def test_select_skills_rejects_prose_instead_of_parsing_it(catalog, monkeypatch):
    class ProseSelector:
        def __call__(self, **_kwargs):
            return SimpleNamespace(invocations="1. leaver-sweep Devon Morrison")

    monkeypatch.setattr(
        workflow_agent.dspy,
        "ChainOfThought",
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
                        "slots": {"identity_query": "Invented Person"},
                    }
                ]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "ChainOfThought",
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
