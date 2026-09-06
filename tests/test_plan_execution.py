"""EXP-028 Arm B/C composite-orchestration execution distinction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import LMTimeoutError

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.binding_normalizers import CONTROL_ALIAS_RESOLVER_V1
from fastworkflow.plan import (
    Invocation,
    bind_captured,
    expand,
    validate_compiled_plan,
)
from fastworkflow.plan_execution import (
    PlanExecutionArm,
    PlanExecutionBinding,
    PlanExecutionOutcome,
    PlanExecutionScope,
    SafetyEnvelopeState,
    alias_resolution_feedback,
    execute_plan,
    plan_execution_arm_from_env,
    render_leaf_instruction,
)
from fastworkflow.runtime_manifest import load_manifest
from fastworkflow.skill_catalog import (
    Skill,
    SkillCatalog,
    load_skill_catalog,
    parse_steps,
)
from fastworkflow.typed_failure import (
    CODE_EXTRACTION_TRUNCATED,
    CODE_PROVIDER_TIMEOUT,
    TypedFailure,
    extraction_truncated_failure,
)

FIXTURE_WORKFLOW = Path(__file__).parent / "fixtures" / "skills_workflow"


def _fixture_catalog() -> SkillCatalog:
    manifest = load_manifest(str(FIXTURE_WORKFLOW))
    assert manifest is not None
    return load_skill_catalog(str(FIXTURE_WORKFLOW), manifest)


def _skill(
    name: str,
    level: str,
    *,
    goal: str,
    uses: tuple[str, ...] = (),
    body: str = "",
) -> Skill:
    return Skill(
        name=name,
        description=f"Exercise {name}.",
        level=level,
        goal=goal,
        slots=(),
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


def _success(leaf, _scope):
    return SimpleNamespace(
        exhausted=False,
        suspended=False,
        command_call_ids=(f"call-{leaf.goal_id}",),
    )


def test_arm_b_ignores_packing_while_arm_c_applies_shared_group_context():
    plan = expand(
        _fixture_catalog(),
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
        plan_id="arm-distinction",
    )
    arm_b_plan = plan.model_copy(deep=True)
    arm_c_plan = plan.model_copy(deep=True)
    arm_b_scopes = []
    arm_c_scopes = []

    def execute_b(leaf, scope):
        arm_b_scopes.append(scope)
        return _success(leaf, scope)

    def execute_c(leaf, scope):
        arm_c_scopes.append(scope)
        return _success(leaf, scope)

    arm_b = execute_plan(
        arm_b_plan,
        execute_leaf=execute_b,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )
    arm_c = execute_plan(
        arm_c_plan,
        execute_leaf=execute_c,
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert arm_b.metadata is not None
    assert arm_c.metadata is not None
    assert arm_b.metadata.packing_applied is False
    assert arm_b.metadata.composite_groups_applied == 0
    assert arm_b.metadata.context_reuse_count == 0
    assert all(scope.composite_group_id is None for scope in arm_b_scopes)
    assert arm_c.metadata.packing_applied is True
    assert arm_c.metadata.composite_groups_applied == 1
    assert arm_c.metadata.grouped_task_count == 2
    assert arm_c.metadata.context_reuse_count == len(arm_c_scopes) - 1
    assert {scope.composite_group_id for scope in arm_c_scopes} == {"pack-1"}
    assert "Private composite orchestration" not in render_leaf_instruction(
        arm_b_plan.leaves[0],
        arm_b_scopes[0],
    )
    c_instruction = render_leaf_instruction(
        arm_c_plan.leaves[0],
        arm_c_scopes[0],
    )
    assert "Private composite orchestration: offboarding-batch" in c_instruction
    assert "execute only the public task goal" in c_instruction
    assert "reused shared input false" in c_instruction
    reused_instruction = render_leaf_instruction(
        arm_c_plan.leaves[1],
        arm_c_scopes[1],
    )
    assert arm_c_scopes[1].shared_bindings == arm_c_scopes[0].shared_bindings
    assert "Devon Morrison" in reused_instruction
    assert "Sean Lyons" in reused_instruction
    assert "reused shared input true" in reused_instruction
    assert arm_b.metadata.schedule_sha256 == arm_c.metadata.schedule_sha256
    assert (
        arm_b_plan.compiled_public_task_keys
        == arm_c_plan.compiled_public_task_keys
        == plan.requested_public_task_keys
    )
    validate_compiled_plan(arm_b_plan)
    validate_compiled_plan(arm_c_plan)


def test_arm_b_leaf_scope_carries_public_binding_and_navigation_context():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
        plan_id="bound-input-envelope",
    )
    scopes = []

    result = execute_plan(
        plan,
        execute_leaf=lambda leaf, scope: (
            scopes.append(scope),
            _success(leaf, scope),
        )[1],
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
        current_navigation_context=lambda: "DirectoryExplorer",
    )

    assert result.outcome is PlanExecutionOutcome.COMPLETED
    assert scopes
    binding = dict(scopes[0].task_bindings)["identity_query"]
    assert binding == PlanExecutionBinding(
        value="Devon Morrison",
        source="utterance",
        kind="exact_text",
        source_spans=((0, 14),),
    )
    assert scopes[0].navigation_context == "DirectoryExplorer"
    instruction = render_leaf_instruction(plan.leaves[0], scopes[0])
    assert "Private task execution envelope" in instruction
    assert "Devon Morrison" in instruction
    assert '"navigation_context":"DirectoryExplorer"' in instruction
    assert "Private composite orchestration" not in instruction
    assert scopes[-1].producer_call_ids == tuple(
        f"call-{leaf.goal_id}" for leaf in plan.leaves[:-1]
    )


def test_declared_alias_resolution_feedback_is_typed_and_provenance_bearing():
    scope = PlanExecutionScope(
        arm=PlanExecutionArm.B,
        task_goal_id="task-one",
        task_bindings=(
            (
                "rule_query",
                PlanExecutionBinding(
                    value=(
                        "Enabled vendor identities with expired ending date "
                        "and owning active accounts"
                    ),
                    source="utterance",
                    kind="exact_text",
                    resolver=CONTROL_ALIAS_RESOLVER_V1,
                    source_spans=((7, 84),),
                ),
            ),
        ),
        navigation_context="RulesMonitor",
    )
    output = CommandOutput(
        command_name="RuleCatalog/list_rules",
        command_call_id="call-list-rules",
        command_response=CommandResponse(
            response="three rules",
            artifacts={
                "control_codes": ["rule-one", "rule-two"],
                "labels": [
                    "Vendor with past end date and active account",
                    "External worker whose manager left",
                ],
            },
        ),
    )

    feedback = alias_resolution_feedback(scope, output)

    assert '"status":"resolved"' in feedback
    assert '"handle":"rule-one"' in feedback
    assert '"producer_call_id":"call-list-rules"' in feedback
    assert "do not ask for this bound input again" in feedback


def test_arm_c_uses_composite_order_while_arm_b_preserves_task_order():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    packet = _skill(
        "packet",
        "composite",
        goal="The packet is done.",
        uses=("alpha", "beta"),
        body="1. beta\n2. alpha",
    )
    plan = expand(
        _catalog(alpha, beta, packet),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
        plan_id="ordering-distinction",
    )

    def run(arm: PlanExecutionArm):
        copied = plan.model_copy(deep=True)
        task_order: list[str] = []

        def execute(leaf, scope):
            task = copied.node(scope.task_goal_id or "")
            assert task is not None
            task_order.append(task.skill or "")
            return _success(leaf, scope)

        result = execute_plan(
            copied,
            execute_leaf=execute,
            arm=arm,
            safety=SafetyEnvelopeState(enabled=False),
        )
        return copied, result, task_order

    arm_b_plan, arm_b, arm_b_order = run(PlanExecutionArm.B)
    arm_c_plan, arm_c, arm_c_order = run(PlanExecutionArm.C)

    assert arm_b_order == ["alpha", "beta"]
    assert arm_c_order == ["beta", "alpha"]
    assert arm_b.metadata is not None
    assert arm_c.metadata is not None
    assert arm_b.metadata.scheduled_task_goal_ids == ("g1", "g2")
    assert arm_c.metadata.scheduled_task_goal_ids == ("g2", "g1")
    assert arm_b.metadata.schedule_sha256 != arm_c.metadata.schedule_sha256
    assert arm_b_plan.compiled_public_task_keys == arm_c_plan.compiled_public_task_keys


def test_arm_c_without_an_exact_pack_does_not_claim_context_reuse():
    source = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
        plan_id="unpackable",
    )
    plan = source.model_copy(deep=True)

    result = execute_plan(
        plan,
        execute_leaf=_success,
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.metadata is not None
    assert result.metadata.packing_applied is False
    assert result.metadata.composite_groups_applied == 0
    assert result.metadata.context_reuse_count == 0
    assert result.metadata.public_task_keys == plan.compiled_public_task_keys
    arm_b = execute_plan(
        source.model_copy(deep=True),
        execute_leaf=_success,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )
    assert arm_b.metadata is not None
    assert arm_b.metadata.schedule_sha256 == result.metadata.schedule_sha256


def test_arm_a_cannot_be_mislabeled_as_compiled_plan_execution():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    with pytest.raises(ValueError, match="flat planner control"):
        execute_plan(
            plan,
            execute_leaf=_success,
            arm=PlanExecutionArm.A,
            safety=SafetyEnvelopeState(enabled=False),
        )


def test_unconfigured_execution_arm_defaults_to_flat_control():
    assert plan_execution_arm_from_env({}) is PlanExecutionArm.A


def test_resume_refuses_to_switch_execution_arms():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )
    completed = execute_plan(
        plan,
        execute_leaf=_success,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )
    assert completed.outcome is PlanExecutionOutcome.COMPLETED

    with pytest.raises(ValueError, match="different execution arm"):
        execute_plan(
            plan,
            execute_leaf=_success,
            arm=PlanExecutionArm.C,
            safety=SafetyEnvelopeState(enabled=False),
        )


def test_arm_c_applies_recursive_composite_group_hierarchy():
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
    plan = expand(
        _catalog(first, second, inner, outer),
        [Invocation(skill_name="task-a"), Invocation(skill_name="task-b")],
        "Do A and B",
        plan_id="recursive-execution",
    )
    scopes = []

    def execute(leaf, scope):
        scopes.append(scope)
        return _success(leaf, scope)

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.metadata is not None
    assert result.metadata.composite_group_ids == ("pack-1", "pack-1.1")
    assert result.metadata.composite_groups_applied == 2
    assert all(
        scope.composite_group_path == ("pack-1", "pack-1.1")
        for scope in scopes
    )
    assert all(
        scope.composite_skill_path == ("a-outer", "z-inner")
        for scope in scopes
    )
    instruction = render_leaf_instruction(plan.leaves[0], scopes[0])
    assert "a-outer > z-inner" in instruction
    assert "pack-1 > pack-1.1" in instruction


def test_safety_envelope_wall_origin_survives_state_round_trip():
    safety = SafetyEnvelopeState(
        wall_time_limit_s=30,
        started_at_epoch_s=100.0,
    )

    assert safety.wall_time_expired(now_epoch_s=129.9) is False
    assert safety.wall_time_expired(now_epoch_s=130.0) is True

    restored = SafetyEnvelopeState.from_state(safety.to_state())
    assert restored.started_at_epoch_s == 100.0
    assert restored.wall_time_expired(now_epoch_s=130.0) is True


def test_resumed_leaf_result_is_consumed_without_reexecuting_that_leaf():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
        plan_id="resume-plan",
    )
    first_leaf = plan.leaves[0]
    first_leaf.status = "needs-user"
    called: list[str] = []

    def execute(leaf, _scope):
        called.append(leaf.goal_id)
        return _success(leaf, _scope)

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
        resumed_from_goal_id=first_leaf.goal_id,
        resumed_leaf_result=SimpleNamespace(
            exhausted=False,
            suspended=False,
            command_call_ids=("resumed-call",),
        ),
    )

    assert result.suspended is False
    assert first_leaf.status == "done"
    assert first_leaf.command_call_ids == ("resumed-call",)
    assert first_leaf.goal_id not in called
    assert called == [plan.leaves[1].goal_id]
    assert plan.budget_consumed == 2


def test_restored_leaf_checkpoint_skips_completed_leaf():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    original = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
        plan_id="restore-leaf-checkpoint",
    )
    original.leaves[0].command_call_ids = ("checkpointed-alpha",)
    original.leaves[0].status = "done"
    restored = type(original).model_validate(
        original.model_dump(mode="json")
    )
    called: list[str] = []

    def execute(leaf, scope):
        called.append(leaf.goal_id)
        return _success(leaf, scope)

    result = execute_plan(
        restored,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert called == [restored.leaves[1].goal_id]
    assert result.outcome is PlanExecutionOutcome.COMPLETED
    assert all(leaf.status == "done" for leaf in restored.leaves)


def test_restored_parent_status_is_reconciled_before_dependency_scheduling():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )
    first_task, second_task = plan.public_nodes
    first_leaf, second_leaf = plan.leaves
    first_leaf.command_call_ids = ("checkpointed-alpha",)
    first_leaf.status = "done"
    assert first_task.status == "not-reached"
    second_leaf.prerequisites = (first_task.goal_id,)
    called: list[str] = []

    result = execute_plan(
        plan,
        execute_leaf=lambda leaf, scope: (
            called.append(leaf.goal_id),
            _success(leaf, scope),
        )[1],
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert called == [second_leaf.goal_id]
    assert first_task.status == "done"
    assert second_task.status == "done"
    assert result.outcome is PlanExecutionOutcome.COMPLETED


def test_completed_leaf_cannot_consume_a_resumed_result():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )
    plan.leaves[0].command_call_ids = ("already-complete",)
    plan.leaves[0].status = "done"

    with pytest.raises(ValueError, match="active needs-user leaf"):
        execute_plan(
            plan,
            execute_leaf=_success,
            arm=PlanExecutionArm.B,
            safety=SafetyEnvelopeState(enabled=False),
            resumed_from_goal_id=plan.leaves[0].goal_id,
            resumed_leaf_result=SimpleNamespace(
                exhausted=False,
                suspended=False,
                command_call_ids=("must-not-replace",),
            ),
        )
    assert plan.leaves[0].command_call_ids == ("already-complete",)


def test_skipped_prerequisite_never_becomes_runnable_after_restore():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )
    first, second = plan.leaves
    first.status = "skipped"
    first.failure_reason = "prerequisite-not-complete:earlier"
    second.prerequisites = (first.goal_id,)
    called: list[str] = []

    result = execute_plan(
        plan,
        execute_leaf=lambda leaf, _scope: called.append(leaf.goal_id),
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert called == []
    assert second.status == "skipped"
    assert result.outcome is PlanExecutionOutcome.PARTIAL


def test_non_budget_plan_account_is_not_mislabeled_as_exhaustion():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    result = execute_plan(
        plan,
        execute_leaf=lambda _leaf, _scope: SimpleNamespace(
            exhausted=False,
            suspended=False,
            command_call_ids=(),
        ),
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.answer
    assert result.exhausted is False
    assert any(leaf.status == "blocked" for leaf in plan.leaves)
    assert all(
        leaf.status in {"blocked", "skipped", "not-reached"}
        for leaf in plan.leaves
    )


def test_leaf_level_safety_censor_is_not_reported_as_exhaustion():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="leaver-sweep",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Devon Morrison is leaving",
    )

    result = execute_plan(
        plan,
        execute_leaf=lambda _leaf, _scope: SimpleNamespace(
            exhausted=False,
            suspended=False,
            censored=True,
            censored_reason="wall-time-cutoff",
            command_call_ids=(),
        ),
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(),
    )

    assert result.censored is True
    assert result.censored_reason == "wall-time-cutoff"
    assert result.exhausted is False


def test_complete_plan_terminalizes_every_leaf_and_public_parent():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )

    result = execute_plan(
        plan,
        execute_leaf=_success,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.outcome is PlanExecutionOutcome.COMPLETED
    assert result.exhausted is False
    assert all(leaf.status == "done" for leaf in plan.leaves)
    assert all(node.status == "done" for node in plan.public_nodes)
    assert "Plan outcome: completed." in result.answer
    assert "2 of 2 leaves done." in result.answer


def test_unresolved_delayed_input_is_partial_needs_user_not_exhaustion():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="account-walk",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Walk Devon Morrison's first account",
    )

    result = execute_plan(
        plan,
        execute_leaf=_success,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.outcome is PlanExecutionOutcome.NEEDS_USER
    assert result.needs_user is True
    assert result.suspended is False
    assert result.exhausted is False
    assert any(node.status == "needs-user" for node in plan.nodes)


def test_delayed_capture_expands_and_executes_newly_runnable_leaves():
    plan = expand(
        _fixture_catalog(),
        [
            Invocation(
                skill_name="account-walk",
                slots={"identity_query": "Devon Morrison"},
            )
        ],
        "Walk Devon Morrison's first account",
    )
    outputs: list[CommandOutput] = []
    executed: list[str] = []
    scopes = {}

    def execute(leaf, scope):
        executed.append(leaf.goal_id)
        scopes[leaf.goal_id] = scope
        if "list_accounts" in (leaf.executable_goal_text or ""):
            outputs.append(
                CommandOutput(
                    command_name="list_accounts",
                    command_call_id="call-list-accounts",
                    command_response=CommandResponse(
                        response="one account",
                        artifacts={"account_uids": ["acct-first"]},
                    ),
                )
            )
        return _success(leaf, scope)

    def resolve(record):
        delayed = next(
            node for node in record.nodes if node.skill == "account-portrait"
        )
        if not delayed.unbound_slots():
            return False
        return (
            bind_captured(
                record,
                delayed,
                "account_uid",
                outputs,
                catalog=_fixture_catalog(),
            )
            is not None
        )

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
        resolve_delayed_bindings=resolve,
        current_navigation_context=lambda: "Identity",
    )

    delayed = next(
        node for node in plan.nodes if node.skill == "account-portrait"
    )
    assert result.outcome is PlanExecutionOutcome.COMPLETED
    assert delayed.bindings["account_uid"].value == "acct-first"
    assert delayed.status == "done"
    assert len(executed) == len(plan.leaves)
    assert all(leaf.status == "done" for leaf in plan.leaves)
    captured_scope = next(
        scope
        for goal_id, scope in scopes.items()
        if goal_id.startswith(f"{delayed.goal_id}.")
    )
    captured = dict(captured_scope.task_bindings)["account_uid"]
    assert captured.source == "captured"
    assert captured.kind == "captured_handle"
    assert captured.command_call_id == "call-list-accounts"
    assert "call-list-accounts" in captured_scope.producer_call_ids
    assert captured_scope.navigation_context == "Identity"


def test_provider_timeout_after_completed_leaf_preserves_checkpoint_evidence():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )
    checkpoints: list[tuple[str, str, tuple[str, ...]]] = []

    def execute(leaf, scope):
        if leaf.goal_id == plan.leaves[1].goal_id:
            raise LMTimeoutError("provider timeout", model="offline-test")
        return _success(leaf, scope)

    def checkpoint(record, leaf, reason):
        checkpoints.append(
            (leaf.goal_id, reason, tuple(record.execution.executed_leaf_goal_ids))
            if record.execution is not None
            else (leaf.goal_id, reason, ())
        )

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
        on_progress=checkpoint,
    )

    assert result.outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
    assert result.provider_timeout is True
    assert result.failure is not None
    assert result.failure.code == CODE_PROVIDER_TIMEOUT
    assert plan.leaves[0].status == "done"
    assert plan.leaves[1].status == "blocked"
    assert result.successful_leaf_goal_ids == (plan.leaves[0].goal_id,)
    assert [reason for _goal, reason, _ids in checkpoints] == [
        "done",
        "provider-timeout",
    ]
    assert "1 of 2 leaves done." in result.answer


def test_provider_timeout_after_partial_leaf_keeps_its_command_references():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )

    def execute(leaf, _scope):
        leaf.command_call_ids = ("call-before-timeout",)
        raise LMTimeoutError("provider timeout", model="offline-test")

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
    assert plan.leaves[0].status == "blocked"
    assert plan.leaves[0].command_call_ids == ("call-before-timeout",)
    assert result.failure is not None
    assert result.failure.completed_work[0]["command_call_ids"] == (
        "call-before-timeout",
    )


def test_provider_timeout_before_first_progress_has_no_false_completion():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )

    result = execute_plan(
        plan,
        execute_leaf=lambda _leaf, _scope: (_ for _ in ()).throw(
            LMTimeoutError("provider timeout", model="offline-test")
        ),
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
    assert result.successful_leaf_goal_ids == ()
    assert plan.budget_consumed == 0
    assert plan.leaves[0].status == "blocked"
    assert plan.leaves[0].command_call_ids == ()


def test_wall_cutoff_is_censoring_with_deterministic_leaf_state():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )
    safety = SafetyEnvelopeState(
        wall_time_limit_s=10,
        started_at_epoch_s=100.0,
    )
    safety.wall_time_expired = lambda now_epoch_s=None: True

    result = execute_plan(
        plan,
        execute_leaf=_success,
        arm=PlanExecutionArm.B,
        safety=safety,
    )

    assert result.outcome is PlanExecutionOutcome.CENSORED
    assert result.censored is True
    assert result.provider_timeout is False
    assert result.exhausted is False
    assert plan.leaves[0].status == "blocked"
    assert plan.leaves[0].failure_reason == "censored:wall-time-cutoff"
    assert plan.public_nodes[0].status == "blocked"


def test_typed_task_failure_is_distinct_from_provider_timeout_and_censor():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )
    task_failure = TypedFailure(
        disposition="permanent",
        code="task-contract-failed",
        detail="offline test",
    )

    def execute(leaf, scope):
        if leaf.goal_id == plan.leaves[0].goal_id:
            return SimpleNamespace(
                exhausted=False,
                suspended=False,
                command_call_ids=("failed-call",),
                failure=task_failure,
            )
        return _success(leaf, scope)

    result = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert result.outcome is PlanExecutionOutcome.FAILED
    assert result.failure == task_failure
    assert result.provider_timeout is False
    assert result.censored is False
    assert plan.leaves[0].status == "blocked"
    assert plan.leaves[1].status == "done"


def test_truncation_type_and_goal_ids_survive_aggregate_plan_round_trip():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    beta = _skill("beta", "task", goal="Beta is done.", body="1. `beta`")
    plan = expand(
        _catalog(alpha, beta),
        [Invocation(skill_name="alpha"), Invocation(skill_name="beta")],
        "Do alpha and beta",
    )
    truncated_goal_id = plan.leaves[0].goal_id

    def execute(leaf, scope):
        result = _success(leaf, scope)
        if leaf.goal_id == truncated_goal_id:
            result.extraction_truncated = True
            result.extraction_failure = extraction_truncated_failure(
                max_tokens=4096
            )
        return result

    first = execute_plan(
        plan,
        execute_leaf=execute,
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert first.outcome is PlanExecutionOutcome.COMPLETED
    assert first.extraction_truncated_goal_ids == (truncated_goal_id,)
    assert first.extraction_failure is not None
    assert first.extraction_failure.code == CODE_EXTRACTION_TRUNCATED
    assert plan.execution is not None
    assert plan.execution.extraction_truncated_goal_ids == (
        truncated_goal_id,
    )
    assert plan.execution.extraction_truncation_failures[
        truncated_goal_id
    ].code == CODE_EXTRACTION_TRUNCATED

    restored = type(plan).model_validate(plan.model_dump(mode="json"))
    resumed = execute_plan(
        restored,
        execute_leaf=lambda *_args: pytest.fail(
            "completed leaves must not be replayed"
        ),
        arm=PlanExecutionArm.B,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert resumed.outcome is PlanExecutionOutcome.COMPLETED
    assert resumed.extraction_truncated_goal_ids == (truncated_goal_id,)
    assert resumed.extraction_failure is not None
    assert resumed.extraction_failure.code == CODE_EXTRACTION_TRUNCATED


def test_provider_timeout_type_survives_aggregate_plan_round_trip():
    alpha = _skill("alpha", "task", goal="Alpha is done.", body="1. `alpha`")
    plan = expand(
        _catalog(alpha),
        [Invocation(skill_name="alpha")],
        "Do alpha",
    )
    first = execute_plan(
        plan,
        execute_leaf=lambda *_args: (_ for _ in ()).throw(
            LMTimeoutError("provider timeout", model="offline-test")
        ),
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert first.outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
    assert plan.execution is not None
    assert plan.execution.terminal_failure is not None
    assert plan.execution.terminal_failure.code == CODE_PROVIDER_TIMEOUT

    restored = type(plan).model_validate(plan.model_dump(mode="json"))
    resumed = execute_plan(
        restored,
        execute_leaf=lambda *_args: pytest.fail(
            "a terminal provider failure must not replay a leaf"
        ),
        arm=PlanExecutionArm.C,
        safety=SafetyEnvelopeState(enabled=False),
    )

    assert resumed.outcome is PlanExecutionOutcome.PROVIDER_TIMEOUT
    assert resumed.failure is not None
    assert resumed.failure.code == CODE_PROVIDER_TIMEOUT
