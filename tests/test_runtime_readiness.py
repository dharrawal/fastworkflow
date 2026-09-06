"""Exact runtime-path introspection for EXP-028 pre-paid readiness."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.plan import PlanConfigurationError
from fastworkflow.runtime_manifest import load_manifest, merge_and_gate
from fastworkflow.runtime_readiness import runtime_readiness_snapshot


FIXTURE_WORKFLOW = Path(__file__).parent / "fixtures" / "skills_workflow"


def _metadata(mode: str = "enforce"):
    manifest = load_manifest(str(FIXTURE_WORKFLOW))
    assert manifest is not None
    return merge_and_gate(
        manifest,
        deployment_features={"skills_v1": mode},
    )


def _env(arm: str, decomposition: str) -> dict[str, str]:
    return {
        "FW_PLAN_DECOMPOSITION": decomposition,
        "FW_PLAN_EXECUTION_ARM": arm,
        "FW_PLAN_STRESS_MODE": "1",
        "FW_PLAN_WALL_TIME_LIMIT_SECONDS": "1800",
        "FW_TURN_DEADLINE_SECONDS": "1980",
    }


def test_arm_a_reports_flat_path_without_loading_catalogue(monkeypatch):
    monkeypatch.setenv("FW_PLAN_EXECUTION_ARM", "c")

    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "off"),
        metadata=_metadata(),
    )

    assert snapshot["configuration_valid"] is True
    assert snapshot["execution_path"] == "flat"
    assert snapshot["packing_configuration"] == "not-compiled"
    assert snapshot["catalog_loaded"] is False
    assert snapshot["plan_execution_arm"] == "a"


@pytest.mark.parametrize(
    ("arm", "execution_path", "packing"),
    (
        ("b", "task-only-unpacked", "ignored"),
        ("c", "task-only-packed", "applied"),
    ),
)
def test_treatment_arms_load_task_cards_and_report_packing_configuration(
    arm,
    execution_path,
    packing,
):
    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env(arm, "enforce"),
        metadata=_metadata(),
    )

    assert snapshot["configuration_valid"] is True
    assert snapshot["execution_path"] == execution_path
    assert snapshot["packing_configuration"] == packing
    assert snapshot["catalog_loaded"] is True
    assert snapshot["catalog_mode"] == "enforce"
    assert snapshot["task_card_count"] > 0
    assert snapshot["selector_surface_task_only"] is True


def test_treatment_readiness_rejects_disabled_manifest_feature():
    with pytest.raises(PlanConfigurationError):
        runtime_readiness_snapshot(
            str(FIXTURE_WORKFLOW),
            env=_env("b", "enforce"),
            metadata=_metadata("off"),
        )


def test_invalid_cross_arm_configuration_is_not_ready():
    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "enforce"),
        metadata=_metadata(),
    )

    assert snapshot["configuration_valid"] is False
    assert snapshot["execution_path"] == "invalid"


def test_ordinary_defaults_remain_a_valid_bounded_flat_path():
    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env={},
        metadata=_metadata(),
    )

    assert snapshot["configuration_valid"] is True
    assert snapshot["plan_decomposition"] == "off"
    assert snapshot["plan_execution_arm"] == "a"
    assert snapshot["execution_path"] == "flat"
    assert snapshot["stress_mode"] is False
    assert snapshot["safety_wall_seconds"] is None
    assert snapshot["turn_deadline_seconds"] is None


def test_invalid_stress_mode_value_is_refused():
    with pytest.raises(ValueError, match="FW_PLAN_STRESS_MODE"):
        runtime_readiness_snapshot(
            str(FIXTURE_WORKFLOW),
            env={"FW_PLAN_STRESS_MODE": "sometimes"},
            metadata=_metadata(),
        )


# ---------------------------------------------------------------------------
# EXP-028 Gate 4 v5 (ido-mn1.6.25 / design section 4): the arm-invariant runtime
# facts that change every prompt in every arm and would be invisible in a
# v4-shaped record.  All of them are env-overridable, so "arm-invariant" is only
# checkable if the per-server snapshot carries them.
# ---------------------------------------------------------------------------

def test_the_snapshot_carries_the_presentation_cap_and_the_eight_constants():
    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "off"),
        metadata=_metadata(),
    )

    assert snapshot["presented_result_max_bytes"] == 32768
    assert set(snapshot["extraction_constants"]) == {
        "render_factor",
        "prose_allowance_tokens",
        "min_tokens",
        "max_tokens_ceiling",
        "bytes_per_token",
        "gen_tokens_per_second",
        "timeout_safety",
        "ttft_allowance_seconds",
    }
    assert snapshot["extraction_constants"]["render_factor"] == 2.0
    assert snapshot["extraction_constants"]["min_tokens"] == 4096
    assert snapshot["extraction_constants"]["max_tokens_ceiling"] == 36864
    assert snapshot["extraction_constants"]["bytes_per_token"] == 2.0


def test_the_reported_constants_are_the_ones_that_will_actually_apply(
    monkeypatch,
):
    """Read through the runtime's own env resolution, not off the module.  A
    snapshot reporting shipped defaults while the server runs an override would
    certify the wrong runtime, which is what the assertion exists to catch."""
    monkeypatch.setenv("FW_PRESENTED_RESULT_MAX_BYTES", "65536")
    monkeypatch.setenv("FW_EXTRACT_RENDER_FACTOR", "3.5")

    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "off"),
        metadata=_metadata(),
    )

    assert snapshot["presented_result_max_bytes"] == 65536
    assert snapshot["extraction_constants"]["render_factor"] == 3.5


def test_the_snapshot_carries_the_command_surface_count():
    metadata = _metadata()
    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "off"),
        metadata=metadata,
    )

    assert snapshot["command_surface_count"] == len(metadata.commands)
    assert snapshot["command_surface_count"] > 0


def test_the_snapshot_pins_the_trained_model_version_and_never_invents_one(
    tmp_path,
):
    """`___command_info` is gitignored and under no hashed root, so a retrain
    moves the system under measurement while the tree hash, the revision and the
    dirty flag all stay identical.  `current.json/version_id` is the one field
    that moves — and a workflow with no trained set reports None rather than a
    placeholder, because a placeholder pin is worse than an absent one."""
    from fastworkflow.runtime_readiness import workflow_model_version

    assert workflow_model_version(str(tmp_path)) is None

    info = tmp_path / "___command_info"
    info.mkdir()
    (info / "current.json").write_text(
        '{"version_id": "20260905T042633Z-46ef78"}', encoding="utf-8"
    )
    assert workflow_model_version(str(tmp_path)) == "20260905T042633Z-46ef78"

    (info / "current.json").write_text("not json", encoding="utf-8")
    assert workflow_model_version(str(tmp_path)) is None

    snapshot = runtime_readiness_snapshot(
        str(FIXTURE_WORKFLOW),
        env=_env("a", "off"),
        metadata=_metadata(),
    )
    assert "workflow_model_version" in snapshot
