"""Credential-free runtime introspection for deployment readiness probes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from fastworkflow.plan import PlanMode, plan_mode_from_env, require_catalog
from fastworkflow.plan_execution import (
    PlanExecutionArm,
    plan_execution_arm_from_env,
    plan_stress_mode_from_env,
)
from fastworkflow.runtime_manifest import RuntimeMetadata, get_runtime_metadata
from fastworkflow.skill_catalog import load_skill_catalog
from fastworkflow import result_handles
from fastworkflow.utils import react


def _positive_int(raw: Any) -> Optional[int]:
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def workflow_model_version(workflow_path: str) -> Optional[str]:
    """The trained model set this workflow is currently serving, or None.

    `___command_info/` is gitignored and is under none of the roots any source
    fingerprint hashes, so a retrain changes the system under measurement while
    the tree hash, the git revision and the dirty flag all stay byte-identical.
    `current.json/version_id` is the one cheap field that moves, and a readiness
    snapshot that does not carry it cannot tell two runs apart across a retrain.

    Read, never cached: only a successful train publishes, so the value is a
    fact about the moment the probe asked. Absent or unreadable answers None
    rather than raising -- a workflow with no trained set is a real deployment
    state, and a readiness probe reports it rather than refusing to describe
    the runtime at all.
    """
    path = Path(workflow_path) / "___command_info" / "current.json"
    try:
        version = json.loads(path.read_text(encoding="utf-8")).get("version_id")
    except (OSError, ValueError):
        return None
    return str(version) if version else None


def extraction_constants() -> dict[str, float]:
    """The eight constants that size one extraction call, as they will apply.

    Read through `react`'s own env resolution rather than off the module
    attributes, because every one of them is env-overridable per process
    (`FW_EXTRACT_*`). A snapshot reporting the shipped defaults while the server
    runs an override would certify the wrong runtime, which is the exact class
    of drift the per-cell snapshot assertion exists to catch (EXP-028 Gate 4 v5
    design section 4).
    """
    return {
        "render_factor": react._extract_env_number(
            "FW_EXTRACT_RENDER_FACTOR", react.EXTRACT_RENDER_FACTOR
        ),
        "prose_allowance_tokens": react._extract_env_number(
            "FW_EXTRACT_PROSE_ALLOWANCE_TOKENS",
            react.EXTRACT_PROSE_ALLOWANCE_TOKENS,
        ),
        "min_tokens": react._extract_env_number(
            "FW_EXTRACT_MIN_TOKENS", react.EXTRACT_MIN_TOKENS
        ),
        "max_tokens_ceiling": react._extract_env_number(
            "FW_EXTRACT_MAX_TOKENS_CEILING", react.EXTRACT_MAX_TOKENS_CEILING
        ),
        "bytes_per_token": react._extract_env_number(
            "FW_EXTRACT_BYTES_PER_TOKEN", react.EXTRACT_BYTES_PER_TOKEN
        ),
        "gen_tokens_per_second": react._extract_env_number(
            "FW_EXTRACT_GEN_TOKENS_PER_SECOND",
            react.EXTRACT_GEN_TOKENS_PER_SECOND,
        ),
        "timeout_safety": react._extract_env_number(
            "FW_EXTRACT_TIMEOUT_SAFETY", react.EXTRACT_TIMEOUT_SAFETY
        ),
        "ttft_allowance_seconds": react._extract_env_number(
            "FW_EXTRACT_TTFT_ALLOWANCE_SECONDS",
            react.EXTRACT_TTFT_ALLOWANCE_SECONDS,
        ),
    }


def runtime_readiness_snapshot(
    workflow_path: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    metadata: Optional[RuntimeMetadata] = None,
) -> dict[str, Any]:
    """Report the effective planner path without exposing deployment secrets."""
    source = os.environ if env is None else env
    effective_metadata = (
        get_runtime_metadata(workflow_path)
        if metadata is None
        else metadata
    )
    mode = plan_mode_from_env(source)
    arm = plan_execution_arm_from_env(source)
    stress_mode = plan_stress_mode_from_env(source)
    safety_seconds = _positive_int(
        source.get("FW_PLAN_WALL_TIME_LIMIT_SECONDS")
    )
    turn_deadline_seconds = _positive_int(
        source.get("FW_TURN_DEADLINE_SECONDS")
    )

    catalog_loaded = False
    catalog_mode = "off"
    catalog_fingerprint = None
    catalog_skill_count = 0
    task_card_count = 0
    selector_surface_task_only = False

    if mode is PlanMode.OFF:
        execution_path = "flat"
        packing_configuration = "not-compiled"
        path_configuration_valid = arm is PlanExecutionArm.A
    else:
        catalog = load_skill_catalog(workflow_path, effective_metadata)
        require_catalog(mode, catalog)
        cards = catalog.cards()
        task_cards = tuple(card for card in cards if card.level == "task")
        catalog_loaded = True
        catalog_mode = catalog.mode
        catalog_fingerprint = catalog.fingerprint
        catalog_skill_count = len(catalog)
        task_card_count = len(task_cards)
        selector_surface_task_only = bool(task_cards)
        if arm is PlanExecutionArm.B:
            execution_path = "task-only-unpacked"
            packing_configuration = "ignored"
            path_configuration_valid = True
        elif arm is PlanExecutionArm.C:
            execution_path = "task-only-packed"
            packing_configuration = "applied"
            path_configuration_valid = True
        else:
            execution_path = "invalid"
            packing_configuration = "invalid"
            path_configuration_valid = False

    deadline_configuration_valid = (
        not stress_mode
        or (
            safety_seconds is not None
            and turn_deadline_seconds is not None
            and turn_deadline_seconds >= safety_seconds
        )
    )
    configuration_valid = (
        path_configuration_valid
        and deadline_configuration_valid
        and (
            mode is PlanMode.OFF
            or (
                catalog_loaded
                and catalog_mode == PlanMode.ENFORCE.value
                and selector_surface_task_only
            )
        )
    )
    return {
        "effective_features": (
            dict(sorted(effective_metadata.feature_modes.items()))
            if effective_metadata is not None
            else {}
        ),
        "plan_decomposition": mode.value,
        "plan_execution_arm": arm.value,
        "execution_path": execution_path,
        "packing_configuration": packing_configuration,
        "stress_mode": stress_mode,
        "safety_wall_seconds": safety_seconds,
        "turn_deadline_seconds": turn_deadline_seconds,
        "catalog_loaded": catalog_loaded,
        "catalog_mode": catalog_mode,
        "catalog_fingerprint": catalog_fingerprint,
        "catalog_skill_count": catalog_skill_count,
        "task_card_count": task_card_count,
        "selector_surface_task_only": selector_surface_task_only,
        # EXP-028 Gate 4 v5 design section 4. Every one of these is
        # env-overridable and every one of them changes every prompt in every
        # arm, so "arm-invariant" is only checkable if the per-cell snapshot
        # carries them and the cross-arm check compares them. None of them
        # would be visible in a v4-shaped record.
        "presented_result_max_bytes": result_handles.presented_max_bytes(),
        "extraction_constants": extraction_constants(),
        "command_surface_count": (
            len(effective_metadata.commands)
            if effective_metadata is not None
            else 0
        ),
        "workflow_model_version": workflow_model_version(workflow_path),
        "configuration_valid": configuration_valid,
    }
