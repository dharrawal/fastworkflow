"""Deployment-level runtime configuration (arch §6.0).

Today the ReAct iteration limit is a default parameter value on
``initialize_workflow_tool_agent`` that no production caller overrides — there
is no configuration surface at all (verification register D7). FW-REQ-001
clause 5 requires one, so this module introduces it, and the default is the
compatible current value of 25.

The precedence is **restrictive**: every layer may lower the limit and none may
raise it.

    effective limit = minimum(deployment maximum,
                              workflow manifest maximum,
                              task-contract limit,
                              host/request limit)

A leaf module by arch §22 — it takes an environment mapping rather than
importing the package to reach ``fastworkflow._env_vars``. Callers wanting the
env-file-then-OS-env precedence the rest of the runtime uses pass
``runtime_manifest.deployment_env(fastworkflow._env_vars)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from typing import Mapping, Optional

# ---------------------------------------------------------------------------
# Where the default budget comes from (`ido-24b.4`)
# ---------------------------------------------------------------------------
# Until 2026-09-02 the default was 3.1.2's `max_iters=25` and nothing behind it
# — a constant nobody derived, carried forward because it was already there.
# The measurement that would justify a number now exists, so the number is
# derived from it and the derivation is checked rather than asserted.
#
# MEASURED, from ido's `g2c-baseline-1` control run (320 attempts, policy off);
# the per-contract table lives in `evaluation/cases/g2a_corpus.py`:
#
#   walk overhead          4 commands   resolve the subject, list the roster,
#                                       and the context moves in between; taken
#                                       as the residual of total commands over
#                                       per-item cost on the same run
#   commands per item      2.20 (cross-system-privilege-audit) …
#                          6.33 (application-recertification), p90 per contract
#
# The p90 rather than the mean, and the dearest contract rather than the
# average one, because the two errors are not symmetric: under-budgeting
# produces a walk that is truncated and then reported as whole (EXP-027 — 13 of
# 15 fabrications were at the ceiling), while over-budgeting only spends
# iterations a finished turn never uses.
MEASURED_WALK_OVERHEAD_COMMANDS = 4
MEASURED_COMMANDS_PER_ITEM_P90 = 6.33

# CHOSEN, and a policy rather than a measurement: how many population items one
# logical turn is expected to cover before the planner has to allocate another
# turn. Three is the modal and median magnitude of the cardinality corpus (15
# of 32 G2C cases), and it is the most the dearest contract fits inside the
# inherited 25 — so this bead records the reasoning without moving any
# deployment. It is deliberately NOT "enough items to finish the largest case":
# the owner's ranking (2026-09-02) is that surfacing exhaustion so a planner can
# allocate more turns beats a bigger budget, because a bigger budget only moves
# the cliff.
ITEMS_PER_TURN_TARGET = 3


def derive_react_max_iterations(
    commands_per_item: float = MEASURED_COMMANDS_PER_ITEM_P90,
    *,
    items_per_turn: int = ITEMS_PER_TURN_TARGET,
    walk_overhead: int = MEASURED_WALK_OVERHEAD_COMMANDS,
) -> int:
    """The smallest budget that completes ``items_per_turn`` items in one turn.

    This is how the default is re-derived when the command surface changes: a
    command split in two, a new confirmation step, or a context move added to a
    walk all change commands-per-item, and the only honest response is to
    re-measure it on a control run and call this again. A deployment whose
    workload is not ido's does the same with its own numbers.

    Note the direction the layers compose in: a manifest or task contract can
    only LOWER the effective limit (``effective_react_max_iterations``), so a
    deployment that re-derives a *larger* budget raises it here or through
    ``FASTWORKFLOW_REACT_MAX_ITERATIONS`` — not in a manifest.
    """
    return max(1, ceil(walk_overhead + items_per_turn * commands_per_item))


def items_within_react_budget(
    limit: int,
    commands_per_item: float = MEASURED_COMMANDS_PER_ITEM_P90,
    *,
    walk_overhead: int = MEASURED_WALK_OVERHEAD_COMMANDS,
) -> int:
    """How many population items ``limit`` iterations can actually walk.

    The inverse of the derivation, and the more useful half when reading a
    result: at 25 the dearest contract reaches 3 items and the cheapest 9, so a
    task over more than that is a multi-turn task whatever it was called.
    """
    if commands_per_item <= 0:
        raise ValueError("commands_per_item must be positive")
    return max(0, int((limit - walk_overhead) // commands_per_item))


#: 23 = 4 + 3 x 6.33, from the constants above.
DERIVED_REACT_MAX_ITERATIONS = derive_react_max_iterations()

# The deployment default. Held at the inherited 25 rather than lowered to the
# derived 23: 25 is the budget every measurement in the programme — including
# the commands-per-item figures the derivation is built on — was taken under,
# and re-baselining that to reclaim two iterations buys nothing and invalidates
# the corpus feasibility split (`G2ACase.within_iteration_budget`). What the
# derivation buys is a floor with provenance: the default must cover
# DERIVED_REACT_MAX_ITERATIONS, and `test_logical_turn_budget.py` fails if a
# re-measured command surface pushes the floor above it.
#
# Changing this default changes every deployment that has not declared one, so
# it moves only with a recorded decision — now with a recorded derivation too.
DEFAULT_REACT_MAX_ITERATIONS = 25

REACT_MAX_ITERATIONS_ENV_VAR = "FASTWORKFLOW_REACT_MAX_ITERATIONS"


@dataclass(frozen=True)
class RuntimeConfig:
    """Deployment-wide runtime limits.

    Frozen because it is read on every turn start and a mutable process-wide
    limit is a limit nobody can attribute a turn to.
    """

    react_max_iterations: int = DEFAULT_REACT_MAX_ITERATIONS

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> tuple["RuntimeConfig", list[str]]:
        """Read the deployment configuration, returning ``(config, problems)``.

        Problems are returned rather than raised so a bad deployment value is
        reported together with manifest problems in one startup conformance
        failure, exactly as ``parse_deployment_features`` does. When a value is
        rejected the default is used, and the *problem* is what stops startup —
        never a silently substituted limit.
        """
        source = os.environ if env is None else env
        problems: list[str] = []
        react_max_iterations = DEFAULT_REACT_MAX_ITERATIONS

        raw = source.get(REACT_MAX_ITERATIONS_ENV_VAR)
        if raw is not None and str(raw).strip():
            parsed = _positive_int(str(raw).strip())
            if parsed is None:
                problems.append(
                    f"{REACT_MAX_ITERATIONS_ENV_VAR} must be a positive integer, "
                    f"got {raw!r}"
                )
            else:
                react_max_iterations = parsed

        return cls(react_max_iterations=react_max_iterations), problems

    def effective_react_max_iterations(
        self,
        *,
        manifest_limit: Optional[int] = None,
        contract_limit: Optional[int] = None,
        host_limit: Optional[int] = None,
    ) -> int:
        """The restrictive minimum over every layer that declared a limit.

        A layer that declares nothing (``None``) does not participate — it is
        not treated as zero and not treated as a licence to raise the
        deployment maximum. A declared value at or above the deployment maximum
        is simply not the minimum, which is how "may lower but cannot exceed"
        falls out of one expression instead of a special case.

        Non-positive declarations are ignored with the same reasoning as
        ``from_env``: a limit of zero would make every turn exhausted before its
        first step, which is a configuration failure rather than a budget.
        """
        candidates = [self.react_max_iterations]
        candidates.extend(
            value
            for value in (manifest_limit, contract_limit, host_limit)
            if value is not None and value > 0
        )
        return min(candidates)


def _positive_int(raw: str) -> Optional[int]:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# Process-wide default, so a caller with no configuration in hand still reads a
# real limit instead of inventing one. Deliberately *not* a cache of the env:
# the entry points resolve configuration at startup and register it, and a
# lazily-read global would let a test's environment leak into a running turn.
DEFAULT_RUNTIME_CONFIG = RuntimeConfig()

_runtime_config: RuntimeConfig = DEFAULT_RUNTIME_CONFIG


def register_runtime_config(config: RuntimeConfig) -> None:
    """Publish the startup-resolved configuration for the process lifetime.

    Modelled on ``runtime_manifest.register_runtime_metadata``: explicit, so a
    build step or a test that resolves configuration for some other purpose
    cannot change what a running turn reads.
    """
    global _runtime_config
    _runtime_config = config


def get_runtime_config() -> RuntimeConfig:
    """The registered configuration, or the compatible defaults."""
    return _runtime_config


def clear_runtime_config() -> None:
    """Drop the registration (test isolation)."""
    global _runtime_config
    _runtime_config = DEFAULT_RUNTIME_CONFIG
