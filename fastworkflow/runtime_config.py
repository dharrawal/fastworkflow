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
from typing import Mapping, Optional

# The compatible current value: 3.1.2's `max_iters=25` default, promoted to a
# deployment setting. Changing this default changes every deployment that has
# not declared one, so it moves only with a recorded decision.
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
