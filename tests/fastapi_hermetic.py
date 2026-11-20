"""Shared helpers so FastAPI tests do not pollute the developer's state tree.

Pending Topology-B blobs under ``<state-root>/workflows/<id>/session_state/`` are
gitignored and survive across pytest runs. A leftover
``nonexistent_user.pending.json`` (awaiting_user=true) is restored on the next
session create and makes unrelated FastAPI tests fail with shifting names
(fix-j83). Every FastAPI ``app_module`` fixture must point the state root at a
per-test temp directory — same pattern as ``tests/test_checkpoint_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import dotenv_values

import fastworkflow


def init_fastapi_hermetic_env(
    env_file: str,
    passwords_file: str,
    speedict_dir: Path | str,
) -> dict[str, Any]:
    """``fastworkflow.init`` with FASTWORKFLOW_STATE_ROOT overridden to ``speedict_dir``.

    Returns the previous ``fastworkflow._env_vars`` so callers can restore in a
    fixture ``finally`` (an interpreter left on a deleted temp dir poisons the
    next test file).
    """
    previous_env = fastworkflow._env_vars
    env_vars = {
        **dotenv_values(env_file),
        **dotenv_values(passwords_file),
        "FASTWORKFLOW_STATE_ROOT": str(speedict_dir),
    }
    fastworkflow.init(env_vars)
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return previous_env or {}


def restore_fastapi_env(previous_env: dict[str, Any]) -> None:
    """Undo ``init_fastapi_hermetic_env`` for the next test / file."""
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.init(previous_env)


PLACEHOLDER_KEY = "<placeholder: no key>"
TEMPLATE_MODEL = "mistral/mistral-small-latest"
_KEY_PREFIX = "LITELLM_API_KEY_"


def write_placeholder_env(directory: Path | str) -> tuple[str, str]:
    """Write ``fastworkflow.env`` and ``fastworkflow.passwords.env`` into *directory*; return both paths.

    Built only from the tracked examples (never ``env/.env`` or ``passwords/.env``), so the result is the same on
    every machine. The env file is the template followed by one block that sets every role's model to the template's
    default, so a developer's local model lines in the working copy cannot change what a test calls. The passwords
    file holds one placeholder per ``LITELLM_API_KEY_<ROLE>`` NAME the passwords template declares and no
    ``JEV_API_KEY``; the placeholder is written in the shipped style (``<``), which ``llm_credentials`` reads as no key.
    """
    examples = Path(fastworkflow.get_fastworkflow_package_path()) / "examples"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    roles = [
        name[len(_KEY_PREFIX):]
        for name in dotenv_values(examples / "fastworkflow.passwords.env")
        if name.startswith(_KEY_PREFIX)
    ]
    env_file = directory / "fastworkflow.env"
    env_file.write_text(
        (examples / "fastworkflow.env").read_text()
        + "\n# Test override: the template's model for every role.\n"
        + "".join(f"LLM_{role}={TEMPLATE_MODEL}\n" for role in roles)
    )
    passwords_file = directory / "fastworkflow.passwords.env"
    passwords_file.write_text("".join(f"{_KEY_PREFIX}{role}={PLACEHOLDER_KEY}\n" for role in roles))
    return str(env_file), str(passwords_file)
