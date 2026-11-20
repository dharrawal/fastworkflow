"""The built-in plan metadata extraction workflow's locally trained intent head, as the test suite sees it.

The head lives under ``fastworkflow/_workflows/plan_metadata_extraction/___command_info/`` which is gitignored, so a clean
checkout and CI have none. Tests that assert the head's own quality carry the ``requires_plan_head`` marker
(registered and enforced in ``tests/conftest.py``); this module is the one place that knows where the head is, what
version it is, and what the message for its absence says. Names only: nothing here reads a model.
"""
from __future__ import annotations

import json
import os

import fastworkflow

PLAN_WORKFLOW = fastworkflow.get_internal_workflow_path("plan_metadata_extraction")
HEAD_CONTEXT = "ProposalReview"
COMMAND_INFO = os.path.join(PLAN_WORKFLOW, "___command_info")
HEAD_DIR = os.path.join(COMMAND_INFO, HEAD_CONTEXT)
REQUIRE_HEADS_ENV = "FW_REQUIRE_HEADS"
HOW_TO_GET = (
    f"train it with `fastworkflow train {PLAN_WORKFLOW} <env_file> <passwords_file>` "
    "(a local, gitignored artifact: a clean checkout and CI do not have it)"
)


def head_trained() -> bool:
    """True when the ProposalReview head has been trained here (its ``threshold.json`` exists)."""
    return os.path.isfile(os.path.join(HEAD_DIR, "threshold.json"))


def head_version_id() -> str | None:
    """The id the ``current`` pointer names, or None. Reads ``current.json``, falling back to the symlink target."""
    try:
        with open(os.path.join(COMMAND_INFO, "current.json"), encoding="utf-8") as handle:
            version = json.load(handle).get("version_id")
        if isinstance(version, str) and version:
            return version
    except (OSError, ValueError):
        pass
    try:
        return os.path.basename(os.readlink(os.path.join(COMMAND_INFO, "current"))) or None
    except OSError:
        return None


def heads_required() -> bool:
    """``FW_REQUIRE_HEADS=1`` in the process environment turns a missing head from a skip into a failure.

    Read from ``os.environ`` directly, not ``fastworkflow.get_env_var``: this is a switch for whoever runs pytest, and
    must not depend on which env files a test happened to load.
    """
    return os.environ.get(REQUIRE_HEADS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def head_status_line() -> str:
    if head_trained():
        return f"plan ProposalReview head: current version {head_version_id() or 'unknown'}"
    return "plan ProposalReview head: NOT TRAINED (head-marked tests " + (
        "will FAIL: FW_REQUIRE_HEADS is set)" if heads_required() else "will skip)")


def missing_head_message() -> str:
    return (
        f"the plan {HEAD_CONTEXT} intent head is not trained here ({HEAD_DIR}); {HOW_TO_GET}. "
        f"{REQUIRE_HEADS_ENV}=1 turns this skip into a failure."
    )
