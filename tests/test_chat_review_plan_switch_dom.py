"""The chat SPA's review form on a step with ambiguous planning-command match (bd fix-1ma8.1).

Discovered by ``tests.browser_validation`` because this module names ``TEST_JSDOM_ROOT``. The page under test is the
assembled chatbot SPA; the reply the select composes is the edit ``tests/test_plan_match_ambiguity.py`` feeds to the
real review reader.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_review_form_plan_switch():
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_review_plan_switch_dom.cjs")
    src = (
        Path(__file__).resolve().parents[1]
        / "fastworkflow"
        / "run_chatbot"
        / "static"
        / "src"
    )
    result = subprocess.run(
        ["node", str(script), dependency, str(src)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
