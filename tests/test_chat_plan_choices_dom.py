"""The chat SPA's plan buttons under a part no-agent mode did not run (bd fix-1ma8.6).

Discovered by ``tests.browser_validation`` because this module names ``TEST_JSDOM_ROOT``. The page under test is the
assembled chatbot SPA; the request a click sends is the one ``tests/test_use_plan.py`` and the HTTP tests run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_plan_buttons_for_a_part_not_run():
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_plan_choices_dom.cjs")
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
