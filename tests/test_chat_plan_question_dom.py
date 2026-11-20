"""The chat SPA's question "which plan should run this part?" (bd fix-yu3l.5).

Discovered by ``tests.browser_validation`` because this module names ``TEST_JSDOM_ROOT``. The page under test is the
assembled chatbot SPA; the question it shows and the replies a click sends are the ones
``tests/test_plan_choice_question.py`` has the core ask and read.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_plan_buttons_reply_to_the_plan_choice_question():
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_plan_question_dom.cjs")
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
