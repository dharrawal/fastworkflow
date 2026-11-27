"""The chat SPA's review form (fix-weih.13).

Discovered by ``tests.browser_validation`` because this module names
``TEST_JSDOM_ROOT``. The page under test is the assembled chatbot SPA; the reply
contract it shares with ``test_review_reply_grammar.py`` is
``tests/fixtures/review_form_edits.json``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_review_form():
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_review_form_dom.cjs")
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
