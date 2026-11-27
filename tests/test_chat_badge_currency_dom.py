"""The chat results view reads ``ui.tones`` on a badge and ``ui.currency`` on a currency value (fix-weih.29, fix-weih.30).

Discovered by ``tests.browser_validation`` because this module names ``TEST_JSDOM_ROOT``. The page under test
is the assembled chatbot SPA.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_author_tones_and_currency_codes_reach_the_drawn_values():
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_badge_currency_dom.cjs")
    src = Path(__file__).resolve().parents[1] / "fastworkflow" / "run_chatbot" / "static" / "src"
    result = subprocess.run(["node", str(script), dependency, str(src)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
