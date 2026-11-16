"""The server's definition of a page, pinned by one shared fixture (fix-weih.18).

``tests/fixtures/page_shape_cases.json`` is read here against the real
``observation_offloading.facts.output_page`` and by ``tests/chatbot_results_dom.cjs`` against the client's
``tmIsPageOutput``, so the two definitions cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastworkflow.observation_offloading.facts import output_page

CASES = json.loads((Path(__file__).parent / "fixtures" / "page_shape_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_output_page_agrees_with_the_shared_case(case):
    assert (output_page(case["output"]) is not None) is case["is_page"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_composed_card_uses_the_same_definition_plus_two_scalar_columns(case):
    """``table_worthy`` is the card's rule: a page that also has two or more non-empty scalar lists."""
    from fastworkflow.run_chatbot.composer import _is_page_shape

    assert _is_page_shape(case["output"]) is case["table_worthy"]
    assert not case["table_worthy"] or case["is_page"]


def test_the_pages_legacy_fallback_names_exactly_the_frameworks_own_review_commands():
    """The results view leaves out an output the server marks ``framework: true``. For a record WITHOUT that key
    (a turn stored before the marker existed) it falls back to ``TM_LEGACY_FRAMEWORK_COMMANDS`` in
    ``254-chat-results.js``, which must be the files of ``fastworkflow/_framework_commands``. Delete the list and
    this test together when no stored turn predates the key."""
    import re

    root = Path(__file__).resolve().parents[1] / "fastworkflow"
    source = (root / "run_chatbot" / "static" / "src" / "254-chat-results.js").read_text(encoding="utf-8")
    assert "TM_FRAMEWORK_COMMANDS" not in source, "the page's own list was renamed the legacy fallback"
    listed = re.search(r"var TM_LEGACY_FRAMEWORK_COMMANDS = \[(.*?)\];", source, re.DOTALL)
    assert listed, "TM_LEGACY_FRAMEWORK_COMMANDS is gone"
    names = sorted(re.findall(r'"([^"]+)"', listed.group(1)))
    on_disk = sorted(p.stem for p in (root / "_framework_commands").glob("*.py") if not p.stem.startswith("_"))
    assert names == on_disk


def test_the_schema_listing_the_page_receives_omits_the_frameworks_own_commands():
    """What the page relies on: ``command_schema_listing`` has no framework command, for any workflow."""
    from fastworkflow.mcp_server import command_schema_listing

    root = Path(__file__).resolve().parents[1]
    framework = {p.stem for p in (root / "fastworkflow" / "_framework_commands").glob("*.py")
                 if not p.stem.startswith("_")}
    listing = command_schema_listing(str(root / "tests" / "example_workflow"))
    assert listing, "the example workflow lists commands"
    assert not {name.split("/")[-1] for name in listing} & framework
