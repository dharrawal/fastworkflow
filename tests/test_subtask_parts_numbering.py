"""The parts of a split turn agree with the numbers the outputs and the headings carry (real library copy, real agent
with a scripted model, as tests/test_command_output_sub_task.py)."""
from __future__ import annotations

import re

import pytest

from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import TURN_PARTS_ARTIFACT
from tests.test_command_output_sub_task import (  # noqa: F401 - the module's fixtures and helpers
    COUNT, HOURS, THREE, _finish, _ran, _split_into, annotated_library, extraction_without_a_model, gated,
    no_draft_extraction, no_live_model, no_model_call_for_parameters, todo)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "recorded_review_router")


def test_the_parts_carry_the_numbers_of_the_outputs_and_the_headings(gated, monkeypatch):
    _split_into(monkeypatch, THREE)
    gated.script.steps = [HOURS, *_finish(), COUNT, *_finish(), HOURS, *_finish()]
    out = gated.ctx.process_turn("do all three", unattended=True)
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    parts = out.command_outputs[-1].command_response.artifacts[TURN_PARTS_ARTIFACT]
    headings = {int(n): title for n, title in re.findall(r"^### (\d+)\. (.*)$", out.answer, re.M)}
    assert [p["number"] for p in parts] == sorted(headings) == sorted({o.sub_task for o in _ran(out)})
    assert [p["text"] for p in parts] == THREE
    for part in parts:
        assert part["answered"] and part["title"] == headings[part["number"]]
        assert out.answer[part["start"]:part["end"]] == "the answer"
    assert out.answer == "\n\n".join(f"### {p['number']}. {p['title']}\nthe answer" for p in parts)
