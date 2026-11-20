"""A recorded command output says whether the command is the framework's own (fix-2nsr).

``CommandOutput.framework`` is a semantic fact about the command, set from the command's metadata
(``CommandMetadata.framework``, the files of ``fastworkflow/_framework_commands``) when the output is
recorded on the turn. It is not a display instruction: any client may read it. ``None`` means "recorded
before the field existed", the same convention as ``ask_user_entry``.
"""
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.utils.signatures import InputForParamExtraction
from tests.test_chat_turn_outputs_contract import _make
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "fastworkflow" / "examples" / "library"


def _output(name: str) -> fastworkflow.CommandOutput:
    return fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response="x"), command_name=name)


def test_a_hand_built_output_does_not_claim_either_way():
    assert _output("find_book").framework is None
    assert _output("find_book").model_dump(mode="json")["framework"] is None


def test_a_record_written_before_the_field_existed_still_validates():
    legacy = fastworkflow.CommandOutput.model_validate(
        {"command_response": {"response": "x", "success": True}, "command_name": "review_subtasks"})
    assert legacy.framework is None


def test_the_metadata_flag_marks_exactly_the_framework_commands_folder(tmp_path):
    folder, _ctx, _wf = _make(tmp_path, LIBRARY, "library")
    flagged = {name for name, meta in CommandDirectory.load(str(folder)).map_command_2_metadata.items()
               if meta.framework}
    on_disk = {p.stem for p in (ROOT / "fastworkflow" / "_framework_commands").glob("*.py")
               if not p.stem.startswith("_")}
    assert flagged == on_disk


def test_recording_marks_the_framework_commands_and_only_those(tmp_path):
    _folder, ctx, _wf = _make(tmp_path, LIBRARY, "library")
    for name in ("review_proposal", "review_subtasks"):
        ctx.append_turn_output(_output(name))
    for name in ("find_book", "library_hours", "ask_user", "Shelf/review_subtasks", "wildcard"):
        ctx.append_turn_output(_output(name))
    recorded = {o.command_name: o.framework for o in ctx._turn_outputs}
    assert recorded == {
        "review_proposal": True, "review_subtasks": True,
        "find_book": False, "library_hours": False, "ask_user": False,
        "Shelf/review_subtasks": False, "wildcard": False,
    }


def test_an_output_already_decided_is_not_overwritten(tmp_path):
    _folder, ctx, _wf = _make(tmp_path, LIBRARY, "library")
    decided = _output("review_subtasks")
    decided.framework = False
    ctx.append_turn_output(decided)
    assert ctx._turn_outputs[-1].framework is False


def test_a_real_turn_carries_the_marker_in_its_turn_output(tmp_path, monkeypatch):
    # the same seam as the turn-output contract test: no model call extracts the parameters
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", lambda _self, cls, *a, **k: cls.model_construct())
    _folder, ctx, _wf = _make(tmp_path, LIBRARY, "library")
    out = ctx.process_turn("/library_hours")
    dumped = out.model_dump(mode="json")["command_outputs"]
    assert dumped and all(o["framework"] is False for o in dumped), dumped


@pytest.fixture(autouse=True)
def _framework(initialized_fastworkflow):  # noqa: F811
    yield
