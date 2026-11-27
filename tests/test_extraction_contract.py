"""The one result contract of every extractor, and the CME parameter stage kept on the shared loop (bd fix-o1lg S6, S7).

``parameter_clarification.Extraction`` is what both kinds of command produce for "the values of a command given some text":
the planning draft (``planning_commands.argument_gaps``, the typed call's checks) and the execution-command extractor
(``parameter_extraction.extraction_of``). Everything downstream reads only it: the ``parameter_request`` artifact, the
question, the progress rule. The CME's pending question is a ``ParameterLoop`` under one context key.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.intent import intent_loop
from fastworkflow.planning import planning_commands as pc
from fastworkflow.parameters import parameter_clarification as loops
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    extraction_of, model_of, park, round_ordinal)
from fastworkflow.parameters.signatures import INVALID_FLOAT_VALUE, NOT_FOUND
from fastworkflow.engine.workflow_execution_context import WorkflowExecutionContext

HELLO = str(Path(__file__).parent.joinpath("hello_world_workflow").resolve())


# ---- the structure ----------------------------------------------------------------------------------------------------

def test_the_states_of_a_parameter():
    extraction = loops.Extraction(values={"a": 1}, missing=("m",), invalid=(("i", "is not a number"),), earlier=("e",))
    assert extraction.names == ("m", "i")
    assert extraction.state_of("a") == (loops.STATE_GIVEN, "")
    assert extraction.state_of("m") == (loops.STATE_MISSING, "")
    assert extraction.state_of("i") == (loops.STATE_INVALID, "is not a number")
    assert extraction.state_of("e") == (loops.STATE_EARLIER, "")
    assert extraction.describe() == "missing required argument m; i is not a number"


def test_a_parameter_an_earlier_part_produces_is_no_longer_open():
    extraction = loops.Extraction(missing=("m", "e"), invalid=(("e", "is odd"), ("i", "is wrong")))
    moved = extraction.from_earlier({"e", "unrelated"})
    assert moved.names == ("m", "i") and moved.earlier == ("e",)
    assert moved.state_of("e") == (loops.STATE_EARLIER, "")
    assert extraction.from_earlier(()) is extraction


def test_the_request_is_one_shape_with_values_as_text_or_as_they_are():
    extraction = loops.Extraction(values={"n": 3, "flag": False}, missing=("m",), invalid=(("i", "bad"),))
    assert extraction.request("cmd") == {"command": "cmd", "missing": ["m"], "invalid": ["i"],
                                         "values": {"n": "3", "flag": "False"}}
    assert extraction.request("cmd", stringify=False)["values"] == {"n": 3, "flag": False}
    assert loops.Extraction(values={"n": 3}).request("cmd") is None          # nothing open: nothing to ask field by field
    assert loops.Extraction(reason="a rule of the command").request("cmd") is None


# ---- both producers return it, and the request has the same keys -------------------------------------------------------------

def test_the_planning_draft_and_the_execution_extractor_produce_the_same_structure():
    folder = str(Path(__file__).parent.joinpath("..", "fastworkflow", "examples", "library").resolve())
    command = pc.planning_command(folder, "Library/note_member") or pc.planning_command(folder, "note_member")
    if command is None:
        pytest.skip("the library example has no note_member planning command here")
    planned = pc.argument_gaps(command, {"member": None, "text": None}, "add a note")
    assert isinstance(planned, loops.Extraction) and planned.missing == ("member", "text")

    executed = extraction_of(SimpleNamespace(model_dump=lambda **_: {"a": 1, "b": NOT_FOUND}), {"missing_fields": ["b"]})
    assert isinstance(executed, loops.Extraction) and executed.missing == ("b",) and executed.values == {"a": 1}

    assert set(planned.request("note_member")) == set(executed.request("cmd")) == {"command", "missing", "invalid", "values"}


# ---- the CME stage on the shared loop -----------------------------------------------------------------------------------------

def test_an_execution_round_is_recorded_on_the_loop_and_progress_is_counted():
    loop = loops.ParameterLoop(command="Cmd/add", kind=loops.KIND_EXECUTION)
    first = loops.Extraction(values={"x": 5}, missing=("y", "z"))
    assert loops.record_round(loop, first, "which y and z?").kind == loops.ASK_AGAIN
    assert (loop.asks, loop.stalled, loop.open, loop.values) == (1, 0, ["y", "z"], {"x": 5})
    assert loop.question == "which y and z?" and loop.request["missing"] == ["y", "z"]

    assert loops.record_round(loop, first, "again?").kind == loops.ASK_AGAIN   # a reply that closed nothing: asked once more
    assert (loop.asks, loop.stalled) == (2, 1)
    loops.record_round(loop, loops.Extraction(values={"x": 5, "y": 1}, missing=("z",)), "z?")   # closed one
    assert (loop.asks, loop.stalled, loop.open) == (3, 0, ["z"])
    assert loops.record_round(loop, loops.Extraction(missing=("z",)), "z?").kind == loops.ASK_AGAIN   # one with none
    ended = loops.record_round(loop, loops.Extraction(missing=("z",)), "z?")   # step 6: the second in a row stops it
    assert ended.kind == loops.EXHAUSTED and ended.missing == ("z",) and ended.describe == "missing required argument z"
    assert loop.stalled == 2 and loops.ParameterLoop.from_state(json.loads(json.dumps(loop.to_state()))) == loop


def test_the_open_fields_come_back_as_sentinels_and_the_given_ones_as_values():
    from pydantic import BaseModel

    class Add(BaseModel):
        x: float
        y: float
        tags: list[str]

    loop = loops.ParameterLoop(command="Cmd/add", values={"x": 5.0}, open=["y", "tags"])
    params = model_of(loop, Add)
    assert (params.x, params.y, params.tags) == (5.0, NOT_FOUND, [])
    assert round_ordinal(loop) is None and round_ordinal(None) == 0       # no question was ever put on this one


def test_a_reply_through_a_real_session_completes_the_parked_command_and_ends_the_loop(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.core.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="o1lg-s7")
    ctx.bind_app_workflow(fastworkflow.Workflow.create(HELLO, workflow_id_str="o1lg-s7"))
    try:
        params_class = RoutingRegistry.get_definition(ctx.app_workflow.folderpath).get_command_class(
            "add_two_numbers", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
        cme = ctx._cme_workflow.context
        cme["command"] = "add 5 and something"
        cme["command_name"] = "add_two_numbers"
        park(ctx._cme_workflow, "add_two_numbers", params_class.model_construct(first_num=5.0, second_num=INVALID_FLOAT_VALUE))
        assert loops.pending(cme).open == ["second_num"]

        out = ctx.process_turn("7")

        assert "12" in str(out.model_dump())
        assert loops.pending(ctx._cme_workflow.context) is None
        assert intent_loop.waiting_for(ctx._cme_workflow.context) is None
    finally:
        ctx.close()
        RoutingRegistry.clear_registry()
