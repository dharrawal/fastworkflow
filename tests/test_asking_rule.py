"""The ONE asking rule for a missing or refused value (bd fix-fpyb.8, migration step 6, ``docs/pending_state_design.md`` 3.6).

Keep asking while each reply makes progress (fills or fixes a value the question listed); after a reply with none ask once
more; after a second reply in a row with none stop; cancel always ends it. The same rule for a planning command and an
execution command, with or without an agent model, in a human or an agent session. Only what "stop" hands over to depends
on the session (``tests/test_parameter_loop.py`` and ``tests/test_no_agent_review.py`` pin the planning side).

These tests drive the deterministic path of the real CME on the hello-world workflow, as
``tests/test_extraction_contract.py`` does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import no_agent_mode
from fastworkflow.parameters import parameter_clarification as loops
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction, park
from fastworkflow.parameters.signatures import INVALID_FLOAT_VALUE, NOT_FOUND
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

HELLO = str(Path(__file__).parent.joinpath("hello_world_workflow").resolve())
COMMAND = "add_two_numbers"


def test_progress_has_one_definition_a_listed_value_is_no_longer_open():
    assert loops.made_progress(["a", "b"], ["b"]) is True            # filled one
    assert loops.made_progress(["a"], []) is True
    assert loops.made_progress(["a", "b"], ["a", "b"]) is False      # nothing closed
    assert loops.made_progress(["a"], ["a", "c"]) is False           # a new open value is not progress
    assert loops.made_progress([], ["a"]) is False


def test_the_loop_has_no_policy_and_no_pending_flag():
    state = loops.ParameterLoop(command="c").to_state()
    assert "policy" not in state and "pending" not in state and state["stalled"] == 0
    assert not [name for name in dir(loops) if name.startswith("POLICY_")]


@pytest.fixture
def session(tmp_path, no_planner_key):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="asking-rule")
    ctx.bind_app_workflow(fastworkflow.Workflow.create(HELLO, workflow_id_str="asking-rule"))
    try:
        yield ctx
    finally:
        ctx.close()
        RoutingRegistry.clear_registry()


def _parked(ctx, **fields):
    from fastworkflow.command_routing import RoutingRegistry

    params_class = RoutingRegistry.get_definition(ctx.app_workflow.folderpath).get_command_class(
        COMMAND, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    cme = ctx._cme_workflow.context
    cme["command"], cme["command_name"] = "add some numbers", COMMAND
    park(ctx._cme_workflow, COMMAND, params_class.model_construct(**fields))
    return cme


def _loop(ctx):
    return loops.pending(ctx._cme_workflow.context)


# ---- (a) the human deterministic session ---------------------------------------------------------------------------------

def test_a_value_question_no_longer_repeats_forever_two_replies_in_a_row_with_no_progress_end_the_command(session):
    _parked(session, first_num=5.0, second_num=INVALID_FLOAT_VALUE)

    first = session.process_turn("not a number")
    assert first.success is False and _loop(session).stalled == 1                      # asked once more
    assert _loop(session).open == ["second_num"]

    second = session.process_turn("still not one")

    assert _loop(session) is None                                                     # no question stays pending
    assert second.success is False
    assert COMMAND in second.answer and "second_num" in second.answer and "was not run" in second.answer
    assert no_agent_mode.MAX_NO_PROGRESS == 2


def test_a_reply_that_makes_progress_resets_the_count(session):
    _parked(session, first_num=INVALID_FLOAT_VALUE, second_num=INVALID_FLOAT_VALUE)

    session.process_turn("umm")                                                       # nothing readable: 1 in a row
    assert _loop(session).stalled == 1
    session.process_turn("first_num: 3")                                              # fixes a listed value: reset
    assert _loop(session).stalled == 0 and _loop(session).open == ["second_num"]
    session.process_turn("seven")                                                     # no progress: 1 in a row again
    assert _loop(session) is not None and _loop(session).stalled == 1                 # (without the reset: stopped)

    done = session.process_turn("4")

    assert _loop(session) is None and "7" in done.answer and "not run" not in done.answer


def test_cancel_still_ends_the_question_at_once(session):
    _parked(session, first_num=5.0, second_num=INVALID_FLOAT_VALUE)

    out = session.process_turn("abort")

    assert _loop(session) is None


# ---- (d) an agent session, the same rule on the same path ----------------------------------------------------------------

def test_an_agent_session_stops_after_two_calls_in_a_row_with_no_progress(session):
    cme = _parked(session, first_num=5.0, second_num=INVALID_FLOAT_VALUE)
    session.app_workflow.context["run_as_agent"] = True
    ask = lambda text: ParameterExtraction(session._cme_workflow, session.app_workflow, COMMAND, f"{COMMAND}: {text}").extract()

    first = ask("not a number")
    assert first.parameters_are_valid is False and first.parameter_request is None      # the agent reads the text
    assert _loop(session).stalled == 1

    second = ask("still not")

    assert second.parameters_are_valid is False and "was not run" in second.error_msg and "second_num" in second.error_msg
    assert _loop(session) is None and loops.pending(cme) is None


def test_an_agent_session_call_that_fixes_a_value_is_not_stopped(session):
    _parked(session, first_num=INVALID_FLOAT_VALUE, second_num=INVALID_FLOAT_VALUE)
    session.app_workflow.context["run_as_agent"] = True
    ask = lambda text: ParameterExtraction(session._cme_workflow, session.app_workflow, COMMAND, f"{COMMAND}: {text}").extract()

    ask("umm")
    ask("first_num: 3")
    third = ask("seven")

    assert third.parameters_are_valid is False and _loop(session) is not None and _loop(session).stalled == 1
