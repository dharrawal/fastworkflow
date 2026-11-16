"""No agent model: a refused planning command call is asked about once, for every argument it lacks, and keeps being asked
while each reply makes progress (fix-1ma8.7).

On the library example's ``note_member`` planning command, which has two required arguments (the member and the note's
text). The pinned route (``use_plan``) and the matched route both go through the draft, the review and the bound run
(``_run_planning_means``); an unattended turn cannot be asked. The shared extractor's model is scripted to find a value
only when the request text writes it, as a real one does; offline like tests/test_use_plan.py."""
from __future__ import annotations

import socket

import pytest

from fastworkflow import approval_binding, no_agent_mode, review_proposals
from fastworkflow.approval_binding import Surface
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import load_planning_command_map
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT, WorkflowExecutionContext
from fastworkflow.workflow_agent import build_query_with_next_steps as REAL_BUILD
from tests.scripted_extractor import ScriptedExtractor
from tests.test_approval_binding import annotated_library, todo  # noqa: F401
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

PLAN = "Library/note_member"
HOST = Surface(kind="host", verified=True, uid="user-1")
REQUEST = "put a note on a record"
MEMBER, NOTE = "Zed Quill", "rehung the banner"


class _FindsWhatTheTextWrites(ScriptedExtractor):
    """The extractor's model: a value is found when the text it reads writes it, else it is NOT_FOUND."""

    def forward(self, prompt=None, messages=None, **kwargs):
        text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
        self.answers = iter([{"reasoning": "scripted", "member": MEMBER if MEMBER in text else "NOT_FOUND",
                              "text": NOTE if NOTE in text else "NOT_FOUND"}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import litellm

    def refuse(*args, **kwargs):
        raise AssertionError("a model call or a network connection was attempted")

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setattr(litellm, "acompletion", refuse)
    real = socket.socket.connect

    def connect(self, address):
        if self.family == socket.AF_UNIX or (isinstance(address, tuple) and str(address[0]) in ("127.0.0.1", "::1", "localhost")):
            return real(self, address)
        refuse()

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture(autouse=True)
def _setup(gated, no_agent_model, no_planner_key, monkeypatch):
    from fastworkflow.utils import signatures

    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "on")
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context",
                        lambda self: pytest.fail("the agent LM was built in no-agent mode"))
    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps",
                        lambda user_query, session, **kwargs: setattr(session, "_initial_plan_text", "") or user_query)
    lm = _FindsWhatTheTextWrites([])
    original = signatures.dspy_utils.get_lm
    monkeypatch.setattr(signatures.dspy_utils, "get_lm",
                        lambda name, *a, **k: lm if name == "LLM_PARAM_EXTRACTION" else original(name, *a, **k))
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: [user_query])


def _reply(gated, message, **kwargs):
    if getattr(gated, "token", None) is None:
        delivery = approval_binding.take_delivery(gated.wf)
        gated.token = delivery["token"] if delivery else None
    return gated.ctx.process_turn(message, approval_surface=HOST, approval_token=gated.token, **kwargs)


def _request(out) -> dict:
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "planning_parameter", artifacts
    return artifacts["parameter_request"]


def _pinned_and_accepted(gated, text=REQUEST):
    """Pinned to the plan: its parameter question (no review opens), which names every missing argument."""
    first = gated.ctx.process_turn(text, use_plan=PLAN, approval_surface=HOST)
    assert first.status is TurnStatus.AWAITING_USER
    return first, first


def _ran_with(gated) -> dict:
    return gated.ctx._turn_subtask_bindings[-1]["ran"]


def _stop(out):
    (stop,) = out.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]
    return stop


def test_one_question_names_every_missing_argument_and_the_data_lists_them_all(gated):
    first, asked = _pinned_and_accepted(gated)

    assert "- member (required): not given" in first.answer and "- text (required): not given" in first.answer
    assert asked.status is TurnStatus.AWAITING_USER
    assert asked.answer.startswith("You chose the planning command Library/note_member: Add a note to a member's record.")
    assert "name: value" in asked.answer and "cancel stops" in asked.answer
    request = _request(asked)
    assert request["command"] == PLAN and request["missing"] == ["member", "text"] and request["values"] == {}


def test_a_reply_that_gives_both_runs_the_part_with_both_values(gated):
    _pinned_and_accepted(gated)
    done = _reply(gated, f"member: {MEMBER}\ntext: {NOTE}")

    assert done.status is TurnStatus.COMPLETED
    assert _ran_with(gated) == {"member": MEMBER, "text": NOTE}
    # The command ran its steps (it stops later at a step only an agent can write, which is the plan's own business).
    assert all(stop["open_step"] is not None for stop in done.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT, []))


def test_a_reply_that_gives_one_leaves_a_second_question_for_the_other_and_keeps_the_first_value(gated):
    _pinned_and_accepted(gated)
    second = _reply(gated, f"member: {MEMBER}")

    assert second.status is TurnStatus.AWAITING_USER
    request = _request(second)
    assert request["missing"] == ["text"] and request["values"] == {"member": MEMBER}
    assert f"- member (required): {MEMBER}" in second.answer and "- text (required): not given" in second.answer
    done = _reply(gated, f"text: {NOTE}")
    assert done.status is TurnStatus.COMPLETED
    assert _ran_with(gated) == {"member": MEMBER, "text": NOTE}


def test_the_draft_keeps_the_value_the_request_gave_and_the_question_asks_only_for_the_other(gated):
    first = gated.ctx.process_turn(f"{REQUEST} for {MEMBER}", use_plan=PLAN, approval_surface=HOST)

    request = _request(first)
    assert request["missing"] == ["text"] and request["values"] == {"member": MEMBER}
    assert [(p["name"], p["value"], p["state"]) for p in request["parameters"]] == [
        ("member", MEMBER, None), ("text", None, "missing")]
    assert f"- member (required): {MEMBER}" in first.answer and "- text (required): not given" in first.answer
    done = _reply(gated, NOTE)                       # one value is open, so a bare reply is that value
    assert done.status is TurnStatus.COMPLETED and _ran_with(gated) == {"member": MEMBER, "text": NOTE}


def test_a_reply_that_makes_no_progress_is_asked_once_more_then_the_part_is_not_run_naming_what_is_missing(gated):
    _pinned_and_accepted(gated)
    again = _reply(gated, "hmm, I am not sure")

    assert again.status is TurnStatus.AWAITING_USER and again.answer.startswith("That did not give a value I could use.")
    assert _request(again)["missing"] == ["member", "text"]
    stopped = _reply(gated, "still not sure")
    assert stopped.status is TurnStatus.COMPLETED
    stop = _stop(stopped)
    assert stop["command"] == PLAN and "missing required arguments member, text" in stop["reason"]


def test_progress_resets_the_no_progress_count(gated):
    _pinned_and_accepted(gated)
    _reply(gated, "hmm")                                      # unclear: asked once more
    assert gated.ctx._refusal_progress["stalled"] == 1
    progressed = _reply(gated, f"member: {MEMBER}")           # progress: the count starts again
    assert progressed.status is TurnStatus.AWAITING_USER and _request(progressed)["missing"] == ["text"]
    assert gated.ctx._refusal_progress["stalled"] == 0


@pytest.mark.parametrize("word", ["cancel", "abort"])
def test_cancel_declines_the_part_and_says_so(gated, word):
    _pinned_and_accepted(gated)
    out = _reply(gated, word)

    assert out.status is TurnStatus.COMPLETED
    stop = _stop(out)
    assert stop["command"] == PLAN and "declined" in stop["reason"] and "member, text" in stop["reason"]


def test_an_unattended_turn_cannot_be_asked_and_the_reason_names_the_missing_arguments(gated):
    out = gated.ctx.process_turn(REQUEST, use_plan=PLAN, approval_surface=HOST, unattended=True)

    assert out.status is TurnStatus.COMPLETED
    stop = _stop(out)
    assert stop["command"] == PLAN and "missing required arguments member, text" in stop["reason"]


def test_the_reason_never_names_an_argument_the_user_gave(gated):
    out = gated.ctx.process_turn(f"{REQUEST} for {MEMBER}", use_plan=PLAN, approval_surface=HOST, unattended=True)

    reason = _stop(out)["reason"]
    assert "missing required argument text" in reason and "member" not in reason.split("(")[1].split(")")[0]


def test_the_progress_survives_a_save_between_the_question_and_the_reply(gated):
    _pinned_and_accepted(gated)
    _reply(gated, f"member: {MEMBER}")
    state = gated.ctx.serialize_state(channel_id="c")

    assert state["refusal_progress"]["command"] == PLAN and state["refusal_progress"]["open"] == ["text"]
    assert state["use_plan"] == PLAN
    assert state["refusal_progress"]["asks"] == 2


def test_a_matched_part_is_asked_the_same_way(gated, monkeypatch):
    command = load_planning_command_map(gated.wf.folderpath)[PLAN]
    monkeypatch.setattr(review_proposals, "planning_match",
                        lambda workflow, text, exclude=None: ("", command, {PLAN: 0.9}))
    first = gated.ctx.process_turn(REQUEST, approval_surface=HOST)
    asked = _reply(gated, "yes")

    assert asked.status is TurnStatus.AWAITING_USER and _request(asked)["missing"] == ["member", "text"]
    second = _reply(gated, f"member: {MEMBER}")
    assert _request(second)["missing"] == ["text"] and _request(second)["values"] == {"member": MEMBER}
    done = _reply(gated, f"text: {NOTE}")
    assert done.status is TurnStatus.COMPLETED and _ran_with(gated) == {"member": MEMBER, "text": NOTE}


def test_the_gaps_of_a_call_are_worked_out_without_a_model(gated):
    from fastworkflow import planning_commands

    command = load_planning_command_map(gated.wf.folderpath)[PLAN]
    gaps = planning_commands.argument_gaps(command, {"member": None, "text": None}, REQUEST)
    assert gaps.missing == ("member", "text") and gaps.invalid == () and gaps.values == {}
    gaps = planning_commands.argument_gaps(command, {"member": MEMBER, "text": None}, f"{REQUEST} for {MEMBER}")
    assert gaps.missing == ("text",) and gaps.values == {"member": MEMBER}
    gaps = planning_commands.argument_gaps(command, {"member": "Nobody Atall", "text": NOTE}, f"{REQUEST}: {NOTE}")
    assert gaps.missing == () and gaps.invalid and gaps.invalid[0][0] == "member"
    assert gaps.values == {"text": NOTE}


def test_the_typed_call_path_asks_about_every_missing_argument_too(gated, monkeypatch):
    """A part named exactly takes the fast path: the planner's typed call is refused, and the question comes from it."""
    import dspy
    from dspy.utils import DummyLM

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", REAL_BUILD)
    command = load_planning_command_map(gated.wf.folderpath)[PLAN]
    monkeypatch.setattr(review_proposals, "planning_match",
                        lambda workflow, text, exclude=None: ("", command, {PLAN: 0.9}))
    class Planner(DummyLM):
        """The planner's model: the typed fill finds nothing, and the text planner (read after a refusal that is not asked
        about) gives a one-line plan."""

        def forward(self, prompt=None, messages=None, **kwargs):
            text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
            self.answers = iter([{"reasoning": "scripted", "next_steps": "1. find the member"} if "next_steps" in text
                                 else {"reasoning": "nothing in the request", "member": "None", "text": "None"}])
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    gated.ctx._current_planner_lm = Planner([], adapter=dspy.JSONAdapter())
    first = gated.ctx.process_turn("note_member", approval_surface=HOST)

    assert first.status is TurnStatus.AWAITING_USER and gated.ctx._turn_subtask_review == "fast_path"
    request = _request(first)
    assert request["missing"] == ["member", "text"] and "does not give the member and text" in first.answer
    second = gated.ctx.process_turn(f"member: {MEMBER}", approval_surface=HOST)
    assert second.status is TurnStatus.AWAITING_USER and _request(second)["missing"] == ["text"]
    assert _request(second)["values"] == {"member": MEMBER}
    again = gated.ctx.process_turn("hmm", approval_surface=HOST)
    assert again.status is TurnStatus.AWAITING_USER and again.answer.startswith("That did not give a value I could use.")
    stopped = gated.ctx.process_turn("hmm", approval_surface=HOST)
    stop = _stop(stopped)
    assert stopped.status is TurnStatus.COMPLETED
    assert stop["command"] == PLAN and "missing required argument text" in stop["reason"]
    assert "member" not in stop["reason"].split("(")[1].split(")")[0]
