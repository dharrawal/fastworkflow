"""Pending prompts are described on every turn that waits for the human (fix-weih.7).

``WorkflowExecutionContext._build_turn_result`` tags the open prompt's entry (its ``artifacts``) with
``question_kind`` ("agent" | "planning_parameter" | "approval" | "review" | "unknown") and the matching
payload (``approval``, ``review`` or ``parameter_request``), including a user-typed "/" approval whose
turn status is COMPLETED. Real WorkflowExecutionContext, real approval and review stores, the real
agent with scripted steps (the ``todo`` fixture of the approval tests). No test calls a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

import fastworkflow
from fastworkflow.planning import planning_commands as pc
from fastworkflow.engine import prompt_payloads
from fastworkflow import workflow_agent
from fastworkflow.review import review_engine, review_proposals
from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _added, _ask, _finish, annotated_library, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

#: The real query builder, taken before the fixtures stub it.
REAL_BUILD_QUERY = workflow_agent.build_query_with_next_steps
PAYLOAD_KEYS = ("approval", "review", "parameter_request")
LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def _last_artifacts(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def _payload_keys(artifacts: dict) -> list[str]:
    return [key for key in PAYLOAD_KEYS if key in artifacts]


# ----------------------------------------------------------------------
# 1. An open-ended agent ask_user
# ----------------------------------------------------------------------


def test_an_open_ended_agent_question_is_tagged_agent_with_no_payload(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    out = todo.ctx.process_turn("find a book")
    assert out.status is TurnStatus.AWAITING_USER
    entry = out.command_outputs[-1]
    assert entry.is_ask_user and entry.command_response.success is False
    assert entry.command_response.artifacts["question_kind"] == "agent"
    assert _payload_keys(entry.command_response.artifacts) == []


# ----------------------------------------------------------------------
# 2. An agent-originated approval
# ----------------------------------------------------------------------


def test_an_agent_originated_approval_is_tagged_approval_with_its_payload(todo, annotated_library, grant_root):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    artifacts = _last_artifacts(out)
    assert artifacts["question_kind"] == "approval"
    assert artifacts["approval"] == {"command": "Record/add_note", "parameters": {"text": "Ann"},
                                     "origin": "agent", "can_always": False}   # fix-qjne.11
    # bd fix-ftdl.7.6: no "irreversible" key: the framework does not know whether a change can be undone.
    assert "review" not in artifacts and "parameter_request" not in artifacts

    # The answered prompt's tag stays on its own entry and does not leak into the turn's final answer.
    done = todo.ctx.process_turn("yes")
    assert done.status is TurnStatus.COMPLETED
    final = done.command_outputs[-1].command_response.artifacts
    assert not any(key in final for key in ("question_kind", *PAYLOAD_KEYS))
    entry = next(o for o in done.command_outputs if o.is_ask_user)
    assert entry.command_response.artifacts["question_kind"] == "approval"


# ----------------------------------------------------------------------
# 3. A user-typed "/" approval: the turn is COMPLETED, the output says it waits
# ----------------------------------------------------------------------


def test_a_user_typed_slash_approval_is_tagged_although_the_turn_is_completed(todo, annotated_library, grant_root):
    out = todo.ctx.process_turn("/add_note text=Ann")
    assert out.status is TurnStatus.COMPLETED
    artifacts = _last_artifacts(out)
    assert artifacts["awaiting_user"] is True
    assert artifacts["question_kind"] == "approval"
    assert artifacts["approval"]["origin"] == "user"
    assert "irreversible" not in artifacts["approval"]            # bd fix-ftdl.7.6
    assert artifacts["approval"]["command"] == "Record/add_note"
    assert artifacts["approval"]["parameters"] == {"text": "Ann"}
    assert _added(todo.chores, todo.base) == []

    done = todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert done.status is TurnStatus.COMPLETED
    # The reply's own turn describes no prompt: no stale tag on its outputs.
    assert not any("question_kind" in o.command_response.artifacts for o in done.command_outputs)


# ----------------------------------------------------------------------
# 4. A refused planning command input
# ----------------------------------------------------------------------


def _command(name: str = "Library/book_by_title") -> pc.PlanningCommandCommand:
    return workflow_agent.load_planning_command_map(str(LIBRARY))[name]


def _gap_refusal(state: str, code: str, why: str, value) -> pc.PlanningCommandInputError:
    """The refusal of a guard for the argument ``title``, its argument carried as data (``gap``)."""
    from fastworkflow.parameters import check

    text = "missing required argument title" if code == "missing" else f"title={value!r} {why}"
    return pc.PlanningCommandInputError(text, gap=check.Gap("title", state, code, why, value, text))


@pytest.mark.parametrize("refusal_factory, missing, invalid", [
    (lambda: pc.PlanningCommandInputError("no such member",
                                          fields=("title", "title")), [], ["title"]),
    (lambda: _gap_refusal("missing", "missing", "", None), ["title"], []),
    (lambda: _gap_refusal("missing", "refused", pc.PRONOUN_WHY, "everyone"), ["title"], []),
    (lambda: _gap_refusal("missing", "refused", pc.PLACEHOLDER_WHY, "x"), ["title"], []),
    (lambda: _gap_refusal("invalid", "refused", "is not one of a, b", "bad"), [], ["title"]),
], ids=["field", "missing", "pronoun", "placeholder", "not-one-of"])
def test_the_parameter_request_of_each_refusal_shape(refusal_factory, missing, invalid):
    command = _command()
    assert pc.planning_refusal_parameter_request(command, refusal_factory()) == {
        "command": command.name, "missing": missing, "invalid": invalid, "values": {}}


def test_a_refusal_that_names_no_input_gives_no_parameter_request():
    command = _command()
    assert pc.planning_refusal_parameter_request(
        command, pc.PlanningCommandInputError("the arguments break a rule of book_by_title: nope")) is None
    # A guard's refusal of a name that is not one of the command's inputs is not a form either.
    from fastworkflow.parameters import check

    assert pc.planning_refusal_parameter_request(
        command, pc.PlanningCommandInputError(
            "nobody='x' is not valid", gap=check.Gap("nobody", check.INVALID, "refused", "is not valid", "x",
                                                    "nobody='x' is not valid"))) is None


def test_a_refused_planning_call_suspends_tagged_planning_parameter_and_the_reply_clears_it(todo, monkeypatch):
    command = _command()
    # The planner's typed call is the model seam: it is replaced by the refusal the real guards give an empty value.
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {p.name: None for p in command.params}, "who wrote that book")
    refusal = refused.value
    expected = pc.planning_refusal_parameter_request(command, refusal)
    assert expected is not None and expected["missing"] == ["title"]

    def refuse(*args, **kwargs):
        raise refusal

    script_matcher(monkeypatch, lambda text: {command.name: 0.9, "none": 0.0})
    monkeypatch.setattr(workflow_agent, "typed_planning_call", refuse)

    # The approval fixtures stub the planner's whole query builder; this test needs the real one, whose typed
    # call (the model seam, replaced above) is what raises the refusal. Its planner LM is never called.
    unused_lm = dspy.LM("openai/unused-planner", api_base="http://127.0.0.1:9/v1", api_key="unused", num_retries=0)
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda *args, **kwargs: REAL_BUILD_QUERY(*args, **{**kwargs, "planner_lm": unused_lm}))
    out = todo.ctx.process_turn("who wrote that book")
    assert out.status is TurnStatus.AWAITING_USER
    assert todo.ctx._resolution_pending is True
    artifacts = _last_artifacts(out)
    assert artifacts["question_kind"] == "planning_parameter"
    assert artifacts["parameter_request"] == expected
    assert "approval" not in artifacts and "review" not in artifacts
    assert todo.ctx._waiting.payload["loop"]["request"] == expected

    # The reply resumes the part that asked (its own call, not the turn from the top); the question is no longer
    # pending when it runs.
    seen = {}

    def call_agent(query, planner_user_query, **kwargs):
        seen["pending"] = todo.ctx._resolution_pending
        seen["planning_name"] = kwargs.get("planning_name")
        return dspy.Prediction(final_answer="done", exhausted=False)

    monkeypatch.setattr(todo.ctx, "_call_agent_for_query", call_agent)
    monkeypatch.setattr(todo.ctx, "_run_agent", lambda message: pytest.fail("the turn was run again from the top"))
    todo.ctx._resume_plan_resolution("Dune")
    assert seen == {"pending": False, "planning_name": command.name}
    assert todo.ctx._resolution_pending is False


# ----------------------------------------------------------------------
# 5. A sub-task review, and the refresh when an edit revises it in place
# ----------------------------------------------------------------------

EMMA = "who wrote Emma and how many copies do we have"
JOKE = "tell Ann a joke about cats"


@pytest.fixture
def gated(todo):
    vars(todo.ctx).pop("_subtask_gate", None)
    todo.wf.record = todo.library
    return todo


def test_a_subtask_review_is_tagged_review_and_an_edit_refreshes_it(gated, monkeypatch):
    script_matcher(monkeypatch, {EMMA: {"none": 0.9}, JOKE: {"none": 0.9}})
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda user_query, **kwargs: [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert out.status is TurnStatus.AWAITING_USER
    artifacts = _last_artifacts(out)
    assert artifacts["question_kind"] == "review"
    review = artifacts["review"]
    assert review["kind"] == "sub_tasks" and review["revision"] == 0
    assert [item["goal"] for item in review["items"]] == [EMMA, JOKE]
    assert len(review["changes_data"]) == 2 and all(isinstance(flag, bool) for flag in review["changes_data"])
    assert "approval" not in artifacts and "parameter_request" not in artifacts
    json.dumps(artifacts)

    revised = gated.ctx.process_turn("change 1: goal=Updated goal")
    assert revised.status is TurnStatus.AWAITING_USER
    refreshed = _last_artifacts(revised)
    assert refreshed["question_kind"] == "review"
    assert refreshed["review"]["revision"] == 1
    assert refreshed["review"]["items"][0]["goal"] == "Updated goal"


# ----------------------------------------------------------------------
# 6. A failed lookup never raises into the turn
# ----------------------------------------------------------------------


def test_a_lookup_failure_tags_unknown_with_no_payload(todo, annotated_library, grant_root, monkeypatch):
    def boom(ctx, held):
        raise RuntimeError("store unavailable")

    # The description's lookup (the record's payload builder; the agent's own question no longer reads the approval store).
    monkeypatch.setattr(prompt_payloads, "payload_of", boom)
    todo.script.steps = [("ask_user", {"clarification_request": "Which book?"})]
    out = todo.ctx.process_turn("find a book")
    assert out.status is TurnStatus.AWAITING_USER
    artifacts = _last_artifacts(out)
    assert artifacts["question_kind"] == "unknown"
    assert _payload_keys(artifacts) == []


# ----------------------------------------------------------------------
# 7. The answered entry keeps its tag
# ----------------------------------------------------------------------


def test_the_answered_entry_keeps_its_question_kind(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    out = todo.ctx.process_turn("find a book")
    entry = out.command_outputs[-1]
    assert entry.command_response.artifacts["question_kind"] == "agent"

    todo.ctx.complete_ask_user_entry("Emma")
    assert entry.command_response.success is True and entry.command_response.response == "Emma"
    assert entry.command_response.artifacts["question_kind"] == "agent"


# ----------------------------------------------------------------------
# Follow-ups: the final answer keeps other artifacts, can_always False, two suspensions, a declined "/" approval
# ----------------------------------------------------------------------


def test_only_the_prompt_description_keys_are_withheld_from_the_final_answer(todo):
    todo.script.steps = [("ask_user", {"clarification_request": "Which book do you mean?"})]
    out = todo.ctx.process_turn("find a book")
    entry = out.command_outputs[-1]
    entry.command_response.artifacts["custom"] = "kept"          # any other artifact of the entry
    # The turn's final answer, built by the finalize chokepoint from the turn's outputs.
    final_output = todo.ctx._finalize_agent_output("find a book", dspy.Prediction(final_answer="ok", exhausted=False))
    final = final_output.command_response.artifacts
    assert final.get("custom") == "kept"
    assert not any(key in final for key in ("question_kind", *PAYLOAD_KEYS))
    assert entry.command_response.artifacts["question_kind"] == "agent"


def test_can_always_is_false_when_the_surface_names_no_grant_identity(todo, annotated_library, grant_root):
    from fastworkflow.review.approval_binding import Surface

    host = Surface("host", verified=True, uid="alice")
    out = _ask(todo, "show_notes", steps=_finish(), approval_surface=host)   # delegatable: fix-qjne.11
    entry = out.command_outputs[-1]
    with_identity = dict(entry.command_response.artifacts["approval"])
    assert with_identity["can_always"] is True
    # The surface can no longer name a user (an unverified host): described again, nothing else changes.
    todo.store.surface = Surface("host", verified=False, uid=None)
    todo.ctx._describe_pending_prompt()
    described = entry.command_response.artifacts["approval"]
    assert described["can_always"] is False
    assert {k: v for k, v in described.items() if k != "can_always"} == \
        {k: v for k, v in with_identity.items() if k != "can_always"}


def test_two_suspensions_in_one_turn_each_carry_their_own_description(todo, annotated_library, grant_root):
    todo.script.steps = [("ask_user", {"clarification_request": "What note?"}),
                         ("execute_workflow_query", {"command": "add_note <text>Ann</text>"}), *_finish()]
    first = todo.ctx.process_turn("add a note")
    assert first.status is TurnStatus.AWAITING_USER
    first_entry = first.command_outputs[-1]
    assert first_entry.command_response.artifacts["question_kind"] == "agent"

    second = todo.ctx.process_turn("Ann")                       # answers the question; the approval then suspends
    assert second.status is TurnStatus.AWAITING_USER
    asks = [o for o in second.command_outputs if o.is_ask_user]
    assert len(asks) == 2
    answered, pending = asks
    assert answered.command_response.success is True
    assert answered.command_response.artifacts["question_kind"] == "agent"
    assert _payload_keys(answered.command_response.artifacts) == []
    assert pending.command_response.artifacts["question_kind"] == "approval"
    assert pending.command_response.artifacts["approval"]["command"] == "Record/add_note"
    assert "review" not in pending.command_response.artifacts
    assert "parameter_request" not in pending.command_response.artifacts


def test_a_declined_slash_approval_leaves_nothing_described_and_does_not_run(todo, annotated_library, grant_root):
    waiting = todo.ctx.process_turn("/add_note text=Ann")
    assert _last_artifacts(waiting)["question_kind"] == "approval"
    declined = todo.ctx.process_turn("no")
    assert _added(todo.chores, todo.base) == []
    for output in declined.command_outputs:
        assert not any(key in output.command_response.artifacts for key in ("question_kind", *PAYLOAD_KEYS))
    assert todo.store.active() is None
