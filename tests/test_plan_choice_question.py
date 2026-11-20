"""The plan-choice question (fix-yu3l.5, ``fastworkflow/intent_clarification.py``).

With no agent model, a turn part that matches no planning command used to end as not run at once. A turn somebody can
answer ASKS which planning command is meant. The question is asked while the turn's parts are being resolved
(``fastworkflow/_workflows/plan_metadata_extraction/resolution.py``, bd fix-o1lg), before anything runs and before the turn's one
review: the picked plan is bound to the part, a missing value is asked for through the parameter loop, and the later
parts follow. ``none`` or ``cancel`` ends the part as the same not-run entry as before, and so does a turn nobody can
answer.

First the reader with no turn around it, then whole turns on the library example, driven as
tests/test_parameter_loop.py drives them: the real context on the annotated library copy, no agent model, the planner's
split, the matcher and the shared extractor's model scripted, and every model call or network connection refused.
"""
from __future__ import annotations

import json

import dspy
import pytest

from fastworkflow import intent_clarification as choice
from fastworkflow import no_agent_mode, workflow_agent
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import build_query_with_next_steps as REAL_BUILD
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT
from tests.test_approval_binding import _finish, annotated_library, todo  # noqa: F401
from tests.test_parameter_loop import (  # noqa: F401 - the no-agent context and its helpers
    ABOUT_DUNE, ABOUT_EMMA, BOOK_PLAN, DUNE_RAN, EMMA_RAN, HOST, MEMBER, NOTE, NOTE_PLAN, NOTE_RAN, _as_a_new_process,
    _lookups, _loop, _offline, _reply, _request, _runs, _TextPlanner, no_agent)
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.test_use_plan import NO_PARAMETERS, _root_plan_added  # noqa: F401 - a planning command with no parameters
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

HAIKU = "write me a haiku about the weather in Paris"
NO_FIT = ("no planning command or command fits it, so it needs free planning, which is not run without an agent "
          "model (LLM_AGENT)")
NAMES = ["Library/book_by_title", "Library/note_member", "Library/overdue_summary", "branch_names"]


# ======================================================================================================================
# The reader
# ======================================================================================================================

def _read(reply, candidates, *, numbered=None):
    """The one reader's answer as ``(kind, name)``."""
    reading = choice.read(reply, candidates, numbered=numbered)
    return reading.kind, reading.name


@pytest.mark.parametrize("reply, expected", [
    # the exact name, its leaf, any case, a leading slash, quotes
    ("Library/note_member", ("pick", "Library/note_member")),
    ("note_member", ("pick", "Library/note_member")),
    ("NOTE_MEMBER.", ("pick", "Library/note_member")),
    ("/note_member", ("pick", "Library/note_member")),
    ("`branch_names`", ("pick", "branch_names")),
    # the humanised name
    ("note member", ("pick", "Library/note_member")),
    ("Book By Title", ("pick", "Library/book_by_title")),
    ("library note member", ("pick", "Library/note_member")),
    # a number of the list
    ("1", ("pick", "Library/book_by_title")),
    ("4.", ("pick", "branch_names")),
    ("#2", ("pick", "Library/note_member")),
    ("0", ("unclear", None)),
    ("5", ("unclear", None)),
    # a misspelt name, within the bound
    ("note_membr", ("pick", "Library/note_member")),
    ("overdue sumary", ("pick", "Library/overdue_summary")),
    # none of these
    ("none", ("none", None)),
    ("None of these.", ("none", None)),
    ("no", ("none", None)),
    ("neither", ("none", None)),
    ("something else", ("none", None)),
    # cancel
    ("cancel", ("abort", None)),
    ("Abort!", ("abort", None)),
    ("never mind", ("abort", None)),
    # nothing readable
    ("", ("unclear", None)),
    ("   ", ("unclear", None)),
    ("hmm, I am not sure which", ("unclear", None)),
    ("the second one maybe", ("unclear", None)),
    ("bo", ("unclear", None)),
])
def test_a_reply_to_the_choice_is_read_in_code(reply, expected):
    assert _read(reply, NAMES) == expected


def test_a_number_picks_only_among_the_options_the_question_numbered():
    assert _read("3", NAMES, numbered=2) == ("unclear", None)
    assert _read("2", NAMES, numbered=2) == ("pick", "Library/note_member")
    # A name that was not numbered is still a candidate.
    assert _read("branch names", NAMES, numbered=2) == ("pick", "branch_names")


def test_two_candidates_that_match_equally_well_are_unclear_and_never_guessed():
    # The matcher scores a candidate on its leading characters: "book" begins both.
    assert _read("book", ["Library/book_by_title", "Library/book_loans"]) == ("unclear", None)
    assert _read("book_loans", ["Library/book_by_title", "Library/book_loans"]) == (
        "pick", "Library/book_loans")
    # The same leaf in two contexts: the leaf picks neither, the full name picks one.
    twins = ["Library/find", "Archive/find"]
    assert _read("find", twins) == ("unclear", None)
    assert _read("archive/find", twins) == ("pick", "Archive/find")


def test_the_word_for_none_is_not_taken_as_the_start_of_a_name():
    assert _read("no", ["Library/note_member", "Library/notify"]) == ("none", None)
    assert _read("none", ["Library/nonesuch"]) == ("none", None)
    # A plan that is really called that is picked by its name.
    assert _read("cancel", ["Library/cancel"]) == ("pick", "Library/cancel")


def test_the_fuzzy_bound_is_the_one_the_cme_uses_for_a_typed_command_name():
    from fastworkflow._workflows.command_metadata_extraction import intent_detection

    assert choice.FUZZY_MAX_DISTANCE == intent_detection._FUZZY_PREMATCH_MAX_DISTANCE


def test_the_question_numbers_the_listed_plans_and_says_how_many_more_there_are():
    plans = [{"name": f"Ctx/plan_{i:02d}", "description": f"does thing {i}", "closest": i < 2} for i in range(15)]
    text = choice.plan_choice_question("do the thing", plans, listed=no_agent_mode.AD_HOC_LIST_CAP)

    assert text.startswith("No planning command matched this request: 'do the thing'. Which one should run it?\n"
                           "1. Ctx/plan_00: does thing 0\n")
    assert "\n12. Ctx/plan_11: does thing 11\n" in text and "13. " not in text
    assert "\nand 3 more; the name of any planning command available here can be given.\n" in text
    assert text.endswith("Reply with a name or number, 'none' if none of these fits, or cancel. "
                         "Only your answer is read; anything else in the reply is ignored.")
    split = choice.plan_choice_question("do the thing", plans[:1], sub_task=2, retry=choice.RETRY)
    assert split.startswith("That did not name one of them. No planning command matched part 2: 'do the thing'.")


# ======================================================================================================================
# Whole turns, no agent model
# ======================================================================================================================

@pytest.fixture(autouse=True)
def _nothing_serves_the_requests_that_are_not_about_the_library(request, monkeypatch):
    """The remote decision model says ``none`` of the jokes and the haiku (nothing of the library serves them), so the
    decision is "none" and the part is ad hoc. Left unanswered, the local model's unsure top three would be an
    ambiguity (asked as such, tests/test_resolve_before_review.py)."""
    if "no_agent" not in request.fixturenames:
        return
    env = request.getfixturevalue("no_agent")
    plans = {ABOUT_EMMA: BOOK_PLAN, ABOUT_DUNE: BOOK_PLAN, "put a note on a record": NOTE_PLAN}

    def match(text):
        name = plans.get(text)
        return {name: 0.9, "none": 0.0} if name else {"none": 1.0}

    env.matched = script_matcher(monkeypatch, match)


def _choice(out) -> dict:
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "plan_choice", artifacts
    assert "parameter_request" not in artifacts and "review" not in artifacts
    return artifacts["plan_choice"]


def _ad_hoc_stops(out) -> list[dict]:
    return [stop for stop in out.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT, [])
            if stop["command"] is None]


def _asked_about_the_joke(env):
    """A turn of one part that no planning command matches: it asks which one is meant."""
    env.parts = [JOKE]
    asked = env.ctx.process_turn(JOKE, approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER, asked.answer
    return asked


def _plans_for(env, text):
    """The plans an ad hoc part lists: those available here, its closest first, from the same decision the turn made
    (``resolution.part_intent``: the local model's ranking when the remote model gives no answer)."""
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    return no_agent_mode.available_plans(env.wf, resolution.part_intent(env.wf, text).planning_scores())


def test_an_attended_ad_hoc_part_asks_which_plan_is_meant(no_agent):
    asked = _asked_about_the_joke(no_agent)
    ctx = no_agent.ctx

    data = _choice(asked)
    plans = _plans_for(no_agent, JOKE)
    assert data == {"request": JOKE, "sub_task": None, "choices": plans, "none_reply": "none", "decline": "cancel",
                    "ambiguous": False, "misunderstood": False}
    assert not data.get("ambiguous")
    assert all(set(plan) == {"name", "description", "closest"} for plan in data["choices"]) and len(plans) > 1
    assert asked.answer == choice.plan_choice_question(JOKE, plans, listed=no_agent_mode.AD_HOC_LIST_CAP)
    assert asked.answer.startswith(f"No planning command matched this request: {JOKE!r}. Which one should run it?\n1. ")
    assert NOT_RUN_ARTIFACT not in asked.command_outputs[-1].command_response.artifacts
    held = ctx._waiting
    record = held.payload["record"]
    assert (record["origin"], record["asks"], record["stalled"], held.kind) == (0, 1, 0, "intent")
    assert "loop" not in held.payload
    assert ctx._resolution_pending and ctx.has_pending_state()
    # Nothing ran.
    assert not [o for o in ctx._turn_outputs if str(o.command_name).startswith("Library/")]


@pytest.mark.parametrize("how", ["name", "leaf", "humanised", "number"])
def test_a_picked_plan_with_nothing_missing_runs_at_once(no_agent, how):
    asked = _asked_about_the_joke(no_agent)
    names = [plan["name"] for plan in _choice(asked)["choices"]]
    reply = {"name": NO_PARAMETERS, "leaf": NO_PARAMETERS.rpartition("/")[2], "humanised": "Branch Names",
             "number": str(names.index(NO_PARAMETERS) + 1)}[how]
    matched = len(no_agent.matched)

    done = _reply(no_agent, reply)

    assert done.status is TurnStatus.COMPLETED, done.answer
    assert done.answer.startswith("Branches: ")
    assert not done.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT)
    ctx = no_agent.ctx
    assert ctx._plan_resolution is None and not ctx._resolution_pending
    # The part's command was chosen by the user, so like a named command it runs with no review.
    assert ctx._turn_subtask_review == "fast_path"
    # The plan was written on the part, so it took the planning path; the matcher was not asked again.
    item = ctx._turn_subtask_plan.items[0]
    assert (item.means.kind, item.means.name) == ("planning", NO_PARAMETERS)
    assert len(no_agent.matched) == matched and len(no_agent.splits) == 1


def test_a_pick_goes_to_the_parameter_question_then_one_review_and_the_turn_keeps_every_part(no_agent):
    """Three parts; the second matches no plan. The question comes first, before any part runs: the pick is asked for
    its values, then the one review shows all three parts, and only then does anything run."""
    no_agent.parts = [ABOUT_EMMA, JOKE, ABOUT_DUNE]
    asked = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {JOKE}; {ABOUT_DUNE}", approval_surface=HOST)
    ctx = no_agent.ctx

    assert asked.status is TurnStatus.AWAITING_USER, asked.answer
    data = _choice(asked)
    assert (data["request"], data["sub_task"]) == (JOKE, 2)
    assert asked.answer.startswith(f"No planning command matched part 2: {JOKE!r}.")
    assert _lookups(no_agent) == [] and ctx._waiting.payload["record"]["origin"] == 1
    splits, matched = len(no_agent.splits), len(no_agent.matched)

    values = _reply(no_agent, "note member")

    # The picked plan needs its parameters only: no task text, no review of the part.
    assert values.status is TurnStatus.AWAITING_USER, values.answer
    assert _request(values)["missing"] == ["member", "text"] and _request(values)["command"] == NOTE_PLAN
    loop = _loop(ctx)
    assert (loop.command, loop.origin, loop.confirm) == (NOTE_PLAN, 1, False)
    assert ctx._waiting.kind == "parameters" and _lookups(no_agent) == []

    review = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")

    assert review.status is TurnStatus.AWAITING_USER, review.answer
    items = review.command_outputs[-1].command_response.artifacts["review"]["items"]
    assert [(item["goal"], item["means"]["name"]) for item in items] == [
        (ABOUT_EMMA, BOOK_PLAN), (JOKE, NOTE_PLAN), (ABOUT_DUNE, BOOK_PLAN)]
    assert _lookups(no_agent) == []
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    assert _lookups(no_agent) == [EMMA_RAN, NOTE_RAN, DUNE_RAN] and len(_runs(no_agent, BOOK_PLAN)) == 2
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2.", "### 3."]
    assert not _ad_hoc_stops(done)
    # No second split, and the matcher was not asked about any part again.
    assert len(no_agent.splits) == splits == 1 and len(no_agent.matched) == matched


@pytest.mark.parametrize("reply, reason", [
    # ``none`` leaves the part ad hoc: it ends as the not-run entry every ad hoc part ends as.
    ("none", NO_FIT),
    ("something else", NO_FIT),
    ("cancel", "it was declined when asked which planning command should run it"),
])
def test_none_or_cancel_ends_the_part_as_the_not_run_entry(no_agent, reply, reason):
    asked = _asked_about_the_joke(no_agent)
    plans = _choice(asked)["choices"]

    done = _reply(no_agent, reply)

    assert done.status is TurnStatus.COMPLETED
    (stop,) = _ad_hoc_stops(done)
    # The entry a turn nobody can answer gets: the same keys, the plans and the request, so use_plan still works.
    assert set(stop) == {"kind", "command", "phase", "open_step", "done", "facts", "remaining", "suggested_plan",
                         "means", "sub_task", "reason", "available_plans", "request"}
    assert (stop["available_plans"], stop["request"], stop["means"], stop["reason"]) == (
        plans, JOKE, "from_scratch", reason)
    assert done.answer == no_agent_mode.explanation(stop)
    assert done.answer.startswith("No matching plan was found for this part.")
    assert not no_agent.ctx._resolution_pending and not no_agent.ctx.has_pending_state()


def test_a_reply_that_cannot_be_read_is_asked_once_more_then_the_part_is_not_run(no_agent):
    asked = _asked_about_the_joke(no_agent)

    again = _reply(no_agent, "hmm, I am not sure which")

    assert again.status is TurnStatus.AWAITING_USER
    assert again.answer == choice.RETRY + asked.answer and _choice(again) == _choice(asked)
    held = no_agent.ctx._waiting.payload["record"]
    assert (held["asks"], held["stalled"]) == (2, 1)
    done = _reply(no_agent, "really, I cannot say")

    assert done.status is TurnStatus.COMPLETED
    (stop,) = _ad_hoc_stops(done)
    assert stop["reason"] == NO_FIT
    assert stop["available_plans"] == _choice(asked)["choices"] and stop["request"] == JOKE


def test_a_turn_nobody_can_answer_ends_the_part_as_before_with_no_question(no_agent):
    no_agent.parts = [JOKE]
    out = no_agent.ctx.process_turn(JOKE, approval_surface=HOST, unattended=True)

    assert out.status is TurnStatus.COMPLETED
    (stop,) = _ad_hoc_stops(out)
    assert stop["request"] == JOKE and stop["available_plans"] == _plans_for(no_agent, JOKE)
    assert "needs free planning" in stop["reason"]
    assert no_agent.ctx._plan_resolution is None and out.answer == no_agent_mode.explanation(stop)


def test_each_ad_hoc_part_of_a_split_turn_is_asked_about_when_the_turn_reaches_it(no_agent):
    no_agent.parts = [JOKE, HAIKU]
    first = no_agent.ctx.process_turn(f"{JOKE} and {HAIKU}", approval_surface=HOST)

    assert first.status is TurnStatus.AWAITING_USER, first.answer
    assert (_choice(first)["request"], _choice(first)["sub_task"]) == (JOKE, 1)
    second = _reply(no_agent, "none")
    assert second.status is TurnStatus.AWAITING_USER, second.answer
    assert (_choice(second)["request"], _choice(second)["sub_task"]) == (HAIKU, 2)
    assert no_agent.ctx._waiting.payload["record"]["origin"] == 1

    # Part 1 stays ad hoc (nothing fits it, shown as not run here) and part 2 is bound: one review of both, then the run.
    review = _reply(no_agent, NO_PARAMETERS)
    assert review.status is TurnStatus.AWAITING_USER, review.answer
    items = review.command_outputs[-1].command_response.artifacts["review"]["items"]
    assert [(item["goal"], item["not_run_here"]) for item in items] == [(JOKE, True), (HAIKU, False)]
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    (stop,) = _ad_hoc_stops(done)
    assert (stop["sub_task"], stop["request"]) == (1, JOKE) and stop["available_plans"]
    assert "Branches: " in done.answer and len(no_agent.splits) == 1


def test_the_question_survives_a_save_and_the_reply_resumes_the_same_part_after_the_restore(no_agent):
    no_agent.parts = [ABOUT_EMMA, JOKE, ABOUT_DUNE]
    asked = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {JOKE}; {ABOUT_DUNE}", approval_surface=HOST)
    ctx = no_agent.ctx
    data = _choice(asked)
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="choice-restore")))

    state, saved = blob["plan_resolution"], blob["waiting"]
    assert saved["kind"] == "intent" and state["index"] == 1
    assert saved["payload"]["record"]["origin"] == 1 and saved["payload"]["record"]["request"] == JOKE
    assert not {"plan_choice", "parameter_loop"} & set(blob)
    _as_a_new_process(ctx, blob)

    assert ctx._plan_resolution is not None
    assert ctx._awaiting_user is True and ctx._resolution_pending and ctx.has_pending_state()
    assert ctx.waiting.question == asked.answer
    kind, payload = ctx._pending_prompt_kind(ctx._turn_outputs[-1])
    assert kind == "plan_choice" and payload["plan_choice"] == data
    review = _reply(no_agent, NO_PARAMETERS)

    assert review.status is TurnStatus.AWAITING_USER, review.answer
    assert [item["means"]["name"] for item in review.command_outputs[-1].command_response.artifacts["review"]["items"]
            ] == [BOOK_PLAN, NO_PARAMETERS, BOOK_PLAN]
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    assert _lookups(no_agent) == [EMMA_RAN, DUNE_RAN] and len(no_agent.splits) == 1
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2.", "### 3."]
    assert "Branches: " in done.answer.split("### 2. ")[1].split("### 3. ")[0]


def test_dropping_what_is_pending_drops_the_question(no_agent):
    _asked_about_the_joke(no_agent)
    ctx = no_agent.ctx

    assert ctx.cancel_pending() is True
    assert not ctx._resolution_pending and not ctx._awaiting_user
    _asked_about_the_joke(no_agent)
    assert ctx.discard_live_pending() is True
    assert ctx._plan_resolution is None and not ctx.has_pending_state()


def test_a_state_saved_before_the_resolution_existed_loads_with_none_out(no_agent):
    """Old saved sessions are not supported: the ``plan_choice`` key an earlier build wrote is ignored."""
    ctx = no_agent.ctx
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="old")))
    blob.pop("plan_resolution")
    blob["plan_choice"] = {"origin": 0, "choices": [{"name": "x"}], "pending": True}
    ctx._plan_resolution = None

    ctx.apply_serialized_state(blob)

    assert ctx._plan_resolution is None and not ctx._resolution_pending
    assert "plan_choice" not in ctx.serialize_state(channel_id="old")


# ======================================================================================================================
# With an agent model nothing changes: the agent plans an ad hoc part itself
# ======================================================================================================================

def test_with_an_agent_model_no_plan_choice_is_asked(todo, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda *args, **kwargs: None)
    script_matcher(monkeypatch, {})                              # the remote model gives no answer
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: [user_query])
    planner = _TextPlanner([], adapter=dspy.JSONAdapter())
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda *args, **kwargs: REAL_BUILD(*args, **{**kwargs, "planner_lm": planner}))
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    ctx = todo.ctx
    asked = []
    real = resolution._ask_choice
    monkeypatch.setattr(resolution, "_ask_choice", lambda *a, **k: asked.append(a) or real(*a, **k))
    todo.script.steps = _finish()

    done = ctx.process_turn(JOKE)

    assert ctx.no_agent_mode is False
    assert done.status is TurnStatus.COMPLETED, done.answer
    # The decision is never reached with an agent model, and no record of a question exists.
    assert asked == [] and ctx._plan_resolution is None
    assert all(NOT_RUN_ARTIFACT not in output.command_response.artifacts for output in done.command_outputs)
