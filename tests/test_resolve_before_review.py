"""Every part of a turn is resolved before the turn's ONE review (bd fix-o1lg, steps S4 and S5).

In order: which command or plan each part gets (asked only when the decision is ambiguous, or with no agent model when
nothing serves the part), then the values a bound planning command still lacks (asked only when it lacks one; a value an
earlier part produces is never asked for); each question can be cancelled, which drops that part while the later parts
go on; then one review of the whole turn with nothing missing; then the run, which asks nothing more (a value refused
at run time is not asked about again). The pending state is one JSON-safe record saved under ``plan_resolution``.

Real components on the annotated library copy; the remote decision model's answer, the planner's split and the shared
extractor's model are the scripted seams, and every model call or network connection is refused (the autouse
``_offline`` fixture of tests/test_parameter_loop.py).
"""
from __future__ import annotations

import json

import pytest

from fastworkflow.review import review_proposals
from fastworkflow.core import waiting
from fastworkflow.parameters.parameter_clarification import ParameterLoop
from fastworkflow._workflows.plan_metadata_extraction import resolution
from fastworkflow.core.turn import TurnStatus
from fastworkflow.engine.workflow_execution_context import NOT_RUN_ARTIFACT
from tests.conftest import script_matcher
from tests.test_approval_binding import annotated_library, todo  # noqa: F401
from tests.test_parameter_loop import (  # noqa: F401 - the no-agent context and its helpers
    ABOUT_DUNE, ABOUT_EMMA, A_NOTE, BOOK_PLAN, DUNE_RAN, EMMA_RAN, HOST, MEMBER, NOTE, NOTE_PLAN, NOTE_RAN,
    _as_a_new_process, _loop, _lookups, _offline, _reply, _request, no_agent)
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FIND_BOOK = "Library/find_book"
AUTHOR = "Library/author_books"
BY_TITLE = BOOK_PLAN


def _artifacts(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def _kind(out) -> str:
    """What the turn is waiting on: the question kind, ``review`` for the review, ``""`` for neither."""
    artifacts = _artifacts(out)
    if artifacts.get("question_kind"):
        return artifacts["question_kind"]
    return "review" if "review" in artifacts else ""


@pytest.fixture
def ambiguous_joke(no_agent, monkeypatch):
    """No agent model. The remote model reads the joke as one of three commands (a plain one among them), the other
    parts as their plans, so the joke is an ambiguity, asked about."""
    plans = {ABOUT_EMMA: BY_TITLE, ABOUT_DUNE: BY_TITLE, A_NOTE: NOTE_PLAN}

    def match(text):
        if text == JOKE:
            return {FIND_BOOK: 0.45, AUTHOR: 0.3, BY_TITLE: 0.25, "none": 0.0}
        return {plans[text]: 0.9, "none": 0.0}

    no_agent.matched = script_matcher(monkeypatch, match)
    return no_agent


# ======================================================================================================================
# No agent model
# ======================================================================================================================

def test_the_questions_come_in_order_before_the_review_and_nothing_runs_until_it_is_accepted(ambiguous_joke):
    env = ambiguous_joke
    env.parts = [ABOUT_EMMA, JOKE, A_NOTE]

    which = env.ctx.process_turn(f"{ABOUT_EMMA}; {JOKE}; {A_NOTE}", approval_surface=HOST)
    # 1. which command: part 2 is an ambiguity, its candidates (a plain command among them) are the choices.
    assert _kind(which) == "plan_choice" and which.status is TurnStatus.AWAITING_USER, which.answer
    data = _artifacts(which)["plan_choice"]
    assert (data["request"], data["sub_task"]) == (JOKE, 2)
    assert [choice["name"] for choice in data["choices"]] == [FIND_BOOK, AUTHOR, BY_TITLE]
    assert which.answer.startswith(f"More than one command fits part 2: {JOKE!r}.")
    assert _lookups(env) == []

    # 2. the picked plain command is asked for its required value (no agent model); part 3's missing values follow.
    query = _reply(env, "find_book")
    assert _kind(query) == "planning_parameter" and _request(query)["command"] == FIND_BOOK
    assert _request(query)["missing"] == ["query"] and _lookups(env) == []
    values = _reply(env, "Emma")
    assert _kind(values) == "planning_parameter" and _request(values)["command"] == NOTE_PLAN
    assert _request(values)["missing"] == ["member", "text"] and _lookups(env) == []

    # 3. the one review, of all three parts, with nothing missing in it.
    review = _reply(env, f"member: {MEMBER}\ntext: {NOTE}")
    assert _kind(review) == "review", review.answer
    items = _artifacts(review)["review"]["items"]
    assert [(item["means"]["kind"], item["means"]["name"]) for item in items] == [
        ("planning", BY_TITLE), ("plain", FIND_BOOK), ("planning", NOTE_PLAN)]
    assert _lookups(env) == []

    # 4. the run asks nothing more.
    done = _reply(env, "yes")
    assert done.status is TurnStatus.COMPLETED, done.answer
    assert EMMA_RAN in _lookups(env) and NOTE_RAN in _lookups(env)
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2.", "### 3."]
    assert len(env.splits) == 1


def test_an_ambiguity_with_one_candidate_is_not_asked_but_bound_and_reviewed(no_agent, monkeypatch):
    """A list of one is no choice: the candidate is bound as drafted and the turn's ONE review (which shows the plan
    and its values, and can be declined) is the confirmation. Nobody chose it, so the review is NOT skipped."""
    no_agent.matched = script_matcher(monkeypatch, lambda text: {BY_TITLE: 0.45, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]

    out = no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)

    assert _kind(out) == "review", out.answer
    assert "More than one" not in out.answer
    items = _artifacts(out)["review"]["items"]
    assert [(item["means"]["kind"], item["means"]["name"]) for item in items] == [("planning", BY_TITLE)]
    assert _lookups(no_agent) == []
    assert _reply(no_agent, "yes").status is TurnStatus.COMPLETED
    assert EMMA_RAN in _lookups(no_agent)


def test_an_unattended_turn_with_one_candidate_asks_nothing_and_binds_nothing(no_agent, monkeypatch):
    """Nobody can answer or review: the part stays as it was (not bound by the one-candidate rule) and no question is
    put."""
    no_agent.matched = script_matcher(monkeypatch, lambda text: {BY_TITLE: 0.45, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]

    out = no_agent.ctx.process_turn(ABOUT_EMMA, unattended=True, approval_surface=HOST)

    assert _kind(out) == "" and out.status is not TurnStatus.AWAITING_USER, out.answer
    assert "More than one" not in out.answer


def test_a_list_of_one_is_a_confirmation_of_that_one_command():
    """The single-candidate form of the ONE question text (the deterministic path has no review to confirm it): it is
    answered by the name, yes or cancel, and never says "more than one"."""
    from fastworkflow.intent import intent_clarification

    one = intent_clarification.plan_choice_question(
        "tell me a joke", [{"name": FIND_BOOK, "description": ""}], ambiguous=True)
    two = intent_clarification.plan_choice_question(
        "tell me a joke", [{"name": FIND_BOOK}, {"name": AUTHOR}], ambiguous=True)
    assert "More than one" not in one and "ambiguous" not in one
    assert one.startswith("Did you mean the command 'find_book'?") and "'yes'" in one
    assert "More than one command fits" in two and "1. " + FIND_BOOK in two
    # and the reader takes the confirmation words
    assert intent_clarification.read("yes", [FIND_BOOK]).name == FIND_BOOK
    assert intent_clarification.read("yes", [FIND_BOOK, AUTHOR]).kind == "unclear"


def _learned(env) -> list[dict]:
    from fastworkflow.intent import intent_learning

    return intent_learning.rows(env.ctx._app_workflow.folderpath)


def test_a_part_the_review_accepted_is_recorded_confirmed_and_a_part_the_user_picked_is_a_user_choice(
        ambiguous_joke):
    """The resolution path writes ONE row per part where the plan is run: the part the user picked at the question is a
    user choice; the others, which the review showed and the user accepted, are confirmed review accepts."""
    env = ambiguous_joke
    env.parts = [ABOUT_EMMA, JOKE]

    _artifacts(env.ctx.process_turn(f"{ABOUT_EMMA}; {JOKE}", approval_surface=HOST))
    query = _reply(env, "find_book")
    assert _kind(query) == "planning_parameter" and _request(query)["command"] == FIND_BOOK
    assert _request(query)["missing"] == ["query"]
    review = _reply(env, "Emma")
    assert _kind(review) == "review"
    assert _learned(env) == []                       # nothing is settled until the plan is run
    assert _reply(env, "yes").status is TurnStatus.COMPLETED

    by_text = {row["text"]: row for row in _learned(env)}
    assert (by_text[JOKE]["how"], by_text[JOKE]["command"], by_text[JOKE]["confirmed"]) == (
        "user_choice", FIND_BOOK, 1)
    assert (by_text[ABOUT_EMMA]["how"], by_text[ABOUT_EMMA]["command"], by_text[ABOUT_EMMA]["confirmed"]) == (
        "review_accept", BY_TITLE, 1)
    assert by_text[ABOUT_EMMA]["remote_label"] == BY_TITLE and by_text[ABOUT_EMMA]["remote_score"] == 0.9


def test_a_one_candidate_part_the_review_accepted_teaches_and_an_unreviewed_name_does_not(no_agent, monkeypatch):
    no_agent.matched = script_matcher(monkeypatch, lambda text: {BY_TITLE: 0.45, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]
    assert _kind(no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)) == "review"
    _reply(no_agent, "yes")
    (row,) = _learned(no_agent)      # the plan's own typed calls (`find_book query=...`) are not requests: not recorded
    assert (row["how"], row["command"], row["confirmed"]) == ("review_accept", BY_TITLE, 1)

    # A command named exactly is the fast path: recorded as a name match, which never teaches.
    no_agent.parts = ["find_book"]
    query = no_agent.ctx.process_turn("find_book", approval_surface=HOST)
    assert _kind(query) == "planning_parameter" and _request(query)["command"] == FIND_BOOK
    assert _request(query)["missing"] == ["query"]
    _reply(no_agent, "Emma")
    named = [each for each in _learned(no_agent) if each["how"] == "name"]
    assert named and named[0]["command"] == FIND_BOOK and named[0]["confirmed"] == 0


def test_cancelling_the_which_command_question_drops_that_part_and_the_later_parts_go_on(ambiguous_joke):
    env = ambiguous_joke
    env.parts = [JOKE, ABOUT_EMMA]

    asked = env.ctx.process_turn(f"{JOKE}; {ABOUT_EMMA}", approval_surface=HOST)
    assert _kind(asked) == "plan_choice"
    review = _reply(env, "cancel")

    # The joke is dropped; the remote model's match for the emma part is reviewed, alone.
    assert _kind(review) == "review"
    assert [item["goal"] for item in _artifacts(review)["review"]["items"]] == [ABOUT_EMMA]
    done = _reply(env, "yes")
    assert done.status is TurnStatus.COMPLETED, done.answer
    (stop,) = [stop for stop in _artifacts(done)[NOT_RUN_ARTIFACT] if stop["command"] is None]
    assert stop["request"] == JOKE and stop["reason"] == "it was declined when asked which command should run it"
    assert _lookups(env) == [EMMA_RAN]


def test_cancelling_a_values_question_drops_that_part_and_the_review_shows_the_rest(no_agent):
    asked = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    assert _kind(asked) == "planning_parameter"

    review = _reply(no_agent, "cancel")

    assert _kind(review) == "review"
    assert [item["goal"] for item in _artifacts(review)["review"]["items"]] == [ABOUT_EMMA, ABOUT_DUNE]
    done = _reply(no_agent, "yes")
    assert done.status is TurnStatus.COMPLETED and _lookups(no_agent) == [EMMA_RAN, DUNE_RAN]
    # The review's items are renumbered: the answer has two parts, not three with a hole.
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2."]


def test_a_value_an_earlier_part_produces_is_never_asked_for(no_agent, monkeypatch):
    """The declared source of the note's member is part 1's output: the resolution asks for the text alone, and a
    cancelled part 1 makes the member the user's to give, so it is asked for then."""
    real = review_proposals._producers
    monkeypatch.setattr(review_proposals, "_producers",
                        lambda workflow, means, earlier: {"member": 1} if means.name == NOTE_PLAN and earlier
                        else real(workflow, means, earlier))
    no_agent.parts = [ABOUT_EMMA, A_NOTE]

    asked = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}", approval_surface=HOST)

    assert _request(asked)["missing"] == ["text"] and _loop(no_agent.ctx).open == ["text"]
    plan = no_agent.ctx._turn_subtask_plan
    assert [(each.name, each.from_subtask) for each in plan.items[1].inputs] == [("member", 1)]
    assert resolution._from_earlier_parts(plan.items[1]) == {"member"}
    # Dropping part 1 (it would have produced the member) leaves the member for the user: the later part is not
    # "from" a part that does not run.
    dropped = resolution._drop(
        resolution.PlanResolution(parts=[{"state": "open"}, {"state": "open"}]), plan, "test")
    assert [each.name for each in dropped.items[1].inputs] == []


def test_a_value_refused_at_run_time_is_not_asked_again(no_agent):
    """The values were answered and the review accepted after an edit that makes the member come from a part that
    cannot produce it: the run refuses it, and the turn ends that part as not run. Nothing is asked a second time."""
    no_agent.parts = [ABOUT_EMMA, A_NOTE]
    first = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}", approval_surface=HOST)
    assert _kind(first) == "planning_parameter"
    review = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")
    assert _kind(review) == "review"
    edited = _reply(no_agent, "change 2: inputs=member=from sub-task 1")
    assert _kind(edited) == "review"

    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    (stop,) = [stop for stop in _artifacts(done).get(NOT_RUN_ARTIFACT, [])
               if stop["command"] == NOTE_PLAN and stop["open_step"] is None]
    assert "member" in stop["reason"]
    assert no_agent.ctx._plan_resolution is None and not no_agent.ctx._resolution_pending
    assert _lookups(no_agent) == [EMMA_RAN]


def test_the_pending_resolution_is_one_json_record_that_round_trips_through_a_save(ambiguous_joke):
    env = ambiguous_joke
    env.parts = [ABOUT_EMMA, JOKE, A_NOTE]
    asked = env.ctx.process_turn(f"{ABOUT_EMMA}; {JOKE}; {A_NOTE}", approval_surface=HOST)
    ctx = env.ctx

    state = ctx._plan_resolution.to_state()
    assert json.loads(json.dumps(state)) == state
    assert resolution.PlanResolution.from_state(state).to_state() == state
    assert (state["index"], state["stage"]) == (1, "intent")
    assert (ctx._waiting.kind, ctx._waiting.owner) == ("intent", "plan")
    assert [part["state"] for part in state["parts"]] == ["resolved", "open", "open"]
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="resolution")))
    assert blob["plan_resolution"] == state and not {"plan_choice", "parameter_loop"} & set(blob)
    _as_a_new_process(ctx, blob)
    assert ctx._plan_resolution.to_state() == state and ctx.waiting.question == asked.answer

    # The reply after the restore carries on exactly where the resolution stood: the next question, then the review.
    query = _reply(env, "find_book")
    assert _kind(query) == "planning_parameter" and _request(query)["command"] == FIND_BOOK
    assert _request(query)["missing"] == ["query"] and _loop(ctx).origin == 1
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="resolution")))
    _as_a_new_process(ctx, blob)
    values = _reply(env, "Emma")
    assert _kind(values) == "planning_parameter" and _loop(ctx).origin == 2
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="resolution")))
    assert blob["waiting"]["kind"] == "parameters" and blob["plan_resolution"]["index"] == 2
    _as_a_new_process(ctx, blob)
    review = _reply(env, f"member: {MEMBER}\ntext: {NOTE}")
    assert _kind(review) == "review"
    done = _reply(env, "yes")
    assert done.status is TurnStatus.COMPLETED and NOTE_RAN in _lookups(env)
    assert ctx._plan_resolution is None and ctx.serialize_state(channel_id="resolution")["plan_resolution"] is None


def test_a_pinned_plans_parameter_question_is_the_review_of_that_turn(no_agent):
    asked = no_agent.ctx.process_turn(A_NOTE, use_plan=NOTE_PLAN, approval_surface=HOST)
    assert _kind(asked) == "planning_parameter" and _request(asked)["confirm"] is True

    done = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")      # every value given: no review follows

    assert done.status is TurnStatus.COMPLETED, done.answer
    assert no_agent.ctx._turn_subtask_review == "not_reviewed" and NOTE_RAN in _lookups(no_agent)


# ======================================================================================================================
# With an agent model: the same order
# ======================================================================================================================

def _split_into(monkeypatch, parts):
    monkeypatch.setattr("fastworkflow.agent.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: list(parts))


def test_with_an_agent_model_an_ambiguous_part_is_asked_before_the_review_and_may_be_a_plain_command(gated, monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    emma = "who wrote Emma and how many copies do we have"
    script_matcher(monkeypatch, lambda text: (
        {FIND_BOOK: 0.45, AUTHOR: 0.3, BY_TITLE: 0.25, "none": 0.0} if text == JOKE else {BY_TITLE: 0.9, "none": 0.0}))
    _split_into(monkeypatch, [emma, JOKE])

    asked = gated.ctx.process_turn(f"{emma} and {JOKE}")

    assert asked.status is TurnStatus.AWAITING_USER and _kind(asked) == "plan_choice"
    assert gated.store.active() is None                    # no review yet
    data = _artifacts(asked)["plan_choice"]
    assert [choice["name"] for choice in data["choices"]] == [FIND_BOOK, AUTHOR, BY_TITLE]
    assert gated.ctx._waiting.payload["record"]["ambiguous"] is True

    review = gated.ctx.process_turn(FIND_BOOK)

    assert review.status is TurnStatus.AWAITING_USER
    record = gated.store.active()
    assert record is not None and record.command == "review_subtasks"
    assert [(item["means"]["kind"], item["means"]["name"]) for item in record.binding["items"]] == [
        ("planning", BY_TITLE), ("plain", FIND_BOOK)]
    gated.script.steps = [("finish", {})] * 2
    done = gated.ctx.process_turn("accept")
    assert done.status is TurnStatus.COMPLETED and gated.ctx._turn_subtask_review == "reviewed"
    assert gated.ctx._plan_resolution is None


def test_with_an_agent_model_cancelling_the_question_drops_the_part_and_the_review_shows_the_rest(gated, monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    emma = "who wrote Emma and how many copies do we have"
    script_matcher(monkeypatch, lambda text: (
        {FIND_BOOK: 0.45, AUTHOR: 0.3, BY_TITLE: 0.25, "none": 0.0} if text == JOKE else {BY_TITLE: 0.9, "none": 0.0}))
    _split_into(monkeypatch, [JOKE, emma, emma + " again"])
    gated.ctx.process_turn(f"{JOKE}; {emma}; {emma} again")

    review = gated.ctx.process_turn("cancel")

    assert review.status is TurnStatus.AWAITING_USER
    record = gated.store.active()
    assert record is not None and [item["goal"] for item in record.binding["items"]] == [emma, emma + " again"]


def _on_the_command_line(ctx):
    """The context as ``fastworkflow run_cli`` holds it: the core of a real ``ChatSession``, with that session's message,
    output and trace queues (the CLI transport, where the agent's own ``ask_user`` blocks on the message queue)."""
    from fastworkflow.engine.chat_session import ChatSession

    session = ChatSession(run_as_agent=True)
    session._core = ctx
    ctx.set_transport_queues(user_message_queue=session.user_message_queue,
                             command_output_queue=session.command_output_queue,
                             command_trace_queue=session.command_trace_queue)
    return session


def test_on_the_command_line_the_gate_asks_which_command_and_cancel_drops_the_part_as_on_the_chat_page(
        gated, monkeypatch):
    """bd fix-ftdl.7.7: the terminal's session has a message queue, and the gate's questions were not put when one was
    set, so an ambiguous part was planned by the agent there while the chat page asked about it. The gate's questions
    suspend the turn like the review does, and the next message is the reply on both."""
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    emma = "who wrote Emma and how many copies do we have"
    script_matcher(monkeypatch, lambda text: (
        {FIND_BOOK: 0.45, AUTHOR: 0.3, BY_TITLE: 0.25, "none": 0.0} if text == JOKE else {BY_TITLE: 0.9, "none": 0.0}))
    _split_into(monkeypatch, [JOKE, emma])
    session = _on_the_command_line(gated.ctx)

    asked = gated.ctx.process_turn(f"{JOKE}; {emma}")

    assert asked.status is TurnStatus.AWAITING_USER and _kind(asked) == "plan_choice", asked.answer
    assert gated.store.active() is None                    # no review yet
    # //cancel at the terminal answers the question with its kind's cancel reply, through the message queue.
    assert session.cancel_pending_prompt() == "clarification"
    review = gated.ctx.process_turn(session.user_message_queue.get_nowait())
    assert review.status is TurnStatus.AWAITING_USER
    record = gated.store.active()
    assert record is not None and [item["goal"] for item in record.binding["items"]] == [emma]


def _note_part_plan():
    means = review_proposals.Means(kind=review_proposals.MEANS_PLANNING, name=NOTE_PLAN)
    item = review_proposals.SubTask(goal=f"{A_NOTE}: {NOTE}", means=means,
                                    inputs=[review_proposals.SubTaskInput(name="text", value=NOTE, drafted=True)],
                                    drafted_means=NOTE_PLAN)
    return review_proposals.SubTaskProposal(items=[item])


def loops_retry() -> str:
    from fastworkflow.parameters.parameter_clarification import RETRY

    return RETRY


@pytest.mark.usefixtures("no_planner_key")      # the reply reader's remote read is not this test's subject (bd fix-xny2)
def test_with_an_agent_model_a_missing_value_is_asked_before_the_review_when_the_inputs_were_drafted_until_the_rule_stops_it(gated):
    plan = _note_part_plan()
    turn = resolution.Turn(workflow=gated.wf, no_agent=False, can_ask=True, port=waiting.Holder())
    state = resolution.PlanResolution(parts=[{"text": A_NOTE, "outcome": "match", "ambiguous": [], "planning_scores": {},
                                              "state": "open", "chosen": False}])

    plan, question = resolution.step(state, plan, turn)

    held = turn.port.get()
    assert question is not None and "member" in question
    assert held.kind == "parameters" and ParameterLoop.from_state(held.payload["loop"]).open == ["member"]
    # The one asking rule (step 6, consequence b; it used to be asked once): a reply that gives nothing readable is asked
    # once more, and the second one in a row ends it: the part goes on and the run decides (the agent takes over).
    plan, question = resolution.take_reply(state, "", plan, turn)
    assert question is not None and question.startswith(loops_retry()) and turn.port.get() is not None
    plan, question = resolution.take_reply(state, "", plan, turn)
    assert question is None and state.parts[0]["state"] == "resolved" and turn.port.get() is None
    assert [each.name for each in plan.items[0].inputs] == ["text"]


def test_with_an_agent_model_a_readable_reply_is_put_on_the_part_and_asked_once(gated):
    plan = _note_part_plan()
    turn = resolution.Turn(workflow=gated.wf, no_agent=False, can_ask=True, port=waiting.Holder())
    state = resolution.PlanResolution(parts=[{"text": A_NOTE, "outcome": "match", "ambiguous": [], "planning_scores": {},
                                              "state": "open", "chosen": False}])
    resolution.step(state, plan, turn)

    plan, question = resolution.take_reply(state, f"member: {MEMBER}", plan, turn)

    assert question is None and state.parts[0]["state"] == "resolved"
    assert {each.name: each.value for each in plan.items[0].inputs} == {"text": NOTE, "member": MEMBER}


def test_with_an_agent_model_inputs_that_were_not_drafted_are_left_to_the_run(gated):
    plan = _note_part_plan()
    plan.items[0] = plan.items[0].model_copy(update={"drafted_means": ""})
    turn = resolution.Turn(workflow=gated.wf, no_agent=False, can_ask=True, port=waiting.Holder())
    state = resolution.PlanResolution(parts=[{"text": A_NOTE, "outcome": "match", "ambiguous": [], "planning_scores": {},
                                              "state": "open", "chosen": False}])

    _, question = resolution.step(state, plan, turn)

    assert question is None and state.parts[0]["state"] == "resolved"


def test_a_review_pending_at_a_save_is_accepted_after_the_restore_and_runs(no_agent, monkeypatch):
    """A session saved while the turn's review is pending, restored into a fresh context and answered "yes", runs the
    plan (it once raised ``UnboundLocalError`` for ``own_steps`` in the phase runner)."""
    no_agent.matched = script_matcher(monkeypatch, lambda text: {BY_TITLE: 0.45, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]
    asked = no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)
    assert _kind(asked) == "review", asked.answer
    blob = json.loads(json.dumps(no_agent.ctx.serialize_state(channel_id="review-restore")))
    no_agent.ctx._turn_part_facts = None        # a new process holds nothing the save did not carry
    _as_a_new_process(no_agent.ctx, blob)

    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    assert EMMA_RAN in _lookups(no_agent)
    # and the accepted review is still recorded for learning, as it is without a save
    (row,) = _learned(no_agent)
    assert (row["how"], row["confirmed"], row["command"]) == ("review_accept", 1, BY_TITLE)
