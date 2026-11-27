"""The parameter clarification loop (fix-yu3l.4, ``fastworkflow/parameters/parameter_clarification.py``).

One loop asks for the values a planning command still lacks, whoever the caller is: a turn part bound to a planning
command, a plan the client pinned (``use_plan``), a typed planning call that was refused. The reply RESUMES THE PART
THAT ASKED: the turn is not run again from its request, so the parts before it are not run a second time, there is no
second split and no second review, and the matcher is not asked about a part whose command is known.

First the pieces with no turn around them (the reader, the progress rule, the state and the upgrade of an old saved
session, the CME merge by name), then whole turns on the library example, driven as tests/test_refusal_gaps.py and
tests/test_use_plan.py are: the real context on the annotated library copy, no agent model, the planner's split, the
matcher and the shared extractor's model scripted, and every model call or network connection refused.
"""
from __future__ import annotations

import json
import socket

import dspy
from dspy.utils import DummyLM
import pytest
from pydantic import BaseModel, Field

from fastworkflow.review import approval_binding
from fastworkflow.core import no_agent_mode
from fastworkflow.parameters import parameter_clarification as loops
from fastworkflow.planning import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow.review.approval_binding import Surface
from fastworkflow.parameters.parameter_clarification import ParameterLoop
from fastworkflow.core.turn import TurnStatus
from fastworkflow.workflow_agent import build_query_with_next_steps as REAL_BUILD
from fastworkflow.workflow_agent import load_planning_command_map
from fastworkflow.engine.workflow_execution_context import NOT_RUN_ARTIFACT, WorkflowExecutionContext
from tests.scripted_extractor import ScriptedExtractor
from tests.test_approval_binding import _finish, annotated_library, todo  # noqa: F401
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

NOTE_PLAN = "Library/note_member"
BOOK_PLAN = "Library/book_by_title"
HOST = Surface(kind="host", verified=True, uid="user-1")
MEMBER, NOTE = "Zed Quill", "rehung the banner"
ABOUT_EMMA = "who wrote Emma and how many copies do we have"
A_NOTE = "put a note on a record"
ABOUT_DUNE = "who wrote Dune and how many copies do we have"
NAMES = ["member", "text"]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Nothing here can reach a model: litellm's calls and every non-local socket connection raise."""
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


# ======================================================================================================================
# The reader
# ======================================================================================================================

@pytest.mark.parametrize("reply, open_names, confirm, expected", [
    ("cancel", NAMES, False, ("decline", {})),
    ("Never mind.", NAMES, True, ("decline", {})),
    ("yes", NAMES, True, ("accept", {})),                                   # every parameter was shown: yes runs
    ("ok", [], False, ("free", {})),                                        # only a confirmation is accepted by yes (rule a)
    ("yes", ["text"], False, ("values", {"text": "yes"})),                  # one value is open: the word is the value
    ("yes", NAMES, False, ("free", {})),                                    # two open, not a confirmation: not readable
    ("member: Zed Quill", NAMES, False, ("values", {"member": "Zed Quill"})),
    ("MEMBER = Zed Quill\ntext: gone fishing", NAMES, True, ("values", {"member": "Zed Quill", "text": "gone fishing"})),
    ("please set member: Zed Quill", NAMES, False, ("free", {})),           # the name must open the line
    ("colour: red", NAMES, False, ("free", {})),                            # not one of the command's own parameters
    ("Zed Quill", ["member"], False, ("values", {"member": "Zed Quill"})),  # a bare value when exactly one is open
    ("Zed Quill", NAMES, False, ("free", {})),
    ("hmm, I am not sure", NAMES, True, ("values", {"member": "hmm", "text": "I am not sure"})),  # two parts, two open (rule e)
    ("", ["member"], False, ("free", {})),
])
def test_a_reply_is_read_strictly_and_in_code(reply, open_names, confirm, expected):
    assert loops.read_reply(reply, NAMES, open_names, confirm=confirm) == expected


def test_a_comma_separated_reply_fills_the_open_values_in_order():
    assert loops.read_reply("Zed Quill, gone fishing", NAMES, NAMES) == (
        "values", {"member": "Zed Quill", "text": "gone fishing"})
    # The part count must equal the number of open values, and no part may be empty.
    assert loops.read_reply("Zed Quill, gone, fishing", NAMES, NAMES) == ("free", {})
    assert loops.read_reply("Zed Quill, ", NAMES, NAMES) == ("free", {})


def test_the_one_reader_reads_a_pinned_plans_reply():
    assert loops.read_reply("yes", NAMES, NAMES, confirm=True) == ("accept", {})
    assert loops.read_reply("cancel", NAMES, NAMES, confirm=True) == ("decline", {})
    assert loops.read_reply("text: gone fishing", NAMES, NAMES, confirm=True) == ("values", {"text": "gone fishing"})
    assert loops.read_reply("hmm", NAMES, NAMES, confirm=True) == ("free", {})


# ======================================================================================================================
# The loop: the progress rule, on the library's note_member (two required arguments), with no turn around it
# ======================================================================================================================

@pytest.fixture
def note_member(annotated_library):  # noqa: F811
    return load_planning_command_map(str(annotated_library))[NOTE_PLAN]


def _opened(command, request=A_NOTE, **kwargs):
    gaps = pc.argument_gaps(command, {p.name: None for p in command.params}, request)
    return loops.open_loop(command, gaps, request_text=request, **kwargs)


def test_the_first_question_names_every_missing_value_and_the_state_holds_them(note_member):
    loop, asked = _opened(note_member, origin=1)

    assert asked.kind == loops.ASK_AGAIN and "does not give the member and text" in asked.question
    assert asked.request == {"command": NOTE_PLAN, "missing": ["member", "text"], "invalid": [], "values": {}}
    assert (loop.command, loop.kind, loop.open, loop.asks, loop.stalled, loop.origin) == (
        NOTE_PLAN, "planning", ["member", "text"], 1, 0, 1)
    assert loop.question == asked.question and loop.request == asked.request
    assert "policy" not in loop.to_state() and "pending" not in loop.to_state()    # one asking rule: no policy, no flag


def test_a_reply_that_closes_one_value_keeps_it_and_asks_for_the_other(note_member):
    loop, _ = _opened(note_member)
    again = loops.advance(loop, f"member: {MEMBER}", note_member)

    assert again.kind == loops.ASK_AGAIN and again.question.endswith("Which text should I use? " + no_agent_mode.ANSWER_ONLY_NOTE)
    assert again.request["missing"] == ["text"] and again.request["values"] == {"member": MEMBER}
    assert loop.values == {"member": MEMBER} and loop.open == ["text"] and (loop.asks, loop.stalled) == (2, 0)
    done = loops.advance(loop, NOTE, note_member)                # one value is open: the bare reply is that value
    assert done.kind == loops.COMPLETE and loop.open == [] and loop.values == {"member": MEMBER, "text": NOTE}


def test_a_reply_that_closes_nothing_is_asked_once_more_then_the_loop_ends_naming_exactly_what_is_missing(note_member):
    loop, _ = _opened(note_member)
    again = loops.advance(loop, "hmm I am not sure", note_member)

    assert again.kind == loops.ASK_AGAIN and again.question.startswith(loops.RETRY) and loop.stalled == 1
    ended = loops.advance(loop, "still not sure", note_member)
    assert ended.kind == loops.EXHAUSTED and ended.missing == ("member", "text")
    assert ended.describe == "missing required arguments member, text"


def test_progress_starts_the_count_again_and_the_reason_never_names_a_value_that_was_given(note_member):
    loop, _ = _opened(note_member)
    loops.advance(loop, "hmm well", note_member)
    assert loop.stalled == 1
    loops.advance(loop, f"member: {MEMBER}", note_member)
    assert loop.stalled == 0 and loop.open == ["text"]
    loops.advance(loop, "", note_member)
    ended = loops.advance(loop, "", note_member)
    assert ended.kind == loops.EXHAUSTED and ended.missing == ("text",)
    assert ended.describe == "missing required argument text"


def test_cancel_declines_and_names_what_was_still_open(note_member):
    loop, _ = _opened(note_member)
    loops.advance(loop, f"member: {MEMBER}", note_member)

    declined = loops.advance(loop, "cancel", note_member)
    assert declined.kind == loops.DECLINED and declined.missing == ("text",)


def test_free_text_is_given_to_the_callers_extractor_and_only_what_was_open_is_taken(note_member):
    loop, _ = _opened(note_member)
    read = []

    def extract(text):
        read.append(text)
        return {"member": MEMBER, "text": "NOT_FOUND"}

    again = loops.advance(loop, f"it is for {MEMBER} I think", note_member, extract=extract)

    # The extractor read the request with the reply after it; the value it found closed one open value.
    assert read == [f"{A_NOTE}\nit is for {MEMBER} I think"]
    assert again.kind == loops.ASK_AGAIN and not again.question.startswith(loops.RETRY)
    assert loop.values == {"member": MEMBER} and loop.open == ["text"] and loop.stalled == 0


def test_an_extractor_that_fails_reads_nothing(note_member):
    loop, _ = _opened(note_member)

    def extract(text):
        raise RuntimeError("the extractor's model is away")

    again = loops.advance(loop, "it is for someone", note_member, extract=extract)
    assert again.kind == loops.ASK_AGAIN and again.question.startswith(loops.RETRY) and loop.stalled == 1


def test_a_confirmation_shows_every_parameter_and_is_strict(note_member):
    request = f"{A_NOTE} for {MEMBER}"
    gaps = pc.argument_gaps(note_member, {"member": MEMBER, "text": None}, request)
    loop, asked = loops.open_loop(note_member, gaps, request_text=request, confirm=True,
                                  description="Add a note to a member's record.", values={"member": MEMBER})

    assert asked.question.startswith(f"You chose the planning command {NOTE_PLAN}: Add a note to a member's record.")
    assert f"- member (required): {MEMBER}" in asked.question and "- text (required): not given" in asked.question
    assert asked.request["confirm"] is True and asked.request["description"] == "Add a note to a member's record."
    assert [(p["name"], p["value"], p["state"]) for p in asked.request["parameters"]] == [
        ("member", MEMBER, None), ("text", None, "missing")]
    # "yes" while a value is missing runs nothing, and free text is not handed to an extractor.
    again = loops.advance(loop, "yes", note_member, extract=lambda text: pytest.fail("a confirmation is strict"))
    assert again.kind == loops.ASK_AGAIN and again.question.startswith(loops.RETRY) and loop.stalled == 1
    done = loops.advance(loop, f"text: {NOTE}", note_member)
    assert done.kind == loops.COMPLETE and loop.given_values() == {"text": NOTE}
    # With every value given, yes runs with the values shown.
    full = pc.argument_gaps(note_member, {"member": MEMBER, "text": NOTE}, f"{request}: {NOTE}")
    shown, _ = loops.open_loop(note_member, full, request_text=f"{request}: {NOTE}", confirm=True,
                               values={"member": MEMBER, "text": NOTE})
    assert shown.open == [] and loops.advance(shown, "yes", note_member).kind == loops.COMPLETE


def test_a_reply_that_reads_as_nothing_is_no_progress_in_every_session_there_is_no_once_policy(note_member):
    """Step 6: the "once" policy (any reply kept as text for the part's own call, ``complete``) is gone. The same loop,
    given a reply nothing reads, asks again and counts it."""
    loop, _ = _opened(note_member)

    again = loops.advance(loop, "It is for Zed Quill.", note_member)
    assert again.kind == loops.ASK_AGAIN and loop.open == ["member", "text"] and (loop.asks, loop.stalled) == (2, 1)


def test_a_value_an_earlier_part_produces_is_never_open(note_member):
    """``not_asked``: the member comes from an earlier part of the turn, so only the text is the user's to give."""
    gaps = pc.argument_gaps(note_member, {"member": None, "text": None}, A_NOTE).from_earlier({"member"})
    assert gaps.names == ("text",)
    assert gaps.state_of("member") == (loops.STATE_EARLIER, "") and gaps.state_of("text") == (loops.STATE_MISSING, "")
    loop, asked = loops.open_loop(note_member, gaps, request_text=A_NOTE)
    assert asked.request["missing"] == ["text"] and asked.question.endswith("Which text should I use? " + no_agent_mode.ANSWER_ONLY_NOTE)

    done = loops.advance(loop, NOTE, note_member, not_asked={"member"})
    assert done.kind == loops.COMPLETE and loop.open == [] and loop.given == ["text"]


def test_a_readable_reply_that_leaves_a_value_open_asks_again_there_is_no_fallback_that_runs_the_part(note_member):
    loop, _ = _opened(note_member, confirm=True)

    again = loops.advance(loop, f"member: {MEMBER}", note_member)
    assert again.kind == loops.ASK_AGAIN and loop.open == ["text"] and loop.stalled == 0


def test_checks_that_still_refuse_after_the_reply_are_measured_against_the_last_question(note_member):
    """``reopen``: the loop said complete, the part ran its own (stricter) checks and they still refuse something."""
    loop, _ = _opened(note_member)
    assert loops.advance(loop, f"member: {MEMBER}\ntext: {NOTE}", note_member).kind == loops.COMPLETE
    still = loops.Extraction(missing=("text",), values={"member": MEMBER})

    again = loops.reopen(loop, note_member, still)             # member was closed: progress, a plain question
    assert again.kind == loops.ASK_AGAIN and not again.question.startswith(loops.RETRY) and loop.open == ["text"]
    assert loops.advance(loop, f"text: {NOTE}", note_member).kind == loops.COMPLETE
    again = loops.reopen(loop, note_member, still)             # nothing closed: asked once more
    assert again.kind == loops.ASK_AGAIN and again.question.startswith(loops.RETRY) and loop.stalled == 1
    assert loops.advance(loop, f"text: {NOTE}", note_member).kind == loops.COMPLETE
    assert loops.reopen(loop, note_member, still).kind == loops.EXHAUSTED


# ======================================================================================================================
# The state: plain data, and the upgrade of a session saved before the loop existed
# ======================================================================================================================

def test_the_loop_is_plain_data_and_survives_json(note_member):
    loop, _ = _opened(note_member, origin=1)
    loops.advance(loop, f"member: {MEMBER}", note_member)

    state = json.loads(json.dumps(loop.to_state()))
    assert ParameterLoop.from_state(state) == loop
    assert ParameterLoop.from_state(None) is None and ParameterLoop.from_state({}) is None
    # A key an earlier build did not write takes its default; a key this build does not know is an error (old saved
    # sessions are not supported).
    assert ParameterLoop.from_state({"command": NOTE_PLAN}) == ParameterLoop(command=NOTE_PLAN)
    with pytest.raises(TypeError):
        ParameterLoop.from_state({"command": NOTE_PLAN, "later_key": 1})


OLD_PINNED = {
    "skill_refusal_pending": True, "skill_refusal_answered": False, "use_plan": NOTE_PLAN,
    "refusal_progress": {"command": NOTE_PLAN, "open": ["text"], "stalled": 1, "asks": 3, "base": A_NOTE,
                         "given": {"member": MEMBER}, "question": "You chose ...", "params": ["member", "text"]},
    "pending_refusal_request": {"command": NOTE_PLAN, "confirm": True, "missing": ["text"], "invalid": [],
                                "values": {"member": MEMBER}},
    "suspended_user_message": f"{A_NOTE}\nmember: {MEMBER}", "subtasks": [],
}


def test_the_keys_of_an_old_saved_question_are_not_a_loop():
    """Old saved sessions are not supported (bd fix-o1lg): the keys a state carried before the loop existed are no
    loop, and nothing converts them."""
    assert not hasattr(loops, "loop_from_old_state")
    assert ParameterLoop.from_state(OLD_PINNED.get("parameter_loop")) is None


# ======================================================================================================================
# The CME stage: a reply written as name: value is merged by name
# ======================================================================================================================

class _TitleAndAuthor(BaseModel):
    title: str = Field(description="the book's title")
    author: str = Field(description="who wrote it")


def _merged(stored, reply):
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction

    merged = ParameterExtraction._extract_and_merge_missing_parameters(stored, reply)
    return merged.title, merged.author


def test_the_cme_merge_reads_a_name_value_reply_by_name():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import NOT_FOUND

    both = _TitleAndAuthor.model_construct(title=NOT_FOUND, author=NOT_FOUND)
    # Stored whole before: "title: Dune" became the title.
    assert _merged(both, "title: Dune") == ("Dune", NOT_FOUND)
    assert _merged(both, "author=Frank Herbert") == (NOT_FOUND, "Frank Herbert")
    assert _merged(both, "title: Dune\nauthor: Frank Herbert") == ("Dune", "Frank Herbert")
    assert _merged(both, "author: Frank Herbert, title: Dune") == ("Dune", "Frank Herbert")
    # A value the user changes on the way is taken too, as long as the reply also fills something that was asked.
    one = _TitleAndAuthor.model_construct(title="Emma", author=NOT_FOUND)
    assert _merged(one, "title: Dune\nauthor: Frank Herbert") == ("Dune", "Frank Herbert")


def test_every_other_reply_takes_the_positional_merge_as_before():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import NOT_FOUND

    both = _TitleAndAuthor.model_construct(title=NOT_FOUND, author=NOT_FOUND)
    assert _merged(both, "Dune, Frank Herbert") == ("Dune", "Frank Herbert")
    # One part for two open values is unclear (rule e): nothing is filled and the question is asked again.
    assert _merged(both, "Dune") == (NOT_FOUND, NOT_FOUND)
    assert _merged(both, "Dune: Part Two") == (NOT_FOUND, NOT_FOUND)
    assert _merged(both, "the title: Dune") == (NOT_FOUND, NOT_FOUND)
    # With one value open the whole reply is it, a colon in it included (rule d).
    one = _TitleAndAuthor.model_construct(title="Emma", author=NOT_FOUND)
    assert _merged(one, "Dune: Part Two") == ("Emma", "Dune: Part Two")
    assert _merged(one, "the title: Dune") == ("Emma", "the title: Dune")
    # A named line sets a field the question showed (rule b): the title was shown with its value.
    assert _merged(one, "title: Dune") == ("Dune", NOT_FOUND)


# ======================================================================================================================
# Whole turns, no agent model: the reply resumes the part that asked
# ======================================================================================================================

class _ReadsTheText(ScriptedExtractor):
    """The shared extractor's model: a value is found when the text it reads writes it, else it is NOT_FOUND."""

    def forward(self, prompt=None, messages=None, **kwargs):
        text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
        title = "Emma" if ABOUT_EMMA in text else ("Dune" if ABOUT_DUNE in text else "NOT_FOUND")
        self.answers = iter([{"reasoning": "scripted", "title": title,
                              "member": MEMBER if MEMBER in text else "NOT_FOUND",
                              "text": NOTE if NOTE in text else "NOT_FOUND"}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def no_agent(gated, no_agent_model, no_planner_key, monkeypatch):  # noqa: F811
    """The library context with no agent model, drafting on, and the model seams scripted. ``splits`` and ``matched``
    record every call of the planner's split and of the matcher."""
    from fastworkflow.parameters import signatures

    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "on")
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context",
                        lambda self: pytest.fail("the agent LM was built in no-agent mode"))
    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps",
                        lambda user_query, session, **kwargs: setattr(session, "_initial_plan_text", "") or user_query)
    lm = _ReadsTheText([])
    original = signatures.dspy_utils.get_lm
    monkeypatch.setattr(signatures.dspy_utils, "get_lm",
                        lambda name, *a, **k: lm if name == "LLM_PARAM_EXTRACTION" else original(name, *a, **k))
    gated.splits, gated.matched, gated.parts = [], [], [ABOUT_EMMA, A_NOTE, ABOUT_DUNE]

    def split(user_query, **kwargs):
        gated.splits.append(user_query)
        return list(gated.parts)

    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", split)
    def match(text):
        name = NOTE_PLAN if text == A_NOTE else (BOOK_PLAN if text in (ABOUT_EMMA, ABOUT_DUNE) else None)
        return {name: 0.9, "none": 0.0} if name else None

    gated.matched = script_matcher(monkeypatch, match)
    gated.script.steps = []
    return gated


def _reply(env, message, **kwargs):
    if getattr(env, "token", None) is None:
        delivery = approval_binding.take_delivery(env.wf)
        env.token = delivery["token"] if delivery else None
    return env.ctx.process_turn(message, approval_surface=HOST, approval_token=env.token, **kwargs)


def _request(out) -> dict:
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "planning_parameter", artifacts
    return artifacts["parameter_request"]


def _runs(env, name: str) -> list:
    """The runs of command *name* in the turn so far (its outputs), in order."""
    return [output for output in env.ctx._turn_outputs if output.command_name == name]


def _lookups(env) -> list[tuple[str, str]]:
    """What the turn's find_book and find_member runs looked for, in order: ``(command, query)``. book_by_title runs
    find_book once and note_member runs find_member once, so these count the runs of each part."""
    found = []
    for output in env.ctx._turn_outputs:
        name = str(output.command_name).rpartition("/")[2]
        if name in ("find_book", "find_member"):
            parameters = output.command_parameters
            found.append((name, str(parameters.get("query") if isinstance(parameters, dict)
                                    else getattr(parameters, "query", None))))
    return found


EMMA_RAN = ("find_book", "Emma")
NOTE_RAN = ("find_member", MEMBER)
DUNE_RAN = ("find_book", "Dune")


def _loop(ctx):
    """The parameter question's loop: the one in the turn's waiting record, or None when no question is out."""
    from fastworkflow.parameters.parameter_clarification import ParameterLoop

    held = ctx._waiting
    return ParameterLoop.from_state(held.payload["loop"]) if held is not None else None


def _asked_at_part_two(env):
    """A turn of three parts: every part is resolved before the review, so part 2 (note_member, no member and no text
    in the request) asks for its values first, with nothing run and no review open."""
    asked = env.ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}", approval_surface=HOST)
    assert asked.status is TurnStatus.AWAITING_USER, asked.answer
    assert "review" not in asked.command_outputs[-1].command_response.artifacts
    assert _request(asked)["missing"] == ["member", "text"]
    return asked


def test_the_reply_completes_the_part_that_asked_then_one_review_then_every_part_runs_once(no_agent):
    asked = _asked_at_part_two(no_agent)
    ctx = no_agent.ctx

    # Nothing has run: every part is resolved before the review, and the question is the first thing the turn says.
    assert not _runs(no_agent, BOOK_PLAN) and _lookups(no_agent) == []
    loop = _loop(ctx)
    assert (loop.command, loop.origin) == (NOTE_PLAN, 1)
    assert "does not give the member and text" in asked.answer
    splits, matched = len(no_agent.splits), len(no_agent.matched)

    review = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")

    # The values are in, nothing is missing, so ONE review of the whole turn opens; still nothing has run.
    assert review.status is TurnStatus.AWAITING_USER, review.answer
    artifacts = review.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") != "planning_parameter" and len(artifacts["review"]["items"]) == 3
    assert _lookups(no_agent) == [] and ctx._plan_resolution is None
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED, done.answer
    # Every part ran once, in order, with the values the user gave.
    assert _lookups(no_agent) == [EMMA_RAN, NOTE_RAN, DUNE_RAN] and len(_runs(no_agent, BOOK_PLAN)) == 2
    bindings = {record["subtask"]: record for record in ctx._turn_subtask_bindings if record["ran"] is not None}
    assert bindings[2]["ran"] == {"member": MEMBER, "text": NOTE} and bindings[3]["ran"] == {"title": "Dune"}
    # No second split, and the matcher was not asked again.
    assert len(no_agent.splits) == splits == 1 and len(no_agent.matched) == matched
    assert ctx._turn_subtask_review == "reviewed"
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2.", "### 3."]
    assert ctx._plan_resolution is None


def test_a_part_that_is_declined_is_dropped_and_the_review_and_the_parts_after_it_go_on(no_agent):
    _asked_at_part_two(no_agent)

    review = _reply(no_agent, "cancel")

    # Cancelling drops that part only: the review shows the other two, and the note is not in it.
    assert review.status is TurnStatus.AWAITING_USER, review.answer
    items = review.command_outputs[-1].command_response.artifacts["review"]["items"]
    assert [item["goal"] for item in items] == [ABOUT_EMMA, ABOUT_DUNE]
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED
    stops = [stop for stop in done.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]
             if stop["command"] == NOTE_PLAN]
    assert len(stops) == 1 and stops[0]["request"] == A_NOTE
    assert stops[0]["reason"] == "it was declined while its arguments member, text were still missing"
    assert _lookups(no_agent) == [EMMA_RAN, DUNE_RAN]


def test_a_reply_with_no_progress_is_asked_once_more_then_the_part_is_not_run_and_the_turn_carries_on(no_agent):
    _asked_at_part_two(no_agent)

    again = _reply(no_agent, "hmm I am not sure")
    assert again.status is TurnStatus.AWAITING_USER and again.answer.startswith("That did not give a value I could use.")
    assert _request(again)["missing"] == ["member", "text"] and _lookups(no_agent) == []
    review = _reply(no_agent, "really I am not sure")
    assert review.status is TurnStatus.AWAITING_USER and _lookups(no_agent) == []   # the part is dropped, the review of the rest opens
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED
    (stop,) = [stop for stop in done.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]
               if stop["command"] == NOTE_PLAN]
    assert "missing required arguments member, text" in stop["reason"] and stop["request"] == A_NOTE
    assert _lookups(no_agent) == [EMMA_RAN, DUNE_RAN]


def _as_a_new_process(ctx, blob: dict) -> None:
    """What is live is dropped and only what the saved state holds comes back (as tests/test_no_agent_decisions.py
    restores a paused step)."""
    ctx._clear_live_pending()
    ctx._turn_outputs, ctx._turn_key, ctx._turn_use_plan = [], None, None
    ctx._turn_user_message = ctx._turn_refined_message = None
    assert ctx._plan_resolution is None and not ctx._subtasks and ctx._turn_subtask_plan is None
    ctx.apply_serialized_state(blob)


def test_the_question_survives_a_save_and_the_reply_resumes_the_same_part_after_the_restore(no_agent):
    asked = _asked_at_part_two(no_agent)
    ctx = no_agent.ctx
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="loop-restore")))

    state, saved = blob["plan_resolution"], blob["waiting"]
    assert saved["kind"] == "parameters" and saved["payload"]["after"] == "resolve_next" and state["index"] == 1
    loop = saved["payload"]["loop"]
    assert loop["command"] == NOTE_PLAN and "pending" not in loop and loop["origin"] == 1
    assert not {"parameter_loop", "plan_choice", "skill_refusal_pending", "skill_refusal_answered", "refusal_progress",
                "pending_refusal_request"} & set(blob)
    _as_a_new_process(ctx, blob)

    assert ctx._awaiting_user is True and ctx._resolution_pending is True
    assert ctx.waiting.question == asked.answer
    assert ctx._pending_prompt_kind(ctx._turn_outputs[-1])[0] == "planning_parameter"
    splits = len(no_agent.splits)
    review = _reply(no_agent, f"member: {MEMBER}\ntext: {NOTE}")
    assert review.status is TurnStatus.AWAITING_USER and _lookups(no_agent) == []
    done = _reply(no_agent, "yes")

    assert done.status is TurnStatus.COMPLETED
    assert _lookups(no_agent) == [EMMA_RAN, NOTE_RAN, DUNE_RAN]
    assert len(no_agent.splits) == splits == 1
    assert [line[:6] for line in done.answer.splitlines() if line.startswith("### ")] == ["### 1.", "### 2.", "### 3."]
    ran = [record for record in ctx._turn_subtask_bindings if record["subtask"] == 2 and record["ran"] is not None]
    assert ran and ran[-1]["ran"] == {"member": MEMBER, "text": NOTE}


def test_a_session_saved_before_the_loop_existed_is_ignored(no_agent):
    """A hand-built old blob: the pending question of a pinned plan as an earlier build saved it (the flags, the
    progress record, the request data, a ``parameter_loop`` and a ``plan_choice`` key), with no ``plan_resolution`` key.
    Old saved sessions are not supported (bd fix-o1lg): those keys are ignored, so no question is pending after the
    restore."""
    ctx = no_agent.ctx
    first = ctx.process_turn(A_NOTE, use_plan=NOTE_PLAN, approval_surface=HOST)
    assert first.status is TurnStatus.AWAITING_USER
    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="old-blob")))
    blob.pop("plan_resolution")
    new = blob.pop("waiting")["payload"]["loop"]
    blob.update({
        "skill_refusal_pending": True, "skill_refusal_answered": False, "use_plan": NOTE_PLAN,
        "refusal_progress": {"command": NOTE_PLAN, "open": ["member", "text"], "stalled": 0, "asks": 1, "base": A_NOTE,
                             "given": {}, "question": first.answer, "params": ["member", "text"]},
        "pending_refusal_request": dict(new["request"]),
        "parameter_loop": dict(new), "plan_choice": {"origin": 0, "choices": [{"name": "x"}], "pending": True},
    })
    _as_a_new_process(ctx, blob)

    assert ctx._plan_resolution is None and ctx._resolution_pending is False
    assert not {"refusal_progress", "skill_refusal_pending", "pending_refusal_request", "parameter_loop",
                "plan_choice"} & set(ctx.serialize_state(channel_id="old-blob"))


def test_a_value_bound_to_an_earlier_part_is_never_asked_for(no_agent, monkeypatch):
    """The note's member is declared to come from part 1's output ("from sub-task 1", what a field's
    ``from_command_output`` gives): the question asks for the text only, and nobody is asked for the member."""
    from fastworkflow.review import review_proposals

    real = review_proposals._producers
    monkeypatch.setattr(review_proposals, "_producers",
                        lambda workflow, means, earlier: {"member": 1} if means.name == NOTE_PLAN and earlier
                        else real(workflow, means, earlier))
    no_agent.parts = [ABOUT_EMMA, A_NOTE]

    asked = no_agent.ctx.process_turn(f"{ABOUT_EMMA}; {A_NOTE}", approval_surface=HOST)

    assert asked.status is TurnStatus.AWAITING_USER
    request = _request(asked)
    assert request["missing"] == ["text"] and "member" not in request["invalid"]
    assert "Which text should I use?" in asked.answer and "member" not in asked.answer.split("yet:")[1]
    assert _loop(no_agent.ctx).open == ["text"]
    review = _reply(no_agent, NOTE)                             # one value is open: the bare reply is that value
    assert review.status is TurnStatus.AWAITING_USER
    item = review.command_outputs[-1].command_response.artifacts["review"]["items"][1]
    assert {"name": "member", "from_subtask": 1} == {k: item["inputs"][0][k] for k in ("name", "from_subtask")}
    assert any(each["name"] == "text" and each["value"] == NOTE for each in item["inputs"])
    done = _reply(no_agent, "yes")
    assert done.status is TurnStatus.COMPLETED
    # What part 1 gave is not a member the checks take (it is a book). The part then ends as not run naming the
    # member, as a part whose earlier part did not produce its value does; it is not asked for, then or now.
    (stop,) = [stop for stop in done.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT, [])
               if stop["command"] == NOTE_PLAN and stop["open_step"] is None]
    assert "member" in stop["reason"] and "text" not in stop["reason"].split("(")[1].split(")")[0]
    assert no_agent.ctx._turn_use_plan is None


# ======================================================================================================================
# A whole turn with an agent model: one question, the reply resumes the part, and no second question
# ======================================================================================================================

class _TextPlanner(DummyLM):
    """The planner's model after the refusal that is not asked about: a one-line text plan."""

    def forward(self, prompt=None, messages=None, **kwargs):
        self.answers = iter([{"reasoning": "scripted", "next_steps": "1. find the book"}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def test_with_an_agent_model_the_question_is_asked_again_while_replies_make_progress_then_the_agent_takes_over(todo, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    request = "who wrote that book"
    command = load_planning_command_map(todo.wf.folderpath)[BOOK_PLAN]
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {p.name: None for p in command.params}, request)
    typed_calls, splits, matched = [], [], []

    def refuse(planning_command, request_text, *args, **kwargs):
        typed_calls.append(request_text)
        raise refused.value

    def split(user_query, **kwargs):
        splits.append(user_query)
        return [user_query]

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda *args, **kwargs: None)
    matched = script_matcher(monkeypatch, lambda text: {command.name: 0.9, "none": 0.0})
    monkeypatch.setattr(workflow_agent, "typed_planning_call", refuse)
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", split)
    planner = _TextPlanner([], adapter=dspy.JSONAdapter())
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda *args, **kwargs: REAL_BUILD(*args, **{**kwargs, "planner_lm": planner}))
    ctx = todo.ctx

    first = ctx.process_turn(request)

    assert first.status is TurnStatus.AWAITING_USER and first.answer.endswith("Which title should I use? " + no_agent_mode.ANSWER_ONLY_NOTE)
    loop = _loop(ctx)
    assert (loop.command, loop.asks, loop.origin) == (BOOK_PLAN, 1, 0)
    assert ctx._waiting.payload["after"] == "resume_part"
    assert typed_calls == [request] and len(splits) == 1 and len(matched) == 1
    todo.script.steps = _finish()

    again = ctx.process_turn("title: Dune")

    # Step 6 (consequence b): the reply filled the listed value, the part's typed call ran again on the request with the
    # reply after it and is refused for it again (the scripted refusal), so the question is asked a second time (it used
    # to go straight to the text planner and the agent).
    assert again.status is TurnStatus.AWAITING_USER and ctx._waiting is not None
    assert typed_calls == [request, f"{request}\ntitle: Dune"] and _loop(ctx).asks == 2

    done = ctx.process_turn("title: Dune")

    # The second reply closed nothing the question listed: the rule stops (two in a row), and this time the text planner
    # and the agent take over, with no third question.
    assert done.status is TurnStatus.COMPLETED
    assert typed_calls[:2] == [request, f"{request}\ntitle: Dune"] and len(typed_calls) >= 3
    assert ctx._waiting is None and ctx._resolution_pending is False
    # The turn was not run again from the top: no second split, and the matcher was not asked about the part again.
    assert len(splits) == 1 and len(matched) == 1
