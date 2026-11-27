"""One clarification behaviour for a planning command and an execution command (bd fix-fpyb.10).

Written from the SPECIFICATION (``docs/pending_state_design.md`` 3.4, 3.6, 3.7 and 7, and the matching paragraphs of
``AGENTS.md``), before the implementation was read. The same scripted conversation runs on (P) a planning command
(``note_member`` of the library, two required plain values ``member`` and ``text``, no agent model, one part) and (E) an
execution command (``record_values`` of ``plain_required_workflow`` given its ratio and ids, so ``name`` and ``count``
are missing, through the CME). Call the two missing values A and B (P: member, text; E: name, count).

Expectation table (identical on both kinds unless a row says "listed difference"). "asked" means a parameter question is
out, its ``parameter_request`` lists the open values, and ``ctx.waiting`` is a ``parameters`` record.

| # | script | expected, on P and on E |
|---|--------|-------------------------|
| 1 | reply "A: v" | asked again, open == [B]; the progress is not a stall: a following unusable reply is still asked |
| 2 | "umm", then "umm again" | first: asked again, open == [A, B]; second: nothing pending, the outcome says not run and names A and B |
| 3 | "umm", "A: v", a reply that is no value of the lone open field (a bare reply fills it: P "...", refused as punctuation; E "umm", not an int) | asked after each; open == [B] after the last (the count is "in a row") |
| 4 | "cancel", "stop", "never mind" (each on a fresh question) | the question ends at once; nothing pending |
| 5 | "A: v" + "B: w" lines | both filled; a line naming an unlisted field ("colour: red", with two values open) fills nothing and is a no-progress reply; a listed line next to it still fills |
| 6 | "A: cancel" (owner question, no expectation set in the spec: AGENTS.md says a reply is `name: value` lines of the listed fields, and cancel is read when it IS the reply) | reported; asserted here as "taken as the value, B is asked next" |
| 7 | "/A: v" while the question is out | read as the reply "A: v": asked again, open == [B] |
| 8 | the CANCEL event's reply (what `//cancel` sends, 3.4 "Cancel" column) | ends it, nothing pending |
| 9 | "you misunderstood" | LISTED DIFFERENCE (section 7): P (a plan part at the gate) asks which command again, E runs the CME's `you_misunderstood`; both are then a which-command question (`plan_choice`) |
| 9b | a reply that is a different request, no remote model (no key) | the remote read does not exist without a key (AGENTS.md: "none with no key"): an unreadable reply, so the same on both: asked again, nothing run |
| 10 | `process_action` and `use_plan` while the question is out | refused with one text; whether it says how to cancel is reported |
"""
from __future__ import annotations

import pytest

import fastworkflow
from fastworkflow.core import waiting as waiting_module
from fastworkflow.engine.workflow_execution_context import AWAITING_ANSWER_REFUSAL_MESSAGE, UsePlanError
from tests.test_parameter_form_reply_contract import session as _cme_session  # noqa: F401 - fixture
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers
    A_NOTE, HOST, no_agent, annotated_library, todo, gated, initialized_fastworkflow, _offline)

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "no_planner_key")


class Kind:
    """What differs between the kinds: how to start the conversation, and the names and values of the two open fields."""

    def __init__(self, name, ctx, start, a, b, a_value, b_value, junk_for_b):
        self.name, self.ctx, self._start = name, ctx, start
        self.junk_for_b = junk_for_b      # a reply that is no value of B when B is the one value open (see test 1)
        self.a, self.b, self.a_value, self.b_value = a, b, a_value, b_value

    def start(self):
        out = self._start()
        assert self.open_of(out) == [self.a, self.b], out.answer
        return out

    def send(self, text):
        return self.ctx.process_turn(text, approval_surface=HOST) if self.name == "plan" else self.ctx.process_turn(text)

    def open_of(self, out):
        """The values the question asks for, from the turn's last output; None when no question is out."""
        if not self.asked():
            return None
        request = out.command_outputs[-1].command_response.artifacts.get("parameter_request")
        return None if request is None else list(request["missing"]) + list(request["invalid"])

    def asked(self):
        held = self.ctx.waiting
        return held is not None and held.kind == "parameters"

    def line(self, field, value=None):
        return f"{field}: {value if value is not None else (self.a_value if field == self.a else self.b_value)}"


@pytest.fixture(params=["plan", "execution"])
def kind(request):
    if request.param == "plan":
        env = request.getfixturevalue("no_agent")
        env.parts = [A_NOTE]
        return Kind("plan", env.ctx, lambda: env.ctx.process_turn(A_NOTE, approval_surface=HOST),
                    "member", "text", "Zed Quill", "rehung the banner", "...")      # punctuation is refused as a text
    ctx = request.getfixturevalue("_cme_session")(False)
    return Kind("execution", ctx, lambda: ctx.process_turn('/record_values ratio=1.5 ids=["a"]'),
                "name", "count", "Bob", "3", "umm")                       # not an int


def _says_not_run_and_names(out, *names):
    text = out.answer.lower()
    assert "not run" in text, out.answer
    for each in names:
        assert each in text.replace("_", " ") or each in text, (each, out.answer)


# ---- 1, 2, 3: the asking rule -----------------------------------------------------------------------------------------

def test_1_a_reply_that_gives_one_of_two_values_is_asked_again_for_the_other(kind):
    kind.start()
    out = kind.send(kind.line(kind.a))

    assert kind.asked() and kind.open_of(out) == [kind.b]
    # Progress is not a stall: a reply that closes nothing after it is still asked, not stopped. (With ONE value open a
    # bare reply IS the value, so the no-progress reply is a line naming a field the question did not list.)
    again = kind.send(kind.junk_for_b)
    assert kind.asked() and kind.open_of(again) == [kind.b]


def test_2_two_replies_in_a_row_with_no_usable_value_stop_it_and_name_what_is_missing(kind):
    kind.start()
    first = kind.send("umm")
    assert kind.asked() and kind.open_of(first) == [kind.a, kind.b]

    second = kind.send("umm again")

    assert not kind.asked() and kind.ctx.waiting is None
    assert kind.open_of(second) is None
    _says_not_run_and_names(second, kind.a, kind.b)


def test_3_the_count_is_in_a_row_progress_resets_it(kind):
    kind.start()
    assert kind.open_of(kind.send("umm")) == [kind.a, kind.b]
    assert kind.open_of(kind.send(kind.line(kind.a))) == [kind.b]
    last = kind.send(kind.junk_for_b)

    assert kind.asked() and kind.open_of(last) == [kind.b]


# ---- 3b: a request for a moment is free once, then counts (fix-b3w2) ------------------------------------------------

def test_3b_a_third_wait_in_a_row_stops_it_and_a_value_between_waits_resets_the_count(kind, monkeypatch):
    from tests.test_one_reply_reader import _Remote

    def still_open():
        return list(kind.ctx.waiting.payload["loop"]["open"])

    _Remote(monkeypatch, "wait")
    kind.start()
    for _ in range(2):          # the first wait is free, the second counts once: still asked
        kind.send("hold on a moment")
        assert kind.asked() and still_open() == [kind.a, kind.b]
    last = kind.send("hold on a moment")

    assert not kind.asked() and kind.ctx.waiting is None
    _says_not_run_and_names(last, kind.a, kind.b)

    kind.start()
    kind.send("hold on a moment")
    assert kind.asked() and still_open() == [kind.a, kind.b]
    assert kind.open_of(kind.send(kind.line(kind.a))) == [kind.b]
    kind.send("hold on a moment")
    assert kind.asked() and still_open() == [kind.b]


# ---- 4: cancel --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["cancel", "stop", "never mind"])
def test_4_cancel_ends_the_question_at_once(kind, word):
    kind.start()
    out = kind.send(word)

    assert not kind.asked() and kind.ctx.waiting is None
    assert kind.open_of(out) is None


# ---- 5: name: value lines ---------------------------------------------------------------------------------------------

def test_5_a_line_naming_an_unlisted_field_fills_nothing(kind):
    kind.start()
    out = kind.send("colour: red")

    assert kind.asked() and kind.open_of(out) == [kind.a, kind.b]


def test_5b_listed_lines_fill_and_an_unlisted_line_beside_them_is_ignored(kind):
    kind.start()
    out = kind.send(kind.line(kind.a) + "\ncolour: red")
    assert kind.asked() and kind.open_of(out) == [kind.b]


def test_5c_lines_for_both_fields_fill_both(kind):
    kind.start()
    out = kind.send(kind.line(kind.a) + "\n" + kind.line(kind.b))

    assert not kind.asked() or kind.open_of(out) is None or kind.open_of(out) == []


# ---- 6: a field whose wanted value is the word cancel (owner question) -------------------------------------------------

def test_6_name_colon_cancel_is_the_value_not_a_cancel(kind):
    kind.start()
    out = kind.send(kind.line(kind.a, "cancel"))

    # Reported behaviour: taken as the value of A, B is asked next.
    assert kind.asked() and kind.open_of(out) == [kind.b]


# ---- 7: "/" while the question is out ---------------------------------------------------------------------------------

def test_7_a_slash_message_is_the_reply_with_the_slash_removed(kind):
    kind.start()
    out = kind.send("/" + kind.line(kind.a))

    assert kind.asked() and kind.open_of(out) == [kind.b]
    assert not kind.ctx.assistant_round or kind.name == "execution"   # the "/" message began no round of its own


def test_7b_a_slash_cancel_ends_it(kind):
    kind.start()
    kind.send("/cancel")
    assert not kind.asked() and kind.ctx.waiting is None


# ---- 8: //cancel ------------------------------------------------------------------------------------------------------

def test_8_the_session_level_cancel_ends_it(kind):
    kind.start()
    move = waiting_module.step(kind.ctx.waiting, waiting_module.Event(waiting_module.CANCEL))
    assert move.action == waiting_module.ACT_REPLY          # what ChatSession.cancel_pending_prompt puts on the queue

    kind.send(move.text)

    assert not kind.asked() and kind.ctx.waiting is None


# ---- 9: replies that differ between the paths (section 7, listed) -----------------------------------------------------

def test_9_you_misunderstood_is_a_listed_difference_but_both_end_in_a_which_command_question(kind):
    kind.start()
    out = kind.send("you misunderstood")

    # Section 7: the plan gate asks which command again; the CME runs you_misunderstood (its own listing). Either way the
    # user is asked which command is meant (plan_choice) and the value question is no longer the pending one.
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("plan_choice") is not None or artifacts.get("question_kind") == "plan_choice"
    assert kind.ctx.waiting is not None and kind.ctx.waiting.kind == "intent"


def test_9b_a_different_request_with_no_remote_model_is_an_unreadable_reply_on_both(kind):
    kind.start()
    out = kind.send("show me every book in the library")

    assert kind.asked() and kind.open_of(out) == [kind.a, kind.b]


# ---- 10: process_action / use_plan while the question is out ----------------------------------------------------------

def test_10_process_action_and_use_plan_are_refused_with_one_text(kind):
    kind.start()

    refused = kind.ctx.process_action(fastworkflow.Action(command_name="list_books", command="list_books"))
    assert not refused.command_response.success
    assert refused.command_response.response == AWAITING_ANSWER_REFUSAL_MESSAGE
    with pytest.raises(UsePlanError) as error:
        kind.ctx.check_use_plan("anything")
    assert str(error.value) == AWAITING_ANSWER_REFUSAL_MESSAGE
    assert kind.asked()


def test_10b_the_refusal_tells_the_sender_how_to_cancel(kind):
    kind.start()
    assert "cancel" in AWAITING_ANSWER_REFUSAL_MESSAGE.lower()

