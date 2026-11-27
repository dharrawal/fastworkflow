"""ONE reader for a reply to a value question (bd fix-fpyb.8, step 7 of ``docs/pending_state_design.md``).

A reply to an execution command's value question (the deterministic path) and to a plan part's value question (at the
gate before the review, and while a part runs) is read by ``parameters.reply.read_reply`` and treated the same way:
values, ``name: value`` lines, a request for a moment (the question stays, nothing is counted), "you misunderstood"
(the which-command question again on the plan path, the deterministic path's own listing there), a different request
(the deterministic path routes it; a plan part is set aside and the reply is NOT run as a request), ``cancel`` and
``abort`` (the same on both paths) and a reply that cannot be read. The remote model's read costs one model call: at
most one per reply, never for ``name: value`` lines or a fixed phrase, never with no key.

Real contexts on the real test workflows (the hello_world assistant session, the annotated library copy); the remote
decision model's read is scripted at the ``dspy.Predict`` seam, as ``tests/test_jev_missing_value_reply.py`` does. No
live model calls.
"""
from __future__ import annotations

import dspy
import pytest

import fastworkflow
from dspy.utils import DummyLM
from types import SimpleNamespace

from fastworkflow import intent_loop, no_agent_mode, parameter_clarification, waiting, workflow_agent
from fastworkflow._workflows.command_metadata_extraction import parameter_reply
from fastworkflow.parameters import read, reply as replies
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT
from tests.conftest import script_matcher
from tests.test_approval_binding import _finish, annotated_library, todo  # noqa: F401 - fixtures
from tests.test_jev_missing_value_reply import (  # noqa: F401 - fixtures and helpers of the execution command's side
    PENDING_COMMAND, _app_workflow, _assistant_session, _cme_state, _framework, _pending, _switch_on)
from tests.test_parameter_loop import (  # noqa: F401 - fixtures and helpers of the plan side
    A_NOTE, ABOUT_DUNE, ABOUT_EMMA, BOOK_PLAN, MEMBER, NOTE, _loop, _offline, _reply, _request, no_agent)
from tests.test_plan_waiting_record import _refused_book_part
from tests.test_round1_gate import gated  # noqa: F401 - fixture
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - fixture
from tests.test_waiting_kinds import _last

DECLINED = "it was declined"
PICKED = "Book/show_book"      # an execution command the which-command question lists for the todo workflow


class _Remote:
    """The remote decision model's read of a reply, scripted at ``dspy.Predict``: every other ``Predict`` is the real
    one. ``answers`` is the word it gives (one for every call); ``calls`` records ``(question, reply)``. With no
    ``answers`` (the no-key cases) a call is a failure."""

    def __init__(self, monkeypatch, answer: str | None = None):
        self.calls: list[tuple[str, str]] = []
        real = dspy.Predict
        if answer is None:
            monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
        else:
            monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))

        def predict(signature, *args, **kwargs):
            if "intent" not in getattr(signature, "output_fields", {}):
                return real(signature, *args, **kwargs)

            def call(question, reply):
                self.calls.append((question, reply))
                assert answer is not None, "the remote model was called with no key"
                return SimpleNamespace(intent=SimpleNamespace(value=answer, probabilities={answer: 0.95},
                                                              confidence=0.95))
            return call

        monkeypatch.setattr(dspy, "Predict", predict)


def _artifacts(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def _stops(out) -> list[dict]:
    return _artifacts(out).get(NOT_RUN_ARTIFACT, [])


# ======================================================================================================================
# The reader itself: pure, no model stack
# ======================================================================================================================

def _said(answer: str, calls: list):
    def remote(question, reply):
        calls.append((question, reply))
        return answer
    return remote


def test_the_reader_is_importable_without_the_model_stack():
    import subprocess
    import sys

    code = "import sys, fastworkflow.parameters.reply; print(sorted(m for m in ('dspy', 'torch', 'litellm', 'transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize("reply, kind", [("cancel", replies.DECLINE), ("abort", replies.DECLINE),
                                         ("never mind", replies.DECLINE)])
def test_cancel_and_abort_are_the_same_reply_and_the_remote_model_is_not_asked(reply, kind):
    calls: list = []
    reading = replies.read_reply(reply, ["a"], ["a"], remote=_said("wait", calls))
    assert reading.kind == kind and calls == []


def test_name_value_lines_are_values_and_the_remote_model_is_not_asked():
    calls: list = []
    reading = replies.read_reply("a: 1", ["a", "b"], ["a", "b"], remote=_said("different_command", calls))
    assert (reading.kind, reading.values) == (replies.VALUES, {"a": "1"}) and calls == []


def test_a_fixed_phrase_is_read_before_the_values_and_asks_no_model():
    calls: list = []
    reading = replies.read_reply("you misunderstood", ["a"], ["a"], phrases=lambda text: "misunderstood" if "misunderstood" in text else None,
                                 remote=_said("answer", calls))
    assert reading.kind == replies.MISUNDERSTOOD and calls == []


@pytest.mark.usefixtures("initialized_fastworkflow")
@pytest.mark.parametrize("reply", ["none", "None", "nope", "Change", "Wrong", "What", "You", "Terminator", "Never",
                                   "Ex", "Stopwatch", "you misunderstood me about the title"])
def test_a_value_that_only_begins_like_a_fixed_phrase_is_not_that_phrase(reply):
    """bd fix-ftdl.9: the framework's phrases of abort and "you misunderstood" match a WHOLE reply (case and trailing
    punctuation ignored), never its opening letters: "none" and "Change" are values, not "you misunderstood", and
    "Terminator" is a title, not abort."""
    assert replies.phrase_of(reply) is None


@pytest.mark.usefixtures("initialized_fastworkflow")
@pytest.mark.parametrize("reply, phrase", [("you misunderstood", "misunderstood"), ("You misunderstood.", "misunderstood"),
                                           ("Wrong command!", "misunderstood"), ("terminate", "abort"),
                                           ("Never mind", "abort"), ("EXIT", "abort")])
def test_a_whole_fixed_phrase_is_still_that_phrase(reply, phrase):
    assert replies.phrase_of(reply) == phrase


@pytest.mark.usefixtures("initialized_fastworkflow")
def test_none_leaves_an_optional_field_empty_through_the_one_reader():
    """Rule c of ``parameters.read`` ("no / none / neither leave an optional field empty") is reachable: the phrases no
    longer read "none" as "you misunderstood" first."""
    calls: list = []
    reading = replies.read_reply("none", ["note"], ["note"], slots={"note": read.Slot(required=False)},
                                 phrases=replies.phrase_of, remote=_said("answer", calls))
    assert (reading.kind, reading.values) == (replies.VALUES, {"note": None})


@pytest.mark.parametrize("word, kind", [("wait", replies.WAIT), ("abort", replies.DECLINE),
                                        ("misunderstood", replies.MISUNDERSTOOD),
                                        ("different_command", replies.DIFFERENT)])
def test_the_remote_read_is_asked_once_and_its_word_is_the_kind(word, kind):
    calls: list = []
    reading = replies.read_reply("hold on", ["a"], ["a"], question="Which a?", remote=_said(word, calls))
    assert reading.kind == kind and reading.intent == word and calls == [("Which a?", "hold on")]


@pytest.mark.parametrize("reply, slot", [
    ("3", read.Slot(number=True)), ("-2.5", read.Slot(number=True)),
    ("HIGH", read.Slot(options=("high", "low"))), ("no", read.Slot(boolean=True)),
])
def test_a_reply_that_fits_a_constrained_field_is_not_sent_to_the_remote_model(reply, slot):
    """bd fix-iw13: a number, an allowed value or a yes/no for the one open field is read in code."""
    def never(question, text):
        raise AssertionError("the remote model was asked")

    reading = replies.read_reply(reply, ["a"], ["a"], slots={"a": slot}, remote=never)
    assert reading.kind == replies.VALUES and "a" in reading.values


@pytest.mark.parametrize("slot", [read.Slot(number=True), read.Slot(options=("high", "low")), read.Slot()])
def test_a_reply_that_does_not_fit_or_a_free_text_field_keeps_the_remote_read(slot):
    calls: list = []
    reading = replies.read_reply("hold on", ["a"], ["a"], slots={"a": slot}, remote=_said("wait", calls))
    assert len(calls) == 1 and reading.kind == replies.WAIT


def test_an_answer_from_the_remote_read_leaves_the_codes_reading():
    calls: list = []
    reading = replies.read_reply("Dune", ["a"], ["a"], remote=_said("answer", calls))
    assert (reading.kind, reading.values, reading.intent) == (replies.VALUES, {"a": "Dune"}, None) and len(calls) == 1


def test_the_own_command_rule_comes_before_the_remote_read():
    calls: list = []
    reading = replies.read_reply("cmd again", ["a"], ["a"], replaces=lambda text: text.startswith("cmd"),
                                 remote=_said("wait", calls))
    assert reading.kind == replies.REPLACE and calls == []


def test_the_callers_answer_rule_is_read_after_cancel_and_before_the_phrases_and_the_remote_read():
    # Step 8: a decision step's yes or no is its own answer (``read_decision``); a reply the rule does not read goes on.
    calls: list = []

    def rule(text):
        return replies.Reading(replies.VALUES, {"answer": "yes"}) if text == "ok" else None

    assert replies.read_reply("ok", [], [], answers=rule, phrases=lambda text: "abort", remote=_said("wait", calls)).values == {
        "answer": "yes"} and calls == []
    assert replies.read_reply("cancel", [], [], answers=rule).kind == replies.DECLINE
    assert replies.read_reply("hmm", [], [], answers=rule, remote=_said("wait", calls)).kind == replies.WAIT and len(calls) == 1


def test_with_no_remote_function_nothing_is_asked_and_the_code_decides():
    assert replies.read_reply("hmm", ["a", "b"], ["a", "b"]).kind == replies.FREE


# ======================================================================================================================
# An execution command's value question (the deterministic path)
# ======================================================================================================================

def test_exec_values_in_name_lines_are_merged_and_the_remote_model_is_not_asked(monkeypatch):
    remote = _Remote(monkeypatch, "wait")
    workflow = _pending()
    reply = parameter_reply.read_reply(workflow, workflow.context["app_workflow"], "user_name: dana")
    assert reply.action == parameter_reply.ACTION_MERGE and remote.calls == []


@pytest.mark.parametrize("value", ["Terminator", "Change"])
def test_exec_a_value_that_begins_like_a_fixed_phrase_is_a_value(monkeypatch, value):
    """The "/" path reads its reply with the same phrases (bd fix-ftdl.9): a value is merged, never abort or the
    misunderstanding listing."""
    _Remote(monkeypatch, "answer")
    workflow = _pending()
    reply = parameter_reply.read_reply(workflow, workflow.context["app_workflow"], value)
    assert reply.action == parameter_reply.ACTION_MERGE and reply.text == value


def test_exec_a_plain_value_is_asked_of_the_remote_model_once_and_merged(monkeypatch):
    remote = _Remote(monkeypatch, "answer")
    workflow = _pending()
    reply = parameter_reply.read_reply(workflow, workflow.context["app_workflow"], "dana.k@corp.io")
    assert reply.action == parameter_reply.ACTION_MERGE and reply.text == "dana.k@corp.io"
    assert [call[1] for call in remote.calls] == ["dana.k@corp.io"]


def test_exec_a_request_for_a_moment_keeps_the_question_and_counts_nothing(_assistant_session, monkeypatch):
    remote = _Remote(monkeypatch, "wait")
    before = _cme_state(_assistant_session)[1]
    out = _assistant_session.process_turn("hmm, hold on a sec")
    stage, stored = _cme_state(_assistant_session)
    assert "OK, take your time." in str(out.model_dump()) and len(remote.calls) == 1
    assert stage == intent_loop.WAITING_PARAMETERS
    assert (stored.asks, stored.stalled, stored.open) == (before.asks, before.stalled, before.open)


def test_exec_you_misunderstood_is_a_fixed_phrase_read_with_no_model_call(_assistant_session, monkeypatch):
    remote = _Remote(monkeypatch, "answer")
    _assistant_session.process_turn("you misunderstood")
    stage, stored = _cme_state(_assistant_session)
    assert remote.calls == [] and stored is None and stage == intent_loop.WAITING_INTENT


def test_exec_the_remote_read_can_say_it_was_misunderstood(_assistant_session, monkeypatch):
    remote = _Remote(monkeypatch, "misunderstood")
    _assistant_session.process_turn("the wrong command seems to have been picked here")
    stage, stored = _cme_state(_assistant_session)
    assert len(remote.calls) == 1 and stored is None and stage == intent_loop.WAITING_INTENT


def test_exec_a_different_request_is_not_run_and_the_question_is_asked_again(_assistant_session, monkeypatch):
    # As on the plan path (bd fix-ftdl.8.10): a reply read as a different request is not run; it counts as a reply
    # with no progress, so the question stays and says the reply was not run.
    remote = _Remote(monkeypatch, "different_command")
    out = _assistant_session.process_turn("what can i do?")
    stage, stored = _cme_state(_assistant_session)
    assert len(remote.calls) == 1 and stored is not None and stage == intent_loop.WAITING_PARAMETERS
    assert stored.stalled == 1
    assert "read as a new request, so it was not run" in str(out.model_dump())


@pytest.mark.parametrize("word", ["cancel", "abort"])
def test_exec_cancel_and_abort_end_the_command_with_no_model_call(_assistant_session, monkeypatch, word):
    remote = _Remote(monkeypatch, "answer")
    _assistant_session.process_turn(word)
    stage, stored = _cme_state(_assistant_session)
    assert remote.calls == [] and stored is None and stage is None


def test_exec_the_remote_models_abort_ends_the_command(_assistant_session, monkeypatch):
    _Remote(monkeypatch, "abort")
    _assistant_session.process_turn("never mind, forget it")
    assert _cme_state(_assistant_session) == (None, None)


def test_exec_an_unreadable_reply_with_no_key_makes_no_call_and_asks_again(monkeypatch):
    remote = _Remote(monkeypatch, None)
    workflow = _pending()
    reply = parameter_reply.read_reply(workflow, workflow.context["app_workflow"], "")
    assert reply.action == parameter_reply.ACTION_MERGE and remote.calls == []


def test_exec_no_key_means_no_call_for_any_reply(_assistant_session, monkeypatch):
    remote = _Remote(monkeypatch, None)
    _assistant_session.process_turn("hold on a moment")
    assert remote.calls == []


# ======================================================================================================================
# A plan part's value question at the gate (before the review)
# ======================================================================================================================

TURN = f"{ABOUT_EMMA}; {A_NOTE}; {ABOUT_DUNE}"


def _asked_at_the_gate(env):
    asked = env.ctx.process_turn(TURN, approval_surface=__import__("tests.test_parameter_loop", fromlist=["HOST"]).HOST)
    assert _request(asked)["missing"] == ["member", "text"]
    return asked


def _review_items(out) -> list[str]:
    return [item["goal"] for item in _artifacts(out)["review"]["items"]]


def test_gate_name_value_lines_are_values_and_the_remote_model_is_not_asked(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "different_command")
    _asked_at_the_gate(no_agent)
    again = _reply(no_agent, f"member: {MEMBER}")
    assert _request(again)["missing"] == ["text"] and remote.calls == []


def test_gate_a_value_is_asked_of_the_remote_model_once_and_kept(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "answer")
    _asked_at_the_gate(no_agent)
    again = _reply(no_agent, f"{MEMBER}, {NOTE}")
    assert _last(again)["question_kind"] == "review" and len(remote.calls) == 1


def test_gate_a_request_for_a_moment_keeps_the_question_and_counts_nothing(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "wait")
    first = _asked_at_the_gate(no_agent)
    before = _loop(no_agent.ctx)
    waited = _reply(no_agent, "hold on, I am checking")
    assert waited.status is TurnStatus.AWAITING_USER and waited.answer.startswith("OK, take your time.")
    assert first.answer in waited.answer and _request(waited)["missing"] == ["member", "text"]
    after = _loop(no_agent.ctx)
    assert (after.asks, after.stalled, after.open) == (before.asks, before.stalled, before.open) and len(remote.calls) == 1
    # The next real answer is still an answer to the same question.
    _Remote(monkeypatch, "answer")
    assert _request(_reply(no_agent, f"member: {MEMBER}"))["missing"] == ["text"]


def test_gate_you_misunderstood_goes_back_to_the_which_command_question_with_no_model_call(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "answer")
    _asked_at_the_gate(no_agent)
    asked = _reply(no_agent, "you misunderstood")
    choice = _artifacts(asked)["plan_choice"]
    assert _last(asked)["question_kind"] == "plan_choice" and choice["misunderstood"] is True
    assert choice["request"] == A_NOTE, [each["name"] for each in choice["choices"]]
    assert not [each["name"] for each in choice["choices"] if each["name"].startswith("IntentDetection/")]
    assert remote.calls == [] and no_agent.ctx._waiting.kind == waiting.KIND_INTENT
    # Answering it carries the resolution on: cancelling drops the part and the review shows the rest.
    review = _reply(no_agent, "cancel")
    assert _last(review)["question_kind"] == "review" and _review_items(review) == [ABOUT_EMMA, ABOUT_DUNE]


def test_gate_the_remote_read_can_say_it_was_misunderstood(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "misunderstood")
    _asked_at_the_gate(no_agent)
    asked = _reply(no_agent, "the wrong command seems to have been picked here")
    assert _artifacts(asked)["plan_choice"]["misunderstood"] is True and len(remote.calls) == 1


def test_gate_a_different_request_sets_the_part_aside_and_is_not_run_as_a_request(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, "different_command")
    _asked_at_the_gate(no_agent)
    review = _reply(no_agent, "show me the users")
    assert _last(review)["question_kind"] == "review" and _review_items(review) == [ABOUT_EMMA, ABOUT_DUNE]
    assert len(remote.calls) == 1
    done = _reply(no_agent, "yes")
    (stop,) = [stop for stop in _stops(done) if stop["command"] == "Library/note_member"]
    assert "asked for something else" in stop["reason"] and "not run as a new request" in stop["reason"]
    assert "not run as a new request" in no_agent_mode.explanation(stop)     # the entry's own answer text


@pytest.mark.parametrize("word", ["cancel", "abort"])
def test_gate_cancel_and_abort_drop_the_part_with_no_model_call(no_agent, monkeypatch, word):
    remote = _Remote(monkeypatch, "answer")
    _asked_at_the_gate(no_agent)
    review = _reply(no_agent, word)
    assert _review_items(review) == [ABOUT_EMMA, ABOUT_DUNE] and remote.calls == []
    done = _reply(no_agent, "yes")
    (stop,) = [stop for stop in _stops(done) if stop["command"] == "Library/note_member"]
    assert stop["reason"].startswith(DECLINED)


def test_gate_the_remote_models_abort_drops_the_part(no_agent, monkeypatch):
    _Remote(monkeypatch, "abort")
    _asked_at_the_gate(no_agent)
    assert _review_items(_reply(no_agent, "never mind, forget it")) == [ABOUT_EMMA, ABOUT_DUNE]


def test_gate_an_unreadable_reply_with_no_key_makes_no_call_and_is_asked_again(no_agent, monkeypatch):
    remote = _Remote(monkeypatch, None)
    _asked_at_the_gate(no_agent)
    again = _reply(no_agent, "")
    assert _request(again)["missing"] == ["member", "text"] and again.answer.startswith(parameter_clarification.RETRY)
    assert remote.calls == []


# ======================================================================================================================
# The value question a running part puts (a refused typed call of the turn's one part)
# ======================================================================================================================

def _asked_while_running(todo, monkeypatch, answer):
    request = _refused_book_part(todo, monkeypatch)
    first = todo.ctx.process_turn(request)
    assert first.status is TurnStatus.AWAITING_USER and _last(first)["question_kind"] == "planning_parameter"
    return _Remote(monkeypatch, answer), first


def test_part_name_value_lines_are_values_and_the_remote_model_is_not_asked(todo, monkeypatch):
    remote, _ = _asked_while_running(todo, monkeypatch, "different_command")
    todo.script.steps = _finish()
    todo.ctx.process_turn("title: Dune")
    assert remote.calls == []


def test_part_a_request_for_a_moment_keeps_the_question_and_counts_nothing(todo, monkeypatch):
    remote, first = _asked_while_running(todo, monkeypatch, "wait")
    before = parameter_clarification.ParameterLoop.from_state(todo.ctx._waiting.payload["loop"])
    waited = todo.ctx.process_turn("hold on, checking")
    assert waited.status is TurnStatus.AWAITING_USER and waited.answer.startswith("OK, take your time.")
    assert first.answer in waited.answer and len(remote.calls) == 1
    held = parameter_clarification.ParameterLoop.from_state(todo.ctx._waiting.payload["loop"])
    assert (held.asks, held.stalled, held.open) == (before.asks, before.stalled, before.open)
    assert todo.ctx._waiting.question == before.question      # the record's question is still the one the user saw


@pytest.mark.parametrize("phrase_or_word, calls", [("you misunderstood", 0), ("the wrong command seems to have been picked here", 1)])
def test_part_you_misunderstood_goes_back_to_the_which_command_question(todo, monkeypatch, phrase_or_word, calls):
    # bd fix-fpyb.12: nothing of the part ran before its question, so "you misunderstood" returns to the which-command
    # question as at the gate (every command the plan path can run), whose reply goes through the resolution.
    remote, _ = _asked_while_running(todo, monkeypatch, "misunderstood")
    asked = todo.ctx.process_turn(phrase_or_word)
    choice = _artifacts(asked)["plan_choice"]
    assert asked.status is TurnStatus.AWAITING_USER and _last(asked)["question_kind"] == "plan_choice"
    assert choice["misunderstood"] is True and choice["request"] == "who wrote that book"
    assert PICKED in [each["name"] for each in choice["choices"]] and len(remote.calls) == calls
    assert todo.ctx._waiting.kind == waiting.KIND_INTENT and todo.ctx._plan_resolution is not None


def _commands_run(out) -> list[str]:
    return [each.command_name for each in out.command_outputs
            if not getattr(each, "framework", False) and each.command_name != "ask_user"]


def test_part_you_misunderstood_then_a_pick_runs_the_picked_command_once(todo, monkeypatch):
    # An agent model is configured: the agent loop builds its LM from LLM_AGENT, and every call it makes is scripted.
    # Before, a developer's .env that litellm loaded into the process named one.
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/agent-model")
    _asked_while_running(todo, monkeypatch, "misunderstood")
    asked = todo.ctx.process_turn("you misunderstood")
    names = [each["name"] for each in _artifacts(asked)["plan_choice"]["choices"]]
    assert PICKED in names
    todo.script.steps = [("execute_workflow_query", {"command": "show book"}), *_finish()]
    done = todo.ctx.process_turn(PICKED)
    # The pick ran with no review and no second question, and the picked command ran exactly once.
    assert done.status is TurnStatus.COMPLETED and _commands_run(done) == [PICKED]
    assert todo.ctx._waiting is None and todo.ctx._plan_resolution is None


@pytest.mark.parametrize("word", ["cancel", "abort"])
def test_part_you_misunderstood_then_cancel_drops_the_part_as_not_run(todo, monkeypatch, word):
    _asked_while_running(todo, monkeypatch, "misunderstood")
    todo.ctx.process_turn("you misunderstood")
    done = todo.ctx.process_turn(word)
    assert done.status is TurnStatus.COMPLETED and "cancelled" in done.answer
    assert _commands_run(done) == [] and todo.ctx._plan_resolution is None and todo.ctx._waiting is None


def test_part_a_different_request_ends_the_part_and_is_not_run_as_a_request(todo, monkeypatch):
    remote, _ = _asked_while_running(todo, monkeypatch, "different_command")
    done = todo.ctx.process_turn("show me the users")
    assert done.status is TurnStatus.COMPLETED and len(remote.calls) == 1
    (stop,) = [stop for stop in _stops(done) if stop["command"] == BOOK_PLAN]
    assert "not run as a new request" in stop["reason"] and "not run as a new request" in done.answer


@pytest.mark.parametrize("word", ["cancel", "abort"])
def test_part_cancel_and_abort_end_the_part_with_no_model_call(todo, monkeypatch, word):
    remote, _ = _asked_while_running(todo, monkeypatch, "answer")
    done = todo.ctx.process_turn(word)
    assert done.status is TurnStatus.COMPLETED and remote.calls == []
    (stop,) = [stop for stop in _stops(done) if stop["command"] == BOOK_PLAN]
    assert stop["reason"].startswith(DECLINED)


def test_part_an_unreadable_reply_with_no_key_makes_no_call(todo, monkeypatch):
    _asked_while_running(todo, monkeypatch, None)
    remote = _Remote(monkeypatch, None)
    again = todo.ctx.process_turn("")
    assert remote.calls == [] and again.status is TurnStatus.AWAITING_USER
