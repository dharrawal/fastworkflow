"""The deterministic path's pending question lives in the turn's ONE waiting record (bd fix-fpyb.8, step 4 of
``docs/pending_state_design.md``), not in the built-in workflow's context.

Real ``WorkflowExecutionContext`` on a copy of ``tests/plain_required_workflow``, as ``tests/test_one_pending_question.py``
drives it (a "/" message opens a parameter question; "you misunderstood" replaces it with the which-command question).
The second half feeds the router's dispatcher (``wildcard._dispatch_reply``) the ``parameter_reply.Reply`` data of each
action in the frozen vocabulary, so every branch is covered whether or not a reader returns it yet.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow import intent_loop, parameter_clarification, waiting
from fastworkflow._workflows.command_metadata_extraction import parameter_reply
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction._commands.wildcard import Signature
from fastworkflow._workflows.command_metadata_extraction.parameter_reply import Reply
from tests.test_parameter_form_reply_contract import PARTIAL, session  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LEGACY_KEYS = ("parameter_loop", "intent_loop", "NLU_Pipeline_Stage", "is_assistant_mode_command")


def _holds_no_question(ctx) -> None:
    context = ctx._cme_workflow.context
    assert not [key for key in LEGACY_KEYS if key in context], sorted(context)
    assert isinstance(context[waiting.PORT_KEY], waiting.OwnerPort)
    assert not [value for value in context.values() if isinstance(value, waiting.Waiting)]


def _parameter_question(session):
    ctx = session(False)
    out = ctx.process_turn(PARTIAL)
    assert ctx.has_open_command() and out.answer
    return ctx


def _which_command_question(session):
    ctx = _parameter_question(session)
    ctx.process_turn("you misunderstood")
    assert ctx._waiting is not None and ctx._waiting.kind == "intent"
    return ctx


# ---- the context holds no pending key; the turn's record does -----------------------------------------------------------

def test_a_parameter_question_is_the_turns_record_and_not_a_key_of_the_context(session):
    ctx = _parameter_question(session)
    _holds_no_question(ctx)
    held = ctx._waiting
    assert held is ctx.waiting and (held.kind, held.owner, held.payload["after"]) == ("parameters", "cme", "run_command")
    assert held.payload["loop"]["command"] == ctx._cme_workflow.context["command_name"]
    assert waiting.port_of(ctx._cme_workflow.context).get() is held
    assert parameter_clarification.pending(ctx._cme_workflow.context).to_state() == held.payload["loop"]
    assert intent_loop.waiting_for(ctx._cme_workflow.context) == intent_loop.WAITING_PARAMETERS


def test_a_which_command_question_is_the_turns_record_and_not_a_key_of_the_context(session):
    ctx = _which_command_question(session)
    _holds_no_question(ctx)
    held = ctx._waiting
    assert (held.kind, held.owner) == ("intent", "cme")
    assert held.payload["record"] is intent_loop.pending_record(ctx._cme_workflow.context)
    assert parameter_clarification.pending(ctx._cme_workflow.context) is None       # one record: the other closed
    assert intent_loop.waiting_for(ctx._cme_workflow.context) == intent_loop.WAITING_INTENT


def test_the_review_slot_question_is_nested_in_the_review_record_and_never_touches_other_records(session):
    ctx = session(False)
    port = waiting.port_of(ctx._plan_cme.context)
    assert isinstance(port, waiting.SlotPort) and port.get() is None
    port.clear()                                       # nothing out: nothing happens
    with pytest.raises(AssertionError, match="no review out"):
        intent_loop.store(ctx._plan_cme.context, {"request": "x", "choices": []})
    ctx._waiting = waiting.Waiting(waiting.KIND_REVIEW, payload={"record_id": "r", "slot": None, "gate": True})
    review = ctx._waiting
    intent_loop.store(ctx._plan_cme.context, {"request": "x", "choices": []})
    assert review.payload["slot"] is port.get() and port.get().kind == "intent" and ctx._waiting is review
    assert not [key for key in LEGACY_KEYS if key in ctx._plan_cme.context]
    port.clear()
    assert review.payload["slot"] is None and ctx._waiting is review
    # The slot is saved inside the review's record, and the turn port's clear never drops the review itself.
    intent_loop.store(ctx._plan_cme.context, {"request": "x", "choices": []})
    saved = json.loads(json.dumps(ctx.serialize_state(channel_id="slot")))
    assert saved["waiting"]["kind"] == "review" and saved["waiting"]["payload"]["slot"]["kind"] == "intent"
    waiting.port_of(ctx._cme_workflow.context).clear()
    assert ctx._waiting is review


# ---- saved and restored --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["parameters", "intent"])
def test_the_record_survives_serialize_state_and_apply_serialized_state(session, which):
    ctx = _which_command_question(session) if which == "intent" else _parameter_question(session)
    saved = json.loads(json.dumps(ctx.serialize_state(channel_id="saved")))
    assert saved["waiting"]["kind"] == which and "cme" not in saved and "plan_cme" not in saved
    # The "/" round rides with the record, for either question: it is kept while the question is out.
    assert saved["waiting"]["assistant"] is ctx.assistant_round is True and "assistant_command" not in saved
    command, name = ctx._cme_workflow.context.get("command"), ctx._cme_workflow.context.get("command_name")

    rebuilt = session(False)
    assert rebuilt._waiting is None
    rebuilt.apply_serialized_state(saved)

    _holds_no_question(rebuilt)
    assert rebuilt._waiting == ctx._waiting and rebuilt.has_open_command()
    assert rebuilt.assistant_round is ctx.assistant_round and rebuilt._waiting.assistant is ctx.assistant_round
    assert rebuilt._cme_workflow.context.get("command") == command
    assert rebuilt._cme_workflow.context.get("command_name") == name
    assert "command" not in rebuilt._waiting.payload and "command_name" not in rebuilt._waiting.payload


def test_nothing_pending_saves_no_record(session):
    ctx = session(False)
    assert ctx.serialize_state(channel_id="idle")["waiting"] is None and "review_slot" not in ctx.serialize_state(channel_id="idle")


# ---- cancel and "/" still work through the record ------------------------------------------------------------------------

def test_abort_ends_the_question_through_the_record(session):
    ctx = _parameter_question(session)
    assert ctx.waiting.cancel_text == "abort"
    done = ctx.process_turn("abort")
    assert "aborted" in done.answer
    assert ctx._waiting is None and ctx.waiting is None and not ctx.has_open_command()
    _holds_no_question(ctx)


def test_the_slash_round_is_a_mark_of_the_turn_and_of_the_record(session):
    ctx = _parameter_question(session)
    assert ctx.assistant_round and ctx._waiting.assistant
    assert waiting.port_of(ctx._cme_workflow.context).assistant
    ctx.end_round()
    assert not ctx.assistant_round and not ctx._waiting.assistant
    assert not waiting.port_of(ctx._cme_workflow.context).assistant
    ctx.start_round()
    waiting.port_of(ctx._cme_workflow.context).end_round()      # the executor's way: through the port
    assert not ctx.assistant_round


def test_a_slash_reply_is_the_reply_to_the_question(session):
    ctx = _parameter_question(session)
    done = ctx.process_turn("/abort")
    assert "aborted" in done.answer and ctx._waiting is None and not ctx.assistant_round


def test_a_new_conversation_parks_the_record_and_drops_the_round(session):
    ctx = _parameter_question(session)
    ctx.park_pending_for_new_conversation()
    assert ctx._waiting is None and not ctx.assistant_round and not ctx.has_open_command()
    assert ctx.has_parked_pending()


# ---- the fallback holder (a context built without an execution context) --------------------------------------------------

def test_a_context_with_no_port_gets_a_holder_that_keeps_the_record_and_the_flag():
    context: dict = {}
    assert intent_loop.waiting_for(context) is None
    holder = context[waiting.PORT_KEY]
    assert isinstance(holder, waiting.Holder)
    loop = parameter_clarification.ParameterLoop(command="c", kind="execution", open=["a"])
    parameter_clarification.store(context, loop)
    assert parameter_clarification.pending(context).open == ["a"] and holder.get().kind == "parameters"
    intent_loop.store(context, {"request": "x"})                     # opening one closes the other
    assert parameter_clarification.pending(context) is None and intent_loop.pending_record(context) == {"request": "x"}
    parameter_clarification.clear(context)                           # clears only its own kind
    assert intent_loop.pending_record(context) is not None
    assert not holder.assistant
    context[waiting.ASSISTANT_KEY] = True
    assert holder.assistant
    holder.put(waiting.Waiting("parameters", owner="cme"))
    assert holder.get().assistant is True
    holder.end_round()
    assert not holder.assistant and holder.get().assistant is False
    holder.clear()
    assert holder.get() is None


def test_end_command_processing_clears_through_the_port_and_keeps_other_kinds(session):
    ctx = _parameter_question(session)
    ctx._cme_workflow.end_command_processing()
    assert ctx._waiting is None
    # A kind the CME does not own is never dropped by its port.
    ctx._waiting = waiting.Waiting(waiting.KIND_REVIEW)
    ctx._cme_workflow.end_command_processing()
    assert ctx._waiting is not None and ctx._waiting.kind == "review"
    ctx._waiting = None


# ---- the dispatcher: one test per action of the frozen vocabulary --------------------------------------------------------

def test_the_vocabulary_is_the_six_actions():
    assert parameter_reply.ACTIONS == ("run", "wait", "different", "replace", "merge", "stop")
    assert Reply("run").in_parameters is True and Reply("run").keep_pending is False


def _dispatch(ctx, reply, text="3", again=lambda: pytest.fail("routed again")):
    return wildcard._dispatch_reply(ctx._cme_workflow, text, reply, again)


def test_dispatch_run_abort_ends_the_question(session):
    ctx = _parameter_question(session)
    ended, extraction = _dispatch(ctx, Reply("run", name=parameter_reply.ABORT, reply_intent="abort"))
    assert extraction is None and ended.handled is True and ended.reply_intent == "abort"
    assert "aborted" in ended.response and ctx._waiting is None


def test_dispatch_run_you_misunderstood_replaces_the_question_with_the_which_command_question(session):
    ctx = _parameter_question(session)
    ended, _ = _dispatch(ctx, Reply("run", name=parameter_reply.MISUNDERSTOOD), text="no, wrong command")
    assert ended.handled is True and ended.reply_intent is None and ended.plan_choice is not None
    assert ctx._waiting.kind == "intent"


def test_dispatch_run_what_can_i_do_in_parameters_keeps_the_question(session):
    ctx = _parameter_question(session)
    held = ctx._waiting
    ended, _ = _dispatch(ctx, Reply("run", name=parameter_reply.WHAT_CAN_I_DO), text="what can i do")
    assert ended.handled is True and ended.response.startswith("Workflow Definition:")
    assert ctx._waiting is held                                     # in_parameters: the question is still open
    ended, _ = _dispatch(ctx, Reply("run", name=parameter_reply.WHAT_CAN_I_DO, keep_pending=True), text="what can i do")
    assert ctx._waiting is held


def test_dispatch_run_what_can_i_do_not_in_parameters_ends_the_question(session):
    ctx = _parameter_question(session)
    ended, _ = _dispatch(ctx, Reply("run", name=parameter_reply.WHAT_CAN_I_DO, in_parameters=False), text="what can i do")
    assert ended.handled is True and ctx._waiting is None


def test_dispatch_wait_answers_and_keeps_waiting(session):
    ctx = _parameter_question(session)
    held = ctx._waiting
    ended, extraction = _dispatch(ctx, Reply("wait", reply_intent="wait", text="OK, take your time.\nMissing: x"))
    assert extraction is None and (ended.handled, ended.routed) == (False, False)
    assert ended.reply_intent == "wait" and ended.response == "OK, take your time.\nMissing: x"
    assert ctx._waiting is held


def test_dispatch_different_ends_the_command_and_routes_the_reply_again(session):
    ctx = _parameter_question(session)
    routed: list[str] = []

    def again():
        routed.append("again")
        assert ctx._waiting is None                                 # the pending command went before the reply was routed
        return Signature.Output(handled=False, routed=False, response="fresh")

    ended, extraction = _dispatch(ctx, Reply("different", reply_intent="different_command"), text="show me the users", again=again)
    assert routed == ["again"] and extraction is None
    assert ended.response == "fresh" and ended.reply_intent == "different_command"


def test_dispatch_different_keeps_the_intent_the_new_route_already_gave(session):
    ctx = _parameter_question(session)
    ended, _ = _dispatch(ctx, Reply("different", reply_intent="different_command"),
                         again=lambda: Signature.Output(handled=True, reply_intent="wait"))
    assert ended.reply_intent == "wait"


def test_dispatch_replace_swaps_the_stored_command_and_goes_on_to_extraction(session):
    ctx = _parameter_question(session)
    new = "record_values name=Zed count=1"
    ended, extraction = _dispatch(ctx, Reply("replace", command=new), text=new)
    assert ended is None and extraction is None
    assert ctx._waiting is None and ctx._cme_workflow.context["command"] == new


def test_dispatch_merge_goes_on_to_extraction_with_the_reply_as_text(session):
    ctx = _parameter_question(session)
    held = ctx._waiting
    name = ctx._cme_workflow.context["command_name"]
    ended, extraction = _dispatch(ctx, Reply("merge", text="Bob"), text="Bob")
    assert ended is None and extraction == f"{name}: Bob" and ctx._waiting is held


def test_dispatch_stop_ends_the_command_failed_with_the_message_and_no_question(session):
    ctx = _parameter_question(session)
    ended, extraction = _dispatch(ctx, Reply("stop", message="Nothing more will be asked: `count` is missing.", reply_intent="answer"))
    assert extraction is None and (ended.handled, ended.routed) == (True, False)
    assert ended.response == "Nothing more will be asked: `count` is missing." and ended.reply_intent == "answer"
    assert ctx._waiting is None and not ctx.has_open_command()


def test_dispatch_refuses_an_action_outside_the_vocabulary(session):
    ctx = _parameter_question(session)
    with pytest.raises(ValueError):
        _dispatch(ctx, Reply("nonsense"))


# ---- the reader behind it returns today's actions -------------------------------------------------------------------------

def test_read_reply_returns_the_action_each_kind_of_reply_had_before(session, monkeypatch):
    ctx = _parameter_question(session)
    workflow, app = ctx._cme_workflow, ctx._cme_workflow.context["app_workflow"]
    name = workflow.context["command_name"]
    monkeypatch.setattr(parameter_reply, "_missing_value_reply_intent", lambda *args: ("answer", []))

    assert parameter_reply.read_reply(workflow, app, "abort") == Reply("run", name=parameter_reply.ABORT)
    assert parameter_reply.read_reply(workflow, app, "you misunderstood") == Reply("run", name=parameter_reply.MISUNDERSTOOD)
    full = f"{name.split('/')[-1]} name=Zed"
    assert parameter_reply.read_reply(workflow, app, full) == Reply("replace", command=full)
    assert parameter_reply.read_reply(workflow, app, "Bob") == Reply("merge", text="Bob")

    monkeypatch.setattr(parameter_reply, "_missing_value_reply_intent", lambda *args: ("wait", ["count"]))
    wait = parameter_reply.read_reply(workflow, app, "hold on")
    assert wait.action == "wait" and wait.reply_intent == "wait" and wait.text.startswith("OK, take your time.")
    # The question comes back as it was, under that line, as on the plan path (bd fix-ftdl.7.8, D9).
    assert wait.text == f"OK, take your time.\n\n{parameter_clarification.pending(workflow.context).question}"
    for intent, name_ in (("abort", parameter_reply.ABORT), ("misunderstood", parameter_reply.MISUNDERSTOOD)):
        monkeypatch.setattr(parameter_reply, "_missing_value_reply_intent", lambda *args, intent=intent: (intent, []))
        assert parameter_reply.read_reply(workflow, app, "x") == Reply("run", name=name_, reply_intent=intent)
    monkeypatch.setattr(parameter_reply, "_missing_value_reply_intent", lambda *args: ("different_command", []))
    # A different request is not run (bd fix-ftdl.8.10): the question comes back once, saying so.
    different = parameter_reply.read_reply(workflow, app, "list the users")
    assert different.action == "wait" and different.reply_intent == "different_command"
    assert "read as a new request" in different.text
    assert different.text.endswith(parameter_clarification.pending(workflow.context).question)
