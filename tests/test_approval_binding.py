"""Approval binding (ADR-EXP-025 sections 6-12 and 15): the pending-action record, the gate, and
the bound resolution of a reply.

Integration tests on a copy of the todo_list test workflow with ``approval`` declared on two of its
commands (one ``mandatory``, one ``delegatable``); the rest are unannotated. The agent is the real
``fastWorkflowReAct`` and its real tools; only the model's step choices are scripted (the same
seam the planning-command tests use), and the command-name resolution that needs a trained
intent model is replaced with a direct lookup. Pure pieces (fingerprint, policy, rendering,
token, state machine transitions) are tested on the real objects.
"""
from __future__ import annotations

import json
import os
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

import fastworkflow
from fastworkflow import approval_binding as ab
from fastworkflow.approval_annotations import REQUIRED_ALWAYS, REQUIRED_UNTIL_DELEGATED
from fastworkflow.approval_binding import Origin, Outcome, Surface
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.examples.library.application.library import Library  # noqa: F401
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import initialize_workflow_tool_agent
from tests.test_planning_command_phases import RecordingSink
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures and helpers
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
)

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
_COPY_IGNORE = shutil.ignore_patterns(
    "___workflow_contexts", "___convo_info", "__pycache__", "versions",
)
MANDATORY = "Record/add_note"            # annotated REQUIRED_ALWAYS (text=...)
DELEGATABLE = "Record/show_notes"        # annotated REQUIRED_UNTIL_DELEGATED (no parameters)
UNANNOTATED = "Book/show_book"           # no annotation: never gated

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


# ----------------------------------------------------------------------
# A copy of the trained library (its model versions are linked, not copied), annotated in the copy
# ----------------------------------------------------------------------


def _annotate(folder: Path, command_file: str, member: str) -> None:
    """Decorate the copy's ResponseGenerator with ``@user_approval_for_execution(UserApproval.<member>)``."""
    path = folder / "_commands" / command_file
    text = path.read_text()
    marker = "class ResponseGenerator:\n"
    assert text.count(marker) == 1, path
    decorator = f"@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.{member})\n"
    path.write_text(text.replace(marker, decorator + marker, 1))


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """No test here calls a model: a LiteLLM call fails the test instead of reaching a provider."""
    import litellm

    def _refuse(*args, **kwargs):
        raise AssertionError("a live model call was attempted from an approval test")

    monkeypatch.setattr(litellm, "completion", _refuse)
    monkeypatch.setattr(litellm, "acompletion", _refuse)


@pytest.fixture(autouse=True)
def no_model_call_for_parameters(monkeypatch):
    """Guard, not a stub: the deterministic name=value path binds these commands. If the model boundary is
    reached, the test fails here rather than calling out."""
    from fastworkflow.parameters.signatures import InputForParamExtraction

    def _refuse(self, *args, **kwargs):
        raise AssertionError(f"the parameter-extraction model was called for {self.command!r}")

    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", _refuse)


def _copy_fix(folder: Path, command_file: str, old: str, new: str) -> None:
    path = folder / "_commands" / command_file
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new, 1))


@pytest.fixture
def annotated_library(tmp_path) -> Path:
    folder = tmp_path / "library_annotated"
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    # The trained versions are ~1.4 GB: link them rather than copy them. Tests here never train.
    os.symlink(LIBRARY / "___command_info" / "versions", folder / "___command_info" / "versions")
    _annotate(folder, "Record/add_note.py", "REQUIRED_ALWAYS")
    _annotate(folder, "Record/show_notes.py", "REQUIRED_UNTIL_DELEGATED")
    # add_note and show_notes are the library's own commands, unchanged: each returns its Signature.Output,
    # and the framework frames it (design 3.14). show_notes takes no Input, so it is called with None.
    return folder


# ----------------------------------------------------------------------
# A real agent with a scripted model. The intent path is the trained one: no stub on invoke_command.
# ----------------------------------------------------------------------


class Script:
    """The model's next steps. Asked for a step nobody scripted, it fails the test.

    Every frame's agent is scripted: a planning command runs in a frame of its own with its own agent (3.4),
    and that agent's steps come from the same script."""

    def __init__(self, agent):
        self.agent = agent
        self.agents = [agent]
        self.steps: list = []

    def __call__(self, module, trajectory, **inputs):
        if any(module is a.react for a in self.agents):
            if not self.steps:
                raise AssertionError(f"the model was asked for a step after {trajectory}")
            tool, args = self.steps.pop(0)
            return dspy.Prediction(next_thought="scripted", next_tool_name=tool, next_tool_args=args)
        return dspy.Prediction(reasoning="", final_answer="the answer")


@pytest.fixture
def todo(annotated_library, monkeypatch, tmp_path):
    """Agent-mode context on the annotated library copy, on book b1 (the record the commands act on)."""
    from fastworkflow.examples.library.application.library import Library

    ctx, wf = _make_agent_ctx(str(annotated_library), monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    ctx.push_active_workflow(wf)
    library = Library()
    wf.root = library
    wf.record = library.books["b1"]
    # Agent mode as a live turn sets it (the parameter path reads it from the app context).
    ctx._cme_workflow.context["run_as_agent"] = True
    wf.context["run_as_agent"] = True
    # A book is identified by its uid in the instance label (as a record is), so two books differ.
    from fastworkflow.context_identity import context_class_for
    monkeypatch.setattr(context_class_for(wf, "Book"), "instance_label_attr", "uid", raising=False)
    agent = initialize_workflow_tool_agent(ctx)
    _set_agents(ctx, agent)
    script = Script(agent)
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", script)
    # A nested planning-command frame builds its own agent: script that one too, through the same Script.
    build_agent = ctx._build_react_agent

    def scripted_build():
        frame_agent = build_agent()
        script.agents.append(frame_agent)
        monkeypatch.setattr(frame_agent, "_call_with_potential_trajectory_truncation", script)
        return frame_agent

    monkeypatch.setattr(ctx, "_build_react_agent", scripted_build)
    # The two seams that call the model before the agent loop: accepted (the LLM boundary).
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda user_query, **kwargs: [user_query])
    monkeypatch.setattr(ctx, "_refine_user_query", lambda message, history: message)
    # The sub-task review gate (fix-3rsq.plan-review.2.6) is not the subject of these approval tests: their turns run
    # as the fast path did, with no review, so each test exercises the approval gate and nothing else. The gate has its
    # own tests (tests/test_round1_gate.py).
    monkeypatch.setattr(ctx, "_subtask_gate", lambda subtasks: None)
    return SimpleNamespace(ctx=ctx, wf=wf, sink=sink, script=script, library=library,
                           chores=library.books["b1"], other=library.books["b2"],
                           base=list(library.books["b1"].notes), base_other=list(library.books["b2"].notes),
                           store=ab.store_for(wf))


def _ask(todo, command: str, *, steps, **kwargs):
    """One agent turn: the model calls ``execute_workflow_query`` with *command*, then the scripted rest."""
    todo.script.steps = [("execute_workflow_query", {"command": command}), *steps]
    return todo.ctx.process_turn("do it", **kwargs)


def _finish():
    return [("finish", {})]


def _added(book, base) -> list:
    """The notes the commands added to *book* since the test began (the fixture's baseline)."""
    return book.notes[len(base):]


def _delivered_token(todo) -> str:
    """The token the host receives: taken once from the delivery seam, never from an output."""
    delivery = ab.take_delivery(todo.wf)
    assert delivery is not None
    return delivery["token"]


# ----------------------------------------------------------------------
# Pure pieces: fingerprint, rendering, policy, token
# ----------------------------------------------------------------------


def _binding(**overrides):
    base = dict(command="TodoList/set_properties", params={"assign_to": "Ann"},
                targets=["TodoList", "TodoList 7"], principal={"class": "channel", "ref": "c1"},
                task="t1", channel="c1")
    base.update(overrides)
    return ab.make_binding(**base)


def test_fingerprint_is_stable_across_key_order_and_changes_with_any_field():
    a = _binding(params={"assign_to": "Ann", "is_complete": True})
    b = _binding(params={"is_complete": True, "assign_to": "Ann"})
    assert ab.fingerprint(a) == ab.fingerprint(b)
    for changed in (
        _binding(params={"assign_to": "Bob", "is_complete": True}),
        _binding(targets=["TodoList", "TodoList 8"]),
        _binding(principal={"class": "channel", "ref": "c2"}),
        _binding(task="t2"),
        _binding(channel="c2"),
        _binding(command="TodoList/mark_completed"),
    ):
        assert ab.fingerprint(changed) != ab.fingerprint(a)


def test_rendering_is_derived_from_the_parameters_and_a_template_only():
    assert ab.render(None, {"assign_to": "Ann", "is_complete": None}) == "assign_to=Ann, is_complete=none"
    assert ab.render(None, {}) == "no parameters"
    assert ab.render("assign to {assign_to}", {"assign_to": "Ann"}) == "assign to Ann"
    # A template that does not fill from the parameters falls back to the default, never to free text.
    assert ab.render("assign to {nobody}", {"assign_to": "Ann"}) == "assign_to=Ann"


def test_values_read_as_people_write_them():
    # true and false read as yes and no; None as none; lists and maps item by item.
    assert ab.render(None, {"is_complete": True, "done": False}) == "is_complete=yes, done=no"
    assert ab.render(None, {"tags": ["a", None], "meta": {"ok": True}}) == "tags=[a, none], meta={ok: yes}"
    assert ab.render("complete={is_complete}", {"is_complete": False}) == "complete=no"
    assert ab.render("count {n}", {"n": 3}) == "count 3"


def test_token_is_256_bits_and_only_its_hash_is_compared():
    token = ab.new_token()
    assert len(token) >= 43          # token_urlsafe(32)
    digest = ab.token_hash(token)
    assert digest != token
    assert ab.token_matches(token, digest)
    assert not ab.token_matches(ab.new_token(), digest)
    assert not ab.token_matches(None, digest)


@pytest.mark.parametrize("annotation, origin, surface, unattended, awaiting, outcome", [
    (None, Origin.AGENT, Surface(), False, False, Outcome.RUN),                       # unannotated
    (REQUIRED_ALWAYS, Origin.USER, Surface(), False, False, Outcome.ASK),                # user-typed: bound too (owner 2026-11-02)
    (REQUIRED_UNTIL_DELEGATED, Origin.USER, Surface(), False, False, Outcome.RUN),              # user's own instruction
    (REQUIRED_ALWAYS, Origin.AGENT, Surface(), False, False, Outcome.ASK),
    (REQUIRED_UNTIL_DELEGATED, Origin.AGENT, Surface(), False, False, Outcome.ASK),              # asks until delegation exists
    (REQUIRED_ALWAYS, Origin.AGENT, Surface(), True, False, Outcome.NOT_ASKED),              # no human channel
    (REQUIRED_ALWAYS, Origin.AGENT, Surface(), False, True, Outcome.NOT_ASKED),              # behind another ask
    (REQUIRED_ALWAYS, Origin.AGENT_DIRECT, Surface(), False, False, Outcome.ASK),        # MCP is agent-originated
    (REQUIRED_ALWAYS, Origin.AGENT, Surface("host", verified=False, uid="u"), False, False, Outcome.REFUSE),
    (REQUIRED_ALWAYS, Origin.AGENT, Surface("host", verified=True, uid=None), False, False, Outcome.REFUSE),
    (REQUIRED_ALWAYS, Origin.AGENT, Surface("host", verified=True, uid="u"), False, False, Outcome.ASK),
])
def test_the_policy_decision_point_returns_graded_outcomes(annotation, origin, surface, unattended,
                                                           awaiting, outcome):
    decision = ab.decide(annotation, origin=origin, surface=surface, unattended=unattended,
                         awaiting_on_channel=awaiting)
    assert decision.outcome is outcome


def test_a_recorded_standing_approval_runs_without_a_prompt_on_any_surface():
    # The grant is an input: it yields run for a delegatable command (bd fix-qjne.11: only for that one;
    # the REQUIRED_ALWAYS counterpart is the test below, it was moved here from "both annotations").
    # (Edited by fix-d6no: the channel cells no longer pass unattended=True; an unattended channel run
    # queues whatever the grant says, see the test below. The host cell is added and is unchanged.)
    annotation = REQUIRED_UNTIL_DELEGATED
    for origin in (Origin.AGENT, Origin.AGENT_DIRECT, Origin.USER):
        decision = ab.decide(annotation, origin=origin, surface=Surface(), unattended=False,
                             awaiting_on_channel=True, auto_granted=True)
        assert decision.outcome is Outcome.RUN
        host = ab.decide(annotation, origin=origin, surface=Surface("host", verified=True, uid="u"),
                         unattended=True, awaiting_on_channel=True, auto_granted=True)
        assert host.outcome is Outcome.RUN        # a verified host's per-end-user grant: unchanged
    # ...but an unverifiable host still has no identity to grant to, so it is refused first.
    refused = ab.decide(annotation, origin=Origin.AGENT, surface=Surface("host", verified=False, uid="u"),
                        unattended=False, awaiting_on_channel=False, auto_granted=True)
    assert refused.outcome is Outcome.REFUSE


def test_a_recorded_standing_approval_never_runs_a_required_always_command_on_any_surface():
    # bd fix-qjne.11: a person confirms every run. The grant is ignored on every surface and attendance.
    host = Surface("host", verified=True, uid="u")
    for origin in (Origin.AGENT, Origin.AGENT_DIRECT, Origin.USER):
        for surface in (Surface(), Surface(kind="channel", origin="chat"), host):
            attended = ab.decide(REQUIRED_ALWAYS, origin=origin, surface=surface, unattended=False,
                                 awaiting_on_channel=False, auto_granted=True)
            assert attended.outcome is Outcome.ASK
            behind = ab.decide(REQUIRED_ALWAYS, origin=origin, surface=surface, unattended=False,
                               awaiting_on_channel=True, auto_granted=True)
            assert behind.outcome is Outcome.NOT_ASKED
            unattended = ab.decide(REQUIRED_ALWAYS, origin=origin, surface=surface, unattended=True,
                                   awaiting_on_channel=False, auto_granted=True)
            assert unattended.outcome is Outcome.NOT_ASKED
            # the same answer with no grant: a grant changes nothing for this annotation
            assert ab.decide(REQUIRED_ALWAYS, origin=origin, surface=surface, unattended=False,
                             awaiting_on_channel=False).outcome is attended.outcome
    refused = ab.decide(REQUIRED_ALWAYS, origin=Origin.AGENT, surface=Surface("host", verified=False, uid="u"),
                        unattended=False, awaiting_on_channel=False, auto_granted=True)
    assert refused.outcome is Outcome.REFUSE


def test_the_policy_accepts_consequence_and_uncertainty_without_changing_its_answer():
    base = ab.decide(REQUIRED_ALWAYS, origin=Origin.AGENT, surface=Surface(), unattended=False,
                     awaiting_on_channel=False)
    with_inputs = ab.decide(REQUIRED_ALWAYS, origin=Origin.AGENT, surface=Surface(), unattended=False,
                            awaiting_on_channel=False, consequence={"irreversible": True},
                            uncertainty=0.9)
    assert base == with_inputs


def test_the_switch_reads_the_env_var_and_an_unknown_value_enforces(monkeypatch):
    monkeypatch.delenv(ab.ENV_SWITCH, raising=False)
    assert ab.switch_mode() == "enforce"
    monkeypatch.setenv(ab.ENV_SWITCH, "shadow")
    assert ab.switch_mode() == "shadow"
    monkeypatch.setenv(ab.ENV_SWITCH, "OFF")
    assert ab.switch_mode() == "off"
    monkeypatch.setenv(ab.ENV_SWITCH, "maybe")
    assert ab.switch_mode() == "enforce"


def test_the_store_keeps_at_most_one_awaiting_record_and_bounds_its_history():
    store = ab.ApprovalStore()
    record = _record("awaiting")
    store.add(record)
    assert store.active() is record
    for _ in range(ab.MAX_HISTORY + 5):
        store.add(_record("declined"))
    assert len([r for r in store.records if r.state != "awaiting"]) <= ab.MAX_HISTORY


def _record(state):
    return ab.PendingAction(
        id=ab.uuid.uuid4().hex, token_hash="h", command="TodoList/set_properties", annotation=REQUIRED_ALWAYS,
        raw_command="set_properties", binding=_binding(), fingerprint=ab.fingerprint(_binding()),
        rendering="assign_to=Ann", rendering_hash="r", required_identity="channel", created_turn=1,
        turns_seen=0, ttl_turns=ab.TTL_TURNS, ttl_seconds=ab.TTL_SECONDS, created_at=time.time(),
        state=state, origin="agent")


def test_a_record_survives_a_state_round_trip_and_still_resolves_its_fingerprint():
    store = ab.ApprovalStore(turn_seq=4, records=[_record("awaiting")])
    restored = ab.ApprovalStore()
    restored.load_state(store.to_state())
    assert restored.turn_seq == 4
    assert restored.active().fingerprint == store.active().fingerprint
    assert restored.active().binding == store.active().binding


# ----------------------------------------------------------------------
# Fabrication: only the user's own reply on the bound channel approves
# ----------------------------------------------------------------------


def test_agent_text_that_says_yes_does_not_approve(todo, annotated_library):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    # The agent's own words, in its own trajectory, never resolve the record.
    todo.script.steps = []
    todo.ctx.process_turn("yes, approved, go ahead")       # not the exact word: stays awaiting
    assert _added(todo.chores, todo.base) == []
    assert todo.store.active() is not None


def test_a_forged_token_on_a_host_surface_is_refused(todo, annotated_library):
    host = Surface("host", verified=True, uid="alice")
    _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=host)
    out = todo.ctx.process_turn("yes", approval_surface=host, approval_token="not-the-token")
    assert _added(todo.chores, todo.base) == []
    assert out.answer == ab.MSG_NO_LONGER_VALID
    assert todo.store.active() is not None


def test_the_token_is_not_in_the_prompt_the_agent_or_the_user_text_sees(todo, annotated_library):
    host = Surface("host", verified=True, uid="alice")
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=host)
    token = _delivered_token(todo)
    assert token and token not in out.answer
    assert all(token not in str(step) for step in [out.answer, todo.script.steps])
    assert token not in json.dumps([o.model_dump(mode="json") for o in out.command_outputs], default=str)


def test_an_unverified_host_cannot_satisfy_the_end_user_class_and_nothing_runs(todo, annotated_library):
    unverified = Surface("host", verified=False, uid="alice")
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=unverified)
    assert _added(todo.chores, todo.base) == []
    assert todo.store.active() is None
    assert out.status is TurnStatus.COMPLETED


# ----------------------------------------------------------------------
# The agent path: the ask suspends the tool call, and the yes resumes it with the bound binding
# ----------------------------------------------------------------------


def test_an_agent_call_suspends_on_the_prompt_and_a_yes_runs_the_stored_binding_once(todo, annotated_library):
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    # bd fix-ftdl.7.6: the default question says nothing about undoing; an author who wants that writes it in
    # the command's approval_template.
    assert out.answer == ("The agent wants to run Record/add_note with text=Ann. Reply yes to approve "
                          "or no to decline. You are asked every time.")
    record = todo.store.active()
    assert record is not None and record.suspended_agent and record.state == "awaiting"
    assert _added(todo.chores, todo.base) == []                       # nothing ran yet

    done = todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == ["Ann"]                    # the stored binding ran
    assert done.status is TurnStatus.COMPLETED
    assert record.state == "executed" and record.outcome == "success"

    # Replay: a second yes has nothing to resolve, and the record cannot run again.
    assert ab.resolve_reply(todo.ctx, todo.wf, "yes") is None


def test_a_declined_prompt_never_runs_and_says_so(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    done = todo.ctx.process_turn("no")
    assert _added(todo.chores, todo.base) == []
    assert record.state == "declined"
    assert done.status is TurnStatus.COMPLETED


def test_a_changed_target_between_prompt_and_yes_invalidates_and_does_not_run(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    todo.wf.record = todo.other            # the target instance is now another list
    todo.ctx.process_turn("yes")
    assert record.state == "invalidated"
    assert _added(todo.chores, todo.base) == [] and _added(todo.other, todo.base_other) == []


def test_a_different_principal_between_prompt_and_yes_invalidates(todo, annotated_library):
    alice = Surface("host", verified=True, uid="alice")
    _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=alice)
    record = todo.store.active()
    token = _delivered_token(todo)
    # A different verified user presents the token: the principal is part of the binding.
    todo.ctx.process_turn("yes", approval_surface=Surface("host", verified=True, uid="mallory"),
                          approval_token=token)
    assert record.state == "invalidated"
    assert _added(todo.chores, todo.base) == []


def test_a_host_confirmation_with_the_token_and_the_same_user_runs(todo, annotated_library):
    host = Surface("host", verified=True, uid="alice")
    _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=host)
    token = _delivered_token(todo)
    done = todo.ctx.process_turn("yes", approval_surface=host, approval_token=token)
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert done.status is TurnStatus.COMPLETED


# ----------------------------------------------------------------------
# Staleness, TTL, unattended, delegatable, unannotated
# ----------------------------------------------------------------------


def test_two_turns_of_other_replies_expire_the_record_by_turns(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    todo.ctx.process_turn("maybe later")
    todo.ctx.process_turn("sure")
    todo.ctx.process_turn("yes")
    assert record.state == "expired"
    assert _added(todo.chores, todo.base) == []


def test_the_record_expires_by_wall_clock(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    record.created_at = time.time() - ab.TTL_SECONDS - 1
    todo.ctx.process_turn("yes")
    assert record.state == "expired"
    assert _added(todo.chores, todo.base) == []


def test_an_unattended_call_is_not_run_and_nothing_waits_for_an_answer(todo, annotated_library):
    # bd fix-ftdl.7.6: it used to be "queued until it expires": a record that was never asked about and that made the
    # session report an approval pending that nobody could answer. One thing is waited on at a time (waiting.py), and
    # a call that cannot be asked now is not run and not kept: the caller is told why, at once.
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish(), unattended=True)
    assert _added(todo.chores, todo.base) == []
    assert out.status is TurnStatus.COMPLETED                  # the caller was told at once
    assert ab.active_approvals(todo.wf) == []
    assert todo.ctx.has_pending_state() is False
    assert len(_approval_spans(todo, "not_asked")) == 1
    # The next turns find nothing to answer and nothing runs on its own.
    todo.script.steps = _finish()
    todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == []


def test_an_unattended_call_says_plainly_that_it_was_not_run_and_why(todo, annotated_library):
    todo.store.unattended = True
    result = CommandExecutor.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=MANDATORY, arguments={"text": "Ann"}))
    assert result.content[0].text == ab.MSG_UNATTENDED.format(command=MANDATORY)
    assert result.content[0].text == ("Record/add_note was not run: it needs your approval, and no one can give it "
                                      "on this turn. Run it again when you can answer.")
    assert ab.active_approvals(todo.wf) == [] and _added(todo.chores, todo.base) == []


def test_a_delegatable_command_asks_with_the_framework_text(todo, annotated_library):
    out = _ask(todo, "show_notes", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    assert out.answer == ("The agent wants to run Record/show_notes with no parameters. "
                          "Reply yes to approve, no to decline, or always to approve this command from now on.")
    assert todo.store.active().annotation == REQUIRED_UNTIL_DELEGATED


def test_an_unannotated_command_is_never_gated(todo, annotated_library):
    out = _ask(todo, "show_book", steps=_finish())
    assert out.status is TurnStatus.COMPLETED
    assert todo.store.records == []


# ----------------------------------------------------------------------
# MCP origin: an agent-originated call that cannot suspend gets its prompt back, and the next
# message on the session resolves it
# ----------------------------------------------------------------------


def test_an_mcp_tool_call_to_an_annotated_command_is_agent_originated_and_asks(todo, annotated_library):
    from fastworkflow.command_executor import CommandExecutor as CE

    result = CE.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=MANDATORY, arguments={"text": "Ann"}))
    text = result.content[0].text
    assert text == ("The agent wants to run Record/add_note with text=Ann. "
                    "Reply yes to approve or no to decline. You are asked every time.")
    assert _added(todo.chores, todo.base) == []
    record = todo.store.active()
    assert record is not None and record.origin == Origin.AGENT_DIRECT.value
    assert record.suspended_agent is False

    done = todo.ctx.process_turn("yes")
    assert record.state == "executed" and record.outcome == "success"
    assert done.status is TurnStatus.COMPLETED


def test_an_mcp_call_to_an_unannotated_command_is_not_gated(todo, annotated_library):
    CommandExecutor.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=UNANNOTATED, arguments={}))
    assert todo.store.records == []


# ----------------------------------------------------------------------
# The switch
# ----------------------------------------------------------------------


def test_switch_off_runs_the_annotated_command_with_no_record(todo, annotated_library, monkeypatch):
    monkeypatch.setenv(ab.ENV_SWITCH, "off")
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert todo.store.records == []
    assert out.status is TurnStatus.COMPLETED


def test_switch_shadow_records_the_decision_and_runs_as_before(todo, annotated_library, monkeypatch):
    monkeypatch.setenv(ab.ENV_SWITCH, "shadow")
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert _added(todo.chores, todo.base) == ["Ann"]                      # executed without a prompt
    assert out.status is TurnStatus.COMPLETED
    [record] = todo.store.records
    assert record.shadow and record.state == "executed"
    assert record.outcome == "shadow"


def test_switch_enforce_is_the_default_and_prompts(todo, annotated_library, monkeypatch):
    monkeypatch.delenv(ab.ENV_SWITCH, raising=False)
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    assert _added(todo.chores, todo.base) == []


# ----------------------------------------------------------------------
# Audit (ADR section 12): one row per transition, and a confirmed row precedes each execution
# ----------------------------------------------------------------------


def test_each_executed_gated_write_has_a_confirmed_audit_row_with_the_same_fingerprint(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    todo.ctx.process_turn("yes")
    rows = [s.attributes for s in todo.sink.spans if s.name == "fw.approval"]
    states = [(row["approval_state_before"], row["approval_state_after"]) for row in rows]
    assert ("", "awaiting") in states
    assert ("awaiting", "confirmed") in states
    assert ("confirmed", "executed") in states
    fingerprints = {row["approval_fingerprint"] for row in rows}
    assert len(fingerprints) == 1
    assert all("Ann" not in str(row) for row in rows)           # no raw parameters in the audit


# ----------------------------------------------------------------------
# Owner amendment 2026-11-02: user-typed invocations are bound too
# ----------------------------------------------------------------------


def test_a_user_typed_mandatory_command_asks_and_a_yes_runs_it_once(todo, annotated_library):
    # Outside the agent's loop: the origin is the user's own instruction, and it is still bound.
    out = CommandExecutor.perform_action(todo.wf, fastworkflow.Action(
        command_name=MANDATORY, command="add_note <text>Ann</text>", parameters={"text": "Ann"}))
    assert out.command_response.artifacts.get("awaiting_user") is True
    assert "You asked to run Record/add_note" in out.command_response.response
    assert _added(todo.chores, todo.base) == []
    todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == ["Ann"]
    [record] = todo.store.records
    assert record.origin == Origin.USER.value and record.state == "executed"


def test_a_user_typed_delegatable_command_is_the_users_own_instruction_and_runs(todo, annotated_library):
    CommandExecutor.perform_action(todo.wf, fastworkflow.Action(command_name=DELEGATABLE, command="show_notes"))
    assert todo.store.records == []


# ----------------------------------------------------------------------
# /cancel_pending and queued-behind
# ----------------------------------------------------------------------


def test_cancel_pending_message_cancels_a_suspended_approval_and_nothing_runs(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    assert todo.ctx.awaiting_user is True
    todo.ctx.process_turn("/cancel_pending")
    assert record.state == "invalidated"
    assert todo.store.active() is None
    assert todo.ctx.awaiting_user is False          # the suspended tool call is abandoned
    assert _added(todo.chores, todo.base) == []


def test_cancel_pending_method_clears_an_mcp_approval(todo, annotated_library):
    CommandExecutor.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=MANDATORY, arguments={"text": "Ann"}))
    record = todo.store.active()
    assert record is not None
    assert todo.ctx.cancel_pending_approvals() == 1
    assert record.state == "invalidated"


def test_a_second_gated_call_while_one_awaits_is_not_run_and_not_kept(todo, annotated_library):
    # bd fix-ftdl.7.6: one question at a time. The second call is not run and not kept as a record nobody asks
    # about: the caller is told to answer the first, then run it again.
    first = CommandExecutor.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=MANDATORY, arguments={"text": "Ann"}))
    assert "Reply yes" in first.content[0].text
    second = CommandExecutor.perform_mcp_tool_call(
        todo.wf, fastworkflow.MCPToolCall(name=MANDATORY, arguments={"text": "Bob"}))
    assert second.content[0].text == ("Record/add_note was not run: it needs your approval, and another approval is "
                                      "already waiting for your answer. Answer that one first, then run it again.")
    assert [r.binding["params"]["text"] for r in ab.active_approvals(todo.wf)] == ["Ann"]
    todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == ["Ann"]           # the second one never ran on its own
    assert ab.active_approvals(todo.wf) == []


# ----------------------------------------------------------------------
# Standing auto-approval: record, use, list and revoke, each audited
# ----------------------------------------------------------------------


@pytest.fixture
def grant_root(tmp_path, monkeypatch):
    import fastworkflow.session.state_paths as state_paths
    monkeypatch.setattr(state_paths, "state_root", lambda: str(tmp_path / "state"))
    return tmp_path / "state"


def _approval_spans(todo, outcome: str):
    return [s.attributes for s in todo.sink.spans
            if s.name == "fw.approval" and s.attributes.get("approval_outcome") == outcome]


def test_always_records_a_standing_approval_that_later_runs_without_a_prompt(todo, annotated_library, grant_root):
    # bd fix-qjne.11: the mechanism is proved on the delegatable command (it was the mandatory one before).
    _ask(todo, "show_notes", steps=_finish())
    todo.ctx.process_turn("always")
    assert todo.store.records[-1].state == "executed"          # this binding ran on the user's reply
    assert len(_approval_spans(todo, "recorded")) == 1

    _ask_out = _ask(todo, "show_notes", steps=_finish())
    assert _ask_out.status is TurnStatus.COMPLETED  # no prompt: the standing approval ran it
    assert len(_approval_spans(todo, "used")) == 1


def test_a_required_always_command_asks_every_time_and_an_always_reply_records_nothing(todo, annotated_library,
                                                                                    grant_root):
    first = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert first.status is TurnStatus.AWAITING_USER
    assert "always" not in first.answer.lower().replace("asked every time", "")
    typed = todo.ctx.process_turn("always")
    assert "always" in typed.answer and "asks every time" in typed.answer     # the question again, no promise
    assert "standing" not in typed.answer and "from now on" not in typed.answer
    assert typed.status is TurnStatus.AWAITING_USER
    assert _added(todo.chores, todo.base) == []                               # not read as yes
    assert todo.store.active() is not None and todo.store.active().state == "awaiting"
    assert _approval_spans(todo, "recorded") == []
    todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert _approval_spans(todo, "recorded") == []

    second = _ask(todo, "add_note <text>Bob</text>", steps=_finish())
    assert second.status is TurnStatus.AWAITING_USER                          # asked again: no grant exists
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert _approval_spans(todo, "used") == []


def test_an_existing_grant_for_a_required_always_command_is_ignored_listed_and_revocable(todo, annotated_library,
                                                                                       grant_root):
    from fastworkflow import approval_grants as ag
    ag.record(f"channel:{ag.local_user()}", state_paths_workflow_id(todo), "Record/add_note", REQUIRED_ALWAYS)
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER and _added(todo.chores, todo.base) == []
    assert _approval_spans(todo, "used") == []
    todo.ctx.process_turn("no")
    listing = todo.ctx.process_turn("/approvals").answer           # still listed, marked as not applying
    assert "Record/add_note" in listing and "ignored" in listing
    assert "is revoked" in todo.ctx.process_turn("/revoke_approval add_note").answer
    assert "Record/add_note" not in todo.ctx.process_turn("/approvals").answer
    assert ag.find(f"channel:{ag.local_user()}", state_paths_workflow_id(todo), "Record/add_note") is None


def test_the_approval_payload_offers_always_only_for_a_delegatable_command(todo, annotated_library, grant_root):
    mandatory = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert mandatory.status is TurnStatus.AWAITING_USER
    assert _pending_can_always(todo) is False
    todo.ctx.process_turn("no")
    delegatable = _ask(todo, "show_notes", steps=_finish())
    assert delegatable.status is TurnStatus.AWAITING_USER
    assert _pending_can_always(todo) is True


def _pending_can_always(todo):
    from fastworkflow import prompt_payloads
    return prompt_payloads.payload_of(todo.ctx, todo.ctx._waiting)[1]["approval"]["can_always"]


def test_a_standing_approval_is_revocable_and_the_next_call_asks_again(todo, annotated_library, grant_root):
    _ask(todo, "show_notes", steps=_finish())
    todo.ctx.process_turn("always")
    listing = todo.ctx.process_turn("/approvals")
    assert "Record/show_notes" in listing.answer and "ignored" not in listing.answer
    revoked = todo.ctx.process_turn("/revoke_approval show_notes")
    assert "is revoked" in revoked.answer
    assert len(_approval_spans(todo, "revoked")) == 1

    out = _ask(todo, "show_notes", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER    # asks again
    assert todo.store.active() is not None


def test_a_standing_approval_belongs_to_one_user_only(grant_root, monkeypatch):
    from fastworkflow import approval_grants as ag
    from fastworkflow.approval_binding import Surface
    alice = ag.identity_key(Surface("host", verified=True, uid="alice"), "c1")
    bob = ag.identity_key(Surface("host", verified=True, uid="bob"), "c1")
    assert alice == "end_user:alice" and bob == "end_user:bob"
    assert ag.identity_key(Surface("host", verified=False, uid="alice"), "c1") is None   # unverified: none
    ag.record(alice, "wf", "Record/add_note", REQUIRED_ALWAYS)
    assert ag.find(alice, "wf", "Record/add_note") is not None
    assert ag.find(bob, "wf", "Record/add_note") is None
    assert ag.revoke(alice, "wf", "Record/add_note") is not None
    assert ag.find(alice, "wf", "Record/add_note") is None


def test_a_corrupt_grant_file_grants_nothing_and_fails_closed(grant_root):
    from fastworkflow import approval_grants as ag
    path = Path(ag.grants_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert ag.find("channel:x", "wf", "Record/add_note") is None
    with pytest.raises(RuntimeError):
        ag.record("channel:x", "wf", "Record/add_note", REQUIRED_ALWAYS)


# ----------------------------------------------------------------------
# Token: never persisted, logged or in an output; delivered once to the host
# ----------------------------------------------------------------------


def test_the_host_token_appears_in_no_persisted_bytes_on_the_real_store_or_session_state(todo, annotated_library, tmp_path):
    from fastworkflow.observability.store import SQLiteTraceSink
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    todo.ctx.set_trace_sink(sink)
    host = Surface("host", verified=True, uid="alice")
    out = _ask(todo, "add_note <text>Ann</text>", steps=_finish(), approval_surface=host)
    token = _delivered_token(todo)                  # the live response only
    awaiting_state = json.dumps(todo.ctx.serialize_state(channel_id="c1"), default=str)
    assert token not in awaiting_state
    assert token not in json.dumps([o.model_dump(mode="json") for o in out.command_outputs], default=str)

    todo.ctx.process_turn("yes", approval_surface=host, approval_token=token)
    assert _added(todo.chores, todo.base) == ["Ann"]
    assert sink.flush()
    sink.close()

    stored = b""
    for name in (db_path, db_path + "-wal", db_path + "-shm"):
        if os.path.exists(name):
            stored += Path(name).read_bytes()
    final_state = json.dumps(todo.ctx.serialize_state(channel_id="c1"), default=str).encode()
    assert token.encode() not in stored
    assert token.encode() not in final_state
    assert b"approval" in stored or b"fw.approval" in stored   # the audit did persist; only the token is absent


# ----------------------------------------------------------------------
# A planning command's step that calls an annotated command gates at that step (ADR section 4, 15)
# ----------------------------------------------------------------------


def test_a_planning_step_that_calls_an_annotated_command_gates_at_that_step_and_runs_once(todo, annotated_library):
    # note_member (a planning command of the library) runs find_member, open_member, add_note. add_note is
    # annotated mandatory, so the step gates; the reply runs it exactly once, and the phase goes on.
    from fastworkflow.approval_annotations import command_approval
    folder = str(annotated_library)
    assert command_approval(folder, "Library/note_member") == REQUIRED_ALWAYS    # derived from its steps
    todo.wf.record = todo.library        # a planning command of the root library
    member = todo.library.members["m1"]
    before = list(member.notes)

    out = _ask(todo, "note_member <member>Ann Lee</member> <text>card renewed</text>", steps=_finish())
    assert out.status is TurnStatus.AWAITING_USER
    assert "Record/add_note with text=card renewed" in out.answer
    assert member.notes == before                          # the gate fired before the step ran
    record = todo.store.active()
    assert record is not None and record.command == "Record/add_note" and record.suspended_agent

    # The reply runs the step. The planning frame then asks the model to finish (its close), and the turn
    # asks for its own finish once the planning command answers: two finishes are scripted. The finish
    # check accepts the frame's finish (the approved run is archived under its step; see the regression
    # test below), so the frame does not re-plan.
    todo.script.steps = _finish() * 2
    done = todo.ctx.process_turn("yes")
    assert member.notes == before + ["card renewed"]       # the step ran, once
    assert record.state == "executed" and record.outcome == "success"
    assert done.status is TurnStatus.COMPLETED
    assert todo.store.active() is None


def test_an_approved_planning_step_run_is_seen_by_the_finish_check(todo, annotated_library, monkeypatch):
    # fix-3rsq.approval-finish-check: the approved add_note runs through the approval reply
    # (approval_binding._execute_stored), not the agent's execute path. Nothing held its run, and the step
    # was not completed, so the finish check read the step as not run and sent the planning frame back to
    # re-plan. The reply must hold the run under the suspended step and complete that step; then every
    # finish check made during the reply's turn accepts the frame's finish.
    from fastworkflow.observation_offloading import finish_check

    notes: list = []
    real_review = finish_check.review_finish

    def spy(**kwargs):
        note = real_review(**kwargs)
        notes.append(note)
        return note

    monkeypatch.setattr(finish_check, "review_finish", spy)
    todo.wf.record = todo.library
    member = todo.library.members["m1"]
    before = list(member.notes)
    _ask(todo, "note_member <member>Ann Lee</member> <text>card renewed</text>", steps=_finish())
    notes.clear()
    todo.script.steps = _finish() * 2
    todo.ctx.process_turn("yes")
    assert member.notes == before + ["card renewed"]
    assert notes, "the planning frame's finish was never checked"
    assert all(note is None for note in notes), notes


def test_a_standing_approval_does_not_run_a_delegatable_command_on_an_unattended_turn(todo, annotated_library, grant_root):
    # fix-d6no (owner 2026-11-13): this test used to assert the opposite (the grant ran the command with no human).
    # An unattended run never uses a channel-surface standing grant: it is not run and not asked (bd fix-ftdl.7.6).
    # (fix-qjne.11: proved on the delegatable command, the only one a grant applies to.)
    _ask(todo, "show_notes", steps=_finish())
    todo.ctx.process_turn("always")
    out = _ask(todo, "show_notes", steps=_finish(), unattended=True)
    assert ab.active_approvals(todo.wf) == []                  # not run, and not kept (bd fix-ftdl.7.6)
    assert _approval_spans(todo, "used") == []
    not_asked = _approval_spans(todo, "not_asked")
    assert len(not_asked) == 1 and not_asked[0]["approval_command"] == "Record/show_notes"
    assert out is not None


# ----------------------------------------------------------------------
# fix-d6no: unattended before the grant (channel), and the origin in the channel grant identity
# ----------------------------------------------------------------------

CHAT = Surface(kind="channel", origin="chat")
CLI = Surface()          # the interactive CLI's default surface: the legacy key


def _always_on(todo, surface):
    """Ask the delegatable command, then answer "always", both on *surface*; the binding runs once and the
    grant is recorded (bd fix-qjne.11: only a delegatable command takes an ``always``)."""
    out = _ask(todo, "show_notes", steps=_finish(), approval_surface=surface)
    assert out.status is TurnStatus.AWAITING_USER
    return todo.ctx.process_turn("always", approval_surface=surface)


@pytest.mark.parametrize("surface", [CLI, CHAT], ids=["cli", "chat"])
def test_a_channel_grant_runs_attended_and_is_not_asked_unattended(todo, annotated_library, grant_root, surface):
    _always_on(todo, surface)
    attended = _ask(todo, "show_notes", steps=_finish(), approval_surface=surface)
    assert attended.status is TurnStatus.COMPLETED               # attended: no prompt, the grant ran it
    unattended = _ask(todo, "show_notes", steps=_finish(), unattended=True,
                      approval_surface=surface)
    assert len(_approval_spans(todo, "not_asked")) == 1           # not run, and not kept (bd fix-ftdl.7.6)
    assert ab.active_approvals(todo.wf) == []
    assert len(_approval_spans(todo, "used")) == 1               # only the attended call used the grant
    assert unattended is not None


def test_the_unattended_policy_cells_for_a_channel_surface_and_a_verified_host():
    # Delegatable (the grant's mechanism); the REQUIRED_ALWAYS cells are in the test of that name above (fix-qjne.11).
    for origin_surface in (Surface(), Surface(kind="channel", origin="chat")):
        granted = ab.decide(REQUIRED_UNTIL_DELEGATED, origin=Origin.AGENT, surface=origin_surface, unattended=True,
                            awaiting_on_channel=False, auto_granted=True)
        assert granted.outcome is Outcome.NOT_ASKED and granted.reason == "no human channel"
        attended = ab.decide(REQUIRED_UNTIL_DELEGATED, origin=Origin.AGENT, surface=origin_surface, unattended=False,
                             awaiting_on_channel=False, auto_granted=True)
        assert attended.outcome is Outcome.RUN
    host = Surface("host", verified=True, uid="u")
    assert ab.decide(REQUIRED_UNTIL_DELEGATED, origin=Origin.AGENT, surface=host, unattended=True,
                     awaiting_on_channel=False, auto_granted=True).outcome is Outcome.RUN
    assert ab.decide(REQUIRED_UNTIL_DELEGATED, origin=Origin.AGENT, surface=host, unattended=True,
                     awaiting_on_channel=False).outcome is Outcome.NOT_ASKED


def test_the_channel_identity_names_the_origin_and_the_cli_keeps_the_legacy_key(grant_root):
    from fastworkflow import approval_grants as ag
    user = ag.local_user()
    assert ag.identity_key(CLI, "c1") == f"channel:{user}"                       # legacy: stored grants stay valid
    assert ag.identity_key(CHAT, "c1") == f"channel:{user}:chat"
    assert ag.identity_key(Surface(kind="channel", origin="somewhere"), "c1") is None   # unknown origin: none


@pytest.mark.parametrize("made_on, used_on", [(CHAT, CLI), (CLI, CHAT)], ids=["chat-then-cli", "cli-then-chat"])
def test_a_grant_made_in_one_origin_does_not_approve_in_the_other(todo, annotated_library, grant_root, made_on,
                                                                   used_on):
    _always_on(todo, made_on)
    out = _ask(todo, "show_notes", steps=_finish(), approval_surface=used_on)
    assert out.status is TurnStatus.AWAITING_USER                # asks: the other origin's grant does not apply
    assert todo.store.active() is not None


def test_a_legacy_channel_grant_still_approves_on_the_cli_and_not_on_the_chat_origin(todo, annotated_library,
                                                                                    grant_root):
    from fastworkflow import approval_grants as ag
    # A grant stored before origins existed: identity channel:<user>, nothing else about it.
    ag.record(f"channel:{ag.local_user()}", state_paths_workflow_id(todo), "Record/show_notes", REQUIRED_UNTIL_DELEGATED)
    chat = _ask(todo, "show_notes", steps=_finish(), approval_surface=CHAT)
    assert chat.status is TurnStatus.AWAITING_USER
    todo.ctx.process_turn("no", approval_surface=CHAT)
    cli = _ask(todo, "show_notes", steps=_finish(), approval_surface=CLI)
    assert cli.status is TurnStatus.COMPLETED


def state_paths_workflow_id(todo) -> str:
    import fastworkflow.session.state_paths as state_paths
    return state_paths.workflow_id(todo.wf.folderpath)


def test_listing_and_revoking_see_only_the_callers_own_origin(todo, annotated_library, grant_root):
    _always_on(todo, CHAT)
    todo.script.steps = []
    on_chat = todo.ctx.process_turn("/approvals", approval_surface=CHAT).answer
    assert "Record/show_notes" in on_chat and "on this channel" in on_chat
    assert "another channel are not shown" in on_chat
    on_cli = todo.ctx.process_turn("/approvals", approval_surface=CLI).answer
    assert "Record/show_notes" not in on_cli and "No standing approvals" in on_cli
    assert "another channel are not shown" in on_cli
    # The CLI cannot revoke the chat's grant; the chat can.
    refused = todo.ctx.process_turn("/revoke_approval show_notes", approval_surface=CLI).answer
    assert "No standing approval for show_notes" in refused
    assert "Record/show_notes" in todo.ctx.process_turn("/approvals", approval_surface=CHAT).answer
    done = todo.ctx.process_turn("/revoke_approval show_notes", approval_surface=CHAT).answer
    assert "is revoked" in done
    assert "Record/show_notes" not in todo.ctx.process_turn("/approvals", approval_surface=CHAT).answer


@pytest.mark.parametrize("surface", [CLI, CHAT], ids=["cli", "chat"])
def test_the_reply_to_always_says_how_to_list_and_revoke_the_grant(todo, annotated_library, grant_root, surface):
    out = _always_on(todo, surface)
    shown = "\n".join(o.command_response.response or "" for o in out.command_outputs)
    assert "/approvals" in shown and "/revoke_approval" in shown       # in the command's reply the user reads
    assert todo.store.records[-1].state == "executed"                 # the command still ran once
