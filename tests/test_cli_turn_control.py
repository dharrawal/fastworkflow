"""The CLI's turn control (fix-1ma8): stop waiting on a running turn (Ctrl-C, a per-turn deadline) and cancel a
pending prompt (``//cancel``), on a real ``ChatSession`` and its worker thread over the bundled hello_world test
workflow, with no model call.

The engine cannot abort a turn in flight, so what these tests pin is what the CLI guarantees instead: the wait ends,
the abandoned turn's output never reaches a later prompt, the turn itself still runs to its end, an abandoned turn
that asks a question is cancelled rather than left blocked, and the session is usable for the next message.

The slow command is the workflow's own ``add_two_numbers`` (``FW_TEST_ADD_SLEEP_SECONDS`` makes it sleep, and
``FW_TEST_ADD_CALL_LOG`` records each run). A question is a blocking ``ask_user`` read made by the same function in a
copy of the workflow (``FW_TEST_ADD_ASK``): the real worker thread, the real ``_execute_message`` conversion of a
cancelled question.
"""
from __future__ import annotations

import _thread
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.chat_session import ChatSession
from fastworkflow.run.turn_wait import (
    WaitOutcome, interrupted_message, timed_out_message, wait_for_turn,
)

HELLO = Path(__file__).parent / "hello_world_workflow"
ADD = "/add_two_numbers first_num={a} second_num={b}"
_COPY_IGNORE = shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__", "versions")

_ASK_HOOK = '''
    if os.environ.get("FW_TEST_ADD_ASK"):
        from fastworkflow.workflow_agent import _ask_user_tool
        import fastworkflow as _fw
        _ask_user_tool("Which one?", _fw.chat_session)
'''


_SLOW_INPUT = """
        from pydantic import model_validator as _mv

        @_mv(mode="before")
        @classmethod
        def _slow_before_the_command_runs(cls, value):
            # Runs while the parameters are bound, i.e. before any approval prompt or missing-value prompt.
            import os as _os, time as _t
            _t.sleep(float(_os.environ.get("FW_TEST_PRE_SLEEP", "0") or 0))
            return value
"""


def _copy_workflow(folder: Path, *, annotate_approval: bool = False, ask_hook: bool = False) -> str:
    shutil.copytree(HELLO, folder, ignore=_COPY_IGNORE, symlinks=True)
    # The trained versions are ~550 MB: link them rather than copy them. Nothing here trains.
    os.symlink(HELLO / "___command_info" / "versions", folder / "___command_info" / "versions")
    if ask_hook:
        path = folder / "application" / "add_two_numbers.py"
        text = path.read_text()
        marker = "    return a + b"
        assert text.count(marker) == 1
        path.write_text(text.replace(marker, _ASK_HOOK.lstrip("\n") + marker, 1))
    path = folder / "_commands" / "add_two_numbers.py"
    text = path.read_text()
    marker = "    class Input(BaseModel):\n"
    assert text.count(marker) == 1
    path.write_text(text.replace(marker, marker + _SLOW_INPUT.lstrip("\n"), 1))
    if annotate_approval:
        path = folder / "_commands" / "add_two_numbers.py"
        text = path.read_text()
        marker = "class ResponseGenerator:\n"
        assert text.count(marker) == 1
        path.write_text(text.replace(
            marker,
            "@fastworkflow.user_approval_for_execution(fastworkflow.UserApproval.REQUIRED_ALWAYS)\n" + marker, 1))
    return str(folder)


@pytest.fixture(scope="module")
def ask_workflow(tmp_path_factory) -> str:
    return _copy_workflow(tmp_path_factory.mktemp("cli_turn_ask") / "wf", ask_hook=True)


@pytest.fixture(scope="module")
def approval_workflow(tmp_path_factory) -> str:
    return _copy_workflow(tmp_path_factory.mktemp("cli_turn_approval") / "wf", annotate_approval=True)


@pytest.fixture(autouse=True)
def _package_chat_session_restored():
    """``_start`` binds the session to ``fastworkflow.chat_session`` (as ``fastworkflow run`` does); put back what the
    name held, so a later test file still finds the submodule there."""
    missing = object()
    previous = vars(fastworkflow).get("chat_session", missing)
    yield
    if previous is missing:
        vars(fastworkflow).pop("chat_session", None)
    else:
        fastworkflow.chat_session = previous


def _start(workflow_path: str) -> ChatSession:
    fastworkflow.init({})
    session = ChatSession(run_as_agent=False)
    fastworkflow.chat_session = session
    session.start_workflow(workflow_path, workflow_id_str=str(uuid.uuid4()), keep_alive=True)
    return session


@pytest.fixture
def session(ask_workflow):
    chat = _start(ask_workflow)
    yield chat
    chat._keep_alive = False


@pytest.fixture
def call_log(tmp_path, monkeypatch) -> Path:
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", str(log))
    return log


def _calls(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


def _send(chat: ChatSession, message: str, **wait) -> WaitOutcome:
    chat.user_message_queue.put(message)
    return wait_for_turn(chat, lambda event: None, **wait)


def _outputs(chat: ChatSession) -> list:
    found = []
    while True:
        try:
            found.append(chat.command_output_queue.get_nowait())
        except queue.Empty:
            return found


def _until(predicate, seconds: float = 15.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


def _interrupt_main_after(seconds: float) -> threading.Timer:
    """A real KeyboardInterrupt in this (the main) thread, as Ctrl-C raises it."""
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.start()
    return timer


def _is_idle(chat: ChatSession) -> bool:
    return not chat.interrupted_turn_running and chat.command_trace_queue.empty() \
        and chat.command_output_queue.empty()


# ----------------------------------------------------------------------
# The default path: no Ctrl-C, no deadline
# ----------------------------------------------------------------------


def test_a_turn_with_no_interrupt_and_no_deadline_ends_done_with_its_one_output(session, call_log):
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    outputs = _outputs(session)
    assert [o.command_response.response for o in outputs] == ['Response: {"sum_of_two_numbers":5.0}']
    assert _calls(call_log) == 1
    assert not session.interrupted_turn_running
    assert session.command_trace_queue.empty()


def test_a_deadline_longer_than_the_turn_changes_nothing(session):
    assert _send(session, ADD.format(a=2, b=3), timeout_seconds=60) is WaitOutcome.DONE
    assert len(_outputs(session)) == 1


def test_abandoning_a_turn_that_has_just_ended_abandons_nothing(session):
    """Ctrl-C landing after the turn's sentinel is queued: the result is read as usual, nothing is discarded."""
    session.user_message_queue.put(ADD.format(a=2, b=3))
    _until(lambda: any(item is None for item in list(session.command_trace_queue.queue)))
    assert session.abandon_turn() is False
    assert not session.interrupted_turn_running
    assert wait_for_turn(session, lambda event: None) is WaitOutcome.DONE
    assert len(_outputs(session)) == 1


# ----------------------------------------------------------------------
# Ctrl-C
# ----------------------------------------------------------------------


def test_ctrl_c_returns_to_the_prompt_and_the_late_result_is_never_delivered(session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "1.5")
    timer = _interrupt_main_after(0.4)
    try:
        started = time.monotonic()
        outcome = _send(session, ADD.format(a=1, b=1))
        waited = time.monotonic() - started
    finally:
        timer.cancel()
    assert outcome is WaitOutcome.INTERRUPTED
    assert waited < 1.2, "the wait must end at the interrupt, not at the turn's end"
    assert session.interrupted_turn_running
    assert _outputs(session) == [] and session.command_trace_queue.empty()

    # The turn runs to its end in the worker: its side effect happens ...
    _until(lambda: not session.interrupted_turn_running)
    assert _calls(call_log) == 1
    # ... and nothing of it is delivered afterwards.
    time.sleep(0.2)
    assert _outputs(session) == [] and session.command_trace_queue.empty()

    # The session is usable: the next message gets its own, and only its own, result.
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0")
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert [o.command_response.response for o in _outputs(session)] == ['Response: {"sum_of_two_numbers":5.0}']
    assert _calls(call_log) == 2


def test_a_message_sent_while_the_interrupted_turn_runs_waits_for_it_and_gets_only_its_own_result(
        session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "1.0")
    timer = _interrupt_main_after(0.3)
    try:
        assert _send(session, ADD.format(a=1, b=1)) is WaitOutcome.INTERRUPTED
    finally:
        timer.cancel()
    assert session.interrupted_turn_running

    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert [o.command_response.response for o in _outputs(session)] == ['Response: {"sum_of_two_numbers":5.0}']
    assert _calls(call_log) == 2, "both turns ran: the interrupted one to its end, in order"
    _until(lambda: _is_idle(session))


def test_interrupting_the_wait_for_an_earlier_interrupted_turn_drops_both_results(session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0.8")
    for delay in (0.2, 0.2):
        timer = _interrupt_main_after(delay)
        try:
            assert _send(session, ADD.format(a=1, b=1)) is WaitOutcome.INTERRUPTED
        finally:
            timer.cancel()
    _until(lambda: not session.interrupted_turn_running)
    assert _calls(call_log) == 2
    time.sleep(0.2)
    assert _outputs(session) == [] and session.command_trace_queue.empty()
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0")
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert len(_outputs(session)) == 1


def test_an_interrupted_turn_that_asks_a_question_is_cancelled_not_left_blocked(session, call_log, monkeypatch):
    """The worker would block on ask_user for an answer nobody knows is wanted, and the next message would become
    that answer. The abandoned turn's question read raises the engine's CommandCancelledError instead, which
    ``_execute_message`` turns into a failed "Command cancelled" output and a cleared suspension."""
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "1.0")
    monkeypatch.setenv("FW_TEST_ADD_ASK", "1")
    timer = _interrupt_main_after(0.3)
    try:
        assert _send(session, ADD.format(a=1, b=1)) is WaitOutcome.INTERRUPTED
    finally:
        timer.cancel()
    _until(lambda: not session.interrupted_turn_running)
    assert not session.user_message_queue.question_pending
    assert _outputs(session) == [] and session.command_trace_queue.empty(), "the question was never shown"

    monkeypatch.delenv("FW_TEST_ADD_ASK")
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0")
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert [o.command_response.response for o in _outputs(session)] == ['Response: {"sum_of_two_numbers":5.0}']


# ----------------------------------------------------------------------
# The per-turn deadline
# ----------------------------------------------------------------------


def test_the_deadline_stops_the_wait_and_the_late_result_is_dropped(session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "1.5")
    started = time.monotonic()
    outcome = _send(session, ADD.format(a=1, b=1), timeout_seconds=0.4)
    waited = time.monotonic() - started
    assert outcome is WaitOutcome.TIMED_OUT
    assert 0.35 <= waited < 1.2
    assert session.interrupted_turn_running

    _until(lambda: not session.interrupted_turn_running)
    assert _calls(call_log) == 1
    time.sleep(0.2)
    assert _outputs(session) == [] and session.command_trace_queue.empty()
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0")
    assert _send(session, ADD.format(a=2, b=3), timeout_seconds=30) is WaitOutcome.DONE


def test_the_messages_name_what_stopped_the_wait_and_what_is_guaranteed():
    assert "--timeout_seconds" in timed_out_message(2.5) and "2.5s" in timed_out_message(2.5)
    for text in (interrupted_message(), timed_out_message(30)):
        assert "still finishing" in text and "commands may still take effect" in text
        assert "will be discarded" in text and "next message runs after it" in text
    assert "\n" not in interrupted_message() and "\n" not in timed_out_message(30)


def test_run_accepts_the_deadline_flag_and_it_defaults_to_off():
    import argparse

    from fastworkflow.cli import add_run_parser

    parser = argparse.ArgumentParser()
    add_run_parser(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["run", "wf"]).timeout_seconds is None
    assert parser.parse_args(["run", "wf", "--timeout_seconds", "90"]).timeout_seconds == 90.0


# ----------------------------------------------------------------------
# //cancel: the chat page's Cancel
# ----------------------------------------------------------------------


def test_cancel_ends_a_pending_question_through_the_engines_cancelled_command(session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_ASK", "1")
    assert _send(session, ADD.format(a=2, b=3), timeout_seconds=30) is WaitOutcome.DONE
    question = _outputs(session)
    assert [o.command_response.response for o in question] == ["Which one?"]

    # Read at once, the worker may not have begun the blocking read yet: the cancel waits for it.
    assert session.cancel_pending_prompt() == "question"
    assert wait_for_turn(session, lambda event: None, timeout_seconds=30) is WaitOutcome.DONE
    cancelled = _outputs(session)
    assert [o.command_response.response for o in cancelled] == ["Command cancelled: the question was cancelled"]
    assert cancelled[0].command_response.success is False
    assert not session.user_message_queue.question_pending
    assert session.cancel_pending_prompt() is None, "nothing is left pending"
    assert _calls(call_log) == 1, "the command was entered (it asked); it did not go on to a result"

    monkeypatch.delenv("FW_TEST_ADD_ASK")
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert [o.command_response.response for o in _outputs(session)] == ['Response: {"sum_of_two_numbers":5.0}']
    assert _calls(call_log) == 2


def test_cancel_with_nothing_pending_does_nothing(session):
    assert session.cancel_pending_prompt() is None
    assert session.user_message_queue.empty()


def test_cancel_is_refused_while_an_interrupted_turn_is_still_finishing(session, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "1.0")
    assert _send(session, ADD.format(a=1, b=1), timeout_seconds=0.2) is WaitOutcome.TIMED_OUT
    assert session.cancel_pending_prompt() is None
    assert session.user_message_queue.empty()
    _until(lambda: not session.interrupted_turn_running)


def _reply_text(chat: ChatSession, message: str) -> tuple[str, bool]:
    assert _send(chat, message, timeout_seconds=30) is WaitOutcome.DONE
    outputs = _outputs(chat)
    assert len(outputs) == 1
    return outputs[0].command_response.response, outputs[0].command_response.success


def test_cancel_of_a_missing_value_prompt_is_what_typing_abort_does(session):
    prompt, _ = _reply_text(session, "/add_two_numbers first_num=2")
    assert "your request does not give the second num" in prompt and session._core.has_open_command()

    assert session.cancel_pending_prompt() == "clarification"
    assert wait_for_turn(session, lambda event: None, timeout_seconds=30) is WaitOutcome.DONE
    via_cancel = [o.command_response.response for o in _outputs(session)]
    assert not session._core.has_open_command()

    _reply_text(session, "/add_two_numbers first_num=2")
    typed, _ = _reply_text(session, "abort")
    assert via_cancel == [typed] == ["command aborted\n"]


def test_cancel_of_an_approval_is_what_typing_no_does(approval_workflow, call_log):
    chat = _start(approval_workflow)
    try:
        prompt, _ = _reply_text(chat, ADD.format(a=2, b=3))
        assert "approv" in prompt.lower() or "yes" in prompt.lower()
        assert _calls(call_log) == 0

        assert chat.cancel_pending_prompt() == "review"
        assert wait_for_turn(chat, lambda event: None, timeout_seconds=30) is WaitOutcome.DONE
        via_cancel = [o.command_response.response for o in _outputs(chat)]
        assert _calls(call_log) == 0, "a declined approval never runs the command"

        _reply_text(chat, ADD.format(a=2, b=3))
        typed, _ = _reply_text(chat, "no")
        assert via_cancel == [typed]
        assert chat.cancel_pending_prompt() is None
    finally:
        chat._keep_alive = False


# ----------------------------------------------------------------------
# An abandoned turn that ends with a prompt the user never saw
# ----------------------------------------------------------------------


def _sum_text() -> str:
    return 'Response: {"sum_of_two_numbers":5.0}'


def _abandon_a_turn_that_ends_in_an_approval(chat, monkeypatch, next_message_queued: bool) -> list:
    monkeypatch.setenv("FW_TEST_PRE_SLEEP", "0.7")
    assert _send(chat, ADD.format(a=2, b=3), timeout_seconds=0.2) is WaitOutcome.TIMED_OUT
    if next_message_queued:
        chat.user_message_queue.put("yes")  # queued while the abandoned turn is still running
    return []


def test_an_abandoned_turn_that_ends_in_an_approval_is_cancelled_before_the_next_message(
        approval_workflow, call_log, monkeypatch):
    """The prompt was never shown, so a following ``yes`` must not approve it: it is cancelled in the worker with the
    replies ``//cancel`` sends, and the ``yes`` is processed as a fresh message."""
    chat = _start(approval_workflow)
    try:
        _abandon_a_turn_that_ends_in_an_approval(chat, monkeypatch, next_message_queued=False)
        _until(lambda: not chat.interrupted_turn_running)
        assert _outputs(chat) == [] and chat.command_trace_queue.empty(), "its cancel output is dropped too"
        assert chat.cancel_pending_prompt() is None, "nothing is left pending"
        assert chat.take_cancelled_unseen_notice() is True
        assert chat.take_cancelled_unseen_notice() is False, "the notice is given once"

        monkeypatch.setenv("FW_TEST_PRE_SLEEP", "0")
        reply, _ = _reply_text(chat, "yes")
        assert _sum_text() not in reply
        assert _calls(call_log) == 0, "the gated command never ran"
    finally:
        chat._keep_alive = False


def test_the_cancel_comes_before_a_message_that_was_already_queued(approval_workflow, call_log, monkeypatch):
    chat = _start(approval_workflow)
    try:
        _abandon_a_turn_that_ends_in_an_approval(chat, monkeypatch, next_message_queued=True)
        assert wait_for_turn(chat, lambda event: None, timeout_seconds=60) is WaitOutcome.DONE
        replies = [o.command_response.response for o in _outputs(chat)]
        assert len(replies) == 1 and _sum_text() not in replies[0], "only the queued message's own result"
        assert _calls(call_log) == 0, "the queued yes approved nothing"
        assert chat.take_cancelled_unseen_notice() is True
        # The queued "yes" matched no command, so the which-command question it opened is what is pending now
        # (one pending record since bd fix-o1lg: it is saved with the session and can be cancelled like any
        # question). The approval is not pending: the queued yes approved nothing, as asserted above.
        assert chat.cancel_pending_prompt() == "clarification"
    finally:
        chat._keep_alive = False


def test_an_abandoned_turn_that_ends_at_a_missing_value_prompt_does_not_take_the_next_message_as_the_value(
        session, call_log, monkeypatch):
    """No model-free path to a plan review exists in the bundled test workflows (a review comes from the agent's
    sub-task gate, which needs the model), so the pending kind covered here is the open command (a missing value);
    a review is classified by the same ``_pending_cancel_reply`` as an approval."""
    # A slow first turn holds the worker; the missing-value message queues behind it and is abandoned too, so it
    # runs (and leaves its prompt open) while abandoned.
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0.7")
    assert _send(session, ADD.format(a=1, b=1), timeout_seconds=0.2) is WaitOutcome.TIMED_OUT
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0")
    assert _send(session, "/add_two_numbers first_num=2", timeout_seconds=0.2) is WaitOutcome.TIMED_OUT
    session.user_message_queue.put("3")  # would be the missing value if the prompt were still open
    calls_before = _calls(call_log)
    assert wait_for_turn(session, lambda event: None, timeout_seconds=60) is WaitOutcome.DONE
    outputs = _outputs(session)
    assert len(outputs) == 1
    assert _calls(call_log) == calls_before == 1, "only the first turn ran; 3 did not complete the command"
    assert session.take_cancelled_unseen_notice() is True


def test_a_turn_with_nothing_left_pending_is_not_touched_when_abandoned(session, call_log, monkeypatch):
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "0.6")
    assert _send(session, ADD.format(a=1, b=1), timeout_seconds=0.2) is WaitOutcome.TIMED_OUT
    _until(lambda: not session.interrupted_turn_running)
    assert session.take_cancelled_unseen_notice() is False


# ----------------------------------------------------------------------
# A reply to an approval reaches the terminal (the fix to the CLI hang)
# ----------------------------------------------------------------------


def test_a_reply_to_an_approval_is_delivered_once(approval_workflow, call_log):
    chat = _start(approval_workflow)
    try:
        _reply_text(chat, ADD.format(a=2, b=3))
        assert _send(chat, "no", timeout_seconds=30) is WaitOutcome.DONE
        assert len(_outputs(chat)) == 1
        time.sleep(0.3)
        assert _outputs(chat) == [] and chat.command_trace_queue.empty(), "no second delivery"
        assert chat._gate.sentinels == 1
    finally:
        chat._keep_alive = False


def test_a_turn_that_delivered_its_own_output_gets_nothing_extra(session):
    assert _send(session, ADD.format(a=2, b=3)) is WaitOutcome.DONE
    assert len(_outputs(session)) == 1
    time.sleep(0.3)
    assert _outputs(session) == [] and session.command_trace_queue.empty()
    assert session._gate.sentinels == 1


# ----------------------------------------------------------------------
# The real entry point, `python -m fastworkflow.run`, over piped stdin
# ----------------------------------------------------------------------


class _Terminal:
    """A real ``fastworkflow.run`` process on piped stdin/stdout (prompt_toolkit accepts a non-tty input and reads
    lines from it), with the output collected by a reader thread."""

    def __init__(self, workflow: str, tmp: Path):
        import subprocess
        import sys

        env_file = tmp / "run.env"
        env_file.write_text(f"LLM_AGENT=mistral/mistral-small-latest\nFASTWORKFLOW_STATE_ROOT={tmp / 'state'}\n")
        passwords = tmp / "run.passwords.env"
        passwords.write_text("LITELLM_API_KEY_AGENT=unused\n")
        self.chunks: list[str] = []
        self.process = subprocess.Popen(
            [sys.executable, "-m", "fastworkflow.run", workflow, str(env_file), str(passwords), "--assistant"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while chunk := self.process.stdout.read1(4096):
            self.chunks.append(chunk.decode(errors="replace"))

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def wait_for(self, marker: str, after: int = 0, seconds: float = 240) -> int:
        """Block until *marker* appears after offset *after*; the offset just past it."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if (at := self.text.find(marker, after)) >= 0:
                return at + len(marker)
            if self.process.poll() is not None and self.text.find(marker, after) < 0:
                break
            time.sleep(0.1)
        raise AssertionError(f"{marker!r} never appeared after offset {after}; output so far:\n{self.text[-3000:]}")

    def send(self, line: str) -> int:
        offset = len(self.text)
        self.process.stdin.write((line + "\n").encode())
        self.process.stdin.flush()
        return offset


def test_run_main_end_to_end_over_piped_stdin_cancel_then_a_normal_command_then_exit(tmp_path):
    workflow = _copy_workflow(tmp_path / "wf")
    terminal = _Terminal(workflow, tmp_path)
    try:
        at = terminal.wait_for("Tips")
        assert "//cancel" in terminal.text[:at + 600], "the tips line names //cancel"
        at = terminal.wait_for("User >", at)
        time.sleep(1.0)

        at = terminal.send("//cancel")
        terminal.wait_for("Nothing to cancel.", at)
        time.sleep(1.0)

        at = terminal.send("/add_two_numbers first_num=2")
        terminal.wait_for("your request does not give the second num", at)
        time.sleep(1.5)

        at = terminal.send("//cancel")
        terminal.wait_for("command aborted", at)
        time.sleep(1.5)

        at = terminal.send("/add_two_numbers first_num=2 second_num=3")
        terminal.wait_for("sum_of_two_numbers", at)
        assert "5.0" in terminal.text[at:]
        time.sleep(1.5)

        terminal.send("//exit")
        assert terminal.process.wait(timeout=60) == 0
    finally:
        if terminal.process.poll() is None:
            terminal.process.kill()
