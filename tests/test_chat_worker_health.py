"""EXP-011: bounded typed failures instead of retries, replays and dead workers.

Fault matrix rows 13-16 (requirements §12.2) plus the two removals that make
them possible:

* row 13 — the model cannot produce a parseable decision;
* row 14 — extraction fails *after* tools have already run;
* row 15 — a turn raises (store access, backend, anything unhandled);
* row 16 — the worker dies, and a saturated / stuck worker.

The acceptance criteria these hold (FW-REQ-008): each fault produces a bounded,
typed failure; a following independent turn runs or gets an explicit
session-failed response; and **no test here waits for a generic timeout after
worker death** — every wait in this file is either sub-second or bounded by an
explicit failure.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.turn_budget import LogicalTurnBudget
from fastworkflow.typed_failure import (
    CODE_ADAPTER_PARSE,
    CODE_EXTRACTION_FAILED,
    CODE_WORKER_FAILED,
    CODE_WORKER_STUCK,
    ControlSignal,
    TurnFailedError,
    TypedFailure,
    classify_exception,
)
from fastworkflow.utils.react import (
    DECISION_PARSE_ATTEMPTS,
    EXTRACT_PARSE_ATTEMPTS,
    fastWorkflowReAct,
)
from fastworkflow.worker_health import (
    TurnRequest,
    WorkerDeadError,
    WorkerHealth,
    WorkerState,
    unwrap_request,
)


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def _parse_error(message="unparseable"):
    """A real ``AdapterParseError``, built the way dspy builds one.

    Its ``__init__`` formats a message from ``signature.output_fields``, so a
    ``None`` signature raises while constructing the error — a stub with the
    one attribute it reads is what makes the fixture the actual exception type
    the decision phase catches, rather than a look-alike.
    """
    from dspy.utils.exceptions import AdapterParseError

    return AdapterParseError(
        adapter_name="ChatAdapter",
        signature=SimpleNamespace(output_fields={"next_tool_name": object()}),
        lm_response=message,
    )


def _bare_agent(**tools):
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.max_iters = 5
    agent._budget = LogicalTurnBudget(iteration_limit=5)
    agent._step_seals = {}
    agent.inputs = {}
    agent.current_trajectory = {}
    agent._suspended = None
    agent._exhausted_last_run = False
    agent.react = object()
    agent.extract = object()
    agent.tools = tools
    return agent


# ----------------------------------------------------------------------
# Row 13 — the model cannot produce a decision
# ----------------------------------------------------------------------


def test_the_decision_phase_retries_the_parse_and_then_fails_typed():
    """Bounded retry where nothing has executed yet (arch §8.4 phase 1)."""
    agent = _bare_agent(finish=lambda: "done")
    calls = []

    def always_unparseable(module, trajectory, **input_args):
        calls.append(module)
        raise _parse_error()

    agent._call_with_potential_trajectory_truncation = always_unparseable
    budget = LogicalTurnBudget(iteration_limit=5)

    with pytest.raises(TurnFailedError) as caught:
        agent._decide({}, {"query": "x"}, budget)

    assert len(calls) == DECISION_PARSE_ATTEMPTS
    assert caught.value.failure.code == CODE_ADAPTER_PARSE
    assert caught.value.failure.disposition == "permanent"
    # Each provider attempt is charged to the turn (arch §8.4), so a turn cannot
    # buy unbounded model calls by failing to parse.
    assert budget.model_calls_consumed == DECISION_PARSE_ATTEMPTS


def test_a_recovered_parse_costs_only_the_failed_attempts():
    """The recovery the removed whole-agent replay was buying, at its own phase."""
    agent = _bare_agent(finish=lambda: "done")
    attempts = {"n": 0}

    def flaky(module, trajectory, **input_args):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _parse_error()
        return SimpleNamespace(
            next_thought="ok", next_tool_name="finish", next_tool_args={}
        )

    agent._call_with_potential_trajectory_truncation = flaky
    budget = LogicalTurnBudget(iteration_limit=5)
    pred = agent._decide({}, {"query": "x"}, budget)

    assert pred.next_tool_name == "finish"
    assert budget.model_calls_consumed == 2


def test_a_non_parse_error_is_not_retried_by_the_decision_phase():
    """Only the parse belongs to this phase; everything else propagates."""
    agent = _bare_agent(finish=lambda: "done")
    calls = []

    def boom(module, trajectory, **input_args):
        calls.append(1)
        raise RuntimeError("provider is down")

    agent._call_with_potential_trajectory_truncation = boom
    with pytest.raises(RuntimeError):
        agent._decide({}, {"query": "x"}, LogicalTurnBudget(iteration_limit=5))
    assert len(calls) == 1


# ----------------------------------------------------------------------
# Row 14 — extraction fails after tools have already run
# ----------------------------------------------------------------------


def test_a_failed_extraction_returns_a_typed_failure_with_the_completed_work():
    """Arch §8.4: it never invokes the agent loop again.

    This is the row the whole-agent replay got wrong: the tools had already
    run, and re-entering the loop re-executed them.
    """
    agent = _bare_agent(do_it=lambda: "did it", finish=lambda: "done")
    tool_calls = []

    def script(module, trajectory, **input_args):
        if module is agent.extract:
            raise _parse_error("cannot extract")
        if tool_calls:
            return SimpleNamespace(
                next_thought="stop", next_tool_name="finish", next_tool_args={}
            )
        tool_calls.append(1)
        return SimpleNamespace(
            next_thought="act", next_tool_name="do_it", next_tool_args={}
        )

    agent._call_with_potential_trajectory_truncation = script
    result = agent.forward(query="x", budget=LogicalTurnBudget(iteration_limit=5))

    assert result.failure.code == CODE_EXTRACTION_FAILED
    assert result.failure.disposition == "permanent"
    # The completed steps ride the failure, so nothing has to re-derive them
    # from a trajectory a later phase may have truncated.
    assert len(result.failure.completed_work) == 2
    assert all("logical_call_key" in seal for seal in result.failure.completed_work)
    # And the tool ran exactly once, across every extraction attempt.
    assert len(tool_calls) == 1


def test_the_extraction_snapshot_is_the_same_for_every_attempt():
    """A truncating retry must not change what the next attempt summarizes."""
    agent = _bare_agent(finish=lambda: "done")
    seen = []

    def script(module, trajectory, **input_args):
        if module is agent.extract:
            seen.append(dict(trajectory))
            raise _parse_error()
        return SimpleNamespace(
            next_thought="stop", next_tool_name="finish", next_tool_args={}
        )

    agent._call_with_potential_trajectory_truncation = script
    agent.forward(query="x", budget=LogicalTurnBudget(iteration_limit=5))

    assert len(seen) == EXTRACT_PARSE_ATTEMPTS
    assert all(snapshot == seen[0] for snapshot in seen)


def test_a_control_signal_from_a_tool_never_becomes_an_observation():
    """Arch §8.4: caught before the generic except, and re-raised."""
    signal = ControlSignal(
        TypedFailure(disposition="outcome-unknown", code="reconciliation-required")
    )

    def unsafe():
        raise signal

    agent = _bare_agent(unsafe=unsafe, finish=lambda: "done")
    agent._call_with_potential_trajectory_truncation = (
        lambda module, trajectory, **kw: SimpleNamespace(
            next_thought="go", next_tool_name="unsafe", next_tool_args={}
        )
    )

    with pytest.raises(ControlSignal):
        agent.forward(query="x", budget=LogicalTurnBudget(iteration_limit=5))


def test_every_tool_call_carries_a_stable_logical_call_key():
    """The same decision at the same step of the same turn derives one key."""
    key = fastWorkflowReAct.logical_call_key("turn-1", 3, "do_it", {"a": 1})
    assert key == fastWorkflowReAct.logical_call_key("turn-1", 3, "do_it", {"a": 1})
    assert key != fastWorkflowReAct.logical_call_key("turn-1", 4, "do_it", {"a": 1})
    assert key != fastWorkflowReAct.logical_call_key("turn-2", 3, "do_it", {"a": 1})
    assert key != fastWorkflowReAct.logical_call_key("turn-1", 3, "do_it", {"a": 2})


# ----------------------------------------------------------------------
# Row 15 — a turn raises, and rows 16 — the worker
# ----------------------------------------------------------------------


class _FailingCore:
    """A WEC stand-in whose turns fail on demand."""

    def __init__(self, failures: list[BaseException | None]):
        self._failures = list(failures)
        self.calls = 0
        self.current_turn_key = "turn-under-test"
        # Read by ChatSession.get_active_workflow on the loop's way out.
        self.app_workflow = None

    def _execute_message(self, message):
        self.calls += 1
        failure = self._failures.pop(0) if self._failures else None
        if failure is not None:
            raise failure
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response=f"ok:{message}")
        )

    def process_action(self, action):
        return self._execute_message(str(action))

    def finalize_turn_for_observability(self, output):
        return None


def _session_with_core(core, keep_alive=True):
    """A ChatSession wired to a stand-in core, without starting a workflow."""
    from fastworkflow.chat_session import ChatSession

    session = ChatSession.__new__(ChatSession)
    session._core = core
    session._user_message_queue = Queue()
    session._command_output_queue = Queue()
    session._command_trace_queue = Queue()
    session._status = None
    session._chat_worker = None
    session._current_workflow = None
    session._keep_alive = keep_alive
    session._startup_command = ""
    session._startup_action = None
    session._workflow_is_complete = False
    session._health = WorkerHealth(state=WorkerState.RUNNING)
    return session


def test_a_failing_turn_is_terminal_and_the_worker_survives(initialized_fastworkflow):
    """FW-REQ-008 clause 3 and its acceptance criterion, in one test.

    Row 15: the turn raises. What used to happen is that the exception left the
    message loop and killed the worker; what has to happen is a terminal failed
    turn and a live worker that runs the next one.
    """
    core = _FailingCore([RuntimeError("state store unavailable"), None])
    session = _session_with_core(core)

    first = TurnRequest(payload="explodes")
    second = TurnRequest(payload="works")
    session._user_message_queue.put(first)
    session._user_message_queue.put(second)

    worker = threading.Thread(target=session._run_workflow_loop, daemon=True)
    worker.start()

    with pytest.raises(TurnFailedError):
        first.wait(timeout=5)
    assert first.failure.disposition == "permanent"

    # The following independent turn runs — the acceptance criterion.
    assert second.wait(timeout=5).command_response.response == "ok:works"
    assert core.calls == 2
    assert session.health.is_alive
    assert session.health.turns_failed == 1

    session._status = __import__(
        "fastworkflow.chat_session", fromlist=["SessionStatus"]
    ).SessionStatus.STOPPING
    session._user_message_queue.put(TurnRequest(payload="drain"))
    worker.join(timeout=5)


def test_the_failed_turn_is_published_on_the_transport(initialized_fastworkflow):
    """A raw-queue caller still gets an answer, output before sentinel."""
    core = _FailingCore([ValueError("boom")])
    session = _session_with_core(core)
    session._user_message_queue.put("explodes")

    worker = threading.Thread(target=session._run_workflow_loop, daemon=True)
    worker.start()

    output = session._command_output_queue.get(timeout=5)
    assert output.command_response.success is False
    assert output.command_response.artifacts["failure"]["disposition"] == "permanent"
    assert session._command_trace_queue.get(timeout=5) is None

    session._status = __import__(
        "fastworkflow.chat_session", fromlist=["SessionStatus"]
    ).SessionStatus.STOPPING
    session._user_message_queue.put("drain")
    worker.join(timeout=5)


def test_a_dead_worker_rejects_submissions_promptly():
    """FW-REQ-008 clause 5. No timeout is waited out anywhere in this test."""
    core = _FailingCore([])
    session = _session_with_core(core)
    session._health.record_worker_failure(
        TypedFailure(disposition="permanent", code=CODE_WORKER_FAILED, detail="gone")
    )

    started = time.monotonic()
    with pytest.raises(WorkerDeadError) as caught:
        session.submit("anything")
    assert time.monotonic() - started < 0.5
    assert caught.value.failure.code == CODE_WORKER_FAILED

    with pytest.raises(WorkerDeadError):
        session.receive_turn(timeout=5)
    assert time.monotonic() - started < 1.0


def test_worker_death_fails_every_queued_envelope():
    """The gap the envelopes close: a queued request with nowhere to be told."""
    session = _session_with_core(_FailingCore([]))
    queued = [TurnRequest(payload=f"m{i}") for i in range(3)]
    for request in queued:
        session._user_message_queue.put(request)
    session._user_message_queue.put("a raw one, which has no delivery path")

    failure = TypedFailure(
        disposition="permanent", code=CODE_WORKER_FAILED, detail="worker died"
    )
    assert session._fail_queued_requests(failure) == 3
    for request in queued:
        assert request.is_done
        with pytest.raises(TurnFailedError):
            request.wait(timeout=0)


def test_a_stuck_worker_stays_poisoned_and_keeps_its_ownership():
    """Arch §13.3: a timed-out join() clears nothing and claims no termination."""
    health = WorkerHealth(state=WorkerState.RUNNING)
    health.mark_stuck("join timed out")

    assert health.is_poisoned
    assert not health.is_alive
    assert health.state is WorkerState.STUCK
    # The classification is `outcome-unknown`, not `permanent`: the thread may
    # still be inside a call that returns, and saying otherwise would invite a
    # retry on top of work that might still be running.
    assert health.rejection_failure().disposition == "outcome-unknown"
    assert health.rejection_failure().code == CODE_WORKER_STUCK

    # Neither a heartbeat nor a stop can relabel it healthy or stopped.
    health.beat(WorkerState.RUNNING)
    assert health.state is WorkerState.STUCK
    health.set_state(WorkerState.STOPPED)
    assert health.state is WorkerState.STUCK


def test_health_is_readable_without_a_thread_stack():
    """FW-REQ-008 clause 4."""
    health = WorkerHealth(state=WorkerState.RUNNING)
    snapshot = health.snapshot()
    assert snapshot["state"] == "running"
    assert snapshot["last_failure"] is None
    assert snapshot["heartbeat_age_seconds"] >= 0

    health.record_turn_failure(classify_exception(ValueError("x")), turn_key="t1")
    snapshot = health.snapshot()
    assert snapshot["turns_failed"] == 1
    assert snapshot["failed_turn_key"] == "t1"
    assert snapshot["last_failure"]["disposition"] == "permanent"


def test_the_legacy_queue_adapter_still_accepts_a_raw_message():
    """Arch §13.3 keeps it; it just cannot deliver a failure."""
    payload, envelope = unwrap_request("plain message")
    assert payload == "plain message"
    assert envelope is None

    request = TurnRequest(payload="wrapped")
    payload, envelope = unwrap_request(request)
    assert payload == "wrapped"
    assert envelope is request


# ----------------------------------------------------------------------
# Row 16 (FastAPI half) — the watchdog, the epoch fence, and readiness
# ----------------------------------------------------------------------


def _registry(**kwargs):
    from fastworkflow.run_fastapi_mcp.turns import TurnRegistry

    return TurnRegistry(**kwargs)


def _execution(registry, channel_id="chan", *, epoch=1, deadline_seconds=0.0):
    from datetime import timedelta

    from fastworkflow.run_fastapi_mcp.turns import TurnExecution, _now

    execn = TurnExecution(
        turn_key=fastworkflow.mint_turn_key(),
        channel_id=channel_id,
        kind="invoke_agent",
        idempotency_key="k",
        epoch=epoch,
        deadline_at=_now() + timedelta(seconds=deadline_seconds),
    )
    registry._by_key[execn.turn_key] = execn
    registry._active_by_channel[channel_id] = execn.turn_key
    registry._epoch_by_channel[channel_id] = epoch
    return execn


def test_an_overdue_execution_is_lost_but_keeps_its_channel(monkeypatch):
    """Arch §13.4: never clear the active pointer merely because a deadline elapsed."""
    import asyncio

    from fastworkflow.run_fastapi_mcp import turns as turns_mod

    registry = _registry()

    async def scenario():
        execn = _execution(registry, deadline_seconds=-1000)
        assert execn.is_overdue

        lost = registry.sweep()

        assert [e.turn_key for e in lost] == [execn.turn_key]
        assert execn.exec_state is turns_mod.ExecState.LOST
        # Unknown, not failed: nothing here learned what the work did.
        assert execn.failure.disposition == "outcome-unknown"
        assert execn.failure.code == CODE_WORKER_STUCK
        # Waiters are released — the whole point.
        assert execn.done_event.is_set()
        # And the channel is still owned by it, so conflicting work is blocked.
        assert registry._active_by_channel["chan"] == execn.turn_key
        assert registry.stuck_count == 1

    asyncio.run(scenario())


def test_a_healthy_execution_is_not_swept():
    import asyncio

    async def scenario():
        registry = _registry()
        execn = _execution(registry, deadline_seconds=600)
        assert registry.sweep() == []
        assert not execn.is_terminal

    asyncio.run(scenario())


def test_a_stale_epoch_cannot_publish():
    """The process-local fence: ownership moved, so this thread is not the owner."""
    import asyncio

    async def scenario():
        registry = _registry()
        execn = _execution(registry, epoch=1, deadline_seconds=600)
        assert registry.owns_channel(execn)

        # A later execution takes the channel.
        registry._epoch_by_channel["chan"] = 2
        assert not registry.owns_channel(execn)

    asyncio.run(scenario())


def test_readiness_goes_false_when_stuck_work_exhausts_capacity():
    import asyncio

    async def scenario():
        registry = _registry(max_stuck_executions=2)
        assert registry.readiness()["ready"] is True

        for i in range(2):
            _execution(registry, channel_id=f"c{i}", deadline_seconds=-1000)
        registry.sweep()

        readiness = registry.readiness()
        assert readiness["ready"] is False
        assert readiness["stuck_executions"] == 2
        # Coarse counts only — nothing here names a channel or a principal.
        assert set(readiness) == {
            "ready",
            "active_executions",
            "stuck_executions",
            "max_stuck_executions",
            "admission_closed",
            "watchdog_running",
        }

    asyncio.run(scenario())


def test_a_lost_execution_is_not_relost_or_reterminalized():
    import asyncio

    async def scenario():
        registry = _registry()
        execn = _execution(registry, deadline_seconds=-1000)
        assert registry.mark_lost(execn, "first") is True
        first_reason = execn.terminalization_reason
        assert registry.mark_lost(execn, "second") is False
        assert execn.terminalization_reason == first_reason

    asyncio.run(scenario())
