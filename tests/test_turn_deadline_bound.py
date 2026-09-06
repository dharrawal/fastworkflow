"""The turn deadline must actually bound the extraction call (ido-mn1.6.33).

Before this, `react.py` tested the 1800 s safety envelope only at the top of
each ReAct step, so it could not interrupt a call already in flight, and no
`external_operations.operation` wrapped the agent run at all.  `_finish()` then
ran the extraction inside `for attempt in range(EXTRACT_PARSE_ATTEMPTS)` and
re-derived its bound on every attempt, with `clamp_timeout()` having no deadline
to clamp against.  A cell reaching `finish` just under the wall could therefore
spend a further 802 s composing, or 2,406 s across three parse attempts, against
a 1,980 s watchdog — and was then recorded as infrastructure-missing after being
paid for.

These tests pin the three halves of the fix, and the one thing it must not be:

  * the whole agent turn runs under one operation, so the derived extraction
    timeout is clamped to what is LEFT of the turn deadline and says so;
  * an attempt the remaining deadline cannot buy is not started at all — no
    provider call, no model call charged to the budget — and comes back as
    infrastructure that keeps its evidence, never as a task failure;
  * the check is re-made before every parse retry, because attempt three has
    less of the turn left than attempt one did;
  * and none of it reads the arm.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import dspy
import pytest

from fastworkflow import external_operations, result_handles
from fastworkflow.turn_budget import LogicalTurnBudget
from fastworkflow.typed_failure import CODE_EXTRACTION_TRUNCATED, TypedFailure
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.react import (
    DEADLINE_INSUFFICIENT_CAUSE,
    EXTRACT_GEN_TOKENS_PER_SECOND,
    EXTRACT_MIN_TOKENS,
    EXTRACT_TIMEOUT_SAFETY,
    EXTRACT_TTFT_ALLOWANCE_SECONDS,
    extraction_bound,
    fastWorkflowReAct,
    minimum_viable_extraction_seconds,
)


PRESENTED_ROWS = "row\n" * 8192  # 32 KiB, the shipped presentation cap


# ---------------------------------------------------------------------------
# 0. The deadline itself is one number, resolved in one place
# ---------------------------------------------------------------------------

def test_the_watchdog_and_the_in_turn_clamp_resolve_the_same_deadline():
    """Two resolvers are two numbers that can disagree, and the whole defect was
    that the watchdog's number never reached the code it was supposed to bound."""
    from fastworkflow.run_fastapi_mcp import turns

    assert turns.TURN_DEADLINE_ENV_VAR == (
        external_operations.TURN_DEADLINE_ENV_VAR
    )
    assert turns.DEFAULT_TURN_DEADLINE_SECONDS == (
        external_operations.DEFAULT_TURN_DEADLINE_SECONDS
    )
    assert turns.resolve_turn_deadline_seconds() == (
        external_operations.resolve_turn_deadline_seconds()
    )


def test_the_deadline_is_read_per_call_and_validated(monkeypatch):
    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "4700")
    assert external_operations.resolve_turn_deadline_seconds() == 4700.0
    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "0")
    with pytest.raises(ValueError):
        external_operations.resolve_turn_deadline_seconds()
    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "soon")
    with pytest.raises(ValueError):
        external_operations.resolve_turn_deadline_seconds()
    monkeypatch.delenv("FW_TURN_DEADLINE_SECONDS")
    assert external_operations.resolve_turn_deadline_seconds() == 900.0


# ---------------------------------------------------------------------------
# 1. The clamp is real and recorded
# ---------------------------------------------------------------------------

def test_the_minimum_viable_attempt_is_the_floor_limit_at_the_measured_rate():
    expected = (
        math.ceil(
            EXTRACT_MIN_TOKENS / EXTRACT_GEN_TOKENS_PER_SECOND
            * EXTRACT_TIMEOUT_SAFETY
        )
        + EXTRACT_TTFT_ALLOWANCE_SECONDS
    )
    assert minimum_viable_extraction_seconds() == expected == 206.0


def _derived_timeout_for(max_tokens: int) -> float:
    """The timeout `extraction_bound` derives for a limit, from the constants
    rather than a literal, so a re-calibration of the constants (2026-09-05:
    BYTES_PER_TOKEN 4.0 -> 2.0, ceiling 20480 -> 36864) cannot silently turn
    these tests into a record of the previous numbers."""
    import math

    from fastworkflow.utils.react import (
        EXTRACT_GEN_TOKENS_PER_SECOND,
        EXTRACT_TIMEOUT_SAFETY,
        EXTRACT_TTFT_ALLOWANCE_SECONDS,
    )

    return float(
        math.ceil(max_tokens / EXTRACT_GEN_TOKENS_PER_SECOND * EXTRACT_TIMEOUT_SAFETY)
        + EXTRACT_TTFT_ALLOWANCE_SECONDS
    )


def test_a_turn_deadline_clamps_the_derived_bound_and_records_both_numbers():
    """Clamping SHORTENS the answer; it does not lose it.  Both numbers are kept
    because `timeout_s` alone cannot tell a bound that was shortened from one
    that was never long."""
    unbounded = extraction_bound(32768, 1200)
    assert unbounded.timeout_clamped is False
    assert unbounded.deadline_remaining_s is None
    assert unbounded.deadline_insufficient is False
    expected = _derived_timeout_for(unbounded.max_tokens)
    assert unbounded.timeout_s == unbounded.derived_timeout_s == expected
    assert expected > 400.0

    with external_operations.operation("model.agent", seconds=400.0):
        bound = extraction_bound(32768, 1200)

    assert bound.derived_timeout_s == expected
    assert bound.timeout_clamped is True
    assert 0 < bound.timeout_s <= 400.0
    assert bound.deadline_remaining_s is not None
    assert bound.deadline_remaining_s <= 400.0
    # Still worth attempting: 400 s buys far more than one floor-sized answer.
    assert bound.deadline_insufficient is False
    assert bound.as_evidence()["derived_timeout_s"] == expected
    assert bound.as_evidence()["deadline_insufficient"] is False


def test_a_deadline_below_one_floor_attempt_is_marked_insufficient():
    with external_operations.operation("model.agent", seconds=100.0):
        bound = extraction_bound(0, 0)
    assert bound.deadline_insufficient is True
    assert bound.timeout_clamped is True


# ---------------------------------------------------------------------------
# 2. `_finish` refuses a doomed attempt rather than paying for it
# ---------------------------------------------------------------------------

class _FakePresented:
    def __init__(self, text: str):
        self._text = text
        self.trimmed = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def field_bytes(self) -> int:
        return len(self._text.encode("utf-8"))

    def as_evidence(self) -> dict:
        return {"handles": [], "bytes": self.field_bytes,
                "field_bytes": self.field_bytes, "max_bytes": 32768,
                "trimmed": False, "unresolved": []}


def _parse_error():
    from dspy.utils.exceptions import AdapterParseError

    return AdapterParseError(
        adapter_name="ChatAdapter",
        signature=dspy.Signature("question -> final_answer"),
        lm_response="unparseable",
    )


def _finish_agent(*, presented_text: str, behaviours):
    """A `fastWorkflowReAct` wired to a fake extract with scripted behaviours.

    `behaviours` is one entry per call: either "ok" or "parse-error".  Same
    `__new__` construction the other `_finish` tests use, for the same reason —
    it exercises `_finish` and nothing it does not need.
    """
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent._exhausted_last_run = False
    agent._step_seals = {}
    agent._safety_envelope = None
    agent.presentation_commands = frozenset()
    agent.policy_point = None

    calls: list[dict] = []

    class _Extract:
        def __call__(self, **kwargs):
            behaviour = behaviours[len(calls)]
            calls.append(dict(kwargs))
            if behaviour == "parse-error":
                raise _parse_error()
            dspy_utils._record_finish_reasons(
                SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])
            )
            return {"reasoning": "r", "final_answer": "the listing"}

    agent.extract = _Extract()
    agent._resolve_presented_results = lambda trajectory: (  # type: ignore[method-assign]
        result_handles.PresentedResults(max_bytes=32768)
        if not presented_text
        else _FakePresented(presented_text)
    )
    agent._consult_finish_policy = lambda *a, **k: None  # type: ignore[method-assign]
    return agent, calls


def _trajectory() -> dict:
    return {
        "thought_0": "first",
        "tool_name_0": "t",
        "tool_args_0": {},
        "observation_0": "o",
        "thought_1": "closing",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }


def test_an_insufficient_deadline_makes_no_provider_call_at_all():
    """The point is that the attempt is not started.  A killed HTTP attempt
    returns nothing and is charged for; refusing returns the same nothing for
    free, and says which it was."""
    agent, calls = _finish_agent(
        presented_text=PRESENTED_ROWS, behaviours=["ok"]
    )
    budget = LogicalTurnBudget(iteration_limit=5)
    before = budget.model_calls_consumed

    with external_operations.operation("model.agent", seconds=60.0):
        result = agent._finish(_trajectory(), {"user_query": "q"}, budget)

    assert calls == []
    assert budget.model_calls_consumed == before


def test_an_insufficient_deadline_is_infrastructure_and_never_a_task_failure():
    agent, _ = _finish_agent(presented_text=PRESENTED_ROWS, behaviours=["ok"])
    with external_operations.operation("model.agent", seconds=60.0):
        result = agent._finish(
            _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
        )

    assert result.extraction_truncated is True
    # The typed reason the durable-turn schema pins, so every existing reader
    # classifies this as infrastructure with no new rule ...
    assert result.extraction_truncated_reason == CODE_EXTRACTION_TRUNCATED
    assert isinstance(result.extraction_failure, TypedFailure)
    assert result.extraction_failure.code == CODE_EXTRACTION_TRUNCATED
    assert result.extraction_failure.disposition == "transient"
    # ... and the cause beside it, because a cut answer and an answer that was
    # never composed are not the same finding.
    assert result.extraction_truncated_cause == DEADLINE_INSUFFICIENT_CAUSE
    assert DEADLINE_INSUFFICIENT_CAUSE in result.extraction_failure.detail
    # `failure` means the turn failed, and a turn the harness ran out of clock
    # on did not.  `censored` ends a plan and discards leaf work.
    assert getattr(result, "failure", None) is None
    assert getattr(result, "censored", False) is False


def test_an_insufficient_deadline_keeps_the_evidence_and_states_the_absence():
    agent, _ = _finish_agent(presented_text=PRESENTED_ROWS, behaviours=["ok"])
    trajectory = _trajectory()
    with external_operations.operation("model.agent", seconds=60.0):
        result = agent._finish(
            trajectory, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
        )

    assert result.trajectory == trajectory
    assert result.presented_results["field_bytes"] == len(
        PRESENTED_ROWS.encode("utf-8")
    )
    assert result.extraction_bound["deadline_insufficient"] is True
    # No answer was composed, so nothing may be presented as one.
    assert result.final_answer != "the listing"
    assert DEADLINE_INSUFFICIENT_CAUSE in result.final_answer


def test_a_sufficient_deadline_still_composes_under_a_clamped_bound():
    """The refusal must not swallow the case the clamp exists for."""
    agent, calls = _finish_agent(
        presented_text=PRESENTED_ROWS, behaviours=["ok"]
    )
    with external_operations.operation("model.agent", seconds=500.0):
        result = agent._finish(
            _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
        )

    assert len(calls) == 1
    assert result.final_answer == "the listing"
    assert result.extraction_bound["timeout_clamped"] is True
    assert result.extraction_bound["deadline_insufficient"] is False
    assert calls[0]["config"]["timeout"] <= 500.0
    assert not getattr(result, "extraction_truncated", False)


# ---------------------------------------------------------------------------
# 3. The parse-retry loop re-checks the clock before every retry
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def test_the_parse_retry_loop_rechecks_the_remaining_time_before_each_retry(
    monkeypatch,
):
    """Attempt three has less of the turn left than attempt one did.  A bound
    computed once stops being true after the first, and a check made once stops
    being true with it."""
    clock = _FakeClock()
    monkeypatch.setattr(
        external_operations, "time", SimpleNamespace(monotonic=clock.monotonic)
    )
    agent, calls = _finish_agent(
        presented_text="", behaviours=["parse-error", "ok", "ok"]
    )
    budget = LogicalTurnBudget(iteration_limit=5)

    with external_operations.operation("model.agent", seconds=1000.0):
        # The first attempt has the whole 1000 s and runs; it fails to parse.
        # By the time the retry is considered only 40 s are left, which cannot
        # buy even a floor-sized answer, so the retry is refused rather than
        # dispatched.
        original = agent.extract

        class _Advancing:
            def __call__(self, **kwargs):
                clock.now = 960.0
                return original(**kwargs)

        agent.extract = _Advancing()
        result = agent._finish(_trajectory(), {"user_query": "q"}, budget)

    assert len(calls) == 1
    assert result.extraction_truncated_cause == DEADLINE_INSUFFICIENT_CAUSE
    assert budget.model_calls_consumed == 1


# ---------------------------------------------------------------------------
# 4. Arm invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "presentation_commands",
    [
        pytest.param(frozenset(), id="flat-arm-a"),
        pytest.param(frozenset({"show_holders"}), id="task-first-leaf-arm-b"),
        pytest.param(
            frozenset({"show_holders", "open_portrait", "list_members"}),
            id="packed-leaf-arm-c",
        ),
    ],
)
def test_the_three_arm_call_sites_take_the_same_deadline_decision(
    presentation_commands,
):
    """No term of the decision reads the arm, the skill or the plan: it reads
    the derived bound and the clock.  Three call sites that agree by inspection
    and not by construction is exactly how an arm-dependent treatment gets into
    a paired comparison."""
    refused, composed = [], []
    for seconds, sink in ((60.0, refused), (500.0, composed)):
        agent, calls = _finish_agent(
            presented_text=PRESENTED_ROWS, behaviours=["ok"]
        )
        agent.presentation_commands = presentation_commands
        with external_operations.operation("model.agent", seconds=seconds):
            result = agent._finish(
                _trajectory(),
                {"user_query": "q"},
                LogicalTurnBudget(iteration_limit=5),
            )
        sink.append((len(calls), result.extraction_bound["max_tokens"]))

    assert refused == [(0, extraction_bound(
        len(PRESENTED_ROWS.encode("utf-8")),
        len(b"first") + len(b"closing"),
        agent._trajectory_bytes(_trajectory()),
    ).max_tokens)]
    assert composed[0][0] == 1
    assert composed[0][1] == refused[0][1]


# ---------------------------------------------------------------------------
# 5. The operation exists at all: WEC opens it around the whole agent run
# ---------------------------------------------------------------------------

def _bare_context():
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._workflow_tool_agent = None
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = None
    ctx._trace_sink = None
    ctx._agent_dspy_context = lambda: (SimpleNamespace(model="m"), None)
    return ctx


def _run_agent_call(ctx, agent_call):
    import fastworkflow.workflow_execution_context as wec

    start, end = wec.tracing.start_span, wec.tracing.end_span
    wec.tracing.start_span = lambda *a, **k: None
    wec.tracing.end_span = lambda *a, **k: None
    try:
        return ctx._call_agent(agent_call)
    finally:
        wec.tracing.start_span, wec.tracing.end_span = start, end


def test_the_agent_run_happens_inside_a_turn_deadline_operation(monkeypatch):
    """This is the whole of (a): before it there was no operation on this path,
    so `clamp_timeout()` had nothing to clamp against however carefully the
    extraction derived its bound."""
    from fastworkflow.plan_execution import SafetyEnvelopeState

    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "1980")
    ctx = _bare_context()
    ctx._turn_safety_envelope = SafetyEnvelopeState(enabled=False)
    seen: list = []

    def agent_call():
        seen.append(external_operations.current_operation())
        return SimpleNamespace(final_answer="x")

    _run_agent_call(ctx, agent_call)

    assert seen and seen[0] is not None
    assert seen[0].kind == "model.agent"
    assert 1970.0 < seen[0].remaining() <= 1980.0


def test_the_operation_is_anchored_on_the_turn_and_not_on_each_leaf(monkeypatch):
    """A planned turn calls the agent once per leaf.  A fresh full deadline per
    leaf would bound each leaf and not the turn, which is not a bound at all."""
    import time as _time

    from fastworkflow.plan_execution import SafetyEnvelopeState

    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "1980")
    ctx = _bare_context()
    ctx._turn_safety_envelope = SafetyEnvelopeState(
        enabled=True,
        wall_time_limit_s=1800,
        started_at_epoch_s=_time.time() - 1700.0,
    )

    assert 270.0 < ctx._remaining_turn_deadline_seconds() <= 280.0

    # And an overrun turn gets a floor rather than a deadline in the past.
    ctx._turn_safety_envelope = SafetyEnvelopeState(
        enabled=True,
        wall_time_limit_s=1800,
        started_at_epoch_s=_time.time() - 9999.0,
    )
    assert ctx._remaining_turn_deadline_seconds() == 1.0


def test_the_extraction_inside_a_turn_operation_is_clamped_to_what_is_left(
    monkeypatch,
):
    """End to end for the defect: a turn that reaches `finish` with 500 s left
    on a 1,980 s deadline composes under a clamped bound, and records it."""
    from fastworkflow.plan_execution import SafetyEnvelopeState
    import time as _time

    monkeypatch.setenv("FW_TURN_DEADLINE_SECONDS", "1980")
    ctx = _bare_context()
    ctx._turn_safety_envelope = SafetyEnvelopeState(
        enabled=True,
        wall_time_limit_s=1800,
        started_at_epoch_s=_time.time() - 1480.0,
    )
    agent, calls = _finish_agent(
        presented_text=PRESENTED_ROWS, behaviours=["ok"]
    )
    ctx._workflow_tool_agent = agent

    result = _run_agent_call(
        ctx,
        lambda: agent._finish(
            _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
        ),
    )

    assert len(calls) == 1
    # 32 KiB of payload plus this trajectory's closing thought, at the measured
    # rate, is far more generation time than the ~500 s left.
    derived = result.extraction_bound["derived_timeout_s"]
    assert derived == _derived_timeout_for(result.extraction_bound["max_tokens"])
    assert derived > 500.0
    assert result.extraction_bound["timeout_clamped"] is True
    assert result.extraction_bound["timeout_s"] <= 500.0
