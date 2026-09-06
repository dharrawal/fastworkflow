"""Contract tests for the extraction call's per-call completion limit (ido-mn1.6.10).

EXP-028 Gate 4 v4 ran every model call at one ``max_tokens`` of 4096. For the
react step that is correct and ido-mn1.6.4 kept it. For the EXTRACTION step —
the composition call whose ``final_answer`` arm A returns verbatim and arms B/C
concatenate per leaf with no further model call — it ended ``finish_reason=length``
69 times, and the cut text was stored as the turn's answer in 17/19 arm-A cells,
5/19 B and 13/19 C.

These tests pin the four things that fix has to be, and one thing it must not be:

  * the limit is DERIVED per call from the payload the runtime measured, and can
    only ever raise the deployment constant, never lower it;
  * it is applied as a per-call kwarg to the extraction call ALONE, because the
    same LM object serves the react step and is shared across turns and threads;
  * the timeout is derived from the limit, since a raised limit under the old
    attempt bound just moves the cut from `max_tokens` to a killed HTTP attempt;
  * a call the provider stops at the limit is marked as harness truncation with
    its answer KEPT, and never as a task failure;
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
    EXTRACT_BYTES_PER_TOKEN,
    EXTRACT_GEN_TOKENS_PER_SECOND,
    EXTRACT_MAX_TOKENS_CEILING,
    EXTRACT_MIN_TOKENS,
    EXTRACT_PROSE_ALLOWANCE_TOKENS,
    EXTRACT_RENDER_FACTOR,
    EXTRACT_TIMEOUT_SAFETY,
    EXTRACT_TTFT_ALLOWANCE_SECONDS,
    ExtractionBound,
    extraction_bound,
    fastWorkflowReAct,
)


# ---------------------------------------------------------------------------
# 1. The formula
# ---------------------------------------------------------------------------

def _expected_tokens(field_bytes: int, thought_bytes: int) -> int:
    derived = (
        math.ceil((field_bytes + thought_bytes) / EXTRACT_BYTES_PER_TOKEN
                  * EXTRACT_RENDER_FACTOR)
        + EXTRACT_PROSE_ALLOWANCE_TOKENS
    )
    return min(max(derived, EXTRACT_MIN_TOKENS), EXTRACT_MAX_TOKENS_CEILING)


def test_empty_payload_gets_the_deployment_floor_and_not_less():
    """The derivation may only ever RAISE the limit.

    A turn that presents nothing derives `PROSE_ALLOWANCE_TOKENS` alone, which is
    less than the 4096 every call gets today. Handing it that number would make
    this change shorten answers that fit now — the opposite of the defect.
    """
    bound = extraction_bound(0, 0)
    assert bound.max_tokens == EXTRACT_MIN_TOKENS == 4096
    assert bound.derived_tokens == EXTRACT_PROSE_ALLOWANCE_TOKENS
    assert bound.floor_applied is True
    assert bound.ceiling_applied is False


def test_the_limit_is_the_stated_formula_of_its_stated_inputs():
    bound = extraction_bound(8000, 500)
    assert bound.derived_tokens == (
        math.ceil(8500 / EXTRACT_BYTES_PER_TOKEN * EXTRACT_RENDER_FACTOR)
        + EXTRACT_PROSE_ALLOWANCE_TOKENS
    )
    assert bound.max_tokens == bound.derived_tokens == _expected_tokens(8000, 500)
    assert bound.max_tokens > EXTRACT_MIN_TOKENS
    assert bound.floor_applied is False
    assert bound.ceiling_applied is False
    # The inputs travel with the number, so a reader can recompute it.
    assert bound.field_bytes == 8000
    assert bound.thought_bytes == 500
    assert bound.render_factor == EXTRACT_RENDER_FACTOR
    assert bound.prose_allowance_tokens == EXTRACT_PROSE_ALLOWANCE_TOKENS


def test_the_limit_is_monotone_in_the_payload():
    """More rows to render can never buy a smaller limit."""
    limits = [extraction_bound(size, 0).max_tokens for size in (0, 4096, 16384, 65536)]
    assert limits == sorted(limits)


def test_the_ceiling_binds_only_beyond_the_presentation_cap():
    """The ceiling is the formula evaluated at the largest field that can exist.

    `presented_max_bytes()` is 32 KiB by default and the closing thought is
    bounded at 1200 chars, so the biggest real field derives 18,008 tokens —
    under the ceiling. The ceiling is a backstop for someone raising
    FW_PRESENTED_RESULT_MAX_BYTES without re-deriving it, and this pins that it
    is exactly that and not a limit on ordinary traffic.
    """
    at_cap = extraction_bound(result_handles.presented_max_bytes(), 1200)
    assert at_cap.ceiling_applied is False
    assert at_cap.max_tokens < EXTRACT_MAX_TOKENS_CEILING

    beyond = extraction_bound(10 * result_handles.presented_max_bytes(), 0)
    assert beyond.ceiling_applied is True
    assert beyond.max_tokens == EXTRACT_MAX_TOKENS_CEILING


@pytest.mark.parametrize(
    "var,value,check",
    [
        ("FW_EXTRACT_RENDER_FACTOR", "4.0", lambda b: b.render_factor == 4.0),
        (
            "FW_EXTRACT_PROSE_ALLOWANCE_TOKENS",
            "3000",
            lambda b: b.prose_allowance_tokens == 3000,
        ),
        ("FW_EXTRACT_MIN_TOKENS", "9000", lambda b: b.max_tokens >= 9000),
        (
            "FW_EXTRACT_MAX_TOKENS_CEILING",
            "5000",
            lambda b: b.max_tokens == 5000,
        ),
    ],
)
def test_every_constant_is_env_overridable(monkeypatch, var, value, check):
    monkeypatch.setenv(var, value)
    assert check(extraction_bound(40000, 0))


def test_a_mis_set_constant_falls_back_rather_than_disabling_the_bound(monkeypatch):
    """"0" almost always means a mis-set variable, and a zero completion limit
    is not a shorter answer — it is no answer at all."""
    monkeypatch.setenv("FW_EXTRACT_RENDER_FACTOR", "0")
    monkeypatch.setenv("FW_EXTRACT_MIN_TOKENS", "not-a-number")
    bound = extraction_bound(8000, 500)
    assert bound.render_factor == EXTRACT_RENDER_FACTOR
    assert bound.max_tokens == _expected_tokens(8000, 500)


def test_a_ceiling_below_the_floor_still_yields_the_floor(monkeypatch):
    monkeypatch.setenv("FW_EXTRACT_MAX_TOKENS_CEILING", "100")
    assert extraction_bound(0, 0).max_tokens == EXTRACT_MIN_TOKENS


# ---------------------------------------------------------------------------
# 2. The timeout derived from the limit
# ---------------------------------------------------------------------------

def test_the_timeout_covers_generating_the_limit_at_the_measured_rate():
    bound = extraction_bound(60000, 0)
    expected = (
        math.ceil(
            bound.max_tokens / EXTRACT_GEN_TOKENS_PER_SECOND * EXTRACT_TIMEOUT_SAFETY
        )
        + EXTRACT_TTFT_ALLOWANCE_SECONDS
    )
    assert bound.timeout_s == pytest.approx(expected)
    assert bound.timeout_clamped is False


def test_the_role_attempt_bound_is_a_floor_and_never_a_ceiling(monkeypatch):
    """A derived timeout SHORTER than the bound the role already promised would
    be a regression wearing a derivation's clothes."""
    monkeypatch.setattr(
        dspy.settings,
        "lm",
        SimpleNamespace(role_policy=SimpleNamespace(attempt_seconds=900.0)),
        raising=False,
    )
    assert extraction_bound(0, 0).timeout_s == 900.0


def test_the_timeout_is_clamped_to_the_turn_deadline_and_says_so():
    """A derived bound that would outlive the turn is not a bound."""
    with external_operations.operation("model.extraction", seconds=5.0):
        bound = extraction_bound(60000, 0)
    assert bound.timeout_clamped is True
    assert 0 < bound.timeout_s <= 5.0


def test_the_call_config_is_exactly_the_two_provider_kwargs():
    bound = extraction_bound(8000, 500)
    assert bound.as_call_config() == {
        "max_tokens": bound.max_tokens,
        "timeout": bound.timeout_s,
    }


# ---------------------------------------------------------------------------
# 3. The per-call kwargs reach the provider, and reach nothing else
# ---------------------------------------------------------------------------

class _RecordingLM(dspy.LM):
    """A `dspy.LM` that records the kwargs each call reaches the provider with."""

    def __init__(self, finish_reason: str = "stop"):
        super().__init__(model="openai/gpt-4o-mini", api_key="x")
        self.seen: list[dict] = []
        self._finish_reason = finish_reason

    def forward(self, prompt=None, messages=None, **kwargs):
        self.seen.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "[[ ## reasoning ## ]]\nr\n\n"
                            "[[ ## final_answer ## ]]\nthe listing\n\n"
                            "[[ ## completed ## ]]"
                        ),
                        tool_calls=None,
                    ),
                    finish_reason=self._finish_reason,
                )
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            model="m",
        )


def test_a_per_call_config_reaches_lm_forward_for_that_call_only():
    """`dspy.Predict(config=...)` is the seam: `_forward_preprocess` merges it
    into `lm_kwargs`, which the adapter hands to `LM.forward`. Mutating the
    shared LM instead would change the react step's limit too, and would race."""
    lm = _RecordingLM()
    with dspy.context(lm=lm):
        cot = dspy.ChainOfThought("question -> final_answer")
        cot(question="q", config={"max_tokens": 9999, "timeout": 42.5})
        cot(question="q")

    assert lm.seen[0]["max_tokens"] == 9999
    assert lm.seen[0]["timeout"] == 42.5
    # The second call, with no config, carries neither.
    assert "max_tokens" not in lm.seen[1]
    assert "timeout" not in lm.seen[1]


def _role_bound(finish_reason: str = "stop", role: str = "LLM_AGENT"):
    """A `_RecordingLM` wearing the same role-bound subclass `get_lm` builds."""
    lm = dspy_utils._role_bound_lm_class(_RecordingLM)(finish_reason=finish_reason)
    lm.role_policy = dspy_utils.role_policy(role)
    return lm


def test_a_per_call_timeout_bounds_one_attempt_and_is_not_multiplied():
    """Before ido-mn1.6.10 a caller-supplied `timeout` took the un-bounded path
    and `dspy.LM.forward` spent it on EACH of `1 + num_retries` attempts, so
    asking for a 600 s bound asked for 2400 s. A caller that computed a bound
    from the work in front of it is entitled to have that bound be the bound."""
    lm = _role_bound()
    lm(messages=[{"role": "user", "content": "hi"}], timeout=600.0)

    assert len(lm.seen) == 1
    # One attempt, and it got the WHOLE derived time — not a quarter of it.
    assert lm.seen[0]["timeout"] == pytest.approx(600.0, abs=0.01)
    assert 600.0 > dspy_utils.role_policy("LLM_AGENT").attempt_seconds


def test_the_role_call_bound_still_holds_when_no_per_call_bound_is_given():
    lm = _role_bound()
    lm(messages=[{"role": "user", "content": "hi"}])
    assert lm.seen[0]["timeout"] == pytest.approx(
        dspy_utils.role_policy("LLM_AGENT").attempt_seconds, abs=0.01
    )


def test_finish_reasons_are_captured_only_inside_the_collector():
    lm = _role_bound(finish_reason="length")

    with dspy_utils.capture_finish_reasons() as reasons:
        lm(messages=[{"role": "user", "content": "hi"}])
    assert reasons == ["length"]

    # Outside the block nothing is collected, and nothing raises.
    lm(messages=[{"role": "user", "content": "hi"}])
    assert reasons == ["length"]


# ---------------------------------------------------------------------------
# 4. `_finish` applies the bound, records it, and marks a truncated answer
# ---------------------------------------------------------------------------

PRESENTED_ROWS = "row\n" * 3000


def _finish_agent(*, presented_text: str, finish_reasons: list[str]):
    """A `fastWorkflowReAct` built without `__init__`, wired to a fake extract.

    Same construction the suspend/resume tests use, for the same reason: this
    exercises `_finish` and nothing it does not need.
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
            calls.append(dict(kwargs))
            for reason in finish_reasons:
                dspy_utils._record_finish_reasons(
                    SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason=reason)]
                    )
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


class _FakePresented:
    """A stand-in with the two properties `_finish` reads off `PresentedResults`."""

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


def _trajectory(thought: str = "closing") -> dict:
    return {
        "thought_0": "first",
        "tool_name_0": "t",
        "tool_args_0": {},
        "observation_0": "o",
        "thought_1": thought,
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }


def test_finish_passes_the_derived_limit_to_the_extraction_call_only():
    agent, calls = _finish_agent(
        presented_text=PRESENTED_ROWS, finish_reasons=["stop"]
    )
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )

    assert len(calls) == 1
    config = calls[0]["config"]
    expected = extraction_bound(
        len(PRESENTED_ROWS.encode("utf-8")),
        len(b"first") + len(b"closing"),
        agent._trajectory_bytes(_trajectory()),
    )
    assert config["max_tokens"] == expected.max_tokens
    assert config["max_tokens"] > EXTRACT_MIN_TOKENS
    assert config["timeout"] == pytest.approx(expected.timeout_s)
    assert result.final_answer == "the listing"
    # And the derived timeout survived the `model.extraction` deadline class,
    # whose 300 s default was chosen against a fixed 4096-token limit and would
    # otherwise become the binding constraint the derivation just removed.
    assert config["timeout"] > 300.0
    assert result.extraction_bound["timeout_clamped"] is False


def test_the_react_step_keeps_the_deployment_constant():
    """ido-mn1.6.4 deliberately did NOT raise the react step's limit: a step that
    overruns is wasting tokens, and a larger cap lengthens the waste. The bound
    lands on one call, so nothing may appear in the react module's config."""
    agent = fastWorkflowReAct(
        __import__(
            "fastworkflow.workflow_agent", fromlist=["WorkflowAgentSignature"]
        ).WorkflowAgentSignature,
        tools=[lambda: "x"],
    )
    assert "max_tokens" not in agent.react.config
    assert "timeout" not in agent.react.config
    assert "max_tokens" not in agent.extract.predict.config
    assert "timeout" not in agent.extract.predict.config


def test_the_bound_is_recorded_as_evidence_on_the_prediction():
    agent, _ = _finish_agent(presented_text=PRESENTED_ROWS, finish_reasons=["stop"])
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )
    evidence = result.extraction_bound
    assert set(evidence) == {
        "max_tokens", "timeout_s", "field_bytes", "thought_bytes",
        "trajectory_bytes", "provider_max_output_tokens", "provider_cap_applied",
        "render_factor", "prose_allowance_tokens", "derived_tokens",
        "floor_applied", "ceiling_applied", "timeout_clamped",
        # ido-mn1.6.33: what the call ASKED for, what the turn had left, and
        # whether what was left could buy an attempt at all. `timeout_s` alone
        # cannot tell a bound that was shortened from one that was not.
        "derived_timeout_s", "deadline_remaining_s", "deadline_insufficient",
    }
    assert evidence["field_bytes"] == len(PRESENTED_ROWS.encode("utf-8"))
    assert evidence["render_factor"] == EXTRACT_RENDER_FACTOR


def test_a_stopped_extraction_carries_no_truncation_marker():
    agent, _ = _finish_agent(presented_text="", finish_reasons=["stop"])
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )
    assert not getattr(result, "extraction_truncated", False)
    assert getattr(result, "failure", None) is None


def test_a_truncated_extraction_keeps_the_answer_and_marks_the_turn():
    """The cut text IS the evidence of what the turn did. Dropping it would
    remove the only record of the work, and calling it a task failure would
    record the harness's limit as the agent's mistake."""
    agent, _ = _finish_agent(presented_text="", finish_reasons=["length"])
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )

    assert result.final_answer == "the listing"
    assert result.extraction_truncated is True
    assert result.extraction_truncated_reason == CODE_EXTRACTION_TRUNCATED
    assert isinstance(result.extraction_failure, TypedFailure)
    assert result.extraction_failure.code == CODE_EXTRACTION_TRUNCATED
    assert result.extraction_failure.disposition == "transient"
    # NOT a task failure, and NOT the safety envelope: `failure` is the field
    # that means "no answer", and `censored` ends a plan and discards leaf work.
    assert getattr(result, "failure", None) is None
    assert getattr(result, "censored", False) is False


def test_a_truncated_extraction_is_transient_and_never_permanent():
    """A permanent code here would license a reader to record the turn as a task
    failure, which is the one reading this classification exists to prevent."""
    agent, _ = _finish_agent(presented_text="", finish_reasons=["length"])
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )
    assert result.extraction_failure.retryable_by_disposition is True
    assert result.extraction_failure.disposition != "permanent"
    # The limit it ran into is named, so the next run can be sized against it.
    assert str(EXTRACT_MIN_TOKENS) in result.extraction_failure.detail


# ---------------------------------------------------------------------------
# 5. Arm invariance
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
def test_the_three_arm_call_sites_derive_the_same_limit(presentation_commands):
    """No term of the formula reads the arm, the skill or the plan: it reads the
    resolved field and the closing thought, which every arm produces the same
    way. Three call sites that agreed by inspection but not by construction is
    exactly how an arm-dependent treatment gets into a paired comparison."""
    agent, calls = _finish_agent(
        presented_text=PRESENTED_ROWS, finish_reasons=["stop"]
    )
    agent.presentation_commands = presentation_commands
    result = agent._finish(
        _trajectory(), {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )
    expected = extraction_bound(
        len(PRESENTED_ROWS.encode("utf-8")),
        len(b"first") + len(b"closing"),
        # 2026-09-05: the trajectory the call receives is the third term, and
        # every arm formats it the same way.
        agent._trajectory_bytes(_trajectory()),
    )
    assert expected.trajectory_bytes > 0
    assert calls[0]["config"]["max_tokens"] == expected.max_tokens
    assert result.extraction_bound["max_tokens"] == expected.max_tokens
    assert result.extraction_bound["field_bytes"] == expected.field_bytes


def test_the_closing_thought_window_is_the_one_the_citation_rule_uses():
    """`CLOSING_THOUGHT_STEPS` thoughts, not the whole trajectory: a thought from
    twenty steps ago is not what the answer is about to present."""
    trajectory = {
        "thought_0": "A" * 10_000,
        "thought_1": "bb",
        "thought_2": "ccc",
    }
    assert fastWorkflowReAct._closing_thought_bytes(trajectory) == 5


def test_the_bound_dataclass_is_frozen_so_evidence_cannot_drift():
    bound = extraction_bound(0, 0)
    assert isinstance(bound, ExtractionBound)
    with pytest.raises(Exception):
        bound.max_tokens = 1  # type: ignore[misc]



def test_the_trajectory_is_part_of_the_payload_and_a_flat_turn_is_not_floored():
    """2026-09-05, Gate 4 v5 shadow blocks: arm A's flat turn resolved a
    123-byte `presented_results` field after 91 commands, so its bound fell to
    the floor and the seven-subtask answer was cut at 4,096 tokens. The
    formatted trajectory the call receives is the third payload term."""
    from fastworkflow.utils.react import extraction_bound

    without = extraction_bound(123, 429)
    assert without.floor_applied is True
    assert without.max_tokens == EXTRACT_MIN_TOKENS
    assert without.trajectory_bytes == 0
    assert without.as_evidence()["trajectory_bytes"] == 0

    with_trajectory = extraction_bound(123, 429, 45_000)
    assert with_trajectory.trajectory_bytes == 45_000
    assert with_trajectory.max_tokens > EXTRACT_MIN_TOKENS
    assert with_trajectory.derived_tokens == (
        math.ceil((123 + 429 + 45_000) / EXTRACT_BYTES_PER_TOKEN * EXTRACT_RENDER_FACTOR)
        + EXTRACT_PROSE_ALLOWANCE_TOKENS
    )
    # A large flat trajectory reaches the ceiling, which is the backstop's job.
    assert with_trajectory.max_tokens <= EXTRACT_MAX_TOKENS_CEILING
    assert with_trajectory.timeout_s >= without.timeout_s



def test_the_provider_ceiling_bounds_the_derived_limit(monkeypatch):
    """2026-09-05, Gate 4 v6: the production route cerebras/gpt-oss-120b
    accepts at most 32,768 output tokens, below the shipped 36,864 ceiling; a
    request above it is refused, not served."""
    from fastworkflow.utils import react as react_module
    from fastworkflow.utils.react import extraction_bound, provider_max_output_tokens

    monkeypatch.setenv("LLM_AGENT", "cerebras/gpt-oss-120b")
    cap = provider_max_output_tokens()
    assert cap == 32768
    assert cap < EXTRACT_MAX_TOKENS_CEILING
    flat = extraction_bound(123, 429, 200_000)
    assert flat.provider_cap_applied is True
    assert flat.provider_max_output_tokens == cap
    assert flat.max_tokens == cap
    assert flat.as_evidence()["provider_cap_applied"] is True

    small = extraction_bound(123, 429)
    assert small.max_tokens == EXTRACT_MIN_TOKENS
    assert small.provider_cap_applied is True  # the cap lowered the ceiling, not this call
    assert small.provider_max_output_tokens == cap

    # An unknown route leaves only the shipped ceiling.
    monkeypatch.setenv("LLM_AGENT", "nobody/unknown-model")
    unknown = extraction_bound(123, 429, 200_000)
    assert unknown.provider_max_output_tokens is None
    assert unknown.provider_cap_applied is False
    assert unknown.max_tokens == EXTRACT_MAX_TOKENS_CEILING

    # Sonnet on Bedrock (the v5 route) is above the shipped ceiling: unchanged.
    monkeypatch.setenv("LLM_AGENT", "bedrock/us.anthropic.claude-sonnet-4-6")
    sonnet = extraction_bound(123, 429, 200_000)
    assert sonnet.provider_max_output_tokens == 64000
    assert sonnet.provider_cap_applied is False
    assert sonnet.max_tokens == EXTRACT_MAX_TOKENS_CEILING
