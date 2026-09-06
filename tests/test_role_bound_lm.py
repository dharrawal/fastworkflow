"""The role timeout has to be the bound that actually fires (arch §13.2.1).

EXP-028 Gate 4 v4 ran with `LLM_AGENT` bounded at 120 s and recorded provider
calls that failed after ~484 s. Nothing was misconfigured: `dspy.LM` hands
`timeout` AND `num_retries` to `litellm.completion`, which spends the timeout on
each of `1 + num_retries` HTTP attempts, so the configured bound was a bound on
an attempt and the call was free to cost four of them. These tests pin the
property the deployment believed it had — one logical model call cannot outlive
its role bound, or what is left of the turn deadline, whichever is smaller — and
the two things that must not be traded away to get it: no added retries, and an
exhausted provider is infrastructure, not a failed task.
"""

import json
import time

import dspy
import litellm
import pytest
from dspy.clients import lm as dspy_lm
from dspy.utils.exceptions import LMConfigurationError, LMTimeoutError
from litellm import ModelResponse

import fastworkflow
from fastworkflow.external_operations import operation
from fastworkflow.typed_failure import classify_exception
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.dspy_utils import SingleCallJSONAdapter, get_lm


# A whole test suite may not spend a real 300-second role bound, so the policy
# is shrunk by the same factor everywhere and the assertions are about the
# RELATIONSHIP between the numbers, which is what the production values also
# have to satisfy. Both stay above `_MIN_ATTEMPT_SECONDS`, because a request for
# less than a second is the one thing the policy refuses to make.
_TOTAL = 4.0
_ATTEMPT = 1.5
_DEADLINE = 1.2


@pytest.fixture
def short_role_bounds(monkeypatch):
    monkeypatch.setitem(dspy_utils._ROLE_TIMEOUTS, "LLM_AGENT", _TOTAL)
    monkeypatch.setitem(dspy_utils._ROLE_ATTEMPT_TIMEOUTS, "LLM_AGENT", _ATTEMPT)
    monkeypatch.setattr(dspy_utils, "_RETRY_BACKOFF_CAP_SECONDS", 0.01)
    monkeypatch.setitem(
        fastworkflow._env_vars, "LLM_AGENT", "openai/generic-agent-model"
    )
    monkeypatch.setitem(fastworkflow._env_vars, "FW_LM_CACHE", "0")
    yield


class _HangingProvider:
    """A provider that answers nothing until the timeout it was given expires.

    The failure mode being reproduced, exactly: the Bedrock Converse handler
    passes the per-call timeout to the httpx client it builds and then never
    hears from the model, so the bound that fires is the one on the attempt.
    """

    def __init__(self):
        self.attempt_timeouts: list[float] = []
        self.retry_counts: list[int] = []

    def __call__(self, *, request, num_retries, cache=None):
        timeout = float(request["timeout"])
        self.attempt_timeouts.append(timeout)
        self.retry_counts.append(num_retries)
        time.sleep(timeout)
        raise litellm.Timeout(
            message=f"Connection timed out after {timeout} seconds.",
            model=request["model"],
            llm_provider="test-provider",
        )


def _call(lm):
    return lm(messages=[{"role": "user", "content": "hello"}])


def test_a_hung_provider_cannot_outlive_the_role_call_bound(
    short_role_bounds, monkeypatch
):
    """The bound is on the CALL. Retries are spent from it, never on top of it."""
    provider = _HangingProvider()
    monkeypatch.setattr(dspy_lm, "litellm_completion", provider)

    lm = get_lm("LLM_AGENT")
    started = time.monotonic()
    with pytest.raises(LMTimeoutError):
        _call(lm)
    elapsed = time.monotonic() - started

    # The property the incident violated: 4 attempts of the configured timeout,
    # which is what the provider library does on its own, would be 4 x _ATTEMPT
    # plus backoff and would sail past _TOTAL.
    assert elapsed <= _TOTAL + 1.0
    assert len(provider.attempt_timeouts) > 1, "retries are not removed, only bounded"
    assert all(t <= _ATTEMPT for t in provider.attempt_timeouts)
    assert sum(provider.attempt_timeouts) <= _TOTAL
    # No retries were ADDED anywhere: the provider is asked for one attempt at a
    # time, and the configured retry count stays visible on the LM.
    assert provider.retry_counts == [0] * len(provider.attempt_timeouts)
    assert lm.num_retries == dspy_utils._PINNED_NUM_RETRIES


def test_the_call_bound_is_clamped_to_what_is_left_of_the_turn(
    short_role_bounds, monkeypatch
):
    """A per-call timeout longer than the turn's own bound is not a bound.

    The clamp has to be read where the CALL happens: `get_lm()` runs outside the
    `external_operations.operation(...)` block, so a timeout frozen at
    construction was clamped against no deadline at all.
    """
    provider = _HangingProvider()
    monkeypatch.setattr(dspy_lm, "litellm_completion", provider)

    lm = get_lm("LLM_AGENT")  # built with no deadline in force
    assert lm.role_policy.attempt_seconds == _ATTEMPT

    with operation("model.agent", seconds=_DEADLINE):
        with pytest.raises(LMTimeoutError):
            _call(lm)

    assert provider.attempt_timeouts, "the provider was never called"
    assert all(t <= _DEADLINE for t in provider.attempt_timeouts)
    assert len(provider.attempt_timeouts) == 1, (
        "no attempt may start once the turn's deadline cannot pay for one"
    )


def test_an_exhausted_provider_is_infrastructure_and_not_a_failed_task(
    short_role_bounds, monkeypatch
):
    """Censoring rule: a turn that never got an answer did not answer wrongly."""
    provider = _HangingProvider()
    monkeypatch.setattr(dspy_lm, "litellm_completion", provider)

    lm = get_lm("LLM_AGENT")
    with pytest.raises(LMTimeoutError) as raised:
        _call(lm)

    failure = classify_exception(raised.value)
    assert failure.disposition == "transient"
    assert failure.code == "provider-timeout"


def test_an_explicit_caller_timeout_still_owns_its_own_bound(
    short_role_bounds, monkeypatch
):
    """`conversation_labeling` pins both numbers itself; the policy defers."""
    provider = _HangingProvider()
    monkeypatch.setattr(dspy_lm, "litellm_completion", provider)

    lm = get_lm("LLM_AGENT", timeout=0.25, num_retries=0)
    assert getattr(lm, "role_policy", None) is None
    with pytest.raises(LMTimeoutError):
        _call(lm)
    assert provider.attempt_timeouts == [0.25]


# ----------------------------------------------------------------------
# The selector's accounting is not collateral damage
# ----------------------------------------------------------------------


class _SelectionSignature(dspy.Signature):
    """Pick a skill."""

    utterance: str = dspy.InputField()
    skill_name: str = dspy.OutputField()


def _model_response(payload: dict) -> ModelResponse:
    return ModelResponse(
        model="generic-agent-model",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(payload)},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_the_selector_path_still_makes_exactly_one_provider_call(
    short_role_bounds, monkeypatch
):
    """`provider_calls == application_attempts`, with zero provider retries."""
    entries = []

    def one_shot_completion(*, request, num_retries, cache=None):
        entries.append((request["timeout"], num_retries))
        return _model_response({"skill_name": "review-packet"})

    monkeypatch.setattr(dspy_lm, "litellm_completion", one_shot_completion)

    lm = get_lm("LLM_AGENT", num_retries=0)
    assert lm.num_retries == 0
    adapter = SingleCallJSONAdapter()

    with dspy.context(lm=lm, adapter=adapter):
        prediction = dspy.Predict(_SelectionSignature)(utterance="Review Casey.")

    assert prediction.skill_name == "review-packet"
    assert len(entries) == 1
    assert entries[0][1] == 0
    assert adapter.provider_call_count == 1
    assert adapter.provider_response_count == 1


def test_a_selector_provider_failure_is_still_counted_once(
    short_role_bounds, monkeypatch
):
    entries = []

    def failing_completion(*, request, num_retries, cache=None):
        entries.append(num_retries)
        for callback in request["failure_callback"]:
            callback.log_failure_event(
                kwargs={"first_api_call_start_time": object()},
                response_obj=None,
                start_time=None,
                end_time=None,
            )
        raise litellm.Timeout(
            message="Connection timed out.",
            model=request["model"],
            llm_provider="test-provider",
        )

    monkeypatch.setattr(dspy_lm, "litellm_completion", failing_completion)

    lm = get_lm("LLM_AGENT", num_retries=0)
    adapter = SingleCallJSONAdapter()

    with dspy.context(lm=lm, adapter=adapter):
        with pytest.raises(LMTimeoutError):
            dspy.Predict(_SelectionSignature)(utterance="Review Casey.")

    assert entries == [0]
    assert adapter.provider_call_count == 1
    assert adapter.provider_response_count == 0


def test_the_selector_adapter_still_refuses_an_lm_that_retries(
    short_role_bounds, monkeypatch
):
    """The zero-retry precondition must not become vacuous under the new loop.

    `RoleBoundLM` asks litellm for one attempt at a time, so `num_retries` on
    the wire is always 0. The number the adapter has to read is the CONFIGURED
    one, which stays on the LM.
    """
    monkeypatch.setattr(dspy_lm, "litellm_completion", _HangingProvider())

    lm = get_lm("LLM_AGENT")
    assert lm.num_retries == dspy_utils._PINNED_NUM_RETRIES

    with dspy.context(lm=lm, adapter=SingleCallJSONAdapter()):
        with pytest.raises(LMConfigurationError, match="provider retries to be zero"):
            dspy.Predict(_SelectionSignature)(utterance="Review Casey.")
