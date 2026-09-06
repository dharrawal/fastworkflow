import asyncio
import contextlib
import contextvars
import copy
import dspy
import math
import time
from dataclasses import dataclass
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.exceptions import (
    LMConfigurationError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
)
from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, Any, Union, get_args, get_origin, Tuple, List

import fastworkflow
from fastworkflow.utils.logging import logger


SELECTOR_ADAPTER_IDENTITY = (
    "dspy-json-object-single-call@2:"
    "native-schema=false,automatic-fallback=false,"
    "provider-retries=0,call-accounting=provider-handoff"
)
LM_TEMPERATURE_ENV_VAR = "FW_LM_TEMPERATURE"
LM_MAX_TOKENS_ENV_VAR = "FW_LM_MAX_TOKENS"


class _ProviderHandoffTracker(CustomLogger):
    """Count failed LiteLLM calls only after provider handoff began."""

    def __init__(self) -> None:
        super().__init__()
        self.provider_call_count = 0

    def log_failure_event(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
    ) -> None:
        if (kwargs or {}).get("first_api_call_start_time") is not None:
            self.provider_call_count += 1


class SingleCallJSONAdapter(JSONAdapter):
    """JSON adapter with one LM dispatch and no automatic fallback.

    DSPy's default ``ChatAdapter`` retries a parse failure through
    ``JSONAdapter``. ``JSONAdapter`` can itself retry a failed native response
    schema call in JSON-object mode. Those are useful interactive defaults but
    violate a phase whose application contract already owns its one explicit
    error-guided retry. This adapter chooses JSON-object mode up front and calls
    the base adapter pipeline directly, so one application attempt means one LM
    dispatch.
    """

    identity = SELECTOR_ADAPTER_IDENTITY

    def __init__(self) -> None:
        super().__init__(use_native_function_calling=False)
        self.provider_call_count = 0
        self.provider_response_count = 0
        self.raw_responses: list[Any] = []

    def __call__(
        self,
        lm,
        lm_kwargs: dict[str, Any],
        signature,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if (
            isinstance(lm, dspy.LM)
            and int(getattr(lm, "num_retries", 0) or 0) != 0
        ):
            raise LMConfigurationError(
                "SingleCallJSONAdapter requires provider retries to be zero",
                model=getattr(lm, "model", None),
            )
        pinned_kwargs = dict(lm_kwargs)
        if "response_format" in (getattr(lm, "supported_params", ()) or ()):
            pinned_kwargs["response_format"] = {"type": "json_object"}
        return Adapter.__call__(
            self,
            lm,
            pinned_kwargs,
            signature,
            demos,
            inputs,
        )

    def _call_lm(self, lm, request):
        data = self._legacy_call_kwargs(request)
        is_litellm = isinstance(lm, dspy.LM)
        tracker = _ProviderHandoffTracker()
        if is_litellm:
            failure_callbacks = data.get("failure_callback")
            if failure_callbacks is None:
                failure_callbacks = []
            elif isinstance(failure_callbacks, list):
                failure_callbacks = list(failure_callbacks)
            else:
                failure_callbacks = [failure_callbacks]
            failure_callbacks.append(tracker)
            data["failure_callback"] = failure_callbacks

        history_before = len(getattr(lm, "history", ()) or ())
        try:
            outputs = lm(messages=data.pop("messages"), **data)
        except BaseException:
            self.provider_call_count += (
                tracker.provider_call_count if is_litellm else 1
            )
            raise

        cache_hit = False
        history = list(getattr(lm, "history", ()) or ())
        if is_litellm and len(history) > history_before:
            entry = history[-1]
            if isinstance(entry, dict):
                cache_hit = bool(
                    getattr(entry.get("response"), "cache_hit", False)
                )
        if not cache_hit:
            self.provider_call_count += 1
            self.provider_response_count += 1
        return self._normalize_legacy_outputs(outputs, request)

    def _call_postprocess(
        self,
        processed_signature,
        original_signature,
        outputs,
        lm,
        lm_kwargs,
    ):
        self.raw_responses.extend(
            output.get("text")
            if isinstance(output, dict) and "text" in output
            else output
            for output in outputs
        )
        return super()._call_postprocess(
            processed_signature,
            original_signature,
            outputs,
            lm,
            lm_kwargs,
        )


def get_lm(model_env_var: str, api_key_env_var: Optional[str] = None, **kwargs):
    """
    Get the dspy LM object.
    
    Supports LiteLLM Proxy routing: if the model string starts with 'litellm_proxy/',
    the call is routed through the LiteLLM Proxy using LITELLM_PROXY_API_BASE and
    LITELLM_PROXY_API_KEY environment variables.
    
    Args:
        model_env_var: Name of the environment variable containing the model string
                       (e.g., 'LLM_AGENT', 'LLM_PARAM_EXTRACTION').
        api_key_env_var: Name of the environment variable containing the API key
                         for direct provider calls. Ignored for litellm_proxy/ models.
        **kwargs: Additional keyword arguments passed to dspy.LM().
    
    Returns:
        dspy.LM: Configured language model instance.
    
    Raises:
        ValueError: If model is not set, or if using litellm_proxy/ without
                    LITELLM_PROXY_API_BASE configured.
    
    Example:
        # Direct provider call (existing behavior):
        # LLM_AGENT=mistral/mistral-small-latest
        # LITELLM_API_KEY_AGENT=sk-...
        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        
        # LiteLLM Proxy call:
        # LLM_AGENT=litellm_proxy/bedrock_mistral_large_2407
        # LITELLM_PROXY_API_BASE=http://127.0.0.1:4000
        # LITELLM_PROXY_API_KEY=proxy-key-...
        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")  # api_key_env_var is ignored for proxy
    """
    # `FW_LM_CACHE=0` defeats the DSPy response cache process-wide
    # (`fix-bn1` `[XR16]`). Without it, k repeated attempts of one task in a
    # pass^k experiment send identical prompts, hit the disk+memory cache that
    # dspy enables by default, and come back byte-identical -- so pass^k equals
    # pass@1 by construction and the run looks like a spectacular result. The
    # lever is process-wide because nothing threads per-call kwargs down to
    # here; `**kwargs` already reaches `dspy.LM`, but no caller passes any.
    #
    # Unset leaves today's behavior exactly as it was.
    if fastworkflow.get_env_var("FW_LM_CACHE", default="1") in ("0", "false", "False"):
        kwargs.setdefault("cache", False)

    # Per-role provider bounds (arch §13.2.1, FW-REQ-008 clause 1). Every model
    # call had the provider's own default timeout and the provider's own default
    # retry count — neither chosen here, neither visible, and neither related to
    # the turn the call belongs to. A hung provider therefore hung a turn for as
    # long as the provider felt like it.
    #
    # `setdefault`, so an explicit caller kwarg still wins, and the values are
    # clamped to whatever remains of the active external-operation deadline: a
    # per-call timeout longer than the turn's own bound is not a bound.
    #
    # `RoleBoundLM` then re-derives the bound on every call, because the clamp
    # has to see the `external_operations.operation(...)` block the CALL runs
    # inside — a value frozen here was clamped against no deadline at all.
    caller_owns_timeout = "timeout" in kwargs
    _apply_role_policy(kwargs, model_env_var)
    _apply_sampling_policy(kwargs)

    def _build(**construction_kwargs):
        lm_class = _role_bound_lm_class(dspy.LM)
        lm = lm_class(**construction_kwargs, **kwargs)
        if not caller_owns_timeout:
            lm.role_policy = role_policy(
                model_env_var,
                max_retries=kwargs.get("num_retries", _PINNED_NUM_RETRIES),
            )
        return lm

    model = fastworkflow.get_env_var(model_env_var)
    if not model:
        logger.critical(f"Critical Error: DSPy Language Model not provided. Set {model_env_var} environment variable.")
        raise ValueError(f"DSPy Language Model not provided. Set {model_env_var} environment variable.")
    
    # Check if this is a LiteLLM Proxy call
    if model.startswith("litellm_proxy/"):
        # Route through LiteLLM Proxy
        proxy_api_base = fastworkflow.get_env_var("LITELLM_PROXY_API_BASE")
        if not proxy_api_base:
            raise ValueError(
                f"Model '{model}' uses litellm_proxy/ prefix but LITELLM_PROXY_API_BASE is not set. "
                "Set LITELLM_PROXY_API_BASE to your LiteLLM Proxy URL (e.g., http://127.0.0.1:4000)."
            )
        
        # Get optional proxy API key (allows no-auth proxies when empty/not set)
        proxy_api_key = fastworkflow.get_env_var("LITELLM_PROXY_API_KEY", default=None)
        
        logger.debug(f"Routing {model_env_var} through LiteLLM Proxy at {proxy_api_base}")
        
        if proxy_api_key:
            return _build(
                model=model,
                api_base=proxy_api_base,
                api_key=proxy_api_key,
            )
        else:
            return _build(model=model, api_base=proxy_api_base)
    
    # Bedrock authenticates through botocore's standard AWS credential chain.
    # A role-specific API key may still be configured for the non-Bedrock
    # default model; never forward that unrelated secret when this role is
    # temporarily routed through Bedrock.
    if model.startswith(("bedrock/", "bedrock_converse/")):
        return _build(model=model)

    # Direct provider call (existing behavior)
    api_key = fastworkflow.get_env_var(api_key_env_var) if api_key_env_var else None
    return (
        _build(model=model, api_key=api_key)
        if api_key
        else _build(model=model)
    )

# Per-role LM policy (arch §13.2.1). The role is read from the env-var name the
# caller asked for, because that is the only thing distinguishing one model call
# from another at this seam — `LLM_AGENT`, `LLM_PLANNER`, `LLM_PARAM_EXTRACTION`
# and the distillation roles all arrive here as a string.
#
# Retries are PINNED rather than left to the provider default, because an
# unpinned provider retry is a retry nobody counted: it multiplies the wall time
# of a call whose deadline was computed for one attempt, and for a write it
# would re-dispatch an effect the runtime never learned about.
# The original rationale above remains true; the two-bound implementation below
# makes the pinned retries visible to one logical-call budget.
#
# Two numbers, not one, because a provider library turns one of them into the
# other and the difference is what EXP-028 Gate 4 paid for. `dspy.LM` hands
# `timeout` AND `num_retries` to `litellm.completion`, which spends the timeout
# on EACH of `1 + num_retries` HTTP attempts
# (dspy/clients/lm.py:253-256 -> litellm/utils.py:1531 ->
# litellm/main.py:5695-5719). A single role timeout is therefore a bound on an
# attempt and never on the call, and the call is what a turn deadline is made
# of. So:
#
#   * `_ROLE_TIMEOUTS` is the bound on ONE LOGICAL MODEL CALL — every provider
#     attempt it makes, together. This is the number that has to be true.
#   * `_ROLE_ATTEMPT_TIMEOUTS` is the bound on one provider HTTP attempt, and it
#     is what actually reaches httpx.
#
# `RoleBoundLM` (below) owns the loop between them and takes a retry only while
# a WHOLE further attempt still fits in what is left. That rule is what keeps
# both properties at once: a transport failure that fails fast (a Bedrock 503)
# is still retried, and a timeout — which by definition consumed a full attempt
# — is not, because retrying it would spend a bound that was already promised.
#
# Attempt sizes are set from observed generation speed, not from taste. At the
# deployment's `max_tokens` of 4096 and the ~35 tok/s measured on
# bedrock/us.anthropic.claude-sonnet-4-6, a full-length answer takes ~117 s of
# generation before the first byte of a non-streaming Converse response is sent.
# The 120 s that used to be here was BELOW the nominal cost of the calls it was
# bounding, so healthy full-length generations were being cut off and silently
# re-dispatched by litellm. That is what the two Gate 4 v4 observations are:
# three calls that died at ~484 s (4 x 120 s of attempts plus 3 s of tenacity
# backoff), holding a turn that then had nothing to show for it; and a call
# recorded at 229 s that nonetheless returned a full 4096-token answer, which no
# single 120 s attempt could have produced and one killed attempt plus one
# successful retry accounts for exactly.
_ROLE_TIMEOUTS: dict[str, float] = {
    "LLM_PLANNER": 300.0,
    "LLM_AGENT": 300.0,
    "LLM_PARAM_EXTRACTION": 90.0,
    "LLM_SUMMARIZATION": 90.0,
    "LLM_CLARIFICATION": 90.0,
    "LLM_TEACHER": 180.0,
    "LLM_STUDENT": 180.0,
    "LLM_INSIGHT_EXTRACTION": 180.0,
}
_DEFAULT_ROLE_TIMEOUT = 120.0

# Per-attempt bounds. 240 s for the two roles that generate to `max_tokens` is
# ~2x the nominal full-length generation above, i.e. it tolerates a sustained
# ~17 tok/s — half the observed rate — plus time to first byte on a large
# prompt. The rest generate short outputs and get a fraction of their call
# bound, so a fast transport failure still leaves room for another attempt
# inside it. A role with no entry falls to `_DEFAULT_ATTEMPT_TIMEOUT`.
_ROLE_ATTEMPT_TIMEOUTS: dict[str, float] = {
    "LLM_PLANNER": 240.0,
    "LLM_AGENT": 240.0,
    "LLM_PARAM_EXTRACTION": 30.0,
    "LLM_SUMMARIZATION": 30.0,
    "LLM_CLARIFICATION": 30.0,
    "LLM_TEACHER": 90.0,
    "LLM_STUDENT": 90.0,
    "LLM_INSIGHT_EXTRACTION": 90.0,
}
_DEFAULT_ATTEMPT_TIMEOUT = 60.0

# Historical one-bound wording retained verbatim beside the split policy.
_ROLE_ATTEMPT_POLICY_DOC = (
    """Set the provider attempt timeout and retry count, clamped to the deadline."""
)

# A deadline with nothing left would ask the provider for a zero-second call,
# which most clients treat as "no timeout" — the opposite of what is meant. The
# floor keeps the request honestly bounded and lets the deadline check that
# follows report the expiry as what it is.
_MIN_ATTEMPT_SECONDS = 1.0

# Pinned at dspy's own default rather than changed. What matters here is that
# the number is CHOSEN and visible — arch §13.2.1 asks for a pinned retry count,
# not for a smaller one — so a provider-library upgrade cannot silently change
# how many times a turn's model call is retried underneath a phase that thinks
# it made one attempt.
#
# Lowering it was tried and reverted the same day: at 1, transient provider 503s
# ("Service temporarily unavailable due to high load") started failing turns
# that had always ridden them out. The retry these attempts represent is
# recovery from a TRANSPORT failure, which is a different question from the
# phase-scoped retry of a bad RESPONSE (arch §8.4), and removing it was
# removing a recovery rather than relocating one.
#
# It is still 3. What changed is that the three retries are now spent out of the
# call's own budget instead of multiplying it.
_PINNED_NUM_RETRIES = 3

# Between attempts. The same schedule litellm applied before the loop moved here
# — tenacity's `wait_exponential(multiplier=1, max=10)`, i.e. 1 s, 2 s, 4 s —
# because a provider under load that is retried immediately is a provider being
# hammered, and because keeping the schedule identical means the change is about
# the bound and not about how hard the provider is pushed. The wait is spent
# from the call budget like everything else, so it can only shorten the run,
# never lengthen it past the role bound.
_RETRY_BACKOFF_CAP_SECONDS = 10.0

# Failures a further provider attempt could plausibly answer. Everything else —
# a bad request, an auth failure, a context-window error — is re-raised at once:
# retrying it spends the turn's budget on a question already answered.
_RETRYABLE_PROVIDER_ERRORS = (
    LMTimeoutError,
    LMTransportError,
    LMRateLimitError,
    LMServerError,
)


@dataclass(frozen=True)
class RolePolicy:
    """The two bounds and the retry count for one role's model calls."""

    role: str
    # The bound on one LOGICAL call: every provider attempt, together.
    total_seconds: float
    # The bound on one provider HTTP attempt.
    attempt_seconds: float
    # How many provider retries this role may take. Never raised here; the
    # budget rule below can only take fewer.
    max_retries: int

    def backoff_seconds(self, attempt: int) -> float:
        return min(float(2 ** attempt), _RETRY_BACKOFF_CAP_SECONDS)


def role_policy(model_env_var: str, max_retries: int = _PINNED_NUM_RETRIES) -> RolePolicy:
    """The policy for a role, with the attempt bound never exceeding the call bound."""
    total = float(_ROLE_TIMEOUTS.get(model_env_var, _DEFAULT_ROLE_TIMEOUT))
    attempt = float(_ROLE_ATTEMPT_TIMEOUTS.get(model_env_var, _DEFAULT_ATTEMPT_TIMEOUT))
    return RolePolicy(
        role=model_env_var,
        total_seconds=total,
        attempt_seconds=min(attempt, total),
        max_retries=max(0, int(max_retries or 0)),
    )


def _remaining_call_budget(policy: RolePolicy, started: float) -> float:
    """What is left of this call's bound, under the deadline in force RIGHT NOW.

    Two things shrink it and both have to be read here rather than at
    construction: the call's own elapsed time, and the active external-operation
    deadline. The second is the reason this is not a constant folded into
    `dspy.LM(...)` — `get_lm()` runs outside the `external_operations.operation`
    block that the call itself runs inside (`react.py` opens the block around the
    call, not around the LM), so a timeout computed at construction was clamped
    against no deadline at all and the clamp was decorative.
    """
    from fastworkflow.external_operations import clamp_timeout

    elapsed = time.monotonic() - started
    clamped = clamp_timeout(policy.total_seconds)
    if clamped is None:
        clamped = policy.total_seconds
    return max(0.0, min(float(clamped), policy.total_seconds - elapsed))


def _attempt_view(lm):
    """A shallow view of `lm` with the provider's own retry loop switched off.

    `dspy.LM.forward` reads `self.num_retries` and hands it to litellm, so the
    only way to keep the retry decision where the deadline is visible is to ask
    the provider for exactly one attempt. A shallow copy rather than a mutation
    because an LM can be shared across turns and threads: `forward()` touches
    neither `history` nor `kwargs` (history is appended by `__call__`, on the
    real object), so the copy is the knob and nothing else. `lm.num_retries`
    stays truthful for anything reading the configured policy — including
    `SingleCallJSONAdapter`, whose zero-retry precondition would otherwise
    become vacuous.
    """
    view = getattr(lm, "_role_bound_attempt_view", None)
    if view is None:
        view = copy.copy(lm)
        view.num_retries = 0
        # Shared, not rebuilt per attempt: a shallow copy shares `kwargs` and
        # `history` with the original, so the only per-call state it carries is
        # dspy's own one-shot warning flags, which are meant to be one-shot.
        lm._role_bound_attempt_view = view
    return view


def _plan_attempt(lm, kwargs: dict) -> tuple[Optional[RolePolicy], Optional[Any]]:
    """(policy, attempt view), or (None, None) when this call is not role-bound.

    A per-call ``timeout`` is the caller sizing ONE PROVIDER ATTEMPT for a call
    whose cost it can compute and the role table cannot — ido-mn1.6.10's
    extraction bound is the only such caller today. It is honoured as an
    override of ``attempt_seconds``, never as a hand-off to litellm's own retry
    loop: before this, a caller-supplied timeout took the ``policy is None``
    branch, and ``dspy.LM.forward`` then spent that timeout on EACH of
    ``1 + num_retries`` attempts, so asking for a 600 s bound asked for 2400 s.
    A caller that computed a bound from the work in front of it is entitled to
    have that bound be the bound.

    ``total_seconds`` is raised to the derived attempt when the derivation is
    the larger of the two, because an attempt that cannot fit inside the call
    budget is an attempt that can never be made. It is never lowered: the role
    table's call bound still holds for every call that does not derive one, and
    `_remaining_call_budget` still clamps whatever is left to the active
    external-operation deadline, so the turn deadline remains the outer bound in
    both directions.

    The kwarg is POPPED. `forward` passes its own `timeout=` to the base, and a
    duplicate keyword would raise before the provider ever saw the request.
    """
    policy = getattr(lm, "role_policy", None)
    if policy is None:
        return None, None
    derived = kwargs.pop("timeout", None)
    if derived is not None:
        try:
            derived = float(derived)
        except (TypeError, ValueError):
            derived = None
    if derived is not None and derived > 0:
        policy = RolePolicy(
            role=policy.role,
            total_seconds=max(policy.total_seconds, derived),
            attempt_seconds=derived,
            max_retries=policy.max_retries,
        )
    return policy, _attempt_view(lm)


# ---------------------------------------------------------------------------
# Provider finish reasons, for the caller that has to know its answer was cut
#
# `dspy.Predict` hands back parsed FIELDS. Whether the provider stopped because
# the model was finished or because it hit `max_tokens` is not one of them, and
# `lm.history` — the usual place to look — is permanently empty in the process
# the chatbot actually runs (`run_fastapi_mcp` constructs its LMs with
# `disable_history=True` as a memory bound). So a caller that must distinguish
# "this is the answer" from "this is the first N tokens of the answer" has
# nowhere to read it from.
#
# This is that place: a collector the caller opens around ONE logical model
# call, which `RoleBoundLM` fills from the provider response it already holds.
# Scoped to a contextvar rather than to the LM object because an LM is shared
# across turns and threads, and "the finish reason of the last call" on a shared
# object is a race, not a fact.
_finish_reason_sink: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "fw_finish_reason_sink", default=None
)


@contextlib.contextmanager
def capture_finish_reasons():
    """Collect the provider finish reasons of the calls made in this block.

    Yields the list it fills, newest last. Empty when the LM is not role-bound
    (a stub, or a model built outside `get_lm`) — an absence of evidence, which
    every caller must read as "not known to be truncated" rather than as
    "not truncated".
    """
    sink: list = []
    token = _finish_reason_sink.set(sink)
    try:
        yield sink
    finally:
        _finish_reason_sink.reset(token)


def _record_finish_reasons(response) -> None:
    """Append this response's per-choice finish reasons to the active sink."""
    sink = _finish_reason_sink.get()
    if sink is None or response is None:
        return
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    for choice in choices or ():
        reason = getattr(choice, "finish_reason", None)
        if reason is None and isinstance(choice, dict):
            reason = choice.get("finish_reason")
        if reason is not None:
            sink.append(str(reason))


_ROLE_BOUND_CLASSES: dict[type, type] = {}


def _role_bound_lm_class(base: type) -> type:
    """A subclass of `base` whose deadline is enforced on the logical call.

    Built from whatever `dspy.LM` currently is, and cached, so a test that
    replaces `dspy.LM` with a stub still gets its stub.
    """
    if not isinstance(base, type):
        # `dspy.LM` replaced by a mock or a factory rather than a class (several
        # build tests do exactly that). There is nothing to subclass, and a
        # stand-in has no deadline to enforce, so hand back what the caller
        # asked for.
        return base

    cached = _ROLE_BOUND_CLASSES.get(base)
    if cached is not None:
        return cached

    class RoleBoundLM(base):  # type: ignore[misc, valid-type]
        """`dspy.LM` whose role timeout bounds the CALL, not one attempt of it.

        The provider is asked for one attempt at a time, with a timeout
        recomputed from what is left of the role bound and of the active
        external-operation deadline. A retry is taken only while a whole further
        attempt still fits, which is what makes the role timeout the bound that
        fires: a timeout consumed a full attempt, so it ends the call, while a
        transport failure that failed in seconds leaves room and is retried.
        """

        def forward(self, prompt=None, messages=None, **kwargs):
            policy, view = _plan_attempt(self, kwargs)
            if policy is None:
                return base.forward(self, prompt=prompt, messages=messages, **kwargs)

            started = time.monotonic()
            for attempt in range(policy.max_retries + 1):
                per_attempt = max(
                    _MIN_ATTEMPT_SECONDS,
                    min(
                        policy.attempt_seconds,
                        _remaining_call_budget(policy, started),
                    ),
                )
                try:
                    response = base.forward(
                        view,
                        prompt=prompt,
                        messages=messages,
                        timeout=per_attempt,
                        **kwargs,
                    )
                    _record_finish_reasons(response)
                    return response
                except _RETRYABLE_PROVIDER_ERRORS:
                    if attempt >= policy.max_retries:
                        raise
                    backoff = policy.backoff_seconds(attempt)
                    if (
                        _remaining_call_budget(policy, started) - backoff
                        < policy.attempt_seconds
                    ):
                        raise
                    time.sleep(backoff)
            raise AssertionError("unreachable: the loop returns or re-raises")

        async def aforward(self, prompt=None, messages=None, **kwargs):
            policy, view = _plan_attempt(self, kwargs)
            if policy is None:
                return await base.aforward(
                    self, prompt=prompt, messages=messages, **kwargs
                )

            started = time.monotonic()
            for attempt in range(policy.max_retries + 1):
                per_attempt = max(
                    _MIN_ATTEMPT_SECONDS,
                    min(
                        policy.attempt_seconds,
                        _remaining_call_budget(policy, started),
                    ),
                )
                try:
                    response = await base.aforward(
                        view,
                        prompt=prompt,
                        messages=messages,
                        timeout=per_attempt,
                        **kwargs,
                    )
                    _record_finish_reasons(response)
                    return response
                except _RETRYABLE_PROVIDER_ERRORS:
                    if attempt >= policy.max_retries:
                        raise
                    backoff = policy.backoff_seconds(attempt)
                    if (
                        _remaining_call_budget(policy, started) - backoff
                        < policy.attempt_seconds
                    ):
                        raise
                    await asyncio.sleep(backoff)
            raise AssertionError("unreachable: the loop returns or re-raises")

    RoleBoundLM.__name__ = f"RoleBound{getattr(base, '__name__', 'LM')}"
    RoleBoundLM.__qualname__ = RoleBoundLM.__name__
    _ROLE_BOUND_CLASSES[base] = RoleBoundLM
    return RoleBoundLM


def _apply_role_policy(kwargs: dict, model_env_var: str) -> None:
    # Set the provider attempt timeout and retry count, clamped to the deadline.
    """Set provider timeout and retry count for this role, clamped to the deadline."""
    from fastworkflow.external_operations import clamp_timeout

    policy = role_policy(model_env_var)
    clamped = clamp_timeout(policy.attempt_seconds)
    if clamped is not None:
        # A deadline with nothing left would ask the provider for a zero-second
        # call, which most clients treat as "no timeout" — the opposite of what
        # is meant. The floor keeps the request honestly bounded and lets the
        # deadline check that follows report the expiry as what it is.
        kwargs.setdefault("timeout", max(_MIN_ATTEMPT_SECONDS, float(clamped)))
    kwargs.setdefault("num_retries", _PINNED_NUM_RETRIES)


def _apply_sampling_policy(kwargs: dict) -> None:
    """Apply optional deployment-pinned sampling values to every live role."""
    raw_temperature = fastworkflow.get_env_var(
        LM_TEMPERATURE_ENV_VAR,
        default=None,
    )
    if raw_temperature not in (None, ""):
        temperature = float(raw_temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"{LM_TEMPERATURE_ENV_VAR} must be a finite non-negative number"
            )
        kwargs.setdefault("temperature", temperature)

    raw_max_tokens = fastworkflow.get_env_var(
        LM_MAX_TOKENS_ENV_VAR,
        default=None,
    )
    if raw_max_tokens not in (None, ""):
        max_tokens = int(raw_max_tokens)
        if max_tokens <= 0:
            raise ValueError(f"{LM_MAX_TOKENS_ENV_VAR} must be positive")
        kwargs.setdefault("max_tokens", max_tokens)


def _process_field(field_info, is_input: bool) -> Tuple[Any, Any, bool]:
    """Process a single field and return its type, DSPy field, and optional status."""
    field_type = field_info.annotation
    field_desc = field_info.description or f"{'Input' if is_input else 'Output'} field"
    
    # Handle Optional types
    is_optional = False
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if type(None) in args:
            is_optional = True
            field_type = next((t for t in args if t is not type(None)), str)
    
    dspy_field = dspy.InputField(desc=field_desc) if is_input else dspy.OutputField(desc=field_desc)
    return field_type, dspy_field, is_optional


def _process_input_fields(model_class: Type[BaseModel], preserve_types: bool) -> Dict[str, Tuple]:
    """Process all input fields from a Pydantic model."""
    fields = {}
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=True)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
    return fields


def _process_output_fields(model_class: Type[BaseModel], preserve_types: bool) -> Tuple[Dict[str, Tuple], List[str]]:
    """Process all output fields from a Pydantic model and generate instructions."""
    fields = {}
    instructions = []
    
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=False)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
        
        # Generate field-specific instructions
        _add_field_instructions(field_name, field_info, instructions)
        
    return fields, instructions


def _add_field_instructions(field_name: str, field_info, instructions: List[str]) -> None:
    """Add instructions for a specific field based on its metadata."""
    if hasattr(field_info, 'default') and field_info.default is not None:
        instructions.append(f"For '{field_name}': Use '{field_info.default}' if not explicitly mentioned.")
    
    if hasattr(field_info, 'examples') and field_info.examples:
        examples_str = ", ".join(f"'{ex}'" for ex in field_info.examples)
        instructions.append(f"Examples for '{field_name}': {examples_str}")


def _create_instructions(custom_instructions: Optional[str], auto_instructions: List[str]) -> str:
    """Create the final instruction string from custom and auto-generated instructions."""
    if custom_instructions:
        return custom_instructions
    
    if not auto_instructions:
        return ""
        
    return "Extract the following fields based on the input:\n\n" + "\n".join(auto_instructions)


def dspySignature(
    Input_class: Type[BaseModel], 
    Output_class: Type[BaseModel],
    instructions: Optional[str] = None,
    preserve_types: bool = True
) -> Type[dspy.Signature]:
    """
    Dynamically creates a dspy.Signature class from Pydantic Input and Output models.

    Args:
        Input_class: A Pydantic BaseModel class defining the input fields.
        Output_class: A Pydantic BaseModel class defining the output fields.
        instructions: Optional custom instructions to include in the signature.
        preserve_types: Whether to preserve field type annotations in the signature.

    Returns:
        A new class that inherits from dspy.Signature.
    """
    if not issubclass(Input_class, BaseModel) or not issubclass(Output_class, BaseModel):
        raise TypeError("Input_class and Output_class must be subclasses of pydantic.BaseModel.")

    # Process fields from both classes
    input_fields = _process_input_fields(Input_class, preserve_types)
    output_fields, auto_instructions = _process_output_fields(Output_class, preserve_types)
    
    # Combine all fields and create instructions
    dspy_fields = {**input_fields, **output_fields}
    final_instructions = _create_instructions(instructions, auto_instructions)
    
    return dspy.Signature(dspy_fields, final_instructions.strip())

############################################
# Steps:
# 1. Define your signature and dspy function
# 2. Get prediction from DSPy module
# 3. Create output directly using ** unpacking

# dspy_signature_class = dspySignature(Signature.Input, Signature.Output)
# dspy_predict_func = dspy.Predict(dspy_signature_class)
# prediction = dspy_predict_func(input)  # Returns a dspy.Prediction object
# return Signature.Output(**prediction)