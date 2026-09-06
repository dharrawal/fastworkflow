import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from litellm import ContextWindowExceededError
from litellm import exceptions as litellm_exceptions

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

from fastworkflow import external_operations, result_handles, tracing
from fastworkflow.plan_execution import SafetyEnvelopeState
from fastworkflow.turn_budget import (LegacyTurnBudget, LogicalTurnBudget,
                                      TurnPartial, budget_from_state)
from fastworkflow.typed_failure import (
    CODE_ADAPTER_PARSE,
    CODE_EXTRACTION_FAILED,
    CODE_EXTRACTION_TRUNCATED,
    ControlSignal,
    TurnFailedError,
    TypedFailure,
    extraction_truncated_failure,
)
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.dspy_logger import DSPyForward

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class AskUserSuspend(BaseException):
    """
    Raised by ask_user when no user_message_queue is configured (Topology B).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; the loop catches this explicitly and returns a suspended sentinel.
    """

    def __init__(self, clarification_request: str):
        self.clarification_request = clarification_request
        super().__init__(clarification_request)


from fastworkflow.policy_decision import (
    AfterObservationInput,
    BeforeFinishInput,
    ContractFacts,
    PolicyDecisionPoint,
    PolicyMode,
    PolicyOutcome,
)


# Architecture §8.4 phase machine. Retry is scoped to the phase that owns it:
#
#   1. DECISION — call `self.react` and retry model/adapter parsing only, and
#      only before a valid tool decision has been accepted. Nothing has been
#      executed yet, so a retry here repeats nothing.
#   2. TOOL — derive the durable logical-call key and execute ONCE. Tool retry
#      belongs to the read/side-effect contract, never to this loop.
#   3. OBSERVATION SEAL — append the observation and seal a digest of the step,
#      so a later phase can prove which steps completed.
#   4. FINISH — seal an immutable snapshot of the completed trajectory and retry
#      only `self.extract` against it.
#
# What this replaces is a retry at the wrong altitude: WEC re-invoked the whole
# `forward()` on an AdapterParseError, which re-executed every tool call the
# first attempt had already made (FW-REQ-008B clause 3).
DECISION_PARSE_ATTEMPTS = 3
EXTRACT_PARSE_ATTEMPTS = 3

# ido-mn1.6.33. Why an extraction was not attempted, when it was not attempted.
# It rides as a CAUSE beside the typed reason rather than replacing it: the
# durable turn schema (`workflow_execution_context`, schema v8) requires
# `extraction_truncated_reason == CODE_EXTRACTION_TRUNCATED` and an
# `extraction_failure` classified the same way, and that requirement is what
# makes every downstream reader treat the cell as infrastructure and never as a
# task failure. Spelling a new code here would take the cell OUT of that rule
# on its way to explaining itself.
DEADLINE_INSUFFICIENT_CAUSE = "deadline-insufficient-for-extraction"

# ---------------------------------------------------------------------------
# Closing-thought bound (ido-mn1.6.4)
#
# EXP-028 Gate 4 v4 (57 cells, 3 arms, Bedrock Sonnet 4.6, max_tokens 4096):
# 131 ReAct LLM calls ended with finish_reason=length. Broken down by what the
# generation actually contained:
#
#   46  `self.react` steps that emitted a perfectly good next_thought /
#       next_tool_name / next_tool_args, then kept writing the whole report
#       AFTER the `[[ ## completed ## ]]` marker (median 8.8 KB of text the
#       adapter parses and throws away — ~63 s of generation per call, pure
#       waste, and the class the three Bedrock timeouts came from).
#   16  `self.react` steps whose next_thought itself ran away (a report, or a
#       hallucinated continuation of the trajectory) until the token cap.
#   69  `self.extract` calls whose `final_answer` — the actual deliverable —
#       ran to the cap. Those are NOT waste and are deliberately left free to
#       render; see `_finish`.
#
# The two bounds below address the first two classes and nothing else. Raising
# max_tokens is explicitly not the fix: it lengthens the wasted generation.
MAX_NEXT_THOUGHT_CHARS = 1200
THOUGHT_TRUNCATION_NOTICE = (
    " …[next_thought truncated by the runtime: it is a working note, not the "
    "final answer]"
)
# `reasoning` on the extraction step is the CoT scratchpad, not the answer
# (median 292 chars in the v4 collection). Bounded for the same reason, and
# `final_answer` deliberately is not.
MAX_EXTRACT_REASONING_CHARS = 1200
REASONING_TRUNCATION_NOTICE = (
    " …[reasoning truncated by the runtime: it is a scratchpad, not the final "
    "answer]"
)
# The ChatAdapter appends `[[ ## completed ## ]]` after the last output field
# and does not read it back (`completed` is not an output field, so
# ChatAdapter.parse ignores its absence). Stopping there is what actually ends
# the 46-call overrun class — an instruction alone cannot, because the tokens
# are already spent by the time anything downstream could reject them. It
# cannot affect next_tool_name / next_tool_args, which the adapter emits
# strictly before the marker.
REACT_STOP_SEQUENCES: tuple[str, ...] = ("[[ ## completed ## ]]",)

# ---------------------------------------------------------------------------
# Presentation resolution at extraction time (ido-mn1.6.6)
#
# The extraction call is the composition step: arm A returns its `final_answer`
# verbatim and arms B/C concatenate one per leaf with no further model call. So
# it is the only place an answer's rows can come from — and once observations
# carry `summary + result_handle + first page` (ido-mn1.6.1), the rows are not
# in the trajectory it reads. `presented_results` is the repair: a dedicated
# input field carrying the payloads of the handles this answer must present,
# resolved from session state, so the composition step is asked to render rows
# it has rather than to remember rows it never received.
#
# It is an input field and not a wider trajectory on purpose. Re-inflating the
# observations would restore the multiplied per-step cost compaction removed;
# this pays once, at the end, for the handles two runtime-checkable rules select.
PRESENTED_RESULTS_FIELD = "presented_results"
PRESENTED_RESULTS_DESC = (
    "Full rows for the result handles this answer must present, resolved from "
    "session state after the loop ended. The trajectory above shows only a "
    "summary and one page of each result; these are the rows to render. Use "
    "them for any listing the request asked for, do not invent rows that are "
    "not here, and where a block says it was trimmed, say the listing is "
    "partial rather than implying it is complete."
)
# How many of the closing steps' thoughts are read for citations. The `finish`
# step's own thought is where ido-mn1.6.4 instructs the agent to name its
# handles, and the step before it is read too because an agent that decided to
# finish on step N-1 and then wrote a terse `finish` thought would otherwise
# have its citation dropped for being one step early. Not the whole trajectory:
# a handle mentioned twenty steps ago and superseded since is not what the
# answer presents, and admitting it is how "do not resolve everything" erodes.
CLOSING_THOUGHT_STEPS = 2


# ---------------------------------------------------------------------------
# The extraction call's completion limit (ido-mn1.6.10)
#
# Every model call in the deployment shares one `max_tokens` (FW_LM_MAX_TOKENS,
# 4096 in EXP-028 Gate 4). For the react step that is right and raising it is
# actively wrong — a step that overruns is wasting tokens, which is why
# ido-mn1.6.4 gave it a stop sequence instead. For the EXTRACTION step it is
# wrong in the other direction: that call is the composition step, its
# `final_answer` IS the deliverable (arm A returns it verbatim; arms B and C
# concatenate one per leaf with no further model call), and in Gate 4 v4 it
# ended `finish_reason=length` 69 times — with the cut text stored as the turn's
# answer in 17/19 arm-A cells, 5/19 B and 13/19 C. A shared constant cannot be
# right for both, because only one of the two has a size that can be computed
# before the call.
#
# This one can. `presented_results` (ido-mn1.6.6) is a field the runtime builds
# and therefore measures: an answer asked to render it cannot be shorter than
# it. So the limit is derived per call from that field rather than pinned:
#
#   max_tokens = clamp(
#       ceil(est_tokens(field_bytes + closing_thought_bytes) * RENDER_FACTOR)
#           + PROSE_ALLOWANCE_TOKENS,
#       EXTRACT_MIN_TOKENS, EXTRACT_MAX_TOKENS_CEILING)
#
# It is arm-invariant by construction: the inputs are the resolved field and the
# closing thought, both of which every arm produces the same way, and no term
# reads the arm, the skill, or the plan.
#
# The constants come from
# evaluation/collections/exp028-extraction-limit-calibration-2026-09-04/ in the
# ido repo, measured over the 712 extraction calls in the Gate 4 v4 collection
# (612 uncut, 69 cut at the limit, 31 with no finish reason recorded).
# Summarised here so a reader does not have to leave the file:
#
#   PROSE_ALLOWANCE_TOKENS  p90 of `completion_tokens` over the 135 uncut
#     extraction calls that had NO presentation payload at all (977.8), rounded
#     up to a multiple of 512. That population is the cost of the answer's PROSE
#     with nothing to render, which is exactly the term a factor multiplied by
#     zero payload has to cover.
#   RENDER_FACTOR  p95 of `completion_tokens / est_tokens(payload)` over the 477
#     uncut calls that DID have a presentation payload (2.041; p90 1.711).
#     p95 and not the max (5.143) because the largest ratios all belong to the
#     smallest payloads — the <250-estimated-token bucket has a median ratio of
#     2.17 while the >=2000 bucket has a median of 0.097 — which is the
#     signature of a fixed prose cost, not of rendering. `PROSE_ALLOWANCE_TOKENS`
#     already pays that cost, so taking the max there would buy the same
#     headroom twice. Measuring the ratio GROSS (allowance not subtracted first)
#     is itself deliberate slack: the residual ratio after subtracting a
#     1024-token allowance has p99 1.117, so 2.0 is nearly twice the worst
#     rendering cost actually observed.
#   EXTRACT_MIN_TOKENS  the deployment's existing 4096. The derivation may only
#     ever RAISE the limit; a call that would compute less than today's constant
#     gets today's constant, so nothing this introduces can shorten an answer
#     that fits now. It is also sufficient on its own for every uncut call in
#     the corpus, whose longest answer was 4021 tokens.
#   EXTRACT_MAX_TOKENS_CEILING  the formula evaluated at the largest field the
#     runtime can hand the call: `presented_max_bytes()`'s 32 KiB default plus
#     the 1200-char closing-thought bound is 18,008 tokens, rounded up to
#     20 x 1024. Deliberately NOT taken from v4's largest observed presentation
#     payload, which was 169,978 bytes — five times the cap. v4 predates result
#     handles and put whole observations in the trajectory; what the extraction
#     call now receives is a field the resolver has already trimmed, so the cap
#     and not the raw demand is what has to be covered. The ceiling therefore
#     binds only if FW_PRESENTED_RESULT_MAX_BYTES is raised without re-deriving
#     it, which is precisely the drift a backstop is for.
#
# What the constants would have done to the corpus, with the payload first
# capped at 32 KiB as the resolver now caps it: 67 of the 69 cut calls would
# have been given more than 4096 tokens (p50 12,806, max 18,008), and the two
# that would not are the two that had no presentation payload at all. Of the 612
# uncut calls, 298 stay at the floor and 314 are given headroom they did not
# need — which costs nothing, because `max_tokens` is a limit and not a target.
#
# est_tokens is bytes/4. Measured on this corpus the model's own OUTPUT runs
# 3.35 bytes/token (p50; p10 2.52, p90 3.91), so bytes/4 UNDER-counts the tokens
# a byte of payload becomes by about 20% — and the payload, dense tabular rows
# full of 32-hex-digit UIDs, tokenizes worse still than prose. Keeping the round
# 4 rather than fitting the measured 3.35 leaves that under-count in place on
# purpose: it is absorbed by a RENDER_FACTOR calibrated against the SAME
# estimator, so the pair is self-consistent, and a reader changing one is told
# by this comment that the other was fitted to it.
# Re-fitted 2026-09-05 on the first Gate 4 v5 shadow block (ido-mn1.6.26;
# evaluation/collections/exp028-extraction-limit-calibration-v2-2026-09-05/).
# The 2026-09-04 calibration measured the model's output at 3.35 bytes per
# token (p10 2.52) on v4 answers and kept the round 4 because RENDER_FACTOR was
# fitted against that estimator. v5 answers render `presented_results`, which
# is denser still -- 32-hex-digit UIDs in markdown tables -- and the first v5
# block measured 2.75 bytes per token at the median and 2.22 at p05 over 56
# extraction calls; 4 of them (both arms B and C, one leaf each) were cut at
# limits the formula had derived at 4 bytes per token, and one cut leaf censors
# its whole cell. BYTES_PER_TOKEN is therefore set to the measured v5 p05,
# rounded down, so `est_tokens` no longer under-counts the payload; RENDER_FACTOR
# is left at 2.0, which is now slack over the payload rather than a fit, and
# the ceiling is the same formula re-evaluated at the same largest field
# (32 KiB + 1200 B): 33,968 / 2.0 * 2.0 + 1024 = 34,992, rounded up to
# 36 x 1024. Its timeout at 35 tok/s is 1,610 s, inside the 4,700 s turn
# deadline after the 1,800 s wall.
EXTRACT_RENDER_FACTOR = 2.0
EXTRACT_PROSE_ALLOWANCE_TOKENS = 1024
EXTRACT_MIN_TOKENS = 4096
EXTRACT_MAX_TOKENS_CEILING = 36864
EXTRACT_BYTES_PER_TOKEN = 2.0

# The derived limit's own timeout. A raised completion limit that keeps the
# role's attempt bound is not a raised limit: the provider is simply killed
# mid-generation instead of stopping at `max_tokens`, and the turn pays the full
# attempt for nothing. 35 tok/s is the rate measured on
# bedrock/us.anthropic.claude-sonnet-4-6 that `_ROLE_ATTEMPT_TIMEOUTS` is
# already sized from; SAFETY covers a provider running at two thirds of it, and
# TTFT covers the time before the first byte of a non-streaming response on a
# large prompt. The role's own attempt bound is a FLOOR, never a ceiling — a
# derived timeout shorter than the bound the role already promised would be a
# regression dressed up as a derivation.
EXTRACT_GEN_TOKENS_PER_SECOND = 35.0
EXTRACT_TIMEOUT_SAFETY = 1.5
EXTRACT_TTFT_ALLOWANCE_SECONDS = 30.0


def _extract_env_string(var: str) -> Optional[str]:
    """The process env, then the workflow env, else None (see `_extract_env_number`)."""
    raw = os.environ.get(var)
    if raw is None or not str(raw).strip():
        try:
            import fastworkflow

            raw = fastworkflow._env_vars.get(var)
        except Exception:  # pragma: no cover - no workflow env loaded
            raw = None
    text = str(raw).strip() if raw is not None else ""
    return text or None


def provider_max_output_tokens(model: Optional[str] = None) -> Optional[int]:
    """The provider's own completion ceiling for the agent model, or None.

    2026-09-05 (Gate 4 v6). The shipped `EXTRACT_MAX_TOKENS_CEILING` was
    re-derived for a route whose provider accepts it; the production route
    (`cerebras/gpt-oss-120b`) accepts at most 32,768 output tokens, and a
    `max_tokens` above that is a rejected request, not a longer answer. The
    limit is read from LiteLLM's model registry for the configured `LLM_AGENT`
    route so the bound follows the model rather than a constant; None when the
    registry does not know the route, in which case only the shipped ceiling
    applies and the bound says so.
    """
    route = model or _extract_env_string("LLM_AGENT")
    if not route:
        return None
    try:
        import litellm

        info = litellm.get_model_info(route)
    except Exception:
        return None
    value = (info or {}).get("max_output_tokens") or (info or {}).get("max_tokens")
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _extract_env_number(var: str, default: float) -> float:
    """A positive float from the process env, then the workflow env, else `default`.

    Read per call for the reason `result_handles._positive_int_env` gives: a
    deployment (or a test) changes one of these without rebuilding a session.
    A non-positive or unparseable value falls back rather than disabling the
    bound — "0" almost always means a mis-set variable, and a zero completion
    limit is not a smaller answer, it is no answer.
    """
    raw = os.environ.get(var)
    if raw is None or not str(raw).strip():
        try:
            import fastworkflow

            raw = fastworkflow._env_vars.get(var)
        except Exception:  # pragma: no cover - defensive
            raw = None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


@dataclass(frozen=True)
class ExtractionBound:
    """The per-call completion limit for one extraction, and how it got there.

    Carries its own inputs because the number alone is not reviewable: a reader
    asking why an answer was 4096 tokens long cannot tell a floor from a
    derivation from a ceiling without them, and that is the same question the
    69 truncated v4 calls could not answer.
    """

    max_tokens: int
    timeout_s: float
    field_bytes: int
    thought_bytes: int
    render_factor: float
    prose_allowance_tokens: int
    derived_tokens: int
    floor_applied: bool
    ceiling_applied: bool
    timeout_clamped: bool
    # ido-mn1.6.33. `timeout_s` is what the call gets; these two are what it
    # asked for and what the turn had left. Without both, a clamped bound and
    # an unclamped one that happen to agree are the same record, and the
    # question "was this answer shortened by the deadline?" has no evidence.
    derived_timeout_s: float = 0.0
    deadline_remaining_s: Optional[float] = None
    # True when what remains of the turn cannot buy even a floor-sized attempt,
    # so starting one would spend the provider call and be killed mid-stream.
    deadline_insufficient: bool = False
    # 2026-09-05 (Gate 4 v5 shadow blocks). The bytes of the formatted
    # trajectory the extraction call is given alongside the field. A flat turn
    # (arm A) composes its whole answer from that trajectory, and its
    # `presented_results` field can be a few hundred bytes while the trajectory
    # runs to tens of KB: sized from the field alone, its limit fell to the
    # floor and a seven-subtask answer was cut. Zero when the caller did not
    # measure it, so every bound taken before this term stays comparable.
    trajectory_bytes: int = 0
    # 2026-09-05 (Gate 4 v6). The provider's completion ceiling for the agent
    # route when LiteLLM knows it, and whether it bound this call. A ceiling
    # above what the provider accepts is a rejected request; below it, the
    # shipped ceiling still applies.
    provider_max_output_tokens: Optional[int] = None
    provider_cap_applied: bool = False

    def as_evidence(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "timeout_s": round(self.timeout_s, 3),
            "field_bytes": self.field_bytes,
            "thought_bytes": self.thought_bytes,
            "trajectory_bytes": self.trajectory_bytes,
            "provider_max_output_tokens": self.provider_max_output_tokens,
            "provider_cap_applied": self.provider_cap_applied,
            "render_factor": self.render_factor,
            "prose_allowance_tokens": self.prose_allowance_tokens,
            "derived_tokens": self.derived_tokens,
            "floor_applied": self.floor_applied,
            "ceiling_applied": self.ceiling_applied,
            "timeout_clamped": self.timeout_clamped,
            "derived_timeout_s": round(self.derived_timeout_s, 3),
            "deadline_remaining_s": (
                None
                if self.deadline_remaining_s is None
                else round(self.deadline_remaining_s, 3)
            ),
            "deadline_insufficient": self.deadline_insufficient,
        }

    def as_call_config(self) -> dict[str, Any]:
        """The per-call kwargs for exactly this one `dspy` invocation.

        Passed as `config=` so it reaches `LM.forward` for THIS call only.
        Mutating the shared LM instead would be wrong twice over: the same LM
        object serves the react step, whose limit ido-mn1.6.4 deliberately did
        not raise, and it is shared across turns and threads, so a mutation is a
        race as well as a policy change.
        """
        return {"max_tokens": self.max_tokens, "timeout": self.timeout_s}


def _role_attempt_floor() -> float:
    """The attempt bound the active LM's role already promises.

    Read off the live LM rather than from the role table directly because that
    is the object whose policy will actually be applied, and a floor taken from
    a different role would be a floor for a call nobody is making. Falls back to
    the module default when the LM is a stub or was built outside `get_lm`.
    """
    try:
        from fastworkflow.utils.dspy_utils import _DEFAULT_ATTEMPT_TIMEOUT

        policy = getattr(dspy.settings.lm, "role_policy", None)
        seconds = getattr(policy, "attempt_seconds", None)
        return float(seconds) if seconds else float(_DEFAULT_ATTEMPT_TIMEOUT)
    except Exception:  # pragma: no cover - defensive
        return 60.0


# ido-mn1.6.33. The reason for the extraction to bear the derived timeout is
# the reason a doomed attempt must not be started: a raised completion limit
# under a bound too short to generate it does not shorten the answer, it kills
# the HTTP attempt mid-generation and returns NOTHING. That is strictly worse
# than a clean `finish_reason=length`, and it arrives after the call has been
# paid for. So the remaining deadline is read in two places with two different
# answers, and they are deliberately not the same test:
#
#   * remaining < the DERIVED timeout  -> clamp. The answer is shortened, which
#     is a measurable, arm-invariant degradation and still an answer.
#   * remaining < the FLOOR attempt    -> refuse. Below the time one
#     deployment-floor answer takes to generate, there is no bound left to
#     shorten to, and the only outcomes are a killed stream or nothing at all.
#
# The floor attempt is `EXTRACT_MIN_TOKENS` at the same rate, safety and TTFT
# the derivation uses everywhere else, so the two thresholds move together when
# a constant is retuned and cannot drift apart.
def minimum_viable_extraction_seconds() -> float:
    """The shortest extraction attempt that can still return an answer."""
    tokens_per_second = _extract_env_number(
        "FW_EXTRACT_GEN_TOKENS_PER_SECOND", EXTRACT_GEN_TOKENS_PER_SECOND
    )
    safety = _extract_env_number("FW_EXTRACT_TIMEOUT_SAFETY", EXTRACT_TIMEOUT_SAFETY)
    ttft = _extract_env_number(
        "FW_EXTRACT_TTFT_ALLOWANCE_SECONDS", EXTRACT_TTFT_ALLOWANCE_SECONDS
    )
    floor_tokens = int(
        _extract_env_number("FW_EXTRACT_MIN_TOKENS", EXTRACT_MIN_TOKENS)
    )
    return math.ceil(floor_tokens / tokens_per_second * safety) + ttft


def extraction_bound(
    field_bytes: int, thought_bytes: int, trajectory_bytes: int = 0
) -> ExtractionBound:
    """Size ONE extraction call from the payload it has been given.

    Call it inside the `external_operations.operation` block the extraction runs
    in: the timeout is clamped to whatever remains of the turn deadline, and
    outside the block there is no deadline to clamp against — which is exactly
    the decorative clamp `_remaining_call_budget` exists to explain.
    """
    factor = _extract_env_number("FW_EXTRACT_RENDER_FACTOR", EXTRACT_RENDER_FACTOR)
    allowance = int(
        _extract_env_number(
            "FW_EXTRACT_PROSE_ALLOWANCE_TOKENS", EXTRACT_PROSE_ALLOWANCE_TOKENS
        )
    )
    floor = int(_extract_env_number("FW_EXTRACT_MIN_TOKENS", EXTRACT_MIN_TOKENS))
    ceiling = int(
        _extract_env_number(
            "FW_EXTRACT_MAX_TOKENS_CEILING", EXTRACT_MAX_TOKENS_CEILING
        )
    )
    bytes_per_token = _extract_env_number(
        "FW_EXTRACT_BYTES_PER_TOKEN", EXTRACT_BYTES_PER_TOKEN
    )
    # A ceiling below the floor is a mis-set pair, not an instruction to emit a
    # limit smaller than the deployment's own constant. The floor wins: it is
    # the number that cannot make anything worse than it already is.
    ceiling = max(ceiling, floor)
    # The provider's own ceiling wins over both: a request above it is refused
    # outright. When it is below the floor the floor gives way too, because a
    # floor the provider will not serve is not a floor.
    provider_cap = provider_max_output_tokens()
    provider_cap_applied = False
    if provider_cap is not None and provider_cap < ceiling:
        ceiling = provider_cap
        provider_cap_applied = True
    floor = min(floor, ceiling)

    # What the call is asked to compose from: the resolved field, the closing
    # thoughts, and (2026-09-05) the formatted trajectory it receives as its
    # `trajectory` input. An answer asked to render what it was given cannot
    # be shorter than what it was given; a flat turn's answer is rendered from
    # the trajectory, not from the field.
    payload_bytes = (
        max(0, int(field_bytes))
        + max(0, int(thought_bytes))
        + max(0, int(trajectory_bytes))
    )
    est_tokens = payload_bytes / bytes_per_token
    derived = math.ceil(est_tokens * factor) + allowance
    max_tokens = min(max(derived, floor), ceiling)

    tokens_per_second = _extract_env_number(
        "FW_EXTRACT_GEN_TOKENS_PER_SECOND", EXTRACT_GEN_TOKENS_PER_SECOND
    )
    safety = _extract_env_number("FW_EXTRACT_TIMEOUT_SAFETY", EXTRACT_TIMEOUT_SAFETY)
    ttft = _extract_env_number(
        "FW_EXTRACT_TTFT_ALLOWANCE_SECONDS", EXTRACT_TTFT_ALLOWANCE_SECONDS
    )
    generation_s = math.ceil(max_tokens / tokens_per_second * safety) + ttft
    derived_timeout_s = max(_role_attempt_floor(), generation_s)
    timeout_s = derived_timeout_s
    remaining = external_operations.remaining_seconds()
    clamped = external_operations.clamp_timeout(timeout_s)
    timeout_clamped = clamped is not None and float(clamped) < timeout_s
    if clamped is not None:
        timeout_s = max(1.0, float(clamped))
    insufficient = (
        remaining is not None
        and remaining < minimum_viable_extraction_seconds()
    )

    return ExtractionBound(
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        field_bytes=max(0, int(field_bytes)),
        thought_bytes=max(0, int(thought_bytes)),
        trajectory_bytes=max(0, int(trajectory_bytes)),
        provider_max_output_tokens=provider_cap,
        provider_cap_applied=provider_cap_applied,
        render_factor=factor,
        prose_allowance_tokens=allowance,
        derived_tokens=derived,
        floor_applied=derived < floor,
        ceiling_applied=derived > ceiling,
        timeout_clamped=timeout_clamped,
        derived_timeout_s=derived_timeout_s,
        deadline_remaining_s=remaining,
        deadline_insufficient=insufficient,
    )


class MissingTurnBudgetError(RuntimeError):
    """``forward()`` was called without the logical turn's budget.

    Architecture §6.4: WEC creates the budget at fresh logical-turn start and
    passes the same object to the planner and to ReAct; ``forward()`` requires
    it and never creates or resets one. Defaulting one here is precisely the
    defect FW-REQ-001 closes — an agent that mints its own budget is an agent
    whose budget nobody can attribute to a turn.
    """


class NoSuspendedAgentStateError(RuntimeError):
    """Resume requested but no suspended ReAct trajectory exists.

    Happens when ``_awaiting_user`` is set after the trajectory was already
    consumed (e.g. a deferred resume still in flight that later failed, or a
    restored blob that lost ``react``) so a second message cannot honestly
    continue the turn. Embedders map this to HTTP 409 Conflict — not 500.
    """


class fastWorkflowReAct(Module):
    #: The executing skill's optional `presents:` OVERRIDE, set per call by the
    #: caller that knows which skill is running. Empty — every arm-A turn, and
    #: every skill that declares nothing — means the arm-invariant default
    #: applies instead: the producing command's own
    #: `ResultHandleSpec.presentation` flag, which every arm can read because it
    #: lives on the stored handle rather than in `_skills/`.
    #:
    #: Declared on the CLASS so that an instance built without `__init__` —
    #: which is how the suspend/resume tests build one — reads the same honest
    #: default rather than raising inside the resolver.
    presentation_commands: frozenset[str] = frozenset()

    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10,
                 on_step_complete: Callable[[int, dict], bool] | None = None,
                 decision_point: "PolicyDecisionPoint | None" = None,
                 contract_facts: "ContractFacts | None" = None):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        # Retained as the *declared default* a caller can read, not as the
        # control: the loop is bounded by the LogicalTurnBudget its caller
        # supplies. Kept because `initialize_workflow_tool_agent(max_iters=...)`
        # is a public constructor argument and removing it would break callers
        # for no gain -- WEC resolves the real limit through RuntimeConfig.
        self.max_iters = max_iters
        # The active logical turn's budget. None between turns: this object
        # outlives a turn, which is exactly why it must not own the counter.
        self._budget: LogicalTurnBudget | None = None
        # FW-REQ-017's single decision point (EXP-025a G3 ADR). Defaulting to a
        # bare PolicyDecisionPoint gives the clause-5 no-op: OFF mode over the
        # empty table, which decides nothing and is byte-for-byte the behaviour
        # this loop had before the hook existed.
        self.decision_point = decision_point or PolicyDecisionPoint()
        # Contract facts for the turn under way, set by the caller that knows
        # them (WEC / the workflow host). Empty facts read as effect_kind
        # "unknown", which §6.6.1 requires to mean write-capable — so a caller
        # that forgets to supply them gets caution, never a free proceed.
        self.contract_facts = contract_facts or ContractFacts()

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                f"next_thought is a working note, not a deliverable: keep it under {MAX_NEXT_THOUGHT_CHARS} characters.",
                "Never render the report, tables, or full listings inside next_thought. State what you have and cite "
                "the observation indices (observation_0, observation_1, ...) and the handles (uids) that carry those "
                "results; the runtime truncates a longer next_thought and the truncated text is what the rest of the "
                "run sees.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        instr.extend(f"({idx + 1}) {tool}" for idx, tool in enumerate(tools.values()))
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")
        # The closing step is the one that overruns (ido-mn1.6.4). Appended
        # AFTER the tool enumeration and the next_tool_args note so the text
        # those two produce is untouched.
        instr.append(
            "When you select `finish`, next_thought must be one or two sentences that state the task is complete "
            "and name the observation indices and handles holding the results. Do NOT write the final answer, a "
            "summary report, or any table there: a separate extraction step composes the final answer from this "
            "trajectory, so anything you render in next_thought is discarded."
        )
        instr.append(
            "Your reply ends at the `[[ ## completed ## ]]` marker. Write nothing after it."
        )

        # Build the ReAct signature with trajectory input.
        # available_commands is injected into system message by CommandsSystemPreludeAdapter
        # (see fastworkflow/utils/chat_adapter.py) and is NOT included in the trajectory
        # formatting to avoid token bloat across iterations.
        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append(
                "next_thought",
                dspy.OutputField(
                    desc=(
                        f"Working note, at most {MAX_NEXT_THOUGHT_CHARS} characters. Cite observation indices and "
                        "handles; never render the report or a table here. On the `finish` step, one or two "
                        "sentences stating completion and where the results are."
                    )
                ),
                type_=str,
            )
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(
            react_signature,
            **({"stop": list(REACT_STOP_SEQUENCES)} if REACT_STOP_SEQUENCES else {}),
        )
        # `reasoning` is bounded here; `final_answer` is NOT. The extraction
        # call IS the composition step for the flat arm and IS the leaf answer
        # arms B/C concatenate deterministically
        # (`WorkflowExecutionContext._compose_plan_answer` makes no model
        # call), so capping the answer would truncate the deliverable rather
        # than remove waste.
        # `presented_results` is appended AFTER `trajectory` so the resolved rows
        # read as a continuation of the evidence rather than as a second request.
        # Appended to the EXTRACT signature only: `react_signature` above is
        # byte-identical to what ido-mn1.6.4 left, which is what
        # tests/test_react_closing_thought_bound.py pins.
        self.extract = dspy.ChainOfThought(
            fallback_signature.append(
                PRESENTED_RESULTS_FIELD,
                dspy.InputField(desc=PRESENTED_RESULTS_DESC),
                type_=str,
            ),
            rationale_field=dspy.OutputField(
                desc=(
                    f"A short plan, at most {MAX_EXTRACT_REASONING_CHARS} characters, naming which observations "
                    "you are drawing on. Do not restate the trajectory and do not draft the answer here — write "
                    "the answer once, in the output fields below."
                )
            ),
        )

        self.inputs = {}
        self.current_trajectory = {}
        # Observation seals (arch §8.4 phase 3): step index -> digest of the
        # sealed step. What they buy is an answer to "which steps completed?"
        # that does not depend on re-reading a trajectory that a later phase may
        # have truncated, and a durable logical-call key per step so a repeated
        # decision at the same step joins the existing call instead of minting a
        # second effect.
        self._step_seals: dict[int, dict[str, Any]] = {}
        # Handle ids each step's thought named, captured before the thought was
        # bounded. Ordered by step; the closing window is a slice off the end.
        self._step_citations: list[tuple[str, ...]] = []
        self._on_step_complete = on_step_complete
        self._suspended: dict[str, Any] | None = None
        # True when the most recent _run_loop ended because max_iters was
        # reached without the agent selecting the `finish` tool.
        self._exhausted_last_run = False
        # EXP-028's emergency wall/no-progress envelope is independent of the
        # iteration budget. It is optional on ordinary turns and shared by all
        # planner arms when a stress collection supplies it.
        self._safety_envelope: SafetyEnvelopeState | None = None
        self._censored_last_run = False

    # ------------------------------------------------------------------
    # Turn budget (arch §6.4)
    # ------------------------------------------------------------------

    @property
    def budget(self) -> LogicalTurnBudget | None:
        """The active logical turn's budget, or None between turns."""
        return self._budget

    @property
    def iteration_counter(self) -> int:
        """Read-only compatibility view over the active budget's spend.

        Was a module-lifetime mutable counter, and the two things done to it —
        reading ``<= 0`` as "this command came from the user" and writing ``-1``
        on every clarification — are the defect (FW-REQ-001 clauses 1 and 3).
        Origin is now stated explicitly by ``workflow_agent.InvocationOrigin``,
        and the budget is per turn, so this survives only for readers that want
        the number. There is deliberately no setter: an assignment would be a
        caller trying to steer the budget from outside the turn that owns it.
        """
        return self._budget.iterations_consumed if self._budget is not None else 0

    def _require_budget(self) -> LogicalTurnBudget:
        if self._budget is None:
            raise MissingTurnBudgetError(
                "fastWorkflowReAct requires the logical turn's LogicalTurnBudget; "
                "the caller that begins the turn must supply it (arch §6.4)"
            )
        return self._budget

    def clear_suspension(self) -> None:
        """Drop any in-memory suspended ReAct state (used on abort/finalize)."""
        self._suspended = None

    def export_suspended(self) -> dict[str, Any] | None:
        """Return a JSON-serializable copy of suspended ReAct state, or None."""
        if self._suspended is None:
            return None
        blob = {
            "trajectory": dict(self._suspended["trajectory"]),
            "idx": self._suspended["idx"],
            "input_args": dict(self._suspended["input_args"]),
            "max_iters": self._suspended["max_iters"],
            "clarification": self._suspended.get("clarification"),
            # Schema 8: a resumed leaf must finish under the same skill-level
            # `presents:` override as its suspended half. Falling back to the
            # class default here can silently omit rows and then let the
            # aggregate read the shorter answer as complete.
            "presentation_commands": sorted(self.presentation_commands),
            # Schema 4: the suspended turn carries its whole budget, so resume
            # restores what the turn had left rather than reconstructing a
            # number. `iteration_counter` is still written for a schema-3
            # reader, and is the only field a v3 build could have used.
            "iteration_counter": self.iteration_counter,
        }
        if self._budget is not None:
            blob["budget"] = self._budget.to_state()
        if self._step_seals:
            # Keyed by string because JSON has no integer keys; restored back to
            # ints on import. Without this a cross-process resume would report
            # zero completed steps and a later typed failure would carry no
            # evidence of the work the turn had already done.
            blob["step_seals"] = {
                str(idx): dict(seal) for idx, seal in self._step_seals.items()
            }
        if citations := getattr(self, "_step_citations", None):
            # The ids the suspended half cited BEFORE its thoughts were bounded.
            # Without this a cross-process resume falls back to re-reading the
            # bounded trajectory, and a citation lost to the truncation stays
            # lost (ido-mn1.6.6).
            blob["step_citations"] = [list(ids) for ids in citations]
        return blob

    def import_suspended(self, data: dict[str, Any]) -> None:
        """Restore suspended ReAct state from export_suspended() output.

        Accepts both shapes. A schema-4 blob carries ``budget`` and restores it
        unchanged. A schema-3 blob carries only ``iteration_counter``, which is
        restored into an explicit ``LegacyTurnBudget`` (arch §9.2) — the counter
        is kept, the reconstruction is not claimed, and the turn stays pinned to
        legacy semantics until it completes or is cancelled.

        Raises ``ValueError`` for state it cannot rebuild exactly; the caller
        turns that into a fail-closed restore rather than resuming a turn on a
        budget nobody set.
        """
        presentation_commands = data.get("presentation_commands", ())
        if not isinstance(presentation_commands, (list, tuple)) or any(
            not isinstance(command, str) or not command
            for command in presentation_commands
        ):
            raise ValueError(
                "presentation_commands must be a list of non-empty command names"
            )
        self.presentation_commands = frozenset(presentation_commands)
        self._suspended = {
            "trajectory": dict(data["trajectory"]),
            "idx": data["idx"],
            "input_args": dict(data["input_args"]),
            "max_iters": data["max_iters"],
            "clarification": data.get("clarification"),
        }
        self._step_seals = {
            int(idx): dict(seal)
            for idx, seal in (data.get("step_seals") or {}).items()
        }
        self._step_citations = [
            tuple(ids) for ids in (data.get("step_citations") or ())
        ]
        if (budget_state := data.get("budget")) is not None:
            self._budget = budget_from_state(budget_state)
        else:
            self._budget = LegacyTurnBudget.from_counter(
                data.get("iteration_counter", 0),
                int(data["max_iters"]) if data.get("max_iters") else self.max_iters,
            )

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    @DSPyForward.intercept
    def forward(self, **input_args):
        """Run one fresh logical turn against the caller's budget.

        ``budget`` is a required keyword. It is popped before the remaining
        arguments reach the DSPy signature, exactly as ``max_iters`` was, and it
        is neither created nor reset here (arch §6.4).
        """
        budget = input_args.pop("budget", None)
        if budget is None:
            raise MissingTurnBudgetError(
                "fastWorkflowReAct.forward() requires budget=LogicalTurnBudget(...); "
                "the caller that begins the logical turn owns it (arch §6.4)"
            )
        self._safety_envelope = input_args.pop("safety_envelope", None)
        self._budget = budget
        if (
            not budget.enforce_iteration_limit
            and (
                self._safety_envelope is None
                or not self._safety_envelope.enabled
            )
        ):
            raise ValueError(
                "an unlimited iteration budget requires an enabled safety envelope"
            )
        self.inputs = input_args
        self.clear_suspension()

        # Reset the full-trajectory mirror at the start of each logical turn.
        # resume() must NOT reset it, so a suspended->resumed turn accumulates one
        # coherent trajectory. current_trajectory is a SEPARATE object from the
        # working `trajectory` below (which is what gets stashed in _suspended),
        # so mirroring into it never corrupts suspend/resume bookkeeping.
        self.current_trajectory = {}
        # Per logical turn, like current_trajectory: resume() must NOT clear
        # these, or a resumed turn would forget which steps it had completed.
        self._step_seals = {}
        # Same rule, same reason: a resumed turn's closing step must still be
        # able to see the handles its suspended half cited (ido-mn1.6.6).
        self._step_citations = []

        trajectory: dict[str, Any] = {}
        # Accepted and discarded: a stale caller passing max_iters must not
        # silently override the turn's budget, and must not reach the DSPy
        # signature either.
        input_args.pop("max_iters", None)
        idx = 0
        exception_count = 0

        suspended = self._run_loop(
            trajectory, idx, input_args, budget, exception_count
        )
        if suspended is not None:
            return suspended
        if self._censored_last_run:
            return self._censored_prediction(trajectory)

        return self._finish(trajectory, input_args, budget)

    def resume(
        self,
        observation: str,
        safety_envelope: SafetyEnvelopeState | None = None,
    ):
        """Resume a suspended run after the user answered an ask_user clarification.

        Ordering is the contract (arch §8.4): the suspension is not discarded
        until the answer has been appended to the same logical turn and the
        suspended step has been sealed. A failed decision parse after that point
        retries only that prediction — ``_decide`` owns it — and cannot consume
        the suspended state a second time, because the state is already gone and
        the answer is already in the trajectory it was consumed into.

        The stash is marked ``answer_appended`` before being dropped, so a
        process that dies between the append and the drop leaves evidence of
        which half happened rather than an ambiguous blob.
        """
        if self._suspended is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )

        stash = self._suspended
        trajectory = stash["trajectory"]
        idx = stash["idx"]
        input_args = stash["input_args"]
        # The suspended turn's own budget, restored unchanged. A clarification
        # answer does not replenish it (arch §6.4) — which is what the old
        # `iteration_counter = -1` did on every round-trip.
        budget = self._require_budget()
        effective_safety = (
            safety_envelope
            if safety_envelope is not None
            else getattr(self, "_safety_envelope", None)
        )
        if (
            not budget.enforce_iteration_limit
            and (
                effective_safety is None
                or not effective_safety.enabled
            )
        ):
            raise ValueError(
                "an unlimited iteration budget requires an enabled safety envelope"
            )
        self._safety_envelope = effective_safety

        # Keep self.inputs pointing at the active run's arg dict so any mid-run refresh
        # (e.g. available_commands re-scoping after a context switch) mutates the same
        # dict this loop unpacks on each step.
        self.inputs = input_args

        trajectory[f"observation_{idx}"] = observation
        # Mirror the resumed observation (the user's ask_user answer) into
        # current_trajectory. Without this the highest-value context — what the
        # user said in response to the clarification — would be missing from the
        # trajectory the planner and distillation see.
        self.current_trajectory[f"observation_{idx}"] = observation

        # Seal the step that suspended, now that its observation exists. The
        # ask_user call is a tool call like any other and gets the same durable
        # logical-call key, so a resumed step is not a hole in the seal record.
        host = tracing.current_host()
        self._seal_step(
            idx,
            self.logical_call_key(
                tracing.get_turn_key(host) if host is not None else None,
                idx,
                trajectory.get(f"tool_name_{idx}", "ask_user"),
                trajectory.get(f"tool_args_{idx}", {}),
            ),
            trajectory.get(f"thought_{idx}"),
            trajectory.get(f"tool_name_{idx}", "ask_user"),
            trajectory.get(f"tool_args_{idx}", {}),
            observation,
        )
        stash["answer_appended"] = True

        idx += 1
        # The suspending step is charged here rather than at suspension: the
        # loop returns before its own increment when ask_user fires, so this is
        # that step's iteration, not a new one bought by resuming.
        budget.consume_iteration()
        # Only now. Everything above is what "durably appended to the same
        # logical turn" means for this object; dropping the stash first would
        # make a failure between the two indistinguishable from a turn that was
        # never resumed.
        self._suspended = None

        suspended = self._run_loop(trajectory, idx, input_args, budget, 0)
        if suspended is not None:
            return suspended
        if self._censored_last_run:
            return self._censored_prediction(trajectory)

        return self._finish(trajectory, input_args, budget)

    # ------------------------------------------------------------------
    # The phase machine (arch §8.4)
    # ------------------------------------------------------------------

    @property
    def step_seals(self) -> dict[int, dict[str, Any]]:
        """The sealed steps of the current turn, by step index."""
        return dict(self._step_seals)

    @staticmethod
    def _is_parse_failure(err: BaseException) -> bool:
        """Whether this is the adapter/model parse failure the decision phase owns.

        Imported at call time: `dspy.utils.exceptions` is not a stable public
        path, and a missing symbol must degrade to "not a parse failure" rather
        than break the loop.
        """
        try:
            from dspy.utils.exceptions import AdapterParseError
        except ImportError:  # pragma: no cover - dspy layout change
            return False
        return isinstance(err, AdapterParseError)

    @staticmethod
    def _bound_text(text: Any, limit: int, notice: str) -> Any:
        """Truncate `text` to `limit` characters, deterministically.

        Returns the value unchanged (identity, not a copy) when it is not an
        over-long string, so callers can test `is` to decide whether anything
        happened. No retry and no model call: the tokens are already spent by
        the time this runs, and asking the model again for a shorter field is
        the retry storm this is meant to avoid.
        """
        if not isinstance(text, str) or len(text) <= limit:
            return text
        keep = max(0, limit - len(notice))
        return text[:keep] + notice

    def _note_thought_citations(self, thought: Any) -> None:
        """Record the handle ids one step's thought named, BEFORE it is bounded.

        `MAX_NEXT_THOUGHT_CHARS` truncates mid-token, so a closing thought that
        ran long can lose a 32-character id to a cut that lands inside it — and
        a half id is not a citation. Parsing here, at the choke point every step
        passes through and before `_bound_text` runs, is what makes the citation
        rule survive the bound; `_cited_handle_ids` prefers this record and falls
        back to re-reading the (bounded) trajectory when there is none, which is
        what a cross-process resume of a pre-ido-mn1.6.6 blob has.

        One entry per accepted decision, in step order, so the closing window is
        a slice off the end. `_decide` retries only PARSE failures and never
        reaches here on one, so a retried step is recorded once.
        """
        recorded = getattr(self, "_step_citations", None)
        if recorded is None:
            recorded = self._step_citations = []
        recorded.append(result_handles.parse_handle_ids(thought))

    def _bound_prediction_thought(self, pred):
        """Cap `next_thought` in place. Touches no other field.

        `next_tool_name` and `next_tool_args` are read and written by nothing
        here, so the tool step is byte-identical to what the model produced.
        """
        if pred is None:
            return pred
        thought = getattr(pred, "next_thought", None)
        self._note_thought_citations(thought)
        bounded = self._bound_text(
            thought, MAX_NEXT_THOUGHT_CHARS, THOUGHT_TRUNCATION_NOTICE
        )
        if bounded is not thought:
            logger.warning(
                "next_thought exceeded %d characters (%d); truncated",
                MAX_NEXT_THOUGHT_CHARS, len(thought),
            )
            try:
                pred.next_thought = bounded
            except Exception:  # pragma: no cover - exotic prediction objects
                logger.warning("could not bound next_thought on %r", type(pred))
        return pred

    def _bound_extraction(self, extract):
        """Cap the extraction step's `reasoning`. Never touches `final_answer`.

        `final_answer` IS the composed deliverable — arm A returns it directly
        and arms B/C concatenate it per leaf in
        `WorkflowExecutionContext._compose_plan_answer`, which makes no model
        call. Capping it would truncate the answer, not remove waste. Only the
        CoT scratchpad is bounded.
        """
        if extract is None:
            return extract
        is_mapping = isinstance(extract, dict)
        reasoning = (
            extract.get("reasoning") if is_mapping
            else getattr(extract, "reasoning", None)
        )
        bounded = self._bound_text(
            reasoning, MAX_EXTRACT_REASONING_CHARS, REASONING_TRUNCATION_NOTICE
        )
        if bounded is not reasoning:
            logger.warning(
                "extraction reasoning exceeded %d characters (%d); truncated",
                MAX_EXTRACT_REASONING_CHARS, len(reasoning),
            )
            try:
                if is_mapping:
                    extract["reasoning"] = bounded
                else:
                    extract.reasoning = bounded
            except Exception:  # pragma: no cover - exotic prediction objects
                logger.warning("could not bound reasoning on %r", type(extract))
        return extract

    def _decide(self, trajectory, input_args, budget):
        """Phase 1: get a tool decision, retrying only the parse.

        Bounded by ``DECISION_PARSE_ATTEMPTS``. Each provider attempt consumes
        model-call budget (arch §8.4), so a turn cannot buy unbounded model
        calls by failing to parse. Nothing has executed at this point, which is
        the whole reason retry is safe *here* and was not safe where it used to
        live.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(DECISION_PARSE_ATTEMPTS):
            budget.consume_model_call()
            try:
                # The decision call, bounded per attempt: `attempts` here is the
                # phase's own retry (arch §8.4), and each attempt gets the class
                # deadline rather than the three of them sharing one.
                external_operations.require_time("before agent decision")
                with external_operations.operation("model.agent"):
                    # Bounded here, at the single choke point every sync step
                    # goes through, so the trajectory, the seal, the step span
                    # and the planner mirror all see the same bounded thought.
                    return self._bound_prediction_thought(
                        self._call_with_potential_trajectory_truncation(
                            self.react, trajectory, **input_args
                        )
                    )
            except BaseException as err:
                if not self._is_parse_failure(err):
                    raise
                last_error = err
                logger.warning(
                    "Decision parse failed (attempt %d/%d): %s",
                    attempt + 1, DECISION_PARSE_ATTEMPTS, err,
                )
        raise TurnFailedError(
            TypedFailure(
                disposition="permanent",
                code=CODE_ADAPTER_PARSE,
                detail=f"no parseable tool decision in {DECISION_PARSE_ATTEMPTS} "
                       f"attempts: {last_error}",
            )
        )

    @staticmethod
    def logical_call_key(turn_key: Optional[str], idx: int, tool_name: str,
                         tool_args: Any) -> str:
        """The durable identity of one tool call at one step of one turn.

        Derived rather than minted, so the SAME decision at the same step of the
        same turn produces the same key however many times it is reached — which
        is what lets a duplicate decision join an existing record or operation
        instead of creating a second effect (arch §8.4 last paragraph). The
        operation journal (EXP-014) is the consumer; this slice's job is that
        the key exists and is stable.

        Falls back to a turn-less key outside an observed turn rather than
        raising: an unobserved run still needs step identity, it just cannot
        claim turn scope.
        """
        payload = json.dumps(
            {"turn": turn_key or "", "step": idx, "tool": tool_name,
             "args": tool_args},
            sort_keys=True, default=repr,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _seal_step(self, idx: int, logical_call_key: str, thought, tool_name,
                   tool_args, observation) -> dict[str, Any]:
        """Phase 3: record what this step did, immutably.

        The digest covers the decision AND its observation, so a seal cannot be
        reconciled with a different observation later. Sealing happens after the
        tool returns and before anything reads the step back.
        """
        digest_source = json.dumps(
            {"thought": thought, "tool": tool_name, "args": tool_args,
             "observation": _as_text(observation)},
            sort_keys=True, default=repr,
        )
        seal = {
            "logical_call_key": logical_call_key,
            "digest": hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:32],
        }
        self._step_seals[idx] = seal
        return seal

    def _turn_partial(self, trajectory, budget):
        """The declared partial for a turn that stopped short, or None.

        Built from the budget's counters and from `trajectory`, which the LOOP
        wrote — a `tool_name_N` entry exists because a tool actually ran, not
        because the model said so. That provenance is the point (EXP-027).
        """
        if not getattr(self, "_exhausted_last_run", False):
            return None
        executed = tuple(
            str(value) for key, value in trajectory.items()
            if key.startswith("tool_name_"))
        return TurnPartial(
            reason="budget-exhausted",
            iterations_consumed=budget.iterations_consumed,
            iteration_limit=budget.iteration_limit,
            commands_executed=executed,
        )

    def _censored_prediction(self, trajectory):
        """Return a deterministic infrastructure censor with no extra model call."""
        safety = getattr(self, "_safety_envelope", None)
        reason = (
            safety.censored_reason
            if safety is not None and safety.censored_reason
            else "unknown"
        )
        commands = sum(
            key.startswith("tool_name_") and value != "finish"
            for key, value in trajectory.items()
        )
        return dspy.Prediction(
            trajectory=trajectory,
            exhausted=False,
            censored=True,
            censored_reason=reason,
            final_answer=(
                "Emergency safety envelope reached. This attempt is censored, "
                "not a task failure. "
                f"{commands} tool decision(s) completed before censoring."
            ),
        )

    @staticmethod
    def _exhaustion_notice(partial: TurnPartial) -> str:
        """What the model is told before it writes the final answer.

        Without this the model is handed a truncated trajectory and asked for a
        final answer with no indication it was truncated — it sees four
        completed walks and a request, and reports eight. The instruction is
        explicit about the failure mode rather than merely factual, because
        "you were cut short" and "do not report what you did not retrieve" are
        different instructions and only the second one changes the answer.
        """
        return (
            "%s You did NOT finish the requested work. Report only what you "
            "actually retrieved above, say plainly which parts you did not "
            "reach, and do not present a partial walk as a complete one."
            % partial.summary
        )

    def _consult_finish_policy(self, extract, trajectory, input_args):
        """The BEFORE_FINISH decision, or None when the table says nothing.

        Reads the answer the agent is about to give, which is the whole reason
        this position exists separately from the tool loop: at finish-SELECTION
        time the answer does not exist yet, and the failure being caught lives
        in its wording.
        """
        point = getattr(self, "decision_point", None)
        if point is None or point.mode is PolicyMode.OFF:
            return None
        # An exhausted turn is exempt, and the exemption is load-bearing rather
        # than tidy. `E-defers-read-only-work` catches an answer that hands the
        # operator commands to run — and an honest partial report legitimately
        # says "open the remaining identities". Measured in EXP-027's stage (c):
        # the row fired on 12 of 13 exhausted turns, on exactly the behaviour
        # EXP-027 exists to produce. Its rewrite tells the agent to "carry out
        # that inspection or say it is out of scope", and the agent can do
        # neither: it has no budget left, and the work is in scope. Left alone,
        # the policy pushes an honest partial back toward a confident whole.
        if getattr(self, "_exhausted_last_run", False):
            return None
        answer = ""
        for key in ("final_answer", "answer", "output"):
            value = extract.get(key) if hasattr(extract, "get") else None
            if value:
                answer = str(value)
                break
        if not answer:
            return None
        decision = point.decide(BeforeFinishInput(
            facts=getattr(self, "contract_facts", ContractFacts()),
            utterance=" ".join(str(v) for v in input_args.values()),
            answer=answer,
            observations=tuple(
                str(value) for key, value in trajectory.items()
                if key.startswith("observation_")),
            commands_run=tuple(
                str(value) for key, value in trajectory.items()
                if key.startswith("tool_name_")),
        ))
        if decision is None or decision.outcome is not PolicyOutcome.PROCEED:
            return None
        return decision if decision.rewrite else None

    def _consult_policy(self, pred, trajectory, idx, input_args):
        """Evaluate the decision point for this step, or return None.

        Returns a decision only when the caller should ACT on it — `OFF` and
        `SHADOW` both return None here, `SHADOW` having recorded what it would
        have done. Only outcomes that carry a rewrite are actioned: a table row
        that says `ASK` is the table agreeing with the agent, and there is
        nothing for this loop to do about it.

        Scoped to `ask_user` deliberately. The G3 ADR authorised one decision
        point, not a general interceptor over every tool, and widening it to
        tools whose failure mode nobody has measured is the unexercised
        generality PHASE-2-ORDER §3 warns about.
        """
        if pred.next_tool_name != "ask_user":
            return None
        point = getattr(self, "decision_point", None)
        if point is None or point.mode is PolicyMode.OFF:
            return None
        observations = tuple(
            str(value) for key, value in trajectory.items()
            if key.startswith("observation_"))
        commands = tuple(
            str(value) for key, value in trajectory.items()
            if key.startswith("tool_name_"))
        decision = point.decide(AfterObservationInput(
            facts=getattr(self, "contract_facts", ContractFacts()),
            utterance=" ".join(str(v) for v in input_args.values()),
            pending_tool=pred.next_tool_name,
            pending_args=pred.next_tool_args or {},
            observations=observations,
            commands_run=commands,
        ))
        if decision is None or decision.outcome is not PolicyOutcome.PROCEED:
            return None
        if not decision.rewrite:
            # A PROCEED with nothing to say would blank the observation and
            # leave the agent to re-select `ask_user` on the next step, which
            # is a loop rather than a policy.
            return None
        return decision

    @staticmethod
    def _ordered_thoughts(trajectory) -> list[str]:
        """The trajectory's thoughts, oldest first, by step index.

        Read by index rather than by insertion order because `truncate_trajectory`
        pops from the front and a resumed turn appends out of band; the step
        number in the key is the only thing that survives both.
        """
        indexed: list[tuple[int, str]] = []
        for key, value in trajectory.items():
            if not str(key).startswith("thought_"):
                continue
            try:
                indexed.append((int(str(key)[len("thought_"):]), _as_text(value)))
            except ValueError:
                continue
        return [text for _, text in sorted(indexed)]

    def _cited_handle_ids(self, trajectory) -> tuple[str, ...]:
        """Handles the agent named on its closing step(s), first mention first.

        Prefers the ids recorded by `_note_thought_citations` before the
        closing thought was bounded; falls back to re-reading the trajectory,
        which is the bounded text, when no such record exists.
        """
        recorded = getattr(self, "_step_citations", None)
        if recorded:
            closing = recorded[-CLOSING_THOUGHT_STEPS:]
            found: list[str] = []
            for ids in closing:
                found.extend(handle_id for handle_id in ids if handle_id not in found)
            return tuple(found)
        thoughts = self._ordered_thoughts(trajectory)
        closing_text = thoughts[-CLOSING_THOUGHT_STEPS:] if thoughts else []
        return result_handles.parse_handle_ids("\n".join(closing_text))

    def _resolve_presented_results(self, trajectory):
        """The `presented_results` field for this turn's extraction call.

        Two selectors, both checkable by the runtime and both arm-invariant:

        * what the agent CITED on its closing step — ido-mn1.6.4's signature
          instructs it to name the handles holding its results there, and this is
          what makes that instruction load-bearing rather than decorative;
        * what the PRODUCING COMMAND declared deliverable
          (`ResultHandleSpec.presentation`), optionally overridden — narrowed or
          extended — by the executing skill's `presents:` list. The default lives
          on the command precisely so arms A, B and C run the same rule: `off`
          never opens `_skills/`, so a skill-only rule would have left the flat
          control arm with strictly less mechanism than the treatment arms.

        Candidates for the second come from this trajectory's own observations,
        so a leaf resolves the handles IT produced and not a sibling leaf's — the
        store is session-scoped and shared across every leaf of a plan, and
        scoping by the trajectory is what keeps arms B and C from leaking one
        leaf's rows into another's answer.

        Never raises. A turn whose tool calls all succeeded must not fail because
        its evidence could not be re-read; the answer degrades to what the
        trajectory carries, which is exactly the pre-ido-mn1.6.6 behaviour.
        """
        try:
            cited = self._cited_handle_ids(trajectory)
            in_trajectory = result_handles.parse_handle_ids(
                "\n".join(_as_text(value) for value in trajectory.values())
            )
            presented = result_handles.presentation_handles_for(
                in_trajectory, self.presentation_commands
            )
            return result_handles.resolve_for_presentation(
                cited=cited, presented=presented
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("presented-result resolution failed: %s", _fmt_exc(exc))
            return result_handles.PresentedResults(
                max_bytes=result_handles.presented_max_bytes()
            )

    @staticmethod
    def _closing_thought_bytes(trajectory) -> int:
        """Bytes of the thoughts the extraction call is composing from the end of.

        The same `CLOSING_THOUGHT_STEPS` window `_resolve_presented_results`
        cites from, and for the same reason: the closing thought is where an
        agent drafts what it is about to report, so a long one is a genuine
        predictor of a long answer that `presented_results` alone does not see.
        It is bounded (`MAX_NEXT_THOUGHT_CHARS`), so this term can add at most a
        few hundred tokens and cannot become the whole derivation on its own.
        """
        thoughts = [
            value
            for key, value in sorted(trajectory.items())
            if key.startswith("thought_") and isinstance(value, str)
        ]
        return sum(
            len(text.encode("utf-8", errors="ignore"))
            for text in thoughts[-CLOSING_THOUGHT_STEPS:]
        )

    def _trajectory_bytes(self, trajectory) -> int:
        """UTF-8 bytes of the trajectory exactly as the extraction call sees it.

        Measured through `_format_trajectory`, the same rendering the call is
        given, so the term counts what the model must read to compose rather
        than an internal representation of it.
        """
        try:
            rendered = self._format_trajectory(trajectory)
        except Exception:  # pragma: no cover - a formatter failure is not a bound failure
            return 0
        if not isinstance(rendered, str):
            rendered = json.dumps(rendered, default=str)
        return len(rendered.encode("utf-8", errors="ignore"))

    def _finish(self, trajectory, input_args, budget):
        """Phase 4: extract against a sealed snapshot, and never re-run the loop.

        The snapshot is a copy taken before the first extract attempt, so a
        retry sees exactly what the first attempt saw — an extract that
        truncated the trajectory on its way to a context-window error must not
        change what the next attempt is asked to summarize.

        A failed extraction returns a typed failure carrying the sealed steps
        (arch §8.4). It does not re-enter the agent loop, because everything the
        loop did already happened and doing it again is the replay FW-REQ-008B
        clause 3 forbids.
        """
        snapshot = dict(trajectory)
        # EXP-027: the runtime knows the loop was cut short and the model does
        # not. Appended to the SEALED snapshot before the first attempt, so
        # every extraction retry sees the same thing — a notice that arrived
        # only on attempt two would make the retries disagree about what
        # happened.
        partial = self._turn_partial(trajectory, budget)
        if partial is not None:
            snapshot[f"observation_{len(snapshot)}"] = self._exhaustion_notice(
                partial)
        # Resolved once, against the sealed snapshot, for the same reason the
        # snapshot itself is sealed: a corrective re-extraction must be asked to
        # compose from exactly what the first attempt was given, and re-resolving
        # between attempts would let two attempts disagree about what the answer
        # is allowed to present.
        presented = self._resolve_presented_results(trajectory)
        extract_args = {**input_args, PRESENTED_RESULTS_FIELD: presented.text}
        thought_bytes = self._closing_thought_bytes(snapshot)
        trajectory_bytes = self._trajectory_bytes(snapshot)
        bound: Optional[ExtractionBound] = None
        truncated = False
        last_error: Optional[BaseException] = None
        corrected = False
        for attempt in range(EXTRACT_PARSE_ATTEMPTS):
            # ido-mn1.6.33 (b) and (c). Derived and tested BEFORE the call and
            # before every parse retry, because attempt three has less of the
            # turn left than attempt one did and a check made once would stop
            # being true after the first. `deadline_insufficient` is not the
            # clamp: a clamped attempt still runs with a shorter bound, while
            # an insufficient one cannot return anything at all and would be
            # a provider call bought to be killed mid-stream.
            bound = extraction_bound(
                presented.field_bytes, thought_bytes, trajectory_bytes
            )
            if bound.deadline_insufficient:
                return self._deadline_insufficient_prediction(
                    trajectory, presented, bound, partial
                )
            budget.consume_model_call()
            try:
                # Final extraction is its own deadline class (arch §13.2): it
                # runs after every tool call is complete, so a provider that
                # hangs here holds a turn whose work is already done.
                #
                # Derived BEFORE the block and used to SIZE it. The class
                # default for `model.extraction` is 300 s, which was chosen
                # against a fixed 4096-token limit and, left in place, would
                # quietly become the binding constraint the moment the limit is
                # derived: 300 s at the measured 35 tok/s with the safety factor
                # is about 6,300 tokens, so every derivation above that would
                # move the cut from `max_tokens` — where the provider stops
                # cleanly and says `length` — to a killed HTTP attempt, which
                # returns nothing at all. That is a strictly worse failure, and
                # it would arrive as a silent one.
                #
                # This is still a bound and still not the caller's to exceed:
                # `operation` nests to the INNER of the two deadlines, so an
                # outer turn deadline continues to win, and `extraction_bound`
                # itself is clamped against whatever operation is already in
                # force here — which is what `timeout_clamped` records.
                #
                # Re-derived per attempt: attempt three has less of the turn
                # left than attempt one did, and a bound computed once would
                # stop being true after the first. The derivation itself now
                # happens above the `try`, so a deadline that cannot fit an
                # attempt is answered before a parse failure could hide it.
                with external_operations.operation(
                    "model.extraction", seconds=bound.timeout_s
                ):
                    # `capture_finish_reasons` is how this call learns it was
                    # cut. `dspy` hands back parsed fields and drops the
                    # provider's `finish_reason`; without it a truncated answer
                    # and a complete one are the same object, which is exactly
                    # how 69 cut answers were stored in v4 as if they were
                    # finished work.
                    with dspy_utils.capture_finish_reasons() as finish_reasons:
                        extract = self._bound_extraction(
                            self._call_with_potential_trajectory_truncation(
                                self.extract,
                                dict(snapshot),
                                config=bound.as_call_config(),
                                **extract_args,
                            )
                        )
                    truncated = any(
                        str(reason) == "length" for reason in finish_reasons
                    )
            except BaseException as err:
                if not self._is_parse_failure(err):
                    raise
                last_error = err
                logger.warning(
                    "Extraction parse failed (attempt %d/%d): %s",
                    attempt + 1, EXTRACT_PARSE_ATTEMPTS, err,
                )
                continue
            if extract is not None:
                # FW-REQ-017 position 4, before successful completion. EXP-025a
                # stage (c) found the agent routing around the ask_user rewrite
                # by DEFERRING in prose instead — finishing with "let me know if
                # you'd like me to..." on read-only work the request had already
                # asked for. The ask was gone; the refusal was not.
                #
                # This re-runs `extract` against the SAME sealed snapshot with a
                # corrective note appended. It re-executes no tool, so it is not
                # the whole-agent replay FW-REQ-008B clause 3 forbids — it is the
                # extraction retry this method already performs, taken for a
                # policy reason instead of a parse failure. Once, and only once:
                # an agent that defers twice is telling us something the table
                # cannot fix by asking again, and an unbounded corrective loop
                # would be a new budget leak in the method that exists to bound
                # this phase.
                decision = self._consult_finish_policy(
                    extract, trajectory, input_args)
                if decision is not None and not corrected:
                    corrected = decision
                    snapshot[f"observation_{len(snapshot)}"] = decision.rewrite
                    continue
                return dspy.Prediction(
                    trajectory=trajectory,
                    exhausted=self._exhausted_last_run,
                    # ido-mn1.6.10. The limit this call was given and where the
                    # number came from, so a short answer can be read as a
                    # floor, a derivation or a ceiling rather than guessed at.
                    extraction_bound=bound.as_evidence() if bound else None,
                    # The provider stopped the DELIVERABLE at its limit. That is
                    # the harness cutting the answer short, so the turn carries
                    # an infrastructure marker and NOT a task failure — and the
                    # answer itself is still returned below, because what was
                    # written before the cut is the only evidence of what the
                    # turn did.
                    **self._truncation_marker(truncated, bound),
                    # Carried out so `fw.agent.execute` can record it: this
                    # decision has no span of its own.
                    finish_policy=corrected or None,
                    # Which handles this answer was allowed to present, and what
                    # the cap did to them. Same road as `finish_policy`: the
                    # resolution happens after the loop, so it has no span of its
                    # own and would otherwise be invisible to a reader asking
                    # why a listing is short (ido-mn1.6.6).
                    presented_results=presented.as_evidence(),
                    # `exhausted` says THAT it stopped; this says what it had
                    # done when it did, which is what a planner needs to decide
                    # whether to allocate more (EXP-027).
                    turn_partial=partial,
                    **extract,
                )
            last_error = last_error or ValueError(
                "extraction returned nothing after trajectory truncation"
            )

        failure = TypedFailure(
            disposition="permanent",
            code=CODE_EXTRACTION_FAILED,
            detail=f"could not extract a final answer in {EXTRACT_PARSE_ATTEMPTS} "
                   f"attempts: {last_error}",
            completed_work=tuple(
                dict(seal, step_index=idx)
                for idx, seal in sorted(self._step_seals.items())
            ),
        )
        return dspy.Prediction(
            trajectory=trajectory,
            exhausted=self._exhausted_last_run,
            failure=failure,
            final_answer=failure.as_observation(),
            presented_results=presented.as_evidence(),
            extraction_bound=bound.as_evidence() if bound else None,
        )

    def _deadline_insufficient_prediction(
        self,
        trajectory,
        presented,
        bound: ExtractionBound,
        partial,
    ):
        """The turn ran out of deadline before the deliverable could be composed.

        ido-mn1.6.33 (b). Everything the loop did already happened and is sealed
        in the trajectory; what is missing is the one call that turns it into an
        answer, and there is no longer time to make it. Three properties, each
        of which the alternatives get wrong:

        * **Infrastructure, never a task failure.** It rides the extraction
          truncation path — same `extraction_truncated` flag, same
          `CODE_EXTRACTION_TRUNCATED` reason, same `transient` disposition — so
          every reader that already treats truncation as infrastructure treats
          this the same way with no new rule. `failure` is deliberately NOT set:
          that field means the turn failed, and a turn the harness ran out of
          clock on did not.
        * **Answer absent, evidence retained.** Unlike a cut answer there is no
          text to keep, so `final_answer` carries the typed failure's own
          observation and says why. The trajectory and `presented_results` go
          out unchanged, because what the turn DID is exactly what is still
          worth reading.
        * **No provider call.** The point is that the attempt is not started.
          The budget's model call is not consumed either, since no model was
          called.
        """
        failure = TypedFailure(
            disposition="transient",
            code=CODE_EXTRACTION_TRUNCATED,
            detail=(
                f"{DEADLINE_INSUFFICIENT_CAUSE}: "
                f"{bound.deadline_remaining_s:.0f}s of the turn deadline "
                f"remained, below the {minimum_viable_extraction_seconds():.0f}s "
                f"one floor-sized extraction attempt takes; no extraction call "
                f"was made and the trajectory is kept as evidence"
                if bound.deadline_remaining_s is not None
                else f"{DEADLINE_INSUFFICIENT_CAUSE}: no extraction call was made"
            ),
            completed_work=tuple(
                dict(seal, step_index=idx)
                for idx, seal in sorted(
                    (getattr(self, "_step_seals", None) or {}).items()
                )
            ),
        )
        logger.warning(
            "Extraction not attempted: %s (remaining=%s, derived=%.0fs)",
            DEADLINE_INSUFFICIENT_CAUSE,
            bound.deadline_remaining_s,
            bound.derived_timeout_s,
        )
        return dspy.Prediction(
            trajectory=trajectory,
            exhausted=self._exhausted_last_run,
            final_answer=failure.as_observation(),
            presented_results=presented.as_evidence(),
            extraction_bound=bound.as_evidence(),
            turn_partial=partial,
            **self._truncation_marker(
                True, bound, cause=DEADLINE_INSUFFICIENT_CAUSE, failure=failure
            ),
        )

    @staticmethod
    def _truncation_marker(
        truncated: bool,
        bound,
        *,
        cause: Optional[str] = None,
        failure: Optional[TypedFailure] = None,
    ) -> dict[str, Any]:
        """The censor fields for an extraction the provider cut at `max_tokens`.

        A field of its own and NOT the existing `censored` flag, which was the
        first thing tried and is wrong in two places at once. `censored` means
        the emergency safety envelope fired, and `plan_execution` acts on it:
        it calls `safety.censor(...)`, blocks the leaf, marks every remaining
        leaf not-reached and ends the plan — so a single leaf whose composition
        ran past its token limit would cancel the rest of a walk that was
        perfectly healthy. `WorkflowExecutionContext._remember_plan_answer` and
        `_checkpoint_plan_progress` then DISCARD a censored leaf's answer, which
        would delete the very evidence this marker exists to preserve.

        So the two stay countable apart. `WorkflowExecutionContext` still ends
        the TURN as `TurnStatus.CENSORED` — an answer the harness cut short is
        not a completed turn, and `censored` is the status the evaluation
        harness already reads as "infrastructure, never task failure" — but it
        gets there by its own route, with its own reason, keeping the answer.

        The `TypedFailure` rides along as `extraction_failure` rather than as
        `failure`: `failure` is the field that means the turn produced no
        answer, and setting it here would make `_finalize` record a turn that
        HAS an answer as failed.
        """
        if not truncated:
            return {}
        marker: dict[str, Any] = {
            "extraction_truncated": True,
            "extraction_truncated_reason": CODE_EXTRACTION_TRUNCATED,
            "extraction_failure": failure or extraction_truncated_failure(
                max_tokens=getattr(bound, "max_tokens", None)
            ),
        }
        if cause is not None:
            # ido-mn1.6.33. WHICH mechanism produced the marker, alongside the
            # typed reason the schema pins. A cell whose answer was cut at
            # `max_tokens` and one whose extraction never ran are both
            # infrastructure and both keep their evidence, but they are not the
            # same finding and the fix for one is not the fix for the other.
            marker["extraction_truncated_cause"] = cause
        return marker

    def _run_loop(
        self,
        trajectory: dict[str, Any],
        idx: int,
        input_args: dict[str, Any],
        budget: LogicalTurnBudget,
        exception_count: int,
    ):
        """
        Run the ReAct tool loop until finish, budget exhaustion, or AskUserSuspend.

        Returns a suspended Prediction, or None when the loop completed normally.
        Sets ``self._exhausted_last_run`` when the loop ends because the logical
        turn's iteration budget ran out without the agent selecting the `finish`
        tool. The budget belongs to the turn and is only ever spent here, never
        created or reset.
        """
        self._exhausted_last_run = False
        self._censored_last_run = False
        self._budget = budget
        # Same reason the `_on_step_complete` read below uses getattr: this
        # method is reachable on an instance built via __new__ (test helpers)
        # that never ran __init__, and a seal store that does not exist would
        # make sealing raise rather than record.
        if getattr(self, "_step_seals", None) is None:
            self._step_seals = {}
        # Host for the fw.agent.step spans, bound by the caller around the whole
        # agent run. None outside an observed turn, where every helper no-ops.
        host = tracing.current_host()
        while True:
            safety = getattr(self, "_safety_envelope", None)
            if safety is not None and not safety.enabled:
                safety = None
            if safety is not None and (
                safety.censored or safety.wall_time_expired()
            ):
                if not safety.censored:
                    safety.censor("wall-time-cutoff")
                self._censored_last_run = True
                break
            # Opened before the reasoning call so a step that fails to pick a
            # tool is still a recorded step rather than a gap in the trace.
            step_span = tracing.start_span(
                host,
                tracing.SPAN_AGENT_STEP,
                attributes={"step_index": idx},
            )
            try:
                # Phase 1 (arch §8.4): bounded parse retry, before any tool has
                # run. `_decide` raises TurnFailedError when it cannot get a
                # parseable decision, which is a bounded typed failure rather
                # than the whole-agent replay this used to become.
                pred = self._decide(trajectory, input_args, budget)
                if pred is None:
                    raise ValueError("Tool returned is None")
                if safety is not None and safety.wall_time_expired():
                    safety.censor("wall-time-cutoff")
                    self._censored_last_run = True
                    tracing.end_span(
                        host,
                        step_span,
                        status=tracing.STATUS_OK,
                        attributes={
                            "step_index": idx,
                            "censored": True,
                            "censored_reason": safety.censored_reason,
                        },
                    )
                    break
            except ValueError as err:
                invalid_tool_obs = (
                    f"Agent failed to select a valid tool: {_fmt_exc(err)}"
                )
                trajectory[f"observation_{idx}"] = invalid_tool_obs
                self.current_trajectory[f"observation_{idx}"] = invalid_tool_obs
                idx += 1
                recovery_thought = (
                    "To execute a command, I should use one of the available tools"
                )
                recovery_obs = (
                    "Use the appropriate tool with proper arguments (correctly formatted)"
                )
                trajectory[f"thought_{idx}"] = recovery_thought
                trajectory[f"observation_{idx}"] = recovery_obs
                self.current_trajectory[f"thought_{idx}"] = recovery_thought
                self.current_trajectory[f"observation_{idx}"] = recovery_obs
                idx += 1
                exception_count += 1
                # Arch §6.4: invalid model/tool selections consume an iteration.
                # They previously cost nothing, so an agent that could not pick
                # a tool could burn the turn's wall clock for free while the
                # counter stood still.
                budget.consume_iteration()
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_ERROR,
                    attributes={
                        "observation": invalid_tool_obs,
                        "recovered": exception_count <= 2,
                    },
                )
                if exception_count > 2:
                    break
                if budget.exhausted:
                    logger.warning("Logical turn budget exhausted")
                    self._exhausted_last_run = True
                    break
                continue
            except BaseException as err:
                # Anything else from the reasoning call — AdapterParseError,
                # provider errors, control signals. The caller's retry loop
                # re-enters this method, and a step span left on the stack
                # would parent the ENTIRE retried attempt under a phantom
                # span that is never emitted. Close it, then propagate.
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_ERROR,
                    attributes={"error_type": type(err).__name__},
                )
                raise

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            step_status = tracing.STATUS_OK
            step_attributes = {
                "step_index": idx,
                "thought": pred.next_thought,
                "tool_name": pred.next_tool_name,
                "tool_args": pred.next_tool_args,
            }
            if safety is not None:
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "tool_name": pred.next_tool_name,
                            "tool_args": pred.next_tool_args,
                        },
                        sort_keys=True,
                        default=repr,
                    ).encode("utf-8")
                ).hexdigest()
                if safety.observe_fingerprint(fingerprint):
                    self._censored_last_run = True
                    tracing.end_span(
                        host,
                        step_span,
                        status=tracing.STATUS_OK,
                        attributes={
                            **step_attributes,
                            "censored": True,
                            "censored_reason": safety.censored_reason,
                        },
                    )
                    break

            # Mirror the full step into current_trajectory (consumed by the planner
            # for replanning and by distillation as the agent trajectory). Keep the
            # legacy action_{idx} entry too for any consumer that still reads it.
            self.current_trajectory[f"thought_{idx}"] = pred.next_thought
            self.current_trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            self.current_trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            self.current_trajectory[f"action_{idx}"] = (
                f"{pred.next_tool_name}: {pred.next_tool_args}"
            )

            # Phase 2: the durable identity of this call, derived before it
            # runs so the record exists whether or not it returns.
            logical_call_key = self.logical_call_key(
                tracing.get_turn_key(host) if host is not None else None,
                idx,
                pred.next_tool_name,
                pred.next_tool_args,
            )
            step_attributes["logical_call_key"] = logical_call_key

            # FW-REQ-017 position 3 (after observations), which is where the
            # G2B attribution puts 91% of attributed failures: the agent has
            # what it needs and selects `ask_user` anyway. Evaluated before the
            # tool runs, because a rewrite that fires after the suspension has
            # nothing left to rewrite. Deterministic and model-free, so it costs
            # no iteration and cannot fail the turn.
            policy_decision = self._consult_policy(
                pred, trajectory, idx, input_args)
            if policy_decision is not None:
                step_attributes["policy_outcome"] = policy_decision.outcome.value
                step_attributes["policy_source"] = policy_decision.source_policy
                step_attributes["policy_table_version"] = \
                    policy_decision.table_version
                # Clause 3: a deterministic rewrite, not a rejection. The agent
                # receives the replacement as an ordinary observation and keeps
                # going; nothing is raised at it, because a policy that ends the
                # turn to prevent a premature hand-back has produced the failure
                # it was preventing.
                trajectory[f"observation_{idx}"] = policy_decision.rewrite
                self.current_trajectory[f"observation_{idx}"] = \
                    policy_decision.rewrite
                step_attributes["observation"] = policy_decision.rewrite
                tracing.end_span(host, step_span,
                                 status=tracing.STATUS_OK,
                                 attributes=step_attributes)
                idx += 1
                budget.consume_iteration()
                if budget.exhausted:
                    logger.warning("Logical turn budget exhausted")
                    self._exhausted_last_run = True
                    break
                continue

            try:
                # Executed exactly once. Tool retry is the read/side-effect
                # contract's to own (arch §8.4 phase 2) — a retry here cannot
                # know whether the first attempt had an effect.
                observation = self.tools[pred.next_tool_name](**pred.next_tool_args)
                trajectory[f"observation_{idx}"] = observation
                self.current_trajectory[f"observation_{idx}"] = observation
                step_attributes["observation"] = _as_text(observation)
            except ControlSignal:
                # Caught BEFORE the generic `except Exception` below, and
                # deliberately not converted to an observation: a model handed
                # "the write may or may not have landed" as ordinary text
                # reasons about it as data and invents a success (arch §8.4).
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_ERROR,
                    attributes={**step_attributes, "control_signal": True},
                )
                raise
            except AskUserSuspend as err:
                self._suspended = {
                    "trajectory": trajectory,
                    "idx": idx,
                    "input_args": input_args,
                    # Kept for the schema-3 blob shape; the budget carries the
                    # limit that actually bounds the resumed loop.
                    "max_iters": budget.iteration_limit,
                    "clarification": err.clarification_request,
                }
                # The step really did end here — the human wait that follows is
                # fw.ask_user's to record, and this span must not stay open
                # across a suspension that may resume in another process.
                step_attributes["clarification"] = err.clarification_request
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_AWAITING_USER,
                    attributes=step_attributes,
                )
                return dspy.Prediction(
                    suspended=True,
                    clarification=err.clarification_request,
                    exhausted=False,
                )
            except Exception as err:
                error_observation = (
                    f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
                )
                trajectory[f"observation_{idx}"] = error_observation
                self.current_trajectory[f"observation_{idx}"] = error_observation
                step_attributes["observation"] = error_observation
                step_attributes["tool_error"] = type(err).__name__
                step_status = tracing.STATUS_ERROR
            except BaseException as err:
                # Control signals from a tool (e.g. CommandCancelledError) end
                # the run — close the step span so a cancelled turn keeps its
                # last step record instead of leaking an open span.
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.status_for_dispatch_exception(err),
                    attributes={**step_attributes, "error_type": type(err).__name__},
                )
                raise

            # Phase 3: seal the completed step before anything reads it back.
            self._seal_step(
                idx,
                logical_call_key,
                pred.next_thought,
                pred.next_tool_name,
                pred.next_tool_args,
                trajectory.get(f"observation_{idx}"),
            )

            tracing.end_span(
                host, step_span, status=step_status, attributes=step_attributes
            )

            # Step-completion callback for distillation: lets external code inspect
            # each completed step and stop execution early (e.g. on trajectory
            # divergence). Placed AFTER the AskUserSuspend catch so it can never
            # swallow a suspension, and it does not touch _suspended state.
            # getattr guard: resume() may run on an instance built via __new__
            # (test helpers) that never set this attribute.
            on_step_complete = getattr(self, "_on_step_complete", None)
            if on_step_complete and not on_step_complete(idx, trajectory):
                break

            if pred.next_tool_name == "finish":
                break

            idx += 1
            budget.consume_iteration()
            if budget.exhausted:
                # Attributable by construction (FW-REQ-001 clause 4): this
                # budget belongs to this logical turn and holds no prior turn's
                # spend, so exhaustion here means *this* turn spent it.
                logger.warning("Logical turn budget exhausted")
                self._exhausted_last_run = True
                break

        return None

    async def aforward(self, **input_args):
        """Async counterpart of ``forward``; the same budget rule applies (§9.2)."""
        budget = input_args.pop("budget", None)
        if budget is None:
            raise MissingTurnBudgetError(
                "fastWorkflowReAct.aforward() requires budget=LogicalTurnBudget(...); "
                "the caller that begins the logical turn owns it (arch §6.4)"
            )
        self._safety_envelope = input_args.pop("safety_envelope", None)
        if (
            not budget.enforce_iteration_limit
            and (
                self._safety_envelope is None
                or not self._safety_envelope.enabled
            )
        ):
            raise ValueError(
                "an unlimited iteration budget requires an enabled safety envelope"
            )
        self._budget = budget
        input_args.pop("max_iters", None)
        self._step_citations = []
        trajectory = {}
        idx = 0
        while not budget.exhausted:
            safety = self._safety_envelope
            if safety is not None and not safety.enabled:
                safety = None
            if safety is not None and (
                safety.censored or safety.wall_time_expired()
            ):
                if not safety.censored:
                    safety.censor("wall-time-cutoff")
                return self._censored_prediction(trajectory)
            try:
                pred = self._bound_prediction_thought(
                    await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
                )
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break
            if safety is not None and safety.wall_time_expired():
                safety.censor("wall-time-cutoff")
                return self._censored_prediction(trajectory)

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            if safety is not None:
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "tool_name": pred.next_tool_name,
                            "tool_args": pred.next_tool_args,
                        },
                        sort_keys=True,
                        default=repr,
                    ).encode("utf-8")
                ).hexdigest()
                if safety.observe_fingerprint(fingerprint):
                    return self._censored_prediction(trajectory)

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            budget.consume_iteration()
            if pred.next_tool_name == "finish":
                break
            idx += 1

        # Same field, same resolver as `_finish`: the async path has no separate
        # composition step, so an extract signature it fed without
        # `presented_results` would be a signature call missing an input.
        presented = self._resolve_presented_results(trajectory)
        # Same bound, same deadline class, same marker as `_finish`: the async
        # path composes with the same call and would otherwise be the one arm of
        # the runtime where the deliverable is still capped at the react step's
        # constant. A limit that holds on three code paths and not the fourth is
        # not a limit, it is a coincidence.
        bound = extraction_bound(
            presented.field_bytes,
            self._closing_thought_bytes(trajectory),
            self._trajectory_bytes(trajectory),
        )
        # ido-mn1.6.33: and the same refusal, for the same reason the bound
        # itself is shared. A limit that holds on three code paths and not the
        # fourth is not a limit, and neither is a deadline.
        if bound.deadline_insufficient:
            return self._deadline_insufficient_prediction(
                trajectory, presented, bound, None
            )
        with external_operations.operation(
            "model.extraction", seconds=bound.timeout_s
        ):
            with dspy_utils.capture_finish_reasons() as finish_reasons:
                extract = self._bound_extraction(
                    await self._async_call_with_potential_trajectory_truncation(
                        self.extract,
                        trajectory,
                        config=bound.as_call_config(),
                        **input_args,
                        **{PRESENTED_RESULTS_FIELD: presented.text},
                    )
                )
            truncated = any(str(reason) == "length" for reason in finish_reasons)
        return dspy.Prediction(
            trajectory=trajectory,
            presented_results=presented.as_evidence(),
            extraction_bound=bound.as_evidence(),
            **self._truncation_marker(truncated, bound),
            **extract,
        )

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except litellm_exceptions.BadRequestError: 
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


def _as_text(value: Any) -> str:
    """A span-safe rendering of a tool observation.

    Tools return whatever their author chose; span attributes are serialized
    to JSON by the store, so an exotic object would poison the write. The
    trajectory keeps the real value — only the trace gets the text.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return repr(type(value))


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adapter format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
