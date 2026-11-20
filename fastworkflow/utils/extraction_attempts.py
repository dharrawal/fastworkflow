"""Retry policy for LLM parameter extraction: the first attempt runs greedy (temperature 0).

``dspy.BestOfN`` copies the language model with ``temperature=1.0`` and a fresh ``rollout_id`` for *every*
attempt, the first included, so every extraction was a temperature-1.0 sample whatever temperature
``LLM_PARAM_EXTRACTION`` was given. :class:`FirstAttemptGreedy` keeps BestOfN's contract (``N``
attempts, the reward function, the threshold, ``fail_count``, the best prediction wins, the first one at or
above the threshold returns at once) but runs attempt 1 at temperature 0 (``FIRST_ATTEMPT_TEMPERATURE``) with no rollout id override. Only
the retries (attempts 2..N, which happen when the reward is below the threshold or an attempt raised) use
BestOfN's diverse sampling: temperature 1.0 and distinct rollout ids. An attempt whose model call timed out (after
the LM's own retries) ends the loop: sampling again would only wait out the timeout again.
"""
import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from dspy.utils.exceptions import AdapterParseError

from fastworkflow.utils import dspy_utils
from fastworkflow.utils.logging import logger

FIRST_ATTEMPT_TEMPERATURE = 0.0  # hard-coded temperature 0 for attempt 1: under test, not yet a decision
RETRY_TEMPERATURE = 1.0


class FirstAttemptGreedy(dspy.BestOfN):
    """``dspy.BestOfN`` whose first attempt is greedy (see the module docstring)."""

    #: After ``forward``: True when no attempt gave an answer and every attempt failed because the answer could not be
    #: parsed into the fields' types (a value outside a ``Literal`` or enum). The values are then not "missing": the
    #: model produced something and the typed answer was refused. Read by ``fill.fill`` (``Filled.unparsed``).
    unparsed = False

    def forward(self, **kwargs):
        self.unparsed = False
        errors = []
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.N)]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, rid in enumerate(rollout_ids):
            mod = self.module.deepcopy()
            if idx == 0:
                mod.set_lm(lm.copy(temperature=FIRST_ATTEMPT_TEMPERATURE))
            else:
                mod.set_lm(lm.copy(rollout_id=rid, temperature=RETRY_TEMPERATURE))

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, pred, trace

                if reward >= self.threshold:
                    break

            except Exception as e:
                logger.warning(f"Extraction attempt {idx + 1} failed (rollout id {rid}): {e}")
                errors.append(e)
                if dspy_utils.is_timeout_error(e):
                    # The model call timed out on every try it was given (``LLM_NUM_RETRIES_PARAM_EXTRACTION``):
                    # a sampling retry would be three more of them, and the longest wait of one extraction
                    # would be three times the one the timeout settings promise. Nothing was extracted:
                    # the caller's not-given handling applies, as for an attempt that raised on every try.
                    break
                if idx > self.fail_count:
                    self.unparsed = best_pred is None and all(isinstance(x, AdapterParseError) for x in errors)
                    raise e
                self.fail_count -= 1

        self.unparsed = best_pred is None and bool(errors) and all(isinstance(e, AdapterParseError) for e in errors)
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred

