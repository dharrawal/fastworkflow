import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from litellm import ContextWindowExceededError
from litellm import exceptions as litellm_exceptions

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

from fastworkflow import external_operations, tracing
from fastworkflow.turn_budget import LegacyTurnBudget, LogicalTurnBudget, budget_from_state
from fastworkflow.typed_failure import (
    CODE_ADAPTER_PARSE,
    CODE_EXTRACTION_FAILED,
    ControlSignal,
    TurnFailedError,
    TypedFailure,
)
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

        # Build the ReAct signature with trajectory input.
        # available_commands is injected into system message by CommandsSystemPreludeAdapter
        # (see fastworkflow/utils/chat_adapter.py) and is NOT included in the trajectory
        # formatting to avoid token bloat across iterations.
        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

        self.inputs = {}
        self.current_trajectory = {}
        # Observation seals (arch §8.4 phase 3): step index -> digest of the
        # sealed step. What they buy is an answer to "which steps completed?"
        # that does not depend on re-reading a trajectory that a later phase may
        # have truncated, and a durable logical-call key per step so a repeated
        # decision at the same step joins the existing call instead of minting a
        # second effect.
        self._step_seals: dict[int, dict[str, Any]] = {}
        self._on_step_complete = on_step_complete
        self._suspended: dict[str, Any] | None = None
        # True when the most recent _run_loop ended because max_iters was
        # reached without the agent selecting the `finish` tool.
        self._exhausted_last_run = False

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
        self._budget = budget
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

        return self._finish(trajectory, input_args, budget)

    def resume(self, observation: str):
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
                    return self._call_with_potential_trajectory_truncation(
                        self.react, trajectory, **input_args
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
        last_error: Optional[BaseException] = None
        corrected = False
        for attempt in range(EXTRACT_PARSE_ATTEMPTS + 1):
            budget.consume_model_call()
            try:
                # Final extraction is its own deadline class (arch §13.2): it
                # runs after every tool call is complete, so a provider that
                # hangs here holds a turn whose work is already done.
                with external_operations.operation("model.extraction"):
                    extract = self._call_with_potential_trajectory_truncation(
                        self.extract, dict(snapshot), **input_args
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
                    # Carried out so `fw.agent.execute` can record it: this
                    # decision has no span of its own.
                    finish_policy=corrected or None,
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
        )

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
        self._budget = budget
        input_args.pop("max_iters", None)
        trajectory = {}
        for idx in range(budget.iterations_remaining):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            budget.consume_iteration()
            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

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
