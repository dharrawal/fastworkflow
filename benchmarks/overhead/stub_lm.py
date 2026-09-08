"""Deterministic stub LM for the fastWorkflow overhead benchmark (fix-49m.2).

No network, no provider, no backend. The stub subclasses ``dspy.utils.DummyLM``
(the idiom the repo's own tests use, e.g. tests/test_dspy_observability.py) and
answers every DSPy call by *reading the prompt*: it parses the output-field
list DSPy's adapters append to the last user message and fabricates a value
for each field. The ReAct step signature (``next_thought`` / ``next_tool_name``
/ ``next_tool_args``) is answered from a per-turn script of workflow commands,
so one user message drives exactly ``len(script)`` ``execute_workflow_query``
tool calls followed by ``finish``.

Prompt formats recognised (both verified against dspy 3.3.0):

* ChatAdapter (and fastWorkflow's ``CommandsSystemPreludeAdapter`` subclass):
  the last user message ends with
  ``Respond with the corresponding output fields, starting with the field
  `[[ ## a ## ]]`, then `[[ ## b ## ]]` (must be formatted as a valid Python
  dict[str, Any]), ... and then ending with the marker for `[[ ## completed ## ]]`.``
* JSONAdapter: ``Respond with a JSON object in the following order of fields:
  `a`, then `b` (must be formatted as a valid Python ...).``

The stub formats its answer with whichever adapter is active in
``dspy.settings`` at call time, so it works under any ``dspy.context(adapter=...)``.

Turn accounting: a turn opens on the first planner (``next_steps``) or ReAct
call after the previous turn finished, and finishes when the stub emits the
``finish`` tool. Per turn it records every call's start/end time and the gaps
between consecutive ReAct calls — the gap after ReAct call *i* is exactly the
time the runtime spent executing the tool call chosen by call *i* (command
dispatch, execution, span/recorder bookkeeping, trajectory formatting), which is
the "per command" latency the benchmark reports.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import dspy
from dspy.utils import DummyLM

DEFAULT_TOOL_NAME = "execute_workflow_query"
FINISH_TOOL_NAME = "finish"

# ChatAdapter tail: `[[ ## name ## ]]` optionally followed by a type hint.
_CHAT_FIELD_RE = re.compile(
    r"`\[\[ ## (\w+) ## \]\]`(?: \(must be formatted as a valid Python (.+?)\))?"
)
_CHAT_TAIL_ANCHOR = "Respond with the corresponding output fields"
# JSONAdapter tail: `name` optionally followed by a type hint.
_JSON_FIELD_RE = re.compile(
    r"`(\w+)`(?: \(must be formatted as a valid Python (.+?)\))?"
)
_JSON_TAIL_ANCHOR = "order of fields:"


@dataclass
class OutputField:
    name: str
    type_hint: Optional[str] = None  # e.g. "dict[str, Any]", "Literal['a', 'b']"


def parse_output_fields(last_message: str) -> list[OutputField]:
    """Extract the ordered output-field list from an adapter-formatted prompt.

    Returns an empty list when the prompt is not recognised; callers then fall
    back to a plain-text answer.
    """
    idx = last_message.rfind(_CHAT_TAIL_ANCHOR)
    if idx >= 0:
        tail = last_message[idx:]
        return [
            OutputField(name, hint or None)
            for name, hint in _CHAT_FIELD_RE.findall(tail)
            if name != "completed"
        ]
    idx = last_message.rfind(_JSON_TAIL_ANCHOR)
    if idx >= 0:
        tail = last_message[idx + len(_JSON_TAIL_ANCHOR):]
        return [OutputField(name, hint or None) for name, hint in _JSON_FIELD_RE.findall(tail)]
    return []


def literal_choices(type_hint: Optional[str]) -> list[Any]:
    """Return the values of a ``Literal[...]`` hint, or [] when not a Literal."""
    if not type_hint or not type_hint.startswith("Literal["):
        return []
    try:
        node = ast.parse(type_hint, mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(node, ast.Subscript):
        return []
    inner = node.slice
    elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
    values = []
    for elt in elts:
        try:
            values.append(ast.literal_eval(elt))
        except (ValueError, SyntaxError):
            return []
    return values


def default_value_for(field_spec: OutputField) -> Any:
    """A parse-safe canned value for a non-scripted output field."""
    hint = (field_spec.type_hint or "").strip()
    choices = literal_choices(hint)
    if choices:
        return choices[0]
    if hint == "bool":
        return True
    if hint == "int":
        return 0
    if hint == "float":
        return 0.0
    if hint.startswith(("list[", "list", "List[")):
        return []
    if hint.startswith(("dict[", "dict", "Dict[")):
        return {}
    return f"stub value for {field_spec.name}"


@dataclass
class StubCall:
    kind: str            # "react_tool" | "react_finish" | "planner" | "extract" | "summary" | "other"
    fields: list[str]
    t_start: float       # perf_counter
    t_end: float
    turn_index: Optional[int]
    command: Optional[str] = None


@dataclass
class StubTurn:
    index: int
    commands: list[str] = field(default_factory=list)
    calls: list[StubCall] = field(default_factory=list)
    finished: bool = False
    # Gap after each ReAct call that selected a tool: runtime time spent
    # executing that tool before the next ReAct call arrived.
    command_gaps_s: list[float] = field(default_factory=list)
    _pending_react_end: Optional[float] = None

    @property
    def model_calls(self) -> int:
        return len(self.calls)

    @property
    def react_calls(self) -> int:
        return sum(1 for c in self.calls if c.kind.startswith("react"))

    @property
    def stub_time_s(self) -> float:
        return sum(c.t_end - c.t_start for c in self.calls)

    def as_dict(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for c in self.calls:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return {
            "index": self.index,
            "commands_issued": len(self.commands),
            "model_calls": self.model_calls,
            "react_calls": self.react_calls,
            "calls_by_kind": kinds,
            "finished": self.finished,
            "stub_time_s": self.stub_time_s,
            "command_gaps_s": list(self.command_gaps_s),
        }


class ScriptedStubLM(DummyLM):
    """A DummyLM that scripts the ReAct loop and fabricates every other answer."""

    def __init__(
        self,
        script: list[str],
        *,
        tool_name: str = DEFAULT_TOOL_NAME,
        final_answer: str = "All scripted commands were executed.",
        plan: str = "1. Execute the scripted commands in order.\n2. Report the results.",
    ):
        super().__init__([])
        self.script = list(script)
        self.tool_name = tool_name
        self.final_answer = final_answer
        self.plan = plan
        self._lock = threading.Lock()
        self.turns: list[StubTurn] = []
        self._current: Optional[StubTurn] = None
        self._remaining: list[str] = []
        self.anomalies: list[str] = []
        self.unparsed_prompts = 0

    # -- script control (main thread, only between turns) -------------------

    def set_script(self, script: list[str]) -> None:
        with self._lock:
            if self._current is not None and not self._current.finished:
                raise RuntimeError("cannot change the script while a turn is in flight")
            self.script = list(script)

    def close_turn(self) -> None:
        """Mark the in-flight turn closed (call after the runtime went idle)."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        self._current = None
        self._remaining = []

    def _open_locked(self) -> StubTurn:
        turn = StubTurn(index=len(self.turns))
        self.turns.append(turn)
        self._current = turn
        self._remaining = list(self.script)
        return turn

    def _turn_for(self, kind: str) -> StubTurn:
        """Return the turn a call belongs to, opening/closing as needed."""
        cur = self._current
        if kind == "planner":
            # A planner call always starts a fresh turn (fastWorkflow plans once
            # per user message, before the ReAct loop).
            if cur is None or cur.finished or cur.react_calls > 0:
                cur = self._open_locked()
            return cur
        if kind.startswith("react"):
            if cur is None or cur.finished:
                cur = self._open_locked()
            return cur
        # extract / summary / other: attach to the current turn if any, else
        # open one so the call is still accounted for.
        if cur is None:
            cur = self._open_locked()
        return cur

    # -- classification ------------------------------------------------------

    @staticmethod
    def classify(fields: list[OutputField]) -> str:
        names = {f.name for f in fields}
        if "next_tool_name" in names:
            return "react"
        if "final_answer" in names:
            return "extract"
        if "next_steps" in names:
            return "planner"
        if "conversation_summary" in names:
            return "summary"
        return "other"

    # -- answer construction --------------------------------------------------

    def _react_values(self, fields: list[OutputField], turn: StubTurn) -> tuple[dict[str, Any], str, Optional[str]]:
        tool_field = next(f for f in fields if f.name == "next_tool_name")
        choices = literal_choices(tool_field.type_hint)
        if choices and self.tool_name not in choices:
            self.anomalies.append(
                f"tool {self.tool_name!r} not offered; offered={choices!r}"
            )
        finish_name = FINISH_TOOL_NAME if not choices or FINISH_TOOL_NAME in choices else choices[-1]

        if self._remaining and (not choices or self.tool_name in choices):
            command = self._remaining.pop(0)
            turn.commands.append(command)
            step = len(turn.commands)
            values = {
                "next_thought": f"Step {step}: execute the next scripted command.",
                "next_tool_name": self.tool_name,
                "next_tool_args": {"command": command},
            }
            return values, "react_tool", command
        turn.finished = True
        values = {
            "next_thought": "All scripted commands have been executed; finishing.",
            "next_tool_name": finish_name,
            "next_tool_args": {},
        }
        return values, "react_finish", None

    def _values_for(self, kind: str, fields: list[OutputField], turn: StubTurn) -> tuple[dict[str, Any], str, Optional[str]]:
        command = None
        if kind == "react":
            react_values, kind, command = self._react_values(fields, turn)
        else:
            react_values = {}
        values: dict[str, Any] = {}
        for f in fields:
            if f.name in react_values:
                values[f.name] = react_values[f.name]
            elif f.name == "reasoning":
                values[f.name] = "Deterministic stub reasoning."
            elif f.name == "final_answer":
                values[f.name] = self.final_answer
            elif f.name == "next_steps":
                values[f.name] = self.plan
            elif f.name == "conversation_summary":
                values[f.name] = "Stub summary: the scripted commands were executed."
            else:
                values[f.name] = default_value_for(f)
        return values, kind, command

    def _format_answer_fields(self, field_names_and_values: dict[str, Any]):
        # Format for the adapter active at call time (CommandsSystemPreludeAdapter,
        # JSONAdapter, ...), not the one captured at construction.
        active = getattr(dspy.settings, "adapter", None)
        if active is not None:
            saved, self.adapter = self.adapter, active
            try:
                return super()._format_answer_fields(field_names_and_values)
            finally:
                self.adapter = saved
        return super()._format_answer_fields(field_names_and_values)

    # -- the LM entry point ----------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):
        t_start = time.perf_counter()
        messages = messages or [{"role": "user", "content": prompt or ""}]
        last = messages[-1].get("content") or ""
        if not isinstance(last, str):
            last = json.dumps(last, default=str)
        fields = parse_output_fields(last)

        with self._lock:
            if fields:
                kind = self.classify(fields)
                turn = self._turn_for(kind)
                values, kind, command = self._values_for(kind, fields, turn)
                content = self._format_answer_fields(values)
            else:
                self.unparsed_prompts += 1
                kind, command = "other", None
                turn = self._turn_for(kind)
                content = "stub response"
            t_end = time.perf_counter()
            if kind.startswith("react") and turn._pending_react_end is not None:
                turn.command_gaps_s.append(t_start - turn._pending_react_end)
            turn._pending_react_end = t_end if kind == "react_tool" else None
            turn.calls.append(
                StubCall(kind=kind, fields=[f.name for f in fields], t_start=t_start,
                         t_end=t_end, turn_index=turn.index, command=command)
            )

        n = kwargs.get("n", 1)
        choices = [
            dspy.utils.dummies.dotdict(
                message=dspy.utils.dummies.dotdict(content=content, tool_calls=None),
                finish_reason="stop",
            )
            for _ in range(n)
        ]
        return dspy.utils.dummies.dotdict(
            choices=choices,
            usage=dspy.utils.dummies.dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="stub/overhead-benchmark",
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    # -- reporting -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "turns": [t.as_dict() for t in self.turns],
            "total_model_calls": sum(t.model_calls for t in self.turns),
            "total_commands_issued": sum(len(t.commands) for t in self.turns),
            "unparsed_prompts": self.unparsed_prompts,
            "anomalies": list(self.anomalies),
        }


# ---------------------------------------------------------------------------
# Network tripwire
# ---------------------------------------------------------------------------

class NetworkCallAttempted(RuntimeError):
    """Raised when something tries to reach a real model provider."""


class NetworkTripwire:
    """Patch the real completion entry points so any escape from the stub fails
    loudly and is counted. ``dspy.LM`` (the litellm-backed client) and litellm's
    completion functions are patched; ``DummyLM`` subclasses are unaffected.
    """

    def __init__(self) -> None:
        self.trips = 0
        self._patched: list[tuple[Any, str, Any]] = []

    def _trip(self, *_args, **_kwargs):
        self.trips += 1
        raise NetworkCallAttempted(
            "overhead benchmark: a real model call was attempted (stub bypassed)"
        )

    def install(self) -> "NetworkTripwire":
        targets: list[tuple[Any, str]] = []
        try:
            import litellm  # noqa: WPS433
            for name in ("completion", "acompletion", "text_completion", "embedding"):
                if hasattr(litellm, name):
                    targets.append((litellm, name))
        except Exception:  # pragma: no cover - litellm always present with dspy
            pass
        try:
            from dspy.clients.lm import LM as RealLM
            for name in ("forward", "aforward"):
                if hasattr(RealLM, name):
                    targets.append((RealLM, name))
        except Exception:  # pragma: no cover
            pass
        for obj, name in targets:
            self._patched.append((obj, name, getattr(obj, name)))
            setattr(obj, name, self._trip)
        return self

    def uninstall(self) -> None:
        for obj, name, original in reversed(self._patched):
            setattr(obj, name, original)
        self._patched.clear()


def install_stub(stub: ScriptedStubLM) -> list[str]:
    """Route every fastWorkflow LM lookup to ``stub``. Returns the patch points hit.

    ``fastworkflow.utils.dspy_utils.get_lm`` is the single factory every
    runtime path calls (verified at 9904df5: workflow_execution_context,
    workflow_agent, utils/signatures, distillation, conversation_labeling);
    ``conversation_labeling`` binds the name at import time so it is patched
    separately when present. ``dspy.configure(lm=...)`` catches any predictor
    run outside a ``dspy.context`` block.
    """
    patched: list[str] = []
    import fastworkflow.utils.dspy_utils as dspy_utils

    def _get_lm(*_args, **_kwargs):
        return stub

    dspy_utils.get_lm = _get_lm
    patched.append("fastworkflow.utils.dspy_utils.get_lm")
    try:
        import fastworkflow.conversation_labeling as conversation_labeling
        if hasattr(conversation_labeling, "get_lm"):
            conversation_labeling.get_lm = _get_lm
            patched.append("fastworkflow.conversation_labeling.get_lm")
    except Exception:
        pass
    dspy.configure(lm=stub)
    patched.append("dspy.configure(lm=stub)")
    return patched
