"""No-agent mode: the framework run without the ReAct agent model (``LLM_AGENT`` not specified).

For a deployment whose client is itself an agent (an MCP or HTTP client). The agent model is the only role
that is absent: routing, parameter extraction, the planner and splitter (``LLM_PLANNER``) and the Jev matcher
are all still used. In this mode:

- a single command and a predefined planning command run as always;
- a planning command runs its code steps; at a DECISION step (a conditional step, or a command step missing one plain
  required value) it pauses like ask_user and resumes on the client's reply (``question_kind`` "phase_step"); at any
  other step code cannot run (a prose step, a value that must be found, a failed step, no path) or where its answer
  would need a model, it ends as a COMPLETED turn that carries a stop (``fw_not_run``), and the wording of the final
  answer is left to the client;
- an ad hoc part (no planning command matched, the means ``from_scratch``) is not run: it comes back as the same
  stop with a suggested plan.

``agent_model_specified`` is the one place that decides the mode. Everything else reads it through
``WorkflowExecutionContext.no_agent_mode``. With ``LLM_AGENT`` set, nothing in this module is consulted.
"""
from __future__ import annotations

from typing import Any, Optional

import fastworkflow

AGENT_MODEL_ENV = "LLM_AGENT"
JEV_KEY_ENV = "JEV_API_KEY"

#: The artifact key under which a turn lists its stopped or refused parts (a list, one entry per part).
NOT_RUN_ARTIFACT = "fw_not_run"

#: The ``kind`` of a stop entry.
KIND_NOT_RUN_HERE = "not_run_here"

#: The kinds of an open step (``open_step["kind"]``).
STEP_CONDITIONAL = "conditional"
STEP_PROSE = "prose"
STEP_MISSING_VALUE = "missing_value"
STEP_FAILED = "failed"
STEP_NO_PATH = "no_path"
STEP_PAGE_CAP = "page_cap"
STEP_NESTED = "nested"
STEP_ANSWER = "answer"
STEP_EXTRA = "extra"

#: The open-step kinds a client can decide (the rest are free work and end the command with the stop).
DECISION_KINDS = (STEP_CONDITIONAL, STEP_MISSING_VALUE)

#: ``question_kind`` of the pending prompt of a decision step, and the artifact key that carries its data.
QUESTION_KIND_PHASE_STEP = "phase_step"
PHASE_STEP_ARTIFACT = "phase_step"
#: The reply that declines to decide (the command then ends with its stop, ``reason`` "declined").
DECLINE_REPLY = "cancel"
#: A decision question is asked once more after a reply that cannot be read; the second one ends with the stop.
MAX_DECISION_ASKS = 2

MODE_BANNER = ("No agent model is set (LLM_AGENT): running in deterministic mode. Ad hoc requests are not planned "
               "here; planning commands run to their first open step.")
MODE_LOG_LINE = ("no-agent mode: LLM_AGENT is not set, so single commands and predefined planning commands run, "
                 "a planning command stops at its first step code cannot run, and an ad hoc request is returned "
                 "as not run here with a suggested plan.")
#: Added to the reply of a request the deterministic path could not route, when no agent model is configured.
NOT_PLANNED_NOTE = ("This request was not planned because no agent model is configured (LLM_AGENT): name a command "
                    "to run it, or configure the agent model.")
STARTUP_PROBLEM = ("no agent model is set (LLM_AGENT) and no Jev key is set (JEV_API_KEY): one of them is needed. "
                   "Set LLM_AGENT to run the ReAct agent as before, or set JEV_API_KEY to run without an "
                   "agent model (Jev then matches each sub-task to a planning command).")


def agent_model_specified() -> bool:
    """Whether the ReAct agent model (``LLM_AGENT``) is specified: set and not empty. The one decision of the mode."""
    return bool(str(fastworkflow.get_env_var(AGENT_MODEL_ENV, default="") or "").strip())


def jev_key_specified() -> bool:
    return bool(str(fastworkflow.get_env_var(JEV_KEY_ENV, default="") or "").strip())


def startup_problem() -> Optional[str]:
    """The startup problem of no-agent mode: neither the agent model nor the Jev key is set. None otherwise."""
    return None if agent_model_specified() or jev_key_specified() else STARTUP_PROBLEM


def runtime_problems() -> list[str]:
    """The problems only a process that serves turns reports (``fastworkflow run``, the server), not training."""
    problem = startup_problem()
    return [problem] if problem else []


def startup_line() -> Optional[str]:
    """The one INFO line naming the mode, or None when the agent model is specified."""
    return None if agent_model_specified() else MODE_LOG_LINE


def run_mode(assistant_requested: bool) -> tuple[bool, Optional[str]]:
    """``(run_as_agent, banner)`` for ``fastworkflow run``: the agent unless ``--assistant`` was asked for or no agent
    model is specified, in which case the run falls back to deterministic (assistant) mode and the banner says so
    plainly (None when nothing fell back)."""
    if assistant_requested:
        return False, None
    if agent_model_specified():
        return True, None
    return False, MODE_BANNER


#: The last line of a continuation: a stop whose command has no code-written answer ends with the answer to write.
ANSWER_LINE = "Write the answer to the request from the facts and the steps that ran."


def numbered(lines: list[str]) -> str:
    """Plan lines as a numbered list, from 1."""
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))


def stop_entry(*, command: Optional[str], phase: Optional[str], open_step: Optional[dict[str, Any]],
               done: list[str], facts: Any, remaining: list[str], suggested_plan: str, means: str,
               sub_task: Optional[int] = None, later_phases: Optional[list[str]] = None,
               nested: Optional[dict[str, Any]] = None, reason: Optional[str] = None) -> dict[str, Any]:
    """One stop, the shape a client reads (``fw_not_run`` is a list of these).

    ``command`` is the planning or plain command (None for an ad hoc part); ``phase`` the phase that stopped (None
    when none had begun); ``open_step`` is ``{"number", "text", "kind", "reason"}`` (None for an ad hoc part);
    ``done`` the plan text of the steps completed; ``facts`` the typed facts of the phases that finished;
    ``remaining`` the plan text of the steps after the open step; ``suggested_plan`` numbered text a client may
    carry out; ``means`` the sub-task's means kind. ``sub_task`` is the part's number in a split turn,
    ``later_phases`` the phases after the stopped one (their steps are planned from facts not yet found), and
    ``nested`` the stop of a nested planning command that this one stopped on, and ``reason`` why a part with
    no open step (an ad hoc part, a command that could not be started) was not run.
    """
    entry: dict[str, Any] = {
        "kind": KIND_NOT_RUN_HERE,
        "command": command,
        "phase": phase,
        "open_step": open_step,
        "done": list(done),
        "facts": facts if facts is not None else {},
        "remaining": list(remaining),
        "suggested_plan": suggested_plan,
        "means": means,
        "sub_task": sub_task,
    }
    if later_phases:
        entry["later_phases"] = list(later_phases)
    if nested:
        entry["nested"] = nested
    if reason:
        entry["reason"] = reason
    return entry


def ran_note(answer: str, command: str) -> str:
    """*answer* of a part that ran bound, with a closing line naming the command that was run (no-agent mode only: the
    client reads the answer text, and the command is otherwise only in the artifacts)."""
    return f"{answer}\n\n(Ran {command}.)" if answer else f"Ran {command}."


def explanation(entry: dict[str, Any]) -> str:
    """The short, plain, code-written answer text of a stop: what ran, where it stopped and why, and that the
    remaining steps are in the data. The wording of the final answer is the client's."""
    command = entry.get("command")
    open_step = entry.get("open_step")
    if open_step is None:
        reason = entry.get("reason") or ("this part needs free planning, which is not run without an agent model "
                                         "(LLM_AGENT)")
        subject = command or "This part"
        suffix = (" A suggested plan is in the data (fw_not_run)." if entry.get("suggested_plan")
                  else " No plan could be suggested.")
        return f"{subject} was not run here: {reason}.{suffix}"
    done = entry.get("done") or []
    ran = f"{len(done)} step{'s' if len(done) != 1 else ''} ran"
    if open_step.get("kind") == STEP_ANSWER:
        where = "its answer needs a model to write it, which is not available here"
    else:
        where = f"it stopped at step {open_step.get('number')} ({open_step.get('reason')})"
    phase = f", in phase {entry['phase']!r}" if entry.get("phase") else ""
    return (f"{command} ran as far as code can run it without an agent model: {ran}{phase}, then {where}. "
            "The remaining steps are listed in the data (fw_not_run); the wording of the final answer is left "
            "to the client.")


# ---- decision steps: the question a paused planning command asks and how its reply is read ------------------------

_YES = frozenset({"yes", "y", "true"})
_NO = frozenset({"no", "n", "false"})
_DECLINE = frozenset({"cancel", "abort", "stop", "decline", "never mind", "nevermind"})


def _bare(reply: str) -> str:
    """*reply* lower-cased, trimmed, and without trailing sentence punctuation."""
    return " ".join(str(reply or "").lower().split()).strip(" .!")


def read_yes_no(reply: str) -> Optional[bool]:
    """True for yes / y / true, False for no / n / false, None for anything else. Strict on purpose: the review
    engine's cues also read ``no changes`` and ``go`` as an accept, which answer a different question."""
    word = _bare(reply)
    return True if word in _YES else False if word in _NO else None


def declines(reply: str) -> bool:
    """Whether the whole reply declines to decide: cancel, abort, stop, decline, never mind."""
    return _bare(reply) in _DECLINE


def decision_question(entry: dict[str, Any], decision: dict[str, Any], *, retry: str = "") -> str:
    """The plain, self-sufficient question a decision step asks: what ran, what is asked, how to answer, how to decline.

    *entry* is the stop's parts (``open_step``, ``done``, ``phase``), *decision* the decision record (``kind``,
    ``command``, and per kind ``condition`` / ``inner`` or ``field`` / ``type`` / ``description``). *retry* opens a
    question asked again after a reply that could not be read.
    """
    open_step = entry["open_step"]
    ran = len(entry.get("done") or [])
    later = list(entry.get("later_phases") or [])
    carries = "its remaining steps" + (f" and its later phases ({', '.join(later)})" if later else "")
    where = (f"The planning command {decision['command']} ran {ran} step{'s' if ran != 1 else ''}"
             + (f" of its phase {entry['phase']!r}" if entry.get("phase") else "")
             + f" and reached step {open_step['number']}: {open_step['text']!r}.")
    if decision["kind"] == STEP_CONDITIONAL:
        ask = (f" That step runs only when {decision['condition']}. Does that hold? Reply yes to run "
               f"{decision['inner']!r} now, or no to skip it; either way the command then carries on with "
               f"{carries}.")
    else:
        kind = f" ({decision['type']}" + (f": {decision['description']}" if decision.get("description") else "") + ")"
        ask = (f" Its command needs a value for {decision['field']}{kind} that the step does not give. Reply with "
               f"that value alone; the command then carries on with {carries}.")
    decline = (f" Reply {DECLINE_REPLY} to stop here without deciding. While this question is open, every message sent "
               f"to the agent (invoke_agent), including one that begins with '/', is read as the reply; to run a "
               f"separate command meanwhile, send it to the assistant (invoke_assistant).")
    return f"{retry}{where}{ask}{decline}".strip()


def phase_step_payload(entry: dict[str, Any], decision: dict[str, Any], facts: Any) -> dict[str, Any]:
    """The pending prompt's data (artifact ``phase_step``, ``question_kind`` "phase_step")."""
    open_step = entry["open_step"]
    if decision["kind"] == STEP_CONDITIONAL:
        asked: dict[str, Any] = {"kind": STEP_CONDITIONAL, "condition": decision["condition"],
                                 "run_if_yes": decision["inner"], "replies": ["yes", "no"]}
    else:
        asked = {"kind": STEP_MISSING_VALUE, "field": decision["field"], "type": decision["type"],
                 "description": decision.get("description") or "", "reply": "the value of the field"}
    return {
        "command": decision["command"], "phase": entry.get("phase"),
        "step": {"number": open_step["number"], "text": open_step["text"], "kind": open_step["kind"]},
        "decision": asked, "decline": DECLINE_REPLY, "asked": decision.get("asked", 1),
        "done": list(entry.get("done") or []), "facts": facts if facts is not None else {},
        "remaining": list(entry.get("remaining") or []),
        # The phases after this one, whose steps are planned when they run (same names as ``fw_not_run``'s).
        "later_phases": list(entry.get("later_phases") or []),
    }
