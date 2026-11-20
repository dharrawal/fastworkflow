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
- an ad hoc part (no planning command matched, the means ``from_scratch``) is not planned here. A turn somebody can
  answer asks which planning command is meant (``question_kind`` "plan_choice", ``fastworkflow/intent_clarification.py``)
  and the reply resumes that part bound to the picked plan; a turn nobody can answer, and a reply that picks none,
  ends the part as not run: it comes back as the same stop with a suggested plan.

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
#: The sentence a question put to a person ends with (which command, a parameter, a decision step): a reply is read
#: by that question's reader for its answer and nothing else in it is acted on (bd fix-rpio).
ANSWER_ONLY_NOTE = "Only your answer is read; anything else in the reply is ignored."
#: A decision question is asked once more after a reply that cannot be read; the second one ends with the stop.
MAX_DECISION_ASKS = 2
#: A question about a planning command's missing arguments is asked again once after a reply that gives none of them;
#: this many replies in a row with no progress end the part.
MAX_NO_PROGRESS = 2

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
               nested: Optional[dict[str, Any]] = None, reason: Optional[str] = None,
               available_plans: Optional[list[dict[str, Any]]] = None,
               request: Optional[str] = None, results: Optional[dict[str, str]] = None,
               partial_read: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One stop, the shape a client reads (``fw_not_run`` is a list of these).

    ``command`` is the planning or plain command (None for an ad hoc part); ``phase`` the phase that stopped (None
    when none had begun); ``open_step`` is ``{"number", "text", "kind", "reason"}`` (None for an ad hoc part);
    ``done`` the plan text of the steps completed; ``facts`` the typed facts of the phases that finished;
    ``remaining`` the plan text of the steps after the open step; ``suggested_plan`` numbered text a client may
    carry out; ``means`` the sub-task's means kind. ``sub_task`` is the part's number in a split turn,
    ``later_phases`` the phases after the stopped one (their steps are planned from facts not yet found), and
    ``nested`` the stop of a nested planning command that this one stopped on, and ``reason`` why a part with
    no open step (an ad hoc part, a command that could not be started) was not run. ``available_plans`` is
    only for an ad hoc part: the planning commands available here, ``{name, description, closest}`` each
    (the closest ones to the part first, see ``available_plans``), and ``request`` the part's own request text (what a
    client sends back with ``use_plan`` to run the part bound to one of them). ``results`` maps a ``done`` line to the
    response text its command gave (when it gave one), and ``partial_read`` is ``{"pages", "rows_read", "total",
    "rows"}`` of the paged read a page-cap stop cut short (``total`` None when the command reported none).
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
    if available_plans is not None:
        entry["available_plans"] = list(available_plans)
    if request is not None:
        entry["request"] = request
    if results:
        entry["results"] = dict(results)
    if partial_read:
        entry["partial_read"] = partial_read
    return entry


#: The most plans an ad hoc part's answer text lists (the data, ``available_plans``, lists them all).
AD_HOC_LIST_CAP = 12


def available_plans(workflow: Any, distribution: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """The planning commands available here (those on *workflow*'s lookup path, the set the matcher chooses from), each ``{name, description, closest}``: the closest ones
    to the part first (the matcher's *distribution* over its options, at or above ``run_listing.UNSURE_CANDIDATE_FLOOR``,
    at most ``review_proposals.MAX_PLAN_ALTERNATIVES``, most probable first), then the rest by name.

    The names are ``workflow_agent.planning_names_on_path`` (the active context, its bases, the live ancestors' contexts and
    the root's, nearest first, each by name: exactly what the matcher offers), the descriptions the matcher's
    (``review_proposals.planning_descriptions``). Empty when none is available here.
    """
    from fastworkflow import review_proposals, run_listing, workflow_agent

    if workflow is None:
        return []
    folder = workflow.folderpath
    names = workflow_agent.planning_names_on_path(workflow)
    if not names:
        return []
    descriptions = review_proposals.planning_descriptions(folder)
    ranked = sorted(((name, float(p)) for name, p in (distribution or {}).items() if name in names),
                    key=lambda pair: (-pair[1], pair[0]))
    closest = [name for name, p in ranked if p >= run_listing.UNSURE_CANDIDATE_FLOOR][
        :review_proposals.MAX_PLAN_ALTERNATIVES]
    ordered = closest + sorted(name for name in names if name not in closest)
    return [{"name": name, "description": descriptions.get(name, ""), "closest": name in closest}
            for name in ordered]


def plan_names(workflow: Any) -> list[str]:
    """The names of the planning commands available here: the ones ``use_plan`` may name."""
    return [plan["name"] for plan in available_plans(workflow)]


def _plan_line(plan: dict[str, Any]) -> str:
    return f"- `{plan['name']}`" + (f": {plan['description']}" if plan.get("description") else "")


def ad_hoc_explanation(entry: dict[str, Any], listed_in: Optional[int] = None) -> str:
    """The code-written answer text of an ad hoc part (no planning command or command fits it): that no matching plan was
    found, that free planning is not available here, then the planning commands available here (markdown lists,
    the closest first and labelled, the rest after, at most ``AD_HOC_LIST_CAP`` in all), and the pointer to the suggested
    plan. *listed_in*: the number of an earlier part of the turn that already lists the same plans in full; this part
    then lists only its own closest ones and points there."""
    plans = entry.get("available_plans") or []
    lines = ["No matching plan was found for this part. Free planning is not available on this server, because no "
             "agent model is set (LLM_AGENT)."]
    closest = [plan for plan in plans if plan.get("closest")]
    rest = [plan for plan in plans if not plan.get("closest")]
    if not plans:
        lines.append("No planning commands are available here.")
    else:
        if closest:
            lines.append("The closest planning commands available here:\n" + "\n".join(map(_plan_line, closest)))
        if rest and listed_in is not None:
            lines.append(f"The other planning commands available here are listed under part {listed_in}.")
        elif rest:
            shown = rest[:max(AD_HOC_LIST_CAP - len(closest), 0)]
            head = "Other planning commands available here:" if closest else "Planning commands available here:"
            block = head + "\n" + "\n".join(map(_plan_line, shown))
            if len(rest) > len(shown):
                block += (f"\n\nand {len(rest) - len(shown)} more; `what can i do` lists everything.")
            lines.append(block)
    lines.append("A suggested plan is in the data (fw_not_run)." if entry.get("suggested_plan")
                 else "No plan could be suggested.")
    return "\n\n".join(lines)


def ran_note(answer: str, command: str) -> str:
    """*answer* of a part that ran bound, with a closing line naming the command that was run (no-agent mode only: the
    client reads the answer text, and the command is otherwise only in the artifacts)."""
    return f"{answer}\n\n(Ran {command}.)" if answer else f"Ran {command}."


def explanation(entry: dict[str, Any], listed_in: Optional[int] = None) -> str:
    """The short, plain, code-written answer text of a stop: what ran, where it stopped and why, and that the
    remaining steps are in the data. The wording of the final answer is the client's."""
    command = entry.get("command")
    open_step = entry.get("open_step")
    if open_step is None and command is None and "available_plans" in entry:
        return ad_hoc_explanation(entry, listed_in)
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
    text = (f"{command} ran as far as code can run it without an agent model: {ran}{phase}, then {where}. "
            "The remaining steps are listed in the data (fw_not_run); the wording of the final answer is left "
            "to the client.")
    if open_step.get("kind") in (STEP_ANSWER, STEP_PAGE_CAP):
        text = f"{ran_statement(entry)}\n\n{text}"
    return text


#: Most values of one list a code-written statement of the facts shows (the data, ``fw_not_run``, has them all).
FACT_LIST_CAP = 10


def _fact_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        shown = ", ".join(str(item) for item in value[:FACT_LIST_CAP])
        more = len(value) - FACT_LIST_CAP
        return f"[{shown}{f', ... and {more} more' if more > 0 else ''}] ({len(value)} in all)"
    return str(value)


#: Most characters of one step's own result text a code-written statement shows (the data, ``fw_not_run``, has it all).
RESULT_TEXT_CAP = 300


def _result_text(text: str) -> str:
    """*text* cut to its first lines within ``RESULT_TEXT_CAP`` characters, saying how much was cut."""
    text = str(text).strip()
    if len(text) <= RESULT_TEXT_CAP:
        return text
    kept = ""
    for line in text.splitlines():
        if len(kept) + len(line) + 1 > RESULT_TEXT_CAP:
            break
        kept += line + "\n"
    kept = kept.rstrip() or text[:RESULT_TEXT_CAP]
    return f"{kept} [... {len(text) - len(kept)} more characters not shown]"


def ran_statement(entry: dict[str, Any]) -> str:
    """Markdown stating what a stopped command's code ran and the facts it recorded, written in code from the stop's
    ``done`` lines and ``facts``. A stop at a paged read's page cap says the read is partial, so a count in the facts is
    of the rows read so far."""
    results = entry.get("results") or {}
    lines = ["**What ran**"]
    for line in entry.get("done") or []:
        lines.append(f"- {line}")
        if results.get(line):
            lines += [f"  > {part}" for part in _result_text(results[line]).splitlines()]
    facts = entry.get("facts") or {}
    if facts:
        lines += ["", "**Facts recorded**"]
        for phase_name, fields in facts.items():
            if isinstance(fields, dict) and fields:
                lines += [f"- {phase_name}:"] + [f"  - {key}: {_fact_text(value)}" for key, value in fields.items()]
            else:
                lines.append(f"- {phase_name}: {_fact_text(fields)}")
    read = entry.get("partial_read")
    if read:
        total = read.get("total")
        rows = read.get("rows") or []
        lines += ["", "**Rows read so far**",
                  f"- {read.get('rows_read', len(rows))} rows read"
                  + (f" of {total} the command reported" if total is not None else " (the command reported no total)")]
        if rows:
            lines.append(f"- rows: {_fact_text(rows)}")
    if (entry.get("open_step") or {}).get("kind") == STEP_PAGE_CAP:
        lines += ["", partial_read_sentence("page cap")]
    return "\n".join(lines)


def partial_read_sentence(limit: str) -> str:
    """The one sentence that says a paged read stopped at *limit* (``page cap`` or ``step limit``) with rows still to
    read, so a count in the answer is of the rows read and not of the whole list."""
    return (f"The read stopped at the {limit} with rows still to read: any count above is of a partial read, not of "
            "the whole list.")


def step_limit_note(command: str, read: dict[str, Any]) -> str:
    """The note a run that ran out of steps while paging *command* ends its answer with: the rows read of the total
    the command reported (``read`` is the ``partial_read`` shape) and the partial-read sentence."""
    return (f"{command}: {read.get('rows_read')} rows read of {read.get('total')} the command reported.\n"
            f"{partial_read_sentence('step limit')}")


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
    decline = (f" Reply {DECLINE_REPLY} to stop here without deciding. While this question is open, every message, "
               f"including one that begins with '/', is read as the reply.")
    return f"{retry}{where}{ask}{decline} {ANSWER_ONLY_NOTE}".strip()


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
