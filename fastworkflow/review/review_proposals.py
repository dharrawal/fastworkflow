"""The framework's review proposal, round 1 (sub-tasks), as a typed Pydantic object.

Each proposal is registered under its spec name, which is the name of the planning command that reviews it
(``review_subtasks``).

``from_text`` reads a proposal from the one-line text a planning command passes to ``review_proposal``:

* sub-tasks: ``goal; goal; ...`` (outcomes, inputs and means are then set by edits or by the candidates builder).

``draft_round1_proposal`` drafts the round-1 proposal of a split result for the turn's gate: each part's means is the
one decision of ``fastworkflow/_workflows/plan_metadata_extraction/resolution.py`` (``part_intent``: exact name, then the
detection chain over execution and planning commands together; bd fix-o1lg). The gate itself is in
workflow_execution_context (``_subtask_gate``).
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import dspy
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer, model_validator

from fastworkflow.utils import subtask_pool
from fastworkflow.observability import tracing
from fastworkflow.core import turn_cache, turn_progress
from fastworkflow.review.review_step_refs import NUMBER_LIST as _NUMBER_LIST, removal_reading
from fastworkflow.review.review_engine import (
    KIND_SUB_TASKS, Add, Change, Drop, EditParseError, EditRefused, Merge, NoEditsRead, Proposal,
    Split, _reader_notes, _SafeValues, note_reader_drop, register_proposal)

logger = logging.getLogger(__name__)

MEANS_PLANNING = "planning"
MEANS_PLAIN = "plain"
MEANS_FROM_SCRATCH = "from_scratch"

class Means(BaseModel):
    """How a sub-task is done: a prebuilt planning command, a plain command, or built from scratch (round 2).

    Text forms (the reply reader and the rendering): ``planning:NAME``, ``plain:NAME`` and ``from_scratch``.
    """

    kind: Literal["planning", "plain", "from_scratch"] = MEANS_FROM_SCRATCH
    name: str = ""

    @model_validator(mode="after")
    def _name_matches_kind(self):
        if self.kind == MEANS_FROM_SCRATCH and self.name:
            raise ValueError("a from_scratch sub-task names no command")
        if self.kind != MEANS_FROM_SCRATCH and not self.name:
            raise ValueError(f"a {self.kind} sub-task names its command")
        return self

    def render(self) -> str:
        return MEANS_FROM_SCRATCH if self.kind == MEANS_FROM_SCRATCH else f"{self.kind}:{self.name}"

    @classmethod
    def from_text(cls, text: str) -> "Means":
        """The text form of a means (``planning:NAME``, ``plain:NAME``, ``from_scratch``); anything else refused."""
        text = text.strip()
        if text == MEANS_FROM_SCRATCH:
            return cls(kind=MEANS_FROM_SCRATCH)
        kind, _, name = text.partition(":")
        if kind in (MEANS_PLANNING, MEANS_PLAIN) and name.strip():
            return cls(kind=kind, name=name.strip())
        raise ValueError("means must be from_scratch, planning:NAME or plain:NAME")

    def __str__(self) -> str:
        return self.render()


_FROM_SUBTASK = re.compile(r"^(?:(?P<name>[A-Za-z_]\w*)\s*=\s*)?from sub-task\s+(?P<number>\d+)$", re.IGNORECASE)
_NAMED_VALUE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>.*)$", re.DOTALL)
_NAMED_COLON = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<value>.*)$", re.DOTALL)


class SubTaskInput(BaseModel):
    """One input of a sub-task: a value the user gave, or ``from sub-task N`` (an earlier sub-task's output).

    Exactly one of ``value`` and ``from_subtask`` is set. ``name`` is the input's name when it has one.
    """

    name: str = ""
    value: Optional[str] = None
    from_subtask: Optional[int] = Field(default=None, ge=1)
    #: The framework drafted it (``draft_inputs``: the shared extractor on the sub-task's text, or a declared
    #: producer that an earlier sub-task runs), not the user. Shown like any input; see ``resolve_inputs``.
    drafted: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.value is None) == (self.from_subtask is None):
            raise ValueError("an input is a value or 'from sub-task N', not both and not neither")
        return self

    def render(self) -> str:
        text = f"from sub-task {self.from_subtask}" if self.from_subtask is not None else str(self.value)
        return f"{self.name}={text}" if self.name else text

    @classmethod
    def from_text(cls, text: str) -> "SubTaskInput":
        """Read one input's text: ``name=value``, ``name=from sub-task N``, ``from sub-task N`` or a bare value."""
        text = text.strip()
        if match := _FROM_SUBTASK.match(text):
            return cls(name=match.group("name") or "", from_subtask=int(match.group("number")))
        if match := _NAMED_VALUE.match(text):
            return cls(name=match.group("name"), value=_strip_outer_quotes(match.group("value")))
        if match := _NAMED_COLON.match(text):
            return cls(name=match.group("name"), value=_strip_outer_quotes(match.group("value")))
        return cls(value=_strip_outer_quotes(text))


#: The most other plans the which-command question names (``resolution.MAX_ALTERNATIVES``; no-agent mode lists with it).
MAX_PLAN_ALTERNATIVES = 3


class SubTask(BaseModel):
    """One sub-task of a request: its goal, its desired outcome, its inputs, and its means.

    ``goal`` is the sub-task's prompt (what the user asked for this piece). ``outcome`` is the desired outcome, as
    text: a predicate about the world ("Ann holds a loan on Dune"), not a command. ``means`` is the bound means: a
    planning command, a plain command, or ``from_scratch``.

    Arch 19.0 invariants 1 and 3: a sub-task is a goal with bound means, never a bare command list, and no goal is
    completed because its bound command returned success. A goal is complete only when the finish check
    (``observation_offloading/finish_check``) confirms its outcome; ``CommandOutput.success`` never marks one done.
    Round 1 keeps ``outcome`` as the user reviewed it and does not verify it.
    """

    goal: str
    outcome: str = ""
    inputs: list[SubTaskInput] = Field(default_factory=list)
    means: Means = Field(default_factory=Means)
    #: What the framework's check said about the means, shown on the line: a refused planning-command argument
    #: (``check_planned_arguments``). Empty when the check passed or no check ran. Not a completion: an advisory note.
    check: str = ""
    #: The required arguments ``check`` says are simply missing (as data: the review line asks for exactly these, never
    #: reads them back from ``check``'s text). Empty when ``check`` is a refusal of something else. Not shown; absent
    #: from the serialized item when empty.
    check_missing: list[str] = Field(default_factory=list, json_schema_extra={"review_hidden": True})
    #: For a name in ``check_missing`` whose extracted value was refused as not given (a pronoun, a word of the workflow's
    #: vocabulary): the value with the reason (``Extraction.screened``), so the question that asks for it says why. Not
    #: shown; absent from the serialized item when empty.
    check_screened: dict[str, str] = Field(default_factory=dict, json_schema_extra={"review_hidden": True})
    #: The arguments ``check`` says are refused (not missing), by name: the review line says how to change each one with a
    #: structured edit, also one the item does not show. Not shown; absent from the serialized item when empty.
    check_refused: list[str] = Field(default_factory=list, json_schema_extra={"review_hidden": True})
    #: The refusal when the draft's values were refused and the refusal names no argument (a validator's
    #: ``Refused(message)`` with no field, a rule of the ``Input``; the same text as ``check`` for a planning means): nothing
    #: is missing to ask for, so the part is not asked about (``resolution.step``). Not shown; absent from the serialized
    #: item when empty.
    refused_without_field: str = Field(default="", json_schema_extra={"review_hidden": True})
    #: The optional fields the duplicate rule left not given in the draft (``ask.left_empty_line``): field -> the note, one
    #: plain line the review shows under the item (not a refusal: the item runs without them). Kept while the item does
    #: not hold the field. Absent from the serialized item when empty.
    check_notes: dict[str, str] = Field(default_factory=dict, json_schema_extra={"review_hidden": True})
    #: The means the draft drafted this sub-task's inputs for (``draft_inputs`` ran on its goal), or "" when it did not
    #: (drafting off, the draft failed, or the user added the sub-task; a split or merged one keeps its first item's).
    #: fix-48np.1: a sub-task whose means is still the drafted one is not drafted again at run time
    #: (``reviewed_arguments``). Not shown.
    drafted_means: str = Field(default="", json_schema_extra={"review_hidden": True})
    #: The sub-tasks this one joins, when the split had more than MAX_SUBTASKS and ``split_merge.reduce_to_cap`` joined some
    #: (``JoinedSubtask``): the review shows the joined item, and a run whose item is still as drafted runs each part as
    #: a job of its own (``expand_joined_parts``). ``parts_fingerprint`` is the item as drafted; an edit of the goal, outcome,
    #: inputs or means makes it differ, and the item then runs as the one job it reads. Not shown.
    parts: list[str] = Field(default_factory=list, json_schema_extra={"review_hidden": True})
    parts_fingerprint: str = Field(default="", json_schema_extra={"review_hidden": True})
    #: The user answered the review with "look it up" for this part (``ProposalReview/look_up``, bd fix-ftdl.4.3): while
    #: it runs, a command that writes (``command_writes.write_commands``) is refused unrun at the one dispatch of every
    #: command (``CommandExecutor._invoke_command_impl``), whatever means the re-pick ended on. Kept by every later edit
    #: of the item. Not shown; absent from the serialized item when False.
    read_only: bool = Field(default=False, json_schema_extra={"review_hidden": True})
    #: The item this one came from before this reply (set by ``apply_edits``; not serialized).
    _before_item: Optional["SubTask"] = PrivateAttr(default=None)
    #: This reply asked for look_up on this item (set by ``apply_edits``; not serialized).
    _look_up: bool = PrivateAttr(default=False)

    @field_validator("inputs", mode="before")
    @classmethod
    def _inputs_from_text(cls, value: Any) -> Any:
        """Edits and the one-line form give inputs as text: each piece is read by ``SubTaskInput.from_text``."""
        if isinstance(value, str):
            value = [value] if value.strip() else []
        return [SubTaskInput.from_text(item) if isinstance(item, str) else item for item in (value or [])]

    @field_validator("means", mode="before")
    @classmethod
    def _means_from_text(cls, value: Any) -> Any:
        return Means.from_text(value) if isinstance(value, str) else value

    @model_serializer(mode="wrap")
    def _omit_empty_check_missing(self, handler):
        data = handler(self)
        if isinstance(data, dict) and not data.get("check_missing"):
            data.pop("check_missing", None)
        if isinstance(data, dict) and not data.get("check_screened"):
            data.pop("check_screened", None)
        if isinstance(data, dict) and not data.get("check_refused"):
            data.pop("check_refused", None)
        if isinstance(data, dict) and not data.get("refused_without_field"):
            data.pop("refused_without_field", None)
        if isinstance(data, dict) and not data.get("check_notes"):
            data.pop("check_notes", None)
        if isinstance(data, dict) and not data.get("read_only"):
            data.pop("read_only", None)
        return data


_META_INPUT_KEYS = frozenset({"means", "goal", "outcome", "desired_outcome", "check"})


def _strip_outer_quotes(value: str) -> str:
    stripped = (value or "").strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1].strip()
    return stripped


def extract_quoted_names(text: str) -> list[str]:
    """Literal entity names the user wrapped in quotes (not contractions like don't)."""
    if not text:
        return []
    found: list[str] = []
    for match in re.finditer(r'"([^"\n]{2,})"', text):
        name = match.group(1).strip()
        if name and name not in found:
            found.append(name)
    for match in re.finditer(r"(?<![A-Za-z0-9])'([^'\n]{2,})'(?![A-Za-z0-9])", text):
        name = match.group(1).strip()
        if name and name not in found:
            found.append(name)
    return found


def _split_input_assignments(stripped: str) -> list[str]:
    """Split key=value fragments without breaking quoted values."""
    if not (_NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped) or _FROM_SUBTASK.match(stripped)):
        return [stripped]
    pieces: list[str] = []
    start = 0
    in_quote: Optional[str] = None
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if in_quote:
            if char == in_quote:
                in_quote = None
            index += 1
            continue
        if char in "\"'":
            in_quote = char
            index += 1
            continue
        if char in ",;":
            rest = stripped[index + 1 :]
            if re.match(r"\s*(?=[A-Za-z_]\w*\s*[=:]|from sub-task\s+\d+)", rest, re.I):
                piece = stripped[start:index].strip()
                if piece:
                    pieces.append(piece)
                start = index + 1
            index += 1
            continue
        match = re.match(r"\s+and\s+(?=[A-Za-z_]\w*\s*[=:])", stripped[index:], re.I)
        if match:
            piece = stripped[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + match.end()
            index = start
            continue
        index += 1
    tail = stripped[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces if pieces else [stripped]


def _input_edit_part_key(part: str) -> Optional[str]:
    stripped = (part or "").strip()
    if match := _NAMED_VALUE.match(stripped):
        return match.group("name").lower()
    if match := _NAMED_COLON.match(stripped):
        return match.group("name").lower()
    return None


def _input_edit_parts(stripped: str) -> list[str]:
    """Split an ``inputs`` edit into separate key=value pieces; commas inside values stay intact."""
    if _NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped) or _FROM_SUBTASK.match(stripped):
        parts = _split_input_assignments(stripped)
        if len(parts) > 1:
            kept: list[str] = []
            for part in parts:
                text = part.strip()
                if not text:
                    continue
                key = _input_edit_part_key(text)
                if key and key in _META_INPUT_KEYS:
                    continue
                kept.append(text)
            return kept if kept else [stripped]
    return [stripped]


def dropped_input_part(existing: list[SubTaskInput], text: str) -> Optional[str]:
    """The part of a packed ``inputs`` edit that applying *text* over *existing* would lose, or None (fix-j6xa).

    ``inputs=member=Z, check=x`` packs two inputs, and ``_input_edit_parts`` drops a part named ``check``, ``goal``,
    ``outcome`` or ``means`` (text a model can put there by mistake), so the edit would lose it without a word. A
    part is lost when applying the pieces one at a time gives another list than applying the text as one edit; the
    caller refuses it. One input per edit (``inputs=check=x``) is read as written.
    """
    stripped = (text or "").strip()
    if not (_NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped) or _FROM_SUBTASK.match(stripped)):
        return None
    parts = [part.strip() for part in _split_input_assignments(stripped) if part.strip()]
    if len(parts) < 2:
        return None
    one_by_one = list(existing)
    for part in parts:
        one_by_one = _merge_one_subtask_input(one_by_one, part)
    as_one = merge_subtask_input_edit(existing, stripped)
    if [(e.name, e.value, e.from_subtask) for e in one_by_one] == [(e.name, e.value, e.from_subtask) for e in as_one]:
        return None
    return next((part for part in parts if _input_edit_part_key(part) in _META_INPUT_KEYS), None)


#: An input's value that clears the input: ``kind=none`` removes ``kind``, and so does an empty value (``kind=``).
_CLEARING_VALUES = frozenset({"none", "null"})


def _clears(value: Optional[str]) -> bool:
    """Whether an input's new *value* removes the input: ``none``, ``null`` or nothing at all (``name=``)."""
    text = (value or "").strip().lower()
    return not text or text in _CLEARING_VALUES


def _is_clearing_change(edit: Change) -> bool:
    """A one-input clearing change (``name=none``, ``name=``): it is applied on its own, never merged with other input
    edits."""
    if edit.field != "inputs":
        return False
    parts = _input_edit_parts((edit.value or "").strip())
    return (len(parts) == 1 and _input_edit_part_key(parts[0]) is not None
            and SubTaskInput.from_text(parts[0]).from_subtask is None
            and _clears(SubTaskInput.from_text(parts[0]).value))


def _raw_input_value(part: str) -> Optional[str]:
    """The text after ``name=`` (or ``name:``) of one input piece, quotes still on; None for a piece with no name."""
    stripped = part.strip()
    if _FROM_SUBTASK.match(stripped):
        return None
    match = _NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped)
    return match.group("value").strip() if match else None


def unreadable_input_value(text: str) -> Optional[str]:
    """Why an ``inputs`` edit cannot be applied as the person wrote it (fix-2vyv), or None when it can.

    Two spellings the reader would silently misread are refused instead, so no edit is lost without a word:

    - a quoted ``none`` or ``null`` (``member="none"``): the quotes cannot make it the literal word, because ``none``
      and ``null`` clear an input wherever an input is rewritten from its text, so it is refused, not read as a clear;
    - a quoted value cut short by the reader: it holds the quote mark it is wrapped in and also ``, name=`` /
      ``, name:`` / `` and name=``, so the reader ends the value at the inner mark and starts another input.
    """
    stripped = (text or "").strip()
    parts = _split_input_assignments(stripped)
    for part in parts:
        raw = _raw_input_value(part)
        if not raw:
            continue
        mark = raw[0]
        if mark in "\"'" and len(raw) >= 2 and raw[-1] == mark and raw[1:-1].strip().lower() in _CLEARING_VALUES:
            return (f"A quoted {raw[1:-1].strip().lower()!r} cannot be set as a value: the words none and null clear an "
                    "input. Use other wording for the value.")
        if len(parts) > 1 and mark in "\"'" and raw[-1] != mark:
            closing = raw.find(mark, 1)
            if closing != -1 and closing < len(raw) - 1:
                return ("A quoted value holds its own quote mark and also ', name=' (or ', name:' or ' and name='), "
                        "so it cannot be read as one value. Quote it with the other kind of quote mark, or reword it.")
    return None


def _merge_one_subtask_input(existing: list[SubTaskInput], text: str) -> list[SubTaskInput]:
    """Update or add one input by name and keep the item's other inputs."""
    new_input = SubTaskInput.from_text(text)
    result = list(existing)
    if new_input.name:
        if new_input.from_subtask is None and _clears(new_input.value):
            # ``name=none`` or ``name=`` clears that one input (``inputs=none`` clears them all): the review shows it
            # removed, as the step runs.
            return [each for each in result if each.name != new_input.name]
        for position, each in enumerate(result):
            if each.name == new_input.name:
                result[position] = new_input
                return result
        result.append(new_input)
        return result
    for position, each in enumerate(result):
        if not each.name:
            result[position] = new_input
            return result
    result.append(new_input)
    return result


def merge_subtask_input_edit(existing: list[SubTaskInput], text: str) -> list[SubTaskInput]:
    """One ``inputs`` change: update or add inputs and keep the item's other inputs (commas stay in values)."""
    stripped = (text or "").strip()
    if stripped.lower() == "none":
        return []
    result = list(existing)
    for part in _input_edit_parts(stripped):
        result = _merge_one_subtask_input(result, part)
    return result


def _inputs_user_kept(before: SubTask | None, after: SubTask) -> list[SubTaskInput]:
    """Inputs *after* carries that differ from *before* (set in the same reply as a goal or means change)."""
    if before is None:
        return list(after.inputs)
    before_named = {each.name: each for each in before.inputs if each.name}
    kept: list[SubTaskInput] = []
    for each in after.inputs:
        if each.name:
            if each.name not in before_named or each != before_named[each.name]:
                kept.append(each)
        else:
            unnamed_before = [inp for inp in before.inputs if not inp.name]
            if not unnamed_before or each != unnamed_before[0]:
                kept.append(each)
    return kept


def _overlay_inputs(
    drafted: SubTask,
    kept: list[SubTaskInput],
    fallback_existing: Optional[list[SubTaskInput]] = None,
) -> SubTask:
    if fallback_existing and kept:
        merged = list(fallback_existing)
        for each in kept:
            merged = merge_subtask_input_edit(merged, each.render())
        return drafted.model_copy(update={"inputs": merged})
    if not kept:
        return drafted
    merged = merge_subtask_input_edit(drafted.inputs, kept[0].render())
    for each in kept[1:]:
        merged = merge_subtask_input_edit(merged, each.render())
    return drafted.model_copy(update={"inputs": merged})


def _input_user_line(inp: SubTaskInput) -> str:
    if inp.from_subtask is not None:
        text = f"from sub-task {inp.from_subtask}"
    else:
        text = inp.value or ""
    return f"{inp.name}: {text}" if inp.name else text


#: The review's badge of a part whose command changes data (" - changes data" on its line, ``badges`` in the payload).
CHANGES_DATA_BADGE = "changes data"
#: The review's badge of an ad hoc part (no command matched it, ``MEANS_FROM_SCRATCH``) when the agent that will plan
#: it can reach a command that changes data: the workflow has one and an agent model is set. Code cannot tell which
#: commands the agent will call (it can move between records), so every such part may change data.
MAY_CHANGE_DATA_BADGE = "may change data"
#: The review's badge of a part the user asked only to look up (``SubTask.read_only``, bd fix-ftdl.4.3): it changes no
#: data, whatever its command, since every write is refused while it runs.
LOOK_UP_ONLY_BADGE = "look up only"

#: Appended to an ad hoc item's line in the rendered review text in no-agent mode (only then).
NOT_RUN_HERE_SUFFIX = " - not run here (no agent model); a suggested plan is returned"


@register_proposal
class SubTaskProposal(Proposal):
    kind = KIND_SUB_TASKS
    spec = "review_subtasks"
    title = "Sub-tasks"
    item_template = "{index}. {goal} (inputs: {inputs}; means: {means}); desired outcome: {outcome}{check}"
    min_items = 1
    items: list[SubTask] = Field(default_factory=list)

    def badges(self, workflow_folderpath: Optional[str] = None) -> list[str]:
        """Each part's badge, the one definition of what the review says a part does to the data: ``LOOK_UP_ONLY_BADGE``
        for a part the user asked only to look up, ``CHANGES_DATA_BADGE`` for any other part that names a command that
        writes (``command_writes.write_commands``), ``MAY_CHANGE_DATA_BADGE`` for an ad hoc part when the workflow has a
        command that writes and an agent model is set (``no_agent_mode.agent_model_specified``: with none, an ad hoc
        part is not run here), else "". The text, the payload and ``changes_data`` read it."""
        from fastworkflow.core.command_writes import write_commands
        from fastworkflow.core.no_agent_mode import agent_model_specified

        writes = write_commands(workflow_folderpath) if workflow_folderpath else frozenset()
        ad_hoc_may_write = bool(writes) and agent_model_specified()
        return [LOOK_UP_ONLY_BADGE if item.read_only
                else (MAY_CHANGE_DATA_BADGE if ad_hoc_may_write else "") if item.means.kind == MEANS_FROM_SCRATCH
                else CHANGES_DATA_BADGE if item.means.name in writes
                else "" for item in self.items]

    def changes_data(self, workflow_folderpath: Optional[str] = None) -> bool:
        """Whether a part changes or may change data (badge ``CHANGES_DATA_BADGE`` or ``MAY_CHANGE_DATA_BADGE``): the
        turn then needs an explicit yes."""
        return bool({CHANGES_DATA_BADGE, MAY_CHANGE_DATA_BADGE} & set(self.badges(workflow_folderpath)))

    def user_rendering(self, workflow_folderpath: Optional[str] = None, *, no_agent: bool = False) -> str:
        lines = [f"{self.title}:"]
        for index, (item, badge) in enumerate(zip(self.items, self.badges(workflow_folderpath)), 1):
            input_bits = [_input_user_line(each) for each in item.inputs]
            suffix = f" ({', '.join(input_bits)})" if input_bits else ""
            write_badge = f" - {badge}" if badge else ""
            if no_agent and item.means.kind == MEANS_FROM_SCRATCH:
                # No agent model: this item will not be run here (its review entry is ``not_run_here``).
                write_badge = NOT_RUN_HERE_SUFFIX
            lines.append(f"{index}. {item.goal}{suffix}{write_badge}")
            if item.check and item.means.kind == MEANS_PLANNING and (note := _check_line(index, item)):
                lines.append(f"   {note}")
            lines.extend(f"   {line}" for line in item.check_notes.values())
        return "\n".join(lines)

    def model_rendering(self) -> str:
        import string

        lines = [f"{self.title}:"]
        for index, item in enumerate(self.items, 1):
            values = self.item_values(item)
            values["index"] = index
            try:
                lines.append(string.Formatter().vformat(self.item_template, (), _SafeValues(values)))
            except (ValueError, AttributeError):
                lines.append(f"{index}. " + ", ".join(f"{k}={v}" for k, v in values.items() if k != "index"))
        return "\n".join(lines)

    def rendering(self) -> str:
        return self.user_rendering()

    def revised(self, host: Any, before: Proposal) -> "SubTaskProposal":
        """With drafting on, re-draft a sub-task when its goal, means, add or look_up changed (fix-48np.1, fix-kk46.3).

        Inputs the user set in the same reply are kept: re-draft, then overlay those inputs.
        """
        workflow = getattr(host, "_app_workflow", None)
        if workflow is None:
            return self
        before_items = list(before.items) if isinstance(before, SubTaskProposal) else []
        if not drafting_on():
            return self.model_copy(update={"items": [
                _rechecked_if_changed(workflow, item, before_items[position] if position < len(before_items) else None)
                for position, item in enumerate(self.items)]})
        planner_lm = getattr(host, "_current_planner_lm", None)
        items: list[SubTask] = []
        for position, item in enumerate(self.items):
            before_item = getattr(item, "_before_item", None)
            if before_item is None and before_items:
                if item in before_items:
                    before_item = before_items[before_items.index(item)]
                else:
                    matched_swapped = next(
                        (b for b in before_items if (b.goal, b.means, b.inputs) == (item.goal, item.means, item.inputs)),
                        None,
                    )
                    if matched_swapped is not None:
                        item = matched_swapped.model_copy()
                        before_item = matched_swapped
                    elif len(self.items) == len(before_items):
                        before_item = before_items[position]
            kept = _inputs_user_kept(before_item, item)
            if getattr(item, "_look_up", False):
                item = redraft_for_look_up(workflow, item, items, planner_lm=planner_lm)
                item = _overlay_inputs(item, kept)
            elif before_item is None and item.means.kind == MEANS_FROM_SCRATCH:
                built = decided_item(workflow, item.goal)
                picked = built.means
                item = item.model_copy(update={"means": picked})
                if item.means.kind != MEANS_FROM_SCRATCH:
                    item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                    item = _overlay_inputs(item, kept)
            elif (before_item is not None and item.goal != before_item.goal
                  and item.means == before_item.means):
                built = decided_item(workflow, item.goal)
                picked = built.means
                if picked.kind != MEANS_FROM_SCRATCH:
                    item = item.model_copy(update={"means": picked})
                item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                overlay_fallback = (
                    before_item.inputs
                    if (kept and item.means == before_item.means)
                    else None
                )
                item = _overlay_inputs(item, kept, fallback_existing=overlay_fallback)
            elif (before_item is not None and item.means.kind != MEANS_FROM_SCRATCH
                  and (item.means != before_item.means or item.drafted_means != item.means.name)):
                item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                item = _overlay_inputs(item, kept)
            items.append(_rechecked_if_changed(workflow, item, before_item))
        return self.model_copy(update={"items": items})

    def item_values(self, item: Any) -> dict[str, Any]:
        values = super().item_values(item)
        values["inputs"] = ", ".join(each.render() for each in item.inputs) or "none"
        values["means"] = item.means.render()
        values["outcome"] = item.outcome or "not stated"
        values["check"] = f" [check: {item.check}]" if item.check else ""
        return values

    @classmethod
    def from_text(cls, text: str) -> "SubTaskProposal":
        """A proposal from its text: ``goal; goal; ...`` (the goals only), or the JSON of a draft (a list of sub-task
        objects, or ``{"items": [...]}``), which is how the gate passes its candidates' draft to the review."""
        stripped = (text or "").strip()
        # A step passes the draft as text quoted by repr (single quotes around JSON that holds double quotes).
        if len(stripped) >= 2 and stripped[0] in "'\"" and stripped[-1] == stripped[0]:
            stripped = stripped[1:-1].strip()
        if stripped[:1] in ("{", "["):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, (dict, list)):
                return cls(items=data["items"] if isinstance(data, dict) else data)
        goals = [part.strip() for part in text.split(";") if part.strip()]
        return cls(items=[SubTask(goal=goal) for goal in goals])


def _check_line(index: int, item: "SubTask") -> str:
    """The review line for a planning means whose arguments the framework refuses (``check``, worked out again whenever
    the item's inputs change: ``rechecked``): what is missing and how to give it (the reply grammar the review reads),
    then a refused value in the check's own words and how to change it (also one the item does not show), else the
    refusal in its own words."""
    lines: list[str] = []
    if item.check_missing:
        given = {each.name for each in item.inputs if each.value or each.from_subtask}
        names = [name for name in item.check_missing if name not in given]
        if names:
            quoted = " and ".join(repr(name) for name in names)
            replies = "; ".join(f"change {index}: {name}=..." for name in names)
            lines.append(f"Cannot run yet: the required value{'s' if len(names) > 1 else ''} {quoted} "
                         f"{'are' if len(names) > 1 else 'is'} missing. "
                         f'Reply "{replies}" to give {"them" if len(names) > 1 else "it"}.')
    if item.check_refused:
        replies = "; ".join(f"change {index}: {name}=..." for name in item.check_refused)
        lines.append(f"Cannot run as shown: {item.check}. "
                     f'Reply "{replies}" to change {"them" if len(item.check_refused) > 1 else "it"}.')
    elif not item.check_missing:
        lines.append(f"Cannot run as shown: {item.check}.")
    return " ".join(lines)


def _input_signature(inputs) -> list[tuple]:
    """What an input list says, for telling whether it changed: each input's name, value and source sub-task."""
    return [(each.name, each.value, each.from_subtask) for each in inputs]


def _rechecked_if_changed(workflow, item: "SubTask", before: Optional["SubTask"]) -> "SubTask":
    """*item* rechecked (``rechecked``) when its inputs or its means differ from *before* (the item before this reply), or
    when its check refuses a value the item does not hold (the draft's refusal of a value it left out: the item as
    shown runs without it, so any reply that revises the review shows what the inputs now earn)."""
    held = {each.name for each in item.inputs if each.value or each.from_subtask is not None}
    if before is not None and item.means == before.means and \
            _input_signature(item.inputs) == _input_signature(before.inputs) and \
            all(name in held for name in item.check_refused):
        return item
    return rechecked(workflow, item)


def rechecked(workflow, item: "SubTask") -> "SubTask":
    """*item* with its check worked out again from its inputs as they stand (no model call), so a refusal the inputs no
    longer earn is gone and one they now earn is shown: the guards on every argument (``planning_commands.argument_gaps``,
    the values an earlier part produces are nobody's to give), and when they pass and no value waits on an earlier part,
    the ``Input``'s validators (``checked_arguments``, the checks the run makes on these values). Only a planning means
    has a check; anything else, or a check that cannot be made, leaves *item* as it is."""
    from fastworkflow.intent import intent_loop
    from fastworkflow.planning import planning_commands
    from fastworkflow.parameters import ask as one_ask

    if item.means.kind != MEANS_PLANNING or workflow is None:
        return item
    try:
        command = planning_commands.planning_command(workflow.folderpath, item.means.name)
        if command is None:
            return item
        earlier = {each.name for each in item.inputs if each.from_subtask is not None and each.name}
        raw = {p.name: next((each.value for each in item.inputs if each.name == p.name and each.value), None)
               for p in command.params}
        check, refused_without_field = "", ""
        with intent_loop.on_owner_record(workflow, command.name):
            # on a tie the gap is the validators' refusal (``one_guard.validators_first``), its message the line
            gaps = planning_commands.argument_gaps(command, raw, item.goal, workflow=workflow).from_earlier(earlier)
            if gaps.invalid:
                name, why = gaps.invalid[0]
                check = gaps.reason or f"{name}={raw.get(name)!r} {why}"
            elif gaps.missing:
                check = "missing required argument " + ", ".join(gaps.missing)
            elif not earlier:
                try:
                    planning_commands.checked_arguments(command, raw, item.goal, workflow)
                except planning_commands.PlanningCommandCallError as refusal:
                    check = str(refusal)
                    if not planning_commands.refusal_names_an_input(command, refusal):
                        refused_without_field = check
    except Exception as exc:  # noqa: BLE001 - the check is advisory: the run makes its own
        logger.warning("could not check the inputs of %s again (%s: %s)", item.means.name, type(exc).__name__, exc)
        return item
    # The duplicate rule leaves an optional field empty when the step runs (``checked_arguments`` applies it to the
    # reviewed values): the review shows that field removed beside its note, never a value that will not run.
    left_empty = one_ask.left_empty_lines(gaps.left_empty)
    inputs = [each for each in item.inputs if each.from_subtask is not None or each.name not in left_empty]
    held = {each.name for each in inputs if each.value or each.from_subtask is not None}
    return item.model_copy(update={
        "inputs": inputs,
        "check": check, "check_missing": list(gaps.missing),
        "check_refused": [name for name, _ in gaps.invalid],
        "check_screened": {name: text for name, text in item.check_screened.items() if name in gaps.missing},
        "check_notes": {**{name: line for name, line in item.check_notes.items() if name not in held},
                        **left_empty},
        "refused_without_field": refused_without_field})


# ----------------------------------------------------------------------
# Stage 3: the round-1 draft's means, from the one decision (resolution.part_intent)
# ----------------------------------------------------------------------


def decided_item(workflow, goal: str, *, exclude=(), part: Optional[int] = None) -> "SubTask":
    """A sub-task for *goal* whose means is what ``resolution.part_intent`` decides (``draft_choice``: a match is
    bound, anything else is ``from_scratch``; an ambiguity is asked about before the review). No inputs are drafted
    here."""
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    name, kind = resolution.draft_choice(resolution.part_intent(workflow, goal, exclude=exclude, part=part))
    if name is None:
        return SubTask(goal=goal, means=Means())
    return SubTask(goal=goal, means=Means(kind=MEANS_PLANNING if kind == resolution.KIND_PLANNING else MEANS_PLAIN,
                                          name=name))


#: The review command's argument values for each proposal (a value is not a command name).
PROPOSAL_KINDS = {1: SubTaskProposal}


# ----------------------------------------------------------------------
# The gate's predicates and candidates (fix-3rsq.plan-review.2.2, .2.3)
# ----------------------------------------------------------------------

#: The ``items`` text the gate passes the review with: the draft itself is handed over through ``draft_scope`` (a step's
#: text is parsed as a command's arguments, which would cut a JSON draft at its first space).
DRAFT_ITEMS = "draft"
_draft_var: contextvars.ContextVar[Optional["SubTaskProposal"]] = contextvars.ContextVar(
    "fw_review_draft", default=None)


@contextlib.contextmanager
def draft_scope(proposal: "SubTaskProposal"):
    """The draft a review opened with ``DRAFT_ITEMS`` puts to the user, for the length of the framework call that opens it."""
    token = _draft_var.set(proposal)
    try:
        yield
    finally:
        _draft_var.reset(token)


_folder_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("fw_review_folder", default=None)


@contextlib.contextmanager
def workflow_scope(folder: Optional[str]):
    """The workflow folder a reply is read for, for the length of the read: the input names its planning commands
    declare (``declared_params``) are what the reply guards know about it. None keeps the enclosing scope; with no scope
    at all, nothing is declared."""
    if folder is None:
        yield
        return
    token = _folder_var.set(folder)
    try:
        yield
    finally:
        _folder_var.reset(token)


def declared_params(plan: Optional[str] = None) -> tuple:
    """The ``Param`` records the scope's workflow declares: those of planning command *plan*, or of every planning
    command when None. Empty outside a ``workflow_scope`` or for a workflow without planning commands."""
    folder = _folder_var.get()
    if not folder:
        return ()
    from fastworkflow import workflow_agent

    commands = workflow_agent.load_planning_command_map(folder)
    chosen = [commands[plan]] if plan in commands else ([] if plan else list(commands.values()))
    return tuple(p for command in chosen for p in command.params)


def declared_input_names() -> frozenset[str]:
    """The input names the workflow's planning commands declare."""
    return frozenset(p.name for p in declared_params())


def entity_input_names() -> frozenset[str]:
    """The declared inputs that hold a value written in the request (``copied_from_command_text`` or
    ``record_named_in_command_text`` on a str): the entity a step is about."""
    return frozenset(p.name for p in declared_params() if p.type == "str" and p.in_request)


def draft_for(items_text: str) -> Optional["SubTaskProposal"]:
    """The gate's draft when a review is asked with ``DRAFT_ITEMS`` inside ``draft_scope``; None otherwise."""
    return _draft_var.get() if items_text == DRAFT_ITEMS else None


def _planning_descriptions(folder: str) -> dict[str, str]:
    """``{planning command name: its one-line description}``: the first line of the description the matcher lists."""
    from fastworkflow.workflow_agent import load_planning_commands

    return {name: (description.strip().splitlines() or [""])[0].strip()
            for name, description, _ in load_planning_commands(folder)}


#: The public name of ``_planning_descriptions`` (no-agent mode lists the plans of a context with it).
planning_descriptions = _planning_descriptions


def _prefetch_intents(workflow, subtasks: list[str]) -> bool:
    """Decide every sub-task's intent together (``resolution.part_intent``), so the one-after-another loop that follows
    finds each in the turn's memo and the remote model's calls wait together. True when it did, and then it also
    announced each sub-task's match stage, in order.

    Only when there is something to wait on and a place to keep it: the remote model can be asked, more than one
    distinct text, pooling on, and the turn's memo bound (the workers write into the caller's memo). A failed prefetch
    is dropped: the loop decides that text again. Never raises.
    """
    from fastworkflow.intent import intent_detector
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    texts = list(dict.fromkeys(subtasks))
    if (len(texts) < 2 or not subtask_pool.is_parallel(len(texts)) or not turn_cache.active()
            or not intent_detector.matcher_available()):
        return False

    numbers = {text: number for number, text in reversed(list(enumerate(subtasks, 1)))}    # a text's first part

    def decide(text: str) -> None:
        try:
            resolution.part_intent(workflow, text, part=numbers[text])
        except Exception as exc:  # noqa: BLE001 - the sequential loop decides again
            logger.info(f"the prefetched intent failed ({type(exc).__name__}: {exc}); it is decided again in order")

    total = len(subtasks)
    for number in range(1, total + 1):
        turn_progress.emit(turn_progress.STAGE_MATCH, number, total)
    subtask_pool.map_in_order([functools.partial(decide, text) for text in texts])
    return True


def draft_round1_proposal(workflow, subtasks: list[str], planner_lm=None, *,
                          pinned_plan: Optional[str] = None) -> SubTaskProposal:
    """The draft round-1 proposal for a split result (fix-3rsq.plan-review.2.3).

    Each sub-task's means is the one decision ``resolution.part_intent`` makes (``decided_item``): a match is bound;
    anything else is ``from_scratch`` here, and an ambiguity is asked about before the review (``resolution.step``,
    bd fix-o1lg S4/S5), which binds the answer. Each sub-task with a command for its means gets its
    inputs drafted from its text (``draft_inputs``, fix-48np: the shared parameter extractor, and for a planning command
    the typed fill fallback on ``planner_lm``; the values the review shows are the ones the run uses, fix-48np.1). A
    planning pick's refusal is shown on its line: the drafted values' refusal, or, with no draft,
    ``check_planned_arguments`` of no arguments.
    """
    total = len(subtasks)
    if pinned_plan:
        # The client chose the plan (``use_plan``) for this one part: nothing is detected and no confidence bar (the
        # choice stands); its inputs are drafted below as for any planning means.
        proposal = SubTaskProposal(items=[
            SubTask(goal=sub_task, means=Means(kind=MEANS_PLANNING, name=pinned_plan)) for sub_task in subtasks])
    else:
        prefetched = _prefetch_intents(workflow, subtasks)
        items = []
        for number, sub_task in enumerate(subtasks, 1):
            if not prefetched:
                turn_progress.emit(turn_progress.STAGE_MATCH, number, total)
            items.append(decided_item(workflow, sub_task, part=number))
        proposal = SubTaskProposal(items=items)

    # A draft reads the sub-tasks before it only through their MEANS (``_producers``), which the proposal already
    # holds, never their drafted inputs: so the drafts are independent and wait on the model together. Each draft is
    # one sub-task's own text, and the items come back in sub-task order.
    def draft(number: int, item: SubTask, sub_task: str) -> SubTask:
        drafted = drafted_item(workflow, item, sub_task, list(proposal.items[:number - 1]), planner_lm=planner_lm,
                               part=number)
        if parts := list(getattr(sub_task, "parts", ()) or ()):
            drafted = drafted.model_copy(update={"parts": parts})
            drafted = drafted.model_copy(update={"parts_fingerprint": _fingerprint(drafted)})
        return drafted

    items = subtask_pool.map_in_order(
        [functools.partial(draft, number, item, sub_task)
         for number, (item, sub_task) in enumerate(zip(proposal.items, subtasks), 1)],
        before=lambda index: turn_progress.emit(turn_progress.STAGE_DRAFT, index + 1, total))
    return SubTaskProposal(items=items)


def _fingerprint(item: SubTask) -> str:
    """The item as the draft made it: what an edit of the goal, outcome, inputs or means changes."""
    return json.dumps([item.goal, item.outcome, item.means.render(), [each.model_dump() for each in item.inputs]],
                      sort_keys=True, default=str)


def expand_joined_parts(workflow, subtasks: list[str], plan: Optional[SubTaskProposal],
                        planner_lm=None) -> tuple[list[str], Optional[SubTaskProposal]]:
    """``(subtasks, plan)`` with each joined item that is still as drafted replaced by its parts, in order.

    One level only: each part is drafted on its own (``draft_round1_proposal``, as a split sub-task is) and is a leaf. An
    item whose goal, outcome, inputs or means the review changed runs as the one job it now reads. ``from sub-task N``
    inputs of the other items follow the renumbering (N names the first part of a joined item N). Nothing changes when
    no item has parts, so a split within the cap is run exactly as before.
    """
    if plan is None or len(plan.items) != len(subtasks) or not any(item.parts for item in plan.items):
        return subtasks, plan
    first_number: dict[int, int] = {}
    expanded: list[SubTask] = []
    goals: list[str] = []
    for number, item in enumerate(plan.items, 1):
        first_number[number] = len(expanded) + 1
        if item.parts and item.parts_fingerprint == _fingerprint(item):
            for leaf in draft_round1_proposal(workflow, list(item.parts), planner_lm=planner_lm).items:
                expanded.append(leaf.model_copy(update={"parts": [], "parts_fingerprint": ""}))
                goals.append(leaf.goal)
        else:
            expanded.append(item.model_copy(update={"parts": [], "parts_fingerprint": ""}))
            goals.append(subtasks[number - 1])
    renumbered = []
    for item in expanded:
        inputs = [each.model_copy(update={"from_subtask": first_number.get(each.from_subtask, each.from_subtask)})
                  if each.from_subtask is not None else each for each in item.inputs]
        renumbered.append(item.model_copy(update={"inputs": inputs}))
    return goals, SubTaskProposal(items=renumbered)


def refused_choice_message(workflow, edited: Proposal) -> Optional[str]:
    """Why *edited* (a proposal after a reply's edits) sets an input to a value outside the choices its command declares,
    or None when none does (fix-j9qk).

    Only an input the edits set is checked: a (name, value) the item already held under the same means is left alone, so
    an earlier draft is never refused here. A choice is a ``Literal`` or ``Enum`` value of the field of the means'
    ``Input`` (scalar fields only; case is not significant). A means whose ``Input`` cannot be read, a from-scratch item,
    an input without a name, a ``from sub-task N`` input and a field with no declared choices are not checked.
    """
    from fastworkflow import ModuleType, RoutingRegistry
    folder = getattr(workflow, "folderpath", None)
    if not folder or not isinstance(edited, SubTaskProposal):
        return None
    for number, item in enumerate(edited.items, 1):
        if item.means.kind == MEANS_FROM_SCRATCH:
            continue
        before = getattr(item, "_before_item", None)
        held = {(each.name, each.value) for each in before.inputs} if before is not None and before.means == item.means else set()
        fresh = [each for each in item.inputs
                 if each.name and each.from_subtask is None and each.value is not None
                 and (each.name, each.value) not in held]
        if not fresh:
            continue
        try:
            params_class = RoutingRegistry.get_definition(folder).get_command_class(
                item.means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
        except Exception:  # noqa: BLE001 - no readable Input: nothing to check against
            continue
        fields = getattr(params_class, "model_fields", None) or {}
        for each in fresh:
            info = fields.get(each.name)
            if info is None:
                continue
            try:
                allowed = _string_choices(info.annotation)
                value = str(each.value).strip()
                if allowed and value.casefold() not in {choice.casefold() for choice in allowed}:
                    return (f"Step {number}: {each.name} must be one of {', '.join(allowed)}; "
                            f"{value!r} is not allowed.")
            except Exception:  # noqa: BLE001 - a check that cannot be made never fails a reply
                continue
    return None


def _string_choices(annotation: Any) -> list[str]:
    """The sorted declared choices of a scalar field, or [] (not checked) when it is a container, declares none, or
    declares any non-string choice (an int or bool ``Literal``, an ``Enum`` with non-string values)."""
    import enum
    from typing import get_args, get_origin

    if _is_container(annotation):
        return []
    raw: list[Any] = []

    def collect(each: Any) -> None:
        if isinstance(each, type) and issubclass(each, enum.Enum):
            raw.extend(member.value for member in each)
        elif get_origin(each) is Literal:
            raw.extend(get_args(each))
        else:
            for argument in get_args(each):
                collect(argument)

    collect(annotation)
    if not raw or not all(isinstance(value, str) for value in raw):
        return []
    return sorted(raw)


def _is_container(annotation: Any) -> bool:
    """Whether *annotation* is a list, tuple or set, alone or inside an ``Optional``/``Union``."""
    from typing import get_args, get_origin

    if get_origin(annotation) in (list, tuple, set, frozenset):
        return True
    return any(_is_container(argument) for argument in get_args(annotation)) and get_origin(annotation) is not Literal


def drafted_item(workflow, item: SubTask, sub_task: str, earlier: list[SubTask], planner_lm=None,
                 part: Optional[int] = None) -> SubTask:
    """*item* with its inputs drafted from *sub_task* for its means (``draft_inputs``), its planning check, and the means
    it was drafted for (``drafted_means``). With no draft (drafting off, from scratch, a failed draft) the inputs are
    empty and a planning pick's check is ``check_planned_arguments`` of no arguments."""
    from fastworkflow.intent import intent_loop
    from fastworkflow import run_listing
    from fastworkflow.planning import planning_commands

    check = ""
    missing: list[str] = []
    refused: list[str] = []
    screened: dict[str, str] = {}
    inputs: list[SubTaskInput] = []
    if item.means.kind == MEANS_FROM_SCRATCH:
        return item.model_copy(update={"check": check, "check_missing": missing, "check_screened": screened,
                                       "check_refused": refused, "check_notes": {},
                                       "inputs": inputs, "refused_without_field": "", "drafted_means": ""})
    # The draft's checks run the command's validators on the record it will run on, as the "/" path's do. Its spans are
    # the part's (``fw.resolve.part``) while the turn is in its resolve stage.
    with intent_loop.on_owner_record(workflow, item.means.name), \
            tracing.part_span(tracing.current_host(), tracing.SPAN_RESOLVE_PART, part):
        drafted = draft_inputs(workflow, item, sub_task, earlier, planner_lm=planner_lm, part=part)
        if drafted is not None:
            inputs = list(drafted.inputs)
        if item.means.kind == MEANS_PLANNING:
            command = planning_commands.planning_command(workflow.folderpath, item.means.name)
            if drafted is not None:
                check = drafted.refusal or ""
                missing = list(drafted.missing)
                refused = list(drafted.refused)
                screened = {name: text for name, text in drafted.screened if name in missing}
            elif command is not None:
                checked = run_listing.check_planned_arguments(command, {}, sub_task)
                check = checked.refusal or ""
                missing = list(checked.missing)
    drafted_item_ = item.model_copy(update={
        "check": check, "check_missing": missing, "check_screened": screened, "check_refused": refused,
        "inputs": inputs, "check_notes": dict(drafted.left_empty) if drafted is not None else {},
        "refused_without_field": drafted.refused_without_field if drafted is not None else "",
        "drafted_means": item.means.name if drafted is not None else ""})
    # A refusal of a value the draft left out of the inputs (the item as shown runs without it) is worked out again from
    # the inputs as they stand, so the review never says "Cannot run as shown" about a value the item does not hold.
    return _rechecked_if_changed(workflow, drafted_item_, drafted_item_)


#: ``FW_DRAFT_ARGUMENTS`` (on by default since fix-48np.1): the draft fills each sub-task's inputs from the shared
#: extractor, and a bound sub-task then runs with the inputs the review shows (``reviewed_values``). ``off`` (or 0,
#: false, no) leaves the inputs empty and the run as before fix-48np.
DRAFT_ARGUMENTS_ENV_VAR = "FW_DRAFT_ARGUMENTS"


def drafting_on() -> bool:
    """Whether ``FW_DRAFT_ARGUMENTS`` is on (the default): the draft fills inputs, and the run takes them as its
    arguments. Only ``off``, ``0``, ``false`` or ``no`` turns it off."""
    import fastworkflow

    value = fastworkflow.get_env_var(DRAFT_ARGUMENTS_ENV_VAR, default="on")
    return str(value if value is not None else "on").strip().lower() not in ("off", "0", "false", "no")


@dataclass(frozen=True)
class DraftedInputs:
    """A sub-task's drafted inputs (each ``drafted``), and why its values do not stand (None when they do)."""

    inputs: tuple[SubTaskInput, ...] = ()
    refusal: Optional[str] = None
    #: The required arguments the refusal says are simply missing (nothing refused), by name.
    missing: tuple[str, ...] = ()
    #: ``(name, "'it' is a pronoun, not a value")`` of the missing ones whose extracted value the screens refused.
    screened: tuple[tuple[str, str], ...] = ()
    #: The refusal when it names no argument (``SubTask.refused_without_field``).
    refused_without_field: str = ""
    #: The arguments the refusal says are refused (not missing), by name (``SubTask.check_refused``).
    refused: tuple[str, ...] = ()
    #: The duplicate rule's notes of the optional fields it left not given (field -> line, ``SubTask.check_notes``).
    left_empty: tuple[tuple[str, str], ...] = ()


def _input_text(value: Any) -> str:
    """An argument as an input's text, in the form ``check_arguments`` reads back (a bool true/false, a list as JSON)."""
    from enum import Enum

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (list, tuple)):
        return json.dumps([str(v) for v in value])
    return str(value)


def _producers(workflow, means: Means, earlier: list[SubTask]) -> dict[str, int]:
    """``{field: N}``: the fields of *means*'s ``Input`` whose declared producer (``from_command_output``, resolved as
    ``available_from``) an earlier sub-task N runs, as its command or as a step of its planning command; the latest N."""
    from fastworkflow import ModuleType, RoutingRegistry
    from fastworkflow.planning import planning_commands

    folder = workflow.folderpath
    params_class = RoutingRegistry.get_definition(folder).get_command_class(
        means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
    if params_class is None:
        return {}
    callees = planning_commands.resolve_field_sources(folder).callees

    def runs(item: SubTask) -> set[str]:
        if item.means.kind == MEANS_FROM_SCRATCH:
            return set()
        names = {item.means.name, *callees.get(item.means.name, ())}
        return names | {name.rpartition("/")[2] for name in names}

    out: dict[str, int] = {}
    for name, info in params_class.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        producers = {str(ref).rpartition(".")[0] for ref in extra.get("available_from") or ()}
        producers |= {p.rpartition("/")[2] for p in producers}
        for number, item in enumerate(earlier, 1):
            if producers & runs(item):
                out[name] = number
    return out


def _given(info: Any, value: Any) -> bool:
    from pydantic_core import PydanticUndefined

    from fastworkflow.parameters.signatures import NOT_FOUND

    if value is None or (isinstance(value, str) and value.strip() in ("", NOT_FOUND, "None")):
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    default = info.default
    return default is PydanticUndefined or default is None or value != default


def draft_inputs(workflow, item: SubTask, sub_task: str, earlier: list[SubTask],
                 planner_lm=None, part: Optional[int] = None) -> Optional[DraftedInputs]:
    """The inputs of *item*'s means drafted from *sub_task* (fix-48np, stage 1), or None when no draft was made.

    A field whose declared producer (``from_command_output``) an earlier sub-task runs is ``from sub-task N``, not extracted. The
    other values are the shared extractor's (``parameter_extraction.extract_arguments``: the extractor execution
    commands use, with its checks, and for a planning command ``check_arguments``, the checks the typed planning call
    runs), each one the extractor gave and the checks kept that is not the field's default. For a planning command
    whose required value is still missing or refused after the extractor's retry, the typed fill fallback runs, as it
    does in ``typed_planning_call`` (fix-48np.1: one planner call, on *planner_lm* or LLM_PLANNER; the same checks
    after it), so the draft and the run follow the same rules. Values that are still refused, or a required value
    that is still missing, give the refusal and no values. None when the switch is off (``FW_DRAFT_ARGUMENTS=off``; on
    by default) or the extraction fails: the draft is advisory and never fails the turn.
    """
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        extract_arguments, traced_extraction)

    if not drafting_on():
        return None
    try:
        from_earlier = _producers(workflow, item.means, earlier)
        # In ONE ``fw.parameters`` span, by the writer the CME's parameter stage uses, with the part's number.
        result = traced_extraction(item.means.name, lambda bag: extract_arguments(workflow, item.means.name, sub_task, bag),
                                   valid=lambda found: found.valid, part=part)
    except Exception as exc:  # noqa: BLE001 - the draft is advisory: no inputs, as before
        # A warning, not a silent pass: a failure here (a provider error, a test's no-model guard) stays visible.
        logger.warning("could not draft the inputs of %s (%s: %s); the draft shows none",
                       item.means.name, type(exc).__name__, exc)
        return None
    inputs = [SubTaskInput(name=name, from_subtask=number, drafted=True) for name, number in from_earlier.items()]
    if result.params is None:
        return DraftedInputs(inputs=tuple(inputs))
    covered = bool(result.missing) and set(result.missing) <= set(from_earlier) and not result.extraction.invalid
    fields = type(result.params).model_fields
    # a marked value a validator rewrote is drafted as the checks accepted it: the later checks read it (the values
    # question's, the run's), and the run's validators rewrite it again
    values = {name: result.unvalidated.get(name, getattr(result.params, name, None)) for name in fields}
    left_empty = dict(result.left_empty)
    if not result.valid and not covered:
        filled = _typed_fallback_values(workflow, item, sub_task, values, result.refusal, planner_lm, left_empty)
        if filled is None:
            kept, refusal, missing = _partial_inputs(workflow, item, sub_task, values, fields, from_earlier,
                                                     result.refusal,
                                                     result.missing if not result.extraction.invalid else ())
            return DraftedInputs(inputs=tuple(inputs + kept), refusal=refusal, missing=missing,
                                 screened=tuple(result.extraction.screened),
                                 left_empty=tuple(result.left_empty.items()),
                                 refused=tuple(name for name, _ in result.extraction.invalid)
                                 if refusal and not missing else (),
                                 refused_without_field=(refusal or "") if not result.extraction.names else "")
        values = filled
    for name, info in fields.items():
        value = values.get(name)
        if name in from_earlier or not _given(info, value):
            continue
        inputs.append(SubTaskInput(name=name, value=_input_text(value), drafted=True))
    return DraftedInputs(inputs=tuple(inputs), left_empty=tuple(left_empty.items()))


def _partial_inputs(workflow, item: SubTask, sub_task: str, values: dict, fields, from_earlier: dict,
                    refusal: Optional[str], missing: Sequence[str] = ()) -> tuple[list[SubTaskInput], Optional[str], tuple]:
    """The inputs of a planning means whose draft is refused, and the refusal: the values the checks stand behind are
    kept (the review shows them, and the run uses them), the missing or refused ones are left out and named in the
    refusal (``planning_commands.argument_gaps``). A refusal that names no argument (a rule of the planning command)
    keeps no value, as before. The third result names the required arguments that are simply missing (*missing*: the
    extractor's, when it found only those), for the review line."""
    from fastworkflow.planning import planning_commands

    if item.means.kind != MEANS_PLANNING:
        return [], refusal, tuple(missing)
    command = planning_commands.planning_command(workflow.folderpath, item.means.name)
    if command is None:
        return [], refusal, tuple(missing)
    try:
        raw = {p.name: planning_commands._given_or_none(values.get(p.name)) for p in command.params}
        gaps = planning_commands.argument_gaps(command, raw, sub_task, workflow=workflow)
    except Exception as exc:  # noqa: BLE001 - the draft is advisory
        logger.warning("could not tell which arguments of %s are still missing (%s: %s)", item.means.name,
                       type(exc).__name__, exc)
        return [], refusal, tuple(missing)
    if not gaps.names:
        return [], refusal, tuple(missing)
    # the values the checks stand behind, as the valid draft keeps them: a default is not a drafted value (``_given``)
    kept = [SubTaskInput(name=name, value=_input_text(value), drafted=True)
            for name, value in gaps.values.items()
            if name in fields and name not in from_earlier and _given(fields[name], value)]
    only_missing = bool(gaps.missing) and not gaps.invalid
    if only_missing:
        refusal = "missing required argument " + ", ".join(gaps.missing)
    return kept, refusal, tuple(gaps.missing) if only_missing else tuple(missing)


def _stopped_at(planning_commands, command, raw: dict, sub_task: str, refusal: Optional[str]):
    """The refusal the typed fill fallback starts from, with the argument the guards stopped at as data (``gap``): the
    guards run again on *raw* (no model call), so the argument is read from their own record, never from the
    refusal's text. A refusal the guards do not give (a validator's) carries no argument."""
    try:
        planning_commands.check_arguments(command, dict(raw), sub_task)
    except planning_commands.PlanningCommandInputError as stop:
        return stop
    except planning_commands.PlanningCommandCallError:
        pass
    return planning_commands.PlanningCommandInputError(refusal or "")


def _typed_fallback_values(workflow, item: SubTask, sub_task: str, values: dict, refusal: Optional[str],
                           planner_lm=None, left_empty: Optional[dict] = None) -> Optional[dict]:
    """The checked arguments of a planning means after the typed fill fallback (``typed_fallback_arguments``, the
    fallback ``typed_planning_call`` runs), or None when the means is not a planning command, the fallback fills
    nothing, or the checks still refuse: the extractor's refusal then stands. Never raises: the draft is advisory.
    *left_empty* (when given) takes the duplicate rule's notes of the fallback's checks."""
    from fastworkflow.planning import planning_commands

    if item.means.kind != MEANS_PLANNING:
        return None
    command = planning_commands.planning_command(workflow.folderpath, item.means.name)
    if command is None or not command.params:
        return None
    from fastworkflow.utils import dspy_utils
    from fastworkflow.workflow_agent import CommandsSystemPreludeAdapter, typed_planner_commands_text

    raw = {p.name: planning_commands._given_or_none(values.get(p.name)) for p in command.params}
    try:
        lm = planner_lm if planner_lm is not None else dspy_utils.get_lm("LLM_PLANNER")
        with dspy_utils.role_context("LLM_PLANNER", lm, adapter=CommandsSystemPreludeAdapter()):
            args, notes = planning_commands.typed_fallback_arguments(
                command, sub_task, typed_planner_commands_text(workflow), raw,
                _stopped_at(planning_commands, command, raw, sub_task, refusal), workflow)
    except planning_commands.PlanningCommandCallError:
        return None                                   # still refused (or the hook raised): the refusal stands
    except Exception as exc:  # noqa: BLE001 - the draft is advisory: the extractor's refusal stands
        logger.warning("the typed fill fallback of %s failed (%s: %s); the draft shows the refusal",
                       item.means.name, type(exc).__name__, exc)
        return None
    logger.info("drafted %s with the typed fill fallback (%s)", item.means.name, "; ".join(notes))
    if left_empty is not None:
        from fastworkflow.parameters import ask as one_ask

        left_empty.clear()
        left_empty.update(one_ask.left_empty_lines(notes))
    return args


def accepted_subtask_proposal(workflow) -> Optional[SubTaskProposal]:
    """The sub-task proposal the user accepted in the turn's review, read from the session's review store, or None.

    The hand-off hook (fix-3rsq.plan-review.2.9): the accepted record is CONFIRMED and was opened in the current turn.
    The gate also keeps it on the context as ``_turn_subtask_plan`` (the draft for a turn that was not reviewed).
    """
    from fastworkflow.review import approval_binding, review_engine

    store = approval_binding.store_for(workflow)
    if store is None:
        return None
    for record in reversed(store.records):
        if (record.command == "review_subtasks" and record.state == review_engine.CONFIRMED
                and record.created_turn == store.turn_seq):
            proposal = review_engine.proposal_of(record)
            return proposal if isinstance(proposal, SubTaskProposal) else None
    return None


def redraft_for_look_up(workflow, item: SubTask, earlier: list[SubTask], *, planner_lm=None) -> SubTask:
    """Goal-preserving re-draft for ``look_up`` (fix-kk46.3): write commands re-match with writes excluded; read-only
    commands keep the means and re-draft inputs from the goal."""
    from fastworkflow.core.command_writes import write_commands

    writes = write_commands(workflow.folderpath)
    if item.means.kind != MEANS_FROM_SCRATCH and item.means.name in writes:
        built = decided_item(workflow, item.goal, exclude=set(writes))
        new_means = built.means
        if new_means.kind == MEANS_FROM_SCRATCH:
            return item.model_copy(update={"means": new_means, "inputs": [], "drafted_means": "", "check": "", "check_missing": [],
                                                         "check_screened": {}, "check_refused": [], "check_notes": {},
                                                         "refused_without_field": ""})
        item = item.model_copy(update={"means": new_means})
        return drafted_item(workflow, item, item.goal, earlier, planner_lm=planner_lm)
    return drafted_item(workflow, item, item.goal, earlier, planner_lm=planner_lm)


# ----------------------------------------------------------------------
# The free-text reader (fix-3rsq.plan-review.2.7): one model call turns the user's words into typed edits
# ----------------------------------------------------------------------


class ProposedEdit(BaseModel):
    """One edit the reader reads from the reply, numbered as the proposal is shown.

    ``change`` sets ``field`` (goal, outcome, inputs or means) to ``value``; ``drop`` removes ``index``; ``add``
    inserts a sub-task with goal ``value`` after ``after`` (0: first); ``merge`` joins ``indexes`` into one with goal
    ``value``; ``split`` cuts ``index`` into one sub-task per ``parts`` goal.
    """

    model_config = ConfigDict(extra="forbid")
    op: Literal["change", "drop", "add", "merge", "split"]
    index: Optional[int] = None
    indexes: list[int] = Field(default_factory=list)
    after: Optional[int] = None
    field: str = ""
    value: str = ""
    parts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_llm_output(self) -> "ProposedEdit":
        if self.index == 0:
            self.index = None
        if (self.index is None and self.op in ("change", "drop", "split")
                and len(self.indexes) == 1 and self.indexes[0] >= 1):
            self.index = self.indexes[0]
        if self.op == "merge" and 0 in self.indexes:
            self.indexes = [each + 1 for each in self.indexes]
        if self.op == "add" and not self.value.strip() and self.parts:
            self.value = self.parts[0].strip()
        return self


class InvalidReadEdit(EditParseError):
    """An edit the reader read that names no field, an input its step does not have, or no goal or parts: a reading that
    cannot be applied as written, which the reader is asked to read again once (``model_edit_reader``)."""


class UnknownMeansEdit(EditRefused):
    """An edit the reader read whose means names no command a step can be bound to (``_known_means``): read again once
    like an ``InvalidReadEdit`` (``model_edit_reader``), and when still wrong the user is told which name is not a
    command and what the commands are, as a step number out of range is (bd fix-ftdl.11)."""


class PendingItemSlot(EditRefused):
    """Edits parsed from free text that need an item index on a multi-item proposal."""

    def __init__(self, message: str, proposed: list[ProposedEdit]):
        super().__init__(message)
        self.proposed = list(proposed)


class ReadSubTaskEdits(dspy.Signature):
    """Read the user's reply on a numbered list of sub-tasks as edits to that list. Give only the edits the reply asks
    for: change a sub-task's means (a planning command written planning:NAME from the list given, or from_scratch when
    no single planning command fits), its inputs (name=value, or none to clear wrong drafted inputs), its desired
    outcome (field='outcome') or its goal (field='goal'); drop one; add one; merge several into one (when two sub-tasks
    are really one record or one job); or split one into parts. When the user says 'yes, but also ...' or 'not just X,
    I want ...' or says a drafted input/kind is wrong or unknown ('use your best judgment'), return the corresponding
    change/add/merge edit rather than an empty list. A reply that asks for no change at all gives an empty list. When
    the user says 'Step N should be NAME as the KIND, not ...' on a sub-task that has inputs name and kind, return
    change edits setting name=NAME and kind=KIND on step N rather than rewriting the goal to a different task. When the
    user asks to split one combined step into separate steps (one per person or item named) or to replace steps with an
    explicit list of steps, use op='split' with 'parts' containing the full goal sentence for each step, or op='change'/
    'add' with the full goal sentence, never short 2-word placeholders. Any text wrapped in single or double quotes in
    the user's reply (e.g. 'Henderson Ramirez and Lewis and Sons' or "Mcclain Miller and Henderson and Sons 1") is a
    single literal entity name: copy the entire quoted name verbatim into the input or goal and never split it on 'and'
    or commas or shorten it."""

    proposal: str = dspy.InputField(desc="the sub-tasks as shown to the user, numbered from 1")
    planning_commands: str = dspy.InputField(desc="the planning commands a means can name, one per line")
    reply: str = dspy.InputField(desc="the user's reply, in their own words")
    edits: list[ProposedEdit] = dspy.OutputField(desc="the edits the reply asks for, in order; empty when it asks for none")


#: The reader once more, when the edits it read cannot be applied (``model_edit_reader``): the same instructions and
#: inputs, and the edits it gave with why they cannot be applied.
RepairSubTaskEdits = ReadSubTaskEdits.append(
    "refused", dspy.InputField(desc="the edits you read from this reply before, and why they cannot be applied; read "
                                    "the reply again and give edits that can be"), type_=str)


def _resolve_proposed_item(proposal: Proposal, index: Optional[int]) -> int:
    count = len(proposal.items)
    if index is None:
        if count == 1:
            return 1
        raise EditRefused(f"Which of the {count} steps do you mean?")
    return index


_MODEL_LINE_SUFFIX = re.compile(
    r"^\s*(?:(?:\d+)\.\s+)?(?P<goal>.*?)\s*"
    r"\(\s*inputs:\s*(?P<inputs>[^;)]*?)\s*;\s*means:\s*(?P<means>[^)]*?)\s*\)"
    r"(?:\s*;\s*desired outcome:\s*(?P<outcome>.*?))?(?:\s*\[check:.*\])?\s*$",
    re.I | re.S,
)


def _strip_leading_step_number(goal: str) -> str:
    return re.sub(r"^\s*\d+\.\s+", "", (goal or "").strip()).strip()


def _clean_goal_value(text: str) -> tuple[str, list[str], Optional[str], Optional[str]]:
    """Strip a copied ``model_rendering()`` line down to the goal phrase only."""
    raw = (text or "").strip()
    match = _MODEL_LINE_SUFFIX.match(raw)
    if match:
        goal = _strip_leading_step_number(match.group("goal"))
        inputs_raw = match.group("inputs") or ""
        inputs_parts = [piece.strip() for piece in inputs_raw.split(",") if piece.strip()]
        means = (match.group("means") or "").strip() or None
        outcome = (match.group("outcome") or "").strip() or None
        return goal, inputs_parts, means, outcome
    return _strip_leading_step_number(raw), [], None, None


_FUNCTION_CALL_EDIT = re.compile(
    r"^(?:planning:)?(?P<cmd>[A-Za-z_/]\w*)\s*\((?P<args>.+)\)\s*$",
    re.DOTALL,
)
_FUNCTION_CALL_ARG = re.compile(
    r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s)]+))",
)


def _parse_function_call_args(args: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in _FUNCTION_CALL_ARG.finditer(args):
        value = match.group(2) or match.group(3) or match.group(4) or ""
        pairs.append((match.group(1), value.strip()))
    return pairs


def _split_function_call_edit(edit: ProposedEdit) -> list[ProposedEdit]:
    field = (edit.field or "").strip().lower()
    if edit.op != "change" or field not in ("goal", "means", "inputs"):
        return [edit]
    value = (edit.value or "").strip()
    match = _FUNCTION_CALL_EDIT.match(value)
    if not match:
        return [edit]
    pairs = _parse_function_call_args(match.group("args"))
    if not pairs:
        return [edit]
    inputs_value = ", ".join(f"{key}={val}" for key, val in pairs)
    return [edit.model_copy(update={"field": "inputs", "value": inputs_value})]


def _clean_proposed_edit(edit: ProposedEdit) -> ProposedEdit:
    field = (edit.field or "").strip().lower()
    if edit.op == "change" and field == "goal" and edit.value.strip():
        return edit.model_copy(update={"value": _clean_goal_value(edit.value)[0]})
    if edit.op == "add" and edit.value.strip() and (means := _clean_goal_value(edit.value)[2]):
        _known_means(means)     # an added step whose means names no command is a failed edit, never an ad hoc step
    if edit.op in ("add", "merge") and edit.value.strip():
        return edit.model_copy(update={"value": _clean_goal_value(edit.value)[0]})
    if edit.op == "split" and edit.parts:
        cleaned = [_clean_goal_value(part)[0] for part in edit.parts if (part or "").strip()]
        if cleaned:
            return edit.model_copy(update={"parts": cleaned})
    return edit


_STEP_FINE_CLAUSE = re.compile(
    r"\bsteps?\s+\d+(?:\s*(?:and|to|through|[-–—])\s*\d+)?\s+(?:look|are|is)\s+(?:fine|good|ok|okay|right)\b",
    re.I,
)
_CLAUSE_SPLIT = re.compile(r"(?:[.;!\n]|\s+[—–-]\s+)")
_READER_DROP_CUE = re.compile(r"\b(?:drop|remove|delete|skip|omit)\b", re.I)


def _reply_clauses(reply_text: str) -> list[str]:
    cleaned = _STEP_FINE_CLAUSE.sub("", reply_text or "")
    parts = [piece.strip() for piece in _CLAUSE_SPLIT.split(cleaned) if piece.strip()]
    return parts if parts else ([cleaned.strip()] if cleaned.strip() else [])


def _extract_step_numbers(segment: str, count: int) -> list[int]:
    from fastworkflow._workflows.plan_metadata_extraction import _item_validation

    return _item_validation.extract_step_numbers(segment, count)


def _clause_edit_overlap(clause: str, edit: ProposedEdit) -> bool:
    words = {word for word in re.findall(r"[a-z0-9]+", clause.lower()) if len(word) > 2}
    blob = f"{edit.field} {edit.value}".lower()
    edit_words = {word for word in re.findall(r"[a-z0-9]+", blob) if len(word) > 2}
    return bool(words & edit_words)


def _explicit_drop_in_reply(reply_text: Optional[str], step: int, count: int) -> bool:
    if not reply_text:
        return False
    for clause in _reply_clauses(reply_text):
        if _READER_DROP_CUE.search(clause) and step in _extract_step_numbers(clause, count):
            return True
    return False


def _expand_edit_to_steps(edit: ProposedEdit, steps: list[int]) -> list[ProposedEdit]:
    if not steps:
        return [edit]
    return [edit.model_copy(update={"index": step, "indexes": []}) for step in steps]


def _infer_missing_indexes_from_reply(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    reply_text: Optional[str],
) -> list[ProposedEdit]:
    if not reply_text:
        return proposed
    count = len(proposal.items)
    if count <= 1:
        return proposed
    if not any(edit.op in ("change", "drop", "split") and edit.index is None for edit in proposed):
        return proposed

    clauses = _reply_clauses(reply_text)
    drop_steps: set[int] = set()
    expanded: list[ProposedEdit] = []

    for edit in proposed:
        if edit.index is not None:
            expanded.append(edit)
            continue
        if edit.op == "drop":
            steps: list[int] = list(removal_reading(reply_text, count).remove)
            for clause in clauses if not steps else ():
                if _READER_DROP_CUE.search(clause):
                    steps.extend(step for step in _extract_step_numbers(clause, count)
                                 if step not in removal_reading(clause, count).keep)
            if not steps and edit.indexes:
                steps = [idx for idx in edit.indexes if 1 <= idx <= count]
            steps = [step for step in dict.fromkeys(steps)]
            if steps:
                drop_steps.update(steps)
                expanded.extend(_expand_edit_to_steps(edit, steps))
            else:
                expanded.append(edit)
            continue
        expanded.append(edit)

    pending_changes = [edit for edit in expanded if edit.op == "change" and edit.index is None]
    resolved_changes: list[ProposedEdit] = []
    for edit in pending_changes:
        matched_steps: list[int] = []
        for clause in clauses:
            if _READER_DROP_CUE.search(clause):
                continue
            clause_steps = _extract_step_numbers(clause, count)
            if clause_steps and _clause_edit_overlap(clause, edit):
                matched_steps = clause_steps
                break
        if not matched_steps:
            for clause in clauses:
                if _READER_DROP_CUE.search(clause):
                    continue
                clause_steps = [
                    step for step in _extract_step_numbers(clause, count) if step not in drop_steps
                ]
                if clause_steps:
                    matched_steps = clause_steps
        matched_steps = [step for step in dict.fromkeys(matched_steps) if step not in drop_steps]
        if matched_steps:
            resolved_changes.extend(_expand_edit_to_steps(edit, matched_steps))
        else:
            resolved_changes.append(edit)

    result = [
        edit for edit in expanded
        if not (edit.op == "change" and edit.index is None)
    ] + resolved_changes
    for edit in expanded:
        if edit.op != "split" or edit.index is not None:
            continue
        nums = [
            step for step in dict.fromkeys(_extract_step_numbers(reply_text, count))
            if step not in drop_steps
        ]
        if len(nums) == 1:
            result.append(edit.model_copy(update={"index": nums[0], "indexes": []}))
        elif len(nums) > 1:
            result.extend(_expand_edit_to_steps(edit, nums))
        else:
            result.append(edit)
    return result


def _dedupe_proposed_edits(proposed: list[ProposedEdit]) -> list[ProposedEdit]:
    seen: set[tuple] = set()
    kept: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "change":
            field = (edit.field or "").strip().lower()
            key = (edit.index, field, edit.value)
            if key in seen:
                continue
            seen.add(key)
        kept.append(edit)
    return kept


def _dedupe_engine_edits(
    edits: list,
    *,
    reply_text: Optional[str] = None,
    proposal: Optional[Proposal] = None,
) -> list:
    from collections import defaultdict

    count = len(proposal.items) if proposal is not None else 0
    drop_by_index: dict[int, Drop] = {}
    split_indexes: set[int] = set()
    merge_indexes: set[int] = set()
    changes: list[Change] = []
    others: list = []

    for edit in edits:
        if isinstance(edit, Drop):
            drop_by_index[edit.index] = edit
        elif isinstance(edit, Split):
            split_indexes.add(edit.index)
            others.append(edit)
        elif isinstance(edit, Merge):
            merge_indexes.update(edit.indexes)
            others.append(edit)
        elif isinstance(edit, Change):
            if _is_clearing_change(edit):
                others.append(edit)
            else:
                changes.append(edit)
        else:
            others.append(edit)

    # A merge of items that the same reply also drops ('make it a single step and drop the second step'): the dropped
    # member is already joined into the first by the merge, so the merge keeps the others, and a merge left with one
    # item is a change of that item's goal. Without this the engine refuses it: 'Item 2 is named by more than one edit'.
    if drop_by_index and merge_indexes & set(drop_by_index):
        rewritten: list = []
        for edit in others:
            if isinstance(edit, Merge) and set(edit.indexes) & set(drop_by_index) and min(edit.indexes) not in drop_by_index:
                rest = [index for index in edit.indexes if index not in drop_by_index]
                if len(rest) >= 2:
                    rewritten.append(edit.model_copy(update={"indexes": rest}))
                elif edit.values.get("goal"):
                    changes.append(Change(index=rest[0], field="goal", value=edit.values["goal"]))
                continue
            rewritten.append(edit)
        others = rewritten
        merge_indexes = {index for edit in others if isinstance(edit, Merge) for index in edit.indexes}

    whole_indexes = split_indexes | merge_indexes
    skip_drop: set[int] = set()
    skip_change: set[int] = set()
    for index in set(drop_by_index) & {change.index for change in changes}:
        if _explicit_drop_in_reply(reply_text, index, count):
            skip_change.add(index)
        else:
            skip_drop.add(index)
    skip_change |= whole_indexes

    surviving_changes: list[Change] = [
        change for change in changes
        if change.index not in skip_change
    ]
    surviving_drops = [
        drop for index, drop in drop_by_index.items()
        if index not in skip_drop
    ]

    by_index_field: dict[tuple[int, str], list[Change]] = defaultdict(list)
    for change in surviving_changes:
        by_index_field[(change.index, change.field)].append(change)

    merged_changes: list[Change] = []
    for (index, field), field_edits in by_index_field.items():
        if field == "inputs":
            merged_inputs: list[SubTaskInput] = []
            for field_edit in field_edits:
                merged_inputs = merge_subtask_input_edit(merged_inputs, field_edit.value)
            joined = ", ".join(item.render() for item in merged_inputs)
            there = list(proposal.items[index - 1].inputs) if proposal is not None and 1 <= index <= count else []
            if dropped_input_part(there, joined) and not any(dropped_input_part(there, each.value) for each in field_edits):
                # Each edit holds one input, but written back as one text a part named like a step field would be
                # dropped: keep the edits as written, one input each (fix-j6xa). A packed edit is read as before.
                merged_changes.extend(field_edits)
            else:
                merged_changes.append(Change(index=index, field="inputs", value=joined))
        else:
            merged_changes.append(field_edits[-1])

    # An input cleared on an item the reply drops, splits or merges away is moot (and the engine would refuse the item as
    # named twice); an input cleared and also set by the same reply is set (the set is what the user wrote).
    gone_items = {drop.index for drop in surviving_drops} | whole_indexes
    set_keys = {
        (change.index, key)
        for change in merged_changes if change.field == "inputs" and not _is_clearing_change(change)
        for key in (_input_edit_part_key(part) for part in _input_edit_parts((change.value or "").strip())) if key}
    others = [
        edit for edit in others
        if not (isinstance(edit, Change) and _is_clearing_change(edit)
                and (edit.index in gone_items
                     or (edit.index, _input_edit_part_key(_input_edit_parts((edit.value or "").strip())[0])) in set_keys))]
    return others + surviving_drops + merged_changes


def _normalize_reader_edits(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    *,
    reply_text: Optional[str] = None,
) -> list[ProposedEdit]:
    count = len(proposal.items)
    normalized: list[ProposedEdit] = []
    for edit in proposed:
        if (
            edit.op == "merge"
            and isinstance(proposal, SubTaskProposal)
            and len(edit.indexes) == 2
            and reply_text
            and re.search(r"\b(?:swap|switch|reorder)\b", reply_text, re.I)
            and not re.search(r"\b(?:merge|combine|join|drop|remove|delete|skip|omit)\b", reply_text, re.I)
        ):
            idx_a, idx_b = edit.indexes[0], edit.indexes[1]
            if 1 <= idx_a <= count and 1 <= idx_b <= count and idx_a != idx_b:
                for src_idx, dst_idx in ((idx_a, idx_b), (idx_b, idx_a)):
                    src_item = proposal.items[src_idx - 1]
                    normalized.append(ProposedEdit(op="change", index=dst_idx, field="goal", value=src_item.goal))
                    normalized.append(
                        ProposedEdit(op="change", index=dst_idx, field="means", value=src_item.means.render())
                    )
                    normalized.append(
                        ProposedEdit(
                            op="change",
                            index=dst_idx,
                            field="inputs",
                            value=", ".join(inp.render() for inp in src_item.inputs) or "none",
                        )
                    )
                continue
        if edit.op == "merge" and (count < 2 or len(set(edit.indexes)) < 2):
            if edit.value.strip() and count == 1:
                normalized.append(
                    _clean_proposed_edit(
                        ProposedEdit(
                            op="change",
                            index=1,
                            field="goal",
                            value=_clean_goal_value(edit.value.strip())[0],
                        )
                    )
                )
            continue
        if edit.op in ("change", "drop", "split") and edit.index is None and len(edit.indexes) >= 2:
            for idx in edit.indexes:
                if 1 <= idx <= count:
                    normalized.append(
                        _clean_proposed_edit(edit.model_copy(update={"index": idx, "indexes": []}))
                    )
            continue
        if edit.op == "drop" and count == 1 and len(proposed) == 1:
            raise EditParseError("I could not read that as an edit.")
        for piece in _split_function_call_edit(_clean_proposed_edit(edit)):
            normalized.append(piece)
    if not normalized:
        raise EditParseError("I could not read that as an edit.")
    return normalized


def _token_overlap(left: str, right: str) -> int:
    left_tokens = set(re.findall(r"[a-z0-9]+", (left or "").lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", (right or "").lower()))
    return len(left_tokens & right_tokens)


def _is_truncated_match(partial: str, full: str) -> bool:
    if not partial or partial.lower() == full.lower():
        return False
    if partial.lower() in full.lower():
        return True
    return _token_overlap(partial, full) >= 2


def _best_quoted_match(value: str, quoted: list[str]) -> Optional[str]:
    for name in quoted:
        if value.lower() == name.lower():
            return name
        if _is_truncated_match(value, name):
            return name
    return None


def _repair_change_inputs_value(value: str, quoted: list[str], reply_text: str = "") -> str:
    parts = _split_input_assignments(value)
    if len(parts) <= 1:
        # ``name=Kline`` with the reply quoting 'Kline and Sons Ltd': the value is compared with the quoted names, not the
        # whole ``name=value`` text.
        name, equals, rest = value.partition("=")
        match = _best_quoted_match(_strip_outer_quotes(rest if equals else value), quoted)
        if match and equals and not _said_literally(_strip_outer_quotes(rest), reply_text, quoted):
            return f"{name.strip()}={match}"
        return value
    repaired: list[str] = []
    for part in parts:
        inp = SubTaskInput.from_text(part)
        if inp.name and inp.value:
            match = _best_quoted_match(inp.value, quoted)
            if match and not _said_literally(inp.value, reply_text, quoted):
                inp = inp.model_copy(update={"value": match})
            repaired.append(inp.render())
        else:
            repaired.append(part)
    return ", ".join(repaired)


_WORD_EDGE = ",.;:?!()[]'\""


def _norm_word(word: str) -> str:
    return word.strip(_WORD_EDGE).lower()


def _repair_goal_value(value: str, quoted: list[str]) -> str:
    """Complete a name the reader cut short in a goal: the words of *value* that are a run of the quoted name (not all of
    it) are replaced by the whole name. Only that run is replaced: the rest of the sentence stays, so a goal that shares
    two words with a name is not overwritten by the name."""
    result = value
    for name in sorted(quoted, key=len, reverse=True):
        if name in result:
            continue
        name_words = [_norm_word(word) for word in name.split()]
        if len(name_words) < 2:
            continue
        words = list(re.finditer(r"\S+", result))
        best: Optional[tuple[int, int]] = None
        for start in range(len(words)):
            for end in range(len(words), start, -1):
                run = [_norm_word(match.group(0)) for match in words[start:end]]
                if len(run) >= len(name_words) or not all(run):
                    continue
                if len(run) < 2 and len(words) > 1:
                    continue
                if any(run == name_words[at:at + len(run)] for at in range(len(name_words) - len(run) + 1)):
                    if best is None or end - start > best[1] - best[0]:
                        best = (start, end)
                    break
        if best is not None:
            first, last = words[best[0]], words[best[1] - 1]
            tail = last.group(0)[len(last.group(0).rstrip(_WORD_EDGE)):]
            result = result[:first.start()] + name + tail + result[last.end():]
    return result


def _clause_containing_quote(reply_text: str, quoted_name: str) -> str:
    for clause in _reply_clauses(reply_text):
        if quoted_name in clause or quoted_name.lower() in clause.lower():
            return clause
    return reply_text


def _infer_kind_from_clause(clause: str, item: "SubTask") -> Optional[tuple[str, str]]:
    """``(input name, value)`` when the clause names exactly one value of one enum input that *item*'s planning command
    declares (and the item carries); None when the workflow declares no such input or the clause is not decisive."""
    if item.means.kind != MEANS_PLANNING:
        return None
    present = {inp.name for inp in item.inputs}
    lowered = clause.lower()
    found = [(p.name, value) for p in declared_params(item.means.name) if p.type == "enum" and p.name in present
             for value in p.values if re.search(rf"(?<![a-z0-9_]){re.escape(value.lower())}(?![a-z0-9_])", lowered)]
    return found[0] if len(found) == 1 else None


def _quoted_already_in_edits(edits: list, quoted_name: str) -> bool:
    for edit in edits:
        if not isinstance(edit, Change) or edit.field != "inputs":
            continue
        for part in _split_input_assignments(edit.value):
            inp = SubTaskInput.from_text(part)
            if inp.value and (inp.value.lower() == quoted_name.lower()
                              or quoted_name.lower() in inp.value.lower()):
                return True
    return False


def _target_index_for_quoted(
    proposal: SubTaskProposal,
    reply_text: str,
    quoted_name: str,
    dropped_indexes: set[int],
) -> Optional[int]:
    clause = _clause_containing_quote(reply_text, quoted_name)
    steps = [
        step for step in _extract_step_numbers(clause, len(proposal.items))
        if step not in dropped_indexes
    ]
    if len(steps) == 1:
        return steps[0]
    surviving = [
        index for index in range(1, len(proposal.items) + 1) if index not in dropped_indexes
    ]
    if len(surviving) == 1:
        return surviving[0]
    for index, item in enumerate(proposal.items, 1):
        if index in dropped_indexes:
            continue
        for inp in item.inputs:
            if inp.name in entity_input_names() and inp.value and inp.value.lower() in quoted_name.lower():
                return index
    return None


def _primary_entity_key(item: SubTask) -> Optional[str]:
    entity = entity_input_names()
    names = [inp.name for inp in item.inputs if inp.name in entity]
    return names[0] if names else None


_PART_OF_NAME = re.compile(r"\bpart\s+of\b", re.I)


def _apply_quoted_names_from_reply(
    proposal: SubTaskProposal,
    edits: list,
    reply_text: str,
    quoted: list[str],
) -> list:
    repaired: list = []
    for edit in edits:
        if isinstance(edit, Change) and edit.field == "inputs":
            repaired.append(
                edit.model_copy(update={"value": _repair_change_inputs_value(edit.value, quoted, reply_text)})
            )
        elif isinstance(edit, Change) and edit.field == "goal":
            repaired.append(
                edit.model_copy(update={"value": _repair_goal_value(edit.value, quoted)})
            )
        elif isinstance(edit, (Merge, Add)) and edit.values.get("goal"):
            repaired.append(edit.model_copy(update={
                "values": {**edit.values, "goal": _repair_goal_value(edit.values["goal"], quoted)}}))
        else:
            repaired.append(edit)

    dropped_indexes = {edit.index for edit in repaired if isinstance(edit, Drop)}
    reader_edits = list(repaired)
    for quoted_name in _wanted_quoted_names(reply_text, quoted):
        if _quoted_already_in_edits(repaired, quoted_name):
            continue
        target_index = _target_index_for_quoted(proposal, reply_text, quoted_name, dropped_indexes)
        if target_index is None:
            continue
        target_item = proposal.items[target_index - 1]
        primary_key = _primary_entity_key(target_item)
        if any(quoted_name.lower() in (inp.value or "").lower() for inp in target_item.inputs):
            continue  # the quoted text is already the value of an input of the step (maybe another one): not the name
        if not primary_key or _reader_touches_input(reader_edits, target_index, primary_key):
            continue
        clause = _clause_containing_quote(reply_text, quoted_name)
        if _PART_OF_NAME.search(clause) and not any(
                quoted_name.lower() in (inp.value or "").lower() for inp in target_item.inputs):
            # 'X is part of the name in step 1': X is a piece of the name, not the name. Setting it would replace the name
            # by its piece, and the whole name was not given, so the user is asked for it.
            read = [f"drop {edit.index}" for edit in repaired if isinstance(edit, Drop)]
            understood = f" I understood: {'; '.join(read)}." if read else ""
            raise EditRefused(
                f"You said '{quoted_name}' is part of the {primary_key} in step {target_index}, but I do not know the "
                f"whole {primary_key}.{understood} Give the whole {primary_key} in quotes, for example "
                f"'change {target_index}: inputs={primary_key}=<whole {primary_key}>'.")
        kv = [f"{primary_key}={quoted_name}"]
        if inferred := _infer_kind_from_clause(clause, target_item):
            kv.append(f"{inferred[0]}={inferred[1]}")
        repaired.append(Change(index=target_index, field="inputs", value=", ".join(kv)))
    return repaired


# ----------------------------------------------------------------------
# Deterministic guards on the reader's edits: the reader (a small model) misreads plain-English corrections in a few
# recurring ways. These act on what it returned, never on the prompt: a drop of a whole step that the reply aimed at an
# input becomes a clearing of that input; an edit for a step the reply never names, whose inputs another named step
# holds, goes to that step; a drop the reply does not ask for is not applied.
# ----------------------------------------------------------------------

#: A declared entity input (it holds the entity a step is about; ``entity_input_names``): a bare mention of its name in a
#: reply does not select it.
_NAME_PREFIXES = ("include_", "with_", "is_", "has_", "show_")
_REMOVE_VERBS = r"(?:drop|remove|delete|skip|omit|cut|leave\s+out|take\s+out|get\s+rid\s+of|replace)"
_STEP_WORD = r"(?:steps?|items?|sub-?tasks?)"
_STEP_LIST = _NUMBER_LIST
_REMOVE_STEP_OBJECT = re.compile(
    rf"\b{_REMOVE_VERBS}\s+(?:(?:the|these|those|this|that|both|all|extra|duplicate|stray|separate)\s+)*"
    rf"(?:{_STEP_WORD}\s+)?(?P<list>{_STEP_LIST})\b", re.I)
_REMOVE_STEP_PRONOUN = re.compile(rf"\b{_REMOVE_VERBS}\s+(?:it|them|those|these|that|this|both)\b", re.I)
_STEP_NOT_WANTED = re.compile(
    rf"\b{_STEP_WORD}\s+(?P<list>{_STEP_LIST})\s+(?:(?:should|must|can|could|needs?\s+to)\s+"
    r"(?:not\s+be\s+there|go|be\s+(?:removed|dropped|deleted))|"
    r"(?:is|are)\s+(?:a\s+|an\s+)?(?:duplicates?|stray|extra|unneeded|unnecessary|not\s+needed|not\s+wanted|not\s+there)|"
    r"(?:isn'?t|aren'?t)\s+needed)", re.I)
_MERGE_REPLY = re.compile(
    r"\b(?:merge|combine|join|single\s+step|one\s+step|one\s+single|a\s+single|just\s+one|only\s+one|"
    r"one\s+(?:name|entity|organi[sz]ation|record)|not\s+two)\b", re.I)
_CLEAR_CUE_BODY = (
    r"drop|remove|delete|clear|omit|skip|leave\s+(?:it\s+|those\s+|them\s+)?out|take\s+out|get\s+rid\s+of|without|"
    r"no\s+longer|(?:did\s+not|didn'?t|never)\s+ask(?:ed)?\s+for|not\s+asked\s+for")
_CLEAR_CUE_STRONG = re.compile(rf"\b(?:{_CLEAR_CUE_BODY})\b", re.I)
_CLEAR_CUE_WEAK = re.compile(rf"\b(?:{_CLEAR_CUE_BODY}|not\s+(?:the|its|their)|not\s+just|not\s+only)\b", re.I)


def _norm_phrase(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (text or "").lower()).strip()


def _list_numbers(text: str, count: int) -> list[int]:
    """The step numbers in a list text such as '2, 3 and 5' or '4 to 6' (1..count)."""
    found: list[int] = []
    for lo, hi in re.findall(r"(\d+)\s*(?:to|through|-|–)\s*(\d+)", text):
        found.extend(range(int(lo), int(hi) + 1))
    found.extend(int(each) for each in re.findall(r"\d+", text))
    return [number for number in dict.fromkeys(found) if 1 <= number <= count]


def _leading_step_numbers(clause: str, count: int) -> list[int]:
    match = re.search(rf"{_STEP_WORD}\s+(?P<list>{_STEP_LIST})", clause, re.I)
    return _list_numbers(match.group("list"), count) if match else []


def _input_mentioned(clause: str, inp: SubTaskInput) -> bool:
    """The clause names this input: its name as a phrase (include_risk: 'risk', whole_list: 'whole list'; generic
    entity names such as unit or right never count) or its value (4 or more characters, not true/false)."""
    text = _norm_phrase(clause)
    name = (inp.name or "").lower()
    if name and name not in entity_input_names():
        variants = {_norm_phrase(name)}
        for prefix in _NAME_PREFIXES:
            if name.startswith(prefix):
                variants.add(_norm_phrase(name[len(prefix):]))
        if any(re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", text) for variant in variants if variant):
            return True
    value = _norm_phrase(inp.value or "")
    if len(value) >= 4 and value not in ("true", "false", "none") and inp.from_subtask is None:
        if re.search(rf"(?<![a-z0-9]){re.escape(value)}", text):
            return True
    return False


def _explicit_removed_steps(reply_text: str, count: int, proposal: Optional["SubTaskProposal"] = None) -> set[int]:
    """Steps the reply explicitly asks to remove ('remove steps 2, 3 and 4', 'step 2 should not be there', 'keep only
    step 1'). 'drop it' counts only when the clause does not name an input of the step it follows ('I did not ask for
    risk, so drop it' is about the risk setting). A step the reply says to keep ('do not remove step 2') is not one."""
    reading = removal_reading(reply_text, count)
    removed: set[int] = set(reading.remove)
    previous: list[int] = []
    for clause in _reply_clauses(reply_text):
        for match in _REMOVE_STEP_OBJECT.finditer(clause):
            removed.update(_list_numbers(match.group("list"), count))
        for match in _STEP_NOT_WANTED.finditer(clause):
            removed.update(_list_numbers(match.group("list"), count))
        if _REMOVE_STEP_PRONOUN.search(clause):
            lead = _leading_step_numbers(clause, count) or previous
            if proposal is not None:
                lead = [step for step in lead
                        if not any(inp.name and _input_mentioned(clause, inp) for inp in proposal.items[step - 1].inputs)]
            removed.update(lead)
        previous = _leading_step_numbers(clause, count) or previous
    if not reading.contradiction:
        removed -= set(reading.keep) - set(reading.remove)
    removed -= set(reading.uncertain)
    return removed


def _mentioned_steps(reply_text: str, count: int) -> set[int]:
    return set(_extract_step_numbers(_STEP_FINE_CLAUSE.sub("", reply_text or ""), count))


_OBJECT_END = re.compile(r"[,;:.!?]|\s+(?:and|but|so|then|because|since|which|instead|as)\b", re.I)
_PRONOUN_OBJECT = re.compile(r"^\s*(?:it|them|those|these|that|this|both|too|as\s+well)?\s*$", re.I)


def _removal_objects(clause: str, *, weak: bool) -> list[str]:
    """The text a removal cue in *clause* points at ('drop the whole-list setting' -> 'the whole-list setting'). A cue
    that points at a pronoun ('set tags to false, or remove it') points at what comes before it. A cue inside a quote
    is not one, and an object that names a step ('the separate Ann step') is a step, not an input."""
    masked = re.sub(r"(['\"])[^'\"]*\1", lambda m: " " * len(m.group(0)), clause)
    cue = _CLEAR_CUE_WEAK if weak else _CLEAR_CUE_STRONG
    objects: list[str] = []
    for match in cue.finditer(masked):
        rest = clause[match.end():]
        end = _OBJECT_END.search(masked[match.end():])
        obj = rest[:end.start()] if end else rest
        if _PRONOUN_OBJECT.match(obj):
            obj = clause[max(0, match.start() - 60):match.start()]
        if re.search(r"\b(?:steps?|items?)\b", obj, re.I):
            continue
        objects.append(obj)
    return objects


def _clear_requests(proposal: "SubTaskProposal", reply_text: str, *, weak: bool = False) -> list[tuple[int, str]]:
    """``(step, input name)`` pairs the reply asks to clear: a removal cue whose object names the input (its setting
    name or its value), in a clause that names the step (or the one step there is)."""
    count = len(proposal.items)
    found: list[tuple[int, str]] = []
    for clause in _reply_clauses(reply_text):
        objects = _removal_objects(clause, weak=weak)
        if not objects:
            continue
        steps = _extract_step_numbers(clause, count)

        def mentioned(inp: SubTaskInput) -> bool:
            return any(_input_mentioned(obj, inp) for obj in objects)

        for step in steps or ([1] if count == 1 else []):
            for inp in proposal.items[step - 1].inputs:
                if inp.name and mentioned(inp) and (step, inp.name) not in found:
                    found.append((step, inp.name))
        if not steps and count > 1:
            holders: dict[str, list[int]] = {}
            for index, item in enumerate(proposal.items, 1):
                for inp in item.inputs:
                    if inp.name and mentioned(inp):
                        holders.setdefault(inp.name, []).append(index)
            for name, where in holders.items():
                if len(where) == 1 and (where[0], name) not in found:
                    found.append((where[0], name))
    return found


def _clear_edit(step: int, name: str) -> ProposedEdit:
    return ProposedEdit(op="change", index=step, field="inputs", value=f"{name}=none")


def _edit_input_keys(edit: ProposedEdit) -> list[str]:
    field = (edit.field or "").strip().lower()
    if edit.op != "change":
        return []
    if field == "inputs":
        keys = [key for key in (_input_edit_part_key(part) for part in _input_edit_parts((edit.value or "").strip())) if key]
        return [key for key in keys if key not in _META_INPUT_KEYS]
    if field.startswith("inputs."):
        return [field[len("inputs."):]]
    if field and field not in ("goal", "outcome", "means", "check", "desired outcome", "desired_outcome"):
        return [field.replace(" ", "_")]
    return []


def _guard_drops_to_clears(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """A drop of step N the reply does not ask for ('remove step N', 'step N should not be there'), where the reply
    names one of N's inputs (a setting or a value) with a removal cue, clears that input instead."""
    count = len(proposal.items)
    if any(edit.op == "drop" for edit in proposed) and not _MERGE_REPLY.search(reply_text):
        explicit = _explicit_removed_steps(reply_text, count, proposal)
        requests = _clear_requests(proposal, reply_text, weak=True)
        result: list[ProposedEdit] = []
        for edit in proposed:
            if edit.op == "drop" and edit.index is not None and 1 <= edit.index <= count and edit.index not in explicit:
                wanted = [name for step, name in requests if step == edit.index]
                if wanted:
                    result.extend(_clear_edit(edit.index, name) for name in wanted)
                    continue
            result.append(edit)
        return result
    return proposed


def _guard_unasked_drops(
    proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str, refused: Optional[list[int]] = None,
) -> list[ProposedEdit]:
    """A drop the reply does not ask for goes to the one step it does ask to remove and the reader did not drop, or,
    when the reply never names that step, is not applied (its step numbers are added to *refused*)."""
    count = len(proposal.items)
    if count < 2 or not any(edit.op == "drop" for edit in proposed) or _MERGE_REPLY.search(reply_text):
        return proposed
    explicit = _explicit_removed_steps(reply_text, count, proposal)
    mentioned = _mentioned_steps(reply_text, count)
    uncovered = [step for step in sorted(explicit)
                 if not any(each.op == "drop" and each.index == step for each in proposed)]
    result: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "drop" and edit.index is not None and 1 <= edit.index <= count and edit.index not in explicit:
            if len(uncovered) == 1 and edit.index not in mentioned:
                result.append(edit.model_copy(update={"index": uncovered.pop()}))
                continue
            if mentioned and edit.index not in mentioned:
                if refused is not None:
                    refused.append(edit.index)
                continue
        result.append(edit)
    return result


def _guard_retarget_changes(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """An input change for a step the reply never names, whose inputs the step lacks, goes to the one named step that
    holds them."""
    count = len(proposal.items)
    mentioned = _mentioned_steps(reply_text, count)
    if count < 2 or not mentioned:
        return proposed
    result: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "change" and edit.index is not None and 1 <= edit.index <= count and edit.index not in mentioned:
            keys = _edit_input_keys(edit)
            if keys and not set(keys) & {inp.name for inp in proposal.items[edit.index - 1].inputs}:
                holders = [step for step in sorted(mentioned)
                           if set(keys) <= {inp.name for inp in proposal.items[step - 1].inputs}]
                if len(holders) == 1:
                    edit = edit.model_copy(update={"index": holders[0]})
        result.append(edit)
    return result


def _guard_unreported_clears(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """An input the reply asks to clear and no edit of the reader touches is cleared."""
    touched = {(each.index, key) for each in proposed for key in _edit_input_keys(each)}
    gone = {each.index for each in proposed if each.op in ("drop", "split")}
    gone.update(step for each in proposed if each.op == "merge" for step in each.indexes)
    result = list(proposed)
    for step, name in _clear_requests(proposal, reply_text):
        if (step, name) not in touched and step not in gone:
            result.append(_clear_edit(step, name))
            touched.add((step, name))
    return result


_ORDINAL_STEP_CUE = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|last|final|previous|next)\b", re.I)


def reconcile_dropped_items(proposal: Proposal, items: list[str], text: str) -> list[int]:
    """The steps a routed ``drop`` removes, checked against the reply's own words (``review_step_refs.removal_reading``).

    *items* are the step numbers another reader (the parameter extractor of the ``drop`` command) took from *text*.
    A reply that names the steps to remove by number removes all of those and no other (a list read short, or a step
    the reply says to keep, is corrected); a step the reply never named is not removed unless it points at a step by
    position ('the last step'). A reply with no certain reading raises ``EditRefused`` (the user is asked again, with the
    removal that was read shown back). Replies that name no step by number keep *items* as given.
    """
    count = len(proposal.items)
    numbers = [int(item) for item in items]
    reading = removal_reading(text, count)
    if unsure := [number for number in numbers if number in reading.uncertain]:
        raise EditRefused(
            f"I am not sure you want step {unsure[0]} removed: your words could mean a change to it, or a condition. "
            f"Reply 'drop {unsure[0]}' to remove it, or say what to change in it.")
    if reading.contradiction:
        clash = sorted(set(reading.remove) & set(reading.keep))
        raise EditRefused(
            f"I could not tell whether to remove or keep step {clash[0]}: you asked for both. "
            f"Which steps should I remove? (I read the removal of {_step_list_text(sorted(reading.remove))}.)")
    if reading.remove:
        # A step the extractor took that the reply neither names nor points at by position is not removed.
        extras = [number for number in numbers if number not in reading.remove and number not in reading.keep]
        if extras and _ORDINAL_STEP_CUE.search(text):
            return list(reading.remove) + extras
        return list(reading.remove)
    kept = [number for number in numbers if number in reading.keep]
    if kept:
        rest = [number for number in numbers if number not in reading.keep]
        if rest:
            return rest
        raise EditRefused(
            f"I read that as removing {_step_list_text(kept)}, but you said to keep "
            f"{'it' if len(kept) == 1 else 'them'}. Which steps should I remove?")
    return numbers


def certain_removals(proposal: Proposal, text: str) -> list[int]:
    """The steps *text* clearly asks to remove by number (``drop steps 2, 3 and 4``), or [] when it does not, or when it
    also says to keep one of them."""
    reading = removal_reading(text, len(proposal.items))
    return list(reading.remove) if reading.certain and not reading.uncertain else []


def _guard_removals(
    proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str, kept_out: Optional[list[int]] = None,
) -> list[ProposedEdit]:
    """The reply's own words decide which steps go (``review_step_refs.removal_reading``), over what the reader returned.

    A drop of a step the reply says to keep ('keep only step 1', 'steps 1 to 4 are fine', 'do not remove step 2') is not
    applied, and a step the reply asks to remove that the reader left out is dropped: a reply that names several steps
    removes all of them. A reply that both removes and keeps the same step has no certain reading: nothing is dropped
    and the user is asked, with the removal that was read shown back."""
    count = len(proposal.items)
    if count < 2:
        return proposed
    reading = removal_reading(reply_text, count)
    if not reading.remove and not reading.keep and not reading.uncertain:
        return proposed
    readers_drops = [edit for edit in proposed if edit.op == "drop"]
    unsure = [step for edit in readers_drops
              for step in ([edit.index] if edit.index is not None else []) + list(edit.indexes)
              if step in reading.uncertain]
    if unsure:
        # A removal verb touched this step without a certain reading ('remove step 2's filter', 'skip step 2 if ...'), and
        # the reader dropped the step: not applied, the user is asked, with the removals that were certain shown back.
        understood = (f" (I did read the removal of {_step_list_text(list(reading.remove))}.)" if reading.remove else "")
        raise EditRefused(
            f"I am not sure you want step {unsure[0]} removed: your words could mean a change to it, or a condition."
            f"{understood} Reply 'drop {unsure[0]}' to remove it, or say what to change in it.")
    if reading.contradiction:
        if not readers_drops and not reading.remove:
            return proposed
        clash = sorted(set(reading.remove) & set(reading.keep))
        raise EditRefused(
            f"I could not tell whether to remove or keep step {clash[0]}: you asked for both. "
            f"Which steps should I remove? (I read the removal of {_step_list_text(sorted(reading.remove))}.)")
    kept = [edit for edit in readers_drops
            if edit.index in reading.keep and edit.index not in reading.remove
            or any(each in reading.keep and each not in reading.remove for each in edit.indexes)]
    result = [edit for edit in proposed if edit not in kept]
    if kept_out is not None:
        kept_out.extend(edit.index for edit in kept if edit.index is not None)
    gone: set[int] = set()
    for edit in result:
        if edit.op == "drop":
            gone.update(([edit.index] if edit.index is not None else []) + list(edit.indexes))
        elif edit.op == "split" and edit.index is not None:
            gone.add(edit.index)
        elif edit.op == "merge":
            gone.update(edit.indexes[1:] if len(edit.indexes) > 1 else [])
    aimed_at_input = {step for step, _ in _clear_requests(proposal, reply_text, weak=True)}
    for step in reading.remove:
        if step not in gone and step not in aimed_at_input:
            result.append(ProposedEdit(op="drop", index=step))
    return result


def _step_list_text(steps: list[int]) -> str:
    labels = [str(step) for step in steps]
    if len(labels) <= 1:
        return f"step {labels[0]}" if labels else "no step"
    return f"steps {', '.join(labels[:-1])} and {labels[-1]}"


def _guard_switch_clause_in_goal(proposed: list[ProposedEdit], reply_text: str) -> None:
    """Refuse a reader edit whose goal is, or ends with, an explicit plan-switch clause copied from the reply (``step 7: use
    NAME``): the clause asks to change a plan, and as goal text it would damage the step it lands in (bd fix-1ma8.1)."""
    from fastworkflow.review.review_engine import switch_clause_in_goal_message
    from fastworkflow.review.review_plan_switch import goal_copies_switch_clause

    for edit in proposed:
        goals = ([edit.value] if edit.op in ("add", "merge") or (edit.op == "change" and edit.field == "goal")
                 else list(edit.parts) if edit.op == "split" else [])
        for goal in goals:
            if goal and (clause := goal_copies_switch_clause(goal, reply_text)):
                raise EditRefused(switch_clause_in_goal_message(clause))


def _guard_reader_edits(
    proposal: Proposal, proposed: list[ProposedEdit], reply_text: Optional[str],
) -> list[ProposedEdit]:
    """The deterministic guards on the reader's edits, in order. Raises ``EditRefused`` (a question for the user) when
    every edit left was a drop the reply does not ask for."""
    if not reply_text or not isinstance(proposal, SubTaskProposal) or not proposed:
        return proposed
    _guard_switch_clause_in_goal(proposed, reply_text)
    if len(proposal.items) == 1:
        proposed = [edit.model_copy(update={"index": 1}) if edit.index is None and edit.op in ("change", "drop") else edit
                    for edit in proposed]
    refused: list[int] = []
    kept_out: list[int] = []
    proposed = _guard_drops_to_clears(proposal, proposed, reply_text)
    proposed = _guard_retarget_changes(proposal, proposed, reply_text)
    proposed = _guard_removals(proposal, proposed, reply_text, kept_out)
    proposed = _guard_unasked_drops(proposal, proposed, reply_text, refused)
    proposed = _guard_unreported_clears(proposal, proposed, reply_text)
    if not proposed and kept_out:
        raise EditRefused(
            f"I read that as removing step {kept_out[0]}, but you said to keep it. Which steps should I remove?")
    if not proposed and refused:
        raise EditRefused(
            f"I read that as removing step {refused[0]}, but you did not ask for that. "
            "Which step or setting do you mean?")
    return proposed


def _reader_touches_input(edits: list, index: int, key: str) -> bool:
    """Whether the reader's own edits already set or clear input *key* of step *index*."""
    for edit in edits:
        if isinstance(edit, Change) and edit.index == index and edit.field == "inputs":
            for part in _input_edit_parts((edit.value or "").strip()):
                if _input_edit_part_key(part) == key.lower():
                    return True
    return False


def _negated_before(text: str, start: int) -> bool:
    return bool(re.search(
        r"(?:\bnot|\bwithout|\binstead\s+of|\brather\s+than|\bno\s+longer|\bnor)\b[^'\"]{0,18}$", text[:start], re.I))


def _wanted_quoted_names(reply_text: str, quoted: list[str]) -> list[str]:
    """The quoted names the reply asks for: one it rules out ('not 'X'', 'without 'X'') is not wanted."""
    wanted: list[str] = []
    for name in quoted:
        positions = [m.start() for m in re.finditer(re.escape(name), reply_text)]
        if positions and all(_negated_before(reply_text, pos) for pos in positions):
            continue
        wanted.append(name)
    return wanted


def _said_literally(value: str, reply_text: str, quoted: list[str]) -> bool:
    """The user wrote *value* as such (outside quotes, and not as 'not just VALUE'): a quoted name is not its completion."""
    if not value or not reply_text:
        return False
    text = reply_text
    for name in quoted:
        text = text.replace(name, " " * len(name))
    for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", text, re.I):
        if not re.search(r"\b(?:not|without|instead\s+of)\s+(?:just\s+|only\s+)?$", text[:match.start()], re.I):
            return True
    return False


def _proposed_edits_to_engine(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    *,
    reply_text: Optional[str] = None,
) -> list:
    proposed = _normalize_reader_edits(proposal, proposed, reply_text=reply_text)
    proposed = _explicit_step_of_reply(proposal, proposed, reply_text)
    proposed = _infer_missing_indexes_from_reply(proposal, proposed, reply_text)
    proposed = _dedupe_proposed_edits(proposed)
    proposed = _guard_reader_edits(proposal, proposed, reply_text)
    proposed = [edit for edit in proposed if not (edit.op == "change" and not edit.field and any(
        other.op == "change" and other.field and other.index == edit.index and other.value == edit.value
        for other in proposed))]  # a field-less change that repeats one naming its field is that change, said twice
    count = len(proposal.items)
    if count > 1 and any(
        edit.op in ("change", "drop", "split") and edit.index is None for edit in proposed
    ):
        raise PendingItemSlot(f"Which of the {count} steps do you mean?", proposed)
    resolved: list = []
    for edit in proposed:
        if edit.op == "change":
            if not edit.field:
                raise InvalidReadEdit("An edit names no field to change.")
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        elif edit.op == "drop":
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        elif edit.op == "split":
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        else:
            resolved.append(_engine_edit(edit, edit.index, proposal))
    resolved = _dedupe_engine_edits(resolved, reply_text=reply_text, proposal=proposal)
    if reply_text and isinstance(proposal, SubTaskProposal):
        quoted = extract_quoted_names(reply_text)
        if quoted:
            resolved = _apply_quoted_names_from_reply(proposal, resolved, reply_text, quoted)
            resolved = _dedupe_engine_edits(resolved, reply_text=reply_text, proposal=proposal)
        resolved = _drop_moved_values(proposal, resolved, reply_text)
    return resolved


#: A step the reply says it is about: ``step 3 should``, ``step 3's``, ``step 3:``, ``in step 3``, ``change 3:``.
_TARGET_STEP = re.compile(
    r"\b(?:in|for|on|change|fix|edit|update|make)\s+(?:step|item)\s+#?(\d+)\b"
    r"|\b(?:step|item)\s+#?(\d+)(?:'s|\s*:|\s+(?:should|must|needs?|wants?)\b)"
    r"|^\s*change\s+(\d+)\s*:",
    re.I | re.M)
#: A step named as where a value comes from, not as the step to change: ``the output of step 1's search``.
_SOURCE_BEFORE = re.compile(r"\b(?:of|from|after|before|than|like|to|as)\s+$", re.I)


def _explicit_step_of_reply(
    proposal: Proposal, proposed: list[ProposedEdit], reply_text: Optional[str]) -> list[ProposedEdit]:
    """The reader's ``change`` edits held to the one step the reply says it is about (``Step 3's name is too long``,
    ``change 3: ...``): when the reply names exactly one such step and the model numbered a change edit otherwise, the
    reply's number stands. A reply about several steps, or none, keeps the model's numbers."""
    count = len(proposal.items)
    if not reply_text or count < 2:
        return proposed
    named = {int(next(group for group in match.groups() if group)) for match in _TARGET_STEP.finditer(reply_text)
             if not _SOURCE_BEFORE.search(reply_text[:match.start()])}
    if len(named) != 1:
        return proposed
    step = next(iter(named))
    if not 1 <= step <= count:
        return proposed
    return [edit.model_copy(update={"index": step, "indexes": []})
            if edit.op == "change" and edit.index is not None and edit.index != step else edit
            for edit in proposed]


def _names_input(reply_text: str, name: str) -> bool:
    """Whether the reply names input *name* (``name``, ``connected name`` or ``connected_name``) as a word."""
    spoken = re.escape(name).replace("_", "[ _]")
    return re.search(rf"(?<![A-Za-z0-9]){spoken}(?![A-Za-z0-9])", reply_text, re.I) is not None


def _drop_moved_values(proposal: "SubTaskProposal", edits: list, reply_text: str) -> list:
    """The reader's input edits without one that copies a value another input of the same step holds (and keeps) into
    an input the reply never names (``GitLab is an application, not a feed`` read as ``name=<the person's value>``): the
    edit is left out and the read-back says so (``note_reader_drop``). An edit of an input the reply names stands,
    so ``the name is X and the person is Y`` is read as written."""
    items = proposal.items
    kept: list = []
    dropped: list[str] = []
    for edit in edits:
        if not (isinstance(edit, Change) and edit.field == "inputs" and 1 <= edit.index <= len(items)):
            kept.append(edit)
            continue
        item = items[edit.index - 1]
        try:
            after = {each.name: each for each in merge_subtask_input_edit(item.inputs, edit.value) if each.name}
        except Exception:  # noqa: BLE001 - an edit that cannot be read is left to apply_edits, which says why
            kept.append(edit)
            continue
        held = {each.name: each.value for each in item.inputs if each.name and each.from_subtask is None and each.value}
        parts = _input_edit_parts((edit.value or "").strip())
        left: list[str] = []
        for part in parts:
            new = SubTaskInput.from_text(part)
            value = (new.value or "").strip()
            source = next((name for name, held_value in held.items()
                           if name != new.name and held_value.strip().casefold() == value.casefold()
                           and after.get(name) is not None and (after[name].value or "") == held_value), None)
            if (new.name and new.from_subtask is None and value and source is not None
                    and held.get(new.name, "").strip().casefold() != value.casefold()
                    and not _names_input(reply_text, new.name)):
                dropped.append(f"I left {new.name} of step {edit.index} as it was: your reply does not name it, and "
                               f"{value!r} is the {source}.")
                note_reader_drop(dropped[-1])
                continue
            left.append(part)
        if len(left) == len(parts):
            kept.append(edit)
        elif left:
            kept.append(Change(index=edit.index, field="inputs", value=", ".join(left)))
    if dropped and not kept:
        raise EditRefused(" ".join(dropped) + " Say which input should change and to what, or reply yes to run it as "
                                              "shown.")
    return kept


_PATH_PIECES = re.compile(r"[.\[\]\s]+")


def _input_names_of(item: "SubTask") -> Optional[frozenset[str]]:
    """The input names *item* can take: the ones it holds and its command's ``Input`` fields (a planning command's
    declared parameters, an execution command's ``Input`` fields); None when nothing bounds the names: the workflow is
    not known here, the part is ad hoc, or its command declares no inputs the workflow can read."""
    names = {each.name for each in item.inputs if each.name}
    folder = _folder_var.get()
    if not folder:
        return None
    if item.means.kind == MEANS_PLANNING:
        declared = declared_params(item.means.name)
        if not declared:
            return None     # a command the workflow does not declare (or one with no inputs): nothing bounds the names
        names |= {p.name for p in declared}
    elif item.means.kind == MEANS_PLAIN:
        from fastworkflow import ModuleType, RoutingRegistry

        try:
            params_class = RoutingRegistry.get_definition(folder).get_command_class(
                item.means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
        except Exception:  # noqa: BLE001 - no readable Input: nothing bounds the names
            params_class = None
        fields = getattr(params_class, "model_fields", None)
        if not fields:
            return None
        names |= set(fields)
    else:
        return None     # an ad hoc part has no command whose inputs bound the names
    return frozenset(names)


def _edited_input_name(proposal: "SubTaskProposal", number: Optional[int], field: str) -> str:
    """The input an edit's *field* names on item *number*: the field itself, or the last piece of a path the reader
    wrote (``sub_tasks[1].inputs.kind`` is ``kind``), whose item number, when the path writes one, must be *number*.
    A field that names no input the item can take raises ``EditParseError`` with the valid names (asked again), never
    an input with no name."""
    pieces = [piece for piece in _PATH_PIECES.split(field.strip()) if piece]
    name = pieces[-1] if pieces else field.strip()
    numbers = [int(piece) for piece in pieces[:-1] if piece.isdigit()]
    item = proposal.items[number - 1] if isinstance(number, int) and 1 <= number <= len(proposal.items) else None
    valid = _input_names_of(item) if item is not None else None
    if numbers and number is not None and any(n != number for n in numbers):
        raise InvalidReadEdit(f"The edit of {field!r} names another step than step {number}.")
    if not re.fullmatch(r"[A-Za-z_]\w*", name) or (valid is not None and name not in valid):
        names = ", ".join(sorted(valid)) if valid else "none"
        raise InvalidReadEdit(f"Step {number} has no input {field!r}. Its inputs are: {names}.")
    return name


def _check_input_names(proposal: "SubTaskProposal", number: Optional[int], value: str) -> None:
    """A whole-input edit (``field="inputs"``) is held to the step's real input names, as an edit of one named input is
    (``_edited_input_name``): each ``name=value`` piece must name an input the step can take, and a bare value with no
    name is refused when the step's inputs are known (it would land in no input, or in the wrong one). ``none`` (clear
    them all) and ``from sub-task N`` stand."""
    text = (value or "").strip()
    if not text or text.lower() == "none":
        return
    item = proposal.items[number - 1] if isinstance(number, int) and 1 <= number <= len(proposal.items) else None
    valid = _input_names_of(item) if item is not None else None
    for part in _input_edit_parts(text):
        piece = SubTaskInput.from_text(part)
        if piece.name:
            if piece.name.lower() not in _META_INPUT_KEYS:
                _edited_input_name(proposal, number, piece.name)
        elif piece.from_subtask is None and valid is not None:
            raise InvalidReadEdit(f"Step {number}: {part.strip()!r} names no input. Its inputs are: "
                                 f"{', '.join(sorted(valid)) or 'none'}.")


def edited_means(proposal: "SubTaskProposal", value: str) -> str:
    """The value of a means edit, read or typed, as it applies (``_known_means``); a means a step of *proposal* already
    holds (a swap moves it) is not checked again. Raises ``UnknownMeansEdit`` for a name that is no command."""
    if value.strip() in {item.means.render() for item in proposal.items}:
        return value
    return _known_means(value)


def _known_means(text: str) -> str:
    """The text form of a means the reader read (``planning:NAME``, ``plain:NAME``, a bare NAME or ``from_scratch``) with
    its name resolved against the commands a step can be bound to (``review_plan_switch.known_plan_names`` and its
    resolver: case, spaces, ``_`` and the context prefix ignored), as ``<kind>:<qualified name>``. A name that resolves
    to no command, or to several, raises ``UnknownMeansEdit`` with the names (bd fix-ftdl.11): the step is never turned
    into an ad hoc one. Unchecked when no workflow is in scope or its commands cannot be read."""
    from fastworkflow.core import no_agent_mode
    from fastworkflow.review import review_plan_switch

    stripped = (text or "").strip()
    if stripped == MEANS_FROM_SCRATCH:
        return stripped
    folder = _folder_var.get()
    universe = review_plan_switch.known_plan_names(_FolderOnly(folder)) if folder else {}
    if not universe:
        return stripped
    kind, colon, rest = stripped.partition(":")
    name = rest.strip() if colon and kind.strip() in (MEANS_PLANNING, MEANS_PLAIN) else stripped
    state, names = review_plan_switch._resolve_name(name, universe)
    if state == "one":
        return f"{universe[names[0]]}:{names[0]}"
    if state == "many":
        raise UnknownMeansEdit(f"{name!r} names several commands: {', '.join(names)}. Name one of them.")
    every = sorted(universe)
    cap = no_agent_mode.AD_HOC_LIST_CAP
    listed = ", ".join(every[:cap]) + (f", and {len(every) - cap} more" if len(every) > cap else "")
    raise UnknownMeansEdit(f"{name!r} is not a command available here. The commands are: {listed}.")


def _engine_edit(edit: ProposedEdit, index: Optional[int] = None, proposal: Optional[Proposal] = None):
    """The review engine's typed edit for one read edit (the same edits the deterministic reader makes)."""
    number = edit.index if index is None else index
    if edit.op == "change":
        if not edit.field:
            raise InvalidReadEdit("An edit names no field to change.")
        field = edit.field.strip()
        field_key = field.lower().replace("_", " ")
        if field_key in ("desired outcome", "desired_outcome"):
            field = "outcome"
        elif field_key in ("title", "step title", "description", "summary", "prompt"):
            field = "goal"
        value = edit.value
        if isinstance(proposal, SubTaskProposal) and field not in ("goal", "outcome", "inputs", "means", "check"):
            name = _edited_input_name(proposal, number, field)
            if value.strip().startswith(f"{field}="):
                value = value.strip()[len(field) + 1:]
            value = f"{name}={value}"
            field = "inputs"
        elif isinstance(proposal, SubTaskProposal) and field == "inputs":
            _check_input_names(proposal, number, value)
        elif isinstance(proposal, SubTaskProposal) and field == "means":
            value = edited_means(proposal, value)
        return Change(index=number, field=field, value=value)
    if edit.op == "drop":
        return Drop(index=number)
    if edit.op == "add":
        if not edit.value.strip():
            raise InvalidReadEdit("An added sub-task needs its goal.")
        after = 0 if edit.after is None else edit.after
        return Add(after=after, values={"goal": edit.value})
    if edit.op == "merge":
        return Merge(indexes=list(edit.indexes), values={"goal": edit.value} if edit.value.strip() else {})
    parts = [part for part in edit.parts if part.strip()]
    if not parts:
        raise InvalidReadEdit("A split needs at least one part.")
    return Split(index=number, parts=[{"goal": part} for part in parts])


def model_edit_reader(planner_lm=None, planning_names: tuple[str, ...] = ()):
    """The reader of free-text replies to a sub-task review, as the review engine calls it: ``(text, proposal) -> edits``.

    One model call (``ReadSubTaskEdits``, typed) on the planner's role. ``planner_lm`` is the planner's LM, or a callable
    giving it at reply time (the turn's stashed planner LM, read when the reply comes); None or a callable giving None
    takes ``LLM_PLANNER``. Any reply it cannot read as edits raises ``EditParseError``, which asks the user again and
    counts toward the review's reply cap. The edits are validated by the engine before they are applied.
    """

    def read(text: str, proposal: Proposal) -> list:
        # The review's own record of the read (bd fix-qjne.34 step 5): a child of the review, its model calls its own.
        host = tracing.current_host()
        span = tracing.start_span(host, tracing.SPAN_REVIEW_EDIT, parent_span_id=tracing.open_stage_id(host))
        facts: dict = {}
        try:
            edits = read_edits(text, proposal, facts)
        except BaseException as exc:
            tracing.end_span(host, span, status=tracing.status_for_dispatch_exception(exc), attributes={
                "edits_read": facts.get("read"), "edits_applied": 0, "repaired": facts.get("repaired", False),
                "refused_names": str(exc) if isinstance(exc, UnknownMeansEdit) else None})
            raise
        tracing.end_span(host, span, attributes={
            "edits_read": facts.get("read"), "edits_applied": len(edits), "repaired": facts.get("repaired", False),
            "refused_names": facts.get("refused_names")})
        return edits

    def read_edits(text: str, proposal: Proposal, facts: dict) -> list:
        from fastworkflow.utils import dspy_utils

        # A callable that is not an LM (dspy.BaseLM is callable too) is asked for the LM now.
        lm = planner_lm() if callable(planner_lm) and not isinstance(planner_lm, dspy.BaseLM) else planner_lm
        if lm is None:
            lm = dspy_utils.get_lm("LLM_PLANNER")
        inputs = {"proposal": proposal.model_rendering() if isinstance(proposal, SubTaskProposal) else proposal.rendering(),
                  "planning_commands": "\n".join(f"planning:{name}" for name in planning_names) or "none",
                  "reply": text}
        proposed = _predicted_edits(lm, ReadSubTaskEdits, inputs)
        facts["read"] = len(proposed)
        if not proposed:
            raise NoEditsRead("I could not read that as an edit.")
        notes = _reader_notes.get()
        said = len(notes) if notes is not None else 0
        try:
            return _validated_reader_edits(proposal, proposed, text)
        except _Unappliable as failed:
            if notes is not None:
                del notes[said:]              # what the first reading left out is not said of the second
            # One more read with the reason the edits cannot be applied, before the user is asked (bd fix-ftdl.8.28):
            # a model call of its own, whose span's input carries ``refused``.
            logger.info("the edits read from the reply cannot be applied (%s); reading it once more", failed.error)
            facts["repaired"] = True
            if isinstance(failed.error, UnknownMeansEdit):
                facts["refused_names"] = str(failed.error)
            given = json.dumps([edit.model_dump(exclude_defaults=True) for edit in proposed], ensure_ascii=False)
            repaired = _predicted_edits(lm, RepairSubTaskEdits, {
                **inputs, "refused": f"Your edits: {given}\nThey cannot be applied: {failed.error}"})
            if not repaired:
                raise failed.error from None
            return _proposed_edits_to_engine(proposal, repaired, reply_text=text)

    return read


def _predicted_edits(lm, signature, inputs: dict) -> list:
    """The edit model's edits for *inputs* (one call); a model failure is an unreadable reply (``EditParseError``)."""
    from fastworkflow.utils import dspy_utils

    try:
        with dspy_utils.role_context("LLM_PLANNER", lm):
            prediction = dspy.Predict(signature)(**inputs)
        return list(prediction.edits or [])
    except Exception as exc:  # noqa: BLE001 - an unreadable reply asks again; it never fails the turn
        logger.info("the reply could not be read as edits (%s)", type(exc).__name__)
        raise EditParseError("I could not read that as an edit.") from exc


def _validated_reader_edits(proposal: Proposal, proposed: list, text: str) -> list:
    """The reader's edits as the engine's typed edits, checked as the review will check them before they are shown: the
    fields, the inputs and the means each edit names (``_engine_edit``: ``InvalidReadEdit``, ``UnknownMeansEdit``), the
    edited proposal's schema and item numbers (``apply_edits``) and the choices its inputs declare (``refused_choice_message``). A failure of these
    checks raises ``_Unappliable`` (the reader is asked once more); the guards' questions to the user raise as they are."""
    from fastworkflow.review.review_engine import apply_edits

    try:
        edits = _proposed_edits_to_engine(proposal, proposed, reply_text=text)
    except (InvalidReadEdit, UnknownMeansEdit) as invalid:
        raise _Unappliable(invalid) from invalid
    try:
        edited = apply_edits(proposal, edits)
    except EditRefused as refused:
        raise _Unappliable(refused) from refused
    folder = _folder_var.get()
    if folder and (message := refused_choice_message(_FolderOnly(folder), edited)) is not None:
        raise _Unappliable(EditRefused(message))
    return edits


class _Unappliable(Exception):
    """The reader's edits fail a check the review makes before it shows them (``error``: what the user would be told)."""

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True)
class _FolderOnly:
    """A workflow as ``refused_choice_message`` reads it: its folder."""

    folderpath: str


# ----------------------------------------------------------------------
# The hand-off (fix-3rsq.plan-review.2.9): a sub-task's inputs and the text its run gets
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedInputs:
    """A sub-task's inputs as values: the named ones, the unnamed ones, and the 'from sub-task N' ones that are not yet
    given (sub-task N has not run)."""

    named: dict[str, str] = field(default_factory=dict)
    unnamed: tuple[str, ...] = ()
    missing: tuple[int, ...] = ()


def resolve_inputs(item: SubTask, answers: Mapping[int, str]) -> ResolvedInputs:
    """The values of *item*'s inputs. ``answers`` maps a sub-task's 1-based number to the output its run gave (its
    Output where it has one, else its answer text); a ``from sub-task N`` input takes N's, and is missing until N ran."""
    named: dict[str, str] = {}
    unnamed: list[str] = []
    missing: list[int] = []
    for each in item.inputs:
        # A drafted input is resolved as a given one (fix-48np.1): the review showed it, so the run uses it.
        if each.from_subtask is not None:
            if each.from_subtask not in answers:
                missing.append(each.from_subtask)
                continue
            value = answers[each.from_subtask]
        else:
            value = each.value or ""
        if each.name:
            named[each.name] = value
        else:
            unnamed.append(value)
    return ResolvedInputs(named=named, unnamed=tuple(unnamed), missing=tuple(missing))


def run_text(item: SubTask, resolved: ResolvedInputs) -> str:
    """The text a sub-task's run gets: its goal, its inputs and its desired outcome."""
    given = [f"{name}={value}" for name, value in resolved.named.items()] + list(resolved.unnamed)
    parts = []
    if given:
        parts.append("inputs: " + ", ".join(given))
    if item.outcome:
        parts.append(f"desired outcome: {item.outcome}")
    return f"{item.goal} ({'; '.join(parts)})" if parts else item.goal


# ----------------------------------------------------------------------
# The run of a reviewed sub-task (fix-48np.1): the inputs the review showed are the values that run
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedValues:
    """A bound sub-task's values at run time: the ones the review showed (``shown``), the ones drafted at run time
    because the review never drafted the sub-task for its means (``drafted_at_run``), and the required fields still
    without a value (``unbound``)."""

    shown: dict[str, str] = field(default_factory=dict)
    drafted_at_run: dict[str, str] = field(default_factory=dict)
    unbound: tuple[str, ...] = ()

    @property
    def values(self) -> dict[str, str]:
        return {**self.drafted_at_run, **self.shown}


def reviewed_values(workflow, item: SubTask, resolved: ResolvedInputs, required: list[str], *,
                    earlier: list[SubTask], answers: Mapping[int, str], planner_lm=None) -> ReviewedValues:
    """The values a bound sub-task runs with, when drafting is on (fix-48np.1).

    The inputs the review bound (*resolved*: drafted, edited by the user, or ``from sub-task N``) are presets: never
    extracted again. Only a sub-task the draft did not draft for its current means (``SubTask.drafted_means``: the
    draft failed, or the user added it) that leaves a *required* field unbound has its unbound fields drafted at run
    time, from the goal, by the draft's own function (``draft_inputs``: the shared extractor, its retry and the typed
    fill fallback), so the draft and the run follow the same rules. A sub-task the draft did draft is not drafted again.
    """
    shown = dict(resolved.named)
    for name, value in _produced_fields(workflow, item, answers).items():
        if value is None:
            # Sub-task N's typed output does not hold this input: N cannot produce it, so it has no value (its whole
            # output is not the value), and the run's checks refuse it as missing.
            shown.pop(name, None)
        else:
            shown[name] = value
    found: dict[str, str] = {}
    if item.drafted_means != item.means.name and any(name not in shown for name in required):
        drafted = draft_inputs(workflow, item, item.goal, earlier, planner_lm=planner_lm)
        unbound = [each for each in (drafted.inputs if drafted is not None else ()) if each.name not in shown]
        found = dict(resolve_inputs(SubTask(goal=item.goal, inputs=unbound), answers).named)
    values = {**found, **shown}
    return ReviewedValues(shown=shown, drafted_at_run=found,
                          unbound=tuple(name for name in required if name not in values))


def _produced_fields(workflow, item: SubTask, answers: Mapping[int, str]) -> dict[str, Optional[str]]:
    """``{field: value}`` for the ``from sub-task N`` inputs of *item* whose sub-task N's output is a JSON object (a typed
    ``Output``): the value of the producer field the input's field declares (``from_command_output="find_book.book_uids"``),
    the one element of a one-element list, or None when that output holds none (N cannot produce the input, a field that
    declares no producer included). An input whose N answered in text keeps N's whole output (``resolve_inputs``)."""
    from fastworkflow import ModuleType, RoutingRegistry

    try:
        params_class = RoutingRegistry.get_definition(workflow.folderpath).get_command_class(
            item.means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:  # noqa: BLE001 - no declared producers: the whole output stands
        return {}
    fields = getattr(params_class, "model_fields", {}) or {}
    out: dict[str, str] = {}
    for each in item.inputs:
        if each.from_subtask is None or not each.name or each.name not in fields \
                or each.from_subtask not in answers:
            continue
        extra = fields[each.name].json_schema_extra
        refs = (extra.get("available_from") or ()) if isinstance(extra, dict) else ()
        try:
            output = json.loads(answers[each.from_subtask])
        except (TypeError, ValueError):
            continue
        if not isinstance(output, dict):
            continue
        out[each.name] = None
        for ref in refs:
            key = str(ref).rpartition(".")[2]
            if key in output:
                value = output[key]
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                out[each.name] = _input_text(value)
                break
    return out


def reviewed_arguments(workflow, command, item: SubTask, resolved: ResolvedInputs, request: str, *,
                       earlier: list[SubTask], answers: Mapping[int, str], earlier_outputs: tuple[str, ...] = (),
                       planner_lm=None):
    """``(ReviewedValues, run_listing.ArgumentCheck)`` for a sub-task bound to planning command *command* (fix-48np.1).

    The values (``reviewed_values``) go through the checks a typed planning call runs (``checked_arguments``):
    the guards (``check_arguments``) against *request* with the turn's earlier outputs after it, so a value an earlier
    part found counts as written, and the ``Input``'s model validators against *request* alone: their ``command_text``
    is what the user said, never the answer text of an earlier part (bd fix-ftdl.10). A refusal is returned: the
    caller hands the sub-task to the agent with it (F2) and records the deviation.
    """
    from fastworkflow.planning import planning_commands
    from fastworkflow.run_listing import ArgumentCheck

    reviewed = reviewed_values(workflow, item, resolved, [p.name for p in command.params if p.required],
                               earlier=earlier, answers=answers, planner_lm=planner_lm)
    notes = ([f"drafted at run time (the review bound no value): {', '.join(sorted(reviewed.drafted_at_run))}"]
             if reviewed.drafted_at_run else [])
    try:
        args, more = planning_commands.checked_arguments(
            command, reviewed.values, "\n\n".join([request, *earlier_outputs]), workflow, user_text=request)
    except planning_commands.PlanningCommandCallError as refused:
        return reviewed, ArgumentCheck(refusal=str(refused), notes=tuple(notes))
    return reviewed, ArgumentCheck(args=args, notes=tuple(notes + more))


def _same_value(shown: str, ran: Any) -> bool:
    """Whether a value that ran is the value the review showed: the same text, ignoring case, surrounding quotes and
    space (the checks may take the request's own spelling of it)."""
    def norm(text: str) -> str:
        return " ".join(str(text).strip().strip("\"'`").split()).casefold()

    return norm(shown) == norm(_input_text(ran) if not isinstance(ran, str) else ran)


def binding_record(number: int, item: SubTask, reviewed: Optional[ReviewedValues], ran: Optional[Mapping[str, Any]],
                   deviation: Optional[str] = None) -> dict:
    """One sub-task's shown == run record for the turn span's ``subtask_bindings`` (fix-48np.1).

    *ran* is the arguments the bound command ran with (None when it did not run with them). A deviation is recorded
    when the review's values were refused or left a required field unbound (the agent then handled the sub-task, F2),
    a value was drafted at run time (it was not shown), or a value that ran is not the one shown.
    """
    shown = dict(reviewed.shown) if reviewed is not None else {}
    reasons = [deviation] if deviation else []
    if reviewed is not None and reviewed.drafted_at_run:
        reasons.append(f"drafted at run time: {', '.join(sorted(reviewed.drafted_at_run))}")
    if ran is None and not reasons:
        reasons.append("the bound command did not run with the reviewed inputs")
    if ran is not None:
        differs = sorted(name for name, value in shown.items() if name not in ran or not _same_value(value, ran[name]))
        if differs:
            reasons.append(f"ran a value other than the one shown: {', '.join(differs)}")
    return {"subtask": number, "means": item.means.render(), "shown": shown,
            "ran": {name: _input_text(value) for name, value in ran.items()} if ran is not None else None,
            "match": not reasons, "deviation": "; ".join(reasons) or None}
