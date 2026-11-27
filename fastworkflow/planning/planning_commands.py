r'''Planning commands as parameterized commands.

A planning command is one Python file, ``_commands/<name>.py``, written with the abstractions of a
command file: a ``Signature`` whose docstring describes the job (what the planning command matcher and
the text planner read) and whose pydantic ``Input`` declares typed parameters, and a
``class PlanGenerator`` (as a command has a ``ResponseGenerator``) whose methods write the
planning command's steps for one set of arguments (``fastworkflow.planning``). The planner fills the
parameters from the request (one typed dspy call), the arguments are checked, and the
``PlanGenerator`` expands them into the numbered plan. A planning command without parameters and
without phases gives fixed steps: a call of it runs those steps as its one phase, with no
planning call, and its steps are also the template the text planner reads for a request
it matches (see ``workflow_agent.load_planning_commands``); one without parameters that has phases is a
planning command call with no arguments (its phases, with no planning call).

A planning command sits in ``_commands`` as an execution command does: a root planning command is the file
``_commands/<name>.py`` whose name is the file stem, and a context planning command in
``_commands/<Context>/<name>.py`` is named ``"<Context>/<stem>"`` (as ``_commands/<Context>/``
names its commands); the file stem is a Python identifier (snake_case: ``show_record.py``).
A ``.py`` file or subfolder whose name starts with ``_`` or ``.`` is skipped (``_kinds.py``
is a helper the planning command files import with ``from ._kinds import FIND``, not a planning command).
``_commands/_workflow.py`` is not a planning command either: it holds the workflow's own settings, as the class
attributes of its ``class Context`` (the whole workflow is a context with this one settings file): the splitter's
examples, the identifier patterns of the split merge, the referent step's subject hints; see
``workflow_agent.split_settings``. No other file is read.

The workflow settings file, every attribute optional::

    import re


    class Context:
        # This workflow's own splitting examples: they replace the built-in splitter's generic ones.
        split_examples = [
            ("Close acct_4421 and acct_4422.", ["Close acct_4421.", "Close acct_4422."]),
        ]

        # Account numbers, as the commands take them: acct_4421
        identifier_patterns = [re.compile(r"\bacct_[0-9]+")]

        subject_hints = """Records in this workflow are accounts and their owners."""

``split_examples`` are given to the splitter as demos, with the built-in rules (a malformed list fails the load).
Any other public member of ``Context`` fails the load (a context hook too: the workflow has no parent), and so does
a pattern that does not compile, a file that does not import, a file with no ``class Context``, and a
``_commands/_split.py`` (the removed spelling of these settings).

A planning command file::

    from typing import Literal

    import fastworkflow
    from pydantic import BaseModel, Field, model_validator

    FIND = {"group": "find_group", "unit": "find_unit"}


    class Signature:
        """Show one named group or unit and its members. One job per record."""

        class Input(BaseModel):
            name: str = Field(description="the record's name",
                              json_schema_extra={"record_named_in_command_text": {"ignore": ["group", "unit"]}})
            kind: Literal["group", "unit"] = Field(
                description="what the request is about",
                json_schema_extra={"classifier_confirms": {"meanings": {"group": "a group of people", "unit": "an organisation unit"}}})
            person: str = Field(default="", description="one named person to check for; empty for everyone",
                                json_schema_extra={"record_named_in_command_text": True})
            include_risk: bool = Field(default=False, description="the request asks how risky it is",
                                       json_schema_extra={"classifier_confirms": True})
            count: int = Field(default=2, ge=0, le=20, description="how many members to open",
                               json_schema_extra={"copied_from_command_text": True})

            @model_validator(mode="after")
            def risk_for_the_whole_record(self, info):
                if self.include_risk and self.person:
                    raise fastworkflow.Refused("the risk is assessed for the whole record, not for one person")
                return self

        plain_utterances = ["who is in the Payroll group", "is Ann Lee in Billing"]


    class PlanGenerator:
        def members(self, input):
            out = [f"{FIND[input.kind]} query={input.name}"]
            out.append(f"list_members filter={input.person}" if input.person else "list_members")
            if input.include_risk:
                out.append("assess_risk")
            for n in range(1, input.count + 1):
                out.append(f"open member {n}")
            return out

- ``Signature`` (required) with a docstring (required; its whitespace is collapsed to one
  line). ``Input`` (a pydantic ``BaseModel``) declares the parameters in order; a planning command
  without parameters leaves it out or empty. A field's type is ``str``, ``bool``, ``int``,
  ``float``, ``list[str]`` or ``Literal[...]`` of text values (an enum), or ``Optional[str] = None``
  (``Optional[int]`` fails the load: an argument the request leaves out is empty text, which a number cannot be).
  ``Field(default=...)`` makes it optional (no default: required; a ``bool`` normally defaults to false, because
  the model sets it true only when the request asks for that part, but a ``default=True`` loads). A ``str``
  default is the value ``steps()`` gets when the request gives none (``default="NOT_FOUND"``, the
  framework's not-given marker, is a load error, ``not_given_default_problems``): an argument the request may leave out defaults to
  ``""``, or is ``Optional[str] = None`` (the ``PlanGenerator``, the hook and the
  validators then get None for it; to the planner it is a str that defaults to empty).
  ``description``
  is what the planner and a ``classifier_confirms`` check read; ``ge``/``le`` bounds are enforced by the
  argument screen (see ``check_arguments``). The planning command options go in ``json_schema_extra``
  (each optional; unknown keys fail the load):

  - ``record_named_in_command_text: True`` (a ``str``): the argument names a record of the request (a person, a
    group, a product, a code). Besides being written in the request it must hold one of the
    request's subjects whole (``split_merge.request_subjects``, with the workflow's identifier
    patterns) or a name-like word (capitalised and not a common word, or holding a digit or
    ``_``) or be a name written in lower case (two or more uncommon words cutting no longer
    run of them: "ann lee"), and cut no subject: "I" of "Before I approve" and "Payroll" of "Payroll_Approver"
    are refused (an optional argument takes its default; a required one fails the call). One
    cut short, mid-word or at a space, at the start or end of exactly one subject becomes
    that subject ("Payroll_Admin" of "Payroll_Administrator", "Ann" of "Ann Lee").
    It replaces the retired ``in_request: "subject"`` and ``subject: True`` spellings.
  - ``classifier_confirms: True`` (a ``bool``, an enum, or a ``str`` with ``record_named_in_command_text``): a classifier
    model decides the argument again from the request alone, when the planner path has one
    (``typed_planning_call``'s ``check_lm``); its answer replaces the planner's when it is at least
    ``VERIFY_THRESHOLD`` sure. It reads the field's ``description``, and an enum's meanings. An enum
    takes ``classifier_confirms: {"meanings": {"group": "a group, people are its members", ...}}`` instead: a
    dict giving each value's meaning, every key one of the enum's values (it replaces the retired ``verify``
    dict and ``means`` spellings).
    For a subject it chooses among the request's subjects (at most ``MAX_SUBJECT_OPTIONS``)
    or ``NO_SUBJECT`` ("none": the default, or a failed call for a required parameter); a
    request naming no subject is not checked.
  - ``copied_from_command_text: True`` (a ``str``, an ``int`` or a ``list[str]``): a ``str`` must appear in the request
    exactly as the request writes it, at word boundaries ("dun" is not written in "dune"); an ``int`` must be a number
    the request writes (digits or the words zero to twenty outside the text of the other
    arguments); a number the request does not write is refused, never replaced by one it does write ("the first
    two" never runs as 3), and an absent value is not checked; a
    ``list[str]`` keeps each item the request writes and drops the others. A marked value given and not written is
    refused, required or optional. Two marked str fields may hold one value only when the request writes it as often.
  - ``copied_from_command_text: {"ignore": [words]}`` (a ``str``; ``record_named_in_command_text`` takes the
    same ``ignore``): words removed from either end of the value before the check (the cleanup the retired
    ``strip`` option did). A field with no check takes its cleanup in code: ``strip_words(text, words)`` in
    a ``field_validator``.
  - No check is implied by a field's type (design 2.46): a field without an option is not
    compared with the request, on either kind of command, and its value stands (a value that is entirely one
    template token, only a pronoun, only a word of the workflow's vocabulary or only punctuation is still refused
    on every ``str``, whatever the options). A check option on a field of a
    type it does not apply to fails the load, naming the option and the type.

- ``PlanGenerator`` (required): its public methods are the planning command's phases, in order, each
  returning its steps as a list of str, one per step (empty ones are dropped); see
  ``fastworkflow.planning`` for the rules. The methods build text only: they must not call a
  backend or read state. A repeated part is a ``for`` loop over ``range()``, bounded by the
  int field's ``le``; steps shared by several planning commands go in a helper module
  (``_commands/_shared/<name>.py``). Lookup tables are plain module-level dicts. A planning command whose one
  phase reads and fills no facts runs flat (one plan); one without parameters whose one
  method takes no ``input`` gives fixed steps.
- ``plain_utterances`` (optional, with or without params): example requests for the
  planning command, written the way people ask for that job. The planning command matcher shows the first
  ``workflow_agent.MATCHER_EXAMPLES`` of them after the description ("e.g. ..."); a planning command
  call also keeps them on its command (``PlanningCommandCommand.plain_utterances``). Write
  them generically: they are in every matcher prompt.
- ``template`` (optional): the template text of the planning command, shown to the planner in the
  planning command call's "expands to:" and used as the text planner's template when a planning command call
  fails. Without it the template is the steps rendered on placeholder inputs
  (``placeholder_template``): every combination of a ``<name>`` for a str (and "" when that
  is its default), true and false for a bool, the default of an int and each value of an
  enum (past ``MAX_GRID`` combinations, the defaults and each one or two changes of them,
  ``input_grid``, and a warning at load), merged into one list in which a step not every
  combination writes says when it is
  written (``(only when person is given)``) and the steps of an enum's values are joined
  (``<find_group or find_unit, by kind> query=<name>``). A repeated part shows only its
  steps for the int's default: declare ``template`` when the planner should see how it
  repeats ("repeat count times (n = 1 to count): open member <n>").
- Phases (with or without params; without, every call has the one plan of the empty
  ``Input``): a ``PlanGenerator`` method that reads an earlier phase's
  facts, or fills its own (its return annotation), makes the call run as phases: each its
  own bounded sub-plan, passing its filled facts model to later phases by parameter name
  (``fastworkflow.planning.planning_phases``; the loader builds its ``Phase`` list,
  ``planning.planned``). At load the phases are built on the placeholder inputs: their
  parameters must bind (``_checked_plan``), and each method's template call must succeed (``PlanningCommandFormatError`` otherwise); every command step
  must name a workflow command and only parameters of its ``Signature.Input``
  (``check_step_commands``: the planning command is skipped at load, logged as an error, and
  ``fastworkflow command preview --check`` fails); a facts field typed with anything but a command's
  Output or Input, a copy (``fact_from`` or ``"from"``) included, is a load error (3.12).
- Rules across fields (optional): ``@model_validator(mode="after")`` on ``Input`` (or a ``field_validator``
  for one field). It is the one place for a rule across fields, and it runs on the arguments the screens
  leave (``check_input``). Its ``info.context`` is ``{"command_text": <the request>, "workflow": <the
  workflow, or None while the template renders>, "command": <the qualified command name>}``, the same for
  both kinds of command. A validator corrects by assigning ``self.<field>`` and returning ``self``, or
  refuses with ``raise fastworkflow.Refused(message, field=...)`` (a plain ``ValueError`` is a refusal with
  no field). Several model validators run in declaration order. Its message says what to fix ("record_a
  and record_b name the same record; record_b is the other one the request names"). A refusal, and a
  correction the validator could not make, is a ``PlanningCommandInputError``: the planner is given the
  message and fills the call once more (``typed_planning_call``); a second refusal falls back. So is a
  required argument the guards of ``check_arguments`` refuse (missing, invalid, not written in the request).
  The validators also run on the placeholder combinations (command text ``""``): one they refuse is left out
  of the template. The option checks (``copied_from_command_text`` and the others) and the field validators run
  before this step.

The file is imported as a command file is (``python_utils.get_module`` under the workflow's
package, so package-relative imports work). A file that fails to import, or whose declaration
is malformed (a ``PlanningCommandFormatError`` naming the file), fails its load: the planning command is logged as
an error and skipped, the workflow's other planning commands still load, and ``fastworkflow command
preview`` fails naming it.

A ``str`` argument marked ``copied_from_command_text`` or ``record_named_in_command_text`` that lies inside a match of
one of the workflow's identifier patterns (``identifier_patterns`` in ``_commands/_workflow.py``, e.g.
``\bacct_[0-9]+``) without being all of it becomes the whole match ("4421" of "acct_4421"); inside several, it is
refused. An unmarked argument is never widened.

The model's ``steps_before`` / ``steps_after`` lose any ``<...>`` placeholder (and the
``key=`` that holds one); a step holding a uid-shaped token the request does not write is
dropped, and so is a step whose commands outside the expansion no open part of the request
asks for, by a word of the command's name, catalog description or plain_utterances
(``filter_extra_steps``, ``command_texts``).

A planning command call's ``str`` and ``int`` arguments first go through the screen a command's
parameters go through (``utils.signatures.screen_parameter_values``: quotes and brackets
removed, bracketed placeholders and out-of-range numbers refused), and a marked str argument is looked up in
the request with Unicode dashes folded to "-" (see ``check_arguments``).
'''
from __future__ import annotations

import decimal
import difflib
import enum
import functools
import inspect
import json
import math
import os
import re
import sys
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import UnionType
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union, get_args, get_origin

from pydantic import BaseModel

from fastworkflow.observability import tracing
from fastworkflow.core.command_context_model import CommandContextModel
from fastworkflow.core.command_directory import CommandDirectory, compute_commands_source_fingerprint, raised_in_helper
from fastworkflow.parameters.field_options import (
    CLASSIFIER_CONFIRMS, CLASSIFIER_DECIDES, COMMAND_TEXT_AFTER_COMMAND_NAME,
    COPIED_FROM_COMMAND_TEXT, EXAMPLES_BY, FROM_COMMAND_OUTPUT, OPTION_KEYS, RECORD_NAMED_IN_COMMAND_TEXT, UI)
from fastworkflow.parameters.parameter_clarification import Extraction
from fastworkflow.planning.planning_command_json_schema_internal import (
    HAND_WRITTEN_AVAILABLE_FROM_KEY as _HAND_WRITTEN_AVAILABLE_FROM_KEY,
    PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS as _PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS,
)
from fastworkflow.parameters.workflow_vocabulary import (
    WorkflowVocabulary, load_workflow_vocabulary)
from fastworkflow.observation_offloading.finish_check import (
    KIND_COMMAND,
    KIND_CONDITIONAL,
    KIND_SKIPPED,
    _CONDITION_RE,
    _NEGATION_RE,
    _outside_quoted_values,
    parse_plan,
    step_chain,
)

#: Like ``finish_check._CONDITION_RE``, but bare ``only`` in a note is not a condition word.
_NOTE_CONDITION_RE = re.compile(
    r"\b(?:if|only\s+(?:if|when)|when|whenever|unless|otherwise|optional(?:ly)?|in case|as needed|"
    r"where (?:needed|applicable)|else)\b", re.IGNORECASE)
from fastworkflow import planning as planning_api
from fastworkflow.planning.planning_phases import (
    FULL_LIST_PHRASES,
    PhaseError,
    _PICK_NAMED_RE,
    active_phases,
    is_flat_plan,
    phase_steps,
    placeholder_steps,
)
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger
from fastworkflow.core.workflow_inheritance_model import (
    WorkflowInheritanceModel,
    WorkflowInheritanceModelError,
)

from fastworkflow.parameters import check as one_guard  # noqa: E402 - the one guard module (bd fix-qoyz step 3)
from fastworkflow.parameters import ask  # noqa: E402 - the one question builder (bd fix-qoyz step 4)
from fastworkflow.parameters import rules  # noqa: E402 - the shared text helpers (bd fix-qoyz step 1)

#: A ``classifier_confirms`` check replaces the planner's argument only at or above this probability.
VERIFY_THRESHOLD = 0.65
#: The value a ``classifier_confirms`` check of a ``record_named_in_command_text`` parameter gives when the request names none.
NO_SUBJECT = "none"
#: A ``classifier_confirms`` check of a record-naming parameter chooses among at most this many of the request's subjects.
MAX_SUBJECT_OPTIONS = 8
#: Names a parameter may not take (they are the splitter's own prompt fields).
RESERVED_NAMES = frozenset({"user_query", "available_commands", "steps_before", "steps_after",
                            "reasoning"})

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_COMMAND_LINE = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*$", re.M)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A 16+ hex-digit run with a digit in it, or a dashed uuid: an id only the run can know.
_UID = re.compile(r"\b(?=[0-9a-f-]*\d)(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,})\b",
                  re.I)
_ARG_PLACEHOLDER = re.compile(r"\s*\b[A-Za-z_][\w.]*\s*[=:]\s*<[^<>]*>")
_BARE_PLACEHOLDER = re.compile(r"\s*<[^<>]*>")


class PlanningCommandFormatError(ValueError):
    """A malformed planning command file: its ``Signature`` or an ``Input`` field."""


class PlanningCommandCallError(ValueError):
    """A planning command call that cannot be used (bad argument, failed guard); the planner falls back."""


class PlanningCommandInputError(PlanningCommandCallError):
    """The arguments are refused: a field of the planning command's ``Input`` breaks its type or a field
    validator, a model validator of the ``Input`` raised ``fastworkflow.Refused``
    (its message), or a guard of
    ``check_arguments`` refused one of the planner's arguments (missing, not a valid value, not
    written in the request). The message says what to fix; ``typed_planning_call`` gives it to the
    planner for one more typed call. ``fields`` names the arguments a ``Refused(field=...)`` named:
    the question put to the user then asks for those arguments."""

    def __init__(self, message: str = "", fields=(), gap: "one_guard.Gap | None" = None):
        super().__init__(message)
        self.fields: tuple[str, ...] = tuple(fields)
        #: The refusal of ONE argument as data (``parameters.check.Gap``: name, state, code, why, value), when a guard
        #: or a rule of the roles refused it. Everything that needs the argument's name or the reason reads it from
        #: here, never from the message text.
        self.gap: "one_guard.Gap | None" = gap
        #: Every argument the call is missing or has refused (``Extraction``), when the raiser worked them out;
        #: ``check_arguments`` itself stops at the first one.
        self.gaps: "Extraction | None" = None
        #: The rule that refused (``parameters.check.Gap.code``) when a guard raised it, else empty. ``missing``
        #: says a required argument is simply not there: nothing the model produced was refused.
        self.code: str = ""
        #: The arguments as the guards held them when they refused (``check_arguments``), else empty.
        self.values: dict = {}


#: The refusals of a free-text value that ``refusal_question`` asks for without quoting the value:
#: a template token or an invented name, a pronoun the request does not tie to one subject, a
#: word of the workflow's own vocabulary (a kind of record, a field name, a declared choice).
PLACEHOLDER_WHY = one_guard.PLACEHOLDER_WHY
PRONOUN_WHY = "is a pronoun the request does not tie to a single subject"
def _role_fits(request: str, value: str, p: "Param") -> bool:
    """Whether the request writes *p*'s role just before *value*: the input's name, as words, among
    the four words before the value's first occurrence in the request."""
    folded = rules.fold(request).lower()
    at = folded.find(rules.fold(value).lower())
    if at < 0:
        return False
    words = re.findall(r"[a-z0-9_]+", folded[:at].replace("_", " "))[-4:]
    role = p.name.lower().replace("_", " ")
    return re.search(rf"\b{re.escape(role)}s?\b", " ".join(words)) is not None


def _planning_commands_workflow_root(path: str) -> str | None:
    """The workflow folder of a planning command file: the parent of the ``_commands`` folder it sits in."""
    for parent in Path(path).parents:
        if parent.name == "_commands":
            return str(parent.parent)
    return None


def _vocabulary_for(command: "PlanningCommandCommand") -> WorkflowVocabulary | None:
    """The vocabulary of the workflow the planning command belongs to (the stored file when fresh, else
    derived in memory); None when the planning command has no workflow folder or the derivation fails."""
    root = _planning_commands_workflow_root(command.path) if command.path else None
    if root is None:
        return None
    try:
        return load_workflow_vocabulary(root)
    except Exception:  # noqa: BLE001 - the workflow's own code; the screens are skipped, not the call
        logger.warning(f"could not derive the vocabulary of {root}", exc_info=True)
        return None


def missing_argument_question(command: PlanningCommandCommand, name: str) -> str:
    """The question put to the user for a required argument the request does not give.

    The same words for every route that finds the argument missing: a planning command call's refusal
    (``refusal_question``), whatever its check found.
    """
    return ask.word(form=ask.FORM_ARGUMENT, command=command, name=name).text


def refusal_question(command: PlanningCommandCommand, refusal: PlanningCommandInputError) -> str:
    """The question put to the user when a refused planning command call cannot be filled from the request.

    It names the recipe's input and says what is wrong with it, in plain words: the request
    does not say it, the value the planner took is not in the request, or it is not one of the
    allowed values. A refusal that names no input (a rule the planning command's own hook or Input raised)
    is put in its own words. When the call is missing or has refused several arguments (``refusal.gaps``), one
    question names them all (``gaps_question``). The words are ``parameters.ask.word``'s.
    """
    return ask.word(form=ask.FORM_REFUSAL, command=command, refusal=refusal).text


def argument_gaps(command: PlanningCommandCommand, raw: dict, request: str, verdicts: dict | None = None,
                  workflow=None) -> Extraction:
    """Every required argument *raw* leaves missing or refused (``check_arguments`` stops at the first): the guards run
    again with each gap's parameter left out, until they pass or refuse something that names no argument. The values
    of the other arguments, as the guards checked them, are kept in ``values`` (the ones the request gave, not defaults).
    No model call.

    On a tie (``one_guard.validators_first``: the duplicate rule's refusal is the ONLY gap, every other value present)
    the ``Input``'s validators run on the same values (``check_input``, *workflow* in their context), and when they
    refuse, their refusal is the gap (``_validators_on_tie``). With any other gap they are not run: validators on
    incomplete values are not reliable, and the framework's wording stands."""
    return _gaps_and_checked(command, raw, request, verdicts, workflow)[0]


def checked_around_gaps(command: PlanningCommandCommand, raw: dict, request: str,
                        verdicts: dict | None = None) -> tuple[dict, list[str]]:
    """``(args, notes)`` of the guards run with every gap's parameter left out (``argument_gaps``' last run): what the
    guards correct in the arguments they do not refuse (a sure verdict, a widened id, the request's spelling, a
    member's declared spelling), so a refusal of one argument does not drop the corrections of the others. Empty
    when a refusal names no argument. No model call."""
    checked = _gaps_and_checked(command, raw, request, verdicts)[1]
    return (dict(checked.values), list(checked.notes)) if checked is not None else ({}, [])


def _gaps_and_checked(command: PlanningCommandCommand, raw: dict, request: str, verdicts: dict | None,
                      workflow=None):
    """``(Extraction, Checked)``: the gaps of ``argument_gaps``, and the guards' result once they pass with the gaps
    left out (None when they stop at a refusal that names no argument)."""
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    screened: list[tuple[str, str]] = []
    work = command
    args: dict = {}
    passed = None
    tie = None        # the values the duplicate rule refused, when its refusal was the first gap
    while True:
        checked = one_guard.check_plan(work, raw, request, verdicts)
        stop = checked.stop
        if stop is None:
            args = checked.values
            passed = checked
            break
        if stop.call:     # the check finds no such argument in the request at all: it names no argument to ask for
            break
        if not missing and not invalid and one_guard.validators_first(stop):
            tie = dict(checked.values)
        (missing.append(stop.name) if stop.state == one_guard.MISSING else invalid.append((stop.name, stop.why)))
        if stop.screened:
            screened.append((stop.name, stop.screened))
        work = replace(work, params=tuple(p for p in work.params if p.name != stop.name))
    # the duplicate rule's notes: a field it left not given is not a value the checks stand behind
    left_empty = ask.left_empty_lines(checked.notes)
    kept = {name: value for name, value in args.items() if not _empty(raw.get(name)) and name not in missing
            and name not in {n for n, _ in invalid} and name not in left_empty}
    gaps = Extraction(missing=tuple(missing), invalid=tuple(invalid), values=kept, screened=tuple(screened),
                      left_empty=tuple(left_empty.values()))
    if tie is not None and passed is not None and len(gaps.names) == 1:
        gaps = _validators_on_tie(command, gaps, tie, raw, request, workflow, left_empty)
    return gaps, passed


def _validators_on_tie(command: PlanningCommandCommand, gaps: Extraction, tie: dict, raw: dict, request: str,
                       workflow, left_empty: dict) -> Extraction:
    """*gaps* (the duplicate rule's refusal alone) on a tie: the ``Input``'s validators run on *tie* (the values the
    rule refused, the duplicated value still in both fields), as ``checked_arguments`` runs them (bd fix-qjne.38). A
    refusal that names arguments is the gap: those arguments are refused with the validator's message (``invalid``,
    with the message as ``reason``, which the question and the review line show) and the other values are kept as the
    guards held them; the validators' rewrites are never kept. A refusal that names no argument is what a call-level
    refusal is here (``stop.call``): nothing to ask and no value kept, and the run reports it. When they pass, or a
    validator raises, *gaps* stands."""
    if command.input_model is None:
        return gaps
    try:
        check_input(command, tie, request, workflow)
    except PlanningCommandInputError as theirs:
        params = {p.name for p in command.params}
        fields = tuple(name for name in dict.fromkeys(theirs.fields) if name in params)
        if not fields:
            return Extraction(left_empty=gaps.left_empty)
        message = getattr(theirs, "refusal", None) or str(theirs)
        values = {name: value for name, value in tie.items() if not _empty(raw.get(name)) and name not in fields
                  and name not in left_empty}
        return Extraction(invalid=tuple((name, message) for name in fields), values=values, reason=message,
                          left_empty=gaps.left_empty)
    except PlanningCommandCallError:   # a validator that raised is no refusal: the duplicate rule's stands
        pass
    return gaps


def refusal_gaps(command: PlanningCommandCommand, refusal: PlanningCommandInputError) -> Extraction:
    """The gaps a refusal stands for: its own ``gaps`` when the raiser worked them out, else the one argument it names
    (a hook's ``Refused(field=...)`` names invalid ones)."""
    if refusal.gaps is not None:
        return refusal.gaps
    if refusal.fields:
        why = getattr(refusal, "refusal", None) or str(refusal)
        return Extraction(invalid=tuple((name, why) for name in dict.fromkeys(refusal.fields)))
    gap = refusal.gap
    if gap is None or gap.name not in {p.name for p in command.params}:
        return Extraction()
    if gap.state == one_guard.MISSING:
        return Extraction(missing=(gap.name,), screened=((gap.name, gap.screened),) if gap.screened else ())
    return Extraction(invalid=((gap.name, gap.why),))


def gaps_question(command: PlanningCommandCommand, gaps: Extraction) -> str:
    """One question for every argument a call lacks: the ones the request does not give, then the ones refused (why)."""
    return ask.word(gaps, form=ask.FORM_GAPS, command=command).text


def refusal_names_an_input(command: PlanningCommandCommand, refusal: PlanningCommandInputError) -> bool:
    """Whether a refused planning command call is something to ask the user: the refusal names one of
    the command's inputs (``Refused(..., field=...)`` from the hook, a missing argument, or a guard's
    refusal of one argument's value).

    A refusal that names no input (``Refused(message)`` without ``field``, or an ``Input`` rule) says
    the call does not fit the request: "owner is read only with kind='application'" is the choice of
    command, not a value the user can supply. The turn then plans the request with the text planner,
    as an unattended turn does, instead of putting the rule to the user.
    """
    if refusal.fields or (refusal.gaps is not None and refusal.gaps.names):
        return True
    return refusal.gap is not None and any(p.name == refusal.gap.name for p in command.params)


def planning_refusal_parameter_request(
    command: PlanningCommandCommand, refusal: PlanningCommandInputError,
) -> Optional[dict]:
    """The refused inputs of a planning command call as data, for a client that renders a form:
    ``{"command", "missing", "invalid", "values"}``, or None when the refusal names no input of the command.

    ``missing`` and ``invalid`` list every argument the call lacks (``refusal.gaps`` when the raiser worked them out,
    else the one the refusal names); ``values`` is what the call already has for the others (kept, not asked again).
    The shapes ``refusal_question`` reads: ``Refused(field=...)`` from the hook (``fields``), a missing argument, and a
    guard's refusal of one argument (a placeholder, pronoun or kind word reads as missing; any other reason as invalid).
    """
    return ask.word(form=ask.FORM_REFUSAL, command=command, refusal=refusal).request


# ---------------------------------------------------------------- the parsed planning command

@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "bool" | "int" | "float" | "list[str]" | "enum"
    values: tuple[str, ...] = ()
    #: A list[str] default is a tuple (a Param is hashable); the others are the value.
    default: object = None
    required: bool = False
    desc: str = ""
    verify: bool = False
    #: ``classifier_decides`` (a bool): the classifier alone decides the field from the request; the typed
    #: call and the parameter extractor never fill it, and it keeps its default when the classifier is
    #: not sure or there is no JEV_API_KEY (also sets ``verify``, internally).
    alone: bool = False
    #: A classifier_confirms check's {value: meaning} (an enum), in the order of ``values``.
    means: tuple[tuple[str, str], ...] = ()
    #: ``copied_from_command_text`` (True) or ``record_named_in_command_text`` (which also sets ``subject``).
    in_request: bool = False
    subject: bool = False
    #: ``ignore`` words of the check option (a str): words removed from either end of the value before the check.
    ignore: tuple[str, ...] = ()

    def type_text(self) -> str:
        return "|".join(self.values) if self.type == "enum" else self.type


@dataclass(frozen=True)
class PlanningCommandCommand:
    """A planning command called with typed arguments: its signature and its ``steps()``. Every planning command
    file has one; a planning command without parameters or ``plan()`` (``fixed_steps``) has no
    parameters to fill and its one phase reads nothing."""

    name: str
    description: str
    params: tuple[Param, ...]
    path: str = field(default="", compare=False)
    #: The workflow's identifier patterns (``workflow_agent.identifier_patterns``), set at load.
    identifiers: tuple = field(default=(), compare=False)
    #: The planning command's pydantic ``Input``. Planning command-call arguments are screened through it like a
    #: command's parameters.
    input_model: type | None = field(default=None, compare=False)
    #: The planning command file's ``steps(input)``.
    steps_fn: Callable | None = field(default=None, compare=False)
    #: The template text (``Signature.template``, or ``steps()`` rendered on placeholder
    #: inputs).
    template: tuple[str, ...] = field(default=(), compare=False)
    #: ``Signature.plain_utterances``: example requests for the planning command.
    plain_utterances: tuple[str, ...] = field(default=(), compare=False)
    #: The planning command file's ``plan(input) -> list[Phase]`` (``fastworkflow.planning.planning_phases``), or None.
    plan_fn: Callable | None = field(default=None, compare=False)
    #: The phases' ``output`` models seen when ``plan()`` ran on the placeholder inputs at load.
    phase_outputs: tuple[type, ...] = field(default=(), compare=False)
    #: The planning command file's ``PlanGenerator`` class (``command_from_module`` only).
    plan_cls: type | None = field(default=None, compare=False)
    #: ``Signature.Output`` when the planning command declares one.
    output_model: type | None = field(default=None, compare=False)
    #: Whether ``PlanGenerator`` defines or inherits ``__result__``.
    uses_result: bool = field(default=False, compare=False)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def signature_text(self) -> str:
        """``name(p: type = default, ...)`` with the description, parameter docs and steps."""
        parts = []
        for p in self.params:
            s = f"{p.name}: {p.type_text()}"
            if not p.required:
                s += f" = {json.dumps(_json_value(p))}"
            parts.append(s)
        lines = [f"{self.name}({', '.join(parts)})", f"  {self.description}"]
        lines += [f"  - {p.name}: {p.desc}" for p in self.params]
        lines.append("  expands to:")
        lines += [f"    {i}. {s}" for i, s in enumerate(self.template, 1)]
        return "\n".join(lines)


# ---------------------------------------------------------------- parameters

def _check_keys(block: dict, allowed: set[str], what: str, where: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise PlanningCommandFormatError(f"{where}: {what} has unknown key(s) {', '.join(unknown)}")


#: The keys a planning command file's ``Input`` field may set in ``json_schema_extra``: the author options
#: (``field_options.OPTION_KEYS``), and ``available_from``, which the resolver writes (a hand-written one is a
#: load error there, ``resolve_field_sources``).
PLANNING_COMMAND_FIELD_KEYS = frozenset(OPTION_KEYS) | {"available_from"}

#: The retired spellings (spec 2.0): using one is a load error that names the option replacing it.
RETIRED_OPTIONS = ("in_request", "verify", "positional", "from", "subject", "means", "strip")

#: Keys of ``json_schema_extra`` removed from every ``Input`` and ``Output`` field of either kind of command (bd fix-qjne.3):
#: the allowed values are the type (``Literal``), nothing read ``used_by``, and a pattern is ``Field(pattern=...)``.
REMOVED_FIELD_KEYS = ("enum", "used_by", "pattern")

#: The options a ``Param`` carries (the check options of spec 2.1 to 2.4), read by ``option_params``.
CHECK_OPTIONS = (CLASSIFIER_CONFIRMS, CLASSIFIER_DECIDES, COPIED_FROM_COMMAND_TEXT, RECORD_NAMED_IN_COMMAND_TEXT)

_CHECK_KEYS_INTERNAL = frozenset(_PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS)


def _shown(value) -> str:
    """A value as an author writes it: True or False, JSON for text, lists and dicts."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value)


def _option_text(key: str, arg) -> str:
    """``"key": arg`` as an author writes it in ``json_schema_extra``."""
    return f'"{key}": {_shown(arg)}'


def _retired_replacement(key: str, value) -> str:
    """The option that replaces a retired spelling, as an author writes it (spec 2.0)."""
    if key == "in_request":
        if value == "subject":
            return _option_text(RECORD_NAMED_IN_COMMAND_TEXT, True)
        if isinstance(value, dict):
            name = RECORD_NAMED_IN_COMMAND_TEXT if value.get("subject") is True else COPIED_FROM_COMMAND_TEXT
            return _option_text(name, {"ignore": value["ignore"]} if "ignore" in value else True)
        if value is True:
            return _option_text(COPIED_FROM_COMMAND_TEXT, True)
        return "leave the key out"
    if key == "verify":
        if value == "alone":
            return _option_text(CLASSIFIER_DECIDES, True)
        if isinstance(value, dict):
            return _option_text(CLASSIFIER_CONFIRMS, {"meanings": value})
        if value is True:
            return _option_text(CLASSIFIER_CONFIRMS, True)
        return "leave the key out"
    if key == "positional":
        return _option_text(COMMAND_TEXT_AFTER_COMMAND_NAME, True)
    if key == "from":
        return _option_text(FROM_COMMAND_OUTPUT, value)
    if key == "subject":
        return _option_text(RECORD_NAMED_IN_COMMAND_TEXT, True)
    if key == "means":
        return _option_text(CLASSIFIER_CONFIRMS, {"meanings": value})
    return _option_text(COPIED_FROM_COMMAND_TEXT, {"ignore": value})   # strip


def _removed_key_advice(key: str, value) -> str:
    """What to write instead of a removed field key of ``REMOVED_FIELD_KEYS``."""
    if key == "used_by":
        return "delete it (nothing reads it)"
    if key == "pattern":
        return f"write Field(pattern={_shown(value)})"
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return f"write the field's type as Literal[{', '.join(_shown(v) for v in value)}]"
    return "write the field's type as Literal[...]"


def check_removed_field_keys(where: str, name: str, extra) -> None:
    """The load error of a key of ``REMOVED_FIELD_KEYS`` among a field's ``json_schema_extra`` *extra*."""
    for key in REMOVED_FIELD_KEYS:
        if key in extra:
            raise _field_error(where, name, f"{key}: {_shown(extra[key])} is removed: {_removed_key_advice(key, extra[key])}")


def _retired_option_error(where: str, name: str, extra) -> None:
    """The load error of a retired spelling among a field's ``json_schema_extra`` *extra*, naming its replacement."""
    check_removed_field_keys(where, name, extra)
    if "db_lookup" in extra:
        from fastworkflow.parameters.signatures import _RETIRED_DB_LOOKUP

        raise _field_error(where, name, _RETIRED_DB_LOOKUP)
    for key in RETIRED_OPTIONS:
        if key in extra:
            value = extra[key]
            replacement = _retired_replacement(key, value)
            advice = replacement if replacement == "leave the key out" else f"write {replacement}"
            raise PlanningCommandFormatError(f"{where}: field {name!r}: {key}: {_shown(value)} is removed: {advice}")


def _field_error(where: str, name: str, message: str) -> PlanningCommandFormatError:
    return PlanningCommandFormatError(f"{where}: field {name!r}: {message}")


def _closest(word: str, options) -> str | None:
    """The option *word* most resembles, or None (difflib)."""
    matches = difflib.get_close_matches(str(word), list(options), n=1, cutoff=0.6)
    return matches[0] if matches else None


#: JSON Schema annotation keywords an execution command's ``Input`` field, any ``Output`` field and a facts field may
#: carry in ``json_schema_extra`` (a planning command's ``Input`` takes the option keys only: ``check_field_keys``).
#: Never read as the misspelling of an option key (``examples`` resembles ``examples_by``).
_SCHEMA_ANNOTATION_KEYS = frozenset({"title", "description", "default", "examples", "deprecated", "readOnly",
                                     "writeOnly", "format"})

#: The option key an ``Output`` field of either kind of command may set (display hints for a client). The resolver's
#: own keys (``from_command_output``, ``available_from``) have their own load errors there (``resolve_signature_field_sources``).
OUTPUT_FIELD_OPTION_KEYS = (UI,)


def _valid_keys_text(options, *, annotations: bool) -> str:
    text = ", ".join(options)
    return f"{text}, and the JSON Schema annotations {', '.join(sorted(_SCHEMA_ANNOTATION_KEYS))}" if annotations else text


def check_field_keys(name: str, extra, where: str, *, planning: bool) -> None:
    """The keys of one ``Input`` field's ``json_schema_extra`` (spec 2.0), one rule stated here (bd fix-ftdl.8.22,
    fix-ftdl.9.4): a retired spelling and an unknown key fail the load on either kind of command. Besides the option
    keys, an execution command's ``Input`` takes the JSON Schema annotation keywords (``_SCHEMA_ANNOTATION_KEYS``),
    because its parameter-example generator reads ``description`` and ``examples`` there
    (``utils/generate_param_examples.py``, a fallback to ``Field(examples=...)``); nothing reads them on a planning
    command's ``Input``, which takes the option keys only. The message names the key, the closest option when one is
    close, and the valid keys."""
    _retired_option_error(where, name, extra)
    for key in extra:
        if key in PLANNING_COMMAND_FIELD_KEYS or key in _CHECK_KEYS_INTERNAL:
            continue
        if not planning and key in _SCHEMA_ANNOTATION_KEYS:
            continue
        close = _closest(key, OPTION_KEYS)
        valid = f"its options are {_valid_keys_text(OPTION_KEYS, annotations=not planning)}"
        hint = f"did you mean {close!r}? ({valid})" if close else valid
        raise _field_error(where, name, f"unknown option {key!r}; {hint}")


def _not_a_dict(where: str, name: str) -> PlanningCommandFormatError:
    """The load error of a field whose ``json_schema_extra`` is not a dict (a function is never read by the checks)."""
    return _field_error(where, name, "json_schema_extra must be a dict of the field's options; a function or another "
                                     "value is not read, so none of its keys would be checked")


def _facts_field_key_problem(model: type, name: str, extra, *, typed: bool) -> Optional[str]:
    """Why one facts model field's ``json_schema_extra`` fails the load, or None. A facts field is typed with a
    command's Output and takes no option: an Input or Output option, and any other key but a JSON Schema annotation,
    is refused. ``from_command_output`` has its own message; so does the retired copy key ``"from"`` (what the removed
    ``fact_from`` wrote) on a field that is not *typed* (``typed_phase_facts``, whose message says copies are not
    read), and on a *typed* field it is refused here."""
    from fastworkflow.planning.typed_phase_facts import COPIES_NOT_READ

    where = f"{model.__name__}"
    if extra is None:
        return None
    if not isinstance(extra, Mapping):
        return str(_not_a_dict(where, name))
    for key in extra:
        if key == "from" and typed:
            return str(_field_error(where, name, f"from is removed: {COPIES_NOT_READ}; the field's type, a command's "
                                                 "Output, says what it holds: leave the key out"))
        if key in _SCHEMA_ANNOTATION_KEYS or key in (FROM_COMMAND_OUTPUT, "from"):
            continue
        if key in OPTION_KEYS or key in OUTPUT_FIELD_OPTION_KEYS:
            return str(_field_error(where, name, f"{key} is an option of a command's Input or Output field; a facts "
                                                 "field takes none (its type, a command's Output, says what it holds)"))
        return str(_field_error(where, name, f"unknown key {key!r}; a facts field takes no option (only the JSON "
                                             f"Schema annotations {', '.join(sorted(_SCHEMA_ANNOTATION_KEYS))})"))
    return None


def check_output_field_keys(name: str, extra, where: str) -> None:
    """The keys of one ``Output`` field's ``json_schema_extra``, on either kind of command (bd fix-ftdl.8.22): a removed
    or retired spelling names its replacement, an ``Input`` option says it is one, and any other key that is not ``ui``
    or a JSON Schema annotation fails the load. ``from_command_output`` and ``available_from`` are left to their own
    messages (``resolve_signature_field_sources``)."""
    _retired_option_error(where, name, extra)
    for key in extra:
        if key in (*OUTPUT_FIELD_OPTION_KEYS, FROM_COMMAND_OUTPUT, "available_from") or key in _SCHEMA_ANNOTATION_KEYS:
            continue
        valid = f"an Output field takes {_valid_keys_text(OUTPUT_FIELD_OPTION_KEYS, annotations=True)}"
        if key in OPTION_KEYS:
            raise _field_error(where, name, f"{key} is an option of an Input field; {valid}")
        close = _closest(key, OUTPUT_FIELD_OPTION_KEYS)
        hint = f"did you mean {close!r}? ({valid})" if close else valid
        raise _field_error(where, name, f"unknown option {key!r}; {hint}")


def _parse_param(name: str, spec, where: str) -> Param:
    if name in RESERVED_NAMES:
        raise PlanningCommandFormatError(
            f"{where}: field {name!r} is reserved for the framework's own planner inputs "
            f"({', '.join(sorted(RESERVED_NAMES))}); rename the field")
    if not re.fullmatch(_NAME, name):
        raise PlanningCommandFormatError(f"{where}: bad parameter name {name!r}")
    if not isinstance(spec, dict):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} must be a block with a type")
    _check_keys(spec, {"type", "default", "desc", "verify", "in_request", "ignore", "optional"},
                f"parameter {name!r}", where)
    type_text = spec.get("type")
    if not isinstance(type_text, str):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} needs a type")
    values: tuple[str, ...] = ()
    m = re.fullmatch(r"enum\[(.*)\]", type_text.strip())
    if m:
        values = tuple(v.strip() for v in m.group(1).split(","))
        if not values or any(not re.fullmatch(r"[\w-]+", v) for v in values) or len(set(values)) != len(values):
            raise PlanningCommandFormatError(f"{where}: parameter {name!r} has a bad enum {type_text!r}")
        kind = "enum"
    elif type_text.strip() in ("str", "bool", "int", "float", "list[str]"):
        kind = type_text.strip()
    else:
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} has unknown type {type_text!r} "
                               "(str, bool, int, float, list[str] or enum[...])")
    desc = spec.get("desc", "")
    if not isinstance(desc, str):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} desc must be text")
    raw_default = spec.get("default")
    if isinstance(raw_default, dict):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} default must be a value")
    required = raw_default is None and kind != "bool" and spec.get("optional") is not True
    default: object = None
    if kind == "bool":
        if raw_default not in (None, "true", "false"):
            raise PlanningCommandFormatError(f"{where}: bool parameter {name!r} default must be true or false")
        default = raw_default == "true"
    elif raw_default is not None:
        if kind == "int":
            if not re.fullmatch(r"-?\d+", raw_default):
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not an int")
            default = int(raw_default)
        elif kind == "float":
            default = _float_text(raw_default)
            if default is None:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not a number")
        elif kind == "list[str]":
            default = _list_text(raw_default)
            if default is None:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not a list of text")
        elif kind == "enum":
            if raw_default not in values:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not one of its values")
            default = raw_default
        else:
            default = raw_default
    # copied_from_command_text (a str, an int or a list[str]) or record_named_in_command_text (a str): the spec's in_request
    in_request_text = spec.get("in_request", "false")
    subject = in_request_text == "subject"
    copy_option = RECORD_NAMED_IN_COMMAND_TEXT if subject else COPIED_FROM_COMMAND_TEXT
    if subject:
        if kind != "str":
            raise option_error(where, RECORD_NAMED_IN_COMMAND_TEXT, name, kind, "a str")
    elif in_request_text not in ("true", "false"):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} has a bad copy option")
    in_request = subject or in_request_text == "true"
    ignore: tuple[str, ...] = ()
    if "ignore" in spec:
        if kind != "str":
            raise option_error(where, f"{copy_option} ignore", name, kind, "a str")
        ignore = tuple(spec["ignore"])
    if in_request and not subject and kind not in ("str", "int", "list[str]"):
        raise option_error(where, COPIED_FROM_COMMAND_TEXT, name, kind, "a str, an int or a list[str]")
    # classifier_confirms: True (a bool, an enum, or a record-named str) or a {value: meaning} dict on an enum;
    # classifier_decides (a bool)
    verify_spec = spec.get("verify", "false")
    means: tuple[tuple[str, str], ...] = ()
    alone = False
    if isinstance(verify_spec, dict):
        block = verify_spec
        if kind != "enum":
            raise option_error(where, f"{CLASSIFIER_CONFIRMS} (a dict of meanings)", name, kind, "an enum")
        if not block:
            raise PlanningCommandFormatError(f"{where}: field {name!r}: {CLASSIFIER_CONFIRMS} meanings must be a "
                                             "dict of its enum values' meanings")
        for key, text in block.items():
            if key not in values or not isinstance(text, str):
                raise PlanningCommandFormatError(f"{where}: field {name!r}: {CLASSIFIER_CONFIRMS} meanings: {key!r} is "
                                                 "not a value of it, or not text")
        means = tuple((v, block[v]) for v in values if v in block)
        verify = True
    elif verify_spec == "alone":
        if kind != "bool":
            raise option_error(where, CLASSIFIER_DECIDES, name, kind, "a bool")
        verify = alone = True
    else:
        verify = _flag({"verify": verify_spec}, "verify", name, where)
        if verify and kind not in ("bool", "enum") and not subject:
            raise option_error(where, CLASSIFIER_CONFIRMS, name, kind,
                               f"a bool, an enum or a str with {RECORD_NAMED_IN_COMMAND_TEXT}")
    return Param(name=name, type=kind, values=values, default=default, required=required,
                 desc=desc, verify=verify, alone=alone, means=means, in_request=in_request,
                 subject=subject, ignore=ignore)


def option_error(where: str, option: str, name: str, kind: str, takes: str) -> PlanningCommandFormatError:
    """The load error of a check option on a field of a type it does not apply to (design 2.46)."""
    return PlanningCommandFormatError(f"{where}: option {option} does not apply to field {name!r} of type "
                                      f"{kind} (it takes {takes})")


def _float_text(text) -> float | None:
    """*text* as a finite float, or None."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _list_text(value) -> tuple[str, ...] | None:
    """A list[str] default: a list, or the list written as JSON text; a tuple of its text, or None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        return None
    return tuple(value)


def _flag(spec: dict, key: str, name: str, where: str) -> bool:
    value = spec.get(key, "false")
    if value not in ("true", "false"):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} {key} must be true or false")
    return value == "true"


# ---------------------------------------------------------------- planning command files (_commands/<name>.py)

def _option_kind(annotation) -> str:
    """The coarse type of a field's annotation for an option check: str, int, bool, float, list[str], enum or other."""
    if rules.is_union(annotation) and type(None) in get_args(annotation):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        annotation = inner[0] if len(inner) == 1 else annotation
    if annotation in (str, int, bool, float):
        return annotation.__name__
    if annotation == list[str]:
        return "list[str]"
    if get_origin(annotation) is Literal:
        return "enum"
    return "other"


def _spec_from_field(name: str, info, where: str, *, planning: bool = False) -> dict:
    """The spec of one ``Input`` field (its type, default, description and planning command options, as
    text), for ``_parse_param``."""
    from pydantic_core import PydanticUndefined

    annotation = info.annotation
    optional = rules.is_union(annotation) and type(None) in get_args(annotation)
    optional_number = False
    if optional:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) != 1 or info.default is not None or inner[0] not in (str, int):
            raise PlanningCommandFormatError(f"{where}: field {name!r}: Optional is for an argument the command text "
                                   "may leave out, Optional[str] = None (or Optional[int] = None on an execution "
                                   "command); give another type a default")
        if inner[0] is int:
            if planning:
                raise PlanningCommandFormatError(
                    f"{where}: field {name!r}: a planning command's arguments reach it as text, and an argument the "
                    "request leaves out is empty text, which a number cannot be; give the int a default, or make it "
                    "Optional[str] = None (an execution command may declare Optional[int] = None)")
            optional_number = True
        annotation = inner[0]
    if annotation in (str, bool, int, float):
        type_text = annotation.__name__
    elif annotation == list[str]:
        type_text = "list[str]"
    elif get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not values or not all(isinstance(v, str) for v in values):
            raise PlanningCommandFormatError(f"{where}: field {name!r}: a Literal of text values only")
        type_text = f"enum[{', '.join(values)}]"
    else:
        raise PlanningCommandFormatError(f"{where}: field {name!r} has type {python_utils.type_text(annotation)}; a planning command field is "
                               "str, bool, int, float, list[str] or Literal[...] of text values")
    spec: dict = {"type": type_text}
    if info.description:
        spec["desc"] = info.description
    default: object = PydanticUndefined
    if info.default_factory is not None:
        if type_text != "list[str]" or info.default_factory is not list:
            raise PlanningCommandFormatError(f"{where}: field {name!r}: give a plain default, not a default_factory")
        default = []
    elif info.default is not PydanticUndefined:
        default = info.default
    if optional_number:
        spec["optional"] = True
    elif optional:
        spec["default"] = ""
    elif default is not PydanticUndefined:
        if default is None or isinstance(default, dict) or (isinstance(default, (list, tuple)) and type_text != "list[str]"):
            raise PlanningCommandFormatError(f"{where}: field {name!r} default must be a value of its type "
                                   "(leave the default out for a required parameter)")
        if type_text == "list[str]":
            if _list_text(default) is None:
                raise PlanningCommandFormatError(f"{where}: field {name!r} default must be a list of text")
            spec["default"] = json.dumps(list(default))
        else:
            spec["default"] = ("true" if default else "false") if isinstance(default, bool) else str(default)
    extra = info.json_schema_extra
    if extra is None:
        return spec
    if not isinstance(extra, dict):
        raise PlanningCommandFormatError(f"{where}: field {name!r} json_schema_extra must be a dict")
    if "requires" in extra:
        raise PlanningCommandFormatError(f"{where}: field {name!r}: requires is not a planning command option; reset the "
                               "argument (or refuse the call) in a model validator on Input")
    check_field_keys(name, extra, where, planning=planning)
    spec.update(check_option_spec(name, extra, where, kind=_option_kind(annotation), planning=planning,
                                  annotation=annotation))
    return spec


def _option_value(name: str, where: str, options: dict, key: str, dict_options: tuple[str, ...] = ()):
    """The value of one option key: None when the key is absent, True, or the dict its options take (spec 2.0:
    False, another value, or an option the key does not take fails the load)."""
    if key not in options:
        return None
    value = options[key]
    if value is True:
        return True
    if value is False:
        raise _field_error(where, name, f"{key}: False is not a value; leave the key out to turn it off")
    if not dict_options or not isinstance(value, dict):
        takes = " or a dict of " + ", ".join(dict_options) if dict_options else ""
        raise _field_error(where, name, f"{key} must be True{takes}; got {_shown(value)}")
    if not value:
        raise _field_error(where, name, f"{key} is empty; give {' or '.join(dict_options)}, or leave the key out")
    for option in value:
        if option not in dict_options:
            close = _closest(option, dict_options)
            hint = f"did you mean {close!r}?" if close else f"its options are {', '.join(dict_options)}"
            raise _field_error(where, name, f"{key} has unknown option {option!r}; {hint}")
    return value


def _ignore_words(name: str, where: str, key: str, value) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(isinstance(w, str) for w in value):
        raise _field_error(where, name, f"{key} ignore must be a list of words")
    return tuple(value)


#: The options of ``ui`` (``field_options.UiOptions``), and the values ``format`` and ``widget`` take.
UI_OPTIONS = ("hidden", "hero", "format", "widget", "table", "tones", "currency")
UI_FORMATS = ("currency", "percent", "code", "badge")
#: ``ui.tones`` values: the meaning of a badge's colour, which a client may draw as it likes (good, warning, bad, neutral).
UI_TONES = ("good", "warning", "bad", "neutral")
#: ``ui.currency``: the shape of an ISO 4217 code. No list of currencies is kept here; a client that does not know
#: a well-formed code falls back to a plain number.
UI_CURRENCY_CODE = re.compile(r"[A-Za-z]{3}")
UI_WIDGETS = ("textarea", "radio")
#: ``ui.table`` names a group of ``Signature.Output`` list fields drawn as the columns of one table.
UI_TABLE_GROUP = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _table_column_item(annotation) -> bool:
    """Whether *annotation* is a list of scalars (``list[str|int|float|bool|Literal[...]]``, or ``Optional`` of
    it) a ``ui.table`` column may be."""
    if get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) != 1:
            return False
        annotation = inner[0]
    if get_origin(annotation) is not list:
        return False
    args = get_args(annotation)
    return len(args) == 1 and (args[0] in (str, int, float, bool) or get_origin(args[0]) is Literal)


def check_table_groups(output_cls, where: str = "Signature.Output") -> list[str]:
    """The load problems of the ``ui.table`` declarations of one ``Signature.Output``, as messages.

    A column field is a list of scalars (``list[str]``, ``list[int]``, ``list[float]``, ``list[bool]`` or
    ``list[Literal[...]]``, optionally ``Optional``); a list of objects is refused. A group has at least two
    member fields, and at least two that are not ``hidden`` (a hidden member is still a column for
    alignment but is not shown). The syntax of each ``ui.table`` value is checked by ``check_ui_option``."""
    problems: list[str] = []
    members: dict[str, list[tuple[str, bool]]] = {}
    refused: set[str] = set()    # a group with a refused column is reported once, for that column
    for field_name, info in output_cls.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
        hint = extra.get(UI)
        if not isinstance(hint, dict) or "table" not in hint:
            continue
        group = hint["table"]
        if not isinstance(group, str) or not UI_TABLE_GROUP.fullmatch(group):
            continue    # reported by check_ui_option
        if not _table_column_item(info.annotation):
            problems.append(str(_field_error(where, field_name, f"{UI}.table is for a list of str, int, float, bool or "
                                             f"Literal values; {python_utils.type_text(info.annotation)} is not one")))
            refused.add(group)
            continue
        members.setdefault(group, []).append((field_name, hint.get("hidden") is True))
    for group, fields in members.items():
        if group in refused:
            continue
        names = ", ".join(repr(n) for n, _ in fields)
        if len(fields) < 2:
            problems.append(f"{where}: {UI}.table group {group!r} has one field ({names}); a table needs at least "
                            "two list fields with the same group name, so give the other column the same name or "
                            "leave the key out")
        elif sum(1 for _, hidden in fields if not hidden) < 2:
            problems.append(f"{where}: {UI}.table group {group!r} ({names}) has fewer than two visible fields; "
                            f"a hidden field is a column for alignment only, so leave {UI}.hidden off one more field "
                            "or leave the key out")
    return problems


def ungrouped_list_notices(output_cls, command: str) -> list[str]:
    """Notices (never errors) for one ``Signature.Output``: two or more visible list-of-scalar fields that belong
    to no ``ui.table`` group. Nothing about a field's name is read; the type and the declared hints decide.

    A page draws such lists as separate lists. The author who means them as the columns of one table declares
    ``json_schema_extra={"ui": {"table": "<group>"}}`` on each of them (the same group name)."""
    loose: list[str] = []
    for field_name, info in output_cls.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
        hint = extra.get(UI)
        hint = hint if isinstance(hint, dict) else {}
        if "table" in hint or hint.get("hidden") is True or not _table_column_item(info.annotation):
            continue
        loose.append(field_name)
    if len(loose) < 2:
        return []
    names = ", ".join(repr(n) for n in loose)
    return [f"notice: {command} output fields {names} are lists drawn as separate lists; if they are the "
            "parallel columns of one table (item i of each list describes the same row), declare it on each of "
            f"them with json_schema_extra={{\"{UI}\": {{\"table\": \"<group name>\"}}}}, the same group name on every "
            "column; otherwise nothing needs to change"]


def _ui_value_annotation(annotation):
    """*annotation* without ``Optional`` and one ``list`` level: what a ``ui.tones`` or ``ui.currency`` value is."""
    for _ in range(2):
        if get_origin(annotation) in (Union, UnionType):
            inner = [a for a in get_args(annotation) if a is not type(None)]
            if len(inner) != 1:
                return annotation
            annotation = inner[0]
        if get_origin(annotation) is list and len(get_args(annotation)) == 1:
            annotation = get_args(annotation)[0]
    return annotation


def check_ui_value_options(output_cls, where: str = "Signature.Output") -> list[str]:
    """The load problems of the ``ui`` hints against the field types of one ``Signature.Output``: ``hero`` is for a
    scalar (never a list), ``format`` ``"percent"`` and ``"currency"`` for a number (``"code"`` draws any scalar and
    ``"badge"`` text or a number), and ``tones`` and ``currency`` as below. A field whose hint fails its own syntax check
    (``check_ui_option``) is reported for that alone.

    ``tones`` is for a ``str``, a ``Literal`` or an ``Enum`` (or a list of one; a ``bool`` is drawn as yes/no, never a
    badge), and on a ``Literal`` or ``Enum`` every key must be one of the declared values (compared as text,
    case-insensitively after trimming). ``currency`` is for an ``int`` or a ``float`` (or a list of one). The syntax of
    each option is checked by ``check_ui_option``."""
    problems: list[str] = []
    for field_name, info in output_cls.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
        hint = extra.get(UI)
        if not isinstance(hint, dict):
            continue
        try:
            check_ui_option(field_name, extra, where, on_output=True)
        except PlanningCommandFormatError:
            continue    # the hint's own syntax is the field's one problem
        inner = _ui_value_annotation(info.annotation)
        shown = python_utils.type_text(info.annotation)
        tones = hint.get("tones")
        if isinstance(tones, dict) and hint.get("format") == "badge":
            declared = None
            if get_origin(inner) is Literal:
                declared = list(get_args(inner))
            elif inspect.isclass(inner) and issubclass(inner, enum.Enum):
                declared = [member.value for member in inner]
            if declared is not None and declared and all(isinstance(v, bool) for v in declared):
                declared = False
            if declared is False or not (inner is str or declared is not None):
                problems.append(str(_field_error(where, field_name, f"{UI}.tones is for a str, Literal or Enum value "
                                                 f"(or a list of one); {shown} is not one")))
                continue
            if declared is not None:
                known = {str(v).strip().lower() for v in declared}
                for text in tones:
                    if isinstance(text, str) and text.strip().lower() not in known:
                        problems.append(str(_field_error(
                            where, field_name, f"{UI}.tones key {text!r} is not one of the declared values "
                            f"{', '.join(repr(str(v)) for v in declared)}")))
        code = hint.get("currency")
        number = inner in (int, float, decimal.Decimal)
        if code is not None and hint.get("format") == "currency" and not number:
            problems.append(str(_field_error(where, field_name, f"{UI}.currency is for an int or float value (or a list "
                                             f"of one); {shown} is not one")))
        elif hint.get("format") in ("currency", "percent") and not number:
            problems.append(str(_field_error(where, field_name, f"{UI}.format \"{hint['format']}\" is for an int or "
                                             f"float value (or a list of one); {shown} is not one")))
        if hint.get("hero") is True and _ui_value_annotation(info.annotation) is not _without_optional(info.annotation):
            problems.append(str(_field_error(where, field_name, f"{UI}.hero marks the one scalar to show first; "
                                             f"{shown} is a list: leave the key out, or draw it with {UI}.table")))
    return problems


def _without_optional(annotation):
    """*annotation* without ``Optional`` (the one-list-level step of ``_ui_value_annotation`` left out)."""
    if get_origin(annotation) in (Union, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return inner[0]
    return annotation


def check_ui_option(name: str, extra, where: str, *, kind: str = "other", on_output: bool = False) -> None:
    """The load check of a field's ``ui`` hint (``field_options.UiOptions``); a no-op when *extra* has no ``ui`` key.

    ``ui`` is a presentation hint for any client that renders results or collects input: no check, extraction or plan reads it. It must be a non-empty dict of
    ``hidden``, ``hero``, ``format`` (a ``Signature.Output`` field) or ``widget`` (a ``Signature.Input`` field,
    *on_output* False; ``"textarea"`` needs *kind* ``str``, ``"radio"`` needs ``enum``, from ``_option_kind``)."""
    if UI not in extra:
        return
    value = extra[UI]
    takes = "hidden, hero, format, table, tones or currency on a Signature.Output field, widget on a Signature.Input field"
    if value is False:
        raise _field_error(where, name, f"{UI}: False is not a value; leave the key out to turn it off")
    if not isinstance(value, dict):
        raise _field_error(where, name, f"{UI} must be a dict of {takes}; got {_shown(value)}")
    if not value:
        raise _field_error(where, name, f"{UI} is empty; give {takes}, or leave the key out")
    for key in value:
        if key not in UI_OPTIONS:
            close = _closest(key, UI_OPTIONS)
            hint = f"did you mean {close!r}?" if close else f"its options are {', '.join(UI_OPTIONS)}"
            raise _field_error(where, name, f"{UI} has unknown option {key!r}; {hint}")
    if on_output:
        if "widget" in value:
            raise _field_error(where, name, f"{UI}.widget is for a Signature.Input field; on a Signature.Output "
                               "field use hidden, hero, format, table, tones or currency, or leave the key out")
        for key in ("hidden", "hero"):
            if key in value and value[key] is not True:
                if value[key] is False:
                    raise _field_error(where, name, f"{UI}.{key}: False is not a value; leave the key out to turn it off")
                raise _field_error(where, name, f"{UI}.{key} must be True; got {_shown(value[key])}")
        if "format" in value and value["format"] not in UI_FORMATS:
            raise _field_error(where, name, f"{UI}.format must be one of {', '.join(map(repr, UI_FORMATS))}; "
                               f"got {_shown(value['format'])}")
        if "tones" in value:
            tones = value["tones"]
            if value.get("format") != "badge":
                raise _field_error(where, name, f"{UI}.tones needs {UI}.format \"badge\" on the same field")
            if not isinstance(tones, dict) or not tones:
                raise _field_error(where, name, f"{UI}.tones must be a non-empty dict of value -> tone, such as "
                                   f"{{\"delivered\": \"good\"}}; got {_shown(tones)}")
            seen: dict[str, str] = {}
            for text, tone in tones.items():
                if not isinstance(text, str) or not text.strip():
                    raise _field_error(where, name, f"{UI}.tones keys must be non-empty text; got {_shown(text)}")
                if tone not in UI_TONES:
                    raise _field_error(where, name, f"{UI}.tones tone for {text!r} must be one of "
                                       f"{', '.join(map(repr, UI_TONES))}; got {_shown(tone)}")
                folded = text.strip().lower()
                if folded in seen:
                    raise _field_error(where, name, f"{UI}.tones keys {seen[folded]!r} and {text!r} are the same "
                                       "value (keys match case-insensitively after trimming); write it once")
                seen[folded] = text
        if "currency" in value:
            code = value["currency"]
            if value.get("format") != "currency":
                raise _field_error(where, name, f"{UI}.currency needs {UI}.format \"currency\" on the same field")
            if not isinstance(code, str) or not UI_CURRENCY_CODE.fullmatch(code):
                raise _field_error(where, name, f"{UI}.currency must be a three-letter ISO 4217 code such as "
                                   f"\"USD\"; got {_shown(code)}")
        if "table" in value:
            group = value["table"]
            if not isinstance(group, str) or not UI_TABLE_GROUP.fullmatch(group):
                raise _field_error(where, name, f"{UI}.table must be a group name, a simple identifier such as "
                                   f"\"rows\" (letters, digits and underscores, not starting with a digit); got {_shown(group)}")
            if "hero" in value:
                raise _field_error(where, name, f"{UI}.hero marks a scalar as the one to show first and {UI}.table makes "
                                   "a list a column of a table; one field cannot be both, so write one")
        return
    for key in ("hidden", "hero", "format", "table", "tones", "currency"):
        if key in value:
            raise _field_error(where, name, f"{UI}.{key} is for a Signature.Output field; on a Signature.Input "
                               "field use widget, or leave the key out")
    widget = value["widget"]
    if widget not in UI_WIDGETS:
        raise _field_error(where, name, f"{UI}.widget must be \"textarea\" or \"radio\"; got {_shown(widget)}")
    if widget == "textarea" and kind != "str":
        raise option_error(where, f'{UI}.widget "textarea"', name, kind, "a str")
    if widget == "radio" and kind != "enum":
        raise option_error(where, f'{UI}.widget "radio"', name, kind, "a Literal of text values")


def _literal_values(annotation) -> tuple:
    """The declared values of a ``Literal`` annotation (or ``Optional`` of one), else ()."""
    if rules.is_union(annotation) and type(None) in get_args(annotation):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        annotation = inner[0] if len(inner) == 1 else annotation
    return get_args(annotation) if get_origin(annotation) is Literal else ()


def check_confirms_words(name: str, words, where: str, values: tuple) -> None:
    """The load check of ``classifier_confirms`` ``words`` (``field_options.ConfirmsOptions.words``; training data
    only, read by the synthetic utterance generator): a non-empty dict from a declared value (*values*) to a non-empty
    list of non-empty words, a word once per value (ignoring case) and under one value only."""
    what = f"{CLASSIFIER_CONFIRMS} words"
    if not isinstance(words, dict) or not words:
        raise _field_error(where, name, f"{what} must be a dict from a value to the words users write for it, such "
                           f"as {{\"<value>\": [\"<word>\", ...]}}; got {_shown(words)}")
    owner: dict[str, tuple[str, str]] = {}
    for key, listed in words.items():
        if key not in values:
            raise _field_error(where, name, f"{what}: {key!r} is not a value of it")
        if isinstance(listed, str) or not isinstance(listed, (list, tuple)) or not listed:
            raise _field_error(where, name, f"{what} for {key!r} must be a non-empty list of words; "
                               f"got {_shown(listed)}")
        seen: dict[str, str] = {}
        for word in listed:
            if not isinstance(word, str) or not word.strip():
                raise _field_error(where, name, f"{what} for {key!r}: {_shown(word)} is not a word (non-empty text)")
            folded = word.strip().casefold()
            if folded in seen:
                raise _field_error(where, name, f"{what} for {key!r} list {seen[folded]!r} and {word!r}, the same "
                                   "word (compared ignoring case); write it once")
            seen[folded] = word
            if folded in owner and owner[folded][0] != key:
                raise _field_error(where, name, f"{what}: {word!r} is listed for both {owner[folded][0]!r} and "
                                   f"{key!r}; a word stands for one value only")
            owner[folded] = (key, word)


def check_option_spec(name: str, options: dict, where: str, *, kind: str = "other",
                      planning: bool = False, annotation=None) -> dict:
    """The ``_parse_param`` spec of a field's option keys (spec 2.1 to 2.6), read from its ``json_schema_extra``.

    The author's keys (``field_options``) translate to the spec's internal options: ``copied_from_command_text``
    is the internal ``in_request`` true (with its ``ignore`` words), ``record_named_in_command_text`` is
    ``in_request`` "subject", ``classifier_confirms`` is ``verify`` (true, or the meanings dict) and
    ``classifier_decides`` is ``verify`` "alone" (the internal names; authors never write them). The keys of the same field must fit together, and the types they apply to are checked
    here (*kind*: the coarse type from ``_option_kind``). ``from_command_output``
    and ``command_text_after_command_name`` are read where they are used; their shape is checked here.

    ``classifier_confirms`` ``words`` is training data only: checked here against the ``Literal`` values of
    *annotation*, and never written into the spec, so a dict with ``words`` and no ``meanings`` is the check
    ``True`` gives. ``examples_by`` is checked on the whole ``Input`` (``check_examples_by``)."""
    spec: dict = {}
    check_ui_option(name, options, where, kind=kind)
    copied = _option_value(name, where, options, COPIED_FROM_COMMAND_TEXT, ("ignore",))
    record = _option_value(name, where, options, RECORD_NAMED_IN_COMMAND_TEXT, ("ignore",))
    confirms = _option_value(name, where, options, CLASSIFIER_CONFIRMS, ("meanings", "words"))
    if isinstance(confirms, dict) and "words" in confirms:
        if kind != "enum":
            raise option_error(where, f"{CLASSIFIER_CONFIRMS} words", name, kind, "a Literal")
        check_confirms_words(name, confirms["words"], where, _literal_values(annotation))
    decides = _option_value(name, where, options, CLASSIFIER_DECIDES)
    positional = _option_value(name, where, options, COMMAND_TEXT_AFTER_COMMAND_NAME)
    if FROM_COMMAND_OUTPUT in options:
        source = options[FROM_COMMAND_OUTPUT]
        if not (isinstance(source, str) or (isinstance(source, (list, tuple)) and all(isinstance(s, str) for s in source))):
            raise _field_error(where, name, f"{FROM_COMMAND_OUTPUT} must be \"[Context/]command.field\" text or a list of it; "
                               f"got {_shown(source)}")
    if copied and record:
        raise _field_error(where, name, f"{COPIED_FROM_COMMAND_TEXT} and {RECORD_NAMED_IN_COMMAND_TEXT} cannot both be "
                           f"on a field: {RECORD_NAMED_IN_COMMAND_TEXT} already copies the command text, so write one")
    if confirms and decides:
        raise _field_error(where, name, f"{CLASSIFIER_CONFIRMS} and {CLASSIFIER_DECIDES} cannot both be on a field: "
                           "the classifier confirms a value the extractor fills, or decides a bool alone")
    if positional and planning:
        raise _field_error(where, name, f"{COMMAND_TEXT_AFTER_COMMAND_NAME} is for an execution command's Input")
    if positional and kind != "str":
        raise option_error(where, COMMAND_TEXT_AFTER_COMMAND_NAME, name, kind, "a str")
    if record:
        spec["in_request"] = "subject"
        if isinstance(record, dict) and "ignore" in record:
            spec["ignore"] = _ignore_words(name, where, RECORD_NAMED_IN_COMMAND_TEXT, record["ignore"])
    elif copied:
        spec["in_request"] = "true"
        if isinstance(copied, dict) and "ignore" in copied:
            spec["ignore"] = _ignore_words(name, where, COPIED_FROM_COMMAND_TEXT, copied["ignore"])
    if confirms:
        if isinstance(confirms, dict) and "meanings" in confirms:
            meanings = confirms["meanings"]
            if not isinstance(meanings, dict) or not meanings:
                raise _field_error(where, name, f"{CLASSIFIER_CONFIRMS} meanings must be a dict of each value's meaning")
            spec["verify"] = meanings
        else:
            spec["verify"] = "true"
    elif decides:
        spec["verify"] = "alone"
    return spec




@dataclass(frozen=True)
class PlanningCommandFile:
    """What a planning command file declares: its description, its parsed command, and its template
    text (``steps``: the steps the planner adapts, and for a fixed-steps planning command the steps it runs)."""

    description: str
    steps: tuple[str, ...]
    command: PlanningCommandCommand
    #: ``Signature.plain_utterances``: example requests for the planning command (with or without params).
    plain_utterances: tuple[str, ...] = ()
    #: The planning command file's path.
    path: str = ""


def fixed_steps(command: PlanningCommandCommand) -> bool:
    """Whether *command* is a planning command with fixed steps: no parameters and no ``plan()``. Its one
    phase reads nothing, so every call of it is the same plan (the phase runner runs it as
    one phase)."""
    return not command.params and command.plan_fn is None


def _absent_as_none(command: PlanningCommandCommand, args: dict) -> dict:
    """*args* as the planning command file's code reads them: an ``Optional[str] = None`` field's
    absent argument (``""``) is None."""
    fields = command.input_model.model_fields
    return {n: None if v == "" and n in fields and fields[n].default is None else v
            for n, v in args.items()}


def input_instance(command: PlanningCommandCommand, args: dict):
    """The planning command's ``Input`` for *args*, unvalidated, as ``steps()``, ``plan()`` and the hook
    get it (``_absent_as_none``)."""
    return command.input_model.model_construct(**_absent_as_none(command, args))


def _call_steps(fn: Callable, instance, where: str) -> list[str]:
    import inspect
    from fastworkflow.planning import Step

    takes_input = bool(inspect.signature(fn).parameters)
    lines = fn(instance) if takes_input else fn()
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)) or \
            not all(isinstance(line, str) for line in lines):
        raise PlanningCommandCallError(f"{where}: steps() must return a list of str")
    return [
        (Step(line.strip(), line.helper) if isinstance(line, Step) else line.strip())
        for line in lines if line.strip()
    ]


#: A planning command's input combinations are each tried up to this many; past it, a bounded set of
#: them (``input_grid``). Shared by ``placeholder_template`` and the planning command preview.
MAX_GRID = 4096
#: Planning command files warned about once in this process (past ``MAX_GRID``).
_warned_grids: set[str] = set()


def input_grid(sizes: list[int], first: tuple | None = None) -> list[tuple]:
    """The combinations (index tuples) of inputs with *sizes* values each that are tried,
    *first* (default all 0) first: every one, in ``itertools.product`` order, up to
    ``MAX_GRID``; past it, *first*, each single change of it and each pair of changes of it
    (every two inputs' values together at least once, the other inputs as in *first*)."""
    import itertools

    first = tuple(0 for _ in sizes) if first is None else tuple(first)
    total = 1
    for size in sizes:
        total *= size
    if total <= MAX_GRID:
        grid = list(itertools.product(*[range(s) for s in sizes]))
        grid.remove(first)
        return [first] + grid
    changes = [[i for i in range(s) if i != first[k]] for k, s in enumerate(sizes)]
    out = [first]
    for k in range(len(sizes)):
        out += [first[:k] + (i,) + first[k + 1:] for i in changes[k]]
    for j, k in itertools.combinations(range(len(sizes)), 2):
        for a, b in itertools.product(changes[j], changes[k]):
            combo = list(first)
            combo[j], combo[k] = a, b
            out.append(tuple(combo))
    return out


def _template_values(p: Param) -> tuple:
    """The inputs ``placeholder_template`` gives one parameter; the first is its base input.

    A str reads ``<name>`` (and "" too when that is its default), a bool is true and false, an
    int takes its default (or 1 and 0 when that is 0), a float its default (or 1.0 and 0.0 when
    that is 0), a list[str] its default (or ``["<name>"]``: one value, never varied), an enum each
    value (its default first).
    """
    if p.type == "str":
        return (f"<{p.name}>",) if p.required or p.default != "" else (f"<{p.name}>", "")
    if p.type == "bool":
        return (True, False)
    if p.type == "int":
        return (p.default,) if p.default else (1, 0)
    if p.type == "float":
        return (p.default,) if p.default else (1.0, 0.0)
    if p.type == "list[str]":
        return (_json_value(p) if p.default else [f"<{p.name}>"],)
    if p.default is None:
        return tuple(p.values)
    return (p.default,) + tuple(v for v in p.values if v != p.default)


def _env_key(env: dict) -> tuple:
    """*env* as a hashable key (a list[str] argument is a list)."""
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in env.items())


def _template_grid(command: PlanningCommandCommand) -> list[dict]:
    """The argument sets ``placeholder_template`` renders: the base first, then the other
    combinations ``input_grid`` tries (past ``MAX_GRID``, a bounded set, logged once per
    planning command file), leaving out a set the planning command's model validators change or
    refuses (``_validate_arguments`` never passes one on to ``steps()``; the set it changes to is
    another of the grid, or none)."""
    values = [_template_values(p) for p in command.params]
    sizes = [len(v) for v in values]
    total = 1
    for size in sizes:
        total *= size
    if total > MAX_GRID and command.path not in _warned_grids:
        from fastworkflow.utils.logging import logger

        _warned_grids.add(command.path)
        logger.warning(f"planning command {command.name!r}: {total} combinations of its inputs, past {MAX_GRID}: "
                       "its template is rendered from its defaults and each one or two changes of them")
    # past MAX_GRID the grid is centred on the defaults (a bool false, a str empty when that is
    # its default), the inputs most calls have
    first = None if total <= MAX_GRID else tuple(
        v.index(p.default) if not p.required and p.default in v and p.type != "int" else 0
        for p, v in zip(command.params, values))
    grid = [{p.name: v[i] for p, v, i in zip(command.params, values, combo)} for combo in input_grid(sizes, first)]
    out, seen = [], set()
    for env in grid:
        key = _env_key(env)
        if key in seen:
            continue
        try:
            if check_input(command, env, "", None):   # a validator changes this combination
                continue
        except PlanningCommandCallError:
            continue
        seen.add(key)
        out.append(env)
    return out


def _merge_runs(runs: list[list[str]]) -> list[list]:
    """``[[line, {grid index, ...}], ...]``: the steps of every grid point in one order. Each
    run is aligned with the lines so far (a line a run writes twice is kept twice); a line
    only some runs write is placed where they write it."""
    import difflib

    merged: list[list] = []
    for i, lines in enumerate(runs):
        ops = difflib.SequenceMatcher(None, [m[0] for m in merged], lines, autojunk=False).get_opcodes()
        new: list[list] = []
        for tag, a0, a1, b0, b1 in ops:
            if tag == "equal":
                for k in range(a0, a1):
                    merged[k][1].add(i)
                new.extend(merged[a0:a1])
            else:
                new.extend(merged[a0:a1])
                new.extend([line, {i}] for line in lines[b0:b1])
        merged = new
    return merged


def _literal_words(p: Param, allowed: frozenset) -> str:
    if p.type == "bool":
        return p.name if True in allowed else f"not {p.name}"
    if p.type == "str":
        return f"{p.name} is empty" if "" in allowed else f"{p.name} is given"
    if p.type in ("int", "float"):
        return f"not {p.name}" if 0 in allowed else p.name
    shown = [v for v in p.values if v in allowed]
    if len(p.values) > 2 and len(shown) == len(p.values) - 1:
        return f"{p.name} is not {next(v for v in p.values if v not in allowed)}"
    return f"{p.name} is " + " or ".join(shown)


def _condition_text(command: PlanningCommandCommand, present: set, grid: list[dict]) -> str | None:
    """The words of a simple condition true exactly on the grid points ``present``: "" when
    that is all of them, else an ``and`` of up to three parameter tests or an ``or`` of up
    to three ``and``s of up to two; None when none fits."""
    return _Conditions(command, grid).text(present)


class _Conditions:
    """``_condition_text`` for the point sets of one grid. A set of points is a bit mask (bit
    i for grid point i), each parameter test (a set of its values) the mask of the points it
    holds on, and a text is kept by its mask. Tests are tried in ``_condition_text``'s order
    (a parameter's sets by size, then in its values' order), so the first that fits is the
    same; an ``and`` that already misses a wanted point, or an ``or`` whose remaining terms
    cannot cover them, is skipped with everything that extends it.

    Only tests that can win are made, from the values a parameter takes on the wanted points
    (``seen``). In an ``and`` every test holds on all of them, so it allows at least ``seen``,
    and the test allowing just ``seen`` comes first. In an ``or`` a winning test allows only
    values of ``seen`` (one more makes a later test), so each set of them is tried while
    ``seen`` has at most ``MAX_OR_VALUES`` values; past it, each one value alone."""

    MAX_OR_VALUES = 8

    def __init__(self, command: PlanningCommandCommand, grid: list[dict]):
        self.size = len(grid)
        self.full = (1 << self.size) - 1
        self.cache: dict[int, str | None] = {}
        index = {_env_key(env): i for i, env in enumerate(grid)}
        # per parameter: (i, j) for each grid point i and each point j that is i with only
        # that parameter's value changed
        self.pairs = [(p, [(i, j) for i, env in enumerate(grid) for v in _template_values(p)
                           if v != env[p.name] for j in [index.get(_env_key({**env, p.name: v}))]
                           if j is not None])
                      for p in command.params]
        # per parameter: (value, mask of the points with that value), in _template_values order
        self.values = {p.name: [(v, sum(1 << i for i, env in enumerate(grid) if env[p.name] == v))
                                for v in _template_values(p)]
                       for p in command.params}

    def text(self, present: set) -> str | None:
        want = 0
        for i in present:
            if 0 <= i < self.size:
                want |= 1 << i
        if want not in self.cache:
            self.cache[want] = self._text(want)
        return self.cache[want]

    def _text(self, want: int) -> str | None:
        import itertools

        if want == self.full:
            return ""
        relevant = [p for p, pairs in self.pairs   # whose value alone can change whether the line is written
                    if any((want >> i & 1) != (want >> j & 1) for i, j in pairs)]
        seen = {p.name: [(v, mask) for v, mask in self.values[p.name] if mask & want] for p in relevant}

        def test(p: Param, chosen) -> tuple:   # (p, allowed values, mask of the points it holds on)
            mask = 0
            for _, m in chosen:
                mask |= m
            return p, frozenset(v for v, _ in chosen), mask

        found = self._and([test(p, seen[p.name]) for p in relevant
                           if len(seen[p.name]) < len(self.values[p.name])], want)
        if found is not None:
            return found
        return self._or([test(p, chosen) for p in relevant
                         for k in range(1, min(len(seen[p.name]), len(self.values[p.name]) - 1) + 1)
                         if k == 1 or len(seen[p.name]) <= self.MAX_OR_VALUES
                         for chosen in itertools.combinations(seen[p.name], k)], want)

    @staticmethod
    def _words(literals: list[tuple], conj) -> str:
        return " and ".join(_literal_words(literals[k][0], literals[k][1]) for k in conj)

    def _and(self, literals: list[tuple], want: int) -> str | None:
        """The first ``and`` of up to three *literals* of different parameters true exactly on *want*."""
        names = [p.name for p, _, _ in literals]
        masks = [mask for _, _, mask in literals]
        n = len(literals)
        for a in range(n):
            if masks[a] == want:
                return self._words(literals, (a,))
        for a in range(n):
            for b in range(a + 1, n):
                if names[a] != names[b] and masks[a] & masks[b] == want:
                    return self._words(literals, (a, b))
        for a in range(n):
            if masks[a] & want != want:
                continue
            for b in range(a + 1, n):
                t = masks[a] & masks[b]
                if t & want != want or names[a] == names[b]:
                    continue
                for c in range(b + 1, n):
                    if t & masks[c] == want and names[c] != names[a] and names[c] != names[b]:
                        return self._words(literals, (a, b, c))
        return None

    def _or(self, literals: list[tuple], want: int) -> str | None:
        """The first ``or`` of two or three ``and``s of up to two *literals* true exactly on *want*."""
        names = [p.name for p, _, _ in literals]
        masks = [mask for _, _, mask in literals]
        n = len(literals)
        small: list[tuple] = []   # (literal indexes, mask): the ands of one or two that hold only within want
        for a in range(n):
            if not masks[a] & ~want:
                small.append(((a,), masks[a]))
        for a in range(n):
            for b in range(a + 1, n):
                t = masks[a] & masks[b]
                if names[a] != names[b] and not t & ~want:
                    small.append(((a, b), t))

        def joined(terms) -> str:
            return " or ".join(f"({self._words(literals, c)})" if len(c) > 1 else self._words(literals, c)
                               for c in terms)

        m = len(small)
        rest = [0] * (m + 1)   # rest[k]: the points the terms from k on hold on
        for k in range(m - 1, -1, -1):
            rest[k] = rest[k + 1] | small[k][1]
        for a in range(m):
            if small[a][1] | rest[a + 1] != want:
                continue
            for b in range(a + 1, m):
                if small[a][1] | small[b][1] == want:
                    return joined((small[a][0], small[b][0]))
        for a in range(m):
            if small[a][1] | rest[a + 1] != want:
                continue
            for b in range(a + 1, m):
                t = small[a][1] | small[b][1]
                if t | rest[b + 1] != want:
                    continue
                for c in range(b + 1, m):
                    if t | small[c][1] == want:
                        return joined((small[a][0], small[b][0], small[c][0]))
        return None


def _words(text: str) -> list[str]:
    """*text* cut into the parts ``_alternatives`` compares: after each space, ``=`` and
    ``(``, and before each ``)`` (``query=``, ``(``, ``identity_uid ``, ``)``), a
    placeholder (``<reach text>``) kept whole, so a varying part never starts inside a
    placeholder."""
    out: list[str] = []
    depth = 0
    for part in re.split(r"(?<=[ =(])|(?=\))", text):
        if not part:
            continue
        if depth and out:
            out[-1] += part
        else:
            out.append(part)
        depth = max(0, depth + part.count("<") - part.count(">"))
    return out


def _alternatives(texts: list[str], by: str, whole: bool = True) -> str | None:
    """One line for the lines ``texts`` (one per value of enum ``by``, in its order) that differ
    only in a middle part: ``<find_group or find_unit, by kind> query=<name>``; None when every
    one differs in more than a few words and in more than they share (different steps), and,
    without ``whole``, when they share no part (lines not at one place in their branches).
    Lines are compared by ``_words``; a middle keeps its brackets balanced and a ``key=``
    its value (the shared ``(``, ``)`` or value it needs is given back to it), so no merge
    cuts a bracket or parts a key from its value."""
    pieces = [_words(t) for t in texts]
    pre = 0
    while all(len(w) > pre for w in pieces) and len({w[pre] for w in pieces}) == 1:
        pre += 1
    suf = 0
    while all(len(w) - pre > suf for w in pieces) and len({w[-1 - suf] for w in pieces}) == 1:
        suf += 1
    raw = ["".join(w[pre:len(w) - suf]) for w in pieces]
    while any(m.count("(") != m.count(")") or m.endswith("=") for m in raw):
        if suf and any(m.count("(") > m.count(")") or m.endswith("=") for m in raw):
            suf -= 1   # a "(" needs its ")"; a "key=" keeps its value
        elif pre and any(m.count(")") > m.count("(") for m in raw):
            pre -= 1
        else:
            break
        raw = ["".join(w[pre:len(w) - suf]) for w in pieces]
    middles = list(dict.fromkeys(m.strip() for m in raw))
    if len(middles) == 1:
        return texts[0]
    if not whole and not pre and not suf:
        return None
    sizes = [len(m.split()) for m in middles]
    if min(sizes) > 3 and max(sizes) > pre + suf:
        return None
    gap = " " if suf and any(m.endswith(" ") for m in raw) else ""
    head, tail = "".join(pieces[0][:pre]), "".join(pieces[0][len(pieces[0]) - suf:])
    return head + "<" + " or ".join(m for m in middles if m) + f", by {by}>" + gap + tail


def _merge_enum_lines(p: Param, merged: list[list], grid: list[dict]) -> list[list]:
    """``merged`` with the branches of enum ``p`` joined where they line up: consecutive blocks
    of lines, each written for its own values of ``p`` alone (the other arguments alike), of
    one length; their n-th lines become one line with ``<a or b, by p>``."""
    own = [env[p.name] for env in grid]
    # the other arguments' values as hashable keys: a list argument's value is a list
    others = [tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in env.items() if k != p.name) for env in grid]

    info = []   # per line: (values of p it is written for, the other arguments it is written for)
    for _line, points in merged:
        values = frozenset(own[i] for i in points)
        rest = frozenset(others[i] for i in points)
        alone = {i for i in range(len(grid)) if own[i] in values and others[i] in rest} == points
        info.append((values, rest) if alone and len(values) < len(p.values) else None)

    def joinable(blocks: list[list[int]]) -> bool:
        sets = [info[b[0]][0] for b in blocks]
        return (sum(len(s) for s in sets) == len(frozenset().union(*sets))
                and len({len(b) for b in blocks}) == 1
                and all(len({info[b[n]][1] for b in blocks}) == 1 for n in range(len(blocks[0]))))

    out: list[list] = []
    i = 0
    while i < len(merged):
        if info[i] is None:
            out.append(merged[i])
            i += 1
            continue
        j = i
        while j < len(merged) and info[j] is not None:
            j += 1
        blocks: list[list[int]] = []
        for k in range(i, j):
            if blocks and info[blocks[-1][-1]][0] == info[k][0]:
                blocks[-1].append(k)
            else:
                blocks.append([k])
        # A block's place in its values' branches (None when its values differ in it): lines
        # that share no part join only at the same place ("check_a", the second line of
        # k0..k2's branches, is not the k3 branch's first line "find_k3"; fix-ajir.38).
        offset: dict[int, int | None] = {}   # by a block's first line
        for b in blocks:
            places = {sum(1 for k in range(i, b[0]) if v in info[k][0]) for v in info[b[0]][0]}
            offset[b[0]] = places.pop() if len(places) == 1 else None
        if joinable(blocks):   # the branches in the enum's order
            blocks.sort(key=lambda b: min(p.values.index(v) for v in info[b[0]][0]))
        start = 0
        while start < len(blocks):
            end = start + 1
            while end < len(blocks) and joinable(blocks[start:end + 1]):
                end += 1
            segment = blocks[start:end]
            if len(segment) == 1:
                out.extend(merged[k] for k in segment[0])
            else:
                segment.sort(key=lambda b: min(p.values.index(v) for v in info[b[0]][0]))
                aligned = len({offset[b[0]] for b in segment}) == 1 and offset[segment[0][0]] is not None
                for n in range(len(segment[0])):
                    members = [merged[b[n]] for b in segment]
                    text = _alternatives([m[0] for m in members], p.name, whole=aligned)
                    if text is not None:
                        out.append([text, set().union(*(m[1] for m in members))])
                    elif aligned:   # different steps, one line each
                        out.extend(members)
                    else:           # the lines at each place in their branches, joined apart
                        by_place: dict[int | None, list] = {}
                        for b, m in zip(segment, members):
                            by_place.setdefault(offset[b[0]], []).append(m)
                        for place, group in by_place.items():
                            text = _alternatives([m[0] for m in group], p.name) \
                                if len(group) > 1 and place is not None else None
                            if text is None:
                                out.extend(group)
                            else:
                                out.append([text, set().union(*(m[1] for m in group))])
            start = end
        i = j
    return out


def _merge_empty_str_lines(p: Param, merged: list[list], grid: list[dict]) -> list[list]:
    """``merged`` with each line written for an empty ``p`` joined to its neighbour written for
    ``<p>`` when it is that line with ``<p>`` left out (``assess_risk{owner=}`` of
    ``assess_risk{owner=<owner>}``): one step whose value may be empty."""
    shown = f"<{p.name}>"

    def written_for(points: set) -> set:
        return {grid[i][p.name] for i in points}

    out: list[list] = []
    for line, points in merged:
        if out and written_for(points) == {""} and written_for(out[-1][1]) == {shown} \
                and shown in out[-1][0] and out[-1][0].replace(shown, "") == line:
            out[-1] = [out[-1][0], out[-1][1] | points]
            continue
        out.append([line, points])
    return out


def placeholder_template(command: PlanningCommandCommand) -> list[str]:
    """A planning command file's template text when it declares none: ``steps()`` rendered on placeholder
    inputs.

    ``steps()`` runs on every combination of the inputs ``_template_values`` gives (a str
    reads ``<name>``, and "" when that is its default; a bool true and false; an int its
    default, or 1 and 0; a float its default, or 1.0 and 0.0; a list[str] one value, its default
    or ``["<name>"]``; an enum each value; past ``MAX_GRID`` combinations, those
    ``input_grid`` covers), and the runs are merged in order. A line not
    every run writes ends with ``(only when ...)``, the condition on the arguments under which
    it is written (``kind is group``, ``person is given``, ``not flag``); lines that differ only
    by an enum's value become one (``<find_group or find_unit, by kind> query=<name>``). A
    repeated part shows its steps for the int's default; declare ``Signature.template`` when
    the planner needs more than that.

    Raises PlanningCommandFormatError when ``steps()`` fails on the base inputs (the first of each, or
    the first combination the model validators leave in).
    """
    from fastworkflow.planning import TemplateError

    grid, runs = [], []
    for env in _template_grid(command):
        try:
            lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
        except TemplateError as exc:
            if hint := _public_base_helper_hint(command, exc):
                raise _StepsFailOnPlaceholders(f"{command.path}: {exc}{hint}") from None
            raise PlanningCommandFormatError(f"{command.path}: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - a set steps() refuses is left out of the template
            if not grid and not runs:
                text = f"{command.path}: steps() fails on placeholder inputs ({type(exc).__name__}: {exc})"
                if hint := _public_base_helper_hint(command, exc):
                    raise _StepsFailOnPlaceholders(text + hint) from None
                if isinstance(exc, PhaseError):     # the shape of a phase is wrong: no template excuses it
                    raise _StepsFailOnPlaceholders(text) from None
                raise PlanningCommandFormatError(f"{text}; declare Signature.template") from None
            continue
        grid.append(env)
        runs.append(lines)
    if not grid:
        raise PlanningCommandFormatError(f"{command.path}: the Input's validators change or refuse every "
                               "placeholder combination; declare Signature.template")
    merged = _merge_runs(runs)
    for p in command.params:
        if p.type == "str" and len(_template_values(p)) > 1:
            merged = _merge_empty_str_lines(p, merged, grid)
    for p in command.params:
        if p.type == "enum":
            merged = _merge_enum_lines(p, merged, grid)
    conditions = _Conditions(command, grid)
    out = []
    for line, points in merged:
        condition = conditions.text(points)
        if condition is None:
            out.append(f"{line}{_SOME_ARGUMENTS}")
        elif condition and line.endswith(")") and _FACT_CONDITION in line:
            # the facts' condition (the planning command's PlanGenerator) and the inputs': one "only when"
            both = f"({condition})" if " or " in condition else condition
            out.append(f"{line[:-1]} and {both})")
        else:
            out.append(f"{line} (only when {condition})" if condition else line)
    return out


#: How a phase's template lines end when its method tested facts (``fastworkflow.planning``).
_FACT_CONDITION = " (only when "
#: How a template line ends when no condition on the arguments says when it is written.
_SOME_ARGUMENTS = " (only for some of the arguments)"


def _is_template_line(text: str) -> bool:
    """Whether *text* carries a condition only the planning command's TEMPLATE writes (``(only when
    find.uids)``, ``(only for some of the arguments)``): the planner copied a template line."""
    return _FACT_CONDITION in text or _SOME_ARGUMENTS in text


def _plan_generator_inherits_foreign(plan_cls: type, path: str) -> None:
    """Load error when *plan_cls* subclasses another command's ``PlanGenerator`` (3.4, 2.36), directly or through
    any class between them (a shared base in a module of ``_commands/_shared/``, or a class of the command file
    itself): every class of its MRO is checked."""
    module = sys.modules.get(plan_cls.__module__)
    own = getattr(module, "PlanGenerator", None) if module is not None else None
    for base in plan_cls.__mro__[1:]:
        if not inspect.isclass(base) or base.__name__ != "PlanGenerator":
            continue
        if base is own:
            continue
        other_path = getattr(sys.modules.get(base.__module__, None), "__file__", "") or ""
        if os.path.basename(other_path).startswith("_"):
            continue
        other_name = _qualified_name_from_command_path(other_path)
        if other_name is None:
            continue
        raise PlanningCommandFormatError(
            f"{path}: PlanGenerator inherits from {other_name}'s PlanGenerator; to reuse its "
            f"whole job, call {other_name.split('/')[-1]} in a step; to share phases, move "
            "them to a base class in a module of _commands/_shared/")


def _qualified_name_from_command_path(fpath: str) -> str | None:
    marker = f"{os.sep}_commands{os.sep}"
    if marker not in fpath:
        return None
    rel = fpath.split(marker, 1)[1]
    if rel.startswith("_") or f"{os.sep}_" in rel:
        return None
    return rel[:-3].replace(os.sep, "/")


def _result_method(plan_cls: type) -> tuple[Callable | None, type | None]:
    for klass in plan_cls.__mro__:
        if "__result__" in klass.__dict__:
            return klass.__dict__["__result__"], klass
    return None, None


def _phase_names_for_command(command: "PlanningCommandCommand") -> set[str]:
    names: set[str] = set()
    if command.plan_cls is not None:
        names.update(name for name, _ in planning_api.phase_methods(command.plan_cls))
    if command.plan_fn is None:
        return names
    for env in _template_grid(command):
        try:
            result = command.plan_fn(input_instance(command, env))
        except Exception:  # noqa: BLE001 - other grid points may still bind
            continue
        if is_flat_plan(result):
            continue
        try:
            phases = active_phases(result, command.path)
        except PhaseError:
            continue
        names.update(phase.name for phase in phases)
    return names


def _check_result_method(plan_cls: type, path: str, command: "PlanningCommandCommand") -> bool:
    result_fn, owner = _result_method(plan_cls)
    if result_fn is None:
        return False
    phase_names = _phase_names_for_command(command)
    for parameter in inspect.signature(result_fn).parameters.values():
        if parameter.name in ("self", "input"):
            continue
        if parameter.name not in phase_names:
            where = "__result__" if owner is plan_cls else f"{owner.__qualname__}.__result__"
            raise PlanningCommandFormatError(
                f"{path}: {where} takes {parameter.name!r}, which is not a phase of this command")
    _check_result_annotation(result_fn, owner, plan_cls, path, command)
    return True


def _check_result_annotation(result_fn, owner: type, plan_cls: type, path: str,
                             command: "PlanningCommandCommand") -> None:
    """A return annotation of ``__result__`` that contradicts the declared ``Signature.Output`` does not load:
    ``-> str`` or another model, where the one result rule (``command_executor.command_response_of``) would refuse
    the value at run time. Only a class annotation is read (``Output``, ``CommandResponse`` or no annotation are
    fine); ``Optional[...]``, an unresolvable name or any other form is left to the run-time rule."""
    if command.output_model is None:
        return
    import typing

    try:
        annotation = typing.get_type_hints(result_fn).get("return")   # a string annotation is resolved
    except Exception:  # noqa: BLE001 - a name that does not resolve here is the run-time rule's to judge
        return
    if not inspect.isclass(annotation):
        return
    import fastworkflow
    from pydantic import BaseModel

    if annotation is command.output_model or issubclass(annotation, fastworkflow.CommandResponse):
        return
    if annotation is str or issubclass(annotation, BaseModel):
        where = "__result__" if owner is plan_cls else f"{owner.__qualname__}.__result__"
        raise PlanningCommandFormatError(
            f"{path}: {where} is annotated -> {annotation.__qualname__}, but the command declares "
            f"Signature.Output ({command.output_model.__qualname__}): return that Output, or "
            "CommandResponse(response=..., output=<Output>)")


def command_from_module(name: str, module, path: str) -> PlanningCommandFile:
    """The planning command a planning command file (``_commands/<name>.py``, already imported) defines.

    Raises PlanningCommandFormatError for a malformed declaration (see the module doc).
    """
    import inspect

    from pydantic import BaseModel

    signature = getattr(module, "Signature", None)
    if not inspect.isclass(signature):
        raise PlanningCommandFormatError(f"{path}: a planning command file needs a class Signature")
    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise PlanningCommandFormatError(f"{path}: Signature needs a docstring (the planning command's description)")
    from fastworkflow import planning as planning_api
    from fastworkflow.planning.planning_phases import PhaseError

    plan_cls = getattr(module, "PlanGenerator", None)
    if not inspect.isclass(plan_cls) or not planning_api.phase_methods(plan_cls):
        raise PlanningCommandFormatError(f"{path}: a planning command file needs a class PlanGenerator with at "
                               "least one phase method (fastworkflow.planning)")
    try:
        steps_fn, plan_fn = planning_api.planning_command_parts(plan_cls)
    except PhaseError as exc:
        raise PlanningCommandFormatError(f"{path}: {exc}") from None
    _plan_generator_inherits_foreign(plan_cls, path)
    return planning_command_from_parts(name, signature, path, steps_fn=steps_fn, plan_fn=plan_fn,
                            plan_cls=plan_cls)


def planning_command_from_parts(name: str, signature, path: str, *, steps_fn: Callable | None = None,
                     plan_fn: Callable | None = None,
                     plan_cls: type | None = None) -> PlanningCommandFile:
    """The planning command a ``Signature`` and its behaviour define: a flat ``steps_fn(input)`` or a
    ``plan_fn(input) -> list[Phase]`` (or both: a plan that may return no phases runs flat,
    from ``steps_fn``). ``command_from_module`` builds them from the file's ``PlanGenerator``
    class; the runtime reads only these. Raises PlanningCommandFormatError."""
    import inspect

    from pydantic import BaseModel

    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise PlanningCommandFormatError(f"{path}: Signature needs a docstring (the planning command's description)")
    own_steps = steps_fn is not None
    if steps_fn is None:
        if plan_fn is None:
            raise PlanningCommandFormatError(f"{path}: a planning command needs its steps or its phases")
        steps_fn = _concatenated_steps_fn(plan_fn)
    input_model = getattr(signature, "Input", None)
    if input_model is None:
        input_model = type(f"{signature.__name__}Input", (BaseModel,), {})
    if not (inspect.isclass(input_model) and issubclass(input_model, BaseModel)):
        raise PlanningCommandFormatError(f"{path}: Signature.Input must be a pydantic BaseModel")
    if message := planning_api.reserved_input_name_message(input_model, path):
        raise PlanningCommandFormatError(message)
    from fastworkflow.parameters.input_validation import guard_input_validators
    guard_input_validators(input_model, path)
    output_model = getattr(signature, "Output", None)
    declared_output = (inspect.isclass(output_model) and issubclass(output_model, BaseModel))
    # Signature.Output ``from_command_output`` is checked at field-source resolution (3.3), where it is this
    # command's load error like every other problem of its file.
    utterances = getattr(signature, "plain_utterances", ())
    if isinstance(utterances, str) or not all(isinstance(u, str) for u in utterances):
        raise PlanningCommandFormatError(f"{path}: plain_utterances must be a list of str")
    from fastworkflow.parameters.signatures import reject_retired_db_lookup, removed_validate_hook_message
    try:
        reject_retired_db_lookup(signature, input_model, path)
    except ValueError as error:     # one command's mistake: it does not load, the rest does
        raise PlanningCommandFormatError(str(error)) from None
    if message := removed_validate_hook_message(signature, path):
        raise PlanningCommandFormatError(message)

    params = {n: _parse_param(n, _spec_from_field(n, info, path, planning=True), path)
              for n, info in input_model.model_fields.items()}
    check_examples_by(input_model, path)
    if not params and plan_fn is None:
        try:
            lines = _call_steps(steps_fn, input_model(), path)
        except PlanningCommandCallError as exc:
            raise PlanningCommandFormatError(str(exc)) from None
        if not lines:
            raise PlanningCommandFormatError(f"{path}: steps() returns no steps")
        _refuse_flat_say_that(path, name, lines)
        command = PlanningCommandCommand(name=name, description=description, params=(), path=path,
                               input_model=input_model, steps_fn=steps_fn,
                               plain_utterances=tuple(utterances), plan_cls=plan_cls,
                               template=tuple(lines))
        return PlanningCommandFile(description=description, steps=tuple(lines), command=command,
                         plain_utterances=tuple(utterances), path=path)

    command = PlanningCommandCommand(name=name, description=description, params=tuple(params.values()), path=path, input_model=input_model, steps_fn=steps_fn,
                           plain_utterances=tuple(utterances), plan_fn=plan_fn,
                           plan_cls=plan_cls,
                           output_model=output_model if declared_output else None)
    if plan_fn is not None:
        command = replace(command, phase_outputs=_checked_plan(command, own_steps=own_steps))
    else:
        for env in _template_grid(command):
            try:
                flat_lines = _call_steps(steps_fn, input_instance(command, env), path)
            except Exception:  # noqa: BLE001 - steps that fail here are the template check's to report
                continue
            _refuse_flat_say_that(path, name, flat_lines)
    uses_result = plan_cls is not None and _check_result_method(plan_cls, path, command)
    if declared_output and not uses_result:
        raise PlanningCommandFormatError(
            f"{path}: Signature.Output is declared, but nothing builds it; return it from "
            "__result__(self, input, ...)")
    command = replace(command, uses_result=uses_result)
    declared = getattr(signature, "template", None)
    if declared is not None:
        if isinstance(declared, str) or not all(isinstance(t, str) for t in declared):
            raise PlanningCommandFormatError(f"{path}: Signature.template must be a list of str")
        template = list(declared)   # as written: it is text for the planner
        try:   # the build still checks the steps (a declared template does not excuse a broken phase)
            placeholder_template(command)
        except _StepsFailOnPlaceholders:
            raise
        except PlanningCommandFormatError:
            pass    # the validators refuse every placeholder input: what a declared template is for
    else:
        template = placeholder_template(command)
    if not template:
        raise PlanningCommandFormatError(f"{path}: a planning command needs steps")
    return PlanningCommandFile(description=description, steps=tuple(template), command=replace(command, template=tuple(template)),
                     plain_utterances=tuple(utterances), path=path)


class _StepsFailOnPlaceholders(PlanningCommandFormatError):
    """A phase of the command is malformed (its steps are not a list of str, or it is a public helper of a base): a
    mistake a declared ``Signature.template`` does not excuse (``placeholder_template``). Steps that only fail on the
    placeholder VALUES (a division by a default of 0) are what a declared template is for."""


def _public_base_helper_hint(command: PlanningCommandCommand, error: Exception) -> str:
    """The fix for a phase that fails while the template is built when the phase is a public method a BASE of the
    PlanGenerator defines (a helper someone forgot to name with ``_``), else ""."""
    match = re.search(r"phase '(\w+)'", str(error))
    plan_cls = command.plan_cls
    if match is None or plan_cls is None:
        return ""
    name = match.group(1)
    owner = next((klass for klass in plan_cls.__mro__ if name in vars(klass)), None)
    if owner is None or owner is plan_cls:
        return ""
    where = os.path.basename(inspect.getsourcefile(owner) or "") or owner.__module__
    return (f"; every public method of a PlanGenerator and of its bases is a phase, and {name} is defined on "
            f"the base {owner.__qualname__} ({where}): if {name} is a helper, name it _{name}")


def _concatenated_steps_fn(plan_fn: Callable) -> Callable:
    """``steps(input)`` of a planning command file that defines only ``plan()``: the phases joined."""
    from fastworkflow.planning.planning_phases import concatenated_steps

    def steps(input):
        return concatenated_steps(plan_fn(input))

    return steps


def _refuse_flat_say_that(path: str, name: str, lines) -> None:
    """A flat call whose answer is only 'say that' lines is refused as a last phase of them is (3.16):
    the answer is ``__result__``'s, and the load check can see the steps the call runs flat."""
    from fastworkflow.planning import planning_phases

    if lines and planning_phases.answer_by_code("\n".join(lines)):
        raise PlanningCommandFormatError(planning_phases.say_that_last_message(path, name))


def _checked_plan(command: PlanningCommandCommand, own_steps: bool = True) -> tuple[type, ...]:
    """Run ``plan()`` on the placeholder inputs (``_template_grid``) and check what it returns.

    A flat result (``None`` or ``[]``, ``planning_phases.is_flat_plan``) is fine when the
    file defines its own ``steps()`` (*own_steps*), which that call then runs.
    Each other result must be a list of ``Phase`` whose parameters can be bound
    (``planning_phases.check_bindings``); a parameter defaulting to None must bind (name an
    earlier phase with an output) in at least one of the results. Returns the output models seen.
    Raises PlanningCommandFormatError.
    """
    from fastworkflow.planning.planning_phases import PhaseError, active_phases, check_bindings, is_flat_plan

    models: list[type] = []
    binds: dict[tuple[str, str, str], bool] = {}
    ran = 0
    for env in _template_grid(command):
        try:
            result = command.plan_fn(input_instance(command, env))
        except Exception as exc:  # noqa: BLE001 - a set plan() refuses, as for steps()
            if not ran:
                raise PlanningCommandFormatError(f"{command.path}: plan() fails on placeholder inputs "
                                       f"({type(exc).__name__}: {exc})") from None
            continue
        ran += 1
        if is_flat_plan(result):
            if not own_steps:
                raise PlanningCommandFormatError(
                    f"{command.path}: plan() returns no phases for some inputs (run flat), so "
                    "the planning command file needs its own steps(input)")
            try:
                flat_lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
            except Exception:  # noqa: BLE001 - steps that fail here are the template check's to report
                flat_lines = []
            _refuse_flat_say_that(command.path, command.name, flat_lines)
            continue
        try:
            phases = active_phases(result, f"{command.path}: plan()")
            for key, bound in check_bindings(phases, f"{command.path}: plan()").items():
                binds[key] = binds.get(key, False) or bound
        except PhaseError as exc:
            raise PlanningCommandFormatError(str(exc)) from None
        if not phases:
            raise PlanningCommandFormatError(f"{command.path}: plan() returns no phase")
        for phase in phases:
            if phase.output is not None and phase.output not in models:
                # The facts models' typing (3.12) and ``from_command_output`` (3.3) are checked at field-source
                # resolution, against the workflow's own commands (``check_typed_facts_model``).
                models.append(phase.output)
    for (phase, what, parameter), bound in binds.items():
        if not bound:
            raise PlanningCommandFormatError(
                f"{command.path}: phase {phase!r}: {what} takes {parameter!r}, which names no "
                "earlier phase in any plan; check its spelling")
    return tuple(models)


# ---------------------------------------------------------------- field sources (from_command_output)

_PRODUCER_REF_RE = re.compile(
    r"^(?:([A-Za-z_][A-Za-z0-9_]*)/)?([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?$"
)


class _ResolvedAvailableFrom(list):
    """List written into ``json_schema_extra["available_from"]`` by :func:`resolve_signature_field_sources`,
    distinct from a hand-written ``available_from`` list so re-running resolution stays idempotent
    while a hand-written ``available_from`` (even alongside ``from_command_output``) is still flagged as an error."""


def _hand_written_available_from_value(extra: Mapping) -> Any | None:
    """The hand-written ``available_from`` value for *extra*, if any (not resolver-written)."""
    if _HAND_WRITTEN_AVAILABLE_FROM_KEY in extra:
        return extra[_HAND_WRITTEN_AVAILABLE_FROM_KEY]
    af = extra.get("available_from")
    if af is not None and not isinstance(af, _ResolvedAvailableFrom):
        return af
    return None


def _spelled_from_text(spelled: Any) -> str:
    """Format a ``from_command_output`` source value for error messages."""
    if isinstance(spelled, list) and len(spelled) == 1:
        return f"{str(spelled[0])!r}"
    if isinstance(spelled, list):
        return f"{[str(s) for s in spelled]!r}"
    return f"{str(spelled)!r}"


@dataclass(frozen=True)
class FieldSourceProblem:
    """One problem reported by :func:`resolve_field_sources`."""

    file: str
    message: str
    severity: Literal["error"] = "error"
    #: The command whose file has the problem: the problem is that command's load error, so it does not load
    #: (``FieldSourceResolution.errors_for``). Empty for a problem of a file the workflow shares.
    command: str = ""

    def format(self) -> str:
        return f"{self.file}: {self.message}" if self.file else self.message

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class FieldSourceResolution:
    """Result of :func:`resolve_field_sources` for one workflow."""

    workflow_folderpath: str
    problems: tuple[FieldSourceProblem, ...]
    #: Command names (execution or planning) that have a load error (``errors_for``): none of them loads.
    skipped_commands: frozenset[str]
    model_to_command: Mapping[type, str]
    signatures: Mapping[str, Any]
    command_files: Mapping[str, str]
    import_errors: Mapping[str, str]
    base_contexts: Mapping[str, tuple[str, ...]]
    #: The planning commands (``_commands/`` with ``PlanGenerator``) that loaded, by
    #: qualified name: what run_planning_command runs and a restored call rebuilds from.
    planning_commands: Mapping[str, PlanningCommandCommand] = field(default_factory=dict)
    #: ``callees[name]``: the qualified names the steps of each planning command call (3.7).
    callees: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[FieldSourceProblem, ...]:
        return tuple(p for p in self.problems if p.severity == "error")

    def errors_for(self, command_name: str) -> list[str]:
        """Why *command_name* does not load: every problem found in its file, one rule for every kind of problem
        and both kinds of command. Empty when it loads."""
        return list(dict.fromkeys(p.format() for p in self.problems
                                  if p.severity == "error" and p.command == command_name))


def _annotation_text(annotation: Any) -> str:
    return python_utils.type_text(annotation)


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        if len(non_none) < len(get_args(annotation)):
            return Union[tuple(non_none)]
    return annotation


def _producer_types_match(declaring_annotation: Any, target_annotation: Any) -> bool:
    """True when *target_annotation* equals *declaring_annotation*, equals it modulo
    ``Optional``, or is ``list[T]`` / ``Optional[list[T]]`` of it."""
    if declaring_annotation == target_annotation:
        return True
    decl_inner = _strip_optional(declaring_annotation)
    targ_inner = _strip_optional(target_annotation)
    if decl_inner == targ_inner:
        return True
    if get_origin(targ_inner) is list:
        args = get_args(targ_inner)
        if len(args) == 1:
            elem = args[0]
            if elem == declaring_annotation or _strip_optional(elem) == decl_inner:
                return True
    return False


def _read_base_contexts(
    workflow_folderpath: str,
) -> tuple[dict[str, tuple[str, ...]], list[FieldSourceProblem]]:
    """``({context: (base, ...)}, problems)`` in depth-first declared order for each context
    of the workflow, using :meth:`CommandContextModel.load_base_contexts`.

    Reads and validates each context's ``inherits`` (literals of its ``class Context``) without
    importing any command or creating ``___command_info``. If validation fails, the error (which
    names the file) is recorded as a :class:`FieldSourceProblem` on the ``_commands`` folder.
    """
    commands_dir = os.path.join(workflow_folderpath, "_commands")
    model_path = commands_dir
    if not os.path.isdir(commands_dir):
        return {}, []
    try:
        bases = CommandContextModel.load_base_contexts(workflow_folderpath)
    except WorkflowInheritanceModelError as exc:
        inh_path = os.path.join(workflow_folderpath, "workflow_inheritance_model.json")
        return {}, [
            FieldSourceProblem(
                file=inh_path,
                message=f"{type(exc).__name__}: {exc}",
                severity="error",
            )
        ]
    except Exception as exc:  # noqa: BLE001
        return {}, [
            FieldSourceProblem(
                file=model_path,
                message=f"{type(exc).__name__}: {exc}",
                severity="error",
            )
        ]
    return bases, []


def _static_lookup_scopes(context: str, base_contexts: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The static lookup path of *context* (``""`` for root): own context, its bases in order,
    then ``""`` (global)."""
    if not context or context == "*":
        return ("",)
    scopes: list[str] = [context]
    for base in base_contexts.get(context, ()):
        if base and base != "*" and base not in scopes:
            scopes.append(base)
    scopes.append("")
    return tuple(scopes)


def _resolve_producer_target(
    ref_context: str | None,
    ref_cmd: str,
    declaring_context: str,
    known_names: set[str],
    base_contexts: Mapping[str, tuple[str, ...]],
) -> tuple[str | None, list[str]]:
    """Resolve ``[ref_context/]ref_cmd`` from *declaring_context*.

    Returns ``(resolved_qualified_name, all_matching_qualified_names)``. When unambiguous,
    *resolved_qualified_name* is the target's qualified name; when 0 or 2+ match on the
    static lookup path, *resolved_qualified_name* is None and *all_matching_qualified_names*
    lists the candidates (on the static path if 2+, else anywhere in the workflow).
    """
    if ref_context:
        qualified = f"{ref_context}/{ref_cmd}"
        if qualified in known_names:
            return qualified, [qualified]
        return None, []

    scopes = _static_lookup_scopes(declaring_context, base_contexts)
    on_path: list[str] = []
    for scope in scopes:
        candidate = f"{scope}/{ref_cmd}" if scope else ref_cmd
        if candidate in known_names and candidate not in on_path:
            on_path.append(candidate)
    if len(on_path) == 1:
        return on_path[0], on_path
    if len(on_path) > 1:
        return None, on_path
    anywhere = sorted(
        q for q in known_names
        if q == ref_cmd or q.rsplit("/", 1)[-1] == ref_cmd
    )
    return None, anywhere


def resolve_signature_field_sources(
    command_name: str,
    signature_cls: type,
    file_path: str,
    resolution: FieldSourceResolution,
    *,
    phase_outputs: Sequence[type] = (),
) -> list[FieldSourceProblem]:
    """Resolve ``from_command_output`` declarations on *signature_cls* (its ``Input``, ``Output``, and
    optional *phase_outputs* facts models) against *resolution*, mutating ``Input`` fields'
    ``json_schema_extra["available_from"]`` in place when valid.

    Can be called per-command (e.g. from ``CommandDirectory.ensure_command_hydrated``) on a
    freshly imported ``Signature`` class. Idempotent if ``available_from`` was already written
    by :func:`resolve_field_sources`.
    """
    problems: list[FieldSourceProblem] = []
    declaring_context = command_name.rpartition("/")[0]
    known_names = set(resolution.signatures) | set(resolution.import_errors)

    # 1. Check Signature.Output fields for illegal from_command_output
    output_cls = getattr(signature_cls, "Output", None)
    if inspect.isclass(output_cls) and issubclass(output_cls, BaseModel):
        for field_name, info in output_cls.model_fields.items():
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
            if FROM_COMMAND_OUTPUT in extra:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Signature.Output.{field_name}: an Output field cannot declare "
                             f"{FROM_COMMAND_OUTPUT}; build the value in __result__"),
                    severity="error",
                    command=command_name,
                ))
            if "available_from" in extra:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Signature.Output.{field_name}: available_from is written by the framework from an "
                             f"Input field's {FROM_COMMAND_OUTPUT}, and an Output field declares neither; remove it"),
                    command=command_name,
                ))
            try:
                if info.json_schema_extra is not None and not isinstance(info.json_schema_extra, Mapping):
                    raise _not_a_dict("Signature.Output", field_name)
                check_output_field_keys(field_name, extra, "Signature.Output")
                check_ui_option(field_name, extra, "Signature.Output", on_output=True)
            except PlanningCommandFormatError as error:
                problems.append(FieldSourceProblem(file=file_path, message=str(error), severity="error",
                                                   command=command_name))
        problems.extend(FieldSourceProblem(file=file_path, message=message, severity="error", command=command_name)
                        for message in (*check_table_groups(output_cls), *check_ui_value_options(output_cls)))

    # 2. Check facts models in phase_outputs for from_command_output (distinct from fact_from)
    from fastworkflow.planning.typed_phase_facts import _strip_optional as _typed_facts_strip

    for model in phase_outputs:
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            continue
        for field_name, info in model.model_fields.items():
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
            if model not in resolution.model_to_command and (   # a command's own Output has its own checks
                    key_problem := _facts_field_key_problem(
                        model, field_name, info.json_schema_extra,
                        typed=_typed_facts_strip(info.annotation) in resolution.model_to_command)) is not None:
                problems.append(FieldSourceProblem(file=file_path, message=key_problem, severity="error",
                                                   command=command_name))
            if FROM_COMMAND_OUTPUT in extra:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"{model.__name__}.{field_name} "
                             f"({FROM_COMMAND_OUTPUT} {_spelled_from_text(extra[FROM_COMMAND_OUTPUT])}): "
                             f"{FROM_COMMAND_OUTPUT} is "
                             "only for a command's Input field; type the field with the "
                             "command's Output instead"),
                    severity="error",
                    command=command_name,
                ))

    # 3. Check Signature.Input fields for hand-written available_from and resolve from_command_output
    input_cls = getattr(signature_cls, "Input", None)
    if not (inspect.isclass(input_cls) and issubclass(input_cls, BaseModel)):
        return problems

    for field_name, info in input_cls.model_fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, Mapping):
            continue
        has_from = FROM_COMMAND_OUTPUT in extra
        hw_af = _hand_written_available_from_value(extra)
        if hw_af is not None:
            af_val = hw_af
            first_cmd = (
                af_val[0] if isinstance(af_val, (list, tuple)) and af_val and isinstance(af_val[0], str)
                else af_val if isinstance(af_val, str) and af_val
                else "<command>"
            )
            if "." not in first_cmd:
                first_cmd += ".<field>"
            problems.append(FieldSourceProblem(
                file=file_path,
                message=(f"Input.{field_name}: available_from is written by {FROM_COMMAND_OUTPUT}; "
                         f'write "{FROM_COMMAND_OUTPUT}": "{first_cmd}"'),
                severity="error",
                command=command_name,
            ))
            # Drop a hand-written edge so consumers do not keep reading it after the error;
            # preserve the value under ``_HAND_WRITTEN_AVAILABLE_FROM_KEY`` for later passes.
            if "available_from" in extra and not isinstance(extra["available_from"], _ResolvedAvailableFrom):
                new_extra = dict(extra)
                new_extra[_HAND_WRITTEN_AVAILABLE_FROM_KEY] = extra["available_from"]
                new_extra.pop("available_from", None)
                info.json_schema_extra = new_extra
            continue

        if not has_from:
            continue

        raw_source = extra[FROM_COMMAND_OUTPUT]
        if isinstance(raw_source, str):
            sources = [str(raw_source)]
        elif isinstance(raw_source, Sequence) and not isinstance(raw_source, (bytes, bytearray)):
            sources = [s for s in raw_source]
        else:
            sources = []

        if not sources or not all(isinstance(s, str) and s.strip() for s in sources):
            problems.append(FieldSourceProblem(
                file=file_path,
                message=(f"Input.{field_name}: {FROM_COMMAND_OUTPUT} must be '[<Context>/]<command>.<field>' "
                         f"or a non-empty list of them, got {raw_source!r}"),
                severity="error",
                command=command_name,
            ))
            continue

        resolved_entries: list[str] = []
        field_ok = True
        for raw_entry in sources:
            entry = str(raw_entry).strip()
            match = _PRODUCER_REF_RE.match(entry)
            if match is None:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): malformed reference; "
                             "expected '[<Context>/]<command>.<field>'"),
                    severity="error",
                    command=command_name,
                ))
                field_ok = False
                continue

            ref_ctx, ref_cmd, ref_field = match.group(1), match.group(2), match.group(3)
            target_qname, candidates = _resolve_producer_target(
                ref_ctx, ref_cmd, declaring_context, known_names, resolution.base_contexts
            )
            if target_qname is None:
                field_ok = False
                if ref_ctx:
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): no command "
                                 f"{f'{ref_ctx}/{ref_cmd}'!r}"),
                        severity="error",
                        command=command_name,
                    ))
                elif len(candidates) > 1 and all(
                    c.rpartition("/")[0] in _static_lookup_scopes(declaring_context, resolution.base_contexts)
                    for c in candidates
                ):
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): ambiguous command "
                                 f"{ref_cmd!r} matches {', '.join(candidates)}; qualify it "
                                 "as <Context>/<command>"),
                        severity="error",
                        command=command_name,
                    ))
                elif candidates:
                    scope_label = (
                        f"in {declaring_context}, its bases or the global commands"
                        if declaring_context else "in the global commands"
                    )
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): no command {ref_cmd} "
                                 f"{scope_label}; {', '.join(candidates)} "
                                 f"{'is' if len(candidates) == 1 else 'are'} in another context. Write "
                                 f"the qualified name only when a user can be in that context before running "
                                 f"this command; a qualified name is not checked for that, so otherwise name a "
                                 f"producer on this command's own path, or leave the declaration out"),
                        severity="error",
                        command=command_name,
                    ))
                else:
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=f"Input.{field_name} (from {entry!r}): no command named {ref_cmd!r}",
                        severity="error",
                        command=command_name,
                    ))
                continue

            if target_qname in resolution.import_errors:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): target command "
                             f"{target_qname!r} could not be imported: "
                             f"{resolution.import_errors[target_qname]}"),
                    severity="error",
                    command=command_name,
                ))
                continue

            # A producer edge on a planning command's Input must not name one of its own
            # callees: the edge sends the agent to run the target first, and the steps run it anyway (3.3).
            if target_qname in resolution.callees.get(command_name, frozenset()):
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}) tells the agent to run "
                             f"{target_qname} first, but this command's own steps call "
                             f"{target_qname}. To describe the value, write "
                             "Field(description=...) instead"),
                    severity="error",
                    command=command_name,
                ))
                continue

            target_sig = resolution.signatures.get(target_qname)
            target_out = getattr(target_sig, "Output", None)
            target_in = getattr(target_sig, "Input", None)
            out_fields = (
                target_out.model_fields
                if inspect.isclass(target_out) and issubclass(target_out, BaseModel)
                else {}
            )
            in_fields = (
                target_in.model_fields
                if inspect.isclass(target_in) and issubclass(target_in, BaseModel)
                else {}
            )

            if ref_field is None:
                field_ok = False
                fitting = [f"{entry}.{name}" for name, finfo in out_fields.items()
                           if _producer_types_match(info.annotation, finfo.annotation)]
                advice = (f"an Output field of {entry} that fits {_annotation_text(info.annotation)}: "
                          f"{', '.join(fitting)}" if fitting else
                          f"{entry} has no Output field of a type that fits "
                          f"{_annotation_text(info.annotation)}; give it one")
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): a bare command name is not accepted; "
                             f'write "<command>.<field>", {advice}'),
                    severity="error",
                    command=command_name,
                ))
                continue

            spelled_target = f"{ref_ctx}/{ref_cmd}" if ref_ctx else ref_cmd
            if ref_field in out_fields:
                target_finfo = out_fields[ref_field]
                if not _producer_types_match(info.annotation, target_finfo.annotation):
                    field_ok = False
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): "
                                 f"{_annotation_text(info.annotation)} does not match "
                                 f"{spelled_target}'s Output.{ref_field}: "
                                 f"{_annotation_text(target_finfo.annotation)}"),
                        severity="error",
                        command=command_name,
                    ))
                    continue
                resolved_entries.append(f"{target_qname}.{ref_field}")
            elif ref_field in in_fields:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): {spelled_target}'s "
                             f"{ref_field} is an Input field; {FROM_COMMAND_OUTPUT} names the Output field a "
                             "value comes from. To describe the value, write "
                             f"Field(description=...); the preview shows "
                             f"{spelled_target}.{ref_field}'s description beside it"),
                    severity="error",
                    command=command_name,
                ))
            else:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): {target_qname} has no "
                             f"Output field {ref_field!r}"),
                    severity="error",
                    command=command_name,
                ))

        if field_ok and resolved_entries:
            new_extra = dict(extra)
            new_extra["available_from"] = _ResolvedAvailableFrom(resolved_entries)
            info.json_schema_extra = new_extra

    return problems


def _typed_phase_output_problems(
    folder: str,
    planning_commands: Mapping[str, PlanningCommandCommand],
    resolution: FieldSourceResolution,
) -> list[FieldSourceProblem]:
    """Load errors for typed phase outputs and step coverage (3.12)."""
    from fastworkflow.planning import typed_phase_facts as tpf

    problems: list[FieldSourceProblem] = []
    signatures = resolution.signatures
    known = set(signatures) | {name.split("/", 1)[-1] for name in signatures}
    for qname, command in sorted(planning_commands.items()):
        fpath = command.path
        for model in command.phase_outputs:
            for message in tpf.check_typed_facts_model(
                fpath, model, resolution.model_to_command, signatures,
            ):
                problems.append(FieldSourceProblem(
                    file=fpath, message=message, severity="error", command=qname,
                ))
        if command.plan_fn is None:
            continue
        for env in _template_grid(command):
            try:
                result = command.plan_fn(input_instance(command, env))
            except Exception:  # noqa: BLE001
                continue
            if is_flat_plan(result):
                continue
            try:
                phases = active_phases(result, fpath)
            except PhaseError:
                continue
            for phase in phases:
                if phase.output is None:
                    continue
                inst = input_instance(command, env)
                # Design 3.12: the steps over the placeholder grid, as the template has them. An
                # earlier phase's facts are not there at load, so the real call would run this phase on
                # its defaults (a ``None`` fact returns no steps) and miss the command it reads.
                try:
                    lines = placeholder_steps(phase)
                except PhaseError:
                    try:
                        lines = phase_steps(phase, {"input": inst})
                    except PhaseError:
                        lines = []
                step_cmds = [name for name, _ in step_commands(lines, known)]
                # An open step that writes the command word whole names it (the agent runs it);
                # a phase that never mentions it stays a load error.
                text = "\n".join(str(line) for line in lines)
                step_cmds.extend(
                    name for name in sorted(known)
                    if name not in step_cmds and re.search(rf"\b{re.escape(name)}\b", text)
                )
                for message in tpf.check_phase_steps_name_commands(
                    fpath, phase.name, step_cmds, phase.output,
                    resolution.model_to_command, signatures,
                ):
                    problems.append(FieldSourceProblem(
                        file=fpath, message=message, severity="error", command=qname,
                    ))
    return problems


def _workflow_contexts(folder: str, files: Mapping[str, Any]) -> frozenset[str]:
    """The workflow's context names: each context folder that holds a command, and each context an
    ``inherits`` declares (a base included). What a ``@descends_to`` target must name."""
    contexts = {name.split("/", 1)[0] for name in files if "/" in name}
    try:
        declared = CommandContextModel.declared_bases(folder)
    except Exception:  # noqa: BLE001 - the file's own error is reported where it is read
        declared = {}
    for context, bases in declared.items():
        contexts.add(context)
        contexts.update(bases)
    return frozenset(contexts - {"*", ""})


def input_validator_mode_problems(input_model: Any) -> list[str]:
    """An Input's model validators that are not ``mode="after"``, one problem each. ``validate_input`` checks the
    fields one by one and then runs the model validators on the built Input, so a ``before`` or ``wrap`` validator
    would switch the whole Input to one pydantic call with other refusal wording."""
    decorators = getattr(input_model, "__pydantic_decorators__", None) if inspect.isclass(input_model) else None
    if decorators is None:
        return []
    return [f"Signature.Input: model validator {name!r} is mode=\"{decorator.info.mode}\"; an Input's model "
            "validators must be mode=\"after\" (def check(self) -> Self, reading the built Input)"
            for name, decorator in decorators.model_validators.items() if decorator.info.mode != "after"]


def not_given_default_problems(input_model: Any) -> list[str]:
    """An Input field whose declared default is the framework's not-given marker (``NOT_FOUND`` for a str,
    ``INVALID_INT_VALUE`` / ``INVALID_FLOAT_VALUE`` for a number), one problem each, on both kinds of command. The
    marker is how the framework holds a value nobody gave, so such a default could never be told from a missing
    value: it was required at one check and optional at others (and written into a planning command's steps)."""
    from fastworkflow.parameters.signatures import INVALID_FLOAT_VALUE, INVALID_INT_VALUE, NOT_FOUND

    fields = getattr(input_model, "model_fields", None) if inspect.isclass(input_model) else None
    problems = []
    for name, info in (fields or {}).items():
        default = info.default
        if isinstance(default, str):
            marker = default == NOT_FOUND
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            marker = default in (INVALID_INT_VALUE, INVALID_FLOAT_VALUE)
        else:
            marker = False
        if marker:
            kind = getattr(info.annotation, "__name__", None) or str(info.annotation)
            problems.append(f"Signature.Input.{name}: default {default!r} is the framework's not-given marker, so a "
                            "value could never be told from a missing one. Declare a required field with no default "
                            f"(`{name}: {kind} = Field(description=...)`, asked for when missing), or "
                            f"`{name}: {kind} | None = None` for an optional one")
    return problems


def _closed_contexts(cmd_dir: Any) -> frozenset[str]:
    """The contexts whose callback class declares ``occupiable = False`` (a base other contexts inherit commands
    from, never a place the workflow stands on). Read from the classes directly; a class that does not load is
    reported where it is read (``command_preview._context_load_problems``)."""
    closed = set()
    for name, metadata in cmd_dir.map_context_2_metadata.items():
        try:
            module = python_utils.get_module(metadata.context_module_path, metadata.workflow_folderpath)
        except Exception:  # noqa: BLE001 - its own load problem
            continue
        if getattr(getattr(module, metadata.context_class, None), "occupiable", None) is False:
            closed.add(name)
    return frozenset(closed)


#: The members a command's ``Signature`` may have, read through its class hierarchy: the only names the framework reads
#: from it. ``template`` is read on a planning command only, ``SlotInput`` in the framework's own workflows only (the
#: plan review's slot question). Any other public member fails that command's load; a helper's name starts with ``_``.
SIGNATURE_MEMBERS = ("Input", "Output", "plain_utterances", "generate_utterances", "template_utterances")
PLANNING_SIGNATURE_MEMBERS = ("template",)
FRAMEWORK_SIGNATURE_MEMBERS = ("SlotInput",)
#: Members with a message of their own, reported where the Signature is read (``utils.signatures``).
_SIGNATURE_MEMBERS_REPORTED_ELSEWHERE = ("validate_extracted_parameters", "db_lookup")
#: The declaration decorators of a generator class, by the attribute each sets.
_DECLARATION_DECORATORS = {"__fw_writes__": "@writes", "__fw_approval__": "@user_approval_for_execution",
                           "__fw_descends_to__": "@descends_to"}


def is_framework_command_folder(workflow_folderpath: str) -> bool:
    """Whether *workflow_folderpath* is one of the framework's own workflows (command metadata extraction, plan
    metadata extraction), compared by resolved path. The light form of
    ``train.intent_model_layout.is_framework_workflow``, which imports the training stack."""
    import fastworkflow

    real = os.path.realpath(workflow_folderpath)
    return any(real == os.path.realpath(fastworkflow.get_internal_workflow_path(name))
               for name in ("command_metadata_extraction", "plan_metadata_extraction"))


def signature_member_problems(signature: Any, *, is_planning: bool, framework: bool) -> list[str]:
    """The load errors of a command's ``Signature`` members, read through its class hierarchy (``object`` left
    out): a public member not in ``SIGNATURE_MEMBERS`` (``template`` on a planning command, ``SlotInput`` in a
    framework workflow), and a declaration decorator put on ``Signature`` instead of the generator class."""
    if not inspect.isclass(signature):
        return []
    names = {name for klass in signature.__mro__ if klass is not object for name in vars(klass)}
    problems = []
    if misplaced := [decorator for attribute, decorator in _DECLARATION_DECORATORS.items() if attribute in names]:
        problems.append(f"{', '.join(misplaced)} {'is' if len(misplaced) == 1 else 'are'} on Signature: a declaration "
                        "goes on the ResponseGenerator or PlanGenerator class")
    allowed = (SIGNATURE_MEMBERS + (PLANNING_SIGNATURE_MEMBERS if is_planning else ())
               + (FRAMEWORK_SIGNATURE_MEMBERS if framework else ()))
    for name in sorted(names):
        if name.startswith("_") or name in allowed or name in _SIGNATURE_MEMBERS_REPORTED_ELSEWHERE:
            continue
        if name in PLANNING_SIGNATURE_MEMBERS:
            problems.append(f"Signature.{name} is read only on a planning command: an execution command has no "
                            f"template; remove it")
            continue
        close = difflib.get_close_matches(name, allowed, n=1)
        problems.append(
            f"Signature.{name} is not a Signature member{f'; did you mean {close[0]!r}?' if close else '.'} The "
            f"members are {', '.join(SIGNATURE_MEMBERS)} (and {', '.join(PLANNING_SIGNATURE_MEMBERS)} on a planning "
            "command); a helper's name starts with an underscore")
    return problems


def _command_class_problems(module: Any, fpath: str, signature: Any, is_planning: bool,
                            contexts: frozenset[str], closed: frozenset[str] = frozenset(),
                            framework: bool = False, context: str | None = None,
                            containment: tuple[Mapping[str, list[str]], Mapping[str, tuple[str, ...]]] | None = None
                            ) -> list[str]:
    """The load errors of a command file's classes, beside its Signature's: a Signature with no generator class
    (``command_directory.missing_generator_problem``), a ``ResponseGenerator.__call__`` that does not take three
    arguments (``command_routing.response_call_arity_problem``) and the declarations on the generator
    (``command_declarations.declaration_problems``), and the Signature's members (``signature_member_problems``). One
    list, read as every other problem of the file."""
    from fastworkflow.core.command_declarations import declaration_problems, declared_descend
    from fastworkflow.core.command_directory import missing_generator_problem
    from fastworkflow.core.command_routing import response_call_arity_problem

    problems = []
    if problem := missing_generator_problem(fpath):
        problems.append(problem[len(fpath):].lstrip(": ") if problem.startswith(fpath) else problem)
    problems += signature_member_problems(signature, is_planning=is_planning, framework=framework)
    problems += input_validator_mode_problems(getattr(signature, "Input", None))
    problems += not_given_default_problems(getattr(signature, "Input", None))
    generator = getattr(module, "PlanGenerator" if is_planning else "ResponseGenerator", None)
    if not inspect.isclass(generator):
        return problems
    if not is_planning and (problem := response_call_arity_problem(generator)):
        problems.append(problem)
    input_model = getattr(signature, "Input", None)
    fields = frozenset(getattr(input_model, "model_fields", {}) or {}) if inspect.isclass(input_model) else frozenset()
    if not is_planning and fields:
        # an execution command's Input field keys, one rule with a planning command's (a planning command's are
        # checked where its file is parsed, ``option_params``)
        for name, info in input_model.model_fields.items():
            if info.json_schema_extra is not None and not isinstance(info.json_schema_extra, dict):
                problems.append(str(_not_a_dict("Signature.Input", name)))
            elif isinstance(info.json_schema_extra, dict):
                try:
                    check_field_keys(name, info.json_schema_extra, "Signature.Input", planning=False)
                except PlanningCommandFormatError as error:
                    problems.append(str(error))
    problems += declaration_problems(generator, is_planning=is_planning, contexts=contexts, input_fields=fields)
    if (target := declared_descend(generator)) in closed:
        problems.append(f"@descends_to(\"{target}\") names a context declared occupiable = False: a base other "
                        "contexts inherit commands from, never a place the workflow stands on. Name the context the "
                        "command enters")
    elif target and context and containment and target in contexts:
        problems += descends_to_containment_problems(context, target, *containment)
    return problems


def _containment(folder: str) -> tuple[dict[str, list[str]], dict[str, tuple[str, ...]]] | None:
    """``(declared parents, base contexts)`` of the workflow for the ``@descends_to`` containment check, or None when
    the declarations do not read (their own load problem is reported where the context model loads)."""
    try:
        return CommandContextModel.declared_parents(folder), CommandContextModel.load_base_contexts(folder)
    except Exception:  # noqa: BLE001 - reported by the context model's load
        return None


def descends_to_containment_problems(context: str, target: str, parents: Mapping[str, list[str]],
                                     bases: Mapping[str, tuple[str, ...]]) -> list[str]:
    """The load error of a command of *context* that declares ``@descends_to(target)`` when a *target* record cannot
    sit in a record the command runs on: one of *target*'s declared ``parents`` must be *context*, a context
    *context* inherits (a TodoList command entering a child whose parents name TodoItem), or a context that inherits
    *context* (a command of a base context runs on the records of the contexts that inherit it). A global command
    runs on the root, whatever its context, and is not checked."""
    declared = parents.get(target)
    if declared and any(parent == context or parent in bases.get(context, ()) or context in bases.get(parent, ())
                        for parent in declared):
        return []
    told = (f"declares parents = {list(declared)!r}" if declared
            else "declares no parents (a top-level context, enclosed only by the global commands)")
    return [f"@descends_to(\"{target}\") enters a context whose records do not sit in a {context}: {target} {told} "
            f"in _commands/{target}/_{target}.py, which names neither {context} nor a context it inherits. Add "
            f"\"{context}\" to the parents of {target}, or name the context the command enters"]


def resolve_field_sources(workflow_folderpath: str) -> FieldSourceResolution:
    """Resolve every ``from_command_output`` declaration across the commands and planning commands of *workflow_folderpath*.

    Runs once per workflow (cached by normalized folder path and command-source fingerprint):
    - reads command files from :class:`CommandDirectory` (``_commands/``, planning commands included),
      recording any import failure rather than dropping the command;
    - builds the ``{Signature.Output | Signature.Input -> qualified_name}`` map (3.12);
    - checks every ``from_command_output`` placement and reference grammar, runs the type checks, flags
      hand-written ``available_from`` declarations, and writes ``available_from`` into
      valid ``Input`` fields' ``json_schema_extra``.
    """
    folder = os.path.abspath(workflow_folderpath)
    fingerprint = compute_commands_source_fingerprint(workflow_folderpath)
    return _resolve_field_sources_cached(folder, fingerprint)


@functools.lru_cache(maxsize=None)
def _resolve_field_sources_cached(folder: str, _source_fingerprint: str) -> FieldSourceResolution:
    problems: list[FieldSourceProblem] = []
    skipped: set[str] = set()
    inh_path = os.path.join(folder, "workflow_inheritance_model.json")
    try:
        inh = WorkflowInheritanceModel.load(folder)
        inh.resolve_base_paths(folder)
    except Exception as exc:  # noqa: BLE001
        problems.append(FieldSourceProblem(
            file=inh_path,
            message=f"{type(exc).__name__}: {exc}",
            severity="error",
        ))
    cmd_dir = CommandDirectory.scan(folder)
    core_names = set(cmd_dir.core_command_names)
    all_files: dict[str, tuple[str, str, bool]] = {}
    for qname, metadata in cmd_dir.map_command_2_metadata.items():
        if qname in core_names:
            continue
        owner = metadata.workflow_folderpath or folder
        all_files[qname] = (
            metadata.response_generation_module_path,
            owner,
            metadata.is_planning,
        )

    signatures: dict[str, Any] = {}
    import_errors: dict[str, str] = {}
    model_to_command: dict[type, str] = {}
    planning_command_phase_outputs: dict[str, tuple[type, ...]] = {}
    planning_commands: dict[str, PlanningCommandCommand] = {}
    contexts = _workflow_contexts(folder, all_files)
    closed = _closed_contexts(cmd_dir)
    containment = _containment(folder)

    for qname in sorted(all_files):
        fpath, owner, is_planning = all_files[qname]
        try:
            module = python_utils.get_module(fpath, owner)
        except Exception as exc:  # noqa: BLE001
            import_errors[qname] = f"{type(exc).__name__}: {exc}{raised_in_helper(exc, fpath, owner)}"
            continue
        if module is None:
            import_errors[qname] = f"could not import {fpath}"
            continue
        sig = getattr(module, "Signature", None)
        problems.extend(FieldSourceProblem(file=fpath, message=message, command=qname)
                        for message in _command_class_problems(module, fpath, sig, is_planning, contexts, closed,
                                                               framework=is_framework_command_folder(owner),
                                                               context=qname.split("/", 1)[0] if "/" in qname
                                                               else None, containment=containment))
        if not inspect.isclass(sig):
            continue
        signatures[qname] = sig
        for attr in ("Input", "Output"):
            model_cls = getattr(sig, attr, None)
            if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
                model_to_command[model_cls] = qname
        if is_planning and getattr(module, "PlanGenerator", None) is not None:
            try:
                parsed = command_from_module(qname, module, fpath)
                planning_commands[qname] = parsed.command
                if parsed.command.phase_outputs:
                    planning_command_phase_outputs[qname] = parsed.command.phase_outputs
            except Exception as exc:  # noqa: BLE001 - any mistake in the file fails that command's load, not the workflow's
                msg = str(exc) if isinstance(exc, PlanningCommandFormatError) else f"{type(exc).__name__}: {exc}"
                prefix = f"{fpath}: "
                if msg.startswith(prefix):
                    msg = msg[len(prefix):]
                problems.append(FieldSourceProblem(
                    file=fpath, message=msg, severity="error", command=qname
                ))

    base_contexts, ctx_problems = _read_base_contexts(folder)
    for cp in ctx_problems:
        if cp not in problems:
            problems.append(cp)
    resolution = FieldSourceResolution(
        workflow_folderpath=folder,
        problems=(),
        skipped_commands=frozenset(),
        model_to_command=model_to_command,
        signatures=signatures,
        command_files={q: f for q, (f, _, _) in all_files.items()},
        import_errors=import_errors,
        base_contexts=base_contexts,
        planning_commands=planning_commands,
    )

    # The callee graph and its cycles (3.7) are built before the from_command_output checks, which read it.
    resolution = replace(resolution, callees=_callee_graph(planning_commands, resolution))
    cycle_problems = _cycle_problems(planning_commands, resolution.callees)
    problems.extend(cycle_problems)
    skipped.update(p.command for p in cycle_problems)

    for qname in sorted(signatures):
        fpath = all_files[qname][0]
        cmd_problems = resolve_signature_field_sources(
            qname,
            signatures[qname],
            fpath,
            resolution,
            phase_outputs=planning_command_phase_outputs.get(qname, ()),
        )
        for p in cmd_problems:
            if p not in problems:
                problems.append(p)
            if p.severity == "error":
                skipped.add(qname)

    problems.extend(_typed_phase_output_problems(folder, planning_commands, resolution))

    return replace(
        resolution,
        problems=tuple(problems),
        skipped_commands=frozenset(skipped),
    )


def is_framework_planning_command(command: "PlanningCommandCommand") -> bool:
    """True for a framework-provided planning command (fastworkflow/review/commands): system machinery, not
    trained, not inventoried, not listed. It stays loadable and callable by name (``planning_command``)."""
    import fastworkflow.core.command_directory as command_directory

    source = os.path.abspath(command.path) if command.path else ""
    root = os.path.abspath(str(command_directory.FRAMEWORK_PLANNING_COMMANDS_DIR))
    return source == root or source.startswith(root + os.sep)


def visible_planning_commands(workflow_folderpath: str) -> dict[str, "PlanningCommandCommand"]:
    """The planning commands an inventory counts (the test-set builder, the scoring listings): every loaded planning
    command except the framework's own."""
    return {name: command for name, command in
            resolve_field_sources(workflow_folderpath).planning_commands.items()
            if not is_framework_planning_command(command)}


def planning_command(workflow_folderpath: str, qualified_name: str) -> PlanningCommandCommand | None:
    """The loaded planning command *qualified_name* of a workflow, or None when it has none
    (not a planning command, or its module does not load)."""
    return resolve_field_sources(workflow_folderpath).planning_commands.get(qualified_name)


def _clear_field_sources_cache() -> None:
    _resolve_field_sources_cached.cache_clear()
    from fastworkflow.core import command_directory  # circular import: planning_commands -> command_context_model -> command_directory

    command_directory.clear_resolved_command_signatures_cache()


resolve_field_sources.cache_clear = _clear_field_sources_cache  # type: ignore[attr-defined]
resolve_field_sources.cache_info = _resolve_field_sources_cached.cache_info  # type: ignore[attr-defined]
resolve_field_sources.cache_parameters = _resolve_field_sources_cached.cache_parameters  # type: ignore[attr-defined]



#: A step that names a command: its first word, snake_case (``find_item``, ``go_up``) or a
#: qualified name (``Resource/show_metadata``); then its chain, cut at ``(``, ``;`` or ``,``
#: outside a quoted value (``go_up, then ...`` is prose after the command).
_STEP_COMMAND = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[A-Z]\w*/\w+)\b")
_STEP_ARG = re.compile(r"(?<![\w<])([A-Za-z_]\w*)=")
#: The note a template line ends with when it is written only under a condition
#: (``find_item query=<q> (only when find.uids)``).
_CONDITION_NOTE = re.compile(r"\s*\(only when .*\)$")


def without_condition_note(line) -> str:
    """*line* without its closing ``(only when ...)`` note."""
    return _CONDITION_NOTE.sub("", str(line).rstrip())


def step_commands(lines, known: set[str]) -> list[tuple[str, tuple[str, ...]]]:
    """``(command, parameters)`` of each command step of *lines*: a step whose first word is
    a snake_case name or one of *known* (``find_item query=Dune (note)`` gives
    ``("find_item", ("query",))``); the parameters are the ``key=`` of its chain, before
    the first ``(``, ``;`` or ``,`` outside a quoted value, as code reads them
    (``planning_phases.code_step``)."""
    out = []
    for line in lines:
        text = without_condition_note(str(line).strip())
        first = re.match(r"[\w/]+", text)
        match = _STEP_COMMAND.match(text)
        if match is None and not (first and first.group(0) in known):
            continue
        name = match.group(1) if match else first.group(0)
        chain = _outside_quoted_values(step_chain(text[len(name):])).split(",", 1)[0]
        out.append((name, tuple(_STEP_ARG.findall(chain))))
    return out


def _note_condition_problem(path: str, phase: str, step_text,
                            known: set[str]) -> str | None:
    """The preview --check warning for a non-conditional command step whose note holds a
    condition word (``_NOTE_CONDITION_RE``: runtime words except bare ``only``), or None."""
    text = str(step_text).strip()
    parsed = parse_plan(f"1. {text}", known)
    if not parsed or parsed[0].kind != KIND_COMMAND or not parsed[0].command:
        return None
    clean = text.replace("`", " ").replace("*", " ")
    m_cmd = re.search(rf"\b{re.escape(parsed[0].command)}\b", clean)
    if m_cmd is None:
        return None
    rest = text[m_cmd.end():]
    chain = step_chain(rest)
    note = rest[len(chain):].strip()
    if not note or not _NOTE_CONDITION_RE.search(note):
        return None
    body = note[1:-1].strip() if note.startswith("(") and note.endswith(")") else note.lstrip(";").strip()
    cond_match = _NOTE_CONDITION_RE.search(body)
    if cond_match is None:
        return None
    tail = body[cond_match.start():]
    cond_text = re.sub(
        r"^(?:only\s+when|only\s+if|whenever|when|if|unless|in\s+case)\s+",
        "",
        tail,
        flags=re.IGNORECASE,
    ).strip() or tail.strip()
    without_note = text[:m_cmd.end() + len(chain)].rstrip()
    return (f"{path}: phase {phase!r}, step {text!r}: the note holds a condition, "
            f"but code never reads a note, so {parsed[0].command} would run every time. "
            f'Decide it in code with if, or write when("{cond_text}", "{without_note}") '
            "for the agent to decide")


def _command_steps(command: PlanningCommandCommand, signatures: dict) -> tuple[dict, list[tuple[str, tuple[str, ...]]]]:
    """The workflow's command fields (*signatures* by name, and by the name's last part) and
    each command step of *command*'s template as ``(command, parameters)``: every placeholder
    combination's ``step_commands``, and each ``run`` call (its keywords, and the row id a
    ``named=`` step opens by). The parse that ``check_step_commands`` checks and the callee graph
    (``_callee_graph``) both read."""
    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, steps):
        fields = fields_map or {}
        known = set(fields)
        seen: dict[tuple[str, tuple[str, ...]], None] = {}
        for env in _template_grid(command):
            steps.clear()
            with planning_api.steps_by_phase_noted() as by_phase:
                try:
                    lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
                except Exception:  # noqa: BLE001 - the template build reports it
                    continue
            for found in step_commands(lines, known) + list(steps):
                seen.setdefault(found)
    return fields, list(seen)


def check_step_commands(command: PlanningCommandCommand, signatures: dict,
                        not_loading: frozenset[str] = frozenset()) -> list[str]:
    """Problems with the commands a planning command's steps name: each command step of its template
    (every placeholder combination; ``step_commands``) and each ``pick`` must name a
    command of the workflow (*signatures*: ``facts.workflow_signatures``), and each
    parameter it passes must be a field of that command's ``Signature.Input``, the schema a
    command call is checked against. Returned as messages: the planning command fails its load
    (``workflow_agent``) and ``fastworkflow command preview --check`` fails."""
    fields, seen = _command_steps(command, signatures)
    problems = []
    for name, args in seen:
        if name not in fields:
            verb = "names" if planning_api._DO_NOT_RUN in args else "runs"
            problem = (f"{command.path}: a step {verb} {name!r}, which does not load: see its own problem"
                       if name in not_loading else
                       f"{command.path}: a step {verb} {name!r}, which is not a command of the workflow")
        elif planning_api._DO_NOT_RUN in args:
            continue
        elif more := next((a for a in args if a.startswith(planning_api._NAMED_TAKES_MORE + ": ")), None):
            problem = (f"{command.path}: a step opens a row of {name!r} by named=, but {name!r} also requires "
                       f"{more.split(': ', 1)[1]}, which such a step cannot pass, so code could never run it "
                       "(pass the row's id and every required input)")
        elif planning_api._NAMED_UNRESOLVED in args:
            problem = (f"{command.path}: a step opens a row of {name!r} by named=, but {name!r} declares no "
                       "row id input the runtime can find (declare the producer of its id field with "
                       "from_command_output, or pass the id input explicitly)")
        else:
            unknown = [a for a in args if not any(a in f for f in fields[name])]
            if not unknown:
                continue
            problem = (f"{command.path}: a step passes {', '.join(unknown)} to {name!r}, whose "
                       "Signature.Input has no such field")
        if problem not in problems:
            problems.append(problem)
    return problems


def _callee_graph(planning_commands: Mapping[str, PlanningCommandCommand],
                  resolution: FieldSourceResolution) -> dict[str, frozenset[str]]:
    """``callees[name]`` for each planning command (3.7): the qualified names its steps call.

    The steps are read by ``_command_steps``, the parse ``check_step_commands`` checks, fixed
    steps included. A step's name resolves on the command's own static lookup path nearest scope
    first, as the runtime resolves it (``_resolve_producer_target`` lists the path's matches in
    that order), so a command that shadows a base's or the global command of its name and runs
    that name calls itself. A name off that path that exactly one command of the workflow has
    resolves to it: a step runs in the context an earlier callee left open (``book_loans`` runs
    ``list_loans`` on the book ``book_by_title`` opened). A name that several contexts off the
    path have is no edge; the cycle guard at run time refuses a cycle the graph cannot see.
    """
    known = set(resolution.signatures)
    graph: dict[str, frozenset[str]] = {}
    for qname, command in sorted(planning_commands.items()):
        _, seen = _command_steps(command, resolution.signatures)
        targets: set[str] = set()
        for name, args in seen:
            if planning_api._DO_NOT_RUN in args:
                continue        # a do_not_run step calls nothing
            ref_context, _, ref_cmd = name.rpartition("/")
            context = qname.rpartition("/")[0]
            target, candidates = _resolve_producer_target(
                ref_context or None, ref_cmd, context, known, resolution.base_contexts)
            if target is None and candidates and (
                    len(candidates) == 1
                    or candidates[0].rpartition("/")[0] in _static_lookup_scopes(context, resolution.base_contexts)):
                target = candidates[0]      # unique off the path, or the nearest scope's on it
            if target is not None:
                targets.add(target)
        graph[qname] = frozenset(targets)
    return graph


def _cycle_through(start: str, callees: Mapping[str, frozenset[str]]) -> list[str] | None:
    """The shortest call path from *start* back to *start* (``[start, ..., start]``), or None
    when *start* is on no cycle."""
    came_from: dict[str, str] = {}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for callee in sorted(callees.get(node, ())):
            if callee == start:
                chain = [node]
                while chain[-1] != start:
                    chain.append(came_from[chain[-1]])
                return list(reversed(chain)) + [start]
            if callee not in came_from:
                came_from[callee] = node
                queue.append(callee)
    return None


def _cycle_problems(planning_commands: Mapping[str, PlanningCommandCommand],
                    callees: Mapping[str, frozenset[str]]) -> list[FieldSourceProblem]:
    """A load error for every planning command on a call cycle, a self-call included (3.7).
    Each names the cycle, so the log says which commands form it."""
    problems = []
    for qname in sorted(planning_commands):
        cycle = _cycle_through(qname, callees)
        if cycle is None:
            continue
        path = " -> ".join(cycle)
        message = (f"a planning command cannot call itself: {path}" if len(cycle) == 2
                   else f"a planning command cannot call itself, even through others: {path}")
        problems.append(FieldSourceProblem(
            file=planning_commands[qname].path, message=message, severity="error", command=qname,
        ))
    return problems


def call_depth(name: str, resolution: FieldSourceResolution, _path: frozenset[str] = frozenset()) -> int:
    """The ``Depth:`` of planning command *name* (3.7): the longest chain of planning commands
    from it, itself counted, an execution command counting 0. A cycle is a load error; the
    guard keeps the walk finite."""
    if name in _path:
        return 0
    below = [call_depth(callee, resolution, _path | {name})
             for callee in sorted(resolution.callees.get(name, ()))
             if callee in resolution.planning_commands]
    return 1 + max(below, default=0)


def _grid_phase_steps(planning_command: PlanningCommandCommand | PlanningCommandFile,
                      signatures: dict) -> tuple[str, list[tuple[str, object]]]:
    """*planning_command*'s path and each ``(phase, step line)`` of its placeholder grid (3.13 checks)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (_, steps):
        seen_steps: list[tuple[str, object]] = []
        command = planning_command.command if isinstance(planning_command, PlanningCommandFile) else planning_command
        path = command.path
        for env in _template_grid(command):
            steps.clear()
            with planning_api.steps_by_phase_noted() as by_phase:
                try:
                    lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
                except Exception:  # noqa: BLE001 - the template build reports it
                    continue
            items = by_phase or [
                (command.name,
                 planning_api.Step(without_condition_note(line), line.helper)
                 if isinstance(line, planning_api.Step) else without_condition_note(line))
                for line in lines
            ]
            for phase_name, step_line in items:
                key = (phase_name, str(step_line), getattr(step_line, "helper", None))
                if key not in {(p, str(s), getattr(s, "helper", None)) for p, s in seen_steps}:
                    seen_steps.append((phase_name, step_line))
        return path, seen_steps


def _hand_typed_warning(path: str, phase: str, step_line: str,
                        known: set[str]) -> str | None:
    """The warning for a step that a helper would write, typed by hand (3.13), or None."""
    from fastworkflow.planning import Step

    if isinstance(step_line, Step):
        return None
    text = str(step_line).strip()
    if not text:
        return None
    parsed = parse_plan(f"1. {text}", known)
    if not parsed or not parsed[0].command:
        return None
    step = parsed[0]
    clean = text.replace("`", " ").replace("*", " ")
    m_cmd = re.search(rf"\b{re.escape(step.command)}\b", clean)
    if m_cmd is None:
        return None
    before = clean[:m_cmd.start()]
    rest = text[m_cmd.end():]
    chain = step_chain(rest)
    note = rest[len(chain):]
    helper: str | None = None
    if step.kind == KIND_SKIPPED and _NEGATION_RE.search(before):
        helper = "do_not_run(...)"
    elif step.kind == KIND_CONDITIONAL and _CONDITION_RE.search(before):
        helper = "when(...)"
    elif step.kind == KIND_COMMAND and _PICK_NAMED_RE.search(text):
        helper = "run(..., named=...)"
    elif step.kind == KIND_COMMAND and any(p in note.lower() for p in FULL_LIST_PHRASES):
        helper = "run(..., all_pages=True)"
    elif step.kind == KIND_COMMAND:
        helper = "run(...)"
    if helper is None:
        return None
    return (f"{path}: phase {phase!r}, step {text!r}: written by hand; "
            f"write {helper} so that code reads it as you mean")


def check_hand_typed_steps(planning_command: PlanningCommandCommand | PlanningCommandFile, signatures: dict) -> list[str]:
    """Warnings for steps of *planning_command*'s placeholder grid that a helper would write,
    typed by hand (3.13)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, _):
        known = set(fields_map or {})
    path, seen_steps = _grid_phase_steps(planning_command, signatures)
    warnings: list[str] = []
    for phase_name, step_line in seen_steps:
        warning = _hand_typed_warning(path, phase_name, step_line, known)
        if warning is not None and warning not in warnings:
            warnings.append(warning)
    return warnings


def check_note_condition_steps(planning_command: PlanningCommandCommand | PlanningCommandFile, signatures: dict) -> list[str]:
    """Warnings for non-conditional command steps whose note holds a condition word (3.13)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, _):
        known = set(fields_map or {})
    path, seen_steps = _grid_phase_steps(planning_command, signatures)
    warnings: list[str] = []
    for phase_name, step_line in seen_steps:
        warning = _note_condition_problem(path, phase_name, step_line, known)
        if warning is not None and warning not in warnings:
            warnings.append(warning)
    return warnings


def phases_for(command: PlanningCommandCommand, args: dict) -> list | None:
    """The phases ``plan()`` gives for ``args`` (complete, checked arguments): those that run.

    ``None`` when ``plan()`` returns ``None`` or ``[]`` (``planning_phases.is_flat_plan``):
    this call runs flat, from ``steps()``. Raises PlanningCommandCallError when ``plan()`` fails,
    returns something else than a list of ``Phase``, or leaves every phase out.
    """
    from fastworkflow.planning.planning_phases import PhaseError, active_phases, is_flat_plan

    instance = input_instance(command, args)
    try:
        raw = command.plan_fn(instance)
        if is_flat_plan(raw):
            return None
        phases = active_phases(raw, f"plan() of {command.name}")
    except PhaseError as exc:
        raise PlanningCommandCallError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise PlanningCommandCallError(f"plan() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    if not phases:
        raise PlanningCommandCallError(f"plan() of {command.name} returns no phase")
    return phases


def call_phases(command: PlanningCommandCommand, args: dict) -> list:
    """The phases a call of ``command`` with ``args`` runs: ``phases_for``'s, or, for a flat
    call (no ``plan()``, or ``plan()`` gives no phases), one phase named after the planning command whose
    steps are ``expand(command, args)``. A planning command without phases runs as one phase.
    Raises PlanningCommandCallError as ``phases_for`` and ``expand`` do."""
    from fastworkflow.planning.planning_phases import Phase

    phases = phases_for(command, args) if command.plan_fn is not None else None
    if phases is not None:
        return phases
    steps = expand(command, args)
    return [Phase(command.name, lambda: list(steps))]


def expand(command: PlanningCommandCommand, args: dict) -> list[str]:
    """The planning command's command steps for ``args`` (complete, checked arguments), unnumbered:
    ``steps(input)``. Raises PlanningCommandCallError when ``steps()`` fails or returns something else
    than a list of text."""
    instance = input_instance(command, args)
    try:
        lines = _call_steps(command.steps_fn, instance, command.path)
    except PlanningCommandCallError:
        raise
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise PlanningCommandCallError(f"steps() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    return lines


def number(steps: list[str]) -> str:
    """Steps as the planner writes them: a numbered list, one per line."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _validate_arguments(command: PlanningCommandCommand, args: dict, request: str, workflow=None,
                        notes: list[str] | None = None) -> dict:
    """The validation step after the options' checks (spec 3.5), for a typed planning call, which has no
    ``Signature`` to run it on: the ``Input``'s field checks and model validators, with the request, workflow and
    command as context (``check_input``, which writes their changes back). Raises PlanningCommandInputError when a
    validator or a rule refuses the call, PlanningCommandCallError when a validator raises."""
    written = check_input(command, args, request, workflow)
    if written and notes is not None:
        notes += [f"{n}={args[n]!r} -> {v!r} (validator)" for n, v in written.items()]
    return {**args, **written}


def check_input(command: PlanningCommandCommand, args: dict, request: str, workflow=None) -> dict:
    """Validate ``args`` against the planning command's ``Input``: its field types and its model validators, which
    get the validation context ``{"command_text": request, "workflow": workflow, "command": command.name}``
    (``parameters.validate.validate``, the one driver; this function words its outcome for a typed planning call).

    Raises PlanningCommandInputError, whose message says which field breaks which rule (a field's type or
    field validator). A ``fastworkflow.Refused`` a model validator raised is the refusal: its message, with its
    ``fields`` (``refusal`` holds the same message). A plain ``ValueError`` or ``AssertionError`` is exactly a
    ``Refused`` that names no field. Any other exception a model validator raises is a PlanningCommandCallError
    ``a validator raised <Type>: <text>`` (``raised`` names its type, ``detail`` its text).
    A refusal carries no writes: what a validator rewrote before it stopped is dropped (one rule, both kinds).

    Returns the fields the validators changed, ``{name: value}``, which the caller writes back onto the
    arguments (a validator writes its correction to ``self`` and returns it, spec 3.4). Empty when none changed.
    """
    from fastworkflow.parameters import validate as one_driver

    passed = _absent_as_none(command, args)
    try:
        outcome = one_driver.validate(command.input_model, passed, command_text=request, workflow=workflow,
                                      command_name=command.name)
    except one_driver.ValidatorReturnError as exc:
        raise PlanningCommandCallError(str(exc)) from None
    refusal = outcome.refusal
    if refusal is None:
        return outcome.written
    if refusal.kind == one_driver.RULE:
        error = PlanningCommandInputError(refusal.message)
        error.refusal = refusal.detail or refusal.message
        raise error from None
    if refusal.kind == one_driver.REFUSED:
        error = PlanningCommandInputError(refusal.message, fields=refusal.fields)
        error.refusal = refusal.message
        raise error from None
    failure = PlanningCommandCallError(f"a validator raised {refusal.raised}: {refusal.message}")
    failure.raised = refusal.raised
    failure.detail = refusal.message
    raise failure from None


# ---------------------------------------------------------------- arguments and guards

def strip_words(text: str, words) -> str:
    """``text`` without the *words* at either end (case-insensitive; the words are whole words of ``text``).

    The value cleanup for a field with no check, as ordinary code: call it from an Input's field
    validator (whose returned value is written back) or from a model validator on ``Input``.
    A field with a check takes ``copied_from_command_text: {"ignore": [words]}`` instead, which runs before the match.
    A possessive ``'s`` left on a name stripped from its end goes with it ("Ann's group" -> "Ann").
    Text of one word is kept whole.
    """
    words = tuple(words)
    if not words:
        return text
    lowered = {w.lower() for w in words}
    tokens = text.split()
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        if tokens[0].lower() in lowered:
            tokens, changed = tokens[1:], True
        elif tokens[-1].lower() in lowered:
            tokens, changed = tokens[:-1], True
            if tokens and re.search(r"['’]s$", tokens[-1]):
                tokens[-1] = tokens[-1][:-2]
    return " ".join(tokens).strip()


def _single(value):
    """``value``, or the element of a one-element list given for it (a list, or the list
    written as JSON text: ``'["Payroll"]'``)."""
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and not isinstance(value[0], (list, tuple, dict)):
        return value[0]
    return value


def _list_items(value) -> list[str]:
    """The elements a list[str] argument holds: a list or tuple, the list written as JSON text, or
    comma-separated text. None elements are left out here; empty ones are left to the caller."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(v) for v in parsed if v is not None]
    return text.split(",") if text else []


def _empty(value) -> bool:
    return value is None or str(value).strip().strip("\"'`").lower() in ("", "none", "null")


def _confident(verdicts: dict | None, name: str):
    """The ``(value, probability)`` check of ``name`` when it is sure enough, else None."""
    verdict = (verdicts or {}).get(name)
    return verdict if verdict is not None and verdict[1] >= VERIFY_THRESHOLD else None


def _request_numbers(request: str, args: dict, command: PlanningCommandCommand) -> set[int]:
    """The numbers ``request`` writes (digits, with a thousands separator or a minus sign, or number words)
    outside the str arguments' text. The execution side's reader (``argument_checks.written_numbers``), so
    "1,000" is 1000 and "-5" is -5 on both sides."""
    from fastworkflow.parameters.rules import written_numbers

    texts = [str(args.get(p.name) or "") for p in command.params if p.type == "str"]
    return written_numbers(request, [text for text in texts if text])


def _spans(request: str, text: str) -> list[tuple[int, int]]:
    """Where ``text`` occurs in ``request`` (case-insensitive; any run of spaces, either apostrophe)."""
    if not text:
        return []
    pattern = re.escape(rules.fold(text)).replace(r"\ ", r"\s+").replace("'", "['’]")
    return [(m.start(), m.end()) for m in re.finditer(pattern, rules.fold(request), re.I)]


def _whole_identifier(request: str, text: str, identifiers: tuple) -> tuple[str | None, bool]:
    """``(match, cut)`` of a str argument against the workflow's identifier matches in ``request``.

    ``cut`` is true when every occurrence of ``text`` lies inside an identifier match without
    being all of it ("4421" of "acct_4421"); ``match`` is then the whole match when exactly one
    match holds it, else None.
    """
    matches = [(m.start(), m.end()) for p in identifiers for m in p.finditer(request) if m.group(0)]
    occurrences = _spans(request, text)
    if not matches or not occurrences:
        return None, False
    holders = set()
    for a, b in occurrences:
        inside = [(s, e) for s, e in matches if s <= a and b <= e and (s, e) != (a, b)]
        if not inside:
            return None, False
        holders |= {request[s:e] for s, e in inside}
    return (holders.pop() if len(holders) == 1 else None), True


def _name_like_word(word: str) -> bool:
    """A capitalised word that is not a common one ("I", "Before"), or one holding a digit or ``_``."""
    from fastworkflow.agent.split_merge import _COMMON
    w = word.strip(".,;:!?()'\"`")
    return bool(w) and (any(c.isdigit() for c in w) or "_" in w
                        or (w[:1].isupper() and w != "I" and w.lower() not in _COMMON))


def _plain_name(request: str, a: int, b: int, kinds: frozenset[str] = frozenset()) -> bool:
    """``request[a:b]`` is a name written in lower case ("ann lee" of "is ann lee in Payroll"):
    two or more words, none a common one, with no such word right before or after it in its
    clause, so that it cuts no longer name ("ann lee" of "ann lee smith"), and not one name of
    a chain joined by "and" that ends on a company word ("henderson ramirez" of "henderson
    ramirez and lewis and sons", as ``split_merge`` reads a capitalised chain). A single
    lower-case word ("roster", "user") reads as a common noun, not a name.

    A neighbour that is a clause word, a verb of ``split_merge._FINITE`` or a preposition of
    ``split_merge._PLACING`` ("does barbara sanchez hold ...", "is alan cooper among ..."), is no
    part of the name; nor is one of the command's ``kinds`` (its enum values, or their plural)
    right before it when a clause word follows it ("every right brent randolph holds")."""
    from fastworkflow.agent.split_merge import _COMPANY_SUFFIX, _FINITE, _NOT_NAMES, _PLACING, _STOP

    def bare(word: str) -> str:
        return word.strip(".,;:!?()'\"`").lower()

    def plain(word: str) -> bool:
        return bool(bare(word)) and bare(word) not in _NOT_NAMES | _STOP

    def company(ws: list[str]) -> bool:
        return any(bare(w) in _COMPANY_SUFFIX for w in ws)

    def clause_word(word: str) -> bool:
        return _FINITE.fullmatch(bare(word)) is not None or re.fullmatch(_PLACING, bare(word)) is not None

    def kind(word: str) -> bool:
        return bare(word) in kinds or (bare(word).endswith("s") and bare(word)[:-1] in kinds)

    words = request[a:b].split()
    clause = r"[.,;:!?()\"]"
    before = re.split(clause, request[:a])[-1].split()
    # a possessive ends the name ("ann lee's login")
    after = [] if re.match(r"['’]s\b", request[b:]) else re.split(clause, request[b:])[0].split()
    verb_after = bool(after) and clause_word(after[0])
    return (len(words) > 1 and all(plain(w) for w in words)
            and not (before and plain(before[-1]) and not clause_word(before[-1])
                     and not (verb_after and kind(before[-1])))
            and not (after and plain(after[0]) and not verb_after)
            and not (bare(after[0] if after else "") in ("and", "&") and company(after))
            and not (bare(before[-1] if before else "") in ("and", "&") and company(words)))


def _opening_tails(request: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The tails of the subject ``spans`` of ``request`` that open a sentence: such a subject
    may hold the word that opens the sentence ("Treat Ledger" of "Treat Ledger as ..."), and its
    tail ("Ledger") stands for it."""
    tails = []
    for s, e in spans:
        first = re.match(r"\S+\s+", request[s:e])
        if first and (s == 0 or re.search(r"[.?!:;]\s*$", request[:s])):
            tails.append((s + first.end(), e))
    return tails


def _check_subjects(command: PlanningCommandCommand, request: str) -> list[str]:
    """The request's subjects as a check offers them: a sentence-opening one with its tail
    after it (``_opening_tails``), so that a check never has to choose between "Treat Ledger"
    and no record at all."""
    out: list[str] = []
    for subject in _request_subjects(command, request):
        tails = [request[s:e] for s, e in _opening_tails(request, _spans(request, subject))]
        out += [n for n in [subject, *tails] if n.lower() not in {o.lower() for o in out}]
    return out


def _subject_without_extra_words(request: str, a: int, b: int, spans: list[tuple[int, int]]) -> str | None:
    """The subject *request[a:b]* holds when the value is that subject with extra words run on: the first subject it
    holds whole, when only lower-case words come before it ("snapshot of IT Support", "access to Confluence") or
    several lower-case words or punctuation follow it ("IT Operations end to end", "X: what it exposes, ..."), else
    None. One extra lower-case word after it stays ("Bamboo HR admins": a word that is part of a name)."""
    held = sorted((s, e) for s, e in set(spans) if a <= s and e <= b and (s, e) != (a, b))
    if not held:
        return None
    s, e = held[0]
    if (e < len(request) and (request[e].isalnum() or request[e] == "_")) or \
            (s > 0 and (request[s - 1].isalnum() or request[s - 1] == "_")):
        return None                      # the subject is part of a longer identifier ("GitLab" of "GitLab_Admin")
    before, after = request[a:s].split(), request[e:b]
    if any(_name_like_word(w) for w in before):
        return None
    value = request[a:b]
    if any(f"{q}{value}{q}" in request for q in ("'", '"', "\u2018", "\u201c", "`")) or \
            (request[a - 1:a] in ("'", '"', "\u2018", "\u201c", "`")):
        return None                      # the request quotes the whole value: it is one name as written
    tail = after.split()
    punctuated = bool(after) and not after[0].isspace() and after[0] in ":,;.-\u2014"
    if s == a and len(request[s:e].split()) == 1 and not punctuated:
        return None                      # one capitalized word opening a sentence-case title ("Active identities who ...")
    if before or len(tail) >= 2 or punctuated:
        return request[s:e]
    return None


def _as_subject(request: str, text: str, subjects: list[str],
                kinds: frozenset[str] = frozenset()) -> tuple[str | None, str]:
    """``(argument, why)`` of a ``subject`` argument: ``text`` when it names a record of the
    request, the whole subject when it is one cut short, else None with ``why``.

    An occurrence of the argument (on word boundaries) names a record when it is a subject, or
    the tail of a sentence-opening one ("Ledger" of "Treat Ledger as ..."), or when it cuts no subject (each subject it overlaps it holds whole) and
    holds a subject or a name-like word (``_name_like_word``), or is a name written in lower case
    (``_plain_name``: "ann lee"): "I" of "Before I approve" names nothing. Otherwise an occurrence inside exactly one subject, at its start or end and cut
    there mid-word or at a space ("Payroll_Admin" of "Payroll_Administrator", "Ann" of "Ann
    Lee"), is that subject cut short; one cut at ``_`` or ``-`` ("Payroll" of "Payroll_Approver")
    may be a record of its own and is refused.
    """
    spans = [(s, e) for subject in subjects for s, e in _spans(request, subject)]
    parts = set(_opening_tails(request, spans))

    def joined(i: int) -> bool:   # request[i - 1] and request[i] are letters or digits of one word
        return 0 < i < len(request) and request[i - 1].isalnum() and request[i].isalnum()

    occurrences = _spans(request, text)
    whole_words = [(a, b) for a, b in occurrences if not joined(a) and not joined(b)]
    for a, b in whole_words:
        if (a, b) in parts:
            return text, ""
        trimmed = _subject_without_extra_words(request, a, b, spans)
        if trimmed is not None:
            return trimmed, ""
        cut = [(s, e) for s, e in spans if s < b and a < e and not (a <= s and e <= b)]
        if not cut and (any(a <= s and e <= b for s, e in spans) or any(_name_like_word(w) for w in text.split())
                        or _plain_name(request, a, b, kinds)):
            return text, ""
    snaps = set()
    for a, b in occurrences:
        for s, e in spans:
            if s <= a and b <= e and (s, e) != (a, b) and (s == a or e == b):
                # the subject is cut between request[edge - 1] and request[edge]
                edge, beyond = (b, request[b]) if s == a else (a, request[a - 1])
                if joined(edge) or beyond.isspace():
                    snaps.add(request[s:e])
    if len(snaps) == 1:
        return snaps.pop(), ""
    return None, "names no record of the request (none of its subjects, whole)"


def _request_subjects(command: PlanningCommandCommand, request: str) -> list[str]:
    from fastworkflow.agent.split_merge import request_subjects
    return request_subjects(request, command.identifiers)


def _screen(command: PlanningCommandCommand, raw: dict, request: str) -> tuple[dict, dict[str, str]]:
    """``(values, refused)``: the ``str`` and ``int`` arguments through the command parameter
    screen (``utils.signatures.screen_parameter_values``): quotes and angle brackets around a
    value removed, a made-up number replaced by the default; ``refused`` says why a placeholder
    or an out-of-range number is refused."""
    from fastworkflow.parameters.signatures import screen_parameter_values

    names = [p.name for p in command.params if p.type in ("str", "int") and raw.get(p.name) is not None]
    if not names or command.input_model is None:
        return {}, {}
    values = command.input_model.model_construct(**{n: raw[n] for n in names})
    screen = screen_parameter_values(values, request, request, names)
    refused = {n: "is a placeholder, not a value" for n in screen.placeholder_fields}
    refused.update({n: f"is out of range ({why})" for n, why in screen.out_of_range_problems.items()})
    return {n: getattr(values, n) for n in names}, refused


# The sentences the extraction model reads about a field after its description (bd fix-ftdl.6.5): each comes from
# one of the field's own declarations, and they are the same for a planning command and an execution command
# (``option_sentences``, the one builder; the parameter extractor and the planner's typed fill both read it).
IN_REQUEST_STR_SENTENCE = "Copy it exactly as the request writes it."   # copied_from_command_text on a str
IN_REQUEST_INT_SENTENCE = "A number the request writes."                # copied_from_command_text on an int
IN_REQUEST_LIST_SENTENCE = "Copy each item exactly as the request writes it."  # copied_from_command_text on a list[str]
SUBJECT_SENTENCE = "Copy it exactly as the request writes it. It must name a record the request names."  # record_named
BOOL_SENTENCE = "True only when the request asks for it."              # the bool type (default false)
BOOL_TRUE_SENTENCE = "False only when the request asks against it."    # the bool type, default true


def extraction_sentences(p: Param) -> tuple[str, ...]:
    """The sentences of the marks *p* carries, as ``check_arguments`` enforces them. A subject field's sentence already
    says what ``in_request`` says, so it stands alone: the extractor reads each rule once."""
    if p.subject:
        return (SUBJECT_SENTENCE,)
    if p.in_request:
        return ({"str": IN_REQUEST_STR_SENTENCE, "int": IN_REQUEST_INT_SENTENCE}.get(p.type, IN_REQUEST_LIST_SENTENCE),)
    return ()


def type_sentences(info) -> tuple[str, ...]:
    """The sentences of a field's type and format: a ``bool`` (``Optional[bool]`` too) is true only when the request asks
    for it (false only when the request asks against it, for a bool that defaults to true), a rule of the type on both
    kinds of command; a declared ``Field(pattern=...)`` is its format (``rules.pattern_sentence``)."""
    members = [a for a in get_args(info.annotation) if a is not type(None)] if rules.is_union(info.annotation) \
        else [info.annotation]
    sentences = []
    if members == [bool]:
        sentences.append(BOOL_TRUE_SENTENCE if info.default is True else BOOL_SENTENCE)
    if pattern := rules.pattern_sentence(info):
        sentences.append(pattern)
    return tuple(sentences)


def check_examples_by(input_model, where: str) -> None:
    """The load check of every ``examples_by`` of *input_model* (``field_options.ExamplesBy``; training data only, read
    by the synthetic utterance generator, never by a check, the extractor, a planner or a client). A
    PlanningCommandFormatError naming the field and the rule broken; cached per ``Input`` and *where*."""
    if (problem := _examples_by_problem(input_model, where)) is not None:
        raise PlanningCommandFormatError(problem)


@functools.lru_cache(maxsize=None)
def _examples_by_problem(input_model, where: str) -> str | None:
    from pydantic import TypeAdapter, ValidationError
    from typing import Annotated

    fields = input_model.model_fields
    for name, info in fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, dict) or EXAMPLES_BY not in extra:
            continue

        def problem(message: str) -> str:
            return str(_field_error(where, name, f"{EXAMPLES_BY} {message}"))

        value = extra[EXAMPLES_BY]
        if not isinstance(value, dict) or not value:
            return problem("must be a dict from the name of a choice (Literal) field of this Input to "
                           "{\"<value>\": [<example>, ...]}, such as {\"kind\": {\"group\": [\"Payroll Approvers\"]}}; "
                           f"got {_shown(value)}")
        if len(value) > 1:
            return problem(f"names {len(value)} choice fields ({', '.join(map(repr, value))}); name one: examples "
                           "that depend on several choices are not supported")
        [(choice, by_value)] = value.items()
        if choice == name:
            return problem(f"names this field itself ({choice!r}); name the choice field its examples depend on")
        if choice not in fields:
            return problem(f"names {choice!r}, which is not a field of this Input; its fields are "
                           f"{', '.join(map(repr, fields))}")
        values = _literal_values(fields[choice].annotation)
        if not values:
            return problem(f"names {choice!r}, which is not a choice field (its type is "
                           f"{python_utils.type_text(fields[choice].annotation)}, not a Literal)")
        if not isinstance(by_value, dict) or not by_value:
            return problem(f"for {choice!r} must be a non-empty dict from a value of {choice!r} to a list of "
                           f"examples; got {_shown(by_value)}")
        adapter = TypeAdapter(Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation)
        for key, examples in by_value.items():
            if key not in values:
                return problem(f"{choice}: {key!r} is not a value of {choice!r}")
            if isinstance(examples, (str, bytes)) or not isinstance(examples, (list, tuple)) or not examples:
                return problem(f"{choice}: the examples for {key!r} must be a non-empty list; got {_shown(examples)}")
            for example in examples:
                if example is None:
                    return problem(f"{choice}: an example for {key!r} is None; an example is a value of this field")
                try:
                    adapter.validate_python(example, strict=True)
                except ValidationError as error:
                    reason = error.errors()[0].get("msg", str(error)) if error.errors() else str(error)
                    return problem(f"{choice}: the example {example!r} for {key!r} is not a valid value of this "
                                   f"field ({python_utils.type_text(info.annotation)}): {reason}")
    return None


def option_params(input_model, where: str, *, planning: bool = False) -> tuple[Param, ...]:
    """A ``Param`` for each field of *input_model* (an ``Input``, of either kind of command) that carries
    a check option (2.46). Other fields are not checked. A load error (PlanningCommandFormatError) when a
    field's option keys are retired, unknown, or do not fit its type (spec 2.0 to 2.6). *planning* says the
    ``Input`` is a planning command's (an execution command's keys are checked the same way, with the
    keys only an execution command takes allowed)."""
    from types import SimpleNamespace

    out = []
    positional: list[str] = []
    for name, info in input_model.model_fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, dict):
            continue
        check_field_keys(name, extra, where, planning=planning)
        if any(key in extra for key in OPTION_KEYS):
            # the shape, the combinations and the type of every option the field carries (2.0)
            check_option_spec(name, extra, where, kind=_option_kind(info.annotation), planning=planning,
                              annotation=info.annotation)
        if extra.get(COMMAND_TEXT_AFTER_COMMAND_NAME) is True:
            positional.append(name)
        if not any(key in extra for key in CHECK_OPTIONS):
            continue
        shadow = SimpleNamespace(annotation=info.annotation, description=info.description, default=info.default,
                                 default_factory=info.default_factory, json_schema_extra=extra)
        out.append(_parse_param(name, _spec_from_field(name, shadow, where, planning=planning), where))
    if len(positional) > 1:
        raise PlanningCommandFormatError(f"{where}: {COMMAND_TEXT_AFTER_COMMAND_NAME} is on {' and '.join(positional)}; "
                                         "one field per Input takes the command text after the command name")
    check_examples_by(input_model, where)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def option_command(input_model, name: str, identifiers: tuple = ()) -> PlanningCommandCommand:
    """The checks of an execution command's ``Input`` as a ``PlanningCommandCommand`` holding only the fields
    that carry an option, so ``check_arguments`` and ``check_verdicts`` run on them unchanged (2.46)."""
    return PlanningCommandCommand(name=name, description="", params=option_params(input_model, name),
                                  identifiers=identifiers, input_model=input_model)


def execution_command(workflow_folderpath: str, qualified_name: str) -> PlanningCommandCommand | None:
    """An execution command's ``Input`` as a ``PlanningCommandCommand`` with a ``Param`` for EVERY field the planning
    grammar can state (str, bool, int, float, list[str], a text ``Literal``), so the one parameter clarification loop
    (``parameter_clarification``) asks for its values as it asks for a planning command's. A field of another type is
    left out (the loop does not ask for it: the command's own extraction stage does, as before). None when the command
    has no ``Input`` or none of its fields can be stated."""
    from fastworkflow import ModuleType, RoutingRegistry

    try:
        model = RoutingRegistry.get_definition(workflow_folderpath).get_command_class(
            qualified_name, ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:  # noqa: BLE001 - a command with no readable Input has no values to ask for
        return None
    return _execution_command(model, qualified_name) if model is not None else None


@functools.lru_cache(maxsize=None)
def _execution_command(input_model, name: str) -> PlanningCommandCommand | None:
    params = []
    for field_name, info in input_model.model_fields.items():
        try:
            params.append(_parse_param(field_name, _spec_from_field(field_name, info, name), name))
        except PlanningCommandFormatError:
            continue
    if not params:
        return None
    return PlanningCommandCommand(name=name, description="", params=tuple(params), input_model=input_model)


def classifier_alone_fields(input_model) -> frozenset[str]:
    """The fields of an ``Input`` (of either kind of command) marked ``classifier_decides``: the parameter
    extractor does not ask for them, as the typed call does not; the classifier decides them."""
    if input_model is None:
        return frozenset()
    return frozenset(p.name for p in option_params(input_model, getattr(input_model, "__name__", "Input"))
                     if p.alone)


def option_sentences(input_model) -> dict[str, tuple[str, ...]]:
    """THE prompt builder of a field (bd fix-ftdl.6.5): ``{field: sentences}`` the extraction model reads after the
    description of each field of an ``Input`` that has any, from the field's own declarations only: its marks
    (``extraction_sentences``), then its type and format (``type_sentences``). The same for both kinds of command; the
    parameter extractor (``InputForParamExtraction.create_signature_from_pydantic_model``) and the planner's typed fill
    (``typed_signature``) read it."""
    if input_model is None:
        return {}
    marks = {p.name: extraction_sentences(p)
             for p in option_params(input_model, getattr(input_model, "__name__", "Input"))}
    out = {}
    for name, info in input_model.model_fields.items():
        if sentences := (*marks.get(name, ()), *type_sentences(info)):
            out[name] = sentences
    return out


def extraction_sentences_for(workflow_folderpath: str | None, qualified_name: str | None) -> dict[str, tuple[str, ...]]:
    """``option_sentences`` of planning command *qualified_name*'s ``Input``. Empty without a workflow."""
    if not workflow_folderpath or not qualified_name:
        return {}
    command = planning_command(workflow_folderpath, qualified_name)
    return {} if command is None else option_sentences(command.input_model)


def verify_field_names(workflow_folderpath: str) -> tuple[str, ...]:
    """``<command>.<field>`` for each planning command's field that carries a classifier check: the checks
    that need ``JEV_API_KEY`` (the INFO line at load, design 3.9)."""
    commands = resolve_field_sources(workflow_folderpath).planning_commands
    planning = tuple(f"{qname}.{p.name}" for qname, command in sorted(commands.items())
                     for p in command.params if p.verify)
    return planning + tuple(f"{qname}.{p.name}" for qname, command in sorted(_execution_option_commands(
        workflow_folderpath).items()) for p in command.params if p.verify)


def _execution_option_commands(workflow_folderpath: str) -> dict[str, PlanningCommandCommand]:
    """``{command: option_command}`` for each execution command of the workflow whose ``Input`` carries an option."""
    from fastworkflow.observation_offloading.facts import workflow_signatures

    planning = resolve_field_sources(workflow_folderpath).planning_commands
    out = {}
    for name, signature in workflow_signatures(workflow_folderpath).items():
        model = getattr(signature, "Input", None)
        if not isinstance(model, type) or name in planning:
            continue
        command = option_command(model, name)
        if command.params:
            out[name] = command
    return out


def check_arguments(command: PlanningCommandCommand, raw: dict, request: str,
                    verdicts: dict | None = None,
                    vocabulary: WorkflowVocabulary | None = None) -> tuple[dict, list[str]]:
    """The guarded, complete arguments of a planning command call; ``(args, notes)``.

    - a one-element list given for an argument (``["Payroll"]``, or that list written as JSON
      text) stands for its element;
    - a ``str`` or ``int`` argument first goes through the command parameter screen
      (``utils.signatures.screen_parameter_values``, as a command's parameters do): quotes
      or angle brackets around the whole value are removed, a bracketed placeholder (``<name>``) or an
      out-of-range number (a negative ``count``, a declared ``ge``/``le`` bound) is refused (an optional
      argument takes its default, a required one fails the call), and the extractor's "not given" number or a
      made-up extreme counts as missing;
    - a missing or empty argument takes its default; a required one fails the call;
    - an enum must be one of its values, an int an integer, a float a number (the extractor's
      "not given" marker counts as missing), a bool true/false; a list[str] is its elements
      (cleaned), a list written as JSON text or comma-separated text;
    - a str argument loses surrounding quotes; a value that is entirely one template token, only punctuation, only a
      pronoun or only a word of the workflow's vocabulary is refused (a pronoun the request ties to one subject
      becomes that subject); otherwise the value stands as the extractor gave it;
    - a MARKED argument (``copied_from_command_text`` or ``record_named_in_command_text``, ``Param.in_request``) is the
      one strict check: a ``str`` must be written in ``request`` at word boundaries (case-insensitive; any Unicode
      dash matches "-") and takes the request's own spelling; one the model cut short inside an identifier match of the
      request (``command.identifiers``) becomes the whole match; a ``record_named`` one must name a record of the
      request (``_as_subject``); an ``int`` must be a number the request writes; a list[str] keeps the items the
      request writes. A marked value given and not written fails the call, required or optional; a marked value not
      given is not checked. Two marked str arguments may hold one value only when the user writes it as often (an
      optional one beyond that count takes its default with a note, ``ask.left_empty_line``; a required one fails);
    - ``verdicts`` (``{param: (value, probability)}``, from ``check_verdicts``): a sure
      verdict on a ``classifier_confirms`` parameter replaces its argument, and stands in for a missing
      or invalid enum (for a subject, ``NO_SUBJECT`` resets it, or fails a required one);
      a ``classifier_decides`` bool ignores any argument given for it: a sure verdict decides it,
      and it keeps its default otherwise.

    An argument that only applies when others take some value is reset afterwards, by the
    planning command's model validators (``_validate_arguments``).

    ``notes`` says what a guard changed. A required argument the guards refuse raises
    PlanningCommandInputError (the planner fills the call once more, ``typed_planning_call``); a sure check
    that the request names no such argument raises PlanningCommandCallError.
    """
    checked = one_guard.check_plan(command, raw, request, verdicts, vocabulary)
    if (stop := checked.stop) is not None:
        error = (PlanningCommandCallError if stop.call else PlanningCommandInputError)(stop.message)
        error.code = stop.code
        error.gap = stop
        error.values = dict(checked.values)   # the arguments as the guards held them when they refused
        raise error
    return checked.values, checked.notes


def command_names(available_commands: str) -> frozenset[str]:
    """Command names of the planner's catalog text (its ``- <name>`` lines)."""
    return frozenset(_COMMAND_LINE.findall(available_commands or ""))


def _without_placeholders(text: str) -> str:
    """``text`` without ``<...>`` placeholders and the ``key=`` holding one."""
    cleaned = _BARE_PLACEHOLDER.sub("", _ARG_PLACEHOLDER.sub("", text))
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:")


#: Words of a command name that say what it does, not what it is about ("open", "list", "uid").
GENERIC_COMMAND_WORDS = frozenset("""open list show find get go up by uid id set add run view read all of to for from
with in on at the a an new next first back current""".split())
_WORD = re.compile(r"[a-z]+")
#: Words that say nothing of what a request asks for ("every", "please", "want").
_STOPWORDS = frozenset("""about also and any are been but can did does each else every has have here how into its
just like many more most much need not now only other over please same some such than that the their them then there
these they this those too very want was were what when where which while who whom whose why will with would you
your""".split())


_SUFFIXES = ("ities", "ity", "ions", "ion", "ings", "ing", "ers", "er", "ies", "ive", "ed", "es", "e", "s", "y")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _same_stem(a: str, b: str) -> bool:
    """Two words of one stem once a common ending is cut off ("holders"/"hold", "nesting"/"nested",
    "sensitivity"/"sensitive"): one stem starts the other and is at least four letters long."""
    x, y = sorted((_stem(a), _stem(b)), key=len)
    return len(x) >= 4 and y.startswith(x)


def _open_request_words(request: str, expanded: list[str]) -> set[str]:
    """The request's content words that no word of the expansion's text shares a stem with: the
    parts of the request the expansion does not cover."""
    covered = {w for s in expanded for w in _WORD.findall(s.lower())}
    return {w for w in _WORD.findall(request.lower())
            if len(w) >= 3 and w not in _STOPWORDS and not any(_same_stem(w, c) for c in covered)}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower())
            if len(w) >= 3 and w not in _STOPWORDS and w not in GENERIC_COMMAND_WORDS}


def _supported(command: str, open_words: set[str], text: str = "") -> bool:
    """A word of the command's name, or of ``text`` (what the command says it does: its
    description, its sample utterances), shares a stem with an open request word
    (``GENERIC_COMMAND_WORDS`` and stop words aside)."""
    words = _content_words(command.replace("_", " ")) | _content_words(text)
    return any(_same_stem(w, r) for w in words for r in open_words)


_CATALOG_ENTRY = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*\n((?:[ \t]+\S.*(?:\n|$))*)", re.M)


def command_descriptions(available_commands: str) -> dict[str, str]:
    """``{name: description}`` of the planner's catalog text: each ``- <name>`` line's first
    indented line, cut at its first sentence (what the command does, not the cautions and
    navigation lines the catalog adds after it). A name listed more than once (one command
    in several contexts) gets each description."""
    out: dict[str, list[str]] = {}
    for name, body in _CATALOG_ENTRY.findall(available_commands or ""):
        first = body.strip().splitlines()[0].strip() if body.strip() else ""
        # A ``parameters:`` line is the typed call's listing (typed_planner_commands_text), not a description.
        if first and not first.endswith(":") and not first.startswith("parameters:"):
            out.setdefault(name, []).append(re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0])
    return {name: " ".join(texts) for name, texts in out.items()}


def command_utterances(workflow_folderpath: str) -> dict[str, list[str]]:
    """``{name: plain_utterances}`` of the workflow's commands, from its routing definition
    (a name in several contexts gets the utterances of each). Planning commands count too:
    their utterances come from their ``Signature``, as an execution command's do. Empty when
    the definition cannot be read; never raises."""
    from fastworkflow.core.command_routing import RoutingRegistry

    out: dict[str, list[str]] = {}
    try:
        crd = RoutingRegistry.get_definition(workflow_folderpath)
        qualified = {q for context, names in crd.contexts.items()
                     for q in (*names, *crd.get_planning_command_names(context))}
        for q in sorted(qualified):
            try:
                meta = crd.command_directory.get_utterance_metadata(q)
            except Exception:  # noqa: BLE001 - a command without utterances adds nothing
                meta = None
            if meta and meta.plain_utterances:
                out.setdefault(q.split("/")[-1], []).extend(meta.plain_utterances)
    except Exception:  # noqa: BLE001 - the command names alone then decide
        from fastworkflow.utils.logging import logger
        logger.warning(f"could not read command utterances of {workflow_folderpath}", exc_info=True)
    return out


def command_texts(available_commands: str, workflow_folderpath: str | None = None) -> dict[str, str]:
    """``{name: words}`` for ``filter_extra_steps``: the words a command says of itself again
    and again: those whose stem recurs in at least two of its texts (its catalog description,
    ``command_descriptions``, and, with ``workflow_folderpath``, each of its plain_utterances).
    A word one utterance happens to use ("check", "whether", "full picture") says nothing
    of what the command is about; a word several do ("invoices" in "show every invoice
    this customer has" and "which invoices are still open") does. A command with one text only
    gives that text's words. A planning command that no execution command shares keeps every
    word of its description too, so its utterances can add words to its hint but never take one away."""
    descriptions = command_descriptions(available_commands)
    sentences = {name: [desc] for name, desc in descriptions.items()}
    planning_only: set[str] = set()
    if workflow_folderpath:
        for name, utterances in command_utterances(workflow_folderpath).items():
            sentences.setdefault(name, []).extend(utterances)
        planning_only = _planning_only_names(workflow_folderpath)
    out = {}
    for name, texts in sentences.items():
        words = [_content_words(t) for t in texts]
        counts: dict[str, int] = {}
        for ws in words:
            for stem in {_stem(w) for w in ws}:
                counts[stem] = counts.get(stem, 0) + 1
        least = 2 if len(texts) > 1 else 1
        found = {w for ws in words for w in ws if counts[_stem(w)] >= least}
        if name in planning_only:
            # A planning command's utterances only add to the words its description gave it:
            # the hint it had before its utterances were read is kept whole.
            found |= _content_words(descriptions.get(name, ""))
        out[name] = " ".join(sorted(found))
    return out


def _planning_only_names(workflow_folderpath: str) -> set[str]:
    """The simple names of the workflow's planning commands that no execution command shares.
    Empty when the routing definition cannot be read."""
    from fastworkflow.core.command_routing import RoutingRegistry

    try:
        crd = RoutingRegistry.get_definition(workflow_folderpath)
    except Exception:  # noqa: BLE001 - without the definition no name is known to be planning
        return set()
    execution = {q.split("/")[-1] for names in crd.contexts.values() for q in names}
    planning = {q.split("/")[-1] for context in crd.contexts
                for q in crd.get_planning_command_names(context)}
    return planning - execution


def filter_extra_steps(extra, expanded: list[str], names: frozenset[str], planning_command_name: str,
                       request: str = "", notes: list[str] | None = None,
                       texts: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` of the model's steps_before/steps_after entries.

    An entry first loses its ``<...>`` placeholders (with the ``key=`` holding one): the
    model cannot know a value the run finds, and the agent passes a placeholder on as one
    (``list_logins login_id=<the person's uid>`` becomes ``list_logins``). Dropped: empty
    entries, lines copied from the planning command's template (``_is_template_line``: the expansion
    already writes them for these arguments, or they do not apply to them), entries
    holding a uid-shaped token ``request`` does not write, entries naming
    the planning command itself (the expansion already runs it), entries naming no available command,
    and entries whose every command the expansion already runs. With ``request`` given, an
    entry is also dropped unless one of the commands it names outside the expansion is asked
    for by a part of the request the expansion leaves open: a word of the command's name
    (``GENERIC_COMMAND_WORDS`` aside) shares a stem with a request word that no word of the
    expansion's text shares one with ("show_tags" for "... and its tags" when the expansion
    says nothing of tags; not a navigation or lookup step the model adds around the
    expansion). ``texts`` (``{name: words}``, see ``command_texts``) widens a command's name
    with what the command says of itself, so a request that names it by a synonym its
    description or sample utterances use still asks for it ("every invoice" for
    ``list_bills`` whose utterances say "invoices"). With no command names known only the
    first four rules apply. ``notes`` (when given) gets one line per placeholder removed.
    """
    expanded_cmds = {w for s in expanded for w in _IDENT.findall(s) if w in names}
    planning_command_stem = planning_command_name.rpartition("/")[2]
    lowered = (request or "").lower()
    open_words = _open_request_words(request, expanded) if request else None
    kept, dropped = [], []
    texts = texts or {}
    for entry in extra or []:
        text = str(entry).strip()
        if not text:
            continue
        if _is_template_line(text) or any(uid.lower() not in lowered for uid in _UID.findall(text)):
            dropped.append(text)
            continue
        cleaned = _without_placeholders(text)
        if not cleaned:
            dropped.append(text)
            continue
        if cleaned != text and notes is not None:
            notes.append(f"extra step {text!r} -> {cleaned!r} (placeholder removed)")
        idents = set(_IDENT.findall(cleaned))
        cmds = idents & names
        if (planning_command_name in cleaned
                or (planning_command_stem in idents and planning_command_stem not in names)
                or (names and (not cmds or cmds <= expanded_cmds))):
            dropped.append(text)
        elif names and open_words is not None and not any(_supported(c, open_words, texts.get(c, ""))
                                                             for c in cmds - expanded_cmds):
            dropped.append(text)
        else:
            kept.append(cleaned)
    return kept, dropped


# ---------------------------------------------------------------- the planner call

def _field_desc(p: Param, sentences: tuple[str, ...] = ()) -> str:
    """A field of the planner's typed fill: its description, the sentences of its declarations (``option_sentences``,
    the extractor's own) and whether it is required."""
    desc = ". ".join([(p.desc or p.name).rstrip(". "), *(s.rstrip(".") for s in sentences)])
    return desc + (" (required)" if p.required else f" (default {json.dumps(_json_value(p))})")


def _json_value(p: Param):
    """``p.default`` as JSON reads it (a list[str] default is a tuple)."""
    return list(p.default) if p.type == "list[str]" else p.default


def typed_signature(command: PlanningCommandCommand):
    """The planning command's dspy signature: user_query in; one typed field per parameter, plus
    steps_before / steps_after, out."""
    import dspy

    fields: dict = {"user_query": (str, dspy.InputField())}
    sentences = option_sentences(command.input_model)
    for p in command.params:
        if p.alone:   # verify "alone": the classifier decides it, the planner is not asked
            continue
        kind = {"str": str, "bool": bool, "int": int, "float": float, "list[str]": list[str]}.get(p.type)
        if p.type == "enum":
            kind = Literal[p.values]
        fields[p.name] = (kind, dspy.OutputField(desc=_field_desc(p, sentences.get(p.name, ()))))
    fields["steps_before"] = (list[str], dspy.OutputField(
        desc="plain command steps to run before the skill, only for parts of the request it does not cover; "
             "only values the request writes, never a <placeholder> or an id; usually []"))
    fields["steps_after"] = (list[str], dspy.OutputField(
        desc="plain command steps to run after the skill, only for parts of the request it does not cover; "
             "only values the request writes, never a <placeholder> or an id; usually []"))
    instructions = (
        "Plan the user_query by calling one skill command and, only if needed, plain command steps.\n"
        "Fill the skill's parameters from the request, as each parameter's description says; an enum is what the "
        "request is about; leave a parameter empty to take its default.\n"
        "Code expands the skill into its command steps. steps_before / steps_after are only for parts of the "
        "request the expansion does not cover, each a short sentence naming one available command; never "
        "repeat a command the expansion already runs; usually both are empty.\n\n"
        f"Skill command:\n{command.signature_text()}")
    return dspy.make_signature(fields, instructions)


def _check_options(p: Param, subjects: list[str] = ()) -> tuple[tuple[str, str], ...]:
    if p.type == "str":
        named = [s for s in subjects if s.lower() != NO_SUBJECT][:MAX_SUBJECT_OPTIONS]
        return tuple((s, f"the request names {s} as the {p.name}") for s in named) + (
            (NO_SUBJECT, f"the request names no {p.name}; each name it writes is something else"),)
    if p.type == "bool":
        if p.default is True:
            # fix-jd9o.9: the value form, not "asks against" (a double negative Jev read near coin-flip:
            # 0.47 on mark_as_complete.is_complete; 0.70 -> 0.97 across default-true bools).
            return (("true", f"true: the request does not mention {p.name}, or says or implies it"),
                    ("false", f"false: the request says not to, or says the opposite of {p.name}"))
        return (("true", f"yes: {p.desc or p.name}"),
                ("false", "no: the request does not ask for this, or says not to"))
    means = dict(p.means)
    return tuple((v, means.get(v, v)) for v in p.values)


def _distinct_labels(options: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """The options with one per label (fix-variance): ``Choice[...]`` refuses two options whose values have the
    same string label, so the first occurrence of each label is kept and the later ones are dropped."""
    kept: dict[str, tuple[str, str]] = {}
    for value, description in options:
        kept.setdefault(str(value), (value, description))
    return tuple(kept.values())


def _check_one(lm, command: PlanningCommandCommand, p: Param, request: str) -> tuple[object, float]:
    import dspy
    from fastworkflow.utils import dspy_utils
    from dspy.experimental import Choice

    if p.type == "bool":
        question = (f"Which value should the argument '{p.name}' take? It means: {p.desc or p.name}."
                    if p.default is True
                    else f"Does the request ask for this: {p.desc or p.name}?")
    elif p.type == "str":
        question = f"Which record named in the request is the {p.name}? {p.desc}".strip()
    else:
        question = f"Which {p.name} does the request mean? {p.desc}".strip()
    options = _distinct_labels(_check_options(p, _check_subjects(command, request) if p.type == "str" else ()))
    signature = dspy.make_signature({
        "request": (str, dspy.InputField(desc="the user's request")),
        "job": (str, dspy.InputField(desc="the job the request is planned as")),
        "value": (Choice[options], dspy.OutputField(desc=question)),
    }, "Decide one argument of the job from what the request says.")
    with dspy_utils.role_context("JEV", lm):
        choice = dspy.Predict(signature)(request=request, job=command.description).value
    value = str(choice.value)
    probability = float((choice.probabilities or {}).get(value, choice.confidence))
    return (value == "true" if p.type == "bool" else value), probability


def check_verdicts(lm, command: PlanningCommandCommand, request: str) -> dict[str, tuple[object, float]]:
    """``{param: (value, probability)}``: ``lm`` (a classifier) decides each classifier-checked
    parameter from the request alone, the parameters in parallel. A failed check is left out
    (the planner's argument then stands); never raises."""
    from fastworkflow.utils.logging import logger

    params = [p for p in command.params if p.verify]
    if any(p.type == "str" for p in params) and not _request_subjects(command, request):
        params = [p for p in params if p.type != "str"]   # nothing to choose from
    if lm is None or not params:
        return {}

    def one(p):
        try:
            return p.name, _check_one(lm, command, p, request)
        except Exception as error:  # noqa: BLE001 - a check is an extra; the planner's argument stands
            logger.info(f"planning command argument {command.name}.{p.name} not checked, the planner's value stands: "
                        f"{type(error).__name__}: {error}")
            return None
    # Each check runs under a copy of this thread's context: the observability host (dspy_logger)
    # and the module frames are ContextVars, unset in a bare worker, so the classifier calls would
    # leave no fw.llm.call span under the turn. One copy per submission: a Context can be entered
    # by one thread at a time.
    with ThreadPoolExecutor(len(params)) as pool:
        futures = [pool.submit(tracing.traced_call(one, p)) for p in params]
        return dict(r for r in (f.result() for f in futures) if r is not None)


def verdict_notes(verdicts: dict | None) -> list[str]:
    """One note per verdict read, for the planner span's ``skill_notes`` (sure or not, and the value and
    probability it had): the verdicts are readable there whether or not an argument changed."""
    notes = []
    for name, (value, probability) in (verdicts or {}).items():
        sure = "sure" if probability >= VERIFY_THRESHOLD else "not sure, not used"
        notes.append(f"verify {name}={value!r} ({probability:.2f}, {sure})")
    return notes


#: Appended to the request when the planner fills a planning command call again after its rules refused
#: the first one (``typed_planning_call``).
REFUSED_CALL_NOTE = ("(A first call of this skill was refused: {refusal}. Fill the call again from the "
                     "request above, correcting that.)")


@dataclass
class PlanningCommandCallPlan:
    plan: str
    args: dict
    notes: list[str]
    prediction: object
    #: The phases that run, in order (``planning_phases.Phase``, see ``call_phases``): a phased
    #: planning command's (``plan()``), or, for a flat call, one phase named after the planning command with its
    #: steps. And the extra steps the request asks for before and after the planning command's own.
    phases: list | None = None
    steps_before: list[str] = field(default_factory=list)
    steps_after: list[str] = field(default_factory=list)
    #: The verify verdicts the call read (``check_verdicts``), kept so a call planned again on a failed first phase
    #: (``reextracted_arguments``) checks its arguments the same way. None when there were none.
    verdicts: dict | None = None
    #: The advisory (``parameters.check.not_written_fields``): the unmarked ``str`` and ``int`` arguments whose value the
    #: request does not write, as field names. Recorded on the planner span; it changes nothing.
    not_in_request: list[str] = field(default_factory=list)


def no_argument_call(command: PlanningCommandCommand) -> PlanningCommandCallPlan:
    """The call of a planning command with no parameters: nothing for a typed planning call to fill, so no
    model call is made. Its phases are ``call_phases``'s (always at least one); it has no
    steps before or after, as those are steps the request needs around a call with arguments."""
    return PlanningCommandCallPlan(plan="", args={}, notes=[], prediction=None,
                         phases=call_phases(command, {}))


def extracted_arguments(command: PlanningCommandCommand, query: str, workflow=None, *, note: str = "",
                        resets: dict | None = None, unparsed: list | None = None) -> dict:
    """The planning command's arguments as the shared parameter extractor gives them for *query*, unchecked
    (fix-48np): ``InputForParamExtraction.extract_parameters`` on the planning command's ``Input``, on
    LLM_PARAM_EXTRACTION, the extractor an execution command's parameters get (``parameters.fill.fill``, the one call of
    it; a *note* is read after *query*). ``typed_planning_call`` checks them. With *resets* (a dict) the fields the
    extractor's own "is this field given" check reset to not given are written into it (name -> probability). With
    *unparsed* (a list) True is appended when the extractor's answers could not be parsed into the fields' types
    (``Filled.unparsed``)."""
    from fastworkflow.parameters import fill

    if command.input_model is None:
        raise PlanningCommandCallError(f"{command.name} has no Input to extract its arguments into")
    folder = getattr(workflow, "folderpath", None)
    filled = fill.fill(None, command.input_model, command.name, folder, text=query, note=note)
    params = filled.params
    if resets is not None and filled.reset_not_given:
        resets.update(filled.reset_not_given)
    if unparsed is not None and filled.unparsed:
        unparsed.append(True)
    return {p.name: _given_or_none(getattr(params, p.name, None)) for p in command.params}


def _given_or_none(value):
    """*value*, or None for the extractor's "not given" text (``NOT_FOUND``, ``<UNKNOWN>``): no value, so the
    guards take the default or refuse a required argument."""
    from fastworkflow.parameters.signatures import NOT_FOUND, is_unknown_marker

    if isinstance(value, str) and (value.strip() == NOT_FOUND or is_unknown_marker(value)):
        return None
    return value


def _call_text(command: PlanningCommandCommand, args: dict, steps: list[str]) -> str:
    """The filled call as the step proposal reads it: the call with its arguments, and its expansion."""
    shown = ", ".join(f"{name}={json.dumps(value, default=str)}" for name, value in args.items())
    return f"{command.name}({shown})\nIt runs these steps:\n" + "\n".join(f"- {step}" for step in steps)


def step_signature(command: PlanningCommandCommand):
    """The step proposal's dspy signature (fix-48np): the request and the filled planning command call (its
    arguments fixed, and the steps code expands it into) in; steps_before / steps_after out."""
    import dspy

    fields: dict = {
        "user_query": (str, dspy.InputField()),
        "skill_call": (str, dspy.InputField(
            desc="the skill command call planned for the request, its arguments already filled, and the "
                 "command steps code expands it into")),
        "steps_before": (list[str], dspy.OutputField(
            desc="plain command steps to run before the skill, only for parts of the request it does not cover; "
                 "only values the request writes, never a <placeholder> or an id; usually []")),
        "steps_after": (list[str], dspy.OutputField(
            desc="plain command steps to run after the skill, only for parts of the request it does not cover; "
                 "only values the request writes, never a <placeholder> or an id; usually []")),
    }
    instructions = (
        "The user_query is planned as one skill command call, already filled; code expands it into its "
        "command steps (skill_call). steps_before / steps_after are only for parts of the request the "
        "expansion does not cover, each a short sentence naming one available command; never repeat a "
        "command the expansion already runs; usually both are empty.\n\n"
        f"Skill command:\n{command.signature_text()}")
    return dspy.make_signature(fields, instructions)


def _typed_fallback(command: PlanningCommandCommand, request: str, available_commands: str, raw: dict,
                    refusal: PlanningCommandInputError, checked) -> tuple[dict, list[str]]:
    """The arguments when the shared extractor, after its retry, still leaves a required argument missing or
    refused (fix-48np): the planner's typed fill (``typed_signature``, the call that filled them before) gives
    those arguments only; every value the extractor gave that the checks did not refuse is kept, and the result
    goes through the same checks (``checked``: ``check_arguments``, the hook, ``check_input``), so a value they
    refuse is still refused. Raises *refusal* when no required argument is missing or refused, or the typed
    fill gives none of them."""
    import dspy
    from fastworkflow.utils.logging import logger

    stopped_at = refusal.gap.name if refusal.gap is not None else None
    gaps = [p.name for p in command.params if p.required and not p.alone
            and (_empty(raw.get(p.name)) or p.name == stopped_at)]
    if not gaps:
        raise refusal
    from dspy.utils.exceptions import AdapterParseError

    try:
        prediction = dspy.ChainOfThought(typed_signature(command))(
            user_query=request, available_commands=available_commands)
    except AdapterParseError as exc:
        # an answer the typed fill's own types refuse (a value outside an enum): the refusal stands, and may be
        # asked. Any other failure (an unreachable model) is not a refusal and is raised as it is.
        logger.info(f"planning command call {command.name}: the typed fill fallback gave no usable answer")
        raise refusal from exc
    filled ={name: _given_or_none(getattr(prediction, name, None)) for name in gaps}
    filled = {name: value for name, value in filled.items() if not _empty(value)}
    if not filled:
        raise refusal
    args, notes = checked({**raw, **filled})
    logger.info(f"planning command call {command.name}: {', '.join(sorted(filled))} from the typed fill "
                f"(the extractor left it: {refusal})")
    notes.insert(0, f"typed fill fallback for {', '.join(sorted(filled))}: {refusal}")
    return args, notes


def checked_arguments(command: PlanningCommandCommand, raw: dict, request: str, workflow=None,
                      verdicts: dict | None = None, *, user_text: str | None = None) -> tuple[dict, list[str]]:
    """``(args, notes)``: *raw* through every check a typed planning call runs on its arguments, in order:
    ``check_arguments`` (the guards, with ``verdicts`` when the call has them), then ``_validate_arguments``
    (the ``Input``'s model validators, the roles).

    *request* is what the guards read. *user_text* is the validators' ``command_text`` when the two differ: a caller
    whose *request* also holds text the framework wrote (the answers of the turn's earlier parts, so that a value an
    earlier part found counts as written) passes the user's own words here, because a validator reads its
    ``command_text`` as what the user said (bd fix-ftdl.10). Not given, the validators read *request*.
    Raises PlanningCommandInputError when a check refuses them. When the duplicate rule refuses and the validators,
    run on the same arguments, refuse too, the validators' refusal is raised (``one_guard.validators_first``)."""
    text = request if user_text is None else user_text
    try:
        args, notes = check_arguments(command, raw, request, verdicts)
    except PlanningCommandInputError as refused:
        if one_guard.validators_first(refused.gap) and command.input_model is not None:
            try:
                check_input(command, refused.values, text, workflow)
            except PlanningCommandInputError as theirs:
                raise theirs from None
            except PlanningCommandCallError:   # a validator that raised is no refusal: the duplicate rule's stands
                pass
        raise refused
    return _validate_arguments(command, args, text, workflow, notes), notes


def typed_fallback_arguments(command: PlanningCommandCommand, request: str, available_commands: str, raw: dict,
                             refusal: PlanningCommandInputError, workflow=None) -> tuple[dict, list[str]]:
    """The typed fill fallback of ``typed_planning_call`` (``_typed_fallback``), for a caller that extracted *raw*
    itself (fix-48np.1: the sub-task draft): the planner's typed fill gives the required arguments *refusal* leaves
    missing or refused, and the result goes through ``checked_arguments``. Runs in the caller's dspy context (the
    planner LM and the adapter that reads ``available_commands``). Raises *refusal* when it fills nothing."""
    return _typed_fallback(command, request, available_commands, raw, refusal,
                           lambda filled: checked_arguments(command, filled, request, workflow))


def typed_planning_call(command: PlanningCommandCommand, request: str, available_commands: str,
                    check_lm=None, workflow=None) -> PlanningCommandCallPlan:
    """Plan ``request`` as a call of ``command``: its arguments, the guards, the expansion, the extra steps.

    A planning command with no parameters is not planned with a model call (``no_argument_call``).

    The arguments come from the shared parameter extractor (``extracted_arguments``, fix-48np: the extractor
    execution commands use, on LLM_PARAM_EXTRACTION), and are checked as before: with ``check_lm`` (a
    classifier LM) the planning command's classifier-checked parameters are decided again from the request, alongside
    the extraction (``check_verdicts``); ``check_arguments`` guards them; a planning command file's
    model validators then see the checked arguments (``_validate_arguments``; ``workflow`` is passed to
    them), and the ``Input``'s rules (``check_input``). When those rules or the guards refuse
    the arguments (PlanningCommandInputError: a made-up name, a missing or invalid value), the extractor
    reads the request once more, with the refusal after it (``argument_checks.REFUSED_EXTRACTION_NOTE``);
    the guards still read the request alone. When a required argument is still missing or refused after that,
    the planner's typed fill gives that argument only (``_typed_fallback``), and the same checks run on it.
    Raises on any other failure, or when the checks still refuse (the caller falls back to the template path).

    The planner (the caller's dspy context: the planner LM and adapter; ``available_commands`` reaches the
    adapter as extra input, as for the text planner) then only proposes the steps before and after the
    call, given the filled call and its expansion (``step_signature``).
    """
    if not command.params:
        return no_argument_call(command)
    import dspy
    from fastworkflow.parameters import pipeline
    from fastworkflow.parameters.argument_checks import REFUSED_EXTRACTION_NOTE

    seen: dict = {"verdicts": None, "reset": {}, "unparsed": []}

    def checked(raw: dict) -> tuple[dict, list[str]]:
        return checked_arguments(command, raw, request, workflow, seen["verdicts"])

    def validate(raw: dict, attempt: int) -> pipeline.Verdict:
        try:
            return pipeline.Verdict(detail=checked(raw))
        except PlanningCommandInputError as refused:
            kind = _input_error_kind(refused, command, raw, request, seen["verdicts"])
            if kind == pipeline.KIND_MISSING and attempt == 1 and seen["reset"]:
                kind = pipeline.KIND_REFUSED   # a value the extractor gave, a check then reset to not given
            return pipeline.Verdict(kind, str(refused), refused,
                                    open=_open_arguments(command, refused, raw, request, seen["verdicts"],
                                                         seen["reset"] if attempt == 1 else {}))

    with ThreadPoolExecutor(1) as pool:
        checks = pool.submit(tracing.traced_call(check_verdicts, check_lm, command, request))

        def fill_text(note: str, attempt: int) -> dict:
            raw = extracted_arguments(command, request, workflow, note=note, resets=seen["reset"] if attempt == 1 else None,
                                      unparsed=seen["unparsed"] if attempt == 1 else None)
            if attempt == 1:
                seen["verdicts"] = checks.result()
            return raw

        # the extractor reads the request once more with the refusal after it; the guards still read the request alone
        extracted = pipeline.extract(
            fill_text, validate=validate, unparsed=lambda: bool(seen["unparsed"]),
            retry_note=REFUSED_EXTRACTION_NOTE,
            unparsed_refusal=pipeline.unparsed_refusal(command.input_model) if command.input_model is not None
            else pipeline.UNPARSED_REFUSAL)
    verdicts, raw = seen["verdicts"], extracted.params
    if extracted.final.ok:
        args, notes = extracted.final.detail
    else:
        try:
            args, notes = _typed_fallback(command, request, available_commands, raw, extracted.final.detail, checked)
        except PlanningCommandInputError as final:
            # ``check_arguments`` stops at the first gap: work out every one, so the question names them all
            # and the values the call already has are kept.
            final.gaps = argument_gaps(command, raw, request, verdicts, workflow)
            raise
    if extracted.retried:
        notes.insert(0, f"first call refused, planned again: {extracted.first.detail}")
    steps = expand(command, args)
    prediction = dspy.ChainOfThought(step_signature(command))(
        user_query=request, skill_call=_call_text(command, args, steps), available_commands=available_commands)
    names = command_names(available_commands)
    texts = command_texts(available_commands, getattr(workflow, "folderpath", None))
    before, dropped_before = filter_extra_steps(getattr(prediction, "steps_before", None), steps, names,
                                                command.name, request, notes, texts)
    after, dropped_after = filter_extra_steps(getattr(prediction, "steps_after", None), steps, names,
                                              command.name, request, notes, texts)
    notes += [f"dropped extra step {s!r}" for s in dropped_before + dropped_after]
    notes += verdict_notes(verdicts)   # the verdicts read, whether or not they changed an argument
    phases = call_phases(command, args)
    return PlanningCommandCallPlan(plan=number(before + steps + after), args=args, notes=notes, prediction=prediction,
                         phases=phases, steps_before=list(before), steps_after=list(after), verdicts=verdicts,
                         not_in_request=_not_in_request(command, args, request))


def _not_in_request(command: PlanningCommandCommand, args: dict, request: str) -> list[str]:
    """The advisory of a typed call (``parameters.check.not_written_fields`` on its checked arguments); never fails a call."""
    try:
        return one_guard.not_written_fields(command.input_model, args, request) if command.input_model is not None else []
    except Exception as exc:  # noqa: BLE001 - recording only
        logger.warning(f"planning command {command.name}: the not-in-request advisory failed: {exc!r}")
        return []


def reextracted_arguments(command: PlanningCommandCommand, request: str, observation: str, args: dict,
                          workflow=None, verdicts: dict | None = None) -> dict | None:
    """The arguments of a call whose first phase's lookup found nothing, extracted once more (fix-variance):
    the shared extractor reads *request* with *observation* (what that phase found) after it, as a refusal is
    put after the request (``argument_checks.PHASE_FAILED_EXTRACTION_NOTE``), and the result goes through the
    same checks. Returns None when the checks refuse them or they equal *args* (the call is not restarted).
    Raises what the extractor raises."""
    from fastworkflow.parameters import pipeline
    from fastworkflow.parameters.argument_checks import PHASE_FAILED_EXTRACTION_NOTE

    def fill_text(note: str, attempt: int) -> dict:
        return extracted_arguments(command, request, workflow, note=note)

    def validate(raw: dict, attempt: int) -> pipeline.Verdict:
        try:
            return pipeline.Verdict(detail=checked_arguments(command, raw, request, workflow, verdicts))
        except PlanningCommandInputError as refused:
            return pipeline.Verdict(_input_error_kind(refused), str(refused), refused)

    # one fill with the phase's note after the request, no retry: a refusal is the end of it
    extracted = pipeline.extract(
        fill_text, validate=validate,
        first_note=PHASE_FAILED_EXTRACTION_NOTE.format(observation=observation))
    if not extracted.final.ok:
        return None
    new_args, _notes = extracted.final.detail
    return None if new_args == args else new_args


def _open_arguments(command: PlanningCommandCommand, refused: "PlanningCommandInputError", raw: dict, request: str,
                    verdicts: dict | None, reset: dict) -> tuple[str, ...] | None:
    """The arguments a refused call leaves open (every gap ``argument_gaps`` finds, the ones the refusal names, the ones
    the not-given check reset): the only ones the one second fill may change. None when the refusal names no input."""
    if not refusal_names_an_input(command, refused):
        return None
    names = [*refusal_gaps(command, refused).names, *argument_gaps(command, raw, request, verdicts).names, *reset]
    return tuple(dict.fromkeys(names)) or None


def _input_error_kind(refused: "PlanningCommandInputError", command: PlanningCommandCommand | None = None,
                      raw: dict | None = None, request: str = "", verdicts: dict | None = None) -> str:
    """How the pipeline classifies a refusal of a planning call's arguments: ``missing`` when only required arguments
    are absent (nothing the model produced was refused), else ``refused``. Read from the guard's own code (``code``
    "missing"), never from the message text; the guard stops at its first gap, so when *command* and *raw* are given a
    value refused after the first missing one counts too (``argument_gaps``, no model call)."""
    from fastworkflow.parameters import pipeline

    if getattr(refused, "code", "") != "missing":
        return pipeline.KIND_REFUSED
    if command is not None and raw is not None and argument_gaps(command, raw, request, verdicts).invalid:
        return pipeline.KIND_REFUSED
    return pipeline.KIND_MISSING


def replanned_call(command: PlanningCommandCommand, call: PlanningCommandCallPlan, args: dict,
                   notes: list[str]) -> PlanningCommandCallPlan:
    """*call* with *args* instead: its steps and phases are read from the new arguments; the extra steps before
    and after the planning command's own are kept as the call had them."""
    steps = expand(command, args)
    before, after = list(call.steps_before), list(call.steps_after)
    return PlanningCommandCallPlan(plan=number(before + steps + after), args=args, notes=notes,
                                   prediction=call.prediction, phases=call_phases(command, args),
                                   steps_before=before, steps_after=after, verdicts=call.verdicts)
