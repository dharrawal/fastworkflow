---
name: author-planning-commands
description: >-
  Write fastWorkflow planning commands (`_commands/<name>.py`, or `_commands/<Context>/<name>.py`):
  parameterized multi-command jobs whose steps are written in code and become the agent's numbered
  plan, flat or as phases that pass facts to each other. Covers when to write a planning command
  instead of a command, the file shape (`Signature` + a top-level `PlanGenerator`), `Input` and
  `from_command_output`, phases and facts typed by their commands, the step helpers (`run`, `when`,
  `do_not_run`, `count`), nesting with its cycle guard and step budget, `__phases__` and `__result__`,
  `copied_from_command_text` (free text that must be the user's words included), an input for every scope users ask
  about and an answer that keeps the description's promise, `@writes` on commands that change data (so a
  read-only planning command's phases make no writes), reuse from `_` modules only, the `command
  preview` / `--check` loop, the step wording the framework reads in code, and descriptions written for
  people. Use when adding, reviewing or debugging a planning command, or when its steps are run, checked
  or answered differently than intended, or when a phase made a write nobody asked for.
---

# Writing planning commands

A planning command is one Python file, `_commands/<name>.py`, or `_commands/<Context>/<name>.py` for
a planning command of a context (named `<Context>/<name>`). It is written like a command file: its
`Signature` describes a job and its `Input` declares the job's typed arguments, exactly as a
command's does. Where a command has a `ResponseGenerator`, a planning command has a top-level
`PlanGenerator`: each public method is a phase that returns its steps as text.

How a planning command runs:

1. **Which command.** A planning command is an intent label like any command: the local intent model
   trains on its `plain_utterances`, its name and the first sentence of its `Signature` docstring, and
   the remote decision model (`JEV_API_KEY`) reads its docstring and its first 9 `plain_utterances`. A
   request reaches it through the one intent decision every command shares (`intent_loop.resolve`): its
   exact name, a close misspelling, the local model, the remote model. A confident local answer that
   names a planning command is never acted on alone: the remote model confirms it, or, with no key, the
   user is asked. When several commands fit, the user is asked which one is meant (`question_kind`
   `plan_choice`); each choice is listed with its docstring, so write the docstring so a reader can tell
   the commands apart.
2. **Arguments.** The planner fills `Input` in one typed call (`planning_commands.typed_planning_call`).
   The guards, the field validators and the model validators then check it (the one parameter pipeline,
   `fastworkflow.parameters`, shared with execution commands). A value a guard or a validator refused is
   filled once more with the refusal in view; a value that is simply missing is not.
3. **Questions.** A required value still missing or refused is asked of the user, when someone can answer,
   in one question naming every such value: the same question and reply reader an execution command
   uses (`parameter_clarification`). It is asked again while each reply fills or fixes a value it listed;
   the second reply in a row with no progress, or `cancel`, ends it. A refusal that names no input (a
   plain `ValueError`) is not asked. When it ends unanswered, the text planner or the agent takes over;
   with no agent model the part is not run and the answer names what is missing.
4. **Review.** Every part of the turn is resolved first, then the turn gets ONE review of all its parts
   before anything runs (none when the request names the command exactly, when the user picked it by
   answering a question, or when nobody can answer).
5. **Run.** The phases run; code runs the steps it can (see the step grammar below). At the first step
   code cannot run, the phase's agent carries on. With no agent model (`LLM_AGENT` unset), a decision
   step (a `when(...)` step whose inner step code can run, or a command step missing one plain required
   value) is asked of the user, and any other open step ends the command as a stop (artifact
   `fw_not_run`) whose answer states what ran. The answer comes from `__result__` or a code-written
   `say that` phase; otherwise a model writes it (with no agent model, the stop).

The one-page guide is `docs/planning_command_file_layout.md` in the fastWorkflow repository (not in
the wheel), with the preview; `docs/planning_command_phases.md` covers phases, facts, code-run steps
and answers. The module documentation of `fastworkflow.planning` (the form),
`fastworkflow.planning_commands` and `fastworkflow.planning_phases` (the runtime) ships with the
package.

Code references below name a module of the `fastworkflow` package and a symbol in it.

## Planning command or command?

| Write a command (`_commands/<name>.py`) when | Write a planning command (`_commands/<name>.py`) when |
|---|---|
| it is one action on the application (wraps code, returns an `Output`) | a family of requests is answered by the same several commands in a known order (find, open, read, answer) |
| the agent should choose it by intent | the agent keeps getting the order, a value or the answer wrong for that family |
| it needs to read state or call a backend | the job only needs text: `PlanGenerator` methods build text and must not call a backend or read state (`planning_commands` module doc) |

A planning command adds no capability: every step it writes runs an existing command (checked at
load). A one-command planning command is a command with a worse name. Fixed steps with no parameters
are allowed (a `PlanGenerator` with one method taking no `input`): the text planner adapts them. A
planning command with no parameters may also have phases: every call has the one plan, and the
planner's call has no arguments (`planning_commands.py` `planning_command_from_parts`).

## The file

- Name: the file stem is the planning command's name, a Python identifier (`member_report.py`;
  `workflow_agent.py` `_read_planning_command_file`). A `.py` file or folder whose name starts with
  `_` or `.` is not a planning command: a helper module that command files import (shared lookup tables
  and phrases, never code that chooses steps) lives in the helper package `_commands/_shared/`
  (`from ._shared.x import Y` from a root command, `from .._shared.x import Y` from a context folder), and
  any other underscore `.py` file or folder of `_commands/` or a context folder, other than `_workflow.py`
  and a context's own `_<Context>.py`, stops the workflow, naming `_commands/_shared/`; `_workflow.py` holds the
  workflow's settings as the class attributes of its `class Context` (`split_examples`, `identifier_patterns`,
  `subject_hints`, nothing else), read only from `_commands/_workflow.py` (one in a context folder is a load
  problem, and a `_split.py`, the removed spelling, stops the workflow); an `identifier_patterns` entry anchored with `^` or `$` is a load problem too (the patterns
  are searched inside a message: use `\b`). A `<Context>/` folder holds the planning commands of that context
  (`workflow_agent.py` `_planning_command_files`).
- Order (`docs/planning_command_file_layout.md`): module docstring, imports, `Signature`, shared facts
  models, `PlanGenerator` (`__phases__` first, then the phases in run order, then `_helpers`),
  module-level tables and shared functions.
- A file that fails to import or is malformed fails the load of that command only: it is not matched and
  the rest of the workflow loads. `fastworkflow run_cli`, `fastworkflow train` (which stops), the server and
  `fastworkflow command preview --check` list it as one line naming the file
  (`command_preview.load_problems`; the one rule is `docs/design_idioms/removed.html#load-rule`).

A complete phased planning command, the library example's `Library/member_report`
(`fastworkflow/examples/library/_commands/Library/member_report.py`, utterances shortened):

```python
"""member_report: the member the request names, with their loans or notes when asked.

find lists the members by name; member opens the one named (skipped when nothing
matched); loans and notes run only when the request asks for them, on the member that
member opened; __result__ says what the phases found. A request about every member is
refused before any step runs.
"""
import fastworkflow
from pydantic import BaseModel, Field, model_validator

from fastworkflow.planning import count, count_and_names_read, run

from ..Member.list_member_loans import Signature as ListMemberLoans
from ..Record.show_notes import Signature as ShowNotes
from .find_member import Signature as FindMember
from .open_member import Signature as OpenMember


class Signature:
    """Show a member's loans, and the notes on their record when asked. One member per call."""

    plain_utterances = [
        "tell me about Raj Patel",
        "what has Ann Lee borrowed",
        "any notes on Mia Chen's record",
    ]

    class Input(BaseModel):
        member: str = Field(description="the member's name",
                            json_schema_extra={"copied_from_command_text": True})
        loans: bool = Field(default=False,
                            description="whether to name the books the member has on loan")
        notes: bool = Field(default=False,
                            description="whether to show the notes on the member's record")

        @model_validator(mode="after")
        def refuse_everyone(self, info):
            if self.member.casefold() in EVERYONE:
                raise fastworkflow.Refused(
                    f"member is {self.member!r}, which names no one member; this "
                    "command reports on one member: give that member's name", field="member")
            return self


class PlanGenerator:
    def __phases__(self, input):
        loans = [self.loans] if input.loans else []
        notes = [self.notes] if input.notes else []
        return [self.find, self.member, *loans, *notes]

    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.member)]

    def member(self, input, find) -> OpenMember.Output:
        if not find.member_uids:
            return []
        return [run("open_member", named=input.member)]

    def loans(self, member) -> ListMemberLoans.Output:
        if member is None:
            return []
        return [run("list_member_loans", all_pages=True)]

    def notes(self, member) -> ShowNotes.Output:
        if member is None:
            return []
        return [run("show_notes")]

    def __result__(self, input, member, loans, notes):
        if member is None:
            return f"No member is named {input.member}."
        on_loan = (count(member.on_loan, "book") if loans is None
                   else count_and_names_read(loans.titles, loans.total, "book"))
        lines = [f"{member.name} has {on_loan} on loan."]
        if notes is not None:
            lines.append(f"{member.name}'s record holds "
                         f"{count_and_names_read(notes.notes, notes.total, 'note')}.")
        return "\n".join(lines)


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

Code runs every step of this command and `__result__` writes the answer, so it needs no model after
its arguments are filled (`fastworkflow command run-code` shows it, below). A planning command of one
method that reads and fills no facts runs flat, as one plan.

## `Input` rules

| Rule | Where |
|---|---|
| A field is `str`, `bool`, `int`, `float`, `list[str]`, `Literal[...]` of text (an enum), or `Optional[str] = None`; any other type or `Optional` fails the load | `planning_commands.py` `_spec_from_field` |
| A `bool` defaults to false; `default=True` fails the load (the model sets it true only when the request asks) | `planning_commands.py` `_parse_param` |
| No default: required. A `str` the request may leave out defaults to `""` or is `Optional[str] = None` (the code then gets `None`); a default of `"NOT_FOUND"` (the framework's not-given marker) is a load error | `planning_commands` module doc; `input_instance` |
| Names `user_query`, `available_commands`, `steps_before`, `steps_after`, `reasoning` are reserved | `planning_commands.py` `RESERVED_NAMES` |
| A rule across fields is an `@model_validator(mode="after")` on `Input` that corrects (assigns `self.<field>` and returns `self`) or refuses: `raise fastworkflow.Refused(message, field=...)` names the input, so the user can be asked for it; a plain `ValueError` names none, so it is not asked. Its `info.context` has `command_text` (the request plus any replies), `workflow` (None while the template renders) and `command` | `planning_commands.py` `check_input` |

### Field options (`json_schema_extra`)

Each option is a key in a field's `json_schema_extra`, written with plain pydantic `Field(...)`
(`fastworkflow.planning.FieldOptions` types them for an IDE). The key names the behaviour; its value is
`True` or a dict of that key's own options. The same keys work on an execution command's `Input`
(`fastworkflow.field_options`). An unknown key, a value that is neither `True` nor a dict of known
options, a key on a type it does not apply to, two keys that conflict, or a retired spelling (`in_request`,
`verify`, `positional`, `from`, `subject`, `means`, `strip`) fails the load, naming the key to write.

With no option at all, the model's `str` or `int` value stands: it is not compared with the request (a
`Field(pattern=...)` only validates). Only `copied_from_command_text` or `record_named_in_command_text` is the strict
check: a `str` must be written in the request at word boundaries (case and dashes ignored; "dun" is not written in
"dune"), an `int` must be a number it writes (refused, never replaced, even when the request writes no number). A
value of only punctuation, or a value that is entirely one template token (`<name>`, `{name}`; `SKU_42` and `{"a": 1}` are values), is refused
on every `str` field; there is no list of stand-in words. A value of an enum needs no grounding.

- `copied_from_command_text: True` (`str`, `int` or `list[str]`): the value is taken in the request's own
  spelling (a `str` is written in the request at word boundaries), and the field's description tells the extractor so
  (`extraction_sentences`; do not repeat that sentence in the description). A `list[str]` keeps each
  item the request writes. An `int` is a number the request writes (digits, or the words zero to twenty
  outside the other arguments' text).
- `copied_from_command_text: {"ignore": ["group", "unit"]}` (`str` only): the named words are removed from
  either end of the value before the match, for example a kind word the model appends to a name
  ("Payroll group" becomes "Payroll").
- `record_named_in_command_text: True` (`str` only): the value names a record of the request (a person, a
  group, a product, a code). It must hold one of the request's subjects whole, or a name-like word, and
  must not cut one. A value cut short at the start or end of exactly one subject becomes that subject
  ("Ann" of "Ann Lee"). A value given and not written is refused, required or optional; one not given is not checked.
  Its `{"ignore": [...]}` form works as the `copied_from_command_text` one does.
- `classifier_confirms: True` (`bool`, an enum, or a `str` with `record_named_in_command_text`): a
  classifier decides the value again from the request alone, and its answer replaces the planner's when
  it is sure enough. It reads the field's `description`. It needs `JEV_API_KEY`; without the key the
  check does not run (`planning_commands.check_verdicts`).
- `classifier_confirms: {"meanings": {"group": "a group of people", "unit": "an organisation unit"}}` (an
  enum): the same check, with each value's meaning. Its keys must be values of the enum.
- `classifier_decides: True` (`bool`): the classifier alone decides the field from the request; the typed
  call never fills it, and it keeps its default when the classifier is not sure or there is no
  `JEV_API_KEY`.
- `from_command_output: "<command>.<field>"` (or a list): the command whose output supplies the value
  (see "Phases and facts" below).
- `command_text_after_command_name` is for an execution command's `Input` only; on a planning command
  it fails the load.

#### Training-only declarations: `words` and `examples_by`

Two declarations feed only the synthetic utterance generator (training data): no check, extractor, planner,
listing or preview reads them, and they change nothing at run time (the extraction prompt, the listing, the
classifier check and the preview are byte-identical). They are visible in the field's JSON Schema
(`GET /command_schemas`). Mistakes, misspellings included, fail the load like any option.

- `words` inside `classifier_confirms` (a `Literal` only): `{"<value>": ["<word users write>", ...]}`. Keys are
  declared values; each list non-empty text; a word once per value (ignoring case) and under one value only; a value
  may have none. `{"words": {...}}` alone is the same check as `classifier_confirms: True`.
- `examples_by` on the field whose examples depend on a choice field of the same `Input`:
  `{"<choice field>": {"<value>": [examples of this field, ...]}}`. One choice field only, a `Literal` and not this
  field; inner keys are its values; each example must be valid for this field's type, `pattern` and bounds.

```python
kind: Literal["identity", "group"] = Field(..., json_schema_extra={"classifier_confirms": {
    "meanings": {"identity": "one person", "group": "a set of people"},
    "words": {"identity": ["person", "employee"], "group": ["team", "role"]}}})
name: str = Field(..., examples=["Maria Okafor", "Payroll Approvers EMEA"], json_schema_extra={
    "copied_from_command_text": True,
    "examples_by": {"kind": {"identity": ["Maria Okafor", "Tom Reyes"], "group": ["Payroll Approvers EMEA"]}}})
```

Adding them edits the command file, so, like any such edit, the workflow's next `fastworkflow train` retrains it.

A field with no check is cleaned in code: `strip_words(text, words)` from `fastworkflow.planning_commands`,
called in the field's `field_validator`. A validator's correction is kept when it passes and dropped
when it refuses. Full rules: `docs/parameter_pipeline_design.md` and the `planning_commands` module doc.

### Declare what a value looks like: examples, bounds, formats, allowed values

The framework reads these declarations in several places, so state them in the field, not only in prose. (The
two training-only declarations above, `words` and `examples_by`, are the other half: they feed generated training
data only and are not repeated here.)

- **`Field(examples=[...])` on an `Input` field.** Read by: the `(e.g., ...)` hint of a refusal (the first example, in a
  type or `pattern` refusal: `signatures.py` `build_type_suggestion`, `rules.pattern_refusal`); the chat form's
  placeholder (the first example of a text field); the sample value of a planning-command preview (the first example,
  unless an optional `str` has a non-empty default: `command_preview.sample_value`); the generated few-shot parameter
  examples (`generate_param_examples.py`); and the command listing the agent and planner read, which renders an Input
  field's examples (`command_metadata_api.py`). The extraction model is NOT shown them (a model copies them as values;
  `signatures.py`). Write 2 to 5 realistic values with the length and shape real values have: a long title field needs
  long examples, an id field ids of the real format (`b1`, `m2`). Never real customer data. Keep them valid for the
  field's own `pattern` and validators (nothing checks this except `examples_by`), few and short where you can: they
  count toward the listing size, and the listing is given whole only up to `run_listing.FULL_LISTING_MAX_CHARS`
  (8000) characters, a shortlist above it. The library's `find_book.query` has `examples=["Middlemarch", "George Eliot"]`.
- **`examples` on an `Output` field** are not rendered in the listing text (`command_metadata_api.py`); they stay in
  the structured metadata and the JSON Schema. Add them where an `Input` declares `from_command_output` pointing at the
  field: `find_book.book_uids` has `examples=[["b1", "b2"], ["b3"]]`, which `open_book.book_uid` takes one of.
- **Bounds and formats.** Declare `ge`, `le`, `gt`, `lt` and `pattern` instead of writing them only in the description.
  A declared bound is enforced: an out-of-range value is refused and the user asked again
  (`signatures.numeric_range_problem`; a field named like a position or count, `start`, `offset`, `limit`, `count`,
  `page`, `size`, that declares no lower bound may not be negative anyway), and a `pattern` must match the whole value
  (`rules.pattern_refusal`) and is also told to the extraction model (`rules.pattern_sentence`: "Must match the
  pattern ..."). So declare only what the command already refuses. A command that clamps today either keeps clamping
  and says so in the description, or declares the bound as a deliberate behaviour change. The library's three `start`
  fields are `Optional[int] = Field(default=None, ge=0, ...)`.
- **Allowed values.** A fixed set known in code is a `Literal[...]` type: the extractor is told the values and another
  value is refused by name. A set that comes from data stays a `str` with a validator plus `examples`. "One of: a, b"
  in a description is only prose, read by nothing.
- **Say what a value looks like** in the description where a shape exists ("the book's uid, such as b1"; "an ISO
  date"), not only what it means. An `Input` field's description is part of the extraction prompt, so editing one
  changes the prompt of every request for that command: re-measure when you do.
- **`plain_utterances` stay generic**: they train intent detection (which command), so a value in one teaches the
  wrong thing; values belong in field `examples`.
- **`personas.json`** in the workflow folder (the persona source for generated utterances: your own `personas` list
  or `domain_keywords` that filter PersonaHub; `train/personas.py`) is optional and costs a regeneration: any
  `personas.json` makes the persona source label non-empty, which is a global input of the training signature
  (`selective_training.py`, `persona_source`) and part of every command's generated-utterance cache key
  (`utterance_cache.py` `persona_source`), so the next `fastworkflow train` regenerates all generated utterances (model
  spend). Adding, changing or removing the file does this; with none, the label is empty and nothing changes.
- **Cost of any of these edits.** A command file edit changes the workflow's training signature: the next
  `fastworkflow train` retrains (seconds in the `embedding` layout), and `fastworkflow command preview <workflow>
  --check` may report a stale preview (a first example is the sample value).

A free-text value the command stores or acts on (a note, a message, a reason) is the field most easily
drafted wrong: as a paraphrase, an invented sentence ("Requested extension for upcoming project deadline"
for "she asked for an extension"), the instruction itself ("Note on Mia Chen's record that she asked for
an extension"), or the note with the next job of a bundled request glued on (measured on the library
example's `note_member`). Declare it in two layers:

1. **`copied_from_command_text: True`.** The note is taken as the request writes it, so a paraphrase or an
   invented note is refused and filled once more with the refusal. The same check runs on a drafted
   sub-task input, on the typed call and on a step that calls the command.
2. **A validator for what the strict check cannot see.** A value that is the whole instruction is
   written in the request too. Refuse it with a rule the command's own meaning gives, in a
   `model_validator` on `Input`. For a note on a member's record, the rule is that the note does not name
   that member: a text that does holds the request around the note. Match the other value as whole
   words, so a short name does not refuse an unrelated note.

Describe the field as what it holds, with a short worked case in prose ("what the note says, without
the request around it (to note that someone called: 'someone called')"). On the library, that wording
took the note's own words in 8 of 9 clean sub-tasks where the bare "the note to add" let through a
paraphrase, an invention and the whole instruction. Likewise a scope field names the values it takes
("such as Central or Riverside").

A free-text field's `examples` are not shown to the extraction model (a model copies them), but a phase agent
reading the listing (which shows an Input field's `examples`, never an Output field's) can still write one into a call (see "Commands that write" below). A value a sub-task holds because the splitter left two jobs in it is a
splitting failure, not one this declaration can catch.

## Inputs for every scope, answers that keep the description's promise

From the library example's `overdue_summary`:

- **Give the command an input for every way people scope the question it answers.** "How many loans are
  overdue at Riverside?" matched `overdue_summary`, which had no branch input, so the call counted the
  whole library. Its callee `count_overdue` took a branch; the planning command must take it too and pass
  it on (`run("count_overdue", branch=input.branch or None)`). An optional scope takes the default that
  means "all" (`""`), and the `Input` maps the words for "all" ("all branches", "the whole library") to
  it and refuses a value that names nothing (a branch that does not exist), naming the values that
  exist. A scope the user writes is not a handle: give it `copied_from_command_text` (with `ignore` for
  the kind word: "the Riverside branch" becomes "Riverside"), not `from_command_output`, which would make
  a draft take the value from an earlier sub-task that merely listed the branches.
- **The answer must deliver everything the description promises.** The last phase's agent sees only its
  own phase's runs and steps, so a value an earlier phase finds reaches the answer only as a fact: type
  that phase with the command's `Output` and write the value into the answer step, or build the answer in
  `__result__`. Read each description sentence against the answer path and check that every noun it
  promises arrives there.
- **A step must not carry a placeholder for a value derived from an earlier step's output in the same
  phase.** Code cannot fill it, so the step is open (the agent fills it; with no agent model the command
  stops there). Put that step's source in its own earlier phase and compute the value in the phase
  method from its fact.

## Phases and facts (typed by commands)

- Each public method of `PlanGenerator` is a phase (`planning.py` `planned`, building the
  runtime's `planning_phases.Phase`), in the order written; `_name` methods are helpers. Each phase
  is a fresh executor run with its own step budget (`PHASE_MAX_STEPS`, 15), except an answer-only
  phase (one model call, or none). The last phase writes the answer; when code ran every step of the
  last phase and `__result__` writes the answer, that phase takes no model step.
- Its return annotation is its facts model. Two forms:
  - (a phase annotated with a command's `Input` alone, or a union with an `Input` member, is a load
    error that stops the command loading; the `Input` goes in a field of a facts model, below)
  - a command's `Signature.Output` class, a phase "typed by its command": its fields are filled from
    the output of that command in the phase's runs (`typed_phase_facts.py` `fill_typed_phase_output`),
    as in `member_report` above;
  - a facts model whose fields are each typed with a command's `Signature.Output` (or `Input`), with
    the other values as properties or methods of the model (a copy, `fact_from` or a `"from"` on a facts
    field, fails the load):

    ```python
    class Book(BaseModel):           # nested right above the one phase that fills it
        show: ShowBook.Output

        @property
        def author(self) -> str:
            return self.show.author

    def book(self, find) -> Book:
        ...
    ```
  Either way the facts come from the runs of that phase only, and a step of the phase must run (or at
  least name) each command its facts are typed by, or the load fails. A required fact no run holds fails
  the phase, and the job stops with the facts found so far. Use the return annotation when the facts
  type is fixed (or a union for "whichever ran"); use `@phase(output=lambda input: ...)` only when the
  type depends on the input (`output=` takes only a function of `input`). A phase with both is a load error (`PhaseError`, naming the phase).
- `from_command_output="<command>.<field>"` on a command's `Input` field names the command that supplies
  the value, so the planner can run it first (`planning_commands.py` `resolve_field_sources`). A
  hand-written `available_from` is an error that `command preview --check` reports. See the
  `declare-parameter-producers` skill.
- Its parameters bind by name: `input`, and earlier phases' facts (`planning_phases.py`
  `bound_arguments`). A phase that may have been skipped reads as `None`. A parameter must name an
  earlier phase with facts in at least one call the inputs give, or the load fails
  (`planning_phases.py` `check_bindings`), so a misspelling is caught.
- `return []` skips the phase. It is always decided when the phase is reached: the runtime calls the
  method once with the real facts (`planning.py` `_phase_functions`). A skipped phase passes nothing on.
- `__phases__(self, input)` returns the phase methods a call runs, in order (default: all). A call of
  one phase that reads and fills no facts runs flat, as one plan (`planning.py` `_runs_flat`). A phase
  repeated for several cases is one method each, calling one shared module function (a `self._helper`
  called with a fact is not run in the template, it reads as its name).
- `@phase(over="<phase>.<list field or property>")` (or a function of `input` giving the items) fans a
  phase out, one sub-plan per item (at most `FANOUT_MAX_ITEMS`, 10, unless `max_items` says otherwise);
  its one parameter that is neither `input` nor a phase gets the item; `max_items`, `noun`, `until` (a
  function of the items' facts so far; true stops the fan-out) and `bind` tune it
  (`docs/planning_command_phases.md`, "Fan-out phases"). `@phase(optional=True)`: the job goes on when it
  fails.
- The template (what the planner sees; `planning_commands.py` `placeholder_template`) calls each
  method once per input with `planning_phases.Placeholder` facts: `<find.total>`, `<find.uids[0]>`,
  `<find.labels[*]>`; a `self._helper` called on one reads `<helper in words>`; every test on one
  (`bool`, `len`, `==`, `<`, `in`) reads as true and is noted, and the phase's lines end
  `(only when find.uids)`. `x is None` is not seen. Write wording that tests facts (`a or b`) in a
  helper.
- At load (`planning_commands.py` `_checked_plan`, `placeholder_template`, `check_step_commands`;
  `workflow_agent.py` `step_command_problems`): a method without `self`, a bad annotation, a parameter
  that cannot bind, a facts field typed with anything but a command's `Output` or `Input`, a method that
  fails on placeholder facts or writes `None` into a step, and a step naming a command the workflow
  lacks or a parameter its `Signature.Input` lacks fail the load.
- A planning command imports what it writes from `fastworkflow.planning`: `run`, `when`, `do_not_run`,
  `phase`, `count`, `count_and_names_read`. The retired helpers (`pick`, `full_list`, `listed`, `num`)
  fail the import, naming their replacement.

## Step helpers

`fastworkflow.planning` writes steps as strings the framework reads (the grammar below). Plain
strings work too; the helpers only make the wording exact.

| Helper | Writes | What it does |
|---|---|---|
| `run("cmd", k=v)` | `cmd k=v` | A command step code runs once every required value is given (a `None` value is left out). `run` on another planning command is a nested call (see below). |
| `run("cmd", k=v, all_pages=True)` | `cmd k=v (full list)` | Code reads every page, up to `FW_CODE_MAX_PAGES` (default 20). Past that, with rows left, the step is open: the agent reads on from the next page; with no agent model the command stops and its answer says the read is partial (the rows read of the total). |
| `run("cmd", named=name)` | `cmd with the uid of the <name> match` | Code opens the one row of a listing this call read that holds `name` whole (or the only row). The row id input is the field that declares `from_command_output`, else the command's one required `str` field; a command with neither is a load error, and no other input may be given. With a concrete id (`run("open_book", named=t, book_uid="b1")`) the step is `cmd book_uid=b1`. |
| `when("condition", step)` | `only when condition, step` | A conditional step the agent decides whether to run; with no agent model, a yes or no question to the user when code can run the inner step. |
| `do_not_run("cmd")` | `do not run cmd` | A negated guard: code passes it over and never holds it to a run. |
| `count(n, noun, plural=None)`, `count_and_names_read(names, total, noun)` | text | How a number or a counted list reads ("2 books", "3 books (a, b, c)", "K of N shown"). Use `count`, not an inline `n == 1`, which the preview cannot render. |

## Nesting, the cycle guard and the step budget

- A step naming another planning command runs it as a nested call, with its own phases and its own
  agent, and the nested call's output is the step's run: `run("book_by_title", title=input.title)`.
  Its steps number after the step that called it.
- A planning command cannot call itself, even through others. A call cycle is reported as an error
  that names the cycle ("a planning command cannot call itself, even through others: A -> B -> A",
  `planning_commands.py` `_cycle_problems`): every planning command on the cycle fails to load, and
  `fastworkflow run_cli`, `train`, the server and `command preview --check` list it (`command_directory.py`
  `command_load_errors`). A fixed step whose first word is the command's own name is a call of itself. At run time, a call
  of a planning command already running in the chain is refused with "is already running; call its
  steps directly."
- The turn-wide step budget is `FW_TURN_MAX_STEPS` (default 400; `plan_frame.py` `turn_max_steps`).
  It counts the model steps the nested planning-command calls of one turn take together, and it
  resets each turn. Once it is spent, a further nested call is refused with "The step budget for
  nested commands is spent; continue with the remaining steps directly." A value that is not a positive
  integer logs a warning and the default applies.
- `PHASE_MAX_STEPS` (15) is a different budget: the steps of one phase's executor run.

## Descriptions written for people

- The `Signature` docstring says the job in one or two sentences, as a person would describe it. Its
  whitespace is collapsed to one line. The remote model reads it, the which-command question shows it
  beside the command's name, and its first sentence is a training row of the local intent model, so make
  the first sentence the plain purpose and put conditions after it.
- A field's `description` says what the field holds, in words a person would use. The planner reads
  it; so does a `classifier_confirms` check.
- `plain_utterances` are example requests in the words people use for that job. Keep them generic,
  and do not copy names from a test workflow. The remote model sees the first 9; the rest still train
  the local model (`command preview --check` notes a command with more).
- Step wording is not written for people: the framework reads it, so use the grammar below.

## `__phases__`, `__result__` and `Signature.Output`

- `__phases__(self, input)` returns the class's own phase methods (`self.find`, ...); a base's method
  the class overrides is returned as `self.<name>`, and the base's is called from it with `super()`.
- A `__call__` method on a `PlanGenerator` is a load error: "PlanGenerator defines __call__; rename
  it to __phases__" (`planning.py` `planned`).
- `__result__(self, input, <phase names>)` builds the call's answer once the phases have run. It
  returns what a command's `__call__` may return, judged by the one rule `command_executor.command_response_of`: a
  `str` or a `fastworkflow.CommandResponse` when no `Output` is declared; when one is declared, exactly that `Output`,
  directly or in `CommandResponse(output=...)` (text or a subclass is an error). Its parameters name `input` or a phase
  of the call; any other name is a load error. A command with `__result__` text needs no model to
  answer, which matters with no agent model.
- A planning command that declares `Signature.Output` must have a `__result__` that builds it: "nothing
  builds it" otherwise. An `Output` field cannot declare `from_command_output`; the value is built in
  `__result__`.
- If `__result__` raises, returns an empty string or returns something else, the call FAILS as an execution
  command's does (`planning_phases.py` `run_call_result`: one error log, a failed output naming the command and the
  error; `raise fastworkflow.Refused(message)` is a refusal with that message); no model writes the answer in its
  place. Return `None` to hand the answer to a model (or, with no agent model, to the `answer` stop). A return
  annotation that is `str` or another model on a command that declares an `Output` fails the load.

## How the chat draws an `Output`: names never decide, hints do

A field's name never decides how it is drawn. Type and shape give the plain default; everything else is a
`ui` hint you declare in `Field(json_schema_extra={"ui": {...}})` (`fastworkflow.field_options.UiOptions`
documents each):

| Hint | Controls |
|---|---|
| `title` (`Field(title=...)`) | the label |
| `hidden` | left out of the result card |
| `hero` | pinned to the stat strip |
| `format` | `currency`, `percent`, `code` or `badge` |
| `tones` | with `format: badge`, a value to `good`, `warning`, `bad` or `neutral` |
| `currency` | with `format: currency`, an ISO code such as `"EUR"` |
| `widget` | an `Input` field as `textarea` or `radio` |
| `table` | list fields of one output with the same group name are the columns of one table |

A badge needs `format: badge`: a field called `status` or `state` is plain text without it. Parallel lists
need the `table` hint to be drawn as one table. `fastworkflow command preview <workflow> --check` prints a
`notice:` for an `Output` with two or more visible list-of-scalar fields in no `table` group; it never
fails the check.

## Reuse: inherit from helper modules only

- A `PlanGenerator` may subclass a `PlanGenerator` defined in a helper module of `_commands/_shared/`
  (`from ._shared.lookup import PlanGenerator as Base`). `self` reaches the base's phases, helpers and fact
  models; a method of the same name overrides a base phase and may extend its steps with
  `super().<phase>(...)`. A subclass that only overrides keeps the base's plan; one that adds phases
  writes `__phases__` (`planning.py` `_order_problem`).
- Subclassing another planning command's `PlanGenerator` is a load error (`planning_commands.py`
  `_plan_generator_inherits_foreign`). To reuse its whole job, call it in a step
  (`run("<name>", ...)`); to share phases, move them to a base class in a `_` module.

## The preview and run-code loop

```sh
fastworkflow command preview <workflow_folder>            # writes <workflow_folder>/../docs/planning_command_previews/<name>.md
fastworkflow command preview <workflow_folder> --check    # writes nothing; exit 1 when not current
fastworkflow command run-code <workflow_folder> <command> --input '{"member": "Ann Lee"}'   # one call, no model
```

1. Write or change the planning command file.
2. Run `command preview`. A load error names the file and its reason; fix it first.
3. Read the preview: the inputs, the phases table ("runs" says when each phase runs), the facts with
   their sources, and the steps of each different plan. A step rendered with `<phase.fact>`
   placeholders shows where a value comes from at run time. Under the steps, a "Pages:" line ("step 2
   reads up to 20 pages, then the agent reads on; step 3 reads page 1 only.") says how code pages each
   step on a paged command (`planning_phases.step_paging`).
4. Check each step against the grammar table below: is it run by code, held by the finish check,
   written into the answer as intended?
5. Run `command run-code` on a development instance (or a copy of the workflow: its steps are the
   workflow's real commands). It runs the command phase by phase with no model, printing each step code
   runs, the facts and the answer, and stops at the first open step with exit 0, which shows exactly
   where code hands over. `--start-with '<step>'` (repeatable) runs steps first to put the workflow
   where the command starts. Exit 1 when the command does not load, a step fails or a hook raises. In
   tests, `fastworkflow.planning.run_code` is the same run.
6. Commit the previews and run `--check` in CI (`command_preview.py` `check_previews`). It fails on a
   stale, missing or extra preview, a planning command file that does not load (a step naming an
   unknown command or parameter included), and a `from_command_output` error reported by
   `planning_commands.resolve_field_sources`. Its `notice:` lines never fail it.

## Step grammar

Each step is one string, one line of the plan. The framework reads it in code: the finish check
(`observation_offloading/finish_check.py`), code-run steps and picks, and the answer-only phase
(`planning_phases.py`). A step is classified by the FIRST declared command it names; the text before
that command decides conditional, negated and answer steps (a condition word in its chain also makes
it conditional).

| Wording | What it triggers | Implemented at |
|---|---|---|
| `cmd k=v k2=v2`, the command the first word, every required field given, values that bind | Mechanical: code runs `cmd k=v k2=v2` before the model takes a step | `planning_phases.py` `_code_step`, `plan_code_steps` |
| `cmd` alone, every field defaulted (or no parameters) | Mechanical: code sends the bare command | `planning_phases.py` `_code_step` |
| `cmd k=v (note)`, `cmd k=v; note` | The chain ends at the first `(` or `;`; the note is never sent and never changes what runs | `planning_phases.py` `_code_step`; `finish_check.py` `step_arguments` |
| a concrete value `k=Alpha` | In a phase, the finish check and the fact fill accept only a run with that value (spacing, outer quotes, case ignored) | `finish_check.py` `step_arguments`; `planning_command_phases.md` "Facts" |
| `k=<the uid>`, `k={x}`, `k=[..]`, `k=...` | A placeholder, not a value: the step is open (the agent fills it); any run of the command with arguments satisfies it | `finish_check.py` `_PLACEHOLDER_RE` |
| `k="Acme (EU)"`, `k='Sales; West'` | A value in quotes right after `=` is kept whole: its `(`, `;` and `, ` do not end it (unquoted, `k=Acme (EU)` sends `k=Acme`) | `finish_check.py` `step_chain`, `_value_text` |
| `k=Ann Lee, no filter` | A value ends at `", "`; prose swallowed into it leaves the step open | `finish_check.py` `_value_text`; `planning_phases.py` `_code_step` ("prose in value") |
| a value whose first token is an id (16+ letters, digits, `_`, `-`; 3+ digits) | The value is that token alone, cut at the first space or comma | `finish_check.py` `_UID_TOKEN_RE` |
| `cmd` then prose not in parentheses (`go_up back to the list`), or the command not the first word (`then cmd k=v`) | Open: the agent runs it | `planning_phases.py` `_code_step` |
| a required field left out (`find_item` with `query` required) | Open | `planning_phases.py` `_code_step` |
| `open_x with the uid of the Alpha match`, after a step code ran (`find_x query=Alpha`) | Pick: code sends `open_x x_uid=<uid>` when exactly one row of that run's Output holds `Alpha` whole, or it has one row. A step that names no lookup value (`open_x with the uid of the match`) runs only when the listing is the step right before it, shows no further page and has one row. The one unwritten required field must be a uid (`uid`, `id`, `*_uid`, `*_id`) or a field that declares its producer (`from_command_output`, read from that command's named list only). A uid containing `(` or `;` is quoted | `planning_phases.py` `pick_step`, `pick_listing`, `_is_uid_field` |
| `cmd x_uid=<the Alpha uid> other=v` after the lookup | Pick with the other values sent as written; the uid's must be the only placeholder | `planning_phases.py` `pick_step` |
| `cmd (full list)`; also `every page`, `all pages`, `until complete`, `page through` (any case) in the note | Code reads page 1, then each later page the output names, as more code-run steps, up to `FW_CODE_MAX_PAGES` (default 20) pages; then the step is open (see `all_pages` above) | `planning_phases.py` `FULL_LIST_PHRASES`, `step_paging`, `_next_page`, `code_max_pages` |
| `cmd start=20` (`start`) | That page only; code never computes a next page | `facts.py` `PAGING_PARAMETERS` |
| `cmd start=<n>`, or a step `next page with start=<n> until the list is read` | Open: the agent pages | `planning_phases.py` `_code_step` |
| any other note, or none, on a paged command | Page 1 is what the step wants; code goes on | `planning_phases.py` `step_paging` |
| a condition word before the command (or in its chain, next row): `if`, `only`, `when`, `whenever`, `unless`, `otherwise`, `optional(ly)`, `in case`, `as needed`, `where needed`, `where applicable`, `else` | Conditional: the agent decides (with no agent model, the user, when code can run the inner step); the finish check does not hold it; it stops code and keeps the phase from being answer-only | `finish_check.py` `_CONDITION_RE`, `_classify` |
| ANYWHERE in the step: `if needed`, `if necessary`, `if required`, `as needed`, `optional(ly)`, `skip when/if/unless`, `skip it when/if/unless` | Conditional, as above | `finish_check.py` `_ANYWHERE_CONDITION_RE` |
| `cmd k=v only when ...` (a condition word in the command's chain: after it, up to the first `(` or `;`) | Conditional, as if the condition came before the command, so the prose never joins the last value. A condition in a note (`cmd k=v (only when overdue)`) is never read: the step stays a plain command step, held to a run | `finish_check.py` `_classify` |
| a negation word before the command: `do not`, `don't`, `dont`, `never`, `none`, `not`, `no`, `nor`, `without`, `avoid`, `skip` | Negated (checked before conditions: `only when there is no match, cmd ...` is negated): code passes it over, never held to a run | `finish_check.py` `_NEGATION_RE`, `_classify` |
| first word an answer verb: `answer report present return say give summarize summarise state tell reply respond compare note identify end stop finish confirm verify conclude offer mention describe include combine format output provide write` (and `list_nothing`, `explain_answer`) | An answer step, even if it names a command later (`say what find_item found`): code passes it over, never held to a run | `finish_check.py` `_ANSWER_VERBS` |
| a step naming no command, no answer verb, no condition word (`read the notes`, `list the parts`) | "Other": not held to a run, but it stops code (an open step) | `finish_check.py` `_classify`; `planning_phases.py` `plan_code_steps` |
| `go_up` or `reset_context` alone | Code runs it; with prose after it, open | `planning_phases.py` `_CODE_NAVIGATION`, `plan_code_steps` |
| a phase whose steps hold no command, conditional or navigation step | Answer-only: no executor run, one model call with no tools (or none, below) | `planning_phases.py` `is_answer_only` |
| (answer-only) `say that P` | Verbatim, the only form code copies: code writes `P.` | `planning_phases.py` `answer_step_kind`, `_answer_line` |
| (answer-only) a head and its facts (`say which is riskier: A 3, B 5`, `say what X holds: a, b`, `compare ...: facts`) | Judgement, like any step but `say that`: only the author knows whether the facts already answer the head. To have code copy facts, write `say that what X holds: a, b` (`What X holds: a, b.`) | `planning_phases.py` `answer_step_kind` |
| (answer-only) no `": "` facts, starting `apply`, `do not`, `don't`, `never`, `make no`, `present them as options only`, `present it as options only`, AND a negation word (`no`, `not`, `none`, `never`, ...) | Guard: adds nothing; dropped from a code-written answer. Without a negation word (`apply a 10% discount to the total`) it is judgement | `planning_phases.py` `ANSWER_STEP_WORDS`; `finish_check.py` `_NEGATION_RE` |
| (answer-only) anything else (`say which is riskier`, `list the parts: a, b`) | Judgement: ONE such step and the model writes the whole answer (with no agent model, the command stops) | `planning_phases.py` `answer_by_code` |
| a step with `name`, `names`, `naming`, `list`, `lists`, `listing`, `every` or `each` outside the items, holding every item of a list fact (2+ items) | That list is checked: rows the answer leaves out are appended after "Also in the facts, not named above:"; the list is never cut | `planning_phases.py` `_NAMING_RE`, `_names_list_in_full`, `missing_list_items` |
| a list fact of more than `FACTS_MAX_LIST_ITEMS` (100) items joined by `", "`, `"; "`, `" \| "`, `","` or `";"` in a step with no naming word | Cut in the prompt to its first items and "(and K more; M in all)" | `planning_phases.py` `_LIST_SEPARATORS`, `capped_steps` |
| 4 or more items in a `", "` run of a code-written answer, each exactly an output value | Laid out as sub-bullets | `planning_phases.py` `MIN_BULLETED_ITEMS`, `bulleted_answer` |
| an empty string | Dropped | `planning_phases.py` `_checked_steps`; `planning_commands.py` `_call_steps` |
| a step whose first word is snake_case (or a known command): its command and the `key=` of its chain (before `(`, `;` or `,`) | Checked at load against the workflow's commands and their `Signature.Input`; an unknown one fails the load | `planning_commands.py` `step_commands`, `check_step_commands` |

Consequences worth designing for:

- Code runs only the LEADING mechanical steps and stops at the first open step. Put the steps that
  need judgement last, or in a phase of their own. With no agent model, an open step that is not a
  decision step ends the command there, so a command meant to run without an agent model is all code
  steps and answers with `__result__` or `say that` lines (`command run-code` shows where it stops).
- Write a filter as an argument (`list_parts kind=gear`), never as a note: the note is not sent.
- Write an answer step's facts as the answer should read them: in a verbatim step every word after
  `say that` is copied, instructions included.
- Make the last phase answer-only when its steps can carry every fact, so code (or one call without
  tools) writes the answer; keep a judgement step only where a judgement is needed.
- Repeat a step with a loop over `range()` bounded by an int field's `le`, and share steps through a
  helper module of `_commands/_shared/`.

## Approval annotation (`@user_approval_for_execution`)

An execution command may carry `@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)`
or `@user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)` on its `ResponseGenerator`,
imported from `fastworkflow` (a planning command cannot: see below). Use `REQUIRED_ALWAYS` for an action that cannot be
undone, such as sending a critical email; `REQUIRED_UNTIL_DELEGATED` when the user may hand the decision
over. Leave the decorator out for everything else: the agent decides, with no gate. Any value that is not
a `UserApproval` member fails the load and names the command and its file.

The annotation is enforced. `REQUIRED_ALWAYS` asks before every run, including a run the user typed, and no standing approval
ever applies to it (its question offers only `yes` and `no`; a typed `always` is asked again);
`REQUIRED_UNTIL_DELEGATED` asks the agent's calls until a standing approval exists (the user's own typed
command runs unasked). The reply is `yes`, `no` or, for `REQUIRED_UNTIL_DELEGATED` only, `always`,
bound to the exact parameters, single-use, and it expires. The listings mark the command. A planning command
cannot carry the decorator (a load error): it needs the strongest approval among the commands its steps reach
(`approval_annotations.command_approval`), and each of those calls is asked about when it runs. It is not read/write gating: an easy-to-undo write needs no annotation. Full rules are in
`docs/planning_command_file_layout.md`.

## Commands that write (`@writes`)

An execution command that changes the application's data declares it on its `ResponseGenerator`:

```python
@writes    # adds a note, cancels an order, sends a message
class ResponseGenerator:
```

Absent means the command only reads, or only moves the workflow between records (navigation is
`@descends_to`). A planning command does not declare it: it writes when one of its steps (directly, or
through a planning command it calls) runs a command that writes, and it is read-only otherwise
(`command_writes.write_commands`). `@writes` on a `PlanGenerator` fails the load.

Inside a planning command's phase, a command that writes runs only when a step of that phase names it,
and no more often than the steps name it: the phase's listing leaves out the other write commands, and
any other call of one is refused unrun with a message the agent reads (`command_writes.phase_write_refusal`,
checked by the executor before the approval gate). So a read-only planning command changes nothing, and a
writing one makes only the writes its steps name. Outside a phase (the turn's own agent, a typed command)
nothing changes.

Declare `writes` on every command that changes data, including the easy-to-undo ones that need no
`@user_approval_for_execution`: the two are independent. Without it, the library's `author_books` (find
books, open each) added notes to the books it opened, the phase agent copying `add_note`'s description
example. A phase that should write names the command in a step (`run("add_note", text=input.text)`); a
step in open words ("add a note saying ...") names none, and the write is refused.

## Modes

In agent mode a planning command runs as above. In assistant mode with an agent model set (`/` messages,
`fastworkflow run_cli --assistant`), a planning command runs only when it is mechanical: code runs every step
and `__result__` writes the answer. One that needs the agent (a prose or conditional step, a phase decided
by facts, a `classifier_confirms` input, no `__result__`) is refused with reasons that name each step
(`planning_needs_agent.planning_command_needs_agent`). With no agent model it runs to its first open step,
as in step 5 at the top.

## Checklist

```
- [ ] The job needs several existing commands in a known order (else: a command)
- [ ] File stem is the planning command's name; the Signature docstring says the job, purpose first; plain_utterances are generic
- [ ] Input types are str / bool / int / float / list[str] / Literal / Optional[str] = None; bools default false
- [ ] An input for every way users scope the question (branch, date range, owner), with the "all" default
- [ ] Every promise of the description reaches the answer (as a fact in the answer step, or __result__)
- [ ] Free text the command stores is copied_from_command_text, plus a validator for the instruction carried along
- [ ] Every execution command its steps run that changes data carries `@writes` on its `ResponseGenerator`
- [ ] Signature + top-level PlanGenerator; each phase a method with self, in run order; helpers _named
- [ ] Every fact field is typed with a command's Output or Input (or the phase returns a command's Output)
- [ ] Every Input handle that another command supplies carries `from_command_output` (not a hand-written `available_from`)
- [ ] Input-dependent shape in __phases__; a skip is `if not <what it needs>: return []`;
      a phase that may be skipped is tested for None where it is read
- [ ] No __call__; a Signature.Output has a __result__ that builds it
- [ ] Inherits only from a PlanGenerator in a _ module; no nested call cycle
- [ ] The template (preview) reads well: wording that tests facts lives in helpers
- [ ] Each step checked against the grammar table (code-run, held, answered as intended)
- [ ] command run-code stops where you expect (or answers, for a command meant to run with no agent model)
- [ ] command preview written and committed; --check passes in CI
```

## Related

- Reading a phased planning command call in a trace (`fw.phase` spans, `code_stop`,
  `answer_by_code`, `facts_error`, `fw_not_run`): the `debug-workflow-conversations` skill.
- The commands a planning command's steps name, and their `Output` fields its facts copy: the
  `integrate-chat-agent` skill's reference for command files.
- The `from_command_output` producer hints on `Input` fields, and their rules: the `declare-parameter-producers`
  skill.
- How the which-command question and the local intent model treat descriptions and utterances: the
  `evaluate-intent-routing` skill.
- Whether a planning command change helped, measured on a fixed task set: the
  `optimize-workflow-with-feedback` skill.
