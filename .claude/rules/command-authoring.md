---
paths:
  - "**/_commands/**"
---

# Authoring fastWorkflow Commands

## Workflow Directory Structure

```
my_workflow/
├── application/                     # Your app code (untouched)
├── _commands/                       # Command implementations (generated + edited)
│   ├── startup.py                   # Global command that builds the root object
│   ├── <command_name>.py            # Global command: single file (preferred)
│   ├── _workflow.py                 # Optional: the workflow's settings (class Context)
│   ├── _shared/                     # Optional: helper modules (__init__.py + modules), never read as commands;
│   │                                #   import with from ._shared.x (root) or from .._shared.x (context folder).
│   │                                #   Any other _name.py or _folder in _commands/ is a load error.
│   └── <Context>/                   # One folder per kind of object, named after its class
│       ├── _<Context>.py            # Optional: class Context (inherits, parents, get_parent, get_displayname, ...)
│       └── <command_name>.py        # A command of that context
├── ___command_info/                 # Generated at train-time (gitignore)
├── ___workflow_contexts/            # Session state at run-time (gitignore)
└── ___convo_info/                   # Conversation logs (gitignore)
```

Add to `.gitignore`: `___workflow_contexts`, `___command_info`, `___convo_info`

## Command File Structure (Single-File Pattern)

New commands use a **single `.py` file** in `_commands/`. Legacy commands with subdirectories (`parameter_extraction/`, `response_generation/`, `utterances/`) are being migrated to this pattern. When you encounter both, use the single file.

```python
# _commands/<command_name>.py

class Signature:
    plain_utterances: list[str] = [...]   # Seed utterances; the framework adds the synthetic variants

    class Input(BaseModel):               # Pydantic params; no default = required (asked for if missing)
        some_param: str = Field(description="...", examples=[...])
        note: Optional[str] = None        # Optional[...] = None = optional

        @model_validator(mode="after")     # Optional; see the validation section below
        def some_rule(self, info):         # info.context: "command_text", "workflow", "command"
            ...                            # assign self.<field> to correct, then return self;
                                           # raise fastworkflow.Refused(message, field=...) to refuse

    class Output(BaseModel):              # Structured result
        ...


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        # input is None when the command has no Signature.Input.
        # Return the Output (the framework fills the envelope), a str (response text),
        # or fastworkflow.CommandResponse (artifacts or next actions only).
        # Refuse with raise fastworkflow.Refused(message).
        ...
```

A two-argument `__call__` (or one whose third argument has a default) is refused with "must take three arguments", but the error is raised only when the command's class is first used: `load_problems` does not list it. `CommandOutput` is not a return value.

## `plain_utterances` — the seed list is the highest-leverage thing you write

`plain_utterances` is not decoration. Training expands it with an LLM into the synthetic
corpus that the intent classifier learns from, so seed count and seed variety are the
single largest measured input to whether the right command gets picked at run time —
larger than the persona/utterance-count dials, larger than anything in the training loop.

Guidance, with its provenance stated honestly:

- **Aim for roughly eight seed utterances per command.** On one 160-command workflow
  scored against a 446-case hand-written benchmark, everything else held constant,
  held-out routing top-1 went 46.2% at 3.2 seeds/command → 70.4% at 8.0 → 73.8% at 9.3:
  steep, then flat. **That is an observation from a single workflow, not a universal
  constant** — the shape (steep early, flattening) is what transfers; the exact knee will
  differ for you. Two or three seeds is the starved case worth fixing first.
- **Vary the phrasing family, not the wording.** Imperative, question, colloquial, terse,
  synonym-heavy, and value-bearing phrasings each buy something. Six paraphrases of one
  sentence buy almost nothing.
- **Never paste a failing benchmark case into `plain_utterances` to "fix" it.** That turns
  your benchmark into a memorisation test. Keep the seeds disjoint from
  `<workflow>/intent_benchmark.json` (see `docs/intent_benchmark_format.md`);
  `heldout_evaluation.assert_benchmark_disjoint_from_seeds` is called during training
  (`fix-eia`, fixed) and aborts the run with `BenchmarkLeakError`, so a leak fails the
  build instead of quietly inflating the score.
- **Count what you actually have** after a run: `seed_utterance_count` per command in
  `<workflow>/___command_info/training_provenance.json`. The same file's `fell_back` flag
  tells you whether a command's utterances degraded to seeds-only because generation was
  rate-limited.

Judging whether adding seeds actually helped is a measurement problem with its own
pitfalls (two training runs are not reproducible): see the
`fastworkflow-intent-training-convergence` skill before comparing two runs.

## Validating parameters: model validators on `Input`

Cross-field rules, context preconditions and value lookups are `@model_validator(mode="after")` methods on the
command's `Input`. That is the one validation mechanism: the `validate_extracted_parameters` hook is removed, and a
`Signature` that still defines it fails the load with the validator form in the message. Pydantic checks types,
patterns and required fields first, so a validator can assume every field is present and of its type.

- Correct by assigning `self.<field>` and returning `self`.
- Refuse with `raise fastworkflow.Refused(message, field=...)`. A plain `ValueError` is a refusal with no field.
  Any other exception becomes a call error naming the command, so raise `Refused` for any refusal you mean the user to see.
- `info.context` has `"command_text"` (the user's text, `""` while a planning template renders), `"workflow"`
  (`None` while a planning template renders) and `"command"` (the qualified name), for both kinds of command.
  Code that builds an `Input` directly, with no context, gets a clear error: use `Input.model_validate(values, context={...})`
  or `Input.model_construct(...)` in tests.

**Upgrade step (migrating a hook).** The old order was option checks, then the hook (which could correct fields), then
the Input's own model validators, then roles. Pydantic runs model validators in declaration order, so when you move a
hook body into a `@model_validator(mode="after")`, declare that validator **first**, before any model validators the
`Input` already has. Otherwise an existing check sees the uncorrected value. (In one workflow, a kind-fit check refused
before the kind resolver ran.)

A value lookup against a live set of legal values goes in a validator too. `db_lookup` is retired: a
`json_schema_extra={"db_lookup": True}` key or a `Signature.db_lookup` method is a load error.

```python
class Signature:
    class Input(BaseModel):
        user_name: str = Field(description="Name of a person")

        @model_validator(mode="after")
        def user_name_is_known(self, info):
            workflow = info.context["workflow"]
            if workflow is None:               # a planning command's template is rendering
                return self
            candidates = workflow.record.list_users()
            matched, corrected, suggestions = DatabaseValidator.fuzzy_match(self.user_name, candidates)
            if matched:
                self.user_name = corrected
            elif suggestions:
                raise fastworkflow.Refused(
                    f"No user named '{self.user_name}'. Did you mean {', '.join(suggestions)}?",
                    field="user_name")
            return self                        # else: accept
```

`DatabaseValidator` is `fastworkflow.utils.signatures.DatabaseValidator`. `workflow.record` is the record this command acts on. The three outcomes:

| `fuzzy_match` result | what the validator does | what the framework records |
|---|---|---|
| `matched` | assigns `self.<field> = corrected` | `changed_fields` lists the field |
| `suggestions` non-empty | `raise fastworkflow.Refused(msg, field="<field>")` | `refused_fields` lists the field; the field is reset to missing and asked again with the message |
| neither | returns `self` | nothing: the value is accepted |

The last outcome means "I have no opinion, leave the extracted value alone", and it is what you want
whenever a field can legitimately hold a value that is not in your candidate list. Any non-empty
suggestion list refuses the field, so feeding real uids to a field whose candidates are display names
rejects valid input. Decline instead, before the lookup:

```python
    if _looks_like_a_uid(self.user_name):
        return self                    # not a name; not mine to judge
```

**Assign the value the field holds, not the value you matched on.** A lookup that returns a display
label is right for a field holding a name. On a `*_uid` field, map the label back to the uid yourself:
`self.user_uid = uid_of[corrected]`.

**Tuning the match, per command.** `DatabaseValidator.fuzzy_match` runs a
case-insensitive exact match, then Levenshtein, then `difflib`, and each stage
short-circuits. Because the validator is your code, you tune it by passing arguments:

- `auto_apply_threshold` (default `0.2`): how far off a *unique* match may be before
  the framework rewrites the user's value. Deliberately strict: at `0.7`, `"Batman"`
  scored 0.333 against a customer list and was silently applied. Distance is
  edits ÷ length, so a one-edit typo on a value under ~5 characters exceeds `0.2` and
  is offered as a suggestion instead. Raise it if an extra clarification turn costs
  more than a wrong substitution; `0.0` accepts only exact and containing matches.
- `suggest_threshold` (default `0.7`): how far off a candidate may be to be offered
  at all.
- `threshold` (default `0.6`): `difflib` cutoff for the final suggestion-only stage.
  Only values that nothing came within `suggest_threshold` of reach it, so loosening
  this yields unrelated suggestions, which refuse the value.

## `command_text_after_command_name`: binding `command value` without the LLM

In agent mode, parameters are parsed in code before the LLM is asked: XML tags, then
`name=value` pairs, then the form `command_name value`. In assistant mode ("/" commands) the
`name=value` form is also parsed in code before the LLM, while the XML and `command_name value` forms
remain agent-mode only. The `command_name value` form binds the text after the command
name to one target field:

- the field marked `json_schema_extra={"command_text_after_command_name": True}`, or, with no mark,
- the class's only `str` field (an `int` paging field does not count).

An unmarked field takes a single token only (`open_by_uid 9f2c…`); a marked field also
takes several words (`find_item Ann Lee 2`, where the trailing `2` stays in the value).
Mark the field a bare value most naturally fills, when the class has more than one `str`
field or the value can contain spaces:

```python
class Input(BaseModel):
    query: Optional[str] = Field(default=None, description="Name or name fragment",
                                 json_schema_extra={"command_text_after_command_name": True})
    start: Optional[int] = Field(default=None, description="Paging offset")
```

Mark at most one field per class; two marks disable the form. Do not mark a free-text
field of a command that changes state, where prose sent by mistake would become the
value. The form declines (and the LLM extracts as before) when the first token is not
the command's own name, when the text holds `=` or XML, when the first word names
another field, or when the value holds `*`, `` ` ``, `?`, `\`, ` / `, `<>{}` or no letter
or digit. A leading word naming the target field itself (`uid 9f2c…`) is dropped, and
matching surrounding quotes are removed. The usual validation runs after.

A lone value that binds to no field (`command_name value` where the LLM fills nothing)
is refused with a message naming the parameters that could take it, instead of running
the command with its defaults.


## How run_chatbot renders commands (zero hints + optional `ui`)

`fastworkflow run_chatbot` draws a command's result and its parameter form from the schemas that
`GET /command_schemas` serves (each command's `Input` and `Output` JSON Schema). With no hints at all the chat is
already presentable; write these, which you write anyway, to improve it:

- `Field(title=..., description=..., examples=[...])`. A `title` is the label; without one the field name is made
  readable (`order_uid` becomes "Order UID"). `description` is the tooltip on a result label and the help line under a
  form field. For a text `Input` field `examples[0]` is the placeholder.
- `from_command_output`. When a later command of the same turn reads a field from an earlier command's `Output`, the
  earlier result is folded under "Supporting lookups" and the later one is the card.
- `classifier_confirms={"meanings": {value: meaning}}` on a `Literal`. The "/" parameter form shows each choice as
  `value - meaning`.
- **A field's name never decides how it is drawn.** Type and shape give the plain default (a stat strip of short
  scalars, a table for lists of records); everything else is a hint you declare. An `Output` field named `status` or
  `state` is plain text unless it carries `format="badge"`. In the results panel several same-length lists form a table only
  by a `table` group you declare (an int `total` beside equal lists no longer makes one there; the answer card still
  infers a table from that until the next step, fix-sush step 4). When a field the user is asked for has a type the parameter form cannot draw (an object),
  that prompt stays plain text.

Optional presentation hints go in `json_schema_extra` under one key, `ui`, with an optional TypedDict
(`FieldOptions(ui=UiOptions(...))`, both exported from `fastworkflow` and `fastworkflow.planning`). A hint changes
presentation only: no check, extraction or plan reads it.

| `ui` option | On | Value | Effect |
|---|---|---|---|
| `title` | any field (a `Field(title=...)` argument, not a `ui` option) | text | the label; without it the field name is made readable |
| `hidden` | `Signature.Output` field | `True` | left out of the result card (still in the JSON) |
| `hero` | `Signature.Output` scalar field | `True` | pinned to the stat strip |
| `format` | `Signature.Output` field | `"currency"`, `"percent"`, `"code"` or `"badge"` | `currency` shows a number with grouping and two decimals in the browser's locale, with no currency symbol (`1,234.50`), `percent` adds `%`, `code` uses monospace, `badge` a status chip (toned only for generic words such as ok, done, pending, failed, rejected; any other value is neutral) |
| `tones` | `Signature.Output` `str`, `Literal` or `Enum` field (or a list of one) with `format="badge"` | a dict of value to `"good"`, `"warning"`, `"bad"` or `"neutral"` | the badge takes the author's tone; any value not in the map keeps the generic vocabulary |
| `currency` | `Signature.Output` `int` or `float` field (or a list of one) with `format="currency"` | a three-letter ISO 4217 code, `"USD"` | the number is shown with that currency (symbol, digits) by the browser's `Intl.NumberFormat`; a code the browser rejects, or none, shows the plain grouped number |
| `table` | `Signature.Output` list-of-scalars field | a group name, a simple identifier (`"rows"`) | the list fields of one output with the same group name are the columns of one table, in field order |
| `widget` | `Signature.Input` field | `"textarea"` on a `str`, `"radio"` on a `Literal` | a multi-line box; radio buttons even for more than five choices |

```python
class Output(BaseModel):
    total: float = Field(description="Order total",
                         json_schema_extra=FieldOptions(ui=UiOptions(hero=True, format="currency")))
    raw_payload: str = Field(default="", json_schema_extra={"ui": {"hidden": True}})
```

`ui` is checked when the workflow loads. A mistake on an `Input` or an `Output` field makes that command fail to load (the rest of the workflow loads), and `fastworkflow run_cli`, `train`, the server and `command preview --check` list it (`docs/design_idioms/removed.html#load-rule`). A `ui` value that is not a dict, is empty, or is `False`; an unknown option (the
error suggests the closest: `did you mean 'widget'?`); `hidden` or `hero` that is not `True`; a `format` or `widget`
outside the lists above; an `Output` option on an `Input` (`hero`, `hidden`, `format`) or `widget` on an `Output`;
`"textarea"` on a field that is not a `str` (`Optional[str]` is fine); and `"radio"` on a field that is not a `Literal` all fail
with a message naming the field and what is accepted. `table` is refused when the name is not a simple identifier, on a field that
is not a list of `str`, `int`, `float`, `bool` or `Literal` values (`Optional` of such a list is fine; a list of objects is
refused, those already draw their own table), together with `hero`, on an `Input`, or when its group has fewer than two
fields or fewer than two that are not `hidden` (a `hidden` member still counts for alignment and is not drawn).

`tones` keys are the value as text, matched case-insensitively after trimming (an `Enum` member by its value, so `Level.LOW = 1`
is keyed `"1"`); on a `Literal` or `Enum` a key that is not a declared value fails the load, so a typo is caught; an empty map,
a tone other than the four, a key that is empty, two keys that differ only by case or spaces, a `bool` or `int` field (drawn as
yes/no or a number, never a text badge) and a missing `format="badge"` are refused. `currency` is stored as written and
upper-cased by the page; it is refused unless the field is an `int` or `float` with `format="currency"` and the code is exactly
three letters (no list of currencies is kept, so an unknown but well-formed code is accepted and the browser decides). Both are
refused on an `Input`. A workflow that declares neither renders as before.

`table` is how parallel lists become one table in the results panel; nothing infers it from a field name or from equal
lengths. (The answer card still infers one from an int `total` beside lists of one length, until the next step.) At render time a group draws a table only when every member is present and the members
share one length of at least 1; with unequal lengths, a missing or null member, or no rows the members render as bullet lists,
never trimmed or padded. Lists outside any group are drawn as plain lists. The panel never infers columns from equal lengths alone: two undeclared
lists of equal length, or two declared groups of equal length, are not merged.

To find the outputs that need the hint, `fastworkflow command preview <workflow> --check` prints a `notice:` (not a
warning or an error, and it never changes the exit status) for each command whose `Output` has two or more visible
list-of-scalar fields in no `table` group: those lists are drawn as separate lists unless you declare them as one table
with the `table` hint. Declare the group where the lists are index-aligned by construction; where they are not, nothing
needs to change (there is no hint that says "separate on purpose", so the notice stays).

```python
branches: list[str] = Field(json_schema_extra={"ui": {"table": "hours"}})
hours: list[str] = Field(json_schema_extra={"ui": {"table": "hours"}})
``` A mistake on an `Input` fails the command's load. A mistake on an
`Output` is reported as a field-source error by `fastworkflow command preview <workflow> --check` and skips the command at
load.

A "/" parameter form also needs your cooperation in two places. A `str` field with `format="password"` (or `writeOnly`)
is drawn masked; its value is sent as ordinary chat text when the user clicks Run, so it is recorded in the
transcript. Do not put a real secret in a "/" command's `Input`. An `Optional[bool]` field is a three-way choice
(unset, yes, no), so leaving it unset is not the same as `False`.

## Declare what a value looks like (checklist; full text in the `author-planning-commands` skill, "Declare what a value looks like")

- `Field(examples=[...])` on every text and id `Input` field: 2 to 5 realistic values with the length and shape real values
  have, never customer data, valid for the field's pattern and validators. Not shown to the extraction model; used for
  the question hint, the chat placeholder, the preview sample, generated few-shot examples, and the listing the agent
  reads (so keep them few: the full listing is whole only up to `FULL_LISTING_MAX_CHARS`).
- `examples` on an `Output` field that an `Input` points at with `from_command_output`; not rendered in the listing.
- Declare `ge`/`le`/`gt`/`lt` and `pattern` rather than prose, but only what the command already refuses: a declared bound
  makes validation refuse an out-of-range value. `pattern` is also told to the extraction model.
- A fixed set known in code is a `Literal`; a set from data is a `str` with a validator plus examples; "One of: a, b" is prose.
- Descriptions say what a value looks like; an `Input` description is part of the extraction prompt, so editing it changes
  that command's prompt.
- `plain_utterances` stay generic (they train which command); values go in `examples`. `words` and `examples_by` below are
  the training-only declarations.
- `personas.json` changes a global training input and every utterance-cache key: the next train regenerates all generated
  utterances (model spend).
- Any command file edit changes the training signature (retrain at the next `fastworkflow train`) and may stale the
  planning-command previews (`fastworkflow command preview <workflow> --check`).

## Training-only declarations: `words` and `examples_by`

Two declarations tell the synthetic utterance generator what only you know. They are **training data only**: no
check, extractor, planner, listing or preview reads them, and the extraction prompt, the command listing, the
classifier check and the preview are byte-identical with or without them (`tests/test_training_only_field_options.py`).
They do appear in the field's JSON Schema, so `GET /command_schemas` carries them. Both are checked when the workflow
loads, on planning and execution commands alike, and a mistake fails that command's load with a message naming the
field and the rule (misspellings too: `example_by` suggests `examples_by`, `word` suggests `words`).

- `words`, inside `classifier_confirms` on a `Literal` field: the words or short phrases users write for a value
  (users rarely write the value itself). Each key is a declared value, each list is non-empty text, a word is listed
  once per value (ignoring case) and under one value only; a value may have none. `{"words": {...}}` without
  `meanings` is the same check as `classifier_confirms: True`.
- `examples_by`, on the field whose example values depend on a choice field of the same `Input`: the one key names
  that `Literal` field (not this field; one choice field only), each inner key is one of its values, and each list
  holds values valid for THIS field by its type, `pattern` and bounds (field validators are not run on them).

```python
kind: Literal["identity", "group"] = Field(..., json_schema_extra={"classifier_confirms": {
    "meanings": {"identity": "one person", "group": "a set of people"},
    "words": {"identity": ["person", "employee"], "group": ["team", "role"]}}})
name: str = Field(..., examples=["Maria Okafor", "Payroll Approvers EMEA"], json_schema_extra={
    "examples_by": {"kind": {"identity": ["Maria Okafor", "Tom Reyes"], "group": ["Payroll Approvers EMEA"]}}})
```

Like any edit of a command file, adding them changes that workflow's training signature, so the next
`fastworkflow train` retrains it.

## Context declarations (`inherits` and `parents` on `class Context`)

A context is a folder of `_commands/`. Its `class Context`, in `_commands/<Context>/_<Context>.py`, declares two
lists of context names, written as string literals in the class body and read without importing the file:

- `inherits = ["Record"]`: the contexts whose commands this context also has (transitively, in list order; its own
  command of the same file name wins). `inherits = ["*"]` adds the global commands to its own set.
- `parents = ["Library"]`: every context a record of this context can sit in, i.e. the contexts whose records its
  `get_parent` returns; the context itself when records nest (`TodoList` declares `parents = ["TodoList",
  "TodoListManager"]`). No `parents` means a top-level context, enclosed only by the global commands; the root record
  needs neither.

From `fastworkflow/examples/library/_commands/Book/_Book.py`:

```python
class Context:
    inherits = ["Record"]
    parents = ["Library"]

    @classmethod
    def get_parent(cls, record: Book) -> Library:
        return record.library
```

Load checks (each stops the workflow with one line naming the file): a name that is no context (with a did-you-mean),
a name given twice, `[]`, a value that is not a list of string literals, a declaration assigned twice or inherited
from a Python base class, a context that inherits itself, an inheritance cycle, a folder with no command file that
inherits none, `parents` without `get_parent` or `get_parent` without `parents`, and `parents` on an
`occupiable = False` context. A command's `@descends_to("X")` target must list the command's context (or one it
inherits, or one inheriting it) among its `parents`, or that command does not load. At run time a `get_parent` that
returns a record of a context its `parents` do not name raises `MissingGetParentError`.

`_commands/context_inheritance_model.json` is a removed spelling and stops the workflow: run
`fastworkflow migrate-contexts <workflow>` (writes `inherits`, deletes the file, lists the contexts with a
`get_parent`) and write their `parents` by hand from what each `get_parent` can return. `fastworkflow build` prints the
`inherits` line it derives for each context and writes it only into a `_<Context>.py` it creates.
`fastworkflow command preview` writes the generated object model to `_object_models/<workflow>.md`.

To add a new command: create its `.py` file in the context's folder; declare `inherits` only to give the context a base.
