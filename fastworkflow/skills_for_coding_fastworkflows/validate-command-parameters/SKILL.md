---
name: validate-command-parameters
description: >-
  Write the model validators of a fastWorkflow command's `Input` so whole-call rules, cross-field
  constraints, context preconditions and value lookups are checked and corrected before the command
  body runs, and so a bad call stays a conversation instead of becoming an exception. Covers the exact
  contract (`@model_validator(mode="after")` that corrects by assigning `self` or refuses with
  `fastworkflow.Refused(message, field=...)`), the validation context (`info.context["command_text"]`,
  `["workflow"]`, `["command"]`), where it sits in the validation order, writing error messages an
  agent can act on, and what belongs here versus in the field declaration. Also covers resolving a typed
  value against live keys (`DatabaseValidator.fuzzy_match`: typo correction, "did you mean" suggestions,
  thresholds, the key set). Use when a command has rules spanning more than one field, needs a precondition
  on the current record, must normalise input before executing, names a real-world entity the user types by
  hand (a person, product, room, tag) whose spelling keeps missing stored data, or when authoring or
  reviewing any command `Signature`.
---

# Validating extracted parameters

Pydantic answers one field at a time: right type, matches the pattern, was supplied. It cannot answer
the questions that only exist once the whole call is on the table: is this end date before its own
start date, is anything selected for this bulk command to act on. Those rules are model validators on
the command's `Input`. That is the one validation mechanism a command has, for an execution command and
a planning command alike: the same validators run on a planning command's arguments. (The old
`validate_extracted_parameters` hook is removed: a `Signature` that still defines it fails the load, and
the load error names the validator form.)

## The contract

A `@model_validator(mode="after")` on the command's `Input`, written as an instance method. Take `info`
only if you read the validation context; `def rule(self)` works too.

```python
from pydantic import BaseModel, model_validator

import fastworkflow


class Signature:
    class Input(BaseModel):
        begin_date: str
        end_date: str

        @model_validator(mode="after")
        def end_is_not_before_begin(self, info):
            if self.end_date < self.begin_date:
                raise fastworkflow.Refused(
                    f"end_date {self.end_date} is before begin_date {self.begin_date}; "
                    "give an end date on or after the begin date.",
                    field="end_date")
            return self
```

- **Correct** a value by assigning to `self.<field>` and returning `self`. The corrected value is what the
  command body reads. A correction is kept when the validators pass and dropped when any of them refuses.
  Return `self`: returning anything else is a call error.
- **Refuse** by raising `fastworkflow.Refused(message, field=...)`:
  - With `field=` (a name or a list of names, each a field of the `Input`): those fields become open
    again, the others are kept, and the user (or the calling agent) is asked for them with your message.
  - Without `field=`: the call does not fit, and no answer can fix it, so nobody is asked. An execution
    command ends as a failed output carrying your message. A planning command's call goes to the text
    planner or the agent.
  - A plain `ValueError` (or `AssertionError`) is a refusal with no field, but its message reaches the
    user prefixed with `validate_extracted_parameters:`. Prefer `Refused`.
  - Any other exception is a bug, not a refusal: it is logged at critical and the call fails with a
    message naming the exception.

The validator reads the validation context from `info.context`, with the same keys for both kinds of
command:

- `info.context["command_text"]`: the text the values were read from (the request, or the agent's
  command text), followed by the user's replies to parameter questions so far. `""` while a planning
  command's template renders.
- `info.context["workflow"]`: the workflow (`workflow.record` is the record this command acts on). It is
  `None` while a planning command's template renders, so a planning command's validator that needs it
  returns `self` early then.
- `info.context["command"]`: the qualified name of the command.

The framework always validates with this context, so no `None` guard on `info.context` is needed. Code
that builds an `Input` directly (a test) has no context: use `Signature.Input.model_construct(...)` or
`model_validate(values, context={...})`, or the framework reports an error naming the validator. Several
model validators run in declaration order, each on the instance the previous one returned.

## What already ran before your validator

Validators run after the checks below, only when they passed (a planning command's guard against two inputs
given one value runs after the validators, so a validator may correct one of them first), so **every required field is present and
of its declared type**. An optional field may be unset (its default, `None` when it declares none): test
it before you use it. Do not re-check what the framework already checks:

- pydantic's type, `pattern`, constraints and field validators, and required-ness;
- a fixed vocabulary: a `Literal` or `Enum` type (the one way to declare allowed values: an `enum` key in
  `json_schema_extra` fails the load, and `One of: a, b, c` in a description is only prose);
- the strict check, only on a field marked `copied_from_command_text` (or `record_named_in_command_text`): a `str`
  must be written in the request (case, dashes, quotes, brackets and trailing punctuation aside), an `int` must be a
  number the request writes (with or without any number in it; never replaced). An unmarked `str` or `int` keeps the
  model's value; `Field(pattern=...)` only validates (a `pattern` key in `json_schema_extra` fails the load), and the
  model is told the pattern. A field's `examples` are not shown to the extraction model and nothing refuses a copied
  example;
- made-up values, refused or treated as not given, on every `str` field: only punctuation, a template token or bracketed
  placeholder, an invented example name, a command's name, a value that is only a pronoun;
- the field options (`copied_from_command_text`, `record_named_in_command_text`, `classifier_confirms`,
  `classifier_decides`).

What is left for a validator is exactly what field declarations cannot express: rules across fields,
preconditions on the workflow, normalisation, and lookups against live data (for a typed name resolved
against live keys, see "Resolve a value against live keys" below).

**Upgrading a `validate_extracted_parameters` hook.** The hook ran before the `Input`'s own model
validators. When you move its body into a `@model_validator(mode="after")`, declare that validator
**first**, or an existing validator sees the uncorrected value.

## What happens after a refusal

- When the model extracted the values, a refusal makes it read the request **once more**, with your message
  after it, and the new values go through every check again. This happens at most once per extraction and
  never for a value that is simply missing. So your message is read by the extractor too: say what the right
  value is, not just that this one is wrong.
- A refusal that still stands with `field=` becomes the parameter question. Each reply is added to
  `command_text` and the values are checked again, validators included. The question is asked again while
  each reply fixes something; a second reply in a row that fixes nothing stops it (the agent or planner takes
  over when there is an agent model, otherwise the command ends naming what is missing), and `cancel` always
  ends it.

## Correct rather than refuse

Prefer normalising to refusing whenever the intent is unambiguous. Every refusal costs a round trip; a
value that can only have meant one thing should just be fixed (from the retail example):

```python
        @model_validator(mode="after")
        def order_id_has_hash(self, info):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'
            return self
```

## Write the message for whoever reads it

The message is all the user, the agent and the extractor have to work from. Name the field, say what is
wrong with the value that arrived, and say where an acceptable one comes from. "Invalid input" makes the
agent retry the same value. Check everything, then raise **one** `Refused` that names every bad field, so one
round trip settles all of it:

```python
        @model_validator(mode="after")
        def dates_and_quantity(self, info):
            errors, fields = [], []
            if self.end_date < self.begin_date:
                errors.append(f"end_date {self.end_date} is before begin_date {self.begin_date}.")
                fields.append("end_date")
            if self.quantity > self.in_stock:
                errors.append(f"quantity {self.quantity} is more than the {self.in_stock} in stock.")
                fields.append("quantity")
            if errors:
                raise fastworkflow.Refused(" ".join(errors), field=fields)
            return self
```

## Keep it local and cheap

The validator runs on **every** attempt at the command: each extraction, the one re-read, and each reply.
Do not call a backend from here. Check against values already in the workflow, or a constant. A live key set
is the one case that needs the workflow's record: see "Resolve a value against live keys" below.

## Context preconditions belong here

This is the only place that sees the workflow before the body runs, so "is this command meaningful right
now" goes here, as a `Refused` with no field (no answer to a question can fix it). Name the command that
fixes it:

```python
        @model_validator(mode="after")
        def something_is_selected(self, info):
            selection = getattr(info.context["workflow"].record, 'selection', None)
            if not selection:
                raise fastworkflow.Refused("Nothing is selected. Use select_items first, then retry.")
            return self
```

It runs on every path the framework runs the command by (a request, an agent's call, a direct action), but
it is a command's own declaration of what it needs, not a security boundary: code that reaches the body
another way bypasses it. Invariants that must hold for correctness or safety belong in the command body as
well.

## Declare the format of id inputs on the field

An input that takes an identifier (a uid, a record number, a key) must say what an identifier looks like,
on the field, with the format the application really mints, and name its producer (from the library
example):

```python
class Input(BaseModel):
    book_uid: str = Field(description="the uid of the book to open",
                          json_schema_extra={"from_command_output": "find_book.book_uids"},
                          pattern=r"^b[0-9]+$")
```

Without it, a `str` id field accepts anything, so given "open the book Dune Messiah" the extractor may fill
`book_uid="Dune Messiah"`, and the run fails on it or opens something else. With the pattern the value is
refused before any validator, on every path, and the id is found the way it would be with no value (the
producer command, or the agent). Match the whole value (`^...$`) and every id the producer returns: read
where the application mints them. Do not move the format into a validator.

## Resolve a value against live keys

A name the user types from memory ("Jon Smyth") rarely matches stored data exactly. Resolve it in a model
validator with `DatabaseValidator.fuzzy_match` (`fastworkflow.utils.signatures`): a typo is corrected in
place, a near miss comes back as "did you mean" suggestions, and a value nothing comes near is accepted for
the command body to judge. (`db_lookup`, a `json_schema_extra` key or a `Signature.db_lookup` method, is retired
and fails the load.) The field needs no special key. This is the shipped `messaging_app_4` command
`ChatRoom/set_current_user`:

```python
from fastworkflow.utils.signatures import DatabaseValidator

        @model_validator(mode="after")
        def user_name_is_in_the_room(self, info):
            key_values = info.context["workflow"].record.list_users()
            matched, corrected, suggestions = DatabaseValidator.fuzzy_match(self.user_name, key_values)
            if matched:
                self.user_name = corrected             # a correction, kept when the validators pass
            elif suggestions:
                raise fastworkflow.Refused(
                    f"No user named '{self.user_name}' in this chat room. Did you mean {', '.join(suggestions)}?",
                    field="user_name")
            return self                                # no suggestions: accepted
```

The refusal is what asks again, and `Refused` has no suggestions argument: the suggestions reach the extractor
and the user only as words in your message, so put them there. A non-empty suggestion list is what refuses
the field, so a value that must exist is enforced by an explicit `Refused` or in the body, not by hoping the
matcher rejects it. On a planning command's `Input`, `info.context["workflow"]` is `None` while its template
renders: return `self` first.

`fuzzy_match(value, key_values, threshold=0.6, suggest_threshold=0.7, auto_apply_threshold=0.2,
near_tie_margin=0.08)` returns `(matched, corrected, suggestions)`. It ignores case, spaces, `@` and `_`, and
runs three stages, each short-circuiting:

1. **Exact** (case-insensitive): matched.
2. **Levenshtein**, edits divided by length against the best-matching stretch of each key, so a value contained
   in a key scores 0 (`John` matches `John Smith` when no other key contains it). Keys within `suggest_threshold`
   are candidates. The best is matched only when it is unique, within `auto_apply_threshold`, and no other
   candidate is within `near_tie_margin` of it; otherwise every close candidate is a suggestion, nearest first
   (`Sarah Coen` against `Sarah Chen` and `Sarah Cohen` is asked, not guessed).
3. **`difflib`** at `threshold`, only when nothing came within `suggest_threshold`: suggestions only.

`auto_apply_threshold` is deliberately strict (a one-edit typo on a value of four characters or fewer exceeds it
and is offered as a suggestion): generous values silently substitute non-members, so raise it per command only
when an extra question costs more than a wrong substitution. A loose `threshold` gives unrelated suggestions,
and any suggestion refuses the field. Keep the defaults without a measured reason, or match with your own logic:
the framework reads only what the validator writes to `self` and whether it raises `Refused`.

The key set is the design decision: read it at call time from `workflow.record`, never a module constant; scope
it to the current context; use the labels the user says, not ids; keep it small (thousands of similar strings
suggest for everything) and cheap (the validator runs on every attempt and reply, so cache a set that needs a
network call on the application object). Do not use it for an opaque handle the user never types
(`from_command_output`: the `declare-parameter-producers` skill), a fixed vocabulary (`Literal` or `Enum`) or a rule between fields (above).

| Symptom | Cause | Fix |
|---|---|---|
| Valid values rejected with irrelevant suggestions | `threshold` too loose, or the key set wrong | tighten `threshold`; check the key set is the one the user names |
| The command acts on the wrong entity | `auto_apply_threshold` too generous, or a short value contained in one key | lower it; scope the key set |
| An obvious typo costs a question | a short value, or a runner-up within the margin | raise `auto_apply_threshold` for that command |

## Authoring checklist

```
- [ ] Type, pattern, allowed values and required-ness are declared on the field, not re-checked here
- [ ] Every id input declares the application's id format (`pattern`), so a name is never a valid id
- [ ] The rules are @model_validator(mode="after") methods that return self
- [ ] Unambiguous values are normalised by assigning to self.<field>, not refused
- [ ] Every refusal is raise fastworkflow.Refused(message, field=...); field= only when an answer can fix it
- [ ] Each message names the field, the bad value, and where a good one comes from
- [ ] Errors accumulate into one Refused
- [ ] Optional fields are tested before use; a planning command's validator returns early when workflow is None
- [ ] No backend calls
- [ ] A typed name is resolved with `fuzzy_match` on a key set read from `workflow.record`: matched assigns, suggestions go in the `Refused` message
- [ ] Safety-critical invariants are also enforced in the command body
```

## Related

- Parameters that never arrived: the `declare-parameter-producers` skill.
