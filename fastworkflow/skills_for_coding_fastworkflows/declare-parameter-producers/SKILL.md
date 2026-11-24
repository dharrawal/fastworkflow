---
name: declare-parameter-producers
description: >-
  Declare producers on fastWorkflow command parameters with `from_command_output` so the planner can
  chain commands into a working sequence and so a missing parameter tells the caller which command
  supplies it. Covers the `from_command_output` form (`<command>.<field>` references),
  how the planner, the parameter listing and the missing-parameter message consume it, choosing a
  producer the caller can reach, matching handle types between producer and consumer, and what
  `command preview --check` verifies. Use when a command takes an id, uid, code or other opaque handle it
  cannot invent, when an agent cannot work out the order to call commands in, or when authoring or
  reviewing any command `Signature.Input`.
---

# Declaring parameter producers with `from_command_output`

A command that takes an opaque handle (an id, uid, code, ticket number) cannot be called until
something produces that handle. `from_command_output` records which command that is, turning a flat list of
commands into a graph the planner can walk.

## The declaration

On the `Input` field, as a `json_schema_extra` key of a plain `Field` (from the shipped `library` example,
`Library/open_book.py`):

```python
class Signature:
    class Input(BaseModel):
        book_uid: str = Field(
            description="the uid of the book to open",
            json_schema_extra={"from_command_output": "find_book.book_uids"},
            pattern=r"^b[0-9]+$",
        )
```

The value takes two forms:

- `"[<Context>/]<command>.<field>"`: a field of the producer's `Output`. It says which value the consumer
  needs, and the loader checks the field's type.
- a list of those, when several commands can supply the handle:
  `json_schema_extra={"from_command_output": ["A/x.id", "B/y.id"]}`.

A bare command name (`"find_book"`, no field) is not accepted: it fails the load of the command that wrote it,
naming the producer's `Output` fields of a fitting type. When the producer has no field that holds the
handle (it returns one block of text), leave the declaration out and describe the handle in `description`,
or add a typed field to the producer's `Output`.

A name without `<Context>/` must be unique on the consumer's own lookup path (its context, that context's
bases, then the global commands). Qualify it when the producer lives elsewhere or the name is ambiguous.

Only an `Input` field may declare it; on an `Output` field or a facts model field it is an error. There is no
`from_` helper (importing `fastworkflow.planning.from_` fails with the spelling above). Do not write
`available_from` yourself: the loader (`planning_commands.resolve_field_sources`) writes it from
`from_command_output`, and a hand-written one is an error.

A declaration that does not resolve (a bare command name, a command that does not exist or is not on the
lookup path, a field the producer's `Output` lacks, a type that does not fit) fails the load of the command
that wrote it, as any command file that does not load does: `command_preview.load_problems` lists it with the
file, the field and the reason (`fastworkflow run` prints it before its first prompt, `fastworkflow train`
stops on it and the server logs it at startup), the command is not available and the rest of the workflow
loads. A call cycle between planning commands, a hand-written `available_from` and a producer on an `Output`
or facts field are other errors: they are logged and `command preview --check` fails on them, and the command
still loads.

## Who reads it

1. **The planner.** Its instructions (`workflow_agent.PLANNER_INSTRUCTIONS`) say to walk the graph of
   these edges to build the command sequence, and the command listing shows each resolved edge as
   `available_from` beside the parameter. The edges are the only structured statement of "call X before
   Y".
2. **The listing of a large workflow.** A listing longer than 8,000 characters is cut to a shortlist when
   the workflow has trained intent models (`run_listing.py`): the commands gathered for the request, the
   producers of their inputs, and the `@descends_to` commands that lead to their contexts. A missing
   edge is a producer the planner never sees there.
3. **A later part of the same request.** When a request is split and an earlier part runs the producer
   (as its command, or as a step of its planning command), the consumer's field is `from sub-task N`:
   it is not extracted and never asked for. With `<command>.<field>` and a JSON output, the value is that
   field (the element itself when the list has one); otherwise it is the part's whole output
   (`review_proposals._producers`, `_produced_fields`).
4. **Steps run by code in a planning command.** A step that leaves one required field open, where that
   field is a uid or declares a producer, is run by code on the one matching row of the preceding
   listing; `run(command, named=...)` opens a row by the field that declares `from_command_output`
   (`planning.row_id_field`). Such a field is a value to be found, never one a decision step asks a
   person for.
5. **An execution command's missing-parameter question.** When the field is missing, the question adds
   `use the <producers> command(s) to get <field> information` (`parameters/ask.py`), preceded by
   `abort and` for a person; an agent (`run_as_agent` set on the workflow context) is told only to use
   it. A planning command's own parameter question does not name producers.
6. **`command preview --check`**, below.

The hint appears only when the field is missing, so it never replaces `description`; write both.

## Rule 1: name a producer the caller can reach

A hint naming a command the caller cannot invoke sends the planner down a dead end. Reachable means on
the lookup path intent detection walks: the current context and its bases, each live ancestor record
and its bases, then the global commands (`context_navigation.exact_name_scopes`). A **sibling** context is
not on it. Verify against the context model, not the folder layout.

## Rule 2: the producer must emit the handle the consumer accepts

Two commands can both return "an id" and still not be interchangeable.

- **A parent-scoped consumer accepts only its own children.** If it checks membership in the parent's
  view, a global search returns handles it rejects. Name the parent-scoped listing, and do not add the
  global search to the list "just in case": every handle it yields is one the consumer rejects.
- **When the handle parameter is optional, the command can be its own producer.** A command that opens
  one child when given its uid and lists the children when not is the only thing that yields uids it
  accepts: point the field's `from_command_output` at the command's own list field. This is not a cycle:
  call it bare first, then again with a returned handle. (A planning command's `Input` may not name a
  command its own steps call; that is a `--check` error.)

## Rule 3: describe the handle at both ends

Have the producer's `Output` field name the input it feeds, as `Library/find_book.py` does:

```python
class Output(BaseModel):
    book_uids: list[str] = Field(description="the uid of each book found, in row order; "
                                             "open_book takes one")
    titles: list[str] = Field(description="the title of each book found, in row order")
```

When an output carries two different handles, say in each description that they are different and
what each opens.

## Rule 4: keep the edges true as the tree changes

Moving or deleting a producer breaks every edge to it; `--check` reports an edge that no longer
resolves. It does not re-check reachability (Rule 1), so re-check that whenever a context's bases or
parents change.

## What `command preview --check` verifies

`fastworkflow command preview <workflow_folder> --check` exits 1 on any error from
`planning_commands.resolve_field_sources` over every execution and planning command, and the first four
checks below also fail the load of their command (`load_problems`). For each edge:

- the reference is well formed (`[<Context>/]<command>.<field>`, a field always named);
- the producer exists: a qualified name anywhere, an unqualified one once on the consumer's own lookup
  path (its context, its bases, the global commands);
- a planning command's `Input` does not name a command its own steps call;
- it is a field of the producer's `Output` (not its `Input`) whose type matches the
  consumer's (equal, equal up to `Optional`, or a list of the consumer's type);
- no hand-written `available_from`, and no `from_command_output` on an `Output` or facts model field.

It does not check Rules 1 to 3: live-ancestor reachability, whether the producer's handles are ones
the consumer accepts, or whether the descriptions say what the handles are.

## Authoring checklist

```
- [ ] Every handle parameter the caller cannot invent carries a from_command_output edge
- [ ] The edge names an Output field (command.field), qualified when not on the static lookup path
- [ ] Each producer is reachable from the consuming context (Rule 1)
- [ ] Each producer emits handles this consumer accepts; a parent-scoped consumer names its own listing
- [ ] An optional-handle command that lists when bare names itself
- [ ] The producer's Output field names the input it feeds in its description
- [ ] `description` is written too: the hint appears only when the field is missing
- [ ] `command preview --check` passes
```

## Related

- For "the parameter arrived unusable" see `validate-command-parameters`; for "the parameter arrived
  nearly right", the same skill's "Resolve a value against live keys".
- Reachability is decided by the context models: `design-context-models`.
- Planning commands that use these edges: `author-planning-commands`.
