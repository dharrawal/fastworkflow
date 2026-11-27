---
name: design-context-models
description: >-
  Design and tune the context inheritance (`inherits`) of a fastWorkflow command tree, and the
  enclosing record each context's `get_parent` returns (declared as `parents`), so intent routing stays accurate as the tree
  grows and a task takes the
  fewest commands. Covers deriving the containment graph from code, sizing base contexts to the
  parameter producers commands actually declare, removing duplicate commands, writing seed
  utterances that separate colliding labels, and verifying a retrain without being fooled by
  training variance. Use when routing accuracy is poor, a context carries too many commands, two
  commands are near-duplicates, wildcard escalation misfires, or a new context, workspace or base
  is being added to a workflow.
---

# Designing context models for routing

## The mechanism being optimized

Two declarations on each context's `class Context` (in `_commands/<Context>/_<Context>.py`), two
unrelated effects. Confusing them makes every later decision guesswork.

| Declaration | Effect |
|---|---|
| `inherits = ["Base", ...]` | Adds the base context's commands to the context's **surface** (transitively). An own command with the same leaf name hides the inherited one. `"*"` adds the global commands. |
| `get_parent` with `parents = ["Ctx", ...]` | The enclosing record: `get_parent` returns it from the live record at run time, and `parents` names every context it can be (the context itself when records nest). The root record has no parent and the framework never asks it: its context class needs neither. No `parents` is a top-level context. |

Both are list literals read without importing the file and checked at load (names are contexts, no
duplicates, no self-inheritance or cycle; `parents` exactly when there is `get_parent`; a
`@descends_to` target's `parents` must name the moving command's context). At run time a
`get_parent` that returns a record of an undeclared context raises `MissingGetParentError`.
`fastworkflow command preview` draws both, with the commands that enter each context, in the
generated `_object_models/<workflow>.md`: read it instead of collecting the declarations by hand.
`_commands/context_inheritance_model.json` is a removed spelling that stops the workflow
(`fastworkflow migrate-contexts <workflow>` moves it into `inherits`; write `parents` by hand), and a
`context_hierarchy_model.json` is ignored with a warning that it can be deleted.

A context is named after its record's class, and its commands are the files in `_commands/<Context>/`.
Files directly under `_commands/` are the global context `*`.

### The lookup path

Every request is looked up along one path, nearest first:

1. the current context, then its bases;
2. each enclosing record on the live chain (`get_parent`, up to and including the root record), each
   followed by its bases;
3. the global context `*`.

A command found on the path runs on the record of the scope it was found in: an enclosing record's
command runs on that record, a global command on the root record.

### How a request is routed along it

In order, the first that decides wins:

1. **Exact name.** The request's first word (an optional leading `/`, any case) names a command:
   the nearest scope that has it. Two commands of that name in one scope are asked about.
2. **A name off the path.** The first word names a command of a context that is not on the path: no
   model is asked, and the reply says how to get there (the commands that enter that context, from
   their `@descends_to`).
3. **Misspelt name.** The whole request against each scope's command names, nearest scope first. Two
   names that match equally well are not a match.
4. **The local model.** In the default `embedding` layout one head is fitted over every command of the
   workflow, and each context reads it through its **mask**: its surface, its planning commands and
   the framework's four core commands (`go_up`, `reset_context`, `what_can_i_do`,
   `what_is_current_context`). The global context is asked first when you stand on a record and it
   has a trained global head: a confident global command wins. Then the current context. The head
   answers one command only when that command is the best of the whole workflow, inside the mask, at
   or above the commit threshold (0.65 by default). It never sends a request up on its own: when a command
   outside the mask is ahead it is not confident, and it hands the request on with its own top three
   plus up to two of those outside commands that are on the lookup path.
5. **The remote model** (when `JEV_API_KEY` is set) decides the handed-on request over the whole path.
   When it picks a command of another scope, the walk climbs to the next enclosing record, whose head
   is asked in turn.
6. **The question.** Otherwise the candidates go to the user (or the agent) as a which-command
   question. The command picked there runs where it lives on the path, on that scope's record.

The head's `wildcard` class is the framework's built-in vague replies ("ask me again later",
"well, you see, um"), the same for every workflow. It is the only "not here" answer the local model
gives.

Consequences that drive everything below:

- A `get_parent` that returns the wrong object, or `None` where an enclosing record exists, takes
  that record's commands off the path: they are neither handed on nor run there. Check it first
  when an enclosing command never runs. When the workflow has a root object, a record that is not the
  root whose context has no callback file or no `get_parent` raises `MissingGetParentError` on the first
  climb (and a command that moves onto it fails); `return None` is the explicit, silent "go to the
  global commands". With no root, all three mean the global commands. A callback file `_<Context>.py`
  that defines no class named exactly `Context` is a load problem (`load_problems`), not a silent skip.
- From inside a context, a command of an enclosing record is never answered by the local model
  alone. It comes back as a handed-on candidate, settled by the remote model or a question.
- Every command of the workflow is a label of the one head, so any two similar commands compete,
  whatever contexts they sit in.

### In the `per_context` layout

An application can still train with `INTENT_MODEL_LAYOUT=per_context` (see `train-and-publish-models`).
Each context then has its own head, whose labels are its mask and whose `wildcard` ("not here")
class holds one hand-written seed of every command that is not in the mask. A lone confident
`wildcard` climbs to the next enclosing record with no model call; `wildcard` beside real candidates
is an ambiguity. The competition Rules 6 to 8 describe then runs through that class: a foreign
command's seed competes with the context's own commands as `wildcard`, and the class grows with the
number of commands in the workflow, not with the depth of the tree.

## Planning commands follow the same visibility

A file `_commands/<Context>/<name>.py` is the planning command `<Context>/<name>`. It is a label of
its context's mask and of every context that inherits that context, and it is found on the same
lookup path as an execution command. A root planning command (directly under `_commands/`) belongs
to the global context. So a planning command in a base context reaches every context that inherits
the base, and one in a child context is not offered from a sibling. A confident local answer that
names a planning command is still confirmed by the remote model or asked about before it runs.
Moving or renaming a planning command changes its qualified name, exactly as for a command, so its
`from_command_output` edges and previews move with it.

## Rule 1: keep `get_parent` honest, and `parents` beside it

The parent of a record is what its context's `get_parent` returns for the live record; `parents`
lists the contexts that can be, and the climb fails when `get_parent` returns a record of any other.
Check each `get_parent` against the record it should return: a record's parent is the record that
holds it, and the root writes none (the framework never asks the root). In the `library` example
`Book` declares `parents = ["Library"]` and its `get_parent` returns `record.library`, and `Library`
has neither.

## Rule 1a: navigation is declared once, with `descends_to`, on the command

A command that moves the workflow into another context (by assigning `workflow.record`) declares it
on its own class: `@fastworkflow.descends_to("<Context>")` on its `ResponseGenerator` or
`PlanGenerator`. It is an unconditional move: there is no `when_parameter` (it is removed and fails
the load); a command that moves only sometimes still assigns `workflow.record` only when it moves.
`enter_command` or `enter_commands` on a context class is a load error that names the fix, and so
is a target whose `parents` name neither the command's context nor a context it inherits. A command may move the
session only onto a context it declares (or one that inherits from it, `inherits`); a move onto
any other context, another record of its own context included, fails the command and the record is put back. Planning, the
run's listing and the "how to get there" reply read only these declarations.

Two other command declarations sit beside it and are not about navigation: `@user_approval_for_execution`
(a gate before a run; `docs/planning_command_file_layout.md`, "Approval annotation") and `@writes`
(the `author-planning-commands` skill, "Commands that write").

## Naming a record: `get_displayname` and `instance_label_attr`

A context class declares two things about the record it manages, and they answer different questions. Keep
them apart:

| | `get_displayname(cls, record)` | `instance_label_attr = "<attribute or property>"` |
|---|---|---|
| What it is | a classmethod that returns any free-form display string | a class attribute that names one attribute (or property) of the record |
| Example value | `"Book: Dune (b12)"` | `"uid"`, whose value on the record is `b12` |
| What it answers | "how should this record look to a person?" | "what is this record's identity?" |
| Who reads it | `workflow.record_displayname` (for the record the command acts on), the CLI and chat context line, observability | `context_identity` (listing tags, `search_memory`, the extract step, the approval binding) |
| When absent | optional; without it the context name is shown | nothing is guessed; the record has no identity in those readers |
| Derived from the other? | no: it may use the identity, but `instance_label_attr` is never read from it | no: it names an attribute, it does not compute one |

A computed identity is a property on the record, which `instance_label_attr` names. A context class
that still defines the removed `instance_label` classmethod fails the load.

The hooks that save and restore a session (`get_state`, `from_state`, `state_version`, `get_locator`,
`find_by_locator`) live on the same class. When several context classes need the same ones, write them once on
a plain base class in a helper module of the workflow's helper package `_commands/_shared/`, which the framework
never reads (for example `_commands/_shared/checkpoint.py` with `class CheckpointHooks:`), and write
`from .._shared.checkpoint import CheckpointHooks` and `class Context(CheckpointHooks):` in each callback file. The framework reads every member
of the class with `getattr`, so these, `get_parent`, `get_displayname`, `instance_label_attr` and `occupiable`
may all be inherited, and a class that defines one of its own overrides it. `inherits` and `parents` may not: they are
read as written in the body of `class Context` itself.

## Rule 2: cutting an inherited command an enclosing record also offers keeps it reachable

- **Reachability comes from `inherits` and from `get_parent`, never from a second file.** A command
  removed from a context's surface is still reachable *in place*, with no navigation, if it lives
  on the lookup path (an enclosing record's context or its bases, or the global context). It becomes
  unreachable without navigation if it only lives in a **sibling**.
- **In place is not free.** From below, the local model no longer answers it alone: it is handed on,
  so it costs a remote-model call or a which-command question. (In the `per_context` layout the
  "not here" class climbs to it with no model call.)

| Context | Parent at runtime | Cutting an inherited command |
|---|---|---|
| Entity nested under a workspace that offers the command | that workspace | reachable in place, through the remote model or a question |
| Top-level workspace, or a command only a sibling offers | the global root | costs a round trip out and back |

Before cutting anything from a `base`, confirm which case applies.

## Rule 3: size each base context to its consumers' declared producers

Inheritance is all-or-nothing per base context, so a context that needs one command from a base must
take all of them. When a fat base is inherited by several contexts, split it.

The evidence for what a context needs is **not intuition**: it is the parameter-producer hints its
own commands declare (`from_command_output` edges, which `command preview --check` resolves). Collect, per context, the producers named by its commands;
that is the minimum base it needs.

Then chain the bases so nothing regresses:

```
FooLookup   (the narrow subset)  <- inherited by the workspaces that need only that
BarCatalog  (another subset)     <- inherited by the workspace that needs only that
Wide        inherits = ["FooLookup", "BarCatalog"] + its own remainder
                                 <- inherited by the one context that needs everything
```

The wide base still resolves to the full set, so its consumer is untouched while the narrow
consumers shrink.

**A command's qualified name is its folder.** Moving a command between base contexts renames it, so
every producer hint, doc, test and routing artifact moves with it, and a retrain is mandatory.

## Rule 4: delete a command that is the same query on another surface

Two commands that issue the same backend query with the same parameters, differing only in which
context they hang from, are one command. Delete the redundant one and accept the longer path: the
extra navigation turn is paid once per task, while the confusion between the two labels is paid on
every request near them.

Keep both only when the surviving one cannot serve the case, for example when a listing must prove
parent-to-child membership by paging a parent-scoped view, which a global search by id cannot do.

## Rule 5: know which token is supposed to discriminate

Near-duplicate labels are the dominant source of routing failure. For each colliding pair, decide
which token class carries the decision, then make **every seed on both sides** carry it.

| Pair shape | Discriminator | Example (`library`) |
|---|---|---|
| Same question asked of different things | the **entity noun** | `Book/list_loans` vs `Member/list_member_loans`: "loans" cannot divide them, "book" and "member" can |
| Different action on the same thing | the **verb** | `Library/open_book` vs `Book/show_book` |

If the seeds on one side name no entity at all ("list what it has", "their items please"), that side
cannot be separated no matter how the other side is worded.

## Rule 6: anchor the ancestor side of a parent-child pair

A parent-child pair asks the same thing at two levels: `Library/book_loans` ("who has borrowed
Dune") and `Book/list_loans` ("list this book's loans"). Inside the child, a request near either one
puts both near the top of the head; when the parent's is ahead, the child's own command is handed on
instead of answered.

- Anchor the **ancestor** side to its entity (the title, the noun) in every seed.
- Let the **descendant** keep bare pronouns ("its loans please", "who has it"). Inside the child
  context a pronoun is unambiguous, and this preserves coverage of terse input.

Anchoring both sides is unnecessary and throws away the low-content register for no gain.

## Rule 7: keep navigation seeds strictly navigational

A command that enters a context (`Library/open_book`) competes with that context's own commands in
every request about the record. A navigation verb whose seeds carry the vocabulary of the context it
opens takes their requests: asked from above, it is answered instead of the command the user meant;
asked from inside, the context's own commands are handed on. A global navigation verb does this
everywhere, and the global context is asked first.

Keep navigation seeds **strictly navigational**: where to go, not what to do once there. "Open book
b1" is navigation; "who has borrowed b1" is `Book/list_loans` asked verbatim. A command that only
enters a context (`@descends_to` with no required input) is never offered to
the remote model as a request's answer, so only its seeds and its name reach it.

## Rule 8: shrinking a surface hands on more requests

Every command cut from a context's mask is one more command outside it. A request near one of them
is then handed on instead of answered (a remote-model call or a question). That is cheap when the
context's commands are lexically distant from the commands left outside, and costly when they are
close, so after narrowing a base, check the narrowed context's own commands are still answered
before assuming the change was free.

## Procedure

```
- [ ] 1. Check each context's get_parent against the live record it should return, and its parents
- [ ] 2. Check which contexts each command enters (`workflow.record` assignments) and that each declares @descends_to
- [ ] 3. Per context, list surface vs. the producers its own commands declare
- [ ] 4. For each unused inherited command, check Rule 2 depth before cutting
- [ ] 5. Split fat base contexts; chain the wide base onto the narrow ones
- [ ] 6. Read the near-duplicate report and the confused pairs; classify each pair by Rules 5-7
- [ ] 7. Rewrite seeds per Rules 5-7
- [ ] 8. Retrain
- [ ] 9. Verify by case-level diff, not by percentages
```

Steps 1-5 are mechanical and safe. Steps 6-7 are where the accuracy is won: in practice a single
colliding pair can cost a context more than its entire label count does.

## Verification discipline

Retraining is mandatory after any of this: the labels and each context's mask are fixed in the
trained artifact. `fastworkflow train` sees the change. In the `embedding` layout any change to any
command refits the one head, in seconds; an artifact in an older format refuses to load and says to
retrain.

Then read the results carefully. Four traps, all of which will otherwise produce a false conclusion:

1. **A re-drawn holdout is not a comparison.** If the evaluation redraws its fold each run, a context
   is never scored on the same phrasings twice: one run may score three of its labels and the next
   six. Per-context deltas are directional only.
2. **A shrinking label space shrinks the denominator.** Totals across runs are not like-for-like when
   the tree changed.
3. **A retrain moves commands nobody touched.** The `embedding` head is deterministic (the same rows
   give the same head), but every change refits it over every command, and regenerated utterances
   change the rows. Fine-tuned layouts add training variance on top. Expect commands the change does
   not explain, framework commands among them, to move.
4. **Only a fixed, hand-written case set compares across runs**, and only if the case set itself was
   not edited. When labels are renamed, the case set changes too, so treat it as an absolute reading.

The reliable signal is the **case-level diff**: list the failing cases before and after, and keep
only the ones the change can actually explain. A change touching one command's seeds cannot sensibly
explain an unrelated framework command moving; that movement is churn. Report the causal set as a
trade, "N cases gained here against M lost there", because a well-targeted seed fix usually does
cost something at the boundary it moved.

## Additional resources

- Diagnostic recipes, a failure-mode catalogue mapping symptom to cause to fix, and the exact
  computations for label space and surface: see [reference.md](reference.md)

## Related

- Step 6's near-duplicate report and the training report's confused pairs: what they measure, the
  reporting bands, and the accept-list for a pair you have decided is deliberate: the
  `detect-duplicate-capabilities` skill.
- Step 8's retrain: what a rename invalidates, which changes retrain what, the layouts, and how to
  roll back: the `train-and-publish-models` skill.
- Step 9's verification: which reported number is comparable across runs and which is not: the
  `evaluate-intent-routing` skill.
