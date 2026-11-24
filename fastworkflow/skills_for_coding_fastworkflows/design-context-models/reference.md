# Reference: diagnostics and failure modes

Companion to [SKILL.md](SKILL.md). Read when a specific number or a specific symptom is needed.

## Computing the numbers

### Surface of a context

Own commands (the `.py` files in `_commands/<Context>/`, excluding leading-underscore files) plus
the transitive closure over `base`, an own command hiding an inherited one with the same leaf name.
Files directly under `_commands/` are the global `*` context.

```python
def surface(ctx):
    out = set(own.get(ctx, []))
    for b in inheritance.get(ctx, {}).get("base", []):
        out |= surface(b)
    return out
```

Cross-check against the framework's own loader rather than trusting the hand computation. It
validates the JSON at the same time:

```python
import fastworkflow
fastworkflow.init(env_vars={})
crd = fastworkflow.RoutingRegistry.get_definition(workflow_path)
crd.context_model.commands(ctx)                # surface: own and inherited execution commands
crd.context_model.inherited_base_contexts(ctx) # its bases, transitively
crd.get_command_names(ctx)                     # the surface plus the four core commands
crd.get_planning_command_names(ctx)            # its planning commands, own and inherited
```

The enclosing contexts are not in any file: they are whatever each `get_parent` returns for the
live record.

### Label space and mask

In the default `embedding` layout the one head separates every label of the workflow at once:
every command, `parameter_value` and `wildcard`. A context's mask is what it may answer:

```
len(surface(ctx))
+ its planning commands (own and inherited)
+ 4 core commands
```

that is, `crd.get_command_names(ctx)` plus `crd.get_planning_command_names(ctx)`. Enclosing and
global commands are outside the mask: a request near one of them is handed on, not answered. The
training report's cross-fit confused pairs are read over the whole workflow.

In the `per_context` layout each context's head has exactly these labels plus `wildcard`, its "not
here" class with one seed per command outside the mask. There, track the sum of label counts over
enterable contexts as the headline size metric (base-tier contexts are not enterable: nothing
navigates into them).

### Deriving the containment graph

Scan each command module for the object assigned to the current context:

```python
pat_ctx = re.compile(r"workflow\.record\s*=\s*(\w+)")
pat_new = re.compile(r"(\w+)\s*=\s*(\w+)\(")   # resolve the local back to its class
```

The result maps `Context/command -> entered class`. Each such command must declare that class
with `@descends_to`; a move it does not declare fails the command at run time. There is no hierarchy
file to diff against: parents are whatever each `get_parent` returns for the live record.

### Which producers a context needs

Collect the producer edges declared by each context's own commands: the `"from_command_output": "<command>.<field>"`
entries of `json_schema_extra` on their `Input` fields (`declare-parameter-producers`). When a
command loads they are resolved into `json_schema_extra["available_from"]`, which is what a script
reads from a loaded command; a source scan looks for the `from_command_output` entries:

```python
re.finditer(r"from_command_output\"\s*:\s*(\[[^\]]*\]|['\"][^'\"]*['\"])", src)
```

Group by owning context. That is the minimum set of producers the context needs on its surface, and
therefore the minimum base it can inherit.

## Failure-mode catalogue

### Own commands are handed on instead of answered

**Symptom.** Requests for a context's own commands come back as a which-command question (or cost a
remote-model call) although the right command is among the candidates. In the `per_context` layout
the same cause shows as own commands coming back as a lone `wildcard`.

**Cause.** A command outside the mask is lexically close and ranks ahead, usually after a surface
was narrowed (Rule 8). The commonest offender is a navigation verb, global or in an enclosing
context, whose seeds describe what the user wants done rather than where they want to be.

**Fix.** Make the navigation seeds strictly navigational (Rule 7). Verify by checking that the
context's own commands recover, not just that the totals moved.

### A command absorbs phrasings belonging to a parent

**Symptom.** A descendant command fires on utterances that name the parent's entity.

**Cause.** The parent's seeds are unanchored, so the descendant's own seeds are the closer match for
anything vague.

**Fix.** Anchor the ancestor side to its entity noun (Rule 6). Leave the descendant's pronouns alone.

### Two same-shaped commands in one context

**Symptom.** A near-duplicate report flags a pair with `shared_contexts` naming a real context, and
that context's top-1 is well below its in-list.

**Cause.** Both labels are on one surface and differ only by a noun the seeds do not emphasise.

**Fix.** Either delete one (Rule 4) or make the differing noun the only thing the seeds vary. A
family of `open_<entity>_by_uid` commands is the classic case: they share `open`, `by`, `uid` and
differ in one word.

**Do not** reflexively merge such a family into one command with an enum parameter. That trades an
intent-detection problem for a parameter slot-filling problem, and slot filling may have no
clarification path equivalent to the one intent detection has. Check whether the framework can
clarify an enum parameter value before proposing the merge.

### A context routes well but never escalates

**Symptom.** High routing accuracy, but requests for an enclosing record's commands never run
there: the commands are not among the candidates, or run on the wrong record.

**Cause.** The enclosing record is not on the live chain: `get_parent` returns `None` or an object
other than the record that holds the current one. That record's commands are off the lookup path.

**Fix.** Check `get_parent` against the live object (Rule 1). No seed work reaches a scope the walk
does not visit.

### Accuracy drops after an unrelated change

**Symptom.** A context nobody touched moves several points; a framework command such as `go_up`
swings from near-perfect to near-zero.

**Cause.** The retrain itself: the one head is refitted over every command on any change, and a
fine-tuned layout adds training variance.

**Fix.** Nothing. Confirm by checking whether the moved cases are reachable from the change at all,
and report them as churn.

## Near-duplicate separability

A pairwise separability score below the report's overlap threshold means a classifier restricted to
just those two commands barely beats chance: the pair is unresolvable by any amount of context.
Treat that report as the primary design signal, ahead of accuracy: accuracy tells you a context is
struggling, separability tells you which two labels to fix.

Read three fields together:

- `separability`: how badly they collide.
- `shared_contexts`: whether both are in one context's mask. Empty does **not** mean safe: in the
  `embedding` layout every command is a label of the one head, so a command outside the mask still
  competes (it makes the context's own command be handed on); in `per_context`, through its seed in
  the "not here" class. The training report's cross-fit confused pairs show the head's own view.
- `terms_only_in_a` / `terms_only_in_b` / `shared_terms`: the shared terms are what the seeds must
  stop relying on; the exclusive terms are what they must lean on.

## What to update when a command moves between contexts

A command's qualified name is `<Folder>/<command>`, so moving it is a rename. Update:

- the module's location on disk, and the scaffold or generator that emits it
- every `from_command_output` producer edge naming it (`"from_command_output": "<Context>/<command>..."`)
- every import of its module by path (a planning command's `from ..Book.list_loans import Signature`)
- the inheritance model, and any `inherits:` prose in generated docs
- benchmark and end-to-end case tables
- navigation docs and command maps

Then retrain. Cached generated utterances are fingerprinted on the seed list *and* the command
name, so a rename invalidates exactly the moved commands and reuses the rest. Prefer a plain
retrain over forcing full regeneration, which is slower and discards reproducibility for
untouched commands. The full list of fingerprint inputs, and what forces a full retrain anyway, is
in the `train-and-publish-models` skill.

## Reporting template

```markdown
## What changed
[the edits, grouped by rule]

## Measured effect
[totals, with the comparability caveat stated]

## Case-level diff
[cases fixed vs cases broken, restricted to what the change can explain]

## Left on the table
[flagged pairs not addressed, and why]
```
