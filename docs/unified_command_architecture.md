# Unifying skills and commands: one idiom at every level of granularity

Status: approved for implementation
Audience: the coding agent implementing it, and fastWorkflow maintainers
Scope: fastworkflow/ runtime, build, train, preview, docs, and the bundled coding skills

Line numbers below were correct when this was written. Locate code by the
function or constant name; the line number is only a hint.

---

## 1. Justification

### 1.1 The pattern that already exists

A user turn is handled the same way at two levels of granularity.

| Step | Skill level (coarse) | Command level (fine) |
|---|---|---|
| 1. Refine | `_run_agent` refines the request with background context | The command metadata extraction (CME) workflow adds the command context and conversation history |
| 2. Split | `split_into_subtasks`, `split_merge.merge_subtasks`, `resolve_referents` | The planner (`build_query_with_next_steps`) writes a plan of command lines |
| 3. Decompose | A skill's phases (`skill_phases.Phase`), each a bounded sub-plan | ReAct agent steps, one per plan line |
| 4. Execute the atomic unit | An executor agent run per phase (`fastWorkflowReAct`) | One formatted command, run by `CommandExecutor._invoke_command_impl` |

The two levels were built separately, so the same ideas are expressed with
different idioms, and some exist at only one level. Each such asymmetry is
something the app developer has to learn twice, and the framework
developer has to maintain twice.

### 1.2 Problems this fixes

The app developer today runs into these problems:

1. **Skills have no contexts.** `_skill_files` (`workflow_agent.py:1119`)
   skips every subfolder of `_skills` with the warning "skill contexts
   (subfolders of _skills) are not loaded yet". Commands are organised by
   `_commands/<Context>/` and `context_inheritance_model.json`; skills are a
   flat global list. A workflow with many skills cannot scope them, and
   every skill competes in every match.
2. **No ask_user while planning a skill.**
   - When `match_skill` is unsure (`SKILL_MATCH_THRESHOLD = 0.5`,
     `workflow_agent.py:1427`), the request silently falls back to the text
     planner.
   - When `plan_skill_call` (`skill_commands.py:2077`) cannot fill a
     required argument after its single retry, it also falls back.
   - At the command level, a missing parameter is asked for. At the skill
     level, the user is never asked.
3. **Three ways to say "this value relates to another command's field".**
   - `available_from` (a command input names its producer command).
   - `fact_from` (a phase facts field copies `<command>.<field>`).
   - A third idiom (`input_from`) was about to be added for skill inputs
     that forward to a callee's input.

   All three are string references stored in `json_schema_extra`. None is
   checked for field-level type compatibility, and all three break silently
   when a command is renamed or moved (Rule 4 of the
   `declare-parameter-producers` coding skill).
4. **`available_from` points at a command, not a field.** It is read by:
   - the planner prompt (`workflow_agent.py:1779`);
   - the ReAct execution agent, through command metadata
     (`command_metadata_api.py:140-155` and `:990-991`);
   - the missing-parameter message (`utils/signatures.py:1260-1267`).

   None of them is told *which output field* supplies the value. That is why
   the coding skill needs prose Rules 2 and 3 ("the producer must emit the
   handle the consumer accepts", "describe the handle at both ends").
5. **Skill arguments are narrower than command arguments.**
   `_spec_from_field` (`skill_commands.py:415`) accepts only str, bool, int,
   Literal and Optional[str]. Authors restate a leaf command's description
   and examples by hand on the skill input, and the two copies drift apart.
6. **A skill cannot call a skill.** Composition today is Python
   inheritance only (`class PlanGenerator(BookByTitle)`,
   `docs/tutorials/skills/11-composing-skills.md`). Calling a skill as a step
   is impossible, because `WorkflowExecutionContext` keeps single-instance
   state:
   - `_subtasks`, `_origins`, `_subtask_index`, `_subtask_answers` and
     `_subtask_exhausted`;
   - `_next_step` and `_evidence_floors`;
   - `_phase_run`, `_phase_live` and `_phased_skill_call`;
   - `_initial_plan_text`;
   - `_workflow_tool_agent` (`workflow_execution_context.py:189-212`).

   A nested call would overwrite all of them. Skills are also not commands:
   they are loaded by `_skill_files` / `load_skill_commands`
   (`workflow_agent.py`), never registered in `CommandDirectory`, so no
   step text can name one and `_explicit_agent_command`
   (`workflow_agent.py:295`) cannot dispatch one.
7. **Two folders for one concept.** `_skills/` and `_commands/` hold the same
   kind of thing, a named capability with a `Signature`, at different
   granularities. Each has its own loader, naming rules and preview.

### 1.3 What success looks like

The app developer learns five things, and uses them identically at every
level:

1. One folder tree: `_commands/<Context>/<name>.py`.
2. One `Signature` with an `Input`, an `Output` and a docstring.
3. A switch made by the class name: `ResponseGenerator` runs code;
   `PlanGenerator` returns steps.
4. One cross-reference idiom: `from_("<Context>/<command>.<field>")`.
5. `ask_user` behaves the same everywhere.

The framework developer maintains four mechanisms:

1. One registry with a two-step route: composites first, then leaves.
2. An `ExecutionFrame` stack with one per-turn step budget shared by all
   nested frames, and a depth cap.
3. One output contract: every frame returns
   `CommandResponse(response, output)`.
4. A callee graph, used only for cycle detection and the preview.

### 1.4 Non-goals

- **Not 100% identical idioms.** The class name stays an explicit switch
  (see 2.2). The target is about 95% shared, with no inferred modes.
- **No layer folders, no layer numbers, no policy based on layers** (see
  2.4).
- **Composites are not trained into the BERT intent classifiers** (see 2.5).
- **`_spec_from_field` is not widened** to list or float types. Forwarding
  (3.3) is allowed only onto target fields whose type the skill machinery
  already supports. Widening is a separate later change.
- **No renaming of public APIs other than those listed here.** `fact_from`
  and the `available_from` key stay valid indefinitely.

---

## 2. Discussion

This section records the arguments behind each decision, so the
implementer does not have to reconstruct them.

### 2.1 Could one recursive pipeline run every level?

Partly. Steps 1 and 2 (refine, split) are the same operation at both
levels. Steps 3 and 4 differ in kind:

- A leaf is Python business logic with no LLM in the loop.
- A composite is a plan an LLM agent executes.

The useful recursion is therefore not "one function that calls itself".
It is **"a composite's steps may name any command, including another
composite"**. That is supported by the frame stack in 3.5. Refining and
splitting stay at the root turn; nested frames receive already-planned
step text.

### 2.2 Why two class names instead of one

`__call__` means two incompatible things today:

- **`ResponseGenerator.__call__(workflow, command, command_parameters)`** is
  the entry point the executor calls (`command_executor.py:306`).
- **`PlanGenerator.__call__(self, input)`** returns the order of the phases
  (`fastworkflow/skill.py`).

An earlier draft proposed telling a leaf from a composite by the presence
of `_process_command`. That is wrong, because `_process_command` is only a
naming convention. Inferring the mode from method names hides behaviour.
Two class names act as a one-word, explicit switch, and everything else is
shared. Readability wins over nominal uniformity.

### 2.3 Why one `from_`, and why its meaning follows the target

`available_from`, `fact_from` and the proposed `input_from` all declare an
edge from this field to a field of another command. They differ in which
kind of field they point at:

| Declared on | Target field kind | Meaning | Today's spelling |
|---|---|---|---|
| A command `Input` field (leaf or composite) | An `Output` field | Producer: call that command first, and its output supplies this value | `available_from` (names the command only) |
| A phase facts model field | An `Output` field, or an `Input` field when no `Output` field has that name (today's `fact_sources` rule) | Runtime copy into the phase's facts | `fact_from` |
| A composite `Input` field | An `Input` field | Forwarding: same parameter as the callee's, so borrow its description, examples and pattern | (proposed `input_from`) |

Because the target's kind is in the reference itself, a single function
with three behaviours stays readable: the reader sees which field is
targeted. Field-level references also turn coding-skill Rules 2–4 into
checks the tooling runs:

- the target field exists;
- the types are compatible;
- the reference still resolves after a rename.

**Forwarding edges must never be shown as `available_from`.** All three
consumers (the planner, the ReAct agent and the missing-parameter message)
read `available_from` as "call X first". For a forwarding edge that is
backwards, because the composite calls X itself. Showing it would make the
agent call `find_book` before calling a composite that already calls
`find_book`.

**A producer edge does not inherit validation or lookup behaviour.** It
means "go call X first", not "resolve this value the way X does".

### 2.4 Why no layers

A layer number is tautological: it is just the longest path in the callee
graph. It also cannot be computed exactly, because an open-ended prose
step such as `f"say who wrote {title}"` names no command. Callees read
from placeholder templates are therefore a lower bound.

Using layers as policy breaks things. For example, "a layer-N composite
sees only layers below N" would hide navigation commands and break
replans, which can legitimately reach any command. The callee graph is
kept for two jobs only:

- Cycle detection, which is a load error.
- Display in `fastworkflow skill preview` (depth shown for information).

### 2.5 Why composites are not trained into the classifiers

"Book *Dune*" (composite) and "find the book *Dune*" (leaf) are
near-duplicates at different granularities. Training them into one
TinyBERT/DistilBERT head creates exactly the confusion that the
duplicate-capability check in `fastworkflow train` exists to prevent.

Composite matching already works: `match_skill` uses a typed Jev choice
with per-option probabilities and a measured threshold (at 0.5, 96.6% of
sub-tasks get a template and 1.1% get a wrong one). So routing is two-step:

1. Match composites first, with `match_skill`.
2. Whatever does not match goes to the leaf path unchanged.

Merging the heads may be revisited later, but only with held-out F1
evidence. That is outside this project.

### 2.6 Why the output contract matters

A parent phase reads typed facts from a child. `CommandResponse.output`
already exists for this purpose (`fastworkflow/__init__.py:41-48`). It is
optional, and many commands leave it `None`; for example,
`examples/extended_workflow_example/_commands/WorkItem/get_status.py`
builds its Output and then returns only text.

A parent can only read what a child emits. So:

- A composite that declares a `Signature.Output` always sets `output`
  when the phase that fills it ran (3.4).
- A field named by a producer `from_` must exist in the target's
  `Signature.Output`, and a field named by a copy `from_` must exist in its
  `Output` or `Input` (3.3). This is checked statically.
- At runtime, a command named as a producer or copy target that returns
  `output=None` logs a warning naming the command. Copy edges onto an
  `Input` field are read from the archived run's parameters and need no
  `output`, exactly as `fact_from` does today.

### 2.7 Runtime cost, memory and scale

- **LLM calls.** Composite matching is one Jev call per sub-task, the same
  as today, and a turn-frame composite keeps its one typed planning call
  (`plan_skill_call`, plus at most one retry). A nested composite adds no
  planning call: its arguments are extracted from the parent's step by the
  same CME parameter extraction a leaf step uses (3.5), which is cheaper
  than a ReAct replan.
- **Step multiplication.** The per-phase cap is `FW_SKILL_PHASE_MAX_STEPS`
  (default 15), and a flat or non-skill executor run is capped by the
  agent's `max_iters` (default 25, `initialize_workflow_tool_agent`,
  `workflow_agent.py:826`). Nesting multiplies the worst case by roughly
  that factor per level. A single turn-wide step counter shared by all
  nested frames, plus a depth cap, bounds what nesting adds (3.5). The
  turn frame keeps today's caps.
- **Memory.** A popped child frame keeps only its `CommandOutput`, and its
  trajectory and agent are released. Live memory is proportional to the
  depth of the current branch, not to the size of the whole call tree.
  Full detail is still on disk in the observability archive.
- **Scale.** Contexts for composites mean the catalog any one match sees is
  bounded by the context surface, which helps both matching accuracy and
  prompt size.
- **Prompt size of richer hints.** Field-level `available_from` text is
  rendered compactly (`Ctx/cmd.field`), never as a sentence.

### 2.8 Ask_user while planning: what is new

Today `AskUserSuspend` / `CommandCancelledError` fire inside command
execution, with an agent trajectory to resume. A question asked during
matching or argument planning happens before any executor run exists. So
it uses the turn-level clarification state that is already serialised:

- `_awaiting_user`
- `_pending_clarification_request`

A new plan-question record is added on the turn frame (3.6). No agent is
suspended for such a question, so the resume path checks for it before
it looks for a suspended agent.

**Unattended turns never ask.** `_turn_unattended`
(`workflow_execution_context.py:264`, set in `_begin_turn`) is already
honoured for ask_user (`_enable_all_tools`, `:1846-1853`). Plan-time
questions honour it too and
keep today's fallback behaviour.

### 2.9 Build safety

`generate_command_files(..., overwrite=False)`
(`build/command_file_generator.py:13`) writes class methods into
`_commands/<Class>/` and module-level functions into the `_commands/` root.
Once hand-written composites live in `_commands`, an `overwrite=True` run
could destroy them in either place. Composite files must be protected
unconditionally.

### 2.10 Rejected alternatives

| Alternative | Why rejected |
|---|---|
| `layer_0/`, `layer_1/` … subfolders | Tautological, inexact, and invites policy that breaks navigation (2.4) |
| Infer leaf or composite from method names | Hidden mode; `_process_command` is only a convention (2.2) |
| Extending leaf-level `db_lookup` (a leaf `Input` field's `json_schema_extra` key, read by CME parameter extraction) to composite inputs | Duplicates the lookup the leaf already does when its step runs; two places to drift |
| A separate `input_from` | Third spelling of the same edge (2.3) |
| One classifier head for composites and leaves | Confuses granularities (2.5) |
| Per-frame step budgets only | Worst case grows roughly 15–25× per level of depth (2.7) |
| One budget over every frame, the turn frame included | Today's turn frame alone can take about 750 model iterations (3.5); a cap low enough to bound nesting would cut off turns that work now |

---

## 3. Proposed architecture and design

### 3.1 File layout (final state)

```
<workflow>/
  _commands/
    _split.py                  # split settings (moved from _skills/_split.py)
    _helpers.py                # any "_"-prefixed file: an importable helper, never a command
    send_greeting.py           # global leaf     (defines ResponseGenerator)
    book_by_title.py           # global composite (defines PlanGenerator)
    Library/
      find_book.py             # leaf in context Library
      reserve_and_notify.py    # composite in context Library
  context_inheritance_model.json
```

The rules:

- **One command per file.** A file defines exactly one of `ResponseGenerator`
  or `PlanGenerator`:
  - both is a load error naming the file;
  - neither is today's behaviour for a broken command file.
- **The name is the file stem**, and the qualified name is
  `<Context>/<stem>` (just `<stem>` at the root), exactly as for leaves
  today.
- **`_`-prefixed files and folders are never commands.** This is already
  true in `command_directory.py:263, 269, 276`.
- **A composite and a leaf with the same qualified name is a load error.**
  This applies to both layouts during the transition (Phase 7).
- **`_skills/` stays readable until Phase 7 ends.** Its root files load as
  global composites, and its subfolders as context composites (Phase 1).
  After Phase 7, loading `_skills/` logs one deprecation warning per
  workflow. It is not removed in this project.

### 3.2 Writing a command at each granularity

Leaf (unchanged):

```python
class Signature:
    """Find books by title or author."""
    class Input(BaseModel):
        query: str = Field(description="Title or author words", examples=["Dune"])
    class Output(BaseModel):
        book_uids: list[str] = Field(description="uid of each match. Pass one to a command that asks for a book_uid.")

class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters) -> CommandOutput:
        lib = workflow.command_context_for_response_generation
        out = Signature.Output(book_uids=lib.find_book(command_parameters.query))
        return CommandOutput(command_response=CommandResponse(
            response=f"Found {len(out.book_uids)} books", output=out))
```

`CommandOutput` takes one `command_response=` (`fastworkflow/__init__.py`);
passing `command_responses=` raises `ValueError`. This is the idiom of
`tests/skill_tutorial_workflow/_commands/find_book.py`.

Flat composite (one phase method that reads only `input` and has no return
annotation, so it runs as one plan, `skill.py:_runs_flat`):

```python
class Signature:
    """Say who wrote a book, given its title."""
    class Input(BaseModel):
        title: str = from_("Library/find_book.query")   # forwarding: borrows description, examples, pattern

class PlanGenerator:
    def book(self, input):
        return [f"find_book query={input.title}",
                f"say who wrote {input.title}"]
```

A flat composite fills no facts, so it cannot declare a `Signature.Output`
(a load error, 3.4) and its `CommandResponse.output` is `None`.

Phased composite (today's skill idiom, with `from_` in place of
`fact_from` and a declared `Output`):

```python
class Signature:
    """Show the author of the book whose title the request names."""
    class Input(BaseModel):
        title: str = from_("Library/find_book.query")
    class Output(BaseModel):
        book_uid: str = from_("Library/show_book.book_uid")   # copy, as on any facts model
        author: str = from_("Library/show_book.author")

class Found(BaseModel):
    book_uids: list[str] = from_("Library/find_book.book_uids")   # copy: today's fact_from

class PlanGenerator:
    def find(self, input) -> Found:
        return [f"find_book query={input.title}"]
    def book(self, input, find) -> Signature.Output:
        return [] if not find.book_uids else [
            pick("show_book", book_uid=None, named=input.title)]
    def answer(self, input, book=None):
        return [f"say who wrote {input.title}" if book else
                f"say that no book matches {input.title}"]
```

The composite's `Output` is the facts model of the one phase whose return
annotation is `Signature.Output` (here `book`); its fields are copy edges
like any facts field.

Producer hint on any command input (replaces a bare `available_from`):

```python
order_id: str = from_("CustomerDesk/find_order.order_ids", description="The order to fetch")
```

### 3.3 `from_`: specification

**Location and exports.** Define `from_` in `fastworkflow/skill_phases.py`,
next to `fact_from`, and export it from `fastworkflow` and from
`fastworkflow.skill`.

```python
def from_(source: str | Sequence[str], *, kind: Literal["auto", "input", "output"] = "auto",
          **field_kwargs) -> Any
```

**What it returns.** `Field(**field_kwargs)`, whose `json_schema_extra` is
the caller's `json_schema_extra` (if any) with
`{"from": source, "from_kind": kind}` merged in.

**Allowed keys on skill inputs.** `_spec_from_field` rejects any
`json_schema_extra` key outside `SKILL_FIELD_KEYS`
(`skill_commands.py:412`, today `strip`, `subject`, `verify`, `means`,
`in_request`). Add `"from"`, `"from_kind"` and `"available_from"` to it, so
composite `Input` fields can carry producer and forwarding edges.

**`fact_from(source, **kw)` becomes an alias** for `from_(source,
kind="output", **kw)`. Its docstring says so. Existing files keep working
byte-for-byte.

**Reference grammar.** `[<Context>/]<command>[.<field>]`, or a list of such
strings. A list is allowed only for producer edges (several commands can
supply one input). Copy and forwarding edges take exactly one string,
because `facts.fact_sources` parses `"from"` as one `"<command>.<field>"`
and raises `FactSourceError` for anything else.

- **Qualification is optional.** An unqualified `<command>` is resolved
  against the *declaring command's* surface. That surface is the
  declaring context's own commands, then inherited commands, then live
  ancestors: the same resolution the router uses for that context.
  - Global commands resolve from any context.
  - Matching two or more commands is an error that lists the qualified
    candidates.
  - A reference that resolves through inheritance is valid; Rule 1
    reachability ("a sibling context is not reachable") is enforced by the
    same resolution.
- **The field is optional for producer edges.** It is backward compatible
  with today's command-only `available_from`.
  - Without a field, no type check is possible, and `--check` reports an
    *info* line suggesting a field.
  - The field is required for copy and forwarding edges.

**Classifying an edge.** Classification happens when the workflow is
loaded, once per process and cached with the existing per-workflow caches:

- With `kind="auto"`, look up `<field>` in the target's `Signature.Output`
  and then in its `Signature.Input`.
  - Found in exactly one: the kind is that one.
  - Found in both: the kind is `output`. This is today's `fact_from`
    resolution (`facts.py`: a field is read from the stored Output, or,
    when the Output has no such field, from the Input), and existing
    files must keep resolving the same way. Pass `kind="input"` to target
    the Input field instead.
  - Found in neither: error.
- A copy edge whose kind is `input` reads the value from the archived
  run's parameters, as `fact_from` does today. It is allowed.
- Then the declaring site and the target kind must form one of these
  combinations:

| Declaring site | Target kind | Edge | Allowed |
|---|---|---|---|
| A leaf `Input` field | output, or command-only | producer | yes |
| A leaf `Input` field | input | — | **error**: a leaf does not call other commands |
| A composite `Input` field | output, or command-only | producer | yes |
| A composite `Input` field | input | forwarding | yes |
| A phase facts field | output | copy | yes |
| A phase facts field | input | copy (from the run's parameters) | yes |
| A composite `Signature.Output` field, when that `Output` is a phase's facts model | output or input | copy | yes |
| Any other `Output` field (every leaf `Output`; a composite `Output` no phase returns) | any | — | **error**: such a field cannot declare `from_` (`available_from` is only read on Input fields) |

**What each edge does.**

- **Producer.**
  - At load, `"available_from"` is written into the field's
    `json_schema_extra` as a list of strings:
    - `<qualified command>.<field>` when a field was given;
    - `<qualified command>` otherwise.
  - If the author also wrote `available_from` by hand, the two lists are
    merged, with duplicates removed and order preserved.
  - The type check: the target field's annotation must equal the
    declaring annotation, or be `list[T]` / `Optional[list[T]]` of it, or
    be equal modulo `Optional`. Anything else is an error.
- **Copy.** Identical to today's `fact_from` runtime behaviour, plus the
  same type check (equality modulo `Optional`).
- **Forwarding.**
  - At load, copy `description`, `examples` and `pattern` from the target
    field into the declaring field *only when the declaring field did not
    set them*.
  - Rebuild the model with `model_rebuild(force=True)` after mutating
    `model_fields`.
  - Do **not** copy `available_from`, `from`, or any other
    `json_schema_extra` key.
  - The declaring annotation must be supported by `_spec_from_field` and
    equal the target's annotation modulo `Optional`; otherwise it is an
    error.
  - Forwarding writes **no** `available_from`.
  - Forwarding runs before `command_from_module`
    (`skill_commands.py:1001`, through `skill_from_parts` at `:1030`)
    builds the composite's parameter specs with `_spec_from_field`. The specs copy each field's description, so
    forwarding applied later would never reach the typed planning
    signature (`typed_signature`, `:1970`).

**Where errors are reported.**

- Every error above makes `fastworkflow skill preview --check` fail.
  Today `--check` (`skill_preview.check_previews`) fails only for
  previews that differ, skill files that did not load, and
  `phase_source_problems`; it never looks at leaf commands. So
  `check_previews` also appends every problem `resolve_field_sources`
  (Phase 2) reports, for leaves and composites alike.
- Loading the workflow logs the error and skips only the offending command.
  That is the current policy of `check_step_commands`.
- `check_phase_sources` (`skill_commands.py:1155`, which today only
  *warns* on a `"from"` naming an unknown command field) is upgraded to the
  same error policy. A malformed `"from"` is already a `SkillFormatError`
  today (`_checked_plan`, via `fact_sources`).

**How consumers render the edges.**

- **`command_metadata_api.py:140-155` and `:990-991`.** Keep the key
  `available_from` and keep the current `str(list)` rendering. Entries now
  carry the `.field` suffix when known. No other change, so the agent and
  planner prompts change only by that suffix. The branch at `:1012-1013`
  that reads `available_from` from *output* fields stays as dead code:
  no Output field can carry `available_from` (the table above), so it
  never fires. Do not remove it in this project.
- **The missing-parameter message (`utils/signatures.py:1260-1267`).**
  Strip any `.field` suffix and the `<Context>/` prefix before joining, so
  the message text stays exactly as it is today for existing workflows.
- **The planner docstring (`workflow_agent.py:1779`).** Unchanged.

### 3.4 Composite commands in the registry

**`CommandDirectory`** (`command_directory.py`):

- Root and context command files are both registered by
  `_register_command` (`command_directory.py:296`, called from the scan
  at `:262-281`). It is deliberately lazy: it does not import the module,
  and it hardcodes `response_generation_class_name="ResponseGenerator"`.
  (`:440-445` is `_register_core_commands`, the CME core commands, which
  are never composites.)
- `_register_command` detects a composite by parsing the file with `ast`,
  without importing it, and checking for a top-level `PlanGenerator`
  class. Importing here is not allowed: it would defeat the lazy load.
  Both a top-level `ResponseGenerator` and a top-level `PlanGenerator` is
  the load error of 3.1.
- For a composite:
  - `response_generation_class_name="PlanGenerator"`;
  - a new metadata field `is_composite: bool = True`.

  Leaves get `is_composite=False`. The field is serialised into
  `command_directory.json`; a snapshot without it loads with
  `is_composite=False`.
- `_skills/` files are registered through the same path (Phase 3), with
  their qualified names.
- **The cache fingerprint.** `_command_source_roots` (`:638`) covers only
  `_commands` roots. Add each contributing `_skills` root to it, so adding,
  editing or removing a skill file invalidates `command_directory.json`
  and `routing_definition.json`. This also changes the function-cache and
  FastAPI checkpoint namespaces that key on the same fingerprint, which is
  the intended meaning ("changes when the workflow's commands change").

**Routing.** Composites are part of the `RoutingDefinition` for their
context, like leaves. That is what lets a step that names a composite
reach it through the CME workflow's exact-prefix name match
(`_workflows/command_metadata_extraction/intent_detection.py`). They are
never in the classifier's label set (Training, below).

**The argument path when a composite is named.** A composite named in a
step (by the executor agent, by a planner step, or by a user typing its
name) takes the same path as a leaf: `_invoke_command_impl` first runs the
CME workflow (`perform_action(chat_session.cme_workflow, Action(command_name="wildcard", ...))`,
`command_executor.py:260`), which detects the command and extracts
`cmd_parameters` into the composite's `Signature.Input`.
`run_composite` then checks those values with `check_arguments`
(`skill_commands.py:1627`), with the invoking step text as its `request`.
The step text is `_invoke_command_impl`'s `command` argument, not the
`raw_user_message` it may substitute for the response generator
(`:296-298`).
There is no typed planning call (`plan_skill_call`) for a named
composite. A `SkillInputError` or `SkillCallError` from the check is
returned as a failed `CommandResponse` whose `response` is the error text,
so the caller can rewrite its step. Only a composite matched at the turn
frame by `match_skill` is planned with `plan_skill_call`, exactly as a
skill is today.

**Dispatch** (`command_executor.py`). `_invoke_command_impl` checks
`is_composite` after parameter extraction and before it instantiates the
response-generation class (`:286-294`); a `PlanGenerator` is never
called with `(workflow, command, command_parameters)`:

- **Agent mode.** The composite is executed by
  `WorkflowExecutionContext.run_composite(qualified_name,
  command_parameters, step_text)` (3.5), which returns a `CommandOutput`.
- **Assistant (deterministic) mode, including `/`-prefixed commands in
  agent mode.** Return `CommandOutput` with
  `CommandResponse(success=False, response="<name> is a composite command;
  it runs in agent mode.")`. Composites need an LLM executor by definition.
- **Direct actions.** `CommandExecutor.perform_action` (`:337`) and
  `perform_mcp_tool_call` (`:466`), which back FastAPI's direct action
  calls and MCP tool calls, return the same refusal when the action names
  a composite of the app workflow. They run one command with given
  parameters and no executor. (`_invoke_command_impl`'s own
  `perform_action` call on the CME workflow names `wildcard`, which is
  never a composite.)

**The composite output contract.** `run_composite` always returns exactly
one `CommandResponse`:

- `response`: the final answer text, which is what the skill returns today.
- `output`: the composite's `Signature.Output` instance, or `None`.
  - No skill reads `Signature.Output` today, so this is new. A composite
    that declares an `Output` must have exactly one phase method whose
    return annotation is `Signature.Output`; zero or several is a load
    error. That phase's facts, filled by the existing `fill_output` like
    any phase's facts, are the output.
  - `output` is `None` when the composite declares no `Output`, or when
    that phase was skipped or did not fill its facts in this call.
  - A flat composite (template-only, or one phase without facts) cannot
    declare an `Output`; that is a load error.
- `success`: `False` when the frame ran out of budget, failed its
  argument check, or failed.

**Catalogs.**

- The ReAct executor's tool catalog (`_what_can_i_do`, passed as
  `available_commands`) and the planner's command catalog
  (`build_query_with_next_steps`) include composites on the current
  context surface when `FW_COMPOSITE_NESTING` is on (Phase 5). They are
  shown with their Signature, like leaves.
- **MCP `list_tools`** (`mcp_server.py`, built from
  `routing.get_command_names`) excludes composites, because every MCP
  tool call goes through `perform_mcp_tool_call`, which refuses them.
- `match_skill` / `load_skills` build their options from composites on the
  current context surface. That includes `_skills/` files until Phase 7
  folds them in.

**Training** (`fastworkflow train`). Commands with `is_composite=True` are
excluded from:

- utterance generation;
- classifier training;
- the duplicate-capability check.

Because composites are in the `RoutingDefinition` (Routing, above), the
filter is explicit at every site that enumerates commands for training:

- the per-context label set in `model_pipeline_training.py` (built from
  `crd.contexts[ctx]` plus the core commands, around `:1345-1351`);
- `_validate_command_inputs` (`train/__main__.py:83`) and
  `_get_commands_with_parameters` (`:671`);
- `train/duplicate_detection.py:683`;
- `train/selective_training.py:355`. Their `plain_utterances`, if declared, are used only by
`match_skill`'s listing, as today (`skill_utterances`).

### 3.5 The `ExecutionFrame` stack

**New module: `fastworkflow/execution_frame.py`.**

```python
@dataclass
class ExecutionFrame:
    kind: Literal["turn", "composite"]
    command_name: Optional[str]          # qualified composite name; None for the turn frame
    depth: int                           # 0 for the turn frame
    alias: str                           # "" for turn; "<parent alias>/<stem>#<n>" otherwise
    subtasks: list[str]
    origins: list[int]
    subtask_index: int
    subtask_answers: list[str]
    subtask_exhausted: bool
    evidence_floors: dict[int, int]      # not serialised (as today)
    initial_plan_text: str               # not serialised (as today)
    phase_run: Optional[dict[str, Any]]
    phase_live: Optional[tuple[Any, list[Any]]]   # not serialised (as today)
    phased_skill_call: Any = None        # not serialised (as today)
    child_step_ranges: list[list[int]] = field(default_factory=list)  # [first, last] per child run
    invoking_step: Optional[int] = None  # the parent step index this frame runs inside
    agent: Any = None                    # per-frame fastWorkflowReAct, built lazily
    plan_question: Optional[dict[str, Any]] = None  # turn frame only (3.6)
    def to_state(self) -> dict: ...      # see Serialisation below
    @classmethod
    def from_state(cls, d: dict) -> "ExecutionFrame": ...
```

`next_step` is deliberately **not** a frame field. `_next_step` is the
turn-wide next free step index (`_run_subtasks_from`, `_record_subtask_answer`
and the phase runner all advance it), and step indexes key the observation
archive (`O{idx}` aliases, `observation_offloading/state.py:current_execute_alias`;
`_archived_command_runs(from_step, to_step)`). One counter keeps every
step index in the turn unique across frames, so it stays a plain attribute
on the context.

**Integrating it into `WorkflowExecutionContext`.**

- Add `self._frames: list[ExecutionFrame]`. It always holds at least the
  turn frame.
- Replace the attributes listed in 1.2 item 6, except `_next_step`, with
  **properties that read and write `self._frames[-1]`**. Existing code
  that refers to `self._subtasks` and the others keeps working, and the
  diff stays small. `_workflow_tool_agent` becomes a property for
  `self._frames[-1].agent`.
- Four places are about the turn, not the current frame, and must name
  `self._frames[0]` explicitly instead of going through the properties:
  - `serialize_state` / `apply_serialized_state` for the top-level keys;
  - `_begin_turn`, which resets `_frames` to a single fresh turn frame;
  - `_hand_off` (`:2650`), which is turn-frame only;
  - `_clear_subtask_state` (`:1834`, called by `_run_agent` at turn
    start). A composite frame's state is released by popping it, never by
    clearing it.
- `_ensure_agent_initialized` builds the turn frame's agent, as today. A
  composite frame's agent is built by `run_composite` with the same
  factory (`initialize_workflow_tool_agent`), so it gets the same tools,
  `max_iters` and adapter.

**Step numbering across frames.** `fastWorkflowReAct.forward` numbers
steps locally from `first_step` and advances `idx` by one per step
(`utils/react.py`). A child run inside the parent's step `k` must not reuse
`k + 1`, … , which the parent would otherwise number next.

- A child agent's run starts at
  `first_step = max(self._next_step, k + 1)`, where `k` is the parent's
  in-flight step (the index `current_execute_alias` reads). The child's
  frame records `invoking_step = k`.
- When the child run ends, `self._next_step` is already past its last step
  (the existing advance at `:2516` and `:2698` runs on the child frame).
- `fastWorkflowReAct` gains one optional hook, `step_floor:
  Callable[[], int] | None`, defaulting to `None`. Wherever `_run_loop`
  advances `idx`, it sets `idx = max(idx + 1, step_floor())` when the hook
  is set. `WorkflowExecutionContext` sets it on every agent it builds to
  `lambda: self._next_step`. A parent therefore resumes numbering after
  the child's steps, and with no nesting the hook returns a value `<=`
  `idx + 1`, so numbering is unchanged.
- The parent frame appends `[first, last]` of the child's steps to
  `child_step_ranges`.

**`run_composite(name, params, step_text)`.**

1. If `len(self._frames) - 1 >= FW_COMPOSITE_MAX_DEPTH`, return
   `CommandResponse(success=False, response="Composite nesting limit
   reached; call the steps of <name> directly.")`. No frame is pushed. The
   agent can continue with leaves.
2. Cycle guard: if `name` is the `command_name` of any frame on the stack,
   return the same kind of failure with "<name> is already running".
3. Push a frame with:
   - `kind="composite"`, `depth = parent.depth + 1`;
   - `alias = f"{parent.alias}/{stem}#{n}"`, where `n` counts calls to
     `stem` within the parent frame, starting at 1;
   - the subtask list set to `[step_text]`, the single step that invoked
     it;
   - `invoking_step` set as described under Step numbering.
4. **Check the arguments.** For a composite with parameters, call
   `check_arguments(skill_command, params.model_dump(), step_text)`
   (3.4, The argument path). On `SkillInputError` or `SkillCallError`, pop
   the frame and return a failed `CommandResponse` with the error text. No
   LLM call is made here.
5. **Build the call and run it** on the top frame, through the properties:
   - For a composite with parameters or phases, build the same
     `SkillCallPlan` `plan_skill_call` returns, without its typed LLM
     call: `plan` from the composite's steps for the checked arguments,
     `phases` from `call_phases(skill_command, args)`
     (`skill_commands.py:1288`), no `steps_before` / `steps_after`. Run it
     with `_start_skill_phases(step_text, skill_command, call,
     first_step=...)` (`:2079`). This is the path every typed skill call
     takes today, flat ones included: `call_phases` always returns at
     least one phase, and a flat call is the one phase named after the
     skill, run with the agent's `max_iters` (`state["flat"]`, `:2106`,
     `:2381-2382`).
   - For a template-only composite (no parameters and no phases), take
     the template branch of `_call_agent_for_query` (`:1972`): one
     `build_query_with_next_steps` call with `plan_template` set to the
     composite's template and `skill_command=None`, then one executor run.
     Pass the template directly; do not re-match with `planner_skill_for`.
6. On completion, failure or exhaustion:
   - build `CommandOutput(command_response=CommandResponse(...))` with the
     response described in 3.4;
   - append the child's step range to the parent's `child_step_ranges`;
   - pop the frame, which releases `agent` and `phase_live`;
   - return the output. The executor then holds it as the run of the
     parent's step `k` (`_hold_command_run`, `workflow_agent.py:96`),
     because after the pop `current_execute_alias` reads the parent's
     in-flight step.

**Ask_user inside a child (Topology B).** `fastWorkflowReAct` catches
`AskUserSuspend` inside its loop and returns a suspended `Prediction`; the
exception does not propagate out of the child's agent. So:

1. The child's phase runner returns its suspended result to
   `run_composite`.
2. `run_composite` does **not** pop. It raises
   `AskUserSuspend(clarification_request)` with the child's clarification.
   `AskUserSuspend` is a `BaseException`, so no `except Exception` on the
   way up swallows it.
3. That exception passes up through the parent's `_execute_workflow_query`
   and is caught by the parent's `_run_loop` at step `k`, which stashes
   the parent's own suspension, exactly as for a leaf that asks.
4. Every frame up to the turn frame returns suspended the same way, and
   `_process_agent_message` ends the turn with `_awaiting_user_output`.
   The stack is serialised with the child on top and every agent on it
   suspended.

**Resuming a nested suspension.** `_resume_agent_message` (`:2790`) gains
one step, after `_post_ask_user_response` builds the observation from the
user's answer and before its existing phased/plain resume:

- While `len(self._frames) > 1`: resume the top frame (its phase runner
  with `resume_observation` when it has `phase_run`, else its agent's
  `resume`).
  - If it suspends again, return the suspended output as today.
  - Otherwise build its `CommandOutput`, record its step range, pop it,
    hold the output as the parent's step `k` run, and turn it into the
    observation that `_execute_workflow_query` returns for a command
    output (factor that formatting into one helper both use). That text
    becomes the observation for the next frame's resume.
- At the turn frame, continue with the existing code using the last
  observation.

`_record_subtask_answer`, `_hand_off` and the "No suspended ReAct state"
check apply to the turn frame only.

**Ask_user inside a child (Topology A).** The CLI path's ask_user blocks on
the user-message queue inside the tool call (`_ask_user_tool`), with no
suspension. A child that asks simply blocks, and nothing above changes.

**Hand-off.** `_hand_off` and `MAX_HANDOFFS = 2` apply only in the turn
frame. A composite frame that exhausts its budget returns `success=False`
to its parent and does not hand off.

**The turn-wide step budget for nested frames.**

- **What it counts.** Model iterations (the steps that increment
  `fastWorkflowReAct.iteration_counter`; code-run steps do not) taken by
  agents of composite frames, depth 1 and below. Steps of the turn frame
  are not counted. The turn frame is already bounded by today's caps:
  `MAX_SUBTASKS = 10` (`workflow_agent.py:1019`) sub-tasks, each with at
  most `MAX_HANDOFFS + 1 = 3` executor runs of `max_iters = 25`, or
  phases of `FW_SKILL_PHASE_MAX_STEPS = 15` (times up to
  `FW_SKILL_FANOUT_MAX_ITEMS = 10` items for a fan-out phase). That is up
  to about 750 model iterations in a turn today, so a turn-wide cap of 200
  on the turn frame would cut off turns that work now. The budget exists
  to bound what nesting adds, and it counts exactly that.
- `FW_TURN_MAX_STEPS`, default `200`, is read through
  `fastworkflow.get_env_var` with the same validation as
  `phase_step_budget()` (`skill_phases.py:432`).
- One counter, `self._turn_steps_used`, is reset in `_begin_turn` and
  serialised.
- `fastWorkflowReAct` gains a second optional hook, `on_iteration:
  Callable[[], bool] | None`, called right after each
  `iteration_counter += 1`. `run_composite` sets it on every composite
  frame's agent; it increments `_turn_steps_used` and returns `False`
  once the counter reaches `FW_TURN_MAX_STEPS`. The loop treats `False`
  exactly like reaching `max_iters`: it sets `_exhausted_last_run` and
  stops. The turn frame's agent has no hook.
- When the budget is spent, the composite frame returns `success=False`
  to its parent with the response "The step budget for nested commands
  is spent; continue with the remaining steps directly." Every further
  `run_composite` in the same turn fails the same way without pushing a
  frame. The parent, and the turn frame, continue under their own caps.

**The depth cap.** `FW_COMPOSITE_MAX_DEPTH`, default `3`. The turn frame is
depth 0, and a composite matched or called from it is depth 1.

**Attributing runs.** `fill_output` and its helpers (`runs_on_plan`,
`planned_record`, `runs_finding_planned_values`, `runs_that_entered_record`,
`runs_standing_in_for_openers` in `skill_phases.py`) take the runs the phase
runner passes them; they do not query the archive themselves. The phase
runner reads them with `_archived_command_runs(from_step, to_step)`
(`workflow_execution_context.py:2324`), which filters by step index. So
attribution is done by step ranges, in the runner, not by a new archive
column:

- Wherever the context reads archived runs for a frame (the phase
  runner's facts read, the finish check's evidence read, and
  `earlier_runs`), it drops every run whose step lies inside one of that
  frame's `child_step_ranges`.
- A child composite therefore contributes exactly one run to its parent:
  the run held under the parent's step `k` (step 6 above), whose
  `command` is the composite's qualified name, whose `input` is its
  checked arguments, and whose `output` is its `CommandResponse.output`.
  A copy `from_("Ctx/composite.field")` on a parent facts field reads that
  output like any command's.
- The observability store (`observability/store.py`, `SCHEMA_VERSION = 10`,
  which refuses older stores and has no migrations) is **not** changed.
  `offload_command_runs` keeps its primary key of turn key, step alias
  and ordinal, which stays unique because step indexes are unique across
  frames.

**Tracing.** Every span opened while a composite frame is on top gets
the attribute `frame_alias` (the frame's `alias`; spans of the turn frame
get `""`). Adding an attribute changes the span contract, so bump
`SPAN_CONTRACT_VERSION` in `tracing.py` from `23` to `24`, in the same
change, following that constant's existing rules.

**Serialisation** (`serialize_state` / `apply_serialized_state`,
`workflow_execution_context.py:977-1139`).

- The turn frame keeps using the existing top-level keys (`subtasks`,
  `subtask_origins`, `subtask_index`, `subtask_answers`,
  `subtask_exhausted`, `next_step`, `skill_phase_run`, `react`), read
  from and written to `self._frames[0]` (and `_next_step`), so old blobs
  restore unchanged.
- Add these keys:
  - `"frames"`: `[frame.to_state() for frame in self._frames[1:]]`.
    `to_state` writes `kind`, `command_name`, `depth`, `alias`,
    `subtasks`, `origins`, `subtask_index`, `subtask_answers`,
    `subtask_exhausted`, `phase_run`, `child_step_ranges` and
    `invoking_step`. It leaves out what the turn frame leaves out today
    (`evidence_floors`, `initial_plan_text`, `phase_live`,
    `phased_skill_call`) and `agent`.
  - `"turn_child_step_ranges"`: the turn frame's `child_step_ranges`.
  - `"frame_react"`: the `export_suspended()` blob of each nested frame's
    agent, as a list aligned with `"frames"`, with `None` where the frame
    has no agent. On restore, each is imported into a fresh agent built
    with the same factory, under the same `awaiting_user` condition as
    `"react"` (`:1121-1125`).
  - `"turn_steps_used"`: the nested-frame step counter.
  - `"plan_question"`: the turn frame's plan question (3.6).
- **Do not bump the session-state `SCHEMA_VERSION` (`4`).** Absent keys
  mean "no nested frames, no child ranges, zero steps used, no plan
  question". This follows the file's existing "absent from blobs written
  before …" pattern.
- `validate_state` (`state_serialization.py:27`) needs no change. It is a
  generic check that the payload round-trips through JSON exactly, not a
  list of allowed keys. Two of its rules constrain the new values: every
  dictionary key must be a string, and no mutable container may be
  reachable by two paths. So `to_state` returns fresh lists and dicts,
  never the frame's own objects.

**Memory.** Popped frames are not retained. Live memory is the frames on
the current branch: at most `FW_COMPOSITE_MAX_DEPTH + 1` frames.

### 3.6 Ask_user while planning

**Where planning happens.** Both triggers fire inside
`_call_agent_for_query` (`workflow_execution_context.py:1972`), which runs
once per sub-task after the request was refined and split: from
`_run_agent` for an unsplit request, and from `_run_subtasks_from` for each
sub-task. Matching is `planner_skill_for` → `match_skill`
(`workflow_agent.py:1466`), and the typed call is `plan_skill_call`
inside `build_query_with_next_steps` (`workflow_agent.py:1873-1888`),
whose `except Exception` falls back to the text planner.

**Two triggers**, both only when all of these hold; otherwise today's
fallback is kept:

- `not self._turn_unattended`;
- the turn frame is on top (`len(self._frames) == 1`);
- the call is not a hand-off continuation (`template_text` is `None` or
  equals the query; a continuation already plans from the template only);
- the new keyword argument `allow_plan_question` of
  `_call_agent_for_query` is `True` (its default; the re-run after a
  plan question passes `False`).

**Trigger A: an ambiguous composite match.**

- Add `match_skill_scored(text, workflow_folderpath)`, returning the
  matched name (or `None`) and the probabilities of the top two
  non-`"none"` options, read from the Jev choice's per-option
  `probabilities` dict (the same dict `_choice_value_and_probability`
  reads). `match_skill` keeps its signature and behaviour by wrapping it.
  When the choice has no `probabilities`, trigger A does not fire.
- `_call_agent_for_query` calls it, through a scored variant of
  `planner_skill_for`, at the turn frame.
- Ask when all three hold:
  - no option reaches `SKILL_MATCH_THRESHOLD`;
  - both of the top two options are `>= FW_SKILL_ASK_MIN_PROBABILITY`
    (default `0.3`);
  - their difference is `< FW_SKILL_ASK_MAX_GAP` (default `0.15`).
- The question text: `"Do you want me to <description A>, or to
  <description B>? (Or say neither.)"`

**Trigger B: a required composite argument that cannot be filled.**

- This applies after `plan_skill_call`'s existing single retry, when the
  remaining `SkillInputError` is about a *required* argument that is
  missing or not written in the request.
- `SkillInputError` (`skill_commands.py:245`) today carries only a
  message. Give it two optional attributes, `param: str | None` and
  `reason: Literal["missing", "not_in_request", "invalid"] | None`, set
  by `check_arguments` where it refuses a required argument. Trigger B
  fires only for `reason` in `{"missing", "not_in_request"}` with `param`
  set. Never parse the message text.
- The hook is the `except Exception` around `plan_skill_call` in
  `build_query_with_next_steps`, before it falls back to the text planner.
  When trigger B applies, that function records the question (step 1
  below), ends its planning span normally with `plan_source="question"`,
  and returns without planning.
- The question text: `"To <composite description>, I need <field
  description>. What is it?"`
- Other refusals keep today's fallback.

**The mechanism.** It applies at the turn frame only. A nested composite
gets its arguments from its caller's step and never asks.

1. Set `frames[0].plan_question` to
   `{"trigger": "match"|"argument", "subtask_index": i, "question": text,
   "candidates": [A, B] | None, "composite": name | None,
   "field": field | None}`.
2. `_call_agent_for_query` sees the recorded question and returns
   `dspy.Prediction(suspended=True, clarification=text, exhausted=False)`
   without running an executor. `_run_subtasks_from` already returns a
   suspended result unchanged, and `_process_agent_message` (`:2764`)
   already turns one into `_awaiting_user = True`,
   `_pending_clarification_request = text` and `_awaiting_user_output`.
   That is the whole exit path; no new one is added.
3. On the next message, `_execute_message` (`:1426-1436`) skips
   `_begin_turn` and routes to `_resume_agent_message` because
   `_awaiting_user` is set.
   `_resume_agent_message` checks `frames[0].plan_question` **first**,
   before its "No suspended ReAct state" check: no agent is suspended for
   a plan question, so that check would otherwise raise
   `NoSuspendedAgentStateError`. For a plan question it:
   - appends the dialog to the action log (`_append_action_record`), as
     `_post_ask_user_response` does, but does not call
     `_post_ask_user_response`, which replans for a suspended executor;
   - classifies the reply with
     `classify_user_reply(reply, question)` (`workflow_agent.py:1532`,
     which uses `ClassifyReply` / `REPLY_INTENTS` and returns `"answer"`
     without a Jev key);
   - clears `plan_question`, then acts on the intent for sub-task `i`:
     - **`abort`**: run nothing for the sub-task; its result is
       `dspy.Prediction(final_answer="Stopped, as you asked.",
       exhausted=False)`. This matches
       today's abort, which leaves an agent only `finish` for the
       current sub-task (`_apply_reply_intent`).
     - **`stop_asking`**: run the sub-task with today's fallback (the text
       planner, with the matched template if any), with
       `allow_plan_question=False`.
     - **`answer` to trigger A**: call `match_skill_scored` on
       `"<sub-task> — <reply>"`. If it returns one of the two candidates
       at or above `SKILL_MATCH_THRESHOLD`, run the sub-task with that
       composite; otherwise use the fallback. Either way pass
       `allow_plan_question=False`, so no sub-task is asked twice.
     - **`answer` to trigger B**: run the sub-task with the request text
       `"<sub-task>\n<field>: <reply>"`, so the reply is written in the
       request that `check_arguments` reads, with
       `allow_plan_question=False`. If the typed call is refused again,
       the existing fallback applies.
   - Each "run the sub-task" is `_call_agent_for_query` with the same
     `first_step`, `template_text` and `evidence_floor` that
     `_run_subtasks_from` passes. The result then goes through the
     existing tail of `_resume_agent_message` (`:2815-2828`):
     `_record_subtask_answer`, hand-off on exhaustion, and
     `_run_subtasks_from(i + 1)`.
4. `plan_question` is serialised (3.5), so a resume in another process
   works.

### 3.7 The callee graph and the preview

- **At load**, build `callees[name] -> set[qualified name]` from every
  command step in each composite's placeholder templates. The existing
  `check_step_commands` already parses them; reuse its parse.
- **A cycle in the graph is a load error** for every composite on the
  cycle, and makes `preview --check` fail.
- **`fastworkflow skill preview`** adds two items per composite:
  - a `Calls:` line listing its callees;
  - `Depth: n (lower bound; open steps name no command)`.

  The preview is the only consumer of the depth number.
- **Rename the user-facing name** to `fastworkflow command preview`, and
  keep `skill preview` as an alias that prints nothing extra (both parsers
  in `cli.py`, next to `add_skill_parser`, call `preview_main`).
  `skill_preview.py` keeps its module name. The generated-file marker
  that `write_previews` looks for (``<!-- Generated by `fastworkflow skill
  preview` ``, `skill_preview.py:112`) keeps its exact text, so existing
  generated previews are still recognised and replaced.
- **Where composites are found.** `render_previews` (`:1494`) today raises
  `PreviewError` when the workflow has no `_skills` folder and lists only
  `_skills/` files. It is changed to list every composite of the workflow
  (from `CommandDirectory`, `is_composite=True`, plus `_skills/` until
  Phase 7 ends), and to raise only when there are none.
- **Context composites in the output folder.** A qualified name
  `Library/reserve_and_notify` is written to
  `<out>/Library/reserve_and_notify.md`. `check_previews` and
  `write_previews` walk subfolders of `<out>` the same way.
- **Pinned previews.** `tests/test_skill_tutorials.py` asserts
  `check_previews(tests/skill_tutorial_workflow,
  docs/tutorials/skills/previews) == []`. Every change in this section,
  and the move in Phase 7 (each preview names its source file), changes
  rendered previews. The phase that makes the change regenerates
  `docs/tutorials/skills/previews/` with `fastworkflow skill preview
  tests/skill_tutorial_workflow --out docs/tutorials/skills/previews` in
  the same commit, and checks that only the expected lines changed.

### 3.8 Protecting composites from the build generator

- **The generated-file header.** `build/command_file_template.py` writes
  `# fastworkflow:generated — regenerate with fastworkflow build; edits may
  be overwritten` as the first line of every file it creates.
- **Refuse to overwrite composites.** `create_command_file`
  (`command_file_template.py:143`) and `create_function_command_file`
  (`:22`), before writing over an existing file, refuse when that file
  defines a top-level `PlanGenerator` (checked with `ast`, without
  importing it). This holds even when `overwrite=True`, and it is the
  only refusal condition; the header does not change it.

  A refusal is a warning naming the file. Existing generated leaf files
  without the header are still overwritten (backward compatible); new
  ones get the header. The header is informational: it tells a reader the
  file is generated, and nothing reads it to decide anything.
- **The build validator.** `validate_command_file_components_in_dir`
  (`build/command_file_generator.py:162`) reports every `.py` file in
  `_commands/` that lacks the leaf components (a `ResponseGenerator`
  with `__call__`, and the rest). It skips a file with a top-level
  `PlanGenerator` for those leaf checks and checks only that it has a
  `Signature`. `verify_commands_against_context_model` (`:275`) and
  `validate_command_imports` (`:349`) treat composite files like leaf
  files: they are commands in their context folder and must import.
- **Name collisions.** If the generator would create a leaf whose
  qualified name equals an existing composite's, it skips that file and
  warns.

---

## 4. Phased implementation plan

General rules for every phase:

- Activate `.venv` first.
- **Tests:**
  - All tests are integration tests against real workflows
    (`tests/example_workflow`, `tests/hello_world_workflow`,
    `tests/todo_list_workflow`, and the skill tutorial workflow
    `tests/skill_tutorial_workflow`, whose files the pages in
    `docs/tutorials/skills/` quote). Do not use mocks.
  - Do not delete any existing test. Where existing behaviour is
    intentionally changed, update the assertion in place and say why in
    the commit message.
  - Never run two full test suites at once. The full suite takes about
    2 hours 7 minutes. Run targeted files while iterating, and run the
    full suite once at the end of each phase.
- **Tracking:** create one bd epic for this project and one child issue per
  phase. Close each child as its phase lands, close the epic after
  Phase 8, and run `bd export -o .beads/issues.jsonl` after every bd
  write.
- **Branches:** every phase is its own commit series and its own PR, in
  this order. Each phase must leave `main` working.
- **Coding skills:** load the `fastworkflow-architecture-contract` and
  `fastworkflow-change-control` skills before Phases 4–6.

### Phase 0: baseline

1. Record the current full-suite result (pass, fail and skip counts) in the
   epic's description. That is the regression baseline for every later
   phase.
2. Record the per-tutorial outputs of `tests/test_skill_tutorials.py` and
   of `fastworkflow skill preview tests/skill_tutorial_workflow --out
   docs/tutorials/skills/previews --check`. `tests/skill_tutorial_workflow`
   is the only workflow in the repository with a `_skills/` folder; no
   workflow under `fastworkflow/examples/` has one, and none has a
   `_skills/_split.py`.

**Done when:** both baselines are recorded.

### Phase 1: skill contexts

1. In `_skill_files` (`workflow_agent.py:1119`), load
   `_skills/<Context>/<name>.py` files. The qualified name is
   `<Context>/<name>`. Keep skipping `_`- and `.`-prefixed entries, and
   remove the "not loaded yet" warning. Root files keep their bare names.
2. **Visibility.** `load_skills` / `_skill_matcher` / `load_skill_commands`
   take the active context name. A context composite is offered when its
   context is in the active context's resolved surface: own or inherited,
   per the existing routing resolution, or a live ancestor. Root
   composites are always offered.
   - The caches (`_build_parsed_skills`, `_skill_matcher`) keep their
     per-workflow key and load every skill once per process; visibility
     filters that full load by the active context. Do not key the caches
     on the context name.
   - The callers pass the active context: `planner_skill_for` /
     `match_skill` are called from `_call_agent_for_query`, which reads it
     from the active workflow's `current_command_context_name`.
3. **Unknown contexts.** `_skills/<Context>/` where `<Context>` is not a
   context of the workflow (no `_commands/<Context>/` folder and not in
   `context_inheritance_model.json`) is a load error for each of its
   files.
4. **Collisions.** A skill with the same qualified name as a command (a
   root skill and a root command, or a context skill and a command in that
   context) is a load error for the skill, reported by `preview --check`.
5. **Preview output.** Write context skills to `<out>/<Context>/<name>.md`
   (3.7), and regenerate `docs/tutorials/skills/previews/` if any preview
   text changed.
6. **Tests.** Add `tests/test_skill_contexts.py`. Use a test workflow with
   `_skills/Library/` and a root skill and verify:
   - the context skill matches only when `Library` (or a context
     inheriting from it) is active;
   - the root skill matches everywhere;
   - preview lists both;
   - a skill folder for an unknown context is a load error.

**Done when:** the new tests pass, the Phase 0 baselines are unchanged,
and the full suite shows no new failures.

### Phase 2: `from_`

1. Implement `from_`, the `fact_from` alias, and the exports (3.3). Add
   `"from"`, `"from_kind"` and `"available_from"` to `SKILL_FIELD_KEYS`.
2. **Load-time resolution.** Add `resolve_field_sources(workflow)` in
   `skill_commands.py`. It runs once per workflow when commands and skills
   are loaded, classifies every `from_`, runs the type checks, and applies
   the forwarding copy and the producer `available_from` write. It
   returns its problems, which `check_previews` reports (3.3). Upgrade
   `check_phase_sources` to the error policy.
3. **Leaf commands.** Run the same resolution for `Signature.Input` fields
   of leaf commands when `CommandDirectory` hydrates a command's
   `Signature` (`ensure_command_hydrated`, the lazy path). Cache the
   result per workflow. The producer `available_from` write happens on
   the hydrated `Signature` class, before `command_metadata_api.py` reads
   it.
4. **Rendering.** Update the missing-parameter message (strip suffix and
   prefix) and leave `command_metadata_api.py` emitting the merged list
   (3.3).
5. **Tests.** Add `tests/test_field_sources.py`, covering:
   - each allowed edge;
   - each error row of the table in 3.3;
   - auto-kind ambiguity;
   - unqualified-name resolution, including the ambiguity error;
   - forwarding not producing `available_from`;
   - the missing-parameter message text being byte-identical for
     `fastworkflow/examples/retail_workflow` before and after the
     migration in step 6 (the message strips the `.field` suffix);
   - metadata output containing `.field` suffixes when `from_` is used.
6. **Migrate one example.** In `fastworkflow/examples/retail_workflow/_commands/`,
   convert the two declarations whose producers declare the field they
   supply:
   - `get_user_details.py` `Input.user_id` →
     `from_(["find_user_id_by_email.user_id",
     "find_user_id_by_name_zip.user_id"], description=...)`;
   - `modify_user_address.py` `Input.user_id` → the same list.

   Both producers declare `Output.user_id: str` but return a
   `CommandResponse` without `output`. Make each also set
   `output=Signature.Output(user_id=...)`, so the edge is readable at
   runtime and Phase 3's warning does not fire. Leave every
   `available_from: ["get_user_details"]` as it is: `get_user_details`'s
   `Output` has only `user_details: str` (no order or payment field), so
   a field-level reference to it is impossible without changing that
   command's Output, which is outside this project.

**Done when:** the tests pass, the retail workflow's existing tests pass,
and the full suite shows no new failures.

### Phase 3: output contract and the composite registry

1. **The skill output.** A matched skill call at the turn frame (through
   `_start_skill_phases`, flat or phased) sets `CommandResponse.output` to
   the skill's `Signature.Output`, taken from the facts of the phase that
   returns it (3.4). Add the two load errors of 3.4: a declared `Output`
   that no phase, or more than one phase, returns; and a flat composite
   that declares an `Output`.
2. **Static checks.** A producer `from_` field must exist in the target's
   `Output`, and a copy `from_` field in its `Output` or `Input` (2.6);
   otherwise it is an error.
3. **Runtime warning.** When a command named as a producer or copy target
   returns `output=None`, log
   `"<cmd> returned no Output; from_ references to it cannot be read"`
   once per command per process.
4. **The registry.** Add `is_composite` to `CommandDirectory` metadata.
   `_skills/` composites are registered through `CommandDirectory` under
   their qualified names, alongside `_commands/` composites. Add the
   `ast`-based `PlanGenerator` detection (3.4) to `_register_command`, and
   add the `_skills` roots to `_command_source_roots` (3.4, The cache
   fingerprint).
5. **Hidden until Phase 5.** In this phase composites are registered and
   routed, but nothing offers them by name:
   - they are left out of the executor's and the planner's command
     catalogs, and out of MCP `list_tools`;
   - they are reached only by `match_skill` at the turn frame, exactly as
     skills are today.
6. **Dispatch.** Add the refusals of 3.4: assistant mode and `/`-prefixed
   commands, `perform_action` and `perform_mcp_tool_call`. In agent mode,
   a step that still names a composite (for example, a user typing its
   name) is refused in this phase with
   `"<name> is a composite command; it cannot be called as a step yet."`
   Phase 5 replaces that refusal with `run_composite`.
7. **Training.** Exclude composites at every site listed in 3.4
   (Training).
8. **Tests.** Add `tests/test_composite_commands.py`, covering:
   - a `_commands/Library/book_by_title.py` composite is matched and run,
     and its `CommandResponse.output` is a filled `Signature.Output`;
   - assistant mode, `/`-prefixed and `perform_action` calls return the
     refusal text;
   - training on the test workflow ignores the composite (the trained
     label set and the duplicate check exclude it);
   - adding a file under `_skills/` changes
     `compute_commands_source_fingerprint`;
   - the producer-without-Output error;
   - the declared-`Output` load errors.

**Done when:** the tests pass, the tutorials are unchanged, and the full
suite shows no new failures.

### Phase 4: the `ExecutionFrame` refactor (no change in behaviour)

1. Add `fastworkflow/execution_frame.py` and convert the
   `WorkflowExecutionContext` attributes to properties over
   `self._frames[-1]`, keeping `_next_step` a plain attribute and making
   the four turn-only sites name `self._frames[0]` (3.5).
2. Add the `step_floor` and `on_iteration` hooks to `fastWorkflowReAct`
   (3.5). Set `step_floor` on every agent the context builds; nothing sets
   `on_iteration` yet.
3. Add the nested-frame step counter, `FW_TURN_MAX_STEPS`, and the new
   serialisation keys. `run_composite` is not yet called from anywhere.
4. Add the step-range exclusion to the context's archived-run reads
   (3.5, Attributing runs). With no child frames the ranges are empty and
   nothing is excluded.
5. Add the `frame_alias` span attribute (always `""` in this phase) and
   bump `SPAN_CONTRACT_VERSION` from `23` to `24` (3.5, Tracing).
6. **Tests.** Add `tests/test_execution_frames.py`, covering:
   - serialise and restore round-trips, including a blob without the new
     keys (an old blob) restoring correctly;
   - `serialize_state` output passing `validate_state`, with the new keys
     present;
   - step indexes of a multi-sub-task turn being identical with the
     `step_floor` hook set and unset;
   - an ask_user suspension and resume across a fresh
     `WorkflowExecutionContext` instance.

**Done when:** the new tests pass and the full suite matches the Phase 0
baseline exactly. This phase must not change behaviour.

### Phase 5: nested composites

1. **Implement `run_composite`** (3.5): depth cap, cycle guard, alias, push
   and pop, the argument check, child step numbering and ranges, the
   nested step budget (`on_iteration`), failure outputs, and no hand-off
   in nested frames. Composite dispatch in agent mode now always goes
   through it, replacing Phase 3's agent-mode refusal.
2. **Nested ask_user** (3.5): the `AskUserSuspend` raise in
   `run_composite`, and the unwinding step in `_resume_agent_message`.
3. **Add composites to the executor's and the planner's command
   catalogs** (3.4), behind `FW_COMPOSITE_NESTING`. It defaults to
   `"on"`; set it to `"off"` to keep Phase 3's behaviour, where
   composites are only matched at the turn frame and a step naming one is
   refused. MCP `list_tools` never lists them.
4. **Callee graph and cycle detection** at load. Add the preview `Calls:`
   and `Depth:` lines, and the `command preview` alias (3.7). Regenerate
   `docs/tutorials/skills/previews/`.
5. **Attribution.** A child contributes exactly one run to its parent
   (3.5).
6. **Tests.** Add `tests/test_nested_composites.py`. Use a test workflow
   modelled on `tests/skill_tutorial_workflow`, with composite A, whose
   step calls composite B, which calls leaves. Verify:
   - A's output is filled from B's typed output;
   - B's internal runs do not satisfy A's facts;
   - step indexes in the turn are unique across A's and B's runs;
   - a step naming B with an argument not written in the step returns
     the `check_arguments` error, and no frame is left on the stack;
   - the depth cap at `FW_COMPOSITE_MAX_DEPTH=1` produces the refusal and
     the agent completes with leaves;
   - the nested budget stopping B at `FW_TURN_MAX_STEPS` set low (for
     example 3), with A continuing;
   - a cycle A→B→A is a load error;
   - an ask_user inside B suspends, and resumes in a new process with the
     stack intact, then finishes A;
   - `FW_COMPOSITE_NESTING=off` restores Phase 3's behaviour;
   - live frames after completion are exactly 1.

**Done when:** the tests pass, the tutorials are unchanged, and the full
suite shows no new failures.

### Phase 6: ask_user while planning

1. Add `match_skill_scored`, the `param` / `reason` attributes of
   `SkillInputError`, the two triggers, the `allow_plan_question`
   argument, `plan_question` handling in `_resume_agent_message`, and the
   two new env settings `FW_SKILL_ASK_MIN_PROBABILITY` and
   `FW_SKILL_ASK_MAX_GAP` (3.6). `SKILL_MATCH_THRESHOLD` stays the module
   constant it is today (`workflow_agent.py:1427`); it is not an env
   setting.
2. **Tests.** Add `tests/test_plan_time_ask_user.py`, covering:
   - trigger A asks, an `answer` naming a candidate runs it, and
     `stop_asking` falls back;
   - trigger B asks for the missing required argument, and the answer
     completes the call;
   - unattended turns never ask, under both triggers;
   - a pending question survives `serialize_state` /
     `apply_serialized_state`;
   - no sub-task is asked twice.

   Both triggers need a real Jev key: trigger A reads Jev's
   probabilities, and trigger B is only reached after `match_skill`
   matched a composite, which returns `None` without a key
   (`_skill_matcher`). When `JEV_API_KEY` is unset, skip both sets of
   tests with a skip reason that says so. The unattended and
   serialisation tests that set `plan_question` directly do not need Jev
   and always run.

**Done when:** the tests pass and the full suite shows no new failures.

### Phase 7: fold `_skills` into `_commands`

1. **The build generator header and refusals** (3.8). Add tests in
   `tests/test_build_protects_composites.py`: an `overwrite=True` build
   over a tree containing a composite leaves the composite byte-identical
   and warns.
2. **Move composites.** The only workflow with `_skills/` is
   `tests/skill_tutorial_workflow`. Move each of its skill files to
   `_commands/` (root files to the root, context folders to the matching
   context folders), and `_`-prefixed helpers with them. Relative imports
   between files that move together keep working (`book_loans.py` has
   `from .book_by_title import PlanGenerator as BookByTitle`); fix any
   import that crosses the old and new trees. No bundled workflow has a
   `_skills/_split.py`, so there is no split file to move.
   - Update every `<!-- file: tests/skill_tutorial_workflow/_skills/... -->`
     marker in `docs/tutorials/skills/*.md` to the new path;
     `tests/test_skill_tutorials.py` reads those markers to compare each
     page's quoted block with the file.
   - Regenerate `docs/tutorials/skills/previews/` (each preview's first
     line names its source file).
3. **`split_settings`** (`workflow_agent.py:1341`) reads
   `_commands/_split.py` first and falls back to `_skills/_split.py`.
   `CommandDirectory` already skips `_`-prefixed files, so
   `_commands/_split.py` is never registered as a command.
4. **Deprecation.** Keep `_skills/` loading with one deprecation warning
   per workflow per process: "_skills/ is deprecated; move these files
   into _commands/". Collisions across the two trees are load errors
   (3.1).
5. **Tests.** Update `tests/test_skill_files.py` and
   `tests/test_skill_tutorials.py` paths in place; do not delete them. Add
   one test that a workflow still using `_skills/` loads with the warning.

**Done when:** every tutorial produces its Phase 0 output from the new
location, and the full suite shows no new failures.

### Phase 8: documentation and coding skills

1. **Update `docs/skill_file_layout.md`, `docs/skill_phases.md`,
   `docs/skill_phases_code_run_steps.md` and every page in
   `docs/tutorials/skills/`:**
   - change `_skills/` to `_commands/`;
   - change `fact_from` to `from_`, noting the alias;
   - describe the composite output contract, nesting, the depth cap, the
     nested step budget, and plan-time ask_user.
2. **Link this document** (`docs/unified_command_architecture.md`) from the
   existing skill docs.
3. **Update the coding skills in
   `fastworkflow/skills_for_coding_fastworkflows/`:**
   - `author-workflow-skills/SKILL.md`: the new layout, `from_`, nesting
     and budgets.
   - `declare-parameter-producers/SKILL.md`: `from_` with field
     references. Rules 2–4 are now checked by `preview --check`; keep them
     as authoring guidance.
   - `design-context-models/SKILL.md`: composites follow context
     visibility.
4. **Update `AGENTS.md`:**
   - add the `fastworkflow command preview` alias;
   - add the new env settings (`FW_TURN_MAX_STEPS`,
     `FW_COMPOSITE_MAX_DEPTH`, `FW_COMPOSITE_NESTING`,
     `FW_SKILL_ASK_MIN_PROBABILITY`, `FW_SKILL_ASK_MAX_GAP`) with their
     defaults.
5. **Close the epic.** Close the remaining bd issues and export them.

**Done when:** the docs build and render, all links resolve, and the full
suite shows no new failures.

### Settings introduced (summary)

| Setting | Default | Meaning |
|---|---|---|
| `FW_TURN_MAX_STEPS` | 200 | Executor model iterations per turn, summed over all composite frames (depth 1 and below); the turn frame is not counted |
| `FW_COMPOSITE_MAX_DEPTH` | 3 | Maximum composite nesting below the turn frame |
| `FW_COMPOSITE_NESTING` | on | Composites appear in the executor's and planner's command catalogs and can be called as steps |
| `FW_SKILL_ASK_MIN_PROBABILITY` | 0.3 | Plan-time trigger A: lower bound for both top candidates |
| `FW_SKILL_ASK_MAX_GAP` | 0.15 | Plan-time trigger A: maximum gap between the top two candidates |

Unchanged settings that keep their meaning: `FW_SKILL_PHASE_MAX_STEPS`
(15, per phase; inside a composite frame also capped by the nested
budget), `FW_SKILL_FANOUT_MAX_ITEMS` (10), and `MAX_HANDOFFS` (2, turn
frame only). Unchanged constants: `SKILL_MATCH_THRESHOLD` (0.5,
`workflow_agent.py`), `MAX_SUBTASKS` (10) and the executor's
`max_iters` (25).
