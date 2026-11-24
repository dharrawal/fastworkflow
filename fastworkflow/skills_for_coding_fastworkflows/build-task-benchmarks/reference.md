# Harness reference

A working skeleton for the two-tier benchmark described in `SKILL.md`. Adapt names to the
workflow; the structure is what matters. The examples use the shipped `library` example's commands.

## Placeholders and turns

A turn's parameters are resolved at run time, because most of them are not known when the task is
written: they come from seeding, or from an earlier turn's output.

```python
class Ref:
    """A value produced by an earlier turn of the same conversation."""
    def __init__(self, key): self.key = key
    def resolve(self, state): return state[self.key]


class Seed:
    """A value discovered during seeding, for chains that need a specific shape
    of object (a container that has children, a record that has a parent)."""
    def __init__(self, key): self.key = key
    def resolve(self, state): return state["_seeds"][self.key]


def resolve(value, state):
    if isinstance(value, (Ref, Seed)):
        return value.resolve(state)
    return value(state) if callable(value) else value


class Turn:
    """One thing the user says, and the command that should handle it.

    write          reversible write; needs --writes
    unsafe         real side effect; validated, never sent
    restore        undo turn; still runs when the chain is BLOCKED
    expect_context the context this turn must land in; also drives static planning
    capture        state key -> handle index into the rendered listing, or a callable
    """
    def __init__(self, say, command, params=None, capture=None, write=False,
                 unsafe=False, restore=False, expect_context=None):
        self.say, self.command = say, command
        self.params, self.capture = params or {}, capture or {}
        self.write, self.unsafe, self.restore = write, unsafe, restore
        self.expect_context = expect_context


class Conversation:
    def __init__(self, name, purpose, turns):
        self.name, self.purpose, self.turns = name, purpose, turns


class Task(Conversation):
    """`combines` names the tier-1 conversations this task re-walks. It is what
    makes a failure attributable: each block is green on its own, so a step that
    only fails at length is a length problem."""
    def __init__(self, name, purpose, combines, turns):
        super().__init__(name, purpose, turns)
        self.combines = combines
```

## Executing one turn

A command path is the routing name: `Context/command_name`, or a bare `command_name` for the
global context. An inherited command keeps the name of the context that defines it (a `Book`
runs `Record/add_note`).

Open a real session once, then send every turn through `CommandExecutor.perform_action`, the
runtime's direct-action path. It runs the command on the record the workflow stands on, runs the
`Input`'s field checks and model validators first, and wraps what the command returns in a
`CommandOutput`.

```python
import json, os, uuid

import fastworkflow
from fastworkflow import Action, ModuleType
from fastworkflow.command_executor import CommandExecutor, call_response_generator


def open_session(workflow_folder):
    """A real Workflow, started as `fastworkflow run` starts it (its startup action, if any)."""
    if fastworkflow.RoutingRegistry is None:
        fastworkflow.init({})
    workflow = fastworkflow.Workflow.create(workflow_folder,
                                            workflow_id_str=f"bench-{uuid.uuid4().hex}")
    startup = os.path.join(workflow_folder, "startup_action.json")
    if os.path.isfile(startup):
        with open(startup, encoding="utf-8") as fh:
            CommandExecutor.perform_action(workflow, Action(**json.load(fh)))
    return workflow


def context_now(workflow):
    """Where the workflow stands after a turn: what `what_is_current_context` reports."""
    return workflow._current_command_context_name


def execute(workflow, turn, params):
    """Run one turn the way the runtime does. The framework navigation commands are below."""
    if turn.command.startswith(CORE):
        return execute_core(workflow, turn)
    try:
        return CommandExecutor.perform_action(
            workflow, Action(command_name=turn.command, command=turn.say, parameters=params))
    except ValueError as refused:
        # A missing or refused value: the message is what the user would be asked.
        return fastworkflow.Refused(str(refused)).as_output()
```

An `unsafe` turn is checked and never sent: run only the `Input`'s checks, with the validation
context the runtime gives them.

```python
from fastworkflow.parameters.validate import validate


def validate_only(workflow, turn, params):
    """The refusal the runtime would give this call, or None when it would be accepted."""
    routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    input_model = routing.get_command_class(turn.command, ModuleType.COMMAND_PARAMETERS_CLASS)
    if input_model is None:
        return None
    return validate(input_model, params, command_text=turn.say, workflow=workflow,
                    command_name=turn.command).refusal
```

The workflow is never reset between turns: `context_now` after one turn is where the next runs.
`perform_action` does not climb: a step whose command lives on an enclosing record needs a go-up
turn first, as a planned step. A planning command is refused on this path (it runs only in an
agent turn), so keep planning-command steps for runs through real turns.

## Reading handles the way an agent does

Parse the handle out of the response text, never out of internal state. A command with a
`Signature.Output` responds with that Output as JSON, so read its handle field with `json.loads`.
A text listing that renders a summary line, a header naming the columns, then one row per line
reads like this:

```python
def handles_from(response):
    lines = response.splitlines()
    if len(lines) < 3 or "  " not in lines[1]:
        return []
    return [line.split("  ")[0] for line in lines[2:] if "  " in line]
```

## Verdicts

```python
def verdict_for(output, response):
    if not output.command_response.success:
        return "SOFT"
    if "is not available yet" in response:
        return "STUB"
    if response.startswith("No ") and "found." in response:
        return "EMPTY"
    if handles_from(response) or response.startswith("Entered "):
        return "OK"
    return "OK" if response.strip() else "EMPTY"
```

`SKIP` and `BLOCKED` are assigned by the runner, not by this function: `SKIP` when the run mode
excludes a write or unsafe turn, `BLOCKED` when an earlier turn already failed.

## Framework navigation commands

These live in the framework's metadata-extraction workflow, not the workflow's own command folder,
and they read and move `workflow.context["app_workflow"]`, so `perform_action` on the app workflow
cannot run them. Run the real bodies through a shim so the parent chain being walked is the
framework's, not a reimplementation. Do not overwrite the app workflow's own context dict: it
already holds live session state.

```python
CORE = "IntentDetection/"
RESET = "IntentDetection/reset_context"
GO_UP = "IntentDetection/go_up"
WHAT_CAN_I_DO = "IntentDetection/what_can_i_do"
WHERE_AM_I = "IntentDetection/what_is_current_context"


class _CmeShim:
    def __init__(self, app_workflow):
        self.context = {"app_workflow": app_workflow}
        self.folderpath = fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def execute_core(workflow, turn):
    routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    generator = routing.get_command_class(turn.command, ModuleType.RESPONSE_GENERATION_INFERENCE)()
    return call_response_generator(generator, _CmeShim(workflow), turn.say, None,
                                   command_name=turn.command)
```

## Static context planning

Navigation is declared, so the context each step runs in is knowable without a backend. Needed for
offline routing; doubles as an assertion in live mode. `root` is the root record's context (what
`context_now` says right after `open_session`): a reset lands there, which is `*` only when the
workflow's root is the global context.

```python
def plan(task, root):
    current, stack, planned = root, [], []
    for turn in task.turns:
        runs_in = current
        if turn.command == RESET:
            current, stack = root, []
        elif turn.command == GO_UP:
            current = stack.pop() if stack else root
        elif turn.expect_context:
            stack.append(current)
            current = turn.expect_context
        planned.append({"turn": turn, "runs_in": runs_in,
                        "ancestors": list(reversed(stack))})
    return planned
```

Order matters: test the two navigation verbs **before** `expect_context`, so a go-up turn that also
asserts its destination is planned as a pop rather than a push.

## Routing scorer

Asks each context's trained local model (`CommandRouter`, cached per folder) and walks the scopes
as the runtime does (see `SKILL.md`). Never touches the backend or a remote model.

```python
from fastworkflow.intent_detector import handed_on
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL


class Router:
    def __init__(self, workflow_folder):
        if fastworkflow.RoutingRegistry is None:
            fastworkflow.init({})
        self.folder = workflow_folder
        self.routing = fastworkflow.RoutingRegistry.get_definition(workflow_folder)

    def commands(self, context):
        known = context in self.routing.contexts
        return set(self.routing.get_command_names(context)) if known else set()

    def labels(self, context, utterance, reachable):
        """One label when the model is confident, else its candidates plus up to two commands
        of other reachable scopes it ranked ahead of its own (as the runtime adds them)."""
        folder = "global" if context == "*" else context
        path = os.path.join(self.folder, "___command_info", folder)
        if not os.path.isdir(path):
            return None
        labels, details = CommandRouter(path).predict_with_details(utterance)
        labels = [str(label) for label in labels]
        if len(labels) > 1:
            labels, _ = handed_on(labels, details, reachable)
        return labels


def _one_command(labels):
    return (labels is not None and len(labels) == 1
            and labels[0] not in (WILDCARD_LABEL, PARAMETER_VALUE_LABEL))


def route_turn(router, entry):
    turn, here = entry["turn"], entry["runs_in"]
    want, walk = turn.command, [here, *entry["ancestors"]]
    reachable = set().union(*(router.commands(c) for c in [*walk, "*"]))

    def outcome(labels, at, climbed=False):
        if _one_command(labels):
            hit = "escalated" if climbed else "direct"
            return {"outcome": hit if labels == [want] else "misrouted", "at": at, "labels": labels}
        if labels and len(labels) > 1 and want in labels:
            return {"outcome": "ambiguous", "at": at, "labels": labels}
        return {"outcome": "lost", "at": at, "labels": labels or []}

    # The global scope first: only a confident application command decides there.
    if "*" not in walk:
        first = router.labels("*", turn.say, reachable)
        if _one_command(first) and not first[0].startswith(CORE):
            return outcome(first, "*", climbed=True)
    last, at = None, here
    for index, context in enumerate(walk):
        labels = router.labels(context, turn.say, reachable)
        if labels is None:
            continue  # no model for this scope: it declines
        last, at = labels, context
        if len(labels) == 1 and labels != [WILDCARD_LABEL]:
            return outcome(labels, at, climbed=index > 0)  # a command, or a bare value
        if index == 0 and len(labels) > 1:
            return outcome(labels, at)  # the own scope is unsure: the runtime asks
    # Everything said "not here": the global scope, asked last.
    if (last is None or last == [WILDCARD_LABEL]) and "*" not in walk:
        final = router.labels("*", turn.say, reachable)
        if final is not None:
            return outcome(final, "*", climbed=True)
    return outcome(last, at, climbed=True)
```

A confident answer that names a planning command is asked at runtime when no remote model
confirms it; score it `ambiguous` if your tasks include planning commands.

## Held-out phrasing check

Run this before anything else and return non-zero on a leak. A phrasing that is also a training
seed measures memory, not routing.

```python
def check_heldout(phrasing_pools, seed_utterances):
    seeds = {s.lower() for s in seed_utterances}
    leaked = sorted({t.lower() for pool in phrasing_pools.values()
                     for t in pool if t.lower() in seeds})
    for text in leaked:
        print(f"  LEAKED {text!r}")
    return not leaked
```

## Phrasing rotation

Keep a pool per command and select deterministically, so variety costs nothing in reproducibility.
Fields let one pool serve every variant of a parameterised question.

```python
PHRASINGS = {
    "Book/list_loans": [
        "who has this {kind} out right now",
        "list the current borrowers here",
        "which members are holding this {kind}",
    ],
}

def say(command, variant=0, **fields):
    text = PHRASINGS[command][variant % len(PHRASINGS[command])]
    return text.format(**fields) if fields else text


def step(command, variant=0, phrase_fields=None, **kwargs):
    """A Turn whose wording comes from the rotation rather than the call site."""
    return Turn(say(command, variant, **(phrase_fields or {})), command, **kwargs)
```

## Block functions

Tier-1 errands become the tier-2 vocabulary. Keep them small and parameterise the wording variant
so the same block reads differently in different tasks.

```python
def book_overview(v=0):
    return [
        step("Book/show_book", v),
        step("Book/list_loans", v),
        step("Record/show_notes", v),
    ]


def note_cycle(text, v=0):
    """Add a working note and prove it landed. The library has no command that removes a note,
    so the write is `unsafe` (validated, never sent). With an undo command, mark both turns
    `write=True` and add the undo as a `restore=True` turn."""
    return [
        step("Record/add_note", v, {"text": text}, params={"text": text}, unsafe=True),
        step("Record/show_notes", v + 1, unsafe=True),
    ]


task = Task("book-review", "...", ["book-360", "notes"],
            [step("Library/open_book", 0, params={"book_uid": Seed("book")},
                  expect_context="Book"),
             *book_overview(0),
             *note_cycle(WORKING_NOTE, 0)])
```

## CLI shape

```
--execute    run every step against the live backend
--route      score intent routing per step (no backend, no LLM)
--writes     run the reversible write turns
--unsafe     dispatch real side effects (not net-neutral)
--report     write the markdown task report
--only NAME  run one task
```

Require at least one of `--execute` / `--route`. Emit a machine-readable results file alongside the
markdown so a report or dashboard can be generated without re-running.

## Reported output

The markdown report should carry, per task: purpose, step count, contexts visited, which tier-1
conversations it combines, and a table of every step with its wording, command, the context it ran
in, and one column per scored axis. Include the ids the run actually used so a reader can replay
the task by hand.

The summary should carry, per task: steps, execution verdict counts, per-step clean routing rate,
`p^n`, and the index of the first derailing step.
