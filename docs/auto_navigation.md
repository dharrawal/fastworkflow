# Auto-navigation: deterministic two-step dispatch on a foreign command name

*Framework half of ido-8ps.9. Successor to R1 (ido-8ps.8, `c976964`) and the
routing hint (`74c348a`).*

## What this is for

An agent writes `list_permissions` while the current context is `Identity`.
`list_permissions` is a real command of the workflow — `Account` owns it — and
the parent walk from `Identity` never passes `Account`. Before R1 the `Identity`
classifier answered it anyway, with the directory-wide catalogue, and the answer
read as a true one: 12 such misroutes across 4 runs, silent. R1 refuses; the hint
says where the command lives. The agent then has to compose two steps itself:
open the account, then list its permissions. In the measured run it did so once
in four.

The two steps are mechanical whenever the workflow has said how a context is
entered. This is the mechanism that composes them — and, just as importantly,
the mechanism that refuses to compose them when composing would require a guess.

## The rule

> Every automatic action is a pure function of the **utterance** and the
> **context model**, never of history.

On a known command name owned by a context the current one does not own and the
parent walk cannot reach, the framework composes
`[enter owner context via its declared entry command; run the original command]`
**only** when:

1. **stateless** — the owning context's declared entry command has no required
   parameters; or
2. **parameters in the utterance** — the entry command's required parameters are
   present in the utterance itself, read with the same XML tag grammar the
   parameter extractor uses; or
3. **explicit handle** — the utterance carries an `O` alias or a value that
   resolves, through the turn-scoped registry, to a context instance entered
   earlier in this turn.

Otherwise the turn **blocks** with a clarification naming the owning context,
the entry command and the missing parameter, in the parameter extractor's own
words (`MISSING_INFORMATION_ERRMSG`), so a summary that counts the existing
missing-parameter path counts these too. The clarification may list
candidate values read off the last few execute observations as a convenience;
the framework acts only on what the agent then supplies.

There is no fourth case. In particular there is no "the uid was in the last
observation", no "the plan said to open that account", and no "only one account
has been opened this turn". Those are inferences from the action log, and 51 of
the 86 misroutes ido-8ps.6.1 counted would have been "fixed" by exactly such an
inference — which is the reason it is forbidden rather than the reason to allow
it.

### Why the rule-3 registry is not history

`decide()` receives the registry as an argument and consults it **only** for a
token the utterance itself contains. A handle the agent wrote is part of the
utterance; the registry answers what that handle denotes, the way the command
inventory answers which context owns a name. An entry nothing in the utterance
names is never selected — not because it is recent, not because it is the only
one. `tests/test_auto_navigation.py::TestNeverFromHistory` parameterises the
same utterances over several prior histories (including one whose registry holds
exactly the uid the agent needs) and asserts the decision cannot move.

## Declaration syntax

One canonical source: a **class attribute on the context's own callback class**,
the mechanism `74c348a` introduced.

```python
# _commands/Account/_Account.py
class Context:
    enter_command = "open_account_by_uid <account_uid>"

# _commands/ControlsMonitor/_ControlsMonitor.py
class Context:
    enter_command = "open_controls_monitor"
```

* Only the **leading command name** is load-bearing. The tail is hint text shown
  to the agent; it is never parsed for values. A declaration carrying a real
  value would make navigation a function of the declaration rather than of the
  utterance.
* `enter_commands = [...]` (plural) is accepted for a context with more than one
  entry command. Auto-navigation then **declines**: the framework will not pick
  between them, the hint names them all, and the validator reports it.
* A context with no declaration keeps exactly the `74c348a` behaviour — the hint
  names the owning context and stops.

**A context-model-file form was considered and rejected.** `enter_command` could
have been added to `_commands/context_inheritance_model.json`, but the callback
class already exists for every context that has one, the loader
(`CommandContextModel.get_context_class`) is already the path both the hint and
the dispatcher take, and a second place to write the fact is a first place to
write it plus a drift. The constant naming the attributes,
`auto_navigation.CONTEXT_ENTER_COMMAND_ATTRS`, is the one definition;
`intent_detection.enter_commands_for` reads through
`auto_navigation.declared_entry_commands`.

**Stateless vs parameterised** is not declared. It is *derived* from the entry
command's `Signature.Input` model: a context is stateless when its entry command
has no required fields. One source for that too.

## The validator

```bash
python -m fastworkflow.auto_navigation <workflow_folderpath>
```

Also run automatically (warn-only, printed) at the start of
`fastworkflow train`, and available as
`auto_navigation.validate_entry_contracts(workflow_folderpath)`.

Offline: it reads the routing definition, `context_hierarchy_model.json` and the
commands' parameter models. No model, no backend, no trained artifact.

Per declaring context `C` it checks that

* the declaration parses to a command name (`unparseable_declaration`);
* exactly one entry command is declared (`several_declarations`);
* the command exists in this workflow (`unknown_command`);
* it is owned by a context from which `C` is reachable — the root `*`, `C`
  itself, or an ancestor of `C` (`unreachable_owner`). A workflow with no
  hierarchy file cannot be checked this way and gets a warning
  (`no_context_hierarchy`) rather than a silent pass;

and classifies `C` stateless or parameterised. Sample output:

```
entry contracts for /path/to/ido_workflow
  Account: 'open_account_by_uid <account_uid>' -> parameterised: account_uid
  ControlsMonitor: 'open_controls_monitor' -> stateless
  ERROR Repository: unreachable_owner: 'open_repo' is owned by Repository; none
        of those is '*', 'Repository' itself, or an ancestor of it (Directory)
```

The validator cannot prove a declared command *transitions into* the context —
that is a runtime fact, visible as `context_before` / `context_after` on the
command's `fw.command.execute` span, and checked by the dispatcher at the moment
it matters (see below). What it proves is that the command exists and could be
run from somewhere that reaches the target.

Run against the live `ido_workflow` it reports eleven contracts and no errors:
four stateless (`ControlsMonitor`, `DirectoryExplorer`, `ReconciliationWorkspace`,
and `ControlFinding` — see the caveat below) and seven parameterised on a
`*_uid`.

## There is no feature flag: the declaration is the switch

`ido-pyw.1` removed `FW_AUTO_NAVIGATION`, `FW_AUTO_NAVIGATION_CANDIDATE_STEPS`
and `FW_AUTO_NAVIGATION_CANDIDATE_MAX`. Dispatch is unconditional, and it is
still opt-in per workflow, because the thing it acts on is a declaration:

* a workflow whose context callback classes declare `enter_command` /
  `enter_commands` gets two-step dispatch and the blocking clarification;
* a workflow that declares nothing gets exactly the declaration-and-hint
  behaviour the flag used to pin — `decide` finds no `EntryContract` for the
  owning context and returns `reason="no_entry_declaration"`, so the R1 hint is
  what the agent is told, byte for byte.

Nothing here is ever inferred from a command's name, and nothing is read from
history, so "no declaration" is a complete answer rather than a degraded one.

The clarification's convenience list is bounded by two module constants,
`auto_navigation.CANDIDATE_STEPS` (3 execute observations scanned) and
`auto_navigation.CANDIDATE_MAX` (10 values listed). They are presentation
bounds, not policy: candidates are listed and never chosen, so they are never
part of the decision.

## What a dispatch looks like

Both steps run through `CommandExecutor.invoke_command`, the ordinary command
path. Each therefore gets its own `fw.command.execute` span, its own execution
record, its own `context_before`/`context_after`, and its own response text; the
A1/A2 conventions hold because nothing about the step path is special. The
agent's observation for the tool call it made carries both, labelled:

```
[auto-navigation rule 2] 'list_permissions' is owned by the Account context;
entered it with 'open_account_by_uid <account_uid>3f2a</account_uid>'.
Account 3f2a: Alan Cooper

O7
2 permissions.
...
```

**If the entry step fails, or succeeds without moving the context, the dispatch
stops** and the original command is not run: running it would be running it in
the context that had already declined it.

That second case is real. IDO declares
`ControlFinding.enter_command = "open_finding <finding_uid>"`, but
`open_finding`'s `finding_uid` has a default (`Omit to list them instead`), so
the validator classifies `ControlFinding` **stateless** and rule 1 fires with a
bare `open_finding` — which lists findings and enters nothing. The offline
validator cannot see this: whether a command *transitions* is a runtime fact.
The runtime check turns it into a legible refusal instead of a confusing second
failure. A workflow that wants such a context auto-entered needs an entry command
whose identifier is required.

**Inner steps never dispatch again.** A foreign name declined inside one of the
two composed steps gets the hint, not a second dispatch — the rule composes two
steps, not a search (`auto_navigation.dispatching`).

## Spans and events

On each of the two `fw.command.execute` spans (contract `fw.command.execute` v2;
absent on a step the agent typed, so a reader counts auto-navigated executes by
the presence of the key rather than by a value):

| attribute | value |
|---|---|
| `auto_navigated` | `true` |
| `auto_navigation_rule` | `1` \| `2` \| `3` |
| `entered_context` | the owning context's name |
| `auto_navigation_step` | `entry` \| `original` |

On the `fw.nlu.intent` span of the declining prediction (contract v3 —
`auto_navigation_enabled` left it in `ido-pyw.1` with the flag it recorded):

| attribute | value |
|---|---|
| `matcher_layer` | `known_name_foreign_context` (R1, unchanged) |
| `known_name_owner_contexts` | contexts owning the name (R1, unchanged) |
| `known_name_foreign_context_hint` | the hint text (R1, unchanged) |

And one `auto_navigation` event per declined name, through the offloading event
log (`FW_OFFLOAD_EVENTS`), carrying the decision kind
(`dispatch`/`clarify`/`none`), the reason, the rule, the entered context, the
entry command, the missing parameters and the resolved handle. The reasons are a
closed vocabulary (`auto_navigation.REASON_*`) so a summary counts them without
parsing prose.

## What is unaffected

* **Root (`*`) commands.** `fetch_result_page` is owned by `*`, which every walk
  reaches, so the prediction resolves before the post-walk block the dispatcher
  lives in is ever entered. ido-8ps.6.1's own case is a walk, now as before.
* **Names the workflow does not own.** No owners, nothing to navigate to; fuzzy,
  cache and classifier run exactly as they did.
* **Names owned by several contexts.** The framework has no ground to prefer one
  and does not choose; the R1 hint names them all.
* **The classifier, the thresholds and the trained artifacts.** Nothing here
  retrains anything or moves a threshold. R1 bypasses the classifier; this
  decides what to do after it has been bypassed.

## Where the code is

| file | what |
|---|---|
| `fastworkflow/auto_navigation.py` | the rule: declaration, contracts, registry, `decide`, candidates, clarification text, validator |
| `fastworkflow/_workflows/command_metadata_extraction/intent_detection.py` | R1's guard: the declined known name and the hint |
| `fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py` | decides after the walk fails; leaves a plan or a clarification |
| `fastworkflow/command_executor.py` | runs the plan's two steps; records context entries for rule 3 |
| `fastworkflow/tracing.py` | the two span contracts |
| `tests/test_auto_navigation.py` | the rules, the property, the clarification, the validator, both seams |
