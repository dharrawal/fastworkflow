# Agent runtime module ownership

**Status:** target design for a behavior-preserving structural refactor  
**Scope:** tool-agent observation storage, paging, evidence readers, command-context navigation,
runtime lifecycle, and observability integration  
**Tracking provenance:** `fix-iq53.1.1`; the design stands without that identifier

## Purpose and constraints

The current implementation distributes ownership across large modules, and some components reach
sideways into their siblings to find shared state or clean it up. This design assigns one owner to
each responsibility before code is moved. It is a module-boundary change, not a change to what the
agent sees or how it decides.

The refactor must preserve:

- the agent's system and tool prompts, including `search_memory` guidance, answer guidance, and any
  existing finish or coverage reminders;
- all context, observation, search-answer, result-page, hot-cache, and iteration budgets, including
  their environment overrides and fallback rules;
- canonical execute aliases, cursor tokens, response text, page headers, error/refusal text, artifact
  shapes, observability fields, and ordering;
- which observations are archived or kept inline, which evidence is selected for retrieval, when a
  model is called, and how missing, partial, failed, or bounded reads are reported;
- suspension and cross-process resumption, execute numbering, scope isolation, failure degradation,
  completion sealing, redaction timing, and retention behavior; live evidence remains raw until the
  existing end-of-turn seal;
- workflow command surfaces and public imports that have actually been released.

This work does not introduce a universal evidence schema, a plan verifier, or a new definition of
answer completeness. Outcome, coverage, and support remain distinct concepts. Existing answer
coverage and finish policies stay unchanged until their separate ablations justify a change.

Selected policy change `fix-iq53.3.5` separately disables answer-coverage instruction injection and
its postcheck by default while retaining finish reminders; `fix-vsxf` is the later cleanup and
remeasurement of that change. The runtime ownership boundaries and all other behavior described
here remain unchanged.

## Current ownership

| Current module | Responsibility it currently carries | Boundary problem |
|---|---|---|
| `observation_offloading.agent` | Builds the concrete ReAct agent, opens the archive, activates tracing enrichment, creates `search_memory`, and binds the step hook | Composition, storage, retrieval, and observability setup meet in one constructor |
| `observation_offloading.compact` | Assigns execute ordinals, annotates and archives observations, decides prompt residency, and compacts trajectories | Correct policy owner, but it reads mutable state and concrete archive types directly |
| `observation_offloading.continuation` | ReAct continuation, scope changes, forced replans, and sealing of the previous scope | Agent algorithm also participates in resource lifecycle |
| `observation_offloading.archive` | Scope model, observation/answer persistence, subjects, navigation entries, sealing, redaction capture, and the unavailable-store fallback | One concrete persistence class is treated as the shared contract; it also calls observability policy code lazily |
| `observation_offloading.state` | Process registries, current-scope discovery, hot observation cache, events, subjects, and sealing | Standalone discovery remains as a released fallback; aggregate cleanup now belongs to the runtime coordinator |
| `observation_offloading.search` | Bounded evidence reads, observation-search model call, answer presentation, and archived answer lookup | Runtime retrieval and reusable evidence reading/presentation are interleaved |
| `observation_offloading.labels` | Canonical observation labels, aliases, escaping, and size estimates | This is already a small shared value/presentation module |
| `observation_offloading.manifest` | Compatibility import for observation-manifest helpers now implemented by observability enrichment | The facade remains for released imports while ownership has moved to observability |
| `result_handles.py` | Result models, resolver registry, concrete store, hot caches, cursor issue/validation, paging, backend continuation, rendering, and events | A single module owns several independently testable layers and imports offloading state for scope, storage, and events |
| `auto_navigation.py` | Workflow declarations, entry contracts, per-scope context-entry registry, persistence, dispatch decisions, clarification, and validation | Navigation policy also locates runtime scope/archive state and owns its own lifecycle cache |
| `answer_rehydration.py`, `answer_coverage.py`, `answer_attribution.py` | Read archived observations, result declarations/pages, and subjects for different consumers | Parallel readers import concrete stores and offloading internals, creating circular dependency pressure |
| `workflow_agent.py` and `WorkflowExecutionContext` | Invoke the agent and expose enough host state for scope creation, suspension, completion, and cleanup | Turn-runtime ownership is implicit and split between the host, agent, and offloading state |
| `observability.*` | Capture policy, execution records, provenance, evidence-run certification, observability storage, and the explicit enrichment registry | Enrichment is activated for process lifetime at the existing agent-construction point; it does not replace tracing functions |
| `run_chatbot.navigation` | Builds a read-only benchmark/experiment/conversation/turn tree from observability readers | Despite its name, this is presentation for recorded runs, not agent command-context navigation |

`RuntimeHandleScope` is the current shared identity model. `SourceDescriptor`, `ResultHandleSpec`,
`Literal`, `ResultPage`, and cursor payloads are the current paging models. Archive lookup functions in
`observation_offloading.state` and `result_handles` are the de facto shared readers. Their concrete
locations should not force higher-level evidence consumers to import runtime policy modules.

## Target ownership

The target uses the existing top-level areas. It does not create a package for every concept.

| Target area | Owns | Does not own |
|---|---|---|
| `agent_runtime` | One turn-runtime object that binds scope, observation archive, result store, navigation registry, event sink, and cleanup callbacks; construction and lifecycle transitions | Compaction rules, paging algorithms, navigation decisions, persistence schemas, prompts, or evidence policy |
| `observation_offloading` | Execute alias/label presentation, archive-before-label behavior, prompt-residency compaction, continuation/replan behavior, and `search_memory` policy | Result paging, workflow-specific backend calls, UI navigation, global trace installation, or sibling cleanup |
| `result_handles` | Generic result models, declaration/page persistence, cursor validation, bounded paging, literal filtering, page rendering, and resolver interfaces | Database/view semantics, credentials, workflow client construction, observation compaction, or lifecycle orchestration |
| `evidence` reader layer | Read-only protocols and shared queries for observations, subjects, declarations, pages, and immutable artifacts, used by rehydration, coverage, and attribution | A universal evidence record, verification policy, answer-completeness decisions, writes, retention, or runtime cleanup |
| `auto_navigation` | Workflow entry declarations/contracts, context-entry values, deterministic dispatch/clarification policy, and build-time validation | Scope discovery, archive selection, result-handle internals, lifecycle orchestration, or observability UI trees |
| `observability` | Capture policy, execution records, provenance, evidence-run certification, and explicit trace enrichment hooks (including observation manifests) | Agent construction, retrieval decisions, or ownership of operational observation/result stores |
| `run_chatbot.navigation` | Read-only UI hierarchy over benchmark, experiment, conversation, and turn readers | Runtime command dispatch; it should keep its qualified import and descriptive module docstring |

`agent_runtime` should start as a small composition/lifecycle module, not a new hierarchy. Add child
modules only where the paging split needs them: models, store/cursors, paging/rendering, and shared
readers are coherent seams. Keep labels, compaction, continuation, and search in the existing
`observation_offloading` package. A generic “evidence framework” is intentionally out of scope.

### Concrete target file layout

The first structural pass should use this layout. The names describe ownership rather than dictating
that every file must become a separately released API:

```text
fastworkflow/
  agent_runtime.py                         # turn composition and lifecycle coordinator
  result_handles/                          # replaces the current result_handles.py implementation
    __init__.py                            # released compatibility imports only
    common.py                              # shared constants, errors, canonical bytes and clock
    models.py                              # SourceDescriptor and ResultHandleSpec
    store.py                               # durable declarations, pages, walks, and cursors
    cursors.py                             # cursor values, issue, decode, and validation
    paging.py                              # declare/fetch algorithm, resolver port, current hot caches
    rendering.py                           # ResultPage, page packing, headers, and outcomes
  evidence_readers.py                     # shared read-only ports and cross-store queries
  observation_offloading/
    agent.py                               # agent/tool wiring using an injected turn runtime
    archive.py                             # observation/answer/subject store implementation
    compact.py                             # aliasing, archive-before-label, residency policy
    continuation.py                       # continuation and replan algorithm
    labels.py                              # observation label and alias presentation
    search.py                              # search policy and model interaction
  auto_navigation.py                      # entry contracts and command-context dispatch policy
  observability/
    enrichment.py                         # explicit registry and observation-manifest implementation
```

`result_handles/__init__.py` preserves `from fastworkflow import result_handles` and re-exports the
documented API while callers migrate away from implementation internals. If splitting `cursors.py` or
`rendering.py` produces files with no independent state or algorithm after extraction, keep that code
in `paging.py`; the ownership boundary matters more than the file count. Likewise,
`evidence_readers.py` remains one module until distinct reader implementations make a package useful.
The existing `observation_offloading.state` responsibilities move either to the observation store
implementation (its own hot cache and writes) or to `agent_runtime.py` (scope binding and coordinated
lifecycle); it does not survive as a second coordinator.

Result walk caches and runtime store selection remain in `paging.py`, with their release coordinated
by `TurnRuntime`. Low-level cursor operations require an explicit scope and store; the released paging
boundary supplies those values and retains the standalone fallback that resolves the current scope and
store. Cursor tokens are released by the cursor component under its own lock. No provider setter or
reverse cursor-to-paging import is part of the runtime path.

### Shared contracts

The shared layer consists of framework-neutral values and reader protocols:

- a turn scope identifies the workflow/channel/conversation/turn/attempt and remains the namespace
  for aliases, pages, navigation entries, and events;
- observation records expose immutable text plus digest, command, ordinal, subject, and seal/capture
  metadata already stored today;
- result declarations and pages expose the existing generic declaration, cursor, and page fields;
- navigation entries describe a recorded context transition without importing dispatch policy;
- readers return explicit missing/unavailable states and never infer absence from a partial read.

Protocols may be implemented by the current sidecar-backed stores, an in-memory implementation, or a
future non-database store. Framework contracts must not expose SQL connections, table names, database
paths, or assume that a result source is a database query.

The framework's backend-continuation contract is a serializable, workflow-defined request plus a
runtime resolver interface. A workflow owns resolver registration, authentication, client lifetime,
source-specific ordering, filters, snapshots/timeslots, and conversion of backend rows into the
generic result-page shape. IDO adapters therefore live in the IDO workflow repository. Current
IDO-derived restrictions in `SourceDescriptor` are not general framework requirements and may be
removed after the workflow adapter is accepted. The IDO integration has not been published, so the
framework must not add a compatibility bridge for its old descriptor or resolver shape.

### Public imports

Keep these supported entry points stable while implementation moves:

- `from fastworkflow import result_handles`, including its documented declaration, paging, cursor,
  resolver-registration, model, error, and reset APIs;
- `fastworkflow.observation_offloading` exports currently used by framework integrations and documented
  operator/evaluation tooling;
- `fastworkflow.auto_navigation` declarations, decision values, validator, and helpers documented in
  `docs/auto_navigation.md`.

New lifecycle objects and shared readers are internal during this refactor. Code inside fastWorkflow
imports their defining modules directly; the package root does not re-export every moved helper.
Compatibility re-exports are added only for released imports, not for private helpers or unpublished
workflow integrations. `run_chatbot.navigation` remains qualified because its dictionaries are a UI
view model, not a general navigation API.

## Dependency direction

Dependencies point from policy and orchestration toward values and ports, and from adapters toward the
framework contracts they implement:

```text
workflow/backend adapters
        |
        v
result_handles models + resolver port <- result paging/rendering
        ^                                  ^
        |                                  |
shared evidence readers <- answer rehydration / coverage / attribution
        ^
        |
observation storage ports <- observation offloading policy
        ^                         ^
        |                         |
        +-------- agent_runtime --+---- auto_navigation registry/policy
                         |
                         v
              explicit observability hooks
```

Lower layers do not discover the active agent, workflow host, or sibling stores. They receive scope,
reader/writer ports, budgets, and event callbacks from `agent_runtime`. `result_handles`, navigation,
and observation storage release only their own caches/resources. They never import one another to
perform coordinated cleanup. Evidence consumers use shared readers and do not import compaction,
continuation, or concrete persistence classes. Observability accepts explicit enrichment callbacks;
it does not import the agent runtime to find them.

The few unavoidable composition edges belong at the top: the workflow host creates the turn runtime;
agent construction consumes it; completion/next-turn/close/eviction ask it to coordinate component
cleanup. Lazy imports used solely to hide a cycle are removed as these ports become explicit.

## Lifecycle

1. **Construct host:** create the workflow session/execution context. No turn scope or archive is
   inferred by a low-level component.
2. **Start or resume turn:** `agent_runtime` creates or restores the exact turn/attempt scope, opens
   component stores, binds event reporting, and supplies the components to the agent. A resumed turn
   keeps its alias numbering, raw live evidence, result cursors, and navigation entries.
3. **Execute step:** the command executor records the context the command ran in. The step hook assigns
   the canonical execute alias, persists the raw observation before any label replacement, then applies
   the existing compaction decision. Result declarations and navigation entries write through their own
   components using the same scope.
4. **Retrieve:** `search_memory`, result paging, rehydration, coverage, and attribution read through
   their appropriate ports. Each keeps its current budgets and decisions; a storage failure or partial
   read retains its current explicit outcome and never becomes evidence of absence.
5. **Suspend:** preserve the runtime binding and durable state needed for resume. Do not seal or redact
   the in-flight turn and do not reset execute ordinals.
6. **Complete or begin the next turn:** the runtime seals the completed scope at the same point as
   today, then asks each component to release only its own hot state. Component failure is reported by
   the existing event/observability paths and does not cause another component to claim cleanup.
7. **Close or evict:** the host asks the runtime owner to release the scope. Retention and erasure remain
   explicit store policy; releasing process resources does not silently delete durable evidence.

## Implementation boundaries and verification

The structural sequence extracts shared models/readers and result paging seams, introduces the
runtime owner, then moves manifest enrichment behind an explicit observability registry. The registry
is activated at the existing agent-construction point for process lifetime; tracing calls it directly,
without function reassignment. Manifest uncapped additions keep their prior behavior.

Focused integration checks must exercise successful completion, transition to the next turn,
session close/eviction, suspension and same-process resume, cross-process resume, storage failure,
scope isolation, result continuation, navigation restoration, and seal timing. Structural equivalence
for the workflow integration should compare prompts, resolved budgets, tool outputs, aliases/cursors,
retrieval choices, stored artifacts/events, and final answers. Those checks establish preservation;
they do not certify a new evidence policy.

## Source evidence in this tree

- `observation_offloading.agent.build_tool_agent` currently selects the sidecar, builds
  `search_memory`, installs span policy, and constructs `StructuredContinuationReAct`.
- `agent_runtime.TurnRuntime` now coordinates finished-scope release across observation state, result
  handles, and auto-navigation. Released standalone cleanup functions delegate to the same coordinator;
  each component's hook releases only its own resources.
- `result_handles.current_scope` and auto-navigation scope helpers discover the active agent and
  offloading archive, demonstrating the inverted dependency.
- `observation_offloading.archive.RuntimeHandleArchive` stores observations, archived search answers,
  subjects, and navigation entries. `result_handles.ResultHandleStore` separately owns result
  declarations, pages, walks, and cursors, even though both stores currently use the same sidecar
  path. `UnavailableHandleArchive` demonstrates that runtime policy already depends on behavior rather
  than successful SQLite access.
- answer rehydration, coverage, and attribution each import archive/label/state internals and
  `result_handles`, demonstrating the need for shared read-only contracts.
- `observation_offloading.manifest.install_span_policy` remains a compatibility import called during
  agent construction. Its implementation activates the explicit registry in
  `observability.enrichment`; tracing functions are not replaced.
- `run_chatbot.navigation` only calls observability-store read methods and builds UI nodes; it is
  separate from `auto_navigation` despite the shared word “navigation.”
