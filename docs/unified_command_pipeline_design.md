# Unified command pipeline: one detector contract, two engines, shared clarification loops

Design for epic `fix-yu3l` (task `fix-yu3l.1`). Status: accepted by the product owner 2026-11-16;
each step below lands only when its pass condition holds. Section 9 records where the build departed
from this text and why.

## 1. Why

Execution commands and planning commands are both "commands", and both already share the parameter
extractor. Everything around it is done twice:

| Concern | Execution command | Planning command |
|---|---|---|
| Which command is meant | Trained head per context, asked scope by scope by a loop in the CME `wildcard` command | Trained head (planning names are labels) **and** a model matcher with its own candidate list, its own scope walk and its own thresholds, combined by hand in `review_proposals` |
| The command is ambiguous | Ambiguity stage: pick one, none of these, abort | "Other plans" inside the plan review |
| Nothing matches | Misunderstanding stage | No-agent listing with a pick (`available_plans`, `use_plan`) |
| A required value is missing | Parameter stage: every missing value named, values kept, reply merged, abort | Refusal question: the turn is re-run from its request with the reply appended; a separate progress counter; a separate strict reply reader |
| Storage | One model pair per context, about 276 MB each | (same heads) |

Consequences seen in practice: a planning command typed by its exact name still needs the model
matcher to bind on the main path; a reply to a missing-argument question re-runs the whole turn and
loses the part position; an untrained workflow cannot run at all even though a matcher exists; a
five-context workflow stores 1.4 GB of models per version.

## 2. The detector contract

One module, `fastworkflow/intent_detector.py`. It owns three things and nothing else.

### 2.1 The scope decision

> Given one scope (a context name) and an utterance, decide among that scope's candidates, or say
> "not mine".

```
ScopeAnswer
  kind        match | ambiguous | not_mine | bare_value
  labels      the one command (match) or the candidates, best first (ambiguous)
  scores      {label: score in 0..1} for every label the engine scored
  confident   True only for a match the engine stands behind without help
  engine      "classifier" | "matcher"
  fallback    True when this engine answered because the preferred one was not available
  details     engine facts for the span (tier, thresholds, artifact version, model name, latency)
```

`not_mine` is the escalation signal (today's reserved `wildcard` label); `bare_value` is today's
`parameter_value` label and ends the walk. An engine implements one method,
`decide(utterance) -> ScopeAnswer`, and declares the candidates it can decide among.

Candidates come from the contract, never from an engine: `candidates(workflow_folder, context)`
returns each command the scope offers (execution, planning and core) with its kind, description and
example utterances, read from the routing definition and the planning command map. The trained
head's label set and the matcher's option list are both checked against it.

### 2.2 The walk

`scopes_for(workflow)` returns the lookup path once, in code, for every engine: the current
context, then each enclosing record's context up to the root object, each with the record a command
found there runs on. Two rules sit on top, exactly as today:

* **Confident global first.** For natural language outside the global scope, the global scope is
  asked first and wins only on a confident match of an application command.
* **Nearest scope wins.** Scopes are asked nearest first; `match`, `ambiguous` and `bare_value`
  stop the walk, `not_mine` moves on. Past the root the result is `none`.

`detect(workflow, utterance) -> Detection` runs the walk and returns one of three results:

```
Detection
  outcome     match | ambiguous | none
  command     qualified name (match)
  candidates  qualified names, best first (ambiguous; empty for none)
  scope       the context and record the command runs on
  answers     every ScopeAnswer of the walk, in order (what the span records)
```

The typed-name layers (exact first token, a known name owned elsewhere, the fuzzy pre-match) are layers of
`intent_loop.resolve` (see the S7 bullet in section 9), which both the plan path and `CommandNamePrediction` call;
`CommandNamePrediction` is the span around it. `run_listing.head_answers` (the planning draft's reading of the heads) asks the same
engines. The `wildcard` command no longer contains a loop.

### 2.3 Engine selection

Per scope, by availability, decided once per process and cached:

1. **classifier**: the scope has trained artifacts (the per-context head today; the workflow model
   of section 6 when it is adopted).
2. **matcher**: no artifacts for the scope, and `JEV_API_KEY` is set.
3. neither: the scope cannot decide natural language; exact names still work. This is today's
   "workflow is not trained" condition.

`INTENT_DETECTION_ENGINE` (workflow env file; default `auto`) pins the choice: `classifier` makes a
missing head a startup problem (today's behaviour, and what a production deployment should set),
`matcher` uses the matcher everywhere (no training run needed), `auto` is the list above.
`FW_REQUIRE_HEADS=1` keeps its meaning for tests and is honoured at run time as `classifier`.

The fallback is never silent:

* at startup (`fastworkflow run`, the server, through `command_preview.load_problems`) one line names
  every scope that runs on the matcher and why;
* each `fw.nlu.intent` span carries `engine` and `engine_fallback`, and the turn record carries the
  set of engines its detections used.

### 2.4 Arbitration: trained execution commands, untrained planning commands

Scores of the two engines are not comparable and are never compared. Production heads label the
planning commands in scope, so a trained scope needs no arbitration. The remaining case is a
planning command the head does not know (added after the last training run). It is resolved by
order, as today: the matcher is asked about the untrained candidates only; a confident matcher
answer (0.5) wins; otherwise the head's answer stands. `fastworkflow train` reports such commands,
and the startup line names them.

The matcher depends on `JEV_API_KEY`. Without it there is no fallback engine, the planning match is
off as today, and a workflow with untrained scopes reports a startup problem.

## 3. The model matcher as an engine

One model call per detection, not one per scope: the prompt lists the candidates of every scope on
the path **grouped by scope**, and the answer is a choice among all of them plus `none`, with
probabilities. Code then applies the walk to the scores: per scope, the best candidate at or above
0.5 is a `match`; no candidate above 0.2 is `not_mine`; anything between is `ambiguous` with the
candidates above 0.2 (at most 3). A `none` at or above 0.5 is `not_mine` everywhere.

The existing planning-command match keeps its prompt and thresholds (0.5 match, 0.9 veto, 0.2
unsure) so that recorded behaviour and cached answers do not move; it becomes this engine restricted
to planning candidates, taking its candidate list from 2.1.

Before the matcher may route execution commands it is scored against the classifier on the same
held-out intent benchmark, including a scope with many commands, with latency recorded (section 8).

## 4. One parameter clarification loop

The loop is the existing parameter stage, lifted out of the CME command so that a caller that
already knows the command can enter it:

```
ParameterClarification  (fastworkflow/parameter_clarification.py)
  open(command, values, missing, invalid, request_text) -> question, parameter_request
  read(reply) -> answer | accept | abort | misunderstood | wait | different_command
  merge(reply) -> values kept, the reply's values added, gaps recomputed
  state        command, values, missing, invalid, asks, stalled   (serialised with the session)
```

* The question names **every** missing and invalid value and shows the values kept
  (`parameter_request`: `command`, `missing`, `invalid`, `values`, and for a planning command
  `description`, `parameters`, `confirm`).
* A reply is read strictly and in code first: abort and its synonyms, `yes` when nothing is missing,
  `name: value` lines, a bare value when one value is open, comma-separated values when their count
  matches. Only then the extractor reads free text. The optional reply-intent check (wait, a
  different request) applies where it applies today.
* Progress rule: the loop goes on while each reply closes at least one open value; a reply that
  closes none is asked once more, then the loop ends naming exactly what is missing. The CME stage
  for a human session keeps its unbounded form (it has `abort`); a turn part uses the bounded form.

Where it is entered:

| Caller | Today | With the loop |
|---|---|---|
| An execution command, or a planning command typed by name (CME stage) | The stage | The stage, on this object. No behaviour change. |
| A turn part bound to a planning command (no agent model) | Refusal question; turn re-run from the request | The part suspends **in place**; the reply resumes that part, its phases start, and the later parts follow. Nothing is re-run. |
| A plan the client pinned (`use_plan`) | Parameter form, its own reader | The same loop with `confirm` (every parameter shown, `yes` runs). |
| A plan step missing one plain value (no agent model) | Decision step with its own reader | The same loop with one open value; the step's question text is kept. |
| A typed planning call refused, with an agent model | Asked once, turn re-run | Asked once (policy kept), through the loop; the part resumes in place. |

Rules that follow from the turn structure:

* **A value an earlier part produces is never asked.** It is bound as "from part N" and is not in
  the open set; if that part did not produce it, the part ends as today.
* **Clarify first, then review complete arguments**, where a review is shown at all (no agent
  model): parts whose open values cannot come from an earlier part are asked in order before the
  review opens, so the review never shows "a required value is missing". With an agent model the
  order is unchanged (review, then at most one question), because the agent can still find a value.
* **A picked plan needs parameters only.** No task text is shown or edited.
* The state replaces `refusal_progress`, `skill_refusal_pending`, `pending_refusal_request` and the
  decision record's value branch in the serialised session; old blobs are read once and upgraded.

Parameter extraction itself is already one function for both kinds
(`parameter_extraction.extract_arguments`). The typed planning call's own extraction and the draft's
extraction both go through it.

## 5. One pair of intent clarification stages

An ambiguous detection and a `none` detection feed the existing stages whatever the kind of command:

* **Ambiguous** (`candidates` non-empty): the ambiguity stage. Pick one, "none of these", abort.
  Planning alternatives stop being a section of the plan review. With an agent model nobody is asked
  for an execution command (the agent chooses, as today); for a planning command in a reviewed turn
  the candidates are data on the review item (`alternatives`) and the pick is read by the same
  stage reader.
* **None**: the misunderstanding stage, whose listing is the scope's commands **and** plans. With no
  agent model this replaces the code-written "no matching plan" listing; the `available_plans` data
  on the `fw_not_run` entry stays, and so does `use_plan` as the way an agent client chooses one.
* A picked plan goes straight to section 4.

What is retired: `no_agent_mode.read_parameter_reply`, `ad_hoc_explanation`'s own listing text,
`WorkflowExecutionContext.refusal_question_for`, `_ask_about_refusal`, `_ask_plan_parameters`,
`_plan_question`, `_rerun_after_planning_command_refusal`, the alternatives section and its switch
reader in the review. What clients rely on and keep: `fw_not_run`, `available_plans`, `use_plan`,
`question_kind` values `planning_parameter` and `phase_step`, the `parameter_request` shape.

## 6. One intent model for the workflow (experiment, `fix-yu3l.8`)

Hypothesis: one classifier over the workflow's whole label set, with its output **masked** to the
scope's candidates, matches the per-context heads.

* **Labels**: every command once (qualified name), plus `parameter_value` and one workflow-wide
  `wildcard` for the built-in vague replies. A command inherited by several contexts is one label
  with one set of rows.
* **Decision in a scope**: softmax over all labels; `p_in` is the best label inside the scope's
  mask, `p_out` the best outside it. `not_mine` when `p_out` beats `p_in` by the calibrated margin
  (the utterance belongs to a command elsewhere, which is what escalation means); otherwise the
  masked, renormalised scores decide `match` or `ambiguous` against thresholds calibrated per scope
  and stored in one file. A tie between a local and a foreign command of the same wording goes to
  the local one: nearest scope wins.
* **Why masked and not context-as-input first**: escalation needs no negatives (they do not
  multiply), a context made of inherited commands needs no training, and there is nothing
  per-context to learn. A second arm feeds the context name as a paired input to test whether it
  adds anything.
* **Tiers**: arm A keeps tiny-then-large; arm B is the large model alone.
* **Artifacts**: `___command_info/versions/<id>/workflow/` holds one model pair, one label encoder,
  one `scopes.json` (mask and thresholds per scope). About 290 MB for the workflow whatever the
  number of contexts.
* **Retraining**: any change retrains the one model (minutes); utterance caches make the data free.
  Publication, the recovery point and the training report are unchanged.

Pre-registration, with predicted numbers and pass conditions, is in
`docs/single_intent_model_preregistration.md` before any run. Swap-in happens only if every pass
condition holds; escalation recall is a condition of its own.

## 7. What changes for a user, by mode

With an agent model (`LLM_AGENT` set), only these:

1. A planning command typed by its exact name is bound in code; the matcher is not asked.
2. The reply to the one question about a refused planning call resumes the part; the turn is not
   re-run from the top.
3. An untrained scope runs on the matcher (when `JEV_API_KEY` is set) and says so, where it used to
   refuse to start. `INTENT_DETECTION_ENGINE=classifier` restores the refusal.
4. Review: `yes, and <request>` is not a plain accept (`fix-yu3l.6`).

With no agent model, additionally: sections 4 and 5 as written.

## 8. Measurement and pass conditions

A held-out benchmark is written first for the bundled `library` example (five contexts, execution
and planning commands, an inheritance edge and nested records) and `retail_workflow` (one large
scope): hand-written phrasings, disjoint from the seeds, routing and escalation cases, checked by
`preflight_benchmark`. Scored with the existing scorer; paired comparisons by exact McNemar.

| Step | Pass condition |
|---|---|
| `.2` contract over the heads | Benchmark verdicts identical case by case; routing, wildcard, intent and listing test files unchanged and green |
| `.3` matcher engine | Planning-match tests unchanged and green (same prompt, same thresholds). Report: matcher against classifier on routing top-1, in-list, escalation recall, and latency per detection. It may be the fallback at any score (the alternative is no routing at all); it becomes selectable by default only if not worse than the classifier at p < 0.05 |
| `.4` parameter loop | The behaviours of the `fix-1ma8.7` tests hold (every missing value named, values kept, progress rule, cancel); a reply resumes the part with no command run twice; session blobs round-trip |
| `.5` intent stages | `fw_not_run`, `available_plans` and `use_plan` contracts hold; endpoint and browser tests green |
| `.6` accept plus request | The listed replies are not plain accepts; existing accept cues unchanged |
| `.8` single model | See the pre-registration |
| `.7` reply that carries a request | Own measurement; last |

No full suite run at any point: the files a step touches and the regression set named in its issue.

## 9. As built: departures from the text above

* **Engine choice is read on every call, not cached per process** (2.3): heads appear and disappear
  during training and in tests, and the check is two file lookups.
* **A fourth setting, `INTENT_DETECTION_ENGINE=hybrid`** (2.3): the head decides when it is
  confident; otherwise the matcher is asked over the scope path and its answer stands (the
  tie-break among the head's three candidates is not called). Added after the measurement in 8:
  the heads are right when confident and unsure on more than half of requests, and the matcher is
  far better exactly there. Measured on the first benchmark, pooled: strict top-1 42% (heads),
  93% (matcher), 93% (hybrid); escalation recall 9%, 100%, 91%; a model call on 0%, 100%, 59% of
  requests. The default stays `auto`.
* **Gathering stays classifier only** (2.2): `run_listing.head_answers` asks every reachable
  context, so it never calls the matcher; a scope without a head offers no candidates there.
* **The matcher is not offered framework core commands** (3): they are matched by the text layers.
* **An extra `not_mine` rule in the matcher** (3): a scope also declines when a command of another
  scope on the path is at or above 0.5.
* **The engines of a turn are on its intent spans, not on the turn record** (2.3).
* **The CME parameter stage is on the shared loop** (4; bd `fix-o1lg` S6 and S7, which replace the earlier departure
  "the CME parameter stage keeps its own state"). See the two bullets after the pending-resolution one. Comma-separated
  replies are still not read by position in a plan part's loop (a hesitant sentence with a comma would be taken as two
  values); the CME's own merge keeps its positional rule.
* **Parts are resolved before the review, in both modes** (4, 5; bd `fix-o1lg.2`; this replaces the
  two earlier departures, "the question stays after the review" and "planning alternatives stay in
  the plan review"). Per turn, in `fastworkflow/_workflows/plan_metadata_extraction/resolution.py`: every part
  is resolved in order, first which command or plan (asked only when `part_intent` is ambiguous,
  the candidates being the choices, execution and planning commands alike, or, with no agent model,
  when nothing serves the part and the turn can ask), then the values a bound planning command
  lacks (asked only when it lacks one; a value an earlier part produces is never asked for; with an
  agent model one ask, and only where the part's inputs were drafted). Each question can be
  cancelled, which drops that part; the later parts go on, and `from sub-task N` references are
  renumbered. Then one review of the whole turn with nothing missing in it, then the run. No review
  for a single part that names a command, or that the local model is confident of for an execution
  command, or whose command the user chose by answering; a pinned plan's parameter question
  (`use_plan`) is that turn's review; none when nobody can answer. The run asks no further intent
  or parameter questions except a decision step inside a plan, and a value it refuses is not asked
  about again (a refused typed planning call of a single unreviewed part keeps its one question).
  Plan alternatives are no longer offered inside the review. The pending state is one JSON-safe
  record, `resolution.PlanResolution`, saved with the session under the key `plan_resolution`; it
  replaces `plan_choice` and `parameter_loop`, and a session saved with those keys is not
  supported. The artifact shapes the clients read are unchanged (`question_kind` `plan_choice` and
  `planning_parameter`, `artifacts.plan_choice`, `parameter_request`, `fw_not_run`, `use_plan`). In
  no-agent mode a part cancelled or left unanswered (`none`, two unreadable replies, two replies
  that give no value) is dropped and comes back as a `fw_not_run` entry with `request`, and
  `available_plans` for an ad hoc part.
* **Two rules of `part_intent` added with it.** (a) A local answer that names a planning command is
  not a decision on its own: the remote model is asked and its answer stands, or, with no key or a
  failed call, the part is an ambiguity (asked); it is never the unreviewed fast path, unless the
  part names the plan exactly. A keyword could carry the local model to a multi-step plan the
  request did not ask for. (b) Commands that only enter a context (`@descends_to` with no
  `when_parameter` and no required input; `context_navigation.context_entry_commands`) are not
  offered to the remote model, so a request that names a workspace by its subject is not sent to
  the command that enters it. On the library plan benchmark (137 parts) the plan parts decided by
  the unreviewed local fast path went from 52 to 0 in the plan group and from 26 to 6 in the
  two-part group, with the remote model agreeing with every local plan pick.
* **The single workflow model** (6): see the three pre-registrations; the masked reading with the
  large model alone is the candidate, and it is opt-in until repeated on a large workflow. As
  shipped (`fix-yu3l.8.5`, `.8.6`): `INTENT_MODEL_LAYOUT=workflow` in the workflow env file, read
  by `fastworkflow train` only (`train/intent_model_layout.py`); the trainer fits the large model
  alone (no tiny model is trained or saved), so `versions/<id>/workflow/` holds `largemodel.pth/`,
  `label_encoder.pkl` and `scopes.json`, and each context folder holds a sentinel
  `threshold.json` and a pointer, `workflow_scope.json`. `CommandRouter` reads the layout from
  the context folder. Any change retrains the one model; a change of layout is a full retrain.
* **The engine setting is gone; hybrid is the only behaviour** (2.3; `fix-o5wi.2`, owner direction
  2026-11-17). `INTENT_DETECTION_ENGINE` and its four modes were removed. `engine_choice` names the
  links a scope has (`chain`: a trained local model and the planner key; `classifier`: a local
  model and no key; `matcher`: no local model and the key; None), and `ChainEngine` is what
  `HybridEngine` was. The local model decides when it is confident; otherwise the matcher is asked
  over the scope path and decides when it is confident; otherwise the ambiguity question, or, when
  every scope declined, the "did not understand" question. Two additions to the hybrid rule: a
  router a test supplies (or one in the router cache) is the scope's local model and an unsure
  answer from it also goes to the matcher; and when the matcher's call fails, the local model's
  answer stands (its candidates go to the question, its "not here" to the walk) where hybrid
  declined. The Jev tie-break among the classifier's candidates can no longer run (with the key
  the matcher is asked; without it there is no Jev) and was removed with its setting
  (`FW_JEV_INTENT_TIEBREAK`). `FW_REQUIRE_HEADS=1` keeps meaning that a missing local model is a
  failure. `startup_report` always returns one line that says what the chain is. The
  `fw.nlu.intent` span is v8: `engine` is `classifier`, `matcher` or `question`; `engine_chain`
  lists the links asked; `engine_mode` and `jev_tiebreak` are no longer written. Section 7's item
  3 now reads: an untrained scope runs on the matcher when the key is set, and nothing but
  `FW_REQUIRE_HEADS=1` restores the refusal.
* **The default local model is an embedding head, not a fine-tuned classifier** (6;
  `docs/single_intent_model_preregistration_v4.md`, `fix-o5wi.2`). `INTENT_MODEL_LAYOUT` has a
  third value, `embedding`, and it is the default for every workflow but the framework's own
  (`intent_model_layout.DEFAULT_LAYOUT`): a frozen sentence encoder (`INTENT_EMBEDDING_MODEL`,
  default `BAAI/bge-small-en-v1.5`) and one logistic-regression head over the rows of section 6,
  read through the same masks without renormalising (`train/embedding_head.py`). `workflow/`
  holds `head.npz` and `scopes.json`; the context folders are those of the `workflow` layout, and
  the pointer's `layout` field says which model it points to. Departures from the pre-registered rule, both in calibration: (a) the
  threshold and margin are set per workflow head from that workflow's cross-fitted rows, where
  the registered run pooled the rows of both example workflows into one pair of numbers (0.53 and
  0.36); (b) the threshold is never below 0.5, the trainer's floor for a single label, because
  the retail example alone calibrates to 0.208 and then answers 22.8% of its benchmark requests
  with one wrong command (8.7% at 0.5). As built and scored through the product trainer and
  router on the third benchmark (264 routing, 44 escalation cases; `scripts/experiments/
  embedding_head.py product`, 2026-11-17): strict top-1 64.4%, in-list 86.7%, wrong lone 8.7%,
  false escalation 2.7%, escalation 42 of 44 (library threshold 0.529 and margin 0.443; retail
  0.500 and 0). The registered arithmetic, reproduced exactly by the same script's `pooled` mode:
  63.6%, 86.7%, 8.7%, 2.7%, 43 of 44 at 0.529 and 0.360. The two extra right answers as built are
  retail requests between 0.5 and 0.529; the one escalation fewer comes from the per-head margin
  (0.443 against 0.360). Operational, CPU, same run: the head 2.4 to 3.0 s per workflow, 49,857
  and 35,463 bytes, a request a median 9.9 to 11.6 ms and 17.4 to 19.7 ms at the 95th percentile.
  Not yet measured on a large workflow, and the chain has not been measured with this head as its
  first link.
* **The similarity cache is gone** (`fix-o5wi.8`, owner direction 2026-11-17). The ladder is exact
  name, fuzzy name, local model, remote model, question. The cache answered a later request from
  a stored clarified one at cosine 0.85; measured in `docs/learning_from_corrections.md` it added
  nothing on unseen requests and 21 to 58 percent of its answers there were wrong. Removed with
  `fastworkflow/cache_matching.py`, `UtteranceCacheStore`, the writes after a clarification and the
  span attributes `cache_similarity` and `cache_similarity_threshold` (span contract v51).
  Learning from corrections will return as recorded rows plus a background refit of the local head
  (`fix-o5wi.6`). A leftover `___convo_info/cache.sqlite3` is never read for the cache again.
* **The local model never passes a request up on its own, and its threshold is fixed** (6, 2.1; `fix-o5wi.9`, owner
  direction 2026-11-17, after `docs/local_intent_model_levers.md`). The embedding head no longer answers a lone
  `not_mine` because a command outside the scope's mask is ahead (the calibrated pass-up margin and its target are removed;
  the `wildcard` class of the built-in vague replies and the `parameter_value` label keep their meaning): it answers
  `ambiguous`, and `details["outside_labels"]` / `["outside_scores"]` name the best commands outside the mask. The detector
  contract is unchanged. `ClassifierEngine(router, reachable=...)` adds up to two of them that the walk can reach
  (`reachable_labels`: the commands of the scopes of the lookup path, the global scope included when asked first) to the
  scope's own candidates, so `Detection.candidates` and the ambiguity question carry the ancestor's command, and the matcher
  (when `JEV_API_KEY` is set) is asked exactly as before. In the clarification stage a scope declines a chosen command it
  does not have, so the walk reaches the scope that has it (the global scope is asked first in that stage) and the command
  runs on that scope's record (`fw.nlu.intent` v10, `chosen_command_elsewhere`). The commit threshold is a constant,
  `INTENT_COMMIT_THRESHOLD` (default 0.65), because the cross-fit calibration did not hit its target on real requests (a
  0.5% target gave 5 to 8% wrong); `calibrated_commit_threshold` is the one place a value calibrated on recorded requests
  will be read (`fix-o5wi.6`). The default encoder is `intfloat/e5-base-v2` (mean pooling, prefix `query: `), the rows add each
  command's name and description, seeds weigh three, and the training report lists the head's cross-fit accuracy and its most
  confused pairs. Measured through the product trainer, library and retail, development benchmarks 1 to 3 pooled: 59.6% right
  outright, 1.8% wrong confident, ancestor among the candidates for 117 of 120 escalation requests; fourth benchmark, once:
  45.8% and 3.0% (the previous head 60.2% and 10.2%), 45 of 46.
* **S6, one extractor contract (bd `fix-o1lg` S6, done).** `parameter_clarification.Extraction` is the one result of "the
  values of a command given some text": `values` (the ones the checks stand behind), `missing`, `invalid` (name and reason),
  `earlier` (a part of the turn produces it) and `reason` (a refusal that names no parameter, or the refusal an extractor
  words). `state_of(name)` gives `given`, `missing`, `invalid` (with why) or `from_earlier`; `from_earlier(names)` replaces
  `without`; `request(command)` is the one builder of the `parameter_request` artifact. Two producers, two mechanisms kept:
  the plan side (`planning_commands.argument_gaps` and `refusal_gaps`, which replace `ArgumentGaps`) and the execution side
  (`parameter_extraction.extraction_of`, which replaces `parameter_request_for`, over the diagnostics `validate_parameters`
  fills). `extract_arguments` returns `ExtractedArguments(params, extraction, retried)`; `valid`, `refusal` and `missing` are
  read from the structure, and the draft (`review_proposals.draft_inputs`) reads the structure. The state "must be found by
  the plan" is not a state of the structure: no producer knows it (the draft knows which fields an earlier part produces,
  not which the plan's own steps find, and showing the second would change the review), so it is not carried.
* **S7, the CME parameter stage on the loop (bd `fix-o1lg` S7, parameters done; intent not done).** The pending question
  of an execution command is one `ParameterLoop` (kind `execution`) under the CME context key `parameter_loop`
  (`CME_LOOP_KEY`): the values kept, the open names, the question as shown (`parameter_error_text`), the
  `parameter_request`, and `asks`, which is the extraction round. `parameter_clarification.record_round` is the one progress
  rule the stage uses; the stage's policy is `unbounded` (human session; the user ends it with `abort`) or `once` (agent
  session; the executor aborts the stage and the agent replans at the first question), so no count of replies without
  progress ends it. Deleted: the context keys `stored_parameters`, `stored_parameter_round` and `missing_value_question`;
  `ParameterExtraction._get_stored_parameters`, `_get_stored_round`, `_store_parameters`, `_clear_parameters`,
  `_extract_missing_fields` (an error-text parser the structure makes redundant) and `_merge_parameters`;
  `parameter_request_for`, `missing_value_question`, `pending_missing_fields`; the model dump and `model_construct`
  rebuild in `WorkflowExecutionContext._serialize_cme_continuation` / `_apply_cme_continuation` (the loop is plain data and
  saved whole; the extractor rebuilds the command's `Input` with `model_of`, and `park` builds the loop from an instance
  for the plan review's slot question). What stays different on purpose: the question text is the command's own
  validation message; the reply merge is `_extract_and_merge_missing_parameters` (the CME's positional rule) after
  `read_reply`'s `name: value` lines; `abort`, `you misunderstood` and the Jev read of the reply act before any merge;
  `NLU_Pipeline_Stage` stayed as a marker until S10 (see the S10 bullet). The intent half of S7
  is the next bullet.
* **S7, the intent half: ONE intent loop (bd `fix-o1lg` S7, done).** `fastworkflow/intent_loop.py` holds the whole-walk decision
  and the reader of a reply to "which of these is meant?" for both paths, replacing the CME's per-scope step
  (`CommandNamePrediction._predict_impl` driven by `ScopeWalk` from `wildcard.py`, with its stage-dependent candidate sets
  and the suggested-commands store in `___convo_info/cache.sqlite3`) and its own reply matching.
  `resolve(workflow, text, core=False)`: (1) exact command name on the lookup path (nearest scope first, the global scope
  last; the core commands first when `core`); (2) the foreign-name guard (a known name no scope of the path owns: no model
  is asked, the routing hint and owners are returned); (3) a fuzzy name (the CME's 0.3 bound; a tie is not a match); (4) the
  local model over the WHOLE walk first; (5) the chain with the remote model, the global scope asked last when the walk did
  not ask it; (6) a local plan pick needs remote confirmation or becomes an ambiguity; (7) a bare value is reported. Commands
  that only enter a context are not offered to the remote model on either path (new for the CME, which offered them on
  purpose). Result: `intent_loop.Intent` (the old `PartIntent`; outcome, command, kind execution / planning / core, ranked
  candidates, `decided_by` including `fuzzy_name` and `foreign_name`, the scope `context` and `record`, evidence, hint,
  owners, bare-value flag). `read` / `answer` is the one reader (`intent_clarification.read`): number, name, leaf, spaced
  name, `none`, cancel, a first word naming a candidate with the rest of the reply kept, and on the CME the plain utterances
  of `abort`, `you_misunderstood` and `what_can_i_do` as aliases; a fuzzy tie is unclear; an unreadable reply is asked once
  more. The pending question is one JSON record (`open_question`), `PlanResolution.choice` on the plan path and the CME
  context key `intent_loop` (saved in the blob's `cme` section), and the client shape is `question_kind` `plan_choice` with
  `artifacts.plan_choice` on both (the "/" path's failed COMPLETED turn carries it in a human session; `intent_choices` is
  gone). `intent_loop.record` is the one learning hook. Departures from the text of sections 2 and 5: the CME's
  misunderstanding stage is the same question with every command as the choices (the `you_misunderstood` command still
  writes the listing the user reads: its file is an input of the CME's training signature), so `none` at the ambiguity
  question opens it; and `NLU_Pipeline_Stage` kept its ambiguity and misunderstanding values only as markers derived from
  the pending record (S10 deleted the enum). The differences that remain: `core` commands are answerable on the CME only;
  one candidate is a bound, reviewed part on the plan path and "Did you mean the command X?" on the CME (no review there);
  two unreadable replies drop the part on the plan path and end the question with the `what_can_i_do` listing on the CME.
  `fw.nlu.intent` v12 is one span per decision; a pick is a span of its own (`matcher_layer` `question_reply`).

* **S8, the consultation workflow is the plan metadata extraction workflow (bd `fix-o1lg` S8).** The built-in review
  workflow moved from `fastworkflow/_workflows/consultation/` to `fastworkflow/_workflows/plan_metadata_extraction/` and every
  name followed: `get_internal_workflow_path("plan_metadata_extraction")`, `is_plan_metadata_extraction_folder`, the test marker
  `requires_plan_head` and `tests/plan_head.py`, the execution context's `_plan_workflow` and `_plan_cme` (the plan workflow's
  own CME instance), the session blob key `plan_cme`. No alias or shim remains. Its pending state sits in one place per
  workflow: the plan workflow's context holds `plan_resolution` (`resolution.PlanResolution`; `WorkflowExecutionContext._plan_resolution`
  is a property over that key and the blob saves it under the same name), while the command metadata extraction workflow's
  context holds the CME's own `intent_loop` and `parameter_loop` (and the plan workflow's CME instance holds its own pair
  for the review reply). The head is a framework embedding head: renaming the folder changes the absolute paths in its
  training signature, so it is retrained in place from the utterance cache (numerically identical, no model call).
* **S10, dead paths (bd `fix-o1lg` S10).** `NLUPipelineStage` and the CME context key `NLU_Pipeline_Stage` are deleted: the
  CME's pending records answer every reader (`intent_loop.waiting_for`). The `aborted` artifact carries `waiting_for`
  (`intent` or `parameters`) in place of `stage`; the session blob has no `nlu_stage` or `plan_nlu_stage`.
  `FW_UTTERANCE_CACHE_SCOPE` and `experiment.runner.UTTERANCE_CACHE_SCOPE_VAR` are removed (inert since S7), and
  `resolution._decide`, a one-line wrapper of `intent_loop.resolve(core=False)`, is gone.
* **S9, a reply that also carries a new request: removed (bd `fix-rpio`).** S9 cut such a reply in code into an answer and a
  new request. It is gone: a reply to a pending question is read by that question's reader for its answer and anything else
  in it is ignored; the which-command, parameter and decision-step questions say so in one sentence. No code turns part of a
  reply into a new request, and `fw_unhandled_request`, `from_reply` and `reply_parts` no longer exist.
