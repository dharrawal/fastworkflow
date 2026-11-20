# Changelog

## Unreleased

- **Docs no longer present plan alternatives in the review as current (docs only; no code, nothing retrains).** The README, the FastAPI architecture doc, the chat-agent skill reference and the run/chatbot parity table said a review item lists "other plans that fit" with `alternatives` and a `switch_plan` form; since bd fix-o1lg.2 an ambiguous part is asked the `plan_choice` question before the review, so those fields are not populated. `step N: use NAME` still switches a step.

- **A value question no longer stays open on endless requests for a moment (bd fix-b3w2; behaviour change; nothing retrains, no `_commands/` file edited).** The first reply in a row that the remote model reads as "give me a moment" still leaves the question out and counts nothing; each further consecutive one now counts as a reply with no progress, so the asking rule's two-in-a-row limit ends the question at the third wait, with the stop a no-progress stop has on each path (a plan part is not run and names what is missing, or the agent hand-over; an execution command ends as a failed output naming what is missing). Any other reply starts the count again. The count is kept in the turn's waiting record (payload key `waited`, `parameter_clarification.waited_of`), not in the loop state, so it is saved with the question and ends with it; a decision step's value or yes/no question follows the same rule (`decision["waited"]`). Who sees it: a client that answers a value question with a request for a moment three times in a row.

- **The unreachable plan review round 2 is gone (bd fix-3rsq.plan-review.7; developers only, no behaviour change; nothing retrains, no `_commands/` file edited).** Nothing opened the plan review of `review_plan`; removed `PlanProposal`, `PlanGoal`, `KIND_PLAN`, the `round` input and output of the `review_proposal` framework command (it reviews sub-tasks only), the `review_plan` framework command file and the round-2 entry of `PROPOSAL_KINDS`. Framework commands are not inputs of any training signature (`compute_command_fingerprints` skips them, `compute_commands_source_fingerprint` reads only `_commands` roots), so no trained head changes. The sub-task review and its typed `step N: use NAME` switch are unchanged.

- **Server-layer tests no longer skip or run on empty config when an env file is absent (bd fix-xd2s, fix-iq53.1.17, fix-iq53.1.19, fix-iq53.1.20; tests only, no product code).** One shared `env_files` fixture (`tests/conftest.py`, writer `tests/fastapi_hermetic.write_placeholder_env`) builds a temp env from the tracked template plus placeholder keys and never reads `env/.env` or `passwords/.env`; the 23 test files that carried their own skip guard, and `tests/test_command_executor.py`, use it. `requires_llm_key` now skips unless a real key is configured (`llm_credentials.real_llm_credentials_available`); the four streaming-ownership tests of `tests/test_manager_shutdown_matrix.py` run on the scripted model seam. A test that asks for `env_files` or `app_module` and skips without `requires_llm_key` is reported as a failure. Seen by contributors and CI: the server tests run on a clean checkout.

- **In agent mode a single fast-path command is run bound to its part, as in no-agent mode (bd fix-jj3k; behaviour change for agent-mode turns; nothing retrains, no `_commands/` file edited).** `WorkflowExecutionContext._subtask_gate` drafts the one-item plan for a single part that names an execution command or that the local model is confident of in both modes, no longer only with no agent model, so `_run_entry` runs it through `_run_plain_means` (code runs it when its values are drafted; the agent's listing is narrowed to that command, as it already was for a bound part of a reviewed turn) instead of the agent choosing the command freely; the narrowing is the listing, not a refusal of other command calls. The review rules are unchanged (none for such a part), a part the local model was not confident of, a multi-part turn and a turn with a pending question run as before, and a bound plain command's answer is still written by the agent; a bound planning command with `__result__` already answers from its own text. A confident local pick that is wrong is now run as picked instead of being corrected by the agent. One drafting extraction call may be added on such a turn when drafting is on.

- **A read that ran out of steps says it is partial (bd fix-cwah; additive answer text; nothing retrains).** When a part's run hits the step cap with no hand-off left, `WorkflowExecutionContext._note_partial_read` ends that part's answer with, for each paged command it read whose reported total is above the rows read, `no_agent_mode.step_limit_note` (the rows read of the total and `no_agent_mode.partial_read_sentence`, the one sentence the page-cap stop's statement uses too). A multi-part turn gets the note under the part that ran out; a command that reports no total, or a run within its steps, adds nothing. `_hand_off` now returns whether a continuation was queued.

- **The split keeps a name with an "and" whole and collapses a stuttered one (bd fix-26pf, fix-hia1.38; nothing retrains).** `split_merge`: a request name that contains "and" (a company chain, a quoted name) counts as named in a sub-task only when written whole, so a name the model cut in two ("Baker" and "Sons and ...") puts the request's sentence back; a name repeated at its own tail ("Blake and Sons and Sons") that the request never writes that long is collapsed to the request's spelling. No model call or review question was added.

- **A split sub-task with a square-bracket placeholder or an invented numbered opener is not kept as a job (bd fix-hia1.41; nothing retrains).** `[Person 1 name] starts on the roster` and `Control 1: who is in it`, when the request does not hold the same text, are visibly broken like a brace placeholder: one retry, then the request runs whole.

- **"The same treatment" and "the same look" are no longer leftover references (bd fix-jtxv; nothing retrains).** They are a request's own wording; `each of them`, `both of them` and `all of them` still flag a split.


- **The observability store's vacuum runs to completion (bd fix-slz6; nothing retrains).** `prune`'s diagnostics cleanup, `forget_channel` and `clear_conversations` ran `PRAGMA incremental_vacuum` without fetching its rows, which frees one page per call; they now fetch them, so the freelist is empty afterwards and the file shrinks as intended.

- **A refused run's `None` is no longer joined into a fact list as `[None]` (bd fix-3rsq.facts-join-none; nothing retrains).** `observation_offloading.facts._join_list` skips `None` values and returns `None` when every run gave `None`.

- **A failed command dispatch records `success=False` on its span (bd fix-zl5b; nothing retrains).** The exception path of `CommandExecutor.invoke_command` ended the dispatch span with only `error_type`; it now adds `success=False`. Control-flow signals (an `ask_user` suspension, a cancel) are not failures and record no `success`. The issue's other half, the "context handles", no longer exists in the code (the span already carries the command and context), so nothing changed for it.

- **An experiment created without a `workflow_name` records its folder's name, not NULL (bd fix-75ex; nothing retrains).** `ExperimentController.create_experiment` defaults the name from `benchmark_setup.workflow_name_for(self.workflow_folderpath)`, the same helper registration uses, so the comparison group recognises the experiment.

- **Internal tracker tags are gone from the chatbot server's comments and two log lines (bd fix-jo9h; no behaviour change, nothing retrains).** The refused-Host and refused-Origin log lines no longer end in a bracketed tag; a test scans `run_chatbot/server.py` for the tag pattern.

- **`make` targets no longer fail when `.env` is absent (bd fix-d7m2; developers only).** The makefile uses `-include ./.env`, and `publish` and `publish-testpypi` check that their token is set and stop with a message when it is not.

- **The typed-phase check accepts an open step that writes the command word (bd fix-rayk; behaviour change for planning-command load checks; nothing retrains).** A phase whose steps are prose now counts as naming a command it hands on when some step contains the command word whole (`find_book`, not `find_books`); a phase that never mentions it is still a load error.

- **`fastworkflow command preview --check` notices a command with more utterances than the matcher shows (bd fix-bh1e.16; output only, `--check` exit status unchanged).** The remote matcher lists the first `MATCHER_EXAMPLES` (9) `plain_utterances` of each candidate; the rest still train the local model. `check_matcher_example_notices` prints one `notice:` line per command with more, next to the ungrouped-list notices.

- **`cancel_pending` drops a question the deterministic path holds, and the refusal text names it (bd fix-u9as; behaviour change for HTTP/MCP clients; nothing retrains).** `WorkflowExecutionContext.cancel_pending` (HTTP and MCP `cancel_pending`) now silently drops every kind of pending question: besides the agent's own and approvals it ends a parameter or which-command question the deterministic path (or a "/" message) holds, returning `cleared: true`; before, it returned false and the next message was read as the answer. `AWAITING_ANSWER_REFUSAL_MESSAGE` names `cancel_pending` and says a reply "cancel" to the agent's own question is only an answer to it.

- **A value only the command word spells is no longer "written in the request" (bd fix-2juj; behaviour change for agent-typed commands; nothing retrains, no `_commands/` file edited).** The grounding guard of an execution command's extraction reads the command text without a leading token that is the command's own name (`fill.without_own_command_word`), so `query=Application` is refused for `find_application SCOM`; the parsers still read the text whole. The issue's integer-sentinel and missing-span observations are not covered.

- **A write a phase's steps name runs at most as many times as they name it (bd fix-ddis; behaviour change for planning-command phases run by an agent; nothing retrains).** `command_writes.phase_write_refusal` counts successful runs of a write per phase run (`_phase_run["writes_run"]`, reset with the phase's plan, so per fan-out item) and refuses a further one unrun (`REPEATED_WRITE_REFUSAL`, artifact `write_refused`); a failed run does not count.

- **A reply that is already a valid number, allowed value or yes/no for the open field is no longer sent to the remote decision model (bd fix-iw13; behaviour change only with a remote model key set; nothing retrains, no `_commands/` file edited).** `parameters.read.fits_constrained` and one condition on the reader's remote gate (`parameters.reply.read_reply`): a reply that fits every open field, each a number, an allowed-value or a yes/no field, is read in code, so it costs no model call and can never be read as a different request. A free-text field, or a reply that does not fit, keeps the remote read.

- **A running part's "you misunderstood" returns to the which-command question; the second home of a running part's loop is gone; the refusal text says how to cancel (bd fix-fpyb.12 and fix-fpyb.11; behaviour change for the first and third; saved-state format change: `SCHEMA_VERSION` 8; nothing retrains, no `_commands/` file edited).** (1) When the value question a running part put (a refused typed planning call, asked before anything of the part ran) is answered with "you misunderstood", the part goes back to the which-command question exactly as it does at the gate (every command the plan can run, `plan_choice` with `misunderstood`), from a one-part resolution built with no model call (`resolution.ask_misunderstood_running_part`, `ask_misunderstood` made public); a pick then asks its values and runs with no review, and `cancel` drops the part as not run. It used to end the part as not run ("you said it was not the command you meant"); a different request and a declined question are unchanged. (2) `WorkflowExecutionContext._part_loop` and the blob key `part_loop` are deleted: the re-run of the part that a reply starts reads the loop from `_rerun_loop`, set and cleared inside that one call, and a re-run that asks again puts a record that carries the whole loop; the asking rule (never asked a third time after two replies in a row with no progress) is unchanged. Old saved sessions are not supported. (3) `AWAITING_ANSWER_REFUSAL_MESSAGE`, what `process_action` and `use_plan` get while a question is out, now says how to drop it: reply `cancel` (`no` for a review or an approval), or `//cancel` at the CLI prompt.
- **The agent's own question and an approval move onto the turn's one waiting record, and one function decides what a message, a cancel and a restore do to it (bd fix-fpyb.8, step 9; saved-state format change: `SCHEMA_VERSION` 7, old saved sessions are refused; nothing retrains, no client-visible shape changes).** `WorkflowExecutionContext._awaiting_user` is a read-only property derived from the record (`Waiting.rides_suspension`: true for every question that rides on the agent's suspension, false for the CME's open command and for an approval raised outside the agent's loop until a reply to it is not understood); nothing sets it. The agent's `ask_user` is the record of kind `agent` (its message in the payload), an approval is the record of kind `approval` pointing at the approval store's record (put by `approval_binding.gate` through the store's owner, `ReviewStore.owner_ref`), and the store ending a record by itself (`review_engine.transition`: accepted, declined, expired, cancelled) ends the turn's record with it, in one place (`_store_record_ended`); `waiting.derive`, `waiting.STRICT` and the autouse test fixture that set it are gone (two questions out at once cannot exist: one slot, `put` replaces). `waiting.step(record, event, facts)` is the transition table of the design note as code (events message, cancel, restore; the "/" message is a message): `_execute_message` runs its move (begin a turn or not, the text the stages read, which stage takes it), `_resume_agent_message` reads the reader from the kind table, `_pending_prompt_kind` the question kind from it (the payload builders moved to `fastworkflow/prompt_payloads.py`), and `ChatSession._pending_cancel_reply` is its cancel event. The saved blob no longer has `awaiting_user`, `suspended_user_message`, `pending_clarification_request` or `assistant_command` (the question is the record; the "/" round is `waiting.assistant`), and a parked conversation parks ONE thing: the record carries the approval store's record it points at (`approval_binding.take_record` / `restore_record` replace `park_reviews` / `unpark_reviews`). A CME open command displaced by an agent's `ask_user` in an agent session is dropped (its stall count restarts).

- **The sub-task review and a paused decision step move onto the turn's one waiting record, and a decision step's reply is read by the one reply reader (bd fix-fpyb.8, step 8; behaviour change for a decision reply that is a request for a moment, "you misunderstood", a different request or `abort`; saved-state format change: `SCHEMA_VERSION` 6; nothing retrains, no `_commands/` file edited, no client-visible shape changes).** A pending review is the turn's `Waiting` record of kind `review` (payload `record_id`, `gate`, `slot`; it points at the approval store's record, which the store keeps), and the review's own slot question is a nested record in its payload (`waiting.SlotPort`, the plan CME's port): `WorkflowExecutionContext._review_slot` and the blob keys `review_slot` and `subtask_review_open` are gone, with `PlanFrame.subtask_review_open` and `PlanFrame.framework_entry` (the same fact twice) and the `"open"` marker of `_turn_subtask_review` (the turn span still says `open` while the gate's review is out). A paused decision step is the record of kind `decision` (`phase_run["decision"]` is gone); the frame loop tests the record's kind and the framework call by its frame. The decision step's reply is read by `parameters.reply.read_decision` (the one reader with the decision's own answer rule plugged in through `read_reply(answers=)`): yes or no for a conditional step, the value for a missing-value step, as before; and now a request for a moment keeps the question (nothing counted), while cancel, abort, "you misunderstood" and a different request end the command at that step as a stop with reason `declined: ...` naming why (the reply is never run as a new request). The asking rule is unchanged (an unusable reply is asked once more, then the stop). With a remote decision model key set, a decision reply that is not a yes or no or a `name: value` line costs one model call. A running part's "you misunderstood" still ends the part as not run, saying you said it was not the command you meant (a which-command question for a running part needs a continuation for a pick that does not exist). `WorkflowExecutionContext._part_loop` stays. `derive` no longer reads the approval store for a review or `phase_run` for a decision.

- **A plan part's value question understands the same replies as an execution command's (bd fix-fpyb.8, step 7; behaviour change for people who answer a plan's questions; one more model call per free-text reply when a remote decision model key is set; nothing retrains, no `_commands/` file edited).** One reader, `parameters.reply.read_reply`, reads a reply to a value question on both paths; `parameter_reply.read_reply` (the deterministic path) is now a thin adapter from its result to the six actions of the dispatcher. A plan part's value question (at the gate before the review, and the question a running part puts) now understands: a request for a moment (the question stays, nothing is counted as a reply without progress), "you misunderstood" (the part goes back to the which-command question, offering every command the plan can run; a running part, which has no resolution to go back to, ends as not run saying so), a different request (the part is set aside as not run, its entry saying the reply asked for something else, and the reply is NOT run as a new request), and `abort` as well as `cancel`. On the deterministic path `cancel`, `stop` and `never mind` now end the command like `abort`. The remote model's read is made at most once per reply, never for `name: value` lines or a fixed phrase, and never when no key is set. `command_choices(core=False)` leaves the CME's navigation commands out.

- **One asking rule for a missing or refused value, for planning and execution commands, in every session (bd fix-fpyb.8, step 6; behaviour change for people and agents who answer parameter questions; saved-state format change: `ParameterLoop` loses `policy` and `pending`; nothing retrains).** The three policies are gone (`ParameterLoop.policy`, `POLICY_BOUNDED/ONCE/UNBOUNDED`, `advance(fallback_runs=)`, `WorkflowExecutionContext._parameter_question_asked`). A question is asked again while each reply makes progress (it fills or fixes a value the question listed; one definition, `parameter_clarification.made_progress`), a reply with none is asked once more, the second one in a row stops it (`ParameterLoop.stalled`), and `cancel` always ends it. What changes per session: a human session on the deterministic path (a `/` message, or no agent model) no longer sees an execution command's value question repeat for ever: after two replies in a row with no progress the command ends as a failed output naming what is missing, with no question pending (`abort` still ends it at once; a reply that makes progress starts the count again); an agent session (the client is an agent) gets the same stop on the same path, where a failing call used to be left to the agent's own iteration limit; with an agent model set, a plan part's value question is asked again after a reply that made progress but left a value open (it went straight to the agent or text planner), every reply is read by the shared extractor (one more model call per reply), `cancel` sent to it cancels the part (it was read as the part's text), and after two replies in a row with no progress the agent or text planner takes over with the refusal, never asked a third time; with no agent model nothing changes (a plan part still asks while replies make progress and then ends as not run naming what is missing).

- **The plan path's which-command and values questions move onto the turn's one waiting record (bd fix-fpyb.8, step 5; saved-state format change for the plan path, no client-visible shape changes, nothing retrains, no behaviour change).** A plan part's which-command question and its values question (at the gate, or asked while a part runs) are `WorkflowExecutionContext._waiting` records (owner `plan`, kinds `intent` and `parameters`; `payload.part`, and for `parameters` `payload.after` = `resolve_next` or `resume_part` and `payload.loop`). `resolution.PlanResolution` lost `pending`, `asking`, `choice`, `loop` and `at_gate` and `run_time` / `question` / `prompt` (now `resolution.prompt(record)`): it holds the resolution's progress only. `resolution.Turn` carries the record's `port`. `waiting.derive` no longer reads the plan resolution. The saved `waiting` section holds the question and `plan_resolution` the progress; a new key `part_loop` keeps the loop of a running part's last question (read by the once-only rule and by the progress rule when the part re-runs). Old saved sessions are not supported.

- **The deterministic path's pending question moves onto the turn's one waiting record (bd fix-fpyb.8, step 4; saved-state format change: old saved sessions are refused, `SCHEMA_VERSION` 5; the built-in workflow needs ONE retrain because `wildcard.py` changed).** A which-command question and an execution command's parameter question are no longer keys (`intent_loop`, `parameter_loop`) of the built-in workflow's context: they are `WorkflowExecutionContext._waiting` (`fastworkflow/waiting.py`, kinds `intent` and `parameters`), and the built-in commands reach it through a port under `context["waiting"]` (`waiting.Port`, `OwnerPort`; a context built without an execution context gets a `Holder`). `intent_loop.INTENT_LOOP_KEY` and `parameter_clarification.CME_LOOP_KEY` are gone; read through `intent_loop.pending_record`, `waiting_for`, `parameter_clarification.pending`. The session blob's `cme` and `plan_cme` sections are replaced by `waiting` and `review_slot` (the record's saved form; the CME's `command` and `command_name` ride in its payload), there is no reader for the old keys, and `waiting_for` no longer raises (one slot cannot hold two). The "/" round flag leaves the context (`is_assistant_mode_command`): it is `WorkflowExecutionContext.assistant_round` and `Waiting.assistant`, ended in one place (`end_round`). The reply to a pending parameter question is read outside `_commands/` (`command_metadata_extraction/parameter_reply.read_reply` returns a `Reply`: an action of the frozen vocabulary `run`, `wait`, `different`, `replace`, `merge`, `stop`) and `wildcard.py` only dispatches it; behaviour is unchanged.

- **A `/` message sent while a question is out is the reply to it (bd fix-fpyb.7, step 2 of fix-fpyb.8; behaviour change for clients and for `/` users; nothing retrains).** One function, `WorkflowExecutionContext._route_message`, decides what an incoming message is. While ANY question is pending (an agent's `ask_user`, a plan's which-command or value question, a sub-task review, an approval, a paused decision step, or the deterministic path's own question) a message that begins with `/` is read as the reply with the leading `/` removed. Before, it was run as a deterministic command beside the question and its output was appended to the suspended turn, which stayed pending with no sign that anything had intervened, while the same message on the deterministic path was read as a reply, so the two paths disagreed. `POST /invoke_assistant` and the MCP `invoke_assistant` tool keep the `/` and the context removes it, as `/invoke_agent` always did. `/approvals`, `/revoke_approval` and `/cancel_pending` keep their meaning. With nothing pending a `/` message is a deterministic command as before. `process_action` and `use_plan` sent while a question is out are refused with one text (`AWAITING_ANSWER_REFUSAL_MESSAGE`, reworded to "reply to it before sending a request, an action or a plan"; `use_plan` raised its own text before). The decision-step question no longer says that a separate command can be sent to the assistant meanwhile. Internal: the five "can anyone answer" predicates are two (`waiting.can_ask`, `waiting.can_review`); `_prepare_message_routing` and `_subtask_no_answerer` are gone.

- **`//cancel` ends every pending question (bd fix-fpyb.6, step 1 of fix-fpyb.8; behaviour change, nothing retrains, no client-visible shape changes).** A session's one pending question is now read in one place: `WorkflowExecutionContext.waiting` (`fastworkflow/waiting.py`), a read-only view derived from the existing flags and records, with one kind table (`question_kind` and the cancel reply of each kind) and a check that at most one question is out (the test suite fails on two; a live session logs one error naming the kinds and acts on the one the earlier code acted on: approval or review, plan resolution, decision, then the CME). `ChatSession._pending_cancel_reply` and the pending-prompt-kind lookup (`question_kind` of an awaiting output) are lookups on it. `//cancel` used to end only an approval, a plan review and the CME's open command; it now also ends a plan's which-command question, a plan's value question and a paused decision step (reply `cancel`, as typing it does). A suspended `ask_user` is still ended by the blocked read's own cancel. `docs/pending_state_design.md` is the agreed design of the whole migration (this is its step 1).

- **The review reply guards read input names from the workflow, not from fixed lists (bd fix-exvh; behaviour change on workflows that do not use the old nouns, nothing retrains).** `review_proposals._ENTITY_INPUT_KEYS`, `_GENERIC_INPUT_NAMES`, the noun branches of `_infer_kind_from_clause` and `review_engine._KNOWN_SUBTASK_INPUT_KEYS` (names of one application) are removed. `ProposalReplyPolicy.decide` opens `review_proposals.workflow_scope(folder)` and the guards read what the workflow's planning commands declare (`declared_params`, `declared_input_names`, `entity_input_names`: a str input with `copied_from_command_text` or `record_named_in_command_text`): a `change N: field=value` on a declared input becomes an inputs edit; a declared entity input is the one a quoted name completes and its bare name never selects a step; a clause that names exactly one value of an enum input the step's planning command declares sets that input (the synonyms the old list mapped, such as organization to unit, are gone). Without a declared name each of these is off, as before on a foreign workflow. `_NO_EFFECT_OBJECT` is English grammar (what "do not run any actions" negates), not application data, and is unchanged.

- **A no-agent stop whose steps all ran by code now states what ran (bd fix-9m0p; behaviour change for the answer text only).** A planning command with no `__result__` text used to end with only the notice that the answer needs a model; `no_agent_mode.explanation` now opens with `ran_statement` (the steps that ran and the facts recorded, in markdown, no model call) and the notice follows. A stop at a paged read's page cap gets the same statement and says any count is of a partial read. The statement now also prints, under each step line, that step's own result text (the command's response, cut at 300 characters with a note of how much), so a write step's outcome is visible; the stop entry gains the additive keys `results` and, at a page cap, `partial_read` (`pages`, `rows_read`, `total` when the command reported one, `rows`: the first list field of each page's Output). At a page cap the facts stay empty, because a phase's facts are filled only when it ends; the rows read so far were held only in the step's runs, and are now read from them.

- **The split and the planner no longer run a planning-command match of their own (bd fix-zvme; behaviour change: the same text gets the same plan in the split, the planner and the resolution that follows).** `workflow_agent.match_planning_command`, its matcher `_planning_command_matcher`, `planner_planning_command_for` and `plan_template_for` are removed. A split over the part cap labels its parts through `workflow_agent.plan_label` (`resolution.part_intent`, memoised for the turn; only a planning match counts, it never asks), so a part labelled during the reduction is not decided again by the resolution, and `split_into_subtasks` takes the app `workflow` for it (without one nothing is labelled). The planner's template was already read from `part_intent`. `PLANNING_COMMAND_MATCH_THRESHOLD` stays as the bar of the run listing's gathering only. Remote calls per split of N parts: at most one per labelled part before (root scope only) and at most one per part now, shared with the resolution.

- **The commit-threshold calibration judges the wrong share of the band, not of all confirmed rows (bd fix-1y36; behaviour change for `fastworkflow learn` and the stored calibration, nothing retrains).** `intent_learning.calibrate` lowers the threshold to the smallest candidate with at least 200 confirmed rows, at least `CALIBRATION_MIN_BAND` = 100 rows in its band and at most `CALIBRATION_MAX_WRONG_SHARE` = 2% of that band wrong (was 30 rows and 1% of ALL confirmed rows, so 2 wrong of a 30-row band, 6.7% of the requests it would newly answer, passed). The evidence entries carry `wrong_share_of_band` (was `wrong_share_of_confirmed`) and `fastworkflow learn` prints the best candidate's wrong count and share. The refit gate's comment now says 19 of the 21 rejected attempts (of 26) failed the base rows condition.

- **One parameter pipeline for planning and execution commands (bd fix-qoyz; behaviour change for command authors and for the people and agents who answer parameter questions; no `_commands/` file edited, nothing retrains).** `fastworkflow/parameters/` holds the rules once (`rules`, `validate`, `check`, `fill`, `pipeline`, `ask`, `read`); `planning_commands.check_arguments`, `utils.argument_checks`, the typed planning call, `extract_arguments` and the CME's parameter stage are callers of it, and no rule differs by kind of command any more. What a user or a command author can see: (1) **reply reading**: one reader for both kinds. `name: value` lines set a field the question listed; with one value open the whole reply is the value (commas included, `Smith, John` is no longer cut to `Smith`); with several open, comma parts fill them in order only when the count matches and each part fits its field; a reply never fills a field the question did not ask for; `no`, `none` and `neither` are the value of a yes/no field or an allowed value, leave an optional field empty and are unclear for a required plain field (they were read as "you misunderstood"); `you misunderstood` is read only by its own phrases; `yes` accepts only a question that showed every parameter. Known limit: prose with a matching comma count sent to several plain-text fields still fills them. (2) **Plans: an `int` the request does not write is refused**, never replaced by the one number it does write; an absent one is not checked; **a `str` or `int` must be written in the request** unless its field is converted or has a pattern (it was opt-in through `copied_from_command_text`); a value of only punctuation or an unwritten description example is refused on both kinds; guards read only values the extractor gave. (3) **A refusal that names no input is not asked**: an execution command ends as a failed output with the validator's message and no pending parameter loop (it used to open a question with nothing to ask); a guard refusal always names its argument, so it is asked like any refused value. (4) **One second extraction pass, the same on both kinds**: only when a guard or a validator refused a value the model produced (or every answer failed to parse), never for a value that is simply missing (a plan used to read a missing value twice); when the second pass fails the first answer stands and the typed planning call goes on to its typed fill instead of raising. Execution commands now get that second pass after a validator's refusal too (it was after the guards only). (5) **Validators see the request plus the replies** as `command_text` on both kinds; a validator's rewrite is kept when it passes and dropped when it refuses. (6) **Wording**: a person gets one plain sentence for a missing or refused value on both kinds; the text an agent reads is unchanged; an agent session is decided by the value of `run_as_agent`, not by the key's presence. (7) `SubTask.check_missing` (additive, absent when empty) carries the names the review line asks for instead of reading them back from `SubTask.check`'s text; `argument_gaps` and the refusals carry `Gap` records, so a refused value containing ` is ` is no longer reported with a garbled reason. Removed: the aliases the two guard modules re-exported while the rules moved (`planning_commands._clean_str`, `_fold`, `_written`, `_placeholder_like`, `NUMBER_WORDS`, `PLACEHOLDER_WORDS`; `argument_checks.written_numbers`, `allowed_values`, `is_grounding_exempt` and the like: import them from `fastworkflow.parameters`), `input_validation.ValidationTrace` and the `trace` argument of `validate_input`, and `pipeline.Retry`. `docs/parameter_pipeline_design.md` has the rules.

- **A reply to a pending question is never cut into an answer and a new request (bd fix-rpio; behaviour change, removes artifacts and a state key).** `fastworkflow/reply_request.py` (`split_reply`) is removed with `WorkflowExecutionContext._cut_reply` and `_cut_cme_reply`, the `reply_parts` state, the parts a reply added to a turn (`PlanResolution.added_from`, `start_added`, `add_parts`), the `from_reply` mark of `fw_sub_tasks` entries, the `fw_unhandled_request` artifact and the reply-with-request benchmark and its tests. The cut invented parts from ordinary replies: a value that begins with a verb (`Get Out`, `Dune and Show Boat`) was split into an answer and a request. Now each reply reaches its question's reader whole, which reads its answer and ignores the rest; the which-command, parameter and decision-step questions say so in one sentence ("Only your answer is read; anything else in the reply is ignored."). The turn's review reader is unchanged.

- **A reply to the agent's own `ask_user` question is no longer cut into an answer and a new request (bd fix-o1lg.8, fix-yu3l.7; behaviour change).** `reply_request.split_reply` returns None for `KIND_FREE` and the rule that a plain yes, no or cancel head is a strict answer to a free question is removed. On a benchmark run on a large private workflow, "Yes, open whatever you need so you can give me the full list ..." was cut after the yes and its rest became an invented second part that needed three review edits. The agent now gets the whole reply and no part is added. The which-command, parameter, review and decision-step replies are still cut. The experiment script's strict numbers are unchanged (dev 0/84 false cuts, 41/56; confirm 2/80, 29/52).

- **A saved session whose plan review is pending can be answered "yes" after a restore (bd fix-o1lg.8; bug fix, additive session key).** The phase runner read `own_steps` only on a fresh phase, so the resumed review phase of a restored no-agent session raised `UnboundLocalError`; it is now kept in the phase state and read back on resume. The settled parts' learning facts (`_turn_part_facts`) are saved under the additive key `turn_part_facts`, so an accepted review after a restore is still recorded for learning. The experiment script `scripts/experiments/real_walk_check.py` patches `intent_loop._walk` (was the removed `resolution._walk`).

- **A reply to a pending question can carry a new request (bd fix-o1lg S9, fix-yu3l.7; behaviour change for every reply of that shape, additive artifacts).** `fastworkflow/reply_request.py` `split_reply` cuts such a reply in code (the strict setting of the benchmark in `docs/reply_with_request_benchmark.md`; no model call) before the question's reader reads it, so a reader that took the whole reply (`book_loans. Then count the overdue loans`, `title: Dune. Also count the overdue loans`) no longer drops the request. Plan path (no-agent and agent mode; the which-command and planning-parameter questions, a refused typed call's question, the turn's review, a plan's decision step, the agent's `ask_user`): the answer part is applied as the whole reply would have been and the request becomes new part(s) of the turn, after the existing ones, through the split, `resolution` and a review before they run (in the turn's one review while it is ahead, else a review of their own after the earlier parts ran; declining drops only them; a surface that cannot answer a review does not run them and says so); `fw_sub_tasks` entries of such a part carry `from_reply: true`; the saved state gains `reply_parts`. Deterministic path (the CME: assistant mode and `/` messages): the request is NOT run (no split or review there); the answer part is applied, the response ends with one sentence quoting the rest and the artifact `fw_unhandled_request` (`{text, answer}`) carries it. Known: restoring a session saved while a plan REVIEW is pending and then accepting it raises an `UnboundLocalError` in the phase runner in no-agent mode (found while testing, independent of this change, not fixed here).

- **Background refit of the intent head is now on by default (bd fix-qcjo; behaviour change).** `INTENT_LEARNING_REFIT` defaults to `background`: after `INTENT_LEARNING_REFIT_EVERY` (25) new confirmed rows one daemon thread per workflow refits the embedding head behind its gates, and the running process picks the new head up without a restart. The live path is covered by `tests/test_learning_live_refit.py` (off-turn thread, gates, file lock, failed gate keeps the head, agent picks teach). Set `INTENT_LEARNING_REFIT=off` to turn it off.

- **The consultation workflow is the plan metadata extraction workflow, and the NLU stage enum is gone (bd fix-o1lg S8 and S10; behaviour change for embedders and clients).** `fastworkflow/_workflows/consultation/` is `fastworkflow/_workflows/plan_metadata_extraction/` (`get_internal_workflow_path("plan_metadata_extraction")`, `is_plan_metadata_extraction_folder`; the test marker is `requires_plan_head`; the session blob keys `consultation_cme` and `consultation_nlu_stage` are `plan_cme` and gone). A head trained in the old folder is retrained in place with `fastworkflow train` (no model call from the utterance cache; the same head). The pending plan resolution lives in the plan workflow's context under `plan_resolution`. `fastworkflow.NLUPipelineStage`, the CME context key `NLU_Pipeline_Stage` and the blob keys `nlu_stage` and `plan_nlu_stage` are deleted: `intent_loop.waiting_for(context)` reads the pending records (`intent`, `parameters`, None), and the `aborted` artifact carries `waiting_for` in place of `stage`. `FW_UTTERANCE_CACHE_SCOPE` (inert) and `resolution._decide` (a one-line wrapper of `intent_loop.resolve`) are removed.

- **One intent loop for the plan path and the deterministic path (bd fix-o1lg S7, the intent half; behaviour change for clients of a failed "/" command and for saved CME sessions).** `fastworkflow/intent_loop.py` is the one decision (`resolve`: exact name, foreign-name guard, fuzzy name, the local model over the whole walk first, then the chain with the remote model, a local plan pick needing remote confirmation) and the one reader of a reply to "which of these is meant?" (`read` / `answer`), used by `CommandNamePrediction` and the `wildcard` router and by `resolution.part_intent`; `resolution.PartIntent` is `intent_loop.Intent`. Client-visible: (1) an ambiguous "/" command no longer carries `intent_choices`; it carries `question_kind` `plan_choice` and `artifacts.plan_choice` (`request`, `sub_task`, `choices`, `none_reply`, `decline`, plus `ambiguous` and `misunderstood`), the shape the plan path already used, and the chat page draws both with one renderer (the pills, "See all commands" and the old label are gone; `none` shows every command, `cancel` ends it); (2) the question text is one function (`intent_clarification.plan_choice_question`), so the ambiguity message reads "More than one command fits ...: Which one is meant?" with numbered, described candidates, and one candidate is "Did you mean the command 'X'?" answered by its name, `yes` or `cancel`; (3) a reply picks by name, leaf, spaced name, number, `none`, `cancel`, or a first word that names a candidate followed by that command's parameter text, and two close misspellings are unclear (the first of a tie is no longer taken); (4) a reply that cannot be read is asked once more and then the CME ends the question with the `what_can_i_do` listing (the plan path drops the part); (5) the plan path gains the fuzzy command-name layer and the foreign-name guard, and the CME stops running the chain scope by scope: a confident local answer anywhere on the walk decides with no remote call, the global scope is on the remote model's list, and a command that only enters a context is no longer offered to the remote model on the deterministic path; (6) a pick runs on the record of the scope that has the command (`chosen_command_elsewhere` on the span), and a reply that names a real command owned off the path gets the routing hint; (7) the typed name of a core command (`abort`, `what_can_i_do`, ...) is matched by name in the intent stage. Removed: the suggested-commands store (`___convo_info/cache.sqlite3` is no longer written), `CommandNamePrediction._predict_impl(context, ..., stage)` and the per-scope `predict(context, command, stage)` (now `predict(command)`, one `fw.nlu.intent` span per decision: span contract v54, `fw.nlu.intent` v12), the plan CME's `convo_path`, `intent_learning.evidence_of_trace`. The pending question is one JSON record under the CME context key `intent_loop`, saved with the session.

- **One result contract for every extractor, and the CME parameter stage on the shared loop (bd fix-o1lg S6 and S7 parameters; behaviour change for saved sessions, none for a live turn).** `parameter_clarification.Extraction` is the values of a command given some text and each parameter's state (given, missing, invalid with the reason, from an earlier part); the plan side (`planning_commands.argument_gaps`, `refusal_gaps`) and the execution side (`parameter_extraction.extraction_of`) both produce it, and the `parameter_request` artifact, the question and the progress rule read only it (`Extraction.request`, `from_earlier`, `record_round`). It replaces `planning_commands.ArgumentGaps`, `parameter_clarification.without` and `parameter_extraction.parameter_request_for`; `extract_arguments` returns `ExtractedArguments(params, extraction, retried)`. The CME's pending parameter question is one `ParameterLoop` (kind `execution`) under the context key `parameter_loop`; the keys `stored_parameters`, `stored_parameter_round` and `missing_value_question` are gone, as are `ParameterExtraction._get_stored_parameters`, `_get_stored_round`, `_store_parameters`, `_clear_parameters`, `_extract_missing_fields`, `_merge_parameters`, `missing_value_question` and `pending_missing_fields`, and the model dump and rebuild in the session serializer (the loop is plain data and saved whole, as `cme.parameter_loop`; a session saved with the old keys is not supported). Two small changes follow: the extraction round (`retry_round_ordinal`) now survives a restore (fix-7gp9), and the Jev read of a reply to a missing-value question is given the question as the user saw it (it was a reconstruction of its first lines). The question text, the artifacts and the reply merge are unchanged. The intent half of S7 is not done (see the bd notes of fix-o1lg S7).

- **Learning from use: the refit gate has two tolerances, and training signatures no longer depend on how the workflow path was spelled (bd fix-o5wi.6; behaviour change for `fastworkflow learn` and the background refit, no signature input changed, nothing retrains).** `head_refit.py`: the base rows' cross-fit accuracy may fall by `GATE_BASE_TOLERANCE` = 1.5 points (was 0.5); on the held-out learned rows right outright must not fall and wrong confident must not rise, except by `GATE_WRONG_TOLERANCE_ROWS` = 1 row when right outright rose by at least `GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS` = 3 rows. The gate report keeps its fields and adds `rule` and `baseline_cached`; the base rows' cross-fit without the learned rows is kept in the learning store (`base_crossfit`) and reused. On a large private workflow right outright rose from 64 to 111.7 of 282 (first gate 82.3), wrong confident fell from 11 to 2.7 (first gate 7.7), 5.7 of 8 refits published (first gate 1.7 of 8.7); on the library example 50.0 to 70.7 percent right outright and wrong confident 3.5 to 3.9; on retail 38.0 to 68.8 and 2.2 to 1.1 (first gate: 0.0). A set of wrong confirmations that costs the base rows under 1.5 points is now published (seed phrasings plus one word, confirmed as another command); seed phrasings confirmed outright as another command still fail by over 20 points. `CommandDirectory.load`, `train_workflow` and the signature functions in `selective_training.py` resolve the workflow path with `os.path.abspath` first: a relative path used to bake relative command-file paths into `command_directory.json` and the training signature, so a later `fastworkflow train` from another directory wanted a retrain and failed importing command files. Signatures of absolute paths are unchanged.

- **Learning from use (bd fix-o5wi.6; additive, recording on by default; a train writes one more file).** Every resolved request is recorded in `<workflow>/___convo_info/intent_learning.sqlite3` (`fastworkflow/intent_learning.py`; `INTENT_LEARNING_RECORD`, default on): the request, context, resolved command, how it was settled (`local`, `remote`, `name`, `user_choice`, `agent_choice`, `review_accept`, `review_switch`), whether a person or agent confirmed it, the local head's facts at decision time and the remote answer, and the answer a switch or a you-misunderstood replaced. It is written from one place per path (the CME's wildcard command, and where the resolution path runs a plan or takes its fast path) and never fails a turn. `fastworkflow learn <workflow> <env>` refits the embedding head with the confirmed rows (`fastworkflow/train/head_refit.py`): the train now stores the vectors it computed (`rows.npz` beside `head.npz`, about 1 MB for the library example), so a refit embeds only the learned texts, and it publishes a new version only when a cross-fit gate passes (the base rows lose at most 0.5 points; the learned rows are no worse right outright and no more often wrong with confidence than without them). The same command calibrates the commit threshold from the recorded requests and can only lower it (0.60, 0.55 or 0.50, with at least 200 confirmed rows, 30 in the band and at most 1% of all confirmed rows wrong; `INTENT_COMMIT_THRESHOLD` set in the env always wins), and the threshold in force and why are in the head's details, the detection span and the startup notices. A server picks up a newly published embedding head within a few seconds without a restart (`intent_detector.refresh_stale_head`). `INTENT_LEARNING_REFIT=background` (default `off`, `INTENT_LEARNING_REFIT_EVERY` default 25) refits on a daemon thread. `fastworkflow learned list|remove|stats` reviews and deletes rows. **The training signature of every embedding-layout workflow changes** (the head's trainer module is a signature input and now writes `rows.npz`): the next `fastworkflow train` retrains them, the built-in plan and command-metadata workflows included (their heads are deterministic from the same rows), and a head trained before has no stored rows and is not refittable until then. Also: a which-command question that would list ONE candidate is no longer asked: the candidate is bound and the turn's one review confirms it (bd fix-o5wi.12; the CME's one-candidate message reads "Did you mean the command X?"), and a list of one never says "More than one command fits".

- **The remote decision model's two bars are 0.9, an outage of it is visible, and the deterministic path reviews a local plan pick (bd fix-o5wi.11; behaviour change: the chain asks the user more often and acts on a wrong remote answer less often).** (1) `intent_detector.DEFAULT_REMOTE_ACCEPT_THRESHOLD` and `DEFAULT_REMOTE_DECLINE_THRESHOLD` are 0.9 (were one 0.5 `MATCHER_MATCH_THRESHOLD`, removed); `INTENT_REMOTE_ACCEPT_THRESHOLD` and `INTENT_REMOTE_DECLINE_THRESHOLD` override them in the workflow env, read in one place (`remote_accept_threshold`, `remote_decline_threshold`). A remote answer below the accept bar is not an answer: the candidates go to the question. The `fw.nlu.intent` `matcher.threshold` is now the accept bar and `matcher.decline_threshold` is new. Measured on 1,102 stored answers (`docs/remote_link_worth.md`). The planning-command match keeps its own 0.5. (2) `fastworkflow/remote_outage.py`: a failed remote call (the command match, the planning-command match, the reply-intent classifier) logs one WARNING with the error class and status code, then at most one per 5 minutes with a count, and the `fw.nlu.intent` span carries `remote_failed` (span contract v53, `fw.nlu.intent` v11); the fallback is unchanged. (3) `CommandNamePrediction` (assistant mode, `/` messages) no longer runs a planning command on a confident local answer alone: the remote model confirms it or the candidates are asked, as `resolution._second_opinion` does (`intent_detector.names_planning_command`); an exact or fuzzy NAME still runs. (4) The review's `drop` command has three more seed utterances and the built-in plan head is retrained (held-out 31/36 as before); "cancel the third one" is now a confident `drop`, "we do not need the joke" is no longer a confident `decline`.

- **The local intent model as chosen: a base encoder, description rows, it never passes a request up alone, a fixed threshold (bd fix-o5wi.9; behaviour change for every `embedding`-layout workflow, retrain to get it).** From `docs/local_intent_model_levers.md`. (1) The default encoder is `intfloat/e5-base-v2` (frozen, mean pooling, the prefix `query: `; one shared download of about 418 MB), `INTENT_EMBEDDING_MODEL` still overrides it; the prefix is recorded in `scopes.json` (`encoder.prefix`). (2) Each command's humanised name and the first sentence of its description are rows, and seed utterances weigh three times a generated row. (3) The head never answers a lone "not here" because a command outside the context is ahead (the pass-up margin and `INTENT_FALSE_ESCALATION_TARGET` are gone): it is not confident and hands the request on, and `ClassifierEngine(router, reachable=...)` puts the best commands of the scopes on the lookup path among the candidates (`details["handed_on"]`; the vague-reply class and the bare-value label keep their meaning). In the ambiguity stage the scope declines a chosen command it does not have, so choosing an ancestor's command runs it on that scope's record. (4) The commit threshold is `INTENT_COMMIT_THRESHOLD`, default 0.65, not a cross-fit calibration (`INTENT_WRONG_COMMIT_TARGET` is gone); `intent_model_layout.calibrated_commit_threshold` is the place a later calibrated value is read. A removed setting left in an env file is ignored with one warning. (5) The training report lists the head's cross-fit accuracy and the ten most confused pairs of commands. Heads trained before this (`scopes.json` `schema_version` 1) are read with the rule they were trained with until retrained. The framework's own workflows keep the pinned encoder `BAAI/bge-small-en-v1.5` (`FRAMEWORK_ENCODER`): the default encoder scored 30 of 36 on the review reply corpus with `look_up` at 2 of 6 and answered "hmm" with `confirm`, under the test gate; bge-small with the new rules scored 31 of 36. Span contract v52 (`fw.nlu.intent` v10: `chosen_command_elsewhere`; the local model's facts carry `outside_labels`, `outside_scores`, `handed_on`). Measured on the library and retail examples through the product trainer (development benchmarks 1 to 3, pooled): 59.6% right outright, 1.8% wrong confident, the ancestor's command among the candidates for 117 of 120 escalation requests; fourth benchmark, scored once: 45.8% right outright and 3.0% wrong confident (the previous head: 60.2% and 10.2%), 45 of 46. A request takes about 25 to 32 ms on CPU (10 threads), up from about 10 ms.

- **The similarity cache is removed from intent detection (bd fix-o5wi.8; behaviour change: a clarification no longer teaches later requests).** `CommandNamePrediction.predict` is now exact name, fuzzy name, local model, remote model, question. Removed: `fastworkflow/cache_matching.py` (`cache_match`, `store_utterance_cache`, the sentence-vector helper), `kvstore.UtteranceCacheStore`, `sentence_embedding` on the embedding head, the store write after an ambiguity or "you misunderstood" answer, and the span attributes `cache_similarity` and `cache_similarity_threshold` (span contract v51, `fw.nlu.intent` v9; `matcher_layer` is never `embedding_cache`, and `RoutingTier` keeps the name so a stored record still loads). Measured in `docs/learning_from_corrections.md`, at cosine 0.85 it added nothing on unseen requests and 21 to 58 percent of its answers there were wrong. Learning from corrections is planned as recorded rows plus a background refit of the head (fix-o5wi.6). An existing `___convo_info/cache.sqlite3` is no longer read for this; its `utterance_cache` table is dead and the file may be deleted (the suggested commands of an ambiguity question live in the same file).

- **The framework's own workflows use the `embedding` intent head (bd fix-o5wi.3; behaviour change for the plan review's reply head).** `intent_model_layout.FRAMEWORK_LAYOUT` is `embedding` (was `per_context`); `fastworkflow train` on the plan and command_metadata_extraction workflows writes one 15 KB and one 12 KB head in seconds, from the utterance cache, instead of a 288 MB classifier pair each; their embedding settings are the defaults, not the application's (`embedding_settings(workflow_folderpath)`, `is_framework_workflow`). Why: a per-context retrain of the review head did not reproduce the previous head (17 of 43 recorded replies answered differently after one retrain). On the review workflow's held-out corpus (36 replies) the new head names the right command first for 31, against 28 for the 13 and 17 November per-context heads, with every `confirm` reply a confident accept and no confident wrong answer; the built-in vague replies are still its "not here" class ("hmm", "not sure" are answered `wildcard`, never a command). `per_context` and `workflow` still train and read when an application asks for them by name. The recorded review-router answers were re-recorded (12 of 43 replies changed: `hmm`, which the old head answered with a confident `confirm`, and `just do it`, a confident `decline`, are now `wildcard` and a confident `confirm`). `train` of the package no longer reports a framework workflow in a one-model layout as untrained (`_internal_workflow_trained`). Tests: the framework-layout test is rewritten for the embedding layout; the plain-cue tests no longer treat `just do it` and `hmm` as head errors; `test_a_correct_reading_of_a_possessive_is_applied` is now `test_a_possessive_the_head_reads_as_a_drop_is_asked_about_and_the_plan_is_unchanged`.

- **Every part of a turn is resolved before the turn's one review; the pending state is `plan_resolution` (bd fix-o1lg.2; behaviour change, both modes).** `fastworkflow/_workflows/plan_metadata_extraction/resolution.py` (`PlanResolution`, `start`, `step`, `take_reply`, `finish`) resolves the parts in order: which command or plan (asked only when `part_intent` is ambiguous, the candidates, execution and planning commands, being the choices; with no agent model also when nothing serves the part), then the values a bound planning command lacks (asked only when it lacks one; once with an agent model, where the inputs were drafted; a value an earlier part produces is never asked for). A question can be cancelled, which drops that part and the later parts go on. Then ONE review of the whole turn, then the run, which asks no more intent or parameter questions (a refused typed planning call of a single unreviewed part keeps its one question). Before this, the review came first and the questions after it, the review offered the other plans of an ambiguous part, and a reply could switch to one: those alternatives are not offered in the review any more. No review for a single part that names a command, that the local model is confident of for an execution command, or whose command the user chose by answering; a pinned plan's parameter question is that turn's review. The state is one JSON record saved under `plan_resolution`; the keys `plan_choice` and `parameter_loop` are gone and old saved sessions are not supported. Client-visible shapes are unchanged (`question_kind` `plan_choice` and `planning_parameter`, `artifacts.plan_choice`, `parameter_request`). In no-agent mode a part dropped by `none`, `cancel` or unreadable replies is a `fw_not_run` entry with `request` (and `available_plans` for an ad hoc part) with `sub_task` null, and a turn whose every part was dropped answers with those entries' explanations. Two `part_intent` rules: a local answer naming a planning command is confirmed by the remote model or asked, never the unreviewed fast path (unless the part names the plan); commands that only enter a context are not offered to the remote model. Not done here (later steps of fix-o1lg): one extractor contract, the CME wildcard on the shared loops, renaming the plan metadata extraction workflow, replies that carry a request, deleting the unreachable alternatives code (`SubTask.alternatives`, `PlanAlternative`, `review_engine` and `review_plan_switch` switch paths).

- **Which command or plan a part gets is one decision (bd fix-o1lg.1; behaviour change).** `fastworkflow/_workflows/plan_metadata_extraction/resolution.py` `part_intent(workflow, text, *, pinned=None, exclude=())` decides it for the turn's gate, the review's draft, look_up and add, the part's run and the no-agent plan list: a pinned plan, then an exact command name, then the detection chain (`intent_detector.detect`) over the execution AND planning commands of the part's scope path (the global scope asked last when the walk does not ask it), so the local model decides when it is confident and the remote decision model (Jev) only when it is unsure. Memoised per part and turn: at most one matcher call per part per turn; the run no longer asks the planning matcher again. Deleted from `review_proposals`: `fast_path_for` and `FastPath`, `_candidates_for_subtask` (the larger of the local and Jev scores per name), `build_round1_proposal` (a planning command preferred among the top three), `plan_alternatives` and `PLAN_AMBIGUITY_CLEAR_MARGIN`, `confident_pick`, `confident_means`, `jev_says_none`, the `none_veto` of `draft_round1_proposal`, `planning_match`, and `workflow_agent.PLANNING_NONE_VETO_THRESHOLD` (fix-6lpc's veto, which corrected the merge). What a draft binds is `resolution.draft_choice`: a match always; an ambiguity's best candidate, with the other plans of the ambiguity as its alternatives, only where somebody reviews the draft; a draft nobody judges (no agent model, or an unreviewed agent-mode draft) leaves an ambiguous part ad hoc. A match has no alternatives. Measured on a new held-out fixture (`tests/planning_test_sets/library_plan_intent_heldout.json`, 137 parts; `scripts/experiments/score_plan_intent.py`, recorded answers): requests a plain command answers given a plan fell from 26 to 0 (no agent model) and from 30 to 0 (agent model), right plain command from 15/42 to 42/42; plan requests 60/60 to 58/60 (two requests a confident local plain command now takes). Also: old saved sessions are no longer converted (`parameter_clarification.loop_from_old_state` deleted; owner ruling). Tests: `tests/test_plan_intent_fixture.py`, rewritten `tests/test_unreviewed_none_veto.py`, and the matcher seam of the review tests is now `script_matcher` / `script_plan` (`tests/conftest.py`).

- **Plan review: a sub-task that matches several planning commands names the other plans, and a reply switches to one (fix-1ma8.1; additive, an unambiguous review is byte-identical).** The matcher already returns a probability for every planning command in the call that picks the winner, but the review showed only the winner. A sub-task's match is now AMBIGUOUS when another planning command scores at least `run_listing.UNSURE_CANDIDATE_FLOOR` (0.2) and the chosen one leads it by less than `PLAN_AMBIGUITY_CLEAR_MARGIN` (0.5); scores are the draft's own (the heads' planning-label scores and Jev's per-option probabilities, the larger for a name, so a head that prefers another plan needs no separate signal). Measured on a private 840-task-run benchmark (1,197 matched sub-tasks, Jev's recorded probabilities): a runner-up at or above 0.2 in 3.8% of sub-tasks; the 0.5 lead marks 3.0% of sub-tasks and 4.1% of the reviews, keeps 36 of the 45; 10 of the 36 were in a run that lost checklist lines (against 4% of all matched sub-tasks of reviewed turns) and 11 of the 36 had a review whose first reply was not an accept (31%, against 13%). At most 3 alternatives, best first, each with the first line of the planning command's description (`SubTask.alternatives`, a hidden field that is absent from the serialized item when empty). Shown: one line under the step in the review text (`   other plans that fit: B, C; reply "step N: use B" to switch`), `alternatives` on the payload item, a `switch_plan` entry at the end of `reply_forms`, and a select on that step of the chat page's review form (it submits `change N: means=planning:B`). The reply reader (`fastworkflow/review_plan_switch.py`, no model) reads `use B`, `step N: use B`, `use B for step N`, `switch step N to B`, `switch to B` and `B instead` after the accept and decline cues and before the typed-edit reader, as the means edit, which re-drafts that step's inputs; the plan it replaced becomes an alternative, so it can be switched back. A name is matched against that step's alternatives first (exact ignoring case, spaces, `_` and the context prefix, then, for a one-word name of at least 4 characters, the intent clarification's fuzzy rule `find_best_matches` with its threshold; a sentence is never fuzzy-matched), and two ambiguous steps with a reply that does not say which are asked again, naming both. Follow-up (a five-round benchmark showed that falling through to the free-text reader for any other name was wrong: it folded "use NAME" into a neighbouring step's goal, or guessed a plan for "switch step 7 to the plan that shows ..."): a clause in an explicit switch form is decided in `review_plan_switch.split_switch_clauses` and never reaches the free-text reader. `step N: use NAME` is always such a clause; `use NAME for step N` and `switch step N to NAME` are when NAME is one word or resolves (`use the 2024 data for step 3` stays a goal edit); the no-step forms (`use NAME`, `NAME instead`) only for the one ambiguous step and only when NAME resolves. NAME may be any planning command or execution command of the workflow, not only a listed alternative (exact, ignoring case, spaces, `_` and the context prefix; no fuzzy match over the whole workflow); it is applied as the same `change N: means=planning:NAME` (or `plain:NAME`) edit, on a step with or without alternatives. A name that resolves to nothing or to several asks again, naming the step, listing its alternatives if it has any and saying how to pick; a step that does not exist asks again naming the valid steps; the plan is unchanged. A reply of several clauses (`;` or new lines) is read clause by clause: switch clauses there, typed edits by the typed parser, any other clause by the usual reader, which sees only that remainder (when the remainder is not read as edits, that question is asked and the switch is not applied either). A request to change which plan a step uses without naming one (`switch step 7 to the plan that shows ...`, `use a different plan for step 3`, `wrong plan for step 2`) is asked again, listing that step's alternatives (a step with none lists nothing: the draft keeps no shortlist, and rebuilding one would call a model); `change step 3 to show the nesting too` is still a goal edit. Defence in depth: a reader (or typed) goal that is, or ends with, a switch clause copied from the reply is refused and the user asked again (`review_proposals._guard_switch_clause_in_goal`, `review_engine._refuse_switch_clause_in_goal`). A clause whose plan name is followed by a sentence (`step 7: use NAME. I want its members ...`, also `, `, `!`, ` - `, `because`, `so`, `since`, and `use NAME for step N. ...`) takes the leading single token as the name when it resolves exactly (an alternative or a command of the workflow; no fuzzy match); text after it that holds no step reference, `=` or edit verb is an explanation and is ignored, otherwise it is read as a clause of its own like any `;` remainder (merged, or asked about with nothing applied); a leading token that does not resolve asks again. The accept path is unchanged. Tests: `tests/test_plan_switch_clauses.py`. An unattended turn runs the winner; the ambiguity is on the turn span as `subtask_plan_ambiguity` (`fw.turn` v5, `SPAN_CONTRACT_VERSION` 46: step number, winner, alternative names; no text, no scores). Nothing is written to the utterance cache, the plan head is not changed and no model call is added. Not built: a "wrong plan" decline that re-drafts with the winner excluded. Tests: `tests/test_plan_match_ambiguity.py`, `tests/test_chat_review_plan_switch_dom.py`.

- **CLI: stop waiting on a turn, a per-turn limit, and `//cancel` (fix-1ma8, audit gaps a to c; additive, with no Ctrl-C and no limit the turn path is unchanged).** (1) Ctrl-C while `fastworkflow run` waits on a turn returns to the prompt with one line; `--timeout_seconds N` (default off; the name and meaning of the server's request field, a delivery deadline that does not abort) does the same after N seconds. The engine cannot abort a turn in flight (the server's turns engine does not either: "wait-or-defer, never wait-or-abort"), so the guarantee is: the wait ends; what the turn already put on the transport queues is cleared and what it puts later is dropped, so nothing of it reaches a later prompt; the turn runs to its end in the worker thread (its commands' side effects happen); an abandoned turn that reaches `ask_user` is ended through the engine's own `CommandCancelledError` (the failed "Command cancelled" output, suspension cleared) instead of blocking for an answer nobody wants; the next message is queued and runs after it, in order; Ctrl-C at an idle prompt exits as before. When the turn had just ended (its sentinel already queued) nothing is abandoned and the result is shown. (2) `//cancel` cancels what is pending with the replies the chat page's Cancel sends: a blocked `ask_user` question ends through `CommandCancelledError`; an open review or approval is answered `no`; an open command (missing value, ambiguous command) is answered `abort`; "Nothing to cancel." otherwise. A plain `cancel` or `abort` typed at a prompt is not intercepted (the review's decline set and the abort command already accept those words). New `fastworkflow/turn_gate.py` (`TurnGate`, `GatedQueue`, `MessageQueue`: the three `ChatSession` queues; one uncontended lock per put), `fastworkflow/run/turn_wait.py` (`wait_for_turn`, the loop `run` always had plus a clock comparison per tick and no watcher thread), `ChatSession.abandon_turn`, `.cancel_pending_prompt`, `.interrupted_turn_running`; the worker loop also ends a cancelled question inside a startup/queued `Action` instead of dying. An abandoned turn that ends with a pending approval, review or open command has it cancelled in the worker (`ChatSession._cancel_unseen_prompts`, classified by `_pending_cancel_reply`, the same classification `//cancel` uses) before the next queued message is taken; its cancel output is dropped and the next prompt says a question from the interrupted turn was cancelled. The messages say the turn is still finishing and its commands may still take effect. Also fixed: a `yes`/`no` reply to an approval or review (and `/approvals`) put no output or sentinel on the queues, so the CLI waited forever (`ChatSession._deliver_if_unsent`; the FastAPI path does not use `ChatSession._run_workflow_loop`). `fastworkflow run` end to end over piped stdin is tested. Not done: printing structured data and conversation resume/listing in the CLI. Tests: `tests/test_cli_turn_control.py`.

- **Progress of a running turn over MCP: polling made first-class (fix-1ma8, audit gap 5; additive).** MCP `notifications/progress` cannot be delivered before the result with the installed mount: fastapi-mcp 0.4.0 runs `StreamableHTTPSessionManager` in JSON-response mode (`mount_http` has no switch), where the SDK drops every notification of a request, and even in SSE mode its `handle_fastapi_request` buffers the whole response and returns it after the request ends (checked on a scratch app; details in `run_fastapi_mcp/README.md`). Nothing was hacked around the library, so no notification is sent and a call with a `progressToken` is answered as one without. Instead a running turn read with `get_turn` / `get_latest_turn` (and the 202 body of an early-returning `invoke_assistant`) carries a new `progress` object `{elapsed_seconds, frames, last_frame?: {type, seq, label?, seconds_ago}}`, read from memory with no store, session or model work: for an `invoke_agent` turn the latest `progress`/`trace`/`timeout` frame (label from the frame's stage label or `<direction>: <command_name>`, never answer text), for an `invoke_assistant` turn only the elapsed time. It is declared in `TURN_RESULT_SCHEMA`, absent once the turn is done, and the final result is unchanged. `TurnStreamChannel.on_frame` feeds `turns.note_stream_frame` (failures swallowed). The `get_turn` and `invoke_agent` tool texts say so. Measured (median of 65 model-free `invoke_agent` calls through `/mcp`): 57.8 ms without a progress token, 58.2 ms with. Tests: `tests/test_mcp_progress.py`.

- **MCP tool results: `structuredContent`, an error flag, an output schema, strict arguments and agent-facing text (fix-1ma8; additive for text readers, no HTTP response the chat page reads changed).** `run_fastapi_mcp/mcp_specific.py` mounts a `FastWorkflowMCP` (a fastapi-mcp subclass overriding its one execution seam). (1) Every tool result keeps its text block unchanged and adds `structuredContent`: the parsed body for a JSON route, `{"result": [...]}` for a list body, and for `invoke_agent` (whose text is the NDJSON/SSE stream) the terminal frame as the same turn object `invoke_assistant`, `get_turn` and `get_latest_turn` return, with `turn_key` the execution key and `logical_turn_key` the stored one; only the last frame is read, so the cost does not grow with the traces. The four turn tools declare that object as `outputSchema` (`TURN_RESULT_SCHEMA`). The two HTTP body shapes are deliberately not unified on the wire (the chat page and HTTP clients read them); `structuredContent` is the one shape. (2) `isError` was already true for an HTTP status of 400 to 599; it is now also true for a turn that ended in failure, defined on existing fields only: an `error` on the turn object (the execution raised, or a stream ended with an `error` frame or no terminal frame), `status` `failed` (a raised turn, or `max_iters_exhausted`), or `exec_state` `lost`. A `running` turn, `awaiting_user` (a question, review or approval), a completed turn with `success: false`, and `cancelled`/`abandoned` are not errors. An HTTP error result gains `structuredContent {http_status, detail, ..., retry_after_seconds?}`. (3) A call with an argument the tool does not list is refused before anything runs, naming it and listing the accepted ones (until now it was ignored). Two schema defects found on the way are fixed: `end_session` advertised no `discard_pending` argument although its route and text have it (the generated schema dropped the optional body), and an explicit `null` for an optional argument (`use_dspy_cache`, `approval_token`) failed the SDK's check because of a leftover `type` beside `anyOf`. (4) Tool and argument descriptions are rewritten for an agent reader (what the tool does, when to call it, the `running` / `awaiting_user` / `done` states, how to answer a pending question, how to recover a timed-out turn, the session limits) and no longer carry internal ticket ids or probe prose; the HTTP OpenAPI text is unchanged. The stale claim that MCP prompts (`format-command`, `clarify-params`) are registered is removed from the READMEs and the spec summary: none ever was. Measured over the in-process MCP route (median of 40 calls, before then two runs after): `invoke_assistant` 22.7 then 23.2 / 22.8 ms, `invoke_agent` 64.4 then 65.4 / 64.4 ms, `get_latest_turn` 3.76 then 3.93 / 3.82 ms, `liveness_probe` 2.08 then 2.18 / 2.10 ms: within noise (at most 0.2 ms). New `tests/test_mcp_structured_results.py`.

- **Plan review: plain-English corrections are applied to the steps the user named (fix-1e5c; the accept path and every typed edit are unchanged).** In a benchmark of 840 runs, 38 plan reviews began with a correction in plain English and 22 of them were misapplied at least once. Four mechanisms, each generic. (1) `drop steps 2, 3 and 4 and keep only step 1` removed step 1: the one step-number reader (`_item_validation.extract_step_numbers`) kept only the first and last of a list of three or more (`2, 3 and 4` read as 2 and 4), and a step named in `keep only step 1` or `steps 1 to 4 are fine` counted as named, so a drop of it was never refused. The new `review_step_refs` module reads step lists whole (`2, 3, 4 and 6`, `2 and 3 and step 5`, `5 to 9`) and what a reply does with them (`removal_reading`: steps to remove, steps to keep, a negated removal is a keep). The reader's guards and the routed `drop` command are held to it: a drop of a step the reply says to keep is not applied, a step the reply asks to remove that the reader or the parameter extractor left out is dropped, and a reply that removes and keeps one step asks again with the removal it read shown back. No new model call. (2) `Item N is named by more than one edit`: a merge plus a drop of one of its members (`make it a single step ... and drop the second step`), or a drop plus a clear of an input of the same step, are now one removal; an input both cleared and set by one reply is set. (3) The reader's completion of a name cut short (`_repair_goal_value`) replaced the whole goal sentence by the quoted name when the two shared two words; it now replaces only the run of words that is the cut-short name, in a change, a merge or an added step, and an input the reader cut short (`name=Kline`) is completed from the quoted name. A quoted piece of a name (`'Sons Ltd' is part of the name in step 1`) is no longer set as the whole name: the user is asked for the whole name. (4) The plan `edit` command passed the reader a stand-in proposal when the session gave it no reader, which silently turned every deterministic guard off; it passes the real proposal. A reply that names its steps by number is no longer asked `Which of the N steps do you mean?` by the `drop` slot. A removal verb is a certain removal only when the step list is its whole object (`remove step 2 entirely`, `... because it repeats`): `remove step 2's filter`, `cut step 3 down to five`, `skip step 2 if ...`, `drop step 3 into step 2` and `remove the date filter in step 2` are uncertain, never overridden, and a reader drop of such a step asks again; `do not keep step 2` is a removal. New: `fastworkflow/review_step_refs.py`, `review_proposals.reconcile_dropped_items`, `certain_removals`. Tests: `tests/test_review_plain_corrections.py`.

- **The MCP session manager starts with the server, not on the first request (`run_fastapi_mcp/mcp_specific.py`, `__main__.py` lifespan; no API change).** fastapi-mcp 0.4.0 starts its StreamableHTTP session manager lazily and marks it started before its task group exists, so several clients sending their first `/mcp` call at the same moment could get HTTP 500 ("Task group is not initialized"; the call never reached the workflow). The lifespan now starts it (`start_mcp_transport`, which returns only when the task group exists) before readiness and stops it at shutdown (`stop_mcp_transport`), so entering the lifespan of the same app again also works. It relies on `FastApiMCP._http_transport`, `_ensure_session_manager_started`, `shutdown` and `_session_manager._task_group`; a library version that changes them fails at startup with a message naming what moved. Tests: `tests/test_mcp_concurrent_first_requests.py`.

- **A hard bound on live sessions, for every workflow (`run_fastapi_mcp/utils.py`, `turns.py`, `session_notices.py`; additive, defaults keep a deployment within its soft target unchanged).** The live-session cap `MAX_LIVE_SESSIONS` (50) is only a soft target: a session whose state cannot be checkpointed (no `get_state` hook, `EPHEMERAL`, a failing hook) is never evicted, so a client opening many channels grew the process without bound. Now: (1) `MAX_LIVE_SESSIONS_HARD` (default 4 x the soft target; below the soft target is a startup error) is a ceiling enforced in `ChannelSessionManager.admit_new_session`, called by `ensure_user_runtime_exists` before anything is built: at the ceiling it retires an idle evictable session, then expires an idle pinned one, then refuses with `503`, `Retry-After: 30` and a message naming the ceiling (the MCP tool result carries the same). Concurrent creators are counted (`_creating`), and a "nothing releasable" verdict is reused for a second so a refusal storm does not rerun every `get_state` hook under the lock. (2) Expiry only above the soft target: `PINNED_SESSION_IDLE_TTL_SECONDS` (1800, 0 never) for a pinned idle session, `PROTECTED_SESSION_IDLE_TTL_SECONDS` (86400, 0 never) for one with a pending question (discarded as `end_session(discard_pending)` does); never a session with a lease or running turn, and never a registered experiment channel (its attempt claim lives on the session and a pinned session has no checkpoint to carry it; reported as `experiment_exempt`). The 5-minute reaper also expires under pressure, at most 25 per pass. (3) Never silent: expiry, and `end_session` of a pinned session, record a notice (`run_fastapi_mcp/session_notices.py`, one small file per channel, reaped after 30 days, fails open); the channel's next session shows `session_notice: {code: "context_dropped", reason, message}` once in its first turn result (all turn endpoints, the `output` frame of `invoke_agent_stream`) and in the `/initialize` response (`InitializeResponse.session_notice`). (4) `/probes/readyz` `sessions` adds `hard_ceiling`, `at_ceiling`, `refused_total`, `expired_total` and both TTLs; `?sessions=true` adds `pending_protected`, `experiment_exempt`, `trace_events_dropped`; the pod is not made unready at the ceiling. Startup logs all four limits and their sources. (5) Per-session retained data: the trace queue is now `BoundedTraceQueue` (10,000 events, oldest dropped and counted; the server drains it every turn, but a turn that failed before its drain left its events behind, now discarded in `_run_turn`), and the deferred conversation-window trim is capped at twice the window. New env: `MAX_LIVE_SESSIONS_HARD`, `PINNED_SESSION_IDLE_TTL_SECONDS`, `PROTECTED_SESSION_IDLE_TTL_SECONDS` (process environment first, as `MAX_LIVE_SESSIONS`). The shipped examples that had no hook now have one: library (`Library`, `Book`, `Member` snapshot the whole library as JSON-native data with `get_locator`/`find_by_locator` for a current book or member; `Record`, which no object is an instance of, returns `fastworkflow.EPHEMERAL`) and messaging_app_4 (`User`, `PremiumUser` snapshot the user with its room; the `ChatRoom` root already had one). `Library.to_state_dict`/`from_state_dict` and `Library(populate=False)` are new. `fastworkflow train` retrains nothing because of these files: its decision reads per-command source hashes (`selective_training.compute_command_fingerprints`), and a context class file is not a command file (the training signature of the library computed before and after the edit is identical). They do change the commands-source fingerprint (snapshots, checkpoint namespaces) and the workflow content hash, as any edit under `_commands` does. tests/todo_list_workflow keeps its hook-less `TodoItem`/`TodoList` on purpose. Tests: `tests/test_session_hard_ceiling.py`, `tests/test_example_context_hooks.py`. Tests: `tests/test_session_hard_ceiling.py`.

- **`use_dspy_cache` on the turn endpoints no longer changes the process-wide DSPy cache by default (`run_fastapi_mcp/utils.py`, `server_memory.py`; the chat page and its request are unchanged).** One client sending `use_dspy_cache=false` rebuilt the shared cache, changed every other channel and made `/probes/readyz` return 503 "DSPy memory policy drifted". Now a server ignores the argument and names a disagreeing value in the `X-FW-DSPy-Cache` response header; the cache is chosen at start with `FW_LM_CACHE=0`. New `fastworkflow run_fastapi_mcp --allow_request_cache_toggle` (default off, forwarded by the CLI) restores the toggle for a single local user: `run_chatbot`'s launcher passes it, so the page's "LLM cache" checkbox works as before, and the toggle re-applies the memory bound so readiness stays ready (a cache a request turned off is shown as `checks.dspy_response_cache`). Retained-turn memory was already bounded (chat turns are dropped on completion, startup turns capped at 20); a test now pins it.

- **An explicit way to end a session: `POST /end_session` and the MCP tool `end_session` (fix-1ma8; additive, nothing the chat page reads changed).** The server kept one live session per channel and nothing ended one: a session whose state cannot be checkpointed (a context class with no `get_state` hook) is pinned and never evicted, and the cap (`max_live_sessions`) is only a target, so a client that opens a channel per job grew the process without bound. `end_session` acts only on the token's own channel, is refused (409) while a turn runs or the channel is in use, and while a question is pending (named in the message) unless `discard_pending: true`; it saves a checkpointable session as an eviction does (the channel restores it on its next call), drops a pinned one's in-memory state (`pinned: true` in the response), evicts the app workflow from the process registry, releases the channel's turn pointer, and touches no stored turn, conversation or observability row (`get_turn` and `get_latest_turn` keep working). Idempotent (`ended: false`). Response `{channel_id, ended, pinned, pending_discarded}`. `ChannelSessionManager.end_session` does the work under the manager lock; `WorkflowExecutionContext.discard_live_pending` is a new public method (approvals, the suspended turn, review and CME run; parked state of other conversations is kept); `TurnRegistry.release_channel` forgets the pointer. `evict_live_session` no longer cancels pending approvals (its docstring said it did not; it delegated to `remove_session`, which does). The tool descriptions of `invoke_assistant` and `invoke_agent` say to call it when finished. Visibility: `GET /probes/readyz` carries `sessions: {live, max_live, over_target}` (and, with `?sessions=true`, `pinned`, `busy_unassessed`), and startup logs one WARNING naming the context classes without a `get_state` hook (read statically from the context model; `checkpoint.startup_pinning_warning`). Eviction policy unchanged: no automatic eviction of pinned sessions, no refusal of new channels at the cap. Tests: `tests/test_fastapi_end_session.py` (including 60 channels opened and ended in a row on a pinned workflow, live count never above 1).

- **Long turns over MCP: an explicit time-out policy and a way to read a turn back (fix-1ma8; additive, no HTTP response the chat page reads changed).** (1) Measured: a turn of 11.5 s completes in full through the mount's own `/mcp` route on both tools (`invoke_assistant` and `invoke_agent`); the 10 s `timeout` of fastapi-mcp's default in-process client is not enforced by httpx's ASGI transport, so this was not a defect. `setup_mcp` now passes its own client (same transport and base URL) with no read time-out (`MOUNT_HTTP_TIMEOUT`), so the behaviour no longer depends on that library detail. (2) The tool descriptions of `invoke_assistant` and `invoke_agent` say what `timeout_seconds` does (a delivery deadline: 202 with `turn_key` for the assistant, a non-terminal `timeout` frame for the agent; the turn keeps running), and that resubmitting re-runs a finished request; the HTTP OpenAPI text is unchanged. (3) `get_turn` is no longer excluded from the MCP tools, and a new `GET /turns/latest` (`get_latest_turn`, the caller's running turn, else its last stored one) serves a client that timed out before it had a key. Both are scoped to the JWT's channel exactly as `GET /turns/{turn_key}` is. Edited: `test_compose_operations_are_excluded_from_mcp` no longer lists `get_turn` as excluded (it asserts it and `get_latest_turn` are not). Tests: `tests/test_mcp_long_turns.py` (one test runs two 11.5 s turns at once, about 24 s for the file).

- **An unreviewed draft no longer binds a planning command Jev is sure does not fit (fix-6lpc; agent mode only).** New bar `workflow_agent.PLANNING_NONE_VETO_THRESHOLD = 0.9`, separate from the 0.5 match bar. When a turn's sub-task draft runs with no one to answer its review (an unattended turn, a host surface) and an agent model is set, a sub-task whose picked means is a planning command is left to the agent (`from_scratch`, as a sub-task with no candidates always was) when Jev's "none" is at or above 0.9. Not applied: when the draft is shown to a reviewer (the chat page, an attended review), to a sub-task that names a command, to a plain command pick (Jev judges planning commands only), with no Jev verdict, or in no-agent mode (unchanged, its own 0.5 rules). No new model call: it reads the match the draft already makes. The bar comes from recorded runs: a veto at 0.5 was a coin flip, no bound sub-task had "none" at 0.9 or more, and out-of-scope requests had 0.93 to 1.00. `review_proposals.jev_says_none` takes an optional `bar`. Tests: `tests/test_unreviewed_none_veto.py`.

- **A context built without `__init__` reads safe defaults (fix: pre-commit sweep).** The per-turn attributes added to `WorkflowExecutionContext` (`_turn_not_run`, `_turn_plan_previews`, `_ask_user_rounds`, `_turn_parts_of_answer`, `_last_nested_stop`, `_pending_refusal_request`, `_planning_command_refusal_request`) have class-level defaults, so a restored or test-built context finalises a turn instead of raising `AttributeError`.

- **BREAKING: the results panel draws parallel lists as a table only for a declared `table` group (fix-sush step 3, `254-chat-results.js`).** A field's name never decides how it is drawn, and the panel no longer reads a field named `total`: an output with an integer `total` beside list fields of one length used to be drawn as one table (a "page"); now those lists are plain lists unless the author puts the fields in one `ui` `table` group. A list of records is still a table, a declared group is unchanged, and a count/total scalar is shown as before. `fastworkflow command preview <folder> --check` prints a notice for each output with two or more visible scalar lists in no group, which is the list of outputs to hint. The answer card and the composer still infer a table from a `total` plus equal lists until step 4 (it changes what the model reads and needs a live measurement).

- **A capability turn carries its data to HTTP and MCP clients (additive; the answer text is unchanged).** The `capabilities` artifact was only on the final answer's own output, which is not in `TurnOutput.command_outputs`. The turn now also carries it on one synthetic output marked `framework` (the chat page leaves such outputs out of cards and the results panel), with or without an agent model. The synthetic output that carries `fw_not_run` for a turn that ran no command is marked `framework` too. Edited: the assertion at the end of `test_a_capability_turn_carries_the_data...` (added earlier the same day, pinned the interim state).

- **No-agent mode, second part: a decision step pauses the command and resumes on the reply (fix-rn9p; additive, with `LLM_AGENT` set nothing changes).** In no-agent mode a conditional step `when("<condition>", run(...))` whose inner step code can run, and a command step that leaves out one plain required value (a `str`, `int`, `float`, `bool`, enum or literal not found from a listing), no longer end the command with `fw_not_run`: the turn is `awaiting_user` (no new status), the pending entry has `question_kind: "phase_step"` and `artifacts.phase_step` (`command`, `phase`, `step`, `decision`, `decline`, `asked`, `done`, `facts`, `remaining`) and the answer is a code-written question. The pause is the phase's state (`_phase_run["decision"]`, serialised like any suspension, nested frames and multi-part turns included) plus a minimal agent stash, so no ReAct resume variant was needed. The reply is read in code (no planner or reply-intent call): `yes`/`no` (also `y`/`true`/`n`/`false`) runs the inner step through `CodeSteps` or skips it; the value is validated as a code-written step is; an unclear reply is asked once more, then the command ends with its stop; `cancel` declines (stop with `reason` "declined"). The phase then carries on from the decided step as a code run of its own (`CodeSteps(resume=...)`), so later decision steps ask again, a free-work step ends the command with the stop (`done` includes what ran after the resume) and `__result__` runs at the end. Asked only on an attended turn of the agent path with no blocking queue; an unattended turn or a `/` command keeps the structured stop. Existing tests edited: three no-agent tests that pin a stop at a conditional step now run unattended (`tests/test_no_agent_mode.py`: the open-step, real-last-output and nested-stop cases). Tests: `tests/test_no_agent_decisions.py`.

- **No-agent mode: run without the ReAct agent model (fix-rn9p, first part; additive, with `LLM_AGENT` set nothing changes).** For a server whose client is itself an agent. `LLM_AGENT` unset or empty (`no_agent_mode.agent_model_specified()`, the one decision; execution contexts that serve users opt in with `detect_no_agent_mode`: `ChatSession` and the FastAPI session) puts the framework in no-agent mode: routing, parameter extraction, `LLM_PLANNER` and Jev stay in use; Jev is mandatory (neither set is a startup problem through `command_preview.load_problems(folder, runtime=True)`, which `fastworkflow run` and the server use and `fastworkflow train` does not; `fastworkflow run` prints it and exits 1, the server refuses to start, with a RuntimeError at startup; both only at the real entry points, never for a context a test builds). (1) No agent LM is built and no model step is taken (`fastWorkflowReAct.no_model`): code steps run, and a planning command that code cannot go on with ends as a COMPLETED turn carrying a stop, the artifact `fw_not_run` (a list, one entry per stopped part: `kind` `not_run_here`, `command`, `phase`, `open_step` `{number, text, kind, reason}`, `done`, `facts`, `remaining`, `suggested_plan`, `means`, `sub_task`; plus `later_phases`, `nested`, `reason` when they apply). It stops at a conditional, prose, missing-value, failed, no-path, page-cap or extra step, when a nested planning command stopped, and where the answer needs a model (no `__result__` text). The turn's answer is a short explanation written in code. (2) An ad hoc part (no planning command matched, a refused or unbound bound command) is not run: it comes back as the same entry with the planner's text plan (`LLM_PLANNER`, as before) as `suggested_plan`; a multi-part turn runs the runnable parts and is COMPLETED. (3) A single plain command inside an agent-endpoint turn answers from its own response. (4) The deterministic (`/`, assistant) path runs a planning command that needs the agent to its first open step instead of refusing it up front. (5) `fastworkflow run` with no agent model starts in assistant mode and says so; the server names the mode once at startup. No `TurnOutput` field, no new status, no span change. Suspending on a decision step: second part, below. Tests: `tests/test_no_agent_mode.py`, fixture `no_agent_model` in `tests/conftest.py`.

- **The core carries data; every code-written part of the answer text is also available as data (fix-szfq, additive: no answer text, heading, label or prompt changed, no `TurnOutput` field, no status, no span change).** (1) A capability answer written in code (`capability_answer.py`) carries the artifact `capabilities` (`capability_data`: planning commands with their example, the current context's commands, the other-context lines with the commands they name and the count they leave out), built from the same arguments as the text, on the answer's own `CommandResponse` (the `CommandOutput` that `process_message` returns and the CLI shows). That turn has no entry in `TurnOutput.command_outputs`, so an HTTP client does not receive it: carrying it there needs a new `CommandOutput` in the turn's outputs, which changes `command_outputs` for clients, the results panel and tests, and was not done (open). (2) A part continued after a step-cap hand-off gains `runs` in its `fw_sub_tasks` entry (`[{index, of, start, end}]`, offsets of each run's answer after its "Part i of n:" label, same convention as `start`/`end`; absent for a part not continued); the composer's `PartData` accepts it and never discards parts for a malformed one, and the model's input is byte-identical. (3) The `review` artifact gains `reply_forms`, the replies the shown hint promises (`REPLY_FORMS_OF_HINT`, one table per hint constant); the reply reader's own cues are separate and wider and were not unified with it. (4) `what_can_i_do` gains `output` (`CapabilityListing`) and the artifact `capabilities` (workflow, contexts, per command: name, context, planning flag, approval marker, description, parameters, outputs, sample utterances, as the text prints them); `abort` gains `aborted` (the stage and command it ended) and `you_misunderstood` `command_choices`. They are attached in `command_executor.py`, not in the built-in command files, because the internal workflow's training fingerprint is a hash of those files: no retraining, no change in routing. Tests: `tests/test_cme_replies_as_data.py`, `tests/test_review_reply_forms.py`, additions to `tests/test_capability_answer.py`, `tests/test_subtask_parts.py`, `tests/test_composer_turn_parts.py`.

- **BREAKING: an un-hinted `status` or `state` field is plain text in the chat; a badge needs `format: badge` (`254-chat-results.js`).** A field's name never decides how it is drawn: type and shape give the plain default, and everything else is a hint the workflow author declares. The old rule badged any short string in an `Output` field named `status` or `state`. Add `json_schema_extra={"ui": {"format": "badge"}}` (and `tones` for a known set of values) where a badge is wanted; the bundled examples needed none (their `status` fields carry JSON text or error sentences). The answer card and the results panel share the same drawing. The page rule for tables (an integer `total` beside list fields of one length) still applies and is scheduled for removal, so `fastworkflow command preview <workflow> --check` now prints a `notice:` (never an error; the exit status is unchanged) for an `Output` with two or more visible list-of-scalar fields in no `table` group (`command_preview.check_ungrouped_list_notices`, `planning_commands.ungrouped_list_notices`). The library example's `find_book`, `find_member`, `list_member_loans` and `list_loans` now declare their parallel lists as `table` groups, which changes those four command files' fingerprints (counted as changed at the next training run).

- **A split turn also returns its parts as data; the composer reads them instead of re-parsing the headings (`workflow_execution_context.py`, `run_chatbot/composer.py`, `258-chat-card.js`, `250-chat-artifacts.js`).** The boundary rule is that rendering intelligence lives in `run_chatbot/` and the core carries data. The core joined the part answers into one text with `### N. <title>` headings and the composer parsed them back. Now, for a turn of two or more sub-tasks, the core ALSO records the parts of the turn, next to the answer text, which is unchanged (the CLI output, the HTTP `answer`, conversation history and the chat bubble are exactly what they were). Where: the artifacts of the turn's last command output (key `fw_sub_tasks`, `TURN_PARTS_ARTIFACT`), the entry `TurnOutput.command_outputs` ends with and the one `_describe_pending_prompt` already tags; no `TurnOutput` field, so no contract or shape-pin change. The final answer's own response is not in `command_outputs`, which is why it is not there. A turn with no command output carries nothing, and a turn whose answer was cleaned afterwards (a DSPy field marker removed) carries nothing, because the offsets would be wrong; both fall back to the headings. Shape: a list with one entry per sub-task in order, answered or not: `number` (1-based, the heading's and `CommandOutput.sub_task`'s number), `text` (the sub-task's request, uncut), `title` (the heading's text, from `_heading_title`), `answered`, and `start` / `end` (offsets in the answer text of the part's own answer, the text after its heading; `null` for a part that gave no answer and so has no heading). Offsets rather than the answers, so a long answer is not stored twice: an artifact over the observability store's inline limit (256 KiB) is offloaded to a reference, and a duplicated 150,000-character answer would be. No status is carried (not run, suspended, failed): at that point the core only knows whether a part gave an answer, anything finer would be a guess. A single sub-task turn has no entry. The page forwards the list as `parts` in the compose request when the last command output carries it, draws nothing from it and does not show it in the artifacts panel. `AnswerCardRequest.parts` is optional and lenient (not a list, fewer than 2 or more than 24 parts, numbers out of order or range, text over 20,000 or a title over 200 characters, or offsets that do not match the answer's own heading lines mean "no parts", never a refused request). With parts the composer reads the answered parts for `sub_tasks`, the section cap, the block cap, the token budget and the long-answer excerpt, and the model's input is byte for byte what the heading parse gives for every answer with well-formed headings (pinned for 2, 9 and 7 long parts, titles cut at 80 characters, a part with no answer and a single answered part of a split turn); the diagnostics (`parts=`, `parts_uncovered=`) also count a part that gave no answer, by its request's title. The one intended difference: a part whose own answer holds a line that looks like `### 3. x` is no longer read as a new part. Without `parts` (an old record, another client) the headings are parsed as before. Tests: `tests/test_subtask_parts.py`, `tests/test_subtask_parts_numbering.py`, `tests/test_composer_turn_parts.py`, `tests/chatbot_turn_parts_dom.cjs`.

- **The core names no client, and the compose 422 text is the real limit (`approval_binding.py`, `planning_commands.py`, `run_fastapi_mcp/__main__.py`).** The `/approvals` reply says "on this channel" and "Approvals made on another channel are not shown or revoked here." instead of naming the chat page or the CLI. The OpenAPI 422 descriptions of `POST /compose_answer_card` and `POST /compose_question_form` are built from the composer's own constants (60 results, 150,000 answer characters, 20,000 question characters); the answer card's text said 20,000. The `ui.tones` comment describes the hint's meaning, not the page's files.

- **The answer-card composer keeps two kinds of card part it used to throw away, so fewer cards need the second model call (fix-weih.35, `run_chatbot/composer.py`).** Replaying the saved raw outputs of 28 many-part passes through the validator (no model), the two commonest triggers of the correction call were a `table` block over a result that is one flat record (`not_a_table`, then `stats_empty` / `empty_section` for the section it emptied) and a lead dropped whole for numbers coordinated before one noun. (1) A `table` block whose ref is one flat record (a non-empty dict of single values only) is now shown as a `record` block, counted in `repaired=` and not a drop; it then passes exactly the checks a model-written record block passes (fields that do not exist are ignored, none left shows the whole record). A list, a dict with a list field or unequal lists, a nested dict, an empty dict and a scalar stay dropped as before. (2) The number check reads numbers joined by `and`/`or` or by commas before one noun (`477 and 513 holders`, `4, 9 and 5 holders`, also in number words) as each counting that noun, then applies the existing anchoring rule to each; every one must anchor, so `4 and 9 holders` (9 is no holder count) and `5 and 2 open findings` (5 is not the open count) are still rejected. The answer's own numbers are not read this way. Replay: first passes that still trigger a correction 11 of 15 before, 9 of 15 after (all 28 passes 19, 17); `not_a_table` drops 11 to 7, `lead_number_unanchored` 2 to 1, `empty_section` 9 to 7, `stats_empty` unchanged at 7 (stats over a `.total` that does not exist are not repaired).

- **The answer-card composer and the chat page read the recorded request part of each result (fix-2nsr piece 2, `run_chatbot/composer.py`, `254-chat-results.js`, `258-chat-card.js`).** Each result the page sends may carry `sub_task`, the `CommandOutput.sub_task` of its recorded output (1-based, as the answer's `### N.` headings; absent when the turn was not split, for framework commands and for old records). The request accepts it as optional and bounds it to 1..24 (anything else is "no part", never a refused request). **With numbers:** the union of the heading numbers and the result numbers (a part's title is its heading's, else the code-written label of its first primary result) is used for the log diagnostics only; what the model reads (`sub_tasks`) and the section, block and token caps come from the answer's headings alone, exactly as without numbers, and a section with an empty heading falls back as it did before (the answer's heading at its position, then the command's name); the digest, the prompt and the section order are NOT changed by the numbers (a blind measurement, 3 runs per arm, 2026-11-13, found the model writes worse cards when the part lines, the `part` / `parts` marks and the prompt sentence are in its input, and ordering sections by part was switched off with them, so the model's order stands; the code was removed, not flagged). The part number reaches the composer for diagnostics only: after validation, a part with a primary result that no kept section binds a block to is counted as `parts_uncovered` (it does not trigger the correction pass, whose triggers are the ones it had), and the INFO line gets `parts=N parts_uncovered=M` (and `cross_part=K`, only when K > 0; `parts_uncovered2=` for the second pass). A section with no usable heading takes the title of the part it binds. A block bound to another part than the rest of its section, in a section whose text names its own part's title and not the other's, is only counted (`cross_part_blocks`); nothing new is dropped. **Without any number nothing changes**: the digest, the prompt, the section cap and the INFO line are what they were (pinned against the pre-change digest). The page (`tmPartOf`, `tmCardSelect`) reads `sub_task` from the last page of a result and sends it only when it is a positive integer; over 60 results the selection is fair across parts (the first primary result of every part is kept first, then the rest in send order; supporting results of a part go before it loses its only primary one; this is the page's choice of what to send, not what the model sees), and without numbers it is the first 60 as before. The footer of results not shown is unchanged (it groups by command), and no span attribute was added (the compose span's attribute set is pinned by the span contract; a bump was not worth it). The number rule for a lead or verdict also reads a count the request enumerates: the length of a list of capitalised names in the message or the answer ("Five people on the list need a check: Ann Lee, Bo Chan, ...") supports a number whose sentence names a word said before that list, so "the five exception holders" is no longer dropped as a number nothing states when the user wrote "Five people" and the data gives "holders" to another number (found as the cause of lost leads on a recorded 9-part turn; the qualifier rule and numbers found nowhere are unchanged). Tests: `tests/test_composer_parts.py` (37) and a DOM block in `chatbot_answer_card_dom.cjs`.

- **The composed answer card replaces the text answer in the chat bubble (fix-weih.36, `258-chat-card.js`, `010-style.css`).** A drawn card with an empty lead was appended below the plain text, so the bubble held both (a card with a lead already folded the text behind a "Show full answer" toggle that opened it under the card). Now every drawn card replaces the text: the bubble holds `div.answerCard`, the full answer in a hidden `div.answerRest` (kept in the DOM, nothing reads it but the toggle; the card request body is built from the turn output and is unchanged) and one quiet `button.answerToggle` under them ("Show full text" / "Show summary", `aria-expanded`, `aria-controls` naming the card and the text, outside the footer row), which swaps the two. Its state is per turn (`dataset.cardText`) and survives a repaint, a cached card and the results panel opening and closing; the default is the summary. While composing, and whenever no card is drawn (skipped, failed, not needed, a lead alone over a results panel, a card with no drawable block, Rich answers off, "/" turns, question and approval turns), the text and the notice are as before. Switching Rich answers off leaves a card on screen alone; the next repaint shows the text. Focus that sat in the replaced text moves to the bubble; the log keeps following only if it was following. Tests: new `tests/chatbot_answer_replace_dom.cjs` (`tests/test_chat_answer_replace_dom.py`). Edited expectations in `tests/chatbot_answer_card_dom.cjs`: the toggle's wording, aria-controls and behaviour (it hid and showed the text under the card; now it swaps card and text), and the empty-lead case (it asserted the text stayed and the card sat after it; now the card comes first and the text is hidden behind the toggle). `tmAnswerToggle` in `254-chat-results.js` is no longer used by the card.

- **The compose span carries the answer and excerpt sizes (fix-weih.35, span contract v45, `run_fastapi_mcp/compose_trace.py`, `tracing.py`).** `fw.compose` gains two integer attributes, `answer_chars` (the answer's length) and `excerpt_chars` (the length of the excerpt the model read; 0 when it read the whole answer), the numbers the composer's INFO line already held. Null for a question form. Purely additive: composing is unchanged. `SPAN_CONTRACT_VERSION` 44 to 45; `fw.compose` v1 to v2, no other span changes. Existing expectation edited: `tests/test_span_contract_versioning.py::test_the_aggregate_number_moves_only_on_purpose` now pins 45 (was 44). New: `tests/test_compose_trace.py` (short and long answer).
- **Long answers get a summary card: an excerpt for the model, a 150,000-character limit, and no cap of 8 sections (fix-weih.34, `run_chatbot/composer.py`, `258-chat-card.js`).** The whole answer text was the only unbounded input of the composer prompt (78% of it for a 9-part turn of 41,800 characters) and its limit, 20,000 characters, stopped such a turn from getting a card. (1) Up to 20,000 characters the model is shown the whole answer, byte for byte as before. Above that it is shown an excerpt of about 12,000 (`answer_for_prompt`): the text before the first heading (at most 1,200), every `### N.` heading in order, and for each part its opening cut at a line or sentence end (a fair share, at least 300 characters, what a short part leaves over going to the long ones), each cut part ending with a marker that holds no number and is in no answer; an answer without headings keeps its head and tail. Every check (headings and their cap, numbers anchored, quotes grounded, label words, the "lead adds something" test) still reads the full text. The INFO line gains `answer_chars=N excerpt_chars=M` when an excerpt was used. The compose span carries them from fix-weih.35 (span contract v45). (2) `AnswerCardRequest.answer` and the page's `TM_CARD_MAX_ANSWER` are 150,000 characters (a test pins the pair); the question form keeps 20,000. The "too long" notice fires only above that. (3) The section cap is the number of parts of the request, at least 5, at most a safety ceiling of 24 (`MAX_SECTIONS`; was 8; the page's `TM_CARD_MAX_SECTIONS`), and blocks are 2 per section up to 48 (was 16; `TM_CARD_MAX_BLOCKS`). The output budget scales with the cap: 1800 tokens up to 5 sections, 250 more per section after (9 sections 2800, 14 sections 4050, 24 sections 6550), clamped to the model's own output limit, with the cut-off retry adding 2400 to that. Tests: `tests/test_generative_ui.py` (`test_w34_*`), `tests/test_compose_endpoints.py`, `tests/chatbot_answer_card_dom.cjs`. Edited expectations: the answer limit 20001 to 150001 (two tests), the section-cap cases for 9 and 12 parts (8 to 9 and 12) and `MAX_SECTIONS == 8`, the instruction text for 12 parts, the token budget of an 8-part card (1800 to 2550), and in the page test the too-long cases (20,001 to 150,001 characters) and the section ceiling (9 sections drawn as 8, now 25 drawn as 24). Live check, 5 composer calls of the configured model on a 9-part turn (41,808 characters) and a 14-part synthetic one (74,746): the prompt fell to about 11,000 tokens (answer excerpt 11,374 characters), 9 of 9 sections kept, first pass 6.2-7.1 s, none cut off.

- **The review reader no longer reads "nah" as an accept or "do it" as a decline: plainer cues, and a polarity guard on the classifier's answer (fix-blui, `review_engine.py`).** Measured with the real reader and the trained review head on 74 natural replies: "nah" accepted the plan (the head routes "nah" and "nope" to confirm), "do it" and "no changes, go ahead" declined it, and "k", "go", "run it", "ship it", "lgtm", "all good", "please proceed", "let's go", "make it so", "nothing to change", a thumbs-up, "nope", "I'll pass" and "scratch that plan" were asked again. (1) New deterministic cues decide before the classifier. Accept, as the whole reply (case, punctuation and apostrophes ignored, a thumbs-up emoji counts as a word): do it, run it, ship it, go, go for it, let's go, let's do it, proceed, continue, carry on, make it so, lgtm, all good, that works, works for me, fine, k, kk, nothing to change, no change(s), no edits, no objections, no problem, as is, run (it, the plan) as is, don't change anything, and a thumbs-up, each with an optional "please" in front and courtesy or go-ahead words behind ("please proceed", "no changes, go ahead", "all good, proceed", "k thanks", "lgtm 👍"). Decline, as the whole reply: no, nope, nah, nay, no thanks, cancel, stop, abort, don't, do not, don't run (this, it), I'll pass, pass, not now, never mind, scratch that (plan), forget it, reject, decline, discard, with "please", "thanks", "this", "the plan", "for now" and the like behind. A cue is the whole reply, never a word found inside a longer one: "go back and change step 2", "pass the id to step 3", "stop after step 2", "fine, but drop the last one", "no, change the second one to weekly" (a comma after "no" makes "change" an edit) and "do it later" are no cue and go on to the classifier and the edit reader as before. There is no typo matching ("proced" asks again). (2) The polarity guard: when the classifier routes `decline` and the reply holds an accept word, routes `confirm` and it holds a decline word, holds both, hedges or defers ("hmm", "maybe", "not sure", "later", "hold on"), or routes `decline` on a reply that names one step, the reader does not act: the reply takes the path of an unrouted one (the edit reader, then ask again), and the decision's new optional `ReplyDecision.note` (and one INFO log line) says "polarity guard: ...". A reply with no polarity word keeps today's behaviour, and typed edits, the reprompt cap, the messages and the `review_engine.render` hash are unchanged. Existing expectations edited: none. New: `tests/test_review_reply_plain_cues.py` and ten recorded head answers added to `tests/fixtures/review_reply_router_answers.json` (nothing re-recorded).

- **A stalled parameter-extraction call no longer holds a turn for two minutes: its own timeout and retries, and the timeout and outcome on the model-call span (fix-vdaj, span contract v44, `utils/dspy_utils.py`, `utils/extraction_attempts.py`, `utils/dspy_logger.py`).** A recorded 9-sub-task turn waited 118 s of its 130 s to the review form for two extraction calls of about 122 s (the model timeout of 120 s, then a retry that answered at once); the other calls took under 2 s. (1) Every role had the one `LLM_TIMEOUT_SECONDS` (default 120). `get_lm` now reads `LLM_TIMEOUT_SECONDS_<ROLE>` and `LLM_NUM_RETRIES_<ROLE>` (`<ROLE>` is the name after `LLM_`), and parameter extraction has built-in limits of 20 s and 2 retries; every other role keeps `LLM_TIMEOUT_SECONDS` (default 120) and DSPy's 3 retries unless its own setting is given. The timeout is per attempt; DSPy makes up to retries + 1 attempts and waits 1 s, then 2 s between them, so one extraction call waits at most 20 x 3 + 3 = 63 s. A bad value logs a warning and the default applies. (2) `FirstAttemptGreedy` (the extractor's three sampling attempts) stops at an attempt whose call timed out after its retries: it used to start the next attempt, which would have made the longest wait three times as long. A timed-out extraction leaves every parameter not given, the handling of an extraction that raised: a required argument is asked for. (3) The `fw.llm.call` span (v2) records `role`, `timeout_s`, `num_retries`, `outcome` (`ok`, `timeout`, `error`), `timed_out` and `exception_type`. The attempts actually made are not recorded: DSPy retries inside one call and does not expose the count. Existing expectation edited: `tests/test_span_contract_versioning.py::test_the_aggregate_number_moves_only_on_purpose` now pins 44 (was 43). New: `tests/test_model_call_timeouts.py`.

- **The chat page says when it skipped the summary card (fix-weih.34, `258-chat-card.js`, `010-style.css`).** A 41,000-character answer got no card and no word about it: the page does not ask for a card over 20,000 characters (the server refuses one) and the user could not tell a skip from a failure. Where the "Composing answer..." line sits, a short muted line (`span.cardNotice`, reason code in `data-reason` and the tooltip, never in the text) now says why: "Summary card skipped: the answer is too long to summarise." (no request is made), "Summary card unavailable (composer model not configured)." (also on later turns while the page remembers it, with no request), "Summary card could not be composed in time." (server or page timeout), and "Summary card could not be composed." (failed, HTTP error such as 422, or a card the page cannot draw). Silent by design: Rich answers off, a "/" turn, a page not connected, a turn ending in a question, a turn that needs no card (one short result, short answer, the composer's `not_needed`). One notice per bubble: a repaint replaces it, a card clears it, and a restored turn decides again when its record loads. The plain answer and the results panel are unchanged. Tests: `tests/chatbot_answer_card_dom.cjs` (two existing expectations changed: the no_model case now expects the notice, and the not_needed check is scoped to its own bubble).

- **Two optional `ui` options: `tones` for a badge and `currency` for a number (fix-weih.29, fix-weih.30; `field_options.UiOptions`, `planning_commands.check_ui_option` and `check_ui_value_options`, `254-chat-results.js`, `258-chat-card.js`).** `ui: {"format": "badge", "tones": {"delivered": "good", "cancelled": "bad", "on hold": "warning"}}` lets the author tone a badge (good, warning, bad or neutral; keys match case-insensitively after trimming; the author's tone wins, any other value keeps the generic vocabulary); `ui: {"format": "currency", "currency": "EUR"}` makes the page format the number with `Intl.NumberFormat` and that code (a code the browser rejects falls back to today's plain grouped number). Both are validated at load with the other ui options (so `run`, `train`, the server and `command preview --check` report them): `tones` needs `format` badge on a `str`, `Literal` or `Enum` field (or a list), a non-empty map, only the four tones, and on a `Literal` or `Enum` every key must be a declared value; `currency` needs `format` currency on an `int` or `float` field (or a list) and exactly three letters (either case; no list of currencies is kept). Neither is allowed on an `Input`. Additive: a workflow that declares neither renders as before; the options travel in the command schema listing unchanged and the answer card draws them wherever it draws a badge or a currency value. Adding the hints edits a command file, so the command snapshot is rebuilt on its next load (its source fingerprint changes). Tests: `tests/test_ui_tones_currency_option.py`, `tests/test_chat_badge_currency_dom.py`. Follow-up: a currency code held in a sibling field.

- **The chat answer footer is one line (fix-weih.32, `250-chat-artifacts.js`, `255-chat-results-panel.js`, `240-chat.js`, `010-style.css`).** The results link (left), the turn status with "view trace" (centre) and the artifacts toggle (right) share one `.turnFooter` grid row per answer (three slots that keep their place when empty); below 560px they stack in the same order.

- **The plan review form shows "1. Goal:" on one line and a goal that wraps and grows (fix-weih.31, `262-chat-review.js`).** The item number used to sit alone on its own line above the "Goal" label, and the goal was a one-line text field that had to be scrolled sideways. The number now belongs to the goal's label ("1. Goal:", one `<label>` for the field, so a screen reader says the same), the other field labels read "member:" / "member (drafted):" without a number, and the goal is a one-row `<textarea>` that wraps to the form's width and grows with its content up to 12 lines, then scrolls vertically; it is measured on first render, on typing and paste, on a repaint, on a window resize, and when the form's width changes or it is shown after being hidden. A goal is still one line: Enter inserts nothing, and a newline that arrives by paste is turned into a space (anything else is removed as a text input removed it), so the reply the form composes, and what counts as an edit, are unchanged.
- **Free-text fields of the "/" parameter form, the question form and the review form's input values wrap and grow in height too (fix-weih.33, `256-chat-forms.js`).** An ordinary text value was a single-line `<input>` that had to be scrolled sideways. It is now the same one-row `<textarea>` as the review goal (one shared helper, `tmGrowingLine`, in `256-chat-forms.js`; up to 12 lines, then it scrolls). It is still one line: Enter adds nothing and sends nothing, a pasted newline becomes a space, and the reply each form composes is byte-identical. Numbers, dates, choices, secrets (not collected) and the explicit `textarea` widget (4 rows, resizable) are unchanged.

- **Tests: the plan review-reply reader no longer needs a trained head (fix-1gq5).** The reader tests (`test_plan_reply_policy.py`, `test_round1_handoff.py`, the reader half of `test_built_in_vague_replies.py`) run on the head's recorded answer for each reply (`tests/recorded_router.py`, `tests/fixtures/review_reply_router_answers.json`, re-recorded with `pytest --record-review-router`), so they run on a clean checkout and a retrain cannot turn them red. The tests of the head's own quality carry one shared marker, `requires_plan_head`, which skips as before and fails when `FW_REQUIRE_HEADS=1` is set; a drift test fails, naming the replies, when the current head answers a recorded reply differently. Test-only; no product change.

- **BEHAVIOUR CHANGE: an unattended run never uses a channel-surface standing approval, and a channel grant remembers where it was made (fix-d6no).** Affects anyone who answered `always` in the CLI or the chat page and then ran the same workflow unattended (an experiment or benchmark task with `unattended`, or an HTTP call with `unattended: true` on a `--chat_approval_surface channel` server): the gated command now queues with "Approval required ... no user is available" instead of running on the old grant; attended turns are unchanged, and a verified host surface's per-end-user grant still runs. A channel grant is now keyed `channel:<os user>:chat` when made on the chat server's channel surface (set by the server, from `approval_surface_for(..., scope="chat")`, never by a request) and keeps the legacy `channel:<os user>` key in the CLI, embedding hosts and experiment runners, so every stored grant still works there and nothing is migrated or deleted; a chat grant does not apply in the CLI and the reverse (the chat re-asks once). `/approvals` and `/revoke_approval <command>` list and revoke only the caller's own origin and say so. The reply to `always` now names both commands. `Surface` gained an optional `origin` field (default `""`); under `--chat_approval_surface channel` the chat endpoints now pass that surface instead of none, and no token travels with it.

- **Compose calls are traced: one `fw.compose` span per answer-card or question-form call (fix-weih.28, span contract v43).** `POST /compose_answer_card` and `POST /compose_question_form` wrote one INFO log line and nothing to the observability store, so a composed answer could not be inspected next to its turn. The server (`run_fastapi_mcp/compose_trace.py`, called from `_compose_response`) now writes one `fw.compose` span per call, after the response is built and off the request path, carrying the log line's counts and reason codes as attributes (kind, outcome, results, model, duration, digest, sections, blocks, lead, drops, retry, repairs, truncation, the correction pass's tally) and no user text, answer text, result values or raw model output. Both requests take an optional `turn_key` (the page sends the one it holds); the span is filed under that turn, as a child of its root, when the store holds the turn for the caller's channel, and under its own `compose-<id>` trace otherwise. A tracer that fails is one log line, never a failed response. `SPAN_CONTRACT_VERSION` 42 to 43; `fw.compose` v1 is new, no other span changes. The composer's model call does not produce an `fw.llm.call` span (no trace host is bound in the request), and none was added, because that span would store the prompt and the raw output.

- **A recorded command output says whether the command is the framework's own: `CommandOutput.framework` (fix-2nsr).** Additive and optional (`None` = recorded before the field existed, the `ask_user_entry` convention; appended last to the `CommandOutput` field pin). `WorkflowExecutionContext.append_turn_output` sets it from the command's metadata (`CommandMetadata.framework`, the files of `fastworkflow/_framework_commands`): `True` for `review_plan`, `review_proposal` and `review_subtasks`, `False` for every other output, including an application command of the same leaf (qualified by its context). It is a semantic fact about the command, not a display instruction, so any client may read it; it travels in `command_outputs` of the turn record, the stream and turn JSON, and the store unchanged. No span contract or record version changes. The chat page hides an output on `framework: true`, shows it on `framework: false`, and only for a record without the key falls back to the bare names `review_plan`/`review_proposal`/`review_subtasks` (`TM_LEGACY_FRAMEWORK_COMMANDS`, to be deleted with its pin test once no stored turn predates the key). Wording only (no behaviour change): the `ui` option docs and `ui` load messages, the `--chat_approval_surface`, `--cors_loopback_only` and `--parent_pid` help and comments and the compose-operation exclusion comment no longer describe a hint or an option as an instruction to one named page or launcher.
- **A recorded command output says which request part it was produced for: `CommandOutput.sub_task` (fix-2nsr).** Additive and optional (`None` = the turn was not split into several sub-tasks, the output belongs to no part, or the record was written before the field existed; appended last to the `CommandOutput` field pin). `WorkflowExecutionContext.append_turn_output` sets it from the turn's execution state, never from text: the 1-based number of the sub-task the framework is executing, the same number the answer's `### N.` heading gives that part (so after a review edit or drop it is the number in the accepted plan). Outputs of nested planning-command calls carry the number of the sub-task that called them; the plan and sub-task review commands, single-command turns and `/` turns carry `None`; a value already set is kept. Execution is sequential, so the number is read from the turn frame with no shared "current" marker, and it survives a review reply turn, a session save and restore and a resume after `ask_user`. A sub-task that itself covers several items is one part. Nothing reads the field yet: the answer text, its headings, the composer and the chat page are unchanged. A `sub_tasks` list on the turn (index, title) was left as a design note.

- **The chat page keeps no domain vocabulary, no currency symbol (fix-2nsr, `254-chat-results.js`).** (1) The framework-command name list stays (a review checked it): the schema listing omits the framework's commands but also the internal metadata-extraction commands and child-workflow commands, whose outputs are real, and no field of a recorded output marks a command as the framework's, so absence from the listing cannot mean "framework". An additive marker (for example `framework: true` on the recorded command output) would let the page drop the list. (2) Badge tones use only a generic vocabulary: good is success, successful, ok, done, active, enabled, passed, approved, complete, completed; warning is warning, pending, waiting, in progress; bad is error, failed, denied, blocked, expired, rejected; every other value is neutral. Removed: delivered, paid, processing, overdue, cancelled, canceled. In the bundled retail workflow `delivered` and `cancelled` order statuses now show a neutral badge (`pending` keeps its warning tone). (3) `format: currency` formats the number with the browser's locale, grouping and two decimals, and no symbol (it was a hard-coded `$`); the bundled examples declare no currency format, so they are unchanged. A currency code on the hint would let the page show a symbol.

- **The answer-card composer, round 4 (fix-weih.25, `fastworkflow/run_chatbot/composer.py`): false or misleading statements are repaired out, never rewritten.** Each check removes a sentence, label or title and writes nothing in its place; none decides layout or wording for the model. (C1) The state-word rule of round 3 now also covers stat labels and block titles: a label or title carrying open, closed, active, pending, failed, new, critical, high, overdue, unresolved, expired or resolved is kept only when the bound field's key, title or description (or the command or context name), or the answer's words beside that same value, carry the word (a table title also when its rows do); otherwise the code-written label replaces the label (`stat_label_unsupported`) or the title is dropped (`title_unsupported`). (C2) A lead, verdict, heading or block title that claims completeness (all, every, each, entire, complete, full, in full, with a word such as listed, shown, below, table) about a table the page caps (`TABLE_ROWS_SHOWN`, 8 rows, mirrored by the page's `TM_CARD_ROWS_SHOWN` and pinned by a test) or about data that says more exist (a `total` above the rows, a not-shown or remaining count above zero) loses that sentence (`completeness_unsupported`); a sentence that says it is partial (first, only, of, page...) and a table whose rows are all the rows are left alone. (C3) A lead or verdict sentence that points at content ("listed below", "shown below", "as follows", "see the table", "are listed") in a section that ends with no block, or a lead on a card with no block, is dropped (`dangling_reference`); a section left with neither verdict nor block is dropped as `empty_section`. (C4) A block ref that is the digest's hint pasted whole (`r14 fields=[labels]`, `table ref=r14 fields=[a,b]`, `record ref=r2 fields=[notes]`) is read back into its ref and fields and counted as `repaired=N`; a hint that does not parse cleanly is dropped as before. The digest's `show` entry is now a list of ready blocks in the output schema's names (`[{"kind":"table","ref":"r14","fields":["a","b"]}]`) instead of one string, and the instructions say to use the result's `show` entry as the block. (C5) A lead, verdict, heading or title using the composer's own words (digest, payload, schema, ref, a result id such as `r12`, primary or supporting result, stats or table block) that the answer, the message and the data do not themselves use is dropped (`internal_wording`); the instructions and field descriptions no longer say "digest" (the wire format is unchanged), and the worked example no longer teaches "every one of them is listed below". (C6) An evaluative adjective (high, low, critical, severe, elevated, moderate...) within three words before risk, level, score, severity, priority, rating, exposure or impact is dropped with its sentence (`judgement_unsupported`) when no text of the turn (answer, message, data, schema titles and descriptions) uses the adjective at all. (C7) A stat label with a word the field does not carry, that the answer does not put beside this value, and that it puts beside another number ("intrinsic score" over the aggregate score's field) is replaced like any unsupported label. The four new codes are not correctable (no second model call). Known limit: a number in prose that shares a modifier word with another stated number ("four intrinsic defects" where the answer gives 4 to "intrinsic risk level") is still kept.

- **The answer-card and question-form composer lives in `fastworkflow/run_chatbot/composer.py` (fix-2nsr).** It moved out of the core package (it was never released as `fastworkflow.generative_ui`); no behaviour change. `run_fastapi_mcp` imports it for `POST /compose_answer_card` and `POST /compose_question_form`; `fastworkflow.run_chatbot` and `run_chatbot.server` do not import it.

- **The answer-card composer stops dropping true statements and scales with the request (fix-weih.25, owner-approved model comparison).** Six generic changes in `fastworkflow/run_chatbot/composer.py` and `258-chat-card.js`. (R1) A number in a lead or verdict is kept when the answer or the user message states it (digits or the number words one to twenty, the tens and `thirty-two`; "four" is 4) and no OTHER number owns the same noun there, so "167 people" stays over an answer's "167 holders" and "four level-4 defects" stays (a hyphenated token such as `level-4` is a label, not a quantity; a written "one thousand fifteen" is not checked); it is still dropped when the answer gives the noun a different number ("32 findings" over "4 findings", or a swapped noun) or states the number nowhere (reason codes unchanged). (R2) `title_unsupported` drops a title only when it misleads: a title that shares no word with the data is kept unless more than half of its words name another result, command or unshown field of the turn; the strict test stays for a stat block whose label code replaced and for a title holding a digit or a qualifier (`not`, `other`, `hidden`...). (R3) The correction pass now triggers only on a lost section, a lost table, record or stats block, or a lost lead (`CORRECTABLE_DROPS`); a dropped title, verdict, quote or a relabelled stat is only a hint to a pass that runs anyway (`CORRECTION_HINT_DROPS`). (R4) The section cap is the number of `### N.` sub-tasks of the answer clamped to 5..8 (`section_cap`; the card block cap is 10, or 2 per section up to 16), the instructions and the `sections` field description state it, and over the cap sections that show a result no earlier section shows are verified first, the cut ones are the repeats; the page mirrors the ceilings (8 sections, 16 blocks). (R5) A footer line names a group by its command and the record it ran for (the result's context, as the results panel shows it; `global` is no label). (R6) Every result line of the digest carries a `label` (context and the first scalar inputs, at most 80 characters, never a hidden field's key or value) so two results of one command can be told apart; a block bound to a sibling result of the same command whose label the section does not name, when the section names exactly the other sibling, is dropped as `sibling_mismatch` (correctable only when it was the section's only block).

- **The answer-card composer, round 3 (fix-weih.25, `fastworkflow/run_chatbot/composer.py`): reasoning models, repaired refs, merged sections, a clearer digest and one example.** (T1) A model that thinks (`lm.supports_reasoning`) is asked not to (`thinking={"type": "disabled"}` when it takes `thinking` and `reasoning_effort`, else the lowest `reasoning_effort` the LiteLLM model map lists) and gets `REASONING_HEADROOM_TOKENS` (3000) more completion tokens, never past its `max_output_tokens`; a temperature of 0 is sent only when the model map and parameter list allow it (no refused call to learn it); a refused reasoning switch is retried once without it and remembered. A cut-off answer (finish reason `length`, completion tokens at the limit, or JSON cut mid-string) sets `truncated` on the diagnostics and the INFO line and earns one second call with `TRUNCATION_RETRY_TOKENS` more room when the model has it and time allows; never a third call. (T2) A ref the model wrote as an index or as the digest's own nesting (`r1.labels.0`, `r2.page.sample.0`, `r2.sample`) is mapped to the list or page it meant when that is unambiguous, a `record` over a list or over two or more page columns becomes the table, and a stat over a list becomes the count only when its label says count, number or total; counted as `repaired=N`, never a drop or a reason to retry. (T3) A section over the cap, or made only of tables and records, about the result an earlier section shows is merged into it instead of dropped (`merged=N`). (T4) The digest collapses supporting results first and primary results last (a collapsed primary line keeps its label, `show` line, first rows and one or two main values); a list result carries `show: table ref=r2 fields=[a,b]` in the syntax the validator accepts, and the nested `page.sample` record is flat `rows` and `first_rows`. (T5) The instructions say a list is a table by ref, a heading states the finding, a verdict is one full sentence, every numbered part is covered or merged, and carry one worked library example whose ids (`r901`, `r902`) cannot occur; a block that points at them is dropped as `example_copied`. (T6) A state word attached to a counted noun (open, closed, active, pending, failed, new, critical, high, overdue, unresolved, expired, resolved) keeps the number only when the answer or a field's key or title pairs that number with the same word ("five open findings" over 5 findings and 2 open is dropped). (T7) A stat label or title made of the words the answer puts right after that same value ("32 rights" on the field that holds 32) is supported, unless the answer also gives the value a name of the field's own (then the field's title is shown).

- **The answer-card composer repairs six habits of small models that a live check found (fix-weih.25).** (1) A `sections` value nested one list level too deep (`[[{section}, {section}], ...]`, with stray `''` or `[['']]` elements) is flattened by exactly one level before validation; stray empty elements are ignored, a non-dict left after flattening is still a `section_not_a_dict` drop, and the repair is counted as `unnested=N` on the composer log line (`ComposeDiagnostics.sections_unnested`), not as a drop, so it never starts the correction pass. (2) A stat label that shares a word with its field but reverses or drops a qualifier of it (not, no, non, without, excluding, missing, remaining, other, hidden) is `stat_label_unsupported`: "Shown" over a field titled "Holders not shown" gets the code-written label; a plain shortening ("Holders") still passes. (3) A number in a lead or verdict that is the length of a list or page in a result ("60 shown") is anchored when the sentence names a word of that result's field keys, titles or descriptions; an integer field's value anchors the word written next to it; a number found nowhere in the data is still dropped. (4) A stats block whose label the code had to write must have every content word of its title supported by the data (same qualifier test), else the title is dropped (`title_unsupported`). (5) A record block on the chat page shows a list-valued field as its first five items and "+N more", or as a count of items for a list of objects, never as raw JSON. (6) A quote block that holds an identifier (a long hex, UUID or prefixed random key, or the value of a field the schema marks `hidden`) is dropped with the new reason code `quote_identifier` (not correctable: it does not start a second model call).

- **A `ui.table` option declares which list fields of an output are the columns of one table (fix-weih.23).** The chat results view tables several list fields only for a page (an int `total` and lists of one length), which loses the table of an output that is two parallel lists with no total, and of an output with two groups of parallel lists of different lengths. Equal length alone cannot tell parallel lists from independent ones, so the chat never infers columns: an author writes `json_schema_extra={"ui": {"table": "<group>"}}` on each column of a `Signature.Output` (a short identifier, columns in field order). Load check (reported like the other `ui` problems by `run`, `train`, the server and `command preview --check`): a list of `str`, `int`, `float`, `bool` or `Literal` values (`Optional` allowed, lists of objects refused), not with `hero`, a group of at least two fields and at least two not `hidden` (a `hidden` member still counts for alignment and is not drawn). At render time declared groups win over the page rule; a group tables only when every member is present with one length of at least 1, otherwise its lists render as bullets, never trimmed or padded; lists outside any group follow the page rule among themselves. An output with no declaration renders as before. The library example's `library_hours` declares its `branches` and `hours`.

- **BREAKING: `--chat_approval_surface` now defaults to `host` (fix-rkr1, step 2).** `POST /invoke_agent_stream`, `POST /invoke_assistant` and so the MCP `invoke_agent` and `invoke_assistant` tools now apply the same strict approval rule as `/invoke_agent`: an approval-gated command needs a signature-checked JWT with a `user_id` (`--expect_encrypted_jwt`), the approval token is delivered in the response and must come back as `approval_token`. On a server without signed JWTs these commands are refused instead of asked, a plain `yes` on the stream endpoint no longer approves, and an unattended HTTP run on an unsigned server gets the refusal instead of a queued record. Migration: pass `--chat_approval_surface channel` to keep the old behaviour; use signed JWTs for approvals over HTTP/MCP. `fastworkflow run_chatbot` already starts its loopback server with `channel`, so the chat is unchanged. The default changed in the server option, the `fastworkflow run_fastapi_mcp` wrapper, and the in-process default used when the app is built without `main()`. Also fixed: a duplicate request that rejoined a running `/invoke_agent`, `/invoke_assistant` or `/perform_action` turn (after a 202) could replace the approval token's delivery with an empty one, so no response carried the token; the executing turn now hands it to the execution.

- **A head with no foreign command gets a "none of these" class from a built-in list of vague replies (fix-1hs8).** A single-context workflow with no core commands (the built-in plan review-reply head) had no wildcard class, so a vague reply such as "hmm" or "it depends" was routed confidently to the nearest command and the review reader acted on it (accepted or declined). Such a head is now seeded with `BUILT_IN_VAGUE_REPLIES` (nine domain-neutral vague replies, used as written, repeated `BUILT_IN_VAGUE_WEIGHT` = 4 times; recorded under the source `built_in/vague_reply` in its not-here samples, signature and fingerprints, so editing the list retrains the head). Heads that already have foreign seeds are byte-identical, and the internal CME still has no class. The per-context held-out report records `wildcard_seeded_from_built_in_vague_replies` and the weight; the "at floor without a wildcard class" warning no longer applies to such a head. Two runtime fixes: a wildcard answer flagged routed with no command name is treated as unrouted (the reply "nope" raised `ValueError`), and the plan metadata extraction workflow's review record is now its root (a wildcard answer made the CME walk to a parent and raised `AttributeError`). Existing plan heads need a retrain to get the class.

- **The train log's reason for a context with no wildcard class names the usual cause (fix-1hs8).** It used to say only "the internal CME has no wildcard class" for any context without foreign commands; it now also says such a context has no foreign commands to supply "not here" rows. Training is unchanged.

### Upgrading to the design idioms (BREAKING: every old spelling fails the load)

Every BREAKING item below gives the old form and the new one. Each old form fails the load, or the call, with a
message that names the replacement. Nothing is kept for compatibility. The release is 4.0.0 (see "Version").

1. **BREAKING: field options are keys of `json_schema_extra`, one per behaviour (spec 2).** Plain pydantic
   `Field(..., json_schema_extra={...})`; there are no helper functions. Optional TypedDicts (`FieldOptions`,
   `CopiedOptions`, `RecordNamedOptions`, `ConfirmsOptions`) give IDE completion and are exported from `fastworkflow`
   and `fastworkflow.planning`. The keys are named "command text" where they used to say "request".

   | Old (in `json_schema_extra`) | New | Notes |
   |---|---|---|
   | `{"in_request": True}` | `{"copied_from_command_text": True}` | the text must be written in the command text |
   | `{"in_request": {"ignore": [...]}}` | `{"copied_from_command_text": {"ignore": [...]}}` | `ignore` strips before the verbatim match |
   | `{"in_request": "subject"}` | `{"record_named_in_command_text": True}` | names a record already written in the command text |
   | `{"subject": True, "ignore": [...]}` | `{"record_named_in_command_text": {"ignore": [...]}}` | |
   | `{"verify": True}` on a `bool` or `Literal` | `{"classifier_confirms": True}` | |
   | `{"verify": {value: meaning}}` on a `Literal` | `{"classifier_confirms": {"meanings": {value: meaning}}}` | keys must be the enum's values |
   | `{"verify": True}` on a subject `str` | `{"record_named_in_command_text": True, "classifier_confirms": True}` | a model re-picks a record name only with `JEV_API_KEY` |
   | `{"means": {...}}` | `{"classifier_confirms": {"meanings": {...}}}` | |
   | `{"verify": "alone"}` | `{"classifier_decides": True}` | |
   | `{"positional": True}` | `{"command_text_after_command_name": True}` | at most one per `Input`; execution commands only |
   | `{"from": "cmd.field"}` / `from_("cmd.field", ...)` | `{"from_command_output": "cmd.field"}` (or a list) | `from_` is removed: importing it fails |
   | `{"strip": [...]}` with a check | `{"copied_from_command_text": {"ignore": [...]}}` | a field with no check: `strip_words(...)` in a `field_validator` |
   | `{"in_request": False}` on an execution command's `Input` (a value converted from the text, e.g. a date) | `{"converted_from_command_text": True}` | |

   Load rules: an old key fails naming the new key; an unknown key suggests the closest; a value must be `True` or a
   dict of known options (`False` is rejected); conflicting pairs fail (`copied_from_command_text` with
   `record_named_in_command_text`, `classifier_confirms` with `classifier_decides`); a wrong type fails naming the
   accepted types; a hand-written `available_from` fails.

2. **BREAKING: one validation mechanism, `@model_validator(mode="after")` on `Input` (spec 3).**
   `Signature.validate_extracted_parameters` is removed; a `Signature` that still defines it fails the load, and the
   message shows the validator form. `db_lookup` (the field key and the `Signature.db_lookup` method) stays retired.

   | Old | New |
   |---|---|
   | `@staticmethod def validate_extracted_parameters(workflow, command, input)`, `return None`; corrects by assigning `input.<field>` | `@model_validator(mode="after") def <name>(self, info)`, corrects by assigning `self.<field>` and `return self` |
   | `raise fastworkflow.Refused(message, field=...)` from the hook | the same raise, from the validator; a plain `ValueError` is a refusal with no field |
   | `workflow` argument | `info.context["workflow"]` (`None` while a planning command's template renders) |
   | `command` argument | `info.context["command"]` (the qualified name) |
   | the raw text (`command_text` parameter) | `info.context["command_text"]` (`""` while a template renders) |

   - **Upgrade step: declare the moved validator first.** The old order was option checks, then the hook (which could
     correct fields), then the `Input`'s own model validators, then roles. Pydantic runs model validators in declaration
     order, so a validator moved from the hook must be declared before the `Input`'s existing model validators. Otherwise
     an existing check sees the uncorrected value. (In one workflow, a kind-fit check refused before the kind resolver ran.)
   - `field_validator` is for one field's cleanup; `strip_words` stays public. A validator cannot replace "required" or
     `ignore`: a missing required field fails before the validators, and `ignore` strips before the verbatim match.
   - Order now: extraction, option checks, model validators (in declaration order), role checks.
   - Building an `Input` directly (`Signature.Input(...)`) without a context fails with a clear message. In tests use
     `Signature.Input.model_construct(...)` or `Input.model_validate(values, context={...})`.
   - Known false positive: the library's `note_member` refuses a note that repeats the member's own name ("Mia Chen to
     call back"). Document it or reword the note.

3. **BREAKING: command class declarations are decorators (spec 4).**

   | Old (class attribute) | New (decorator, from `fastworkflow`) |
   |---|---|
   | `approval = "mandatory"` | `@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)` |
   | `approval = "delegatable"` | `@user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)` |
   | `writes = True` on `ResponseGenerator` | `@writes` on the execution command's `ResponseGenerator` (on a `PlanGenerator` it is a load error) |
   | `descends_to = "X"` | `@descends_to("X")` |
   | `descend_parameter = "p"` | `@descends_to("X", when_parameter="p")` |

   The old attributes fail the load naming the decorator. Hand-written private attributes fail too. `always` remains
   the reply that records the delegation for that user, workflow and command. `command_directory.json` stores
   `"required_always"` and `"required_until_delegated"` (it stored `"mandatory"` and `"delegatable"`).

4. **BREAKING: the record names (spec 5 and 6).**

   | Old | New |
   |---|---|
   | `workflow.command_context_for_response_generation` (read) | `workflow.record` (the record this command acts on) |
   | `workflow.current_command_context = x` (move) | `workflow.record = x` |
   | `workflow.root_command_context` | `workflow.root` (set once, in the startup command) |
   | `workflow.is_command_context_for_response_generation_root` | `workflow.is_record_root` |
   | `workflow.current_command_context_displayname` | `workflow.record_displayname` |
   | `command_context_object` parameter of `get_parent`, `get_displayname`, `get_state`, `get_locator` | `record` |
   | `instance_label` classmethod on a context class | a property on the record, named by `instance_label_attr` (a context that defines `instance_label` fails the load) |

   `find_by_locator(anchor, locator)` and `from_state(state, workflow, *, state_version)` are unchanged. "Context" still
   names the kind of object (the folder, `base`, `class Context`). `get_displayname` and `instance_label_attr` are kept
   side by side; see `docs/planning_command_file_layout.md`, "Naming a record".

5. **BREAKING: planning step helpers (spec 7, section 13).**

   | Old | New |
   |---|---|
   | `pick("cmd", named=n, item_uid=u)` | `run("cmd", named=n, item_uid=u)` (the command's own input name; `named` and `all_pages` are reserved input names) |
   | `full_list("cmd", ...)` | `run("cmd", ..., all_pages=True)` (reads pages until the rows reach `Output.total`, up to `CODE_MAX_PAGES` = 20) |
   | `listed(...)` | `count_and_names_read(...)` (same behaviour) |
   | `num(...)` | an inline f-string, or `count(n, noun, plural=None)`, which is placeholder-safe (use it rather than an inline `n == 1`) |
   | `from_(...)` | `json_schema_extra={"from_command_output": ...}` |

   Removed: `pick`, `full_list`, `listed`, `num`, `from_` (importing them fails, the first four naming their replacement).
   The paging parameter is `start` only. `run` takes `named=` and `all_pages=`, and a `None` value is left out.
   `when`, `do_not_run`, plain strings and `@phase(...)` are unchanged. Skip a phase with `return []`. `__phases__` is
   only for shared phases.

6. **BREAKING: "request" is "command text" in every author-visible name.** The validation context key `"request"` is
   `"command_text"`. The option keys, load errors and documentation say "command text". Text a model reads is unchanged:
   the sentences the framework appends to field descriptions, refusal messages and step wording. A validator that still
   reads `info.context["request"]` (or `.get("request")`) fails the call with `raised KeyError` and the message
   `'request' is removed from the validation context: read info.context['command_text']`.

7. **Diagnostics: SPAN_CONTRACT_VERSION 41.** The `validation_hook` record keeps its keys (`ran`, `is_valid`, `message`,
   `changed_fields`, `refused_fields`), but its facts come from the `Input`'s model validators: `changed_fields` are the
   fields a validator wrote back, and `refused_fields` the fields of a `fastworkflow.Refused` a validator raised. The
   record is always emitted when the validation step runs, with `ran=False` when the `Input` declares no validator.
   A span recorded before version 41 carries the same keys with the hook's meaning.

8. **Binding and direct construction (`fastworkflow/utils/input_validation.py`).** A `name=value` binding whose value
   fails its type or a field validator is dropped: the binding is no binding, and the LLM extracts the value instead.
   A model validator runs later, with the context. Direct construction of an `Input` without context gives a clear
   error naming the validator, not `'NoneType' object is not subscriptable`.

9. **Retired spellings name their replacement in the load error.** For example:
   `field 'who': in_request: "subject" is removed: write record_named_in_command_text: True`. Each old spelling in
   the tables above fails the load in the same way; the planning-command troubleshooting tutorial
   (`docs/tutorials/planning_commands/15-troubleshooting.md`) shows the load error for each.

**Version.** 4.0.0. The change removes the authoring spellings every workflow uses, so it is breaking for every
planning command and every command with an `Input` validator. The change-control skill reserves a major for the
breaking wire/model collapse, and its version rules predate 3.x (it still says 2.22.2). The major is the team's call
to confirm; the bump follows semver for a breaking public contract.

### Upgrading from copy edges (fact_from, "from" on a facts field) to typed facts (Phase 15)

1. **BREAKING: copy edges are retired (design 2.33, 3.12).** `fact_from` is removed from
   `fastworkflow.planning` and `fastworkflow.planning_phases`, and a `"from"` key on a facts model's
   field is no longer read. A facts field typed with anything other than a command's `Output` or
   `Input` is a load error, and copy fields fail with the message that copies are no longer read.
   `facts.fact_sources` no longer resolves copies, and `check_phase_sources`, `facts_from_runs`,
   `facts_from_archive`, `check_fact_sources`, `FactSourceError` and `AmbiguousFactSourceError` are
   removed. `from_` on a command's `Input` field (a producer edge) is unchanged.

   | Old (copy edges) | New (typed facts) |
   |---|---|
   | `class Found(BaseModel): book_uids: list[str] = fact_from("find_book.book_uids")` with `def find(self, input) -> Found` | `def find(self, input) -> FindBook.Output` (`from .find_book import Signature as FindBook`) |
   | `class Book(BaseModel): name: str = fact_from("show_book.title")` | `class Book(BaseModel): show: ShowBook.Output`, with `@property def name(self) -> str: return self.show.title` |
   | `total: int = fact_from("list_loans.total", min_length=1)` | the field typed `ListLoans.Output`, its check in a `@model_validator` or the phase method |
   | a value a copy computed by name (`count: int = Field(json_schema_extra={"from": ...})`) | a property or method of the facts model, over a field typed with the command's `Output` |

   A value a copy renamed, filtered or joined across commands is a property or method of the facts model,
   over the fields typed with the commands' `Output`s. The previews show each field's origin.

   `over=` may name a property as well as a field: `over="roster.identity_uids"` reads a property of the
   facts model an earlier phase outputs, and `over="roster.unit.identity_uids"` reads a property of a typed
   field's model (`unit`). The load checks each name as a field or a property; a property's return
   annotation types the names after it (a list's items are its element type), and a property without one
   is read at run time, unchecked past it. A name that is neither a field nor a property still fails the load.

2. **Nested lists and joined pages (Phase 15 follow-ups).** A list a nested Output of a facts model holds is
   read by the same rule, each nested model its own column scope: a nested list that drops a row fails the
   check, and a nested complete list passes. A phase's joined pages keep every key an earlier page carries,
   so a key the last page omits survives; a scalar takes the last page that gives a value.

3. **Plain names in types (Phase 15 follow-up).** Types in previews, load messages and the command-error
   text are written by plain class name (`Optional[X]` as `X | None`), with no module path or path-derived
   digest. Two copies of one workflow at two paths give the same previews.

4. **Previews: a fan-out over a property shows its later items.** A `@phase(over="unit.identity_uids")`
   whose path ends in a property over a typed field now renders the "For each later item" block, as a
   fan-out over a field does. This changes the preview text only, not the run.

### Upgrading from Phase 11 (check options, one hook, short forms, plain required fields)

Every BREAKING item below gives the old form and the new one. Each old form is a load error or a call
error that names the new form; there is no legacy support, except where an item says a form still works.

1. **BREAKING: the check options are two (fix-3rsq.two-options).** The Phase 11 options `subject`,
   `means` and `strip` are retired. Each one that is still declared fails the load, and the message
   names the new spelling. Migrate each declaration as the table says:

   | Old (Phase 11) | New | Notes |
   |---|---|---|
   | `subject: True` on a `str` | `in_request: "subject"` | same check; it implies `in_request` |
   | `means: {v: text}` on an enum, with `verify: True` | `verify: {v: text}` | the dict is the check; each key must be a value of the enum, else a load error |
   | `strip: ["group", "unit"]` on a `str`, with a check (`in_request` or `subject`) | `in_request: {"ignore": ["group", "unit"]}` (or `{"subject": True, "ignore": [...]}`) | the same cleanup, before the match, as before |
   | `strip: [...]` on a `str` with no check | `strip_words(name, ("group", "unit"))` in the field's validator or in `validate_extracted_parameters` | the field had no check, so `in_request` would add one; the cleanup is code |
   | `verify: True` on a `str` with `subject` | `verify: True` on a `str` with `in_request: "subject"` | unchanged meaning |

   - `in_request: {"ignore": [...]}` runs the cleanup that `strip` ran: before the match, on the
     extracted value. The outcomes match the old `strip` on the cases checked (required, optional and
     `subject` fields alike).
   - The rule sentences the extractor reads are the same. The previews show the new spelling: the
     column `in_request` holds `yes` or `subject`, followed by `ignore` and the words, and a `verify`
     dict's meanings are the preview's notes.
   - Each field a check was declared on needs the option now: a `str` or `int` is checked only when it
     carries `in_request` or `subject`. A field with neither is taken as the extractor gives it.

2. **BREAKING where it changes a value: Input validators write back.** A value that an `Input` validator
   returns instead of its input is written back to the arguments, for both kinds of command. Before
   this release the returned value was checked and then discarded. Inventory: no `Input` validator in
   the app, the bundled examples, the test workflows or the tests returns a different value, so none
   changes its behaviour. A validator that refuses still refuses. The Facts models of the app's
   `_phase_facts.py` are outputs, not inputs.
   A validator's wording is the hook's: a `fastworkflow.Refused` is `validate_extracted_parameters: <message>`
   (with its `field`), and an exception is `validate_extracted_parameters raised <Type>: <text>` on the planning
   path, `Exception in <command>'s validate_extracted_parameters function: <text>` on the execution path.
   A plain `ValueError` or `AssertionError` is a refusal with no field, worded the same way; any other exception
   is the call error `validate_extracted_parameters raised <Type>: <text>`.

3. **BREAKING: short forms only.** Each old form is a load or call error that names the new form.

   | Job | Old form | New form |
   |---|---|---|
   | Call a command | `def __call__(self, workflow, command)` | `def __call__(self, workflow, command, input)`; `input` is the Input, or `None` for a command with no Input |
   | Return a result | `return CommandOutput(command_response=CommandResponse(response=X))` | `return X` (a `str`), or the command's `Signature.Output` instance |
   | Return with artifacts or next actions | `return CommandOutput(command_response=CommandResponse(response=X, artifacts=...))` | `return fastworkflow.CommandResponse(response=X, artifacts=...)` |
   | Refuse a call | `return CommandOutput(command_response=CommandResponse(response=M, success=False))`, or a `CommandResponse(..., success=False)` return | `raise fastworkflow.Refused(M)` |
   | Refuse one value | `return (False, M)` from `validate_extracted_parameters`, or a `db_lookup` rejection | `raise fastworkflow.Refused(M, field="name")` (the field is reset and asked for again; `field=` takes a list too) |
   | Accept | `return (True, "")`, `return True, ""` | `return None` (or fall off the end) |
   | Generated command file | `_process_command` plus a `CommandOutput` `__call__`, and a `validate_extracted_parameters` stub | one `__call__(self, workflow, command, input) -> Signature.Output` (regenerate) |

   - `CommandOutput` is not a return value. Commands that return one fail with an error naming the
     command. A handled-but-not-successful router answer is the routing `Signature.Output` (handled and
     routed flags); the framework builds the turn's `CommandOutput` from it.
   - A call error names the command and the value it returned, for example "... returned a
     CommandResponse with success=False ..." or "must return None to accept and raise
     fastworkflow.Refused(message, field=...) to refuse; it returned tuple".
   - The bundled simple workflow template raises `Refused` for a failed create. The Phase 10 item
     "Check command return values" is superseded by this item.

4. **BREAKING: one validation hook; `db_lookup` is retired.** `validate_extracted_parameters` is the
   one hook for checks that need the workflow, the request text, several parameters together, or
   suggestions. Its return is `None` (accept). A non-None return is a `TypeError`. An exception other
   than `fastworkflow.Refused` becomes a call error that names the command, so raise `Refused` for any
   refusal the user should see. Pydantic `Input` validators are for self-contained per-value rules
   (format, range) that need no context.

   | Old | New |
   |---|---|
   | `json_schema_extra={'db_lookup': True}` on a field, or a `Signature.db_lookup` method | a lookup in `validate_extracted_parameters`: `matched, corrected, suggestions = DatabaseValidator.fuzzy_match(...)`; assign `input.<field> = corrected`, or `raise fastworkflow.Refused("... Did you mean ...?", field="<field>")` |
   | a `*_uid` field corrected from a display name | map the label back to the uid yourself: `input.user_uid = uid_of[corrected]` |

   - A lookup that matches nothing and offers no suggestion returns `None`: the value is accepted.
   - The param-extraction span no longer writes `db_lookup`. A value the hook rewrites is recorded as
     the slot binding `validation_hook`, and a refusal with `field=` is recorded as an invalid field.
     Span contract v32 (`fw.nlu.param_extraction` v8); the decision signal contract is v2. A span
     recorded before this carries `db_lookup` and no change list.
   - Planning commands follow the same hook rule. Their `__result__` forms are unchanged: a `str`, the
     `Output`, or a `CommandResponse`.
   - The bundled `messaging_app_4` example's `set_current_user` does its lookup in its hook.
   - The coding skills (`resolve-parameter-values`, `validate-command-parameters`,
     `integrate-chat-agent`) teach only these forms. Projects that copied the old forms migrate by the table.

5. **BREAKING: required fields are plain fields (fix-3rsq.plain-required).** A field with no default and
   no `Optional[...]` is required. Write it plainly:

   | Old | New |
   |---|---|
   | `name: str = Field(default="NOT_FOUND", description="...")` | `name: str = Field(description="...")` |

   - The framework supplies `NOT_FOUND` (or the field type's own sentinel) internally while the value is
     missing, so the Input can be built, then asks the user as before. Both command kinds, and the
     str, int, float and list fields, work this way.
   - `default="NOT_FOUND"` still loads but logs one WARNING at load for each field that declares it.
     Planning commands are exempt from that warning.
   - A direct call (MCP, REST, `perform_action`) that omits a required field now gets the same ask as the
     chat path, not a raw pydantic `Field required` error. The MCP tool schema lists plain required fields
     under `required`.
   - Editing a command's source changes its fingerprint, so the next `fastworkflow train` retrains that
     command.

6. **Environment sentinels are constants, not env keys.** `NOT_FOUND`, `INVALID`,
   `MISSING_INFORMATION_ERRMSG`, `INVALID_INFORMATION_ERRMSG` and `PARAMETER_EXTRACTION_ERROR_MSG` are
   constants in `fastworkflow.utils.signatures`. The framework does not read them from the environment,
   so delete any such line from `fastworkflow.env`. Import the constant where a command needs it. The env template
   lists `FW_TURN_MAX_STEPS` (commented; default 400).

7. **No generated utterance boilerplate.** A generated command file no longer contains
   `generate_utterances` or `template_utterances`. The framework default generates the synthetic
   variants from `plain_utterances`, so a file needs only the seed list. Files that already define them
   keep working; delete them when you next touch the file. The stub's zero-argument
   `generate_utterances()` that returned `[]` is gone.

8. **Paging recognises only `start`, and a paged command declares `total`.** A paging field is named
   `start` and is `int` or `Optional[int]`; `offset` and `skip` are no longer paging names. A command with
   a paging field whose Output has no `total: int` is a load error that names the field. A field that is
   not a page (for example a `datetime` named `start`) is an ordinary field. Rename a paging field that is
   not a page.

   | Old | New |
   |---|---|
   | `offset: Optional[int]` on Input | `start: Optional[int]` on Input |
   | Output without a total | `total: int` on the Output |

   - When rows remain, the response text ends with a paging line, for example
     `Rows 1-20 of 57; next page: start=20.` **Clients that parse the response as JSON must read
     `structuredContent` instead** (MCP), which carries the Output without that line.

9. **Schemas for clients (additive).** `GET /command_schemas` (operation `get_command_schemas`) lists
   every listed command with its kind and schemas; framework commands and the internal command-metadata
   commands are hidden. A per-command MCP tool declares `outputSchema` for a command that has an Output,
   and returns `structuredContent` (the Output as JSON) beside its text content. The REST turn body keeps
   `next_actions` and `recommendations` in each `command_response`.

10. **The Input is named `input`.** Docs, the scaffold and the bundled examples name the Input instance
    `input`. The framework passes it positionally, so a command that names it differently still runs. New
    code uses `input`.

11. **BREAKING: a declared Output must be returned (fix-3rsq.declared-output-required).** Before this
    release, a command that declared `Signature.Output` but returned a `str`, or a `CommandResponse`
    without that output, only logged a warning, and its MCP tool advertised `outputSchema` with no
    `structuredContent`. Now it is a call error that names the fix. Return the Output, or
    `CommandResponse(response=<the same text>, output=<Output>)` when the response text must stay as it
    was. A command that declares no Output is unchanged. Planning commands are not affected: their
    `__result__` fallback still writes text without an output, and that is still only logged.

12. **BREAKING: a last phase of 'say that' lines is a load error (fix-3rsq.p14.1, Phase 14).** A planning
    command whose last phase is made of `say that` lines (and guards) no longer loads: its message reads
    `phase '<name>' writes the answer as 'say that' lines, which code no longer copies; return the text from
    __result__(self, input, <phase facts...>) instead`. Return the answer from
    `__result__(self, input, <phase facts...>)` (a phase's name is a parameter, its facts the value). A flat
    planning command whose answer is `say that` lines is refused the same way, at load where the load check
    can see its steps and at run time otherwise (fix-3rsq.p14.6). A non-last phase made only of `say that`
    lines is skipped: no model call, and its text reaches the answer only through `__result__` (fix-3rsq.p14.6).
    `__result__` returning `None` means no code-written answer on that path (fix-3rsq.p14.6): the answer is
    written as if the command had no `__result__`, with no warning. A flat call's agent answers. A phased
    call's last phase runs its steps only, and its answer is one model call with the phases' facts (a span
    `fw.skill.answer`, span contract v34).

### Upgrading to Phase 11 (check options, one validation step)

1. **Declare the checks a field needs.** The framework no longer checks a `str` or an `int` by its
   type alone. A `str` or `int` field is checked only when it carries `in_request` (the value must be
   written in the request) or `subject` (a `str` naming a record the request names). Add those options
   to each field that needed the check before. A field with neither is taken as the extractor gives it.
2. **Call one validation step.** `validate_call` is removed. A caller runs
   `check_arguments` and then the command's `validate_parameters`, which is the only validation step
   for both kinds of command. Replace each `validate_call` call with those two.
3. **Fix any check option on the wrong type.** A load error names the command, the field and the
   option. `in_request` takes a `str`, an `int` or a `list[str]`; `in_request: "subject"` takes a `str`.
   `verify` takes a `bool`, an enum, or a `str` with `in_request: "subject"`; on an enum it may be
   a dict of each value's meaning (see Upgrading from Phase 11, item 1).
4. **Check the defaults of `bool` fields with `verify`.** A `bool` may now default to true. The
   extractor is then told "False only when the request asks against it."
5. **Check the previews.** An open-ended type (such as a `float`, a `list[str]` or a `str`) shows one
   sample value in the preview.

### Upgrading to Phase 10 (matching contract, approvals, return forms)

1. **Retrain every workflow that has planning commands**, with `fastworkflow train`. Their
   earlier artifacts refuse to load and say to retrain. The first train after the upgrade is a
   full retrain, and it **generates** training utterances for each planning command, because
   planning commands are now ordinary classifier labels. That is an LLM cost earlier releases did
   not incur. Later trains reuse the generated utterances. A workflow without planning commands
   needs no retrain for this change.
2. **Declare navigation on the command, not the context.** Delete `enter_command` and
   `enter_commands` from each context class, and set `descends_to = "<Context>"` on each
   `ResponseGenerator` or `PlanGenerator` that enters that context. A context class that still
   declares either one is a load error. It names the file, the command and the fix. A command that
   moves the workflow into a context it does not declare logs one warning per command and target.
3. **Review `approval` annotations.** Before this release the attribute was declared and shown but
   not enforced. Now `approval = "mandatory"` asks before every run, including a run you typed
   yourself, and `approval = "delegatable"` asks unless a standing approval is recorded. Tell users
   that the prompt takes `yes`, `no` or `always`. `FW_APPROVAL_BINDING` takes `off`, `shadow` or
   `enforce` (the default). `off` restores the earlier path, and `shadow` records each decision
   without asking.
4. **Check command return values.** A command may return its own `Signature.Output`, a `str`, or a
   `CommandResponse`. Phase 12 removed `CommandOutput` as a return value (see Upgrading from Phase 11,
   item 3). Any other type fails that call with an error that names the command. A command with no
   `Input` receives `None` as its parameters when its `__call__` declares a third argument.
5. **Assistant mode** runs a planning command only when you name it exactly and it is mechanical.
   Any other planning command is refused, with the reasons. Direct API and MCP calls still refuse
   planning commands.
6. **Known issue:** `fastworkflow command preview --check` looks for the bundled examples' previews
   in `fastworkflow/examples/docs/planning_command_previews`, a folder that is not committed. It
   exits 1 for the library example. That exit does not mean the output is stale.

### Upgrading to Phase 9 scoped lookup

1. Retrain every workflow with `fastworkflow train`. Old artifacts refuse to load until you do,
   and the first train is a full retrain.
2. Delete `context_hierarchy_model.json` from each workflow, if it is still there.
3. Review one-word command names of each context. A request that starts with a name that
   matches a command of an enclosing context now reaches that command.

### Fixed

- **The large tier's ambiguity threshold no longer drops to the 0.5 floor when the model makes few or no test
  errors (fix-1hs8, `fastworkflow/model_pipeline_training.py`).** It was the mean confidence of the large model's
  misclassified test rows, so a better in-distribution fit gave a looser threshold. With fewer than 5 misclassified
  rows it is now the 10th percentile of the correct rows' confidences (clamped to [0.5, 0.99]); with 5 or more it is
  the failed mean as before. `analyze_model_confidence` adds `count` and `low_quantile`, and
  `heldout_evaluation.json` records `large_ambiguity` (failed rows, rule, threshold, wildcard presence) per context.
  `write_ambiguity_thresholds` takes an optional `has_wildcard_class`; `False` with a large threshold exactly at
  the floor logs one WARNING naming the context (the thresholds are still written, because raising would abort
  training for every later context). `train()` passes it (the head has an escalation class) and the per-context
  `large_ambiguity` record gains `at_floor_without_wildcard`. The warning is meant to become an error once a retrain
  shows the built-in heads do not trip it. Retrain to pick up the new thresholds.

### Added

- **A '/' parameter round keeps its deterministic routing when it is parked or rebuilt from saved state** (fix-icce). The serialized session state and the parked pending snapshot now carry `assistant_command`, the flag a failed '/' command sets, and restore it with the round, so a bare reply (`abort`, a value) after returning to the conversation or after a restart stays off the agent in an agent session. The flag never leaks into the conversation that parked it, and state saved before this change (no key) loads exactly as before. Test: `tests/test_slash_round_routing_survives_restore.py`.
- **`--chat_approval_surface {channel,host}` on `run_fastapi_mcp` chooses the approval surface of the stream and assistant endpoints** (fix-rkr1, step 1). `POST /invoke_agent_stream` (the MCP `invoke_agent` tool) and `POST /invoke_assistant` passed no approval surface, so they fell back to the channel surface (a plain `yes` approves, no token), while `/invoke_agent` and `/perform_action` are always the host surface (signed JWT with a `user_id`, a token, refused otherwise). **The default is `channel` and nothing changes at the default**: those two endpoints pass exactly what they passed before. `host` is the documented intent for a standalone server: the two endpoints then apply the `/invoke_agent` rule, deliver the approval token in the stream's `output` frame (`data.approval`) or the `/invoke_assistant` body (live only, never in a `TurnOutput`, a stored record, an observability row or `GET /turns`), and read `approval_token` from the request. `/invoke_agent` and `/perform_action` are never loosened by it. One helper, `run_fastapi_mcp.utils.approval_surface_for`, now builds the surface for all four endpoints. `fastworkflow run_chatbot` starts its loopback server with `--chat_approval_surface channel` explicitly, so the chat does not change when the default later moves to `host`. The chat page sends an `approval_token` back on the reply that answers an approval prompt when one was delivered, and never renders, logs or persists it. Tests: `tests/test_approval_endpoint_surfaces.py` (the endpoint x JWT mode x option truth table) and `tests/chatbot_approval_card_dom.cjs`.
- **The answer-card composer is diagnosable from the server log** (fix-weih.25, `ComposeDiagnostics` in `run_chatbot/composer.py`). The `compose answer_card:` INFO line now also carries `digest_chars`, `collapsed`, `sections=kept/proposed`, `blocks=kept/proposed`, `lead=kept|dropped|none` and `drops=<reason>:<count>,...` (counts and stable reason codes only, never text). At `LOG_LEVEL=DEBUG` the raw model `lead` and `sections` (first 4000 characters) and each drop's reason with the offending ref or first 80 characters are logged too. The response to the page is unchanged.
- **A composed answer card no longer attaches a label to data that does not match it** (fix-weih.25, `run_chatbot/composer.py`). A stat label is kept only when it shares a content word (case-insensitive, plural/singular, no stop words or generic words such as total, count, list, items) with the field's key, schema title or description, the command name or the result's context; otherwise code writes it: the field's schema title when specific, else `<Command>: <Field>` (`List Orders: Total`). A table, record or stats title that the data does not support is dropped. Both are counted in `drops=` as `stat_label_unsupported` and `title_unsupported`; the `ComposeAnswerCard` instruction now says a stat label must name what its ref holds.
- **A composed answer card no longer lets a number in prose attach to the wrong thing** (fix-weih.25, `run_chatbot/composer.py`). A quantity in the `lead` or a section `verdict` is kept only when the answer or the user message states the same number with a content word that the number owns in the answer and in the model's sentence too (a word belongs to the nearest number within 3 words, a tie going to the number before it, and clauses end at `. ; | ,` and line breaks, so `4 control findings and 32 rights` does not anchor `32 findings`) (`4 findings were proposed` is anchored by `4 control findings` or `findings: 4`, not by `32 rights` or by a total that exists only in the data). A number that is part of a name copied whole from the answer, message or data (`Administrator 1`), an ID (`#A1`, `user_88`, `3rd`) and a date copied whole stay allowed; written numbers are not read. A dropped verdict keeps its section's heading and blocks; a dropped lead leaves the existing lead-less card rules. Counted in `drops=` as `verdict_number_unanchored` and `lead_number_unanchored`; the `ComposeAnswerCard` instruction now says to state outcomes in words and put counts and amounts in stats blocks.
- **A composed answer card gets one correction pass when validation dropped something a second look can fix** (fix-weih.27, `run_chatbot/composer.py`). After the first pass, if the tally holds a reason in `CORRECTABLE_DROPS` (unresolved refs, not a table/record, unsupported labels and titles, unanchored or ungrounded lead/verdict/quote, empty or unparseable sections; never caps, duplicates or a failed call) and the first pass took at most 17.5 s, the model is called once more with a `corrections` input (at most 800 characters of plain facts built from the recorded drops: `ref r1.owner does not exist; r1 has fields: ...`). The second answer is validated with a fresh tally and kept only if strictly better by (sections kept, blocks kept, lead kept, fewer correctable drops); a failed second call is ignored; never a third call. The first call's prompt is unchanged (the second uses a separate signature). The INFO line gains `retry=none|used|kept_first|failed|skipped_slow` and, after a retry, `drops2=`, `sections2=` and `blocks2=`.
- **Short sub-task headings** (fix-weih.26, `_heading_title` in `workflow_execution_context.py`). A multi-part answer's
  `### N.` heading is the sub-task's first clause (at most 80 characters, cut at a word with an ellipsis) instead of the
  whole sentence; a short text is unchanged. No model call; the sub-task still runs with its full text.

- **The live panel shows the running stage and the elapsed time** (fix-kq49, `fastworkflow/turn_progress.py`, part `261`).
  A turn that carries a multi-part request now announces the start of each model-backed stage (split, referents, match,
  draft, review, with "i of n" for the per-sub-task ones) as a `progress` frame on `/invoke_agent_stream`, and the chat
  shows the stage and a timer that ticks each second from the send. No added model call; a payload is a stage key, a
  generic label and counts. Additive: a client that does not know the `progress` frame ignores it, the polling and
  non-stream endpoints drop it, and the CLI spinner shows the label. `run_process_message_with_trace_stream` takes an
  optional `on_progress`.
- **`run_chatbot` draws a command's result as a card, with no model call and no author hint** (fix-weih.3, fix-weih.4,
  `static/src/252-chat-schemas.js`, `254-chat-results.js`). A command's `command_response.output` and `GET /command_schemas`
  give a stat strip for up to four short scalar fields, tables for lists of records and for same-length arrays, a badge for
  a `status` or `state` string, labels from `Field(title=...)` (else the field name made readable, acronyms kept), JSON
  strings and one-key wrappers decoded (up to 5 levels), a result that a later command reads through `from_command_output`
  folded under "Supporting lookups". The whole answer stays in the bubble; the results sit in a panel to its right (below it on a
  narrow window) behind a "Show results (N)" link under the answer, closed by default (fix-weih.24). The schema listing is cached per
  connection.

- **`ui` hints on `Output` and `Input` fields** (fix-weih.14, `field_options.UiOptions`, `planning_commands.check_ui_option`).
  `json_schema_extra={"ui": {...}}` or `FieldOptions(ui=UiOptions(...))`: `hidden`, `hero` and `format`
  (`currency`, `percent`, `code`, `badge`) on a `Signature.Output` field, `widget` (`textarea` on a `str`, `radio` on a
  `Literal`) on a `Signature.Input` field. `UiOptions` is exported from `fastworkflow` and `fastworkflow.planning`. A hint
  is presentation only; a wrong one is a load error on an `Input`, and a field-source error (read by `fastworkflow command
  preview --check`) on an `Output`.

- **Waiting turns say what they wait for** (fix-weih.7, `WorkflowExecutionContext._describe_pending_prompt`). The last entry of
  a waiting turn's `command_outputs` carries `artifacts.question_kind` (`agent`, `planning_parameter`, `approval`, `review`,
  `unknown`) and, for three kinds, a payload: `approval` (`command`, `parameters`, `origin`, `irreversible`, `can_always`),
  `review` (`kind`, `title`, `revision`, `items`, `changes_data`) and `parameter_request` (`command`, `missing`, `invalid`,
  `values`). A user-typed `/` approval has `status` `"completed"` and `artifacts.awaiting_user` true; both shapes are tagged.
  No field of `TurnOutput`, the span contract or the store schema changed.

- **A failed `/` command carries `parameter_request` or `intent_choices`** (fix-weih.2). In assistant mode only, a parameter
  failure that names fields adds `artifacts.parameter_request`, and an ambiguous command name adds `artifacts.intent_choices`
  (the qualified candidates). The text of the messages is unchanged and the agent never sees either key.

- **`run_chatbot` forms for every question it asks** (fix-weih.5, fix-weih.11, fix-weih.12, fix-weih.13, parts `256` to `262`).
  A `/` command's missing or rejected parameters become a form (**Run** sends `/command name=value ...`), an ambiguous `/`
  command becomes buttons, an agent's refused planning-command input becomes a form of the asked fields (sends
  `<field name>: <value>` lines), an approval becomes an Approve / Always allow / Decline card (sends exactly `yes`, `always`,
  `no`), and a sub-task or plan review becomes an editable form (sends `yes`, `no` or edits in the `review_engine.parse_edits`
  grammar). None calls a model. The server's text stays under **Details**. Limits: a password field's value on a `/` form
  is sent as chat text and recorded in the transcript, and the review form refuses edits the server grammar would
  misread. The chat's artifacts link no longer lists the prompt keys.

- **`POST /compose_answer_card` and `POST /compose_question_form`, and `run_chatbot`'s "Rich answers"** (fix-weih.8, fix-weih.9,
  fix-weih.10, fix-weih.11, `fastworkflow/run_chatbot/composer.py`). A small model on `LLM_RESPONSE_GEN` writes a short lead and
  chooses blocks for a non-`/` turn with more than one primary result or an answer over 400 characters (the chat requests it
  by itself once per eligible turn, shows the whole plain answer and a "Composing answer..." line meanwhile, caches the card by
  turn, and for a reopened conversation asks only for turns whose record is loaded; fix-weih.24), and turns an agent's
  `ask_user` question into a form. The model only chooses: blocks are bound to the turn's data by reference, a lead with a
  number or ID not in the data is dropped, a quote must be in the answer, and a form option must be a value or dict key in the
  results or appear in the question. The form is built by tuned instructions (a `known_values` input, yes/no questions as one
  boolean, a list in the results as options) and guards that drop a field restating result data, require one field only on a
  one-field form, and, last, a precision gate: a form with no pick of two or more options, boolean or date is not shown. The
  card repairs a record or table pointed at one value (a stat bound to it) and a list-index path (a table over that list), shows
  all columns of a table that names fewer than two, builds a combined table only from a page as the server defines it
  (`output_page`: an integer `total` and lists of one length), and is not shown when it is only a lead restating the answer.
  Both endpoints answer `200` with the value, or `null` and `reason` (`no_model`, `failed`, `timeout`, or `not_needed` when the
  composer chose the plain view; the client treats it like any null); `422` over 60 results (was 20) or 20000 characters; neither is an MCP tool. The **Rich answers** checkbox
  (on by default) turns the composed card and form off; the deterministic card and forms do not depend on it. Counting
  the questions asked per logical turn: the query in the README ("Counting clarification rounds").

- **The composed answer card is summarized, complete and organized** (fix-weih.25, steps 1 and 2). The compose request takes
  up to 60 results (was 20) and each carries `descriptions`, `hints` (a `hidden` key never reaches the model) and `supporting`;
  the model is shown a 9000-character digest (was 6000) that puts primary results first, shares the budget fairly between
  commands, writes a page as one line of column keys with 3 sample rows and "60 of 167", collapses a group over its share to
  one line, and names what is left as `also: <command> xN`. The answer's `### N.` headings are sent as `sub_tasks`. The model
  now writes `lead` and `sections` (`{heading, verdict, blocks}`; at most 5 sections, 4 blocks each, 10 in all); heading and
  verdict go through the same number-and-ID grounding as the lead, a section needs a verdict or a surviving block, and a
  heading that is dropped becomes the matching sub-task title or the command's name. The response is `{v: 2, lead, sections,
  blocks}` where `blocks` repeats the sections' blocks flat, so an older page still works, and a page without `sections` in
  the reply draws the flat card as before. A card table shows 8 rows and then "+M more - open in results"; the card ends with
  one page-written line per primary result group that no block shows (a button that opens the results panel). The model
  output field `blocks` is replaced by `sections` (a model that wrote flat blocks is no longer read). Timeouts: page 25 s
  to 35 s, server compose 30 s to 40 s (shared with the question form); `max_tokens` 1024 to 1800 for the card.

- **`writes = True` declares a command that changes data; a planning command's phases make only the writes
  their steps name** (fix-fixv). An execution command's `ResponseGenerator` declares `writes = True`; a planning
  command writes when its steps reach such a command and is read-only otherwise (`command_writes.write_commands`).
  Inside a planning command's phase, a write command no step of the phase names is left out of the phase's
  listing and refused unrun by the executor, before the approval gate. `writes` on a `PlanGenerator`, or a value
  that is not a bool, is a load error. Outside a phase nothing changes. The library example declares it on
  `Record/add_note` (its one write): `author_books` no longer adds notes to the books it opens.

- **A drift test keeps the planning and execution argument checkers in agreement** (fix-48np.3,
  `tests/test_argument_checker_drift.py`). One table of argument cases (the typed-call strictness corpus, the
  execution checks' cases adapted to fields both checkers read, and cases per deliberate difference) runs
  through `planning_commands.checked_arguments` and `argument_checks.check_extracted_values` on the same
  `Input`s, and their verdicts (accepted, accepted as another value, refused, not given) must be equal, except
  on a documented list of deliberate differences (an int the request does not write, an absent int, exemptions,
  which values are checked, planning-only concepts), each of which a case must show exactly. A new disagreement
  fails the test; the fix is to align the checkers or to add a list entry by an explicit decision.

- **A planning command's `str` argument with a `One of: a, b` description list must be one of them** (fix-48np.3,
  the first rule both argument checkers share). The allowed-values rule (`argument_checks.allowed_member`: a
  case-insensitive match, the declared spelling taken, a value outside the list refused) now runs in
  `planning_commands.check_arguments` as well as on execution commands; the drift test found the planning path
  did not check the list. A refused value is refused like any other (an optional argument takes its default, a
  required one fails the call); the planning checks after it are unchanged. A planning command's choice declared
  as a `Literal` is checked as before. Execution commands are unchanged.

- **Sub-tasks are reviewed before a multi-part request runs** (fix-3rsq.plan-review.2, round 1). In agent mode,
  a request split into several sub-tasks, or a sub-task with no confident match, is shown as a proposed plan
  before anything runs. Each sub-task has its goal, its inputs (values, or `from sub-task N`) and its means: a
  planning command, a plain command, or build from scratch. The proposal is drawn from the head candidates
  (execution and planning top-k), the shortlist and, with `JEV_API_KEY`, Jev's planning match. Replies are
  `accept`, `no` (the turn ends and nothing runs), typed edits (`change N: means=...`, `change N: inputs=...`,
  `drop N`), or free text: one model call reads the free text into structured edits, which are shown back for
  confirmation. A review admits 8 replies or 15 minutes.
- **Fast path.** One sub-task with a confident head match, or an exact command name, runs directly, with no review.
- **No answerer runs the draft.** An unattended turn, or a host surface (MCP/REST, agent-to-agent), runs the draft
  plan as proposed and marks it `not_reviewed` (span attribute `subtask_review`). Commands with an approval
  annotation still need their approval.
- **Hand-off of the reviewed plan.** Each sub-task runs with its means: a planning command through the phase runner
  with its checked arguments, a plain command with a listing narrowed to it (its producers and the navigation to its
  context included), or build from scratch. A `from sub-task N` input takes N's typed Output when N produced one,
  and N's answer text only when it did not. An accept after a restart resumes the review.
- **Goals complete only through the finish check.** A successful command does not complete its sub-task's goal.

- **`fastworkflow command run-code`, and `fastworkflow.planning.run_code`** (fix-3rsq.17.1). One call of a command,
  run with no model: a planning command phase by phase, as the tutorial's runs show it, stopping at the first
  open step with exit 0; `--start-with` puts the workflow on a record first; a step naming a planning command
  runs it indented; an execution command runs and prints its `CommandOutput`. A hook that raises exits 1, naming
  it. The steps are the workflow's real commands on a real session, so the help says to use a development
  instance. The tutorial test's `dry_run` is now a call to `run_code`, and the pages' traces are unchanged except
  the open-step line, which now reads `step <n> is open: the agent would run it and the rest`.
- **Load problems where developers look** (fix-3rsq.17.3). `fastworkflow run` prints the command files that do not
  load before its first prompt, `fastworkflow train` stops with them before training, and `run_fastapi_mcp` logs
  them at startup. All three use `command_preview.load_problems`.
- **Jev chooses among an ambiguous intent's candidates** (fix-jd9o.1). When the intent classifier
  returns several candidates, Jev picks one before the "command is ambiguous" prompt is shown. The
  turn routes to Jev's pick when Jev is at least 0.9 sure (`INTENT_TIEBREAK_THRESHOLD`); otherwise the
  prompt is shown as before. A confident `wildcard` takes the existing not-here path to the parent
  context. Jev's pick is never written to the utterance cache, and the span records it under
  `jev_tiebreak` with `matcher_layer` set to `jev_tiebreak`. It runs only with `JEV_API_KEY`; set
  `FW_JEV_INTENT_TIEBREAK=off` to turn it off.
- **Jev checks the fields an LLM extraction filled** (fix-jd9o.3). After LLM parameter extraction,
  Jev decides for each filled field whether the request gives that value. A field Jev is at least 0.9
  sure is not given (`FIELD_GIVEN_THRESHOLD`) is reset to its not-given value: an optional field takes
  its default, and a required field is asked for. Values the extractor invented, such as
  `filter='Jane Doe'` from a bare `show_group_members`, are therefore no longer run. The deterministic
  forms (tags, `name=value`, positional) are not checked. The reset fields are in the extraction span's
  diagnostics as `fields_reset_not_given`. It runs only with `JEV_API_KEY`; set `FW_JEV_FIELD_GIVEN=off`
  to turn it off.
- **`DatabaseValidator.fuzzy_match_with_request`: Jev picks among close values** (fix-jd9o.5). The helper
  returns the same three-tuple as `fuzzy_match`, and a unique fuzzy match still stands. When fuzzy
  matching leaves several candidates, Jev reads the request and each candidate's description. It
  returns the intended value when Jev is at least 0.9 sure (`VALUE_PICK_THRESHOLD`); otherwise the hook
  asks as before. The helper takes the request from the hook's `command_text` parameter and a
  `{value: description}` dict. Descriptions are required, and with none Jev is not called: from bare
  names, Jev picked wrongly. In the pre-registered test it made 97 confident picks with 1 wrong, and
  unnecessary questions fell from 89 to 41. It runs only with `JEV_API_KEY`; set
  `FW_JEV_VALUE_PICK=off` to turn it off.
- **Assistant mode reads the reply to a missing-value prompt before merging it** (fix-jd9o.6). Jev
  decides what the reply does. An answer is merged as before. A request to stop runs the existing
  abort, and a "wrong command" reply runs the existing "you misunderstood". A request for something
  else leaves the pending command and runs as a new request. A hesitation ("hmm, hold on") gets
  "OK, take your time." and the missing fields; the pending command and its values stay as they
  were. A non-answer needs probability 0.5 (`MISSING_VALUE_REPLY_THRESHOLD`). Measured on 280
  persona replies to real prompts: 0 of 130 answers diverted, 30 of 30 hesitations recognised. A
  question back is, for now, treated as a request for something else. It runs only with
  `JEV_API_KEY` and only in assistant mode; set `FW_JEV_MISSING_VALUE_REPLY=off` to turn it off.
- **Agent mode: a hesitation in reply to `ask_user` keeps the question open** (fix-jd9o.11).
  `classify_user_reply` has a fourth intent, `wait`. On a hesitation the agent does not replan;
  it says it will wait and asks the same question again, and `ask_user` stays available. Before
  this, 23 of 30 hesitations were read as "stop asking", which removed `ask_user` for the rest of
  the turn. Set `FW_JEV_REPLY_WAIT=off` to keep the three earlier intents.
- **Check options on any command's `Input`** (fix-3rsq.11). `in_request` and `verify` apply to an
  execution command's `Input` field as well as a planning command's. Each option is checked by the same
  rule for both kinds. (Phase 12 cut the five Phase 11 options to these two; see Upgrading from Phase 11.)
- **Typed inputs `float` and `list[str]`** (fix-3rsq.11). A planning command's `Input` takes a
  `float` or a `list[str]`. A `list[str]` with `in_request` keeps only the items the request writes.
  A `bool` may default to true.
- **Planning commands are ordinary classifier labels** (fix-3rsq.10.12). A planning command is a
  label of every classifier whose scope holds it, so a request reaches it the way it reaches an
  execution command. The experiment switch `PLANNING_HEADS_LABELLED` remains for
  re-measurement. Old artifacts of such a workflow refuse to load (see the upgrade checklist).

- **Approval annotation, enforced** (fix-3rsq.10.13, fix-3rsq.10.14). A command class may declare
  `approval = "mandatory"` or `approval = "delegatable"`. Absent means the agent decides, with no
  gate. The annotation is not a read/write gate. Each approval is bound to the exact parameters it
  was asked for. A reply is single-use and expires after 2 turns or 15 minutes, and a changed
  parameter invalidates it. A reply of `always` records a standing approval for that user, that
  workflow and that command. `/approvals` lists standing approvals, `/revoke_approval <command>`
  revokes one, and `/cancel_pending` cancels the pending approval. A host receives the approval token
  in the live response only, and it is never stored or logged. Intent confirmations and plan review
  are not approvals. The listings mark annotated commands: "(needs your approval before it runs)" for
  `mandatory` and "(may need your approval)" for `delegatable`.

- **Command return forms** (fix-3rsq.12.early). A command returns its own `Signature.Output`, a
  `str` or a `CommandResponse`. The framework fills the envelope, so the command does not build a
  `CommandOutput`. (`CommandOutput` was removed as a return value in Phase 12.) `fastworkflow.Refused(message)` gives `success=False` with that
  message. A command with no `Input` receives `None` as its parameters when its `__call__` declares a
  third argument.

- **Full run listing by default** (fix-3rsq.10.15). A sub-task's run is shown the full listing of
  commands. A size-based command shortlist exists for a larger workflow, above
  `FULL_LISTING_MAX_CHARS` (7000 characters). Round 1 of plan review switched production to the shortlist
  above that size (see the production listing entry under Changed).

- **The library example has eight seeds per command** (fix-3rsq.10.17). Its
  `plain_utterances` are the worked example for the seed count. Follow it, and keep each command's
  seeds distinct from its neighbours' and from its callees'.

- **Scoped lookup: a request finds its command along the live chain** (fix-3rsq.9).
  A command is found on the path a request walks: the current context and its bases, then
  each enclosing record on the live object chain and its bases, then the global commands.
  The first scope that recognises the request wins, and the command runs on that scope's
  object; a run of an enclosing record's command is filed under that record. The same path
  serves a command named exactly. A qualified name must name a context on the path.
  Upgrade checklist below.

- **Not-here seeds are recorded per head** (fix-3rsq.9.8, `docs/tutorials/planning_commands/14`).
  Each classifier records which seed it used for each command that is not one of its labels.
  Editing one command's seeds retrains only the classifiers that hold that command (and any
  that recorded its seed); adding or removing a command retrains every classifier.
  `fastworkflow train --regenerate-utterances` chooses the seeds again.

- **`fastworkflow command preview` renders a union-typed phase output as variants** (fix-3rsq.phase7-union-preview). A phase whose output is a union of models (`FindBook.Output |
  FindMember.Output`) is rendered as one variant per member instead of failing.


- **A migration guide for app workflows** (fix-3rsq.8,
  `docs/migrating_skills_to_commands.md`): moving `_skills/` into
  `_commands/`, keeping an app's own generator from deleting planning
  commands, import depth, `from_`, `__phases__`, the step helpers, the field
  options, and the renames below.

- **The build generator protects planning commands** (fix-3rsq.7,
  `fastworkflow/build/command_file_template.py`). Every file `fastworkflow
  build` creates starts with `# fastworkflow:generated — regenerate with
  fastworkflow build; edits may be overwritten`. The build will not write over
  a file that defines a top-level `PlanGenerator`, even with
  `overwrite=True`; it logs a warning naming the file instead. The check reads
  the file with `ast` and falls back to a line check when the file does not
  parse. The build validator checks only `Signature` on a planning command.

- **Policy verifiers** (`docs/policy_verifiers_design.md`). A workflow can ship a
  versioned table of deterministic rules in `_policy/table.py`; a workflow that
  has one has its rules enforced, with nothing to configure. The rules are checked:
  - before a command the model chose runs, on its real name and validated
    parameters;
  - before a question goes to the user;
  - before the answer is given.

  A rule that fires replaces the step's result with its own text, and the agent
  carries on. Rules that need to know what the user asked for read it from one
  model call per user message (`LLM_POLICY_INTENT`, else `LLM_AGENT`). A workflow
  without a table runs unchanged. `fw.agent.step` v8 (span contract v38) records which rule fired; the
  debug UI shows it on the step. Worked example:
  `fastworkflow/examples/retail_workflow/_policy/`.
- **`fastworkflow policy replay`**: replays a recorded experiment through a policy
  table and reports every step it would block, with the totals on passing and on
  failing attempts. This is the audit a table must pass before it is turned on.
- **`fastworkflow distill --self` and `--pool-students/--pool-teachers`**: distil
  failing trials against a passing trial of the same model, or against a passing
  teacher trial when no student trial passed. `--max-pairs-per-task` caps the pairs;
  `--provenance FILE` writes which pair produced which insights.
- **Coding skills `distill-policy-verifiers` and `insights-to-failure-modes`**: the
  end-to-end loop for any workflow, and clustering distilled insights into a
  failure-mode taxonomy (with a checker).
- **Sealed experiments can be selected and acted on together** (Tools → Sealed
  experiments). Select all, or pick rows, to seal complete experiments, delete
  sealed ones, or release invalid ones in one confirmation. A released
  experiment can be deleted the same way; the page calls that removal delete.

- **Skills are written as `Signature` + `PlanGenerator`** (`fastworkflow/skill.py`,
  `docs/skill_file_layout.md`). A skill file reads like a command file: `class
  Signature` declares it as a command's does, and `class PlanGenerator` generates its
  plan. Each public method is a phase in the order written (`_name` methods are
  helpers); its parameters name `input` and earlier phases; its return annotation is
  the facts model it fills; it returns its steps as text, and `return []` skips it
  (decided when the phase is reached). `@phase(over=..., max_items=..., until=...,
  noun=..., optional=...)` for fan-outs and optional phases; `pick` and the filters
  `num`, `count`, `listed` for steps. Skills are unreleased, so the old form is gone.
  A step naming a command or parameter the workflow lacks is a load error.
- **`fastworkflow.Refused(message)`** (commands): raised from a `ResponseGenerator.__call__` or from a
  `validate_extracted_parameters` hook it is a failed answer with that message. `Refused(message,
  field=...)` also names the refused field. The Phase 10 `(False, message)` return is retired. A `Signature` without `generate_utterances` gets the standard body as a
  default; a defined or inherited one wins.
- **`fact_from("<command>.<field>")`** (fix-ajir.15, `docs/skill_phases.md`):
  the field of a facts model copied from a command field,
  `Field(json_schema_extra={"from": ...}, **kwargs)`, in place of the `_from`
  helper each skill file wrote. `fastworkflow command preview --check` now fails
  on a `"from"` naming a command or field the workflow does not have (at load
  it stays a warning, so a command's `Output` can come later).

- **Phase facts: stand-in openers, re-bind to the planned record, plainer
  step values; `name=value` parameters parsed in code** (fix-hia1.10).
  In agent mode a command written as `cmd key=value key2=value2`, every key a
  field of its Input, is parsed before the LLM fallback: a value runs to the
  next ` key=` or the end, quotes and a trailing "(...)" note go, types come
  from the Input model; any doubt falls back as before (the span records it as
  `xml_regex`, so no contract changes). A plan step's value also ends at ", " prose, and
  an identifier value at its first token (`step_arguments`). A planned opener
  with concrete values (`list_x x_uid=U`) is done by a successful run of
  another command that entered the phase's record with the same value under
  the same name (`runs_standing_in_for_openers`); facts from the planned
  command's Input take it. A non-fan-out phase whose step commands all ran on
  one record, and that ended elsewhere, is filled on that record
  (`planned_record`).
- **A plural noun for a fan-out's items** (fix-hia1, `docs/skill_phases.md`).
  `Phase(..., over=..., noun="people")` makes the answer's notes count in that
  noun and leave out the internal phase name: "Note: only the first 3 of 5
  people were covered." and "Note: the work stopped after 3 of 5 people
  because its stop condition was met." (`cap_note` / `fan_out_note` take
  `noun`). Without a noun the wording is unchanged. A noun on a phase that
  does not fan out, or one that is not a non-empty str, fails the load. Skill
  previews show it in the "fans out" column. A note already in the answer is
  not repeated (two fan-outs over the same items with one noun say it once).
- **Skill phases: flat for some inputs, conditions on earlier facts, fan-out
  that stops on a condition** (fix-hia1.9, `docs/skill_phases.md`). `plan()`
  may return `None` or `[]` to run that call flat, from the file's own
  `steps()` (`is_flat_plan`; `phases_for` returns `None`). `Phase.when` may
  be a callable bound by name to earlier outputs, decided when the phase is
  reached (`runs_now`); a skipped phase counts as not run.
  `Phase(..., over=..., until=callable)` stops a fan-out once the outputs so
  far (its parameter named after the phase) meet a condition (`stops_after`);
  `max_items` still caps, and `fan_out_note` says "stopped" or "capped".
  `fw.skill.phase` v3 (span contract v18) adds `skipped` and
  `stop_condition_met`. Skill previews show a callable `when` / `until` by its
  source (`condition_text`), add a "fans out" column, mark phases whose steps
  read earlier facts as rendered with placeholder facts, render a flat
  variant's `steps()`, and pass a fan-out phase's output on as a list.
- **Answer-only skill phases** (fix-hia1.7, `docs/skill_phases.md`). A phase
  whose steps name no command (`skill_phases.is_answer_only`: no command,
  conditional or navigation step as `finish_check.parse_plan` reads them) is
  not an executor run: one model call with no tools (`write_answer`) writes its
  answer from the request and its steps, told "Do not run any command; write
  the answer from these facts only." It cannot call a workflow command. New
  `facts_text(model)` (`observation_offloading.facts`, re-exported from
  `skill_phases`) renders a facts model as `field: value` text for steps.
  `fw.skill.phase` v2 (span contract v17) adds `answer_only`.
- **Fan-out skill phases with record binding** (fix-hia1.3).
  `Phase(..., over=<list> | "<phase>.<field>", item="item", max_items=None,
  bind=True)` runs the phase once per item, each its own sub-plan; a later
  phase gets the list of per-item outputs. At most `max_items` /
  `FW_SKILL_FANOUT_MAX_ITEMS` (default 10) items run, and the answer says when
  it was capped. Each item is bound to the record that names it: its output
  is filled there, and its finish check flags a step on another item's record
  and an item never opened (`item_record_issues`; span `record_issues`).
  `fw.skill.phase` v2 adds `item`, `item_index`, `item_count`, `items_total`,
  `record_issues`.
- **Skill previews and a skill-file layout convention** (fix-553q,
  `fastworkflow.skill_preview`, `docs/skill_file_layout.md`).
  `fastworkflow command preview <workflow_folder> [--out <dir>] [--check]` loads
  every skill with the skill loader and writes `<dir>/<skill>.md` (default
  `<workflow_folder>/../docs/skill_previews`): description, inputs (type,
  default, sample value, `verify` / `subject` flags, description, other
  options), `plain_utterances`, and the steps for sample inputs, rendered once
  per combination of the inputs that change them; a phased skill also lists
  its phases, when each runs, and the facts each output holds with their
  `"from"` sources, with earlier facts rendered as placeholders
  (`<find.item_uids[0]>`). Byte-stable; `--check` writes nothing and exits 1
  when a preview is stale, missing or extra. The layout doc fixes one order
  for a skill file (docstring, `Signature`, fact models, `plan()` / `steps()`
  as the table of contents, then the step functions), naming, plain-string
  steps, no step logic in helpers, and conditions in `plan()`.
- **Skill previews that read alone** (fix-553q follow-up). Each different plan
  is rendered once, titled by the fewest inputs that decide it ("Plan when
  compare = true"); the "runs" column names the deciding input, and one line
  says when `plan()` returns phases at all. Output models, `when` and fan-outs
  that differ by input carry the inputs that give them. Past 16 plans, a
  pairwise cover of the inputs plus one plan per phase layout is rendered.
  Int inputs are varied (the sample and 0 or 3), and samples respect `ge` /
  `le`. A fan-out renders each listed item (never `None`), or the first and a
  later item of an earlier phase's list; an empty list says it runs nothing.
  Placeholder facts are consistent (a uid fact is the entry of the earlier
  uid list, a fan-out item's output carries its item), str placeholders
  survive `split`, and a float fact is 999999 in arithmetic so `int()` /
  `float()` never print a plausible 0. A callable `when` shows its closure
  values and parameter defaults, joined without "( " gaps.
- **Skill phases: a skill run as bounded sub-plans** (fix-hia1.2,
  `fastworkflow.skill_phases`, `docs/skill_phases.md`). A skill file with params
  may define `plan(input) -> list[Phase]`, with `Phase(name, steps_fn,
  output=None, when=True)`; a skill without it is unchanged. `steps_fn`'s
  parameters are bound by name to earlier phases' filled outputs (typed facts
  models); `when=False` leaves a phase out; `steps(input)` stays the flat
  concatenation (`concatenated_steps`) that matching and the template read, and
  a file with only `plan()` gets it. The loader runs `plan()` on the placeholder
  inputs: bad phases, unbindable parameters or a malformed `"from"` fail the
  load; a `"from"` naming a command field the workflow lacks is a logged warning
  (`check_phase_sources`). A skill call of a phased skill runs each phase as a
  fresh executor run with a step budget of `FW_SKILL_PHASE_MAX_STEPS` (default
  15), its own steps (not rewritten by the planner) and its own finish check;
  after it, the phase's output is filled in code from the phase's own command
  runs on the record it ended on (plus the run that entered that record). Only
  runs that do one of the phase's own steps count (fix-hia1.5,
  `runs_on_plan`, `finish_check.satisfies`): a step giving a concrete value
  (`find_x query=Ann`) takes only runs carrying that value, after
  normalisation, and each page of that query; a placeholder or prose argument
  keeps the looser match. An off-plan lookup of another value never fills the
  fact; with only that run, the fill fails saying the planned run was not
  found. A planned opening step the workflow was already on (refused unrun
  with "You are already in context ...", which archives no run; e.g. an
  earlier phase opened the record) is done by the turn's earlier run that
  satisfies the step, entered the phase's record, and whose parameters are
  words of the record's clause (fix-hia1.6, `runs_that_entered_record`); the
  opening command's Input and Output facts come from that run, and a run
  that only read the record, or entered another one, never counts.
  A phase's finish check applies the same value rule to its steps
  (`values=True`); the finish check of any other run is unchanged. A
  missing fact stops the call: no later phase runs, and the answer is what the
  phases that ran found plus a line naming the failed phase and field. The
  sub-task's answer is the last phase's. A phased skill is never handed off
  (`_hand_off`); other sub-tasks keep the hand-off and the 25-step cap. A phase
  suspended on ask_user resumes and the rest run (the call's state rides the
  session state, `skill_phase_run`). New span `fw.skill.phase` (contract v1;
  aggregate span contract 16) per phase run. `SkillCallPlan` gains `phases`,
  `steps_before`, `steps_after`.

- **Command outputs archived as JSON; facts filled from them** (fix-hia1.1,
  `CommandResponse.output`, `observation_offloading.facts`). A command may
  return its `Signature.Output` as `CommandResponse(response=..., output=...)`;
  the field is optional and appended (default `None`), and the agent's
  observation and prompt are unchanged. Each command an archived execute step
  ran is stored beside the step's text in the new additive
  `offload_command_runs` table (name, success, record context, parameters and
  Output as JSON, redacted like the text), read with
  `RuntimeHandleArchive.command_runs`; older stores gain the table on open and
  their rows read as before. `facts_from_runs` / `facts_from_archive` fill a
  pydantic model whose fields declare `json_schema_extra={"from":
  "<command>.<field>"}` from the Output or Input of successful runs, with no
  model call: latest run for a scalar, runs joined in order for a list, a
  missing required field is a `MissingFactError`, an unknown or ambiguous
  source a `FactSourceError` (`check_fact_sources`, `workflow_signatures`). A
  plain command name resolves against the record's context; `Context/command`
  is accepted. A scalar comes from the latest run of the broadest query (fewest
  given parameters, page position ignored), as list pages do, so a later
  filtered read no longer replaces the whole list's `total`.

- **Finish check: every command step of the plan ran** (fix-3xts,
  `observation_offloading.finish_check`). When the agent picks `finish`, the
  run's initial plan is read against the outputs it archived, in code (no model
  call): a command step whose command never ran successfully is unmet. Unmet
  steps become one note (at most 1 KB, "Step N (cmd) was not run: run it now.")
  that replaces the finish observation, once per agent run and only with two or
  more iterations left (`fastWorkflowReAct.finish_reviewer`). Every checked
  finish records a `finish_check` offload event. It fails open: no plan, no
  archive or an error means no note.

- **Split settings as one command-form file, `_skills/_split.py`**
  (`workflow_agent.split_settings`, `SplitSettings`): the split stage's
  per-workflow settings are one Python file written like a command file, every
  attribute optional (absent file: the built-in defaults). `class Signature`'s
  docstring (`inspect.cleandoc`) replaces `SplitIntoSubtasks`'s instructions;
  optional pydantic `Signature.Input` (`job_types`, `user_query`) and
  `Signature.Output` (`subtasks`) give a splitter field a new description (a
  field the splitter does not have fails the load); `identifier_patterns` is a
  list of compiled patterns (`[re.compile(r"\bacct_[0-9]+")]`) for the split
  merge and skill-call arguments; `subject_hints` is the referent step's text.
  The file is imported as a skill file is (package-relative imports work); the
  leading `_` keeps it from being loaded as a skill. A file that does not
  import (an invalid pattern raises in `re.compile`) or declares a malformed
  attribute fails the load with a `SkillFormatError` naming it, logged; an
  invalid pattern is no longer skipped with a warning.

- **The skill matcher reads a skill's `plain_utterances` as examples**
  (`workflow_agent._matcher_job_types`, `MATCHER_EXAMPLES = 9`): in the
  `MatchSkill` job-type listing a skill's line is followed by
  `e.g. "...", "..."` with its first nine `Signature.plain_utterances`
  (whitespace collapsed); a skill that declares none keeps its plain
  `- name: description` line, and the `job_types` field description mentions
  examples only when some skill has them. Skills without params now carry
  their utterances too (`SkillFile.plain_utterances`;
  `workflow_agent.skill_utterances(folder)` gives them by name). The
  splitter's job types (`_job_types_for`) stay description-only. No span
  fields change. Nine, not four: measured offline on an 18-skill workflow
  with nine utterances each, nine shown matched 0.863 of a tuning set against
  0.834 with four (skill sub-tasks right 562 vs 552 of 567), for about 1.4k
  more prompt tokens.
- **Skills as Python skill files** (`fastworkflow/skill_commands.py`,
  `workflow_agent.load_skills`): a skill is now a file `_skills/<name>.py`,
  laid out and written like a command file at the root of `_commands`: the
  file stem is the skill's name, so it is a Python identifier (snake_case; a
  stem that is not one fails the load with a `SkillFormatError`), a `.py` file
  whose name starts with `_` is a helper the skill files import
  (`from ._kinds import ...`), not a skill, and a subfolder of `_skills` is
  reserved for a future skill context (like `_commands/<Context>/`): it is
  logged and skipped, its files not loaded. `class Signature` carries the description (its
  docstring, what the skill matcher and the text planner read), a pydantic
  `Input` (`str`, `bool`, `int`, `Literal[...]` for an enum; `Field(default=...,
  description=...)`; the check options `in_request` and `verify` in
  `json_schema_extra`; the Phase 11 options were cut to these two), and optionally
  `plain_utterances`, a `template` (the text the planner falls back to) and
  `validate_extracted_parameters(workflow, command, cmd_parameters,
  command_text=None)`. A module-level `steps(input) -> list[str]` returns the
  steps, written with Python f-strings, `if` and `for n in repeat(input.count)`
  (`repeat` refuses more than `REPEAT_LIMIT`); lookup tables are plain dicts; a
  step `@<skill>` includes a skill without params. A skill
  without params is a file whose `steps()` returns its fixed lines. Without a
  declared `template`, the template is `steps()` rendered on placeholder inputs
  (`skill_commands.placeholder_template`): every combination of `<name>` for a
  str (and "" when that is its default), true and false for a bool, an int's
  default and each enum value, merged into one list where a step not every
  combination writes ends with `(only when ...)` (`person is given`,
  `kind is group`, `not flag`) and the steps of an enum's values are joined
  (`<find_group or find_unit, by kind> query=<name>`); a repeated part shows
  its steps for the int's default, so a skill declares `template` when the
  planner should see how it repeats. The file is imported like a command file
  (package-relative imports work); one that fails to import is logged and
  skipped. The one file replaces SKILL.md (see Removed).
- **Skill arguments go through the command parameter screen**
  (`utils.signatures.screen_parameter_values`, factored out of
  `InputForParamExtraction.validate_parameters`, which calls it unchanged): a
  skill call's `str` and `int` arguments lose quotes and angle brackets around
  the whole value, a placeholder or an out-of-range number (a negative count,
  a declared `ge`/`le` bound) is refused (an optional argument takes its
  default, a required one fails the call), and the extractor's "not given"
  number or a made-up extreme counts as not given. A str argument is found in
  the request with Unicode dashes (a non-breaking hyphen) folded to `-`. A skill
  file's `validate_extracted_parameters` sees the checked arguments
  (`skill_commands.validate_call`; `plan_skill_call` takes `workflow=`) and can
  refuse the call (the planner falls back to the template) or correct a field.
  An argument that only applies when other arguments take some value is reset
  there (`if not cmd_parameters.include_risk: cmd_parameters.owner = ""`); skill
  files have no condition option of their own, and a field declaring `requires`
  in `json_schema_extra` fails the load with a `SkillFormatError` pointing to
  the hook. The rendered template runs the hook on each placeholder combination
  and leaves out the ones it changes, so its `(only when ...)` conditions follow
  the hook; a combination the hook refuses is left out too.
- **Skill-call argument checks** (`fastworkflow/skill_commands.py`), all optional
  per parameter (in a skill file, an `Input` field's `json_schema_extra`), so a
  skill without them plans as before:
  - `verify: true` on a `bool` or `enum` parameter: when Jev is configured, the
    planner path (`plan_skill_call(..., check_lm=...)`, passed by
    `build_query_with_next_steps`) has Jev decide that argument again from the
    request alone (one `Choice` call per verified parameter, in parallel with the
    planner's typed call); a verdict at or above `VERIFY_THRESHOLD` (0.65)
    replaces the planner's argument and stands in for a missing or invalid enum.
    An enum's `verify` dict (a `{value: meaning}` map) gives Jev each value's meaning (`"group": "a
    group, which people are members of"`). A failed check leaves the planner's argument.
    This targets a wrong record kind for a name ("Payroll" planned as a feed) and
    flags set or left unset against what the request asks.
  - `in_request: true` on an `int` parameter: when the request writes numbers
    (digits or the words zero to twenty) outside the other arguments' text, the
    argument must be one of them, else it takes the only one written (or its
    default when written) and with several others the call falls back; "the
    first two" no longer runs as the default 3.
  - The model's `steps_before` / `steps_after` lose `<...>` placeholders (with
    the `key=` holding one): `list_logins login_uid=<Ann Lee's uid>` becomes
    `list_logins`, which the agent runs on the record in context instead of
    passing the placeholder on as a value. A step holding a uid-shaped token the
    request does not write is dropped; their descriptions now say so.
  - `in_request: "subject"` on a `str` parameter: the argument must name a record of
    the request: hold one of its subjects (`split_merge.request_subjects`, with
    the workflow's identifier patterns) whole, or a name-like word, and cut no
    subject. "I" of "Before I approve" or "Payroll" of "Payroll_Approver" is
    refused (an optional argument takes its default, a required one falls back);
    a name cut short mid-word or at a space at a subject's start or end
    ("Payroll_Admin", "Ann" of "Ann Lee") becomes the whole subject. `verify: true` on a subject has Jev choose among the
    request's subjects or "none" (a name/person swap, a right's name planned as
    a group's).
  - Any `str` argument inside a match of the workflow's identifier patterns
    becomes the whole match ("4421" of "acct_4421"); commands loaded by
    `load_skill_commands` carry the patterns (`SkillCommand.identifiers`).
  - An extra step naming a command outside the expansion is kept only when a
    part of the request the expansion leaves open asks for it: a word of the
    command's name (not `open`, `list`, `find`, `uid`, ...) shares a stem with a
    request word no word of the expansion's text shares one with. Navigation and
    lookup steps the model adds around the expansion are dropped.

- **Skills as parameterized commands** (`fastworkflow/skill_commands.py`): a
  skill may declare typed parameters (its skill file's `Input`: `str`, `bool`,
  `int` or an enum, with a default, a description and, for a string, the words to
  ignore in `in_request`) that `steps(input)` expands into its steps. The
  description is still what the skill matcher sees; a skill without parameters
  is read as before. An `@<skill>` include is plain text and cannot name a skill
  with params. A malformed skill fails the load with a `SkillFormatError`
  naming its file. On a turn's initial plan, when the sub-task's matched skill
  has params, `build_query_with_next_steps(..., skill_command=...)` asks the
  planner LM for the skill's typed arguments (one `ChainOfThought` over a
  signature built from the params, plus `steps_before` / `steps_after` for parts
  the skill does not cover; same adapter and catalog as the text planner),
  applies the argument guards and expands the skill in code into the numbered
  plan. Guards: a string argument loses quotes and its strip words (a kind word
  appended to a name: "NORTHFEED feed" -> "NORTHFEED", "Ann Lee's login" ->
  "Ann Lee") and must then be written in the request (case-insensitive; the
  request's spelling is used) or the call falls back (an optional one takes its
  default); an optional string repeating another argument takes its default; a
  bool always defaults to false and the signature tells the model to set it only
  when the request asks; an enum must be one of its values; extra steps naming no
  available command, naming the skill, or naming only commands the expansion
  already runs are dropped; a repeat is capped at `REPEAT_LIMIT`. Any failure
  (bad arguments, a guard, an exception) falls back to the text planner with
  the skill's template; replans and hand-off continuations are unchanged.
  `_call_agent_for_query` gets the template and the command from one skill
  match (`workflow_agent.planner_skill_for`; `plan_template_for` is its
  template half). Whether a chosen record kind fits what the find command
  returns is not checked at execution time.
  Span contract: `fw.planner.plan` / `fw.planner.replan` v4 (aggregate 14) add
  `skill`, `skill_args` and `skill_fallback`; `plan_source` may be `skill_call`.

- **Deterministic split merge** (`fastworkflow/split_merge.py`): the
  splitter's sub-tasks now go through `merge_subtasks(user_query, subtasks)`,
  code only, after the model (and its one retry). It drops meta lines ("(No
  tag action is requested.)") and sub-tasks contained in another, joins a
  sub-task that names no subject, or the same subjects as the one before it,
  to that one ("How risky is Ann Lee?" + "Which findings are driving that?"),
  and joins per-record sub-tasks that the request asks to compare ("the
  riskier of Billing and Shipping") into the request's own sentence. When one
  job is left, the request itself runs, verbatim, so qualifiers the model
  dropped ("Do not apply anything") are back. Subjects are found in the text
  alone (names, quoted and back-ticked spans, uids, and the identifiers the
  workflow declares in an optional `_skills/identifier_patterns.txt`, one
  regular expression per line, since identifier formats are the workflow's
  own; "X and Y Ltd" is one unit). A failing merge keeps the model's split.
  Limits on the request that the model dropped ("Apply none of them.", "I do
  not need the full population.") are appended to the sub-tasks they apply to
  (all of them, or the ones carrying the sentence a pointing-back limit
  qualifies), and a name extending another name of the request
  ("Payroll_Approver" of "Payroll") joins that name's sub-task when it points
  back ("among them", in the sub-task or in the request sentence naming it,
  since the model may drop it). A sub-task naming part of the subjects before
  it joins them when it checks a change ("Show me the tags on Ann Lee before
  and after" after "Put `on-leave` on Ann Lee").

- **Split merge, more clean-ups** (`split_merge.py`, code only): a sub-task
  quoting a value the request never gives ("Apply the tag 'offboarding'") is
  dropped; a request sentence naming a record no sub-task names (a job the
  model dropped) is put back, verbatim, in place; a name of a sub-task that is
  a run of words of exactly one request name ("Sons" of "Hart and Sons and
  Sons") stands for that name.

- **Referents written into sub-tasks** (`workflow_agent.resolve_referents`):
  after the merge, a sub-task that leaves some of the request's records
  unnamed gets one Jev call choosing which of them, if any, its "it", "its",
  "that team" or "the customer" stands for, told what the workflow counts as a
  record by its optional `_skills/subject_hints.md` (natural language). The
  choice is written in (`split_merge.with_referent`): the one pointing phrase
  is replaced ("its holders" -> "Payroll_Approver's holders"), or the record is
  appended ("(for Ann Lee)") when there are none or several. Without a Jev key
  nothing changes.

- **Referents next to their description, and cut subjects put back**
  (`split_merge.py`, `workflow_agent.resolve_referents`): when Jev has chosen
  the record a sub-task points back to and the sub-task holds a description
  of it ("the customer `on-leave`"), a second Jev call picks that phrase and
  the record is written right after it ("Tag the customer Ann Lee
  `on-leave`"), instead of only into "its tags". A sub-task taken from a
  request sentence that lost one of that sentence's subjects is replaced by
  the sentence itself, even when another sub-task names the subject.

- **Placeholder arguments refused** (`utils/signatures.py`,
  `InputForParamExtraction.validate_parameters`): a parameter value that is
  one bracketed token is unwrapped when it holds a real value
  (`<9a1b2c3d...>`) and refused, with an error telling the agent to pass the
  real value from an earlier output, when it is only a placeholder
  (`<item_uid>`, `<uid>`). Every way a command gets its parameters passes
  through this check.
  - A value wholly inside one quote pair (straight, typographic, guillemet or
    backtick) loses the quotes: a filter of `"Ann Lee"` runs as `Ann Lee`.
    Quotes the request itself writes inside quotes are kept.
    `unwrap_quoted_value` is public for other callers.
  - A placeholder-shaped value without brackets (`desired-tag`, `your_uid`,
    `uid` for an `item_uid`) that the request never writes is refused the same
    way. In agent mode it is checked against the user's message, not the
    agent's command.
  - The extractor's "not given" marker for an int (`-sys.maxsize`) no longer
    reaches the command on a field with a default or an Optional one: the
    field gets its default (`None`). The same goes for a number the model
    made up (int32/int64 limits, beyond 2**53, inf/nan); on a field with no
    default it is a parameter error. Declared `ge`/`gt`/`le`/`lt` bounds are
    enforced, and a field named as a position, count or size (`start`,
    `offset`, `limit`, `page_size`...) with no declared lower bound may not
    be negative.
  - `fw.nlu.param_extraction` v4 records `unwrapped_quoted_values`,
    `defaulted_sentinel_values` and `out_of_range_values` (aggregate span
    contract 15).

- **The repeat block covers the same call only**: a command that returned
  the same result twice in a row is refused in the next step only when it is
  sent with the same text again; it stays in the available commands, and with
  other arguments (`start=42`, the next page) it runs. Before, the whole
  command was blocked for that step, so its next page could not be read.

- **Composite skills**: a step that is exactly `@<other-skill>` is replaced,
  when the skills load, by that skill's steps (recursively, in order; its
  description is not included). The
  splitter's job types, the skill matcher and the planner template see only
  the expanded steps. An unknown skill or a cycle raises `ValueError` naming
  the include chain. A skill without `@` lines loads as before.

- **Planning commands can run other planning commands** (fix-3rsq.5,
  `fastworkflow/workflow_execution_context.py`). A planning command's step may
  name another one. `run_planning_command` runs it as a nested frame, and the
  calling step gets the child's whole run back as one run. A command already
  running is refused: "<name> is already running; call its steps directly."
  A call cycle is a load error (see below).
  - Nested calls share one turn-wide step budget, `FW_TURN_MAX_STEPS` (default
    400), which is now enforced. Once it is spent, a nested call returns "The
    step budget for nested commands is spent; continue with the remaining steps
    directly."
  - `ask_user` inside a nested call suspends the turn with its stack intact.
    The user's reply resumes it there, also after the session is restored.

- **Planning commands are in the executor's and planner's catalogs**
  (fix-3rsq.5, `fastworkflow/workflow_agent.py`). They are listed by the same
  context rule as execution commands, and an agent step that calls one from
  another context is refused, as it is for an execution command. MCP
  `list_tools`, the capability answer and training still leave them out.

- **Load errors for call cycles and for `from_` on a callee** (fix-3rsq.5,
  `fastworkflow/skill_commands.py`). The loader builds a graph of the planning
  commands each one's steps call. A call cycle is a load error for every
  command on it. A producer `from_` on a planning command's Input that names
  one of its own callees is also a load error; the message says to write
  `Field(description=...)` instead.

- **A refused planning-command call is asked as a question** (fix-3rsq.6,
  `fastworkflow/workflow_agent.py`). When the planner's typed call of a
  matched planning command is refused, an attended turn that has not yet asked
  or run a command puts the refusal to the user, e.g. "I can't run member
  report yet: your request does not say which member to use. Which member
  should I use?", and suspends on it; the reply completes the call. An
  unattended turn plans from the command's template as before. The planner
  span records `skill_refusal_asked`.

- **`fastworkflow train` writes the workflow's vocabulary** (fix-3rsq.6,
  `fastworkflow/utils/workflow_vocabulary.py`). `___command_info/
  workflow_vocabulary.json` is a top-level, unversioned file derived from
  command metadata (kind words, field names, kind selectors), keyed by a source
  fingerprint. The runtime reads it while the fingerprint matches and derives
  it in memory otherwise. Intent detection and the DSPy examples do not read it.

### Removed

- **`enter_command` and `enter_commands` on a context class** (fix-3rsq.10.4). A context class that
  declares either one is a load error, which names `descends_to` and the command. Declare the move
  with `descends_to` on the command that makes it.

- **`context_hierarchy_model.json`** (fix-3rsq.9). The file is no longer read. A workflow
  that still has one loads, logs one warning that the file is ignored and can be deleted,
  and routes the same way. Parents come from each context's `get_parent`, read from the live
  object chain. `fastworkflow build` does not generate the file.


- **`fastworkflow.skill`** (fix-3rsq.8): the module planning commands import
  is `fastworkflow.planning`; there is no alias.

- **`FW_SKILL_PHASE_MAX_STEPS`, `FW_SKILL_FANOUT_MAX_ITEMS`,
  `FW_SKILL_FACTS_MAX_LIST_ITEMS`** (fix-3rsq.8): no longer read. They are
  the constants `PHASE_MAX_STEPS` (15), `FANOUT_MAX_ITEMS` (10) and
  `FACTS_MAX_LIST_ITEMS` (100) in `fastworkflow.planning_phases`. Remove them
  from env files.

- **`_skills/`** (fix-3rsq.7): a workflow that still has a `_skills/`
  folder fails to load with "_skills/ is no longer read; move these files into
  _commands/ (first make sure any script that regenerates _commands/ preserves
  them; see docs/migrating_skills_to_commands.md)". The checks that existed
  only while two trees did (a skill named like a command, a skill folder for an
  unknown context, the nested-folder warning) are gone with it.

- **Skill `repeat()` / `REPEAT_LIMIT` and `@<skill>` include lines**
  (fix-ajir.18): a skill file's `steps()` loops with `range()` (bound the int
  field with `le`) and shares steps through a helper module
  (`_skills/_<name>.py`). A step written `@<name>` is now plain text.

- **`run_fastapi_mcp --keep_dspy_history`** (fix-50pd): the bounded-history
  "inspect mode" of the server's DSPy memory policy is gone, with
  `install_policy(keep_history=..., history_entries=...)`,
  `INSPECT_POLICY_SETTINGS`, `BoundedHistory` and `SERVER_DSPY_HISTORY_ENTRIES`.
  `fw.llm.call` usage, cost and cache status come from each call's own result
  (fix-sgsp), so DSPy history is now off in every server, the chatbot's
  included; the chatbot no longer passes the flag, and passing it is an error.

- **`FW_SPLIT_REQUESTS`**: request splitting is always on. The switch already
  defaulted to `1` and was on in every evaluation run; `"0"` no longer turns
  it off (the variable is not read), and the commented line is gone from
  `examples/fastworkflow.env`.

- **The finish check's model pass and its switch** (fix-3xts follow-up,
  fix-uyab): the one model call that judged whether each executed step's
  output held what the step asked for is gone, with its filters, prompt and
  "Step N asks for ..." note lines. In a live run it helped alone 3 times, was
  ignored as a false alarm 19 times and led to 1 made-up answer. Gone with it:
  `FW_FINISH_CHECK` (the command check is always on when a run has a plan; no
  `commands`/`full` modes and no off switch), the `LLM_FINISH_CHECK` model
  role and `LITELLM_API_KEY_FINISH_CHECK`, and the `context_budget.FINISH_CHECK`
  budget (`finish_check_max_bytes`, `FW_FINISH_CHECK_MAX_BYTES`), which sized
  only that model's prompt. `finish_check` events no longer carry
  `unmet_by_model`, `vetoes`, `model_called`, `prompt_bytes`, `error` or
  `usage`.

- **The split stage's text files**: `_skills/split_instructions.md`,
  `_skills/identifier_patterns.txt` and `_skills/subject_hints.md` are no
  longer read (nor `workflow_agent._skill_file_text`); their contents go in
  `_skills/_split.py` (`Signature` docstring, `identifier_patterns`,
  `subject_hints`), word for word.

- **SKILL.md skills**: `_skills/<name>/SKILL.md` is no longer read; a skill is
  the file `_skills/<name>.py` only. This covers the plain form of 3.6.0 (the
  description on the first line, one step per line), which becomes a file with
  `class Signature` (the description as its docstring) and
  `def steps(): return [...]` (the step lines), and the front-matter form with
  its `{param}` / `[if]` / `[repeat]` step language, which was never released.
  A folder under `_skills` is not a skill (see the skill-context note above).

### Changed

- **Assistant mode binds `name=value` in code** (fix-weih.1, `ParameterExtraction._extract_impl`). `/add_two_numbers first_num=2
  second_num=3` is now parsed by code before any model, in a `/` command as it already was in agent mode: one LLM call
  fewer, and a `/` message that happens to have that shape is no longer read by the model. XML and positional forms still
  go to the model in assistant mode. The one existing test that pinned the old behaviour,
  `test_assistant_mode_does_not_parse_in_code`, was replaced by `test_assistant_mode_parses_only_key_value_in_code`.

- **A leading `/` is stripped before the exact command-name match** (fix-weih.2, `CommandNamePrediction`). A first message
  `/what_can_i_do` now exact-matches a core command instead of going to the fuzzy and classifier layers, and `/<command>`
  for a command owned by another context returns the same routing hint as the bare name.

- **`LLM_RESPONSE_GEN` is an active model role** (fix-weih.8, `observability.provenance.LLM_ROLE_VARS`). It was recorded but
  read by nothing. `run_chatbot/composer.py` now reads it (key `LITELLM_API_KEY_RESPONSE_GEN`), it is no longer in
  `ORPHAN_ROLE_VARS`, and `config_hash` in model provenance includes the role, so a hash stored before this release will
  differ from the same configuration's hash now. The shipped `fastworkflow.env` sets it to `mistral/mistral-small-latest` (the repo's `env/.env` uses `bedrock/nvidia.nemotron-nano-3-30b`, which needs no `LITELLM_API_KEY_*` entry); the code has no fallback, so unset it turns the composed card and form off.

- **"Rich answers" sends command outputs to `LLM_RESPONSE_GEN`** (fix-weih.10). With the switch on and the variable set (the shipped template sets it), a
  composed card sends the user's message, the answer and each result's command, parameters and output to that model. This
  includes a long (over 400 characters) non-`/` answer in an assistant-mode session, whose command outputs no model had
  seen before. The model is called automatically, once, on an eligible turn (fix-weih.24); a `/` turn
  never composes. Uncheck **Rich answers** to keep the plain answer and send nothing.

- **The final answer of a turn no longer carries an answered prompt's description keys** (fix-weih.7,
  `_without_prompt_description`). `question_kind`, `approval`, `review` and `parameter_request` on an answered `ask_user`
  entry are not merged into the turn's final artifacts; any other artifact still is. The entry keeps `question_kind`.

- **The first parameter-extraction attempt now runs at temperature 0 (greedy); retries keep temperature 1.0** (`FirstAttemptGreedy`, `FIRST_ATTEMPT_TEMPERATURE` in `fastworkflow/utils/extraction_attempts.py`; under test). Before, `dspy.BestOfN` ran every attempt, the first included, at temperature 1.0 with a fresh rollout id. Retries (reward 0 or an exception) still use temperature 1.0 and distinct rollout ids.
- **A DSPy field marker in the splitter's reply is not part of a sub-task** (`workflow_agent._split_items`): a `[[ ## completed ## ]]` the model carried into a sub-task's text is cut before the plan review shows it.
- **A record name with extra words run on is trimmed to the record** (`planning_commands._subject_without_extra_words`, in the `record_named_in_command_text` check): "snapshot of IT Support" and "X: what it exposes, ..." become the name the request writes. A value the request quotes whole, a subject inside a longer identifier, and a sentence-case title opening with one capitalized word are kept.
- **A split longer than `MAX_SUBTASKS` is reduced, never run whole** (`split_merge.reduce_to_cap`): exact and contained copies are dropped, then near-copies (Jev, pairs sharing subjects, probability >= 0.9), then the existing merge joins, then adjacent items of the same planning command (or most similar) are joined until the list fits. Jev calls are bounded (`REDUCTION_JEV_CALLS`). The request runs whole only when the splitter listed nothing usable. A joined item keeps its parts (`JoinedSubtask`): the review shows it as one item, and unless the review edits it, each part runs as its own job (`review_proposals.expand_joined_parts`). Splits at or under the cap are unchanged.
- **Review reader guards** (`review_proposals`, `review_engine`): deterministic checks on the reader's edits. A drop that names a setting of the step clears that setting instead of removing the step; a change is moved to the one step that has the named input; drops the reply did not ask for are not applied (the user is asked); a ruled-out quoted name is not applied; `name=none` clears an input; "yes, but do not apply anything" is an accept.
- **The finish check reviews up to two finishes per run** (`react.FINISH_REVIEWS_PER_RUN`, `finish_check`): a second note reopens the loop once more, a third finish ends the run; the note's closing line tells the agent to run an unmet step, and to report one not done only when its command ran and failed.
- **Drafting is on by default, and a bound sub-task runs with the inputs its review showed** (fix-48np.1).
  `FW_DRAFT_ARGUMENTS` now defaults to on; `FW_DRAFT_ARGUMENTS=off` (or 0, false, no) restores the previous
  behaviour exactly. `resolve_inputs` no longer skips drafted inputs. Measured before the flip (24 multi-sub-task
  requests on the library example and the planning tutorial, 2 fresh unattended runs per arm, Bedrock
  `nvidia.nemotron-nano-3-30b`, blind-judged): task success on 38/48 against off 35/48, about 2.2x faster (median
  turn 8.5 s against 18.9 s) at about a quarter of the model cost; shown == run for 80 of 83 bound sub-tasks, the
  rest an id input drafted with a title (fixed below) and one refusal the agent handled.
  - The library example and the planning tutorial now declare the format of their id inputs (`open_book.book_uid`,
    `open_member.member_uid`, and the tutorial's `show_book`, `list_loans` and `list_member_loans`: `pattern`
    `^b[0-9]+$` / `^m[0-9]+$`, the ids their applications mint). A title drafted into a uid now breaks the
    field's declared rule, so the draft leaves it unbound and the run finds the id as it did before. No framework
    change was needed: a declared constraint was already enforced on drafted values (`validate_parameters` for an
    execution command, `check_input` for a planning command). The validate-command-parameters skill says to
    declare id formats, and why.
  - A sub-task bound to a planning command runs its phases with the reviewed inputs (drafted, edited by the user, or
    `from sub-task N`) as its arguments (`review_proposals.reviewed_arguments`), never extracted again but still
    checked as a typed planning call checks them (`planning_commands.checked_arguments`: `check_arguments`, the
    `validate_extracted_parameters` hook, the `Input`'s rules).
  - A sub-task bound to an execution command runs that command with its reviewed inputs: code runs it as the plan's
    one step (`planning_phases.CodeSteps`, through the agent's command tool and the CME parameter path and its
    validation), and the agent only answers from its output. A `from sub-task N` input whose field declares its
    producer field (`from_("find_book.book_uids")`) takes that field of N's typed output, not the whole output.
  - Reviewed inputs that are refused, or that leave a required field unbound, are handled as before: the agent runs
    the sub-task, with the refusal in its context. Only a sub-task the draft never drafted for its means
    (`SubTask.drafted_means`) has its unbound values drafted at run time, by the draft's own function.
  - Every bound sub-task's shown and run values are recorded on the turn span as `subtask_bindings` (fw.turn v4,
    `SPAN_CONTRACT_VERSION` 37), with the reason for any deviation (refused or unbound so the agent ran it, drafted
    at run time, a value other than the one shown).
  - The review re-drafts a sub-task's inputs when an edit changes its means (`Proposal.revised`), so it never
    shows inputs drafted for another command. `drafted_means` is bookkeeping: it is not rendered, not listed in
    the review's field errors and not editable (`review_engine.shown_fields`, `review_hidden`).
  - The draft runs the same typed fill fallback as `typed_planning_call` for a planning command whose required
    value the extractor leaves missing or refused (`planning_commands.typed_fallback_arguments`, one planner call).
  - Measured latency (Bedrock `nvidia.nemotron-nano-3-30b`, `FW_LM_CACHE=0`, library example): drafting adds a
    median 1.2 s per sub-task (one extractor call in 19 of 22 drafts), 4.3 s median per request; a sub-task that
    reaches the typed fill fallback costs three calls, 2.6 to 3.8 s.
- **Library example and planning tutorial workflow: `overdue_summary` takes an optional `branch` and its answer
  names the branches** (fix-1744). The input (`in_request`, ignoring "the"/"branch"; words for every branch mean
  the whole library; a name that is no branch is refused with the branches) is passed to `count_overdue`, and the
  `branches` phase hands `list_branches`'s `Output` on to the answer step. Tutorial page 1 is updated. No
  `plain_utterances` changed, so no utterance is generated again; the `Signature` docstring (a seed row of the
  Library planning head) did change, so the next `fastworkflow train` retrains that head selectively. The
  current heads keep working until then.
- **Library example: `note_member.text` is `in_request`, and a note that names the member is refused**
  (fix-46mm). An invented or paraphrased note is refused and extracted once more; a text that holds the request
  around the note ("Note on Mia Chen's record that ...") is refused by an `Input` validator. Tutorial page 2 is
  updated. The `author-planning-commands` skill has the lessons of all three fixes.

- **A planning command's arguments come from the shared parameter extractor, not the planner's typed call** (fix-48np,
  stage 2). `typed_planning_call` now fills a planning command's arguments with `extracted_arguments` (the extractor
  execution commands use, on `LLM_PARAM_EXTRACTION`, with the copy-verbatim sentence for `str` fields) and runs them
  through the same `check_arguments`, `validate_extracted_parameters` hook and `Input` rules as before; a refusal is
  extracted once more with the refusal appended. The planner call now only proposes `steps_before` / `steps_after`
  for the filled call, and `check_verdicts` is unchanged. When a REQUIRED argument is still missing or refused after
  that retry, its value is taken from the planner's typed fill (`_typed_fallback`) and checked the same way; the
  fallback never overrides a value the extractor gave, never adds a value the checks refuse, and records a
  "typed fill fallback" note in the plan span's `skill_notes`. Measured (pre-registered noise-aware gate, Haiku 5.5
  on Bedrock, 149 planning requests, two fresh runs per arm, hand-judged references): arguments right 199 and 198 of
  205 against 192 and 196 for the typed call; right->wrong changes 0 and 2, within the typed call's own run-to-run
  noise (6), and in a class the typed call produces itself (a value where the request gives none); no bare surname
  dropped (14 per run); nothing the planning checks refused is accepted (38-case strictness corpus). The fallback
  fired on 6 of 149 requests per run. Tests that script the planner's typed answer now script the extractor too
  (`tests/scripted_extractor.py`).

- **The sub-task review's draft shows each sub-task's arguments, from the shared parameter extractor** (fix-48np,
  stage 1: display only). `draft_round1_proposal` fills each sub-task's inputs with the values the shared extractor
  gives on the sub-task's text (`parameter_extraction.extract_arguments`: the LLM extractor execution commands use,
  with their checks, and for a planning command `check_arguments`, its hook and its `Input`'s rules, as
  `typed_planning_call` checks them; a refused or missing planning value is read once more with the refusal after the
  text). A field whose declared producer (`from_`) an earlier sub-task runs, as its command or as a step of its
  planning command, is shown as `from sub-task N` instead of being extracted. A refusal is shown on the line as
  before. The drafted inputs (`SubTaskInput.drafted`) are shown and not run: `resolve_inputs` skips them, so the
  sub-tasks run as they did. **Off by default**: `FW_DRAFT_ARGUMENTS=on` turns it on. While `typed_planning_call`
  still produces the values that run, the two disagree on about 1 argument in 20, so the review could show a value
  other than the one that runs; stage 2 (the typed call takes the drafted values) is not landed. An extraction that
  fails leaves the inputs empty and logs a warning. Measured (gate 1b, Haiku 5.5 on Bedrock for both arms, no Jev, 149 planning requests
  from the tutorial and library test sets, two runs, hand-judged references): the shared extractor agreed with the
  reference on 403 of 410 scored arguments, `typed_planning_call` on 389 (discordant 19 vs 5; exact McNemar
  p=0.146 and 0.039 per run). The typed call accepted deictic values ("this book", "the books") that the
  extractor leaves missing; the extractor missed a bare surname twice and paraphrased a free-text note three times
  (the typed call tells the model every `str` is copied verbatim; the extractor said so only for `in_request`).
  The extractor now tells the model that each `str` field of a planning command's `Input` is copied exactly as the
  request writes it (`signatures.planning_verbatim_sentences`, the `IN_REQUEST_STR_SENTENCE`), as the typed call
  did; an execution command's fields are unchanged. Re-measured the same way: 405 of 410 against 392 (discordant
  16 vs 3; per run p=0.109 and 0.039); the paraphrased notes are gone, and the extractor still misses "Austen" in
  "list every Austen book we own" (answers it not given about one time in three).

- **Production listing is the shortlist above 7000 characters** (fix-3rsq.plan-review.2.10). The agent's listing
  uses the threshold arm: the full list up to 7000 characters, and the shortlist of the matching commands with their
  producers and navigation above it. A workflow with no trained heads falls back to the full list at every size.

- **`verify` on a bool that defaults to true asks which value the argument takes** (fix-jd9o.9).
  Before, the check asked "Does the request ask against this?". Jev read that double negative close
  to a coin flip: 0.47 on one field, and 0.70 across bools that default to true. The check now asks
  "Which value should the argument take?", with the options "true: the request does not mention
  it, or says or implies it" and "false: the request says not to, or says the opposite". On a fresh,
  blind-labelled set that scored 0.97, and no field got worse. A bool that defaults to false keeps
  its wording, because neither form won there: each failed a different kind of case.
- **Navigation warns once when a command moves the workflow into a context it does not declare.** A
  warning names the command and the target, and says to declare `descends_to`. It is logged once per
  command and target in each process (fix-3rsq.10.4).
- **Assistant mode uses the same rule for planning commands** (fix-3rsq.10.8). An exactly named
  planning command that is mechanical runs. One that needs the agent is refused with the reasons,
  such as a step the agent carries out in prose, a branch decided at run time, or a `verify` check.
- **Planning commands in the listings** name their approval marker, and a planning command's
  template shows the annotated commands it calls.

- **Every trained workflow must be retrained** (fix-3rsq.9). A workflow trained by an
  earlier release refuses to load and says "retrain with `fastworkflow train`". The first
  train after upgrading is a full retrain.

- **The reserved "not here" class is one seed per foreign command** (fix-3rsq.9.6). Each
  context's `wildcard` (not-here) class is built from one hand-written seed of every command
  that is not one of the context's own or inherited labels, planning commands included.
  Ancestor sampling and the ratio fill are gone. The internal CME workflow has no not-here
  class. `--regenerate-utterances` chooses the seeds again.

- **Confident global commands run first for natural language** (fix-3rsq.9). For a request
  in natural language, the global scope is asked before the current context. If it names one
  command with confidence, that command runs, bound to the root object. Otherwise the
  context's own scope is asked and the climb continues as before. Exact command names are
  unaffected: they take the nearest scope.

- **A bare value ends the lookup** (fix-3rsq.9). When a context answers `parameter_value`, no
  enclosing scope is asked about the request.

- **The planner and executor listings show enclosing and global commands** (fix-3rsq.9). The
  planner's list shows the commands of enclosing contexts and the global commands as callable
  here, each with its scope. A plan never names a command the executor would refuse.

- **Navigation messages name the root context** (fix-3rsq.9). A message about going back
  up names the root context on the way back, for example when a record was opened under the
  root.

- **Re-issuing "open this record" from inside that record runs in place** (fix-3rsq.9). A
  command that opens the record the workflow already stands on runs there, instead of being
  refused "already in context".

- **Authors: one-word command names reach across the path** (fix-3rsq.9). The first word of
  a request is matched against command names along the whole path. A one-word command name in
  an enclosing context can capture a request that starts with that word. Review such names.


- **Skills are planning commands, by name** (fix-3rsq.8). Modules:
  `skill_commands` → `planning_commands`, `skill_phases` → `planning_phases`,
  `skill_preview` → `command_preview`. Classes: `Skill*` → `PlanningCommand*`
  (`SkillFormatError` → `PlanningCommandFormatError`, `SkillCommand` →
  `PlanningCommand`). Functions: `load_skills` → `load_planning_commands`,
  `warm_skills` → `warm_planning_commands`, `match_skill` →
  `match_planning_command`, `plan_skill_call` → `typed_planning_call`.
  Folders: `docs/tutorials/skills/` → `docs/tutorials/planning_commands/`,
  `docs/skill_*.md` → `docs/planning_command_*.md`, the
  `author-workflow-skills` coding skill → `author-planning-commands`. Log,
  error and CLI messages say "planning command". Span names, span attributes
  and saved session keys keep their `skill` names.

- **`PlanGenerator.__call__` is `__phases__`** (fix-3rsq.8): the method that
  orders a planning command's phases is `__phases__`, and a `PlanGenerator`
  that defines `__call__` fails to load with "PlanGenerator defines __call__;
  rename it to __phases__".

- **Planning commands live in `_commands/`** (fix-3rsq.7,
  `fastworkflow/workflow_agent.py`, `fastworkflow/command_directory.py`,
  `fastworkflow/skill_preview.py`). A planning command is a file in
  `_commands/` (root or one context folder) with a top-level `PlanGenerator`,
  beside the execution commands; the split settings file is
  `_commands/_split.py`. To migrate, move each `_skills/` file, with the `_`
  helpers it imports, to the same place under `_commands/`, make sure any
  script that regenerates `_commands/` preserves them, and regenerate pinned
  previews (`fastworkflow command preview <workflow> --out <dir>`), whose
  source line names the new path.

- **A step may run a planning command at load** (fix-3rsq.7,
  `fastworkflow/workflow_agent.py`). The load-time step and phase-source checks
  read the signatures of planning commands as well as execution commands, so a
  planning command whose step runs another planning command loads.

- **A prompt stored as pieces is no longer also kept as a 16 KB prefix**
  (fix-ajir.37): an `fw.llm.call` that records `prompt_slots_ref` keeps only
  the cut envelope of `messages` (`value: ""`, `replaced_by:
  "prompt_slots_ref"`; length and sha256 unchanged), as a call blanked by the
  trajectory manifest already did. The prompt is read from the pieces
  (`prompt_as_sent`, `/api/prompt/…`, "LLM input as sent"). About 5% of a
  heavy store's file. When two enrichers blank one envelope, the first keeps
  its `replaced_by`.

- **Skill API surface** (fix-ajir.19): `fastworkflow.skill_phases.__all__` is
  the author-facing API (`Phase`, `concatenated_steps`, `fact_from`,
  `facts_text`, `PhaseError`, and the helpers a workflow's tests use to check
  its skills without a model); the other names stay importable but are the
  framework's. A skill `Input` field may be `Optional[str] = None`: the planner
  sees a str that defaults to empty, and `steps()`, `plan()`, the hook and the
  validators get None when the request leaves it out. A `str` default such as
  `"NOT_FOUND"` is still a value written into the steps (now documented).

- **The finish check runs on every run that has a plan** (fix-3xts follow-up):
  it no longer waits for `FW_FINISH_CHECK`, and it reads plan steps against
  their commands' runs only. A run with no durable archive gets no note
  (`reason: "no archive"`) instead of every step reported as not run.

- **Finish check: fewer false "not run" notes** (fix-uyab, part):
  an archived run counts for the command its text resolved to, read from the
  session's action log (`resolved_command_names`; `check`, `attribute_rows`
  and `review_finish` take `resolved=`), so an alias satisfies its command's
  step; a step with "(skip when ...)", "(skip if ...)" or "(skip unless ...)"
  after its command is conditional; and a step that writes an argument right
  after its command (`command key=<...>`) is not satisfied by a run of the
  command with no arguments at all.

- **A skill call's extra step is kept for a synonym its command uses**
  (`fix-9x93`; `skill_commands.command_texts`, `command_descriptions`,
  `command_utterances`): `filter_extra_steps` (`texts=`) matches the request's
  open words against a command's name and the words that recur in at least two
  of its texts (the first sentence of its catalog description and each of its
  `plain_utterances`; a command with one text gives that text's words), not
  the name alone, so "every invoice" now keeps `list_bills` whose utterances
  say "invoices". `plan_skill_call` builds the texts from the catalog and the
  workflow's routing definition (no model call). "please", "want", "need" and
  "has" joined the filter's stop words.

- **Every experiment pins a benchmark version** (`fix-ujat`): the store and
  `ExperimentController.create_experiment` refuse an experiment without
  `benchmark_id`, `benchmark_version` and `benchmark_digest_sha256`
  (`MissingBenchmarkPin`). An unpinned experiment could carry any task ids
  and sat in the Benchmarks tab beside the real benchmarks, with a contest
  and a winner of its own. Ad-hoc work is plain conversations (`/initialize`
  with no experiment bootstrap), listed under **ad-hoc conversations**.
  Removed: the "Experiments without a benchmark" rail branch and the
  ad-hoc comparison group (`selection.GROUP_ADHOC`);
  `comparison_group_identity` refuses an unpinned row. An older store's
  unpinned experiments are left off the rail and their turns list with the
  ad-hoc conversations.

- **Experiments driven by a user simulator** (`fix-va9v`): an experiment can
  be created with `user_simulator: true` (UI registration
  `POST /api/benchmarks/<id>/experiments`, duplicate, or
  `ExperimentController.create_experiment(user_simulator=True)`). Its
  simulator turns each task prompt into its own utterances, so its runs'
  first messages are not checked against the prompts: validity reads
  `simulated_user` instead of valid | invalid. Completeness is still checked.
  The flag is set once, at creation; a registered experiment takes it from
  its registration. It shows read-only on the experiment and registration
  pages and in their API responses. **Schema v10** adds
  `experiments.user_simulator` and `experiment_registrations.user_simulator`.
  A v9 store is refused, as with every older schema, and is not migrated.

- **Benchmark experiments are checked against their version** (`fix-mgoi`):
  every experiment that names a benchmark version must declare exactly that
  version's task IDs, whether or not it was registered in the UI (before, only
  UI-registered ones were checked). Each recorded experiment now reports two
  verdicts, derived from its attempts and recorded turns: **complete |
  incomplete** (every benchmark task got its declared runs) and **valid |
  invalid** (every run's first message matched its task's saved prompt; tasks
  without a saved prompt are not checked). Both show on the benchmark page's
  experiment cards with "N of M tasks", and on the experiment page with the
  runs that are short or that sent another prompt. A store verdict of
  `invalid` for missing runs (`attempt_shortfall`, `never_completed`) now
  reads "incomplete" in the UI.

- **Experiments and winners are partitioned by benchmark version**
  (`fix-873a`): each version of a benchmark is its own contest, so the Studio
  lists one version's experiments at a time (rail: benchmark → version →
  experiments) and each version has its own winner. The winner is now plain:
  `POST /api/experiments/<id>/winner` sets it, `DELETE` removes it, and having
  no winner is normal. Removed: the automatic first-experiment winner, the
  `keep`/`undecided` decisions and `expected_selection_id` on the winner,
  `POST .../winner/decisions`, `GET /api/benchmarks/<id>/experiments` (now
  `.../versions/<v>/experiments`), and the refusal to delete the current
  winner (deleting it leaves the version with no winner). Existing live DBs
  are not migrated: experiments already in a cross-version group stay there.
  `create_experiment`'s
  `initialize_winner` is renamed `join_comparison_group`. Per-task best runs
  are unchanged.

- **The observability record bounds itself in long-lived servers**
  (`fix-nczr`): the writer re-runs the retention prune hourly instead of only
  at recorder startup, and once only turn records are left over
  `FW_OBS_DB_MAX_BYTES` the oldest finished non-experiment turns are evicted
  whole, so the cap is a real bound. No new settings. A malformed `FW_OBS_*`
  integer now logs a warning, and so does an unsealed experiment holding the
  file over the cap.

- **`fastworkflow skill preview` is now `fastworkflow command preview`**
  (fix-3rsq.5, `fastworkflow/cli.py`). There is no `skill preview` alias. Every
  app workflow's pinned previews go stale, so regenerate them with
  `fastworkflow command preview <workflow_folder> --out <dir>`.

- **Planning command previews show calls and depth** (fix-3rsq.5,
  `fastworkflow/skill_preview.py`). A preview adds a `Calls:` line (the commands
  its steps call, or `none`) and a `Depth:` line: the longest chain of
  planning commands from it, a lower bound while open steps name no command.
  The `Kind:` line is gone.

- **`process_action` refuses while a question waits for the user** (fix-3rsq.5,
  `fastworkflow/workflow_execution_context.py`). A direct action sent while
  `ask_user` waits for the reply returns "A question is waiting for the user's
  answer; reply to it before running an action." The suspended turn is left as
  it was; the user's reply resumes it.

- **The typed planning call is shown a trimmed list of commands** (fix-3rsq.6,
  `fastworkflow/workflow_agent.py`). Each command appears with its name, the
  first sentence of its description and its `parameters:` line, which cut the
  call's prompt by about 79% with no measured loss. The text planner still
  sees the full list.

- **Parameter extraction treats placeholder values as not given**
  (fix-3rsq.6, `fastworkflow/utils/signatures.py`). A value that is the name
  of a command (the called one's, or another's in identifier form), the string
  `None`, or a free-text value that is only a pronoun or only a kind word or
  field name of the workflow counts as missing, so the missing-parameter path
  asks for it. A field whose description lists its choices ("One of: ...") or
  whose labeled examples hold kind words is exempt from the word screen.
  `fw.nlu.param_extraction` records `command_name_values` and
  `word_screen_values`.

- **A planning command typed by name is matched by name** (fix-3rsq.6,
  `fastworkflow/_workflows/command_metadata_extraction/intent_detection.py`).
  Planning command names join the exact-match set and the cross-context
  inventory, so they no longer fall to the classifier, which has no labels for
  them.

- **Span contract v30** (fix-3rsq.6, `fastworkflow/tracing.py`):
  `fw.planner.plan` / `fw.planner.replan` v8, `fw.agent.execute` v3 and
  `fw.nlu.param_extraction` v7.
- **`@phase` with both `output=` and a return annotation is a load error** (fix-3rsq epic note, D). The error names the phase: use the return annotation when the facts type is fixed (or a union), and `output=` only when the type depends on the input.
- **`fastworkflow command preview --check` warns on a `str` input whose description lists kind words** (fix-3rsq epic note, H). A warning, not a failure; if the set of kinds is fixed, type the field as a `Literal`.

### Fixed

- **Typed review edits are no longer silently lost or misread** (fix-2vyv). Several `change N: name=...` edits of one
  step are merged by applying them in order over the step's inputs; when the merge cannot be written back as one text
  without losing a part, the edits are now applied as written instead. Before, `change 1: inputs=check=strict; change 1:
  member=Zed` applied `member` and dropped the `check` edit (a goal, outcome, means, check or desired_outcome input, a
  cleared input, or a quoted value that split again were all lost), and `change 9: a=1; change 9: b=2` raised
  `IndexError` instead of saying there are only N steps. A reply that holds a `change N:`, `drop N` or `add after N`
  clause is no longer an acceptance: `change 1: goal=Run it as shown` sets that goal (before, the phrase accepted the
  plan and the edit was dropped), and `change 1: goal=X; run it as shown` is asked again instead of accepting and
  dropping the edit. A quoted `"none"` or `"null"` is refused with a message (before, it cleared the input; unquoted
  `none` still clears), `change 1: goal=none` says why it is refused, and a quoted value that holds its own quote mark
  and a `, name=` separator (`book='it's "a", b: 2'`) is refused with a message (before, it split into a stray input).
  The chat review form now composes the cases the server reads exactly (an input named check, goal, outcome, means or
  desired_outcome beside another input edit, a quoted value beside another input edit, accept-sounding wording in a
  value). A plain `yes` or `run it as shown` is accepted as before. Limits: the literal words none and null cannot be
  set as an input value from typed text, and an unquoted `, name=` still starts another input.
- **A packed `inputs=` edit no longer drops an input named like a step field** (fix-j6xa). `change 1: inputs=member=Z,
  check=x` applied `member` and silently dropped `check` (also `goal`, `outcome`, `means`); it is now refused, naming the
  part and saying how to write it (`change 1: inputs=check=x`, one input per edit, or `change 1: check=x` for the step's
  field). The path that turns a model's reading of a plain-language reply into edits merged several one-input edits of a
  step into one text and lost the part named like a field; it now keeps those edits as written. A packed part in a
  model's reading is still dropped as before: the model echoes a shown step's means and outcome into its inputs text, and
  dropping that echo keeps the reading correct. Every other reading is unchanged.
- **`fastworkflow run_fastapi_mcp` forwards `--expect_encrypted_jwt`, `--cors_loopback_only` and `--enable_admin_endpoints`** (fix-qavj). The wrapper had no such arguments, so the signed-token setting and the other two server options were reachable only through `python -m fastworkflow.run_fastapi_mcp`. Each is a flag with the server's own default (off) and is appended to the server command line only when given, so leaving it out is exactly the server's default. The server reads none of the three from an environment file, so there is no env value for the flag to override. `tests/test_cli_run_fastapi_mcp_forwarding.py` also compares the two parsers, so a public server option the wrapper cannot take fails a test.
- **The chat `/` parameter form no longer collects a password parameter** (fix-weih.16). A `string` parameter with
  `format: password` or `writeOnly` (what a pydantic `SecretStr` gives) used to get a masked input, but **Run** sent
  `/command name=value` as ordinary chat text, so the secret was echoed in the user bubble and recorded in the
  transcript and the observability store. The form now shows a labelled notice row in place of the input ("Not
  collected here: ... Supply it another way, or type the command yourself if you accept that."). **Run** and **Edit as
  text** leave the parameter out; when it is required, **Run** is disabled and a status line names the parameter.
  The question form and the approval card already refused or masked it, and now share the one detector
  (`tmIsSecretProperty`).
- **A `/` command held for approval no longer leaves later messages on the deterministic path** (fix-weih.15). A user-typed
  `/` command that waited for approval left the session in assistant mode, so the next agent message was routed as if it
  were a `/` command.

- **FW_LM_CACHE=0 also turns off the Jev classifier's response cache** (`dspy_utils.lm_cache_enabled`, `workflow_agent._jev_lm`). Jev's LM was built with `cache=True` regardless, so repeated experiment attempts replayed earlier Jev answers (937 of 1,233 Jev calls in one three-run benchmark were cache hits) and run-to-run variance was understated.

- **A split whose sub-tasks field has no numbered item is retried once, not taken as the request** (fix-variance, `workflow_agent.split_into_subtasks`): a visibly broken reply gets the existing retry at temperature 0.7; two such replies leave the request whole, and a genuine one-item split is still returned verbatim.
- **A planning command call whose first phase's lookup finds nothing is planned again once** (fix-variance, `workflow_execution_context._restart_after_empty_first_phase`, `planning_commands.reextracted_arguments` / `replanned_call`, `argument_checks.PHASE_FAILED_EXTRACTION_NOTE`): the arguments are extracted again with the phase's observation after the request, and the call restarts only when they differ; the restarted phases carry `reextracted` (span contract v40).
- **Verify options with duplicate labels build, so the check runs** (fix-variance, `planning_commands._distinct_labels`, `_check_one`): the first option of each label is kept, and a check that is still skipped (`check_verdicts`) is logged at INFO with its reason.
- **A backslash escape of punctuation in a key=value value is read as the character** (fix-variance, `parameter_extraction.unescape_punctuation`, `ParameterExtraction._extract_parameters_from_key_values`): `\@` reads as `@` and `\_` as `_`; other backslashes are kept.
- **Verify classifier calls are traced under the turn, and their verdicts are on the planner span** (fix-verify-finish, `planning_commands.check_verdicts`, `typed_planning_call`): the worker threads run under the submitter's context, so each call leaves an `fw.llm.call` span; `skill_notes` lists each verdict read (span contract v39).
- **A request split is cleaned before it runs** (fix-5kz5, `workflow_agent.split_into_subtasks`, `split_merge.without_repeated_clauses`): a numbered list written inline after the model's reasoning gives one sub-task per item, a clause the model wrote twice inside a sub-task is written once, a sub-task the model listed twice is one job, and a request that comes back as one job is returned as the user wrote it.
- **The agent's command tool refuses a blank command, or one that names the tool itself, unrun** (fix-wfxt, `workflow_agent.execute_workflow_query`, `command_executor.invoke_command`, `run_listing.head_answers`). Blank text reached intent detection, whose predict raises `Input text cannot be empty` and ended the turn, and `execute_workflow_query` as a command was classified as an utterance; both now return a failed step that tells the agent to name a workflow command.

- **A facts field typed with a command's `Input` fills from the run's input** (fix-3rsq.15.5, `typed_phase_facts._role_for_model`). The run-time fill passes no command signatures, so such a field was filled from the command's Output and failed validation, breaking the phase. A command's `Signature.Input` is now known by its class name.

- **A planner that takes only one temperature now splits requests** (fix-z62t, `dspy_utils.call_dropping_rejected_params`, `workflow_agent.split_into_subtasks`). The splitter calls the planner at temperature 0.0 and retries at 0.7. Bedrock Claude Haiku 5.5 takes only `temperature=1`, so litellm refused both calls before sending them. The error was caught, every request ran whole, and each turn logged the same warning. Nothing set `litellm.drop_params` outside `ChatSession.start_workflow`, so the FastAPI server and other direct `WorkflowExecutionContext` users always hit this. Now a call that litellm refuses for a parameter (`UnsupportedParamsError`) is sent once more with `drop_params=True`, and later calls on that model in the process are sent with it from the start. A model that takes the temperature is called exactly as before, with no `drop_params` added (adding it to every LM would move native `openai/` and `anthropic/` models onto LiteLLM). When the temperature is left out, the splitter's retry skips the response cache so it is a fresh answer. The observation search (`search_memory`, temperature 0) uses the same fallback. A split that fails is logged at WARNING once per process for each planner model and root cause, and later at DEBUG. Live: 5 two-job library requests with Haiku 5.5 as planner, 0/5 split before and 5/5 after, for $0.0016.

- **A plain `str` field that declares a default takes it when the extractor leaves it unstated**, instead of being reported missing and asked for. `default_unfilled_optional_strs` gave defaults only to `Optional[str]` fields, and the missing check counts only `Optional[...]` fields as optional, so `reason: str = "ordered by mistake"` (retail `cancel_pending_order`) or `branch: str = ''` (library `overdue_summary`, `count_overdue`, `library_hours`) asked the user when the model answered NOT_FOUND. A field whose default is the NOT_FOUND marker itself (the old required idiom) is still reported missing. Planning commands already gave a missing argument its default.

- **Two independent jobs in one request are no longer joined back into one sub-task** (fix-bydz, `split_merge.merge_subtasks` `same_job`, `workflow_agent._same_request_check`). The split model returned "Who wrote Dune?" and "Who has Dune out?" for "Who wrote Dune and who has it out?", and the merge's rule 3 (a part naming no subject, or the same subjects as the part before it, is no new job) joined them, so only one job ran. With `JEV_API_KEY` set, rule 3 now asks Jev whether the two parts are separate requests or steps of one, and joins them only when Jev says one request with probability at least `SAME_REQUEST_THRESHOLD` (0.2); "find X, then open it" is still joined. Without the key, or when the Jev call fails, rule 3 joins as before. A limit sentence ("Apply none of them.") is joined without asking. `split_merge` itself still makes no model call: the decision is passed in as a callable.

- **The conversation summary is no longer part of the request the splitter splits and the merge returns** (fix-bydz, `split_into_subtasks(..., conversation=)`, `merge_subtasks(..., context=)`). The earlier turns' summaries were prepended to the request ("conversation summary: ... new_user_query: ..."), so when the merge joined everything back it returned that whole text as the one sub-task, and the summary's names changed how a request merged. The splitter now gets the user's new message as the request and the earlier turns as a separate `conversation` input (present only when there are earlier turns, so a first turn's prompt is unchanged), which it uses to resolve words such as "its loans" after a turn about Dune. The merge treats only the new message as the request; it reads the conversation only so that a name a sub-task takes from it is not dropped as an invented record. An unsplit request still runs with the summary, as before.

- **Parked pending state of a deleted or archived conversation is dropped** (fix-ayez, `WorkflowExecutionContext.drop_parked_pending`, `run_fastapi_mcp/utils.py` `prune_parked_pending`). The state is pruned against the observability store when it is saved, before a conversation id is minted, and when a saved blob is restored, so a reused id cannot pick up a deleted conversation's state. The archive and delete routes live in the separate chatbot server, so the prune runs at save time rather than at the delete.

- **A new conversation no longer answers a previous conversation's paused question or approval** (fix-newconv-pending). Questions and reviews are parked with their conversation and come back when it is active again; approvals are cancelled.

- **An execution command's extracted values are checked against the request, as a planning command's arguments are** (fix-qjf4, `fastworkflow/utils/argument_checks.py`). When the LLM extractor fills an execution command's parameters, each value it gives is checked against the command text it read (the user's message in assistant mode, the agent's command in agent mode; earlier user messages are not counted, since the extractor never sees them): a `str` must be written there, ignoring case and Unicode dashes, and takes the request's spelling; punctuation alone, a template token (`{name}`, `NAME_1`, "unspecified"), an invented example value ("John Doe" the request does not write) or a copied description example is refused; a `str` whose allowed values live in `json_schema_extra={"enum": [...]}` or in a `One of: a, b` description must be one of them (the declared spelling is taken, and a `json_schema_extra` enum is now listed to the model as its valid values: fix-1pfu.enum-in-extra); an `int` must be a number the request writes, when it writes any. A refused value sends the extraction back once with the refusal appended (as `typed_planning_call` does with `REFUSED_CALL_NOTE`); a value still refused is treated as not given, so a required field is asked for and an optional one takes its default. A value that is not given (including an absent `Optional[int] = None`, which the planning rule refused when the request wrote two numbers) or equal to its default is not checked, and an `int` is refused, never replaced by the one number the request writes. Exempt from the request check: types other than `str` and `int` (dates, floats, lists), a field with a `pattern`, and a field marked `json_schema_extra={"in_request": False}` for a value converted on purpose (a date from "next Monday"). Planning commands keep `check_arguments`; `FW_EXTRACTION_ARGUMENT_CHECKS=off` turns the checks off. Span contract v36: `fw.nlu.param_extraction` v10 records `argument_check_refusals`, `argument_check_retry`, `argument_check_refusals_final` and `argument_check_respelled`. Measured offline on the app's extraction replay: 0 of 1,189 values refused in passing Haiku runs (0 of 974 calls would retry); on 468 Nemotron cases, 32 of 107 wrong values caught with 0 correct values refused; retries on 101 of 738 Nemotron calls.

- **An HTML-escaped `&lt;UNKNOWN&gt;` from the extractor is the not-given marker too** (fix-3koe). Claude on Bedrock sometimes writes the fix-1pfu marker with its brackets escaped; the value is now HTML-decoded (repeatedly, at most three rounds, so `&amp;lt;UNKNOWN&amp;gt;` counts) before it is matched as the marker or as a bracketed placeholder, for a string and for list items. A value with entities but no bracket shape (`AT&amp;T`) keeps its text.

- **Two workflows in one process no longer share command classes or modules** (fix-3rsq.phase10-routing-class-cache). The command-class cache is keyed by workflow folder and command, and each folder's command modules load under a namespace from its resolved path, so two same-named folders no longer share modules or relative imports. Persisted parameter-example cache keys are unchanged.

- **Parameter extraction maps `<UNKNOWN>` to "not given", so unstated arguments are asked for** (fix-1pfu). A `<UNKNOWN>` (or bracketed unknown variant) from the model now becomes "not given" (`NOT_FOUND`, or `[]` for a list), instead of running as the value `UNKNOWN`. The JSON schema is kept for its type checks. Measured on Bedrock: retail 96.6% exact cases vs 71-76% before; airline 96-97%.

- **Held-out planning requests no longer repeat training rows** (fix-3rsq.10.17). The test-set
  builder for planning-command matching dropped held-out requests that matched rows the trained
  version was built from. Its measurements were optimistic before this fix.

- **Live distillation's student pass no longer starts from the teacher's end
  state.** The workflow snapshot shared the live context dict, so restoring it
  restored nothing. The student's commands are also kept off the command-trace
  queue, so a host reading it sees one trajectory, not two interleaved.

- **A clarification reply fills the field it was asked for** (fix-mbkw).
  When parameter extraction asks for a missing or invalid field, the reply is
  merged into the asked fields (stored as `NOT_FOUND`, or a required field
  still unset) before any unset Optional field: "Ann" for a missing `name` no
  longer lands in an Optional `filter` declared before it while `name` is
  asked again. A comma-separated reply with one part per asked field fills
  those fields. With no Optional field unset, the merge is unchanged.
- **An unmentioned `Optional[str]` reaches the command as its default**
  (fix-3t6p, the str analogue of fix-7t0q). On the LLM path an `Optional[str]`
  field the request does not give came back as the text `NOT_FOUND` (or
  `None`) and reached the command as that value, while the key=value and XML
  paths gave the default. `validate_parameters` now gives such a field its
  declared default (None when it declares none;
  `signatures.default_unfilled_optional_strs`). Required fields are unchanged:
  `NOT_FOUND` there is still reported missing and asked for.
- **A misspelled phase parameter with a default fails the load** (fix-ajir.11).
  A `steps_fn`, `when` or `until` parameter defaulting to `None` must name an
  earlier phase with an output in at least one plan the placeholder inputs
  give; `lambda find, recrod=None: ...` no longer loads and silently takes the
  "not found" branch. A parameter with another default (`kind=kind`) keeps its
  value and is not checked. `check_bindings` returns what it saw of them.

- **An answer-only reply that cannot be parsed twice no longer fails the turn**
  (fix-ajir.8): `answer_only_answer` writes the answer from the steps
  (`answer_from_steps`), as it does for an empty reply; the list check and the
  code-written layout (`bulleted_answer`) apply to it.

- **One malformed skill file no longer fails every turn** (fix-ajir.3). A skill
  file whose declaration is malformed (or whose name is not an identifier) is
  logged as an error and skipped, as one that does not import already was; the
  workflow's other skills load, and the result is kept for the process instead
  of every skill being imported again on each turn. `fastworkflow command preview`
  (with or without `--check`) fails naming each skipped file and its error.

- **Command modules that import other command modules load safely from
  several threads** (fix-nhhq, `python_utils.get_module`). Loading a command
  file by path always executed a fresh copy and swapped it into `sys.modules`
  without Python's import lock, so a concurrent `from ..Ctx.mod import X` in
  another command module could get the half-executed copy ("cannot import name
  X"); and a load that failed left its broken module in `sys.modules`, so every
  later import of it failed the same way for the rest of the process. The
  loader now takes the module's import lock, reuses a module already imported
  from the same file (re-executing it in place only if the file changed on
  disk), marks a fresh module initialising while it runs, and removes it from
  `sys.modules` if it fails.

- **The size-cap prune no longer deletes a running turn's offload evidence**
  (fix-q2fy, `ObservabilityStore.prune`). A turn's record is written when it
  ends, so the cap now evicts offload evidence only of turns with a finished
  turn record; evidence of a turn that may still be running (or suspended)
  goes only by the age horizon, which also clears a crashed turn's orphans.
  The bytes it holds while the DB stays over the cap are reported as
  `in_flight_bytes` in the `prune_over_cap_bound_bytes` diagnostic. No schema
  change.

- **A question back or a hesitation no longer reads as stop_asking** (fix-jd9o.12, `workflow_agent.classify_user_reply`, `ClassifyReply`). `stop_asking` now needs Jev's reading and an explicit request to stop being asked ("stop asking", "just do it", "don't ask me again", "never mind"), so `ask_user` stays available for the turn after a question or a hesitation.
- **A near tie between two values is no longer auto-applied** (fix-jd9o.10, `DatabaseValidator.fuzzy_match`, `near_tie_margin`). A unique match is applied only when no other candidate scores within 0.08 of it (edits / length); otherwise the result is a suggestion list with every candidate within the margin, so the hook asks the user to choose. A clear winner still applies.
- **`--startup_action` is a JSON file path, or an inline JSON object** (fix-3rsq epic note, F). The help and the server comment said "JSON string"; `fastworkflow.load_startup_action` reads a path, or an object when the value starts with `{`, for `run` and `run_fastapi_mcp`.
- **Planning number reading matches the execution side** (fix-3rsq epic note, numbers). `"1,000"` was refused for 1000 and `"-5"` was read as 5; the planning check now uses `argument_checks.written_numbers`.
- **The turn-frame path refuses one value given to two required str inputs of a planning command** (fix-3rsq.check-roles-one-step). `validate_parameters` runs the role check the typed call already runs, with the same rule: every required `str` input.
- **Dependencies** (Dependabot): black 26.5.1 → 26.10.1 and setuptools 83.0.0 → 84.2.1 in `poetry.lock`; the resolver also moved urllib3 2.8.0, httpcore2/httpx2 2.13.1, litellm 1.95.1 and typesafe-sdk 0.7.4.

## 3.6.0 — DSPy 3.4, request splitting, workflow facts on classes

### Added

- **Request splitting** (`b647f25`): the agent splits a request into
  independent sub-tasks (workflow skills from `_skills/` as job types,
  optional `_skills/split_instructions.md`) and runs them one by one; the
  25-step cap hands a sub-task to a fresh executor (at most 2 hand-offs).
- **Execution guards**: multi-command calls are refused, a command that
  returns the same result again is blocked for a step, and unparsable
  replies and LLM timeouts are recovered as bad steps.
- **`LLM_TIMEOUT_SECONDS`** bounds each LLM call (default 120); parameter
  validation hooks may declare `command_text` to receive the command as written.

### Changed

- **dspy is required at `>=3.4.0,<3.5`** with the `typesafe` extra. Every
  module imports litellm before dspy, and `import fastworkflow` repairs
  DSPy's lazy `openai` stand-in when dspy was imported first, so a bare
  `import fastworkflow` no longer loads litellm. Provider cache misses are
  recorded as misses again (DSPy 3.4 flags only hits).
- **`workflow_runtime.json` is retired** (`c4c901b`): a context's callback
  class declares `occupiable = False`, a command's ResponseGenerator declares
  `descends_to` / `descend_parameter`; workflow identity is the
  framework-computed content hash.
- **Observation offloading is simplified** (`4257d2a`): observations are
  saved once per step and offloaded when the whole executor prompt exceeds
  `FW_TRAJECTORY_MAX_BYTES`. **Observability schema v9**: offload rows lose
  `scope_id` / `offload_order`, `offload_subjects` is dropped.
- **The initial planner gets every prior turn summary** instead of the last
  trajectory; the planner and agent tools get prescriptive context navigation.
- `POST /post_feedback` and `GET /feedback` accept a retired execution key
  ([R9]); `/post_feedback` returns the turn's comments.
- **Observation handles use the ReAct step index** (`O0`, `O1`, …): execute ordinals,
  `ordinal_offset`, forced-replan continuation segments, and `alias_conflict` are
  removed; max-iters exhaustion follows the base ReAct loop.

### Security

- Floors: `transformers>=5.17.0` (CVE-2026-75104, CVE-2026-80047; pulls
  tokenizers 0.23.3), `anyio>=4.14.2` (CVE-2026-63374, CVE-2026-64847),
  `fsspec>=2026.6.0` (CVE-2026-104851), `multidict>=6.9.1` (CVE-2026-104874).
- OpenVEX v7 (`security/vex/fastworkflow.openvex.json`) marks not_affected:
  litellm CVE-2026-12772/12795/12796 (proxy-only, fixed after 1.82.2), the
  transformers ZDI checkpoint batch CVE-2025-14921/14924/14926–14930, and
  torch CVE-2026-4538, CVE-2025-3000/3121/3136/4287/55554.

Releases before 3.4.0 were announced in their merge-commit subjects
(`feat: v3.2.0 — observability store, chatbot debug UI, …`) and are recoverable
with `git tag` and `git log --first-parent main`. This file starts at 3.4.0; it
does not backfill them.

## 3.5.1 — agent observation and tool-selection fixes, exact prompts in the debug UI

### Added

- **The exact prompt of an over-cap LLM call** (`fix-ra63`): an `fw.llm.call`
  whose `messages` exceed the 16 KB attribute cap is split at its DSPy field
  markers and each distinct piece is stored once per turn in a new
  `prompt_slots` table (additive; no schema version bump). The span carries a
  `prompt_slots_ref`, `ObservabilityStore.prompt_as_sent` rebuilds the messages
  and checks them against the recorded digest, and the debug UI shows them
  under "LLM input as sent" (`/api/prompt/…`).
  Pieces are credential-scrubbed and removed by prune,
  forget-channel, clear-conversations and restart-attempt.

### Changed

- **Capture profiles and context handles are gone**: `FW_OBS_CAPTURE_PROFILE`
  and the manifest `capture` field are removed (a manifest that still declares
  `capture` or `handle_projector` fails validation); recorded values get the
  credential scrub only. Span contract 10 (`fw.command.execute` v4,
  `fw.agent.tool_call` v2) records `context_before` / `context_after` as the
  plain context type name instead of a handle mapping. **Observability schema v8**: `experiments` and
  `offload_evidence` lose `capture_profile` / `capture_policy_version`, and the
  `sealed_turn_comments` control table is dropped (see Removed); a v7
  store is refused on open like every older one (a one-off
  `migrate_obs_v7_to_v8.py` at the repo root converts existing stores).
- **The observation line says where a command ran and whether it moved the
  context** (#86): `Observation O3 (execute_workflow_query ran in Directory)`,
  with `; and resulted in a context change` when it did. The root context
  prints as `global`, and a command rejected before dispatch still gets its
  clause.
- **dspy is required at `>=3.3.0,<3.4`** (`fix-5sm7`, `fix-cbjx`): 3.3.0 added
  the JSON-retry hook that keeps the available commands in the agent's retry
  prompt, and 3.4.0's lazy `openai` import breaks `import dspy; import litellm`.

### Removed

- **The workspace viewer and the blinded review sidecar** (`fix-0gh0`):
  `run_chatbot --workspace-manifest`, the picker's workspace-manifest list,
  `/api/workspace/*`, `/api/select_workspace`, `/api/review/*`, `/trace`,
  `fastworkflow/review/` and `observability/workspace.py` are gone, as are the
  navigation nodes' `source` field and the feedback response's `read_only` /
  `annotated` flags. Comments always go to the workflow's live database, so
  the `sealed_turn_comments` table that only workspace mode wrote is dropped.
  Experiment sealing is unchanged: `seal_workspace_evidence` still writes
  `sealed_archives`, and `selection.store_for` still reads a sealed member's
  archive.
- **`fw.agent.step` contract v4**: adds `repaired_tool_name`.

### Fixed

- **A workflow command named as the agent's tool runs** (`fix-8q7a`): when the
  model writes `next_tool_name: open_directory` and that command is listed for
  the current context, the step runs as the `execute_workflow_query` call it
  meant instead of becoming an invalid-tool step (three of which end a turn).
- **A moved or reinstalled fastworkflow no longer breaks every turn**
  (`fix-u2zw`): a persisted `command_directory.json` whose recorded module
  paths no longer exist is rebuilt instead of failing with "Could not import
  module from path".
- **`finish` called with arguments finishes** (`fix-njhf`): arguments the model
  passes to `finish` (`analysis=...`, `command=...`) are ignored instead of
  raising "Arg X is not in the tool's args", so the finish check runs.

## 3.5.0 — observability debug UI hardening and winner-selection fixes

The adversarial review of the `run_chatbot` observability UI (`fix-hzux`) is
resolved, and experiment winner selection no longer has a dead end on deletion
or recovery.

### Added

- **Process logs in the UI** (`fix-hzux.3`): the spawned FastAPI server's and
  training's logs are reachable from the debug UI instead of "the terminal".
- **Parent watch for the FastAPI child** (`fix-hzux.4`):
  `run_fastapi_mcp --parent_pid` exits when `run_chatbot` dies uncleanly
  (SIGHUP, SIGKILL, crash), so the next launch no longer finds an orphan on its
  port.
- **Derived-turn cache** (`fastworkflow.observability.derived_cache`,
  `fix-hzux.1`): `/api/turns` reuses per-turn derivations keyed on span count
  and last row, so a warm scan recomputes only turns that gained spans.
- **`fastworkflow.observability.turn_derivations`** (`fix-cnoc`): the pure turn
  derivations (`execution_ledger`, `turn_decision_signals`,
  `is_low_confidence`, LLM-call cost helpers) live in core observability, with
  no dependency on the debug UI. `cost_rollup` lives in
  `fastworkflow.observability.comparison`. The old `run_chatbot` import paths
  re-export the same objects.

### Changed

- **`run_chatbot` split into per-domain modules** (`fix-hzux.11`): `server.py`
  and `selection_api.py` become `handler_*` modules behind one route table and
  one auth chokepoint (`fix-hzux.10`).
- **The SPA ships as ordered source parts** (`fix-hzux.12`):
  `run_chatbot/static/index.html` is replaced by `static/src/*`, concatenated
  at serve time; the wheel includes `static/src/*`.
- **The "stdlib-only" chatbot is stdlib-only** (`fix-hzux.17`): `fastworkflow`
  and `fastworkflow.experiment` import their heavy members lazily (PEP 562), so
  starting the chatbot no longer loads dspy, litellm or fastapi.
- **Navigation and the workflow picker are lighter** (`fix-hzux.6`, `.7`):
  `/api/navigation` sends summaries with an ETag, and the picker stops polling
  when it is not shown.
- **Workflow activation no longer blocks a request for seconds** or mutates
  session state other threads read without a lock (`fix-hzux.5`).
- **Initial winner election picks the group's oldest member** (`fix-kkod`), not
  whichever registration happens to trigger it.
- **A group's sole experiment can be deleted even when it is the automatic
  winner** (`fix-65ik`, part): the pointer is withdrawn with an ordinary
  `retire` decision, and the next experiment of the group is elected
  automatically. A winner whose group has other members is still refused (409).
- **Clearer selection errors**: `ExperimentSelected` and `NoCurrentSelection`
  say why and what to do.

### Fixed

- **Deleted unused experiments stay out of winner selection** (`fix-jfy5`).
- **FastAPI feedback enforces channel ownership** on reads and writes; an
  unknown turn and another channel's turn get the same 404 (`fix-bnym`).
- **FastAPI feedback resolves recorded pass selectors** and refuses spans
  outside the named pass (`fix-jxkk`).
- **Paired task feedback filters on the queried side's attempt** (`fix-ptu1`).
- **Leaked SQLite connections**: per-request observability store connections
  are closed deterministically instead of by cyclic GC (`fix-hzux.8`).
- **Request bodies are size-capped**, and a negative `Content-Length` no longer
  reads to EOF (`fix-hzux.9`).
- **`run_chatbot` reports its own failures**: access log restored,
  `logger.exception` on 500s (`fix-hzux.2`); the SPA no longer swallows errors
  in empty catches (`fix-hzux.15`).
- **Trace view and transcript stay bounded**: multi-MB span payloads render
  lazily and superseded fetches are cancelled (`fix-hzux.16`).
- **The debug rail is keyboard-operable** (`fix-hzux.14`).

### Known limits

- A permanently lost evidence store still leaves new comparison groups in its
  workflow without a winner, and a winner-less group cannot be promoted into
  (`fix-65ik`, open).

## 3.4.0 — observation offloading and search

**Observation offloading and answer-time rehydration become the framework's
behaviour for every workflow.** They are no longer modes a deployment opts into.
A fastWorkflow tool agent compacts its trajectory, keeps every observation
reachable, and answers over the evidence behind its labels rather than over the
labels themselves.

### Added

- **Observation offloading** (`fastworkflow.observation_offloading`): canonical
  `O{n}` aliases on every execute observation, an eager SQLite archive, a packed
  trajectory target with offload labels, `search_memory` over one stored
  observation, and segmented continuation with forced replans.
- **Context-instance line** (`fastworkflow.context_identity`): every execute
  observation names the context instance the command ran in, so a listing
  produced inside a context is still attributable to that instance by a reader
  that cannot use the order of the commands.
- **Answer-time rehydration** (`fastworkflow.answer_rehydration`): the extract
  call is given its own copy of the trajectory with the evidence behind offload
  labels put back, under a byte budget, so the answer is written over evidence
  rather than over pointers.
- **Finish reminder**: a `finish` action that never opened a named item of the
  request goes back to the loop once, and only while iterations remain.
  (Replaced in this release by the finish-time execution check below; the
  roster nudge that implemented it, `fastworkflow.answer_coverage`, is removed.)
- **Finish-time execution check** (`observation_offloading/finish_check.py`,
  fix-4dsr): when the agent chooses `finish`, a decision model judges, for
  every step of the turn's initial plan and every subject the request names,
  whether the turn's record shows the step carried out; unexecuted steps are
  named in one note and the agent returns to the loop -- once per turn, only
  with iterations left, and it may finish anyway. It checks execution, not
  whether the request was answered. Off unless `FW_FINISH_CHECK=jev` and
  `JEV_API_KEY` are both set (it sends the request, the plan and a redacted
  summary of every step to TypeSafe); fails open to an unchecked finish; one
  `finish_check` offload event per finish and an `fw.finish_check` span.
  Measured offline on 82 recorded downstream-workflow attempts it was not tuned on (910
  labelled step x subject pairs, pre-registered): precision 0.77, recall 0.89.
  Install with the `jev` extra.
- **Structured plan** (`fastworkflow.turn_plan`): with the finish check on, the
  planner returns its steps (commands, sub-steps, optional and user-gated
  flags) and the request's subjects as data, and the agent still reads a
  numbered list. A plan that does not parse falls back to the plain-text
  planner. Without the check the planner is unchanged, because the structured
  call is slower (median 12.1 s against 3.7 s on three todo-list requests with
  `cerebras/gpt-oss-120b`).
- **Context budgets** (`fastworkflow.context_budget`): one input — the model's
  context window — and every byte budget derived from it as a fixed fraction.
  `budget_provenance()` returns the input, its source and every budget.
- **Searches answered without the search model.** An archived observation of at
  most 256 bytes is returned verbatim with up to three better-matching handles
  of the turn (status `short_verbatim`). A request for every row of a listing is
  answered by copying the rows (`observation_offloading/listing.py`: aligned,
  markdown and tab-separated listings, served only when the parse is provably
  complete), cut at the answer bound with a line saying how many were shown
  (status `rows_served`).
- **Optional search router** (`observation_offloading/search_router.py`, the
  `jev` extra; `router` is kept as a deprecated alias of it): a decision-model call (TypeSafe Jev) that decides whether a
  search over a listing wants every row. Off unless `FW_SEARCH_ROUTER=jev` and
  `JEV_API_KEY` are set; what it sends is redacted by the capture policy; one
  attempt, a 2-second timeout, and it fails open to the search model. Every
  search event records its verdict. A workflow may add examples in
  `search_router_examples.json`.
- **`fw.search.route` span** (contract v1) for each routing call under a traced
  turn: the verdict, latency and tokens, never the question or the evidence.

### Changed

- **Structured planning is disabled** (2026-09-28, owner decision; supersedes
  the "Structured plan" entry under Added and the later structured-planner
  notes in this release). The planner always makes a plain-text plan, with the
  finish check on or off; with it on, the check's plan is `parse_text_plan`'s
  (`source="text"`, no subjects). The structured signatures, the
  `STRUCTURED_PLAN_GUIDE` prompt, `turn_plan.render`, the structured adapter,
  the zero-steps / parse-error fallback and the planner span's subject
  redaction are commented out, not deleted, and not behind a flag: nothing
  re-enables them. `fw.planner.plan` / `.replan` keep contract v2 (and
  `SPAN_CONTRACT_VERSION` 8): `plan_source` is only `text` or `none`, and
  `subjects` is always `[]`. The `TurnPlan` / `PlanStep` / `PlanSubject`
  models, the plan's persistence across a resume (fix-ju1v) and the
  adapter's `use_json_adapter_fallback` parameter (fix-6jzi) are unchanged.
  The tests of the structured path are skipped, not removed.
- **Known-name guard**: a known command name is never answered by a context
  that does not own it; the declining prediction carries a hint naming where the
  command lives. When no context on the chain owns it, that hint is the whole
  response and the next message goes through ordinary intent detection, rather
  than `you_misunderstood` and its clarification stage, which only matches the
  current context's commands and so could not route the command the hint names.
  The guard also covers a reply to `you_misunderstood`, which is matched against
  the same full command set; the ambiguity stage, which matches a short
  suggestion list, is still excluded. With several owners, the hint names each
  entering command beside the context it enters.
- **Threshold separation**: `write_ambiguity_thresholds` is the single writer
  for both ambiguity files and establishes a non-empty ambiguity band where the
  artifacts are produced, with `TIER_AMBIGUITY_MIN_SEPARATION` and
  `SINGLE_LABEL_RESOLUTION_FLOOR`. A tier threshold of exactly 1.0, which a
  float32-saturated sweep can pick, publishes the flat 0.99 cap instead of
  aborting the workflow's training run.
- **Workflow fingerprint scope rule v2**: a root `benchmarks/` tree and the
  runtime observability store leave `workflow_content_entries`.
  `WORKFLOW_SCOPE_RULE_VERSION` 1 → 2, so a v1 declaration reads as
  `incomparable`, not stale.
- **Synthetic utterance generation** sends `temperature` only; `top_p` is gone,
  because current Bedrock Claude models reject both together and every
  generation call was a hard `BadRequestError`.
- `fw.nlu.intent` span contract v2 → v3.
- `fw.command.execute` span contract v2 → v3: the four auto-navigation
  attributes are gone with the two-step dispatch that wrote them.
- The aggregate span-contract version is 5. Version 4's note named
  auto-navigation keys that `SPAN_CONTRACTS` does not contain; 5 is the number
  that matches those contracts. It became 6 when `fw.search.route` was added
  (no existing span changed), 7 with `fw.finish_check` and `fw.agent.step` v3,
  and is now 8: `fw.planner.plan` / `fw.planner.replan` v2 (fix-3x76).
- **Offload labels say the evidence comes back.** A label reads "Offloaded
  observation O… returned by … It is restored in full when the final answer is
  written, so search it with search_memory only for a value you need for your
  next step", and the agent's instructions say the same. The earlier wording,
  which opened with an instruction to search, is still recognised. In recorded
  runs this cut requests to search whole tables by 70-95%. (Since fix-2hxv a
  label ends with the short "Restored in full for the final answer."; the
  promise and the search guidance are stated once, in the agent's instructions
  and the `search_memory` description. Both earlier wordings still parse.)
- **Answers about an incomplete listing name a real narrowing input.** The
  search signature gains a `narrowing` input listing the producing command's
  optional, non-selecting inputs; an answer that needs rows not shown ends with
  `To reach them: <command> <input>=<value>`, or says a narrower re-run is
  needed when the command declares none.
- **Continuation allows 3 forced replans** (4 segments, a 100-step ceiling),
  up from 2 (3 segments, 75 steps). `FW_MAX_FORCED_REPLANS` moves it (see
  below).
- **Intent `signal_version`** no longer carries a threshold-semantics segment.
  It now reads `intent-classifier/<artifact version>/...`, so a version string
  identifies the artifact behind a signal and nothing else.
- **Offloading evidence lives in the observability store.** Archived
  observations, their subjects and the offloading runtime's diagnostic events
  are rows of `offload_evidence`, `offload_subjects` and `offload_events` in the
  workflow's `observability.sqlite3` (feature markers `offload_evidence_v1`,
  `offload_events_v1`; no schema-version bump). They are keyed by turn, and
  `forget_channel`, Clear conversations and retention pruning delete them in
  the same transactions that delete the turn record. Events are read back with
  `ObservabilityStore.offload_events(...)`.
- **Evidence is redacted when it is written.** With
  `FW_OFFLOAD_EVIDENCE_REDACTION=on` (the default) responses and event text are
  stored as the trace sink's credential scrub and capture policy leave them;
  `off` stores them verbatim for development. The process running a turn keeps
  the raw text of its redacted observations in memory until the turn is over,
  so the agent's own reads stay exact; a turn resumed in another process reads
  the redacted text.
- **Observability recording is always on**, for fastWorkflow's entry points and
  for programs that embed the library alike. The database is owner-only (0600
  file, 0700 directory) and pruned by `FW_OBS_RETENTION_DAYS` and
  `FW_OBS_DB_MAX_BYTES`. `get_observability_sink()` no longer takes
  `entry_point`, and returns `None` only when the store cannot be opened.
- **An execution context opens its own sink.** A `WorkflowExecutionContext`
  given no sink opens the bound app workflow's sink in `bind_app_workflow()`,
  and follows a rebind to another workflow's database. A sink passed to the
  constructor or to `set_trace_sink()` is never replaced, and an explicit
  `tracing.NoOpTraceSink()` records no spans or turn records;
  `set_trace_sink(None)` returns the context to its automatic sink. Offloading
  evidence is written to the workflow's database whatever the sink, and the
  offloading archive prunes that database once per process when it opens it.
- The `search_memory` input bound has no tuning override; it is derived from the
  search model's context window only. It is a quarter of that window (131,072
  bytes at the reference window), up from 3/128 (12,288 bytes). (Superseded
  in part by fix-deus below: the derived value is now capped at 131,072 bytes
  and `FW_SEARCH_OBSERVATION_MAX_BYTES` overrides it.)
- **An observability database from an older schema version is replaced, not
  refused.** The store has not shipped in a release before, so such a file can
  only be a local development database: opening it with the writer deletes it
  (and its `-wal`/`-shm`) and creates a fresh store in place, with a warning
  naming the path and both versions. A database from a newer build is still
  refused and never touched, and the read-only viewer refuses an older one
  without deleting it.
- **The Jev features share one client module** (`observation_offloading/jev_client.py`,
  fix-xeg1): the optional SDK import, `JEV_API_KEY`, the default model,
  redaction of what is sent, one-attempt clients and a per-process cache, for
  both the search router and the finish check.
- **A Jev feature that is asked for but cannot start says so** (fix-jdcm).
  `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` set to anything other than `jev`, set
  without `typesafe-sdk` installed, or set without `JEV_API_KEY`, logs one
  warning per cause per process and the feature stays off; an unset flag is
  still silent. Clients are cached by a fingerprint of the key, never the key,
  so a changed key builds a new client instead of reusing the old one. (More
  causes since fix-2so9 and fix-7tz5, and a flag may name a registered
  provider since fix-5uva; see below.)
- **The forced-replan bound is a setting** (fix-vd5c): `FW_MAX_FORCED_REPLANS`
  overrides the default of 3, clamped to `[0, 10]`; a non-integer is ignored,
  with a warning either way. `forced_replan` and `forced_replan_wall` events
  carry `reached_limit`, and `agent_installed` records the bound in force. The
  default rests on one downstream-workflow task, not on a measured tail across workflows.
- **Planner spans say which planner produced the plan** (fix-3x76):
  `fw.planner.plan` / `fw.planner.replan` v2 add `plan_source` (`structured`,
  `text`, `text_fallback`, `none`) and the structured plan's `subjects`,
  redacted by the capture policy. A `finish_check` event also carries a score
  table, one entry per checked step x subject (x part), capped at 200.
- **`finish_check` events are recorded for every finish an attached check
  sees** (fix-7pba), with a reason even when nothing was asked: `disabled`
  (`FW_EVAL_FINISH_REMINDERS=0`), `cap reached`, `nothing to check` (every step
  optional or user-gated), `no plan`, `no room to act`, `error`. The step count
  the note states includes the room later forced-replan segments still hold
  (`shown_iterations_left`); the two-iteration gate is still the current
  segment's.
- **Stored `finish_check` events name steps and subjects by index**
  (fix-7scj): `flagged_steps` holds the step index, the subject's index into the
  plan's subjects, its kind and `p_unmet`, never plan-step text or subject
  names.
- **Search events say which path a search took** (fix-wheb): `router_enabled`,
  `listing_parsed`, `listing_shape`, `listing_skip_reason` and `for_report` on
  every search event, so a `router` of `None` is never ambiguous;
  `related_scores` and `subject_recorded` on `short_verbatim`;
  `served_over_bound` and `trailing_lines_dropped` on `rows_served`. The
  search-observation budget moves into `context_budget` as
  `SEARCH_OBSERVATION`, and `budget_provenance()` reports it as
  `search_observation_max_bytes` beside `search_window_tokens` and
  `search_window_source`, the window it is cut from.
- **The `search_memory` description matches the path the agent has**
  (fix-8uj0): it is built with the agent, states the read bound as the byte
  count the search will apply, and, when a router is attached, says an all-rows
  search may be answered by copying rows instead of forbidding such searches.
  The router's `for_report` verdict is recorded and does not change the path.
- **Offload labels and served listings are shorter** (fix-2hxv): both end with
  "Restored in full for the final answer." instead of the full restore promise,
  which cost ~100 bytes per label and made fewer observations worth offloading.
  (Since fix-94m9 the mark is "Normally restored for the final answer."; see
  below.)
- **Narrowing inputs include defaulted ones** (fix-swqi): an input with a
  non-None default (`limit: int = 50`) is offered as a way to narrow a listing,
  not only one typed `Optional`. A required field's sentinel default
  (`NOT_FOUND`, `INVALID_INT_VALUE`, `INVALID_FLOAT_VALUE`) does not count.
- **Fewer archive reads** (fix-314g): `list_summaries` measures stored rows
  with `length()` on the BLOB, without reading their content, and an alias the
  archive has no subject for is remembered (bounded) instead of re-read on every
  related-handle lookup.
- The settings above, `FW_SEARCH_ROUTER_MODEL`, `FW_FINISH_CHECK_MODEL` and
  `JEV_API_KEY` are in the example env templates, commented out, with a
  data-egress warning (fix-9l9e); stale docs, comments and this changelog's
  span-contract number are corrected (fix-42ig).
- **Vendor calls have hard wall-clock bounds** (fix-i94q): the router's 2 s,
  the finish check's 4 s per call and 8 s per check are enforced during the
  call, not checked before it, and the check's 8 s now include building the
  ledger. Each call runs on its own daemon thread, at most 4 process-wide; a
  call with every worker busy fails at once as `VendorBusy`, one cut off at its
  cap is `CallTimedOut`, one cut off by a deadline or budget is `OutOfTime`. An
  abandoned request is never retried and never delays interpreter exit, but
  may still be billed vendor-side. The SDK's own timeout is the cap plus 0.5 s.
- **One vendor budget per turn** (fix-sotm): 10 s of time actually spent on
  vendor calls, router and finish check together, and at most 3 routing calls.
  Past either, `route` returns `router_budget` with no call and no span, the
  search model answers, and the search records `listing_skip_reason`
  `router_budget`. `vendor_ms` (the turn's running total) is on every router
  record and on a checked `finish_check` event.
- **The Jev endpoint is `FW_JEV_BASE_URL`** (fix-2so9): default
  `https://api.typesafe.ai`; https only, or plain http to a loopback address;
  no credentials, query or fragment. A rejected value warns once, without the
  URL, and turns both features off; a non-default host is logged once. The
  SDK's `TYPESAFE_BASE_URL` is ignored with one warning, so it can no longer
  redirect the key and payloads.
- **`typesafe-sdk` is pinned exactly to 0.7.2** (fix-em0l), with an SDK
  contract test on a loopback stand-in as the review gate for a bump.
- **Jev failures are diagnosable** (fix-conv): an `error` `finish_check` event
  and a router error record carry `error_status`, `error_request_id`,
  `error_code` (one machine-readable token from the body, never its text) and
  `error_stage` (`ledger`, `request`, `budget`, `check`, `note` on the check;
  `redaction`, `request`, `budget` on the router). A failure is warned about at
  most once per 5 minutes per (feature, error type, HTTP status), counting the
  ones not logged, instead of once per process; a ledger-build failure is
  logged with its traceback. No span-contract change.
- **The Jev features stay off when the capture policy would withhold what they
  send** (fix-7tz5): under a capture profile that withholds command output
  (e.g. `evidence`), an unknown profile, or
  `FW_OFFLOAD_EVIDENCE_REDACTION=off`, one warning per cause. A declared field
  policy that keeps observations whole is still allowed. A value the policy
  withholds at call time sends nothing: the finish check records the new
  reason `policy_withheld` (unwarned, `requests` 0) and the router returns the
  error `policy_withheld` (`error_stage` `redaction`).
- **One decision says whether the finish check is on for a plan** (fix-rm98):
  `workflow_agent.finish_check_active(agent)`, read by the planner. With
  `FW_EVAL_FINISH_REMINDERS=0` nothing the check adds runs -- no note and no
  structured plan -- while the checker stays attached and records `disabled`.
  The switch is read from the workflow's `fastworkflow.env` first, then the
  process; any other set value still fails the agent build, and an empty value
  now counts as unset.
- **The structured planner runs only for a turn's initial plan** (fix-5vtw);
  replans after a parameter-extraction error or an `ask_user` reply use the
  plain-text planner, and the check keeps checking the initial plan. **Its
  call no longer takes DSPy's hidden JSON-adapter retry** (fix-7eu6), which
  dropped the available-commands prelude: an unparseable reply goes straight
  to the text planner, at most 3 calls instead of 6.
  `CommandsSystemPreludeAdapter` gains `use_json_adapter_fallback` (default
  `True`, unchanged for the agent loop and intent clarification). (Since
  fix-6jzi that retry, where it stays on, keeps the command list; see Fixed.)
- **Large finish checks are bounded** (fix-az9q): a plan naming more than 12
  subjects is checked step by step only (reason `subjects capped`, no subject
  name sent); the ledger is kept under 96 KiB by dropping rows' `head`, then
  `refers_to`, oldest first; an over-long ledger is halved at most once, and a
  half still too large is an `error`. Checked events gain `subjects_capped`,
  `ledger_bytes` and `ledger_rows_trimmed`. No recorded turn is trimmed (max
  52 KB).
- **The plan survives a cross-process `ask_user` resume** (fix-ju1v): with a
  checker attached, `turn_plan` and `turn_plan_status` are session state and
  the resumed agent's step record is seeded from the suspension. A "no plan"
  event carries `no_plan_cause` (`planner_empty`, `plan_unreadable`,
  `lost_on_resume`, `not_planned`); a resume from a suspension the
  context-window fallback had cut is skipped as `ledger incomplete`.
- **The finish-check note says to skip what the user declined, and not to
  repeat or bypass** (fix-hfbr, partial; fix-03lt). Its tail now reads "Run any
  that are still needed, or say in your answer why not. Do not repeat a change
  this turn's record shows was already made, and do not make a change the user
  has not confirmed: ask them instead. Skip any step the user has since
  declined or changed. You have N steps left." Every `finish_check` event
  carries `user_replies`, the turn's `ask_user` step count. The checked plan is
  still the initial one. fix-hfbr stays open on a framework-owned approval
  gate (fix-47a9). (Since 2026-09-28 that gate is deferred and fix-hfbr no
  longer waits on it; see the next entry.)
- **The finish check only asks about provably read-only steps** (fix-hfbr).
  A step is checked only if every command it and its parts name is declared
  `read_only` in the workflow's runtime manifest (the metadata registered at
  startup, else `workflow_runtime.json`; a qualified name counts by its last
  `/` segment, at the most severe kind any key of that name declares). A step
  naming a `write` command or one the manifest does not declare is never
  asked about, its text is not sent, and it is never named in the note, so a
  false flag cannot drive a repeated or unconfirmed change. A step naming no
  command is still checked. Undeclared is treated as not read-only here (the
  safe direction for a note that asks the agent to act); the owner's
  "treat undeclared as read-only" decision was for fix-47a9's approval gate.
  **Behaviour change:** a workflow without a manifest, or with one that
  cannot be read, now has only its command-less steps checked. Every
  `finish_check` event recorded after the plan is read carries
  `unchecked_for_effect`, the steps skipped this way; when that is every
  step the check would hold the agent to, the reason is the new
  `no read-only steps` and nothing is sent. Replayed on the 82 recorded
  downstream attempts with that workflow's manifest (cached answers): write-step flags 7 to 0,
  precision 0.80, recall 0.88 to 0.87. No span contract changes; with
  `FW_FINISH_CHECK` unset nothing changes.
- **The listing parser is fail-closed** (fix-zoup, fix-36gv, fix-k56c; see
  Fixed): it returns a listing only when it is sure the rows are the whole
  listing. Its recognition is widened conservatively (fix-1593): Unicode
  column names, the compact `|-|-|` markdown separator and pipe-less
  `a | b` / `--- | ---` tables. On 4,000 random listings it refuses 702 the old
  parser accepted and accepts none the old parser refused; with the router
  off only `listing_parsed` telemetry changes.
- **Served rows fit the answer bound exactly, or go to the model** (fix-3zxk):
  see Fixed. The closing reads "rows 1-N of the M rows listed in O5".
- **Short-observation hints compare against the observation itself**
  (fix-lzdz): the header names the observation's own subject, and another
  handle is offered only when it mentions the question's words strictly more;
  all hint wordings changed. The event gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`.
  Relatedness reads Unicode letters and digits (fix-1593).
- **Broad scopes are not enumerated** (fix-y570): under the process-default
  scope and the between-turns fallback, `list_summaries` returns nothing and the
  hint says other observations are not listed in this scope; it also filters
  by `channel_id` now. The archive's `get()`/`list()` cross-channel read under
  those scopes is fix-tyzj, still open. (Fixed since by fix-tyzj; see Fixed.)
- **Backend lines shaped like framework markers are quoted** (fix-znxq,
  verbatim-quoting part): in a short observation and in served rows (preamble,
  column line, rows), a line containing `[search_memory`, `Observation O<n> (`
  or an offload-label prefix is printed after a visible `> `. The finish-check
  part of fix-znxq is folded into fix-hfbr. (Since 2026-09-28 an adversarial
  test covers it: with a Jev stand-in answering "not executed, and applies" to
  every question and injected tool output claiming a write step did not run,
  the note names only the read-only and command-less steps, never the write
  steps, their text or the injected text;
  `test_a_ledger_that_claims_writes_went_unexecuted_never_gets_a_write_step_named`.)
- **The search evidence bound has a ceiling and an override** (fix-deus): the
  derived quarter of the search window is capped at 131,072 bytes
  (`SEARCH_OBSERVATION_CEILING_BYTES`), `FW_SEARCH_OBSERVATION_MAX_BYTES`
  overrides it (and may exceed the cap), and the search window is the smaller
  of `FW_MODEL_CONTEXT_TOKENS` and the search model's metadata instead of the
  setting outright. `budget_provenance()["overrides"]` reports the override.
- **The finish check's identifier resolution is generic, and its calibration
  is stated** (fix-ft18): `refers_to` also resolves from aligned, markdown and
  tabbed listings (first 64 KiB of each response, block by block), parameter
  values and clause tokens; recorded downstream ledgers are unchanged. Every event
  records `calibration` (the calibration tag), and a non-default
  `FW_FINISH_CHECK_MODEL` warns once per process. The structured planner's
  `kind` description no longer uses workflow-specific examples.
- **Decision-model features go through a provider interface** (fix-5uva,
  stages 1-2): `fastworkflow.observation_offloading.decision` (`YesNo`,
  `OneOf`, `DecisionProvider`, `ProviderFailure`, `StateTooLarge`,
  `OutOfTime`, `register_decision_provider`, `unregister_decision_provider`),
  with Jev as `jev_client.JevProvider`. `FW_FINISH_CHECK` and
  `FW_SEARCH_ROUTER` accept a provider name registered from code, never a
  module path; selecting one warns once that the thresholds were calibrated
  with Jev. `finish_check` events and router records gain `provider`. What Jev
  receives is the same request bodies as before, key order included, pinned by
  a golden test.
  A LiteLLM provider (stage 3) is deferred.
- **Integration tests reach real HTTP** (fix-dln2, partial): a threaded
  loopback stand-in for TypeSafe (`tests/jev_stub.py`, fixture `jev_stub`)
  drives the real SDK; the structured planner is tested with DSPy's `DummyLM`
  and the plan's cross-process resume through real `process_turn` turns.
- **The restore promise says "normally"** (fix-94m9). Default path,
  agent-visible, and **not re-measured**: the offload-label and served-rows
  mark (`labels.LABEL_RESTORE_MARK`) is now "Normally restored for the final
  answer." (was "Restored in full for the final answer."; +1 byte per label);
  the `search_memory` tool description and the `WorkflowAgentSignature`
  docstring (agent and extractor instructions) say every observation "is
  normally restored in full" and add "If the answer's evidence limit is
  reached, the oldest observations are not restored and the answer names
  them."; and the rehydration note's prefix (`NOT_REHYDRATED_PREFIX`) is now
  "Not rehydrated for the answer (evidence exists under these observations,
  but the answer's evidence limit was reached; say in the final answer that
  the rows of these observations are not included in it): ", so the extractor
  states the omission. The note appears only when the evidence budget binds.
  Labels in every earlier wording still parse.
- **`search_memory`'s narrowing inputs come from the command dispatched**
  (fix-v9jy): `_execute_workflow_query` files the qualified command each
  execute alias ran (`remember_dispatched_command`, on
  `agent.dispatched_commands`, current turn's scope only; CME commands such as
  `abort` and `go_up` are not filed), and `describe_command_inputs` uses it
  when its bare name matches, so two contexts' same-named commands are told
  apart. Otherwise, and after a cross-process resume, it falls back to the
  previous rule. No prompt, schema or event field changes.
- **`search.py` is split** (fix-vpe3): the relatedness score, related-handle
  suggestions and short-observation answer move to
  `observation_offloading/related.py`; `search.py` re-exports every name. The
  search event is built once from a shared base, so its keys come out in
  another order with the same values (stored with `sort_keys`). No behaviour
  change.
- **The decision-model wording is pinned to its measurement** (fix-8ee5):
  `tests/fixtures/decision_measurements.json` records the offline measurement
  behind the finish check's and router's published figures (calibration
  tag, model, thresholds; finish check precision 0.767
  [0.722, 0.810], recall 0.888 [0.832, 0.936] on 910 pairs over 82 attempts;
  router all-rows precision 1.0 [0.970, 1.0] on 126, recall 0.962
  [0.914, 0.984] on 131, at threshold 0.5; each with n and 95% interval) and
  the sha256 of every evidence file. `tests/test_decision_measurements.py`
  re-renders every finish-check and router question template with fixed
  arguments and fails when a digest changes, even if the golden request bodies
  were regenerated to match. The evidence itself is not in the repo; see
  Migration.

### Fixed

- A malformed model reply that arrives after a tool has already run now fails
  the turn instead of silently re-running the whole trajectory. The re-run could
  repeat the side effects of the commands already executed and leave the
  archived evidence out of step with the answer.
- A known command name followed by a newline or a tab is now recognised by the
  known-name guard and by the owning context's exact match, not only when the
  name is followed by a space.
- A command whose name has capital letters is matched by its own context's
  exact match whatever case it is typed in, and is never refused by the
  known-name guard as belonging to another context. The guard compared a
  lowercased name against the context's command names as spelled, so it named
  the current context as the foreign owner.
- A failure while releasing the previous turn's process-local evidence no
  longer aborts the turn that is starting.
- An observability database that no trace sink ever opens -- a context built
  with `tracing.NoOpTraceSink()` -- is now pruned too: the offloading archive
  prunes each database once per process when it first opens it. Its offload
  tables used to grow without bound.
- The finish-time roster nudge no longer fires in a workflow whose contexts
  declare no instance identity, where every named item looked untouched, and
  it no longer tells the agent that nothing about those items was retrieved;
  it says only that they were never the subject of a command. (The roster
  nudge was then removed in favour of the finish-time execution check.)
- The foreign-context hint no longer ends with "Then run it there" when no
  entry command is declared; it says to move into the owning context first.
- Replanning and answer-time rehydration apply the agent's own execute
  numbering to a handle line or offload label, as compaction already did, so a
  command response shaped like one on a step that was never annotated is kept
  as response and never resolved to another observation's evidence. Each such
  line is recorded as a `foreign_line_ignored` event.
- The replan skeleton's policy label is `greedy_trajectory_budget`; it named
  a fixed 28 KB bound it does not use outside the reference window.
- A reply to `you_misunderstood` that matches none of the current context's
  commands now lists what can be done there. It used to raise
  `KeyError: 'what can i do?'`, because the fallback it substitutes was
  registered only for the ambiguity clarification stage.
- The finish check asks about the whole step when a step concerns a subject
  but none of its sub-steps does; such a step was never asked about, so it
  could not be flagged (fix-ba7k).
- Text plans parse a plan written on one numbered line ("1. a. 2. b. 3. c")
  into its steps, and read "optional" and "needs the user" from the whole step
  rather than its first 40 characters, without taking "with optional filters"
  for an optional step (fix-nzmy).
- A structured planner call that returns no steps falls back to the plain-text
  planner instead of leaving the turn unchecked, and a text plan that cannot be
  parsed leaves the check without a plan instead of failing the turn. With the
  check off no text plan is parsed at all (fix-j8yn).
- The async agent loop runs the finish check off the event loop, so its HTTP
  calls no longer block other coroutines (fix-13lt).
- `FinishChecker.note()` never raises and always closes its `fw.finish_check`
  span, also when the check fails after the span opens (fix-re11).
- A step output whose first sentence opens by saying it is empty ("No accounts
  found for Alan.", "Identity not found") still counts as empty, but one that
  goes on to a counted clause or picks from a counted set is a result: "0
  errors, 12 updated" and "None of the 5 are overdue". Subject
  names are found in an output as whole words, so "Al" is not found in "Alan"
  (fix-tu2g).
- Finish-check tests assert structure and key facts; one golden test pins the
  measured note wording (fix-mbcb).
- An over-long finish-check ledger is recognised from the error body's
  `max_tokens_exceeded` code, not from `str(error)`, so a refusal worded for
  humans still halves the ledger instead of failing the check (fix-oivq).
- The TypeSafe SDK's DEBUG logging no longer writes whole, unredacted request
  and response bodies: a filter on the `typesafe_sdk` logger drops those
  records whatever the log level and keeps its INFO status line (fix-qgeu).
- Finish-check subject names and kinds, `refers_to` keys and
  `names_in_output` now pass the same outbound filter as every other value
  sent; a credential used as a subject name or listing id never reaches the
  wire. The note shown to the agent keeps the raw names (fix-hu4f).
- A step the NLU stage stopped before any command ran (parameter extraction,
  an ambiguous or misunderstood command) is recorded at dispatch and shows in
  the ledger as outcome `error`, instead of reading as executed (fix-lnzw).
- A short-observation search no longer blocks for 30 s and raises when the
  related-handle lookup cannot read the archive: `list_summaries` waits at most
  0.5 s, any failure is caught, and the hint says other observations could not
  be listed (fix-kvq0).
- A refused listing parse is final: `parse_table` no longer retries from every
  later line, so a 104 KB listing refused at its end takes ~2 ms instead of
  2.5-4.6 s, and a short observation is not parsed at all (fix-zoup,
  fix-n4z9).
- An aligned listing no longer ends on an ordinary row -- a wrapped label, a
  row with separators collapsed to one space, or empty trailing cells -- while
  claiming to be complete (fix-36gv).
- The first block of a listing is no longer presented as the whole
  observation: a second group or table after it, a `total`/`sum`/`count` last
  row, or text saying there is more (`remaining=`, `has_more=true`, `N of M`,
  `page=2`, `next cursor`, `truncated`, …) refuses the parse (fix-k56c).
- Served rows never exceed the answer bound: `served_rows` picks the largest
  row count whose closing line fits and returns `None` when none fits, and the
  search goes to the model with `listing_skip_reason` `rows_do_not_fit`. A
  large preamble used to give a 5,410 B answer against a 3,000 B bound
  (fix-3zxk).
- **Cold-resume archive location.** An agent built while a context restores a
  suspended turn has no active workflow yet, so it opened the offload archive
  from an empty workflow path: a database named after the working directory's
  basename (unless `FASTWORKFLOW_WORKFLOW_ID` was set). This hit the default
  path. Evidence archived before the suspension was unreachable by
  `search_memory` and answer rehydration, and everything archived after the
  resume went to the stray database for the context's life, where channel
  erasure never reached it. `build_tool_agent` now reads the session's bound
  app workflow first, then the active one; the router's examples and the
  finish check's effect lookup use the same path. Warm turns open the same
  database as before.
- The `ask_user` replan after a cross-process resume plans from the request
  and trajectory again (fix-ksdu): it ran before `resume()` restored them and
  got `{}` for both. `ContinuationReAct.planner_view()` falls back to a copy of
  the suspended stash's `input_args` and `trajectory` (offload labels, no
  `action_N` keys); a same-process replan prompt is byte-identical. Session
  state is unchanged.
- DSPy's JSON-adapter retry after an unparseable chat-format reply keeps the
  available-commands list (fix-6jzi), in the agent loop, intent clarification
  and the plain-text planner; call counts are unchanged. It relies on DSPy's
  **private** `ChatAdapter._make_json_adapter_fallback()` hook (checked on
  DSPy 3.3.0; the pin still allows `^3.0.1`), overridden to return the new
  `CommandsSystemPreludeJSONAdapter`; a guard in
  `tests/test_chat_adapter_commands.py` fails if DSPy renames it. On an
  installed DSPy without the hook, importing `fastworkflow.utils.chat_adapter`
  logs one warning that the JSON retry will drop the available commands list;
  behaviour is otherwise unchanged.
- The offload archive no longer reads across channels (fix-tyzj): `get`,
  `list`, `get_subject`, `capture_record`, `forget_subject` and the
  stored-digest check filter on `channel_id` as well as the turn key, and
  `put_subject` updates only a row of the same channel. A scope pairing another
  channel with this turn's key reads nothing, cannot replace or forget this
  channel's subjects, and still collides on persist. The collision's
  `PersistenceError` now reads "runtime handle alias is already stored for this
  turn (different text or another channel)"; it said "collides with different
  text", which was wrong for identical text from another channel.
- A labelled last row -- first cell ending with `:`, or one word then `:` and a
  space (`Note:  2 items`) -- is a footer, not a row (fix-4riz): the strict
  parse refuses the listing, as for a `total`/`sum`/`count` row, and the lax
  read drops it. Over 3,832 recorded downstream-workflow texts, no accept/refuse decision
  changed in either mode.
- Test isolation (fix-4fw3): the SDK timeout contract asserts the private
  `_config.timeout` the pinned 0.7.2 SDK actually has, and tests that register a
  litellm model restore litellm's model tables and caches afterwards
  (`restore_litellm_model_cost` fixture).

### Migration

- **Structured planning is disabled** (2026-09-28). Its code is commented
  out, not deleted, and there is no setting or flag to turn it back on. With
  `FW_FINISH_CHECK` on, the finish check now gets text plans only: those have
  no subjects, so the per-subject questions are never asked of a live plan,
  only per-step ones. Planning with the check on is no slower than without it
  (a single plain-text planner call). A reader of `fw.planner.plan` spans sees
  `plan_source` `text` / `none` and `subjects` `[]`; records written before
  this change may still hold `structured` or `text_fallback`.
- `LLM_OBSERVATION_SEARCH` is the recommended setting for the model that answers
  `search_memory`. When it is unset, search runs on `LLM_AGENT`, which also
  fixes the budget the evidence is cut to, since that budget is sized from the
  search model's own context window.
- `LITELLM_API_KEY_OBSERVATION_SEARCH` is the recommended credential for that
  role. When it is unset, search uses the credential configured for `LLM_AGENT`.
- Offloading evidence now lives in `observability.sqlite3`. An evidence file
  left by an earlier build beside it (`observability.sqlite3.offload-handles.sqlite3`,
  its write-ahead-log files and any `.preserve` marker) is deleted the first time
  the store opens; its contents are not migrated.
- Experiment evidence is no longer preserved: Clear conversations, and
  forgetting an experiment run's channel, erase its offloading evidence like any
  other turn's.
- Recording can no longer be turned off. A program that embeds the library gets
  the same owner-only, pruned record as the entry points; to keep it elsewhere,
  set `FASTWORKFLOW_STATE_ROOT`. A `WorkflowExecutionContext` built without a
  sink now records into its workflow's database on its own; pass
  `trace_sink=tracing.NoOpTraceSink()` where a context must record no spans or
  turn records. That does not stop observation offloading, which still writes
  its evidence, subjects and events to the same database.
- The offloading runtime's events are not written to a separate file; read them
  with `ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.
- To correct the `search_memory` input bound, set `FW_MODEL_CONTEXT_TOKENS` or
  point `LLM_OBSERVATION_SEARCH` at the intended model. (Since fix-deus, set
  `FW_SEARCH_OBSERVATION_MAX_BYTES` to move the bound itself; a
  `FW_MODEL_CONTEXT_TOKENS` larger than the search model's own window no longer
  raises it.)
- The roster nudge is gone: without `FW_FINISH_CHECK=jev` and `JEV_API_KEY` a
  `finish` is never interrupted. `FW_EVAL_FINISH_REMINDERS=0` still turns the
  note off when the check is on (and, since fix-rm98, the structured planner
  too). `fastworkflow.answer_coverage` no longer
  exists, and a step span marks a note with `finish_check_note` where it used to
  say `roster_nudge` (`fw.agent.step` v3; span contract 7).
- The aggregate span contract is 8 (was 7): `fw.planner.plan` and
  `fw.planner.replan` are v2 and gain `plan_source` and `subjects`; with the
  finish check on, `plan` is the structured plan rendered as a numbered list
  with "(optional)" / "(needs the user)" flags. Spans recorded before read with
  both keys absent. A reader that pins the aggregate number must accept 8.
- `finish_check` events recorded before this change carry step text and
  subject names in `flagged_steps`; new ones carry indices. Events are now also
  recorded for skipped finishes, so count checks by `reason`, not by event.
- Search events gain the path fields above; `listing_shape` is no longer only
  on `rows_served` events.
- Install the Jev features with `fastworkflow[jev]`. `fastworkflow[router]`
  still works as a deprecated alias.
- `JEV_API_KEY` keeps its name; it does not follow `LITELLM_API_KEY_<ROLE>`
  because it is the TypeSafe SDK's own key, not a litellm role's.
- Turning on `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` sends turn content to
  TypeSafe with only credential patterns scrubbed by default (since fix-7tz5,
  neither turns on at all under a withholding capture profile or with
  redaction off), and the tokens
  those calls use are not in `fw.llm.call` cost totals. The finish check's
  8-second budget is best-effort: it is checked before each call, so one call
  can run up to its 4-second timeout past it. (No longer true since fix-i94q:
  every bound is a hard wall-clock cutoff; see the migration notes below.)
- **Session state gains two additive keys**, `turn_plan` (the plan as JSON, or
  `null`) and `turn_plan_status` (`planned`, `planner_empty`,
  `plan_unreadable`, `lost_on_resume`, `not_planned`); there is no
  `SCHEMA_VERSION` bump. A state written before this change resumes with the
  plan lost (`no_plan_cause` `lost_on_resume`) rather than failing. With the
  finish check off every blob carries `turn_plan: null` and
  `turn_plan_status: "not_planned"`. A react suspension may carry an optional
  `dispatch_outcomes` key.
- **`typesafe-sdk` is pinned to exactly 0.7.2.** `pip install
  'fastworkflow[jev]'` no longer resolves a later 0.7.x; an environment that
  already has another version installed is downgraded or conflicts. A bump
  must rerun `tests/test_jev_client.py::test_the_sdk_surface_fastworkflow_uses`
  and the golden-body tests in `tests/test_decision_provider.py`.
- **`TYPESAFE_BASE_URL` is ignored.** A deployment that pointed the SDK at
  another endpoint with it must set `FW_JEV_BASE_URL` instead (https, or http
  only to loopback); until then the features call the default endpoint and log
  one warning naming the variable.
- **Event shapes gain fields; none is removed or renamed.** `finish_check`
  events: `provider`, `calibration` and `user_replies` on every event;
  `no_plan_cause` on "no plan"; `vendor_ms`, `subjects_capped`, `ledger_bytes`
  and `ledger_rows_trimmed` on a checked finish; `error_status`,
  `error_request_id`, `error_code` and `error_stage` on "error"; new reasons
  `policy_withheld`, `subjects capped` and `ledger incomplete` (a capped check
  reports `subjects capped` where it used to report `every step executed` or
  `unexecuted steps`; read `fired` for the outcome). `search_memory` events:
  the `router` record gains `vendor_ms`, `provider` and the four `error_*`
  fields, with new error values `router_budget`, `policy_withheld`,
  `CallTimedOut`, `OutOfTime` and `VendorBusy`; `listing_skip_reason` gains
  `router_budget` and `rows_do_not_fit`; `short_verbatim` gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`,
  and now always has `listing_parsed: false`. Span attributes and the span
  contract are unchanged.
- **Later on 2026-09-28**: `finish_check` events and router records gain
  `vendor_calls_in_flight` (vendor worker slots held, only when above 0);
  `listing_skip_reason` gains `policy_withheld` (until then counted as
  `router_error`), and a withheld route no longer uses one of the turn's 3
  routing calls. Every vendor call now passes the SDK a timeout of its cutoff
  plus 0.5 s (the client's own if shorter), and an abandoned call gives its
  worker slot back at most 0.5 s after its cutoff even if a trickled body keeps
  it running (`jev_client.calls_orphaned`); while 8 such requests still run
  (`jev_client.ORPHANED_CALLS_MAX`) every new call fails at once as
  `VendorBusy` and sends nothing, so a trickling endpoint cannot pile up
  threads and sockets, and `finish_check` events and router records carry
  `vendor_calls_orphaned` (only when above 0). The finish check's ledger reads
  labels from a page or one group of a longer listing
  (`parse_table(text, require_complete=False)`; rows are still served only from
  a complete one), its archive reads wait at most 0.5 s for a locked database
  and a failed read now ends the check as `error` at stage `ledger` (until
  then the inline text was used). Its context-clause reads share the process
  caches every other clause read uses, the "no subject" answer included, so a
  second check on a turn reads no alias from the archive again. With `FW_EVAL_FINISH_REMINDERS=0` a check
  that is attached no longer seeds a resumed trajectory, records dispatch
  outcomes or restores the suspended plan. The async entry (`aforward`) now
  resets the vendor budget and dispatch outcomes per turn, as `forward` does.
  `calibration` stays on events only: it is not a `fw.finish_check` span
  attribute, and the span contract is unchanged.
- **`FW_EVAL_FINISH_REMINDERS` is read from the workflow's `fastworkflow.env`
  first**, then the process, and `0` now also turns off the structured planner:
  a control arm run with it gets the plain-text planner and no plan markers. A
  value in the env file other than `0` now fails the agent build, as one in the
  process did; an empty value is unset.
- **The search evidence bound halves on the example model.** The example
  configuration's `mistral/mistral-small-latest` reports a 262,144-token
  window, which gave a 262,144-byte bound; the 131,072-byte ceiling now applies,
  so observations of 128-256 KB are searched as a disclosed prefix. Set
  `FW_SEARCH_OBSERVATION_MAX_BYTES=262144` to keep the old bound. No recorded
  downstream observation is over 5.4 KB.
- **`listing.served_rows` returns `Optional[tuple]`**: `None` when not one row
  fits the bound. A caller outside the framework must handle it.
  `parse_table` keeps its return shape and returns `None` more often.
- **Vendor-call limits are module constants**, not settings
  (`jev_client.VENDOR_WORKERS`, `TURN_VENDOR_SECONDS`,
  `ROUTER_CALLS_PER_TURN`; `finish_check.SUBJECTS_MAX`, `LEDGER_MAX_BYTES`,
  `LISTING_PARSE_MAX_BYTES`). A turn resumed in another process starts a fresh
  vendor budget.
- **Agent-visible wording changed on the default path without re-measurement**
  (fix-94m9): the offload-label mark, the `search_memory` description, the
  `WorkflowAgentSignature` docstring and the rehydration note prefix (texts
  under Changed). A deployment that compared agent runs against a baseline
  taken before this change is comparing different prompts. A reader that
  matched the exact old mark "Restored in full for the final answer." must
  accept "Normally restored for the final answer."; `labels.LABEL_RE` parses
  both.
- **fix-6jzi depends on a private DSPy hook**
  (`ChatAdapter._make_json_adapter_fallback`). A DSPy upgrade must rerun
  `tests/test_chat_adapter_commands.py`; if the hook is gone the retry silently
  loses the command list again.
- **Changing a finish-check or router question now fails
  `tests/test_decision_measurements.py`** (fix-8ee5). Update the digests in
  `tests/fixtures/decision_measurements.json` only together with a new
  measurement recorded there. **The measurement evidence lives only in local
  directories outside this repository** (with the manifest-gate replay
  scripts and the manifest generator) and must be archived somewhere durable;
  the manifest holds only their sha256 and paths.
- `fastworkflow.observation_offloading.related` is a new module (fix-vpe3);
  every moved name is still importable from `observation_offloading.search`.

### Known limits

Offloading and search ship with documented limits rather than silent ones: what
a turn resumed in another process reads, which part of the evidence is stored
unredacted, when pruning runs, the worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.
