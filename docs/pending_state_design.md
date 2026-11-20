# One waiting record and one transition table for a turn

Status: agreed design (revision 3, with the owner's final decisions of 2026-11-19; they are marked CONFIRMED). Covers bd
fix-fpyb.8, and folds in fix-fpyb.6 (`//cancel` must know every pending kind) and fix-fpyb.7 (what a "/" message does
while a plan question is pending). Line numbers are from the tree of 2026-11-19 (other agents are editing); `WEC` is
`fastworkflow/workflow_execution_context.py`.

How this serves the product goal. A planning command and an execution command must get the same intent detection,
the same parameter extraction and the same clarification experience, with no switch on the kind of command or on the
path. Today the two paths keep their pending question in different places, read it with different chains and stop
asking by different rules. This design gives both paths ONE record type in ONE home (the turn object), ONE `parameters`
kind, ONE transition table and ONE asking rule, so that what differs between the paths is only what is inherent
(section 7).

Steps 1 to 9 are done (section 3.8 records step 4 as built, the corrections it made to this note, and the frozen reply-dispatch contract; section 3.9 records step 5; section 3.10 records step 6; section 3.11 records step 7; section 3.12 records step 8; section 3.13 records step 9). Step 10 is the cleanup. Sections 1 and 2 describe the tree before step 1; where a later step changed a fact, its own section says so.

Already fixed under the epic: fix-fpyb.1, .2 (the review-open flag is cleared in one place, WEC:2739), .3 (turn flags
saved), .5, and the first half of .4: `intent_loop.store` / `clear` and `parameter_clarification.store` / `clear` are
now the only writers of the two CME keys, and `waiting_for` raises when both are set (`intent_loop.py:774-799`,
`parameter_clarification.py:290-313`). The bead for .8 also names `reply_parts` and `review_from`; they were removed
with fix-rpio and are not in the tree.

## 1. The problem, and every pending kind today

A turn has no single answer to "what is this session waiting for". Four if-chains work it out again each time
(`WEC._execute_message`, `_resume_agent_message`, `_pending_prompt_kind`, `ChatSession._pending_cancel_reply`) and a
fifth lives in the built-in `wildcard.py`. They read about twelve flags in six records, held in THREE places: the WEC,
the plan resolution, and two CME workflow contexts (`_cme_workflow` and `_plan_cme`, the second one holds the plan
review's slot question, `review_engine.py:1771`, `:1968`). The chains agree only because records are cleared in a
particular order; nothing asserts it. The chains already disagree: `//cancel` knows two of the nine kinds below, and a
"/" message is routed by the CME's records only.

"Saved" means written by `serialize_state` (WEC:1543) and restored by `apply_serialized_state` (WEC:1649).

| # | Kind (client `question_kind`) | Where its state lives | Set by | Cleared by | Reply reaches its reader through | Cancel today | Saved |
|---|---|---|---|---|---|---|---|
| 1 | Agent question (`agent`) | `_awaiting_user` (WEC:390), `_suspended_user_message`, `_pending_clarification_request`, the agent's suspended react state | `_settle_agent_result` (WEC:5574); framework call (WEC:4426) | `_reset_agent_suspension` (WEC:2739); WEC:5608 | `_execute_message` -> `_resume_agent_message` (WEC:5668) -> `_post_ask_user_response` | REST `cancel_pending` (WEC:1927); CLI `post_cancel` of the blocked read (`chat_session.py:388`); not in `_pending_cancel_reply` | yes |
| 2 | Plan which-command (`plan_choice`) | `PlanResolution.choice`, `.asking`, `.pending` (`resolution.py:127-135`) plus #1's flags | `_ask_choice` (`resolution.py:217`) | `take_reply` (`resolution.py:436`); `_after_resolution` (WEC:3107); `_reset_agent_suspension` only sets `pending` False (WEC:2745-2751) | `_resume_agent_message` -> `_resolution_pending` (WEC:3062) -> `_resume_plan_resolution` (WEC:3066) -> `intent_loop.answer` | not in `_pending_cancel_reply`; the reader reads `cancel` | yes |
| 3 | Plan values at the gate (`planning_parameter`) | `PlanResolution.loop` (`ParameterLoop`, `parameter_clarification.py:157`), `.asking`, `.pending` AND `loop.pending` (one fact twice) | `_values_question` (`resolution.py:268`) | as #2 | as #2, then `parameter_clarification.advance` | as #2 | yes |
| 4 | Part values, asked while a part runs (`planning_parameter`) | `PlanResolution.run_time(loop)` (`at_gate` False) | `_open_parameter_loop` (WEC:4230), `_suspend_on_parameter_loop` (WEC:4224) | `_resume_parameter_loop_unlistened` (WEC:4279); `_begin_turn` (WEC:902) | `_resume_plan_resolution` -> `not at_gate` -> `_resume_parameter_loop_unlistened` | as #2 | yes |
| 5 | Sub-task review (`review`) | an approval-store record of kind `sub_tasks` (`approval_binding.py:625`), AND `turn.subtask_review_open` (set WEC:3163), AND `turn.framework_entry` (set WEC:4420), AND `_turn_subtask_review == "open"` (WEC:3159); its slot question sits in `_plan_cme`'s `parameter_loop` | `_open_review` (WEC:3155) via `run_framework_planning_command` (WEC:4397, its only caller) | accept (WEC:3214), decline or expiry (WEC:3236), `_reset_agent_suspension` | `_resume_agent_message` approval branch -> `approval_binding.resolve_reply` (`approval_binding.py:458`); frame loop tests `framework_entry` and `subtask_review_open` (WEC:5728-5729) | `//cancel` sends `no` (`chat_session.py:357-373`) | record yes; `subtask_review_open` yes; `framework_entry` rebuilt (WEC:1702); slot via `plan_cme` |
| 6 | Approval of an action (`approval`) | approval-store record of kind `action`, `suspended_agent` | the approval gate | `resolve_reply`, `cancel_all` (`approval_binding.py:597`) | not suspended: `_resume_approval_direct` (WEC:5598); suspended in an agent run: as #5 | `no`; `/cancel_pending` (`approval_binding.py:125`) | yes |
| 7 | Decision step (`phase_step`) | `turn.phase_run["decision"]` (set WEC:4122, popped WEC:4159, :4188, :4193), `agent._suspended` (WEC:4131), plus #1's flags | `_suspend_for_decision` (WEC:4081) | `_read_decision_reply` (WEC:4145) | `_resume_agent_message` `elif phase_run.decision` (WEC:5699) | not in `_pending_cancel_reply`; the reader reads `cancel` | yes |
| 8 | CME which-command (`plan_choice`) | CME context key `intent_loop` (`intent_loop.py:77`) | `wildcard._detect` (`wildcard.py:251`), `_open_misunderstanding` (`:143`), via `intent_loop.store` | `intent_loop.clear` (`wildcard.py:243`); `Workflow.end_command_processing` (`workflow.py:395-410`) | `_process_message` -> `invoke_command` -> `wildcard.__call__` -> `waiting_for` -> `_answer_question` (`wildcard.py:304-312`) | `//cancel` sends `abort` because `has_open_command` (WEC:1343) | yes (`cme.intent_loop`) |
| 9 | CME parameters (`parameter_request`) | CME keys `parameter_loop` (`parameter_clarification.py:38`), `command`, `command_name` | `parameter_extraction.py:252`, `:500` via `parameter_clarification.store` | `parameter_clarification.clear` (`parameter_extraction.py:466`, `:508`, `wildcard.py:332`), `end_command_processing` | as #8, then alias check, own-command check, Jev reply read, merge (`wildcard.py:320-345`) | as #8 | yes (`cme.parameter_loop`) |

Also: `_turn_subtask_review` (seven string values, written at WEC:2994, :3110-3132, :3159, :3231, :3248) mixes one waiting
value ("open") with six outcomes; `_open_parameter_loop` reads it as a guard (WEC:4248).

## 2. The dispatch chains as they are

Chain A, `_execute_message` (WEC:2163-2205), in order, as it was before step 2 (`_route_message` now runs first and replaces A5; A3 and A4 read the message as the user typed it for `/approvals` and `/cancel_pending`, a "/" reply otherwise has its "/" removed).

| Step | Check | Result | Line |
|---|---|---|---|
| A1 | state belongs to another conversation | park it (approvals cancelled); the message starts a new turn | 2175 |
| A2 | not `_awaiting_user` | `_begin_turn` (a reply never begins a turn) | 2181 |
| A3 | no approval, or a non-suspended one | `management_reply` (`/approvals` ...) | 2188 |
| A4 | approval active and (not suspended in an agent run, or text is `/cancel_pending`) | `_resume_approval_direct` | 2193 |
| A5 | `_prepare_message_routing`: no CME record pending and text starts with "/" | set `is_assistant_mode_command` | 2668-2673 |
| A6 | `run_as_agent` and flag absent | `_awaiting_user` -> `_resume_agent_message`, else `_process_agent_message` | 2196, :2675 |
| A7 | otherwise | `_process_message` -> deterministic path | 2201 |

Chain B, `_resume_agent_message` (WEC:5668-5730): B1 `_resolution_pending` (plan resolution, or the part's question when
`not at_gate`); B2 no suspended agent state raises `NoSuspendedAgentStateError`; B3 approval record with
`suspended_agent` -> `resolve_reply`; B4 `phase_run["decision"]` -> decision reply; B5 else `_post_ask_user_response`;
B6 frame loop tests `framework_entry` and `subtask_review_open`.

Chain C, `_pending_prompt_kind` (WEC:2354-2417): review or approval record, then `_resolution_pending`, then
`phase_run["decision"]`, then ask_user. A third copy of B's order and the only producer of `question_kind`.

Chain D, `wildcard.__call__` (`wildcard.py:295-345`): `waiting_for` -> intent question -> `_answer_question`; parameters
-> alias for abort / you_misunderstood -> own command token -> Jev reply read -> merge; nothing -> `_detect`.

Chain E, `ChatSession._pending_cancel_reply` (`chat_session.py:357`): any active approval or review -> `no`; else
`has_open_command` -> `abort`; else None. Used by `//cancel` (`:375`) and `_cancel_unseen_prompts` (`:395`).

Two more refusals say "a question is open" in a different idiom: `process_action` returns a failed output
(`AWAITING_ANSWER_REFUSAL_MESSAGE`, WEC:2613) and `check_use_plan` raises `UsePlanError` (WEC:2153). "Can anyone answer"
is answered five times: `workflow_agent.py:936`, `:951`, `:960`, `_can_ask_decision` (WEC:4070), `_subtask_no_answerer`
(WEC:2960).

### Known surprises

1. A "/" message during a plan question (fix-fpyb.7). A5 looks only at the CME's records, which are empty for kinds 1 to
   7, so the flag is set; A2 does not begin a turn; A6 falls to A7. The command runs on the deterministic path, its
   output is appended to the suspended turn, and the plan question stays pending with no sign that anything intervened.
   On the deterministic path the same message in the same position is read as a reply, so the two paths disagree.
   Nothing tests it.
2. `//cancel` blind spots (fix-fpyb.6). Chain E does not know kinds 2, 3, 4, 7 or a Topology B agent question. It is
   safe today only because those questions are not put when a blocking queue is set (`workflow_agent.py:946`, `:956`;
   WEC:4077), and the CLI always sets one. With an approval and an open CME command both present, E picks `no` and
   leaves the command open.
3. "Pending" has two meanings: `_reset_agent_suspension` sets `pending` False but keeps `_plan_resolution` so an agent
   turn asks once (WEC:2745-2751); `res.pending` and `res.loop.pending` are set together (WEC:4226).
4. `framework_entry` and `subtask_review_open` are the same fact (one caller, WEC:3167; restore copies one into the
   other, WEC:1702).
5. A review record opens only after `_plan_resolution` is None (WEC:3107), and B3 finds it only because B1 is false.
6. The "/" flag is set at one place (WEC:2673, restore WEC:1763) and cleared at about eleven (WEC:1479, :1493, :5612,
   :5642; `command_executor.py:667, :687, :707, :723, :771, :815`).

## 3. The proposal

### 3.1 One record, one home

A new module `fastworkflow/waiting.py` (no model imports) holds a dataclass.

```python
KINDS = ("intent", "parameters", "review", "approval", "decision", "agent")

@dataclass
class Waiting:
    kind: str        # one of KINDS
    assistant: bool  # the round was started by a "/" message
    question: str    # the text the user saw, as the reader needs it
    payload: dict    # per kind, below; plain JSON data only
    def to_state(self) -> dict: ...
    @classmethod
    def from_state(cls, state) -> "Waiting | None": ...
```

"No record" is None. A record in the slot means a question is out; it is never saved otherwise (see 3.5 for the one
in-call exception). The kinds map onto the nine rows: `intent` = 2 and 8, `parameters` = 3, 4, 9 and the review's slot,
`review` = 5, `approval` = 6, `decision` = 7, `agent` = 1.

| Kind | Payload |
|---|---|
| `intent` | `record`: the `intent_loop.open_question` dict, unchanged (`intent_loop.py:665`); `part` (plan path: index into the resolution) |
| `parameters` | `loop`: `ParameterLoop.to_state()`, unchanged; `after`: what follows the answer (below); `part`; for `after == "run_command"` also `command_name` and `command` (the CME's scratch text) |
| `review` | `record_id` (the approval-store record; the store is not folded in); `slot`: a nested `Waiting` of kind `parameters` with `after == "edit_slot"`, or None |
| `approval` | `record_id`, `suspended_agent` |
| `decision` | the dict now at `phase_run["decision"]`, typed |
| `agent` | `clarification`, `suspended_message`; the react blob stays where it is |

`after` is the one field that replaces the old `part_parameters` kind. Its values: `resolve_next` (plan gate: carry
the resolution on to the next part, the review or the run), `resume_part` (a part was running: resume it in place,
today `_resume_parameter_loop_unlistened`), `run_command` (the CME: run the command with the values) and `edit_slot`
(the review's slot question). Which one applies is decided where the record is opened, so what the reader reads and
how the question is worded are the same for all four.

The home is ONE attribute, `WEC._waiting` (the turn object; it moves with the turn and is parked with it). The plan
resolution's progress record (`PlanResolution`: `parts`, `index`, `stage`, `pinned`, `asked`) stays beside it and loses
`pending`, `asking`, `choice`, `loop` and `at_gate`.

### 3.2 How the built-in commands reach the record (item 1)

The built-in commands run in a CME `Workflow` whose context is a dict they already read (`context["app_workflow"]`);
they do not see the WEC (checked: no reference to the execution context in `wildcard.py`, `parameter_extraction.py`,
`intent_detection.py`). The record stays on the WEC and the CME is given a port to it.

- `waiting.py` also defines `Port` (methods `get() -> Waiting | None`, `put(record)`, `clear()`, and the read-only
  `assistant`) and `Holder`, a port that keeps its record in itself.
- The WEC builds two ports: the turn port (reads and writes `WEC._waiting`; `put` replaces any record, so opening a
  record closes the other, which is what `intent_loop.store` does for the two keys today) and the review-slot port
  (reads and writes `WEC._waiting.payload["slot"]` while the record is a `review`, else raises). It stores the first in
  `_cme_workflow.context["waiting"]` where it sets `context["app_workflow"]` (WEC:2025) and the second in
  `_plan_cme.context["waiting"]` where `_plan_cme` is created (WEC:564). A port is a live object in a context, as
  `app_workflow` already is; `Workflow.flush` does not serialise contexts (`workflow.py:504`).
- A port clears only the kinds its holder owns (`intent`, `parameters`): `clear()` on the turn port never drops a
  `review`, `approval`, `decision` or `agent` record.
- The helpers keep their names and become thin: `intent_loop.waiting_for(context)`, `store`, `clear` and a new
  `pending_record(context)`; `parameter_clarification.pending`, `store`, `clear`. Each does `context["waiting"]`
  (installing a `Holder` when the context has none, so a CME built without a WEC, as in the tests, still works) and
  reads or writes it. The "at most one" rule is then structural (one slot) and the raise in `waiting_for` goes.
- `Workflow.end_command_processing` (`workflow.py:395`) drops the two key pops and calls `self._context["waiting"].clear()`
  when a port is present, and still deletes `command`.
- The session blob's `cme` section (`_serialize_cme_continuation`, WEC:1308; `_apply_cme_continuation`, WEC:1857) and
  `plan_cme` section are deleted. The record is saved once under `waiting` (the `review` payload carries the slot).
  `has_open_command` (WEC:1343) becomes `waiting.kind in ("intent", "parameters")`; `_clear_cme_continuation`
  (WEC:1399) becomes "drop that record, pop `command` and `command_name`". On restore the WEC re-installs the CME's
  scratch keys `command` and `command_name` from the `parameters` payload (they stay CME scratch, see section 7).
- The "/" flag leaves the CME context: it is `Waiting.assistant` for a question that is out, and a turn field
  `WEC._assistant_round` between the message and the question. The CME reads it as `context["waiting"].assistant`
  (the port answers from the turn field). The eleven pops and the saved `assistant_command` key are replaced by one
  `end_round`.

Cost in the built-in workflow. Only files under `_workflows/*/_commands/` are training-signature inputs. The edits
there are in `wildcard.py` alone and are small: two direct reads of the key (`wildcard.py:80`, `:208`, to
`intent_loop.pending_record(context)`) and the `_agentic` flag read (`wildcard.py:133`, to the port's `assistant`),
about 6 lines. Everything else the CME does with the record already goes through `store` / `clear` / `pending` /
`waiting_for`, which live outside `_commands/` (`intent_loop.py`, `parameter_clarification.py`,
`parameter_extraction.py`, `intent_detection.py`, `workflow.py`). No other `_commands/` file touches the keys
(checked: `ErrorCorrection/*`, `IntentDetection/*`). So this step's retrain is one (step 4), batched with the
fix-fpyb.4 / fix-kc83 edit if that has not shipped yet; step 7 edits `wildcard.py` once more (section 6, risk 1).

No hard obstacle was found. The nearest ones: (a) `review_engine.py:1235`, `:1771`, `:1968` read and write the plan
CME's record, now through the slot port; the port raises if the WEC's record is not a `review`, which would be a real
bug, not a state to tolerate. (b) A `Workflow` created again under one id keeps its context, so the ports are installed
after every `Workflow.create` (as `app_workflow` is). (c) The CME's scratch keys `command` and `command_name` are read
unconditionally (`wildcard.py:314`); they are left as CME scratch rather than moved, to keep the `wildcard.py` edit at 6
lines (section 7).

### 3.3 One transition function

```python
def step(record: Waiting | None, event: Event, turn: TurnFacts) -> Move: ...
# Event: Reply(text) | Cancel | Restore(state) | NewTurn(message) | Slash(message)
# Move : next: Waiting | None, then: Read(reader) | Resume | Done(output) | Refuse(text) | Begin
```

Pure over data, no model call. `Then` names which reader to call (`intent_loop.answer`, `parameters.read`,
`approval_binding.resolve_reply`, `_read_decision_reply`, `_post_ask_user_response`; all unchanged) and which resume to
run. `TurnFacts` is the one predicate set: `attended`, `queue_blocks`, `agent_model`. Five "can anyone answer"
predicates become two: `can_ask` (attended and no blocking queue) and `can_review` (today's `_subtask_no_answerer`,
which adds the host and no-agent rules).

### 3.4 The transition table

Rows = kind. Cells = what happens, and the next record.

| Kind | Reply | Cancel | Restore | New turn (nothing out) | "/" message |
|---|---|---|---|---|---|
| none | not a reply | `//cancel` says nothing is pending | no record | `_begin_turn`; `assistant` set if the text starts with "/" | the same |
| `intent` | `intent_loop.answer`: pick -> bind the part (plan) or move on to parameter extraction (CME) -> next record; none/abort -> drop the part / end the command; unclear -> ask once more, second unclear -> drop / `what_can_i_do` | cancel text `abort`; same as the reader's abort | put the record back | not reachable | reply, "/" removed |
| `parameters` | `parameters.read` -> the asking rule (3.6): progress -> `after` carries on (`resolve_next`: next question, review or run; `resume_part`: the part resumes in place; `run_command`: the command runs; `edit_slot`: the edit is applied) and the record is dropped; ask again -> same kind; stop -> hand-over (3.6) | cancel text `cancel` (plan) / `abort` (CME): the part or command ends as declined; later parts go on | put the record back; `run_command` re-installs the scratch keys | not reachable | reply, "/" removed |
| `review` | `resolve_reply`: not done -> stays (re-prompt); accepted -> run the accepted parts (WEC:3214) -> none; declined or expired -> WEC:3236 -> none; while `slot` is out the reader asks the slot's `parameters` row first | cancel text `no` -> declined -> none | store record restored, free-text reader re-armed | not reachable | reply, "/" removed; `/cancel_pending` unchanged |
| `approval` | `resolve_reply`: not done -> stays; approved -> run (the agent's observation if suspended) -> none; declined -> none | cancel text `no`; a suspended one abandons the agent's call | as `review` | not reachable | `/cancel_pending` cancels; any other "/" text is the reply |
| `decision` | `_read_decision_reply`: go -> the phase resumes; ask -> stays, `asked` + 1; stop -> the command ends with its stop (`fw_not_run`) | cancel text `cancel` (declines) | stash restored with `react` | not reachable | reply, "/" removed |
| `agent` | `_post_ask_user_response`: the agent resumes -> none or another record | blocked read: `post_cancel`; otherwise `cancel_pending` (`_reset_agent_suspension`) | restored with `react`; none -> `NoSuspendedAgentStateError` | not reachable | reply, "/" removed |

Rules in every row:

1. A message while a record is out is its reply or a cancel. It never begins a turn, a request or an action;
   `process_action` and `check_use_plan` refuse with one shared text.
2. `put` replaces the slot, so two records cannot exist. A restored blob holds one `waiting`.
3. Cancel asks the table for the kind's `cancel_text` and delivers it as a reply through the same path (the CLI puts it
   on the user queue). `agent` is the one cancel that is not a reply.
4. Park for a conversation change (A1) moves one `Waiting.to_state()` whole, with its `assistant` flag, instead of the
   list of parked keys.

### 3.5 What each existing flag or key becomes

| Today | Becomes |
|---|---|
| `_awaiting_user` (WEC:390, about 40 references) | read-only property: a record is out and its kind is not `intent` / `parameters` on the CME |
| `_suspended_user_message`, `_pending_clarification_request` | `agent` payload |
| `PlanResolution.pending`, `.asking`, `.choice`, `.loop`, `.at_gate`, `ParameterLoop.pending` | deleted |
| `turn.subtask_review_open`, `turn.framework_entry`, `_turn_subtask_review == "open"` | deleted; `kind == "review"`. The other six values stay as `turn.review_outcome` |
| `turn.phase_run["decision"]` | `decision` payload |
| CME keys `intent_loop`, `parameter_loop` and the `cme` / `plan_cme` blob sections | deleted; the one record and the port (3.2) |
| `is_assistant_mode_command` | `Waiting.assistant` and `_assistant_round`; one `end_round` |
| `_pending_prompt_kind`, `_pending_cancel_reply`, `waiting_for` ladders | one table lookup: kind -> `question_kind`, payload builder, `cancel_text` |
| `PlanResolution.run_time`, `_parameter_question_asked` (WEC:4207), `ParameterLoop.policy` | deleted (3.6) |

In-call exception: when a reply arrives for `parameters` with `after == "resume_part"`, the record stays in the slot
while the part re-runs inside the same engine call (its own checks may refuse again, and `reopen` must measure progress
against the last question, `parameter_clarification.py:351`). No state is saved while it is so held; the engine
asserts at return that a record is either absent or out. This replaces the `out=False` state of revision 1, which is
therefore not part of the model or the saved form.

### 3.6 The asking rule (item 3). CONFIRMED

One rule for a missing or refused value, for planning and execution commands alike:

1. Keep asking while each reply makes progress (fills or fixes at least one listed value).
2. After a reply with no progress, ask once more. After a second such reply in a row, stop.
3. Cancel always ends it.
4. The only thing that depends on the session is what "stop" hands over to: an agent model if one is set (it gets the
   refusal as today: the text planner or the agent), otherwise the part or command ends as not run and names exactly
   what is missing. A pinned plan's confirmation (`confirm`) has nobody to hand to and always ends as not run.

Where the three policies are implemented today:

| Policy | Where |
|---|---|
| Constants and doc | `parameter_clarification.py:28-34`; field `ParameterLoop.policy` (`:184`); `MAX_NO_PROGRESS = 2` (`no_agent_mode.py:62`) |
| Bounded (the shared rule) | `_settle` (`parameter_clarification.py:361-369`), `advance` (`:372-447`), `reopen` (`:351`); used by `resolution._take_values` (`resolution.py:395-433`) and WEC:4265-4277 |
| Once | `advance` early return (`parameter_clarification.py:409-420`); `fallback_runs` (`:387`, `:438`; WEC:4299); WEC:4257-4262 (`POLICY_ONCE` for a part); `resolution.py:408-427` branches on `turn.no_agent or loop.confirm`; `_parameter_question_asked` (WEC:4207) read by `workflow_agent.py:947`; `parameter_extraction.py:497` (the CME in an agent session) |
| Unbounded | `record_round` never ends (`parameter_clarification.py:339-348`); `parameter_extraction.py:242` and `:498` (a human session's CME stage); `park` default |

What changes, by session type:

| Session | Today | After |
|---|---|---|
| Human, deterministic ("/" or no agent model), execution command | unbounded: asked until `abort` | stops after two replies in a row with no progress, ends as a failed output naming what is missing; `abort` still works. This is the one change a human can see, in the rare case of repeated unusable replies |
| Agent model set, plan part or gate | one question; the reply is passed as text to the part's own typed call; a refusal after it goes to the text planner or agent with no second question; `cancel` is handed to the agent | a second question is possible while replies make progress; each reply is read by the shared extractor (not the part's own typed call); `cancel` ends the part; on stop the refusal goes to the agent or text planner as today |
| Agent model set, CME in an agent session (the audience is the agent) | the question is the refusal text returned to the agent, never ended by a count | the same text; two no-progress calls in a row end it as a failed output naming what is missing, instead of leaving the agent to its iteration limit |
| No agent model | bounded | unchanged |

`out=False` held record: not needed. Its two uses were the once-only rule (`_parameter_question_asked`), which is
deleted, and `reopen` across a part's re-run, which uses the in-call exception of 3.5. The count of replies without
progress is `ParameterLoop.stalled`, already in the record.

Tests that pin the old policies (found by `POLICY_`, `policy=`, "once" in tests): `tests/test_parameter_loop.py` (6
hits), `tests/test_planning_command_refusal_ask.py` (3), `tests/test_no_agent_review.py` (2),
`tests/test_extraction_contract.py` (1). Behavioural pins to run and read: `test_jev_missing_value_reply.py`,
`test_misunderstood_during_parameter_question.py`, `test_clarification_reply_merge.py`, `test_param_retry_round_ordinal.py`.
Expectations change; none is deleted.

Measure before and after, on the attended benchmark (`docs/benchmark_setup.md`, the identity-risk-analyst persona), the
same sample both times, counted from the turns' outputs (`question_kind` of awaiting outputs) and the saved loop
(`asks`, `stalled`): questions per turn; first-reply-not-accept rate (loops whose first reply closed no open value);
turns ending as not run; turns ending by hand-over to the agent; extra model calls per turn in agent sessions (the
extractor now reads free-text replies). Ship the rule only if questions per turn do not rise by more than the owner's
bar and not-run endings do not rise. It is a separate step (6) so it can be measured, held back or reverted alone.

Baseline, measured before any code of this migration (one round of 119 tasks on a large private workflow, 2026-11-19):
with no agent model 123 questions, and 2 turns with more than one question; with an agent model set 126 questions.
The same round is the "before" of the measurement at step 6.

### 3.7 The rule for fix-fpyb.7. CONFIRMED

Decided: while any question is out, a "/" message is the reply to it, with the leading "/" removed. It never starts
a command and never runs beside the question. `/cancel_pending` keeps its meaning. Why: it is the rule the
deterministic path already has, so the two paths agree and there is one idiom; nothing is appended to a suspended turn;
the way out of a question is already one thing. Cost: one small rule, stripping "/" before the reader, because the CME
reader saw the raw text. Done in step 2 (`WEC._route_message`, every leading "/" removed as `/invoke_agent` does). One
consequence the note did not list: `process_action` and `use_plan` are now refused while the CME's own question is out
too (they used to run, because the CME's kinds are not `_awaiting_user`); the refusal asks the `waiting` view, not the flag.
The decision-step question's text no longer offers `invoke_assistant` as a way to run a command meanwhile.

Considered and rejected: refusing the message with the text `process_action` uses (safe and needs no stripping, but a
user who typed a command gets a refusal where the reader would say "unclear", and the CME path would have to change to
match), and cancelling the question and running the command (drops state unasked; makes "/" a second way to cancel).

### 3.8 Step 4 as built (DONE), and what it corrected

What exists now. `waiting.Waiting` gained `assistant`; `waiting.Port` (`get`, `put`, `clear`, `assistant`, `end_round`),
`OwnerPort` (a weak reference to the execution context and the attribute it reads and writes: `_waiting`, or
`_review_slot`), `Holder` (the fallback: its record in itself, its "/" flag under `ASSISTANT_KEY` in the context it sits in),
`port_of(context)` (installs a `Holder` when there is no port). The WEC builds two `OwnerPort`s after creating its two
built-in workflows (`_install_waiting_ports`): the turn port (`context["waiting"]` of the CME, home `WEC._waiting`) and the
review-slot port (`context["waiting"]` of the plan CME, home `WEC._review_slot`, no "/" round). The helpers
(`intent_loop.store` / `clear` / `pending_record` / `waiting_for`; `parameter_clarification.pending` / `requestion` /
`store(after=)` / `clear`) read and write the port; `INTENT_LOOP_KEY` and `CME_LOOP_KEY` are deleted. A CME record is
`Waiting(kind="intent", owner="cme", payload={"record": <the intent dict>})` or `Waiting(kind="parameters", owner="cme",
payload={"loop": <ParameterLoop.to_state()>, "after": "run_command" | "edit_slot"})`. `Workflow.end_command_processing`
clears through the port (only the kinds a CME port owns). `waiting.derive` (gone at step 9) read `WEC._waiting` for the CME, so two CME
questions can no longer be out at once and `waiting_for` no longer raises. The session blob has `waiting` and `review_slot`
(each `Waiting.to_state()` with `command` and `command_name` added to its payload at save time and re-installed as CME scratch
keys on restore), keeps `assistant_command`, and has no `cme` or `plan_cme`; `SCHEMA_VERSION` is 5. The "/" flag is
`WEC.assistant_round` (turn field) mirrored on the record; `start_round` (the route), `end_round` (every other place, through
`WEC.end_round` or `port.end_round()`; the executor's pops are `command_executor._end_round`).

Corrections to this note.

1. The note's `Waiting` had `assistant` and no `owner`; step 1 built `owner` (kept) and step 4 added `assistant`.
2. The review's slot port cannot read `WEC._waiting.payload["slot"]`: a review is not a `Waiting` record until step 8 (it is
   still derived from the approval store), so `_waiting` cannot be of kind `review` yet. The slot has its own attribute,
   `WEC._review_slot`, saved as `review_slot`; step 8 folds it into `review.payload["slot"]` (done, section 3.12). The slot is not counted by
   `derive` (it never was: the plan CME was not read).
3. `wildcard.py` did not need "about 6 lines": it read the key at three places, the flag, and held the whole reply block. As
   built, the reply block (alias read, own-command check, remote read, merge) is `parameter_reply.read_reply` and `wildcard.py`
   keeps `_dispatch_reply`. The risk 1 two-retrain problem is gone: nothing later edits `wildcard.py` (the asking rule is the
   `stop` action, the shared reader is `read_reply`). The which-command reply block (`_answer_question`) stays in
   `wildcard.py`: no later step changes how that reply is read (step 5 is the plan path, which reads it with
   `intent_loop.answer`; its two differences from the CME, section 7, are inherent).
4. The port holds the owner weakly: a strong reference would make a reference cycle from the execution context through its
   workflow's context back to it.
5. A context with no port keeps the "/" flag under `is_assistant_mode_command` in itself (the fallback `Holder` reads it), which is
   what unit tests that build a bare CME context set; a real session never has that key.
6. The blob keeps `assistant_command` (the plan path's questions have no record yet, so the flag cannot live only in a record);
   it goes with the saved-state cleanup of step 9.

The frozen reply-dispatch contract (`command_metadata_extraction/parameter_reply.py`; `wildcard._dispatch_reply`). `read_reply`
returns `Reply(action, name="", in_parameters=True, keep_pending=False, reply_intent=None, text="", command="", message="")`.
`wildcard.py` handles EVERY action, and cannot change afterwards: later steps change `read_reply` only.

| action | meaning | fields | what the dispatcher does |
|---|---|---|---|
| `run` | run a built-in command | `name` (`ErrorCorrection/abort`, `ErrorCorrection/you_misunderstood`, `IntentDetection/what_can_i_do`), `in_parameters`, `keep_pending`, `reply_intent` | `_run_cme_command(...)`, then sets `reply_intent` when given |
| `wait` | answer with text, keep waiting | `text`, `reply_intent` | `Output(handled=False, routed=False, ...)`; the question stays |
| `different` | the reply is a new request | `reply_intent` | end the pending command, route the reply again, set `reply_intent` when the new route gave none |
| `replace` | the stored command is replaced | `command` | clear the loop, store `command`, go on to extraction |
| `merge` | the reply is an answer | `text` | go on to extraction with `"<command_name>: <text>"` |
| `stop` | nothing more will be asked | `message`, `reply_intent` | end the command; `Output(handled=True, routed=False, response=message)`; no question stays |

Today `read_reply` returns `run`, `wait`, `different`, `replace` and `merge` exactly as the block did; `stop` is for section 3.6.

### 3.9 Step 5 as built (DONE), and what it corrected

What exists now. The plan path's which-command question is `Waiting(kind="intent", owner="plan", payload={"record": <the
intent dict>, "part": <index>})` and its values question `Waiting(kind="parameters", owner="plan", payload={"loop":
ParameterLoop.to_state(), "after": "resolve_next" | "resume_part", "part": <index>})` (`parameter_clarification.waiting_record`,
`AFTER_RESOLVE_NEXT`, `AFTER_RESUME_PART`), in `WEC._waiting`. `PlanResolution` is `parts`, `index`, `stage`, `pinned`, `asked`
(`pending`, `asking`, `choice`, `loop`, `at_gate`, `run_time`, `question` and `prompt()` are gone; `resolution.prompt(record)` builds
the client artifacts). The stage functions put and take the record through `Turn.port` (`resolution.Turn(..., port=)`; the
WEC passes `OwnerPort(self, "_waiting")`): `take_reply` removes the record first (the question is no longer out) and puts it back
when it asks again. A restored plan record goes straight back to `_waiting` (`_apply_waiting`; no CME scratch keys). There is one
`parameters` kind: no `part_parameters`.

Corrections to this note.

1. The in-call exception of 3.5 ("the record stays in the slot while the part re-runs") is NOT built. Held in the slot, the record
   would make `has_open_command` / `derive` report a question out while the part runs, and a second suspension in the part (a
   decision step, an `ask_user`) would be two questions at once. The record leaves the slot at the reply (as `pending = False` did);
   the loop of the last question is kept as `WEC._part_loop` (a live `ParameterLoop`, set by `_put_part_loop`, saved as the blob key
   `part_loop`, reset by `_begin_turn`). `reopen` measures progress against it, and `_parameter_question_asked` (the once-only rule)
   reads it. Step 6 deletes `_parameter_question_asked` and, with the once policy, the second use of `_part_loop`; the progress use
   stays unless `reopen` is rebuilt on the record. The old `PlanResolution.run_time(loop)` (a resolution with no parts, made only to
   hold a loop) is gone: while a part's question is out or answered `_plan_resolution` is None.
2. `has_open_command` is "owner is the CME and the kind is `intent` or `parameters`": the plan path's records have kinds in
   `CME_KINDS` too (`CME_KINDS` is the kinds the turn port may drop, not an owner test).
3. `derive` reads `_waiting` after the decision step, not before it (the plan-resolution branch used to come before the decision
   in the "first one acted on" order; a test pins decision before the CME's record, and the plan record shares the slot with it).
4. `ParameterLoop.pending` was kept (a loop in a record is always pending, the flag is redundant); step 6 deleted it.

### 3.10 Step 6 as built (DONE), and what it corrected

What exists now. ONE asking rule in `parameter_clarification.py`: progress is `made_progress(asked, names)` (a value the last
question listed is no longer open), the stop is `stopped(loop)` (`loop.stalled >= no_agent_mode.MAX_NO_PROGRESS`), cancel is the
reader's decline. `ParameterLoop.policy`, `ParameterLoop.pending`, `POLICY_*`, `open_loop(policy=)`, `advance(fallback_runs=)`, the
"once" branch of `advance`, `WorkflowExecutionContext._parameter_question_asked` and the once-only gate of
`workflow_agent._refusal_may_ask_attended` are deleted. `advance`, `reopen` and `record_round` all measure progress with
`made_progress`; `advance` (the reply to a plan's question) and `reopen` (a part's refusal after its re-run) return `exhausted`
through `_settle` / `stopped`, and `reopen` returns it at once for a loop already stopped (the hand-over is made once, so a re-run
after the stop, or a planner's second call of the command, is never asked a third time). `record_round` (the CME stage) returns an
`Outcome` that is `exhausted` on the second reply in a row with no progress, and `ParameterExtraction._extract_impl` then clears the
loop and returns the failed output naming what is missing. The one hand-over place of a plan part is `_resume_parameter_loop_unlistened`
and `resolution._take_values` (agent model: run on with what was given, so the text planner or the agent takes the refusal; none:
not run); `_open_parameter_loop` is one code path for both modes, with `_refusal_stop_reason` set only with no agent model.

Corrections to this note.

1. The stop on the deterministic path is NOT delivered through the dispatcher's `stop` action. `parameter_reply.read_reply` runs
   before the extraction and progress is known only after it (what the values read as, what the validators refused), so the reader
   cannot return `stop` without running the extraction itself. `ParameterExtraction._extract_impl` ends the command (the way it
   already ends a refusal that names no input) and `wildcard.py` returns its failed output unchanged. `ACTION_STOP` stays in the frozen
   vocabulary, still unused by a reader; nothing needs it.
2. `_part_loop` stays, and 3.5's "`stalled` is already in the record" is not enough to delete it. The record leaves the slot at the
   reply (3.9 correction 1), the part re-runs, and its typed call can be refused again, possibly after a second suspension (the agent's
   own `ask_user`, saved and restored); `reopen` needs the names the last question listed and the stall count at that moment, and no
   record is in the slot to carry them. It now holds only that: its `pending` and once-only uses are gone.
3. The note's "Agent model set, plan part or gate" row: the gate (`resolution._take_values`) already read every free reply with the
   extractor; only the running part's question (`_resume_parameter_loop_unlistened`) read the reply as the part's own text. And the
   first question of a refused typed call is now asked when "the part ran nothing", as with no agent model (`_part_ran_something`),
   not when "the turn has no outputs": an `ask_user` entry of the first question is a turn output, which would have blocked a second.
4. A reply `read` takes as the value of the single open field is a reply that "made progress" only if the value is then accepted: a
   refused value is still open, so it is no progress (it is how a human session's `bad` / `worse` replies stop).
5. 3.6's list of pinned tests named `tests/test_plan_choice_question.py`, `tests/test_refusal_gaps.py`, `tests/test_use_plan.py`,
   `tests/test_resolve_before_review.py`, `tests/test_plan_waiting_record.py`, `tests/test_waiting_kinds.py`,
   `tests/test_parameter_pipeline.py` / `tests/test_parameter_ask_read.py` (their recorded `parameter_loop` state kept `policy` and
   `pending`; the corpus modules now put those two keys back when they record, so the frozen tables are unchanged) as well.

### 3.11 Step 7 as built (DONE), and what it corrected

What exists now. ONE reader of a reply to a value question, `fastworkflow/parameters/reply.py` `read_reply`, used by both paths and importable without the model stack (a test imports it in a fresh process and checks that `dspy`, `torch`, `litellm` and `transformers` are not loaded). Order: (1) `cancel` and its synonyms (`abort`, `stop`, `never mind`; the code rule of `parameters.read`) are `decline`, and a confirmation's `yes` is `accept`; (2) the caller's `phrases` function: the plain utterances of `abort` (`decline`) and of `you_misunderstood` (`misunderstood`), `reply.phrase_of`; (3) the caller's own rule, `replaces` (an execution command's pending command sent again in full: `replace`); (4) a reply that read as `name: value` lines is values and is read no further; (5) any other reply goes to the caller's `remote` function ONCE (`reply.remote_intent`: the CME's `classify_missing_value_reply`, so its threshold, prompt and `FW_JEV_MISSING_VALUE_REPLY` switch are unchanged; no call with no key, with the switch off or when the call fails): `wait`, `abort` (`decline`), `misunderstood`, `different_command` (`different`); `answer` leaves the code's reading; (6) else `parameters.read`'s `values` or `free`. The result is `Reading(kind, values, intent)`; `intent` is the remote model's word (the CME's `reply_intent`), None for a phrase or the code's reading.

The vocabulary: `decline`, `accept`, `values`, `free` (`parameters.read`'s) and `wait`, `misunderstood`, `different`, `replace`. `parameter_clarification.advance(live=True)` (a caller that serves a user) reads with the phrases and the remote function and returns the `Outcome` kinds `declined`, `wait`, `misunderstood`, `different` (new) besides `complete`, `ask_again`, `exhausted`; a `wait` or any of the new three returns BEFORE the loop is touched, so a wait keeps the question and counts nothing. What the callers do:

| Reply | Execution command (`parameter_reply.read_reply` -> action) | Plan part at the gate (`resolution._take_values`) | Running part (`WEC._resume_parameter_loop_unlistened`) |
|---|---|---|---|
| `name: value` lines, values | `merge`, no remote read | values kept, the next question or the review | the part resumes with them |
| a plain value | one remote read, then `merge` | one remote read, then values (or the extractor) | same |
| wait | `wait` (the question stays; its text, unchanged) | the record is put back unchanged; the answer is "OK, take your time." and the question | same, through `_put_part_loop` |
| `you misunderstood` (phrase or remote) | `run` `you_misunderstood` (the CME's listing question) | back to the which-command question of that part: `resolution._ask_misunderstood` (the intent loop's misunderstanding question, `command_choices(core=False)`), the part unbound until picked | the part ends as not run, "you said it was not the command you meant" (see correction 3) |
| a different request | `different` (the reply is routed as a request) | the part is dropped, reason `DIFFERENT_REQUEST_REASON`; the reply is NOT run as a request | the part ends as not run with that reason |
| `cancel`, `abort`, `stop`, `never mind` | `run` `abort` | the part is dropped ("it was declined") | the part ends as not run ("it was declined") |
| unreadable | `merge`; the extraction asks again | asked again, `RETRY`, counted | same |

Corrections to this note.

1. 3.1's step 7 row said the remote read and the phrases move into `parameters.read`. They did not: `parameters.read` stays code-only (the design constraint of this step), and the two live in `parameters/reply.py` behind functions the caller passes in (`phrases`, `remote`), with `phrase_of` and `remote_intent` as the usual ones, so the model stack is imported only when a call is made.
2. The remote read is NOT made for a reply that read as `name: value` lines, but it is for a lone value taken from the whole reply and for comma parts (a reply like "show me the users" is a lone value to the code). "Did not read as values" was too weak a gate for the single-field question: it would have disabled wait and different-command there. Where the deterministic path used to read a named reply with the remote model it does not any more.
3. A running part has no `PlanResolution` (3.9, correction 1: while it runs `_plan_resolution` is None), so "back to the which-command question" is not available there; it ends as not run, which is what the gate's `none` would also do to a part nothing serves. Making it ask again needs a one-part resolution built for the running part; it was not built.
4. `cancel` on the deterministic path now ends the command (`abort`) with no key; before, a reply of `cancel` was merged, read as a decline by `parameters.read`, and the question was asked again unless the remote model called it an abort or it was one of `abort`'s plain utterances. `stop` and `never mind` behave the same.
5. `what_can_i_do` is not a fixed phrase of this reader: the deterministic reader filtered it out before (it keeps only `abort` and `you_misunderstood`), and the frozen `keep_pending` field of the `run` action is still unused.
6. The decision step's reply (`WEC._read_decision_reply`, a plan step missing one plain value) still read with `parameter_clarification.read_reply` (the code rule) at step 7: it is a step of a running plan, not a value question of a part, and an unreadable reply there has its own ladder. It did not understand wait / misunderstood / a different request. Step 8 moved it onto the one reader (3.12).
7. The text and the word "cancel": the plan path's question says `cancel`, the CME's says `abort` (`no_agent_mode.DECLINE_REPLY`, `ErrorCorrection/abort`); both words are accepted on both paths. One word for both would need the CME's question text, which is in `wildcard.py`'s training signature; proposal: say "cancel or abort" in `no_agent_mode.DECLINE_REPLY` (it is outside `_commands/`), if the owner wants it.

### 3.12 Step 8 as built (DONE), and what it corrected

What exists now.

- A pending review is the turn's `Waiting` record of kind `review` (owner `plan`) in `WEC._waiting`, payload `{"record_id": <the approval store's record>, "slot": <nested Waiting or None>, "gate": <bool>}`. It is put by `WEC._put_review_record` when the framework call (`run_framework_planning_command`) suspends on its review, so the sub-task gate's review and a stand-alone `review_plan` / `review_subtasks` call both have one. The store keeps its record (owner decision); the record points at it and holds nothing of it. `gate` says the turn's sub-task gate opened the review (its accept runs the turn's sub-tasks, `_accept_subtask_review`); a stand-alone call's accept ends in `_finish_framework_call`. The record is dropped in one function, `_drop_review_record` (accept: through `_finish_framework_call`; decline or expiry: `_end_subtask_review_unrun`; a stand-alone call finishing), and by `_reset_agent_suspension` (cancel, `//cancel`, a conversation change).
- The review's slot question is a nested `Waiting` of kind `parameters` (`after` `edit_slot`) in `payload["slot"]`. The plan CME's port is `waiting.SlotPort(wec)`: `get` reads the slot of the turn's review record (None with no review out), `put` sets it and raises with no review out (a defect: the slot is only asked while the review is), `clear` drops only the slot. `Waiting.to_state` / `from_state` nest it; the blob carries it under `waiting.payload.slot` with the plan CME's scratch keys `command` and `command_name` in its payload, and `_apply_waiting` / `_apply_question` put it back. `WEC._review_slot`, the keys `review_slot` and `subtask_review_open`, `PlanFrame.subtask_review_open`, `PlanFrame.framework_entry` and `OwnerPort`'s `attr` / `rounds` arguments are deleted. `SCHEMA_VERSION` is 6.
- `framework_entry` was read in two places, both now stateless: the frame loop asks whether the one nested frame is a framework command (`_is_framework_command(frame.command_name)`), then whether the review record is the gate's (`_gate_review_out`); `_can_ask_decision` asks whether any nested frame is a framework command.
- `_turn_subtask_review` keeps its six outcomes. `"open"` is not stored: the turn span's `subtask_review` attribute (a client-visible value pinned by `tests/test_round1_gate.py`) is `_subtask_review_mark()`, `open` while the gate's review record is out.
- A paused decision step is the record of kind `decision` (owner `plan`), payload `{"decision": <the dict that was phase_run["decision"]>}`, `question` the question as the user saw it. `_pause_for_decision` puts it (the first question, an asked-again one and a wait), `_read_decision_reply` drops it on go and stop, `_run_phase` reads it when the phase resumes, `_resume_agent_message` tests its kind. `phase_run` no longer carries `decision` (so `skill_phase_run` does not either). `waiting.derive` reads the approval store for kind `action` only (step 9 folds that in) and `_waiting` for everything else; two questions out at once can now only be an approval and the record.
- The decision step's reply is read by `parameters.reply.read_decision`, the one reader with the decision's own answer rule plugged in (`read_reply(answers=)`, after cancel / accept and before the phrases, so a yes or no is never a phrase and never costs a model call):

| Reply | Conditional step | Command step missing one plain value |
|---|---|---|
| `yes` / `y` / `true`, `no` / `n` / `false` | the answer (run the inner step, or skip it); no remote read | `yes` is the value like any text (a boolean field's value too); a `no` is no value for a required field (unreadable); no remote read for either |
| `name: value` line of the field | no answer form: read as any other text | the value; no remote read |
| any other text | one remote read when a key is set; with none, unreadable | the value, after one remote read when a key is set |
| a request for a moment (remote read `wait`) | the question stays (`"OK, take your time."` and the question, built once: a second wait does not stack it), `asked` unchanged | same |
| `cancel`, `abort`, `stop`, `never mind` (code); `abort`'s plain utterances, or the remote read `abort` | the command stops at the step: reason "declined: the client chose not to decide this step; carry it out as its text says" | same |
| "you misunderstood" (phrase or remote read) | the command stops at the step: reason "declined: you said this is not what you meant, so the step was not decided; ..." | same |
| a different request (remote read) | the command stops at the step: reason "declined: the reply asked for something else, ... so it was not run as a new request (send it as its own message); ..." | same |
| unreadable (no usable answer) | asked once more (`asked` 2), then the stop "the reply could not be read as an answer, twice" | same (the value must also pass the step's own checks) |

"You misunderstood" and a different request end the command and do not return to a which-command question or run the reply as a request: a decision step belongs to a plan that is already running (its command is settled, nothing of it can be re-bound from the middle), and a reply answers its question and nothing else (AGENTS.md). The asking rule for a decision step is unchanged: asked again once after a reply that gave nothing usable (`MAX_DECISION_ASKS` = `MAX_NO_PROGRESS` = 2 asks in all), cancel ends it, a wait counts nothing; the step has no `ParameterLoop`.

Corrections to this note.

1. 3.5 and section 1 call `framework_entry` and `subtask_review_open` "the same fact". They are not: they differ for a stand-alone framework call (`within_turn=False`, `review_plan`; the tests drive it), where the first was set and the second was not. After a restore the old code made them equal, so a restored stand-alone review finished through the wrong branch. Now the framework call is read from the frame and the gate from the record's `gate` field, which the note's payload did not have.
2. 3.5 says `"open"` is deleted. It is a value of a client-visible span attribute, so the span derives it from the record instead.
3. 3.1 puts the review record at `_open_review`. The record is put where the framework call suspends (`run_framework_planning_command`), because a stand-alone review has no `_open_review`.
4. 3.2 says the slot port raises when the record is not a review. Built as: `put` raises, `get` and `clear` are quiet (they are called on every review reply and on every `end_command_processing` of the plan CME, review or not). A test that drives the review engine on the store directly (no framework call) must put the review record itself (`_wire_plan` in `tests/test_plan_reply_policy.py`); a production review always has one.
5. 3.1 types the decision payload as the dict itself; built as `{"decision": dict}` so `_pending_prompt_kind`, `phase_step_payload` and `waiting.derive` readers are unchanged.
6. 3.11 correction 6 and section 7 said the decision step reads by the code rule only: it no longer does (the table above). A decision reply now costs one remote read when a key is set; `tests/test_no_agent_decision_route.py` started a server whose env files set the key and asserted the key-less behaviour, so its helper now removes the key as it removes `LLM_AGENT`.
7. The note's step 8 row did not list the slot question or `_review_slot`; they were step 4's temporary home and are gone here.
8. `_part_loop` stays (item 5 of the task, below). A running part's "you misunderstood" stays "not run" (item 4, below).

What was looked at again and left.

- `WEC._part_loop` (and the blob key `part_loop`). It carries two facts the reply's re-run needs after the record has left the slot: the names the last question listed (`asked`) and the count of replies in a row with no progress (`stalled`), and the replies and given values the re-run appends to the part's text (`tail`). Every home found for it is the same thing under another name: the turn frame (`PlanFrame` is saved as top-level keys, so it would be a field and the key would stay); a part's plan item (`review_proposals.SubTaskInput` is the reviewed plan's schema; progress is not a property of an input); a resolution for the running part (the old `run_time(loop)`, removed at step 5); the record held in the slot while the part re-runs (rejected at 3.9: a second suspension in the part would be two questions out, and `derive` / `has_open_command` would report one while the part runs). To delete it, one of these would have to be built: (a) a second, non-question slot in the turn that outlives the record (it is `_part_loop` renamed), or (b) a `reopen` that needs no history, which means ending the part's second refusal without the progress rule (a change to the asking rule of 3.6, which is confirmed). Neither is smaller than what is there.
- A running part's "you misunderstood" (3.11 correction 3). Going back to the which-command question needs a one-part resolution built for the running part (a `PlanResolution` with one part and the part's item, `resolution._ask_misunderstood`, which builds without the frozen file), and then somewhere for the PICK to go: `_resume_plan_resolution` ends in `_after_resolution`, which drafts, reviews and runs the WHOLE turn from its sub-tasks, not the one part that is running while earlier parts have already run. Resuming only that part after a pick (bind the command, ask its values, re-run it from its start through `_run_loop_part`, then the later parts) is a second continuation beside `resolve_next` and `resume_part` on the intent record (`after` is a field of the `parameters` record only), and the fast path has no plan item to bind. That is a new idiom, so it is not built; the part ends as not run with the reason "you said it was not the command you meant" (pinned by `tests/test_one_reply_reader.py::test_part_you_misunderstood_ends_the_part_as_not_run_naming_why`, which also checks the not-run entry).

What step 9 should know.

- `_awaiting_user` is still set by the framework call (`run_framework_planning_command`), by `_settle_agent_result` and by `_resume_approval_direct`, and read about forty times; a review and a decision record both ride on the same agent suspension (`export_suspended` is non-None), so the `agent` kind is "`_awaiting_user` and no record", as `derive` already reads it.
- `_resume_agent_message` still tests three different sources in order: `_resolution_pending` (the record), the approval store (`awaiting.suspended_agent`: a review AND an action approval), then the decision record. With the review on the record, the approval branch could read `kind in (review, approval)` of the one record; `approval_resumed` and `was_suspended` (WEC 2217-2221, 5672, 5712, 5750) still call `approval_binding.active_record`.
- The review's accept, decline and expiry are decided by the store's record state (`resolve_reply`), while the WEC's record is dropped by `_accept_subtask_review`, `_end_subtask_review_unrun` and `_finish_framework_call`: if step 9 folds the approval into the record, expiry observed on the store (`approval_binding.on_new_turn`, `time_expired`) must end the record too; today a stale review record can only exist until the next `resolve_reply` or `_reset_agent_suspension`.
- A parked review snapshot carries its record (`waiting` is in `_PARKED_PENDING_KEYS`) next to the parked store records (`park_reviews`); `unpark_reviews` runs before `apply_serialized_state`, so the record's `record_id` finds its record. If the approval kind moves into the record, park one thing instead of two.
- The saved keys left to delete at step 9: `awaiting_user`, `suspended_user_message`, `pending_clarification_request`, `assistant_command` (3.8 correction 6), and `part_loop` unless it is replaced.

### 3.13 Step 9 as built (DONE), and what it corrected

What exists now.

- The agent's own question is the record of kind `agent` (owner `plan`), `question` the clarification, payload `{"message": <the turn's user message>}` (`WEC._ask_agent_question`, called by `_settle_agent_result` and by the re-suspension at the end of `_resume_agent_message`). A question that rides on the same suspension (a review, a decision step, a plan question, an approval) is already the record and stays; the CME's open command, left by a refused call in an agent session, gives way to it (it is dropped: its stall count restarts, which nothing pinned). `WEC._suspended_message()` reads the message back (the agent record's, else `_turn_user_message`; the three places that read `_suspended_user_message` now read it).
- `WEC._awaiting_user` is a read-only property: `held is not None and held.rides_suspension` (`Waiting.rides_suspension`: false for the CME's open command; for an approval, `suspended_agent` or `asked_again`, the mark `_resume_approval_direct` puts when a reply to an approval raised outside the agent's loop was not understood, which is the one case the old flag was set outside a suspension; true for every other kind). The flag, `_suspended_user_message` and `_pending_clarification_request` are gone; assigning `_awaiting_user` raises.
- An approval is the record of kind `approval`, payload `{"record_id", "suspended_agent"}`, `question` the rendering. `approval_binding.gate` puts it when it asks (`WEC._put_approval_record`), through the store's owner: `ReviewStore.owner_ref`, a weak reference set by `WEC.bind_app_workflow` (`review_engine.question_owner(store, host)`, the host only when no context bound the store). The host cannot be used: `process_action` (the REST and MCP direct action) runs with no trace host, and `CommandExecutor.perform_action` called directly has none.
- The store ends a record by itself in one place: `review_engine.transition` calls the owner's `_store_record_ended` when a record leaves the active states (accepted, declined, expired, invalidated, cancelled), which drops the turn's record that points at it (`_drop_store_record`, which also clears the review slot's CME scratch). `_end_subtask_review_unrun` and `_finish_framework_call` no longer drop it. A reply that ends the record is read after it ended, so `_resume_agent_message` reads what it needs first (the message, the question, the store's record, `gate_review`).
- `waiting.derive`, `waiting.STRICT` and the autouse `strict_waiting` fixture are deleted. `WEC.waiting` is `_waiting`. Two questions cannot be out: one slot, `_put_record` replaces (known surprise 2, an approval beside the CME's open command, is now "the approval replaces it"). An approval on the store with no record: none in a session (queued approvals are not questions and never were in `derive`); it can exist only on a store no context bound (a unit test of the engine on a bare store), so no lookup remains.
- `waiting.step(record, event, facts)` (`Event` MESSAGE, CANCEL, RESTORE; `Facts` `agent_path`, `assistant_round`, `manage`, `cancel_pending`; `Move` `action`, `text`, `begin`, `assistant`, `reader`, `label`, `record`) is the table of section 3.4 as code. `WEC._route` runs it for a message and applies the "/" round; `_execute_message` begins a turn when `move.begin` and takes the action (`ACT_MANAGE`, `ACT_APPROVAL`, `ACT_RESUME_AGENT`, `ACT_START_AGENT`, else the deterministic path); `ChatSession._pending_cancel_reply` is the CANCEL event; `_apply_waiting` takes the RESTORE event's record. The kind table (`KindSpec.reader`, `cme_reader`) names where a reply goes, and `_resume_agent_message` and `_pending_prompt_kind` (its payload builders moved to `fastworkflow/prompt_payloads.py`) read it. What stayed outside `step`, and why: the readers (`resolution.take_reply`, `approval_binding.resolve_reply`, `_read_decision_reply`, `_post_ask_user_response`, the CME's `wildcard`); the branches inside `_resume_agent_message` that run the agent after the reply (they are what the reply does, not which reader); the "can anyone answer" predicates (`waiting.can_ask`, `can_review`, section 3.3: asked before a question exists, not when a message arrives); `process_action` and `check_use_plan` (they refuse when any question is out: `_question_is_out`, one test).
- Parking: `_pending_snapshot` takes the store's record out of the store into the turn's record (`payload["store_record"]`, `approval_binding.take_record`) and `_restore_parked_pending` puts it back (`restore_record`): one thing is parked. `park_reviews` / `unpark_reviews` and the snapshot key `reviews` are gone; an approval is cancelled, never parked, as before.
- Saved state, `SCHEMA_VERSION` 7: gone are `awaiting_user`, `suspended_user_message`, `pending_clarification_request` and `assistant_command` (the "/" round is `waiting.assistant`, restored before the record so the port keeps it). `part_loop` stays (3.12).

Corrections to this note.

1. 3.5 said `_awaiting_user` is "a record is out and its kind is not `intent` / `parameters` on the CME". Built as `rides_suspension`: the plan path's `intent` and `parameters` records do ride on the suspension; an approval raised outside the agent's loop does not, until it is asked again.
2. 3.1's `agent` payload had `clarification` and `suspended_message`. The question is `Waiting.question`; the message is the one payload key.
3. 3.4 had the CME's row a plain "not reachable" for the new-turn column and put `step` in front of the CME's own reader. The CME's reader still runs behind the deterministic path (`reader` "cme" is named, not called by `step`).
4. The note said an approval folded into the record needs no store lookup. It needs the store to know its owner (`owner_ref`), because the host of a gate call is not always the context.
5. `step` does not carry the `Move.next` record the note's signature listed: the readers put the next record themselves (`take_reply`, `_put_record`), and a RESTORE returns the record. The transitions that put a next record live where the work is, as 3.12 left them.
6. The review record keeps its `gate` field (3.12) and gains `suspended_agent` (copied from the store's record, as an approval's is): a review opened by the framework call suspends the agent, one opened by `review_engine.open_review` with the default `suspend=False` (a caller outside the agent's loop, the engine's own tests) does not, and its reply is the approval path's (`ACT_APPROVAL`). The record is put by `open_review` itself (`WEC._put_review_record(record)`, through the store's owner, as `gate` puts an approval's); `run_framework_planning_command` only sets its `gate` field. Both kinds of store-backed record ride the suspension iff `suspended_agent` or `asked_again`.
7. A paused decision step's record question now includes the "OK, take your time." line of a wait (it is what the user saw, and what `_pending_clarification_request` held), not the bare question.

For the cleanup step (10).

- The frozen dispatcher's `stop` action (`ACTION_STOP`, `Reply(action="stop")` in `parameter_reply`, handled in `wildcard._dispatch_reply`): no reader returns it (3.10 correction 1); deleting it edits `wildcard.py` (a retrain), so it stays unless a retrain is wanted anyway.
- `WEC._part_loop` and the blob key `part_loop` (3.12).
- `waiting.Holder` / `Port.clear` / `CME_KINDS` naming: `CME_KINDS` is the kinds a CME port may drop, not an owner test (3.9); a rename would touch the built-in commands' imports.
- `WEC._route_message` is kept for `tests/test_planning_commands.py` (it calls it); `ctx._route` is the one the engine uses.
- `approval_binding.active_reviews` has no caller outside the tests.
- `tests/parameter_pipeline_corpus.py` and `tests/parameter_ask_read_corpus.py` still put the removed `policy` and `pending` keys of `ParameterLoop` back when they record (3.10 correction 5).

## 4. What does not change, and what does

Does not change: `question_kind` values (`agent`, `approval`, `review`, `plan_choice`, `planning_parameter`,
`phase_step`, `parameter_request`, `unknown`); the artifacts (`plan_choice`, `parameter_request`, `phase_step`,
`approval`, `review`, `fw_not_run`, `fw_sub_tasks`, `aborted.waiting_for` with `intent` / `parameters`); the readers and
their words (`abort`, `you misunderstood`, `none`, `cancel`, `no`, `yes`); span attributes; `intent_learning` rows; the
approval store format; `ParameterLoop.to_state` minus `policy` and `pending` (both removed at step 6); the intent record dict; the REST and MCP
surfaces.

Changes:

| Change | Kind | Notes |
|---|---|---|
| Saved state: keys `awaiting_user`, `suspended_user_message`, `pending_clarification_request`, `subtask_review_open`, `assistant_command`, the `cme` and `plan_cme` sections and `plan_resolution`'s pending fields are replaced by one `waiting`; `skill_phase_run.decision` leaves `phase_run` | format | `SCHEMA_VERSION` bump; old blobs refused (owner ruling, no shim) |
| "/" message during any open question is read as its reply | behaviour | today it runs alongside (3.7) |
| `//cancel` cancels every kind | behaviour | no visible change in the CLI, where those questions are not asked |
| Human deterministic session stops after two unusable replies in a row | behaviour | confirmed (3.6) |
| Agent model set: a second question is possible; `cancel` ends the part; replies read by the shared extractor | behaviour | confirmed (3.6) |
| A plan part's value question reads a reply as an execution command's does: the remote model's read (wait, different command) and the `abort` / `you misunderstood` phrases | behaviour | confirmed (step 7) |
| `_turn_subtask_review` loses "open" (the span's `subtask_review` still says `open`, derived from the review record); five predicates become two | internal | tests asserting the string `"open"` change |
| A decision step's reply is read by the one reader: a request for a moment keeps the question; cancel, abort, "you misunderstood" and a different request end the command as declined, saying why | behaviour | step 8 (3.12) |
| Tests that read the old flags or CME keys read `waiting` | tests | expectations only |

## 5. Migration

Every step leaves the suite green. Sizes are lines touched, tests included. The full migration (steps 1 to 10) is the
goal. Steps 1 to 3 are a checkpoint only: the work is not finished at it, and nothing in this plan stops there.

| Step | What | Files | Size | `wildcard.py` |
|---|---|---|---|---|
| 0 | Done in the tree: `store` / `clear` helpers, `waiting_for` raises (fix-fpyb.4). If its `wildcard.py` edit has not shipped, it joins step 4's batch | | | batch with 4 |
| 1 (DONE) | `waiting.py`: the type and the kind table (`question_kind`, `cancel_text`; `Port` and `Holder` are step 4's); a read-only `WEC.waiting` DERIVED from today's flags (two records at once: the tests fail, a live session logs one error and acts on the first in the old order; `derive(strict=)`, `waiting.STRICT`, set by `tests/conftest.py`); replace Chain C and Chain E with the table. Fixes fix-fpyb.6. Tests: a `//cancel` test per kind on a real session | `waiting.py` (new), WEC, `chat_session.py`, `tests/test_waiting_kinds.py` | ~320 | no |
| 2 (DONE) | fix-fpyb.7: one `route(message)` replaces A5 (`WEC._route_message`: a question out, any kind, is the reply with the leading "/" removed, management messages excepted; nothing out, a "/" message sets the assistant flag); `process_action` / `use_plan` share one refusal text (`AWAITING_ANSWER_REFUSAL_MESSAGE`); pinned by a test per kind (`tests/test_slash_message_during_question.py`) | WEC, one test file | ~90 | no |
| 3 (DONE) | Two "can anyone answer" predicates: `waiting.can_ask`, `waiting.can_review`; each call site keeps its own extra conditions (`_refusal_may_ask_attended` adds "not asked yet", `_planning_command_refusal_may_ask` adds "nothing ran", `_can_ask_decision` the agent path and the rest). `resolution.py` needed no edit | `waiting.py`, `workflow_agent.py`, WEC | ~110 | no |
| 4 (DONE) | The deterministic path on the WEC record: `WEC._waiting` holds `intent` / `parameters` for the CME; the turn port and the review-slot port; helpers read the port; `end_command_processing` clears the port; the `cme` / `plan_cme` blob sections go; `has_open_command`, `_clear_cme_continuation`, parking; the "/" flag moves (`Waiting.assistant`, `end_round`, the eleven pops). **The one `wildcard.py` batch, one retrain (as built it also moves the whole reply-to-a-parameter-question block out of `wildcard.py`, so step 7 and step 10 need no `wildcard.py` edit; section 3.8)** | `waiting.py`, WEC, `intent_loop.py`, `parameter_clarification.py`, `parameter_extraction.py`, `parameter_reply.py` (new), `workflow.py`, `command_executor.py`, `session_state_store.py`, `wildcard.py`, tests | ~420 | yes, once |
| 5 (DONE) | The plan path on the same record: `PlanResolution` loses its pending fields; the plan's which-command and values questions are `intent` / `parameters` with `after`; `part_parameters` and `run_time` go | `resolution.py`, WEC, `parameter_clarification.py`, tests | ~300 | no |
| 6 (DONE) | The asking rule (3.6), behind the measurement: delete `ParameterLoop.policy`, `POLICY_*`, `fallback_runs`, `_parameter_question_asked`; one hand-over place; `stalled` ends the CME stage | `parameter_clarification.py`, `resolution.py`, WEC, `parameter_extraction.py`, `workflow_agent.py`, the pinned tests | ~280 | no |
| 7 (DONE) | One reply reader (CONFIRMED), built as section 3.11 records: `parameters/reply.py` (`read_reply`, the fixed phrases and the remote read passed in as functions), `parameter_clarification.advance(live=)`, `resolution._take_values` and WEC `_resume_parameter_loop_unlistened` act on the new outcomes, `parameter_reply.read_reply` is the adapter. NOTE: this note first said the remote read and the phrases would move INTO `parameters.read`; they did not (3.11, correction 1). | `parameters/reply.py` (new), `parameters/read.py`, `parameter_clarification.py`, `resolution.py`, WEC, `intent_loop.py`, `parameter_reply.py`, tests | ~330 | no (done at step 4) |
| 8 (DONE) | Review and decision, built as section 3.12 records: delete `subtask_review_open`, `framework_entry`, `"open"` (the span derives it); `phase_run["decision"]` into the record; the review's slot nested in the review record (`SlotPort`; `_review_slot` deleted); the decision step's reply on the one reader (`parameters.reply.read_decision`); `SCHEMA_VERSION` 6. `_part_loop` and a running part's "you misunderstood" left as they are (3.12) | `waiting.py`, `parameters/reply.py`, WEC, `plan_frame.py`, `session_state_store.py`, review and decision tests | ~420 | no |
| 9 (DONE) | Agent and approval kinds; `_awaiting_user` a property; chains A, B, C and E are `waiting.step` and the kind table; parking moves one record; `SCHEMA_VERSION` 7, old keys deleted; `derive` and `STRICT` deleted. Built as section 3.13 records | `waiting.py`, `prompt_payloads.py` (new), WEC, `approval_binding.py`, `review_engine.py`, `chat_session.py`, `session_state_store.py`, tests | ~500 | no |
| 10 | Delete leftovers (fix-kc83), one sentence each in `AGENTS.md` and `README.md`; close fix-fpyb.8, .6, .7 and the epic | docs, tests | ~60 | no |

Total about 2,400 lines, about 760 of them tests. That is more than the bead's 600 to 800 because the bead did not count
tests, the CME side, the asking rule or the reader. Steps 4 and 7 are the only ones that change a built-in file under
`_commands/` (`wildcard.py`, so two retrains; see the risk in section 6); steps 6 and 7 are the ones that change what a
user or an agent sees beyond "/" and `//cancel`. Step 7's tests that pin today's behaviour: `test_jev_missing_value_reply.py`,
`test_misunderstood_during_parameter_question.py`, `test_clarification_reply_merge.py`, `test_param_retry_round_ordinal.py`,
`test_ask_user_reply_intent.py` and `test_parameter_loop.py`; their expectations change, none is deleted. Steps 5 to 9
each run the plan, review, restore and parking test files (`test_plan_resolution*`, `test_cancel_pending_review`,
`test_pending_state_*`, `test_conversation_pending_isolation`, `test_one_pending_question`); the full suite is run once,
at step 10, alone on the machine.

## 6. Open risks

Decided and therefore not listed here: the asking rule (3.6), "/" read as the reply (3.7), the reply reader's move (step
7), the pointer to the approval store (3.1), `command` and `command_name` stay CME scratch keys (section 7), and the full
migration as the goal.

1. RESOLVED at step 4: one retrain, `wildcard.py` is not edited again (section 3.8). It was: two retrains. `wildcard.py` is edited at step 4 (the port, about 6 lines) and again at step 7 (the alias and Jev block
   leaves it). Each edit changes a training-signature input of the CME. To make it one, step 7's `wildcard.py` deletion
   can wait for step 10's batch, with the new reader in place and the old block unreachable in between; at the cost of
   two readers coexisting for three steps. Decide at step 4 which way the batch goes.
2. The asking rule's cost (step 6). Measured against the baseline in 3.6 (123 questions and 2 turns with more than one
   question with no agent model; 126 questions with one). Held back or reverted alone if questions per turn or not-run
   endings rise by more than the owner's bar.
3. (Built at step 7, 3.11.) The reader's move puts the remote model's read on a path that had none: a plan part's value question with the
   remote model set makes one more model call per reply. It is made only when the key is set and its result is the same
   threshold-gated read as on the CME; measure the extra calls per reply in the same round as step 6.
4. Size and risk: steps 8 and 9 edit WEC, whose health is 1.0/10 and which has had six bug fixes in six months. Each gets a
   full review before it merges. The suite is run once, alone on the machine, at step 10.
5. RESOLVED at step 9: the check that at most one question is out read six sources, and two at once (known surprise 2: an
   approval and an open CME command) was reported and the first in the old order acted on. The record is one slot and
   `_put_record` replaces, so two cannot exist; `waiting.derive`, `waiting.STRICT` and the test fixture that set it are
   deleted (section 3.13).

## 7. Differences between the paths that remain after the migration

| Difference | Why it stays |
|---|---|
| The CME can also pick the framework's core commands (`core`; `abort`, `what_can_i_do` ...); a plan cannot | they are the framework's own commands, answerable only where there is no plan |
| One candidate: the plan path binds it and the turn's one review confirms; the CME asks "Did you mean the command 'X'?" | the deterministic path has no review |
| Two unreadable replies to which-command: the plan path drops the part (a not-run entry); the CME ends with the `what_can_i_do` listing | a plan turn has later parts to go on to; the CME has nothing else to do |
| `after`: resolve the next part, resume the running part, run the command, or apply the edit | where the answer goes next is inherent to the caller |
| An execution command's question carries its validators' messages and suggestions and opens with the shared sentence; a plan's is built from `argument_gaps` | different checks; one builder (`parameters.ask`) and one `Extraction` contract |
| Confirmation mode (a pinned plan shows every value, `yes` runs) | only a planning command can be pinned; at stop it always ends as not run |
| The audience of the wording (an agent reads the unchanged agent text, a person one plain sentence, `ask.audience_of`) | who reads it, not what is asked |
| The turn gate with parts, one review and the fast path exists on the plan path only | a "/" or deterministic message is one command |
| `command` and `command_name` remain CME scratch keys, saved in the `parameters` payload | keeps the `wildcard.py` edit small (decided) |
| A reply that says the command was misunderstood: the plan gate asks which command again (every command the plan can run); a running part ends as not run; the CME runs `you_misunderstood` (its listing, every command including navigation) | a plan has no `you_misunderstood` command; a running part has no resolution to return to (3.11, correction 3) |
| A different request: the CME routes the reply as a new request; a plan part is set aside as not run and the reply is not run | a reply answers its question and nothing else (AGENTS.md); the deterministic path's message IS a request |
| The remote read is skipped in an agent session on the CME (the agent's own text is the reply); a plan part's reply (a person's) is always read when a key is set | who writes the reply |
| A decision step's "you misunderstood" and a different request end the command at that step as declined (no which-command question, the reply is not run); a part's value question at the gate asks which command again | a decision step belongs to a plan that is already running: its command is settled and nothing can be re-bound from the middle of it (3.12) |
| A running part's "you misunderstood" ends the part as not run | the which-command question for a running part needs a continuation for a pick that does not exist (3.12) |
| `WEC._part_loop` is a second home of the last question's loop while a part runs | the record leaves the slot at the reply and the re-run needs the loop (3.10 correction 2, 3.12) |
| `cancel` on the plan path ends the part; on the CME ends the command | the same word, the same meaning; what ends differs |
