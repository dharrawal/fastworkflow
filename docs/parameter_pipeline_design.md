# One parameter pipeline for planning and execution commands (fix-qoyz)

Status: done. The staged plan (shared helpers, one validation driver, one guard module, one question builder and
reader, one fill and pipeline, the behaviour decisions, the tidy) is complete. This note says what exists and which rules
hold now. The code is the authority; each module's docstring has the detail.

## 1. The package

`fastworkflow/parameters/` is the only home of the rules. No module imports a model stack at import time.

| Module | What it is |
|---|---|
| `rules.py` | Text helpers both guards share: `clean_str`, `fold`, `written` (at word boundaries), `written_numbers`, `placeholder_like` (a whole template token), `pattern_refusal` (the one meaning of `Field(pattern=...)`), `pattern_sentence`, `field_extra`. |
| `validate.py` | `validate`: the one driver that runs an `Input`'s field checks and model validators and returns a `Refusal` value (`refused`, `plain`, `rule`, `raised`). The typed planning call (`planning_commands.check_input`) and the execution turn (`InputForParamExtraction._check_input_rules`) word its outcome. |
| `check.py` | Every guard rule once. `check_plan` is the ONE guard of a marked field on both kinds, and a planning command's guard of every argument (stops at the first refusal); `check_execution` is an execution command's screen of what needs no knowledge of the request (collects every refusal). `Gap` is one refused or missing argument as data (name, state, code, why, value, message). |
| `fill.py` | `fill`: the one call of the model extractor, with an optional note after the text. `fill_from_agent_text`: the four forms an agent's own command text is read in code (xml tags, `name=value`, one guarded positional value, the bare command name). |
| `pipeline.py` | `extract`: fill, guard, settle, validate, and at most one more fill. |
| `ask.py` | `word`: the one builder of every question about a missing or refused value; `audience_of` says who reads it. |
| `read.py` | `read`: the one reader of a reply to such a question. |

Callers: `ParameterExtraction._llm_extract` (the CME's execution path), `parameter_extraction.extract_arguments` (the shared
extractor outside the CME, used by the draft of the plan review), `planning_commands.typed_planning_call` and
`reextracted_arguments`. `planning_commands.check_arguments`, `utils.argument_checks` (kept as a thin layer: the
`ExtractionCheck` form of `check_execution`, `checks_enabled`, the retry notes), `parameter_clarification.read_reply` and the
CME's `_extract_and_merge_missing_parameters` call the package.

## 2. The rules

**Guards (`check.py`).** There is no switch between the two kinds.
- A value the extractor did not give (a default, `NOT_FOUND`, the unknown marker, an empty value, the not-given number) is not
  guarded. Guards read only values the extractor gave.
- The extractor's value of a `str` or `int` stands by default (bd fix-qjne.21). Only a field marked `copied_from_command_text`
  (or `record_named_in_command_text`) is checked against the request, by ONE guard on both kinds (`check_plan`; an execution
  command's marked fields reach it through `planning_commands.option_command` in `validate_parameters`, which runs it on
  every value given, also when another value is missing; bd fix-ftdl.6.3): a `str` must be written in it at word boundaries
  (the value's first and last characters do not continue into a letter or digit of the request: "dun" is not written in
  "dune"; bd fix-ftdl.6.6), up to case and Unicode dashes and after quotes, brackets and trailing punctuation are removed, and
  takes its spelling; an id the model cut short inside one identifier match is still widened to it, and a record name cut
  short is completed by the subject rule; an `int` must be a number the request writes, whether or not it writes any number
  (never replaced by another). A marked value given and not written is refused, required or optional; a marked value not
  given is not checked. Two marked `str` fields may hold one value only when the user writes it as often ("New York, New
  York"), the one duplicate rule (bd fix-qjne.20). Otherwise the fields beyond that count give it up, one at a time, the
  first whose role the request does not write before the value, an optional one first: an OPTIONAL one is left not given
  (its default) with a note, `ask.left_empty_line` ("connected_name left empty: 'X' is written once in your request and is
  the name."), which goes on the check's notes (`argument_notes`), names the field on the extraction span
  (`argument_left_empty`), and is one plain line on the review item (`SubTask.check_notes`, drawn by the chat page's form)
  and before a parameter question; it is not a refusal (no second fill, no "Cannot run as shown"). A REQUIRED one, or a
  value the user does not write at all, is refused; when the `Input`'s validators, run on the same values (the value
  still in both fields), refuse too, their refusal is the one reported, handled as that path handles a validator's
  refusal (validators first on a tie, `check.validators_first`, bd fix-qjne.38; a validator that raises is no refusal,
  and the duplicate rule's stands). The count is of the user's own request for the turn when the turn
  binds it (`check.user_written`, bound on the plan and agent path to the turn's message; `check.written_times`): a split
  part or an agent's text that repeats a name the user wrote once does not let two fields hold it, and a value the
  user's request does not write (an earlier step's output) is counted in the checked text. A value written only inside
  another marked field's longer value is not counted apart from it (no rule covers it). Unmarked fields are never
  compared with each other: nothing is reset silently. `converted_from_command_text` does not exist.
- `Field(pattern=...)` means one thing on every path (bd fix-ftdl.6.4): a given text value must match the WHOLE pattern
  (`rules.pattern_refusal`, `re.fullmatch`; a compiled pattern keeps its flags), checked after the guard has widened a marked
  id: by the validators' driver (`validate.validate`) on the typed planning call, and by `validate_parameters` on the "/" path
  and the draft, for every value not already open. The refusal names the field, the value, the pattern and the field's first
  example. A pattern neither marks nor exempts a field. A `Literal` field's value is not grounded: the extractor's answer is read with a text `Literal` as plain text, so a value outside it refuses that field only (a `Gap` naming it, with the allowed values; every other field of the answer keeps its value), and a member in another case or spacing takes its declared spelling (`rules.allowed_member`, the one rule for an execution field and a planning argument). An absent `Optional[int]` is not checked.
- Checked on every `str` field, marked or not, because they need no knowledge of the request: a value of only punctuation and
  a value that is ENTIRELY one template token: a variable name in braces or angle brackets, the inner text judged by its
  shape alone (`{name}`, `{{repo_uid}}`, `<name>`, `<ID>`: empty, or letters joined by `_` or `-` in one case, no digit or
  space). `<Dune>`, `<9a1b>` and `<Ann Lee>` are values in brackets and are unwrapped; `{"a": 1}` (any JSON object string),
  `{Ann Lee}` and `{x1}` are values left exactly as written (braces are never unwrapped). ONE function decides (`rules.placeholder_like`, read through
  `signatures.classify_bracketed_value`), and every path reaches it: the execution command's first fill (its check stage),
  `validate_parameters` (the "/" reply merge, the agent's typed call and every code-read form), and a plan's draft and typed
  call (`check_plan`, through its screen and again on the value it uses, such as a classifier's choice). Without brackets, a
  value the request does not write that only echoes the field's own name is refused (`uid` or `item_uid` for `item_uid`,
  one word before the whole name: `your_item_uid`, `desired-tag` for `tag`; `signatures.is_ungrounded_placeholder`). There
  is no list of stand-in words (bd fix-ftdl.6.1): "Adam Smith" or "John Doe" is a value, so is an id such as `SKU_42` or
  `ORDER_123`, and so is a value that merely contains a bracket. A generic stand-in in an unmarked field ("value",
  "your_name") is the remote "was this value given" check's business; in a marked field the strict check refuses it.
  A free-text value that is only a pronoun or only a word of the workflow's vocabulary reads as missing and its question says
  why; a field's description exempts no word from that screen ("One of: a, b" is prose; allowed values are the `Literal` type).
- The remote "was this value given" check (`signatures.fields_not_given`, `FW_JEV_FIELD_GIVEN`, on with `JEV_API_KEY`) asks
  about every filled UNMARKED field and resets it to not given at 0.9 or more sure (measured in bd fix-jd9o.3: a reset was
  right 266 of 267 times); a marked field is not asked about, the strict check decides it (bd fix-ftdl.6.2). The fields it
  reset are the span attribute `fields_reset_not_given`.
- No example check: a field's `examples` are not shown to the extraction model (`create_signature_from_pydantic_model` leaves
  them out; they feed the generated few-shot examples, the "(e.g., ...)" hint after a pattern mismatch and the JSON schema),
  so a value equal to one is the model's own: the extractor's reward step (it once ran another attempt for such a value) and the readers of an agent's own typed text (`fill.from_xml`, `from_key_values`, `from_positional`, which once treated it as not given) no longer compare with it either.
- What the extraction model is told about a field comes from the field's own declarations and is the same on both kinds,
  from one builder (`planning_commands.option_sentences`, read by the extractor and by the planner's typed fill; bd
  fix-ftdl.6.5): the description, then the sentence of a mark ("Copy it exactly as the request writes it." only for a marked
  field), the rule of the `bool` type ("True only when the request asks for it.", or "False only when the request asks
  against it." for a bool that defaults to true), and a declared pattern (`Must match the pattern <regex>`,
  `rules.pattern_sentence`). A `Literal` is shown as its type.
- Advisory, no behaviour: the names of the unmarked `str` and `int` fields whose value the request does not write
  (`check.not_written_fields`) are the span attribute `argument_check_not_in_request`, on `fw.nlu.param_extraction` and on
  `fw.planner.plan` / `fw.planner.replan`, so an author can see which fields may need `copied_from_command_text`.
- Rules that read the workflow (subjects and pronouns, identifier expansion, kind and vocabulary words, classifier options,
  `ignore` words, list items) stay helpers of `planning_commands` called from `check_plan`; an execution field that carries
  an option is checked by `check_plan` through `planning_commands.option_command` (`validate_parameters`).
- A guard refusal always names its argument (`Gap.name`). `validate_parameters` puts it among the missing or invalid fields,
  so it is asked like any refused value.

**One second pass (`pipeline.extract`).** The same on both kinds. The request is read once more, with the refusal after it
(`argument_checks.REFUSED_EXTRACTION_NOTE`), only when a guard or a validator refused a value the model produced, or every
attempt failed to parse into the fields' types; never for a value that is simply missing. At most one second pass in all:
the first refusal of the guard, else of the validators, is the one that is read again, and the new values go through the guard,
its settling and the validators again. When the second pass raises, the first answer stands (logged, no raise) and its
refusal is handled as any refusal. When it is refused again, the second answer's refusal is the one handled. A caller that
passes no `retry_note` (`reextracted_arguments`, which puts the phase's note on the first fill) has no second pass.
A verdict is classed "missing" from structured information (a guard's code, the validator's diagnostics, `Filled.unparsed`),
never from message text. The typed planning call, whose second pass used to propagate an error, now goes on to its typed fill
(`_typed_fallback`) with the first answer.

The second pass answers the refusal and nothing else. Every verdict says which fields it leaves open (`Verdict.open`: the
refused, the reset and the missing ones, read from the guard's refused names, the validator's diagnostics or the plan
call's gaps), and only those are taken from the second answer (`pipeline.answered`); every other field keeps the first
fill's value, merged into the first answer as the model gave it (a snapshot before any stage wrote on it), and the merged
values go through every stage again, so what the guards correct is corrected again. A verdict that names no field (a
refusal of the call as a whole) gets no second pass at all (bd fix-ftdl.9): nobody asked for different values, so the
first answer and its refusal stand. A first answer that did not parse takes the second answer whole. A guard's refusal of
one argument keeps its corrections of the others: `Signature._check_option_arguments` writes the values of the guards run
with every gap left out (`planning_commands.checked_around_gaps`, `argument_gaps`' last run), as it writes them when they all
pass.

**The remote given-check** (`signatures.fields_not_given`) asks about an unmarked filled field only when the request does
not write its value at word boundaries (`_written_in`, the strict check's `rules.written_at`): a model call cannot overrule
a value the text writes.

**Validators.** They see the request plus the replies as `command_text` on both kinds (`ParameterLoop.text`). What a validator
rewrote is kept when it passes and dropped when it refuses. A later check of the values never reads a marked value a
validator wrote: the draft's inputs hold the value the checks accepted (`ExtractedArguments.unvalidated`), and the run's
validators write theirs again. A default written as text ("0" for 0, "false" for False: a review input holds every value as
text) is not a given value (`check.is_given_value`), and a refused draft keeps no default as an input.

**A refusal that names no input is not asked** (a validator's `Refused(message)` with no field, a plain `ValueError`, a raised
exception, a call-level refusal): an execution command ends as a failed output carrying the message and no pending loop
(`refusal_names_no_input`); a plan call goes to the text planner or the agent (`refusal_names_an_input`).

**Questions (`ask.py`).** A person gets one plain sentence for a missing or refused value on both kinds, naming every gap in
one question. The text an agent reads is unchanged. The audience is decided by the VALUE of `run_as_agent`
(`audience_of`); one test of the key's presence is left in the CME's `wildcard.py`, and editing that file forces a CME
retrain.

**Replies (`read.py`).** One rule, for both kinds:
a. `cancel` and its synonyms decline; `yes` and its synonyms accept only a question that showed every parameter (`confirm`).
b. `name: value` or `name=value` lines set the fields the question listed (open ones and ones it showed with a value); a
   one-line reply whose comma parts are all such lines is read as those lines.
c. With one value open, `yes`, `no`, `none` and the like are that value when the field is yes/no or the word is an allowed
   value; otherwise a bare `no`, `none`, `neither`, `n` or `nope` leaves an optional field empty and is unclear for a required
   plain field. `you misunderstood` is read only by its own phrases.
d. With one value open and no name line, the whole reply is the value, commas included, quotes removed.
e. With several open and no name line, the comma parts fill them in the order the question listed only when the count equals
   the number open, none is empty, no part is a `name: value` line, and each part fits its field where code can tell (yes/no,
   allowed values, numbers). Otherwise the reply is unclear and asked again. Accepted limit: prose with a matching comma
   count sent to several plain-text fields still fills them, because no structural test separates it from real multi-word values.
f. A reply never fills a field the question did not list.
g. Anything else in the reply is ignored (the question says so).

**Gap records.** `PlanningCommandInputError.gap` and `.gaps` carry the refused or missing arguments as data
(`parameter_clarification.Extraction`); `argument_gaps` works out every gap of a refused plan call by running the guards again
with each gap's parameter left out (no model call), and nothing reads an argument name or a reason back from message text.
The review line of a sub-task whose required values are missing reads `SubTask.check_missing` (names), not the text of
`SubTask.check`; a refused one reads `SubTask.check_refused` (names) and says how to change each with `change N: <field>=...`.
The check is worked out again from the item's inputs whenever they change (a review edit, a reply to the values question),
with no model call (`review_proposals.rechecked`: `argument_gaps`, then `checked_arguments` when the guards pass and no value
waits on an earlier part), and also when the draft's refusal is of a value the item does not hold, so no edit leaves a
stale refusal on the line. A review edit's field written as a path (`sub_tasks[1].inputs.kind`) names the input of its last
piece; one that names no input of the item's command is asked again with the valid names (`_edited_input_name`).

## 3. What still differs between the two kinds

These are differences of what a caller runs, not of a rule:
- Stages. The CME runs the screens (`check_execution`) before `validate_parameters`, which runs the one guard (`check_plan`)
  on the fields with options; the typed planning call and the draft run `check_plan` inside their validation stage. The retry
  rule is the same.
- Guard shape. `check_plan` stops at its first refusal and `argument_gaps` collects the rest; `check_execution` collects all.
- Typed fill fallback. Only a planning command whose required value is still missing or refused after the second pass gets the
  planner's typed fill (`_typed_fallback`).
- Agent text. The four code forms (`fill_from_agent_text`) apply to an execution command in agent mode.
- Question content. The execution forms carry what only an execution command knows (where a value can be looked up, the
  validators' own messages and suggestions, what was extracted so far, how to leave the question); only a plan chosen by the
  client is presented whole with `confirm`.
- The loop around the reply. One asking rule for both (`parameter_clarification`, step 6 of `pending_state_design.md`): a
  loop counts replies without progress and stops after two in a row, in every session; the CME reads `abort`, `you misunderstood` and Jev's read of the reply
  (wait, different command) before the reader.

## 4. Constraints that remain

- Training-signature inputs. Editing a command file under `fastworkflow/_workflows/*/_commands/` or
  `train/embedding_head.py`, `train/workflow_model.py`, `train/generate_synthetic.py` or `model_pipeline_training.py` forces a
  retrain. Nothing in this package is one.
- Span attributes of `fw.nlu.param_extraction` keep their meaning: `extraction_method`, `deterministic_form`, `retry_round`,
  `retry_round_ordinal`, `placeholder_values`, `defaulted_sentinel_values`, `out_of_range_values`, `command_name_values`,
  `word_screen_values`, `missing_fields`, `invalid_fields`, `validation_hook`, `fields_reset_not_given`,
  `argument_check_refusals`, `argument_check_retry`, `argument_check_refusals_final` (the refusals of the check stage's
  screens, and those that remain after the second pass, if any; a marked field's refusal is `argument_refusal`),
  `argument_check_respelled`, `argument_check_not_in_request`, `argument_refusal`, `argument_notes`, `unbound_value`, `decision_uncertainty`,
  `parameters_valid`. The stages write the same keys.
- The remote reply classifier reads the question text: rewording a parameter question needs a live check that a request for a
  different command still reads as `different_command`.
- Saved sessions from before are not supported (`parameter_loop`, `plan_resolution` and `SubTask.check_missing` are the shapes now).

## 5. Gates

Frozen tables of the behaviour, regenerated on purpose when a rule changes and diffed case by case:
`tests/fixtures/parameter_check_equivalence.json` (guards), `parameter_ask_read_equivalence.json` (questions and replies),
`parameter_pipeline_equivalence.json` (every call site with a scripted model, recording each text sent to it). The live
harness `scripts/experiments/parameter_pipeline_gate.py score --split dev --variant baseline` scores right, missing and made-up
values and extractor calls with a real model.
