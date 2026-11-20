# One parameter pipeline for planning and execution commands (fix-qoyz)

Status: done. The staged plan (shared helpers, one validation driver, one guard module, one question builder and
reader, one fill and pipeline, the behaviour decisions, the tidy) is complete. This note says what exists and which rules
hold now. The code is the authority; each module's docstring has the detail.

## 1. The package

`fastworkflow/parameters/` is the only home of the rules. No module imports a model stack at import time.

| Module | What it is |
|---|---|
| `rules.py` | Text helpers both guards share: `clean_str`, `fold`, `written`, `written_numbers`, `placeholder_like`, `allowed_values`, `allowed_member`, `not_allowed_why`, `field_extra`. |
| `validate.py` | `validate`: the one driver that runs an `Input`'s field checks and model validators and returns a `Refusal` value (`refused`, `plain`, `rule`, `raised`). The typed planning call (`planning_commands.check_input`) and the execution turn (`InputForParamExtraction._check_input_rules`) word its outcome. |
| `check.py` | Every guard rule once. `check_plan` is a planning command's guard (stops at the first refusal); `check_execution` is an execution command's (collects every refusal). `Gap` is one refused or missing argument as data (name, state, code, why, value, message). |
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
- One grounding rule: a `str` or `int` must be written in the request, up to case and Unicode dashes and after quotes,
  brackets and trailing punctuation are removed, unless the field opts out (`converted_from_command_text`, or a `pattern`).
  A value taken from the field's own allowed list needs no grounding.
- An `int` the request does not write is refused, never replaced by the one number the request does write. An absent
  `Optional[int]` is not checked.
- A value of only punctuation, a template token or a bracketed placeholder, an invented example name, and a description
  example the request does not write are refused.
- Plan-only rules (subjects and pronouns, identifier expansion, kind and vocabulary words, classifier options, roles,
  `ignore` words, list items) stay helpers of `planning_commands` called from `check_plan`; an execution field that carries
  such an option is checked by `check_plan` through `planning_commands.option_command` (`validate_parameters`).
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

**Validators.** They see the request plus the replies as `command_text` on both kinds (`ParameterLoop.text`). What a validator
rewrote is kept when it passes and dropped when it refuses.

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
`SubTask.check`.

## 3. What still differs between the two kinds

These are differences of what a caller runs, not of a rule:
- Stages. The CME runs the guard (`check_execution`) before `validate_parameters`, which runs the plan guard (`check_plan`) only
  for fields with options; the typed planning call and the draft run `check_plan` inside their validation stage. The retry
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
  `argument_check_refusals`, `argument_check_retry`, `argument_check_refusals_final` (the refusals that remain after the second
  pass, if any), `argument_check_respelled`, `argument_refusal`, `argument_notes`, `unbound_value`, `decision_uncertainty`,
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
