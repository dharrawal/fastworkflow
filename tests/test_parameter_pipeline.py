"""The one fill stage and the one pipeline with its one retry (bd fix-qoyz step 5).

1. ``fill_from_agent_text``: the four forms an agent's own command text is read in code, direct cases, and the
   diagnostics each leaves.
2. ``fill``: the one call of the model extractor, with a note after the text.
3. ``pipeline.extract``: direct cases of the order of the stages and of the one retry.
4. The equivalence table, ``tests/fixtures/parameter_pipeline_equivalence.json``: the inputs, EVERY TEXT SENT TO THE
   (scripted) MODEL IN ORDER, the result, the diagnostics and the exception that every old call site gave BEFORE the
   sequence was written once (``ParameterExtraction._extract_impl`` / ``_llm_extract``, ``extract_arguments``,
   ``extracted_arguments``, ``typed_planning_call``, ``reextracted_arguments`` and the four code forms). Each must
   still give exactly those outputs with the same number and order of model calls. Steps 6 on of the staged plan change
   behaviour on purpose, one decision at a time: a step that moves an output edits the case it moves, in its own commit.
5. The one retry rule (a second pass after a guard's or a validator's refusal, never after a missing value; a failing
   second pass leaves the first answer) is the same on every caller.

Real components only (the tutorial workflow's commands, the real DSPy extractor with a scripted ``DummyLM`` as the model
boundary, the real guards and validators). No model is called.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from fastworkflow.parameters import fill, pipeline
from tests import parameter_pipeline_corpus as corpus

FIXTURE = Path(__file__).parent / "fixtures" / "parameter_pipeline_equivalence.json"


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    with corpus.Environment(tmp_path_factory.mktemp("pipeline_table")) as env:
        yield env


# ---------------------------------------------------------------- 0. no model stack, no cycle

@pytest.mark.parametrize("module", ["fill", "pipeline"])
def test_the_modules_import_no_model_stack(module):
    code = (f"import sys; import fastworkflow.parameters.{module}; "
            "heavy = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'dspy')]; "
            "print(','.join(sorted(heavy)))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""


# ---------------------------------------------------------------- 1. fill_from_agent_text

class _Find(BaseModel):
    query: str = Field(default="x")
    start: int = Field(default=0)


class _Required(BaseModel):
    query: str = Field(description="words")


class _Defaults(BaseModel):
    page: int = Field(default=1)


def test_xml_is_read_first():
    filled = fill.fill_from_agent_text("find <query>Ann</query><start>2</start>", _Find, "find")
    assert filled.source == "xml"
    assert (filled.params.query, filled.params.start) == ("Ann", "2")


def test_name_value_is_the_next_form():
    filled = fill.fill_from_agent_text("find query=Ann start=2", _Find, "find")
    assert filled.source == "key_value"
    assert (filled.params.query, filled.params.start) == ("Ann", 2)


def test_a_code_written_quoted_value_is_taken_whole():
    text = 'find query="x <y> ..."'
    assert fill.fill_from_agent_text(text, _Find, "find", by_code=True).params.query == "x <y> ..."
    assert fill.fill_from_agent_text(text, _Find, "find", by_code=False) is None


def test_one_positional_value_is_the_next_form():
    filled = fill.fill_from_agent_text("find Ann", _Required, "find")
    assert filled.source == "positional" and filled.params.query == "Ann"


def test_the_bare_command_name_gives_the_defaults_when_nothing_is_required():
    filled = fill.fill_from_agent_text("page", _Defaults, "page")
    assert filled.source == "bare" and filled.params.page == 1


def test_a_text_no_form_reads_is_none_and_the_model_reads_it():
    assert fill.fill_from_agent_text("find me the thing about Ann Lee", _Required, "find") is None


def test_the_forms_are_pure_and_the_class_keeps_its_old_names():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction as pe

    for text in ("find query=Ann", "find Ann", "<query>Ann</query>"):
        old = pe._extract_parameters_from_key_values(text, _Required, "find")
        new = fill.from_key_values(text, _Required, "find")
        assert (old is None) == (new is None)
        assert old is None or old.model_dump() == new.model_dump()
    assert pe._extract_parameters_from_positional("find Ann", _Required, "find").query == "Ann"
    assert pe._extract_parameters_from_xml("<query>Ann</query>", _Required).query == "Ann"
    assert pe._extract_parameters_from_bare("page", _Defaults, "page").page == 1


# ---------------------------------------------------------------- 2. fill

def test_fill_reads_the_text_with_the_note_after_it_and_puts_the_text_back(environment):
    from fastworkflow.parameters.signatures import InputForParamExtraction

    case = {"answers": [corpus.A(query="Ann Lee"), corpus.A(query="Ann Lee")]}
    environment.begin(case)
    workflow = environment.workflow(case)
    params_class = corpus.FORM_CLASSES["OneStr"]
    extractor = InputForParamExtraction.create(workflow, "find_member", "find the member Ann Lee")
    first = fill.fill(extractor, params_class, "find_member", workflow.folderpath)
    second = fill.fill(extractor, params_class, "find_member", workflow.folderpath, note="(a note)")
    assert first.source == second.source == "llm" and first.reset_not_given is None
    assert extractor.command == "find the member Ann Lee"
    assert len(environment.prompts) == 2
    assert "find the member Ann Lee\n\n(a note)" not in environment.prompts[0]
    assert "find the member Ann Lee\n\n(a note)" in environment.prompts[1]


def test_fill_without_an_extractor_makes_one_for_the_text(environment):
    case = {"answers": [corpus.A(query="Ann")]}
    environment.begin(case)
    filled = fill.fill(None, corpus.FORM_CLASSES["OneStr"], "find_member", environment.folder, text="find Ann",
                       note="(n)")
    assert filled.params.query == "Ann"
    assert "find Ann\n\n(n)" in environment.prompts[0]


def test_with_note():
    assert fill.with_note("text") == "text"
    assert fill.with_note("text", "note") == "text\n\nnote"


# ---------------------------------------------------------------- 3. pipeline.extract, direct

def _script(*values):
    """A fill that gives *values* in order and records what it was asked."""
    asked: list = []

    def fill_text(note, attempt):
        asked.append((note, attempt))
        return values[len(asked) - 1]
    return fill_text, asked


def _refuse_text(*bad):
    """A stage that refuses the values in *bad* with the value as the refusal, naming the field it refused (a refusal
    that names none is not read again: ``test_a_refusal_that_names_no_field_gets_no_second_fill``)."""
    def stage(value, attempt):
        return (pipeline.Verdict(pipeline.KIND_REFUSED, str(value), value, open=("value",)) if value in bad
                else pipeline.Verdict(detail=value))
    return stage


RETRY = "(refused: {refusal})"


def test_a_clean_first_pass_is_one_fill_and_no_retry():
    fill_text, asked = _script("a")
    done = pipeline.extract(fill_text, validate=_refuse_text("bad"), retry_note=RETRY)
    assert (done.params, done.final.ok, done.retried, asked) == ("a", True, False, [("", 1)])


def test_a_refusal_is_read_once_more_with_the_note_after_the_request():
    fill_text, asked = _script("bad", "good")
    done = pipeline.extract(fill_text, validate=_refuse_text("bad"), retry_note=RETRY)
    assert asked == [("", 1), ("(refused: bad)", 2)]
    assert (done.params, done.final.ok, done.retried_at, done.first.refusal) == ("good", True, "validate", "bad")


def test_a_refusal_twice_is_the_end_of_it():
    fill_text, asked = _script("bad", "bad")
    done = pipeline.extract(fill_text, validate=_refuse_text("bad"), retry_note=RETRY)
    assert len(asked) == 2 and not done.final.ok and done.retried


def test_no_retry_when_none_is_given():
    fill_text, asked = _script("bad", "good")
    done = pipeline.extract(fill_text, validate=_refuse_text("bad"), retry_note=None)
    assert len(asked) == 1 and not done.final.ok and not done.retried


def test_the_first_note_is_on_the_first_fill():
    fill_text, asked = _script("a")
    pipeline.extract(fill_text, first_note="(first)")
    assert asked == [("(first)", 1)]


def test_a_refusal_of_the_check_stage_is_read_once_more_too():
    fill_text, asked = _script("bad", "good")
    done = pipeline.extract(fill_text, check=_refuse_text("bad"), retry_note=RETRY)
    assert len(asked) == 2 and done.checked.ok and done.retried_at == "check"


def test_one_second_pass_in_all_whichever_stage_refuses():
    """The guard's refusal used the one pass; the validator's refusal of the new values is then not read again."""
    fill_text, asked = _script("bad", "worse")
    done = pipeline.extract(fill_text, check=_refuse_text("bad"), validate=_refuse_text("worse"), retry_note=RETRY)
    assert len(asked) == 2 and not done.final.ok and done.retried_at == "check"


def test_a_validator_refusal_after_a_clean_check_is_read_once_more_and_the_new_values_are_checked_again():
    fill_text, asked = _script("bad", "good")
    checked: list = []
    settled: list = []

    def check(value, attempt):
        checked.append((value, attempt))
        return pipeline.Verdict(detail=value)

    done = pipeline.extract(fill_text, check=check, settle=lambda value, verdict: settled.append(value),
                            validate=_refuse_text("bad"), retry_note=RETRY)
    assert asked == [("", 1), ("(refused: bad)", 2)] and done.final.ok and done.retried_at == "validate"
    assert checked == [("bad", 1), ("good", 2)] and settled == ["bad", "good"]


def test_the_check_stage_retries_and_settles_before_validation():
    fill_text, asked = _script("bad", "good")
    settled: list = []
    order: list = []

    def validate(value, attempt):
        order.append(("validate", value))
        return pipeline.Verdict(detail=value)

    done = pipeline.extract(
        fill_text, check=_refuse_text("bad"), settle=lambda value, verdict: (settled.append(value), order.append("settle")),
        validate=validate, retry_note="({refusal})")
    assert settled == ["good"] and order == ["settle", ("validate", "good")]
    assert done.retried_at == "check" and done.checked.ok


MISSING = pipeline.Verdict(pipeline.KIND_MISSING, "missing required argument x")


def test_a_missing_value_is_never_read_once_more():
    fill_text, asked = _script("m", "good")
    done = pipeline.extract(
        fill_text, validate=lambda value, attempt: MISSING if value == "m" else pipeline.Verdict(detail=value),
        retry_note=RETRY)
    assert len(asked) == 1 and done.final.kind == pipeline.KIND_MISSING and not done.retried


def _failing_second(retry):
    calls: list = []

    def fill_text(note, attempt):
        calls.append(attempt)
        if attempt == 2:
            raise RuntimeError("outage")
        return "bad"

    judged: list = []

    def stage(value, attempt):
        judged.append(attempt)
        return pipeline.Verdict(pipeline.KIND_REFUSED, "bad", open=("value",))

    return fill_text, stage, judged, calls


@pytest.mark.parametrize("stage", ["check", "validate"])
def test_an_error_in_the_second_fill_leaves_the_first_answer_and_its_refusal(stage):
    """The first answer stands (no raise) and its refusal is the final verdict, as any refusal."""
    fill_text, judge, judged, calls = _failing_second(None)
    done = pipeline.extract(fill_text, retry_note=RETRY, **{stage: judge})
    assert done.params == "bad" and not done.retried and calls == [1, 2] and judged == [1]
    assert not done.final.ok and done.final.refusal == "bad"


# ---------------------------------------------------------------- 4. the equivalence table

IDS = [case["id"] for case in corpus.CASES]


def test_the_table_has_every_case_and_every_branch(table):
    assert set(table) == set(IDS) and len(IDS) >= 150
    sites = {case["site"] for case in corpus.CASES}
    assert sites == {"impl", "llm", "args", "typed", "extracted", "reextract", "form"}
    # the retry branches of the table of when each caller retries (design note 3d)
    results = [table[i]["result"] for i in IDS]
    assert any(r.get("diagnostics", {}).get("argument_check_retry") for r in results)             # guard retry
    assert any(r.get("diagnostics", {}).get("argument_check_refusals_final") for r in results)    # guard refused twice
    assert any(r.get("result", {}).get("retried") for r in results)                               # extract_arguments retry
    assert any(r["model_calls"] == 2 and r.get("call") for r in results)                          # typed retry
    assert any(r.get("planner_prompts") and len(r["planner_prompts"]) == 2 for r in results)       # typed fallback used
    assert any(r.get("exception", {}).get("type") == "RuntimeError" for r in results)             # a retry that raises
    assert any(r.get("diagnostics", {}).get("fields_reset_not_given") for r in results)


@pytest.mark.parametrize("case_id", IDS)
def test_every_old_call_site_still_gives_the_frozen_outputs_and_model_calls(environment, table, case_id):
    frozen = table[case_id]
    case = next(c for c in corpus.CASES if c["id"] == case_id)
    assert json.loads(json.dumps(case)) == frozen["case"], "the corpus moved: edit the case with its measurement"
    result = corpus.run_case(environment, case)
    assert result["prompts"] == frozen["result"]["prompts"], "the texts sent to the model, in order"
    assert result == frozen["result"]


# ---------------------------------------------------------------- 5. the one retry rule is the same everywhere

def _rerun(environment, table, ids, monkeypatch, patch):
    """The ids whose result differs from the frozen one once *patch* (a function of monkeypatch) is applied."""
    patch(monkeypatch)
    moved = []
    for case_id in ids:
        case = next(c for c in corpus.CASES if c["id"] == case_id)
        if corpus.run_case(environment, case) != table[case_id]["result"]:
            moved.append(case_id)
    return moved


def test_the_one_rule_for_the_retry_refused_is_read_once_more_missing_is_not(environment, table):
    """A refused value is read once more on both plan paths and a missing one is not (the frozen table says so)."""
    refused = table["args-ab-invented-then-clean"]["result"]
    missing = table["args-ab-missing-then-clean"]["result"]
    assert refused["model_calls"] == 2 and refused["result"]["valid"] is True
    assert missing["model_calls"] == 1 and missing["result"]["valid"] is False


def test_the_note_of_the_one_retry_is_live(environment, table, monkeypatch):
    """Both kinds of command read the same note after the request on the second pass."""
    from fastworkflow.parameters import argument_checks

    moved = _rerun(environment, table, ["impl-fm-guard-then-clean", "llm-fm-guard-then-clean", "args-ab-invented-then-clean"],
                   monkeypatch, lambda mp: mp.setattr(argument_checks, "REFUSED_EXTRACTION_NOTE",
                                                      "(changed: {refusal})"))
    assert moved


def test_the_retry_of_the_plan_paths_and_the_execution_path_is_one_rule(environment, table):
    """A second pass that raises leaves the first answer on the typed call, the draft and the execution path alike:
    the error does not propagate. What remains is the first answer's refusal, handled as any refusal (the typed call
    goes on to its typed fill and ends in the refusal, a PlanningCommandInputError, not the model's error)."""
    for case_id in ("args-ab-retry-raises", "impl-fm-guard-then-retry-raises", "llm-fm-retry-raises"):
        assert "exception" not in table[case_id]["result"], case_id
    assert table["typed-ab-retry-raises"]["result"]["exception"]["type"] == "PlanningCommandInputError"


# ---------------------------------------------------------------- 6. a parse failure is a refusal (bd fix-qoyz step 6, item 1)

def _unparsed_case(site: str, command: str, text: str, bad: dict, good: dict, **more) -> dict:
    """A case whose first extractor answer is outside the field's Literal (read as text: the guard refuses that one
    field, every other field of the answer keeps its value), then one good answer for the second fill. A text
    ``Literal`` no longer fails a whole answer; the whole-answer retry is covered in ``test_one_spelling_fields``."""
    return {"id": f"unparsed-{site}-{command}", "site": site, "command": command, "text": text,
            "answers": [bad, good], **more}


def test_every_attempt_failing_to_parse_gets_the_one_second_pass_on_the_execution_path(environment):
    case = _unparsed_case("impl", "lend_book", "place a hold on Dune for Ann Lee",
                          corpus.LEND(kind="borrow"), corpus.LEND(kind="hold"))
    out = corpus.run_case(environment, case)
    assert out["model_calls"] == 2
    assert out["output"]["valid"] is True and out["output"]["params"]["kind"] == "hold"
    # the second pass carries the one refused field and its allowed values, in the refusal note
    assert "kind='borrow' is not one of loan, hold" in out["prompts"][1]
    assert out["output"]["params"]["member"] == "Ann Lee"


def test_a_missing_value_still_gets_no_second_pass_on_the_execution_path(environment):
    case = {"id": "m", "site": "impl", "command": "find_member", "text": "find", "answers": [corpus.A(query="NOT_FOUND")]}
    assert corpus.run_case(environment, case)["model_calls"] == 1


def test_every_attempt_failing_to_parse_gets_the_one_second_pass_on_the_planning_paths(environment):
    good = corpus.A(member="Ann Lee", branch="Main", entity_type="account", note="")
    bad = corpus.A(member="Ann Lee", branch="Main", entity_type="savings", note="")
    text = "Move the loan of member Ann Lee to branch Main"
    for site in ("args", "typed"):
        case = _unparsed_case(site, "transfer_loan", text, bad, good)
        if site == "typed":
            case["planner"] = [corpus.A(steps_before=[], steps_after=[])]
        out = corpus.run_case(environment, case)
        assert out["model_calls"] == 2, (site, out)
        # the refusal's words differ by where it is found (the guard: "is not one of ..."; the validator's list of
        # "Valid values"), and either names the field and every allowed value
        note = out["prompts"][1].split("(A first extraction", 1)[1]
        assert "entity_type" in note and "identity" in note and "account" in note, site


# ---------------------------------------------------------------- 7. a refusal that names no input is not a question

@pytest.mark.parametrize("mode", ["assist", "agent", "slash"])
def test_a_refusal_naming_no_input_ends_the_execution_command_with_no_pending_loop(environment, mode):
    case = {"id": "nf", "site": "impl", "command": "shelf_code", "text": "label the shelf NOFIELD", "mode": mode,
            "answers": [corpus.A(code="NOFIELD", floor=0), corpus.A(code="NOFIELD", floor=0)]}
    out = corpus.run_case(environment, case)
    assert out["output"]["valid"] is False
    assert out["output"]["error_msg"] == "this does not fit the request"
    assert "parameter_loop" not in out["cme_context"]
    assert out["output"]["parameter_request"] is None


@pytest.mark.parametrize("mode", ["assist", "agent", "slash"])
def test_a_refusal_naming_no_input_is_not_read_again_so_a_rewritten_answer_does_not_run(environment, mode):
    """bd fix-ftdl.9: the second answer would pass the validator, but nobody asked for it: one model call, and the
    command ends with the validator's message."""
    case = {"id": "nf2", "site": "impl", "command": "shelf_code", "text": "label the shelf NOFIELD", "mode": mode,
            "answers": [corpus.A(code="NOFIELD", floor=0), corpus.A(code="GOOD", floor=0)]}
    out = corpus.run_case(environment, case)
    assert out["model_calls"] == 1
    assert out["output"]["valid"] is False and out["output"]["error_msg"] == "this does not fit the request"


def test_a_planning_refusal_naming_no_input_is_not_read_again(environment):
    """The same rule on a planning command's arguments: no second fill, the refusal stands."""
    case = {"id": "nf3", "site": "args", "command": "ledger_pc", "text": "write a line for Nobody and Ann Lee",
            "answers": [corpus.LEDGER(owner="Nobody"), corpus.LEDGER(owner="Ann Lee")]}
    out = corpus.run_case(environment, case)
    assert out["model_calls"] == 1
    assert out["result"]["valid"] is False, out


def test_a_refusal_naming_a_field_is_still_asked(environment):
    case = {"id": "f", "site": "impl", "command": "shelf_code", "text": "label the shelf BAD-1",
            "answers": [corpus.A(code="BAD-1", floor=0), corpus.A(code="BAD-1", floor=0)]}
    out = corpus.run_case(environment, case)
    assert out["output"]["valid"] is False and "parameter_loop" in out["cme_context"]


def test_a_guard_refusal_names_the_argument_so_it_is_asked(environment):
    """The guards always know the argument they refused: it is asked for like any refused value."""
    case = {"id": "g1", "site": "impl", "command": "author_books", "text": "list the books by Frank Herbert",
            "answers": [corpus.A(author="Stephen King"), corpus.A(author="Stephen King")]}
    out = corpus.run_case(environment, case)
    loop = out["cme_context"]["parameter_loop"]
    assert out["output"]["valid"] is False and loop["open"] == ["author"]
    assert out["output"]["parameter_request"]["invalid"] == ["author"]


@pytest.mark.parametrize("mode", ["assist", "agent"])
def test_a_guard_refusal_naming_no_argument_ends_the_command_with_no_pending_loop(environment, monkeypatch, mode):
    from fastworkflow.planning import planning_commands as pc

    def refuse(*args, **kwargs):
        raise pc.PlanningCommandCallError("the call as a whole is refused")

    monkeypatch.setattr(pc, "check_arguments", refuse)
    case = {"id": "g2", "site": "impl", "command": "author_books", "text": "list the books by Frank Herbert",
            "mode": mode, "answers": [corpus.A(author="Frank Herbert"), corpus.A(author="Frank Herbert")]}
    out = corpus.run_case(environment, case)
    assert out["output"]["valid"] is False
    assert out["output"]["error_msg"] == "the call as a whole is refused"
    assert "parameter_loop" not in out["cme_context"]
    assert out["output"]["parameter_request"] is None


# ---------------------------------------------------------------- 8. a validator sees the request plus the replies

def _record_validator_text(monkeypatch) -> list:
    from fastworkflow.parameters import validate as one_driver

    seen: list = []
    real = one_driver.validate

    def recording(input_model, values, *, command_text, workflow, command_name):
        seen.append(command_text)
        return real(input_model, values, command_text=command_text, workflow=workflow, command_name=command_name)

    monkeypatch.setattr(one_driver, "validate", recording)
    return seen


def test_an_execution_validator_sees_the_request_and_every_reply(environment, monkeypatch):
    from fastworkflow.parameters import parameter_clarification as loops
    from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe

    seen = _record_validator_text(monkeypatch)
    environment.begin({"answers": []})
    workflow = environment.workflow({})
    cme = types.SimpleNamespace(context={})
    params_class = pe.parameters_class_of(workflow.folderpath, "shelf_code")
    pe.park(cme, "shelf_code", params_class.model_construct(code="NOT_FOUND", floor=0), question="Q?",
            request_text="label the shelf")
    pe.ParameterExtraction(cme, workflow, "shelf_code", "shelf_code: BAD-3")._extract_impl({})
    assert loops.pending(cme.context).replies == ["BAD-3"]
    pe.ParameterExtraction(cme, workflow, "shelf_code", "shelf_code: BAD-4")._extract_impl({})
    assert [t for t in seen if t] == ["label the shelf\nBAD-3", "label the shelf\nBAD-3\nBAD-4"]
    # Step 6 (consequence a): the second reply in a row that fixed no listed value ends the command (it used to leave the
    # question pending with both replies on the loop); the validator had seen both.
    assert loops.pending(cme.context) is None


def test_a_planning_validator_sees_the_request_as_it_was_given_which_is_the_loops_text(environment, monkeypatch):
    from fastworkflow.parameters import parameter_clarification as loops

    seen = _record_validator_text(monkeypatch)
    loop = loops.ParameterLoop(command="ledger_pc", request_text="write a line for Ann Lee", replies=["3 items"])
    text = loop.text()
    assert text == "write a line for Ann Lee\n3 items"
    case = {"id": "p", "site": "typed", "command": "ledger_pc", "text": text, "planner": [corpus.A(steps_before=[], steps_after=[])],
            "answers": [corpus.A(owner="Ann Lee", count=3, code="")]}
    corpus.run_case(environment, case)
    assert set(seen) - {""} == {text}   # the empty text is the template builder's own probe


# ---------------------------------------------------------------- 9. the second fill answers the refusal and nothing else

def test_the_second_fill_does_not_replace_a_value_the_first_verdict_did_not_name():
    fill_text, asked = _script({"name": "Ann Lee", "code": "BAD"}, {"name": "Ann_Lee", "code": "OK"})

    def validate(values, attempt):
        if values["code"] == "BAD":
            return pipeline.Verdict(pipeline.KIND_REFUSED, "code='BAD' is closed", values, open=("code",))
        return pipeline.Verdict(detail=values)

    done = pipeline.extract(fill_text, validate=validate, retry_note=RETRY)
    assert len(asked) == 2 and done.final.ok and done.retried_at == "validate"
    assert done.params == {"name": "Ann Lee", "code": "OK"}


def test_the_second_fill_is_merged_into_the_answer_as_the_model_gave_it_not_as_a_stage_wrote_it():
    """A stage that writes on the values (a respelling, a cleared field) does not leave its writes in the merged
    answer: the stages judge the merged answer again and write what they write again."""
    fill_text, _ = _script({"name": "ann lee", "code": "BAD"}, {"name": "x", "code": "OK"})

    def validate(values, attempt):
        values["name"] = values["name"].title()        # a stage writing in place
        if values["code"] == "BAD":
            return pipeline.Verdict(pipeline.KIND_REFUSED, "closed", values, open=("code",))
        return pipeline.Verdict(detail=values)

    done = pipeline.extract(fill_text, validate=validate, retry_note=RETRY)
    assert done.params == {"name": "Ann Lee", "code": "OK"}


def test_a_refusal_that_names_no_field_gets_no_second_fill():
    """bd fix-ftdl.9: a refusal that names no field is not read again (its second answer used to be taken whole, so the
    command could run with values the model rewrote and nobody was asked); the first answer and its refusal stand, for
    the caller to end the command (execution) or hand the call on (plan)."""
    fill_text, asked = _script({"name": "Ann Lee", "code": "BAD"}, {"name": "Bob", "code": "OK"})
    for check in (None, lambda v, a: pipeline.Verdict(pipeline.KIND_REFUSED, "no fit", v)):
        asked.clear()
        done = pipeline.extract(
            fill_text, retry_note=RETRY, check=check,
            validate=lambda v, a: pipeline.Verdict(pipeline.KIND_REFUSED, "no fit", v) if v["code"] == "BAD"
            else pipeline.Verdict(detail=v))
        assert len(asked) == 1 and not done.retried and not done.final.ok
        assert done.params == {"name": "Ann Lee", "code": "BAD"} and done.final.refusal == "no fit"


def test_an_unparsed_first_answer_takes_the_second_answer_whole():
    fill_text, _ = _script({"name": None, "code": None}, {"name": "Ann Lee", "code": "OK"})
    done = pipeline.extract(
        fill_text, retry_note=RETRY, unparsed=lambda: True,
        validate=lambda v, a: pipeline.Verdict(pipeline.KIND_REFUSED, "bad", v, open=("code",)) if v["code"] != "OK"
        else pipeline.Verdict(detail=v))
    assert done.params == {"name": "Ann Lee", "code": "OK"}


def test_answered_merges_a_model_by_field():
    class _M(BaseModel):
        name: str
        code: str

    merged = pipeline.answered(_M(name="Ann Lee", code="BAD"), _M(name="Ann_Lee", code="OK"), ("code",))
    assert (merged.name, merged.code) == ("Ann Lee", "OK")
    assert pipeline.answered("a", "b", ("code",)) == "b"


def test_a_marked_value_the_first_fill_had_right_survives_a_second_fill_for_another_field_on_the_execution_path(
        environment):
    """The guard refuses the placeholder note; the second fill writes the member with an underscore. Only the note is
    taken from it: the member the first fill had right stands, and the command's values are valid."""
    case = {"id": "x", "site": "impl", "command": "lend_book", "text": "lend Dune to Ann Lee",
            "answers": [corpus.LEND(note="<note>"), corpus.LEND(member="Ann_Lee", note=None)]}
    out = corpus.run_case(environment, case)
    assert out["model_calls"] == 2
    assert out["output"]["valid"] is True and out["output"]["params"]["member"] == "Ann Lee"


@pytest.mark.parametrize("site", ["typed", "args"])
def test_a_marked_value_the_first_fill_had_right_survives_a_second_fill_for_another_field_on_the_planning_paths(
        environment, site):
    case = {"id": "x", "site": site, "command": "ledger_pc", "text": "write a line for Ann Lee with code BAD",
            "answers": [corpus.LEDGER(code="BAD"), corpus.LEDGER(owner="Ann_Lee", code="OK")],
            "planner": [corpus.A(steps_before=[], steps_after=[])]}
    out = corpus.run_case(environment, case)
    assert out["model_calls"] == 2
    values = out["call"]["args"] if site == "typed" else out["result"]["params"]
    assert (values["owner"], values["code"]) == ("Ann Lee", "OK")


def test_the_guards_corrections_of_other_fields_are_kept_when_one_field_is_refused(environment):
    """The count is refused (the request writes 3); the guard's respelling of the owner to the request's spelling is
    still applied, as when every guard passes."""
    case = {"id": "x", "site": "args", "command": "ledger_pc", "text": "write a line for Ann Lee, 3 items",
            "answers": [corpus.LEDGER(owner="ann lee", count=8)] * 2}
    out = corpus.run_case(environment, case)
    assert out["result"]["valid"] is False and out["result"]["params"]["owner"] == "Ann Lee"
    assert out["result"]["extraction"]["values"].get("owner") == "Ann Lee"
