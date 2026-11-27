"""Evaluation set and scorer for composed question forms (no model calls in this module).

`tests/fixtures/question_form_eval.json` holds generic cases: a question, the turn's results and what a good
form contains. `score_question_form` is a pure, strict function: a form that is safe but carries a field the
question did not ask for FAILS.
"""
import json
import re
from pathlib import Path
from typing import Any, Optional

from fastworkflow.run_chatbot.composer import QuestionForm, QuestionFormRequest, ResultData

FIXTURES = Path(__file__).parent / "fixtures"


def load_cases() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "question_form_eval.json").read_text())["cases"]


def request_for(case: dict[str, Any]) -> QuestionFormRequest:
    return QuestionFormRequest(
        question=case["question"], user_message=case["user_message"],
        results=[ResultData(**r) for r in case["results"]], use_dspy_cache=False,
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _matches(field, spec: dict[str, Any]) -> bool:
    kinds = spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]]
    if field.kind not in kinds:
        return False
    values = {o.value for o in field.options}
    if not set(spec.get("options_must_include", [])) <= values:
        return False
    if len(values) < spec.get("min_options", 0):
        return False
    return not (set(spec.get("options_must_not_include", [])) & values)


def _assign(fields: list, specs: list[dict[str, Any]], used: frozenset = frozenset()) -> bool:
    """Whether every spec can be given its own matching field (small sets, so plain backtracking)."""
    if not specs:
        return True
    for index, field in enumerate(fields):
        if index not in used and _matches(field, specs[0]) and _assign(fields, specs[1:], used | {index}):
            return True
    return False


def score_question_form(form: Optional[QuestionForm], case: dict[str, Any]) -> tuple[bool, list[str]]:
    """``(passed, reasons)``; reasons is empty on a pass."""
    expect = case["expect"]
    if form is None:
        return (True, []) if expect.get("allow_none") else (False, ["no form, but the question asks for something"])
    reasons: list[str] = []
    fields = form.fields
    specs = expect["fields"]
    if len(fields) > expect.get("max_fields", len(specs)):
        reasons.append(f"{len(fields)} fields, at most {expect.get('max_fields', len(specs))} wanted")
    if not _assign(fields, specs):
        reasons.append("no assignment of fields to the asked things: " + ", ".join(
            f"{spec['about']} -> {spec['kind']}" for spec in specs) + "; got " + ", ".join(
            f"{f.kind}({','.join(o.value for o in f.options)})" for f in fields))
    forbidden = {_norm(word) for word in expect.get("forbid_field_names_or_labels", [])}
    for field in fields:
        for text in (field.name, field.label):
            if any(f"_{word}_" in f"_{_norm(text)}_" for word in forbidden):
                reasons.append(f"field restates result data: {text!r}")
                break
    if len(fields) > 1 and any(f.required for f in fields):
        reasons.append("a required field in a multi-field form")
    if any(not f.label.strip() for f in fields):
        reasons.append("empty label")
    return (not reasons), reasons


def outcome_of_question_form(form: Optional[QuestionForm], case: dict[str, Any]) -> str:
    """What the user sees, in three outcomes (fix-weih.17, the precision gate).

    ``good``: a form is shown and `score_question_form` passes it. ``plain``: no form (``None``), the user
    keeps the question and the composer; acceptable, and it is correct-by-fallback when the case says
    ``allow_none`` (a single free-text answer is expected). ``bad``: a form is shown and it fails the scorer.
    """
    if form is None:
        return "plain"
    return "good" if score_question_form(form, case)[0] else "bad"


def plain_was_expected(case: dict[str, Any]) -> bool:
    """Whether a ``plain`` outcome is the labelled-correct one (the case allows no form)."""
    return bool(case["expect"].get("allow_none"))
