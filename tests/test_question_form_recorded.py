"""Replays recorded model outputs through parsing, the grounding guards and the scorer. No network.

WHAT THIS PINS: given the exact text a model once returned for each case in
`tests/fixtures/question_form_eval.json`, `COMPOSER_ADAPTER` parses it, `ground_question_form` (guards G1-G7)
cleans it and `score_question_form` judges it. A change to the guards, the parser or the scorer that turns
a recorded pass into a fail, or the baseline numbers into something else, shows here.

WHAT THIS DOES NOT PROVE: that the live model still behaves like this. The outputs were recorded once from
one model (bedrock/nvidia.nemotron-nano-3-30b, temperature 0, cache off). A new model, a new prompt or a
provider change needs a new live run (see `tests/fixtures/question_form_recorded/`).
"""
import json

import pytest

from fastworkflow.run_chatbot import composer as gui
from tests.question_form_eval import (
    FIXTURES, load_cases, outcome_of_question_form, plain_was_expected, request_for, score_question_form,
)

CASES = {case["name"]: case for case in load_cases()}


def recorded(label: str) -> list[dict]:
    return json.loads((FIXTURES / "question_form_recorded" / f"{label}.json").read_text())["runs"]


def replay(run: dict):
    case = CASES[run["name"]]
    parsed = gui.COMPOSER_ADAPTER.parse(gui.ComposeQuestionForm, run["raw"])["form_fields"]
    form = gui.ground_question_form(gui.QuestionForm(fields=parsed), request_for(case))
    return form, score_question_form(form, case)


def test_the_eval_set_is_what_the_recordings_were_made_from():
    assert len(CASES) >= 16 and sum(bool(c.get("holdout")) for c in CASES.values()) == 4
    assert {r["name"] for r in recorded("final")} == set(CASES) == {r["name"] for r in recorded("baseline")}


@pytest.mark.parametrize("name", sorted(CASES))
def test_final_recorded_output_is_good_or_plain_never_bad(name):
    """Three outcomes since G7: a form that passes, or no form (the plain question); never a wrong form."""
    run = next(r for r in recorded("final") if r["name"] == name)
    form, _ = replay(run)
    assert outcome_of_question_form(form, CASES[name]) in ("good", "plain")
    if form is not None and len(form.fields) > 1:
        assert not any(f.required for f in form.fields)


def test_the_recorded_final_outputs_have_the_pinned_three_way_counts():
    outcomes = {r["name"]: outcome_of_question_form(replay(r)[0], CASES[r["name"]]) for r in recorded("final")}
    counts = {k: sum(v == k for v in outcomes.values()) for k in ("good", "plain", "bad")}
    assert counts == {"good": 14, "plain": 6, "bad": 0}
    # 3 of the 6 plain outcomes are the labelled-correct fallback (a single free-text answer is expected)
    assert sum(plain_was_expected(CASES[n]) for n, o in outcomes.items() if o == "plain") == 3
    # without G7 the same outputs were 20 good, which is what the gate trades for (3 forms the labels wanted)
    assert all(score_question_form(replay_without_gate(r), CASES[r["name"]])[0] for r in recorded("final"))


def replay_without_gate(run: dict):
    case = CASES[run["name"]]
    parsed = gui.COMPOSER_ADAPTER.parse(gui.ComposeQuestionForm, run["raw"])["form_fields"]
    return gui.ground_question_form(gui.QuestionForm(fields=parsed), request_for(case), precision_gate=False)


def test_the_guards_alone_lift_the_old_prompt_outputs():
    """The baseline outputs came from the old prompt and the old grounding: 11 of 20 passed then.

    Guards G1-G6 (no gate) lift that to 16. With G7 it is 10 good, 9 plain, 1 bad: the gate gives up forms
    rather than show a wrong one.
    """
    runs = recorded("baseline")
    assert sum(r["live_pass"] for r in runs) == 11
    assert sum(score_question_form(replay_without_gate(r), CASES[r["name"]])[0] for r in runs) >= 16
    counts = [outcome_of_question_form(replay(r)[0], CASES[r["name"]]) for r in runs]
    assert (counts.count("good"), counts.count("plain"), counts.count("bad")) == (10, 9, 1)


def test_the_prompt_example_values_never_leak_into_recorded_outputs():
    for label in ("baseline", "final"):
        assert not any("QX-7" in r["raw"] or "QX-8" in r["raw"] for r in recorded(label))
    assert "QX-7" in gui.ComposeQuestionForm.output_fields["form_fields"].json_schema_extra["desc"]
