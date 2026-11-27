"""Replays recorded model outputs for 10 generic answer-card turns through parsing and validation. No network.

WHAT THIS PINS: given the exact text a model once returned for each case in
`tests/fixtures/answer_card_eval.json`, `COMPOSER_ADAPTER` parses it, `validate_answer_card` (the data-bound
salvage of scalar refs and list-index paths, the lead-only rule, the table column rule) checks it and
`score_answer_card` grades it good, plain or bad. A change to the validation that makes a recorded outcome
worse, or turns a plain into a card, shows here. Before these rules the same recorded outputs graded
6 good, 0 plain, 4 bad (measured once with a copy of the old functions; the four bad cards were a lead
alone that hid the answer behind a toggle).

WHAT THIS DOES NOT PROVE: that the live model still behaves like this. The outputs were recorded once from
one model (nvidia.nemotron-nano-3-30b, temperature 0, cache off). A new model or prompt needs a new live run.
"""
import json
from typing import Any

import litellm  # noqa: F401  (must precede dspy)
import dspy
import pytest

from fastworkflow.run_chatbot import composer as gui
from tests.answer_card_eval import FIXTURES, block_refs, load_cases, request_for, score_answer_card

CASES = {case["name"]: case for case in load_cases()}


class RecordedSignature(dspy.Signature):
    """The signature the outputs were recorded under (fix-weih.25 replaced it with ``gui.ComposeAnswerCard``, whose
    outputs are ``lead`` and ``sections``). Only its output fields matter: they decide how the raw text parses."""

    user_message: str = dspy.InputField()
    answer: str = dspy.InputField()
    results: str = dspy.InputField()
    lead: str = dspy.OutputField()
    blocks: list[Any] = dspy.OutputField()


def recorded() -> list[dict]:
    return json.loads((FIXTURES / "answer_card_recorded" / "before.json").read_text())["runs"]


def replay(run: dict):
    case = CASES[run["name"]]
    parsed = gui.COMPOSER_ADAPTER.parse(RecordedSignature, run["raw"])
    card = gui.validate_answer_card(gui.AnswerCard(lead=parsed["lead"], blocks=parsed["blocks"]), request_for(case))
    return card, score_answer_card(card, case)


def test_the_cases_and_the_recordings_match():
    assert len(CASES) == 10 and {r["name"] for r in recorded()} == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_recorded_output_makes_a_bad_card(name):
    run = next(r for r in recorded() if r["name"] == name)
    card, (outcome, reasons) = replay(run)
    assert outcome in ("good", "plain"), reasons
    if card is not None:
        assert card.lead or card.blocks


def test_the_recorded_outputs_have_the_pinned_three_way_counts():
    outcomes = {r["name"]: replay(r)[1][0] for r in recorded()}
    counts = {k: sum(v == k for v in outcomes.values()) for k in ("good", "plain", "bad")}
    # fix-weih.25 round 3 (T2): a record block over a list of records is repaired into the table of that list, so
    # "list_of_records" is a card now (it was plain: 8 good, 2 plain)
    assert counts == {"good": 9, "plain": 1, "bad": 0}, outcomes
    assert {n for n, o in outcomes.items() if o == "plain"} == {"nothing_useful_in_result"}


def test_scalar_refs_on_record_blocks_are_salvaged_as_data_bound_stats():
    by_name = {r["name"]: replay(r)[0] for r in recorded()}
    status = by_name["status_like"]
    assert [(b.kind, [s.ref for s in b.stats]) for b in status.blocks] == [("stats", ["r1.status", "r1.reference"])]
    assert set(block_refs(by_name["supporting_lookup"])) == {"r1.amount", "r1.due"}   # nothing from the lookup r2
