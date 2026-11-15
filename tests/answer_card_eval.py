"""Evaluation set and three-way scorer for composed answer cards (no model calls in this module).

`tests/fixtures/answer_card_eval.json` holds generic cases: a turn (message, answer, results) and what a card
may show. `score_answer_card` is pure: ``good`` (at least one data-bound block that shows data relevant to the
answer, no forbidden block, a grounded non-empty lead), ``plain`` (no card: the deterministic view with the
full answer stays) or ``bad`` (a card that hides the answer while showing nothing useful, or shows something
the case forbids, or has no lead).
"""
import json
from pathlib import Path
from typing import Any, Optional

from fastworkflow.run_chatbot.composer import AnswerCard, AnswerCardRequest, ResultData

FIXTURES = Path(__file__).parent / "fixtures"


def load_cases() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "answer_card_eval.json").read_text())["cases"]


def request_for(case: dict[str, Any]) -> AnswerCardRequest:
    return AnswerCardRequest(
        user_message=case["user_message"], answer=case["answer"],
        results=[ResultData(**r) for r in case["results"]], use_dspy_cache=False,
    )


def _under(ref: str, prefixes: list[str]) -> bool:
    return any(ref == p or ref.startswith(p + ".") or p.startswith(ref + ".") for p in prefixes)


def block_refs(card: AnswerCard) -> list[str]:
    refs: list[str] = []
    for block in card.blocks:
        refs.extend(stat.ref for stat in block.stats)
        if block.ref:
            refs.append(block.ref)
    return refs


def score_answer_card(card: Optional[AnswerCard], case: dict[str, Any]) -> tuple[str, list[str]]:
    """``(outcome, reasons)``; outcome is ``good``, ``plain`` or ``bad``."""
    if card is None:
        return "plain", []
    expect = case["expect"]
    refs = block_refs(card)
    reasons: list[str] = []
    forbidden = [r for r in refs if _under(r, expect["forbid"])]
    if forbidden:
        reasons.append(f"shows forbidden data: {forbidden}")
    if not card.blocks:
        if not expect["lead_only_ok"]:
            reasons.append("a lead only: hides the answer and shows no data")
    elif not [r for r in refs if _under(r, expect["relevant"])] and any(b.kind != "quote" for b in card.blocks):
        reasons.append("no block shows data relevant to the answer")
    if not card.lead.strip():
        reasons.append("no grounded lead")
    return ("bad" if reasons else "good"), reasons
