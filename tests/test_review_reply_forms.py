"""A review's artifact lists the replies the shown prompt's hint promises, as data (``review["reply_forms"]``, fix-szfq).

The forms are what the HINT TEXT says (the reply reader's own cues are separate and wider); the text itself is
untouched. The agreement tests read each form's word or pattern out of the very hint the user was shown, in order.
Real execution context and review store (the fixtures of tests/test_pending_prompt_description.py); no model call.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow import review_engine
from fastworkflow.review_engine import (
    PROMPT_HINT, REPLY_FORMS_OF_HINT, SUB_TASK_PROMPT_HINT, SUB_TASK_STRUCTURED_PROMPT_HINT, reply_forms)
from tests.test_approval_binding import (  # noqa: F401
    annotated_library, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_pending_prompt_description import (  # noqa: F401
    EMMA, JOKE, _last_artifacts, gated, no_draft_extraction)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401


pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _assert_forms_agree_with_hint(forms, hint):
    """Every word and pattern a form carries is in the hint, and they come in the hint's order."""
    spelled = [form.get("reply") or form.get("pattern") for form in forms if form.get("reply") or form.get("pattern")]
    positions = [hint.index(text) for text in spelled]
    assert positions == sorted(positions)
    if any("pattern" in form for form in forms):
        assert all(form["separator"] in hint for form in forms if "pattern" in form)
    json.dumps(forms)


@pytest.mark.parametrize("hint", [PROMPT_HINT, SUB_TASK_PROMPT_HINT, SUB_TASK_STRUCTURED_PROMPT_HINT])
def test_each_hints_forms_agree_with_its_text(hint):
    forms = dict((text, f) for text, f in REPLY_FORMS_OF_HINT)[hint]
    shown = reply_forms(f"Some rendering\n{hint}", proposal=None)
    assert [f["kind"] for f in shown] == [f["kind"] for f in forms]
    _assert_forms_agree_with_hint(shown, hint)


def test_a_text_with_no_hint_falls_back_to_the_hint_the_proposal_would_show():
    from fastworkflow.review_proposals import SubTaskProposal
    proposal = SubTaskProposal(items=[])
    assert [f.get("reply") for f in reply_forms("I could not read that.", proposal)][:1] == ["yes"]
    assert [f.get("reply") for f in reply_forms("I could not read that.", proposal, structured=True)][:2] == ["accept", "no"]
    assert [f.get("pattern") for f in reply_forms("x", None)] == [None, None, "change N: field=value", "drop N"]


def test_a_subtask_review_carries_its_reply_forms_and_an_edit_refreshes_them(gated, monkeypatch):
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda user_query, **kwargs: [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    review = _last_artifacts(out)["review"]
    record = review_engine.active_record(gated.wf)
    assert record.rendering.endswith(SUB_TASK_PROMPT_HINT) and out.answer.endswith(SUB_TASK_PROMPT_HINT)
    assert [f["kind"] for f in review["reply_forms"]] == ["accept", "edit"]
    assert review["reply_forms"][0]["reply"] == "yes" and review["reply_forms"][1]["free_text"] is True
    # the other keys are as they were
    assert review["kind"] == "sub_tasks" and review["revision"] == 0 and len(review["items"]) == 2
    json.dumps(review)

    # Each later prompt: the forms are those of the hint the text ends with (the hint changes after the first reply).
    hints = {text: forms for text, forms in REPLY_FORMS_OF_HINT}
    seen = {SUB_TASK_PROMPT_HINT}
    for reply in ("change 1: goal=Updated goal", "change 1: goal=Again", "change 1: goal=Third"):
        revised = gated.ctx.process_turn(reply)
        record = review_engine.active_record(gated.wf)
        assert revised.answer == record.rendering
        hint = next(text for text in hints if revised.answer.endswith(text))
        seen.add(hint)
        forms = _last_artifacts(revised)["review"]["reply_forms"]
        assert [(f["kind"], f.get("reply") or f.get("pattern")) for f in forms] == [
            (f["kind"], f.get("reply") or f.get("pattern")) for f in hints[hint]]
        _assert_forms_agree_with_hint(forms, hint)
    assert SUB_TASK_STRUCTURED_PROMPT_HINT in seen, "a later prompt shows the structured hint"
