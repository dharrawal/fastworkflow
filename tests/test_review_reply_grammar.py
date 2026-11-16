"""The review form's replies, bound by the real server side (fix-weih.13).

``tests/fixtures/review_form_edits.json`` lists, per case, the review payload items, what the person did
in the chat review form (``edits``) and the exact text the form sends (``reply``). The DOM test
(``chatbot_review_form_dom.cjs``) asserts that ``tmComposeReviewEdits`` produces ``reply``; this module
feeds ``reply`` to the real ``parse_edits``, ``_normalize_parsed_subtask_edits`` and ``apply_edits`` on
real ``SubTaskProposal`` / ``PlanProposal`` instances (and to ``ProposalReplyPolicy``, which resolves a
reply) and asserts what came out.

A case with ``refused_fields`` is something the form refuses to send, because the server would not do
what the person meant. The test shows what the server does with the text the form would otherwise have
composed (``naive_reply``): it must not give the ``intended`` result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastworkflow.review_engine import (
    Change,
    Drop,
    EditParseError,
    EditRefused,
    ProposalReplyPolicy,
    Verdict,
    _normalize_parsed_subtask_edits,
    apply_edits,
    parse_edits,
)
from fastworkflow.review_proposals import PlanProposal, SubTaskProposal

CASES = json.loads((Path(__file__).parent / "fixtures" / "review_form_edits.json").read_text(encoding="utf-8"))
ACCEPTED = [case for case in CASES if "reply" in case]
REFUSED = [case for case in CASES if "refused_fields" in case]


def _proposal(case):
    cls = SubTaskProposal if case["kind"] == "sub_tasks" else PlanProposal
    return cls(items=case["items"])


def _named_values(item) -> dict:
    return {each.name: each.value for each in item.inputs if each.name and each.from_subtask is None}


def _shape(item, kind: str) -> dict:
    if kind == "plan":
        return {"goal": item.goal, "steps": list(item.steps)}
    return {"goal": item.goal, "inputs": _named_values(item)}


def _intended(case, expected: dict) -> list[dict]:
    """What the kept steps must look like: the original, with the expected goals and inputs laid over it."""
    proposal = _proposal(case)
    result = []
    for step, item in enumerate(proposal.items, 1):
        if step in expected.get("dropped", []):
            continue
        shape = _shape(item, case["kind"])
        shape["goal"] = expected.get("goals", {}).get(str(step), shape["goal"])
        if case["kind"] == "sub_tasks":
            shape["inputs"] = {**shape["inputs"], **expected.get("inputs", {}).get(str(step), {})}
        result.append(shape)
    return result


def _applied(proposal, reply: str):
    edits = _normalize_parsed_subtask_edits(proposal, parse_edits(reply))
    return apply_edits(proposal, edits)


def _shapes(proposal, kind: str) -> list[dict]:
    return [_shape(item, kind) for item in proposal.items]


@pytest.mark.parametrize("case", ACCEPTED, ids=[c["name"] for c in ACCEPTED])
def test_the_composed_reply_does_what_the_form_meant(case):
    reply = case["reply"]
    proposal = _proposal(case)
    want = _intended(case, case["expected"])
    if reply == "yes":
        # "yes" is not an edit: parse_edits does not read it, the reply policy accepts it.
        with pytest.raises(EditParseError):
            parse_edits(reply)
        assert ProposalReplyPolicy().decide(proposal, reply).verdict == Verdict.CONFIRM
        assert want == _shapes(proposal, case["kind"])
        return
    edits = parse_edits(reply)
    # Only changes and drops: no clause of the form's text is misread as "add after N" or a continuation.
    assert all(isinstance(edit, (Change, Drop)) for edit in edits), edits
    assert len(edits) == len(reply.split("; ")), (reply, edits)
    result = _applied(proposal, reply)
    assert _shapes(result, case["kind"]) == want
    # The reply policy (how the server resolves the reply) reaches the same proposal.
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.EDIT, decision
    assert _shapes(apply_edits(proposal, list(decision.edits)), case["kind"]) == want
    # The proposal the form was built from is never modified.
    assert _shapes(proposal, case["kind"]) == _shapes(_proposal(case), case["kind"])


@pytest.mark.parametrize("case", REFUSED, ids=[c["name"] for c in REFUSED])
def test_what_the_form_refuses_would_not_do_what_the_person_meant(case):
    """The form refuses these. The server must not silently do something else with the text it would have sent:
    it applies what the grammar documents (``clears``) or refuses with a message (``refuses``), fix-2vyv."""
    proposal = _proposal(case)
    want = _intended(case, case["intended"])
    decision = ProposalReplyPolicy().decide(proposal, case["naive_reply"])
    if decision.verdict == Verdict.EDIT:
        result = apply_edits(proposal, list(decision.edits))
        assert _shapes(result, case["kind"]) != want, case["why"]
    outcome = case["server_outcome"]
    if outcome == "refuses":
        assert decision.verdict in (Verdict.REFUSE, Verdict.REPROMPT), decision
        assert len(decision.message) > 20, decision
        assert proposal.binding() == _proposal(case).binding()
    else:
        assert outcome == "clears"
        assert decision.verdict == Verdict.EDIT, decision
        assert all(each.name != "member" for each in apply_edits(proposal, list(decision.edits)).items[0].inputs)


def test_dropping_all_n_steps_is_refused_and_n_minus_1_is_not():
    for items_case in ("one_input_edit", "plan_no_edits_is_yes", "single_step_proposal_edit"):
        case = next(c for c in CASES if c["name"] == items_case)
        proposal = _proposal(case)
        count = len(proposal.items)
        everything = "; ".join(f"drop {n}" for n in range(1, count + 1))
        with pytest.raises(EditRefused):
            _applied(proposal, everything)
        assert ProposalReplyPolicy().decide(proposal, everything).verdict == Verdict.REFUSE
        if count > 1:
            most = "; ".join(f"drop {n}" for n in range(2, count + 1))
            assert len(_applied(proposal, most).items) == 1


def test_the_fixture_covers_what_the_review_asked_for():
    names = {case["name"] for case in CASES}
    for required in (
        "no_edits_is_yes", "one_input_edit", "several_inputs_on_one_step", "goal_edit",
        "goal_and_input_on_same_step", "skip_one_step", "skip_plus_edits_elsewhere",
        "input_named_check_is_sent_through_inputs", "input_named_goal_is_sent_through_inputs",
        "semicolon_in_a_value", "newline_in_a_value", "equals_in_a_value", "colon_space_in_a_value",
        "drop_n_minus_1_of_n", "plan_goal_edit_and_drop",
    ):
        assert required in names, required
    assert {case["kind"] for case in CASES} == {"sub_tasks", "plan"}


# ---- fix-2vyv: a typed edit is applied exactly as written, or refused with a message; it is never lost or misread ----

import itertools  # noqa: E402
import random  # noqa: E402

from fastworkflow.review_proposals import SubTask, SubTaskInput, unreadable_input_value  # noqa: E402

_META_NAMES = ("check", "goal", "outcome", "means", "desired_outcome")
_INPUT_NAMES = ("member", "book") + _META_NAMES


def _one_item_proposal() -> SubTaskProposal:
    inputs = [SubTaskInput(name=name, value=f"old_{name}") for name in _INPUT_NAMES]
    return SubTaskProposal(items=[SubTask(goal="List loans", inputs=inputs), SubTask(goal="Show them")])


def _named(proposal, step: int = 1) -> dict:
    return {each.name: each.value for each in proposal.items[step - 1].inputs}


def _clause(name: str, written: str) -> str:
    """The text of one input edit as a person types it (a name that is a SubTask field goes through inputs=)."""
    reserved = name in ("goal", "outcome", "means", "check")
    return f"change 1: {'inputs=' if reserved else ''}{name}={written}"


#: (what is typed after name=, what the input must hold afterwards; None: the input is cleared)
_VALUES = [
    ("Zed", "Zed"), ("Ann B. Lee", "Ann B. Lee"), ("a=b", "a=b"), ("Dune: Messiah", "Dune: Messiah"),
    ('"A, book: c"', "A, book: c"), ("'x, member: y'", "x, member: y"), ('"A and member=B"', "A and member=B"),
    ("'it said \"hi\"'", 'it said "hi"'), ("none", None), ("null", None), ("Run it as shown", "Run it as shown"),
    ('"from sub-task 2"', "from sub-task 2"),
]


def _expected_after(old: dict, picks: list) -> dict:
    want = dict(old)
    for name, (_, meant) in picks:
        if meant is None:
            want.pop(name, None)
        else:
            want[name] = meant
    return want


def _decide_applied(proposal, reply: str):
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.EDIT, (reply, decision)
    return apply_edits(proposal, list(decision.edits))


def test_several_input_edits_of_one_step_lose_no_part_for_any_combination():
    """merge(parse(text)) loses nothing: every pair and triple of inputs (goal, outcome, means, check and
    desired_outcome included), every value kind (quoted, splitting, none), in two orders."""
    proposal = _one_item_proposal()
    old = _named(proposal)
    generated = 0
    for size in (2, 3):
        for names in itertools.combinations(_INPUT_NAMES, size):
            for values in ([_VALUES[0]] * size, _VALUES[4:6] + _VALUES[:1], _VALUES[8:9] + _VALUES[2:3] + _VALUES[7:8]):
                picks = list(zip(names, values[:size]))
                for ordering in (picks, picks[::-1]):
                    reply = "; ".join(_clause(name, written) for name, (written, _) in ordering)
                    result = _decide_applied(proposal, reply)
                    assert _named(result) == _expected_after(old, picks), reply
                    assert result.items[0].goal == "List loans" and len(result.items) == 2, reply
                    generated += 1
    rng = random.Random(20261112)
    for _ in range(300):
        names = rng.sample(_INPUT_NAMES, rng.randint(2, len(_INPUT_NAMES)))
        picks = [(name, rng.choice(_VALUES)) for name in names]
        reply = "; ".join(_clause(name, written) for name, (written, _) in picks)
        assert _named(_decide_applied(proposal, reply)) == _expected_after(old, picks), reply
        generated += 1
    assert generated > 400


@pytest.mark.parametrize("reply, kept", [
    ("change 1: inputs=check=strict; change 1: member=Zed", {"check": "strict", "member": "Zed"}),
    ('change 1: member="A, book: c"; change 1: book=Z', {"member": "A, book: c", "book": "Z"}),
    ("change 1: member=none; change 1: inputs=goal=G", {"goal": "G"}),
    ("change 1: inputs=outcome=O; change 1: inputs=means=M; change 1: book=B", {"outcome": "O", "means": "M", "book": "B"}),
])
def test_the_merge_keeps_every_part_of_the_reply(reply, kept):
    result = _decide_applied(_one_item_proposal(), reply)
    named = _named(result)
    for name, value in kept.items():
        assert named.get(name) == value, (reply, named)
    if "member=none" in reply:
        assert "member" not in named


def test_two_edits_of_the_same_input_are_read_as_the_last_or_refused_never_misread():
    proposal = _one_item_proposal()
    decision = ProposalReplyPolicy().decide(proposal, "change 1: member=A; change 1: member=B")
    if decision.verdict == Verdict.EDIT:
        assert _named(apply_edits(proposal, list(decision.edits)))["member"] == "B"
    else:
        assert decision.verdict == Verdict.REFUSE and decision.message


def test_edits_of_a_step_that_does_not_exist_are_refused_not_a_crash():
    proposal = _one_item_proposal()
    for reply in ("change 9: member=A; change 9: book=B", "change 0: member=A; change 0: book=B"):
        decision = ProposalReplyPolicy().decide(proposal, reply)
        assert decision.verdict == Verdict.REFUSE and "only 2 steps" in decision.message, reply


@pytest.mark.parametrize("reply", [
    "change 1: goal=Run it as shown please",
    "change 1: goal=keep it as is",
    "change 1: goal=Yes, that is right",
    "change 1: member=that matches what I want",
    "change 1: member=that matches what I want; change 2: goal=Keep it as shown",
    "drop 2; change 1: goal=Run it as shown",
    "change 1: goal=A\nchange 1: book=that is right",
    "add after 1: goal=run it as shown",
])
def test_an_accept_cue_inside_an_edit_is_part_of_the_edit(reply):
    proposal = _one_item_proposal()
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.EDIT, decision
    if reply.startswith("change 1: goal=Run it as shown please"):
        assert apply_edits(proposal, list(decision.edits)).items[0].goal == "Run it as shown please"
    if reply == "change 1: member=that matches what I want":
        assert _named(apply_edits(proposal, list(decision.edits)))["member"] == "that matches what I want"


def test_an_accept_cue_beside_an_edit_does_not_accept_and_drop_the_edit():
    proposal = _one_item_proposal()
    for reply in ("change 1: goal=Other; run it as shown", "run it as shown; drop 2", "drop 2\nkeep it as is"):
        decision = ProposalReplyPolicy().decide(proposal, reply)
        assert decision.verdict in (Verdict.REPROMPT, Verdict.REFUSE) and decision.message, (reply, decision)


@pytest.mark.parametrize("reply", [
    "yes", "accept", "Run it as shown", "run it as shown.", "Keep it as is", "keep it as shown", "That is right",
    "that matches what I want", "Yes, that is right, run it as shown", "yes, go ahead", "sounds good",
    "Yes, go ahead, but do not run any of the actions.",
])
def test_plain_acceptances_are_still_accepted(reply):
    for proposal in (_one_item_proposal(), PlanProposal(items=[{"goal": "Fix it", "steps": ["a", "b"]}])):
        assert ProposalReplyPolicy().decide(proposal, reply).verdict == Verdict.CONFIRM, reply


@pytest.mark.parametrize("reply", ["no", "decline"])
def test_plain_declines_are_still_declined(reply):
    assert ProposalReplyPolicy().decide(_one_item_proposal(), reply).verdict == Verdict.DECLINE


@pytest.mark.parametrize("reply", [
    'change 1: member="none"', "change 1: member='null'", 'change 1: inputs=book: "None"',
    'change 1: member="none"; change 1: book=Z', "change 1: member='NULL'; change 1: inputs=check=strict",
])
def test_a_quoted_none_cannot_be_the_literal_word_and_is_refused_not_read_as_a_clear(reply):
    proposal = _one_item_proposal()
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.REFUSE, decision
    assert "none and null clear an input" in decision.message
    assert proposal.binding() == _one_item_proposal().binding()


def test_an_unquoted_none_still_clears_and_other_none_words_are_plain_values():
    proposal = _one_item_proposal()
    assert "member" not in _named(_decide_applied(proposal, "change 1: member=none"))
    assert "book" not in _named(_decide_applied(proposal, "change 1: book=NULL"))
    assert _named(_decide_applied(proposal, "change 1: member=none of them"))["member"] == "none of them"
    assert _named(_decide_applied(proposal, "change 1: member=\"nonesuch\""))["member"] == "nonesuch"
    # a goal cannot be none: refused, and the message says why
    decision = ProposalReplyPolicy().decide(proposal, "change 1: goal=none")
    assert decision.verdict == Verdict.REFUSE and "cannot be the text" in decision.message


@pytest.mark.parametrize("reply", [
    "change 1: book='it's \"a\", b: 2'",
    "change 1: book='it's \"a\" and member=x'",
    "change 1: book='it's a, member: 2'; change 1: member=Z",
])
def test_a_quoted_value_with_both_quote_kinds_that_would_split_is_refused(reply):
    proposal = _one_item_proposal()
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.REFUSE, decision
    assert "quote mark" in decision.message and "reword" in decision.message
    assert proposal.binding() == _one_item_proposal().binding()


@pytest.mark.parametrize("reply, held", [
    ("change 1: book='it's \"a\"'", 'it\'s "a"'),         # wrapped, both kinds, nothing to split: read as written
    ("change 1: book=\"A, member: x\"", "A, member: x"),
    ("change 1: book='Tis the season", "'Tis the season"),   # an apostrophe that opens no quote: a plain value
    ("change 1: book=\"Dune\" Messiah", '"Dune" Messiah'),   # nothing splits: the value is kept as written
])
def test_quoted_values_that_the_reader_gets_right_are_read_as_before(reply, held):
    assert _named(_decide_applied(_one_item_proposal(), reply))["book"] == held


def test_unreadable_input_value_names_only_the_two_spellings():
    assert unreadable_input_value("member=Zed") is None
    assert unreadable_input_value("member=A, book=B") is None
    assert unreadable_input_value("member='A', book=\"B\"") is None
    assert unreadable_input_value("member=\"none\"") is not None
    assert unreadable_input_value("member='it's \"a\", b: 2'") is not None
    assert unreadable_input_value("none") is None


def test_the_fixture_marks_every_case_the_form_refuses():
    for case in REFUSED:
        assert case["server_outcome"] in ("clears", "refuses"), case["name"]
    assert {c["name"] for c in ACCEPTED} >= {
        "reserved_input_with_another_input_edit", "splitting_value_with_another_input_edit", "goal_that_reads_as_accept"}


# ---- fix-j6xa: one packed inputs edit never loses a part named like a step field; the model-read path agrees ----

from fastworkflow.review_proposals import ProposedEdit, _proposed_edits_to_engine  # noqa: E402


@pytest.mark.parametrize("reply, lost", [
    ("change 1: inputs=member=Z, check=x", "check=x"),
    ("change 1: inputs=check=x, goal=G", "check=x"),
    ("change 1: inputs=book=B, means=M, member=Z", "means=M"),
    ("change 1: inputs=member=Z and outcome=O", "outcome=O"),
])
def test_a_packed_inputs_edit_with_a_field_named_part_is_refused_naming_the_part(reply, lost):
    proposal = _one_item_proposal()
    decision = ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.REFUSE, decision
    assert f"'{lost}'" in decision.message and "its own edit" in decision.message
    assert proposal.binding() == _one_item_proposal().binding()


@pytest.mark.parametrize("reply, held", [
    ("change 1: inputs=check=x", {"check": "x"}),                       # one input per edit is read as written
    ("change 1: inputs=member=Z; change 1: inputs=check=x", {"member": "Z", "check": "x"}),
    ("change 1: member=Z, book=B", {"member": "Z", "book": "B"}),        # no field-named part: unchanged reading
])
def test_the_readings_a_packed_refusal_must_not_touch(reply, held):
    named = _named(_decide_applied(_one_item_proposal(), reply))
    assert {key: named[key] for key in held} == held


def _reader_outcome(edits):
    proposal = _one_item_proposal()
    try:
        resolved = _proposed_edits_to_engine(proposal, edits, reply_text="as above")
        return "applied", _named(apply_edits(proposal, resolved))
    except EditRefused as refused:
        return "refused", str(refused)


def _read(field, value):
    return ProposedEdit(op="change", index=1, field=field, value=value)


def test_the_model_reader_path_keeps_its_reading_of_an_echoed_part_and_keeps_inputs_it_would_have_merged_away():
    # one reader edit that packs a field-named part: the reader echoes a shown step's means/outcome into its inputs
    # text, and dropping that echo is what keeps the reading correct, so it is read as before (not refused)
    kind, named = _reader_outcome([_read("inputs", "member=Z, means=planning:x; desired outcome: not stated")])
    assert kind == "applied" and named["member"] == "Z" and named["means"] == "old_means"
    # two reader edits, one input each: the merge re-rendered them as one text and lost the second; both are kept now
    kind, named = _reader_outcome([_read("inputs", "member=Z"), _read("inputs", "check=x")])
    assert kind == "applied" and named["member"] == "Z" and named["check"] == "x"
    # the reader's own spelling of a step field and of plain inputs reads as before
    kind, named = _reader_outcome([_read("check", "strict"), _read("inputs", "member=Z, book=B")])
    assert kind == "applied" and named["member"] == "Z" and named["book"] == "B"
