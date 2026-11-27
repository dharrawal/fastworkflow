"""The composer reads the recorded request part of each result (``ResultData.sub_task``) as data (fix-2nsr, piece 2).

With numbers the composer derives the parts of the turn (the union of the answer's ``### N.`` headings and the
numbers) for the log diagnostics only, counts the parts no section covers and the blocks bound
across parts (diagnostics only). The ``sub_tasks`` input, the section cap and the headings fallback read the answer's headings alone. The part lines, the ``part`` / ``parts`` marks, the prompt sentence about them, the
ordering of sections by part and the correction trigger for an uncovered part are switched off (measured worse, blind,
2026-11-13): what the model sees in its digest and how its sections are ordered are what they were without numbers. The
model is a scripted ``DummyLM`` (the repo's one allowed seam); no test calls a live model.
"""
import json

import litellm  # noqa: F401  (must precede dspy)
import pytest

from fastworkflow.run_chatbot import composer as gui
from fastworkflow.run_chatbot.composer import AnswerCard, AnswerCardRequest, ResultData
from fastworkflow.utils import dspy_utils
from tests.test_generative_ui import SentLM


def result(rid, command, part=None, supporting=False, context="", parameters=None, output=None):
    return ResultData(
        id=rid, command=command, context=context, parameters=parameters or {}, supporting=supporting,
        output=output or {"status": f"ok {rid}", "count": 3}, sub_task=part)


def heading_answer(*titles):
    return "\n\n".join(f"### {n}. {title}\nDone {n}." for n, title in enumerate(titles, 1))


def card(*sections):
    return AnswerCard(lead="", sections=list(sections))


def section(heading, *refs, verdict=""):
    return {"heading": heading, "verdict": verdict,
            "blocks": [{"kind": "record", "ref": ref, "fields": ["status"]} for ref in refs]}


def validated(raw_sections, request):
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        diag.parts = len(gui.part_titles(request)) if gui.has_part_numbers(request) else 0
        out = gui.validate_answer_card(card(*raw_sections), request)
    finally:
        gui.DIAGNOSTICS.reset(token)
    return out, diag


# ----------------------------------------------------------------------
# The request: the number is optional and bounded
# ----------------------------------------------------------------------


def test_a_result_without_a_part_number_is_as_before_and_old_pages_send_none():
    assert ResultData(id="r1", output={"a": 1}).sub_task is None
    request = AnswerCardRequest(answer="x", results=[{"id": "r1", "output": {"a": 1}}])
    assert request.results[0].sub_task is None and not gui.has_part_numbers(request)


@pytest.mark.parametrize("sent,kept", [(1, 1), (7, 7), (gui.MAX_SUB_TASKS, gui.MAX_SUB_TASKS), (0, None), (-2, None),
                                       (gui.MAX_SUB_TASKS + 1, None), ("3", None), (2.0, None), (True, None), (None, None)])
def test_the_part_number_is_bounds_checked_and_never_refuses_the_request(sent, kept):
    request = AnswerCardRequest(answer="x", results=[{"id": "r1", "output": {"a": 1}, "sub_task": sent}])
    assert request.results[0].sub_task == kept


# ----------------------------------------------------------------------
# The parts: headings, numbers, or both
# ----------------------------------------------------------------------


def test_without_numbers_the_parts_are_exactly_the_heading_parse():
    request = AnswerCardRequest(answer=heading_answer("Find the book", "Pay the fine"), results=[result("r1", "find_book")])
    assert gui.request_sub_tasks(request) == gui.sub_tasks(request.answer) == ["1. Find the book", "2. Pay the fine"]
    assert gui.section_cap(request) == gui.MIN_SECTIONS


def test_numbers_alone_make_the_parts_and_title_them_from_the_label_of_their_primary_result():
    results = [
        result("r1", "Library/find_book", 1, parameters={"title": "Dune"}),
        result("r2", "Library/get_fine", 2, supporting=True, context="Member: Ann"),
        result("r3", "Library/renew_loan", 2, context="Member: Ann"),
        result("r4", "Library/get_fine", 4),
    ]
    request = AnswerCardRequest(answer="All done.", results=results)
    assert gui.part_titles(request) == {1: "Find Book: title=Dune", 2: "Renew Loan: Member: Ann", 4: "Get Fine"}
    assert gui.request_sub_tasks(request) == []      # what the model reads and the caps count: headings alone


def test_a_part_with_only_a_supporting_result_takes_its_label():
    request = AnswerCardRequest(answer="x", results=[result("r1", "find_member", 3, supporting=True, context="Ann")])
    assert gui.part_titles(request) == {3: "Find Member: Ann"}


def test_headings_and_numbers_are_united_and_a_heading_title_wins_over_a_label():
    results = [result("r1", "find_book", 1), result("r2", "pay_fine", 3), result("r3", "list_loans", 2)]
    request = AnswerCardRequest(answer=heading_answer("Locate the book", "Check the loans"), results=results)
    assert gui.part_titles(request) == {1: "Locate the book", 2: "Check the loans", 3: "Pay Fine"}


def test_disagreeing_counts_use_the_union_for_the_diagnostics_but_the_cap_follows_the_headings():
    results = [result(f"r{n}", f"cmd_{n}", n) for n in range(1, 13)]
    request = AnswerCardRequest(answer=heading_answer("First", "Second"), results=results)
    assert list(gui.part_titles(request)) == list(range(1, 13))
    assert gui.section_cap(request) == gui.MIN_SECTIONS
    few = AnswerCardRequest(answer=heading_answer(*[f"Part {n}" for n in range(1, 10)]), results=[result("r1", "cmd", 1)])
    assert gui.section_cap(few) == 9


def test_the_model_input_and_the_caps_are_the_same_with_and_without_part_numbers(sent):
    answer = heading_answer("Find the book", "Check the loan", "Pay the fine")
    numbered = [result("r1", "find_book", 1), result("r2", "list_loans", 2), result("r3", "get_fine", 3), result("r4", "extra", 7)]
    with_numbers = AnswerCardRequest(user_message="q", answer=answer, results=numbered)
    without = AnswerCardRequest(user_message="q", answer=answer, results=bare(numbered))
    assert gui.request_sub_tasks(with_numbers) == gui.request_sub_tasks(without) == gui.sub_tasks(answer)
    assert gui.section_cap(with_numbers) == gui.section_cap(without)
    assert gui.card_token_budget(gui.section_cap(with_numbers)) == gui.card_token_budget(gui.section_cap(without))
    assert gui.results_digest(with_numbers.results) == gui.results_digest(without.results)
    prompts = []
    for request in (with_numbers, without):
        lm = sent([{"lead": "", "sections": [section("Book", "r1")]}])
        gui.compose_answer_card(request)
        prompts.append(lm.sent[0])
    assert prompts[0] == prompts[1]


def test_a_heading_number_beyond_the_bound_is_not_a_part():
    answer = "### 1. A\nx\n\n### 99. B\ny"
    request = AnswerCardRequest(answer=answer, results=[result("r1", "cmd", 1)])
    assert list(gui.part_titles(request)) == [1]


# ----------------------------------------------------------------------
# The digest
# ----------------------------------------------------------------------


NUMBERLESS = [result("r1", "find_book", output={"title": "Dune", "pages": 412}), result("r2", "get_fine", output={"amount": 12.5})]
# the digest the module built before part numbers existed (taken from the pre-change code, byte for byte)
NUMBERLESS_DIGEST = (
    '[{"id":"r1","command":"find_book","context":"","parameters":{},"output":[{"key":"title","title":"title","type":"text",'
    '"sample":"Dune"},{"key":"pages","title":"pages","type":"number","sample":412}]},{"id":"r2","command":"get_fine","context":"",'
    '"parameters":{},"output":[{"key":"amount","title":"amount","type":"number","sample":12.5}]}]')


def test_without_numbers_the_digest_is_the_one_it_always_was():
    assert gui.results_digest(NUMBERLESS) == NUMBERLESS_DIGEST


def big(rid, command, part, fields=40):
    return result(rid, command, part, output={f"field_{i}": "x" * 40 for i in range(fields)})


def lines_of(text):
    return [line for line in json.loads(text) if isinstance(line, dict) and "id" in line]


def bare(results):
    return [r.model_copy(update={"sub_task": None}) for r in results]


def test_numbers_change_nothing_in_the_digest_no_part_lines_no_marks_no_ordering():
    results = [result("r1", "get_fine", 3), result("r2", "find_book", 1), result("r3", "get_fine", 2, supporting=True),
               result("r4", "find_book", 2)]
    text = gui.results_digest(results)
    assert text == gui.results_digest(bare(results))
    digest = json.loads(text)
    assert all("part" not in line and "parts" not in line for line in digest) and '"part' not in text
    assert [line["id"] for line in digest] == ["r1", "r2", "r4", "r3"]    # primary in input order, then supporting


def test_a_collapsed_group_lists_no_parts_and_every_form_is_the_one_without_numbers():
    results = [big(f"r{n * 2 - 1}", "cmd_a", n) for n in range(1, 13)] + [big(f"r{n * 2}", "cmd_b", n) for n in range(1, 13)]
    text = gui.results_digest(results)
    assert len(text) <= gui.DIGEST_MAX_CHARS and text == gui.results_digest(bare(results))
    assert all("parts" not in line and "part" not in line for line in lines_of(text))


def test_when_the_groups_do_not_fit_even_collapsed_the_also_line_is_the_one_without_numbers():
    results = [result(f"r{n}_{k}", f"cmd_{k}", n, output={f"key_{i}": 1 for i in range(12)}, parameters={"q": "v" * 20})
               for n in range(1, 25) for k in range(40)]
    text = gui.results_digest(results)
    assert len(text) <= gui.DIGEST_MAX_CHARS and "also: " in text
    assert text == gui.results_digest(bare(results))


# ----------------------------------------------------------------------
# The prompt
# ----------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch):
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())

    def install(answers):
        lm = SentLM(answers)
        monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
        return lm
    return install


def three_parts():
    return AnswerCardRequest(
        user_message="three things", answer=heading_answer("Find the book", "Check the loan", "Pay the fine"),
        results=[result("r1", "find_book", 1), result("r2", "list_loans", 2), result("r3", "get_fine", 3)])


def test_a_request_without_numbers_sends_the_prompt_it_always_sent(sent):
    request = AnswerCardRequest(user_message="q", answer=heading_answer("A", "B"), results=NUMBERLESS)
    lm = sent([{"lead": "", "sections": [section("Books", "r1")]}])
    gui.compose_answer_card(request)
    prompt = lm.sent[0]
    assert NUMBERLESS_DIGEST.replace('"', '\\"') in prompt
    assert '\\"part\\"' not in prompt
    assert gui._card_signature(5, False).instructions == f"{gui.ComposeAnswerCard.instructions} Write at most 5 sections; " \
        "when the request has more parts than that, merge related parts into one section."
    assert gui.request_sub_tasks(request) == ["1. A", "2. B"]


def test_with_numbers_the_prompt_carries_no_part_lines_marks_or_instruction_sentence(sent):
    request = three_parts()
    lm = sent([{"lead": "", "sections": [section("Book", "r1"), section("Loan", "r2"), section("Fine", "r3")]}])
    gui.compose_answer_card(request)
    prompt = lm.sent[0]
    assert '\\"part\\"' not in prompt and "carry a \\\"part\\\" number" not in prompt.replace("\\\\", "\\")
    assert not hasattr(gui, "PARTS_INSTRUCTION")
    assert gui._card_signature(5, False).instructions == f"{gui.ComposeAnswerCard.instructions} Write at most 5 sections; " \
        "when the request has more parts than that, merge related parts into one section."
    assert "1. Find the book" in prompt          # the sub_tasks input is the headings


# ----------------------------------------------------------------------
# Verification and repair
# ----------------------------------------------------------------------


def test_a_part_with_a_primary_result_and_no_bound_section_is_counted_uncovered_and_triggers_no_correction():
    request = three_parts()
    out, diag = validated([section("Book", "r1"), section("Fine", "r3")], request)
    assert diag.parts == 3 and diag.parts_uncovered == 1 and diag.uncovered == [(2, "Check the loan", ["r2"])]
    assert diag.correctable() == 0 and gui.corrections_text(diag, request) == ""     # diagnostics only
    assert "parts=3 parts_uncovered=1" in diag.summary()


def test_a_part_covered_by_a_merged_section_or_only_supported_is_not_uncovered():
    results = [result("r1", "find_book", 1), result("r2", "list_loans", 2), result("r3", "get_fine", 3, supporting=True)]
    request = AnswerCardRequest(answer="x", results=results)
    out, diag = validated([section("Book and loans", "r1", "r2")], request)
    assert diag.parts_uncovered == 0 and diag.correctable() == 0      # part 3 has no primary result; part 2 shares a section
    assert "parts=3 parts_uncovered=0" in diag.summary()


def test_the_uncovered_parts_are_all_counted_with_their_ids_in_the_diagnostics():
    results = [result(f"r{i}", "cmd", 4) for i in range(1, 7)] + [result("r7", "other", 5)]
    request = AnswerCardRequest(answer=heading_answer("a", "b", "c", "Fourth part"), results=results)
    out, diag = validated([section("Only", "r7")], request)
    assert diag.parts_uncovered == 1 and diag.uncovered == [(4, "Fourth part", ["r1", "r2", "r3", "r4", "r5", "r6"])]
    assert gui.corrections_text(diag, request) == ""


def test_a_lost_part_is_logged_but_triggers_no_correction_pass(sent):
    request = three_parts()
    first = {"lead": "", "sections": [section("Book", "r1"), section("Fine", "r3")]}
    lm = sent([first, first])
    diag = gui.ComposeDiagnostics()
    out = gui.compose_answer_card(request, diag)
    assert len(lm.sent) == 1 and diag.retry == "none" and diag.parts_uncovered == 1
    assert [s.heading for s in out.sections] == ["Book", "Fine"]
    assert "parts=3 parts_uncovered=1" in diag.summary() and "parts_uncovered2" not in diag.summary()


def test_every_part_covered_is_one_call(sent):
    request = three_parts()
    lm = sent([{"lead": "", "sections": [section("Book", "r1"), section("Loan", "r2"), section("Fine", "r3")]}] * 2)
    diag = gui.ComposeDiagnostics()
    gui.compose_answer_card(request, diag)
    assert len(lm.sent) == 1 and diag.retry == "none" and diag.parts_uncovered == 0


def test_a_request_without_numbers_has_no_part_fields_in_its_log_line(sent):
    request = AnswerCardRequest(answer=heading_answer("A", "B"), results=NUMBERLESS)
    diag = gui.ComposeDiagnostics()
    sent([{"lead": "", "sections": [section("Books", "r1")]}])
    gui.compose_answer_card(request, diag)
    assert "parts" not in diag.summary() and "cross_part" not in diag.summary() and diag.parts_uncovered == 0


def test_kept_sections_keep_the_order_the_model_wrote_them_in_whatever_parts_they_bind():
    request = three_parts()
    out, _ = validated([section("Fine", "r3"), section("Book", "r1"), section("Loan", "r2")], request)
    assert [s.heading for s in out.sections] == ["Fine", "Book", "Loan"]


def test_sections_with_and_without_numbered_results_all_stay_where_the_model_put_them():
    results = [result("r1", "find_book", 1), result("r2", "list_loans", 2), result("r3", "get_fine", 2), result("r4", "free_cmd")]
    request = AnswerCardRequest(answer="x", results=results)
    out, _ = validated([section("Last part", "r3"), section("Free", "r4"), section("Loan", "r2"), section("Book", "r1")], request)
    assert [s.heading for s in out.sections] == ["Last part", "Free", "Loan", "Book"]


def test_a_block_bound_to_another_part_is_counted_and_never_dropped():
    results = [result("r1", "find_book", 1, context="Dune"), result("r2", "get_fine", 2, context="Ann"), result("r3", "list_loans", 1)]
    request = AnswerCardRequest(answer=heading_answer("Find the book Dune", "Pay the fine of Ann"), results=results)
    out, diag = validated([section("The book Dune", "r1", "r3", "r2")], request)
    assert diag.cross_part_blocks == 1 and len(out.sections[0].blocks) == 3 and not diag.drops
    assert "cross_part=1" in diag.summary()
    out, diag = validated([section("Dune and the fine of Ann", "r1", "r3", "r2")], request)
    assert diag.cross_part_blocks == 0          # the heading names both parts: nothing to flag


def test_an_empty_heading_falls_back_to_the_answers_heading_then_the_command_not_the_part_number():
    results = [result("r1", "find_book", 1), result("r2", "get_fine", 4, parameters={"member": "Ann"})]
    request = AnswerCardRequest(answer="x", results=results)
    out, _ = validated([section("", "r2"), section("", "r1")], request)
    assert [s.heading for s in out.sections] == ["Get Fine", "Find Book"]


# ----------------------------------------------------------------------
# A count the request enumerates (the lead that lost "five exception holders")
# ----------------------------------------------------------------------

ROSTER = ("Five reviewers on this quarter's audit list need the same check: Ann Lee, Bo Chan, Cy Diaz, Di Park and Ed Ruiz. "
          "Also check the Vault role.")


def lead_request(message=ROSTER, answer="The Vault role is held by 477 holders. 4 findings were raised, 2 open."):
    return AnswerCardRequest(user_message=message, answer=answer, results=[result("r1", "list_holders", 1)])


def unanchored(text, request):
    return gui._unanchored_sentence(text, gui._anchor_sets(request))


def test_the_number_of_names_the_request_enumerates_supports_a_count_of_what_the_list_is_about():
    request = lead_request()
    assert unanchored("The five audit holders are checked.", request) == ""      # "holders" is 477's noun, "audit" the list's
    assert unanchored("Five reviewers each get a check.", request) == ""        # the message's own word, capitalised
    assert gui.lead_is_grounded("The five audit holders are checked.", request)
    assert gui._enumerations(request.user_message) == [(5, gui._content_words("Five reviewers on this quarter's audit list need the same check:") - set(gui._NUMBER_WORDS))]


def test_the_enumerated_count_is_not_a_licence_for_other_numbers_or_other_nouns():
    request = lead_request()
    assert unanchored("The six audit holders are checked.", request) != ""         # six names are enumerated nowhere
    assert unanchored("Five vault roles were raised.", request) != ""              # nothing in the list sentence says roles
    assert unanchored("There are five holders.", request) != ""                    # 477's noun, and no word of the list sentence


def test_the_original_bug_class_and_the_qualifier_rule_are_still_caught():
    request = lead_request()
    assert unanchored("32 findings were raised.", request) != ""                   # the answer says 4
    assert unanchored("Four findings were raised.", request) == ""
    assert unanchored("Five open findings were raised.", request) != ""            # the answer pairs "open" with 2
    assert unanchored("Two open findings were raised.", request) == ""


HOLDERS = "Cloud Admin: 477 holders in all. Compliance Officer: 513 holders in all. The Vault role has 4 holders. 9 findings were raised, 2 open."


def test_numbers_coordinated_before_one_noun_each_count_that_noun():
    request = lead_request(answer=HOLDERS)
    assert unanchored("Both rights are widely held (477 and 513 holders).", request) == ""
    assert unanchored("Both rights are held, with 477 and 513 holders respectively.", request) == ""
    assert unanchored("The roles have 477, 513 and 4 holders.", request) == ""             # a comma list
    assert unanchored("The roles have 477 or 513 holders.", request) == ""
    assert unanchored("The roles have four and 477 holders.", request) == ""               # number words
    assert unanchored("There are two and 2 open findings.", request) == ""                 # both are the open count


def test_coordinated_numbers_must_each_anchor_the_shared_noun():
    request = lead_request(answer=HOLDERS)
    assert unanchored("The roles have 4 and 9 holders.", request) != ""                     # 9 counts findings, not holders
    assert unanchored("The roles have 477 and 32 holders.", request) != ""                  # 32 is stated nowhere
    assert unanchored("The roles have 477, 513 and 9 holders.", request) != ""
    assert unanchored("There are 9 and 2 open findings.", request) != ""                    # 9 findings are not all open
    assert unanchored("There are 2 and 9 open findings.", request) != ""
    assert unanchored("There are five and two open findings.", request) != ""               # 5 is stated nowhere


def test_coordination_does_not_loosen_the_other_checks():
    request = lead_request(answer=HOLDERS)
    assert unanchored("32 findings were raised.", request) != ""
    assert unanchored("Five open findings were raised.", request) != ""
    assert unanchored("Two open findings were raised.", request) == ""
    assert unanchored("There are 477 findings and 513 holders.", request) != ""             # a noun between them: not coordinated
    assert unanchored("Five reviewers each get a check.", lead_request()) == ""             # the enumerated count still anchors
    pair = lead_request(answer="Total 186.32 and 4 findings.")                              # the answer's own pair is no pair of counts
    assert unanchored("186.32 findings were raised.", pair) != "" and unanchored("4 findings were raised.", pair) == ""


def test_a_list_of_lowercase_words_or_a_single_name_is_not_an_enumeration():
    assert gui._enumerations("Check the right, the system and the feed. Ann Lee needs a login.") == []
    assert gui._enumerations("Ann Lee: these things") == []
