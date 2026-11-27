"""The generative-UI composer: what a small model returns is validated, salvaged and grounded.

The only mock is `_composer_lm`, replaced by `dspy.utils.DummyLM`; no test calls a live model.
Everything else is the real module. The grounding tests come first: an invented number, ID, ref or
option must never survive.
"""
import json
import logging

import litellm  # noqa: F401  (must precede dspy)
import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow.run_chatbot import composer as gui
from fastworkflow.run_chatbot.composer import (
    AnswerCard, AnswerCardRequest, CardBlock, ComposerUnavailable, FormField, FormOption, QuestionForm,
    QuestionFormRequest, ResultData, StatItem, clean_and_verify_lead, lead_is_grounded,
    normalize_output_dict, normalize_ref, resolve_ref, results_digest, salvage_list, validate_answer_card,
)
from fastworkflow.utils import dspy_utils


def ground_question_form(form, request):
    """Guards G1-G6 only: the tests below pin those; the precision gate G7 has its own tests (section 5b)."""
    return gui.ground_question_form(form, request, precision_gate=False)


ORDER = {
    "order_id": "#W2041120",
    "status": "delivered",
    "total": 186.32,
    "items": ["shirt", "shoe", "hat"],
    "payment_methods": {"credit_card_4421890": {"brand": "visa"}},
    "lines": [{"name": "shirt", "price": 20.5}, {"name": "shoe", "price": 100.0}],
}
RETAIL_WRAPPED = {"order_details": json.dumps(ORDER)}


def one_result() -> list[ResultData]:
    return [ResultData(id="r1", command="get_order_details", output=RETAIL_WRAPPED)]


def card_request(answer="Order #W2041120 is delivered. Total 186.32.", message="where is my order") -> AnswerCardRequest:
    return AnswerCardRequest(user_message=message, answer=answer, results=one_result())


def form_request(question="Do you want to exchange or return?") -> QuestionFormRequest:
    return QuestionFormRequest(question=question, results=one_result())


# ----------------------------------------------------------------------
# 4. lead grounding: the safety property
# ----------------------------------------------------------------------


def test_grounded_lead_is_kept():
    request = card_request()
    lead = "Order #W2041120 is delivered, total $186.32."
    assert lead_is_grounded(lead, request)
    # a lead alone no longer makes a card when the answer is short (fix-weih.22), so it comes with a block
    card = validate_answer_card(AnswerCard(lead=lead, blocks=[{"kind": "record", "ref": "r1"}]), request)
    assert card is not None and card.lead == lead


def test_hallucinated_id_empties_the_lead_and_keeps_valid_blocks():
    request = card_request()
    block = {"kind": "record", "ref": "r1"}
    card = validate_answer_card(AnswerCard(lead="Order #W2041129 is delivered.", blocks=[block]), request)
    assert card is not None
    assert card.lead == ""
    assert [b.kind for b in card.blocks] == ["record"]


def test_hallucinated_amount_empties_the_lead():
    request = card_request()
    assert not lead_is_grounded("You paid $150.00 for it.", request)
    card = validate_answer_card(AnswerCard(lead="You paid $150.00 for it.", blocks=[{"kind": "record", "ref": "r1"}]), request)
    assert card is not None and card.lead == ""


def test_card_is_none_when_lead_is_invented_and_no_block_survives():
    request = card_request()
    assert validate_answer_card(AnswerCard(lead="Order #W2041129 shipped."), request) is None
    assert validate_answer_card(AnswerCard(lead="Order #W2041129", blocks=[{"kind": "table", "ref": "r9.lines"}]), request) is None


@pytest.mark.parametrize("lead", [
    "You have 7 items.",            # 7 is no count anywhere
    "Order W2041121 is on its way.",  # bare ID off by one digit
    "Delivered on 2024-05-03.",     # a date that is not in the data
    "Your total is 186.50.",        # decimal too far from 186.32
])
def test_other_invented_numbers_and_ids_are_not_grounded(lead):
    assert not lead_is_grounded(lead, card_request())


def test_bare_id_matches_hash_id_and_comma_numbers_and_rounding():
    request = card_request(answer="Refund of 1,234.50 for #W2041120")
    assert lead_is_grounded("Order W2041120 refunded 1234.5", request)
    assert lead_is_grounded("Total about 186.32", request)          # a grounded float
    assert lead_is_grounded("Total about 186.3", request)           # 186.32 rounded to fewer places
    assert not lead_is_grounded("Total about 186.33", request)
    assert not lead_is_grounded("Total about 186.5", request)
    assert lead_is_grounded("There are 3 items", request)           # list length
    assert lead_is_grounded("One result", request)                  # no digits at all


def _grounded_amount(data_value, lead_text) -> bool:
    request = AnswerCardRequest(answer="x", results=[ResultData(id="r1", output={"amount": data_value})])
    return lead_is_grounded(lead_text, request)


@pytest.mark.parametrize("data_value, text, expected", [
    (186.3249, "186.32", True),         # 2-decimal rounding of a grounded float
    (186.32, "186.33", False),          # a mis-copied amount is not rounding
    (186.32, "186.31", False),
    (12.5, "12.50", True),              # the same value, written with a trailing zero
    (186.32, "186.3", True),            # rounding to fewer places
    (0.1 + 0.2, "0.3", True),           # float noise in the data
    (2.5, "3", False),                  # no rounding to a whole number
    (186.32, "186.320", True),
    (186.325, "186.33", True),          # half up
    (186.325, "186.32", True),          # half even
    (186.325, "186.34", False),
    (1e300, "1000000000000000052504760255204420248704468581108159154915854115111802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472196386865459400540160.5", False),
])
def test_decimal_grounding_is_rounding_not_tolerance(data_value, text, expected):
    assert _grounded_amount(data_value, text) is expected


def test_lead_with_nothing_to_check_is_grounded_and_empty_data_means_numbers_fail():
    request = AnswerCardRequest(answer="ok")
    assert lead_is_grounded("All done.", request)
    assert not lead_is_grounded("All 5 done.", request)


def test_dspy_markers_are_stripped_and_lead_is_cut_at_a_word():
    request = card_request()
    assert clean_and_verify_lead("Order delivered. [[ ## completed ## ]]", request) == "Order delivered."
    long_lead = "word " * 100
    cut = clean_and_verify_lead(long_lead, request)
    assert len(cut) <= 300 and cut.endswith("word") and not cut.endswith(" ")


def test_a_long_lead_is_checked_after_it_is_cut():
    request = card_request()
    filler = "word " * 59                       # 295 characters
    long_filler = "word " * 62                  # 310 characters
    assert clean_and_verify_lead(filler + "3 more", request).endswith("3")      # cut at a word; 3 is a list length
    assert clean_and_verify_lead(long_filler + "999", request).endswith("word")   # the invented 999 is cut off, not shown
    assert clean_and_verify_lead("999 " + filler, request) == ""                # the invented 999 is kept, so the lead goes


# ----------------------------------------------------------------------
# 4. blocks and quotes
# ----------------------------------------------------------------------


def test_quote_tolerates_markdown_and_whitespace_but_not_invention_or_echo():
    answer = ("Thanks for waiting.\n\nReturns are accepted within **30 days**   of delivery if the item is unused. "
              "Refunds go to the original payment method after the item is inspected by our team. " * 2)
    request = AnswerCardRequest(answer=answer, results=[])
    ok = {"kind": "quote", "quote": "Returns are accepted within 30 days of delivery if the item is unused."}
    invented = {"kind": "quote", "quote": "Returns are accepted within 90 days of delivery if the item is unused."}
    echo = {"kind": "quote", "quote": answer}
    short = {"kind": "quote", "quote": "30 days"}
    card = validate_answer_card(AnswerCard(lead="", blocks=[ok, invented, echo, short]), request)
    assert card is not None
    assert [b.quote for b in card.blocks] == [ok["quote"]]


def test_stats_keep_only_resolvable_scalars_and_use_canonical_refs():
    request = card_request()
    block = {"kind": "metrics", "stats": [
        {"label": "Status", "ref": "$r1['status']", "tone": "success"},
        {"label": "Order total", "ref": "total"},
        {"label": "Ghost", "ref": "r1.nope"},
        {"label": "Items", "ref": "r1.items"},          # a list is not a scalar
        {"label": "order " + "x" * 34 + " tail", "ref": "r1.order_id"},
        "garbage",
    ]}
    card = validate_answer_card(AnswerCard(blocks=[block]), request)
    assert card is not None
    stats = card.blocks[0].stats
    assert [(s.ref, s.tone) for s in stats] == [("r1.status", "good"), ("r1.total", "neutral"), ("r1.order_id", "neutral")]
    assert [s.label for s in stats[:2]] == ["Status", "Order total"]      # clean labels are kept verbatim
    assert len(stats[2].label) == 40


def test_stat_label_with_an_invented_number_or_id_falls_back_to_a_derived_label():
    request = card_request()
    block = {"kind": "stats", "stats": [
        {"label": "Order #W2041129", "ref": "r1.status"},
        {"label": "3 orders over $500", "ref": "r1.total"},
        {"label": "", "ref": "r1.order_id"},
        {"label": "Order status (3)", "ref": "r1.status"},
    ]}
    marked = {"kind": "stats", "stats": [{"label": "[[ ## completed ## ]] Total order", "ref": "r1.total"}]}
    card = validate_answer_card(AnswerCard(blocks=[block, marked]), request)
    assert [(s.label, s.ref) for s in card.blocks[0].stats] == [
        ("Get Order Details: Status", "r1.status"), ("Get Order Details: Total", "r1.total"),
        ("Get Order Details: Order Id", "r1.order_id"), ("Order status (3)", "r1.status")]
    assert [s.label for s in card.blocks[1].stats] == ["Total order"]


def test_block_title_is_checked_cleaned_and_capped():
    request = card_request()
    blocks = [
        {"kind": "record", "ref": "r1", "title": "Order #W2041129 details"},     # invented ID
        {"kind": "record", "ref": "r1", "title": "3 orders over $500"},          # invented numbers
        {"kind": "record", "ref": "r1", "title": "Order #W2041120 details"},     # clean
        {"kind": "record", "ref": "r1", "title": "  [[ ## completed ## ]]  Order shipping  "},
        {"kind": "record", "ref": "r1", "title": " ".join(["order"] + ["tt"] * 100)},
        {"kind": "quote", "quote": "x" * 5, "title": "Order #W2041129"},
    ]
    card = validate_answer_card(AnswerCard(blocks=blocks), request)
    assert [b.title for b in card.blocks] == ["", "", "Order #W2041120 details", "Order shipping"] + [card.blocks[4].title]
    assert 0 < len(card.blocks[4].title) <= 80


def test_stats_are_capped_at_four_and_an_empty_stats_block_is_dropped():
    request = card_request()
    many = {"kind": "stats", "stats": [{"label": f"s{i}", "ref": "r1.status"} for i in range(9)]}
    card = validate_answer_card(AnswerCard(blocks=[many]), request)
    assert len(card.blocks[0].stats) == 4
    assert validate_answer_card(AnswerCard(blocks=[{"kind": "stats", "stats": [{"label": "a", "ref": "r1.zzz"}]}]), request) is None


def test_table_needs_rows_or_page_shape_and_fields_are_filtered():
    request = card_request()
    rows = {"kind": "table", "ref": "get_order_details.lines", "fields": "name, price, invented"}
    scalar = {"kind": "table", "ref": "r1.status"}
    plain_list = {"kind": "table", "ref": "r1.items"}      # list of scalars, not records
    card = validate_answer_card(AnswerCard(blocks=[rows, scalar, plain_list]), request)
    # the scalar is salvaged as a stat (fix-weih.22); the list of scalars is still dropped
    assert [(b.kind, b.ref, b.fields, [s.ref for s in b.stats]) for b in card.blocks] == [
        ("table", "r1.lines", ["name", "price"], []), ("stats", "", [], ["r1.status"])]


def test_table_page_shape_needs_two_equal_length_scalar_lists():
    page = {"names": ["a", "b"], "prices": [1, 2], "total": 5}      # a page has an integer total (fix-weih.20)
    uneven = {"names": ["a", "b"], "prices": [1]}
    single = {"names": ["a", "b"]}
    results = [ResultData(id="r1", output=page), ResultData(id="r2", output=uneven), ResultData(id="r3", output=single)]
    request = AnswerCardRequest(answer="x", results=results)
    blocks = [{"kind": "table", "ref": r} for r in ("r1", "r2", "r3")]
    card = validate_answer_card(AnswerCard(blocks=blocks), request)
    assert [b.ref for b in card.blocks] == ["r1"]


def test_two_unrelated_lists_of_one_length_are_not_a_page_without_a_total():
    unrelated = {"owners": ["Ann", "Bo"], "managers": ["Cy", "Di"]}
    boolean_total = {"owners": ["Ann", "Bo"], "managers": ["Cy", "Di"], "total": True}
    request = AnswerCardRequest(answer="x", results=[
        ResultData(id="r1", output=unrelated), ResultData(id="r2", output=boolean_total)])
    card = validate_answer_card(AnswerCard(blocks=[{"kind": "table", "ref": "r1"}, {"kind": "table", "ref": "r2"}]), request)
    assert card is None


def test_record_needs_a_nonempty_dict_and_unknown_kinds_are_dropped():
    request = card_request()
    blocks = [
        {"kind": "details", "ref": "r1.payment_methods"},
        {"kind": "record", "ref": "r1.status"},
        {"kind": "chart", "ref": "r1"},
        {"kind": "record", "ref": "r1.nothing"},
    ]
    card = validate_answer_card(AnswerCard(blocks=blocks), request)
    # `record` on the scalar r1.status is salvaged as a one-item stats block (fix-weih.22)
    assert [(b.kind, b.ref, [s.ref for s in b.stats]) for b in card.blocks] == [
        ("record", "r1.payment_methods", []), ("stats", "", ["r1.status"])]


# ---- fix-weih.22: deterministic salvage of the habits of small models (hand-written model outputs) ----

ORDERS = {"customer": "Ann", "orders": [{"id": "A1", "total": 5.0, "due": "May 1"}, {"id": "B2", "total": 7.5, "due": "May 9"}],
          "loans": [{"id": "L1", "due_date": "June 1"}], "tags": ["x", "y"], "status": "open", "count": 2, "paid": True}


def orders_request(answer="Done.", n=1):
    return AnswerCardRequest(answer=answer, results=[
        ResultData(id=f"r{i}", output=dict(ORDERS)) for i in range(1, n + 1)])


def test_a_record_or_table_on_a_scalar_becomes_a_stat_labelled_by_its_grounded_title_or_its_key():
    request = orders_request()
    blocks = [{"kind": "record", "title": "Case status", "ref": "r1.status"},
              {"kind": "table", "title": "Orders open: 99", "ref": "r1.customer"},    # title not grounded: the key
              {"kind": "record", "ref": "r1.paid"}, {"kind": "record", "ref": "r1.count", "quote": "stray words"}]
    card = validate_answer_card(AnswerCard(lead="Done", blocks=blocks), orders_request("Done." + " Details here." * 40))
    assert card is not None and len(card.blocks) == 1 and card.blocks[0].kind == "stats"
    assert [(s.label, s.ref) for s in card.blocks[0].stats] == [
        ("Case status", "r1.status"), ("customer", "r1.customer"), ("paid", "r1.paid"), ("count", "r1.count")]
    assert card.blocks[0].quote == "" and card.blocks[0].title == "" and card.blocks[0].ref == ""
    assert request is not None


def test_salvaged_scalars_merge_per_result_up_to_the_stat_cap_and_do_not_repeat():
    blocks = [{"kind": "record", "ref": f"r1.{key}"} for key in ("status", "customer", "paid", "count", "status")]
    blocks += [{"kind": "record", "ref": "r2.status"}, {"kind": "record", "ref": "r2.customer"}]
    card = validate_answer_card(AnswerCard(blocks=blocks), orders_request(n=2))
    assert [[s.ref for s in b.stats] for b in card.blocks] == [
        ["r1.status", "r1.customer", "r1.paid", "r1.count"], ["r2.status", "r2.customer"]]
    # a block of another kind between two scalars ends the merge
    split = validate_answer_card(AnswerCard(blocks=[
        {"kind": "record", "ref": "r1.status"}, {"kind": "table", "ref": "r1.orders"}, {"kind": "record", "ref": "r1.paid"}]),
        orders_request())
    assert [b.kind for b in split.blocks] == ["stats", "table", "stats"]


def test_a_list_index_path_over_a_list_of_records_becomes_one_table_over_that_list():
    request = orders_request()
    for ref in ("r1.orders.0", "r1.orders[0]", "r1.orders[1].total", "r1.orders.0.total", "orders.1"):
        card = validate_answer_card(AnswerCard(blocks=[{"kind": "record", "ref": ref}]), request)
        assert [(b.kind, b.ref) for b in card.blocks] == [("table", "r1.orders")], ref
    # an unambiguous trailing key alone would be one column: all columns are shown
    assert card.blocks[0].fields == []
    # more than one block over the same list: one table
    twice = validate_answer_card(AnswerCard(blocks=[
        {"kind": "record", "ref": "r1.orders.0"}, {"kind": "record", "ref": "r1.orders[1].id"},
        {"kind": "table", "ref": "r1.orders", "fields": "id,total"}, {"kind": "table", "ref": "r1.loans[0].due_date"}]), request)
    assert [(b.kind, b.ref, b.fields) for b in twice.blocks] == [("table", "r1.orders", []), ("table", "r1.loans", [])]


def test_an_index_path_that_does_not_lead_to_a_list_of_records_is_dropped_as_before():
    request = orders_request()
    for ref in ("r1.tags.0", "r1.status.0", "r1.nothing.0", "r9.orders.0", "r1.0"):
        blocks = [{"kind": "record", "ref": ref}]
        assert validate_answer_card(AnswerCard(blocks=blocks), request) is None, ref


LONG = ("Your account has two open orders, one due in May and one later. " + "Each order lists its own shipping steps and dates. " * 9
        + "\n\nSecond paragraph with more detail about delivery and returns.")


def test_a_lead_only_card_needs_a_long_answer_and_a_lead_that_does_not_restate_the_opening():
    def card(lead, answer):
        # a turn with no results: nothing else to show, so a lead may stand alone
        return validate_answer_card(AnswerCard(lead=lead), AnswerCardRequest(answer=answer))

    assert len(LONG) > gui.LONG_ANSWER_CHARS
    assert card("Two orders are open.", LONG) is not None                           # a real summary
    assert card("Two orders are open.", "Two orders are open. Short.") is None      # the answer is short
    assert card("Your account has two open orders, one due in May and one later.", LONG) is None   # its own opening
    assert card("Your account has two open orders", LONG) is None                   # a prefix of the opening
    short_opening = "Your account has two open orders, one due in May and one later.\n\n" + "More detail follows here. " * 20
    assert card("Your account has two open orders and one is due in May", short_opening) is None   # covers > 60% of paragraph 1
    assert card("Two orders are open.", short_opening) is not None
    assert card("  YOUR   account has two open orders! ", LONG) is None            # case, punctuation, spaces ignored
    assert card("x " * 130, LONG[:450]) is None                                     # not materially shorter
    # blocks make a card whatever the lead says
    assert validate_answer_card(AnswerCard(lead="Your account has two open orders", blocks=[
        {"kind": "stats", "stats": [{"label": "Count", "ref": "r1.a"}]}]), AnswerCardRequest(
            answer=LONG, results=[ResultData(id="r1", output={"a": 1})])) is not None


def test_a_lead_alone_never_replaces_the_answer_of_a_turn_that_has_results():
    """The owner's verdict on a lead plus a wall of footer chips: with a results panel to open, the plain
    answer stays (by design: ``NOT_NEEDED``), however good the lead is."""
    results = [ResultData(id="r1", output={"a": 1})]
    lead = AnswerCard(lead="Two orders are open.")
    request = AnswerCardRequest(answer=LONG, results=results)
    assert gui._validated_card(lead, request) == (None, True)
    assert validate_answer_card(lead, request) is None
    assert validate_answer_card(lead, AnswerCardRequest(answer=LONG)) is not None          # no results: it may stand
    # the same for a v2 response whose sections all fail
    junk = AnswerCard(lead="Two orders are open.", sections=[{"heading": "x", "blocks": [{"kind": "stats", "stats": [
        {"label": "n", "ref": "r9.n"}]}]}])
    assert gui._validated_card(junk, request) == (None, True)


def test_a_table_that_selects_fewer_than_two_columns_shows_all_of_them():
    request = orders_request()
    cases = {"id": [], "id,total": ["id", "total"], "nope": [], "id,nope": [], "": [], "total,due,id": ["total", "due", "id"]}
    for fields, expected in cases.items():
        card = validate_answer_card(AnswerCard(blocks=[{"kind": "table", "ref": "r1.orders", "fields": fields}]), request)
        assert card.blocks[0].fields == expected, fields
    page = AnswerCardRequest(answer="x", results=[ResultData(id="r1", output={"ids": ["a", "b"], "names": ["A", "B"], "total": 2})])
    one = validate_answer_card(AnswerCard(blocks=[{"kind": "table", "ref": "r1", "fields": "ids"}]), page)
    assert one.blocks[0].fields == []
    # a record may still show one field
    rec = validate_answer_card(AnswerCard(blocks=[{"kind": "record", "ref": "r1", "fields": "ids"}]), page)
    assert rec.blocks[0].fields == ["ids"]


def test_compose_reports_not_needed_for_a_restating_lead_and_none_for_a_failure(composer):
    composer([{"lead": "Two orders are open.", "sections": []}])
    assert gui.compose_answer_card(orders_request(LONG)) is gui.NOT_NEEDED         # results to show: the answer stays
    composer([{"lead": "Two orders are open.", "sections": []}])
    assert gui.compose_answer_card(AnswerCardRequest(answer=LONG)) is not None     # nothing else to show
    composer([{"lead": LONG.split(".")[0] + ".", "sections": []}])
    assert gui.compose_answer_card(orders_request(LONG)) is gui.NOT_NEEDED
    composer([{"lead": "", "sections": []}])
    assert gui.compose_answer_card(orders_request(LONG)) is None        # nothing came back: not by design
    assert not gui.NOT_NEEDED and gui.NOT_NEEDED is not None


def test_compose_question_form_reports_not_needed_when_no_form_is_worth_showing(composer, caplog):
    composer([{"form_fields": [{"label": "Name", "kind": "text"}]}])
    with caplog.at_level(logging.WARNING):
        assert gui.compose_question_form(form_request("Please give your name.")) is gui.NOT_NEEDED
    assert not [r for r in caplog.records if "composition failed" in r.getMessage()]


def test_at_most_six_blocks():
    request = card_request()
    card = validate_answer_card(AnswerCard(blocks=[{"kind": "record", "ref": "r1"}] * 9), request)
    assert len(card.blocks) == 6


# ----------------------------------------------------------------------
# 5. question form grounding: the second safety property
# ----------------------------------------------------------------------


def choice(*options, name="pick", kind="choice") -> FormField:
    return FormField(name=name, label="Pick one", kind=kind, options=list(options))


def test_dict_key_and_question_word_are_kept_and_invented_option_is_dropped():
    request = form_request()
    field = choice(
        {"value": "credit_card_4421890", "label": "Visa"},
        {"value": "exchange", "label": "Exchange"},
        {"value": "paypal_5521", "label": "PayPal"},
    )
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert [(o.value, o.label) for o in form.fields[0].options] == [
        ("credit_card_4421890", "Visa"), ("exchange", "Exchange")]


def test_compound_option_is_rescued_to_the_one_grounded_id():
    request = form_request()
    field = choice({"value": "#W2041120 (delivered)", "label": ""})
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert form.fields[0].options == [FormOption(value="#W2041120", label="#W2041120 (delivered)")]


def test_option_label_must_not_name_a_different_id_than_the_value():
    request = form_request()
    field = choice(
        {"value": "#W2041120", "label": "#W2041120 (delivered)"},       # matching label kept
        {"value": "credit_card_4421890", "label": "Visa ending 4421890"},  # digits only, no ID token: kept
        {"value": "exchange", "label": "Swap for #W1994898"},            # names another order
    )
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert [(o.value, o.label) for o in form.fields[0].options] == [
        ("#W2041120", "#W2041120 (delivered)"),
        ("credit_card_4421890", "Visa ending 4421890"),
        ("exchange", "exchange")]
    swapped = choice({"value": "#W2041120", "label": "#W1994898 (delivered)"})
    form = ground_question_form(QuestionForm(fields=[swapped]), request)
    assert form.fields[0].options == [FormOption(value="#W2041120", label="#W2041120")]
    rescued = choice({"value": "#W2041120 (delivered)"})
    form = ground_question_form(QuestionForm(fields=[rescued]), request)
    assert form.fields[0].options == [FormOption(value="#W2041120", label="#W2041120 (delivered)")]


def test_compound_option_with_two_or_zero_grounded_ids_is_dropped():
    results = [ResultData(id="r1", output={"orders": ["#W1111111", "#W2222222"]})]
    request = QuestionFormRequest(question="Which?", results=results)
    field = choice({"value": "#W1111111 or #W2222222"}, {"value": "#W3333333 (shipped)"})
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert form.fields[0].kind == "text" and form.fields[0].options == []


def test_rescue_needs_a_whole_token_not_a_substring_of_a_longer_id():
    results = [ResultData(id="r1", output={"id": "#W2041120"})]
    request = QuestionFormRequest(question="Which?", results=results)
    field = choice({"value": "#W20411209"}, {"value": "x#W2041120"})
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert form.fields[0].kind == "text"


def test_numeric_dict_keys_and_values_ground_options():
    results = [ResultData(id="r1", output={"accounts": {"1008292230": {"type": "checking"}}, "limit": 500})]
    request = QuestionFormRequest(question="Which?", results=results)
    field = choice({"value": "1008292230 (checking)"}, {"value": 500}, {"value": "999"})
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert [o.value for o in form.fields[0].options] == ["1008292230", "500"]


def test_empty_and_single_letter_options_never_ground_through_the_question():
    request = QuestionFormRequest(question="Is it an exchange?", results=[])
    field = choice({"value": ""}, {"value": " "}, {"value": "e"}, {"value": "exchange"})
    form = ground_question_form(QuestionForm(fields=[field]), request)
    assert [o.value for o in form.fields[0].options] == ["exchange"]


def test_choice_without_surviving_options_becomes_text_and_none_when_no_field():
    request = form_request()
    form = ground_question_form(QuestionForm(fields=[choice({"value": "nope"}, kind="multichoice")]), request)
    assert form.fields[0].kind == "text" and form.fields[0].options == []
    assert ground_question_form(QuestionForm(fields=[]), request) is None
    assert ground_question_form(QuestionForm(fields=[FormField(label="   ")]), request) is None


def test_options_are_deduplicated_capped_and_labels_cut():
    values = [f"id_{i}" for i in range(20)]
    results = [ResultData(id="r1", output={"ids": values})]
    request = QuestionFormRequest(question="Which?", results=results)
    field = choice({"value": "id_1", "label": "L" * 200}, *({"value": v} for v in values))
    form = ground_question_form(QuestionForm(fields=[field]), request)
    options = form.fields[0].options
    assert len(options) == 12 and len({o.value for o in options}) == 12
    assert len(options[0].label) == 80


def test_names_are_derived_deduplicated_and_fields_capped_at_six():
    request = form_request()
    fields = [
        FormField(label="Order Number"),
        FormField(name="order_number", label="Order No."),
        FormField(name="Bad Name!", label="Reason"),
        FormField(name="good_name", label="Why", help="h" * 300),
        FormField(label="1st"),
        FormField(label="L" * 100),
        FormField(label="seventh"),
    ]
    form = ground_question_form(QuestionForm(fields=fields), request)
    names = [f.name for f in form.fields]
    assert len(form.fields) == 6 and len(set(names)) == 6
    assert names[0] == "order_number" and names[1] == "order_number_2"
    assert names[2] == "bad_name" and names[3] == "good_name"
    assert names[4] == "f_1st"
    assert len(form.fields[3].help) == 200 and len(form.fields[5].label) == 60
    assert all(__import__("re").fullmatch(r"[a-z][a-z0-9_]{0,39}", n) for n in names)


# ----------------------------------------------------------------------
# 5b. question form guards G1-G6: they hold whatever the model returned (hand-written model outputs)
# ----------------------------------------------------------------------

ACCOUNTS = [
    ResultData(id="r1", command="get_account", output={"account_id": "ACC-1001", "status": "active", "balance": 2450.75}),
    ResultData(id="r2", command="get_account", output={"account_id": "ACC-2002", "status": "frozen", "balance": 310.0}),
]


def account_request(question: str) -> QuestionFormRequest:
    return QuestionFormRequest(question=question, results=ACCOUNTS)


def ground(fields, request):
    return ground_question_form(QuestionForm(fields=fields), request)


def labels_of(form):
    return [f.label for f in form.fields]


def test_g1_drops_fields_that_restate_result_data_the_question_does_not_ask_for():
    request = account_request("Which account do you want to close, ACC-1001 or ACC-2002, and why?")
    form = ground([
        FormField(name="account_id", label="Account ID", kind="choice",
                  options=[{"value": "ACC-1001"}, {"value": "ACC-2002"}]),
        FormField(name="status", label="Status", kind="text"),
        FormField(name="balance", label="Balance", kind="number"),
        FormField(name="reason", label="Reason", kind="longtext"),
    ], request)
    assert labels_of(form) == ["Account ID", "Reason"]


def test_g1_keeps_a_field_the_question_asks_about_and_a_pick_among_named_values():
    # a word of the key is in the question
    request = account_request("Which account_id do you mean?")
    form = ground([FormField(name="account_id", label="Account ID", kind="text")], request)
    assert labels_of(form) == ["Account ID"]
    # the question names two of the key's values, though no word of the key is in it
    request = account_request("Should I close ACC-1001 or ACC-2002?")
    form = ground([FormField(name="account_id", label="Account ID", kind="choice",
                             options=[{"value": "ACC-1001"}, {"value": "ACC-2002"}])], request)
    assert form.fields[0].kind == "choice" and len(form.fields[0].options) == 2
    # a field that is not a key of the results is never touched
    form = ground([FormField(name="reason", label="Reason")], request)
    assert labels_of(form) == ["Reason"]


def test_g1_single_known_value_quoted_as_context_or_not_asked_again():
    results = [ResultData(id="r1", output={"order_id": "ORD-7781", "status": "shipped", "total": 54.2})]
    quoted = QuestionFormRequest(question="Your order (status shipped, total 54.20) cannot change. Cancel it instead?",
                                 results=results)
    form = ground([FormField(name="status", label="Status"), FormField(name="total", label="Total", kind="number"),
                   FormField(name="cancel", label="Cancel it?", kind="boolean")], quoted)
    assert labels_of(form) == ["Cancel it?"]
    # the key's word is in the question and its one value is known: still dropped unless a new value is asked for
    again = QuestionFormRequest(question="What status do you expect?", results=results)
    assert ground([FormField(name="status", label="Status")], again) is None
    changed = QuestionFormRequest(question="What new status should it have?", results=results)
    assert labels_of(ground([FormField(name="status", label="Status")], changed)) == ["Status"]


def test_g1_matches_names_and_labels_after_normalizing_at_any_depth():
    results = [ResultData(id="r1", output={"profile": {"Flagged Logins": 3, "tier": ["gold"]}})]
    request = QuestionFormRequest(question="Anything else to add?", results=results)
    form = ground([FormField(name="x", label="flagged-logins"), FormField(name="tier", label="Plan"),
                   FormField(name="note", label="Note")], request)
    assert labels_of(form) == ["Note"]


def test_g2_required_only_when_the_form_has_one_field():
    request = account_request("Why, and how many?")
    many = ground([FormField(label="Reason", required=True), FormField(label="Count", kind="number", required=True)], request)
    assert [f.required for f in many.fields] == [False, False]
    one = ground([FormField(label="Reason", required=False)], request)
    assert one.fields[0].required is True


def test_g3_named_ids_in_a_plain_text_field_become_a_choice_in_question_order():
    request = account_request("Which one should I close, ACC-2002 or ACC-1001?")
    form = ground([FormField(name="target", label="Account", kind="text")], request)
    field = form.fields[0]
    assert field.kind == "choice" and [(o.value, o.label) for o in field.options] == [
        ("ACC-2002", "ACC-2002"), ("ACC-1001", "ACC-1001")]


def test_g3_does_not_guess_when_unsure():
    both = account_request("Which one should I close, ACC-2002 or ACC-1001?")
    # two competing plain text fields
    form = ground([FormField(label="Account"), FormField(label="Note")], both)
    assert [f.kind for f in form.fields] == ["text", "text"]
    # a longtext field is not a target, so the one plain text field is
    form = ground([FormField(label="Account"), FormField(label="Why", kind="longtext")], both)
    assert [f.kind for f in form.fields] == ["choice", "longtext"]
    # another choice already offers these values
    form = ground([FormField(label="Acct", kind="choice", options=[{"value": "ACC-1001"}, {"value": "ACC-2002"}]),
                   FormField(label="Note")], both)
    assert [f.kind for f in form.fields] == ["choice", "text"]
    # one value, a value that is not an ID, or a context mention far from the other: not alternatives
    for question in ("Which one, ACC-1001?", "Close ACC-1001. The balance is 2450.75 and later we may see ACC-2002 again.",
                     "Is the account active or frozen?"):
        assert ground([FormField(label="Account")], account_request(question)).fields[0].kind == "text"


def test_g3b_a_text_field_with_two_grounded_options_is_a_choice():
    request = account_request("Pick one.")
    form = ground([FormField(label="Account", kind="text", options=[{"value": "ACC-1001"}, {"value": "ACC-2002"}, {"value": "nope"}])], request)
    assert form.fields[0].kind == "choice" and [o.value for o in form.fields[0].options] == ["ACC-1001", "ACC-2002"]
    one = ground([FormField(label="Account", kind="text", options=[{"value": "ACC-1001"}])], request)
    assert one.fields[0].kind == "text" and one.fields[0].options == []


def test_g4_duplicate_and_empty_labels_are_dropped_and_nothing_left_is_none():
    request = account_request("Why?")
    form = ground([FormField(label="Reason"), FormField(name="why", label="  reason "), FormField(label="  ")], request)
    assert labels_of(form) == ["Reason"]
    assert ground([FormField(label="Balance")], request) is None


@pytest.mark.parametrize("question, expected", [
    ("Do you want me to cancel order ORD-5521 now?", True),
    ("Would you like me to renew the loans?", True),
    ("Should I go ahead?", True),
    ("Do you want a refund or an exchange?", False),
    ("Do you want to cancel ORD-1 or ORD-2?", False),
    ("Can you provide the account name?", False),
    ("Which account do you want to use?", False),
    ("Do you want it sent to which address?", False),
    ("Do you want it? Really?", False),
    ("Do you want it", False),
])
def test_g5_yes_no_detection(question, expected):
    assert gui.is_yes_no_question(question) is expected


def test_g5_a_yes_no_question_gets_one_boolean_whatever_the_model_returned():
    request = account_request("Do you want me to close ACC-1001 now?")
    junk = ground([FormField(label="Account", kind="text"), FormField(label="Balance", kind="number")], request)
    assert [(f.kind, f.label, f.required) for f in junk.fields] == [("boolean", "Yes or no", True)]
    own = ground([FormField(name="close_it", label="Close it?", kind="boolean"), FormField(label="Why")], request)
    assert [(f.name, f.kind, f.label) for f in own.fields] == [("close_it", "boolean", "Close it?")]
    assert ground(QuestionForm(fields=[]).fields, request).fields[0].kind == "boolean"


def test_g6_a_text_field_named_like_one_item_of_a_list_becomes_a_choice():
    results = [ResultData(id="r1", command="list_tickets", output={
        "ticket_ids": ["TCK-1", "TCK-2", "TCK-3"], "labels": ["Login", "Export", "Billing"], "total": 3})]
    request = QuestionFormRequest(question="Which one should I open?", results=results)
    form = ground([FormField(name="ticket_id", label="Ticket", kind="text")], request)
    assert form.fields[0].kind == "choice"
    assert [(o.value, o.label) for o in form.fields[0].options] == [("TCK-1", "Login"), ("TCK-2", "Export"), ("TCK-3", "Billing")]
    # one item or a differently named field: no promotion
    other = ground([FormField(name="reason", label="Reason")], request)
    assert other.fields[0].kind == "text"
    single = [ResultData(id="r1", output={"ticket_ids": ["TCK-1"]})]
    kept = ground_question_form(QuestionForm(fields=[FormField(name="ticket_id", label="Ticket")]),
                                QuestionFormRequest(question="Which?", results=single))
    assert kept.fields[0].kind == "text"


# ---- G7: the precision gate (hand-written model outputs; the real function, no shadowing) ----


def gate(fields, question="Which one do you want?", results=None):
    request = QuestionFormRequest(question=question, results=results if results is not None else one_result())
    return gui.ground_question_form(QuestionForm(fields=fields), request)


def opts(*values):
    return [{"value": v, "label": v} for v in values]


def test_g7_a_choice_with_zero_or_one_option_becomes_text_and_a_text_only_form_is_not_shown():
    zero = FormField(label="Pick", kind="choice")                               # no options at all
    nothing_grounded = FormField(label="Pick", kind="choice", options=opts("nope", "never"))
    one = FormField(label="Pick", kind="choice", options=opts("exchange"))      # one grounded option
    for field in (zero, nothing_grounded, one):
        assert gate([field], "Do you want to exchange or return?") is None
    # with a structured field beside it the demoted field is a text field (and is the only free-form one)
    form = gate([one, FormField(label="Done", kind="boolean")], "Do you want to exchange or return?")
    assert [(f.kind, f.options) for f in form.fields] == [("text", []), ("boolean", [])]


def test_g7_free_form_only_forms_return_none():
    question = "Please give the details."
    for kinds in (["text"], ["longtext"], ["number"], ["text", "longtext", "number"]):
        assert gate([FormField(label=f"Field {i}", kind=k) for i, k in enumerate(kinds)], question) is None


def test_g7_a_boolean_a_date_or_a_real_choice_is_worth_showing():
    assert gate([FormField(label="Sure", kind="boolean")], "Please confirm.").fields[0].kind == "boolean"
    assert gate([FormField(label="When", kind="date")], "Please give the date.").fields[0].kind == "date"
    picked = gate([FormField(label="Pick", kind="choice", options=opts("exchange", "return"))],
                  "Do you want to exchange or return?")
    assert [o.value for o in picked.fields[0].options] == ["exchange", "return"] and picked.fields[0].required is True
    many = gate([FormField(label="Pick", kind="multichoice", options=opts("#W2041120", "credit_card_4421890"))], "Which ones?")
    assert many.fields[0].kind == "multichoice"


def test_g7_a_choice_nobody_asked_to_pick_from_becomes_text():
    results = [ResultData(id="r1", output={"ids": ["AA-100", "BB-200", "CC-300"]})]
    menu = FormField(label="Item", kind="choice", options=opts("AA-100", "BB-200", "CC-300"))
    # no pick word, and the question names none of the options: an invented menu
    assert gate([menu], "Do you want more details or anything else?", results) is None
    assert gate([menu, FormField(label="Sure", kind="boolean")], "Do you want more details or anything else?", results).fields[0].kind == "text"
    # a pick word, or two options named in the question, keep it
    assert gate([menu], "Which one?", results).fields[0].kind == "choice"
    assert gate([menu], "Should it be AA-100 or BB-200?", results).fields[0].kind == "choice"
    # a single named option is not an ask to pick
    assert gate([menu], "Is AA-100 the right one?", results) is None


def test_g7_a_g6_list_choice_stays_unless_the_question_is_an_offer():
    results = [ResultData(id="r1", output={"ticket_ids": ["TCK-1", "TCK-2", "TCK-3"]})]
    field = FormField(name="ticket_id", label="Ticket", kind="text")
    assert gate([field], "Please give the ticket.", results).fields[0].kind == "choice"
    assert gate([field], "Do you want more about this ticket or something else?", results) is None


def test_g7_only_the_first_free_form_field_stays_beside_the_structured_ones():
    fields = [
        FormField(label="Why", kind="longtext"), FormField(label="Amount", kind="number"),
        FormField(label="Sure", kind="boolean"), FormField(label="Name", kind="text"),
        FormField(label="When", kind="date"),
    ]
    form = gate(fields, "Please confirm.")
    assert [f.label for f in form.fields] == ["Why", "Sure", "When"]
    assert not any(f.required for f in form.fields)
    # one free-form field beside a choice is untouched (two asks are a legitimate form)
    two = gate([FormField(label="Ticket", kind="choice", options=opts("exchange", "return")),
                FormField(label="Reason", kind="text")], "Which one, exchange or return, and why?")
    assert [f.kind for f in two.fields] == ["choice", "text"]


def test_g7_a_yes_no_question_still_gets_its_boolean():
    form = gate([], "Do you want a refund?")
    assert [f.kind for f in form.fields] == ["boolean"]


def test_known_values_are_id_like_newest_first_capped_and_empty_unless_the_question_picks():
    results = [ResultData(id="r1", output={"ids": ["AA-1"], "note": "hello world"}),
               ResultData(id="r2", output={"orders": {"ORD_9": {"status": "ok"}}, "ids": [f"ZZ-{i}" for i in range(30)]})]
    picked = json.loads(gui.known_values(results, "Which one do you want?"))
    assert picked[0] == "ORD_9" and len(picked) == 20 and "hello world" not in picked and "ok" not in picked
    assert gui.known_values(results, "Do you want it?") == "[]"
    assert gui.known_values(results, "") == "[]"


def test_the_composer_passes_known_values_and_the_signature_keeps_its_public_names(composer):
    composer([{"form_fields": [{"label": "Account", "kind": "choice", "options": ["ACC-1001", "ACC-2002"]}]}])
    form = gui.compose_question_form(account_request("Which account do you want to use?"))
    assert [o.value for o in form.fields[0].options] == ["ACC-1001", "ACC-2002"] and form.fields[0].required is True
    assert list(gui.ComposeQuestionForm.input_fields) == ["question", "user_message", "results", "known_values"]
    assert list(gui.ComposeQuestionForm.output_fields) == ["form_fields"]


# ----------------------------------------------------------------------
# 1. request models
# ----------------------------------------------------------------------


def test_retail_json_string_unwraps_and_digest_shows_dict_keys():
    results = one_result()
    assert results[0].output["order_id"] == "#W2041120"
    assert results[0].output["payment_methods"] == {"credit_card_4421890": {"brand": "visa"}}
    digest = results_digest(results)
    assert "credit_card_4421890" in digest and "list of 3" in digest and "list of 2 records" in digest
    assert json.loads(digest)[0]["id"] == "r1"


def test_normalize_output_dict_rules():
    assert normalize_output_dict({"a": "{\"x\": 1}"}) == {"x": 1}
    assert normalize_output_dict({"a": "[1, 2]"}) == {"a": [1, 2]}            # one key, but a list: kept wrapped
    assert normalize_output_dict({"a": "{not json", "b": "plain"}) == {"a": "{not json", "b": "plain"}
    assert normalize_output_dict({"a": "{\"x\": 1}", "b": 2}) == {"a": {"x": 1}, "b": 2}
    assert normalize_output_dict({"a": "{}"}) == {"a": {}}


def test_normalize_output_dict_peels_nested_wrappers_like_the_browser():
    # a real one-field dict, and a wrapper inside a wrapper (mirrors tmNormalizeOutputObject's tests)
    assert normalize_output_dict({"book": {"title": "Dune", "year": 1965}}) == {"title": "Dune", "year": 1965}
    assert normalize_output_dict({"order_details": '{"order": {"status": "delivered"}}'}) == {"status": "delivered"}
    assert normalize_output_dict({"a": {"b": {"c": {"x": 1, "y": 2}}}}) == {"x": 1, "y": 2}
    # a JSON string inside the peeled record is decoded on the way
    assert normalize_output_dict({"w": {"inner": '{"k": "v"}'}}) == {"k": "v"}
    # the peel stops after five steps, as the browser's does
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"x": 1}}}}}}}
    assert normalize_output_dict(deep) == {"f": {"x": 1}}


@pytest.mark.parametrize("raw", [{"k": 5}, {"k": [1]}, {"k": {}}, {"k": "[1]"}, {"k": "text"}, {"k": "{}"}])
def test_normalize_output_dict_does_not_peel_scalars_lists_or_empty_dicts(raw):
    out = normalize_output_dict(raw)
    assert set(out) == {"k"}


@pytest.mark.parametrize("raw", [
    {"book": {"title": "Dune"}},
    {"order_details": '{"order": {"status": "x"}}'},
    {"a": 1, "b": 2},
    {"a": {"b": {"c": {"d": {"x": 1, "y": 2}}}}},
    {"order_details": json.dumps(ORDER)},
])
def test_normalize_output_dict_is_idempotent(raw):
    once = normalize_output_dict(raw)
    assert normalize_output_dict(once) == once
    assert ResultData(id="r1", output=once).output == once


def test_digest_is_bounded_deterministic_and_collapses_a_group_over_its_share():
    big = {f"field_{i}": "x" * 500 for i in range(40)}
    results = [ResultData(id=f"r{i}", command="cmd", output=big) for i in range(1, 31)]
    digest = results_digest(results)
    assert len(digest) <= 9000 and digest == results_digest(results)
    [line] = json.loads(digest)             # one group, over its share: ONE line, every id kept
    assert line["command"] == "cmd" and line["ids"] == [f"r{i}" for i in range(1, 31)] and line["rows"] == 30
    assert line["keys"][0] == "field_0" and len(line["keys"]) == 12 and "sample" not in line
    huge = [ResultData(id="r1", output={f"k{i}": "y" * 400 for i in range(200)})]
    assert len(results_digest(huge)) <= 9000


def test_digest_types_and_samples():
    out = {"n": 3, "b": True, "z": None, "t": "t" * 300, "l": list(range(9)), "d": {str(i): i for i in range(20)}}
    item = json.loads(results_digest([ResultData(id="r1", output=out, titles={"n": "Count"})]))[0]
    by_key = {f["key"]: f for f in item["output"]}
    assert by_key["n"]["title"] == "Count" and by_key["n"]["type"] == "number"
    assert by_key["b"]["type"] == "boolean" and by_key["z"]["type"] == "null"
    assert len(by_key["t"]["sample"]) == 120 and by_key["l"]["sample"] == [0, 1, 2, 3, 4]
    assert by_key["l"]["type"] == "list of 9" and by_key["d"]["type"] == "record" and len(by_key["d"]["sample"]) == 12


def test_request_limits():
    with pytest.raises(ValueError):
        AnswerCardRequest(answer="x" * 150001)         # was 20001 before fix-weih.34
    with pytest.raises(ValueError):
        QuestionFormRequest(question="x" * 20001)
    many = [{"id": f"r{i}", "output": {"a": 1}} for i in range(61)]
    with pytest.raises(ValueError):
        AnswerCardRequest(answer="x", results=many)
    with pytest.raises(ValueError):
        QuestionFormRequest(question="x", results=many)
    assert len(AnswerCardRequest(answer="x", results=many[:31]).results) == 31      # was refused above 20
    assert len(AnswerCardRequest(answer="x", results=many[:60]).results) == 60


# ----------------------------------------------------------------------
# 2. references
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ref", [
    "r1.status", "$r1['status']", "results.r1.status", "get_order_details.status", "status", "r1.order_details.status",
    '  r1["status"] ',
])
def test_refs_normalize_to_the_canonical_form(ref):
    assert resolve_ref(ref, one_result()) == ("r1.status", "delivered")


@pytest.mark.parametrize("ref", ["r9.status", "r1.items.0", "", "items.0", "order_details.status", "r9.order_details.status", "status.x", "r1.payment_methods.nope"])
def test_unresolvable_refs_raise_key_error(ref):
    with pytest.raises(KeyError):
        resolve_ref(ref, one_result())


def test_bare_key_and_command_do_not_resolve_when_ambiguous():
    results = [
        ResultData(id="r1", command="cmd/get_order", output={"status": "a"}),
        ResultData(id="r2", command="cmd/get_order", output={"status": "b"}),
    ]
    for ref in ("status", "get_order.status"):
        with pytest.raises(KeyError):
            resolve_ref(ref, results)
    assert resolve_ref("r2.status", results) == ("r2.status", "b")


def test_unique_command_among_several_results_resolves():
    results = [
        ResultData(id="r1", command="a/get_user", output={"name": "n"}),
        ResultData(id="r2", command="a/get_order", output={"status": "s"}),
    ]
    assert resolve_ref("get_order.status", results) == ("r2.status", "s")
    assert resolve_ref("a/get_user.name", results) == ("r1.name", "n")


def test_normalize_ref_leaves_unmatched_text_alone():
    assert normalize_ref("whatever.x", one_result()) == "whatever.x"


# ----------------------------------------------------------------------
# 3. synonyms and salvage
# ----------------------------------------------------------------------


def test_form_field_synonyms_and_options():
    kinds = {"dropdown": "choice", "checkboxes": "multichoice", "int": "number", "textarea": "longtext",
             "slider": "text", "Bool": "boolean", "DATE": "date", None: "text"}
    for raw, expected in kinds.items():
        assert FormField.model_validate({"label": "L", "kind": raw}).kind == expected
    assert FormField.model_validate({"label": "Order ID"}).name == ""
    field = FormField.model_validate({"label": "L", "kind": "select", "options": [
        "#W2041120", 5, 1.5, {"id": "a"}, {"key": "b", "label": "B"}, {"value": "c", "label": None}, None, [], {}, True]})
    assert [(o.value, o.label) for o in field.options] == [
        ("#W2041120", "#W2041120"), ("5", "5"), ("1.5", "1.5"), ("a", ""), ("b", "B"), ("c", "")]
    assert FormField.model_validate({"label": "L", "options": "not a list"}).options == []


def test_form_field_required_coercion():
    for raw, expected in {"true": True, "Yes": True, 1: True, "false": False, "no": False, 0: False, None: False}.items():
        assert FormField.model_validate({"label": "L", "required": raw}).required is expected


def test_stat_and_block_synonyms():
    assert [StatItem(label="a", ref="r1.a", tone=t).tone for t in ("OK", " warn ", "failed", "weird", None)] == [
        "good", "warning", "bad", "neutral", "neutral"]
    kinds = {"KPIs": "stats", "grid": "table", "details": "record", "callout": "quote", "pie": "unknown", None: "unknown"}
    for raw, expected in kinds.items():
        assert CardBlock.model_validate({"kind": raw}).kind == expected
    assert [CardBlock(kind="x", tone=t).tone for t in ("good", "warn", "error", "other")] == [
        "success", "warning", "danger", "info"]
    assert CardBlock(kind="table", fields="name, price").fields == ["name", "price"]
    assert CardBlock(kind="table", fields=None).fields == [] and CardBlock(kind="table", fields=3).fields == []
    assert CardBlock(kind="table", fields=["a", 1, {}, ""]).fields == ["a", "1"]


@pytest.mark.parametrize("garbage", [123, {}, "text", None, [1]])
def test_salvage_keeps_valid_items_when_one_is_garbage(garbage):
    blocks = [{"kind": "record", "ref": "r1"}] * 3 + [garbage]
    assert len(salvage_list(blocks, CardBlock)) == 3
    fields = [{"label": "A"}, {"label": "B"}, {"label": "C"}, garbage]
    assert len(salvage_list(fields, FormField)) == 3
    assert len(AnswerCard(blocks=blocks).blocks) == 3 and len(QuestionForm(fields=fields).fields) == 3


def test_salvage_accepts_a_single_dict_and_rejects_other_shapes():
    assert len(salvage_list({"label": "A"}, FormField)) == 1
    assert salvage_list("nope", FormField) == [] and salvage_list(None, FormField) == []


def test_nested_stats_are_salvaged_too():
    block = CardBlock.model_validate({"kind": "stats", "stats": [{"label": "a", "ref": "r1.a"}, 5, {"label": "b"}]})
    assert [s.ref for s in block.stats] == ["r1.a"]


# ----------------------------------------------------------------------
# 6. the model, through DummyLM
# ----------------------------------------------------------------------


class RecordingLM(DummyLM):
    """A DummyLM that remembers the kwargs of every copy made of it, and may refuse temperature once."""

    def __init__(self, answers, refuse_temperature=False):
        super().__init__(answers)
        self.copies = []
        self.refuse_temperature = refuse_temperature

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied.copies = self.copies
        copied.refuse_temperature = self.refuse_temperature
        self.copies.append({**copied.kwargs, "_cache": copied.cache})
        return copied

    def forward(self, prompt=None, messages=None, **kwargs):
        if self.refuse_temperature and not self.kwargs.get("drop_params"):
            raise litellm.UnsupportedParamsError("temperature", model="dummy", llm_provider="dummy")
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def composer(monkeypatch):
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())

    def install(answers, **kwargs):
        lm = RecordingLM(answers, **kwargs)
        monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
        return lm
    return install


def test_composer_constants():
    assert isinstance(gui.COMPOSER_ADAPTER, dspy.ChatAdapter) and gui.COMPOSER_MAX_TOKENS == 1024


def test_compose_answer_card_end_to_end(composer):
    lm = composer([{
        "lead": "Order #W2041120 is delivered, total $186.32. [[ ## completed ## ]]",
        "sections": [{"heading": "", "blocks": [
            {"kind": "KPIs", "stats": [{"label": "Status", "ref": "status"}, {"label": "Fake", "ref": "r1.fake"}]},
            {"kind": "grid", "ref": "r1.lines", "fields": "name"},
            {"kind": "chart"},
            42,
        ]}],
    }])
    card = gui.compose_answer_card(card_request())
    assert card.lead == "Order #W2041120 is delivered, total $186.32."
    assert [b.kind for b in card.blocks] == ["stats", "table"]
    assert [s.ref for s in card.blocks[0].stats] == ["r1.status"]
    assert lm.copies and lm.copies[0]["temperature"] == 0.0 and lm.copies[0]["max_tokens"] == gui.CARD_MAX_TOKENS == 1800


def test_compose_answer_card_drops_an_invented_lead_end_to_end(composer):
    composer([{"lead": "Order #W2041129 was refunded $150.00.", "sections": [{"blocks": [{"kind": "record", "ref": "r1"}]}]}])
    card = gui.compose_answer_card(card_request())
    assert card.lead == "" and [b.kind for b in card.blocks] == ["record"]


def test_compose_question_form_end_to_end(composer):
    composer([{"form_fields": [
        {"label": "Payment method", "kind": "dropdown",
         "options": ["credit_card_4421890", "paypal_5521", {"value": "#W2041120 (delivered)"}]},
        {"label": "Reason", "kind": "textarea"},
        99,
    ]}])
    form = gui.compose_question_form(form_request("Which payment method do you want to use, and why?"))
    assert [f.kind for f in form.fields] == ["choice", "longtext"]
    assert [o.value for o in form.fields[0].options] == ["credit_card_4421890", "#W2041120"]
    assert form.fields[0].name == "payment_method"


def test_use_dspy_cache_false_reaches_the_lm_copy_as_a_disabled_cache_attribute(composer):
    lm = composer([{"lead": "Done.", "sections": []}])
    request = card_request()
    request.use_dspy_cache = False
    gui.compose_answer_card(request)
    assert lm.copies[0]["_cache"] is False          # the copy's `cache` attribute is off ...
    assert "cache" not in lm.copies[0]              # ... and is not a request kwarg sent to the provider


def test_unsupported_params_is_retried_with_drop_params(composer):
    lm = composer([{"lead": "Done.", "sections": [{"blocks": [{"kind": "record", "ref": "r1"}]}]}], refuse_temperature=True)
    card = gui.compose_answer_card(card_request())
    assert card is not None and card.lead == "Done."
    assert lm.copies[0].get("drop_params") is None and lm.copies[-1]["drop_params"] is True
    assert lm.copies[-1]["temperature"] == 0.0


def test_malformed_model_output_returns_none_and_warns_once(composer, caplog):
    composer([{"lead": "ok", "sections": "this is not a json list"}] * 3)
    with caplog.at_level(logging.WARNING):
        card = gui.compose_answer_card(card_request())
    warnings = [r for r in caplog.records if "composition failed" in r.getMessage()]
    assert card is None and len(warnings) == 1


def test_model_error_returns_none_and_logs_one_warning_each(monkeypatch, caplog):
    class Exploding(RecordingLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            raise RuntimeError("boom")
    lm = Exploding([])
    monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
    with caplog.at_level(logging.WARNING):
        assert gui.compose_answer_card(card_request()) is None
        assert gui.compose_question_form(form_request()) is None
    warnings = [r.getMessage() for r in caplog.records if "composition failed" in r.getMessage()]
    assert len(warnings) == 2 and all("RuntimeError" in w for w in warnings)


def test_composer_unavailable_propagates(monkeypatch):
    def unavailable():
        raise ComposerUnavailable("no model")
    monkeypatch.setattr(gui, "_composer_lm", unavailable)
    with pytest.raises(ComposerUnavailable):
        gui.compose_answer_card(card_request())
    with pytest.raises(ComposerUnavailable):
        gui.compose_question_form(form_request())


def test_composer_lm_raises_unavailable_when_the_model_is_not_set(monkeypatch):
    def missing(model_env_var, **kwargs):
        # The key follows the model (dspy_utils.resolve_role): get_lm takes the role's model setting only.
        assert model_env_var == "LLM_RESPONSE_GEN"
        raise ValueError("DSPy Language Model not provided. Set LLM_RESPONSE_GEN environment variable.")
    monkeypatch.setattr(dspy_utils, "get_lm", missing)
    with pytest.raises(ComposerUnavailable):
        gui._composer_lm()


# ----------------------------------------------------------------------
# fix-weih.25 step 1: what the model is shown
# ----------------------------------------------------------------------


def _digest(results):
    return json.loads(results_digest(results))


def test_digest_carries_descriptions_and_hints_and_leaves_hidden_keys_out_entirely():
    result = ResultData(
        id="r1", command="report", output={"internal_id": "x1", "total": 186.32, "rate": 12.5, "note": "hi"},
        titles={"total": "Total"},
        descriptions={"total": "Order total including tax", "rate": "y" * 300, "internal_id": "never shown"},
        hints={"internal_id": "hidden", "total": "hero,currency", "rate": "percent"})
    [item] = _digest([result])
    assert [f["key"] for f in item["output"]] == ["total", "rate", "note"]          # hidden key: no field ...
    assert item["descriptions"] == {"total": "Order total including tax", "rate": "y" * 100}
    assert item["hints"] == {"total": "hero,currency", "rate": "percent"}
    assert "internal_id" not in results_digest([result]) and "never shown" not in results_digest([result])
    bare = _digest([ResultData(id="r1", output={"a": 1})])[0]
    assert "descriptions" not in bare and "hints" not in bare and "supporting" not in bare


def test_a_page_is_one_line_of_column_keys_with_three_sample_rows_and_its_counts():
    page = {"ids": [f"u{i}" for i in range(60)], "names": [f"Name {i}" for i in range(60)], "total": 167}
    [item] = _digest([ResultData(id="r1", command="find_book", output=page)])
    assert [f["key"] for f in item["output"]] == ["total"]                           # no parallel-array previews
    # fix-weih.25 round 3: flat keys and a ``show`` line, not a nested ``page`` record whose path the model copies
    assert "page" not in item
    assert (item["show"], item["rows"], item["first_rows"]) == (
        [{"kind": "table", "ref": "r1", "fields": ["ids", "names"]}], "60 of 167", [["u0", "Name 0"], ["u1", "Name 1"], ["u2", "Name 2"]])
    full = {"ids": ["a", "b"], "names": ["A", "B"], "total": 2}
    [whole] = _digest([ResultData(id="r1", output=full)])
    assert (whole["show"], whole["rows"], whole["first_rows"]) == (
        [{"kind": "table", "ref": "r1", "fields": ["ids", "names"]}], "2", [["a", "A"], ["b", "B"]])
    other = {"ids": ["a", "b"], "names": ["A", "B"]}                                  # no total: not a page
    assert "first_rows" not in _digest([ResultData(id="r1", output=other)])[0]
    hidden = _digest([ResultData(id="r1", output=full, hints={"names": "hidden"})])[0]   # a hidden column is no column
    assert "first_rows" not in hidden and "names" not in json.dumps(hidden)


def test_primary_results_come_before_supporting_ones():
    results = [
        ResultData(id="r1", command="lookup", output={"a": 1}, supporting=True),
        ResultData(id="r2", command="main", output={"b": 2}),
        ResultData(id="r3", command="lookup", output={"a": 3}, supporting=True),
    ]
    items = _digest(results)
    assert [i["id"] for i in items] == ["r2", "r1", "r3"] and [i.get("supporting") for i in items] == [None, True, True]


def _crowd(commands, per_command=1, size=25):
    return [
        ResultData(id=f"r{c * per_command + i + 1}", command=f"cmd_{c}", output={f"k{n}": "z" * 80 for n in range(size)})
        for c in range(commands) for i in range(per_command)
    ]


def test_each_command_group_gets_a_fair_share_and_an_oversized_one_collapses_to_a_line():
    small = [ResultData(id="r1", command="small", output={"a": 1}), ResultData(id="r2", command="small", output={"a": 2})]
    huge = [ResultData(id=f"r{i}", command="huge", output={f"field_{i}": "x" * 200 for i in range(100)}) for i in (3, 4)]
    items = _digest([*small, *huge])
    assert len(results_digest([*small, *huge])) <= 9000
    assert [i["id"] for i in items[:2]] == ["r1", "r2"] and "output" in items[0]     # the small group is in full
    # fix-weih.25 round 3 (T4): a primary group over its share steps down to one RICH line per result (its keys and its
    # first two scalar values) before it would become one line for the group
    rich = items[-2:]
    assert [i["id"] for i in rich] == ["r3", "r4"] and all("output" not in i and i["command"] == "huge" for i in rich)
    assert all(len(i["keys"]) == 12 and len(i["values"]) == 2 for i in rich)
    # still bound to the data: a ref to a key the digest did not show resolves
    assert resolve_ref("r3.field_99", [*small, *huge])[1] == "x" * 200


def test_a_single_oversized_result_first_shows_fewer_fields_before_it_collapses():
    one = ResultData(id="r1", command="huge", output={f"field_{i}": "x" * 200 for i in range(100)})
    [item] = _digest([one])
    assert 0 < len(item["output"]) <= 20 and len(results_digest([one])) <= 9000


def _crowd(commands):
    return [
        ResultData(id=f"r{c + 1}", command=f"cmd_{c}", output={f"{'k' * 30}{n}": "z" * 80 for n in range(25)})
        for c in range(commands)
    ]


def test_results_past_the_budget_become_one_also_line_never_a_silent_drop():
    digest = results_digest(_crowd(commands=40))
    items = json.loads(digest)
    assert len(digest) <= 9000
    also = [i for i in items if isinstance(i, str)]
    assert len(also) == 1 and items[-1] == also[0] and also[0].startswith("also: cmd_")
    shown = {i["command"] for i in items if isinstance(i, dict)}
    named = {part.strip().rsplit(" x", 1)[0] for part in also[0].removeprefix("also: ").split(",")}
    assert shown | named == {f"cmd_{c}" for c in range(40)} and not shown & named    # every command is somewhere
    assert items[0]["command"] == "cmd_0"                                              # the first ones are the ones kept
    many = [ResultData(id=f"r{i}", command="lookup", output={"v": i}, supporting=True) for i in range(1, 4)]
    assert "also" not in results_digest(many)                                          # nothing past the budget: no line


def test_the_also_line_counts_results_per_command_and_supporting_goes_last():
    supporting = [ResultData(id=f"s{i}", command="lookup_extra", output={"v": 1}, supporting=True) for i in range(1, 5)]
    items = json.loads(results_digest([*supporting, *_crowd(commands=40)]))          # the input order does not matter
    assert items[-1].startswith("also: ") and items[-1].endswith("lookup_extra x4")
    assert not any(isinstance(i, dict) and i["command"] == "lookup_extra" for i in items)


def test_sub_tasks_are_read_from_the_numbered_headings_of_the_answer():
    answer = "Intro\n\n### 1. find the blue book\nfound.\n\n### 2. renew it  \nDone.\n\n#### 3. not a sub-task\n\n###4. no space"
    assert gui.sub_tasks(answer) == ["1. find the blue book", "2. renew it"]
    assert gui.sub_tasks("A plain answer.\n\n## Summary\n1. a list") == []
    long_title = "### 1. " + "word " * 40
    assert len(gui.sub_tasks(long_title)[0]) <= 83
    assert gui.sub_tasks("") == []


def test_the_model_is_given_the_sub_tasks_and_the_digest_with_hints(monkeypatch):
    sent: list[str] = []

    class Capturing(RecordingLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            sent.append(json.dumps(messages))
            return super().forward(prompt=prompt, messages=messages, **kwargs)
    lm = Capturing([{"lead": "Done.", "sections": []}])
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())
    monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
    request = AnswerCardRequest(answer="### 1. find X\nok\n\n### 2. renew Y\nok", results=[
        ResultData(id="r1", command="find_x", output={"a": 1}, descriptions={"a": "The A count"}, hints={"a": "hero"})])
    gui.compose_answer_card(request)
    assert len(sent) == 1 and "1. find X" in sent[0] and "2. renew Y" in sent[0] and "The A count" in sent[0]
    assert list(gui.ComposeAnswerCard.input_fields) == ["user_message", "answer", "sub_tasks", "results"]
    assert list(gui.ComposeAnswerCard.output_fields) == ["lead", "sections"]
    assert len((gui.ComposeAnswerCard.instructions or "").split()) <= 250       # was 150 before fix-weih.25 round 3 (T5)


# ----------------------------------------------------------------------
# fix-weih.25 step 2: sections
# ----------------------------------------------------------------------


def two_command_request(answer="### 1. find the order\nThe order total is 186.32.\n\n### 2. check the loan\nThe balance is 2,450.75."):
    return AnswerCardRequest(user_message="status", answer=answer, results=[
        ResultData(id="r1", command="Orders/get_order_details", output={"status": "delivered", "total": 186.32}),
        ResultData(id="r2", command="Loans/get_loan", output={"due": "2026-05-01", "balance": 2450.75}),
    ])


def sec(heading="", verdict="", *blocks):
    return {"heading": heading, "verdict": verdict, "blocks": list(blocks)}


STATS_R1 = {"kind": "stats", "stats": [{"label": "Total", "ref": "r1.total"}]}
STATS_R2 = {"kind": "stats", "stats": [{"label": "Balance", "ref": "r2.balance"}]}


def test_a_grounded_section_keeps_heading_verdict_and_blocks_and_the_flat_blocks_follow():
    request = two_command_request()
    card = validate_answer_card(AnswerCard(lead="Two items.", sections=[
        sec("Order status", "The order total is 186.32.", STATS_R1),
        sec("Loan", "The balance is 2,450.75.", STATS_R2),
    ]), request)
    assert card.v == 2
    assert [(s.heading, s.verdict, [b.kind for b in s.blocks]) for s in card.sections] == [
        ("Order status", "The order total is 186.32.", ["stats"]), ("Loan", "The balance is 2,450.75.", ["stats"])]
    assert [stat.ref for b in card.blocks for stat in b.stats] == ["r1.total", "r2.balance"]       # flat, for an old page
    dumped = card.model_dump(mode="json")
    assert set(dumped) == {"v", "lead", "sections", "blocks"} and dumped["blocks"] == [
        b for s in dumped["sections"] for b in s["blocks"]]


def test_an_ungrounded_verdict_is_dropped_and_an_ungrounded_heading_falls_back():
    request = two_command_request()
    card = validate_answer_card(AnswerCard(sections=[
        sec("Order 999 status", "The order total is 999.99.", STATS_R1),       # both invent a number
        sec("Loan 88 summary", "", STATS_R2),
    ]), request)
    first, second = card.sections
    assert first.verdict == "" and second.verdict == ""
    assert first.heading == "find the order" and second.heading == "check the loan"    # the answer's ### N. titles


def test_a_heading_without_a_sub_task_falls_back_to_the_first_valid_refs_command():
    request = two_command_request(answer="One plain answer without numbered headings. The order total is 186.32.")
    card = validate_answer_card(AnswerCard(sections=[
        sec("", "", {"kind": "record", "ref": "r9.nope"}, STATS_R2),                  # the bad ref is skipped
        sec("   ", "The order total is 186.32.", STATS_R1),
    ]), request)
    assert [s.heading for s in card.sections] == ["Get Loan", "Get Order Details"]
    # no blocks left at all, so nothing to name: a section with a grounded verdict keeps an empty heading
    only_verdict = validate_answer_card(AnswerCard(sections=[sec("", "The order total is 186.32.")]), request)
    assert only_verdict.sections[0].heading == "" and only_verdict.sections[0].blocks == []


def test_the_heading_fallback_follows_the_sections_position_in_the_models_order():
    request = two_command_request()
    card = validate_answer_card(AnswerCard(sections=[sec("x", "", {"kind": "record", "ref": "r9.nope"}), sec("", "", STATS_R2)]), request)
    # the first section is empty and dropped; the second one was written second, so it takes sub-task 2
    assert [s.heading for s in card.sections] == ["check the loan"]


def test_a_section_with_no_verdict_and_no_surviving_block_is_dropped():
    request = two_command_request()
    card = validate_answer_card(AnswerCard(lead="Hi", sections=[
        sec("Empty", ""), sec("Bad ref", "Total 999.", {"kind": "record", "ref": "r1.invented"}),
        sec("Good", "", STATS_R1)]), request)
    assert [s.heading for s in card.sections] == ["Good"]


def test_sections_are_capped_at_five_with_four_blocks_each_and_ten_in_all():
    results = [ResultData(id=f"r{i}", command=f"cmd_{i}", output={"v": i, "w": i + 100}) for i in range(1, 15)]
    request = AnswerCardRequest(answer="Done.", results=results)
    stat = lambda i: {"kind": "stats", "stats": [{"label": f"L{i}", "ref": f"r{i}.v"}]}  # noqa: E731
    six = [sec(f"S{i}", "", stat(i)) for i in range(1, 7)]
    assert len(validate_answer_card(AnswerCard(sections=six), request).sections) == 5
    one = [sec("Big", "", *[{"kind": "record", "ref": f"r{i}"} for i in range(1, 9)])]
    assert len(validate_answer_card(AnswerCard(sections=one), request).sections[0].blocks) == 4
    # 5 sections of 4 would be 20 blocks: the card stops at 10, the later sections keep what room is left
    wide = [sec(f"S{i}", "ok", *[{"kind": "record", "ref": f"r{i * 4 + j + 1}"} for j in range(4)]) for i in range(5)]
    card = validate_answer_card(AnswerCard(sections=wide), request)
    assert [len(s.blocks) for s in card.sections] == [4, 4, 2, 0, 0] and len(card.blocks) == 10
    long_head = validate_answer_card(AnswerCard(sections=[sec("w" * 100, "v. " + "y" * 300, STATS_R1)]), two_command_request())
    assert len(long_head.sections[0].heading) <= 60 and long_head.sections[0].verdict == "v."


def test_a_verdict_is_one_sentence_of_at_most_200_characters():
    request = two_command_request()
    card = validate_answer_card(AnswerCard(sections=[
        sec("A", "The order total is 186.32. The balance is 2,450.75. More text follows.", STATS_R1)]), request)
    assert card.sections[0].verdict == "The order total is 186.32."
    long = validate_answer_card(AnswerCard(sections=[sec("A", "word " * 80, STATS_R1)]), request)
    assert len(long.sections[0].verdict) <= 200


def test_no_surviving_section_falls_back_to_the_flat_card_or_not_needed():
    request = two_command_request()
    junk = [sec("Nothing", "", {"kind": "record", "ref": "r9.nope"})]
    card = validate_answer_card(AnswerCard(lead="Two items.", sections=junk, blocks=[STATS_R1]), request)
    assert card.sections == [] and [b.kind for b in card.blocks] == ["stats"]
    assert validate_answer_card(AnswerCard(lead="Two items.", sections=junk), request) is None   # lead alone, short answer
    card, by_design = gui._validated_card(AnswerCard(lead="Short.", sections=junk), request)
    assert card is None and by_design is True
    assert validate_answer_card(AnswerCard(lead="", sections=junk), request) is None


def test_a_bare_block_in_sections_becomes_a_section_of_its_own():
    request = two_command_request(answer="Plain answer.")
    card = validate_answer_card(AnswerCard(sections=[STATS_R1, "junk", 7, sec("Loan", "", STATS_R2)]), request)
    assert [s.heading for s in card.sections] == ["Get Order Details", "Loan"]


def test_a_table_is_shown_once_across_sections():
    request = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", output={"orders": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]})])
    table = {"kind": "table", "ref": "r1.orders"}
    card = validate_answer_card(AnswerCard(sections=[sec("One", "", table), sec("Two", "", table)]), request)
    assert [s.heading for s in card.sections] == ["One"]


def test_compose_answer_card_end_to_end_returns_wire_v2(composer):
    composer([{"lead": "Two items.", "sections": [
        {"heading": "Order", "verdict": "The order total is 186.32.", "blocks": [STATS_R1]},
        {"heading": "Loan", "verdict": "", "blocks": [STATS_R2]}]}])
    card = gui.compose_answer_card(two_command_request())
    dumped = card.model_dump(mode="json")
    assert dumped["v"] == 2 and len(dumped["sections"]) == 2 and len(dumped["blocks"]) == 2
    # no section at all and only a lead on a short answer: the flat rule applies, nothing is shown
    composer([{"lead": "Two items.", "sections": []}])
    assert gui.compose_answer_card(two_command_request()) is gui.NOT_NEEDED


PAGE_REQUEST = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", output={
    "permission_uids": ["p1", "p2", "p3"], "labels": ["Admin", "Reader", "Writer"], "total": 3})])


def test_a_table_ref_naming_one_column_of_a_page_is_the_table_of_that_page():
    """The digest shows a page as one line of column keys, so ``r1.labels`` is the ref a small model forms."""
    for ref in ("r1.labels", "r1[\"labels\"]", "labels"):
        card = validate_answer_card(AnswerCard(sections=[sec("Rights", "", {"kind": "table", "ref": ref})]), PAGE_REQUEST)
        assert card is not None, ref
        block = card.sections[0].blocks[0]
        assert (block.kind, block.ref, block.fields) == ("table", "r1", []), ref   # one column is no table: all are shown
    # a list that is not a column of a page stays dropped
    plain_lists = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", output={"tags": ["a", "b"], "ids": ["x"]})])
    assert validate_answer_card(AnswerCard(sections=[sec("T", "", {"kind": "table", "ref": "r1.tags"})]), plain_lists) is None
    # the page is still shown once
    twice = validate_answer_card(AnswerCard(sections=[
        sec("One", "", {"kind": "table", "ref": "r1.labels"}), sec("Two", "", {"kind": "table", "ref": "r1"})]), PAGE_REQUEST)
    assert [s.heading for s in twice.sections] == ["One"]


# ---- fix-weih.25: the composer is diagnosable (counts and reason codes; raw output at DEBUG only) ----

def _composed(composer, sections, lead="", request=None):
    composer([{"lead": lead, "sections": sections}])
    diag = gui.ComposeDiagnostics()
    card = gui.compose_answer_card(request or orders_request(), diagnostics=diag)
    return card, diag


def test_diagnostics_tally_a_mixed_card_by_reason_code(composer):
    sections = [
        {"heading": "Good", "verdict": "The case is open.", "blocks": [{"kind": "record", "ref": "r1.status"}]},
        {"heading": "Bad ref", "verdict": "", "blocks": [{"kind": "table", "ref": "r9.nothing"}]},
        {"heading": "Ungrounded", "verdict": "There are 987 orders.",
         "blocks": [{"kind": "record", "ref": "r1.customer"}]},
        {"heading": "Empty", "verdict": "", "blocks": []},
    ]
    card, diag = _composed(composer, sections, lead="Case 12345 is fine.")
    assert card is not None and len(card.sections) == 2
    assert (diag.sections_proposed, diag.sections_kept) == (4, 2)
    assert (diag.blocks_proposed, diag.blocks_kept) == (3, 2)
    assert diag.lead == "dropped"
    assert diag.drops == {"lead_ungrounded": 1, "ref_unresolved": 1, "verdict_ungrounded": 1, "empty_section": 2}
    assert diag.results == 1 and diag.digest_chars > 0 and diag.raw_sections_chars > 0
    assert "sections=2/4 blocks=2/3 lead=dropped" in diag.summary()
    assert "empty_section:2" in diag.drops_text()


def test_diagnostics_record_a_lead_alone_with_results_and_no_card(composer):
    card, diag = _composed(composer, [], lead="Two orders are open.", request=orders_request(LONG))
    assert card is gui.NOT_NEEDED
    assert diag.drops == {"lead_only_with_results": 1} and diag.lead == "kept"
    assert (diag.sections_proposed, diag.sections_kept, diag.blocks_kept) == (0, 0, 0)


def test_diagnostics_record_parse_failures_and_a_failed_model(composer):
    card, diag = _composed(composer, "not a list")      # the adapter cannot parse it: the call failed
    assert card is None and diag.drops == {"model_failed": 1}
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        gui._note_raw_sections(7, [], diag)
    finally:
        gui.DIAGNOSTICS.reset(token)
    assert diag.drops == {"sections_not_a_list": 1} and diag.sections_proposed == 0
    card, diag = _composed(composer, ["text", {"verdict": "The case is open.", "blocks": ["x", 3]}])
    assert diag.drops["section_not_a_dict"] == 1 and diag.drops["block_not_a_dict"] == 2
    assert diag.sections_proposed == 2 and diag.sections_kept == 1


def test_raw_output_and_drop_details_reach_only_the_debug_log(composer, caplog):
    sections = [{"heading": "H", "verdict": "", "blocks": [{"kind": "table", "ref": "r9.zzz_marker"}]}]
    with caplog.at_level(logging.INFO, logger="fastWorkflow"):
        _composed(composer, sections, lead="LEADMARKER")
    assert not [r for r in caplog.records if "zzz_marker" in r.getMessage() or "LEADMARKER" in r.getMessage()]
    caplog.clear()
    composer([{"lead": "LEADMARKER", "sections": sections}])
    with caplog.at_level(logging.DEBUG, logger="fastWorkflow"):
        gui.compose_answer_card(orders_request(), diagnostics=gui.ComposeDiagnostics())
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("raw lead" in m and "LEADMARKER" in m for m in debug)
    assert any("raw sections" in m and "zzz_marker" in m for m in debug)
    assert any("reason=ref_unresolved" in m and "r9.zzz_marker" in m for m in debug)
    assert all(len(m) < 4200 for m in debug)


# ---- fix-weih.25: a model-written label must be supported by the data it is bound to ----


def _listing_request():
    return AnswerCardRequest(answer="Ann holds 32 rights; 4 findings.", results=[
        ResultData(id="r1", command="Identity/list_entitlements", context="Identity: Ann",
                   output={"total": 32, "uid": "u1"},
                   titles={"total": "Total", "uid": "Identity UID"},
                   descriptions={"total": "How many entitlements the backend reported."}),
        ResultData(id="r2", command="Resource/propose_fix", output={"finding_total": 4, "n_open": 2},
                   titles={"finding_total": "Finding Total"},
                   descriptions={"finding_total": "How many control findings the backend reported."}),
    ])


def _stat_labels(stats, request=None):
    block = {"kind": "stats", "stats": stats}
    card = validate_answer_card(AnswerCard(blocks=[block]), request or _listing_request())
    return [s.label for s in card.blocks[0].stats]


def _with_diag(fn):
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        return fn(), diag
    finally:
        gui.DIAGNOSTICS.reset(token)


def test_a_stat_label_naming_another_results_thing_is_replaced_and_recorded():
    labels, diag = _with_diag(lambda: _stat_labels([{"label": "Findings", "ref": "r1.total"}]))
    assert labels == ["List Entitlements: Total"]
    assert diag.drops == {"stat_label_unsupported": 1}


def test_a_stat_label_supported_by_key_title_description_command_or_context_is_kept():
    labels, diag = _with_diag(lambda: _stat_labels([
        {"label": "Findings", "ref": "r2.finding_total"},          # key, plural against singular
        {"label": "Identity UIDs", "ref": "r1.uid"},               # schema title
        {"label": "Entitlements held", "ref": "r1.total"},         # description and command, plural
        {"label": "Open", "ref": "r2.n_open"},                     # key split on _
    ]))
    more, diag2 = _with_diag(lambda: _stat_labels([
        {"label": "Ann", "ref": "r1.total"},                       # context name
        {"label": "Fix", "ref": "r2.n_open"},                      # command name
    ]))
    assert labels == ["Findings", "Identity UIDs", "Entitlements held", "Open"] and more == ["Ann", "Fix"]
    assert diag2.drops == {}
    assert diag.drops == {}


def test_a_generic_only_stat_label_is_qualified_by_the_command_and_a_specific_title_is_used_as_is():
    labels, diag = _with_diag(lambda: _stat_labels([
        {"label": "Total", "ref": "r1.total"},
        {"label": "Count of all", "ref": "r2.n_open"},
        {"label": "Rights", "ref": "r2.finding_total"},        # unsupported: the specific schema title
    ]))
    assert labels == ["List Entitlements: Total", "Propose Fix: N Open", "Finding Total"]
    assert diag.drops == {"stat_label_unsupported": 1}


def test_a_salvaged_scalar_block_title_that_the_field_does_not_support_gets_the_derived_label():
    block = {"kind": "record", "title": "Findings", "ref": "r1.total"}
    card = validate_answer_card(AnswerCard(blocks=[block]), _listing_request())
    assert [s.label for s in card.blocks[0].stats] == ["List Entitlements: Total"]


def test_a_table_title_the_data_does_not_support_is_dropped_and_recorded():
    request = AnswerCardRequest(answer="Two orders.", results=[ResultData(
        id="r1", command="list_orders", output={"orders": [{"id": "A1", "total": 5.0}, {"id": "B2", "total": 7.5}]},
        titles={"orders": "Orders"}, descriptions={"id": "The order reference."}),
        ResultData(id="r2", command="list_refunds", output={"refunds": [1]})])    # "Refunds" names this one: it competes
    blocks = [{"kind": "table", "ref": "r1.orders", "title": "Refunds"},
              {"kind": "record", "ref": "r1", "title": "Recent orders"},   # command word, plural
              {"kind": "record", "ref": "r1", "title": "Items"}]           # generic only: names nothing wrong
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(blocks=blocks), request))
    assert [(b.kind, b.ref, b.title) for b in card.blocks] == [
        ("table", "r1.orders", ""), ("record", "r1", "Recent orders"), ("record", "r1", "Items")]
    assert diag.drops.get("title_unsupported") == 1


def test_label_replacements_count_in_the_drops_tally(composer):
    sections = [{"heading": "H", "verdict": "", "blocks": [
        {"kind": "stats", "stats": [{"label": "Findings", "ref": "r1.total"}]}]}]
    card, diag = _composed(composer, sections, request=_listing_request())
    assert diag.drops == {"stat_label_unsupported": 1}
    assert "stat_label_unsupported:1" in diag.summary()
    assert card.sections[0].blocks[0].stats[0].label == "List Entitlements: Total"


# ----------------------------------------------------------------------
# fix-weih.25: a number in a lead or a verdict must be anchored (or be part of a name or ID)
# ----------------------------------------------------------------------


def anchor_request(answer, message="how did the review go", output=None):
    return AnswerCardRequest(user_message=message, answer=answer, results=[
        ResultData(id="r1", command="review_findings", output=output or {"proposed": 32, "findings": 4})])


STATS_PROPOSED = {"kind": "stats", "stats": [{"label": "Proposed", "ref": "r1.proposed"}]}


def verdict_of(verdict, request, *blocks):
    card = validate_answer_card(AnswerCard(sections=[sec("Review", verdict, *(blocks or (STATS_PROPOSED,)))]), request)
    return card.sections[0].verdict


@pytest.mark.parametrize("answer, verdict", [
    ("The review found 4 control findings and 32 rights.", "4 findings were proposed."),              # within the window
    ("Findings: 4. Rights: 32.", "There were 4 findings."),                                          # label first
    ("The review has 4 finding to fix.", "4 findings were proposed."),                               # singular / plural
    ("Review: 4 findings, 32 rights.", "Only 4 findings need work."),
])
def test_an_anchored_quantity_in_a_verdict_is_kept(answer, verdict):
    assert verdict_of(verdict, anchor_request(answer)) == verdict


def test_the_users_message_anchors_a_quantity_too():
    request = anchor_request("Done.", message="show me the 4 findings")
    assert verdict_of("4 findings were proposed.", request) == "4 findings were proposed."


def test_an_unanchored_quantity_is_dropped_with_a_reason_and_the_section_keeps_its_blocks():
    request = anchor_request("The review covered 32 rights and 4 control findings.")
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        card = validate_answer_card(AnswerCard(sections=[
            sec("Review", "32 findings were proposed.", STATS_PROPOSED)]), request)
    finally:
        gui.DIAGNOSTICS.reset(token)
    assert card.sections[0].verdict == "" and [b.kind for b in card.sections[0].blocks] == ["stats"]
    assert card.sections[0].heading == "Review"
    assert diag.drops == {"verdict_number_unanchored": 1}


def test_a_number_that_exists_only_in_the_data_does_not_anchor():
    request = anchor_request("Review finished.")             # 32 is in the data, not in the answer
    assert verdict_of("32 proposals were made.", request) == ""


def test_a_section_with_only_an_unanchored_verdict_is_dropped():
    request = anchor_request("The review covered 32 rights and 4 findings.")   # 4 findings: "32 findings" is contradicted
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        card = validate_answer_card(AnswerCard(lead="Reviewed.", sections=[
            sec("Review", "32 findings were proposed.")]), request)
    finally:
        gui.DIAGNOSTICS.reset(token)
    assert card is None
    assert diag.drops["verdict_number_unanchored"] == 1 and diag.drops["empty_section"] == 1


def test_a_number_that_is_part_of_a_name_is_kept():
    request = anchor_request("Done.", output={"proposed": 7, "user": "Administrator 1", "role": "Administrator 12"})
    assert verdict_of("Administrator 1 owns the account.", request) == "Administrator 1 owns the account."
    # "Administrator 1" is a name here only because that exact phrase is in the data; a lone "1" is a quantity
    assert verdict_of("Administrator 1 owns 1 account.", request) == ""
    # the phrase must match whole: "Administrator 12" is not "Administrator 1"
    assert verdict_of("Administrator 2 owns the account.", request) == ""


def test_a_generic_word_before_a_number_is_not_a_name():
    request = anchor_request("Total 32 rights and 4 findings.")
    assert verdict_of("Total 32 findings.", request) == ""


def test_ids_and_ordinals_stay_under_the_id_grounding():
    request = anchor_request("Order #A1 is delivered; uid user_88 and the 3rd review.")
    assert verdict_of("Order #A1 is delivered.", request) == "Order #A1 is delivered."
    assert verdict_of("The user_88 account was reviewed.", request) == "The user_88 account was reviewed."
    assert verdict_of("The 3rd review passed.", request) == "The 3rd review passed."
    assert verdict_of("Order #A9 is delivered.", request) == ""             # still ungrounded


def test_a_date_is_kept_only_when_copied_whole():
    request = anchor_request("The review was due 2026-05-01.", output={"proposed": 7, "due": "2026-05-01"})
    assert verdict_of("The review was due 2026-05-01.", request) == "The review was due 2026-05-01."
    assert verdict_of("The review was due 2026-05-02.", request) == ""


def test_a_lead_with_an_anchored_quantity_is_kept_and_an_unanchored_one_is_dropped():
    request = anchor_request("The review found 4 control findings and 32 rights.")
    block = {"kind": "record", "ref": "r1"}
    kept = validate_answer_card(AnswerCard(lead="4 findings were proposed.", blocks=[block]), request)
    assert kept.lead == "4 findings were proposed."
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        dropped = validate_answer_card(AnswerCard(lead="Done. 32 findings were proposed.", blocks=[block]), request)
    finally:
        gui.DIAGNOSTICS.reset(token)
    assert dropped.lead == "" and [b.kind for b in dropped.blocks] == ["record"]
    assert diag.drops == {"lead_number_unanchored": 1} and diag.lead == "dropped"


def test_a_lead_dropped_for_a_number_leaves_a_lead_less_card_to_the_existing_rules():
    request = anchor_request("The review covered 32 rights.")
    assert validate_answer_card(AnswerCard(lead="32 findings were proposed."), request) is None   # nothing else left


@pytest.mark.parametrize("answer", [
    "The review found 4 control findings and 32 rights.",
    "Findings: 4, Rights: 32.",
    "findings 4 | rights 32",
    "Findings: 4. Rights: 32.",
])
def test_a_word_belongs_to_the_nearest_number_so_a_swapped_noun_is_dropped(answer):
    request = anchor_request(answer)
    assert verdict_of("4 findings were proposed.", request) == "4 findings were proposed."
    assert verdict_of("32 rights were checked.", request) == "32 rights were checked."
    assert verdict_of("32 findings were proposed.", request) == ""
    assert verdict_of("4 rights were checked.", request) == ""


def test_the_swap_is_caught_on_the_models_side_too():
    request = anchor_request("The review found 4 control findings and 32 rights.")
    assert verdict_of("4 rights and 32 findings were reviewed.", request) == ""
    assert verdict_of("4 findings and 32 rights were reviewed.", request) == "4 findings and 32 rights were reviewed."


# ----------------------------------------------------------------------
# fix-weih.27: one correction pass when validation drops something a second look can fix
# ----------------------------------------------------------------------


class SentLM(RecordingLM):
    """A RecordingLM that keeps the messages of every call (shared by its copies)."""

    def __init__(self, answers):
        super().__init__(answers)
        self.sent = []

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied.sent = self.sent
        return copied

    def forward(self, prompt=None, messages=None, **kwargs):
        self.sent.append(json.dumps(messages))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def sent_composer(monkeypatch):
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())

    def install(answers):
        lm = SentLM(answers)
        monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
        return lm
    return install


GOOD_SECTION = {"heading": "Order", "verdict": "", "blocks": [{"kind": "record", "ref": "r1.status"}]}
BAD_BLOCK = {"kind": "record", "ref": "r1.owner"}
FIRST_WITH_A_BAD_REF = {"lead": "Your order is delivered.", "sections": [
    {"heading": "Order", "verdict": "", "blocks": [{"kind": "record", "ref": "r1.status"}, BAD_BLOCK]}]}
CLEAN_SECOND = {"lead": "Your order is delivered.", "sections": [GOOD_SECTION]}


def run_card(lm_answers, sent_composer, request=None):
    lm = sent_composer(lm_answers)
    diag = gui.ComposeDiagnostics()
    card = gui.compose_answer_card(request or card_request(), diag)
    return lm, diag, card


def test_nothing_dropped_means_one_call(sent_composer):
    lm, diag, card = run_card([CLEAN_SECOND, CLEAN_SECOND], sent_composer)
    assert len(lm.sent) == 1 and diag.retry == "none" and diag.second is None and card is not None
    assert "retry=none" in diag.summary() and "drops2" not in diag.summary()


def test_only_caps_dropped_means_one_call(sent_composer):
    capped = {"lead": "", "sections": [
        {"heading": f"S{i}", "verdict": "", "blocks": [{"kind": "stats", "stats": [{"label": "", "ref": f"r{i}.v"}]}]}
        for i in range(1, 8)]}
    request = AnswerCardRequest(answer="Done.", results=[
        ResultData(id=f"r{i}", command=f"cmd_{i}", output={"v": i, "w": i + 100}) for i in range(1, 9)])
    lm, diag, card = run_card([capped, capped], sent_composer, request)
    assert diag.drops == {"section_cap": 2} and len(lm.sent) == 1 and diag.retry == "none"
    assert not (set(diag.drops) & gui.CORRECTABLE_DROPS) and "section_cap" in gui.NOT_CORRECTABLE_DROPS


def test_a_correctable_drop_makes_a_second_call_with_the_facts_and_a_better_second_answer_is_kept(sent_composer):
    lm, diag, card = run_card([FIRST_WITH_A_BAD_REF, CLEAN_SECOND], sent_composer)
    assert len(lm.sent) == 2 and diag.retry == "used"
    assert diag.drops == {"ref_unresolved": 1} and diag.second.drops == {}
    assert len(card.blocks) == 1
    expected = ("- ref r1.owner does not exist; r1 has fields: order_id, status, total, items, payment_methods, lines")
    assert gui.corrections_text(diag, card_request()) == expected
    assert "corrections" not in lm.sent[0] and expected.lstrip("- ") in lm.sent[1]
    assert "retry=used drops2=none sections2=1/1 blocks2=1/1" in diag.summary()


def test_the_first_calls_prompt_is_the_same_with_and_without_the_retry_machinery(sent_composer, monkeypatch):
    first_lm, _, _ = run_card([FIRST_WITH_A_BAD_REF, CLEAN_SECOND], sent_composer)
    monkeypatch.setattr(gui, "RETRY_ENABLED", False)
    plain_lm, _, _ = run_card([FIRST_WITH_A_BAD_REF], sent_composer)
    assert first_lm.sent[0] == plain_lm.sent[0]
    assert list(gui.ComposeAnswerCard.input_fields) == ["user_message", "answer", "sub_tasks", "results"]


def test_a_worse_or_equal_second_answer_keeps_the_first(sent_composer):
    worse = {"lead": "", "sections": [{"heading": "x", "verdict": "", "blocks": [BAD_BLOCK]}]}
    lm, diag, card = run_card([FIRST_WITH_A_BAD_REF, worse], sent_composer)
    assert len(lm.sent) == 2 and diag.retry == "kept_first" and len(card.blocks) == 1 and card.lead
    lm, diag, card = run_card([FIRST_WITH_A_BAD_REF, FIRST_WITH_A_BAD_REF], sent_composer)
    assert diag.retry == "kept_first" and len(lm.sent) == 2               # equal is not better


def test_a_second_call_that_fails_keeps_the_first_silently(sent_composer, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="fastWorkflow"):
        lm, diag, card = run_card([FIRST_WITH_A_BAD_REF], sent_composer)      # the script has no second answer
    assert diag.retry == "failed" and card is not None and len(card.blocks) == 1 and diag.second is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_slow_first_pass_skips_the_retry(sent_composer, monkeypatch):
    monkeypatch.setattr(gui, "RETRY_SKIP_AFTER_SECONDS", -1)
    lm, diag, card = run_card([FIRST_WITH_A_BAD_REF, CLEAN_SECOND], sent_composer)
    assert len(lm.sent) == 1 and diag.retry == "skipped_slow" and "retry=skipped_slow" in diag.summary()


def test_the_retry_budget_is_derived_from_the_page_and_server_constants():
    assert gui.RETRY_SKIP_AFTER_SECONDS == gui.PAGE_GIVE_UP_SECONDS / 2 < gui.SERVER_COMPOSE_BUDGET_SECONDS


def test_a_failed_first_call_is_never_retried(sent_composer):
    lm, diag, card = run_card([], sent_composer)
    assert card is None and diag.drops == {"model_failed": 1} and diag.retry == "none"


def test_corrections_name_only_facts_the_digest_already_shows_and_stay_under_800_characters(sent_composer):
    request = card_request(answer="Order #W2041120 is delivered. Total 186.32 and 4 findings.")
    diag = gui.ComposeDiagnostics()
    token = gui.DIAGNOSTICS.set(diag)
    try:
        card = AnswerCard(lead="186.32 findings", sections=[
            sec("H", "186.32 findings were proposed.", {"kind": "stats", "stats": [{"label": "Findings", "ref": "r1.total"}]},
                {"kind": "quote", "quote": "words that the answer never said anywhere at all"},
                {"kind": "table", "ref": "r1.status"}),
            sec("Gone", "", {"kind": "table", "ref": f"r1.zz{'z' * 60}"})])
        validate_answer_card(card, request)
    finally:
        gui.DIAGNOSTICS.reset(token)
    text = gui.corrections_text(diag, request)
    assert "label 'Findings' does not describe r1.total (total)" in text
    assert "quote 'words that the answer never said anywhere at all' is not in the answer" in text
    assert "verdict '186.32 findings were proposed.' uses a number the answer does not state that way; say it in words or use a stats block" in text
    assert "lead '186.32 findings' uses a number the answer does not state that way" in text
    assert "delivered" not in text and "W2041120" not in text and len(text) <= gui.CORRECTIONS_MAX_CHARS
    assert len(diag.details) <= gui.DROP_DETAILS_MAX


def test_the_drop_details_are_kept_on_the_record_but_not_on_the_info_summary():
    diag = gui.ComposeDiagnostics()
    for i in range(20):
        diag.add("ref_unresolved", f"r9.f{i}", f"r9.f{i}")
    assert len(diag.details) == gui.DROP_DETAILS_MAX and diag.drops == {"ref_unresolved": 20}
    assert "r9.f0" not in diag.summary()


# ----------------------------------------------------------------------
# fix-weih.25 live-check defects G1-G6 (invented data; the shapes are what a small model really wrote)
# ----------------------------------------------------------------------

SEC_A = sec("Orders", "", {"kind": "record", "ref": "r1.status"})
SEC_B = sec("Customer", "", {"kind": "record", "ref": "r1.customer"})
BARE_BLOCK = {"kind": "record", "ref": "r1.count"}


def test_g1_a_sections_value_nested_one_level_too_deep_is_flattened_and_is_not_a_drop(composer):
    card, diag = _composed(composer, [[SEC_A, SEC_B], [['']]])
    assert card is not None and [s.heading for s in card.sections] == ["Orders", "Customer"]
    assert (diag.sections_proposed, diag.sections_kept, diag.sections_unnested) == (2, 2, 1)
    assert diag.drops == {} and diag.correctable() == 0 and "unnested=1" in diag.summary()
    assert diag.retry == "none"


def test_g1_a_stray_block_inside_the_nested_list_becomes_a_section_and_later_siblings_are_kept(composer):
    card, diag = _composed(composer, [[SEC_A, BARE_BLOCK], SEC_B, "", []])
    assert card is not None and len(card.sections) == 3
    assert (diag.sections_proposed, diag.sections_unnested, diag.drops) == (3, 1, {})


def test_g1_only_one_list_level_is_flattened_and_a_non_dict_left_is_still_dropped(composer):
    card, diag = _composed(composer, [["just text", SEC_A], [[SEC_B]]])
    assert [s.heading for s in card.sections] == ["Orders"]
    assert diag.drops == {"section_not_a_dict": 2} and diag.sections_unnested == 2
    assert diag.correctable() == 2


def test_g1_unnest_sections_ignores_stray_empty_elements():
    assert gui.unnest_sections([[{"a": 1}], "", None, [], [[""]], [""], {"b": 2}]) == ([{"a": 1}, {"b": 2}], 1)


def _remaining_request(answer="Group Alpha has 167 holders, listed 60 per page."):
    return AnswerCardRequest(answer=answer, results=[ResultData(
        id="r1", command="Group/list_holders", output={"total": 167, "remaining": 107, "labels": [f"h{i}" for i in range(60)]},
        titles={"total": "Holders", "remaining": "Holders not shown", "labels": "Holders"},
        descriptions={"remaining": "How many holders the page does not show."})])


def test_g2_a_label_that_drops_the_negation_of_the_field_is_replaced_and_recorded():
    labels, diag = _with_diag(lambda: _stat_labels([
        {"label": "Shown", "ref": "r1.remaining"},                 # shares "shown", loses "not"
        {"label": "Shown holders", "ref": "r1.remaining"},
    ], _remaining_request()))
    assert labels == ["Holders not shown", "Holders not shown"]
    assert diag.drops == {"stat_label_unsupported": 2}


def test_g2_a_shortening_and_a_label_with_the_same_qualifier_are_kept_and_the_converse_fails():
    labels, diag = _with_diag(lambda: _stat_labels([
        {"label": "Holders", "ref": "r1.remaining"},               # a plain shortening: no word is reversed
        {"label": "Not shown", "ref": "r1.remaining"},
        {"label": "Holders not shown", "ref": "r1.remaining"},
        {"label": "Remaining", "ref": "r1.remaining"},             # the key itself
    ], _remaining_request()))
    assert labels == ["Holders", "Not shown", "Holders not shown", "Remaining"] and diag.drops == {}
    request = AnswerCardRequest(answer="Done.", results=[ResultData(
        id="r1", command="Group/list_holders", output={"shown": 60}, titles={"shown": "Holders shown"})])
    labels, diag = _with_diag(lambda: _stat_labels([{"label": "Holders not shown", "ref": "r1.shown"}], request))
    assert labels == ["Holders shown"] and diag.drops == {"stat_label_unsupported": 1}


def test_g3_a_page_size_the_data_supports_anchors_a_verdict_the_answer_words_differently():
    request = _remaining_request()
    block = {"kind": "record", "ref": "r1.total"}
    assert verdict_of("Group Alpha has 167 holders; 60 shown.", request, block) == "Group Alpha has 167 holders; 60 shown."


def test_g3_a_count_field_value_anchors_when_the_sentence_owns_a_word_of_that_field():
    request = _remaining_request()
    block = {"kind": "record", "ref": "r1.total"}
    assert verdict_of("There are 107 holders hidden.", request, block) == "There are 107 holders hidden."


def test_g3_numbers_found_nowhere_or_on_the_wrong_noun_stay_unanchored():
    request = _remaining_request()
    block = {"kind": "record", "ref": "r1.total"}
    assert verdict_of("Group Alpha has 167 holders; 61 shown.", request, block) == ""   # 61 is in no list, no field
    assert verdict_of("There are 107 findings.", request, block) == ""                  # 107 belongs to holders
    assert verdict_of("The group lists 60 invoices.", request, block) == ""              # names nothing the list holds


def _risk_request():
    return AnswerCardRequest(answer="The aggregate risk score is 4000100.", results=[ResultData(
        id="r4", command="Identity/assess_risk", output={"score": 4000100, "total": 5},
        titles={"score": "Score", "total": "Findings"})])


def test_g4_a_block_whose_label_code_wrote_loses_a_title_that_names_what_the_data_does_not():
    block = {"kind": "stats", "title": "Risk level", "stats": [{"label": "Level", "ref": "r4.score"}]}
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(blocks=[block]), _risk_request()))
    assert card.blocks[0].title == "" and [s.label for s in card.blocks[0].stats] == ["Score"]
    assert diag.drops == {"stat_label_unsupported": 1, "title_unsupported": 1}


def test_g4_a_title_over_a_supported_label_is_judged_as_before_and_a_reversed_title_goes_with_its_label():
    ok = {"kind": "stats", "title": "Risk level", "stats": [{"label": "Score", "ref": "r4.score"}]}
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(blocks=[ok]), _risk_request()))
    assert card.blocks[0].title == "Risk level" and diag.drops == {}
    reversed_ = {"kind": "stats", "title": "Shown holders", "stats": [{"label": "Shown", "ref": "r1.remaining"}]}
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(blocks=[reversed_]), _remaining_request()))
    assert card.blocks[0].title == "" and card.blocks[0].stats[0].label == "Holders not shown"
    assert diag.drops == {"stat_label_unsupported": 1, "title_unsupported": 1}


HEX_ID = "6f37f40029429a3e4b28a925da45affb"
ID_ANSWER = (
    "The review is done. Note: the group published by the Alpha application (permission "
    f"{HEX_ID}) has sensitivity level four. Note: the Alpha application publishes the group to every site. "
    + "Further detail follows for each of the sub-tasks that the request listed. " * 6)


def _quote_request(hidden=()):
    return AnswerCardRequest(answer=ID_ANSWER, results=[ResultData(
        id="r1", command="list_things", output={"label": "Alpha", "ref_code": "ACC-77-XYZ"},
        hints={"ref_code": "hidden"} if hidden else {})])


def _quote_card(quote, request):
    return _with_diag(lambda: validate_answer_card(
        AnswerCard(sections=[sec("Q", "", {"kind": "quote", "quote": quote}, {"kind": "record", "ref": "r1.label"})]),
        request))


def test_g6_a_quote_that_carries_an_identifier_is_dropped_and_recorded():
    quote = f"the group published by the Alpha application (permission {HEX_ID}) has sensitivity level four."
    card, diag = _quote_card(quote, _quote_request())
    assert [b.kind for b in card.sections[0].blocks] == ["stats"] and diag.drops == {"quote_identifier": 1}
    plain = "the Alpha application publishes the group to every site."
    card, diag = _quote_card(plain, _quote_request())
    assert [b.kind for b in card.sections[0].blocks] == ["quote", "stats"] and diag.drops == {}


def test_g6_a_quote_that_holds_a_value_of_a_hidden_field_is_dropped_whatever_its_shape():
    answer = ("Report done. Note: the code ACC-77-XYZ was given to the site and the Alpha application kept it. "
              + "More detail is in the sub-tasks. " * 10)
    request = AnswerCardRequest(answer=answer, results=[ResultData(
        id="r1", command="list_things", output={"label": "Alpha", "ref_code": "ACC-77-XYZ"}, hints={"ref_code": "hidden"})])
    quote = "the code ACC-77-XYZ was given to the site and the Alpha application kept it."
    card, diag = _with_diag(lambda: validate_answer_card(
        AnswerCard(sections=[sec("Q", "", {"kind": "quote", "quote": quote}, {"kind": "record", "ref": "r1.label"})]), request))
    assert diag.drops == {"quote_identifier": 1}
    shown = AnswerCardRequest(answer=answer, results=[ResultData(
        id="r1", command="list_things", output={"label": "Alpha", "ref_code": "ACC-77-XYZ"})])
    card, diag = _with_diag(lambda: validate_answer_card(
        AnswerCard(sections=[sec("Q", "", {"kind": "quote", "quote": quote}, {"kind": "record", "ref": "r1.label"})]), shown))
    assert diag.drops == {}


def test_g6_quote_identifier_does_not_trigger_the_correction_pass():
    assert "quote_identifier" in gui.NOT_CORRECTABLE_DROPS and "quote_identifier" not in gui.CORRECTABLE_DROPS


# ----------------------------------------------------------------------
# fix-weih.25 generic composer rules: R1 numbers, R2 titles, R3 retry trigger, R4 cap, R6 siblings
# ----------------------------------------------------------------------


def holders_request(answer, message="who holds the right"):
    return AnswerCardRequest(user_message=message, answer=answer, results=[
        ResultData(id="r1", command="list_holders", output={"total": 167, "remaining": 107, "labels": ["a", "b"]},
                   titles={"total": "Holders", "remaining": "Holders not shown"})])


HOLDERS_STATS = {"kind": "stats", "stats": [{"label": "Holders", "ref": "r1.total"}]}


@pytest.mark.parametrize("answer, verdict", [
    ("The right has 167 holders, 107 not shown.", "The right is held by 167 people."),             # a synonym noun
    ("There are 4 defects at level 4.", "There are four level-4 defects."),                         # word + label
    ("Found four findings.", "4 findings were found."),                                            # word in the answer
    ("Found 4 findings.", "Four findings were found."),                                            # word in the verdict
    ("Twenty entitlements.", "20 entitlements are listed."),
    ("1 account, 30 entitlements.", "The identity holds one account."),
])
def test_r1_a_number_the_answer_states_is_kept_when_no_other_number_owns_the_noun(answer, verdict):
    assert verdict_of(verdict, holders_request(answer), HOLDERS_STATS) == verdict


@pytest.mark.parametrize("answer, verdict", [
    ("The review found 4 findings and 32 rights.", "32 findings were proposed."),                  # the original bug
    ("The review found 4 findings and 32 rights.", "Thirty-two findings were proposed."),          # ... in words
    ("The review found 4 findings and 32 rights.", "Four rights were reviewed."),                  # swapped noun
    ("The review found 4 findings and 32 rights.", "Five findings were proposed."),                # stated nowhere
    ("The review found 4 findings.", "Seven findings were proposed."),
])
def test_r1_a_number_another_number_contradicts_or_nobody_states_is_dropped(answer, verdict):
    assert verdict_of(verdict, anchor_request(answer)) == ""


def test_r1_a_hyphenated_token_is_a_label_and_a_date_still_has_to_be_copied():
    request = anchor_request("Done: 4 defects.")
    assert verdict_of("The level-4 defects were listed.", request) == "The level-4 defects were listed."
    assert verdict_of("A high-risk review.", request) == "A high-risk review."
    assert verdict_of("Due 2026-05-01.", request) == ""


def test_r1_a_loose_one_is_not_a_quantity():
    request = anchor_request("Done: 4 findings.")
    for verdict in ("One of the findings is open.", "No one owns it.", "It is the one that matters."):
        assert verdict_of(verdict, request) == verdict


def test_r1_a_lead_follows_the_same_rule():
    request = holders_request("The right has 167 holders.")
    block = {"kind": "record", "ref": "r1"}
    assert validate_answer_card(AnswerCard(lead="167 people hold it.", blocks=[block]), request).lead == "167 people hold it."
    request = anchor_request("The review found 4 findings and 32 rights.")
    assert validate_answer_card(AnswerCard(lead="32 findings.", blocks=[block]), request).lead == ""
    assert validate_answer_card(AnswerCard(lead="32 rights.", blocks=[block]), request).lead == "32 rights."


def titled(title, request, block):
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(blocks=[{**block, "title": title}]), request))
    return card.blocks[0].title, diag


def test_r2_a_descriptive_title_that_names_no_competing_thing_is_kept():
    request = holders_request("167 holders.")
    for title in ("Reach", "Who can sign in"):
        kept, diag = titled(title, request, {"kind": "record", "ref": "r1"})
        assert kept == title and diag.drops == {}


def test_r2_a_title_that_names_another_result_or_field_is_dropped():
    request = AnswerCardRequest(answer="167 holders.", results=[
        ResultData(id="r1", command="list_holders", output={"total": 167, "remaining": 107}, titles={"total": "Holders"}),
        ResultData(id="r2", command="get_feed", output={"feed": "x", "accounts": 497})])
    stat = {"kind": "stats", "stats": [{"label": "Holders", "ref": "r1.total"}]}
    dropped, diag = titled("Feed", request, stat)                    # names the other result's command
    assert dropped == "" and diag.drops == {"title_unsupported": 1}
    dropped, diag = titled("Accounts", request, stat)                # names a field of the other result
    assert dropped == "" and diag.drops == {"title_unsupported": 1}
    dropped, diag = titled("Remaining", request, {"kind": "record", "ref": "r1", "fields": ["total"]})   # a field of this result the block does not show
    assert dropped == "" and diag.drops == {"title_unsupported": 1}
    kept, diag = titled("Holders on the right", request, stat)       # shares a word with the field
    assert kept == "Holders on the right" and diag.drops == {}


def test_r2_a_title_with_a_number_or_a_negation_keeps_the_strict_test():
    request = holders_request("167 holders.")
    dropped, _ = titled("Top 5", request, {"kind": "record", "ref": "r1"})             # a number that shares no word
    assert dropped == ""
    dropped, _ = titled("Not shown", request, {"kind": "stats", "stats": [{"label": "Holders", "ref": "r1.total"}]})
    assert dropped == ""                                                               # not a qualifier of the field
    kept, _ = titled("Holders not shown", request, {"kind": "stats", "stats": [{"label": "Holders not shown", "ref": "r1.remaining"}]})
    assert kept == "Holders not shown"


def test_r2_a_stat_whose_label_code_wrote_keeps_the_strict_title_test():
    request = holders_request("167 holders.")
    stat = {"kind": "stats", "stats": [{"label": "Rights", "ref": "r1.total"}]}          # label replaced by code
    dropped, diag = titled("Reach", request, stat)
    assert dropped == "" and diag.drops == {"stat_label_unsupported": 1, "title_unsupported": 1}


def test_r3_the_retry_triggers_are_a_lost_section_block_or_lead_and_nothing_smaller():
    assert gui.CORRECTABLE_DROPS.isdisjoint(gui.NOT_CORRECTABLE_DROPS)
    for reason in ("title_unsupported", "verdict_number_unanchored", "verdict_ungrounded", "stat_label_unsupported",
                   "quote_ungrounded", "stat_not_scalar", "stat_ref_unresolved", "sibling_mismatch"):
        assert reason not in gui.CORRECTABLE_DROPS
    for reason in ("ref_unresolved", "not_a_table", "stats_empty", "empty_section", "lead_ungrounded", "lead_number_unanchored"):
        assert reason in gui.CORRECTABLE_DROPS


def test_r3_small_drops_alone_do_not_make_a_second_call(sent_composer):
    request = AnswerCardRequest(answer="167 holders and 4 findings.", results=[
        ResultData(id="r1", command="list_holders", output={"total": 167}, titles={"total": "Holders"}),
        ResultData(id="r2", command="list_findings", output={"count": 4})])
    first = {"lead": "", "sections": [
        {"heading": "Holders", "verdict": "167 findings were found.",                      # 4 findings: contradicted
         "blocks": [{"kind": "stats", "title": "Findings", "stats": [{"label": "Rights", "ref": "r1.total"}]},   # wrong label and title
                    {"kind": "quote", "quote": "words that the answer never said anywhere at all"}]}]}
    lm, diag, card = run_card([first, CLEAN_SECOND], sent_composer, request)
    assert len(lm.sent) == 1 and diag.retry == "none" and card is not None
    assert {"title_unsupported", "stat_label_unsupported", "quote_ungrounded", "verdict_number_unanchored"} <= set(diag.drops)
    assert diag.correctable() == 0


def test_r3_a_lost_lead_or_block_makes_a_second_call_that_also_hears_the_small_drops(sent_composer):
    request = card_request(answer="Order #W2041120 is delivered. Total 186.32.")
    first = {"lead": "There are 987 orders.", "sections": [GOOD_SECTION]}
    lm, diag, _ = run_card([first, CLEAN_SECOND], sent_composer, request)
    assert len(lm.sent) == 2 and diag.correctable() == 1
    first = {"lead": "", "sections": [{"heading": "x", "verdict": "987 orders.", "blocks": [BAD_BLOCK, GOOD_SECTION["blocks"][0]]}]}
    lm, diag, _ = run_card([first, CLEAN_SECOND], sent_composer, request)
    assert len(lm.sent) == 2
    assert "does not exist" in lm.sent[1] and "holds a number or ID" in lm.sent[1]      # the verdict hint rides along


def parts_answer(n):
    return "\n\n".join(f"### {i}. part {i}\nDone {i}." for i in range(1, n + 1))


def parts_request(n, results=None):
    return AnswerCardRequest(answer=parts_answer(n) if n else "Done.", results=results or [
        ResultData(id=f"r{i}", command=f"cmd_{i}", output={"v": i, "w": i + 100}) for i in range(1, 13)])


@pytest.mark.parametrize("parts, cap", [(0, 5), (3, 5), (5, 5), (6, 6), (8, 8), (9, 9), (12, 12), (14, 14), (30, 24)])
def test_r4_the_section_cap_follows_the_parts_of_the_request(parts, cap):
    assert gui.section_cap(parts_request(parts)) == cap
    assert (gui.MIN_SECTIONS, gui.MAX_SECTIONS) == (5, 24)


NAMES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"]


def stat_section(i):
    return sec(NAMES[i - 1], "", {"kind": "stats", "stats": [{"label": f"L{i}", "ref": f"r{i}.v"}]})


def test_r4_an_eight_part_request_keeps_eight_sections_in_order_and_a_ninth_is_cut():
    request = parts_request(8)
    nine = [stat_section(i) for i in range(1, 10)]
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=nine), request))
    assert [s.heading for s in card.sections] == NAMES[:8]
    assert diag.drops == {"section_cap": 1}
    five = validate_answer_card(AnswerCard(sections=nine), parts_request(0))
    assert [s.heading for s in five.sections] == NAMES[:5]
    assert len(card.blocks) == 8 and gui.card_block_cap(5) == 10 and gui.card_block_cap(8) == 16


def test_r4_over_the_cap_a_section_showing_new_data_beats_a_repeat_of_earlier_data():
    request = parts_request(0)
    sections = [stat_section(1), stat_section(1), stat_section(1), stat_section(2), stat_section(3),
                stat_section(4), stat_section(5)]
    sections[1] = sec("Repeat A", "", {"kind": "record", "ref": "r1"})
    sections[2] = sec("Repeat B", "", {"kind": "record", "ref": "r1"})
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=sections), request))
    assert [s.heading for s in card.sections] == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]      # request order
    # fix-weih.25 round 3 (T3): the repeats are no longer cut, their blocks move into the section about r1
    assert diag.drops == {} and diag.sections_merged == 2
    assert [b.kind for b in card.sections[0].blocks] == ["stats", "stats", "stats"] or len(card.sections[0].blocks) == 3


def test_r4_the_instruction_text_states_the_cap_of_this_request(sent_composer):
    for parts, cap in ((0, 5), (7, 7), (12, 12)):        # was (12, 8): the cap no longer stops at 8 (fix-weih.34)
        lm, _, _ = run_card([CLEAN_SECOND], sent_composer, parts_request(parts))
        assert f"Write at most {cap} sections" in lm.sent[0] and f"up to {cap} sections" in lm.sent[0]


def test_r4_the_card_call_runs_with_the_card_token_limit_not_the_form_limit(sent_composer):
    lm, _, _ = run_card([CLEAN_SECOND], sent_composer, parts_request(5))
    assert gui.CARD_MAX_TOKENS == 1800 and lm.copies[-1]["max_tokens"] == 1800        # not COMPOSER_MAX_TOKENS (1024)
    lm, _, _ = run_card([CLEAN_SECOND], sent_composer, parts_request(8))              # 8 parts: scaled (fix-weih.34), was 1800
    assert lm.copies[-1]["max_tokens"] == gui.card_token_budget(8) == 2550 > gui.COMPOSER_MAX_TOKENS


def sibling_request(first_context="Identity: Ann Lee", second_context="Identity: Bob Roy", answer="Ann 3 rights, Bob 5."):
    return AnswerCardRequest(answer=answer, results=[
        ResultData(id="r1", command="get_profile", context=first_context, parameters={"person": "ann"},
                   output={"rights": 3, "team": "red"}),
        ResultData(id="r2", command="get_profile", context=second_context, parameters={"person": "bob"},
                   output={"rights": 5, "team": "blue"}),
        ResultData(id="r3", command="list_teams", output={"teams": ["red", "blue"], "total": 2}),
    ])


def test_r6_every_result_line_of_the_digest_tells_siblings_apart():
    lines = json.loads(results_digest(sibling_request().results))
    labels = {line["id"]: line.get("label", "") for line in lines}
    assert "Ann" in labels["r1"] and "person=ann" in labels["r1"] and "Bob" in labels["r2"] and "person=bob" in labels["r2"]
    assert all(len(label) <= gui.DISCRIMINATOR_MAX_CHARS for label in labels.values())
    assert "label" not in next(line for line in lines if line["id"] == "r3")      # nothing to tell apart


def test_r6_a_hidden_input_never_labels_a_result():
    results = [ResultData(id="r1", command="get_profile", parameters={"token": "sekret-value", "person": "ann"},
                          hints={"token": "hidden"}, output={"rights": 3})]
    line = json.loads(results_digest(results))[0]
    assert "sekret" not in line.get("label", "") and "person=ann" in line["label"]


def test_r6_a_block_bound_to_the_sibling_the_section_does_not_name_is_dropped():
    request = sibling_request()
    block_bob = {"kind": "stats", "stats": [{"label": "Rights", "ref": "r2.rights"}]}
    block_ann = {"kind": "stats", "stats": [{"label": "Rights", "ref": "r1.rights"}]}
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[
        sec("Ann Lee's reach", "She holds 3 rights.", block_bob, block_ann)]), request))
    assert [stat.ref for b in card.sections[0].blocks for stat in b.stats] == ["r1.rights"]
    assert diag.drops == {"sibling_mismatch": 1} and diag.correctable() == 0      # one of two blocks: not correctable


def test_r6_a_sole_block_lost_to_a_sibling_mismatch_is_correctable():
    request = sibling_request()
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[
        sec("Ann Lee's reach", "She holds 3 rights.", {"kind": "record", "ref": "r2"})]), request))
    assert [s.blocks for s in card.sections] == [[]] and diag.drops == {"sibling_mismatch": 1}
    assert diag.correctable() == 1


@pytest.mark.parametrize("heading, verdict, first, second", [
    ("Reach", "Three rights are held.", "Identity: Ann Lee", "Identity: Bob Roy"),               # names neither
    ("Ann Lee and Bob Roy", "Compared.", "Identity: Ann Lee", "Identity: Bob Roy"),              # names both
    ("Ann Lee's reach", "She holds 3 rights.", "", ""),                                          # no labels of their own
    ("Ann Lee's reach", "She holds 3 rights.", "Identity: Ann Lee", "Identity: Ann Lee"),        # the same label twice
    ("Ann Lee's reach", "She holds 3 rights.", "Ann Lee", "Ann Lee Roy"),                        # the second adds a word the section lacks
])
def test_r6_it_is_conservative_when_the_labels_do_not_say_which_sibling(heading, verdict, first, second):
    request = sibling_request(first, second)
    request.results[0].parameters = {}
    request.results[1].parameters = {}
    card = validate_answer_card(AnswerCard(sections=[
        sec(heading, verdict, {"kind": "record", "ref": "r2"})]), request)
    assert [len(s.blocks) for s in card.sections] == [1]


# ----------------------------------------------------------------------
# fix-weih.25 round 3 (invented library data; the shapes are what the small and the strong model really wrote)
# ----------------------------------------------------------------------

class ThinkingLM(RecordingLM):
    """A recording LM that reports what litellm would: it thinks, and takes these parameters."""

    model_name = "fw-test-thinker"
    params = frozenset({"temperature", "max_tokens", "thinking", "reasoning_effort"})
    finish_reason = ""
    refuse_switch = False

    def __init__(self, answers, **kwargs):
        super().__init__(answers, **kwargs)
        self.model = self.model_name

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied.model = self.model
        copied.params, copied.finish_reason, copied.refuse_switch = self.params, self.finish_reason, self.refuse_switch
        return copied

    @property
    def supports_reasoning(self):
        return True

    @property
    def supported_params(self):
        return self.params

    def forward(self, prompt=None, messages=None, **kwargs):
        if self.refuse_switch and "thinking" in self.kwargs:
            raise ValueError("thinking.type: this model does not take a thinking setting")
        if self.refuse_temperature and "temperature" in self.kwargs and not self.kwargs.get("drop_params"):
            raise litellm.UnsupportedParamsError("temperature", model="dummy", llm_provider="dummy")
        response = DummyLM.forward(self, prompt=prompt, messages=messages, **kwargs)
        if self.finish_reason:
            response.choices[0].finish_reason = self.finish_reason
        return response


@pytest.fixture
def thinker(monkeypatch):
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())
    monkeypatch.setattr(gui, "_SWITCH_REFUSED", set(), raising=False)

    def install(answers, **attrs):
        lm = ThinkingLM(answers)
        for name, value in attrs.items():
            setattr(lm, name, value)
        monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
        return lm
    return install


OK_SECTIONS = {"lead": "Done.", "sections": [{"blocks": [{"kind": "record", "ref": "r1.status"}]}]}


def test_t1_a_model_that_thinks_is_asked_not_to_and_gets_room_for_the_thinking_it_still_does(thinker):
    lm = thinker([OK_SECTIONS])
    assert gui.compose_answer_card(card_request()) is not None
    options = lm.copies[0]
    assert options["thinking"] == {"type": "disabled"} and "reasoning_effort" not in options
    assert options["max_tokens"] == gui.CARD_MAX_TOKENS + gui.REASONING_HEADROOM_TOKENS and options["temperature"] == 0.0


def test_t1_the_lowest_reasoning_effort_the_model_map_lists_is_used_when_there_is_no_thinking_switch(thinker, monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "fw-test-effort", {"supports_minimal_reasoning_effort": True})
    lm = thinker([OK_SECTIONS], model_name="fw-test-effort", params=frozenset({"temperature", "reasoning_effort"}))
    lm.model = "fw-test-effort"
    gui.compose_answer_card(card_request())
    assert lm.copies[0]["reasoning_effort"] == "minimal" and "thinking" not in lm.copies[0]


def test_t1_a_model_the_map_says_took_sampling_away_is_sent_no_temperature_and_no_call_is_wasted(thinker, monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "fw-test-nosampling", {"supports_sampling_params": False})
    lm = thinker([OK_SECTIONS], refuse_temperature=True)
    lm.model = "fw-test-nosampling"
    assert gui.compose_answer_card(card_request()) is not None
    assert "temperature" not in lm.copies[0] and len(lm.copies) == 1 and lm.copies[0].get("drop_params") is None


def test_t1_a_model_whose_parameter_list_lacks_temperature_is_sent_none(thinker):
    lm = thinker([OK_SECTIONS], params=frozenset({"max_tokens", "thinking", "reasoning_effort"}))
    gui.compose_answer_card(card_request())
    assert "temperature" not in lm.copies[0]


def test_t1_the_limit_never_exceeds_the_models_own_output_ceiling(thinker, monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "fw-test-ceiling", {"max_output_tokens": 2500})
    lm = thinker([OK_SECTIONS])
    lm.model = "fw-test-ceiling"
    gui.compose_answer_card(card_request())
    assert lm.copies[0]["max_tokens"] == 2500


def test_t1_a_provider_that_refuses_the_switch_is_called_once_more_without_it_and_remembered(thinker):
    lm = thinker([OK_SECTIONS, OK_SECTIONS], refuse_switch=True)
    assert gui.compose_answer_card(card_request()) is not None
    assert "thinking" in lm.copies[0] and "thinking" not in lm.copies[1] and lm.copies[1]["max_tokens"] == 4800
    assert "fw-test-thinker" in gui._SWITCH_REFUSED
    lm.copies.clear()
    gui.compose_answer_card(card_request())
    assert len(lm.copies) == 1 and "thinking" not in lm.copies[0]


def test_t1_a_cut_off_first_answer_is_recorded_and_retried_once_with_a_larger_limit(thinker):
    lm = thinker([{"lead": "Done.", "sections": []}, OK_SECTIONS], finish_reason="length")
    diag = gui.ComposeDiagnostics()
    card = gui.compose_answer_card(card_request(), diag)
    assert diag.truncated and "truncated=1" in diag.summary()
    limits = [c["max_tokens"] for c in lm.copies]
    assert len(limits) == 2 and limits[1] > limits[0]
    assert limits[1] == gui.CARD_MAX_TOKENS + gui.TRUNCATION_RETRY_TOKENS + gui.REASONING_HEADROOM_TOKENS
    assert diag.retry in ("used", "kept_first") and card is not None


def test_t1_a_cut_off_answer_is_not_retried_when_no_larger_limit_exists_or_time_is_short(thinker, monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "fw-test-small-ceiling", {"max_output_tokens": 4800})
    lm = thinker([{"lead": "Done.", "sections": []}, OK_SECTIONS], finish_reason="length")
    lm.model = "fw-test-small-ceiling"
    diag = gui.ComposeDiagnostics()
    gui.compose_answer_card(card_request(), diag)
    assert diag.truncated and len(lm.copies) == 1 and diag.retry == "none"
    monkeypatch.undo()


def test_t1_an_answer_whose_json_stops_mid_string_counts_as_cut_off():
    assert gui._json_is_cut('[[ ## lead ## ]]\nx\n\n[[ ## sections ## ]]\n[{"heading": "Over')
    assert gui._json_is_cut('[[ ## sections ## ]]\n[{"a": [1, 2')
    assert not gui._json_is_cut('[[ ## sections ## ]]\n[{"a": 1}]\n\n[[ ## completed ## ]]')
    assert not gui._json_is_cut("no markers at all")


def test_t1_a_complete_answer_is_not_truncated(thinker):
    thinker([OK_SECTIONS])
    diag = gui.ComposeDiagnostics()
    gui.compose_answer_card(card_request(), diag)
    assert not diag.truncated and "truncated" not in diag.summary()


# ---- T2: refs the model wrote as index or digest paths are repaired when the meaning is unambiguous ----

def loans_request(answer="### 1. loans\nThe member holds 3 loans.\n\n### 2. fine\nThe fine is 12.5.", extra=()):
    return AnswerCardRequest(user_message="loans and fine", answer=answer, results=[
        ResultData(id="r1", command="Library/list_loans", output={
            "loan_ids": ["L1", "L2", "L3"], "titles": ["Dune", "Emma", "Ulysses"], "due": ["May 1", "May 2", "May 3"], "total": 3},
            titles={"titles": "Titles", "total": "Loans"}),
        ResultData(id="r2", command="Library/get_fine", output={"fine_amount": 12.5, "notes": ["late", "damaged"], "total": 2}),
        ResultData(id="r3", command="Library/list_reservations", output={"reservations": [
            {"title": "Dune", "rank": 1}, {"title": "Emma", "rank": 2}], "total": 2}),
        *extra])


def _one_block(block, request=None, heading="Loans"):
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[sec(heading, "", block)]), request or loans_request()))
    return card, diag


@pytest.mark.parametrize("ref", ["r1.titles.0", "r1.titles[0]", "r1.titles", "r1.page.titles.0", "r1.page.sample.0", "r1.sample",
                                 "r1.first_rows.0", "r1.rows"])
@pytest.mark.parametrize("kind", ["record", "table"])
def test_t2_index_and_digest_paths_over_a_page_become_the_table_of_the_page(ref, kind):
    card, diag = _one_block({"kind": kind, "ref": ref})
    block = card.sections[0].blocks[0]
    assert (block.kind, block.ref) == ("table", "r1")
    assert diag.drops == {} and diag.correctable() == 0
    assert diag.refs_repaired == (0 if (ref == "r1.titles" and kind == "table") else 1)


def test_t2_a_page_path_naming_a_column_keeps_that_column_and_the_models_fields_win():
    card, diag = _one_block({"kind": "record", "ref": "r1.page.sample.0", "fields": ["titles", "due"]})
    assert (card.sections[0].blocks[0].ref, card.sections[0].blocks[0].fields) == ("r1", ["titles", "due"])
    assert diag.refs_repaired == 1 and diag.repairs == [("r1.page.sample.0", "r1")]


def test_t2_an_index_path_over_a_list_of_records_is_the_table_of_that_list():
    for ref in ("r3.reservations.0", "r3.reservations[1].title"):
        card, diag = _one_block({"kind": "record", "ref": ref})
        assert [(b.kind, b.ref) for b in card.sections[0].blocks] == [("table", "r3.reservations")], ref
        assert diag.refs_repaired == 1 and diag.drops == {}


def test_t2_a_record_over_two_or_more_columns_of_a_page_is_a_table_and_one_column_stays_a_record():
    card, diag = _one_block({"kind": "record", "ref": "r1", "fields": ["titles", "due"]})
    assert [(b.kind, b.fields) for b in card.sections[0].blocks] == [("table", ["titles", "due"])] and diag.refs_repaired == 1
    card, diag = _one_block({"kind": "record", "ref": "r1", "fields": ["titles"]})
    assert card.sections[0].blocks[0].kind == "record" and diag.refs_repaired == 0
    card, diag = _one_block({"kind": "record", "ref": "r1"})                            # no fields: a record of the page
    assert card.sections[0].blocks[0].kind == "record" and diag.refs_repaired == 0


def member_request():
    return loans_request(extra=(
        ResultData(id="r4", command="Library/get_member", output={"member": "Ann", "tier": "gold", "fine_amount": 12.5, "active": True}),
        ResultData(id="r5", command="Library/get_counts", output={"a": [1, 2], "b": ["x"], "total": 2}),
        ResultData(id="r6", command="Library/get_profile", output={"member": {"name": "Ann"}, "tier": "gold"}),
        ResultData(id="r7", command="Library/get_nothing", output={})))


def test_t2_a_table_over_one_flat_record_is_that_records_record_block():
    card, diag = _one_block({"kind": "table", "ref": "r4", "fields": ["member", "tier", "no_such_field"]}, member_request())
    block = card.sections[0].blocks[0]
    assert (block.kind, block.ref, block.fields) == ("record", "r4", ["member", "tier"])
    assert diag.drops == {} and diag.correctable() == 0 and diag.refs_repaired == 1 and diag.repairs == [("r4", "r4")]
    written, written_diag = _one_block({"kind": "record", "ref": "r4", "fields": ["member", "tier", "no_such_field"]}, member_request())
    assert written.sections[0].blocks[0] == block and written_diag.refs_repaired == 0       # exactly what a model-written record block is
    card, _ = _one_block({"kind": "table", "ref": "r4", "fields": ["labels"]}, member_request())
    assert (card.sections[0].blocks[0].kind, card.sections[0].blocks[0].fields) == ("record", [])   # no field exists: the whole record, as a record does


def test_t2_a_table_over_anything_but_one_flat_record_is_still_dropped():
    for ref in ("r2", "r5", "r6", "r7", "r2.notes"):          # a list field, unequal lists, nested, empty, a list of scalars
        card, diag = _one_block({"kind": "table", "ref": ref}, member_request())
        assert card is None and diag.refs_repaired == 0, ref
        assert diag.drops.get("not_a_table") == 1 and diag.correctable() >= 1, (ref, diag.drops)
    card, diag = _one_block({"kind": "table", "ref": "r1"}, member_request())           # a real page is a table and no repair
    assert card.sections[0].blocks[0].kind == "table" and diag.refs_repaired == 0


def test_t2_the_flat_record_repair_is_counted_not_dropped_and_asks_for_no_second_call(composer):
    sections = [sec("Member", "", {"kind": "table", "ref": "r4", "fields": ["member", "tier"]}),
                sec("Loans", "", {"kind": "table", "ref": "r1", "fields": ["titles", "due"]})]
    card, diag = _composed(composer, sections, request=member_request())
    assert [b.kind for b in card.blocks] == ["record", "table"]
    assert diag.refs_repaired == 1 and diag.drops == {} and diag.retry == "none" and "repaired=1" in diag.summary()


def test_t2_a_stat_over_a_list_becomes_the_count_only_when_its_label_says_count():
    for label in ("Number of loans", "Loan count", "Total loans", "How many loans"):
        card, diag = _one_block({"kind": "stats", "stats": [{"label": label, "ref": "r1.titles.0"}]})
        assert [s.ref for s in card.sections[0].blocks[0].stats] == ["r1.total"], label
        assert diag.refs_repaired == 1
    card, diag = _one_block({"kind": "stats", "stats": [{"label": "Titles", "ref": "r1.titles"}]})
    assert card is None and diag.drops["stat_not_scalar"] == 1 and diag.refs_repaired == 0


def test_t2_ambiguous_refs_are_dropped_as_before():
    for ref in ("r1.nothing.0", "r9.titles.0", "r2.notes.0", "r2.fine_amount.0", "r1.page.nothing"):
        card, diag = _one_block({"kind": "record", "ref": ref})
        assert card is None, ref
        assert diag.refs_repaired == 0 and diag.drops.get("ref_unresolved") == 1, (ref, diag.drops)


def test_t2_a_repair_is_counted_not_dropped_and_never_asks_for_a_second_call(composer):
    sections = [sec("Loans", "", {"kind": "record", "ref": "r1.page.sample.0"}),
                sec("Reservations", "", {"kind": "record", "ref": "r3.reservations.0"})]
    card, diag = _composed(composer, sections, request=loans_request())
    assert diag.refs_repaired == 2 and diag.drops == {} and diag.retry == "none" and "repaired=2" in diag.summary()
    assert [b.kind for b in card.blocks] == ["table", "table"]


# ---- T3: a section about a result an earlier section shows is merged into it, not cut ----

def test_t3_sections_over_the_cap_about_an_earlier_result_are_merged_in_order():
    request = parts_request(0)
    sections = [stat_section(i) for i in range(1, 6)]
    sections.append(sec("Alpha table", "", {"kind": "record", "ref": "r1", "fields": ["v", "w"]}))
    sections.append(sec("Other", "", {"kind": "record", "ref": "r2"}))
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=sections), request))
    assert [s.heading for s in card.sections] == NAMES[:5]
    assert [len(s.blocks) for s in card.sections] == [2, 2, 1, 1, 1]
    assert diag.sections_merged == 2 and diag.drops == {} and "merged=2" in diag.summary()


def test_t3_a_table_only_section_about_the_same_result_with_the_same_subject_is_merged():
    sections = [sec("Loans on the account", "The member holds loans.", {"kind": "stats", "stats": [{"label": "Loans", "ref": "r1.total"}]}),
                sec("Loans list", "", {"kind": "table", "ref": "r1", "fields": ["titles", "due"]}),
                sec("Fine", "The fine is 12.5.", {"kind": "stats", "stats": [{"label": "Fine amount", "ref": "r2.fine_amount"}]})]
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=sections), loans_request()))
    assert [s.heading for s in card.sections] == ["Loans on the account", "Fine"]
    assert [b.kind for b in card.sections[0].blocks] == ["stats", "table"] and diag.sections_merged == 1


def test_t3_sections_about_different_results_are_never_merged_and_a_different_subject_stays_apart():
    request = parts_request(0)
    over = [stat_section(i) for i in range(1, 6)] + [sec("Sixth", "", {"kind": "record", "ref": "r6"})]
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=over), request))
    assert len(card.sections) == 5 and diag.sections_merged == 0 and diag.drops == {"section_cap": 1}
    two = [sec("Loans", "", {"kind": "stats", "stats": [{"label": "Loans", "ref": "r1.total"}]}),
           sec("Due dates", "", {"kind": "table", "ref": "r1", "fields": ["titles", "due"]})]     # same result, another part
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=two), loans_request()))
    assert len(card.sections) == 2 and diag.sections_merged == 0


def test_t3_the_per_section_block_cap_holds_and_the_overflow_is_dropped_with_the_cap_reason():
    request = parts_request(0)
    first = sec("Alpha", "", *[{"kind": "stats", "stats": [{"label": "L", "ref": "r1.v"}]},
                               {"kind": "record", "ref": "r1"}, {"kind": "quote", "quote": "x" * 5}])
    extra = sec("Alpha more", "", {"kind": "record", "ref": "r1", "fields": ["v"]}, {"kind": "record", "ref": "r1", "fields": ["w"]})
    rest = [stat_section(i) for i in range(2, 6)]
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[first, *rest, extra]), request))
    assert len(card.sections[0].blocks) <= gui.MAX_SECTION_BLOCKS and diag.sections_merged <= 1


# ---- T4: the digest ----

def _crowded_primary(n=14):
    return [ResultData(id=f"r{i}", command=f"Library/cmd_{i}", context="Ann Lee", output={
        "name": f"Item {i}", "count": i, "ids": [f"i{j}" for j in range(30)],
        "labels": [f"label {j} " + "z" * 60 for j in range(30)], "total": 30 + i,
        **{f"note_{k}": "n" * 150 for k in range(12)}}, supporting=False) for i in range(1, n + 1)]


def test_t4_supporting_results_collapse_before_primary_ones_and_a_collapsed_primary_keeps_its_label_and_values():
    primary = _crowded_primary(10)
    supporting = [ResultData(id=f"s{i}", command=f"Library/lookup_{i}", output={f"f{n}": "q" * 150 for n in range(12)}, supporting=True)
                  for i in range(1, 6)]
    items = json.loads(results_digest([*primary, *supporting]))
    assert len(results_digest([*primary, *supporting])) <= gui.DIGEST_MAX_CHARS
    by_id = {i["id"]: i for i in items if isinstance(i, dict)}
    assert all("output" not in by_id[f"s{i}"] for i in range(1, 6))                    # supporting: collapsed first
    assert all("output" in by_id[f"r{i}"] for i in range(1, 11))                        # primary: still in full


def test_t4_a_primary_line_that_does_collapse_carries_label_show_line_and_the_main_values():
    items = json.loads(results_digest(_crowded_primary(40)))
    collapsed = [i for i in items if isinstance(i, dict) and "output" not in i]
    assert collapsed, "the crowd must force a collapse"
    line = collapsed[0]
    assert line["label"] == "Ann Lee" and line["show"][0]["kind"] == "table" and line["show"][0]["ref"].startswith("r") and "rows" in line
    assert 1 <= len(line["values"]) <= 2 and "ids" not in line["values"]            # not an identifier column


def test_t4_a_list_result_says_how_to_show_it_in_the_syntax_the_validator_accepts():
    items = json.loads(results_digest(loans_request().results))
    shows = {i["id"]: i.get("show") for i in items}
    assert shows["r1"] == [{"kind": "table", "ref": "r1", "fields": ["loan_ids", "titles", "due"]}]
    assert shows["r3"] == [{"kind": "table", "ref": "r3.reservations", "fields": ["title", "rank"]}]
    assert shows["r2"] == [{"kind": "record", "ref": "r2", "fields": ["notes"]}]
    assert not any("page" in i or "sample" in json.dumps(list(i)) for i in items)           # no nesting to copy a path from
    for ref, block in (("r1", {"kind": "table", "ref": "r1", "fields": ["loan_ids", "titles", "due"]}),
                       ("r3", {"kind": "table", "ref": "r3.reservations", "fields": ["title", "rank"]}),
                       ("r2", {"kind": "record", "ref": "r2", "fields": ["notes"]})):
        card, diag = _one_block(block)
        assert card.sections[0].blocks[0].ref == block["ref"] and diag.refs_repaired == 0, ref


# ---- T5: instructions and the worked example ----

def test_t5_the_instructions_state_the_new_rules_and_the_example_uses_ids_no_request_has():
    text = gui.ComposeAnswerCard.instructions
    for phrase in ("table block", "never the command's name", "70 to 95 characters", "Cover every numbered part", "prefer a table"):
        assert phrase in text, phrase
    example = gui.ComposeAnswerCard.output_fields["sections"].json_schema_extra["desc"]
    assert all(f'"{rid}' in example for rid in gui.EXAMPLE_RESULT_IDS)
    assert all(int(rid[1:]) > gui.MAX_RESULTS for rid in gui.EXAMPLE_RESULT_IDS)


def test_t5_a_block_copied_from_the_example_is_dropped_as_example_copied():
    sections = [sec("Overdue loans", "", {"kind": "table", "ref": "r901", "fields": ["loan_titles", "loan_due_dates"]}),
                sec("Fine", "", {"kind": "stats", "stats": [{"label": "Fine owed", "ref": "r902.fine_amount"}]}),
                sec("Real", "", {"kind": "table", "ref": "r1", "fields": ["titles", "due"]})]
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=sections), loans_request()))
    assert [s.heading for s in card.sections] == ["Real"] and diag.drops["example_copied"] == 2
    assert "example_copied" in gui.CORRECTION_HINT_DROPS and diag.correctable() == 2          # the two empty sections
    # an id the request really has is never mistaken for the example's
    real = AnswerCardRequest(answer="Done.", results=[ResultData(id="r901", output={"a": [{"x": 1}, {"x": 2}]})])
    card = validate_answer_card(AnswerCard(sections=[sec("A", "", {"kind": "table", "ref": "r901.a"})]), real)
    assert card is not None and card.sections[0].blocks[0].ref == "r901.a"


# ---- T6: a state word attached to the counted noun ----

def findings_request():
    return AnswerCardRequest(user_message="findings", results=[
        ResultData(id="r1", command="Audit/list_findings", output={"titles": ["a", "b", "c", "d", "e"], "total": 5}, titles={"total": "Findings"}),
        ResultData(id="r2", command="Audit/open_findings", output={"open_total": 2}, titles={"open_total": "Open findings"})],
        answer="5 control findings hit; open control findings on this identity: 2.")


@pytest.mark.parametrize("verdict, kept", [
    ("The identity has five open control findings.", False),           # 5 is the total, not the open count
    ("The identity has two open control findings.", True),             # the answer pairs 2 with "open"
    ("The identity has five control findings.", True),
    ("The identity has 5 critical control findings.", False),          # nothing pairs 5 with critical
])
def test_t6_a_state_word_needs_a_number_the_answer_or_the_data_pairs_with_it(verdict, kept):
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[
        sec("Findings", verdict, {"kind": "stats", "stats": [{"label": "Findings", "ref": "r1.total"}]})]), findings_request()))
    assert (card.sections[0].verdict == verdict) is kept
    assert (diag.drops.get("verdict_number_unanchored", 0) == 0) is kept


def test_t6_a_data_field_whose_title_carries_the_state_word_anchors_it_and_people_vs_holders_is_untouched():
    request = AnswerCardRequest(answer="Report ready.", results=[
        ResultData(id="r1", output={"open_total": 4}, titles={"open_total": "Open items"})])
    assert gui._unanchored_sentence("There are 4 open items.", gui._anchor_sets(request)) == ""
    assert gui._unanchored_sentence("There are 4 closed items.", gui._anchor_sets(request)) != ""
    holders = AnswerCardRequest(answer="167 holders were found.", results=[ResultData(id="r1", output={"total": 167})])
    assert gui._unanchored_sentence("167 people hold it.", gui._anchor_sets(holders)) == ""


# ---- T7: label words the answer itself uses for that value ----

def rights_request(answer):
    return AnswerCardRequest(answer=answer, results=[
        ResultData(id="r1", command="Identity/reach", output={"entitlement_total": 32, "group_total": 7, "level": 4},
                   titles={"entitlement_total": "Entitlements", "group_total": "Groups", "level": "Level"})])


def _label_for(answer, label, ref="r1.entitlement_total"):
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[
        sec("Reach", "", {"kind": "stats", "stats": [{"label": label, "ref": ref}]})]), rights_request(answer)))
    return card.sections[0].blocks[0].stats[0].label, diag


def test_t7_a_label_made_of_the_words_the_answer_puts_beside_that_value_is_supported():
    label, diag = _label_for("The identity holds 32 rights through 7 groups.", "Rights")
    assert label == "Rights" and diag.drops == {}


def test_t7_the_answers_words_for_another_value_do_not_support_the_label():
    label, diag = _label_for("The identity holds 32 rights through 7 groups.", "Groups")        # "groups" owns 7, the field holds 32
    assert label == "Entitlements" and diag.drops == {"stat_label_unsupported": 1}               # code wrote the field's title
    other, diag = _label_for("The identity holds 32 rights through 7 groups.", "Rights", ref="r1.level")   # the field holds 4
    assert other != "Rights" and diag.drops == {"stat_label_unsupported": 1}


def test_t7_a_state_word_in_the_label_must_be_beside_the_value_too():
    label, diag = _label_for("The identity holds 32 rights.", "Open rights")
    assert label != "Open rights" and diag.drops == {"stat_label_unsupported": 1}


def test_t7_a_value_the_answer_gives_two_names_does_not_lend_either_to_a_label():
    request = AnswerCardRequest(answer="It reaches 32 rights; groups: 32 groups.", results=[
        ResultData(id="r1", output={"group_total": 32}, titles={"group_total": "Groups"})])
    card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=[
        sec("Reach", "", {"kind": "stats", "stats": [{"label": "Rights", "ref": "r1.group_total"}]})]), request))
    assert card.sections[0].blocks[0].stats[0].label == "Groups" and diag.drops == {"stat_label_unsupported": 1}


# ---- fix-weih.25 round 4 (C1-C7): repairs and truth checks that decide neither layout nor wording ----

def _card(sections, request, lead=""):
    return _with_diag(lambda: validate_answer_card(AnswerCard(lead=lead, sections=sections), request))


def invoices_request(rows=12, extra_output=None, answer="Invoices were read.", descriptions=None):
    output = {"invoices": [{"number": f"INV-{i}", "amount": 10 + i} for i in range(rows)], **(extra_output or {})}
    return AnswerCardRequest(answer=answer, results=[
        ResultData(id="r1", command="Billing/list_invoices", output=output, descriptions=descriptions or {})])


INVOICE_TABLE = {"kind": "table", "ref": "r1.invoices", "fields": ["number", "amount"]}


# C1: a state word in a stat label or a block title

def issues_request(answer="5 issues in total; 2 are open."):
    return AnswerCardRequest(answer=answer, results=[
        ResultData(id="r1", command="Audit/list_issues", output={"total": 5, "open_total": 2}, titles={"total": "Issues", "open_total": "Open issues"})])


def test_c1_a_stat_label_with_a_state_word_over_the_total_is_replaced_by_the_fields_own_title():
    card, diag = _card([sec("Issues", "", {"kind": "stats", "stats": [{"label": "Open Issues", "ref": "r1.total"}]})], issues_request())
    assert card.sections[0].blocks[0].stats[0].label == "Issues"
    assert diag.drops == {"stat_label_unsupported": 1}


@pytest.mark.parametrize("label, ref, answer", [
    ("Open Issues", "r1.open_total", "5 issues in total; 2 are open."),        # the field's own title carries "open"
    ("Open Issues", "r1.total", "There are 5 open issues."),                  # the answer pairs this very value with "open"
    ("Issues", "r1.total", "5 issues in total; 2 are open."),                  # no state word: nothing to check
])
def test_c1_a_state_word_the_field_or_the_answer_pairs_with_the_value_is_kept(label, ref, answer):
    card, diag = _card([sec("Issues", "", {"kind": "stats", "stats": [{"label": label, "ref": ref}]})], issues_request(answer))
    assert card.sections[0].blocks[0].stats[0].label == label and diag.drops == {}


def test_c1_a_block_title_with_a_state_word_the_data_does_not_carry_is_dropped():
    request = issues_request()
    card, diag = _card([sec("Issues", "", {"kind": "stats", "title": "Open issues", "stats": [{"label": "Issues", "ref": "r1.total"}]})], request)
    assert card.sections[0].blocks[0].title == "" and diag.drops == {"title_unsupported": 1}
    card, diag = _card([sec("Issues", "", {"kind": "stats", "title": "Open issues", "stats": [{"label": "Open issues", "ref": "r1.open_total"}]})], request)
    assert card.sections[0].blocks[0].title == "Open issues" and diag.drops == {}


def test_c1_a_table_title_with_a_state_word_needs_that_word_in_the_fields_or_the_rows():
    plain = invoices_request(3)
    card, diag = _card([sec("Invoices", "", {**INVOICE_TABLE, "title": "Overdue invoices"})], plain)
    assert card.sections[0].blocks[0].title == "" and diag.drops == {"title_unsupported": 1}
    rows = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", command="Billing/list_invoices", output={
        "invoices": [{"number": "INV-1", "status": "overdue"}, {"number": "INV-2", "status": "paid"}]})])
    card, diag = _card([sec("Invoices", "", {"kind": "table", "ref": "r1.invoices", "title": "Overdue invoices"})], rows)
    assert card.sections[0].blocks[0].title == "Overdue invoices" and diag.drops == {}


# C2: a completeness claim over a table the page caps

def test_c2_the_row_cap_is_the_pages_and_the_two_are_kept_in_step():
    import re
    from pathlib import Path
    source = (Path(gui.__file__).parent / "static" / "src" / "258-chat-card.js").read_text(encoding="utf-8")
    assert int(re.search(r"var TM_CARD_ROWS_SHOWN = (\d+);", source).group(1)) == gui.TABLE_ROWS_SHOWN == 8


def test_c2_an_all_shown_verdict_over_a_capped_table_loses_that_sentence_only():
    card, diag = _card([sec("Invoices", "All 12 invoices are shown below.", INVOICE_TABLE)], invoices_request(12))
    assert card.sections[0].verdict == "" and len(card.sections[0].blocks) == 1
    assert diag.drops == {"completeness_unsupported": 1}


@pytest.mark.parametrize("extra, description", [
    ({"total": 40}, None),                          # a total larger than the rows on the page
    ({"not_shown": 3}, None),                       # a not-shown count above zero
    ({"remaining": 2}, None),
])
def test_c2_data_that_says_more_exist_makes_a_completeness_claim_misleading_even_under_the_cap(extra, description):
    card, diag = _card([sec("Invoices", "Every invoice is listed below.", INVOICE_TABLE)], invoices_request(4, extra))
    assert card.sections[0].verdict == "" and diag.drops == {"completeness_unsupported": 1}


def test_c2_a_page_whose_total_exceeds_its_rows_is_not_complete():
    page = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", command="Billing/list_holders", output={
        "names": ["a", "b", "c"], "ids": ["1", "2", "3"], "total": 40})])
    card, diag = _card([sec("Holders", "All holders are listed below.", {"kind": "table", "ref": "r1", "fields": ["names", "ids"]})], page)
    assert card.sections[0].verdict == "" and diag.drops == {"completeness_unsupported": 1}


@pytest.mark.parametrize("verdict, rows, extra", [
    ("All 5 invoices are listed below.", 5, None),                       # the rows shown ARE all the rows
    ("All 8 invoices are listed below.", 8, {"total": 8}),               # exactly the cap
    ("The first 8 of 12 invoices are listed below.", 12, None),          # no completeness claim
    ("Of 12 invoices, all of the first 8 are shown.", 12, None),         # says it is partial
    ("All invoices are paid.", 12, None),                                # nothing about what is shown
])
def test_c2_a_true_or_unrelated_statement_is_kept(verdict, rows, extra):
    card, diag = _card([sec("Invoices", verdict, INVOICE_TABLE)], invoices_request(rows, extra, answer=f"{verdict} Invoices were read."))
    assert card.sections[0].verdict == verdict and "completeness_unsupported" not in diag.drops


def test_c2_a_lead_and_a_block_title_that_claim_completeness_over_a_capped_table_are_repaired():
    request = invoices_request(12)
    card, diag = _card([sec("Invoices", "", {**INVOICE_TABLE, "title": "Complete list of invoices"})], request,
                       lead="Billing was read. Every invoice is listed below.")
    assert card.lead == "Billing was read." and card.sections[0].blocks[0].title == ""
    assert diag.drops == {"completeness_unsupported": 2}


def test_c2_a_second_table_that_is_complete_keeps_the_claim():
    request = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", command="Billing/list_invoices", output={
        "invoices": [{"number": f"INV-{i}"} for i in range(12)], "credits": [{"ref_no": "C1"}, {"ref_no": "C2"}]})])
    card, diag = _card([sec("Billing", "All credits are listed below.", INVOICE_TABLE, {"kind": "table", "ref": "r1.credits"})], request)
    assert card.sections[0].verdict == "All credits are listed below." and diag.drops == {}


# C3: a sentence that points at content that is not there

def test_c3_a_verdict_that_points_below_in_a_section_with_no_block_is_dropped_and_the_section_goes_with_it():
    card, diag = _card([sec("Invoices", "The invoices are listed below.", {"kind": "table", "ref": "r9"}),
                        sec("Totals", "Billing is up to date.")], invoices_request(3))
    assert [s.heading for s in card.sections] == ["Totals"]
    assert diag.drops["dangling_reference"] == 1 and diag.drops["empty_section"] == 1


@pytest.mark.parametrize("verdict", ["The findings are listed with their labels.", "The totals are as follows.",
                                     "See the table for the details.", "The holders listed are active."])
def test_c3_other_ways_of_pointing_at_nothing(verdict):
    card, diag = _card([sec("Invoices", verdict, {"kind": "stats", "stats": [{"label": "Invoices", "ref": "r9.x"}]}),
                        sec("Totals", "Billing is up to date.")], invoices_request(3))
    assert diag.drops.get("dangling_reference") == 1, verdict


@pytest.mark.parametrize("verdict, block", [
    ("The invoices are listed below.", INVOICE_TABLE),                                      # a table is there
    ("The total is shown below.", {"kind": "stats", "stats": [{"label": "Invoices", "ref": "r1.total"}]}),   # a block is there
    ("The balance sits below the limit.", None),                                            # 'below' as a comparison
    ("The account is listed as dormant.", None),                                            # another sense of listed
])
def test_c3_a_reference_that_has_its_content_is_kept(verdict, block):
    request = invoices_request(3, {"total": 3})
    blocks = [block] if block else []
    card, diag = _card([sec("Invoices", verdict, *blocks)], request)
    assert card.sections[0].verdict == verdict and "dangling_reference" not in diag.drops


def test_c3_a_lead_that_points_at_blocks_in_a_card_with_none_is_repaired():
    card, diag = _card([sec("Billing", "Billing is up to date.")], invoices_request(3), lead="Billing was read. The details are as follows.")
    assert card.lead == "Billing was read." and diag.drops == {"dangling_reference": 1}


# C4: the digest's show hint pasted into the ref

@pytest.mark.parametrize("ref, fields, kind, want_ref, want_fields", [
    ("r1 fields=[titles,due]", [], "table", "r1", ["titles", "due"]),
    ("table ref=r1 fields=[titles,due]", [], "table", "r1", ["titles", "due"]),
    ("table ref=r3.reservations fields=[title, rank]", [], "table", "r3.reservations", ["title", "rank"]),
    ("record ref=r2 fields=[notes]", [], "record", "r2", ["notes"]),
    ("table ref=r1 fields=[titles]", ["titles", "due"], "table", "r1", ["titles", "due"]),     # the block's own fields win
    ("ref=r1", [], "table", "r1", []),
])
def test_c4_a_pasted_show_hint_is_read_back_into_ref_and_fields(ref, fields, kind, want_ref, want_fields):
    card, diag = _one_block({"kind": kind, "ref": ref, "fields": fields})
    block = card.sections[0].blocks[0]
    assert (block.ref, block.fields) == (want_ref, want_fields) and diag.drops == {} and diag.refs_repaired == 1


@pytest.mark.parametrize("ref", ["r1 fields=[titles,due", "table ref=r1 fields=titles", "table ref=r1 fields=[a] | table ref=r3.reservations fields=[b]",
                                 "r1 fields=[titles] please", "show r1"])
def test_c4_a_hint_that_does_not_parse_cleanly_is_dropped_as_before(ref):
    card, diag = _one_block({"kind": "table", "ref": ref})
    assert card is None or not card.sections
    assert diag.refs_repaired == 0 and diag.drops.get("ref_unresolved") == 1


def test_c4_the_digests_show_hint_is_block_shaped_so_it_is_not_one_string_to_paste_into_a_ref():
    items = json.loads(results_digest(loans_request().results))
    shows = {i["id"]: i.get("show") for i in items}
    assert shows["r1"] == [{"kind": "table", "ref": "r1", "fields": ["loan_ids", "titles", "due"]}]
    assert shows["r2"] == [{"kind": "record", "ref": "r2", "fields": ["notes"]}]
    for entry in [entry for show in shows.values() if show for entry in show]:
        card, diag = _one_block(entry)
        assert diag.drops == {} and diag.refs_repaired == 0, entry


# C5: the composer's own vocabulary in the card's text

@pytest.mark.parametrize("verdict", [
    "The right is listed in the digest.", "Details sit in the primary result.", "A supporting result confirms it.",
    "The payload holds three invoices.", "Shown by the table block.", "The schema lists the fields.", "See ref r1 for the rows.",
])
def test_c5_a_verdict_in_the_composers_own_vocabulary_is_dropped(verdict):
    card, diag = _card([sec("Invoices", verdict, INVOICE_TABLE), ], invoices_request(3))
    assert card.sections[0].verdict == "" and diag.drops.get("internal_wording") == 1, verdict


def test_c5_a_lead_sentence_and_a_title_with_internal_words_are_dropped_and_the_rest_kept():
    card, diag = _card([sec("Invoices", "Three invoices exist.", {**INVOICE_TABLE, "title": "Digest of invoices"})], invoices_request(3),
                       lead="Billing was read. The primary result lists them.")
    assert card.lead == "Billing was read." and card.sections[0].blocks[0].title == ""
    assert diag.drops["internal_wording"] == 2


def test_c5_a_word_the_answer_or_data_itself_uses_is_not_internal():
    request = invoices_request(3, answer="The mail digest was sent; the schema is v2.")
    card, diag = _card([sec("Invoices", "The mail digest was sent.", INVOICE_TABLE)], request)
    assert card.sections[0].verdict == "The mail digest was sent." and diag.drops == {}


def test_c5_the_model_is_not_taught_to_talk_about_a_digest_and_the_example_makes_no_completeness_claim():
    seen = [gui.ComposeAnswerCard.instructions, gui.ComposeAnswerCard.output_fields["sections"].json_schema_extra["desc"],
            gui.ComposeAnswerCard.input_fields["results"].json_schema_extra["desc"]]
    assert not any("digest" in text.lower() for text in seen)
    example = seen[1]
    assert "every one of" not in example and "listed below" not in example


# C6: an evaluative adjective the data and the answer never use

def level_request(answer="The review gave a risk level of 4.", extra=""):
    return AnswerCardRequest(answer=answer + extra, results=[ResultData(id="r1", command="Review/assess", output={"level": 4})])


@pytest.mark.parametrize("verdict, kept", [
    ("The review gives a high risk level of 4.", False),
    ("The score is a low risk rating.", False),
    ("The review gives a risk level of 4.", True),
])
def test_c6_a_judgement_word_beside_risk_or_level_needs_a_source(verdict, kept):
    card, diag = _card([sec("Review", verdict, {"kind": "stats", "stats": [{"label": "Level", "ref": "r1.level"}]})], level_request())
    assert (card.sections[0].verdict == verdict) is kept
    assert diag.drops == ({} if kept else {"judgement_unsupported": 1})


def test_c6_the_word_in_the_answer_or_the_data_makes_it_the_answers_own():
    request = level_request(extra=" The risk is high for this identity.")
    card, diag = _card([sec("Review", "The review gives a high risk level of 4.")], request)
    assert card.sections[0].verdict == "The review gives a high risk level of 4." and diag.drops == {}


# C7: a label that borrows the words of another number

def score_request():
    return AnswerCardRequest(answer="aggregate score: 400\nintrinsic score: 100\nrisk level: 4\n5 findings", results=[
        ResultData(id="r1", command="Review/assess_risk", output={"score": 400, "total": 5},
                   titles={"score": "Score", "total": "Findings"}, descriptions={"score": "The aggregate score of the entity."})])


@pytest.mark.parametrize("label, ref, kept", [
    ("Intrinsic score", "r1.score", False),       # "intrinsic" is the answer's word for 100, not for 400
    ("Aggregate score", "r1.score", True),
    ("Risk level", "r1.total", False),            # shares "risk" with the command, but "level" is the answer's word for 4
    ("Findings", "r1.total", True),
])
def test_c7_a_stat_label_with_a_word_the_answer_gives_another_number_is_replaced(label, ref, kept):
    card, diag = _card([sec("Risk", "", {"kind": "stats", "stats": [{"label": label, "ref": ref}]})], score_request())
    assert (card.sections[0].blocks[0].stats[0].label == label) is kept
    assert diag.drops == ({} if kept else {"stat_label_unsupported": 1})


def test_c2_a_claim_about_two_tables_one_of_them_cut_is_untrue():
    request = AnswerCardRequest(answer="Done.", results=[ResultData(id="r1", command="Billing/list_invoices", output={
        "invoices": [{"number": f"INV-{i}"} for i in range(12)], "credits": [{"ref_no": "C1"}, {"ref_no": "C2"}]})])
    card, diag = _card([sec("Billing", "All invoices and credits are listed below.", INVOICE_TABLE, {"kind": "table", "ref": "r1.credits"})], request)
    assert card.sections[0].verdict == "" and diag.drops == {"completeness_unsupported": 1}


# ----------------------------------------------------------------------
# fix-weih.34: a long answer is composed. The model sees an excerpt of it; every check reads the full text.
# ----------------------------------------------------------------------

def long_parts_answer(parts=9, lines=60, tail=""):
    """A multi-part answer (invented, neutral): each part opens with a summary line then `lines` routine lines."""
    out = ["The request asked for several checks.\n"]
    for part in range(1, parts + 1):
        body = "\n".join(f"Entry {part}.{n} was reviewed and recorded without change." for n in range(1, lines + 1))
        out.append(f"### {part}. Check number {part}\nPart {part} opens with its summary sentence. It continues here.\n{body}\n")
    return "\n".join(out) + tail


def test_w34_an_answer_up_to_the_old_limit_goes_into_the_prompt_whole_and_unchanged():
    exactly = ("### 1. Only part\n" + "Line of text that is long enough.\n" * 700)[:gui.ANSWER_PROMPT_WHOLE_CHARS]
    assert len(exactly) == 20000 == gui.ANSWER_PROMPT_WHOLE_CHARS
    assert gui.answer_for_prompt(exactly) is exactly
    assert gui.answer_for_prompt("") == "" and gui.answer_for_prompt("short") == "short"
    over = exactly + "x"
    assert gui.answer_for_prompt(over) != over and len(gui.answer_for_prompt(over)) <= gui.ANSWER_EXCERPT_CHARS


def test_w34_the_prompt_of_an_answer_at_the_old_limit_is_the_same_as_before(sent_composer):
    exactly = ("### 1. Only part\n" + "Line of text that is long enough.\n" * 700)[:20000]
    lm = sent_composer([CLEAN_SECOND])
    diag = gui.ComposeDiagnostics()
    gui.compose_answer_card(card_request(answer=exactly), diag)
    assert json.dumps(exactly)[1:-1] in lm.sent[0]                     # the whole answer, character for character
    assert diag.excerpt_chars == 0 and diag.answer_chars == 20000
    assert "excerpt_chars" not in diag.summary() and "answer_chars" not in diag.summary()


def test_w34_an_excerpt_is_bounded_keeps_every_heading_in_order_and_a_non_empty_opening_of_each_part():
    answer = long_parts_answer(9, 120)
    assert len(answer) > 40000
    excerpt = gui.answer_for_prompt(answer)
    assert len(excerpt) <= gui.ANSWER_EXCERPT_CHARS
    headings = [f"### {n}. Check number {n}" for n in range(1, 10)]
    positions = [excerpt.index(h) for h in headings]
    assert positions == sorted(positions) and excerpt.startswith("The request asked for several checks.")
    pieces = [p for p in excerpt.split("### ")[1:]]
    assert len(pieces) == 9
    for number, piece in enumerate(pieces, 1):
        body = piece.split("\n", 1)[1]
        assert f"Part {number} opens with its summary sentence." in body            # the opening is there
        assert body.rstrip().endswith(gui.EXCERPT_CUT_MARKER)                       # one marker per cut part
        kept_lines = body.rstrip().split("\n")[:-1]
        assert len(kept_lines) >= 3
        assert all(line in answer for line in kept_lines)                           # cut at a line end: only whole lines
    assert excerpt.count(gui.EXCERPT_CUT_MARKER) == 9


def test_w34_a_short_part_is_kept_whole_and_its_unused_share_goes_to_the_long_parts():
    long_a = "\n".join(f"Long row {n} of the first long part holds routine text." for n in range(400))
    long_b = "\n".join(f"Long row {n} of the second long part holds routine text." for n in range(400))
    answer = f"### 1. Short\nThis part is short and complete.\n\n### 2. Long A\n{long_a}\n\n### 3. Long B\n{long_b}\n"
    excerpt = gui.answer_for_prompt(answer)
    first, second, third = excerpt.split("### ")[1:]
    assert first.rstrip().endswith("This part is short and complete.") and gui.EXCERPT_CUT_MARKER not in first
    assert gui.EXCERPT_CUT_MARKER in second and gui.EXCERPT_CUT_MARKER in third
    assert abs(len(second) - len(third)) < 200 and len(second) > gui.ANSWER_EXCERPT_CHARS * 0.4     # they split what is left
    assert len(excerpt) <= gui.ANSWER_EXCERPT_CHARS


def test_w34_the_text_before_the_first_heading_is_bounded_and_the_parts_still_all_appear():
    answer = "Background sentence. " * 400 + "\n\n" + long_parts_answer(5, 200)
    excerpt = gui.answer_for_prompt(answer)
    assert excerpt.index("### 1.") < 1400 and all(f"### {n}. Check number {n}" in excerpt for n in range(1, 6))
    assert len(excerpt) <= gui.ANSWER_EXCERPT_CHARS


def test_w34_more_headings_than_the_section_ceiling_still_gives_a_bounded_excerpt():
    answer = long_parts_answer(40, 40)
    excerpt = gui.answer_for_prompt(answer)
    assert len(excerpt) <= gui.ANSWER_EXCERPT_CHARS
    assert all(f"### {n}. Check number {n}" in excerpt for n in range(1, gui.MAX_SECTIONS + 1))


def test_w34_an_answer_without_headings_keeps_its_head_and_its_tail_with_a_marker_between():
    answer = "\n".join(f"Row {n} of a plain answer that has no headings at all." for n in range(1, 900))
    excerpt = gui.answer_for_prompt(answer)
    assert len(answer) > 20000 and len(excerpt) <= gui.ANSWER_EXCERPT_CHARS
    assert excerpt.startswith("Row 1 of") and excerpt.rstrip().endswith("Row 899 of a plain answer that has no headings at all.")
    assert excerpt.count(gui.EXCERPT_GAP_MARKER) == 1
    head, tail = excerpt.split(gui.EXCERPT_GAP_MARKER)
    assert all(line in answer for line in head.strip().split("\n")) and all(line in answer for line in tail.strip().split("\n"))
    assert len(head) > len(tail) > 1000


def test_w34_the_markers_hold_no_number_and_a_quote_made_of_one_is_not_grounded():
    for marker in (gui.EXCERPT_CUT_MARKER, gui.EXCERPT_GAP_MARKER):
        assert not any(ch.isdigit() for ch in marker) and "#" not in marker
        assert not gui._quote_is_grounded(marker, long_parts_answer(9, 120))
        assert marker not in long_parts_answer(9, 120)


def test_w34_the_checks_read_the_full_answer_not_the_excerpt(sent_composer):
    quote = "The final tally for all parts reached seven hundred and seventy seven widgets."
    answer = long_parts_answer(9, 120, tail=f"\nTotal of 777 widgets counted. {quote}\n")
    request = AnswerCardRequest(answer=answer, results=[
        ResultData(id="r1", command="cmd_a", output={"status": "ok"}), ResultData(id="r2", command="cmd_b", output={"status": "ok"})])
    assert "777" not in gui.answer_for_prompt(answer) and quote not in gui.answer_for_prompt(answer)    # beyond the excerpt
    card = {"lead": "", "sections": [{"heading": "Check number 9", "verdict": "The checks counted 777 widgets in all.",
            "blocks": [{"kind": "record", "ref": "r1"}, {"kind": "quote", "quote": quote}]}]}
    lm, diag, result = run_card([card], sent_composer, request)
    assert "777" not in lm.sent[0] and "seventy seven" not in lm.sent[0]                                 # the model never saw it
    assert diag.excerpt_chars and diag.answer_chars == len(answer) and diag.excerpt_chars <= gui.ANSWER_EXCERPT_CHARS
    assert f"answer_chars={len(answer)} excerpt_chars={diag.excerpt_chars}" in diag.summary()
    section = result.sections[0]
    assert section.verdict == "The checks counted 777 widgets in all."       # anchored in the full text
    assert [b.kind for b in section.blocks] == ["record", "quote"] and section.blocks[1].quote == quote
    assert "quote_ungrounded" not in diag.drops and "verdict_number_unanchored" not in diag.drops


def test_w34_a_number_that_is_nowhere_in_the_full_answer_is_still_dropped(sent_composer):
    answer = long_parts_answer(9, 120)
    request = AnswerCardRequest(answer=answer, results=[ResultData(id="r1", command="cmd_a", output={"status": "ok"})])
    card = {"lead": "", "sections": [{"heading": "Check number 1", "verdict": "The checks counted 777 widgets in all.",
            "blocks": [{"kind": "record", "ref": "r1"}, {"kind": "quote", "quote": gui.EXCERPT_CUT_MARKER + " plus more words"}]}]}
    _, diag, result = run_card([card, card], sent_composer, request)
    assert result.sections[0].verdict == "" and "verdict_ungrounded" in diag.drops and "quote_ungrounded" in diag.drops


def test_w34_the_answer_card_limit_is_150000_and_the_question_form_limit_is_unchanged():
    assert gui.MAX_ANSWER_CHARS == 150000 and gui.MAX_TEXT_CHARS == 20000
    assert AnswerCardRequest(answer="x" * 150000).answer
    with pytest.raises(ValueError, match="150000"):
        AnswerCardRequest(answer="x" * 150001)
    assert QuestionFormRequest(question="x" * 20000).question
    with pytest.raises(ValueError, match="20000"):
        QuestionFormRequest(question="x" * 20001)


@pytest.mark.parametrize("parts, cap", [(0, 5), (3, 5), (5, 5), (9, 9), (14, 14), (24, 24), (30, 24)])
def test_w34_the_section_cap_is_the_number_of_parts_between_5_and_the_safety_ceiling(parts, cap):
    assert gui.section_cap(parts_request(parts)) == cap


@pytest.mark.parametrize("cap, tokens", [(3, 1800), (5, 1800), (9, 2800), (14, 4050), (24, 6550)])
def test_w34_the_output_budget_grows_with_the_section_cap_from_exactly_1800_at_5(cap, tokens):
    assert gui.card_token_budget(cap) == tokens
    assert gui.card_token_budget(5) == gui.CARD_MAX_TOKENS == 1800


@pytest.mark.parametrize("cap, blocks", [(5, 10), (9, 18), (14, 28), (24, 48), (40, 48)])
def test_w34_the_block_cap_is_two_per_section_up_to_twice_the_section_ceiling(cap, blocks):
    assert gui.card_block_cap(cap) == blocks and gui.MAX_CARD_BLOCKS_CEILING == 2 * gui.MAX_SECTIONS == 48


def test_w34_a_fourteen_part_request_keeps_fourteen_sections_and_a_thirty_part_one_is_cut_at_the_ceiling():
    names = [f"Part {i}" for i in range(1, 31)]
    results = [ResultData(id=f"r{i}", command=f"cmd_{i}", output={"v": i}) for i in range(1, 31)]
    sections = [sec(names[i - 1], "", {"kind": "stats", "stats": [{"label": "V", "ref": f"r{i}.v"}]}) for i in range(1, 31)]
    for parts, kept in ((14, 14), (30, 24)):
        request = AnswerCardRequest(answer="\n\n".join(f"### {i}. part {i}\nDone." for i in range(1, parts + 1)), results=results)
        card, diag = _with_diag(lambda: validate_answer_card(AnswerCard(sections=sections), request))
        assert len(card.sections) == kept and [s.heading for s in card.sections] == names[:kept]
        assert diag.drops.get("section_cap", 0) == 30 - kept


def test_w34_the_instructions_and_the_field_state_the_cap_of_the_request():
    for cap in (5, 9, 14, 24):
        signature = gui._card_signature(cap, False)
        assert f"Write at most {cap} sections" in signature.instructions
        assert f"up to {cap} sections" in signature.output_fields["sections"].json_schema_extra["desc"]
        assert f"Write at most {cap} sections" in gui._card_signature(cap, True).instructions


def test_w34_a_nine_part_card_is_sent_the_scaled_budget_and_its_retry_gets_the_scaled_extra(thinker):
    request = parts_request(9)
    lm = thinker([OK_SECTIONS], finish_reason="")
    gui.compose_answer_card(request)
    assert lm.copies[0]["max_tokens"] == 2800 + gui.REASONING_HEADROOM_TOKENS
    lm = thinker([{"lead": "Done.", "sections": []}, OK_SECTIONS], finish_reason="length")
    diag = gui.ComposeDiagnostics()
    gui.compose_answer_card(request, diag)
    limits = [c["max_tokens"] for c in lm.copies]
    assert limits == [2800 + gui.REASONING_HEADROOM_TOKENS, 2800 + gui.TRUNCATION_RETRY_TOKENS + gui.REASONING_HEADROOM_TOKENS]


def test_w34_a_five_section_card_is_sent_exactly_the_old_budget(composer):
    lm = composer([OK_SECTIONS])
    gui.compose_answer_card(parts_request(5))
    assert lm.copies[0]["max_tokens"] == 1800


def test_w34_the_page_mirrors_the_server_constants():
    import re
    from pathlib import Path
    src = Path(gui.__file__).parent / "static" / "src"
    card = (src / "258-chat-card.js").read_text(encoding="utf-8")
    question = (src / "259-chat-question.js").read_text(encoding="utf-8")

    def js(text, name):
        return int(re.search(rf"var {name} = (\d+);", text).group(1))
    assert js(card, "TM_CARD_MAX_ANSWER") == gui.MAX_ANSWER_CHARS == 150000
    assert js(card, "TM_CARD_MAX_SECTIONS") == gui.MAX_SECTIONS == 24
    assert js(card, "TM_CARD_MAX_BLOCKS") == gui.MAX_CARD_BLOCKS_CEILING == 48
    assert js(card, "TM_CARD_MAX_SECTION_BLOCKS") == gui.MAX_SECTION_BLOCKS
    assert js(card, "TM_CARD_MAX_RESULTS") == gui.MAX_RESULTS
    assert js(question, "TM_QUESTION_MAX_CHARS") == gui.MAX_TEXT_CHARS == 20000
