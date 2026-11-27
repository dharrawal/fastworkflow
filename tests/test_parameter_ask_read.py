"""The one question builder and the one reply reader (bd fix-qoyz step 4): ``parameters/ask.py`` and ``parameters/read.py``.

1. Direct cases of ``ask.word`` and ``read.read``, with the texts written out.
2. The equivalence table: ``tests/fixtures/parameter_ask_read_equivalence.json`` holds the outputs the two wording sides
   and the two reply readers gave BEFORE they were merged (see ``tests/parameter_ask_read_corpus.py`` for what the
   table covers). Both sides must still give exactly those outputs, character for character. Step 6 of the staged plan
   changes wording and reading on purpose, one decision at a time: a step that moves an output edits the case it
   moves here, in its own commit, with its measurement.
3. The reader has one rule (a, b, c ... g of ``parameters/read.py``): its arguments are the ones the rule needs.

Real components only: the tutorial workflow's planning commands, real ``Input`` classes, the real guards and
extractor. No model is called.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.core import no_agent_mode
from fastworkflow.parameters import parameter_clarification as pcl
from fastworkflow.planning import planning_commands as pc
from fastworkflow.parameters.parameter_clarification import Extraction
from fastworkflow.parameters import ask, read
from tests import parameter_ask_read_corpus as corpus
from tests import parameter_check_corpus as check_corpus

FIXTURE = Path(__file__).parent / "fixtures" / "parameter_ask_read_equivalence.json"
NOTE = no_agent_mode.ANSWER_ONLY_NOTE


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture(scope="module")
def commands(tmp_path_factory) -> dict:
    return check_corpus.build_commands(tmp_path_factory.mktemp("ask_read_commands"))


def test_the_modules_import_no_model_stack():
    code = ("import sys; import fastworkflow.parameters.ask, fastworkflow.parameters.read; "
            "heavy = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'dspy')]; "
            "print(','.join(sorted(heavy)))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""


# ---------------------------------------------------------------- 1. direct cases of word

def _MISSING_MEMBER() -> pc.PlanningCommandInputError:
    """The refusal of a guard that finds ``member`` simply not there: the one argument it names is data (``gap``)."""
    from fastworkflow.parameters import check

    return pc.PlanningCommandInputError("missing required argument member",
                                        gap=check.Gap("member", check.MISSING, "missing", "", None,
                                                      "missing required argument member"))


def test_the_one_argument_question(commands):
    question = ask.word(form=ask.FORM_ARGUMENT, command=commands["member_loans"], name="member")
    assert question == ask.Question("I can't run member loans yet: your request does not say which member to use. "
                                    "Which member should I use?")


def test_the_gaps_question_names_every_gap(commands):
    gaps = Extraction(missing=("member", "branch"), invalid=(("entity_type", "is not one of identity, account"),))
    text = ask.word(gaps, form=ask.FORM_GAPS, command=commands["transfer_loan"]).text
    assert text == ("I can't run transfer loan yet: your request does not give the member and branch; "
                    "the entity type is not one of identity, account. What should the member, branch and entity type be? "
                    "Reply with name: value lines (member: ...; branch: ...), or with the values in order separated by commas.")


def test_a_refusal_question_has_no_note_the_one_who_puts_it_to_a_person_adds_it(commands):
    command = commands["member_loans"]
    refusal = _MISSING_MEMBER()
    question = ask.word(command=command, refusal=refusal)
    assert NOTE not in question.text
    assert ask.with_answer_only_note(question.text) == f"{question.text} {NOTE}"
    assert pcl.planning_question(command, {}, Extraction(missing=("member",)), refusal=refusal)[0] == \
        f"{question.text} {NOTE}"
    assert question.request == {"command": "member_loans", "missing": ["member"], "invalid": [], "values": {}}


def test_a_retry_opens_the_question(commands):
    refusal = _MISSING_MEMBER()
    text = ask.word(command=commands["member_loans"], refusal=refusal, retry=pcl.RETRY).text
    assert text.startswith("That did not give a value I could use. I can't run member loans yet")


def test_a_refusal_with_none_is_made_from_the_gaps(commands):
    gaps = Extraction(missing=("member",))
    made = ask.word(gaps, command=commands["member_loans"])
    given = ask.word(command=commands["member_loans"], refusal=_MISSING_MEMBER())
    assert made == given


def test_the_confirm_form_presents_every_parameter(commands):
    command = commands["transfer_loan"]
    gaps = Extraction(missing=("branch",))
    question = ask.word(gaps, command=command, confirm=True, values={"member": "Ann"}, description="Moves a loan.")
    assert question.text == (
        "You chose the planning command transfer_loan: Moves a loan.\nIts parameters:\n"
        "- member (required): Ann\n- branch (required): not given\n- entity_type (optional): not given\n"
        "- note (optional): not given\n"
        "Reply with one name: value line for each value to set or change, or yes to run with the values shown "
        f"(the ones not given must be given first); cancel stops. {NOTE}")
    assert question.request["confirm"] is True and question.request["missing"] == ["branch"]
    assert [p["state"] for p in question.request["parameters"]] == [None, "missing", None, None]


def test_the_validation_message_of_a_human_session():
    parts = ask.Parts(missing=["uid", "name"], invalid=["code 'x'"],
                      available_from={"uid": ["Library/find_book.py", "Library/list_books"]},
                      notes=["a note"], refusals=["a rule"], range_notes=["a range"],
                      suggestions={"code": ["Please use the format matching pattern ^a$ (e.g., a)"]})
    assert ask.word(form=ask.FORM_VALIDATION, command_name="Library/find_member", parts=parts).text == (
        "I can't run find member yet: your request does not give the uid and name; a rule. "
        "What should the uid, name and code be? "
        "Reply with name: value lines (uid: ...; name: ...), or with the values in order separated by commas.\n"
        "abort and use the find_book or list_books command(s) to get uid information.\n"
        "a note\na range\n"
        "code: Please use the format matching pattern ^a$ (e.g., a)")


def test_the_validation_message_of_a_human_session_with_one_open_value_teaches_the_value():
    assert ask.word(form=ask.FORM_VALIDATION, command_name="find_member", parts=ask.Parts(missing=["uid"])).text == (
        "I can't run find member yet: your request does not give the uid. Which uid should I use?")


def test_the_validation_message_of_an_agent_session_teaches_no_comma_protocol():
    parts = ask.Parts(missing=["uid"], available_from={"uid": ["Library/find_book.py"]})
    assert ask.word(form=ask.FORM_VALIDATION, audience=ask.AGENT, parts=parts).text == (
        "Missing parameter values: uid\nuse the find_book command(s) to get uid information. OR...\n"
        "\nProvide corrected parameter values in the exact order specified below, separated by commas:\nuid")


def test_a_suggestion_that_is_not_a_format_is_offered_as_a_choice():
    parts = ask.Parts(missing=["x"], suggestions={"x": ["a", "b"], "y": []})
    assert ask.word(form=ask.FORM_VALIDATION, command_name="find_member", parts=parts).text == (
        "I can't run find member yet: your request does not give the x. Which x should I use?\n"
        "Did you mean one of these xs? a, b")


def test_a_validation_with_nothing_open_has_no_how_to_reply():
    assert ask.word(form=ask.FORM_VALIDATION, parts=ask.Parts(refusals=["the call does not fit"])).text == \
        "the call does not fit\n"


def test_the_frame_of_a_human_and_an_agent_session():
    kwargs = dict(form=ask.FORM_FRAME, message="MSG", refusals=["x was not used"], shown="a: 1")
    assert ask.word(**kwargs).text == (
        "Extracted parameters so far:\na: 1\n\nValues not used: x was not used\nMSG\n"
        f"{NOTE}\nEnter 'abort' to get out of this error state and/or execute a different command."
        "\nEnter 'you misunderstood' if the wrong command was executed.")
    assert ask.word(audience=ask.AGENT, **kwargs).text == (
        "Extracted parameters so far:\na: 1\n\nValues not used: x was not used\nMSG\n"
        "Check your command name if the wrong command was executed.")


def test_the_lead_line_names_the_command():
    assert ask.word(form=ask.FORM_LEAD, command_name="Book/find", message="MSG").text == \
        "PARAMETER EXTRACTION ERROR FOR COMMAND 'Book/find'\nMSG"


def test_open_names_are_the_missing_then_the_refused_without_their_values():
    assert ask.open_names(["a", "b"], ["c 'x y'", "d ''"]) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------- 1. direct cases of read

NAMES = ["title", "author", "year"]


SLOTS = {"flag": read.Slot(boolean=True), "mode": read.Slot(options=("fast", "slow")), "note": read.Slot(required=False),
         "tag": read.Slot()}
LISTED = ["flag", "mode", "note", "tag"]


def _read(reply, open_names, listed=None, **kwargs):
    reading = read.read(reply, NAMES if listed is None else listed, open_names, **kwargs)
    return reading.kind, reading.values


@pytest.mark.parametrize("reply, open_names, kwargs, expected", [
    # a. cancel declines; yes accepts only a confirmation
    ("cancel", ["title"], {}, ("decline", {})),
    ("Never  Mind.", ["title"], {}, ("decline", {})),
    ("cancel", ["title", "author"], {"confirm": True}, ("decline", {})),
    ("yes", ["title"], {}, ("values", {"title": "yes"})),
    ("yes", ["title"], {"confirm": True}, ("accept", {})),
    ("yes", [], {}, ("free", {})),
    ("yes", [], {"confirm": True}, ("accept", {})),
    # b. name lines set listed fields; an unlisted field is never set
    ("TITLE = Dune", ["title"], {}, ("values", {"title": "Dune"})),
    ("title: Dune\nunknown: x\nauthor: Herbert", ["title", "author"], {}, ("values", {"title": "Dune", "author": "Herbert"})),
    ("title: Dune, author: Herbert", ["title", "author"], {}, ("values", {"title": "Dune", "author": "Herbert"})),
    ("title: Dune, Part One", ["title"], {}, ("values", {"title": "Dune, Part One"})),
    # g. anything else is ignored
    ("title: Dune\nand some prose", ["title"], {}, ("values", {"title": "Dune"})),
    # d. one open value: the whole reply, commas included, quotes removed
    ("Smith, John", ["title"], {}, ("values", {"title": "Smith, John"})),
    ("\"Dune, Part One\"", ["title"], {}, ("values", {"title": "Dune, Part One"})),
    ("'Dune'", ["title"], {}, ("values", {"title": "Dune"})),
    ("a long reply, with several commas, and words", ["title"], {}, ("values", {"title": "a long reply, with several commas, and words"})),
    # e. several open values: comma parts in order, only when the counts match
    ("Dune, Herbert", ["title", "author"], {}, ("values", {"title": "Dune", "author": "Herbert"})),
    ("Dune, Herbert, 1965", ["title", "author"], {}, ("free", {})),
    ("Dune", ["title", "author"], {}, ("free", {})),
    ("Dune, , Herbert", ["title", "author", "year"], {}, ("free", {})),
    ("\"Dune, Part One\", Herbert", ["title", "author"], {}, ("values", {"title": "Dune, Part One", "author": "Herbert"})),
    ("\"Dune, Part One\"", ["title", "author"], {}, ("free", {})),
    # f. a value without a name fills open fields only: title is listed (shown) but not open
    ("Herbert", ["author"], {}, ("values", {"author": "Herbert"})),
    ("Dune, Herbert", ["author"], {}, ("values", {"author": "Dune, Herbert"})),
    ("hello there", ["title", "author"], {}, ("free", {})),
    ("", ["title"], {}, ("free", {})),
    ("Dune", [], {}, ("free", {})),
])
def test_the_one_reading(reply, open_names, kwargs, expected):
    assert _read(reply, open_names, **kwargs) == expected


def test_a_named_line_sets_only_a_listed_field():
    assert _read("year: 1965", ["title"], listed=["title"]) == ("values", {"title": "year: 1965"})
    assert _read("year: 1965\ntitle: Dune", ["title"], listed=["title"]) == ("values", {"title": "Dune"})
    assert _read("year: 1965\ntitle: Dune", ["title"], listed=["title", "year"]) == ("values", {"title": "Dune", "year": "1965"})


@pytest.mark.parametrize("reply, open_names, expected", [
    # c. a yes/no field: the word is the value
    ("no", ["flag"], ("values", {"flag": "no"})),
    ("None", ["flag"], ("values", {"flag": "None"})),
    ("yes", ["flag"], ("values", {"flag": "yes"})),
    ("nope", ["flag"], ("values", {"flag": "nope"})),
    # an enum field: an allowed value, whatever its case, as declared; other no-words are not values of it
    ("FAST", ["mode"], ("values", {"mode": "fast"})),
    ("slow", ["mode"], ("values", {"mode": "slow"})),
    ("no", ["mode"], ("free", {})),
    ("yes", ["mode"], ("values", {"mode": "yes"})),
    # an optional plain field: no-words leave it empty
    ("no", ["note"], ("values", {"note": None})),
    ("None", ["note"], ("values", {"note": None})),
    ("neither", ["note"], ("values", {"note": None})),
    ("n", ["note"], ("values", {"note": None})),
    ("nope", ["note"], ("values", {"note": None})),
    ("yes", ["note"], ("values", {"note": "yes"})),
    ("no thanks", ["note"], ("values", {"note": "no thanks"})),
    # a required plain field: no-words are unclear, anything else is the value
    ("no", ["tag"], ("free", {})),
    ("none", ["tag"], ("free", {})),
    ("neither", ["tag"], ("free", {})),
    ("yes", ["tag"], ("values", {"tag": "yes"})),
    ("none of them", ["tag"], ("values", {"tag": "none of them"})),
    # several open values: a bare no is one part for several, unclear
    ("no", ["flag", "tag"], ("free", {})),
    ("no, red", ["flag", "tag"], ("values", {"flag": "no", "tag": "red"})),
    ("flag: no", ["flag", "tag"], ("values", {"flag": "no"})),
])
def test_a_bare_yes_or_no(reply, open_names, expected):
    assert _read(reply, open_names, listed=LISTED, slots=SLOTS) == expected


GUARD_SLOTS = {"flag": read.Slot(boolean=True), "mode": read.Slot(options=("fast", "slow")),
               "count": read.Slot(number=True), "tag": read.Slot(), "who": read.Slot()}


@pytest.mark.parametrize("reply, open_names, expected", [
    # (i) count, (ii) no named part, (iii) every part fits its field
    ("Jane Austen, Pride and Prejudice", ["who", "tag"], ("values", {"who": "Jane Austen", "tag": "Pride and Prejudice"})),
    ("yes, 3", ["flag", "count"], ("values", {"flag": "yes", "count": "3"})),
    ("FAST, red", ["mode", "tag"], ("values", {"mode": "fast", "tag": "red"})),
    ("slow, 2.5", ["mode", "count"], ("values", {"mode": "slow", "count": "2.5"})),
    ("maybe, 3", ["flag", "count"], ("free", {})),           # a bool field takes a yes/no word
    ("no, three", ["flag", "count"], ("free", {})),          # a number field takes a number
    ("medium, red", ["mode", "tag"], ("free", {})),          # an allowed-values field takes one of them
    ("name: Ann, dept: Sales", ["who", "tag"], ("free", {})),  # named parts with names the question does not have
    ("Dune: Part Two, Herbert", ["who", "tag"], ("free", {})),
    ("Dune, Herbert, 1965", ["who", "tag"], ("free", {})),
])
def test_several_open_values_by_commas_only_when_the_parts_fit(reply, open_names, expected):
    assert _read(reply, open_names, listed=list(GUARD_SLOTS), slots=GUARD_SLOTS) == expected


def test_slots_are_made_from_annotations_and_type_text():
    from typing import Literal, Optional
    import enum

    class Colour(enum.Enum):
        RED = "red"

    assert read.slot_of(bool) == read.Slot(boolean=True)
    assert read.slot_of(Optional[bool], required=False) == read.Slot(boolean=True, required=False)
    assert read.slot_of(Literal["a", "b"]) == read.Slot(options=("a", "b"))
    assert read.slot_of(Colour) == read.Slot(options=("red",))
    assert read.slot_of(str) == read.Slot()
    assert read.slot_of_text("bool") == read.Slot(boolean=True)
    assert read.slot_of_text("Literal['a', 'b'] | None") == read.Slot(options=("a", "b"))
    assert read.slot_of_text("str") == read.Slot()


def test_split_values_keeps_quoted_commas_together():
    assert read.split_values('a, "b, c", d') == ["a", "b, c", "d"]
    assert read.split_values("a,") == ["a", ""]
    assert read.split_values("O'Brien, Pat") == ["O'Brien", "Pat"]


def test_a_bare_no_is_not_you_misunderstood():
    """The CME's check of a reply to a parameter question (it passes no candidates) no longer reads a bare no as
    "you misunderstood"; the command's own phrases and abort still are."""
    from fastworkflow.intent import intent_loop

    aliases = {"you misunderstood": intent_loop.intent_clarification.NONE, "abort": intent_loop.ABORT}
    for word in ("no", "none", "neither", "n", "nope", "something else", "yes"):
        assert intent_loop.read(word, [], aliases=aliases).kind == intent_loop.UNCLEAR, word
    assert intent_loop.read("you misunderstood", [], aliases=aliases).kind == intent_loop.intent_clarification.NONE
    assert intent_loop.read("cancel", [], aliases=aliases).kind == intent_loop.ABORT
    # with candidates the which-command question still reads none of these
    assert intent_loop.read("none", ["a/b"], aliases=aliases).kind == intent_loop.intent_clarification.NONE


# ---------------------------------------------------------------- 2. the equivalence table

@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_table_is_broad(table):
    counts = table["counts"]
    assert counts["wording_plan_entries"] + counts["exec_wording"] + counts["frame"] >= 120
    assert counts["read_reply"] + counts["merge"] >= 150


@pytest.mark.usefixtures("no_planner_key")
def test_both_sides_still_give_exactly_the_frozen_outputs(table, tmp_path):
    current = corpus.build_table(tmp_path)
    moved = {section: [key for key in table[section] if table[section][key] != current[section].get(key)]
             for section in ("wording_plan", "wording_exec", "frame", "read_reply", "merge")}
    assert {section: keys[:3] for section, keys in moved.items() if keys} == {}
    for section in moved:
        assert set(current[section]) == set(table[section])
    assert current["counts"] == table["counts"]


# ---------------------------------------------------------------- 3. every difference-carrying argument is live

def test_the_reading_arguments_are_the_one_rule():
    import inspect

    assert list(inspect.signature(read.read).parameters) == ["reply", "names", "open_names", "confirm", "slots"]
    assert "positional" not in inspect.signature(pcl.read_reply).parameters


def test_confirm_decides_whether_yes_accepts():
    assert read.read("yes", NAMES, ["title"], confirm=True) != read.read("yes", NAMES, ["title"], confirm=False)


def test_slots_decide_a_bare_no():
    assert read.read("no", LISTED, ["note"], slots=SLOTS) != read.read("no", LISTED, ["tag"], slots=SLOTS)
    assert read.read("no", LISTED, ["flag"], slots=SLOTS) != read.read("no", LISTED, ["tag"], slots=SLOTS)


def _gaps_case(commands) -> dict:
    command = commands["transfer_loan"]
    return dict(command=command, refusal=_MISSING_MEMBER())


def test_every_wording_argument_changes_a_case(commands):
    base = _gaps_case(commands)
    assert ask.word(retry="", **base) != ask.word(retry=pcl.RETRY, **base)
    gaps = Extraction(missing=("member",))
    plain = dict(command=commands["transfer_loan"], values={"branch": "North"})
    assert ask.word(gaps, confirm=False, **plain) != ask.word(gaps, confirm=True, **plain)
    assert ask.word(gaps, confirm=True, description="", **plain) != ask.word(gaps, confirm=True, description="A.", **plain)
    assert ask.word(gaps, confirm=True, **plain) != ask.word(gaps, confirm=True, **{**plain, "values": {"branch": "South"}})
    parts = ask.Parts(missing=["uid"], available_from={"uid": ["a/b.py"]})
    assert ask.word(form=ask.FORM_VALIDATION, audience=ask.HUMAN, parts=parts) != \
        ask.word(form=ask.FORM_VALIDATION, audience=ask.AGENT, parts=parts)
    framed = dict(form=ask.FORM_FRAME, message="M")
    assert ask.word(audience=ask.HUMAN, **framed) != ask.word(audience=ask.AGENT, **framed)
    assert ask.word(shown="a: 1", **framed) != ask.word(shown="", **framed)
    assert ask.word(refusals=["r"], **framed) != ask.word(refusals=[], **framed)
    # one wording for a person on both sides (an agent is taught the comma protocol instead)
    person = ask.word(gaps, form=ask.FORM_GAPS, command=commands["transfer_loan"]).text
    assert person == ask.word(form=ask.FORM_VALIDATION, command_name="Ctx/transfer_loan",
                              parts=ask.Parts(missing=["member"])).text
    assert person != ask.word(form=ask.FORM_VALIDATION, command_name="transfer_loan", audience=ask.AGENT,
                              parts=ask.Parts(missing=["member"])).text


# ---------------------------------------------------------------- the audience is the VALUE of run_as_agent

@pytest.mark.parametrize("context, expected", [
    ({}, ask.HUMAN), ({"run_as_agent": False}, ask.HUMAN), ({"run_as_agent": None}, ask.HUMAN),
    ({"run_as_agent": True}, ask.AGENT), (None, ask.HUMAN)])
def test_the_audience_is_decided_by_the_value_of_run_as_agent_not_the_keys_presence(context, expected):
    assert ask.audience_of(context) == expected


# ---------------------------------------------------------------- the gaps are structured, never read from a message

def test_a_refused_value_that_contains_is_keeps_its_value_and_its_reason(commands):
    """The refusal of one argument is a ``Gap`` (name, why, value), end to end: a value with " is " in it used to be
    cut at that word when the message text was parsed."""
    command = commands["member_loans"]
    request = "Which books does that member have on loan?"
    value = "the cat is on the mat"
    gaps = pc.argument_gaps(command, {"member": value}, request)
    assert gaps.invalid == (("member", "is not written in the request"),) and gaps.missing == ()
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {"member": value}, request)
    assert refused.value.gap.name == "member" and refused.value.gap.value == value
    assert refused.value.gap.why == "is not written in the request"
    question = pc.refusal_question(command, refused.value)
    assert question == ("I can't run member loans yet: the member 'the cat is on the mat' is not in your request. "
                        "Which member should I use?")
    assert pc.refusal_gaps(command, refused.value).invalid == (("member", "is not written in the request"),)


def test_no_message_parser_is_left_in_the_planning_side():
    import fastworkflow.planning.planning_commands as module

    assert not hasattr(module, "_refusal_gap") and not hasattr(module, "_ARGUMENT_REFUSAL")
    assert not hasattr(module, "_MISSING_ARGUMENT")
