"""The composer reads the parts of a split turn from data when the page sends them (``AnswerCardRequest.parts``) and
from the answer's ``### N.`` headings otherwise.

The model's input must be byte-identical either way for every answer whose headings are well formed: the prompt, the
``sub_tasks`` list, the section cap, the block cap, the token budget and the long-answer excerpt. The one intended
difference is a part whose own answer holds a line that looks like a heading. The parts are built the way the core
builds them (``WorkflowExecutionContext._answer_parts``, tests/test_subtask_parts.py). The model is a scripted
``DummyLM`` (the repo's one allowed seam); no test calls a live model.
"""
import litellm  # noqa: F401  (must precede dspy)
import pytest

from fastworkflow.run_chatbot import composer as gui
from fastworkflow.run_chatbot.composer import AnswerCardRequest, ResultData
from fastworkflow.engine.workflow_execution_context import _heading_title
from tests.test_composer_parts import result, section, sent  # noqa: F401


def build(entries):
    """``(answer, parts)`` for ``[(request text, answer or "")]``, as the core builds them: the combined text with a
    heading for every part that has an answer (a single answered part has none) and the parts with offsets."""
    answered = [(n, text, body) for n, (text, body) in enumerate(entries, 1) if body]
    pieces, spans, position = [], {}, 0
    for number, text, body in answered:
        if len(answered) == 1:
            spans[number] = (0, len(body))
            pieces.append(body)
            continue
        heading = f"### {number}. {_heading_title(text)}\n"
        position += len(heading)
        spans[number] = (position, position + len(body))
        position += len(body) + 2
        pieces.append(heading + body)
    answer = "\n\n".join(pieces)
    parts = [{"number": n, "text": text, "title": _heading_title(text), "answered": n in spans,
              "start": spans[n][0] if n in spans else None, "end": spans[n][1] if n in spans else None}
             for n, (text, _) in enumerate(entries, 1)]
    return answer, parts


def requests(entries, results=None, user_message="q"):
    answer, parts = build(entries)
    results = results or [result("r1", "find_book", 1, output={"title": "Dune", "pages": 412})]
    return (AnswerCardRequest(user_message=user_message, answer=answer, results=results, parts=parts),
            AnswerCardRequest(user_message=user_message, answer=answer, results=results))


def model_input(request, sent):
    lm = sent([{"lead": "", "sections": [section("Book", "r1")]}])
    gui.compose_answer_card(request)
    return lm.sent, lm.copies


LONG = "An answer sentence that goes on. " * 120       # about 4000 characters of prose
LONG_REQUEST = ("Please look up the long and winding record of the member who borrowed the oldest book in the "
                "collection last spring and tell me everything about it")

CASES = {
    "two parts": [("find the book", "Dune is on shelf 4."), ("check the fine", "The fine is 12.")],
    "nine parts": [(f"do thing number {n}: with a clause after it", f"Done {n}.\nSecond line {n}.") for n in range(1, 10)],
    "a long answer over the excerpt threshold": [(f"look up record {n}", LONG) for n in range(1, 8)],
    "titles that hit the 80 character cut": [(LONG_REQUEST + f" number {n}", f"Result {n}.") for n in range(1, 4)],
    "a part with an empty answer": [("find the book", "Dune is on shelf 4."), ("say hello", ""), ("check the fine", "The fine is 12.")],
    "a single answered part of a split turn": [("find the book", ""), ("say hello", "Hello there."), ("check the fine", "")],
    "a long answer with an empty part": [("look up record 1", LONG), ("nothing", ""), ("look up record 3", LONG),
                                        ("look up record 4", LONG), ("look up record 5", LONG), ("look up record 6", LONG)],
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_model_input_is_the_same_with_the_parts_and_without_them(name, sent):
    with_parts, without = requests(CASES[name])
    assert with_parts.parts is not None and without.parts is None, "the parts fit their answer"
    assert gui.request_sub_tasks(with_parts) == gui.request_sub_tasks(without)
    assert gui.section_cap(with_parts) == gui.section_cap(without)
    cap = gui.section_cap(with_parts)
    assert gui.card_block_cap(cap) == gui.card_block_cap(gui.section_cap(without))
    assert gui.card_token_budget(cap) == gui.card_token_budget(gui.section_cap(without))
    assert gui.answer_for_prompt(with_parts.answer, with_parts.parts) == gui.answer_for_prompt(without.answer)
    prompts_with, kwargs_with = model_input(with_parts, sent)
    prompts_without, kwargs_without = model_input(without, sent)
    assert prompts_with == prompts_without and prompts_with, "the whole prompt sent to the model"
    assert kwargs_with == kwargs_without, "the call's settings, the token budget among them"


def test_the_cases_cover_what_they_claim():
    two, _ = requests(CASES["two parts"])
    nine, _ = requests(CASES["nine parts"])
    long, _ = requests(CASES["a long answer over the excerpt threshold"])
    cut, _ = requests(CASES["titles that hit the 80 character cut"])
    empty, _ = requests(CASES["a part with an empty answer"])
    single, _ = requests(CASES["a single answered part of a split turn"])
    assert len(gui.request_sub_tasks(two)) == 2
    assert len(gui.request_sub_tasks(nine)) == 9 and gui.section_cap(nine) == 9
    assert len(long.answer) > gui.ANSWER_PROMPT_WHOLE_CHARS
    assert gui.answer_for_prompt(long.answer) != long.answer and gui.EXCERPT_CUT_MARKER in gui.answer_for_prompt(long.answer)
    assert all(len(title.split(". ", 1)[1]) <= 80 for title in gui.request_sub_tasks(cut))
    assert all(part.title.endswith("\u2026") for part in cut.parts), "the heading itself was cut at 80"
    assert [p.answered for p in empty.parts] == [True, False, True] and gui.request_sub_tasks(empty) == [
        "1. find the book", "3. check the fine"]
    assert gui.request_sub_tasks(single) == [] and gui.section_cap(single) == gui.MIN_SECTIONS


def test_an_unanswered_part_is_a_title_for_the_diagnostics_only():
    with_parts, without = requests(CASES["a part with an empty answer"])
    assert gui.part_titles(with_parts) == {1: "find the book", 2: "say hello", 3: "check the fine"}
    assert gui.part_titles(without) == {1: "find the book", 3: "check the fine"}


def test_a_part_that_wrote_a_heading_line_is_not_read_as_a_new_part():
    entries = [("find the book", "Dune.\n### 3. not a part\nMore."), ("check the fine", "The fine is 12.")]
    with_parts, without = requests(entries)
    assert with_parts.parts is not None
    # today's reading, kept as the fallback: the line in the body is a part of its own
    assert gui.request_sub_tasks(without) == ["1. find the book", "3. not a part", "2. check the fine"]
    assert gui.section_cap(without) == gui.MIN_SECTIONS
    # with the data it is what the turn was
    assert gui.request_sub_tasks(with_parts) == ["1. find the book", "2. check the fine"]


def test_the_heading_line_in_a_part_does_not_split_the_long_answer_excerpt():
    entries = [("look up record 1", "Start.\n### 9. fake\n" + LONG * 2), ("look up record 2", LONG * 2), ("look up record 3", LONG * 2)]
    with_parts, without = requests(entries)
    assert len(with_parts.answer) > gui.ANSWER_PROMPT_WHOLE_CHARS
    shown_with = gui.answer_for_prompt(with_parts.answer, with_parts.parts)
    shown_without = gui.answer_for_prompt(without.answer)
    assert "\n\n### 9. fake" in shown_without       # the fallback reads the line as a part of its own
    assert "\n\n### 9. fake" not in shown_with and "Start.\n### 9. fake\n" in shown_with   # the data: still the part's text
    assert shown_with.index("### 1.") < shown_with.index("### 2.") < shown_with.index("### 3.")


def test_a_single_answered_part_whose_text_holds_a_heading_line_has_no_parts_with_the_data():
    entries = [("find the book", ""), ("say hello", "Hi.\n### 2. x\nthere\n### 5. y\nmore")]
    with_parts, without = requests(entries)
    assert gui.request_sub_tasks(without) == ["2. x", "5. y"]       # today's mis-reading, the fallback
    assert gui.request_sub_tasks(with_parts) == [] and gui.section_cap(with_parts) == gui.MIN_SECTIONS


# ----------------------------------------------------------------------
# The request: bounded, lenient, checked against the answer
# ----------------------------------------------------------------------

def test_a_request_without_parts_is_what_it_always_was():
    request = AnswerCardRequest(answer="### 1. a\nx\n\n### 2. b\ny")
    assert request.parts is None and gui.request_sub_tasks(request) == ["1. a", "2. b"]


def good_parts():
    return build([("find the book", "Dune."), ("check the fine", "Twelve.")])


def accepted(parts, answer=None):
    built_answer, _ = good_parts()
    return AnswerCardRequest(answer=answer if answer is not None else built_answer, parts=parts).parts


def test_parts_that_fit_are_kept():
    _, parts = good_parts()
    assert [p.number for p in accepted(parts)] == [1, 2]


@pytest.mark.parametrize("break_it", [
    lambda parts: "not a list",
    lambda parts: parts[:1],                                                    # fewer than two parts
    lambda parts: parts * 13,                                                   # more than the ceiling
    lambda parts: [parts[0], {**parts[1], "number": 1}],                        # numbers not increasing
    lambda parts: [parts[0], {**parts[1], "number": gui.MAX_SUB_TASKS + 1}],    # a number out of range
    lambda parts: [parts[0], {**parts[1], "number": 0}],
    lambda parts: [parts[0], "x"],                                              # an item that is not a part
    lambda parts: [parts[0], {**parts[1], "title": "t" * (gui.MAX_PART_TITLE_CHARS + 1)}],
    lambda parts: [parts[0], {**parts[1], "text": "t" * (gui.MAX_TEXT_CHARS + 1)}],
    lambda parts: [parts[0], {**parts[1], "start": None}],                      # answered, no offsets
    lambda parts: [parts[0], {**parts[1], "start": parts[1]["start"] + 1}],     # offsets that miss the heading
    lambda parts: [parts[0], {**parts[1], "end": 10 ** 6}],                     # past the end of the answer
    lambda parts: [parts[0], {**parts[1], "title": "another title"}],           # a heading the answer does not have
    lambda parts: [parts[1], parts[0]],                                         # out of order
])
def test_parts_that_are_malformed_or_do_not_fit_are_ignored_not_refused(break_it):
    _, parts = good_parts()
    assert accepted(break_it(parts)) is None


def test_parts_of_another_answer_are_ignored():
    _, parts = good_parts()
    assert accepted(parts, answer="### 1. find the book\nDune.\n\n### 2. check the fine\nTwelve!!") is not None  # same layout
    assert accepted(parts, answer="Dune. Twelve.") is None


# ----------------------------------------------------------------------
# A part continued after a hand-off carries runs; the model never sees them
# ----------------------------------------------------------------------

def _continued_entries():
    one = "Part 1 of 2:\nDune is on shelf 4.\n\nPart 2 of 2:\nIt is checked in."
    return [("find the book", one), ("check the fine", "The fine is 12.")]


def _with_runs(entries):
    """``build`` parts, the continued first part given its runs the way the core offsets them."""
    answer, parts = build(entries)
    first = entries[0][1]
    label_one, label_two = "Part 1 of 2:\n", "\n\nPart 2 of 2:\n"
    a = parts[0]["start"] + len(label_one)
    b = first.index(label_two) + parts[0]["start"] + len(label_two)
    parts[0]["runs"] = [{"index": 1, "of": 2, "start": a, "end": first.index(label_two) + parts[0]["start"]},
                        {"index": 2, "of": 2, "start": b, "end": parts[0]["end"]}]
    return answer, parts


def test_the_model_input_is_the_same_with_runs_on_the_parts(sent):
    entries = _continued_entries()
    answer, parts = _with_runs(entries)
    results = [result("r1", "find_book", 1, output={"title": "Dune", "pages": 412})]
    with_runs = AnswerCardRequest(user_message="q", answer=answer, results=results, parts=parts)
    plain_parts = [{k: v for k, v in p.items() if k != "runs"} for p in parts]
    without_runs = AnswerCardRequest(user_message="q", answer=answer, results=results, parts=plain_parts)
    assert with_runs.parts is not None and with_runs.parts[0].runs and len(with_runs.parts[0].runs) == 2
    assert without_runs.parts is not None and without_runs.parts[0].runs is None
    assert gui.request_sub_tasks(with_runs) == gui.request_sub_tasks(without_runs)
    assert gui.answer_for_prompt(answer, with_runs.parts) == gui.answer_for_prompt(answer, without_runs.parts)
    prompts_a, kwargs_a = model_input(with_runs, sent)
    prompts_b, kwargs_b = model_input(without_runs, sent)
    assert prompts_a == prompts_b and prompts_a and kwargs_a == kwargs_b


@pytest.mark.parametrize("bad", ["not a list", [{"index": 1}], [{"index": "x", "of": 2, "start": 0, "end": 1}], [3]])
def test_malformed_runs_never_discard_the_parts(bad):
    answer, parts = _with_runs(_continued_entries())
    parts[0]["runs"] = bad
    request = AnswerCardRequest(answer=answer, parts=parts)
    assert request.parts is not None and [p.number for p in request.parts] == [1, 2]
    assert request.parts[0].runs is None


# ----------------------------------------------------------------------
# A part marked not_run (no-agent mode) is the same to the model; a malformed mark never discards the parts
# ----------------------------------------------------------------------

def test_the_model_input_is_the_same_with_not_run_on_the_parts(sent):
    entries = [("find the book", "Dune is on shelf 4."), ("plan a trip", "Not run here: it needs a plan."),
               ("check the fine", "The fine is 12.")]
    answer, parts = build(entries)
    marked = [{**p, "not_run": p["number"] == 2} for p in parts]
    results = [result("r1", "find_book", 1, output={"title": "Dune", "pages": 412})]
    with_mark = AnswerCardRequest(user_message="q", answer=answer, results=results, parts=marked)
    without_mark = AnswerCardRequest(user_message="q", answer=answer, results=results, parts=parts)
    assert [p.not_run for p in with_mark.parts] == [False, True, False]
    assert [p.not_run for p in without_mark.parts] == [False, False, False]
    assert gui.request_sub_tasks(with_mark) == gui.request_sub_tasks(without_mark)
    assert gui.answer_for_prompt(answer, with_mark.parts) == gui.answer_for_prompt(answer, without_mark.parts)
    prompts_a, kwargs_a = model_input(with_mark, sent)
    prompts_b, kwargs_b = model_input(without_mark, sent)
    assert prompts_a == prompts_b and prompts_a and kwargs_a == kwargs_b


def test_a_malformed_not_run_never_discards_the_parts():
    _, parts = good_parts()
    parts[0]["not_run"] = "yes"
    request = AnswerCardRequest(answer=good_parts()[0], parts=parts)
    assert request.parts is not None and request.parts[0].not_run is False
