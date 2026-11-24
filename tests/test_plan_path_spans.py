"""The plan path writes the spans the "/" path writes (span contract v58): one ``fw.nlu.intent`` span per decision of a
part (never one per read of the turn's memo) and one ``fw.nlu.param_extraction`` span for the extraction of the part's
own command, both carrying the part's number, written by the same functions the CME's stages call
(``intent_detection.traced_decision``, ``parameter_extraction.traced_extraction``).

Library example, no agent model, the matcher and the extractor scripted (``tests/test_parameter_loop.py``)."""
from __future__ import annotations

import inspect

import pytest

from fastworkflow import tracing
from fastworkflow.utils.signatures import InputForParamExtraction
from tests.conftest import script_matcher
from tests.test_parameter_loop import HOST, no_agent, _offline, BOOK_PLAN, ABOUT_EMMA, ABOUT_DUNE  # noqa: F401
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_approval_binding import annotated_library, grant_root, todo, no_live_model, no_model_call_for_parameters  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FIND_BOOK, AUTHOR = "Library/find_book", "Library/author_books"


def _spans(env, name: str, since: int = 0) -> list:
    return [span for span in env.sink.spans[since:] if span.name == name]


def test_one_intent_span_per_decision_with_the_part_number_and_the_chain_facts(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA, ABOUT_DUNE]
    no_agent.ctx.process_turn(f"{ABOUT_EMMA}. {ABOUT_DUNE}", approval_surface=HOST)
    spans = _spans(no_agent, tracing.SPAN_NLU_INTENT)
    by_part = {span.attributes["part"]: span for span in spans}
    assert sorted(by_part) == [1, 2] and len(spans) == 2, [s.attributes.get("utterance") for s in spans]   # not one per memo read
    for number, text in ((1, ABOUT_EMMA), (2, ABOUT_DUNE)):
        attributes = by_part[number].attributes
        assert attributes["utterance"] == text and attributes["stage"] == "INTENT_DETECTION"
        assert attributes["engine"] and attributes["engine_chain"]          # the chain's facts, as the CME's span has them
        assert "resolved" in attributes and "command_name" in attributes
        assert attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == 13 if tracing.ATTR_SPAN_CONTRACT_VERSION in attributes else True


def test_an_ambiguity_is_traced_with_its_candidates(no_agent, monkeypatch):
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0})
    no_agent.parts = [JOKE]
    no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    (span,) = _spans(no_agent, tracing.SPAN_NLU_INTENT)
    assert span.attributes["part"] == 1 and span.attributes["ambiguous"] is True
    assert span.attributes["engine"] == "question" and set(span.attributes["candidates"]) >= {FIND_BOOK, AUTHOR}


def test_the_extraction_of_the_part_s_own_command_is_traced_with_the_part_number(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]
    no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)
    spans = [s for s in _spans(no_agent, tracing.SPAN_NLU_PARAM_EXTRACTION) if s.attributes.get("part") == 1]
    assert spans, "the part's own extraction wrote no span"
    attributes = spans[0].attributes
    assert attributes["command_name"] == BOOK_PLAN and attributes["parameters_valid"] is True
    assert attributes["extraction_method"] == "llm"


def test_a_refused_value_is_traced_as_the_cme_traces_it(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="it"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    text = "who wrote it and how many copies"
    no_agent.parts = [text]
    no_agent.ctx.process_turn(text, approval_surface=HOST)
    span = next(s for s in _spans(no_agent, tracing.SPAN_NLU_PARAM_EXTRACTION) if s.attributes.get("part") == 1)
    assert span.attributes["parameters_valid"] is False and span.attributes["missing_fields"] == ["title"]


def test_the_not_in_request_advisory_is_on_the_plan_path_s_extraction_span(no_agent, monkeypatch):
    """The CME's span carries ``argument_check_not_in_request`` for an unmarked field (``start``) whose value the request
    does not write; the plan path's span for an execution command's own extraction carries it from the same guard."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(query="windmills", start=7))
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.95, "none": 0.0})
    text = "find a book about windmills"
    no_agent.parts = [text]
    no_agent.ctx.process_turn(text, approval_surface=HOST)
    span = next(s for s in _spans(no_agent, tracing.SPAN_NLU_PARAM_EXTRACTION) if s.attributes.get("part") == 1)
    assert span.attributes.get("argument_check_not_in_request") == ["start"], span.attributes


def test_both_paths_call_the_one_writer():
    from fastworkflow._workflows.command_metadata_extraction import intent_detection, parameter_extraction
    from fastworkflow._workflows.plan_metadata_extraction import resolution
    from fastworkflow import review_proposals

    assert "traced_decision" in inspect.getsource(intent_detection.CommandNamePrediction.predict)
    assert "traced_decision" in inspect.getsource(resolution._decided)
    assert "traced_extraction" in inspect.getsource(parameter_extraction.ParameterExtraction.extract)
    assert "traced_extraction" in inspect.getsource(review_proposals.draft_inputs)
