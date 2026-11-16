"""CME prompt artifacts for a human ("/") session: parameter_request and intent_choices (fix-weih.2).

Real workflows and the real extractor, router and executor. The doubles are the LLM extractor
(``InputForParamExtraction.extract_parameters``), ``CommandNamePrediction.predict`` for the
ambiguity passthrough, and a ``CommandRouter`` stand-in that fails if a classifier is asked.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import fastworkflow
from fastworkflow import CommandOutput, NLUPipelineStage
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
    parameter_request_for,
)
from fastworkflow.command_executor import _wildcard_command_output
from fastworkflow.utils.signatures import (
    INVALID_FLOAT_VALUE,
    INVALID_INT_VALUE,
    NOT_FOUND,
    InputForParamExtraction,
)

TESTS = Path(__file__).parent
PLAIN = str((TESTS / "plain_required_workflow").resolve())
ONE_HOOK = str((TESTS / "one_hook_workflow").resolve())

ASSISTANT = {"is_assistant_mode_command": True}
CANDIDATES = ["Library/find_book", "Library/find_member"]


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture(autouse=True)
def _no_llm():
    """Every extraction here is bound in code; a call to the LLM extractor is a failure."""
    def fake(self, cls, command_name, folder):
        raise AssertionError("the LLM ran")

    with patch.object(InputForParamExtraction, "extract_parameters", fake):
        yield


def _app(folder: str):
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    return fastworkflow.Workflow.create(folder, workflow_id_str=f"pr_{uuid.uuid4().hex[:8]}")


def _extract(wf, command_name, text, cme_context, run_as_agent=True):
    if run_as_agent:
        c = wf.context
        c["run_as_agent"] = True
        wf.context = c
    return ParameterExtraction(
        SimpleNamespace(context=dict(cme_context)), wf, command_name, text).extract()


RECORD = "record_values"
FULL = f'{RECORD}: {RECORD} name=Bob count=3 ratio=1.5 ids=["a","b"]'
PARTIAL = f"{RECORD}: {RECORD} name=Bob count=3"


# 1
def test_assistant_mode_failure_carries_a_parameter_request():
    out = _extract(_app(PLAIN), RECORD, PARTIAL, ASSISTANT)
    assert out.parameters_are_valid is False
    request = out.parameter_request
    assert set(request["missing"]) == {"ratio", "ids"}
    assert request == {
        "command": RECORD, "missing": request["missing"], "invalid": [],
        "values": {"name": "Bob", "count": 3},
    }
    json.dumps(request)  # JSON-safe


# 2
def test_agent_mode_has_no_parameter_request_and_the_same_text():
    wf = _app(PLAIN)
    agent = _extract(wf, RECORD, PARTIAL, {})
    assert agent.parameters_are_valid is False
    assert agent.parameter_request is None
    assert "ratio" in agent.error_msg and "ids" in agent.error_msg
    human = _extract(_app(PLAIN), RECORD, PARTIAL, ASSISTANT)
    assert human.error_msg == agent.error_msg


# 3
def test_a_field_refusal_is_reported_as_invalid():
    out = _extract(_app(ONE_HOOK), "pick_seat", "pick_seat: pick_seat seat=B9 note=window side", ASSISTANT)
    assert out.parameters_are_valid is False
    assert out.parameter_request == {
        "command": "pick_seat", "missing": [], "invalid": ["seat"], "values": {"note": "window side"},
    }


# 4
def test_a_whole_call_refusal_has_no_parameter_request():
    out = _extract(_app(ONE_HOOK), "strict_check", "strict_check: strict_check count=5", ASSISTANT)
    assert out.parameters_are_valid is False
    assert out.parameter_request is None


# 5
def test_valid_input_has_no_parameter_request():
    out = _extract(_app(PLAIN), RECORD, FULL, ASSISTANT)
    assert out.parameters_are_valid is True
    assert out.parameter_request is None


# 6
@pytest.fixture
def todo_cme(tmp_path):
    folder = tmp_path / "todo_list_workflow"
    shutil.copytree(
        TESTS / "todo_list_workflow", folder,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    app = fastworkflow.Workflow.create(
        workflow_folderpath=str(folder), workflow_id_str=f"amb-{tmp_path.name}")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    return app, cme


def _route_ambiguous(app, cme, *, assistant: bool):
    c = cme.context
    if assistant:
        c["is_assistant_mode_command"] = True
    cme.context = c
    c = app.context
    c["run_as_agent"] = True
    app.context = c
    ambiguous = CommandNamePrediction.Output(error_msg="ambiguous", candidates=list(CANDIDATES) + CANDIDATES[:1])
    with patch.object(CommandNamePrediction, "predict", return_value=ambiguous):
        route = wildcard.ResponseGenerator()(cme, "frobnicate", None)
    framed = CommandOutput(command_response=fastworkflow.CommandResponse(
        response=route.model_dump_json(), output=route))
    return route, _wildcard_command_output(framed)


def test_ambiguity_choices_reach_the_artifacts_in_assistant_mode(todo_cme):
    route, out = _route_ambiguous(*todo_cme, assistant=True)
    assert route.intent_choices == CANDIDATES  # duplicates removed, order kept
    assert out.command_response.artifacts["intent_choices"] == CANDIDATES
    assert out.command_response.response.startswith("Ambiguous intent error for command 'frobnicate'\nambiguous")


def test_ambiguity_choices_never_reach_the_agent(todo_cme):
    route, out = _route_ambiguous(*todo_cme, assistant=False)
    assert route.intent_choices is None
    assert "intent_choices" not in out.command_response.artifacts
    assert out.command_response.response.startswith("Ambiguous intent error for command 'frobnicate'\nambiguous")


def test_parameter_request_is_copied_to_the_artifacts():
    route = wildcard.Signature.Output(
        handled=False, routed=False, response="x",
        parameter_request={"command": "c", "missing": ["a"], "invalid": [], "values": {}})
    framed = CommandOutput(command_response=fastworkflow.CommandResponse(response="x", output=route))
    out = _wildcard_command_output(framed)
    assert out.command_response.artifacts["parameter_request"]["missing"] == ["a"]
    assert "intent_choices" not in out.command_response.artifacts


# 7
class _RouterThatMustNotClassify:
    built = 0

    def __init__(self, path):
        type(self).built += 1
        self.modelpipeline = None

    def predict_with_details(self, command):
        raise AssertionError("the classifier was asked")

    predict = predict_with_details


@pytest.mark.parametrize("reply", ["find_book", "/find_book"])
def test_clarification_reply_matches_the_stored_candidates_without_classifying(todo_cme, monkeypatch, reply):
    _, cme = todo_cme
    monkeypatch.setattr(intent_detection, "CommandRouter", _RouterThatMustNotClassify)
    _RouterThatMustNotClassify.built = 0
    c = cme.context
    c["command"] = "find something"
    cme.context = c
    predictor = CommandNamePrediction(cme)
    CommandNamePrediction._store_suggested_commands(predictor.path, CANDIDATES, 1)

    trace: dict = {}
    result = predictor._predict_impl(
        "TodoListManager", reply, NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION, trace)

    assert result.command_name == "Library/find_book"
    assert trace["matcher_layer"] == "exact_prefix"
    # No ``___command_info`` was copied, so a real router could not have loaded; the stand-in is built
    # once, after the match, only to embed the reply for the clarification cache.
    assert _RouterThatMustNotClassify.built <= 1


# 8
def test_parameter_request_for_deduplicates_and_prefers_missing():
    params = SimpleNamespace(model_dump=lambda **kw: {"a": 1, "b": 2, "c": 3, "d": 4})
    request = parameter_request_for(
        "cmd", params, {"missing_fields": ["a", "a", "b"], "invalid_fields": ["b", "c", "c"]})
    assert request == {
        "command": "cmd", "missing": ["a", "b"], "invalid": ["c"], "values": {"d": 4}}


def test_parameter_request_for_is_none_without_fields():
    params = SimpleNamespace(model_dump=lambda **kw: {"a": 1})
    assert parameter_request_for("cmd", params, {}) is None
    assert parameter_request_for("cmd", params, {"missing_fields": [], "invalid_fields": None}) is None


def test_parameter_request_values_exclude_not_given_sentinels():
    dumped = {"n": NOT_FOUND, "i": INVALID_INT_VALUE, "f": INVALID_FLOAT_VALUE, "ok": 0, "flag": False, "x": "y"}
    params = SimpleNamespace(model_dump=lambda **kw: dict(dumped))
    request = parameter_request_for("cmd", params, {"missing_fields": ["m"]})
    assert request["values"] == {"ok": 0, "flag": False, "x": "y"}


# Other ambiguity exits: the clarification stages never re-ask (fix-weih.2 follow-up)
def test_unmatched_clarification_reply_ends_the_question_instead_of_asking_again(todo_cme, monkeypatch):
    _, cme = todo_cme
    monkeypatch.setattr(intent_detection, "CommandRouter", _RouterThatMustNotClassify)
    c = cme.context
    c["command"] = "find something"
    cme.context = c
    predictor = CommandNamePrediction(cme)
    CommandNamePrediction._store_suggested_commands(predictor.path, CANDIDATES, 1)

    result = predictor.predict(
        "TodoListManager", "zzzz qqqq nothing", NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION)

    assert result.error_msg is None and result.candidates is None
    assert result.command_name == "IntentDetection/what_can_i_do"


# A leading "/" that reaches a first message's intent detection (fix-weih.2 follow-up)
def test_slash_core_command_name_on_a_first_message_is_matched_exactly(todo_cme, monkeypatch):
    _, cme = todo_cme
    _RouterThatMustNotClassify.built = 0
    monkeypatch.setattr(intent_detection, "CommandRouter", _RouterThatMustNotClassify)
    trace: dict = {}
    result = CommandNamePrediction(cme)._predict_impl(
        "TodoListManager", "/what_can_i_do", NLUPipelineStage.INTENT_DETECTION, trace)
    assert result.command_name == "IntentDetection/what_can_i_do"
    assert trace["matcher_layer"] == "exact_prefix"
    assert _RouterThatMustNotClassify.built == 0


def test_slash_name_owned_by_another_context_gets_the_routing_hint(todo_cme, monkeypatch):
    _, cme = todo_cme
    _RouterThatMustNotClassify.built = 0
    monkeypatch.setattr(intent_detection, "CommandRouter", _RouterThatMustNotClassify)
    result = CommandNamePrediction(cme).predict(
        "TodoItem", "/create_todo_list my list", NLUPipelineStage.INTENT_DETECTION)
    assert result.command_name is None
    assert result.routing_hint
    assert _RouterThatMustNotClassify.built == 0
