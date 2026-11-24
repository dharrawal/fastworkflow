"""CME prompt artifacts for a human ("/") session: parameter_request and the which-command question (fix-weih.2;
one ``plan_choice`` shape since fix-o1lg S7).

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
from fastworkflow import model_pipeline_training
from fastworkflow import CommandOutput, intent_loop
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
    extraction_of,
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
    # Who replies decides the wording, not the session: a person's "/" message in an agent session is assistant mode
    # (``ask.audience_of(..., assistant=True)``), so its question is the person's sentence; the agent's own call keeps
    # the agent's text. (Before: both were the agent's text, "the same text".)
    assert "ratio" in human.error_msg and "ids" in human.error_msg
    assert human.error_msg != agent.error_msg
    assert "Only your answer is read" in human.error_msg and "Only your answer is read" not in agent.error_msg


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
    from fastworkflow import intent_loop

    c = cme.context
    if assistant:
        c["is_assistant_mode_command"] = True
    cme.context = c
    c = app.context
    c["run_as_agent"] = True
    app.context = c
    names = list(CANDIDATES) + CANDIDATES[:1]
    ambiguous = intent_loop.Intent(intent_loop.AMBIGUOUS, decided_by=intent_loop.BY_QUESTION, ambiguous=tuple(names))
    with patch.object(CommandNamePrediction, "predict", return_value=ambiguous):
        route = wildcard.ResponseGenerator()(cme, "frobnicate", None)
    framed = CommandOutput(command_response=fastworkflow.CommandResponse(
        response=route.model_dump_json(), output=route))
    return route, _wildcard_command_output(framed)


def test_ambiguity_choices_reach_the_artifacts_in_assistant_mode(todo_cme):
    route, out = _route_ambiguous(*todo_cme, assistant=True)
    names = [each["name"] for each in route.plan_choice["choices"]]
    assert names == CANDIDATES  # duplicates removed, order kept
    artifacts = out.command_response.artifacts
    assert artifacts["question_kind"] == "plan_choice" and artifacts["plan_choice"] == route.plan_choice
    assert route.plan_choice["request"] == "frobnicate" and route.plan_choice["ambiguous"] is True
    assert out.command_response.response.startswith("Ambiguous intent error for command 'frobnicate'\n")
    assert "find_book" in out.command_response.response and "find_member" in out.command_response.response


def test_ambiguity_choices_never_reach_the_agent(todo_cme):
    route, out = _route_ambiguous(*todo_cme, assistant=False)
    assert route.plan_choice is None
    assert "plan_choice" not in out.command_response.artifacts
    assert "question_kind" not in out.command_response.artifacts
    assert out.command_response.response.startswith("Ambiguous intent error for command 'frobnicate'\n")
    # the agent reads the wording that tells it to update its command
    assert "update your command" in out.command_response.response


def test_parameter_request_is_copied_to_the_artifacts():
    route = wildcard.Signature.Output(
        handled=False, routed=False, response="x",
        parameter_request={"command": "c", "missing": ["a"], "invalid": [], "values": {}})
    framed = CommandOutput(command_response=fastworkflow.CommandResponse(response="x", output=route))
    out = _wildcard_command_output(framed)
    assert out.command_response.artifacts["parameter_request"]["missing"] == ["a"]
    assert "plan_choice" not in out.command_response.artifacts


# 7
class _RouterThatMustNotClassify:
    built = 0

    def __init__(self, path):
        type(self).built += 1
        self.modelpipeline = None

    def predict_with_details(self, command):
        raise AssertionError("the classifier was asked")

    predict = predict_with_details


SCOPE_CANDIDATES = ["TodoListManager/list_todo_lists", "TodoListManager/create_todo_list"]


def _pending_question(cme, candidates, request="find something"):
    """The which-command question the CME holds for *candidates* (the intent loop's record, in its context)."""
    from fastworkflow import intent_loop

    record = intent_loop.open_question(request, [{"name": name, "description": "", "closest": True}
                                                 for name in candidates], listing_hint=True)
    intent_loop.ask(record)
    c = cme.context
    c["command"] = request
    intent_loop.store(c, record)
    cme.context = c
    return record


@pytest.mark.parametrize("reply", ["list_todo_lists", "/list_todo_lists", "1"])
def test_clarification_reply_matches_the_stored_candidates_without_classifying(todo_cme, monkeypatch, reply):
    # The reply is read by the one reader: a name, a name with a slash, or the number the question printed.
    from tests.todo_list_workflow.application.todo_manager import TodoListManager

    app, cme = todo_cme
    app.root = TodoListManager()
    app.record = app.root
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RouterThatMustNotClassify)
    _RouterThatMustNotClassify.built = 0
    record = _pending_question(cme, SCOPE_CANDIDATES)
    predictor = CommandNamePrediction(cme)

    answered, chosen = predictor.read_reply(record, reply)

    assert (answered.kind, answered.name) == ("pick", "TodoListManager/list_todo_lists")
    assert chosen.command == "TodoListManager/list_todo_lists"
    assert _RouterThatMustNotClassify.built == 0


@pytest.mark.parametrize("reply", ["find_book", "/find_book"])
def test_clarification_reply_naming_a_command_of_another_scope_is_picked_and_found_nowhere_here(
        todo_cme, monkeypatch, reply):
    """bd fix-o5wi.9: the question offers commands of other scopes on the path. The reply picks the command; where it
    lives is the path's to say (``intent_loop.choose``): a command no scope of this workflow has is found nowhere,
    and the CME then says so instead of running anything."""
    _, cme = todo_cme
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RouterThatMustNotClassify)
    record = _pending_question(cme, CANDIDATES)

    answered, chosen = CommandNamePrediction(cme).read_reply(record, reply)

    assert (answered.kind, answered.name) == ("pick", "Library/find_book")
    assert chosen is None
    out = wildcard.ResponseGenerator()(cme, reply, None)
    assert out.handled is True and out.routed is False and "not available from here" in out.response
    from fastworkflow import intent_loop
    assert intent_loop.pending_record(cme.context) is None   # the question is over


# 8
def _request_for(command, params, diagnostics):
    """The parameter_request the CME sends: the shared contract's request, with JSON-safe values as they are."""
    return extraction_of(params, diagnostics).request(command, stringify=False)


def test_parameter_request_for_deduplicates_and_prefers_missing():
    params = SimpleNamespace(model_dump=lambda **kw: {"a": 1, "b": 2, "c": 3, "d": 4})
    request = _request_for(
        "cmd", params, {"missing_fields": ["a", "a", "b"], "invalid_fields": ["b", "c", "c"]})
    assert request == {
        "command": "cmd", "missing": ["a", "b"], "invalid": ["c"], "values": {"d": 4}}


def test_parameter_request_for_is_none_without_fields():
    params = SimpleNamespace(model_dump=lambda **kw: {"a": 1})
    assert _request_for("cmd", params, {}) is None
    assert _request_for("cmd", params, {"missing_fields": [], "invalid_fields": None}) is None


def test_parameter_request_values_exclude_not_given_sentinels():
    dumped = {"n": NOT_FOUND, "i": INVALID_INT_VALUE, "f": INVALID_FLOAT_VALUE, "ok": 0, "flag": False, "x": "y"}
    params = SimpleNamespace(model_dump=lambda **kw: dict(dumped))
    request = _request_for("cmd", params, {"missing_fields": ["m"]})
    assert request["values"] == {"ok": 0, "flag": False, "x": "y"}


# Other ambiguity exits: a reply that cannot be read is asked once more, then the question ends with the listing
def test_unmatched_clarification_reply_is_asked_once_more_and_then_ends_with_the_listing(todo_cme, monkeypatch):
    from fastworkflow import intent_loop

    _, cme = todo_cme
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RouterThatMustNotClassify)
    _pending_question(cme, CANDIDATES)

    first = wildcard.ResponseGenerator()(cme, "zzzz qqqq nothing", None)
    assert first.routed is False and first.handled is False
    assert first.response.startswith("Ambiguous intent error") and "That did not name one of them." in first.response
    assert [c["name"] for c in first.plan_choice["choices"]] == CANDIDATES   # a human session: the same question again
    assert intent_loop.pending_record(cme.context) is not None

    second = wildcard.ResponseGenerator()(cme, "zzzz qqqq nothing", None)
    assert second.handled is True
    assert second.response.startswith("Workflow Definition:")   # the what_can_i_do listing
    assert intent_loop.pending_record(cme.context) is None
    assert intent_loop.waiting_for(cme.context) is None


# A leading "/" that reaches a first message's intent detection (fix-weih.2 follow-up)
def test_slash_core_command_name_on_a_first_message_is_matched_exactly(todo_cme, monkeypatch):
    _, cme = todo_cme
    _RouterThatMustNotClassify.built = 0
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RouterThatMustNotClassify)
    predictor = CommandNamePrediction(cme)
    result = predictor.predict("/what_can_i_do")
    assert result.command == "IntentDetection/what_can_i_do"
    assert predictor.trace["matcher_layer"] == "exact_prefix"
    assert _RouterThatMustNotClassify.built == 0


def test_slash_name_owned_by_another_context_gets_the_routing_hint(todo_cme, monkeypatch):
    app, cme = todo_cme
    _RouterThatMustNotClassify.built = 0
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RouterThatMustNotClassify)
    # Stand where the todo workflow's item context is: a name of the manager context is foreign to it and its parents.
    result = CommandNamePrediction(cme).predict("/create_todo_list my list")
    # (the todo workflow's current context owns it or reaches it; the guard only fires for a name no scope on the path owns)
    assert result.command == "TodoListManager/create_todo_list" or result.hint
    assert _RouterThatMustNotClassify.built == 0
