"""Integration tests for the fuzzy pre-match tie guard in intent detection.

Covers fix-0xn. Before the guard, the CME's intent stage took
`best_matched_commands[0]` from a list that `find_best_matches` documents as
*all* candidates tying for the minimum distance. Commands sharing a prefix tied
at distance 0.0 (that scoring compared only the leading `len(input)` characters),
and `command_name_dict` is built by iterating a set — so the pick varied between
processes and bypassed both the classifier and the ambiguity prompt. Since bd
fix-ftdl.9 the name layer scores the whole request against each whole name
(`intent_clarification.misspelt`), and the shared stem still ties the two names.

Real components throughout: a real `RoutingDefinition` for a real test workflow
and the real `predict` path. Only the classifier result is injected, the same
seam `test_wildcard_escalation.py` uses, so no trained model artifacts are
needed.
"""

import logging
import os
import shutil
from io import StringIO

import pytest

import fastworkflow
from fastworkflow.train import model_pipeline_training
from fastworkflow.core.command_routing import RoutingDefinition
from fastworkflow.intent import intent_loop
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow.intent.intent_clarification import misspelt

# tests/todo_list_workflow exposes two commands sharing the 'add_child_todo'
# prefix, which is the ordinary shape of a generated command set.
TIED_PREFIX = "add_child_todo"
TIED_COMMANDS = {"add_child_todoitem", "add_child_todolist"}

import logging as _logging

intent_logger = _logging.getLogger("fastWorkflow")
FUZZY_THRESHOLD = 0.3  # the literal at the call site under test


@pytest.fixture(scope="module")
def todolist_command_names() -> dict[str, str]:
    """Real short-name -> fully-qualified map for the TodoList context."""
    workflow_path = os.path.join(os.path.dirname(__file__), "todo_list_workflow")
    routing_definition = RoutingDefinition.build(workflow_path)
    valid = set(routing_definition.get_command_names("TodoList"))
    assert valid, "todo_list_workflow exposes no commands in 'TodoList'"
    return {fqn.split("/")[-1]: fqn for fqn in valid}


@pytest.fixture
def todolist_workflows(tmp_path):
    """A real app workflow plus the real CME workflow bound to it."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"fuzzy-tie-{tmp_path.name}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    # The workflow stands on a TodoList below its manager, with a head for each of the two scopes.
    from fastworkflow.intent import intent_detector
    from tests.todo_list_workflow.application.todo_list import TodoList
    from tests.todo_list_workflow.application.todo_manager import TodoListManager

    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    app_workflow.root = manager
    app_workflow.record = todo_list
    for context in ("TodoList", "TodoListManager"):
        folder = intent_detector.head_folder(str(workflow_path), context)
        os.makedirs(os.path.join(folder, "tinymodel.pth"), exist_ok=True)
        for name in ("label_encoder.pkl", "threshold.json"):
            with open(os.path.join(folder, name), "w") as marker:
                marker.write("{}")
    return app_workflow, cme_workflow


def _router_returning(predictions, modelpipeline=...):
    class PredictingRouter:
        def __init__(self, model_artifact_path):
            # The real router's pipeline; None keeps the double free of a transformer.
            self.modelpipeline = self if modelpipeline is ... else modelpipeline

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            # Delegation direction mirrors the real CommandRouter, so the two
            # methods cannot drift apart — that drift is what broke this
            # double when production moved to predict_with_details.
            #
            # The details are deliberately empty rather than plausible-looking.
            # The real router DERIVES its labels from confidence; this seam
            # INJECTS labels, so there is no confidence here that is not
            # invented, and an invented number is something a later assertion
            # could pin as if it were true. Nothing reads these: the dict's
            # only destination is nlu_trace["classifier"], which reaches a
            # trace span that no-ops when no host is bound, as here.
            return predictions, {}

    return PredictingRouter


# ---------------------------------------------------------------------------
# The precondition: the tie is real, and it is reachable with real names
# ---------------------------------------------------------------------------


def test_prefix_sharing_commands_tie_at_the_same_distance(todolist_command_names):
    matches, distance = misspelt(TIED_PREFIX, todolist_command_names.keys())
    assert set(matches) == TIED_COMMANDS
    assert distance == 0.25 <= FUZZY_THRESHOLD     # four letters of sixteen, for each of the two


def test_tie_is_also_reachable_from_a_spaced_utterance(todolist_command_names):
    """A user typing words, not a command name, reaches the same tie: spaces and
    underscores are ignored."""
    utterance = "add child todo"
    matches, distance = misspelt(utterance, todolist_command_names.keys())
    assert set(matches) == TIED_COMMANDS
    assert distance == 0.25


# ---------------------------------------------------------------------------
# fix-0xn — INTENT_DETECTION defers to the classifier instead of guessing
# ---------------------------------------------------------------------------


def test_tie_defers_to_the_classifier_rather_than_picking_one(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """The whole point: a tied pre-match must not commit to one of the tied
    commands. Here the classifier is confident about a third command, and that
    is what must win."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        model_pipeline_training, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict(TIED_PREFIX)

    assert result.command == "TodoList/get_properties"
    assert result.outcome == "match"


# With the planner key an unsure local answer goes to the remote decision model (the chain, fix-o5wi.2);
# this asserts what happens without one.
@pytest.mark.usefixtures("no_planner_key")
def test_tie_reaches_the_ambiguity_prompt_when_the_classifier_is_unsure(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """Deferring restores the prompt that exists to ask which one was meant."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        model_pipeline_training, "CommandRouter",
        _router_returning(
            ["TodoList/add_child_todoitem", "TodoList/add_child_todolist"]))

    result = CommandNamePrediction(cme_workflow).predict(TIED_PREFIX)

    assert result.command is None
    assert result.outcome == "ambiguous"
    assert set(result.ambiguous) == {"TodoList/add_child_todoitem", "TodoList/add_child_todolist"}


def test_tie_is_logged(todolist_workflows, monkeypatch, setup_test_environment):
    """The pre-guard behaviour was invisible; a discarded tie must not be."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        model_pipeline_training, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    log_output = StringIO()
    log_handler = logging.StreamHandler(log_output)
    intent_logger.addHandler(log_handler)
    try:
        CommandNamePrediction(cme_workflow).predict(TIED_PREFIX)
    finally:
        intent_logger.removeHandler(log_handler)

    warning = log_output.getvalue()
    assert "Fuzzy pre-match tied across" in warning
    assert "add_child_todoitem" in warning
    assert "Deferring to the chain" in warning


# ---------------------------------------------------------------------------
# Positive controls: the guard must not disarm the pre-match
# ---------------------------------------------------------------------------


def test_unique_fuzzy_match_still_routes_without_the_classifier(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """An unambiguous near-miss must still short-circuit the classifier. The
    stub router would return a different command if it were consulted."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        model_pipeline_training, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict("add_child_todoitm")

    assert result.command == "TodoList/add_child_todoitem"
    assert result.outcome == "match" and result.decided_by == "fuzzy_name"


def test_exact_command_name_still_routes(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """One of the tied names, given in full, is an exact dict hit and never
    reaches the fuzzy branch at all."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        model_pipeline_training, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict("add_child_todolist")

    assert result.command == "TodoList/add_child_todolist" and result.decided_by == "exact_name"


def test_a_tie_in_a_reply_to_the_which_command_question_is_unclear_never_the_first(
    todolist_command_names
):
    """fix-0xn's open half, decided by fix-o1lg S7: the reply reader is ONE reader, and two names that match equally
    well are unclear on both paths (the CME used to take whichever came first)."""
    from fastworkflow.intent import intent_loop

    tied = sorted(f"TodoList/{name}" for name in TIED_COMMANDS)
    assert intent_loop.read(TIED_PREFIX, tied).kind == "unclear"
    # a misspelling that is closer to one of them still picks it
    assert intent_loop.read("add_child_todoitm", tied).name == "TodoList/add_child_todoitem"
