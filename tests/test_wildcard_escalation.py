"""Integration tests for the runtime half of the `wildcard` label work.

Covers two defects from ``docs/intent_training_improvements_spec.md``:

- **F6 / R7.1** — ``wildcard`` carried two meanings across two NLU stages. The
  parameter-extraction placeholders now live under their own label
  (``parameter_value``) so the escalation classifier is no longer trained on
  ``"france"``. The runtime must treat every reserved label as non-routable.
- **F7 / R7.4** — a ``wildcard`` inside a low-confidence top-k list was silently
  discarded in favour of a local prompt. The fixed conservative behavior now logs
  that suppression and offers only the local routable candidates.

These are integration tests against real components: the real CME
``CommandDirectory`` (which is how the trainer reaches the wildcard command's
utterances), a real ``RoutingDefinition`` for a real test workflow, and the real
``CommandNamePrediction.predict`` path. The predict regression injects only the
classifier result so it does not need trained model artifacts.
"""

import logging
import os
import shutil
from io import StringIO

import pytest

import fastworkflow
from fastworkflow import model_pipeline_training
from fastworkflow.core.command_directory import CommandDirectory
from fastworkflow.intent.nlu_labels import (
    ESCALATION_LABELS,
    NON_ROUTABLE_LABELS,
    PARAMETER_VALUE_LABEL,
    PARAMETER_VALUE_PLACEHOLDERS,
    WILDCARD_LABEL,
    is_escalation,
    is_non_routable,
)
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow._workflows.command_metadata_extraction import intent_detection


# Spec F7 failure: context ReviewTicket, utterance
# "approve everything from this app at once".
ESCALATION_FAILURE_TOPK = [
    "wildcard",
    "ReviewTicket/certify_approve",
    "ReviewTicket/show_review_item",
]


@pytest.fixture(scope="module")
def cme_command_directory() -> CommandDirectory:
    """The real CME command directory, rebuilt from source (not the snapshot)."""
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    cmd_dir = CommandDirectory.load(cme_path)
    cmd_dir.ensure_command_hydrated("wildcard")
    return cmd_dir


@pytest.fixture(scope="module")
def wildcard_generated_utterances(cme_command_directory) -> list[str]:
    """Reach the wildcard command's utterances exactly the way the trainer does.

    ``model_pipeline_training._get_utterances`` resolves
    ``UtteranceMetadata.get_generated_utterances_func`` and calls it, so this is
    the same seam that builds the ``wildcard`` training class.
    """
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    utterance_metadata = cme_command_directory.get_utterance_metadata("wildcard")
    assert utterance_metadata is not None, "CME has no utterance metadata for 'wildcard'"

    generate = utterance_metadata.get_generated_utterances_func(cme_path)
    assert generate is not None, "wildcard's generate_utterances could not be resolved"

    # Signature.generate_utterances ignores its workflow argument; the trainer
    # passes a real Workflow, but constructing one is not needed to exercise it.
    return generate(None, "wildcard")


# ---------------------------------------------------------------------------
# R7.1 — the label split
# ---------------------------------------------------------------------------


def test_parameter_value_placeholders_are_intact():
    """All seven bare-value literals survived the move out of wildcard.py."""
    assert PARAMETER_VALUE_PLACEHOLDERS == [
        "3",
        "france",
        "16.7,.002",
        "John Doe, 56, 281-995-6423",
        "/path/to/my/object",
        "id=3636",
        "25.73 and Howard St",
    ]


def test_reserved_labels_are_distinct_and_classified():
    assert WILDCARD_LABEL != PARAMETER_VALUE_LABEL
    assert NON_ROUTABLE_LABELS == {WILDCARD_LABEL, PARAMETER_VALUE_LABEL}
    # Only the escalation signal escalates; a bare value says nothing about
    # whether an ancestor context can serve the utterance.
    assert ESCALATION_LABELS == {WILDCARD_LABEL}
    assert is_non_routable(WILDCARD_LABEL)
    assert is_non_routable(PARAMETER_VALUE_LABEL)
    assert is_escalation(WILDCARD_LABEL)
    assert not is_escalation(PARAMETER_VALUE_LABEL)
    assert not is_non_routable("ReviewTicket/certify_approve")


def test_generate_utterances_excludes_parameter_value_placeholders(
    wildcard_generated_utterances,
):
    """The escalation class must no longer be taught that "france" means escalate."""
    assert wildcard_generated_utterances == ["wildcard"]
    leaked = [
        placeholder
        for placeholder in PARAMETER_VALUE_PLACEHOLDERS
        if placeholder in wildcard_generated_utterances
    ]
    assert not leaked, f"parameter-extraction literals leaked into the wildcard class: {leaked}"


def test_wildcard_plain_utterances_still_expose_the_placeholders(cme_command_directory):
    """Consumers that read `plain_utterances` keep seeing the seven literals.

    ``CommandDirectory.ensure_command_hydrated`` treats an empty
    ``plain_utterances`` as "not hydrated yet", so emptying it would make
    hydration re-run on every call.
    """
    utterance_metadata = cme_command_directory.get_utterance_metadata("wildcard")
    assert utterance_metadata.plain_utterances == PARAMETER_VALUE_PLACEHOLDERS


# ---------------------------------------------------------------------------
# R7.1 — runtime safety for the new label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(NON_ROUTABLE_LABELS))
def test_non_routable_label_is_never_a_candidate(label):
    """A reserved label is not looked up as a command and never offered as one (the intent loop's filter)."""
    from fastworkflow.intent import intent_loop

    assert intent_loop._usable([label, "Todo/add_item"], set()) == ["Todo/add_item"]
    assert intent_loop._usable([label], set()) == []


# ---------------------------------------------------------------------------
# R7.1 — no reserved label may ever be offered to a user
# ---------------------------------------------------------------------------


def _question_for(labels, *, for_agent):
    from fastworkflow.intent import intent_clarification, intent_loop

    choices = [{"name": name, "description": "", "closest": True} for name in intent_loop._usable(labels, set())]
    return intent_clarification.plan_choice_question("approve all", choices, ambiguous=True, for_agent=for_agent)


def test_ambiguity_message_hides_every_non_routable_label():
    predictions = [
        WILDCARD_LABEL,
        PARAMETER_VALUE_LABEL,
        "ReviewTicket/certify_approve",
        "ReviewTicket/show_review_item",
    ]
    message = _question_for(predictions, for_agent=False)
    for label in NON_ROUTABLE_LABELS:
        assert label not in message, f"non-routable label '{label}' offered as a command"
    assert "certify_approve" in message
    assert "show_review_item" in message


@pytest.mark.parametrize("run_as_agent", [True, False])
def test_ambiguity_message_lists_only_local_candidates(run_as_agent):
    message = _question_for(ESCALATION_FAILURE_TOPK, for_agent=run_as_agent)
    listed = [line.split(". ", 1)[1].split("/")[-1] for line in message.splitlines() if line[:2] in {"1.", "2.", "3."}]
    assert listed == ["certify_approve", "show_review_item"]
    assert WILDCARD_LABEL not in message


# ---------------------------------------------------------------------------
# R7.4 — fixed top-k wildcard behaviour
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R7.4 — production-path suppression diagnostics
# ---------------------------------------------------------------------------


def _fake_heads(app_workflow, *contexts):
    """A trained-looking head for each context: the router double answers for it."""
    from fastworkflow.intent import intent_detector

    for context in contexts:
        folder = intent_detector.head_folder(app_workflow.folderpath, context)
        os.makedirs(os.path.join(folder, "tinymodel.pth"), exist_ok=True)
        for name in ("label_encoder.pkl", "threshold.json"):
            with open(os.path.join(folder, name), "w") as marker:
                marker.write("{}")


def _todo_pair(tmp_path):
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"),
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"wildcard-predict-{tmp_path.name}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    return app_workflow, cme_workflow


class _PredictingRouter:
    labels: list = []

    def __init__(self, model_artifact_path):
        self.modelpipeline = self

    def predict(self, command):
        return self.predict_with_details(command)[0]

    def predict_with_details(self, command):
        # The details stay empty because this seam injects labels rather than deriving them from a confidence the
        # double does not have.
        return list(type(self).labels), {}


# With the planner key an unsure local answer goes to the remote decision model (the chain, fix-o5wi.2);
# this asserts what happens without one.
@pytest.mark.usefixtures("no_planner_key")
def test_predict_mixed_topk_logs_suppressed_escalation_and_prompts_locally(
    tmp_path, monkeypatch, setup_test_environment
):
    """Exercise the production ambiguity branch for the original F7 behaviour.

    A mixed top-k remains an ambiguity rather than silently changing context.
    The diagnostic adds a log line and nothing else: the pre-change path took
    the ambiguity branch and filtered the literal ``wildcard`` from the prompt.
    This fixed conservative behaviour is product logic, not an environment or
    per-workflow policy. A top-k ``parameter_value`` likewise carries no
    evidence that an ancestor can help.
    """
    app_workflow, cme_workflow = _todo_pair(tmp_path)
    _fake_heads(app_workflow, app_workflow._current_command_context_name, "*")

    class Router(_PredictingRouter):
        labels = [WILDCARD_LABEL, "TodoList/add_child_todoitem", "TodoList/get_properties"]

    monkeypatch.setattr(model_pipeline_training, "CommandRouter", Router)
    prediction = CommandNamePrediction(cme_workflow)
    log_output = StringIO()
    log_handler = logging.StreamHandler(log_output)
    intent_detection.logger.addHandler(log_handler)
    try:
        result = prediction.predict("approve everything from this app at once")
    finally:
        intent_detection.logger.removeHandler(log_handler)

    assert result.command is None
    assert result.outcome == "ambiguous"
    assert list(result.ambiguous) == ["TodoList/add_child_todoitem", "TodoList/get_properties"]
    warning = log_output.getvalue()
    assert "Top-k escalation signal discarded" in warning
    assert "suppressed=['wildcard']" in warning
    assert prediction.trace["escalation_labels_discarded"] == [WILDCARD_LABEL]


@pytest.mark.usefixtures("no_planner_key")
@pytest.mark.parametrize("label", sorted(NON_ROUTABLE_LABELS))
def test_a_reserved_label_from_the_head_resolves_to_no_command(tmp_path, monkeypatch, setup_test_environment, label):
    """The head's reserved answer (the escalation signal, a bare value) is never looked up as a command: the decision
    names no command, and a bare value is reported as one."""
    app_workflow, cme_workflow = _todo_pair(tmp_path)
    _fake_heads(app_workflow, app_workflow._current_command_context_name, "*")

    class Router(_PredictingRouter):
        labels = [label]

    monkeypatch.setattr(model_pipeline_training, "CommandRouter", Router)
    intent = CommandNamePrediction(cme_workflow).predict("france")
    assert intent.command is None and intent.outcome == "none"
    assert intent.bare_value is (label == PARAMETER_VALUE_LABEL)


def test_escalation_signals_are_reported_for_diagnostics():
    assert CommandNamePrediction.escalation_signals_in(ESCALATION_FAILURE_TOPK) == [
        WILDCARD_LABEL
    ]
    assert not CommandNamePrediction.escalation_signals_in(
        ["ReviewTicket/certify_approve"]
    )
