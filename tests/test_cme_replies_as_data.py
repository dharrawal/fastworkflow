"""The built-in replies that were text only are also data (fix-szfq): the ``what_can_i_do`` listing
(``capabilities``), what ``abort`` ended (``aborted``) and the commands ``you_misunderstood`` offers
(``command_choices``). The data is attached in ``command_executor``, not in the built-in command files, so the
internal workflow's training fingerprint (a hash of those files) does not move. Real workflows and the real
executor; the response texts are the commands' own, untouched. No model call, no training."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

import fastworkflow
from fastworkflow.train import model_pipeline_training
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow.agent.capability_answer import CapabilityListing
from fastworkflow.engine.command_executor import CommandExecutor
from fastworkflow.train import selective_training

TODO = os.path.abspath(os.path.join(os.path.dirname(__file__), "todo_list_workflow"))
CME = "command_metadata_extraction"


class _RefusingRouter:
    def __init__(self, model_artifact_path):
        self.modelpipeline = None

    def predict(self, command):
        raise AssertionError(f"the classifier was consulted for {command!r}")

    predict_with_details = predict


@pytest.fixture
def pair(setup_test_environment, monkeypatch):
    fastworkflow.init({})
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _RefusingRouter)
    app = fastworkflow.Workflow.create(workflow_folderpath=TODO, workflow_id_str=f"cme-data-{uuid4()}")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(CME),
        parent_workflow_id=app.id, workflow_context={"app_workflow": app})
    yield app, cme
    app.close()


def _wildcard(cme, text):
    return CommandExecutor.perform_action(cme, fastworkflow.Action(command_name="wildcard", command=text))


def test_what_can_i_do_carries_its_listing_and_the_text_is_the_commands_own(pair):
    app, cme = pair
    direct = CommandExecutor.perform_action(
        cme, fastworkflow.Action(command_name="IntentDetection/what_can_i_do", command="what can you do"))
    text = direct.command_response.response
    listing = direct.command_response.output
    assert isinstance(listing, CapabilityListing)
    assert direct.command_response.artifacts["capabilities"] == listing.model_dump(mode="json")

    # the text is what it always was: the workflow block, then the commands
    from fastworkflow.core.command_context_model import get_workflow_info
    from fastworkflow.core.command_metadata_api import CommandMetadataAPI
    expected = (CommandMetadataAPI.get_workflow_definition_display_text(get_workflow_info(app.folderpath))
                + "\n\n" + CommandMetadataAPI.get_command_display_text(
                    subject_workflow_path=app.folderpath, cme_workflow_path=cme.folderpath,
                    active_context_name=app._current_command_context_name, for_agents=False))
    assert text == expected

    # agreement: same commands, same order, same count, same names, parameters and descriptions as the text
    assert listing.commands, "the todo workflow lists commands"
    heads = [line[2:] for line in text.splitlines() if line.startswith("- ") and not line.startswith("- global")
             and line[2:].split(" ")[0] in {c["name"] for c in listing.commands}]
    names_in_text = [h for h in heads if h in {c["name"] + c["approval_marker"] for c in listing.commands}]
    assert names_in_text == [c["name"] + c["approval_marker"] for c in listing.commands]
    for command in listing.commands:
        assert command["context"] == listing.context and command["qualified_name"].endswith(command["name"])
        assert command["planning"] is False
        if command["description"]:
            assert command["description"] in text
        for parameter in command["parameters"]:
            assert (parameter["description"] or parameter["name"]) in text
    assert listing.workflow["name"] == os.path.basename(TODO)
    for context in listing.workflow["contexts"]:
        assert f"- {context}" in text
    assert text.startswith("Workflow Definition:") and f"({listing.context}):" in text


def test_what_can_i_do_through_the_router_carries_the_listing_too(pair):
    app, cme = pair
    routed = _wildcard(cme, "what_can_i_do")
    assert routed.command_response.artifacts["command_handled"] is True
    listing = routed.command_response.artifacts["capabilities"]
    assert listing["commands"] and routed.command_response.output.model_dump(mode="json") == listing
    assert [c["name"] for c in listing["commands"]] == sorted(c["name"] for c in listing["commands"])


def test_you_misunderstood_carries_the_commands_it_offers_in_the_order_it_lists_them(pair):
    app, cme = pair
    routed = _wildcard(cme, "you_misunderstood")
    choices = routed.command_response.artifacts["command_choices"]
    assert choices and choices == routed.command_response.output.valid_command_names == sorted(choices)
    listed = routed.command_response.response.split("enter its command name:\n", 1)[1].split("\n\n", 1)[0].split("\n")
    assert listed == choices


def test_abort_says_what_it_ended(pair):
    app, cme = pair
    first = _wildcard(cme, "you_misunderstood")      # leaves the CME in the misunderstanding stage with the command
    assert first.command_response.artifacts["command_choices"]
    aborted = _wildcard(cme, "abort")
    assert aborted.command_response.response == "command aborted\n"
    ended = aborted.command_response.artifacts["aborted"]
    assert ended["waiting_for"] == "intent" and ended["command"] == "you_misunderstood"
    assert aborted.command_response.output.command_name == "abort"


def test_the_edited_files_are_not_what_the_internal_workflows_training_fingerprint_hashes(pair):
    """Fingerprints are hashes of the CME command files and two trainer modules; the data is attached elsewhere."""
    import fastworkflow.agent.capability_answer as capability_answer
    import fastworkflow.engine.command_executor as command_executor

    sources = {fp.source_path for fp in selective_training.compute_command_fingerprints(
        fastworkflow.get_internal_workflow_path(CME)).values() if fp.source_path}
    assert sources and not sources & {command_executor.__file__, capability_answer.__file__}
    assert not {selective_training.TRAINER_SOURCE_MODULE, selective_training.GENERATOR_SOURCE_MODULE} & {
        "fastworkflow.engine.command_executor", "fastworkflow.agent.capability_answer"}
