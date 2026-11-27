"""The capability listing reads the context model through ``CommandContextModel`` and prints every docstring.

fix-ftdl.7.1 / fix-w6jy: the listing printed ``Book: [base]`` because two readers iterated the raw
``{"base": [...]}`` entry of ``context_inheritance_model.json``. fix-ftdl.7.18 (first half): a command's
docstring was left out of the listings when the command's name in words occurred in it.
Real components: the library example, never trained.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.agent.capability_answer import what_can_i_do_data
from fastworkflow.core.command_context_model import CommandContextModel, get_workflow_info
from fastworkflow.core.command_metadata_api import CommandMetadataAPI

LIBRARY = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "library"


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def library(tmp_path):
    target = tmp_path / "library"
    shutil.copytree(LIBRARY, target, ignore=shutil.ignore_patterns("___command_info", "___convo_info", "__pycache__"))
    return str(target)


def test_the_workflow_info_names_each_contexts_bases_not_the_key_of_its_entry(library):
    info = get_workflow_info(library)
    assert info["context_inheritance_model"] == {"Book": ["Record"], "Member": ["Record"]}
    assert CommandContextModel.declared_bases(library) == {"Book": ["Record"], "Member": ["Record"]}


def test_the_listing_text_prints_the_bases(library):
    text = CommandMetadataAPI.get_workflow_definition_display_text(get_workflow_info(library))
    assert "Book: [Record]" in text and "Member: [Record]" in text
    assert "[base]" not in text


def test_the_listing_data_names_the_bases_and_prints_every_docstring(library):
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    app = fastworkflow.Workflow.create(workflow_folderpath=library, workflow_id_str="listing-identity-app")
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=cme, parent_workflow_id=app.id, workflow_context={"app_workflow": app})
    try:
        listing = what_can_i_do_data(cme_workflow, for_agents=False)
    finally:
        app.close()
    assert listing.workflow["context_inheritance"] == {"Book": ["Record"], "Member": ["Record"]}
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=library, cme_workflow_path=cme, active_context_name=app._current_command_context_name)
    docs = {c["name"]: " ".join(str(c.get("doc_string") or "").split()) for c in meta["commands"]}
    assert any(docs.values())
    for entry in listing.commands:
        assert entry["description"] == docs[entry["name"]]


def test_a_docstring_that_contains_the_command_name_in_words_is_printed_in_the_text():
    text = CommandMetadataAPI._to_yaml_like(
        {"commands": [{"name": "find_books", "doc_string": "Find books by title or author."}]})
    assert "Find books by title or author." in text
