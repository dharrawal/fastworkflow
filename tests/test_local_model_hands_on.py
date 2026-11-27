"""bd fix-o5wi.9: the local model never passes a request up on its own, and the ancestor command is not lost.

A COPY of the library example is trained into the embedding layout from its utterance cache with no model call (a
cache miss raises), then read through the product router and the detection chain. The current record is a Book; its
parent Library has `list_branches`, and the global scope has `library_hours`.

* the Book head, asked for an ancestor's command, is not confident and hands the request on; the commands of the
  scopes on the lookup path that it ranks ahead of its own are among the candidates;
* the remote decision model, when scripted, still decides (it is asked exactly as before);
* with no remote model the ambiguity question offers the ancestor command, and choosing it runs it where it lives.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.intent import intent_detector, intent_loop
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import CommandRouter, train
from fastworkflow.core.workflow import running_command
from fastworkflow.train import determinism, utterance_cache
from fastworkflow.train import intent_model_layout as layout
from fastworkflow.train import workflow_model as wm
from tests.conftest import script_matcher

LIBRARY = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")
ENV_TEMPLATE = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "fastworkflow.env")

#: The branches command's seed utterances (the head was fitted on them, with weight 3).
BRANCH_REQUEST = "which branches does the library have"
HOURS_REQUEST = "what are your opening hours"


@pytest.fixture(scope="module")
def trained_library(tmp_path_factory):
    """The library example copied and trained in the embedding layout, offline."""
    from dotenv import dotenv_values

    if not os.path.isdir(os.path.join(LIBRARY, "___command_info", "utterance_cache")):
        pytest.skip("the library example has no utterance cache here; nothing can be rebuilt without a model call")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))
    import single_intent_model as runner

    root = tmp_path_factory.mktemp("hands_on")
    copy = runner.copy_workflow(LIBRARY, str(root))
    env = {key: value for key, value in dotenv_values(ENV_TEMPLATE).items() if value is not None}
    env["LLM_SYNDATA_GEN"] = wm.cached_generation_model(copy) or env.get("LLM_SYNDATA_GEN", "")
    env.pop(layout.SETTING, None)
    fastworkflow.init(env_vars=env)
    RoutingRegistry.clear_registry()
    try:
        cache = wm._ReadOnlyUtteranceCache(copy, mode=utterance_cache.MODE_REUSE)
        handle = fastworkflow.Workflow.create(copy, workflow_id_str="hands-on-train")
        previous = (determinism.get_provenance_recorder(), utterance_cache.get_utterance_cache())
        try:
            determinism.set_provenance_recorder(determinism.ProvenanceRecorder(copy))
            utterance_cache.set_utterance_cache(cache)
            with wm.no_model_calls():
                train(handle, intent_model_layout=layout.EMBEDDING)
        finally:
            determinism.set_provenance_recorder(previous[0])
            utterance_cache.set_utterance_cache(previous[1])
            handle.close()
    except Exception as error:  # noqa: BLE001 - no encoder or no cache: nothing to test with
        pytest.skip(f"the library copy could not be trained offline here ({type(error).__name__}: {error})")
    assert not cache.stats["miss"] and not cache.stats["stored"], cache.stats
    yield copy
    for key in [key for key in CommandRouter._instances_cache if key.startswith(copy)]:
        CommandRouter._instances_cache.pop(key, None)
    RoutingRegistry.clear_registry()


@pytest.fixture
def library_on_a_book(trained_library):
    fastworkflow.init({})
    app = fastworkflow.Workflow.create(workflow_folderpath=trained_library, workflow_id_str="hands-on-app")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    if app.root is None:
        from fastworkflow.examples.library.application.library import Library

        app.root = Library()
    library = app.root
    app.record = library.books["b1"]
    yield app, cme, library
    app.close() if hasattr(app, "close") else None


def _engine(app, context="Book", path=None):
    return intent_detector.engine_for(
        app.folderpath, context, scope_path=path or intent_detector.walk_path(app))


# ---------------------------------------------------------------- the head never passes up alone

@pytest.mark.usefixtures("no_planner_key")
def test_the_head_trained_here_is_the_new_rule_and_never_answers_not_here_for_a_command_elsewhere(
        library_on_a_book):
    app, _cme, _library = library_on_a_book
    scopes = layout_scopes(app)
    assert scopes["schema_version"] == 2 and scopes["encoder"]["model"] == "intfloat/e5-base-v2"
    assert scopes["threshold"] == 0.65 and "margin" not in scopes
    assert scopes["rows"]["description"] > 0 and scopes["rows"]["seeds_weighted"] > 0

    router = CommandRouter(intent_detector.head_folder(app.folderpath, "Book"))
    labels, details = router.predict_with_details(BRANCH_REQUEST)
    assert labels != [fastworkflow.intent.nlu_labels.WILDCARD_LABEL] and len(labels) > 1
    assert details["confident"] is False and details["p_out"] > details["p_in"]
    assert details["outside_labels"][0] == "Library/list_branches"


def layout_scopes(app) -> dict:
    import json

    folder = intent_detector.head_folder(app.folderpath, "Book")
    model_dir = layout.workflow_model_dir(folder)
    with open(os.path.join(model_dir, layout.SCOPES_FILENAME), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.usefixtures("no_planner_key")
def test_the_candidates_handed_on_include_the_ancestor_command_when_the_walk_can_reach_it(library_on_a_book):
    app, _cme, _library = library_on_a_book
    path = intent_detector.walk_path(app)
    assert "Library" in path

    answer = _engine(app).decide(BRANCH_REQUEST)

    assert answer.kind == intent_detector.AMBIGUOUS and not intent_detector.head_is_confident(answer)
    assert answer.labels[0] == "Library/list_branches"             # the head's best command, best first
    assert "Library/list_branches" in answer.details["handed_on"]
    assert set(answer.labels) - set(answer.details["handed_on"]) <= {
        label for label in answer.details["topk_labels"]}           # the rest are the scope's own candidates
    assert answer.scores["Library/list_branches"] == pytest.approx(answer.details["p_out"])

    # A path that does not reach the Library scope does not offer its command: nothing unreachable is asked.
    narrow = _engine(app, path=("Book",)).decide(BRANCH_REQUEST)
    assert "Library/list_branches" not in narrow.labels and "handed_on" not in narrow.details
    # A request the scope itself serves is answered outright and hands nothing on.
    own = _engine(app).decide("show me this book")
    assert own.kind == intent_detector.MATCH and own.labels == ["Book/show_book"] and own.confident


@pytest.mark.usefixtures("no_planner_key")
def test_detect_carries_the_ancestor_command_among_its_candidates(library_on_a_book):
    app, _cme, _library = library_on_a_book

    detection = intent_detector.detect(app, BRANCH_REQUEST)

    assert detection.outcome == intent_detector.AMBIGUOUS
    assert "Library/list_branches" in detection.candidates
    assert detection.scope.context_name == "Book"


# ---------------------------------------------------------------- the remote model still decides

def test_the_remote_decision_model_still_decides_when_the_head_hands_a_request_on(library_on_a_book, monkeypatch):
    app, _cme, _library = library_on_a_book
    asked = script_matcher(monkeypatch, {BRANCH_REQUEST: {"Library/list_branches": 0.9}})

    answer = _engine(app).decide(BRANCH_REQUEST)

    assert asked, "the remote model was not asked"
    assert answer.engine == intent_detector.ENGINE_MATCHER
    # The matcher chose a command of another scope on the path: the Book scope declines so the walk climbs there.
    assert answer.kind == intent_detector.NOT_MINE
    assert answer.details["classifier"]["handed_on"][0] == "Library/list_branches"


# ---------------------------------------------------------------- with no remote model: the question

@pytest.mark.usefixtures("no_planner_key")
def test_choosing_the_ancestor_command_at_the_question_runs_it_in_the_ancestor_scope(library_on_a_book):
    app, cme, library = library_on_a_book
    assert not intent_detector.matcher_available()

    asked = wildcard.ResponseGenerator()(cme, BRANCH_REQUEST, None)

    assert asked.routed is False and asked.handled is False
    assert "ambiguous" in asked.response.lower()
    assert "list_branches" in asked.response
    assert app._command_context_for_response_generation is not library                     # nothing ran: the question is open
    assert intent_loop.waiting_for(cme.context) == intent_loop.WAITING_INTENT

    with running_command():     # as the dispatcher runs the CME's command
        chosen = wildcard.ResponseGenerator()(cme, "list_branches", None)

    assert chosen.routed is True
    assert chosen.command_name == "Library/list_branches"
    assert chosen.ran_on_context_name == "Library"       # bound to the ancestor's record, as a climb would
    assert app._command_context_for_response_generation is library


@pytest.mark.usefixtures("no_planner_key")
def test_a_global_command_chosen_at_the_question_is_bound_to_the_root_record(library_on_a_book):
    """A pick is found where it lives: the global scope is asked first, so a global command the question offered (the
    head ranked it ahead of the Book's own) runs on the root record. Only the routing is read: nothing is executed."""
    from fastworkflow.intent import intent_loop

    app, cme, library = library_on_a_book
    assert intent_detector.global_first_applies(app)

    chosen = intent_loop.choose(app, "library_hours")
    assert (chosen.command, chosen.context) == ("library_hours", "*") and chosen.record is library
    # An own command chosen at the same question is not taken by the global scope.
    own = intent_loop.choose(app, "Book/show_book")
    assert own.context == "Book" and own.record is app._current_command_context


@pytest.mark.usefixtures("no_planner_key")
def test_a_pick_of_a_command_the_scope_does_not_have_runs_in_the_scope_that_has_it(library_on_a_book):
    """The Book scope does not run `list_branches`: the pick is found in the scope that has it (its record is bound);
    a command the scope has stays there; a framework command is answered by the current scope; a name no scope of
    the path has is found nowhere. The span records a pick of another scope (`chosen_command_elsewhere`)."""
    from fastworkflow.intent import intent_loop

    app, cme, library = library_on_a_book
    elsewhere = intent_loop.choose(app, "Library/list_branches")
    assert (elsewhere.context, elsewhere.record) == ("Library", library)
    assert intent_loop.choose(app, "Book/show_book").context == "Book"
    core = intent_loop.choose(app, "ErrorCorrection/abort")
    assert core.kind == intent_loop.KIND_CORE and core.context == app._current_command_context_name
    assert intent_loop.choose(app, "Nowhere/at_all") is None

    predictor = intent_detection.CommandNamePrediction(cme)
    record = intent_loop.open_question("branches?", intent_loop.choices_of(
        app, ["Library/list_branches", "Book/show_book"]))
    answered, chosen = predictor.read_reply(record, "list_branches")
    assert (answered.kind, answered.name) == ("pick", "Library/list_branches") and chosen.context == "Library"
