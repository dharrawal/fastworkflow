"""The detector contract (``fastworkflow/intent/intent_detector.py``): the scope decision, the candidates
of a scope, the engine of a scope, and the walk over scopes.

Integration tests against real components: a COPY of the bundled library example (five contexts, an
inheritance edge, nested records, planning commands) with its real routing definition, real
application records and the real ``CommandNamePrediction``. No head is trained and none of the
example's own ``___command_info`` is read or written: where a head must answer, the test supplies a
router that returns the labels it was told to (the seam ``tests/test_wildcard_escalation.py`` and
``tests/test_intent_detection_foreign_context.py`` use), and the walk's per-scope step is a plain
function that returns real ``CommandNamePrediction.Output`` objects.
"""

import os
import shutil

import pytest

import dataclasses
from typing import Optional
import fastworkflow
from fastworkflow.intent import intent_detector
from fastworkflow import model_pipeline_training, run_listing
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import CommandNamePrediction
from fastworkflow.intent.intent_detector import (
    Candidate,
    ClassifierEngine,
    Engine,
    Scope,
    ScopeAnswer,
    ScopeWalk,
)
from fastworkflow.intent.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixture
    initialized_fastworkflow,
)

LIBRARY = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")


@dataclasses.dataclass
class Output:
    """What a per-scope step returns to ``ScopeWalk``: the shape the walk reads."""

    command_name: Optional[str] = None
    error_msg: Optional[str] = None
    is_cme_command: bool = False
    routing_hint: Optional[str] = None
    answered_bare_value: bool = False


# ---------------------------------------------------------------------------
# Doubles and fixtures
# ---------------------------------------------------------------------------

def _router(labels, details=None):
    """A router that answers every utterance with *labels* and *details* (empty by default, as the
    label-injecting doubles of the neighbouring test files are)."""
    class LabellingRouter:
        def __init__(self, model_artifact_path=None):
            self.model_artifact_path = model_artifact_path
            self.modelpipeline = None

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            return labels, ({} if details is None else details)

    return LabellingRouter


def _router_by_context(labels_by_context: dict, asked: list):
    """A router class whose answer depends on the head folder it was built for."""
    class ByContextRouter:
        def __init__(self, model_artifact_path):
            self.context = os.path.basename(model_artifact_path)
            self.modelpipeline = None

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            asked.append(self.context)
            return labels_by_context[self.context], {}

    return ByContextRouter


@pytest.fixture
def library_copy(tmp_path, initialized_fastworkflow):  # noqa: F811 - fixture
    """The library example copied without its trained heads or session state."""
    folder = tmp_path / "library_wf"
    shutil.copytree(
        LIBRARY, folder,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"),
    )
    return str(folder)


@pytest.fixture
def on_a_book(library_copy):
    """The copy as a workflow whose root is a real Library and whose current record is a real Book."""
    app = fastworkflow.Workflow.create(
        workflow_folderpath=library_copy, workflow_id_str=f"detector-{os.path.basename(os.path.dirname(library_copy))}")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    if app.root is None:
        from fastworkflow.examples.library.application.library import Library

        app.root = Library()
    library = app.root
    book = library.books["b1"]
    app.record = book
    return app, cme, library, book


def _fake_head(folder: str) -> None:
    """The three markers that make a folder a trained head, with nothing a model could load."""
    os.makedirs(os.path.join(folder, "tinymodel.pth"))
    for name in ("label_encoder.pkl", "threshold.json"):
        with open(os.path.join(folder, name), "w") as marker:
            marker.write("{}")


def _heads(app, *contexts: str) -> None:
    """A trained-looking head for each context (the router double answers for it)."""
    for context in contexts:
        folder = intent_detector.head_folder(app.folderpath, context)
        if not intent_detector.has_head(folder):
            _fake_head(folder)


def _scripted_step(answers: dict):
    """A per-scope step that returns the scripted answer of each context (a decline when unscripted)."""
    def step(context_name: str):
        return answers.get(context_name, Output(command_name=None))
    return step


# ---------------------------------------------------------------------------
# The scope decision: ClassifierEngine
# ---------------------------------------------------------------------------

DETAILS = {
    "model_tier": "tiny",
    "confidence": 0.93,
    "ambiguous_threshold": 0.5,
    "confident": True,
    "top_label": "Book/show_book",
    "topk_labels": ["Book/show_book", "Book/list_loans", "wildcard"],
    "topk_scores": [0.93, 0.05, 0.02],
}


def test_the_classifier_engine_satisfies_the_engine_contract():
    engine = ClassifierEngine(_router(["Book/show_book"])())
    assert isinstance(engine, Engine)
    assert engine.name == "classifier"


def test_one_command_label_is_a_match_with_the_routers_details_unchanged():
    router = _router(["Book/show_book"], DETAILS)()
    answer = ClassifierEngine(router).decide("show me this book")
    assert answer.kind == "match"
    assert answer.labels == ["Book/show_book"]
    assert answer.confident is True
    assert answer.engine == "classifier" and answer.fallback is False
    assert answer.details is DETAILS
    assert answer.scores == {"Book/show_book": 0.93, "Book/list_loans": 0.05, "wildcard": 0.02}


def test_one_label_the_router_was_not_confident_on_is_a_match_that_is_not_confident():
    """A head with one label in its top-k returns it below the threshold too: still the one command."""
    details = {**DETAILS, "confident": False, "topk_labels": ["Book/show_book"], "topk_scores": [0.31]}
    answer = ClassifierEngine(_router(["Book/show_book"], details)()).decide("hmm")
    assert answer.kind == "match" and answer.confident is False


def test_the_escalation_label_alone_is_not_mine():
    answer = ClassifierEngine(_router([WILDCARD_LABEL], {**DETAILS, "top_label": "wildcard"})()).decide("x")
    assert answer.kind == "not_mine"
    assert answer.labels == [WILDCARD_LABEL]   # kept: the caller resolves it to no command, as before
    assert answer.confident is False


def test_the_parameter_value_label_alone_is_a_bare_value():
    answer = ClassifierEngine(_router([PARAMETER_VALUE_LABEL], DETAILS)()).decide("france")
    assert answer.kind == "bare_value"
    assert answer.labels == [PARAMETER_VALUE_LABEL]
    assert answer.confident is False


def test_several_labels_are_an_ambiguity_that_keeps_every_label_in_order():
    labels = [WILDCARD_LABEL, "Book/show_book", PARAMETER_VALUE_LABEL, "Book/list_loans"]
    details = {**DETAILS, "confident": False, "topk_labels": labels, "topk_scores": [0.4, 0.3, 0.2, 0.1]}
    answer = ClassifierEngine(_router(labels, details)()).decide("something about this book")
    assert answer.kind == "ambiguous"
    assert answer.labels == labels
    assert answer.confident is False
    assert list(answer.scores.items()) == list(zip(labels, [0.4, 0.3, 0.2, 0.1]))


@pytest.mark.parametrize("labels, kind, confident", [
    (["Book/show_book"], "match", True),
    ([WILDCARD_LABEL], "not_mine", False),
    ([PARAMETER_VALUE_LABEL], "bare_value", False),
    ([WILDCARD_LABEL, "Book/show_book"], "ambiguous", False),
    ([], "ambiguous", False),
])
def test_a_router_that_reports_no_details_still_answers(labels, kind, confident):
    answer = ClassifierEngine(_router(labels)()).decide("anything")
    assert (answer.kind, answer.labels, answer.confident) == (kind, labels, confident)
    assert answer.scores == {} and answer.details == {}


def test_a_fallback_engine_says_so_on_its_answer():
    answer = ClassifierEngine(_router(["Book/show_book"])(), fallback=True).decide("x")
    assert answer.fallback is True


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def test_candidates_of_a_context_are_its_execution_core_and_planning_commands(library_copy):
    offered = intent_detector.candidates(library_copy, "Book")
    assert all(isinstance(candidate, Candidate) for candidate in offered)
    by_name = {candidate.name: candidate for candidate in offered}
    kinds = {name: candidate.kind for name, candidate in by_name.items()}

    # Its own commands, the ones it inherits from its base (Record), and the framework's.
    assert kinds["Book/show_book"] == "execution"
    assert kinds["Book/list_loans"] == "execution"
    assert kinds["Record/add_note"] == "execution"
    assert kinds["IntentDetection/go_up"] == "core"
    assert kinds["IntentDetection/what_can_i_do"] == "core"
    # Its planning commands, and not another context's.
    assert kinds["Book/loan_report"] == "planning"
    assert kinds["Book/shelf_report"] == "planning"
    assert "Library/overdue_summary" not in kinds
    # The reserved label is not a command a scope offers.
    assert "wildcard" not in kinds

    # Examples are the seed utterances; descriptions the Signature docstrings.
    crd = fastworkflow.RoutingRegistry.get_definition(library_copy)
    assert by_name["Book/list_loans"].examples == tuple(
        crd.command_directory.get_utterance_metadata("Book/list_loans").plain_utterances)
    assert "who has this book" in by_name["Book/loan_report"].examples
    assert by_name["Book/loan_report"].description.startswith("Say who has this book on loan")

    # Execution and core commands first, in routing order; planning commands after.
    names = [candidate.name for candidate in offered]
    execution = crd.get_command_names("Book")
    assert names[:len(execution)] == execution
    assert names[len(execution):] == crd.get_planning_command_names("Book")


def test_candidates_of_the_global_scope_and_of_a_context_with_planning_commands(library_copy):
    global_kinds = {c.name: c.kind for c in intent_detector.candidates(library_copy, "*")}
    assert global_kinds["library_hours"] == "execution"
    assert "planning" not in global_kinds.values()

    library_kinds = {c.name: c.kind for c in intent_detector.candidates(library_copy, "Library")}
    assert library_kinds["Library/find_book"] == "execution"
    assert library_kinds["Library/overdue_summary"] == "planning"
    assert "Book/loan_report" not in library_kinds


def test_candidates_of_an_unknown_context_is_an_error(library_copy):
    with pytest.raises(ValueError):
        intent_detector.candidates(library_copy, "NoSuchContext")


def test_reading_candidates_leaves_the_bundled_example_alone(library_copy):
    """The tests above read a copy. The bundled example's own trained folder is never the target."""
    assert not library_copy.startswith(LIBRARY)


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("no_planner_key")   # with the key, the model matcher stands in (fix-yu3l.3)
def test_a_scope_with_no_trained_head_has_no_engine(library_copy):
    built = []
    assert intent_detector.engine_for(library_copy, "Book", router_factory=built.append) is None
    assert built == []   # nothing was constructed for a scope that cannot answer


@pytest.mark.usefixtures("no_planner_key")   # with the key, the head is the first link of a chain (fix-o5wi.2)
def test_a_scope_with_a_trained_head_gets_the_classifier_built_from_its_folder(library_copy):
    head = os.path.join(library_copy, "___command_info", "Book")
    _fake_head(head)
    built = []

    def factory(folder):
        built.append(folder)
        return _router(["Book/show_book"])(folder)

    engine = intent_detector.engine_for(library_copy, "Book", router_factory=factory)
    assert isinstance(engine, ClassifierEngine)
    assert engine.fallback is False
    assert built == [head]
    assert engine.decide("show me this book").kind == "match"


@pytest.mark.usefixtures("no_planner_key")   # with the key, the model matcher stands in (fix-yu3l.3)
def test_the_global_scope_reads_the_global_folder_and_an_incomplete_head_is_no_head(library_copy):
    head = os.path.join(library_copy, "___command_info", "global")
    os.makedirs(head)
    with open(os.path.join(head, "threshold.json"), "w") as marker:
        marker.write("{}")
    assert intent_detector.engine_for(library_copy, "*", router_factory=_router([])) is None

    shutil.rmtree(head)
    _fake_head(head)
    engine = intent_detector.engine_for(library_copy, "*", router_factory=_router(["library_hours"]))
    assert engine is not None and engine.router.model_artifact_path == head


def test_the_default_router_is_the_trainers_as_named_when_the_engine_is_built(library_copy, monkeypatch):
    """A head placed under the trainer's ``CommandRouter`` name is the one the engine asks."""
    from fastworkflow import model_pipeline_training

    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router(["Book/list_loans"]))
    engine = intent_detector.engine_for(library_copy, "Book")
    assert engine.decide("loans").labels == ["Book/list_loans"]


@pytest.mark.usefixtures("no_planner_key")   # with the key, the model matcher stands in (fix-yu3l.3)
def test_heads_of_another_folder_are_read_with_command_info_root(library_copy, tmp_path):
    other_root = tmp_path / "other_heads"
    _fake_head(str(other_root / "Member"))
    assert intent_detector.engine_for(library_copy, "Member", router_factory=_router([])) is None
    engine = intent_detector.engine_for(
        library_copy, "Member", router_factory=_router([]), command_info_root=str(other_root))
    assert engine is not None


# ---------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------

def test_scopes_for_a_nested_record_are_its_context_then_each_enclosing_record(on_a_book):
    app, _, library, book = on_a_book
    assert intent_detector.scopes_for(app) == [Scope("Book", book), Scope("Library", library)]
    # Reading the path moves nothing.
    assert app._current_command_context is book


def test_scopes_for_the_root_record_is_the_root_alone(on_a_book):
    app, _, library, _ = on_a_book
    app.record = library
    assert intent_detector.scopes_for(app) == [Scope("Library", library)]


def test_scopes_for_no_record_is_the_global_scope(on_a_book):
    app, _, _, _ = on_a_book
    app.record = None
    assert intent_detector.scopes_for(app) == [Scope("*", None)]


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_a_confident_global_command_wins_and_runs_on_the_root(on_a_book):
    app, _, library, book = on_a_book
    result = intent_detector.walk(
        app, _scripted_step({"*": Output(command_name="library_hours"),
                             "Book": Output(command_name="Book/show_book")}),
        global_first=True, bind=True)
    assert result.asked == ("*",)
    assert result.answer.command_name == "library_hours"
    assert result.scope == Scope("*", library)
    assert app._command_context_for_response_generation is library               # the command runs on the root record
    assert app._current_command_context is book   # and the position did not move


@pytest.mark.parametrize("global_answer", [
    Output(command_name=None),                                           # a decline
    Output(command_name=None, error_msg="Ambiguous intent"),              # unsure
    Output(command_name=None, answered_bare_value=True),                  # a bare value
    Output(command_name="IntentDetection/go_up", is_cme_command=True),    # a framework command
])
def test_anything_else_from_the_global_scope_is_discarded(on_a_book, global_answer):
    app, _, _, book = on_a_book
    result = intent_detector.walk(
        app, _scripted_step({"*": global_answer, "Book": Output(command_name="Book/show_book")}),
        global_first=True, bind=True)
    assert result.asked == ("*", "Book")
    assert result.answer.command_name == "Book/show_book"
    assert result.scope == Scope("Book", book)
    assert app.record is book


def test_the_nearest_scope_wins_and_no_enclosing_scope_is_asked(on_a_book):
    app, _, _, book = on_a_book
    result = intent_detector.walk(
        app, _scripted_step({"Book": Output(command_name="Book/list_loans"),
                             "Library": Output(command_name="Library/find_book")}),
        global_first=False)
    assert result.asked == ("Book",)
    assert result.scope == Scope("Book", book)


def test_a_decline_moves_to_the_enclosing_scope_which_is_bound_when_asked(on_a_book):
    app, _, library, book = on_a_book
    bound_when_asked = {}

    def step(context_name):
        bound_when_asked[context_name] = app._command_context_for_response_generation
        return (Output(command_name="Library/find_book") if context_name == "Library"
                else Output(command_name=None))

    result = intent_detector.walk(app, step, global_first=False, bind=True)
    assert result.asked == ("Book", "Library")
    assert result.answer.command_name == "Library/find_book"
    assert result.scope == Scope("Library", library)
    # Each enclosing scope is the workflow's record while it is asked, and stays so when it answers.
    assert bound_when_asked["Library"] is library
    assert app._command_context_for_response_generation is library
    assert app._current_command_context is book


def test_without_binding_the_walk_only_reads_the_workflow(on_a_book):
    app, _, library, _ = on_a_book
    before = app.record
    result = intent_detector.walk(
        app, _scripted_step({"Library": Output(command_name="Library/find_book")}), global_first=False)
    assert result.scope == Scope("Library", library)
    assert app.record is before


def test_a_bare_value_stops_the_walk_where_it_is_given(on_a_book):
    app, _, _, book = on_a_book
    result = intent_detector.walk(
        app, _scripted_step({"Book": Output(command_name=None, answered_bare_value=True),
                             "Library": Output(command_name="Library/find_book")}),
        global_first=False, bind=True)
    assert result.asked == ("Book",)
    assert result.answer.command_name is None and result.answer.answered_bare_value
    assert result.scope == Scope("Book", book)


def test_past_the_root_record_the_result_is_the_last_decline(on_a_book):
    app, _, library, _ = on_a_book
    result = intent_detector.walk(app, _scripted_step({}), global_first=False, bind=True)
    assert result.asked == ("Book", "Library")   # the root record is the last scope; '*' is not climbed to
    assert result.answer.command_name is None
    assert result.scope == Scope("Library", library)
    assert result.routing_hint is None


def test_the_first_routing_hint_of_the_walk_is_the_one_carried(on_a_book):
    app, _, _, _ = on_a_book
    both = intent_detector.walk(
        app, _scripted_step({"Book": Output(command_name=None, routing_hint="from the book"),
                             "Library": Output(command_name=None, routing_hint="from the library")}),
        global_first=False)
    assert both.routing_hint == "from the book"

    later = intent_detector.walk(
        app, _scripted_step({"Library": Output(command_name=None, routing_hint="from the library")}),
        global_first=False)
    assert later.routing_hint == "from the library"


@pytest.mark.parametrize("first_answer", [
    Output(command_name=None, error_msg="Ambiguous intent"),
    Output(command_name="IntentDetection/go_up", is_cme_command=True),
])
def test_an_ambiguity_or_a_framework_command_ends_the_walk_and_binds_nothing(on_a_book, first_answer):
    app, _, _, _ = on_a_book
    before = app.record
    result = intent_detector.walk(app, _scripted_step({"Book": first_answer}), global_first=False, bind=True)
    assert result.asked == ("Book",)
    assert result.answer is first_answer
    assert app.record is before


def test_the_two_halves_are_the_walk_the_cme_stages_run(on_a_book):
    """``first`` then ``climb``, with the caller's own work between them, as the wildcard command does."""
    app, _, library, book = on_a_book
    scope_walk = ScopeWalk(app, _scripted_step({"Library": Output(command_name="Library/open_member")}))
    answer, scope = scope_walk.first(Scope("Book", book), global_first=False)
    assert answer.command_name is None and scope == Scope("Book", book)
    answer, scope, hint = scope_walk.climb(answer, scope)
    assert (answer.command_name, scope, hint) == ("Library/open_member", Scope("Library", library), None)
    assert scope_walk.asked == ["Book", "Library"]

    # A command from the first half is bound to its scope by the climb and asks nothing more.
    scope_walk = ScopeWalk(app, _scripted_step({"Book": Output(command_name="Book/show_book")}))
    answer, scope = scope_walk.first(Scope("Book", book), global_first=False)
    answer, scope, hint = scope_walk.climb(answer, scope)
    assert scope_walk.asked == ["Book"] and app.record is book


def test_the_global_scope_is_not_asked_first_without_a_trained_global_head(on_a_book):
    app, _, _, _ = on_a_book
    assert intent_detector.global_head_trained(app) is False
    assert intent_detector.global_first_applies(app) is False
    result = intent_detector.walk(app, _scripted_step({"*": Output(command_name="library_hours")}))
    assert "*" not in result.asked

    # The rule reads the trainer's threshold file for '*', and applies only away from the global scope.
    head = os.path.join(app.folderpath, "___command_info", "global")
    os.makedirs(head)
    with open(os.path.join(head, "threshold.json"), "w") as marker:
        marker.write("{}")
    assert intent_detector.global_first_applies(app) is True
    assert intent_detector.walk(
        app, _scripted_step({"*": Output(command_name="library_hours")})).asked == ("*",)
    app.record = None
    assert intent_detector.global_first_applies(app) is False


# ---------------------------------------------------------------------------
# The real per-scope step: CommandNamePrediction over the engine
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("no_planner_key")
def test_the_cme_decision_walks_the_heads_nearest_first_through_the_engine(on_a_book, monkeypatch):
    app, cme, library, _ = on_a_book
    _heads(app, "Book", "Library")
    asked = []
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router_by_context(
        {"Book": [WILDCARD_LABEL], "Library": ["Library/list_branches"]}, asked))
    predictor = CommandNamePrediction(cme)

    intent = predictor.predict("which branches does the library have")

    assert asked[:2] == ["Book", "Library"]
    assert intent.command == "Library/list_branches"
    assert intent.context == "Library" and intent.record is library


@pytest.mark.usefixtures("no_planner_key")
def test_the_cme_decision_stops_on_a_bare_value_from_the_head(on_a_book, monkeypatch):
    app, cme, _, _ = on_a_book
    _heads(app, "Book", "Library")
    asked = []
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router_by_context(
        {"Book": [PARAMETER_VALUE_LABEL], "Library": ["Library/list_branches"]}, asked))

    intent = CommandNamePrediction(cme).predict("france")

    assert set(asked) == {"Book"}, "a bare value ends the walk where it is given"
    assert intent.command is None and intent.bare_value


@pytest.mark.usefixtures("no_planner_key")
def test_the_classifier_layer_records_its_engine_on_the_trace(on_a_book, monkeypatch):
    app, cme, _, _ = on_a_book
    _heads(app, "Book", "Library")
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router(["Book/show_book"]))
    predictor = CommandNamePrediction(cme)
    intent = predictor.predict("tell me about this one")
    trace = predictor.trace
    assert intent.command == "Book/show_book"
    assert trace["matcher_layer"] == "classifier"
    # span contract v63: the engine that decided is `decided_by` on fw.intent ("local" for the local model)
    assert trace["decided_by"] == "local" and trace["engine_fallback"] is False

    # A layer above the classifier answers without an engine: the name decided, and no engine is named.
    predictor.predict("show_book")
    trace = predictor.trace
    assert trace["matcher_layer"] == "exact_prefix"
    assert trace["decided_by"] == "name" and "engine_fallback" not in trace and "engine_chain" not in trace


# ---------------------------------------------------------------------------
# detect: the walk over the engines alone
# ---------------------------------------------------------------------------

def _engines(labels_by_context: dict):
    def engine_of(context_name):
        labels = labels_by_context.get(context_name)
        return None if labels is None else ClassifierEngine(_router(labels)())
    return engine_of


def test_detect_returns_the_match_its_scope_and_every_answer_of_the_walk(on_a_book):
    app, _, library, book = on_a_book
    detection = intent_detector.detect(
        app, "which branches are there",
        engine_of=_engines({"Book": [WILDCARD_LABEL], "Library": ["Library/list_branches"]}))
    assert detection.outcome == "match"
    assert detection.command == "Library/list_branches"
    assert detection.scope == Scope("Library", library)
    assert [answer.kind for answer in detection.answers] == ["not_mine", "match"]
    assert all(isinstance(answer, ScopeAnswer) for answer in detection.answers)
    assert app.record is not library and app._current_command_context is book   # read only


def test_detect_names_the_routable_candidates_of_an_ambiguity(on_a_book):
    app, _, _, book = on_a_book
    detection = intent_detector.detect(
        app, "something about loans",
        engine_of=_engines({"Book": [WILDCARD_LABEL, "Book/list_loans", "Book/show_book"]}))
    assert detection.outcome == "ambiguous"
    assert detection.candidates == ["Book/list_loans", "Book/show_book"]
    assert detection.scope == Scope("Book", book)
    assert detection.answers[0].labels[0] == WILDCARD_LABEL   # the answer itself keeps what the head said


@pytest.mark.usefixtures("no_planner_key")   # with the key, the model matcher stands in (fix-yu3l.3)
def test_detect_is_none_when_every_scope_declines_or_has_no_engine(on_a_book):
    app, _, library, _ = on_a_book
    declined = intent_detector.detect(app, "x", engine_of=_engines({"Book": [WILDCARD_LABEL]}))
    assert declined.outcome == "none" and declined.command is None and declined.candidates == []
    assert declined.scope == Scope("Library", library)
    assert [answer.kind for answer in declined.answers] == ["not_mine"]   # Library has no engine to answer

    bare = intent_detector.detect(
        app, "france", engine_of=_engines({"Book": [PARAMETER_VALUE_LABEL], "Library": ["Library/find_book"]}))
    assert bare.outcome == "none" and [answer.kind for answer in bare.answers] == ["bare_value"]

    # With no trained head anywhere (this copy), the default engines answer nothing.
    assert intent_detector.detect(app, "who has this book").outcome == "none"


# ---------------------------------------------------------------------------
# run_listing.head_answers asks the same engines
# ---------------------------------------------------------------------------

def test_head_answers_reads_each_trained_head_through_the_contract(on_a_book, monkeypatch):
    from fastworkflow import model_pipeline_training

    app, _, _, _ = on_a_book
    assert run_listing.head_answers(app, "who has this book") == {}   # no head, no answer

    _fake_head(os.path.join(app.folderpath, "___command_info", "Book"))
    details = {
        "model_tier": "tiny", "confidence": 0.4, "ambiguous_threshold": 0.5, "confident": False,
        "top_label": "Book/loan_report",
        "topk_labels": ["Book/loan_report", "wildcard", "Book/list_loans"],
        "topk_scores": [0.4, 0.35, 0.25],
    }
    labels = ["Book/loan_report", "wildcard", "Book/list_loans"]
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router(labels, details))

    assert run_listing.head_answers(app, "who has this book") == {
        "Book": {
            "labels": labels,
            "confident": False,
            "topk_labels": ["Book/loan_report", "wildcard", "Book/list_loans"],
            "topk_scores": [0.4, 0.35, 0.25],
        },
    }
    assert run_listing.head_answers(app, "   ") == {}   # blank text is not put to any head
