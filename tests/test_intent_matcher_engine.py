"""The model matcher behind the detector contract (bd fix-yu3l.3), and the one detection chain
(bd fix-o5wi.2): the trained local model when it is confident, then the matcher when the planner key is
set, then the question to the user or the agent.

Integration tests against real components: a COPY of the bundled library example (its real routing
definition, application records, planning commands and the real ``CommandNamePrediction`` and CME
``wildcard`` command). Nothing inside the example's own ``___command_info`` is written or removed: the copy
has no trained heads, and the one test that needs real heads links them read-only into its copy.

Only the matcher's model call is scripted, the way the planner-key tests script Jev:
``workflow_agent._jev_lm`` is a ``DummyLM`` and
``workflow_agent.dspy.Predict`` returns a predictor that answers with the probabilities the test gives.
Tests of behaviour WITHOUT the planner key use the ``no_planner_key`` fixture.
"""

import contextvars
import json
import logging
import os
import shutil
from types import SimpleNamespace

import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import model_pipeline_training
from fastworkflow import intent_detector, run_listing, tracing, turn_cache, workflow_agent
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction, command_identity_uncertainty)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.intent_detector import AMBIGUOUS, MATCH, NOT_MINE, ClassifierEngine, MatcherEngine, Scope
from fastworkflow.nlu_labels import WILDCARD_LABEL
from tests.test_intent_detector import (  # noqa: F401 - shared fixtures and doubles
    LIBRARY, _fake_head, _router, library_copy, on_a_book)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

BRANCHES = "which branches does the library have"
CHAIN_LINE = ("Intent detection: the trained local model decides when it is confident; otherwise the remote "
              "decision model (JEV_API_KEY) decides when it is confident; otherwise the user or the agent is asked.")
NO_KEY_LINE = ("Intent detection: the trained local model decides when it is confident; otherwise the user or the "
               "agent is asked (JEV_API_KEY is not set, so there is no remote decision model).")


def _untrained(*folders: str) -> str:
    """The chain line of a workflow whose contexts *folders* have no trained local model."""
    many = len(folders) > 1
    return (f"{CHAIN_LINE} Context{'s' if many else ''} {', '.join(folders)} {'have' if many else 'has'} no "
            f"trained local model: the remote decision model decides there (slower, one model call per request). "
            f"Run 'fastworkflow train' for the local model.")


NOTICE = _untrained("Book", "Library", "Member", "Record", "global")


# ---------------------------------------------------------------------------
# The scripted model
# ---------------------------------------------------------------------------

class ScriptedMatcher:
    """What the matcher's model answers: ``scores`` is ``{option: probability}``, or a function of the
    message, or an exception to raise. ``calls`` holds every call made: ``(signature, inputs)``."""

    def __init__(self, scores):
        self.scores = scores
        self.calls: list[tuple] = []

    def predict(self, signature):
        def answer(**inputs):
            self.calls.append((signature, inputs))
            scores = self.scores(inputs["message"]) if callable(self.scores) else self.scores
            if isinstance(scores, Exception):
                raise scores
            value = max(scores, key=scores.get)
            choice = SimpleNamespace(value=value, probabilities=dict(scores), confidence=scores[value])
            return SimpleNamespace(command=choice)

        return answer

    def options(self, call: int = 0) -> dict:
        """The Choice options of one call, as ``{value: description}``."""
        return json.loads(self.calls[call][0].output_fields["command"].annotation._criteria_json)

    def listing(self, call: int = 0) -> str:
        return self.calls[call][1]["commands"]


@pytest.fixture
def matcher_model(monkeypatch):
    """Script the matcher's model: ``matcher_model(scores)`` returns the ``ScriptedMatcher`` that answers."""
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        name: value for name, value in fastworkflow._env_vars.items() if name != "INTENT_DETECTION_ENGINE"})

    def script(scores) -> ScriptedMatcher:
        scripted = ScriptedMatcher(scores)
        monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
        monkeypatch.setattr(workflow_agent.dspy, "Predict", scripted.predict)
        return scripted

    return script


def _setting(monkeypatch, value: str) -> None:
    monkeypatch.setitem(fastworkflow._env_vars, "INTENT_DETECTION_ENGINE", value)


def _outside_a_turn(function):
    """Run *function* in a context of its own, so a turn memo it opens does not outlive the test."""
    return contextvars.copy_context().run(function)


# ---------------------------------------------------------------------------
# One grouped distribution, answered per scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scores, book, library", [
    # the nearest scope's command at or above the accept bar (0.9): a match there, nothing for the enclosing scope
    ({"Book/list_loans": 0.9, "none": 0.1}, (MATCH, ["Book/list_loans"]), (NOT_MINE, [WILDCARD_LABEL])),
    # just under it (the old 0.5 bar): not an answer, the scope's candidates go to the question
    ({"Book/list_loans": 0.89, "Book/show_book": 0.06, "none": 0.05},
     (AMBIGUOUS, ["Book/list_loans"]), (NOT_MINE, [WILDCARD_LABEL])),
    ({"Book/list_loans": 0.5, "Book/show_book": 0.3, "none": 0.2},
     (AMBIGUOUS, ["Book/list_loans", "Book/show_book"]), (NOT_MINE, [WILDCARD_LABEL])),
    # the model chose a command of the enclosing scope at or above the decline bar (0.9): the nearest declines
    ({"Library/list_branches": 0.92, "Book/show_book": 0.05, "none": 0.03},
     (NOT_MINE, [WILDCARD_LABEL]), (MATCH, ["Library/list_branches"])),
    # a command of the enclosing scope below it: the nearest asks about its own candidates, the enclosing scope
    # asks about its own, no scope is declined
    ({"Library/list_branches": 0.6, "Book/show_book": 0.3, "none": 0.1},
     (AMBIGUOUS, ["Book/show_book"]), (AMBIGUOUS, ["Library/list_branches"])),
    # nothing reaches the accept bar: the candidates above 0.2, best first
    ({"Book/show_book": 0.35, "Book/list_loans": 0.4, "none": 0.25},
     (AMBIGUOUS, ["Book/list_loans", "Book/show_book"]), (NOT_MINE, [WILDCARD_LABEL])),
    ({"Book/show_book": 0.3, "Library/find_book": 0.45, "none": 0.25},
     (AMBIGUOUS, ["Book/show_book"]), (AMBIGUOUS, ["Library/find_book"])),
    # at most three candidates
    ({"Book/show_book": 0.22, "Book/list_loans": 0.26, "Record/add_note": 0.24, "Record/show_notes": 0.23,
      "none": 0.05},
     (AMBIGUOUS, ["Book/list_loans", "Record/add_note", "Record/show_notes"]), (NOT_MINE, [WILDCARD_LABEL])),
    # "none" at or above the decline bar (0.9) declines everywhere, whatever else is above 0.2
    ({"none": 0.9, "Book/list_loans": 0.06, "Library/find_book": 0.04},
     (NOT_MINE, [WILDCARD_LABEL]), (NOT_MINE, [WILDCARD_LABEL])),
    # "none" below it does not decline: the candidates above 0.2 are asked
    ({"none": 0.5, "Book/list_loans": 0.3, "Library/find_book": 0.2},
     (AMBIGUOUS, ["Book/list_loans"]), (NOT_MINE, [WILDCARD_LABEL])),
    # no candidate above 0.2 (0.2 itself is not above it)
    ({"Book/list_loans": 0.2, "Book/show_book": 0.2, "Library/find_book": 0.2, "none": 0.4},
     (NOT_MINE, [WILDCARD_LABEL]), (NOT_MINE, [WILDCARD_LABEL])),
])
def test_each_scope_is_answered_from_the_one_grouped_distribution(library_copy, matcher_model, scores, book, library):
    model = matcher_model(scores)
    memo: dict = {}
    path = ("Book", "Library")

    book_answer = MatcherEngine(library_copy, "Book", path, memo=memo).decide("anything")
    library_answer = MatcherEngine(library_copy, "Library", path, memo=memo).decide("anything")

    assert (book_answer.kind, book_answer.labels) == book
    assert (library_answer.kind, library_answer.labels) == library
    assert len(model.calls) == 1                                   # one model call for both scopes
    for answer in (book_answer, library_answer):
        assert answer.engine == "matcher" and answer.fallback is True
        assert answer.confident is (answer.kind == MATCH)


def test_the_answer_carries_the_engines_facts(library_copy, matcher_model):
    matcher_model({"Book/list_loans": 0.92, "Book/show_book": 0.04, "Library/find_book": 0.02, "none": 0.02})
    answer = MatcherEngine(library_copy, "Book", ("Book", "Library")).decide("what is out on loan")

    details = answer.details
    assert details["confident"] is True
    assert details["top_label"] == "Book/list_loans" and details["confidence"] == pytest.approx(0.92)
    assert details["topk_labels"][:2] == ["Book/list_loans", "Book/show_book"]
    assert details["topk_scores"][:2] == pytest.approx([0.92, 0.04])
    assert len(details["topk_labels"]) == len(details["topk_scores"]) <= 5
    assert details["model"] == "dummy" and details["latency_ms"] >= 0.0
    assert details["none_score"] == pytest.approx(0.02) and details["scope_path"] == ["Book", "Library"]
    assert details["calls"] == 1 and "error" not in details
    # the scores are this scope's candidates and "none", never another scope's commands
    assert answer.scores["Book/list_loans"] == pytest.approx(0.92) and answer.scores["none"] == pytest.approx(0.02)
    assert "Library/find_book" not in answer.scores
    assert isinstance(MatcherEngine(library_copy, "Book"), intent_detector.Engine)


def test_the_prompt_lists_every_scope_of_the_path_grouped_and_never_a_core_command(library_copy, matcher_model):
    model = matcher_model({"none": 1.0})
    MatcherEngine(library_copy, "Book", ("Book", "Library", "*")).decide("anything")

    offered = {name: [c for c in intent_detector.candidates(library_copy, name) if c.kind != "core"]
               for name in ("Book", "Library", "*")}
    names = [c.name for group in offered.values() for c in group]
    options = model.options()
    assert list(options) == names + ["none"]                       # nearest scope first, "none" last
    assert not any(name.startswith(("IntentDetection/", "ErrorCorrection/")) for name in options)
    assert {"Book/list_loans", "Record/add_note", "Book/loan_report", "Library/find_book",
            "Library/author_books", "library_hours"} <= set(options)        # execution and planning commands
    assert options["none"] == "None of these commands fits the message."

    listing = model.listing()
    blocks = listing.split("\n\n")
    assert [block.splitlines()[0] for block in blocks] == ["Scope Book:", "Scope Library:", "Scope global:"]
    list_branches = next(c for c in offered["Library"] if c.name == "Library/list_branches")
    assert intent_detector.candidate_line(
        list_branches.name, list_branches.description, list_branches.examples,
        workflow_agent.MATCHER_EXAMPLES) in blocks[1].splitlines()
    assert '- Library/list_branches: List the library\'s branches. e.g. "which branches does the library have", ' \
        in listing
    for line in listing.splitlines():                              # the cap on examples holds for every command
        assert line.count('", "') < workflow_agent.MATCHER_EXAMPLES
    assert model.calls[0][1]["message"] == "anything"


def test_a_command_two_scopes_offer_is_listed_once_under_the_nearest(library_copy, matcher_model):
    model = matcher_model({"Record/add_note": 0.9, "none": 0.1})
    memo: dict = {}
    path = ("Book", "Record")
    book = MatcherEngine(library_copy, "Book", path, memo=memo).decide("note this")
    record = MatcherEngine(library_copy, "Record", path, memo=memo).decide("note this")

    assert list(model.options()).count("Record/add_note") == 1
    assert "Scope Record:" not in model.listing()                  # everything Record offers, Book already listed
    assert book.labels == ["Record/add_note"] and record.labels == ["Record/add_note"]


def test_a_path_with_no_command_is_not_asked(library_copy, matcher_model, monkeypatch):
    model = matcher_model({"none": 1.0})
    monkeypatch.setattr(intent_detector, "matcher_candidates", lambda folder, context: [])
    answer = MatcherEngine(library_copy, "Book").decide("anything")
    assert answer.kind == NOT_MINE and model.calls == [] and answer.details["calls"] == 0


def test_a_path_above_the_option_limit_is_asked_scope_by_scope(library_copy, matcher_model, monkeypatch):
    """The fallback for a model that cannot take every option in one choice: one call per scope."""
    monkeypatch.setattr(intent_detector, "MATCHER_MAX_OPTIONS", 5)
    model = matcher_model(lambda message: {"Library/list_branches": 0.9, "none": 0.1})
    memo: dict = {}
    path = ("Book", "Library")
    book = MatcherEngine(library_copy, "Book", path, memo=memo).decide(BRANCHES)
    library = MatcherEngine(library_copy, "Library", path, memo=memo).decide(BRANCHES)

    assert len(model.calls) == 2 and book.details["calls"] == 2
    assert all(not name.startswith("Library/") for name in model.options(0))
    assert "Library/list_branches" in model.options(1) and "Book/list_loans" not in model.options(1)
    assert library.kind == MATCH and library.labels == ["Library/list_branches"]


def test_a_failed_model_call_declines_with_the_error_and_never_raises(library_copy, matcher_model, caplog):
    model = matcher_model(RuntimeError("the model is down"))
    with caplog.at_level(logging.WARNING):
        answer = MatcherEngine(library_copy, "Book", ("Book", "Library")).decide("anything")

    assert answer.kind == NOT_MINE and answer.labels == [WILDCARD_LABEL] and answer.confident is False
    assert answer.details["error"] == "RuntimeError: the model is down"
    assert answer.scores == {} and len(model.calls) == 1
    assert answer.details["error_class"] == "RuntimeError" and answer.details["status_code"] is None
    notices = [r for r in caplog.records if "remote decision model failed" in r.getMessage()]
    assert len(notices) == 1 and "RuntimeError" in notices[0].getMessage()
    assert "the model is down" not in notices[0].getMessage()      # the class and the status, never the message


class _CreditsExhausted(Exception):
    """What litellm raises for HTTP 402: an error carrying ``status_code``."""
    status_code = 402


@pytest.fixture
def fresh_outage_counter():
    from fastworkflow import remote_outage

    remote_outage.reset()
    yield remote_outage
    remote_outage.reset()


def test_an_outage_is_logged_first_then_at_most_once_per_interval_with_a_count(
        library_copy, matcher_model, caplog, monkeypatch, fresh_outage_counter):
    outage = fresh_outage_counter
    clock = [1000.0]
    monkeypatch.setattr(outage.time, "monotonic", lambda: clock[0])
    matcher_model(_CreditsExhausted("402 Payment Required: sk-secret-key-123"))

    def ask(text: str):
        return MatcherEngine(library_copy, "Book", ("Book", "Library")).decide(text)

    with caplog.at_level(logging.WARNING):
        first = ask("one")
        ask("two")
        ask("three")
        clock[0] += outage.NOTICE_INTERVAL_SECONDS - 1
        ask("four")
        notices_before = [r.getMessage() for r in caplog.records if "remote decision model failed" in r.getMessage()]
        clock[0] += 2
        ask("five")

    notices = [r.getMessage() for r in caplog.records if "remote decision model failed" in r.getMessage()]
    assert len(notices_before) == 1 and len(notices) == 2          # the first failure, then one after the interval
    assert "_CreditsExhausted, status 402" in notices[0] and "1 failure(s)" in notices[0]
    assert "4 failure(s)" in notices[1]                               # the four since the first notice
    assert all("sk-secret-key-123" not in message for message in notices)
    assert first.details["error_class"] == "_CreditsExhausted" and first.details["status_code"] == 402


def test_a_failed_matcher_call_leaves_the_fallback_unchanged_and_marks_the_span(
        on_a_book, matcher_model, monkeypatch, fresh_outage_counter):
    """The chain's fallback is as it was (the local answer stands); the span trace now says the remote call failed."""
    app, cme, _, _ = on_a_book
    matcher_model(_CreditsExhausted("out of credits"))
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book", "Book/list_loans", WILDCARD_LABEL], UNSURE)
    trace: dict = {}

    output = predictor.predict("what is out on loan")

    trace = predictor.trace

    assert output.command is None or output.candidates                   # the question, as before the marker
    assert trace["remote_failed"] == {"error_class": "_CreditsExhausted", "status_code": 402}
    assert trace["engine"] == "question" and trace["engine_chain"][-1] == "matcher"


def test_the_chain_fallback_carries_the_failure_class_and_status(library_copy, matcher_model, fresh_outage_counter):
    matcher_model(_CreditsExhausted("out of credits"))
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    chain = intent_detector.engine_for(
        library_copy, "Book", router_factory=_router(["Book/show_book", "Book/list_loans"], UNSURE),
        scope_path=("Book", "Library"))

    answer = chain.decide("anything")

    assert (answer.kind, answer.labels, answer.engine) == ("ambiguous", ["Book/show_book", "Book/list_loans"], "classifier")
    assert answer.details["matcher_error_class"] == "_CreditsExhausted" and answer.details["matcher_status_code"] == 402


def test_the_chain_reports_an_outage_in_the_shared_notice_and_still_answers(
        library_copy, matcher_model, caplog, fresh_outage_counter):
    """(was the planning-command match's notice; the one remote decision shares the outage notice and never raises.)"""
    matcher_model(_CreditsExhausted("out of credits"))
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    chain = intent_detector.engine_for(
        library_copy, "Book", router_factory=_router(["Book/show_book", "Book/list_loans"], UNSURE),
        scope_path=("Book", "Library"))
    with caplog.at_level(logging.WARNING):
        assert chain.decide("anything").kind == "ambiguous"
    messages = [r.getMessage() for r in caplog.records if "remote decision model failed" in r.getMessage()]
    assert len(messages) == 1 and "command match" in messages[0] and "status 402" in messages[0]


def test_the_two_remote_thresholds_are_constants_with_one_setting_each(monkeypatch, caplog):
    from fastworkflow.train import intent_model_layout

    assert intent_detector.DEFAULT_REMOTE_ACCEPT_THRESHOLD == 0.9 == intent_detector.DEFAULT_REMOTE_DECLINE_THRESHOLD
    assert intent_detector.remote_accept_threshold() == 0.9 and intent_detector.remote_decline_threshold() == 0.9
    monkeypatch.setitem(fastworkflow._env_vars, "INTENT_REMOTE_ACCEPT_THRESHOLD", "0.7")
    monkeypatch.setitem(fastworkflow._env_vars, "INTENT_REMOTE_DECLINE_THRESHOLD", "0.8")
    assert intent_detector.remote_accept_threshold() == 0.7 and intent_detector.remote_decline_threshold() == 0.8
    monkeypatch.setitem(fastworkflow._env_vars, "INTENT_REMOTE_ACCEPT_THRESHOLD", "high")
    monkeypatch.setattr(intent_model_layout, "_warned_values", set())
    with caplog.at_level(logging.WARNING):
        assert intent_detector.remote_accept_threshold() == 0.9
        assert intent_detector.remote_accept_threshold() == 0.9
    warnings = [r.getMessage() for r in caplog.records if "INTENT_REMOTE_ACCEPT_THRESHOLD" in r.getMessage()]
    assert len(warnings) == 1                                         # one warning, the default applies


def test_the_settings_move_the_bars_and_the_answer_records_them(library_copy, matcher_model, monkeypatch):
    matcher_model({"Book/list_loans": 0.7, "Book/show_book": 0.2, "Library/find_book": 0.06, "none": 0.04})
    path = ("Book", "Library")
    default = MatcherEngine(library_copy, "Book", path).decide("what is out on loan")
    assert default.kind == AMBIGUOUS and default.details["threshold"] == 0.9
    assert default.details["decline_threshold"] == 0.9
    monkeypatch.setitem(fastworkflow._env_vars, "INTENT_REMOTE_ACCEPT_THRESHOLD", "0.5")
    lowered = MatcherEngine(library_copy, "Book", path).decide("what is out on loan again")
    assert lowered.kind == MATCH and lowered.labels == ["Book/list_loans"] and lowered.details["threshold"] == 0.5


# ---------------------------------------------------------------------------
# One model call per detection
# ---------------------------------------------------------------------------

def test_detect_walks_to_the_enclosing_scope_on_one_model_call(on_a_book, matcher_model):
    app, _, library, book = on_a_book
    model = matcher_model({"Library/list_branches": 0.92, "Book/show_book": 0.05, "none": 0.03})

    detection = intent_detector.detect(app, BRANCHES)

    assert detection.outcome == "match" and detection.command == "Library/list_branches"
    assert detection.scope == Scope("Library", library)
    assert [answer.kind for answer in detection.answers] == [NOT_MINE, MATCH]
    assert all(answer.engine == "matcher" and answer.fallback for answer in detection.answers)
    assert len(model.calls) == 1
    assert app._current_command_context is book                    # detect reads the workflow, never moves it


def test_detect_takes_the_nearest_scope_when_it_matches(on_a_book, matcher_model):
    app, _, _, book = on_a_book
    model = matcher_model({"Book/list_loans": 0.92, "Library/find_book": 0.05, "none": 0.03})
    detection = intent_detector.detect(app, "what is out on loan")
    assert detection.command == "Book/list_loans" and detection.scope == Scope("Book", book)
    assert len(detection.answers) == 1 and len(model.calls) == 1


def test_detect_reports_an_ambiguity_and_a_none(on_a_book, matcher_model):
    app, _, _, _ = on_a_book
    matcher_model({"Book/list_loans": 0.4, "Book/show_book": 0.35, "none": 0.25})
    ambiguous = intent_detector.detect(app, "the book")
    assert ambiguous.outcome == "ambiguous" and ambiguous.candidates == ["Book/list_loans", "Book/show_book"]

    matcher_model({"none": 0.97, "Book/list_loans": 0.03})
    assert intent_detector.detect(app, "sing me a song").outcome == "none"


def test_the_call_is_memoised_for_the_turn_when_a_turn_is_open(library_copy, matcher_model):
    model = matcher_model({"Library/list_branches": 0.9, "none": 0.1})
    path = ("Book", "Library")

    def in_a_turn():
        turn_cache.begin_turn()
        for context in ("Book", "Library", "Book"):                # separate engines, no shared memo dict
            MatcherEngine(library_copy, context, path).decide(BRANCHES)
        first = len(model.calls)
        MatcherEngine(library_copy, "Book", path).decide("another request")          # another utterance
        MatcherEngine(library_copy, "Book", ("Book",)).decide(BRANCHES)              # another path
        return first, len(model.calls)

    assert _outside_a_turn(in_a_turn) == (1, 3)


def test_without_a_turn_the_memo_is_the_one_the_engines_share(library_copy, matcher_model, request):
    model = matcher_model({"Library/list_branches": 0.9, "none": 0.1})
    path = ("Book", "Library")
    # A test that ran a turn earlier in the session leaves its memo bound; this test is about no turn at all.
    token = turn_cache._memo_var.set(None)
    request.addfinalizer(lambda: turn_cache._memo_var.reset(token))
    assert not turn_cache.active()
    engine = MatcherEngine(library_copy, "Book", path)
    engine.decide(BRANCHES)
    engine.decide(BRANCHES)
    assert len(model.calls) == 1                                   # the engine's own memo
    MatcherEngine(library_copy, "Library", path).decide(BRANCHES)
    assert len(model.calls) == 2                                   # an engine with a memo of its own asks again


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head, expected", [(True, "chain"), (False, "matcher")])
def test_the_links_of_a_scope_with_the_planner_key(library_copy, matcher_model, head, expected):
    matcher_model({"none": 1.0})
    if head:
        _fake_head(os.path.join(library_copy, "___command_info", "Book"))

    assert intent_detector.engine_choice(library_copy, "Book") == expected
    engine = intent_detector.engine_for(library_copy, "Book", router_factory=_router(["Book/show_book"]))
    if head:
        assert isinstance(engine, ChainEngine) and isinstance(engine, intent_detector.Engine)
        assert engine.matcher.fallback is False
    else:
        assert isinstance(engine, MatcherEngine)
        assert engine.fallback is True                             # it stands in for a local model that is not there


@pytest.mark.parametrize("head, expected", [(True, "classifier"), (False, None)])
def test_the_links_of_a_scope_without_the_planner_key(library_copy, no_planner_key, monkeypatch, head, expected):
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    if head:
        _fake_head(os.path.join(library_copy, "___command_info", "Book"))

    assert intent_detector.matcher_available() is False
    assert intent_detector.engine_choice(library_copy, "Book") == expected
    engine = intent_detector.engine_for(library_copy, "Book", router_factory=_router(["Book/show_book"]))
    assert (engine is None) if expected is None else isinstance(engine, ClassifierEngine)
    if engine is not None:
        assert engine.fallback is False


@pytest.mark.parametrize("setting", ["auto", "classifier", "matcher", "hybrid", "fastest-please"])
def test_the_removed_engine_setting_is_ignored_with_one_warning(
        library_copy, matcher_model, monkeypatch, caplog, setting):
    """``INTENT_DETECTION_ENGINE`` chose among engines. The chain is the only behaviour now: a value left in
    an env file changes nothing, and one warning per process says it is ignored."""
    matcher_model({"none": 1.0})
    _setting(monkeypatch, setting)
    monkeypatch.setattr(intent_detector, "_warned_engine_setting", False)
    with caplog.at_level(logging.WARNING):
        assert intent_detector.engine_choice(library_copy, "Book") == "matcher"       # no local model: the matcher
        _fake_head(os.path.join(library_copy, "___command_info", "Book"))
        assert intent_detector.engine_choice(library_copy, "Book") == "chain"         # a local model: the chain
        intent_detector.startup_report(library_copy, [])
    warnings = [r.getMessage() for r in caplog.records if "INTENT_DETECTION_ENGINE" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"INTENT_DETECTION_ENGINE={setting!r} is ignored: the setting was removed.")
    assert not hasattr(intent_detector, "engine_setting") and not hasattr(intent_detector, "engine_mode")


def test_fw_require_heads_makes_a_missing_local_model_a_failure(library_copy, matcher_model, monkeypatch):
    matcher_model({"none": 1.0})
    monkeypatch.setenv("FW_REQUIRE_HEADS", "1")

    assert intent_detector.heads_required() is True
    assert intent_detector.engine_choice(library_copy, "Book") is None           # the matcher does not stand in
    assert intent_detector.engine_for(library_copy, "Book") is None
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    assert intent_detector.engine_choice(library_copy, "Book") == "chain"        # a trained scope keeps its chain
    assert intent_detector.startup_report(library_copy, ["Member"])[0]          # a missing head is a problem


def test_gathering_stays_classifier_only(on_a_book, matcher_model, monkeypatch):
    """``run_listing.head_answers`` asks every reachable context: a context with no head answers nothing,
    and no model call is made for it."""
    app, _, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.9, "none": 0.1})
    assert intent_detector.engine_for(app.folderpath, "Book", allow_matcher=False) is None
    assert run_listing.head_answers(app, "what is out on loan") == {}
    assert model.calls == []

    _fake_head(os.path.join(app.folderpath, "___command_info", "Book"))
    engine = intent_detector.engine_for(
        app.folderpath, "Book", router_factory=_router(["Book/list_loans"]), allow_matcher=False)
    assert isinstance(engine, ClassifierEngine)                    # the local model alone, though the key is set


# ---------------------------------------------------------------------------
# The CME path
# ---------------------------------------------------------------------------

def test_the_cme_step_asks_the_matcher_for_a_scope_with_no_head(on_a_book, matcher_model):
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.92, "Book/show_book": 0.05, "none": 0.03})
    predictor = CommandNamePrediction(cme)
    trace: dict = {}

    output = predictor.predict("what is out on loan")

    trace = predictor.trace

    assert output.command == "Book/list_loans" and output.outcome != "ambiguous"
    assert output.kind != "core" and output.bare_value is False
    assert trace["matcher_layer"] == "matcher" and trace["engine"] == "matcher" and trace["engine_fallback"] is True
    assert trace["engine_chain"] == ["matcher"] and "engine_mode" not in trace
    assert trace["matcher"]["top_label"] == "Book/list_loans" and trace["matcher"]["model"] == "dummy"
    assert trace["matcher"]["scope_path"] == ["Book", "Library", "*"] and trace["candidate_count"] == 1
    assert "classifier" not in trace and "jev_tiebreak" not in trace
    assert len(model.calls) == 1
    uncertainty = command_identity_uncertainty(trace)
    assert uncertainty.signals == () and uncertainty.signals_absent_reason == "not-instrumented"
    assert uncertainty.candidate_count == 1


def test_the_cme_walk_climbs_on_a_decline_with_one_model_call(on_a_book, matcher_model):
    app, cme, library, _ = on_a_book
    model = matcher_model({"Library/list_branches": 0.9, "Book/show_book": 0.05, "none": 0.05})
    predictor = CommandNamePrediction(cme)

    intent = predictor.predict(BRANCHES)

    assert intent.command == "Library/list_branches"
    assert intent.context == "Library" and intent.record is library
    assert len(model.calls) == 1                                   # the walk asked two scopes on one call


def test_the_cme_step_turns_an_ambiguity_into_the_standard_question(on_a_book, matcher_model):
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/show_book": 0.35, "Book/list_loans": 0.4, "none": 0.25})
    predictor = CommandNamePrediction(cme)
    trace: dict = {}

    output = predictor.predict("the book")

    trace = predictor.trace

    labels = ["Book/list_loans", "Book/show_book"]
    assert output.command is None and list(output.ambiguous) == labels
    assert output.outcome == "ambiguous" and output.decided_by == "question"
    assert trace["candidates"] == labels and trace["reducible"] is True and trace["candidate_count"] == 2
    assert trace["engine"] == "question" and trace["engine_chain"] == ["matcher"]     # nobody was confident
    assert "jev_tiebreak" not in trace and len(model.calls) == 1


def test_a_failed_matcher_call_in_the_cme_step_is_a_decline(on_a_book, matcher_model):
    app, cme, _, _ = on_a_book
    matcher_model(RuntimeError("the model is down"))
    predictor = CommandNamePrediction(cme)
    output = predictor.predict("what is out on loan")
    trace = predictor.trace
    assert output.command is None and output.outcome != "ambiguous" and not output.bare_value
    assert trace["matcher"]["error"] == "RuntimeError: the model is down" and trace["candidate_count"] == 0
    assert trace["remote_failed"] == {"error_class": "RuntimeError", "status_code": None}


def test_an_exact_name_needs_no_matcher(on_a_book, matcher_model):
    app, cme, _, _ = on_a_book
    model = matcher_model({"none": 1.0})
    predictor = CommandNamePrediction(cme)
    output = predictor.predict("list_loans")
    trace = predictor.trace
    assert output.command == "Book/list_loans" and trace["matcher_layer"] == "exact_prefix"
    assert model.calls == [] and "engine" not in trace


def test_a_router_a_test_supplies_is_asked_and_the_matcher_is_not(on_a_book, matcher_model, monkeypatch):
    """A scope with a head on disk has a local model: it answers first, and a confident answer needs no matcher.
    (A router behind the trainer's ``CommandRouter`` name is what answers; a scope with no head on disk has none.)"""
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.9, "none": 0.1})
    _fake_head(os.path.join(app.folderpath, "___command_info", "Book"))

    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router(["Book/show_book"]))
    predictor = CommandNamePrediction(cme)
    output = predictor.predict("what is out on loan")
    trace = predictor.trace
    assert output.command == "Book/show_book" and trace["engine"] == "classifier"
    assert model.calls == []
    assert intent_detector.engine_choice(app.folderpath, "Book") == "chain"
    assert intent_detector.engine_choice(app.folderpath, "Member") == "matcher"


def test_without_the_planner_key_a_scope_with_no_head_has_no_matcher(on_a_book, no_planner_key, monkeypatch):
    app, cme, _, _ = on_a_book
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    assert intent_detector.engine_for(app.folderpath, "Book") is None        # no head, no key: nothing can decide
    intent = CommandNamePrediction(cme).predict("what is out on loan")
    assert intent.command is None and intent.outcome == "none"              # the misunderstanding question follows


def test_the_span_contract_names_the_matchers_keys():
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_NLU_INTENT]
    assert contract.version == 12
    assert {"engine", "engine_fallback", "engine_chain", "matcher", "matcher_layer", "remote_failed"} <= contract.attributes
    assert not {"engine_mode", "jev_tiebreak"} & contract.attributes        # gone with the setting and the tie-break
    assert not {"cache_similarity", "cache_similarity_threshold"} & contract.attributes   # gone with the similarity cache


def test_the_wildcard_command_routes_an_untrained_workflow_end_to_end(on_a_book, matcher_model):
    """From a Book of a workflow with no trained head at all, a request for a Library command: the CME
    ``wildcard`` command walks Book then Library on the matcher's one call, and the command it names is the
    Library's, bound to the Library record."""
    app, cme, library, book = on_a_book
    model = matcher_model({"Library/list_branches": 0.92, "Book/show_book": 0.05, "none": 0.03})

    out = CommandExecutor.perform_action(cme, fastworkflow.Action(command_name="wildcard", command=BRANCHES))

    artifacts = out.command_response.artifacts
    assert artifacts["command_name"] == "Library/list_branches"
    assert artifacts["ran_on_context_name"] == "Library"
    assert app.record is library and app._current_command_context is book
    assert len(model.calls) == 1


@pytest.fixture
def library_without_the_book_head(tmp_path, initialized_fastworkflow):  # noqa: F811 - fixture
    """A copy of the library whose Library, Member and Record heads are the example's trained ones, linked
    read-only, and whose Book head (and global head) is not there. Skips when the example is not trained."""
    source = os.path.join(LIBRARY, "___command_info")
    heads = {name: os.path.realpath(os.path.join(source, name)) for name in ("Library", "Member", "Record")}
    if not all(intent_detector.has_head(folder) for folder in heads.values()):
        pytest.skip("the library example has no trained heads here (run `fastworkflow train` on it)")
    folder = tmp_path / "library_wf"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    os.makedirs(folder / "___command_info")
    for name, target in heads.items():
        os.symlink(target, folder / "___command_info" / name)
    return str(folder)


def test_one_context_without_its_head_runs_on_the_matcher_and_the_rest_on_their_heads(
        library_without_the_book_head, matcher_model, monkeypatch):
    folder = library_without_the_book_head
    app = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str="matcher-mixed")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id, workflow_context={"app_workflow": app})
    if app.root is None:
        from fastworkflow.examples.library.application.library import Library

        app.root = Library()
    app.record = app.root.books["b1"]
    model = matcher_model({"Library/list_branches": 0.9, "Book/show_book": 0.05, "none": 0.05})
    predictor = CommandNamePrediction(cme)

    assert intent_detector.engine_choice(folder, "Book") == "matcher"
    assert intent_detector.engine_choice(folder, "Library") == "chain"
    # One decision for the whole walk, the local model over every scope first: Book has no head (nothing local to
    # ask), Library has one and is confident, so it decides with no model call; the matcher is not asked at all.
    intent = predictor.predict("Show me all of the library's branches")
    trace = predictor.trace

    assert intent.command == "Library/list_branches" and intent.context == "Library"
    assert trace["engine"] == "classifier" and trace["engine_chain"] == ["classifier"]
    assert model.calls == []
    problems, notices = intent_detector.startup_report(folder)
    assert problems == [] and notices == [_untrained("Book", "global")]


# ---------------------------------------------------------------------------
# Startup: never silent
# ---------------------------------------------------------------------------

MISSING = ["Book", "Library", "Member", "Record", "global"]


def test_untrained_contexts_the_matcher_decides_start_with_one_notice(library_copy, matcher_model):
    matcher_model({"none": 1.0})
    assert intent_detector.startup_report(library_copy, MISSING) == ([], [NOTICE])
    assert intent_detector.startup_report(library_copy, ["Book"]) == ([], [_untrained("Book")])
    # Every context trained: still one line, the chain.
    assert intent_detector.startup_report(library_copy, []) == ([], [CHAIN_LINE])


def test_a_gate_that_loads_the_workflow_builds_its_routing_to_find_the_untrained_contexts(
        library_copy, matcher_model):
    matcher_model({"none": 1.0})
    # A gate that only looks does not build: a workflow that was never built stays refused there.
    problems, notices = intent_detector.startup_report(library_copy)
    assert notices == [] and "<workflow not built>" in problems[0]
    assert not os.path.exists(os.path.join(library_copy, "___command_info"))

    assert intent_detector.startup_report(library_copy, build=True) == ([], [NOTICE])
    assert os.path.isfile(os.path.join(library_copy, "___command_info", "routing_definition.json"))
    assert intent_detector.startup_report(library_copy) == ([], [NOTICE])      # built now: found without building


def test_a_workflow_that_was_never_built_is_not_built_for_a_refusal(library_copy, no_planner_key, monkeypatch):
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    problems, notices = intent_detector.startup_report(library_copy, build=True)
    assert notices == [] and "<workflow not built>" in problems[0]
    assert not os.path.exists(os.path.join(library_copy, "___command_info"))


def test_fw_require_heads_keeps_the_refusal(library_copy, matcher_model, monkeypatch):
    matcher_model({"none": 1.0})
    monkeypatch.setenv("FW_REQUIRE_HEADS", "1")
    problems, notices = intent_detector.startup_report(library_copy, MISSING)
    assert notices == [] and problems == [
        "Workflow 'library_wf' is not trained (missing model artifacts for context(s): "
        "Book, Library, Member, Record, global)."]


def test_without_the_planner_key_the_refusal_stays(library_copy, no_planner_key, monkeypatch):
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    problems, notices = intent_detector.startup_report(library_copy, MISSING)
    assert notices == [] and len(problems) == 1 and "is not trained" in problems[0]
    # Every context trained and no key: the chain is the local model, then the question.
    assert intent_detector.startup_report(library_copy, []) == ([], [NO_KEY_LINE])


def test_a_workflow_that_was_never_built_is_refused_by_a_gate_that_only_looks(library_copy, matcher_model):
    matcher_model({"none": 1.0})
    problems, notices = intent_detector.startup_report(library_copy, ["<workflow not built>"])
    assert notices == [] and "<workflow not built>" in problems[0]


def test_one_context_the_matcher_cannot_decide_refuses_the_start(library_copy, matcher_model, monkeypatch):
    matcher_model({"none": 1.0})
    monkeypatch.setattr(
        intent_detector, "engine_choice",
        lambda folder, context, **_: None if context == "*" else "matcher")
    problems, notices = intent_detector.startup_report(library_copy, MISSING)
    assert notices == [] and len(problems) == 1


def test_the_chat_page_lists_an_untrained_workflow_as_runnable_only_when_the_matcher_decides(
        on_a_book, matcher_model, monkeypatch):
    from fastworkflow.run_chatbot.workflow_discovery import _workflow_is_trained

    app, _, _, _ = on_a_book
    matcher_model({"none": 1.0})
    assert _workflow_is_trained(app.folderpath) is False           # never built: the listing does not build it
    fastworkflow.RoutingRegistry.get_definition(app.folderpath)    # loaded once: its routing is on disk
    assert _workflow_is_trained(app.folderpath) is True
    monkeypatch.setenv("FW_REQUIRE_HEADS", "1")
    assert _workflow_is_trained(app.folderpath) is False


def test_the_mcp_server_still_refuses_an_untrained_workflow_when_heads_are_required(
        on_a_book, matcher_model, monkeypatch):
    from fastworkflow import mcp_server

    app, _, _, _ = on_a_book
    matcher_model({"none": 1.0})
    monkeypatch.setenv("FW_REQUIRE_HEADS", "1")
    with pytest.raises(RuntimeError) as refusal:
        mcp_server.create_mcp_server_for_workflow(app.folderpath)
    assert str(refusal.value).startswith(
        "Workflow 'library_wf' is not trained (missing model artifacts for context(s): "
        "Book, Library, Member, Record, global). Train it first with: fastworkflow train ")


# ---------------------------------------------------------------------------
# The planning match takes its candidates from the contract, byte for byte
# ---------------------------------------------------------------------------

def _old_job_types(workflow_folderpath: str, examples: int = workflow_agent.MATCHER_EXAMPLES,
                   scope: str | None = None) -> str:
    """The planning match's text builder as it was before it read the detector contract (reference copy)."""
    utterances = workflow_agent.planning_command_utterances(workflow_folderpath)
    lines = []
    for name, description, _ in workflow_agent.load_planning_commands(workflow_folderpath):
        if scope is not None and workflow_agent._planning_scope(name) != scope:
            continue
        shown = [" ".join(u.split()) for u in utterances.get(name, ())[:max(0, examples)]]
        shown = [u for u in shown if u]
        lines.append(f"- {name}: {description}" + (
            f" e.g. {', '.join(json.dumps(u, ensure_ascii=False) for u in shown)}" if shown else ""))
    return "\n".join(lines)


def _old_options(workflow_folderpath: str) -> dict[str, tuple]:
    """The planning match's option lists by scope, as they were built from the planning command files."""
    by_scope: dict[str, list[tuple[str, str]]] = {}
    for name, description, _ in workflow_agent.load_planning_commands(workflow_folderpath):
        by_scope.setdefault(workflow_agent._planning_scope(name), []).append((name, description))
    return {scope: tuple(commands) + (("none", "None of the job types fits this sub-task."),)
            for scope, commands in by_scope.items()}


def test_the_planning_match_text_is_byte_identical_for_every_scope(library_copy):
    contexts = list(fastworkflow.RoutingRegistry.get_definition(library_copy).contexts)
    scopes = [None, "", *contexts]
    assert {"Book", "Library", "Member", "Record", "*"} <= set(contexts)
    for scope in scopes:
        for examples in (workflow_agent.MATCHER_EXAMPLES, 0, 2):
            assert workflow_agent._matcher_job_types(library_copy, examples, scope) == \
                _old_job_types(library_copy, examples, scope), (scope, examples)
    assert workflow_agent._matcher_job_types(library_copy, scope="Library").count("\n") == 7      # eight job types
    assert workflow_agent._matcher_job_types(library_copy, scope="Member") == ""


def test_the_planning_bar_is_the_listings_and_the_veto_stays_deleted():
    # (was: the planning match's options and signature; the match itself is the intent loop's, bd fix-zvme)
    assert workflow_agent.PLANNING_COMMAND_MATCH_THRESHOLD == 0.5
    assert not hasattr(workflow_agent, "PLANNING_NONE_VETO_THRESHOLD")       # the veto was deleted (bd fix-o1lg)


def test_the_planning_candidates_are_the_contracts(library_copy):
    by_scope = intent_detector.planning_candidates_by_scope(library_copy)
    assert list(by_scope) == ["Book", "Library"]
    for scope, offered in by_scope.items():
        contract = [c for c in intent_detector.candidates(library_copy, scope) if c.kind == "planning"]
        assert offered == [c for c in contract if c.name.startswith(f"{scope}/")]
    assert [c.name for c in intent_detector.planning_candidates(library_copy)] == [
        name for name, _, _ in workflow_agent.load_planning_commands(library_copy)]


# ---------------------------------------------------------------------------
# The chain: the trained local model when it is confident, the model matcher otherwise, then the question
# ---------------------------------------------------------------------------

from fastworkflow.intent_detector import BARE_VALUE, ChainEngine  # noqa: E402
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL  # noqa: E402

HEAD_DETAILS = {"confident": True, "confidence": 0.97, "model_tier": "tiny", "top_label": "Book/show_book",
                "topk_labels": ["Book/show_book"], "topk_scores": [0.97]}
UNSURE = {**HEAD_DETAILS, "confident": False, "confidence": 0.41}


def _book_with_a_head(monkeypatch, cme, head_labels, head_details):
    """A Book head on disk that answers *head_labels* (the router behind the module's name)."""
    app = cme.context["app_workflow"]
    _fake_head(os.path.join(app.folderpath, "___command_info", "Book"))
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _router(head_labels, head_details))
    return CommandNamePrediction(cme)


def test_the_chain_is_chosen_for_a_scope_with_a_head_and_the_matcher_for_one_without(
        library_copy, matcher_model):
    matcher_model({"none": 1.0})
    assert intent_detector.engine_choice(library_copy, "Book") == "matcher"            # no local model
    assert intent_detector.engine_for(library_copy, "Book").fallback is True
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    assert intent_detector.engine_choice(library_copy, "Book") == "chain"
    engine = intent_detector.engine_for(library_copy, "Book", router_factory=_router(["Book/show_book"]))
    assert isinstance(engine, ChainEngine) and isinstance(engine, intent_detector.Engine)
    # gathering stays classifier only
    assert isinstance(intent_detector.engine_for(
        library_copy, "Book", router_factory=_router(["Book/show_book"]), allow_matcher=False), ClassifierEngine)


@pytest.mark.parametrize("head_labels, head_details, scores, kind, labels, engine, calls", [
    # a confident head decides: no model call
    (["Book/show_book"], HEAD_DETAILS, {"Book/list_loans": 0.9, "none": 0.1}, MATCH, ["Book/show_book"], "classifier", 0),
    ([PARAMETER_VALUE_LABEL], HEAD_DETAILS, {"Book/list_loans": 0.9, "none": 0.1},
     BARE_VALUE, [PARAMETER_VALUE_LABEL], "classifier", 0),
    # an unsure head (a list): the matcher is asked once and its answer stands
    (["Book/show_book", "Book/list_loans", WILDCARD_LABEL], UNSURE, {"Book/list_loans": 0.9, "none": 0.1},
     MATCH, ["Book/list_loans"], "matcher", 1),
    (["Book/show_book", "Book/list_loans"], UNSURE, {"Library/find_book": 0.9, "none": 0.1},
     NOT_MINE, [WILDCARD_LABEL], "matcher", 1),
    (["Book/show_book", "Book/list_loans"], UNSURE, {"Book/show_book": 0.4, "Book/list_loans": 0.35, "none": 0.25},
     AMBIGUOUS, ["Book/show_book", "Book/list_loans"], "matcher", 1),
    # one label the head is not confident of
    (["Book/show_book"], UNSURE, {"Book/list_loans": 0.9, "none": 0.1}, MATCH, ["Book/list_loans"], "matcher", 1),
    # a lone escalation label: the matcher is asked, and may keep the request here
    ([WILDCARD_LABEL], HEAD_DETAILS, {"Book/list_loans": 0.9, "none": 0.1}, MATCH, ["Book/list_loans"], "matcher", 1),
    ([WILDCARD_LABEL], HEAD_DETAILS, {"Library/find_book": 0.9, "none": 0.1}, NOT_MINE, [WILDCARD_LABEL], "matcher", 1),
])
def test_the_chain_asks_the_matcher_only_when_the_local_model_is_not_confident(
        library_copy, matcher_model, monkeypatch, head_labels, head_details, scores, kind, labels, engine, calls):
    model = matcher_model(scores)
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    chain = intent_detector.engine_for(
        library_copy, "Book", router_factory=_router(head_labels, head_details), scope_path=("Book", "Library"))

    answer = chain.decide("anything")

    assert (answer.kind, answer.labels, answer.engine) == (kind, labels, engine)
    assert len(model.calls) == calls and answer.fallback is False
    if engine == "matcher":
        assert answer.details["classifier"] == head_details        # the head's facts are kept beside the matcher's
        assert answer.details["scope_path"] == ["Book", "Library"]


@pytest.mark.parametrize("head_labels, kind", [
    (["Book/show_book", "Book/list_loans"], AMBIGUOUS),            # the local candidates go to the question
    ([WILDCARD_LABEL], NOT_MINE),                                  # the local "not here" stands: the walk climbs
])
def test_when_the_matchers_call_fails_the_local_answer_stands(library_copy, matcher_model, head_labels, kind):
    model = matcher_model(RuntimeError("the model is down"))
    _fake_head(os.path.join(library_copy, "___command_info", "Book"))
    chain = intent_detector.engine_for(
        library_copy, "Book", router_factory=_router(head_labels, UNSURE), scope_path=("Book", "Library"))

    answer = chain.decide("anything")

    assert (answer.kind, answer.labels, answer.engine) == (kind, head_labels, "classifier")
    assert answer.details["matcher_error"] == "RuntimeError: the model is down" and len(model.calls) == 1


def test_in_the_cme_step_a_confident_head_decides_with_no_model_call(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.9, "none": 0.1})
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book"], HEAD_DETAILS)
    trace: dict = {}

    output = predictor.predict("tell me about this book")

    trace = predictor.trace

    assert output.command == "Book/show_book" and model.calls == []
    assert trace["matcher_layer"] == "classifier" and trace["engine"] == "classifier"
    assert trace["engine_chain"] == ["classifier"] and "engine_mode" not in trace
    assert "matcher" not in trace and "jev_tiebreak" not in trace


def test_in_the_cme_step_an_unsure_head_hands_the_scope_to_the_matcher(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.92, "Book/show_book": 0.05, "none": 0.03})
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book", "Book/list_loans", WILDCARD_LABEL], UNSURE)
    trace: dict = {}

    output = predictor.predict("what is out on loan")

    trace = predictor.trace

    assert output.command == "Book/list_loans" and output.outcome != "ambiguous"
    assert len(model.calls) == 1 and model.calls[0][0].__doc__.strip() == \
        "Decide which command a message to a workflow asks for."          # the matcher's one call
    assert trace["matcher_layer"] == "matcher" and trace["engine"] == "matcher"
    assert trace["engine_fallback"] is False and trace["engine_chain"] == ["classifier", "matcher"]
    assert trace["classifier"] == UNSURE and trace["matcher"]["top_label"] == "Book/list_loans"
    assert "jev_tiebreak" not in trace and trace["candidate_count"] == 1
    assert command_identity_uncertainty(trace).signals_absent_reason == "not-instrumented"


def test_in_the_cme_walk_a_lone_escalation_label_asks_the_matcher_and_the_walk_costs_one_call(
        on_a_book, matcher_model, monkeypatch):
    app, cme, library, _ = on_a_book
    model = matcher_model({"Library/list_branches": 0.9, "none": 0.1})
    predictor = _book_with_a_head(monkeypatch, cme, [WILDCARD_LABEL], HEAD_DETAILS)       # every head: "not here"
    _fake_head(os.path.join(app.folderpath, "___command_info", "Library"))

    intent = predictor.predict(BRANCHES)

    # Book: the head escalates and the matcher agrees (its choice is the Library's). Library: the head escalates,
    # and the matcher, asked from the same grouped call, keeps the request there. One model call in all.
    assert intent.command == "Library/list_branches"
    assert intent.context == "Library" and intent.record is library and len(model.calls) == 1


def test_in_the_cme_step_the_matcher_may_keep_here_what_the_head_escalated(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    matcher_model({"Book/list_loans": 0.9, "none": 0.1})
    predictor = _book_with_a_head(monkeypatch, cme, [WILDCARD_LABEL], HEAD_DETAILS)
    output = predictor.predict("what is out on loan")
    trace = predictor.trace
    assert output.command == "Book/list_loans" and trace["escalation_outcome"] == "absent"


def test_in_the_cme_step_the_matchers_ambiguity_is_the_standard_question(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    matcher_model({"Book/show_book": 0.4, "Book/list_loans": 0.35, "none": 0.25})
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/list_loans", "Book/show_book", "Record/add_note"], UNSURE)
    output = predictor.predict("the book")
    trace = predictor.trace
    labels = ["Book/show_book", "Book/list_loans"]                 # the matcher's candidates, not the head's three
    assert list(output.ambiguous) == labels
    # Neither link was confident: the question decides, and the trace says which links were asked.
    assert trace["engine"] == "question" and trace["engine_chain"] == ["classifier", "matcher"]


def test_in_the_cme_step_a_failed_matcher_call_leaves_the_local_candidates_for_the_question(
        on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    model = matcher_model(RuntimeError("the model is down"))
    head = ["Book/list_loans", "Book/show_book", "Record/add_note"]
    predictor = _book_with_a_head(monkeypatch, cme, head, UNSURE)
    trace: dict = {}

    output = predictor.predict("the book")

    trace = predictor.trace

    assert output.outcome == "ambiguous" and list(output.ambiguous) == head and len(model.calls) == 1
    assert trace["engine"] == "question" and trace["engine_chain"] == ["classifier", "matcher"]
    assert trace["matcher_layer"] == "classifier" and trace["matcher"]["error"] == "RuntimeError: the model is down"


def test_in_the_cme_step_a_failed_matcher_call_leaves_a_local_not_here_to_the_walk(
        on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    matcher_model(RuntimeError("the model is down"))
    predictor = _book_with_a_head(monkeypatch, cme, [WILDCARD_LABEL], HEAD_DETAILS)
    output = predictor.predict("which branches are there")
    trace = predictor.trace
    assert output.command is None and output.outcome != "ambiguous"        # a decline: the walk climbs
    assert trace["engine"] == "classifier" and trace["escalation_outcome"] == "honored"


def test_without_the_planner_key_the_chain_is_the_local_model_then_the_question(
        on_a_book, no_planner_key, monkeypatch):
    app, cme, _, _ = on_a_book
    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book", "Book/list_loans"], UNSURE)

    assert intent_detector.engine_choice(app.folderpath, "Book") == "classifier"
    assert intent_detector.engine_choice(app.folderpath, "Library") is None
    assert not intent_detector.matcher_available()
    output = predictor.predict("the book")
    trace = predictor.trace
    assert output.outcome == "ambiguous" and list(output.ambiguous) == ["Book/show_book", "Book/list_loans"]      # the head's ambiguity
    assert trace["engine"] == "question" and trace["engine_chain"] == ["classifier"] and "matcher" not in trace
    assert intent_detector.startup_report(app.folderpath, []) == ([], [NO_KEY_LINE])
    assert intent_detector.startup_report(app.folderpath, ["Member"])[0]                # untrained: still refused


def test_the_startup_line_says_what_the_chain_is(library_copy, matcher_model, monkeypatch):
    matcher_model({"none": 1.0})
    assert intent_detector.startup_report(library_copy, []) == ([], [CHAIN_LINE])
    assert intent_detector.startup_report(library_copy, ["Book"]) == ([], [_untrained("Book")])
    monkeypatch.setenv("FW_REQUIRE_HEADS", "1")                    # a missing local model is a failure
    assert intent_detector.startup_report(library_copy, []) == ([], [CHAIN_LINE])
    problems, notices = intent_detector.startup_report(library_copy, ["Book"])
    assert notices == [] and len(problems) == 1


@pytest.mark.parametrize("setting", ["auto", "classifier"])
def test_the_removed_setting_does_not_keep_an_unsure_head_from_the_matcher(
        on_a_book, matcher_model, monkeypatch, setting):
    """Under the old ``auto`` and ``classifier`` settings an unsure head never reached the matcher. The setting
    is ignored now: the chain is the only behaviour."""
    app, cme, _, _ = on_a_book
    model = matcher_model({"Book/list_loans": 0.9, "none": 0.1})
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book", "Book/list_loans"], UNSURE)
    _setting(monkeypatch, setting)
    output = predictor.predict("the book")
    trace = predictor.trace
    assert output.command == "Book/list_loans" and len(model.calls) == 1
    assert trace["engine"] == "matcher" and trace["engine_chain"] == ["classifier", "matcher"]
    assert "engine_mode" not in trace


# ---------------------------------------------------------------------------
# The deterministic path: a confident local answer naming a planning command is never acted on unreviewed
# (the same rule as resolution._second_opinion; exact and fuzzy NAME matches are out of scope and still run)
# ---------------------------------------------------------------------------

PLAN = "Book/loan_report"
PLAN_HEAD = {**HEAD_DETAILS, "top_label": PLAN, "topk_labels": [PLAN, "Book/show_book", WILDCARD_LABEL],
             "topk_scores": [0.97, 0.02, 0.01]}
LOAN_TEXT = "what is out on loan"


@pytest.mark.usefixtures("no_planner_key")
def test_with_no_remote_model_a_confident_local_plan_is_asked_not_run(on_a_book, monkeypatch):
    app, cme, _, _ = on_a_book
    predictor = _book_with_a_head(monkeypatch, cme, [PLAN], PLAN_HEAD)
    trace: dict = {}

    output = predictor.predict(LOAN_TEXT)

    trace = predictor.trace

    assert output.command is None and output.ambiguous[0] == PLAN and "Book/show_book" in output.ambiguous
    assert trace["engine"] == "question" and trace["candidates"][0] == PLAN


def test_a_remote_confirmation_lets_a_confident_local_plan_run(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    model = matcher_model({PLAN: 0.95, "Book/show_book": 0.03, "none": 0.02})
    predictor = _book_with_a_head(monkeypatch, cme, [PLAN], PLAN_HEAD)
    trace: dict = {}

    output = predictor.predict(LOAN_TEXT)

    trace = predictor.trace

    assert output.command == PLAN and len(model.calls) == 1
    assert trace["engine"] == "matcher" and trace["engine_chain"] == ["classifier", "matcher"]


def test_a_remote_answer_below_the_accept_bar_does_not_run_the_plan(on_a_book, matcher_model, monkeypatch):
    app, cme, _, _ = on_a_book
    matcher_model({PLAN: 0.8, "Book/show_book": 0.15, "none": 0.05})
    predictor = _book_with_a_head(monkeypatch, cme, [PLAN], PLAN_HEAD)

    output = predictor.predict(LOAN_TEXT)

    assert output.command is None and output.ambiguous[0] == PLAN


def test_a_failed_remote_check_of_a_confident_local_plan_is_a_question(
        on_a_book, matcher_model, monkeypatch, fresh_outage_counter):
    app, cme, _, _ = on_a_book
    matcher_model(_CreditsExhausted("out of credits"))
    predictor = _book_with_a_head(monkeypatch, cme, [PLAN], PLAN_HEAD)
    trace: dict = {}

    output = predictor.predict(LOAN_TEXT)

    trace = predictor.trace

    assert output.command is None and output.ambiguous[0] == PLAN
    assert trace["remote_failed"] == {"error_class": "_CreditsExhausted", "status_code": 402}


@pytest.mark.usefixtures("no_planner_key")
def test_a_confident_local_execution_command_still_runs_alone_on_the_deterministic_path(on_a_book, monkeypatch):
    app, cme, _, _ = on_a_book
    predictor = _book_with_a_head(monkeypatch, cme, ["Book/show_book"], HEAD_DETAILS)

    output = predictor.predict(LOAN_TEXT)

    assert output.command == "Book/show_book"


@pytest.mark.usefixtures("no_planner_key")
def test_an_exact_planning_command_name_still_runs_with_no_review(on_a_book, monkeypatch):
    app, cme, _, _ = on_a_book
    predictor = _book_with_a_head(monkeypatch, cme, [PLAN], PLAN_HEAD)

    output = predictor.predict("loan_report")

    assert output.command == PLAN
