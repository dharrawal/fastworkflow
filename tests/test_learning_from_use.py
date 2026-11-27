"""Learning from use (bd fix-o5wi.6): what a workflow records about the requests it resolves, which rows teach, the
refit of the embedding head behind its gate, the lower-only calibration of the commit threshold, the reload of a
published head without a restart, the background refit (on by default) and the review commands.

Real components throughout. The trained pieces run on temporary COPIES of the library example (the example's own
generated-utterance cache is read back, so no model is called, as in tests/test_intent_model_layout_training.py);
nothing under ``fastworkflow/examples`` or ``tests`` is trained into or changed. The tests that need a trained head
skip, saying why, when the example's cache or the encoder is not on this machine. A test that asserts what happens
with no remote model uses ``no_planner_key``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.intent import intent_detector
from fastworkflow.intent import intent_learning as learning
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import embedding_head as eh
from fastworkflow.train import head_refit
from fastworkflow.train import intent_model_layout as layout
from tests.test_intent_model_layout_training import (
    LIBRARY,
    _copy_library,
    _forget_loaded_routers,
    _rekey_parameter_examples,
    _session_on_a_book,
    _train,
    _training_env,
)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

BENCHMARKS = Path(__file__).parent / "fixtures" / "intent_benchmarks"


def _benchmark(name: str) -> list[dict]:
    return [case for case in json.loads((BENCHMARKS / f"{name}.json").read_text())["cases"]
            if case.get("kind", "routing") == "routing"]


# ======================================================================================================================
# The store, the settings and the selection of teaching rows (no head needed)
# ======================================================================================================================

@pytest.fixture
def folder(tmp_path, initialized_fastworkflow):  # noqa: F811
    path = tmp_path / "workflow"
    path.mkdir()
    fastworkflow.init(env_vars={})
    return str(path)


def _write(folder, text, command, how=learning.HOW_USER_CHOICE, context="Book", **more):
    return learning.record(folder, text=text, context=context, command=command, how=how, **more)


def test_a_row_carries_how_it_was_settled_and_whether_somebody_confirmed_it(folder):
    evidence = {"head": {"version": "v1", "top": "Book/show_book", "p_in": 0.41, "p_out": 0.05, "confident": False,
                         "threshold": 0.65, "threshold_source": "trained"},
                "remote": {"label": "Book/list_loans", "score": 0.3}}
    assert _write(folder, "Show me the book", "Book/show_book", learning.HOW_LOCAL, evidence=evidence)
    assert _write(folder, "which loans", "Book/list_loans", learning.HOW_REVIEW_SWITCH,
                  rejected_label="Book/show_book")
    first, second = learning.rows(folder)
    assert (first["how"], first["confirmed"], first["norm"]) == ("local", 0, "show me the book")
    assert (first["head_version"], first["head_top"], first["p_in"], first["head_confident"], first["threshold"]) == (
        "v1", "Book/show_book", 0.41, 0, 0.65)
    assert (first["remote_label"], first["remote_score"]) == ("Book/list_loans", 0.3)
    assert (second["confirmed"], second["rejected_label"]) == (1, "Book/show_book")
    assert os.path.isfile(os.path.join(folder, "___convo_info", "intent_learning.sqlite3"))


def test_recording_can_be_turned_off_and_then_writes_nothing(folder):
    fastworkflow.init(env_vars={learning.RECORD_SETTING: "off"})
    assert _write(folder, "show the book", "Book/show_book") is False
    assert learning.rows(folder) == []
    assert not os.path.exists(learning.store_path(folder))
    fastworkflow.init(env_vars={})
    assert _write(folder, "show the book", "Book/show_book") is True


def test_a_store_that_cannot_be_written_never_raises(folder):
    os.makedirs(learning.store_path(folder))          # a directory where the database file should be
    assert _write(folder, "show the book", "Book/show_book") is False


def test_only_confirmed_rows_teach_never_a_name_a_bare_value_or_a_vague_reply(folder):
    _write(folder, "show me the book", "Book/show_book")
    _write(folder, "list the loans", "Book/list_loans", learning.HOW_LOCAL)              # unconfirmed
    _write(folder, "show_book", "Book/show_book", learning.HOW_NAME)                      # a name match
    _write(folder, "12345", "Book/show_book")                                             # a bare value
    _write(folder, "Ask me again later", "Book/show_book")                                     # a built-in vague reply
    assert [(row.norm, row.command) for row in learning.teaching_rows(folder)] == [
        ("show me the book", "Book/show_book")]


def test_conflicting_confirmations_are_excluded_unless_one_leads_by_two(folder):
    for _ in range(2):
        _write(folder, "what is on it", "Book/show_book")
    _write(folder, "What is on it?", "Book/list_loans")                  # same normalised text: a conflict, lead of 1
    _write(folder, "tell me more", "Book/show_book")
    for _ in range(3):
        _write(folder, "tell me more", "Book/list_loans")                # leads by 2
    rows = {row.norm: row for row in learning.teaching_rows(folder)}
    assert "what is on it" not in rows
    assert (rows["tell me more"].command, rows["tell me more"].confirmations) == ("Book/list_loans", 3)


def test_a_row_counts_after_the_minimum_confirmations(folder):
    _write(folder, "show me the book", "Book/show_book")
    _write(folder, "list the loans", "Book/list_loans")
    _write(folder, "list the loans", "Book/list_loans")
    fastworkflow.init(env_vars={learning.MIN_CONFIRMATIONS_SETTING: "2"})
    assert [row.norm for row in learning.teaching_rows(folder)] == ["list the loans"]
    fastworkflow.init(env_vars={})
    assert len(learning.teaching_rows(folder)) == 2


def test_a_command_the_head_does_not_have_does_not_teach(folder):
    _write(folder, "show me the book", "Book/show_book")
    _write(folder, "do the thing", "Gone/command")
    assert [row.command for row in learning.teaching_rows(folder, labels=["Book/show_book"])] == ["Book/show_book"]


# ======================================================================================================================
# The calibration: lower only
# ======================================================================================================================

def _band_rows(folder, count, p_in, right=True, version="v1"):
    for index in range(count):
        top = "Book/show_book"
        _write(folder, f"request number {p_in} {right} {index}", top if right else "Book/list_loans",
               evidence={"head": {"version": version, "top": top, "p_in": p_in, "p_out": 0.01, "confident": False,
                                  "threshold": 0.65}})


def _lowered(folder):
    return learning.calibrate(folder, base_threshold=0.65, lineage=["v1"])


def test_the_calibration_lowers_to_the_smallest_candidate_with_enough_evidence(folder):
    _band_rows(folder, 100, 0.62)                                 # in [0.60, 0.65), all right
    _band_rows(folder, 100, 0.20)                                 # confirmed rows below every band
    decision = _lowered(folder)
    assert decision["threshold"] == 0.50 and decision["confirmed"] == 200
    assert [entry["candidate"] for entry in decision["evidence"]] == [0.50, 0.55, 0.60]
    # Three wrong answers 0.50 would newly give (3 of 103 is over 2% of its band) keep it at the next candidate.
    _band_rows(folder, 3, 0.52, right=False)
    decision = _lowered(folder)
    assert decision["threshold"] == 0.55
    first = decision["evidence"][0]
    assert (first["band_rows"], first["band_wrong"]) == (103, 3) and round(first["wrong_share_of_band"], 3) == 0.029


def test_the_calibration_needs_200_confirmed_100_in_the_band_and_a_wrong_share_of_at_most_two_percent_of_the_band(folder):
    _band_rows(folder, 100, 0.62)
    assert _lowered(folder)["threshold"] is None                  # 100 confirmed only
    _band_rows(folder, 100, 0.20)
    assert _lowered(folder)["threshold"] == 0.50
    _band_rows(folder, 3, 0.62, right=False)                      # 3 wrong of a band of 103 is over 2%
    assert _lowered(folder)["threshold"] is None
    # Rows of a head outside the lineage do not count.
    assert learning.calibrate(folder, base_threshold=0.65, lineage=["other"])["confirmed"] == 0


def test_two_wrong_of_a_thirty_row_band_is_not_one_percent_of_anything(folder):
    """The wrong share is the band's own: 2 of 30 (6.7%) used to pass as 1% of 200 confirmed rows."""
    _band_rows(folder, 28, 0.62)
    _band_rows(folder, 2, 0.62, right=False)
    _band_rows(folder, 170, 0.20)
    decision = _lowered(folder)
    assert decision["confirmed"] == 200 and decision["threshold"] is None
    assert decision["evidence"][-1]["band_rows"] == 30 and decision["evidence"][-1]["band_wrong"] == 2


def test_a_band_of_100_may_hold_two_wrong_but_not_three(folder):
    _band_rows(folder, 98, 0.62)
    _band_rows(folder, 2, 0.62, right=False)
    _band_rows(folder, 100, 0.20)
    assert _lowered(folder)["threshold"] == 0.50
    _band_rows(folder, 1, 0.62, right=False)                      # 3 of 101
    assert _lowered(folder)["threshold"] is None


def test_a_band_of_99_is_too_small_even_when_clean(folder):
    _band_rows(folder, 99, 0.62)
    _band_rows(folder, 101, 0.20)
    decision = _lowered(folder)
    assert decision["threshold"] is None and decision["evidence"][-1]["band_rows"] == 99


def test_a_clean_higher_band_does_not_carry_a_dirty_lower_one(folder):
    _band_rows(folder, 100, 0.62)                                 # 0.60 band: clean
    _band_rows(folder, 100, 0.57)                                 # adds to the 0.55 band with 5 wrong rows
    _band_rows(folder, 5, 0.57, right=False)
    decision = _lowered(folder)
    by_candidate = {entry["candidate"]: entry for entry in decision["evidence"]}
    assert by_candidate[0.60]["enough"] and not by_candidate[0.55]["enough"] and not by_candidate[0.50]["enough"]
    assert decision["threshold"] == 0.60


def test_the_calibration_never_raises_the_threshold(folder):
    _band_rows(folder, 250, 0.52)
    decision = learning.calibrate(folder, base_threshold=0.50, lineage=["v1"])
    assert decision["threshold"] is None and decision["evidence"] == []     # no candidate is below 0.50
    lowered = learning.calibrate(folder, base_threshold=0.65, lineage=["v1"])
    assert lowered["threshold"] == 0.50


# ======================================================================================================================
# A trained workflow: the row store, recording from the CME path, the refit, reload, background, CLI
# ======================================================================================================================

@pytest.fixture(scope="module")
def trained_library(tmp_path_factory):
    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets package not installed; intent-detection training is skipped.")
    cache_root = LIBRARY / "___command_info" / "utterance_cache"
    if not cache_root.is_dir() or not any(cache_root.glob("*.json")):
        pytest.skip(f"the library example has no generated-utterance cache at {cache_root}, so training a copy "
                    "would call a model")
    from fastworkflow.train import workflow_model

    base = tmp_path_factory.mktemp("learning") / "library"
    path = _copy_library(base)
    fastworkflow.init(env_vars=_training_env(path, None))
    try:
        eh.SentenceEncoder.get(layout.DEFAULT_ENCODER)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"the default sentence encoder cannot be loaded here ({type(error).__name__}: {error})")
    _rekey_parameter_examples(path)
    try:
        _train(path, None)
    except workflow_model.ModelCallAttemptedError as attempted:
        pytest.skip(f"the copied caches are not warm for this checkout: {attempted}")
    RoutingRegistry.clear_registry()
    return path


@pytest.fixture
def library(trained_library, tmp_path, initialized_fastworkflow, monkeypatch, no_planner_key):  # noqa: F811
    """A fresh copy of the trained library per test: versions, store and routers are the test's own."""
    path = str(tmp_path / "library")
    shutil.copytree(trained_library, path, symlinks=True)
    # The tests refit in the foreground and count the rows themselves: the default background refit would race them.
    fastworkflow.init(env_vars={**_training_env(path, None), learning.REFIT_SETTING: "off"})
    RoutingRegistry.clear_registry()
    _forget_loaded_routers(path)
    monkeypatch.setattr(intent_detector, "_last_head_check", {})
    from fastworkflow.train import workflow_model

    with workflow_model.no_model_calls():          # a model call (a remote one) raises here instead of being made
        yield path
    _forget_loaded_routers(path)
    RoutingRegistry.clear_registry()


def _current_model_dir(path) -> Path:
    return av.version_dir(path, av.resolve_current_version(path)) / layout.WORKFLOW_MODEL_DIRNAME


def _confirm_cases(path, cases, limit=None):
    version = av.resolve_current_version(path)
    count = 0
    for case in cases[:limit]:
        evidence = {"head": {"version": version, "top": case["expected_label"], "p_in": 0.5, "p_out": 0.1,
                             "confident": False, "threshold": 0.65}}
        count += int(learning.record(path, text=case["utterance"], context=case["context"],
                                     command=case["expected_label"], how=learning.HOW_USER_CHOICE,
                                     evidence=evidence))
    return count


def test_the_train_stores_the_vectors_it_computed_beside_the_head(trained_library):
    model_dir = _current_model_dir(trained_library)
    store = eh.read_row_store(str(model_dir))
    scopes = json.loads((model_dir / "scopes.json").read_text())
    assert len(store["texts"]) == scopes["rows"]["total"] == store["vectors"].shape[0]
    assert store["vectors"].shape[1] == scopes["encoder"]["dimensions"]
    assert scopes["rows"]["row_store_bytes"] == (model_dir / "rows.npz").stat().st_size
    assert set(store["kinds"].tolist()) <= {0, 1, 2} and 1 in store["kinds"].tolist()
    assert (model_dir / "rows.npz").stat().st_size < 3_000_000          # about 1 MB for the library example


def test_a_confident_local_answer_on_the_cme_path_is_recorded_unconfirmed_with_the_heads_facts(library):
    from tests.test_intent_model_layout_training import OWN_CANDIDATES, _confident, _route

    app, cme = _session_on_a_book(library, "learning-local")
    try:
        hits = _confident(cme, OWN_CANDIDATES, "Book")
        assert hits, "no probed Book request is confident"
        phrasing, command = max(hits, key=lambda hit: len(hit[0]))      # not close to a command name
        assert _route(cme, phrasing).command_response.artifacts["command_name"] == command
        _route(cme, "show_book")                                                              # a typed name
    finally:
        app.close()
    local = [row for row in learning.rows(library) if row["how"] == "local"]
    assert local and local[0]["text"] == phrasing and local[0]["command"] == command
    assert (local[0]["confirmed"], local[0]["head_confident"]) == (0, 1)
    assert local[0]["head_version"] == av.resolve_current_version(library)
    assert local[0]["p_in"] >= local[0]["threshold"] == 0.65 and local[0]["head_top"] == command
    named = [row for row in learning.rows(library) if row["how"] == "name"]
    assert named and named[0]["command"].endswith("show_book") and named[0]["confirmed"] == 0
    assert learning.teaching_rows(library) == []                    # an unconfirmed answer and a name teach nothing


def test_a_pick_at_the_which_command_question_is_a_confirmed_row_and_a_misunderstanding_corrects_one(library):
    from tests.test_intent_model_layout_training import _route

    app, cme = _session_on_a_book(library, "learning-pick")
    try:
        vague = next((text for text in ("show me the report", "give me the details", "what about this one",
                                        "tell me about it", "report") if "ambiguous" in
                      _route(cme, text).command_response.response.lower()), None)
        assert vague, "none of the probed requests is an ambiguity for the head"
        # The ambiguity question is out; its reply (a command name) is the pick.
        out = _route(cme, "show_book")
        assert out.command_response.artifacts.get("command_name", "").endswith("show_book")
    finally:
        app.close()
    picks = [row for row in learning.rows(library) if row["how"] == "user_choice"]
    assert picks and (picks[0]["confirmed"], picks[0]["text"]) == (1, vague)
    assert picks[0]["command"].endswith("show_book")
    assert picks[0]["p_in"] is not None and picks[0]["head_confident"] == 0       # the head's facts at the question
    assert [row.norm for row in learning.teaching_rows(library)] == [learning.normalise(vague)]


def test_a_misunderstood_answer_followed_by_a_choice_keeps_the_first_answer_as_the_rejected_label(library):
    from tests.test_intent_model_layout_training import OWN_CANDIDATES, _confident, _route

    app, cme = _session_on_a_book(library, "learning-misunderstood")
    try:
        hits = [hit for hit in _confident(cme, OWN_CANDIDATES, "Book") if hit[1] == "Book/show_book"]
        assert hits
        phrasing, first = max(hits, key=lambda hit: len(hit[0]))
        _route(cme, phrasing)
        _route(cme, "you_misunderstood")
        _route(cme, "shelf_report")
    finally:
        app.close()
    corrected = [row for row in learning.rows(library) if row["rejected_label"]]
    assert corrected and corrected[-1]["rejected_label"] == first and corrected[-1]["confirmed"] == 1
    assert corrected[-1]["text"] == phrasing and corrected[-1]["command"] == "Book/shelf_report"
    assert corrected[-1]["how"] == "user_choice"
    assert corrected[-1]["head_top"] == first                       # the first answer's facts are kept with the row


def test_a_turn_survives_a_store_that_cannot_be_written(library):
    from tests.test_intent_model_layout_training import OWN_CANDIDATES, _confident, _route

    os.makedirs(learning.store_path(library), exist_ok=True)
    shutil.rmtree(learning.store_path(library))
    os.makedirs(learning.store_path(library))                        # a directory where the database should be
    app, cme = _session_on_a_book(library, "learning-broken-store")
    try:
        hits = _confident(cme, OWN_CANDIDATES, "Book")
        assert hits
        longest = max(hits, key=lambda hit: len(hit[0]))
        assert _route(cme, longest[0]).command_response.artifacts["command_name"] == longest[1]
    finally:
        app.close()


def test_a_refit_adds_the_learned_rows_embeds_only_their_texts_and_publishes_a_new_version(library, monkeypatch):
    base_version = av.resolve_current_version(library)
    embedded: list[list[str]] = []
    original = eh.SentenceEncoder.embed

    def counting(self, texts, *args, **kwargs):
        embedded.append(list(texts))
        return original(self, texts, *args, **kwargs)

    monkeypatch.setattr(eh.SentenceEncoder, "embed", counting)
    cases = _benchmark("library_v2")
    assert _confirm_cases(library, cases, 60) == 60

    report = head_refit.refit(library)

    assert report.published and report.base_version == base_version and report.new_version != base_version
    assert av.resolve_current_version(library) == report.new_version
    assert report.learned_rows == 60 and report.gate["passed"]
    # Only the learned texts were embedded: never a base row.
    assert sum(len(texts) for texts in embedded) == 60 == report.embedded_texts
    manifest = av.read_manifest(library, report.new_version)
    assert (manifest["refit_of"], manifest["learned_rows"]) == (base_version, 60)
    assert manifest["lineage"] == [base_version, report.new_version] and manifest["refit_gate"]["passed"]
    scopes = json.loads((_current_model_dir(library) / "scopes.json").read_text())
    assert scopes["rows"]["learned"] == 60 and scopes["refit"]["of"] == base_version
    assert scopes["threshold"] == 0.65 and scopes["labels"] == json.loads(
        (av.version_dir(library, base_version) / "workflow" / "scopes.json").read_text())["labels"]
    # The same rows again is nothing new; more rows embed only the new texts.
    assert not head_refit.refit(library).published
    embedded.clear()
    assert _confirm_cases(library, cases[60:80]) == 20
    second = head_refit.refit(library)
    assert second.published, second.reason
    assert sum(len(texts) for texts in embedded) == 20
    # Retention keeps the current version, the previous one and the full train the lineage descends from.
    assert {info.version_id for info in av.list_versions(library)} == {base_version, report.new_version,
                                                                       second.new_version}
    assert _confirm_cases(library, cases[80:90]) == 10
    third = head_refit.refit(library)
    assert third.published, third.reason
    assert base_version in {info.version_id for info in av.list_versions(library)}
    assert report.new_version not in {info.version_id for info in av.list_versions(library)}


def test_a_refit_that_fails_its_gate_is_not_published_and_says_why(library):
    base_version = av.resolve_current_version(library)
    model_dir = _current_model_dir(library)
    store = eh.read_row_store(str(model_dir))
    labels = json.loads((model_dir / "scopes.json").read_text())["labels"]
    # Deliberately wrong rows: the seed phrasings of every Library command but one, confirmed as that one. They
    # contradict the base rows outright (cross-fit accuracy on the base rows falls by over 20 points). Seeds with a
    # word added ("... thanks", 32 to 104 rows) cost only 0.6 to 1.2 points, which the 1.5 point tolerance now lets
    # through, as the held-out learned rows then agree with each other.
    wrong = "Library/count_books"
    index = 0
    library_commands = sorted({labels[target] for kind, target in zip(store["kinds"], store["targets"])
                               if kind == 1 and labels[target].startswith("Library/")} - {wrong})
    assert len(library_commands) >= 6
    for command in library_commands:
        seeds = [text for text, kind, target in zip(store["texts"], store["kinds"], store["targets"])
                 if kind == 1 and labels[target] == command]
        for text in seeds:
            learning.record(library, text=text, context="Library", command=wrong,
                            how=learning.HOW_USER_CHOICE)
            index += 1
    assert index >= 20
    report = head_refit.refit(library)

    assert not report.published and "gate failed" in report.reason
    assert report.gate["passed"] is False
    assert av.resolve_current_version(library) == base_version
    assert [info.version_id for info in av.list_versions(library)] == [base_version]
    # The attempt counts for the background trigger: the next one waits for new rows, it does not retry at once.
    assert learning.new_teaching_count(library) == 0


def test_a_head_without_a_row_store_is_not_refittable_until_it_is_trained_again(library):
    (_current_model_dir(library) / "rows.npz").unlink()
    _confirm_cases(library, _benchmark("library_v2"), 20)
    report = head_refit.refit(library)
    assert not report.published and "cannot be refitted until the workflow is trained again" in report.reason


def test_too_few_teaching_rows_are_not_refitted(library):
    _confirm_cases(library, _benchmark("library_v2"), 3)
    assert "distinct texts are needed" in head_refit.refit(library).reason


def test_a_calibrated_threshold_is_in_force_unless_the_setting_names_one(library):
    model_dir = str(_current_model_dir(library))
    version = av.resolve_current_version(library)
    from fastworkflow.intent.router import CommandRouter

    context = os.path.join(library, "___command_info", "Book")
    assert CommandRouter(context)._workflow_router.threshold == 0.65
    learning.store_calibration(library, {"threshold": 0.55, "base": 0.65, "confirmed": 240, "evidence": [],
                                        "needs": learning._calibration_needs()}, [version])
    assert layout.calibrated_commit_threshold(library) == 0.55
    assert layout.threshold_in_force(model_dir, 0.65) == (0.55, "calibrated")
    _forget_loaded_routers(library)
    router = CommandRouter(context)._workflow_router
    assert (router.threshold, router.threshold_source) == (0.55, "calibrated")
    _labels, details = router.predict_with_details("show me this book")
    assert details["ambiguous_threshold"] == 0.55 and details["threshold_source"] == "calibrated"
    assert details["head_version"] == version
    notices = intent_detector.startup_report(library, [])[1]
    assert any("calibrated from 240 confirmed requests (trained 0.65)" in line for line in notices)
    # An explicit setting always wins, and the calibration never raises a threshold.
    fastworkflow.init(env_vars={**_training_env(library, None), layout.COMMIT_THRESHOLD_SETTING: "0.65"})
    assert layout.threshold_in_force(model_dir, 0.65) == (0.65, "setting")
    fastworkflow.init(env_vars=_training_env(library, None))
    assert layout.threshold_in_force(model_dir, 0.50) == (0.50, "trained")
    # A calibration made for another lineage is not read.
    learning.store_calibration(library, {"threshold": 0.55, "base": 0.65, "confirmed": 240, "evidence": [],
                                        "needs": learning._calibration_needs()}, ["gone"])
    assert layout.calibrated_commit_threshold(library) is None
    # Nor is one judged by another evidence rule (the earlier 30-row band and 1% of all confirmed rows).
    learning.store_calibration(library, {"threshold": 0.55, "base": 0.65, "confirmed": 240, "evidence": [],
                                        "needs": {"confirmed": 200, "band": 30, "max_wrong_share": 0.01}}, [version])
    assert layout.calibrated_commit_threshold(library) is None


def test_a_published_head_is_picked_up_without_a_restart_and_a_turn_in_flight_keeps_its_router(library):
    folder_of_book = os.path.join(library, "___command_info", "Book")
    before = intent_detector._command_router(folder_of_book)
    old_version = before._workflow_router.head_version
    _confirm_cases(library, _benchmark("library_v2"), 40)
    report = head_refit.refit(library)
    assert report.published

    intent_detector._last_head_check.clear()                          # the check is rate limited; a new process has none
    after = intent_detector._command_router(folder_of_book)

    assert after is not before and after._workflow_router.head_version == report.new_version
    assert before._workflow_router.head_version == old_version        # the router a turn started with is untouched
    assert intent_detector._command_router(folder_of_book) is after   # and the new one is cached
    labels, _details = after.predict_with_details("show me this book")
    assert labels


def test_the_other_layouts_are_not_reloaded(library):
    """A router that is not an embedding head is left where it is: nothing for the reload to look at."""
    from fastworkflow.intent.router import CommandRouter

    context = os.path.join(library, "___command_info", "Book")
    router = CommandRouter(context)
    router._workflow_router = SimpleNamespace(head_version=None)      # the shape of a router with no embedding head
    intent_detector._last_head_check.clear()
    assert intent_detector._command_router(context) is router


def test_the_background_refit_starts_after_enough_rows_never_twice_at_once_and_only_when_asked(library, monkeypatch):
    cases = _benchmark("library_v2")
    base_version = av.resolve_current_version(library)
    # Off when asked: rows arrive, nothing is refitted (the default is background: tests/test_learning_live_refit.py).
    assert learning.refit_mode() == learning.REFIT_OFF            # the fixture turns it off
    fastworkflow.init(env_vars=_training_env(library, None))
    assert learning.refit_mode() == learning.REFIT_BACKGROUND     # the default
    fastworkflow.init(env_vars={**_training_env(library, None), learning.REFIT_SETTING: "off",
                                learning.REFIT_EVERY_SETTING: "5"})
    _confirm_cases(library, cases, 8)
    learning.wait_for_background(library)
    assert av.resolve_current_version(library) == base_version

    env = {**_training_env(library, None), learning.REFIT_SETTING: "background", learning.REFIT_EVERY_SETTING: "5"}
    fastworkflow.init(env_vars=env)
    # A refit is running somewhere else (the file lock is held): the trigger starts a thread that does not refit.
    with head_refit.refit_lock(library) as acquired:
        assert acquired
        _confirm_cases(library, cases[8:], 6)
        learning.wait_for_background(library)
        assert av.resolve_current_version(library) == base_version
    # The lock is free: the next row triggers one refit, on a daemon thread.
    _confirm_cases(library, cases[14:], 1)
    learning.wait_for_background(library)
    published = av.resolve_current_version(library)
    assert published != base_version
    assert av.read_manifest(library, published)["refit_of"] == base_version
    thread = learning._background[os.path.realpath(library)]
    assert thread.daemon and not thread.is_alive()
    # Fewer than the interval since the refit: nothing more.
    _confirm_cases(library, cases[15:], 2)
    learning.wait_for_background(library)
    assert av.resolve_current_version(library) == published


def test_learn_runs_the_refit_and_the_calibration_and_prints_a_short_report(library, capsys):
    _confirm_cases(library, _benchmark("library_v2"), 40)
    result = head_refit.learn(library)
    text = head_refit.format_report(result)
    assert result["refit"].published and "Commit threshold stays 0.65" in text
    assert text.startswith("Recorded requests: 40 (40 confirmed") and "refit published as" in text
    stored = learning.get_meta(library, "calibration")
    assert stored["threshold"] is None and stored["lineage_root"] == result["refit"].base_version


def test_the_review_commands_list_remove_and_count_the_rows(folder, capsys, monkeypatch):
    from fastworkflow import cli

    _write(folder, "show me the book", "Book/show_book")
    _write(folder, "list the loans", "Book/list_loans", learning.HOW_LOCAL)
    monkeypatch.setattr(sys, "argv", ["fastworkflow", "learned", "list", folder, "--confirmed"])
    cli.main()
    listing = capsys.readouterr().out
    assert "show me the book" in listing and "list the loans" not in listing and "confirmed" in listing
    first_id = learning.rows(folder)[0]["id"]
    monkeypatch.setattr(sys, "argv", ["fastworkflow", "learned", "remove", folder, str(first_id)])
    cli.main()
    assert "Removed 1 of 1" in capsys.readouterr().out
    assert [row["text"] for row in learning.rows(folder)] == ["list the loans"]
    monkeypatch.setattr(sys, "argv", ["fastworkflow", "learned", "stats", folder])
    cli.main()
    assert "rows: 1" in capsys.readouterr().out
    cli._learned_main(Namespace(action="stats", workflow_folderpath=folder))


# ======================================================================================================================
# The resolution path
# ======================================================================================================================

def _plan_of(*pairs):
    """A plan as `_record_planned_parts` reads it: items with a goal and a means name."""
    return SimpleNamespace(items=[SimpleNamespace(goal=goal, means=SimpleNamespace(name=name)) for goal, name in pairs])


def test_a_reviewed_part_is_a_confirmed_accept_or_switch_and_an_unreviewed_one_is_the_links_answer(
        initialized_fastworkflow, tmp_path):  # noqa: F811
    from fastworkflow.engine.workflow_execution_context import WorkflowExecutionContext

    fastworkflow.init(env_vars={})
    wf = tmp_path / "wf"
    wf.mkdir()
    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._app_workflow = SimpleNamespace(folderpath=str(wf), _current_command_context_name="Book")
    facts = {"a": {"how": "local", "evidence": {"head": {"top": "Book/x", "p_in": 0.7}, "context": "Book"},
                   "drafted": "Book/x"},
             "b": {"how": "remote", "evidence": {"remote": {"label": "Book/y", "score": 0.95}}, "drafted": "Book/y"},
             "c": {"how": "user_choice", "evidence": {}, "drafted": "Book/z"}}

    ctx._turn_part_facts = dict(facts)
    ctx._record_planned_parts(_plan_of(("a", "Book/x"), ("b", "Book/other"), ("c", "Book/z")), reviewed=True)
    accept, switch, chosen = learning.rows(str(wf))
    assert (accept["how"], accept["confirmed"], accept["rejected_label"]) == ("review_accept", 1, None)
    assert (switch["how"], switch["command"], switch["rejected_label"]) == ("review_switch", "Book/other", "Book/y")
    assert (chosen["how"], chosen["confirmed"]) == ("user_choice", 1)
    assert accept["head_top"] == "Book/x" and switch["remote_label"] == "Book/y"

    ctx._turn_part_facts = dict(facts)
    ctx._record_planned_parts(_plan_of(("a", "Book/x"), ("b", "Book/y"), ("edited text", "Book/q")), reviewed=False)
    later = learning.rows(str(wf))[3:]
    assert [(row["how"], row["confirmed"]) for row in later] == [("local", 0), ("remote", 0)]   # an edited text is skipped
    ctx._record_planned_parts(_plan_of(("a", "Book/x")), reviewed=False)                         # once per turn
    assert len(learning.rows(str(wf))) == 5


# --- the gate's two tolerances, at their boundaries (bd fix-o5wi.6) -------------------------------------------------

def _counts(right: int, wrong: int) -> dict:
    return {"right_rows": right, "wrong_rows": wrong}


def test_the_base_rows_tolerance_is_one_and_a_half_points_and_no_more():
    assert head_refit.GATE_BASE_TOLERANCE == 0.015
    assert head_refit.base_gate_verdict(200, 200, 200) == (True, "no loss")
    assert head_refit.base_gate_verdict(200, 203, 200) == (True, "no loss")
    assert head_refit.base_gate_verdict(200, 197, 200) == (True, "base tolerance")     # exactly 1.5 points of 200
    assert head_refit.base_gate_verdict(200, 196, 200) == (False, "")                  # 2 points
    assert head_refit.base_gate_verdict(1000, 985, 1000) == (True, "base tolerance")   # exactly 15 rows of 1000
    assert head_refit.base_gate_verdict(1000, 984, 1000) == (False, "")


def test_the_learned_rows_gate_allows_one_more_wrong_row_only_for_three_more_right_rows():
    verdict = head_refit.learned_gate_verdict
    assert verdict(_counts(10, 3), _counts(10, 3)) == (True, "strict")
    assert verdict(_counts(10, 3), _counts(14, 2)) == (True, "strict")
    assert verdict(_counts(10, 3), _counts(9, 3)) == (False, "")                        # right outright lower
    assert verdict(_counts(10, 3), _counts(13, 4)) == (True, "wrong confident tolerance")   # +3 right buys +1 wrong
    assert verdict(_counts(10, 3), _counts(12, 4)) == (False, "")                       # +2 right does not
    assert verdict(_counts(10, 3), _counts(20, 5)) == (False, "")                       # two more wrong never passes
    assert verdict(_counts(10, 3), _counts(9, 4)) == (False, "")


def test_a_refit_passes_with_the_rule_named_and_reuses_the_baseline_crossfit(library, monkeypatch):
    cases = _benchmark("library_v2")
    assert _confirm_cases(library, cases, 60) == 60
    calls: list[int] = []
    original = head_refit._fit_probabilities

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(head_refit, "_fit_probabilities", counting)
    first = head_refit.refit(library)
    assert first.published, first.reason
    gate = first.gate["base_rows"]
    assert gate["rule"] in {"no loss", "base tolerance"} and gate["baseline_cached"] is False
    assert first.gate["learned_rows"]["rule"] in {"strict", "wrong confident tolerance"}
    assert learning.get_meta(library, head_refit.BASE_CROSSFIT_META_KEY)["counted"] == gate["rows"]
    first_calls = len(calls)
    assert _confirm_cases(library, cases[60:80]) == 20
    calls.clear()
    second = head_refit.refit(library)
    assert second.gate["base_rows"]["baseline_cached"] is True
    # The cached side saves one fit per fold; the refit side is computed again.
    assert len(calls) < first_calls
    assert second.gate["base_rows"]["accuracy_before"] == gate["accuracy_before"]
