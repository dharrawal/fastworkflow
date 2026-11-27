"""The live path of learning from use (bd fix-qcjo): requests are resolved on real CME sessions of a trained temporary
copy of the library example, the confirmed ones are recorded, and with ``INTENT_LEARNING_REFIT=background`` the daemon
thread refits the embedding head behind its gate while the process keeps running.

Real components throughout, on the same trained copy as tests/test_learning_from_use.py. The CME is driven directly
(``CommandExecutor.perform_action`` on the ``wildcard`` command, as a deterministic turn does): a FastAPI app would
need a second trained workflow and an HTTP stack for nothing the recording and the refit read. The only wrapper is a
spy on ``head_refit.refit`` that calls through, to see which thread runs it and to hold it open at a chosen point.
No model is called (``no_planner_key`` and ``workflow_model.no_model_calls`` in the shared ``library`` fixture).
"""

from __future__ import annotations

import os
import threading
import time

import pytest

import fastworkflow
from fastworkflow.intent import intent_detector
from fastworkflow.intent import intent_learning as learning
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import embedding_head as eh
from fastworkflow.train import head_refit
from tests.test_intent_model_layout_training import _route, _session_on_a_book, _training_env
from tests.test_learning_from_use import (  # noqa: F401  (fixtures)
    _benchmark,
    _current_model_dir,
    library,
    trained_library,
)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

EVERY = 8


def _background_env(library_path, every=EVERY):
    fastworkflow.init(env_vars={**_training_env(library_path, None), learning.REFIT_SETTING: learning.REFIT_BACKGROUND,
                                learning.REFIT_EVERY_SETTING: str(every)})


def _book_cases():
    return [case for case in _benchmark("library_v2")
            if case["context"] == "Book" and case["expected_label"].startswith("Book/")]


def _unanswered_by_the_head(library_path, cases):
    """The cases the local head alone does not answer outright in a Book context."""
    engine = intent_detector.engine_for(library_path, "Book", allow_matcher=False)
    return [case for case in cases if not engine.decide(case["utterance"]).confident]


def _ask_and_pick(cme, case):
    """A request the walk cannot settle alone, and the reply that picks its command: the which-command question."""
    asked = _route(cme, case["utterance"]).command_response
    if "mbiguous" not in asked.response:
        return False
    picked = _route(cme, case["expected_label"].split("/")[-1]).command_response
    return picked.artifacts.get("command_name") == case["expected_label"]


def _pick_rows(cme, cases, wanted):
    """Resolve requests until *wanted* picks have been made; returns the cases whose pick was recorded."""
    done = []
    for case in cases:
        if len(done) == wanted:
            break
        if _ask_and_pick(cme, case):
            done.append(case)
    return done


class _SpiedRefit:
    """Calls ``head_refit.refit`` through; says which thread ran it and can hold it open until released."""

    def __init__(self, monkeypatch, hold=False):
        self.threads, self.reports = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        if not hold:
            self.release.set()
        original = head_refit.refit

        def spy(*args, **kwargs):
            self.threads.append(threading.current_thread())
            self.entered.set()
            assert self.release.wait(120), "the test never released the held refit"
            report = original(*args, **kwargs)
            self.reports.append(report)
            return report

        monkeypatch.setattr(head_refit, "refit", spy)


@pytest.fixture
def session(library):
    app, cme = _session_on_a_book(library, "live-refit")
    try:
        yield app, cme
    finally:
        learning.wait_for_background(library)
        app.close()


def test_the_live_background_refit_runs_off_the_turn_is_gated_locked_and_picked_up_without_a_restart(
        library, session, monkeypatch):
    app, cme = session
    base_version = av.resolve_current_version(library)
    cases = _book_cases()
    unanswered = _unanswered_by_the_head(library, cases)
    assert len(unanswered) >= EVERY, "the probed Book requests the head leaves open are too few"
    _background_env(library)
    spy = _SpiedRefit(monkeypatch, hold=True)
    caller = threading.current_thread()

    # 1. Confirmed picks from real turns; the EVERY-th one starts the thread, and the turn returns while it is held.
    picked = _pick_rows(cme, unanswered, EVERY - 1)
    assert len(picked) == EVERY - 1 and not spy.entered.is_set()
    assert learning.new_teaching_count(library) == EVERY - 1
    started = time.monotonic()
    picked += _pick_rows(cme, unanswered[unanswered.index(picked[-1]) + 1:], 1)
    assert len(picked) == EVERY, "no further probed request was a pick"
    assert spy.entered.wait(60), "the background refit never started"
    assert not spy.reports, "the turn returned only after the refit finished"
    assert spy.threads[0] is not caller and spy.threads[0].daemon and spy.threads[0].is_alive()
    assert time.monotonic() - started < 60
    assert av.resolve_current_version(library) == base_version            # nothing published while it is held open

    # 4. The refit holds the file lock: another refit of the workflow (the CLI's `learn`) is turned away.
    assert os.path.isfile(os.path.join(library, "___convo_info", "intent_learning.lock"))
    with head_refit.refit_lock(library) as acquired:
        assert acquired is False
    turned_away = head_refit.learn(library)["refit"]
    assert not turned_away.published and "another refit of this workflow is running" in turned_away.reason
    assert len(spy.threads) == 1                                              # and nothing else started a refit

    # 2. Released: the thread refits through the gates and publishes a new version.
    spy.release.set()
    learning.wait_for_background(library)
    report = spy.reports[0]
    assert report.published, report.reason
    new_version = av.resolve_current_version(library)
    assert new_version == report.new_version != base_version
    manifest = av.read_manifest(library, new_version)
    assert manifest["refit_of"] == base_version and manifest["learned_rows"] == EVERY
    assert manifest["refit_gate"]["passed"] and manifest["refit_gate"]["base_rows"]["rule"]
    assert len(spy.threads) == 1 and spy.threads[0].name.startswith("intent-learning-refit-")
    with head_refit.refit_lock(library) as acquired:                         # the lock is free again
        assert acquired

    # 3. The same process, no restart: the old head's open requests are now answered outright by the new head.
    time.sleep(intent_detector.HEAD_RELOAD_CHECK_SECONDS + 0.2)              # the reload check is rate limited
    engine = intent_detector.engine_for(library, "Book", allow_matcher=False)
    outright = [case for case in picked if engine.decide(case["utterance"]).confident]
    assert len(outright) == len(picked), [case["utterance"] for case in picked if case not in outright]
    router = intent_detector._command_router(os.path.join(library, "___command_info", "Book"))
    assert router._workflow_router.head_version == new_version
    answered = _route(cme, picked[0]["utterance"]).command_response
    assert answered.artifacts.get("command_name") == picked[0]["expected_label"]    # the live session routes by it


def test_a_refit_that_fails_its_gate_in_the_background_leaves_the_current_head_in_place(library, session, monkeypatch):
    app, cme = session
    base_version = av.resolve_current_version(library)
    model_dir = _current_model_dir(library)
    store = eh.read_row_store(str(model_dir))
    import json

    labels = json.loads((model_dir / "scopes.json").read_text())["labels"]
    wrong = "Library/count_books"
    for command in sorted({labels[t] for k, t in zip(store["kinds"], store["targets"])
                           if k == 1 and labels[t].startswith("Library/")} - {wrong}):
        for text in [x for x, k, t in zip(store["texts"], store["kinds"], store["targets"])
                     if k == 1 and labels[t] == command]:
            learning.record(library, text=text, context="Library", command=wrong, how=learning.HOW_USER_CHOICE)
    old_router = intent_detector._command_router(os.path.join(library, "___command_info", "Book"))
    _background_env(library, every=1)
    spy = _SpiedRefit(monkeypatch)
    # A real turn's pick is the row that starts the thread.
    case = _unanswered_by_the_head(library, _book_cases())[0]
    assert _ask_and_pick(cme, case)
    learning.wait_for_background(library)

    assert len(spy.reports) == 1
    report = spy.reports[0]
    assert not report.published and "gate failed" in report.reason and report.gate["passed"] is False
    assert av.resolve_current_version(library) == base_version
    assert [info.version_id for info in av.list_versions(library)] == [base_version]
    time.sleep(intent_detector.HEAD_RELOAD_CHECK_SECONDS + 0.2)
    intent_detector.refresh_stale_head(os.path.join(library, "___command_info", "Book"))
    assert intent_detector._command_router(os.path.join(library, "___command_info", "Book")) is old_router
    assert learning.new_teaching_count(library) == 0         # the attempt counts: the next one waits for new rows


def test_what_an_agent_picks_from_the_candidate_list_is_a_confirmed_row_and_teaches_a_refit(library, session):
    app, cme = session
    app.context["run_as_agent"] = True                       # an agent's session: the pick is the agent's own
    base_version = av.resolve_current_version(library)
    unanswered = _unanswered_by_the_head(library, _book_cases())
    picked = _pick_rows(cme, unanswered, 8)
    assert len(picked) == 8

    rows = [row for row in learning.rows(library) if row["how"] == learning.HOW_AGENT_CHOICE]
    assert [row["text"] for row in rows] == [case["utterance"] for case in picked]
    assert all(row["confirmed"] == 1 for row in rows)
    assert [row["command"] for row in rows] == [case["expected_label"] for case in picked]
    assert not [row for row in learning.rows(library) if row["how"] == learning.HOW_USER_CHOICE]
    assert len(learning.teaching_rows(library)) == 8

    report = head_refit.refit(library)
    assert report.published, report.reason
    assert report.learned_rows == 8 and report.new_version != base_version
    assert av.read_manifest(library, report.new_version)["learned_rows"] == 8
    intent_detector._last_head_check.clear()
    engine = intent_detector.engine_for(library, "Book", allow_matcher=False)
    assert all(engine.decide(case["utterance"]).confident for case in picked)
