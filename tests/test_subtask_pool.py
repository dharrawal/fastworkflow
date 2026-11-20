"""The independent per-sub-task model calls of a turn run side by side with the outcome of running them in order (fix-awg8).

Two layers, both on real components:

- ``subtask_pool.map_in_order`` with real threads, ordered by ``threading.Event`` instead of sleeps: results keep call
  order, the first exception in call order is raised and no worker outlives the call, one call (or one worker) makes no
  pool, and each call sees the caller's context (turn memo, progress listener, the observability host of the trace).
- ``review_proposals.draft_round1_proposal`` on the library copy with its trained heads: the real drafts (the shared
  parameter extractor, its checks) against a thread-safe model that answers by the text it is asked about, so the answer
  does not depend on call order. The proposal is identical to the sequential one, and a sub-task whose draft is held back
  does not reorder the items.
"""
from __future__ import annotations

import contextvars
import queue
import threading

import dspy
import pytest

from tests.conftest import script_matcher
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import review_proposals, subtask_pool, turn_cache, turn_progress
from fastworkflow.turn_progress import ProgressEvent
from fastworkflow.utils import dspy_logger, signatures
from tests.test_drafted_subtask_inputs import (  # noqa: F401 - the library copy with its trained heads, and the guards
    EMMA, _framework, annotated_library, library, library_heads)
from tests.test_planning_command_commands import RecordingTraceSink
from tests.test_text_planner_and_split import _session, workflow_env  # noqa: F401

DUNE = "who wrote Dune and how many copies do we have"
WAIT = 20


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    """Four workers: the drafted-input fixtures imported above set one, because they script a model in call order."""
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "4")


@pytest.fixture
def turn_memo():
    """A turn's memo bound for the test, and unbound after it (``begin_turn`` would leak into later tests)."""
    token = turn_cache._memo_var.set({})
    yield
    turn_cache._memo_var.reset(token)


# ---------------------------------------------------------------- map_in_order


def test_results_keep_call_order_when_the_second_call_finishes_first():
    second_done = threading.Event()
    finished: list[int] = []

    def first() -> int:
        assert second_done.wait(WAIT), "the second call never ran beside the first"
        finished.append(1)
        return 1

    def second() -> int:
        finished.append(2)
        second_done.set()
        return 2

    assert subtask_pool.map_in_order([first, second]) == [1, 2]
    assert finished == [2, 1]


def test_the_first_exception_in_call_order_is_raised_and_every_call_is_awaited(caplog):
    first_may_fail = threading.Event()
    finished: list[str] = []

    def first() -> None:
        assert first_may_fail.wait(WAIT)
        finished.append("first")
        raise ValueError("first failed")

    def second() -> None:
        finished.append("second")
        first_may_fail.set()
        raise KeyError("second failed")

    def third() -> str:
        finished.append("third")
        return "ok"

    # A pool thread another test left behind is not this call's: only threads started here are counted.
    before = set(threading.enumerate())
    with pytest.raises(ValueError, match="first failed"):
        subtask_pool.map_in_order([first, second, third])
    # none is left running: all three finished before the exception left map_in_order
    assert sorted(finished) == ["first", "second", "third"]
    assert not [t for t in threading.enumerate()
                if t not in before and t.name.startswith("ThreadPoolExecutor")]


def test_the_exception_is_the_same_as_the_sequential_one_for_the_first_failing_call():
    def boom(label: str):
        def call():
            raise RuntimeError(label)
        return call

    with pytest.raises(RuntimeError, match="a"):
        subtask_pool.map_in_order([boom("a"), boom("b")])


def test_one_call_makes_no_pool_and_runs_in_the_callers_thread(monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("a single call must not create a pool")

    monkeypatch.setattr(subtask_pool, "ThreadPoolExecutor", no_pool)
    caller = threading.get_ident()
    assert subtask_pool.map_in_order([lambda: threading.get_ident()]) == [caller]
    assert subtask_pool.map_in_order([]) == []


def test_one_worker_runs_every_call_in_order_in_the_callers_thread_and_stops_at_a_failure(monkeypatch):
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "1")
    monkeypatch.setattr(subtask_pool, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("no pool with one worker"))
    seen: list[tuple[str, int]] = []

    def call(label: str):
        def run():
            seen.append((label, threading.get_ident()))
            if label == "b":
                raise ValueError(label)
            return label
        return run

    with pytest.raises(ValueError):
        subtask_pool.map_in_order([call("a"), call("b"), call("c")])
    assert seen == [("a", threading.get_ident()), ("b", threading.get_ident())]


@pytest.mark.parametrize("value", ["0", "-2", "many", ""])
def test_a_bad_worker_setting_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, value)
    assert subtask_pool.worker_limit() == subtask_pool.DEFAULT_SUBTASK_WORKERS


def test_no_more_workers_than_the_limit_run_at_once(monkeypatch):
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "2")
    lock = threading.Lock()
    running = peak = 0
    gate = threading.Event()

    def call() -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        gate.wait(0.2)
        with lock:
            running -= 1
        return 0

    assert subtask_pool.map_in_order([call] * 6) == [0] * 6
    assert peak == 2


def test_before_runs_once_per_call_in_order_in_the_callers_thread():
    caller = threading.get_ident()
    started: list[tuple[int, int]] = []
    subtask_pool.map_in_order([lambda: None] * 3, before=lambda index: started.append((index, threading.get_ident())))
    assert started == [(0, caller), (1, caller), (2, caller)]


def test_each_call_runs_in_a_copy_of_the_callers_context(turn_memo):
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("test_marker", default="unset")
    marker.set("turn")

    def call(n: int):
        def run():
            turn_cache.memo(("key", n), lambda: n * 10)           # lands in the caller's memo
            marker.set(f"worker {n}")                              # stays in the copy
            return marker.get()
        return run

    assert subtask_pool.map_in_order([call(1), call(2)]) == ["worker 1", "worker 2"]
    assert marker.get() == "turn"
    assert turn_cache.memo(("key", 1), lambda: "recomputed") == 10
    assert turn_cache.memo(("key", 2), lambda: "recomputed") == 20


def test_progress_events_and_llm_spans_from_worker_threads_belong_to_the_turn(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, _wf = _session(sink)
    ctx._begin_turn("pooled spans")
    events: queue.Queue = queue.Queue()
    ctx.set_transport_queues(command_trace_queue=events)
    evidence = {"probabilities": {"true": 0.9, "false": 0.1}, "confidence": 0.9}

    def call(n: int):
        def run():
            turn_progress.emit(turn_progress.STAGE_DRAFT, n, 2)
            with dspy.context(lm=DummyLM([{"answer": f"a{n}"}])):
                return dspy.Predict("question -> answer")(question=f"q{n}").answer
        return run

    with turn_progress.bound(ctx), dspy_logger.observe_dspy_host(ctx):
        assert subtask_pool.map_in_order([call(1), call(2)]) == ["a1", "a2"]

    spans = sink.named("fw.llm.call")
    assert spans, "a worker's model calls left no fw.llm.call span"
    assert all(span.trace_id == ctx._turn_key for span in spans)
    assert all(any(f"a{n}" in str(span.attributes) for span in spans) for n in (1, 2)), "a call left no span"
    progress = []
    while not events.empty():
        item = events.get_nowait()
        if isinstance(item, ProgressEvent):
            progress.append((item.stage, item.index, item.total))
    assert sorted(progress) == [("draft", 1, 2), ("draft", 2, 2)]


# ---------------------------------------------------------------- draft_round1_proposal


class KeyedLM(DummyLM):
    """The extractor's model, answering by the text it is asked about (so thread order does not matter). A text in
    ``hold`` waits for the text it names in ``release`` to be asked first; every request is recorded. A copy of it (the
    extractor copies its LM per attempt) shares the same records, as ``ScriptedExtractor``'s do."""

    def __init__(self, answers: dict[str, dict], hold: dict[str, threading.Event] | None = None,
                 raises: set[str] | None = None):
        super().__init__([{}], adapter=dspy.JSONAdapter())
        self.answers = answers
        self.hold = hold or {}
        self.raises = raises or set()
        self.asked: list[str] = []
        self.finished: list[str] = []
        self.started = {key: threading.Event() for key in answers}

    def forward(self, prompt=None, messages=None, **kwargs):
        text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
        key = next(key for key in self.answers if key in text)
        self.asked.append(key)
        self.started[key].set()
        if key in self.hold:
            assert self.hold[key].wait(WAIT), f"{key!r} was held and never released"
        try:
            if key in self.raises:
                raise RuntimeError(f"the model failed on {key!r}")
            inner = DummyLM([{"reasoning": "scripted", **self.answers[key]}], adapter=dspy.JSONAdapter())
            return inner.forward(prompt=prompt, messages=messages, **kwargs)
        finally:
            self.finished.append(key)


def _keyed(monkeypatch, lm: KeyedLM) -> KeyedLM:
    monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
    return lm


ANSWERS = {"Emma": {"title": "Emma"}, "Dune": {"title": "Dune"}}


def _items(draft) -> list[dict]:
    return [item.model_dump() for item in draft.items]


def test_the_pooled_draft_is_the_sequential_draft(library_heads, monkeypatch):
    sub_tasks = [EMMA, DUNE]
    _keyed(monkeypatch, KeyedLM(ANSWERS))
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "1")
    sequential = review_proposals.draft_round1_proposal(library_heads, sub_tasks)
    lm = _keyed(monkeypatch, KeyedLM(ANSWERS))
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "4")

    pooled = review_proposals.draft_round1_proposal(library_heads, sub_tasks)

    assert _items(pooled) == _items(sequential)
    assert pooled.rendering() == sequential.rendering()
    assert [item.means.name for item in pooled.items] == ["Library/book_by_title", "Library/book_by_title"]
    assert [item.inputs[0].value for item in pooled.items] == ["Emma", "Dune"]
    assert sorted(lm.asked) == ["Dune", "Emma"]


def test_the_items_keep_sub_task_order_when_the_second_draft_finishes_first(library_heads, monkeypatch):
    first_may_go = threading.Event()
    lm = _keyed(monkeypatch, KeyedLM(ANSWERS, hold={"Emma": first_may_go}))
    watcher = threading.Thread(target=lambda: (lm.started["Dune"].wait(WAIT), first_may_go.set()))
    watcher.start()

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, DUNE])
    watcher.join()

    assert lm.finished == ["Dune", "Emma"], "the second sub-task's draft did not finish first"
    assert [item.goal for item in draft.items] == [EMMA, DUNE]
    assert [item.inputs[0].value for item in draft.items] == ["Emma", "Dune"]


def test_a_failed_draft_is_shown_as_before_and_the_other_draft_is_kept(library_heads, monkeypatch):
    """A draft is advisory: its failure leaves that sub-task with no inputs (a warning), the same pooled or not."""
    _keyed(monkeypatch, KeyedLM(ANSWERS, raises={"Emma"}))
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "1")
    sequential = review_proposals.draft_round1_proposal(library_heads, [EMMA, DUNE])
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "4")

    pooled = review_proposals.draft_round1_proposal(library_heads, [EMMA, DUNE])

    assert _items(pooled) == _items(sequential)
    assert pooled.items[0].inputs == [] and pooled.items[1].inputs[0].value == "Dune"


def test_a_single_sub_task_draft_makes_no_pool(library_heads, monkeypatch):
    _keyed(monkeypatch, KeyedLM(ANSWERS))
    monkeypatch.setattr(subtask_pool, "ThreadPoolExecutor", lambda *a, **k: pytest.fail("one sub-task: no pool"))

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA])

    assert draft.items[0].inputs[0].value == "Emma"


def test_the_stage_events_of_a_pooled_draft_are_one_per_sub_task_and_in_order(library_heads, monkeypatch):
    _keyed(monkeypatch, KeyedLM(ANSWERS))
    events: queue.Queue = queue.Queue()

    class Context:
        command_trace_queue = events

    with turn_progress.bound(Context()):
        review_proposals.draft_round1_proposal(library_heads, [EMMA, DUNE])

    stages = []
    while not events.empty():
        item = events.get_nowait()
        if isinstance(item, ProgressEvent):
            stages.append((item.stage, item.index, item.total))
    assert stages == [("match", 1, 2), ("match", 2, 2), ("draft", 1, 2), ("draft", 2, 2)]


def test_a_failed_prefetched_match_is_asked_again_in_order(library_heads, monkeypatch, turn_memo):
    """Every sub-task's intent is decided together; one that fails there is dropped and the in-order loop decides that
    text again, so the draft is made as it would have been."""
    from fastworkflow import intent_loop

    calls: list[str] = []
    real = intent_loop.resolve

    def flaky(workflow, text, core=False):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("the first match failed")
        return real(workflow, text, core=core)

    # the remote model's answer is the library_heads fixture's: a local plan pick is decided by it (fix-o1lg.2)
    monkeypatch.setattr(intent_loop, "resolve", flaky)
    _keyed(monkeypatch, KeyedLM(ANSWERS))

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, DUNE])

    assert len(calls) == 3 and sorted(calls[:2]) == sorted([EMMA, DUNE])     # prefetch (one failed), then in order
    assert [item.inputs[0].value for item in draft.items] == ["Emma", "Dune"]
