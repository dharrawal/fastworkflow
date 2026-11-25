"""Spans opened by work run in a thread pool are parented by the span that was current when the work was submitted,
never by a sibling worker's span (bd fix-qjne.34, step 1).

The host's parenting stack is one list shared by every thread; ``tracing.traced_call`` gives each submitted call its own
copy of it. Real threads, the real tracer (``start_span``/``end_span`` under ``host_scope``, as the NLU stages and the
model-call logger open them), and every pool site reachable without a model: ``subtask_pool.map_in_order``,
``planning_commands.check_verdicts`` and ``signatures.fields_not_given`` (their one model call replaced by a leaf span,
the shape ``dspy_logger`` writes). Workers wait on a barrier, so every worker's span is open before any is closed: the
interleaving that nested sibling spans under each other before the fix.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from fastworkflow import planning_commands, subtask_pool, tracing
from fastworkflow.utils import signatures

WAIT = 20


class _Sink:
    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def emit_span(self, span):
        with self._lock:
            self.spans.append(span)

    def emit_turn_record(self, record):
        return True

    def record_conversation_label(self, *a):
        pass


@pytest.fixture(autouse=True)
def pooled(monkeypatch):
    monkeypatch.setenv(subtask_pool.SUBTASK_WORKERS_ENV_VAR, "4")


@pytest.fixture
def host():
    host = SimpleNamespace(trace_sink=_Sink(), current_turn_key="turn-parentage", trace_span_stack=[])
    with tracing.host_scope(host):
        yield host


def _by_id(host) -> dict:
    return {span.span_id: span for span in host.trace_sink.spans}


def _model_call(host, label: str):
    """A model call as ``dspy_logger`` opens it: a leaf, parented by the top of the stack, never pushed."""
    leaf = tracing.start_span(host, tracing.SPAN_LLM_CALL, kind=tracing.KIND_LLM, attributes={"module": label},
                              use_stack=False)
    tracing.end_span(host, leaf)


def _part_work(host, barrier: threading.Barrier, number: int):
    """One part's stage: its span is open while every sibling's is, and a model call is made inside it."""
    def call():
        span = tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": number})
        barrier.wait(WAIT)                       # every sibling has pushed its span now
        _model_call(tracing.current_host(), f"part {number}")
        barrier.wait(WAIT)
        tracing.end_span(tracing.current_host(), span)
        return number
    return call


def test_each_worker_span_is_parented_by_the_span_open_at_submission_and_its_model_call_by_it(host):
    stage = tracing.start_span(host, tracing.SPAN_PLAN)
    barrier = threading.Barrier(3)
    assert subtask_pool.map_in_order([_part_work(host, barrier, n) for n in (1, 2, 3)]) == [1, 2, 3]
    tracing.end_span(host, stage)

    spans = _by_id(host)
    parts = {s.attributes["part"]: s for s in spans.values() if s.name == tracing.SPAN_INTENT}
    assert sorted(parts) == [1, 2, 3]
    for span in parts.values():   # (a) the stage span, never a sibling part's span
        assert span.parent_span_id == stage.span_id, (span.attributes["part"], spans[span.parent_span_id].name)
    calls = [s for s in spans.values() if s.name == tracing.SPAN_LLM_CALL]
    assert len(calls) == 3
    for call in calls:            # (b) each model call under its own part's span
        number = int(call.attributes["module"].split()[1])
        assert call.parent_span_id == parts[number].span_id
    assert stage.parent_span_id == tracing.root_span_id("turn-parentage")


def test_the_caller_s_stack_is_unchanged_by_the_pool_and_a_span_around_it_keeps_its_children(host):
    outer = tracing.start_span(host, tracing.SPAN_PLAN)
    before = list(host.trace_span_stack)
    seen_by_caller = []
    barrier = threading.Barrier(3)

    def watch(call):
        def run():
            result = call()
            seen_by_caller.append(list(host.trace_span_stack))   # the host's own list, read from a worker
            return result
        return run

    subtask_pool.map_in_order([watch(_part_work(host, barrier, n)) for n in (1, 2, 3)])
    assert host.trace_span_stack == before == [outer]           # (c) nothing pushed or left by a worker
    assert all(stack == [outer] for stack in seen_by_caller)
    after = tracing.start_span(host, tracing.SPAN_PARAMETERS)   # the caller's thread parents as before
    assert after.parent_span_id == outer.span_id
    tracing.end_span(host, after)
    tracing.end_span(host, outer)
    assert host.trace_span_stack == []
    children = [s for s in host.trace_sink.spans if s.parent_span_id == outer.span_id]
    assert len(children) == 4    # three parts and the span opened after the pool


def test_a_worker_that_submits_work_parents_the_inner_spans_to_its_own_span(host):
    outer = tracing.start_span(host, tracing.SPAN_PLAN)
    inner_barrier = threading.Barrier(4)    # two outer workers x two inner calls, all open at once

    def outer_call(number):
        def run():
            span = tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": number})
            inner = [_part_work(host, inner_barrier, number * 10 + k) for k in (1, 2)]
            out = subtask_pool.map_in_order(inner)            # (d) a nested pool
            tracing.end_span(tracing.current_host(), span)
            return out
        return run

    assert subtask_pool.map_in_order([outer_call(1), outer_call(2)]) == [[11, 12], [21, 22]]
    tracing.end_span(host, outer)
    spans = _by_id(host)
    parts = {s.attributes["part"]: s for s in spans.values() if s.name == tracing.SPAN_INTENT}
    assert parts[1].parent_span_id == parts[2].parent_span_id == outer.span_id
    for number in (11, 12, 21, 22):
        assert parts[number].parent_span_id == parts[number // 10].span_id
    for call in (s for s in spans.values() if s.name == tracing.SPAN_LLM_CALL):
        assert call.parent_span_id == parts[int(call.attributes["module"].split()[1])].span_id
    assert host.trace_span_stack == []


def test_a_worker_that_raises_leaves_no_span_on_any_stack(host):
    outer = tracing.start_span(host, tracing.SPAN_PLAN)

    def failing():
        tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": 1})
        raise RuntimeError("part 1 failed with its span open")

    def after_it(number):
        def run():
            span = tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": number})
            tracing.end_span(tracing.current_host(), span)
            return span
        return run

    with pytest.raises(RuntimeError):
        subtask_pool.map_in_order([failing, after_it(2)])
    assert host.trace_span_stack == [outer]                    # (e) the span left open by the failure is not here
    (second,) = subtask_pool.map_in_order([after_it(3)])        # one call: the caller's thread
    assert second.parent_span_id == outer.span_id
    (pooled_again, _) = subtask_pool.map_in_order([after_it(4), after_it(5)])
    assert pooled_again.parent_span_id == outer.span_id
    tracing.end_span(host, outer)
    assert host.trace_span_stack == []


def test_background_work_parents_to_the_span_open_at_submission_while_the_caller_moves_on(host):
    """The argument checks run in the background while the caller keeps opening spans (``_start_argument_checks``)."""
    first = tracing.start_span(host, tracing.SPAN_INTENT)
    caller_moved = threading.Event()

    def background():
        assert caller_moved.wait(WAIT)
        _model_call(tracing.current_host(), "background 0")

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(tracing.traced_call(background))
        later = tracing.start_span(host, tracing.SPAN_PARAMETERS)
        caller_moved.set()
        future.result()
    tracing.end_span(host, later)
    tracing.end_span(host, first)
    (call,) = [s for s in host.trace_sink.spans if s.name == tracing.SPAN_LLM_CALL]
    assert call.parent_span_id == first.span_id


def test_without_a_host_a_traced_call_is_a_plain_context_copy():
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(tracing.traced_call(lambda a, b: a + b, 2, 3)).result() == 5


def test_the_argument_check_pool_files_each_model_call_under_the_span_open_at_submission(host, monkeypatch):
    barrier = threading.Barrier(2)

    def check_one(lm, command, p, request):
        span = tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": p.name})
        barrier.wait(WAIT)
        _model_call(tracing.current_host(), f"check {p.name}")
        barrier.wait(WAIT)
        tracing.end_span(tracing.current_host(), span)
        return 1, 0.99

    monkeypatch.setattr(planning_commands, "_check_one", check_one)
    command = SimpleNamespace(name="plan", params=[SimpleNamespace(name=n, verify=True, type="int") for n in ("a", "b")])
    stage = tracing.start_span(host, tracing.SPAN_PARAMETERS)
    assert planning_commands.check_verdicts(object(), command, "two numbers") == {"a": (1, 0.99), "b": (1, 0.99)}
    tracing.end_span(host, stage)
    spans = _by_id(host)
    checks = {s.attributes["part"]: s for s in spans.values() if s.name == tracing.SPAN_INTENT}
    assert {c.parent_span_id for c in checks.values()} == {stage.span_id}
    for call in (s for s in spans.values() if s.name == tracing.SPAN_LLM_CALL):
        assert call.parent_span_id == checks[call.attributes["module"].split()[1]].span_id
    assert host.trace_span_stack == []


class _Statement(BaseModel):
    title: str = ""
    author: str = ""


def test_the_field_given_check_pool_files_each_model_call_under_the_span_open_at_submission(host, monkeypatch):
    import fastworkflow.workflow_agent as workflow_agent

    barrier = threading.Barrier(2)

    def check_field_given(lm, statement, name, info):
        span = tracing.start_span(tracing.current_host(), tracing.SPAN_INTENT, attributes={"part": name})
        barrier.wait(WAIT)
        _model_call(tracing.current_host(), f"field {name}")
        barrier.wait(WAIT)
        tracing.end_span(tracing.current_host(), span)
        return "given", 0.9

    monkeypatch.setattr(signatures, "_field_given_enabled", lambda: True)
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: object())
    monkeypatch.setattr(signatures, "_check_field_given", check_field_given)
    stage = tracing.start_span(host, tracing.SPAN_PARAMETERS)
    assert signatures.fields_not_given("find something", _Statement, {"title": "Emma", "author": "Austen"}) == {}
    tracing.end_span(host, stage)
    spans = _by_id(host)
    checks = {s.attributes["part"]: s for s in spans.values() if s.name == tracing.SPAN_INTENT}
    assert sorted(checks) == ["author", "title"]
    assert {c.parent_span_id for c in checks.values()} == {stage.span_id}
    for call in (s for s in spans.values() if s.name == tracing.SPAN_LLM_CALL):
        assert call.parent_span_id == checks[call.attributes["module"].split()[1]].span_id
    assert host.trace_span_stack == []
