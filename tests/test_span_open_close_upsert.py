"""A long-lived span reaches the sink twice and must be stored exactly once.

`tracing.start_span(..., emit_open=True)` (tracing.py) emits the *same* `Span`
object at open and again at close, so every span of `fw.turn` / `fw.ask_user`
shape arrives at the sink twice.  That is deliberate -- the open event has to
be visible before a suspension and closable after it [R6] -- but it means the
sink boundary is emission-counted while the store is span-counted, and any
consumer that appends rather than upserts double-counts real traffic.

These tests pin the two halves of that contract:

* the emission surface really does deliver two events (so nobody "fixes" it by
  suppressing the open), and
* the durable surface holds one row, carrying the *closed* status, end and
  attributes, in either arrival order.

The ordering half matters because the two emissions cross an asynchronous
writer queue: a close that lands before its own open must not be reopened.
"""

from __future__ import annotations

import uuid

import pytest

import fastworkflow.observability_store as obs
from fastworkflow import tracing


class RecordingTraceSink:
    """Counts emissions, which is not the same thing as counting spans."""

    def __init__(self) -> None:
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def emit_distillation_record(self, *args) -> None:
        pass


class Host:
    """The duck-typed surface `tracing` resolves a span's context from."""

    def __init__(self, sink, turn_key: str) -> None:
        self.trace_sink = sink
        self.current_turn_key = turn_key
        self.observability_channel_id = "open-close-upsert"
        self.trace_span_stack: list = []


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


def _stored_spans(db_path: str, trace_id: str) -> list[dict]:
    store = obs.ReadOnlyObservabilityStore(db_path)
    return [
        row
        for row in store.get_spans(trace_id)
        if row["name"] == tracing.SPAN_TURN
    ]


def test_start_span_with_emit_open_delivers_two_emissions():
    """The emission surface is two events; this is the documented behaviour."""

    sink = RecordingTraceSink()
    host = Host(sink, f"turn-{uuid.uuid4().hex}")

    span = tracing.start_span(
        host,
        tracing.SPAN_TURN,
        attributes={"phase": "open"},
        emit_open=True,
        use_stack=False,
    )
    assert span is not None
    tracing.end_span(host, span, attributes={"phase": "closed"})

    assert len(sink.spans) == 2
    assert {id(recorded) for recorded in sink.spans} == {id(span)}
    assert sink.spans[0] is sink.spans[1]


def test_the_store_holds_one_row_with_the_closed_attributes(db_path):
    sink = obs.SQLiteTraceSink(db_path)
    turn_key = f"turn-{uuid.uuid4().hex}"
    host = Host(sink, turn_key)
    try:
        span = tracing.start_span(
            host,
            tracing.SPAN_TURN,
            attributes={"phase": "open"},
            emit_open=True,
            use_stack=False,
        )
        assert span is not None
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK,
            attributes={"phase": "closed"},
        )
        assert sink.flush()
    finally:
        sink.close()

    rows = _stored_spans(db_path, turn_key)

    assert len(rows) == 1
    assert rows[0]["span_id"] == span.span_id
    assert rows[0]["status"] == tracing.STATUS_OK
    assert rows[0]["end_ns"] is not None
    assert '"closed"' in rows[0]["attributes"]
    assert '"open"' not in rows[0]["attributes"]


def test_an_open_arriving_after_its_close_does_not_reopen_the_row(db_path):
    """The two emissions cross a queue; the close must survive re-ordering."""

    sink = obs.SQLiteTraceSink(db_path)
    turn_key = f"turn-{uuid.uuid4().hex}"
    try:
        closed = tracing.Span(
            span_id="deterministic-span",
            trace_id=turn_key,
            name=tracing.SPAN_TURN,
            start_ns=1,
            end_ns=2,
            status=tracing.STATUS_OK,
            attributes={"phase": "closed"},
        )
        sink.emit_span(closed)
        assert sink.flush()

        stale_open = tracing.Span(
            span_id="deterministic-span",
            trace_id=turn_key,
            name=tracing.SPAN_TURN,
            start_ns=1,
            end_ns=None,
            status=tracing.STATUS_OPEN,
            attributes={"phase": "open"},
        )
        sink.emit_span(stale_open)
        assert sink.flush()
    finally:
        sink.close()

    rows = _stored_spans(db_path, turn_key)

    assert len(rows) == 1
    assert rows[0]["status"] == tracing.STATUS_OK
    assert rows[0]["end_ns"] == 2
    assert '"closed"' in rows[0]["attributes"]
