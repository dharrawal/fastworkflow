"""The fw.compose span: one trace record per compose call (fix-weih.28).

Same real in-process app and scripted composer model as ``tests/test_compose_endpoints.py``. The turn a span is attached to
is a real turn (``/perform_action`` on the hello_world workflow), read back from the real observability store.
"""

from __future__ import annotations

import json
import time

import pytest

from fastworkflow.observability import tracing
from fastworkflow.observability.store import get_observability_sink
from fastworkflow.run_chatbot import composer as gui
from tests.test_compose_endpoints import (  # noqa: F401  (fixtures)
    _card_body, _form_body, app_module, client, composer, headers,
)

_CARD = [{"lead": "Your order is delivered.", "sections": [{"heading": "Order", "blocks": [
    {"kind": "stats", "stats": [{"label": "Status", "ref": "r1.status"}]}]}]}]
# Text the page sent that must never reach the store.
_USER_TEXT = "where is my order"
_VALUES = ("credit_card_4421890", "186.32", "Bike", "Your order is delivered.")


def _sink(app_module):
    sink = get_observability_sink(app_module.ARGS.workflow_path)
    assert sink is not None
    return sink


def _compose_spans(app_module, trace_id=None, wait=10.0) -> list[dict]:
    """Every fw.compose span in the store (or under one trace), waiting for the background writer."""
    sink = _sink(app_module)
    deadline = time.time() + wait
    while True:
        sink.flush()
        with sink.store._connect() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM spans WHERE name=?", (tracing.SPAN_COMPOSE,))]
        rows = [r for r in rows if trace_id is None or r["trace_id"] == trace_id]
        if rows or time.time() > deadline:
            return rows
        time.sleep(0.1)


def _attrs(row: dict) -> dict:
    return json.loads(row["attributes"])


def _real_turn(client, headers) -> str:
    resp = client.post("/perform_action", headers=headers, json={
        "action": {"command_name": "add_two_numbers", "parameters": {"first_num": 2.0, "second_num": 3.0}},
        "timeout_seconds": 60})
    assert resp.status_code == 200, resp.text
    key = resp.json().get("logical_turn_key") or resp.json()["turn_key"]
    # The turn row is written synchronously at finalize; resolve the logical key the store knows.
    done = client.get(f"/turns/{key}", headers=headers)
    assert done.status_code == 200, done.text
    return done.json().get("logical_turn_key") or key


def test_a_card_compose_call_writes_one_span_with_counts_and_no_text(client, headers, composer, app_module):
    composer(_CARD)
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)
    assert resp.status_code == 200 and resp.json()["card"] is not None

    rows = _compose_spans(app_module)
    assert len(rows) == 1
    attrs = _attrs(rows[0])
    assert attrs["compose_kind"] == "answer_card" and attrs["outcome"] == "ok"
    assert attrs["results"] == 1 and attrs["model"] and attrs["duration_ms"] >= 0
    assert attrs["sections_kept"] == 1 and attrs["sections_proposed"] == 1
    assert attrs["blocks_kept"] == 1 and attrs["blocks_proposed"] == 1
    assert attrs["lead"] == "kept" and attrs["retry"] == "none" and attrs["drops"] == {}
    assert attrs["span_contract_version"] == tracing.span_contract_versions()[tracing.SPAN_COMPOSE]
    assert set(attrs) - tracing.ENVELOPE_ATTRIBUTES <= tracing.SPAN_CONTRACTS[tracing.SPAN_COMPOSE].attributes
    assert rows[0]["status"] == tracing.STATUS_OK and rows[0]["end_ns"] >= rows[0]["start_ns"]
    stored = json.dumps(rows[0])
    for text in (_USER_TEXT, *_VALUES):
        assert text not in stored


def test_a_drop_reason_code_reaches_the_span(client, headers, composer, app_module):
    composer([{"lead": "Your order is delivered.", "sections": [{"blocks": [
        {"kind": "record", "ref": "r1.invented_field"},
        {"kind": "stats", "stats": [{"label": "Status", "ref": "r1.status"}]},
    ]}]}])
    assert client.post("/compose_answer_card", json=_card_body(), headers=headers).status_code == 200
    attrs = _attrs(_compose_spans(app_module)[0])
    assert attrs["drops"].get("ref_unresolved", 0) >= 1
    assert attrs["blocks_kept"] == 1 and attrs["blocks_proposed"] == 2


def test_a_question_form_span_has_its_kind_and_no_card_counts(client, headers, composer, app_module):
    composer([{"fields": [{"name": "payment_method", "label": "Refund to", "kind": "choice",
                           "options": ["paypal_5521", "credit_card_4421890"]}]}])
    resp = client.post("/compose_question_form", json=_form_body(), headers=headers)
    assert resp.status_code == 200
    rows = _compose_spans(app_module)
    assert len(rows) == 1
    attrs = _attrs(rows[0])
    assert attrs["compose_kind"] == "question_form" and attrs["results"] == 1
    assert attrs["outcome"] in ("ok", "not_needed", "failed")
    assert attrs["drops"] is None and attrs["model"]
    assert "paypal_5521" not in json.dumps(rows[0]) and "Which payment method" not in json.dumps(rows[0])


def test_the_span_carries_the_answer_and_excerpt_sizes(client, headers, composer, app_module):
    """A short answer is read whole (excerpt 0); a long one is read as an excerpt shorter than the answer."""
    short = "Your order is delivered."
    long_answer = "".join(f"### {n}. Part {n}\n" + ("The order moved through the depot. " * 60) + "\n\n" for n in range(1, 13))
    assert len(long_answer) > gui.ANSWER_PROMPT_WHOLE_CHARS
    composer(_CARD)
    assert client.post("/compose_answer_card", json=_card_body(answer=short), headers=headers).status_code == 200
    short_attrs = _attrs(_compose_spans(app_module)[0])
    assert short_attrs["answer_chars"] == len(short) and short_attrs["excerpt_chars"] == 0

    composer(_CARD)
    resp = client.post("/compose_answer_card", json=_card_body(answer=long_answer), headers=headers)
    assert resp.status_code == 200
    deadline = time.time() + 10
    while len(rows := _compose_spans(app_module)) < 2 and time.time() < deadline:
        time.sleep(0.1)
    long_attrs = next(a for a in map(_attrs, rows) if a["answer_chars"] == len(long_answer))
    assert 0 < long_attrs["excerpt_chars"] < long_attrs["answer_chars"]
    assert set(long_attrs) - tracing.ENVELOPE_ATTRIBUTES <= tracing.SPAN_CONTRACTS[tracing.SPAN_COMPOSE].attributes


def test_a_question_form_span_has_no_answer_sizes(client, headers, composer, app_module):
    composer([{"fields": [{"name": "payment_method", "label": "Refund to", "kind": "choice",
                           "options": ["paypal_5521", "credit_card_4421890"]}]}])
    assert client.post("/compose_question_form", json=_form_body(), headers=headers).status_code == 200
    attrs = _attrs(_compose_spans(app_module)[0])
    assert attrs["answer_chars"] is None and attrs["excerpt_chars"] is None


@pytest.mark.parametrize("reason", ["no_model", "failed"])
def test_a_call_without_a_card_is_recorded_with_its_reason(client, headers, composer, monkeypatch, app_module, reason):
    if reason == "no_model":
        def unavailable():
            raise gui.ComposerUnavailable("no model")
        monkeypatch.setattr(gui, "_composer_lm", unavailable)
    else:
        composer([{"lead": "ok", "sections": "this is not a json list"}] * 3)
    assert client.post("/compose_answer_card", json=_card_body(), headers=headers).json()["reason"] == reason
    rows = _compose_spans(app_module)
    assert _attrs(rows[0])["outcome"] == reason and rows[0]["status"] == tracing.STATUS_ERROR


def test_the_span_is_attached_to_the_turn_when_its_key_is_sent(client, headers, composer, app_module):
    turn_key = _real_turn(client, headers)
    composer(_CARD)
    resp = client.post("/compose_answer_card", json=_card_body(turn_key=turn_key), headers=headers)
    assert resp.status_code == 200

    rows = _compose_spans(app_module, trace_id=turn_key)
    assert len(rows) == 1
    assert rows[0]["parent_span_id"] == tracing.root_span_id(turn_key)
    assert _attrs(rows[0])["turn_attached"] is True
    # the trace viewer's own endpoint lists it beside the turn's other spans
    spans = client.get(f"/turns/{turn_key}/trace", headers=headers).json()["spans"]
    assert tracing.SPAN_COMPOSE in [span["name"] for span in spans]


def test_a_question_form_span_is_filed_under_the_question_it_composes(client, headers, composer, app_module):
    """bd fix-qjne.34: the form composes the question the turn waits on, so its span is that ``fw.question``'s child."""
    turn_key = _real_turn(client, headers)
    sink = _sink(app_module)
    question_id = tracing.deterministic_span_id(turn_key, tracing.SPAN_QUESTION, 1)
    sink.emit_span(tracing.Span(
        span_id=question_id, trace_id=turn_key, name=tracing.SPAN_QUESTION,
        parent_span_id=tracing.root_span_id(turn_key), kind=tracing.KIND_HUMAN_WAIT, status=tracing.STATUS_AWAITING_USER,
        start_ns=time.time_ns(), attributes={"kind": "agent", "asked": "Which payment method?"}))
    sink.flush()
    composer([{"fields": [{"name": "payment_method", "label": "Refund to", "kind": "choice",
                           "options": ["paypal_5521", "credit_card_4421890"]}]}])
    resp = client.post("/compose_question_form", json=_form_body(turn_key=turn_key), headers=headers)
    assert resp.status_code == 200
    rows = _compose_spans(app_module, trace_id=turn_key)
    assert len(rows) == 1
    assert rows[0]["parent_span_id"] == question_id


def test_the_span_is_still_recorded_unattached_without_a_key_or_with_a_foreign_one(client, headers, composer, app_module):
    composer(_CARD * 2)
    assert client.post("/compose_answer_card", json=_card_body(), headers=headers).status_code == 200
    assert client.post("/compose_answer_card", json=_card_body(turn_key="no-such-turn"), headers=headers).status_code == 200
    rows = _compose_spans(app_module)
    deadline = time.time() + 10
    while len(rows) < 2 and time.time() < deadline:
        rows = _compose_spans(app_module)
    assert len(rows) == 2
    for row in rows:
        assert row["trace_id"].startswith("compose-") and row["parent_span_id"] is None
        assert _attrs(row)["turn_attached"] is False


def test_a_turn_of_another_channel_is_not_claimed(client, headers, composer, app_module):
    turn_key = _real_turn(client, headers)
    other = client.post("/initialize", json={"channel_id": "compose_other", "user_id": "x"}).json()["access_token"]
    composer(_CARD)
    assert client.post("/compose_answer_card", json=_card_body(turn_key=turn_key),
                       headers={"Authorization": f"Bearer {other}"}).status_code == 200
    rows = _compose_spans(app_module)
    assert [r["trace_id"] for r in rows if r["trace_id"] == turn_key] == []
    assert len(rows) == 1 and _attrs(rows[0])["turn_attached"] is False


def test_a_malformed_turn_key_never_refuses_the_request(client, headers, composer):
    composer(_CARD * 2)
    assert client.post("/compose_answer_card", json=_card_body(turn_key="x" * 5000), headers=headers).status_code == 200
    assert client.post("/compose_answer_card", json=_card_body(turn_key=12), headers=headers).status_code == 200


def test_a_failing_tracer_does_not_fail_the_response(client, headers, composer, app_module, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("store is gone")
    monkeypatch.setattr(app_module.compose_trace.tracing, "start_span", broken)
    composer(_CARD)
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)
    assert resp.status_code == 200 and resp.json()["card"] is not None
    deadline = time.time() + 10
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    assert calls, "the tracer was never reached, so the test proves nothing"
