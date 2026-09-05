from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

from fastworkflow import observability_store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server


def _turn(turn_key: str, channel_id: str) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": None,
        "ordinal": None,
        "user_message": "waiting",
        "refined_user_message": None,
        "entry_workflow_name": "workflow",
        "entry_context": "context",
        "status": "awaiting_user",
        "success": 0,
        "failure_reason": None,
        "answer": "",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": None,
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps({"turn_output": {"turn_key": turn_key}}),
    }


def _get(server, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_feedback_is_readable_without_a_conversation_summary(tmp_path):
    db_path = tmp_path / "observability.sqlite3"
    store = obs.ObservabilityStore(str(db_path))
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn, _turn("turn/no-summary", "channel-a"), [], obs.Redactor()
        )
        conn.commit()
    verdict = json.dumps({"verdict": "useful", "score": 1})
    store.upsert_feedback("turn/no-summary", verdict)

    assert store.get_feedback("turn/no-summary") == {
        "turn_key": "turn/no-summary",
        "feedback_json": verdict,
        "updated_at": store.get_feedback("turn/no-summary")["updated_at"],
    }
    assert (
        store.list_feedback(channel_id="channel-a", limit=10)[0]["feedback_json"]
        == verdict
    )

    server = run_chatbot_server.ChatbotServer(str(db_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        encoded = urllib.parse.quote("turn/no-summary", safe="")
        status, payload = _get(server, f"/api/feedback/{encoded}")
        assert status == 200
        assert payload["feedback"]["feedback_json"] == verdict
        assert "conversation_summary" not in payload["feedback"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_feedback_list_filters_by_channel_without_usable_turn_filter(tmp_path):
    store = obs.ObservabilityStore(str(tmp_path / "observability.sqlite3"))
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for turn_key, channel_id in (("turn-a", "a"), ("turn-b", "b")):
            assert store.upsert_turn_row(
                conn, _turn(turn_key, channel_id), [], obs.Redactor()
            )
        conn.commit()
    store.upsert_feedback("turn-a", '{"verdict":"a"}')
    store.upsert_feedback("turn-b", '{"verdict":"b"}')

    assert [row["turn_key"] for row in store.list_feedback(channel_id="a")] == [
        "turn-a"
    ]
