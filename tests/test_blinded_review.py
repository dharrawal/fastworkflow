from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastworkflow.observability_store import ObservabilityStore
from fastworkflow.observability_workspace import WORKSPACE_SCHEMA
from fastworkflow.run_chatbot.server import ChatbotServer


def _turn_row(turn_key: str, task_id: str, success: bool) -> dict:
    outcome = "pass" if success else "fail"
    return {
        "turn_key": turn_key,
        "channel_id": f"channel-{task_id}",
        "conversation_id": None,
        "ordinal": None,
        "user_message": f"run {task_id}",
        "refined_user_message": None,
        "entry_workflow_name": "blinded-review-test",
        "entry_context": "test",
        "status": "completed" if success else "failed",
        "success": int(success),
        "failure_reason": None if success else "command exited with code 1",
        "answer": f"system-default-{outcome}",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": "2026-09-04T00:00:01+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": "experiment-1",
        "task_id": task_id,
        "attempt": 1,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(
            {
                "turn_output": {
                    "turn_key": turn_key,
                    "status": "completed" if success else "failed",
                    "success": success,
                    "answer": f"nested-system-default-{outcome}",
                    "outcomeSource": "machine-predicate",
                    "machine_predicate": {
                        "predicate_verdict": outcome,
                        "passed": success,
                    },
                    "command_outputs": [
                        {"command": "run-check", "observation": "trace observation"}
                    ],
                }
            }
        ),
    }


def _seed_workspace(tmp_path: Path) -> Path:
    source = tmp_path / "live.sqlite3"
    store = ObservabilityStore(str(source))
    store.create_experiment(
        "experiment-1",
        "Blinded review",
        declared_tasks=2,
        declared_attempts=1,
    )
    for turn_key, task_id, success in (
        ("turn-pass", "task-pass", True),
        ("turn-fail", "task-fail", False),
    ):
        store.start_attempt("experiment-1", task_id, 1, f"channel-{task_id}")
        store.finish_attempt(
            "experiment-1",
            task_id,
            1,
            outcome="pass" if success else "fail",
            outcome_source="machine-predicate",
        )
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            assert store.upsert_turn_row(
                conn,
                _turn_row(turn_key, task_id, success),
                [],
                store._store_redactor(),
            )
            conn.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, kind, channel_id,
                    command_name, context, start_ns, end_ns, status, attributes,
                    experiment_id, task_id, attempt)
                   VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"span-{turn_key}",
                    turn_key,
                    "fw.command.execute",
                    "internal",
                    f"channel-{task_id}",
                    "run-check",
                    "test",
                    1,
                    2,
                    "completed" if success else "error",
                    json.dumps(
                        {
                            "raw_command": "run-check",
                            "response_text": (
                                "check completed"
                                if success
                                else "failure observation: exit code 1"
                            ),
                            "success": success,
                            "evaluationResult": "pass" if success else "fail",
                            "predicate_results": [
                                {
                                    "name": "machine-check",
                                    "verdict": "pass" if success else "fail",
                                }
                            ],
                            "final_answer": f"trace-default-{'pass' if success else 'fail'}",
                        }
                    ),
                    "experiment-1",
                    task_id,
                    1,
                ),
            )
            conn.commit()
    archive = store.archive_to(str(tmp_path / "sealed.sqlite3"))
    manifest = tmp_path / "workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "workspace-1",
                "label": "Blinded review workspace",
                "stores": [
                    {
                        "store_id": "store-a",
                        "label": "Sealed evidence",
                        "path": Path(archive["path"]).name,
                        "mode": "sealed",
                        "sha256": archive["sha256"],
                        "store_identity": archive["store_identity"],
                    }
                ],
                "experiments": [
                    {
                        "experiment_id": "logical-experiment",
                        "segments": [
                            {
                                "store_id": "store-a",
                                "local_experiment_id": "experiment-1",
                            }
                        ],
                    }
                ],
                "projected_attempts": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _assignment(assignment_id: str, *, blinded: bool) -> dict:
    return {
        "id": assignment_id,
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": ["adjudicator"],
        "blinded": blinded,
        "rows": [
            {
                "id": "row-fail-first",
                "turn_ref": {
                    "store_id": "store-a",
                    "logical_turn_key": "turn-fail",
                },
            },
            {
                "id": "row-pass-second",
                "turn_ref": {
                    "store_id": "store-a",
                    "logical_turn_key": "turn-pass",
                },
            },
        ],
        "questions": [
            {
                "id": "verdict",
                "prompt": "Choose an outcome.",
                "type": "single-select",
                "vocabulary": ["pass", "fail"],
            }
        ],
    }


@contextmanager
def _serve(manifest: Path) -> Iterator[ChatbotServer]:
    server = ChatbotServer(port=0, workspace_manifest_path=str(manifest))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _request(
    server: ChatbotServer,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    capability: str | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {server.token}",
            "Content-Type": "application/json",
        },
    )
    if capability is not None:
        request.add_header("X-Review-Capability", capability)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested_key for item in value.values() for nested_key in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested_key for item in value for nested_key in _all_keys(item)}
    return set()


def test_blinded_progress_keeps_assignment_order_and_only_own_answers(tmp_path):
    manifest = _seed_workspace(tmp_path)
    with _serve(manifest) as server:
        created = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("blinded", blinded=True),
        )[1]
        sidecar = server.open_review_sidecar()
        sidecar.capture_answer(
            created["rater_capabilities"]["rater-a"],
            "row-fail-first",
            "verdict",
            "fail",
        )
        sidecar.capture_answer(
            created["rater_capabilities"]["rater-b"],
            "row-pass-second",
            "verdict",
            "pass",
        )

        status, payload = _request(
            server,
            "GET",
            "/api/review/assignments/blinded/progress",
            capability=created["rater_capabilities"]["rater-a"],
        )

    assert status == 200
    progress = payload["progress"]
    assert [row["id"] for row in progress["assignment"]["rows"]] == [
        "row-fail-first",
        "row-pass-second",
    ]
    assert {answer["rater_slot_id"] for answer in progress["current_answers"]} == {
        "rater-a"
    }
    assert "outcome" not in _all_keys(progress)
    assert "outcome_source" not in _all_keys(progress)


def test_blinded_review_turn_payload_strips_system_outcomes(tmp_path):
    manifest = _seed_workspace(tmp_path)
    with _serve(manifest) as server:
        created = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("blinded", blinded=True),
        )[1]
        status, payload = _request(
            server,
            "GET",
            "/api/review/assignments/blinded/rows/row-fail-first/turn",
            capability=created["rater_capabilities"]["rater-a"],
        )

    assert status == 200
    turn = payload["turn"]
    assert turn["failure_reason"] == "command exited with code 1"
    assert turn["record"]["turn_output"]["command_outputs"] == [
        {"command": "run-check", "observation": "trace observation"}
    ]
    assert _all_keys(turn).isdisjoint(
        {
            "answer",
            "outcomeSource",
            "outcome",
            "outcome_source",
            "passed",
            "predicate_results",
            "predicate_verdict",
            "status",
            "success",
            "verdict",
        }
    )
    assert "system-default" not in json.dumps(payload)


def test_blinded_review_trace_payload_keeps_failure_trace_without_verdicts(tmp_path):
    manifest = _seed_workspace(tmp_path)
    with _serve(manifest) as server:
        created = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("blinded", blinded=True),
        )[1]
        status, payload = _request(
            server,
            "GET",
            "/api/review/assignments/blinded/rows/row-fail-first/trace",
            capability=created["rater_capabilities"]["rater-a"],
        )

    assert status == 200
    assert payload["spans"][0]["status"] == "error"
    assert payload["spans"][0]["attributes"] == {
        "raw_command": "run-check",
        "response_text": "failure observation: exit code 1",
    }
    assert _all_keys(payload).isdisjoint(
        {
            "evaluationResult",
            "final_answer",
            "predicate_results",
            "success",
            "verdict",
        }
    )


def test_non_blinded_review_evidence_retains_outcomes(tmp_path):
    manifest = _seed_workspace(tmp_path)
    with _serve(manifest) as server:
        created = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("open-review", blinded=False),
        )[1]
        capability = created["rater_capabilities"]["rater-a"]
        turn = _request(
            server,
            "GET",
            "/api/review/assignments/open-review/rows/row-pass-second/turn",
            capability=capability,
        )[1]["turn"]
        trace = _request(
            server,
            "GET",
            "/api/review/assignments/open-review/rows/row-pass-second/trace",
            capability=capability,
        )[1]["spans"]

    assert turn["status"] == "completed"
    assert turn["success"] == 1
    assert turn["answer"] == "system-default-pass"
    assert turn["record"]["turn_output"]["machine_predicate"] == {
        "predicate_verdict": "pass",
        "passed": True,
    }
    assert trace[0]["attributes"]["predicate_results"][0]["verdict"] == "pass"
