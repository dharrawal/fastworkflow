"""Phase 4 observability: fastWorkflow Chatbot debug mode (bead fix-kw7.5).

Server tests run against a real ObservabilityStore in tmp_path (design §6 —
no mocks for stores), seeded through the store's own write methods, with the
stdlib server started on an ephemeral port and exercised via urllib.

Covers the §3.4 invariants: token gate [R5], Host/Origin allowlist [R18],
restrictive CSP [R22], read-only API surface, artifact delivery, the
writer-health endpoint [R13], the prune/forget-channel CLI paths [R12][R21],
and the SPA packaging assertion [R23].
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow import observability_store as obs
from fastworkflow.cli import add_run_chatbot_parser
from fastworkflow.run_chatbot import server as run_chatbot_server


# ----------------------------------------------------------------------
# Seeded store + running server fixtures
# ----------------------------------------------------------------------

TURN1 = "20260825T120000-t1"  # completed, success, conversation 1
TURN2 = "20260825T120500-t2"  # failed, conversation 1
TURN3 = "20260825T121000-t3"  # awaiting_user, conversation-less [R17]
HTML_ARTIFACT_ID = "a" * 32
TEXT_ARTIFACT_ID = "b" * 32
HTML_PAYLOAD = "<html><body><script>alert(1)</script>hi</body></html>"
TEXT_PAYLOAD = "plain text artifact payload"


def _turn_row(
    turn_key: str,
    channel_id: str,
    conversation_id,
    ordinal,
    status: str,
    success: int,
    record: dict,
    failure_reason=None,
    completed_at="2026-08-25T12:01:00+00:00",
) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": conversation_id,
        "ordinal": ordinal,
        "user_message": f"user message for {turn_key}",
        "refined_user_message": None,
        "entry_workflow_name": "todo_list",
        "entry_context": "TodoList",
        "status": status,
        "success": success,
        "failure_reason": failure_reason,
        "answer": f"answer for {turn_key}" if status == "completed" else "",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-08-25T12:00:00+00:00",
        "completed_at": completed_at,
        "suspended_ms": 1500 if status == "awaiting_user" else 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps(record),
    }


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def seeded_db(workflow_path) -> str:
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()

    conv_id = store.mint_conversation_id("chan1")
    assert conv_id == 1
    store.record_conversation_label("chan1", conv_id, "Groceries", "About milk")

    record1 = {
        "turn_output": {
            "turn_key": TURN1,
            "status": "completed",
            "success": True,
            "command_outputs": [
                {
                    "command_name": "add_todo",
                    "command_response": {
                        "response": "done",
                        "success": True,
                        "artifacts": {
                            "note": "hello inline artifact",
                            "report": {
                                "__fw_artifact_ref__": HTML_ARTIFACT_ID,
                                "size": len(HTML_PAYLOAD),
                                "content_type": "text/html",
                                "content_encoding": None,
                                "error": None,
                            },
                            "log": {
                                "__fw_artifact_ref__": TEXT_ARTIFACT_ID,
                                "size": len(TEXT_PAYLOAD),
                                "content_type": "text/plain",
                                "content_encoding": None,
                                "error": None,
                            },
                        },
                    },
                }
            ],
        }
    }
    artifact_rows = [
        {
            "artifact_id": HTML_ARTIFACT_ID,
            "turn_key": TURN1,
            "channel_id": "chan1",
            "span_id": None,
            "key": "report",
            "content_type": "text/html",
            "size_bytes": len(HTML_PAYLOAD),
            "sha256": "x",
            "inline_value": HTML_PAYLOAD.encode(),
            "error": None,
        },
        {
            "artifact_id": TEXT_ARTIFACT_ID,
            "turn_key": TURN1,
            "channel_id": "chan1",
            "span_id": None,
            "key": "log",
            "content_type": "text/plain",
            "size_bytes": len(TEXT_PAYLOAD),
            "sha256": "y",
            "inline_value": TEXT_PAYLOAD.encode(),
            "error": None,
        },
    ]

    now_ns = time.time_ns()
    spans = [
        tracing.Span(
            span_id="s-root", trace_id=TURN1, name="fw.turn", kind="internal",
            channel_id="chan1", start_ns=now_ns, end_ns=now_ns + 5_000_000_000,
            status="ok", attributes={"user_message": "add milk"},
        ),
        tracing.Span(
            span_id="s-cmd", trace_id=TURN1, name="fw.command.execute",
            kind="tool", channel_id="chan1", command_name="add_todo",
            context="TodoList", parent_span_id="s-root",
            start_ns=now_ns + 1_000_000_000, end_ns=now_ns + 2_000_000_000,
            status="ok", attributes={"parameters": {"description": "milk"}},
        ),
        tracing.Span(
            span_id="s-ask", trace_id=TURN1, name="fw.ask_user",
            kind="human_wait", channel_id="chan1", parent_span_id="s-root",
            start_ns=now_ns + 2_000_000_000, end_ns=now_ns + 4_000_000_000,
            status="ok", attributes={"agent_query": "which list?"},
        ),
        # Open span on the in-progress turn (rendered honestly as open).
        tracing.Span(
            span_id="s-open", trace_id=TURN3, name="fw.ask_user",
            kind="human_wait", channel_id="chan2",
            start_ns=now_ns, end_ns=None, status="awaiting_user",
            attributes={"agent_query": "still waiting"},
        ),
    ]

    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            _turn_row(TURN1, "chan1", 1, 1, "completed", 1, record1),
            artifact_rows,
            redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                TURN2, "chan1", 1, 2, "failed", 0,
                {"turn_output": {"turn_key": TURN2, "success": False}},
                failure_reason="command exploded",
            ),
            [],
            redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                TURN3, "chan2", None, None, "awaiting_user", 0,
                {"turn_output": {"turn_key": TURN3, "success": False}},
                completed_at=None,
            ),
            [],
            redactor,
        )
        store.upsert_span_rows(conn, spans, redactor)
        store.set_diagnostic(
            conn,
            "writer_health",
            {"spans_dropped": 2, "records_dropped": 1, "write_errors": 3,
             "busy_retries": 0, "refused_terminal_writes": 0,
             "last_error": "disk full"},
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def server(seeded_db, workflow_path):
    srv = run_chatbot_server.ChatbotServer(seeded_db, workflow_path=workflow_path, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _get(server, path, token=..., headers=None):
    """GET helper returning (status, headers, body_bytes); never raises on 4xx."""
    if token is ...:
        token = server.token
    url = f"http://127.0.0.1:{server.port}{path}"
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def _get_json(server, path, **kwargs):
    status, headers, body = _get(server, path, **kwargs)
    assert status == 200, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


def _row_counts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("conversations", "turns", "spans", "artifacts", "feedback")
        }
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Access control [R5][R18]
# ----------------------------------------------------------------------


class TestAccessControl:
    def test_missing_token_is_401(self, server):
        status, _, body = _get(server, "/api/channels", token=None)
        assert status == 401
        assert b"token" in body

    def test_wrong_token_is_401(self, server):
        status, _, _ = _get(server, "/api/channels", token="not-the-token")
        assert status == 401
        status, _, _ = _get(server, "/", token="not-the-token")
        assert status == 401

    def test_token_accepted_via_query_param(self, server):
        status, _, body = _get(
            server, f"/api/channels?token={server.token}", token=None
        )
        assert status == 200
        assert json.loads(body)["channels"]

    def test_bad_host_is_403(self, server):
        status, _, body = _get(
            server, "/api/channels", headers={"Host": "evil.example.com"}
        )
        assert status == 403
        assert b"forbidden" in body

    def test_bad_origin_is_403(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Origin": "http://evil.example.com"}
        )
        assert status == 403

    def test_localhost_host_allowed(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Host": f"localhost:{server.port}"}
        )
        assert status == 200

    def test_forwarded_loopback_port_allowed(self, server):
        # WSL relays and IDE port forwards re-expose the server on a DIFFERENT
        # local port; the browser's Host names that port. Loopback hosts pass
        # regardless of port — the token stays the authentication [R18].
        for host in (
            f"127.0.0.1:{server.port + 1}",
            "localhost:9999",
            f"[::1]:{server.port}",
            "127.0.0.1",
        ):
            status, _, _ = _get(server, "/api/channels", headers={"Host": host})
            assert status == 200, host

    def test_loopback_origin_any_port_allowed_https_and_null_refused(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Origin": "http://localhost:9999"}
        )
        assert status == 200
        for origin in (f"https://127.0.0.1:{server.port}", "null"):
            status, _, _ = _get(
                server, "/api/channels", headers={"Origin": origin}
            )
            assert status == 403, origin

    def test_403_body_names_the_offending_host(self, server):
        status, _, body = _get(
            server, "/api/channels", headers={"Host": "evil.example.com"}
        )
        assert status == 403
        assert b"evil.example.com" in body

    def test_url_embeds_token(self, server):
        assert server.url.startswith(f"http://127.0.0.1:{server.port}/?token=")
        assert server.token in server.url


# ----------------------------------------------------------------------
# SPA page + CSP [R22]
# ----------------------------------------------------------------------


class TestPage:
    def test_index_served_with_restrictive_csp(self, server):
        status, headers, body = _get(server, "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"fastWorkflow Chatbot" in body
        csp = headers["Content-Security-Policy"]
        assert "default-src 'none'" in csp
        assert "script-src 'self'" in csp
        assert "connect-src 'self'" in csp
        assert "frame-src 'self'" in csp
        # The page's inline script is hash-sourced, not 'unsafe-inline'.
        assert "'sha256-" in csp
        assert "script-src 'self' 'unsafe-inline'" not in csp

    def test_page_never_uses_innerhtml(self, server):
        # [R22]: record-derived text renders via textContent only.
        assert b"innerHTML" not in server.index_html
        assert b'id="tabTurns"' not in server.index_html
        assert b'id="tabConvs"' not in server.index_html
        assert b'id="clearConvsBtn"' in server.index_html
        # The rail nests channel > conversation > turns, with no channel picker.
        assert b'id="channelSel"' not in server.index_html
        assert b"/api/channels" not in server.index_html
        assert b"chanGroup" in server.index_html
        assert b"renderConversationGroups" in server.index_html
        assert b"renderNestedTurns" in server.index_html
        # Empty nodes are hidden at both levels: conversations with no turns,
        # and channels left with no conversation.
        assert b"channelHasConversation" in server.index_html

    def test_unknown_path_404(self, server):
        status, _, _ = _get(server, "/nope")
        assert status == 404
        status, _, _ = _get(server, "/api/nope")
        assert status == 404


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------


class TestApi:
    def test_meta(self, server, seeded_db, workflow_path):
        meta = _get_json(server, "/api/meta")
        assert meta["db_path"] == seeded_db
        assert meta["workflow_name"] == "my_workflow"
        assert meta["db_size_bytes"] > 0

    def test_channels(self, server):
        assert _get_json(server, "/api/channels")["channels"] == ["chan1", "chan2"]

    def test_conversations_with_labels(self, server):
        convs = _get_json(server, "/api/conversations?channel=chan1")["conversations"]
        assert len(convs) == 1
        assert convs[0]["topic"] == "Groceries"
        assert convs[0]["summary"] == "About milk"
        assert convs[0]["last_turn_at"]  # stamped by the turn upserts

    def test_turns_by_conversation(self, server):
        turns = _get_json(
            server, "/api/turns?channel=chan1&conversation=1"
        )["turns"]
        assert [t["turn_key"] for t in turns] == [TURN2, TURN1]
        assert all("record_json" not in t for t in turns)

    def test_turns_filter_status(self, server):
        turns = _get_json(server, "/api/turns?status=awaiting_user")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN3]
        assert turns[0]["conversation_id"] is None  # turns-first view row [R17]

    def test_turns_filter_success(self, server):
        turns = _get_json(server, "/api/turns?success=1")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN1]
        turns = _get_json(server, "/api/turns?success=0")["turns"]
        assert {t["turn_key"] for t in turns} == {TURN2, TURN3}

    def test_turns_filter_command(self, server):
        turns = _get_json(server, "/api/turns?command=add_todo")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN1]
        assert _get_json(server, "/api/turns?command=nonexistent")["turns"] == []

    def test_turn_detail_parses_record_json(self, server):
        turn = _get_json(server, f"/api/turn/{TURN1}")["turn"]
        assert "record_json" not in turn
        record = turn["record"]
        outputs = record["turn_output"]["command_outputs"]
        assert outputs[0]["command_name"] == "add_todo"
        artifacts = outputs[0]["command_response"]["artifacts"]
        assert artifacts["note"] == "hello inline artifact"
        assert artifacts["report"]["__fw_artifact_ref__"] == HTML_ARTIFACT_ID
        assert turn["failure_reason"] is None
        assert turn["status"] == "completed"

    def test_turn_detail_404(self, server):
        status, _, _ = _get(server, "/api/turn/no-such-turn")
        assert status == 404

    def test_spans_for_turn(self, server):
        spans = _get_json(server, f"/api/spans/{TURN1}")["spans"]
        assert [s["name"] for s in spans] == [
            "fw.turn", "fw.command.execute", "fw.ask_user"
        ]
        by_name = {s["name"]: s for s in spans}
        assert by_name["fw.ask_user"]["kind"] == "human_wait"
        assert by_name["fw.command.execute"]["command_name"] == "add_todo"
        # Attributes come back parsed, not as a JSON string.
        assert by_name["fw.command.execute"]["attributes"]["parameters"] == {
            "description": "milk"
        }

    def test_open_span_kept_open(self, server):
        spans = _get_json(server, f"/api/spans/{TURN3}")["spans"]
        assert spans[0]["end_ns"] is None
        assert spans[0]["status"] == "awaiting_user"

    def test_text_artifact_served(self, server):
        status, headers, body = _get(server, f"/api/artifact/{TEXT_ARTIFACT_ID}")
        assert status == 200
        assert headers["Content-Type"] == "text/plain"
        assert body == TEXT_PAYLOAD.encode()

    def test_html_artifact_sandboxed(self, server):
        status, headers, body = _get(server, f"/api/artifact/{HTML_ARTIFACT_ID}")
        assert status == 200
        assert headers["Content-Type"] == "text/html"
        assert body == HTML_PAYLOAD.encode()
        # Direct navigation to the artifact URL must be inert [R22].
        assert headers["Content-Security-Policy"] == "default-src 'none'; sandbox"
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_artifact_404(self, server):
        status, _, _ = _get(server, "/api/artifact/" + "f" * 32)
        assert status == 404

    def test_health_reflects_diagnostics(self, server):
        health = _get_json(server, "/api/health")["writer_health"]
        assert health["spans_dropped"] == 2
        assert health["records_dropped"] == 1
        assert health["write_errors"] == 3
        assert health["last_error"] == "disk full"
        assert health["updated_at"]

    def test_clear_all_conversations_requires_explicit_confirmation(
        self, server, seeded_db
    ):
        status, body = _post(server, "/api/clear_conversations", {})
        assert status == 400
        assert "confirmation required" in body["error"]
        assert _row_counts(seeded_db)["turns"] == 3

    def test_clear_all_conversations_removes_turn_observability(
        self, server, seeded_db
    ):
        status, body = _post(
            server,
            "/api/clear_conversations",
            {"confirm": "clear all conversations"},
        )
        assert status == 200
        assert body["deleted"]["turns"] == 3
        assert body["deleted"]["conversations"] == 1
        assert _row_counts(seeded_db) == {
            "conversations": 0,
            "turns": 0,
            "spans": 0,
            "artifacts": 0,
            "feedback": 0,
        }
        # Clearing data never rewinds conversation identity.
        assert obs.ObservabilityStore(seeded_db).mint_conversation_id("chan1") == 2


# ----------------------------------------------------------------------
# Read-only guarantee
# ----------------------------------------------------------------------


class TestReadOnly:
    def test_non_get_methods_rejected(self, server, seeded_db):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/api/turns",
                method=method,
                data=b"{}",
                headers={"Authorization": f"Bearer {server.token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
            except urllib.error.HTTPError as err:
                status = err.code
            assert status == 405

    def test_gets_do_not_mutate_state(self, server, seeded_db):
        before = _row_counts(seeded_db)
        for path in (
            "/", "/api/meta", "/api/channels", "/api/conversations",
            "/api/turns", f"/api/turn/{TURN1}", f"/api/spans/{TURN1}",
            f"/api/artifact/{TEXT_ARTIFACT_ID}", "/api/health",
        ):
            status, _, _ = _get(server, path)
            assert status == 200
        assert _row_counts(seeded_db) == before


# ----------------------------------------------------------------------
# CLI paths: internal maintenance helpers and the browser-owned launch surface
# ----------------------------------------------------------------------


class TestCliPaths:
    def _parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        add_run_chatbot_parser(subparsers)
        return parser

    def test_run_forget_channel(self, seeded_db):
        deleted = run_chatbot_server.run_forget_channel(seeded_db, "chan1")
        assert deleted["turns"] == 2
        assert deleted["conversations"] == 1
        assert deleted["spans"] == 3
        assert deleted["artifacts"] == 2
        counts = _row_counts(seeded_db)
        assert counts["turns"] == 1  # chan2's turn survives
        assert counts["artifacts"] == 0

    def test_run_prune_returns_counts(self, seeded_db):
        deleted = run_chatbot_server.run_prune(seeded_db)
        assert set(deleted) == {"spans", "artifacts"}
        # Everything seeded is recent; nothing crosses the retention horizon.
        assert deleted["spans"] == 0
        counts = _row_counts(seeded_db)
        assert counts["spans"] == 4

    @pytest.mark.parametrize("flag", ["--prune", "--forget-channel"])
    def test_run_chatbot_cli_rejects_removed_maintenance_flags(self, flag):
        args = ["run_chatbot", flag]
        if flag == "--forget-channel":
            args.append("chan2")
        with pytest.raises(SystemExit):
            self._parser().parse_args(args)

    def test_workflow_path_is_rejected_even_when_the_db_is_missing(self, tmp_path):
        # A missing DB is a normal cold start for the chatbot itself (the
        # auto-spawned server creates it on the first turn). Workflow selection
        # now belongs to the browser, regardless of whether that DB exists.
        wf = tmp_path / "never_ran"
        wf.mkdir()
        with pytest.raises(SystemExit):
            self._parser().parse_args(["run_chatbot", str(wf)])

    def test_run_chatbot_main_opens_the_picker(self, monkeypatch, capsys):
        monkeypatch.setattr(
            run_chatbot_server.ChatbotServer, "serve_forever", lambda self: None
        )
        monkeypatch.setattr(
            run_chatbot_server.ChatbotServer,
            "shutdown",
            lambda self: self.httpd.server_close(),
        )
        monkeypatch.setattr(signal, "signal", lambda *_args: None)
        rc = run_chatbot_server.run_chatbot_main(
            SimpleNamespace(
                server_port=8000,
                expect_encrypted_jwt=False,
            )
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "pick a workflow in the browser" in out

    def test_open_in_browser_is_a_noop_under_pytest(self, monkeypatch):
        # The pytest skip is env-based, not a CLI flag: a test that drives
        # run_chatbot_main must not pop a browser, and must not need --no-browser.
        import os
        import sys

        opened = []

        class _FakeBrowser:
            def open(self, url):
                opened.append(url)

        monkeypatch.setitem(sys.modules, "webbrowser", _FakeBrowser())
        assert "PYTEST_CURRENT_TEST" in os.environ
        run_chatbot_server._open_in_browser("http://127.0.0.1:1/?token=x")
        assert opened == []


# ----------------------------------------------------------------------
# Packaging [R23]
# ----------------------------------------------------------------------


class TestPackaging:
    def test_spa_ships_as_package_data(self):
        import importlib.resources

        resource = (
            importlib.resources.files("fastworkflow.run_chatbot") / "static" / "index.html"
        )
        assert resource.is_file()
        page = resource.read_bytes()
        assert b"fastWorkflow Chatbot" in page
        # Self-contained: no external origins anywhere in the page. Test mode
        # legitimately names loopback origins (the local FastAPI server), so
        # every http(s):// occurrence must be 127.0.0.1 or localhost [R19].
        import re

        for match in re.findall(rb"https?://[^\s\"'`<>)]*", page):
            assert match.startswith(
                (b"http://127.0.0.1", b"http://localhost")
            ), f"non-loopback origin referenced by the SPA: {match!r}"
        assert b"https://" not in page  # loopback is always plain http
        assert b"innerHTML" not in page
        assert b"startTrain" in page
        assert "Training…".encode() in page

    def test_legend_chips_sit_inline_with_labels(self):
        """Waterfall legend color chips must sit next to their span-type labels.

        Chips used to reuse `.wfBar` (`position: absolute`), which pinned them
        to the viewport instead of the legend (fix-kw7.14).
        """
        page = (
            Path(__file__).parent.parent
            / "fastworkflow"
            / "run_chatbot"
            / "static"
            / "index.html"
        ).read_text()
        assert 'el("span", "chip wfBar "' not in page
        assert 'el("span", "chip " + pair[0])' in page
        chip_rule = page.split(".legend .chip {", 1)[1].split("}", 1)[0]
        assert "position: static" in chip_rule
        # Category colors apply without requiring .wfBar, so chips keep their fill.
        assert ".cat-turn { background: var(--bar-turn); }" in page
        assert ".wfBar.cat-turn" not in page

    def test_pyproject_includes_spa(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        includes = data["tool"]["poetry"]["include"]
        assert "fastworkflow/run_chatbot/static/index.html" in includes

    def test_server_module_is_stdlib_only(self):
        # [R23]: debug mode must work on a base install (no [server] extra).
        import ast

        tree = ast.parse(Path(run_chatbot_server.__file__).read_text())
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        for forbidden in ("fastapi", "uvicorn", "starlette", "flask", "aiohttp"):
            assert forbidden not in imported_roots, (
                f"chatbot server imports {forbidden}"
            )


# ----------------------------------------------------------------------
# Read-only viewer discipline: the debug layer never creates or writes the
# DB it inspects (epic review follow-up; [R12] + module invariant).
# ----------------------------------------------------------------------


class TestReadOnlyViewer:
    def test_viewer_does_not_create_a_missing_db(self, workflow_path):
        db_path = state_paths.observability_db(workflow_path)
        assert not Path(db_path).exists()
        srv = run_chatbot_server.ChatbotServer(
            db_path, workflow_path=workflow_path, port=0
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            meta = _get_json(srv, "/api/meta")
            assert meta["db_available"] is False
            assert _get_json(srv, "/api/turns")["turns"] == []
            assert _get_json(srv, "/api/conversations")["conversations"] == []
            assert _get_json(srv, "/api/health")["db_available"] is False
        finally:
            srv.shutdown()
            thread.join(timeout=5)
        # The whole point: inspecting must not have created the file.
        assert not Path(db_path).exists()

    def test_viewer_opens_a_read_only_snapshot(self, seeded_db):
        import os as _os

        # Checkpoint so the snapshot has no -wal sidecar, then drop write perms:
        # a post-mortem copy the developer does not own must still open.
        conn = sqlite3.connect(seeded_db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        _os.chmod(seeded_db, 0o400)
        try:
            srv = run_chatbot_server.ChatbotServer(seeded_db, port=0)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                turns = _get_json(srv, "/api/turns")["turns"]
                assert len(turns) == 3
                assert _get_json(srv, "/api/meta")["db_available"] is True
            finally:
                srv.shutdown()
                thread.join(timeout=5)
        finally:
            _os.chmod(seeded_db, 0o600)


class TestServerSideContextFilter:
    def test_context_filter_is_applied_by_the_server(self, server):
        # Seeded turns carry entry_context "TodoList"; the match is a
        # case-insensitive substring, applied in SQL so it reaches rows older
        # than one page.
        turns = _get_json(server, "/api/turns?context=todolist")["turns"]
        assert len(turns) == 3
        assert _get_json(server, "/api/turns?context=nomatch")["turns"] == []
        # LIKE metacharacters in the filter are literals, not wildcards.
        assert _get_json(server, "/api/turns?context=%25")["turns"] == []


class TestForgetChannelLegacyErasure:
    def test_forget_channel_deletes_legacy_conversation_db(
        self, seeded_db, workflow_path
    ):
        # Ruling C1: during the Phase-A dual-write window, erasure must also
        # remove the legacy per-channel conversations/<channel_id>.sqlite3.
        legacy_dir = Path(state_paths.conversations_dir(workflow_path))
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy = legacy_dir / "chan1.sqlite3"
        legacy.write_bytes(b"legacy payload")
        (legacy_dir / "chan1.sqlite3-wal").write_bytes(b"wal")
        deleted = run_chatbot_server.run_forget_channel(
            seeded_db, "chan1", workflow_path
        )
        assert deleted["legacy_conversation_db_files"] == 2
        assert not legacy.exists()
        assert not (legacy_dir / "chan1.sqlite3-wal").exists()

    def test_forget_channel_refuses_path_traversal_channel_ids(
        self, seeded_db, workflow_path, tmp_path
    ):
        outside = tmp_path / "outside.sqlite3"
        outside.write_bytes(b"do not delete")
        deleted = run_chatbot_server.run_forget_channel(
            seeded_db, "../../outside", workflow_path
        )
        assert "legacy_conversation_db_files" not in deleted
        assert outside.exists()


# ----------------------------------------------------------------------
# Control plane: /api/session, workflow picker, POST /api/select_workflow
# ----------------------------------------------------------------------


def _post(server, path, body, token=...):
    if token is ...:
        token = server.token
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


class TestControlPlane:
    def test_session_reports_managed_identity(self, server):
        s = _get_json(server, "/api/session")["session"]
        # Single-user tool: the channel is chatbot-managed, never typed, and
        # fixed across launches so restarts share one history.
        assert s["channel_id"] == "chatbot"
        assert s["user_id"] == "developer"
        assert s["workflow_name"] == "my_workflow"
        assert s["server_running"] is False  # fixtures never spawn

    def test_workflow_candidates_include_bundled_examples(self, server):
        import fastworkflow

        wfs = _get_json(server, "/api/workflows")["workflows"]
        bundled = str(
            Path(fastworkflow.__file__).resolve().parent / "examples" / "hello_world"
        )
        hello = next(w for w in wfs if w["path"] == bundled)
        assert hello["source"] == "examples"
        assert hello["name"] == "hello_world"
        assert hello["trainable"] is False
        assert "training" in hello
        assert "rel" in hello
        assert Path(hello["path"]).is_dir()

    def test_nested_local_workflows_carry_rel_paths(self, tmp_path, monkeypatch):
        cwd = tmp_path / "proj"
        (cwd / "top_wf" / "_commands").mkdir(parents=True)
        (cwd / "apps" / "team" / "deep_wf" / "_commands").mkdir(parents=True)
        monkeypatch.chdir(cwd)
        wfs = run_chatbot_server.list_workflow_candidates()
        local = {w["name"]: w for w in wfs if w["source"] == "local"}
        assert local["top_wf"]["rel"] == "top_wf"
        assert local["deep_wf"]["rel"] == "apps/team/deep_wf"

    def test_parent_of_nested_workflow_is_omitted(self, tmp_path, monkeypatch):
        cwd = tmp_path / "proj"
        (cwd / "pkg" / "_commands").mkdir(parents=True)
        (cwd / "pkg" / "apps" / "child_wf" / "_commands").mkdir(parents=True)
        monkeypatch.chdir(cwd)
        wfs = run_chatbot_server.list_workflow_candidates()
        local = {w["name"]: w for w in wfs if w["source"] == "local"}
        assert "child_wf" in local
        assert "pkg" not in local
        assert local["child_wf"]["rel"] == "pkg/apps/child_wf"

    def test_rel_under_rejects_paths_outside_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        inside = root / "apps" / "wf"
        assert run_chatbot_server._rel_under(str(inside), str(root)) == "apps/wf"
        assert run_chatbot_server._rel_under(str(root), str(root)) == ""
        assert run_chatbot_server._rel_under(str(tmp_path), str(root)) == ""

    def test_browse_lists_directories_flagging_workflows(self, server):
        import urllib.parse

        import fastworkflow

        examples = str(
            Path(fastworkflow.__file__).resolve().parent / "examples"
        )
        data = _get_json(
            server, "/api/browse?dir=" + urllib.parse.quote(examples)
        )
        assert data["dir"] == examples
        by_name = {e["name"]: e for e in data["entries"]}
        assert by_name["hello_world"]["is_workflow"] is True

    def test_select_workflow_requires_token(self, server, tmp_path):
        status, body = _post(
            server, "/api/select_workflow", {"path": str(tmp_path)}, token=None
        )
        assert status == 401

    def test_select_workflow_switches_the_active_db(self, server, tmp_path):
        other = tmp_path / "other_wf"
        (other / "_commands").mkdir(parents=True)
        status, body = _post(
            server, "/api/select_workflow", {"path": str(other)}
        )
        assert status == 200
        s = body["session"]
        assert s["workflow_path"] == str(other)
        assert s["server_running"] is False  # no_server fixtures
        assert server.db_path.endswith("observability.sqlite3")
        assert "other_wf" in server.db_path

    def test_select_workflow_rejects_non_workflow_dirs(self, server, tmp_path):
        plain = tmp_path / "not_a_workflow"
        plain.mkdir()
        status, body = _post(
            server, "/api/select_workflow", {"path": str(plain)}
        )
        assert status == 400
        assert "_commands" in body["error"]

    def test_missing_env_files_are_configured_from_templates(
        self, workflow_path, tmp_path, monkeypatch
    ):
        wf = tmp_path / "needs_env"
        (wf / "_commands").mkdir(parents=True)
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.launcher.missing_server_packages",
            lambda: ["fastapi"],
        )
        srv = run_chatbot_server.ChatbotServer(
            workflow_path=str(wf),
            port=0,
            spawn_options={"no_server": False},
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            session = srv.activate_workflow(str(wf))
            assert session["env_setup_required"] is True
            assert session["server_running"] is False

            status, body = _post(
                srv, "/api/configure_env", {"create_from_templates": True}
            )
            assert status == 200
            assert body["session"]["env_setup_required"] is False
            assert (wf / "fastworkflow.env").is_file()
            assert (wf / "fastworkflow.passwords.env").is_file()
            assert "LLM_AGENT" in (wf / "fastworkflow.env").read_text()
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_selected_env_file_contents_are_copied_locally(self, tmp_path):
        wf = tmp_path / "uploaded_env"
        (wf / "_commands").mkdir(parents=True)
        srv = run_chatbot_server.ChatbotServer(
            workflow_path=str(wf),
            port=0,
            spawn_options={"no_server": True},
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            status, _body = _post(
                srv,
                "/api/configure_env",
                {
                    "env_content": "LLM_AGENT=test/model\n",
                    "passwords_content": "LITELLM_API_KEY_AGENT=test-secret\n",
                },
            )
            assert status == 200
            env_file = wf / "fastworkflow.env"
            passwords_file = wf / "fastworkflow.passwords.env"
            assert env_file.read_text() == "LLM_AGENT=test/model\n"
            assert (
                passwords_file.read_text()
                == "LITELLM_API_KEY_AGENT=test-secret\n"
            )
            assert env_file.stat().st_mode & 0o777 == 0o600
            assert passwords_file.stat().st_mode & 0o777 == 0o600
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_other_posts_are_still_405(self, server):
        status, _ = _post(server, "/api/turns", {})
        assert status == 405
        status, _ = _post(server, "/api/channels", {})
        assert status == 405


# ----------------------------------------------------------------------
# Distillation review surface (`fix-sb8.6`, `fix-sb8.9`, `fix-sb8.10`,
# `fix-sb8.12`, `fix-sb8.13`) — §12.1's route inventory [DR55].
# ----------------------------------------------------------------------

DISTILL_TURN = "20260826T090000-d1"


@pytest.fixture
def distilled_db(seeded_db):
    """A `[DR54]`-shaped corpus on top of the seeded DB.

    Four runs — one comparable and citing an action divergence, one
    non-comparable with the same command and kind, one replay of the first,
    and one comparable run that agreed — plus a run-level divergence with a
    NULL `command_name`. That is the minimum fixture `[DR54]` will accept as
    evidence that a provenance query is right rather than merely parseable.
    """
    store = obs.ObservabilityStore(seeded_db)
    redactor = obs.Redactor()

    def run(conn, run_id, **overrides):
        payload = {
            "run_id": run_id,
            "turn_key": DISTILL_TURN if run_id == "runA" else f"t-{run_id}",
            "channel_id": "chan1",
            "conversation_id": 1,
            "user_message": "finish my laundry task",
            "workflow_name": "todo_list",
            "entry_context": "TodoList",
            "comparable": 1,
            "isolation_verified": 1,
            "started_at": "2026-08-26T09:00:00+00:00",
            "completed_at": "2026-08-26T09:00:20+00:00",
            "exec_diverged": 1,
            "material_divergences": 1,
            "execution_insights": 1,
            "run_json": json.dumps({"run_id": run_id, "status": "ok"}),
        }
        payload.update(overrides)
        store.upsert_distillation_row(conn, "run", payload, redactor)

    def divergence(conn, divergence_id, run_id, **overrides):
        payload = {
            "divergence_id": divergence_id,
            "run_id": run_id,
            "level": "action",
            "left_pass": "teacher",
            "right_pass": "student",
            "align_index": 0,
            "kind": "missing-in-student",
            "material": 1,
            "command_key": "TodoList/complete_task",
            "command_name": "complete_task",
            "left_span_id": "sd-teacher",
            "right_span_id": "sd-student",
            "param_diff_json": json.dumps({"task_id": {"left": 1, "right": 2}}),
            "detail_json": json.dumps({"left": {}, "right": {}}),
        }
        payload.update(overrides)
        store.upsert_distillation_row(conn, "divergence", payload, redactor)

    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        # The user-visible turn the run belongs to: the marker is what the SPA
        # renders on the turn list, so it has to be a real turn row.
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                DISTILL_TURN, "chan1", 1, 3, "completed", 1,
                {"turn_output": {"turn_key": DISTILL_TURN, "success": True}},
            ),
            [],
            redactor,
        )
        run(conn, "runA")
        run(conn, "runB", comparable=0, comparable_reason="fingerprint-differs")
        run(conn, "runC", replay_of="runA", replay_trace_id=f"{DISTILL_TURN}~replay.1")
        run(conn, "runD", exec_diverged=0, material_divergences=0, execution_insights=0)
        divergence(conn, "dA", "runA")
        divergence(conn, "dB", "runB")
        divergence(conn, "dC", "runC")
        # A run-level divergence keys on nothing: no command, and no step pair
        # to point at either — it is about the two passes' final ANSWERS.
        divergence(
            conn, "dRun", "runA",
            level="run", kind="different-answer-same-actions",
            command_key=None, command_name=None, align_index=1,
            left_span_id=None, right_span_id=None,
        )
        for run_id in ("runA", "runD"):
            for label, seq in (("teacher", 0), ("student", 1)):
                store.upsert_distillation_row(
                    conn,
                    "pass",
                    {
                        "run_id": run_id,
                        "pass_label": label,
                        "role": label,
                        "seq": seq,
                        "trace_id": (
                            DISTILL_TURN if run_id == "runA" else f"t-{run_id}"
                        ),
                        "agent_model": f"model-{label}",
                        "entry_fingerprint": "fp-entry",
                        "exit_fingerprint": f"fp-exit-{label}",
                        "wall_ms": 500,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "cache_hits": 0,
                        "cache_misses": 2,
                        "model_params_json": json.dumps({"temperature": 0}),
                        "entry_inputs_json": json.dumps({"user_message": "x"}),
                    },
                    redactor,
                )
        for insight_id, divergence_id in (("ins-1", "dA"), ("ins-2", "dRun")):
            store.upsert_distillation_row(
                conn,
                "insight",
                {
                    "insight_id": insight_id,
                    "run_id": "runA",
                    "kind": "execution",
                    "text": f"text for {insight_id}",
                    "text_hash": f"hash-{insight_id}",
                    "created_at": "2026-08-26T09:00:30+00:00",
                },
                redactor,
            )
            store.upsert_distillation_row(
                conn,
                "citation",
                {"insight_id": insight_id, "divergence_id": divergence_id},
                redactor,
            )
        store.upsert_span_rows(
            conn,
            [
                tracing.Span(
                    span_id="sd-teacher", trace_id=DISTILL_TURN,
                    name="fw.command.execute", command_name="complete_task",
                    start_ns=1, end_ns=2, status="ok",
                    distillation_pass="teacher",
                ),
                tracing.Span(
                    span_id="sd-student", trace_id=DISTILL_TURN,
                    name="fw.command.execute", command_name="add_todo",
                    start_ns=3, end_ns=4, status="ok",
                    distillation_pass="student",
                ),
                tracing.Span(
                    span_id="sd-turn", trace_id=DISTILL_TURN,
                    name="fw.turn", start_ns=0, end_ns=9, status="ok",
                ),
                # runD's student DID run the cited command and did not diverge:
                # the contradiction case.
                tracing.Span(
                    span_id="sd-agree", trace_id="t-runD",
                    name="fw.command.execute", command_name="complete_task",
                    start_ns=1, end_ns=2, status="ok",
                    distillation_pass="student",
                ),
                # A NULL-command failed span, the [DR54] three-valued-logic trap.
                tracing.Span(
                    span_id="sd-null", trace_id="t-runD",
                    name="fw.command.execute", start_ns=3, end_ns=4,
                    status="error", distillation_pass="student",
                ),
            ],
            redactor,
        )
        conn.commit()
    finally:
        conn.close()
    return seeded_db


@pytest.fixture
def distill_server(distilled_db, workflow_path):
    srv = run_chatbot_server.ChatbotServer(
        distilled_db, workflow_path=workflow_path, port=0
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


class TestDistillationReadApi:
    def test_run_list_excludes_replays_by_default(self, distill_server):
        runs = _get_json(distill_server, "/api/distillation/runs")["runs"]
        assert [r["run_id"] for r in runs] == ["runA", "runB", "runD"]
        with_replays = _get_json(
            distill_server, "/api/distillation/runs?include_replays=1"
        )["runs"]
        assert "runC" in {r["run_id"] for r in with_replays}

    def test_run_list_filters(self, distill_server):
        assert [
            r["run_id"]
            for r in _get_json(
                distill_server, "/api/distillation/runs?comparable=0"
            )["runs"]
        ] == ["runB"]
        assert [
            r["run_id"]
            for r in _get_json(
                distill_server, "/api/distillation/runs?diverged=0"
            )["runs"]
        ] == ["runD"]
        assert [
            r["run_id"]
            for r in _get_json(
                distill_server, "/api/distillation/runs?channel=nobody"
            )["runs"]
        ] == []

    def test_run_detail_carries_passes_and_retention(self, distill_server):
        payload = _get_json(distill_server, "/api/distillation/run/runA")
        assert payload["run"]["run_id"] == "runA"
        # run_json is parsed server-side, the /api/turn/<k> precedent.
        assert payload["run"]["record"]["status"] == "ok"
        assert [(p["pass_label"], p["seq"]) for p in payload["passes"]] == [
            ("teacher", 0),
            ("student", 1),
        ]
        assert payload["passes"][0]["model_params"] == {"temperature": 0}
        # `fix-sb8.13`: "why is this trace still here", answered.
        assert payload["retention"]["pin_class"] == "produced-an-insight"
        assert payload["retention"]["pin_expected"] is True

    def test_run_detail_404s_for_an_unknown_run(self, distill_server):
        status, _, _ = _get(distill_server, "/api/distillation/run/nope")
        assert status == 404

    def test_divergences_are_ordered_and_parsed(self, distill_server):
        rows = _get_json(
            distill_server, "/api/distillation/divergences/runA"
        )["divergences"]
        assert [r["divergence_id"] for r in rows] == ["dA", "dRun"]
        assert rows[0]["param_diff"] == {"task_id": {"left": 1, "right": 2}}
        only_run = _get_json(
            distill_server, "/api/distillation/divergences/runA?level=run"
        )["divergences"]
        assert [r["divergence_id"] for r in only_run] == ["dRun"]

    def test_insights_resolve_provenance_in_both_directions(self, distill_server):
        forward = _get_json(distill_server, "/api/distillation/insights?run=runA")
        assert {i["insight_id"] for i in forward["insights"]} == {"ins-1", "ins-2"}
        assert {c["divergence_id"] for c in forward["citations"]} == {"dA", "dRun"}

        reverse = _get_json(
            distill_server, "/api/distillation/insights?insight=ins-1"
        )
        # [DR54]: the non-comparable run and the replay are BOTH excluded.
        assert {r["run_id"] for r in reverse["runs"]["support"]} == {"runA"}
        assert {r["run_id"] for r in reverse["runs"]["contradict"]} == {"runD"}

        # A run-level insight has no command to key on: its contradiction set
        # is defined over outcomes, and it must not silently return nothing.
        run_level = _get_json(
            distill_server, "/api/distillation/insights?insight=ins-2"
        )
        assert {
            r["run_id"] for r in run_level["runs"]["contradict_run_level"]
        } == {"runD"}

        by_hash = _get_json(
            distill_server, "/api/distillation/insights?text_hash=hash-ins-1"
        )
        assert [i["insight_id"] for i in by_hash["insights"]] == ["ins-1"]

    def test_insights_requires_exactly_one_selector(self, distill_server):
        status, _, _ = _get(distill_server, "/api/distillation/insights")
        assert status == 400
        status, _, _ = _get(
            distill_server, "/api/distillation/insights?run=runA&insight=ins-1"
        )
        assert status == 400

    def test_spans_can_be_filtered_to_one_pass(self, distill_server):
        """The whole of "neither waterfall interleaves the other": the passes
        share a trace_id by [DR1] and are separated by the column."""
        every = _get_json(distill_server, f"/api/spans/{DISTILL_TURN}")["spans"]
        assert len(every) == 3
        teacher = _get_json(
            distill_server, f"/api/spans/{DISTILL_TURN}?pass=teacher"
        )["spans"]
        assert [s["span_id"] for s in teacher] == ["sd-teacher"]
        student = _get_json(
            distill_server, f"/api/spans/{DISTILL_TURN}?pass=student"
        )["spans"]
        assert [s["span_id"] for s in student] == ["sd-student"]
        outside = _get_json(
            distill_server, f"/api/spans/{DISTILL_TURN}?pass=none"
        )["spans"]
        assert [s["span_id"] for s in outside] == ["sd-turn"]

    def test_the_turn_list_marks_distillation_turns(self, distill_server):
        turns = _get_json(distill_server, "/api/turns")["turns"]
        marked = {
            t["turn_key"]: t.get("distillation")
            for t in turns
            if t.get("distillation")
        }
        assert list(marked) == [DISTILL_TURN]
        assert marked[DISTILL_TURN]["run_id"] == "runA"
        assert marked[DISTILL_TURN]["exec_diverged"] == 1

    def test_corpus_aggregates(self, distill_server):
        corpus = _get_json(distill_server, "/api/distillation/corpus")
        promotion = {row["insight_id"]: row for row in corpus["promotion"]}
        # An insight with no corroboration is LISTED with a zero, not dropped.
        assert promotion["ins-1"]["support_runs"] == 1
        assert promotion["ins-2"]["support_runs"] == 0
        assert corpus["promotion_blocked"] is False
        by_command = {row["command_name"]: row for row in corpus["by_command"]}
        assert by_command["complete_task"]["n"] == 1
        assert {row["kind"] for row in corpus["by_kind"]} == {
            "missing-in-student",
            "different-answer-same-actions",
        }
        assert {row["role"] for row in corpus["cost"]} == {"teacher", "student"}
        assert corpus["weekly"], "no weekly rate rows"

    def test_export_is_self_contained(self, distill_server):
        payload = _get_json(distill_server, "/api/distillation/export/runA")
        assert payload["export_version"] == 1
        assert payload["run"]["run_id"] == "runA"
        assert [d["divergence_id"] for d in payload["divergences"]] == ["dA", "dRun"]
        assert {i["insight_id"] for i in payload["insights"]} == {"ins-1", "ins-2"}
        # The evidence itself, so an extraction agent can work from the file.
        assert {s["span_id"] for s in payload["spans"][DISTILL_TURN]} == {
            "sd-teacher", "sd-student", "sd-turn",
        }

    def test_a_pre_distillation_db_404s_rather_than_500s(self, server):
        """[DR29]: the seeded DB has the tables (this build wrote it), so the
        marker is what is interrogated — the viewer must answer with a reason,
        never with `no such table` behind a 500."""
        for path in (
            "/api/distillation/runs",
            "/api/distillation/run/nope",
            "/api/distillation/corpus",
        ):
            status, _, _ = _get(server, path)
            assert status in (200, 404), path
        status, _, _ = _get(server, "/api/distillation/nope")
        assert status == 404


class TestDistillationUi:
    """`fix-sb8.7` / `.8` / `.9` / `.10` in the SPA.

    Byte-level assertions rather than a browser, following the existing
    `test_page_never_uses_innerhtml` precedent: the page is 3k lines of
    framework-free vanilla JS with no build step, so what a test can hold onto
    is that the seams exist and the [R22] ban still holds over the new code.
    """

    def test_the_distillation_surface_is_present(self, distill_server):
        page = distill_server.index_html
        for marker in (
            b"renderDistillationCard",   # the run header
            b"renderPassesView",         # one clean waterfall at a time
            b"renderDiffView",           # the aligned two-pane diff
            b"renderInsightsView",       # the ledger beside its evidence
            b"showCorpus",               # the aggregates
            b"/api/distillation/verdict",
            b"firstDivergence",          # jump-to-first-divergence
            b"nonComparable",            # the loud banner
        ):
            assert marker in page, marker

    def test_the_new_code_keeps_the_innerhtml_ban(self, distill_server):
        assert b"innerHTML" not in distill_server.index_html
        # And the banned rail ids from the parent design are still absent.
        assert b'id="channelSel"' not in distill_server.index_html
        assert b"/api/channels" not in distill_server.index_html

    def test_the_page_still_loads_under_its_own_csp(self, distill_server):
        """The CSP hashes are computed from the page, so an edit to the inline
        script flows through that path — but only if the page is still served
        as one hash-sourced inline block."""
        status, headers, body = _get(distill_server, "/")
        assert status == 200
        assert b"renderDistillationCard" in body
        csp = headers["Content-Security-Policy"]
        assert "'sha256-" in csp
        # Only STYLE may be inline; the script block stays hash-sourced.
        assert "script-src 'self' 'unsafe-inline'" not in csp


class TestDistillationVerdicts:
    def test_a_verdict_is_appended_and_supersedes_the_previous(
        self, distill_server, distilled_db
    ):
        status, first = _post(
            distill_server,
            "/api/distillation/verdict",
            {"insight_id": "ins-1", "verdict": "supported", "actor": "human"},
        )
        assert status == 200, first
        status, second = _post(
            distill_server,
            "/api/distillation/verdict",
            {
                "insight_id": "ins-1",
                "verdict": "overfit-to-single-turn",
                "actor": "agent:reviewer",
                "note": "one turn is an anecdote",
            },
        )
        assert status == 200, second

        rows = _rows_of(
            distilled_db,
            "SELECT verdict, actor, superseded FROM distillation_verdicts "
            "WHERE insight_id='ins-1' ORDER BY created_at, verdict_id",
        )
        # Append-only WITH supersede: the history of judgements is evidence.
        assert len(rows) == 2
        assert {r["superseded"] for r in rows} == {0, 1}
        live = [r for r in rows if r["superseded"] == 0]
        assert live[0]["verdict"] == "overfit-to-single-turn"
        assert live[0]["actor"] == "agent:reviewer"

    def test_the_verdict_shows_up_in_the_provenance_view(self, distill_server):
        _post(
            distill_server,
            "/api/distillation/verdict",
            {"insight_id": "ins-1", "verdict": "supported", "actor": "human"},
        )
        payload = _get_json(
            distill_server, "/api/distillation/insights?insight=ins-1"
        )
        assert [v["verdict"] for v in payload["verdicts"]] == ["supported"]

    @pytest.mark.parametrize(
        "body",
        [
            {"insight_id": "ins-1", "verdict": "looks-fine", "actor": "human"},
            {"insight_id": "ins-1", "verdict": "supported", "actor": "root"},
            {"insight_id": "ins-1", "verdict": "supported", "actor": "agent:"},
            {"insight_id": "nope", "verdict": "supported", "actor": "human"},
            {
                "insight_id": "ins-1",
                "verdict": "supported",
                "actor": "human",
                "note": "x" * 4097,
            },
        ],
    )
    def test_rejected_bodies_are_400(self, distill_server, distilled_db, body):
        status, _ = _post(distill_server, "/api/distillation/verdict", body)
        assert status == 400
        assert _rows_of(distilled_db, "SELECT * FROM distillation_verdicts") == []

    def test_the_verdict_route_still_needs_the_token(self, distill_server):
        status, _ = _post(
            distill_server,
            "/api/distillation/verdict",
            {"insight_id": "ins-1", "verdict": "supported", "actor": "human"},
            token=None,
        )
        assert status == 401

    def test_the_verdict_route_cannot_reach_recorded_evidence(
        self, distill_server, distilled_db
    ):
        """§12 rule 5: the read-only property is RE-ASSERTED, not relaxed.

        The route touches `distillation_verdicts` and nothing else — not
        spans, turns or artifacts, and not the `pinned` column either, whose
        consequential change `prune()` derives ([DR52]).
        """
        before = _row_counts(distilled_db)
        before_runs = _rows_of(
            distilled_db, "SELECT run_id, pinned FROM distillation_runs"
        )
        before_divergences = _rows_of(
            distilled_db, "SELECT * FROM distillation_divergences"
        )
        status, _ = _post(
            distill_server,
            "/api/distillation/verdict",
            {"insight_id": "ins-1", "verdict": "supported", "actor": "human"},
        )
        assert status == 200
        assert _row_counts(distilled_db) == before
        assert (
            _rows_of(distilled_db, "SELECT run_id, pinned FROM distillation_runs")
            == before_runs
        )
        assert (
            _rows_of(distilled_db, "SELECT * FROM distillation_divergences")
            == before_divergences
        )

    def test_other_posts_are_still_405(self, distill_server):
        for path in ("/api/turns", "/api/distillation/runs", "/api/spans/x"):
            status, _ = _post(distill_server, path, {})
            assert status == 405


class TestShippedSqlRecipes:
    """`[DR54]`: a recipe is not verified until it has been EXECUTED against a
    fixture containing the shapes that silently break it.

    Parse-checking is retired as a standard of evidence here — it is what let a
    promotion query ship returning three supporters where the answer was one,
    and a contradiction query ship returning zero rows with no error. Every SQL
    block in the skill's Distillation section is extracted from the shipped
    markdown and run against `distilled_db`, which holds a comparable run, a
    non-comparable one, a replay, a no-divergence run, a run-level
    NULL-command divergence and a NULL-command failed span.
    """

    def _recipes(self) -> list[str]:
        from fastworkflow import state_paths as _sp  # noqa: F401

        reference = (
            Path(run_chatbot_server.__file__).resolve().parents[1]
            / "skills_for_coding_fastworkflows"
            / "debug-workflow-conversations"
            / "reference.md"
        )
        text = reference.read_text()
        section = text.split("### Distillation recipes", 1)[1]
        block = section.split("```sql", 1)[1].split("```", 1)[0]
        statements = []
        for chunk in block.split(";"):
            # A recipe's `--` title lines belong to the statement that FOLLOWS
            # them, so they land at the head of the next chunk and are dropped
            # here. Trailing same-line comments stay: SQLite reads them fine
            # and they carry the reason each filter is there.
            sql = "\n".join(
                line
                for line in chunk.strip().splitlines()
                if line.strip() and not line.strip().startswith("--")
            ).strip()
            if sql:
                statements.append(sql)
        assert len(statements) >= 8, f"only {len(statements)} recipes extracted"
        return statements

    def test_every_documented_recipe_executes(self, distilled_db):
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        params = {"insight_id": "ins-1", "text_hash": "hash-ins-1",
                  "turn_key": DISTILL_TURN, "span_id": "sd-teacher"}
        try:
            for statement in self._recipes():
                needed = {
                    name: value
                    for name, value in params.items()
                    if ":" + name in statement
                }
                conn.execute(statement, needed).fetchall()
        finally:
            conn.close()

    def test_a_span_resolves_back_to_the_rules_drawn_from_it(self, distilled_db):
        """Acceptance criterion 2's reverse direction: given a tool call that
        looks wrong, has a rule already been written about it?"""
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            recipe = next(
                st for st in self._recipes() if "d.left_span_id = :span_id" in st
            )
            rows = conn.execute(recipe, {"span_id": "sd-teacher"}).fetchall()
        finally:
            conn.close()
        assert {r["insight_id"] for r in rows} == {"ins-1"}

    def test_the_support_recipe_excludes_replays_and_non_comparable_runs(
        self, distilled_db
    ):
        """The specific defect `[DR54]` caught: runB (non-comparable) and runC
        (a replay of runA) carry the same command and kind as the cited
        divergence, so a recipe missing either filter reports three supporters
        for an insight that has one."""
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            support = next(
                st for st in self._recipes() if "cited.command_key = d.command_key" in st
            )
            rows = conn.execute(support, {"insight_id": "ins-1"}).fetchall()
        finally:
            conn.close()
        assert {r["run_id"] for r in rows} == {"runA"}

    def test_the_contradiction_recipe_survives_a_null_command_name(
        self, distilled_db
    ):
        """Without the `cited.command_name IS NOT NULL` guard, three-valued
        logic makes EXISTS false for every run and the query returns nothing
        at all — which reads as "no contradictions found"."""
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            recipe = next(
                st for st in self._recipes() if "CROSS JOIN cited" in st
            )
            rows = conn.execute(recipe, {"insight_id": "ins-1"}).fetchall()
        finally:
            conn.close()
        assert {r["run_id"] for r in rows} == {"runD"}

    def test_the_promotion_recipe_keeps_an_uncorroborated_insight(
        self, distilled_db
    ):
        """A run-level insight keys on nothing, so a join-chain form drops it
        entirely. It has to appear with a zero — an insight that silently
        disappears from the promotion list is the failure this epic exists to
        remove."""
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            recipe = next(st for st in self._recipes() if "AS support_runs" in st)
            rows = {r["insight_id"]: r for r in conn.execute(recipe).fetchall()}
        finally:
            conn.close()
        assert set(rows) == {"ins-1", "ins-2"}
        assert rows["ins-1"]["support_runs"] == 1
        assert rows["ins-2"]["support_runs"] == 0


class TestDocumentedSqlMatchesExecutedSql:
    """The UI and the agent surface must not drift apart.

    `fix-sb8.10`'s design note is explicit about the risk: the corpus view is
    worth having only if it "runs exactly the §15 recipes, so the UI and the
    agent surface cannot drift". They are two copies of the same query — one
    in `observability_store`, one in the shipped skill — and nothing stops
    someone fixing a bug in one of them.

    Equality of TEXT would be the wrong assertion: the executed constants
    carry a `{scope}` placeholder for channel filtering that has no place in
    documentation. Equality of ANSWERS is the property that matters, so both
    are run against the same fixture and their result sets compared.
    """

    def _doc_recipe(self, needle: str) -> str:
        return next(
            st for st in TestShippedSqlRecipes()._recipes() if needle in st
        )

    @pytest.mark.parametrize(
        "needle, constant, params",
        [
            ("cited.command_key = d.command_key", "_SUPPORT_RUNS_SQL",
             {"insight_id": "ins-1"}),
            ("CROSS JOIN cited", "_CONTRADICT_RUNS_SQL", {"insight_id": "ins-1"}),
            ("d.level = 'run'", "_CONTRADICT_RUN_LEVEL_SQL",
             {"insight_id": "ins-2"}),
        ],
    )
    def test_provenance_recipes_agree(
        self, distilled_db, needle, constant, params
    ):
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            documented = [
                tuple(r) for r in conn.execute(self._doc_recipe(needle), params)
            ]
            executed = [
                tuple(r)
                for r in conn.execute(getattr(obs, constant), params)
            ]
        finally:
            conn.close()
        assert documented == executed, (
            f"{constant} and its shipped documentation no longer answer the "
            "same question"
        )

    @pytest.mark.parametrize(
        "needle, constant",
        [
            ("strftime('%Y-W%W'", "_WEEKLY_RATE_SQL"),
            ("missing-in-student", "_BY_COMMAND_SQL"),
            ("GROUP BY d.level, d.kind", "_BY_KIND_SQL"),
            ("p.role IN ('teacher','student')", "_COST_SQL"),
        ],
    )
    def test_aggregate_recipes_agree(self, distilled_db, needle, constant):
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        try:
            documented = [tuple(r) for r in conn.execute(self._doc_recipe(needle))]
            # The unscoped form: `{scope}` is the channel filter the route adds.
            executed = [
                tuple(r)
                for r in conn.execute(getattr(obs, constant).replace("{scope}", ""))
            ]
        finally:
            conn.close()
        assert documented == executed, (
            f"{constant} and its shipped documentation no longer agree"
        )

    def test_the_promotion_recipe_agrees_on_support_counts(self, distilled_db):
        """The store's constant additionally computes `material_support_runs`,
        which the documented version omits for readability — so the comparison
        is over the columns they share."""
        conn = sqlite3.connect(distilled_db)
        conn.row_factory = sqlite3.Row
        keys = ("insight_id", "kind", "verdict", "support_runs")
        try:
            documented = [
                tuple(r[k] for k in keys)
                for r in conn.execute(self._doc_recipe("AS support_runs"))
            ]
            executed = [
                tuple(r[k] for k in keys)
                for r in conn.execute(obs._PROMOTION_SQL)
            ]
        finally:
            conn.close()
        assert documented == executed


def _rows_of(db_path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
