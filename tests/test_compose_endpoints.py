"""POST /compose_answer_card and POST /compose_question_form (fix-weih.9).

Real FastAPI app over a copy of ``tests/hello_world_workflow``, real auth, real composers. The only seam is
``fastworkflow.run_chatbot.composer._composer_lm``, replaced by ``dspy.utils.DummyLM``; no test calls a live model.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import litellm  # noqa: F401  (must precede dspy)
import pytest
from dspy.utils import DummyLM
from fastapi.testclient import TestClient

from fastworkflow.run_chatbot import composer as gui
from fastworkflow.run_fastapi_mcp import mcp_specific
from fastworkflow.utils import dspy_utils

HELLO = Path(__file__).resolve().parent / "hello_world_workflow"
_COPY_IGNORE = shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__")


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(env_files, tmp_path):
    env_file, passwords_file = env_files
    workflow = tmp_path / "hello"
    shutil.copytree(HELLO, workflow, ignore=_COPY_IGNORE, symlinks=True)
    sys.argv = ["pytest", "--workflow_path", str(workflow), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _auth(client: TestClient, channel_id: str) -> dict:
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": "compose"})
    assert init.status_code == 200, init.text
    return {"Authorization": f"Bearer {init.json()['access_token']}"}


@pytest.fixture
def client(app_module):
    return TestClient(app_module.app)


@pytest.fixture
def headers(client):
    return _auth(client, f"compose_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def composer(monkeypatch):
    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())

    def install(answers):
        lm = DummyLM(answers)
        monkeypatch.setattr(gui, "_composer_lm", lambda: lm)
        return lm
    return install


def _results(n: int = 1) -> list[dict]:
    return [
        {"id": f"r{i}", "command": "get_order_details", "output": {
            "status": "delivered", "total": 186.32, "payment_method": "credit_card_4421890",
            "lines": [{"name": "Bike", "qty": 1}, {"name": "Lock", "qty": 2}],
        }}
        for i in range(1, n + 1)
    ]


def _card_body(**extra) -> dict:
    return {"user_message": "where is my order", "answer": "Your order is delivered.", "results": _results(), **extra}


def _form_body(**extra) -> dict:
    return {"question": "Which payment method should I refund to: paypal_5521 or the original?",
            "user_message": "refund my order", "results": _results(), **extra}


def _refs(card: dict) -> list[str]:
    refs = []
    for block in card["blocks"]:
        refs.extend(stat["ref"] for stat in block["stats"])
        if block["ref"]:
            refs.append(block["ref"])
    return refs


def _resolves(ref: str, results: list[dict]) -> bool:
    first, *rest = ref.split(".")
    value = next(r["output"] for r in results if r["id"] == first)
    for key in rest:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


# ---- 1, 2, 3, 4: answer card ------------------------------------------------

def test_answer_card_requires_a_token(client):
    assert client.post("/compose_answer_card", json=_card_body()).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.post("/compose_answer_card", json=_card_body(), headers=bad).status_code == 401


def test_answer_card_returns_a_verified_card_without_a_reason(client, headers, composer):
    composer([{"lead": "Your order is delivered.", "sections": [{"heading": "Order", "blocks": [
        {"kind": "stats", "stats": [{"label": "Status", "ref": "status"}, {"label": "Total", "ref": "r1.total"}]},
        {"kind": "table", "ref": "r1.lines", "fields": "name,qty"},
    ]}]}])
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"card"}                       # no "reason" key on success
    card = body["card"]
    assert card["lead"] == "Your order is delivered."
    assert [b["kind"] for b in card["blocks"]] == ["stats", "table"]
    refs = _refs(card)
    assert refs == ["r1.status", "r1.total", "r1.lines"]
    assert all(_resolves(ref, _results()) for ref in refs)


def test_answer_card_response_is_wire_v2_with_flat_blocks_for_an_old_page(client, headers, composer):
    composer([{"lead": "Your order is delivered.", "sections": [
        {"heading": "Order", "verdict": "The order total is 186.32.", "blocks": [
            {"kind": "stats", "stats": [{"label": "Total", "ref": "r1.total"}]}]},
        {"heading": "Lines", "verdict": "", "blocks": [{"kind": "table", "ref": "r1.lines", "fields": "name,qty"}]},
    ]}])
    body = _card_body(answer="Your order is delivered. The order total is 186.32.")   # the verdict's number is the answer's
    resp = client.post("/compose_answer_card", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    card = resp.json()["card"]
    assert set(card) == {"v", "lead", "sections", "blocks"} and card["v"] == 2
    assert [(s["heading"], s["verdict"]) for s in card["sections"]] == [("Order", "The order total is 186.32."), ("Lines", "")]
    # an old page reads only lead and blocks: the sections' blocks, flat, in order
    assert card["blocks"] == [b for s in card["sections"] for b in s["blocks"]]
    assert _refs(card) == ["r1.total", "r1.lines"]


def test_openapi_422_descriptions_carry_the_composers_own_limits(client):
    paths = client.get("/openapi.json").json()["paths"]
    card = paths["/compose_answer_card"]["post"]["responses"]["422"]["description"]
    form = paths["/compose_question_form"]["post"]["responses"]["422"]["description"]
    assert f"{gui.MAX_RESULTS} results" in card and f"{gui.MAX_ANSWER_CHARS} characters" in card
    assert f"{gui.MAX_RESULTS} results" in form and f"{gui.MAX_TEXT_CHARS} characters" in form


def test_answer_card_accepts_up_to_sixty_results(client, headers, composer, app_module):
    assert app_module.COMPOSE_TIMEOUT_SECONDS == 40          # was 30: the plain answer is already on screen
    composer([{"lead": "Your order is delivered.", "sections": [
        {"heading": "Order", "verdict": "", "blocks": [{"kind": "record", "ref": "r31"}]}]}] * 2)
    resp = client.post("/compose_answer_card", json={**_card_body(), "results": _results(31)}, headers=headers)
    assert resp.status_code == 200 and resp.json()["card"]["sections"][0]["blocks"][0]["ref"] == "r31"
    resp = client.post("/compose_answer_card", json={**_card_body(), "results": _results(60)}, headers=headers)
    assert resp.status_code == 200 and resp.json()["card"] is not None


def test_answer_card_drops_a_block_with_an_invented_ref(client, headers, composer):
    composer([{"lead": "Your order is delivered.", "sections": [{"blocks": [
        {"kind": "record", "ref": "r1.invented_field"},
        {"kind": "stats", "stats": [{"label": "Status", "ref": "r1.status"}]},
    ]}]}])
    card = client.post("/compose_answer_card", json=_card_body(), headers=headers).json()["card"]
    assert [b["kind"] for b in card["blocks"]] == ["stats"]
    assert "r1.invented_field" not in _refs(card)


def test_answer_card_no_model_and_failed_reasons(client, headers, composer, monkeypatch):
    def unavailable():
        raise gui.ComposerUnavailable("no model")
    monkeypatch.setattr(gui, "_composer_lm", unavailable)
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"card": None, "reason": "no_model"}

    composer([{"lead": "ok", "sections": "this is not a json list"}] * 3)
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"card": None, "reason": "failed"}


def test_answer_card_timeout_reason_and_orphan_thread_ends_quietly(
        client, headers, composer, app_module, monkeypatch, caplog):
    monkeypatch.setattr(app_module, "COMPOSE_TIMEOUT_SECONDS", 0.05)
    finished = threading.Event()
    lm = DummyLM([{"lead": "Your order is delivered.", "sections": []}])

    def slow():
        time.sleep(0.4)             # outlives the timeout; the thread cannot be cancelled
        finished.set()
        return lm
    monkeypatch.setattr(gui, "_composer_lm", slow)

    started = time.monotonic()
    resp = client.post("/compose_answer_card", json=_card_body(), headers=headers)
    assert time.monotonic() - started < 1.5
    assert resp.status_code == 200 and resp.json() == {"card": None, "reason": "timeout"}
    assert finished.wait(2), "the abandoned composer call should run to completion in the threadpool"
    time.sleep(0.2)                 # let its result be discarded
    assert not [r for r in caplog.records if r.levelname in ("ERROR", "CRITICAL")]


def test_compose_logs_one_info_line_without_user_text(client, headers, composer, caplog):
    import logging
    composer([{"lead": "Your order is delivered.", "sections": []}])
    with caplog.at_level(logging.INFO):
        client.post("/compose_answer_card", json=_card_body(answer="SECRET-ANSWER-TEXT"), headers=headers)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("compose ")]
    assert len(lines) == 1
    assert "answer_card" in lines[0] and "results=1" in lines[0] and "duration_ms=" in lines[0] and "outcome=" in lines[0]
    assert "SECRET" not in lines[0] and "delivered" not in lines[0]


# ---- 5: question form -------------------------------------------------------

def test_question_form_requires_a_token(client):
    assert client.post("/compose_question_form", json=_form_body()).status_code == 401


def test_question_form_grounds_options_and_has_no_reason(client, headers, composer):
    composer([{"form_fields": [
        {"label": "Payment method", "kind": "dropdown",
         "options": ["credit_card_4421890", "paypal_5521", "bitcoin_99999"]},
        {"label": "Reason", "kind": "textarea"},
    ]}])
    resp = client.post("/compose_question_form", json=_form_body(), headers=headers)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"form"}
    fields = body["form"]["fields"]
    assert [f["kind"] for f in fields] == ["choice", "longtext"]
    values = [o["value"] for o in fields[0]["options"]]
    assert values == ["credit_card_4421890", "paypal_5521"]   # in the results / in the question
    assert "bitcoin_99999" not in values                        # in neither


def test_a_by_design_nothing_is_not_needed_not_failed_and_logs_no_warning(client, headers, composer, caplog):
    """The composer chose the plain view: the gate drops a text-only form, a lead restates a long answer."""
    import logging

    composer([{"form_fields": [{"label": "Reason", "kind": "text"}]}])
    with caplog.at_level(logging.WARNING):
        resp = client.post("/compose_question_form", json=_form_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"form": None, "reason": "not_needed"}

    long_answer = "Your order is delivered and the bike arrived on time. " + "More shipping detail follows here. " * 20
    composer([{"lead": "Your order is delivered and the bike arrived on time.", "sections": []}])
    resp = client.post("/compose_answer_card", json=_card_body(answer=long_answer), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"card": None, "reason": "not_needed"}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    # a lead that does summarize the answer still shows nothing when the turn has results to open
    composer([{"lead": "Delivered on time.", "sections": []}])
    resp = client.post("/compose_answer_card", json=_card_body(answer=long_answer), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"card": None, "reason": "not_needed"}
    composer([{"lead": "Delivered on time.", "sections": []}])
    resp = client.post("/compose_answer_card", json=_card_body(answer=long_answer, results=[]), headers=headers)
    assert resp.status_code == 200 and resp.json()["card"]["lead"] == "Delivered on time."


def test_question_form_no_model_failed_and_timeout_reasons(client, headers, composer, app_module, monkeypatch):
    def unavailable():
        raise gui.ComposerUnavailable("no model")
    monkeypatch.setattr(gui, "_composer_lm", unavailable)
    resp = client.post("/compose_question_form", json=_form_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"form": None, "reason": "no_model"}

    composer([{"form_fields": "this is not a json list"}] * 3)
    resp = client.post("/compose_question_form", json=_form_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"form": None, "reason": "failed"}

    monkeypatch.setattr(app_module, "COMPOSE_TIMEOUT_SECONDS", 0.05)
    lm = DummyLM([{"form_fields": []}])

    def slow():
        time.sleep(0.3)
        return lm
    monkeypatch.setattr(gui, "_composer_lm", slow)
    resp = client.post("/compose_question_form", json=_form_body(), headers=headers)
    assert resp.status_code == 200 and resp.json() == {"form": None, "reason": "timeout"}
    time.sleep(0.4)


# ---- 6: limits --------------------------------------------------------------

def test_over_the_limits_is_422(client, headers):
    assert client.post("/compose_answer_card", json={**_card_body(), "results": _results(61)},
                       headers=headers).status_code == 422
    assert client.post("/compose_question_form", json={**_form_body(), "results": _results(61)},
                       headers=headers).status_code == 422
    assert client.post("/compose_answer_card", json=_card_body(answer="x" * 150001),
                       headers=headers).status_code == 422
    assert client.post("/compose_question_form", json=_form_body(question="x" * 20001),
                       headers=headers).status_code == 422
    # exactly 60 is accepted (the model is not consulted for the status check: any 200 will do)


# ---- 7: MCP -----------------------------------------------------------------

def test_compose_operations_are_excluded_from_mcp(app_module):
    assert "compose_answer_card" in mcp_specific.EXCLUDED_OPERATIONS
    assert "compose_question_form" in mcp_specific.EXCLUDED_OPERATIONS
    for name in ("root", "dump_all_conversations", "generate_mcp_token", "rest_initialize", "perform_action",
                 "rest_invoke_agent", "refresh_token", "get_turn_trace"):
        assert name in mcp_specific.EXCLUDED_OPERATIONS
    # fix-1ma8: the turn lookups are the recovery path of an MCP client whose time-out fired, so they are tools.
    for name in ("get_turn", "get_latest_turn"):
        assert name not in mcp_specific.EXCLUDED_OPERATIONS
    operations = {op["operationId"] for path in app_module.app.openapi()["paths"].values() for op in path.values()}
    assert {"compose_answer_card", "compose_question_form"} <= operations


def test_compose_info_line_carries_the_composer_diagnostics(client, headers, composer, caplog):
    import logging
    composer([{"lead": "Your order is delivered.", "sections": [
        {"heading": "Order", "verdict": "", "blocks": [{"kind": "stats", "stats": [{"label": "Total", "ref": "r1.total"}]}]},
        {"heading": "Lost", "verdict": "", "blocks": [{"kind": "table", "ref": "r9.nothing"}]},
    ]}])
    with caplog.at_level(logging.INFO, logger="fastWorkflow"):
        resp = client.post("/compose_answer_card", json=_card_body(answer="SECRET-ANSWER-TEXT"), headers=headers)
    assert resp.json()["card"] is not None and set(resp.json()) == {"card"}      # the wire response is unchanged
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("compose ")]
    assert len(lines) == 1
    line = lines[0]
    import re
    assert re.fullmatch(
        r"compose answer_card: results=1 digest_chars=\d+ collapsed=0 sections=1/2 blocks=1/2 lead=kept "
        r"drops=empty_section:1,ref_unresolved:1 retry=failed outcome=card duration_ms=\d+", line), line   # no second answer scripted
    assert "SECRET" not in line and "nothing" not in line


def test_compose_question_form_keeps_its_plain_info_line(client, headers, composer, caplog):
    import logging
    composer([{"form_fields": [{"label": "Method", "kind": "text"}]}])
    with caplog.at_level(logging.INFO, logger="fastWorkflow"):
        client.post("/compose_question_form", json=_form_body(), headers=headers)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("compose ")]
    assert len(lines) == 1 and lines[0].startswith("compose question_form: results=1 outcome=")
    assert "drops=" not in lines[0] and "duration_ms=" in lines[0]


def test_compose_info_line_carries_the_retry_fields(client, headers, composer, caplog):
    import logging
    import re
    first = {"lead": "Your order is delivered.", "sections": [
        {"heading": "Order", "verdict": "", "blocks": [{"kind": "stats", "stats": [{"label": "Total", "ref": "r1.total"}]},
                                                       {"kind": "table", "ref": "r9.nothing"}]}]}
    second = {"lead": "Your order is delivered.", "sections": [
        {"heading": "Order", "verdict": "", "blocks": [{"kind": "stats", "stats": [{"label": "Total", "ref": "r1.total"}]}]}]}
    composer([first, second])
    with caplog.at_level(logging.INFO, logger="fastWorkflow"):
        resp = client.post("/compose_answer_card", json=_card_body(answer="SECRET-ANSWER-TEXT"), headers=headers)
    assert resp.json()["card"] is not None and set(resp.json()) == {"card"}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("compose ")]
    assert len(lines) == 1
    assert re.fullmatch(
        r"compose answer_card: results=1 digest_chars=\d+ collapsed=0 sections=1/1 blocks=1/2 lead=kept "
        r"drops=ref_unresolved:1 retry=used drops2=none sections2=1/1 blocks2=1/1 outcome=card duration_ms=\d+",
        lines[0]), lines[0]
    assert "SECRET" not in lines[0] and "nothing" not in lines[0]
