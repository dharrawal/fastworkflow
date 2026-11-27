"""Per-role model timeout and retries, and what a stalled model call leaves on its span (fix-vdaj).

A recorded turn waited about 120 s for two parameter-extraction calls the provider never answered, because every
role had the one global timeout. Parameter extraction now has its own (20 s, 2 retries); every other role keeps
``LLM_TIMEOUT_SECONDS``. The stall tests run the real ``get_lm`` LM against a local HTTP server that accepts the
request and never answers: no provider, no key, no mock.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest
from pydantic import BaseModel

import fastworkflow
from fastworkflow.observability import tracing
from fastworkflow.observability.store import ObservabilityStore, SQLiteTraceSink
from fastworkflow.utils import dspy_utils
from fastworkflow.parameters import signatures
from fastworkflow.utils.dspy_logger import observe_dspy_host
from fastworkflow.parameters.signatures import NOT_FOUND, InputForParamExtraction


@pytest.fixture
def env():
    saved = dict(fastworkflow._env_vars)
    for key in list(fastworkflow._env_vars):
        if key.startswith(("LLM_TIMEOUT_SECONDS", "LLM_NUM_RETRIES")):
            del fastworkflow._env_vars[key]
    yield fastworkflow._env_vars
    fastworkflow._env_vars.clear()
    fastworkflow._env_vars.update(saved)


@pytest.fixture(autouse=True)
def no_process_env(monkeypatch):
    for name in ("LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS_PARAM_EXTRACTION", "LLM_NUM_RETRIES_PARAM_EXTRACTION",
                 "LLM_TIMEOUT_SECONDS_AGENT", "LLM_NUM_RETRIES_AGENT"):
        monkeypatch.delenv(name, raising=False)


# ---- settings: pure -------------------------------------------------------------------------------------------


def test_extraction_has_its_own_defaults_and_other_roles_the_global_one(env):
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION") == (20.0, 2)
    for role in ("LLM_AGENT", "LLM_PLANNER", "LLM_SYNDATA_GEN", "LLM_RESPONSE_GEN", "LLM_CONVERSATION_STORE"):
        assert dspy_utils.role_limits(role) == (120.0, None), role


def test_the_global_timeout_moves_every_role_but_extraction(env):
    env["LLM_TIMEOUT_SECONDS"] = "45"
    assert dspy_utils.role_limits("LLM_AGENT") == (45.0, None)
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION") == (20.0, 2)


def test_a_role_setting_beats_the_default_and_the_global(env):
    env["LLM_TIMEOUT_SECONDS"] = "45"
    env["LLM_TIMEOUT_SECONDS_PARAM_EXTRACTION"] = "8.5"
    env["LLM_NUM_RETRIES_PARAM_EXTRACTION"] = "1"
    env["LLM_TIMEOUT_SECONDS_AGENT"] = "90"
    env["LLM_NUM_RETRIES_AGENT"] = "0"
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION") == (8.5, 1)
    assert dspy_utils.role_limits("LLM_AGENT") == (90.0, 0)
    assert dspy_utils.role_limits("LLM_PLANNER") == (45.0, None)


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "nan", "inf", ""])
def test_a_bad_timeout_warns_and_the_default_applies(env, bad):
    env["LLM_TIMEOUT_SECONDS_PARAM_EXTRACTION"] = bad
    env["LLM_TIMEOUT_SECONDS"] = bad
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION")[0] == 20.0
    assert dspy_utils.role_limits("LLM_AGENT")[0] == 120.0


@pytest.mark.parametrize("bad", ["abc", "-1", "1.5", "two"])
def test_a_bad_retry_count_warns_and_the_default_applies(env, bad):
    env["LLM_NUM_RETRIES_PARAM_EXTRACTION"] = bad
    env["LLM_NUM_RETRIES_AGENT"] = bad
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION")[1] == 2
    assert dspy_utils.role_limits("LLM_AGENT")[1] is None


def test_zero_retries_is_a_valid_setting(env):
    env["LLM_NUM_RETRIES_PARAM_EXTRACTION"] = "0"
    assert dspy_utils.role_limits("LLM_PARAM_EXTRACTION") == (20.0, 0)


# ---- the LMs get_lm builds -------------------------------------------------------------------------------------


def test_the_extraction_lm_carries_the_role_limits_and_the_others_the_global_one(env):
    env["LLM_PARAM_EXTRACTION"] = "openai/extractor"
    env["LLM_AGENT"] = "openai/agent"
    extraction = dspy_utils.get_lm("LLM_PARAM_EXTRACTION")
    agent = dspy_utils.get_lm("LLM_AGENT")
    assert extraction.kwargs["timeout"] == 20.0
    assert extraction.num_retries == 2
    assert agent.kwargs["timeout"] == 120.0
    assert agent.num_retries == 3  # DSPy's own default: this role's retries are not changed
    assert extraction.fastworkflow_role == "LLM_PARAM_EXTRACTION"
    assert agent.fastworkflow_role == "LLM_AGENT"


def test_a_caller_given_limit_still_wins(env):
    env["LLM_AGENT"] = "openai/agent"
    lm = dspy_utils.get_lm("LLM_AGENT", timeout=5.0, num_retries=0)
    assert lm.kwargs["timeout"] == 5.0
    assert lm.num_retries == 0


# ---- a stalled endpoint ----------------------------------------------------------------------------------------


class _Stall(BaseHTTPRequestHandler):
    """Accepts the request, reads it, never answers (for 30 s, longer than any test waits)."""

    def do_POST(self):  # noqa: N802 - http.server's name
        self.server.hits.append(time.monotonic())
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.release.wait(30)

    def log_message(self, *args):
        pass


@pytest.fixture
def stalled_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stall)
    server.daemon_threads = True
    server.hits = []
    server.release = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()


class _TraceHost:
    def __init__(self, sink):
        self.trace_sink = sink
        self.current_turn_key = "20261113T120000.000000Z-vdajtimeout1"
        self.observability_channel_id = "vdaj-test-channel"
        self.trace_span_stack = []


def _recorded_llm_spans(tmp_path, run):
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    try:
        with observe_dspy_host(host):
            result = run()
        assert sink.flush()
    finally:
        sink.close()
    spans = [s for s in ObservabilityStore(db_path).get_spans(host.current_turn_key)
             if s["name"] == tracing.SPAN_LLM_CALL]
    return result, spans


def _point_extraction_at(env, server, route):
    env["LLM_PARAM_EXTRACTION"] = f"{route}/stalled-model"
    env["LITELLM_PROXY_API_BASE"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    env["LITELLM_PROXY_API_KEY"] = "not-a-key"
    env["LITELLM_API_KEY_PARAM_EXTRACTION"] = "not-a-key"
    env["FW_LM_CACHE"] = "0"
    env["LLM_TIMEOUT_SECONDS_PARAM_EXTRACTION"] = "1"
    env["LLM_NUM_RETRIES_PARAM_EXTRACTION"] = "1"


@pytest.mark.parametrize("route", ["litellm_proxy", "openai"])
def test_a_stalled_extraction_call_fails_in_about_two_timeouts_and_the_span_says_so(
        env, stalled_endpoint, tmp_path, route):
    """1 s timeout, 1 retry: two attempts (the server saw two requests), 1 s of backoff between them: about 3 s,
    not 120. The span names the role, the limits in force, and that it timed out."""
    _point_extraction_at(env, stalled_endpoint, route)
    if route == "openai":
        env["LLM_PARAM_EXTRACTION"] = "openai/stalled-model"
        env.pop("LITELLM_PROXY_API_BASE")  # an openai/ model has no proxy: its base is the api_base below
    lm = dspy_utils.get_lm("LLM_PARAM_EXTRACTION",
                           **({"api_base": f"http://127.0.0.1:{stalled_endpoint.server_address[1]}/v1"}
                              if route == "openai" else {}))

    def call():
        started = time.monotonic()
        with pytest.raises(Exception) as caught:
            with dspy.context(lm=lm):
                lm("hello")
        return time.monotonic() - started, caught.value

    (elapsed, error), spans = _recorded_llm_spans(tmp_path, call)

    assert 2.0 <= elapsed < 8.0, elapsed
    assert len(stalled_endpoint.hits) == 2  # one try and one retry, no more
    assert dspy_utils.is_timeout_error(error)
    assert len(spans) == 1
    assert spans[0]["status"] == tracing.STATUS_ERROR
    attributes = json.loads(spans[0]["attributes"])
    assert attributes["role"] == "LLM_PARAM_EXTRACTION"
    assert attributes["timeout_s"] == 1.0
    assert attributes["num_retries"] == 1
    assert attributes["outcome"] == "timeout"
    assert attributes["timed_out"] is True
    assert attributes["exception_type"] == type(error).__name__
    assert "exception" in attributes


class _Params(BaseModel):
    order_id: str
    address: str


def test_extraction_that_times_out_after_its_retries_is_not_given_and_is_not_repeated(
        env, stalled_endpoint, monkeypatch):
    """The caller's handling: the timeout is logged, nothing is extracted, every field takes its not-given value
    (so a required one is asked for, as when the model answers with nothing). The sampling retries of
    FirstAttemptGreedy do not repeat the wait: the server sees 1 + retries requests, not 3 x that."""
    monkeypatch.setattr(signatures, "_field_given_enabled", lambda: False)
    _point_extraction_at(env, stalled_endpoint, "litellm_proxy")

    started = time.monotonic()
    result = InputForParamExtraction(command="cancel order W1").extract_parameters(
        CommandParameters=_Params, subject_command_name=None, workflow_folderpath=None)
    elapsed = time.monotonic() - started

    assert 2.0 <= elapsed < 8.0, elapsed
    assert len(stalled_endpoint.hits) == 2
    assert result.order_id == NOT_FOUND
    assert result.address == NOT_FOUND
