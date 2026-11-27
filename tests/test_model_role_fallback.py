"""One required model (bd fix-ygxo): every model role falls back to the base model (``LLM_AGENT``, else
``LLM_PLANNER``), and the key follows the model, never across a provider. ``dspy_utils.resolve_role`` is the one
owner of the rule; ``get_lm``, the generators' ``role_completion``, provenance and the startup checks read it."""
from __future__ import annotations

import logging

import litellm
import pytest

import fastworkflow
from fastworkflow.utils import dspy_utils, model_roles

ROLE_VARS = [*model_roles.ROLE_KEY_VARS, *model_roles.ROLE_KEY_VARS.values(), "LITELLM_PROXY_API_KEY",
             "LITELLM_PROXY_API_BASE"]


@pytest.fixture
def env(monkeypatch):
    """An env-file layer with no model or key of any role, and none in the process either."""
    settings: dict = {}
    monkeypatch.setattr(fastworkflow, "_env_vars", settings)
    for name in ROLE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dspy_utils, "_ANNOUNCED_FALLBACKS", set())
    return settings


# ---------------------------------------------------------------- the model


def test_agent_mode_every_fallback_role_takes_the_agent_model_and_key(env):
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key"})

    for role in sorted(model_roles.FALLS_BACK):
        resolved = dspy_utils.resolve_role(role)
        assert (resolved.model, resolved.inherited_from) == ("mistral/agent", "LLM_AGENT"), role
        assert (resolved.key, resolved.key_var) == ("agent-key", "LITELLM_API_KEY_AGENT"), role
    agent = dspy_utils.resolve_role("LLM_AGENT")
    assert (agent.model, agent.inherited_from, agent.key) == ("mistral/agent", None, "agent-key")


def test_no_agent_mode_the_planner_is_the_base_model(env):
    env.update({"LLM_PLANNER": "mistral/planner", "LITELLM_API_KEY_PLANNER": "planner-key"})

    assert dspy_utils.resolve_role("LLM_AGENT").model is None   # the agent never inherits
    for role in sorted(model_roles.FALLBACK_ROLES):
        resolved = dspy_utils.resolve_role(role)
        assert (resolved.model, resolved.inherited_from, resolved.key) == (
            "mistral/planner", "LLM_PLANNER", "planner-key"), role


def test_the_agent_model_is_the_base_when_both_are_set(env):
    env.update({"LLM_AGENT": "mistral/agent", "LLM_PLANNER": "mistral/planner"})

    assert dspy_utils.resolve_role("LLM_PARAM_EXTRACTION").model == "mistral/agent"
    assert dspy_utils.resolve_role("LLM_PLANNER").inherited_from is None


def test_a_role_of_its_own_and_the_roles_with_no_fallback(env):
    env.update({"LLM_AGENT": "mistral/agent", "LLM_PARAM_EXTRACTION": "mistral/extractor"})

    assert dspy_utils.resolve_role("LLM_PARAM_EXTRACTION").inherited_from is None
    assert dspy_utils.resolve_role("LLM_PARAM_EXTRACTION").model == "mistral/extractor"
    for role in ("LLM_RESPONSE_GEN", "LLM_DISTILLATION", "LLM_TEACHER_AGENT", "LLM_STUDENT_PLANNER"):
        assert dspy_utils.resolve_role(role).model is None, role


def test_an_empty_role_setting_falls_back_like_an_unset_one(env):
    env.update({"LLM_AGENT": "mistral/agent", "LLM_SYNDATA_GEN": ""})

    assert dspy_utils.resolve_role("LLM_SYNDATA_GEN").model == "mistral/agent"


def test_no_base_model_leaves_every_role_without_a_model(env):
    for role in model_roles.ROLE_KEY_VARS:
        assert dspy_utils.resolve_role(role).model is None
    with pytest.raises(ValueError, match="LLM_PARAM_EXTRACTION"):
        dspy_utils.get_lm("LLM_PARAM_EXTRACTION")


def test_get_lm_builds_the_fallback_model_with_the_key_that_follows_it(env):
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key"})

    lm = dspy_utils.get_lm("LLM_CONVERSATION_STORE")

    assert lm.model == "mistral/agent"
    assert lm.kwargs.get("api_key") == "agent-key"
    assert lm.fastworkflow_role == "LLM_CONVERSATION_STORE"


def test_a_fallback_is_said_once_per_role_per_process(env):
    env.update({"LLM_AGENT": "mistral/agent"})
    seen: list[str] = []

    class Sink(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    sink = Sink(level=logging.INFO)
    previous = dspy_utils.logger.level
    dspy_utils.logger.addHandler(sink)
    dspy_utils.logger.setLevel(logging.INFO)
    try:
        dspy_utils.resolve_role("LLM_SYNDATA_GEN")
        dspy_utils.resolve_role("LLM_SYNDATA_GEN")
        dspy_utils.resolve_role("LLM_AGENT")
    finally:
        dspy_utils.logger.removeHandler(sink)
        dspy_utils.logger.setLevel(previous)
    fallback_lines = [line for line in seen if "LLM_SYNDATA_GEN is not set" in line]
    assert len(fallback_lines) == 1
    assert "LLM_AGENT" in fallback_lines[0] and "mistral/agent" in fallback_lines[0]


# ---------------------------------------------------------------- the key


def test_a_role_with_its_own_model_on_the_base_provider_takes_the_base_key(env):
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key",
                "LLM_PARAM_EXTRACTION": "mistral/extractor"})

    resolved = dspy_utils.resolve_role("LLM_PARAM_EXTRACTION")
    assert (resolved.key, resolved.key_var) == ("agent-key", "LITELLM_API_KEY_AGENT")


def test_a_key_never_crosses_a_provider(env):
    """The extractor's model is another provider's: the agent's key is not sent to it."""
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key",
                "LLM_PARAM_EXTRACTION": "openai/extractor"})

    resolved = dspy_utils.resolve_role("LLM_PARAM_EXTRACTION")
    assert (resolved.key, resolved.key_var) == (None, None)
    assert resolved.needs_key
    assert "api_key" not in dspy_utils.get_lm("LLM_PARAM_EXTRACTION").kwargs


def test_a_role_own_key_wins(env):
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key",
                "LITELLM_API_KEY_SYNDATA_GEN": "syndata-key"})

    resolved = dspy_utils.resolve_role("LLM_SYNDATA_GEN")
    assert (resolved.model, resolved.key, resolved.key_var) == (
        "mistral/agent", "syndata-key", "LITELLM_API_KEY_SYNDATA_GEN")


def test_a_role_with_no_fallback_never_takes_the_base_key(env):
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key",
                "LLM_RESPONSE_GEN": "mistral/composer"})

    assert dspy_utils.resolve_role("LLM_RESPONSE_GEN").key is None


def test_bedrock_needs_no_key(env):
    env.update({"LLM_PLANNER": "bedrock/nvidia.model"})

    resolved = dspy_utils.resolve_role("LLM_SYNDATA_GEN")
    assert (resolved.model, resolved.key) == ("bedrock/nvidia.model", None)
    assert not resolved.needs_key


def test_a_proxy_model_takes_the_proxy_key_and_base_never_a_role_key(env):
    env.update({"LLM_AGENT": "litellm_proxy/agent", "LITELLM_API_KEY_AGENT": "agent-key",
                "LITELLM_PROXY_API_KEY": "proxy-key", "LITELLM_PROXY_API_BASE": "http://127.0.0.1:4000"})

    resolved = dspy_utils.resolve_role("LLM_PARAM_EXTRACTION")
    assert (resolved.model, resolved.key, resolved.key_var) == (
        "litellm_proxy/agent", "proxy-key", "LITELLM_PROXY_API_KEY")
    lm = dspy_utils.get_lm("LLM_PARAM_EXTRACTION")
    assert lm.kwargs.get("api_base") == "http://127.0.0.1:4000"
    assert lm.kwargs.get("api_key") == "proxy-key"


def test_a_direct_role_does_not_take_a_proxy_base_model_key(env):
    """litellm_proxy is its own provider: a direct model on mistral gets no key from a proxied base."""
    env.update({"LLM_AGENT": "litellm_proxy/agent", "LITELLM_PROXY_API_KEY": "proxy-key",
                "LITELLM_API_KEY_AGENT": "agent-key", "LLM_PARAM_EXTRACTION": "mistral/extractor"})

    assert dspy_utils.resolve_role("LLM_PARAM_EXTRACTION").key is None


# ---------------------------------------------------------------- the generators' completion


def test_the_generators_call_litellm_with_the_role_model_and_key_per_call(env, monkeypatch):
    """``role_completion`` replaces the model and key a caller passed (the parameter-example generator passes the raw
    ``LLM_SYNDATA_GEN`` it read) with the resolved ones, as arguments of the call, and leaves ``litellm.api_key``
    alone."""
    env.update({"LLM_AGENT": "mistral/agent", "LITELLM_API_KEY_AGENT": "agent-key"})
    calls: list[dict] = []
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: calls.append(kwargs) or "answer")
    monkeypatch.setattr(litellm, "api_key", None)

    completion = dspy_utils.role_completion("LLM_SYNDATA_GEN")
    assert completion(model=None, api_key=None, messages=[]) == "answer"

    assert calls == [{"model": "mistral/agent", "api_key": "agent-key", "messages": []}]
    assert litellm.api_key is None
    assert getattr(completion, dspy_utils.ROLE_COMPLETION_MARK) == "LLM_SYNDATA_GEN"


def test_train_stops_at_start_with_no_generator_model(env):
    from fastworkflow.train.__main__ import NO_SYNDATA_MODEL, syndata_key_notice, syndata_model_problem

    assert syndata_model_problem() == NO_SYNDATA_MODEL
    env["LLM_PLANNER"] = "mistral/planner"
    assert syndata_model_problem() is None
    notice = syndata_key_notice("pw.env")
    assert "LITELLM_API_KEY_PLANNER" in notice and "mistral/planner" in notice
    env["LITELLM_API_KEY_PLANNER"] = "planner-key"
    assert syndata_key_notice("pw.env") is None
