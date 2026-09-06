"""Tests for direct Amazon Bedrock routing through the AWS credential chain."""

import fastworkflow

from fastworkflow.utils.dspy_utils import (
    LM_MAX_TOKENS_ENV_VAR,
    LM_TEMPERATURE_ENV_VAR,
    get_lm,
)


def test_bedrock_model_ignores_unrelated_role_api_key():
    original_env_vars = dict(fastworkflow._env_vars)
    try:
        fastworkflow._env_vars["LLM_PLANNER"] = (
            "bedrock/us.anthropic.claude-sonnet-4-6"
        )
        fastworkflow._env_vars["LITELLM_API_KEY_PLANNER"] = (
            "non-bedrock-key-must-not-be-forwarded"
        )

        lm = get_lm(
            "LLM_PLANNER",
            "LITELLM_API_KEY_PLANNER",
            temperature=0.0,
        )

        assert lm.model == "bedrock/us.anthropic.claude-sonnet-4-6"
        assert "api_key" not in lm.kwargs
        assert lm.kwargs["temperature"] == 0.0
    finally:
        fastworkflow._env_vars.clear()
        fastworkflow._env_vars.update(original_env_vars)


def test_optional_sampling_policy_pins_every_role():
    original_env_vars = dict(fastworkflow._env_vars)
    try:
        fastworkflow._env_vars.update(
            {
                "LLM_AGENT": "bedrock/us.anthropic.claude-sonnet-4-6",
                LM_TEMPERATURE_ENV_VAR: "0",
                LM_MAX_TOKENS_ENV_VAR: "4096",
            }
        )

        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")

        assert lm.kwargs["temperature"] == 0.0
        assert lm.kwargs["max_tokens"] == 4096
    finally:
        fastworkflow._env_vars.clear()
        fastworkflow._env_vars.update(original_env_vars)


def test_explicit_sampling_values_override_deployment_policy():
    original_env_vars = dict(fastworkflow._env_vars)
    try:
        fastworkflow._env_vars.update(
            {
                "LLM_PLANNER": "bedrock/us.anthropic.claude-sonnet-4-6",
                LM_TEMPERATURE_ENV_VAR: "0.5",
                LM_MAX_TOKENS_ENV_VAR: "4096",
            }
        )

        lm = get_lm(
            "LLM_PLANNER",
            "LITELLM_API_KEY_PLANNER",
            temperature=0.0,
            max_tokens=256,
        )

        assert lm.kwargs["temperature"] == 0.0
        assert lm.kwargs["max_tokens"] == 256
    finally:
        fastworkflow._env_vars.clear()
        fastworkflow._env_vars.update(original_env_vars)
