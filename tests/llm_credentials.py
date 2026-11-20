"""Shared gate for tests that need a working LLM_SYNDATA_GEN model."""


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def syndata_llm_available(env_vars) -> bool:
    """True when LLM_SYNDATA_GEN can authenticate.

    Either a real LITELLM_API_KEY_SYNDATA_GEN is present, or the model is a
    ``bedrock/`` model, which authenticates from the ambient AWS credential chain
    and must be left without a key (a non-empty key is sent to Bedrock as a bearer token).
    """
    if _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        return True
    return str(env_vars.get("LLM_SYNDATA_GEN") or "").startswith("bedrock/")


def real_llm_credentials_available() -> bool:
    """True when the process can authenticate to some model: a real ``LITELLM_API_KEY_*`` or a ``bedrock/`` role.

    Reads the tracked examples, ``env/.env``, ``passwords/.env`` and the process environment (values are only tested,
    never printed). Placeholders in the shipped style (``<...>``) count as absent. This is the one gate of the
    ``requires_llm_key`` marker (``tests/conftest.py``).
    """
    import os
    from pathlib import Path

    from dotenv import dotenv_values

    root = Path(__file__).resolve().parent.parent
    settings: dict = {}
    for path in (
        root / "fastworkflow" / "examples" / "fastworkflow.env",
        root / "fastworkflow" / "examples" / "fastworkflow.passwords.env",
        root / "env" / ".env",
        root / "passwords" / ".env",
    ):
        settings.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    settings.update(os.environ)
    for name, value in settings.items():
        if name.startswith("LITELLM_API_KEY_") and _looks_like_real_key(value):
            return True
        if name.startswith("LLM_") and str(value).startswith("bedrock/"):
            return True
    return False
