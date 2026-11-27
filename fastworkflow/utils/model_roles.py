"""Which model and which key a model role uses: the one fallback rule (bd fix-ygxo).

The rule, read by every model reader through ``dspy_utils.resolve_role`` (and by provenance, a leaf module, through
``resolve`` with its own mapping):

- The BASE model is ``LLM_AGENT`` when set, else ``LLM_PLANNER`` (no-agent mode). ``LLM_PLANNER`` falls back to
  ``LLM_AGENT``. Every role of ``FALLBACK_ROLES`` falls back to the base. ``LLM_AGENT`` never inherits, and
  ``LLM_RESPONSE_GEN`` and the distillation roles have no fallback: they are used only when set.
- The KEY follows the model: the role's own ``LITELLM_API_KEY_*``; else, when the model came from another role, that
  role's key; else (for the roles that fall back) the base role's key, only when the two models name the same provider (the part before the first
  ``/``). A key is never sent to another provider, and there is no global key. A ``litellm_proxy/`` model is its own
  provider and always uses ``LITELLM_PROXY_API_KEY``. A ``bedrock/`` model needs no key (it authenticates from the
  AWS credential chain), so a missing key is no problem there.

A role whose setting is the empty string has no model of its own and falls back like an unset one.

Leaf module: the standard library only, so ``observability.provenance`` can read it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

AGENT = "LLM_AGENT"
PLANNER = "LLM_PLANNER"
RESPONSE_GEN = "LLM_RESPONSE_GEN"

#: The key setting of each role's model. Two names do not follow ``LITELLM_API_KEY_<ROLE>``:
#: ``LLM_COMMAND_METADATA_GEN`` pairs with ``LITELLM_API_KEY_COMMANDMETADATA_GEN``.
ROLE_KEY_VARS: dict[str, str] = {
    "LLM_SYNDATA_GEN": "LITELLM_API_KEY_SYNDATA_GEN",
    "LLM_PARAM_EXTRACTION": "LITELLM_API_KEY_PARAM_EXTRACTION",
    PLANNER: "LITELLM_API_KEY_PLANNER",
    AGENT: "LITELLM_API_KEY_AGENT",
    "LLM_CONVERSATION_STORE": "LITELLM_API_KEY_CONVERSATION_STORE",
    "LLM_COMMAND_METADATA_GEN": "LITELLM_API_KEY_COMMANDMETADATA_GEN",
    "LLM_OBSERVATION_SEARCH": "LITELLM_API_KEY_OBSERVATION_SEARCH",
    "LLM_POLICY_INTENT": "LITELLM_API_KEY_POLICY_INTENT",
    "LLM_DISTILLATION": "LITELLM_API_KEY_DISTILLATION",
    "LLM_TEACHER_AGENT": "LITELLM_API_KEY_TEACHER_AGENT",
    "LLM_TEACHER_PLANNER": "LITELLM_API_KEY_TEACHER_PLANNER",
    "LLM_STUDENT_AGENT": "LITELLM_API_KEY_STUDENT_AGENT",
    "LLM_STUDENT_PLANNER": "LITELLM_API_KEY_STUDENT_PLANNER",
    RESPONSE_GEN: "LITELLM_API_KEY_RESPONSE_GEN",
}

#: The roles that fall back to the base model (``LLM_AGENT``, else ``LLM_PLANNER``).
FALLBACK_ROLES: frozenset[str] = frozenset({
    "LLM_PARAM_EXTRACTION",
    "LLM_CONVERSATION_STORE",
    "LLM_SYNDATA_GEN",
    "LLM_OBSERVATION_SEARCH",
    "LLM_POLICY_INTENT",
    "LLM_COMMAND_METADATA_GEN",
})

#: The roles that take part in the fallback rule: their model, and their key, may come from the base role.
FALLS_BACK: frozenset[str] = FALLBACK_ROLES | {PLANNER}

PROXY_PROVIDER = "litellm_proxy"
PROXY_KEY_VAR = "LITELLM_PROXY_API_KEY"
#: Providers that authenticate without a ``LITELLM_API_KEY_*`` (the AWS credential chain).
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"bedrock"})


@dataclass(frozen=True)
class RoleModel:
    """What a role resolved to. ``model`` is None when the role has no model (its own or a fallback);
    ``inherited_from`` names the role whose model it is when not its own; ``key_var`` the setting the key came from
    (None with no key)."""

    role: str
    model: Optional[str]
    key: Optional[str]
    inherited_from: Optional[str]
    key_var: Optional[str]

    @property
    def provider(self) -> str:
        return provider(self.model)

    @property
    def needs_key(self) -> bool:
        """Whether a missing key is a problem: a model whose provider authenticates otherwise (Bedrock), and a proxy
        (whose key is optional), need none."""
        return bool(self.model) and self.provider not in KEYLESS_PROVIDERS and self.provider != PROXY_PROVIDER


def key_var_of(role: str) -> str:
    """The key setting of *role*: its ``ROLE_KEY_VARS`` entry, else ``LITELLM_API_KEY_<ROLE>``."""
    return ROLE_KEY_VARS.get(role) or f"LITELLM_API_KEY_{role[4:] if role.startswith('LLM_') else role}"


def provider(model: Optional[str]) -> str:
    """The provider prefix of a LiteLLM model string (``mistral`` of ``mistral/mistral-small-latest``), "" for none."""
    text = str(model or "").strip()
    return text.split("/", 1)[0].lower() if "/" in text else ""


def base_role(read: Callable[[str], Optional[str]]) -> Optional[str]:
    """The role that holds the base model: ``LLM_AGENT`` when set, else ``LLM_PLANNER`` when set, else None."""
    for role in (AGENT, PLANNER):
        if _value(read, role):
            return role
    return None


def _value(read: Callable[[str], Optional[str]], name: str) -> str:
    return str(read(name) or "").strip()


def resolve(role: str, read: Callable[[str], Optional[str]]) -> RoleModel:
    """The model and key of *role*, reading settings with *read* (a name -> value-or-None function)."""
    own = _value(read, role)
    source: Optional[str] = role if own else None
    if not own:
        if role == PLANNER:
            source = AGENT if _value(read, AGENT) else None
        elif role in FALLBACK_ROLES:
            source = base_role(read)
    if source is None:
        return RoleModel(role=role, model=None, key=None, inherited_from=None, key_var=None)
    model = _value(read, source)
    inherited_from = None if source == role else source

    if provider(model) == PROXY_PROVIDER:
        key = _value(read, PROXY_KEY_VAR)
        return RoleModel(role, model, key or None, inherited_from, PROXY_KEY_VAR if key else None)

    candidates = [role]
    if inherited_from:
        candidates.append(inherited_from)
    base = base_role(read)
    if (role in FALLS_BACK and base and base not in candidates and provider(model)
            and provider(_value(read, base)) == provider(model)):
        candidates.append(base)
    for candidate in candidates:
        key_var = key_var_of(candidate)
        if key := _value(read, key_var):
            return RoleModel(role, model, key, inherited_from, key_var)
    return RoleModel(role, model, None, inherited_from, None)
