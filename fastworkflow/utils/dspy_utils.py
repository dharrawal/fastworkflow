# Before anything imports dspy: dspy 3.4 puts a lazy stand-in for `openai` in
# sys.modules, and litellm importing an openai submodule through that stand-in
# fails with a circular import. Loaded first, the real openai is used instead.
# Every module that imports dspy carries the litellm line directly above its dspy import.
# The dspy-first case (dspy imported before fastworkflow) is repaired in fastworkflow/__init__.py.
import contextlib

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, Any, Union, get_args, get_origin, Tuple, List

import fastworkflow
from fastworkflow.utils.python_utils import is_union
from fastworkflow.utils import model_roles
from fastworkflow.utils.logging import logger


def lm_cache_enabled() -> bool:
    """Whether model responses may come from the DSPy response cache: False when ``FW_LM_CACHE`` is 0 or
    false (the env file ``init`` loaded, else the process environment). Every LM the framework builds asks
    this, so one switch turns the cache off for all of them (an experiment's repeated attempts must not
    replay an earlier attempt's answers)."""
    return fastworkflow.get_env_var("FW_LM_CACHE", default="1") not in ("0", "false", "False")


#: The model timeout of every role without its own setting, in seconds (``LLM_TIMEOUT_SECONDS``).
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

#: Roles whose built-in limits differ from the global one, as ``{role env var: (timeout seconds, retries)}``.
#: Parameter extraction answers in under 4 s when it answers; a provider request that stalls past 20 s
#: will not answer, and one retry of a fresh request usually does (fix-vdaj). The retries are DSPy's: the
#: timeout is per attempt, a failed attempt waits 1 s, then 2 s (``dspy.clients.execution._delay``), so
#: 20 s x (1 + 2) + 3 s = 63 s is the longest one call can hold a turn.
ROLE_LIMIT_DEFAULTS: dict[str, tuple[float, int]] = {"LLM_PARAM_EXTRACTION": (20.0, 2)}

_WARNED_LIMITS: set[tuple[str, str]] = set()


def _limit_setting(name: str, parse, default):
    """``name`` from the env (env files first, then the process: ``get_env_var``), parsed by ``parse``
    (which raises ``ValueError`` for a value out of range); unset is ``default``, and a value that does
    not parse logs one warning and is ``default``."""
    raw = fastworkflow.get_env_var(name, default="")
    if raw is None or not str(raw).strip():
        return default
    try:
        return parse(str(raw).strip())
    except ValueError:
        if (name, str(raw)) not in _WARNED_LIMITS:
            _WARNED_LIMITS.add((name, str(raw)))
            logger.warning(f"{name}={raw!r} is not a valid setting; using {default}")
        return default


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not value > 0 or value == float("inf"):
        raise ValueError(text)
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise ValueError(text)
    return value


def role_limits(model_env_var: str) -> tuple[float, Optional[int]]:
    """``(timeout seconds per attempt, retries)`` for the role whose model is ``model_env_var``.

    ``LLM_TIMEOUT_SECONDS_<ROLE>`` and ``LLM_NUM_RETRIES_<ROLE>``, ``<ROLE>`` being the name after ``LLM_`` (the
    suffix of ``LITELLM_API_KEY_<ROLE>``): ``LLM_TIMEOUT_SECONDS_PARAM_EXTRACTION``. Without them the role's
    ``ROLE_LIMIT_DEFAULTS`` entry applies (parameter extraction: 20 s, 2 retries); a role with no entry takes
    ``LLM_TIMEOUT_SECONDS`` (default 120) and the retries are ``None``, DSPy's own default. A value that is not a
    positive number of seconds, or not a non-negative integer, logs a warning and the default applies.
    """
    role = model_env_var[4:] if model_env_var.startswith("LLM_") else model_env_var
    built_in = ROLE_LIMIT_DEFAULTS.get(model_env_var)
    if built_in is not None:
        timeout_default, retries_default = built_in
    else:
        timeout_default = _limit_setting("LLM_TIMEOUT_SECONDS", _positive_seconds, DEFAULT_LLM_TIMEOUT_SECONDS)
        retries_default = None
    return (
        _limit_setting(f"LLM_TIMEOUT_SECONDS_{role}", _positive_seconds, timeout_default),
        _limit_setting(f"LLM_NUM_RETRIES_{role}", _non_negative_int, retries_default),
    )


def is_timeout_error(error: BaseException) -> bool:
    """Whether ``error``, or anything it was raised from, is a model request that timed out.

    DSPy raises ``LMTimeoutError`` when LiteLLM times out (Bedrock and the other LiteLLM routes) and
    ``LMTransportError`` ("read timed out ...") from its own client (``openai/`` and the other native routes);
    both carry the cause. Checked by type, then by the words "timed out" in the message.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (TimeoutError, litellm.Timeout)):
            return True
        if type(error).__name__ in ("LMTimeoutError", "TimeoutError", "ReadTimeout", "APITimeoutError"):
            return True
        if "timed out" in str(error).lower():
            return True
        error = error.__cause__ or error.__context__
    return False


@contextlib.contextmanager
def role_context(role: str, lm, **settings):
    """``dspy.context(lm=lm, **settings)`` for a call of the model setting *role* (``LLM_PLANNER``, ``JEV``, ...): the
    caller names its own model's setting, and each model-call span of the block carries it as ``role``
    (``tracing.model_role``), whichever LM object it was handed."""
    from fastworkflow.observability import tracing

    with dspy.context(lm=lm, **settings), tracing.model_role(role):
        yield


_ANNOUNCED_FALLBACKS: set[tuple[str, str]] = set()


def resolve_role(model_var: str) -> model_roles.RoleModel:
    """The model and key of the role whose model setting is *model_var*: THE fallback rule
    (``fastworkflow.utils.model_roles``), read with the one precedence rule (``fastworkflow.raw_env_var``).

    The base model is ``LLM_AGENT``, else ``LLM_PLANNER``; ``LLM_PLANNER`` falls back to ``LLM_AGENT``; every role
    of ``model_roles.FALLBACK_ROLES`` falls back to the base; ``LLM_RESPONSE_GEN`` and the distillation roles do not.
    The key follows the model and never crosses a provider. A model taken from another role is said once per role
    per process, at INFO, naming the role it came from.
    """
    resolved = model_roles.resolve(model_var, fastworkflow.raw_env_var)
    if resolved.inherited_from and (model_var, resolved.inherited_from) not in _ANNOUNCED_FALLBACKS:
        _ANNOUNCED_FALLBACKS.add((model_var, resolved.inherited_from))
        logger.info(f"{model_var} is not set: using {resolved.inherited_from}'s model ({resolved.model})"
                    + (f" and key ({resolved.key_var})" if resolved.key_var else ""))
    return resolved


def role_completion(model_var: str):
    """``litellm.completion`` for the role whose model setting is *model_var*, for the code that calls litellm directly
    (the two training-data generators): each call gets ``resolve_role``'s model and key (and the proxy base for a
    ``litellm_proxy/`` model) as arguments, whatever model or key the caller passed, and never through the global
    ``litellm.api_key``. ``litellm.completion`` is looked up at each call. The function carries
    ``ROLE_COMPLETION_MARK``, so a cache fingerprint can name it as the production backend."""

    def completion(**kwargs):
        resolved = resolve_role(model_var)
        if not resolved.model:
            raise ValueError(f"DSPy Language Model not provided. Set {model_var} environment variable.")
        kwargs["model"] = resolved.model
        kwargs.pop("api_key", None)
        if resolved.key:
            kwargs["api_key"] = resolved.key
        if resolved.provider == model_roles.PROXY_PROVIDER:
            if proxy_api_base := fastworkflow.get_env_var("LITELLM_PROXY_API_BASE", default=""):
                kwargs["api_base"] = proxy_api_base
        return litellm.completion(**kwargs)

    setattr(completion, ROLE_COMPLETION_MARK, model_var)
    return completion


#: The attribute ``role_completion`` marks its function with (the role's model setting).
ROLE_COMPLETION_MARK = "fastworkflow_role_completion"


def get_lm(model_env_var: str, **kwargs):
    """The role's LM, tagged with the role (``fastworkflow_role``) so a model-call span can say whose it is.

    ``_build_lm`` builds it; see there for the routing.
    """
    lm = _build_lm(model_env_var, **kwargs)
    try:
        lm.fastworkflow_role = model_env_var
    except Exception:  # noqa: BLE001 - a tag is an extra: an LM that refuses it is still the role's LM
        pass
    return lm


def _build_lm(model_env_var: str, **kwargs):
    """The dspy LM of the role whose model setting is *model_env_var* (``LLM_AGENT``, ``LLM_PARAM_EXTRACTION``, ...).

    The model and key are ``resolve_role``'s: the role's own model, else the one it falls back to, and the key that
    follows it. A ``litellm_proxy/`` model is routed through the LiteLLM Proxy at ``LITELLM_PROXY_API_BASE``, with
    ``LITELLM_PROXY_API_KEY`` when set (a no-auth proxy needs none). ``**kwargs`` go to ``dspy.LM``.

    Raises ValueError when the role has no model (its own or a fallback), or when a ``litellm_proxy/`` model has no
    ``LITELLM_PROXY_API_BASE``.
    """
    # `FW_LM_CACHE=0` defeats the DSPy response cache process-wide
    # (`fix-bn1` `[XR16]`). Without it, k repeated attempts of one task in a
    # pass^k experiment send identical prompts, hit the disk+memory cache that
    # dspy enables by default, and come back byte-identical -- so pass^k equals
    # pass@1 by construction and the run looks like a spectacular result.
    if not lm_cache_enabled():
        kwargs.setdefault("cache", False)
    # Without one, a provider request that stalls blocks the turn for minutes.
    timeout_s, num_retries = role_limits(model_env_var)
    kwargs.setdefault("timeout", timeout_s)
    if num_retries is not None:
        kwargs.setdefault("num_retries", num_retries)

    resolved = resolve_role(model_env_var)
    model = resolved.model
    if not model:
        logger.critical(f"Critical Error: DSPy Language Model not provided. Set {model_env_var} environment variable.")
        raise ValueError(f"DSPy Language Model not provided. Set {model_env_var} environment variable.")

    if resolved.provider == model_roles.PROXY_PROVIDER:
        proxy_api_base = fastworkflow.get_env_var("LITELLM_PROXY_API_BASE", default="")
        if not proxy_api_base:
            raise ValueError(
                f"Model '{model}' uses litellm_proxy/ prefix but LITELLM_PROXY_API_BASE is not set. "
                "Set LITELLM_PROXY_API_BASE to your LiteLLM Proxy URL (e.g., http://127.0.0.1:4000)."
            )
        logger.debug(f"Routing {model_env_var} through LiteLLM Proxy at {proxy_api_base}")
        kwargs["api_base"] = proxy_api_base

    return (
        dspy.LM(model=model, api_key=resolved.key, **kwargs)
        if resolved.key
        else dspy.LM(model=model, **kwargs)
    )

#: Models litellm refused a parameter for in this process (``params_rejected``). Their later calls
#: go with ``drop_params`` from the start (``call_dropping_rejected_params``).
_PARAMS_REJECTED: set[str] = set()


def params_rejected(error: BaseException) -> bool:
    """Whether ``error`` is litellm refusing, before it sends anything, a parameter the model does not take.

    That is ``litellm.UnsupportedParamsError`` anywhere in the cause chain (DSPy wraps it twice). Bedrock
    Claude Haiku 5.5, for one, takes only ``temperature=1``, so ``temperature=0.0`` is refused (fix-z62t).
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, litellm.UnsupportedParamsError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def drops_rejected_params(lm) -> bool:
    """Whether calls on ``lm`` leave out the parameters its model does not take (so its temperature may not apply)."""
    return bool((getattr(lm, "kwargs", None) or {}).get("drop_params")) or getattr(lm, "model", None) in _PARAMS_REJECTED


def call_dropping_rejected_params(lm, call):
    """``call(lm)``; when litellm refuses one of ``lm``'s parameters, ``call`` once more with ``drop_params=True``.

    With ``drop_params`` litellm leaves out what the model does not take (a temperature other than 1 on a
    model that takes only 1) and sends the rest. The refusal is raised before anything is sent, so the
    first try costs no tokens. The model is remembered for the process: its later calls go with
    ``drop_params`` from the start, and the first refusal is logged once at WARNING.

    A model that takes the parameters is called exactly as before: ``drop_params`` is never added to
    it. Adding it to every LM is not neutral: DSPy sends an LM with ``drop_params`` through LiteLLM
    even when it would call the provider natively (``openai/``, ``anthropic/``). An LM litellm refused
    is already on LiteLLM.
    """
    dropping = bool((getattr(lm, "kwargs", None) or {}).get("drop_params"))
    if getattr(lm, "model", None) in _PARAMS_REJECTED and not dropping:
        return call(lm.copy(drop_params=True))
    try:
        return call(lm)
    except Exception as error:  # noqa: BLE001 - re-raised unless litellm refused a parameter
        if dropping or not params_rejected(error):
            raise
        if lm.model not in _PARAMS_REJECTED:
            _PARAMS_REJECTED.add(lm.model)
            reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
            logger.warning(f"{lm.model} refused a parameter ({reason}); its calls now leave out the "
                           "parameters it does not take")
    return call(lm.copy(drop_params=True))


def _process_field(field_info, is_input: bool) -> Tuple[Any, Any, bool]:
    """Process a single field and return its type, DSPy field, and optional status."""
    field_type = field_info.annotation
    field_desc = field_info.description or f"{'Input' if is_input else 'Output'} field"
    
    # Handle Optional types
    is_optional = False
    if is_union(field_type):
        args = get_args(field_type)
        if type(None) in args:
            is_optional = True
            field_type = next((t for t in args if t is not type(None)), str)
    
    dspy_field = dspy.InputField(desc=field_desc) if is_input else dspy.OutputField(desc=field_desc)
    return field_type, dspy_field, is_optional


def _process_input_fields(model_class: Type[BaseModel], preserve_types: bool) -> Dict[str, Tuple]:
    """Process all input fields from a Pydantic model."""
    fields = {}
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=True)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
    return fields


def _process_output_fields(model_class: Type[BaseModel], preserve_types: bool) -> Tuple[Dict[str, Tuple], List[str]]:
    """Process all output fields from a Pydantic model and generate instructions."""
    fields = {}
    instructions = []
    
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=False)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
        
        # Generate field-specific instructions
        _add_field_instructions(field_name, field_info, instructions)
        
    return fields, instructions


def _add_field_instructions(field_name: str, field_info, instructions: List[str]) -> None:
    """Add instructions for a specific field based on its metadata."""
    if hasattr(field_info, 'default') and field_info.default is not None:
        instructions.append(f"For '{field_name}': Use '{field_info.default}' if not explicitly mentioned.")
    
    if hasattr(field_info, 'examples') and field_info.examples:
        examples_str = ", ".join(f"'{ex}'" for ex in field_info.examples)
        instructions.append(f"Examples for '{field_name}': {examples_str}")


def _create_instructions(custom_instructions: Optional[str], auto_instructions: List[str]) -> str:
    """Create the final instruction string from custom and auto-generated instructions."""
    if custom_instructions:
        return custom_instructions
    
    if not auto_instructions:
        return ""
        
    return "Extract the following fields based on the input:\n\n" + "\n".join(auto_instructions)


def dspySignature(
    Input_class: Type[BaseModel], 
    Output_class: Type[BaseModel],
    instructions: Optional[str] = None,
    preserve_types: bool = True
) -> Type[dspy.Signature]:
    """
    Dynamically creates a dspy.Signature class from Pydantic Input and Output models.

    Args:
        Input_class: A Pydantic BaseModel class defining the input fields.
        Output_class: A Pydantic BaseModel class defining the output fields.
        instructions: Optional custom instructions to include in the signature.
        preserve_types: Whether to preserve field type annotations in the signature.

    Returns:
        A new class that inherits from dspy.Signature.
    """
    if not issubclass(Input_class, BaseModel) or not issubclass(Output_class, BaseModel):
        raise TypeError("Input_class and Output_class must be subclasses of pydantic.BaseModel.")

    # Process fields from both classes
    input_fields = _process_input_fields(Input_class, preserve_types)
    output_fields, auto_instructions = _process_output_fields(Output_class, preserve_types)
    
    # Combine all fields and create instructions
    dspy_fields = {**input_fields, **output_fields}
    final_instructions = _create_instructions(instructions, auto_instructions)
    
    return dspy.Signature(dspy_fields, final_instructions.strip())

############################################
# Steps:
# 1. Define your signature and dspy function
# 2. Get prediction from DSPy module
# 3. Create output directly using ** unpacking

# dspy_signature_class = dspySignature(Signature.Input, Signature.Output)
# dspy_predict_func = dspy.Predict(dspy_signature_class)
# prediction = dspy_predict_func(input)  # Returns a dspy.Prediction object
# return Signature.Output(**prediction)