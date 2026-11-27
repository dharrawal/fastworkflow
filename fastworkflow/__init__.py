import os as _os
import sys as _sys


def _stop_litellm_loading_a_dotenv_file() -> None:
    """litellm, when imported, copies the first ``.env`` it finds searching upward from its own installed file into
    ``os.environ``, unless ``LITELLM_MODE`` is ``PRODUCTION``: that is the variable's only effect in litellm. Under the
    precedence rule (the process environment beats the env files) such a file would silently override the workflow's
    own env files, so fastworkflow turns that load off. A ``LITELLM_MODE`` the user set is left as it is; when litellm
    was imported before fastworkflow, its load has already happened and this can only say so."""
    _os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    litellm = _sys.modules.get("litellm")
    # litellm records the mode it was imported under; anything but PRODUCTION means it ran its .env load.
    if litellm is not None and getattr(litellm, "litellm_mode", "DEV") != "PRODUCTION":
        import logging
        logging.getLogger("fastWorkflow").warning(
            "litellm was imported before fastworkflow, so it may already have loaded a .env file into the process "
            "environment, where its values beat the workflow's env files. Import fastworkflow first, or set "
            "LITELLM_MODE=PRODUCTION before litellm is imported.")


_stop_litellm_loading_a_dotenv_file()

# dspy imported first leaves a lazy `openai` stand-in that breaks litellm's import;
# touching it loads the real package. Only then, so `import fastworkflow` stays light.
if "dspy" in _sys.modules and "openai" in _sys.modules:
    _sys.modules["openai"].__version__
import contextlib
from dataclasses import dataclass
import json
from datetime import datetime
from enum import Enum
import os
import time
from typing import Any, Optional, Union

from pydantic import BaseModel, model_validator
import mmh3


class Action(BaseModel):
    command_name: str
    command: str = ""   # the request text: the command text the checks and the ResponseGenerator are given
    parameters: dict[str, Optional[Union[str, bool, int, float, BaseModel]]] = {}
    workflow_id: Optional[int] = None    # when creating a new action, this is set by the workflow

def load_startup_action(value: str) -> "Action":
    """Build the Action for a ``--startup_action`` value.

    The value is a path to a JSON file holding the action object. An inline JSON
    object is also accepted when the value starts with "{". A missing file raises
    ValueError naming the path, so the CLI and the server report the same error.
    """
    text = value.strip()
    if text.startswith("{"):
        data = json.loads(text)
    else:
        try:
            with open(value, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise ValueError(
                f"--startup_action file not found: {value!r}. Pass a path to a JSON file, "
                "or an inline JSON object starting with '{'."
            ) from None
    return Action(**data)

_STARTUP_ACTION_FILE = "startup_action.json"


def workflow_startup_action(workflow_folder: str) -> Optional["Action"]:
    """The startup Action in ``<workflow_folder>/startup_action.json``, or None when the folder has no such file.

    A file that exists but is not a JSON object holding a valid Action is a ValueError naming the file: it is
    never skipped silently. The one reader of the workflow's own startup file, for every entry point.
    """
    path = os.path.join(workflow_folder, _STARTUP_ACTION_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("it must hold one JSON object")
        if unknown := sorted(set(data) - set(Action.model_fields)):
            import difflib

            close = difflib.get_close_matches(unknown[0], list(Action.model_fields), n=1)
            raise ValueError(f"unknown key {unknown[0]!r}{f'; did you mean {close[0]!r}?' if close else ''} (its "
                             f"keys: {', '.join(Action.model_fields)})")
        return Action(**data)
    except (ValueError, TypeError) as error:   # json errors and pydantic's ValidationError are ValueErrors
        raise ValueError(f"startup action file {path!r} is not valid: {error}") from None


def resolve_startup_action(
    workflow_folder: str, startup_command: Optional[str], startup_action: Optional[str]
) -> Optional["Action"]:
    """The startup Action a process starts with: the ``--startup_action`` value when given, nothing when only
    ``--startup_command`` was given, else the workflow folder's ``startup_action.json`` when it has one.
    Used by the entry points (``run``, the server); a caller that builds a session directly gets no startup.
    """
    if startup_action:
        return load_startup_action(startup_action)
    if startup_command:
        return None
    return workflow_startup_action(workflow_folder)


class Recommendation(BaseModel):
    summary: str
    suggested_actions: list[Action] = []

class CommandResponse(BaseModel):
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []
    recommendations: list[Recommendation] = []
    #: The command's ``Signature.Output`` instance, when the command built one.
    #: Optional and appended (arch §12.2): a command that does not set it behaves
    #: exactly as before. The agent reads ``response``, which for an ``Output`` the command
    #: returns directly is that Output as JSON; the observation archive also stores it as JSON beside the
    #: text (``RuntimeHandleArchive.command_runs``), so later code can read exact
    #: ids, labels and counts without parsing ``response``. A pydantic model in
    #: memory; a dict once a record has been dumped and read back.
    output: Any = None

class MCPToolCall(BaseModel):
    """MCP-compliant tool call request format"""
    name: str
    arguments: dict[str, Any] = {}

class MCPContent(BaseModel):
    """MCP content block"""
    type: str  # "text", "image", etc.
    text: Optional[str] = None
    # Add other content type fields as needed

class MCPToolResult(BaseModel):
    """MCP-compliant tool result format"""
    content: list[MCPContent]
    isError: bool = False
    # The command's Output as JSON (MCP ``structuredContent``); absent when the command returned none.
    structuredContent: Optional[dict[str, Any]] = None

class CommandTraceEventDirection(str, Enum):
    AGENT_TO_WORKFLOW = "agent_to_workflow"
    WORKFLOW_TO_AGENT = "workflow_to_agent"

@dataclass
class CommandTraceEvent:
    direction: CommandTraceEventDirection
    raw_command: str | None               # for AGENT_TO_WORKFLOW
    command_name: str | None              # for WORKFLOW_TO_AGENT
    parameters: dict | str | None
    response_text: str | None
    success: bool | None
    timestamp_ms: int
    # Logical-turn correlation key (additive, observability design §3.1 [X7]).
    # None only for events emitted outside any logical turn.
    turn_key: str | None = None

class Refused(ValueError):
    """The one way a command, or a planning command, declines: ``raise fastworkflow.Refused(message, field=None)``.

    A decision, not a crash. ``message`` is what the user is told. Without ``field``, the
    call answers ``CommandOutput(CommandResponse(response=message, success=False))``. With
    ``field`` (a field name, or a list of names), the framework resets those fields to
    missing, keeps the others, and asks the user for those fields again, with the message
    (the targeted re-ask a ``db_lookup`` rejection gave before it was retired). Raised from
    a model validator of the command's ``Input`` (``@model_validator``), it is a refusal of the arguments; raised from
    a ``ResponseGenerator.__call__`` or a planning command's ``__result__``, it is the call's answer: the command has
    run, so nothing is asked again, and ``field`` is named on the failed answer as its artifact ``refused_fields``
    (``as_output``). A ``ValueError``, so code that caught ``ValueError`` around a command's helpers keeps catching it.
    """

    def __init__(self, message: str = "", field=None):
        super().__init__(message)
        if field is None:
            names: tuple[str, ...] = ()
        elif isinstance(field, str):
            names = (field,)
        else:
            names = tuple(field)
            if not all(isinstance(name, str) and name for name in names):
                raise TypeError("Refused(field=...) takes a field name or a list of field names")
        self.fields: tuple[str, ...] = names

    def as_output(self) -> "CommandOutput":
        """The failed answer of a call that refused from its body: the message, and the fields it named as the
        artifact ``refused_fields`` (none named: no artifact)."""
        artifacts = {"refused_fields": list(self.fields)} if self.fields else {}
        return CommandOutput(command_response=CommandResponse(response=str(self), success=False, artifacts=artifacts))


class CommandOutput(BaseModel):
    """The result of one command execution.

    Note on ask_user entries (A7 role inversion): when ``command_name`` is
    ``"ask_user"``, the usual roles are inverted — ``command_parameters``
    holds the *agent's question* to the user, and the command response's
    ``response`` holds the *user's answer*. ``success=False`` on an ask_user
    entry means the question is still unanswered, not that anything failed.

    ``command_parameters`` honesty [A10]: in memory this is the typed Pydantic
    params instance (or a ``str`` question for ask_user). Record serialization
    emits ``model_dump()`` as a dict; restore therefore accepts ``dict`` as
    well. Declared as ``Any`` so dump→validate round-trips do not lie as ``str``.

    As of v3.0, each command carries exactly one ``command_response``. Turn-level
    multiplicity lives on ``TurnOutput.command_outputs`` / ``TurnResult``. Passing
    the legacy ``command_responses=[...]`` keyword raises ``ValueError``.
    """

    command_response: CommandResponse
    workflow_name: str = ""
    context: str = ""
    command_name: str = ""
    command_parameters: Any = None  # typed model in memory; dict in records [A10]
    started_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    # Joins this outcome to the span that produced it (arch §12.0 delta 1).
    # Optional with a default because §12.2 requires CommandOutput stay
    # compatible: every existing constructor call and every already-serialized
    # record must keep validating, and both do — an absent key reads as None.
    # None means "not dispatched through a call-id-stamping path", which is the
    # honest answer for a hand-built CommandOutput; it never means "no span".
    command_call_id: Optional[str] = None
    # [A7] structural marker for an ask_user exchange. Optional with a None
    # default for the same §12.2 reason as command_call_id: every existing
    # constructor call and every already-serialized record must keep
    # validating. None means "written before this field existed" and defers to
    # the name comparison in `is_ask_user`; True and False are authoritative.
    ask_user_entry: Optional[bool] = None
    # The record a command ran on, when a parent-context walk ran it on an enclosing
    # record rather than the current one (design 3.9, "Filing the run"). Both None when the
    # command ran on the current record. ``ran_in_context_name`` is "" for the global
    # scope and ``ran_in_context_clause`` is "" for a root with no printable identity.
    ran_in_context_name: Optional[str] = None
    ran_in_context_clause: Optional[str] = None
    # True when the command is the framework's own (a file of ``fastworkflow/_framework_commands``, the
    # metadata's ``framework`` flag): machinery the framework runs for itself, such as the plan and
    # sub-task reviews, not an application result. A semantic fact about the command, set from its
    # metadata when the output is recorded on the turn; it says nothing about how any client shows
    # it. Optional with a default for the same §12.2 reason as ``ask_user_entry``: None means "written
    # before this field existed" (a hand-built output, an old record), True and False are authoritative.
    framework: Optional[bool] = None
    # The request part this output was produced for: the 1-based number of the sub-task the framework
    # executed it under, the same number the turn's answer gives that part in its ``### N.`` heading. Nested
    # planning-command outputs carry the number of the sub-task that called them. None when the turn was not
    # split into several sub-tasks, for outputs that belong to no part (the plan and sub-task reviews) and for
    # records written before the field existed. A semantic fact about the execution, not a display instruction.
    # A sub-task that itself covers several items is one part. Set when the output is recorded on the turn,
    # never parsed from text; a value already set is kept.
    sub_task: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_command_responses(cls, data: Any) -> Any:
        """Reject the pre-v3.0 ``command_responses`` list keyword with a clear error."""
        if isinstance(data, dict) and "command_responses" in data:
            raise ValueError(
                "CommandOutput no longer accepts command_responses=[...]; "
                "pass command_response=CommandResponse(...) (singular). "
                "Turn-level multiplicity lives on TurnOutput.command_outputs."
            )
        return data

    @property
    def success(self) -> bool:
        return self.command_response.success

    @property
    def is_ask_user(self) -> bool:
        """True if this entry is an ask_user clarification exchange. [A7]

        `ask_user_entry` is authoritative when set; the name comparison is the
        fallback for records serialized before the field existed. It cannot be
        the primary test any more: since fix-ajv.16 a FAILURE output carries the
        real routed command name with `success=False`, which is byte-identical
        to an unanswered question, and root-context command names are
        UNQUALIFIED — so a workflow defining a root command called `ask_user`
        would have its failures collected by `complete_ask_user_entry`, which
        would overwrite the error with the user's answer and mark it successful.
        The guarantee now rests on a field the framework sets, not on no
        workflow ever choosing a name. fix-ajv.17.
        """
        if self.ask_user_entry is not None:
            return self.ask_user_entry
        return self.command_name == "ask_user"

    @property
    def question(self) -> Optional[str]:
        """The agent's question to the user (ask_user entries only). [A7]"""
        return self.command_parameters if self.is_ask_user else None

    @property
    def user_reply(self) -> Optional[str]:
        """The user's answer to an ask_user question, if any. [A7]"""
        if self.is_ask_user:
            return self.command_response.response
        return None

    @property
    def command_aborted(self) -> bool:
        return self.command_response.artifacts.get("command_name", None) == "abort"

    @property
    def command_handled(self) -> bool:
        return self.command_response.artifacts.get("command_handled", False) is True

    @property
    def not_what_i_meant(self) -> bool:
        return (
            self.command_response.artifacts.get("command_name", None)
            == "misunderstood_intent"
        )

    def to_mcp_result(self) -> MCPToolResult:
        """Convert CommandOutput to MCP-compliant format.

        The text content is the response text. When the command returned its ``Signature.Output``
        (or a dict of one, after a record is read back), ``structuredContent`` carries that Output
        as JSON, the data its tool's ``outputSchema`` describes (MCP spec field names).
        """
        output = self.command_response.output
        if isinstance(output, BaseModel):
            structured = output.model_dump(mode="json")
        elif isinstance(output, dict):
            structured = output
        else:
            structured = None
        return MCPToolResult(
            content=[
                MCPContent(type="text", text=self.command_response.response)
            ],
            isError=not self.success,
            structuredContent=structured,
        )

class _Ephemeral:
    """Sentinel: this state is deliberately not persisted.

    Returned from a serialization hook to say "I know about this and I choose
    nothing", which pins the session WITHOUT the warning that an absent hook
    earns. It is a distinct object rather than None so that a hook which falls
    off the end of a function is a reportable bug instead of silent consent.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "fastworkflow.EPHEMERAL"

    def __bool__(self) -> bool:
        return False


EPHEMERAL = _Ephemeral()


class UnsupportedStateVersion(Exception):
    """Raised by a from_state hook that will not migrate the version it was given.

    The framework treats this as "this snapshot is not restorable": the record is
    quarantined and the request fails, rather than state being partly applied.
    """

    def __init__(self, state_version: int, message: Optional[str] = None):
        self.state_version = state_version
        super().__init__(
            message or f"cannot migrate state written at version {state_version}"
        )


class ModuleType(Enum):
    """Specifies which part of a command's implementation to load."""
    INPUT_FOR_PARAM_EXTRACTION_CLASS = 0
    COMMAND_PARAMETERS_CLASS = 1
    RESPONSE_GENERATION_INFERENCE = 2
    CONTEXT_CLASS = 3

_chat_session: Optional["ChatSession"] = None
class ChatSessionDescriptor:
    """Descriptor for accessing the global chat session.""" 
    def __get__(self, obj, objtype = None):
        global _chat_session
        return _chat_session
    
    def __set__(self, obj, value):
        global _chat_session
        if _chat_session:
            raise RuntimeError("Cannot set chat session. It is already set.")
        _chat_session = value
# Create the descriptor instance
chat_session = ChatSessionDescriptor()


_env_vars: dict = {}
CommandContextModel = None
RoutingDefinition = None
RoutingRegistry = None
ModelPipelineRegistry=None


def init(env_vars: dict):
    global _env_vars, CommandContextModel, RoutingDefinition, RoutingRegistry, ModelPipelineRegistry
    _env_vars = env_vars

    # The import-time LOG_LEVEL could see only the process environment; now that the env files are loaded, the one
    # rule (``get_env_var``: process, then the files) applies to it too.
    if log_level := raw_env_var("LOG_LEVEL"):
        from .utils.logging import reconfigure_log_level
        reconfigure_log_level(log_level)

    # init before importing other modules so env vars are available
    from .command_context_model import CommandContextModel as CommandContextModelClass
    from .command_routing import RoutingDefinition as RoutingDefinitionClass
    from .command_routing import RoutingRegistry as RoutingRegistryClass
    from .model_pipeline_training import ModelPipeline

    # Assign to global variables
    CommandContextModel = CommandContextModelClass
    RoutingDefinition = RoutingDefinitionClass
    RoutingRegistry = RoutingRegistryClass
    ModelPipelineRegistry = ModelPipeline

    # Ensure DSPy logging is properly configured after all imports
    # This needs to happen after DSPy is imported by other modules
    import logging
    logging.getLogger("dspy").setLevel(logging.ERROR)
    logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)

    # ------------------------------------------------------------
    # Eager imports for heavy libraries that otherwise trigger lock
    # contention during the first wildcard command.  Importing them
    # once here (at server start-up) shifts the cost out of the
    # request path and takes advantage of Python's module cache.
    # ------------------------------------------------------------
    with contextlib.suppress(Exception):
        import datasets  # noqa: F401 – pre-load Hugging Face datasets

#: Where a setting's value came from (``env_source``).
SOURCE_PROCESS = "process environment"
SOURCE_ENV_FILE = "workflow env file"


def env_source(var_name: str) -> Optional[str]:
    """Which layer holds *var_name* under the one precedence rule (``get_env_var``): ``SOURCE_PROCESS``,
    ``SOURCE_ENV_FILE``, or None when neither sets it."""
    if var_name in os.environ:
        return SOURCE_PROCESS
    if _env_vars.get(var_name) is not None:
        return SOURCE_ENV_FILE
    return None


def raw_env_var(var_name: str) -> Optional[str]:
    """The text of *var_name* under the one precedence rule, or None when unset; no conversion and no warning.

    The rule: the process environment (a shell export, a container or CI variable) wins over the env files
    ``fastworkflow.init`` loaded, and the env files win over the code default. A process value that is the empty
    string counts as set, as ``os.environ`` has it: it hides the env file's value.
    """
    value = os.environ.get(var_name)
    if value is None:
        value = _env_vars.get(var_name)
    return None if value is None else str(value)


def get_env_var(var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """The value of *var_name*, converted to *var_type*: from the process environment, else the env files
    ``fastworkflow.init`` loaded, else *default* (returned as given). The one precedence rule of every setting
    (``raw_env_var``)."""
    value = raw_env_var(var_name)
    if value is None and default is not None:
        return default

    if value is None:
        from fastworkflow.utils.logging import logger
        logger.warning(f"Environment variable '{var_name}' does not exist and no default value is provided.")

    try:
        if value is None:
            return None
        if var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        elif var_type is bool:
            if value.lower() in ('true', '1'):
                return True
            elif value.lower() in ('false', '0'):
                return False
            else:
                raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.")
        return str(value)  # Default case for str
    except ValueError as e:
        raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.") from e

def get_fastworkflow_package_path() -> str:
    """Get the fastworkflow package directory.
    
    This works both in development (when working in the fastworkflow repo)
    and when fastworkflow is pip installed.
    
    Returns:
        str: Path to the fastworkflow package directory
    """
    return os.path.dirname(os.path.abspath(__file__))

def get_internal_workflow_path(workflow_name: str) -> str:
    """Get the path to an internal fastworkflow workflow.
    
    Args:
        workflow_name: Name of the workflow in the _workflows directory
        
    Returns:
        str: Full path to the internal workflow
    """
    return os.path.join(get_fastworkflow_package_path(), "_workflows", workflow_name)

def get_workflow_id(workflow_id_str: str) -> int:
    return int(mmh3.hash(workflow_id_str))

from .active_workflow import (
    get_active_workflow,
    push_active_workflow,
    pop_active_workflow,
    clear_workflow_stack,
)

# Turn-level result types (fastworkflow.turn). Imported here — after
# CommandResponse and CommandOutput are defined — so the forward references
# inside TurnResult can be resolved against this module's types.
from fastworkflow.turn import (
    TurnResult,
    TurnOutput,
    TurnStatus,
    FW_ARTIFACT_REF_KEY,
    mint_turn_key,
)

_turn_types_namespace = {
    "CommandResponse": CommandResponse,
    "CommandOutput": CommandOutput,
    "TurnOutput": TurnOutput,
}
TurnOutput.model_rebuild(_types_namespace=_turn_types_namespace)
TurnResult.model_rebuild(_types_namespace=_turn_types_namespace)

# Observability sinks (fastworkflow.observability.tracing / fastworkflow.observability.metrics, both
# stdlib-only). Embedders implement TraceSink/MetricsSink and wire them via
# WorkflowExecutionContext (observability design §3.1).
from fastworkflow.observability.tracing import (
    Span as Span,
    TraceSink as TraceSink,
    NoOpTraceSink as NoOpTraceSink,
)
from fastworkflow.observability.metrics import (
    MetricsSink as MetricsSink,
    NoOpMetricsSink as NoOpMetricsSink,
    LoggingMetricsSink as LoggingMetricsSink,
)

# The ChatSessionDescriptor instance used to occupy this name until the eager
# `from .chat_session import ChatSession` replaced it with the submodule.
# Leave the name unbound so attribute access loads that submodule on demand
# instead of publishing the descriptor, which is not the public object.
del chat_session

# Command class declarations (fastworkflow.command_declarations): the decorators a
# ResponseGenerator or PlanGenerator carries. Stdlib-only, so imported eagerly.
from fastworkflow.command_declarations import (
    UserApproval,
    descends_to,
    user_approval_for_execution,
    writes,
)

# First-use re-exports. Workflow pulls numpy via kvstore; ChatSession and
# WorkflowExecutionContext pull dspy, litellm, fastapi, and starlette.
_LAZY_ATTRS = {
    "Workflow": ("fastworkflow.workflow", "Workflow"),
    "ChatSession": ("fastworkflow.chat_session", "ChatSession"),
    "WorkflowExecutionContext": (
        "fastworkflow.workflow_execution_context",
        "WorkflowExecutionContext",
    ),
    "CommandCancelledError": (
        "fastworkflow.workflow_execution_context",
        "CommandCancelledError",
    ),
    "FieldOptions": ("fastworkflow.parameters.field_options", "FieldOptions"),
    "CopiedOptions": ("fastworkflow.parameters.field_options", "CopiedOptions"),
    "RecordNamedOptions": ("fastworkflow.parameters.field_options", "RecordNamedOptions"),
    "ConfirmsOptions": ("fastworkflow.parameters.field_options", "ConfirmsOptions"),
    "UiOptions": ("fastworkflow.parameters.field_options", "UiOptions"),
}

__all__ = (
    "Action",
    "Any",
    "BaseModel",
    "ChatSession",
    "ChatSessionDescriptor",
    "CommandCancelledError",
    "CommandContextModel",
    "CommandOutput",
    "CommandResponse",
    "CommandTraceEvent",
    "CommandTraceEventDirection",
    "EPHEMERAL",
    "Enum",
    "FW_ARTIFACT_REF_KEY",
    "LoggingMetricsSink",
    "MCPContent",
    "MCPToolCall",
    "MCPToolResult",
    "MetricsSink",
    "ModelPipelineRegistry",
    "ModuleType",
    "NoOpMetricsSink",
    "NoOpTraceSink",
    "Optional",
    "Recommendation",
    "RoutingDefinition",
    "RoutingRegistry",
    "Span",
    "TraceSink",
    "TurnOutput",
    "TurnResult",
    "TurnStatus",
    "Union",
    "UnsupportedStateVersion",
    "UserApproval",
    "Workflow",
    "descends_to",
    "user_approval_for_execution",
    "writes",
    "WorkflowExecutionContext",
    "active_workflow",
    "chat_session",
    "clear_workflow_stack",
    "context_budget",
    "contextlib",
    "dataclass",
    "datetime",
    "FieldOptions",
    "CopiedOptions",
    "RecordNamedOptions",
    "ConfirmsOptions",
    "UiOptions",
    "get_active_workflow",
    "get_env_var",
    "get_fastworkflow_package_path",
    "get_internal_workflow_path",
    "get_workflow_id",
    "init",
    "mint_turn_key",
    "mmh3",
    "model_validator",
    "observability",
    "os",
    "pop_active_workflow",
    "push_active_workflow",
    "time",
    "turn",
    "utils",
    "workflow",
    "workflow_execution_context",
)


def __getattr__(name: str):
    # PEP 562: Workflow/ChatSession/WorkflowExecutionContext import numpy, dspy, litellm, and fastapi.
    import importlib

    spec = _LAZY_ATTRS.get(name)
    if spec is not None:
        module_name, attr_name = spec
        module = importlib.import_module(module_name)
        for public_name, (lazy_module, lazy_attr) in _LAZY_ATTRS.items():
            if lazy_module == module_name:
                globals()[public_name] = getattr(module, lazy_attr)
        return globals()[name]
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{name}":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
        raise
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(__all__) | {key for key in globals() if not key.startswith("_")})
