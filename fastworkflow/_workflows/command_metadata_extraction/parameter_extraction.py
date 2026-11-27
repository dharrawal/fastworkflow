import contextlib
import dataclasses
import sys
import re
from typing import Any, Callable, Dict, List, Optional, get_args

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import fastworkflow
import dspy
from fastworkflow.utils.logging import logger
from fastworkflow import ModuleType
from fastworkflow.observability import tracing
from fastworkflow.observability.decision_signals import (
    DecisionUncertainty,
    UncertaintySignal,
    slot_binding_source,
)

from fastworkflow import no_agent_mode, waiting
from fastworkflow.parameters import parameter_clarification
from fastworkflow.parameters.parameter_clarification import Extraction, ParameterLoop
from fastworkflow.parameters import ask, fill, pipeline, read
from fastworkflow.parameters import argument_checks, signatures
from fastworkflow.utils import dspy_utils
from fastworkflow.parameters.signatures import (
    InputForParamExtraction,
    MISSING_INFORMATION_ERRMSG,
    NOT_FOUND, PARAMETER_EXTRACTION_ERROR_MSG, _planning_command_of)


INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max

# Versions of the mechanisms that bind a parameter, one per member of
# ``SLOT_BINDING_SOURCES``. ``signal_version`` identifies the producer of a value,
# and for an enumerated provenance that means the extractor itself: a change to the
# XML tag grammar or to the sentinel-driven merge changes what "xml_regex" or
# "stored_merge" is evidence of, without changing the recorded enum.
#
# ``llm`` is resolved at call time instead, because the model string is the thing
# that identifies that extractor and it comes from configuration.
_STATIC_SLOT_BINDING_SIGNAL_VERSIONS = {
    "stored_merge": "param-extraction/stored-merge/1",
    "xml_regex": "param-extraction/xml-regex/1",
    "validation_hook": "param-extraction/validation-hook/2",
}

# Resolved lazily and once: env access on every extraction logs a warning per turn when the var is unset.
LLM_PARAM_EXTRACTION_MODEL = None


def _slot_binding_signal_version(extraction_method: str) -> str:
    """Which version of which mechanism produced a binding-source signal.

    FW-REQ-021 clause 13 asks for thresholds to be re-validated when a model,
    prompt, or artifact that produces a signal changes; naming the LLM here is what
    makes an ``llm`` binding from one model distinguishable from another's.
    """
    global LLM_PARAM_EXTRACTION_MODEL
    if static_version := _STATIC_SLOT_BINDING_SIGNAL_VERSIONS.get(extraction_method):
        return static_version
    if not LLM_PARAM_EXTRACTION_MODEL:
        from fastworkflow.utils import dspy_utils

        LLM_PARAM_EXTRACTION_MODEL = dspy_utils.resolve_role("LLM_PARAM_EXTRACTION").model or "unknown"
    return f"param-extraction/llm/{LLM_PARAM_EXTRACTION_MODEL}"


def slot_binding_uncertainty(diagnostics: dict) -> Optional[DecisionUncertainty]:
    """The §6.6.1 record for the slot-binding decision *diagnostics* describes.

    A pure function of the facts ``_extract_impl`` recorded, and one-way: it reads
    the capture bag and writes nothing back, so extraction cannot come to depend on
    what is being measured about it (FW-REQ-021 P0 / arch §17.3 — capture only).

    Returns None when no binding decision was reached: a command with no parameters
    class binds no slots, and an extraction that raised before choosing a mechanism
    made no decision to characterise. Neither is an uninstrumented decision, so
    neither gets a record saying it was one.
    """
    extraction_method = diagnostics.get("extraction_method")
    if extraction_method is None:
        return None

    signals: list[UncertaintySignal] = [
        slot_binding_source(
            extraction_method,
            signal_version=_slot_binding_signal_version(extraction_method),
        )
    ]

    # Only a value the hook rewrote in place makes the hook the thing that bound the slot,
    # which is the sense in which SLOT_BINDING_SOURCES admits "validation_hook" as a source.
    # A refusal binds nothing and is recorded as an invalid field.
    if (diagnostics.get("validation_hook") or {}).get("changed_fields"):
        signals.append(
            slot_binding_source(
                "validation_hook",
                signal_version=_slot_binding_signal_version("validation_hook"),
            )
        )

    # The extractor produces one binding per slot and enumerates no alternatives to it: a
    # refusal's suggestions are words in its message, not structured candidates.
    candidate_count = 1

    # Requirements §4.14: uncertainty is reducible when evidence gathered inside the
    # task can lower it. A missing or invalid field is re-prompted and the next
    # round fills it, which is exactly that. Anything else is left as None — nobody
    # assessed it, which is not the same as "no".
    unresolved = diagnostics.get("missing_fields") or diagnostics.get("invalid_fields")
    return DecisionUncertainty(
        decision_kind="slot-binding",
        signals=tuple(signals),
        candidate_count=candidate_count,
        reducible=True if unresolved else None,
    )


def _recorded_slot_binding_uncertainty(span, diagnostics: dict) -> Optional[dict]:
    """``slot_binding_uncertainty`` in span-attribute form, never raising.

    Skipped when *span* is None: ``start_span`` declines without a sink or an open
    turn and ``end_span`` then discards its attributes, so assembling a record
    nobody will store is pure overhead against this slice's declared budget.

    ``tracing`` wraps every one of its own calls so a broken recorder degrades to
    a log line rather than a failed turn; this assembly runs just outside that
    guard, and capture that can lose a user's turn is the behavior change Phase 0
    exists to avoid.

    None also means no slot-binding decision occurred, which is a real outcome — so
    a capture failure reports itself instead of collapsing into the same value. The
    pure function above still raises, which is what the tests exercise.
    """
    if span is None:
        return None
    try:
        uncertainty = slot_binding_uncertainty(diagnostics)
    except Exception as exc:
        logger.warning(f"slot-binding uncertainty capture failed: {exc!r}")
        return {"capture_error": type(exc).__name__}
    return None if uncertainty is None else uncertainty.model_dump(mode="json")


#: A backslash before a character that is neither a letter, a digit, a backslash nor a space: an escape a model
#: writes in a tool call (``name\@domain``), which the value means without (fix-variance).
_ESCAPED_PUNCTUATION = re.compile(r"\\([^A-Za-z0-9\\\s])")


def unescape_punctuation(value: str) -> str:
    """*value* with each backslash escape of a non-alphanumeric character removed (``\\@`` -> ``@``, ``\\_`` -> ``_``).
    Other backslashes stay: ``C:\\new`` and a doubled backslash are not escapes of punctuation."""
    return _ESCAPED_PUNCTUATION.sub(r"\1", value)


_unwrap_optional = fill.unwrap_optional


def _unique(names) -> list:
    """*names* without repeats, in first-seen order."""
    return list(dict.fromkeys(names or []))


def _is_not_given(value: Any) -> bool:
    """Whether *value* is one of the extractor's not-given sentinels."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value == NOT_FOUND
    if isinstance(value, int):
        return value == INVALID_INT_VALUE
    if isinstance(value, float):
        return value == INVALID_FLOAT_VALUE
    return False


def extraction_of(params: Any, diagnostics: dict, reason: str = "") -> Extraction:
    """What an extraction of *params* found, as the shared contract (``parameter_clarification.Extraction``).

    ``missing`` and ``invalid`` come from the diagnostics ``validate_parameters`` filled (an invalid field that is also
    missing is reported once, as missing). ``values`` are the other fields as the user gave them, JSON-safe, without the
    not-given sentinels. *reason* is the refusal a caller words for the whole call (what a refusal that names no field is
    asked with, and the refusal of an extraction run outside the CME)."""
    missing = tuple(_unique(diagnostics.get("missing_fields")))
    invalid = tuple((name, "") for name in _unique(diagnostics.get("invalid_fields")) if name not in missing)
    unresolved = {*missing, *(name for name, _ in invalid)}
    try:
        dumped = params.model_dump(mode="json", warnings=False)
    except Exception:  # noqa: BLE001 - the values are optional; they must not fail the turn
        logger.warning("extracted values could not be dumped", exc_info=True)
        dumped = {}
    return Extraction(
        values={key: value for key, value in dumped.items() if key not in unresolved and not _is_not_given(value)},
        missing=missing, invalid=invalid, reason=reason)


def refusal_names_no_input(diagnostics: dict) -> bool:
    """Whether the validation of the values ended in a refusal that names no input: the ``Input``'s validators refused
    (the structured ``validation_hook`` fact ``validate_parameters`` writes) and named no field, or a guard or a rule
    of the roles refused the call and named no argument (``argument_refusal``, written only then or with a name:
    the names are the missing and invalid fields, which are written last)."""
    hook = diagnostics.get("validation_hook") or {}
    if hook.get("message") is not None and not hook.get("refused_fields"):
        return True
    return bool(diagnostics.get("argument_refusal")) and not (
        diagnostics.get("missing_fields") or diagnostics.get("invalid_fields"))


def parameter_error_text(command_name: str, error_msg: str) -> str:
    """The question the CME puts for a command whose values are not all given or are refused: the lead line, then the
    extractor's own message (what it extracted so far, what is missing or refused and why, how to get out)."""
    return ask.word(form=ask.FORM_LEAD, command_name=command_name, message=error_msg).text


def parameters_class_of(folderpath: str, command_name: str, *, pending: bool = False):
    """The class a command's parameters are read into: its ``Input``; and, when a parameter question of the command is
    *pending*, the slot input of a command the plan review asks one value of (``Signature.SlotInput``, which no fresh
    extraction reads). None when neither resolves."""
    routing = fastworkflow.RoutingRegistry.get_definition(folderpath)
    params_class = routing.get_command_class(command_name, ModuleType.COMMAND_PARAMETERS_CLASS)
    if params_class is None and pending:
        signature = routing.get_command_class(command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
        params_class = getattr(signature, "SlotInput", None) if signature else None
    return params_class


def _sentinel_for(params_class: type[BaseModel], name: str):
    """The not-given value of one field: an empty list for a list field, ``NOT_FOUND`` for the rest."""
    info = params_class.model_fields.get(name)
    _, origin = _unwrap_optional(info.annotation) if info is not None else (None, None)
    return [] if origin in (list, List) else NOT_FOUND


def model_of(loop: ParameterLoop, params_class: type[BaseModel]) -> BaseModel:
    """The pending loop as the extractor's working form: an instance of the command's ``Input`` built without
    validation (``model_construct``), the values kept and each open field holding its not-given value."""
    data = {name: value for name, value in loop.values.items() if name not in loop.open}
    data.update({name: _sentinel_for(params_class, name) for name in loop.open})
    return params_class.model_construct(**data)


def park(cme_workflow: fastworkflow.Workflow, command_name: str, params: BaseModel, *, question: str = "",
         request_text: str = "") -> ParameterLoop:
    """Leave the CME parked at *command_name*'s parameter question: *params* (built with ``model_construct``, the fields
    it still lacks holding their not-given values) become the pending loop, with *question* as the text the user saw.
    The one way a caller outside the extractor (the plan review's slot question) starts that state."""
    fields = type(params).model_fields
    asked = [name for name, info in fields.items() if ParameterExtraction._is_asked_field(info, getattr(params, name, None))]
    dumped = params.model_dump(mode="json", warnings=False)
    loop = ParameterLoop(command=command_name, kind=parameter_clarification.KIND_EXECUTION,
                         request_text=request_text, values={name: value for name, value in dumped.items() if name not in asked})
    parameter_clarification.record_round(loop, Extraction(values=loop.values, missing=tuple(asked)), question)
    parameter_clarification.store(cme_workflow.context, loop, after=parameter_clarification.AFTER_EDIT_SLOT)
    return loop


def round_ordinal(loop: Optional[ParameterLoop]) -> Optional[int]:
    """How many extraction attempts already ran for the pending loop: the 0-based ordinal of the attempt about to run (0
    is the first attempt, 1 the first retry). None when the loop's count is not a positive integer (a loop is only ever
    saved after a question, so a count below one is corruption, not a first attempt)."""
    if loop is None:
        return 0
    return loop.asks if type(loop.asks) is int and loop.asks > 0 else None


def _llm_guard(request: str, diagnostics: dict):
    """``(check, settle)`` of the execution command's LLM path, for ``pipeline.extract``: the guard
    (``argument_checks.check_extracted_values``) on the values against *request*, writing the refusals it finds
    (``argument_check_refusals`` and ``argument_check_retry`` on the first look) and, once the check stage is resolved
    (after the one second pass, if there was one), the refusals that remain (``argument_check_refusals_final``), the
    respellings and the not-given value of each refused field."""
    def check(params: BaseModel, attempt: int) -> pipeline.Verdict:
        found = argument_checks.check_extracted_values(params, request)
        if not found.refused:
            return pipeline.Verdict(detail=found)
        if attempt == 1:
            diagnostics["argument_check_refusals"] = found.refusal_texts()
            diagnostics["argument_check_retry"] = True
        return pipeline.Verdict(pipeline.KIND_REFUSED, "; ".join(found.refusal_texts()), found,
                                open=tuple(found.refused))

    def settle(params: BaseModel, verdict: pipeline.Verdict) -> None:
        found = verdict.detail
        if found.refused:   # what is still refused once the one second pass (if any) is over
            diagnostics["argument_check_refusals_final"] = found.refusal_texts()
        if found.respelled:
            diagnostics["argument_check_respelled"] = {
                name: str(spelled) for name, (_, spelled) in found.respelled.items()}
        if found.not_in_request:   # the advisory: field names only, never the values
            diagnostics["argument_check_not_in_request"] = list(found.not_in_request)
        found.apply(params)

    return check, settle


#: ``fw.command``'s ``parameters_read_by``, by the extraction's method (any other is its own value).
PARAMETERS_READ_BY = {None: "none", "xml_regex": "typed", "llm": "model"}
#: An extraction that found any of these decided something: its span is written even under a command.
_DECISION_KEYS = ("missing_fields", "invalid_fields", "argument_refusal", "argument_check_refusals",
                  "argument_check_refusals_final", "argument_check_retry", "fields_reset_not_given")


def traced_extraction(command_name: str, extract: Callable[[dict], Any], *, valid: Callable[[Any], bool],
                      part: Optional[int] = None):
    """``extract(diagnostics)``'s result, run in ONE ``fw.parameters`` span (D3 as amended; a child of the part it reads
    the values of, as ``intent_detection.traced_decision``'s span is). THE writer of
    that span: the CME's parameter stage (``ParameterExtraction.extract``) and the plan path's extraction for a part's own
    command (``review_proposals.draft_inputs``, with *part*, the part's 1-based number) both call it.

    *extract* fills the bag it is given as it works (the extraction method, whether this is a NOT_FOUND retry round, the
    STRUCTURED validation outcome from ``validate_parameters``: missing/invalid fields, the validation hook's facts, the
    argument checks' refusals and ``argument_check_not_in_request``); *valid* says from its result whether the values
    are valid. A failed extraction is a conversational state, not a span error: status stays ok and ``parameters_valid``
    carries the signal; only an exception marks the span error. Emission never affects extraction (no-op without a
    host). It also carries the §6.6.1 ``DecisionUncertainty`` for the slot-binding decision, assembled by
    ``slot_binding_uncertainty`` from those same diagnostics, written to the span and read by nothing else (FW-REQ-021
    P0: representation only)."""
    host = tracing.current_host()
    span = tracing.start_span(
        host,
        tracing.SPAN_PARAMETERS,
        command_name=command_name,
        attributes={"command_name": command_name, **({"part": part} if part is not None else {})},
        parent_span_id=tracing.direct_part_id(host) if part is None else None,
    )
    diagnostics: dict = {}
    try:
        output = extract(diagnostics)
    except BaseException:
        # No DecisionUncertainty here: the extraction did not complete, so there is no binding decision to
        # characterise. The raw diagnostics gathered so far still go out.
        tracing.end_span(host, span, status=tracing.STATUS_ERROR, attributes=diagnostics)
        raise
    # The command these values are for records how they were read (bd fix-qjne.34 step 6); a command's extraction
    # that decided nothing (a typed call read in code, or no values to read, all valid, no model call), on any path, is
    # not written: the command's span says it.
    method = diagnostics.get("extraction_method")
    is_valid = valid(output)
    tracing.note(tracing.SPAN_COMMAND, parameters_read_by=PARAMETERS_READ_BY.get(method, method),
                 parameters_valid=is_valid)
    if (is_valid and method in (None, "xml_regex") and not diagnostics.get("retry_round")
            and not any(diagnostics.get(key) for key in _DECISION_KEYS)
            and not tracing.had_children(span) and tracing.is_command_decision(host, span)):
        tracing.discard_span(host, span)
        return output
    tracing.end_span(
        host,
        span,
        status=tracing.STATUS_OK,
        attributes={
            **diagnostics,
            "decision_uncertainty": _recorded_slot_binding_uncertainty(span, diagnostics),
            "parameters_valid": is_valid,
            # the extraction reads the reply to a parameter question (a round resumed from the values kept)
            "reads_reply": bool(diagnostics.get("retry_round")),
            **({"part": part} if part is not None else {}),
        },
    )
    return output


class ParameterExtraction:
    class Output(BaseModel):
        parameters_are_valid: bool
        cmd_parameters: Optional[BaseModel] = None
        error_msg: Optional[str] = None
        suggestions: Optional[Dict[str, List[str]]] = None
        # Structured form of the error, for a human ("/") session only; None in agent mode (fix-weih.2).
        parameter_request: Optional[Dict[str, Any]] = None

    def __init__(self, cme_workflow: fastworkflow.Workflow, app_workflow: fastworkflow.Workflow, command_name: str, command: str):
        self.cme_workflow = cme_workflow
        self.app_workflow = app_workflow
        self.command_name = command_name
        self.command = command
        #: ``validate_parameters``' answer on the values ``_llm_extract`` returned (None when they did not come from it)
        self._validation = None

    def extract(self) -> "ParameterExtraction.Output":
        """Extract, wrapped in a ``fw.parameters`` span (D3 as
        amended). The span records the extraction method (stored-merge /
        xml-regex / llm), whether this is a NOT_FOUND retry round, and the
        STRUCTURED validation outcome from ``validate_parameters`` —
        missing/invalid fields, the validation hook's facts (the Input's model validators), and the
        validation_hook verdict. A failed extraction is a
        conversational state, not a span error: status stays ok and
        ``parameters_valid`` carries the signal; only an exception marks the
        span error. Emission never affects extraction (no-op without a host).

        It also carries the §6.6.1 ``DecisionUncertainty`` for the slot-binding
        decision, assembled by ``slot_binding_uncertainty`` from those same
        diagnostics. That record is written to the span and read by nothing else
        (FW-REQ-021 P0: representation only).
        """
        return traced_extraction(self.command_name, self._extract_impl, valid=lambda out: out.parameters_are_valid)

    def _extract_impl(self, diagnostics: dict) -> "ParameterExtraction.Output":
        app_workflow_folderpath = self.app_workflow.folderpath
        loop = parameter_clarification.pending(self.cme_workflow.context)
        if loop is not None and loop.command != self.command_name:
            loop = None
        command_parameters_class = parameters_class_of(
            app_workflow_folderpath, self.command_name, pending=loop is not None)
        if not command_parameters_class:
            return self.Output(parameters_are_valid=True)
        stored_params = model_of(loop, command_parameters_class) if loop is not None else None
        self._validation = None

        # The caller prefixes "<command_name>: " (wildcard.py). Strip exactly that,
        # so the parsers below see the text as sent ("cmd k=v", not ": cmd k=v")
        # and a value that happens to contain the command name is left alone
        # (fix-hia1.11).
        prefix = f"{self.command_name}:"
        self.command = (
            self.command[len(prefix):] if self.command.startswith(prefix)
            else self.command.replace(self.command_name, "")
        ).strip()

        input_for_param_extraction = InputForParamExtraction.create(
            self.app_workflow, self.command_name, 
            self.command)

        # If we have missing fields (in parameter extraction error state), try to apply the command directly
        diagnostics["retry_round"] = loop is not None
        # WHICH round, not just whether this is one (fix-8ko2): the loop counts its questions, and a loop is cleared
        # with the command it belongs to, so a count never outlives it. The attribute is omitted when the count cannot
        # be read, rather than sent as a 0 that would read as "first attempt".
        round_ordinal_value = round_ordinal(loop)
        if round_ordinal_value is not None:
            diagnostics["retry_round_ordinal"] = round_ordinal_value
        # Agentic mode is "not an assistant-mode command, in an agent session". Computed once: it
        # picks the extraction branch below and gates the parameter_request artifact (fix-weih.2).
        audience = self._audience()
        is_agentic_mode = audience == ask.AGENT
        if stored_params is not None:
            new_params = self._extract_and_merge_missing_parameters(stored_params, self.command)
            diagnostics["extraction_method"] = "stored_merge"
        else:
            if is_agentic_mode:
                # The agent's own command text is read in code first (``fill.fill_from_agent_text``): xml, the
                # ``name=value`` form (fix-hia1.10), the guarded positional form (fix-hia1.16) and the bare command
                # name. They stay the deterministic regex source, so the slot-binding signal contract does not
                # change; ``deterministic_form`` says which one bound.
                diagnostics["extraction_method"] = "xml_regex"
                filled = fill.fill_from_agent_text(
                    self.command, command_parameters_class, self.command_name,
                    by_code=bool(self.app_workflow.context.get("by_code")))
                new_params = None if filled is None else filled.params
                if filled is not None and filled.source != "xml":
                    diagnostics["deterministic_form"] = filled.source

                # If regex extraction fails, fall back to LLM-based extraction
                if new_params is None:
                    new_params = self._llm_extract(
                        input_for_param_extraction, command_parameters_class,
                        app_workflow_folderpath, diagnostics)

                # A lone value that bound to nothing is refused, not dropped: run
                # with defaults, the command does something other than what was
                # asked and reports success (fix-hia1.14).
                if unbound := self._unbound_lone_value(
                        self.command, new_params, command_parameters_class,
                        self.command_name):
                    diagnostics["unbound_value"] = unbound
                    return self.Output(
                        parameters_are_valid=False,
                        error_msg=(
                            self._unbound_value_message(unbound, command_parameters_class)
                            + "\nCheck your command name if the wrong command was executed."
                        ),
                        cmd_parameters=new_params)
            else:
                # Assistant mode ("/" commands): bind a plain ``name=value`` form in
                # code first (the same signal values as the agent branch); anything
                # else goes to the LLM as before.
                new_params = self._extract_parameters_from_key_values(
                    self.command, command_parameters_class, self.command_name)
                if new_params is not None:
                    diagnostics["extraction_method"] = "xml_regex"
                    diagnostics["deterministic_form"] = "key_value"
                else:
                    new_params = self._llm_extract(
                        input_for_param_extraction, command_parameters_class,
                        app_workflow_folderpath, diagnostics)

        # The Input's validators see the request with the user's replies after it (the text a planning command's
        # validators see: ``ParameterLoop.text``); the reply of this round joins the loop's replies first.
        if loop is not None and self.command:
            loop.replies.append(self.command)
        validation, self._validation = self._validation, None
        is_valid, error_msg, suggestions, missing_invalid_fields = validation if validation is not None else \
            input_for_param_extraction.validate_parameters(
            self.app_workflow, self.command_name, new_params,
            diagnostics=diagnostics,
            command_text=loop.text() if loop is not None else None,
            audience=audience,
        )

        if not is_valid and refusal_names_no_input(diagnostics):
            # A refusal that names no input (a validator's ``Refused(message)`` with no field, an ``Input`` rule that
            # names none, a validator that raised) is not a question: no single answer is being asked for. The command
            # ends as a failed output carrying the validator's message, with no pending parameter loop (the user reads
            # why and sends a new request; the agent reads it as the command's failure). A planning command's call is
            # treated the same way (``planning_commands.refusal_names_an_input``).
            parameter_clarification.clear(self.cme_workflow.context)
            return self.Output(parameters_are_valid=False, error_msg=error_msg, cmd_parameters=new_params)

        # Set all the missing and invalid fields to appropriate sentinel values before storing
        current_values = {
            field_name: getattr(new_params, field_name, None)
            for field_name in list(type(new_params).model_fields.keys())
        }
        for field_name in missing_invalid_fields:
            if field_name in current_values:
                # Determine appropriate sentinel value based on field type
                field_info = type(new_params).model_fields[field_name]
                _, origin = _unwrap_optional(field_info.annotation)

                # Use empty list for list fields, NOT_FOUND for others
                if origin in (list, List):
                    current_values[field_name] = []
                else:
                    current_values[field_name] = NOT_FOUND
        # Reconstruct the model instance without validation
        new_params = new_params.__class__.model_construct(**current_values)

        if not is_valid:
            error_msg = ask.word(
                form=ask.FORM_FRAME, message=error_msg, refusals=diagnostics.get("argument_check_refusals_final") or (),
                shown=self._format_parameters_for_display(new_params),
                audience=audience).text
            extraction = extraction_of(new_params, diagnostics)
            if loop is None:
                loop = ParameterLoop(
                    command=self.command_name, kind=parameter_clarification.KIND_EXECUTION, request_text=self.command)
            round_ = parameter_clarification.record_round(
                loop, extraction, parameter_error_text(self.command_name, error_msg))
            if round_.kind == parameter_clarification.EXHAUSTED:
                # The asking rule's stop (``parameter_clarification``): the second reply in a row that fixed none of the
                # values the last question listed. The command ends as a failed output naming what is missing, with no
                # pending question, in every session type (progress is known only here, after the extraction, so it
                # is not the reader's ``stop`` action).
                parameter_clarification.clear(self.cme_workflow.context)
                return self.Output(
                    parameters_are_valid=False,
                    error_msg=(f"{self.command_name} was not run: no usable answer was given in "
                               f"{no_agent_mode.MAX_NO_PROGRESS} replies in a row ({round_.describe})."),
                    cmd_parameters=new_params)
            parameter_clarification.store(self.cme_workflow.context, loop)
            return self.Output(
                parameters_are_valid=False,
                error_msg=error_msg,
                cmd_parameters=new_params,
                suggestions=suggestions,
                parameter_request=None if is_agentic_mode else loop.request)

        parameter_clarification.clear(self.cme_workflow.context)
        return self.Output(
            parameters_are_valid=True,
            cmd_parameters=new_params)

    def _audience(self) -> str:
        """Who is replying to this command's questions: the agent for its own tool call, a person for a "/" message
        (``ask.audience_of``: the one definition), also in an agent session."""
        assistant = self.cme_workflow is not None and waiting.port_of(self.cme_workflow.context).assistant
        return ask.audience_of(self.app_workflow.context, assistant=assistant)

    def _llm_extract(self, input_for_param_extraction: InputForParamExtraction,
                     command_parameters_class: type[BaseModel], app_workflow_folderpath: str,
                     diagnostics: dict) -> BaseModel:
        """The LLM extractor's values, checked against the command text it read (fix-qjf4).

        ``argument_checks.check_extracted_values`` checks the values (grounding, placeholders,
        allowed values, numbers). When it refuses one, the extractor is asked once more with the
        refusal appended to the command text (``REFUSED_EXTRACTION_NOTE``, as ``typed_planning_call``
        does); the second answer is checked against the original command text, and a value still
        refused takes the field's not-given value, so a required field is asked for (the NOT_FOUND
        path) and an optional one takes its default. The command's validators (``validate_parameters``) then judge
        the values, and ``pipeline.extract`` makes ONE second pass for whichever of the guard or the validators
        refuses a value first (never for a value that is simply missing). A planning command's ``Input`` is left to
        ``check_arguments``; ``FW_EXTRACTION_ARGUMENT_CHECKS=off`` turns the guard off.
        """
        first_unparsed: list = []

        def fill_text(note: str, attempt: int) -> BaseModel:
            filled = fill.fill(input_for_param_extraction, command_parameters_class, self.command_name,
                               app_workflow_folderpath, note=note)
            if attempt == 1 and filled.unparsed:
                first_unparsed.append(True)
            diagnostics["extraction_method"] = "llm"
            if filled.reset_not_given is not None:
                diagnostics["fields_reset_not_given"] = filled.reset_not_given
            return filled.params

        def validated(params: BaseModel, attempt: int) -> pipeline.Verdict:
            if attempt == 2:
                diagnostics.pop("missing_fields", None)
                diagnostics.pop("invalid_fields", None)
            result = input_for_param_extraction.validate_parameters(
                self.app_workflow, self.command_name, params, diagnostics=diagnostics, command_text=None,
                audience=self._audience())
            return _validation_verdict(result[0], result[1], diagnostics, result)

        guarded = argument_checks.checks_enabled() and \
            _planning_command_of(self.app_workflow, self.command_name) is None
        # The request is the command text the extractor read (self.command, prefix stripped, which
        # InputForParamExtraction.create was given): see argument_checks for why it is not more.
        check, settle = _llm_guard(fill.without_own_command_word(self.command, self.command_name), diagnostics) if guarded else (None, None)
        extracted = pipeline.extract(
            fill_text, check=check, settle=settle, validate=validated,
            retry_note=argument_checks.REFUSED_EXTRACTION_NOTE, unparsed=lambda: bool(first_unparsed),
            unparsed_refusal=pipeline.unparsed_refusal(command_parameters_class))
        self._validation = extracted.final.detail   # the validators' answer on these very values: read, not asked again
        return extracted.params

    @staticmethod
    def _format_parameters_for_display(params):
        """
        Format parameters for display in the error message.
        """
        if not params:
            return ""

        lines = []

        all_fields = list(type(params).model_fields.keys())

        for field_name in all_fields:
            value = getattr(params, field_name, None)

            if value in [
                NOT_FOUND,
                None,
                INVALID_INT_VALUE,
                INVALID_FLOAT_VALUE
            ]:
                continue

            # Skip empty lists (sentinel for missing list fields)
            if isinstance(value, list) and len(value) == 0:
                continue

            display_name = " ".join(word.capitalize() for word in field_name.split('_'))

            # Format fields appropriately based on type
            if (
                isinstance(value, bool)
                or not hasattr(value, 'value')
                and isinstance(value, (int, float))
                or not hasattr(value, 'value')
                and isinstance(value, str)
                or not hasattr(value, 'value')
            ):
                lines.append(f"{display_name}: {value}")
            else:  # Handle enum types
                lines.append(f"{display_name}: {value.value}")
        return "\n".join(lines)

    # The four forms an agent's own command text is read in code live in ``fastworkflow.parameters.fill`` (bd fix-qoyz
    # step 5); these names stay because the planning code steps and the CME's commands call them.
    @staticmethod
    def _extract_parameters_from_xml(command: str, command_parameters_class: type[BaseModel]) -> Optional[BaseModel]:
        """``fill.from_xml``: parameters from an XML-formatted command, or None."""
        return fill.from_xml(command, command_parameters_class)

    @staticmethod
    def _is_own_command_token(token: str, command_name: str) -> bool:
        """``fill.is_own_command_token``: whether *token* is this command's name, short or qualified (``Ctx/cmd``)."""
        return fill.is_own_command_token(token, command_name)

    @staticmethod
    def _bound_model(command_parameters_class: type[BaseModel],
                     converted: dict[str, Any]) -> Optional[BaseModel]:
        """``fill.bound_model``: the parameters model holding *converted* (already type-converted), or None."""
        return fill.bound_model(command_parameters_class, converted)

    @staticmethod
    def _extract_parameters_from_key_values(
            command: str, command_parameters_class: type[BaseModel],
            command_name: str = "", *, code_written: bool = False) -> Optional[BaseModel]:
        """``fill.from_key_values``: parameters from a ``name=value`` command, or None."""
        return fill.from_key_values(command, command_parameters_class, command_name, code_written=code_written)

    @staticmethod
    def _extract_parameters_from_bare(
            command: str, command_parameters_class: type[BaseModel],
            command_name: str = "") -> Optional[BaseModel]:
        """``fill.from_bare``: the parameters' defaults for a command sent with nothing after its name, or None."""
        return fill.from_bare(command, command_parameters_class, command_name)

    @staticmethod
    def _is_positional_field(field_info) -> bool:
        """``fill.is_positional_field``: a field marked ``command_text_after_command_name``."""
        return fill.is_positional_field(field_info)

    @staticmethod
    def _extract_parameters_from_positional(
            command: str, command_parameters_class: type[BaseModel],
            command_name: str) -> Optional[BaseModel]:
        """``fill.from_positional``: parameters from a bare ``command_name value`` command, or None."""
        return fill.from_positional(command, command_parameters_class, command_name)

    @staticmethod
    def _unbound_lone_value(command: str, params: Optional[BaseModel],
                            command_parameters_class: type[BaseModel],
                            command_name: str) -> Optional[str]:
        """The value of ``command_name value`` when it bound to no field, else None.

        Only a single letters-and-digits token counts (``_`` and ``-`` allowed), and
        only when every field of *params* is still unset: None, NOT_FOUND, an empty
        list or its default, with the token in none of them (fix-hia1.14).
        """
        parts = (command or "").split()
        if len(parts) != 2 or not ParameterExtraction._is_own_command_token(
                parts[0], command_name):
            return None
        token = parts[1]
        if not re.fullmatch(r"[A-Za-z0-9_-]*[A-Za-z0-9][A-Za-z0-9_-]*", token):
            return None
        if params is None:
            return token
        for name, info in command_parameters_class.model_fields.items():
            value = getattr(params, name, None)
            if value is None or value == NOT_FOUND or (isinstance(value, list) and not value):
                continue
            if info.default is not PydanticUndefined and value == info.default \
                    and token.lower() not in str(value).lower():
                continue
            return None
        return token

    @staticmethod
    def _unbound_value_message(token: str, command_parameters_class: type[BaseModel]) -> str:
        """Names the parameters that could take *token*, as ``name=token``."""
        from pydantic import TypeAdapter

        fields = command_parameters_class.model_fields
        takers = []
        for name, info in fields.items():
            with contextlib.suppress(Exception):
                TypeAdapter(info.annotation).validate_python(token)
                takers.append(name)
        names = takers or list(fields)
        return (f"'{token}' was not bound to any parameter; use "
                + " or ".join(f"{name}={token}" for name in names))

    @staticmethod
    def _is_asked_field(field_info, value) -> bool:
        """Whether a stored field is one the clarification asked the user for.

        ``_extract_impl`` stores each missing or invalid field as ``NOT_FOUND``
        (an empty list for a list field), so those are asked; so is a required
        field (annotation without ``None``) still holding a "not given" value.
        An Optional field left at ``None`` or an empty list was not asked for.
        """
        if value in [NOT_FOUND, INVALID_INT_VALUE, INVALID_FLOAT_VALUE]:
            return True
        is_optional = type(None) in get_args(field_info.annotation)
        return not is_optional and (
            value is None or (isinstance(value, list) and len(value) == 0))

    @staticmethod
    def _is_shown_value(value) -> bool:
        """Whether the question showed *value* ("Extracted parameters so far"): a real value, not a "not given" one."""
        return not (value in [NOT_FOUND, None, INVALID_INT_VALUE, INVALID_FLOAT_VALUE]
                    or (isinstance(value, list) and len(value) == 0))

    @staticmethod
    def _extract_and_merge_missing_parameters(stored_params: BaseModel, command: str):
        """
        Identify fields to fill by scanning for sentinel values and merge values
        parsed from the command string into a new params instance. This preserves
        existing behavior for token/field count mismatches and leaves values as
        strings (no type coercion).

        The reply is read by the one reader of parameter replies (``parameters.read``, the same rule a planning command's
        reply is read by): ``name: value`` lines set the fields the question listed (the asked ones and the ones it
        showed), one asked field takes the whole reply, several take its comma-separated parts in order when the counts
        match, and only asked fields are ever filled from a value without a name. A reply that does not read leaves the
        parameters as they were (the question is asked again).
        """
        # Initialize with existing values to avoid triggering validation
        model_fields = type(stored_params).model_fields
        field_names = list(model_fields.keys())
        params_data = {
            field_name: getattr(stored_params, field_name, None)
            for field_name in field_names
        }

        # Determine which fields still need user-provided input based on sentinels
        asked_fields = []
        for field_name in field_names:
            value = getattr(stored_params, field_name, None)
            if value in [
                NOT_FOUND,
                None,
                INVALID_INT_VALUE,
                INVALID_FLOAT_VALUE,
            ] or (isinstance(value, list) and len(value) == 0):
                if ParameterExtraction._is_asked_field(model_fields[field_name], value):
                    asked_fields.append(field_name)
        if not asked_fields:
            return stored_params

        listed = list(asked_fields) + [name for name in field_names if name not in asked_fields and
                                       ParameterExtraction._is_shown_value(getattr(stored_params, name, None))]
        slots = {name: read.slot_of(model_fields[name].annotation,
                                    required=type(None) not in get_args(model_fields[name].annotation))
                 for name in asked_fields}
        reading = read.read(command, listed, asked_fields, slots=slots)
        if reading.kind != read.VALUES:
            return stored_params
        params_data.update(reading.values)   # a value of None leaves an optional field empty

        # Return a new instance without validation
        return stored_params.__class__.model_construct(**params_data)


# ---------------------------------------------------------------------------
# Jev's reading of a reply to a missing-value prompt (fix-jd9o.6).
#
# In assistant mode the PARAMETER_EXTRACTION stage asks for the missing values and
# merges the next message as an answer. Jev reads that message first: an answer
# is merged as before; a hesitation waits; a request to stop, a wrong-command
# report or a different request leaves the pending round (wildcard.py dispatches it).
# ---------------------------------------------------------------------------

#: A non-answer intent applies only at or above this Jev probability (measured: 30/30 hesitations).
MISSING_VALUE_REPLY_THRESHOLD = 0.5
_MISSING_VALUE_REPLY_OFF = frozenset({"off", "0", "false"})
MISSING_VALUE_REPLY_OPTIONS = (
    ("answer", "The reply gives the value asked for, even if unsure or with a question tag ('X, I think?'); it may add or change other values."),
    ("different_command", "The reply asks the system to do something else instead, even when phrased as a question ('can you show me the users?')."),
    ("misunderstood", "The reply says the system picked the wrong command."),
    ("abort", "The reply asks to stop or cancel."),
    ("wait", "The reply asks for a moment or says the user is checking; it gives no value and asks nothing."),
)


def missing_value_reply_enabled() -> bool:
    """False when FW_JEV_MISSING_VALUE_REPLY is set to off, 0 or false (case-insensitive)."""
    value = str(fastworkflow.get_env_var("FW_JEV_MISSING_VALUE_REPLY", default="on"))
    return value.strip().lower() not in _MISSING_VALUE_REPLY_OFF


def classify_missing_value_reply(question: str, reply: str) -> str:
    """Jev's intent for a reply to the missing-value prompt: a non-answer intent at or above
    MISSING_VALUE_REPLY_THRESHOLD, otherwise "answer". Never raises: no Jev LM or any failure
    means "answer", which is the behaviour before this check.
    """
    if no_agent_mode._bare(reply) in read.YES_NO_VALUES:
        # A bare yes or no is the value the question asked for when its field is a yes/no one, and the reader of the reply
        # decides what it is otherwise (``parameters.read``): never "you misunderstood", and no model call.
        return "answer"
    try:
        from dspy.experimental import Choice
        from fastworkflow.workflow_agent import _jev_lm

        jev_lm = _jev_lm()
        if jev_lm is None:
            return "answer"

        class ClassifyMissingValueReply(dspy.Signature):
            """Decide what a user's reply to the system's question does. A reply that is a command written with a leading
            slash and a command name (``/find_book``) asks for that command instead: it is not the value asked for."""

            question: str = dspy.InputField(desc="the question the system asked the user")
            reply: str = dspy.InputField(desc="the user's reply")
            intent: Choice[MISSING_VALUE_REPLY_OPTIONS] = dspy.OutputField(desc="What does the reply do?")

        # The classifier reads the question without the note that only the answer is read: with the note in it, a
        # reply that names another command ("/find_book") was classified as an answer (measured, bd fix-rpio).
        question = "\n".join(line for line in str(question).splitlines()
                             if line.strip() != no_agent_mode.ANSWER_ONLY_NOTE)
        with dspy_utils.role_context("JEV", jev_lm):
            choice = dspy.Predict(ClassifyMissingValueReply)(question=question, reply=reply).intent
        value = str(choice.value)
        probability = (choice.probabilities or {}).get(value, choice.confidence)
        logger.info(f"missing-value reply {reply!r}: {value} (probability {float(probability):.2f})")
        if value != "answer" and probability >= MISSING_VALUE_REPLY_THRESHOLD:
            return value
        return "answer"
    except Exception:  # noqa: BLE001 - the reply is then an answer; never fail the turn
        logger.warning("could not classify the missing-value reply; treating it as an answer",
                       exc_info=True)
        return "answer"


class ExtractedArguments(BaseModel):
    """What ``extract_arguments`` found: the checked values (``params``, None when the command takes none) and the
    shared contract for them (``extraction``: what is still open, and the refusal as ``reason``); ``retried`` says the
    extractor was asked a second time with the refusal."""

    params: Optional[BaseModel] = None
    extraction: Extraction = Extraction()
    retried: bool = False
    #: A marked value the ``Input``'s validators rewrote, as the checks accepted it before (name -> value): the value a
    #: later check of these values reads (a draft's inputs), since the strict check reads the request's value and the
    #: run's validators write theirs again. ``params`` holds the rewrite.
    unvalidated: dict = {}
    #: The duplicate rule's notes of the optional fields it left not given (``ask.left_empty_line``, name -> line): the
    #: draft shows them on the review item.
    left_empty: dict = {}

    @property
    def valid(self) -> bool:
        return not self.extraction.names and not self.extraction.reason

    @property
    def refusal(self) -> Optional[str]:
        return self.extraction.reason or None

    @property
    def missing(self) -> List[str]:
        return list(self.extraction.missing)


def _start_argument_checks(app_workflow, extractor, command_name: str, params_class, text: str):
    """``(future, pool)`` of ``check_verdicts`` for *text*, started in a copy of this thread's context; or
    ``(None, None)`` when there is no Jev LM or the command has no ``verify`` parameter (nothing to start)."""
    import concurrent.futures
    from fastworkflow.planning import planning_commands
    from fastworkflow.workflow_agent import _jev_lm

    lm = _jev_lm()
    if lm is None:
        return None, None
    try:
        command = extractor.checked_command(app_workflow, command_name, params_class)
    except Exception as exc:  # noqa: BLE001 - validation resolves it again and raises where it always did
        logger.debug(f"argument checks not started early: {exc!r}")
        return None, None
    if not any(p.verify for p in command.params):
        return None, None
    pool = concurrent.futures.ThreadPoolExecutor(1)
    return pool.submit(tracing.traced_call(planning_commands.check_verdicts, lm, command, text)), pool


#: The diagnostics keys that record a value the model produced and a screen refused (it is then cleared, so the
#: field reads as "missing"): a placeholder, a number outside its bounds, a command's own name, a pronoun or a word
#: of the workflow's vocabulary, or a value the not-given classifier reset. The extractor's own "not given" marker is not
#: one of them.
_SCREENED_VALUE_KEYS = ("placeholder_values", "out_of_range_values", "command_name_values", "word_screen_values",
                        "fields_reset_not_given")


def _screen_refused_a_value(diagnostics: dict) -> bool:
    """Whether validation refused something the model produced (a field it names invalid, or a value a screen cleared)
    rather than finding a required value simply absent. Read from the diagnostics' structured keys."""
    return bool(diagnostics.get("invalid_fields")) or any(diagnostics.get(key) for key in _SCREENED_VALUE_KEYS)


def _validation_verdict(valid: bool, message: str, diagnostics: dict, detail) -> pipeline.Verdict:
    """The pipeline's verdict of one ``validate_parameters`` result: ok, ``missing`` when only required values are
    simply absent (nothing the model produced was refused: no second pass), else ``refused``. Read from the
    diagnostics' structured keys, never from the message text. *detail* is what the caller reads from it."""
    if valid:
        return pipeline.Verdict(detail=detail)
    missing = list(diagnostics.get("missing_fields") or [])
    reason = diagnostics.get("argument_refusal")   # a guard's own text (the question words it for its reader)
    refusal = (("missing required argument " + ", ".join(missing)) if missing and not
               diagnostics.get("invalid_fields") else
               reason if reason else (message or "").strip()) or "the arguments are refused"
    kind = pipeline.KIND_MISSING if missing and not _screen_refused_a_value(diagnostics) else pipeline.KIND_REFUSED
    return pipeline.Verdict(kind, refusal, detail, open=_open_fields(diagnostics))


def _open_fields(diagnostics: dict) -> Optional[tuple]:
    """The fields a refused validation leaves open (missing, invalid, or a value a screen cleared or the not-given
    check reset): the only ones the one second fill may change. None when the refusal names no input."""
    if refusal_names_no_input(diagnostics):
        return None
    names = [*(diagnostics.get("missing_fields") or []), *(diagnostics.get("invalid_fields") or []),
             *(name for key in _SCREENED_VALUE_KEYS for name in (diagnostics.get(key) or {}))]
    return tuple(dict.fromkeys(names)) or None


def extract_arguments(app_workflow: fastworkflow.Workflow, command_name: str, text: str,
                      diagnostics: Optional[dict] = None) -> ExtractedArguments:
    """The shared extractor's arguments for *command_name* on *text*, outside the CME workflow (fix-48np).

    The extraction a command's parameters get on the LLM path (``ParameterExtraction._llm_extract``: the
    DSPy extractor on LLM_PARAM_EXTRACTION, and for an execution command ``argument_checks`` with its one
    retry) and the validation they then get (``InputForParamExtraction.validate_parameters``: the screen,
    and for a planning command ``check_arguments``, its hook and its ``Input``'s rules, the checks
    ``typed_planning_call`` runs). Nothing is read from or written to a CME workflow: no stored parameters,
    no NLU stage, no xml/key=value parsing (that is for the agent's own command text). The app workflow's
    context is read as validation reads it (``raw_user_message``, ``run_as_agent``) and is not written.

    A planning command whose values are refused, or miss a required one, is extracted once more with the
    refusal after the text (``argument_checks.REFUSED_EXTRACTION_NOTE``, as ``typed_planning_call`` fills
    its call once more); the second answer is checked against *text* alone.
    """
    if diagnostics is None:    # a plain assignment, so the span-contract scan can follow the bag
        diagnostics = {}
    folder = app_workflow.folderpath
    params_class = fastworkflow.RoutingRegistry.get_definition(folder).get_command_class(
        command_name, ModuleType.COMMAND_PARAMETERS_CLASS)
    if not params_class:
        return ExtractedArguments()
    extractor = InputForParamExtraction.create(app_workflow, command_name, text)

    def validated(params: BaseModel, attempt: int) -> pipeline.Verdict:
        if attempt == 2:
            diagnostics.pop("missing_fields", None)
            diagnostics.pop("invalid_fields", None)
        as_extracted = params.model_copy()      # the screens clear a refused value in place: keep what was extracted
        valid, message, _, fields = extractor.validate_parameters(
            app_workflow, command_name, params, diagnostics, checks)
        verdict = _validation_verdict(valid, message, diagnostics, None)
        refusal = verdict.refusal
        found = extraction_of(params, {} if valid else diagnostics, "" if valid else refusal)
        if found.missing:
            # A value only a pronoun or a word of the workflow's vocabulary reads as missing; its reason goes with it,
            # so a question that asks for it says why the extracted one was not taken.
            reasons = signatures.word_screen_reasons(
                as_extracted, signatures._workflow_vocabulary(app_workflow), command_name)
            found = dataclasses.replace(found, screened=tuple((n, t) for n, t in reasons if n in found.missing))
        return dataclasses.replace(verdict, detail=ExtractedArguments(
            params=params, extraction=found, unvalidated=dict(getattr(extractor, "unvalidated", None) or {}),
            left_empty=dict(getattr(extractor, "left_empty", None) or {})))

    planning = _planning_command_of(app_workflow, command_name) is not None
    guarded = argument_checks.checks_enabled() and not planning

    first_unparsed: list = []

    def fill_text(note: str, attempt: int) -> BaseModel:
        filled = fill.fill(extractor, params_class, command_name, folder, note=note)
        if attempt == 1 and filled.unparsed:
            first_unparsed.append(True)
        # the first fill and the guard's retry report to the span's diagnostics; a planning command's retry does not
        if not (planning and attempt == 2):
            diagnostics["extraction_method"] = "llm"
            if filled.reset_not_given is not None:
                diagnostics["fields_reset_not_given"] = filled.reset_not_given
        return filled.params

    # The option checks read only the Jev LM, the static command and *text*: start them beside the
    # extraction (as typed_planning_call does) and join where validation reads them.
    checks, pool = _start_argument_checks(app_workflow, extractor, command_name, params_class, text)
    try:
        check, settle = _llm_guard(text, diagnostics) if guarded else (None, None)
        extracted = pipeline.extract(
            fill_text, check=check, settle=settle, validate=validated, unparsed=lambda: bool(first_unparsed),
            retry_note=argument_checks.REFUSED_EXTRACTION_NOTE, unparsed_refusal=pipeline.unparsed_refusal(params_class))
        result = extracted.final.detail
        result.retried = extracted.retried
        return result
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
