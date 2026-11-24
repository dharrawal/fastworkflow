"""The one fill stage: values from the model extractor, or from the forms an agent writes (bd fix-qoyz step 5).

``fill`` is the ONE call of the model extractor (``InputForParamExtraction.extract_parameters``, on LLM_PARAM_EXTRACTION)
for every kind of command and every caller: the execution path of the CME (``ParameterExtraction._llm_extract``), the
shared extractor outside the CME (``parameter_extraction.extract_arguments``), and a planning command's typed call
(``planning_commands.extracted_arguments``). It reads the text it is given with an optional *note* after it (what a
retry appends: the refusal, or what a phase found). It writes no diagnostics: the caller records what the span reads
(``extraction_method``, ``fields_reset_not_given``) from the ``Filled`` it gets back.

``fill_from_agent_text`` is the four forms an agent's own command text is read in CODE, before any model (xml tags,
``name=value``, one positional value, the bare command name), moved here from ``ParameterExtraction`` as pure
functions: the same results; the ``Filled`` it returns names the form (``source``), from which the caller writes the same
``extraction_method`` / ``deterministic_form`` diagnostics as before.

This module imports no model stack at import time: ``utils.signatures`` (which imports dspy) is read inside ``fill``.
"""
from __future__ import annotations

import ast
import contextlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from fastworkflow.field_options import COMMAND_TEXT_AFTER_COMMAND_NAME
from fastworkflow.parameters.rules import is_union
from fastworkflow.utils.input_validation import bound_fields, field_checked_class


def unwrap_optional(field_type):
    """
    Unwrap ``Optional[X]`` / ``Union[X, None]`` to its underlying type.

    Returns ``(inner_type, origin)`` where ``origin`` is ``get_origin`` of the
    unwrapped type (e.g. ``list`` for ``Optional[List[str]]``). Non-optional
    types are returned unchanged.
    """
    origin = get_origin(field_type)
    if is_union(field_type):
        if non_none_types := [
            t for t in get_args(field_type) if t is not type(None)
        ]:
            field_type = non_none_types[0]
            origin = get_origin(field_type)
    return field_type, origin


def with_note(text: str, note: str = "") -> str:
    """*text* with *note* after it, the way every retry puts a refusal (or what a phase found) after the request."""
    return f"{text}\n\n{note}" if note else text


@dataclass(frozen=True)
class Filled:
    """What a fill found: the values as an instance of the command's ``Input`` (built without validation), where
    they came from (``llm``, or the form that read them: ``xml``, ``key_value``, ``positional``, ``bare``), and the
    fields the extractor's own "is this field given" check reset to not given (name -> the check's probability), or None
    when the extractor reported no reset (the span then carries no ``fields_reset_not_given``)."""

    params: Any
    source: str
    reset_not_given: Optional[Dict[str, float]] = None
    #: True when no attempt gave an answer because every answer failed to parse into the fields' types (a value outside
    #: a ``Literal`` or enum): a produced value the typed answer refused, not a value that is missing.
    unparsed: bool = False


def fill(extractor, params_class: type, command_name: str, folderpath: Optional[str], *, text: Optional[str] = None,
         note: str = "") -> Filled:
    """The model extractor's values for *command_name*: ONE call of ``extract_parameters``.

    *extractor* is an ``InputForParamExtraction`` that holds the request text (``extractor.command``), or None, and
    then one is made for *text*. With a *note* the model reads the text with the note after it; the extractor's own
    text is put back afterwards, even when the call raises, because validation reads the request from there."""
    from fastworkflow.utils.signatures import InputForParamExtraction

    if extractor is None:
        extractor = InputForParamExtraction(command=with_note(text or "", note))
        request = None
    else:
        request = extractor.command
        if note:
            extractor.command = with_note(request, note)
    try:
        params = extractor.extract_parameters(params_class, command_name, folderpath)
    finally:
        if request is not None and note:
            extractor.command = request
    reset = getattr(extractor, "fields_reset_not_given", None)
    return Filled(params=params, source="llm",
                  reset_not_given={name: float(p) for name, p in reset.items()} if reset else None,
                  unparsed=bool(getattr(extractor, "unparsed", False)))


def fill_from_agent_text(command: str, params_class: type, command_name: str, *,
                         by_code: bool = False) -> Optional[Filled]:
    """The values an agent's own command text gives in code, or None (the model reads it then).

    Tried in order: ``<tag>value</tag>`` (``from_xml``), ``name=value`` (``from_key_values``, with *by_code* when the
    command text was written by code, which takes a quoted value whole), the one positional value
    (``from_positional``) and the bare command name (``from_bare``). The ``Filled`` names the form that answered
    (``source``: ``xml``, ``key_value``, ``positional``, ``bare``); the span's ``deterministic_form`` is that name for
    every form but xml."""
    params = from_xml(command, params_class)
    source = "xml"
    if params is None:
        params = from_key_values(command, params_class, command_name, code_written=by_code)
        source = "key_value"
    if params is None:
        params = from_positional(command, params_class, command_name)
        source = "positional"
    if params is None:
        params = from_bare(command, params_class, command_name)
        source = "bare"
    return None if params is None else Filled(params=params, source=source)


def from_xml(command: str, command_parameters_class: type[BaseModel]) -> Optional[BaseModel]:
    """
    Extract parameters from XML-formatted command using regex.

    Returns:
        BaseModel instance with extracted parameters, or None if parsing fails
    """
    field_names = list(command_parameters_class.model_fields.keys())

    # If no parameters are defined, return empty model immediately
    if not field_names:
        return command_parameters_class.model_construct()

    extracted_data = {}

    # Try to extract each parameter using XML tags
    if len(field_names) == 1:
        # If there's only one field, extract content from first XML tag
        pattern = r'<[^>]+>(.+?)</[^>]+>'
        if match := re.search(pattern, command, re.DOTALL):
            parameter_value = match[1].strip()
            extracted_data[field_names[0]] = parameter_value
    else:
        # Try to extract each parameter using XML tags
        for field_name in field_names:
            # Look for <field_name>value</field_name> pattern
            pattern = rf'<{re.escape(field_name)}>(.+?)</{re.escape(field_name)}>'
            # For list-typed fields, an agent may repeat the tag once per item
            # (e.g. <item_ids>a</item_ids> <item_ids>b</item_ids>). re.search
            # would keep only the first, silently dropping the rest, so collect
            # ALL occurrences with findall and hand the list parser every value.
            _, forigin = unwrap_optional(
                command_parameters_class.model_fields[field_name].annotation
            )
            if forigin in (list, List):
                matches = re.findall(pattern, command, re.DOTALL)
                if len(matches) > 1:
                    # Multiple repeated tags -> JSON array so the list parser
                    # below collects every value (not just the first).
                    extracted_data[field_name] = json.dumps([m.strip() for m in matches])
                elif len(matches) == 1:
                    extracted_data[field_name] = matches[0].strip()
            elif match := re.search(pattern, command, re.DOTALL):
                parameter_value = match[1].strip()
                extracted_data[field_name] = parameter_value

    # Check if we extracted values for ALL fields (safest criteria for LLM fallback)
    all_fields_extracted = len(extracted_data) == len(field_names)

    if all_fields_extracted:
        # Initialize all fields with their default values (if they exist) or None
        params_data = {}
        for field_name in field_names:
            field_info = command_parameters_class.model_fields[field_name]
            if field_info.default is not PydanticUndefined:
                params_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                params_data[field_name] = field_info.default_factory()
            else:
                params_data[field_name] = None

        # Parse and type-correct extracted values
        for field_name, raw_value in extracted_data.items():
            field_info = command_parameters_class.model_fields[field_name]
            field_type, origin = unwrap_optional(field_info.annotation)

            # Handle list types with robust parsing
            if origin in (list, List):
                inner_type = get_args(field_type)[0] if get_args(field_type) else str
                parsed_list = None

                # Try JSON array format
                if raw_value.startswith('[') and raw_value.endswith(']'):
                    with contextlib.suppress(Exception):
                        parsed = json.loads(raw_value)
                        if isinstance(parsed, list):
                            parsed_list = parsed

                # Try Python literal format
                if parsed_list is None:
                    with contextlib.suppress(Exception):
                        parsed = ast.literal_eval(raw_value)
                        if isinstance(parsed, list):
                            parsed_list = parsed

                # Try comma-separated format
                if parsed_list is None and ',' in raw_value:
                    parsed_list = [item.strip().strip('"').strip("'") for item in raw_value.split(',')]

                # Try space-separated format
                if parsed_list is None and ' ' in raw_value and not any(c in raw_value for c in ['[', ']', '{', '}', '"', "'"]):
                    parsed_list = [item.strip() for item in raw_value.split() if item.strip()]

                # Single value treated as single-item list
                if parsed_list is None:
                    cleaned = raw_value.strip().strip('"').strip("'")
                    parsed_list = [cleaned] if cleaned else []

                # Type-convert list items to match the inner type
                if parsed_list:
                    typed_list = []
                    for item in parsed_list:
                        if inner_type is str:
                            typed_list.append(str(item))
                        elif inner_type is int:
                            with contextlib.suppress(ValueError, TypeError):
                                typed_list.append(int(item))
                        elif inner_type is float:
                            with contextlib.suppress(ValueError, TypeError):
                                typed_list.append(float(item))
                        elif isinstance(inner_type, type) and issubclass(inner_type, BaseModel) and isinstance(item, dict):
                            with contextlib.suppress(Exception):
                                typed_list.append(inner_type(**item))
                        else:
                            typed_list.append(item)
                    params_data[field_name] = typed_list
                else:
                    params_data[field_name] = []
            else:
                # Non-list types: use raw value as-is
                params_data[field_name] = raw_value

        # Construct model without validation
        return command_parameters_class.model_construct(**params_data)

    return None

def is_own_command_token(token: str, command_name: str) -> bool:
    """Whether *token* is this command's name, short or qualified (``Ctx/cmd``)."""
    if not token or not command_name:
        return False
    token = token.lstrip("/").lower()
    return token in (command_name.lower(), command_name.split("/")[-1].lower())

def without_own_command_word(text: str, command_name: str) -> str:
    """*text* without a leading token that is this command's own name (``find_book SCOM`` -> ``SCOM``).

    What the grounding guard reads as the request: the command word an agent's text starts with is
    not something the user wrote, so a value only it spells (``Application`` in ``find_application``)
    is not written in the request. The parsers read the text whole.
    """
    head, _, rest = text.lstrip().partition(" ")
    return rest.strip() if is_own_command_token(head, command_name) else text

def bound_model(command_parameters_class: type[BaseModel],
                 converted: dict[str, Any]) -> Optional[BaseModel]:
    """The parameters model holding *converted* (already type-converted), or None.

    Each given value is checked against its own field (``bound_fields``: type, constraints and
    the field's own validators): a value that does not check is no binding. No model validator
    runs here (an ``Input``'s validators need the validation context, and ``validate_parameters``
    runs them with it), so a model validator's refusal reaches the caller as a refusal, with
    the binding kept. Fields not given take their defaults; a required one not given is left
    for validation to report as missing.
    """
    from pydantic import ValidationError

    try:
        checked = bound_fields(command_parameters_class, converted)
    except ValidationError:
        return None
    params_data: dict[str, Any] = {}
    for name, info in command_parameters_class.model_fields.items():
        if name in checked:
            params_data[name] = checked[name]
        elif info.default is not PydanticUndefined:
            params_data[name] = info.default
        elif info.default_factory is not None:
            params_data[name] = info.default_factory()
        else:
            params_data[name] = None
    return command_parameters_class.model_construct(**params_data)

def from_key_values(
        command: str, command_parameters_class: type[BaseModel],
        command_name: str = "", *, code_written: bool = False) -> Optional[BaseModel]:
    """Parameters from a ``name=value`` command (``query=Ann Lee start=2``), or None.

    Parsed in code, before the LLM fallback, only when the form is plain:

    - The text is an optional leading command token, then ``name=value``
      pairs; every name is a field of the parameters class, each at most once.
      The leading token must be *command_name* itself, short or qualified, so
      the arguments of a call routed to some other command are not bound
      here (fix-hia1.12).
    - A value runs to the next `` name=`` whose name is a field, or to the
      end. Surrounding quotes go, and so does a trailing parenthetical note
      (``" (this opens it)"``). ``None``/``null`` is no value.
    - Values are converted by the parameters class: the given fields must
      validate, and the only errors allowed are fields that were not given
      and are required (validation later reports those as missing).

    With *code_written* (``planning_phases.code_step`` and ``pick_step``), a quoted value is taken whole,
    with nothing inside it read as a key, a placeholder or a note (3.13). Without it, any doubt -- a ``word=``
    that is not a field, an XML tag, a placeholder (``<...>``), an
    empty value, a validation error -- returns None, and extraction goes on as before.
    """
    from pydantic import TypeAdapter, ValidationError
    from fastworkflow.observation_offloading.finish_check import _QUOTED_VALUE_RE

    fields = command_parameters_class.model_fields
    text = (command or "").strip()
    if not fields or "=" not in text:
        return None
    probe = (_QUOTED_VALUE_RE.sub(lambda m: "=" + "x" * (len(m.group(0)) - 1), text)
             if code_written else text)
    if re.search(r"</?[A-Za-z_][\w.-]*>", probe):
        return None
    keys = list(re.finditer(r"(?:^|(?<=\s))([A-Za-z_][A-Za-z0-9_]*)\s*=", probe))
    if not keys:
        return None
    lead = text[:keys[0].start()].strip()
    if lead and not is_own_command_token(lead, command_name):
        return None
    # Every ``word=`` must name a field: a value holding ``x=y`` is not plain.
    if any(m.group(1) not in fields for m in keys):
        return None
    names = [m.group(1) for m in keys]
    if len(set(names)) != len(names):
        return None

    raw: dict[str, Any] = {}
    for i, match in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(text)
        value = text[match.end():end].strip()
        was_quoted = False
        if code_written and len(value) >= 2 and value[0] in "\"'":
            q = re.match(r"""^("[^"\n]*"|'[^'\n]*')\s*$""", value)
            if q:
                value = q.group(1)[1:-1]
                was_quoted = True
            elif i + 1 == len(keys):
                # Try stripping a trailing parenthetical note after the closing quote
                trimmed = re.sub(r"\s*\([^()]*\)\s*\.?$", "", value).strip()
                q = re.match(r"""^("[^"\n]*"|'[^'\n]*')\s*$""", trimmed)
                if q:
                    value = q.group(1)[1:-1]
                    was_quoted = True
        if not was_quoted:
            if i + 1 == len(keys):
                value = re.sub(r"\s*\([^()]*\)\s*\.?$", "", value)
            value = value.strip().rstrip(",;").strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
                value = value[1:-1].strip()
            if not value or re.search(r"[<>{}]|\.\.\.|\u2026", value):
                return None
        elif not value:
            return None
        # unescape_punctuation (fix-6lt7) disabled after benchmark pass 4, where no escaped value occurred:
        # raw[match.group(1)] = (None if (not was_quoted and value in ("None", "null"))
        #                        else unescape_punctuation(value))
        raw[match.group(1)] = None if (not was_quoted and value in ("None", "null")) else value

    converted: dict[str, Any] = {}
    for name, value in raw.items():
        annotation = fields[name].annotation
        _, origin = unwrap_optional(annotation)
        if origin in (list, List) and isinstance(value, str):
            if value.startswith("["):
                try:
                    value = json.loads(value)
                except ValueError:
                    return None
            else:
                value = [part.strip().strip("\"'") for part in value.split(",")
                         if part.strip()]
        try:
            converted[name] = TypeAdapter(annotation).validate_python(value)
        except (ValidationError, TypeError, ValueError):
            return None
    return bound_model(command_parameters_class, converted)

def from_bare(
        command: str, command_parameters_class: type[BaseModel],
        command_name: str = "") -> Optional[BaseModel]:
    """The parameters' defaults for a command sent with nothing after its name, or None.

    Only when the text is empty or just *command_name* (short or qualified) and
    every field has a default; a command with a required field goes on as before.
    """
    text = (command or "").strip()
    if text and not is_own_command_token(text, command_name):
        return None
    if any(info.is_required() for info in command_parameters_class.model_fields.values()):
        return None
    # The defaults are checked field by field, with no model validator: an Input's validators need
    # the validation context, and a bare command's binding must not depend on them. A default that
    # does not check is no binding.
    try:
        checked = field_checked_class(command_parameters_class)()
    except Exception:  # noqa: BLE001 - a default that does not validate is no binding
        return None
    return command_parameters_class.model_construct(
        _fields_set=set(),
        **{name: getattr(checked, name) for name in command_parameters_class.model_fields})

def is_positional_field(field_info) -> bool:
    """A field marked ``json_schema_extra={'command_text_after_command_name': True}``."""
    extra = getattr(field_info, "json_schema_extra", None)
    return isinstance(extra, dict) and extra.get(COMMAND_TEXT_AFTER_COMMAND_NAME) is True

def from_positional(
        command: str, command_parameters_class: type[BaseModel],
        command_name: str) -> Optional[BaseModel]:
    """Parameters from a bare ``command_name value`` command, or None (fix-hia1.16).

    Parsed in code, before the LLM fallback, only when nothing is in doubt:

    - The text has no ``=`` and no XML, its first token is *command_name*
      itself (short or qualified), and a value follows.
    - The value goes to one target field: the field marked
      ``json_schema_extra={'command_text_after_command_name': True}``, or else the class's only
      ``str`` field (other types, such as an int ``start``, do not count).
      Two marked fields, or two ``str`` fields and no mark, is no target.
    - A leading word naming the target (``uid 203b``, ``query Ann Lee``) is
      dropped; a leading word naming another field (``other_uid 2b55``)
      declines.
    - A value holding ``* ` ? \\``, `` / ``, ``<>{}`` or no letter or digit
      declines. Matching surrounding quotes go. The whole remainder is the
      value: a trailing number is not split off into another field.
    - Only a marked field takes a value of several words; an unmarked field
      takes a single token, so prose sent to a free-text field still goes to
      the LLM.

    The value is converted by the field's type; the usual validation runs
    after, as for any other form.
    """
    from pydantic import TypeAdapter, ValidationError

    fields = command_parameters_class.model_fields
    text = (command or "").strip()
    if not fields or "=" in text or re.search(r"</?[A-Za-z_][\w.-]*>", text):
        return None
    parts = text.split(None, 1)
    if len(parts) < 2 or not is_own_command_token(
            parts[0], command_name):
        return None
    rest = parts[1].strip()

    marked = [name for name, info in fields.items()
              if is_positional_field(info)]
    if marked:
        if len(marked) != 1:
            return None
        target, multi_word = marked[0], True
    else:
        strings = [name for name, info in fields.items()
                   if unwrap_optional(info.annotation)[0] is str]
        if len(strings) != 1:
            return None
        target, multi_word = strings[0], False

    tail = target.rsplit("_", 1)[-1]
    words = rest.split(None, 1)
    first = words[0].rstrip(":").lower()
    if first in (target.lower(), tail.lower()):
        if len(words) < 2:
            return None
        rest = words[1].strip()
    elif first in {name.lower() for name in fields} or (
            len(words) > 1 and first.endswith(f"_{tail.lower()}")):
        return None

    if re.search(r"[*`?\\<>{}]| / ", rest) or not re.search(r"[A-Za-z0-9]", rest):
        return None
    if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
        rest = rest[1:-1].strip()
        if not re.search(r"[A-Za-z0-9]", rest):
            return None
    if not multi_word and len(rest.split()) > 1:
        return None
    try:
        value = TypeAdapter(fields[target].annotation).validate_python(rest)
    except (ValidationError, TypeError, ValueError):
        return None
    return bound_model(command_parameters_class, {target: value})
