"""The author-facing field options of a planning command or execution command ``Input`` (spec section 2).

An author writes plain pydantic ``Field(..., json_schema_extra={...})``. Each key names one behaviour; its
value is ``True`` or a dict of that key's own options. The loader (``planning_commands.option_params``,
``planning_commands.check_option_spec``) reads these keys and writes the same internal options as the
spellings they replace, so the runtime behaves as before.

``FieldOptions`` is a plain dict at runtime. Its TypedDicts give IDE completion and typo detection:

    json_schema_extra=FieldOptions(copied_from_command_text={"ignore": ["the"]})
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

#: ``copied_from_command_text``: the explicit strict check. The value is written in the command text (a ``str``, an
#: ``int`` or a ``list[str]``): a ``str`` takes the request's spelling, an ``int`` must be a number the request writes.
#: Its ``ignore`` lists words stripped from either end of a ``str`` before the match. A field WITHOUT it keeps the
#: extractor's value, which is not compared with the command text.
COPIED_FROM_COMMAND_TEXT = "copied_from_command_text"
#: ``record_named_in_command_text``: a ``str`` naming a record the command text names (a subject).
RECORD_NAMED_IN_COMMAND_TEXT = "record_named_in_command_text"
#: ``classifier_confirms``: a classifier checks the value again from the command text. Its dict form
#: (``ConfirmsOptions``) also takes ``words``, which is training data only (see ``ConfirmsOptions.words``).
CLASSIFIER_CONFIRMS = "classifier_confirms"
#: ``classifier_decides``: a bool the classifier alone decides from the command text.
CLASSIFIER_DECIDES = "classifier_decides"
#: ``command_text_after_command_name``: a ``str`` that takes the text after the command name (execution commands).
COMMAND_TEXT_AFTER_COMMAND_NAME = "command_text_after_command_name"
#: ``from_command_output``: the command and Output field (``"<command>.<field>"``) whose output supplies the value.
FROM_COMMAND_OUTPUT = "from_command_output"
#: ``ui``: presentation hints an application declares for any client that renders results or collects input
#: (``UiOptions``). Read by no check, extractor or planner; a client may follow them or ignore them.
UI = "ui"
#: ``examples_by``: training data only: read by the synthetic utterance generator; no check, extractor, planner or
#: client reads it. On the field whose example values depend on a choice (``Literal``) field of the same ``Input``:
#: ``{"<choice field>": {"<choice value>": [example values of this field, ...]}}`` (``ExamplesBy``).
EXAMPLES_BY = "examples_by"

#: Every author-facing option key.
OPTION_KEYS = (
    COPIED_FROM_COMMAND_TEXT,
    RECORD_NAMED_IN_COMMAND_TEXT,
    CLASSIFIER_CONFIRMS,
    CLASSIFIER_DECIDES,
    COMMAND_TEXT_AFTER_COMMAND_NAME,
    FROM_COMMAND_OUTPUT,
    UI,
    EXAMPLES_BY,
)


class CopiedOptions(TypedDict, total=False):
    """Options of ``copied_from_command_text`` (a dict form)."""

    ignore: list[str]


class RecordNamedOptions(TypedDict, total=False):
    """Options of ``record_named_in_command_text`` (a dict form)."""

    ignore: list[str]


class ConfirmsOptions(TypedDict, total=False):
    """Options of ``classifier_confirms`` (a dict form, on a ``Literal`` only): each value's meaning, and the words
    users write for a value. A dict with ``words`` and no ``meanings`` is the same check as ``True``."""

    #: Each declared value's meaning, read by the classifier check.
    meanings: dict[str, str]
    #: Training data only: read by the synthetic utterance generator; no check, extractor, planner or client reads
    #: it. A map from a declared value to the words or short phrases users write for it (``{"identity": ["person",
    #: "employee"], "group": ["team"]}``). Each list is non-empty, a word is listed once per value (ignoring case) and
    #: under one value only; a value may have no words.
    words: dict[str, list[str]]


#: ``examples_by`` (training data only: read by the synthetic utterance generator; no check, extractor, planner or
#: client reads it): ``{"<choice field>": {"<choice value>": [example values of this field, ...]}}``. The one outer
#: key names another field of the same ``Input`` whose type is a ``Literal`` (or ``Optional`` of one); each inner key
#: is a declared value of it, with a non-empty list of values valid for THIS field (its type, ``pattern`` and
#: bounds). A choice value may have no entry.
ExamplesBy = dict[str, dict[str, list]]


class UiOptions(TypedDict, total=False):
    """Options of ``ui`` (a dict form): presentation hints the application declares for any client that renders its
    results or collects its input. The framework only validates and passes them through in the field's JSON Schema; they
    change no check or extraction, and a client is free to ignore them.

    ``hidden``, ``hero``, ``format``, ``table``, ``tones`` and ``currency`` are for a ``Signature.Output`` field; ``widget`` is for a
    ``Signature.Input`` field.
    """

    #: Output: a hint that the field is not for a human-readable view of the result (it is still in the Output JSON).
    hidden: Literal[True]
    #: Output (a scalar): a hint that the field is the one to show first.
    hero: Literal[True]
    #: Output: a hint for how a client shows the value.
    format: Literal["currency", "percent", "code", "badge"]
    #: Input: ``"textarea"`` on a ``str`` field, ``"radio"`` on a ``Literal`` field.
    widget: Literal["textarea", "radio"]
    #: Output (a list of scalars): a group name. The list fields of one output with the same name are the
    #: columns of one table, in field order. Needs two visible members; a client should not infer columns from
    #: equal lengths alone, and should show no table when the declared lists differ in length.
    table: str
    #: Output (with ``format="badge"`` on a ``str``, ``Literal`` or ``Enum`` field, or a list of one): a map from a value
    #: to one of ``"good"``, ``"warning"``, ``"bad"``, ``"neutral"``. Keys are the value as text, matched
    #: case-insensitively after trimming (an ``Enum`` member by its value); on a ``Literal`` or ``Enum`` every key must
    #: be a declared value. A client uses the author's tone first and its own vocabulary for any other value.
    tones: dict[str, Literal["good", "warning", "bad", "neutral"]]
    #: Output (with ``format="currency"`` on an ``int`` or ``float`` field, or a list of one): a three-letter ISO 4217
    #: code such as ``"USD"`` (either case; a client upper-cases it). A client that cannot format the code shows a plain
    #: number.
    currency: str


class FieldOptions(TypedDict, total=False):
    """The ``json_schema_extra`` keys of a field of a planning command or execution command ``Input``.

    Optional: a plain dict with the same keys works. Leave a key out to turn its behaviour off.
    """

    copied_from_command_text: Union[Literal[True], CopiedOptions]
    record_named_in_command_text: Union[Literal[True], RecordNamedOptions]
    classifier_confirms: Union[Literal[True], ConfirmsOptions]
    classifier_decides: Literal[True]
    command_text_after_command_name: Literal[True]
    from_command_output: Union[str, list[str]]
    ui: UiOptions
    examples_by: ExamplesBy
