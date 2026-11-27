import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
# Compose an answer card and a question form with a small model, then verify everything it returned.
#
# The model (``LLM_RESPONSE_GEN``, usually a 7B-14B model) only *chooses*: a one-line lead, which result
# fields to show, which options to offer. Everything it returns is then checked in plain Python, and the
# failure direction of every check is "drop it":
#
# * a lead, a section heading or a section verdict holding a number or an ID that is not in the turn's data is
#   replaced by ``""`` (an empty heading becomes the answer's ``### N.`` title or the command's name; a section
#   with no verdict and no surviving block is dropped); a quantity in a lead or verdict that the answer does not
#   anchor (``_unanchored_sentence``) drops it too, so a total that exists elsewhere cannot attach to another noun;
# * a stat, table or record whose ``ref`` does not resolve to real data is dropped, and the ones that
#   survive carry the canonical ref ``r1.<key>``, so the browser binds to the data, not to model text;
# * a quote that is not in the answer is dropped;
# * a form option that is not a value or dict key in the results, or in the question, is dropped.
#
# Callers: ``POST /compose_answer_card`` (``compose_answer_card``) and ``POST /compose_question_form``
# (``compose_question_form``) in ``fastworkflow/run_fastapi_mcp/__main__.py``. Model calls go only through
# ``_composer_lm()`` and ``dspy_utils.call_dropping_rejected_params``.

import contextvars
import functools
import json
import re
import math
import time
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from fastworkflow.observation_offloading.facts import output_page
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.logging import logger

MAX_RESULTS = 60                # the page's TM_CARD_MAX_RESULTS
MAX_TEXT_CHARS = 20000          # the question form's limit on its question (the page's TM_QUESTION_MAX_CHARS)
MAX_ANSWER_CHARS = 150000       # the answer card's limit on the answer text (the page's TM_CARD_MAX_ANSWER)
# The model reads the whole answer up to ANSWER_PROMPT_WHOLE_CHARS; above it, an excerpt of about
# ANSWER_EXCERPT_CHARS (``answer_for_prompt``). Every check reads the full text, whatever the model was shown.
ANSWER_PROMPT_WHOLE_CHARS = 20000
ANSWER_EXCERPT_CHARS = 12000
ANSWER_EXCERPT_MIN_SECTION_CHARS = 300      # the least of a part's opening a section keeps in the excerpt
ANSWER_EXCERPT_PREAMBLE_CHARS = 1200        # the most of the text before the first heading
DIGEST_MAX_CHARS = 9000
LEAD_MAX_CHARS = 300
MAX_BLOCKS = 6                  # a card without sections (wire v1)
MIN_SECTIONS = 5                # the section cap of a request with up to 5 parts ...
MAX_SECTIONS = 24               # ... scaling with the parts of the request, only a safety ceiling against runaway output (the page mirrors it)
MAX_SECTION_BLOCKS = 4
MAX_CARD_BLOCKS = 10            # all sections together, up to the base cap of MIN_SECTIONS sections
MAX_CARD_BLOCKS_CEILING = 2 * MAX_SECTIONS    # and 2 per section after that, up to this many (the page mirrors it)
DISCRIMINATOR_MAX_CHARS = 80    # a result's label in the digest
HEADING_MAX_CHARS = 60
VERDICT_MAX_CHARS = 200
DESCRIPTION_MAX_CHARS = 100
MAX_STATS = 4
TABLE_ROWS_SHOWN = 8            # rows of a table the page shows before "+M more" (the page's TM_CARD_ROWS_SHOWN)
MAX_FIELDS = 6
MAX_OPTIONS = 12
COMPOSER_MAX_TOKENS = 1024
CARD_MAX_TOKENS = 1800                  # the output budget of a card of up to MIN_SECTIONS sections
CARD_TOKENS_PER_EXTRA_SECTION = 250     # and this much more for each section the cap allows beyond that (``card_token_budget``)
# A model that thinks spends part of ``max_tokens`` on reasoning the caller never sees: a measured Bedrock Haiku
# 5.5 used all 1800 in 4 of 4 runs and its JSON was cut mid-string. Such a model (``litellm.supports_reasoning``)
# gets this much more room, on top of asking it not to think (``composer_call_options``).
REASONING_HEADROOM_TOKENS = 3000
# The extra room of the one call after a first one that was cut off by its token limit.
TRUNCATION_RETRY_TOKENS = 2400
SERVER_COMPOSE_BUDGET_SECONDS = 40      # the endpoint's overall bound on one composition
PAGE_GIVE_UP_SECONDS = 35               # the page stops waiting after this long
# One correction call is skipped when the first pass alone took more than half the page's patience: a second
# call of that length would end past the point where the page has given up.
RETRY_SKIP_AFTER_SECONDS = PAGE_GIVE_UP_SECONDS / 2
RETRY_ENABLED = True                    # a switch for tests only
CORRECTIONS_MAX_CHARS = 800
DROP_DETAILS_MAX = 12                   # reason + detail pairs kept on the diagnostics record
# The drops that make a second look worth its latency: a section or a table, record or stats block lost for a
# reason the model can fix (a wrong ref, a kind or shape it got wrong, a section it wrote badly), or the lead
# lost. A dropped title, verdict sentence or quote, or a relabelled stat, loses little and never triggers one.
CORRECTABLE_DROPS = frozenset({
    "ref_unresolved", "not_a_table", "not_a_record", "bad_kind", "stats_empty", "empty_section",
    "section_unparseable", "block_unparseable", "section_not_a_dict", "block_not_a_dict", "sections_not_a_list",
    "lead_ungrounded", "lead_number_unanchored",
})
# Drops the correction call is TOLD about when it happens (the model sees them with the fixes above) but that do
# not trigger it alone: part of a block (a stat) or a sentence or a title.
CORRECTION_HINT_DROPS = frozenset({
    "stat_ref_unresolved", "stat_not_scalar", "stat_label_unsupported", "title_unsupported",
    "verdict_number_unanchored", "verdict_ungrounded", "quote_ungrounded", "sibling_mismatch",
    "example_copied",
})
NOT_CORRECTABLE_DROPS = frozenset({
    "block_cap", "section_cap", "stat_cap", "duplicate_stat", "duplicate_table", "no_content",
    "lead_only_with_results", "lead_restates", "model_failed",
    "quote_identifier",              # the block is harmless to lose; a second call would only cost time
    # a sentence or title the code removed because it said something the data does not (the page is right without it)
    "completeness_unsupported", "dangling_reference", "internal_wording", "judgement_unsupported",
    *CORRECTION_HINT_DROPS,
})
COMPOSER_ADAPTER = dspy.ChatAdapter()  # in DSPy 3.4 it falls back to JSONAdapter on a parse failure

_SCALARS = (str, int, float, bool, type(None))


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------


UNWRAP_LIMIT = 5


def _decode_json_values(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                value = parsed
        out[key] = value
    return out


def normalize_output_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON-looking string values, then unwrap one-key wrappers holding a record.

    ``retail_workflow`` returns ``{"order_details": "<json text>"}``; the useful fields are inside.
    This is what the chat's ``tmNormalizeOutputObject`` does, step for step (decode; then, at most
    ``UNWRAP_LIMIT`` times, replace a one-key output whose value is a non-empty dict by that dict,
    decoded again), so a ref written against this copy resolves in the browser. The browser sends
    its own normalized copy, and normalizing that again changes nothing.
    """
    out = _decode_json_values(raw)
    for _ in range(UNWRAP_LIMIT):
        if len(out) != 1:
            break
        inner = next(iter(out.values()))
        if not isinstance(inner, dict) or not inner:
            break
        out = _decode_json_values(inner)
    return out


class ResultData(BaseModel):
    id: str
    command: str = ""
    context: str = ""
    parameters: dict[str, Any] = {}
    output: dict[str, Any]
    titles: dict[str, str] = {}
    descriptions: dict[str, str] = {}   # schema description per output key (the page sends them)
    hints: dict[str, str] = {}          # schema ui hint per key: "hero", "badge", "hidden", a format, joined by ","
    supporting: bool = False            # an upstream lookup the page classed as supporting
    sub_task: Optional[int] = None      # the request part (1-based, as the answer's "### N." headings) the command ran for

    @field_validator("output", mode="after")
    @classmethod
    def _unwrap_output(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_output_dict(value)

    @field_validator("sub_task", mode="before")
    @classmethod
    def _part_lenient(cls, value: Any) -> Optional[int]:
        """The part number the recorded output carried: an integer in ``1..MAX_SUB_TASKS``, else None. Lenient on
        purpose (never a refused request): a missing or out of range number means "no part known", which is the
        heading-parsing behaviour every older page and turn has."""
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= MAX_SUB_TASKS:
            return value
        return None


MAX_TURN_KEY_CHARS = 200


def _turn_key_or_none(value: Any) -> Optional[str]:
    """The turn key a page sent so the server can tie the compose call to its turn's trace, or None.

    Lenient on purpose: a missing, non-text or oversized key means "unattached", never a refused request, because
    the key only decides where a diagnostics span is filed and a composed answer must not depend on it.
    """
    if isinstance(value, str) and 0 < len(value) <= MAX_TURN_KEY_CHARS:
        return value
    return None


MAX_PART_TITLE_CHARS = 200


class PartRunData(BaseModel):
    """One run of a part continued after a step-cap hand-off: ``index`` of ``of``, its answer at ``start``..``end``."""
    index: int
    of: int
    start: int
    end: int


class PartData(BaseModel):
    """One part of a split turn as the core recorded it (``fw_sub_tasks`` in the final answer's artifacts)."""
    number: int
    text: str = ""
    title: str = ""
    answered: bool
    start: Optional[int] = None         # where the part's answer begins in the request's ``answer``; None when unanswered
    end: Optional[int] = None
    runs: Optional[list[PartRunData]] = None   # the runs of a continued part; read by no prompt, a malformed list is None

    not_run: bool = False   # no-agent mode: the part's text is a stop's explanation; read by no prompt, ignored

    @field_validator("not_run", mode="before")
    @classmethod
    def _not_run_lenient(cls, value: Any) -> bool:
        return value is True

    @field_validator("runs", mode="before")
    @classmethod
    def _runs_lenient(cls, value: Any) -> Optional[list[PartRunData]]:
        """Runs are optional detail for a client: a malformed list never discards the part (or the parts)."""
        try:
            return [PartRunData.model_validate(item) for item in value] if isinstance(value, list) else None
        except ValidationError:
            return None


def _parts_or_none(value: Any) -> Optional[list[PartData]]:
    """The parts a page sent, or None. Lenient on purpose, like ``turn_key``: parts are an optional, more exact source of
    what the answer's ``### N.`` headings say, so a missing or malformed list (not a list, fewer than two or more than
    ``MAX_SUB_TASKS`` parts, an item that is not a part, numbers not strictly increasing within ``1..MAX_SUB_TASKS``, a
    text over ``MAX_TEXT_CHARS`` or a title over ``MAX_PART_TITLE_CHARS``) means "no parts": the headings are read as
    they always were, never a refused request."""
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_SUB_TASKS:
        return None
    try:
        parts = [item if isinstance(item, PartData) else PartData.model_validate(item) for item in value]
    except ValidationError:
        return None
    numbers = [part.number for part in parts]
    if numbers != sorted(set(numbers)) or numbers[0] < 1 or numbers[-1] > MAX_SUB_TASKS:
        return None
    if any(len(p.text) > MAX_TEXT_CHARS or len(p.title) > MAX_PART_TITLE_CHARS for p in parts):
        return None
    return parts


def _parts_fit(parts: list[PartData], answer: str) -> bool:
    """Whether the parts describe *answer*: each answered part's offsets lie in order inside it, and two or more answered
    parts are each preceded by their own ``### N. title`` heading line. The core builds the parts from the very text it
    returns; an answer that was changed on the way (or parts of another turn) fails this, and the headings are read."""
    answered = [p for p in parts if p.answered]
    floor = 0
    for part in answered:
        if part.start is None or part.end is None or not floor <= part.start <= part.end <= len(answer):
            return False
        if len(answered) >= 2:
            heading = f"### {part.number}. {part.title}\n"
            if part.start < floor + len(heading) or answer[part.start - len(heading):part.start] != heading:
                return False
        floor = part.end
    return all(p.start is None and p.end is None for p in parts if not p.answered)


class AnswerCardRequest(BaseModel):
    user_message: str = ""
    answer: str
    status: str = "completed"
    results: list[ResultData] = []
    use_dspy_cache: Optional[bool] = None
    turn_key: Optional[str] = None      # the logical turn key the page holds; the compose span is filed under it
    parts: Optional[list[PartData]] = None   # the parts of a split turn, as data; None: read the answer's headings

    @field_validator("parts", mode="before")
    @classmethod
    def _parts_lenient(cls, value: Any) -> Optional[list[PartData]]:
        return _parts_or_none(value)

    @model_validator(mode="after")
    def _parts_match_the_answer(self) -> "AnswerCardRequest":
        if self.parts is not None and not _parts_fit(self.parts, self.answer):
            self.parts = None
        return self

    @field_validator("turn_key", mode="before")
    @classmethod
    def _turn_key_lenient(cls, value: Any) -> Optional[str]:
        return _turn_key_or_none(value)

    @field_validator("answer", mode="after")
    @classmethod
    def _answer_limit(cls, value: str) -> str:
        if len(value) > MAX_ANSWER_CHARS:
            raise ValueError(f"answer is longer than {MAX_ANSWER_CHARS} characters")
        return value

    @field_validator("results", mode="after")
    @classmethod
    def _results_limit(cls, value: list[ResultData]) -> list[ResultData]:
        if len(value) > MAX_RESULTS:
            raise ValueError(f"more than {MAX_RESULTS} results")
        return value


class QuestionFormRequest(BaseModel):
    question: str
    user_message: str = ""
    results: list[ResultData] = []
    use_dspy_cache: Optional[bool] = None
    turn_key: Optional[str] = None      # see AnswerCardRequest.turn_key

    @field_validator("turn_key", mode="before")
    @classmethod
    def _turn_key_lenient(cls, value: Any) -> Optional[str]:
        return _turn_key_or_none(value)

    @field_validator("question", mode="after")
    @classmethod
    def _question_limit(cls, value: str) -> str:
        if len(value) > MAX_TEXT_CHARS:
            raise ValueError(f"question is longer than {MAX_TEXT_CHARS} characters")
        return value

    @field_validator("results", mode="after")
    @classmethod
    def _results_limit(cls, value: list[ResultData]) -> list[ResultData]:
        if len(value) > MAX_RESULTS:
            raise ValueError(f"more than {MAX_RESULTS} results")
        return value


# ----------------------------------------------------------------------
# Permissive specs: synonyms are mapped, never rejected
# ----------------------------------------------------------------------


def _synonym(value: Any, table: dict[str, tuple[str, ...]], default: str) -> str:
    word = str(value).strip().lower() if value is not None else ""
    for canonical, synonyms in table.items():
        if word in synonyms:
            return canonical
    return default


_STAT_TONES = {
    "good": ("good", "ok", "success", "positive"),
    "warning": ("warning", "warn", "caution", "pending"),
    "bad": ("bad", "danger", "error", "negative", "failed"),
}
_BLOCK_KINDS = {
    "stats": ("stats", "stat", "kpi", "kpis", "metric", "metrics", "summary"),
    "table": ("table", "rows", "list", "grid"),
    "record": ("record", "kv", "object", "detail", "details"),
    "quote": ("quote", "callout", "note", "warning"),
}
_BLOCK_TONES = {
    "success": ("success", "good", "ok"),
    "warning": ("warning", "warn"),
    "danger": ("danger", "error", "bad"),
}
_FIELD_KINDS = {
    "choice": ("choice", "select", "dropdown", "radio", "enum", "single_select"),
    "multichoice": ("multichoice", "checkbox", "checkboxes", "multi", "multi_select", "multiselect"),
    "number": ("number", "int", "integer", "float", "decimal"),
    "longtext": ("longtext", "textarea", "multiline", "paragraph"),
    "boolean": ("boolean", "bool", "yes_no"),
    "date": ("date",),
}


def salvage_list(raw_items: Any, item_model: type[BaseModel]) -> list:
    """Each item validated on its own; only the items that fail are skipped."""
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []
    kept = []
    for item in raw_items:
        try:
            kept.append(item_model.model_validate(item))
        except ValidationError:
            continue
    return kept


class StatItem(BaseModel):
    label: str
    ref: str
    tone: str = "neutral"

    @field_validator("tone", mode="before")
    @classmethod
    def _tone(cls, value: Any) -> str:
        return _synonym(value, _STAT_TONES, "neutral")


class CardBlock(BaseModel):
    kind: str
    title: str = ""
    tone: str = "info"
    ref: str = ""
    fields: list[str] = []
    stats: list[StatItem] = []
    quote: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: Any) -> str:
        return _synonym(value, _BLOCK_KINDS, "unknown")

    @field_validator("tone", mode="before")
    @classmethod
    def _tone(cls, value: Any) -> str:
        return _synonym(value, _BLOCK_TONES, "info")

    @field_validator("title", "ref", "quote", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("fields", mode="before")
    @classmethod
    def _fields(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if isinstance(part, (str, int)) and str(part).strip()]
        return []

    @field_validator("stats", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> list[StatItem]:
        return salvage_list(value, StatItem)


class CardSection(BaseModel):
    heading: str = ""
    verdict: str = ""
    blocks: list[CardBlock] = []

    @field_validator("heading", "verdict", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("blocks", mode="before")
    @classmethod
    def _blocks(cls, value: Any) -> list[CardBlock]:
        return salvage_list(value, CardBlock)


def _is_stray(item: Any) -> bool:
    """An element that says nothing: ``None``, a blank string, or a list of only such elements (``[['']]``)."""
    if item is None or (isinstance(item, str) and not item.strip()):
        return True
    return isinstance(item, (list, tuple)) and all(_is_stray(part) for part in item)


def unnest_sections(value: list) -> tuple[list, int]:
    """``(items, lists_flattened)``: ``value`` with exactly one extra list level removed and stray empty elements
    ignored. Small models often write ``[[{section}, {section}], ...]``; each list among the items is spliced
    in place, the elements inside it are not flattened again, so an element that is still not a dict is
    reported by the caller as ``section_not_a_dict``. Only a list that held something counts as flattened."""
    items: list = []
    flattened = 0
    for item in value:
        if isinstance(item, (list, tuple)):
            if _is_stray(item):
                continue
            flattened += 1
            items.extend(part for part in item if not _is_stray(part))
        elif not _is_stray(item):
            items.append(item)
    return items, flattened


def _salvage_sections(value: Any) -> list[CardSection]:
    """Sections from what the model wrote. A bare block (it has a ``kind`` and no ``blocks``) becomes a section
    of its own, so a model that forgot the wrapper still shows its data. A sections value nested one list level
    too deep is flattened (``unnest_sections``)."""
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    value, _ = unnest_sections(value)
    wrapped = [
        {"blocks": [item]} if isinstance(item, dict) and "kind" in item and "blocks" not in item else item
        for item in value
    ]
    return salvage_list(wrapped, CardSection)


class AnswerCard(BaseModel):
    """What the model writes is ``lead`` and ``sections``. What the server returns (wire v2) is
    ``{v, lead, sections, blocks}``, where ``blocks`` repeats the sections' blocks in one flat list for a
    page that predates sections, and a card with no ``sections`` is the v1 card (``lead`` and ``blocks``)."""

    v: int = 2
    lead: str = ""
    sections: list[CardSection] = []
    blocks: list[CardBlock] = []

    @field_validator("lead", mode="before")
    @classmethod
    def _lead(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("sections", mode="before")
    @classmethod
    def _sections(cls, value: Any) -> list[CardSection]:
        return _salvage_sections(value)

    @field_validator("blocks", mode="before")
    @classmethod
    def _blocks(cls, value: Any) -> list[CardBlock]:
        return salvage_list(value, CardBlock)


class FormOption(BaseModel):
    value: str
    label: str = ""


def _coerce_option(item: Any) -> Optional[dict[str, str]]:
    if isinstance(item, FormOption):
        return item.model_dump()
    if isinstance(item, bool):
        return None
    if isinstance(item, (str, int, float)):
        text = str(item)
        return {"value": text, "label": text}
    if isinstance(item, dict):
        value = None
        for key in ("value", "id", "key", "option", "name"):
            if isinstance(item.get(key), (str, int, float)) and not isinstance(item.get(key), bool):
                value = str(item[key])
                break
        if value is None:
            return None
        label = item.get("label")
        return {"value": value, "label": str(label) if isinstance(label, (str, int, float)) else ""}
    return None


class FormField(BaseModel):
    name: str = ""
    label: str
    kind: str = "text"
    options: list[FormOption] = []
    required: bool = False
    help: str = ""

    @field_validator("name", "help", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: Any) -> str:
        return _synonym(value, _FIELD_KINDS, "text")

    @field_validator("options", mode="before")
    @classmethod
    def _options(cls, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        coerced = (_coerce_option(item) for item in value)
        return [option for option in coerced if option is not None]

    @field_validator("required", mode="before")
    @classmethod
    def _required(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value) if value is not None else False


class QuestionForm(BaseModel):
    fields: list[FormField] = []

    @field_validator("fields", mode="before")
    @classmethod
    def _fields(cls, value: Any) -> list[FormField]:
        return salvage_list(value, FormField)


# ----------------------------------------------------------------------
# References: r1.<key>
# ----------------------------------------------------------------------

_BRACKET_KEY = re.compile(r"\[\s*['\"]([^'\"]+)['\"]\s*\]")


def normalize_ref(ref: str, results: list[ResultData]) -> str:
    """The reference a small model meant, as ``r1.<key>...``, or the cleaned text when nothing fits."""
    text = str(ref).strip()
    text = text.removeprefix("$").strip()
    text = text.removeprefix("results.")
    text = _BRACKET_KEY.sub(r".\1", text).strip(".")
    segments = [part for part in text.split(".")]
    if not segments or not segments[0]:
        return text

    ids = {result.id for result in results}
    target: Optional[ResultData] = None
    if segments[0] in ids:
        target = next(result for result in results if result.id == segments[0])
    else:
        by_command = [
            result for result in results
            if segments[0] in (result.command, result.command.split("/")[-1]) and result.command
        ]
        if len(by_command) == 1:
            target = by_command[0]
            segments[0] = target.id
        elif len(results) == 1:
            only = results[0]
            if segments[0] in only.output:
                target = only
                segments.insert(0, only.id)
    if target is None:
        return ".".join(segments)

    # "r1.order_details.status" when `status` is directly on the unwrapped output
    if len(segments) >= 3 and segments[1] not in target.output and segments[2] in target.output:
        del segments[1]
    return ".".join(segments)


def resolve_ref(ref: str, results: list[ResultData]) -> tuple[str, Any]:
    """``(canonical_ref, value)`` for a reference into the results; ``KeyError`` when it points at nothing.

    Only dict keys are followed. A list index never resolves.
    """
    canonical = normalize_ref(ref, results)
    segments = canonical.split(".")
    result = next((r for r in results if r.id == segments[0]), None)
    if result is None:
        raise KeyError(ref)
    value: Any = result.output
    for segment in segments[1:]:
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        else:
            raise KeyError(ref)
    return canonical, value


# ----------------------------------------------------------------------
# What the model is shown
# ----------------------------------------------------------------------


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return f"list of {len(value)} records"
        return f"list of {len(value)}"
    return "record"


def _preview(value: Any) -> Any:
    if isinstance(value, str):
        return value[:60]
    if isinstance(value, (dict, list)):
        return _type_of(value)
    return value


def _sample(value: Any) -> Any:
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, list):
        return [_preview(item) for item in value[:5]]
    if isinstance(value, dict):
        return {str(key): _preview(item) for key, item in list(value.items())[:12]}
    return value


def _hint_words(hint: str) -> list[str]:
    return [word for word in str(hint).replace(" ", "").split(",") if word]


def _visible_keys(result: ResultData) -> list[str]:
    """The output keys the model may be shown: a key the schema marks ``hidden`` is left out entirely."""
    return [key for key in result.output if "hidden" not in _hint_words(result.hints.get(key, ""))]


def _cell(value: Any) -> Any:
    return value[:60] if isinstance(value, str) else value


def _page_summary(result: ResultData, keys: list[str]) -> Optional[dict[str, Any]]:
    """One line for a page (parallel lists and a ``total``): its column keys, "60 of 167" and 3 sample rows."""
    value = result.output
    if not _is_page_shape(value):
        return None
    columns = [key for key in _scalar_lists(value) if key in keys]
    if len(columns) < 2:
        return None
    rows, total = len(value[columns[0]]), value["total"]
    return {
        "columns": columns,
        "rows": f"{rows} of {total}" if total > rows else str(rows),
        "sample": [[_cell(value[column][i]) for column in columns] for i in range(min(3, rows))],
    }


def _discriminator(result: ResultData) -> str:
    """A short text that tells this result from a sibling of the same command: its context (entity) label and
    the values of its first few scalar inputs. A hidden field's key or value, an identifier-like token and a
    value that is not a short scalar are never used."""
    hidden_keys = {key for key, hint in result.hints.items() if "hidden" in _hint_words(hint)}
    hidden_values = _hidden_values([result])
    parts: list[str] = []
    context = _clean_text(result.context, 40)
    if context and context.lower() not in hidden_values:
        parts.append(context)
    for key, value in list(result.parameters.items())[:4]:
        if key in hidden_keys or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        text = _clean_text(str(value), 24)
        if not text or text.lower() in hidden_values or _is_long_identifier(text):
            continue
        parts.append(f"{key}={text}")
    return _clean_text("; ".join(parts), DISCRIMINATOR_MAX_CHARS)


MAX_SHOW_FIELDS = 8
RICH_MAX_MEMBERS = 8        # a command group of more results than this collapses straight to one line
_IDENTIFIER_KEY = re.compile(r"(^|_)(u?id|uuid|guid)s?$", re.IGNORECASE)


def _show_hint(result: ResultData, keys: list[str], page: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """How to show the list-like parts of a result, as ready blocks in the names of the output schema (``kind``,
    ``ref``, ``fields``): a page or a list of records as a ``table`` and lists of plain values as a ``record``. Each
    part is its own object, never one string such as ``table ref=r2 fields=[a,b]`` that a small model pastes whole
    into the ``ref`` field. ``[]`` when the result holds no list."""
    parts: list[dict[str, Any]] = []
    columns = _page_columns(result)
    if columns:
        parts.append({"kind": "table", "ref": result.id, "fields": columns[:MAX_SHOW_FIELDS]})
    plain: list[str] = []
    for key in keys:
        value = result.output[key]
        if key in columns or not isinstance(value, list) or not value:
            continue
        if _is_row_list(value):
            parts.append({"kind": "table", "ref": f"{result.id}.{key}", "fields": _table_keys(value)[:MAX_SHOW_FIELDS]})
        elif all(_is_scalar(item) for item in value):
            plain.append(key)
    if plain:
        parts.append({"kind": "record", "ref": result.id, "fields": plain[:MAX_SHOW_FIELDS]})
    return parts[:3]


def _add_table_view(item: dict[str, Any], result: ResultData, keys: list[str], page: Optional[dict[str, Any]]) -> None:
    """The page of a result as flat keys (``rows``, ``first_rows``) and a ``show`` list, not as a nested
    ``page.sample`` record whose path a small model copies into a ref."""
    show = _show_hint(result, keys, page)
    if show:
        item["show"] = show
    if page:
        item["rows"] = page["rows"]
        item["first_rows"] = page["sample"]


def _headline_values(result: ResultData, keys: list[str], limit: int = 2) -> dict[str, Any]:
    """The one or two most informative scalar values of a result: a ``hero`` field first, then numbers, then short
    text; never an identifier, a hidden value or a long text."""
    hidden = _hidden_values([result])
    ranked: list[tuple[int, int, int, str, Any]] = []
    for position, key in enumerate(keys):
        value = result.output[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or _IDENTIFIER_KEY.search(key):
            continue
        if isinstance(value, str):
            value = _clean_text(value, 60)
            if not value or value.lower() in hidden or _is_long_identifier(value):
                continue
        hero = "hero" in _hint_words(result.hints.get(key, ""))
        ranked.append((0 if hero else 1, 0 if isinstance(value, (int, float)) else 1, position, key, value))
    return {key: value for *_, key, value in sorted(ranked, key=lambda row: row[:3])[:limit]}


def _rich_line(result: ResultData) -> dict[str, Any]:
    """A result shrunk to one line that still says what it is: its label, how to show it, the first row of its
    page and its one or two most informative scalar values."""
    keys = _visible_keys(result)
    line: dict[str, Any] = {"id": result.id, "command": result.command}
    label = _discriminator(result) or _clean_text(result.context, DISCRIMINATOR_MAX_CHARS)
    if label:
        line["label"] = label
    page = _page_summary(result, keys)
    if page:
        page = {**page, "sample": page["sample"][:1]}
    _add_table_view(line, result, keys, page)
    values = _headline_values(result, keys)
    if values:
        line["values"] = values
    line["keys"] = [key[:40] for key in keys[:12]]
    return line


def _digest_result(result: ResultData, max_fields: int) -> dict[str, Any]:
    keys = _visible_keys(result)
    page = _page_summary(result, keys)
    columns = set(page["columns"]) if page else set()
    shown = [key for key in keys if key not in columns][:max_fields]
    fields = [
        {"key": key, "title": result.titles.get(key, key), "type": _type_of(result.output[key]),
         "sample": _sample(result.output[key])}
        for key in shown
    ]
    item: dict[str, Any] = {
        "id": result.id,
        "command": result.command,
        "context": result.context,
        "parameters": dict(list(result.parameters.items())[:6]),
    }
    label = _discriminator(result)
    if label and label != result.context:
        item["label"] = label           # what tells it from a sibling result of the same command
    if result.supporting:
        item["supporting"] = True
    described = {
        key: result.descriptions[key].strip()[:DESCRIPTION_MAX_CHARS]
        for key in [*shown, *columns] if result.descriptions.get(key, "").strip()
    }
    hints = {key: result.hints[key] for key in [*shown, *columns] if result.hints.get(key, "").strip()}
    if described:
        item["descriptions"] = described
    if hints:
        item["hints"] = hints
    item["output"] = fields
    _add_table_view(item, result, keys, page)
    return item


def _dump(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"), default=str)


def _rows_of(result: ResultData) -> int:
    page = _page_summary(result, _visible_keys(result))
    if page:
        return len(result.output[next(iter(_scalar_lists(result.output)))])
    lengths = [len(value) for value in result.output.values() if isinstance(value, list)]
    return max(lengths) if lengths else 1


def _collapsed(results: list[ResultData]) -> dict[str, Any]:
    """The one line a command group shrinks to when it is over its share of the digest."""
    keys: list[str] = []
    for result in results:
        for key in _visible_keys(result):
            if key not in keys:
                keys.append(key)
    line: dict[str, Any] = {"id": results[0].id, "command": results[0].command}
    if len(results) > 1:
        line["ids"] = [result.id for result in results]
        labels = [_discriminator(result)[:30] for result in results[:8]]
        if any(labels):
            line["labels"] = labels
    elif label := _discriminator(results[0]):
        line["label"] = label
    line["keys"] = [key[:40] for key in keys[:12]]
    line["rows"] = sum(_rows_of(result) for result in results)
    return line


def _fair_level(sizes: list[int], budget: int) -> int:
    """The largest size a group may keep when the budget is shared fairly: the groups smaller than their
    share keep all of it, and the rest split what is left."""
    remaining = budget
    ordered = sorted(sizes)
    for index, size in enumerate(ordered):
        share = remaining // (len(ordered) - index)
        if size > share:
            return share
        remaining -= size
    return budget


ALSO_LINE_CHARS = 380


def _also_line(groups: list[list[ResultData]]) -> str:
    text = "also: " + ", ".join(f"{group[0].command} x{len(group)}" for group in groups)
    if len(text) <= ALSO_LINE_CHARS:
        return text
    cut = text[:ALSO_LINE_CHARS - 30].rsplit(", ", 1)[0]
    return f"{cut}, and more"


def _rungs(members: list[ResultData], supporting: bool) -> list[list[dict[str, Any]]]:
    """The forms one command group may take, each strictly smaller than the one before: every result in full;
    for a single result, fewer fields; for a primary group of up to ``RICH_MAX_MEMBERS``, one rich line per result
    (``_rich_line``); at last one line for the whole group (``_collapsed``)."""
    rungs = [[_digest_result(result, len(result.output)) for result in members]]
    candidates: list[list[dict[str, Any]]] = []
    if len(members) == 1:
        candidates += [[_digest_result(members[0], max_fields)] for max_fields in (20, 10, 5, 2)]
    if not supporting and len(members) <= RICH_MAX_MEMBERS:
        candidates.append([_rich_line(result) for result in members])
    candidates.append([_collapsed(members)])
    for items in candidates:
        if len(_dump(items)) < len(_dump(rungs[-1])):
            rungs.append(items)
    return rungs


def results_digest(results: list[ResultData]) -> str:
    """A deterministic JSON summary of the results, at most ``DIGEST_MAX_CHARS`` characters.

    Primary results come before supporting ones. Every command group starts in full (fields, schema descriptions
    and hints, a ``show`` line for what is a list, a page as ``rows`` and ``first_rows``). While the whole is over
    the budget, one group steps down to a smaller form (``_rungs``): supporting groups first, and among the groups
    of a kind the biggest first, so a PRIMARY result is the last to lose detail and when it does it keeps a line
    with its label, how to show it and its main values. Groups past the budget are never dropped silently: they
    become ONE closing line, ``"also: <command> xN, ..."``. A ref to anything named here, or in a collapsed group,
    still resolves: the refs are checked against the whole data, not the digest.

    The recorded request part (``ResultData.sub_task``) is not in the digest: no part lines, no part marks, no
    ordering by part (measured worse, blind, 2026-11-13).
    """
    ordered = sorted(results, key=lambda result: result.supporting)
    groups: dict[tuple[str, bool], list[ResultData]] = {}
    for result in ordered:
        groups.setdefault((result.command, result.supporting), []).append(result)
    rungs = {key: _rungs(members, key[1]) for key, members in groups.items()}
    level = {key: 0 for key in groups}

    def lines() -> list[Any]:
        return [item for key in groups for item in rungs[key][level[key]]]

    def size(key: tuple[str, bool]) -> int:
        return len(_dump(rungs[key][level[key]]))

    while len(_dump(lines())) > DIGEST_MAX_CHARS:
        open_groups = [key for key in groups if level[key] < len(rungs[key]) - 1]
        if not open_groups:
            break
        biggest = max([key for key in open_groups if key[1]] or open_groups, key=size)
        level[biggest] += 1
    diag = DIAGNOSTICS.get()
    if diag is not None:
        diag.collapsed += sum(1 for key in groups if level[key] > 0)
    if len(_dump(lines())) <= DIGEST_MAX_CHARS:
        return _dump(lines())
    kept: list[Any] = []
    cut = len(groups)
    for index, key in enumerate(groups):
        items = rungs[key][level[key]]
        if len(_dump([*kept, *items])) > DIGEST_MAX_CHARS - ALSO_LINE_CHARS - 2:
            cut = index
            break
        kept.extend(items)
    leftover = list(groups.values())[cut:]
    if diag is not None:
        diag.collapsed += len(leftover)
    return _dump([*kept, _also_line(leftover)])


# ----------------------------------------------------------------------
# Diagnostics: what the validator kept and dropped (server log only, never on the wire)
# ----------------------------------------------------------------------

DEBUG_RAW_CHARS = 4000
DEBUG_DETAIL_CHARS = 80


class ComposeDiagnostics:
    """What one answer-card composition proposed, kept and dropped, by stable reason code.

    ``sections_unnested`` counts the lists a model nested one level too deep that were flattened: a repair, not a
    drop, so it never triggers the correction pass.

    Counts and codes only: no data values and no user text, so the endpoint may log it at INFO. The offending
    ref or text of each drop goes to the DEBUG log alone (``note_drop``). Reason codes: ``sections_not_a_list``,
    ``section_not_a_dict``, ``section_unparseable``, ``block_not_a_dict``, ``block_unparseable``,
    ``section_cap``, ``empty_section``, ``verdict_ungrounded``, ``lead_ungrounded``, ``block_cap``,
    ``bad_kind``, ``ref_unresolved``, ``not_a_table``, ``not_a_record``, ``duplicate_table``,
    ``duplicate_stat``, ``stats_empty``, ``stat_ref_unresolved``, ``stat_not_scalar``, ``stat_cap``,
    ``quote_ungrounded``, ``quote_identifier``, ``stat_label_unsupported``, ``title_unsupported``,
    ``verdict_number_unanchored``, ``lead_number_unanchored``, ``no_content``,
    ``sibling_mismatch``, ``example_copied``, ``lead_only_with_results``, ``lead_restates``, ``model_failed``,
    and the sentence and title checks ``completeness_unsupported`` (an "all shown" claim over a table the page caps,
    or over data that says more exist), ``dangling_reference`` ("listed below" where no block shows it),
    ``internal_wording`` (the composer's own vocabulary in the text) and ``judgement_unsupported`` (an evaluative
    adjective beside a risk, level or score that the answer and the data never use).

    Not drops: ``sections_unnested`` (above), ``refs_repaired`` (a ref the model wrote as an index or digest path
    that code mapped to the data it meant, or a table block over one flat record shown as its record block;
    ``note_repair``), ``sections_merged`` (a section about a result an
    earlier section already shows, whose blocks moved into that section) and ``truncated`` (the provider cut the
    answer at its token limit; the second call gets a larger one when the time allows).
    """

    def __init__(self) -> None:
        self.results = 0
        self.digest_chars = 0
        self.collapsed = 0
        self.raw_sections_chars = 0
        self.sections_proposed = 0
        self.sections_kept = 0
        self.blocks_proposed = 0
        self.blocks_kept = 0
        self.sections_unnested = 0      # lists flattened out of the sections value (a repair, not a drop)
        self.lead = "none"              # "kept" | "dropped" | "none" (the model wrote none)
        self.drops: dict[str, int] = {}
        self.details: list[tuple[str, str, str]] = []   # (reason, detail, ref): first DROP_DETAILS_MAX, never logged at INFO
        self.retry = "none"             # "none" | "used" | "kept_first" | "failed" | "skipped_slow"
        self.second: Optional["ComposeDiagnostics"] = None      # the correction pass's own tally
        self.sole_sibling_drops = 0     # sibling_mismatch drops that emptied a section of its only block
        self.truncated = False          # the provider cut the answer at its token limit (or the JSON is cut)
        self.token_limit = 0            # the completion limit this call was sent with
        self.refs_repaired = 0          # refs the code corrected without a drop (``note_repair``)
        self.repairs: list[tuple[str, str]] = []    # (ref as written, ref meant): first DROP_DETAILS_MAX, DEBUG only
        self.sections_merged = 0        # sections whose blocks went into an earlier section about the same result
        self.answer_chars = 0           # the full answer's length (every check reads all of it)
        self.excerpt_chars = 0          # the length of the excerpt the model was shown instead; 0 when it saw the whole answer
        self.parts = 0                  # parts of the request, when the results carry part numbers (``request_parts``); else 0
        self.parts_uncovered = 0        # parts with a primary result that no kept section binds a block to
        self.uncovered: list[tuple[int, str, list[str]]] = []   # (part, title, its result ids): diagnostics only
        self.cross_part_blocks = 0      # blocks bound to another part than the rest of their section (counted, never dropped)

    def add(self, reason: str, detail: Any = "", ref: str = "") -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1
        if len(self.details) < DROP_DETAILS_MAX:
            self.details.append((reason, str(detail or "")[:DEBUG_DETAIL_CHARS], ref))

    def correctable(self) -> int:
        """The drops that trigger the correction pass (``CORRECTABLE_DROPS``, and a sibling mismatch that took the
        only block of its section)."""
        return (sum(count for reason, count in self.drops.items() if reason in CORRECTABLE_DROPS)
                + self.sole_sibling_drops)

    def drops_text(self) -> str:
        return ",".join(f"{reason}:{count}" for reason, count in sorted(self.drops.items())) or "none"

    def summary(self) -> str:
        """The key=value fields of the INFO line; ints, short codes and nothing else."""
        return (
            f"digest_chars={self.digest_chars} collapsed={self.collapsed} "
            f"sections={self.sections_kept}/{self.sections_proposed} blocks={self.blocks_kept}/{self.blocks_proposed} "
            f"lead={self.lead} drops={self.drops_text()} retry={self.retry}"
            + (f" parts={self.parts} parts_uncovered={self.parts_uncovered}" if self.parts else "")
            + (f" cross_part={self.cross_part_blocks}" if self.cross_part_blocks else "")
            + (f" unnested={self.sections_unnested}" if self.sections_unnested else "")
            + (f" repaired={self.refs_repaired}" if self.refs_repaired else "")
            + (f" merged={self.sections_merged}" if self.sections_merged else "")
            + (" truncated=1" if self.truncated else "")
            + (f" answer_chars={self.answer_chars} excerpt_chars={self.excerpt_chars}" if self.excerpt_chars else "")
            + ("" if self.second is None else (
                f" drops2={self.second.drops_text()} sections2={self.second.sections_kept}/{self.second.sections_proposed}"
                f" blocks2={self.second.blocks_kept}/{self.second.blocks_proposed}"
                + (f" parts_uncovered2={self.second.parts_uncovered}" if self.parts else "")))
        )


DIAGNOSTICS: "contextvars.ContextVar[Optional[ComposeDiagnostics]]" = contextvars.ContextVar(
    "compose_diagnostics", default=None)


def note_drop(reason: str, detail: Any = "", ref: str = "") -> None:
    """Record one drop in the current composition's diagnostics (no-op without one); the offending ref or the
    first 80 characters of the offending text are logged at DEBUG only."""
    diag = DIAGNOSTICS.get()
    if diag is not None:
        diag.add(reason, detail, ref)
    if detail not in ("", None):
        logger.debug(f"compose answer_card drop: reason={reason} detail={str(detail)[:DEBUG_DETAIL_CHARS]!r}")
    else:
        logger.debug(f"compose answer_card drop: reason={reason}")


def note_repair(written: str, meant: str) -> None:
    """Record one ref the code corrected (counted, never a drop and never a reason for a second call)."""
    diag = DIAGNOSTICS.get()
    if diag is not None:
        diag.refs_repaired += 1
        if len(diag.repairs) < DROP_DETAILS_MAX:
            diag.repairs.append((str(written)[:DEBUG_DETAIL_CHARS], str(meant)[:DEBUG_DETAIL_CHARS]))
    logger.debug(f"compose answer_card repair: {str(written)[:DEBUG_DETAIL_CHARS]!r} -> {str(meant)[:DEBUG_DETAIL_CHARS]!r}")


def _raw_chars(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(json.dumps(value, default=str, ensure_ascii=False))


def _raw_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text[:DEBUG_RAW_CHARS]


def _note_raw_sections(raw: Any, parsed: list["CardSection"], diag: ComposeDiagnostics) -> None:
    """Count what the model proposed, and record the items pydantic dropped before the validator saw them."""
    items = [raw] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        diag.sections_proposed = 0
        if raw not in (None, "", []):
            note_drop("sections_not_a_list", type(raw).__name__)
        return
    items, flattened = unnest_sections(items)
    diag.sections_unnested += flattened
    diag.sections_proposed = len(items)
    for item in items:
        if not isinstance(item, dict):
            note_drop("section_not_a_dict", item)
            continue
        blocks = item.get("blocks") if "blocks" in item or "kind" not in item else [item]
        if isinstance(blocks, dict):
            blocks = [blocks]
        diag.blocks_proposed += len(blocks) if isinstance(blocks, list) else 0
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    note_drop("block_not_a_dict", block)
    if len(parsed) < sum(isinstance(item, dict) for item in items):
        for _ in range(sum(isinstance(item, dict) for item in items) - len(parsed)):
            note_drop("section_unparseable")
    parsed_blocks = sum(len(section.blocks) for section in parsed)
    dict_blocks = diag.blocks_proposed - diag.drops.get("block_not_a_dict", 0)
    for _ in range(max(0, dict_blocks - parsed_blocks)):
        note_drop("block_unparseable")


# ----------------------------------------------------------------------
# Answer card verification
# ----------------------------------------------------------------------

_MARKER = re.compile(r"\[\[\s*##.*?##\s*\]\]", re.DOTALL)
_ID_TOKEN = re.compile(r"[#]?[A-Za-z_]+\d[\w-]*|\b\d+[A-Za-z_][\w-]*")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _clean_text(text: str, limit: int) -> str:
    """``text`` without DSPy markers and extra whitespace, cut to ``limit`` characters at a word."""
    text = " ".join(_MARKER.sub(" ", text or "").split())
    if len(text) > limit:
        cut = text[:limit]
        if text[limit] != " " and " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.rstrip()
    return text


def _verified_text(text: str, limit: int, ground: "_Ground") -> str:
    """The cleaned text, or ``""`` when it holds an ID or number that is not in the turn's data."""
    text = _clean_text(text, limit)
    return text if text and _text_is_grounded(text, ground) else ""


def clean_and_verify_lead(lead: str, request: AnswerCardRequest) -> str:
    """The lead without DSPy markers, cut to 300 characters at a word, or ``""`` when it is not grounded."""
    return _verified_text(lead, LEAD_MAX_CHARS, _ground_sets(request))


def _walk(value: Any, texts: list[str], numbers: set[float]) -> None:
    """Collect every dict key and scalar as text, and every list and dict length as a number."""
    if isinstance(value, dict):
        numbers.add(float(len(value)))
        for key, item in value.items():
            texts.append(str(key))
            _walk(item, texts, numbers)
    elif isinstance(value, list):
        numbers.add(float(len(value)))
        for item in value:
            _walk(item, texts, numbers)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        numbers.add(float(value))
        texts.append(str(value))
    else:
        texts.append(str(value))


def _bare_id(token: str) -> str:
    return token.lower().lstrip("#")


_Ground = tuple[set[str], list[Decimal]]


def _ground_sets(request: AnswerCardRequest) -> _Ground:
    """The bare IDs and the numbers (as Decimals) the turn's answer, message and results contain."""
    texts = [request.answer, request.user_message]
    numbers: set[float] = {float(i) for i in range(1, len(request.results) + 1)}
    for result in request.results:
        _walk(result.output, texts, numbers)
    ids: set[str] = set()
    decimals = {Decimal(str(number)) for number in numbers if math.isfinite(number)}
    for text in texts:
        ids.update(_bare_id(token) for token in _ID_TOKEN.findall(text))
        decimals.update(Decimal(token.replace(",", "")) for token in _NUMBER.findall(text))
        for tokens in _clauses(text):       # "four" is 4
            decimals.update(_quantities(tokens).values())
    return ids, list(decimals)


def _decimal_places(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _number_is_grounded(token: str, ground_numbers: list[Decimal]) -> bool:
    """Equal to a grounded number, or that number rounded to the places the token is written with.

    Rounding applies only to a token with decimals, and only from a grounded number that has more places
    (186.3249 shown as 186.32). 186.33 is not 186.32, and 3 is not 3.4.
    """
    value = Decimal(token.replace(",", ""))
    places = _decimal_places(value)
    for ground in ground_numbers:
        if ground == value:
            return True
        if places and _decimal_places(ground) > places:
            step = Decimal(1).scaleb(-places)
            try:
                rounded = (ground.quantize(step, ROUND_HALF_UP), ground.quantize(step, ROUND_HALF_EVEN))
            except InvalidOperation:        # too many digits to round: not grounded
                continue
            if value in rounded:
                return True
    return False


def _text_is_grounded(text: str, ground: _Ground) -> bool:
    ground_ids, ground_numbers = ground
    if any(_bare_id(token) not in ground_ids for token in _ID_TOKEN.findall(text)):
        return False
    return all(_number_is_grounded(token, ground_numbers) for token in _NUMBER.findall(text))


# A lead or a verdict may state a quantity only when the data supports it. The decision, per quantity:
#   (a) the number is stated in the answer or the user's message (digits or the number word: "four" is 4) and
#       the answer does not give a DIFFERENT number to the same content word. "167 people" is kept when the
#       answer says "167 holders"; "32 findings" is dropped when the answer says "4 findings" or "32 rights and
#       4 findings", and "4 rights and 32 findings" is dropped when the nouns are swapped. A word belongs to the
#       nearest number (see ``_owned_words``: within ANCHOR_WINDOW words, a tie goes to the number before the
#       word unless a link word (of, is ...) precedes the one after, a number that owns none keeps its whole
#       window, clauses end at . ; | , and line breaks), on both sides.
#   (b) otherwise the number is a count field's value owned by a word of that field, or the length of a list or
#       page in a result, or the number of names the message or the answer enumerates ("Ann Lee, Bo Chan and Cy Diaz"),
#       when the sentence names a word of that field or result, or one said before the enumeration in its sentence
#       (the command name does not count: it names the whole result);
#   (c) otherwise it is unanchored and the sentence is dropped.
# A number that is part of a name ("Administrator 1": a capitalised or identifier word, then the number, copied
# verbatim from the answer, message or data) is not a quantity; nor is an ID ("A1", "#32", "3rd"), which the ID
# grounding checks; nor a hyphenated label ("level-4", "high-risk"). A date, time or range ("2026-11-12",
# "10:30") is allowed only when copied whole from those texts, else it is unanchored. The number words one to
# twenty and the tens are quantities (the loose "one" of "one of", "no one", "the one" is not).
ANCHOR_WINDOW = 3
_WORD_TOKEN = re.compile(r"[#\w][\w#-]*(?:[.,:/]\d+)*")
_HYPHEN_LABEL = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)+")
_NUMBER_WORDS = {
    word: number for number, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({word: 10 * tens for tens, word in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)})
_COMPOUND_NUMBER = re.compile(
    r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine)")
_MAGNITUDES = frozenset({"hundred", "thousand", "million", "billion"})
_ONE_NOT_BEFORE = frozenset({"no", "any", "every", "each", "the", "this", "that", "which", "another"})
# What the data itself says about a number: ``(list lengths, count values)``, each ``{number: [content words]}``.
# A length is the number of items of a list or page in a result (a page size: "60
# shown"); a count is an integer field's value ("167" in ``total``), anchoring only the word it is written next
# to. The words are those of the field's key, schema title and description (a length: those of every key of the
# result, a page's columns and its total name one thing).
_DataNumbers = tuple[dict[Decimal, list[set[str]]], dict[Decimal, list[set[str]]], dict[Decimal, list[set[str]]]]
# ``(owned words per number, every text joined, the data numbers, every word of the clause of each occurrence)``
# plus the words of the noun phrase right after each occurrence (what the answer calls that number)
_Anchors = tuple[dict[Decimal, list[set[str]]], str, _DataNumbers, dict[Decimal, list[set[str]]], dict[Decimal, list[set[str]]]]
# A state word attached to a counted noun ("five OPEN findings") changes what the number counts, so the number is
# kept only when the answer or the data pairs THAT number with the same state word. Generic and small on purpose.
_STATE_WORDS = frozenset(
    "open closed active inactive pending failed new critical high overdue unresolved resolved expired".split())


def _word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN.findall(text or "")


_CLAUSE_BREAK = re.compile(r"(?<=[.!?])\s+|[;|\n]|,\s+")


_LIST_NUMBER = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|(?:%s)(?:-\w+)?)" % "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_NUMBER_LIST_COMMA = re.compile(rf"(?<![\w.,])({_LIST_NUMBER}),\s+(?={_LIST_NUMBER}\b)", re.IGNORECASE)


def _clauses(text: str, number_lists: bool = False) -> list[list[str]]:
    """The word tokens of each clause (a sentence, a comma part, a ``|`` cell or a line) of ``text``. With
    ``number_lists``, a comma between two bare numbers joins them (``4, 9 and 5 holders`` is one clause)."""
    if number_lists:
        text = _NUMBER_LIST_COMMA.sub(r"\1 and ", text or "")
    return [_word_tokens(part) for part in _CLAUSE_BREAK.split(text or "")]


def _loose_words(text: str) -> set[str]:
    """Like ``_content_words`` but keeping generic words (total, items ...)."""
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    words = {_stem(word) for word in re.sub(r"[^A-Za-z0-9]+", " ", spaced).lower().split()}
    return {word for word in words if len(word) > 1 and not word.isdigit() and word not in _STOP_WORDS}


_COORDINATORS = frozenset({"and", "or"})
_LINKS = frozenset({"of", "is", "are", "was", "were"})     # "tax of 98.75", "findings is 4": the word goes with the number after


def _quantities(tokens: list[str]) -> dict[int, Decimal]:
    """``{index: value}`` of the quantities of a clause: numbers written in digits and number words."""
    found: dict[int, Decimal] = {}
    for index, token in enumerate(tokens):
        if _NUMBER.fullmatch(token):
            found[index] = Decimal(token.replace(",", ""))
            continue
        lower = token.lower()
        compound = _COMPOUND_NUMBER.fullmatch(lower)
        if compound:                        # thirty-two
            found[index] = Decimal(_NUMBER_WORDS[compound.group(1)] + _NUMBER_WORDS[compound.group(2)])
            continue
        if lower not in _NUMBER_WORDS:
            continue
        around = {tokens[i].lower() for i in (index - 1, index + 1) if 0 <= i < len(tokens)}
        if around & _MAGNITUDES:            # "one thousand fifteen": a written large number cannot be checked
            continue
        if lower == "one":
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            before = tokens[index - 1].lower() if index else ""
            if not _content_words(following) or before in _ONE_NOT_BEFORE:
                continue
        found[index] = Decimal(_NUMBER_WORDS[lower])
    return found


def _tie_rank(tokens: list[str], number: int, word: int) -> int:
    if number < word:
        return 1
    return 0 if tokens[number - 1].lower() in _LINKS else 2


def _owned_words(tokens: list[str], found: Optional[dict[int, Decimal]] = None, coordinated: bool = False) -> dict[int, set[str]]:
    """For each quantity in a clause, the words it owns, generic ones (total, items) included. A word within
    ``ANCHOR_WINDOW`` tokens of several quantities belongs to the nearest one; on a tie it belongs to the one
    before it (``4 control findings and 32 rights``: findings is 4's, rights is 32's). With ``coordinated``, numbers
    joined by ``and`` or ``or`` before one noun (``477 and 513 holders``) each own that noun: used for the sentence
    being checked only, never for the answer's own numbers (``Total 186.32 and 4 findings`` is no pair of counts)."""
    values = _quantities(tokens) if found is None else found
    quantities = list(values)
    owned: dict[int, set[str]] = {i: set() for i in quantities}
    for index, token in enumerate(tokens):
        if index in owned:
            continue
        near = [i for i in quantities if abs(i - index) <= ANCHOR_WINDOW]
        if near:
            owned[min(near, key=lambda i: (abs(i - index), _tie_rank(tokens, i, index)))
                  ] |= _loose_words(token)
    for i in quantities:
        if not owned[i]:                    # a number that was nobody's nearest (tax of 98.75): its whole window
            for index in range(max(0, i - ANCHOR_WINDOW), min(len(tokens), i + ANCHOR_WINDOW + 1)):
                owned[i] |= set() if index in owned else _loose_words(tokens[index])
    shared_by: dict[int, set[str]] = {}
    for here, after in reversed(list(zip(quantities, quantities[1:]))) if coordinated else []:
        if after == here + 2 and tokens[here + 1].lower() in _COORDINATORS:
            # "477 and 513 holders": numbers coordinated before one noun each count that noun
            shared = shared_by.get(after) or _phrase_words(tokens, after, values)
            if shared:
                owned[here] = shared_by[here] = set(shared)
    return owned


def _shares_owned_word(model_words: set[str], source_words: set[str]) -> bool:
    """A content word both own; when the model's number owns only generic words (``total $186.32``), a
    generic word both own."""
    specific = {word for word in model_words if word not in _GENERIC_WORDS}
    return bool((specific or model_words) & source_words)


def _adjacent_words(tokens: list[str], index: int, found: dict[int, Decimal]) -> set[str]:
    """The content words of the token right after the quantity at ``index`` (else right before it): what the
    number counts. "107 holders hidden" counts holders; "32 findings were proposed" counts findings."""
    for position in (index + 1, index - 1):
        if 0 <= position < len(tokens) and position not in found:
            words = _content_words(tokens[position])
            if words:
                return words
    return set()


def _phrase_words(tokens: list[str], index: int, found: dict[int, Decimal]) -> set[str]:
    """The loose words of the noun phrase that follows the quantity at ``index`` (up to two tokens, stopping at a
    stop word or another quantity): "32 rights through 7 groups" gives ``{right}`` for 32. When nothing follows,
    the one token before it."""
    words: set[str] = set()
    for position in (index + 1, index + 2):
        if position >= len(tokens) or position in found or tokens[position].lower() in _STOP_WORDS:
            break
        words |= _loose_words(tokens[position])
    if not words and index and index - 1 not in found:
        words = _loose_words(tokens[index - 1])
    return words


_NAME = r"[A-Z][\w'-]*(?:[ \t]+[A-Z][\w'-]*)*"
_ENUMERATION = re.compile(rf"{_NAME}(?:(?:,[ \t]+|,?[ \t]+(?:and|or)[ \t]+){_NAME})+")
_ENUMERATION_SEPARATOR = re.compile(rf"(?:,[ \t]+|,?[ \t]+(?:and|or)[ \t]+)(?={_NAME})")


def _enumerations(text: str) -> list[tuple[int, set[str]]]:
    """``(length, head words)`` of each enumeration of capitalised names ("Ann Lee, Bo Chan and Cy Diaz") in
    ``text``. The head words are the content words said before the list in its sentence ("five people on the exception
    list need the same treatment: ..."): what the list is a list of. A count of a request's own enumerated items is
    a number the request states without ever writing the digit."""
    found: list[tuple[int, set[str]]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        for match in _ENUMERATION.finditer(sentence):
            head = {word for word in _content_words(sentence[:match.start()]) if word not in _NUMBER_WORDS}
            if head:
                found.append((len(_ENUMERATION_SEPARATOR.findall(match.group(0))) + 1, head))
    return found


def _data_numbers(request: AnswerCardRequest) -> _DataNumbers:
    lengths: dict[Decimal, list[set[str]]] = {}
    counts: dict[Decimal, list[set[str]]] = {}
    named: dict[Decimal, list[set[str]]] = {}      # a count's value with only the words of its key and title
    for text in (request.user_message, request.answer):
        for length, head in _enumerations(text):
            lengths.setdefault(Decimal(length), []).append(head)
    for result in request.results:
        every_key = _key_words(result, list(result.output))     # not the command: it names the whole result
        for key, value in result.output.items():
            if isinstance(value, list) and value:
                lengths.setdefault(Decimal(len(value)), []).append(every_key)
            elif isinstance(value, int) and not isinstance(value, bool):
                counts.setdefault(Decimal(value), []).append(_key_words(result, [key]))
                named.setdefault(Decimal(value), []).append(_names_of(result, [key]))
    return lengths, counts, named


def _anchor_sets(request: AnswerCardRequest) -> _Anchors:
    """The numbers the answer and the message state, each with the content words around every occurrence, and
    every text of the turn (answer, message, data keys and values) joined for verbatim phrase checks."""
    anchored: dict[Decimal, list[set[str]]] = {}
    clauses: dict[Decimal, list[set[str]]] = {}
    phrases: dict[Decimal, list[set[str]]] = {}
    for text in (request.answer, request.user_message):
        for tokens in _clauses(text):
            found = _quantities(tokens)
            whole = set().union(*(_loose_words(token) for token in tokens)) if tokens else set()
            for index, words in _owned_words(tokens, found).items():
                anchored.setdefault(found[index], []).append(words)
                clauses.setdefault(found[index], []).append(whole)
                phrases.setdefault(found[index], []).append(_phrase_words(tokens, index, found))
    texts = [request.answer, request.user_message]
    for result in request.results:
        _walk(result.output, texts, set())
    return anchored, "\x00".join(" ".join(str(text).split()) for text in texts), _data_numbers(request), clauses, phrases


def _occurs_verbatim(phrase: str, corpus: str) -> bool:
    return re.search(rf"(?<![\w#]){re.escape(phrase)}(?![\w]|[.,:/]\d)", corpus, re.IGNORECASE) is not None


def _is_name_prefix(word: str) -> bool:
    return bool(word) and (word[0].isupper() or any(c.isdigit() or c == "_" for c in word)) \
        and bool(_content_words(word))


def _unanchored_sentence(text: str, anchors: _Anchors) -> str:
    """The first sentence of ``text`` holding a quantity that neither the answer, the message nor the data
    supports (the decision order above), or ``""``."""
    anchored, corpus, (lengths, counts, named), clauses, _ = anchors
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_words = _content_words(sentence)
        for tokens in _clauses(sentence, number_lists=True):
            found = _quantities(tokens)
            owned = _owned_words(tokens, found, coordinated=True)
            for index, token in enumerate(tokens):
                if token.startswith("#") or _ID_TOKEN.fullmatch(token):
                    continue                # an ID: the ID grounding decides
                if index not in found:
                    if any(c.isdigit() for c in token) and not _occurs_verbatim(token, corpus) \
                            and not _HYPHEN_LABEL.fullmatch(token):     # "level-4" is a label, not a quantity
                        return sentence     # a date, time or range that is not copied whole
                    continue
                if index and _is_name_prefix(tokens[index - 1]) \
                        and _occurs_verbatim(f"{tokens[index - 1]} {token}", corpus):
                    continue                # part of a name
                number = found[index]
                states = owned[index] & _STATE_WORDS
                if states and not any(states <= words for words in [*clauses.get(number, []), *named.get(number, [])]):
                    return sentence         # "five open findings": nothing pairs this number with "open"
                if any(_shares_owned_word(owned[index], words) for words in anchored.get(number, [])):
                    continue                # the answer puts the number next to the same word
                if number in anchored and not any(
                        other != number and any(_shares_owned_word(owned[index], words) for words in many)
                        for other, many in anchored.items()):
                    continue                # (a) stated, and no other number owns this noun
                # (b) the data: a count field's value, owned by a word of that field; the length of a list or
                # page (a page size), when the sentence as a whole names what the list holds
                adjacent = _adjacent_words(tokens, index, found)
                if any(adjacent & words for words in counts.get(number, [])):
                    continue
                if any(sentence_words & words for words in lengths.get(number, [])):
                    continue
                return sentence
    return ""


def lead_is_grounded(lead: str, request: AnswerCardRequest) -> bool:
    """Whether every ID-with-digits and every number in ``lead`` is in the turn's answer, message or results."""
    return _text_is_grounded(lead, _ground_sets(request))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, _SCALARS)


def _scalar_lists(value: dict[str, Any]) -> dict[str, list]:
    return {
        key: item for key, item in value.items()
        if isinstance(item, list) and all(_is_scalar(entry) for entry in item)
    }


def _is_page_shape(value: Any) -> bool:
    """A page as the server defines it (``observation_offloading.facts.output_page``: an int ``total`` and
    every list field of one length), which also holds at least two non-empty scalar lists, the columns of a
    combined table. The client mirrors this in ``tmCardIsPage``; ``tests/fixtures/page_shape_cases.json``
    pins both. Two unrelated lists of one length, without a ``total``, are not a page."""
    if not isinstance(value, dict) or output_page(value) is None:
        return False
    lists = _scalar_lists(value)
    return len(lists) >= 2 and all(len(item) >= 1 for item in lists.values())


def _table_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return list(_scalar_lists(value))
    keys: list[str] = []
    for row in value:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _is_row_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, dict) for row in value)


def _existing(fields: list[str], available: list[str]) -> list[str]:
    return [name for name in fields if name in available]


def _clean_quote(text: str) -> str:
    return " ".join(re.sub(r"[*_`]", "", text).split()).lower()


_LOWER_RUN = re.compile(r"[a-z]{6,}")
_HEX_ID = re.compile(r"[0-9A-Fa-f][0-9A-Fa-f-]{15,}")
ID_LIKE_MIN_CHARS = 20
HIDDEN_VALUE_MIN_CHARS = 6


def _is_long_identifier(token: str) -> bool:
    """A token the ID grounding already recognises (``_ID_TOKEN``: letters with a digit) that is long and
    machine-made: a hex or UUID string, or a prefixed random key. A long name made of real words has a run of
    lower-case letters; an identifier does not."""
    if _HEX_ID.fullmatch(token) and any(c.isdigit() for c in token):
        return True
    return len(token) >= ID_LIKE_MIN_CHARS and not _LOWER_RUN.search(token)


def _hidden_values(results: list[ResultData]) -> set[str]:
    """The text of every scalar (and scalar list item) under a key the schema marks ``hidden``."""
    values: set[str] = set()
    for result in results:
        for key, value in result.output.items():
            if "hidden" not in _hint_words(result.hints.get(key, "")):
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, (str, int)) and not isinstance(item, bool):
                    text = str(item).strip()
                    if len(text) >= HIDDEN_VALUE_MIN_CHARS:
                        values.add(text.lower())
    return values


def _quote_carries_identifier(quote: str, results: list[ResultData]) -> bool:
    """Whether a quote consists of or contains an identifier: the value of a field the schema hides, or a long
    machine-made token (``_is_long_identifier``). The page hides such fields; a verbatim quote must not bring
    them back."""
    lowered = quote.lower()
    if any(value in lowered for value in _hidden_values(results)):
        return True
    return any(_is_long_identifier(token) for token in _ID_TOKEN.findall(quote))


def _quote_is_grounded(quote: str, answer: str) -> bool:
    cleaned, whole = _clean_quote(quote), _clean_quote(answer)
    return 20 <= len(cleaned) <= 600 and len(cleaned) <= 0.8 * len(whole) and cleaned in whole


_STOP_WORDS = frozenset(
    "a an the of for and or in on to by with per is are was from at as this that it its".split())
_GENERIC_WORDS = frozenset(
    "total count number num all list items item results result value values data info detail details "
    "summary overview amount sum records record rows row entries entry".split())
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _stem(word: str) -> str:
    """Light stemming: ``findings`` and ``finding`` meet, ``entries`` and ``entry`` meet."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _content_words(text: Any) -> set[str]:
    """The lower-case stemmed words of ``text`` (snake_case, camelCase and spaces split) that say something:
    no stop words, generic words (total, count, list, items ...) or bare numbers."""
    spaced = _CAMEL_BOUNDARY.sub(" ", str(text or ""))
    words = {_stem(word) for word in re.sub(r"[^A-Za-z0-9]+", " ", spaced).lower().split()}
    return {word for word in words if len(word) > 1 and not word.isdigit()
            and word not in _STOP_WORDS and _stem(word) not in _GENERIC_WORDS and word not in _GENERIC_WORDS}


def _result_for(ref: str, results: list[ResultData]) -> Optional[ResultData]:
    return next((r for r in results if r.id == str(ref).split(".")[0]), None)


def _result_words(result: Optional[ResultData]) -> set[str]:
    """Words of the command name and of the result's context name."""
    return set() if result is None else _content_words(result.command) | _content_words(result.context)


def _key_words(result: Optional[ResultData], keys: list[str]) -> set[str]:
    """Words of each key, of its schema title and of its schema description."""
    words: set[str] = set()
    for key in keys:
        words |= _content_words(key)
        if result is not None:
            words |= _content_words(result.titles.get(key, "")) | _content_words(result.descriptions.get(key, ""))
    return words


def _pretty_key(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in _CAMEL_BOUNDARY.sub(" ", text).replace("_", " ").split())


def _derived_stat_label(canonical: str, result: Optional[ResultData]) -> str:
    """The code-written label of a stat: the field's schema title when it names something specific, else
    ``<Command noun phrase>: <Field>`` (``List Orders: Total``); without a command, the title or the key."""
    key = canonical.split(".")[-1]
    title = (result.titles.get(key, "") if result is not None else "").strip()
    if title and _content_words(title):
        return _clean_text(title, 60)
    if result is None or not result.command:
        return _clean_text(title or key, 60)    # nothing to qualify it with: the key, as before
    return _clean_text(f"{_pretty_command(result.command)}: {_pretty_key(title or key)}", 60)


# A qualifier changes what the word after it counts: "holders not shown" is not "holders shown". A label or
# title that shares a word with the data must carry the same qualifiers on that word as the data does.
_QUALIFIERS = {
    "not": "not", "no": "not", "non": "not", "without": "not", "n't": "not",
    "excluding": "excl", "excluded": "excl", "exclude": "excl",
    "missing": "missing", "remaining": "remaining", "other": "other", "hidden": "hidden",
}
_PURE_NEGATORS = frozenset({"not", "no", "non", "without"})
_QUALIFIER_WORDS = frozenset(_QUALIFIERS) - {"n't"}


def _qualified_words(text: Any) -> dict[str, set[str]]:
    """Each stemmed word of ``text`` with the qualifiers (``_QUALIFIERS``) attached to it: a qualifier attaches
    to the next word that is neither a stop word nor a qualifier, at most two tokens on."""
    spaced = _CAMEL_BOUNDARY.sub(" ", str(text or "").replace("n't", " not"))
    tokens = re.sub(r"[^A-Za-z0-9]+", " ", spaced).lower().split()
    out: dict[str, set[str]] = {}
    for index, token in enumerate(tokens):
        word = _stem(token)
        out.setdefault(word, set())
        if token not in _QUALIFIERS:
            continue
        for follower in tokens[index + 1:index + 3]:
            if follower in _STOP_WORDS or follower in _QUALIFIERS:
                continue
            out.setdefault(_stem(follower), set()).add(_QUALIFIERS[token])
            break
    return out


def _field_texts(result: Optional[ResultData], keys: list[str]) -> list[str]:
    texts: list[str] = []
    for key in keys:
        texts.append(key)
        if result is not None:
            texts += [result.titles.get(key, ""), result.descriptions.get(key, "")]
    return texts


def _qualifiers_agree(text: str, shared: set[str], result: Optional[ResultData], keys: list[str]) -> bool:
    """Whether ``text`` and the fields' key, title and description agree on the qualifiers of every word they
    share (``shared``, bare negators aside): each word is qualified by both or by neither, so ``Shown`` over
    ``Holders not shown`` fails and ``Holders`` over ``Holders not shown`` (a plain shortening) passes."""
    theirs: dict[str, set[str]] = {}
    for field_text in _field_texts(result, keys):
        for word, qualifiers in _qualified_words(field_text).items():
            theirs.setdefault(word, set()).update(qualifiers)
    ours = _qualified_words(text)
    words = {word for word in shared if word not in _PURE_NEGATORS} or shared
    return all(ours.get(word, set()) == theirs.get(word, set()) for word in words)


def _answer_words_for(value: Any, anchors: Optional[_Anchors]) -> set[str]:
    """The words of the noun phrase the answer or the user's message put right after this number ("32 rights");
    nothing for a value that is not a number, so a label can lean on the answer only for the value it names."""
    if anchors is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return set()
    words: set[str] = set()
    for owned in anchors[4].get(Decimal(str(value)), []):
        words |= owned
    return words


def _value_of(ref: str, results: list[ResultData]) -> Any:
    try:
        return resolve_ref(ref, results)[1]
    except KeyError:
        return None


def _number_of(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return Decimal(str(value))


def _words_beside(value: Any, anchors: Optional[_Anchors]) -> set[str]:
    """Every word the answer or the message owns together with this number (the words next to it)."""
    number = _number_of(value)
    if anchors is None or number is None:
        return set()
    words: set[str] = set()
    for owned in [*anchors[0].get(number, []), *anchors[4].get(number, [])]:
        words |= owned
    return words


def _borrowed_words(label_words: set[str], support: set[str], value: Any, anchors: Optional[_Anchors]) -> set[str]:
    """The words of a stat label that the field does not carry, that the answer does not put beside this value, and
    that it puts beside ANOTHER number: ``intrinsic score`` over the field that holds the aggregate score. The
    value of a stat always comes from its ref; what such a label gets wrong is which number it names."""
    number = _number_of(value)
    if anchors is None or number is None:
        return set()
    others: set[str] = set()
    for other, owned_sets in anchors[0].items():
        if other != number:
            for owned in owned_sets:
                others |= owned
    return (label_words - support - _words_beside(value, anchors)) & others


def _supported_stat_label(
        raw: str, canonical: str, results: list[ResultData], ground: _Ground,
        replaced: Optional[set[str]] = None, anchors: Optional[_Anchors] = None) -> str:
    """The model's label when its data supports it, else a code-written one. Supported: grounded, and it
    shares a content word with the field's key, schema title or description, the command name or the
    result's context name, and does not reverse or drop a qualifier the field puts on that word (``Shown``
    over a field titled ``Holders not shown``, ``_qualifiers_agree``). A label that is empty or generic only
    (``Total``) is replaced without a note. ``replaced`` collects the ref of every label that code wrote."""
    result = _result_for(canonical, results)
    label = _verified_text(raw, 40, ground)
    words = _content_words(label)
    keys = canonical.split(".")[1:]
    if not label or not words:
        return _derived_stat_label(canonical, result)
    support = _key_words(result, keys) | _result_words(result)
    value = _value_of(canonical, results)
    states = words & _STATE_WORDS
    # A state word (open, closed, failed ...) changes what the number counts: the field's own names, or the answer's
    # words beside that same value, must carry it; otherwise ``Open Issues`` over a total of all issues is false.
    states_ok = not states or states <= support or states <= _words_beside(value, anchors)
    if words & support and states_ok and _qualifiers_agree(label, words & support, result, keys) \
            and not _borrowed_words(words, support, value, anchors):
        return label
    # The answer's own words for this same value: "32 rights" supports the label "Rights" on the field that holds 32,
    # when every state or negation word of the label is there too ("Open ..." needs "open" beside that number). When
    # the answer also calls that value by a word of the field's own names (32 rights AND 32 groups), the value is a
    # coincidence of two things: the label is not taken from the answer, the field's own title is shown.
    spoken = _answer_words_for(value, anchors)
    if words & spoken and not spoken & support and ((_STATE_WORDS | _QUALIFIER_WORDS) & set(_words(label))) <= spoken:
        return label
    note_drop("stat_label_unsupported", label, canonical)
    if replaced is not None:
        replaced.add(canonical)
    return _derived_stat_label(canonical, result)


def _title_supported(
        block: CardBlock, request: AnswerCardRequest, strict: bool = False, anchors: Optional[_Anchors] = None) -> bool:
    """Whether a table, record or stats title does not MISLEAD about the data it heads.

    A title that shares a content word with the data passes: the command name and context of the referenced
    result(s), and the keys, schema titles and descriptions of the fields shown. A title of only generic words
    (``Items``) names nothing wrong and passes. A title that shares no word passes too, unless it is *competing*:
    it names another field, result or command of the turn (``Refunds`` over the orders). A descriptive grouping
    phrase that names none of them (``Reach``) misleads nobody.

    The strict test (every content word supported, qualifiers agreeing) applies to a stats block whose label code
    had to write (``strict``: a model that got the label wrong cannot be trusted with the title over the same
    value), and to a title that holds a digit or a qualifier (``not``, ``other``, ``hidden`` ...): a number or a
    negation that the data does not carry is a claim."""
    words = _content_words(block.title)
    if not words:
        return True
    if block.kind == "stats":
        refs = [stat.ref for stat in block.stats]
    else:
        refs = [block.ref]
    claims = any(c.isdigit() for c in block.title) or bool(_QUALIFIER_WORDS & set(_words(block.title)))
    support: set[str] = set()
    spoken: set[str] = set()                # the answer's own words for the numbers a stats block shows
    shown: dict[str, set[str]] = {}         # result id -> the keys of it this block shows
    for ref in refs:
        if block.kind == "stats":
            spoken |= _answer_words_for(_value_of(ref, request.results), anchors)
        result = _result_for(ref, request.results)
        keys = ref.split(".")[1:]
        if block.kind != "stats":
            keys = [*keys, *block.fields]
            if not block.fields:
                try:
                    value = resolve_ref(ref, request.results)[1]
                    keys += _table_keys(value) if block.kind == "table" else list(value)
                except (KeyError, TypeError, AttributeError):
                    pass
        ref_support = _key_words(result, keys) | _result_words(result)
        support |= ref_support
        if result is not None:
            shown.setdefault(result.id, set()).update(keys)
        if (strict or claims) and words & ref_support \
                and not _qualifiers_agree(block.title, words & ref_support, result, keys):
            return False
    support |= spoken
    states = words & _STATE_WORDS
    if states and not states <= support and not (block.kind != "stats" and _states_in_rows(states, refs, request)):
        return False                        # "Open issues" over data that never says open
    if strict:
        return words <= support
    if words & support:
        return True
    if claims:
        return False
    return not _competes(words, request.results, shown)


def _states_in_rows(states: set[str], refs: list[str], request: AnswerCardRequest) -> bool:
    """Whether the data a table or record shows holds every state word somewhere in its values (a ``status`` column
    of ``overdue`` rows supports the title ``Overdue invoices``)."""
    texts: list[str] = []
    for ref in refs:
        try:
            _walk(resolve_ref(ref, request.results)[1], texts, set())
        except KeyError:
            continue
    return states <= _content_words(" ".join(texts))


def _names_of(result: ResultData, keys: list[str]) -> set[str]:
    """The words that NAME a result's fields: keys and schema titles (a description is prose, with every
    everyday word in it)."""
    words: set[str] = set()
    for key in keys:
        words |= _content_words(key) | _content_words(result.titles.get(key, ""))
    return words


def _competes(words: set[str], results: list[ResultData], shown: dict[str, set[str]]) -> bool:
    """Whether a title that names nothing the block shows names something else of the turn instead: more than
    half of its content words are the command, context, field keys or field titles of another result, or the
    name of a field of a shown result that the block does not show."""
    need = len(words) // 2 + 1
    for result in results:
        if result.id in shown:
            for key in result.output:
                if key not in shown[result.id] and len(words & _names_of(result, [key])) >= need:
                    return True
        elif len(words & (_result_words(result) | _names_of(result, list(result.output)))) >= need:
            return True
    return False


def _verified_stats(
        block: CardBlock, results: list[ResultData], ground: _Ground, replaced: Optional[set[str]] = None,
        anchors: Optional[_Anchors] = None,
) -> Optional[CardBlock]:
    stats = []
    for position, stat in enumerate(block.stats):
        try:
            canonical, value = resolve_ref(stat.ref, results)
        except KeyError:
            canonical, value = "", None
            counted = _count_ref(stat.ref, stat.label, results)
            if counted is None:
                note_drop("stat_ref_unresolved", stat.ref, stat.ref)
                continue
        else:
            counted = _count_ref(stat.ref, stat.label, results) if not _is_scalar(value) else None
        if counted is not None:
            note_repair(stat.ref, counted)
            canonical, value = resolve_ref(counted, results)
        if not _is_scalar(value):
            note_drop("stat_not_scalar", stat.ref, stat.ref)
            continue
        label = _supported_stat_label(stat.label, canonical, results, ground, replaced, anchors)
        stats.append(StatItem(label=label, ref=canonical, tone=stat.tone))
        if len(stats) == MAX_STATS:
            for _ in block.stats[position + 1:]:
                note_drop("stat_cap")
            break
    if not stats:
        note_drop("stats_empty")
        return None
    return block.model_copy(update={"stats": stats, "ref": "", "fields": [], "quote": ""})


def _verified_block(
        block: CardBlock, request: AnswerCardRequest, ground: _Ground, anchors: Optional[_Anchors] = None,
        wording: Optional["_Wording"] = None,
) -> Optional[CardBlock]:
    replaced: set[str] = set()              # stat refs whose label code wrote: their block's title is held to a stricter test
    verified = _verified_block_body(block, request, ground, replaced, anchors)
    if verified is None:
        return None
    title = _verified_text(block.title, 80, ground)
    if title and verified.kind != "quote" and not _title_supported(
            verified.model_copy(update={"title": title}), request, strict=bool(replaced), anchors=anchors):
        note_drop("title_unsupported", title)
        title = ""                          # the card needs no title: the page shows the data under the section
    if title and wording is not None and verified.kind != "quote":
        title = _checked_title(title, [verified], request, wording)
    return verified.model_copy(update={"title": title})


def _copied_from_example(ref: str, request: AnswerCardRequest) -> bool:
    """Whether a ref points at a result id of the worked example (``EXAMPLE_RESULT_IDS``) that this request does not
    have: the model copied the example's block instead of reading the results."""
    head = str(ref).strip().removeprefix("$").split(".")[0].split("[")[0]
    return head in EXAMPLE_RESULT_IDS and not any(r.id == head for r in request.results)


def _verified_block_body(
        block: CardBlock, request: AnswerCardRequest, ground: _Ground, replaced: Optional[set[str]] = None,
        anchors: Optional[_Anchors] = None,
) -> Optional[CardBlock]:
    copied = next((r for r in [*[stat.ref for stat in block.stats], block.ref] if _copied_from_example(r, request)), None)
    if copied is not None:
        note_drop("example_copied", copied, copied)
        return None
    if block.kind == "stats":
        return _verified_stats(block, request.results, ground, replaced, anchors)
    if block.kind == "quote":
        if not _quote_is_grounded(block.quote, request.answer):
            note_drop("quote_ungrounded", block.quote)
            return None
        if _quote_carries_identifier(block.quote, request.results):
            note_drop("quote_identifier", block.quote)
            return None
        return block.model_copy(update={"ref": "", "fields": [], "stats": []})
    if block.kind not in ("table", "record"):
        note_drop("bad_kind", block.kind)
        return None
    wanted = block.fields
    kind = block.kind
    try:
        canonical, value = resolve_ref(block.ref, request.results)
    except KeyError:
        fixed = _repair_unresolved(block.ref, request.results)
        if fixed is None:
            note_drop("ref_unresolved", block.ref, block.ref)
            return None
        kind, canonical, value, meant = fixed
        wanted = meant or block.fields
        note_repair(block.ref, canonical)
    else:
        if kind == "record":
            as_table = _record_as_table(canonical, value, block.fields, request.results)
            if as_table is not None:
                kind = "table"
                note_repair(block.ref, as_table[0])
                canonical, value, wanted = as_table
        elif kind == "table" and _is_flat_record(value):
            kind = "record"                 # the reverse repair: a table over one flat record is its record block
            note_repair(block.ref, canonical)
    if kind == "table":
        column = _page_column(canonical, value, request.results)
        if column is not None:
            canonical, value, wanted = column
        if not (_is_page_shape(value) or _is_row_list(value)):
            note_drop("not_a_table", canonical, canonical)
            return None
        available = _table_keys(value)
    else:
        if not isinstance(value, dict) or not value:
            note_drop("not_a_record", canonical, canonical)
            return None
        available = list(value)
    fields = _existing(wanted, available)
    if kind == "table" and len(fields) < 2:
        fields = []                         # one column is no table: show them all (the client does the same)
    return block.model_copy(update={"kind": kind, "ref": canonical, "fields": fields, "stats": [], "quote": ""})


def _page_column(canonical: str, value: Any, results: list[ResultData]) -> Optional[tuple[str, dict, list[str]]]:
    """``(page ref, page, [column])`` for a table ref that names one column of a page (``r6.labels``, a list of
    scalars whose parent is a page), else None. The model meant the table of the page: a column is not a
    table by itself, and the digest shows a page as one line of column keys, so this is the ref it forms."""
    segments = canonical.split(".")
    if len(segments) < 2 or not isinstance(value, list) or not all(_is_scalar(item) for item in value):
        return None
    try:
        parent_ref, parent = resolve_ref(".".join(segments[:-1]), results)
    except KeyError:
        return None
    if not _is_page_shape(parent) or segments[-1] not in _scalar_lists(parent):
        return None
    return parent_ref, parent, [segments[-1]]


_INDEX_SEGMENT = re.compile(r"\[\s*(\d+)\s*\]")
# The names the digest gave its own structure, which a small model sometimes copies into a ref
# (``r2.page.sample.0``): they hold no data, so a ref passing through them means the thing they describe.
_DIGEST_MARKERS = frozenset({"page", "sample", "samples", "first_rows", "rows", "columns", "show", "table"})
_COUNT_LABEL = re.compile(r"(?<![A-Za-z])(count|number|total|how many)(?![A-Za-z])", re.IGNORECASE)


def _follow_ref(ref: str, results: list[ResultData]) -> Optional[tuple[list[str], Any, bool, str]]:
    """``(real segments, value there, skipped, trailing key)`` for a ref that is a path of real keys with list
    indexes and digest names (``_DIGEST_MARKERS``) among them, or None when it names something else, or nothing.
    ``skipped`` says an index or digest name was passed over; the trailing key is the one row key of a list of
    records written after the list (``r1.loans.0.due_date``)."""
    canonical = normalize_ref(_INDEX_SEGMENT.sub(r".\1", str(ref)), results)
    segments = canonical.split(".")
    result = _result_for(segments[0], results) if segments[0] else None
    if result is None:
        return None
    real, value, skipped, trailing = [result.id], result.output, False, ""
    for position, segment in enumerate(segments[1:], start=1):
        if isinstance(value, dict) and segment in value:
            value = value[segment]
            real.append(segment)
        elif segment.isdigit() or (segment in _DIGEST_MARKERS and isinstance(value, (dict, list))):
            skipped = True
        elif _is_row_list(value) and segment in _table_keys(value) and position == len(segments) - 1:
            trailing = segment
        else:
            return None
    return real, value, skipped, trailing


def _page_columns(result: ResultData) -> list[str]:
    """The visible columns of a result that is a page (its hidden columns left out, one visible column is enough)."""
    if not _is_page_shape(result.output):
        return []
    visible = _visible_keys(result)
    return [key for key in _scalar_lists(result.output) if key in visible]


def _repair_unresolved(ref: str, results: list[ResultData]) -> Optional[tuple[str, str, Any, list[str]]]:
    """``(kind, canonical ref, value, wanted fields)`` of what an unresolved ref meant, when that is unambiguous,
    else None (the block is dropped as before).

    The model wrote an index (``r1.labels.0``, ``r1.orders[0].total``) or the digest's own nesting
    (``r2.page.sample.0``, ``r2.page.labels.0``, ``r2.sample``) into a ref. It meant the list or page it passes
    through: a list of records is the table of that list; a column of a page, or the page itself, is the table of
    the page (with that column, or its columns); anything else (a scalar, a record, a list that is no column of a
    page) is not guessed."""
    followed = _follow_ref(ref, results)
    if followed is None or not followed[2]:
        return None
    real, value, _, trailing = followed
    canonical = ".".join(real)
    result = _result_for(real[0], results)
    if _is_row_list(value):
        return "table", canonical, value, [trailing] if trailing else []
    if len(real) == 1:
        columns = _page_columns(result) if result is not None and _is_page_shape(value) else []
        return ("table", canonical, value, []) if columns else None
    if isinstance(value, list):
        column = _page_column(canonical, value, results)
        if column is not None:
            return "table", column[0], column[1], column[2]
    return None


def _record_as_table(canonical: str, value: Any, fields: list[str], results: list[ResultData]) -> Optional[tuple[str, Any, list[str]]]:
    """``(ref, value, fields)`` of the table a RECORD block meant, when it points at a list of records, at a column
    of a page, or at two or more columns of a page; None for a record (a single list field stays a record: a
    one-column table is no table)."""
    if _is_row_list(value):
        return canonical, value, fields
    if isinstance(value, list):
        return _page_column(canonical, value, results)
    if _is_page_shape(value) and len(fields) >= 2 and all(name in _scalar_lists(value) for name in fields):
        return canonical, value, fields
    return None


def _is_flat_record(value: Any) -> bool:
    """Whether ``value`` is one flat record: a non-empty dict whose every value is a scalar (no list, no nested
    dict), so a record block would show it as it is and a table block cannot."""
    return isinstance(value, dict) and bool(value) and all(_is_scalar(item) for item in value.values())


def _count_ref(ref: str, label: str, results: list[ResultData]) -> Optional[str]:
    """The ref of the count stat for a stats ref that points at a list (or at a path through one), when the label
    already says count, number or total and the list's own record has an integer ``total``; else None."""
    if not _COUNT_LABEL.search(label or ""):
        return None
    followed = _follow_ref(ref, results)
    if followed is None:
        return None
    real, value, _, _ = followed
    if not isinstance(value, list):
        if len(real) != 1 or not _is_page_shape(value):
            return None
        parent = real[:1]
    else:
        parent = real[:-1] or real[:1]
    try:
        _, holder = resolve_ref(".".join(parent), results)
    except KeyError:
        return None
    total = holder.get("total") if isinstance(holder, dict) else None
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return ".".join([*parent, "total"])


def _salvaged_stat(
        block: CardBlock, request: AnswerCardRequest, ground: _Ground, anchors: Optional[_Anchors] = None,
) -> Optional[StatItem]:
    """A record or table block whose ref resolves to one scalar, as the stat it should have been.

    The label is the block's title when it is grounded, else the ref's last key. Bound to the data, like
    every stat: the browser draws the value from the ref.
    """
    if block.kind not in ("table", "record"):
        return None
    try:
        canonical, value = resolve_ref(block.ref, request.results)
    except KeyError:
        return None
    if not _is_scalar(value):
        return None
    return StatItem(label=_supported_stat_label(block.title, canonical, request.results, ground, None, anchors), ref=canonical)


# ----------------------------------------------------------------------
# Repairs and truth checks on the card's own words (round 4): none decides layout or wording for the model.
# A sentence or title that says something the data does not is removed; nothing is ever written in its place.
# ----------------------------------------------------------------------

# The digest's ``show`` entry copied into a block's ref as one string: ``r14 fields=[labels]``,
# ``table ref=r14 fields=[a,b]``, ``record ref=r2 fields=[notes]``. Exactly that syntax is read back; a ref that
# does not parse cleanly (a second part after ``|``, a bracket left open, words after it) is left to be dropped.
_SHOW_TEXT = re.compile(
    r"^\s*(?:(?P<kind>table|record)\s+)?(?:(?P<keyword>ref)\s*=\s*)?(?P<ref>r\d+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?:\s+fields\s*=\s*\[(?P<fields>[^\[\]|]*)\])?\s*$")


def _unpasted_block(block: CardBlock) -> CardBlock:
    """``block`` with a pasted ``show`` hint in its ref read back into ref and fields (counted as ``ref_repaired``);
    the block's own fields win when it wrote some. Any other block as it is."""
    if block.kind not in ("table", "record"):
        return block
    parsed = _SHOW_TEXT.match(block.ref)
    if parsed is None or not (parsed["kind"] or parsed["keyword"] or parsed["fields"] is not None):
        return block
    fields = block.fields or [
        name.strip().strip("'\"") for name in (parsed["fields"] or "").split(",") if name.strip().strip("'\"")]
    note_repair(block.ref, parsed["ref"])
    return block.model_copy(update={"ref": parsed["ref"], "fields": fields})


_INTERNAL_WORDS = re.compile(
    r"\b(digests?|payloads?|schemas?|refs?|primary\s+results?|supporting\s+results?|"
    r"(?:stat|stats|table|record|quote)\s+blocks?|block\s+kinds?)\b", re.IGNORECASE)
_RESULT_ID_WORD = re.compile(r"\br\d{1,3}\b")
_JUDGEMENT = re.compile(
    r"\b(high|higher|highest|low|lower|lowest|critical|severe|extreme|serious|elevated|moderate|minimal|dangerous|alarming)"
    r"\s+(?:[A-Za-z-]+\s+){0,2}?(?:risk|level|score|severity|priority|rating|exposure|impact)s?\b", re.IGNORECASE)
_JUDGEMENT_ROOT = {"higher": "high", "highest": "high", "lower": "low", "lowest": "low"}
_COMPLETE_CLAIM = re.compile(
    r"\b(all|every|each|entire|whole|complete|completely|full|fully|everything|in\s+full)\b", re.IGNORECASE)
_SHOWS_ROWS = re.compile(
    r"\b(listed|shown|displayed|presented|included|enumerated|itemi[sz]ed|below|here|table|list|rows?|visible)\b", re.IGNORECASE)
_SAYS_PARTIAL = re.compile(
    r"\b(first|only|top|partial|partially|sample|page|remaining|rest|others?|more|out\s+of|at\s+most|up\s+to)\b", re.IGNORECASE)
# What a sentence points at that must then be on the card: "listed below", "as follows", "see the table".
_POINTS_AT_A_BLOCK = re.compile(
    r"\b(?:(?:listed|shown|displayed|presented|detailed|given|set\s+out|laid\s+out|outlined|provided|included)\s+"
    r"(?:below|above|underneath|beneath|here|next)|as\s+follows|(?:is|are|was|were|be|been|all|as|\d+)\s+(?:listed(?!\s+as\b)|shown|displayed|presented)\b"
    r"|(?:the|these|those)\s+\w+\s+listed\b|(?:see|check|refer\s+to)\s+the\s+(?:table|list|rows|details|figures|data|card|breakdown)\b"
    r"|(?:the|in\s+the)\s+(?:table|list)\s+(?:below|above))", re.IGNORECASE)
_NOT_SHOWN_KEY = re.compile(
    r"(not[_ ]?(shown|returned|listed|included|displayed)|remaining|hidden|omitted|unshown|truncated|overflow|beyond|more[_ ]?(results|rows|items|entries|records))",
    re.IGNORECASE)


class _Wording:
    """What the sentence checks read about a request: its texts (answer, message, data), every word the answer, the
    message, the data and the schema texts use, and its result ids."""

    def __init__(self, request: AnswerCardRequest, corpus: str) -> None:
        self.corpus = corpus
        self.ids = {result.id for result in request.results}
        texts = [corpus]
        for result in request.results:
            texts += [*result.titles.values(), *result.descriptions.values(), *result.hints.values(), result.command, result.context]
        self.vocab = set(re.findall(r"[a-z]+", " ".join(texts).lower()))


def _table_is_cut(block: CardBlock, request: AnswerCardRequest) -> Optional[bool]:
    """Whether a table block shows only part of its rows: more rows than the page shows (``TABLE_ROWS_SHOWN``), a
    ``total`` above the rows it holds, or a not-shown or remaining count above zero beside it. ``None`` for a block
    that is not a table (or one that does not resolve)."""
    if block.kind != "table":
        return None
    try:
        canonical, value = resolve_ref(block.ref, request.results)
    except KeyError:
        return None
    if isinstance(value, dict):
        columns = list(_scalar_lists(value).values())
        rows = len(columns[0]) if columns else 0
        containers = [value]
    elif isinstance(value, list):
        rows = len(value)
        parent = ".".join(canonical.split(".")[:-1])
        containers = [_value_of(parent, request.results)] if "." in parent else []
    else:
        return None
    result = _result_for(canonical, request.results)
    if result is not None:
        containers.append(result.output)
    if rows > TABLE_ROWS_SHOWN:
        return True
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key, number in container.items():
            if isinstance(number, bool) or not isinstance(number, int):
                continue
            if (key == "total" and number > rows) or (number > 0 and _NOT_SHOWN_KEY.search(key)):
                return True
    return False


def _table_words(block: CardBlock, request: AnswerCardRequest) -> set[str]:
    """The content words that name what a table block shows: its title, the command and context of its result and
    the keys, schema titles and descriptions of its fields."""
    try:
        canonical, value = resolve_ref(block.ref, request.results)
        keys = [*canonical.split(".")[1:], *(block.fields or _table_keys(value))]
    except (KeyError, TypeError, AttributeError):
        return _content_words(block.title)
    result = _result_for(canonical, request.results)
    return _content_words(block.title) | _key_words(result, keys) | _result_words(result)


def _claims_more_than_shown(blocks: list[CardBlock], request: AnswerCardRequest, sentence: str) -> bool:
    """Whether what ``sentence`` says is shown includes a table that is cut. The tables it is about are those that
    share a content word with it (all of them when none does); one cut table among them makes "all shown" untrue."""
    tables = [(block, cut) for block in blocks if (cut := _table_is_cut(block, request)) is not None]
    if not tables:
        return False
    words = _content_words(sentence)
    about = [cut for block, cut in tables if words & _table_words(block, request)]
    return any(about) if about else all(cut for _, cut in tables)


def _words_problem(sentence: str, wording: _Wording) -> str:
    """``internal_wording`` or ``judgement_unsupported`` when ``sentence`` uses the composer's own vocabulary
    (a word the answer, the message and the data never use) or an evaluative adjective beside risk, level or score
    that nothing in the turn uses; else ``""``."""
    for found in _INTERNAL_WORDS.finditer(sentence):
        if not _occurs_verbatim(found.group(0), wording.corpus):
            return "internal_wording"
    for found in _RESULT_ID_WORD.finditer(sentence):
        if found.group(0) in wording.ids and not _occurs_verbatim(found.group(0), wording.corpus):
            return "internal_wording"
    for found in _JUDGEMENT.finditer(sentence):
        word = found.group(1).lower()
        if _JUDGEMENT_ROOT.get(word, word) not in wording.vocab and word not in wording.vocab:
            return "judgement_unsupported"
    return ""


def _claims_problem(
        sentence: str, blocks: list[CardBlock], request: AnswerCardRequest, wording: _Wording, dangling: bool = True) -> str:
    """``dangling_reference`` when ``sentence`` points at content (``listed below``) that ``blocks`` do not hold, or
    ``completeness_unsupported`` when it claims everything is shown over tables that are all cut; else ``""``."""
    if dangling:
        if not blocks and _POINTS_AT_A_BLOCK.search(sentence):
            return "dangling_reference"
    if _COMPLETE_CLAIM.search(sentence) and _SHOWS_ROWS.search(sentence) and not _SAYS_PARTIAL.search(sentence) \
            and _claims_more_than_shown(blocks, request, sentence):
        return "completeness_unsupported"
    return ""


def _checked_sentences(
        text: str, blocks: list[CardBlock], request: AnswerCardRequest, wording: _Wording, checks: str,
        dangling: bool = True) -> str:
    """``text`` without the sentences the ``checks`` ("words" or "claims") reject, each noted as a drop; the others
    as written."""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        problem = _words_problem(sentence, wording) if checks == "words" \
            else _claims_problem(sentence, blocks, request, wording, dangling)
        if problem:
            note_drop(problem, sentence)
        else:
            kept.append(sentence)
    return " ".join(kept)


def _checked_title(title: str, blocks: list[CardBlock], request: AnswerCardRequest, wording: _Wording) -> str:
    """A block title or section heading, or ``""`` when it uses the composer's vocabulary, an unsupported judgement
    or claims completeness over a cut table (``blocks`` are the blocks it heads: one for a title, the section's for
    a heading)."""
    problem = (_words_problem(title, wording) or _claims_problem(title, blocks, request, wording, dangling=False)) if title else ""
    if problem:
        note_drop(problem, title)
        return ""
    return title


LONG_ANSWER_CHARS = 400         # the client's TM_CARD_LONG_ANSWER: below it a one-result turn composes nothing
LEAD_ONLY_COVERAGE = 0.6


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", _clean_quote(text)).split()


def _lead_adds_something(lead: str, answer: str) -> bool:
    """Whether a card of only a lead may replace the answer on screen (the lead then hides the full answer
    behind a toggle, so it must be a real summary).

    All of: the answer is longer than ``LONG_ANSWER_CHARS``; the lead is at most half as long as the
    answer; the lead's words (normalized: lower case, letters and digits) are not a prefix of the answer's
    words; and the lead's distinct words cover at most 60% of the distinct words of the answer's first
    paragraph (more means it restates the opening).
    """
    paragraphs = [part for part in re.split(r"\n\s*\n", answer) if _words(part)]
    lead_words, answer_words = _words(lead), _words(answer)
    if len(answer) <= LONG_ANSWER_CHARS or not paragraphs or not lead_words:
        return False
    if len(lead) * 2 > len(answer) or answer_words[:len(lead_words)] == lead_words:
        return False
    first = set(_words(paragraphs[0]))
    return len(first & set(lead_words)) <= LEAD_ONLY_COVERAGE * len(first)


def _verified_blocks(
        raw_blocks: list[CardBlock], request: AnswerCardRequest, ground: _Ground,
        tables: set[str], salvaged_refs: set[str], cap: int, anchors: Optional[_Anchors] = None,
        wording: Optional[_Wording] = None) -> list[CardBlock]:
    """The blocks that resolve against the data, at most ``cap``. ``tables`` and ``salvaged_refs`` belong to
    the whole card: one table per list, one stat per ref, however many sections there are."""
    blocks: list[CardBlock] = []
    salvaged_from: dict[int, str] = {}      # index of a merged stats block -> its result id
    for block in raw_blocks:
        block = _unpasted_block(block)
        stat = _salvaged_stat(block, request, ground, anchors)
        if stat is not None:
            if stat.ref in salvaged_refs:
                note_drop("duplicate_stat", stat.ref)
                continue
            salvaged_refs.add(stat.ref)
            result_id = stat.ref.split(".")[0]
            last = len(blocks) - 1
            if salvaged_from.get(last) == result_id and len(blocks[last].stats) < MAX_STATS:
                blocks[last] = blocks[last].model_copy(update={"stats": [*blocks[last].stats, stat]})
            elif len(blocks) < cap:
                salvaged_from[len(blocks)] = result_id
                blocks.append(CardBlock(kind="stats", stats=[stat]))
            else:
                note_drop("block_cap", stat.ref)
            continue
        verified = _verified_block(block, request, ground, anchors, wording)
        if verified is not None and verified.kind == "table":
            if verified.ref in tables:
                note_drop("duplicate_table", verified.ref)
                continue                    # one table per list
            tables.add(verified.ref)
        if verified is not None and len(blocks) < cap:
            blocks.append(verified)
        elif verified is not None:
            note_drop("block_cap", verified.ref)
    return blocks


def _pretty_command(command: str) -> str:
    """``Library/find_book`` as ``Find Book``: the leaf, words apart, each first letter up (the page's footer
    does the same, and also spells acronyms such as ID)."""
    return " ".join(word[:1].upper() + word[1:] for word in command.split("/")[-1].replace("_", " ").split())


SUB_TASK_HEADING = re.compile(r"^#{3}[ \t]+(\d+)\.[ \t]+(.+?)[ \t]*$", re.MULTILINE)
MAX_SUB_TASKS = MAX_SECTIONS

EXCERPT_CUT_MARKER = "[... the rest of this part is in the results]"
EXCERPT_GAP_MARKER = "[... part of the answer is left out here; it is in the results]"


def _cut_at_boundary(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` characters at the last line break, else the last sentence end, else the
    last space in its second half; a hard cut only for text with none of them."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    floor = limit // 2
    line = window.rfind("\n")
    if line >= floor:
        return window[:line].rstrip()
    sentence = max((m.end() for m in re.finditer(r"[.!?](?=\s)", window)), default=-1)
    if sentence >= floor:
        return window[:sentence].rstrip()
    space = window.rfind(" ")
    return (window[:space] if space >= floor else window).rstrip()


def _fair_shares(lengths: list[int], budget: int, floor: int) -> list[int]:
    """How many characters each part may keep: an equal share of ``budget``, at least ``floor``, and what a short
    part does not use goes to the longer ones (a part that fits keeps its whole length)."""
    shares = [0] * len(lengths)
    pending = sorted(range(len(lengths)), key=lambda index: lengths[index])
    left = budget
    for position, index in enumerate(pending):
        share = max(floor, left // (len(pending) - position))
        shares[index] = min(lengths[index], share)
        left -= shares[index]
    return shares


def _answered_parts(parts: Optional[list[PartData]]) -> list[PartData]:
    return [part for part in parts or [] if part.answered]


def _heading_spans(answer: str, parts: Optional[list[PartData]]) -> list[tuple[str, int, int]]:
    """``[(heading line, where it starts, where its part's answer starts)]`` for the answer's parts, at most
    ``MAX_SUB_TASKS``. With ``parts`` (data the core recorded, already checked against the answer): the answered parts
    when there are two or more (an answer of one part has no heading, whatever its text holds). Without: the
    ``### N.`` heading lines found in the text."""
    if parts is not None:
        answered = _answered_parts(parts)
        if len(answered) < 2:
            return []
        spans = []
        for part in answered[:MAX_SUB_TASKS]:
            heading = f"### {part.number}. {part.title}"
            spans.append((heading, part.start - len(heading) - 1, part.start))
        return spans
    spans = []
    for match in list(SUB_TASK_HEADING.finditer(answer))[:MAX_SUB_TASKS]:
        line_end = answer.find("\n", match.end())
        spans.append((match.group(0), match.start(), len(answer) if line_end < 0 else line_end + 1))
    return spans


def answer_for_prompt(answer: str, parts: Optional[list[PartData]] = None) -> str:
    """The text of the answer the MODEL is shown; every check reads the full ``request.answer`` instead.

    At or under ``ANSWER_PROMPT_WHOLE_CHARS`` the whole answer, unchanged. Above it an excerpt of about
    ``ANSWER_EXCERPT_CHARS``: the text before the first ``### N.`` heading (at most
    ``ANSWER_EXCERPT_PREAMBLE_CHARS``), then every heading in order, each with the opening of its part cut at a
    line or sentence end (a fair share of the budget, at least ``ANSWER_EXCERPT_MIN_SECTION_CHARS``; what a short
    part leaves unused goes to the long ones). A cut part ends with ``EXCERPT_CUT_MARKER``. An answer with no
    headings keeps its head and its tail with ``EXCERPT_GAP_MARKER`` between. The markers hold no digit and are in
    no answer, so a number or a quote taken from one fails the checks.

    ``parts`` (``AnswerCardRequest.parts``) says where the headings are; None reads them from the text."""
    if len(answer) <= ANSWER_PROMPT_WHOLE_CHARS:
        return answer
    headings = _heading_spans(answer, parts)
    if not headings:
        room = ANSWER_EXCERPT_CHARS - len(EXCERPT_GAP_MARKER) - 4
        head = _cut_at_boundary(answer, room * 2 // 3)
        tail_part = answer[len(answer) - room // 3:]
        newline = tail_part.find("\n")
        tail = tail_part[newline + 1:] if 0 <= newline < len(tail_part) // 2 else tail_part
        return f"{head}\n\n{EXCERPT_GAP_MARKER}\n\n{tail.lstrip()}"
    starts = [start for _, start, _ in headings] + [len(answer)]
    pieces: list[str] = []
    preamble = answer[:starts[0]].strip()
    if preamble:
        kept = _cut_at_boundary(preamble, ANSWER_EXCERPT_PREAMBLE_CHARS)
        pieces.append(kept if kept == preamble else f"{kept}\n{EXCERPT_CUT_MARKER}")
    kept_parts: list[tuple[str, str]] = []
    for index, (line, _, body_start) in enumerate(headings):
        kept_parts.append((line, answer[body_start:starts[index + 1]].strip()))
    used = sum(len(piece) + 2 for piece in pieces) + sum(len(head) + 2 + len(EXCERPT_CUT_MARKER) + 2 for head, _ in kept_parts)
    shares = _fair_shares([len(body) for _, body in kept_parts], max(0, ANSWER_EXCERPT_CHARS - used), ANSWER_EXCERPT_MIN_SECTION_CHARS)
    for (heading, body), share in zip(kept_parts, shares):
        kept = _cut_at_boundary(body, share) if share < len(body) else body
        text = heading + ("\n" + kept if kept else "")
        pieces.append(text if len(kept) == len(body) else f"{text}\n{EXCERPT_CUT_MARKER}")
    return "\n\n".join(pieces)


def sub_tasks(answer: str) -> list[str]:
    """``["1. find X", "2. ..."]``: the ``### N. <title>`` headings of a multi-sub-task answer, in order, each
    title cut to 80 characters; ``[]`` for an answer without them. Read by code, never by the model."""
    return [
        f"{number}. {_clean_text(title, 80)}" for number, title in SUB_TASK_HEADING.findall(answer or "")
    ][:MAX_SUB_TASKS]


def has_part_numbers(request: AnswerCardRequest) -> bool:
    """Whether any result carries the request part it ran for (``ResultData.sub_task``)."""
    return any(result.sub_task for result in request.results)


def _part_label(result: ResultData) -> str:
    """A part's title when its answer has no heading: the code-written label of one of its results, its command
    spelled out and the discriminator (the record it ran for), never a wording the model or the code invents."""
    name = _pretty_command(result.command) if result.command else ""
    label = _discriminator(result)
    return _clean_text(f"{name}: {label}" if name and label else (name or label), 80)


def part_titles(request: AnswerCardRequest) -> dict[int, str]:
    """``{part number: title}`` for the parts of the request, in number order: the union of the numbers of the
    answer's ``### N.`` headings and of the results' ``sub_task``. A part's title is its heading's when it has one,
    else the label of its first primary result (of its first result when it has only supporting ones)."""
    titles: dict[int, str] = {}
    if request.parts is not None:
        # The data lists every part, one that gave no answer (no heading) too: its title is its request's heading.
        for part in request.parts:
            if part.title:
                titles.setdefault(part.number, _clean_text(part.title, 80))
    else:
        for number, title in SUB_TASK_HEADING.findall(request.answer or ""):
            if 1 <= int(number) <= MAX_SUB_TASKS:
                titles.setdefault(int(number), _clean_text(title, 80))
    first: dict[int, ResultData] = {}
    for result in sorted(request.results, key=lambda r: r.supporting):
        if result.sub_task:
            first.setdefault(result.sub_task, result)
    for number, result in first.items():
        if not titles.get(number):
            titles[number] = _part_label(result)
    return dict(sorted(titles.items()))


def request_sub_tasks(request: AnswerCardRequest) -> list[str]:
    """The parts of the request as ``["1. title", ...]``, what the model is shown as ``sub_tasks`` and what
    ``section_cap`` counts: the answer's ``### N.`` headings alone, whether or not the results carry part numbers
    (the numbers are for the log diagnostics only; ``part_titles``). With ``request.parts`` the answered parts, when
    two or more answered (the answer has headings only then); the same strings, read from data."""
    if request.parts is not None:
        answered = _answered_parts(request.parts)
        if len(answered) < 2:
            return []
        return [f"{part.number}. {_clean_text(part.title, 80)}" for part in answered][:MAX_SUB_TASKS]
    return sub_tasks(request.answer)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", " ".join(str(text).split()), maxsplit=1)[0]


def _section_heading(
        raw: str, index: int, titles: list[str], blocks: list[CardBlock], request: AnswerCardRequest,
        ground: _Ground) -> str:
    """The model's heading when it is grounded; else the answer's own ``### N.`` title when the section's
    position (0-based, in the order the model wrote the sections) is a sub-task; else the prettified command
    of the first result the section's blocks point at; else ``""``."""
    heading = _verified_text(raw, HEADING_MAX_CHARS, ground)
    if heading:
        return heading
    if index < len(titles):
        return _clean_text(titles[index].split(". ", 1)[-1], HEADING_MAX_CHARS)
    for block in blocks:
        refs = [stat.ref for stat in block.stats] if block.kind == "stats" else [block.ref]
        for ref in refs:
            result = next((r for r in request.results if r.id == ref.split(".")[0]), None)
            if result is not None and result.command:
                return _clean_text(_pretty_command(result.command), HEADING_MAX_CHARS)
    return ""


def section_cap(request: AnswerCardRequest) -> int:
    """The most sections a card for ``request`` may have: one per part of the request (the ``### N.`` sub-task
    headings of its answer): at least ``MIN_SECTIONS``, with no other limit than the safety ceiling
    ``MAX_SECTIONS`` against runaway output. The parts are ``request_sub_tasks``: the headings alone."""
    return max(MIN_SECTIONS, min(MAX_SECTIONS, len(request_sub_tasks(request))))


def card_block_cap(cap: int) -> int:
    """The most blocks in all: ``MAX_CARD_BLOCKS`` up to ``MIN_SECTIONS`` sections, then 2 per section, never more
    than ``MAX_CARD_BLOCKS_CEILING`` (2 per section of the section ceiling)."""
    return min(MAX_CARD_BLOCKS_CEILING, max(MAX_CARD_BLOCKS, 2 * cap))


def card_token_budget(cap: int) -> int:
    """The output budget of a card that may have ``cap`` sections: ``CARD_MAX_TOKENS`` up to ``MIN_SECTIONS``
    sections (1800: what a 5-section card has always had), then ``CARD_TOKENS_PER_EXTRA_SECTION`` more for each
    section beyond that. The model's own output ceiling clamps it when it is sent (``effective_max_tokens``).
    9 sections 2800, 14 sections 4050, 24 sections 6550."""
    return CARD_MAX_TOKENS + CARD_TOKENS_PER_EXTRA_SECTION * max(0, cap - MIN_SECTIONS)


def _raw_result_ids(section: CardSection) -> set[str]:
    """The ids of the results the blocks of a not yet verified section point at."""
    ids: set[str] = set()
    for block in section.blocks:
        refs = [stat.ref for stat in block.stats] if block.kind == "stats" else [block.ref]
        ids.update(str(ref).split(".")[0] for ref in refs if ref)
    return ids


def _section_order(sections: list[CardSection], cap: int) -> list[int]:
    """The order to verify the sections in: as written when they fit the cap; with more than the cap, first those
    that show a result no earlier section shows (a section repeating the data of an earlier one goes last), each
    group in the order written. Verification stops at the cap, so the repeats are what the cap cuts."""
    if len(sections) <= cap:
        return list(range(len(sections)))
    fresh: list[int] = []
    seen: set[str] = set()
    for index, section in enumerate(sections):
        ids = _raw_result_ids(section)
        if ids - seen:
            fresh.append(index)
            seen |= ids
    return fresh + [index for index in range(len(sections)) if index not in fresh]


_SIBLING_WORDS = re.compile(r"[a-z0-9]+")


def _label_words(result: ResultData) -> set[str]:
    """The lower-case words of a result's context label (else of its short scalar inputs): what names the entity
    the result is about."""
    if result.context.strip():
        text = result.context
    else:
        text = " ".join(str(v) for v in result.parameters.values() if isinstance(v, (str, int)) and not isinstance(v, bool))
    return {word for word in _SIBLING_WORDS.findall(text.lower()) if word not in _STOP_WORDS}


def _block_result_ids(block: CardBlock) -> list[str]:
    refs = [stat.ref for stat in block.stats] if block.kind == "stats" else [block.ref]
    return [ref.split(".")[0] for ref in refs if ref]


def _sibling_mismatch(block: CardBlock, section_text: str, request: AnswerCardRequest) -> Optional[ResultData]:
    """The sibling the section is about when ``block`` shows a different result of the same command, else None.

    Fires only when every sibling of the block's result (same command) that the section names has a label of its
    own and exactly one of them is named in the section's heading, verdict or titles while the block's own
    result's label is named nowhere in it: two ``get_record`` results for two people, a section about one, a
    block bound to the other. A label is told from its siblings' by the words only it has."""
    text_words = set(_SIBLING_WORDS.findall(section_text.lower()))
    for result_id in _block_result_ids(block):
        bound = next((r for r in request.results if r.id == result_id), None)
        if bound is None or not bound.command:
            continue
        siblings = [r for r in request.results if r.command == bound.command and r.id != bound.id]
        own = _label_words(bound)
        if not siblings or not own:
            continue
        named: list[ResultData] = []
        for sibling in siblings:
            theirs = _label_words(sibling)
            distinct = theirs - own
            if theirs and own - theirs and distinct and distinct <= text_words:
                named.append(sibling)
        if len(named) == 1 and not ((own - _label_words(named[0])) & text_words):
            return named[0]
    return None


def _primary_id(blocks: list[CardBlock]) -> str:
    """The id of the result the first block of a section points at (``""`` for a quote)."""
    for block in blocks:
        ids = _block_result_ids(block)
        if ids:
            return ids[0]
    return ""


def _raw_primary_id(section: CardSection) -> str:
    for block in section.blocks:
        refs = [stat.ref for stat in block.stats] if block.kind == "stats" else [block.ref]
        if refs and refs[0]:
            return str(refs[0]).split(".")[0]
    return ""


def _merge_target(raw: CardSection, over_cap: bool, sections: list[CardSection]) -> int:
    """The index (in ``sections``) of the earlier kept section a raw section's blocks go into instead of being
    dropped or standing alone, or ``-1``. Only a section about the SAME result: the same first result id as the
    earlier one (or the same heading and no other result than the earlier section shows). It qualifies when it is
    over the section cap, or when it holds nothing but tables and records and says the same thing about that result
    (its heading shares a content word with the earlier heading): two sections about
    one result under different headings are two parts of the request and stay two."""
    if not raw.blocks or not (over_cap or all(block.kind in ("table", "record") for block in raw.blocks)):
        return -1
    wanted = _raw_primary_id(raw)
    if not wanted:
        return -1
    heading = _loose_words(raw.heading)
    ids = _raw_result_ids(raw)
    for index, earlier in enumerate(sections):
        shown = {result_id for block in earlier.blocks for result_id in _block_result_ids(block)}
        same_result = _primary_id(earlier.blocks) == wanted
        if same_result and not over_cap:
            theirs = _content_words(earlier.heading)
            same_result = bool(_content_words(raw.heading) & theirs)
        if same_result or (heading and heading == _loose_words(earlier.heading) and ids <= shown):
            return index
    return -1


def _without_sibling_mismatches(blocks: list[CardBlock], section_text: str, request: AnswerCardRequest) -> list[CardBlock]:
    kept: list[CardBlock] = []
    for block in blocks:
        other = _sibling_mismatch(block, section_text, request)
        if other is None:
            kept.append(block)
            continue
        note_drop("sibling_mismatch", other.context or other.id, _block_result_ids(block)[0])
        if len(blocks) == 1 and (diag := DIAGNOSTICS.get()) is not None:
            diag.sole_sibling_drops += 1
    return kept


def _verified_sections(
        card: AnswerCard, request: AnswerCardRequest, ground: _Ground, anchors: _Anchors,
        wording: Optional[_Wording] = None) -> list[CardSection]:
    """At most ``section_cap(request)`` sections of at most ``MAX_SECTION_BLOCKS`` verified blocks each,
    ``card_block_cap`` in all. A verdict that holds a number or ID not in the data, or a quantity the data does
    not support, is dropped; a block bound to a sibling result the section does not name is dropped; a section
    with no verdict and no block is dropped.

    A section past the cap, or one that holds only tables and records, whose first result is the one an earlier
    kept section is about, is not dropped: its blocks move into that section (``_merge_target``), up to the
    section's block cap, the rest dropped as ``block_cap``."""
    titles = request_sub_tasks(request)
    cap = section_cap(request)
    block_cap = card_block_cap(cap)
    tables: set[str] = set()
    salvaged_refs: set[str] = set()
    sections: list[CardSection] = []
    total = 0
    order = _section_order(card.sections, cap)
    kept_at: list[int] = []
    diag = DIAGNOSTICS.get()
    for index in order:
        raw = card.sections[index]
        over_cap = len(sections) >= cap
        target = _merge_target(raw, over_cap, sections)
        if over_cap and target < 0:
            note_drop("section_cap")
            continue
        room = min(MAX_SECTION_BLOCKS, block_cap - total)
        blocks = _verified_blocks(raw.blocks, request, ground, tables, salvaged_refs, room, anchors, wording) if room > 0 else []
        written = _first_sentence(raw.verdict)
        if wording is not None:
            written = _checked_sentences(written, [], request, wording, "words")
        verdict = _verified_text(written, VERDICT_MAX_CHARS, ground)
        if written.strip() and not verdict:
            note_drop("verdict_ungrounded", raw.verdict)
        elif verdict and (loose := _unanchored_sentence(verdict, anchors)):
            note_drop("verdict_number_unanchored", loose)
            verdict = ""
        if target >= 0 and blocks:
            into = sections[target]
            space = max(0, MAX_SECTION_BLOCKS - len(into.blocks))
            for _ in blocks[space:]:
                note_drop("block_cap", raw.heading)
            moved = _without_sibling_mismatches(
                blocks[:space], " ".join([into.heading, into.verdict, *[b.title for b in blocks[:space]]]), request)
            if moved:
                if verdict and wording is not None:
                    verdict = _checked_sentences(verdict, [*into.blocks, *moved], request, wording, "claims")
                sections[target] = into.model_copy(update={
                    "blocks": [*into.blocks, *moved], "verdict": into.verdict or verdict})
                total += len(moved)
                if diag is not None:
                    diag.sections_merged += 1
                continue
            blocks = []
        if over_cap:
            note_drop("section_cap")
            continue
        if blocks:
            blocks = _without_sibling_mismatches(blocks, " ".join([raw.heading, raw.verdict, *[block.title for block in blocks]]), request)
        if verdict and wording is not None:
            verdict = _checked_sentences(verdict, blocks, request, wording, "claims")
        if not verdict and not blocks:
            note_drop("empty_section", raw.heading)
            continue
        total += len(blocks)
        written_heading = _checked_title(raw.heading, blocks, request, wording) if wording is not None else raw.heading
        heading = _section_heading(written_heading, index, titles, blocks, request, ground)
        sections.append(CardSection(heading=heading, verdict=verdict, blocks=blocks))
        kept_at.append(index)
    return [section for _, section in sorted(zip(kept_at, sections), key=lambda pair: pair[0])]


def _block_parts(block: CardBlock, by_id: dict[str, ResultData]) -> set[int]:
    return {by_id[i].sub_task for i in _block_result_ids(block) if i in by_id and by_id[i].sub_task}


def _section_words(section: CardSection) -> set[str]:
    return _content_words(" ".join([section.heading, section.verdict, *[block.title for block in section.blocks]]))


def _cross_part_blocks(section: CardSection, by_id: dict[str, ResultData], titles: dict[int, str]) -> int:
    """Blocks bound to another part than every other numbered block of their section, in a section whose text names
    its own part's title (a word only that part's title has) and none that only the block's part has. Only counted."""
    per_block = [_block_parts(block, by_id) for block in section.blocks]
    words = _section_words(section)
    count = 0
    for index, parts in enumerate(per_block):
        rest = set().union(*(other for i, other in enumerate(per_block) if i != index and other))
        if len(rest) != 1 or not parts or parts == rest:
            continue
        own = _content_words(titles.get(next(iter(rest)), ""))
        other = _content_words(" ".join(titles.get(part, "") for part in parts - rest))
        if (own - other) & words and not ((other - own) & words):
            count += 1
    return count


def _parts_tallied(sections: list[CardSection], request: AnswerCardRequest) -> list[CardSection]:
    """With part numbers on the results: tally, for the diagnostics only, the parts no section covers
    (``parts_uncovered``; not a trigger of the correction pass) and the blocks bound across parts. The sections come
    back in the order the model wrote them.

    A part is uncovered when it has a primary result and no block of any kept section is bound to any result of
    that part."""
    if not sections or not has_part_numbers(request):
        return sections
    diag = DIAGNOSTICS.get()
    if diag is None:
        return sections
    titles = part_titles(request)
    by_id = {result.id: result for result in request.results}
    bound = {rid for section in sections for block in section.blocks for rid in _block_result_ids(block)}
    diag.uncovered = []
    for number, title in titles.items():
        mine = [result for result in request.results if result.sub_task == number]
        primary = [result.id for result in mine if not result.supporting]
        if primary and not any(result.id in bound for result in mine):
            diag.uncovered.append((number, title, primary))
    diag.parts_uncovered = len(diag.uncovered)
    diag.cross_part_blocks = sum(_cross_part_blocks(section, by_id, titles) for section in sections)
    return sections


def _validated_card(card: AnswerCard, request: AnswerCardRequest) -> tuple[Optional[AnswerCard], bool]:
    """``(card, by_design)``: ``by_design`` is True when None is the intended answer (a lead-only card
    that restates the answer), not a model that produced nothing usable.

    With sections that survive, the card is wire v2: the sections plus their blocks flattened into
    ``blocks``. With none, it is today's flat card (``lead`` and ``blocks``). A card of only a lead is kept
    solely for a turn with no results (nothing else to show) and a lead that summarizes a long answer."""
    ground = _ground_sets(request)          # once per card
    anchors = _anchor_sets(request)
    wording = _Wording(request, anchors[1])
    written = _checked_sentences(card.lead, [], request, wording, "words")
    lead = _verified_text(written, LEAD_MAX_CHARS, ground)
    diag = DIAGNOSTICS.get()
    if written.strip() and not lead:
        note_drop("lead_ungrounded", card.lead)
    elif lead and (loose := _unanchored_sentence(lead, anchors)):
        note_drop("lead_number_unanchored", loose)
        lead = ""

    def claims_checked(blocks: list[CardBlock]) -> str:
        """The lead without the sentences that point at, or claim completeness over, blocks the card does not have."""
        checked = _checked_sentences(lead, blocks, request, wording, "claims") if lead else lead
        if diag is not None:
            diag.lead = "kept" if checked else ("dropped" if card.lead.strip() else "none")
        return checked
    sections = _parts_tallied(_verified_sections(card, request, ground, anchors, wording), request)
    if sections:
        flat = [b for s in sections for b in s.blocks]
        if diag is not None:
            diag.sections_kept, diag.blocks_kept = len(sections), len(flat)
        return AnswerCard(lead=claims_checked(flat), sections=sections, blocks=flat), False
    blocks = _verified_blocks(card.blocks, request, ground, set(), set(), MAX_BLOCKS, anchors, wording)
    lead = claims_checked(blocks)
    if diag is not None:
        diag.blocks_kept = len(blocks)
    if not lead and not blocks:
        note_drop("no_content")
        return None, False
    if not blocks and (request.results or not _lead_adds_something(lead, request.answer)):
        note_drop("lead_only_with_results" if request.results else "lead_restates", lead)
        return None, True               # a lead alone never hides the answer of a turn that has results to show
    return AnswerCard(lead=lead, blocks=blocks), False


def validate_answer_card(card: AnswerCard, request: AnswerCardRequest) -> Optional[AnswerCard]:
    """The card with an ungrounded lead emptied and every block checked against the data; None when nothing
    is left, or when only a lead is left that does not clearly summarize a long answer (``_lead_adds_something``)."""
    return _validated_card(card, request)[0]


# ----------------------------------------------------------------------
# Question form grounding
# ----------------------------------------------------------------------


def _data_candidates(results: list[ResultData]) -> set[str]:
    """Every non-empty string or number leaf and every dict key in the results' outputs, as stripped text."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                found.add(str(key).strip())
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            found.add(str(value).strip())

    for result in results:
        walk(result.output)
    found.discard("")
    return found


def _in_text(value: str, text: str) -> bool:
    """Whether ``value`` stands as a whole word or phrase in ``text`` (case-insensitive)."""
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE))


def _rescued_value(raw_value: str, candidates: set[str]) -> Optional[str]:
    """The one data ID inside a compound option value such as ``"#W2041120 (delivered)"``, else None."""
    inside = [
        c for c in candidates
        if len(c) >= 3 and (c.isascii() and (any(ch.isdigit() for ch in c) or "_" in c)) and _in_text(c, raw_value)
    ]
    # "2041120" inside "#W2041120" is the same ID: keep the longest only
    inside = [c for c in inside if not any(c != other and c in other for other in inside)]
    return inside[0] if len(inside) == 1 else None


def _label_for(value: str, label: str) -> str:
    """``label``, unless it names an ID that is not the option's value: the user must not read one thing and send another."""
    bare_value = value.lower()
    for token in _ID_TOKEN.findall(label):
        if _bare_id(token) != _bare_id(value) and _bare_id(token) not in bare_value:
            return value
    return label


def _grounded_options(field: FormField, candidates: set[str], question: str) -> list[FormOption]:
    options: list[FormOption] = []
    seen: set[str] = set()
    for option in field.options:
        raw_value = option.value.strip()
        raw_label = (option.label or raw_value).strip()[:80]
        if not raw_value:
            continue
        if raw_value in candidates or _in_text(raw_value, question):
            value = raw_value
        else:
            rescued = _rescued_value(raw_value, candidates)
            if rescued is None:
                continue
            value = rescued
        if value in seen:
            continue
        seen.add(value)
        options.append(FormOption(value=value, label=_label_for(value, raw_label)))
        if len(options) == MAX_OPTIONS:
            break
    return options


def _field_name(raw: str, taken: set[str], position: int) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if name and not name[0].isalpha():
        name = "f_" + name
    name = name[:40] or f"field_{position}"
    base, counter = name, 2
    while name in taken:
        suffix = f"_{counter}"
        name = base[:40 - len(suffix)] + suffix
        counter += 1
    return name


def _norm_key(text: str) -> str:
    """Lower-case, every run of non-alphanumerics as one ``_``: ``"Flagged Logins"`` -> ``flagged_logins``."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _key_values(results: list[ResultData]) -> dict[str, set[str]]:
    """Normalized dict key (any depth) -> the scalar values stored under it, as text; scalar lists count by item."""
    found: dict[str, set[str]] = {}

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                found.setdefault(_norm_key(name), set())
                walk(item, _norm_key(name))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif key and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            found[key].add(str(value).strip())

    for result in results:
        walk(result.output)
    return found


def _value_in_text(value: str, text: str) -> bool:
    """Whether ``value`` is in ``text`` as a whole word, or (for a number) as an equal number such as 54.2 and 54.20."""
    if _in_text(value, text):
        return True
    try:
        target = float(value)
    except ValueError:
        return False
    return any(float(token.replace(",", "")) == target for token in _NUMBER.findall(text))


def _question_mentions_key(question: str, key: str) -> bool:
    """Whether any word of ``key`` (``flagged_logins``: flagged, logins) starts a word of the question."""
    for word in key.split("_"):
        stem = word[:-1] if len(word) > 3 and word.endswith("s") else word
        if stem and re.search(r"(?<![a-z0-9])" + re.escape(stem), question, re.IGNORECASE):
            return True
    return False


_CHANGE_WORDS = re.compile(r"(?<![a-z])(new|different|another|other|correct|updated?|replacement)(?![a-z])", re.IGNORECASE)


def _restates_result(field: FormField, question: str, key_values: dict[str, set[str]]) -> bool:
    """Whether ``field`` only repeats a key of the results that the question does not ask for (guard G1).

    The field's normalized ``name`` or ``label`` must equal a key of the results (any depth). It is then
    KEPT when the question asks about that key, which is the case when
      * a word of the key is in the question (key ``order_id``, question "Which order ..."), or
      * the question names two or more of the key's values, or of the field's own options (the user is
        picking among alternatives the question lists);
    except that a text or number field is still dropped when the key holds ONE value in the results and
    either the question quotes that value as context ("Your order (status shipped) ...") or the question
    has no word asking for a different one (new, different, another, other, correct, updated,
    replacement): the answer is already known. Everything else is dropped: nobody is asked for it.
    """
    for text in (field.name, field.label):
        key = _norm_key(text)
        if key not in key_values:
            continue
        values = key_values[key]
        named = [value for value in values if value and _value_in_text(value, question)]
        offered = {o.value.strip() for o in field.options if _value_in_text(o.value.strip(), question)}
        if len(named) >= 2 or len(offered) >= 2:
            return False
        if not _question_mentions_key(question, key):
            return True
        if len(values) == 1 and field.kind in ("text", "number") and (
                named or not _CHANGE_WORDS.search(question)):
            return True
    return False


_NAMED_GAP = re.compile(r"^[^.?!;:]{0,40}$")
_ALTERNATIVE_WORDS = re.compile(r"(,|/|(?<![A-Za-z0-9_])(or|and)(?![A-Za-z0-9_]))", re.IGNORECASE)


def _named_alternatives(question: str, candidates: set[str]) -> list[str]:
    """The data values the question names as alternatives, in the order it names them; ``[]`` when unclear.

    A value counts when it is an ID-like leaf or key of the results (ASCII, at least 3 characters, with a
    digit or an underscore, and not a decimal or a number under 5 digits) standing as a whole word in the question. The values must form ONE run: each
    gap between neighbours is short, holds no sentence end, and has a comma, "/", "or" or "and".
    """
    hits: list[tuple[int, int, str]] = []
    for value in candidates:
        if len(value) < 3 or not value.isascii() or not (any(ch.isdigit() for ch in value) or "_" in value):
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", value) and ("." in value or len(value) < 5):
            continue                        # an amount or a small count is not an ID
        for match in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])", question, re.IGNORECASE):
            hits.append((match.start(), match.end(), value))
    hits.sort()
    kept: list[tuple[int, int, str]] = []
    for hit in hits:                        # "1001" inside "ACC-1001" is the same ID: keep the longer one
        if any(other != hit and other[0] <= hit[0] and hit[1] <= other[1] for other in hits):
            continue
        kept.append(hit)
    runs: list[list[tuple[int, int, str]]] = []
    for hit in kept:
        gap = question[runs[-1][-1][1]:hit[0]] if runs else ""
        if runs and _NAMED_GAP.match(gap) and _ALTERNATIVE_WORDS.search(gap):
            runs[-1].append(hit)
        else:
            runs.append([hit])
    big = [run for run in runs if len({item[2] for item in run}) >= 2]
    if len(big) != 1:
        return []
    ordered: list[str] = []
    for _, _, value in big[0]:
        if value not in ordered:
            ordered.append(value)
    return ordered


_LABEL_KEYS = ("labels", "names", "titles")


def _list_options(field: FormField, results: list[ResultData]) -> list[FormOption]:
    """Guard G6: the options for a text field named like ONE item of a list in the results, else ``[]``.

    ``ticket_id`` (name or label) matches the key ``ticket_ids`` holding at least two scalars: the answer
    can only be one of them. The newest result that has the key wins. An item's label is the entry at the
    same position of a sibling list named ``labels``, ``names`` or ``titles`` when it has the same length,
    else the value. At most ``MAX_OPTIONS`` options.
    """
    wanted = {_norm_key(field.name), _norm_key(field.label)} - {""}
    found: list[FormOption] = []

    def walk(value: Any) -> None:
        if found:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                norm = _norm_key(key)
                singular = norm[:-1] if norm.endswith("s") and len(norm) > 3 else norm
                if (norm in wanted or singular in wanted) and isinstance(item, list) and len(item) >= 2 \
                        and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in item):
                    names = next((value[k] for k in _LABEL_KEYS if isinstance(value.get(k), list) and len(value[k]) == len(item)), None)
                    for index, entry in enumerate(item[:MAX_OPTIONS]):
                        text = str(entry).strip()
                        label = str(names[index]).strip()[:80] if names and isinstance(names[index], (str, int, float)) else ""
                        if text and text not in {o.value for o in found}:
                            found.append(FormOption(value=text, label=_label_for(text, label or text)))
                    return
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for result in reversed(results):
        walk(result.output)
        if found:
            break
    return found if len(found) >= 2 else []


_YES_NO_START = re.compile(
    r"^\s*(do you (want|need|like)|would you like|should i|shall i|did you|have you|is there|are you sure)\b", re.IGNORECASE)
_WH_WORD = re.compile(r"(?<![A-Za-z])(which|what|who|whom|whose|where|when|how|why)(?![A-Za-z])", re.IGNORECASE)


def is_yes_no_question(question: str) -> bool:
    """Whether the question can only be answered yes or no (guard G5).

    It starts with an offer or a check ("Do you want", "Would you like", "Should I", "Did you", "Have you",
    "Is there", "Are you sure"), ends with "?", has no question word (which, what, who, where, when, how,
    why) and no " or ": "Do you want a refund or an exchange?" is a pick, not a yes/no. A request such as
    "Can you provide the name?" does not start that way.
    """
    text = " ".join(question.split())
    return (bool(_YES_NO_START.match(text)) and text.endswith("?") and text.count("?") == 1
            and not _WH_WORD.search(text) and not re.search(r"(?<![A-Za-z])or(?![A-Za-z])", text, re.IGNORECASE)
            and not _named_alternatives(text, set(re.findall(r"[A-Za-z0-9_#-]{3,}", text))))


def ground_question_form(
        form: QuestionForm, request: QuestionFormRequest, *, precision_gate: bool = True) -> Optional[QuestionForm]:
    """The form the user is shown: only what the question asks for, with only grounded options.

    In this order, per field and then per form:
      * a field with an empty label, or the same label as an earlier field, is dropped;
      * G1, ``_restates_result``: a field that only repeats a key of the results is dropped;
      * a choice's options are grounded (an option that is no value or key in the results and not in the
        question is dropped); a choice left without options becomes text;
      * G6, ``_list_options``: a text field named like one item of a list in the results is a choice of the
        list's items, and G1 does not drop it;
      * G3b: a text field whose own options, after grounding, number two or more is a choice (the model
        meant a choice and wrote the wrong kind);
      * G3, ``_named_alternatives``: when no choice already offers those values and exactly one field is a plain text field and the question names two
        or more data values as alternatives, that field is a choice of exactly those values, in the
        order the question names them, labelled with the value. Two competing text fields: no guess;
      * at most ``MAX_FIELDS`` fields;
      * G2: ``required`` is cleared on every field of a form with two or more fields, so a form never
        blocks "Send answer" on something the question did not need; a form of one field is required;
      * G4: no field left means ``None``;
      * G5, ``is_yes_no_question``: a yes/no question gets exactly one boolean field (the model's own
        boolean field if it made one, else "Yes or no"), whatever else the model returned;
      * G7, ``_precision_gate``, the last step: a form is shown only when it clearly helps. See there.

    Why G7: a genuine open question is rare (about 1 turn in 80 in the recorded agent runs) and the user can
    always type in the composer, so a wrong form costs more than no form. Every G7 rule can only turn a
    form into a smaller one or into ``None`` (the client then keeps the plain question); none adds anything.
    ``precision_gate=False`` skips G7 (tests of the earlier guards only; the endpoint never does).
    """
    if is_yes_no_question(request.question):
        own = next((f for f in form.fields if f.kind == "boolean" and f.label.strip()), None)
        label = " ".join(own.label.split())[:60] if own else "Yes or no"
        name = _field_name(own.name if own and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", own.name or "") else label, set(), 1)
        return QuestionForm(fields=[FormField(name=name, label=label, kind="boolean", required=True)])
    candidates = _data_candidates(request.results)
    key_values = _key_values(request.results)
    fields: list[FormField] = []
    taken: set[str] = set()
    labels: set[str] = set()
    listed_names: set[str] = set()
    for position, field in enumerate(form.fields, start=1):
        label = " ".join(field.label.split())[:60]
        if not label or label.lower() in labels:
            continue
        listed = _list_options(field, request.results) if field.kind == "text" and not field.options else []
        if not listed and _restates_result(field, request.question, key_values):
            continue
        labels.add(label.lower())
        valid_name = re.fullmatch(r"[a-z][a-z0-9_]{0,39}", field.name or "")
        name = _field_name(field.name if valid_name else (field.name or label), taken, position)
        taken.add(name)
        kind, options = field.kind, []
        if listed:
            kind, options = "choice", listed
            listed_names.add(name)
        elif field.options:
            options = _grounded_options(field, candidates, request.question)
            if kind in ("choice", "multichoice") and not options:
                kind = "text"
            elif kind == "text" and len(options) >= 2:
                kind = "choice"
            elif kind not in ("choice", "multichoice"):
                options = []
        fields.append(FormField(
            name=name, label=label, kind=kind, options=options, required=field.required, help=field.help.strip()[:200],
        ))
        if len(fields) == MAX_FIELDS:
            break
    _promote_to_named_choice(fields, request.question, candidates)
    if precision_gate:
        fields = _precision_gate(fields, request.question, listed_names)
    if not fields:
        return None
    return QuestionForm(fields=[f.model_copy(update={"required": len(fields) == 1}) for f in fields])


_STRUCTURED = ("choice", "multichoice", "boolean", "date")
_FREE_FORM = ("text", "longtext", "number")


def _asks_to_pick(field: FormField, question: str) -> bool:
    """Whether the question asks to pick: it has a pick word, or names at least two of the field's options."""
    if _PICK_WORDS.search(question):
        return True
    return sum(_in_text(o.value.strip(), question) for o in field.options) >= 2


def _precision_gate(fields: list[FormField], question: str, listed_names: set[str]) -> list[FormField]:
    """Guard G7, the precision gate: ``[]`` (no form) unless the form clearly helps the user. In this order:

      * a choice or multichoice that the question does not ask to pick from becomes a text field: the
        question has no pick word (which, select, choose, pick, one of, any of, among, from the list) and
        names fewer than two of the options. A model that, for "Do you want more details?", lists a dozen
        IDs from the results has invented a menu. Exempt: a field G6 built from a list in the results,
        unless the question is an offer or a check ("Do you want", "Would you like", ...);
      * a choice or multichoice with fewer than 2 options becomes a text field (0 and 1 alike: nothing
        is being chosen);
      * the form is worth showing only if some field gives the user what the composer does not: a choice
        or multichoice (it has 2 or more options now), a boolean or a date. A form of only text, longtext
        and number fields is dropped;
      * of the free-form fields (text, longtext, number) of a form that stays, only the first is kept.
        On the recorded outputs this removed 3 fields that no label wanted and none that a label wanted.
    """
    gated: list[FormField] = []
    for field in fields:
        if field.kind in ("choice", "multichoice") and (
                len(field.options) < 2 or (not _asks_to_pick(field, question) and (
                    field.name not in listed_names or _YES_NO_START.match(" ".join(question.split()))))):
            field = field.model_copy(update={"kind": "text", "options": []})
        gated.append(field)
    if not any(f.kind in _STRUCTURED for f in gated):
        return []
    kept: list[FormField] = []
    for field in gated:
        if field.kind in _FREE_FORM and any(f.kind in _FREE_FORM for f in kept):
            continue
        kept.append(field)
    return kept


def _promote_to_named_choice(fields: list[FormField], question: str, candidates: set[str]) -> None:
    """Guard G3, in place: see ``ground_question_form``."""
    plain = [index for index, field in enumerate(fields) if field.kind == "text" and not field.options]
    if len(plain) != 1:
        return
    named = _named_alternatives(question, candidates)
    offered = {o.value for f in fields if f.kind in ("choice", "multichoice") for o in f.options}
    if len(named) >= 2 and not set(named) & offered:
        index = plain[0]
        fields[index] = fields[index].model_copy(update={
            "kind": "choice", "options": [FormOption(value=value, label=value) for value in named[:MAX_OPTIONS]],
        })


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------


class ComposerUnavailable(Exception):
    """``LLM_RESPONSE_GEN`` is not usable (no model, or a proxy model without its base URL)."""


def _composer_lm():
    """The composer's language model, or ``ComposerUnavailable`` when it is not configured."""
    try:
        return dspy_utils.get_lm("LLM_RESPONSE_GEN")
    except ValueError as error:
        raise ComposerUnavailable(str(error)) from error


def model_id() -> str:
    """The composer model's id for a diagnostics record, or "" when none is configured. Never raises."""
    try:
        return str(getattr(_composer_lm(), "model", "") or "")
    except Exception:  # noqa: BLE001 - a diagnostics label must not fail a compose call
        return ""


# The ids of the worked example in the instructions. Real results are ``r1`` ... ``r60``, so these never occur in a
# request: a block that points at one copied the example (``example_copied``).
EXAMPLE_RESULT_IDS = frozenset({"r901", "r902"})


class ComposeAnswerCard(dspy.Signature):
    """Summarize the results of a turn as an answer card: a lead, then sections.

    The lead is one or two sentences that state the outcome in words. Cover every numbered part of the request
    (see sub_tasks) with one section each, or merge parts into one section on purpose. A section has a heading that
    states the finding for that part (3 to 6 words, never the command's name), a verdict (one full sentence that
    says the outcome, about 70 to 95 characters, not a label) and up to 4 blocks that show the data. A result that
    is a list or a page is shown as a table block, by ref with fields; prefer a table to a bare count whenever rows
    exist, and use the result's "show" entry as the block (its kind, ref and fields). Put counts and amounts in stats blocks, not in the lead or
    a verdict; a number may appear there only exactly as the answer states it. A block has a kind: "stats" (with
    stats: a list of {label, ref}), "table" or "record" (with ref), or "quote" (with quote: words copied from the
    answer). A ref points at data as r1.<field>, for example r1.status; never write an index or a path through
    the results' structure (r2.page.sample.0 is wrong). A stat label must name what its ref holds (use the field's title).
    Copy every ID exactly; never invent a number. Show only what helps. The example only shows the shape.
    """

    user_message: str = dspy.InputField(desc="What the user asked.")
    answer: str = dspy.InputField(desc="The answer already given to the user.")
    sub_tasks: list[str] = dspy.InputField(desc="The answer's numbered sub-task headings; empty when there are none.")
    results: str = dspy.InputField(
        desc="JSON list of the command results: ids, fields with descriptions and hints, types, samples, row counts "
             "and a \"show\" entry with the blocks that show a list.")
    lead: str = dspy.OutputField(desc="One or two sentences; state outcomes in words; a number only as the answer states it.")
    sections: list[Any] = dspy.OutputField(
        desc=f"JSON list of up to {MIN_SECTIONS} sections: {{heading, verdict, blocks}}. Example for a library request "
             "(invented ids; lead: \"The member has loans past their due date and a fine to settle.\"): "
             '[{"heading": "Overdue loans and due dates", "verdict": "The member has overdue loans, each with a due date '
             'already past.", "blocks": [{"kind": "table", "title": "Overdue loans", '
             '"ref": "r901", "fields": ["loan_titles", "loan_due_dates"]}]}, {"heading": "Fine owed on the account", '
             '"verdict": "The account owes a fine for the late returns.", "blocks": [{"kind": "stats", '
             '"title": "", "stats": [{"label": "Fine owed", "ref": "r902.fine_amount"}]}]}]')


# The correction call: the same signature plus one input. The first call keeps ``ComposeAnswerCard`` as it was, so
# its prompt (and every cached or recorded answer) is byte-identical to a composition with no retry.
ComposeAnswerCardCorrected = ComposeAnswerCard.append(
    "corrections", dspy.InputField(desc="What was wrong with a first attempt at this card; do not repeat it."), str)


@functools.lru_cache(maxsize=4 * MAX_SECTIONS)
def _card_signature(cap: int, corrected: bool) -> type[dspy.Signature]:
    """The card signature for a request that allows ``cap`` sections: its instructions and its ``sections``
    field both state the cap (``section_cap``), and ask to merge related parts when there are more than that."""
    base = ComposeAnswerCardCorrected if corrected else ComposeAnswerCard
    desc = base.output_fields["sections"].json_schema_extra["desc"].replace(
        f"up to {MIN_SECTIONS} sections", f"up to {cap} sections")
    return base.with_instructions(
        f"{base.instructions} Write at most {cap} sections; when the request has more parts than that, merge "
        "related parts into one section."
    ).with_updated_fields("sections", desc=desc)


class ComposeQuestionForm(dspy.Signature):
    """Turn a question put to a person into the fields they must fill to answer it.

    Most questions need exactly ONE field. Add a second field only when the question asks a second thing.
    A question that can be answered yes or no gets ONE boolean field, even when it offers to do more.
    Never make a field for context, background or data that is already in the results.
    If the question names two or more alternatives, or asks to pick among items in the results, make ONE
    field of kind choice (pick one) or multichoice (more than one may apply at once), with ALL the
    alternatives as options, also when they are plain words in the question. Copy each option value exactly
    from the question or the results, such as an ID. Write a short readable label for each option.
    Use date for a date, number for a quantity, longtext for an explanation, and text for a name, ID or any
    other free answer.
    Set required to false. Keep each label short and in the words of the question.
    If the question asks for nothing to fill in, return an empty list.
    """

    question: str = dspy.InputField(desc="The question being asked of the person.")
    user_message: str = dspy.InputField(desc="What the person asked earlier.")
    results: str = dspy.InputField(desc="JSON list of data already known: ids, fields, types and samples.")
    known_values: str = dspy.InputField(
        desc="JSON list of IDs and keys found in the results. Offer them as option values only when the question asks to pick.")
    form_fields: list[Any] = dspy.OutputField(  # not `fields`: dspy.Signature already has that attribute
        desc="JSON list of fields: {name, label, kind, options, required, help}. kind is one of text, longtext, "
             "number, boolean, date, choice, multichoice. options is [{value, label}] for choice and multichoice "
             "only. Example for a question that asks which of two items to use: "
             '[{"name": "item", "label": "Item to use", "kind": "choice", "options": [{"value": "QX-7", '
             '"label": "QX-7"}, {"value": "QX-8", "label": "QX-8"}], "required": false, "help": ""}]')


def _catalog(model: str) -> dict[str, Any]:
    """litellm's model-map entry for ``model`` (``{}`` when it has none): the id as given, without its routing
    prefix, and its last path segment."""
    costs = getattr(litellm, "model_cost", None) or {}
    name = str(model or "")
    for key in (name, name.split("/", 1)[-1], name.rsplit("/", 1)[-1]):
        entry = costs.get(key)
        if isinstance(entry, dict):
            return entry
    return {}


def _lm_facts(lm) -> tuple[bool, frozenset[str]]:
    """``(the model thinks, the OpenAI-style parameters it takes)`` as the installed DSPy (which asks LiteLLM or
    the native client) reports them; ``(False, {})`` when unknown. An empty set means "unknown", not "none"."""
    try:
        reasoning = bool(lm.supports_reasoning)
    except Exception:  # noqa: BLE001 - a client that cannot say is a client we change nothing for
        reasoning = False
    try:
        params = frozenset(lm.supported_params)
    except Exception:  # noqa: BLE001
        params = frozenset()
    return reasoning, params


# Models whose reasoning switch the provider refused in this process: later calls go without it.
_SWITCH_REFUSED: set[str] = set()
_SWITCH_KEYS = ("thinking", "reasoning_effort")


def _reasoning_switch(model: str, params: frozenset[str], entry: dict[str, Any]) -> dict[str, Any]:
    """The setting that asks a model that thinks not to (or to think least), from what LiteLLM says the model takes.

    An Anthropic-style model takes ``thinking`` as well as ``reasoning_effort``; LiteLLM reads any
    ``reasoning_effort`` as thinking ON for it, so the explicit ``thinking={"type": "disabled"}`` is the switch.
    Otherwise the lowest ``reasoning_effort`` the model map says the model accepts. ``{}`` when there is none."""
    if model in _SWITCH_REFUSED:
        return {}
    if "thinking" in params and "reasoning_effort" in params:
        return {"thinking": {"type": "disabled"}}
    if "reasoning_effort" in params:
        for level in ("none", "minimal", "low"):
            if entry.get(f"supports_{level}_reasoning_effort") is True:
                return {"reasoning_effort": level}
    return {}


def effective_max_tokens(lm, budget: int) -> int:
    """The completion limit to send for an answer of ``budget`` tokens: more for a model that thinks
    (``REASONING_HEADROOM_TOKENS``), never more than the model's own output ceiling when the model map has one."""
    reasoning, _ = _lm_facts(lm)
    limit = budget + (REASONING_HEADROOM_TOKENS if reasoning else 0)
    ceiling = _catalog(getattr(lm, "model", "")).get("max_output_tokens")
    if isinstance(ceiling, int) and ceiling > 0:
        limit = min(limit, ceiling)
    return limit


def composer_call_options(lm, budget: int) -> dict[str, Any]:
    """The settings of one composer call, as ``lm.copy(**options)``: the completion limit, a temperature of 0 only
    when the model takes one (the model map says it removed sampling parameters, or its parameter list lacks
    ``temperature``: no call is wasted on a refusal), and for a model that thinks the switch that turns thinking off.
    Nothing is decided by a model name; a model LiteLLM knows nothing about is called as before."""
    reasoning, params = _lm_facts(lm)
    entry = _catalog(getattr(lm, "model", ""))
    options: dict[str, Any] = {"max_tokens": effective_max_tokens(lm, budget)}
    takes_temperature = entry.get("supports_sampling_params") is not False and (not params or "temperature" in params)
    options["temperature"] = 0.0 if takes_temperature else None     # None removes the setting from the request
    if reasoning:
        options.update(_reasoning_switch(str(getattr(lm, "model", "")), params, entry))
    return options


_TRUNCATED_FINISH = frozenset({"length", "max_tokens", "max_output_tokens", "model_length"})


def _json_is_cut(text: str) -> bool:
    """Whether a DSPy-formatted answer stopped inside its ``sections`` JSON: no closing marker and an unclosed
    bracket, brace or string."""
    marker = "[[ ## sections ## ]]"
    if marker not in text or "[[ ## completed ## ]]" in text:
        return False
    depth, in_string, escaped = 0, False, False
    for char in text.split(marker, 1)[1]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
    return in_string or depth > 0


def _was_truncated(lm, limit: int) -> bool:
    """Whether the last call on ``lm`` ended at its token limit: the provider's finish reason says so, its
    completion tokens reached the limit, or the answer's JSON is cut."""
    try:
        entry = (getattr(lm, "history", None) or [])[-1]
        choices = getattr(entry.get("response"), "choices", None) or []
        if choices and str(getattr(choices[0], "finish_reason", "") or "").lower() in _TRUNCATED_FINISH:
            return True
        tokens = (entry.get("usage") or {}).get("completion_tokens")
        if isinstance(tokens, int) and limit and tokens >= limit:
            return True
        return any(_json_is_cut(output) for output in entry.get("outputs") or [] if isinstance(output, str))
    except Exception:  # noqa: BLE001 - no history is no evidence
        return False


def _mentions_reasoning(error: Optional[BaseException]) -> bool:
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if any(word in str(error).lower() for word in ("thinking", "reasoning")):
            return True
        error = error.__cause__ or error.__context__
    return False


def _predict(
        signature: type[dspy.Signature], use_cache: Optional[bool], max_tokens: int = COMPOSER_MAX_TOKENS,
        **inputs: Any) -> dspy.Prediction:
    """One prediction on a deterministic copy of the composer model (``composer_call_options``); litellm's refused
    parameters are dropped. When the provider refuses the reasoning switch the call is made once more without it
    (and the model is remembered). A call cut off at its token limit sets ``truncated`` on the diagnostics."""
    base_lm = _composer_lm()
    changes = composer_call_options(base_lm, max_tokens)
    if use_cache is False:
        changes["cache"] = False
    diag = DIAGNOSTICS.get()
    used: list = []

    def run(settings: dict[str, Any]) -> dspy.Prediction:
        def call(lm):
            used.append(lm)
            with dspy_utils.role_context("LLM_RESPONSE_GEN", lm, adapter=COMPOSER_ADAPTER):
                return dspy.Predict(signature)(**inputs)
        try:
            return dspy_utils.call_dropping_rejected_params(base_lm.copy(**settings), call)
        finally:
            if used and diag is not None:
                diag.token_limit = int(settings.get("max_tokens") or 0)
                if _was_truncated(used[-1], diag.token_limit):
                    diag.truncated = True

    try:
        return run(changes)
    except Exception as error:  # noqa: BLE001 - re-raised unless the provider refused the reasoning switch
        switch = [key for key in _SWITCH_KEYS if key in changes]
        if not switch or not _mentions_reasoning(error):
            raise
        _SWITCH_REFUSED.add(str(getattr(base_lm, "model", "")))
        logger.warning(f"{getattr(base_lm, 'model', '')} refused the reasoning switch; its calls now go without it")
        return run({key: value for key, value in changes.items() if key not in switch})


class _NotNeeded:
    """What a composer returns when it chose, by design, to show nothing (not a failure)."""

    def __repr__(self) -> str:
        return "NOT_NEEDED"

    def __bool__(self) -> bool:
        return False


NOT_NEEDED = _NotNeeded()


def _compose_once(
        request: AnswerCardRequest, digest: str, diag: ComposeDiagnostics, corrections: str = "",
        budget: Optional[int] = None,
) -> "Optional[AnswerCard] | _NotNeeded":
    """One model call and its validation, tallied in ``diag`` (which must be the current ``DIAGNOSTICS``).

    ``budget`` is the output budget (default: ``card_token_budget`` of the request's section cap). The model is
    shown ``answer_for_prompt(request.answer)``; ``_validated_card`` and everything under it read ``request.answer``."""
    extra: dict[str, Any] = {"corrections": corrections} if corrections else {}
    shown = answer_for_prompt(request.answer, request.parts)
    diag.answer_chars = len(request.answer)
    diag.excerpt_chars = len(shown) if shown is not request.answer else 0
    diag.parts = len(part_titles(request)) if has_part_numbers(request) else 0
    try:
        prediction = _predict(
            _card_signature(section_cap(request), bool(corrections)), request.use_dspy_cache,
            budget or card_token_budget(section_cap(request)), user_message=request.user_message, answer=shown,
            sub_tasks=request_sub_tasks(request), results=digest, **extra,
        )
        diag.raw_sections_chars = _raw_chars(prediction.sections)
        logger.debug(f"compose answer_card raw lead={_raw_text(prediction.lead)!r}")
        logger.debug(f"compose answer_card raw sections={_raw_text(prediction.sections)}")
        parsed = AnswerCard(lead=prediction.lead, sections=prediction.sections)
        _note_raw_sections(prediction.sections, parsed.sections, diag)
        card, by_design = _validated_card(parsed, request)
        return NOT_NEEDED if card is None and by_design else card
    except ComposerUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed composition falls back to the deterministic card
        diag.add("model_failed")
        log = logger.debug if corrections else logger.warning
        log(f"answer card composition failed: {type(exc).__name__}")
        return None


def _fields_text(result: Optional[ResultData]) -> str:
    return ", ".join(list(result.output)[:MAX_FIELDS * 2]) if result is not None else ""


def _correction_for(reason: str, detail: str, ref: str, request: AnswerCardRequest) -> str:
    """One plain-language fix, from facts the code knows and the model was shown (field names, its own text)."""
    result = _result_for(ref, request.results) if ref else None
    if reason in ("ref_unresolved", "stat_ref_unresolved"):
        if result is None:
            return f"ref {ref} does not exist; the results are {', '.join(r.id for r in request.results)}"
        return f"ref {ref} does not exist; {result.id} has fields: {_fields_text(result)}"
    if reason == "not_a_table":
        return f"ref {ref} is not a list of records; show it as a record or a stat"
    if reason == "not_a_record":
        return f"ref {ref} is not a record; show it as a table or a stat"
    if reason == "stat_not_scalar":
        return f"ref {ref} is not a single value; a stat needs one"
    if reason == "stat_label_unsupported":
        key = ref.split(".")[-1]
        title = (result.titles.get(key, "") if result is not None else "") or key
        return f"label '{detail}' does not describe {ref} ({title})"
    if reason == "title_unsupported":
        return f"title '{detail}' does not describe the data it heads"
    if reason in ("verdict_number_unanchored", "lead_number_unanchored"):
        place = "verdict" if reason.startswith("verdict") else "lead"
        return (f"{place} '{detail}' uses a number the answer does not state that way; "
                "say it in words or use a stats block")
    if reason in ("verdict_ungrounded", "lead_ungrounded"):
        return f"{reason.split('_')[0]} '{detail}' holds a number or ID that is not in the answer or the results"
    if reason == "quote_ungrounded":
        return f"quote '{detail}' is not in the answer"
    if reason == "example_copied":
        return f"ref {ref} comes from the example in the instructions, not from the results"
    if reason == "bad_kind":
        return f"block kind '{detail}' is not one of stats, table, record, quote"
    if reason == "empty_section":
        return f"section '{detail}' had nothing valid to show"
    return "a section or block was not a JSON object with the expected fields"


def corrections_text(diag: ComposeDiagnostics, request: AnswerCardRequest) -> str:
    """The correctable drops of a first pass as a short list of fixes, at most ``CORRECTIONS_MAX_CHARS``."""
    lines: list[str] = []
    for reason, detail, ref in diag.details:
        if reason in CORRECTABLE_DROPS or reason in CORRECTION_HINT_DROPS:
            line = _correction_for(reason, detail, ref, request)
            if line not in lines:
                lines.append(line)
    text = ""
    for line in lines:
        if len(text) + len(line) + 3 > CORRECTIONS_MAX_CHARS:
            break
        text += f"- {line}\n"
    return text.strip()


CUT_OFF_CORRECTION = "- the first attempt was cut off at the output limit; write fewer, shorter sections"


def _larger_budget(diag: ComposeDiagnostics, elapsed: float, request: Optional[AnswerCardRequest] = None) -> int:
    """The answer budget of a second call after a cut-off first one, or 0 when it cannot help: the model has no
    larger limit to give (its own output ceiling is reached), or a call that may take as many times longer as
    the limit grows would end after the page has given up (``PAGE_GIVE_UP_SECONDS``). Never a third call."""
    try:
        base = card_token_budget(section_cap(request)) if request is not None else CARD_MAX_TOKENS
        bigger = base + TRUNCATION_RETRY_TOKENS
        limit = effective_max_tokens(_composer_lm(), bigger)
    except ComposerUnavailable:
        return 0
    if limit <= diag.token_limit or elapsed + elapsed * limit / max(diag.token_limit, 1) > PAGE_GIVE_UP_SECONDS:
        return 0
    return bigger


def _card_score(card: Any, diag: ComposeDiagnostics) -> tuple[int, int, int, int]:
    """Higher is better: sections kept, blocks kept, a lead kept, fewer correctable drops."""
    if not isinstance(card, AnswerCard):
        return (0, 0, 0, -diag.correctable())
    return (diag.sections_kept, diag.blocks_kept, int(diag.lead == "kept"), -diag.correctable())


def compose_answer_card(
        request: AnswerCardRequest, diagnostics: Optional[ComposeDiagnostics] = None,
) -> "Optional[AnswerCard] | _NotNeeded":
    """A verified card; None when the model failed or nothing it returned survived; ``NOT_NEEDED`` when the
    only thing left was a lead that restates the answer (the deterministic view is the better card).

    ``diagnostics`` (or the one the endpoint put in ``DIAGNOSTICS``) is filled with what was proposed, kept and
    dropped; the raw model output goes to the DEBUG log only.

    When the first pass drops something a second look can fix (``CORRECTABLE_DROPS``) and took less than
    ``RETRY_SKIP_AFTER_SECONDS``, the model is asked once more with the fixes (``corrections_text``). The
    second answer is validated with a fresh tally and kept only when strictly better (``_card_score``); a second
    call that fails is ignored. Never a third call."""
    diag = diagnostics or DIAGNOSTICS.get() or ComposeDiagnostics()
    token = DIAGNOSTICS.set(diag)
    diag.results = len(request.results)
    try:
        digest = results_digest(request.results)
        diag.digest_chars = len(digest)
        started = time.monotonic()
        first = _compose_once(request, digest, diag)
        elapsed = time.monotonic() - started
        bigger = _larger_budget(diag, elapsed, request) if diag.truncated else 0
        if not RETRY_ENABLED or not (diag.correctable() or bigger):
            return first
        if "model_failed" in diag.drops and not bigger:
            return first
        if elapsed > RETRY_SKIP_AFTER_SECONDS:
            diag.retry = "skipped_slow"
            return first
        second_diag = ComposeDiagnostics()
        second_diag.results, second_diag.digest_chars = diag.results, diag.digest_chars
        DIAGNOSTICS.set(second_diag)
        try:
            second = _compose_once(
                request, digest, second_diag,
                "\n".join(part for part in (corrections_text(diag, request), CUT_OFF_CORRECTION if diag.truncated else "") if part),
                bigger or card_token_budget(section_cap(request)))
        except ComposerUnavailable:
            second = None
            second_diag.add("model_failed")
        finally:
            DIAGNOSTICS.set(diag)
        if "model_failed" in second_diag.drops:
            diag.retry = "failed"
            return first
        diag.second = second_diag
        if _card_score(second, second_diag) > _card_score(first, diag):
            diag.retry = "used"
            return second
        diag.retry = "kept_first"
        return first
    finally:
        DIAGNOSTICS.reset(token)


KNOWN_VALUES_MAX = 20


_PICK_WORDS = re.compile(r"(?<![A-Za-z])(which|select|choose|pick|one of|any of|among|from the list)(?![A-Za-z])", re.IGNORECASE)


def known_values(results: list[ResultData], question: str = "") -> str:
    """JSON list of up to 20 ID-like values and dict keys of the results, newest result first; ``[]`` unless
    the question asks to pick (which, select, choose, pick, one of, any of, among, from the list).

    Offered for a yes/no question the list tempts a small model into a choice nobody asked for.

    The digest shows lists of records only as a count, so a model that must offer "one of the orders"
    could not see the IDs. ID-like: ASCII, 3 to 40 characters, with a digit or an underscore, no spaces.
    """
    found: list[str] = []
    if not _PICK_WORDS.search(question):
        return "[]"

    def add(text: str) -> None:
        text = text.strip()
        ok = 3 <= len(text) <= 40 and text.isascii() and " " not in text and (
            any(ch.isdigit() for ch in text) or "_" in text)
        if ok and text not in found and len(found) < KNOWN_VALUES_MAX:
            found.append(text)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    add(str(key))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            add(str(value))

    for result in reversed(results):
        walk(result.output)
    return json.dumps(found, ensure_ascii=False)


def compose_question_form(request: QuestionFormRequest) -> "Optional[QuestionForm] | _NotNeeded":
    """A grounded form; None when the model failed; ``NOT_NEEDED`` when the model answered but no form is
    worth showing (no field survived, or G7 judged the form not worth showing): the plain question stays."""
    try:
        prediction = _predict(
            ComposeQuestionForm, request.use_dspy_cache,
            question=request.question, user_message=request.user_message, results=results_digest(request.results),
            known_values=known_values(request.results, request.question),
        )
        form = QuestionForm(fields=prediction.form_fields)
        return ground_question_form(form, request) or NOT_NEEDED
    except ComposerUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed composition falls back to the deterministic question
        logger.warning(f"question form composition failed: {type(exc).__name__}")
        return None
