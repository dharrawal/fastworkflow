"""Answer coverage statement: the extractor is told what the run never retrieved.

``ido-8ps.22`` / ``ido-8ps.23``. Two measured failures share one shape.

* ``ido-8ps.22`` (D1, ``exp-ido-gqv-2-20260915T225309``): answers assert that
  data is *unavailable* for queries the turn never ran -- "no finding
  description containing Christopher Hubbard was available" after searching
  finding labels and never running ``show_affected_entities``. An assertion of
  absence is a claim about the world; the run only ever earned a claim about
  itself.
* ``ido-8ps.23`` (neutral-v2, ``exp-ido-8ps-20-20260915T221729``): the two
  attempts that ended at the iteration limit completed their per-person tables
  from queries never issued; the three that finished normally fabricated
  nothing. A writer who is not told the run was cut short writes as though it
  was not.

Both are answered by one deterministic statement, built from the request text
and the turn's own stores and prepended to the extract input:

    Coverage of this run: the loop ended normally.
    These named items from the request appear in no retrieved observation:
    Christopher Hubbard.
    For each of them report "not retrieved" and nothing else - no value, no
    unavailability, no absence.
    These named items of the request DO appear in this run's observations:
    Alan Cooper; Brandon Miller.
    Every other named item of the request WAS retrieved: it appears in this
    run's observations and must be reported from them. Do not write "not
    retrieved", "not available", "no data", or any other statement of absence
    about an item that is not named in the unobserved list above.
    For items that appear, report only what the observations show.

``ido-8ps.24`` is the third and fourth sentences, added after the D3+D4 cell
(``exp-ido-gqv-5-20260916T035444``) measured the first version. That version
named the unobserved set and then said only "For items that appear, report only
what the observations show"; one attempt read it as licence to write "not
retrieved" against three identity uids that were in its own holder pages. Naming
a set is not the same claim as saying what the REST of the set is, so the block
now says what the rest of the set is, and names it where it fits.

Rules this module does not bend:

* **No model chooses anything.** Entities come out of the request by regex;
  presence is a normalised substring test against bytes a command in this turn
  produced and the store kept under a digest.
* **Nothing is invented.** An item the extractor is told to mark "not retrieved"
  is named verbatim from the request; nothing is said about what it *would* have
  been.
* **Rehydration is unchanged.** This runs immediately after it, on the copy it
  returned, and adds one key. The ReAct loop's own trajectory is never touched.
* **The post-check is measurement, never a gate.** It counts unavailability
  phrasing near unobserved items in the finished answer and records the count.
  It never edits, rejects or retries an answer.

Flag: ``FW_ANSWER_COVERAGE=1`` (default ``0`` -- off, and with it off the extract
call receives byte-for-byte what it received at ``4832b3c``).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import (
    is_search_answer_key,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    default_scope,
    stored_handles,
)

logger = logging.getLogger(__name__)

#: The feature flag. Off is the ``ido-8ps.18`` accepted stack, unchanged.
ANSWER_COVERAGE_ENV = "FW_ANSWER_COVERAGE"

#: The key the statement is stored under in the extractor's trajectory copy.
#: Deliberately not an ``observation_`` key: it is a statement ABOUT the run, not
#: a tool result, and it is inserted FIRST so the adapter renders it before the
#: trajectory the extractor is being asked to read.
COVERAGE_KEY = "coverage_statement"

#: ``build_query_with_next_steps`` hands the agent ``<request>\n\nExecute these
#: next steps:\n<plan>``. The plan is a model's text about the request, not the
#: request, so entities are taken from the part before this marker. Splitting is
#: literal and deterministic; a query without the marker is used whole.
PLAN_MARKER = "\n\nExecute these next steps:\n"
#: The prefix the same builder adds when it passes agent inputs and trajectory.
REQUEST_PREFIX = "User Query:\n"

#: Bounds on what counts as one named item, so a runaway span cannot become a
#: paragraph-long "entity" and a 300-name request cannot become the prompt.
MIN_ENTITY_CHARS = 3
MAX_ENTITY_CHARS = 120
MAX_ENTITIES = 64

#: Only these kinds are ever INSTRUCTED as "not retrieved". A quoted phrase is
#: extracted and measured, never instructed, and the offline replay
#: (``evaluation/artifacts/result-search/honesty-replay.md``) is why: the pinned
#: card quotes its control as 'Active contractor identities whom manager left'
#: while the catalogue's label is ``Contractor whom manager left``, so the exact
#: phrase appears in no observation of 83 of 88 stored attempts -- INCLUDING the
#: 28 in which that branch was judged CORRECT. A quoted phrase in a request is
#: narrative framing; a name, a uid and an address are handles the workflow
#: itself prints. Instructing "not retrieved" on framing would manufacture the
#: very false absence ``ido-8ps.22`` is about.
INSTRUCTED_KINDS = frozenset({"name", "uid", "email"})

# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

from fastworkflow.answer_rehydration import env_value as _env_value  # noqa: E402


def answer_coverage_enabled() -> bool:
    """True when ``FW_ANSWER_COVERAGE`` is set to a truthy value.

    Read env file first, then the process environment -- the rule
    ``auto_navigation`` and ``answer_rehydration`` use, so one run's flags are
    read one way.
    """
    return _env_value(ANSWER_COVERAGE_ENV).lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(text: Any) -> str:
    """NFKC, casefolded, whitespace collapsed -- the one comparison form.

    Presence is decided on this form and nothing else, so a name broken across a
    line in a rendered row still matches the name written on one line in the
    request, and a full-width or ligature variant matches its plain spelling.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(folded.split())


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

def request_text(user_query: Any) -> str:
    """The user's request, without the planner's todo list.

    The agent's ``user_query`` is the refined request with a generated plan
    appended under a fixed marker. The plan is a model's paraphrase: a name it
    invents is not a named item of the request, and telling the extractor to
    report "not retrieved" for one would be this module inventing work. So the
    text before the first marker is the request, and a query that never went
    through the planner is its own request.
    """
    text = str(user_query or "")
    if text.startswith(REQUEST_PREFIX):
        text = text[len(REQUEST_PREFIX):]
    marker = text.find(PLAN_MARKER)
    return text if marker < 0 else text[:marker]


# ---------------------------------------------------------------------------
# Named entities
# ---------------------------------------------------------------------------

#: A uid as this workflow prints them: a long unbroken hex run.
_UID_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{16,64}(?![0-9A-Za-z])")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]")
#: ``'...'`` where the opening quote follows a space or an opener and the closing
#: quote precedes a space or punctuation. That rule is what keeps the apostrophe
#: of "this quarter's" from opening a quotation.
_SQUOTE_RE = re.compile(r"(?<![^\s(\[{])'([^'\n]{2,120})'(?=[\s.,;:!?)\]}]|$)")
_DQUOTE_RE = re.compile(r"[\"“]([^\"”\n]{2,120})[\"”]")
_CURLY_SQUOTE_RE = re.compile(r"‘([^’\n]{2,120})’")

#: Stripped off the ends of a token; never from inside it, so
#: ``Active Directory_Cloud`` and ``O'Brien`` survive intact.
_LEAD_STRIP = "\"'“”‘’([{"
_TRAIL_STRIP = "\"'“”‘’)]}"
#: A token ending in one of these ends the run it is in: a comma or a full stop
#: separates two names, it does not join them.
_BREAK_AFTER = ",;:.!?—–"
_SENTENCE_END = ".!?"


def _clean_token(raw: str) -> str:
    return raw.strip(_LEAD_STRIP).rstrip(_TRAIL_STRIP + _BREAK_AFTER).strip(_LEAD_STRIP)


def _is_capitalised(token: str) -> bool:
    """First letter uppercase, and there is a letter. Digits alone are not names."""
    for char in token:
        if char.isalpha():
            return char.isupper()
    return False


@dataclass(frozen=True)
class Entity:
    """One named item of the request, with where it was found."""

    text: str
    kind: str  # "name" | "quoted" | "uid" | "email"
    start: int

    @property
    def key(self) -> str:
        return normalise(self.text)


def _name_spans(text: str) -> list[Entity]:
    """Maximal runs of two or more consecutive capitalised tokens.

    A run is broken by a lowercase token and by the punctuation that ends a
    token, so "Alan Cooper, Alisha Ochoa" is two names rather than one. A run
    that starts a sentence loses its first token when two or more remain --
    "Two Active Directory rights" is about *Active Directory*, and the capital on
    "Two" is grammar, not a name. "Christopher Hubbard is one of the people it
    names" keeps both of its tokens: dropping the sentence capital there would
    leave a bare surname, which is a worse handle than the name itself.
    """
    tokens: list[tuple[str, int, bool, bool]] = []  # cleaned, start, sentence_start, breaks
    previous_end = 0
    previous_raw = ""
    for match in re.finditer(r"\S+", text):
        raw = match.group(0)
        gap = text[previous_end:match.start()]
        sentence_start = (
            not tokens
            or "\n" in gap
            or previous_raw.rstrip(_TRAIL_STRIP).endswith(tuple(_SENTENCE_END))
        )
        cleaned = _clean_token(raw)
        breaks = raw.rstrip(_TRAIL_STRIP).endswith(tuple(_BREAK_AFTER))
        tokens.append((cleaned, match.start(), sentence_start, breaks))
        previous_end, previous_raw = match.end(), raw

    spans: list[Entity] = []
    run: list[tuple[str, int, bool]] = []

    def flush() -> None:
        if len(run) >= 2:
            body = run
            if body[0][2] and len(body) >= 3:
                body = body[1:]
            spans.append(
                Entity(" ".join(part[0] for part in body), "name", body[0][1])
            )
        run.clear()

    for cleaned, start, sentence_start, breaks in tokens:
        if cleaned and _is_capitalised(cleaned):
            run.append((cleaned, start, sentence_start))
            if breaks:
                flush()
        else:
            flush()
    flush()
    return spans


def named_entities(request: str) -> list[Entity]:
    """Every named item of *request*, in the order it is written.

    Proper names (capitalised multi-token spans), quoted strings, uids and email
    addresses. Deduplicated on the normalised form, first spelling kept, capped.
    Nothing here consults a model, a dictionary or the workflow: the same request
    always yields the same list.
    """
    found: list[Entity] = list(_name_spans(request))
    for pattern in (_SQUOTE_RE, _DQUOTE_RE, _CURLY_SQUOTE_RE):
        found.extend(
            Entity(match.group(1).strip(), "quoted", match.start(1))
            for match in pattern.finditer(request)
        )
    found.extend(
        Entity(match.group(0), "uid", match.start()) for match in _UID_RE.finditer(request)
    )
    found.extend(
        Entity(match.group(0), "email", match.start())
        for match in _EMAIL_RE.finditer(request)
    )

    ordered: list[Entity] = []
    seen: set[str] = set()
    for entity in sorted(found, key=lambda item: (item.start, item.kind)):
        key = entity.key
        if not key or key in seen:
            continue
        if not (MIN_ENTITY_CHARS <= len(entity.text) <= MAX_ENTITY_CHARS):
            continue
        seen.add(key)
        ordered.append(entity)
        if len(ordered) >= MAX_ENTITIES:
            break
    return ordered


# ---------------------------------------------------------------------------
# What the run retrieved
# ---------------------------------------------------------------------------

def aliased_executes(trajectory: Mapping[str, Any]) -> set[str]:
    """Every alias the extractor can see on an execute observation.

    The coverage list is only sound when the archive holds all of these: an
    alias the extractor is reading but the archive never kept is text this
    module cannot search, and "appears in no retrieved observation" would then
    be a statement about the archive rather than about the run.
    """
    from fastworkflow.observation_offloading.labels import (
        is_offload_label,
        label_alias,
        printed_alias,
    )

    found: set[str] = set()
    for key, value in trajectory.items():
        name = str(key)
        if not name.startswith("observation_") or not isinstance(value, str):
            continue
        index = name.removeprefix("observation_")
        if not index.isdigit():
            continue
        if str(trajectory.get(f"tool_name_{index}") or "") != "execute_workflow_query":
            continue
        alias = label_alias(value) if is_offload_label(value) else printed_alias(value)
        if alias:
            found.add(alias)
    return found


def retrieved_corpus(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
) -> tuple[str, list[str]]:
    """Every byte this turn retrieved, normalised, as one haystack.

    Two sources, both stores, both read-only:

    (a) the archived text of every execute observation of the turn, with the
        context clause recorded for its alias (``ido-8ps.13``) -- the archive
        keeps the command *response*, never the command, so a name the agent
        merely typed into a query can never make that name look retrieved;
    (b) every stored row behind every result handle the turn declared -- the
        rows ``answer_rehydration`` puts in front of the extractor, whole.

    It is deliberately a superset of what the extract prompt can hold: when the
    rehydration budget leaves an alias as a pointer, its evidence still counts as
    retrieved. The error that matters is telling a writer that something was
    never retrieved when it was, and this cannot make it.

    Returns ``(normalised_haystack, archived_aliases)``.
    """
    from fastworkflow import answer_rehydration

    selected = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()

    parts: list[str] = []
    aliases: list[str] = []
    hot = stored_handles(selected)
    try:
        rows = archive.list(selected)
    except Exception:  # noqa: BLE001 - an unreadable archive is an empty one, never a failed turn
        logger.debug("answer coverage could not list the archive", exc_info=True)
        rows = []
    def take(alias: str, handle: Mapping[str, Any]) -> None:
        # A search answer is archived under ``O5#a1``. It is a MODEL's summary and
        # it repeats the agent's own question, so a name the agent searched for
        # would look retrieved through it. Evidence only.
        if not alias or alias in aliases or is_search_answer_key(alias):
            return
        aliases.append(alias)
        parts.append(strip_alias_line(str(handle.get("text") or "")))
        parts.append(context_clause_of(selected, alias) or "")

    for handle in rows:
        take(str(handle.get("alias") or ""), handle)
    for alias, handle in hot.items():
        take(str(alias), handle or {})

    if handle_store is None:
        try:
            from fastworkflow import result_handles

            handle_store = result_handles.store()
        except Exception:  # noqa: BLE001
            handle_store = None
    if handle_store is not None:
        for alias in aliases:
            try:
                declaration = handle_store.get_declaration(selected, alias)
            except Exception:  # noqa: BLE001
                continue
            if declaration is None or str(declaration.get("parent_alias") or ""):
                continue
            try:
                parts.append(
                    answer_rehydration.stored_rows_block(
                        alias, scope=selected, store=handle_store
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("answer coverage could not read rows for %s", alias,
                             exc_info=True)
    return normalise("\n".join(parts)), aliases


def retrieved_text(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
) -> str:
    """``retrieved_corpus`` without the alias list."""
    return retrieved_corpus(scope=scope, archive=archive, handle_store=handle_store)[0]


def split_by_presence(
    entities: Iterable[Entity], haystack: str
) -> tuple[list[Entity], list[Entity]]:
    """``(observed, unobserved)``: a normalised substring test, nothing more."""
    observed: list[Entity] = []
    unobserved: list[Entity] = []
    for entity in entities:
        (observed if entity.key and entity.key in haystack else unobserved).append(entity)
    return observed, unobserved


# ---------------------------------------------------------------------------
# The statement
# ---------------------------------------------------------------------------

NORMAL_ENDING = "the loop ended normally"
EXHAUSTED_ENDING = "the loop ended at the iteration limit after {steps} steps"
NONE_MARKER = "none"

#: How many bytes of OBSERVED names the block will spell out before it falls
#: back to the general sentence. The observed list is a convenience -- the rule
#: it illustrates ("everything not in the unobserved list was retrieved") is
#: complete without it -- so it must never be the reason a coverage statement
#: grows without bound. Entities come from a regex over one request, so this is
#: a backstop and not a budget.
OBSERVED_LIST_MAX_BYTES = 1024

#: The sentence ido-8ps.24 exists for. Kept as a constant because the post-check
#: and the offline replay both quote it, and a wording that drifts between the
#: instruction and the measure would make the measure meaningless.
RETRIEVED_RULE = (
    "Every other named item of the request WAS retrieved: it appears in this "
    "run's observations and must be reported from them. "
    'Do not write "not retrieved", "not available", "no data", or any other '
    "statement of absence about an item that is not named in the unobserved "
    "list above."
)


def coverage_block(
    *,
    unobserved: Iterable[str],
    exhausted: bool,
    steps: int,
    observed: Iterable[str] = (),
) -> str:
    """The exact text prepended to the extract input.

    One paragraph, no markup, always the same sentences in the same order, so
    two runs of the same turn produce the same block byte for byte.

    ``ido-8ps.24`` is the fourth and fifth sentences. The first version of this
    block named the unobserved items and then said only "For items that appear,
    report only what the observations show", which one D3+D4 attempt read as
    permission to apply "not retrieved" to three identity uids that were in its
    own holder pages. Naming the unobserved set is not the same claim as saying
    what the rest of the set IS, and the block now says it: everything else was
    retrieved, must be reported from the observations, and may not be called
    absent. The observed names are spelled out where they fit, because the
    failure was about specific items and a list is harder to misread than a
    quantifier.
    """
    ending = (
        EXHAUSTED_ENDING.format(steps=int(steps)) if exhausted else NORMAL_ENDING
    )
    names = [str(name).strip() for name in unobserved if str(name).strip()]
    listed = "; ".join(names) if names else NONE_MARKER
    seen: list[str] = []
    for name in observed:
        text = str(name).strip()
        if text and text not in seen:
            seen.append(text)
    shown = "; ".join(seen)
    named = (
        f"These named items of the request DO appear in this run's "
        f"observations: {shown}. "
        if seen and len(shown.encode("utf-8")) <= OBSERVED_LIST_MAX_BYTES
        else ""
    )
    return (
        f"Coverage of this run: {ending}. "
        f"These named items from the request appear in no retrieved observation: "
        f"{listed}. "
        'For each of them report "not retrieved" and nothing else - no value, no '
        "unavailability, no absence. "
        f"{named}"
        f"{RETRIEVED_RULE} "
        "For items that appear, report only what the observations show."
    )


@dataclass
class CoverageReport:
    """What the statement said, and what it cost."""

    flag: bool = True
    exhausted: bool = False
    steps: int = 0
    entities_total: int = 0
    entities_observed: int = 0
    entities_unobserved: int = 0
    observed: list[str] = field(default_factory=list)
    #: The observed items the statement NAMED back (ido-8ps.24) -- the
    #: instructed kinds only, so `observed_named` and `unobserved` partition one
    #: set and the post-check can measure both halves of the same instruction.
    observed_named: list[str] = field(default_factory=list)
    unobserved: list[str] = field(default_factory=list)
    phrases_total: int = 0
    phrases_unmatched: list[str] = field(default_factory=list)
    kinds: dict[str, int] = field(default_factory=dict)
    complete: bool = True
    incomplete_reason: str = ""
    archived_observations: int = 0
    aliased_executes: int = 0
    statement: str = ""
    statement_bytes: int = 0
    haystack_bytes: int = 0
    request_bytes: int = 0

    def as_event(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "exhausted": self.exhausted,
            "steps": self.steps,
            "entities_total": self.entities_total,
            "entities_observed": self.entities_observed,
            "entities_unobserved": self.entities_unobserved,
            "observed": list(self.observed),
            "unobserved": list(self.unobserved),
            "phrases_total": self.phrases_total,
            "phrases_unmatched": list(self.phrases_unmatched),
            "entity_kinds": dict(self.kinds),
            "complete": self.complete,
            "incomplete_reason": self.incomplete_reason,
            "archived_observations": self.archived_observations,
            "aliased_executes": self.aliased_executes,
            "statement_bytes": self.statement_bytes,
            "haystack_bytes": self.haystack_bytes,
            "request_bytes": self.request_bytes,
        }


def _steps(trajectory: Mapping[str, Any]) -> int:
    return sum(1 for key in trajectory if str(key).startswith("tool_name_"))


def build_statement(
    trajectory: Mapping[str, Any],
    *,
    user_query: Any,
    exhausted: bool,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
    haystack: Optional[str] = None,
) -> tuple[dict[str, Any], CoverageReport]:
    """``(trajectory_copy_with_the_block_first, report)``.

    The input mapping is never mutated and no observation in it is changed: the
    copy holds one extra key, and it holds it FIRST, because ``_format_trajectory``
    renders the keys in order and a coverage rule read after 60 KB of evidence is
    a rule read too late.
    """
    request = request_text(user_query)
    entities = named_entities(request)
    if haystack is not None:
        text, archived = haystack, []
    else:
        text, archived = retrieved_corpus(
            scope=scope, archive=archive, handle_store=handle_store
        )
    observed, unobserved = split_by_presence(entities, text)
    instructed = [e.text for e in unobserved if e.kind in INSTRUCTED_KINDS]
    phrases = [e.text for e in unobserved if e.kind not in INSTRUCTED_KINDS]
    steps = _steps(trajectory)

    # The list is a claim about the RUN; it is only sound when the archive holds
    # every observation the extractor is reading. An alias on the extractor's
    # trajectory that the archive never kept is text this module cannot search,
    # and naming its contents "not retrieved" would be a statement about the
    # archive. When that happens the run still gets the exhaustion sentence --
    # which needs no archive -- and names nothing.
    seen_aliases = aliased_executes(trajectory)
    missing = sorted(seen_aliases - set(archived)) if haystack is None else []
    complete = True
    reason = ""
    if haystack is None and not archived:
        complete, reason = False, "no archived observations for this scope"
    elif missing:
        complete = False
        reason = "archive is missing %d of %d observations (%s)" % (
            len(missing), len(seen_aliases), ", ".join(missing[:8]),
        )
    if not complete:
        logger.warning("answer coverage names no items: %s", reason)
        instructed = []
        phrases = []
    statement = coverage_block(
        unobserved=instructed,
        exhausted=bool(exhausted),
        steps=steps,
        # ido-8ps.24: only the kinds that are ever INSTRUCTED as "not retrieved"
        # are listed back as retrieved, so the two lists partition one set. A
        # quoted request phrase is measured and never instructed, and naming one
        # here would turn a measure into an instruction by the back door.
        observed=[
            entity.text for entity in observed if entity.kind in INSTRUCTED_KINDS
        ],
    )

    kinds: dict[str, int] = {}
    for entity in entities:
        kinds[entity.kind] = kinds.get(entity.kind, 0) + 1

    report = CoverageReport(
        exhausted=bool(exhausted),
        steps=steps,
        entities_total=len(entities),
        entities_observed=len(observed),
        entities_unobserved=len(instructed),
        observed=[entity.text for entity in observed],
        observed_named=[
            entity.text for entity in observed if entity.kind in INSTRUCTED_KINDS
        ],
        unobserved=instructed,
        phrases_total=sum(
            1 for entity in entities if entity.kind not in INSTRUCTED_KINDS
        ),
        phrases_unmatched=phrases,
        kinds=kinds,
        complete=complete,
        incomplete_reason=reason,
        archived_observations=len(archived),
        aliased_executes=len(seen_aliases),
        statement=statement,
        statement_bytes=len(statement.encode("utf-8")),
        haystack_bytes=len(text.encode("utf-8")),
        request_bytes=len(request.encode("utf-8")),
    )
    copy: dict[str, Any] = {COVERAGE_KEY: statement}
    copy.update(trajectory)
    return copy, report


# ---------------------------------------------------------------------------
# The post-check (measurement only)
# ---------------------------------------------------------------------------

#: Phrasings that assert something about the WORLD rather than about this run.
#: A curated list, not a classifier: it is read only to count, never to gate, so
#: a phrasing it misses costs a measure and never an answer.
_UNAVAILABILITY_RE = re.compile(
    r"(?:"
    r"not (?:available|present|provided|found|listed|returned|disclosed|exposed"
    r"|accessible|surfaced|supported|defined|configured|in the)"
    r"|unavailable|no longer available"
    r"|does not (?:provide|expose|disclose|return|list|contain|support|have|exist)"
    r"|do not (?:provide|expose|disclose|return|list|contain|support|have|exist)"
    r"|(?:cannot|can't|could not|couldn't) be (?:determined|retrieved|found|located"
    r"|identified|established|resolved|obtained)"
    r"|(?:is|are|was|were) absent|absent from"
    r"|no (?:such|record|finding|entry|data|value|information|match|result)"
    r"|none (?:available|found|present|exist)"
    r"|there (?:is|are|was|were) no"
    r"|has no|have no"
    r"|not exist"
    r")"
)
_NOT_RETRIEVED_RE = re.compile(r"not retrieved")
#: How far from a name a claim still counts as being about it. One long sentence.
CLAIM_WINDOW_CHARS = 200


def _windows(haystack: str, needle: str) -> list[str]:
    out: list[str] = []
    start = 0
    while True:
        at = haystack.find(needle, start)
        if at < 0:
            return out
        out.append(
            haystack[max(0, at - CLAIM_WINDOW_CHARS): at + len(needle) + CLAIM_WINDOW_CHARS]
        )
        start = at + max(1, len(needle))


@dataclass
class PostCheck:
    """What the finished answer said about the items the statement named.

    Two directions, and ``ido-8ps.24`` is the second one. ``*_on_unobserved``
    counts the items the run really never retrieved; ``*_on_observed`` counts
    the opposite and worse error, an absence claim about an item that IS in the
    run's own observations. Both are counts, never gates.
    """

    answer_bytes: int = 0
    unobserved_total: int = 0
    unobserved_mentioned: int = 0
    unavailability_claim_on_unobserved: int = 0
    not_retrieved_on_unobserved: int = 0
    silent_on_unobserved: int = 0
    observed_total: int = 0
    observed_mentioned: int = 0
    unavailability_claim_on_observed: int = 0
    not_retrieved_on_observed: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    observed_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def misuse_on_observed(self) -> int:
        """The ido-8ps.24 number: absence phrasing about a retrieved item."""
        return (
            self.unavailability_claim_on_observed + self.not_retrieved_on_observed
        )

    def as_event(self) -> dict[str, Any]:
        return {
            "answer_bytes": self.answer_bytes,
            "unobserved_total": self.unobserved_total,
            "unobserved_mentioned": self.unobserved_mentioned,
            "unavailability_claim_on_unobserved": self.unavailability_claim_on_unobserved,
            "not_retrieved_on_unobserved": self.not_retrieved_on_unobserved,
            "silent_on_unobserved": self.silent_on_unobserved,
            "observed_total": self.observed_total,
            "observed_mentioned": self.observed_mentioned,
            "unavailability_claim_on_observed": self.unavailability_claim_on_observed,
            "not_retrieved_on_observed": self.not_retrieved_on_observed,
            "misuse_on_observed": self.misuse_on_observed,
            "per_item": list(self.details),
            "per_observed_item": list(self.observed_details),
        }


def post_check(
    answer: Any,
    unobserved: Iterable[str],
    observed: Iterable[str] = (),
) -> PostCheck:
    """Count absence phrasing near each named item. Measurement only.

    Nothing here changes the answer, retries the call, or fails the turn. It
    exists so ``ido-8ps.22`` can be scored by a number rather than by reading
    five answers by hand, and so a later run can be compared to this one.
    ``observed`` is ``ido-8ps.24``: the same window test applied to the items
    the statement said WERE retrieved, where any hit at all is a defect.
    """
    text = normalise(answer)
    check = PostCheck(answer_bytes=len(str(answer or "").encode("utf-8")))
    for name in unobserved:
        key = normalise(name)
        check.unobserved_total += 1
        if not key:
            continue
        windows = _windows(text, key)
        if not windows:
            check.silent_on_unobserved += 1
            check.details.append({"item": name, "mentioned": False,
                                  "unavailability_claims": 0, "not_retrieved": 0})
            continue
        check.unobserved_mentioned += 1
        claims = sum(1 for window in windows if _UNAVAILABILITY_RE.search(window))
        marked = sum(1 for window in windows if _NOT_RETRIEVED_RE.search(window))
        check.unavailability_claim_on_unobserved += claims
        check.not_retrieved_on_unobserved += marked
        check.details.append({"item": name, "mentioned": True,
                              "unavailability_claims": claims, "not_retrieved": marked})
    for name in observed:
        key = normalise(name)
        check.observed_total += 1
        if not key:
            continue
        windows = _windows(text, key)
        if not windows:
            check.observed_details.append({"item": name, "mentioned": False,
                                           "unavailability_claims": 0,
                                           "not_retrieved": 0})
            continue
        check.observed_mentioned += 1
        claims = sum(1 for window in windows if _UNAVAILABILITY_RE.search(window))
        marked = sum(1 for window in windows if _NOT_RETRIEVED_RE.search(window))
        check.unavailability_claim_on_observed += claims
        check.not_retrieved_on_observed += marked
        check.observed_details.append({"item": name, "mentioned": True,
                                       "unavailability_claims": claims,
                                       "not_retrieved": marked})
    return check


__all__ = [
    "ANSWER_COVERAGE_ENV",
    "COVERAGE_KEY",
    "CLAIM_WINDOW_CHARS",
    "CoverageReport",
    "OBSERVED_LIST_MAX_BYTES",
    "RETRIEVED_RULE",
    "Entity",
    "INSTRUCTED_KINDS",
    "MAX_ENTITIES",
    "MAX_ENTITY_CHARS",
    "MIN_ENTITY_CHARS",
    "PLAN_MARKER",
    "PostCheck",
    "aliased_executes",
    "answer_coverage_enabled",
    "build_statement",
    "coverage_block",
    "named_entities",
    "normalise",
    "post_check",
    "request_text",
    "retrieved_corpus",
    "retrieved_text",
    "split_by_presence",
]
