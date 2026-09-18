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

``ido-8ps.28`` is one further sentence, added after the
observed-items rule: for each named item of the request this run made the
SUBJECT of a command, the other named items of the request that appear in that
subject's own observations. Positive half only -- what a subject's evidence
lacks is never stated, because a bounded portrait would make that a false
absence -- and no conclusion is instructed. It answers the failure the
attribution replay measured: the run retrieves the right rows for a person and
the answer still credits that person with a property their own rows do not
carry.

Unconditional since ``ido-pyw.1``: the coverage statement, the roster nudge and
the evidence sentence are what an answer-time extract call gets, for every
workflow. See ``docs/answer_coverage.md``.
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
# Normalisation
# ---------------------------------------------------------------------------

#: ``ido-c7m``/F36. A zero-width space is a break opportunity -- the character
#: is literally named a space -- so it folds to one rather than being removed.
#: Deleting it would turn ``Alan<ZWSP>Cooper`` into ``alancooper`` and leave
#: the request's ``Alan Cooper`` unobserved, which is the very false absence
#: this is about. The cost is the other reading: a zero-width space INSIDE a
#: token splits it, so ``Al<ZWSP>an Cooper`` still does not match. That reading
#: is what the soft hyphen is for and what every other invisible character here
#: gets, and a fold to a space can only split a run, never join two -- which is
#: the side a presence test may err on.
_IGNORABLE_TO_SPACE = "\u200b"


def _strip_format_characters(text: str) -> str:
    """Drop the format (``Cf``) characters, folding a zero-width space to a space.

    ``ido-c7m``/F36. Soft hyphen (U+00AD), byte-order mark (U+FEFF), the zero-
    width joiner and non-joiner, the word joiner and the bidi controls are
    default-ignorable: they render as nothing, so a row carrying one inside a
    name prints exactly the name the request wrote, while the request's
    spelling reads as unobserved and the coverage block instructs a false
    absence. They are removed here, BEFORE the compatibility fold, so a fold
    that spans one still composes.

    Removal is by Unicode category, not by the list of characters one review
    happened to try: any other ``Cf`` character is invisible for the same
    reason. It cannot make two visibly different names collide -- every
    character it removes renders as nothing -- and the one character that could
    have joined two visible runs into a third spelling is folded to a space
    instead.
    """
    if not any(unicodedata.category(ch) == "Cf" for ch in text):
        return text
    return "".join(
        " " if ch in _IGNORABLE_TO_SPACE
        else "" if unicodedata.category(ch) == "Cf"
        else ch
        for ch in text
    )


def normalise(text: Any) -> str:
    """NFKC, casefolded, whitespace collapsed -- the one comparison form.

    Presence is decided on this form and nothing else, so a name broken across a
    line in a rendered row still matches the name written on one line in the
    request, and a full-width or ligature variant matches its plain spelling.
    Invisible format characters go first (:func:`_strip_format_characters`), so
    a soft hyphen or a byte-order mark inside a name in a row cannot hide that
    name from the presence test.
    """
    stripped = _strip_format_characters(str(text or ""))
    folded = unicodedata.normalize("NFKC", stripped).casefold()
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
#: ``ido-jf6``/F29. These separate two named items with no space around them, so
#: ``Alan Cooper/Brandon Miller`` is two people and ``Anna Garcia—contractor``
#: is a person and a word. They never belong to a token and always end a run.
_SEPARATORS = "/—–"
_TOKEN_RE = re.compile(r"[^\s/—–]+")
#: Taken off the end of a token BEFORE the possessive test. The apostrophes are
#: deliberately absent: a trailing apostrophe is stripped as a closing quote,
#: which would make a plural possessive invisible to that test.
_TRAIL_BEFORE_POSSESSIVE = "\"“”)]}" + _BREAK_AFTER
#: ``ido-jf6``/F29. "Cooper's" is a name plus a grammatical marker, not a
#: two-word name, and the marker ENDS the name: "Alan Cooper's Active Directory
#: rights" is Alan Cooper AND Active Directory, never one item. Straight and
#: curly apostrophes, and the upper-case spelling of a shouted request.
_POSSESSIVE_SUFFIXES = ("'s", "’s", "'S", "’S")
#: ``ido-jf6``/F29. A sentence that OPENS with one of these opens with a verb,
#: so the capital is grammar and not a name: "List Identities whose manager
#: left" and "Compare Alan and Brandon" name nobody. The list is deliberately
#: small, explicit, and holds only words that are not also ordinary given names
#: or surnames -- no "Mark", "Bill", "Grant", "Will", "Rose", "May" -- because
#: dropping a real first name would cost the run a handle, the more expensive
#: mistake of the two. Only the FIRST token of a sentence is tested against it,
#: so "Alan Cooper met Barbara List" is untouched.
_IMPERATIVE_VERBS = frozenset({
    "audit", "check", "compare", "count", "describe", "display", "explain",
    "fetch", "find", "get", "identify", "list", "report", "retrieve", "review",
    "search", "show", "summarise", "summarize", "tell", "verify",
})


def _clean_token(raw: str) -> tuple[str, bool]:
    """The token itself, and whether it carried a possessive marker.

    The possessive comes off before the capitalisation test, because "Cooper's"
    ends in a letter and would otherwise read as an ordinary capitalised token
    and join its run to whatever capitalised word follows it.
    """
    trimmed = raw.lstrip(_LEAD_STRIP).rstrip(_TRAIL_BEFORE_POSSESSIVE)
    possessive = False
    for suffix in _POSSESSIVE_SUFFIXES:
        if len(trimmed) > len(suffix) and trimmed.endswith(suffix):
            trimmed, possessive = trimmed[: -len(suffix)], True
            break
    else:
        # Plural possessive: a bare apostrophe after an s ("the Hendersons'
        # rights"). Only that shape, so a closing quote on any other word stays
        # an ordinary closing quote.
        if len(trimmed) > 2 and trimmed[-1] in "'’" and trimmed[-2] in "sS":
            trimmed, possessive = trimmed[:-1], True
    cleaned = trimmed.strip(_LEAD_STRIP).rstrip(_TRAIL_STRIP + _BREAK_AFTER)
    return cleaned.strip(_LEAD_STRIP), possessive


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

    A run also ends at a possessive marker and at a ``/`` or a dash, and a run
    that opens a sentence with an imperative verb loses that verb however short
    the run is -- ``ido-jf6``/F29, where "List Alan Cooper's Active Directory
    rights" came out as one named item that appears in no observation ever
    retrieved, and the coverage block then instructed a false absence about the
    person the run had in fact opened.
    """
    tokens: list[tuple[str, int, bool, bool]] = []  # cleaned, start, sentence_start, breaks
    previous_end = 0
    previous_raw = ""
    matches = list(_TOKEN_RE.finditer(text))
    for index, match in enumerate(matches):
        raw = match.group(0)
        gap = text[previous_end:match.start()]
        sentence_start = (
            not tokens
            or "\n" in gap
            or previous_raw.rstrip(_TRAIL_STRIP).endswith(tuple(_SENTENCE_END))
        )
        cleaned, possessive = _clean_token(raw)
        following = (
            text[match.end():matches[index + 1].start()]
            if index + 1 < len(matches)
            else text[match.end():]
        )
        breaks = (
            possessive
            or raw.rstrip(_TRAIL_STRIP).endswith(tuple(_BREAK_AFTER))
            or any(char in _SEPARATORS for char in following)
        )
        tokens.append((cleaned, match.start(), sentence_start, breaks))
        previous_end, previous_raw = match.end(), raw

    spans: list[Entity] = []
    run: list[tuple[str, int, bool]] = []

    def flush() -> None:
        body = run
        if body and body[0][2] and (
            len(body) >= 3 or body[0][0].casefold() in _IMPERATIVE_VERBS
        ):
            body = body[1:]
        if len(body) >= 2:
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


def issued_commands(trajectory: Mapping[str, Any]) -> str:
    """Every command this run ISSUED, normalised, as one haystack.

    ``ido-8ps.27`` (b). The archive keeps command RESPONSES and never commands,
    which is what makes ``retrieved_corpus`` sound. This is the deliberate other
    half, read for one purpose only: an item that is in no response but IS in a
    command was asked for and did not come back, and that is a different fact
    from an item nobody asked for. It can never make an item look retrieved --
    it is consulted only about items ``split_by_presence`` already put in the
    unobserved list.

    The trajectory handed to ``build_statement`` is the extractor's copy, so a
    step the context-window fallback truncated is gone from it. A truncated
    attempt therefore reads as "never attempted", which is the wording this
    module already used before ``ido-8ps.27`` and never a new claim.
    """
    parts: list[str] = []
    for key, value in trajectory.items():
        name = str(key)
        if not name.startswith("tool_args_"):
            continue
        if isinstance(value, Mapping):
            parts.extend(str(item) for item in value.values())
        else:
            parts.append(str(value))
    return normalise("\n".join(parts))


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
        merely typed into a query can never make that name look retrieved. A
        response may QUOTE the command, though: a filtered listing prints its
        own filter literal in its header and a backend may quote the literal it
        did not match, so the contract is finished by
        :func:`strip_query_echoes`, which ``split_by_presence`` applies before
        it decides presence (``ido-mng``), and by
        :func:`drop_zero_match_echo`, which is applied HERE, per alias, for the
        echoes no wording rule can recognise: what a page retrieved is a fact
        its own record already holds (``ido-3f8``);
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
    # Where each alias's OWN archived text sits in ``parts``. The haystack is
    # still one string; this is only so the declaration pass below can correct
    # the text of the alias it is reading (``ido-3f8``) without disturbing the
    # order anything else here produces.
    text_at: dict[str, int] = {}
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
        text_at[alias] = len(parts)
        parts.append(strip_alias_line(str(handle.get("text") or "")))
        parts.append(
            context_clause_of(selected, alias, selected_archive=archive) or "")

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
            if declaration is None:
                continue
            if str(declaration.get("parent_alias") or ""):
                # A page observation of a listing declared elsewhere. Its rows
                # are read under that listing, so there is nothing to add here
                # -- but if it is a FILTERED page that carried no rows, its own
                # text quotes a literal it never retrieved, and that echo comes
                # out (``ido-3f8``).
                index = text_at.get(alias)
                if index is not None:
                    parts[index] = drop_zero_match_echo(parts[index], declaration)
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


def subject_corpus(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
) -> str:
    """The CONTEXT CLAUSES of this turn's observations, normalised, as one haystack.

    ``ido-8ps.27``. ``retrieved_corpus`` answers "does this name appear anywhere
    in what the run retrieved". The roster diagnosis
    (``evaluation/artifacts/result-search/roster-diagnosis.md``) showed that is
    the wrong question for a finish-time check: ``find_identity <name>`` puts a
    name in the corpus, so three attempts that audited ONE person and closed four
    others as "not retrieved" reported zero or one unobserved names.

    The clause answers the right question. ``ido-8ps.13`` records, per archived
    observation, the context INSTANCE the command ran against -- "Identity
    28c5aeb5... Alan Cooper", "Account e8a0c3a1... Alan Cooper". A name in a
    clause is a name the run made the SUBJECT of a command; a name only in
    observation text is a row in somebody else's listing. Replayed over the ten
    stored pinned attempts the clause test fired on 7 of 7 voluntary early stops
    and on neither attempt that completed the roster.

    Clauses only. No observation text, no stored rows: this haystack may never
    be the reason a name is called retrieved, and it never is -- it is read only
    to decide whether the agent has yet turned to that subject.
    """
    selected = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()
    try:
        rows = archive.list(selected)
    except Exception:  # noqa: BLE001 - an unreadable archive is an empty one
        logger.debug("roster nudge could not list the archive", exc_info=True)
        rows = []
    aliases: list[str] = []
    for handle in rows:
        alias = str(handle.get("alias") or "")
        if alias and alias not in aliases and not is_search_answer_key(alias):
            aliases.append(alias)
    for alias in stored_handles(selected):
        if alias and alias not in aliases and not is_search_answer_key(alias):
            aliases.append(alias)
    parts = [
        context_clause_of(selected, alias, selected_archive=archive) or ""
        for alias in aliases
    ]
    return normalise("\n".join(parts))


def evidence_by_subject(
    entities: Iterable[Entity],
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
    observations: Optional[Iterable[Any]] = None,
) -> list[tuple[Entity, list[Entity]]]:
    """``ido-8ps.28``: per subject, the other *entities* its own observations contain.

    One line of work, because the reader already exists.
    ``answer_attribution.observations`` is ``retrieved_corpus``'s three readers
    kept PER ALIAS instead of concatenated -- archived observation text, stored
    handle pages, and the ``ido-8ps.13`` context clause -- which is exactly what
    "in that subject's own observations" needs and what one haystack cannot
    give. A third reader would be a third thing to keep true.

    Imported inside the function: ``answer_attribution`` imports this module, and
    the dependency only ever runs one way at import time.
    """
    from fastworkflow import answer_attribution

    found = (
        observations if observations is not None
        else answer_attribution.observations(
            scope=scope, archive=archive, handle_store=handle_store
        )
    )
    return answer_attribution.subject_evidence(entities, found)


#: ``ido-mng``. A rendered result page states the query in its own header --
#: ``result_handle=O2 filter="Christopher Hubbard" page 1 rows 0 of 12 ...`` --
#: and a backend that finds nothing may say so in the same words the agent typed
#: ("No identity matching 'Christopher Hubbard' was found."). Both are the
#: agent's OWN text quoted back, never a retrieved row, and both are in the
#: archived response, which is why ``retrieved_corpus``'s "the archive keeps the
#: command response, never the command" was not by itself enough to keep the
#: contract. A filtered miss that echoes its literal would otherwise make the
#: typed name look retrieved and leave the writer instructed AGAINST the one
#: true statement -- that the filtered listing returned nothing.
_FILTER_ECHO_RE = re.compile(r"""filter=(?:"[^"\n]*"|'[^'\n]*')""", re.IGNORECASE)

#: A miss, however the backend words it: "no rows matched", "no identity
#: matching", "no accounts were found", "0 results returned".
_MISS_CUE = (
    r"no(?:t)?\s+(?:[\w'\u2019\-]+\s+){0,4}?"
    r"(?:match(?:ed|es|ing)?|found|returned|exist(?:s|ed)?)"
)

#: The literal a miss quotes back, and ONLY that literal: the cue, a short gap
#: that crosses no sentence boundary, then one quoted span. Nothing else on the
#: line is touched, because removing more than the echo is how a haystack starts
#: reporting that something retrieved was never retrieved, which is the error
#: this module exists to prevent.
#: The quote pairs a backend may use. Typographic quotes survive NFKC, so a
#: miss written with them echoes just as loudly as one written with ASCII.
_QUOTED_SPAN = (
    r"\"[^\"\n]*\"|'[^'\n]*'|\u201c[^\u201d\n]*\u201d|\u2018[^\u2019\n]*\u2019"
)

_MISS_ECHO_RE = re.compile(
    _MISS_CUE + r"(?:[^\"'\u201c\u2018\n.;]{0,24}?)(" + _QUOTED_SPAN + r")",
    re.IGNORECASE,
)


def strip_query_echoes(text: str) -> str:
    """*text* without the run's own query quoted back at it (``ido-mng``).

    Two spans go, both of them the agent's typed literal and neither of them a
    retrieved value: the ``filter="..."`` echo a result page prints in its
    header, and the quoted literal a "nothing matched" sentence repeats. What is
    removed is the echo itself, never the line around it: a page that DID match
    rows still carries the name in its rows, so this cannot turn a retrieved
    name into an absent one. Idempotent, and safe on text whose line structure
    normalisation has already collapsed, because neither pattern spans a line.
    """
    out = _FILTER_ECHO_RE.sub(" ", str(text or ""))
    return _MISS_ECHO_RE.sub(lambda match: match.group(0)[:match.start(1) - match.start(0)], out)


def drop_zero_match_echo(text: str, declaration: Optional[Mapping[str, Any]]) -> str:
    """*text* without the filter literal of a page that carried no rows (``ido-3f8``).

    :func:`strip_query_echoes` is textual: it knows the two shapes the framework
    itself writes and the one a miss quotes. It cannot know the shape a BACKEND
    writes -- "No identity matching Christopher Hubbard was found" quotes
    nothing -- and widening it to unquoted spans was refused on purpose, because
    normalisation collapses newlines and a removal bounded by a sentence end
    could swallow real rows and cause the very false absence this module exists
    to prevent.

    This is the structural half, and it needs no wording at all. The page layer
    already records, per page observation, that a filtered page carried no rows
    (``result_handles.page_matched_nothing``). A page that carried no rows
    retrieved nothing, so EVERY occurrence of its filter literal in its own
    observation is the run's own query quoted back -- by the header, by the
    framework's zero message, by whatever sentence the backend chose -- and
    removing the literal there cannot remove a retrieved row, because that
    observation has none.

    Two bounds keep it honest. The removal is confined to the ONE observation
    the marker is about, so a name retrieved in some other observation, or on a
    page of the same listing that did match, is untouched and stays observed.
    And the literal is the one the store proves that page ran, not any name the
    text mentions: an unfiltered page, a page that matched rows, and an alias
    with no declaration are all returned unchanged.
    """
    try:
        from fastworkflow import result_handles

        literal = result_handles.echoed_literal(declaration, text)
    except Exception:  # noqa: BLE001 - an unreadable marker is no marker
        logger.debug("answer coverage could not read a page's filter", exc_info=True)
        return text
    if not literal:
        return text
    hunted = normalise(literal)
    return normalise(text).replace(hunted, " ") if hunted else text


def split_by_presence(
    entities: Iterable[Entity], haystack: str
) -> tuple[list[Entity], list[Entity]]:
    """``(observed, unobserved)``: a normalised substring test, nothing more.

    Nothing more, on a haystack of what the run RETRIEVED: the run's own query
    echoes are stripped first (:func:`strip_query_echoes`), so a name the agent
    merely typed into a filter can never come back as observed even when the
    backend quotes it in a header or a "nothing matched" sentence (``ido-mng``).
    """
    hunted = strip_query_echoes(haystack)
    observed: list[Entity] = []
    unobserved: list[Entity] = []
    for entity in entities:
        (observed if entity.key and entity.key in hunted else unobserved).append(entity)
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


#: ``ido-8ps.28``. How many bytes of EVIDENCE the block will spell out. Its own
#: constant, sized like ``OBSERVED_LIST_MAX_BYTES`` and for the same reason: the
#: sentence is built from a regex over one request crossed with the subjects
#: that request named, so this is a backstop and not a budget. Whole subjects are
#: taken in order until the cap is reached and the rest are COUNTED, never
#: silently dropped -- a half-written subject would read as a short list for that
#: subject, and a short list is how a positive statement turns into an absence.
EVIDENCE_LIST_MAX_BYTES = 1024

#: The ``ido-8ps.28`` sentence. It states evidence back and instructs nothing.
#:
#: The failure it answers is not retrieval and not coverage: the run retrieves
#: the right rows for a person and the answer still credits that person with a
#: property their own rows do not carry, copied off the request's premise. The
#: attribution replay
#: (``evaluation/artifacts/result-search/attribution-replay.md``) measured that
#: after the fact and reached 6 of the 9 rows, missing every answer that asserts
#: the property without naming it. Asked FORWARD, of the evidence alone, the same
#: question has no such blind spot: the writer is told which named items of the
#: request each subject's own observations contain before it writes anything.
#:
#: POSITIVE HALF ONLY, and this is the whole of the ``ido-8ps.22`` lesson. What a
#: subject's evidence LACKS is never stated, because a bounded portrait or an
#: unpaged listing makes "not in the evidence" a claim about the run dressed as a
#: claim about the world -- exactly the false absence ``ido-8ps.24`` repaired. So
#: a subject whose evidence contains none of the other items is not listed with
#: an empty list; it is not listed at all.
#:
#: And no conclusion is instructed (the ``ido-8ps.24`` pattern). The sentence
#: does not say to drop a claim, to prefer the evidence, or to check anything. It
#: says what the observations of each subject contain, and stops.
EVIDENCE_HEAD = (
    "Evidence by subject - for each named item of the request that this run "
    "made the subject of a command, the OTHER named items of the request that "
    "appear in that subject's own observations: "
)
EVIDENCE_MORE = "; and {count} more subjects"


def evidence_sentence(
    evidence: Iterable[tuple[str, Iterable[str]]]
) -> tuple[str, int, int]:
    """``(text, subjects_listed, items_listed)``; ``("", 0, 0)`` when silent.

    Deterministic: the same subjects with the same items produce the same bytes.
    A subject with no items is dropped here rather than printed empty -- see
    ``EVIDENCE_HEAD`` for why that is the one rule this sentence cannot bend.
    """
    wanted: list[tuple[str, list[str]]] = []
    for subject, items in evidence:
        name = str(subject).strip()
        found = [str(item).strip() for item in items if str(item).strip()]
        if name and found:
            wanted.append((name, found))
    if not wanted:
        return "", 0, 0
    taken: list[str] = []
    counted = 0
    for name, found in wanted:
        entry = f"{name}: {', '.join(found)}"
        dropped = len(wanted) - len(taken) - 1
        suffix = EVIDENCE_MORE.format(count=dropped) if dropped else ""
        candidate = "; ".join([*taken, entry]) + suffix
        if len(candidate.encode("utf-8")) > EVIDENCE_LIST_MAX_BYTES:
            break
        taken.append(entry)
        counted += len(found)
    if not taken:
        return "", 0, 0
    listed = "; ".join(taken)
    if len(taken) < len(wanted):
        listed += EVIDENCE_MORE.format(count=len(wanted) - len(taken))
    return f"{EVIDENCE_HEAD}{listed}. ", len(taken), counted


#: ``ido-8ps.27`` (b). The two reasons an item can be missing from the corpus,
#: as one deterministic sentence. It is emitted only when the second kind
#: actually occurred, so a run where every missing item was simply never
#: attempted -- seven of the eight in the roster diagnosis -- gets the block it
#: got at ``9e5e9d9``, byte for byte.
ATTEMPT_SPLIT = (
    "Of those, these WERE attempted and the attempt returned nothing about "
    "them: {unavailable}. The rest were never attempted - no command of this "
    "run named them: {never}. "
)


def coverage_block(
    *,
    unobserved: Iterable[str],
    exhausted: bool,
    steps: int,
    observed: Iterable[str] = (),
    unavailable: Iterable[str] = (),
    evidence: Iterable[tuple[str, Iterable[str]]] = (),
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

    ``ido-8ps.27`` is ``unavailable``: the subset of ``unobserved`` the run DID
    name in a command. An item that was asked for and did not come back is a
    different fact about the run from an item nobody ever asked for, the
    extractor is now told which is which, and ``post_check`` counts the two
    separately. The sentence appears only when the second kind exists, so the
    common case is unchanged text.

    ``ido-8ps.28`` is ``evidence``: one further sentence, after the rule about
    the observed items, saying per subject which OTHER named items of the
    request that subject's own observations contain. Positive half only, no
    conclusion instructed, and emitted only when at least one subject has at
    least one item -- so an empty ``evidence`` leaves this block byte for byte
    what it was at ``90a1565``.
    """
    ending = (
        EXHAUSTED_ENDING.format(steps=int(steps)) if exhausted else NORMAL_ENDING
    )
    names = [str(name).strip() for name in unobserved if str(name).strip()]
    listed = "; ".join(names) if names else NONE_MARKER
    tried = [str(name).strip() for name in unavailable if str(name).strip()]
    tried = [name for name in tried if name in names]
    split = ""
    if tried:
        untried = [name for name in names if name not in tried]
        split = ATTEMPT_SPLIT.format(
            unavailable="; ".join(tried),
            never="; ".join(untried) if untried else NONE_MARKER,
        )
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
    stated, _, _ = evidence_sentence(evidence)
    return (
        f"Coverage of this run: {ending}. "
        f"These named items from the request appear in no retrieved observation: "
        f"{listed}. "
        f"{split}"
        'For each of them report "not retrieved" and nothing else - no value, no '
        "unavailability, no absence. "
        f"{named}"
        f"{RETRIEVED_RULE} "
        f"{stated}"
        "For items that appear, report only what the observations show."
    )


# ---------------------------------------------------------------------------
# The roster nudge (ido-8ps.27)
# ---------------------------------------------------------------------------

#: The nudge is bounded like the observed list and for the same reason: it is
#: built from a regex over one request, so this is a backstop, not a budget. A
#: list that does not fit is cut and counted, never dropped silently.
NUDGE_MAX_BYTES = 1024

#: The nudge costs one iteration and is worthless unless the agent can act on
#: it. Below this many further actions the loop ends as it always did.
NUDGE_MIN_ITERS_LEFT = 2

NUDGE_HEAD = (
    "Harness check before this turn ends. You selected finish, and this run "
    "has not made the following named items of the request the subject of any "
    "command, so nothing about them has been retrieved: "
)
NUDGE_TAIL = (
    ". You have {left} more actions available before this turn ends - the "
    "budget is not spent. Retrieve for each of them what the request asks, or "
    "select finish again and say in your final answer why you could not. This "
    "note is from the harness, not from the user: do not ask the user about "
    "it, and it is shown once per turn."
)
NUDGE_MORE = "; and {count} more"


def nudge_block(names: Iterable[str], iterations_left: int) -> tuple[str, int]:
    """``(text, named)`` -- the nudge, capped at ``NUDGE_MAX_BYTES``.

    Deterministic: the same missing set and the same budget produce the same
    bytes. Names are taken in order until the cap is reached and the remainder
    is counted, so a request naming forty people still yields one bounded note.
    """
    wanted = [str(name).strip() for name in names if str(name).strip()]
    if not wanted:
        return "", 0
    tail = NUDGE_TAIL.format(left=max(0, int(iterations_left)))
    frame = len(NUDGE_HEAD.encode("utf-8")) + len(tail.encode("utf-8"))
    room = NUDGE_MAX_BYTES - frame
    taken: list[str] = []
    for name in wanted:
        candidate = "; ".join([*taken, name])
        dropped = len(wanted) - len(taken) - 1
        suffix = NUDGE_MORE.format(count=dropped) if dropped else ""
        if len(candidate.encode("utf-8")) + len(suffix.encode("utf-8")) > room:
            break
        taken.append(name)
    if not taken:
        return "", 0
    listed = "; ".join(taken)
    if len(taken) < len(wanted):
        listed += NUDGE_MORE.format(count=len(wanted) - len(taken))
    return f"{NUDGE_HEAD}{listed}{tail}", len(taken)


@dataclass
class NudgeReport:
    """What the finish-time check saw, whether or not it fired."""

    fired: bool = False
    reason: str = ""
    entities_total: int = 0
    subjects_total: int = 0
    subjects_missing: list[str] = field(default_factory=list)
    subjects_named: int = 0
    iterations_left: int = 0
    text_bytes: int = 0
    clause_bytes: int = 0

    def as_event(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "reason": self.reason,
            "entities_total": self.entities_total,
            "subjects_total": self.subjects_total,
            "subjects_missing": list(self.subjects_missing),
            "subjects_named": self.subjects_named,
            "iterations_left": self.iterations_left,
            "text_bytes": self.text_bytes,
            "clause_bytes": self.clause_bytes,
        }


def build_nudge(
    *,
    user_query: Any,
    iterations_left: int,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    clauses: Optional[str] = None,
) -> tuple[str, NudgeReport]:
    """``(text, report)``. ``text`` is ``""`` when nothing should be injected.

    The whole decision, in one deterministic place, so the loop's own code is a
    call and a branch. Nothing here calls a model, reads a backend or consults
    the user; it reads the request by the same regex ``build_statement`` uses
    and the same context clauses the archive already holds.
    """
    report = NudgeReport(iterations_left=int(iterations_left))
    if int(iterations_left) < NUDGE_MIN_ITERS_LEFT:
        report.reason = "no room to act"
        return "", report
    entities = named_entities(request_text(user_query))
    instructed = [e for e in entities if e.kind in INSTRUCTED_KINDS]
    report.entities_total = len(entities)
    report.subjects_total = len(instructed)
    if not instructed:
        report.reason = "the request names no items"
        return "", report
    haystack = (
        clauses if clauses is not None
        else subject_corpus(scope=scope, archive=archive)
    )
    report.clause_bytes = len(haystack.encode("utf-8"))
    if not haystack:
        # No clause was recorded at all, so "never the subject of a command" is
        # a statement about the archive rather than about the run. Say nothing,
        # exactly as build_statement names nothing on an incomplete archive.
        report.reason = "no context clauses recorded"
        return "", report
    _, missing = split_by_presence(instructed, haystack)
    report.subjects_missing = [entity.text for entity in missing]
    if not missing:
        report.reason = "every named item was already a subject"
        return "", report
    text, named = nudge_block(report.subjects_missing, iterations_left)
    if not text:
        report.reason = "the note would not fit its byte cap"
        return "", report
    report.fired = True
    report.subjects_named = named
    report.text_bytes = len(text.encode("utf-8"))
    return text, report


@dataclass
class CoverageReport:
    """What the statement said, and what it cost."""

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
    #: ido-8ps.27 (b): the two halves of ``unobserved``. ``unavailable`` is the
    #: subset the run named in a command; ``never_attempted`` is the rest.
    unavailable: list[str] = field(default_factory=list)
    never_attempted: list[str] = field(default_factory=list)
    #: ido-8ps.28. ``evidence`` is EVERY subject this run stamped a command
    #: against, with the other named items its own observations contain --
    #: including subjects whose list is empty, because post_check measures those
    #: too and a measure is allowed to know what a sentence may not say.
    #: ``evidence_named`` is the subset the block actually PRINTED: non-empty,
    #: inside the byte cap, positive half only.
    evidence: list[tuple[str, list[str]]] = field(default_factory=list)
    evidence_named: list[str] = field(default_factory=list)
    evidence_subjects: int = 0
    evidence_items: int = 0
    evidence_bytes: int = 0
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

    @property
    def instructed_items(self) -> list[str]:
        """Every named item of the INSTRUCTED kinds, both halves of the split.

        ``observed_named`` and ``unobserved`` partition one set by construction
        (``ido-8ps.24``), so their union is the request's named items and is the
        candidate set the ``ido-8ps.28`` measure tests claims against.
        """
        out: list[str] = list(self.observed_named)
        out.extend(item for item in self.unobserved if item not in out)
        return out

    def as_event(self) -> dict[str, Any]:
        return {
            "exhausted": self.exhausted,
            "steps": self.steps,
            "entities_total": self.entities_total,
            "entities_observed": self.entities_observed,
            "entities_unobserved": self.entities_unobserved,
            "observed": list(self.observed),
            "unobserved": list(self.unobserved),
            "unavailable": list(self.unavailable),
            "never_attempted": list(self.never_attempted),
            "evidence": [
                {"subject": subject, "items": list(items)}
                for subject, items in self.evidence
            ],
            "evidence_named": list(self.evidence_named),
            "evidence_subjects": self.evidence_subjects,
            "evidence_items": self.evidence_items,
            "evidence_bytes": self.evidence_bytes,
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
    observations: Optional[Iterable[Any]] = None,
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
    # ido-8ps.27 (b): split the instructed list by whether the run ever named
    # the item in a command. Order is preserved on both sides so the block is
    # deterministic.
    issued = issued_commands(trajectory)
    by_text = {entity.text: entity for entity in unobserved}
    unavailable = [
        text for text in instructed
        if (key := getattr(by_text.get(text), "key", "")) and key in issued
    ]
    never_attempted = [text for text in instructed if text not in unavailable]

    # ido-8ps.28. A failure is an empty list and never a failed turn: this
    # sentence adds evidence to a statement that is already complete without it.
    #
    # It is NOT gated on `complete`. The completeness refusal exists because
    # "appears in no retrieved observation" would otherwise be a claim about the
    # archive; a statement that only ever says what IS in an observation cannot
    # make that mistake, and on a partial archive it simply says less.
    evidence: list[tuple[str, list[str]]] = []
    try:
        evidence = [
            (subject.text, [item.text for item in items])
            for subject, items in evidence_by_subject(
                [e for e in entities if e.kind in INSTRUCTED_KINDS],
                scope=scope,
                archive=archive,
                handle_store=handle_store,
                observations=observations,
            )
        ]
    except Exception:  # noqa: BLE001 - a sentence must never fail a turn
        logger.debug("answer coverage could not read subject evidence",
                     exc_info=True)
        evidence = []
    printed = [(subject, items) for subject, items in evidence if items]
    stated, stated_subjects, stated_items = evidence_sentence(printed)

    statement = coverage_block(
        unobserved=instructed,
        exhausted=bool(exhausted),
        steps=steps,
        unavailable=unavailable,
        # ido-8ps.24: only the kinds that are ever INSTRUCTED as "not retrieved"
        # are listed back as retrieved, so the two lists partition one set. A
        # quoted request phrase is measured and never instructed, and naming one
        # here would turn a measure into an instruction by the back door.
        observed=[
            entity.text for entity in observed if entity.kind in INSTRUCTED_KINDS
        ],
        evidence=printed,
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
        unavailable=unavailable,
        never_attempted=never_attempted,
        evidence=evidence,
        evidence_named=[subject for subject, _ in printed][:stated_subjects],
        evidence_subjects=stated_subjects,
        evidence_items=stated_items,
        evidence_bytes=len(stated.encode("utf-8")),
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
    #: ido-8ps.27 (b): the same three counts, split by WHY the item is missing.
    #: They partition the ``*_on_unobserved`` totals above.
    unavailable_total: int = 0
    unavailable_mentioned: int = 0
    unavailability_claim_on_unavailable: int = 0
    not_retrieved_on_unavailable: int = 0
    never_attempted_total: int = 0
    never_attempted_mentioned: int = 0
    unavailability_claim_on_never_attempted: int = 0
    not_retrieved_on_never_attempted: int = 0
    observed_total: int = 0
    observed_mentioned: int = 0
    unavailability_claim_on_observed: int = 0
    not_retrieved_on_observed: int = 0
    #: ido-8ps.28. Per subject, the items the answer CLAIMS of that subject,
    #: split by whether the evidence sentence listed that item for that subject.
    #: An item is claimed of a subject when it is written within
    #: ``CLAIM_WINDOW_CHARS`` of the subject's name -- the same proximity test
    #: every other count in this class uses, and it is a count and not a
    #: judgement: "unlisted" is where the premise-copy rows live, not a verdict
    #: that any one of them is wrong.
    evidence_subjects_total: int = 0
    evidence_subjects_mentioned: int = 0
    evidence_claims_listed: int = 0
    evidence_claims_unlisted: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    observed_details: list[dict[str, Any]] = field(default_factory=list)
    evidence_details: list[dict[str, Any]] = field(default_factory=list)

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
            "unavailable_total": self.unavailable_total,
            "unavailable_mentioned": self.unavailable_mentioned,
            "unavailability_claim_on_unavailable": self.unavailability_claim_on_unavailable,
            "not_retrieved_on_unavailable": self.not_retrieved_on_unavailable,
            "never_attempted_total": self.never_attempted_total,
            "never_attempted_mentioned": self.never_attempted_mentioned,
            "unavailability_claim_on_never_attempted": self.unavailability_claim_on_never_attempted,
            "not_retrieved_on_never_attempted": self.not_retrieved_on_never_attempted,
            "observed_total": self.observed_total,
            "observed_mentioned": self.observed_mentioned,
            "unavailability_claim_on_observed": self.unavailability_claim_on_observed,
            "not_retrieved_on_observed": self.not_retrieved_on_observed,
            "misuse_on_observed": self.misuse_on_observed,
            "evidence_subjects_total": self.evidence_subjects_total,
            "evidence_subjects_mentioned": self.evidence_subjects_mentioned,
            "evidence_claims_listed": self.evidence_claims_listed,
            "evidence_claims_unlisted": self.evidence_claims_unlisted,
            "per_item": list(self.details),
            "per_observed_item": list(self.observed_details),
            "per_evidence_subject": list(self.evidence_details),
        }


def post_check(
    answer: Any,
    unobserved: Iterable[str],
    observed: Iterable[str] = (),
    unavailable: Iterable[str] = (),
    evidence: Iterable[tuple[str, Iterable[str]]] = (),
    items: Iterable[str] = (),
) -> PostCheck:
    """Count absence phrasing near each named item. Measurement only.

    Nothing here changes the answer, retries the call, or fails the turn. It
    exists so ``ido-8ps.22`` can be scored by a number rather than by reading
    five answers by hand, and so a later run can be compared to this one.
    ``observed`` is ``ido-8ps.24``: the same window test applied to the items
    the statement said WERE retrieved, where any hit at all is a defect.
    ``unavailable`` is ``ido-8ps.27`` (b): the subset of ``unobserved`` the run
    did name in a command, counted separately from the ones it never asked for.

    ``evidence`` and ``items`` are ``ido-8ps.28``: every subject the run stamped
    a command against with the items its own observations contain, and the
    request's named items as the candidates. Per subject, each candidate written
    within ``CLAIM_WINDOW_CHARS`` of that subject's name is counted as claimed of
    it, and split by whether the evidence sentence listed it for that subject.
    Both halves are counts. The unlisted half is where a claim copied off the
    request's premise lands, but a proximity test is not a judgement and this
    number is never read as one.
    """
    text = normalise(answer)
    check = PostCheck(answer_bytes=len(str(answer or "").encode("utf-8")))
    tried = {normalise(name) for name in unavailable if normalise(name)}
    for name in unobserved:
        key = normalise(name)
        check.unobserved_total += 1
        # ido-8ps.27 (b): every unobserved item belongs to exactly one kind, so
        # the two sets of counts partition the totals above.
        kind = "unavailable" if key in tried else "never_attempted"
        if kind == "unavailable":
            check.unavailable_total += 1
        else:
            check.never_attempted_total += 1
        if not key:
            continue
        windows = _windows(text, key)
        if not windows:
            check.silent_on_unobserved += 1
            check.details.append({"item": name, "mentioned": False, "kind": kind,
                                  "unavailability_claims": 0, "not_retrieved": 0})
            continue
        check.unobserved_mentioned += 1
        claims = sum(1 for window in windows if _UNAVAILABILITY_RE.search(window))
        marked = sum(1 for window in windows if _NOT_RETRIEVED_RE.search(window))
        check.unavailability_claim_on_unobserved += claims
        check.not_retrieved_on_unobserved += marked
        if kind == "unavailable":
            check.unavailable_mentioned += 1
            check.unavailability_claim_on_unavailable += claims
            check.not_retrieved_on_unavailable += marked
        else:
            check.never_attempted_mentioned += 1
            check.unavailability_claim_on_never_attempted += claims
            check.not_retrieved_on_never_attempted += marked
        check.details.append({"item": name, "mentioned": True, "kind": kind,
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
    candidates = [(str(item), normalise(item)) for item in items]
    for subject, listed in evidence:
        name = str(subject)
        key = normalise(name)
        shown = {normalise(item) for item in listed if normalise(item)}
        check.evidence_subjects_total += 1
        if not key:
            continue
        windows = _windows(text, key)
        if not windows:
            check.evidence_details.append(
                {"subject": name, "mentioned": False,
                 "listed": [str(item) for item in listed],
                 "claimed_listed": [], "claimed_unlisted": []}
            )
            continue
        check.evidence_subjects_mentioned += 1
        claimed_listed: list[str] = []
        claimed_unlisted: list[str] = []
        for spelling, candidate in candidates:
            if not candidate or candidate == key:
                continue
            if not any(candidate in window for window in windows):
                continue
            (claimed_listed if candidate in shown else claimed_unlisted).append(
                spelling
            )
        check.evidence_claims_listed += len(claimed_listed)
        check.evidence_claims_unlisted += len(claimed_unlisted)
        check.evidence_details.append(
            {"subject": name, "mentioned": True,
             "listed": [str(item) for item in listed],
             "claimed_listed": claimed_listed,
             "claimed_unlisted": claimed_unlisted}
        )
    return check


__all__ = [
    "ATTEMPT_SPLIT",
    "COVERAGE_KEY",
    "CLAIM_WINDOW_CHARS",
    "CoverageReport",
    "EVIDENCE_HEAD",
    "EVIDENCE_LIST_MAX_BYTES",
    "EVIDENCE_MORE",
    "OBSERVED_LIST_MAX_BYTES",
    "RETRIEVED_RULE",
    "Entity",
    "INSTRUCTED_KINDS",
    "MAX_ENTITIES",
    "MAX_ENTITY_CHARS",
    "MIN_ENTITY_CHARS",
    "NUDGE_MAX_BYTES",
    "NUDGE_MIN_ITERS_LEFT",
    "NudgeReport",
    "PLAN_MARKER",
    "PostCheck",
    "aliased_executes",
    "build_nudge",
    "build_statement",
    "coverage_block",
    "drop_zero_match_echo",
    "evidence_by_subject",
    "evidence_sentence",
    "issued_commands",
    "named_entities",
    "normalise",
    "nudge_block",
    "post_check",
    "request_text",
    "retrieved_corpus",
    "retrieved_text",
    "split_by_presence",
    "strip_query_echoes",
    "subject_corpus",
]
