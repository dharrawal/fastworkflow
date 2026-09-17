"""Attribution check: a claim that pairs a subject with a property literal.

``ido-8ps.28``, step 1. Measurement only, offline, nothing wired to any loop.

THE FAILURE. On the pinned card of the ``ido-8ps`` series the run retrieves the
right rows and the answer still writes the wrong claim: three people are each
credited with a property the run's own evidence for THAT person does not
contain, in a table row that names the person and the property together. The
evidence is correct in every attempt; what is wrong is the pairing.

THE RULE, and it is a rule about text and stamps, not about any workflow. Take
the named items of the request (``answer_coverage.named_entities`` over
``answer_coverage.request_text``). Cut the answer into units -- a table row, or
a sentence. In one unit, two named items written in order are a SUBJECT and a
PROPERTY LITERAL unless the text between them is nothing but enumeration. The
claim is supported when the property literal appears in the text of at least one
observation of this turn whose per-observation SUBJECT CLAUSE (the
``ido-8ps.13`` context-instance line, the same stamp the roster nudge reads)
names that subject. Otherwise the pair is flagged as an unsupported attribution.

What the check refuses to do:

* **It never decides what a thing IS.** It has no vocabulary, no schema and no
  command names. A "subject" is an item some command of this run was stamped
  against; a "property" is any other named item of the request written after it
  in the same unit. Swap the workflow and the same code runs.
* **It never reads a negative as a claim.** "X does not hold Y" is not an
  attribution, so a negation cue between the two mentions (or just before the
  subject) drops the pair.
* **It never confuses "never looked" with "looked and it was not there".** A
  subject that was the subject of no command of this run is reported under its
  own kind, ``subject_not_observed``. A turn with no clauses at all therefore
  raises no unsupported flag, which is the same refusal
  ``answer_coverage.build_statement`` makes on an incomplete archive.
* **It never edits, gates or retries an answer.** It returns counts and flags.

Segments: a named item written ``A_B``, ``A/B`` or ``A:B`` is often written in
prose by its last segment alone. ``allow_segments`` lets the last segment stand
for the whole item, on both sides of the test, when it is at least
``MIN_SEGMENT_CHARS`` long. It is a property of how qualified names are written,
not of any one workflow, and it is a flag on the function so a caller can
measure the check with and without it.

A tail only stands for its item when it stands for NOTHING ELSE the request
named (``ido-rf3``). "Okta_Cloud Administrator" and "Active Directory_Cloud
Administrator" share the tail "Cloud Administrator", so the tail names neither
of them, and reading it as either would make two different rights
interchangeable: a subject holding only the Okta right would be credited with
the Active Directory one. So a tail that occurs in any OTHER item of the same
request is dropped, and the item is then matched by its full name alone. The
convenience survives wherever it is unambiguous, which is the only place it was
ever sound. Uniqueness is a property of the request's item SET, so the forms are
built for the whole set at once by :func:`match_forms_index`; ``match_forms``
keeps its per-entity signature and takes the set as ``among``.

The SET is not the whole ambiguity, because the evidence and the answer contain
qualified names the request never mentions. So a tail is also read as its item
only where the text writes the item unqualified: "Cloud Administrator" inside
"SCOM_Cloud Administrator" is the SCOM right written in full, and crediting the
requested Active Directory right with it would be the same false support from
the other side. :func:`writes` is the one containment test both halves use.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow.answer_coverage import (
    INSTRUCTED_KINDS,
    Entity,
    named_entities,
    normalise,
    request_text,
)
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

#: Bounds. Every one of them is a backstop against a runaway answer, never a
#: budget: the entities come from one request and the units from one answer.
MAX_ANSWER_BYTES = 262_144
MAX_UNITS = 4_000
MAX_UNIT_CHARS = 8_000
MAX_MENTIONS_PER_UNIT = 64
MAX_PAIRS_PER_UNIT = 512
MAX_FLAGS = 256
SPAN_MAX_CHARS = 400
SPAN_CONTEXT_CHARS = 120

#: The shortest last segment of a qualified name that may stand for the whole
#: name. Short tails ("Officer", "1") are ambiguous in any workflow.
MIN_SEGMENT_CHARS = 8
_SEGMENT_SEPARATORS = "_/:\\"

#: Kinds that can carry an attribution. The same set ``answer_coverage``
#: instructs on: a quoted phrase of a request is narrative framing, and the
#: honesty replay is why (the pinned card quotes a control label the catalogue
#: spells differently).
ATTRIBUTION_KINDS = INSTRUCTED_KINDS

#: Between two mentions, THIS and nothing else means the two were listed, not
#: predicated. Other mentions, bracketed asides, code spans and markup tags are
#: removed before the test, so "A (obs 33) and B" is one list and not a claim.
#: ``|`` is deliberately absent: a table cell boundary is where attribution
#: happens in a table.
_ENUMERATOR_SEPARATORS = r"[\s,;&/+·•*\-–—]"
_ENUMERATOR_RE = re.compile(
    rf"^{_ENUMERATOR_SEPARATORS}*"
    r"(?:(?:and|or|nor|plus|with|&|as well as|along with)\s*)?"
    r"(?:for|of|to|in|by|from|at|on)?"
    rf"{_ENUMERATOR_SEPARATORS}*$"
)

#: Cut out of a gap before the enumeration test. A parenthetical, a code span
#: and a markup tag are asides: none of them predicates one item of another.
_ASIDE_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|`[^`]*`|<[^<>]*>|\{[^{}]*\}")

#: A table row is read the way a table is read: the first cell that names
#: something is the row's subject, and the cells after it are what the row says
#: about it. Two items in the SAME cell are a list, not a predication.
CELL_SEPARATOR = "|"

#: A negation cue anywhere in the gap (or just before the subject) means the
#: sentence is denying the pairing, and a denial is not an attribution.
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|without|lacks|lacking|lacked|absent|"
    r"excluded|missing|none|except|other than|rather than|instead of|"
    r"unlike|cannot|can't|couldn't|doesn't|does not|do not|don't|"
    r"isn't|is not|aren't|are not|wasn't|was not|weren't|were not|"
    r"hasn't|has not|haven't|have not)\b"
)
NEGATION_LOOKBEHIND_CHARS = 40
#: And just after the property literal: "X - Y does not appear in her list" denies
#: the pairing from the other side.
NEGATION_LOOKAHEAD_CHARS = 40

REASON_UNSUPPORTED = "unsupported_attribution"
REASON_SUBJECT_NOT_OBSERVED = "subject_not_observed"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One observation of the turn, with the subject it was stamped against.

    ``text`` is the evidence -- archived observation text, and the stored rows
    behind a declared result handle. ``clause`` is the ``ido-8ps.13``
    context-instance line recorded for that alias, and it is read for one thing
    only: which subject this evidence belongs to.
    """

    alias: str
    clause: str
    text: str


def observations(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
) -> list[Observation]:
    """This turn's evidence, one entry per alias, subject clause attached.

    The readers are the ones ``answer_coverage.retrieved_corpus`` and
    ``answer_rehydration`` already use -- the archive listing, the hot handles,
    ``context_clause_of`` and ``stored_rows_block``. The only difference is that
    the text is kept per alias instead of being concatenated, because "in an
    observation whose subject is X" is a question the single haystack cannot
    answer. Search answers (``O5#a1``) are excluded here for the same reason
    ``retrieved_corpus`` excludes them: a model's summary repeats the agent's own
    question back.
    """
    from fastworkflow import answer_rehydration

    selected = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()
    try:
        rows = archive.list(selected)
    except Exception:  # noqa: BLE001 - an unreadable archive is an empty one
        logger.debug("attribution check could not list the archive", exc_info=True)
        rows = []

    texts: dict[str, list[str]] = {}
    order: list[str] = []
    def take(alias: str, handle: Mapping[str, Any]) -> None:
        if not alias or is_search_answer_key(alias):
            return
        if alias not in texts:
            texts[alias] = []
            order.append(alias)
        texts[alias].append(strip_alias_line(str(handle.get("text") or "")))

    for handle in rows:
        take(str(handle.get("alias") or ""), handle)
    for alias, handle in stored_handles(selected).items():
        take(str(alias), handle or {})

    if handle_store is None:
        try:
            from fastworkflow import result_handles

            handle_store = result_handles.store()
        except Exception:  # noqa: BLE001
            handle_store = None
    if handle_store is not None:
        for alias in order:
            try:
                declaration = handle_store.get_declaration(selected, alias)
            except Exception:  # noqa: BLE001
                continue
            if declaration is None or str(declaration.get("parent_alias") or ""):
                continue
            try:
                texts[alias].append(
                    answer_rehydration.stored_rows_block(
                        alias, scope=selected, store=handle_store
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("attribution check could not read rows for %s", alias,
                             exc_info=True)
    return [
        Observation(
            alias=alias,
            clause=normalise(
                context_clause_of(selected, alias, selected_archive=archive) or ""),
            text=normalise("\n".join(texts[alias])),
        )
        for alias in order
    ]


# ---------------------------------------------------------------------------
# Match forms
# ---------------------------------------------------------------------------

def match_forms(
    entity: Entity,
    *,
    allow_segments: bool = True,
    among: Optional[Iterable[Entity]] = None,
) -> list[str]:
    """The normalised strings that count as writing *entity*.

    The item itself always; and, when ``allow_segments``, its last segment after
    ``_``, ``/``, ``:`` or ``\\`` if that segment is long enough to be a handle
    on its own. Nothing else: no stemming, no synonyms, no abbreviation table.

    ``among`` is the other named items of the same request. A tail that occurs
    in any of their full names is ambiguous -- it does not say WHICH item was
    written -- so it is dropped and the item keeps its full name only
    (``ido-rf3``). Callers that hold the whole set should use
    :func:`match_forms_index`, which applies this to every item at once.
    """
    forms = [entity.key]
    if not allow_segments:
        return forms
    tail = _segment_tail(entity)
    if not tail:
        return forms
    for other in among or ():
        if other.key != entity.key and tail in other.key:
            return forms
    forms.append(tail)
    return forms


def _segment_tail(entity: Entity) -> str:
    """The last segment of *entity* when it is long enough to stand alone."""
    tail = entity.key
    for separator in _SEGMENT_SEPARATORS:
        tail = tail.rsplit(separator, 1)[-1]
    tail = tail.strip()
    if tail and tail != entity.key and len(tail) >= MIN_SEGMENT_CHARS:
        return tail
    return ""


def match_forms_index(
    entities: Iterable[Entity], *, allow_segments: bool = True
) -> dict[str, list[str]]:
    """``{entity key: its match forms}`` for a whole request at once.

    The one place segment tails are tested for ambiguity, so the forward
    evidence sentence and the after-the-fact check can never disagree about what
    a tail means.
    """
    items = list(entities)
    return {
        entity.key: match_forms(entity, allow_segments=allow_segments, among=items)
        for entity in items
    }


def _tail_of_a_longer_name(text: str, at: int) -> bool:
    """Is the match at *at* the last segment of some OTHER qualified name?

    ``ido-rf3``. A tail stands for its item only where the item is written
    unqualified. "Cloud Administrator" inside "SCOM_Cloud Administrator" is the
    SCOM right written in full, not the Active Directory right written short, so
    reading it as the requested item would credit a claim to evidence that says
    something else. The prefix is not compared to anything: ANY prefix means the
    text named a different qualified thing.
    """
    return at > 0 and text[at - 1] in _SEGMENT_SEPARATORS


def writes(text: str, forms: Sequence[str]) -> bool:
    """Does *text* write the item whose match ``forms`` are?

    ``forms[0]`` is the item's full name and any other form is a tail
    (:func:`match_forms`), and a tail only counts where it is not itself the tail
    of a longer qualified name. The one containment test of this module, so the
    clause side, the evidence side and the answer side cannot disagree.
    """
    for index, form in enumerate(forms):
        if not form:
            continue
        at = text.find(form)
        while at >= 0:
            if index == 0 or not _tail_of_a_longer_name(text, at):
                return True
            at = text.find(form, at + 1)
    return False


@dataclass(frozen=True)
class _Mention:
    entity: Entity
    start: int
    end: int


def _mentions(unit: str, forms: Mapping[str, list[str]],
              entities: Mapping[str, Entity]) -> list[_Mention]:
    """Every occurrence of a named item in *unit*, longest match wins.

    Overlaps are resolved by length so "Active Directory" inside "Active
    Directory_Cloud Administrator" is one mention of the longer item and not two
    items sitting on the same characters.
    """
    found: list[_Mention] = []
    for key, entity in entities.items():
        for index, form in enumerate(forms[key]):
            if not form:
                continue
            at = unit.find(form)
            while at >= 0:
                if index == 0 or not _tail_of_a_longer_name(unit, at):
                    found.append(_Mention(entity, at, at + len(form)))
                at = unit.find(form, at + 1)
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[_Mention] = []
    for mention in found:
        if any(mention.start < other.end and other.start < mention.end
               and (other.end - other.start) >= (mention.end - mention.start)
               and other is not mention
               for other in kept):
            continue
        kept.append(mention)
        if len(kept) >= MAX_MENTIONS_PER_UNIT:
            break
    return kept


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#: A sentence ends at ``.!?``; a clause ends at ``;``. Both are boundaries here,
#: because "A holds X; B holds Y" is two claims and reading it as one would
#: attribute Y to A.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|;")


def units(answer: Any) -> list[str]:
    """The answer as claim-sized units: one table row, or one sentence.

    A line carrying ``|`` is a table row and stays whole -- the row IS the claim,
    and cutting it at the cell boundary would throw away exactly the pairing this
    module is about. Every other line is split into sentences.
    """
    text = str(answer or "")
    if len(text.encode("utf-8")) > MAX_ANSWER_BYTES:
        text = text.encode("utf-8")[:MAX_ANSWER_BYTES].decode("utf-8", "ignore")
    out: list[str] = []
    for line in text.splitlines():
        pieces = [line] if "|" in line else _SENTENCE_SPLIT.split(line)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                out.append(piece[:MAX_UNIT_CHARS])
            if len(out) >= MAX_UNITS:
                return out
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Flag:
    """One claim the evidence does not carry."""

    subject: str
    property: str
    answer_span: str
    reason: str
    unit: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "property": self.property,
            "answer_span": self.answer_span,
            "reason": self.reason,
            "unit": self.unit,
        }


@dataclass
class AttributionReport:
    """What the answer paired, and which pairings its evidence carries."""

    entities_total: int = 0
    subjects_observed: int = 0
    observations_total: int = 0
    clauses_total: int = 0
    units_total: int = 0
    units_with_pairs: int = 0
    pairs_total: int = 0
    pairs_enumerated: int = 0
    pairs_negated: int = 0
    pairs_supported: int = 0
    pairs_unsupported: int = 0
    pairs_subject_not_observed: int = 0
    answer_bytes: int = 0
    truncated: bool = False
    segments: bool = True
    flags: list[Flag] = field(default_factory=list)
    flags_dropped: int = 0

    @property
    def unsupported(self) -> int:
        """The number the bead is about: claims the run's own evidence denies."""
        return self.pairs_unsupported

    def as_event(self) -> dict[str, Any]:
        return {
            "entities_total": self.entities_total,
            "subjects_observed": self.subjects_observed,
            "observations_total": self.observations_total,
            "clauses_total": self.clauses_total,
            "units_total": self.units_total,
            "units_with_pairs": self.units_with_pairs,
            "pairs_total": self.pairs_total,
            "pairs_enumerated": self.pairs_enumerated,
            "pairs_negated": self.pairs_negated,
            "pairs_supported": self.pairs_supported,
            "pairs_unsupported": self.pairs_unsupported,
            "pairs_subject_not_observed": self.pairs_subject_not_observed,
            "unsupported": self.unsupported,
            "answer_bytes": self.answer_bytes,
            "truncated": self.truncated,
            "segments": self.segments,
            "flags": [flag.as_dict() for flag in self.flags],
            "flags_dropped": self.flags_dropped,
        }


def _span(unit: str, start: int, end: int) -> str:
    left = max(0, start - SPAN_CONTEXT_CHARS)
    right = min(len(unit), end + SPAN_CONTEXT_CHARS)
    text = unit[left:right]
    if len(text) > SPAN_MAX_CHARS:
        text = text[:SPAN_MAX_CHARS]
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(unit) else ""
    return f"{prefix}{text}{suffix}"


def _is_enumeration(gap: str, inner: Sequence[tuple[int, int]], offset: int) -> bool:
    """True when the only thing between the two mentions is a list.

    Other mentions are cut out of the gap first, so "A, B and C" is one
    enumeration of three items rather than a claim about A and C; then the
    asides are cut, so "A (observation O33) and B" is the same list.
    """
    cleaned: list[str] = []
    cursor = 0
    for start, end in inner:
        start, end = max(0, start - offset), max(0, end - offset)
        if start > cursor:
            cleaned.append(gap[cursor:start])
        cursor = max(cursor, end)
    cleaned.append(gap[cursor:])
    text = _ASIDE_RE.sub(" ", "".join(cleaned))
    while True:
        shorter = _ASIDE_RE.sub(" ", text)
        if shorter == text:
            break
        text = shorter
    return bool(_ENUMERATOR_RE.match(text))


#: A negation window stops at a cell boundary or a sentence end: "…Cloud
#: Administrator | metadata empty (no collection info)" denies nothing about the
#: cell before it.
_WINDOW_STOP = "|.!?;"


def _clip_ahead(text: str) -> str:
    for position, char in enumerate(text):
        if char in _WINDOW_STOP:
            return text[:position]
    return text


def _clip_behind(text: str) -> str:
    for position in range(len(text) - 1, -1, -1):
        if text[position] in _WINDOW_STOP:
            return text[position + 1:]
    return text


def _cell_index(unit: str, position: int) -> int:
    """Which cell of a table row *position* falls in. 0 for a sentence."""
    return unit.count(CELL_SEPARATOR, 0, position)


def enumeration_groups(
    unit: str, mentions: Sequence[_Mention]
) -> list[list[_Mention]]:
    """Consecutive mentions joined by nothing but enumeration, as one group.

    "Alan Cooper, Alisha Ochoa and Anna Garcia" is one group; the group is what
    a following claim is said of, so a distributive list gets each of its
    members checked rather than only the last.
    """
    groups: list[list[_Mention]] = []
    for mention in mentions:
        if groups:
            previous = groups[-1][-1]
            gap = unit[previous.end:mention.start]
            if _is_enumeration(gap, [], previous.end):
                groups[-1].append(mention)
                continue
        groups.append([mention])
    return groups


def pairs_of(unit: str, mentions: Sequence[_Mention]) -> list[tuple[_Mention, _Mention]]:
    """Which mention is the subject of which, by the shape of the unit.

    TABLE ROW. The row's subject is the first cell that names something, and
    every named item in a LATER cell is something the row says about it. Items
    inside one cell are a list.

    SENTENCE. A named item is said of the nearest enumeration group written
    before it. English predicates on what it has just named, and the
    alternative -- pairing every earlier item with every later one -- reads
    "A holds X; B holds Y" as a claim that A holds Y.
    """
    if CELL_SEPARATOR in unit:
        cells = [(_cell_index(unit, m.start), m) for m in mentions]
        if not cells:
            return []
        first = min(index for index, _ in cells)
        subjects = [m for index, m in cells if index == first]
        return [
            (subject, m) for index, m in cells if index > first
            for subject in subjects if m.entity.key != subject.entity.key
        ]
    groups = enumeration_groups(unit, mentions)
    return [
        (subject, mention)
        for before, group in zip(groups, groups[1:])
        for mention in group
        for subject in before
        if subject.entity.key != mention.entity.key
    ]


# ---------------------------------------------------------------------------
# Subjects, and what each subject's own evidence contains
# ---------------------------------------------------------------------------

def _normalised(observations: Iterable[Observation]) -> list[Observation]:
    """The evidence in the one comparison form.

    Idempotent, so a caller that has already normalised loses nothing by asking
    again and a caller that has not cannot get a wrong answer.
    """
    return [
        Observation(alias=item.alias, clause=normalise(item.clause),
                    text=normalise(item.text))
        for item in observations
    ]


def subject_index(
    entities: Iterable[Entity],
    observations: Iterable[Observation],
    *,
    allow_segments: bool = True,
) -> dict[str, list[Observation]]:
    """``{entity key: the observations this turn stamped against it}``.

    An observation belongs to a subject when the subject is written in that
    observation's CLAUSE -- the per-observation context-instance line, the stamp
    recording which thing the command ran against. Observation TEXT is never
    read here, so a name that merely appears in somebody else's listing is not
    thereby a subject.

    Every entity gets an entry; an entity no clause names gets an empty list,
    which is the only honest thing to say about it. Generic: this function knows
    nothing about what a subject or a property means in any workflow.
    """
    items = list(entities)
    evidence = _normalised(observations)
    forms = match_forms_index(items, allow_segments=allow_segments)
    return {
        entity.key: [
            item for item in evidence
            if item.clause and writes(item.clause, forms[entity.key])
        ]
        for entity in items
    }


def subject_evidence(
    entities: Iterable[Entity],
    observations: Iterable[Observation],
    *,
    allow_segments: bool = True,
) -> list[tuple[Entity, list[Entity]]]:
    """``[(subject, the OTHER given items that subject's own observations contain)]``.

    The question :func:`check_attribution` asks of a finished answer -- "does
    this property literal appear in an observation whose subject is this
    subject?" -- asked FORWARD, of the evidence alone. No answer is read and no
    claim is judged: this reports what the run's own evidence says per subject.

    Subjects keep the order they were given in, and only subjects some
    observation was stamped against appear at all: an item no clause names has
    no evidence OF ITS OWN to report, and that is not the same as evidence that
    contains nothing. A subject that IS stamped but whose evidence contains none
    of the other items is returned with an empty list, because the difference
    between the two cases is real and belongs to the caller. Nothing here
    phrases anything, and nothing here decides what an empty list means.
    """
    items = list(entities)
    evidence = _normalised(observations)
    index = subject_index(items, evidence, allow_segments=allow_segments)
    forms = match_forms_index(items, allow_segments=allow_segments)
    out: list[tuple[Entity, list[Entity]]] = []
    for entity in items:
        where = index.get(entity.key) or []
        if not where:
            continue
        out.append((
            entity,
            [
                other for other in items
                if other.key != entity.key
                and any(writes(item.text, forms[other.key]) for item in where)
            ],
        ))
    return out


def check_attribution(
    *,
    request: Any,
    answer: Any,
    observations: Iterable[Observation],
    allow_segments: bool = True,
) -> AttributionReport:
    """Pure. Same request, answer and evidence in, same report out.

    No store is opened here, no model is called, no clock is read.
    """
    evidence = _normalised(observations)
    raw_answer = str(answer or "")
    report = AttributionReport(
        answer_bytes=len(raw_answer.encode("utf-8")),
        truncated=len(raw_answer.encode("utf-8")) > MAX_ANSWER_BYTES,
        segments=bool(allow_segments),
        observations_total=len(evidence),
        clauses_total=sum(1 for item in evidence if item.clause),
    )
    items = [
        entity for entity in named_entities(request_text(request))
        if entity.kind in ATTRIBUTION_KINDS
    ]
    report.entities_total = len(items)
    if not items:
        return report

    entities = {entity.key: entity for entity in items}
    forms = match_forms_index(items, allow_segments=allow_segments)

    subjects = subject_index(items, evidence, allow_segments=allow_segments)
    report.subjects_observed = sum(1 for key in subjects if subjects[key])

    seen: set[tuple[str, str, str]] = set()
    for index, unit in enumerate(units(raw_answer)):
        report.units_total += 1
        folded = normalise(unit)
        mentions = _mentions(folded, forms, entities)
        if len(mentions) < 2:
            continue
        counted_unit = False
        for subject, property_mention in pairs_of(folded, mentions)[:MAX_PAIRS_PER_UNIT]:
            report.pairs_total += 1
            if not counted_unit:
                report.units_with_pairs += 1
                counted_unit = True
            gap = folded[subject.end:property_mention.start]
            inner = [
                (m.start, m.end) for m in mentions
                if m.start >= subject.end and m.end <= property_mention.start
            ]
            if _is_enumeration(gap, inner, subject.end):
                report.pairs_enumerated += 1
                continue
            behind = _clip_behind(
                folded[max(0, subject.start - NEGATION_LOOKBEHIND_CHARS):
                       subject.start]
            )
            ahead = _clip_ahead(
                folded[property_mention.end:
                       property_mention.end + NEGATION_LOOKAHEAD_CHARS]
            )
            if (_NEGATION_RE.search(gap) or _NEGATION_RE.search(behind)
                    or _NEGATION_RE.search(ahead)):
                report.pairs_negated += 1
                continue
            key = (subject.entity.key, property_mention.entity.key, folded)
            where = subjects[subject.entity.key]
            if not where:
                report.pairs_subject_not_observed += 1
                reason = REASON_SUBJECT_NOT_OBSERVED
            elif any(
                writes(item.text, forms[property_mention.entity.key])
                for item in where
            ):
                report.pairs_supported += 1
                continue
            else:
                report.pairs_unsupported += 1
                reason = REASON_UNSUPPORTED
            if key in seen:
                continue
            seen.add(key)
            if len(report.flags) >= MAX_FLAGS:
                report.flags_dropped += 1
                continue
            report.flags.append(
                Flag(
                    subject=subject.entity.text,
                    property=property_mention.entity.text,
                    answer_span=_span(folded, subject.start, property_mention.end),
                    reason=reason,
                    unit=index,
                )
            )
    return report


def attribution_report(
    *,
    request: Any,
    answer: Any,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
    allow_segments: bool = True,
) -> AttributionReport:
    """``check_attribution`` over the turn's own stores. Read-only."""
    return check_attribution(
        request=request,
        answer=answer,
        observations=observations(
            scope=scope, archive=archive, handle_store=handle_store
        ),
        allow_segments=allow_segments,
    )


__all__ = [
    "ATTRIBUTION_KINDS",
    "AttributionReport",
    "Flag",
    "MAX_ANSWER_BYTES",
    "MAX_FLAGS",
    "MAX_MENTIONS_PER_UNIT",
    "MAX_UNITS",
    "MIN_SEGMENT_CHARS",
    "NEGATION_LOOKAHEAD_CHARS",
    "NEGATION_LOOKBEHIND_CHARS",
    "Observation",
    "REASON_SUBJECT_NOT_OBSERVED",
    "REASON_UNSUPPORTED",
    "SPAN_MAX_CHARS",
    "attribution_report",
    "check_attribution",
    "match_forms",
    "match_forms_index",
    "writes",
    "subject_evidence",
    "subject_index",
    "enumeration_groups",
    "observations",
    "pairs_of",
    "units",
]
