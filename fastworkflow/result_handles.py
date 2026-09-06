"""Result handles: the full payload stays in evidence, the agent reads a page.

**The problem this exists for.** A ReAct step's observation is whatever string
the tool returned, and that string is appended to the trajectory, which is
re-sent as prompt context on every subsequent step. So the cost of a large
command response is not the one observation — it is that observation multiplied
by every step that follows it. Gate 4 v4 measured the consequence directly: in
one Arm C cell, command response text totalled 922k characters and agent-step
observations totalled 932k; `show_holders` returns all 477 holders in one 23k
string; `open_portrait` returns ~4k each and was called 519 times across 19
cells. Arm A prompts reached 148k input tokens for the same reason.

**What must NOT change.** The full response is the evidence. An evaluator scoring
whether the agent actually saw a holder cannot score a summary, so the
`fw.command.execute` span's `response_text` and the turn record's
`command_outputs` keep the whole payload exactly as before — the 256 KiB
artifact-offload path in `observability_store` still applies to it. Only the
ReAct trajectory, and therefore the LLM context, gets the compact rendering.
That split is the whole design: *evidence is not the agent's context*.

**Opt-in, per command, by declaration.** A command opts in by putting one
artifact in its `CommandResponse.artifacts` — see `declare()`. A command that
does not is not touched anywhere: no store write, no span attribute, no change
to the observation string, byte for byte. This is checked by a parity test
rather than asserted here, because "off path" claims are exactly the kind that
rot silently.

**The handle IS the producing command_call_id.** `plan.py` already binds a
captured handle to `command_call_id` — the id of the command whose
`CommandOutput.artifacts` produced it (`CapturedBinding.command_call_id`). Using
a second identifier here would mean a later composition step had to learn which
of two ids a citation was written in. So `handle_id == CommandOutput.
command_call_id`, minted by `CommandExecutor.invoke_command`, and a cited handle
resolves against the same span the plan binder resolves against.

**Ordering is the producer's responsibility, and the contract carries it.** This
module cannot know whether a backend view returns rows in a stable order, so it
does not pretend to sort them; it stores what the command handed over, pages it
by offset, and records the `ordering` string the command declared so a reader
can see what the paging is resting on. An unstable producer yields unstable
pages, and the record says whose fault that is.

**Bounded by BYTES, not by count, and a live turn is protected.** The obvious
bound — keep the last N handles — is wrong here, and measurably so. EXP-028 runs
in stress mode with no iteration budget: Gate 4 v4 Arm C cells ran 177 agent
steps each, with ~27 `open_portrait` calls per cell on top of `show_holders`,
`list_members`, `who_has_access_to` and `what_can_i_do`, all of which opt in
under ido-mn1.6.2. A count of a few dozen would evict handles the same turn was
still using. Worse, ido-mn1.6.6 resolves the handles the agent CITED at
composition time, at the END of the turn, so an evicted handle is not a retry —
it is a silently missing table in the final answer.

So the bound is a byte budget over stored payloads
(`FW_RESULT_HANDLE_MAX_BYTES`, default 64 MiB), with a count ceiling
(`MAX_STORED_HANDLES`, 4096) as a distant secondary backstop.

The budget is scoped to the SESSION rather than reset per turn, and that is the
stricter choice, not a looser one: a per-turn budget would let a long session
accumulate 64 MiB per turn with no total bound at all, which is not a bound. What
the live turn actually needs — that its own handles survive until composition
reads them — is delivered by the protection rule below plus the sizing: a whole
Arm C cell's handles are on the order of 1 MB, so a turn never approaches the
budget and never has to compete with itself. Two rules follow:

* **A handle produced in the CURRENT turn is never evicted to satisfy the count
  ceiling.** Only the byte budget can reach into the live turn, and only after
  everything older is gone — because at that point the alternative is unbounded
  memory, and a bound that can be exceeded is not a bound.
* **Eviction is recorded in evidence.** The `fw.command.execute` span of the
  command whose storage displaced them lists the evicted ids
  (`result_handles_evicted`). Without it, a later "handle not found" is a dead
  end: nothing would say the handle had ever existed, when it was dropped, or
  what dropped it.

A fetch against an evicted handle still raises `UnknownResultHandle` rather than
returning an empty page, because "this handle is gone, run the command again" and
"this handle has no rows" are different facts and an agent told the second would
report an empty answer as a complete one.

Leaf-ish module (arch §22): standard library, Pydantic, and `tracing` for host
resolution. The capture policy is resolved lazily so that importing this module
does not drag in the observability store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from fastworkflow import tracing
from fastworkflow.runtime_manifest import DataClassification
from fastworkflow.utils.logging import logger

# The artifact key a command sets to opt in. Dunder-ish for the same reason as
# `__fw_artifact_ref__` and `__fw_capture__`: a reader walking `artifacts` needs
# to tell a framework declaration from workflow data that happens to be a dict.
RESULT_HANDLE_ARTIFACT_KEY = "__fw_result_handle__"

# The cursor's own version, carried in every cursor string. A cursor is handed to
# a model, which may echo it back many steps later — possibly across a
# suspend/resume, possibly into a build whose paging has changed. An unversioned
# opaque offset would then be silently reinterpreted; a versioned one is refused.
CURSOR_VERSION = "v2"

# Historical cursor documentation retained verbatim so recorded v1 semantics
# remain reviewable beside the v2 handle/filter-bound contract.
LEGACY_ENCODE_CURSOR_DOC = (
    """`v1:<offset>`. Versioned so a stale cursor is refused, not misread."""
)
LEGACY_DECODE_CURSOR_DOC = (
    """The offset a cursor names. None/empty is the first page."""
)

# A producer can hand the store fewer rows than the backend says exist. This is
# not ordinary page truncation: no cursor over the stored rows can materialize
# the missing backend page, so every rendering has to say that the source is
# incomplete rather than presenting the end of the stored list as completion.
SOURCE_INCOMPLETE_REASON = "backend-total-exceeds-materialized"

# The real bound: total bytes of stored payload per session. Chosen over a count
# because payload sizes differ by two orders of magnitude between an
# `open_portrait` (~4 KB) and a `show_holders` (~23 KB projected, far more raw),
# so a count bounds nothing in particular. 64 MiB holds every handle a Gate 4
# Arm C cell produces — 177 steps' worth — with room to spare, and is small
# enough that a runaway session is still bounded.
DEFAULT_MAX_STORED_BYTES = 64 * 1024 * 1024
MAX_STORED_BYTES_VAR = "FW_RESULT_HANDLE_MAX_BYTES"

# A distant secondary ceiling, not a working limit. It exists so that a session
# producing millions of tiny handles cannot grow the dict without bound while
# staying under the byte budget; it is deliberately far above any observed turn
# (Arm C's worst cell stores on the order of 200), because a count that binds in
# practice is the failure this bound was rewritten to avoid.
MAX_STORED_HANDLES = 4096

# Default rows in one page when the producer does not say. Small enough that a
# page costs ~1 KB rather than the 23 KB a full holder listing costs, large
# enough that a three-item walk finishes in one page.
DEFAULT_PAGE_SIZE = 20

# A page nobody bounded is still bounded: a producer asking for 10_000 rows a
# page has opted out of compaction while appearing to opt in.
MAX_PAGE_SIZE = 200


# The presentation cap (ido-mn1.6.6). Bounds the payload `resolve_for_presentation`
# hands the extraction call, and nothing else — the store's own budget above is
# about MEMORY, this one is about one prompt. 32 KiB holds the whole projected
# 477-holder listing (~23 KB) plus a portrait, at a cost of roughly 8k input
# tokens, and is small enough that a turn holding 200 handles cannot rebuild the
# 922k-character prompt compaction removed. It is deliberately NOT derived from
# any completion limit: ido-mn1.6.10 sizes the extraction call's completion
# limit PER CALL from `PresentedResults.field_bytes`, so a constant tied to a
# fixed max_tokens here would be a second, stale opinion about the same number.
# Arm-invariant by construction: one constant, read the same way on the flat and
# the leaf paths.
DEFAULT_PRESENTED_MAX_BYTES = 32 * 1024
PRESENTED_MAX_BYTES_VAR = "FW_PRESENTED_RESULT_MAX_BYTES"

# A cap that removes rows or metadata is an infrastructure limitation, not a
# claim about the result. Kept as a closed string so runtime evidence, replay
# evidence and endpoint analysis do not invent near-synonyms for the same event.
PRESENTATION_TRUNCATION_CLASSIFICATION = "infrastructure-truncated"
PRESENTATION_TRUNCATION_REASON = "presentation-payload-cap"

# Individual producer-authored metadata fields are bounded before they enter
# the global presentation allocator. The global 32 KiB cap remains authoritative;
# this local cap only prevents one pathological summary or ordering description
# from starving every row and every later handle.
MAX_PRESENTATION_METADATA_BYTES = 1024

# How many distinct page views one handle remembers. A cap, not a working limit:
# an agent paging a 477-row result at 20 rows a page records 24 views, and the
# resolver needs every one of them to reconstruct what the agent actually saw.
# It exists so a runaway paging loop cannot grow a stored record without bound.
MAX_RECORDED_VIEWS = 128


def _positive_int_env(var: str, default: int) -> int:
    """A positive integer from the process env, then the workflow env, else `default`.

    Read per call rather than cached at construction so a deployment (or a test)
    can change it without rebuilding every live session, and resolved through the
    same process-env-then-workflow-env order the other `FW_*` knobs use. A
    non-positive or unparseable value falls back to the default rather than
    disabling the bound, because "0" almost always means a mis-set variable and
    an unbounded store is the failure mode these budgets exist to prevent.
    """
    raw = os.environ.get(var)
    if raw is None or not str(raw).strip():
        try:
            import fastworkflow

            raw = fastworkflow._env_vars.get(var)
        except Exception:  # pragma: no cover - defensive
            raw = None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def max_stored_bytes() -> int:
    """The per-session payload budget, from the environment or the default."""
    return _positive_int_env(MAX_STORED_BYTES_VAR, DEFAULT_MAX_STORED_BYTES)


def presented_max_bytes() -> int:
    """The per-extraction presentation budget, from the environment or the default."""
    return _positive_int_env(PRESENTED_MAX_BYTES_VAR, DEFAULT_PRESENTED_MAX_BYTES)


class ResultHandleError(ValueError):
    """Something is wrong with a handle or a cursor the agent presented."""


class UnknownResultHandle(ResultHandleError):
    """No stored payload for this handle: never issued, or evicted since."""


class InvalidResultCursor(ResultHandleError):
    """A cursor this build cannot read — wrong version, or not a cursor."""


# ----------------------------------------------------------------------
# What a command declares
# ----------------------------------------------------------------------


class ResultHandleSpec(BaseModel):
    """The opt-in declaration a command puts in its response artifacts.

    `items` are the agent-facing rows, already rendered by the command, in the
    order the command wants them paged. `detail` is anything else the command
    wants kept out of the observation but retrievable — a per-row dict, a raw
    body — and is never rendered into the compact observation.

    `total` is what the *backend* reported, which can exceed `len(items)` when
    the view itself paged; it is carried separately for that reason and defaults
    to the item count when the command has nothing better to say.
    """

    model_config = ConfigDict(extra="forbid")

    #: What kind of thing was listed ("holders", "members", ...). Free text; it
    #: appears in the observation so the agent can tell two handles apart.
    kind: str
    #: One-line agent-facing description of the whole result.
    summary: str
    #: The rows, pre-rendered by the producer, in the producer's order.
    items: list[str] = Field(default_factory=list)
    #: How the producer ordered them. Recorded, not verified — see module doc.
    ordering: str = "producer-defined"
    #: Backend-reported total; defaults to len(items).
    total: Optional[int] = None
    #: Whether `items` materializes the backend source completely. None asks the
    #: framework to infer it from `total <= len(items)`. An explicit True can
    #: never override a backend total larger than the materialized item list.
    source_complete: Optional[bool] = None
    #: Rows per page. Clamped to [1, MAX_PAGE_SIZE].
    page_size: int = DEFAULT_PAGE_SIZE
    #: Filters the producing command ALREADY applied, echoed so the agent never
    #: reads a page as if it were the unfiltered population.
    filters: dict[str, str] = Field(default_factory=dict)
    #: Structured payload kept out of the observation, retrievable by handle.
    detail: dict[str, Any] = Field(default_factory=dict)
    #: The workflow's data classification for `items`/`detail`, if it declared
    #: one. Fed to the capture policy exactly as a parameter's classification is.
    #: The same closed classification governs summary, ordering, producer/view
    #: filters, response storage, and extraction-time resolved rows.
    classification: Optional[DataClassification] = None
    #: True when these rows are DELIVERABLE — the thing an answer about this
    #: request is expected to list, not a lookup the agent made on the way
    #: (ido-mn1.6.6). It lives on the PRODUCING COMMAND rather than on a skill
    #: because that is the only place every arm can read it: with
    #: `FW_PLAN_DECOMPOSITION=off` the loader never opens `_skills/`, so a
    #: skill-only rule would give the flat control arm citation alone while the
    #: decomposed arms got citation plus declaration — a mechanism asymmetry the
    #: endpoint would score as an effect of decomposition. A command flag is read
    #: identically in A, B and C.
    presentation: bool = False


def declare(spec: ResultHandleSpec) -> dict[str, Any]:
    """The artifact entry that opts a command's response into compaction.

    Merge into `CommandResponse.artifacts`::

        artifacts={**payload, **result_handles.declare(spec)}
    """
    return {RESULT_HANDLE_ARTIFACT_KEY: spec.model_dump(mode="json")}


def spec_from_artifacts(artifacts: Any) -> Optional[ResultHandleSpec]:
    """The declaration in a response's artifacts, or None if it did not opt in.

    A malformed declaration is reported and then ignored rather than raised: a
    command whose compaction metadata is wrong must still return its answer.
    Here "ignored" means none of its untrusted fields is admitted raw. The
    declaration remains an opt-in and becomes a classification-less spec whose
    prompt-facing fields fail closed to capture envelopes; returning None would
    restore the full response to the trajectory.
    """
    if not isinstance(artifacts, Mapping):
        return None
    raw = artifacts.get(RESULT_HANDLE_ARTIFACT_KEY)
    if raw is None:
        return None
    if isinstance(raw, ResultHandleSpec):
        return raw
    try:
        return ResultHandleSpec.model_validate(raw)
    except Exception as exc:
        logger.warning(
            f"failing closed malformed {RESULT_HANDLE_ARTIFACT_KEY}: {exc!r}"
        )
        candidate = dict(raw) if isinstance(raw, Mapping) else {}
        candidate["classification"] = None
        try:
            return ResultHandleSpec.model_validate(candidate)
        except Exception:
            return ResultHandleSpec(
                kind="withheld",
                summary=(
                    "Result metadata withheld because its declaration was "
                    "malformed or unclassified."
                ),
                ordering="withheld",
                classification=None,
            )


# ----------------------------------------------------------------------
# What the store keeps
# ----------------------------------------------------------------------


class ResultView(BaseModel):
    """One page of a stored result that was actually RENDERED to the agent.

    The resolver's whole bound depends on this record: "the rows the agent saw"
    is not derivable from the stored payload, which holds the population, nor
    from the trajectory, which holds a compact rendering the resolver would have
    to parse back. So the two seams that render a page — the first page in
    `compact_observation_for`, every later page in `fetch_page` — say so here,
    and `resolve_for_presentation` replays the slices.

    `count` rather than an end offset because a page can be short (the last one),
    and a resolver that recomputed the end from `page_size` would over-resolve
    exactly at the boundary where the agent saw least.
    """

    model_config = ConfigDict(extra="forbid")

    offset: int = 0
    count: int = 0
    #: The `contains` filter this view was taken under, if any. A page fetched
    #: under a filter is a page of the FILTERED rows, and replaying it against
    #: the population would resolve rows the agent never saw.
    contains: Optional[str] = None
    #: Policy-projected fetch-time filters. Unlike `contains`, this is safe to
    #: persist and is explicit about being view provenance rather than a
    #: producer filter.
    view_filters: Any = Field(default_factory=dict)
    #: Cursor identity of the normalized fetch-time filter.
    filter_identity: str = ""
    #: Exact producer-order item indices rendered in this page. Persisting the
    #: slice avoids replaying a raw filter after cold resume.
    item_indices: tuple[int, ...] = ()
    #: Every materialized item index matched by this view's filter, so the
    #: aggregate matched count remains exact across several distinct filters.
    matched_indices: tuple[int, ...] = ()
    matched: int = 0


class StoredResult(BaseModel):
    """One command's full payload, keyed by the producing command_call_id.

    `items`, `detail` and `response` are `Any` because the capture policy may
    have replaced any of them with an envelope (`__fw_capture__`). A withheld
    payload is kept — the envelope carries the size and digest — so the
    observation can say "withheld" instead of "empty".
    Summary, ordering, and filters use the same envelope-capable shape; treating
    them as harmless metadata would let producer-authored tenant text bypass the
    policy around otherwise withheld rows.
    """

    model_config = ConfigDict(extra="forbid")

    handle_id: str
    command_name: str
    kind: Any
    summary: Any
    ordering: Any
    total: int
    #: Explicit on new records; None on a restored pre-v2 record, where
    #: `is_source_complete` conservatively infers from total and item count.
    source_complete: Optional[bool] = None
    page_size: int
    filters: Any = Field(default_factory=dict)
    #: Whether the producing command applied a filter before materialization.
    #: Kept separately because the policy may replace the filter values with an
    #: envelope, while presentation scope still needs the boolean fact.
    producer_filter_applied: bool = False
    classification: Optional[DataClassification] = None
    items: Any = Field(default_factory=list)
    detail: Any = Field(default_factory=dict)
    #: The command's FULL response text, as the agent would have seen it before
    #: compaction. Kept so a command can hand the whole thing back deliberately.
    response: Any = ""
    response_bytes: int = 0
    response_digest: str = ""
    created_at: str = ""
    #: The logical turn that produced this handle. What makes "do not evict a
    #: handle the live turn is still using" expressible at all; None means the
    #: record was stored outside any turn, which nothing protects.
    turn_key: Optional[str] = None
    #: Bytes this record's payload occupies, against the session byte budget.
    #: Persisted rather than recomputed so a restored store accounts for exactly
    #: what the writing process accounted for — a re-encode could differ.
    stored_bytes: int = 0
    #: The producer's `presentation` flag, carried so the resolver can select on
    #: it without re-reading the command's artifacts.
    presentation: bool = False
    #: The pages of this result the agent was actually shown, in fetch order.
    #: Deliberately NOT charged against `stored_bytes`: a view is three small
    #: scalars, and re-measuring the record on every page would make a handle's
    #: cost rise because it was read, which `put`'s docstring rejects. Persisted
    #: with the record so a resumed turn still knows what its suspended half saw.
    views: list[ResultView] = Field(default_factory=list)

    @property
    def item_list(self) -> list[str]:
        """The rows, or [] when the policy withheld them."""
        return list(self.items) if isinstance(self.items, list) else []

    @property
    def payload_withheld(self) -> bool:
        """True when the capture policy replaced the rows with an envelope."""
        return not isinstance(self.items, list)

    @property
    def materialized_count(self) -> int:
        """Rows the handle can actually page, independently of backend total."""
        return len(self.item_list)

    @property
    def is_source_complete(self) -> bool:
        """Whether reaching the stored tail proves the backend source ended."""
        if self.source_complete is not None:
            return bool(self.source_complete) and (
                self.payload_withheld
                or self.total <= self.materialized_count
            )
        return self.total <= self.materialized_count

    @property
    def source_incomplete_reason(self) -> Optional[str]:
        """Typed reason a stored tail cannot establish full backend coverage."""
        return None if self.is_source_complete else SOURCE_INCOMPLETE_REASON


class ResultHandleStore:
    """Session-scoped, insertion-ordered, byte-bounded. Survives suspend/resume.

    Insertion order is eviction order, and `put` re-inserts, so a handle the
    agent keeps paging is not thereby made younger — the store bounds MEMORY, and
    a payload's cost does not fall because it is being read.
    """

    def __init__(
        self,
        max_entries: int = MAX_STORED_HANDLES,
        max_bytes: Optional[int] = None,
    ) -> None:
        self._records: dict[str, StoredResult] = {}
        self._max_entries = max_entries
        # None means "ask the environment on every put"; an explicit value pins
        # the budget, which is what a test wants and a deployment does not.
        self._max_bytes = max_bytes

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, handle_id: object) -> bool:
        return handle_id in self._records

    @property
    def total_bytes(self) -> int:
        return sum(record.stored_bytes for record in self._records.values())

    def _budget(self) -> int:
        return self._max_bytes if self._max_bytes is not None else max_stored_bytes()

    def _oldest_other_turn(self, current_turn_key: Optional[str], keep: str):
        """The oldest handle that is neither `keep` nor from the live turn."""
        for handle_id, record in self._records.items():
            if handle_id == keep:
                continue
            if current_turn_key is not None and record.turn_key == current_turn_key:
                continue
            return handle_id
        return None

    def _oldest_other(self, keep: str):
        """The oldest handle that is not `keep`, live turn or not."""
        return next((h for h in self._records if h != keep), None)

    def put(
        self, record: StoredResult, *, current_turn_key: Optional[str] = None
    ) -> tuple[str, ...]:
        """Store one payload; return the handle ids this displaced, oldest first.

        Two passes, in this order and not the other, because they answer to
        different rules:

        1. **The count ceiling** — a distant backstop — never touches a handle
           the current turn produced. ido-mn1.6.6 resolves cited handles at the
           END of a turn, so evicting one mid-turn turns a citation into a
           missing table rather than into a retry.
        2. **The byte budget** is a hard bound and therefore may reach into the
           live turn, oldest first, once everything older is gone. The record
           just stored is never the victim while any other remains: dropping the
           payload whose observation the agent is about to read would make the
           very next step unresolvable.
        """
        record = _policy_project_record(record)
        self._records.pop(record.handle_id, None)
        self._records[record.handle_id] = record

        evicted: list[str] = []
        while len(self._records) > self._max_entries:
            victim = self._oldest_other_turn(current_turn_key, record.handle_id)
            if victim is None:
                break
            evicted.append(victim)
            self._records.pop(victim)

        budget = self._budget()
        while self.total_bytes > budget and len(self._records) > 1:
            victim = self._oldest_other_turn(
                current_turn_key, record.handle_id
            ) or self._oldest_other(record.handle_id)
            if victim is None:
                break
            evicted.append(victim)
            self._records.pop(victim)

        if evicted:
            logger.warning(
                "result handle store evicted %d handle(s) to stay within "
                "%d bytes / %d entries: %s",
                len(evicted),
                budget,
                self._max_entries,
                ", ".join(evicted),
            )
        return tuple(evicted)

    def get(self, handle_id: str) -> StoredResult:
        record = self._records.get(handle_id)
        if record is None:
            raise UnknownResultHandle(
                f"no stored result for handle {handle_id!r}; it was never issued "
                "or has expired. Re-run the command that produced it."
            )
        projected = _policy_project_record(record)
        self._records[handle_id] = projected
        return projected

    def peek(self, handle_id: str) -> Optional[StoredResult]:
        record = self._records.get(handle_id)
        if record is None:
            return None
        projected = _policy_project_record(record)
        self._records[handle_id] = projected
        return projected

    def record_view(self, handle_id: str, page: "ResultPage") -> None:
        """Remember that `page` of `handle_id` was rendered to the agent.

        Idempotent on an identical slice — an agent re-fetching the same page
        records nothing new — and a no-op for a handle this store does not hold,
        because a view of an evicted result is not evidence of anything and
        raising here would fail a rendering over bookkeeping. Never re-inserts
        the record: reading a handle must not make it younger (see `put`).
        """
        record = self._records.get(handle_id)
        if record is None:
            return
        view = ResultView(
            offset=page.offset,
            count=len(page.items),
            view_filters=page.view_filters,
            filter_identity=page.filter_identity,
            item_indices=page.item_indices,
            matched_indices=page.matched_indices,
            matched=page.matched,
        )
        if view in record.views or len(record.views) >= MAX_RECORDED_VIEWS:
            return
        record.views.append(view)

    def handle_ids(self) -> tuple[str, ...]:
        """Every handle held, oldest first — which is eviction order."""
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()

    def to_state(self) -> list[dict[str, Any]]:
        """JSON-able, oldest first, so restore rebuilds the same eviction order."""
        return [
            self.get(handle_id).model_dump(mode="json")
            for handle_id in self._records
        ]

    def apply_state(self, state: Any) -> None:
        """Replace the contents from `to_state()` output. Never raises on junk.

        A blob whose handle records this build cannot read means the agent gets
        `UnknownResultHandle` and re-runs a command — recoverable — whereas
        raising here would fail an otherwise restorable turn.
        """
        self._records.clear()
        if not isinstance(state, list):
            return
        for item in state:
            try:
                record = StoredResult.model_validate(item)
            except Exception as exc:
                logger.warning(f"dropping unreadable result handle: {exc!r}")
                continue
            # `current_turn_key=record.turn_key` so a restore of a suspended
            # turn's own handles cannot evict them on the way back in: the whole
            # point of persisting them is that the resumed half of the turn still
            # cites them. The byte budget still applies.
            self.put(record, current_turn_key=record.turn_key)


# ----------------------------------------------------------------------
# Cursors
# ----------------------------------------------------------------------


def _normalized_cursor_identity(
    handle_id: str,
    filters: Optional[Mapping[str, Any]] = None,
) -> str:
    """Digest the handle/version and the filters under their matching semantics."""
    normalized_filters = sorted(
        (
            str(key).strip().casefold(),
            str(value).strip().casefold(),
        )
        for key, value in (filters or {}).items()
        if value is not None and str(value).strip()
    )
    payload = {
        "version": CURSOR_VERSION,
        "handle": str(handle_id).strip().casefold(),
        "filters": normalized_filters,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_cursor(
    offset: int,
    *,
    handle_id: str,
    filters: Optional[Mapping[str, Any]] = None,
) -> str:
    """A versioned offset bound to one handle and normalized filter identity."""
    identity = _normalized_cursor_identity(handle_id, filters)
    return f"{CURSOR_VERSION}:{int(offset)}:{identity}"


def decode_cursor(
    cursor: Optional[str],
    *,
    handle_id: str,
    filters: Optional[Mapping[str, Any]] = None,
) -> int:
    """The bound offset a cursor names. None/empty restarts at the first page."""
    if cursor is None:
        return 0
    text = str(cursor).strip()
    if not text:
        return 0
    parts = text.split(":")
    if len(parts) != 3:
        raise InvalidResultCursor(
            f"cursor {cursor!r} is not a cursor; expected "
            f"'{CURSOR_VERSION}:<offset>:<identity>'"
        )
    version, offset, identity = parts
    if version != CURSOR_VERSION:
        raise InvalidResultCursor(
            f"cursor {cursor!r} was written at version {version!r}; this build "
            f"reads {CURSOR_VERSION!r}. Re-run the command to get a fresh handle."
        )
    expected = _normalized_cursor_identity(handle_id, filters)
    if identity != expected:
        raise InvalidResultCursor(
            f"cursor {cursor!r} does not belong to this result handle and "
            "normalized filter. Omit the cursor to restart filtering."
        )
    try:
        value = int(offset)
    except ValueError as exc:
        raise InvalidResultCursor(
            f"cursor {cursor!r} has a non-integer offset"
        ) from exc
    if value < 0:
        raise InvalidResultCursor(f"cursor {cursor!r} has a negative offset")
    return value


# ----------------------------------------------------------------------
# The observation contract
# ----------------------------------------------------------------------

#: Bumped when the rendered observation's field set changes, and carried in the
#: rendering itself: a trajectory recorded before a change and one recorded
#: after are otherwise indistinguishable to a reader.
OBSERVATION_CONTRACT_VERSION = "2"


def _render_policy_value(value: Any) -> str:
    """Stable prompt text for a value or a capture envelope."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _render_filter_value(value: Any) -> str:
    """Readable filters without assuming the policy left a mapping."""
    if isinstance(value, Mapping) and not _is_capture_envelope(value):
        return (
            ", ".join(
                f"{key}={_render_policy_value(item)}"
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
            or "none"
        )
    return _render_policy_value(value) if value else "none"


def _combine_filter_provenance(producer_filters: Any, view_filters: Any) -> Any:
    """Compatibility rendering plus explicit producer/view fields."""
    if (
        isinstance(producer_filters, Mapping)
        and not _is_capture_envelope(producer_filters)
        and isinstance(view_filters, Mapping)
        and not _is_capture_envelope(view_filters)
    ):
        overlap = set(producer_filters) & set(view_filters)
        if not overlap:
            return {**producer_filters, **view_filters}
    if not producer_filters:
        return view_filters
    if not view_filters:
        return producer_filters
    return {"producer": producer_filters, "view": view_filters}


@dataclass(frozen=True)
class ResultPage:
    """One page of a stored result — the exact agent-facing contract.

    Every field here appears in `as_observation()`. Nothing else does: the full
    payload reaches evidence by a different road (`fw.command.execute`, the turn
    record) and must not leak back into the trajectory through this rendering.
    """

    handle_id: str
    kind: Any
    summary: Any
    ordering: Any
    #: Rows matching the filters, across all pages. Equals `total` unfiltered.
    #: The historical equality above applies only to a completely materialized
    #: source; the v2 completeness fields below make the partial case explicit.
    #: Rows matching the filters in the materialized source. Complete only when
    #: `matched_complete` is True.
    matched: int
    #: What the backend said the population is. Can exceed `matched`.
    total: int
    #: Rows actually held by this handle before a fetch-time filter.
    materialized: int
    #: Whether the producer materialized every backend row.
    source_complete: bool
    #: Whether `matched` is an exact count rather than a lower bound.
    matched_complete: bool
    offset: int
    items: tuple[str, ...]
    has_more: bool
    #: The cursor for the NEXT page, or None when this is the last one.
    next_cursor: Optional[str]
    #: Producer filters plus anything this fetch added.
    filters: Any
    #: `cursor`, `complete`, or `source-incomplete`.
    continuation: str
    producer_filters: Any
    view_filters: Any
    filter_identity: str
    item_indices: tuple[int, ...]
    matched_indices: tuple[int, ...]
    #: Typed when the backend reported rows the handle does not materialize.
    incomplete_reason: Optional[str] = None
    #: True when the capture policy withheld the rows from the store.
    withheld: bool = False

    def as_observation(self) -> str:
        shown = (
            f"{self.offset + 1}-{self.offset + len(self.items)}"
            if self.items
            else "none"
        )
        filters = _render_filter_value(self.filters)
        header = [
            _render_policy_value(self.summary),
            (
                f"result_handle={self.handle_id} "
                f"kind={_render_policy_value(self.kind)} "
                f"contract=v{OBSERVATION_CONTRACT_VERSION}"
            ),
            (
                f"total={self.total} materialized={self.materialized} "
                f"source_complete={'true' if self.source_complete else 'false'}"
            ),
            (
                f"matched={self.matched} "
                f"matched_complete={'true' if self.matched_complete else 'false'} "
                f"shown={shown} "
                f"has_more={'true' if self.has_more else 'false'}"
            ),
            f"continuation={self.continuation}",
            f"ordering={_render_policy_value(self.ordering)}",
            f"filters={filters}",
            f"producer_filters={_render_filter_value(self.producer_filters)}",
            f"view_filters={_render_filter_value(self.view_filters)}",
        ]
        if self.next_cursor:
            header.append(
                f"next_cursor={self.next_cursor} "
                "(fetch the next page with this handle and cursor)"
            )
        if self.incomplete_reason:
            header.append(
                f"incomplete_source={self.incomplete_reason}; the stored handle "
                "cannot fetch the unseen backend rows. Narrow or re-run the "
                "producing command with backend pagination and treat this result "
                "as partial."
            )
        if self.withheld:
            header.append(
                "rows=withheld by the capture policy; the full response is in "
                "the execution record, not in this observation"
            )
        return "\n".join(header + list(self.items))


def page_of(
    record: StoredResult,
    *,
    cursor: Optional[str] = None,
    contains: Optional[str] = None,
    page_size: Optional[int] = None,
) -> ResultPage:
    """Slice one page out of a stored result. Pure; the store is not consulted.

    `contains` is a case-insensitive substring test over the rendered row. It is
    deliberately the only filter: a richer predicate would have to be evaluated
    on structure this module does not model, and a filter the agent can express
    but the store cannot honor exactly is worse than none.
    """
    record = _policy_project_record(record)
    indexed_rows = list(enumerate(record.item_list))
    contains_text = str(contains).strip() if contains is not None else ""
    view_filter_raw: dict[str, str] = {}
    if contains_text:
        needle = contains_text.casefold()
        indexed_rows = [
            (index, row)
            for index, row in indexed_rows
            if needle in row.casefold()
        ]
        view_filter_raw["contains"] = contains_text

    view_filters = _policy_applier(
        record.command_name,
        record.classification,
    )("view_filters", view_filter_raw)
    producer_filters = record.filters
    filters = _combine_filter_provenance(producer_filters, view_filters)

    # Decode after the effective filter exists. The cursor identity includes
    # this mapping, so changing `contains` while reusing a cursor is rejected;
    # omitting the cursor intentionally restarts the new filter at row zero.
    offset = decode_cursor(
        cursor,
        handle_id=record.handle_id,
        filters=view_filter_raw,
    )

    size = _clamp_page_size(page_size if page_size is not None else record.page_size)
    window_pairs = tuple(indexed_rows[offset : offset + size])
    window = tuple(row for _index, row in window_pairs)
    end = offset + len(window)
    materialized_more = end < len(indexed_rows)
    source_complete = record.is_source_complete
    has_more = materialized_more or not source_complete
    continuation = (
        "cursor"
        if materialized_more
        else "complete"
        if source_complete
        else "source-incomplete"
    )
    return ResultPage(
        handle_id=record.handle_id,
        kind=record.kind,
        summary=record.summary,
        ordering=record.ordering,
        matched=len(indexed_rows),
        # ALWAYS the producer's declared backend total, filtered or not. The two
        # numbers answer two different questions and collapsing them loses one of
        # them either way round: `total` alone lets the agent read a filtered
        # page as covering the population, and `matched` alone hides from an
        # evaluator how much of the population the filter excluded. Carrying both
        # on every page is what makes `filters=` legible rather than decorative.
        total=record.total,
        materialized=record.materialized_count,
        source_complete=source_complete,
        matched_complete=source_complete and not record.payload_withheld,
        offset=offset,
        items=window,
        has_more=has_more,
        next_cursor=(
            encode_cursor(
                end,
                handle_id=record.handle_id,
                filters=view_filter_raw,
            )
            if materialized_more
            else None
        ),
        filters=filters,
        continuation=continuation,
        producer_filters=producer_filters,
        view_filters=view_filters,
        filter_identity=_normalized_cursor_identity(
            record.handle_id,
            view_filter_raw,
        ),
        item_indices=tuple(index for index, _row in window_pairs),
        matched_indices=tuple(index for index, _row in indexed_rows),
        incomplete_reason=record.source_incomplete_reason,
        withheld=record.payload_withheld,
    )


def _clamp_page_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return DEFAULT_PAGE_SIZE
    return max(1, min(size, MAX_PAGE_SIZE))


# ----------------------------------------------------------------------
# Host resolution (duck-typed, exactly like `tracing`)
# ----------------------------------------------------------------------


def store_for(host: Any = None) -> Optional[ResultHandleStore]:
    """The session's handle store, or None when the host does not keep one.

    `host` defaults to the trace host bound by `CommandExecutor.invoke_command`,
    which is what lets a command's `ResponseGenerator` — several frames down,
    holding only a `Workflow` — reach the store without a new parameter on every
    command signature.
    """
    if host is None:
        host = tracing.current_host()
    if host is None:
        return None
    store = getattr(host, "result_handles", None)
    if isinstance(store, ResultHandleStore):
        return store
    core = getattr(host, "_core", None)
    if core is not None:
        store = getattr(core, "result_handles", None)
        if isinstance(store, ResultHandleStore):
            return store
    return None


# ----------------------------------------------------------------------
# The public API a workflow command wraps
# ----------------------------------------------------------------------


def fetch_page(
    handle_id: str,
    *,
    cursor: Optional[str] = None,
    contains: Optional[str] = None,
    page_size: Optional[int] = None,
    host: Any = None,
) -> ResultPage:
    """Page / filter a previously issued handle. Raises `ResultHandleError`.

    This is the seam a workflow wraps in its own `fetch_result_page` command:
    fastWorkflow deliberately does NOT register one as a core command, because a
    core command joins every workflow's command surface and therefore every
    workflow's trained intent model and `what_can_i_do` output — which would
    make the off-path parity claim false for workflows that never opt in.
    """
    store = store_for(host)
    if store is None:
        raise UnknownResultHandle(
            "this session keeps no result handles; the command that would have "
            "produced one did not opt in, or ran outside a workflow session"
        )
    page = page_of(
        store.get(handle_id), cursor=cursor, contains=contains, page_size=page_size
    )
    # Recorded HERE rather than in `page_of`, which is pure and is also called by
    # readers that are not showing the agent anything. This function is the seam
    # a workflow's paging command wraps, so reaching it means a page is on its
    # way into the trajectory (ido-mn1.6.6).
    store.record_view(handle_id, page)
    return page


def get_result(handle_id: str, *, host: Any = None) -> StoredResult:
    """The whole stored record. Raises `UnknownResultHandle`."""
    store = store_for(host)
    if store is None:
        raise UnknownResultHandle("this session keeps no result handles")
    return store.get(handle_id)


def full_response(handle_id: str, *, host: Any = None) -> str:
    """The producing command's complete response text, if the policy kept it."""
    record = get_result(handle_id, host=host)
    return record.response if isinstance(record.response, str) else ""


# ----------------------------------------------------------------------
# Framework seams
# ----------------------------------------------------------------------


#: What `store_from_command_output` returns: the handle it issued (None when the
#: command did not opt in) and the handles that storing it displaced. The second
#: half exists so the caller can put it in evidence — an eviction nobody recorded
#: makes a later `UnknownResultHandle` undiagnosable.
StoreOutcome = tuple[Optional[str], tuple[str, ...]]


def store_from_command_output(host: Any, command_output: Any) -> StoreOutcome:
    """Store a command's payload if it opted in.

    Called from `CommandExecutor.invoke_command` after the call id is stamped, so
    the handle IS that call id. Never raises: capture must not fail a command.
    """
    try:
        response = getattr(command_output, "command_response", None)
        spec = spec_from_artifacts(getattr(response, "artifacts", None))
        if spec is None:
            return None, ()
        handle_id = getattr(command_output, "command_call_id", None)
        if not handle_id:
            return None, ()
        store = store_for(host)
        if store is None:
            return None, ()
        turn_key = _turn_key_of(host)
        evicted = store.put(
            _record_for(handle_id, spec, command_output, response, turn_key),
            current_turn_key=turn_key,
        )
        return handle_id, evicted
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"result handle capture failed: {exc!r}")
        return None, ()


def _turn_key_of(host: Any) -> Optional[str]:
    """The logical turn in flight, or None. Duck-typed and never raising."""
    try:
        return tracing.get_turn_key(host)
    except Exception:  # pragma: no cover - defensive
        return None


def _record_for(handle_id, spec, command_output, response, turn_key) -> StoredResult:
    text = getattr(response, "response", "") or ""
    raw = text.encode("utf-8")
    command_name = getattr(command_output, "command_name", "") or "unknown"
    apply_policy = _policy_applier(command_name, spec.classification)
    items = apply_policy("items", list(spec.items))
    detail = apply_policy("detail", dict(spec.detail))
    stored_response = apply_policy("response", text)
    summary = apply_policy("summary", spec.summary)
    kind = apply_policy("kind", spec.kind)
    ordering = apply_policy("ordering", spec.ordering)
    filters = apply_policy("filters", dict(spec.filters))
    total = spec.total if spec.total is not None else len(spec.items)
    source_complete = (
        total <= len(spec.items)
        if spec.source_complete is None
        else bool(spec.source_complete) and total <= len(spec.items)
    )
    return StoredResult(
        handle_id=handle_id,
        command_name=command_name,
        kind=kind,
        summary=summary,
        ordering=ordering,
        total=total,
        source_complete=source_complete,
        page_size=_clamp_page_size(spec.page_size),
        filters=filters,
        producer_filter_applied=bool(spec.filters),
        classification=spec.classification,
        presentation=bool(spec.presentation),
        items=items,
        detail=detail,
        response=stored_response,
        response_bytes=len(raw),
        response_digest=hashlib.sha256(raw).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(),
        turn_key=turn_key,
        # Measured on what is actually KEPT, after the capture policy has run: a
        # withheld payload is an envelope of a few hundred bytes and must not be
        # charged for the megabytes it stands in for.
        stored_bytes=json_size(
            [kind, summary, ordering, filters, items, detail, stored_response]
        ),
    )


def _is_capture_envelope(value: Any) -> bool:
    return isinstance(value, dict) and value.get("__fw_capture__") is True


def _fail_closed_envelope(value: Any, *, reason: str) -> Any:
    """A capture-policy-compatible omission when no safe policy result exists."""
    if value is None or _is_capture_envelope(value):
        return value
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    except Exception:
        encoded = repr(value).encode("utf-8", errors="ignore")
    return {
        "__fw_capture__": True,
        "classification": None,
        "disposition": "omit",
        "original_bytes": len(encoded),
        "digest": f"sha256:{hashlib.sha256(encoded).hexdigest()[:16]}",
        "reason": reason,
        "policy_version": "1",
        "retention_class": "diagnostic",
    }


def _policy_applier(
    command_name: str,
    classification: Optional[DataClassification],
):
    """A `field -> value -> value` projection under the process capture policy.

    Resolved lazily and defensively. Under the `debug` profile — the default —
    `CapturePolicy.apply` returns every value whole, so this seam is a no-op and
    the stored payload is byte-identical to what the command produced. Under
    `evidence`, an unclassified payload is withheld, which is the same
    default-deny the trace sink applies; the observation then says so rather than
    reporting an empty result.

    `for_prompt=True` because this payload's destination IS the agent's prompt.
    """
    if classification is None:
        return lambda field, value: _fail_closed_envelope(
            value,
            reason=(
                "omitted because the result-handle classification is missing "
                f"or invalid ({field})"
            ),
        )
    try:
        from fastworkflow.observability_store import resolve_capture_policy

        policy = resolve_capture_policy()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"capture policy unavailable for result handles: {exc!r}")
        return lambda field, value: _fail_closed_envelope(
            value,
            reason=f"omitted because capture policy was unavailable ({field})",
        )

    def apply(field: str, value: Any) -> Any:
        if _is_capture_envelope(value):
            return value
        try:
            return policy.apply(
                f"command.{command_name}.result_handle.{field}",
                value,
                classification=classification,
                for_prompt=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"capture policy failed on {field}: {exc!r}")
            return _fail_closed_envelope(
                value,
                reason=f"omitted because capture policy failed ({field})",
            )

    return apply


def _same_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


def _policy_project_record(record: StoredResult) -> StoredResult:
    """Apply the prompt policy at every store/read boundary, idempotently."""
    apply_policy = _policy_applier(record.command_name, record.classification)
    raw_rows = record.item_list
    projected_views: list[ResultView] = []
    for view in record.views:
        raw_view_filters = (
            {"contains": view.contains}
            if view.contains
            else {}
        )
        item_indices = tuple(view.item_indices)
        matched_indices = tuple(view.matched_indices)
        if not view.filter_identity and not item_indices and raw_rows:
            if view.contains:
                needle = view.contains.casefold()
                candidates = tuple(
                    index
                    for index, row in enumerate(raw_rows)
                    if needle in row.casefold()
                )
            else:
                candidates = tuple(range(len(raw_rows)))
            item_indices = tuple(
                candidates[view.offset : view.offset + view.count]
            )
            matched_indices = candidates
        filter_identity = view.filter_identity or _normalized_cursor_identity(
            record.handle_id,
            raw_view_filters,
        )
        source_view_filters = raw_view_filters or view.view_filters
        projected_filters = (
            apply_policy("view_filters", source_view_filters)
            if source_view_filters
            else source_view_filters
        )
        projected_views.append(
            view.model_copy(
                update={
                    "contains": None,
                    "view_filters": projected_filters,
                    "filter_identity": filter_identity,
                    "item_indices": item_indices,
                    "matched_indices": matched_indices,
                    "matched": (
                        len(matched_indices)
                        if matched_indices
                        else view.matched
                    ),
                }
            )
        )

    projected = {
        "kind": apply_policy("kind", record.kind),
        "summary": apply_policy("summary", record.summary),
        "ordering": apply_policy("ordering", record.ordering),
        "filters": apply_policy("filters", record.filters),
        "items": apply_policy("items", record.items),
        "detail": apply_policy("detail", record.detail),
        "response": apply_policy("response", record.response),
    }
    changed = any(
        not _same_value(projected[field], getattr(record, field))
        for field in projected
    )
    producer_filter_applied = record.producer_filter_applied or (
        isinstance(record.filters, Mapping)
        and not _is_capture_envelope(record.filters)
        and bool(record.filters)
    )
    stored_bytes = record.stored_bytes
    if changed:
        stored_bytes = json_size(
            [
                projected["summary"],
                projected["kind"],
                projected["ordering"],
                projected["filters"],
                projected["items"],
                projected["detail"],
                projected["response"],
            ]
        )
    return record.model_copy(
        update={
            **projected,
            "producer_filter_applied": producer_filter_applied,
            "views": projected_views,
            "stored_bytes": stored_bytes,
        }
    )


def compact_observation_for(host: Any, command_output: Any) -> Optional[str]:
    """The compact observation for an opted-in command's output, or None.

    None means "this command did not opt in, or nothing was stored" — and the
    caller must then return the full response text unchanged, which is the whole
    off-path guarantee.
    For an opted-in declaration, a storage/rendering failure returns a fixed
    withholding observation instead. Falling back to None there would expose
    exactly the raw payload the declaration asked this seam to compact.
    """
    handle_id = getattr(command_output, "command_call_id", None)
    try:
        response = getattr(command_output, "command_response", None)
        if spec_from_artifacts(getattr(response, "artifacts", None)) is None:
            return None
        store = store_for(host)
        if not handle_id or store is None:
            return _withheld_observation(handle_id, "result store unavailable")
        record = store.peek(handle_id)
        if record is None:
            return _withheld_observation(handle_id, "result payload unavailable")
        page = page_of(record)
        store.record_view(handle_id, page)
        return page.as_observation()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"result handle rendering failed: {exc!r}")
        return _withheld_observation(handle_id, "result rendering failed")


def _withheld_observation(handle_id: Any, reason: str) -> str:
    """Fixed fail-closed observation for an opted-in payload."""
    return "\n".join(
        (
            "Result payload withheld by the capture policy.",
            (
                f"result_handle={handle_id or 'unavailable'} "
                f"contract=v{OBSERVATION_CONTRACT_VERSION}"
            ),
            f"rows=withheld; reason={reason}",
            "The full response remains in the execution record, not in this observation.",
        )
    )


def json_size(value: Any) -> int:
    """Bytes of the canonical JSON encoding; what the byte budget counts.

    Falls back to `repr` for anything JSON cannot encode. A payload whose size
    could not be measured must still be CHARGED for — returning 0 would let an
    unmeasurable value sit in the store forever while the budget said the store
    was empty.
    """
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(value)
    return len(encoded.encode("utf-8", errors="ignore"))


def iter_handles(host: Any = None) -> Iterable[str]:
    """Handle ids this session still holds, oldest first."""
    store = store_for(host)
    return () if store is None else store.handle_ids()


# ----------------------------------------------------------------------
# Presentation resolution (ido-mn1.6.6)
# ----------------------------------------------------------------------
#
# The endpoint rates what the final answer PRESENTS. Compaction moved the rows
# out of the trajectory, so the extraction call that composes that answer can no
# longer list them — not because the runtime lost them, but because the model
# never saw them. That is a measurement artefact, and this section is the repair:
# at extraction time, resolve the payloads of the handles the answer must present
# and hand them to the extraction call as a dedicated input field.
#
# What is NOT done here, deliberately:
#
# * The trajectory is not re-inflated. The compact observations stay compact —
#   they are what bounded the loop's context in the first place, and re-expanding
#   them would give back every step's multiplied cost to save one call's.
# * Not everything is resolved. `iter_handles` may hold 200 records in an Arm C
#   turn; resolving them all would be the 922k-character prompt with extra steps.
#   Two rules select, and both are things the RUNTIME can check rather than
#   things the model asserts: the agent cited the handle (ido-mn1.6.4 tells it to
#   cite ids on the closing step), or the PRODUCING COMMAND declared its rows
#   deliverable (`ResultHandleSpec.presentation`). The second lives on the
#   command, not on a skill, because arm A runs with `FW_PLAN_DECOMPOSITION=off`
#   and never opens `_skills/`: a skill-only rule would have given the control
#   arm citation alone while the decomposed arms got citation plus declaration,
#   and the endpoint would have read that mechanism gap as an effect of
#   decomposition. A skill's `presents:` list is an optional override on top,
#   able to narrow the flagged set or add a command the producer did not flag.
# * The population is not resolved in place of the view. A handle whose page the
#   agent read once resolves to that page, not to all 477 rows: the answer is
#   bounded by the request, and an answer that lists rows nobody asked for is a
#   different failure from the one this fixes.

#: What `ResultPage.as_observation()` writes, and therefore what a citation of a
#: handle looks like in agent text. Kept beside the bare form rather than folded
#: into it so that the rendered contract is visibly the thing being parsed back.
_HANDLE_CITATION = re.compile(r"result_handle\s*=\s*([0-9a-fA-F]{32})")
#: A bare id. `tracing.new_command_call_id()` is `uuid.uuid4().hex`, so 32 hex
#: characters with no separators; the word boundaries keep this from matching
#: inside a longer digest.
_BARE_HANDLE = re.compile(r"\b[0-9a-fA-F]{32}\b")


def parse_handle_ids(text: Any) -> tuple[str, ...]:
    """Handle ids mentioned in a piece of agent text, first mention first.

    The inverse of `as_observation`, and it lives beside it for that reason: the
    rendering and the parse of a citation are one contract, and splitting them
    across modules is how the two drift.

    Case is normalized because a model that retypes an id will sometimes upcase
    it, and refusing a citation over letter case would turn a correct citation
    into a missing table.
    """
    if not text:
        return ()
    blob = text if isinstance(text, str) else str(text)
    hits: list[tuple[int, str]] = [
        (match.start(1), match.group(1).lower())
        for match in _HANDLE_CITATION.finditer(blob)
    ]
    hits.extend(
        (match.start(), match.group(0).lower()) for match in _BARE_HANDLE.finditer(blob)
    )
    # By position, not by pattern: the two patterns overlap on every rendered
    # citation, and ordering by pattern would report the second id in a thought
    # before the first whenever only one of them carried the `result_handle=`
    # prefix. Order is the cap's allocation order, so it has to be the text's.
    found: list[str] = []
    seen: set[str] = set()
    for _, handle_id in sorted(hits):
        if handle_id not in seen:
            seen.add(handle_id)
            found.append(handle_id)
    return tuple(found)


#: Why a handle was resolved. Recorded per entry so evidence says which rule
#: admitted it, not merely that something was admitted.
REASON_CITED = "cited"
REASON_PRESENTS = "presents"

#: How the rows were bounded. `fetched-pages` replays the views the agent was
#: actually shown; `producer-filtered` takes every row the PRODUCING command's
#: own filters matched, which is only ever wider than a page when the command
#: narrowed the population itself.
SCOPE_FETCHED = "fetched-pages"
SCOPE_PRODUCER_FILTERED = "producer-filtered"


def _truncate_utf8(text: Any, max_bytes: int, marker: str = "") -> tuple[str, bool]:
    """Bound text on a character boundary and leave an explicit marker."""
    value = str(text)
    encoded = value.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return value, False
    marker_bytes = marker.encode("utf-8", errors="ignore")
    if len(marker_bytes) >= max_bytes:
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore"), True
    prefix = encoded[: max_bytes - len(marker_bytes)].decode(
        "utf-8", errors="ignore"
    )
    return prefix + marker, True


def _bounded_metadata(value: Any) -> tuple[str, bool]:
    """One producer-authored header field under its local starvation guard."""
    marker = (
        f"[… classification={PRESENTATION_TRUNCATION_CLASSIFICATION} "
        "metadata-truncated]"
    )
    return _truncate_utf8(value, MAX_PRESENTATION_METADATA_BYTES, marker)


def _bounded_filters(filters: Any) -> tuple[Any, bool]:
    """A deterministic bounded rendering copy of producer filter metadata."""
    if not isinstance(filters, Mapping) or _is_capture_envelope(filters):
        rendered, trimmed = _bounded_metadata(_render_policy_value(filters))
        return ({"…": rendered} if trimmed else filters), trimmed
    rendered: dict[str, str] = {}
    trimmed = False
    spent = 0
    for key, value in sorted(filters.items(), key=lambda item: str(item[0])):
        bounded_key, key_trimmed = _bounded_metadata(key)
        bounded_value, value_trimmed = _bounded_metadata(value)
        cost = len(f"{bounded_key}={bounded_value}, ".encode("utf-8"))
        if spent + cost > MAX_PRESENTATION_METADATA_BYTES:
            trimmed = True
            break
        rendered[bounded_key] = bounded_value
        spent += cost
        trimmed = trimmed or key_trimmed or value_trimmed
    if trimmed:
        rendered["…"] = (
            f"classification={PRESENTATION_TRUNCATION_CLASSIFICATION}; "
            "metadata-truncated"
        )
    return rendered, trimmed


def _bounded_view_filters(
    views: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Bound each view's policy-projected filter metadata."""
    bounded: list[dict[str, Any]] = []
    trimmed = False
    for view in views:
        filters, filters_trimmed = _bounded_filters(view.get("filters", {}))
        bounded.append({**view, "filters": filters})
        trimmed = trimmed or filters_trimmed
    return tuple(bounded), trimmed


@dataclass(frozen=True)
class ResolvedResult:
    """One handle's rows, as the extraction call will see them."""

    handle_id: str
    command_name: str
    kind: str
    summary: Any
    ordering: Any
    scope: str
    reasons: tuple[str, ...]
    filters: Any
    view_filters: tuple[dict[str, Any], ...]
    total: int
    materialized: int
    matched: int
    matched_complete: bool
    source_complete: bool
    incomplete_reason: Optional[str]
    rows: tuple[str, ...]
    #: Rows the scope selected, before the byte cap trimmed. `>= len(rows)`.
    rows_available: int
    trimmed: bool
    metadata_trimmed: bool
    withheld: bool
    bytes: int

    def as_text(self) -> str:
        filters = _render_filter_value(self.filters)
        header = [
            (
                f"result_handle={self.handle_id} command={self.command_name} "
                f"kind={_render_policy_value(self.kind)} "
                f"reason={'+'.join(self.reasons)}"
            ),
            _render_policy_value(self.summary),
            (
                f"total={self.total} materialized={self.materialized} "
                f"source_complete={'true' if self.source_complete else 'false'}"
            ),
            (
                f"matched={self.matched} "
                f"matched_complete={'true' if self.matched_complete else 'false'} "
                f"rows_here={len(self.rows)} of {self.rows_available} "
                f"scope={self.scope}"
            ),
            f"ordering={_render_policy_value(self.ordering)}",
            f"filters={filters}",
            "view_filters="
            + (
                json.dumps(
                    self.view_filters,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                if self.view_filters
                else "none"
            ),
        ]
        if self.withheld:
            header.append(
                "rows=withheld by the capture policy; do not claim rows this "
                "field does not carry"
            )
        if self.incomplete_reason:
            header.append(
                f"incomplete_source={self.incomplete_reason}; rows absent from "
                "the materialized handle are not available to extraction. Treat "
                "the result as partial."
            )
        lines = header + list(self.rows)
        if self.trimmed:
            lines.append(
                f"[classification={PRESENTATION_TRUNCATION_CLASSIFICATION} "
                f"reason={PRESENTATION_TRUNCATION_REASON}; "
                f"{self.rows_available - len(self.rows)} further row(s) of this "
                "result are not shown here; say the listing is partial rather "
                "than implying it is complete]"
            )
        if self.metadata_trimmed:
            lines.append(
                f"[classification={PRESENTATION_TRUNCATION_CLASSIFICATION} "
                f"reason={PRESENTATION_TRUNCATION_REASON}; producer metadata "
                "was shortened and must not be treated as complete]"
            )
        return "\n".join(lines)

    def as_evidence(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "command": self.command_name,
            "reason": "+".join(self.reasons),
            "scope": self.scope,
            "rows": len(self.rows),
            "rows_available": self.rows_available,
            "materialized": self.materialized,
            "source_complete": self.source_complete,
            "matched": self.matched,
            "matched_complete": self.matched_complete,
            "incomplete_reason": self.incomplete_reason,
            "view_filters": list(self.view_filters),
            "bytes": self.bytes,
            "trimmed": self.trimmed,
            "metadata_trimmed": self.metadata_trimmed,
            "truncation_classification": (
                PRESENTATION_TRUNCATION_CLASSIFICATION
                if self.trimmed or self.metadata_trimmed
                else None
            ),
            "withheld": self.withheld,
        }


#: What the extraction call sees when nothing was resolved. A sentence rather
#: than an empty string: an empty input field reads to a model as a field it may
#: fill, and this one is the runtime speaking.
PRESENTED_RESULTS_NONE = (
    "(none — no result handle was cited or marked for presentation on this turn; "
    "compose the answer from the trajectory alone)"
)


@dataclass(frozen=True)
class PresentedResults:
    """The resolved payload plus the evidence of what resolving it selected.

    `total_bytes` is what the CAP counted (rows only). `field_bytes` is what the
    extraction call actually receives, headers and trim notices included, and is
    the number ido-mn1.6.10 sizes the per-call completion limit from: an answer
    asked to render this field cannot be shorter than the field, and a limit
    derived from the row bytes alone would under-size every multi-handle turn.
    """

    entries: tuple[ResolvedResult, ...] = ()
    total_bytes: int = 0
    max_bytes: int = 0
    trimmed: bool = False
    #: Selected handles omitted completely because even their bounded headers
    #: did not fit after higher-priority entries.
    omitted_entries: int = 0
    omitted_handles: tuple[str, ...] = ()
    #: Closed classification whenever the presentation cap changed the field.
    truncation_classification: Optional[str] = None
    #: Handles that were selected but could not be resolved (evicted, or issued
    #: by a session that no longer holds them). Recorded because a citation that
    #: silently resolved to nothing is exactly the undiagnosable case the store's
    #: eviction logging exists to prevent.
    unresolved: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The value of the extraction call's `presented_results` input field."""
        rendered = _render_presented_payload(
            self.entries,
            max_bytes=self.max_bytes,
            trimmed=self.trimmed,
            unresolved=self.unresolved,
        )
        if self.max_bytes and len(rendered.encode("utf-8")) > self.max_bytes:
            # Last-resort guard for an explicitly tiny test/deployment cap. The
            # normal allocator fits complete headers and rows; this branch keeps
            # even a pathological cap hard rather than silently exceeding it.
            marker = (
                f"[classification={PRESENTATION_TRUNCATION_CLASSIFICATION}]"
            )
            rendered, _ = _truncate_utf8(rendered, self.max_bytes, marker)
        return rendered

    @property
    def field_bytes(self) -> int:
        """Bytes of the rendered field — the seam ido-mn1.6.10 sizes from."""
        return len(self.text.encode("utf-8", errors="ignore"))

    def as_evidence(self) -> dict[str, Any]:
        return {
            "handles": [entry.as_evidence() for entry in self.entries],
            "bytes": self.total_bytes,
            "field_bytes": self.field_bytes,
            "max_bytes": self.max_bytes,
            "trimmed": self.trimmed,
            "omitted_entries": self.omitted_entries,
            "omitted_handles": list(self.omitted_handles),
            "truncation_classification": self.truncation_classification,
            "truncation_reason": (
                PRESENTATION_TRUNCATION_REASON if self.trimmed else None
            ),
            "unresolved": list(self.unresolved),
        }


def _presentation_cap_notice(max_bytes: int) -> str:
    """The typed notice reserved by the global payload allocator."""
    return (
        f"[classification={PRESENTATION_TRUNCATION_CLASSIFICATION} "
        f"reason={PRESENTATION_TRUNCATION_REASON} cap_bytes={max_bytes}; "
        "the presentation field is partial and must not be used as proof of "
        "complete coverage]"
    )


def _resolution_incomplete_notice(count: int) -> str:
    """The bounded field notice for selected handles that were unavailable."""
    return (
        "[result-handle resolution incomplete: "
        f"{count} selected handle(s) were unavailable; "
        "do not imply that the resulting listing is complete]"
    )


def _render_presented_payload(
    entries: Iterable[ResolvedResult],
    *,
    max_bytes: int,
    trimmed: bool,
    unresolved: Iterable[str] = (),
) -> str:
    blocks = [
        f"[presented result {index}]\n{entry.as_text()}"
        for index, entry in enumerate(entries, start=1)
    ]
    if trimmed:
        blocks.append(_presentation_cap_notice(max_bytes))
    unresolved_count = sum(1 for _handle_id in unresolved)
    if unresolved_count:
        blocks.append(_resolution_incomplete_notice(unresolved_count))
    if not blocks:
        return PRESENTED_RESULTS_NONE
    return "\n\n".join(blocks)


def _fits_presentation(
    entries: Iterable[ResolvedResult],
    *,
    budget: int,
    trimmed: bool,
    unresolved: Iterable[str] = (),
) -> bool:
    rendered = _render_presented_payload(
        entries,
        max_bytes=budget,
        trimmed=trimmed,
        unresolved=unresolved,
    )
    return len(rendered.encode("utf-8", errors="ignore")) <= budget


def _fit_presented_entries(
    entries: list[ResolvedResult],
    budget: int,
    *,
    unresolved: Iterable[str] = (),
) -> tuple[tuple[ResolvedResult, ...], int, bool]:
    """Greedily fit complete headers and whole rows under the total field cap."""
    unresolved = tuple(unresolved)
    metadata_trimmed = any(entry.metadata_trimmed for entry in entries)
    if _fits_presentation(
        entries,
        budget=budget,
        trimmed=metadata_trimmed,
        unresolved=unresolved,
    ):
        return tuple(entries), 0, metadata_trimmed

    accepted: list[ResolvedResult] = []
    omitted = 0
    for position, entry in enumerate(entries):
        empty = replace(
            entry,
            rows=(),
            bytes=0,
            trimmed=entry.rows_available > 0,
        )
        if not _fits_presentation(
            [*accepted, empty],
            budget=budget,
            trimmed=True,
            unresolved=unresolved,
        ):
            omitted = len(entries) - position
            break

        kept: list[str] = []
        spent = 0
        for row in entry.rows:
            cost = len(row.encode("utf-8", errors="ignore")) + 1
            candidate_rows = (*kept, row)
            candidate = replace(
                entry,
                rows=candidate_rows,
                bytes=spent + cost,
                trimmed=len(candidate_rows) < entry.rows_available,
            )
            if not _fits_presentation(
                [*accepted, candidate],
                budget=budget,
                trimmed=True,
                unresolved=unresolved,
            ):
                break
            kept.append(row)
            spent += cost

        fitted = replace(
            entry,
            rows=tuple(kept),
            bytes=spent,
            trimmed=len(kept) < entry.rows_available,
        )
        accepted.append(fitted)
        if fitted.trimmed:
            omitted = len(entries) - position - 1
            break

    return tuple(accepted), omitted, True


def _rows_the_agent_saw(record: StoredResult) -> tuple[str, ...]:
    """The union of the pages actually rendered for this handle, producer order.

    Producer order, not fetch order: `ordering` is the record's claim about how
    the rows are sequenced, and a resolver that re-ordered them by when the agent
    happened to page would make that claim false in the one place it is read.

    A record with no recorded views resolves to its first page — the page
    `compact_observation_for` renders, which is what a handle in a trajectory
    written before views were recorded (or restored from such a state) must mean.
    """
    rows = record.item_list
    if not record.views:
        return page_of(record).items
    indices: list[int] = []
    seen: set[int] = set()
    for view in record.views:
        if view.item_indices:
            candidates = list(view.item_indices)
        elif view.contains:
            needle = view.contains.casefold()
            candidates = [i for i, row in enumerate(rows) if needle in row.casefold()]
            candidates = candidates[view.offset : view.offset + view.count]
        else:
            candidates = list(range(len(rows)))[
                view.offset : view.offset + view.count
            ]
        for index in candidates:
            if index not in seen:
                seen.add(index)
                indices.append(index)
    indices.sort()
    return tuple(rows[index] for index in indices)


def _scope_rows(record: StoredResult, reasons: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """`(rows, scope)` for one selected handle.

    A `presents:` handle whose PRODUCER already narrowed the population resolves
    to everything that narrowing matched, not to the page the agent stopped on:
    the skill declared this command's output the thing the answer presents, and
    the command was called with the request's own filters, so the whole filtered
    set is bounded by the request by construction. Without producer filters the
    same handle falls back to the pages the agent fetched — that is the branch
    that keeps an unfiltered 477-holder listing out of the field unless the agent
    actually walked it.

    A cited-only handle is always the fetched view. Citing an id says "this is
    where the result is", not "widen it".
    """
    if REASON_PRESENTS in reasons and record.producer_filter_applied:
        return tuple(record.item_list), SCOPE_PRODUCER_FILTERED
    return _rows_the_agent_saw(record), SCOPE_FETCHED


def _view_filter_provenance(
    record: StoredResult,
) -> tuple[tuple[dict[str, Any], ...], int]:
    """Distinct rendered view filters and their exact post-filter union count."""
    if not record.views:
        page = page_of(record)
        return (), page.matched

    grouped: dict[str, dict[str, Any]] = {}
    fallback_matched = 0
    all_matched_indices: set[int] = set()
    for view in record.views:
        identity = view.filter_identity or _normalized_cursor_identity(
            record.handle_id,
            {"contains": view.contains} if view.contains else {},
        )
        entry = grouped.setdefault(
            identity,
            {
                "filter_identity": identity,
                "filters": view.view_filters,
                "matched": 0,
            },
        )
        if view.matched_indices:
            all_matched_indices.update(view.matched_indices)
            entry.setdefault("_matched_indices", set()).update(
                view.matched_indices
            )
            entry["matched"] = len(entry["_matched_indices"])
        else:
            entry["matched"] = max(int(entry["matched"]), int(view.matched))
            fallback_matched = max(fallback_matched, int(view.matched))

    rendered: list[dict[str, Any]] = []
    for entry in grouped.values():
        entry.pop("_matched_indices", None)
        rendered.append(entry)
    matched = (
        len(all_matched_indices)
        if all_matched_indices
        else fallback_matched
    )
    return tuple(rendered), matched


def resolve_for_presentation(
    *,
    cited: Iterable[str] = (),
    presented: Iterable[str] = (),
    host: Any = None,
    max_bytes: Optional[int] = None,
) -> PresentedResults:
    """Resolve the handles an answer must present, under one byte cap.

    `cited` are ids the agent named on its closing step; `presented` are ids
    whose producing command the executing skill marked `presents:`. Both are
    ordered, cited first, and a handle in both is resolved once carrying both
    reasons — order is the cap's allocation order, so a citation is never starved
    by a declaration.

    Rows already passed the capture policy on the way IN (`_record_for` applies
    the same `for_prompt=True` projection the observation is built under), so a
    withheld payload arrives here as an envelope and is reported as withheld
    rather than silently as empty. Nothing here re-reads a raw command response.

    Never raises: a resolution failure must degrade the answer, not fail a turn
    whose work is already done.
    """
    budget = max(1, int(max_bytes if max_bytes is not None else presented_max_bytes()))
    store = store_for(host)
    selected: dict[str, list[str]] = {}
    for handle_id in cited:
        selected.setdefault(str(handle_id).lower(), []).append(REASON_CITED)
    for handle_id in presented:
        reasons = selected.setdefault(str(handle_id).lower(), [])
        if REASON_PRESENTS not in reasons:
            reasons.append(REASON_PRESENTS)
    if store is None or not selected:
        return PresentedResults(max_bytes=budget)

    entries: list[ResolvedResult] = []
    unresolved: list[str] = []
    for handle_id, reason_list in selected.items():
        record = store.peek(handle_id)
        if record is None:
            unresolved.append(handle_id)
            continue
        try:
            record = _policy_project_record(record)
            reasons = tuple(reason_list)
            available, scope = _scope_rows(record, reasons)
            view_filters, view_matched = _view_filter_provenance(record)
            matched = (
                record.materialized_count
                if scope == SCOPE_PRODUCER_FILTERED
                else view_matched
            )
            apply_policy = _policy_applier(
                record.command_name,
                record.classification,
            )
            resolved_rows = apply_policy(
                "resolved_rows",
                list(available),
            )
            safe_available = (
                tuple(str(row) for row in resolved_rows)
                if isinstance(resolved_rows, list)
                else ()
            )
            command_name, command_trimmed = _bounded_metadata(record.command_name)
            kind, kind_trimmed = _bounded_metadata(record.kind)
            summary_value = apply_policy("summary", record.summary)
            ordering_value = apply_policy("ordering", record.ordering)
            filters_value = apply_policy("filters", record.filters)
            summary, summary_trimmed = _bounded_metadata(summary_value)
            ordering, ordering_trimmed = _bounded_metadata(ordering_value)
            filters, filters_trimmed = _bounded_filters(filters_value)
            resolved_view_filters, view_filters_trimmed = _bounded_view_filters(
                view_filters
            )
            metadata_trimmed = any(
                (
                    command_trimmed,
                    kind_trimmed,
                    summary_trimmed,
                    ordering_trimmed,
                    filters_trimmed,
                    view_filters_trimmed,
                )
            )
            row_bytes = sum(
                len(row.encode("utf-8", errors="ignore")) + 1
                for row in safe_available
            )
            entries.append(
                ResolvedResult(
                    handle_id=handle_id,
                    command_name=command_name,
                    kind=kind,
                    summary=summary,
                    ordering=ordering,
                    scope=scope,
                    reasons=reasons,
                    filters=filters,
                    view_filters=resolved_view_filters,
                    total=record.total,
                    materialized=record.materialized_count,
                    matched=matched,
                    matched_complete=(
                        record.is_source_complete
                        and isinstance(resolved_rows, list)
                    ),
                    source_complete=record.is_source_complete,
                    incomplete_reason=record.source_incomplete_reason,
                    rows=safe_available,
                    rows_available=len(available),
                    trimmed=False,
                    metadata_trimmed=metadata_trimmed,
                    withheld=(
                        record.payload_withheld
                        or not isinstance(resolved_rows, list)
                    ),
                    bytes=row_bytes,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"could not resolve handle {handle_id!r}: {exc!r}")
            unresolved.append(handle_id)
    if unresolved:
        logger.warning(
            "presentation resolution found no payload for %d handle(s): %s",
            len(unresolved),
            ", ".join(unresolved),
        )
    fitted, omitted, trimmed = _fit_presented_entries(
        entries,
        budget,
        unresolved=unresolved,
    )
    omitted_handles = tuple(
        entry.handle_id for entry in entries[len(fitted) :]
    )
    return PresentedResults(
        entries=fitted,
        total_bytes=sum(entry.bytes for entry in fitted),
        max_bytes=budget,
        trimmed=trimmed,
        omitted_entries=omitted,
        omitted_handles=omitted_handles,
        truncation_classification=(
            PRESENTATION_TRUNCATION_CLASSIFICATION if trimmed else None
        ),
        unresolved=tuple(unresolved),
    )


def presentation_handles_for(
    handle_ids: Iterable[str],
    command_names: Iterable[str] = (),
    *,
    host: Any = None,
) -> tuple[str, ...]:
    """Those of `handle_ids` whose producer says their rows are deliverable.

    **The default is the command's own flag** (`ResultHandleSpec.presentation`),
    and that is what makes this rule arm-invariant: every arm reads the same bit
    off the same stored record, with no catalogue involved. Arm A cannot read a
    skill — `off` never opens `_skills/` — so a skill-only rule would have given
    the control arm strictly less mechanism than the treatment arms.

    `command_names` is the executing skill's optional `presents:` OVERRIDE. When
    it is non-empty it replaces the flag for this call, which is exactly the two
    edits an author can want: NARROW (name a subset of the flagged commands, so
    a skill that only reports holders does not also drag in every portrait) and
    ADD (name a command whose producer did not flag itself). When it is empty —
    which is every arm-A turn and every skill that declares nothing — the flag
    stands, so A, B and C run the same rule.

    Resolved through the store rather than through the plan: the handle IS the
    producing `command_call_id` and `StoredResult` records both the command name
    and the flag, so neither half of this needs a second index.
    """
    wanted = {str(name).strip() for name in command_names if str(name).strip()}
    store = store_for(host)
    if store is None:
        return ()
    found: list[str] = []
    for handle_id in handle_ids:
        normalized = str(handle_id).lower()
        record = store.peek(normalized)
        if record is None:
            continue
        selected = record.command_name in wanted if wanted else record.presentation
        if selected:
            found.append(normalized)
    return tuple(found)
