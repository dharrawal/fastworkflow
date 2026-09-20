"""One scoped-execution reference and one comparison projection (`fix-9eg.4`).

Three things the product wants to compare turn out to be the same thing:

- the current experiment winner against a candidate,
- the selected best run of a task against its other attempts,
- a recorded teacher pass against the student pass that followed it.

They differ only in HOW the two sides are named, so this module defines ONE
reference — `ExecutionRef` — and one projection over it, rather than three
views that would drift. A comparison is a pure read: nothing here writes to an
evidence store, and nothing here is a judgement. "Reference", "best" and
"teacher" are labels a caller chose; a difference from the left side is a
difference, not an error.

Reference vocabulary is borrowed, not invented. `store_id` + logical turn key
is exactly what `workspace.py` already uses for a portable evidence address,
and experiment/task/attempt are the `experiments` / `experiment_attempts`
columns. A reference that DECLARES an experiment, task or attempt is checked
against the turn rows it names: a scope the evidence does not record is
refused, because a forged or stale scope would otherwise ride along on real
evidence and label it — and anchor feedback to it — as something it is not.

The only genuinely new part is `pass_id`, for a turn that holds more than one
recorded pass. Membership is resolved ONLY from recorded spans, through a
`PassSelector` the caller supplies and this module verifies: a rule naming a
span or a stamp the turn does not contain is an unknown pass and is refused
rather than answered with an empty view. Pass identity and pass scope travel
together — a `pass_id` with no selector, or a selector with no `pass_id`, is
refused, because either alone yields a projection whose contents and whose
`ref_id` disagree about what was shown.

WHAT THIS REPO RECORDS TODAY, stated plainly because the pass machinery above
reads as if a producer used it: nothing does. `distillation.py` runs the agent
twice for one user message, but it stamps no pass marker on its spans, records
no per-pass turn row, and its per-pass answer, plan and action log stay in
process (`summarize_and_record_turn` appends to in-memory conversation history);
`ObservabilityStore.list_distillation_runs` is a stub that returns `[]`. So a
pass-scoped projection carries only the STEPS and the LLM cost that recorded
spans attribute to the pass. The turn's answer, status and wall time belong to
the turn and are shared by every pass in it, so they are labelled
`shared_across_passes` rather than reported as that pass's own. Producer-side
stamping is filed as `fix-txxy`; until it lands, `discover_pass_selectors`
returns `[]` on every real turn and every caller compares whole turns.

Reads go through an injected reader: no filesystem lookup, no cross-store
search, and a reference naming an unknown store fails rather than being
resolved somewhere else. Absence is preserved rather than repaired -- an
unreadable turn is reported in `unavailable` and the rest still renders, so an
execution with no scores, a half-pruned trace, or one missing side can still be
inspected.

The per-turn step list is the execution ledger from `run_chatbot/server.py`,
injected rather than re-derived, because a second implementation of "what
dispatches happened in this turn" is exactly the semantic mismatch this work
must not introduce. Nested wrappers therefore appear exactly once, as the
ledger files them, and every roll-up here that could double-count a parent and
its span-less inner hop counts ROOT rows only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

# The span name the ledger builds its rows from, restated here (as
# `run_chatbot/server.py` restates it) so that reading a stored trace does not
# import the HTTP layer.
SPAN_COMMAND_EXECUTE = "fw.command.execute"

# How a matched pair was decided. `recorded` means a stored structured
# alignment said so; the rest are this module's deterministic fallback, named
# after the recorded fields they agreed on. Cross-run `command_call_id`
# equality is deliberately NOT a basis: call ids are minted per run, so equal
# ids across two runs are either meaningless or a copied trace.
BASIS_RECORDED = "recorded"
BASIS_COMMAND_CONTEXT_PARAMETERS = "command+context+parameters"
BASIS_COMMAND_CONTEXT = "command+context"
BASIS_COMMAND = "command"
# Unmatched steps carry a basis too, so a reader can tell "we looked and found
# no counterpart" from "we could not form a key for this step at all".
BASIS_UNMATCHED = "unmatched"
BASIS_UNKNOWN = "unknown"

PAIR_MATCHED = "matched"
PAIR_LEFT_ONLY = "left_only"
PAIR_RIGHT_ONLY = "right_only"

# Whose content a projected value describes. A whole-turn projection owns its
# turn's answer and wall time; a pass-scoped one does not -- the passes share
# one turn row, so its text is `shared_across_passes` and saying otherwise
# would attribute the student's answer to the teacher.
ATTRIBUTION_TURN = "turn"
ATTRIBUTION_SHARED = "shared_across_passes"
# An artifact whose `CommandOutput` recorded no `command_call_id` cannot be
# joined to a dispatch, so a pass-scoped projection can say only that the turn
# produced it. It is listed apart rather than claimed by both passes.
ATTRIBUTION_PASS = "pass"
ATTRIBUTION_UNATTRIBUTED = "unattributed"

# Beyond this many step pairs the quadratic alignment is skipped in favour of
# an order-preserving greedy pass, and the result says so. A multi-turn attempt
# with thousands of dispatches is rare; silently spending minutes on one in a
# request handler is worse than an explicitly degraded answer.
_MAX_ALIGNMENT_CELLS = 250_000


class ComparisonError(RuntimeError):
    """Base class for comparison failures."""


class InvalidExecutionRef(ComparisonError, ValueError):
    """A reference does not name a readable scope."""


class ExecutionScopeMismatch(InvalidExecutionRef):
    """A reference declares an experiment/task/attempt the evidence does not record.

    Raised rather than dropped to a label, because the failure it prevents is
    silent: a reference that claims `attempt=2` over a turn recorded under
    attempt 1 would render real evidence under a false heading, and every
    feedback anchor and review-pair key derived from it would carry that claim.
    """


class UnknownRecordedPass(InvalidExecutionRef):
    """A pass selector resolves against no recorded evidence in a turn.

    A selector is external input. When its rules name a span id or a stamp the
    turn does not contain, the honest answer is "this turn records no such
    pass" -- not an empty pass view, which reads as "the pass did nothing".
    """


class InvalidRecordedAlignment(ComparisonError, ValueError):
    """A supplied alignment names steps that are not in the recorded evidence."""


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _exact_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_exact_int(value: Any, field: str) -> int:
    """An attempt number, or a refusal. Never a truncation and never a bool.

    `int(value)` was wrong here: it turns 1.9 into attempt 1 and `True` into
    attempt 1, so a malformed reference would silently name a DIFFERENT
    attempt's evidence. Exact integers only; the string form is accepted at the
    wire boundary alone (`from_mapping`), where query parameters arrive as text.
    """
    exact = _exact_int(value)
    if exact is None:
        raise InvalidExecutionRef(
            f"{field} must be an exact integer, not {value!r}"
        )
    return exact


def _canonical_json(value: Any) -> str:
    """Stable text for a recorded value, for digesting only.

    `default=repr` rather than raising: a record that round-tripped through
    JSON is already plain data, but a hand-built one may hold anything, and a
    step whose parameters cannot be serialized still deserves a key.
    """
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers it
        return repr(value)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# The reference
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionRef:
    """One scoped recorded execution: the unit both sides of a comparison name.

    `store_id` + `turn_keys` is the mandatory part and is the same address
    `workspace.py` uses (`logical_turn_key` within a named store). `turn_keys`
    is a SEQUENCE, so a multi-turn attempt is one reference and stays navigable
    in full rather than collapsing to its final answer. Everything else narrows
    it:

    - `experiment_id` / `task_id` / `attempt` when the turns belong to an
      experiment attempt. Optional, because an ad-hoc chat turn and a
      teacher/student pass recorded outside any experiment are both legitimate
      things to compare; declared, they are CHECKED against the turn rows by
      `project_execution` rather than taken as labels.
    - `pass_id` when one turn holds more than one recorded pass. Two passes
      sharing a trace id are two references differing only in this field.

    `label` is display-only and is excluded from `ref_id`, so renaming
    "Reference" to "Best run" does not invalidate review progress or feedback
    recorded against the pair.
    """

    store_id: str
    turn_keys: tuple[str, ...]
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    pass_id: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        store_id = _clean(self.store_id)
        if not store_id:
            raise InvalidExecutionRef(
                "store_id is required; executions are never searched across stores"
            )
        keys = tuple(key for key in (_clean(k) for k in self.turn_keys) if key)
        if not keys:
            raise InvalidExecutionRef("at least one turn key is required")
        if len(set(keys)) != len(keys):
            raise InvalidExecutionRef("turn keys must be distinct")
        attempt = self.attempt
        if attempt is not None:
            attempt = _required_exact_int(attempt, "attempt")
            if attempt < 0:
                raise InvalidExecutionRef("attempt must not be negative")
        object.__setattr__(self, "store_id", store_id)
        object.__setattr__(self, "turn_keys", keys)
        object.__setattr__(self, "experiment_id", _clean(self.experiment_id))
        object.__setattr__(self, "task_id", _clean(self.task_id))
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "pass_id", _clean(self.pass_id))
        object.__setattr__(self, "label", _clean(self.label))

    def ref_id(self) -> str:
        """A stable id for this scope, for anchors and review-pair keys.

        Derived, never minted, so two processes that build the same reference
        agree without coordinating, and a reference reconstructed from a URL
        keys the same review row it did yesterday. `label` is excluded on
        purpose (see the class docstring).
        """
        return "xr-" + _digest(
            self.store_id,
            "\x1e".join(self.turn_keys),
            self.experiment_id or "",
            self.task_id or "",
            "" if self.attempt is None else str(self.attempt),
            self.pass_id or "",
        )

    def as_dict(self) -> dict[str, Any]:
        """The wire shape. Human UI and agent reads use this same object."""
        return {
            "ref_id": self.ref_id(),
            "store_id": self.store_id,
            "turn_keys": list(self.turn_keys),
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "pass_id": self.pass_id,
            "label": self.label,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionRef":
        """Parse the wire shape, tolerating the workspace's field names.

        `logical_turn_keys` / `logical_turn_key` are accepted because that is
        what a workspace attempt row's `turn_refs` are called; they name the
        same strings.

        `attempt` may arrive as the text an HTTP query carries, and only as the
        exact decimal form of an integer: `"2"` is attempt 2, while `"2.0"`,
        `2.5` and `True` are refused here rather than truncated into a
        reference that names somebody else's attempt.
        """
        if not isinstance(value, Mapping):
            raise InvalidExecutionRef("an execution reference must be an object")
        raw_keys: Any = None
        for key in ("turn_keys", "logical_turn_keys", "turns"):
            if value.get(key) is not None:
                raw_keys = value[key]
                break
        if raw_keys is None:
            single = value.get("turn_key", value.get("logical_turn_key"))
            raw_keys = [single] if single is not None else []
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        if not isinstance(raw_keys, (list, tuple)):
            raise InvalidExecutionRef("turn_keys must be an array of turn keys")
        attempt = value.get("attempt")
        if isinstance(attempt, str):
            text = attempt.strip()
            if not (text.lstrip("-").isdigit()):
                raise InvalidExecutionRef(
                    f"attempt must be an exact integer, not {attempt!r}"
                )
            attempt = int(text)
        return cls(
            store_id=value.get("store_id"),
            turn_keys=tuple(str(k) for k in raw_keys if k is not None),
            experiment_id=value.get("experiment_id"),
            task_id=value.get("task_id"),
            attempt=attempt,
            pass_id=value.get("pass_id"),
            label=value.get("label"),
        )


def review_pair_key(left: ExecutionRef, right: ExecutionRef) -> str:
    """The identity of one review pair: the EXACT two executions compared.

    Ordered, not a set: "this candidate reviewed against that reference" is
    not the same statement as the reverse, and pinning a different reference
    must produce new pairs rather than relabel old ones.
    """
    return f"{left.ref_id()}|{right.ref_id()}"


# ----------------------------------------------------------------------
# Recorded passes within one turn
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PassSelector:
    """How to tell one recorded pass's activity from another's, within a turn.

    No rule infers a pass from ordering, timing or model name: the failure this
    module must avoid is a confident wrong attribution, and showing the
    student's calls under the teacher's heading is worse than showing them as
    unattributed.

    Every rule names RECORDED SPANS and is checked against the turn's spans
    before use (`resolve_against`), so a selector is evidence a caller can point
    at rather than an assertion this module takes on trust. Rules are ORed:

    - `attribute_key`/`attribute_value`: a span attribute recorded on the
      step's own `fw.command.execute` span or on any of its ancestors. This is
      the rule a producer that stamps its passes should use, and the one
      `discover_pass_selectors` builds.
    - `root_span_ids`: the step's span lies in the subtree of one of these. A
      producer that opens one span per pass needs nothing else.
    - `span_ids`: an explicit membership list of recorded span ids.

    `exclude_span_ids` removes activity that belongs to neither pass -- insight
    extraction runs after both and its LLM calls would otherwise land in
    whichever pass a subtree rule happened to cover. Exclusion is checked
    against a span's whole ancestry, so excluding a root excludes its subtree,
    and exclusion wins over every inclusion rule.

    There is deliberately no rule keyed on `command_call_id` and no separate
    subtree-exclusion set: the first would let a caller assert pass membership
    for dispatches the span tree does not place in the pass (an in-process
    action log is not recorded evidence), and the second is what
    `exclude_span_ids` already does.
    """

    pass_id: str
    attribute_key: Optional[str] = None
    attribute_value: Optional[str] = None
    root_span_ids: frozenset[str] = frozenset()
    span_ids: frozenset[str] = frozenset()
    exclude_span_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        pass_id = _clean(self.pass_id)
        if not pass_id:
            raise ValueError("pass_id is required")
        object.__setattr__(self, "pass_id", pass_id)
        for name in ("root_span_ids", "span_ids", "exclude_span_ids"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        if (self.attribute_key is None) != (self.attribute_value is None):
            raise ValueError(
                "attribute_key and attribute_value are set together or not at all"
            )
        if not (self.attribute_key or self.root_span_ids or self.span_ids):
            raise ValueError(
                f"pass selector {pass_id!r} has no membership rule; a selector "
                "that matches everything cannot separate two passes"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
            "root_span_ids": sorted(self.root_span_ids),
            "span_ids": sorted(self.span_ids),
            "exclude_span_ids": sorted(self.exclude_span_ids),
        }

    def resolve_against(self, tree: "_SpanTree", turn_key: str) -> None:
        """Refuse unless every rule names evidence this turn actually recorded.

        Checked per turn, before any step is attributed. An id or a stamp the
        turn does not contain means the caller is describing a pass that is not
        in this evidence, and the answer it would otherwise get -- a projection
        with no steps -- is indistinguishable from a pass that ran and did
        nothing.
        """
        missing = sorted(
            (self.root_span_ids | self.span_ids | self.exclude_span_ids)
            - set(tree.by_id)
        )
        if missing:
            raise UnknownRecordedPass(
                f"pass {self.pass_id!r} names span(s) {missing} that turn "
                f"{turn_key!r} does not record"
            )
        if self.attribute_key is not None and not any(
            tree.attributes(span).get(self.attribute_key) == self.attribute_value
            for span in tree.by_id.values()
        ):
            raise UnknownRecordedPass(
                f"pass {self.pass_id!r} is selected by "
                f"{self.attribute_key}={self.attribute_value!r}, which no span of "
                f"turn {turn_key!r} records"
            )


class _SpanTree:
    """Parent links and decoded attributes for one turn's spans."""

    def __init__(self, spans: Sequence[Mapping[str, Any]]) -> None:
        self.by_id: dict[str, Mapping[str, Any]] = {}
        for span in spans:
            span_id = _text_or_none(span.get("span_id"))
            if span_id:
                self.by_id[span_id] = span

    def attributes(self, span: Mapping[str, Any]) -> dict[str, Any]:
        """Span attributes as a mapping, whether the reader decoded them.

        `workspace.trace` decodes the JSON column; `ObservabilityStore.get_spans`
        hands back the raw text. Both are legitimate readers, so accept both
        rather than making the caller normalize.
        """
        raw = span.get("attributes")
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except (ValueError, TypeError):
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    def ancestry(self, span_id: Optional[str]) -> list[Mapping[str, Any]]:
        """The span and its ancestors, nearest first. Cycle-safe."""
        chain: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        cursor = span_id
        while cursor and cursor in self.by_id and cursor not in seen:
            seen.add(cursor)
            span = self.by_id[cursor]
            chain.append(span)
            cursor = _text_or_none(span.get("parent_span_id"))
        return chain


def discover_pass_selectors(
    spans: Iterable[Mapping[str, Any]],
    *,
    attribute_key: str,
    exclude_values: Iterable[str] = (),
) -> list[PassSelector]:
    """Selectors for every distinct recorded value of one span attribute.

    Discovery, not inference: it reports the pass labels a producer actually
    stamped, in sorted order so two callers agree, and returns `[]` when
    nothing was stamped. A caller that gets `[]` has a turn with no recorded
    pass identity and must compare it whole.
    """
    tree = _SpanTree(list(spans))
    excluded = {str(value) for value in exclude_values}
    found: set[str] = set()
    for span in tree.by_id.values():
        value = tree.attributes(span).get(attribute_key)
        if isinstance(value, str) and value and value not in excluded:
            found.add(value)
    return [
        PassSelector(
            pass_id=value, attribute_key=attribute_key, attribute_value=value
        )
        for value in sorted(found)
    ]


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------


class ExecutionReader(Protocol):
    """What a projection needs, and nothing more.

    Both methods take an explicit `store_id`. A reader is free to refuse one
    it was not constructed for; none of them may go looking for the turn
    elsewhere.
    """

    def turn(self, store_id: str, turn_key: str) -> Optional[Mapping[str, Any]]:
        """The turn row with its decoded `record`, or None if absent."""

    def trace(self, store_id: str, turn_key: str) -> list[Mapping[str, Any]]:
        """The turn's span rows, or `[]`."""


class StoreExecutionReader:
    """Reader over ONE opened `ObservabilityStore`, bound to one store id.

    The store is opened and named by the caller -- typically a
    `ReadOnlyObservabilityStore`, so inspecting evidence cannot write to it.
    A reference naming a different store raises instead of being answered from
    this one.
    """

    def __init__(self, store_id: str, store: Any) -> None:
        store_id = _clean(store_id)
        if not store_id:
            raise ValueError("store_id is required")
        self.store_id = store_id
        self._store = store

    def _check(self, store_id: str) -> None:
        if store_id != self.store_id:
            raise InvalidExecutionRef(
                f"this reader serves store {self.store_id!r}, not {store_id!r}; "
                "executions are never searched across stores"
            )

    def turn(self, store_id: str, turn_key: str) -> Optional[dict[str, Any]]:
        self._check(store_id)
        row = self._store.get_turn(turn_key)
        if row is None:
            return None
        result = dict(row)
        raw = result.pop("record_json", None)
        try:
            result["record"] = json.loads(raw) if isinstance(raw, str) else None
        except (ValueError, TypeError):
            result["record"] = None
        result["store_id"] = store_id
        result["logical_turn_key"] = turn_key
        return result

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        self._check(store_id)
        return [dict(span) for span in self._store.get_spans(turn_key)]


class WorkspaceExecutionReader:
    """Reader over a loaded `ObservabilityWorkspace`.

    The workspace already enforces manifest-bound, per-store, read-only access
    and already decodes the turn record and span attributes, so this is a
    two-line adapter rather than a second access path.
    """

    def __init__(self, workspace: Any) -> None:
        self._workspace = workspace

    def turn(self, store_id: str, turn_key: str) -> Optional[dict[str, Any]]:
        return self._workspace.turn(store_id, turn_key)

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        return list(self._workspace.trace(store_id, turn_key))


# ----------------------------------------------------------------------
# Injected projections that already exist elsewhere
# ----------------------------------------------------------------------

LedgerProjection = Callable[[Any, Iterable[Mapping[str, Any]]], Mapping[str, Any]]
CostRollup = Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]]


def default_ledger_projection() -> LedgerProjection:
    """`run_chatbot/server.py`'s `execution_ledger` -- the one implementation.

    Imported lazily so this module stays usable without the HTTP layer and so
    the dependency runs one way at import time. Callers that already hold the
    function (the server does) should pass it instead.
    """
    from fastworkflow.run_chatbot.server import execution_ledger

    return execution_ledger


def default_cost_rollup() -> CostRollup:
    """`run_chatbot/server.py`'s `cost_rollup`, for the same reason.

    Recorded cost only: it answers `total: None`, never 0, when no LLM call
    recorded a cost, and this module passes that through unchanged.
    """
    from fastworkflow.run_chatbot.server import cost_rollup

    return cost_rollup


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRef:
    """One artifact a command produced, as the record refers to it.

    `artifact_id` is set when the value was offloaded to the `artifacts` table
    (the `__fw_artifact_ref__` envelope `serialize_turn_result` writes); it is
    None for a value small enough to have stayed inline, which is a fact about
    size, not about whether the artifact exists. Either way the artifact is
    reachable: by id from the store, or from the record itself.

    `attribution` says WHOSE artifact this is: the turn's (`turn`), the selected
    pass's, because its dispatch is in that pass (`pass`), or nobody's in
    particular (`unattributed`) -- a `CommandOutput` that recorded no
    `command_call_id` cannot be joined to a dispatch, so in a pass-scoped
    projection it is reported apart instead of being shown under both passes as
    if each had produced it.
    """

    turn_key: str
    command_call_id: Optional[str]
    command_name: Optional[str]
    key: str
    artifact_id: Optional[str] = None
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    inline: bool = True
    error: Optional[str] = None
    attribution: str = ATTRIBUTION_TURN

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_key": self.turn_key,
            "command_call_id": self.command_call_id,
            "command_name": self.command_name,
            "key": self.key,
            "artifact_id": self.artifact_id,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "inline": self.inline,
            "error": self.error,
            "attribution": self.attribution,
        }


@dataclass(frozen=True)
class ExecutionStep:
    """One dispatch, as the ledger filed it plus what the record recorded.

    Everything up to `asked_user` is the ledger row verbatim. `parameters` and
    `response_success` are added here from the two places a dispatch's inputs
    and outcome are recorded -- the turn record's `CommandOutput` (durable,
    joined on `command_call_id`) and the execute span's attributes
    (best-effort) -- with the record preferred. `parameters_digest` is what
    alignment keys on; it is None when neither source recorded parameters, and
    a step with no digest is never matched on a parameter basis.
    """

    turn_index: int
    turn_key: str
    position: int
    command_call_id: str
    parent_call_id: Optional[str]
    command_ordinal: Optional[int]
    span_id: Optional[str]
    command_name: Optional[str]
    context: Optional[str]
    status: Optional[str]
    success: Optional[bool]
    start_ns: Optional[int]
    duration_ns: Optional[int]
    in_record: bool
    span_recorded: bool
    child_call: bool
    asked_user: int
    parameters: Optional[Any] = None
    parameters_source: Optional[str] = None
    parameters_digest: Optional[str] = None
    response_success: Optional[bool] = None
    pass_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_key": self.turn_key,
            "position": self.position,
            "command_call_id": self.command_call_id,
            "parent_call_id": self.parent_call_id,
            "command_ordinal": self.command_ordinal,
            "span_id": self.span_id,
            "command_name": self.command_name,
            "context": self.context,
            "status": self.status,
            "success": self.success,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "in_record": self.in_record,
            "span_recorded": self.span_recorded,
            "child_call": self.child_call,
            "asked_user": self.asked_user,
            "parameters": self.parameters,
            "parameters_source": self.parameters_source,
            "parameters_digest": self.parameters_digest,
            "response_success": self.response_success,
            "pass_id": self.pass_id,
        }


@dataclass(frozen=True)
class TurnProjection:
    """One logical turn of the execution: its answer and what it cost.

    `answer`, `status`, `success`, `failure_reason`, `user_message` and the
    timestamps are the TURN ROW's, quoted as recorded. When this projection is
    scoped to a pass, they are still the turn row's -- the passes share it --
    so `content_attribution` says `shared_across_passes` and
    `pass_content_recorded` is False. Nothing in this repo records a per-pass
    answer or plan (see the module docstring), and a pass-scoped projection that
    presented the turn's answer as the pass's own would be inventing the one
    thing a teacher/student comparison is read for.

    `cost` is genuinely pass-scoped when a selector is in play: it rolls up only
    the LLM spans the span tree attributes to that pass.
    """

    turn_index: int
    turn_key: str
    status: Optional[str]
    success: Optional[bool]
    failure_reason: Optional[str]
    answer: Optional[str]
    user_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    suspended_ms: Optional[int]
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    ledger_summary: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    pass_id: Optional[str] = None
    content_attribution: str = ATTRIBUTION_TURN
    pass_content_recorded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_key": self.turn_key,
            "status": self.status,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "answer": self.answer,
            "user_message": self.user_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "suspended_ms": self.suspended_ms,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "ledger_summary": dict(self.ledger_summary),
            "cost": dict(self.cost),
            "pass_id": self.pass_id,
            "content_attribution": self.content_attribution,
            "pass_content_recorded": self.pass_content_recorded,
        }


@dataclass(frozen=True)
class ExecutionProjection:
    """Everything one side of a comparison shows, from read-only evidence.

    Partial by design. `unavailable` lists what could not be read (a pruned
    turn, a turn the store never held) and the projection still carries the
    turns that were readable, because half an execution is inspectable and
    refusing to render it would hide the half that survived.
    """

    ref: ExecutionRef
    turns: tuple[TurnProjection, ...]
    steps: tuple[ExecutionStep, ...]
    artifacts: tuple[ArtifactRef, ...]
    timing: Mapping[str, Any]
    cost: Mapping[str, Any]
    unavailable: tuple[str, ...] = ()
    unassigned_steps: tuple[ExecutionStep, ...] = ()
    unattributed_artifacts: tuple[ArtifactRef, ...] = ()
    pass_selector: Optional[PassSelector] = None

    @property
    def readable(self) -> bool:
        """At least one named turn was found. False is inspectable, not fatal."""
        return bool(self.turns)

    @property
    def content_attribution(self) -> str:
        """Whose the turn-level text is: the turn's, or shared between passes."""
        return ATTRIBUTION_SHARED if self.pass_selector else ATTRIBUTION_TURN

    def answers(self) -> list[dict[str, Any]]:
        """The answer of each turn, in order. The default view of a run.

        Each row carries its `attribution`, so a pass-scoped view cannot be read
        as "this is what the teacher answered" when what is recorded is what the
        turn answered.
        """
        return [
            {
                "turn_index": turn.turn_index,
                "turn_key": turn.turn_key,
                "answer": turn.answer,
                "status": turn.status,
                "success": turn.success,
                "failure_reason": turn.failure_reason,
                "pass_id": turn.pass_id,
                "attribution": turn.content_attribution,
                "pass_content_recorded": turn.pass_content_recorded,
            }
            for turn in self.turns
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.as_dict(),
            "readable": self.readable,
            "content_attribution": self.content_attribution,
            "turns": [turn.as_dict() for turn in self.turns],
            "answers": self.answers(),
            "steps": [step.as_dict() for step in self.steps],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "timing": dict(self.timing),
            "cost": dict(self.cost),
            "unavailable": list(self.unavailable),
            "unassigned_steps": [step.as_dict() for step in self.unassigned_steps],
            "unattributed_artifacts": [
                artifact.as_dict() for artifact in self.unattributed_artifacts
            ],
            "pass_selector": (
                self.pass_selector.as_dict() if self.pass_selector else None
            ),
        }


def _command_outputs(record: Any) -> list[Mapping[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    turn_output = record.get("turn_output")
    if not isinstance(turn_output, Mapping):
        return []
    outputs = turn_output.get("command_outputs")
    if not isinstance(outputs, list):
        return []
    return [value for value in outputs if isinstance(value, Mapping)]


def _check_scope(ref: ExecutionRef, turn_key: str, row: Mapping[str, Any]) -> None:
    """Refuse a reference whose declared scope the turn row does not record.

    Only DECLARED fields are checked: a reference that names no experiment is
    not claiming one, and the recorded scope is reported on the projected turn
    either way. A declared field that the row leaves NULL is a mismatch too --
    "attempt 2 of task-1" over a turn recorded outside any experiment is not a
    partial truth, it is a different statement from the evidence's.
    """
    declared = (
        ("experiment_id", ref.experiment_id, _text_or_none(row.get("experiment_id"))),
        ("task_id", ref.task_id, _text_or_none(row.get("task_id"))),
        ("attempt", ref.attempt, _exact_int(row.get("attempt"))),
    )
    for field_name, claimed, recorded in declared:
        if claimed is None or claimed == recorded:
            continue
        raise ExecutionScopeMismatch(
            f"reference declares {field_name}={claimed!r} but turn {turn_key!r} "
            f"in store {ref.store_id!r} records "
            + (
                f"{field_name}={recorded!r}"
                if recorded is not None
                else f"no {field_name}"
            )
        )


def _artifacts_from_record(turn_key: str, record: Any) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for output in _command_outputs(record):
        response = output.get("command_response")
        artifacts = response.get("artifacts") if isinstance(response, Mapping) else None
        if not isinstance(artifacts, Mapping):
            continue
        call_id = _text_or_none(output.get("command_call_id"))
        command_name = _text_or_none(output.get("command_name"))
        for key in sorted(artifacts):
            value = artifacts[key]
            envelope = value if isinstance(value, Mapping) else {}
            artifact_id = _text_or_none(envelope.get("__fw_artifact_ref__"))
            refs.append(
                ArtifactRef(
                    turn_key=turn_key,
                    command_call_id=call_id,
                    command_name=command_name,
                    key=str(key),
                    artifact_id=artifact_id,
                    size_bytes=_exact_int(envelope.get("size")),
                    content_type=_text_or_none(envelope.get("content_type")),
                    inline=artifact_id is None,
                    error=_text_or_none(envelope.get("error")),
                )
            )
    return refs


def _recorded_parameters(record: Any) -> dict[str, tuple[Any, Optional[bool]]]:
    """`{command_call_id: (parameters, response success)}` from the turn record.

    Keyed on `command_call_id` because that is the join the substrate already
    maintains between a `CommandOutput` and the dispatch that produced it. An
    output with no call id (a hand-built one, or a path that never stamped it)
    contributes nothing rather than being matched by position.
    """
    found: dict[str, tuple[Any, Optional[bool]]] = {}
    for output in _command_outputs(record):
        call_id = _text_or_none(output.get("command_call_id"))
        if call_id is None:
            continue
        response = output.get("command_response")
        success = (
            response.get("success") if isinstance(response, Mapping) else None
        )
        found[call_id] = (
            output.get("command_parameters"),
            success if isinstance(success, bool) else None,
        )
    return found


def _pass_id_for(
    selector: Optional[PassSelector],
    tree: _SpanTree,
    span_id: Optional[str],
) -> Optional[str]:
    """The pass a span belongs to, or None when the evidence does not say.

    A step with no recorded span can never be attributed: the span tree is the
    only thing that says which pass ran it, so such a step is withheld from
    every pass view (and counted in `unassigned_steps`) rather than assigned to
    the pass whose neighbours it sat between.
    """
    if selector is None:
        return None
    chain = tree.ancestry(span_id)
    chain_ids = {
        _text_or_none(span.get("span_id"))
        for span in chain
        if _text_or_none(span.get("span_id"))
    }
    if selector.exclude_span_ids & chain_ids:
        return None
    if span_id is not None and span_id in selector.span_ids:
        return selector.pass_id
    if selector.root_span_ids & chain_ids:
        return selector.pass_id
    if selector.attribute_key is not None:
        for span in chain:
            value = tree.attributes(span).get(selector.attribute_key)
            if isinstance(value, str) and value == selector.attribute_value:
                return selector.pass_id
    return None


def _spans_in_pass(
    selector: Optional[PassSelector],
    tree: _SpanTree,
    spans: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The spans attributable to the selected pass, for cost roll-up.

    A span with no attributable pass is left out rather than shared between
    passes: a cost that cannot be attributed is better reported as the
    `unrecorded` it already is than split by guesswork.
    """
    if selector is None:
        return list(spans)
    selected = []
    for span in spans:
        span_id = _text_or_none(span.get("span_id"))
        if _pass_id_for(selector, tree, span_id) == selector.pass_id:
            selected.append(span)
    return selected


def _root_duration_ns(steps: Sequence[ExecutionStep]) -> tuple[Optional[int], int]:
    """Summed duration of ROOT dispatches only, and how many had none.

    Root means "no parent among the steps in this list". A nested dispatch's
    time is already inside its parent's span, and the ledger deliberately
    lists both -- so summing every row would double-count exactly the nested
    wrappers this projection must not double-count. Returns None, never 0,
    when nothing recorded a duration.
    """
    known = {step.command_call_id for step in steps}
    total = 0
    counted = 0
    missing = 0
    for step in steps:
        if step.parent_call_id in known:
            continue
        if step.duration_ns is None:
            missing += 1
            continue
        total += step.duration_ns
        counted += 1
    return (total if counted else None), missing


def project_execution(
    ref: ExecutionRef,
    reader: ExecutionReader,
    *,
    ledger: Optional[LedgerProjection] = None,
    cost_rollup: Optional[CostRollup] = None,
    pass_selector: Optional[PassSelector] = None,
) -> ExecutionProjection:
    """Read one scoped execution into the shared projection.

    Read-only and side-effect free. Every turn named by the reference is
    attempted; one that cannot be read is recorded in `unavailable` and the
    rest still project. A turn that IS readable but whose recorded
    experiment/task/attempt contradicts the reference is a refusal, not a
    partial answer (`_check_scope`).

    Pass scope and pass identity must agree, in all three directions:

    - `ref.pass_id` with no `pass_selector` is refused. Nothing here can resolve
      pass membership without one, so the projection would be the whole turn
      under a pass's name -- the wrong answer this module exists to avoid.
    - a `pass_selector` with no `ref.pass_id` is refused: the steps would be one
      pass's while `ref_id()` -- and so every feedback anchor and review-pair
      key derived from it -- would be the whole turn's.
    - both set and disagreeing is refused.

    The selector is then resolved against each turn's recorded spans, so a pass
    the evidence does not contain fails loudly (`UnknownRecordedPass`).
    """
    if ref.pass_id is not None and pass_selector is None:
        raise InvalidExecutionRef(
            f"reference names pass {ref.pass_id!r} but no pass selector was "
            "supplied; pass membership is resolved from recorded spans and is "
            "never assumed"
        )
    if pass_selector is not None and ref.pass_id is None:
        raise InvalidExecutionRef(
            f"a pass selector for {pass_selector.pass_id!r} was supplied for a "
            "reference that names no pass; the projection would be pass-scoped "
            "while its ref_id, anchors and review pairs would be the whole "
            "turn's"
        )
    if pass_selector is not None and pass_selector.pass_id != ref.pass_id:
        raise InvalidExecutionRef(
            f"reference names pass {ref.pass_id!r} but the selector resolves "
            f"pass {pass_selector.pass_id!r}"
        )
    ledger_fn = ledger or default_ledger_projection()
    cost_fn = cost_rollup or default_cost_rollup()

    turns: list[TurnProjection] = []
    steps: list[ExecutionStep] = []
    unassigned: list[ExecutionStep] = []
    artifacts: list[ArtifactRef] = []
    unattributed_artifacts: list[ArtifactRef] = []
    unavailable: list[str] = []
    cost_parts: list[Mapping[str, Any]] = []
    wall_ms_total = 0
    wall_ms_known = 0

    for turn_index, turn_key in enumerate(ref.turn_keys):
        row = reader.turn(ref.store_id, turn_key)
        if row is None:
            unavailable.append(f"turn {turn_key!r} is not in store {ref.store_id!r}")
            continue
        _check_scope(ref, turn_key, row)
        spans = list(reader.trace(ref.store_id, turn_key))
        tree = _SpanTree(spans)
        if pass_selector is not None:
            pass_selector.resolve_against(tree, turn_key)
        record = row.get("record")
        ledger_rows = ledger_fn(record, spans)
        recorded_params = _recorded_parameters(record)

        by_span: dict[str, Mapping[str, Any]] = {
            _text_or_none(span.get("span_id")): span
            for span in spans
            if _text_or_none(span.get("span_id"))
        }

        turn_steps: list[ExecutionStep] = []
        for raw in ledger_rows.get("rows") or []:
            call_id = _text_or_none(raw.get("command_call_id"))
            if call_id is None:
                continue
            span_id = _text_or_none(raw.get("span_id"))
            parameters: Any = None
            source: Optional[str] = None
            response_success: Optional[bool] = None
            if call_id in recorded_params:
                parameters, response_success = recorded_params[call_id]
                if parameters is not None:
                    source = "record"
            if parameters is None and span_id in by_span:
                span = by_span[span_id]
                if span.get("name") == SPAN_COMMAND_EXECUTE:
                    candidate = tree.attributes(span).get("parameters")
                    if candidate is not None:
                        parameters = candidate
                        source = "span"
            step = ExecutionStep(
                turn_index=turn_index,
                turn_key=turn_key,
                position=int(raw.get("position") or 0),
                command_call_id=call_id,
                parent_call_id=_text_or_none(raw.get("parent_call_id")),
                command_ordinal=_exact_int(raw.get("command_ordinal")),
                span_id=span_id,
                command_name=_text_or_none(raw.get("command_name")),
                context=_text_or_none(raw.get("context")),
                status=_text_or_none(raw.get("status")),
                success=raw.get("success") if isinstance(raw.get("success"), bool) else None,
                start_ns=_exact_int(raw.get("start_ns")),
                duration_ns=_exact_int(raw.get("duration_ns")),
                in_record=bool(raw.get("in_record")),
                span_recorded=bool(raw.get("span_recorded")),
                child_call=bool(raw.get("child_call")),
                asked_user=int(raw.get("asked_user") or 0),
                parameters=parameters,
                parameters_source=source,
                parameters_digest=(
                    _digest(_canonical_json(parameters)) if parameters is not None else None
                ),
                response_success=response_success,
                pass_id=_pass_id_for(pass_selector, tree, span_id),
            )
            if pass_selector is not None and step.pass_id != pass_selector.pass_id:
                unassigned.append(step)
                continue
            turn_steps.append(step)

        pass_spans = _spans_in_pass(pass_selector, tree, spans)
        turn_cost = dict(cost_fn(pass_spans))
        cost_parts.append(turn_cost)

        started = _text_or_none(row.get("started_at"))
        completed = _text_or_none(row.get("completed_at"))
        wall = _wall_ms(started, completed)
        if wall is not None:
            wall_ms_total += wall
            wall_ms_known += 1

        turns.append(
            TurnProjection(
                turn_index=turn_index,
                turn_key=turn_key,
                status=_text_or_none(row.get("status")),
                success=_bool_column(row.get("success")),
                failure_reason=_text_or_none(row.get("failure_reason")),
                answer=row.get("answer") if isinstance(row.get("answer"), str) else None,
                user_message=(
                    row.get("user_message")
                    if isinstance(row.get("user_message"), str)
                    else None
                ),
                started_at=started,
                completed_at=completed,
                suspended_ms=_exact_int(row.get("suspended_ms")),
                experiment_id=_text_or_none(row.get("experiment_id")),
                task_id=_text_or_none(row.get("task_id")),
                attempt=_exact_int(row.get("attempt")),
                ledger_summary={
                    key: ledger_rows.get(key)
                    for key in (
                        "record_rows",
                        "span_rows",
                        "rows_not_in_record",
                        "rows_without_span",
                        "asked_user_outside_dispatch",
                    )
                },
                cost=turn_cost,
                pass_id=ref.pass_id,
                content_attribution=(
                    ATTRIBUTION_SHARED if pass_selector else ATTRIBUTION_TURN
                ),
                # Nothing in this repo records a per-pass answer or plan, so a
                # pass-scoped turn projection never carries pass-specific
                # content. Flipping this is the producer-side work in fix-txxy.
                pass_content_recorded=False,
            )
        )
        steps.extend(turn_steps)
        # Artifacts follow their dispatch. In a pass-scoped projection an
        # artifact whose CommandOutput recorded no command_call_id cannot be
        # joined to one, so it is reported as the turn's unattributed output
        # rather than shown under both passes as if each had produced it.
        in_pass = {step.command_call_id for step in turn_steps}
        for artifact in _artifacts_from_record(turn_key, record):
            if pass_selector is None:
                artifacts.append(artifact)
            elif artifact.command_call_id is None:
                unattributed_artifacts.append(
                    replace(artifact, attribution=ATTRIBUTION_UNATTRIBUTED)
                )
            elif artifact.command_call_id in in_pass:
                artifacts.append(replace(artifact, attribution=ATTRIBUTION_PASS))

    duration_ns, steps_without_duration = _root_duration_ns(steps)
    turn_wall_ms = wall_ms_total if wall_ms_known else None
    timing = {
        "turns": len(turns),
        # Wall time is recorded per TURN. A pass has none of its own -- two
        # passes share one turn row's started_at/completed_at -- so a
        # pass-scoped projection reports None here and the shared turn figure
        # beside it, rather than presenting the whole turn's duration as the
        # pass's. `root_step_duration_ns` and `cost` below ARE pass-scoped:
        # they are summed over the steps and spans attributed to the pass.
        "wall_ms": None if pass_selector else turn_wall_ms,
        "wall_ms_attribution": (
            ATTRIBUTION_SHARED if pass_selector else ATTRIBUTION_TURN
        ),
        "turn_wall_ms": turn_wall_ms,
        "wall_ms_turns_recorded": wall_ms_known,
        "wall_ms_turns_unrecorded": len(turns) - wall_ms_known,
        "root_step_duration_ns": duration_ns,
        "root_steps_without_duration": steps_without_duration,
        "steps": len(steps),
    }
    return ExecutionProjection(
        ref=ref,
        turns=tuple(turns),
        steps=tuple(steps),
        artifacts=tuple(artifacts),
        timing=timing,
        cost=_merge_cost(cost_parts),
        unavailable=tuple(unavailable),
        unassigned_steps=tuple(unassigned),
        unattributed_artifacts=tuple(unattributed_artifacts),
        pass_selector=pass_selector,
    )


def _bool_column(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _wall_ms(started_at: Optional[str], completed_at: Optional[str]) -> Optional[int]:
    """Recorded wall time of one turn, or None. Never a negative duration."""
    if not started_at or not completed_at:
        return None
    from datetime import datetime

    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
    except (TypeError, ValueError):
        return None
    delta = int((end - start).total_seconds() * 1000)
    return delta if delta >= 0 else None


def _merge_cost(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-turn roll-ups, keeping `total: None` when nothing recorded one."""
    calls = recorded = unrecorded = 0
    total = 0.0
    for part in parts:
        calls += int(part.get("calls") or 0)
        recorded += int(part.get("recorded") or 0)
        unrecorded += int(part.get("unrecorded") or 0)
        value = part.get("total")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return {
        "calls": calls,
        "recorded": recorded,
        "unrecorded": unrecorded,
        "total": total if recorded else None,
    }


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedPair:
    """One row of the comparison: two steps, one step, or an ambiguous match.

    `ambiguous` is not a soft form of `matched`. It means the recorded
    evidence admits more than one correspondence -- a weaker key, or a key
    that repeats -- and the UI is expected to show both sides without
    asserting they are the same call.
    """

    kind: str
    basis: str
    left: Optional[ExecutionStep] = None
    right: Optional[ExecutionStep] = None
    ambiguous: bool = False
    ambiguity_reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "basis": self.basis,
            "ambiguous": self.ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "left": self.left.as_dict() if self.left else None,
            "right": self.right.as_dict() if self.right else None,
        }


@dataclass(frozen=True)
class Alignment:
    """The pairs plus the counts a reader needs before reading them."""

    pairs: tuple[AlignedPair, ...]
    degraded: bool = False
    degraded_reason: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        matched = sum(1 for pair in self.pairs if pair.kind == PAIR_MATCHED)
        return {
            "pairs": len(self.pairs),
            "matched": matched,
            "ambiguous": sum(1 for pair in self.pairs if pair.ambiguous),
            "left_only": sum(1 for pair in self.pairs if pair.kind == PAIR_LEFT_ONLY),
            "right_only": sum(1 for pair in self.pairs if pair.kind == PAIR_RIGHT_ONLY),
            "unknown": sum(1 for pair in self.pairs if pair.basis == BASIS_UNKNOWN),
            "recorded_matches": sum(
                1 for pair in self.pairs if pair.basis == BASIS_RECORDED
            ),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def _keys_for(step: ExecutionStep) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """The three key tiers for one step, strongest first; None where unusable.

    `child_call` is part of every tier: a span-less inner hop and a top-level
    dispatch of the same command are different rows in the ledger and must not
    be matched to each other, or a wrapper on one side would silently absorb a
    real call on the other.
    """
    name = step.command_name
    if name is None:
        return (None, None, None)
    marker = "child" if step.child_call else "root"
    context = step.context or ""
    weak = _digest(marker, name)
    medium = _digest(marker, name, context)
    strong = (
        _digest(marker, name, context, step.parameters_digest)
        if step.parameters_digest is not None
        else None
    )
    return (strong, medium, weak)


def _lcs_pairs(
    left: Sequence[int],
    right: Sequence[int],
    left_keys: Sequence[Optional[str]],
    right_keys: Sequence[Optional[str]],
) -> list[tuple[int, int]]:
    """Longest common subsequence over key equality, ties to the earlier index.

    Order-preserving and content-based: nothing here matches by list position,
    and inserted, removed and repeated calls fall out as gaps rather than
    being paired up to make the lists the same length.
    """
    n, m = len(left), len(right)
    if n == 0 or m == 0:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ki = left_keys[left[i]]
        row = table[i]
        nxt = table[i + 1]
        for j in range(m - 1, -1, -1):
            if ki is not None and ki == right_keys[right[j]]:
                row[j] = nxt[j + 1] + 1
            else:
                row[j] = nxt[j] if nxt[j] >= row[j + 1] else row[j + 1]
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        ki = left_keys[left[i]]
        if ki is not None and ki == right_keys[right[j]]:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _greedy_pairs(
    left: Sequence[int],
    right: Sequence[int],
    left_keys: Sequence[Optional[str]],
    right_keys: Sequence[Optional[str]],
) -> list[tuple[int, int]]:
    """Order-preserving first-fit, used only when the LCS would be too large."""
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for li in left:
        key = left_keys[li]
        if key is None:
            continue
        for offset in range(cursor, len(right)):
            if right_keys[right[offset]] == key:
                pairs.append((li, right[offset]))
                cursor = offset + 1
                break
    return pairs


def _align_segment(
    left_idx: list[int],
    right_idx: list[int],
    left_steps: Sequence[ExecutionStep],
    right_steps: Sequence[ExecutionStep],
) -> tuple[list[tuple[int, int, str, bool, Optional[str]]], bool]:
    """Align one gap by descending key strength; returns pairs and a degraded flag.

    Three passes, strongest key first, each one only over what the previous
    left unmatched. A pass that matches on a weaker key says so in its basis,
    so "these two are the same call" and "these two are both `add_todo`" are
    never reported as the same claim.
    """
    matched: list[tuple[int, int, str, bool, Optional[str]]] = []
    remaining_left = list(left_idx)
    remaining_right = list(right_idx)
    degraded = False
    tiers = (
        (0, BASIS_COMMAND_CONTEXT_PARAMETERS),
        (1, BASIS_COMMAND_CONTEXT),
        (2, BASIS_COMMAND),
    )
    left_keys_all = [_keys_for(step) for step in left_steps]
    right_keys_all = [_keys_for(step) for step in right_steps]
    for tier, basis in tiers:
        if not remaining_left or not remaining_right:
            break
        left_keys = [keys[tier] for keys in left_keys_all]
        right_keys = [keys[tier] for keys in right_keys_all]
        if len(remaining_left) * len(remaining_right) > _MAX_ALIGNMENT_CELLS:
            degraded = True
            found = _greedy_pairs(remaining_left, remaining_right, left_keys, right_keys)
        else:
            found = _lcs_pairs(remaining_left, remaining_right, left_keys, right_keys)
        if not found:
            continue
        # A key that occurs more than once on either side inside this segment
        # cannot distinguish which repetition is which. The pair still shows,
        # labelled, rather than being withheld or silently asserted.
        left_counts: dict[str, int] = {}
        for index in remaining_left:
            key = left_keys[index]
            if key is not None:
                left_counts[key] = left_counts.get(key, 0) + 1
        right_counts: dict[str, int] = {}
        for index in remaining_right:
            key = right_keys[index]
            if key is not None:
                right_counts[key] = right_counts.get(key, 0) + 1
        for li, ri in found:
            key = left_keys[li]
            repeated = (
                key is not None
                and (left_counts.get(key, 0) > 1 or right_counts.get(key, 0) > 1)
            )
            weak = basis == BASIS_COMMAND
            reason = None
            if repeated:
                reason = "repeated-key"
            elif weak:
                reason = "command-name-only"
            matched.append((li, ri, basis, repeated or weak, reason))
        paired_left = {li for li, _, _, _, _ in matched}
        paired_right = {ri for _, ri, _, _, _ in matched}
        remaining_left = [i for i in remaining_left if i not in paired_left]
        remaining_right = [i for i in remaining_right if i not in paired_right]
    return matched, degraded


def align_steps(
    left_steps: Sequence[ExecutionStep],
    right_steps: Sequence[ExecutionStep],
    *,
    recorded_alignment: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Alignment:
    """The shared alignment every consumer reads -- UI, API and extractor alike.

    A supplied structured alignment is authoritative: its pairs are emitted with
    `basis="recorded"` and are never second-guessed. Steps it does not mention
    are then aligned deterministically between its anchors, so a partial
    alignment adds information instead of hiding the rest of the run.

    It is also VERIFIED against the projected steps: every entry must name a
    `command_call_id` present on its side, and one that does not is refused.
    Nothing in this repo writes such an alignment today -- there is no alignment
    table and `list_distillation_runs` is a stub -- so the input is external by
    definition, and dropping the entries that do not resolve would report a
    rejected alignment as an accepted one.

    With no supplied alignment the whole thing is one deterministic segment,
    which is what every caller gets today. Either way the result is a function
    of the recorded steps alone, so a browser and an agent reading the same
    evidence get the same pairs and neither recomputes a diff of its own.
    """
    recorded_pairs: list[tuple[int, int]] = []
    out_of_order: list[tuple[int, int]] = []
    if recorded_alignment is not None:
        left_by_call = {step.command_call_id: i for i, step in enumerate(left_steps)}
        right_by_call = {step.command_call_id: i for i, step in enumerate(right_steps)}
        raw: list[tuple[int, int]] = []
        for entry in recorded_alignment:
            if not isinstance(entry, Mapping):
                raise InvalidRecordedAlignment(
                    f"an alignment entry must be an object, not {entry!r}"
                )
            left_call = _text_or_none(entry.get("left_command_call_id"))
            right_call = _text_or_none(entry.get("right_command_call_id"))
            li = left_by_call.get(left_call)
            ri = right_by_call.get(right_call)
            if li is None or ri is None:
                unknown = [
                    f"left_command_call_id={left_call!r}" if li is None else None,
                    f"right_command_call_id={right_call!r}" if ri is None else None,
                ]
                raise InvalidRecordedAlignment(
                    "alignment entry names "
                    + " and ".join(part for part in unknown if part)
                    + ", which the projected steps do not contain"
                )
            raw.append((li, ri))
        raw.sort()
        seen_left: set[int] = set()
        seen_right: set[int] = set()
        last_right = -1
        for li, ri in raw:
            if li in seen_left or ri in seen_right:
                continue
            seen_left.add(li)
            seen_right.add(ri)
            if ri > last_right:
                recorded_pairs.append((li, ri))
                last_right = ri
            else:
                # A stored alignment that crosses itself cannot also order the
                # gaps around it. The pair is kept -- it is recorded evidence --
                # but it stops being an anchor.
                out_of_order.append((li, ri))

    anchored_left = {li for li, _ in recorded_pairs} | {li for li, _ in out_of_order}
    anchored_right = {ri for _, ri in recorded_pairs} | {ri for _, ri in out_of_order}

    pairs: list[AlignedPair] = []
    degraded = False

    def emit_segment(l_start: int, l_end: int, r_start: int, r_end: int) -> None:
        nonlocal degraded
        left_idx = [
            i for i in range(l_start, l_end) if i not in anchored_left
        ]
        right_idx = [
            j for j in range(r_start, r_end) if j not in anchored_right
        ]
        matched, seg_degraded = _align_segment(
            left_idx, right_idx, left_steps, right_steps
        )
        degraded = degraded or seg_degraded
        by_left = {li: (ri, basis, amb, reason) for li, ri, basis, amb, reason in matched}
        by_right = {ri: li for li, ri, _, _, _ in matched}
        li_cursor = 0
        ri_cursor = 0
        left_list = left_idx
        right_list = right_idx
        while li_cursor < len(left_list) or ri_cursor < len(right_list):
            if li_cursor < len(left_list):
                li = left_list[li_cursor]
                if li in by_left:
                    ri, basis, amb, reason = by_left[li]
                    # Everything on the right before this partner is an
                    # insertion, and is emitted before the pair so the reader
                    # sees it where it happened.
                    while ri_cursor < len(right_list) and right_list[ri_cursor] != ri:
                        rj = right_list[ri_cursor]
                        if rj not in by_right:
                            pairs.append(_unmatched(right_steps[rj], PAIR_RIGHT_ONLY))
                        ri_cursor += 1
                    pairs.append(
                        AlignedPair(
                            kind=PAIR_MATCHED,
                            basis=basis,
                            left=left_steps[li],
                            right=right_steps[ri],
                            ambiguous=amb,
                            ambiguity_reason=reason,
                        )
                    )
                    li_cursor += 1
                    ri_cursor += 1
                    continue
                pairs.append(_unmatched(left_steps[li], PAIR_LEFT_ONLY))
                li_cursor += 1
                continue
            rj = right_list[ri_cursor]
            if rj not in by_right:
                pairs.append(_unmatched(right_steps[rj], PAIR_RIGHT_ONLY))
            ri_cursor += 1

    previous_left = 0
    previous_right = 0
    for li, ri in recorded_pairs:
        emit_segment(previous_left, li, previous_right, ri)
        pairs.append(
            AlignedPair(
                kind=PAIR_MATCHED,
                basis=BASIS_RECORDED,
                left=left_steps[li],
                right=right_steps[ri],
            )
        )
        previous_left = li + 1
        previous_right = ri + 1
    emit_segment(previous_left, len(left_steps), previous_right, len(right_steps))

    for li, ri in out_of_order:
        pairs.append(
            AlignedPair(
                kind=PAIR_MATCHED,
                basis=BASIS_RECORDED,
                left=left_steps[li],
                right=right_steps[ri],
                ambiguous=True,
                ambiguity_reason="recorded-alignment-out-of-order",
            )
        )

    return Alignment(
        pairs=tuple(pairs),
        degraded=degraded,
        degraded_reason=(
            "step count exceeded the quadratic alignment budget; matches are "
            "order-preserving first-fit"
            if degraded
            else None
        ),
    )


def _unmatched(step: ExecutionStep, kind: str) -> AlignedPair:
    """An unmatched step, saying whether it had a usable key at all."""
    strong, _, weak = _keys_for(step)
    unknown = weak is None
    return AlignedPair(
        kind=kind,
        basis=BASIS_UNKNOWN if unknown else BASIS_UNMATCHED,
        left=step if kind == PAIR_LEFT_ONLY else None,
        right=step if kind == PAIR_RIGHT_ONLY else None,
        ambiguous=unknown,
        ambiguity_reason="no-recorded-command-name" if unknown else None,
    )


# ----------------------------------------------------------------------
# Evidence anchors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAnchor:
    """Where a comment attaches, in the vocabulary the store already accepts.

    `target_kind` and `span_ids` are exactly what
    `ObservabilityStore.add_human_feedback` validates, so a feedback worker
    can pass these straight through without translating. This module defines
    the anchor and does NOT record feedback: storage, the three categories and
    their subcategories belong to the feedback slice.

    `anchorable` is False for a dispatch the trace has no span for (a span-less
    inner hop). The store requires at least one recorded span for anything
    finer than turn scope, so such a step can only be commented on at
    `fallback_target_kind` -- which is a real limitation of the evidence, worth
    showing rather than working around.
    """

    ref_id: str
    store_id: str
    turn_key: str
    target_kind: str
    span_ids: tuple[str, ...]
    command_call_id: Optional[str] = None
    step_position: Optional[int] = None
    pass_id: Optional[str] = None

    @property
    def anchorable(self) -> bool:
        return self.target_kind == "turn" or bool(self.span_ids)

    @property
    def fallback_target_kind(self) -> str:
        return "turn"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "store_id": self.store_id,
            "turn_key": self.turn_key,
            "target_kind": self.target_kind,
            "span_ids": list(self.span_ids),
            "command_call_id": self.command_call_id,
            "step_position": self.step_position,
            "pass_id": self.pass_id,
            "anchorable": self.anchorable,
        }


def anchor_for_step(ref: ExecutionRef, step: ExecutionStep) -> EvidenceAnchor:
    return EvidenceAnchor(
        ref_id=ref.ref_id(),
        store_id=ref.store_id,
        turn_key=step.turn_key,
        target_kind="step" if step.span_id else "turn",
        span_ids=(step.span_id,) if step.span_id else (),
        command_call_id=step.command_call_id,
        step_position=step.position,
        pass_id=step.pass_id,
    )


def anchor_for_turn(ref: ExecutionRef, turn: TurnProjection) -> EvidenceAnchor:
    return EvidenceAnchor(
        ref_id=ref.ref_id(),
        store_id=ref.store_id,
        turn_key=turn.turn_key,
        target_kind="turn",
        span_ids=(),
        pass_id=ref.pass_id,
    )


def anchors_for_pair(
    comparison: "ExecutionComparison", pair: AlignedPair
) -> dict[str, Any]:
    """Anchors for one comparison row: one side, the other, or both.

    A comment on a matched pair is about both executions, which is why both
    anchors are returned rather than a single merged one -- the two sides live
    in different turns and possibly different stores, and the store anchors
    feedback per turn.
    """
    return {
        "left": (
            anchor_for_step(comparison.left.ref, pair.left).as_dict()
            if pair.left
            else None
        ),
        "right": (
            anchor_for_step(comparison.right.ref, pair.right).as_dict()
            if pair.right
            else None
        ),
    }


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionComparison:
    """Two projections and the alignment between them.

    `left` is the pinned side -- the winner, the selected best run, the
    teacher. That is a caller's choice of viewpoint and carries no claim that
    the left side is correct.
    """

    left: ExecutionProjection
    right: ExecutionProjection
    alignment: Alignment

    def summary(self) -> dict[str, Any]:
        summary = dict(self.alignment.summary())
        summary.update(
            {
                "left_readable": self.left.readable,
                "right_readable": self.right.readable,
                "left_steps": len(self.left.steps),
                "right_steps": len(self.right.steps),
                "left_unavailable": len(self.left.unavailable),
                "right_unavailable": len(self.right.unavailable),
                "left_turns": len(self.left.turns),
                "right_turns": len(self.right.turns),
            }
        )
        return summary

    def differences(self) -> list[AlignedPair]:
        """The rows a "Differences only" view shows.

        A matched pair counts as a difference when the recorded outcome or the
        recorded parameters differ; an unmatched or ambiguous row always does.
        """
        rows = []
        for pair in self.alignment.pairs:
            if pair.kind != PAIR_MATCHED or pair.ambiguous:
                rows.append(pair)
                continue
            left, right = pair.left, pair.right
            if left is None or right is None:  # pragma: no cover - matched has both
                rows.append(pair)
                continue
            if (
                left.status != right.status
                or left.success != right.success
                or left.parameters_digest != right.parameters_digest
            ):
                rows.append(pair)
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "alignment": self.alignment.as_dict(),
            "summary": self.summary(),
            "review_pair_key": review_pair_key(self.left.ref, self.right.ref),
        }


def compare_executions(
    left: ExecutionRef,
    right: ExecutionRef,
    reader: ExecutionReader,
    *,
    ledger: Optional[LedgerProjection] = None,
    cost_rollup: Optional[CostRollup] = None,
    left_pass: Optional[PassSelector] = None,
    right_pass: Optional[PassSelector] = None,
    recorded_alignment: Optional[Iterable[Mapping[str, Any]]] = None,
) -> ExecutionComparison:
    """Project both sides and align them. Read-only; records nothing.

    Both sides may be unreadable, one side may be, or both may be scoreless --
    none of that is refused. Comparing an execution with itself is legal and
    useful (it is how a reviewer confirms the alignment is behaving).
    """
    left_projection = project_execution(
        left,
        reader,
        ledger=ledger,
        cost_rollup=cost_rollup,
        pass_selector=left_pass,
    )
    right_projection = project_execution(
        right,
        reader,
        ledger=ledger,
        cost_rollup=cost_rollup,
        pass_selector=right_pass,
    )
    alignment = align_steps(
        left_projection.steps,
        right_projection.steps,
        recorded_alignment=recorded_alignment,
    )
    return ExecutionComparison(
        left=left_projection, right=right_projection, alignment=alignment
    )


def comparison_digest(comparison: ExecutionComparison) -> dict[str, Any]:
    """Metadata, ids and counts only -- no user text, answers or parameters.

    For probes and logs against corpora whose content must not be printed.
    Everything here is either a count, a boolean, an opaque digest or an
    identifier the store itself already treats as non-content.
    """

    def side(projection: ExecutionProjection) -> dict[str, Any]:
        return {
            "ref_id": projection.ref.ref_id(),
            "store_id": projection.ref.store_id,
            "turn_count": len(projection.ref.turn_keys),
            "experiment_id": projection.ref.experiment_id,
            "task_id": projection.ref.task_id,
            "attempt": projection.ref.attempt,
            "pass_id": projection.ref.pass_id,
            "turns_read": len(projection.turns),
            "unavailable": len(projection.unavailable),
            "steps": len(projection.steps),
            "unassigned_steps": len(projection.unassigned_steps),
            "steps_with_parameters": sum(
                1 for step in projection.steps if step.parameters_digest is not None
            ),
            "steps_without_span": sum(
                1 for step in projection.steps if not step.span_recorded
            ),
            "child_call_steps": sum(1 for step in projection.steps if step.child_call),
            "artifacts": len(projection.artifacts),
            "artifacts_offloaded": sum(
                1 for artifact in projection.artifacts if artifact.artifact_id
            ),
            "unattributed_artifacts": len(projection.unattributed_artifacts),
            "content_attribution": projection.content_attribution,
            "answers_recorded": sum(
                1 for turn in projection.turns if turn.answer is not None
            ),
            "timing": dict(projection.timing),
            "cost": dict(projection.cost),
        }

    return {
        "left": side(comparison.left),
        "right": side(comparison.right),
        "summary": comparison.summary(),
        "differences": len(comparison.differences()),
        "review_pair_key": review_pair_key(
            comparison.left.ref, comparison.right.ref
        ),
    }


__all__ = [
    "ATTRIBUTION_PASS",
    "ATTRIBUTION_SHARED",
    "ATTRIBUTION_TURN",
    "ATTRIBUTION_UNATTRIBUTED",
    "BASIS_COMMAND",
    "BASIS_COMMAND_CONTEXT",
    "BASIS_COMMAND_CONTEXT_PARAMETERS",
    "BASIS_RECORDED",
    "BASIS_UNKNOWN",
    "BASIS_UNMATCHED",
    "PAIR_LEFT_ONLY",
    "PAIR_MATCHED",
    "PAIR_RIGHT_ONLY",
    "AlignedPair",
    "Alignment",
    "ArtifactRef",
    "ComparisonError",
    "EvidenceAnchor",
    "ExecutionComparison",
    "ExecutionProjection",
    "ExecutionReader",
    "ExecutionRef",
    "ExecutionScopeMismatch",
    "ExecutionStep",
    "InvalidExecutionRef",
    "InvalidRecordedAlignment",
    "PassSelector",
    "StoreExecutionReader",
    "TurnProjection",
    "UnknownRecordedPass",
    "WorkspaceExecutionReader",
    "align_steps",
    "anchor_for_step",
    "anchor_for_turn",
    "anchors_for_pair",
    "compare_executions",
    "comparison_digest",
    "default_cost_rollup",
    "default_ledger_projection",
    "discover_pass_selectors",
    "project_execution",
    "review_pair_key",
]
