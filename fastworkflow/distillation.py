"""
Runtime insights distillation module for planning and execution agents.

Runs a teacher (large LLM) and student (small LLM) on the same user query,
compares their planning decisions and execution actions, and extracts insights
when the student diverges from the teacher.

Two types of insights are extracted:
- **Planning insights** ("what TO DO"): Prescriptive rules for the planner,
  stored in `planning_agent_insights.md`
- **Execution insights** ("what NOT to do"): Anti-patterns for the execution agent,
  stored in `execution_agent_anti_patterns.md`

Planning comparison uses the generated plans from `build_query_with_next_steps`.
Execution comparison uses actual resolved command_name and parameters from the
in-process WEC action log (`ctx.action_log`), snapshotted per pass.
Full ReAct trajectories are passed to the insight extraction LLM for richer context.
"""

import copy
import hashlib
import inspect
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import dspy

import fastworkflow
from fastworkflow import tracing
from fastworkflow import distillation_alignment as alignment
from fastworkflow.observability_store import COUNT_LIVE_SPANS
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils

# Envelope version for distillation_runs.run_json, so a reader can pin the
# shape it parses.
_RUN_JSON_VERSION = 1


def _new_run_id() -> str:
    """Mint a distillation run id, mirroring [DR31]'s 'ins-<12 hex>' shape."""
    return f"run-{uuid.uuid4().hex[:12]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Insight identity (§13.1, [DR31])
#
# The id carries `run_id` on purpose: two runs producing identical text are two
# independent pieces of evidence, not one row, and support counting needs both.
# It is deliberately NOT the file entry number — `_append_numbered_insights`
# renumbers by regex over the file, so a hand edit or a renumber would orphan
# every citation built on it. `file_entry_number` is stored for display only.
# Deduplication across runs is a view grouped on `text_hash`, which is also the
# reverse index from a markdown line (where only the text survives).
# ---------------------------------------------------------------------------

INSIGHT_KIND_PLANNING = "planning"
INSIGHT_KIND_EXECUTION = "execution"

# fw.distill.extract.empty_reason (§13.3). The two are not the same failure:
# EXTRACTOR_RETURNED_EMPTY is the model judging the difference context-justified
# or a duplicate, PARSE_YIELDED_NOTHING is the model answering in a shape the
# parser kept none of — one says the extractor is too conservative, the other
# says the parser is too strict. Today both are the same silence.
EMPTY_REASON_EXTRACTOR = "extractor-returned-empty"
EMPTY_REASON_PARSE = "parse-yielded-nothing"

# Which alignment level feeds which extractor, so a stored record can be cited
# by the insight the summary describing it produced (§13.2).
_INSIGHT_KIND_BY_LEVEL = {
    alignment.LEVEL_PLAN: INSIGHT_KIND_PLANNING,
    alignment.LEVEL_ACTION: INSIGHT_KIND_EXECUTION,
}


def normalize_insight_text(text: str) -> str:
    """The normalized form both the id and the text hash are taken over (§13.1)."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".;,")


def insight_text_hash(text: str) -> str:
    """`text_hash`: the cross-run reverse index from a markdown line (§13.1)."""
    return hashlib.sha256(
        normalize_insight_text(text).encode("utf-8")
    ).hexdigest()[:16]


def insight_id(run_id: str, kind: str, text: str) -> str:
    """A stable `ins-<12 hex>` id for one emitted insight (§13.1)."""
    seed = f"{run_id}|{kind}|{normalize_insight_text(text)}"
    return f"ins-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def dedupe_insight_texts(texts: list[str]) -> list[str]:
    """One entry per distinct `normalize_insight_text` form, first spelling kept.

    `[DR31]` seeds the id with `run_id|kind|normalized`, so two bullets of ONE
    extractor call that differ only in case or a trailing period — exactly what
    `normalize_insight_text` folds — mint the SAME `insight_id`. Without this
    the file gets two lines carrying one marker while
    `ON CONFLICT(insight_id) DO UPDATE` keeps one row: the marker round trip is
    ambiguous forwards (one id, two lines) and lossy backwards (the surviving
    row's `file_entry_number` names the later line only).

    The fix is deduplication, not a richer seed: §13.1 keeps the file entry
    number OUT of the id on purpose — "a hand edit or a renumber would orphan
    every reference" — so it must not come back as a disambiguator. Cross-run
    duplicates already differ by `run_id` and are a `text_hash` view, not a
    collision.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        normalized = normalize_insight_text(text)
        if normalized in seen:
            logger.debug(
                "Distillation: dropping a duplicate insight in one extraction "
                f"({text!r} normalizes onto an earlier entry)"
            )
            continue
        seen.add(normalized)
        kept.append(text)
    return kept


@dataclass
class _ExtractionOutcome:
    """What one `fw.distill.extract` call produced, for the ledger to write.

    A negative outcome is a row, not an absence (§13.3): an extraction that
    kept nothing still carries its `empty_reason` and still sets
    `distillation_runs.extractor_empty`.
    """
    kind: str
    span_id: Optional[str]
    insights: list[str]
    insight_ids: list[str]
    empty_reason: Optional[str]
    cited_divergence_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Comparability fingerprints (§6.1, [DR14] [DR47])
#
# Two projections, not one. Revision 1 of the design hashed context *and* a
# conversation-history tail with one function and used it for four jobs; two of
# them degenerate, because every pass calls summarize_and_record_turn INSIDE
# the pass (an LLM call that appends its own generated summary to
# _conversation_history.messages) and teacher and student run different models.
# A history-bearing hash therefore differs across passes on every single run,
# which would make `material` and `restore_ok` constants.
# ---------------------------------------------------------------------------

# Hash-payload version, so a stored fingerprint is never compared against one
# computed under a different rule.
_FINGERPRINT_VERSION = 2

# The history tail that actually reaches the planner and agent prompts.
_HISTORY_TAIL_MESSAGES = 4

# Keys dropped from every canonicalized mapping, each for a stated reason
# (§6.1): a live-object handle whose contents are already hashed directly; the
# two keys the pass itself writes at entry, which would make every pass differ
# from every other by construction; and mid-pipeline scratch that is empty at a
# pass boundary and would only ever add noise.
_FINGERPRINT_DROP_KEYS = frozenset(
    {
        "app_workflow",
        "raw_user_message",
        "is_user_command",
        "stored_parameters",
        "NLU_Pipeline_Stage",
    }
)
# Wall clock, in the spellings this codebase uses.
_FINGERPRINT_DROP_KEY_RE = re.compile(r"_ts$|_at$|^started_|^updated_")


def _fingerprint_drops(key: Any) -> bool:
    return isinstance(key, str) and (
        key in _FINGERPRINT_DROP_KEYS or bool(_FINGERPRINT_DROP_KEY_RE.search(key))
    )


def _canonical_key(key: Any) -> str:
    """Mapping keys, rendered the same way values are (never a repr)."""
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, (int, float)):
        return _canonical(key)
    return f"<type:{type(key).__name__}>"


def _canonical(value: Any, _path: frozenset = frozenset()) -> Any:
    """Render *value* as a stable, JSON-safe projection ([DR47]).

    Keys are sorted, numbers become decimal strings, datetimes RFC-3339 UTC,
    list order is preserved, and **anything that is not a JSON scalar / list /
    dict becomes the structural token ``<type:Name>``** — never its ``repr``,
    never its ``str``. That last rule is the whole point of this function:
    ``cme._context`` always carries the live ``app_workflow`` object
    (`workflow_execution_context.py:1036`) and `fastworkflow/workflow.py`
    defines no ``__repr__``/``__str__``, so hashing through
    ``json.dumps(..., default=str)`` was hashing a **heap address** — unstable
    across processes, and therefore a gate no replay could ever pass.

    The consequence, published rather than hidden (§6.2): an application object
    living in a context dict contributes its type and nothing else, so an equal
    fingerprint attests comparable *inputs*, never a verified application world.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr is the shortest round-tripping form and is stable across
        # processes; str() is the same in py3 but repr says why.
        return repr(value)
    if isinstance(value, datetime):
        stamped = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return stamped.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        if id(value) in _path:
            # App-authored context may hold a cycle; a fingerprint must not hang.
            return "<cycle>"
        inner = _path | {id(value)}
        if isinstance(value, dict):
            return {
                _canonical_key(key): _canonical(item, inner)
                for key, item in sorted(
                    value.items(), key=lambda kv: _canonical_key(kv[0])
                )
                if not _fingerprint_drops(key)
            }
        return [_canonical(item, inner) for item in value]
    return f"<type:{type(value).__name__}>"


def _digest(payload: Any) -> str:
    """SHA-256 over the canonical JSON of *payload*, truncated to 32 hex.

    Deliberately no ``default=``: `_canonical` has already replaced everything
    JSON cannot render, so a ``TypeError`` here means the projection missed a
    case and must be fixed, rather than being papered over with an address.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]


def state_fingerprint(chat_session) -> str:
    """Stable hash of pass-entry (or pass-exit) WORLD state. [DR14][DR47].

    Deliberately excludes conversation history: every pass appends its own
    LLM-generated summary before it exits, so a history-bearing hash can never
    compare equal across passes.

    This is the **only** implementation in the codebase, by [DR13]: fix-35m.3
    consumes it for its teacher-mutated-nothing post-condition and must not
    define a second one.
    """
    workflow = chat_session.get_active_workflow()
    cme = chat_session.cme_workflow
    payload = {
        "v": _FINGERPRINT_VERSION,
        "workflow_context": _canonical(workflow._context) if workflow else None,
        "cme_context": _canonical(cme._context) if cme else None,
        "command_context": (
            workflow.current_command_context_name if workflow else None
        ),
        "is_complete": bool(workflow._is_complete) if workflow else None,
    }
    return _digest(payload)


def prompt_fingerprint(chat_session, *, history_bound: int) -> str:
    """Hash of the inputs a pass's prompts actually see. [DR47].

    `history_bound` is the length of conversation_history.messages captured at
    the ENTRY of the pass being measured; entry and exit are always computed at
    the same bound, so a pass's own appended summary is never inside its own
    exit hash.
    """
    msgs = chat_session.conversation_history.messages[:history_bound]
    payload = {
        "v": _FINGERPRINT_VERSION,
        "history_tail": [
            _canonical(msg) for msg in msgs[-_HISTORY_TAIL_MESSAGES:]
        ],
        "refined_user_message": getattr(
            chat_session, "current_refined_message", None
        ),
    }
    return _digest(payload)


def _safe_state_fingerprint(chat_session) -> Optional[str]:
    """`state_fingerprint`, or None when it could not be taken.

    None is what makes the run non-comparable ([DR40]: silence is never read as
    agreement), so a fingerprint that cannot be computed costs the run its
    aggregates rather than the user their turn.
    """
    try:
        return state_fingerprint(chat_session)
    except Exception as exc:
        logger.warning(f"Distillation: state_fingerprint failed: {exc!r}")
        return None


def _safe_prompt_fingerprint(chat_session, history_bound: int) -> Optional[str]:
    try:
        return prompt_fingerprint(chat_session, history_bound=history_bound)
    except Exception as exc:
        logger.warning(f"Distillation: prompt_fingerprint failed: {exc!r}")
        return None


def _seed_live_handles(value: Any, memo: dict, seen: set) -> None:
    """Map every non-data value in *value* to itself in a deepcopy memo."""
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        items = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return
    for item in items:
        if isinstance(item, (dict, list, tuple, set, frozenset)):
            _seed_live_handles(item, memo, seen)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            memo[id(item)] = item


def _deepcopy_context(context: Any) -> Any:
    """`copy.deepcopy` of a workflow context dict, live handles kept by identity.

    The deep copy is what makes a snapshot a snapshot (§5): `Workflow._to_dict()`
    returns `self._context` **by reference** (`workflow.py:450`), so an aliased
    snapshot is mutated by the very pass it is supposed to outlive — the restore
    becomes a no-op and any fingerprint taken across it reports agreement by
    construction.

    It cannot be a blanket deepcopy, though: the CME context holds the live
    `app_workflow` object that `intent_detection.py:34` reads back, and cloning
    it would install a duplicate Workflow — under the same workflow id — on
    restore. Non-data values are therefore seeded into deepcopy's own memo so
    they come back by reference, which is the same line `_canonical` draws when
    it renders them as opaque structural tokens.
    """
    if not isinstance(context, dict):
        return context
    memo: dict[int, Any] = {}
    try:
        _seed_live_handles(context, memo, set())
        return copy.deepcopy(context, memo)
    except Exception as exc:
        # A context that will not copy must not cost the user their turn; the
        # shallow copy still detaches the mapping itself, which is what the
        # restore path writes back.
        logger.warning(
            f"Distillation: context deep copy failed ({exc!r}); "
            "snapshotting one level only"
        )
        return dict(context)


def _announce(title: str, subtitle: str = "", style: str = "cyan") -> None:
    """
    Print an observability banner for a distillation phase.

    Distillation runs the agent twice (teacher then student) for a single user
    message, so without this the user cannot tell which model produced which
    output. Uses rich when available, else a plain print; never raises.
    """
    line = f"{title} — {subtitle}" if subtitle else title
    try:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(line, style=style, expand=False))
    except Exception:
        print(f"\n=== {line} ===")


# Per-pass token/cost/cache rollup (§6.3). Verified as a QUERY over attributes
# `utils/dspy_logger.py` already writes — `model` (:374), `usage` (:425), `cost`
# (:426), `cache_hit` (:429-430) — so distillation adds no LLM instrumentation
# of its own. Note the NESTED json_extract on `usage`: dspy_logger stores it
# through `_json_text(...)`, so it is a JSON *string* inside the attributes
# JSON, and the single-level form returns NULL silently.
_LLM_ROLLUP_SQL = """
SELECT distillation_pass AS pass_label,
       SUM(CASE WHEN json_extract(attributes,'$.cache_hit') THEN 1 ELSE 0 END) AS cache_hits,
       SUM(CASE WHEN json_extract(attributes,'$.cache_hit') THEN 0 ELSE 1 END) AS cache_misses,
       SUM(COALESCE(json_extract(attributes,'$.cost'), 0.0))                   AS cost_usd,
       SUM(COALESCE(json_extract(json_extract(attributes,'$.usage'),
                                 '$.total_tokens'), 0))                        AS tokens
FROM spans
WHERE trace_id = :turn_key AND name = 'fw.llm.call'
  AND distillation_pass IS NOT NULL
GROUP BY distillation_pass
"""

# The rollup reads spans the writer thread owns, so it takes the sink's own
# flush barrier first ([DR49]). Both bounds are short: this runs on the turn
# thread, and a rollup that cannot be taken costs the run its cost columns,
# never the user their turn.
_ROLLUP_FLUSH_TIMEOUT_S = 5.0
_ROLLUP_READ_TIMEOUT_S = 5.0

# entry_inputs_json envelope version, and the cap past which the diagnostic
# context snapshot is dropped rather than bloating a per-pass row (§10.2
# budgets ~6 KB for all of a run's pass rows together).
_ENTRY_INPUTS_VERSION = 1
_ENTRY_INPUTS_MAX_BYTES = 32 * 1024


# The writer-health counters a run's evidence can go missing through. Only
# `spans_dropped` was ever consulted, and only per pass, so the two record
# losses were unmonitored: a dropped `divergence` leaves a citation pointing at
# nothing, and a dropped `pass` pair leaves a run row with no passes — which
# also makes that run unpinnable, because the pinned-trace set joins through
# `distillation_passes`. `write_errors` covers the swallow in
# `_apply_distillation` (an IntegrityError from a NOT NULL column is counted,
# never raised) as well as serialize failures on the span path.
_WRITER_COUNTER_KEYS = ("spans_dropped", "records_dropped", "write_errors")


def _writer_counters(chat_session) -> Optional[dict[str, int]]:
    """The sink's live loss counters, or None when unavailable.

    Read off the sink's in-memory health rather than `writer_health()`, which
    is a DB read of a snapshot the writer flushes on its own schedule.
    """
    sink = tracing.get_sink(chat_session)
    health = getattr(sink, "_health", None)
    if not isinstance(health, dict):
        return None
    counters: dict[str, int] = {}
    for key in _WRITER_COUNTER_KEYS:
        try:
            counters[key] = int(health.get(key) or 0)
        except (TypeError, ValueError):
            counters[key] = 0
    return counters


def _spans_dropped(chat_session) -> Optional[int]:
    """The sink's live `spans_dropped` counter, or None when unavailable."""
    counters = _writer_counters(chat_session)
    return None if counters is None else counters["spans_dropped"]


def _text_digest(text: Optional[str]) -> Optional[dict]:
    """Byte length + SHA-256 of an insight corpus, never its body (§8)."""
    if not text:
        return None
    raw = text.encode("utf-8", "replace")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


# ---------------------------------------------------------------------------
# The comparable-unit spans a pass emitted ([DR17], [DR49])
#
# §7.1 aligns over `fw.command.execute` and `fw.ask_user` SPANS, not over
# `chat_session.action_log`: spans are the persisted record, each carries the
# `span_id` a divergence row needs for provenance, and they inherit none of
# `is_valid_action`'s filtering — which drops every failed command (the early
# returns in `command_executor._invoke_command_impl` leave `command_name`
# falsy), every ask_user record and every ErrorCorrection/* record.
#
# [DR49] then fixes WHERE they are read: the in-process `Span` objects the
# pass emitted, never a SELECT racing the writer thread. A span that is merely
# LATE would otherwise become a fabricated `missing-in-student` divergence,
# stored as structured evidence and, under §10.3, pinned forever. The DB read
# of `spans` stays a rendering concern (fix-sb8.8).
# ---------------------------------------------------------------------------

# The step spans of one pass, per level. fw.agent.execute is not a step: it
# supplies the run-level record's span pair (§7.3 step 6).
_ACTION_SPAN_NAMES = (tracing.SPAN_COMMAND_EXECUTE, tracing.SPAN_ASK_USER)
_PLAN_SPAN_NAMES = (tracing.SPAN_PLANNER_PLAN, tracing.SPAN_PLANNER_REPLAN)
_COLLECTED_SPAN_NAMES = frozenset(
    _ACTION_SPAN_NAMES + _PLAN_SPAN_NAMES + (tracing.SPAN_AGENT_EXECUTE,)
)

# The [DR49] barrier is on the turn thread, so it is bounded like the §6.3
# rollup's: a barrier that cannot be taken costs the run its comparability,
# never the user their turn.
_ALIGN_FLUSH_TIMEOUT_S = 5.0


def _flush_trace(flush: Any, trace_id: Optional[str], timeout: float) -> bool:
    """Take a [DR49] barrier scoped to `trace_id` where the sink supports it.

    Scoping is what keeps the barrier off other channels' backlogs
    (`fix-sb8.15`); a sink predating the keyword — or any test double with the
    older signature — still gets the unscoped barrier, which is correct, only
    starvable.

    Support is decided by INSPECTING the signature rather than by catching the
    `TypeError` a wrong call raises: a `TypeError` from inside `flush` would be
    indistinguishable from an unsupported keyword, and swallowing it would turn
    a real failure into a silently unscoped barrier.
    """
    if trace_id and _accepts_trace_id(flush):
        return bool(flush(timeout=timeout, trace_id=trace_id))
    return bool(flush(timeout=timeout))


def _accepts_trace_id(flush: Any) -> bool:
    """Whether this sink's `flush` takes the `trace_id` keyword."""
    try:
        parameters = inspect.signature(flush).parameters
    except (TypeError, ValueError):
        # A builtin or an unintrospectable callable: assume the older shape.
        return False
    return "trace_id" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _attr_text(value: Any) -> Optional[str]:
    """A string span attribute, seen past [R10]'s truncation envelope."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("truncated"):
        return value.get("value")
    return None


class _PassSpanCollector:
    """Holds this run's step spans as the sink emits them ([DR49]).

    Installed by shadowing the sink's own `emit_span` with `collect` for the
    length of the run, rather than by wrapping the sink in a proxy: the sink
    object, its class and every other attribute stay exactly what they were,
    so `isinstance(sink, SQLiteTraceSink)` and `sink.store` keep answering the
    way the rest of the process expects. The delegation is to the bound method
    captured at install time, so the real sink still does all the work.

    Spans are bucketed by `Span.distillation_pass`, which `start_span` stamps
    from the same ambient label that opened them — so the bucketing does not
    depend on §9 item 12's ask_user reparenting, which is not this child's.
    """

    def __init__(self, sink: Any):
        self._sink = sink
        self._emit_span = sink.emit_span
        self._installed: Any = None
        self._by_pass: dict[str, dict[str, Any]] = {}

    def collect(self, span) -> None:
        """The sink's `emit_span`, with this run's step spans kept on the way."""
        try:
            pass_label = getattr(span, "distillation_pass", None)
            if pass_label and span.name in _COLLECTED_SPAN_NAMES:
                # Keyed by span_id: fw.ask_user is emitted at open AND at
                # close, and one question must not become two steps.
                self._by_pass.setdefault(pass_label, {})[span.span_id] = span
        except Exception as exc:  # collection must never cost a turn its span
            logger.warning(f"Distillation: span collection failed: {exc!r}")
        self._emit_span(span)

    def install(self) -> bool:
        """Shadow `emit_span` on the sink instance. False if already shadowed."""
        if "emit_span" in vars(self._sink):
            return False
        # Held rather than re-derived: `self.collect` mints a fresh bound
        # method on every access, so `remove` could never recognise its own.
        self._installed = self.collect
        self._sink.emit_span = self._installed
        return True

    def remove(self) -> None:
        """Unshadow, leaving the class's own `emit_span` in place."""
        if vars(self._sink).get("emit_span") is self._installed:
            del self._sink.emit_span

    def spans_of(self, pass_label: str, names) -> list:
        """That pass's collected spans named in `names`, in `start_ns` order."""
        return sorted(
            (
                span
                for span in self._by_pass.get(pass_label, {}).values()
                if span.name in names
            ),
            key=lambda span: span.start_ns or 0,
        )


@dataclass
class PlanningStep:
    """Captures a single planning decision during agent execution."""
    step_number: int           # For correlating multi-turn planning decisions
    user_query: str
    generated_plan: list[str]  # List of next steps from planner
    reasoning: str = ""        # Chain-of-thought reasoning from the planner


@dataclass
class DistillationResult:
    """Result of a distillation run for a single message."""
    command_output: fastworkflow.CommandOutput
    planning_insights_extracted: int = 0
    execution_insights_extracted: int = 0
    # The distillation_runs row this pass pair was recorded under, so the CLI
    # summary can point at the record (fix-sb8.2). None when the run never got
    # far enough to mint one.
    run_id: Optional[str] = None

    @property
    def insights_extracted(self) -> int:
        """Total insights (planning + execution) for backward compatibility."""
        return self.planning_insights_extracted + self.execution_insights_extracted


class PlanningInsightExtractionSignature(dspy.Signature):
    """You are analyzing why a student planner generated an inferior plan compared to the teacher.

    The planner's job is to break down a user request into a sequence of workflow commands.
    A good plan identifies the RIGHT commands in the RIGHT order to fulfill the user's intent.

    You are given:
    - The user's original query
    - Teacher's plan (the correct approach)
    - Student's plan (may have issues)
    - A summary of how the plans diverged
    - Teacher's executed actions (what the teacher actually did — the ground truth)
    - Student's executed actions (what the student actually did)

    Use the executed actions to validate whether a plan led to correct behavior.
    A plan that looks different but leads to the same correct actions may be acceptable.

    CRITICAL — Context divergence awareness:
    In multi-turn conversations, teacher and student may have different conversation
    history context from prior turns. This can legitimately cause different planning
    decisions. Before extracting a rule, analyze whether:
    1. The planning difference was a genuine mistake (extract a rule)
    2. The difference was justified by different context (return EMPTY)

    YOUR TASK:

    1. **Analyze and Find Root Cause**:
    - Analyze the context each planner had access to
    - Identify if there was a genuine planning mistake vs context-justified difference
    - Look at Step 0 (initial planning) - this is usually where things go wrong
    - What did the student plan that was incorrect?
    - What did the teacher plan that was correct?
    - What actions are MISSING from student's execution?

    2. **Generate 1-3 Specific, Actionable Rules if genuine mistakes are found**:
    - Rules should be PRESCRIPTIVE
    - Rules should reference SPECIFIC command names
    - Rules should prevent THIS EXACT failure from happening again
    - Focus on the ROOT CAUSE, not symptoms

    Return rules as: 1. [rule], 2. [rule], 3. [rule]
    Return EMPTY if:
    - Student's plan was reasonable given its context
    - Rules duplicate existing insights"""

    user_query: str = dspy.InputField()
    teacher_plan: str = dspy.InputField(
        desc="The correct plan generated by the teacher planner"
    )
    student_plan: str = dspy.InputField(
        desc="The plan generated by the student planner (may have issues)"
    )
    divergence_summary: str = dspy.InputField(
        desc="Summary of how teacher and student plans differ"
    )
    teacher_actions: str = dspy.InputField(
        desc="Actions actually executed by the teacher agent (command names + parameters)"
    )
    student_actions: str = dspy.InputField(
        desc="Actions actually executed by the student agent (command names + parameters)"
    )
    existing_insights: str = dspy.InputField(
        desc="Already known planning rules (avoid duplicates)"
    )
    insights: str = dspy.OutputField(
        desc="1-3 prescriptive, workflow-general rules as numbered list. "
             "Return EMPTY if no genuine mistakes, context-justified, or duplicates."
    )


class InsightExtractionSignature(dspy.Signature):
    """Analyze teacher vs student execution agent ReAct trajectories for the same user query.
    Both trajectories contain the full sequence of thoughts, tool calls, arguments, and
    observations (including any ask_user interactions and conversation history context).

    A divergence summary describes the concrete differences in executed actions
    (command names + parameters). Use the full trajectories to understand WHY
    each agent made its choices.

    CRITICAL — Multi-turn context awareness:
    In multi-turn conversations, teacher and student may have received different
    conversation context from prior turns (visible in observation_about_conversation_history
    and ask_user observations). This can legitimately cause different action sequences.
    Before extracting an insight, determine whether the student's actions were
    reasonable given the context it actually had. Only extract insights for
    genuine mistakes — NOT for context-justified differences.

    Return insights as single-line bullet points (one per line, prefixed with "- ").
    You may return multiple insights if there are multiple distinct mistakes.
    Return EMPTY if differences are not genuine mistakes or duplicate existing insights."""
    user_query: str = dspy.InputField()
    teacher_trajectory: str = dspy.InputField(
        desc="Teacher's full ReAct trajectory (reference/correct behavior)"
    )
    student_trajectory: str = dspy.InputField(
        desc="Student's full ReAct trajectory (may contain mistakes)"
    )
    divergence_summary: str = dspy.InputField(
        desc="Human-readable summary of action-level differences between teacher and student "
             "(which commands/params differed, ordering differences, etc.)"
    )
    existing_insights: str = dspy.InputField(
        desc="Already known anti-patterns (avoid duplicates)"
    )
    insights: str = dspy.OutputField(
        desc="One or more anti-pattern insights as bullet points (each line prefixed with '- '). "
             "Each insight should be a concise, actionable single-line point. "
             "Return EMPTY if no genuine mistakes found, differences are context-justified, "
             "or insights duplicate existing ones."
    )


class DistillationSession:
    """Orchestrates a single distillation comparison for one user message."""

    def __init__(self, wec: "fastworkflow.WorkflowExecutionContext"):
        # `wec` is the WorkflowExecutionContext (session engine). Distillation is
        # CLI/Topology-A only and drives its own agent passes over the WEC's state.
        self.chat_session = wec
        self.planning_insights_extracted: int = 0
        self.execution_insights_extracted: int = 0
        # One distillation_runs row per compared message; the pass rows and
        # every fw.distill.* span carry it (§9).
        self.run_id: str = _new_run_id()
        # Everything written to that row so far. The row is opened when the run
        # starts and completed when it ends, and SQLite evaluates NOT NULL on
        # the row an upsert *attempts to insert* — before the conflict is
        # resolved — so a completion write carrying only the changed columns is
        # rejected for a missing user_message, and rejected silently, since the
        # writer thread counts a malformed record rather than raising [DR46].
        # Accumulating makes every write a whole row.
        self._run_row: dict[str, Any] = {}
        # Same trap on distillation_passes, whose role/seq/trace_id are also
        # NOT NULL: the cache/cost rollup lands after the pass row is first
        # written (§6.3), so that write must carry the whole row too.
        self._pass_rows: dict[str, dict[str, Any]] = {}
        # The comparability evidence itself, per pass label: the entry/exit
        # fingerprints [DR47], the history bound they were taken at, and the
        # writer's spans_dropped delta across the pass [DR49]. Ordered by seq
        # when the run row's verdict is computed.
        self._passes: dict[str, dict[str, Any]] = {}
        # Sampling parameters of the LMs a pass actually built, captured where
        # dspy_utils.get_lm resolves them (§6.2's model_params_json).
        self._lm_params: dict[str, Any] = {}
        # The in-process step spans the passes emit ([DR49]); None when there
        # is no sink to wrap, which is also what selects the legacy
        # action-log comparison in `alignment_available`.
        self._span_collector: Optional[_PassSpanCollector] = None
        # Set when the [DR49] read barrier could not be taken. Missing
        # evidence, never agreement: it forces `evidence-incomplete` exactly
        # like a pass whose spans_dropped counter moved.
        self._barrier_failed: bool = False
        # The writer's loss counters as they stood when the run opened, and
        # whatever they moved by while it ran. The per-pass `spans_dropped`
        # delta covers only the inside of a pass and is sampled BEFORE the
        # closing barrier runs, so on its own it cannot see a batch the writer
        # discarded at the barrier — nor either record-loss counter at all.
        self._writer_counters_entry: Optional[dict[str, int]] = None
        self._writer_loss: dict[str, int] = {}
        # What each extractor call produced, by insight kind (§13.3). Present
        # even when it produced nothing — that is the row the negative outcome
        # is recorded as.
        self._extractions: dict[str, _ExtractionOutcome] = {}
        # The divergence ids the summary handed to each extractor described,
        # by insight kind. They become that kind's citations (§13.2). v1 is one
        # run per insight, which is what the extractor is actually given;
        # cross-run consolidation is a view over `text_hash`, not a stored
        # relation (§18).
        self._cited_divergences: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------------
    # Run/pass records ([DR46])
    # ------------------------------------------------------------------

    def _emit_run_record(self, **fields: Any) -> None:
        """Upsert this run's distillation_runs row through the sink.

        Never a direct `ObservabilityStore._connect()` on the turn thread
        ([DR46]): the write is queued onto the sink's writer thread, so lock
        contention or an OperationalError can never surface inside
        `_execute_message`. Upsert on run_id, so the row is opened when the
        run starts and completed when it ends.
        """
        turn_key = self.chat_session.current_turn_key
        if not turn_key:
            # No open turn means nothing to key the record to; a live turn
            # always has one, so this is the no-observability path only.
            return
        # [DR48]: fix-sb8.3 may write NULL or 0 into isolation_verified and
        # nothing else. Only fix-35m.3's read-only surface check can write 1,
        # and until it exists a NULL must never be readable as a pass.
        if fields.get("isolation_verified") == 1:
            logger.warning(
                "Distillation: refusing to write isolation_verified=1 ([DR48]); "
                "only fix-35m.3 may assert isolation"
            )
            fields = {k: v for k, v in fields.items() if k != "isolation_verified"}
        self._run_row.update(fields)
        tracing.emit_distillation_record(
            self.chat_session,
            "run",
            {"run_id": self.run_id, "turn_key": turn_key, **self._run_row},
        )

    @property
    def trace_key(self) -> Optional[str]:
        """The trace this session's spans and rows belong to ([DR41]).

        The turn key for every live run; the derived `<turn_key>~replay.<n>`
        while a counterfactual replay is bound, where `current_turn_key` is
        deliberately None so the replay cannot write a `turns` row.
        """
        chat = self.chat_session
        return (
            getattr(chat, "current_replay_trace_id", None) or chat.current_turn_key
        )

    def _emit_pass_record(self, pass_label: str, **fields: Any) -> None:
        """Upsert one distillation_passes row, keyed (run_id, pass_label).

        Per-pass facts live here rather than as run columns, which is what
        makes the schema N-pass capable: adding a student adds a row, never a
        column (§4).
        """
        turn_key = self.trace_key
        if not turn_key:
            return
        row = self._pass_rows.setdefault(pass_label, {})
        row.update(fields)
        tracing.emit_distillation_record(
            self.chat_session,
            "pass",
            {
                "run_id": self.run_id,
                "pass_label": pass_label,
                "trace_id": turn_key,
                **row,
            },
        )

    # ------------------------------------------------------------------
    # State snapshot / restore
    # ------------------------------------------------------------------

    def snapshot_workflow_state(self) -> dict:
        """Capture current workflow + CME state for later restoration.

        Both context dicts are deep-copied. `Workflow._to_dict()` returns
        `self._context` **by reference** (`workflow.py:450`), so without this
        the "snapshot" aliases the live dict: the teacher's own mutations reach
        it, restoring it is a no-op, and `restore_ok_pre_student` — a
        fingerprint compared against a state that was never rolled back —
        reports agreement by construction. This prerequisite belongs to
        fix-sb8.3 rather than to fix-35m.3 because it is a measurement bug, and
        the deep copy and the fingerprint are worth nothing apart (§5).
        """
        workflow = self.chat_session.get_active_workflow()
        cme = self.chat_session.cme_workflow
        workflow_dict = dict(workflow._to_dict())
        workflow_dict["workflow_context"] = _deepcopy_context(
            workflow_dict.get("workflow_context")
        )
        cme_dict = dict(cme._to_dict())
        cme_dict["workflow_context"] = _deepcopy_context(
            cme_dict.get("workflow_context")
        )
        return {
            "workflow_dict": workflow_dict,
            "cme_dict": cme_dict,
            "conversation_history_len": len(
                self.chat_session.conversation_history.messages
            ),
        }

    def restore_workflow_state(self, snapshot: dict):
        """Restore workflow + CME state from a prior snapshot."""
        workflow = self.chat_session.get_active_workflow()
        wd = snapshot["workflow_dict"]
        workflow._context = wd["workflow_context"]
        workflow._is_complete = wd["is_complete"]
        workflow._save()
        workflow._dirty = False

        cme = self.chat_session.cme_workflow
        cd = snapshot["cme_dict"]
        cme._context = cd["workflow_context"]
        cme._save()
        cme._dirty = False

        orig_len = snapshot["conversation_history_len"]
        # Create a new History instance instead of modifying frozen messages
        import dspy
        self.chat_session._conversation_history = dspy.History(
            messages=self.chat_session._conversation_history.messages[:orig_len]
        )

        # Clear the in-process action log so the next pass starts from a clean
        # slate. In-place clear on the WEC's live list — callers holding action
        # snapshots (list(...)) are unaffected (ruling I8).
        self.chat_session.clear_action_log()

    # ------------------------------------------------------------------
    # Comparability evidence (§6, fix-sb8.3)
    # ------------------------------------------------------------------

    def _restore_matches(self, baseline: Optional[str]) -> int:
        """1 iff the post-restore `state_fingerprint` equals *baseline* (§6.2).

        The two restore_ok columns share this check and nothing else: they are
        different assertions against different baselines —
        `restore_ok_pre_student` against the PRE-teacher entry state, which
        `restore_workflow_state(initial_snapshot)` restores toward, and
        `restore_ok_post_compare` against the teacher's EXIT state, which the
        divergence-path restores restore toward. Revision 1's single column
        used the pre-teacher baseline for all three call sites and would have
        reported 0 on every divergent run.
        """
        if not baseline:
            return 0
        current = _safe_state_fingerprint(self.chat_session)
        return int(current is not None and current == baseline)

    def pass_fingerprint(self, pass_label: str, which: str) -> Optional[str]:
        """One recorded pass fingerprint ('entry_fingerprint'/'exit_fingerprint')."""
        return self._passes.get(pass_label, {}).get(which)

    def comparability_fields(
        self, *, forced_reason: Optional[str] = None
    ) -> dict[str, Any]:
        """The run row's comparability verdict ([DR15], [DR47]).

        `comparable = 1` iff every pass entered on an equal `state_fingerprint`.
        What that attests is bounded and published (§6.2): the **context dicts,
        command context and completion flag** were equal at pass entry. It does
        not attest that the application world was equal and it cannot —
        `Workflow._to_dict()` never captures `_current_command_context` or the
        application's own objects — which is why the UI label is "comparable
        inputs; application state not verified" and never a bare "comparable".

        Missing evidence is never agreement: an absent fingerprint, a pass that
        never ran, a barrier that could not be taken, or a writer that dropped
        a span or a distillation record anywhere in the run all yield
        `evidence-incomplete`. A run whose divergence rows cite spans the
        writer discarded must never be recorded `comparable = 1`: every §15
        recipe filters on that column, so it would be first-class evidence.
        """
        ordered = sorted(self._passes.items(), key=lambda kv: kv[1].get("seq", 0))
        fields: dict[str, Any] = {
            # Denormalized onto the run row so the loud banner and the
            # list-level warning need no join (§6.2).
            "fingerprint_teacher": self.pass_fingerprint("teacher", "entry_fingerprint"),
            "fingerprint_student": self.pass_fingerprint("student", "entry_fingerprint"),
        }
        if forced_reason:
            return {**fields, "comparable": 0, "comparable_reason": forced_reason}
        entries = [row.get("entry_fingerprint") for _label, row in ordered]
        if len(entries) < 2 or any(fp is None for fp in entries):
            reason = "evidence-incomplete"
        elif (
            self._barrier_failed
            or self._writer_loss
            or any(row.get("spans_dropped_delta") for _label, row in ordered)
        ):
            # [DR49]: a span that never landed is missing evidence, and a
            # merely late one reads downstream as a divergence rather than as
            # silence — which is worse than the silence [DR40] already forbids.
            # A flush barrier that timed out leaves the same hole: the rows
            # the divergence records cite may not be there to be read.
            reason = "evidence-incomplete"
        elif len(set(entries)) > 1:
            reason = "fingerprint-differs"
        else:
            reason = None
        fields["comparable"] = int(reason is None)
        fields["comparable_reason"] = reason
        return fields

    def finalize_pass_metrics(self) -> Optional[int]:
        """Roll per-pass tokens/cost/cache up from `fw.llm.call` spans (§6.3).

        Computed once, at run completion, and written onto `distillation_passes`
        so the UI and the aggregates never rescan spans. Returns
        `cache_asymmetric` ([DR16]) — a hit in one pass and a miss in the other
        — or None when no rollup could be taken. Asymmetry does **not** force
        `comparable = 0`: a cache hit returns the same completion, so the
        trajectory is still comparable; it is a *cost* confound, and it is the
        cost columns that must not be compared across it.

        Every known pass row is re-emitted whether or not the rollup produced
        anything (`fix-sb8.16`). The re-emission is an idempotent upsert on
        `(run_id, pass_label)`, so its value is a SECOND chance for a row the
        record queue dropped at pass exit — and an empty rollup (no
        `fw.llm.call` spans, a barrier that could not be taken, a read that
        failed) is exactly the situation in which the first write is most
        likely to have been the one that was lost. Returning early on it left a
        run row with no pass rows: a run with no models, no fingerprints and no
        entry inputs, which every §15 recipe joins through.
        """
        rollup = self._llm_rollup()
        for pass_label in list(self._pass_rows):
            self._emit_pass_record(pass_label, **rollup.get(pass_label, {}))
        if not rollup:
            return None
        hits = [
            totals["cache_hits"]
            for pass_label, totals in rollup.items()
            if pass_label in self._pass_rows
        ]
        if len(hits) < 2:
            return 0
        return int(any(h > 0 for h in hits) and any(h == 0 for h in hits))

    def _llm_rollup(self) -> dict[str, dict[str, Any]]:
        """Execute `_LLM_ROLLUP_SQL` behind the sink's flush barrier.

        A read, not a write: [DR46] routes every distillation *write* through
        the sink's writer thread, and this stays on that side of the line by
        never opening a transaction. Any failure returns an empty rollup, which
        leaves the cost columns NULL rather than failing `_execute_message`.
        """
        chat = self.chat_session
        turn_key = chat.current_turn_key
        sink = tracing.get_sink(chat)
        store = getattr(sink, "store", None)
        if not turn_key or store is None:
            return {}
        try:
            flush = getattr(sink, "flush", None)
            if flush is not None and not _flush_trace(
                flush, turn_key, _ROLLUP_FLUSH_TIMEOUT_S
            ):
                logger.warning(
                    "Distillation: span flush barrier timed out; "
                    "per-pass cost rollup skipped"
                )
                return {}
            conn = store._connect(timeout=_ROLLUP_READ_TIMEOUT_S)
            try:
                rows = conn.execute(
                    _LLM_ROLLUP_SQL, {"turn_key": turn_key}
                ).fetchall()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"Distillation: per-pass cost rollup failed: {exc!r}")
            return {}
        return {
            row["pass_label"]: {
                "cache_hits": int(row["cache_hits"] or 0),
                "cache_misses": int(row["cache_misses"] or 0),
                "cost_usd": float(row["cost_usd"] or 0.0),
                "tokens": int(row["tokens"] or 0),
            }
            for row in rows
            if row["pass_label"]
        }

    # ------------------------------------------------------------------
    # Retention pin (§10.3, [DR43])
    # ------------------------------------------------------------------

    def pin_fields(self, completion: dict[str, Any]) -> dict[str, Any]:
        """§10.3's pin-class table, evaluated at write time.

        "Implemented as `pinned = 1` at write time with the pin cleared by a
        bounded sweep in `prune()`" — the sweep, the prune predicate, the
        ceiling and the shortfall API are all built; this is the writer they
        were missing, and without it `prune()` (which runs opportunistically at
        every sink startup, on a 30-day default) deletes an insight's cited
        divergence rows, its citations and both passes' spans at the horizon.

        The classes, in the order they are resolved:

        * **Non-comparable — no pin.** Its divergences are unusable by
          contract (§6.2), so retaining the trace retains a confound rather
          than evidence. This row of the table is an exclusion and is therefore
          resolved first; §6.2 obligation 5 keeps the run and its insights as
          rows, which is what "recorded but quarantined" means.
        * **Produced an insight — pinned.** Every insight is unadjudicated
          when it is written, and pruning would pre-empt adjudication. A
          `supported` verdict keeps it pinned; a rejected-only run is unpinned
          later by `prune()`'s sweep, which derives it from the verdicts (§12
          rule 1 forbids the verdict route from writing this column).
        * **No divergence at all — pinned for 90 days.** The contradiction set
          for every future rule (§15). `_release_distillation_pins` selects
          exactly `pinned=1 AND planning_diverged=0 AND exec_diverged=0` and
          orders on `COALESCE(pinned_at, started_at)`, both of which this write
          supplies.
        * A run that diverged and produced no insight is in none of the five
          classes, and is left unpinned: nothing cites its evidence and it is
          not the contradiction pool.
        """
        if not completion.get("comparable"):
            return {}
        produced_insight = (
            self.planning_insights_extracted + self.execution_insights_extracted
        ) > 0
        diverged = bool(
            completion.get("planning_diverged") or completion.get("exec_diverged")
        )
        no_divergence = not diverged and bool(completion.get("completed_at"))
        if not (produced_insight or no_divergence):
            return {}
        # [DR43]: the count is what a later shortfall is measured against, so
        # a loss caused by a build without the pin predicate is detected rather
        # than discovered. It is a SENTINEL, not a value: the producer used to
        # read it with its own sqlite3 connection, which put a synchronous DB
        # read on the turn thread at the completion of every pinned run, on top
        # of the barrier waits (`fix-sb8.18`). The writer thread resolves it
        # inside the transaction that writes this row, where the connection is
        # already open and the pass rows are already applied.
        return {
            "pinned": 1,
            "pinned_at": _utc_now(),
            "pinned_span_count": COUNT_LIVE_SPANS,
        }

    def _record_lm_params(self, which: str, lm: Any) -> None:
        """Capture a pass's resolved sampling parameters where get_lm made them."""
        try:
            kwargs = getattr(lm, "kwargs", None)
            self._lm_params[which] = {
                "model": getattr(lm, "model", None),
                "kwargs": _canonical(dict(kwargs)) if kwargs else None,
            }
        except Exception:
            # A diagnostic column must never cost a pass its run.
            self._lm_params[which] = None

    def _entry_inputs_json(self, message: str, history_bound: int) -> Optional[str]:
        """`entry_inputs_json` — PROMPT INPUTS, not restorable state ([DR45]).

        The context dicts ride along under an explicit `diagnostic_only` flag:
        they are here to *explain* a divergence to a reader, never to be loaded
        back into a `Workflow`. `_canonical` renders numbers as decimal strings
        and bounds the history tail, and `[R20]` redaction rewrites anything
        matching a loaded secret — a column that read as restorable state and
        was not would be worse than no column at all.
        """
        chat = self.chat_session
        try:
            workflow = chat.get_active_workflow()
            cme = chat.cme_workflow
            msgs = chat.conversation_history.messages[:history_bound]
            payload: dict[str, Any] = {
                "v": _ENTRY_INPUTS_VERSION,
                "raw_user_message": message,
                "refined_user_message": None,
                "plan": None,
                "history_bound": history_bound,
                "history_tail": [
                    _canonical(msg) for msg in msgs[-_HISTORY_TAIL_MESSAGES:]
                ],
                "insight_set": self.insight_set(),
                "context_snapshot": {
                    "diagnostic_only": True,
                    "workflow_context": (
                        _canonical(workflow._context) if workflow else None
                    ),
                    "cme_context": _canonical(cme._context) if cme else None,
                },
            }
            return json.dumps(payload, separators=(",", ":"))
        except Exception as exc:
            logger.warning(f"Distillation: entry inputs capture failed: {exc!r}")
            return None

    def insight_set(self) -> dict[str, Any]:
        """The insight corpora as loaded at agent init, by size and hash.

        `_planning_insights` / `_execution_insights` are loaded once at agent
        init and never reloaded, so the corpus a run actually used is not the
        file's current contents ([DR34]). The bodies are not stored — §8 makes
        the same call for the extractor span, and the corpus is reconstructible
        from the ledger.
        """
        chat = self.chat_session
        return {
            "planning": _text_digest(getattr(chat, "_planning_insights", None)),
            "execution": _text_digest(getattr(chat, "_execution_insights", None)),
        }

    @staticmethod
    def _with_pass_prompt_inputs(
        entry_inputs_json: Optional[str], planning_steps: list
    ) -> Optional[str]:
        """Fold the refined query and the plan into a pass's entry inputs.

        Both are produced *inside* the pass — `_refine_user_query` and then
        `build_query_with_next_steps` — so they cannot be captured at the entry
        boundary, but they are prompt inputs of that pass and [DR45] names them.
        """
        if not entry_inputs_json or not planning_steps:
            return entry_inputs_json
        try:
            payload = json.loads(entry_inputs_json)
            first = planning_steps[0]
            payload["refined_user_message"] = getattr(first, "user_query", None)
            payload["plan"] = list(getattr(first, "generated_plan", None) or [])
            rendered = json.dumps(payload, separators=(",", ":"))
        except Exception:
            return entry_inputs_json
        if len(rendered.encode()) > _ENTRY_INPUTS_MAX_BYTES:
            payload["context_snapshot"] = {
                "diagnostic_only": True,
                "omitted": "over-cap",
            }
            rendered = json.dumps(payload, separators=(",", ":"))
        return rendered

    # ------------------------------------------------------------------
    # Divergence alignment and recording (§7, fix-sb8.4)
    # ------------------------------------------------------------------

    def install_span_collector(self) -> bool:
        """Start holding this run's step spans as the sink emits them ([DR49]).

        The alignment must never read `spans` while the writer thread is still
        draining: a merely late span would come out as a fabricated
        `missing-in-student` divergence. Holding the emitted `Span` objects is
        what makes that race structurally impossible.
        """
        sink = tracing.get_sink(self.chat_session)
        if sink is None:
            return False
        collector = _PassSpanCollector(sink)
        try:
            if not collector.install():
                return False
        except Exception as exc:
            logger.warning(f"Distillation: span collector not installed: {exc!r}")
            return False
        self._span_collector = collector
        return True

    def remove_span_collector(self) -> None:
        """Leave the sink as it was found, on every exit path.

        The collected spans stay readable afterwards — the collector is the
        capture mechanism, not the store.
        """
        collector = self._span_collector
        if collector is None:
            return
        try:
            collector.remove()
        except Exception as exc:
            logger.warning(f"Distillation: span collector not removed: {exc!r}")

    def alignment_available(self) -> bool:
        """True when spans were collected, so the records are the source (§7.6).

        With observability off there are no spans to align and no table to
        write them into; such a run falls back to the legacy action-log
        comparison below. §7.6 replaces the prose summary's *source*, it does
        not make distillation require a sink.
        """
        return self._span_collector is not None

    def _pass_spans(self, pass_label: str, names) -> list:
        collector = self._span_collector
        return collector.spans_of(pass_label, names) if collector is not None else []

    def action_steps(self, pass_label: str) -> list:
        """§7.1's comparable-unit sequence for one pass, in `start_ns` order.

        Every `fw.command.execute` span plus every `fw.ask_user` span the pass
        emitted — failed commands included, which is the whole point of
        aligning over spans ([DR50] keys them off `raw_command`).
        """
        steps = []
        for span in self._pass_spans(pass_label, _ACTION_SPAN_NAMES):
            attributes = span.attributes or {}
            if span.name == tracing.SPAN_ASK_USER:
                steps.append(
                    alignment.make_ask_user_step(
                        span.span_id,
                        _attr_text(attributes.get("agent_query")) or "",
                        context=span.context,
                        start_ns=span.start_ns,
                    )
                )
                continue
            parameters = attributes.get("parameters")
            steps.append(
                alignment.make_command_step(
                    span.span_id,
                    command_name=span.command_name or None,
                    context=span.context,
                    parameters=parameters if isinstance(parameters, dict) else None,
                    raw_command=_attr_text(attributes.get("raw_command")),
                    start_ns=span.start_ns,
                )
            )
        return steps

    def plan_steps(self, pass_label: str) -> list:
        """§7.1's plan level: one step per planner span, the FULL plan string.

        Not `PlanningStep.generated_plan`, which is `next_steps.split()` — a
        whitespace split into individual *words* (`workflow_agent.py:686`) —
        so aligning over it produces word-level noise.
        """
        return [
            alignment.make_plan_step(
                span.span_id,
                _attr_text((span.attributes or {}).get("plan")) or "",
                context=span.context,
                start_ns=span.start_ns,
            )
            for span in self._pass_spans(pass_label, _PLAN_SPAN_NAMES)
        ]

    def agent_span_id(self, pass_label: str) -> Optional[str]:
        """The pass's `fw.agent.execute` span id — the run record's pair (§7.3)."""
        spans = self._pass_spans(pass_label, (tracing.SPAN_AGENT_EXECUTE,))
        return spans[0].span_id if spans else None

    def read_barrier(self) -> bool:
        """[DR49]'s barrier: block until this turn's spans are written.

        Taken before ANY divergence row is written, so a stored record can
        never cite a span id the table does not hold yet. A barrier that
        cannot be satisfied is missing evidence, not agreement: it sets
        `comparable = 0` / `evidence-incomplete`, the same verdict a moved
        `spans_dropped` counter earns.

        Scoped to this turn's trace (`fix-sb8.15`). The sink is shared by
        every channel on the workflow DB, so an unscoped barrier waits on
        other channels' backlogs and can be starved into a spurious
        `evidence-incomplete` — a loss report that also costs the run its
        §10.3 pin. The rows this run is about to cite all live under
        `current_turn_key`, so that is the only trace it needs settled.
        """
        sink = tracing.get_sink(self.chat_session)
        flush = getattr(sink, "flush", None) if sink is not None else None
        if flush is None:
            # No sink, or one predating flush(): nothing was persisted, so
            # nothing can be cited and there is no race to lose.
            return True
        try:
            satisfied = bool(
                _flush_trace(
                    flush,
                    self.chat_session.current_turn_key,
                    _ALIGN_FLUSH_TIMEOUT_S,
                )
            )
        except Exception as exc:
            logger.warning(f"Distillation: span flush barrier failed: {exc!r}")
            satisfied = False
        if not satisfied:
            logger.warning(
                "Distillation: span flush barrier not satisfied; divergence "
                "evidence is incomplete ([DR49])"
            )
            self._barrier_failed = True
        return satisfied

    def snapshot_writer_counters(self) -> None:
        """Record the writer's loss counters at run open ([DR49], §19)."""
        self._writer_counters_entry = _writer_counters(self.chat_session)

    def check_writer_loss(self) -> dict[str, int]:
        """Compare the loss counters against run open; remember any movement.

        Called AFTER the closing read barrier, which is the whole point: the
        per-pass `spans_dropped_delta` is sampled at the pass's own exit, so a
        batch the writer discards while the barrier drains — the `[R8]`
        multi-process `SQLITE_BUSY` rollback, which drops every span of the
        batch outright — moves no per-pass counter and would otherwise leave
        the run recorded `comparable = 1` while the spans its divergence rows
        cite are gone. `records_dropped` and `write_errors` cover the same hole
        on the record queue, which nothing consulted at all.

        Returns the counters that moved, and folds them into
        `comparability_fields` as `evidence-incomplete`.
        """
        entry = self._writer_counters_entry
        current = _writer_counters(self.chat_session)
        if entry is None or current is None:
            return {}
        moved = {
            key: current[key] - entry[key]
            for key in _WRITER_COUNTER_KEYS
            if current.get(key, 0) > entry.get(key, 0)
        }
        if moved:
            self._writer_loss = moved
            logger.warning(
                "Distillation: the writer lost evidence during this run "
                f"({moved}); the run is recorded comparable=0 / "
                "evidence-incomplete ([DR49])"
            )
        return moved

    def align_and_record(
        self,
        *,
        level: str,
        left_pass: str,
        right_pass: str,
        left_steps: list,
        right_steps: list,
        comparable: bool,
        left_answer: Optional[str] = None,
        right_answer: Optional[str] = None,
    ) -> alignment.AlignmentResult:
        """Align one level of two passes and persist every record (§7, [DR46]).

        `identical` records are stored too: they are what makes the aligned
        diff renderable without recomputation, and they are the denominator of
        every rate in fix-sb8.10 (§7.3). Materiality reads the two exit
        `state_fingerprint`s and the run's `comparable` flag and is NULL when
        the run is non-comparable ([DR20]) — the aligner never computes it.
        """
        chat = self.chat_session
        span = tracing.start_span(
            chat,
            tracing.SPAN_DISTILL_COMPARE,
            attributes={
                "run_id": self.run_id,
                "level": level,
                "left_pass": left_pass,
                "right_pass": right_pass,
            },
        )
        try:
            result = alignment.align_passes(
                run_id=self.run_id,
                left_pass=left_pass,
                right_pass=right_pass,
                left_steps=left_steps,
                right_steps=right_steps,
                comparable=comparable,
                left_exit_state_fingerprint=self.pass_fingerprint(
                    left_pass, "exit_fingerprint"
                ),
                right_exit_state_fingerprint=self.pass_fingerprint(
                    right_pass, "exit_fingerprint"
                ),
                level=level,
                left_answer=left_answer,
                right_answer=right_answer,
                left_run_span_id=self.agent_span_id(left_pass),
                right_run_span_id=self.agent_span_id(right_pass),
            )
        except BaseException as exc:
            tracing.end_span(
                chat,
                span,
                status=tracing.STATUS_ERROR,
                attributes={"error_type": type(exc).__name__},
            )
            raise
        for record in result.records:
            tracing.emit_distillation_record(chat, "divergence", record.to_row())
        # The citations this level's insights will carry (§13.2). Exactly the
        # records `render_divergence_summary` puts in front of the extractor —
        # `identical` records are stored but are not described to it, so they
        # are not evidence any insight was drawn from.
        insight_kind = _INSIGHT_KIND_BY_LEVEL.get(level)
        if insight_kind is not None:
            self._cited_divergences[insight_kind] = tuple(
                record.divergence_id
                for record in result.records
                if record.kind != alignment.KIND_IDENTICAL
            )
        tracing.end_span(
            chat,
            span,
            # Spelled out rather than `result.compare_attributes(...)` so the
            # §12.0-delta-5 contract test can recover this emitter's key set by
            # AST: its resolver follows helpers defined in the emitting module
            # or in `tracing`, and a method on an imported dataclass is neither.
            attributes={
                "level": level,
                "left_pass": left_pass,
                "right_pass": right_pass,
                "left_steps": result.left_steps,
                "right_steps": result.right_steps,
                "matched_pairs": result.matched_pairs,
                "divergence_counts": dict(result.divergence_counts),
                "material_count": result.material_count,
                "algorithm": result.algorithm,
            },
        )
        return result

    @staticmethod
    def divergence_verdict(
        result: alignment.AlignmentResult,
    ) -> tuple[bool, str]:
        """(diverged, prose summary) for one alignment, both off the RECORDS.

        §7.6: the extractor prompt, the UI and the aggregate queries read one
        source of truth, so the prose is *rendered from* the stored records
        rather than computed beside them.

        `different-answer-same-actions` is stored but does not set the
        `exec_diverged` column: that column has always meant "the two passes
        executed different actions", and teacher and student are different
        models whose wording differs on nearly every run. Letting it flip the
        column would change how often extraction fires, which this epic's
        "record and review, not re-tune" scope forbids.
        """
        diverged = any(
            record.level != alignment.LEVEL_RUN
            and record.kind != alignment.KIND_IDENTICAL
            for record in result.records
        )
        return diverged, alignment.render_divergence_summary(result)

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_exec_steps(trajectory: dict) -> list[dict]:
        """
        Extract execute_workflow_query steps from a ReAct trajectory.

        Returns a list of dicts with keys: step_idx, thought, tool_name, tool_args, observation.
        Only includes steps where tool_name == "execute_workflow_query".
        """
        steps = []
        idx = 0
        while True:
            tool_name_key = f"tool_name_{idx}"
            if tool_name_key not in trajectory:
                break
            tool_name = trajectory[tool_name_key]
            if tool_name == "execute_workflow_query":
                steps.append({
                    "step_idx": idx,
                    "thought": trajectory.get(f"thought_{idx}", ""),
                    "tool_name": tool_name,
                    "tool_args": trajectory.get(f"tool_args_{idx}", {}),
                    "observation": trajectory.get(f"observation_{idx}", ""),
                })
            idx += 1
        return steps

    @staticmethod
    def _format_trajectory_for_llm(trajectory: dict) -> str:
        """Format a ReAct trajectory dict into a readable string for the insight LLM."""
        lines = []
        idx = 0
        while True:
            thought_key = f"thought_{idx}"
            if thought_key not in trajectory and f"tool_name_{idx}" not in trajectory:
                break
            if thought_key in trajectory:
                lines.append(f"[Step {idx}] Thought: {trajectory[thought_key]}")
            if f"tool_name_{idx}" in trajectory:
                lines.append(f"[Step {idx}] Tool: {trajectory[f'tool_name_{idx}']}")
            if f"tool_args_{idx}" in trajectory:
                args = trajectory[f"tool_args_{idx}"]
                lines.append(f"[Step {idx}] Args: {json.dumps(args, default=str)}")
            if f"observation_{idx}" in trajectory:
                obs = str(trajectory[f"observation_{idx}"])
                # Truncate very long observations
                if len(obs) > 500:
                    obs = f"{obs[:500]}... [truncated]"
                lines.append(f"[Step {idx}] Observation: {obs}")
            lines.append("")
            idx += 1

        # Include conversation history context if present
        if "observation_about_conversation_history" in trajectory:
            lines.insert(0, f"[Context] Conversation history: {trajectory['observation_about_conversation_history']}\n")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trajectory comparison off the in-process action log — the NO-SINK path
    #
    # [DR17] moved the live comparison onto spans, and §7.6 made the prose a
    # rendering of the stored records. These two comparators remain the
    # fallback for a run with no observability sink: there are no spans to
    # align and no `distillation_divergences` table to write, and
    # distillation must keep detecting divergence exactly as it did. Their
    # set semantics are why they are not the source of truth when spans
    # exist — a set loses repeats, cannot express "same command, one wrong
    # parameter", and cannot say whether the difference changed the end state
    # or only the path.
    # ------------------------------------------------------------------

    @staticmethod
    def _action_signature(action: dict) -> tuple[str, str]:
        """Return (command_name, sorted-params-json) as a comparable unit."""
        cmd = action.get("command_name", "")
        params = action.get("parameters", {})
        return (cmd, json.dumps(params, sort_keys=True, default=str))

    @staticmethod
    def _format_action(action: dict) -> str:
        """Human-readable representation of an action (command + params)."""
        cmd = action.get("command_name", "")
        params = action.get("parameters", {})
        return f"{cmd}({json.dumps(params, default=str)})" if params else cmd

    def compare_trajectories(
        self,
        teacher_actions: list[dict],
        student_actions: list[dict],
    ) -> tuple[bool, str]:
        """
        Compare teacher and student action lists (per-pass action-log snapshots).

        The no-sink fallback for `align_and_record` (§7.6). Each action is
        treated as a (command_name, parameters) unit.
        Instead of strict step-by-step ordering, this produces a human-readable
        summary of all differences and lets the insight extraction LLM judge
        whether those differences constitute genuine mistakes or are justified
        by multi-turn conversation context divergence.

        Returns:
            (has_divergence: bool, divergence_summary: str)
            divergence_summary is empty when has_divergence is False.
        """
        # Filter out internal error correction actions (abort commands from loop detection)
        def is_valid_action(a: dict) -> bool:
            cmd = a.get("command_name", "")
            # Exclude ErrorCorrection/abort and ask_user records (which have agent_query key)
            return cmd and not cmd.startswith("ErrorCorrection/") and "agent_query" not in a

        teacher_actions = [a for a in teacher_actions if is_valid_action(a)]
        student_actions = [a for a in student_actions if is_valid_action(a)]

        teacher_sigs = [self._action_signature(a) for a in teacher_actions]
        student_sigs = [self._action_signature(a) for a in student_actions]

        # Fast path: identical sequences (same actions, same order)
        if teacher_sigs == student_sigs:
            return False, ""

        differences: list[str] = []

        # Compare actions as (command_name, params) units
        teacher_sig_set = set(teacher_sigs)
        student_sig_set = set(student_sigs)

        only_teacher = teacher_sig_set - student_sig_set
        only_student = student_sig_set - teacher_sig_set

        if only_teacher:
            only_teacher_strs = [
                self._format_action(a)
                for a in teacher_actions
                if self._action_signature(a) in only_teacher
            ]
            differences.append(
                f"Actions executed only by teacher: {only_teacher_strs}"
            )

        if only_student:
            only_student_strs = [
                self._format_action(a)
                for a in student_actions
                if self._action_signature(a) in only_student
            ]
            differences.append(
                f"Actions executed only by student: {only_student_strs}"
            )

        # Check ordering of shared actions (only if there are actual shared actions)
        shared_teacher_sigs = [s for s in teacher_sigs if s in student_sig_set]
        shared_student_sigs = [s for s in student_sigs if s in teacher_sig_set]

        # Only report ordering differences if there are meaningful shared actions
        if shared_teacher_sigs and shared_student_sigs and shared_teacher_sigs != shared_student_sigs:
            teacher_order = [
                self._format_action(a)
                for a in teacher_actions
                if self._action_signature(a) in student_sig_set
            ]
            student_order = [
                self._format_action(a)
                for a in student_actions
                if self._action_signature(a) in teacher_sig_set
            ]
            # Only add if we actually have formatted actions (not empty strings)
            if teacher_order and student_order and any(teacher_order) and any(student_order):
                differences.append(
                    f"Different execution order — "
                    f"teacher: {teacher_order}, student: {student_order}"
                )

        if not differences:
            # Sequences differ in some way we didn't classify—still flag it
            differences.append(
                "Action sequences differ (unclassified difference)"
            )

        return True, "\n".join(differences)

    # ------------------------------------------------------------------
    # Planning trace helpers
    # ------------------------------------------------------------------

    def compare_planning_traces(
        self,
        teacher_steps: list[PlanningStep],
        student_steps: list[PlanningStep],
    ) -> tuple[bool, str]:
        """
        Compare teacher and student planning traces.

        The no-sink fallback for the plan-level alignment (§7.1), which reads
        the whitespace-normalized full plan string off the planner spans
        rather than `PlanningStep.generated_plan`'s word split.

        Returns:
            (has_divergence: bool, divergence_summary: str)
        """
        if not teacher_steps and not student_steps:
            return False, ""

        differences = []

        # Compare plans at each step
        for i in range(max(len(teacher_steps), len(student_steps))):
            t_plan = teacher_steps[i].generated_plan if i < len(teacher_steps) else []
            s_plan = student_steps[i].generated_plan if i < len(student_steps) else []

            # Normalize for comparison
            t_normalized = [p.lower().strip() for p in t_plan]
            s_normalized = [p.lower().strip() for p in s_plan]

            if t_normalized != s_normalized:
                differences.append(
                    f"Step {i}: Teacher planned {t_plan}, Student planned {s_plan}"
                )

        return (True, "\n".join(differences)) if differences else (False, "")

    @staticmethod
    def _format_planning_traces_for_llm(steps: list[PlanningStep]) -> str:
        """Format planning steps into readable string for insight LLM."""
        lines = []
        for step in steps:
            lines.append(f"[Step {step.step_number}] Query: {step.user_query}")
            if step.reasoning:
                lines.append(f"[Step {step.step_number}] Reasoning: {step.reasoning}")
            lines.extend((f"[Step {step.step_number}] Plan: {step.generated_plan}", ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent run helper
    # ------------------------------------------------------------------

    def _run_agent_pass(
        self,
        message: str,
        agent_lm_role: str,
        agent_api_key_role: str,
        planner_lm_role: str,
        planner_api_key_role: str,
        pass_label: Optional[str] = None,
        role: Optional[str] = None,
        seq: int = 0,
    ) -> tuple[fastworkflow.CommandOutput, dict, list[dict], list[PlanningStep]]:
        """
        Run a full agent pass with the specified LLMs for planner and agent.

        With `pass_label` set (every pass of a real run), the pass body runs
        inside its own `fw.distill.pass` span and with the label ambient, so
        every span the pass emits is both parented under that span and stamped
        with `spans.distillation_pass` — the two halves of §8's separation,
        one structural and one indexed. It also writes the pass's
        `distillation_passes` row, on the raising path as well as the clean
        one. `pass_label=None` runs the pass exactly as before, with no
        distillation structure at all.

        Returns:
            (command_output, trajectory_dict, actions, planning_steps)
            actions: snapshot (copy) of the WEC's in-process action log for this
                pass — dicts with keys command, command_name, parameters, response
            planning_steps: list of PlanningStep objects capturing planning decisions
        """
        if pass_label is None:
            return self._run_agent_pass_body(
                message,
                agent_lm_role,
                agent_api_key_role,
                planner_lm_role,
                planner_api_key_role,
            )

        chat = self.chat_session
        # The replay trace where one is bound ([DR41]): the pass span's
        # deterministic id and the pass row both key off the trace the spans
        # are actually going into, not off a turn key a replay does not have.
        turn_key = self.trace_key
        agent_model = fastworkflow.get_env_var(agent_lm_role)
        planner_model = fastworkflow.get_env_var(planner_lm_role)

        # The pass ENTRY boundary — the one call per boundary [DR13] allows,
        # with two consumers (this row, and fix-35m.3's teacher post-condition).
        # `history_bound` is captured here and reused at exit, so the summary
        # this pass appends to conversation history from inside itself is never
        # inside its own exit prompt hash ([DR47]).
        history_bound = len(chat.conversation_history.messages)
        entry_fingerprint = _safe_state_fingerprint(chat)
        entry_prompt_fingerprint = _safe_prompt_fingerprint(chat, history_bound)
        spans_dropped_entry = _spans_dropped(chat)
        entry_inputs_json = self._entry_inputs_json(message, history_bound)
        evidence = self._passes.setdefault(pass_label, {})
        evidence.update(
            role=role or pass_label,
            seq=seq,
            history_bound=history_bound,
            entry_fingerprint=entry_fingerprint,
            entry_prompt_fingerprint=entry_prompt_fingerprint,
        )
        self._lm_params = {}

        # The label goes ambient BEFORE the span opens: fw.distill.pass carries
        # its own label ([DR7]), unlike the run-level wrappers.
        with chat.distillation_pass_scope(pass_label):
            span = tracing.start_span(
                chat,
                tracing.SPAN_DISTILL_PASS,
                # Deterministic [DR51] — the ask_user close can only parent
                # onto an id it can recompute from the turn key and the label.
                span_id=(
                    tracing.distill_pass_span_id(turn_key, pass_label)
                    if turn_key
                    else None
                ),
                attributes={
                    "run_id": self.run_id,
                    "pass_label": pass_label,
                    "role": role or pass_label,
                    "seq": seq,
                    "agent_model": agent_model,
                    "planner_model": planner_model,
                },
            )
            started_ns = time.time_ns()
            status = tracing.STATUS_OK
            error_type = None
            outcome = None
            try:
                outcome = self._run_agent_pass_body(
                    message,
                    agent_lm_role,
                    agent_api_key_role,
                    planner_lm_role,
                    planner_api_key_role,
                )
                return outcome
            except BaseException as exc:
                status = tracing.STATUS_ERROR
                error_type = type(exc).__name__
                raise
            finally:
                wall_ms = (time.time_ns() - started_ns) // 1_000_000
                # The pass EXIT boundary. Taken at the same `history_bound` as
                # the entry, and history-free for the state projection, which
                # is what keeps §7.4's "same end state, different path" branch
                # from being dead code.
                exit_fingerprint = _safe_state_fingerprint(chat)
                exit_prompt_fingerprint = _safe_prompt_fingerprint(
                    chat, history_bound
                )
                spans_dropped_delta = (
                    None
                    if spans_dropped_entry is None
                    else (_spans_dropped(chat) or 0) - spans_dropped_entry
                )
                if outcome is not None:
                    entry_inputs_json = self._with_pass_prompt_inputs(
                        entry_inputs_json, outcome[3]
                    )
                evidence.update(
                    exit_fingerprint=exit_fingerprint,
                    exit_prompt_fingerprint=exit_prompt_fingerprint,
                    spans_dropped_delta=spans_dropped_delta,
                )
                tracing.end_span(
                    chat,
                    span,
                    status=status,
                    attributes={
                        "wall_ms": wall_ms,
                        "error_type": error_type,
                        "entry_fingerprint": entry_fingerprint,
                        "exit_fingerprint": exit_fingerprint,
                    },
                )
                model_params = self._lm_params or None
                self._emit_pass_record(
                    pass_label,
                    role=role or pass_label,
                    seq=seq,
                    agent_model=agent_model,
                    planner_model=planner_model,
                    model_params_json=(
                        json.dumps(model_params, separators=(",", ":"))
                        if model_params
                        else None
                    ),
                    wall_ms=wall_ms,
                    # The pass wrapper is by construction the first span of the
                    # pass and the anchor its whole subtree hangs off.
                    first_span_id=span.span_id if span is not None else None,
                    entry_fingerprint=entry_fingerprint,
                    exit_fingerprint=exit_fingerprint,
                    entry_prompt_fingerprint=entry_prompt_fingerprint,
                    exit_prompt_fingerprint=exit_prompt_fingerprint,
                    history_bound=history_bound,
                    spans_dropped_delta=spans_dropped_delta,
                    entry_inputs_json=entry_inputs_json,
                )

    def _run_agent_pass_body(
        self,
        message: str,
        agent_lm_role: str,
        agent_api_key_role: str,
        planner_lm_role: str,
        planner_api_key_role: str,
    ) -> tuple[fastworkflow.CommandOutput, dict, list[dict], list[PlanningStep]]:
        """One agent pass, with no distillation structure around it."""
        # Clean prior action log so this pass's records stand alone.
        self.chat_session.clear_action_log()

        # Store raw message in workflow context (mirrors _process_agent_message)
        self.chat_session.get_active_workflow().context["raw_user_message"] = message

        # Create agent with the specified LLM
        from fastworkflow.workflow_agent import (
            initialize_workflow_tool_agent,
            build_query_with_next_steps,
            _what_can_i_do,
        )

        # Load execution insights for the agent
        execution_insights = getattr(
            self.chat_session, "_execution_insights", None
        )

        agent = initialize_workflow_tool_agent(
            self.chat_session,
            execution_insights=execution_insights,
        )

        # Temporarily install this agent
        original_agent = self.chat_session._workflow_tool_agent
        self.chat_session._workflow_tool_agent = agent

        try:
            refined_message = self.chat_session._refine_user_query(
                message, self.chat_session.conversation_history
            )

            # Get planning insights for injection into planner prompt
            planning_insights = getattr(self.chat_session, '_planning_insights', None)

            # Set up planner LM for this pass
            planner_lm = dspy_utils.get_lm(planner_lm_role, planner_api_key_role)
            self._record_lm_params("planner", planner_lm)

            # Store planner_lm on the session so it can be used for replanning
            self.chat_session._current_planner_lm = planner_lm

            # Initialize capture list on the session to capture ALL plans
            # (initial + replanning during agent execution)
            self.chat_session._planning_steps_capture = []

            # Build initial query with next steps using the PLANNER LLM
            # The hook in build_query_with_next_steps will auto-capture the plan
            command_info = build_query_with_next_steps(
                refined_message, self.chat_session, planning_insights=planning_insights, planner_lm=planner_lm
            )

            # Get available commands for current context
            available_commands = _what_can_i_do(self.chat_session)

            # Run the agent with the specified AGENT LLM, reusing the WEC's shared
            # agent-invocation contract (dspy.context + AdapterParseError retry).
            agent_lm = dspy_utils.get_lm(agent_lm_role, agent_api_key_role)
            self._record_lm_params("agent", agent_lm)
            agent_result = self.chat_session._call_agent_with_retry(
                lambda: agent(
                    user_query=command_info,
                    available_commands=available_commands,
                ),
                lm=agent_lm,
            )

            # Extract result text
            result_text = (
                agent_result.final_answer
                if hasattr(agent_result, "final_answer")
                else str(agent_result)
            )

            # Build CommandOutput
            command_response = fastworkflow.CommandResponse(response=result_text)
            command_output = fastworkflow.CommandOutput(
                command_response=command_response
            )
            command_output.workflow_name = (
                self.chat_session.get_active_workflow().folderpath.split("/")[-1]
            )

            # Snapshot the in-process action log (actual resolved command_name +
            # parameters). MUST copy: the live list is cleared between passes,
            # so holding it across a clear would empty this pass's actions
            # (ruling I8 snapshot discipline).
            actions = list(self.chat_session.action_log)

            # Capture the full ReAct trajectory
            trajectory = dict(agent.current_trajectory)

            self.chat_session.summarize_and_record_turn(message, actions, result_text)

            # Flush workflow state
            if workflow := self.chat_session.get_active_workflow():
                workflow.flush()

            # Collect all captured planning steps (initial + replanning)
            planning_steps = list(
                getattr(self.chat_session, '_planning_steps_capture', [])
            )

            return command_output, trajectory, actions, planning_steps

        finally:
            # Restore original agent and clean up distillation-specific attributes
            self.chat_session._workflow_tool_agent = original_agent
            if hasattr(self.chat_session, '_current_planner_lm'):
                delattr(self.chat_session, '_current_planner_lm')
            if hasattr(self.chat_session, '_planning_steps_capture'):
                delattr(self.chat_session, '_planning_steps_capture')

    # ------------------------------------------------------------------
    # Insight extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Insight extraction: the fw.distill.extract span (§8) and the ledger (§13)
    # ------------------------------------------------------------------

    def _open_extract_span(
        self,
        kind: str,
        *,
        existing_insights: str,
        divergence_summary: str,
    ):
        """Open the `fw.distill.extract` span for one extractor call (§8).

        The step that actually decides what the rule is was the one step of a
        distillation run with no span at all, so the model invocation inside it
        had no association with the run it explains. Opening the span here also
        makes the `fw.llm.call` the extractor emits a CHILD of it, which is
        what "the extractor call must be observable" reduces to (§8).

        `existing_insights` is stored as LENGTH + SHA-256, never as body: the
        whole markdown corpus is pasted into the prompt and grows without
        bound, and "what did it dedupe against" needs the corpus's identity,
        which the ledger can reconstruct.
        """
        corpus = (existing_insights or "").encode("utf-8")
        return tracing.start_span(
            self.chat_session,
            tracing.SPAN_DISTILL_EXTRACT,
            attributes={
                "run_id": self.run_id,
                "kind": kind,
                "extractor_model": fastworkflow.get_env_var("LLM_DISTILLATION"),
                "divergence_summary": divergence_summary,
                "existing_insights_bytes": len(corpus),
                "existing_insights_sha256": hashlib.sha256(corpus).hexdigest(),
            },
        )

    def _record_extraction(
        self,
        span,
        kind: str,
        raw_output: str,
        insights: list[str],
    ) -> list[str]:
        """Close the extract span and record the outcome — empty or not (§13.3).

        An extraction that kept nothing is evidence, not an absence, and which
        of the two ways it kept nothing is what says whether the extractor or
        the parser is the problem.

        Two bullets of one call that normalize onto the same text are ONE
        insight ([DR31]): they mint one id, so keeping both would write two
        file lines against a single ledger row. `parsed_count` therefore counts
        what was kept, which is what the ledger and the file agree on.
        """
        insights = dedupe_insight_texts(insights)
        if insights:
            empty_reason = None
        elif not raw_output or raw_output.upper() == "EMPTY":
            empty_reason = EMPTY_REASON_EXTRACTOR
        else:
            empty_reason = EMPTY_REASON_PARSE
        insight_ids = [insight_id(self.run_id, kind, text) for text in insights]
        tracing.end_span(
            self.chat_session,
            span,
            attributes={
                "raw_output": raw_output,
                "parsed_count": len(insights),
                "empty_reason": empty_reason,
                "insight_ids": insight_ids,
            },
        )
        self._extractions[kind] = _ExtractionOutcome(
            kind=kind,
            span_id=getattr(span, "span_id", None),
            insights=list(insights),
            insight_ids=insight_ids,
            empty_reason=empty_reason,
            cited_divergence_ids=self._cited_divergences.get(kind, ()),
        )
        return list(insights)

    def _emit_insight_records(
        self,
        kind: str,
        insights: list[str],
        insights_file: Path,
        entry_numbers: list[int],
    ) -> None:
        """Write one `distillation_insights` row per insight, plus its citations.

        The citations are what makes §13.2's chain resolve in one query in both
        directions: insight -> divergence -> the teacher/student span pair, and
        back from either span to the insights drawn from it.
        """
        outcome = self._extractions.get(kind)
        span_id = outcome.span_id if outcome else None
        cited = (
            outcome.cited_divergence_ids
            if outcome
            else self._cited_divergences.get(kind, ())
        )
        created_at = _utc_now()
        for text, entry_number in zip(insights, entry_numbers):
            iid = insight_id(self.run_id, kind, text)
            tracing.emit_distillation_record(
                self.chat_session,
                "insight",
                {
                    "insight_id": iid,
                    "run_id": self.run_id,
                    "kind": kind,
                    "text": text,
                    "text_hash": insight_text_hash(text),
                    "extractor_span_id": span_id,
                    "insight_file": str(insights_file),
                    # DISPLAY ONLY (§13.1): the file renumbers, the id does not.
                    "file_entry_number": entry_number,
                    "created_at": created_at,
                },
            )
            for divergence_id in cited:
                tracing.emit_distillation_record(
                    self.chat_session,
                    "citation",
                    {"insight_id": iid, "divergence_id": divergence_id},
                )

    @staticmethod
    def _parse_execution_insights(raw_insights: str) -> list[str]:
        """Parse the execution extractor's raw output into insight lines.

        Split out from `extract_insights` unchanged, so the parser and the
        model can be told apart when the result is empty (§13.3).
        """
        # Treat "EMPTY" or empty string as no insights
        if not raw_insights or raw_insights.upper() == "EMPTY":
            return []

        # Parse bullet points: each line starting with "- "
        insights = []
        for line in raw_insights.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                insights.append(line[2:].strip())
            elif line and not line.startswith("#"):
                # Accept non-prefixed lines as insights too
                insights.append(line)

        return [i for i in insights if i]

    @staticmethod
    def _parse_planning_insights(raw_insights: str) -> list[str]:
        """Parse the planning extractor's raw output into insight lines.

        Keeps only lines starting with a digit followed by `.` within the first
        three characters, which is why a bullet-formatted answer yields `[]`
        and reads as EMPTY unless `empty_reason` separates them (§13.3).
        """
        if not raw_insights or raw_insights.upper() == "EMPTY":
            return []

        # Parse numbered rules (1. rule, 2. rule, etc.)
        insights = []
        for line in raw_insights.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line[:3]:
                if rule_text := line.split(".", 1)[1].strip():
                    insights.append(rule_text)

        return insights

    def extract_insights(
        self,
        teacher_traj: dict,
        student_traj: dict,
        divergence_summary: str,
        user_query: str,
    ) -> list[str]:
        """
        Use LLM_DISTILLATION to analyze the trajectory delta and extract insights.

        Uses the full ReAct trajectories (which include conversation context via
        observation_about_conversation_history and ask_user observations).

        The divergence_summary describes concrete action-level differences;
        the LLM uses the full trajectories to judge whether differences are
        genuine mistakes or context-justified.

        Returns a list of insight strings, or empty list if no genuine mistakes.
        """
        # Load existing insights for dedup
        from fastworkflow.utils.insights_loader import load_workflow_insights

        workflow = self.chat_session.get_active_workflow()
        existing_insights = (
            load_workflow_insights(workflow.folderpath, "execution_agent") or ""
        )

        # Format trajectories for the LLM
        teacher_formatted = self._format_trajectory_for_llm(teacher_traj)
        student_formatted = self._format_trajectory_for_llm(student_traj)

        # The label goes ambient around the whole call, so the extractor's own
        # fw.llm.call lands under `distillation_pass = 'extractor'` (§8).
        with self.chat_session.distillation_pass_scope("extractor"):
            span = self._open_extract_span(
                INSIGHT_KIND_EXECUTION,
                existing_insights=existing_insights,
                divergence_summary=divergence_summary,
            )
            try:
                lm = dspy_utils.get_lm(
                    "LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION"
                )

                with dspy.context(lm=lm):
                    extractor = dspy.ChainOfThought(InsightExtractionSignature)
                    result = extractor(
                        user_query=user_query,
                        teacher_trajectory=teacher_formatted,
                        student_trajectory=student_formatted,
                        divergence_summary=divergence_summary,
                        existing_insights=existing_insights,
                    )

                raw_insights = (getattr(result, "insights", None) or "").strip()
                insights = self._parse_execution_insights(raw_insights)
            except BaseException as exc:
                tracing.end_span(
                    self.chat_session,
                    span,
                    status=tracing.STATUS_ERROR,
                    attributes={"error_type": type(exc).__name__},
                )
                raise

            return self._record_extraction(
                span, INSIGHT_KIND_EXECUTION, raw_insights, insights
            )

    # ------------------------------------------------------------------
    # Insight persistence
    # ------------------------------------------------------------------

    def _insights_dir(self) -> Path:
        """Return (creating if needed) the workflow's Insights directory."""
        workflow = self.chat_session.get_active_workflow()
        workflow_name = Path(workflow.folderpath).name
        insights_dir = Path(workflow.folderpath) / "Insights" / workflow_name
        insights_dir.mkdir(parents=True, exist_ok=True)
        return insights_dir

    @staticmethod
    def _append_numbered_insights(
        insights: list[str],
        insights_file: Path,
        header: str,
        number_pattern: str,
        entry_format: str,
        insight_ids: Optional[list[str]] = None,
    ) -> list[int]:
        """
        Append ``insights`` to ``insights_file`` as a numbered list, continuing
        the numbering already present in the file, and return the entry number
        each one was written under.

        ``number_pattern`` is a regex (with one capturing group for the number)
        used to find existing entries; ``entry_format`` is a format string taking
        ``num`` and ``insight``. The two MUST correspond so the numbering the file
        is read with matches the numbering it is written with.

        ``insight_ids``, when given, appends `[DR31]`'s HTML-comment marker to
        each written line, which is what lets a markdown line resolve back to
        its ledger row after the file has been renumbered or hand-edited. The
        marker sits outside ``number_pattern`` by construction and is stripped
        by ``load_workflow_insights`` before any prompt sees it `[DR56]`.
        """
        from fastworkflow.utils.insights_loader import format_insight_marker

        if insights_file.exists():
            content = insights_file.read_text(encoding="utf-8")
            numbers = re.findall(number_pattern, content, re.MULTILINE)
            next_num = max(int(n) for n in numbers) + 1 if numbers else 1
        else:
            insights_file.write_text(header, encoding="utf-8")
            next_num = 1

        entry_numbers: list[int] = []
        with open(insights_file, "a", encoding="utf-8") as f:
            for index, insight in enumerate(insights):
                marked = insight
                if insight_ids is not None and index < len(insight_ids):
                    marked = f"{insight}{format_insight_marker(insight_ids[index])}"
                f.write(entry_format.format(num=next_num, insight=marked))
                entry_numbers.append(next_num)
                next_num += 1
        return entry_numbers

    def append_insights(self, insights: list[str]):
        """Append new insights to execution_agent_anti_patterns.md."""
        if not insights:
            return

        insights_file = self._insights_dir() / "execution_agent_anti_patterns.md"
        # [DR31]: one id per distinct normalized text, so a file line and a
        # ledger row are one-to-one even when a caller hands this the raw
        # parse. `_record_extraction` already folds the extractor's own output.
        insights = dedupe_insight_texts(insights)
        insight_ids = [
            insight_id(self.run_id, INSIGHT_KIND_EXECUTION, text)
            for text in insights
        ]
        entry_numbers = self._append_numbered_insights(
            insights,
            insights_file,
            header=(
                "# Execution Agent Anti-Patterns\n\n"
                "Critical mistakes to avoid when executing workflows, "
                "derived from distillation.\n\n"
            ),
            number_pattern=r"^(\d+)\.\s",
            entry_format="{num}. {insight}\n",
            insight_ids=insight_ids,
        )
        self._emit_insight_records(
            INSIGHT_KIND_EXECUTION, insights, insights_file, entry_numbers
        )

        self.execution_insights_extracted += len(insights)

    # ------------------------------------------------------------------
    # Planning insight extraction
    # ------------------------------------------------------------------

    def extract_planning_insights(
        self,
        teacher_steps: list[PlanningStep],
        student_steps: list[PlanningStep],
        divergence_summary: str,
        user_query: str,
        teacher_actions: list[dict],
        student_actions: list[dict],
    ) -> list[str]:
        """Extract prescriptive planning rules using LLM_DISTILLATION.

        Uses both planning traces and executed actions to give the insight
        extraction LLM full context for judging plan quality.
        """
        # Load existing planning insights for dedup
        from fastworkflow.utils.insights_loader import load_workflow_insights

        workflow = self.chat_session.get_active_workflow()
        existing_insights = (
            load_workflow_insights(workflow.folderpath, "planning_agent") or ""
        )

        # Format planning traces
        teacher_plan_str = self._format_planning_traces_for_llm(teacher_steps)
        student_plan_str = self._format_planning_traces_for_llm(student_steps)

        # Format executed actions for context
        teacher_actions_str = "\n".join(
            self._format_action(a) for a in teacher_actions
        ) or "(no actions executed)"
        student_actions_str = "\n".join(
            self._format_action(a) for a in student_actions
        ) or "(no actions executed)"

        with self.chat_session.distillation_pass_scope("extractor"):
            span = self._open_extract_span(
                INSIGHT_KIND_PLANNING,
                existing_insights=existing_insights,
                divergence_summary=divergence_summary,
            )
            try:
                lm = dspy_utils.get_lm(
                    "LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION"
                )

                with dspy.context(lm=lm):
                    extractor = dspy.ChainOfThought(PlanningInsightExtractionSignature)
                    result = extractor(
                        user_query=user_query,
                        teacher_plan=teacher_plan_str,
                        student_plan=student_plan_str,
                        divergence_summary=divergence_summary,
                        teacher_actions=teacher_actions_str,
                        student_actions=student_actions_str,
                        existing_insights=existing_insights,
                    )

                raw_insights = (getattr(result, "insights", None) or "").strip()
                insights = self._parse_planning_insights(raw_insights)
            except BaseException as exc:
                tracing.end_span(
                    self.chat_session,
                    span,
                    status=tracing.STATUS_ERROR,
                    attributes={"error_type": type(exc).__name__},
                )
                raise

            return self._record_extraction(
                span, INSIGHT_KIND_PLANNING, raw_insights, insights
            )

    def append_planning_insights(self, insights: list[str]):
        """Append new planning insights to planning_agent_insights.md."""
        if not insights:
            return

        insights_file = self._insights_dir() / "planning_agent_insights.md"
        # [DR31]: one id per distinct normalized text, so a file line and a
        # ledger row are one-to-one even when a caller hands this the raw
        # parse. `_record_extraction` already folds the extractor's own output.
        insights = dedupe_insight_texts(insights)
        insight_ids = [
            insight_id(self.run_id, INSIGHT_KIND_PLANNING, text)
            for text in insights
        ]
        entry_numbers = self._append_numbered_insights(
            insights,
            insights_file,
            header=(
                "# Planning Agent Insights\n\n"
                "Key insights for planning workflow execution strategies, "
                "derived from distillation training.\n\n"
            ),
            number_pattern=r"^## (\d+)\.",
            entry_format="## {num}. {insight}\n\n",
            insight_ids=insight_ids,
        )
        self._emit_insight_records(
            INSIGHT_KIND_PLANNING, insights, insights_file, entry_numbers
        )

        self.planning_insights_extracted += len(insights)


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------


def distill_message(
    chat_session: "fastworkflow.WorkflowExecutionContext", message: str
) -> DistillationResult:
    """
    Run teacher and student agents, extract BOTH planning and execution insights.

    Flow:
    1. Snapshot state
    2. Run teacher - capture planning traces + actions
    3. Save teacher's final state
    4. Restore pre-teacher state
    5. Run student - capture planning traces + actions
    6a. Compare planning traces → extract planning insights
    6b. Compare executed actions → extract execution insights
    7. On divergence: restore teacher's state; else keep student's
    8. Return teacher's output

    The whole of it runs inside one `fw.distill.run` span parenting a
    `fw.distill.pass` span per pass (§8), and is recorded as one
    `distillation_runs` row plus one `distillation_passes` row per pass
    (fix-sb8.2), written through the sink's writer thread ([DR46]).
    """
    ds = DistillationSession(chat_session)

    # Shed prior-turn action records at entry: the distillation branch bypasses
    # _run_agent (the only other clearer), so without this the previous turn's
    # actions would leak into the first pass (Phase 7 §2.7, ruling I8
    # clear-point discipline; replaces the old cwd action.jsonl file reset).
    chat_session.clear_action_log()

    teacher_model = fastworkflow.get_env_var("LLM_TEACHER_AGENT") or "LLM_TEACHER_AGENT"
    student_model = fastworkflow.get_env_var("LLM_STUDENT_AGENT") or "LLM_STUDENT_AGENT"
    teacher_planner_model = fastworkflow.get_env_var("LLM_TEACHER_PLANNER")
    student_planner_model = fastworkflow.get_env_var("LLM_STUDENT_PLANNER")
    extractor_model = fastworkflow.get_env_var("LLM_DISTILLATION")

    # The run wrapper: run-level, so distillation_pass stays NULL on it ([DR7]).
    # It parents both passes, which is what makes the pass boundary a
    # structural fact in the waterfall rather than only a filter (fix-kw7.11).
    run_span = tracing.start_span(
        chat_session,
        tracing.SPAN_DISTILL_RUN,
        attributes={
            "run_id": ds.run_id,
            "user_message": message,
            "teacher_agent_model": teacher_model,
            "teacher_planner_model": teacher_planner_model,
            "student_agent_model": student_model,
            "student_planner_model": student_planner_model,
        },
    )

    # The writer's loss counters as the run opens. Everything this run enqueues
    # — its own row included — is measured against this at completion, so a
    # discarded batch or a swallowed record write cannot leave the run claiming
    # comparable evidence that is not in the table ([DR49]).
    ds.snapshot_writer_counters()

    # Open the row now, not at the end: a run that dies mid-flight is still a
    # run somebody has to be able to find.
    ds._emit_run_record(
        channel_id=chat_session.observability_channel_id,
        conversation_id=chat_session.observability_conversation_id,
        user_message=message,
        workflow_name=getattr(chat_session, "_turn_entry_workflow_name", None),
        entry_context=getattr(chat_session, "_turn_entry_context", None),
        # comparable is NOT NULL and means "every pass entered on an equal
        # state_fingerprint" ([DR15]). At OPEN no pass has reached a boundary,
        # so the only honest value is 0/'evidence-incomplete'; the completion
        # write below replaces it with the verdict the fingerprints support.
        comparable=0,
        comparable_reason="evidence-incomplete",
        # The corpora as loaded at agent init, by size and hash: they are never
        # reloaded, so the prompt a run actually used is not the file's current
        # contents ([DR34]).
        insight_set_json=json.dumps(ds.insight_set(), separators=(",", ":")),
        extractor_model=extractor_model,
        started_at=_utc_now(),
        run_json=json.dumps(
            {"version": _RUN_JSON_VERSION, "status": "started", "run_id": ds.run_id}
        ),
    )

    completion: dict[str, Any] = {
        "comparable": 0,
        "comparable_reason": "evidence-incomplete",
    }
    run_json: dict[str, Any] = {
        "version": _RUN_JSON_VERSION,
        "run_id": ds.run_id,
        "status": "completed",
        "teacher_agent_model": teacher_model,
        "student_agent_model": student_model,
        "passes": [
            {"pass_label": "teacher", "role": "teacher", "seq": 0},
            {"pass_label": "student", "role": "student", "seq": 1},
        ],
    }
    run_attributes: dict[str, Any] = {}
    run_status = tracing.STATUS_OK
    teacher_raised = False
    # A pass that RAISED is not a pass that agreed. Without its own recorded
    # state such a run is stored `comparable=1, planning_diverged=0,
    # exec_diverged=0` with `completed_at` set — indistinguishable in SQL from
    # "the student matched the teacher", so §15's aggregates read a crash as
    # agreement and every rate they publish is wrong by commission.
    student_raised = False
    run_raised = False

    # [DR49]: hold the in-process Span objects the passes emit, so the
    # alignment reads what the pass produced instead of racing the writer
    # thread for it. Removed on every exit path in the finally below.
    ds.install_span_collector()

    try:
        try:
            # 1. Snapshot initial state
            initial_snapshot = ds.snapshot_workflow_state()

            # 2. Run teacher (returns planning_steps too now)
            _announce("TEACHER pass", teacher_model, style="bold magenta")
            teacher_output, teacher_traj, teacher_actions, teacher_plans = (
                ds._run_agent_pass(
                    message,
                    agent_lm_role="LLM_TEACHER_AGENT",
                    agent_api_key_role="LITELLM_API_KEY_TEACHER_AGENT",
                    planner_lm_role="LLM_TEACHER_PLANNER",
                    planner_api_key_role="LITELLM_API_KEY_TEACHER_PLANNER",
                    pass_label="teacher",
                    role="teacher",
                    seq=0,
                )
            )
        except BaseException as exc:
            # The half run (§18, fix-sb8.2). The teacher raising before the
            # student ran — dspy_utils.get_lm on an unset LLM_TEACHER_AGENT is
            # the live case — still writes its row: a run that could not start
            # is itself a fact worth seeing in the list. completed_at stays
            # NULL, and the raise propagates exactly as it does today.
            teacher_raised = True
            run_json["status"] = "teacher-raised"
            run_json["error_type"] = type(exc).__name__
            run_status = tracing.STATUS_ERROR
            run_attributes["error_type"] = type(exc).__name__
            raise

        # 3. Save teacher's final state
        teacher_final_state = ds.snapshot_workflow_state()
        teacher_entry = ds.pass_fingerprint("teacher", "entry_fingerprint")
        teacher_exit = ds.pass_fingerprint("teacher", "exit_fingerprint")

        # 4. Restore initial state for student run. This site restores toward
        # the PRE-teacher entry state, so that — and not the teacher's exit —
        # is what its restore_ok column is measured against (§6.2). 0 until the
        # restore both returns and lands.
        completion["restore_ok_pre_student"] = 0
        ds.restore_workflow_state(initial_snapshot)
        completion["restore_ok_pre_student"] = ds._restore_matches(teacher_entry)

        # 5. Run student
        _announce("STUDENT pass", student_model, style="bold cyan")
        try:
            student_output, student_traj, student_actions, student_plans = (
                ds._run_agent_pass(
                    message,
                    agent_lm_role="LLM_STUDENT_AGENT",
                    agent_api_key_role="LITELLM_API_KEY_STUDENT_AGENT",
                    planner_lm_role="LLM_STUDENT_PLANNER",
                    planner_api_key_role="LITELLM_API_KEY_STUDENT_PLANNER",
                    pass_label="student",
                    role="student",
                    seq=1,
                )
            )
        except Exception as e:
            logger.warning(f"Distillation: student agent failed: {e}")
            # No comparison happened: `align_and_record` never runs on this
            # path, so there are no divergence rows and the NOT NULL
            # `planning_diverged`/`exec_diverged` columns keep their DDL
            # default of 0. `student-raised` is the same precedent §18 sets for
            # `teacher-raised` — a queryable state of its own, and one every
            # §15 recipe's `comparable = 1` filter excludes.
            student_raised = True
            # This restore, and the divergence one below, restore toward the
            # TEACHER'S EXIT state — a different assertion from the column
            # above, against a different baseline (§6.2).
            completion["restore_ok_post_compare"] = 0
            ds.restore_workflow_state(teacher_final_state)
            completion["restore_ok_post_compare"] = ds._restore_matches(teacher_exit)
            run_json["status"] = "student-failed"
            run_json["error_type"] = type(e).__name__
            completion["completed_at"] = _utc_now()
            run_attributes["error_type"] = type(e).__name__
            return DistillationResult(
                command_output=teacher_output, run_id=ds.run_id
            )

        any_divergence = False

        # 6. Compare. [DR49]'s read barrier comes FIRST: no divergence row may
        # be written before every span it can cite has landed, or a merely
        # late span reads downstream as a fabricated `missing-in-student`
        # divergence — which §10.3 would then pin forever as the evidence for
        # an insight describing a bug that never happened.
        ds.read_barrier()

        if ds.alignment_available():
            # `comparable` is needed BEFORE the records are written, because
            # materiality is NULL on a non-comparable run ([DR20]). It is a
            # pure function of the pass evidence, which is complete by now, so
            # this agrees with the verdict the completion write takes below.
            comparable = bool(ds.comparability_fields().get("comparable"))

            plan_result = ds.align_and_record(
                level=alignment.LEVEL_PLAN,
                left_pass="teacher",
                right_pass="student",
                left_steps=ds.plan_steps("teacher"),
                right_steps=ds.plan_steps("student"),
                comparable=comparable,
            )
            action_result = ds.align_and_record(
                level=alignment.LEVEL_ACTION,
                left_pass="teacher",
                right_pass="student",
                left_steps=ds.action_steps("teacher"),
                right_steps=ds.action_steps("student"),
                comparable=comparable,
                left_answer=teacher_output.command_response.response,
                right_answer=student_output.command_response.response,
            )
            planning_diverged, planning_summary = ds.divergence_verdict(plan_result)
            exec_diverged, exec_summary = ds.divergence_verdict(action_result)
            divergence_counts = dict(plan_result.divergence_counts)
            for kind, count in action_result.divergence_counts.items():
                divergence_counts[kind] = divergence_counts.get(kind, 0) + count
            completion.update(
                # The aligner's OWN step counts, so a later reader can detect
                # a stored sequence truncated by retention by comparing them
                # against the rows that survived ([DR49]).
                left_steps=action_result.left_steps,
                right_steps=action_result.right_steps,
                material_divergences=(
                    plan_result.material_count + action_result.material_count
                ),
            )
            run_attributes.update(
                divergence_counts=divergence_counts,
                material_count=(
                    plan_result.material_count + action_result.material_count
                ),
            )
        else:
            # No sink: nothing to align and nowhere to record it (§7.6).
            planning_diverged, planning_summary = ds.compare_planning_traces(
                teacher_plans, student_plans
            )
            exec_diverged, exec_summary = ds.compare_trajectories(
                teacher_actions, student_actions
            )

        # 6a. Extract PLANNING insights
        if planning_diverged:
            planning_insights = ds.extract_planning_insights(
                teacher_plans, student_plans,
                planning_summary, message,
                teacher_actions, student_actions,
            )
            if planning_insights:
                ds.append_planning_insights(planning_insights)
            any_divergence = True

        # 6b. Extract EXECUTION insights
        if exec_diverged:
            exec_insights = ds.extract_insights(
                teacher_traj, student_traj,
                exec_summary, message
            )
            if exec_insights:
                ds.append_insights(exec_insights)
            any_divergence = True

        # 7. State restoration
        if any_divergence:
            completion["restore_ok_post_compare"] = 0
            ds.restore_workflow_state(teacher_final_state)
            completion["restore_ok_post_compare"] = ds._restore_matches(teacher_exit)
        # else: keep student's state (equivalent to teacher's), and the column
        # stays NULL because this site did not execute (§6.2).

        extracted = (
            ds.planning_insights_extracted + ds.execution_insights_extracted
        )
        # An extraction that kept nothing, and WHY it kept nothing (§13.3):
        # `extractor-returned-empty` is the model judging the difference
        # context-justified, `parse-yielded-nothing` is the parser discarding
        # an answer it got. One says the extractor is too conservative, the
        # other says the parser is too strict; today both are the same silence.
        empty_reasons = sorted(
            {
                outcome.empty_reason
                for outcome in ds._extractions.values()
                if outcome.empty_reason
            }
        )
        if any_divergence:
            _announce(
                "DISTILLATION: divergence found",
                (
                    f"{ds.planning_insights_extracted} planning + "
                    f"{ds.execution_insights_extracted} execution insight(s) extracted"
                    if extracted
                    else "no insights extracted"
                    + (f" — {', '.join(empty_reasons)}" if empty_reasons else "")
                ),
                style="bold yellow",
            )
        else:
            _announce(
                "DISTILLATION: no divergence",
                "student matched teacher — no insights extracted",
                style="green",
            )

        completion.update(
            planning_diverged=int(planning_diverged),
            exec_diverged=int(exec_diverged),
            planning_insights=ds.planning_insights_extracted,
            execution_insights=ds.execution_insights_extracted,
            # Diverged but the extractor produced nothing. The second arm
            # catches the case one extractor delivered and the other did not,
            # which the run-wide total alone reads as a success.
            extractor_empty=int(
                (any_divergence and extracted == 0) or bool(empty_reasons)
            ),
            completed_at=_utc_now(),
        )
        run_json["planning_summary"] = tracing.cap_attr_value(planning_summary)
        run_json["execution_summary"] = tracing.cap_attr_value(exec_summary)
        run_attributes.update(
            planning_diverged=int(planning_diverged),
            exec_diverged=int(exec_diverged),
            planning_insights=ds.planning_insights_extracted,
            execution_insights=ds.execution_insights_extracted,
        )

        return DistillationResult(
            command_output=teacher_output,
            planning_insights_extracted=ds.planning_insights_extracted,
            execution_insights_extracted=ds.execution_insights_extracted,
            run_id=ds.run_id,
        )
    except BaseException as exc:
        # Anything that escaped after the teacher pass returned — the extractor
        # LLM raising is the live case, since `extract_planning_insights` /
        # `extract_insights` are called without a guard. The divergence rows
        # were already written by then, so leaving `run_json["status"]` at the
        # "completed" it was initialized with would describe a run that never
        # finished comparing as a finished comparison.
        if not teacher_raised:
            run_raised = True
            run_json["status"] = "raised"
            run_json["error_type"] = type(exc).__name__
            run_status = tracing.STATUS_ERROR
            run_attributes["error_type"] = type(exc).__name__
        raise
    finally:
        # One completion write on every exit path, the teacher raise included.
        # The per-pass cost rollup runs first, because cache_asymmetric is one
        # of the run row's own columns (§6.3).
        cache_asymmetric = ds.finalize_pass_metrics()
        if cache_asymmetric is not None:
            completion["cache_asymmetric"] = cache_asymmetric
        # The closing barrier, then the loss counters — in that order. Nothing
        # this run enqueued may still be in a queue when the counters are read,
        # or a batch discarded at the barrier would be charged to nobody, and
        # the run would be completed as comparable evidence for spans and
        # records that are not in the table ([DR49]).
        ds.read_barrier()
        writer_loss = ds.check_writer_loss()
        if writer_loss:
            # Which counter moved, so "evidence-incomplete" can be diagnosed
            # rather than only observed. run_json is the run row's own
            # free-form column; the verdict itself is `comparable_reason`.
            run_json["writer_loss"] = writer_loss
        if teacher_raised:
            forced_reason = "teacher-raised"
        elif student_raised:
            forced_reason = "student-raised"
        elif run_raised:
            # The comparison did not run to completion. `evidence-incomplete`
            # is literally true here and needs no new enum value.
            forced_reason = "evidence-incomplete"
        else:
            forced_reason = None
        completion.update(ds.comparability_fields(forced_reason=forced_reason))
        # isolation_verified is deliberately absent: it stays NULL until
        # fix-35m.3's read-only surface check exists, and sb8 never writes 1
        # ([DR48]). A NULL is not readable as a pass — fix-sb8.10's promotion
        # view and fix-sb8.11's replay both refuse while it is not 1.
        run_json["comparable"] = completion["comparable"]
        run_json["comparable_reason"] = completion["comparable_reason"]
        run_json["fingerprints"] = {
            label: {
                "entry": row.get("entry_fingerprint"),
                "exit": row.get("exit_fingerprint"),
            }
            for label, row in ds._passes.items()
        }
        # §10.3's pin, at write time. Nothing else in the producer writes this
        # column, and `prune()` runs at every sink startup on a 30-day default,
        # so without it an insight's cited divergences, its citations and both
        # passes' spans are deleted at the horizon while the markdown rule and
        # the insight row survive, pointing at nothing (AC9, and AC2 with it).
        completion.update(ds.pin_fields(completion))
        completion["run_json"] = json.dumps(run_json, default=str)
        ds._emit_run_record(**completion)
        run_attributes.update(
            run_id=ds.run_id,
            comparable=completion["comparable"],
            comparable_reason=completion["comparable_reason"],
            isolation_verified=None,
            restore_ok_pre_student=completion.get("restore_ok_pre_student"),
            restore_ok_post_compare=completion.get("restore_ok_post_compare"),
            cache_asymmetric=completion.get("cache_asymmetric"),
        )
        tracing.end_span(
            chat_session, run_span, status=run_status, attributes=run_attributes
        )
        ds.remove_span_collector()
