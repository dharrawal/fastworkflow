"""OTel-shaped trace emission at the agent↔workflow boundary.

Implements the flight-recorder half of the observability design
(docs/fastworkflow_observability_studio_design.md §3.1): a ``TraceSink``
protocol with a no-op default, the v1 span taxonomy, and safe emission
helpers that NEVER raise to the caller — a broken sink degrades to a log
line, not a failed turn.

Spans are OTel-*aligned* records, not wire-conformant OTel:
``trace_id`` is the logical turn_key, span ids are opaque strings, and the
translation to real OTel ids is an external script's contract.

Sink discovery is duck-typed off the host object (WorkflowExecutionContext,
or ChatSession delegating to its core) via ``trace_sink`` /
``current_turn_key`` / ``trace_span_stack`` — deliberately NOT the
transport-queue contract, so queue-less embedders still trace.

This module is stdlib-only by design: it is imported by core runtime
modules and must never pull torch/dspy/transformers.

"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from fastworkflow.observability import enrichment

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Host propagation for deep emission sites (v2 spans, D3 as amended)
# ----------------------------------------------------------------------
#
# The NLU internals (intent detection, parameter extraction) run several call
# frames below CommandExecutor.invoke_command and have no reference to the
# WEC/ChatSession that owns the trace sink. Rather than threading a host
# parameter through the whole pipeline, invoke_command binds the host into a
# ContextVar for the duration of the call; deep sites read it back with
# ``current_host()``. The pipeline is synchronous on the calling thread, so
# the binding is race-free per turn — and it honors [R28]: the sink is still
# reached via the WEC, never via a transport queue.

_current_host: contextvars.ContextVar = contextvars.ContextVar(
    "fw_trace_host", default=None
)


def current_host() -> Any:
    """The trace host bound by the nearest enclosing ``host_scope`` (or None)."""
    return _current_host.get()


@contextlib.contextmanager
def host_scope(host: Any) -> Iterator[None]:
    """Bind *host* as the current trace host for the enclosed call stack."""
    token = _current_host.set(host)
    try:
        yield
    finally:
        _current_host.reset(token)


# A worker thread's own parenting stack. The host's stack is one plain list shared by every thread, so work run in a
# pool (the per-part stages, the argument checks) would push onto and parent under each other's spans. ``traced_call``
# gives each submitted call ``(host_stack, own_stack)``: while the host's stack is ``host_stack``, ``_get_stack``
# answers ``own_stack``, a copy of the caller's stack taken at submission. The call's spans then parent to the span
# that was current when it was submitted, its pushes are seen by no sibling and not by the caller, and the copy ends
# with the call's context.
_forked_stack: contextvars.ContextVar = contextvars.ContextVar("fw_trace_forked_stack", default=None)


def traced_call(fn: Any, *args: Any) -> Any:
    """A no-argument callable that runs ``fn(*args)`` under a copy of this thread's context with its own parenting
    stack seeded from the current one. Build it in the submitting thread: ``pool.submit(tracing.traced_call(f, x))``."""
    context = contextvars.copy_context()
    host_stack = _host_stack(current_host())
    if host_stack is not None:
        own = list(_get_stack(current_host()) or [])
        context.run(_forked_stack.set, (host_stack, own))
    return lambda: context.run(fn, *args)


# ----------------------------------------------------------------------
# Command call identity (arch §12.0 delta 1, §12.1 item 5)
# ----------------------------------------------------------------------
#
# A ``CommandOutput`` and the ``fw.command.execute`` span that produced it shared
# no key: the span carried a random uuid4 and the output carried nothing at all,
# so a reader holding a turn record could only guess which span belonged to which
# command outcome, from ordering and timestamps. ``command_call_id`` is that key,
# minted once per dispatch and stamped on both sides.
#
# The ContextVar is for the second half of the problem. An application command is
# dispatched through ``CommandExecutor.invoke_command``, which reaches the
# application only via an internal CME ``perform_action`` hop, and that hop may
# itself dispatch another core command — ``command_metadata_extraction/_commands/
# wildcard.py`` does exactly this for the ``IntentDetection/*`` and
# ``ErrorCorrection/*`` commands. Those inner calls have no span of their own,
# because capture rides the emission sites that already exist rather than adding
# new ones, so their correlation to the enclosing command is recorded as a
# child-call ledger stamped on the parent's span.

ATTR_COMMAND_CALL_ID = "command_call_id"
ATTR_PARENT_CALL_ID = "parent_call_id"
ATTR_CHILD_CALLS = "child_calls"
# The active command context's TYPE name (a context class name such as
# "Order") before and after the command ran; None when no workflow or context
# could be read. A type, not an instance: two equal values do not prove the
# command stayed on the same object.
ATTR_CONTEXT_BEFORE = "context_before"
ATTR_CONTEXT_AFTER = "context_after"

# The emitter's own attribute-contract version, stamped on every span by `_emit`
# (arch §12.0 delta 5). It is an ATTRIBUTE rather than a `Span` field because
# `spans` has fixed columns and `attributes` is the JSON bag that survives the
# store boundary — a span that describes its own contract has to carry the
# version where the store actually keeps it.
ATTR_SPAN_CONTRACT_VERSION = "span_contract_version"

# The alias of the planning-command frame a span was opened under (design 3.5): "" for
# the turn frame. It is stamped by `start_span` from the host's top frame, so no emitter
# writes it and a span opened inside a nested frame keeps that frame's alias when it
# closes after the frame has been popped.
ATTR_FRAME_ALIAS = "frame_alias"

_current_call_id: contextvars.ContextVar = contextvars.ContextVar(
    "fw_command_call_id", default=None
)

_child_call_ledger: contextvars.ContextVar = contextvars.ContextVar(
    "fw_child_call_ledger", default=None
)


def new_command_call_id() -> str:
    """Mint one command-call id."""
    return uuid.uuid4().hex


def current_call_id() -> Optional[str]:
    """The command call this frame runs under, or None at the outermost dispatch."""
    return _current_call_id.get()


@contextlib.contextmanager
def call_scope(call_id: str, *, command_name: Optional[str] = None) -> Iterator[list]:
    """Bind *call_id* as the current command call; yield the child-call ledger.

    The ledger is created by the OUTERMOST scope and shared by every nested one,
    so one dispatch produces one flat list in which each entry names its own
    parent. Yielding it rather than making the caller read it back through the
    ContextVar matters: the span closes *after* the scope has exited, and by then
    the ContextVar has been reset.

    A scope with no parent adds nothing to the ledger — it is not a child of
    anything, and its own id is already on the span.
    """
    parent = _current_call_id.get()
    ledger = _child_call_ledger.get()
    ledger_token = None
    if ledger is None:
        ledger = []
        ledger_token = _child_call_ledger.set(ledger)
    if parent is not None:
        ledger.append(
            {
                "call_id": call_id,
                "parent_call_id": parent,
                "command_name": command_name,
            }
        )
    call_token = _current_call_id.set(call_id)
    try:
        yield ledger
    finally:
        _current_call_id.reset(call_token)
        if ledger_token is not None:
            _child_call_ledger.reset(ledger_token)


# ----------------------------------------------------------------------
# Span-name taxonomy
# ----------------------------------------------------------------------

# The version of the span ATTRIBUTE contract — which keys each emitter sets and
# what they mean. Distinct from the DB's `user_version` (which versions tables and
# columns) and from `turns.record_version` (which versions the TurnResult
# envelope): `spans.attributes` is an unversioned JSON bag, so a replay or a
# calibration report comparing two runs has no way to notice that an emitter
# started spelling a key differently or changed a value's units.
#
# Recorded in run provenance by the evidence-grade run mode (arch §12.0 delta 6).
# Bump this whenever an emitter's attribute keys or their meanings change.
#
# This is deliberately ONE number for the whole taxonomy. Arch §12.0 delta 5 asks
# for a version *per emitter*, which is fix-ajv.8's work; a single number is what
# the evidence-grade run mode needs to detect drift between two runs, and it is
# honest about what it can and cannot localize — it says "something changed",
# not which emitter.
#
# v2 (EXP-003 capture slice): fw.command.execute and fw.agent.tool_call gained
# command_call_id, parent_call_id, child_calls, context_before, context_after and
# consequence. Bumped under the rule stated directly above — a run recorded
# before this and a run recorded after it are not attribute-comparable, and a
# calibration report joining them would silently read the older run's missing
# handles as "no context transition" rather than as "not captured".
#
# Amendment (fix-ajv.8): the per-emitter versions arch §12.0 delta 5 asks for now
# exist in `SPAN_CONTRACTS` below, and this number is KEPT rather than replaced.
# The two answer different questions and a reader needs both: a run-to-run
# comparison reads this one to learn that something moved, and the per-emitter map
# localizes what. Replacing it would force every comparison to diff a map to
# answer a yes/no question, and dropping it from provenance would break every
# existing reader of `ObservabilityProvenance.span_contract_version`.
#
# v3: every span now carries `span_contract_version` in its attributes;
# fw.agent.tool_call gained the §12.1.1 capture keys at its third emission site
# (workflow_agent.py, previously the only unmigrated one); and fw.nlu.intent's
# `classifier` attribute gained `topk_scores`.
#
# v4: routing. fw.nlu.intent's known-name refusal keys were declared here.
# The emitter had already been writing them. An intermediate draft of this
# same number also named an auto-navigation flag and four fw.command.execute
# keys for a composed step. Those keys did not remain in the contracts.
#
# v5: the aggregate moves so this number and SPAN_CONTRACTS describe the same
# taxonomy. fw.command.execute v3 has no composed-step attributes.
# fw.nlu.intent v3 keeps the known-name keys and does not carry
# auto_navigation_enabled. Auto-navigation was removed.
#
# v6: fw.search.route added -- a removed optional decision-model routing step
# on search_memory over listings (span name kept for historical traces).
# No existing span changed.
#
# v7: fw.finish_check added (historical; no longer emitted) -- and fw.agent.step v3 replaces the
# `roster_nudge` attribute with `finish_check_note` (the roster nudge was removed).
#
# v8: fw.planner.plan / fw.planner.replan v2 -- `plan` may be the structured plan
# rendered with "(optional)" / "(needs the user)" flags, and the span gains
# `plan_source` and the redacted `subjects`.
# Structured planning is disabled (2026-09-28): the keys are unchanged, so the
# version is too; `plan_source` is now only "text" or "none" and `subjects` [].
#
# v9 (fix-txxy): `fw.distillation.pass` joins the taxonomy. A turn that ran a
# teacher pass and a student pass used to record both under one flat trace with
# nothing saying which activity was whose, so a run recorded before this and a
# run recorded after it are not comparable on that question: the older one is
# "no pass identity recorded", not "one pass".
#
# v10: fw.command.execute v4 and fw.agent.tool_call v2 -- `context_before` /
# `context_after` change from a context-handle mapping to the plain context type
# name string.
#
# v11: fw.command.execute v5 and fw.agent.tool_call v3 -- the `consequence`
# attribute is gone. It was graded from the workflow's runtime manifest, which
# no longer exists, so a run recorded before this and after it differ in which
# keys they carry.
#
# v12: fw.agent.step v5 and fw.planner.plan / fw.planner.replan v3 -- the
# `finish_check_note` attribute on fw.agent.step and the `subjects` attribute on
# the planner spans are no longer written (the finish check and structured
# planning were removed). fw.search.route is no longer declared: nothing emits
# it, so it has no contract left to version.
#
# v13: fw.turn v2 -- the turn records `unattended`: whether it ran with no
# ask_user tool. A turn recorded before this carries no flag, which reads as
# "not recorded", not as attended.
#
# v14: fw.planner.plan / fw.planner.replan v4 -- `plan_source` may be
# "skill_call", and the spans gain `skill`, `skill_args` and `skill_fallback`
# (planning commands as parameterized commands).
#
# v15: fw.nlu.param_extraction v4 -- `unwrapped_quoted_values`,
# `defaulted_sentinel_values` and `out_of_range_values` (fix-7t0q).
#
# v16: fw.skill.phase added (fix-hia1.2) -- one span per phase run of a phased
# planning command call (`fastworkflow.planning_phases`). No existing span changed.
#
# v17: fw.skill.phase v2 (fix-hia1.7, fix-hia1.3) -- `answer_only` (a phase whose
# steps name no command, answered by one model call with no tools), and for one
# item of a fan-out phase `item`, `item_index`, `item_count`, `items_total` and
# `record_issues` (what the item record check flagged).
#
# v18: fw.skill.phase v3 (fix-hia1.9) -- `skipped` (a phase whose callable `when`
# was false when reached: no executor ran) and, for an item of a fan-out phase
# with `until`, `stop_condition_met`.
#
# v19: fw.nlu.param_extraction v5 (fix-hia1.15, fix-hia1.14) --
# `deterministic_form` and `unbound_value`.
#
# v20: fw.agent.step v6 (fix-5jss) -- `folded_command`: the command text the
# command tool built from arguments the model passed beside `command`.
#
# v21: fw.agent.step v7 and fw.skill.phase v4 (fix-6llp) -- code runs a phase's
# leading mechanical steps: `by_code` on a step code ran, and on the phase
# `steps_by_code` and `code_stop`. fw.nlu.param_extraction's `deterministic_form`
# may be "bare" (no new attribute).
#
# v22: fw.skill.phase v5 (fix-hia1.30) -- `answer_by_code` on an answer-only phase:
# code wrote its answer from its steps (every step verbatim or a guard), no model call.
#
# v23: fw.planner.plan / fw.planner.replan v5 (fix-hia1.34) -- `skill_notes`: what
# the argument guards and verify verdicts changed in a planning command call's arguments.
#
# v24: `frame_alias` on every span (design 3.5, Phase 4 of fix-3rsq, fix-3rsq.4.6): the
# alias of the planning-command frame the span was opened under, "" for the turn frame.
# It is an envelope attribute, stamped by the emission machinery rather than declared by
# an emitter, so no per-emitter version moved for it. fw.skill.phase v6 (fix-3rsq.4.10)
# declares `uses_result`, which it has written since Phase 3 without declaring it.
#
# v25: fw.planner.plan / fw.planner.replan v6 (Phase 6 arm selector, fix-3rsq.6.7) --
# `experiment_arm` (the measurement arm in force: "A" today's path, "B" the fast path)
# and `fast_path` (this initial plan is one planning-command call the fast path runs), and
# `recipe_choice` (arm B: the planner's yes or no on whether a matched recipe covers the request;
# None when no check ran). All three are logged on every planner span; a span recorded before
# this carries none of them.
#
# v26: fw.planner.plan / fw.planner.replan v7 -- `skill_refusal_asked`: a refused planning command call's
# question was put to the user, and the turn suspends on it (see the contract's v7 note). A span
# recorded before this carries no such key, which reads as "not asked".
#
# v27: fw.nlu.param_extraction v6 (command-name guard) -- `command_name_values`: a parameter value
# that was the name of a command (the called command's own, or another's in identifier form) and
# was treated as not given. A span recorded before this carries no such key.
#
# v28: fw.agent.execute v2 -- `fast_path_refusal_asked`: the fast path's one planning-command call
# (arm B) was refused for a missing argument, and the question was put to the user, so the run
# suspends on it instead of the agent going on from its plan. The fast path has no span of its
# own: its planner span closes before the run, so the run's executor span carries the fact. A span
# recorded before this carries no such key, which reads as "not asked".
#
# v29: fw.nlu.param_extraction v7 (word screens) -- `word_screen_values`: a free-text value that was
# only a pronoun or only a kind word or field name of the workflow, and was treated as not given. A
# span recorded before this carries no such key.
#
# v34: fw.skill.answer v1 (fix-3rsq.p14.6) -- new: the deferred model call that writes a planning
# command call's answer when its __result__ returns None (3.16). One span per call, a child of the
# turn's planning command work; a call whose answer is text or __result__'s makes no such span.
#
# v33: fw.nlu.intent v4 (fix-jd9o.1) -- `jev_tiebreak`: Jev's choice among an ambiguous intent's
# candidates, with `matcher_layer` "jev_tiebreak" when it routed. fw.nlu.param_extraction v9
# (fix-jd9o.3) -- `fields_reset_not_given`: fields an LLM extraction filled that Jev judged not given,
# reset to their not-given value. A span recorded before this carries neither key.
#
# v32: fw.nlu.param_extraction v8 (fix-3rsq.one-hook) -- the validation hook replaces db_lookup as the one
# place a parameter is corrected or refused. `db_lookup` is no longer written; `validation_hook` records
# `changed_fields` (a value the hook rewrote in place binds the slot, signal "validation_hook") and
# `refused_fields` (the fields of a fastworkflow.Refused). A span recorded before this carries `db_lookup`
# and no change list.
# v31: fw.approval v1 (new) -- one span per approval-binding transition (ADR-EXP-025 section 12):
# the gate's refuse, queue, ask, shadow and each resolution. It carries hashes, state and the
# identity class and reference, never the token or the raw parameters.
# v30: fw.planner.plan / fw.planner.replan v8 and fw.agent.execute v3 -- the Phase 6 arms were removed
# by owner decision 2026-10-30. `experiment_arm`, `fast_path` and `recipe_choice` (the v25 keys of the
# planner span) and `fast_path_refusal_asked` (the v28 key of the executor span) are no longer declared
# or written. A span recorded before this carries them as it wrote them; nothing reads them as a live path.
#
# v35: fw.turn v3 (fix-3rsq.plan-review.2.8) -- `subtask_review`: how the turn's sub-tasks were reviewed before they
# ran: "fast_path" (one sub-task, a confident head or an exact name: no review), "open" (the review was asked and
# the turn was still awaiting it), "reviewed" (accepted), "declined", "expired", "not_reviewed" (no one could answer:
# the draft ran as proposed), or "not_applicable" (the turn never reached the gate). A span recorded before this
# carries no such key.
#
# v36: fw.nlu.param_extraction v10 (fix-qjf4) -- the argument checks of an LLM extraction
# (``fastworkflow.utils.argument_checks``): `argument_check_refusals` (the values the first extraction
# gave that were refused), `argument_check_retry` (the extractor was asked once more with the refusal),
# `argument_check_refusals_final` (the values still refused after it, treated as not given) and
# `argument_check_respelled` ({field: value}: the request's or the declared spelling taken). A span
# recorded before this carries none of them.
#
# v37: fw.turn v4 (fix-48np.1) -- `subtask_bindings`: with drafting on (FW_DRAFT_ARGUMENTS), one record per bound
# sub-task the turn ran: `subtask`, `means`, `shown` (the values the review showed), `ran` (the arguments the bound
# command ran with, or null), `match` and `deviation` (why the run did not use the shown values: refused or unbound,
# so the agent ran it; drafted at run time; a value other than the one shown). Null when there is none. A span
# recorded before this carries no such key.
#
# v38: fw.agent.step v8 (main's policy verifiers, merged into the unified command architecture) -- the
# policy verifier's decision on a step it rewrote (fastworkflow.policy.policy_runtime): `policy_mode`, `policy_outcome`,
# `policy_source`, `policy_table_version`, `policy_intent`, `policy_requests`, `policy_judgement`, and `answer`
# (the answer a finish policy sent back). Main numbered this v24; on this line it follows v37.
#
# v39: fw.planner.plan / fw.planner.replan v9 (fix-verify-finish) -- `skill_notes` also carries one note per verify
# verdict the planning command call read ("verify <param>=<value> (<probability>, sure | not sure, not used)"), so a
# verdict that changed no argument is still on the span. The attribute set is unchanged; the content is. The
# classifier calls behind the verdicts now run under the turn's context, so each leaves an fw.llm.call span.
#
# v40: fw.skill.phase v7 (fix-variance) -- `reextracted`: True on the phases of a planning command call planned
# again after its first phase's lookup found nothing (the arguments extracted once more, the call restarted once).
# Null on every other phase. The attribute set gains this key; no other span changes.
#
# v41: fw.nlu.param_extraction v11 (spec 3.1-3.7) -- the validation step is the Input's model validators
# (`@model_validator(mode="after")`), and the Signature's `validate_extracted_parameters` hook is gone. The
# `validation_hook` record keeps its keys (ran, is_valid, message, changed_fields, refused_fields), but its
# facts now come from the Input's model validators: `changed_fields` are the fields a validator wrote back,
# and `refused_fields` the fields of a fastworkflow.Refused a validator raised. The record is always emitted
# when the validation step runs, with ran=False when the Input declares no validator (no facts to report).
# The key and the attribute set are unchanged; a span recorded before this carries the same keys with the
# hook's meaning.
# v42: fw.ask_user v2 (fix-4da9) -- `user_responses`: the ordered list of every reply the question took, the final
# one last. A review (plan or sub-task approval) can take several replies before it settles, and `user_response`
# keeps only the final one. `user_response` is unchanged. A span recorded before this has no `user_responses`.
#
# v43: fw.compose v1 (new; fix-weih.28) -- one span per compose call of the run_chatbot page (POST /compose_answer_card,
# POST /compose_question_form). It carries the counts and reason codes of the composer's INFO log line, the model id,
# the outcome and the duration, and never user text, result values or raw model output. It is written by the server
# after the turn finished, filed under the turn's trace id when the page sent the turn key (parent: the turn root)
# and under its own `compose-<id>` trace otherwise. A span recorded before this does not exist.
#
# v44: fw.llm.call v2 (fix-vdaj) -- what a stalled call needed to be seen without inference. `role` (the model role the
# LM was built for: LLM_PARAM_EXTRACTION, LLM_AGENT, ...; absent for an LM not built by get_lm), `timeout_s` (the
# per-attempt timeout in effect), `num_retries` (the retries configured; DSPy makes up to num_retries + 1 attempts),
# `outcome` ("ok", "timeout" or "error"), `timed_out` (a timeout anywhere in the exception's cause chain) and
# `exception_type` (the class name of the failure; `exception` keeps the repr). The attempts actually made are NOT
# recorded: DSPy retries inside one call and its callbacks fire once around all the attempts, so the library does not
# expose the count. A span recorded before this carries none of the six keys.
#
# v45: fw.compose v2 (fix-weih.35) -- `answer_chars` (the length of the answer the page sent, which every composer
# check reads in full) and `excerpt_chars` (the length of the excerpt the model was shown instead of the whole answer;
# 0 when it saw the whole answer). They are the two numbers the composer's INFO line already carried for a long answer.
# Null for a question form. A span recorded before this carries neither key.
#
# v46: fw.turn v5 (fix-1ma8.1) -- `subtask_plan_ambiguity`: one record per sub-task of the turn's draft whose
# planning-command match was ambiguous (another planning command cleared the shortlist floor and the winner was not a clear
# leader): `subtask` (its number), `means` (the winner, as the review shows it) and `alternatives` (the names of the other
# plans that fit). Names only, never user text or scores. Recorded whether or not the review was opened (an unattended
# turn runs the winner and carries the record). Null when none was ambiguous. A span recorded before this carries no
# such key.
#
# v47: fw.nlu.intent v5 (fix-yu3l.2) -- `engine` (which detector engine answered when the classifier layer decided:
# "classifier" today) and `engine_fallback` (that engine answered because the preferred one was not available; False
# today). Both are absent when a layer above the engine decided (an exact name, the fuzzy pre-match, the cache). A span
# recorded before this carries neither key.
#
# v48: fw.nlu.intent v6 (fix-yu3l.3) -- `matcher`: the model matcher's facts when it decided the scope
# (`matcher_layer` "matcher", `engine` "matcher"): {confident, top_label, confidence, topk_labels, topk_scores,
# none_score, threshold, unsure_threshold, scope_path, model, latency_ms, calls, and error for a failed call}.
# `engine_fallback` is then True when the matcher stood in for a head that does not exist. Absent when the
# classifier or a layer above decided. A span recorded before this carries no such key.
#
# v49: fw.nlu.intent v7 (hybrid intent detection) -- `engine_mode`: the INTENT_DETECTION_ENGINE mode in force
# when an engine decided ("auto" | "classifier" | "matcher" | "hybrid"; "hybrid" without the planner key is "auto").
# In hybrid, a span whose `engine` is "matcher" and which also carries `classifier` is a scope whose trained head
# was not confident: the head's facts are kept beside the matcher's. Absent when a layer above the engine decided.
#
# v50: fw.nlu.intent v8 (one detection chain, fix-o5wi.2) -- the engine setting is gone, so `engine_mode` is no
# longer written, and neither is `jev_tiebreak` (the tie-break among an ambiguous intent's candidates was removed;
# `matcher_layer` is never "jev_tiebreak"). `engine` says who decided: "classifier" (the scope's trained local
# model), "matcher" (the remote decision model) or "question" (no link was confident: the candidates were put to
# the user or the agent). `engine_chain` lists the links that were asked, in order (["classifier"],
# ["classifier", "matcher"] or ["matcher"]); with `engine` "question" the candidates are the last link's. A span
# with `classifier` and `matcher` is a scope whose local model was not confident; when the matcher's call failed
# (`matcher.error`) the local model's answer stood and `engine` is not "matcher". `classifier.model_tier` may be
# "embedding" (the embedding head). A span recorded before this may carry `engine_mode` and `jev_tiebreak`.
#
# v51: fw.nlu.intent v9 (fix-o5wi.8) -- the similarity cache was removed from intent detection: the ladder is
# exact name, fuzzy name, local model, remote model, question. `cache_similarity` and `cache_similarity_threshold`
# are no longer written and `matcher_layer` is never "embedding_cache". A span recorded before this may carry
# them; a reader must show an unknown layer as recorded.
#
# v52: fw.nlu.intent v10 (fix-o5wi.9) -- the local model never passes a request up on its own, so it no longer answers a
# lone "not here" because a command of another scope is ahead (`escalation_outcome` is then not "escalate" for that
# reason). It hands the request on with the best reachable commands of other scopes among the candidates:
# `classifier.outside_labels` / `outside_scores` (the commands outside the scope it ranks ahead of its own, best first),
# `classifier.handed_on` (the ones added to the candidates), and `classifier.escalation_margin` is no longer written
# for a head trained under the new rule. `chosen_command_elsewhere`: in the ambiguity stage, the reply named a command of
# another scope on the path; this scope declined it so the walk reaches the scope that has it (the qualified name).
# A span recorded before this has none of these.
#
# v53: fw.nlu.intent v11 (remote outage visibility) -- `remote_failed` ({error_class, status_code}, never the message):
# the remote decision model was asked for this scope and its call failed, so the local answer or the question stood in
# (the fallback is unchanged). `matcher.error` carries the failure text as before, and `matcher.threshold` is now the
# accept threshold (INTENT_REMOTE_ACCEPT_THRESHOLD, default 0.9; it was a fixed 0.5) with `matcher.decline_threshold`
# beside it (INTENT_REMOTE_DECLINE_THRESHOLD, default 0.9); `matcher.error_class` and `matcher.status_code` join
# `matcher.error`. A span recorded before this has no `remote_failed`, and its `matcher.threshold` is 0.5.
#
# v54: fw.nlu.intent v12 (one intent loop, fix-o1lg S7) -- the span is one DECISION, not one prediction per scope of the
# parent walk: `intent_loop.resolve` decides the whole walk (exact name, foreign-name guard, fuzzy name, the local model over
# every scope first, then the chain), so there is one span per request and `context` is the scope the decision settled in
# (the current context when nothing was decided). `stage` is `INTENT_DETECTION` for it. A reply to the which-command
# question is read by the same loop and recorded as its own span with `stage` `INTENT_AMBIGUITY_CLARIFICATION` or
# `INTENT_MISUNDERSTANDING_CLARIFICATION`, `matcher_layer` `question_reply` (replacing `exact_prefix`,
# `fuzzy_prematch` and `clarification_default` for a reply), `engine` `user`, and `chosen_command_elsewhere` when the picked
# command lives in another scope than the current one. `fuzzy_prematch_tie` now also reaches the plan path's decision. A
# remote model that answered for the local model's scope can no longer be asked once per scope: `engine_chain` is the links
# asked for the whole decision. A span recorded before this is per scope and its `clarification_default` layer is gone.
#
# v55: fw.nlu.param_extraction v12 -- `argument_refusal`: a refusal of the call as a whole that names no input. A span recorded
# before this does not have it.
#
# v56: fw.turn v6 (fix-hfqv, recording only) -- `subtask_split`: for a turn whose request the model was asked to split
# (a turn run with `use_plan` is never split), one record {request, raw, merged, changed_by, attempts}: `request` is the
# text the splitter saw, `raw` the model's sub-tasks of the attempt that was used (a list of strings, before any code
# rule), `merged` the list the turn runs with after the code rules (one entry, the request itself, when it was left whole;
# also when the model's call failed, then `raw` is empty), `changed_by` the code steps that changed the list, in order
# (merge rules `stutter`, `invented_or_meta`, `contained`, `attach_context`, `restore_cut_subjects`, `group`,
# `comparison`, `restore_dropped`, `one_per_record`, `carry_limits`, and `reduced_to_cap`, `whole_if_placeholder`; absent
# when none changed it) and `attempts` the model calls made (1 or 2). Each text is capped at `MAX_ATTR_BYTES` like
# `user_message`. User content: stored, redacted and erased with the turn like `user_message` and the `fw.llm.call` that
# holds the same request. Null when the turn was not split. A span recorded before this carries no such key.
#
# v58: fw.nlu.intent v13 and fw.nlu.param_extraction v14 (bd fix-o1lg, one decision on both paths): the plan path writes
# the two spans for a part's own decision and its own command's extraction, as the "/" path always did, from the same
# writers (`intent_detection.traced_decision`, `parameter_extraction.traced_extraction`): one `fw.nlu.intent` span per
# decision (a memo read writes none; a pinned plan is not detected, so it writes none) and one `fw.nlu.param_extraction`
# span for the extraction of the part's command (the draft's, including the review's edits that draft again).
# New attribute `part` on both: the 1-based number of the part of the turn the decision or extraction is for, present on
# the plan path when the caller knows it, absent on the "/" path and on a nested step command's spans (which are not a
# part's own). A span recorded before this carries no such key, and before this the plan path wrote neither span for a part.
#
# v57: fw.nlu.param_extraction v13 and fw.planner.plan / fw.planner.replan v10 (bd fix-qjne.21, recording only) --
# `argument_check_not_in_request`: the sorted names of the UNMARKED `str` and `int` fields (no `copied_from_command_text`
# or `record_named_in_command_text`) whose extracted value the request does not write, the test the copy check applies to
# a marked field. Names only, never the values; the key is absent on a parameter extraction span and null on a planner span
# when there is none. Advisory: the value stands, nothing is refused. A span recorded before this carries no such key.
#
# v59: fw.turn v7 -- `subtask_split.failed`: why the split gave no answer and the request ran as one part, `cut_off`
# (every answer stopped at the splitter's output cap, the retry without the reasoning field included) or `error` (the
# model call raised); absent when the split answered. A span recorded before this carries no such key.
#
# v60: fw.nlu.param_extraction v15 -- `argument_left_empty`: the sorted names of the OPTIONAL marked fields the duplicate
# rule left not given (two marked fields held one value the user writes fewer times; the note on `argument_notes` and on
# the review item or question says why). Names only, never the values; absent when none. A span recorded before this
# carries no such key, and before this such a field was refused (`argument_check_refusals`, `invalid_fields`).
#
# v61: fw.turn v8 (bd fix-qjne.26 stage 0, recording only) -- `subtask_split.prompt_digest` and
# `subtask_split.prompt_source`: which splitter prompt produced the split. `prompt_digest` is the first 12 hex
# characters of the SHA-256 of the splitter signature's instructions and field descriptions
# (`workflow_agent.split_prompt`); `prompt_source` is `built_in` or `workflow` (the workflow's `_commands/_workflow.py`
# replaces the instructions or a field description). Absent when the prompt was never built. A span recorded before
# this carries no such keys.
#
# v62: fw.turn v9 (bd fix-qjne.26 stage 3) -- `subtask_split.prompt_source` gains the value `built_in+workflow`: the
# built-in splitter rules (`workflow_agent.SPLIT_RULES`) with the workflow's `split_examples` (`_commands/_workflow.py`)
# given as demos in place of the generic examples; `prompt_digest` then also covers each example's request and parts.
# The value `workflow` is no longer written: a `Signature` in that file (a replacement prompt) is a load error, so
# `prompt_source` is `built_in` or `built_in+workflow`. A span recorded before this never carries the new value. Also `subtask_split.gate`: the one-job gate's verdict, {score, label, threshold, skipped}
# (`score` the remote decision model's probability of "one", `label` its answer, `one` or `several`, `threshold` the bar
# `FW_SPLIT_GATE_THRESHOLD`, `skipped` whether the splitter was skipped), present when the gate was asked and absent
# when it was not (no remote decision model, the gate `off`, a failed call, a `use_plan` turn). A request the gate kept
# whole is recorded with `raw` empty, `merged` the request, `attempts` 0 and no `prompt_digest` or `prompt_source`
# (they are present only when the splitter's prompt was built). A span recorded before this carries no such key.
#
# v63 (bd fix-qjne.34, one revision): the span tree follows the turn's stages. Under `fw.turn` (v10) every turn has, in
# order, the stages it went through, each ONE span with a deterministic id (`stage_span_id`) that a turn waiting on a
# person closes with status `awaiting_user` and reopens when it resumes, in any process (`resumed`, `segments`):
# `fw.split` (new; the record that was the turn's `subtask_split`, `changed_by` now `rules`) with `fw.split.gate`,
# `fw.split.splitter` (one per model attempt), `fw.split.merge` and `fw.split.referent` (all new); `fw.resolve` (new;
# `plan_ambiguity`, was the turn's `subtask_plan_ambiguity`) with one `fw.resolve.part` per part (new); `fw.review`
# (new; `outcome`, was the turn's `subtask_review`; it replaces the review's fake `fw.skill.phase phase=review` and its
# agent loop) with `fw.review.reply`, `fw.review.edit` and `fw.review.revision` (new; the reply's read replaces the
# review router's `fw.nlu.intent` / `fw.nlu.param_extraction`) and the review record's `fw.approval`; `fw.run` (new)
# with one `fw.run.part` per part (new; `means`, `outcome` ran / not_run / refused, `reason`, `answered_by`, and the
# agent loop's facts that were `fw.agent.execute`'s; a part dropped at resolve is a run part not run, reason
# `dropped_at_resolve: ...`, numbered as the split numbered it); `fw.answer` (new; `written_by`, `chars`). The direct
# ("/") path has `fw.resolve` and `fw.run` with one part each and no split or review. `fw.turn` gains `path` (`plan`,
# `direct`) and `mode` (`agent`, `no_agent`) and loses `subtask_split`, `subtask_review` and `subtask_plan_ambiguity`.
# Renamed (every key kept unless said): `fw.nlu.intent` -> `fw.intent` (v14; `stage` -> `reads_reply` +
# `question_kind`, `engine` -> `decided_by`: name, fuzzy, foreign_name, local, remote, question, reply);
# `fw.nlu.param_extraction` -> `fw.parameters` (v16; + `reads_reply`); `fw.ask_user` -> `fw.question` (v3; every wait
# for a person, `kind` which_command / parameters / review / approval / decision / agent, `part`, `agent_query` ->
# `asked`; under the resolve part, the review or the run part it asks for); `fw.planner.plan` and `fw.planner.replan`
# -> `fw.plan` (v11; under the run part it plans for, `replan_trigger` -> `trigger` with `initial`, `skill*` ->
# `command*`); `fw.skill.phase` -> `fw.phase` (v8; `skill` -> `command`, + the agent loop's `attempts` and
# `clarification`); `fw.agent.step` with its `fw.agent.tool_call`, and `fw.skill.answer` -> `fw.step` (v9; `run_by`
# code / agent / model, `kind` command / ask / finish / tool / answer, `plan_span_id`, the dispatch's facts; `by_code`
# gone); `fw.command.execute` -> `fw.command` (v6; + `resolved_by`, `parameters_read_by`, `parameters_valid`).
# Removed: `fw.agent.execute` (its facts are on the run part or the phase) and `fw.agent.tool_call` (on the step; its
# command is the step's child). A command's `fw.intent` / `fw.parameters` that decided nothing (its exact name, a typed
# call read in code, no model call, nothing missing or refused), on any path, is no longer written: the command's
# span says it. `fw.llm.call` (v3) always has `role` (`JEV` for the remote decision model) and may have `purpose`
# (`conversation_summary`, under `fw.answer`). `fw.compose` hangs under `fw.answer` (answer card) or the `fw.question`
# it composes (question form). The parents of the spans a pooled worker opens were corrected before this bump (each
# worker has its own parenting stack, `traced_call`): part N's spans no longer nest under part N-1's. No legacy path
# (owner decision): the framework has no old-to-new name map, its old name constants are removed, and a store written
# before contract 63 is not read by the framework's tools (the diagnosis, the comparisons, the debug page, the replay
# and distillation readers read the names above only). The support API answers the new names only: `SPAN_CONTRACTS`
# has no entry for an old name, and the store's `spans_for_turns(keys, name=)` / `count_spans_without_turn(name=)`
# match the name given, so a caller outside the framework that asks for `fw.command.execute` (a shared-utterance
# harvest) must ask for `SPAN_COMMAND` (`fw.command`).
SPAN_CONTRACT_VERSION = 63

# The turn's root span, and one span per model call (written at the DSPy callback level).
SPAN_TURN = "fw.turn"
SPAN_LLM_CALL = "fw.llm.call"
# A reserved prefix with no emitter.
SPAN_TRAIN_PREFIX = "fw.train."

# ----------------------------------------------------------------------
# Recorded passes within one turn (fix-txxy)
# ----------------------------------------------------------------------
#
# Distillation runs the agent twice for ONE user message — a teacher pass and a
# student pass — inside one turn, so both passes' dispatches, planner calls and
# LLM calls share one trace id. Nothing recorded which was whose: a reader
# holding the trace could only guess from ordering and model names, and
# `comparison.PassSelector` (which resolves pass membership ONLY from recorded
# spans) had nothing to resolve against.
#
# One span per pass answers both halves of that. The stamp is on the span, so a
# selector keyed on `fw.pass` is evidence rather than an assertion; and because
# the pass span is on the parenting stack for the duration of the pass,
# everything the pass did is in its subtree and inherits the stamp through
# ancestry — no emitter downstream has to learn about passes.
#
# Activity that belongs to NEITHER pass stays outside the span by construction:
# insight extraction runs after both passes have closed, so its LLM calls parent
# to the turn root and no pass's cost roll-up can claim them.
SPAN_DISTILLATION_PASS = "fw.distillation.pass"

# One span per approval-binding transition (approval_binding._audit).
SPAN_APPROVAL = "fw.approval"

# One span per compose call of the run_chatbot page (the answer card and the question form). Written by the
# server after the turn finished; see ``fastworkflow.run_fastapi_mcp.compose_trace``.
SPAN_COMPOSE = "fw.compose"

# The turn's stages (bd fix-qjne.34, design section 4): one span per stage and per part of a stage, each with a
# deterministic id (``stage_span_id``), so a turn that waits for the user keeps ONE span per stage and part across the
# wait and across processes. A stage span's parent is the turn root, a part span's its stage's span. Opened by
# ``enter_stage`` and ``part_span`` below; closed through the host's rebuilders (``close_stage_span``).
SPAN_SPLIT = "fw.split"
SPAN_RESOLVE = "fw.resolve"
SPAN_REVIEW = "fw.review"
SPAN_RUN = "fw.run"
SPAN_ANSWER = "fw.answer"
SPAN_RESOLVE_PART = "fw.resolve.part"
SPAN_RUN_PART = "fw.run.part"
STAGE_SPAN_NAMES = (SPAN_SPLIT, SPAN_RESOLVE, SPAN_REVIEW, SPAN_RUN, SPAN_ANSWER)
#: The stages pushed on the parenting stack while open, so the spans their work opens are their children.
PUSHED_STAGES = frozenset(STAGE_SPAN_NAMES)
#: The stage a part span belongs to.
STAGE_OF_PART = {SPAN_RESOLVE_PART: SPAN_RESOLVE, SPAN_RUN_PART: SPAN_RUN}

# The split stage's own steps: the one-job gate, the splitter's call, the merge of its list, a part's referent.
SPAN_SPLIT_GATE = "fw.split.gate"
SPAN_SPLIT_SPLITTER = "fw.split.splitter"
SPAN_SPLIT_MERGE = "fw.split.merge"
SPAN_SPLIT_REFERENT = "fw.split.referent"

# A decision of which command (was fw.nlu.intent), a reading of a command's values (was fw.nlu.param_extraction), and
# every wait for a person (was fw.ask_user), with its ``kind``.
SPAN_INTENT = "fw.intent"
SPAN_PARAMETERS = "fw.parameters"
SPAN_QUESTION = "fw.question"

# The run's own records (bd fix-qjne.34 step 6): a phase of a planning command (was fw.skill.phase), one step of a run,
# whoever ran it (`run_by` code, agent or model; was fw.agent.step with its fw.agent.tool_call, and fw.skill.answer as
# `kind` answer), one command run (was fw.command.execute), and a plan the text planner made (was fw.planner.plan and
# fw.planner.replan). fw.agent.execute is gone: its facts are on the part or the phase it ran for.
SPAN_PHASE = "fw.phase"
SPAN_STEP = "fw.step"
SPAN_COMMAND = "fw.command"
SPAN_PLAN = "fw.plan"

# The review's own records (step 5): how a reply read, the free-text edit read, a revision of the reviewed items.
SPAN_REVIEW_REPLY = "fw.review.reply"
SPAN_REVIEW_EDIT = "fw.review.edit"
SPAN_REVIEW_REVISION = "fw.review.revision"

#: The spans of the work a run does, which are not recorded while the framework's own planning command (the turn's
#: review) runs: the review is recorded as the ``fw.review`` stage and its own spans, never as a phase, a step or a
#: command (``quiet``).
WORK_SPAN_NAMES = frozenset({SPAN_PHASE, SPAN_STEP, SPAN_COMMAND, SPAN_INTENT, SPAN_PARAMETERS, SPAN_PLAN})

# The attribute a pass-stamping producer records, and the one
# `comparison.discover_pass_selectors` / `selection_api.DEFAULT_PASS_ATTRIBUTE`
# already look for. Dotted like a span name rather than named `pass` because it
# is a cross-emitter label, not one emitter's field.
ATTR_PASS = "fw.pass"

PASS_TEACHER = "teacher"
PASS_STUDENT = "student"

# A pass whose outcome the producer did not record. Deliberately NOT a
# `TurnStatus` member: "completed" is a claim, and a pass that ended in a way
# the producer could not classify has not made it. Defaulting an unclassified
# outcome to completed is how a failure becomes invisible in a roll-up.
PASS_STATUS_UNKNOWN = "unknown"

# ----------------------------------------------------------------------
# Per-emitter attribute contracts (arch §12.0 delta 5, FW-REQ-019 clause 3)
# ----------------------------------------------------------------------
#
# `SPAN_CONTRACT_VERSION` above says "something in the taxonomy changed". This
# table says what each emitter promises to write, and versions that promise
# separately, so a reader that notices the aggregate moved can find out which
# emitter moved and which of its keys it should stop expecting.
#
# `attributes` is not documentation. tests/test_span_contract_versioning.py
# recovers the key set every emission site actually writes — by AST, including
# keys added to an attribute bag several frames away — and fails when the two
# disagree. So adding a key to an emitter forces an edit here, which puts the
# version right under the cursor. That is as close to "you cannot add a key
# without deciding about the version" as a static check gets; the hand-written
# list it replaces would have gone stale the first time a bag gained a key
# somewhere other than the emission site.
#
# The version numbers all start at 1 on purpose. This counter begins here, and
# back-dating it would invent per-emitter history that was never recorded on any
# span — a span with no `span_contract_version` attribute at all is exactly the
# "recorded before per-emitter versioning" case, and that is already legible.
# Bump one when its emitter's keys, or their meanings, change.


@dataclass(frozen=True)
class SpanContract:
    """One emitter's attribute promise, and the version of that promise."""

    version: int
    attributes: frozenset[str]


SPAN_CONTRACTS: dict[str, SpanContract] = {
    # v2: `unattended` (the turn ran with no ask_user tool).
    # v3: `subtask_review` (SPAN_CONTRACT_VERSION v35): the sub-task review gate's outcome for the turn.
    # v4: `subtask_bindings` (SPAN_CONTRACT_VERSION v37): shown == run per bound sub-task (fix-48np.1).
    # v5: `subtask_plan_ambiguity` (SPAN_CONTRACT_VERSION v46): the sub-tasks whose planning-command match was ambiguous.
    # v6: `subtask_split` (SPAN_CONTRACT_VERSION v56): the model's split, the merged list and the rules that changed it.
    # v7: `subtask_split.failed` (SPAN_CONTRACT_VERSION v59): why a split gave no answer.
    # v8: `subtask_split.prompt_digest` and `.prompt_source` (SPAN_CONTRACT_VERSION v61): which splitter prompt split.
    # v9: `subtask_split.prompt_source` value `built_in+workflow` (SPAN_CONTRACT_VERSION v62): split_examples as demos,
    #     and `workflow` (a replacement prompt) is no longer written;
    #     `subtask_split.gate`: the one-job gate's verdict, and a split the gate skipped (`attempts` 0).
    # v10 (SPAN_CONTRACT_VERSION v63): `subtask_split` moved to `fw.split`, `subtask_review` to `fw.review`'s
    #     `outcome`, `subtask_plan_ambiguity` to `fw.resolve`'s `plan_ambiguity`; new `path` (`plan` or `direct`)
    #     and `mode` (`agent` or `no_agent`).
    SPAN_TURN: SpanContract(
        version=10,
        attributes=frozenset(
            {
                "turn_key",
                "channel_id",
                "conversation_id",
                "user_message",
                "unattended",
                "status",
                "success",
                "failure_reason",
                "suspended_ms",
                "context_mutations",
                "subtask_bindings",
                "path",
                "mode",
            }
        ),
    ),
    # v1: one per compose call (SPAN_CONTRACT_VERSION v43). Counts, reason codes and short labels only.
    # v2: `answer_chars` and `excerpt_chars` (SPAN_CONTRACT_VERSION v45): the answer's length and the excerpt's.
    SPAN_COMPOSE: SpanContract(
        version=2,
        attributes=frozenset(
            {
                "compose_kind",
                "outcome",
                "results",
                "model",
                "duration_ms",
                "turn_attached",
                "digest_chars",
                "answer_chars",
                "excerpt_chars",
                "collapsed",
                "sections_kept",
                "sections_proposed",
                "blocks_kept",
                "blocks_proposed",
                "lead",
                "drops",
                "retry",
                "sections_unnested",
                "refs_repaired",
                "sections_merged",
                "truncated",
                "token_limit",
                "second_pass",
            }
        ),
    ),
    # v1: one per approval-binding transition (ADR-EXP-025 section 12).
    SPAN_APPROVAL: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "approval_record_id",
                "approval_command",
                "approval_annotation",
                "approval_state_before",
                "approval_state_after",
                "approval_reason",
                "approval_outcome",
                "approval_fingerprint",
                "approval_rendering_hash",
                "approval_channel",
                "approval_identity_class",
                "approval_identity_ref",
                "approval_surface",
                "approval_origin",
                "approval_mode",
                "approval_shadow",
            }
        ),
    ),
    # v2: `user_responses` (SPAN_CONTRACT_VERSION v42): all replies of the question in order, the final one last.
    # v3 (SPAN_CONTRACT_VERSION v63) was `fw.ask_user`: every wait for a person, with its `kind` (which_command,
    # parameters, review, approval, decision, agent) and `part` (a resolve part's question); `agent_query` is `asked`.
    SPAN_QUESTION: SpanContract(
        version=3,
        attributes=frozenset(
            {"kind", "part", "asked", "attempt", "user_response", "user_responses", "human_wait_ms"}
        ),
    ),
    # The turn's stages and parts (bd fix-qjne.34 steps 2 to 4). `resumed` and `segments` on each: how many times the
    # span was opened (a turn that waits for the user reopens the same span when it resumes).
    # fw.split: the split record (was the turn's `subtask_split`): the request, the model's list (`raw`), the list the
    # turn runs (`merged`), the code rules that changed it (`rules`, was `changed_by`), `failed`, the model calls
    # (`attempts`), the splitter prompt's `prompt_digest` and `prompt_source`, and `parts` (how many).
    SPAN_SPLIT: SpanContract(
        version=1,
        attributes=frozenset(
            {"request", "raw", "merged", "rules", "failed", "attempts", "prompt_digest", "prompt_source", "parts",
             "resumed", "segments"}
        ),
    ),
    # The one-job gate's verdict (was `subtask_split.gate`): p(one), the answer, the bar, whether the splitter was skipped.
    SPAN_SPLIT_GATE: SpanContract(
        version=1, attributes=frozenset({"score", "label", "threshold", "skipped"})),
    # One model attempt of the splitter (the first, or its one retry): its number, the list it gave, why it gave none.
    SPAN_SPLIT_SPLITTER: SpanContract(
        version=1, attributes=frozenset({"attempt", "raw", "failed"})),
    # The merge of the model's list: the list after it, the rules that changed it, the rewrites it did not make.
    SPAN_SPLIT_MERGE: SpanContract(
        version=1, attributes=frozenset({"merged", "rules", "doubts"})),
    # A part's referent: the record it acts on without naming it, and the description that stands for it.
    SPAN_SPLIT_REFERENT: SpanContract(
        version=1, attributes=frozenset({"part", "subject", "description"})),
    # fw.resolve: how many parts, how many which-command questions were answered, and the parts whose planning-command
    # match was ambiguous (`plan_ambiguity`, was the turn's `subtask_plan_ambiguity`).
    SPAN_RESOLVE: SpanContract(
        version=1, attributes=frozenset({"parts", "asked", "plan_ambiguity", "resumed", "segments"})),
    # fw.resolve.part: one part's resolution: its text, `outcome` (bound, ad_hoc, dropped, asked), `means`
    # (planning_command, command, ad_hoc), the command, and `fast_path`: it runs with no review (the gate's fast path
    # took it, or it is the turn's one part and the user picked its command: the turn's review mark was `fast_path`).
    SPAN_RESOLVE_PART: SpanContract(
        version=1,
        attributes=frozenset(
            {"part", "request", "outcome", "means", "command", "fast_path", "resumed", "segments"}
        ),
    ),
    # fw.review: the review's outcome (was the turn's `subtask_review` while the review is out or after it).
    SPAN_REVIEW: SpanContract(
        version=1, attributes=frozenset({"outcome", "rounds", "items", "resumed", "segments"})),
    # (step 5) How one reply to the review read: `read_as` (accept, decline, edit, unclear, refused, wait, ...), `by`
    # (code, head: the review head's routing, model: the free-text edit reader), and whether only an explicit yes accepts
    # the turn (`explicit_yes_required`). Replaces the review router's fw.intent / fw.parameters.
    SPAN_REVIEW_REPLY: SpanContract(
        version=1, attributes=frozenset({"read_as", "by", "reason", "explicit_yes_required"})),
    # (step 5) The free-text edit reader's read of a reply: how many edits it read and kept, whether it read again to
    # repair them, the means it named that are no command here.
    SPAN_REVIEW_EDIT: SpanContract(
        version=1, attributes=frozenset({"edits_read", "edits_applied", "repaired", "refused_names"})),
    # (step 5) A revision of the reviewed items: its number, how many items changed, whether their checks ran again.
    SPAN_REVIEW_REVISION: SpanContract(
        version=1, attributes=frozenset({"round", "items_changed", "rechecked"})),
    # fw.run: how many parts the run has, and how many ran, were not run, or were refused (step 6).
    SPAN_RUN: SpanContract(
        version=1, attributes=frozenset({"parts", "ran", "not_run", "refused", "resumed", "segments"})),
    # (step 6) One part of the run: `means` (planning_command, command, ad_hoc), the `command`, `outcome` (ran, not_run,
    # refused), who answered it (`answered_by`: code, agent), `reason` and `open_step` of a part not run or refused, and
    # the agent loop's facts when an agent ran it (were fw.agent.execute's): `agent_input`, `model`, `attempts`,
    # `final_answer`, `exhausted`, `error_type`; `decided_by` caller for a direct action.
    SPAN_RUN_PART: SpanContract(
        version=1,
        attributes=frozenset(
            {"part", "means", "command", "outcome", "answered_by", "reason", "open_step", "decided_by",
             "agent_input", "model", "attempts", "final_answer", "exhausted", "error_type", "resumed", "segments"}
        ),
    ),
    # (step 7) The turn's answer: who wrote it (`written_by` code, agent), its length (`chars`), and whether the
    # conversation summary failed (`summary_failed`); the summary's model call is its child.
    SPAN_ANSWER: SpanContract(
        version=1, attributes=frozenset({"written_by", "chars", "summary_failed", "resumed", "segments"})),
    # v6 (SPAN_CONTRACT_VERSION v63) One command run (was fw.command.execute, v5 keys): the command text, its parameters, its
    # response and success, its call ids and contexts, and how it was resolved: `resolved_by` (exact_name, fuzzy, model,
    # caller for a direct action, or the deciding layer), `parameters_read_by` (typed, none, model, or the extraction's
    # method) and `parameters_valid`. For a command a step dispatched, its fw.intent / fw.parameters are written only when they decided
    # something (a layer other than an exact name, a model call, missing or invalid values, an error).
    SPAN_COMMAND: SpanContract(
        version=6,
        attributes=frozenset(
            {
                "raw_command",
                "parameters",
                "response_text",
                "success",
                "error_type",
                "resolved_by",
                "parameters_read_by",
                "parameters_valid",
                ATTR_COMMAND_CALL_ID,
                ATTR_PARENT_CALL_ID,
                ATTR_CHILD_CALLS,
                ATTR_CONTEXT_BEFORE,
                ATTR_CONTEXT_AFTER,
            }
        ),
    ),
    # v8 (SPAN_CONTRACT_VERSION v63) One phase run of a planning command (was fw.skill.phase, v7 keys; `skill` is
    # `command`): the phase,
    # its index of the call's phases, its step budget and first step, `resumed` after a wait; closed with its numbered
    # `steps`, `steps_used`, `steps_by_code` and `code_stop`, `exhausted` or `suspended`, the `record` and `facts` it
    # filled (`facts_error` when it failed), `answer_only` / `answer_by_code` / `uses_result` / `skipped`, the fan-out
    # item keys, `stop_condition_met`, `reextracted`, `error_type`; and the agent loop's `attempts` and `clarification`
    # (was its fw.agent.execute child's).
    SPAN_PHASE: SpanContract(
        version=8,
        attributes=frozenset(
            {
                "answer_only", "answer_by_code", "uses_result", "skipped", "stop_condition_met", "steps_by_code",
                "code_stop", "item", "item_index", "item_count", "items_total", "record_issues", "command", "phase",
                "phase_index", "phase_count", "step_budget", "first_step", "resumed", "reextracted", "steps",
                "steps_used", "exhausted", "suspended", "record", "facts", "facts_error", "error_type", "attempts",
                "clarification",
            }
        ),
    ),
    # v9 (SPAN_CONTRACT_VERSION v63) One step of a run (was fw.agent.step with its fw.agent.tool_call, and
    # fw.skill.answer): `run_by` code,
    # agent (the agent model chose it; its model call is the step's child) or model (a planning command's deferred
    # answer); `kind` command, ask, finish, tool or answer; `plan_span_id` the fw.plan it carries out, when a plan was
    # made for its part; the agent step's v8 keys (minus `by_code`); the command it dispatched (`raw_command`,
    # `response_text`, `success`, its call id and contexts, was the tool call's); for an answer step the `command` and
    # `phase` whose facts it writes from. Every opener writes the one record (``step_record``).
    SPAN_STEP: SpanContract(
        version=9,
        attributes=frozenset(
            {
                "step_index", "run_by", "kind", "plan_span_id", "thought", "tool_name", "tool_args", "observation",
                "clarification", "recovered", "tool_error", "error_type", "repaired_tool_name", "folded_command",
                "policy_mode", "policy_outcome", "policy_source", "policy_table_version", "policy_intent",
                "policy_requests", "policy_judgement", "answer", "raw_command", "response_text", "success",
                ATTR_COMMAND_CALL_ID, ATTR_CONTEXT_BEFORE, ATTR_CONTEXT_AFTER, "command", "phase", "phase_index",
                "phase_count",
            }
        ),
    ),
    # v11 (SPAN_CONTRACT_VERSION v63) A plan of the text planner (was fw.planner.plan and fw.planner.replan, v10 keys), a child of the part it
    # plans for: `trigger` (initial, or what planned again: ask_user_response, parameter_extraction_error, ...; was
    # `replan_trigger`), `command`, `command_args`, `command_notes`, `command_fallback`, `command_refusal_asked` (were
    # `skill*`). The steps it led to are its siblings and carry its id (`plan_span_id`).
    SPAN_PLAN: SpanContract(
        version=11,
        attributes=frozenset({"model", "trigger", "plan", "plan_source",
                              "command", "command_args", "command_notes", "command_fallback",
                              "command_refusal_asked", "argument_check_not_in_request"}),
    ),
    # One distillation pass. `fw.pass` is the membership stamp every descendant
    # inherits through ancestry; the rest is the pass's OWN content, recorded
    # here because the turn row cannot hold it -- two passes share one turn row,
    # so its answer and status are the turn's and belong to neither pass.
    #
    # `answer` is the pass's final answer as the user would have seen it, and it
    # is recorded by the producer that generated it rather than copied from the
    # turn: the whole point of a teacher/student view is that the two answers
    # differ, and reporting the turn's answer under both headings would hide
    # exactly the difference the view is read for.
    SPAN_DISTILLATION_PASS: SpanContract(
        version=1,
        attributes=frozenset(
            {ATTR_PASS, "model", "answer", "plan", "status", "failure_reason"}
        ),
    ),
    # v2: the known-name refusal keys. The emitter wrote the three
    # `known_name_*` keys before they were declared here.
    # v3: `auto_navigation_enabled` is absent. Auto-navigation was removed, so
    # there is no flag and no composed step for a measured run to carry.
    # v4 (fix-jd9o.1): `jev_tiebreak` records Jev's choice among an ambiguous
    # intent's candidates ({choice, probability, threshold, candidates when it
    # routed}); `matcher_layer` is then "jev_tiebreak". Absent on earlier versions.
    # v5 (fix-yu3l.2): `engine` and `engine_fallback` name the detector engine behind
    # the classifier layer's answer. Absent when another layer decided, and on earlier versions.
    # v6 (fix-yu3l.3): `matcher` carries the model matcher's probabilities when it decided
    # (`matcher_layer` "matcher"). Absent otherwise, and on earlier versions.
    # v7: `engine_mode` names the INTENT_DETECTION_ENGINE mode in force. Absent on earlier versions.
    # v8 (fix-o5wi.2): one detection chain. `engine_mode` and `jev_tiebreak` are no longer written;
    # `engine` is "classifier", "matcher" or "question"; `engine_chain` lists the links asked.
    # v9 (fix-o5wi.8): the similarity cache is gone, so `cache_similarity` and `cache_similarity_threshold` are
    # no longer written and `matcher_layer` is never "embedding_cache". A span recorded before this may carry them.
    # v10 (fix-o5wi.9): `chosen_command_elsewhere`; the local model's facts carry `outside_labels`, `outside_scores`
    # and `handed_on` (see the v52 note above).
    # v11 (SPAN_CONTRACT_VERSION v53): `remote_failed`, the remote model's failed call (see the v53 note above).
    # v12 (SPAN_CONTRACT_VERSION v58): `part`, the 1-based part of the turn a plan-path decision is for (see the v58 note).
    # v14 (SPAN_CONTRACT_VERSION v63) was `fw.nlu.intent`: `stage` is `reads_reply` (the span reads the reply to a which-command
    # question) and `question_kind` (`ambiguity` or `misunderstanding` when it does); `engine` is `decided_by` (`name`,
    # `fuzzy`, `foreign_name`, `local`, `remote`, `question`, `reply`).
    SPAN_INTENT: SpanContract(
        version=14,
        attributes=frozenset(
            {
                "context",
                "reads_reply",
                "question_kind",
                "utterance",
                "matcher_layer",
                "known_name_foreign_context",
                "known_name_owner_contexts",
                "known_name_foreign_context_hint",
                "chosen_command_elsewhere",
                "escalation_outcome",
                "fuzzy_distance",
                "fuzzy_threshold",
                "fuzzy_prematch_tie",
                "candidate_count",
                "classifier",
                "classifier_signal_version",
                "decided_by",
                "engine_fallback",
                "engine_chain",
                "matcher",
                "remote_failed",
                "candidates",
                "escalation_labels_discarded",
                "reducible",
                "decision_uncertainty",
                "command_name",
                "is_cme_command",
                "ambiguous",
                "resolved",
                "part",
            }
        ),
    ),
    SPAN_PARAMETERS: SpanContract(
        # v16 (SPAN_CONTRACT_VERSION v63) was `fw.nlu.param_extraction`; `reads_reply`: the extraction reads the
        # reply to a parameter question (`retry_round`).
        # v2 (fix-8ko2): `retry_round_ordinal` joins the boolean `retry_round`.
        # The flag says an extraction resumed from stored parameters; the
        # ordinal says which attempt it was, which a consumer previously had to
        # guess by counting spans. A span carrying v1 has the flag only, and a
        # reader must keep treating its round as unrecorded rather than
        # inferring one.
        # v3 (fix-7t0q): `unwrapped_bracketed_values` and `placeholder_values`
        # record a value sent as one bracketed token: unwrapped to the real value
        # it held, or refused as a placeholder. Absent on earlier versions.
        # v4 (fix-7t0q): `unwrapped_quoted_values` records a value whose
        # surrounding quotes were removed; `defaulted_sentinel_values` a made-up
        # extreme number (int32/int64 limit, beyond 2**53) replaced by the
        # field's default; `out_of_range_values` a number refused as a parameter
        # error (declared ge/le bounds, a negative position or count, an extreme
        # for a field with no default). `placeholder_values` may now also hold a
        # placeholder-shaped value with no brackets that the request never
        # writes (`desired-tag`). Absent on earlier versions.
        # v5 (fix-hia1.15, fix-hia1.14): `deterministic_form` says which code
        # parser bound the parameters when `extraction_method` is "xml_regex"
        # ("key_value", "positional" or, since span contract v21, "bare": nothing
        # after the command name and every field defaulted; absent for XML tags), and
        # `unbound_value` is a lone value that bound to no field and was
        # refused. Absent on earlier versions.
        # v6 (command-name guard): `command_name_values` records a parameter value
        # that was the name of a command (the called command's own, or another's in
        # identifier form) and was treated as not given. Absent on earlier versions.
        # v7 (word screens): `word_screen_values` records a free-text value that was
        # only a pronoun or only a kind word or field name of the workflow and was
        # treated as not given. Absent on earlier versions.
        # v8 (fix-3rsq.one-hook): `db_lookup` is no longer written (the lookup moved into the
        # validate_extracted_parameters hook, and db_lookup is a load error). `validation_hook`
        # now also carries `changed_fields` (fields the hook rewrote in place) and `refused_fields`
        # (the fields a fastworkflow.Refused named). A span carrying v7 has `db_lookup` and no
        # `validation_hook` change list, and a reader must not read one as the other.
        # v9 (fix-jd9o.3): `fields_reset_not_given` ({field: probability}) records the fields an LLM
        # extraction filled that Jev judged the request does not give, reset to their not-given
        # value. Absent on earlier versions.
        # v11 (spec 3.7): `validation_hook` is the Input's model validators' record (the Signature hook is
        # removed). The keys and their meanings are the same; the producer changed. A span carrying v10
        # recorded the hook's verdict under the same key.
        # v10 (fix-qjf4): `argument_check_refusals`, `argument_check_retry`,
        # `argument_check_refusals_final` and `argument_check_respelled` record the argument checks of
        # an LLM extraction (fastworkflow.utils.argument_checks): the refused values ("name='value'
        # why"), the one retry with the refusal appended, what was still refused after it (treated as
        # not given), and {field: value} for a value that took the request's or the declared spelling.
        # Absent on earlier versions.
        # v12 (SPAN_CONTRACT_VERSION v55): `argument_refusal` (a refusal of the call as a whole that names no input;
        # written by an earlier step of fix-qoyz, declared here). Absent on earlier versions.
        # v13 (SPAN_CONTRACT_VERSION v57): `argument_check_not_in_request` (see the v57 note). Absent on earlier versions.
        # v14 (SPAN_CONTRACT_VERSION v58): `part`, the 1-based part of the turn a plan-path extraction is for (see the v58 note).
        # v15 (SPAN_CONTRACT_VERSION v60): `argument_left_empty` (see the v60 note). Absent on earlier versions.
        version=16,
        attributes=frozenset(
            {
                "command_name",
                "deterministic_form",
                "unbound_value",
                "retry_round",
                "retry_round_ordinal",
                "unwrapped_bracketed_values",
                "unwrapped_quoted_values",
                "placeholder_values",
                "defaulted_sentinel_values",
                "out_of_range_values",
                "extraction_method",
                "missing_fields",
                "invalid_fields",
                "validation_hook",
                "decision_uncertainty",
                "parameters_valid",
                "command_name_values",
                "word_screen_values",
                "fields_reset_not_given",
                "argument_check_refusals",
                "argument_check_retry",
                "argument_check_refusals_final",
                "argument_check_respelled",
                "argument_refusal",
                "argument_check_not_in_request",
                "part",
                "argument_left_empty",
                "reads_reply",
            }
        ),
    ),
    # v3 (SPAN_CONTRACT_VERSION v63): `role` is always set (`JEV` for the remote decision model), and `purpose` says
    # what a call is for when its caller says (`conversation_summary`).
    SPAN_LLM_CALL: SpanContract(
        version=3,
        attributes=frozenset(
            {
                "module",
                "capture_source",
                "model",
                "messages",
                "prompt",
                "call_kwargs",
                "module_chain",
                "module_input",
                "module_output",
                "module_exception",
                "reasoning",
                "output",
                "usage",
                "cost",
                "history_uuid",
                "response_model",
                "cache_hit",
                "provider_response",
                "usage_capture",
                "exception",
                "role",
                "timeout_s",
                "num_retries",
                "outcome",
                "timed_out",
                "exception_type",
                # (bd fix-qjne.34 step 7) what the call is for, when its caller says (``model_purpose``):
                # `conversation_summary` for the turn's summary.
                "purpose",
            }
        ),
    ),
}

# `fw.train.*` is a reserved PREFIX with no emitter — nothing in the package opens
# a span under it — so it has no contract to version. It gets one when something
# emits one, and the conformance test is what will notice.

# Written by the emission machinery rather than by an emitter, so it is not part
# of any emitter's declared set: it describes the record, not the thing recorded.
ENVELOPE_ATTRIBUTES = frozenset({ATTR_SPAN_CONTRACT_VERSION, ATTR_FRAME_ALIAS})


def span_contract_versions() -> dict[str, int]:
    """`span name -> attribute-contract version`, for run provenance (§12.4)."""
    return {name: contract.version for name, contract in SPAN_CONTRACTS.items()}


# Span kinds (spans.kind column): internal | llm | human_wait | tool.
KIND_INTERNAL = "internal"
KIND_LLM = "llm"
KIND_HUMAN_WAIT = "human_wait"
KIND_TOOL = "tool"

# Span statuses. "open" means started and not yet ended; a store treats a
# re-emission of the same span_id as an idempotent upsert ([R2][R6]).
STATUS_OPEN = "open"
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_AWAITING_USER = "awaiting_user"

#: The cap on ONE span-attribute value written to the observability store, in
#: UTF-8 bytes. A constant: it bounds a database row, not a
#: model prompt, so it is not one of the context-window budgets in
#: ``fastworkflow.context_budget`` and does not scale with a model.
MAX_ATTR_BYTES = 16384
#: The pre-rename name, kept as an alias for readers of the provenance
#: record.
_DEFAULT_MAX_ATTR_BYTES = MAX_ATTR_BYTES


def is_control_signal(exc: BaseException) -> bool:
    """True for the BaseExceptions that mean "pause", not "fail".

    `CommandCancelledError` and `AskUserSuspend` both unwind a dispatch without
    anything having gone wrong: the first suspends a trajectory for ask_user
    resume, the second is raised by ask_user when no user_message_queue is
    configured. Both subclass BaseException precisely so ordinary
    `except Exception` handlers do not swallow them.

    Imports are deferred: both defining modules import this one.
    """
    from fastworkflow.utils.react import AskUserSuspend
    from fastworkflow.workflow_execution_context import CommandCancelledError

    return isinstance(exc, (AskUserSuspend, CommandCancelledError))


def status_for_dispatch_exception(exc: BaseException) -> str:
    """The span status for an exception that ended a command dispatch.

    ONE mapping in ONE place, because five sites used to each carry their own
    isinstance check and they did not agree. The same
    AskUserSuspend closed as `error` in three of them, `awaiting_user` in a
    fourth, and escaped a fifth without closing its span at all — so an
    ordinary pause for input drew as a red ERROR node in the chatbot waterfall
    (`renderWaterfall`, run_chatbot/static/src/220-turn-nav.js), and what a
    reader saw depended on which layer happened
    to catch it.

    Control signals map to CANCELLED rather than AWAITING_USER deliberately.
    `awaiting_user` is a TURN-level state in this codebase: the store's
    non-terminal turn status, and the only thing the SPA tests it for
    (`statusBadge` in 210-turn-detail.js and the chat status lines in
    240-chat.js / 260-chat-live.js, all under run_chatbot/static/src, read the
    turn's status). Nothing anywhere reads a
    SPAN status of awaiting_user. A span status describes that span's own
    outcome — this dispatch was cut short — while "we are waiting on a human"
    is recorded once, on the turn, where readers already look for it.
    """
    return STATUS_CANCELLED if is_control_signal(exc) else STATUS_ERROR


@dataclass
class Span:
    """One OTel-shaped span record (spans table shape, design §3.2)."""

    span_id: str
    trace_id: str  # = turn_key
    name: str
    kind: str = KIND_INTERNAL
    parent_span_id: Optional[str] = None
    channel_id: Optional[str] = None
    command_name: Optional[str] = None
    context: Optional[str] = None
    start_ns: int = 0
    end_ns: Optional[int] = None
    status: str = STATUS_OPEN
    attributes: dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    claim_epoch: Optional[int] = None
    server_incarnation: Optional[str] = None
    #: ``sha256 -> text`` of the prompt pieces an over-cap ``messages`` was
    #: split into (``observability.prompt_slots``). Carried beside the
    #: attributes, never inside them, so a sink that persists the attribute
    #: bag does not persist the full prompt text with it.
    prompt_slots: Optional[dict[str, str]] = None


#: The attribute the prompt-slot enricher hands its piece texts over in.
#: ``_emit`` moves it out of the attributes onto ``Span.prompt_slots`` before
#: any sink sees the span.
ATTR_PROMPT_SLOT_TEXTS = "_fw_prompt_slot_texts"


# ----------------------------------------------------------------------
# Sink protocol
# ----------------------------------------------------------------------


@runtime_checkable
class TraceSink(Protocol):
    """Trace/record sink (design §3.1). Implementations must never raise to
    the caller; the emission helpers below additionally guard every call."""

    def emit_span(self, span: Span) -> None: ...

    def emit_turn_record(self, record: Any) -> None:
        """Receive the internal TurnResult at turn finalize (typed ``Any`` to
        keep this module import-light)."""
        ...

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None: ...  # [R15]


class NoOpTraceSink:
    """Default sink: tracing structurally present, nothing recorded."""

    def emit_span(self, span: Span) -> None:
        pass

    def emit_turn_record(self, record: Any) -> None:
        pass

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None:
        pass


# ----------------------------------------------------------------------
# Span identity
# ----------------------------------------------------------------------


def deterministic_span_id(turn_key: str, span_name: str, attempt: int = 0) -> str:
    """Deterministic span id for spans that must close in a different process
    than the one that opened them (fw.turn, fw.ask_user)."""
    digest = hashlib.sha256(f"{turn_key}|{span_name}|{attempt}".encode()).hexdigest()
    return digest[:32]


def root_span_id(turn_key: str) -> str:
    """The fw.turn root span id for a logical turn."""
    return deterministic_span_id(turn_key, SPAN_TURN, 0)


def cap_attr_value(value: Any) -> Any:
    """Cap one attribute value at ``MAX_ATTR_BYTES``.

    Truncation is lossy-and-counted: an over-limit string becomes an
    envelope carrying ``truncated: True``, the original byte length, and the
    sha256 of the original — never a silent prefix.
    """
    if not isinstance(value, str):
        return value
    limit = MAX_ATTR_BYTES
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return {
        "truncated": True,
        "original_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "value": raw[:limit].decode("utf-8", errors="ignore"),
    }


def split_stage_attributes(record: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The ``fw.split`` span's attributes from the split recorder's dict (``workflow_agent.recording_split``); every key
    None when the request was not split. Every text is capped like any attribute string (``cap_attr_value``). The gate's
    verdict and the merge's doubts are their own spans' (``fw.split.gate``, ``fw.split.merge``)."""
    record = record or {}
    merged = record.get("merged")
    return {
        "request": cap_attr_value(record["request"]) if record.get("request") is not None else None,
        "raw": [cap_attr_value(text) for text in record["raw"]] if record.get("raw") is not None else None,
        "merged": [cap_attr_value(text) for text in merged] if merged is not None else None,
        "rules": list(record.get("changed_by") or []) or None,
        "failed": record.get("failed"),
        "attempts": record.get("attempts"),
        "prompt_digest": record.get("prompt_digest"),
        "prompt_source": record.get("prompt_source"),
        "parts": len(merged) if merged is not None else None,
    }


def _capped(attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not attributes:
        return {}
    capped = {key: cap_attr_value(value) for key, value in attributes.items()}
    enrichment._apply(attributes, capped)
    return capped


# ----------------------------------------------------------------------
# Duck-typed host access (WEC directly, or ChatSession via _core)
# ----------------------------------------------------------------------


def _resolve(host: Any, attr: str) -> Any:
    value = getattr(host, attr, None)
    if value is not None:
        return value
    core = getattr(host, "_core", None)
    if core is not None:
        return getattr(core, attr, None)
    return None


def get_sink(host: Any) -> Optional[TraceSink]:
    sink = _resolve(host, "trace_sink")
    return sink if sink is not None and not isinstance(sink, NoOpTraceSink) else None


def get_turn_key(host: Any) -> Optional[str]:
    return _resolve(host, "current_turn_key")


def get_channel_id(host: Any) -> Optional[str]:
    return _resolve(host, "observability_channel_id")


def get_experiment_claim(host: Any) -> dict[str, Any]:
    return _resolve(host, "observability_experiment_claim") or {}


def _host_stack(host: Any) -> Optional[list]:
    return _resolve(host, "trace_span_stack") if host is not None else None


def _get_stack(host: Any) -> Optional[list]:
    """The parenting stack of *host* in this thread: a pooled call's own copy (``traced_call``), else the host's."""
    stack = _host_stack(host)
    forked = _forked_stack.get()
    if forked is not None and stack is not None and forked[0] is stack:
        return forked[1]
    return stack


def get_frame_alias(host: Any) -> str:
    """The alias of the planning frame on top of *host*; "" for the turn frame."""
    alias = _resolve(host, "current_frame_alias")
    return alias if isinstance(alias, str) else ""


# ----------------------------------------------------------------------
# Emission helpers — never raise to the caller
# ----------------------------------------------------------------------


def _emit(sink: TraceSink, span: Span) -> None:
    # Stamped here rather than at each emission site: this is the single funnel
    # every span passes through on its way to a sink — including the ones
    # reconstructed as a bare `Span` to close across a process boundary — so no
    # emitter can forget, and a span read back out of the store describes its own
    # attribute contract without a lookup table. An unregistered name (a future
    # fw.train.* span) is left unstamped rather than given a made-up version.
    contract = SPAN_CONTRACTS.get(span.name)
    if contract is not None:
        span.attributes[ATTR_SPAN_CONTRACT_VERSION] = contract.version
    slot_texts = span.attributes.pop(ATTR_PROMPT_SLOT_TEXTS, None)
    if isinstance(slot_texts, dict) and slot_texts:
        span.prompt_slots = slot_texts
    try:
        sink.emit_span(span)
    except Exception as exc:  # a broken sink must never fail a turn
        logger.warning(f"TraceSink.emit_span failed (span {span.name}): {exc!r}")


def datetime_to_ns(value: Any) -> Optional[int]:
    """Epoch nanoseconds for a datetime, or None when absent/unparseable."""
    try:
        return int(value.timestamp() * 1_000_000_000) if value is not None else None
    except Exception:
        return None


def start_span(
    host: Any,
    name: str,
    *,
    kind: str = KIND_INTERNAL,
    attributes: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    emit_open: bool = False,
    use_stack: bool = True,
) -> Optional[Span]:
    """Open a span under the host's current turn; returns None when there is
    no active sink or no open turn. Never raises.

    Short-lived spans are emitted once, at ``end_span``; pass
    ``emit_open=True`` for long-lived spans (fw.turn, fw.ask_user) whose open
    event must be visible before — and closable after — a suspension.
    ``use_stack=False`` keeps a span off the parenting stack (fw.turn and
    fw.ask_user: children reach the root via its deterministic id, which
    survives suspension where the in-memory stack does not).
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        turn_key = get_turn_key(host)
        if not turn_key:
            return None

        if name in WORK_SPAN_NAMES and _resolve(host, "trace_quiet"):
            return None     # the framework's own planning command runs: its work is recorded as its stage's (``quiet``)

        stack = _get_stack(host)
        if parent_span_id is None:
            if stack:
                parent_span_id = stack[-1].span_id
                stack[-1]._fw_has_children = True       # read by ``had_children`` (a trivial span is discarded)
            elif name != SPAN_TURN:
                parent_span_id = root_span_id(turn_key)

        claim = get_experiment_claim(host)
        span = Span(
            span_id=span_id or uuid.uuid4().hex,
            trace_id=turn_key,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            channel_id=get_channel_id(host),
            command_name=command_name,
            context=context,
            start_ns=time.time_ns(),
            status=STATUS_OPEN,
            attributes=_capped(attributes) | {ATTR_FRAME_ALIAS: get_frame_alias(host)},
            experiment_id=claim.get("experiment_id"),
            task_id=claim.get("task_id"),
            attempt=claim.get("attempt"),
            claim_epoch=claim.get("epoch"),
            server_incarnation=claim.get("server_incarnation"),
        )

        if use_stack and stack is not None:
            stack.append(span)

        if emit_open:
            _emit(sink, span)
        return span
    except Exception as exc:
        logger.warning(f"start_span({name}) failed: {exc!r}")
        return None


def end_span(
    host: Any,
    span: Optional[Span],
    *,
    status: str = STATUS_OK,
    attributes: Optional[dict[str, Any]] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    close: bool = True,
) -> None:
    """Finish (or, with ``close=False``, update-in-place) a span and emit it.
    Never raises. A None span (start_span declined) is a silent no-op."""
    if span is None:
        return
    try:
        if close:
            span.end_ns = time.time_ns()
        span.status = status
        if attributes:
            span.attributes.update(_capped(attributes))
        if command_name is not None:
            span.command_name = command_name
        if context is not None:
            span.context = context

        stack = _get_stack(host)
        if stack is not None and span in stack:
            stack.remove(span)

        sink = get_sink(host)
        if sink is not None:
            # A span rebuilt to close across a suspension was never opened here, so it
            # has no alias yet and takes the frame on top at close. A span from
            # start_span already has its own, and keeps it.
            span.attributes.setdefault(ATTR_FRAME_ALIAS, get_frame_alias(host))
            _emit(sink, span)
    except Exception as exc:
        logger.warning(f"end_span({span.name}) failed: {exc!r}")


def discard_span(host: Any, span: Optional[Span]) -> None:
    """Drop *span*, opened and never to be emitted: it leaves the parenting stack (bd fix-qjne.34 step 6: a per-command
    decision that decided nothing is not written; its summary is on the command's span, ``note``)."""
    if span is not None:
        _without(_get_stack(host), span)


def had_children(span: Optional[Span]) -> bool:
    """Whether a span was opened under *span* through the parenting stack (a model call, a nested span)."""
    return bool(getattr(span, "_fw_has_children", False))


def is_command_decision(host: Any, span: Optional[Span]) -> bool:
    """Whether *span* (a decision or a reading of values) is a command's own: it runs inside an open ``fw.command``
    whose facts bag takes its summary (``facts_for``), the command a step dispatched or the direct ("/") path's command
    (whose decision is filed under its part). Such a record is written only when it decided something (bd fix-qjne.34
    decision b); its facts are on the command's span (``resolved_by``, ``parameters_read_by``, ``parameters_valid``)."""
    if span is None or (_fact_bags.get() or {}).get(SPAN_COMMAND) is None:
        return False
    return any(each.name == SPAN_COMMAND for each in (_get_stack(host) or ()))


# Facts that code several frames below a span hands to it (bd fix-qjne.34 step 6): the command a step dispatched, how a
# command was resolved. The span's opener binds a bag for its name (``facts_for``) and writes what the bag holds when it
# closes; the code below fills it (``note``), and does nothing when no such span is open.
_fact_bags: contextvars.ContextVar = contextvars.ContextVar("fw_trace_fact_bags", default=None)


@contextlib.contextmanager
def facts_for(name: str) -> Iterator[dict]:
    """A bag for the facts of the span *name* the block runs under (the innermost one of that name)."""
    bag: dict = {}
    bags = dict(_fact_bags.get() or {})
    bags[name] = bag
    token = _fact_bags.set(bags)
    try:
        yield bag
    finally:
        _fact_bags.reset(token)


def note(name: str, **facts: Any) -> None:
    """Add *facts* to the bag of the innermost open span *name* (``facts_for``); nothing when there is none."""
    bag = (_fact_bags.get() or {}).get(name)
    if bag is not None:
        bag.update(facts)


# What a model call is for (step 7): its caller says (``model_purpose``), and the call's span carries it.
_purpose: contextvars.ContextVar = contextvars.ContextVar("fw_trace_model_purpose", default=None)


@contextlib.contextmanager
def model_purpose(purpose: str) -> Iterator[None]:
    """The model calls of the block are for *purpose* (their span's ``purpose``)."""
    token = _purpose.set(purpose)
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> Optional[str]:
    return _purpose.get()


# Whose model a call is (the span's `role`): the caller names its own model setting (``LLM_PLANNER``, ``JEV``, ...)
# where it makes the call (``model_role``, ``dspy_utils.role_context``), so the span says it whatever LM object the
# caller holds; an LM ``get_lm`` built carries the same name as its tag, read when no caller named one.
_role: contextvars.ContextVar = contextvars.ContextVar("fw_trace_model_role", default=None)


@contextlib.contextmanager
def model_role(role: str) -> Iterator[None]:
    """The model calls of the block are *role*'s (their span's ``role``): the model setting's name."""
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


def current_role() -> Optional[str]:
    return _role.get()


def step_record(*, step_index: Any = None, run_by: Any = None, kind: Any = None, plan_span_id: Any = None,
                thought: Any = None, tool_name: Any = None, tool_args: Any = None, raw_command: Any = None,
                response_text: Any = None, success: Any = None, command_call_id: Any = None,
                context_before: Any = None, context_after: Any = None, command: Any = None, phase: Any = None,
                phase_index: Any = None, phase_count: Any = None, error_type: Any = None) -> dict[str, Any]:
    """The ``fw.step`` record every opener of a step writes (a step the agent loop runs, a planning command's answer
    step), so one contract describes them: the given facts that are not None. The keys only a step's close writes are
    listed with None so that every opener declares the one key set (a later ``end_span`` adds those its step has)."""
    return {key: value for key, value in {
        "step_index": step_index, "run_by": run_by, "kind": kind, "plan_span_id": plan_span_id, "thought": thought,
        "tool_name": tool_name, "tool_args": tool_args, "observation": None, "clarification": None,
        "recovered": None, "tool_error": None, "error_type": error_type, "repaired_tool_name": None,
        "folded_command": None, "policy_mode": None, "policy_outcome": None, "policy_source": None,
        "policy_table_version": None, "policy_intent": None, "policy_requests": None, "policy_judgement": None,
        "answer": None, "raw_command": raw_command, "response_text": response_text, "success": success,
        ATTR_COMMAND_CALL_ID: command_call_id, ATTR_CONTEXT_BEFORE: context_before,
        ATTR_CONTEXT_AFTER: context_after, "command": command, "phase": phase, "phase_index": phase_index,
        "phase_count": phase_count,
    }.items() if value is not None}


# ----------------------------------------------------------------------
# The turn's stages and parts (bd fix-qjne.34, design section 4.2)
# ----------------------------------------------------------------------
#
# A stage span (``STAGE_SPAN_NAMES``) and a part span (``STAGE_OF_PART``) have a deterministic id, so the turn has ONE
# of each however many times it waits for the user and in however many processes it runs: a wait closes the open stage
# with the turn's status (``awaiting_user``) and the resumed segment opens the same id again (the store keeps the first
# start and takes the last close). Children reach them by id (the explicit parent) or, while they are open in this
# process, through the parenting stack: an open stage is pushed on the host's stack, and a part is pushed on the stack
# of the thread that works on it (a pooled worker's own, ``traced_call``) for as long as ``part_span`` runs.
#
# Opening is this module's (the spans carry no emitter attribute when they open); every close is a rebuild by the host
# (``_trace_close_stage(name, part, span, status)``, which calls ``close_stage_span`` with the attributes it computes
# from the turn's saved state), so a close in a later process does not lose an earlier segment's facts.


@dataclass
class TurnStages:
    """The stage bookkeeping of the host's turn: the open stage, the spans open in this process (by ``name:part``),
    how many times each was opened (saved with the turn), the stage that was open when the turn began to wait (opened
    again when it resumes), and the direct path's part (the "/" path has one part and no pool)."""

    current: Optional[str] = None
    spans: dict[str, Span] = field(default_factory=dict)
    segments: dict[str, int] = field(default_factory=dict)
    waiting_in: Optional[str] = None
    waiting_parts: list[str] = field(default_factory=list)
    direct_part: Optional[str] = None
    #: The part of the open stage its work runs for now (``enter_part``: the run's parts, one after another).
    current_part: Optional[str] = None
    #: The parts closed before their stage (``enter_part`` closes the one before), not closed again with it.
    closed: list[str] = field(default_factory=list)
    #: Facts of a part the code that ran it noted (``note_part``), written by the part's rebuilder.
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The id of the last plan made for the current part (``note_plan``), carried by the steps that follow it.
    last_plan: Optional[str] = None
    #: The number (1-based, as the turn's split numbered it) of each part a stage's work runs, by part span name, when
    #: they are not 1, 2, ...: the run leaves out the parts dropped while the turn was resolved (``number_parts``).
    numbers: dict[str, list[int]] = field(default_factory=dict)
    #: Whether the turn took the direct ("/") path (``open_direct_part``): its ``fw.turn`` says ``path`` direct.
    direct: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def to_state(self) -> dict[str, Any]:
        return {"segments": dict(self.segments), "waiting_in": self.waiting_in,
                "waiting_parts": list(self.waiting_parts), "current_part": self.current_part,
                "closed": list(self.closed), "facts": {key: dict(value) for key, value in self.facts.items()},
                "last_plan": self.last_plan, "numbers": {key: list(value) for key, value in self.numbers.items()},
                "direct": self.direct}

    @classmethod
    def from_state(cls, state: Any) -> "TurnStages":
        state = state if isinstance(state, dict) else {}
        return cls(segments={str(k): int(v) for k, v in (state.get("segments") or {}).items()},
                   waiting_in=state.get("waiting_in"), waiting_parts=list(state.get("waiting_parts") or []),
                   current_part=state.get("current_part"), closed=list(state.get("closed") or []),
                   facts={str(k): dict(v) for k, v in (state.get("facts") or {}).items() if isinstance(v, dict)},
                   last_plan=state.get("last_plan"),
                   numbers={str(k): [int(n) for n in v] for k, v in (state.get("numbers") or {}).items()
                            if isinstance(v, list)},
                   direct=bool(state.get("direct")))


def stage_span_id(turn_key: str, name: str, part: int = 0) -> str:
    """The deterministic id of a stage span (*part* 0) or of a part span."""
    return deterministic_span_id(turn_key, name, part)


def _stages(host: Any) -> Optional[TurnStages]:
    return _resolve(host, "trace_stages") if host is not None else None


def _stage_key(name: str, part: int) -> str:
    return f"{name}:{part}"


def _key_parts(key: str) -> tuple[str, int]:
    name, _, part = key.rpartition(":")
    return name, int(part)


def _without(stack: Optional[list], span: Span) -> None:
    """Remove *span* (the object, not an equal one) from *stack*: its last entry."""
    if stack is None:
        return
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is span:
            del stack[index]
            return


def _open_stage_span(host: Any, stages: TurnStages, name: str, part: int) -> Optional[Span]:
    """The open span of *name* (and *part*) in this process, opened (and counted) when it is not."""
    turn_key = get_turn_key(host)
    key = _stage_key(name, part)
    with stages.lock:
        if key in stages.spans:
            return stages.spans[key]
        parent = root_span_id(turn_key) if not part else stage_span_id(turn_key, STAGE_OF_PART[name])
        span = start_span(host, name, span_id=stage_span_id(turn_key, name, part), parent_span_id=parent,
                          attributes={"part": part} if part else None, use_stack=False, emit_open=True)
        if span is not None:
            stages.spans[key] = span
            stages.segments[key] = stages.segments.get(key, 0) + 1
        return span


def _live(host: Any) -> Optional[TurnStages]:
    """The host's stages when a turn is being recorded, else None."""
    stages = _stages(host)
    if stages is None or get_sink(host) is None or not get_turn_key(host):
        return None
    return stages


def enter_stage(host: Any, name: str, *, push: Optional[bool] = None) -> None:
    """Make *name* the turn's open stage: the stage open before it (and its parts) closes, and *name* opens with its
    deterministic id under the turn root, pushed on the host's stack when it is one of ``PUSHED_STAGES`` and *push* is
    not False (the "/" path passes False: its command spans keep their own parents). Nothing when it is already the open
    stage, or nothing is recorded."""
    push = name in PUSHED_STAGES if push is None else push
    stages = _live(host)
    if stages is None or (stages.current == name and _stage_key(name, 0) in stages.spans):
        return
    close_stages(host, STATUS_OK)
    span = _open_stage_span(host, stages, name, 0)
    stages.current = name
    stack = _host_stack(host)
    if push and span is not None and stack is not None:
        stack.append(span)


def resume_stage(host: Any) -> None:
    """Open again the stage the turn was waiting in, and the parts of it that were open (a reply resumes them); the
    part its work ran for goes back on the host's stack. Nothing when it was waiting in none."""
    stages = _live(host)
    if stages is not None and stages.waiting_in and stages.current is None:
        name, stages.waiting_in = stages.waiting_in, None
        parts, stages.waiting_parts = stages.waiting_parts, []
        enter_stage(host, name)
        for key in parts:
            span = _open_stage_span(host, stages, *_key_parts(key))
            stack = _host_stack(host)
            if key == stages.current_part and span is not None and stack is not None:
                stack.append(span)


def enter_part(host: Any, name: str, part: int) -> None:
    """Make part *part* (1-based) the part the open stage's work runs for: the part before it closes (its facts are
    complete), and this one opens with its deterministic id, on the host's stack until its stage closes or the next
    part is entered. Nothing when its stage is not the open stage or it is already the current part."""
    stages = _live(host)
    if stages is None or not part or stages.current != STAGE_OF_PART.get(name):
        return
    key = _stage_key(name, part)
    if stages.current_part == key and key in stages.spans:
        return
    if stages.current_part is not None and stages.current_part != key and stages.current_part in stages.spans:
        previous = stages.current_part
        _close_one(host, stages, *_key_parts(previous), STATUS_OK)
        stages.closed.append(previous)
    if key in stages.closed:
        stages.closed.remove(key)
    stages.current_part, stages.last_plan = key, None
    span = _open_stage_span(host, stages, name, part)
    stack = _host_stack(host)
    if span is not None and stack is not None:
        stack.append(span)


def note_part(host: Any, name: str, part: Optional[int] = None, **facts: Any) -> None:
    """Keep *facts* of part *part* of span *name* (the current part when *part* is None) for its rebuilder
    (``part_facts``); saved with the turn, so a close in a later process has them."""
    stages = _live(host)
    if stages is None:
        return
    key = _stage_key(name, part) if part else stages.current_part
    if key is None or _key_parts(key)[0] != name:
        return
    stages.facts.setdefault(key, {}).update(facts)


def part_facts(host: Any, name: str, part: int) -> dict[str, Any]:
    """The facts noted for part *part* of span *name* (``note_part``)."""
    stages = _stages(host)
    return dict(stages.facts.get(_stage_key(name, part)) or {}) if stages is not None else {}


def number_parts(host: Any, name: str, numbers: Optional[list[int]]) -> None:
    """The parts of span *name* the stage's work runs are, in order, the turn's parts *numbers* (None: 1, 2, ...)."""
    stages = _live(host)
    if stages is not None:
        if numbers:
            stages.numbers[name] = [int(number) for number in numbers]
        else:
            stages.numbers.pop(name, None)


def part_number(host: Any, name: str, index: int) -> int:
    """The number of the part of span *name* the stage's work runs as its *index*-th (0-based): the turn's own number
    for it (``number_parts``), else *index* + 1."""
    stages = _stages(host)
    numbers = stages.numbers.get(name) if stages is not None else None
    return numbers[index] if numbers and 0 <= index < len(numbers) else index + 1


def noted_parts(host: Any, name: str) -> list[int]:
    """The numbers of the parts of span *name* that have noted facts (``note_part``), opened or not, in order."""
    stages = _stages(host)
    keys = list(stages.facts) if stages is not None else []
    return sorted(part for key in keys for span_name, part in [_key_parts(key)] if span_name == name and part)


def part_numbers(host: Any, name: str) -> list[int]:
    """The numbers of the parts of span *name* opened in this turn (in any process), in order."""
    stages = _stages(host)
    keys = list(stages.segments) if stages is not None else []
    return sorted(part for key in keys for span_name, part in [_key_parts(key)] if span_name == name and part)


def current_part_id(host: Any) -> Optional[str]:
    """The id of the part the open stage's work runs for now (``enter_part``), or None."""
    stages = _live(host)
    if stages is None or stages.current_part is None or stages.current_part not in stages.spans:
        return None
    return stages.spans[stages.current_part].span_id


def note_plan(host: Any, span: Optional[Span]) -> None:
    """*span* (a ``fw.plan``) is the plan the current part's next steps carry out (``current_plan_id``)."""
    stages = _live(host)
    if stages is not None and span is not None:
        stages.last_plan = span.span_id


def current_plan_id(host: Any) -> Optional[str]:
    """The id of the last plan made for the current part, or None."""
    stages = _live(host)
    return stages.last_plan if stages is not None else None


@contextlib.contextmanager
def part_span(host: Any, name: str, part: Optional[int]) -> Iterator[Optional[Span]]:
    """The part span *name* of part *part* (1-based) open, and on this thread's stack for the block, when its stage is
    the open stage; else nothing (the block's spans parent as they would). The span stays open until its stage closes."""
    stages = _live(host)
    if stages is None or not part or stages.current != STAGE_OF_PART.get(name):
        yield None
        return
    span = _open_stage_span(host, stages, name, part)
    stack = _get_stack(host)
    if span is not None and stack is not None:
        stack.append(span)
    try:
        yield span
    finally:
        if span is not None:
            _without(stack, span)


def open_direct_part(host: Any, name: str) -> None:
    """The direct ("/") path's one part: part 1 of *name*, open, not pushed; its decisions find it by
    ``direct_part_id``."""
    stages = _live(host)
    if stages is None or stages.current != STAGE_OF_PART.get(name):
        return
    if _open_stage_span(host, stages, name, 1) is not None:
        stages.direct_part = _stage_key(name, 1)
        stages.direct = True


def turn_path(host: Any) -> Optional[str]:
    """The ``fw.turn``'s ``path``: ``direct`` when the turn took the direct ("/") path (a person's typed command, an
    action), ``plan`` when it went through the turn's stages otherwise, None when it opened no stage."""
    stages = _stages(host)
    if stages is None or not stages.segments:
        return None
    return "direct" if stages.direct else "plan"


def enter_direct_run(host: Any, command: Optional[str]) -> None:
    """The direct ("/") path bound its one command *command*: its resolve stage closes (its part bound to it), and the
    run opens with its one part (not pushed: the command's span opened before it, ``direct_part_id`` finds it).
    Nothing on any other path."""
    stages = _live(host)
    if stages is None or stages.direct_part is None or stages.current != SPAN_RESOLVE:
        return
    note_part(host, SPAN_RESOLVE_PART, 1, command=command)
    enter_stage(host, SPAN_RUN, push=False)
    open_direct_part(host, SPAN_RUN_PART)
    note_part(host, SPAN_RUN_PART, 1, direct=True, command=command)


def direct_part_id(host: Any) -> Optional[str]:
    """The id of the direct path's open part, when its stage is the open stage; else None (the caller's span then
    parents as it would)."""
    stages = _live(host)
    if stages is None or stages.direct_part is None or stages.direct_part not in stages.spans:
        return None
    name, part = _key_parts(stages.direct_part)
    if stages.current != STAGE_OF_PART.get(name):
        return None
    return stages.spans[stages.direct_part].span_id


def part_span_id(host: Any, name: str, part: int) -> Optional[str]:
    """The deterministic id of part *part*'s span *name* (whether open or not), or None with no turn."""
    turn_key = get_turn_key(host)
    return stage_span_id(turn_key, name, part) if turn_key else None


def open_stage_id(host: Any) -> Optional[str]:
    """The id of the turn's open stage span, or None."""
    stages = _live(host)
    if stages is None or stages.current is None:
        return None
    return stage_span_id(get_turn_key(host), stages.current)


def leave_stage(host: Any, name: str) -> None:
    """Close *name* (and its parts) when it is the open stage, leaving no stage open."""
    stages = _live(host)
    if stages is not None and stages.current == name:
        close_stages(host, STATUS_OK)


def close_stages(host: Any, status: str, *, waiting: bool = False) -> None:
    """Close the open stage and every part of it opened in this turn (in any process), each rebuilt by the host
    (``_trace_close_stage``) with *status*. *waiting*: the turn now waits for the user, and the stage opens again when
    it resumes (``resume_stage``)."""
    stages = _live(host)
    if stages is None or stages.current is None:
        return
    name, stages.current = stages.current, None
    stages.direct_part = None
    if waiting:
        stages.waiting_in = name
        stages.waiting_parts = [key for key in stages.spans if key != _stage_key(name, 0)]
    else:
        stages.current_part = None
    keys = [key for key in stages.segments if STAGE_OF_PART.get(_key_parts(key)[0]) == name
            and key not in stages.closed]
    keys.sort(key=lambda key: _key_parts(key)[1])
    for key in [*keys, _stage_key(name, 0)]:
        span_name, part = _key_parts(key)
        _close_one(host, stages, span_name, part, status)


def _close_one(host: Any, stages: TurnStages, name: str, part: int, status: str) -> None:
    span = stages.spans.pop(_stage_key(name, part), None)
    if span is None:
        turn_key = get_turn_key(host)
        claim = get_experiment_claim(host)
        parent = root_span_id(turn_key) if not part else stage_span_id(turn_key, STAGE_OF_PART[name])
        span = Span(span_id=stage_span_id(turn_key, name, part), trace_id=turn_key, name=name, parent_span_id=parent,
                    channel_id=get_channel_id(host), experiment_id=claim.get("experiment_id"),
                    task_id=claim.get("task_id"), attempt=claim.get("attempt"), claim_epoch=claim.get("epoch"),
                    server_incarnation=claim.get("server_incarnation"))
    _without(_host_stack(host), span)
    closer = _resolve(host, "_trace_close_stage")
    try:
        if callable(closer):
            closer(name, part, span, status)
        else:
            end_span(host, span, status=status)
    except Exception as exc:  # a close must never fail a turn
        logger.warning(f"closing the {name} span failed: {exc!r}")


def stage_segments(host: Any, name: str, part: int = 0) -> dict[str, Any]:
    """``resumed`` and ``segments`` of a stage or part span: how many times it was opened in this turn."""
    stages = _stages(host)
    count = (stages.segments.get(_stage_key(name, part), 0) if stages is not None else 0) or 1
    return {"resumed": count > 1, "segments": count}


def close_stage_span(host: Any, name: str, span: Span, *, status: str, attributes: dict[str, Any],
                     command_name: Optional[str] = None) -> None:
    """Close the stage or part span *span* (a ``name`` span) with the host's *attributes*: the host's rebuilders call
    this, one per span name, so each name's keys are written in one place."""
    if span.name != name:
        raise ValueError(f"close_stage_span({name}) was handed a {span.name} span")
    end_span(host, span, status=status, attributes=attributes, command_name=command_name)


# ----------------------------------------------------------------------
# Capture projection (arch §12.0 deltas 2 and 4; §6.6.1)
# ----------------------------------------------------------------------
#
# The helpers below are pure projections onto span attributes. Nothing in
# fastWorkflow reads what they return: that is EXP-003's no-control-flow-read
# exit criterion and architecture §17.3's stop condition, and it is asserted
# both structurally and behaviorally by tests/test_no_capture_control_flow.py.

def context_type(workflow: Any) -> Optional[str]:
    """The name of *workflow*'s active command context, or None. Never raises:
    a capture failure must not fail a turn."""
    try:
        name = getattr(workflow, "_current_command_context_name", None) if workflow else None
        return str(name) if name else None
    except Exception as exc:
        logger.warning(f"context type read failed: {exc!r}")
        return None


def context_before(span: Optional[Span], workflow: Any) -> Optional[str]:
    """The active context type before a command runs, or None.

    Gated on a span having actually opened, matching the attribute-prep rule at
    every emission site: with observability off this must cost nothing.
    """
    if span is None:
        return None
    return context_type(workflow)


def active_workflow(host: Any) -> Any:
    """The workflow whose command context a dispatch on *host* acts on, or None.

    Duck-typed and never raising, like the rest of this seam: it is called only to
    build capture attributes, and a host that cannot answer must degrade to an
    absent assessment rather than fail the command. The fallback to ``app_workflow``
    matters for a bare WorkflowExecutionContext, whose ``get_active_workflow``
    reads a ContextVar stack that is empty outside a dispatch.
    """
    try:
        getter = getattr(host, "get_active_workflow", None)
        workflow = getter() if callable(getter) else None
    except Exception:
        workflow = None
    return workflow if workflow is not None else _resolve(host, "app_workflow")


def capture_attributes(
    span: Optional[Span],
    command_output: Any,
    context_type_before: Optional[str],
    workflow: Any,
    *,
    command_name: Optional[str] = None,
) -> dict[str, Any]:
    """Call-id and context type before/after for one executed command.

    §12.1.1 requires the dispatch paths to capture the same things, so the
    projection lives here once. All three ``fw.agent.tool_call`` emitters call it
    (``workflow_agent`` and both ``WorkflowExecutionContext`` paths), which is
    what makes a single contract version for that span name true rather than
    merely declared.

    ``command_output`` and ``workflow`` are read duck-typed, never imported: this
    module is on the import path of the core runtime and may not reach the models.

    The call id is read off the CommandOutput rather than minted here — the
    dispatcher that ran the command already stamped it, and a second id would join
    nothing. ``command_name`` is accepted from callers on the direct-action path
    but no longer read: the only consumer was the consequence lookup.
    """
    if span is None:
        return {}
    return {
        ATTR_COMMAND_CALL_ID: getattr(command_output, "command_call_id", None),
        ATTR_CONTEXT_BEFORE: context_type_before,
        ATTR_CONTEXT_AFTER: context_type(workflow),
    }


def emit_turn_record(host: Any, record: Any) -> Optional[bool]:
    """Hand the finalized internal TurnResult to the sink. Never raises.

    Returns the sink's "stored" ack (Phase 7 ruling I1), or None when there is
    no sink at all. The three values are distinct on purpose: True and False
    both mean a durable store exists and say whether the row reached it, while
    None means nothing is recording turns, so a caller gating on durability has
    nothing to wait for and must not defer forever.
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        return bool(sink.emit_turn_record(record))
    except Exception as exc:
        logger.warning(f"TraceSink.emit_turn_record failed: {exc!r}")
        return False
