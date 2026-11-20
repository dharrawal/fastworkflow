from typing import Optional
import os

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import tracing
from fastworkflow.observability.decision_signals import (
    DecisionUncertainty,
    UncertaintySignal,
    ambiguity_set_size,
    classifier_confidence,
    classifier_topk_margin,
    fuzzy_distance,
)
from fastworkflow import intent_clarification, intent_detector, intent_loop
from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER
from fastworkflow.nlu_labels import is_escalation
from fastworkflow.train.artifact_versioning import VERSIONS_DIRNAME


# A low-confidence top-k prediction containing an escalation label remains an
# ambiguity: prompt with the local command candidates and log that the parent signal
# was discarded. This is product behaviour, not workflow configuration.


# What became of an escalation signal on this prediction. FW-REQ-021 clause 1 asks
# for a first-class escalation outcome rather than only the list of labels that were
# thrown away: "no escalation label was predicted" and "we never ran the classifier
# that could predict one" are different facts, and `escalation_labels_discarded`
# being absent cannot tell them apart.
#
# Only the classifier can produce an escalation label, so every other matching layer
# reports NOT_EVALUATED rather than ABSENT.
ESCALATION_OUTCOME_NOT_EVALUATED = "not_evaluated"
ESCALATION_OUTCOME_ABSENT = "absent"
# The escalation label won outright: `resolve_fully_qualified_command_name` maps it
# to None and the CME wildcard command walks the parent chain, which is the signal
# being acted on.
ESCALATION_OUTCOME_HONORED = "honored"
# The classifier ranked an escalation label alongside local candidates and was not
# confident. The user is prompted with the local candidates only (the ambiguity
# message filters non-routable labels) and the "try my parent" signal is dropped —
# GAP-18, recorded here as an outcome instead of inferred from a log line.
ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY = "suppressed_by_ambiguity"

ESCALATION_OUTCOMES: frozenset[str] = frozenset({
    ESCALATION_OUTCOME_NOT_EVALUATED,
    ESCALATION_OUTCOME_ABSENT,
    ESCALATION_OUTCOME_HONORED,
    ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY,
})

# Maximum normalized Levenshtein distance the fuzzy pre-match will accept. Named so
# the span can record what a distance was compared against — a distance with no
# threshold beside it cannot be binned by a calibration report. The value is
# unchanged.
_FUZZY_PREMATCH_MAX_DISTANCE = intent_clarification.FUZZY_MAX_DISTANCE

# Identifies the matcher that produced a fuzzy distance, so a curve drawn from one
# matcher is not silently continued by another. The name carries the two properties
# that fix the units: a normalized Levenshtein distance, and `best_window=False`,
# which scores only the candidate's LEADING len(input) characters. Turning
# `best_window` on can only lower distances, so it would shift every bin without
# changing the field name.
_FUZZY_MATCHER_VERSION = "levenshtein-leading-window/1"

# The matching layer between the context-scoped exact-name match and the fuzzy
# pre-match: the first token names a real command of this workflow, and this
# context is not one that owns it. Recorded as its own layer
# rather than folded into "no matcher claimed it", because the two are opposite
# facts -- this one is a deterministic refusal to answer, and a span that says
# so is how a misroute is counted after the fact.
MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT = "known_name_foreign_context"

# Reported when the classifier artifacts are not under the versioned layout. A
# tree that has never been trained under versioning has no version to report, and
# saying so is better than inventing one that would look comparable across runs.
_UNVERSIONED_ARTIFACT = "unversioned"

# Reported when the router did not say which model answered. Named rather than
# defaulted to "tiny", because a signal_version that claims the wrong tier is worse
# than one that admits it does not know.
_UNKNOWN_MODEL_TIER = "unknown-tier"


#: The hint is composed by the intent loop (the one place a foreign name is decided); kept under its name here.
foreign_context_hint = intent_loop.foreign_context_hint


def escalation_outcome_of(predictions: list[str]) -> str:
    """What became of an escalation signal in *predictions*.

    The classifier returns one label when it was confident and its top-k when it was
    not (``CommandRouter.predict_with_details``), so a lone escalation label is one
    the runtime acted on — it resolves to ``command_name=None`` and the CME wildcard
    command walks the parent chain. An escalation label ranked among several is the
    GAP-18 case: the ambiguity message filters non-routable labels out, so the user
    sees the local candidates and the "try my parent" signal goes nowhere.

    Reports an outcome, never a decision: the caller stores the result on the span
    and the resolution path never reads it back.
    """
    if not any(is_escalation(prediction) for prediction in predictions):
        return ESCALATION_OUTCOME_ABSENT
    if len(predictions) == 1:
        return ESCALATION_OUTCOME_HONORED
    return ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY


def _classifier_signal_version(model_artifact_path: str, model_tier: str) -> str:
    """Identity of the trained artifact behind a classifier signal.

    `signal_version` exists so that a retrained classifier is distinguishable: a
    confidence of 0.8 from one artifact is not the same measurement as a
    confidence of 0.8 from the next, and FW-REQ-021 clause 13 requires thresholds
    be re-validated when the producing artifact changes.

    Under the versioned artifact layout the per-context entry in
    ``___command_info`` is a compatibility link into
    ``___command_info/versions/<version>/<context>``, so one
    ``realpath`` recovers the published version without importing the trainer's
    resolver or re-reading its pointer file on every prediction. The ``*`` to
    ``global`` mapping mirrors ``CommandRouter.__init__``, which does the same
    substitution before opening the artifacts.
    """
    resolved = os.path.realpath(
        model_artifact_path.replace('*', GLOBAL_CONTEXT_FOLDER)
    )
    versions_parent = os.path.dirname(os.path.dirname(resolved))
    version = (
        os.path.basename(os.path.dirname(resolved))
        if os.path.basename(versions_parent) == VERSIONS_DIRNAME
        else _UNVERSIONED_ARTIFACT
    )
    return (
        f"intent-classifier/{version}"
        f"/{os.path.basename(resolved)}/{model_tier}"
    )


def _topk_margin_signals(
    classifier_details: dict, signal_version: str
) -> list[UncertaintySignal]:
    """The gap between the top two label probabilities, as a 0- or 1-item list.

    A list rather than an Optional so the caller has nothing to test. That is not
    style: `nlu_trace["classifier"]` belongs to `CommandRouter`, and the test
    doubles that inject labels return it EMPTY on purpose, so the emitter has to
    tolerate a router that reports no top-k exactly as it tolerates one that
    reports no `model_tier`. Pairing the kept scores with their own tail yields one
    pair for two scores and no pair at all for fewer, which gets that tolerance
    without comparing a captured measurement against anything (arch §17.3).

    The scores are positionally aligned with `topk_labels` and come from the single
    forward pass `predict_batch` already made, so this is a projection of a number
    that was computed and thrown away, not a new measurement.
    """
    top_two = list(classifier_details.get("topk_scores") or ())[:2]
    return [
        classifier_topk_margin(
            float(first) - float(second), signal_version=signal_version
        )
        for first, second in zip(top_two, top_two[1:])
    ]


def command_identity_uncertainty(nlu_trace: dict) -> DecisionUncertainty:
    """The §6.6.1 record for the command-identity decision *nlu_trace* describes.

    A pure function of the facts ``_predict_impl`` recorded, and deliberately
    one-way: it reads the capture bag and writes nothing back, so the resolution
    path cannot come to depend on what is being measured about it. That is the
    architecture §17.3 stop condition — capture only, no threshold, no branch.

    It reads ``matcher_layer``, which is the name of the branch that has already
    run, not a measurement of it. No confidence, distance, count, or assembled
    record is read by anything other than ``tracing.end_span``.
    """
    matcher_layer = nlu_trace.get("matcher_layer")
    signals: list[UncertaintySignal] = []
    signals_absent_reason = None
    # Every tier that enumerates candidates records how many; the tiers below that
    # do not enumerate resolved to exactly one command, or to none at all.
    candidate_count = nlu_trace.get("candidate_count", 1)

    if matcher_layer == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT:
        # The first token is a command name this workflow owns elsewhere. That
        # is a lookup against an inventory, not a measurement: there is no
        # confidence, distance or candidate set behind it, and the decision it
        # produced was "decline, let the parent chain answer". Reporting it as
        # deterministic keeps it out of every calibration curve, where a
        # confidence of 1.0 (or an uncaptured-measurement marker) would both be
        # false.
        signals_absent_reason = "deterministic-resolution"
        candidate_count = 0
    elif matcher_layer == "exact_prefix":
        # An exact command-name match has nothing to be unsure about. Emitting
        # confidence 1.0 here would enter a calibration curve as a real
        # measurement of a classifier that never ran.
        signals_absent_reason = "deterministic-resolution"
    elif matcher_layer in ("clarification_default", "question_reply"):
        # A reply to the which-command question is read by code (a name, a number, a word), and 'what can i do?'
        # was a constant substituted when no matcher claimed the utterance in a clarification stage. Nothing was
        # measured and nothing could have been, so this is deterministic in the same sense as an exact match rather
        # than an uncaptured measurement.
        signals_absent_reason = "deterministic-resolution"
    elif matcher_layer == "fuzzy_prematch":
        signals.extend([
            fuzzy_distance(
                nlu_trace["fuzzy_distance"], signal_version=_FUZZY_MATCHER_VERSION
            ),
            ambiguity_set_size(candidate_count, signal_version=_FUZZY_MATCHER_VERSION),
        ])
    elif matcher_layer == "matcher":
        # The model matcher answered for the scope (the local model was not confident, or the
        # scope has none). Its probabilities are on the span as `matcher`, but §6.6.1 has no
        # signal kind for an LLM's probability. The candidate count is what the matcher kept.
        signals_absent_reason = "not-instrumented"
    elif matcher_layer == "classifier":
        signal_version = nlu_trace["classifier_signal_version"]
        signals.append(
            classifier_confidence(
                nlu_trace["classifier"]["confidence"], signal_version=signal_version
            )
        )
        # classifier-topk-margin is missing on purpose: `predict_with_details`
        # returns the winning label's probability and the top-k label NAMES, and
        # `predict_single_sentence` drops `top_k_scores` from what `predict_batch`
        # computed. There is no second-best probability to subtract, and inventing
        # one is worse than its absence.
        #
        # That is no longer true: `predict_single_sentence` carries `top_k_scores`
        # through to `predict_with_details`, so the second-best probability is a
        # fact the same forward pass already produced. It stays absent when the
        # details dict does not carry it — a stubbed router, or an older record —
        # which is why the helper returns a list instead of raising on a missing
        # key.
        signals.extend(
            _topk_margin_signals(nlu_trace["classifier"], signal_version)
        )
        signals.append(
            ambiguity_set_size(candidate_count, signal_version=signal_version)
        )
    else:
        # No matcher claimed the utterance. The fuzzy pre-match did run and compared
        # a distance against the threshold, but `find_best_matches` returns
        # ``([], None)`` above the threshold and discards the value, so the one
        # measurement taken here is not recoverable at this call site.
        signals_absent_reason = "not-instrumented"
        candidate_count = 0

    return DecisionUncertainty(
        decision_kind="command-identity",
        signals=tuple(signals),
        candidate_count=candidate_count,
        reducible=nlu_trace.get("reducible"),
        signals_absent_reason=signals_absent_reason,
    )


def _recorded_command_identity_uncertainty(
    span, nlu_trace: dict
) -> Optional[dict]:
    """``command_identity_uncertainty`` in span-attribute form, never raising.

    Skipped entirely when *span* is None. ``start_span`` declines without a sink
    or an open turn and ``end_span`` then discards its attributes, so assembling a
    record nobody will store is pure overhead against this slice's declared
    budget — and it keeps the promise the test doubles in
    ``test_intent_detection_fuzzy_tie.py`` were written against, that a stubbed
    classifier's details reach nothing but a span that no-ops.

    ``tracing`` wraps every one of its own calls because a broken recorder must
    degrade to a log line rather than a failed turn, and this assembly runs just
    outside that guard, in the caller's frame. Capture that can lose a user's turn
    is precisely the behavior change Phase 0 exists to avoid.

    A record that violates its own contract is a bug in the recorder, so it is
    logged loudly and the attribute says what happened rather than going quietly
    absent — which would read as "this span predates the capture" — or null, which
    on the parameter-extraction side means "no decision". The pure function above
    still raises, which is what the tests exercise.
    """
    if span is None:
        return None
    try:
        return command_identity_uncertainty(nlu_trace).model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"command-identity uncertainty capture failed: {exc!r}")
        return {"capture_error": type(exc).__name__}


class CommandNamePrediction:
    """The CME's intent stage: ``intent_loop.resolve`` for the request, wrapped in the ``fw.nlu.intent`` span.

    The decision is the intent loop's (one decision for the whole walk, the same one the plan path makes: exact name,
    foreign-name guard, fuzzy name, the local model over the whole walk, then the chain). What is this class's: the CME
    may answer the framework's core commands (``core=True``), the span, and the facts the span reads."""

    def __init__(self, cme_workflow: fastworkflow.Workflow):
        self.cme_workflow = cme_workflow
        self.app_workflow = cme_workflow.context["app_workflow"]
        self.app_workflow_folderpath = self.app_workflow.folderpath
        self.app_workflow_id = self.app_workflow.id
        #: The span facts of the last decision.
        self.trace: dict = {}

    def predict(self, command: str) -> intent_loop.Intent:
        """The intent of *command* from where the application workflow stands, in one ``fw.nlu.intent`` span.

        One span per decision, with the context the command lives in (or the current context when nothing was
        decided). It carries which layer decided (exact prefix / fuzzy pre-match / foreign name / classifier /
        matcher), the classifier's confidence and threshold when it ran, and the candidate set on an ambiguity. Emission
        never affects the decision: the helpers no-op without a bound host/sink.

        It also carries the §6.6.1 ``DecisionUncertainty`` for the command-identity decision, assembled by
        ``command_identity_uncertainty`` from the facts below. That record is written to the span and read by nothing
        else (FW-REQ-021 P0: representation only).
        """
        host = tracing.current_host()
        span = tracing.start_span(
            host,
            tracing.SPAN_NLU_INTENT,
            attributes={
                "context": self.app_workflow._current_command_context_name,
                "stage": "INTENT_DETECTION",
                "utterance": command,
            },
        )
        nlu_trace: dict = {}
        try:
            intent = self._predict_impl(command, nlu_trace)
        except BaseException:
            # No DecisionUncertainty here: the prediction did not complete, so there is no decision to characterise.
            # The raw facts gathered so far still go out.
            tracing.end_span(host, span, status=tracing.STATUS_ERROR, attributes=nlu_trace)
            raise
        self.trace = nlu_trace
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK,
            attributes={
                **nlu_trace,
                "context": intent.context or self.app_workflow._current_command_context_name,
                "decision_uncertainty": _recorded_command_identity_uncertainty(span, nlu_trace),
                "command_name": intent.command,
                "is_cme_command": intent.kind == intent_loop.KIND_CORE,
                "ambiguous": intent.outcome == intent_loop.AMBIGUOUS,
                # None = no command: the CME files a misunderstanding (or a routing hint) — exactly the routing
                # signal a debugging agent needs.
                "resolved": intent.command is not None,
            },
        )
        return intent

    def _predict_impl(self, command: str, nlu_trace: dict) -> intent_loop.Intent:
        intent = intent_loop.resolve(self.app_workflow, command, core=True)
        nlu_trace.update(self._facts_of(intent))
        if intent.outcome == intent_loop.AMBIGUOUS and intent.decided_by == intent_loop.BY_QUESTION:
            if escalation_signals := self.escalation_signals_in(self._labels_of(intent)):
                logger.warning(
                    f"Top-k escalation signal discarded for utterance '{command}'. "
                    f"predictions={self._labels_of(intent)}, suppressed={escalation_signals}. The classifier ranked "
                    "an escalation label alongside local candidates; the user will be prompted with the local "
                    "candidates only and the 'this belongs to an ancestor context' signal is dropped.")
                nlu_trace["escalation_labels_discarded"] = [str(label) for label in escalation_signals]
        return intent

    @staticmethod
    def _labels_of(intent: intent_loop.Intent) -> list[str]:
        deciding = intent.deciding
        return [str(label) for label in deciding.labels] if deciding is not None else []

    def _facts_of(self, intent: intent_loop.Intent) -> dict:
        """The span facts of a decision, read off its engine answers (the keys the span contract names)."""
        trace: dict = {"escalation_outcome": ESCALATION_OUTCOME_NOT_EVALUATED}
        # What the name layers found, key by key (the span contract is read from these lines).
        facts = intent.facts
        if "matcher_layer" in facts:
            trace["matcher_layer"] = facts["matcher_layer"]
        if "known_name_foreign_context" in facts:
            trace["known_name_foreign_context"] = facts["known_name_foreign_context"]
            trace["known_name_owner_contexts"] = facts["known_name_owner_contexts"]
            trace["known_name_foreign_context_hint"] = facts["known_name_foreign_context_hint"]
        if "fuzzy_distance" in facts:
            trace["fuzzy_distance"] = facts["fuzzy_distance"]
            trace["fuzzy_threshold"] = facts["fuzzy_threshold"]
        if "fuzzy_prematch_tie" in facts:
            trace["fuzzy_prematch_tie"] = facts["fuzzy_prematch_tie"]
        if "candidate_count" in facts:
            trace["candidate_count"] = facts["candidate_count"]
        answers = intent.answers
        head = matcher = None
        for answer in answers:
            details = answer.details or {}
            if answer.engine == intent_detector.ENGINE_MATCHER:
                matcher = answer
                if isinstance(details.get("classifier"), dict):
                    head = head or details["classifier"]
            else:
                head = details
        classifier_answer = next((a for a in reversed(answers) if a.engine == intent_detector.ENGINE_CLASSIFIER), None)
        if not answers:
            return trace
        deciding = intent.deciding
        context = intent.context or self.app_workflow._current_command_context_name
        if head is not None:
            trace["classifier"] = head
            trace["classifier_signal_version"] = _classifier_signal_version(
                f"{self.app_workflow_folderpath}/___command_info/{context}",
                head.get("model_tier", _UNKNOWN_MODEL_TIER))
            labels = [str(label) for label in (classifier_answer.labels if classifier_answer is not None else ())]
            if labels:
                trace["candidate_count"] = len(labels)
                trace["escalation_outcome"] = escalation_outcome_of(labels)
        failed = next((a.details for a in answers if a.details.get("matcher_error")), None)
        if matcher is None and failed is not None:
            # The remote model was asked and its call failed: the local answer stands, carrying what happened.
            trace["matcher"] = {"error": failed["matcher_error"], "error_class": failed.get("matcher_error_class"),
                                "status_code": failed.get("matcher_status_code")}
            trace["engine_chain"] = [intent_detector.ENGINE_CLASSIFIER, intent_detector.ENGINE_MATCHER]
        elif matcher is not None:
            trace["matcher"] = matcher.details
            trace["engine_chain"] = [intent_detector.ENGINE_CLASSIFIER, intent_detector.ENGINE_MATCHER] \
                if head is not None else [intent_detector.ENGINE_MATCHER]
            trace["escalation_outcome"] = escalation_outcome_of([str(label) for label in matcher.labels])
            trace["candidate_count"] = len(matcher.labels) if matcher.kind != intent_detector.NOT_MINE else 0
        else:
            trace["engine_chain"] = [intent_detector.ENGINE_CLASSIFIER]
        errored = next((a.details for a in answers if a.details.get("error_class") or a.details.get("matcher_error_class")), None)
        if errored is not None:
            trace["remote_failed"] = {
                "error_class": errored.get("error_class") or errored.get("matcher_error_class"),
                "status_code": errored.get("status_code") or errored.get("matcher_status_code")}
        trace["engine"] = deciding.engine
        trace["engine_fallback"] = deciding.fallback
        trace["matcher_layer"] = "matcher" if deciding.engine == intent_detector.ENGINE_MATCHER else "classifier"
        if intent.decided_by == intent_loop.BY_QUESTION:
            trace["candidates"] = list(intent.ambiguous)
            trace["candidate_count"] = len(intent.ambiguous)
            # The runtime is about to ask, and the answer resolves this: requirements §4.14's reducible uncertainty,
            # recorded where the asking happens.
            trace["reducible"] = True
            trace["engine"] = intent_detector.ENGINE_QUESTION
        return trace

    @staticmethod
    def escalation_signals_in(route_choice_list: list[str]) -> list[str]:
        """Return the escalation labels present in a prediction list."""
        return [route_choice for route_choice in route_choice_list if is_escalation(route_choice)]

    def read_reply(self, record: dict, reply: str) -> tuple[intent_loop.Answer, Optional[intent_loop.Intent]]:
        """The reply to the pending which-command question, read by the intent loop's one reader, in one
        ``fw.nlu.intent`` span (stage ``INTENT_AMBIGUITY_CLARIFICATION`` or ``INTENT_MISUNDERSTANDING_CLARIFICATION``).
        The span says how the reply read (``matcher_layer`` ``question_reply``) and which command it picked. Returns the
        answer and, for a pick, where the command lives (``intent_loop.choose``; None for any other answer, and for a
        command no scope of the path has). A command of another scope than the current one is recorded on the span as
        ``chosen_command_elsewhere``: it runs on that scope's record, as a climb would."""
        stage = ("INTENT_MISUNDERSTANDING_CLARIFICATION" if record.get("misunderstood")
                 else "INTENT_AMBIGUITY_CLARIFICATION")
        host = tracing.current_host()
        span = tracing.start_span(
            host, tracing.SPAN_NLU_INTENT,
            attributes={"context": self.app_workflow._current_command_context_name, "stage": stage,
                        "utterance": reply})
        nlu_trace: dict = {}
        try:
            answered = intent_loop.answer(record, reply, aliases=intent_loop.core_aliases())
            chosen = intent_loop.choose(self.app_workflow, answered.name) if answered.kind == intent_loop.PICK else None
            if chosen is not None and chosen.context != self.app_workflow._current_command_context_name \
                    and chosen.kind != intent_loop.KIND_CORE:
                nlu_trace["chosen_command_elsewhere"] = answered.name
        except BaseException:
            tracing.end_span(host, span, status=tracing.STATUS_ERROR, attributes=nlu_trace)
            raise
        nlu_trace["matcher_layer"] = "question_reply"
        nlu_trace["escalation_outcome"] = ESCALATION_OUTCOME_NOT_EVALUATED
        nlu_trace["candidate_count"] = len(record.get("choices") or ())
        tracing.end_span(
            host, span, status=tracing.STATUS_OK,
            attributes={**nlu_trace, "engine": "user", "command_name": answered.name,
                        "resolved": answered.name is not None, "ambiguous": answered.kind == intent_loop.ASK_AGAIN,
                        "is_cme_command": chosen is not None and chosen.kind == intent_loop.KIND_CORE,
                        "decision_uncertainty": _recorded_command_identity_uncertainty(span, nlu_trace)})
        return answered, chosen

