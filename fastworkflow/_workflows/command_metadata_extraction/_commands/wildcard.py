from typing import Any, Optional

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import Action
from fastworkflow import intent_loop
from fastworkflow.parameters import parameter_clarification
from fastworkflow import waiting
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.nlu_labels import PARAMETER_VALUE_PLACEHOLDERS

from fastworkflow.utils.logging import logger

from ..intent_detection import CommandNamePrediction
from ..parameter_extraction import ParameterExtraction
from ..parameter_reply import (
    ACTION_DIFFERENT, ACTION_MERGE, ACTION_REPLACE, ACTION_RUN, ACTION_STOP, ACTION_WAIT, read_reply,
)

class Signature:
    class Output(BaseModel):
        """The routing result of one message. The executor reads it and builds the turn's CommandOutput.

        handled: a CME or core command already answered the message (or a routing hint did), so the
            turn ends here with ``response``. A handled answer carries the artifact ``command_handled``.
        routed: True routes the message on to an application command (run by the executor next);
            False refuses it, and ``response`` is the answer (success=False).
        command / command_name / cmd_parameters / ran_on_context_*: the extraction, when the message
            is routed on to the application command ``command_name`` (run by the executor next).
        """

        handled: bool = False
        routed: bool = True
        response: str = ""
        command: Optional[str] = None
        command_name: Optional[str] = None
        cmd_parameters: Optional[Any] = Field(default=None, exclude=True)
        ran_on_context_name: Optional[str] = None
        ran_on_context_clause: Optional[str] = None
        reply_intent: Optional[str] = None
        # The nested command's Output, when a CME command answered (the handled path only).
        output: Optional[Any] = Field(default=None, exclude=True)
        # Structured prompts for a human ("/") session only; None in agent mode (fix-weih.2).
        # parameter_request: the failed extraction's missing/invalid fields and the values given.
        # plan_choice: the which-command question's data (the same artifact the plan path's question carries,
        # ``question_kind`` "plan_choice"): the candidates an ambiguous message could mean, or every command.
        parameter_request: Optional[dict] = None
        plan_choice: Optional[dict] = None

    # These are the PARAMETER_EXTRACTION stage's bare-value literals. They belong
    # to nlu_labels.PARAMETER_VALUE_LABEL, not to this command's INTENT_DETECTION
    # escalation class; see fastworkflow/nlu_labels.py for the stage split.
    plain_utterances = list(PARAMETER_VALUE_PLACEHOLDERS)

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        # Only the humanised command name. The placeholders above are excluded on
        # purpose: training them into the 'wildcard' label taught the escalation
        # classifier that a bare value like "france" means "escalate to my
        # parent". The trainer labels them under PARAMETER_VALUE_LABEL instead.
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ]


def _run_cme_command(workflow, command: str, command_name: str, *, evidence: Optional[dict] = None,
                     opened: bool = False, correction: bool = True, keep_pending: bool = False,
                     in_parameters: bool = False) -> Signature.Output:
    """Run a framework (CME) command the message routed to, and wrap its result as the routing output.

    ``you_misunderstood`` opens the misunderstanding question (*opened*: the caller already did; *correction*: it
    corrects the last settled request). *keep_pending*: the answer does not end the pending which-command question
    (``what_can_i_do`` asked in reply to it). *in_parameters*: the message arrived while a parameter question was
    pending (an ``abort`` still ends it; ``you_misunderstood`` replaces it with the which-command question)."""
    workflow_context = workflow.context
    plan_choice = record = None
    if command_name == 'ErrorCorrection/you_misunderstood':
        record = (intent_loop.pending_record(workflow_context) if opened else
                  _open_misunderstanding(workflow, workflow_context["app_workflow"], command, correction=correction,
                                         evidence=evidence))
    elif (not in_parameters or command_name == 'ErrorCorrection/abort') and not keep_pending:
        workflow.end_command_processing()
    workflow.context = workflow_context

    startup_action = Action(
        command_name=command_name,
        command=command,
    )
    command_output = CommandExecutor.perform_action(workflow, startup_action)
    if record is not None:
        # The text the user read is the question, and the reply is read against what they saw.
        record["question"] = command_output.command_response.response
        workflow.context = workflow.context
        plan_choice = intent_loop.payload(record) if not _agentic(workflow) else None
    return Signature.Output(
        handled=True,       # a CME command's own answer: nothing is run next, whatever stage it arrived in
        routed=command_output.success,
        response=command_output.command_response.response,
        output=command_output.command_response.output,
        plan_choice=plan_choice,
    )


def _agentic(workflow) -> bool:
    """Agentic mode: not an assistant-mode ("/") command, in an agent session. Only a human's session gets the
    structured prompts (the agent sees none)."""
    app_workflow = workflow.context["app_workflow"]
    return not waiting.port_of(workflow.context).assistant and bool(app_workflow.context.get("run_as_agent"))


def _json_safe(value):
    """*value* as plain JSON data (the pending question is saved with the session)."""
    import json

    return json.loads(json.dumps(value, default=float))


def _open_misunderstanding(workflow, app_workflow, request: str, *, correction: bool = False,
                           evidence: Optional[dict] = None) -> dict:
    """Open the misunderstanding question: nothing fitted (or the user said so), so the candidates are every command.
    The record is the same plain-data record the ambiguity question keeps (``intent_loop``); the text the user reads is
    the ``you_misunderstood`` command's listing. ``correction``: the user said the last answer was wrong, so a pick
    corrects the last settled request."""
    record = intent_loop.open_question(
        request, intent_loop.command_choices(app_workflow), ambiguous=False, misunderstood=True, listing_hint=True)
    record["evidence"], record["correction"] = _json_safe(evidence or {}), correction
    context = workflow.context
    context["command"] = request
    intent_loop.store(context, record)
    workflow.context = context
    return record


def _settle(app_workflow, intent, command: str) -> None:
    """Record the command a request settled on (`intent_loop.record`, the one hook), called where the message moves on
    to parameter extraction: a name match, the local head or the remote model (an unconfirmed answer). Not a request:
    the text is a typed call of a named command with its parameters (what a plan's steps and an agent's tool calls
    are). Never raises."""
    try:
        how = intent_loop.how_settled(intent.decided_by)
        if how is None or (intent.decided_by == intent_loop.BY_EXACT_NAME and len(command.split(None, 1)) > 1):
            return
        app_workflow.context["intent_last"] = {"text": command, "command": intent.command,
                                               "evidence": intent.evidence}
        intent_loop.record(app_workflow, command, intent.command, how, intent.evidence, context=intent.context)
    except Exception as error:  # noqa: BLE001 - recording never fails a turn
        logger.debug(f"intent learning: not recorded ({error})")


def _record_choice(workflow, app_workflow, record: dict, command_name: str, context_name: Optional[str]) -> None:
    """Record the pick that answered the which-command question (a confirmed row). A pick after an explicit
    you_misunderstood corrects the last settled request (kept in the app workflow's context, which outlives the
    command): that request is the row's text, and the answer it got is the row's ``rejected_label``. Never raises."""
    try:
        from fastworkflow import intent_learning as learning

        how = learning.HOW_AGENT_CHOICE if _agentic(workflow) else learning.HOW_USER_CHOICE
        text, evidence, rejected = record.get("request") or "", record.get("evidence"), None
        if record.get("correction"):
            last = app_workflow.context.get("intent_last") or {}
            if last.get("text"):
                text, evidence = last["text"], last.get("evidence")
                rejected = last.get("command") if last.get("command") != command_name else None
        intent_loop.record(app_workflow, text, command_name, how, evidence, rejected, context=context_name)
    except Exception as error:  # noqa: BLE001 - recording never fails a turn
        logger.debug(f"intent learning: not recorded ({error})")


def _ambiguous_output(workflow, command: str, record: dict, question: str) -> Signature.Output:
    return Signature.Output(
        handled=False,
        routed=False,
        response=f"Ambiguous intent error for command '{command}'\n{question}",
        plan_choice=None if _agentic(workflow) else intent_loop.payload(record),
    )


def _answer_question(workflow, app_workflow, command: str, predictor) -> "Signature.Output | None":
    """The reply to the pending which-command question: read by the intent loop's one reader. Returns the turn's output
    when the reply ends or repeats the question, or None when it picked a command (the message goes on to parameter
    extraction, ``workflow.context`` holding the command and the text its parameters are read from)."""
    context = workflow.context
    record = intent_loop.pending_record(context)
    answered, chosen = predictor.read_reply(record, command)
    intent_loop.store(context, record)     # the counts the reader kept
    workflow.context = context
    kind = answered.kind
    if kind in (intent_loop.ASK_AGAIN, intent_loop.EXHAUSTED) and (foreign := intent_loop.foreign_hint(app_workflow, command)):
        # The reply names a real command of this workflow that no scope of the path owns: a misroute, not an
        # unreadable answer. Say where it lives and how to get there; the question is over.
        workflow.end_command_processing()
        return Signature.Output(handled=True, routed=False, response=foreign[0])
    if kind == intent_loop.ASK_AGAIN:
        return _ambiguous_output(workflow, record["request"], record, answered.question)
    if kind == intent_loop.ABORT or (kind == intent_loop.NONE and record.get("misunderstood")):
        # The user stopped, or said none of the listed commands fits after being shown every command.
        return _run_cme_command(workflow, command, "ErrorCorrection/abort")
    if kind == intent_loop.NONE:
        # None of the candidates: show every command, the same question with the full list.
        return _run_cme_command(workflow, record["request"], "ErrorCorrection/you_misunderstood",
                                evidence=record.get("evidence"), correction=False)
    if kind == intent_loop.LIST:
        # The question stays open: the listing is an answer to a different question.
        return _run_cme_command(workflow, command, "IntentDetection/what_can_i_do", keep_pending=True)
    if kind == intent_loop.EXHAUSTED:
        # The second reply that could not be read: the question ends with the listing of what can be done.
        return _run_cme_command(workflow, command, "IntentDetection/what_can_i_do")
    if chosen is None:
        workflow.end_command_processing()
        return Signature.Output(handled=True, routed=False,
                                response=f"'{answered.name.split('/')[-1]}' is not available from here.")
    if chosen.kind == intent_loop.KIND_CORE:
        return _run_cme_command(workflow, command, answered.name)
    # The question may have offered a command of another scope on the path: it runs where it lives.
    app_workflow._command_context_for_response_generation = chosen.record
    _record_choice(workflow, app_workflow, record, answered.name, chosen.context)
    context = workflow.context
    intent_loop.clear(context)
    # The reply's own words after the command name are its parameter text; else the request that was asked about.
    context["command"] = answered.rest or record["request"]
    context["command_name"] = answered.name
    workflow.context = context
    return None


def _detect(workflow, app_workflow, command: str, predictor) -> "Signature.Output | None":
    """The intent stage: the intent loop's decision for the request. Returns the turn's output when it ends here (a
    core command answered, a routing hint, the which-command question, the misunderstanding question), or None when a
    command was chosen (the message goes on to parameter extraction)."""
    intent = predictor.predict(command)
    context = workflow.context
    if intent.outcome == intent_loop.MATCH and intent.kind == intent_loop.KIND_CORE:
        if intent.command != 'ErrorCorrection/you_misunderstood':
            context["command"] = command
            workflow.context = context
        return _run_cme_command(workflow, command, intent.command, evidence=intent.evidence)
    if intent.outcome == intent_loop.MATCH:
        # The command runs on the record of the scope it lives in (a decline climbs: the walk binds the answering scope).
        app_workflow._command_context_for_response_generation = intent.record
        context["command"] = command
        context["command_name"] = intent.command
        workflow.context = context
        _settle(app_workflow, intent, command)
        return None
    if intent.outcome == intent_loop.AMBIGUOUS and intent.ambiguous:
        record = intent_loop.open_question(
            command, intent_loop.choices_of(app_workflow, list(dict.fromkeys(intent.ambiguous))), ambiguous=True,
            for_agent="run_as_agent" in app_workflow.context, listing_hint=True)
        record["evidence"], record["correction"] = _json_safe(intent.evidence), False
        question = intent_loop.ask(record)
        context["command"] = command
        intent_loop.store(context, record)
        workflow.context = context
        return _ambiguous_output(workflow, command, record, question)
    if intent.hint:
        # The name IS a command of this workflow; the walk simply never passed a context that owns it. "Nothing
        # matched" is true and useless here, so say where it lives and how to get there. A hint only: nothing here
        # navigates, and navigating on a guess about what was meant would change the workflow's state on the strength of
        # that guess. Not a misunderstanding either: the entering command the hint names usually lives elsewhere, so
        # command processing ends and the next message is routed normally.
        workflow.end_command_processing()
        return Signature.Output(handled=True, routed=False, response=intent.hint)
    # Out of scope: nothing fits. Show every command (the misunderstanding question).
    _open_misunderstanding(workflow, app_workflow, command, evidence=intent.evidence)
    return _run_cme_command(workflow, command, 'ErrorCorrection/you_misunderstood', opened=True)


def _dispatch_reply(workflow, command: str, reply, again) -> "tuple[Optional[Signature.Output], Optional[str]]":
    """Turn a ``parameter_reply.Reply`` (the reading of a reply to the pending parameter question) into this router's
    own calls. Returns ``(ended, extraction_text)``: the turn's output when the reply ends here, else None and, for a
    reply that is an answer, the text to merge into the stored command. *again* routes the reply as a new request.
    It holds no rule: what a reply means is decided in ``parameter_reply.read_reply``, and this handles every action of
    that vocabulary (``parameter_reply``'s table), whether or not a reader returns it yet."""
    action = reply.action
    if action == ACTION_RUN:
        routed = _run_cme_command(workflow, command, reply.name, in_parameters=reply.in_parameters,
                                  keep_pending=reply.keep_pending)
        if reply.reply_intent is not None:
            routed.reply_intent = reply.reply_intent
        return routed, None
    if action == ACTION_WAIT:
        return Signature.Output(handled=False, routed=False, reply_intent=reply.reply_intent, response=reply.text), None
    if action == ACTION_DIFFERENT:
        # The pending command is left: the stored parameters go with it (end_command_processing), and the reply runs
        # as a fresh request in INTENT_DETECTION.
        workflow.end_command_processing()
        out = again()
        if out.reply_intent is None:
            out.reply_intent = reply.reply_intent
        return out, None
    if action == ACTION_REPLACE:
        parameter_clarification.clear(workflow.context)
        workflow_context = workflow.context
        workflow_context["command"] = reply.command
        workflow.context = workflow_context
        return None, None
    if action == ACTION_MERGE:
        return None, f'{workflow.context["command_name"]}: {reply.text}'
    if action == ACTION_STOP:
        workflow.end_command_processing()
        return Signature.Output(handled=True, routed=False, reply_intent=reply.reply_intent, response=reply.message), None
    raise ValueError(f"unknown reply action {action!r}")


class ResponseGenerator:
    def __call__(
        self, 
        workflow: fastworkflow.Workflow, 
        command: str,
        input: None,
    ) -> Signature.Output:  # sourcery skip: hoist-if-from-if
        app_workflow = workflow.context["app_workflow"]   # type: fastworkflow.Workflow
        # What the message answers is read from the pending records: a which-command question (the intent loop's
        # record), a parameter question (the parameter loop), or nothing, and then the message is a request.
        waiting_for = intent_loop.waiting_for(workflow.context)
        pending_question = waiting_for == intent_loop.WAITING_INTENT
        in_parameters = waiting_for == intent_loop.WAITING_PARAMETERS

        if not in_parameters:
            predictor = CommandNamePrediction(workflow)
            ended = (_answer_question(workflow, app_workflow, command, predictor) if pending_question
                     else _detect(workflow, app_workflow, command, predictor))
            if ended is not None:
                return ended

        command_name = workflow.context["command_name"]
        extraction_text = None
        if in_parameters:
            ended, extraction_text = _dispatch_reply(
                workflow, command, read_reply(workflow, app_workflow, command), lambda: self(workflow, command, input))
            if ended is not None:
                return ended
        # Use the preserved original command (with parameters) if available
        preserved_command = f'{command_name}: {workflow.context.get("command", command)}'
        extractor = ParameterExtraction(
            workflow, app_workflow, command_name, extraction_text or preserved_command)
        pe_output = extractor.extract()
        if not pe_output.parameters_are_valid:
            return Signature.Output(
                handled=False,
                routed=False,
                reply_intent="answer",
                response=(
                    f"PARAMETER EXTRACTION ERROR FOR COMMAND '{command_name}'\n"
                    f"{pe_output.error_msg}"
                ),
                parameter_request=pe_output.parameter_request,
            )

        workflow.end_command_processing()

        # The walk bound an enclosing record: the command runs on that record, so it is
        # filed under it (design 3.9, "Filing the run"). Absent when it runs here.
        ran_on_name = ran_on_clause = None
        ran_on = app_workflow.record
        if ran_on is not app_workflow._current_command_context:
            from fastworkflow.context_identity import identity_of
            from fastworkflow.observation_offloading.labels import context_clause

            ran_on_name, ran_on_label = identity_of(app_workflow, ran_on)
            ran_on_clause = (
                context_clause(ran_on_name, ran_on_label) if ran_on is not None else "")

        reply_intent = "answer" if in_parameters else None
        return Signature.Output(
            handled=False,
            routed=True,
            reply_intent=reply_intent,
            response="",
            command=preserved_command,
            command_name=command_name,
            cmd_parameters=pe_output.cmd_parameters,
            ran_on_context_name=ran_on_name,
            ran_on_context_clause=ran_on_clause,
        )
