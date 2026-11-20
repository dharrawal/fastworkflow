"""The reply to a pending parameter question of the deterministic path, read as plain data.

``read_reply`` decides what the reply is and returns a ``Reply``: an ACTION plus its arguments. The router
(``_commands/wildcard.py``) only turns that data into its own calls (``_dispatch_reply``); it holds no rule of its own.
This is where the rules live, outside ``_commands/``, so a change of how a reply is read never edits (and so never
retrains) the built-in workflow (``docs/pending_state_design.md``, steps 4, 6 and 7).

The action vocabulary is FROZEN: the router handles every action below, including those no reader returns yet.

========== ================================================================ ======================================
action     meaning                                                          fields used
========== ================================================================ ======================================
RUN        run the built-in command *name* (``ErrorCorrection/abort``,      name, in_parameters, keep_pending,
           ``ErrorCorrection/you_misunderstood``,                           reply_intent
           ``IntentDetection/what_can_i_do``)
WAIT       answer with *text* and keep waiting: the question stays open     text, reply_intent
DIFFERENT  the reply is a new request: the pending command ends and the     reply_intent
           reply is routed from the start
REPLACE    the pending command is replaced by *command* (its stored         command
           parameters go), and extraction goes on
MERGE      the reply *text* is an answer: it is merged into the stored      text
           command, and extraction goes on
STOP       nothing more will be asked: the command ends, failed, with       message, reply_intent
           *message*, and no question stays pending
========== ================================================================ ======================================

``read_reply`` is a thin adapter from the one reader of a value question's reply (``parameters.reply.read_reply``, which a
plan part's value question is read by too) to RUN, WAIT, DIFFERENT, REPLACE and MERGE. STOP is not used by a reader: the
asking rule's end of the question is made by the extraction stage (design note, 3.10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastworkflow import parameter_clarification
from fastworkflow.parameters import reply as replies
from fastworkflow.utils.logging import logger
from fastworkflow.utils.signatures import MISSING_INFORMATION_ERRMSG

from .parameter_extraction import (
    ParameterExtraction,
    classify_missing_value_reply,
    missing_value_reply_enabled,
)

ACTION_RUN = "run"
ACTION_WAIT = "wait"
ACTION_DIFFERENT = "different"
ACTION_REPLACE = "replace"
ACTION_MERGE = "merge"
ACTION_STOP = "stop"
ACTIONS = (ACTION_RUN, ACTION_WAIT, ACTION_DIFFERENT, ACTION_REPLACE, ACTION_MERGE, ACTION_STOP)

ABORT = "ErrorCorrection/abort"
MISUNDERSTOOD = "ErrorCorrection/you_misunderstood"
WHAT_CAN_I_DO = "IntentDetection/what_can_i_do"


@dataclass(frozen=True)
class Reply:
    """What a reply to a pending parameter question is: ``action`` (one of ``ACTIONS``) and the fields that action uses
    (see the module's table); the rest keep their defaults."""

    action: str
    name: str = ""                       # RUN: the built-in command's qualified name
    in_parameters: bool = True           # RUN: the message arrived while a parameter question was pending
    keep_pending: bool = False           # RUN: the command's answer does not end the pending question
    reply_intent: Optional[str] = None   # RUN, WAIT, DIFFERENT, STOP: what the reply was read as (Jev's word)
    text: str = ""                       # WAIT: the answer text; MERGE: the reply merged into the stored command
    command: str = ""                    # REPLACE: the command text that replaces the stored one
    message: str = ""                    # STOP: why the command ended


def _missing_value_reply_intent(workflow, app_workflow, command_name: str,
                                command: str) -> tuple[str, list[str]]:
    """Jev's reading of a reply to the pending missing-value prompt, with the fields it asked for.

    Returns ("answer", []) whenever the check does not apply or fails: the switch is off, the
    session is in agent mode, nothing is pending, no Jev LM is configured, or any step raises.
    """
    if app_workflow.context.get("run_as_agent") or not missing_value_reply_enabled():
        return "answer", []
    loop = parameter_clarification.pending(workflow.context)
    if loop is None:
        return "answer", []
    try:
        from fastworkflow.workflow_agent import _jev_lm

        if _jev_lm() is None:
            return "answer", []
        return classify_missing_value_reply(loop.question, command), list(loop.open)
    except Exception:  # noqa: BLE001 - a failed check is an answer, as before this check existed
        logger.warning("missing-value reply check failed; treating the reply as an answer", exc_info=True)
        return "answer", []


def read_reply(workflow, app_workflow, command: str) -> Reply:
    """Read *command*, a reply to the CME's pending parameter question, as a ``Reply``: an adapter from the one reader's
    result (``parameters.reply.read_reply``, the reader a plan part's value question uses too) to the six actions.

    In order: ``cancel`` and its synonyms and the plain utterance of ``abort`` are ``abort`` (the pending command is left
    either way); ``you_misunderstood``'s utterance is that command; the pending command sent again in full replaces the
    extraction (fix-hia1.17, this path's own rule: ``replaces``); else, unless the reply is ``name: value`` lines, the
    remote model's read of the reply, once (a hesitation waits, a request to stop, a wrong-command report or a different
    request leaves the pending round); anything else is an answer to be merged."""
    command_name = workflow.context["command_name"]
    loop = parameter_clarification.pending(workflow.context)
    open_names = list(loop.open) if loop is not None else []
    listed = [*open_names, *(name for name in (loop.values if loop is not None else {}) if name not in open_names)]
    missing_fields: list[str] = []

    def remote(_question: str, text: str) -> str:
        intent, missing_fields[:] = _missing_value_reply_intent(workflow, app_workflow, command_name, text)
        return intent

    def replaces(text: str) -> bool:
        first_token = text.split(None, 1)[0].split("(", 1)[0] if text.strip() else ""
        # The pending command sent again in full is a new extraction, not an answer to the pending round: it replaces
        # the stored command and the parameters collected for it.
        return ParameterExtraction._is_own_command_token(first_token, command_name)

    reading = replies.read_reply(
        command, listed, open_names, question=loop.question if loop is not None else "",
        phrases=replies.phrase_of, replaces=replaces, remote=remote)
    if reading.kind == replies.DECLINE:
        return Reply(ACTION_RUN, name=ABORT, reply_intent=reading.intent)
    if reading.kind == replies.MISUNDERSTOOD:
        return Reply(ACTION_RUN, name=MISUNDERSTOOD, reply_intent=reading.intent)
    if reading.kind == replies.REPLACE:
        return Reply(ACTION_REPLACE, command=command)
    if reading.kind == replies.WAIT:
        return Reply(ACTION_WAIT, reply_intent=replies.INTENT_WAIT, text="\n".join(
            [replies.WAIT_TEXT, f"{MISSING_INFORMATION_ERRMSG}{', '.join(missing_fields)}"]))
    if reading.kind == replies.DIFFERENT:
        return Reply(ACTION_DIFFERENT, reply_intent=replies.INTENT_DIFFERENT)
    return Reply(ACTION_MERGE, text=command)
