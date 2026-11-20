"""The ONE reader of a reply to a value question (bd fix-fpyb.8, step 7).

A reply to a parameter question reads the same way for an execution command (the deterministic path) and for a plan
part (the gate before the review, and a part that is running): ``read_reply`` is the only reader, and each path only
turns its result into what it does next. ``parameters.read`` stays the code-only rule for values; this module adds what
a reply can be besides values, in this order:

1. ``cancel`` and its synonyms (``abort``, ``stop``, ``never mind``): ``decline``. A confirmation's ``yes``: ``accept``.
2. the caller's own answer rule (``answers``: what only the question knows, such as a decision step's yes or no); a
   reading it returns is final, with no phrase and no model call;
3. the fixed phrases of ``you misunderstood`` (``misunderstood``) and of ``abort`` (``decline``), by the caller's
   ``phrases`` function (the framework's own utterances for them);
3b. the caller's own rule (``replaces``: an execution command's pending command sent again in full: ``replace``);
4. a reply that read as ``name: value`` lines is values and is read no further;
5. any other reply goes to the remote decision model, ONCE, through the caller's ``remote`` function (None: no read, and
   so no model call). Its answer is ``wait`` (the user asks for a moment: the question stays), ``decline`` (it asked to
   stop), ``misunderstood`` (the command was the wrong one) or ``different`` (the reply is another request); ``answer``
   or a read below the threshold leaves the code's reading as it was.
6. else the reading of ``parameters.read``: ``values`` or ``free``.

Imports no model stack: the remote read is a function passed in (``remote_intent`` is the usual one and imports its
model stack only when it is called), and so are the phrases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

from fastworkflow import no_agent_mode
from fastworkflow.parameters import read

#: How a reply reads (``Reading.kind``): the four of ``parameters.read`` and the ones the remote read and the phrases add.
DECLINE, ACCEPT, VALUES, FREE = read.DECLINE, read.ACCEPT, read.VALUES, read.FREE
WAIT = "wait"
MISUNDERSTOOD = "misunderstood"
DIFFERENT = "different"
REPLACE = "replace"
KINDS = (DECLINE, ACCEPT, VALUES, FREE, WAIT, MISUNDERSTOOD, DIFFERENT, REPLACE)

#: What the fixed phrases and the remote model call a reply (the words of ``Reading.intent``).
INTENT_ABORT = "abort"
INTENT_MISUNDERSTOOD = "misunderstood"
INTENT_WAIT = "wait"
INTENT_DIFFERENT = "different_command"
_BY_INTENT = {INTENT_WAIT: WAIT, INTENT_ABORT: DECLINE, INTENT_MISUNDERSTOOD: MISUNDERSTOOD, INTENT_DIFFERENT: DIFFERENT}

#: The first line of the answer to a reply that asked for a moment.
WAIT_TEXT = "OK, take your time."


@dataclass(frozen=True)
class Reading:
    """What a reply is: ``kind`` (one of ``KINDS``), the ``values`` it gave (``values``), and ``intent``, the word the
    remote model read the reply as when it decided (``abort``, ``misunderstood``, ``wait``, ``different_command``), else
    None (a fixed phrase or the code's own reading)."""

    kind: str
    values: dict = field(default_factory=dict)
    intent: Optional[str] = None


def phrase_of(reply: str) -> Optional[str]:
    """``abort`` or ``misunderstood`` when *reply* is the plain utterance of the framework's ``abort`` or
    ``you_misunderstood`` command (the intent loop's one reader, with those two aliases), else None."""
    from fastworkflow import intent_loop

    stop = intent_loop.read(reply, [], aliases={
        phrase: kind for phrase, kind in intent_loop.core_aliases().items()
        if kind in (intent_loop.ABORT, intent_loop.NONE)})
    if stop.kind == intent_loop.ABORT:
        return INTENT_ABORT
    return INTENT_MISUNDERSTOOD if stop.kind == intent_loop.NONE else None


def remote_intent(question: str, reply: str) -> str:
    """The remote decision model's read of *reply* to *question*: ``wait``, ``abort``, ``misunderstood``,
    ``different_command`` or ``answer``. ``answer`` with no model call when no remote model is set, when the read is
    switched off (``FW_JEV_MISSING_VALUE_REPLY``) or when it fails."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        classify_missing_value_reply, missing_value_reply_enabled)

    return classify_missing_value_reply(question, reply) if missing_value_reply_enabled() else "answer"


def read_reply(reply: str, names: Iterable[str], open_names: Iterable[str], *, confirm: bool = False,
               slots: Optional[Mapping[str, read.Slot]] = None, question: str = "",
               phrases: Optional[Callable[[str], Optional[str]]] = None,
               replaces: Optional[Callable[[str], bool]] = None,
               answers: Optional[Callable[[str], Optional[Reading]]] = None,
               remote: Optional[Callable[[str, str], str]] = None) -> Reading:
    """How *reply* reads (the module docstring is the order). *names*, *open_names*, *confirm* and *slots* are
    ``parameters.read``'s. *phrases* maps a reply to ``abort`` / ``misunderstood`` / None, *replaces* says whether the
    reply replaces the pending command (an execution command's own rule), *answers* maps a reply to the ``Reading`` of
    the question's own answer form or None (a decision step's yes or no), *remote* maps ``(question, reply)`` to the
    remote model's word for the reply (made at most once per reading, and only for a reply that is not ``name: value``
    lines); a function left None is a step skipped."""
    base = read.read(reply, names, open_names, confirm=confirm, slots=slots)
    if base.kind in (read.DECLINE, read.ACCEPT):
        return Reading(base.kind)
    if answers is not None and (answer := answers(reply)) is not None:
        return answer
    if phrases is not None and (phrase := phrases(reply)) is not None:
        return Reading(_BY_INTENT[phrase])
    if replaces is not None and replaces(reply):
        return Reading(REPLACE)
    if remote is not None and not base.named:
        intent = remote(question, reply)
        if intent in _BY_INTENT:
            return Reading(_BY_INTENT[intent], intent=intent)
    return Reading(base.kind, dict(base.values))


def read_decision(decision: Mapping, reply: str, *, phrases: Optional[Callable[[str], Optional[str]]] = phrase_of,
                  remote: Optional[Callable[[str, str], str]] = remote_intent) -> Reading:
    """How *reply* reads to a paused planning command's decision step (``decision``: the record in the turn's waiting
    record). The same reader as every value question, with the decision's own answer rule plugged in: a conditional step
    is answered yes or no (``values`` ``{"answer": "yes" | "no"}``), a command step missing one plain value takes the
    reply as that field's value (``values`` ``{<field>: value}``, or a ``<field>: value`` line). Everything else reads as
    it does there: ``decline`` (cancel, abort, stop, never mind and the framework's ``abort`` phrases), ``misunderstood``,
    ``wait`` and ``different`` (the phrases and the remote model's read, made at most once and never for a yes or no or a
    named line), else ``free`` (nothing usable). A function passed as None is a step skipped."""
    if decision["kind"] == no_agent_mode.STEP_CONDITIONAL:
        def yes_no(text: str) -> Optional[Reading]:
            answer = no_agent_mode.read_yes_no(text)
            return None if answer is None else Reading(VALUES, {"answer": "yes" if answer else "no"})

        return read_reply(reply, [], [], question=decision.get("question", ""), answers=yes_no, phrases=phrases,
                          remote=remote)
    field_name = decision["field"]
    return read_reply(reply, [field_name], [field_name], slots={field_name: read.slot_of_text(decision["type"])},
                      question=decision.get("question", ""), phrases=phrases, remote=remote)
