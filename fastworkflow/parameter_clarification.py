"""The parameter clarification loop: one way to ask for the values a command still lacks and to read the replies.

A caller that already knows the command enters it here: a turn part bound to a planning command, a plan a client pinned
(``use_plan``), a typed planning call that was refused, a plan step missing one plain value, and the CME parameter
stage of an execution command. They used to have a reader and a progress rule each, and the reply to a planning
command's question re-ran the whole turn. Here there is one result contract (``Extraction``: the values of a command
given some text and each parameter's state), one reader (``read_reply``), one question builder (``planning_question``),
one asking rule (``made_progress``, ``advance`` / ``reopen`` / ``record_round``) and one piece of state (``ParameterLoop``), which is plain
data so that it is saved with the session and the reply resumes the part that asked.

Both kinds of command produce an ``Extraction`` and everything after it reads only that: the planning draft
(``planning_commands.argument_gaps``, the typed call's checks) and the execution-command extractor
(``parameter_extraction.extraction_of``). The two extraction mechanisms stay what they are; what they return does not.

Nothing in this module calls a model. Free text is read by an extractor the caller passes in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable, Optional

from fastworkflow import no_agent_mode, waiting
from fastworkflow.parameters import read, reply as replies

KIND_PLANNING = "planning"
KIND_EXECUTION = "execution"

#: THE asking rule, for a planning command and an execution command alike, in every session type: the question is asked
#: again while each reply makes progress (``made_progress``); a reply with none is asked once more; the second reply in a
#: row with none stops it (``no_agent_mode.MAX_NO_PROGRESS``, counted in ``ParameterLoop.stalled``); the kind's cancel
#: reply always ends it. What "stop" hands over to is the caller's one decision: with an agent model the agent or the text
#: planner gets the refusal, otherwise the part or command ends as not run naming exactly what is missing.

#: What follows the answer to a parameter question (the ``after`` of the waiting record, ``docs/pending_state_design.md``
#: 3.1): the CME runs the command with the values, or the plan review applies the edit its slot question asked for.
AFTER_RUN_COMMAND = "run_command"
AFTER_EDIT_SLOT = "edit_slot"
#: On the plan path: the resolution goes on to the next part, the review or the run, or the part that was running
#: resumes in place (a refused typed planning call asked about its values).
AFTER_RESOLVE_NEXT = "resolve_next"
AFTER_RESUME_PART = "resume_part"

#: How a reply reads (``read_reply``).
REPLY_DECLINE = read.DECLINE
REPLY_ACCEPT = read.ACCEPT
REPLY_VALUES = read.VALUES
REPLY_FREE = read.FREE

#: What a reply did to the loop (``Outcome.kind``).
COMPLETE = "complete"
ASK_AGAIN = "ask_again"
DECLINED = "declined"
EXHAUSTED = "exhausted"
#: The reply was not an answer (``parameters.reply``, the one reader): the user asked for a moment (``wait``: the
#: question stays, nothing is counted), said the command was the wrong one (``misunderstood``) or sent another request
#: (``different``). What each does next is the caller's (the plan path: ``resolution`` and the execution context).
WAIT = replies.WAIT
MISUNDERSTOOD = replies.MISUNDERSTOOD
DIFFERENT = replies.DIFFERENT

#: Opens a question asked again after a reply that closed no open value.
RETRY = "That did not give a value I could use. "

#: The state of one parameter of a command given some text (``Extraction.state_of``): it has a value the checks stand
#: behind, it is required and not given, it was given and refused (with the reason), or an earlier part of the turn
#: produces it, so it is nobody's to give here.
STATE_GIVEN = "given"
STATE_MISSING = "missing"
STATE_INVALID = "invalid"
STATE_EARLIER = "from_earlier"


@dataclass(frozen=True)
class Extraction:
    """The values of one command given some text, and what they still lack: the ONE result of every extractor.

    ``values`` are the values the checks stand behind (not the ones still open). ``missing`` names the required
    parameters not given, ``invalid`` the ones given and refused (each with why), ``earlier`` the ones an earlier part of
    the turn produces (``from_earlier``). ``reason`` says why the whole call is refused when that names no parameter (a
    rule of the command) or, for an extractor that words its refusal, the refusal itself. A call with no open name and no
    reason is complete."""

    values: dict = field(default_factory=dict, compare=False)
    missing: tuple[str, ...] = ()
    invalid: tuple[tuple[str, str], ...] = ()
    earlier: tuple[str, ...] = ()
    reason: str = ""
    #: For a missing name whose value the guards refused as not given (a pronoun, a vocabulary word, a placeholder): the
    #: value with the reason, as the question says it (``Gap.screened``). Wording only: it does not change what is open.
    screened: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    #: The duplicate rule's notes of the optional fields it left not given (``ask.left_empty_line``), one line each: said
    #: before a question about these values. Wording only: such a field is not open.
    left_empty: tuple[str, ...] = field(default=(), compare=False)

    @property
    def names(self) -> tuple[str, ...]:
        """The parameters still open (missing, then refused), in order, once each."""
        return tuple(dict.fromkeys([*self.missing, *(name for name, _ in self.invalid)]))

    def state_of(self, name: str) -> tuple[str, str]:
        """``(state, why)`` of one parameter: ``STATE_MISSING``, ``STATE_INVALID`` (with the reason),
        ``STATE_EARLIER`` or ``STATE_GIVEN``."""
        if name in self.missing:
            return STATE_MISSING, ""
        for each, why in self.invalid:
            if each == name:
                return STATE_INVALID, why
        return (STATE_EARLIER, "") if name in self.earlier else (STATE_GIVEN, "")

    def describe(self) -> str:
        """The gaps in a plain phrase: ``missing required argument a, b`` / ``c is not ...`` (the not-run reason)."""
        parts = []
        if self.missing:
            parts.append(f"missing required argument{'s' if len(self.missing) > 1 else ''} {', '.join(self.missing)}")
        parts.extend(f"{name} {why}".strip() for name, why in self.invalid)
        return "; ".join(parts)

    def from_earlier(self, names: Iterable[str]) -> "Extraction":
        """This extraction with *names* (the parameters an earlier part of the turn produces) no longer open: they are
        nobody's to give, so no question asks for them."""
        names = set(names)
        if not names:
            return self
        moved = tuple(name for name in self.names if name in names)
        return Extraction(values=self.values, missing=tuple(n for n in self.missing if n not in names),
                          invalid=tuple((n, why) for n, why in self.invalid if n not in names),
                          earlier=tuple(dict.fromkeys([*self.earlier, *moved])), reason=self.reason,
                          screened=tuple((n, text) for n, text in self.screened if n not in names),
                          left_empty=self.left_empty)

    def request(self, command: str, *, stringify: bool = True) -> Optional[dict]:
        """The open parameters as data for a client that renders a form (``parameter_request``: ``command``, ``missing``,
        ``invalid``, ``values``), or None when none is open (a refusal that names no parameter has nothing to ask field by
        field). *stringify*: the values as text (a planning command's) or as they are (JSON-safe ones)."""
        if not self.names:
            return None
        return {"command": command, "missing": list(self.missing), "invalid": [name for name, _ in self.invalid],
                "values": {name: str(value) if stringify else value for name, value in self.values.items()}}


_PARAMETER_LINE = read.PARAMETER_LINE


# ---- the reader ------------------------------------------------------------------------------------------------------

def named_values(reply: str, names: Iterable[str]) -> dict[str, str]:
    """The ``name: value`` (or ``name=value``) lines of *reply* that name one of *names*, one per line; the name is
    matched without regard to case and returned as *names* spells it. A line with no value is skipped."""
    return read.named_values(reply, names)


def slots_of(command) -> dict[str, read.Slot]:
    """The ``read.Slot`` of each parameter of planning command *command* (what a bare yes or no needs to know)."""
    fields = command.input_model.model_fields
    return {p.name: read.slot_of(fields[p.name].annotation, required=bool(p.required)) for p in command.params}


def read_reply(reply: str, names: Iterable[str], open_names: Iterable[str], *, confirm: bool = False,
               slots: Optional[dict] = None) -> tuple[str, dict[str, Optional[str]]]:
    """How a reply to a parameter question reads, strictly and in code: ``(kind, values)``. The one rule is
    ``parameters.read``'s (its docstring): ``decline`` (cancel and its synonyms), ``accept`` (yes, only when *confirm*:
    every parameter was shown), ``values`` (``name: value`` lines of the fields the question listed, *names*; or the
    whole reply as the value when one value is open; or comma-separated parts in order when several are and the counts
    match; a value of ``None`` leaves an optional field empty) and ``free`` (anything else: the caller decides whether an
    extractor reads it or it counts as a reply that made no progress). *slots*: ``slots_of(command)``."""
    reading = read.read(reply, names, open_names, confirm=confirm, slots=slots)
    return reading.kind, reading.values


# ---- the state -------------------------------------------------------------------------------------------------------

@dataclass
class ParameterLoop:
    """One command's open parameter question, as plain data (held in the turn's waiting record, kind ``parameters``, and saved with the session).

    ``command`` is the qualified name and ``kind`` what it is ("planning" or "execution"). ``values`` are the values
    kept so far, ``open`` the names still missing or invalid in order, ``invalid`` the ones of those that have a value
    that was refused (name -> why). ``asks`` counts the questions put and ``stalled`` the replies in a row that closed
    nothing. ``request_text`` is the text the values were read from, ``origin`` the turn part the loop belongs to (None
    for a part that had not started: a pinned plan asked before its run). ``confirm``: every parameter is shown and
    ``yes`` runs.

    Kept beside those: ``asked`` (the names the last question asked about, which the progress rule compares against;
    ``made_progress``), ``given`` (the names whose value came from the user's replies, which are the ones
    put on the part when it resumes), ``replies`` (free-text replies, read after the request text), and the last
    ``question`` and its ``request`` data, so the pending prompt can be described without rebuilding them.
    """

    command: str
    kind: str = KIND_PLANNING
    values: dict[str, Any] = field(default_factory=dict)
    open: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)
    asks: int = 0
    stalled: int = 0
    request_text: str = ""
    origin: Optional[int] = None
    confirm: bool = False
    asked: list[str] = field(default_factory=list)
    given: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    question: str = ""
    request: Optional[dict[str, Any]] = None

    def to_state(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(cls, state: Any) -> Optional["ParameterLoop"]:
        """The loop a saved state holds, or None when it holds none (old saved sessions are not supported: a state with
        a key this build does not know is an error)."""
        if not isinstance(state, dict) or not state.get("command"):
            return None
        return cls(**state)

    def given_values(self) -> dict[str, str]:
        """The values the user's replies gave, by name."""
        return {name: self.values[name] for name in self.given if name in self.values}

    def tail(self) -> str:
        """What the replies add after the request text: the free-text replies, then one ``name: value`` line for each
        value a reply gave."""
        lines = [*self.replies, *(f"{name}: {value}" for name, value in self.given_values().items())]
        return "\n".join(line for line in lines if line)

    def text(self) -> str:
        """What the values are read from and checked against: the request with the replies after it (``tail``), so a
        check that a value is written in the request holds for a value the user typed."""
        return "\n".join(part for part in (self.request_text, self.tail()) if part)

    def _keep(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if value is None:
                self.values.pop(name, None)      # left empty on purpose
            else:
                self.values[name] = str(value)
            if name not in self.given:
                self.given.append(name)


@dataclass(frozen=True)
class Outcome:
    """What a reply did: ``complete`` (run with ``loop.values``; nothing is open), ``ask_again`` (``question`` and
    ``request``), ``declined``, or ``exhausted`` (the loop ended on replies that made no progress). ``missing`` names
    exactly what is still open for the last two, ``describe`` says it as the not-run reason does, and ``reply_kind`` is
    how the last reply read."""

    kind: str
    question: str = ""
    request: Optional[dict[str, Any]] = None
    missing: tuple[str, ...] = ()
    describe: str = ""
    reply_kind: str = ""
    waited: int = 0          # 1 when the reply was a request for a moment, else 0: the question's record keeps it


# ---- the question ----------------------------------------------------------------------------------------------------

def made_progress(asked: Iterable[str], names: Iterable[str]) -> bool:
    """THE definition of progress, used on both paths: a reply made progress when it filled or fixed at least one of the
    values the last question listed (*asked*), that is, one of them is no longer open (*names*, what is open now)."""
    return bool(set(asked) - set(names))


def refusal_from_gaps(gaps, values: Optional[dict] = None):
    """A refusal that stands for *gaps*, worded as the argument checks word theirs, for the question builders that read
    a refusal. It carries the gaps, so a question about several arguments names them all."""
    from fastworkflow import planning_commands

    from fastworkflow.parameters import check

    values = values or {}
    gap = None
    if gaps.missing:
        text = f"missing required argument {gaps.missing[0]}"
        gap = check.Gap(gaps.missing[0], check.MISSING, "missing", "", None, text)
    elif gaps.invalid:
        name, why = gaps.invalid[0]
        shown = str(values.get(name, ""))
        text = f"{name}={shown!r} {why}"
        gap = check.Gap(name, check.INVALID, "refused", why, shown, text)
    else:
        text = ""
    error = planning_commands.PlanningCommandInputError(text, gap=gap)
    error.gaps = gaps
    return error


def planning_question(command, values: dict, gaps, *, confirm: bool = False, description: str = "", retry: str = "",
                      refusal=None) -> tuple[str, Optional[dict]]:
    """``(question, parameter_request)`` for planning command *command* with *gaps* still open.

    With *confirm* (a plan the client chose) every parameter is presented: its name and *description*, each parameter
    with the value kept for it or what it still lacks, and how to reply; the data names the same (``command``,
    ``description``, ``parameters``, ``missing``, ``invalid``, ``values``, ``confirm``), so a client can show a form with
    the values filled in. Without it the question names what is missing or refused, in the words of
    ``planning_commands.refusal_question`` (from *refusal* when the caller has the checks' own, else from the gaps), and
    the data is ``command``, ``missing``, ``invalid``, ``values``. *retry* opens a question asked again.
    """
    from fastworkflow.parameters import ask

    question = ask.word(gaps, form=ask.FORM_REFUSAL, command=command, refusal=refusal, values=values, confirm=confirm,
                        description=description, retry=retry)
    return (question.text if confirm else ask.with_answer_only_note(question.text)), question.request


# ---- the loop --------------------------------------------------------------------------------------------------------

def pending(context: dict) -> Optional[ParameterLoop]:
    """The loop the CME's turn holds (the waiting record of kind ``parameters``), or None."""
    held = waiting.port_of(context).get()
    return ParameterLoop.from_state(held.payload.get("loop")) if held is not None and held.kind == waiting.KIND_PARAMETERS else None


def requestion(context: dict, question: str) -> None:
    """Make *question* the text of the pending loop's question (the question a caller asked in its own words), so the
    reply is read against what the user saw. Nothing when no loop is pending."""
    if (loop := pending(context)) is not None:
        loop.question = question
        held = waiting.port_of(context).get()
        store(context, loop, after=held.payload.get("after", AFTER_RUN_COMMAND), waited=waited_of(held))


def waited_of(held) -> int:
    """The waits in a row the question's record *held* carries (payload key ``waited``, absent when none). It is kept
    with the question, not on the loop, and ends with it: a request for a moment is free once, and each further one in
    a row counts as a reply with no progress (``advance``; the CME's ``parameter_reply.read_reply``)."""
    return int(held.payload.get("waited") or 0) if held is not None else 0


def wait_question(question: str) -> str:
    """What a reply that asks for a moment gets, for every question that reads it (a plan part's values, the CME's
    parameter question, a plan's decision step): one line, then the same question as it was (bd fix-ftdl.7.8, D9)."""
    return f"{replies.WAIT_TEXT}\n\n{question}"


def store(context: dict, loop: ParameterLoop, after: str = AFTER_RUN_COMMAND, waited: int = 0) -> None:
    """Make *loop* the turn's one pending question: the parameter loop replaces whatever question the CME held (the port
    holds one record at a time). *after* says what follows the answer (``AFTER_RUN_COMMAND``, ``AFTER_EDIT_SLOT``)."""
    payload = {"loop": loop.to_state(), "after": after, **({"waited": waited} if waited else {})}
    waiting.port_of(context).put(waiting.Waiting(waiting.KIND_PARAMETERS, owner=waiting.OWNER_CME, payload=payload))


def waiting_record(loop: ParameterLoop, after: str, part: Optional[int] = None, waited: int = 0) -> waiting.Waiting:
    """The plan path's waiting record for *loop*'s question (owner ``plan``): ``after`` says what follows the answer
    (``AFTER_RESOLVE_NEXT``, ``AFTER_RESUME_PART``), ``part`` the turn part it belongs to and ``waited`` the requests
    for a moment in a row (``Outcome.waited``)."""
    return waiting.Waiting(waiting.KIND_PARAMETERS, question=loop.question,
                           payload={"loop": loop.to_state(), "after": after, "part": part,
                                    **({"waited": waited} if waited else {})})


def clear(context: dict) -> None:
    """Close the parameter question (nothing when none is pending)."""
    if pending(context) is not None:
        waiting.port_of(context).clear()


def _take_gaps(loop: ParameterLoop, gaps) -> None:
    """Put what the checks found on the loop: what is open, why the refused ones are refused, and the values the
    checks stand behind (in the checks' own spelling)."""
    loop.open = list(gaps.names)
    loop.invalid = {name: why for name, why in gaps.invalid}
    loop.values.update({name: str(value) for name, value in gaps.values.items()})


def _ask(loop: ParameterLoop, command, gaps, *, description: str, retry: str, refusal=None) -> Outcome:
    shown = {name: value for name, value in loop.values.items() if name not in gaps.names}
    question, request = planning_question(command, shown if loop.confirm else loop.values, gaps, confirm=loop.confirm,
                                          description=description, retry=retry, refusal=refusal)
    return _put(loop, gaps, question, request)


def _put(loop: ParameterLoop, gaps: Extraction, question: str, request: Optional[dict]) -> Outcome:
    """Put *question* out: one more question about the names *gaps* still has open."""
    loop.asks += 1
    loop.asked = list(gaps.names)
    loop.question, loop.request = question, request
    return Outcome(ASK_AGAIN, question=question, request=request)


def stopped(loop: ParameterLoop) -> bool:
    """Whether the asking rule says the loop ends here: ``MAX_NO_PROGRESS`` replies in a row made no progress."""
    return loop.stalled >= no_agent_mode.MAX_NO_PROGRESS


def record_round(loop: ParameterLoop, extraction: Extraction, question: str) -> Outcome:
    """One more round of an execution command's parameter stage: its extractor produced *extraction* and worded its own
    *question* (the command's validation message, with its hook text and suggestions). The values it found are kept,
    progress is counted against the names the last question asked about (``made_progress``) and the question is put.
    The outcome is ``exhausted`` (naming exactly what is still open) when this is the ``MAX_NO_PROGRESS``-th reply in a
    row with none (``stopped``): the caller ends the command instead of keeping the question."""
    progressed = not loop.asked or made_progress(loop.asked, extraction.names)
    loop.values = dict(extraction.values)
    loop.open, loop.invalid = list(extraction.names), dict(extraction.invalid)
    loop.stalled = 0 if progressed else loop.stalled + 1
    outcome = _put(loop, extraction, question, extraction.request(loop.command, stringify=False))
    if stopped(loop):
        return Outcome(EXHAUSTED, missing=tuple(extraction.names), describe=extraction.describe())
    return outcome


def open_loop(command, gaps, *, request_text: str, origin: Optional[int] = None, confirm: bool = False,
              description: str = "", refusal=None, values: Optional[dict] = None,
              kind: str = KIND_PLANNING) -> tuple[ParameterLoop, Outcome]:
    """A new loop for *command* (a planning command, or an execution command as ``planning_commands.execution_command``
    states it: *kind*) and its first question. *gaps* is what the call lacks (``planning_commands.argument_gaps``); its
    checked values, and *values* when given, are the values kept."""
    loop = ParameterLoop(command=command.name, kind=kind, request_text=str(request_text or ""), origin=origin,
                         confirm=confirm,
                         values={name: str(value) for name, value in (values or {}).items() if value is not None})
    _take_gaps(loop, gaps)
    return loop, _ask(loop, command, gaps, description=description, retry="", refusal=refusal)


def reopen(loop: ParameterLoop, command, gaps, *, description: str = "", refusal=None) -> Outcome:
    """The part ran again with the loop's values and its own checks still refuse something (*gaps*): the next question,
    or ``exhausted``. Progress is measured against the names the last question asked about. A loop whose last question
    asked about nothing (a confirmation with every value given) asks its first question about the gaps. A loop that
    already ended on replies with no progress stays ended: it is never asked again (the hand-over was made)."""
    first_about_it = not loop.asked and not loop.stalled
    progressed = first_about_it or made_progress(loop.asked, gaps.names)
    _take_gaps(loop, gaps)
    if stopped(loop):
        return Outcome(EXHAUSTED, missing=tuple(gaps.names), describe=gaps.describe())
    return _settle(loop, command, gaps, progressed, description=description, refusal=refusal, reply_kind="")


def _settle(loop: ParameterLoop, command, gaps, progressed: bool, *, description: str, refusal=None,
            reply_kind: str) -> Outcome:
    """The progress rule: a reply that closed an open value starts the count again; one that closed none is asked once
    more, and ``MAX_NO_PROGRESS`` of those in a row end the loop naming exactly what is still open."""
    loop.stalled = 0 if progressed else loop.stalled + 1
    if stopped(loop):
        return Outcome(EXHAUSTED, missing=tuple(gaps.names), describe=gaps.describe(), reply_kind=reply_kind)
    return _ask(loop, command, gaps, description=description, retry="" if progressed else RETRY, refusal=refusal)


def advance(loop: ParameterLoop, reply: str, command, *,
            extract: Optional[Callable[[str], dict]] = None, description: str = "",
            not_asked: Iterable[str] = (), live: bool = False, waited: int = 0) -> Outcome:
    """Take the reply to the loop's question.

    The reply is read by the one reader (``parameters.reply.read_reply``). With *live* (a caller that serves a user) it
    reads the fixed phrases of ``abort`` and ``you misunderstood`` and, for a reply that is not ``name: value`` lines,
    asks the remote decision model once (no call when none is set): ``abort`` is ``declined``, and ``wait``,
    ``misunderstood`` and ``different`` come back as outcomes of those kinds with the loop untouched (the first wait in
    a row keeps the question and counts nothing; *waited* is the waits in a row so far, held in the question's record,
    and each further one is a reply that made no progress, so ``MAX_NO_PROGRESS`` of them end the loop). Without it the reply is read in code only. Values are kept on the loop; the gaps are worked out again by
    ``planning_commands.argument_gaps`` on the values kept, against the request text with the replies after it; and the
    progress rule decides between ``complete``, ``ask_again`` and ``exhausted``. Free text is given to *extract* (text ->
    {name: value}, the caller's extractor) when the loop is not a confirmation; a confirmation is strict, so free text
    there is a reply that made no progress.

    *not_asked* names the arguments that are not the user's to give (an earlier part produces them): they are never
    open here. The rule is the one above (``made_progress``, ``MAX_NO_PROGRESS``): what the caller does with
    ``exhausted`` is its one hand-over.
    """
    from fastworkflow import planning_commands

    names = [p.name for p in command.params]
    listed = [*loop.open, *(name for name in names if loop.confirm or loop.values.get(name) not in (None, ""))]
    slots = slots_of(command)
    reading = replies.read_reply(
        reply, listed, loop.open, confirm=loop.confirm, slots=slots, question=loop.question,
        phrases=replies.phrase_of if live else None, remote=replies.remote_intent if live else None)
    kind, values = reading.kind, reading.values
    before = list(loop.asked)
    if kind == REPLY_DECLINE:
        return Outcome(DECLINED, missing=tuple(loop.open), reply_kind=kind)
    if kind == replies.WAIT and not waited:
        return Outcome(WAIT, question=wait_question(loop.question), request=loop.request,
                       missing=tuple(loop.open), reply_kind=kind, waited=1)
    if kind in (replies.MISUNDERSTOOD, replies.DIFFERENT):
        return Outcome(kind, missing=tuple(loop.open), reply_kind=kind)
    if kind == REPLY_FREE and not loop.confirm and extract is not None and str(reply or "").strip():
        loop.replies.append(str(reply).strip())
        try:
            found = extract(loop.text()) or {}
        except Exception:  # noqa: BLE001 - an extractor that fails read nothing: the reply made no progress
            found = {}
        values = {name: str(found[name]) for name in loop.open
                  if not planning_commands._empty(planning_commands._given_or_none(found.get(name)))}
    if values:
        loop._keep(values)
    gaps = planning_commands.argument_gaps(command, {name: loop.values.get(name) for name in names},
                                           loop.text()).from_earlier(not_asked)
    _take_gaps(loop, gaps)
    # A confirmation takes only what it can read exactly; free text there is never a reason to run.
    unreadable = kind in (REPLY_FREE, replies.WAIT) and (loop.confirm or not values)
    if not gaps.names and not unreadable:
        # The count of replies with no progress is left as it is: the part's own checks have the last word, and when
        # they still refuse something ``reopen`` measures this reply against what the last question asked.
        return Outcome(COMPLETE, reply_kind=kind)
    # A confirmation that had nothing open and now has (the reply set a value the checks refuse) asks its first
    # question about it; it is not a reply that made no progress.
    first_about_it = not before and not loop.stalled and kind == REPLY_VALUES
    progressed = not unreadable and (first_about_it or made_progress(before, gaps.names))
    outcome = _settle(loop, command, gaps, progressed, description=description, reply_kind=kind)
    return replace(outcome, waited=1) if kind == replies.WAIT else outcome
