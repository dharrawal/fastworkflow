"""The stages in order, with the ONE retry: fill -> check -> validate -> (one more fill) -> result (bd fix-qoyz step 5).

The sequence "fill the values, guard them, validate them, maybe read the request once more with a note, check again"
was written in four places (``ParameterExtraction._llm_extract``, ``parameter_extraction.extract_arguments``,
``planning_commands.typed_planning_call`` and ``planning_commands.reextracted_arguments``). It is written here once,
and each of those calls ``extract`` with its own stages and its own values for the few things on which they differ.

``extract`` is a driver. It owns the order of the stages, the decision whether to retry, the second fill and what
becomes of an error in it; the stages themselves are callables the caller supplies, and they are the one fill
(``fastworkflow.parameters.fill.fill``) and the existing guards and validators (``check.py`` and ``validate.py``
through ``argument_checks.check_extracted_values``, ``InputForParamExtraction.validate_parameters`` and
``planning_commands.checked_arguments``). Nothing here changes what a stage says; it decides only what happens
after a stage refuses.

The one rule for the retry, the same for every caller: ONE second fill, only when a guard or a validator refused a
value the model produced, or every answer failed to parse into the fields' types (``unparsed``: a value outside a
``Literal`` was produced and refused); never when a required value is simply missing (``KIND_MISSING``: nothing was
refused). The second fill reads the request with the refusal after it (``retry_note``); its values go through the
stages again (guard, ``settle``, validators). When the second fill raises, or its values are refused again, the
first answer stands (no raise) and the refusal is handled as any refusal: the caller reads ``final`` and, for the
guard, the values that are still refused are treated as not given. A caller classes a verdict "missing" from
structured information (a plan's guard code, the execution validator's diagnostics, ``Filled.unparsed``), never
from message text.

What still differs between the callers is only which stages they run: ``check`` (the guard, execution commands),
``validate`` (the validators; both kinds), ``first_note`` (``reextracted_arguments`` puts the phase's note on the
first fill) and ``retry_note`` None (no second fill: ``reextracted_arguments``).

This module imports no model stack at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastworkflow.utils.logging import logger

KIND_OK = "ok"
#: A guard or a validator refused a value the model produced (or a rule of the command refused the whole call).
KIND_REFUSED = "refused"
#: Nothing was refused: a required value is simply not there.
KIND_MISSING = "missing"

#: The refusal a retry's note carries when the extractor's answers could not be read into the fields' types.
UNPARSED_REFUSAL = "the answer did not fit the allowed values or types of the fields"


@dataclass(frozen=True)
class Verdict:
    """What a stage found. ``refusal`` is the text a retry's note carries; ``detail`` is whatever the stage's own
    result is (the guard's ``ExtractionCheck``, the validated arguments, the exception) for the caller to read."""

    kind: str = KIND_OK
    refusal: str = ""
    detail: Any = None

    @property
    def ok(self) -> bool:
        return self.kind == KIND_OK


@dataclass(frozen=True)
class Extracted:
    """What ``extract`` found. ``params`` is what the last fill returned that the stages judged; ``first`` is the
    verdict of the first fill (of the stage that refused it, else the last stage run); ``final`` is the last
    verdict; ``retried_at`` names the stage whose refusal made the second fill run ("check" or "validate") when it ran
    and its answer was judged, else it is empty."""

    params: Any
    first: Verdict
    final: Verdict
    retried_at: str = ""
    checked: Verdict = Verdict()

    @property
    def retried(self) -> bool:
        return bool(self.retried_at)


def _unparsed_as_refusal(verdict: Verdict) -> Verdict:
    """*verdict* of a fill whose answers could not be parsed: a refusal (a value was produced and its own types
    refused it), unless a stage already refused something."""
    return verdict if verdict.kind == KIND_REFUSED else Verdict(KIND_REFUSED, UNPARSED_REFUSAL, verdict.detail)


def extract(fill: Callable[[str, int], Any], *, check: Optional[Callable[[Any, int], Verdict]] = None,
            settle: Optional[Callable[[Any, Verdict], None]] = None,
            validate: Optional[Callable[[Any, int], Verdict]] = None, retry_note: Optional[str] = None,
            first_note: str = "", unparsed: Optional[Callable[[], bool]] = None) -> Extracted:
    """Fill, check, validate, and at most one more fill.

    ``fill(note, attempt)`` returns the values (attempt 1 gets *first_note*, attempt 2 the retry's note);
    ``check(values, attempt)`` and ``validate(values, attempt)`` return a ``Verdict`` (a stage that is None is
    skipped); ``settle(values, verdict)`` runs after the check stage is resolved, before validation (the CME writes
    the respellings and gives each refused value its not-given value there). *retry_note* (a template with
    ``{refusal}``; None: no second fill) is the one retry rule of the module doc: the first refusal of the check
    stage or, when it passed, of the validate stage, is read once more; a missing value never is.

    ``unparsed()`` says whether the fill just run gave nothing because its answers could not be parsed into the
    fields' types: the first verdict is then a refusal, not "missing", so it gets the one second pass like any other
    produced value that was refused."""
    params = fill(first_note, 1)
    unread = bool(unparsed and unparsed())
    state = {"used": False, "retried_at": ""}

    def second_pass(stage: str, refused: Verdict, judge: Callable[[Any, int], Verdict], current: Any):
        """``(params, verdict)`` after the one more fill; the first answer and verdict when it raises."""
        state["used"] = True
        try:
            again = fill(retry_note.format(refusal=refused.refusal), 2)
        except Exception as exc:  # noqa: BLE001 - the second fill is an extra: the first answer stands
            logger.warning(f"parameter extraction retry failed: {exc!r}")
            return current, refused
        state["retried_at"] = stage
        return again, judge(again, 2)

    def may_retry(verdict: Verdict) -> bool:
        return retry_note is not None and not state["used"] and not verdict.ok and verdict.kind != KIND_MISSING

    first = verdict = Verdict()
    checked = Verdict()
    if check is not None:
        verdict = first = check(params, 1)
        if unread and retry_note is not None:
            verdict = first = _unparsed_as_refusal(verdict)
        if may_retry(verdict):
            params, verdict = second_pass("check", verdict, check, params)
        checked = verdict
        if settle is not None:
            settle(params, verdict)
    if validate is not None:
        verdict = validate(params, 1)
        if unread and retry_note is not None and not state["used"]:
            verdict = _unparsed_as_refusal(verdict)
        if check is None or first.ok:
            first = verdict
        if may_retry(verdict):
            def judge(again: Any, attempt: int) -> Verdict:
                # the new values pass the guard and its settling before the validators see them
                if check is not None:
                    state["checked"] = recheck = check(again, attempt)
                    if settle is not None:
                        settle(again, recheck)
                return validate(again, attempt)
            params, verdict = second_pass("validate", verdict, judge, params)
            checked = state.get("checked", checked)
    return Extracted(params=params, first=first, final=verdict, retried_at=state["retried_at"], checked=checked)
