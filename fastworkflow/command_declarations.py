"""Command class declarations: decorators on a command's ``ResponseGenerator`` (or ``PlanGenerator``).

A command states a fact about itself with a decorator, not a class attribute::

    @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)   # or REQUIRED_UNTIL_DELEGATED
    @writes                                                      # execution commands only
    @descends_to("Account")                                      # an unconditional move
    class ResponseGenerator:
        ...

Each decorator stores its value on a private attribute (``__fw_approval__``, ``__fw_writes__``,
``__fw_descends_to__``) together with a mark that only the decorator sets. :func:`declaration_problems` lists what
fails the load of the command (and of that command only, ``planning_commands.resolve_field_sources``): the retired
public attributes (``approval``, ``writes``, ``descends_to``, ``descend_parameter``), each naming its decorator, the
removed ``when_parameter`` option of ``@descends_to``, a private attribute written by hand without its decorator's
mark, ``@writes`` on a ``PlanGenerator``, a ``@descends_to`` target that is no context of the workflow, and an
``approval_template`` that cannot be filled.

This module imports nothing from the rest of fastworkflow, so the package and the planning module can
both re-export it.
"""
from __future__ import annotations

import enum
from typing import Any, Callable, Optional, TypeVar

_C = TypeVar("_C", bound=type)


class UserApproval(enum.Enum):
    """How a command needs the user's say before it runs (approval_annotations).

    ``REQUIRED_ALWAYS``: a person confirms every run, even a command they typed themselves; a standing
    ``always`` approval is neither offered nor honoured, so it can never be delegated.
    ``REQUIRED_UNTIL_DELEGATED``: the agent needs the user's say, and the user's own typed command or a
    standing ``always`` approval counts as that say. A command with no ``@user_approval_for_execution``
    is left to the agent, with no gate.
    """

    REQUIRED_ALWAYS = "required_always"
    REQUIRED_UNTIL_DELEGATED = "required_until_delegated"


class _DecoratorMark:
    """The value of a decorator's mark. Only this module creates one, so a hand-written mark is never it."""

    def __repr__(self) -> str:
        return "<set by a fastworkflow command decorator>"


_DECORATED = _DecoratorMark()

#: The private attributes the decorators store, and the mark each one is stored with.
APPROVAL_ATTRIBUTE = "__fw_approval__"
WRITES_ATTRIBUTE = "__fw_writes__"
DESCENDS_TO_ATTRIBUTE = "__fw_descends_to__"
_MARKS = {
    APPROVAL_ATTRIBUTE: "__fw_approval_decorated__",
    WRITES_ATTRIBUTE: "__fw_writes_decorated__",
    DESCENDS_TO_ATTRIBUTE: "__fw_descends_to_decorated__",
}
_DECORATOR_FOR = {
    APPROVAL_ATTRIBUTE: "@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)",
    WRITES_ATTRIBUTE: "@writes",
    DESCENDS_TO_ATTRIBUTE: '@descends_to("<Context>")',
}

#: Set by ``@descends_to`` when it is given the removed ``when_parameter`` option, so the load can name it.
RETIRED_WHEN_PARAMETER_ATTRIBUTE = "__fw_descends_to_when_parameter__"

_UNCONDITIONAL_MOVE = ("@descends_to always declares an unconditional move; declare it without the option "
                       "(a command that moves only sometimes still assigns workflow.record only when it moves)")

#: The public class attributes the decorators replace. Declaring one is a load error.
RETIRED_ATTRIBUTES = ("approval", "writes", "descends_to", "descend_parameter")

_APPROVAL_BY_RETIRED_VALUE = {
    "mandatory": "REQUIRED_ALWAYS",
    "delegatable": "REQUIRED_UNTIL_DELEGATED",
}


def _mark(command_class: _C, attribute: str, value: Any) -> _C:
    if not isinstance(command_class, type):
        raise TypeError(f"{_DECORATOR_FOR[attribute]} decorates a class (a ResponseGenerator or "
                        f"PlanGenerator), not {command_class!r}")
    setattr(command_class, attribute, value)
    setattr(command_class, _MARKS[attribute], _DECORATED)
    return command_class


def user_approval_for_execution(approval: UserApproval) -> Callable[[_C], _C]:
    """Gate a command's execution on the user's approval (``UserApproval.REQUIRED_ALWAYS`` or
    ``REQUIRED_UNTIL_DELEGATED``). Any other argument raises TypeError when the module is imported."""
    if not isinstance(approval, UserApproval):
        raise TypeError("@user_approval_for_execution takes a UserApproval member "
                        "(UserApproval.REQUIRED_ALWAYS or UserApproval.REQUIRED_UNTIL_DELEGATED), "
                        f"not {approval!r}")

    def decorate(command_class: _C) -> _C:
        return _mark(command_class, APPROVAL_ATTRIBUTE, approval)

    return decorate


def writes(command_class: _C) -> _C:
    """Declare that running an execution command's ``ResponseGenerator`` changes the application's data.

    A ``PlanGenerator`` cannot carry it: a planning command writes when a step of it runs a command that
    does. That is a load error of the planning command (:func:`declaration_problems`).
    """
    return _mark(command_class, WRITES_ATTRIBUTE, True)


_NOT_GIVEN: Any = object()


def descends_to(context: str, *, when_parameter: Any = _NOT_GIVEN) -> Callable[[_C], _C]:
    """Declare the context a command enters, whenever it runs. That the name is a non-empty string is checked when
    the module is imported; a name that is no context of the workflow fails the load of the command
    (:func:`declaration_problems`). ``when_parameter`` is removed: passing it is accepted here only so the load can
    fail naming it."""
    if not isinstance(context, str) or not context.strip():
        raise TypeError(f"@descends_to takes the name of a context, not {context!r}")
    target = context.strip()

    def decorate(command_class: _C) -> _C:
        marked = _mark(command_class, DESCENDS_TO_ATTRIBUTE, target)
        if when_parameter is not _NOT_GIVEN:
            setattr(marked, RETIRED_WHEN_PARAMETER_ATTRIBUTE, when_parameter)
        return marked

    return decorate


# ---------------------------------------------------------------------------
# Readers (getattr: a subclass inherits its base's declaration, as the old attributes did)
# ---------------------------------------------------------------------------

def declared_approval(command_class: Any) -> Optional[UserApproval]:
    """The command's ``UserApproval``, or None when it declares none (agent discretion)."""
    value = getattr(command_class, APPROVAL_ATTRIBUTE, None) if command_class is not None else None
    return value if isinstance(value, UserApproval) else None


def declares_writes(command_class: Any) -> bool:
    """Whether *command_class* carries ``@writes``."""
    return command_class is not None and getattr(command_class, WRITES_ATTRIBUTE, False) is True


def declared_descend(command_class: Any) -> Optional[str]:
    """The context named by ``@descends_to``, or None."""
    if command_class is None:
        return None
    value = getattr(command_class, DESCENDS_TO_ATTRIBUTE, None)
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# The load check: retired spellings, hand-written private attributes, targets and templates
# ---------------------------------------------------------------------------

def _q(text: Any) -> str:
    return f'"{text}"'


def _retired_problems(command_class: type) -> list[str]:
    own = vars(command_class)
    problems: list[str] = []
    if "approval" in own:
        value = own["approval"]
        decorator_value = _APPROVAL_BY_RETIRED_VALUE.get(value) if isinstance(value, str) else None
        if decorator_value is not None:
            fix = f"write @user_approval_for_execution(UserApproval.{decorator_value})"
        else:
            fix = ("write @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS) or "
                   "@user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)")
        shown = _q(value) if isinstance(value, str) else str(value)
        problems.append(f"approval = {shown} is removed: {fix}")
    if "writes" in own:
        problems.append(f"writes = {own['writes']} is removed: write @writes on the ResponseGenerator "
                        "(execution commands only)")
    if "descends_to" in own:
        target = own["descends_to"]
        shown = _q(target) if isinstance(target, str) else '"<Context>"'
        problems.append(f"descends_to = {shown} is removed: write @descends_to({shown}) on the ResponseGenerator")
    if "descend_parameter" in own:
        problems.append(f"descend_parameter = {_q(own['descend_parameter'])} is removed: {_UNCONDITIONAL_MOVE}")
    if RETIRED_WHEN_PARAMETER_ATTRIBUTE in own:
        problems.append(f"when_parameter is removed from @descends_to (it was given "
                        f"{own[RETIRED_WHEN_PARAMETER_ATTRIBUTE]!r}): {_UNCONDITIONAL_MOVE}")
    return problems


def _hand_written_problems(command_class: type) -> list[str]:
    own = vars(command_class)
    problems = []
    for attribute, mark in _MARKS.items():
        if attribute in own and own.get(mark) is not _DECORATED:
            problems.append(f"{attribute} is written by hand; declare it with "
                            f"{_DECORATOR_FOR[attribute]} instead")
    return problems


def _descends_to_problems(command_class: type, contexts: frozenset[str]) -> list[str]:
    target = declared_descend(command_class)
    if target is None or target in contexts:
        return []
    import difflib

    close = difflib.get_close_matches(target, sorted(contexts), n=1)
    hint = f"did you mean {_q(close[0])}?" if close else f"the workflow's contexts are {', '.join(sorted(contexts))}"
    return [f"@descends_to({_q(target)}) names no context of the workflow; {hint}"]


def _approval_template_problems(command_class: type, input_fields: frozenset[str]) -> list[str]:
    import string

    if "approval_template" not in dir(command_class):
        return []
    template = getattr(command_class, "approval_template")
    if not isinstance(template, str):
        return [f"approval_template must be a str (a str.format text over the Input fields); got {template!r}"]
    if declared_approval(command_class) is None:
        return ["approval_template is read only for a command that asks for approval: add "
                "@user_approval_for_execution(...), or remove approval_template"]
    try:
        names = [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]
    except ValueError as error:
        return [f"approval_template {template!r} is not a str.format text: {error}"]
    fields = ", ".join(sorted(input_fields)) or "none"
    return [f"approval_template {template!r} names {{{name}}}, which is not a field of Input; write a field "
            f"name in braces (its fields: {fields})"
            for name in names if name not in input_fields]


def declaration_problems(command_class: Any, *, is_planning: bool, contexts: frozenset[str],
                         input_fields: frozenset[str]) -> list[str]:
    """Why the declarations on a command's ``ResponseGenerator`` or ``PlanGenerator`` fail its load, one message
    per mistake, each saying what to write instead; empty when they are fine. *contexts* are the workflow's
    context names (a ``@descends_to`` target must be one), *input_fields* the names of the command's
    ``Signature.Input`` fields (an ``approval_template`` fills from them)."""
    if not isinstance(command_class, type):
        return []
    problems = [*_retired_problems(command_class), *_hand_written_problems(command_class)]
    if is_planning and WRITES_ATTRIBUTE in vars(command_class):
        problems.append("a planning command cannot carry @writes. It writes when its steps run a command that "
                        "carries @writes: put the decorator on that execution command")
    if is_planning and APPROVAL_ATTRIBUTE in vars(command_class):   # one rule for both marks (bd fix-ftdl.7.6)
        problems.append("a planning command cannot carry @user_approval_for_execution. Its steps are asked about "
                        "when they run a command that carries it: put the decorator on that execution command")
    problems += _descends_to_problems(command_class, contexts)
    problems += _approval_template_problems(command_class, input_fields)
    return problems
