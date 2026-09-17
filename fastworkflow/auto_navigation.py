"""Auto-navigation (ido-8ps.9): deterministic two-step dispatch on a foreign name.

R1 (ido-8ps.8) made a misroute loud: a command name this workflow really owns,
typed in a context that does not own it, is refused instead of being adjudicated
by that context's classifier, and the walk ends with a hint saying where the
command lives. 74c348a added the declaration the hint reads -- ``enter_command``
on a context's callback class.

This module decides what to do next. When the owning context can be entered
*without guessing*, the framework composes the two steps the agent would have
had to compose itself: enter the owning context via its declared entry command,
then run the original command. When it cannot, it blocks and says exactly what
it needs.

THE RULE (owner-approved, non-negotiable). Every automatic action is a pure
function of the UTTERANCE and the CONTEXT MODEL, never of history. Two-step
dispatch happens ONLY when

1. the owning context is stateless -- its declared entry command has no
   required parameters; or
2. the entry command's required parameters are present in the utterance itself,
   read by the same XML tag grammar the parameter extractor uses; or
3. the utterance carries an explicit handle -- an ``O`` alias or a value that
   entered a context earlier in this turn -- that resolves to a context instance
   recorded in the turn-scoped registry below.

Otherwise the turn BLOCKS with a clarification naming the owning context, the
entry command and the missing parameter. The clarification may list candidate
values read off the last few execute observations as a convenience; the
framework acts only on what the agent then supplies. Nothing here ever infers a
value from the action log, from what the agent did earlier, or from what a
previous turn did.

Why the registry is not history. ``decide`` is given the registry as an
argument, and the registry is consulted only for a handle the utterance itself
names. A handle the agent wrote is part of the utterance; the registry only says
which context instance that handle denotes, the same way the command inventory
says which context owns a name. No entry is ever selected because it is recent,
because it is the only one, or because a plan mentioned it.

Unconditional since ``ido-pyw.1``: a workflow that declares ``enter_command`` on
a context callback class gets two-step dispatch and the blocking clarification;
one that declares nothing gets the declaration-and-hint behaviour, because
``decide`` has no entry contract to act on. The declaration is the switch.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

#: How many of the most recent execute observations a clarification may read
#: candidate values off. Convenience only -- candidates are listed, never chosen,
#: so this is a presentation bound and not a policy.
CANDIDATE_STEPS = 3
#: Most candidate values one clarification may list.
CANDIDATE_MAX = 10

#: Class attributes a workflow's context callback class may declare to say which
#: command enters that context. THE canonical source for the fact: nothing in the
#: routing definition records which command sets the current context, and
#: inferring it from a command's NAME would bake one workflow's spelling
#: conventions into the framework. A context-model-file form was considered and
#: rejected in docs/auto_navigation.md -- two sources for one fact is one source
#: plus a drift.
#:
#: Syntax: ``enter_command = "<command_name>"`` or
#: ``enter_command = "<command_name> <param>value-shaped hint</param>"``. Only the
#: leading command name is load-bearing; the rest is hint text shown to the agent.
#: ``enter_commands`` takes a list when a context has more than one entry command,
#: in which case dispatch declines and the hint names them all.
CONTEXT_ENTER_COMMAND_ATTRS = ("enter_command", "enter_commands")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

RULE_STATELESS = 1
RULE_UTTERANCE_PARAMETERS = 2
RULE_EXPLICIT_HANDLE = 3

#: Decision kinds. ``NONE`` is "auto-navigation had nothing to say", which
#: leaves the R1 hint exactly as it was.
DISPATCH = "dispatch"
CLARIFY = "clarify"
NONE = "none"

#: Span attributes. ``auto_navigated`` marks the two execute steps a dispatch
#: produced; the decision and its rule go on the routing event whether or not
#: anything was dispatched, so a run's routing is readable from the trace.
ATTR_AUTO_NAVIGATED = "auto_navigated"
ATTR_AUTO_NAVIGATION_RULE = "auto_navigation_rule"
ATTR_ENTERED_CONTEXT = "entered_context"
ATTR_AUTO_NAVIGATION_DECISION = "auto_navigation_decision"
ATTR_AUTO_NAVIGATION_STEP = "auto_navigation_step"

#: The two steps a dispatch produces, named on their own spans.
STEP_ENTRY = "entry"
STEP_ORIGINAL = "original"

#: Artifact key carrying a dispatch plan from the CME wildcard command to
#: `CommandExecutor.invoke_command`, the only frame that can run the two steps
#: through the ordinary command path. Declared here rather than in either of
#: them because `wildcard` imports `command_executor`.
AUTO_NAVIGATION_ARTIFACT = "auto_navigation_plan"

#: Reasons a decision came out the way it did. One vocabulary, so a summary can
#: count them without parsing prose.
REASON_NOT_A_KNOWN_NAME = "not_a_known_name"
REASON_NO_ENTRY_DECLARATION = "no_entry_declaration"
REASON_SEVERAL_OWNING_CONTEXTS = "several_owning_contexts"
REASON_SEVERAL_ENTRY_COMMANDS = "several_entry_commands"
REASON_UNKNOWN_ENTRY_COMMAND = "unknown_entry_command"
REASON_DISPATCH_IN_FLIGHT = "dispatch_in_flight"
REASON_STATELESS = "stateless_entry_command"
REASON_PARAMETERS_IN_UTTERANCE = "parameters_in_utterance"
REASON_HANDLE_RESOLVED = "handle_resolved"
REASON_MISSING_ENTRY_PARAMETERS = "missing_entry_parameters"
REASON_AMBIGUOUS_HANDLE = "ambiguous_handle"


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

_COMMAND_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def entry_command_name(declaration: str) -> Optional[str]:
    """The command name at the head of an ``enter_command`` declaration.

    ``"open_account_by_uid <account_uid>"`` -> ``"open_account_by_uid"``. The
    tail is hint text for the agent and never reaches the dispatcher: a
    declaration that carried real parameter VALUES would make navigation a
    function of the declaration rather than of the utterance.
    """
    if not isinstance(declaration, str):
        return None
    head = declaration.strip().split(" ", 1)[0].split("(", 1)[0].split("<", 1)[0]
    return head if _COMMAND_NAME_RE.match(head or "") else None


def declared_entry_commands(workflow_folderpath: str, context_name: str) -> list[str]:
    """The ``enter_command`` declarations *context_name* carries, verbatim.

    Read off the context's own callback class, the one canonical source. Empty
    when the workflow declares nothing, when the context has no callback class,
    or when loading it fails -- a hint that names the context alone is worth more
    than a failed turn, so nothing here is allowed to raise.
    """
    try:
        import fastworkflow

        app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
        context_class = app_crd.context_model.get_context_class(
            context_name, fastworkflow.ModuleType.CONTEXT_CLASS
        )
        for attribute in CONTEXT_ENTER_COMMAND_ATTRS:
            value = getattr(context_class, attribute, None)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            if isinstance(value, (list, tuple)) and value:
                return [str(v).strip() for v in value if str(v).strip()]
    except Exception as exc:  # noqa: BLE001 - a hint must not fail a turn
        logger.debug(
            "no enter_command declaration readable for context %r: %r",
            context_name, exc,
        )
    return []


@dataclass(frozen=True)
class EntryContract:
    """What entering one context costs, as the context model states it."""

    context: str
    #: The declaration verbatim, so a hint can quote what the workflow wrote.
    declaration: str
    command_name: str
    qualified_command_name: Optional[str]
    #: Contexts that own the entry command, from the routing definition.
    owner_contexts: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()

    @property
    def stateless(self) -> bool:
        """No required parameters: rule 1 applies and nothing has to be guessed."""
        return not self.required_parameters


def _command_owners(app_crd, simple_name: str) -> tuple[tuple[str, ...], Optional[str]]:
    """Contexts owning *simple_name*, and its qualified name."""
    owners: list[str] = []
    qualified: Optional[str] = None
    for context_name, qualified_names in app_crd.contexts.items():
        for name in qualified_names:
            if name.split("/")[-1] == simple_name:
                owners.append(context_name)
                qualified = qualified or name
                break
    return tuple(sorted(owners)), qualified


def _parameters_of(app_crd, qualified_command_name: Optional[str]):
    """``(required, optional)`` field names of a command's parameters model.

    A command with no parameters class has neither, which is what makes its
    context stateless. Failure to load one answers "unknown" by raising -- the
    callers that must not fail wrap this.
    """
    if not qualified_command_name:
        return (), ()
    import fastworkflow

    parameters_class = app_crd.get_command_class(
        qualified_command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )
    if parameters_class is None:
        return (), ()
    required: list[str] = []
    optional: list[str] = []
    for name, info in parameters_class.model_fields.items():
        (required if info.is_required() else optional).append(name)
    return tuple(required), tuple(optional)


def entry_contract_for(
    workflow_folderpath: str, context_name: str
) -> Optional[EntryContract]:
    """The entry contract *context_name* declares, or None.

    None covers every way a context can decline to be auto-entered: no
    declaration, a declaration that does not parse, more than one entry command
    (the framework will not pick between them), or a name the workflow does not
    own. Each of those is a real state and each leaves the R1 hint in place.
    """
    declarations = declared_entry_commands(workflow_folderpath, context_name)
    if len(declarations) != 1:
        return None
    declaration = declarations[0]
    command_name = entry_command_name(declaration)
    if not command_name:
        return None
    try:
        import fastworkflow

        app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
        owners, qualified = _command_owners(app_crd, command_name)
        if not owners:
            return None
        required, optional = _parameters_of(app_crd, qualified)
    except Exception as exc:  # noqa: BLE001 - never fail a turn over a contract
        logger.debug("entry contract for %r unreadable: %r", context_name, exc)
        return None
    return EntryContract(
        context=context_name,
        declaration=declaration,
        command_name=command_name,
        qualified_command_name=qualified,
        owner_contexts=owners,
        required_parameters=required,
        optional_parameters=optional,
    )


def entry_contracts(workflow_folderpath: str) -> dict[str, EntryContract]:
    """Every declared entry contract in the workflow, keyed by context."""
    contracts: dict[str, EntryContract] = {}
    try:
        import fastworkflow

        app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
        context_names = list(app_crd.contexts)
    except Exception as exc:  # noqa: BLE001
        logger.debug("no routing definition at %r: %r", workflow_folderpath, exc)
        return contracts
    for context_name in sorted(context_names):
        contract = entry_contract_for(workflow_folderpath, context_name)
        if contract is not None:
            contracts[context_name] = contract
    return contracts


# ---------------------------------------------------------------------------
# The turn-scoped registry (rule 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextEntry:
    """One context instance entered in this turn, and what entered it.

    Recorded by ``CommandExecutor.invoke_command`` whenever a command moved the
    current command context. It holds the context name, the parameter values the
    entering command was given, and the ``O`` alias of the execute step that did
    it -- the three things rule 3 needs to rebuild that entry exactly.

    It is never scanned for "the most recent X". It is a lookup table from a
    handle the AGENT wrote to the context instance that handle denotes.
    """

    sequence: int
    context: str
    command_name: str
    parameters: Mapping[str, str]
    alias: Optional[str]

    def handles(self) -> tuple[str, ...]:
        """Every token that denotes this entry: its alias and its values."""
        found: list[str] = []
        if self.alias:
            found.append(str(self.alias))
        found.extend(str(v) for v in self.parameters.values() if str(v).strip())
        return tuple(found)


_lock = threading.Lock()
_entries: dict[str, list[ContextEntry]] = {}
_sequence: dict[str, int] = {}


def record_context_entry(
    scope_id: str,
    *,
    context: str,
    command_name: str,
    parameters: Optional[Mapping[str, Any]] = None,
    alias: Optional[str] = None,
) -> ContextEntry:
    """Remember that *command_name* entered *context* in this turn."""
    values = {
        str(name): str(value)
        for name, value in (parameters or {}).items()
        if value is not None and str(value).strip()
    }
    with _lock:
        _sequence[scope_id] = _sequence.get(scope_id, 0) + 1
        entry = ContextEntry(
            sequence=_sequence[scope_id],
            context=str(context),
            command_name=str(command_name),
            parameters=values,
            alias=str(alias) if alias else None,
        )
        _entries.setdefault(scope_id, []).append(entry)
    return entry


def context_entries(scope_id: str) -> tuple[ContextEntry, ...]:
    with _lock:
        return tuple(_entries.get(scope_id, ()))


def reset_auto_navigation_state() -> None:
    """Drop every turn-scoped entry. Nothing here outlives the turn."""
    with _lock:
        _entries.clear()
        _sequence.clear()


def current_scope_id() -> str:
    """The turn this call belongs to, in the offloading runtime's own terms."""
    try:
        from fastworkflow.result_handles import current_scope

        return str(current_scope().scope_id)
    except Exception:  # noqa: BLE001
        return "unbound"


# ---------------------------------------------------------------------------
# Reading the utterance
# ---------------------------------------------------------------------------

#: The agentic parameter grammar, as `ParameterExtraction._extract_parameters_from_xml`
#: writes it. Rule 2 reads the utterance with the SAME grammar the extractor will
#: use on the dispatched command, so a rule-2 dispatch cannot be composed out of
#: values the extractor would then fail to find.
def xml_parameter(utterance: str, field_name: str) -> Optional[str]:
    match = re.search(
        rf"<{re.escape(field_name)}>(.+?)</{re.escape(field_name)}>",
        utterance or "",
        re.DOTALL,
    )
    return match[1].strip() if match else None


_ALIAS_TOKEN_RE = re.compile(r"^[OD][1-9]\d*$")
_TOKEN_SPLIT_RE = re.compile(r"[\s,;]+")


def utterance_tokens(utterance: str) -> tuple[str, ...]:
    """Every token of *utterance* that could be an explicit handle.

    XML tag VALUES and bare words, in the order they appear, tag names dropped.
    Deliberately syntactic: a handle is something the agent wrote down, and the
    only question this asks is which of the things it wrote the registry knows.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().strip("`'\"<>[](){}.,;:")
        if token and token not in seen:
            seen.add(token)
            found.append(token)

    text = utterance or ""
    for value in re.findall(r"<[^<>/]+>(.*?)</[^<>]+>", text, re.DOTALL):
        add(value)
    stripped = re.sub(r"<[^<>]*>", " ", text)
    for token in _TOKEN_SPLIT_RE.split(stripped):
        add(token)
    return tuple(found)


def _is_alias(token: str) -> bool:
    return bool(_ALIAS_TOKEN_RE.match(token.upper()))


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoNavigationDecision:
    """What auto-navigation decided, and why, in terms a summary can count."""

    kind: str
    reason: str
    command_name: str
    rule: Optional[int] = None
    entered_context: Optional[str] = None
    entry_command: Optional[str] = None
    entry_utterance: Optional[str] = None
    entry_parameters: Mapping[str, str] = field(default_factory=dict)
    missing_parameters: tuple[str, ...] = ()
    handle: Optional[str] = None
    owner_contexts: tuple[str, ...] = ()

    @property
    def dispatches(self) -> bool:
        return self.kind == DISPATCH

    def event(self) -> dict[str, Any]:
        """The routing-event payload, filed for every declined KNOWN name."""
        return {
            "kind": "auto_navigation",
            ATTR_AUTO_NAVIGATION_DECISION: self.kind,
            "reason": self.reason,
            "command_name": self.command_name,
            ATTR_AUTO_NAVIGATION_RULE: self.rule,
            ATTR_ENTERED_CONTEXT: self.entered_context,
            "entry_command": self.entry_command,
            "missing_parameters": list(self.missing_parameters),
            "handle": self.handle,
            "owner_contexts": list(self.owner_contexts),
        }


def _entry_utterance(command_name: str, values: Mapping[str, str]) -> str:
    """The entry command, written the way the agent would have had to write it."""
    tail = "".join(
        f" <{name}>{values[name]}</{name}>" for name in sorted(values)
    )
    return f"{command_name}{tail}"


def decide(
    *,
    command_name: str,
    utterance: str,
    owner_contexts: Sequence[str],
    contracts: Mapping[str, EntryContract],
    entries: Sequence[ContextEntry] = (),
) -> AutoNavigationDecision:
    """Rule 1, 2, 3 or a blocking clarification. A pure function.

    Inputs are the UTTERANCE (``utterance``, ``command_name``) and the CONTEXT
    MODEL (``owner_contexts``, ``contracts``), plus the registry that says which
    context instance a handle written IN the utterance denotes. Nothing else --
    no observations, no action log, no clock, no environment. That is what makes
    the 'never from history' property testable: hold these arguments fixed and
    the result cannot move.
    """
    owners = tuple(owner_contexts)
    nothing = AutoNavigationDecision(
        kind=NONE, reason=REASON_NOT_A_KNOWN_NAME,
        command_name=command_name, owner_contexts=owners,
    )
    if not owners:
        return nothing
    if len(owners) > 1:
        # Two contexts own the name and the framework has no ground to prefer
        # one. The R1 hint already names them all; that stays the answer.
        return AutoNavigationDecision(
            kind=NONE, reason=REASON_SEVERAL_OWNING_CONTEXTS,
            command_name=command_name, owner_contexts=owners,
        )

    owner = owners[0]
    contract = contracts.get(owner)
    if contract is None:
        return AutoNavigationDecision(
            kind=NONE, reason=REASON_NO_ENTRY_DECLARATION,
            command_name=command_name, owner_contexts=owners,
            entered_context=owner,
        )

    common = {
        "command_name": command_name,
        "owner_contexts": owners,
        "entered_context": owner,
        "entry_command": contract.command_name,
    }

    # Rule 1 -- stateless. Nothing to supply, so nothing can be guessed wrong.
    if contract.stateless:
        return AutoNavigationDecision(
            kind=DISPATCH, reason=REASON_STATELESS, rule=RULE_STATELESS,
            entry_utterance=_entry_utterance(contract.command_name, {}),
            entry_parameters={}, **common,
        )

    # Rule 2 -- the agent wrote the values in the utterance it just sent.
    from_utterance = {
        name: value
        for name in contract.required_parameters
        if (value := xml_parameter(utterance, name)) is not None
    }
    if len(from_utterance) == len(contract.required_parameters):
        return AutoNavigationDecision(
            kind=DISPATCH, reason=REASON_PARAMETERS_IN_UTTERANCE,
            rule=RULE_UTTERANCE_PARAMETERS,
            entry_utterance=_entry_utterance(contract.command_name, from_utterance),
            entry_parameters=from_utterance, **common,
        )

    # Rule 3 -- the agent named a handle, and the registry says what it denotes.
    # The leading token is the command name, not a handle: a workflow whose uid
    # happened to spell a command name would otherwise resolve every call of
    # that command to it.
    tokens = utterance_tokens(utterance.split(" ", 1)[1] if " " in utterance else "")
    matched: list[tuple[str, ContextEntry]] = []
    for token in tokens:
        for entry in entries:
            if entry.context != owner:
                continue
            if not all(name in entry.parameters for name in contract.required_parameters):
                continue
            aliases = {h.upper() if _is_alias(h) else h for h in entry.handles()}
            probe = token.upper() if _is_alias(token) else token
            if probe in aliases:
                matched.append((token, entry))
    distinct = {
        tuple(sorted(entry.parameters.items())) for _, entry in matched
    }
    if len(distinct) == 1:
        token, entry = matched[0]
        values = {
            name: entry.parameters[name] for name in contract.required_parameters
        }
        return AutoNavigationDecision(
            kind=DISPATCH, reason=REASON_HANDLE_RESOLVED, rule=RULE_EXPLICIT_HANDLE,
            entry_utterance=_entry_utterance(contract.command_name, values),
            entry_parameters=values, handle=token, **common,
        )

    missing = tuple(
        name for name in contract.required_parameters if name not in from_utterance
    )
    reason = REASON_AMBIGUOUS_HANDLE if len(distinct) > 1 else REASON_MISSING_ENTRY_PARAMETERS
    return AutoNavigationDecision(
        kind=CLARIFY, reason=reason, missing_parameters=missing, **common,
    )


# ---------------------------------------------------------------------------
# The impure wrapper: gather the model, then decide
# ---------------------------------------------------------------------------

#: Depth guard. A dispatch runs two more commands through the ordinary path, and
#: either of them could decline a foreign name of its own. One level is all the
#: rule describes, so the inner steps never auto-navigate again.
_dispatch_depth = threading.local()


def dispatch_in_flight() -> bool:
    return bool(getattr(_dispatch_depth, "value", 0))


class dispatching:
    """Context manager marking the two dispatched steps as inner steps."""

    def __enter__(self):
        _dispatch_depth.value = getattr(_dispatch_depth, "value", 0) + 1
        return self

    def __exit__(self, *exc_info):
        _dispatch_depth.value = max(0, getattr(_dispatch_depth, "value", 0) - 1)
        return False


def plan(
    workflow_folderpath: str,
    *,
    command_name: str,
    utterance: str,
    owner_contexts: Sequence[str],
    scope_id: Optional[str] = None,
) -> AutoNavigationDecision:
    """``decide`` with the context model and registry read from the runtime."""
    owners = tuple(owner_contexts or ())
    if dispatch_in_flight():
        return AutoNavigationDecision(
            kind=NONE, reason=REASON_DISPATCH_IN_FLIGHT,
            command_name=command_name, owner_contexts=owners,
        )
    contracts: dict[str, EntryContract] = {}
    for owner in owners:
        contract = entry_contract_for(workflow_folderpath, owner)
        if contract is not None:
            contracts[owner] = contract
    entries = context_entries(scope_id or current_scope_id())
    return decide(
        command_name=command_name,
        utterance=utterance,
        owner_contexts=owners,
        contracts=contracts,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# The clarification and its candidates
# ---------------------------------------------------------------------------

#: How the framework's own listing observations render a row: identifier, two
#: spaces, label (`result_handles.ROW_SEPARATOR`). Candidates are read off these
#: first because they are the shape a uid actually arrives in.
_ROW_RE = re.compile(r"^(\S+)  +(\S.*)$")
#: The fallback token shape when an observation carries no rows.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._:-]{3,}\b")


def recent_execute_observations(agent: Any = None, steps: Optional[int] = None) -> list[str]:
    """The text of the last *steps* execute observations, newest last.

    Read off the live ReAct trajectory. Used ONLY to list candidate values in a
    clarification; no decision in this module is given this text.
    """
    limit = CANDIDATE_STEPS if steps is None else steps
    if limit <= 0:
        return []
    try:
        from fastworkflow.result_handles import _current_agent

        agent = agent if agent is not None else _current_agent()
        trajectory = getattr(agent, "current_trajectory", None)
        if not isinstance(trajectory, Mapping):
            return []
        indexes = sorted(
            int(key.removeprefix("tool_name_"))
            for key in trajectory
            if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()
        )
        texts = [
            str(trajectory[f"observation_{index}"])
            for index in indexes
            if str(trajectory.get(f"tool_name_{index}") or "") == "execute_workflow_query"
            and isinstance(trajectory.get(f"observation_{index}"), str)
        ]
    except Exception as exc:  # noqa: BLE001 - a convenience must not fail a turn
        logger.debug("no trajectory to read candidates from: %r", exc)
        return []
    return texts[-limit:]


def candidate_values(observations: Sequence[str], *, limit: Optional[int] = None) -> list[str]:
    """Values the agent could plausibly mean, listed newest first.

    A convenience and nothing more. The framework never chooses one of these --
    it prints them and waits. That is the whole difference between this and
    inferring a parameter from the action log.
    """
    maximum = CANDIDATE_MAX if limit is None else limit
    if maximum <= 0:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            found.append(value)

    for text in reversed(list(observations)):
        rows = [
            match[1]
            for line in str(text).splitlines()
            if (match := _ROW_RE.match(line.rstrip()))
        ]
        for value in rows:
            add(value)
        if not rows:
            for value in _IDENTIFIER_RE.findall(str(text)):
                add(value)
        if len(found) >= maximum:
            break
    return found[:maximum]


def missing_information_errmsg() -> str:
    """The parameter extractor's own words for a missing parameter.

    Borrowed rather than invented so the blocking clarification reads as, and is
    matched by, the existing missing-parameter path: a summary that counts
    `MISSING_INFORMATION_ERRMSG` in responses counts this one too.
    """
    try:
        import fastworkflow

        value = fastworkflow.get_env_var(
            "MISSING_INFORMATION_ERRMSG", default="Missing parameter values: ")
    except Exception:  # noqa: BLE001
        value = "Missing parameter values: "
    return str(value or "Missing parameter values: ")


def clarification_text(
    decision: AutoNavigationDecision, candidates: Sequence[str] = ()
) -> str:
    """The blocking message: what is needed, from where, and what was seen.

    It names the owning context, the entry command and the missing parameter --
    the three facts the agent needs to compose the step itself. Candidates, when
    there are any, are listed as "seen recently" and nothing more: the framework
    acts on what the agent supplies next, never on this list.
    """
    missing = ", ".join(decision.missing_parameters) or "its required parameters"
    lines = [
        f"'{decision.command_name}' is a command of the {decision.entered_context} "
        f"context, which this context and its parents do not provide. Enter it "
        f"with '{decision.entry_command}' first.",
        f"{missing_information_errmsg()}{missing}",
        f"Send: {decision.entry_command} "
        + " ".join(
            f"<{name}>VALUE</{name}>" for name in
            (decision.missing_parameters or ("PARAMETER",))
        )
        + f", then '{decision.command_name}'.",
    ]
    if decision.reason == REASON_AMBIGUOUS_HANDLE:
        lines.insert(
            1,
            "More than one recorded entry matches the handle in your request, so "
            "none was used.",
        )
    if candidates:
        lines.append(
            "Values seen in recent observations (listed, not chosen; supply the "
            "one you mean): " + ", ".join(candidates)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The validator (part a), offline
# ---------------------------------------------------------------------------

ISSUE_UNPARSEABLE = "unparseable_declaration"
ISSUE_SEVERAL_DECLARATIONS = "several_declarations"
ISSUE_UNKNOWN_COMMAND = "unknown_command"
ISSUE_UNREACHABLE_OWNER = "unreachable_owner"
ISSUE_NO_HIERARCHY = "no_context_hierarchy"


@dataclass(frozen=True)
class EntryContractIssue:
    context: str
    code: str
    detail: str
    fatal: bool = True


@dataclass(frozen=True)
class EntryContractReport:
    """What a workflow's entry declarations say, and what is wrong with them."""

    workflow_folderpath: str
    contracts: Mapping[str, EntryContract]
    issues: tuple[EntryContractIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(issue.fatal for issue in self.issues)

    @property
    def stateless_contexts(self) -> tuple[str, ...]:
        return tuple(sorted(c for c, k in self.contracts.items() if k.stateless))

    @property
    def parameterised_contexts(self) -> dict[str, tuple[str, ...]]:
        return {
            context: contract.required_parameters
            for context, contract in sorted(self.contracts.items())
            if not contract.stateless
        }

    def render(self) -> str:
        lines = [f"entry contracts for {self.workflow_folderpath}"]
        if not self.contracts and not self.issues:
            lines.append("  (no context declares enter_command)")
        for context, contract in sorted(self.contracts.items()):
            kind = "stateless" if contract.stateless else (
                "parameterised: " + ", ".join(contract.required_parameters)
            )
            lines.append(f"  {context}: {contract.declaration!r} -> {kind}")
        for issue in self.issues:
            mark = "ERROR" if issue.fatal else "warn "
            lines.append(f"  {mark} {issue.context}: {issue.code}: {issue.detail}")
        return "\n".join(lines)


def validate_entry_contracts(workflow_folderpath: str) -> EntryContractReport:
    """Check every ``enter_command`` declaration against the context model.

    Offline: it reads the routing definition, the context hierarchy and the
    commands' parameter models. No model, no backend, no trained artifact.

    Checks, per declaring context C:

    * the declaration parses to a command name;
    * exactly one entry command is declared (dispatch will not choose);
    * the command exists in this workflow;
    * it is owned by a context from which C is reachable -- the root ``*``
      (available everywhere through the walk), C itself, or an ancestor of C in
      ``context_hierarchy_model.json``. A workflow with no hierarchy file cannot
      be checked this way, which is reported as a warning rather than asserted
      as a pass.

    It also classifies each declaring context stateless or parameterised from
    the entry command's required parameters -- the same classification rule 1
    applies at runtime, computed from the same place.
    """
    import fastworkflow

    contracts: dict[str, EntryContract] = {}
    issues: list[EntryContractIssue] = []
    app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    context_model = app_crd.context_model
    hierarchy = context_model._load_context_hierarchy()

    for context_name in sorted(app_crd.contexts):
        declarations = declared_entry_commands(workflow_folderpath, context_name)
        if not declarations:
            continue
        if len(declarations) > 1:
            issues.append(EntryContractIssue(
                context_name, ISSUE_SEVERAL_DECLARATIONS,
                f"{len(declarations)} entry commands declared "
                f"({', '.join(declarations)}); dispatch will not choose between them",
            ))
            continue
        declaration = declarations[0]
        command_name = entry_command_name(declaration)
        if not command_name:
            issues.append(EntryContractIssue(
                context_name, ISSUE_UNPARSEABLE,
                f"{declaration!r} does not begin with a command name",
            ))
            continue
        owners, qualified = _command_owners(app_crd, command_name)
        if not owners:
            issues.append(EntryContractIssue(
                context_name, ISSUE_UNKNOWN_COMMAND,
                f"'{command_name}' is not a command of this workflow",
            ))
            continue
        try:
            required, optional = _parameters_of(app_crd, qualified)
        except Exception as exc:  # noqa: BLE001
            required, optional = (), ()
            issues.append(EntryContractIssue(
                context_name, ISSUE_UNKNOWN_COMMAND,
                f"parameters of '{command_name}' could not be read: {exc!r}",
            ))
        contracts[context_name] = EntryContract(
            context=context_name, declaration=declaration,
            command_name=command_name, qualified_command_name=qualified,
            owner_contexts=owners, required_parameters=required,
            optional_parameters=optional,
        )
        if not hierarchy:
            issues.append(EntryContractIssue(
                context_name, ISSUE_NO_HIERARCHY,
                "no context_hierarchy_model.json, so reachability of "
                f"'{command_name}' from a context that owns it is unchecked",
                fatal=False,
            ))
            continue
        ancestors = set(context_model.get_ancestor_contexts(context_name))
        reachable = {"*", context_name} | ancestors
        if not (set(owners) & reachable):
            issues.append(EntryContractIssue(
                context_name, ISSUE_UNREACHABLE_OWNER,
                f"'{command_name}' is owned by {', '.join(owners)}; none of those "
                f"is '*', '{context_name}' itself, or an ancestor of it "
                f"({', '.join(sorted(ancestors)) or 'none'})",
            ))

    return EntryContractReport(
        workflow_folderpath=str(workflow_folderpath),
        contracts=contracts,
        issues=tuple(issues),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m fastworkflow.auto_navigation <workflow_folderpath>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="fastworkflow.auto_navigation",
        description="Validate a workflow's enter_command declarations (offline).",
    )
    parser.add_argument("workflow_folderpath")
    args = parser.parse_args(argv)

    import fastworkflow

    fastworkflow.init(env_vars={})
    report = validate_entry_contracts(args.workflow_folderpath)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
