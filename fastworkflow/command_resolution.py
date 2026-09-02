"""Exact command identity, span parsing, and effective capability (arch §10).

Three defects, one layer (FW-REQ-003/004/005):

**A command the caller named exactly still goes through the model.** There is no
exact-identity path: a qualified input like ``Identity/show_properties`` is not
recognised as an identity at all, so it falls through to fuzzy matching and then
to the classifier — a model call, and a chance to be wrong, for text that named
its target unambiguously.

**A command that is not callable here becomes a misroute.** Nothing answers "you
named a real command, but not one you can call from where you are", so the
question the runtime asks instead is "which of the commands you *can* call did
you probably mean" — and it answers it.

**The identity information is thrown away before anyone can use it.** Both
``RoutingDefinition.contexts`` and ``CommandContextModel.commands()`` flatten
inherited commands onto simple names and keep one winner per name. By the time
those structures exist, "``show_properties`` is ambiguous between two bases" is
no longer expressible. So the index below is built from the raw definitions
(arch §10.1), and the flattened views stay exactly as they are — they are what
routing and training data use, and this slice does not touch either.

Nothing here grants execution authority. Resolution answers "what did this name
refer to"; the dispatcher rechecks authorization, preconditions and eligibility
immediately before every command, reads included (arch §10.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict

# The global surface, spelled the way the command tree spells it.
GLOBAL_CONTEXT = "*"

# Arch §10.5. P0 returns these; it does not navigate on any of them.
ResolutionFailure = Literal[
    "unknown-command",
    "not-callable-here",
    "missing-target-handle",
    "ambiguous-route",
    "unauthorized",
    "unsupported-dispatch",
]

# Where a definition came from, in precedence order (arch §10.1).
CapabilitySource = Literal["own", "inherited", "core"]

_SOURCE_RANK: dict[CapabilitySource, int] = {"own": 0, "inherited": 1, "core": 2}


class CommandDefinitionRef(BaseModel):
    """The stable identity of the implementation that owns a command (§6.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    definition_id: str
    contract_version: Optional[str] = None

    @property
    def owner_context(self) -> str:
        return self.definition_id.split("/")[0] if "/" in self.definition_id else GLOBAL_CONTEXT

    @property
    def simple_name(self) -> str:
        return self.definition_id.split("/")[-1]


class EffectiveCapability(BaseModel):
    """One command, as callable from one concrete occupiable context (§6.5).

    The pair that matters is (``definition``, ``effective_context_name``): the
    same definition is a different capability from every context that inherits
    it, and the display alias names the context you are actually in rather than
    the class that happens to implement it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    definition: CommandDefinitionRef
    effective_context_name: str
    display_alias: str
    simple_name: str
    source: CapabilitySource
    override_rank: int
    # The chain by which this context reached the definition. Empty for `own`.
    # Kept because "why is this command here?" is otherwise unanswerable once
    # inheritance has been resolved.
    inheritance_path: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        definition_id: str,
        effective_context_name: str,
        source: CapabilitySource,
        inheritance_path: tuple[str, ...] = (),
        depth: int = 0,
    ) -> "EffectiveCapability":
        definition = CommandDefinitionRef(definition_id=definition_id)
        simple_name = definition.simple_name
        alias_context = effective_context_name or GLOBAL_CONTEXT
        return cls(
            definition=definition,
            effective_context_name=alias_context,
            display_alias=f"{alias_context}/{simple_name}",
            simple_name=simple_name,
            source=source,
            # Lower wins. Source class first, then inheritance depth, so a
            # concrete own definition beats a shallow base which beats a deep
            # one which beats core (arch §10.1 precedence).
            override_rank=_SOURCE_RANK[source] * 1000 + depth,
            inheritance_path=inheritance_path,
        )


@dataclass(frozen=True)
class ExactResolution:
    """What a token referred to, or why it did not resolve.

    A resolution is never an authorization. `capability` says the name refers to
    something callable from here; whether it may be called is asked again, later,
    by the dispatcher.
    """

    token: str
    capability: Optional[EffectiveCapability] = None
    failure: Optional[ResolutionFailure] = None
    detail: str = ""
    # Populated for `ambiguous-route`: the candidates that tied, so the caller
    # can ask about them by name instead of guessing.
    candidates: tuple[EffectiveCapability, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.capability is not None

    @property
    def is_unknown(self) -> bool:
        """True when nothing recognised this token at all.

        The one case that may continue to fuzzy matching and the classifier
        (arch §10.3 step 6). Every other outcome is an answer, and passing it to
        a model would be asking a question that has already been settled.
        """
        return self.failure == "unknown-command"


# ----------------------------------------------------------------------
# Parsing (arch §10.2)
# ----------------------------------------------------------------------

# The leading token ends at whitespace or an opening parenthesis, matching what
# `intent_detection` has always treated as a command name.
_LEADING_TOKEN = re.compile(r"^\s*(?P<token>[^\s(]+)")


class ParsedCommand(BaseModel):
    """A command line split into its leading token and everything after it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: str
    command_token: str
    argument_text: str
    explicit_assistant_prefix: bool = False


def is_qualified_identity(token: str) -> bool:
    """Whether this token is a well-formed ``Context/name`` identity.

    A bare ``"/" in token`` test is not this question. `/add_two_numbers` is a
    slash-command prefix — the convention chat clients use — and treating it as
    a qualified identity made the runtime answer `not-callable-here` for a
    command that was perfectly callable, because the "context" it read was the
    empty string. Both halves have to be non-empty, and there is exactly one
    separator.
    """
    parts = (token or "").split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def parse_command(raw: str, *, assistant_prefix: str = "@") -> ParsedCommand:
    """Split off the leading command token, byte-for-byte.

    Replaces `command.replace(command_name, "")`, which removed **every**
    occurrence of the token anywhere in the line. A parameter whose value
    contains the command name — a search for `show_properties`, a tag named
    after a command — was silently mutilated, and the corruption was invisible
    because what came back was still a plausible-looking string.

    Only the first token's span is removed. The remainder is unchanged except
    for the whitespace that separated it from the token (arch §10.2).
    """
    text = raw or ""
    explicit_prefix = text.lstrip().startswith(assistant_prefix)
    if explicit_prefix:
        text = text.lstrip()[len(assistant_prefix):]

    match = _LEADING_TOKEN.match(text)
    if match is None:
        return ParsedCommand(
            raw=raw or "",
            command_token="",
            argument_text=text.strip(),
            explicit_assistant_prefix=explicit_prefix,
        )
    token = match.group("token")
    # `end()` of the token span, not `find(token)`: the same string may appear
    # later in the arguments and that occurrence is data.
    remainder = text[match.end():]
    return ParsedCommand(
        raw=raw or "",
        command_token=token,
        # lstrip only: trailing content is the caller's, and a parameter value
        # ending in whitespace is still that value.
        argument_text=remainder.lstrip(),
        explicit_assistant_prefix=explicit_prefix,
    )


def strip_command_token(raw: str, command_name: str) -> str:
    """Remove a *known* command name from the front of a line, once.

    The narrow replacement for the two `str.replace` sites. Matches the leading
    token against the command's simple name, its qualified name, or its display
    alias — and if the line does not start with any of them, returns the line
    unchanged rather than deleting a mid-string occurrence.
    """
    parsed = parse_command(raw)
    if not parsed.command_token:
        return (raw or "").strip()
    simple = command_name.split("/")[-1]
    token = parsed.command_token
    if token.lower() in {command_name.lower(), simple.lower()}:
        return parsed.argument_text
    return (raw or "").strip()


# ----------------------------------------------------------------------
# The capability index (arch §10.1)
# ----------------------------------------------------------------------


@dataclass
class CommandCapabilityIndex:
    """Every command, as callable from every context, with nothing flattened.

    Built from raw declarations — `{context: {"/": [...], "base": [...]}}` plus
    the framework's core command names — and never from
    `RoutingDefinition.contexts` or `CommandContextModel.commands()`, which have
    already picked a winner per simple name and discarded the collisions
    (arch §10.1).
    """

    # effective context -> simple name -> capabilities, best-ranked first.
    _by_context: dict[str, dict[str, list[EffectiveCapability]]] = field(
        default_factory=dict
    )
    # Every definition id the workflow has, for `is_known_identity`.
    _definitions: set[str] = field(default_factory=set)
    _occupiable: Optional[frozenset[str]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        raw_contexts: Mapping[str, Mapping[str, list[str]]],
        *,
        core_command_names: tuple[str, ...] = (),
        occupiable_contexts: Optional[frozenset[str]] = None,
    ) -> "CommandCapabilityIndex":
        index = cls(_occupiable=occupiable_contexts)
        for context_name in raw_contexts:
            index._build_context(context_name, raw_contexts)
        # Core/global definitions are callable everywhere, at the lowest
        # precedence: a workflow command of the same simple name overrides them
        # (arch §10.1 precedence 3).
        for qualified in core_command_names:
            index._definitions.add(qualified)
            for context_name in list(index._by_context) or [GLOBAL_CONTEXT]:
                index._add(
                    context_name,
                    EffectiveCapability.build(qualified, context_name, "core"),
                )
        return index

    def _build_context(
        self,
        context_name: str,
        raw_contexts: Mapping[str, Mapping[str, list[str]]],
        *,
        depth: int = 0,
        path: tuple[str, ...] = (),
        visiting: Optional[frozenset[str]] = None,
    ) -> None:
        visiting = visiting or frozenset()
        if context_name in visiting:
            # The context model already rejects cycles at load; refusing to
            # recurse here keeps a malformed hand-built index from hanging.
            return
        declaration = raw_contexts.get(context_name) or {}

        for qualified in declaration.get("/") or []:
            self._definitions.add(qualified)
            if depth == 0:
                self._add(
                    context_name,
                    EffectiveCapability.build(qualified, context_name, "own"),
                )
            else:
                self._add(
                    path[0],
                    EffectiveCapability.build(
                        qualified, path[0], "inherited", path[1:] + (context_name,), depth
                    ),
                )

        for base in declaration.get("base") or []:
            self._build_context(
                base,
                raw_contexts,
                depth=depth + 1,
                path=(context_name,) + path[1:] if depth == 0 else path,
                visiting=visiting | {context_name},
            )

    def _add(self, context_name: str, capability: EffectiveCapability) -> None:
        by_name = self._by_context.setdefault(context_name, {})
        bucket = by_name.setdefault(capability.simple_name, [])
        if any(
            existing.definition.definition_id == capability.definition.definition_id
            for existing in bucket
        ):
            return
        bucket.append(capability)
        bucket.sort(key=lambda cap: (cap.override_rank, cap.definition.definition_id))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def is_known_identity(self, token: str) -> bool:
        """Whether this token names a command that exists *somewhere*.

        The distinction between `unknown-command` and `not-callable-here`: only
        a token nothing recognises may go on to the classifier.
        """
        if token in self._definitions:
            return True
        simple = token.split("/")[-1]
        return any(
            definition.split("/")[-1] == simple for definition in self._definitions
        )

    def effective_capabilities(
        self, context_name: str
    ) -> tuple[EffectiveCapability, ...]:
        """Everything callable from this context, best-ranked winner per name."""
        by_name = self._by_context.get(context_name or GLOBAL_CONTEXT, {})
        return tuple(
            sorted(
                (bucket[0] for bucket in by_name.values() if bucket),
                key=lambda cap: cap.display_alias,
            )
        )

    def collisions(self, context_name: str, simple_name: str) -> tuple[EffectiveCapability, ...]:
        """Every candidate for a simple name here, including the losers."""
        return tuple(
            self._by_context.get(context_name or GLOBAL_CONTEXT, {}).get(simple_name, [])
        )

    def resolve_exact(self, token: str, current_context: str) -> ExactResolution:
        """Arch §10.3, in order. Only step 6 leaves this function unanswered.

        1. effective-context alias exact match;
        2. canonical definition ID exact match;
        3. unique simple name in the current effective capability set;
        4. known but inactive identity -> `not-callable-here`;
        5. ambiguous current simple name -> `ambiguous-route`;
        6. unknown text -> `unknown-command`, which is the caller's licence to
           continue to fuzzy matching and the classifier.
        """
        token = (token or "").strip()
        if not token:
            return ExactResolution(token=token, failure="unknown-command")

        context_name = current_context or GLOBAL_CONTEXT
        here = self._by_context.get(context_name, {})
        simple = token.split("/")[-1]
        bucket = here.get(simple, [])

        # 1. effective-context alias: `Identity/show_properties` while in Identity.
        if is_qualified_identity(token):
            alias_matches = [cap for cap in bucket if cap.display_alias == token]
            if alias_matches:
                # Arch §10.1 precedence 4: an equal-rank collision has no
                # simple-name alias and no unqualified answer. Two bases that
                # both define `find_thing` give the inheritor ONE alias and TWO
                # definitions, so resolving it to whichever sorted first would
                # be picking a winner the precedence rules deliberately refuse
                # to pick.
                best = alias_matches[0]
                tied = [
                    cap for cap in alias_matches if cap.override_rank == best.override_rank
                ]
                if len(tied) == 1:
                    return ExactResolution(token=token, capability=best)
                return ExactResolution(
                    token=token,
                    failure="ambiguous-route",
                    detail=(
                        f"alias '{token}' maps to "
                        + ", ".join(cap.definition.definition_id for cap in tied)
                        + " at equal precedence; name the definition"
                    ),
                    candidates=tuple(tied),
                )
            # 2. canonical definition id: `Resource/show_properties`, which is
            # callable here through inheritance even though the alias differs.
            for capability in bucket:
                if capability.definition.definition_id == token:
                    return ExactResolution(token=token, capability=capability)
            # A qualified name that names a real definition, but not one
            # reachable from here.
            if token in self._definitions or self.is_known_identity(token):
                return ExactResolution(
                    token=token,
                    failure="not-callable-here",
                    detail=f"'{token}' is not callable in context '{context_name}'",
                )
            return ExactResolution(token=token, failure="unknown-command")

        # 3/5. simple name in the current capability set.
        if bucket:
            if len(bucket) == 1 or bucket[0].override_rank < bucket[1].override_rank:
                return ExactResolution(token=token, capability=bucket[0])
            return ExactResolution(
                token=token,
                failure="ambiguous-route",
                detail=(
                    f"'{token}' is ambiguous in '{context_name}' between "
                    + ", ".join(
                        cap.definition.definition_id
                        for cap in bucket
                        if cap.override_rank == bucket[0].override_rank
                    )
                    + "; qualify it"
                ),
                candidates=tuple(
                    cap for cap in bucket if cap.override_rank == bucket[0].override_rank
                ),
            )

        # 4. known somewhere, not here.
        if self.is_known_identity(token):
            return ExactResolution(
                token=token,
                failure="not-callable-here",
                detail=f"'{token}' is not callable in context '{context_name}'",
            )

        # 6. genuinely unknown; the classifier's turn.
        return ExactResolution(token=token, failure="unknown-command")


# ----------------------------------------------------------------------
# The runtime entry point (arch §10.3)
# ----------------------------------------------------------------------

FEATURE_ID = "command_identity_v1"


def index_for_workflow(workflow_folderpath: str) -> Optional[CommandCapabilityIndex]:
    """The capability index for a workflow, or None when it cannot be built.

    None rather than an exception: resolution is an optimisation over the
    existing NLU path, and a workflow whose model will not load must fall
    through to that path rather than fail a turn.
    """
    try:
        from fastworkflow.command_context_model import CommandContextModel

        return CommandContextModel.load(workflow_folderpath).capability_index()
    except Exception:  # pragma: no cover - defensive; see docstring
        return None


def simple_name_resolution_enforced(workflow_folderpath: str) -> bool:
    """Whether bare simple names are resolved by identity as well.

    **Why this is gated and qualified-token resolution is not.**

    A qualified token (`Identity/show_properties`, `Resource/show_properties`)
    is unambiguously an identity: no natural-language utterance begins with one,
    so resolving it changes nothing a classifier was previously asked about, and
    the classifier-parity stop condition (requirements §12.3) is satisfied by
    construction.

    A bare simple name is not. `description` and `show_properties` are command
    names AND ordinary words, so answering `not-callable-here` for a leading
    word that happens to match a command in some other context would
    reinterpret natural-language utterances the classifier used to handle —
    exactly the prediction movement §12.3 forbids in a non-training slice.

    So the half of FW-REQ-003 that can move predictions rides the manifest
    feature gate the architecture provides for this (§7.1 dual gating): it runs
    only where a workflow declares `command_identity_v1` and the deployment
    enables that exact version. Every workflow has it `off` today, which is why
    this returns False and P0 keeps current behavior for bare names.
    """
    from fastworkflow.runtime_manifest import get_runtime_metadata

    metadata = get_runtime_metadata(workflow_folderpath)
    if metadata is None:
        return False
    return metadata.feature_mode(FEATURE_ID) == "enforce"
