"""Useful pointers to persisted observations, without payload-sized metadata."""
from __future__ import annotations

import math
import re

CHARS_PER_TOKEN = 4
OFFLOAD_MARK = "Use search_memory tool to search inside Observation "
LABEL_RE = re.compile(r"^Use search_memory tool to search inside Observation (O[1-9]\d*) returned by ")
#: The A1 handle line, with the ido-8ps.13 context clause optional. The clause
#: can never contain a parenthesis or a newline (``context_clause`` removes
#: both), so the closing ``)`` is unambiguous and a line printed before the
#: clause existed still matches.
ALIAS_LINE_RE = re.compile(
    r"^Observation (O[1-9]\d*) \(execute_workflow_query(?:, in ([^()\n]*))?\)\n")
#: Longest instance identity printed. A uid plus a display name, not a payload.
MAX_INSTANCE_LABEL_CHARS = 80
#: Longest context name printed, for the same reason.
MAX_CONTEXT_NAME_CHARS = 60
SEARCH_ANSWER_KEY_RE = re.compile(r"^(O[1-9]\d*)#a([1-9]\d*)$")


def search_answer_key(alias: str, sequence: int) -> str:
    r"""Archive key for one complete search answer, under the searched handle.

    Deliberately NOT an O alias. The agent-visible ``O`` namespace is execute
    ordinals only, and ``search_memory`` validates its ``alias`` argument
    against ``O[1-9]\d*``, so this key can never be passed back as a handle: a
    bounded answer's marking names a record, not a searchable observation. The
    searched alias is kept as the prefix so the archived answer is filed under
    the observation that produced it, and ``sequence`` separates repeated
    searches of the same observation within one scope.
    """
    if not sequence >= 1:
        raise ValueError("search answer sequence must be a positive integer")
    return f"{alias}#a{sequence}"


def is_search_answer_key(key: str) -> bool:
    return SEARCH_ANSWER_KEY_RE.match(key) is not None


def _clipped(value: str, limit: int) -> str:
    """*value* with no parenthesis, no newline, collapsed spaces, capped."""
    cleaned = " ".join(str(value or "").replace("(", " ").replace(")", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def context_clause(context_name: str, instance_label: str = "") -> str:
    """The ``<ContextName> <instance label>`` clause, or ``""``.

    The single place a context identity is made printable, so the format, the
    character rules the regex depends on and the length cap have one
    definition. An empty context name yields an empty clause (the root context
    prints no clause at all); a context with no declared identity yields its
    name alone -- an identifier is never invented to fill the gap.
    """
    name = _clipped(context_name, MAX_CONTEXT_NAME_CHARS)
    if not name:
        return ""
    label = _clipped(instance_label, MAX_INSTANCE_LABEL_CHARS)
    return f"{name} {label}" if label else name


def alias_line(alias: str, context: str = "") -> str:
    """The canonical handle line printed above an inline execute observation.

    This is the only identifier the agent is ever asked to pass to
    search_memory, so it must read the same here and in an offload label. It is
    presentation only: archived text never carries it (see strip_alias_line).

    ``context`` is the clause from ``context_clause`` -- the context the command
    RAN IN and, where the workflow declares one, that context's instance
    identity (``ido-8ps.13``). It is empty at the root context, and the line is
    then byte-for-byte what A1 printed.
    """
    clause = _clipped(context, MAX_CONTEXT_NAME_CHARS + MAX_INSTANCE_LABEL_CHARS + 1)
    suffix = f", in {clause}" if clause else ""
    return f"Observation {alias} (execute_workflow_query{suffix})\n"


def printed_alias(text: str) -> str | None:
    """The alias already printed on this observation, or None."""
    match = ALIAS_LINE_RE.match(text)
    return match.group(1) if match else None


def printed_context(text: str) -> str | None:
    """The context clause printed on this observation, or None.

    ``None`` both when there is no alias line and when the line carries no
    clause: a root-context observation and an unannotated one are separated by
    ``printed_alias``, not by this.
    """
    match = ALIAS_LINE_RE.match(text)
    if match is None:
        return None
    return match.group(2) or None


def strip_alias_line(text: str) -> str:
    """The original command response, without a printed alias line."""
    match = ALIAS_LINE_RE.match(text)
    return text[match.end():] if match else text


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def output_description(response: str) -> str:
    """Truthful fallback when a command has no authored Output description."""
    heading = next((line.strip() for line in response.splitlines() if line.strip()), "")
    return f"command output beginning with: {heading[:200]}" if heading else "an empty command result"


def offload_label(*, alias: str, command_name: str, response: str,
                  description: str = "") -> str:
    description = description.strip() or output_description(response)
    return (
        f"{OFFLOAD_MARK}{alias} returned by {command_name}. "
        f"It was offloaded to memory and contains {description.rstrip('.')}. "
    ).rstrip()


def is_offload_label(text: str) -> bool:
    return LABEL_RE.match(text) is not None


def label_alias(text: str) -> str | None:
    match = LABEL_RE.match(text)
    return match.group(1) if match else None


def replacement_saves_space(original: str, replacement: str) -> bool:
    return (len(replacement) < len(original)
            and len(replacement.encode('utf-8')) < len(original.encode('utf-8')))


def offload_saving_bytes(original: str, replacement: str) -> int:
    """UTF-8 bytes the trajectory loses by swapping a response for its label.

    This is the quantity offloading exists to buy, so it is what eligibility is
    decided on: a token estimate of the response alone cannot tell a 3 KB page
    worth replacing from a 300 B fact whose label is bigger than it is. Negative
    when the label is the larger of the two.
    """
    return len(original.encode("utf-8")) - len(replacement.encode("utf-8"))
