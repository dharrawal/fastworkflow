"""Answer a capability question ("what can you do?") in code, in agent mode.

When the app workflow has planning commands (``_commands/<name>.py``) and the whole user message asks
what the assistant can do, the turn is answered here, before the split and the planner,
with no model call: each planning command in plain words with its shortest ``plain_utterance`` as the
example, then the current context's commands, one line each, then one line for the commands
that more than half of the other contexts add, then one line per other context with the
commands only it (or a minority) adds. Each other-context line names commands in order while
it fits ``LINE_WIDTH`` characters (the shared line two lines' worth) and counts the rest
("and 3 more"); it always names at least one, and never leaves out just one (it names that
one too, past the width).

A capability question is one of the ``plain_utterances`` of the CME command
``IntentDetection/what_can_i_do``, up to minor wording deviations. Both sides are
normalised (``_words``: lowercase, contractions expanded, punctuation dropped, filler and
politeness words dropped, then a few plain rewrites such as "help me with" -> "do"), and the
message then matches an utterance when they have the same number of words and each word
pair is equal or one typo apart (``_close_word``). Matching word by word, rather than on
the whole string, keeps a message that adds a topic ("what can I do about the logins?")
from ever matching: it has more words than any utterance. The leading-window matcher of
``utils.fuzzy_match.find_best_matches`` does not fit here, because it scores a candidate
on its first ``len(message)`` characters, so "what can" would match "what can you do?".

Workflows without planning commands, the executor's ``what_can_i_do`` tool, the CME command itself
and assistant mode ('/' commands) are untouched.
"""
from __future__ import annotations

import functools
import re
from typing import Iterable, Optional

import Levenshtein
from pydantic import BaseModel

import fastworkflow
from fastworkflow.core.command_metadata_api import CommandMetadataAPI, _approval_marker

#: The CME command whose ``plain_utterances`` define a capability question.
WHAT_CAN_I_DO = "IntentDetection/what_can_i_do"

#: The characters an other-context line of the answer fits in (the shared line gets twice
#: that); the commands that do not fit are counted ("and 3 more"), unless only one is left.
LINE_WIDTH = 90

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "what're": "what are", "i'd": "i would",
    "i'm": "i am", "you're": "you are", "u": "you", "ya": "you", "r": "are",
    "we": "i", "our": "my", "could": "can", "pls": "please", "plz": "please",
}

#: Words that carry no topic, dropped wherever they stand.
_FILLER = frozenset(
    "please so hey hi hello ok okay well um uh oh then just actually exactly really again "
    "anyway right all else here the things stuff".split())

#: Phrases rewritten after the filler is dropped (the message is lowercase and unpunctuated).
_REWRITES = (
    (re.compile(r"^(?:can you tell me|tell me|i want to know|i would like to know) "), ""),
    (re.compile(r"\b(?:kinds?|sorts?) of\b"), ""),
    (re.compile(r"\bhelp(?: me)?(?: with)?\b"), "do"),
    (re.compile(r"\bfor me$"), ""),
    (re.compile(r"^what (you|i) can\b"), r"what can \1"),
)


def _words(text: str) -> tuple[str, ...]:
    """*text* as the words a capability question is compared on (see the module docstring)."""
    words = re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'")).split()
    words = " ".join(_CONTRACTIONS.get(word, word) for word in words).replace("'", "").split()
    text = " ".join(word for word in words if word not in _FILLER)
    for pattern, replacement in _REWRITES:
        text = pattern.sub(replacement, text).strip()
    return tuple(text.split())


def _close_word(typed: str, expected: str) -> bool:
    """Equal, or (for a word of 3+ letters) one edit or one swap of neighbours apart."""
    if typed == expected:
        return True
    if len(expected) < 3:
        return False
    if Levenshtein.distance(typed, expected) <= 1:
        return True
    diff = [i for i, (a, b) in enumerate(zip(typed, expected)) if a != b]
    return (len(typed) == len(expected) and len(diff) == 2 and diff[1] == diff[0] + 1
            and typed[diff[0]] == expected[diff[1]] and typed[diff[1]] == expected[diff[0]])


@functools.lru_cache(maxsize=8)
def _utterance_words(utterances: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(words for words in map(_words, utterances) if words)


def is_capability_question(message: str, utterances: Iterable[str]) -> bool:
    """True when the whole *message* is one of *utterances* up to minor wording deviations."""
    typed = _words(message)
    return bool(typed) and any(
        len(words) == len(typed) and all(map(_close_word, typed, words))
        for words in _utterance_words(tuple(utterances)))


def capability_utterances(cme_workflow_folderpath: str) -> tuple[str, ...]:
    """The ``plain_utterances`` of the CME command ``IntentDetection/what_can_i_do``."""
    crd = fastworkflow.RoutingRegistry.get_definition(cme_workflow_folderpath)
    return tuple(crd.command_directory.map_command_2_utterance_metadata[WHAT_CAN_I_DO].plain_utterances)


def _plain_name(name: str, capitalize: bool = True) -> str:
    text = name.replace("_", " ").strip().lower()
    text = text.capitalize() if capitalize else text
    return re.sub(r"\bi\b", "I", text)


def _plain_context_name(name: str) -> str:
    return _plain_name(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name))


def _line(prefix: str, plain: list[str], shown: int) -> str:
    rest = len(plain) - shown
    return prefix + ", ".join(plain[:shown]) + (f" and {rest} more" if rest else "")


def _shown_count(prefix: str, plain: list[str], width: int) -> int:
    """How many of *plain* the line names: as many in order as keep the whole line within *width*, at least one;
    when just one would be left out it is named too (never "and 1 more"), and the line may pass *width*."""
    shown = next((n for n in range(len(plain), 1, -1) if len(_line(prefix, plain, n)) <= width), min(1, len(plain)))
    return len(plain) if len(plain) - shown == 1 else shown


def _command_list(prefix: str, names: list[str], width: int) -> str:
    """*prefix* then *names* in plain words, as many in order as keep the whole line, with
    " and K more" for the rest, within *width* characters; at least one name. When just one
    would be left out it is named too (never "and 1 more"), and the line may pass *width*."""
    plain = [_plain_name(name, capitalize=False) for name in names]
    return _line(prefix, plain, _shown_count(prefix, plain, width))


def _context_groups(other_contexts: dict[str, list[str]]) -> list[tuple[str, Optional[str], list[str], int]]:
    """The lines of the other contexts as ``(prefix, context, names, width)``: first the commands that more than half
    of the other contexts add (``context`` None, twice the width), then one per context with the rest of its commands
    (a context left with none gets none). Order is kept."""
    seen: dict[str, int] = {}
    for names in other_contexts.values():
        for name in dict.fromkeys(names):
            seen[name] = seen.get(name, 0) + 1
    shared = [name for name, count in seen.items() if count * 2 > len(other_contexts)]
    groups = [("- In several places: ", None, shared, 2 * LINE_WIDTH)] if shared else []
    for context, names in other_contexts.items():
        own = [name for name in names if name not in shared]
        if own:
            groups.append((f"- In {_plain_context_name(context)}: ", context, own, LINE_WIDTH))
    return groups


def _context_lines(other_contexts: dict[str, list[str]]) -> list[str]:
    """One line (up to ``2 * LINE_WIDTH`` characters) for the commands that more than half of
    the other contexts add, then one line (up to ``LINE_WIDTH``) per context with the rest of
    its commands; a context left with none gets no line. Order is kept."""
    return [_command_list(prefix, names, width) for prefix, _context, names, width in _context_groups(other_contexts)]


def _planning_example(utterances: tuple[str, ...]) -> Optional[str]:
    examples = [u.strip() for u in utterances if u.strip()]
    return min(examples, key=len) if examples else None


def render_capability_answer(planning_commands: dict[str, tuple[str, ...]], commands: list[str],
                             other_contexts: Optional[dict[str, list[str]]] = None) -> str:
    """Markdown: each planning command with its shortest example request, then one line per command,
    then one line for the commands more than half of the other contexts add, then one line per
    other context with the rest of its commands (each of these lines cut to ``LINE_WIDTH``
    characters, the shared line to twice that, with the rest counted: "and 3 more", never
    "and 1 more").

    *planning_commands* maps a planning command's name to its ``plain_utterances`` (written as users talk, unlike
    the ``Signature`` docstring, which is written for the matcher; the shortest by length is
    shown, the first of equals); *commands* are the names of the current context's commands;
    *other_contexts* maps each other context's name to the names of the commands it adds
    (``CommandMetadataAPI.get_other_context_commands``).
    """
    lines = ["Here is what I can do. Ask in your own words, for example:", ""]
    for name, utterances in planning_commands.items():
        example = f': "{_planning_example(utterances)}"' if _planning_example(utterances) else ""
        lines.append(f"- **{_plain_name(name)}**{example}")
    lines += ["", "Or ask for a single action:", ""]
    lines += [f"- {_plain_name(name)}" for name in commands]
    lines += _context_lines(other_contexts or {})
    return "\n".join(lines)


def capability_data(planning_commands: dict[str, tuple[str, ...]], commands: list[str],
                    other_contexts: Optional[dict[str, list[str]]] = None) -> dict:
    """What ``render_capability_answer`` says, as data, from the same arguments and in the same order (fix-szfq).

    ``{"planning_commands": [{name, title, example}], "commands": [{name, title}], "other_contexts": [{context, title,
    commands: [{name, title}], more}]}``. ``name`` is the command's name as the workflow spells it, ``title`` the plain
    words the text shows, ``example`` the shortest example request the text quotes (None when it quotes none). An
    ``other_contexts`` group is one of the text's "In ..." lines: ``context`` its context name (None for the line of
    commands that more than half of the other contexts add, titled "In several places"), ``commands`` the commands the
    line names and ``more`` the number it counts ("and 3 more"). Nothing the text does not say."""
    def entry(name: str) -> dict:
        return {"name": name, "title": _plain_name(name, capitalize=False)}

    groups = []
    for prefix, context, names, width in _context_groups(other_contexts or {}):
        plain = [_plain_name(name, capitalize=False) for name in names]
        shown = _shown_count(prefix, plain, width)
        groups.append({"context": context, "title": prefix[len("- "):-len(": ")],
                       "commands": [entry(name) for name in names[:shown]], "more": len(names) - shown})
    return {
        "planning_commands": [{"name": name, "title": _plain_name(name), "example": _planning_example(utterances)}
                              for name, utterances in planning_commands.items()],
        "commands": [{"name": name, "title": _plain_name(name)} for name in commands],
        "other_contexts": groups,
    }


def context_commands(app_workflow, cme_workflow_folderpath: str) -> list[str]:
    """The names of the app commands of the current context (no parameters, no docstrings:
    a command's docstring is written for the agent)."""
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=app_workflow.folderpath,
        cme_workflow_path=cme_workflow_folderpath,
        active_context_name=app_workflow._current_command_context_name,
        include_planning=False,
    )
    return [cmd["name"] for cmd in meta.get("commands", [])
            if not cmd["qualified_name"].startswith("IntentDetection/")]


def other_context_commands(app_workflow) -> dict[str, list[str]]:
    """The commands each other context adds, by context name; the CME navigation commands are
    in every context, so the current one already lists (and skips) them. Empty on failure."""
    try:
        return CommandMetadataAPI.get_other_context_commands(
            app_workflow.folderpath, app_workflow._current_command_context_name)
    except Exception:
        return {}


def capability_turn(message: str, app_workflow, cme_workflow_folderpath: str) -> Optional[tuple[str, dict]]:
    """``(answer, data)`` when *message* is a capability question and the workflow has planning commands, else None.
    *data* is ``capability_data`` of the very arguments the answer text is rendered from."""
    from fastworkflow.agent.workflow_agent import planning_command_utterances

    planning_commands = planning_command_utterances(app_workflow.folderpath)
    if not planning_commands or not is_capability_question(message, capability_utterances(cme_workflow_folderpath)):
        return None
    commands = context_commands(app_workflow, cme_workflow_folderpath)
    others = other_context_commands(app_workflow)
    return (render_capability_answer(planning_commands, commands, others),
            capability_data(planning_commands, commands, others))


def capability_answer(message: str, app_workflow, cme_workflow_folderpath: str) -> Optional[str]:
    """The answer to *message* when it is a capability question and the workflow has planning commands, else None."""
    turn = capability_turn(message, app_workflow, cme_workflow_folderpath)
    return None if turn is None else turn[0]


class CapabilityListing(BaseModel):
    """The listing ``IntentDetection/what_can_i_do`` writes as text, as data (fix-szfq).

    ``workflow``: ``{name, contexts, context_inheritance}`` as the "Workflow Definition:" block shows them (``global``
    for ``*``; only contexts with parents in ``context_inheritance``). ``context``: the display name of the current
    context. ``commands``: one entry per command the text lists, in its order, each ``{name, qualified_name, context,
    planning, approval_marker, description, parameters, outputs, sample_utterances}``: ``name`` without the approval
    marker the text appends to it (that is ``approval_marker``, "" for none), ``planning`` True for a planning
    command, ``description`` the docstring line the text prints ("" when the command has none), ``parameters``/``outputs`` lists of ``{name, type, description, examples, available_from}`` (a parameter
    also ``pattern`` in agent mode, as the text; an output's ``examples`` are in this data but not in the text, which
    renders only a parameter's, fix-xj0x), ``sample_utterances`` the first two."""
    workflow: dict = {}
    context: str = ""
    commands: list[dict] = []


def what_can_i_do_data(workflow, for_agents: bool) -> CapabilityListing:
    """The listing of the CME command ``what_can_i_do`` run on *workflow* (the CME workflow), from the metadata its
    text is rendered from (``CommandMetadataAPI.get_enhanced_command_info``, the same sort and filter)."""
    from fastworkflow.core.command_context_model import get_workflow_info

    app_workflow = workflow.context["app_workflow"]
    active = app_workflow._current_command_context_name
    info = get_workflow_info(app_workflow.folderpath)

    def shown(name: str) -> str:
        return "global" if name == "*" else name

    inheritance = {shown(ctx): [shown(p) for p in parents]
                   for ctx, parents in sorted((info.get("context_inheritance_model") or {}).items()) if parents}
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=app_workflow.folderpath, cme_workflow_path=workflow.folderpath,
        active_context_name=active)
    planning = set(fastworkflow.RoutingRegistry.get_definition(
        app_workflow.folderpath).get_planning_command_names(active))

    def fields(items: list[dict], with_pattern: bool = False) -> list[dict]:
        listed = []
        for item in items or []:
            entry = {"name": item.get("name", ""),
                     "type": CommandMetadataAPI._simplify_type_str(item.get("type", "")),
                     "description": item.get("description", "") or "",
                     "examples": list(item.get("examples", []) or []),
                     "available_from": item.get("available_from")}
            if with_pattern and (pattern := item.get("pattern")):
                entry["pattern"] = str(getattr(pattern, "pattern", pattern))
            listed.append(entry)
        return listed

    commands = []
    for cmd in sorted(meta.get("commands", []), key=lambda x: x.get("name", "")):
        qualified = cmd.get("qualified_name", "")
        doc = " ".join(str(cmd.get("doc_string") or "").split())
        marker = _approval_marker(app_workflow.folderpath, qualified)
        commands.append({
            "name": cmd["name"], "qualified_name": qualified, "context": shown(active),
            "planning": qualified in planning,
            "approval_marker": marker,
            "description": doc,
            "parameters": fields(cmd.get("inputs"), with_pattern=for_agents),
            "outputs": fields(cmd.get("outputs")),
            "sample_utterances": list(cmd.get("plain_utterances", []) or [])[:2],
        })
    return CapabilityListing(
        workflow={"name": info.get("workflow_name"),
                  "contexts": [shown(ctx) for ctx in info.get("available_contexts") or []],
                  "context_inheritance": inheritance},
        context=shown(active), commands=commands)
