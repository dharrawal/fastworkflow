"""Intent clarification: how a reply to "which of these is meant?" is read, and the question that asks it.

ONE reader (``read``, reached as ``intent_loop.read``) for a reply that picks one of a list of named options, on the plan
path and on the deterministic (CME) path alike (bd fix-o1lg S7): the command of a part whose decision was an ambiguity,
a planning command for a part that matched none, or the full command list of a request nothing matched. The question
is ONE function (``plan_choice_question``). Both are code only: no model is called, so the same reply always reads the
same way and a client that is itself an agent can rely on the forms the question names.

What the CME used to match for itself is options of this reader: a reply whose first word names a candidate returns the
pick AND the rest of the reply (``Reading.rest``; the CME uses it as that command's parameter text, the plan path does
not read it), the plain utterances of ``abort``, ``you_misunderstood`` and ``what_can_i_do`` are *aliases* (abort, none
of these, list the commands), a number picks, and a fuzzy tie is unclear (never the first of the tied).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from fastworkflow import no_agent_mode
from fastworkflow.utils.fuzzy_match import find_best_matches

from typing import NamedTuple

PICK = "pick"
NONE = "none"
ABORT = "abort"
UNCLEAR = "unclear"
#: A reply that asks for the list of commands (an alias, see ``read``): the CME runs ``what_can_i_do``.
LIST = "list"

#: The largest normalised edit distance at which a reply is taken as a misspelt name: the bound the CME uses for a
#: typed command name (``intent_detection._FUZZY_PREMATCH_MAX_DISTANCE``; a test holds the two equal). It is repeated
#: here because importing the CME module loads the model stack, which a reader of plain text must not need.
FUZZY_MAX_DISTANCE = 0.3
#: A reply shorter than this is never matched as a misspelt name. The matcher scores a candidate on its leading
#: characters only, so one or two letters match every name that begins with them at distance 0.
FUZZY_MIN_LENGTH = 3

#: The reply the question names for "none of these fits".
NONE_REPLY = "none"
_NONE_WORDS = frozenset({
    "none", "none of these", "none of them", "none of the above", "no", "n", "neither", "something else",
    "nothing", "other", "another one"})

QUESTION_KIND_PLAN_CHOICE = "plan_choice"
PLAN_CHOICE_ARTIFACT = "plan_choice"
#: Opens a question asked again after a reply that could not be read.
RETRY = "That did not name one of them. "


class Reading(NamedTuple):
    """How a reply read: ``kind`` (``pick``, ``none``, ``abort``, ``list`` or ``unclear``), the picked ``name``, and for
    a pick made by the reply's first word the ``rest`` of the reply (else "")."""

    kind: str
    name: Optional[str] = None
    rest: str = ""


def _leaf(name: str) -> str:
    return str(name).rpartition("/")[2]


def _humanised(name: str) -> str:
    """*name* as a person writes it: lower case, underscores and the context separator as spaces."""
    return " ".join(str(name).lower().replace("_", " ").replace("/", " ").split())


def _only(names: Iterable[str]) -> Optional[str]:
    """The one name of *names*, or None when there is none or more than one."""
    unique = list(dict.fromkeys(names))
    return unique[0] if len(unique) == 1 else None


def read(reply: str, candidates: Iterable[str], *, numbered: Optional[int] = None,
         aliases: Optional[Mapping[str, str]] = None) -> Reading:
    """How a reply to "which of *candidates* is meant?" reads. *candidates* are the names in the order the question
    listed them. THE reader, for every which-command question on either path.

    Read in this order, the first that applies:

    - the exact name or its leaf (the part after the last ``/``), without regard to case, with or without a leading
      ``/`` and surrounding quotes or backticks;
    - the humanised name (``book by title`` for ``Library/book_by_title``);
    - an *alias*: *aliases* maps a phrase (lower case) to ``abort``, ``none`` or ``list`` (the CME gives the plain
      utterances of ``abort``, ``you_misunderstood`` and ``what_can_i_do``); a candidate's name wins over an alias;
    - a 1-based number of a listed option (*numbered*: how many of the candidates the question numbered; all of them
      when not given). A number outside the list is unclear;
    - cancel and its synonyms (``no_agent_mode.declines``): abort;
    - ``yes`` when the question offered exactly one candidate (it is then a confirmation): that candidate;
    - ``none``, ``no``, ``neither``, ``something else`` and the like: none of these;
    - a first word that is a candidate's name, the rest of the reply kept in ``Reading.rest`` (``find_book 5``);
    - a misspelt name or alias, by the matcher and bound the CME uses for a typed command name. Two that match
      equally well are unclear: the reader never guesses between them.

    A leaf that two candidates share (the same command name in two contexts) picks neither; the full name does.
    The words for cancel and none are read before the fuzzy match because that match scores a candidate on its leading
    characters, so ``no`` would otherwise pick a name that begins with "no".
    """
    names = [str(name) for name in candidates if str(name)]
    text = str(reply or "").strip().strip("`'\"").strip()
    word = no_agent_mode._bare(text).lstrip("/")
    if not word:
        return Reading(UNCLEAR)

    exact = _only(name for name in names if word in (name.lower(), _leaf(name).lower()))
    if exact is not None:
        return Reading(PICK, exact)
    if any(word in (name.lower(), _leaf(name).lower()) for name in names):
        return Reading(UNCLEAR)   # a leaf two candidates share
    spoken = _humanised(word)
    humanised = [name for name in names if spoken in (_humanised(name), _humanised(_leaf(name)))]
    if humanised:
        picked = _only(humanised)
        return Reading(PICK, picked) if picked is not None else Reading(UNCLEAR)
    aliases = {str(key).lower(): value for key, value in (aliases or {}).items()}
    if word in aliases:
        return Reading(aliases[word])

    number = word.lstrip("#").rstrip(".)")
    if number.isdigit():
        index, listed = int(number), len(names) if numbered is None else min(int(numbered), len(names))
        return Reading(PICK, names[index - 1]) if 1 <= index <= listed else Reading(UNCLEAR)

    if no_agent_mode.declines(text):
        return Reading(ABORT)
    if len(names) == 1 and no_agent_mode.read_yes_no(text) is True:
        return Reading(PICK, names[0])
    if word in _NONE_WORDS and names:
        # "none of these" answers a question that listed candidates: with no candidates (the CME's check of a reply to a
        # parameter question, which passes only aliases) a bare no or none is a value for the parameter reader, not
        # "you misunderstood".
        return Reading(NONE)

    parts = text.split(None, 1)
    if len(parts) == 2:
        first = no_agent_mode._bare(parts[0]).lstrip("/")
        by_first = _only(name for name in names if first in (name.lower(), _leaf(name).lower()))
        if by_first is not None:
            return Reading(PICK, by_first, parts[1].strip())

    if len(word.replace(" ", "")) < FUZZY_MIN_LENGTH or not (names or aliases):
        return Reading(UNCLEAR)
    # The leaf and the full name are both offered to the matcher, as a person may type either. Aliases join them: the
    # closest of all wins, and two that tie are unclear.
    by_key: dict[str, list[tuple[str, Optional[str]]]] = {}
    for name in names:
        for key in (name, _leaf(name)):
            by_key.setdefault(key, []).append((PICK, name))
    for phrase, kind in aliases.items():
        by_key.setdefault(phrase, []).append((kind, None))
    matched, _ = find_best_matches(word.replace(" ", "_"), list(by_key), threshold=FUZZY_MAX_DISTANCE)
    targets = list(dict.fromkeys(target for key in matched for target in by_key[key]))
    if len(targets) == 1:
        return Reading(targets[0][0], targets[0][1])
    return Reading(UNCLEAR)


# ---- the which-command question, one text for every path ------------------------------------------------------------

def plan_choice_question(request: str, choices: list[dict[str, Any]], *, sub_task: Optional[int] = None,
                         listed: Optional[int] = None, retry: str = "", ambiguous: bool = False,
                         misunderstood: bool = False, for_agent: bool = False, listing_hint: bool = False) -> str:
    """The plain, self-sufficient question: which command should run this request. *choices* are
    ``no_agent_mode.available_plans`` entries (the closest first) for a part no planning command matched, or, with
    *ambiguous*, the candidates of a request more than one command fits (execution and planning commands), or, with
    *misunderstood*, every command of the current context (nothing fitted); the first *listed* are numbered with their
    descriptions and the count of the rest is said. *retry* opens a question asked again.

    One candidate is not a list: it is a confirmation of that one command, answered by its name, ``yes`` or cancel.
    *for_agent* is the wording an agent reads (it updates its command; the page never shows it); *listing_hint* adds
    that ``what can i do`` lists every command (the deterministic path, where the reply is read)."""
    listed = len(choices) if listed is None else min(listed, len(choices))
    subject = f"part {sub_task}" if sub_task is not None else "this request"
    lines = [f"{index}. {plan['name']}" + (f": {plan['description']}" if plan.get("description") else "")
             for index, plan in enumerate(choices[:listed], 1)]
    more = len(choices) - listed
    if more > 0:
        noun = "command" if ambiguous or misunderstood else "planning command"
        lines.append(f"and {more} more; the name of any {noun} available here can be given.")
    shown = {name for name in (str(plan["name"]).rpartition("/")[2] for plan in choices)}
    if ambiguous and len(choices) == 1:
        leaf = next(iter(shown))
        body = f"Did you mean the command '{leaf}'?"
        tail = f" Reply with its name ({leaf}) to run it, 'yes', or {no_agent_mode.DECLINE_REPLY}."
        return f"{retry}{body}{tail}" + ("" if for_agent else f" {no_agent_mode.ANSWER_ONLY_NOTE}")
    if misunderstood:
        opening = (f"I couldn't determine which available command matches {subject}: {request!r}. "
                   "If one of these applies, say which:\n")
    elif ambiguous:
        opening = (f"More than one command fits {subject}: {request!r}. "
                   + ("Choose the correct command name from these possible options and update your command:\n"
                      if for_agent else "Which one is meant?\n"))
    else:
        opening = f"No planning command matched {subject}: {request!r}. Which one should run it?\n"
    closing = (f"\nReply with a name or number, '{NONE_REPLY}' if none of these fits, or {no_agent_mode.DECLINE_REPLY}."
               + (" 'what can i do' lists every command." if listing_hint else "")
               + ("" if for_agent else f" {no_agent_mode.ANSWER_ONLY_NOTE}"))
    return f"{retry}{opening}" + "\n".join(lines) + closing


def plan_choice_payload(record: dict[str, Any]) -> dict[str, Any]:
    """The pending prompt's data (artifact ``plan_choice``, ``question_kind`` "plan_choice"): the same shape on every
    path. ``ambiguous`` says the choices are the candidates of a request more than one command fits (not a list of
    plans for a part none matched); ``misunderstood`` that nothing fitted and they are every command."""
    return {
        "request": record.get("request") or "",
        "sub_task": record.get("sub_task"),
        "choices": [{"name": plan["name"], "description": plan.get("description") or "",
                     "closest": bool(plan.get("closest"))} for plan in record.get("choices") or []],
        "none_reply": NONE_REPLY,
        "decline": no_agent_mode.DECLINE_REPLY,
        "ambiguous": bool(record.get("ambiguous")),
        "misunderstood": bool(record.get("misunderstood")),
    }
