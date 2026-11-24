"""THE intent loop: which command a request means, and the question asked when that is not settled (bd fix-o1lg S7).

One loop for both paths. The plan path (``plan_metadata_extraction/resolution.py``, a part of a turn) and the deterministic path
(the command metadata extraction workflow, CME: ``CommandNamePrediction`` and the ``wildcard`` router) decide a request
text with ``resolve`` and read the reply to "which of these is meant?" with ``read`` / ``answer``. Nothing else decides
an intent, and nothing else matches a reply.

``resolve(workflow, text)`` is the whole-walk decision. Its layers, in order, each living once, here:

1. an exact command name on the lookup path (``context_navigation.exact_name_scopes``, nearest scope first, the global
   scope last): the request's first word, up to whitespace or ``(``, case-insensitively, with or without a leading ``/``;
2. the foreign-name guard: the first word names a command of this workflow that no scope on the path owns. No model is
   asked; the owning contexts and the routing hint (what to do to get there) are returned;
3. a fuzzy name (a misspelt command name): the whole request against the names of each scope, at the CME's bound
   (``intent_clarification.FUZZY_MAX_DISTANCE``). Two names that match equally well are NOT a match (they used to be
   decided by set order): the chain decides;
4. the local model over the whole walk first: a confident answer anywhere on the walk decides with no model call;
5. then the chain (``intent_detector.detect`` with ``engine_for``): the local model, then the remote decision model
   (Jev) when it is unsure and ``JEV_API_KEY`` is set, the global scope asked last when the walk did not ask it and
   everything declined. The remote model is never offered a command that only enters a context
   (``context_navigation.context_entry_commands``; the local model, exact names and fuzzy names still reach them);
6. a confident LOCAL answer that names a planning command is never acted on unreviewed: the remote model confirms it or
   the candidates are asked (``_second_opinion``);
7. a bare value (``parameter_value``) is reported as such (``Intent.bare_value``).

The product differences, the only parameters:

- ``core``: the CME may answer the framework's core commands (``what_can_i_do``, ``go_up``, ``reset_context``,
  ``what_is_current_context``, ``abort``, ``you_misunderstood``): by name, the plain utterances of
  ``you_misunderstood``, and as a confident answer or a candidate of the chain. A plan part may not (they are not
  something a plan runs), so with ``core`` False a core command is never a match and never a candidate.
- the callers of ``resolution.part_intent`` add ``pinned`` (a plan the client chose) and ``exclude`` (a review's
  look_up re-match); they are not part of the decision and live there.

The pending question is ONE piece of plain data, a JSON-safe ``dict`` made by ``open_question`` (both paths keep it in the turn's
waiting record, kind ``intent``; the plan path's ``owner`` is ``plan``): ``ask`` writes its text, ``answer``
reads a reply against it and counts the replies that could not be read (``no_agent_mode.MAX_NO_PROGRESS``). The client
shape is ``question_kind`` ``plan_choice`` with the artifact ``plan_choice`` (``payload``).

Nothing here imports the model stack at import time.
"""
from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow import intent_clarification, waiting

MATCH = "match"
AMBIGUOUS = "ambiguous"
NONE = "none"

KIND_EXECUTION = "execution"
KIND_PLANNING = "planning"
KIND_CORE = "core"

#: Who decided: a pinned plan, an exact command name, a fuzzy command name, a link of the detection chain, the user's
#: pick at a question, or nobody (the candidates are put to the user or the agent, ``intent_detector.ENGINE_QUESTION``).
BY_PINNED = "pinned"
BY_EXACT_NAME = "exact_name"
BY_FUZZY_NAME = "fuzzy_name"
BY_FOREIGN_NAME = "foreign_name"
BY_CLASSIFIER = "classifier"
BY_MATCHER = "matcher"
BY_QUESTION = "question"
BY_CHOICE = "choice"
BY_NOTHING = ""

#: The deciders a single part runs on with no review (the fast path): the signals the local side stands behind. A close
#: spelling of a command's name is one, as a confident local model is; neither is the user naming a command.
FAST_PATH_DECIDERS = frozenset({BY_EXACT_NAME, BY_FUZZY_NAME, BY_CLASSIFIER})

#: The deciders by which the user (or a client) named the command: its exact name, a pinned plan, a pick at the
#: which-command question (a client's direct call is not decided here at all). A close spelling of a name is NOT one: it
#: is the closest command, so a command that changes data found that way is confirmed like a model's choice. A command
#: that changes data or carries an approval mark runs only when it was named, or after a yes (``unconfirmed_pick``).
NAMED_DECIDERS = frozenset({BY_EXACT_NAME, BY_PINNED, BY_CHOICE})

#: The most other plans a drafted item names when its match was ambiguous.
MAX_ALTERNATIVES = 3

#: What a CME workflow context is waiting for (``waiting_for``).
WAITING_INTENT = "intent"
WAITING_PARAMETERS = "parameters"


@dataclass(frozen=True)
class Ranked:
    """One candidate command for a request: its qualified name, its kind and the score of the link that scored it."""

    name: str
    kind: str
    score: float = 0.0


@dataclass(frozen=True)
class Intent:
    """The decision for one request (``resolution.PartIntent`` is this).

    outcome: ``match`` (one command), ``ambiguous`` (several candidates, none confident) or ``none`` (nothing here
        serves it: an ad hoc part, a bare value, a foreign name).
    command, kind: the command of a match and whether it is an execution, planning or core command (None otherwise).
    candidates: every command of the lookup path a link scored, best first: the deciding link's ranking first (for an
        ambiguity, its candidates in its order), then the rest by score.
    decided_by: ``pinned``, ``exact_name``, ``fuzzy_name``, ``foreign_name``, ``classifier``, ``matcher``, ``question``
        (an ambiguity), ``choice`` (the user's pick) or "" (none).
    ambiguous: the qualified names an ambiguity asks about, in order.
    context, record: where the command lives: the context of the scope that answered and the record a command found
        there runs on (None for the global scope's root when it has none, and for an outcome that is not a match).
    hint, owners: a foreign name's routing hint and the contexts that own it.
    bare_value: the request is a value, not a command here or anywhere else (it ended the walk).
    evidence: what the chain knew when it decided (`intent_learning.evidence_of`, and the ``context`` the walk ended
        in): recorded with the settled command.
    answers: every engine answer of the walk, in the order asked (the span reads them); facts: the name layers' facts
        for the span (fuzzy distance, tie, owners).
    """

    outcome: str
    command: Optional[str] = None
    kind: Optional[str] = None
    candidates: tuple[Ranked, ...] = ()
    decided_by: str = BY_NOTHING
    ambiguous: tuple[str, ...] = field(default=())
    evidence: dict = field(default_factory=dict, compare=False)
    context: Optional[str] = None
    record: Any = field(default=None, compare=False)
    hint: Optional[str] = None
    owners: tuple[str, ...] = ()
    bare_value: bool = False
    answers: tuple = field(default=(), compare=False)
    facts: dict = field(default_factory=dict, compare=False)

    def planning_scores(self) -> dict[str, float]:
        """``{planning command: score}`` of the planning candidates, best first: what orders the plans an ad hoc part
        lists (``no_agent_mode.available_plans``) and what the run's listing gathers."""
        return {each.name: each.score for each in self.candidates if each.kind == KIND_PLANNING}

    @property
    def deciding(self):
        """The engine answer the decision stands on, or None: the last of the walk for a match or an ambiguity (the
        scope that answered), the first for a decision nobody made (the nearest scope's, which the walk declined)."""
        if not self.answers:
            return None
        return self.answers[0] if self.outcome == NONE else self.answers[-1]


# ======================================================================================================================
# Kinds, scopes and candidates
# ======================================================================================================================

def _kinds(workflow) -> tuple[set[str], set[str]]:
    """``(planning names, core names)`` of the workflow."""
    from fastworkflow import planning_commands
    from fastworkflow.workflow_agent import core_command_names

    return (set(planning_commands.resolve_field_sources(workflow.folderpath).planning_commands),
            set(core_command_names()))


def _offers(workflow, context: str) -> list[str]:
    from fastworkflow import intent_detector

    try:
        return [c.name for c in intent_detector.matcher_candidates(workflow.folderpath, context)]
    except ValueError:
        return []


def scope_path(workflow) -> tuple[tuple[str, ...], bool]:
    """``(the request's scope path, whether the global scope was added to it)``: the detection's own walk path, then the
    global scope when the walk does not ask it and it offers commands (a root-level planning command, a global command),
    asked last: nearest scope first, so it decides only a request no scope of the walk took."""
    from fastworkflow import intent_detector

    path = intent_detector.walk_path(workflow)
    if intent_detector.GLOBAL_CONTEXT not in path and _offers(workflow, intent_detector.GLOBAL_CONTEXT):
        return (*path, intent_detector.GLOBAL_CONTEXT), True
    return path, False


def _kind_of(name: str, planning: set[str], core_names: Iterable[str] = ()) -> str:
    return KIND_PLANNING if name in planning else KIND_CORE if name in core_names else KIND_EXECUTION


def _ranked(detection, planning: set[str], offered: set[str], deciding, core_names: set[str]) -> tuple[Ranked, ...]:
    """The candidates of a detection: the deciding answer's labels first, by its score, then the others' by score.
    Reserved labels, ``none`` and names off the scope path are left out."""
    from fastworkflow.intent_detector import MATCHER_NONE
    from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, is_escalation, label_of

    def scored(answer) -> list[tuple[str, float]]:
        pairs = [(str(label), float(score)) for label, score in answer.scores.items()]
        return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))

    answers = ([deciding] if deciding is not None else []) + [a for a in detection.answers if a is not deciding]
    out: dict[str, Ranked] = {}
    for answer in answers:
        for name, score in scored(answer):
            if (name in out or name == MATCHER_NONE or is_escalation(name)
                    or label_of(name) == PARAMETER_VALUE_LABEL or (name not in offered and name not in core_names)):
                continue
            out[name] = Ranked(name, _kind_of(name, planning, core_names), score)
    if deciding is None:
        return tuple(sorted(out.values(), key=lambda each: -each.score))
    first = [each for each in out.values() if each.name in deciding.scores]
    rest = sorted((each for each in out.values() if each.name not in deciding.scores), key=lambda each: -each.score)
    return tuple(first + rest)


def _entry_only(workflow) -> frozenset[str]:
    """The commands that only enter a context (``context_navigation.context_entry_commands``): never offered to the
    remote decision model as a request's answer. Empty when they cannot be read."""
    from fastworkflow import context_navigation

    try:
        return context_navigation.context_entry_commands(workflow.folderpath)
    except Exception:  # noqa: BLE001 - an unreadable declaration leaves nothing out
        return frozenset()


def _walk(workflow, text: str, engine_of, path, global_last: bool, detection=None):
    """``intent_detector.detect`` with *engine_of* (or the *detection* already made with the same engines), then the
    global scope, asked last, when the walk does not ask it and declined everywhere (a root-level planning command)."""
    from fastworkflow import intent_detector

    if detection is None:
        detection = intent_detector.detect(workflow, text, engine_of=engine_of)
    if detection.outcome != intent_detector.NONE or not global_last:
        return detection
    engine = engine_of(intent_detector.GLOBAL_CONTEXT)
    answer = engine.decide(text) if engine is not None else None
    if answer is None:
        return detection
    answers = [*detection.answers, answer]
    scope = intent_detector.Scope(intent_detector.GLOBAL_CONTEXT, workflow.root)
    if answer.kind == intent_detector.MATCH:
        return intent_detector.Detection(intent_detector.MATCH, command=answer.labels[0], scope=scope, answers=answers)
    if answer.kind == intent_detector.AMBIGUOUS:
        return intent_detector.Detection(intent_detector.AMBIGUOUS, candidates=list(answer.labels), scope=scope,
                                         answers=answers)
    return intent_detector.Detection(intent_detector.NONE, scope=detection.scope, answers=answers)


def _usable(names, excluded) -> list[str]:
    from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, is_escalation, label_of

    return [name for name in names
            if name not in excluded and not is_escalation(name) and label_of(name) != PARAMETER_VALUE_LABEL]


def _evidence(workflow, answers: Iterable, scope) -> dict:
    """The evidence of a decision (`intent_learning.evidence_of`) with the context the walk ended in."""
    from fastworkflow import intent_learning

    context = getattr(scope, "context_name", None) or workflow._current_command_context_name
    return {**intent_learning.evidence_of(answers), "context": context}


# ======================================================================================================================
# The name layers: exact, foreign, fuzzy
# ======================================================================================================================

def first_word(text: str) -> str:
    """The word a command name ends at: ANY whitespace or a ``(``, so ``list_orders\\n<scope>all</scope>`` does not make
    the whole line the name. "" for a blank text."""
    stripped = str(text or "").strip()
    return stripped.split(None, 1)[0].split("(", 1)[0] if stripped else ""


def _is_framework_planning(folderpath: str, qualified: str) -> bool:
    from fastworkflow.planning_commands import is_framework_planning_command, planning_command

    command = planning_command(folderpath, qualified)
    return command is not None and is_framework_planning_command(command)


def _scope_names(workflow, context_name: str, core_names: set[str]) -> list[str]:
    """The application commands a scope offers, without the core ones (they are the framework's) and without the
    framework's own planning commands (they are not app commands)."""
    from fastworkflow import context_navigation

    return [name for name in context_navigation.scope_command_names(workflow.folderpath, context_name)
            if name not in core_names and not _is_framework_planning(workflow.folderpath, name)]


def _current_scope(workflow):
    from fastworkflow import intent_detector

    return intent_detector.scopes_for(workflow)[0]


def command_inventory(workflow_folderpath: str) -> dict[str, tuple[str, ...]]:
    """Every command name this application owns (lower-cased leaf), and the contexts owning it. Reserved labels name no
    command and are left out. Read from the same routing definition the per-context candidate sets are built from;
    planning commands too, so a planning name owned by another context is guarded like a routed one."""
    import fastworkflow
    from fastworkflow.nlu_labels import is_non_routable

    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    owners: dict[str, set[str]] = {}
    for context_name, qualified_names in crd.contexts.items():
        for qualified_name in (*qualified_names, *crd.get_planning_command_names(context_name)):
            simple_name = qualified_name.split("/")[-1]
            if not is_non_routable(simple_name):
                owners.setdefault(simple_name.lower(), set()).add(context_name)
    return {name: tuple(sorted(contexts)) for name, contexts in owners.items()}


def foreign_context_hint(command_name: str, owner_contexts: list[str], enter_commands: dict[str, list[str]]) -> str:
    """What to say when a real command name is typed where no scope of the path can run it.

    The name IS a command, the runtime knows exactly which contexts have it, and the caller is one navigation away from
    being able to run it. A hint, never an action: this composes text and nothing here navigates, because
    auto-navigating on a misrouted call would change the workflow's state on the strength of a guess about what was
    meant. Generic by construction: the owning contexts come from the routing definition and the entering command only
    from a command's own ``@descends_to`` declaration, never from the shape of a command's name.

    *enter_commands* is keyed by owning context. With one owner the entering commands are simply listed; with several,
    each is named beside the context it enters. Owners that declare nothing are left out of the entry text, and still
    named among the owners.
    """
    owners = ", ".join(owner_contexts)
    plural = "contexts" if len(owner_contexts) > 1 else "context"
    declared = [(context_name, enter_commands[context_name])
                for context_name in owner_contexts if enter_commands.get(context_name)]
    entry = ""
    if len(owner_contexts) == 1 and declared:
        entry = " Enter it with: " + ", ".join(f"'{c}'" for c in declared[0][1]) + "."
    elif declared:
        entry = "".join(f" Enter {context_name} with: " + ", ".join(f"'{c}'" for c in commands) + "."
                        for context_name, commands in declared)
    if entry:
        step = f"{entry} Then run '{command_name}' there."
    else:
        where = "that context" if len(owner_contexts) == 1 else "one of those contexts"
        step = f" Move into {where} first, then run '{command_name}' there."
    return (f"'{command_name}' is a command of the {owners} {plural}, which this context and its parents do not "
            f"provide.{step} Use 'what_can_i_do' to list the commands available where you are now.")


def routing_hint_for(workflow_folderpath: str, command_name: str, owner_contexts: list[str]) -> str:
    """The hint for one foreign known name, the entering commands resolved (never raises: ``entering_commands``)."""
    from fastworkflow.context_navigation import entering_commands

    return foreign_context_hint(command_name, owner_contexts,
                                {name: entering_commands(workflow_folderpath, name) for name in owner_contexts})


def foreign_owners(workflow_folderpath: str, name: str, on_path: Iterable[str],
                   core_leaves: Iterable[str] = ()) -> list[str]:
    """The contexts that own the command *name* (a first word, any case) when no scope of *on_path* does: the
    foreign-name condition. Empty for free text (a word that names no command of this workflow), for a name a scope of
    the path owns (the exact layer found it), and for a framework command (*core_leaves*, lower case). A context
    is never its own foreign owner."""
    bare = str(name).lstrip("/").lower()
    if not bare or bare in {leaf.lower() for leaf in core_leaves}:
        return []
    path = set(on_path)
    owners = command_inventory(workflow_folderpath).get(bare, ())
    return [] if any(context in path for context in owners) else list(owners)


def foreign_hint(workflow, text: str) -> Optional[tuple[str, list[str]]]:
    """``(routing hint, owning contexts)`` when the first word of *text* names a command no scope of the lookup path
    owns, else None. The guard of ``resolve`` as a question of its own: the CME applies it to a reply to the which-command
    question too, since a reply that names a real command owned elsewhere is a misroute, not an unreadable answer."""
    from fastworkflow import context_navigation

    _, core_names = _kinds(workflow)
    bare = first_word(text).lstrip("/").lower()
    on_path = [scope.context_name for scope in context_navigation.exact_name_scopes(workflow)]
    owners = foreign_owners(workflow.folderpath, bare, on_path, [name.split("/")[-1] for name in core_names])
    return (routing_hint_for(workflow.folderpath, bare, owners), owners) if owners else None


def _core_fuzzy_keys() -> dict[str, str]:
    """``{key: qualified core name}`` the CME's fuzzy name layer also reads: the context-independent navigation commands
    by their names, and ``you_misunderstood`` by its plain utterances."""
    import fastworkflow
    from fastworkflow.workflow_agent import core_command_names

    internal = fastworkflow.RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    keys = {name.split("/")[-1]: name for name in core_command_names() if name.startswith("IntentDetection/")}
    metadata = internal.command_directory.map_command_2_utterance_metadata["ErrorCorrection/you_misunderstood"]
    keys.update({utterance: "ErrorCorrection/you_misunderstood" for utterance in metadata.plain_utterances})
    return keys


def _named(workflow, qualified: str, planning: set[str], core_names: set[str], decided_by: str, scope, facts: dict,
           ) -> Intent:
    kind = _kind_of(qualified, planning, core_names)
    return Intent(MATCH, qualified, kind, (Ranked(qualified, kind, 1.0),), decided_by,
                  evidence=_evidence(workflow, (), scope), context=scope.context_name,
                  record=getattr(scope, "record", None), facts=facts)


def _by_name(workflow, text: str, planning: set[str], core_names: set[str], core: bool,
             ) -> tuple[Optional[Intent], dict]:
    """Layers 1 to 3: an exact name, the foreign-name guard, a fuzzy name. ``(intent, facts)``: the intent is None when
    none of them decides, and the facts (a fuzzy tie) are then kept for the span of the chain's decision."""
    from fastworkflow import context_navigation

    token = first_word(text)
    bare = token.lstrip("/").lower()
    facts: dict = {}

    # 1. exact
    if core and bare:
        for qualified in sorted(core_names):
            if bare in (qualified.lower(), qualified.split("/")[-1].lower()):
                return _named(workflow, qualified, planning, core_names, BY_EXACT_NAME, _current_scope(workflow),
                              {"matcher_layer": "exact_prefix"}), {}
    scopes = context_navigation.exact_name_scopes(workflow)
    if bare:
        for scope in scopes:
            hits = [name for name in _scope_names(workflow, scope.context_name, core_names)
                    if context_navigation.token_names_command(bare, name)]
            if hits:
                facts = {"matcher_layer": "exact_prefix", "candidate_count": len(hits)}
                if len(hits) == 1:
                    return _named(workflow, hits[0], planning, core_names, BY_EXACT_NAME, scope, facts), {}
                kinds = tuple(Ranked(name, _kind_of(name, planning, core_names), 1.0) for name in hits)
                return Intent(AMBIGUOUS, candidates=kinds, decided_by=BY_QUESTION, ambiguous=tuple(hits),
                              evidence=_evidence(workflow, (), scope), context=scope.context_name,
                              record=scope.record, facts=facts), {}

    # 2. the foreign-name guard
    if bare:
        owners = foreign_owners(workflow.folderpath, bare, [scope.context_name for scope in scopes],
                                [name.split("/")[-1] for name in core_names])
        if owners:
            hint = routing_hint_for(workflow.folderpath, bare, owners)
            facts = {"matcher_layer": "known_name_foreign_context", "known_name_foreign_context": True,
                     "known_name_owner_contexts": owners, "known_name_foreign_context_hint": hint}
            return Intent(NONE, decided_by=BY_FOREIGN_NAME, hint=hint, owners=tuple(owners),
                          evidence=_evidence(workflow, (), None), facts=facts), {}

    # 3. fuzzy: the WHOLE request is a misspelling of one WHOLE name (its leaf or its qualified name; spaces, underscores
    # and case ignored), nearest scope first (``intent_clarification.misspelt``). A request that only begins like a
    # name is not one. A match is the closest command, not the user naming it (``NAMED_DECIDERS``).
    for index, scope in enumerate(scopes):
        keys: dict[str, str] = {}
        for name in _scope_names(workflow, scope.context_name, core_names):
            keys.update({name: name, name.split("/")[-1]: name})
        if core and index == 0:
            keys = {**_core_fuzzy_keys(), **keys}
        if not keys:
            continue
        matched, distance = intent_clarification.misspelt(text, keys)
        commands = list(dict.fromkeys(keys[key] for key in matched))
        if not commands:
            continue
        facts = {"fuzzy_distance": distance, "fuzzy_threshold": intent_clarification.FUZZY_MAX_DISTANCE,
                 "candidate_count": len(commands)}
        if len(commands) > 1:
            # Two names the request is equally close to: picking the first would choose between them by set order. A tie
            # is not a match: the chain decides.
            from fastworkflow.utils.logging import logger

            logger.warning(f"Fuzzy pre-match tied across {commands} for utterance '{text}' in context "
                           f"'{scope.context_name}'. Deferring to the chain instead of picking one.")
            return None, {**facts, "fuzzy_prematch_tie": [str(c).split("/")[-1] for c in commands]}
        facts["matcher_layer"] = "fuzzy_prematch"
        return _named(workflow, commands[0], planning, core_names, BY_FUZZY_NAME,
                      scope if commands[0] not in core_names else _current_scope(workflow), facts), {}
    return None, {}


# ======================================================================================================================
# The decision
# ======================================================================================================================

def _second_opinion(workflow, text: str, local: str, local_ranked: tuple, path, global_last: bool,
                    planning: set[str], core_names: set[str], offered: set[str], entry_only: frozenset,
                    excluded_core: set[str], local_answers: Iterable = ()) -> Intent:
    """A confident LOCAL answer that names a planning command is not a decision on its own: a keyword can carry the
    local model to a multi-step plan the request does not ask for. The remote decision model is asked over the same
    scope path (when the key is set) and its answer stands, the local plan kept as the top candidate; with no key, or
    when its call fails, the request is an ambiguity, asked like any other, never the unreviewed fast path."""
    from fastworkflow import intent_detector

    answers = tuple(local_answers)
    if intent_detector.matcher_available():
        memo: dict = {}

        def remote_of(context_name: str):
            return intent_detector.MatcherEngine(workflow.folderpath, context_name, path, fallback=False, memo=memo,
                                                 exclude=entry_only)

        remote = _walk(workflow, text, remote_of, path, global_last)
        answers = (*local_answers, *remote.answers)
        evidence = _evidence(workflow, answers, remote.scope)
        if not any(intent_detector.matcher_could_not_answer(answer) for answer in remote.answers):
            deciding = remote.answers[-1] if remote.answers else None
            ranked = _ranked(remote, planning, offered, deciding, core_names)
            first = tuple(each for each in local_ranked if each.name == local)
            ranked = (*first, *(each for each in ranked if each.name != local))
            context = getattr(remote.scope, "context_name", None)
            record = getattr(remote.scope, "record", None)
            if remote.outcome == intent_detector.MATCH and remote.command not in excluded_core:
                command = remote.command
                ranked = (*(each for each in ranked if each.name == command),
                          *(each for each in ranked if each.name != command))
                if not any(each.name == command for each in ranked):
                    ranked = (Ranked(command, _kind_of(command, planning, core_names), 1.0), *ranked)
                return Intent(MATCH, command, _kind_of(command, planning, core_names), ranked, BY_MATCHER,
                              evidence=evidence, context=context, record=record, answers=answers)
            if remote.outcome == intent_detector.AMBIGUOUS:
                names = [local, *(name for name in _usable(remote.candidates, excluded_core) if name != local)]
                return Intent(AMBIGUOUS, candidates=ranked, decided_by=BY_QUESTION,
                              ambiguous=tuple(names[:MAX_ALTERNATIVES]), evidence=evidence, answers=answers)
            return Intent(NONE, candidates=ranked, decided_by=BY_MATCHER, evidence=evidence, answers=answers)
    names = [local, *(each.name for each in local_ranked if each.name != local)]
    return Intent(AMBIGUOUS, candidates=local_ranked, decided_by=BY_QUESTION, ambiguous=tuple(names[:MAX_ALTERNATIVES]),
                  evidence=_evidence(workflow, local_answers, None), answers=answers)


def resolve(workflow, text: str, *, core: bool = False) -> Intent:
    """Which command *text* means from where *workflow* stands (see the module docstring for the layers).

    *workflow* is the application workflow. *core* lets the CME answer the framework's core commands (see the module
    docstring); a plan part passes False. Never raises for a model that cannot be reached: the chain then stands on the
    local model. Not memoised here: ``resolution.part_intent`` memoises for the turn.
    """
    planning, core_names = _kinds(workflow)
    if not str(text).strip():
        return Intent(NONE)
    named, facts = _by_name(workflow, text, planning, core_names, core)
    if named is not None:
        return named
    return dataclasses.replace(_chain(workflow, text, planning, core_names, core), facts=facts)


def _chain(workflow, text: str, planning: set[str], core_names: set[str], core: bool) -> Intent:
    """Layers 4 to 7: the local model over the whole walk, then the chain."""
    from fastworkflow import intent_detector

    excluded_core = set() if core else core_names
    path, global_last = scope_path(workflow)
    entry_only = _entry_only(workflow)
    memo: dict = {}

    def engine_of(context_name: str):
        # The remote link is never offered a command that only enters a context: a request that names a workspace by the
        # subject it is about would otherwise be sent there instead of to the plan that serves it.
        return intent_detector.engine_for(workflow.folderpath, context_name, scope_path=path, memo=memo,
                                          exclude=entry_only)

    def local_of(context_name: str):
        return intent_detector.engine_for(workflow.folderpath, context_name, allow_matcher=False, scope_path=path)

    # The local model first, over the whole walk: a confident answer anywhere on it decides, with no model call (the
    # chain would otherwise ask the remote model at a scope the local model is unsure of, the global-first step among
    # them, before a nearer scope's confident answer is reached).
    detection = intent_detector.detect(workflow, text, engine_of=local_of)
    if detection.outcome != intent_detector.MATCH or detection.command in excluded_core:
        # With no remote model the chain IS the local model: the answers already made stand (no second inference).
        detection = _walk(workflow, text, engine_of, path, global_last,
                          detection=None if intent_detector.matcher_available() else detection)
    deciding = detection.answers[-1] if detection.answers else None
    offered = {name for context in path for name in _offers(workflow, context)}
    ranked = _ranked(detection, planning, offered, deciding, core_names)
    context = getattr(detection.scope, "context_name", None)
    record = getattr(detection.scope, "record", None)
    answers = tuple(detection.answers)
    if detection.outcome == intent_detector.MATCH and detection.command not in excluded_core:
        command = detection.command
        kind = _kind_of(command, planning, core_names)
        if not any(each.name == command for each in ranked):
            ranked = (Ranked(command, kind, 1.0), *ranked)
        decided_by = deciding.engine if deciding is not None else BY_CLASSIFIER
        if decided_by == BY_CLASSIFIER and command in planning:
            return _second_opinion(workflow, text, command, ranked, path, global_last, planning, core_names, offered,
                                   entry_only, excluded_core, local_answers=detection.answers)
        return Intent(MATCH, command, kind, ranked, decided_by,
                      evidence=_evidence(workflow, detection.answers, detection.scope), context=context, record=record,
                      answers=answers)
    if detection.outcome == intent_detector.AMBIGUOUS:
        names = _usable(detection.candidates, excluded_core)
        if names:
            order = {name: index for index, name in enumerate(names)}
            ranked = tuple(sorted(ranked, key=lambda each: (order.get(each.name, len(order)),)))
            return Intent(AMBIGUOUS, candidates=ranked, decided_by=BY_QUESTION, ambiguous=tuple(names),
                          evidence=_evidence(workflow, detection.answers, detection.scope), answers=answers)
    bare = bool(deciding is not None and deciding.kind == intent_detector.BARE_VALUE)
    return Intent(NONE, candidates=ranked, decided_by=deciding.engine if deciding is not None else BY_NOTHING,
                  evidence=_evidence(workflow, detection.answers, detection.scope), bare_value=bare, answers=answers)


# ======================================================================================================================
# Where a chosen command lives
# ======================================================================================================================

def choose(workflow, qualified: str, *, decided_by: str = BY_CHOICE) -> Optional[Intent]:
    """The intent of a command somebody CHOSE by name (a pick at the question): where it lives on the lookup path, so it
    runs on that scope's record (the climb's binding), or None when no scope of the path has it.

    A global command is taken from the global scope when the walk asks that scope first (``global_first_applies``),
    as the walk does for natural language; otherwise the nearest scope that has it. A framework command runs anywhere,
    on the current scope."""
    from fastworkflow import context_navigation, intent_detector

    planning, core_names = _kinds(workflow)
    kind = _kind_of(qualified, planning, core_names)
    if qualified in core_names:
        scope = _current_scope(workflow)
    else:
        scope = None
        if intent_detector.global_first_applies(workflow) and qualified in set(
                context_navigation.scope_command_names(workflow.folderpath, intent_detector.GLOBAL_CONTEXT)):
            scope = intent_detector.Scope(intent_detector.GLOBAL_CONTEXT, workflow.root)
        for each in () if scope is not None else context_navigation.exact_name_scopes(workflow):
            if qualified in context_navigation.scope_command_names(workflow.folderpath, each.context_name):
                scope = intent_detector.Scope(each.context_name, each.record)
                break
    if scope is None:
        return None
    return Intent(MATCH, qualified, kind, (Ranked(qualified, kind, 1.0),), decided_by,
                  evidence={"context": scope.context_name}, context=scope.context_name, record=scope.record)


@contextlib.contextmanager
def on_owner_record(workflow, qualified: str):
    """Run what is inside (a command's argument checks made outside a dispatch: the plan path's draft, its values
    question and reply, its run's check) with ``workflow.record`` the record *qualified* will run on, as the dispatcher
    binds it before the "/" path extracts (``choose``: the nearest scope of the lookup path that has it, the root for a
    global command; a command no scope of the path has keeps the current position). The binding is context-local
    (``workflow.running_command``), so parts checked side by side each see their own; the position does not move."""
    from fastworkflow.workflow import running_command

    chosen = choose(workflow, qualified) if qualified else None
    record = chosen.record if chosen is not None else workflow._current_command_context
    with running_command(workflow, record):
        yield


def unconfirmed_pick(workflow, intent: "Intent", origin) -> bool:
    """Whether *intent* is a match nobody named, of a command that needs the user's say, that nothing else will confirm:
    a model (the local or the remote one) or a close spelling of a name (``BY_FUZZY_NAME``) chose a command that writes
    (``command_writes.write_commands``: an execution command with ``@writes``, or a planning command whose steps reach
    one) or that carries an approval mark (``approval_annotations.command_approval``, while the gate enforces), and the
    approval gate will not ask about it for a call by *origin* (``approval_binding.would_ask``: that question names the
    command, so it is the yes; a person's own "/" message runs a ``REQUIRED_UNTIL_DELEGATED`` command unasked, so a
    model's pick of one is asked here). Such a match needs the user's yes: the plan path reviews it
    (``resolution.is_fast_path``), the "/" path asks "Did you mean the command 'X'?" (``CommandNamePrediction.predict``).
    The one rule both paths read: "a command that changes data or needs approval runs only when the user named it
    (typed its exact name, picked it at the which-command question, or a client pinned or called it directly), or
    after the user says yes to it". Raises what reading the workflow's routing raises."""
    from fastworkflow import approval_annotations, approval_binding, command_writes

    if intent.outcome != MATCH or intent.decided_by in NAMED_DECIDERS or not intent.command:
        return False
    folder = workflow.folderpath
    if (intent.command not in command_writes.write_commands(folder)
            and not (approval_binding.switch_mode() == approval_binding.MODE_ENFORCE
                     and approval_annotations.command_approval(folder, intent.command) is not None)):
        return False
    return not approval_binding.would_ask(workflow, intent.command, origin)


# ======================================================================================================================
# Learning: the one hook
# ======================================================================================================================

def how_settled(decided_by: str) -> Optional[str]:
    """How a request's command was settled, for `intent_learning`: None when nobody settled it (a question comes next)."""
    from fastworkflow import intent_learning as learning

    return {BY_CLASSIFIER: learning.HOW_LOCAL, BY_MATCHER: learning.HOW_REMOTE, BY_EXACT_NAME: learning.HOW_NAME,
            BY_FUZZY_NAME: learning.HOW_NAME, BY_PINNED: learning.HOW_USER_CHOICE,
            BY_CHOICE: learning.HOW_USER_CHOICE}.get(decided_by)


def record(workflow, text: str, command: str, how: str, evidence: Optional[dict] = None,
           rejected: Optional[str] = None, context: Optional[str] = None) -> None:
    """Record a request's settled command (`intent_learning.record`; never raises). THE one writer, called where a decision
    is settled: by the plan path for a part (``WorkflowExecutionContext``) and by the CME's router for a request."""
    from fastworkflow import intent_learning

    where = context or (evidence or {}).get("context") or workflow._current_command_context_name
    intent_learning.record(workflow.folderpath, text=text, context=where, command=command, how=how,
                           evidence=evidence, rejected_label=rejected)


# ======================================================================================================================
# The reader and the pending question
# ======================================================================================================================

read = intent_clarification.read
PICK, ABORT, UNCLEAR, LIST = (intent_clarification.PICK, intent_clarification.ABORT, intent_clarification.UNCLEAR,
                              intent_clarification.LIST)

#: What ``answer`` returns besides the readings: the question is asked again, or the replies ran out.
ASK_AGAIN = "ask_again"
EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class Answer:
    """A reply applied to the pending question: ``kind`` is ``pick`` (``name``), ``none``, ``abort``, ``list``,
    ``ask_again`` (``question`` is the question asked once more) or ``exhausted`` (the second reply that could not be
    read).

    ``rest`` is always empty: a reply answers its question and nothing else (the question says so), so the words after
    a picked name are not the command's parameter text on either path; the request that was asked about is (bd
    fix-ftdl.7.8, D5). The field stays only because the CME's router (a frozen ``_commands/`` file) reads it; it goes
    with the next retrain of that workflow (bd fix-qjne.32)."""

    kind: str
    name: Optional[str] = None
    rest: str = ""
    question: str = ""


def open_question(request: str, choices: list[dict[str, Any]], *, sub_task: Optional[int] = None,
                  ambiguous: bool = True, misunderstood: bool = False, origin: Optional[int] = None,
                  for_agent: bool = False, listing_hint: bool = False) -> dict[str, Any]:
    """The record of a which-command question, as plain data. *choices* are ``{name, description, closest}``. Call
    ``ask`` to write its text."""
    return {"origin": origin, "request": request, "sub_task": sub_task, "choices": choices, "ambiguous": ambiguous,
            "misunderstood": misunderstood, "asks": 0, "stalled": 0, "question": "", "for_agent": for_agent,
            "listing_hint": listing_hint}


def ask(record: dict[str, Any], *, retry: str = "") -> str:
    """The question's text, written into the record (``record["question"]``) and counted."""
    from fastworkflow import no_agent_mode

    record["question"] = intent_clarification.plan_choice_question(
        record["request"], record["choices"], sub_task=record.get("sub_task"), listed=no_agent_mode.AD_HOC_LIST_CAP,
        retry=retry, ambiguous=bool(record.get("ambiguous")), misunderstood=bool(record.get("misunderstood")),
        for_agent=bool(record.get("for_agent")), listing_hint=bool(record.get("listing_hint")))
    record["asks"] = int(record.get("asks") or 0) + 1
    return record["question"]


def payload(record: dict[str, Any]) -> dict[str, Any]:
    """The client's data for the question (artifact ``plan_choice``, with ``question_kind`` ``plan_choice``)."""
    return intent_clarification.plan_choice_payload(record)


def prompt_artifacts(record: dict[str, Any]) -> dict[str, Any]:
    """``{question_kind, plan_choice}``: the artifacts a client reads the pending question from."""
    return {"question_kind": intent_clarification.QUESTION_KIND_PLAN_CHOICE,
            intent_clarification.PLAN_CHOICE_ARTIFACT: payload(record)}


def answer(record: dict[str, Any], reply: str, *, aliases: Optional[Mapping[str, str]] = None) -> Answer:
    """Apply *reply* to the pending question (*aliases*: see ``read``). A reply that cannot be read is asked once more
    (``no_agent_mode.MAX_NO_PROGRESS`` replies in all); after the last one the replies are ``exhausted`` and the
    caller ends the question (the plan path drops the part; the CME lists the commands). Counts in the record."""
    from fastworkflow import no_agent_mode

    names = [each["name"] for each in record["choices"]]
    reading = read(reply, names, numbered=no_agent_mode.AD_HOC_LIST_CAP, aliases=aliases)
    if reading.kind == intent_clarification.UNCLEAR:
        record["stalled"] = int(record.get("stalled") or 0) + 1
        if record["stalled"] < no_agent_mode.MAX_NO_PROGRESS:
            return Answer(ASK_AGAIN, question=ask(record, retry=intent_clarification.RETRY))
        return Answer(EXHAUSTED)
    return Answer(reading.kind, reading.name)


def describe(workflow, names: Iterable[str]) -> dict[str, str]:
    """``{name: one-line description}`` for the commands of the request's scope path."""
    from fastworkflow import intent_detector

    wanted = set(names)
    found: dict[str, str] = {}
    for context in scope_path(workflow)[0]:
        try:
            offered = intent_detector.candidates(workflow.folderpath, context)
        except ValueError:
            continue
        for candidate in offered:
            if candidate.name in wanted and candidate.name not in found:
                found[candidate.name] = (candidate.description.strip().splitlines() or [""])[0].strip()
    return found


def choices_of(workflow, names: Sequence[str]) -> list[dict[str, Any]]:
    """``{name, description, closest}`` for the ambiguity's candidates, in order."""
    described = describe(workflow, names)
    return [{"name": name, "description": described.get(name, ""), "closest": True} for name in names]


def core_aliases(*, listing: bool = True) -> dict[str, str]:
    """The phrases a reply to the question reads besides a candidate's name: the plain utterances of ``abort`` (abort),
    ``you_misunderstood`` (none of these) and, with *listing*, ``what_can_i_do`` (list the commands), and those commands'
    names. The one source of them: the CME reads all three, the plan path (no listing of commands there) the first two.
    What the CME used to merge into its own matching, now an argument of the one reader."""
    import fastworkflow

    internal = fastworkflow.RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    metadata = internal.command_directory.map_command_2_utterance_metadata
    aliases: dict[str, str] = {}
    for command, kind in (("IntentDetection/what_can_i_do", intent_clarification.LIST),
                          ("ErrorCorrection/you_misunderstood", intent_clarification.NONE),
                          ("ErrorCorrection/abort", intent_clarification.ABORT)):
        if kind == intent_clarification.LIST and not listing:
            continue
        for phrase in (*metadata[command].plain_utterances, command.split("/")[-1].replace("_", " ")):
            aliases[str(phrase).lower()] = kind
    return aliases


def command_choices(app_workflow, *, core: bool = True) -> list[dict[str, Any]]:
    """Every command the user can name from where the workflow stands, as the choices of the misunderstanding question
    (nothing fitted, so the candidates are all of them): the CME's navigation commands, the current context's commands
    and its planning commands, by name. Without *core* the CME's navigation commands are left out (a plan does not run
    them)."""
    import fastworkflow
    from fastworkflow.nlu_labels import is_non_routable

    internal = fastworkflow.RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    app = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
    context = app_workflow._current_command_context_name
    navigation = set(internal.get_command_names("IntentDetection"))
    names = {*navigation, *app.get_command_names(context), *app.get_planning_command_names(context)}
    if not core:
        names -= navigation
    names = sorted((name for name in names if not is_non_routable(name.split("/")[-1])),
                   key=lambda name: (name.split("/")[-1].lower(), name))
    described = describe(app_workflow, names)
    return [{"name": name, "description": described.get(name, ""), "closest": False} for name in names]


def pending_record(context: dict) -> Optional[dict]:
    """The which-command question the CME's turn holds (the plain-data record ``open_question`` built), or None."""
    held = waiting.port_of(context).get()
    return held.payload["record"] if held is not None and held.kind == waiting.KIND_INTENT else None


def waiting_for(context: dict) -> Optional[str]:
    """What the CME is waiting for, read from the turn's one waiting record and nothing else: ``WAITING_INTENT`` (a
    which-command question is open), ``WAITING_PARAMETERS`` (a parameter question is open) or None (nothing is pending:
    the next message is a request)."""
    held = waiting.port_of(context).get()
    if held is None:
        return None
    return {waiting.KIND_INTENT: WAITING_INTENT, waiting.KIND_PARAMETERS: WAITING_PARAMETERS}.get(held.kind)


def store(context: dict, record: dict) -> None:
    """Make *record* the turn's one pending question: the which-command record replaces whatever question the CME held
    (the port holds one record at a time)."""
    waiting.port_of(context).put(waiting.Waiting(waiting.KIND_INTENT, owner=waiting.OWNER_CME, payload={"record": record}))


def clear(context: dict) -> None:
    """Close the which-command question (nothing when none is pending)."""
    if pending_record(context) is not None:
        waiting.port_of(context).clear()
