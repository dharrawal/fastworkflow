"""The detector contract: which command of one scope an utterance means, and the walk over scopes.

This module owns three things and nothing else (design: ``docs/unified_command_pipeline_design.md``,
section 2):

* **The scope decision.** ``Engine.decide(utterance) -> ScopeAnswer``: given one scope (a context
  name), decide among that scope's commands, or say "not mine". ``ClassifierEngine`` is the trained
  per-context head (``CommandRouter``) behind that contract. ``candidates`` lists what a scope offers,
  from the routing definition and the planning command map, never from an engine.
* **The walk.** ``scopes_for`` is the lookup path; ``ScopeWalk`` asks the scopes on it, with the two
  rules that sit on top ("confident global first", "nearest scope wins"). The per-scope step is a
  callable, so the CME stages walk with ``CommandNamePrediction.predict`` (which keeps the typed-name
  layers and calls the engine where it called the router), and ``detect`` walks with the engines alone.
* **The detection chain.** There is one way a scope is decided, and no setting chooses it:

  1. the scope's trained local intent model decides when it is confident (one command, or a bare value),
     with no model call. It never passes a request up on its own: when its best command lies outside the
     scope it is not confident, and hands the request on with that command among the candidates;
  2. otherwise (a list of candidates, the vague-reply "not here", or no local model for the scope) the remote
     decision model (``MatcherEngine``, Jev) is asked over the scope path when ``JEV_API_KEY`` is set, and
     decides when it is confident;
  3. otherwise the candidates go to the user or the agent (the ambiguity question), and a walk that every
     scope declined ends in the "did not understand" question.

  ``engine_choice`` names the links a scope has by what is available; ``engine_for`` builds them
  (``ChainEngine`` for a scope with both). ``startup_report`` is the one place that says whether a
  workflow with untrained scopes may start, and the one line that says what its chain is.

It imports nothing heavy at module level: the trainer's ``CommandRouter`` and the planning command
loader are imported where they are used, so the CME commands can import this module freely.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Protocol, Sequence, runtime_checkable

import fastworkflow
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL, is_escalation, label_of


# ---------------------------------------------------------------- the scope decision

MATCH = "match"
AMBIGUOUS = "ambiguous"
NOT_MINE = "not_mine"
BARE_VALUE = "bare_value"
NONE = "none"

#: The scope's trained local intent model (whatever its training layout).
ENGINE_CLASSIFIER = "classifier"
#: The remote decision model (Jev), asked when ``JEV_API_KEY`` is set.
ENGINE_MATCHER = "matcher"
#: ``engine_choice``'s name for a scope that has both links: the local model, then the matcher.
ENGINE_CHAIN = "chain"
#: The ``engine`` a span records when no link was confident and the candidates are put to the user or the
#: agent. Never an ``Engine``'s name: it says who decides, and then it is the one who is asked.
ENGINE_QUESTION = "question"

GLOBAL_CONTEXT = "*"


@dataclass(frozen=True)
class ScopeAnswer:
    """One engine's answer for one scope.

    kind: ``match`` (one command), ``ambiguous`` (several candidates), ``not_mine`` (the escalation
        signal: an enclosing scope may serve the utterance) or ``bare_value`` (the utterance is a
        value, not a command here or anywhere else; it ends the walk).
    labels: what the engine returned, best first, unchanged: the one command for a match, the
        candidates for an ambiguity, and the reserved label itself for ``not_mine`` and
        ``bare_value``. An ambiguity keeps any reserved label the engine ranked among the candidates,
        because the callers filter, log and record those themselves.
    scores: ``{label: score in 0..1}`` for every label the engine scored (empty when it gave none).
    confident: True only for a match the engine stands behind without help.
    engine: the name of the link that gave the answer (``classifier`` or ``matcher``).
    fallback: True when the matcher answered for a scope that has no trained local model.
    details: the engine's own facts for the span, as the engine wrote them.
    """

    kind: Literal["match", "ambiguous", "not_mine", "bare_value"]
    labels: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    confident: bool = False
    engine: str = ENGINE_CLASSIFIER
    fallback: bool = False
    details: dict = field(default_factory=dict)


@runtime_checkable
class Engine(Protocol):
    """What decides one scope. ``name`` is what the span records as ``engine``."""

    name: str

    def decide(self, utterance: str) -> ScopeAnswer:
        ...


@dataclass(frozen=True)
class Candidate:
    """One command a scope offers: its qualified name, its kind, and the texts that say what it is for."""

    name: str
    kind: Literal["execution", "planning", "core"]
    description: str = ""
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scope:
    """One scope of the walk: a context, and the record a command found there runs on."""

    context_name: str
    record: Optional[object] = None


@dataclass(frozen=True)
class Detection:
    """The result of a walk.

    outcome: ``match``, ``ambiguous`` or ``none`` (every scope declined, or a bare value ended the walk).
    command: the qualified command name (match only).
    candidates: the qualified names, best first (ambiguous only).
    scope: the context and record the command runs on; for ``none``, the last scope asked.
    answers: every ``ScopeAnswer`` of the walk, in the order the scopes were asked.
    """

    outcome: Literal["match", "ambiguous", "none"]
    command: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    scope: Optional[Scope] = None
    answers: list[ScopeAnswer] = field(default_factory=list)


#: The most commands of other scopes a hand-on adds to the scope's own candidates.
HANDED_ON_COMMANDS = 2


def reachable_labels(workflow_folderpath: str, scope_path: Sequence[str]) -> frozenset[str]:
    """The qualified commands a request asked in the walk *scope_path* can be run as: the commands of every
    scope on the path (execution and planning, never the framework's core commands). A scope the workflow
    does not have contributes nothing."""
    from fastworkflow import workflow_agent

    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    core = workflow_agent.core_command_names()
    names: set[str] = set()
    for context_name in scope_path:
        try:
            names.update(crd.get_command_names(context_name))
            names.update(crd.get_planning_command_names(context_name))
        except ValueError:
            continue
    return frozenset(names - set(core))


def handed_on(labels: Sequence[str], details: dict, reachable: Iterable[str]) -> tuple[list[str], dict[str, float]]:
    """A local model's candidates plus the best commands of other scopes it ranked ahead of its own.

    The model reports them as ``details["outside_labels"]`` / ``["outside_scores"]`` (best first) when it
    handed the request on because a command outside the scope was ahead. Up to ``HANDED_ON_COMMANDS`` of
    them that are in *reachable* (the walk can get there) and not already listed join *labels*; the result
    is ordered by score, the scope's own candidate first on a tie. Returns the labels and the scores of the
    ones added."""
    own = list(labels)
    own_scores = {str(label): float(score)
                  for label, score in zip(details.get("topk_labels", ()), details.get("topk_scores", ()))}
    reachable = frozenset(reachable)
    added: dict[str, float] = {}
    for label, score in zip(details.get("outside_labels", ()), details.get("outside_scores", ())):
        if len(added) == HANDED_ON_COMMANDS:
            break
        if label in reachable and label not in own and label not in added:
            added[str(label)] = float(score)
    if not added:
        return own, {}
    scored = {**own_scores, **added}
    merged = sorted(own + list(added), key=lambda label: -scored.get(label, 0.0))
    return merged, added


class ClassifierEngine:
    """The trained head of one context (a ``CommandRouter``) behind the contract.

    The router is handed in, never built here: its caller owns how it is constructed and cached
    (``CommandNamePrediction`` builds it through its own module's ``CommandRouter`` name).

    *reachable* is what lets a hand-on carry an ancestor's command: the qualified commands the walk can
    run (``reachable_labels``), or a zero-argument function giving them (asked only when needed). When
    the head is not confident and ranks commands of other scopes ahead of its own, the best reachable
    ones join the candidates (``handed_on``) and ``details["handed_on"]`` names them. Without it the
    answer is the head's own, as before.
    """

    name = ENGINE_CLASSIFIER

    def __init__(self, router: Any, *, fallback: bool = False,
                 reachable: Optional[Iterable[str] | Callable[[], Iterable[str]]] = None):
        self.router = router
        self.fallback = fallback
        self.reachable = reachable

    def decide(self, utterance: str) -> ScopeAnswer:
        labels, details = self.router.predict_with_details(utterance)
        labels = list(labels)
        # `.get` throughout: the details dict belongs to the router, and a router that reports
        # nothing (the test doubles that inject labels return an empty dict on purpose) still answers.
        scores = {
            str(label): float(score)
            for label, score in zip(details.get("topk_labels", ()), details.get("topk_scores", ()))
        }
        if len(labels) > 1 and self.reachable is not None and details.get("outside_labels"):
            reachable = self.reachable() if callable(self.reachable) else self.reachable
            labels, added = handed_on(labels, details, reachable)
            if added:
                scores.update(added)
                details = {**details, "handed_on": list(added)}
        if len(labels) != 1:
            # No label at all is treated as the callers treat it today: not one command, so an ambiguity.
            kind = AMBIGUOUS
        elif is_escalation(labels[0]):
            kind = NOT_MINE
        elif label_of(labels[0]) == PARAMETER_VALUE_LABEL:
            kind = BARE_VALUE
        else:
            kind = MATCH
        return ScopeAnswer(
            kind=kind,
            labels=labels,
            scores=scores,
            # The router returns one label only when it is confident; a router that says nothing
            # about confidence is taken at its word for the one label it gave.
            confident=kind == MATCH and bool(details.get("confident", True)),
            engine=self.name,
            fallback=self.fallback,
            details=details,
        )


# ---------------------------------------------------------------- candidates

def candidates(workflow_folderpath: str, context_name: str) -> list[Candidate]:
    """The commands the scope *context_name* offers: its execution commands (the framework's core
    commands among them, flagged ``core``) in routing order, then its planning commands.

    Read from the routing definition and the planning command map, so the trained head's label set and
    a matcher's option list can both be checked against one list. A description or example that cannot
    be read is left empty: the name alone is still a candidate. Raises ``ValueError`` for a context the
    workflow does not have, as the routing definition does.
    """
    from fastworkflow import workflow_agent

    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    core = workflow_agent.core_command_names()
    offered: list[Candidate] = []
    for name in crd.get_command_names(context_name):
        offered.append(Candidate(
            name=name,
            kind="core" if name in core else "execution",
            description=_execution_description(crd, name),
            examples=_execution_examples(crd, name),
        ))
    offered.extend(_planning_candidates(workflow_folderpath, context_name))
    return offered


def _planning_candidates(workflow_folderpath: str, context_name: str) -> list[Candidate]:
    """The planning commands runnable in *context_name* (its own, then the inherited ones), as candidates."""
    from fastworkflow import workflow_agent

    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    planning_names = crd.get_planning_command_names(context_name)
    if not planning_names:
        return []
    descriptions = {name: description
                    for name, description, _ in workflow_agent.load_planning_commands(workflow_folderpath)}
    utterances = workflow_agent.planning_command_utterances(workflow_folderpath)
    return [Candidate(name=name, kind="planning", description=descriptions.get(name, ""),
                      examples=tuple(utterances.get(name, ())))
            for name in planning_names]


def _execution_examples(crd, name: str) -> tuple[str, ...]:
    try:
        metadata = crd.command_directory.get_utterance_metadata(name)
    except Exception:  # noqa: BLE001 - a command without utterance metadata has no examples
        return ()
    return tuple(metadata.plain_utterances) if metadata and metadata.plain_utterances else ()


def _execution_description(crd, name: str) -> str:
    """The command's ``Signature`` docstring on one line, as a planning command's description is read."""
    import inspect

    try:
        signature = crd.get_command_class(name, fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    except Exception:  # noqa: BLE001 - a command whose module does not load is described by its name
        return ""
    return " ".join(inspect.cleandoc(getattr(signature, "__doc__", None) or "").split())


# ---------------------------------------------------------------- engine selection

def head_folder(workflow_folderpath: str, context_name: str, command_info_root: Optional[str] = None) -> str:
    """Where the trained head of *context_name* lives (``global`` for ``*``)."""
    from fastworkflow.train.artifact_versioning import context_folder_name

    root = command_info_root or os.path.join(workflow_folderpath, "___command_info")
    return os.path.join(root, context_folder_name(context_name))


def has_head(folder: str) -> bool:
    """Whether *folder* holds a trained head. The models are folders, hence ``exists``.

    Either layout counts: the per-context files, or a context of a version trained with
    ``INTENT_MODEL_LAYOUT=workflow``, whose folder holds a pointer to the workflow's one model
    (``CommandRouter`` reads both, so the engine built for the scope is the same)."""
    from fastworkflow.train.intent_model_layout import workflow_head_present
    from fastworkflow.train.score_lookup import HEAD_MARKERS

    return (all(os.path.exists(os.path.join(folder, marker)) for marker in HEAD_MARKERS)
            or workflow_head_present(folder))


#: How often (seconds) a process looks at a workflow's ``current`` pointer to see whether a newer head was published.
HEAD_RELOAD_CHECK_SECONDS = 3.0
_last_head_check: dict[str, float] = {}


def refresh_stale_head(folder: str) -> None:
    """Drop the loaded routers of *folder*'s workflow whose embedding head is no longer the current version, so the
    next ``CommandRouter(folder)`` builds the new one (a refit, `train.head_refit`, or a retrain, published while the
    process runs). Looks at the pointer at most once every `HEAD_RELOAD_CHECK_SECONDS` per workflow, and only for
    workflows whose cached router is an embedding head: the ``workflow`` and ``per_context`` layouts are unchanged
    (a retrain of them still needs a restart). A turn already running keeps the router it started with. Never raises."""
    import time

    from fastworkflow import model_pipeline_training
    from fastworkflow.train import artifact_versioning
    from fastworkflow.utils.logging import logger

    try:
        router_cache = model_pipeline_training.CommandRouter._instances_cache
        key = folder.replace("*", model_pipeline_training.GLOBAL_CONTEXT_FOLDER)
        inner = getattr(router_cache.get(key), "_workflow_router", None)
        if getattr(inner, "head_version", None) is None:
            return
        workflow = os.path.dirname(os.path.dirname(key))
        now = time.monotonic()
        if now - _last_head_check.get(workflow, -HEAD_RELOAD_CHECK_SECONDS) < HEAD_RELOAD_CHECK_SECONDS:
            return
        _last_head_check[workflow] = now
        current = artifact_versioning.resolve_current_version(workflow)
        if current is None:
            return
        with model_pipeline_training.CommandRouter._build_lock:
            prefix = os.path.join(workflow, "___command_info") + os.sep
            for cached_key in [k for k, v in router_cache.items() if k.startswith(prefix)
                               and getattr(getattr(v, "_workflow_router", None), "head_version", current) != current]:
                router_cache.pop(cached_key, None)
    except Exception as error:  # noqa: BLE001 - a reload that cannot be checked leaves the loaded head in place
        logger.debug(f"head reload check failed: {error}")


def _command_router(folder: str):
    # Looked up on the module at call time, so a caller that replaces the trainer's CommandRouter
    # (or places an instance in its cache) is honoured here as it is everywhere else.
    from fastworkflow import model_pipeline_training

    refresh_stale_head(folder)
    return model_pipeline_training.CommandRouter(folder)


#: The setting that used to choose among engines. It is gone: the chain is the only behaviour. A value
#: left in a workflow env file is ignored, and one warning says so.
ENGINE_SETTING = "INTENT_DETECTION_ENGINE"
_warned_engine_setting = False


def warn_if_engine_setting() -> None:
    """Log one warning per process when ``INTENT_DETECTION_ENGINE`` is still set: it is ignored."""
    global _warned_engine_setting
    if _warned_engine_setting:
        return
    try:
        raw = str(fastworkflow.get_env_var(ENGINE_SETTING, default="") or "").strip()
    except Exception:  # noqa: BLE001 - an env that cannot be read has no stale setting to report
        return
    if raw:
        _warned_engine_setting = True
        from fastworkflow.utils.logging import logger

        logger.warning(
            f"{ENGINE_SETTING}={raw!r} is ignored: the setting was removed. Intent detection is one chain "
            f"(the trained local model when it is confident, then the remote decision model when "
            f"JEV_API_KEY is set, then the user or the agent). Remove the line from the workflow env file.")


def heads_required() -> bool:
    """``FW_REQUIRE_HEADS=1`` in the process environment (read directly, as the test switch is): a scope
    with no trained local model is a failure, and the matcher does not stand in for it."""
    return os.environ.get("FW_REQUIRE_HEADS") == "1"


def head_is_confident(answer: "ScopeAnswer") -> bool:
    """Whether the local model's answer ends the chain at its first link: one label it is confident of, a
    command or a bare value. A list of candidates, or a lone escalation label, goes to the next link."""
    return (len(answer.labels) == 1 and answer.kind in (MATCH, BARE_VALUE)
            and bool(answer.details.get("confident", True)))


def matcher_available() -> bool:
    """Whether the model matcher can be asked: the planner key (``JEV_API_KEY``) is set. Never raises."""
    try:
        from fastworkflow import workflow_agent

        return workflow_agent._jev_lm() is not None
    except Exception:  # noqa: BLE001 - a matcher that cannot be built is a matcher that is not there
        return False


def engine_choice(
    workflow_folderpath: str,
    context_name: str,
    *,
    command_info_root: Optional[str] = None,
) -> Optional[str]:
    """The links of the chain the scope *context_name* has, or None when nothing can decide it.

    * ``chain``: a trained local model and the planner key: the local model, then the matcher.
    * ``classifier``: a trained local model and no planner key: the local model alone.
    * ``matcher``: no trained local model, and the planner key is set: the matcher alone. Not under
      ``FW_REQUIRE_HEADS=1``, where a missing local model is a failure.
    * None: no trained local model and nothing to stand in for it (a startup problem).

    The artifacts are looked at on every call, not remembered: a head that is trained, replaced or removed
    while the process runs is seen at once.
    """
    warn_if_engine_setting()
    trained = has_head(head_folder(workflow_folderpath, context_name, command_info_root))
    if trained:
        return ENGINE_CHAIN if matcher_available() else ENGINE_CLASSIFIER
    if matcher_available() and not heads_required():
        return ENGINE_MATCHER
    return None


def engine_for(
    workflow_folderpath: str,
    context_name: str,
    router_factory: Optional[Callable[[str], Any]] = None,
    *,
    command_info_root: Optional[str] = None,
    allow_matcher: bool = True,
    scope_path: Optional[Sequence[str]] = None,
    memo: Optional[dict] = None,
    exclude: Iterable[str] = (),
) -> Optional[Engine]:
    """The engine that decides the scope *context_name* (``engine_choice``); None when nothing can.

    1. chain: a ``ChainEngine``: the local model when it is confident, else the matcher.
    2. classifier: *router_factory* builds the router from the head's folder (default: the trainer's
       ``CommandRouter``, which caches per folder).
    3. matcher: a ``MatcherEngine`` over *scope_path* (the ordered context names of the walk this scope is
       part of; default: the scope alone), sharing *memo* with the other scopes of the detection so the walk
       costs one model call. Its ``fallback`` is True: it stands in for a local model that does not exist.
    4. neither: the scope cannot decide natural language; exact names still work.

    ``allow_matcher=False`` is for a caller that asks MANY scopes and must not turn each into a model call
    (``run_listing.head_answers``) or that asks the local model alone first (``intent_loop.resolve``): it gets the
    trained local model or None (with *scope_path*, one that hands on the commands of the other scopes).
    *command_info_root* reads the heads of another folder than the workflow's own ``___command_info``.
    *exclude* names commands the matcher is never to offer (the local model is not affected).
    """
    folder = head_folder(workflow_folderpath, context_name, command_info_root)
    if not allow_matcher:
        if not has_head(folder):
            return None
        # With a *scope_path* the head hands on commands of the other scopes it can reach, as it does in the chain.
        path = tuple(dict.fromkeys(scope_path)) if scope_path else None
        return ClassifierEngine(
            (router_factory or _command_router)(folder),
            reachable=None if path is None else (lambda: reachable_labels(workflow_folderpath, path)))
    choice = engine_choice(workflow_folderpath, context_name, command_info_root=command_info_root)
    path = tuple(dict.fromkeys(scope_path or (context_name,)))

    def classifier() -> ClassifierEngine:
        return ClassifierEngine(
            (router_factory or _command_router)(folder),
            reachable=lambda: reachable_labels(workflow_folderpath, path))

    if choice == ENGINE_CLASSIFIER:
        return classifier()
    if choice == ENGINE_MATCHER:
        return MatcherEngine(workflow_folderpath, context_name, scope_path, fallback=True, memo=memo,
                             exclude=exclude)
    if choice == ENGINE_CHAIN:
        return ChainEngine(
            classifier(),
            MatcherEngine(workflow_folderpath, context_name, scope_path, fallback=False, memo=memo,
                          exclude=exclude))
    return None


# ---------------------------------------------------------------- the model matcher as an engine

#: The remote decision model's two bars, constants of the product (measured on 1,102 stored answers,
#: docs/remote_link_worth.md: at 0.5 a large workflow got 13.4% confident wrong answers, at 0.9 6.5%).
#: In a scope, the best candidate at or above ACCEPT is a match; "none", or a command of another scope on the
#: path, at or above DECLINE declines the scope. Below ACCEPT the remote answer is not an answer: the candidates
#: go to the question. Each is overridable by one workflow-env setting; ``remote_accept_threshold`` and
#: ``remote_decline_threshold`` are the one place they are read.
DEFAULT_REMOTE_ACCEPT_THRESHOLD = 0.9
DEFAULT_REMOTE_DECLINE_THRESHOLD = 0.9
REMOTE_ACCEPT_THRESHOLD_SETTING = "INTENT_REMOTE_ACCEPT_THRESHOLD"
REMOTE_DECLINE_THRESHOLD_SETTING = "INTENT_REMOTE_DECLINE_THRESHOLD"


def _remote_threshold(setting: str, default: float) -> float:
    from fastworkflow.train.intent_model_layout import parse_threshold

    return parse_threshold(fastworkflow.get_env_var(setting, default=""), default, setting)


def remote_accept_threshold() -> float:
    """The probability at or above which the remote model's best command is the answer."""
    return _remote_threshold(REMOTE_ACCEPT_THRESHOLD_SETTING, DEFAULT_REMOTE_ACCEPT_THRESHOLD)


def remote_decline_threshold() -> float:
    """The probability at or above which the remote model's "none" (or a command of another scope) declines a scope."""
    return _remote_threshold(REMOTE_DECLINE_THRESHOLD_SETTING, DEFAULT_REMOTE_DECLINE_THRESHOLD)


#: A candidate above this is kept as a candidate of an ambiguity; a scope with none above it declines.
MATCHER_UNSURE_THRESHOLD = 0.2
#: An ambiguity names at most this many candidates, best first.
MATCHER_MAX_CANDIDATES = 3
#: The option that says no command on the path fits.
MATCHER_NONE = "none"
#: The most options (commands plus ``none``) one grouped call offers. A path with more is asked scope by
#: scope instead: one call per scope, each offering that scope's commands. A guard, not a measured limit:
#: the largest paths of the bundled examples (21 options, and 16 in one scope) are answered by one call
#: (``scripts/experiments/score_matcher_engine.py``), and nothing larger has been measured.
MATCHER_MAX_OPTIONS = 64
_MATCHER_TOPK = 5
_MATCHER_MEMO_SIZE = 64


def matcher_candidates(workflow_folderpath: str, context_name: str) -> list[Candidate]:
    """What the matcher is offered in a scope: its execution and planning commands. The framework's core
    commands are left out: the typed-name layers match those, and no request is routed to them by meaning."""
    return [candidate for candidate in candidates(workflow_folderpath, context_name) if candidate.kind != "core"]


def candidate_line(name: str, description: str, examples: Sequence[str], limit: int) -> str:
    """One listing line of the matcher's prompt: ``- name: description``, followed by ``e.g. "...", "..."``
    for at most *limit* example requests (whitespace collapsed, blank ones dropped). The planning match
    (``workflow_agent._matcher_job_types``) writes its job types with this function, so the two cannot drift."""
    shown = [" ".join(example.split()) for example in tuple(examples)[:max(0, limit)]]
    shown = [example for example in shown if example]
    return f"- {name}: {description}" + (
        f" e.g. {', '.join(json.dumps(example, ensure_ascii=False) for example in shown)}" if shown else "")


def planning_candidates(workflow_folderpath: str) -> list[Candidate]:
    """The workflow's planning commands as candidates, in the order the planning command files load: what
    the planning match offers. Read from the planning command map alone (the scope that owns a planning
    command is in its name), so the match needs no routing definition, as it never did. A planning command
    whose file did not load is not offered, as the planning match has always had it."""
    from fastworkflow import workflow_agent

    utterances = workflow_agent.planning_command_utterances(workflow_folderpath)
    return [Candidate(name=name, kind="planning", description=description,
                      examples=tuple(utterances.get(name, ())))
            for name, description, _ in workflow_agent.load_planning_commands(workflow_folderpath)]


def planning_candidates_by_scope(workflow_folderpath: str) -> dict[str, list[Candidate]]:
    """``planning_candidates`` by the scope that owns each (``""`` for a global one, else the context
    name): the option lists of the planning match, one per scope."""
    by_scope: dict[str, list[Candidate]] = {}
    for candidate in planning_candidates(workflow_folderpath):
        by_scope.setdefault(candidate.name.rpartition("/")[0], []).append(candidate)
    return by_scope


def _scope_title(context_name: str) -> str:
    return "global" if context_name == GLOBAL_CONTEXT else context_name


def _matcher_groups(workflow_folderpath: str, scope_path: Sequence[str],
                    exclude: frozenset = frozenset()) -> list[tuple[str, list[Candidate]]]:
    """``(context, its candidates)`` for each scope of the path, in path order, without the commands *exclude*
    names. A context the workflow does not have offers nothing."""
    groups = []
    for context_name in dict.fromkeys(scope_path):
        try:
            offered = [c for c in matcher_candidates(workflow_folderpath, context_name) if c.name not in exclude]
        except ValueError:
            offered = []
        groups.append((context_name, offered))
    return groups


def _described(candidate: Candidate) -> str:
    return candidate.description or candidate.name.rpartition("/")[2].replace("_", " ")


def matcher_listing(groups: Sequence[tuple[str, Sequence[Candidate]]], examples: int) -> str:
    """The commands of a path as the matcher reads them: one ``Scope <name>:`` block per scope, nearest
    first, each command once (under the nearest scope that offers it) as a ``candidate_line``."""
    listed: set[str] = set()
    blocks = []
    for context_name, offered in groups:
        lines = [candidate_line(c.name, _described(c), c.examples, examples)
                 for c in offered if c.name not in listed]
        listed.update(c.name for c in offered)
        if lines:
            blocks.append(f"Scope {_scope_title(context_name)}:\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _ask_matcher(utterance: str, groups: Sequence[tuple[str, Sequence[Candidate]]]) -> dict:
    """One model call: which of the commands of *groups* does *utterance* ask for. Returns
    ``{"scores": {option: probability}, "model", "latency_ms", "calls"}``, plus ``"error"`` when the call
    failed (the scores are then empty). Never raises."""
    from fastworkflow import remote_outage, workflow_agent
    from fastworkflow.utils.logging import logger

    result: dict = {"scores": {}, "model": None, "latency_ms": 0.0, "calls": 0}
    offered = {c.name: c for _, group in groups for c in reversed(list(group))}
    names = list(dict.fromkeys(c.name for _, group in groups for c in group))
    if not names:
        return result
    started = time.perf_counter()
    try:
        jev_lm = workflow_agent._jev_lm()
        if jev_lm is None:
            result["error"] = "JEV_API_KEY is not set"
            return result
        result["model"] = str(getattr(jev_lm, "model", None) or type(jev_lm).__name__)
        dspy = workflow_agent.dspy
        options = tuple((name, _described(offered[name])) for name in names) + (
            (MATCHER_NONE, "None of these commands fits the message."),)

        listing = matcher_listing(groups, workflow_agent.MATCHER_EXAMPLES)
        result["calls"] = 1
        with dspy.context(lm=jev_lm):
            choice = dspy.Predict(workflow_agent.command_match_signature(options))(
                message=utterance, commands=listing).command
        value, probability = workflow_agent._choice_value_and_probability(choice)
        result["scores"] = {str(option): float(score)
                            for option, score in (choice.probabilities or {value: probability}).items()}
        logger.info(f"command match for {utterance!r} over {[context for context, _ in groups]}: "
                    f"{value} (confidence {probability:.2f})")
    except Exception as error:  # noqa: BLE001 - a failed match declines; it never fails the turn
        marker = remote_outage.report(error, "command match")
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_class"] = marker["error_class"]
        result["status_code"] = marker["status_code"]
        result["scores"] = {}
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return result


def _matcher_distribution(workflow_folderpath: str, scope_path: tuple[str, ...], utterance: str,
                          exclude: frozenset = frozenset()) -> dict:
    """The matcher's scores for every command on *scope_path* but the ones *exclude* names: one grouped call, or,
    for a path with more than ``MATCHER_MAX_OPTIONS`` options, one call per scope (``"calls"`` says how many were
    made)."""
    groups = _matcher_groups(workflow_folderpath, scope_path, exclude)
    option_count = len({c.name for _, offered in groups for c in offered}) + 1
    if option_count <= MATCHER_MAX_OPTIONS or len(groups) < 2:
        return _ask_matcher(utterance, groups)
    merged: dict = {"scores": {}, "model": None, "latency_ms": 0.0, "calls": 0, "per_scope": True}
    none_scores = []
    for group in groups:
        if not group[1]:
            continue
        answer = _ask_matcher(utterance, [group])
        merged["calls"] += answer["calls"]
        merged["model"] = merged["model"] or answer["model"]
        merged["latency_ms"] = round(merged["latency_ms"] + answer["latency_ms"], 1)
        if "error" in answer:
            merged["error"] = answer["error"]
            merged["error_class"] = answer.get("error_class")
            merged["status_code"] = answer.get("status_code")
            continue
        none_scores.append(answer["scores"].pop(MATCHER_NONE, 0.0))
        for option, score in answer["scores"].items():
            merged["scores"][option] = max(merged["scores"].get(option, 0.0), score)
    if none_scores:
        merged["scores"][MATCHER_NONE] = min(none_scores)   # "none" overall only when every scope said so
    return merged


class MatcherEngine:
    """The model matcher (Jev) behind the contract: the chain's second link, and the only link of a scope
    with no trained local model.

    One model call per detection, not per scope. The call lists the commands of every scope on *scope_path*
    grouped by scope and returns a probability for each command and for ``none``; ``decide`` reads this
    scope's answer from those scores:

    * the scope's best command at or above ``remote_accept_threshold()`` is a ``match``;
    * ``none`` or a command of ANOTHER scope on the path (the model chose elsewhere, so the walk should get
      there) at or above ``remote_decline_threshold()``, or no command of this scope above
      ``MATCHER_UNSURE_THRESHOLD`` is ``not_mine``;
    * anything else is ``ambiguous`` with the scope's commands above ``MATCHER_UNSURE_THRESHOLD``, at most
      ``MATCHER_MAX_CANDIDATES``, best first.

    The call is memoised per detection (same workflow folder, same ordered scope path, same utterance): in
    the turn's memo when a turn is open (``turn_cache``), else in *memo*, a small dict the engines of one
    detection share (default: this engine's own). It never answers ``bare_value``, and it never raises into
    a turn: a failed call is logged and answers ``not_mine`` with ``details["error"]``.
    """

    name = ENGINE_MATCHER

    def __init__(
        self,
        workflow_folderpath: str,
        context_name: str,
        scope_path: Optional[Sequence[str]] = None,
        *,
        fallback: bool = True,
        memo: Optional[dict] = None,
        exclude: Iterable[str] = (),
    ):
        self.workflow_folderpath = workflow_folderpath
        #: Commands never offered as the answer (``resolution.part_intent``: the commands that only enter a context).
        self.exclude = frozenset(exclude)
        self.context_name = context_name
        path = tuple(dict.fromkeys(scope_path or (context_name,)))
        self.scope_path = path if context_name in path else (context_name, *path)
        self.fallback = fallback
        self._memo = memo if memo is not None else {}

    def distribution(self, utterance: str) -> dict:
        """The memoised model answer for *utterance* over this engine's scope path."""
        from fastworkflow import turn_cache

        key = ("matcher_engine", self.workflow_folderpath, self.scope_path, utterance, tuple(sorted(self.exclude)))

        def compute() -> dict:
            if self.exclude:
                return _matcher_distribution(self.workflow_folderpath, self.scope_path, utterance, exclude=self.exclude)
            return _matcher_distribution(self.workflow_folderpath, self.scope_path, utterance)

        if turn_cache.active():
            return turn_cache.memo(key, compute)
        if key not in self._memo:
            while len(self._memo) >= _MATCHER_MEMO_SIZE:
                self._memo.pop(next(iter(self._memo)))
            self._memo[key] = compute()
        return self._memo[key]

    def decide(self, utterance: str) -> ScopeAnswer:
        answered = self.distribution(utterance)
        all_scores: dict[str, float] = answered["scores"]
        try:
            own = [c.name for c in matcher_candidates(self.workflow_folderpath, self.context_name)
                   if c.name not in self.exclude]
        except ValueError:
            own = []
        scores = {name: float(all_scores.get(name, 0.0)) for name in own}
        ranked = sorted(own, key=lambda name: -scores[name])          # stable: listing order breaks a tie
        none_score = float(all_scores.get(MATCHER_NONE, 0.0))
        elsewhere = max((score for option, score in all_scores.items()
                         if option != MATCHER_NONE and option not in scores and option not in self.exclude), default=0.0)
        best = scores[ranked[0]] if ranked else 0.0
        kept = [name for name in ranked if scores[name] > MATCHER_UNSURE_THRESHOLD][:MATCHER_MAX_CANDIDATES]
        accept, decline = remote_accept_threshold(), remote_decline_threshold()
        if "error" in answered:
            kind, labels = NOT_MINE, [WILDCARD_LABEL]
        elif ranked and best >= accept:
            kind, labels = MATCH, [ranked[0]]
        elif none_score >= decline or elsewhere >= decline or not kept:
            kind, labels = NOT_MINE, [WILDCARD_LABEL]
        else:
            kind, labels = AMBIGUOUS, kept
        details = {
            "confident": kind == MATCH,
            "top_label": ranked[0] if ranked else None,
            "confidence": best,
            "topk_labels": ranked[:_MATCHER_TOPK],
            "topk_scores": [scores[name] for name in ranked[:_MATCHER_TOPK]],
            "none_score": none_score,
            "threshold": accept,
            "decline_threshold": decline,
            "unsure_threshold": MATCHER_UNSURE_THRESHOLD,
            "scope_path": list(self.scope_path),
            "model": answered.get("model"),
            "latency_ms": answered.get("latency_ms", 0.0),
            "calls": answered.get("calls", 0),
        }
        if "error" in answered:
            details["error"] = answered["error"]
            details["error_class"] = answered.get("error_class")
            details["status_code"] = answered.get("status_code")
        return ScopeAnswer(
            kind=kind,
            labels=labels,
            scores={**scores, MATCHER_NONE: none_score} if all_scores else {},
            confident=kind == MATCH,
            engine=self.name,
            fallback=self.fallback,
            details=details,
        )


def matcher_could_not_answer(answer: ScopeAnswer) -> bool:
    """Whether the matcher's answer is "the call failed" (it then answers ``not_mine`` with
    ``details["error"]``). In a chain the local model's answer stands in that case."""
    return answer.engine == ENGINE_MATCHER and "error" in answer.details


class ChainEngine:
    """The chain for a scope with both links: the local model decides when it is confident
    (``head_is_confident``: one command or a bare value), with no model call; in every other case (a list
    of candidates, a lone escalation label) the model matcher is asked over the scope path and ITS answer
    for the scope stands, carrying the local model's facts as ``details["classifier"]``. When the matcher's
    call fails, the local model's answer stands and carries ``details["matcher_error"]``. ``engine`` on the
    answer names the link that gave it."""

    name = ENGINE_CHAIN

    def __init__(self, classifier: ClassifierEngine, matcher: "MatcherEngine"):
        self.classifier = classifier
        self.matcher = matcher

    def decide(self, utterance: str) -> ScopeAnswer:
        head = self.classifier.decide(utterance)
        if head_is_confident(head):
            return head
        answer = self.matcher.decide(utterance)
        if matcher_could_not_answer(answer):
            return ScopeAnswer(
                kind=head.kind, labels=head.labels, scores=head.scores, confident=head.confident,
                engine=head.engine, fallback=head.fallback,
                details={**head.details, "matcher_error": answer.details["error"],
                         "matcher_error_class": answer.details.get("error_class"),
                         "matcher_status_code": answer.details.get("status_code")})
        return ScopeAnswer(
            kind=answer.kind, labels=answer.labels, scores=answer.scores, confident=answer.confident,
            engine=answer.engine, fallback=answer.fallback,
            details={**answer.details, "classifier": head.details})


# ---------------------------------------------------------------- startup: may an untrained scope run?

def _contexts_without_a_head(workflow_folderpath: str) -> list[str]:
    """The context folder names (``global`` for ``*``) of the workflow's own contexts that have no
    ``threshold.json``, read from its routing definition, which is built and saved when it is not there yet
    (what loading the workflow does). The contexts ``is_workflow_trained`` checks, without the trainer."""
    from fastworkflow.train.artifact_versioning import context_folder_name

    definition = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    internal = set(fastworkflow.CommandContextModel.load(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))._command_contexts)
    root = os.path.join(workflow_folderpath, "___command_info")
    folders = {context_folder_name(name) for name in (set(definition.contexts) - internal) | {GLOBAL_CONTEXT}}
    return sorted(folder for folder in folders
                  if not os.path.isfile(os.path.join(root, folder, "threshold.json")))


def startup_report(
    workflow_folderpath: str,
    missing_contexts: Optional[Sequence[str]] = None,
    *,
    build: bool = False,
) -> tuple[list[str], list[str]]:
    """``(problems, notices)`` about the workflow's detection chain, for every gate that refuses an
    untrained workflow (``fastworkflow run``, the MCP server, the chat page's workflow listing, the FastAPI
    server). The one decision of "may it start": a gate refuses, with its own words, when *problems* is not
    empty, and otherwise prints or logs the notice.

    *missing_contexts* is what the gate's own check found (``is_workflow_trained``'s second value: context
    folder names, ``global`` for ``*``, or a ``<...>`` marker for a workflow that was never built); it is
    computed here when not given. A workflow that may start gets ONE notice: the line that says what its
    chain is (the local model, then the remote decision model when the planner key is set, then the user or
    the agent), naming any context that has no trained local model and is decided by the matcher. A missing
    local model with nothing to stand in for it (no planner key, ``FW_REQUIRE_HEADS=1``, or a workflow that
    was never built) is one problem, and there is no notice. Never raises.

    ``build=True`` is for a gate that is about to load the workflow anyway (``fastworkflow run``, the
    servers): a workflow that was never built (no routing definition yet, so nobody knows its contexts) gets
    its routing definition built and saved here when the matcher could decide its scopes, and is then
    reported like any other. A gate that only looks (the chat page's listing of many folders) leaves it
    False, and a workflow that was never built stays "not trained" there.
    """
    warn_if_engine_setting()
    problems, untrained = _untrained_report(workflow_folderpath, missing_contexts, build)
    if problems:
        return problems, []
    notices = [chain_line(untrained)]
    try:
        from fastworkflow import intent_learning

        if line := intent_learning.threshold_line(workflow_folderpath):
            notices.append(line)
    except Exception:  # noqa: BLE001 - a notice that cannot be made never stops a start
        pass
    return [], notices


def chain_line(untrained: Sequence[str] = ()) -> str:
    """The one line that says what the detection chain is, given the context folder names that have no
    trained local model (they are decided by the matcher, so the planner key is set when there are any)."""
    if matcher_available():
        line = ("Intent detection: the trained local model decides when it is confident; otherwise the "
                "remote decision model (JEV_API_KEY) decides when it is confident; otherwise the user or "
                "the agent is asked.")
    else:
        line = ("Intent detection: the trained local model decides when it is confident; otherwise the "
                "user or the agent is asked (JEV_API_KEY is not set, so there is no remote decision model).")
    if untrained:
        many = len(untrained) > 1
        line += (f" Context{'s' if many else ''} {', '.join(untrained)} {'have' if many else 'has'} no "
                 f"trained local model: the remote decision model decides there (slower, one model call "
                 f"per request). Run 'fastworkflow train' for the local model.")
    return line


def _untrained_report(
    workflow_folderpath: str,
    missing_contexts: Optional[Sequence[str]],
    build: bool,
) -> tuple[list[str], list[str]]:
    """``startup_report``'s reading of the untrained contexts: ``(problems, the context folder names the
    matcher decides)``."""
    name = os.path.basename(os.path.normpath(workflow_folderpath))
    try:
        from fastworkflow.train.artifact_versioning import context_folder_name

        if missing_contexts is None:
            from fastworkflow.model_pipeline_training import is_workflow_trained

            missing_contexts = is_workflow_trained(workflow_folderpath)[1]
        missing = sorted(str(context) for context in missing_contexts)
        if not missing:
            return [], []
        problem = (f"Workflow '{name}' is not trained "
                   f"(missing model artifacts for context(s): {', '.join(missing)}).")
        if (build and missing == ["<workflow not built>"]
                and not heads_required() and matcher_available()):
            missing = _contexts_without_a_head(workflow_folderpath)
            if not missing:
                return [], []
            problem = (f"Workflow '{name}' is not trained "
                       f"(missing model artifacts for context(s): {', '.join(missing)}).")
        if any(context.startswith("<") for context in missing):
            return [problem], []
        global_folder = context_folder_name(GLOBAL_CONTEXT)
        for folder_name in missing:
            context_name = GLOBAL_CONTEXT if folder_name == global_folder else folder_name
            if engine_choice(workflow_folderpath, context_name) != ENGINE_MATCHER:
                return [problem], []
        return [], missing
    except Exception as error:  # noqa: BLE001 - a report that cannot be made must not be what stops a start
        return [f"Workflow '{name}': could not check its trained intent models: {type(error).__name__}: {error}"], []


# ---------------------------------------------------------------- the walk

def enclosing_scopes(workflow, record: Optional[object]) -> Iterator[Scope]:
    """The scopes above *record*, nearest first, up to and including the root record.

    Lazy on purpose: ``workflow.get_parent`` is asked for the next record only when the walk needs
    it. A record of None is the global scope and has nothing above it.
    """
    while record is not None and not (record == workflow.root):
        record = workflow.get_parent(record)
        yield Scope(fastworkflow.Workflow.get_command_context_name(record), record)


def scopes_for(workflow) -> list[Scope]:
    """The lookup path of natural language: the current context, then each enclosing record's context
    up to the root record, each with the record a command found there runs on."""
    current = workflow._current_command_context
    return [
        Scope(fastworkflow.Workflow.get_command_context_name(current), current),
        *enclosing_scopes(workflow, current),
    ]


def walk_path(workflow) -> tuple[str, ...]:
    """The ordered context names a walk from where *workflow* stands may ask: ``scopes_for``, then the global
    scope when it is asked first (``global_first_applies``). What the matcher's one grouped call lists."""
    names = [scope.context_name for scope in scopes_for(workflow)]
    if GLOBAL_CONTEXT not in names and global_first_applies(workflow):
        names.append(GLOBAL_CONTEXT)
    return tuple(dict.fromkeys(names))


def global_head_trained(workflow) -> bool:
    """Whether the workflow has a global head to ask: '*' has execution commands, and the trainer
    wrote its artifacts (``___command_info/global``, the folder ``CommandRouter`` reads for '*').
    Without one the global-first step is skipped silently, so no model is loaded for it.

    The model matcher does not change this: a workflow with no global head has no global-first step, and no
    extra matcher call is made for one. Where the step does run and the global head is not confident, '*' is
    on the matcher's scope path (``walk_path``), so its answer comes from the walk's one grouped call."""
    from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER

    try:
        commands = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath).get_command_names(GLOBAL_CONTEXT)
    except Exception:  # noqa: BLE001 - a workflow whose routing cannot be read has no global head to ask
        return False
    if not commands:
        return False
    return os.path.isfile(os.path.join(
        workflow.folderpath, "___command_info", GLOBAL_CONTEXT_FOLDER, "threshold.json"))


def global_first_applies(workflow) -> bool:
    """Whether the global scope is asked before the current one: the workflow stands on a record
    outside the global scope, it has a root record for a global command to run on, and it has a
    trained global head. The caller adds what only it knows (natural language, not a typed name)."""
    return (workflow._current_command_context_name != GLOBAL_CONTEXT
            and workflow._current_command_context is not None
            and workflow.root is not None
            and global_head_trained(workflow))


def confident_application_command(answer) -> bool:
    """A confident answer naming one application command: not a decline, not an ambiguity, not a
    bare value, and not a framework (CME) command. The reserved labels resolve to no command name."""
    return (answer.command_name is not None
            and answer.error_msg is None
            and not answer.answered_bare_value
            and not answer.is_cme_command)


class ScopeWalk:
    """The walk over scopes, with the per-scope step supplied by the caller.

    *step* takes a context name and returns that scope's answer, an object with ``command_name`` (None
    for a decline), ``error_msg`` (set for an ambiguity), ``answered_bare_value``, ``is_cme_command``
    and ``routing_hint`` (``CommandNamePrediction.Output`` is one). The walk is in two halves because
    the CME stages do work between them: an ambiguity or a framework command ends the turn after
    ``first`` and must leave the workflow's bound record alone.

    With *bind* the walk binds each scope it asks in the climb as the workflow's record, so the command
    that is found runs on the record of the scope that answered. Without it the workflow is only read.
    """

    def __init__(self, workflow, step: Callable[[str], Any], *, bind: bool = True):
        self.workflow = workflow
        self.step = step
        self.bind = bind
        #: The context names asked so far, in order.
        self.asked: list[str] = []

    def _ask(self, context_name: str):
        self.asked.append(context_name)
        return self.step(context_name)

    def first(self, scope: Scope, *, global_first: bool) -> tuple[Any, Scope]:
        """The answer of the nearest scope, and the scope it stands for.

        Confident global first: with *global_first* the global scope is asked before *scope*. A
        confident application command of the global scope is the answer and runs on the root record.
        Anything else it says is discarded and *scope* is asked as if the global scope never was.
        """
        if global_first:
            global_answer = self._ask(GLOBAL_CONTEXT)
            if confident_application_command(global_answer):
                return global_answer, Scope(GLOBAL_CONTEXT, self.workflow.root)
        return self._ask(scope.context_name), scope

    def climb(self, answer, scope: Scope) -> tuple[Any, Scope, Optional[str]]:
        """Nearest scope wins: while *answer* is a decline, ask the next enclosing scope.

        Returns the last answer, the scope that gave it, and the routing hint of the walk. A command
        stops the climb, and so does a bare value: a scope that called the utterance a value has
        said it is not a command anywhere, so no enclosing scope is asked. Past the root record the
        last decline is returned.

        The hint is the first one a declining scope composed. Every scope on the path declines the
        same token for the same reason, so the first is the whole story; it is kept here because each
        step replaces the answer.
        """
        if self.bind:
            self.workflow._command_context_for_response_generation = scope.record
        if answer.command_name is not None:
            return answer, scope, None
        routing_hint = answer.routing_hint
        parents = self._enclosing(scope.record)
        while not answer.command_name and not answer.answered_bare_value:
            parent = next(parents, None)
            if parent is None:
                break
            scope = parent
            answer = self._ask(scope.context_name)
            routing_hint = routing_hint or answer.routing_hint
        return answer, scope, routing_hint

    def _enclosing(self, record: Optional[object]) -> Iterator[Scope]:
        """The scopes above *record*, nearest first, each bound as the workflow's record when the walk binds.

        A binding walk asks the workflow where it stands at every step (``record``, ``is_record_root``)
        and moves its bound record one parent up, so the workflow's own idea of its root decides
        where the climb ends. A reading walk follows the same chain without touching the workflow.
        """
        if not self.bind:
            yield from enclosing_scopes(self.workflow, record)
            return
        workflow = self.workflow
        while workflow.record is not None and not workflow.is_record_root:
            workflow._command_context_for_response_generation = workflow.get_parent(workflow.record)
            yield Scope(fastworkflow.Workflow.get_command_context_name(workflow.record), workflow.record)


@dataclass(frozen=True)
class WalkResult:
    """What a whole walk returns: the last answer, its scope, the routing hint, the contexts asked."""

    answer: Any
    scope: Scope
    routing_hint: Optional[str] = None
    asked: tuple[str, ...] = ()


def walk(
    workflow,
    step: Callable[[str], Any],
    *,
    scope: Optional[Scope] = None,
    global_first: Optional[bool] = None,
    bind: bool = False,
) -> WalkResult:
    """Both halves of the walk in one call, for a caller with nothing to do between them.

    *scope* is where the walk starts (default: the workflow's current context and record), and
    *global_first* whether the global scope is asked first (default: ``global_first_applies``). An
    ambiguity or a framework command from the first scope asked ends the walk there.
    """
    if scope is None:
        scope = scopes_for(workflow)[0]
    if global_first is None:
        global_first = global_first_applies(workflow)
    scope_walk = ScopeWalk(workflow, step, bind=bind)
    answer, scope = scope_walk.first(scope, global_first=global_first)
    routing_hint = None
    if not answer.error_msg and not answer.is_cme_command:
        answer, scope, routing_hint = scope_walk.climb(answer, scope)
    return WalkResult(answer, scope, routing_hint, tuple(scope_walk.asked))


# ---------------------------------------------------------------- detect: the walk over the engines alone

@dataclass(frozen=True)
class _EngineStep:
    """A ``ScopeAnswer`` in the shape the walk reads. ``answer`` is None for a scope with no engine."""

    answer: Optional[ScopeAnswer]
    is_cme_command: bool = False
    routing_hint: Optional[str] = None

    @property
    def command_name(self) -> Optional[str]:
        return self.answer.labels[0] if self.answer is not None and self.answer.kind == MATCH else None

    @property
    def error_msg(self) -> Optional[str]:
        return AMBIGUOUS if self.answer is not None and self.answer.kind == AMBIGUOUS else None

    @property
    def answered_bare_value(self) -> bool:
        return self.answer is not None and self.answer.kind == BARE_VALUE


def detect(
    workflow,
    utterance: str,
    *,
    engine_of: Optional[Callable[[str], Optional[Engine]]] = None,
) -> Detection:
    """Which command *utterance* means from where *workflow* stands, asking the engines alone.

    The walk of ``ScopeWalk`` with each scope's engine as the step: no typed-name layers, no cache,
    no tie-break (those are ``CommandNamePrediction``'s, which the CME stages walk with). The workflow
    is read, never changed. *engine_of* gives the engine of a context name (default: ``engine_for``);
    a scope with no engine declines.
    """
    from fastworkflow import workflow_agent

    if engine_of is None:
        scope_path = walk_path(workflow)
        memo: dict = {}

        def engine_of(context_name: str) -> Optional[Engine]:
            return engine_for(workflow.folderpath, context_name, scope_path=scope_path, memo=memo)

    core = workflow_agent.core_command_names()
    answers: list[ScopeAnswer] = []

    def step(context_name: str) -> _EngineStep:
        engine = engine_of(context_name)
        if engine is None:
            return _EngineStep(None)
        answer = engine.decide(utterance)
        answers.append(answer)
        return _EngineStep(answer, is_cme_command=answer.kind == MATCH and answer.labels[0] in core)

    result = walk(workflow, step)
    answer = result.answer.answer
    if answer is not None and answer.kind == MATCH:
        return Detection(MATCH, command=answer.labels[0], scope=result.scope, answers=answers)
    if answer is not None and answer.kind == AMBIGUOUS:
        return Detection(
            AMBIGUOUS,
            candidates=[label for label in answer.labels
                        if not is_escalation(label) and label_of(label) != PARAMETER_VALUE_LABEL],
            scope=result.scope, answers=answers)
    return Detection(NONE, scope=result.scope, answers=answers)
