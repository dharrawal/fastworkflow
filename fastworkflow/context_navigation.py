"""The command sequence that reaches a context the caller is not in.

The agent tool refuses a command name that belongs to some other context. Naming
that context is not enough: ``go_up`` and ``reset_context`` only climb, and the
command the caller just attempted often lives *down* from here (``find_identity``
is reached from global by ``open_directory``). This module turns the declarations
the workflow already made into that sequence, and names the parameters a step
cannot fill in.

Nothing here navigates. A misnamed command must not move the workflow.
"""
from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.entry_declarations import declared_entry_commands
from fastworkflow.runtime_manifest import load_manifest, merge_and_gate

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")

#: ``go_up`` is tried before ``reset_context`` when both cost the same, because
#: ``go_up`` is the move that lands on the live parent. ``reset_context`` still
#: wins when it is shorter (two climbs versus one reset).
_KIND_ORDER = {"ascend": 0, "descend": 1, "reset": 2}


def display_context_name(context: str) -> str:
    return "global" if context == "*" else context


@dataclass(frozen=True)
class DescendEdge:
    """One command that enters ``target`` when run from ``source``."""

    source: str
    command: str
    target: str
    needs: tuple[str, ...] = ()
    required_to_enter: tuple[str, ...] = ()


@dataclass(frozen=True)
class NavStep:
    """One command on a path, and why it may not be runnable as written."""

    command: str
    enters: str
    needs: tuple[str, ...] = ()
    required_to_enter: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Move:
    command: str
    target: str
    needs: tuple[str, ...]
    required_to_enter: tuple[str, ...]
    kind: str


def live_context_chain(workflow) -> tuple[str, ...]:
    """The contexts ``go_up`` will actually visit, current first, global last.

    The hierarchy file lists every parent a context *may* have. ``go_up`` follows
    the object in hand, which has one parent. A path that climbed a parent the
    live object does not have would send the caller the wrong way.
    """
    current = getattr(workflow, "current_command_context_name", "*")
    if current == "*":
        return ("*",)
    names = [current]
    seen = {current}
    obj = getattr(workflow, "current_command_context", None)
    root = getattr(workflow, "root_command_context", None)
    while obj is not None and obj is not root:
        try:
            parent = workflow.get_parent(obj)
        except Exception:
            logger.debug("parent walk stopped at %r", names[-1], exc_info=True)
            break
        if parent is None or parent is root or parent is obj:
            break
        parent_name = type(parent).__name__
        if parent_name in seen:
            break
        seen.add(parent_name)
        names.append(parent_name)
        obj = parent
    if names[-1] != "*":
        names.append("*")
    return tuple(names)


def plan_navigation(
    current: str,
    targets: list[str],
    live_chain: tuple[str, ...],
    edges: tuple[DescendEdge, ...] | list[DescendEdge],
) -> dict[str, tuple[NavStep, ...] | None]:
    """Shortest command sequence from ``current`` to each target.

    Cost is the number of commands, then the number of parameters those commands
    cannot fill in, then the number of ``reset_context`` steps. An empty tuple
    means the caller is already there. ``None`` means nothing declared reaches it.
    """
    if not live_chain or live_chain[0] != current:
        live_chain = (current, "*") if current != "*" else ("*",)

    graph: dict[str, list[_Move]] = {}

    def add(source: str, move: _Move) -> None:
        graph.setdefault(source, []).append(move)

    for here, parent in zip(live_chain, live_chain[1:]):
        add(here, _Move("go_up", parent, (), (), "ascend"))

    nodes = set(live_chain)
    for edge in edges:
        nodes.add(edge.source)
        nodes.add(edge.target)
        if edge.source == edge.target:
            continue
        add(edge.source, _Move(
            edge.command, edge.target, edge.needs, edge.required_to_enter, "descend"))
    for node in nodes:
        if node != "*":
            add(node, _Move("reset_context", "*", (), (), "reset"))

    for source in graph:
        graph[source].sort(key=lambda move: (
            len(move.needs) + len(move.required_to_enter),
            _KIND_ORDER[move.kind],
            move.command,
        ))

    best: dict[str, tuple[int, int, int]] = {current: (0, 0, 0)}
    prev: dict[str, tuple[str, _Move] | None] = {current: None}
    heap: list[tuple[int, int, int, str]] = [(0, 0, 0, current)]
    while heap:
        steps, caveats, resets, node = heapq.heappop(heap)
        if (steps, caveats, resets) != best.get(node):
            continue
        for move in graph.get(node, ()):
            cand = (
                steps + 1,
                caveats + len(move.needs) + len(move.required_to_enter),
                resets + (1 if move.kind == "reset" else 0),
            )
            if move.target not in best or cand < best[move.target]:
                best[move.target] = cand
                prev[move.target] = (node, move)
                heapq.heappush(heap, (*cand, move.target))

    planned: dict[str, tuple[NavStep, ...] | None] = {}
    for target in targets:
        if target not in best:
            planned[target] = None
            continue
        steps_rev: list[NavStep] = []
        node = target
        while prev[node] is not None:
            source, move = prev[node]
            steps_rev.append(NavStep(
                move.command, move.target, move.needs, move.required_to_enter))
            node = source
        steps_rev.reverse()
        planned[target] = tuple(steps_rev)
    return planned


def navigation_steps_to_context(workflow, current: str, target: str) -> str | None:
    """The command sequence to reach *target* from *current*, or None if undeclared.

    Returns an empty string when *current* is already *target*. Returns ``None``
    when no path exists.
    """
    if current == target:
        return ""
    try:
        chain = live_context_chain(workflow)
        edges = collect_descend_edges(workflow.folderpath)
        path = plan_navigation(current, [target], chain, edges).get(target)
    except Exception:
        logger.debug("navigation steps not computed", exc_info=True)
        return None
    if path is None:
        return None
    if not path:
        return ""
    return ", then ".join(_format_step(step) for step in path)


def format_how_to_enter_context(workflow, current: str, target: str) -> str:
    """Prescriptive steps to reach *target* from the workflow's current context."""
    current_label = display_context_name(current)
    target_label = display_context_name(target)
    steps = navigation_steps_to_context(workflow, current, target)
    if steps is None:
        return (
            f"How to enter {target_label} from {current_label}: "
            "no declared navigation path."
        )
    if steps == "":
        return f"You are already in {target_label}."
    return f"How to enter {target_label} from {current_label}: {steps}."


def render_unavailable_command(
    token: str,
    current: str,
    homes: list[str],
    paths: dict[str, tuple[NavStep, ...] | None],
) -> str:
    """The refusal the agent tool raises."""
    current_label = display_context_name(current)
    home_labels = ", ".join(repr(display_context_name(home)) for home in homes)
    text = (
        f"Command {token!r} is not available in the current context "
        f"{current_label!r}. It is available in: {home_labels}."
    )
    sentences = []
    for home in homes:
        label = display_context_name(home)
        path = paths.get(home)
        if path is None:
            sentences.append(
                f"No declared navigation path from {current_label!r} to {label!r}."
            )
        elif not path:
            sentences.append(
                f"{label!r} is the current context. Retry {token!r}."
            )
        else:
            rendered = ", then ".join(_format_step(step) for step in path)
            sentences.append(
                f"From here to {label!r}: {rendered}. Then retry {token!r}."
            )
    return text + " " + " ".join(sentences)


def unavailable_command_message(
    workflow, token: str, current: str, homes: list[str],
) -> str:
    """Path from ``workflow``'s current context to each context that has ``token``."""
    try:
        chain = live_context_chain(workflow)
        edges = collect_descend_edges(workflow.folderpath)
        paths = plan_navigation(current, homes, chain, edges)
    except Exception:
        logger.debug("navigation path not computed", exc_info=True)
        paths = {home: None for home in homes}
    return render_unavailable_command(token, current, homes, paths)


def collect_descend_edges(workflow_folderpath: str) -> tuple[DescendEdge, ...]:
    """Descend commands declared by the manifest and by ``enter_command``.

    A manifest entry whose kind is not ``descend`` is not an edge, even when a
    context names that command as its ``enter_command``: the command's own
    effect is the fact, and a placeholder command must not be described as a
    way in. ``enter_command`` supplies the parameter names the manifest does
    not (``open_account_by_uid <account_uid>``) and is the only edge when the
    workflow has no manifest.
    """
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
        contexts: dict[str, list[str]] = dict(routing.contexts)
    except Exception:
        logger.debug("routing definition unread", exc_info=True)
        return ()

    effects, occupiable = _navigation_effects(workflow_folderpath)
    raw: dict[tuple[str, str, str], _RawEdge] = {}

    for manifest_key, effect in effects.items():
        if effect.kind != "descend":
            continue
        sources = _contexts_for_manifest_key(manifest_key, contexts)
        routing_name = _routing_name(manifest_key, contexts)
        gate = effect.when_parameter_present
        for target in effect.declared_targets():
            if not _enterable(target, occupiable):
                continue
            for source in sources:
                if source == target:
                    continue
                shown = _shown_command(source, manifest_key, contexts)
                _put(raw, _RawEdge(source, shown, routing_name or shown, target, gate, ()))

    for context_name, names in contexts.items():
        if not _enterable(context_name, occupiable):
            continue
        for declaration in declared_entry_commands(workflow_folderpath, context_name):
            token, placeholders = _parse_enter(declaration)
            if not token:
                continue
            effect = _effect_for_token(token, effects, contexts)
            if effect is not None and effect.kind != "descend":
                continue
            if effect is not None and context_name not in effect.declared_targets():
                continue
            for source, routing_name in _sources_for_token(token, contexts):
                if source == context_name:
                    continue
                shown = _shown_command(source, routing_name, contexts)
                _put(raw, _RawEdge(
                    source, shown, routing_name, context_name, None, placeholders))

    schema: dict[str, tuple[str, ...] | None] = {}
    edges: dict[tuple[str, str, str], DescendEdge] = {}
    for record in raw.values():
        if record.routing_name not in schema:
            schema[record.routing_name] = _required_parameters(routing, record.routing_name)
        needs, required = _classify(
            schema[record.routing_name], record.placeholders, record.gate)
        key = (record.source, record.command, record.target)
        edge = DescendEdge(record.source, record.command, record.target, needs, required)
        existing = edges.get(key)
        edges[key] = edge if existing is None else _merge_edge(existing, edge)
    return tuple(edges.values())


@dataclass(frozen=True)
class _RawEdge:
    source: str
    command: str
    routing_name: str
    target: str
    gate: str | None
    placeholders: tuple[str, ...]


def _put(raw: dict[tuple[str, str, str], _RawEdge], record: _RawEdge) -> None:
    key = (record.source, record.command, record.target)
    existing = raw.get(key)
    if existing is None:
        raw[key] = record
        return
    raw[key] = _RawEdge(
        record.source,
        record.command,
        existing.routing_name or record.routing_name,
        record.target,
        existing.gate or record.gate,
        _union(existing.placeholders, record.placeholders),
    )


def _navigation_effects(workflow_folderpath: str):
    """Every declared transition, including those that do not descend.

    A ``none`` or ``temporary`` effect has to stay visible. Dropping it makes
    ``enter_command`` look like the only statement about that command, and a
    placeholder that does not navigate gets offered as a way in.
    """
    try:
        # Navigation effects are facts about commands, not a feature gate.
        # An ambient FASTWORKFLOW_RUNTIME_FEATURES value must not erase them.
        metadata = merge_and_gate(load_manifest(workflow_folderpath), env={})
    except Exception:
        logger.debug("runtime manifest unread", exc_info=True)
        return {}, {}
    effects = {}
    for name, declaration in metadata.commands.items():
        effect = declaration.navigation_effect
        if effect is not None:
            effects[name] = effect
    occupiable = {
        name: metadata.is_occupiable(name) for name in metadata.contexts
    }
    return effects, occupiable


def _enterable(context_name: str, occupiable: dict[str, bool | None]) -> bool:
    """A context the manifest marks non-occupiable is a mixin, not a place."""
    return occupiable.get(context_name) is not False


def _contexts_for_manifest_key(
    manifest_key: str, contexts: dict[str, list[str]],
) -> tuple[str, ...]:
    """Contexts whose command list contains this manifest command.

    Global commands are stored unqualified on ``*`` (``open_directory``) while a
    generator may key the same command by its spec root (``IDO/open_directory``).
    The prefix is not a context, so the unqualified command is the match. A
    prefix that *is* a context only matches that qualified name, so
    ``Identity/list_accounts`` is not ``Application/list_accounts``.
    """
    known = set(contexts)
    prefix, separator, short = manifest_key.partition("/")
    if not separator:
        short = prefix
        prefix = "*"
    matched: list[str] = []
    for context_name, names in contexts.items():
        for name in names:
            if name == manifest_key:
                matched.append(context_name)
                break
            name_prefix, name_sep, name_short = name.partition("/")
            if not name_sep:
                name_prefix, name_short = "*", name
            if name_short != short:
                continue
            if name_prefix == prefix or (prefix not in known and name_prefix == "*"):
                matched.append(context_name)
                break
    return tuple(matched)


def _routing_name(manifest_key: str, contexts: dict[str, list[str]]) -> str | None:
    """The command-directory key ``get_command_class`` can load."""
    for names in contexts.values():
        if manifest_key in names:
            return manifest_key
    prefix, separator, short = manifest_key.partition("/")
    if not separator:
        return manifest_key if any(manifest_key in names for names in contexts.values()) else None
    if prefix in contexts:
        return None
    for names in contexts.values():
        if short in names:
            return short
    return None


def _sources_for_token(
    token: str, contexts: dict[str, list[str]],
) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for context_name, names in contexts.items():
        for name in names:
            if name.split("/")[-1] == token:
                found.append((context_name, name))
                break
    return tuple(found)


def _effect_for_token(token: str, effects: dict, contexts: dict[str, list[str]]):
    for manifest_key, effect in effects.items():
        if manifest_key.split("/")[-1] != token:
            continue
        if _contexts_for_manifest_key(manifest_key, contexts):
            return effect
    return None


def _shown_command(
    source: str, manifest_or_routing_name: str, contexts: dict[str, list[str]],
) -> str:
    """Short name, unless this context has two commands with that tail."""
    short = manifest_or_routing_name.split("/")[-1]
    collisions = [
        name for name in contexts.get(source, ())
        if name.split("/")[-1] == short
    ]
    if len(collisions) <= 1:
        return short
    if manifest_or_routing_name in collisions:
        return manifest_or_routing_name
    return collisions[0]


def _parse_enter(declaration: str) -> tuple[str, tuple[str, ...]]:
    parts = declaration.split(None, 1)
    if not parts:
        return "", ()
    return parts[0], tuple(_PLACEHOLDER.findall(declaration))


def _required_parameters(routing, command_name: str) -> tuple[str, ...] | None:
    """Input fields with no default. ``None`` when the signature cannot be read.

    ``None`` keeps the declaration's placeholders. An empty tuple means the
    signature was read and the command can be invoked with no arguments, which
    is a different fact from not having looked.
    """
    try:
        model = routing.get_command_class(
            command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:
        logger.debug("parameters unread for %r", command_name, exc_info=True)
        return None
    fields = getattr(model, "model_fields", None)
    if not fields:
        return ()
    required = []
    for name, field in fields.items():
        is_required = getattr(field, "is_required", None)
        if callable(is_required) and is_required():
            required.append(name)
    return tuple(required)


def _classify(
    schema_required: tuple[str, ...] | None,
    placeholders: tuple[str, ...],
    gate: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split parameters into 'must pass to run' and 'must pass to enter'.

    A required input is ``needs``. ``when_parameter_present`` on an optional
    field is the second kind: the command runs without it and stays put.
    Placeholders on ``enter_command`` join that second kind when the schema
    does not already call them required, so a signature that failed to import
    still names the value.
    """
    needs = tuple(schema_required or ())
    required: list[str] = []
    if gate and gate not in needs:
        required.append(gate)
    for name in placeholders:
        if name in needs or name in required:
            continue
        required.append(name)
    return needs, tuple(required)


def _union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for name in (*left, *right):
        if name not in seen:
            seen.add(name)
            merged.append(name)
    return tuple(merged)


def _merge_edge(left: DescendEdge, right: DescendEdge) -> DescendEdge:
    needs = _union(left.needs, right.needs)
    required = tuple(
        name for name in _union(left.required_to_enter, right.required_to_enter)
        if name not in needs
    )
    return DescendEdge(left.source, left.command, left.target, needs, required)


def _format_step(step: NavStep) -> str:
    holes = list(step.needs)
    holes.extend(name for name in step.required_to_enter if name not in step.needs)
    text = step.command
    if holes:
        text += "".join(f" <{name}>" for name in holes)
    notes = [f"needs {name}" for name in step.needs]
    if step.required_to_enter:
        target = display_context_name(step.enters)
        joined = ", ".join(step.required_to_enter)
        notes.append(f"{joined} required to enter {target}")
    if notes:
        text += " (" + "; ".join(notes) + ")"
    return text
