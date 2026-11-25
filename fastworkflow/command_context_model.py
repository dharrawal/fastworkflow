from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any
import os
from pathlib import Path

import fastworkflow
from fastworkflow.command_directory import get_cached_command_directory, is_visible_planning
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_inheritance_model import WorkflowInheritanceModel

#: Workflow folders already told that their ``context_hierarchy_model.json`` is ignored.
#: ``CommandContextModel.load`` runs on every ``Workflow.get_parent`` call, so the warning is
#: deduplicated per folder for the life of the process: one warning per workflow load.
_HIERARCHY_FILE_WARNED: set[str] = set()

"""Utility for loading and traversing the workflow command context model.

A context is a folder of ``_commands/`` (``context_folders``). Its facts are declared on the ``class Context`` of its
``_commands/<Context>/_<Context>.py`` and read as literals from the class body (no import, ``read_declarations``):

- ``inherits = ["Base", ...]``: contexts whose commands this context also has (``"*"``: the global commands).
- ``parents = ["Ctx", ...]``: every context a record of this context can sit in (the contexts ``get_parent`` returns
  records of; the context itself when records nest). No ``parents``: a top-level context, enclosed only by the global
  commands. A context declares ``parents`` exactly when it has ``get_parent``.

``_commands/context_inheritance_model.json`` is a removed spelling: its presence is a load error.

This loader performs:
1. Reading and checking the declarations (names, duplicates, self-inheritance, the removed file).
2. Discovery of contexts and their direct commands from the ``_commands/`` filesystem structure.
3. Merging of filesystem-discovered commands with the declared inheritance (``"base"`` in the merged model).
4. Cycle detection for the inheritance graph.
5. Resolution of effective command lists per context (own commands + inherited commands).
"""


#: The callback-class methods whose first parameter is the record (renamed from ``command_context_object``).
_RECORD_PARAMETER_METHODS = ("get_parent", "get_displayname", "get_state", "get_locator")


#: Every public member a context callback class may have: the hooks the framework reads. Anything else public is a
#: misspelling (a hook the framework would never call); a helper's name starts with ``_``.
CONTEXT_HOOKS = ("get_parent", "get_displayname", "instance_label_attr", "get_state", "from_state", "state_version",
                 "get_locator", "find_by_locator", "occupiable", "inherits", "parents")
#: The two declarations of a context, read as literals from the body of its ``class Context`` (``read_declarations``).
CONTEXT_DECLARATIONS = ("inherits", "parents")
#: The removed file that declared inheritance before the declarations (``removed_inheritance_file_problem``).
REMOVED_INHERITANCE_FILE = "context_inheritance_model.json"
#: Retired members, each with its own load error elsewhere (``instance_label`` here, ``enter_command(s)`` in
#: ``entry_declarations``).
_RETIRED_CONTEXT_MEMBERS = ("instance_label", "enter_command", "enter_commands")


def context_class_problems(context_class: Any) -> list[str]:
    """The load errors of a context callback class, empty when it loads.

    Two removed spellings fail the load, each naming its replacement: a classmethod
    ``instance_label`` (make the label a property on the record object and name it with
    ``instance_label_attr``), and a parameter ``command_context_object`` on
    ``get_parent``, ``get_displayname``, ``get_state`` or ``get_locator`` (name it ``record``).
    """
    problems = []
    if hasattr(context_class, "instance_label"):
        problems.append(
            "instance_label is removed: make the label a property on the record object "
            "and name it with instance_label_attr = \"<property name>\" (e.g. \"instance_label\")")
    for method_name in _RECORD_PARAMETER_METHODS:
        method = getattr(context_class, method_name, None)
        if method is None:
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            continue
        if "command_context_object" in parameters:
            problems.append(
                f'{method_name}: parameter "command_context_object" is removed: name it "record"')
    problems += containment_problems(context_class)
    for name in sorted(dir(context_class)):
        if name.startswith("_") or name in CONTEXT_HOOKS or name in _RETIRED_CONTEXT_MEMBERS:
            continue
        close = difflib.get_close_matches(name, CONTEXT_HOOKS, n=1)
        problems.append(
            f"Context.{name} is not a context hook{f'; did you mean {close[0]!r}?' if close else '.'} The hooks are "
            f"{', '.join(CONTEXT_HOOKS)}; a helper starts with _")
    return problems


def containment_problems(context_class: Any) -> list[str]:
    """The load errors of a context class's ``parents`` against its hooks: ``parents`` is declared exactly when the
    class has ``get_parent`` (one inherited from a Python base class counts), a context declared ``occupiable =
    False`` declares none, and both declarations are written in the class body itself (they are read there as
    literals, ``read_declarations``)."""
    problems = []
    for name in CONTEXT_DECLARATIONS:
        if hasattr(context_class, name) and name not in vars(context_class):
            problems.append(f"{name} is inherited from a base class: write {name} = [...] in the body of class "
                            f"Context itself (it is read there, as a literal, when the workflow loads)")
    declared = "parents" in vars(context_class)
    has_get_parent = hasattr(context_class, "get_parent")
    if getattr(context_class, "occupiable", None) is False:
        if declared:
            problems.append("parents is declared on a context declared occupiable = False: the workflow never "
                            "stands on one of its records, so nothing contains them. Remove parents")
        return problems
    if declared and not has_get_parent:
        problems.append("parents is declared but there is no get_parent: write '@classmethod def get_parent(cls, "
                        "record)' returning the enclosing record, or remove parents for a top-level context")
    elif has_get_parent and not declared:
        problems.append("get_parent is written but no parents is declared: list every context a record of this "
                        "context can sit in, parents = [\"<Context>\", ...] (the contexts get_parent returns records "
                        "of; remove get_parent instead when it only returns None)")
    return problems


def removed_inheritance_file_problem(workflow_path: str | Path) -> str | None:
    """The load error of a workflow (or a base workflow) that still has ``_commands/context_inheritance_model.json``,
    or None. The file is a removed spelling and is not read."""
    workflow_path_obj = Path(workflow_path)
    base_paths = [Path(p) for p in WorkflowInheritanceModel.load(str(workflow_path_obj))
                  .resolve_base_paths(str(workflow_path_obj))]
    for workflow_dir in (workflow_path_obj, *base_paths):
        path = workflow_dir / "_commands" / REMOVED_INHERITANCE_FILE
        if path.is_file():
            return (f"{path}: {REMOVED_INHERITANCE_FILE} is removed and not read: declare each context's bases as "
                    f"inherits = [\"<Context>\", ...] and what contains its records as parents = [\"<Context>\", ...] "
                    f"in the class Context of _commands/<Context>/_<Context>.py, then delete the file "
                    f"(`fastworkflow migrate-contexts <workflow>` writes the inherits declarations from it)")
    return None


@dataclass(frozen=True)
class ContextDeclarations:
    """What a context's ``class Context`` declares (None: not declared) and the file it is written in."""

    file: Path
    inherits: tuple[str, ...] | None = None
    parents: tuple[str, ...] | None = None


_DECLARATIONS_CACHE: dict[tuple, dict[str, ContextDeclarations]] = {}


def _literal_names(node: Any, name: str, where: str) -> tuple[str, ...]:
    """The names of a ``inherits`` / ``parents`` value, which must be a non-empty list of string literals."""
    import ast

    if not isinstance(node, (ast.List, ast.Tuple)) or not all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts):
        raise CommandContextModelValidationError(
            f"{where}: {name} must be a list of context names written as string literals, "
            f"{name} = [\"<Context>\", ...] (it is read as written, without importing the file)")
    if not node.elts:
        raise CommandContextModelValidationError(
            f"{where}: {name} = [] declares nothing: remove it (omitting {name} is the only spelling of none)")
    return tuple(item.value for item in node.elts)


def _declarations_in_file(path: Path) -> ContextDeclarations:
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return ContextDeclarations(file=path)   # the file's own load error is reported where it is imported
    found: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Context"):
            continue
        for statement in node.body:
            targets = (statement.targets if isinstance(statement, ast.Assign)
                       else [statement.target] if isinstance(statement, ast.AnnAssign) else [])
            for target in targets:
                if isinstance(target, ast.Name) and target.id in CONTEXT_DECLARATIONS:
                    where = f"{path}:{statement.lineno}"
                    if target.id in found:
                        raise CommandContextModelValidationError(
                            f"{where}: {target.id} is assigned twice in class Context: write it once")
                    found[target.id] = _literal_names(statement.value, target.id, where)
    return ContextDeclarations(file=path, inherits=found.get("inherits"), parents=found.get("parents"))


def _check_declared_names(declarations: dict[str, ContextDeclarations], folders: set[str]) -> None:
    """Raise :class:`CommandContextModelValidationError` with one line per mistake in the names the declarations
    give: a name that is no context (with the folder it probably meant), a name given twice, a context that inherits
    itself. ``"*"`` (the global commands) may be inherited, never a parent."""
    lines = []
    for context, declared in sorted(declarations.items()):
        for name, values, allowed in (("inherits", declared.inherits, folders | {"*"}),
                                      ("parents", declared.parents, folders)):
            for value in values or ():
                if value in allowed:
                    continue
                close = difflib.get_close_matches(value, sorted(allowed), n=1)
                lines.append(f"{declared.file}: {name} names {value!r}, which is no context"
                             f"{f'; did you mean {close[0]!r}?' if close else '.'} A context is a folder of "
                             f"_commands/; the contexts are: {', '.join(sorted(folders)) or 'none'}")
            for value in sorted({v for v in values or () if list(values).count(v) > 1}):
                lines.append(f"{declared.file}: {name} names {value!r} twice: name each context once")
        if context in (declared.inherits or ()):
            lines.append(f"{declared.file}: inherits names {context!r}, the context itself: a context has its own "
                         f"commands already; remove it")
    if lines:
        raise CommandContextModelValidationError("\n".join(lines))


def read_declarations(workflow_path: str | Path) -> dict[str, ContextDeclarations]:
    """``{context: ContextDeclarations}`` for every context folder (``context_folders``) whose
    ``_<Context>.py`` exists, read as literals from the body of its ``class Context`` without importing it.

    Raises :class:`CommandContextModelValidationError` for the removed ``context_inheritance_model.json``, a
    declaration that is not a non-empty list of string literals, and a name that is no context, given twice or the
    context itself (inherits). Cached on the files' modification times."""
    workflow_path_obj = Path(workflow_path)
    if problem := removed_inheritance_file_problem(workflow_path_obj):
        raise CommandContextModelValidationError(problem)
    folders = context_folders(workflow_path_obj)
    files = _context_class_files(workflow_path_obj, folders)
    key = (str(workflow_path_obj.resolve()), tuple(sorted(folders)),
           tuple((str(path), path.stat().st_mtime_ns) for _, path in sorted(files.items()) if path.is_file()))
    if (cached := _DECLARATIONS_CACHE.get(key)) is not None:
        return cached
    declarations = {name: _declarations_in_file(path) for name, path in files.items() if path.is_file()}
    _check_declared_names(declarations, set(folders))
    _DECLARATIONS_CACHE[key] = declarations
    return declarations


def _context_class_files(workflow_path: Path, folders: dict[str, Path]) -> dict[str, Path]:
    """The ``_<Context>.py`` the context class of each context is loaded from: the last of (base workflows, this
    workflow) that has one, as ``CommandDirectory`` registers it. A folder of this workflow with no ``_<Context>.py``
    keeps a base workflow's context class, and so its declarations."""
    base_paths = [Path(p) for p in WorkflowInheritanceModel.load(str(workflow_path))
                  .resolve_base_paths(str(workflow_path))]
    files = {name: folder / f"_{name}.py" for name, folder in folders.items()}
    for name in folders:
        for workflow_dir in (workflow_path, *reversed(base_paths)):
            candidate = workflow_dir / "_commands" / name / f"_{name}.py"
            if candidate.is_file():
                files[name] = candidate
                break
    return files


def context_folders(workflow_path: str | Path) -> dict[str, Path]:
    """Every context of the workflow, as its folder: a directory of ``_commands/`` whose name does not start with
    ``_`` or ``.``, in this workflow or a base workflow (``workflow_inheritance_model.json``; this workflow's folder
    wins). A folder is a context whether or not it holds a command file of its own. The one definition of a context
    folder, read by :meth:`CommandContextModel.load` and :meth:`CommandContextModel.load_base_contexts`."""
    workflow_path_obj = Path(workflow_path)
    base_paths = [Path(p) for p in WorkflowInheritanceModel.load(str(workflow_path_obj))
                  .resolve_base_paths(str(workflow_path_obj))]
    folders: dict[str, Path] = {}
    for workflow_dir in (*base_paths, workflow_path_obj):
        commands_dir = workflow_dir / "_commands"
        if commands_dir.is_dir():
            for entry in sorted(commands_dir.iterdir()):
                if entry.is_dir() and not entry.name.startswith(("_", ".")) and entry.name != "Core":
                    folders[entry.name] = entry
    return folders


def _check_no_empty_folder(folders: dict[str, Path], contexts: dict[str, dict[str, list[str]]],
                           cmd_dir: Any) -> None:
    """Raise :class:`CommandContextModelValidationError` naming every context folder that holds no command file and
    inherits none: no scope of ``(folder, *its bases)`` holds a command file (``*`` the global ones). Command files
    count, loaded or not: a command that does not load is that command's own problem. One line per folder."""
    # a core command is in no folder of an application (its folders are the core workflow's own), but the global
    # core command (wildcard) gives an application's "*" nothing
    core = set(cmd_dir.core_command_names)
    owning = {name.split("/", 1)[0] if "/" in name else "*"
              for name in (*cmd_dir.map_command_2_metadata, *cmd_dir.load_errors) if "/" in name or name not in core}
    lines = [
        f"{path}: the context folder {name!r} holds no command file and inherits none (no inherits in its class "
        f"Context, or every context it inherits from is empty): add a command, declare inherits = [\"<Context>\"] "
        f"naming a context that has commands in the class Context of {path}/_{name}.py, or move helper modules to "
        f"_commands/_shared/"
        for name, path in sorted(folders.items())
        if not ({name, *CommandContextModel._ordered_bases_from_map(name, contexts)} & owning)]
    if lines:
        raise CommandContextModelValidationError("\n".join(lines))


class CommandContextModelError(Exception):
    """Base class for errors in this module."""


class CommandContextModelValidationError(CommandContextModelError):
    """Raised when the context model fails validation."""


@dataclass
class CommandContextModel:
    """Represents the command context model for a workflow."""

    _workflow_path: str
    _command_contexts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    #: ``{context: [parent, ...]}`` for each context that declares ``parents`` (``read_declarations``).
    _parents: dict[str, list[str]] = field(default_factory=dict)
    _resolved_commands: dict[str, list[str]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        # Inheritance is resolved lazily on first `commands()` call to
        # avoid upfront DFS across all contexts during cold-start.
        pass

    @classmethod
    def load(cls, workflow_path: str | Path) -> CommandContextModel:
        """Loads and validates the command context model from a workflow path,
        augmenting with contexts and commands discovered from the _commands directory. A context is a folder of
        ``_commands/`` (``context_folders``); a key of the inheritance file must name one, and a folder must hold a
        command file, execution or planning, or inherit one. Its ``"/"`` list holds its execution commands only
        (planning commands are read apart, ``RoutingDefinition.get_planning_command_names``)."""
        workflow_path_obj = Path(workflow_path)

        # 1. The declarations of each class Context (inherits, parents), read without importing
        declarations = read_declarations(workflow_path_obj)
        raw_contexts_from_json = cls._inheritance_of(declarations)
        
        # 2. Use cached CommandDirectory to get filesystem-derived commands
        cmd_dir = get_cached_command_directory(str(workflow_path_obj))
        fs_derived_direct_commands = {}
        for qualified_cmd_name in cmd_dir.get_execution_commands():
            if "/" in qualified_cmd_name:
                context_part, _ = qualified_cmd_name.split("/", 1)
                if context_part not in fs_derived_direct_commands:
                    fs_derived_direct_commands[context_part] = {"/": []}
                fs_derived_direct_commands[context_part]["/"].append(qualified_cmd_name)
            else:
                if "*" not in fs_derived_direct_commands:
                    fs_derived_direct_commands["*"] = {"/": []}
                fs_derived_direct_commands["*"]["/"].append(qualified_cmd_name)
        # A folder named after a class is a context whatever kind of command it holds: one that holds only planning
        # commands is a context with no execution command of its own (bd fix-ftdl.7.3).
        for qualified_cmd_name, metadata in cmd_dir.map_command_2_metadata.items():
            if is_visible_planning(metadata) and "/" in qualified_cmd_name:
                fs_derived_direct_commands.setdefault(qualified_cmd_name.split("/", 1)[0], {"/": []})
        fs_derived_direct_commands.pop('Core', None)
        for context_data in fs_derived_direct_commands.values():
            context_data["/"] = sorted(list(set(context_data["/"])))

        # Every context is a folder (this workflow's or a base workflow's): a key that is none is a mistake.
        folders = context_folders(workflow_path_obj)

        # 3. Merge JSON-defined inheritance with Filesystem-derived direct commands
        merged_contexts = {}
        all_context_names = (set(raw_contexts_from_json.keys()) | set(fs_derived_direct_commands.keys())
                             | set(folders))
        for name in all_context_names:
            # A JSON entry always has "base" (validated on load) and a filesystem-derived entry holds a command
            # (its execution commands are its "/" list, which is empty for a folder of planning commands only).
            current_context_definition: dict[str, list[str]] = {}
            if "base" in (json_def := raw_contexts_from_json.get(name, {})):
                current_context_definition["base"] = json_def["base"]
            current_context_definition["/"] = fs_derived_direct_commands.get(name, {}).get("/", [])
            merged_contexts[name] = current_context_definition

        # Every declared base is checked now, at load, not when a context's commands are first asked for; the
        # error names the file and the context that declares the base.
        for name in sorted(raw_contexts_from_json):
            try:
                cls._validate_context_inheritance(name, merged_contexts)
            except CommandContextModelValidationError as error:
                raise CommandContextModelValidationError(f"{declarations[name].file}: {error}") from None
        _check_no_empty_folder(folders, merged_contexts, cmd_dir)

        # Create the instance with the merged command contexts
        instance = cls(_workflow_path=str(workflow_path_obj), _command_contexts=merged_contexts,
                       _parents={name: list(declared.parents) for name, declared in declarations.items()
                                 if declared.parents})

        # 4. The parent chain is not read from a file. The runtime climbs the live chain
        #    (Workflow.get_parent -> the context class's get_parent). A leftover
        #    context_hierarchy_model.json is ignored, and warned about once per workflow.
        _warn_if_hierarchy_file_ignored(workflow_path_obj)

        return instance

    @staticmethod
    def _inheritance_of(declarations: dict[str, ContextDeclarations]) -> dict[str, dict[str, list[str]]]:
        """``{context: {"base": [...]}}`` for each context that declares ``inherits``: the merged model's shape."""
        return {name: {"base": list(declared.inherits)} for name, declared in declarations.items()
                if declared.inherits}

    @classmethod
    def declared_bases(cls, workflow_path: str | Path) -> dict[str, list[str]]:
        """``{context: [base, ...]}`` exactly as each context's ``inherits`` declares them (direct bases, in
        declared order), validated: the one reader of the declared inheritance for display and for the scoring
        tools (``train/score_listing.py``, ``train/score_lookup.py``)."""
        return {name: list(entry["base"]) for name, entry in cls._inheritance_of(read_declarations(workflow_path)).items()}

    @classmethod
    def declared_parents(cls, workflow_path: str | Path) -> dict[str, list[str]]:
        """``{context: [parent, ...]}`` exactly as each context's ``parents`` declares them, validated, without
        importing anything. A context with no entry is top level."""
        return {name: list(declared.parents) for name, declared in read_declarations(workflow_path).items()
                if declared.parents}

    def parents(self, context_name: str) -> list[str] | None:
        """The declared ``parents`` of *context_name*, or None when it declares none (a top-level context)."""
        return self._parents.get(context_name)

    @classmethod
    def _validate_context_inheritance(
        cls,
        context_name: str,
        contexts: dict[str, dict[str, list[str]]],
        valid_contexts: set[str] | None = None,
        visiting: set[str] | None = None,
    ) -> None:
        """Validate that *context_name* and its transitive ``base`` contexts exist in
        *valid_contexts* (defaults to ``set(contexts)``) and contain no inheritance cycle."""
        # "*" (the global context) is always a valid base: routing gives it a context even when no
        # top-level command file exists (command_routing: resolved_contexts['*']), so it is never missing.
        allowed = (set(contexts) | {"*"}) if valid_contexts is None else valid_contexts
        if context_name not in allowed:
            # A context must be defined either by having command files
            # or by being part of a command inheritance structure. If it's not
            # in _command_contexts, it's an unknown/invalid context.
            raise CommandContextModelValidationError(f"Context '{context_name}' not found in model.")

        if visiting is None:
            visiting = set()

        if context_name in visiting:
            raise CommandContextModelValidationError(
                f"Inheritance cycle detected: {' -> '.join(list(visiting) + [context_name])}"
            )

        next_visiting = visiting.copy()
        next_visiting.add(context_name)
        for base_context_name in contexts.get(context_name, {}).get("base") or []:
            if base_context_name not in allowed:
                known = ", ".join(sorted(allowed - {"*"})) or "none"
                raise CommandContextModelValidationError(
                    f"context {context_name!r} declares the base {base_context_name!r}, which is no context of "
                    f"the workflow (a context is a folder of _commands/; its contexts: "
                    f"{known})")
            cls._validate_context_inheritance(
                base_context_name, contexts, allowed, next_visiting
            )

    @staticmethod
    def _ordered_bases_from_map(
        context_name: str,
        contexts: dict[str, dict[str, list[str]]],
    ) -> list[str]:
        """Depth-first declared base contexts for *context_name*, each once, from *contexts*."""
        ordered: list[str] = []
        seen: set[str] = {context_name}

        def walk(name: str) -> None:
            for base in contexts.get(name, {}).get("base") or []:
                if base in seen:
                    continue
                seen.add(base)
                ordered.append(base)
                walk(base)

        walk(context_name)
        return ordered

    @classmethod
    def load_base_contexts(
        cls,
        workflow_path: str | Path,
        known_contexts: set[str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Read and validate each context's ``inherits`` (``read_declarations``) and return
        ``{context: (base, ...)}`` in depth-first declared order without importing any command
        or creating ``___command_info``.

        Verifies that every referenced base context exists (a context folder, or in *known_contexts*
        when given) and rejects inheritance cycles with :class:`CommandContextModelValidationError`.
        """
        workflow_path_obj = Path(workflow_path)
        declarations = read_declarations(workflow_path_obj)
        raw_contexts = cls._inheritance_of(declarations)
        if not raw_contexts:
            return {}

        known = set(context_folders(workflow_path_obj)) if known_contexts is None else set(known_contexts)

        # The global context "*" always exists (command routing gives it a context even when no
        # global command does), so a base of "*" names the global commands and is never missing.
        valid_contexts = set(raw_contexts.keys()) | known | {"*"}
        for ctx_name in raw_contexts:
            try:
                cls._validate_context_inheritance(ctx_name, raw_contexts, valid_contexts)
            except CommandContextModelValidationError as error:
                raise CommandContextModelValidationError(f"{declarations[ctx_name].file}: {error}") from None

        all_contexts = (set(raw_contexts.keys()) | known) - {"*", ""}
        return {
            ctx: bases
            for ctx in sorted(all_contexts)
            if (bases := tuple(cls._ordered_bases_from_map(ctx, raw_contexts)))
        }

    def _resolve_inheritance(self) -> None:
        """Resolves command inheritance and detects cycles."""
        for context_name in self._command_contexts:
            self._validate_context_inheritance(context_name, self._command_contexts)
            self.commands(context_name)

    def commands(self, context_name: str, visiting: set[str] | None = None) -> list[str]:
        """Returns the effective list of commands for a given command context, including inherited ones."""
        if context_name in self._resolved_commands:
            return self._resolved_commands[context_name]

        self._validate_context_inheritance(
            context_name, self._command_contexts, visiting=visiting
        )
        if visiting is None:
            visiting = set()
        visiting.add(context_name)

        context_def = self._command_contexts.get(context_name, {})
        own_commands_list = context_def.get("/") or []

        simple_to_qualified = {}

        for own_cmd_qualified in own_commands_list:
            own_cmd_simple = own_cmd_qualified.split('/')[-1]
            simple_to_qualified[own_cmd_simple] = own_cmd_qualified

        base_contexts_list = context_def.get("base") or []
        for base_context_name in base_contexts_list:
            inherited_commands_qualified_list = self.commands(base_context_name, visiting.copy()) 

            for inherited_cmd_qualified in inherited_commands_qualified_list:
                inherited_cmd_simple = inherited_cmd_qualified.split('/')[-1]
                if inherited_cmd_simple not in simple_to_qualified:
                    simple_to_qualified[inherited_cmd_simple] = inherited_cmd_qualified

        final_effective_commands_list = sorted(simple_to_qualified.values())
        self._resolved_commands[context_name] = final_effective_commands_list

        visiting.remove(context_name)

        return final_effective_commands_list

    def inherited_base_contexts(self, context_name: str) -> set[str]:
        """Every context whose commands *context_name* inherits, transitively.

        Unknown contexts (including the global ``*``) inherit nothing. Cycles
        are rejected when the model loads, so the walk terminates.
        """
        bases: set[str] = set()
        pending = list(self._command_contexts.get(context_name, {}).get("base") or [])
        while pending:
            base = pending.pop()
            if base not in bases:
                bases.add(base)
                pending.extend(self._command_contexts.get(base, {}).get("base") or [])
        return bases

    def ordered_base_contexts(self, context_name: str) -> list[str]:
        """Every context whose commands *context_name* inherits, transitively,
        depth-first in declared ``base`` order, each once.

        Matches the order :meth:`commands` resolves inherited commands in.
        Unknown contexts (including the global ``*``) inherit nothing and give
        ``[]``. Cycles are rejected when the model loads, so the walk
        terminates.
        """
        return self._ordered_bases_from_map(context_name, self._command_contexts)

    # ---------------------------------------------------------------------
    # Context callback class resolution
    # ---------------------------------------------------------------------

    def get_context_class(self, context_name: str, module_type: fastworkflow.ModuleType):
        """Retrieve the callback class implementation for a given context.

        Currently only supports ModuleType.CONTEXT_CLASS. Returns None if not found.
        """

        if module_type != fastworkflow.ModuleType.CONTEXT_CLASS:
            raise ValueError(
                "CommandContextModel.get_context_class only supports ModuleType.CONTEXT_CLASS"
            )

        # Load command directory for metadata
        cmd_dir = get_cached_command_directory(self._workflow_path)

        try:
            context_metadata = cmd_dir.map_context_2_metadata[context_name]
        except KeyError:
            return None

        if module := python_utils.get_module(
            context_metadata.context_module_path,
            context_metadata.workflow_folderpath or self._workflow_path,
        ):
            context_class = getattr(module, context_metadata.context_class, None)
            if context_class is not None:
                problems = context_class_problems(context_class)
                if problems:
                    raise CommandContextModelError(
                        f"{context_metadata.context_module_path}: context {context_name!r} does not load: "
                        + "; ".join(problems))
            return context_class
        else:
            return None


def _warn_if_hierarchy_file_ignored(workflow_path: Path) -> None:
    """Warn once per workflow that a leftover ``context_hierarchy_model.json`` is ignored.

    The file used to declare each context's ``parent``. Parents now come only from the live
    chain, so the file has no effect on routing or training and can be deleted.
    """
    hierarchy_path = workflow_path / "context_hierarchy_model.json"
    if not hierarchy_path.is_file():
        return
    key = os.path.realpath(str(workflow_path))
    if key in _HIERARCHY_FILE_WARNED:
        return
    _HIERARCHY_FILE_WARNED.add(key)
    logger.warning(
        f"Workflow '{workflow_path.name}' has a context_hierarchy_model.json, which is "
        f"ignored: parents come from the live context chain and training no longer reads "
        f"the file. It can be deleted ({hierarchy_path})."
    )


# ---------------------------------------------------------------------
# Workflow info helper (module-level)
# ---------------------------------------------------------------------

def get_workflow_info(chat_session_or_path: fastworkflow.ChatSession | str) -> dict[str, Any]:
    """
    Return workflow-level metadata for the active workflow in the given chat session or workflow path.

    Shape:
    {
      "workflow_name": str,
      "available_contexts": list[str],
      "context_inheritance_model": dict[str, list[str]]   # context -> its declared bases
    }

    Notes:
    - available_contexts are derived from CommandContextModel.load(workflow_path)
      so they reflect filesystem + inheritance.
    - current context and purpose/description are intentionally omitted.
    - The declared bases of each context (``CommandContextModel.declared_bases``, the
      ``inherits`` of its class Context) are returned. The parent hierarchy is not described here:
      it lives in the live chain, and context_hierarchy_model.json is ignored.
    """
    if isinstance(chat_session_or_path, str):
        workflow_path_str = chat_session_or_path
    else:
        workflow_path_str = chat_session_or_path.get_active_workflow().folderpath
    workflow_name = os.path.basename(os.path.abspath(workflow_path_str))

    available_contexts: list[str]
    inheritance: dict[str, list[str]] = {}

    try:
        # Derive contexts via consolidated model
        context_model = CommandContextModel.load(workflow_path_str)
        # Do not normalize names beyond keeping what the model exposes
        available_contexts = sorted([key for key in context_model._command_contexts.keys() if key != 'IntentDetection'])
    except Exception:
        available_contexts = ["/"]

    # The declared bases of each context, through the one reader of the declarations
    try:
        inheritance = CommandContextModel.declared_bases(workflow_path_str)
    except Exception:
        inheritance = {}

    return {
        "workflow_name": workflow_name,
        "available_contexts": available_contexts,
        "context_inheritance_model": inheritance,
    }
