"""``fastworkflow build``: each context's ``inherits``, derived from the application's class hierarchy.

A context's bases are declared as ``inherits = [...]`` on the ``class Context`` of ``_commands/<Context>/_<Context>.py``
(``command_context_model.read_declarations``); ``context_inheritance_model.json`` is a removed spelling and build never
writes it. Build never edits a ``_<Context>.py`` that existed before it ran: it prints the line it derives, the file the
line belongs in, and whether the file declares it, declares something else or declares nothing. A ``_<Context>.py`` the
build itself creates (none existed: a new workflow, or a new class) gets the derived line, written by the migration
helper's writer (``migrate_context_declarations.write_inherits``). ``parents`` cannot be derived from the classes:
every context whose ``get_parent`` has no ``parents`` beside it is listed for the author.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.core.command_context_model import REMOVED_INHERITANCE_FILE, _declarations_in_file
from fastworkflow.build.migrate_context_declarations import inherits_line, needs_parents, write_inherits


def derive_inherits(classes: Dict[str, ClassInfo]) -> dict[str, list[str]]:
    """``{class: [base, ...]}`` for every analysed class: its direct bases that are analysed classes too, in order."""
    names = set(classes)
    return {name: [base for base in info.bases if base in names] for name, info in sorted(classes.items())}


def context_file(workflow_folder: str | Path, context: str) -> Path:
    """``<workflow>/_commands/<Context>/_<Context>.py``: the file a context's declarations belong in."""
    return Path(workflow_folder) / "_commands" / context / f"_{context}.py"


def existing_context_files(workflow_folder: str | Path) -> set[Path]:
    """The ``_<Context>.py`` files the workflow has now: the hand-written ones build never edits."""
    commands = Path(workflow_folder) / "_commands"
    if not commands.is_dir():
        return set()
    return {folder / f"_{folder.name}.py" for folder in commands.iterdir()
            if folder.is_dir() and (folder / f"_{folder.name}.py").is_file()}


@dataclass
class InheritsReport:
    """What build derived for each context and what it did with it."""

    workflow_folder: Path
    derived: dict[str, list[str]] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)                     # files build created with the line
    declared: dict[str, list[str] | None] = field(default_factory=dict)   # hand-written files: what they declare
    needs_parents: dict[str, str] = field(default_factory=dict)          # context -> get_parent return annotation
    removed_file: bool = False

    def differs(self) -> dict[str, tuple[list[str], list[str] | None]]:
        """``{context: (derived, declared)}`` for each hand-written file whose ``inherits`` differs from the derived
        one (None: it declares none)."""
        return {name: (self.derived.get(name, []), declared) for name, declared in sorted(self.declared.items())
                if (declared or []) != self.derived.get(name, [])}

    def lines(self) -> list[str]:
        out = []
        for name, bases in sorted(self.derived.items()):
            where = context_file(self.workflow_folder, name).relative_to(self.workflow_folder)
            line = inherits_line(bases) if bases else "no inherits"
            if name in self.written:
                out.append(f"{name}: {line}  ({where}: written, a new file)")
            elif name in self.declared:
                declared = self.declared[name]
                if (declared or []) == bases:
                    out.append(f"{name}: {line}  ({where}: declared)")
                else:
                    told = inherits_line(declared) if declared else "no inherits"
                    out.append(f"{name}: {line}  ({where} declares {told}: differs, not changed)")
            elif bases:
                out.append(f"{name}: {line}  (belongs in class Context of {where}, which does not exist)")
            else:
                out.append(f"{name}: {line}")
        out += [f"{name}: has get_parent, write parents = [...] in its class Context by hand "
                f"(get_parent returns {hint})" for name, hint in sorted(self.needs_parents.items())]
        if self.removed_file:
            out.append(f"_commands/{REMOVED_INHERITANCE_FILE} is removed and not read: run "
                       f"`fastworkflow migrate-contexts {self.workflow_folder}`")
        return out


def declare_inherits(workflow_folder: str | Path, derived: dict[str, list[str]],
                     hand_written: Iterable[Path] | None = None) -> InheritsReport:
    """Write the derived ``inherits`` of each context into a ``_<Context>.py`` that is not hand-written (one in
    *hand_written*, by default every one that exists now), creating it when the context folder has none; read what
    each hand-written one declares, never changing it. Contexts with a ``get_parent`` and no ``parents`` are listed."""
    folder = Path(workflow_folder)
    kept = {Path(path) for path in (existing_context_files(folder) if hand_written is None else hand_written)}
    report = InheritsReport(workflow_folder=folder, derived={name: list(b) for name, b in derived.items()},
                            removed_file=(folder / "_commands" / REMOVED_INHERITANCE_FILE).is_file())
    for name, bases in sorted(derived.items()):
        path = context_file(folder, name)
        if path in kept:
            report.declared[name] = list(_declarations_in_file(path).inherits or []) or None
        elif bases and path.parent.is_dir():
            write_inherits(path, bases)
            report.written.append(name)
    commands = folder / "_commands"
    for context_folder in sorted(commands.iterdir()) if commands.is_dir() else ():
        if context_folder.is_dir() and not context_folder.name.startswith(("_", ".")):
            if (hint := needs_parents(context_folder / f"_{context_folder.name}.py")) is not None:
                report.needs_parents[context_folder.name] = hint
    return report


def generate_context_model(classes: Dict[str, ClassInfo], workflow_folder: str,
                           hand_written: Iterable[Path] | None = None) -> dict[str, dict[str, list[str]]]:
    """Derive each class's ``inherits``, declare it in every ``_<Context>.py`` that is not hand-written
    (``declare_inherits``), print one line per context, and return the model the other build generators read:
    ``{context: {"base": [...]}}`` for every analysed class. Nothing else is written."""
    derived = derive_inherits(classes)
    report = declare_inherits(workflow_folder, derived, hand_written)
    for line in report.lines():
        print(line)
    return {name: {"base": bases} for name, bases in derived.items()}
