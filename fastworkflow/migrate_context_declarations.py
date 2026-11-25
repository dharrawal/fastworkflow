"""Move a workflow's ``_commands/context_inheritance_model.json`` into ``inherits`` declarations.

``fastworkflow migrate-contexts <workflow>`` writes, for each context the file gives a base, ``inherits = [...]`` into
the ``class Context`` of ``_commands/<Context>/_<Context>.py`` (creating the file, or the class, when there is none),
then deletes the file. It is mechanical: the resolved command set of every context is the same before and after.

``parents`` cannot be derived: a ``get_parent`` is code whose return value is known only at run time (a nested list
returns a list or the manager; a return annotation is a hint, not the truth). The helper lists every context that has
a ``get_parent`` with the return annotation as a hint, and the author writes its ``parents`` by hand.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from fastworkflow.command_context_model import REMOVED_INHERITANCE_FILE, context_folders


@dataclass
class MigrationReport:
    written: dict[str, list[str]] = field(default_factory=dict)        # context -> inherits written
    created: list[str] = field(default_factory=list)                   # context class files created
    needs_parents: dict[str, str] = field(default_factory=dict)        # context -> get_parent return annotation
    deleted: str | None = None

    def lines(self) -> list[str]:
        out = [f"{name}: inherits = {bases!r}{' (new file)' if name in self.created else ''}"
               for name, bases in sorted(self.written.items())]
        out += [f"{name}: has get_parent, write parents = [...] by hand (get_parent returns {hint})"
                for name, hint in sorted(self.needs_parents.items())]
        if self.deleted:
            out.append(f"deleted {self.deleted}")
        return out


def inherits_line(bases: list[str]) -> str:
    return "inherits = [" + ", ".join(json.dumps(base) for base in bases) + "]"


def write_inherits(path: Path, bases: list[str]) -> bool:
    """Write ``inherits`` into the ``class Context`` of *path*; True when the file was created."""
    if not path.is_file():
        path.write_text(f"class Context:\n    {inherits_line(bases)}\n", encoding="utf-8")
        return True
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    context = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Context"), None)
    if context is None:
        suffix = "" if text.endswith("\n") or not text else "\n"
        path.write_text(f"{text}{suffix}\n\nclass Context:\n    {inherits_line(bases)}\n", encoding="utf-8")
        return False
    first = context.body[0]
    indent = " " * first.col_offset
    is_docstring = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str))
    if isinstance(first, ast.Pass):
        lines[first.lineno - 1] = f"{indent}{inherits_line(bases)}\n"
    else:
        start = min([first.lineno, *(decorator.lineno for decorator in getattr(first, "decorator_list", ()))])
        at = first.end_lineno if is_docstring else start - 1
        block = f"{indent}{inherits_line(bases)}\n" + ("\n" if not is_docstring else "")
        if is_docstring:
            block = "\n" + block
        lines.insert(at, block)
    path.write_text("".join(lines), encoding="utf-8")
    return False


def needs_parents(path: Path) -> str | None:
    """When the ``class Context`` of *path* has a ``get_parent`` and declares no ``parents``: the return annotation
    of that ``get_parent`` as a hint (``"not annotated"`` without one); otherwise None."""
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Context":
            names = {target.id for statement in node.body
                     for target in (statement.targets if isinstance(statement, ast.Assign)
                                    else [statement.target] if isinstance(statement, ast.AnnAssign) else [])
                     if isinstance(target, ast.Name)}
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "get_parent"
                        and "parents" not in names):
                    return ast.unparse(item.returns) if item.returns is not None else "not annotated"
    return None


def migrate(workflow_folder: str | Path) -> MigrationReport:
    """Write the ``inherits`` of each context from the workflow's ``context_inheritance_model.json`` and delete it.
    Contexts with a ``get_parent`` and no ``parents`` are listed for the author."""
    folder = Path(workflow_folder)
    path = folder / "_commands" / REMOVED_INHERITANCE_FILE
    report = MigrationReport()
    folders = context_folders(folder)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
        if "*" in raw:
            raise ValueError(f"{path}: the '*' key (the global commands inheriting a context) has no declaration; "
                             f"remove it by hand first")
        unknown = sorted(set(raw) - set(folders))
        if unknown:
            raise ValueError(f"{path}: {', '.join(unknown)} is no context folder of the workflow; fix the file first")
        for name, entry in sorted(raw.items()):
            bases = list((entry or {}).get("base") or [])
            if not bases:
                continue
            if write_inherits(folders[name] / f"_{name}.py", bases):
                report.created.append(name)
            report.written[name] = bases
        path.unlink()
        report.deleted = str(path)
    for name, context_folder in sorted(folders.items()):
        if (hint := needs_parents(context_folder / f"_{name}.py")) is not None:
            report.needs_parents[name] = hint
    return report


def migrate_main(args) -> int:
    report = migrate(args.workflow_folder)
    for line in report.lines() or ["nothing to migrate"]:
        print(line)
    return 0
