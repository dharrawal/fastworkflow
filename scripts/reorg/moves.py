#!/usr/bin/env python3
"""The package reorganisation (bd fix-bl97): the mapping table and the tool that applies it.

Every module that moves is one row of ``MOVES``: its old dotted path under ``fastworkflow.``, its new
dotted path, and the step that moves it. The tool never decides anything the table does not say.

    python scripts/reorg/moves.py --step 2 --plan                    # what step 2 moves and rewrites
    python scripts/reorg/moves.py --step 2 --print-map               # old -> new dotted paths, plain text
    python scripts/reorg/moves.py --step 2 --apply --root . --root ../app   # rewrite references only
    python scripts/reorg/moves.py --step 2 --apply --move --root .   # rewrite, then `git mv`

``--apply`` rewrites, in every ``--root``: Python files with libcst (absolute and relative imports, ``import
fastworkflow.x as y``, attribute chains ``fastworkflow.x.y``, string literals and comments that contain an
old dotted path or an old ``fastworkflow/x.py`` path), and text files (markdown, rules, toml, yaml, shell)
by the same two text rules. It runs before the move, so a relative import is resolved against the file's
present location; a moved file's relative imports become absolute. Running it again changes nothing.
``--move`` then moves the files with ``git mv`` in the repo that holds the package (``--repo``, default the
repo of this script) and creates the empty ``__init__.py`` of each new package, which it lists for ``git
add`` (it adds nothing itself). CHANGELOG.md is history and is left as written unless ``--changelog``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "fastworkflow"
REPO = Path(__file__).resolve().parents[2]

# kind: "module" (a .py file), "package" (a folder, moved whole), "to_package" (a module that becomes its
# own package's __init__.py: the dotted path does not change), "delete" (removed, no new path).


@dataclass(frozen=True)
class Move:
    step: int
    old: str
    new: str | None
    kind: str = "module"


def _rows(step: int, new_parent: str, olds: str, kind: str = "module") -> list[Move]:
    return [Move(step, old, f"{new_parent}.{old.rsplit('.', 1)[-1]}", kind) for old in olds.split()]


MOVES: list[Move] = [
    # Step 2 (fix-bl97.3): leaf areas.
    *_rows(2, "observability", "tracing metrics"),
    *_rows(2, "session", "session_state_store checkpoint_store state_serialization serialization_hooks state_paths "
                         "storage_keys kvstore conversation_history_io conversation_summary conversation_labeling"),
    *_rows(2, "policy", "policy_decision policy_intent policy_loader policy_runtime"),
    *_rows(2, "evaluation", "distillation distillation_compare distillation_trace policy_replay utils.insights_loader"),
    *_rows(2, "utils", "subtask_pool"),
    Move(2, "utils.parameterize_func_decorator", None, "delete"),
    # Step 3 (fix-bl97.4): the parameter area.
    *_rows(3, "parameters", "parameter_clarification field_options utils.signatures utils.argument_checks "
                            "utils.input_validation utils.extraction_attempts utils.workflow_vocabulary"),
    *_rows(3, "train", "utils.generate_param_examples"),
    # Step 4 (fix-bl97.5): planning and review.
    Move(4, "planning", "planning", "to_package"),
    *_rows(4, "planning", "planning_commands planning_phases plan_frame typed_phase_facts planning_code_run "
                          "planning_needs_agent planning_command_json_schema_internal command_preview"),
    *_rows(4, "review", "review_engine review_proposals review_plan_switch review_step_refs approval_binding "
                        "approval_grants"),
    Move(4, "_framework_commands", "review.commands", "package"),
    # Step 5 (fix-bl97.6): the hub.
    *_rows(5, "core", "workflow workflow_inheritance_model command_context_model command_directory command_routing "
                      "command_metadata_api command_interfaces context_identity context_navigation "
                      "entry_declarations command_declarations command_writes approval_annotations "
                      "object_model_view turn turn_cache turn_progress waiting no_agent_mode utils.context_utils"),
    *_rows(5, "engine", "workflow_execution_context command_executor chat_session prompt_payloads turn_gate "
                        "user_message_queues active_workflow"),
    *_rows(5, "intent", "intent_loop intent_detector intent_clarification intent_learning nlu_labels remote_outage"),
    # Step 6 (fix-bl97.7): the agent and the rest.
    *_rows(6, "agent", "workflow_agent split_merge run_listing capability_answer utils.react utils.chat_adapter"),
    *_rows(6, "observation_offloading", "context_budget answer_rehydration"),
    *_rows(6, "train", "model_pipeline_training"),
    *_rows(6, "run_fastapi_mcp", "mcp_server"),
    *_rows(6, "build", "migrate_context_declarations"),
    # Steps 7 and 8 move no module (extractions, docs). Step 9 (optional, fix-bl97.10).
    Move(9, "experiment", "evaluation.experiment", "package"),
    Move(9, "benchmark", "evaluation.benchmark", "package"),
    # observation_offloading stays where it is: planning modules import it at module level, so under agent/
    # it would sit above planning (tests/test_import_layers.py).
]

def selected(steps: list[int] | None) -> list[Move]:
    return [m for m in MOVES if steps is None or m.step in steps]


def dotted_map(moves: list[Move]) -> dict[str, str]:
    """Full old dotted path -> full new dotted path, for the rows whose path changes."""
    return {f"{PACKAGE}.{m.old}": f"{PACKAGE}.{m.new}" for m in moves
            if m.new is not None and m.new != m.old}


def deleted(moves: list[Move]) -> list[str]:
    return [f"{PACKAGE}.{m.old}" for m in moves if m.kind == "delete"]


def _rel_path(dotted: str, kind: str) -> str:
    """``fastworkflow/a/b.py`` (a module) or ``fastworkflow/a/b`` (a package) of a dotted path under it."""
    parts = dotted.split(".")
    return "/".join(parts) + ("" if kind in ("package",) else ".py")


def source_path(m: Move) -> str:
    return _rel_path(f"{PACKAGE}.{m.old}", m.kind)


def target_path(m: Move) -> str | None:
    if m.new is None:
        return None
    if m.kind == "to_package":
        return "/".join(f"{PACKAGE}.{m.new}".split(".")) + "/__init__.py"
    return _rel_path(f"{PACKAGE}.{m.new}", m.kind)


def remap(dotted: str, mapping: dict[str, str]) -> str | None:
    """The new dotted path of *dotted* (a module or anything under it), or None when it does not move."""
    for old in sorted(mapping, key=len, reverse=True):
        if dotted == old or dotted.startswith(old + "."):
            return mapping[old] + dotted[len(old):]
    return None


# ---------------------------------------------------------------- the two text rules


def text_rules(moves: list[Move]) -> list[tuple[re.Pattern, str]]:
    """Dotted paths and file paths, longest old path first, each bounded so ``fastworkflow.turn`` does not
    match ``fastworkflow.turn_cache`` and an already rewritten path is never matched again."""
    rules: list[tuple[re.Pattern, str]] = []
    rows = [m for m in moves if m.new is not None and m.new != m.old]
    for m in sorted(rows, key=lambda r: len(r.old), reverse=True):
        old_dotted, new_dotted = f"{PACKAGE}.{m.old}", f"{PACKAGE}.{m.new}"
        rules.append((re.compile(r"(?<![\w.])" + re.escape(old_dotted) + r"(?!\w)"), new_dotted))
        old_path = "/".join(old_dotted.split("."))
        new_path = "/".join(new_dotted.split("."))
        if m.kind == "package":
            rules.append((re.compile(r"(?<![\w/.])" + re.escape(old_path) + r"(?![\w.])"), new_path))
        else:
            rules.append((re.compile(r"(?<![\w/.])" + re.escape(old_path) + r"\.py(?!\w)"), new_path + ".py"))
    return rules


def rewrite_text(text: str, rules) -> str:
    for pattern, new in rules:
        text = pattern.sub(new, text)
    return text


# ---------------------------------------------------------------- Python files


def _module_of(path: Path, root: Path) -> tuple[str | None, bool]:
    """The dotted module name of a file inside the ``fastworkflow`` package under *root*, or None."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None, False
    parts = list(rel.with_suffix("").parts)
    if not parts or parts[0] != PACKAGE:
        return None, False
    is_init = parts[-1] == "__init__"
    if is_init:
        parts = parts[:-1]
    return ".".join(parts), is_init


def rewrite_python(source: str, module: str | None, is_init: bool, mapping: dict[str, str], rules,
                   moving: set[str]) -> str:
    import libcst as cst
    from libcst.helpers import get_full_name_for_node

    def dotted_node(name: str):
        return cst.parse_expression(name)

    def resolve(node) -> str | None:
        base = get_full_name_for_node(node.module) if node.module is not None else ""
        level = len(node.relative)
        if not level:
            return base
        if module is None:
            return None
        pkg = module.split(".") if is_init else module.split(".")[:-1]
        if level > 1:
            pkg = pkg[: len(pkg) - (level - 1)]
        return ".".join(pkg + ([base] if base else []))

    file_moves = module is not None and (module in moving)

    class Rewriter(cst.CSTTransformer):
        def visit_Import(self, node):
            return False

        def visit_ImportFrom(self, node):
            return False

        def leave_Attribute(self, original, updated):
            name = get_full_name_for_node(updated)
            if name and name.startswith(PACKAGE + "."):
                new = remap(name, mapping)
                if new is not None:
                    return dotted_node(new)
            return updated

        def leave_SimpleString(self, original, updated):
            new = rewrite_text(updated.value, rules)
            return updated if new == updated.value else updated.with_changes(value=new)

        def leave_FormattedStringText(self, original, updated):
            new = rewrite_text(updated.value, rules)
            return updated if new == updated.value else updated.with_changes(value=new)

        def leave_Comment(self, original, updated):
            new = rewrite_text(updated.value, rules)
            return updated if new == updated.value else updated.with_changes(value=new)

        def leave_Assign(self, original, updated):
            # The package's ``__all__`` names its submodules, which ``__getattr__`` imports by name: a moved
            # top-level module leaves it (its new path is reached through its new package).
            if module != PACKAGE or not is_init or not isinstance(updated.value, (cst.Tuple, cst.List)):
                return updated
            if [get_full_name_for_node(t.target) for t in updated.targets] != ["__all__"]:
                return updated
            gone = {old.split(".", 1)[1] for old in moving if old.count(".") == 1}
            kept = [e for e in updated.value.elements
                    if not (isinstance(e.value, cst.SimpleString) and e.value.evaluated_value in gone)]
            if len(kept) == len(updated.value.elements):
                return updated
            return updated.with_changes(value=updated.value.with_changes(elements=kept))

        def leave_SimpleStatementLine(self, original, updated):
            if len(updated.body) != 1:
                return updated
            stmt = updated.body[0]
            if isinstance(stmt, cst.Import):
                names = []
                changed = False
                for alias in stmt.names:
                    full = get_full_name_for_node(alias.name)
                    new = remap(full, mapping) if full else None
                    if new is not None:
                        alias = alias.with_changes(name=dotted_node(new))
                        changed = True
                    names.append(alias)
                return updated.with_changes(body=[stmt.with_changes(names=names)]) if changed else updated
            if not isinstance(stmt, cst.ImportFrom):
                return updated
            base = resolve(stmt)
            if base is None or not (base == PACKAGE or base.startswith(PACKAGE + ".")):
                return updated
            relative = bool(stmt.relative)
            new_base = remap(base, mapping) or base
            if isinstance(stmt.names, cst.ImportStar):
                if new_base == base and not (relative and file_moves):
                    return updated
                return updated.with_changes(body=[stmt.with_changes(
                    module=dotted_node(new_base), relative=[])])
            # Group the names: a name that is itself a moving module goes under its new parent.
            groups: dict[str, list] = {}
            for alias in stmt.names:
                full = f"{base}.{get_full_name_for_node(alias.name)}"
                moved = full in mapping
                if moved:
                    parent, leaf = mapping[full].rsplit(".", 1)
                    alias = alias.with_changes(name=cst.Name(leaf))
                    groups.setdefault(parent, []).append(alias)
                else:
                    groups.setdefault(new_base, []).append(alias)
            if list(groups) == [base] and not (relative and file_moves):
                return updated
            if len(groups) == 1:
                # One group: only the module changes (a moved name keeps its leaf), formatting kept.
                (parent,) = groups
                return updated.with_changes(body=[stmt.with_changes(module=dotted_node(parent), relative=[])])
            lines = []
            for i, (parent, aliases) in enumerate(groups.items()):
                aliases = [a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in aliases]
                line = updated.with_changes(body=[cst.ImportFrom(module=dotted_node(parent), names=aliases)])
                if i:
                    line = line.with_changes(leading_lines=[])
                lines.append(line)
            return cst.FlattenSentinel(lines) if len(lines) > 1 else lines[0]

    tree = cst.parse_module(source)
    return tree.visit(Rewriter()).code


# ---------------------------------------------------------------- walking the roots

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".repowise", ".beads", "worktrees",
             "___command_info", "___workflow_contexts", "___convo_info", ".mypy_cache", ".pytest_cache"}
TEXT_SUFFIXES = {".md", ".mdc", ".txt", ".rst", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".sh", ".ipynb"}
TEXT_NAMES = {"Makefile", "Dockerfile"}


def files_under(root: Path, changelog: bool = False):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if name == "CHANGELOG.md" and not changelog:
                continue
            if path.suffix == ".py":
                yield path, "python"
            elif path.suffix in TEXT_SUFFIXES or name in TEXT_NAMES or name.startswith("Dockerfile"):
                yield path, "text"


def apply(moves: list[Move], roots: list[Path], changelog: bool, dry: bool = False) -> list[Path]:
    mapping = dotted_map(moves)
    rules = text_rules(moves)
    moving = {f"{PACKAGE}.{m.old}" for m in moves if m.kind in ("module", "to_package")}
    moving |= {f"{PACKAGE}.{m.old}" for m in moves if m.kind == "package"}
    changed: list[Path] = []
    # Cheap pre-filter: a file that names no moving module's leaf (and has no relative import) is untouched.
    quick = re.compile("|".join(sorted({re.escape(m.old.rsplit(".", 1)[-1]) for m in moves})) or "^$")
    for root in roots:
        for path, kind in files_under(root, changelog):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if PACKAGE not in text or not quick.search(text):
                if not (kind == "python" and "from ." in text):
                    continue
            if kind == "python":
                module, is_init = _module_of(path, root)
                # A file in a moving package (step 9) is moving too.
                file_moving = moving | ({module} if module and any(
                    module.startswith(p + ".") for p in moving) else set())
                try:
                    new = rewrite_python(text, module, is_init, mapping, rules, file_moving)
                except Exception as exc:  # a file libcst cannot parse is reported, never guessed at
                    print(f"  SKIPPED (does not parse): {path}: {exc.__class__.__name__}", file=sys.stderr)
                    continue
            else:
                new = rewrite_text(text, rules)
            if new != text:
                changed.append(path)
                if not dry:
                    path.write_text(new, encoding="utf-8")
    return changed


def move_files(moves: list[Move], repo: Path) -> list[Path]:
    """`git mv` (and `git rm` for a deleted module); returns the new __init__.py files to `git add`."""
    created: list[Path] = []

    def ensure_package(folder: Path) -> None:
        package_root = repo / PACKAGE
        while folder != package_root and package_root in folder.parents:
            init = folder / "__init__.py"
            if not folder.exists():
                folder.mkdir(parents=True)
            if not init.exists():
                init.write_text("", encoding="utf-8")
                created.append(init)
            folder = folder.parent

    for m in moves:
        src = repo / source_path(m)
        if m.kind == "delete":
            if src.exists():
                subprocess.run(["git", "-C", str(repo), "rm", "-q", str(src)], check=True)
            continue
        dst = repo / target_path(m)
        if not src.exists():
            if dst.exists():
                continue  # already moved
            raise SystemExit(f"neither {src} nor {dst} exists")
        if m.kind == "to_package":
            dst.parent.mkdir(parents=True, exist_ok=True)
            # git cannot move planning.py into planning/ while both names exist; go through a temp name.
            tmp = src.with_name(src.stem + "__moving.py")
            subprocess.run(["git", "-C", str(repo), "mv", str(src), str(tmp)], check=True)
            subprocess.run(["git", "-C", str(repo), "mv", str(tmp), str(dst)], check=True)
            continue
        ensure_package(dst.parent)
        subprocess.run(["git", "-C", str(repo), "mv", str(src), str(dst)], check=True)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, action="append", help="a step number (repeatable); none = every step")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--plan", action="store_true", help="print the moves and the files a rewrite would change")
    what.add_argument("--print-map", action="store_true", help="print old -> new dotted paths, one per line")
    what.add_argument("--apply", action="store_true", help="rewrite references in every --root")
    parser.add_argument("--move", action="store_true", help="with --apply: git mv the files afterwards")
    parser.add_argument("--root", action="append", type=Path, default=[], help="a tree to rewrite (repeatable)")
    parser.add_argument("--repo", type=Path, default=REPO, help="the repo holding the package (for --move)")
    parser.add_argument("--changelog", action="store_true", help="also rewrite CHANGELOG.md")
    args = parser.parse_args(argv)
    moves = selected(args.step)

    if args.print_map:
        for m in moves:
            new = f"{PACKAGE}.{m.new}" if m.new else "(deleted)"
            print(f"{PACKAGE}.{m.old} -> {new}")
        return 0

    roots = args.root or [args.repo]
    if args.plan:
        for m in moves:
            print(f"step {m.step}  {m.kind:10s} {source_path(m)} -> {target_path(m) or '(deleted)'}")
        # A moved file that finds files next to itself (``__file__``) needs a hand edit: list them.
        for m in moves:
            src = args.repo / source_path(m)
            for path in ([src] if src.is_file() else sorted(src.rglob("*.py")) if src.is_dir() else []):
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if re.search(r"(?<![\w.'\"])__file__(?!['\"])", line):
                        print(f"  HAND EDIT: {path}:{number}: {line.strip()}")
        changed = apply(moves, roots, args.changelog, dry=True)
        print(f"\n{len(changed)} file(s) would be rewritten:")
        for path in changed:
            print(f"  {path}")
        return 0

    if args.move and args.repo.resolve() not in [r.resolve() for r in roots]:
        print("--move: the repo holding the package must be one of the --root trees", file=sys.stderr)
        return 2
    changed = apply(moves, roots, args.changelog)
    print(f"rewrote {len(changed)} file(s)")
    if args.move:
        created = move_files(moves, args.repo)
        if created:
            print("new package files to add:")
            print("  git add " + " ".join(str(p.relative_to(args.repo)) for p in created))
    return 0


if __name__ == "__main__":
    sys.exit(main())
