"""The package's layers (bd fix-bl97, gate G5): a module-level import never reaches a higher layer.

Each module's area is read from the reorganisation's mapping table (``scripts/reorg/moves.py``): a module that
moves belongs to the area it moves to, so the order holds before, during and after the moves. An import inside a
function is how the code breaks a cycle today and is not checked. The known exceptions are listed with the
extraction that removes each; an exception that no longer occurs must leave the list.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "fastworkflow"


def _load_table():
    spec = importlib.util.spec_from_file_location("reorg_moves", REPO / "scripts" / "reorg" / "moves.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses read the defining module
    spec.loader.exec_module(module)
    return module


TABLE = _load_table()
MAPPING = TABLE.dotted_map(TABLE.MOVES)

# Lowest first. Equal ranks may import each other.
RANKS = {
    "utils": 0, "observability_leaf": 0,
    "session": 1,
    "core": 2,
    "observability": 3, "observation_offloading": 3, "intent": 3, "parameters": 3, "policy": 3,
    "planning": 4,
    "review": 5,
    "engine": 6,
    "agent": 7,
    "evaluation": 8,
    "train": 9, "build": 9, "refine": 9, "run_cli": 9, "run_chatbot": 9, "run_fastapi_mcp": 9, "experiment": 9,
    "benchmark": 9, "cli": 9,
}
# Modules that import nothing from fastworkflow, so every layer may use them.
LEAVES = {"fastworkflow.observability.tracing", "fastworkflow.observability.metrics",
          "fastworkflow.observability.enrichment"}
NOT_CHECKED = ("examples", "_workflows", "skills_for_coding_fastworkflows")

# (importer, imported) as old dotted paths; each is removed by the extraction named beside it.
KNOWN_EXCEPTIONS = {
    ("fastworkflow.workflow_execution_context", "fastworkflow.utils.react"):
        "AskUserSuspend and NoSuspendedAgentStateError move out of the ReAct module (step 7)",
    ("fastworkflow.parameters.signatures", "fastworkflow.model_pipeline_training"):
        "CommandRouter and its path helpers move to intent/ (step 7)",
    ("fastworkflow.parameters.signatures", "fastworkflow.planning_commands"):
        "the planning-argument gap code moves into parameters/ (later)",
    ("fastworkflow.observation_offloading.facts", "fastworkflow.plan_frame"):
        "runs_outside_ranges moves into observation_offloading (step 7)",
    ("fastworkflow.observation_offloading.agent", "fastworkflow.utils.react"):
        "the offloading wiring of the ReAct loop moves to agent/ (later)",
    ("fastworkflow.utils.dspy_utils", "fastworkflow.parameters.rules"):
        "is_union, a typing helper, moves to utils (step 7)",
}


def canonical(dotted: str) -> str:
    """Where *dotted* lives once every move is made."""
    return TABLE.remap(dotted, MAPPING) or dotted


def area(dotted: str) -> str | None:
    final = canonical(dotted)
    if final in LEAVES:
        return "observability_leaf"
    parts = final.split(".")
    if len(parts) < 2 or parts[0] != "fastworkflow":
        return None
    # ``fastworkflow.<x>`` is a package's own __init__ (or cli.py); a name with no rank is no area
    return parts[1] if len(parts) > 2 or parts[1] in RANKS else None


def _is_module(dotted: str) -> bool:
    path = REPO.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported(node, module: str, is_init: bool) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        package = module.split(".") if is_init else module.split(".")[:-1]
        package = package[: len(package) - (node.level - 1)] if node.level > 1 else package
        base = ".".join(package + ([base] if base else []))
    names = []
    for alias in node.names:
        full = f"{base}.{alias.name}"
        names.append(full if _is_module(full) else base)
    return names


def _module_level_imports(tree: ast.Module):
    """Imports that run when the module loads: not inside a function, not under ``if TYPE_CHECKING``."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            stack.extend(node.orelse)
        else:
            stack.extend(ast.iter_child_nodes(node))


def upward_imports() -> dict[tuple[str, str], str]:
    found = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(PACKAGE_DIR).parts
        if rel[0].startswith(NOT_CHECKED) or "__pycache__" in rel or rel == ("__init__.py",):
            continue
        module = _module_name(path)
        source_area = area(module)
        if source_area is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _module_level_imports(tree):
            for target in _imported(node, module, path.name == "__init__.py"):
                target_area = area(target)
                if target_area is None or target_area == source_area or canonical(target) in LEAVES:
                    continue
                if RANKS[target_area] > RANKS[source_area]:
                    found[(canonical(module), canonical(target))] = f"{source_area} -> {target_area}"
    return found


def test_every_area_has_a_rank():
    areas = {area(f"fastworkflow.{m.new}") for m in TABLE.MOVES if m.new}
    for child in PACKAGE_DIR.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file() and not child.name.startswith(NOT_CHECKED):
            areas.add(area(f"fastworkflow.{child.name}.x"))
    assert areas - set(RANKS) == set()


def test_module_level_imports_follow_the_layer_order():
    allowed = {(canonical(a), canonical(b)) for a, b in KNOWN_EXCEPTIONS}
    new = {edge: why for edge, why in upward_imports().items() if edge not in allowed}
    assert new == {}, "a module-level import reaches a higher layer: " + "; ".join(
        f"{a} imports {b} ({why})" for (a, b), why in sorted(new.items()))


def test_the_known_exceptions_still_occur():
    found = set(upward_imports())
    stale = [f"{a} -> {b}" for a, b in KNOWN_EXCEPTIONS if (canonical(a), canonical(b)) not in found]
    assert stale == [], "these exceptions no longer occur, remove them from KNOWN_EXCEPTIONS: " + "; ".join(stale)
