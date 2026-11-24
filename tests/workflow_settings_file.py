"""Write a workflow's settings file, ``_commands/_workflow.py`` (bd fix-qjne.26), for tests.

The settings are the class attributes of its ``class Context``. A test states them as it would write them at module
level; ``workflow_file_text`` keeps the import lines at the top and puts every other line into the class.
"""
from __future__ import annotations

import textwrap
from pathlib import Path


def workflow_file_text(settings: str) -> str:
    """The ``_workflow.py`` text declaring *settings* (module-level lines) as the attributes of ``class Context``."""
    lines = settings.splitlines()
    imports = [line for line in lines if line.startswith(("import ", "from "))]
    body = [line for line in lines if not line.startswith(("import ", "from "))]
    head = "\n".join(imports) + "\n\n\n" if imports else ""
    return head + "class Context:\n" + (textwrap.indent("\n".join(body), "    ").rstrip() or "    pass") + "\n"


def write_workflow_file(commands_dir: Path | str, settings: str) -> Path:
    """Write ``<commands_dir>/_workflow.py`` declaring *settings*; returns its path."""
    path = Path(commands_dir) / "_workflow.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(workflow_file_text(settings), encoding="utf-8")
    return path
