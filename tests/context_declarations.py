"""Write a context's ``inherits`` declaration into a temp workflow (bd fix-qjne.37.4).

Tests that build a workflow of their own used to write ``_commands/context_inheritance_model.json``; the bases are now
declared on the ``class Context`` of ``_commands/<Context>/_<Context>.py``. ``declare_inherits`` writes them there.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_INHERITS_LINE = re.compile(r"^[ \t]+inherits = \[.*\][ \t]*\n", re.M)


def _line(bases: list[str]) -> str:
    return "    inherits = [" + ", ".join(json.dumps(base) for base in bases) + "]\n"


def declare_inherits(workflow_folder: str | Path, bases_by_context: dict[str, list[str]]) -> None:
    """Make each context's ``class Context`` declare exactly ``inherits = bases`` (none for an empty list), creating
    ``_<Context>.py`` (and the context folder) when there is none."""
    for context, bases in bases_by_context.items():
        path = Path(workflow_folder) / "_commands" / context / f"_{context}.py"
        bases = list(bases)
        if not path.is_file():
            if bases:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("class Context:\n" + _line(bases), encoding="utf-8")
            continue
        text = _INHERITS_LINE.sub("", path.read_text(encoding="utf-8"))
        if bases:
            if "\nclass Context:\n" in f"\n{text}":
                text = re.sub(r"^class Context:\n", lambda m: m.group(0) + _line(bases), text, count=1, flags=re.M)
            else:
                text = text + ("" if text.endswith("\n") or not text else "\n") + "\n\nclass Context:\n" + _line(bases)
        path.write_text(text, encoding="utf-8")
