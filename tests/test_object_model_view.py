"""The generated object-model view (bd fix-qjne.37.2): drawn from each context's ``inherits`` / ``parents`` and each
command's ``@descends_to``, written by ``fastworkflow command preview`` beside the planning command previews."""
import os
import re

from fastworkflow.planning import command_preview
from fastworkflow.core import object_model_view
from fastworkflow.core.command_context_model import CommandContextModel, context_folders

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
TODO = os.path.join(ROOT, "tests", "todo_list_workflow")
TUTORIAL = os.path.join(ROOT, "tests", "planning_tutorial_workflow")
TUTORIAL_PREVIEWS = os.path.join(ROOT, "docs", "tutorials", "planning_commands", "previews")


def _rows(text: str) -> dict[str, list[str]]:
    """``{context: [inherits, parents, entered by]}`` from the rendered table."""
    rows = {}
    for line in text.splitlines():
        if match := re.match(r"^\| `(\w+)` \| (.*) \| (.*) \| (.*) \|$", line):
            assert match.group(1) not in rows, f"{match.group(1)} is listed twice"
            rows[match.group(1)] = [match.group(2), match.group(3), match.group(4)]
    return rows


def _names(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def _check_against_declarations(folder: str) -> str:
    text = object_model_view.render_object_model(folder)
    rows = _rows(text)
    assert sorted(rows) == sorted(context_folders(folder))           # every context once
    bases, parents = CommandContextModel.declared_bases(folder), CommandContextModel.declared_parents(folder)
    for name, (inherits, contained_by, _) in rows.items():
        assert _names(inherits) == bases.get(name, []), name
        assert _names(contained_by) == parents.get(name, []), name
    assert text.count("No file declares the root") == 1
    assert object_model_view.render_object_model(folder) == text        # deterministic
    return text


def test_the_library_model_matches_its_declarations():
    text = _check_against_declarations(LIBRARY)
    rows = _rows(text)
    assert rows["Book"] == ["`Record`", "`Library`", "`Library/open_book`"]
    assert rows["Member"] == ["`Record`", "`Library`", "`Library/open_member`"]
    assert rows["Library"] == ["-", "top level", "-"] and rows["Record"] == ["-", "never entered", "-"]
    model = object_model_view.object_model(LIBRARY)
    assert model.top_level == ("Library",) and model.never_entered == ("Record",)
    assert "- global commands\n  - Library\n    - Book (entered by `Library/open_book`)\n" in text


def test_a_record_nested_in_its_own_kind_is_drawn_once():
    text = _check_against_declarations(TODO)
    assert text.count("- TodoList (entered by") == 1 and "      - TodoList (see above)" in text
    assert object_model_view.object_model(TODO).top_level == ("TodoListManager",)


def test_the_committed_object_models_are_current():
    assert object_model_view.check_object_model(LIBRARY, command_preview.default_out_dir(LIBRARY)) == []
    assert object_model_view.check_object_model(TUTORIAL, TUTORIAL_PREVIEWS) == []
    assert "No contexts: every command is a global command." in object_model_view.render_object_model(TUTORIAL)


def test_check_reports_missing_and_stale_and_ignores_other_workflows_models(tmp_path):
    out = str(tmp_path)
    assert object_model_view.check_object_model(LIBRARY, out) == ["missing: _object_models/library.md"]
    assert object_model_view.write_object_model(LIBRARY, out) is True
    assert object_model_view.write_object_model(LIBRARY, out) is False
    assert object_model_view.check_object_model(LIBRARY, out) == []
    path = tmp_path / "_object_models" / "library.md"
    path.write_text(path.read_text().replace("Library/open_book", "Library/open_tome"))
    assert object_model_view.check_object_model(LIBRARY, out) == ["stale: _object_models/library.md"]
    object_model_view.write_object_model(LIBRARY, out)
    (tmp_path / "_object_models" / "other_workflow.md").write_text("another workflow's model\n")
    assert object_model_view.check_object_model(LIBRARY, out) == []


def test_previews_write_and_check_the_object_model_beside_them(tmp_path):
    out = str(tmp_path / "previews")
    command_preview.write_previews(LIBRARY, out)
    assert (tmp_path / "previews" / "_object_models" / "library.md").is_file()
    (tmp_path / "previews" / "_object_models" / "library.md").unlink()
    assert "missing: _object_models/library.md" in command_preview.check_previews(LIBRARY, out)
