"""EXP-028 skill-catalogue loading and conformance."""

from __future__ import annotations

import builtins
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastworkflow.runtime_manifest import RuntimeManifest, load_manifest, merge_and_gate
from fastworkflow.skill_catalog import (
    APPROVED_LEVELS,
    CORE_VERBS,
    SkillCatalogError,
    load_skill_catalog,
)

FIXTURE_WORKFLOW = Path(__file__).parent / "fixtures" / "skills_workflow"
IDO_WORKFLOW = Path("/home/drawal/rl/ido/ido_workflow")


def _manifest(
    *,
    mode: str | None = "enforce",
    commands: tuple[str, ...] = ("known",),
    skills_fingerprint: str | None = None,
):
    features = {} if mode is None else {"skills_v1": mode}
    return SimpleNamespace(
        features=features,
        commands={f"Context/{name}": SimpleNamespace() for name in commands},
        skills_fingerprint=skills_fingerprint,
    )


def _write_skill(
    workflow: Path,
    name: str,
    *,
    directory_name: str | None = None,
    level: str = "atomic",
    goal: str | None = None,
    slots: tuple[dict[str, object], ...] = (),
    uses: tuple[str, ...] = (),
    presents: tuple[str, ...] = (),
    body: str = "1. `known`\n",
) -> Path:
    directory = workflow / "_skills" / (directory_name or name)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: Exercise {name}.",
        f"level: {level}",
    ]
    if goal is not None:
        lines.append(f"goal: {goal}")
    for index, slot in enumerate(slots):
        if index == 0:
            lines.append("slots:")
        lines.extend(
            [
                f"  - name: {slot['name']}",
                f"    required: {str(slot.get('required', False)).lower()}",
            ]
        )
        if slot.get("on_repeat") is not None:
            lines.append(f"    on_repeat: {slot['on_repeat']}")
        lines.append(f"    description: {slot.get('description', 'A test slot')}")
        if slot.get("list"):
            lines.append("    list: true")
        if slot.get("binding_kind"):
            lines.append(f"    binding_kind: {slot['binding_kind']}")
        if slot.get("normalizer"):
            lines.append(f"    normalizer: {slot['normalizer']}")
        if slot.get("resolver"):
            lines.append(f"    resolver: {slot['resolver']}")
    if uses:
        lines.append("uses:")
        lines.extend(f"  - {target}" for target in uses)
    if presents:
        lines.append("presents:")
        lines.extend(f"  - {command}" for command in presents)
    lines.extend(["---", "", f"# {name}", "", body.rstrip(), ""])
    path = directory / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _goal(name: str) -> str:
    return f"The {name} goal is satisfied."


def test_name_must_equal_its_directory(tmp_path):
    path = _write_skill(tmp_path, "declared-name", directory_name="directory-name")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "name-matches-directory"


@pytest.mark.parametrize("level", ["task", "composite"])
def test_task_and_composite_require_a_goal(tmp_path, level):
    path = _write_skill(tmp_path, "parent", level=level)

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "goal-required"


def test_atomic_is_exempt_from_the_goal_requirement(tmp_path):
    _write_skill(tmp_path, "leaf", level="atomic")

    catalog = load_skill_catalog(str(tmp_path), _manifest())

    assert catalog["leaf"].goal is None


def test_every_uses_target_must_exist(tmp_path):
    path = _write_skill(
        tmp_path,
        "parent",
        level="task",
        goal=_goal("parent"),
        uses=("missing-child",),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "uses-target-exists"


@pytest.mark.parametrize(
    ("parent_level", "child_level"),
    [
        ("composite", "composite"),
        ("composite", "task"),
        ("composite", "atomic"),
        ("task", "task"),
        ("task", "atomic"),
    ],
)
def test_corrected_level_order_allows_only_downward_or_same_level_edges(
    tmp_path, parent_level, child_level
):
    _write_skill(
        tmp_path,
        "parent",
        level=parent_level,
        goal=_goal("parent"),
        uses=("child",),
    )
    _write_skill(
        tmp_path,
        "child",
        level=child_level,
        goal=None if child_level == "atomic" else _goal("child"),
    )

    catalog = load_skill_catalog(str(tmp_path), _manifest())

    assert catalog["parent"].uses == ("child",)


@pytest.mark.parametrize(
    ("parent_level", "child_level"),
    [
        ("task", "composite"),
        ("atomic", "composite"),
        ("atomic", "task"),
        ("atomic", "atomic"),
    ],
)
def test_corrected_level_order_rejects_upward_edges_and_atomic_children(
    tmp_path, parent_level, child_level
):
    path = _write_skill(
        tmp_path,
        "parent",
        level=parent_level,
        goal=None if parent_level == "atomic" else _goal("parent"),
        uses=("child",),
    )
    _write_skill(
        tmp_path,
        "child",
        level=child_level,
        goal=None if child_level == "atomic" else _goal("child"),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule in {"level-ordering", "atomic-is-leaf"}


def test_uses_graph_must_be_acyclic(tmp_path):
    _write_skill(
        tmp_path,
        "first",
        level="task",
        goal=_goal("first"),
        uses=("second",),
    )
    _write_skill(
        tmp_path,
        "second",
        level="task",
        goal=_goal("second"),
        uses=("first",),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "acyclic"
    assert "first -> second -> first" in str(excinfo.value)


def test_uses_graph_depth_is_at_most_three(tmp_path):
    _write_skill(
        tmp_path,
        "root",
        level="composite",
        goal=_goal("root"),
        uses=("middle-one",),
    )
    _write_skill(
        tmp_path,
        "middle-one",
        level="task",
        goal=_goal("middle one"),
        uses=("middle-two",),
    )
    _write_skill(
        tmp_path,
        "middle-two",
        level="task",
        goal=_goal("middle two"),
        uses=("leaf",),
    )
    _write_skill(tmp_path, "leaf", level="atomic")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "max-depth"
    assert "depth 4" in str(excinfo.value)


def test_depth_includes_a_terminal_task_command_leaf(tmp_path):
    _write_skill(
        tmp_path,
        "root",
        level="composite",
        goal=_goal("root"),
        uses=("middle",),
        body="1. middle",
    )
    _write_skill(
        tmp_path,
        "middle",
        level="task",
        goal=_goal("middle"),
        uses=("terminal",),
        body="1. terminal",
    )
    _write_skill(
        tmp_path,
        "terminal",
        level="task",
        goal=_goal("terminal"),
        body="1. `known`",
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "max-depth"
    assert "depth 4" in str(excinfo.value)


def test_required_slot_must_declare_on_repeat(tmp_path):
    path = _write_skill(
        tmp_path,
        "task",
        level="task",
        goal="The {subject} task is satisfied.",
        slots=(
            {
                "name": "subject",
                "required": True,
                "description": "The subject",
            },
        ),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "required-slot-needs-on-repeat"


def test_normalized_enum_slot_requires_a_registered_versioned_normalizer(tmp_path):
    path = _write_skill(
        tmp_path,
        "inspect",
        slots=(
            {
                "name": "entity_type",
                "binding_kind": "normalized_enum",
                "normalizer": "unknown@1",
            },
        ),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "slot-normalizer-known"


def test_registered_normalizer_is_exposed_on_the_selector_card(tmp_path):
    _write_skill(
        tmp_path,
        "inspect",
        slots=(
            {
                "name": "entity_type",
                "binding_kind": "normalized_enum",
                "normalizer": "entity-type@1",
            },
        ),
    )

    card = load_skill_catalog(str(tmp_path), _manifest()).cards()[0].as_dict()

    assert card["slots"][0]["binding_kind"] == "normalized_enum"
    assert card["slots"][0]["normalizer"] == "entity-type@1"


def test_candidate_resolver_must_be_registered_and_versioned(tmp_path):
    path = _write_skill(
        tmp_path,
        "investigate",
        slots=(
            {
                "name": "rule_query",
                "resolver": "unknown@1",
            },
        ),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "slot-resolver-known"


def test_registered_candidate_resolver_is_exposed_on_selector_card(tmp_path):
    _write_skill(
        tmp_path,
        "investigate",
        slots=(
            {
                "name": "rule_query",
                "resolver": "control-alias@1",
            },
        ),
    )

    card = load_skill_catalog(str(tmp_path), _manifest()).cards()[0].as_dict()

    assert card["slots"][0]["binding_kind"] == "exact_text"
    assert card["slots"][0]["resolver"] == "control-alias@1"


def test_candidate_resolver_and_presents_schema_are_unioned(tmp_path):
    _write_skill(
        tmp_path,
        "investigate",
        slots=(
            {
                "name": "rule_query",
                "resolver": "control-alias@1",
            },
        ),
        presents=("known",),
    )

    skill = load_skill_catalog(str(tmp_path), _manifest())["investigate"]
    card = skill.card().as_dict()

    assert skill.presents == ("known",)
    assert card["slots"][0]["binding_kind"] == "exact_text"
    assert card["slots"][0]["resolver"] == "control-alias@1"
    assert "presents" not in card


def test_for_each_may_only_iterate_a_list_slot(tmp_path):
    path = _write_skill(
        tmp_path,
        "batch",
        level="composite",
        goal="Every {subjects} subject is handled.",
        slots=(
            {
                "name": "subjects",
                "required": True,
                "on_repeat": "known",
                "description": "Subjects",
            },
        ),
        uses=("child",),
        body="1. for each {subject} in {subjects}: child subject={subject}",
    )
    _write_skill(
        tmp_path,
        "child",
        level="task",
        goal="The {subject} child is handled.",
        slots=(
            {
                "name": "subject",
                "required": True,
                "on_repeat": "known",
                "description": "Subject",
            },
        ),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "for-each-iterates-a-list-slot"


def test_for_each_may_only_name_a_child_in_uses(tmp_path):
    path = _write_skill(
        tmp_path,
        "batch",
        level="composite",
        goal="Every {subjects} subject is handled.",
        slots=(
            {
                "name": "subjects",
                "required": True,
                "on_repeat": "known",
                "description": "Subjects",
                "list": True,
            },
        ),
        uses=("declared-child",),
        body=("1. for each {subject} in {subjects}: " "other-child subject={subject}"),
    )
    _write_skill(
        tmp_path,
        "declared-child",
        level="task",
        goal=_goal("declared child"),
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "for-each-names-a-used-child"


def test_for_each_must_bind_its_loop_variable_to_the_child(tmp_path):
    path = _write_skill(
        tmp_path,
        "batch",
        level="composite",
        goal="Every {subjects} subject is handled.",
        slots=(
            {
                "name": "subjects",
                "required": True,
                "on_repeat": "known",
                "description": "Subjects",
                "list": True,
            },
        ),
        uses=("child",),
        body="1. for each {subject} in {subjects}: child fixed=value",
    )
    _write_skill(
        tmp_path,
        "child",
        level="task",
        goal="The child is handled.",
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "for-each-binds-loop-variable"


def test_body_may_name_only_manifest_commands(tmp_path):
    path = _write_skill(tmp_path, "leaf", body="1. `not_declared`")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest(commands=()))

    assert excinfo.value.path == str(path)
    assert excinfo.value.rule == "body-names-a-declared-command"
    assert "not_declared" in str(excinfo.value)


def test_body_accepts_manifest_commands_by_bare_name(tmp_path):
    _write_skill(tmp_path, "leaf", body="1. `known`")

    catalog = load_skill_catalog(str(tmp_path), _manifest(commands=("known",)))

    assert catalog.names == ("leaf",)


def test_core_verbs_are_always_allowed(tmp_path):
    _write_skill(
        tmp_path,
        "leaf",
        body="\n".join(
            f"{index}. `{verb}`"
            for index, verb in enumerate(sorted(CORE_VERBS), start=1)
        ),
    )

    catalog = load_skill_catalog(str(tmp_path), _manifest(commands=()))

    assert catalog.names == ("leaf",)


def test_absent_skills_folder_is_an_empty_catalogue(tmp_path):
    catalog = load_skill_catalog(str(tmp_path), _manifest())

    assert len(catalog) == 0
    assert catalog.mode == "enforce"


@pytest.mark.parametrize("mode", [None, "off"])
def test_undeclared_or_off_feature_opens_nothing_under_skills(
    tmp_path, monkeypatch, mode
):
    invalid = tmp_path / "_skills" / "broken" / "SKILL.md"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("this is deliberately invalid", encoding="utf-8")
    opened: list[str] = []
    original_builtin_open = builtins.open
    original_path_open = Path.open

    def recording_builtin_open(file, *args, **kwargs):
        try:
            opened.append(os.fspath(file))
        except TypeError:
            opened.append(repr(file))
        return original_builtin_open(file, *args, **kwargs)

    def recording_path_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_builtin_open)
    monkeypatch.setattr(Path, "open", recording_path_open)

    catalog = load_skill_catalog(str(tmp_path), _manifest(mode=mode))

    assert len(catalog) == 0
    assert not any("_skills" in Path(path).parts for path in opened)


def test_fingerprint_is_stable_and_one_byte_sensitive(tmp_path):
    workflow = tmp_path / "workflow"
    shutil.copytree(FIXTURE_WORKFLOW, workflow)
    manifest = load_manifest(str(workflow))

    first = load_skill_catalog(str(workflow), manifest)
    second = load_skill_catalog(str(workflow), manifest)
    assert first.fingerprint == second.fingerprint

    path = workflow / "_skills" / "inspect-thing" / "SKILL.md"
    before = path.read_text(encoding="utf-8")
    after = before.replace("portrait.", "portrait!", 1)
    assert len(after.encode("utf-8")) == len(before.encode("utf-8"))
    path.write_text(after, encoding="utf-8")

    changed = load_skill_catalog(str(workflow), manifest)
    assert changed.fingerprint != first.fingerprint


def test_declared_skills_fingerprint_must_match_loaded_content(tmp_path):
    _write_skill(tmp_path, "leaf")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(
            str(tmp_path),
            _manifest(skills_fingerprint="sha256:not-the-catalogue"),
        )

    assert excinfo.value.rule == "skills-fingerprint-matches"
    assert excinfo.value.path == str(tmp_path / "workflow_runtime.json")


def test_cards_never_carry_bodies():
    catalog = load_skill_catalog(
        str(FIXTURE_WORKFLOW), load_manifest(str(FIXTURE_WORKFLOW))
    )

    cards = catalog.cards()

    assert cards
    assert all(not hasattr(card, "body") for card in cards)
    assert all(
        set(card.as_dict()) == {"name", "description", "level", "slots"}
        for card in cards
    )
    assert all("body" not in card.as_dict() for card in cards)


def test_cards_expose_required_and_list_slot_shape():
    catalog = load_skill_catalog(
        str(FIXTURE_WORKFLOW), load_manifest(str(FIXTURE_WORKFLOW))
    )

    batch = next(card for card in catalog.cards() if card.name == "offboarding-batch")

    assert batch.as_dict()["slots"] == [
        {
            "name": "identity_queries",
            "description": "The leavers the request names",
            "required": True,
            "list": True,
            "binding_kind": "exact_text",
        }
    ]


def test_unknown_frontmatter_field_is_rejected(tmp_path):
    path = _write_skill(tmp_path, "leaf")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "level: atomic\n",
            "level: atomic\nsurprise: value\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "front-matter"
    assert "surprise" in str(excinfo.value)


def test_unknown_slot_field_is_rejected(tmp_path):
    path = _write_skill(
        tmp_path,
        "task",
        level="task",
        goal=_goal("task"),
        slots=(
            {
                "name": "subject",
                "required": False,
                "description": "Subject",
            },
        ),
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    description: Subject\n",
            "    description: Subject\n    surprise: value\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "front-matter"
    assert "surprise" in str(excinfo.value)


def test_invalid_boolean_is_rejected_instead_of_becoming_false(tmp_path):
    path = _write_skill(
        tmp_path,
        "task",
        level="task",
        goal=_goal("task"),
        slots=(
            {
                "name": "subject",
                "required": False,
                "description": "Subject",
            },
        ),
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    required: false\n",
            "    required: sometimes\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert excinfo.value.rule == "boolean-field"
    assert "sometimes" in str(excinfo.value)


def test_character_counts_are_reported_per_artifact():
    catalog = load_skill_catalog(
        str(FIXTURE_WORKFLOW), load_manifest(str(FIXTURE_WORKFLOW))
    )
    expected = {
        path.relative_to(FIXTURE_WORKFLOW / "_skills").as_posix(): len(
            path.read_text(encoding="utf-8")
        )
        for path in sorted((FIXTURE_WORKFLOW / "_skills").glob("*/SKILL.md"))
    }

    assert catalog.character_counts == expected
    assert catalog.total_characters == sum(expected.values())


def test_approved_levels_are_in_corrected_order():
    assert APPROVED_LEVELS == ("composite", "task", "atomic")


def test_runtime_manifest_carries_skills_gate_and_fingerprint():
    declared = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        features={"skills_v1": "enforce"},
        skills_fingerprint="sha256:declared-skills",
    )

    metadata = merge_and_gate(
        declared,
        deployment_features={"skills_v1": "shadow"},
    )

    assert metadata.feature_mode("skills_v1") == "shadow"
    assert metadata.skills_fingerprint == "sha256:declared-skills"


def test_real_ido_catalogue_loads_read_only():
    manifest = load_manifest(str(IDO_WORKFLOW))
    assert manifest is not None
    manifest_data = manifest.model_dump(mode="python")
    manifest_data["features"] = {
        **manifest_data["features"],
        "skills_v1": "enforce",
    }
    enabled_manifest = RuntimeManifest.model_validate(manifest_data)

    catalog = load_skill_catalog(str(IDO_WORKFLOW), enabled_manifest)

    expected_names = {
        "access-request-triage",
        "application-recertification",
        "control-failure-investigation",
        "control-sweep",
        "cross-system-privilege-audit",
        "department-quarterly-review",
        "department-roster-walk",
        "entity-portrait-review",
        "explain-finding",
        "finding-explanation",
        "inspect-entity",
        "leaver-batch",
        "leaver-offboarding-sweep",
        "permission-holder-sweep",
        "privilege-exception-review",
        "unit-review-packet",
    }
    assert set(catalog.names) == expected_names
    by_level = {
        level: {skill.name for skill in catalog.skills.values() if skill.level == level}
        for level in APPROVED_LEVELS
    }
    assert {level: len(names) for level, names in by_level.items()} == {
        "composite": 4,
        "task": 10,
        "atomic": 2,
    }
    rank = {"composite": 0, "task": 1, "atomic": 2}
    for composite in (
        skill for skill in catalog.skills.values() if skill.level == "composite"
    ):
        assert all(
            rank[catalog[child].level] > rank[composite.level]
            for child in composite.uses
        )
