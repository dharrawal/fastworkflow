"""The skill catalogue: `<workflow>/_skills/<name>/SKILL.md`, loaded and validated.

EXP-028 decision 1. Ten SKILL.md files have sat in `ido_workflow/_skills` with
`level`, `slots` and `uses` metadata and **no consumer** — `provenance.py`'s
comment saying no Python reads it was the whole of the integration. This module
is the consumer, and it is a *loader plus a validator* rather than a reader: the
metadata only becomes load-bearing once something can fail on it.

**Enablement is by manifest, never by file presence** (FW-REQ-006 clause 1,
FW-REQ-019C clause 2). Two no-op cases, not one: the folder is absent, and the
folder is present while the manifest does not declare `skills_v1`. The second is
the one that matters, and `load_skill_catalog` does not merely return an empty
catalogue for it — it returns before touching the filesystem at all. A loader
that reads in `off` is a loader that can *fail* in `off`, and the byte-identical
guarantee EXP-028 decision 6 rests on is asserted with a spy on `open`. That
guarantee is only spy-able if there is no read to spy on.

**Bodies never reach the selector** (FW-REQ-006 clause 3). `cards()` is the
selection surface — name, description, level, slot names and slot descriptions —
and it is a separate projection rather than a documented convention, because a
convention that the prompt builder is supposed to honour is one refactor away
from shipping ten skill bodies to a model. `character_counts` reports per-
artifact size (clause 4) so the cost of the catalogue is a measured number
rather than an impression.

**`goal` is the one schema addition.** FW-REQ-010 clause 2 forbids a plan whose
nodes are command invocations, and a node built from `name` + `uses` is exactly
that. A `task` or `composite` skill without a `goal` fails validation here, at
the workflow conformance gate, rather than at the turn. `atomic` skills are
exempt: they are private goals by construction (§4.12) and their predicate is
their parent's.

**The frontmatter parser is deliberately small**, and deliberately the same
shape as the hand parser in ido's `tests/test_skill_conformance.py`: the front
matter is a fixed shape, the checks are about what the fields say, and adding a
YAML dependency to read four scalars and two lists would be paying a supply
chain for a regex. Once this module exists it is the schema authority and that
test asserts *content* against it — leaving both parsers as authorities is a
guarantee they will disagree (EXP-028 "Two parsers is one too many").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow.runtime_manifest import MANIFEST_FILENAME, canonical_content_hash

#: The manifest feature id that turns this module on. `skills` at version 1,
#: spelled the way `runtime_manifest._feature` spells every other feature.
SKILLS_FEATURE_ID = "skills_v1"

#: Where skills live, relative to the workflow root.
SKILLS_DIRNAME = "_skills"

SKILL_FILENAME = "SKILL.md"

#: Requirements §4.6. `block` was in use once and is not one of them.
APPROVED_LEVELS = ("composite", "task", "atomic")

#: The ordering the `uses` graph may not run backwards along:
#: ``task`` -> ``composite`` -> ``atomic``, never upward (decision 1). A target's
#: level must rank at or below its parent's, so a `task` may use a `composite`
#: and a `composite` may not use a `task`.
#: The review correction supersedes the pre-amendment sentence above:
#: ``composite`` -> ``task`` -> ``atomic``. A composite may use any level, a
#: task may use task or atomic, and an atomic is a leaf.
LEVEL_RANK: dict[str, int] = {"composite": 0, "task": 1, "atomic": 2}

#: FW-REQ-012 clause 3.
MAX_DEPTH = 3

#: The four core verbs every workflow inherits from the CME workflow. They are
#: always nameable by a body: they exist in `CORE_MANIFEST` rather than in the
#: workflow's own manifest, so checking a body against the workflow manifest
#: alone would reject `go_up` — which three ido skills name, correctly.
CORE_VERBS = frozenset(
    {"go_up", "reset_context", "what_can_i_do", "what_is_current_context"}
)

#: Framework vocabulary a body may name in code voice without naming a command.
#: `available_from` is the parameter hint the expander reads (decision 3) and
#: three ido bodies tell the agent to walk it; it is a manifest key, not a verb.
#: Kept explicit and small: the alternative is a heuristic that guesses whether
#: a snake_case token is a command, and a heuristic here would silently stop
#: catching the defect this rule exists for — a body naming a command that does
#: not exist.
FRAMEWORK_VOCABULARY = frozenset(
    {
        "available_from",
        "entity_type",
        "navigation_effect",
        "on_repeat",
        "query",
        "resource_type",
        "target_context",
        "uid",
    }
)

_FRONT_MATTER_OPEN = "---\n"
_FRONT_MATTER_CLOSE = "\n---\n"

# `2. inspect-entity ...` — a numbered step. Prose around the steps is not a
# step (ido's conformance suite makes the same distinction, for the same
# reason: `application-recertification` names a command in a note saying the
# step is deliberately absent).
_NUMBERED_STEP = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")

# The review amendment's one new step form:
#   for each {x} in {xs}: <child-skill> <slot>={x}
_FOR_EACH = re.compile(
    r"^for\s+each\s+\{(?P<var>[^{}]+)\}\s+in\s+\{(?P<list_slot>[^{}]+)\}\s*:\s*(?P<rest>.*\S)\s*$",
    re.IGNORECASE,
)

# `entity_type=identity`, `query={identity_query}`, backticks optional.
_SLOT_ARGUMENT = re.compile(
    r"`?(?P<slot>[a-z][a-z0-9_]*)\s*=\s*(?P<value>\{[^{}]+\}|[^\s`]+)`?"
)

# A bare identifier in code voice. Anything with a space, a `*`, an `=`, braces
# or a capital is not a command reference: `find_*` is a family, `{uid}` a
# placeholder, `ControlFinding` a context, `visited N of T` a sentence.
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BARE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

_FRONT_MATTER_FIELDS = frozenset(
    {"name", "description", "level", "goal", "slots", "uses"}
)
_LIST_FIELDS = frozenset({"slots", "uses"})
_SLOT_FIELDS = frozenset({"name", "required", "on_repeat", "description", "list"})


class SkillCatalogError(ValueError):
    """A skill that does not load, naming the file and the rule it broke.

    Both halves are the point. "invalid skill" sends an author reading ten
    files; the rule name is what makes the message actionable at the
    conformance gate, which is where decision 1 puts these failures — at load,
    failing the workflow, rather than at the turn.
    """

    def __init__(self, path: Any, rule: str, detail: str) -> None:
        self.path = str(path)
        self.rule = rule
        self.detail = detail
        super().__init__(f"{self.path}: [{rule}] {detail}")


@dataclass(frozen=True)
class Slot:
    """One declared slot of a skill.

    `on_repeat` is the deterministic fallback AGENTS.md promises and
    FW-REQ-011 clause 6 requires: ask once, then act on a declared default. It
    is mandatory on a required slot (decision 1) because a required slot with
    nowhere to go leaves the second half of that rule undefined.

    `list` is the review amendment's addition: a slot the selector binds to
    several values at once, which a `for each` step fans out. Named `list`
    rather than `multi` because that is what the amendment writes in the
    frontmatter, and a schema field that does not match its own document is a
    schema field authors get wrong.
    """

    name: str
    required: bool = False
    on_repeat: Optional[str] = None
    description: str = ""
    list: bool = False

    @property
    def is_list(self) -> bool:
        """`list` under a name that does not shadow the builtin at call sites."""
        return self.list


@dataclass(frozen=True)
class Step:
    """One step of a skill body, already classified by the grammar.

    Three kinds, and the classification is deterministic:

    * ``skill`` — the step *begins* with a name in `uses`. Begins, not
      mentions: `cross-system-privilege-audit` step 4 is "If the permission
      portrait names an application, inspect-entity that application", a
      conditional whose condition only the executor can evaluate, and treating
      it as a child invocation would bind that child's two required slots from
      nothing. A conditional mention is a command sequence.
    * ``for_each`` — the amendment's fan-out form.
    * ``commands`` — everything else, executed as a command sequence.
    """

    ordinal: int
    kind: str  # "skill" | "for_each" | "commands"
    text: str
    skill: Optional[str] = None
    arguments: Mapping[str, str] = field(default_factory=dict)
    loop_variable: Optional[str] = None
    list_slot: Optional[str] = None


@dataclass(frozen=True)
class SkillCard:
    """What the selector sees. Never a body (FW-REQ-006 clause 3)."""

    name: str
    description: str
    level: str
    slots: tuple[tuple[str, str], ...]  # (slot name, slot description)
    required_slots: frozenset[str] = frozenset()
    list_slots: frozenset[str] = frozenset()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level,
            "slots": [
                {
                    "name": name,
                    "description": description,
                    "required": name in self.required_slots,
                    "list": name in self.list_slots,
                }
                for name, description in self.slots
            ],
        }


@dataclass(frozen=True)
class Skill:
    """One validated SKILL.md."""

    name: str
    description: str
    level: str
    goal: Optional[str]
    slots: tuple[Slot, ...]
    uses: tuple[str, ...]
    body: str
    path: str
    content_hash: str
    steps: tuple[Step, ...] = ()

    def slot(self, name: str) -> Optional[Slot]:
        return next((s for s in self.slots if s.name == name), None)

    @property
    def required_slots(self) -> tuple[Slot, ...]:
        return tuple(s for s in self.slots if s.required)

    @property
    def is_leaf_level(self) -> bool:
        """`atomic` — a leaf of the decomposition tree (decision 3)."""
        return self.level == "atomic"

    def card(self) -> SkillCard:
        return SkillCard(
            name=self.name,
            description=self.description,
            level=self.level,
            slots=tuple((s.name, s.description) for s in self.slots),
            required_slots=frozenset(s.name for s in self.slots if s.required),
            list_slots=frozenset(s.name for s in self.slots if s.list),
        )


class SkillCatalog:
    """An immutable, validated index of one workflow's skills.

    `fingerprint` is `sha256` over the sorted (relative path, content) pairs,
    computed with `runtime_manifest.canonical_content_hash` rather than a second
    recipe here — an experiment whose treatment is partly workflow *content* and
    whose ledger cannot say which content ran is not a treatment arm, and two
    implementations of "the fingerprint" is how a ledger starts lying about it.
    """

    def __init__(
        self,
        skills: Mapping[str, Skill] = (),  # type: ignore[assignment]
        *,
        fingerprint: Optional[str] = None,
        mode: str = "off",
        character_counts: Optional[Mapping[str, int]] = None,
        root: Optional[str] = None,
    ) -> None:
        self._skills: dict[str, Skill] = dict(skills or {})
        self.mode = mode
        self.root = root
        self.fingerprint = (
            fingerprint if fingerprint is not None else canonical_content_hash(())
        )
        self.character_counts: Mapping[str, int] = dict(character_counts or {})

    # -- mapping-ish access ------------------------------------------------
    @property
    def skills(self) -> Mapping[str, Skill]:
        return dict(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(sorted(self._skills))

    def __getitem__(self, name: str) -> Skill:
        return self._skills[name]

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    @property
    def is_enabled(self) -> bool:
        """Anything other than `off`. Shadow counts: it loads, it just cannot act."""
        return self.mode != "off"

    @property
    def total_characters(self) -> int:
        return sum(self.character_counts.values())

    def cards(self) -> tuple[SkillCard, ...]:
        """The selection surface, in name order. Bodies are not in it."""
        return tuple(self._skills[name].card() for name in sorted(self._skills))

    def depth(self, name: str) -> int:
        """Longest expanded node chain from `name`, counting it as 1."""
        return _depth(name, self._skills, {})


def load_skill_catalog(
    workflow_path: Optional[str], manifest: Any = None
) -> SkillCatalog:
    """Load `<workflow>/_skills`, gated on the manifest feature `skills_v1`.

    `manifest` is duck-typed on purpose: a `RuntimeMetadata` (the merged,
    dual-gated view, which is what a running workflow holds) answers
    `feature_mode`, a raw `RuntimeManifest` carries `features`, and `None` is a
    workflow with no manifest at all. All three are legitimate callers and none
    of them should have to be converted before asking a yes/no question.

    Returns an empty catalogue — **without opening the folder** — when the
    feature is `off` or undeclared. See the module docstring.
    """
    mode = skills_mode(manifest)
    if mode == "off":
        return SkillCatalog(mode="off")
    if not workflow_path:
        return SkillCatalog(mode=mode)

    root = Path(workflow_path) / SKILLS_DIRNAME
    if not root.is_dir():
        # FW-REQ-006 clause 7: a missing file is a no-op. A manifest that
        # declares the feature over a workflow with no skills gets an empty
        # catalogue, and `plan.require_catalog` is what turns that into the
        # hard error when a mode was configured to act on it.
        return SkillCatalog(mode=mode, root=str(root))

    parsed: dict[str, Skill] = {}
    entries: list[tuple[str, bytes]] = []
    counts: dict[str, int] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        path = directory / SKILL_FILENAME
        if not path.is_file():
            raise SkillCatalogError(
                path,
                "skill-file-missing",
                f"'{directory.name}' is a skill directory with no {SKILL_FILENAME}",
            )
        text = path.read_text(encoding="utf-8")
        skill = _parse_skill(text, path=path, directory_name=directory.name)
        parsed[skill.name] = skill
        relative_path = f"{directory.name}/{SKILL_FILENAME}"
        entries.append((relative_path, text.encode("utf-8")))
        counts[relative_path] = len(text)

    _validate_graph(parsed)
    _validate_bodies(parsed, manifest)

    fingerprint = canonical_content_hash(entries)
    declared_fingerprint = getattr(manifest, "skills_fingerprint", None)
    if declared_fingerprint and declared_fingerprint != fingerprint:
        raise SkillCatalogError(
            Path(workflow_path) / MANIFEST_FILENAME,
            "skills-fingerprint-matches",
            f"manifest declares {declared_fingerprint}, but the loaded "
            f"catalogue hashes to {fingerprint}",
        )

    return SkillCatalog(
        parsed,
        fingerprint=fingerprint,
        mode=mode,
        character_counts=counts,
        root=str(root),
    )


def skills_mode(manifest: Any) -> str:
    """The effective `skills_v1` mode of `manifest`: off | shadow | enforce."""
    if manifest is None:
        return "off"
    feature_mode = getattr(manifest, "feature_mode", None)
    if callable(feature_mode):
        return feature_mode(SKILLS_FEATURE_ID) or "off"
    features = getattr(manifest, "features", None)
    if isinstance(features, Mapping):
        return features.get(SKILLS_FEATURE_ID, "off") or "off"
    return "off"


# ----------------------------------------------------------------------
# Frontmatter
# ----------------------------------------------------------------------


def parse_front_matter(text: str, path: Any) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into ``(fields, body)``.

    The YAML subset: `key: scalar` at column 0, `  - item` under a list key, and
    `  - name: x` / `    key: value` for the slot mappings. That is the whole of
    what these files use and the whole of what this reads.
    """
    if not text.startswith(_FRONT_MATTER_OPEN):
        raise SkillCatalogError(path, "front-matter", "no front matter")
    remainder = text[len(_FRONT_MATTER_OPEN) :]
    if _FRONT_MATTER_CLOSE not in remainder:
        raise SkillCatalogError(path, "front-matter", "front matter is not closed")
    front, body = remainder.split(_FRONT_MATTER_CLOSE, 1)

    fields: dict[str, Any] = {}
    section: Optional[str] = None
    slot: Optional[dict[str, str]] = None
    for line in front.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if match := re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line):
            key, value = match.group(1), match.group(2).strip()
            if key not in _FRONT_MATTER_FIELDS:
                raise SkillCatalogError(path, "front-matter", f"unknown field '{key}'")
            if key in fields:
                raise SkillCatalogError(
                    path, "front-matter", f"field '{key}' is declared twice"
                )
            section = key
            slot = None
            if value:
                if key in _LIST_FIELDS:
                    raise SkillCatalogError(
                        path,
                        "front-matter",
                        f"list field '{key}' must use indented '- ' entries",
                    )
                fields[key] = _unquote(value)
            else:
                fields[key] = [] if key in _LIST_FIELDS else ""
            continue
        if section is None:
            raise SkillCatalogError(path, "front-matter", f"unrecognized line {line!r}")
        if match := re.match(r"^\s*-\s+(\w+):\s*(.*)$", line):
            # A mapping entry of a list: `- name: identity_query`.
            key = match.group(1)
            if section != "slots" or key not in _SLOT_FIELDS:
                raise SkillCatalogError(
                    path,
                    "front-matter",
                    f"mapping entry '{key}' is not valid under '{section}'",
                )
            slot = {key: _unquote(match.group(2).strip())}
            fields[section].append(slot)
            continue
        if match := re.match(r"^\s*-\s+(\S.*)$", line):
            # A scalar entry of a list: `- inspect-entity`.
            if section != "uses":
                raise SkillCatalogError(
                    path,
                    "front-matter",
                    f"scalar list entry is not valid under '{section}'",
                )
            slot = None
            fields[section].append(_unquote(match.group(1).strip()))
            continue
        if slot is not None and (match := re.match(r"^\s+(\w+):\s*(.*)$", line)):
            key = match.group(1)
            if section != "slots" or key not in _SLOT_FIELDS:
                raise SkillCatalogError(
                    path, "front-matter", f"unknown slot field '{key}'"
                )
            if key in slot:
                raise SkillCatalogError(
                    path,
                    "front-matter",
                    f"slot field '{key}' is declared twice",
                )
            slot[key] = _unquote(match.group(2).strip())
            continue
        raise SkillCatalogError(path, "front-matter", f"unrecognized line {line!r}")
    return fields, body


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _as_bool(value: Any, *, path: Any, field_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("true", "yes", "1"):
        return True
    if normalized in ("false", "no", "0"):
        return False
    raise SkillCatalogError(
        path,
        "boolean-field",
        f"'{field_name}' must be true or false, got {value!r}",
    )


def _parse_skill(text: str, *, path: Any, directory_name: str) -> Skill:
    fields, body = parse_front_matter(text, path)

    for required in ("name", "description", "level"):
        value = fields.get(required)
        if not isinstance(value, str) or not value.strip():
            raise SkillCatalogError(
                path, "required-field", f"'{required}' is missing or empty"
            )

    name = fields["name"].strip()
    level = fields["level"].strip()
    if name != directory_name:
        raise SkillCatalogError(
            path,
            "name-matches-directory",
            f"declares name '{name}' in directory '{directory_name}'",
        )
    if level not in APPROVED_LEVELS:
        raise SkillCatalogError(
            path,
            "approved-level",
            f"declares level '{level}'; approved levels are "
            + ", ".join(APPROVED_LEVELS),
        )

    goal = fields.get("goal")
    goal = goal.strip() if isinstance(goal, str) else None
    if level in ("task", "composite") and not goal:
        raise SkillCatalogError(
            path,
            "goal-required",
            f"level '{level}' requires a 'goal': FW-REQ-010 clause 2 forbids a "
            "plan whose nodes are command invocations, and a node built from "
            "name and uses is exactly that",
        )

    slots: list[Slot] = []
    for entry in fields.get("slots") or ():
        if not isinstance(entry, dict) or not entry.get("name"):
            raise SkillCatalogError(
                path, "slot-shape", f"slot entry {entry!r} declares no name"
            )
        slots.append(
            Slot(
                name=entry["name"].strip(),
                required=_as_bool(
                    entry.get("required", "false"),
                    path=path,
                    field_name=f"{entry['name']}.required",
                ),
                on_repeat=(entry.get("on_repeat") or "").strip() or None,
                description=(entry.get("description") or "").strip(),
                list=_as_bool(
                    entry.get("list", "false"),
                    path=path,
                    field_name=f"{entry['name']}.list",
                ),
            )
        )

    seen: set[str] = set()
    for slot in slots:
        if slot.name in seen:
            raise SkillCatalogError(
                path, "slot-shape", f"slot '{slot.name}' is declared twice"
            )
        seen.add(slot.name)
        if slot.required and not slot.on_repeat:
            raise SkillCatalogError(
                path,
                "required-slot-needs-on-repeat",
                f"slot '{slot.name}' is required but does not say what to do "
                "when the user repeats without answering; FW-REQ-011 clause 6 "
                "is ask once, then act on a declared default",
            )

    uses: list[str] = []
    for entry in fields.get("uses") or ():
        target = entry.get("name") if isinstance(entry, dict) else entry
        if not isinstance(target, str) or not target.strip():
            raise SkillCatalogError(path, "uses-shape", f"uses entry {entry!r}")
        if target.strip() not in uses:
            uses.append(target.strip())

    steps = parse_steps(body, uses=tuple(uses), path=path)
    _validate_for_each(steps, slots=tuple(slots), uses=tuple(uses), path=path)

    return Skill(
        name=name,
        description=fields["description"].strip(),
        level=level,
        goal=goal,
        slots=tuple(slots),
        uses=tuple(uses),
        body=body,
        path=str(path),
        content_hash=canonical_content_hash(
            [(f"{directory_name}/{SKILL_FILENAME}", text.encode("utf-8"))]
        ),
        steps=steps,
    )


# ----------------------------------------------------------------------
# Body grammar
# ----------------------------------------------------------------------


def parse_steps(
    body: str, *, uses: Sequence[str] = (), path: Any = ""
) -> tuple[Step, ...]:
    """The numbered steps of a body, classified. Prose is not a step.

    A body with no numbered steps has no steps, and expansion turns such a node
    into a single command-sequence leaf carrying the whole body — see
    `plan.expand`. `control-failure-investigation` is that shape today.
    """
    steps: list[Step] = []
    for line in body.splitlines():
        stripped = line.strip()
        match = _NUMBERED_STEP.match(line)
        text = match.group(2) if match else None
        if text is None and stripped.lower().startswith("for each "):
            # The amendment's form is admitted unnumbered too: a fan-out is
            # often the whole of a body.
            text = stripped
        if text is None:
            continue
        steps.append(_classify_step(text, ordinal=len(steps) + 1, uses=uses))
    return tuple(steps)


def _classify_step(text: str, *, ordinal: int, uses: Sequence[str]) -> Step:
    if match := _FOR_EACH.match(text):
        rest = match.group("rest")
        child, arguments = _leading_skill(rest, uses)
        return Step(
            ordinal=ordinal,
            kind="for_each",
            text=text,
            skill=child,
            arguments=arguments,
            loop_variable=match.group("var").strip(),
            list_slot=match.group("list_slot").strip(),
        )
    child, arguments = _leading_skill(text, uses)
    if child is not None:
        return Step(
            ordinal=ordinal, kind="skill", text=text, skill=child, arguments=arguments
        )
    return Step(ordinal=ordinal, kind="commands", text=text)


def _leading_skill(
    text: str, uses: Sequence[str]
) -> tuple[Optional[str], dict[str, str]]:
    """`(child skill, slot arguments)` when `text` *begins* with a `uses` name."""
    candidate = text.strip().strip("`")
    for name in sorted(uses, key=len, reverse=True):
        if candidate == name or candidate.startswith(f"{name} "):
            return name, _slot_arguments(candidate[len(name) :])
    return None, {}


def _slot_arguments(text: str) -> dict[str, str]:
    return {
        match.group("slot"): match.group("value").strip("`")
        for match in _SLOT_ARGUMENT.finditer(text)
    }


def _validate_for_each(
    steps: Sequence[Step], *, slots: Sequence[Slot], uses: Sequence[str], path: Any
) -> None:
    """The amendment's two rules, and they are the whole of its validation."""
    by_name = {slot.name: slot for slot in slots}
    for step in steps:
        if step.kind != "for_each":
            continue
        slot = by_name.get(step.list_slot or "")
        if slot is None or not slot.list:
            raise SkillCatalogError(
                path,
                "for-each-iterates-a-list-slot",
                f"step {step.ordinal} iterates '{step.list_slot}', which is not "
                "a slot declared 'list: true'",
            )
        if step.skill is None or step.skill not in uses:
            raise SkillCatalogError(
                path,
                "for-each-names-a-used-child",
                f"step {step.ordinal} fans out to "
                f"'{step.skill or step.text}', which is not in uses",
            )
        loop_reference = "{%s}" % (step.loop_variable or "")
        if loop_reference not in step.arguments.values():
            raise SkillCatalogError(
                path,
                "for-each-binds-loop-variable",
                f"step {step.ordinal} does not bind {loop_reference} to a "
                "child slot",
            )


# ----------------------------------------------------------------------
# Graph validation
# ----------------------------------------------------------------------


def _validate_graph(skills: Mapping[str, Skill]) -> None:
    for name in sorted(skills):
        skill = skills[name]
        if skill.level == "atomic" and skill.uses:
            raise SkillCatalogError(
                skill.path,
                "atomic-is-leaf",
                f"atomic skill '{skill.name}' uses {', '.join(skill.uses)}; "
                "atomic skills are decomposition leaves and cannot use children",
            )
        for target in skill.uses:
            child = skills.get(target)
            if child is None:
                raise SkillCatalogError(
                    skill.path,
                    "uses-target-exists",
                    f"uses '{target}', which is not a skill in this catalogue",
                )
            if LEVEL_RANK[child.level] < LEVEL_RANK[skill.level]:
                raise SkillCatalogError(
                    skill.path,
                    "level-ordering",
                    f"level '{skill.level}' uses '{target}' at level "
                    f"'{child.level}'; the ordering is composite -> task -> "
                    "atomic and a target's level must be at or below its "
                    "parent's, never upward",
                )

    _validate_acyclic(skills)

    for name in sorted(skills):
        depth = _depth(name, skills, {})
        if depth > MAX_DEPTH:
            raise SkillCatalogError(
                skills[name].path,
                "max-depth",
                f"expands to depth {depth}; FW-REQ-012 clause 3 bounds it at "
                f"{MAX_DEPTH}",
            )


def _validate_acyclic(skills: Mapping[str, Skill]) -> None:
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {name: WHITE for name in skills}

    def visit(name: str, trail: list[str]) -> None:
        colour[name] = GREY
        for target in skills[name].uses:
            if colour.get(target) == GREY:
                cycle = " -> ".join(trail[trail.index(target) :] + [target])
                raise SkillCatalogError(
                    skills[name].path, "acyclic", f"uses cycle: {cycle}"
                )
            if colour.get(target) == WHITE:
                visit(target, trail + [target])
        colour[name] = BLACK

    for name in sorted(skills):
        if colour[name] == WHITE:
            visit(name, [name])


def _depth(name: str, skills: Mapping[str, Skill], memo: dict[str, int]) -> int:
    if name in memo:
        return memo[name]
    skill = skills.get(name)
    if skill is None or skill.level == "atomic":
        memo[name] = 1
        return 1
    # Guarded by `_validate_acyclic`, which runs first; the sentinel keeps a
    # caller that reordered them from recursing forever instead of failing.
    memo[name] = 1
    child_depths = [_depth(target, skills, memo) for target in skill.uses]
    if not skill.steps or any(
        step.kind == "commands" or step.skill is None for step in skill.steps
    ):
        child_depths.append(1)
    memo[name] = 1 + max(child_depths, default=0)
    return memo[name]


# ----------------------------------------------------------------------
# Bodies name only commands the manifest declares
# ----------------------------------------------------------------------


def _declared_commands(manifest: Any) -> set[str]:
    """Bare command names the manifest declares, by last path segment.

    `Identity/list_accounts` and `Repository/list_accounts` are two commands
    with one bare name, and a skill body says `list_accounts`. Matching on the
    last segment is what the bodies actually reference; qualifying them would
    be a body-authoring change, not a validation rule.
    """
    commands = getattr(manifest, "commands", None) or {}
    return {str(name).split("/")[-1] for name in commands}


def _validate_bodies(skills: Mapping[str, Skill], manifest: Any) -> None:
    declared = _declared_commands(manifest)
    if manifest is None:
        # No manifest, or one that declares no commands: there is nothing to
        # check a body against, and inventing a check would reject every skill
        # in a workflow whose manifest is command-silent.
        # An enabled, command-silent manifest is nevertheless an authority: the
        # only valid command references in that case are the inherited core
        # verbs. The broader historical no-op above now applies only when there
        # is no manifest object at all.
        return
    allowed = declared | CORE_VERBS | FRAMEWORK_VOCABULARY
    for name in sorted(skills):
        skill = skills[name]
        for token in sorted(_command_tokens(skill.body)):
            if token not in allowed:
                raise SkillCatalogError(
                    skill.path,
                    "body-names-a-declared-command",
                    f"names '{token}', which the runtime manifest does not "
                    "declare as a command",
                )


def _command_tokens(body: str) -> set[str]:
    """Bare identifiers in code voice — the body's command references."""
    return {span for span in _CODE_SPAN.findall(body) if _BARE_IDENTIFIER.match(span)}


def workflow_skills_fingerprint(workflow_path: str) -> str:
    """The fingerprint of a tree's `_skills`, computed without validating it.

    For a generator (`gen_ido_scaffold`) that has to write `skills_fingerprint`
    into a manifest before the workflow it describes is loadable.
    """
    root = Path(workflow_path) / SKILLS_DIRNAME
    if not root.is_dir():
        return canonical_content_hash(())
    entries: list[tuple[str, bytes]] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        path = directory / SKILL_FILENAME
        if path.is_file():
            entries.append((f"{directory.name}/{SKILL_FILENAME}", path.read_bytes()))
    return canonical_content_hash(entries)
