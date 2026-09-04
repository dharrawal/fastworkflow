---
name: author-workflow-skills
description: >-
  Author or review workflow `_skills/*/SKILL.md` files and choose the correct skill level:
  composite, task, or atomic. Use when someone asks why there are three levels instead of two,
  when writing a new SKILL.md, when a skill is labeled composite but has no child skills, when
  `uses` points the wrong way up the tree, or when a task is missing a `goal`. Do not use for
  context inheritance/hierarchy JSON (design-context-models) or for `available_from` on command
  parameters (declare-parameter-producers).
---

# Authoring workflow skills

Skills live at `<workflow>/_skills/<name>/SKILL.md`. The loader is
`fastworkflow.skill_catalog`. Enablement is by the runtime manifest feature
`skills_v1`, never by file presence.

The intern-length explanation of the three levels is
[docs/skill_levels.md](../../../docs/skill_levels.md). Read that before inventing a
fourth level or collapsing `task` into `composite`.

## The three levels

| Level | What it is | `uses:` | `goal:` |
|---|---|---|---|
| `atomic` | Shared how-to (find/open/portrait). Not a ticket. | empty | omitted (private goal; parent owns the predicate) |
| `task` | One named operator job | atomics or other tasks | required |
| `composite` | Several of those jobs in one utterance | child skills, usually tasks | required |

Tiny in-tree examples: `tests/fixtures/skills_workflow/_skills/`
(`inspect-thing` / `leaver-sweep` / `offboarding-batch`).

`uses` may not run up the tree: composite → task → atomic. Depth ≤ 3.
An atomic body names only commands and deterministic control flow.

## Do not confuse with frozen composite *commands*

A command whose internals already bundle several operations (IDO `open_portrait`)
is a **command**. Do not mark a skill `composite` because its body runs more than
one command. No child skills → it is atomic.

## Selection surface

`cards()` is what the selector sees: name, description, level, slot names and
slot descriptions. Skill **bodies never reach the selector**. Overpromising in
`description` relative to the body is a content defect, not a runtime feature.
