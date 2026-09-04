# Why workflow skills have three levels

**Audience:** anyone authoring `_skills/*/SKILL.md`, or asking why the vocabulary is `composite`, `task`, and `atomic` rather than just composite and atomic.

**Normative one-liner:** a skill is a workflow-authored procedure with metadata, slots, and steps. The three levels are defined in IDO requirements §4.6 and enforced by `fastworkflow.skill_catalog` (`APPROVED_LEVELS`, `LEVEL_RANK`, `MAX_DEPTH`).

Worked examples in this repo: `tests/fixtures/skills_workflow/_skills/`. Worked examples in the IDO reference workflow: `ido_workflow/_skills/`.

---

## The question this document answers

The usual recipe is two layers: **big jobs call small jobs**. That intuition is right. fastWorkflow still needs a third name because “big job” is doing two different jobs:

- naming **one operator goal** you can put on a ticket and score;
- naming **several of those goals in one utterance**.

Calling both of those “composite” makes the catalogue, the contracts, and the expander lie to each other.

---

## Atomic — a reusable how-to, not a job

An **atomic** skill is a shared lookup or navigation recipe. It never `uses:` another skill. Its body names only commands and deterministic control flow (if/then, “already in this context”, ask-once).

`inspect-thing` in the fixture (IDO: `inspect-entity`) is the poster child: bind a type and a name, find it, open it, `open_portrait`. Operators can ask for that directly (“show me Alice”), and every larger skill can reuse it instead of rewriting find/open.

Atomic skills are **private goals** by construction. They do not need a `goal:` predicate of their own; the parent’s goal is the one that is scored. That is why the loader requires `goal` on `task` and `composite` and exempts `atomic`.

If you label an atomic skill `composite` just because it runs several commands, you have mixed two different ideas. A **frozen composite command** (for example IDO’s `open_portrait`) is already one command and must not be expanded back into its constituent commands (§4.6 last sentence). An atomic *skill* is still a skill that *calls* commands. EXP-020 caught `inspect-entity` being mislabeled `composite` while it had no child skills; under §4.6 that made it atomic.

---

## Task — one named job the operator would put on a ticket

A **task** skill is **one complete operator goal**, with a `goal:` sentence you can evaluate. Examples: “offboard this one person,” “quarterly-review this one department.”

It may `uses:` an atomic as a helper, then continue with its own domain steps. Fixture `leaver-sweep` (IDO: `leaver-offboarding-sweep`) does not stop after inspecting the person. It walks a login and proposes remediations — and it does not apply them.

That is the catalogue entry. Supported-task contracts, Pass@1, “did we finish the job?” hang off **task**, not off the helper.

If you only had composite vs atomic, this would have to pretend to be composite because it uses `inspect-thing`. Then “one leaver” and “seven leavers plus a recert in the same message” would look like the same kind of object.

---

## Composite — several of those jobs in one utterance

A **composite** skill does not invent new domain work. It **fans out** to existing skills, usually tasks.

Fixture `offboarding-batch` (IDO: `leaver-batch`) is the simple case: for each name, run `leaver-sweep`. Three names are three sibling goals. One name running out of budget does not cancel the next.

IDO `unit-review-packet` is the messy case: one department review plus recerts, audits, findings, leavers, a request — whatever the operator hung on the same sentence.

Scoring matches that shape. A composite contract is composed from child-task verdicts; it is not a ninth skill body with its own collection plan.

---

## Why two levels were not enough

The two-level story collapses three different questions into one word, “composite”:

| Question | Atomic | Task | Composite |
|---|---|---|---|
| What is this? | Shared means | One operator goal | A packet of goals |
| What does it call? | Commands only | Commands + atomics (or another task) | Child skills, usually tasks |
| How do you score it? | Parent’s goal; simple lookup if invoked alone | Its own contract | Composition of child contracts |
| How should it run? | One short procedure | One job | Several jobs, expanded separately |

The live failure that made this load-bearing was flattening a multi-skill operator message into **one prose todo list and one ReAct loop under one iteration budget**. A 30-step job and a 90-step job both died around the same command count. Three levels exist so expansion can be **deterministic**: composite → several tasks; each task → atomics and commands; each atomic → commands. Recursion here means that expansion (P-01), not a language model invoking itself as a tool.

The `uses:` graph may not run backwards up the tree. `LEVEL_RANK` is `composite` → `task` → `atomic`. A composite may use any level; a task may use task or atomic; an atomic is a leaf. Depth is capped at 3 (`MAX_DEPTH`).

---

## Intern cheat sheet

Think of a restaurant:

- **Atomic** = “how we plate a steak” (always the same motions).
- **Task** = “table 4 ordered the steak dinner” (one ticket, one success condition).
- **Composite** = “the wedding party’s tasting menu” (many tickets, same kitchen).

You expected composite + atomic because you were thinking about **call structure**. The third level exists because the system also cares about **operator intent, contracts, and budgets**. A task is allowed to use atomics and still be “one job,” not “a batch.”

---

## Naming traps

| Phrase | What it is | What it is not |
|---|---|---|
| `level: composite` on a SKILL.md | A skill that `uses:` child skills | A command that internally does several things |
| Frozen composite command | One command whose internals are frozen (crystallization) | A composite skill |
| Surface-walk / E2E walk | A test construct | An operator task or a skill level |

---

## See also

- Loader and validator: `fastworkflow/skill_catalog.py`
- Tiny three-level tree: `tests/fixtures/skills_workflow/_skills/`
- IDO requirements §4.6 and the IDO copy of this note: `docs/fastworkflow-skill-levels.md` in the IDO repo
- IDO content conformance: `tests/test_skill_conformance.py` (`Levels`)
