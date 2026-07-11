# Case study: the fix-vof / TurnResult review pipeline — the change-control angle

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

This file covers ONLY what fix-vof teaches about gating: when to classify a change T2, and
how to argue the cost/benefit. The full material lives in sibling skills — do not restate
it here:

- **Reusable procedure** (the seven stages, review mechanics, commit shapes, scaling the
  template down): `fastworkflow-research-methodology/references/adversarial-review-runbook.md`.
- **Historical chronicle** (timeline, findings by name — R37/R1/X1/X3/X6 — the Topic-5
  near-miss, what shipped as v2.21.0–v2.21.2, v3.0 roadmap status):
  `fastworkflow-failure-archaeology/references/turn-result-saga.md`.

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

## Why this change was gated T2 (the trigger analysis)

Symptom: in agent mode, the xray example's `POST /invoke` always returned `payload=None`.
A T0 patch could have hacked artifacts through the agent boundary. Dhar classified it T2
because the real defect was the **shape of the per-turn record** — a system-of-record
question that everything downstream (persistence, projections, wire contract, metrics)
would sit on. This is the worked example behind SKILL.md §3's gate triggers: the deciding
question is not the patch size, it is how many future decisions will sit on the answer.

## The finding-resolution contract

Each of the 48 findings was resolved by ONE docs commit updating three artifacts in
lockstep: `docs/turn_result_design.md` (+ amendment), `docs/turn_result_design_review.md`
(finding marked resolved), `.beads/issues.jsonl` (child closed). That triple-update IS the
change-control contract for T2 findings. The verified commit format is in SKILL.md §3
item 4; the full step-by-step is adversarial-review-runbook Step 4.
Re-verify: `git show 42e9b3a --stat --format='%s%n%b'`.

## Cost/benefit for future gate decisions

- Cost: ~2 days of review (2026-06-10/11), 48 + 12 findings, ~230 KB of docs, ~55 commits.
- Benefit: at least 3 shipped-quality catches (X1, X3, the feedback-session near-miss)
  plus one unrelated open bug (fix-5fv) found by reading, not running. Each layer caught a
  class of error the previous layer was structurally blind to — details per catch:
  turn-result-saga.md "Findings worth knowing by name" and its near-miss section.
- Use this ratio when arguing whether a change is T1 or T2: if a wrong decision would be
  *silent* and *load-bearing* (data loss, wire shape, identity), the review pays for
  itself. If a wrong decision fails loudly in tests, T0/T1 suffices.

## Scope honesty (what a T2 "done" looks like)

fix-vof was review-only; the implementation shipped separately as v2.21.0–v2.21.2 under a
*different* epic, `fix-yy1` — still OPEN (SKILL.md §4 stale-epic table; owner adjudication
required, do not close it yourself). The entire designed persistence layer is v3.0 roadmap
per `turn_result_design_final.md` §14–15 and is **unbuilt on main** — no
`fastworkflow/stores/` exists. A T2 process that ends in a minimal slice plus a
ready-to-implement spec is a *success*, not an unfinished failure.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

- Epic + children status: `bd show fix-vof --json`
- Resolution-commit shape: `git show 42e9b3a --stat --format='%s%n%b'`
- Review window: `git show -s --format='%ci %s' fe3981d 920d0de` (2026-06-10 → 2026-06-11)
- fix-5fv still open: `bd show fix-5fv --json`
- v3.0 layer still unbuilt: `ls fastworkflow/stores 2>&1` (expect: No such file or directory)
