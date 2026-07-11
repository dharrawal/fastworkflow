# Experiment cards (E-cards) and pre-registration

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Source of the format: `docs/tau2_retail_reliability_implementation_plan.md` (untracked, team-private, Draft v2.0 dated 2026-07-09), §6–§7 and Appendix G. The plan defines 26 cards (E0–E25). This reference extracts the reusable format so new experiments — inside or outside the tau2 program — follow the same discipline.

## 1. The card format

Header line (plan §7, line 299):

> **ID. Name** | Effectiveness (E) toward the goal: High/Med/Low, **with mechanism** | Complexity (C): XS/S/M/L/XL (≈ person-days/weeks/months) | Dependencies | Parallelism

Body sections, in order (extracted from the E0–E25 cards):

| Section | Requirement |
|---|---|
| **Hypothesis** | Falsifiable, **numeric**, written before any run. Not "invariants help" but "convert ≥4 of the 10 post-mitigation failures" (E3), "recovers ≥1 task at zero design cost" (E1), "serves ≥95% of recall queries... no slower end-to-end" (E2). |
| **Design** | Concrete mechanism, priority-ordered where the card has multiple parts. Names the exact code surfaces (e.g. E3: "All implemented in `validate_extracted_parameters` / Pydantic validators / `workflow.context`. Tool layer untouched (benchmark parity)."). |
| **Metrics** | Must include the ways the idea could be **net-negative**. E3: "false-positive blocks on currently-correct writes (report prominently — a validator that blocks correct actions is worse than none)". E4: "turn count per task (watch for budget pressure)". |
| **Failure modes before → after** | What failure class disappears, and what NEW failure mode the change introduces (E4: "New failure mode to watch: turn-budget exhaustion (measure!)"). |
| **Contribution** | What is claimable externally if the hypothesis holds — ties the card to §9 Research Outputs. |
| **Dependencies / gate criteria** | Real ordering constraints. The serial spine (plan §8.3): repo sync before everything; E0 before any measured claim; E1 before E3; E3 before E20; E20 before production/MCP deployment; Gate-3 attempt once, at the end, on a frozen stack. |

## 2. Worked example — E3 (deterministic invariant layer), abridged

```
### E3. Deterministic invariant layer
E: High (attacks exec/planner 3/15 + part of deviation 5/15 + schema 2/15)
C: S–M (2–3 weeks incl. tests) | Depends: E1 (else invariants don't fire), E0
Parallel with: E2, E5, E6.

Hypothesis: deterministic pre-commit invariants convert >=4 of the 10
post-mitigation failures from wrong-writes into blocked-then-repaired
successes, and eliminate wrong-write VARIANCE on currently-passing tasks.

Design (priority order): M2+G4 attribute-delta guard and echo; CN3
whole-vs-part cancel guard; G1/G2 session auth + ownership; P1 ID-type
confusion guard; R3 coverage echo; remaining catalog (Appendix A).
All in validate_extracted_parameters / Pydantic validators /
workflow.context. Tool layer untouched (benchmark parity). Every
rejection returns machine-readable, actionable errors with valid
candidates — blocked without a repair path is still a failed task.

Metrics: invalid-write attempts blocked (per class); post-block recovery
rate; pass deltas; false-positive blocks (report prominently).

Failure modes: before — silent wrong writes. After — blocked +
candidate-bearing error; residual failure shifts to "agent failed to
repair," which E4/E7 address.

Contribution: quantifies deterministic-guard value in isolation (the
ablation the articles never had).
```

Note what makes this a *scientific* card and not a feature ticket: the predicted number (≥4 of 10), the named variance claim, the mandatory reporting of its own failure mode (false positives), the parity disclosure, and the residual-failure handoff to a sibling card.

## 3. E0 is the meta-card: no harness, no science

E0's hypothesis is about the methodology itself (plan §6): "with k≥5 runs per configuration and paired statistics, the true effect of each intervention is distinguishable from run-to-run noise — which single-run scores demonstrably are not (the 2/15 and 5/15 pass sets were disjoint)."

E0's components define the standing measurement bar for every later card:
1. Declarative runner: {configuration × task × k runs}, fully logged (every prompt, completion, tool call, memory op, token count, latency, per-action gold check).
2. Determinization: temperature 0 + fixed seeds for every role we control; **measure the residual provider nondeterminism** (k runs of an identical config) — this "irreducible variance floor... belongs in every report."
3. Metrics: pass@1, pass^k (k=3 and 5), per-action check pass rate, failure-bucket attribution, tokens/task/role, latency, cost, turn count.
4. Statistics module: Clopper-Pearson/Wilson CIs on all proportions; McNemar's exact test for paired before/after; a pre-registration template committed before the run.
5. Re-attribution protocol: every failed run classified by **two raters independently** (one may be an LLM; disagreements resolve by human adjudication), **blind** to configuration; report Cohen's κ; primary + contributing labels.
6. Gold-label adjudication list: contestable benchmark conventions annotated (never disputed in the score).
7. Full-split runs (~115 retail tasks) as mandatory context for any hard-set claim.

Status (verified 2026-07-09): E0 is unbuilt — no `eval/` dir, no `docs/experiments/` on main. Everything above is the *plan's* standard, adopted as this repo's methodology going forward; label it "planned" when citing externally.

## 4. Pre-registration and the staged gates

- **Per experiment:** Appendix G section (1) — hypothesis, configuration diff vs. ladder predecessor, metrics, analysis plan, date, "committed **before** the run."
- **Per program:** the staged gates are themselves pre-registered commitments (plan §3.3): Gate 1 = 15/15 pass@1 (k=5 measurement); Gate 2 = ≥13/15 pass^3 (the honest primary endpoint); Gate 3 = 15/15 pass^3, single-shot, estimated 25–40% in advance.
- **Anti-p-hacking rule (verbatim, plan §3.3):** "these 15 tasks have been our tuning set for months. Repeatedly running the pass^3 evaluation and patching between attempts until a sweep lands is test-set optimization. The credible protocol: freeze, pre-register, run once, report — and always pair with the full retail split (~115 tasks) so the 15 read as the stress subset, not the headline."
- **The most valuable claimable result** is pre-declared too — not "15/15" but: "the gap between our pass@1 and pass^3 is attributable almost entirely to simulator stochasticity rather than agent stochasticity."

## 5. The integration-queue rule (ablation by construction)

Plan §8.4, the rule that prevents the Article-3 mistake structurally:

> Development branches merge into the measured stack **one experiment at a time, in a declared order, with a paired k=5 before/after evaluation (McNemar) at each step.** The baseline ladder B2, B3, … *is* the ablation table. If two experiments interact, the interaction is measured by the ladder order and, where suspicion warrants, one extra leave-one-out run. **Nothing enters the stack unmeasured; nothing is measured while entangled.**

## 6. The experiment document (Appendix G, verbatim standard)

Each `docs/experiments/EXX_<name>.md` MUST contain, in order:

1. **Pre-registration** — hypothesis, configuration diff vs. ladder predecessor, metrics, analysis plan, date, committed BEFORE the run
2. **Baseline numbers with CIs**
3. **Results with CIs + McNemar vs. predecessor**
4. **Failure modes before/after** (bucket table)
5. **Surprises / negative results** (negative results are results)
6. **Links to run logs + traces**
7. **bd issue ID**
8. **Reproduction command line**

Rule: "if it isn't written here with a number and an interval, it didn't happen."

Lint a draft with the skill helper:

```bash
python3 .claude/skills/fastworkflow-research-methodology/scripts/lint_experiment_doc.py docs/experiments/E3_invariants.md
```

## 7. Variance-aware promotion (the RSI extension of the bar)

When experiments are proposed by an automated loop (E21), the evidence bar tightens rather than relaxes (`docs/rsi_harness_agent_report.md`):

- Promotion objective is **pass^k improvement with variance accounting**, not pass@1: a candidate wins only if (a) it flips ≥1 target-cluster task to consistently-passing across all stage-3 runs, (b) no previously-consistent task becomes inconsistent, (c) the held-out split is statistically no-worse, (d) audits are clean.
- Evaluation is a 5-stage cascade (static → OPE replay → targeted k=2 → full paired k=5 McNemar+SPRT → nightly held-out) — stages 0–2 are useful standalone as pre-integration checks for *human* patches too.
- Three firewalls: evaluator/gates/promotion rules out-of-loop; held-out split in every promotion; the Gate-3 run permanently invisible to the loop.
- A kill criterion is stated in advance (~4 weeks of plateau below Gate 2 → stop).

## Re-verification

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). The plan and RSI report are untracked; diff-check them before trusting line numbers.

```bash
grep -n "^##\|^###" docs/tau2_retail_reliability_implementation_plan.md   # section map
sed -n '274,296p' docs/tau2_retail_reliability_implementation_plan.md     # E0 card
sed -n '343,362p' docs/tau2_retail_reliability_implementation_plan.md     # E3 card
sed -n '793,797p' docs/tau2_retail_reliability_implementation_plan.md     # integration queue
sed -n '943,946p' docs/tau2_retail_reliability_implementation_plan.md     # Appendix G
grep -n "Promote on\|cascade\|kill criterion\|firewall" docs/rsi_harness_agent_report.md
ls eval docs/experiments 2>&1                                             # still absent = E0 not started
```
