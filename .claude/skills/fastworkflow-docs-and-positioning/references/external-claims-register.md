# External-claims register

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Every public-facing claim the project currently makes, its backing, and its status.
Verified 2026-07-09 against v2.22.2 (commit c33b9a5). For the statistical evidence bar
that a NEW claim must clear, see `fastworkflow-research-methodology`; for what pass@1 /
pass^k mean and how tau-bench runs work, see `fastworkflow-taubench-reference`.

## Status legend

- **BACKED** — peer-reviewed or in-repo evidence exists; may be repeated with citation
- **PUBLISHED-BUT-UNDOCUMENTED** — already public, but provenance is not in this repo;
  do not extend or re-derive from it
- **TEAM-PRIVATE** — must not appear in any public artifact until Dhar approves
- **OPEN DEFECT** — known error awaiting correction; never propagate

## The register

| # | Claim | Where it appears | Status | Notes |
|---|---|---|---|---|
| 1 | "Build AI agents your application can actually trust in production"; small models punch above their weight AND frontier models get more reliable | README.md:6-12, failure-modes table :69-74 | BACKED (as framing) | Marketing framing of claim 3; keep it qualitative |
| 2 | CAIS '26 paper: Satija, Bhatt, Jani, Rawal 2026, *fastWorkflow: Closing the Performance Gap Between Small and Frontier Language Models for Conversational Agents* | README.md:101 | BACKED | THE citable artifact. DOI dispute below (claim 5) |
| 3 | Tau Bench Retail Pass@1: fastWorkflow Mistral Small 24B 86.95% (also Mistral 7B 57.40%, Qwen3 14B 74.00%, GPT-OSS 20B 80.20%) vs leaderboard (Claude Opus 4 80.50% … Claude-Sonnet-4.5 86.20%) | `fastWorkflow - Tau Bench Retail.jpg` embedded at README.md:88 | PUBLISHED-BUT-UNDOCUMENTED | No in-repo run counts, harness commit, config, or task-subset record; leaderboard-baseline provenance (re-run vs quoted) unrecorded. The paper is the only backing. Do NOT derive new claims from the chart |
| 4 | Tau Bench Airline Pass@1: fastWorkflow GPT-OSS 20B 96.00%, Mistral Small 24B 90.00%, Mistral 7B 82.00%, Qwen3 14B 80.00% — all four beat every leaderboard entry (best: Gemini 3.0 Pro 73.00%) | `fastWorkflow - TauBench Airline.jpg` at README.md:92; "matches frontier models" at :99 | PUBLISHED-BUT-UNDOCUMENTED | Same gap as claim 3 |
| 5 | The paper's DOI | README.md:101 + Article 1 PDF: `10.1145/3786335.3813158`; Article 2 & 3 PDFs: `10.1145/3738331.3738402` | OPEN DEFECT | Verified by direct PDF text extraction 2026-07-09. tau2 plan Appendix E (line 933): "verify the single correct ACM DOI across all three articles". Until resolved: cite the README form, flag the dispute, resolve before any new publication |
| 6 | RLM ("Recursive Language Models") citation in the Article PDFs: arXiv 2512.24601 | Article PDFs | OPEN DEFECT | Correct ID is arXiv 2510.04871 (tau2 plan line 931: "the articles' 2512.24601 citation was erroneous and is being fixed") |
| 7 | RLM memory-layer results: 15 enriched τ²-retail tasks, baseline 2/15 → 5/15 post-mitigation; the 2 original passes NOT among the 5; ~10 min/task | Article PDFs, tau2 plan | TEAM-PRIVATE | Also methodologically fragile per the plan's own critical review (unmeasured pre-memory baseline, bundled mitigations, disjoint pass sets = variance). Raw traces are already public (github.com/Programiz-007/fastworkflow_RLM_Traces) but the analysis is not. E19 (publish fork + artifacts) is the path to citability |
| 8 | Everything in the tau2 plan / RSI report / Forge spec (E0–E25, Option N v2, pass^3 gates, "RSI for reliability" novelty claim) | untracked docs/ | TEAM-PRIVATE | The novelty claim ("no treatment of determinism/variance/pass^k in Weng 2026's cited RSI systems") is a literature-gap assertion, not yet a result |
| 9 | Any pass^k / reliability number for fastWorkflow | nowhere yet | (none exists) | E0 harness is unbuilt (no `eval/`); until it exists no reliability number may be stated, even internally-measured ones |
| 10 | "1-shot adaptation from intent-detection mistakes… corrections can be persisted" | README.md:373 | BACKED (mechanism), unaudited (extent) | Implemented: `fastworkflow/cache_matching.py:53 store_utterance_cache` persists utterance→label corrections with frequency; consumed by the CME intent pipeline. Test coverage of the cross-session persistence claim unaudited — don't elaborate beyond the README sentence |
| 11 | "Your application code is never modified" | README.md:61, 186, 407 | BACKED | Architecture invariant AND marketing promise — breaking it is a positioning event, not just a code change |
| 12 | integrate-chat-agent skill is "the fastest path for a non-trivial app" | README.md:179, 407 | BACKED (as recommendation) | A deliberate June 2026 repositioning (commit 1d3a6aa) of hand-authored commands over `fastworkflow build`. Whether Forge supersedes both is an open strategy question |

## Rules for adding a claim

1. **A new benchmark number requires, committed in-repo BEFORE the claim ships:** run
   count k with per-run results; exact harness fork commit + config; task subset
   (full split vs enriched — the 15-task hard set is adversarially enriched, never
   representative); statistical treatment (CIs, paired tests) per
   `fastworkflow-research-methodology`. The claim-3/4 gap is the anti-pattern: fix it
   for new claims, do not repeat it.
2. **Benchmark parity is sacred.** Never modify tau-bench/τ²-bench tools or tasks for a
   benchmark run. Any nonstandard trade — pinning user-simulator temperature or seed,
   task filtering, retry policies — must be DISCLOSED in the claim's provenance record,
   never silent. (The user simulator is a sampled LLM; per the tau2 plan, even 2%
   sim-induced failure per task-run caps a perfect agent's Gate-3 sweep odds near 40% —
   which is why the temptation to pin it exists and why hiding it would be fatal to
   credibility.)
3. **Enriched subsets must be labeled.** "5/15 on hand-selected hard tasks" and "X% on
   the full retail split" are different universes; conflating them in public is an
   integrity failure.
4. **Runtime model of record vs dev-time tooling separation** (from the articles and
   the RSI report): benchmark runs use one runtime model for all roles (gpt-oss-20b on
   Groq in the failure series) as a deliberate confound control; frontier models may be
   used at dev time (proposers, task generators). Every report must state the
   separation.
5. **The confidentiality boundary is per-turn explicit consent.** The repo is public;
   untracked ≠ safe (the 2026-07-08 incident was a private planning doc auto-committed
   and pushed, requiring a history rewrite). Never commit or push anything without
   the developer'sexplicit request in that turn. New strategy docs get a team-private status
   line at the top on day one.

## Re-verification one-liners

| Volatile fact | Command |
|---|---|
| README citation + DOI | `grep -n "doi.org" README.md` |
| PDF DOIs still discrepant | `python3` zlib-extraction one-liner in SKILL.md Provenance |
| JPG provenance still missing | `grep -in -e 'pass@1' -e trial -e harness README.md` (expect no provenance hits) |
| Plan errata still open | `grep -n -e "verify the single correct ACM DOI" -e 2512.24601 docs/tau2_retail_reliability_implementation_plan.md` |
| Trace repo still the only public RLM artifact | `grep -rn "recall_from_memory" fastworkflow/` (expect empty) |
| E0 harness existence | `ls eval/ docs/experiments/ 2>&1` |
| cache_matching backing for claim 10 | `grep -n "def store_utterance_cache" fastworkflow/cache_matching.py` |
