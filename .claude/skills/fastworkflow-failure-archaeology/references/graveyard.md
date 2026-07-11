# The graveyard — abandoned features and dead code, in detail

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Companion to SKILL.md entry 12. Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).
Purpose: stop two failure modes — (a) resurrecting a ghost because a spec/branch/file makes it
look alive, and (b) deleting a "harmless orphan" that is actually a live compatibility shim.
When in doubt, run the re-verification command and check with Dhar before deleting anything.

## 1. Command dependency graph (built v2.10–v2.13, excised v2.15.8–v2.17.19)

- **What it was:** "a GenAI based backward chaining engine" to help agents find commands that
  could provide missing parameter values. Implemented in v2.10.0 (55a8668, alongside workflow
  inheritance); the v2.13.0 refine tool (ce5ada5) built a dependency-graph JSON from
  input/output parameter names and descriptions.
- **Death:** never adopted by the agent. v2.15.8 (ed21d77) "Removed outdated tests related to
  command dependency graph"; v2.17.19 (ad68d9b) "Removed the sentencetransformers package that
  was introduced when we built the now unused command dependency graph functionality".
- **Why:** superseded by a simpler approach — injecting command info into the agent's system
  prompt (v2.17.6 5f0e66b, v2.17.13 40b598d, per git archaeology).
- **Ghosts still on main:** `fastworkflow/build/dependency_manager.py` and
  `fastworkflow/build/command_dependency_resolver.py` exist with ZERO callers outside
  themselves (verified by grep). `tests/__pycache__/` still holds a
  `test_command_dependency_graph.*.pyc` ghost of the deleted test.
- **If tempted to revive:** the underlying need ("agent can't find the command that produces a
  missing parameter") is real; the settled answer is prompt injection + the NOT_FOUND
  ask-don't-guess convention. Bring evidence the settled answer fails before rebuilding.
- Re-verify: `grep -rn "dependency_manager\|resolve_command_dependencies" fastworkflow/ --include=*.py | grep -v "build/dependency_manager.py\|build/command_dependency_resolver.py"` (expect empty).

## 2. The one-week standalone FastAPI service (v2.16.0)

- **What it was:** v2.16.0 (447db70, 2025-10-11) shipped `services/run_fastapi` — a 1092-line
  main.py, multi-user, conversation/trace storage, feedback recording, SSE streaming.
- **Death:** seven days later, v2.17.0 (299739e, 2025-10-18) replaced it wholesale with
  `fastworkflow/run_fastapi_mcp` ("automagically an MCP server"). `services/` no longer exists.
- **Why:** the MCP-integrated design subsumed it; the v2.17 line then absorbed 35 patch
  releases (JWT, channels, LiteLLM proxy, k8s probes, docker slimming) and became the product.
- **Ghosts:** `run_fastapi_mcp/README.md` still instructs `uvicorn services.run_fastapi.main:app`,
  and `redoc_2_standalone_html.py` imports `services.run_fastapi.main` — both reference the
  deleted layout. The real entry point is `fastworkflow run_fastapi_mcp ...` (module
  `fastworkflow.run_fastapi_mcp.__main__`).
- Re-verify: `ls services 2>&1` (expect "No such file or directory"); `grep -rn "services.run_fastapi" fastworkflow/ *.py 2>/dev/null`.

## 3. Probabilistic response generation (branch only; spec on main)

- **What it was:** branch `origin/cursor/implement-probabilistic-response-generation-and-pass-tests-7586`
  (7c7d1cf "Add probabilistic response generation framework and build hook", a4d3d4d lazy-import
  stubs; forked from ddab25c, Aug 2025).
- **State:** never merged. `fastworkflow/probabilistic_response.py` does not exist on main.
- **Trap:** the spec `docs/probabilistic_response_generation.md` IS on main (landed in ddab25c
  "specs for new functionality" — spec docs can precede and outlive unimplemented features).
- **Why abandoned:** unknown (no beads issue, no revert commit, no discussion found in-repo).
  Candidate explanations — priority shift or supersession by agent-mode response generation —
  are unverified. Ask Dhar before treating the spec as roadmap.
- Re-verify: `ls fastworkflow/probabilistic_response.py 2>&1`; `git branch -r --merged main | grep probabilistic` (expect empty).

## 4. Rich CLI (branch only)

- **What it was:** branch `cursor/enhance-fastworkflow-cli-aesthetics-1752` (4d2d9a7 "Replace
  colorama with rich for enhanced CLI output styling" + c4cf167 background-composer dump;
  forked from v2.5.2 f841075, 2025-06-22/23).
- **State:** never merged; main still depends on colorama (pyproject.toml:49) and
  `fastworkflow/cli.py` uses it.
- **Why abandoned:** unknown.
- Re-verify: `grep -n colorama pyproject.toml`.

## 5. 43-agents cloud-agent experiment (branches only)

- **What it was:** branches `origin/cursor/new-cloud-agent-1d34` and `-af26` (Feb 2026, forked
  from v2.17.33 51cf15e): 2a0214f "feat: Add agent categorization system with 43 agents in 4
  collections", plus heavy beads/Dolt database churn, authored by Cursor Agent.
- **State:** never merged; nothing of it exists on main. (The .beads Dolt-remote setup landed
  on main separately.)
- **Why abandoned:** unknown — an exploratory Cursor cloud-agent experiment.
- Re-verify: `git branch -r --merged main | grep new-cloud-agent` (expect empty).

## 6. `majority_vote_predictions` (dead code, revival blocked on temperature)

- **Where:** `fastworkflow/_workflows/command_metadata_extraction/intent_detection.py:304`;
  sole call site commented out at line 122.
- **What it was:** run 5 parallel intent predictions and majority-vote to reduce prediction
  variance.
- **Why dead:** its own TODOs (lines 302–303): "generation is deterministic. They all return
  the same answer" / "Need 'temperature' for intent detection pipeline". Ensemble voting over
  a deterministic pipeline is a no-op.
- **Status: open question** — revive once temperature support lands (relevant to the tau2
  reliability program's variance work) or delete. Neither has been decided; do not do either
  silently.
- Re-verify: `grep -n majority_vote fastworkflow/_workflows/command_metadata_extraction/intent_detection.py`.

## 7. `ambiguous_threshold.json` legacy files (harmless orphans)

- **Where:** trained `___command_info` folders, e.g.
  `fastworkflow/_workflows/command_metadata_extraction/___command_info/{global,IntentDetection}/ambiguous_threshold.json`
  and `fastworkflow/examples/simple_workflow_template/___command_info/*/ambiguous_threshold.json`.
- **Why they exist:** a rename split the artifact into `tiny_ambiguous_threshold.json` and
  `large_ambiguous_threshold.json`; the loader reads only the tiny_/large_ variants
  (`fastworkflow/model_pipeline_training.py:299-300`) and the writer writes only those
  (:1179, :1184-ish). The unprefixed files are pre-rename leftovers in committed/trained dirs.
- **Status:** harmless; not proof of a live code path. Safe to ignore. If cleaning up, note
  most `___command_info` dirs are gitignored dev-local state — only a few (CME, some examples)
  are checked in.
- Re-verify: `find fastworkflow -name ambiguous_threshold.json`; `grep -rn '"ambiguous_threshold.json"\|/ambiguous_threshold' fastworkflow/*.py` (expect only tiny_/large_ prefixed hits).

## 8. Honorable mentions (documented elsewhere, listed so you don't re-discover them)

- **CLI cross-restart context resume** — deliberately dropped in v2.21.4 (6880b0a), an
  accepted trade-off, not a bug (SKILL.md entry 6).
- **`_extract_missing_fields` in
  `fastworkflow/_workflows/command_metadata_extraction/parameter_extraction.py:140-160`** — zero callers and
  would raise ValueError if revived (unpacks a 4-tuple into 3); a large commented-out merge
  block nearby suggests an abandoned richer merge strategy. Unverified beyond static reading.
- **ChatSessionDescriptor** (`fastworkflow/__init__.py`, ~lines 151–164) — a set-once guard
  that has never worked (descriptors don't fire as module attributes, and a later submodule
  import rebinds the name). Latent, no beads issue. Pass `chat_session_obj`/WEC explicitly
  instead of relying on `fastworkflow.chat_session`.
- **Task Master residue** — `.windsurfrules`, `.roo/rules`, `.taskmaster/` still describe the
  pre-beads tracker; beads (bd) is the only sanctioned tracker (AGENTS.md).
- **Deleted-test ghosts** — `tests/__pycache__` holds .pyc files for removed tests
  (`test_agent_integration`, `test_ambient_credentials_apikey`,
  `test_command_dependency_graph`); the "never remove pytest tests without approval" rule
  exists because tests HAVE been removed.

## Rules of thumb distilled from this graveyard

1. **A spec doc in `docs/` is not proof the feature exists** (items 3, and TurnResult v3.0).
2. **A branch is not roadmap** — four of five abandoned branches have no recorded
   why-abandoned; ask before building on them.
3. **"Removed" features can leave live-looking modules** (item 1) and stale READMEs (item 2);
   grep for callers before trusting a file's existence.
4. **Deletion is change control too** — dead code here is catalogued, not condemned; several
   items are open questions (item 6) or deliberate trade-offs (CLI resume).
