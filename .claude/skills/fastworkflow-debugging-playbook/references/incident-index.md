# Incident index — the stories behind the traps

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

**The canonical chronicle of these incidents is the sibling skill `fastworkflow-failure-archaeology`**
(its SKILL.md numbered entries + `references/`) — full symptom → root cause → fix → status narratives,
plus all dead ends and abandoned features, live there and are maintained there. This index exists for
*trap triage only*: it maps each trap ID from `SKILL.md` to its incident. Incidents already chronicled
in archaeology get one routing row (plus any playbook-specific delta); full entries appear below only
for incidents the archaeology chronicle does not cover. Verify commits with `git show <sha> --stat`.

## Routing table — incidents chronicled in fastworkflow-failure-archaeology

| Trap(s) | Incident | Commit / issue | Canonical entry | Playbook-specific triage delta |
|---|---|---|---|---|
| T1 | Stale command caches (mtime check blind to deletions/renames → fingerprint invalidation) | `b5747df` (v2.22.1) | failure-archaeology entry 7 | Fixed on disk only; the in-process `lru_cache`/registry staleness (T1b) remains open by design — the fingerprint is checked only on cache miss. |
| T2, T13 | The self-destructing test suite (fixture trained against + `rmtree`'d the REAL bundled hello_world model) | fix-0hb, `fa97b48` (2026-06-15) | failure-archaeology entry 5 | Derived non-negotiable: tests/experiments never mutate bundled examples' trained models — train into a temp copy. |
| T6 | The 504-retry double execution (timeout unwound the lock while the orphaned executor kept mutating the context) | fix-85g Step 1, `cf3eeae` (v2.22.0) | failure-archaeology entry 3 | Steps 2–3 OPEN (`bd show fix-85g`); `/invoke_agent_stream` still uses the old `lock.locked()` pattern (`run_fastapi_mcp/__main__.py:963`). |
| T10 | speedict hot-path removal and the same-day WeakValueDictionary leak fix | `6880b0a` + `033132f` (v2.21.4/.5, 2026-06-15) | failure-archaeology entry 6 | Closed (fix-2jo/fix-04r); full speedict excision is open epic fix-7kp — fix-7kp and closed fix-2jo were created 9 seconds apart on 2026-06-16 with byte-identical titles, likely collateral of the bd flakiness first observed 2026-06-11; unadjudicated — reconcile with Dhar before citing fix-7kp. |
| T11 | Import-time env read (`SPEEDDICT_FOLDERNAME` read at module import, before `fastworkflow.init()` loaded the env file; masked in-repo by litellm's `load_dotenv`) | `79e6986` (v2.21.3) | failure-archaeology entry 6 (v2.21.3 item) | Rule: never call `get_env_var` at import time. |

## Full entries — incidents NOT in the archaeology chronicle

### transformers 5.x broke training — commit d9511a5 (v2.18.0) — train-phase context for T2/T8

- **Symptom:** `fastworkflow train` failed loading the default tiny model.
- **Root cause:** old default `prajjwal1/bert-tiny` shipped neither `model_type` nor a fast tokenizer;
  transformers 5.x removed both fallback paths.
- **Fix:** default swapped to `google/bert_uncased_L-4_H-128_A-2` (same WordPiece vocab); base models became
  env-configurable via `INTENT_DETECTION_TINY_MODEL` / `INTENT_DETECTION_LARGE_MODEL` (env-file only — code
  defaults shadow shell exports, trap T11).

### Untrained-workflow crash-on-first-command — commit d7853a1 (v2.19.0) — trap T2

- **Symptom:** starting a chat on a never-trained workflow crashed mid-conversation inside `CommandRouter`
  opening the missing `___command_info/global/threshold.json`.
- **Fix:** `is_workflow_trained()` filesystem pre-check fails fast at session start in `run` and `mcp_server`,
  printing the exact train command. Paths that embed WEC directly still lack the guard.

### PyJWT empty-key landmine — commit 5a63790 (v2.21.2) — trap T12

- **Symptom:** trusted-network "unsigned" tokens failed with `HMAC key must not be empty` after the
  python-jose → PyJWT security migration.
- **Fix:** unsigned mode now uses JWT `alg="none"` (`run_fastapi_mcp/jwt_manager.py:203,252`); verification skips
  signatures and checks only expiry/type. Found while validating the suite for an unrelated release.
- **Consequence to remember:** the default server posture accepts forgeable tokens; `--expect_encrypted_jwt`
  exists only on `python -m fastworkflow.run_fastapi_mcp` (the CLI wrapper never forwards it).

### The predicted CLI ask_user hang — fix-5fv (OPEN, filed 2026-06-11) — trap T5

(Listed as review collateral in failure-archaeology entry 2; the triage detail lives here.)

- **Origin:** found by *code reading* during the turn-result design review (finding R43), not by a runtime repro.
- **Hypothesis:** Topology A blocking `ask_user` enqueues the clarification `CommandOutput` without a trace
  sentinel; the CLI drains the trace queue for a `None` sentinel before reading the output queue → spinner
  deadlock; the question never prints.
- **Status:** OPEN, explicitly flagged "NEEDS RUNTIME VERIFICATION". If you reproduce (or fail to reproduce) it,
  record the runtime evidence in the bead (one bd write, verify `.beads/issues.jsonl` afterward).

---

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verify with:
`git log --oneline | grep -E "^(b5747df|fa97b48|cf3eeae|6880b0a|033132f|79e6986|d9511a5|d7853a1|5a63790)"`
and `bd show fix-85g fix-5fv fix-7kp fix-5ka`.
