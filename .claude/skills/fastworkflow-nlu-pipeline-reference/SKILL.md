---
name: fastworkflow-nlu-pipeline-reference
description: >
  Domain reference for fastWorkflow's NLU stack AS IMPLEMENTED IN THIS REPO: two-tier
  TinyBERT/DistilBERT intent classifiers, confidence thresholds, synthetic utterance
  generation (litellm + PersonaHub), DSPy parameter extraction, runtime matching layers
  (fuzzy/embedding-cache/classifier), and per-role litellm model routing. Load this when
  you hear: "wrong command predicted", "ambiguous intent", "intent detection", "parameter
  extraction returned NOT_FOUND", "threshold.json", "confidence threshold", "utterance
  generation", "DSPy signature/cache", "LabeledFewShot", "litellm_proxy", "why did the
  classifier pick X", or when modifying anything under model_pipeline_training.py,
  intent_detection.py, signatures.py, generate_synthetic.py, or cache_matching.py.
  Do NOT load for step-by-step failure triage (use fastworkflow-debugging-playbook),
  the full env-var catalog (fastworkflow-config-and-flags), or tau-bench benchmark
  mechanics (fastworkflow-taubench-reference).
---

# fastWorkflow NLU Pipeline Reference

Everything below is verified against source at v2.22.2 (commit c33b9a5), 2026-07-09.
File paths are repo-relative. **Trust this document over CLAUDE.md and README where they
disagree** — known doc rot is listed at the end.

## When to use / when NOT to use

| You need... | Use |
|---|---|
| How intent detection / parameter extraction actually work here; what a threshold means; where a magic number lives | **This skill** |
| A symptom-to-fix triage table for a live failure | `fastworkflow-debugging-playbook` |
| Every env var with defaults and consumers | `fastworkflow-config-and-flags` |
| tau-bench / tau2-bench harness, pass^k, simulator mechanics | `fastworkflow-taubench-reference` |
| Why the architecture is shaped this way; invariants | `fastworkflow-architecture-contract` |
| Running train/build/CLI commands operationally | `fastworkflow-run-and-operate` |
| Measuring model quality instead of eyeballing | `fastworkflow-diagnostics-and-tooling` |
| Variance/pass^k math, calibration analysis recipes | `fastworkflow-proof-and-analysis-toolkit` |

## Glossary (one line each, used throughout)

| Term | Meaning here |
|---|---|
| fine-tuning | Continuing training of a pretrained model's weights on your small labeled dataset |
| logits | Raw per-class scores from the classifier head, before normalization |
| softmax confidence | `max(softmax(logits))` — the top class's probability; this repo's only "confidence" signal |
| calibration | How well confidence tracks actual correctness. This repo tunes *decision thresholds* on held-out data but never calibrates the probabilities themselves (no temperature scaling) |
| embedding | Fixed-length vector representing text; here the DistilBERT `[CLS]` token's last hidden state (`cache_matching.py:51`) |
| cosine similarity | Angle-based vector similarity in [~-1, 1]; 1.0 = same direction |
| Levenshtein distance | Minimum single-character edits between strings; "normalized" = divided by the longer length, so 0.0 = identical |
| LabelEncoder | sklearn utility mapping label strings ↔ integer class ids (the ONLY load-bearing sklearn use in intent detection) |
| weighted F1 | Per-class harmonic mean of precision/recall, averaged weighted by class frequency |
| NDCG@3 | Ranking score: 1.0 if the true label is ranked 1st, discounted by 1/log2(rank+1) if 2nd/3rd, 0 if absent from top-3 |
| DSPy | Framework that compiles typed "Signatures" (input/output field specs) into LLM prompts via modules like `ChainOfThought` |
| ChainOfThought | DSPy module that makes the LLM emit reasoning before the output fields |
| LabeledFewShot | DSPy optimizer that stuffs labeled examples into the prompt (few-shot = examples-in-prompt) |
| litellm | Client library exposing one API over many LLM providers via `provider/model` strings |
| NOT_FOUND | String sentinel (env var `NOT_FOUND`, value `"NOT_FOUND"`) marking an unextracted parameter |

## The 30-second mental model

Every user turn in deterministic mode runs a pseudo-command called `wildcard` on an
internal "command_metadata_extraction" (CME) workflow. That one command
(`fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py`) drives the
whole NLU state machine:

```
user text
  └─> Stage 1  Intent detection   (intent_detection.py — 4-layer matching ladder)
  └─> Stage 2  Parameter extraction (parameter_extraction.py + utils/signatures.py — regex/DSPy)
  └─> Stage 3  Command execution  (your command's ResponseGenerator — outside NLU)
```

Any gap (ambiguous intent, missing/invalid parameter) produces a **clarification turn**,
never a guess — that is the framework's core reliability contract.

## 1. Intent detection: the 4-layer matching ladder

Order, from `_workflows/command_metadata_extraction/intent_detection.py` (`predict`, :36-167):

| # | Layer | Mechanism | Threshold | Code |
|---|---|---|---|---|
| 1 | Exact first-token | First whitespace/`(`-delimited token, lowercased, looked up in valid command names | exact | :100-105 |
| 2 | Fuzzy match | Normalized Levenshtein **distance** vs command names (input truncated-prefix compare) | distance ≤ 0.3 | :108-114 → `utils/fuzzy_match.py:17` |
| 3 | Embedding cache | Cosine similarity vs previously-clarified utterances in `<app_workflow>/___convo_info/cache.db` | ≥ 0.85 | :118 → `cache_matching.py:131` |
| 4 | Transformer classifier | `CommandRouter.predict` (two-tier BERT, next section) | trained thresholds | :121 |

Sharp edges, all verified:

- Layer 2's threshold is a **distance** (lower = closer), not a similarity. `find_best_matches`
  also compares against each candidate *truncated to the input's length* (`fuzzy_match.py:49`)
  — effectively prefix-biased matching.
- Layer 3: the `cache_match` function *default* is 0.90 (`cache_matching.py:131`) but the only
  call site passes **0.85** (`intent_detection.py:118`). The 0.90 default is never used.
- Layer 3 is an O(n) linear scan opening the RocksDB (`speedict.Rdict`) per message, one
  `cosine_similarity` call per cached entry (`cache_matching.py:168-179`). Embeddings are
  memoized in-process via `lru_cache(256)` keyed on `(id(pipeline), text)` (:18-27).
- **Learning loop**: after a successful ambiguity/misunderstanding clarification, the ORIGINAL
  utterance (from `cme_workflow.context["command"]`) is stored with the resolved label and its
  DistilBERT embedding (`intent_detection.py:151-162` → `store_utterance_cache`), so the next
  similar utterance short-circuits at layer 3. Multiple labels per utterance are resolved by
  highest frequency, tie-broken by most recent feedback date (`cache_matching.py:189-208`).
  This is the implementation behind README's "1-shot adaptation" claim.
- Special commands bypass the classifier entirely: `ErrorCorrection/abort` and
  `ErrorCorrection/you_misunderstood` (and `what_can_i_do` during ambiguity clarification) are
  matched only via layers 1-2 against their `plain_utterances` (:69-97).
- `majority_vote_predictions` (:304-360) is **dead code** — the call is commented out at :122
  and its own TODOs (:302-303) admit predictions are deterministic, so voting is pointless
  without a temperature mechanism. Status: stalled reliability idea, relevant to tau2 work.

## 2. Two-tier intent models (TinyBERT + DistilBERT)

**Not sklearn.** Despite CLAUDE.md's wording, intent models are two HuggingFace
`AutoModelForSequenceClassification` checkpoints fine-tuned with torch
(`fastworkflow/model_pipeline_training.py` — note: repo root of the package, NOT under
`train/`). sklearn contributes only `LabelEncoder`, `train_test_split`, `f1_score`
(the `PCA` import at :5 is unused).

| | Tier 1 "tiny" | Tier 2 "large" |
|---|---|---|
| Default checkpoint | `google/bert_uncased_L-4_H-128_A-2` | `distilbert-base-uncased` |
| Env override | `INTENT_DETECTION_TINY_MODEL` | `INTENT_DETECTION_LARGE_MODEL` (:892-895; env-FILE only — see §6 quirk) |
| Optimizer / LR | AdamW, 1e-4 (:956) | AdamW, 5e-5 (:1019) |
| Epochs | 12 (:957) | 5 (:1020) |
| Batch size | 10 (:936) | 10 |
| Max token length | 128 (train and inference) | 128 |
| Split | `train_test_split(test_size=0.25, random_state=42)` (:914) | same data |

One model **pair per command context**, trained by `train()` (:803). Contexts trained =
workflow's contexts minus the internal CME contexts, plus `'*'` (mapped to folder `global`,
:604, :820-823). Labels per context = that context's commands + core commands, PLUS a
`wildcard` out-of-scope class built from ancestor-context utterances (minus the context's
own) plus the `wildcard` command's deliberately-junky seed utterances (:869-877).
Commands without `Signature.Input` are excluded from training entirely (:741-757) — they
are `perform_action`-only. Labels are fully-qualified (`Context/command`); the runtime
takes `.split('/')[-1]` (`intent_detection.py:125`).

**Runtime tiering** (`ModelPipeline.predict_batch` :416-508, `CommandRouter.predict` :321-334):

1. TinyBERT predicts. If softmax confidence ≥ `confidence_threshold` (from
   `threshold.json`), use its answer; else DistilBERT re-predicts (that sample only).
2. Whichever model answered: if its confidence > that model's *ambiguous* threshold
   (`tiny_ambiguous_threshold.json` / `large_ambiguous_threshold.json`), return ONE label;
   otherwise return top-k labels (k = 3, or 2 when only 2 classes; :581-582, :887-888)
   → triggers the ambiguity-clarification flow.

**Threshold tuning** (all per context, written at train time):

- `confidence_threshold` ← `find_optimal_threshold` (:222-258): sweeps 20 `linspace` points
  between TinyBERT's mean-failed-confidence and mean-successful-confidence, maximizing
  `f1 * ndcg * (1 - 0.15 * distil_usage%/100)` (alpha=0.15 at :255) — i.e., accuracy
  discounted by how often the slower model is consulted. If a context has zero
  misclassifications (no failed-confidence mean), it writes threshold **-1** (:229-241) —
  every input then goes to whichever branch `-1` implies (tiny always confident).
- `tiny/large_ambiguous_threshold` ← simply the **mean confidence of that model's
  misclassified test samples** (:1174-1190), 0.0 if it never misclassified.
- Real values (CME workflow `global` context, committed in-repo): confidence 0.4129,
  tiny-ambiguous 0.4010, large-ambiguous 0.5468. Inspect any workflow with
  `scripts/show_intent_thresholds.py` (in this skill).
- `find_optimal_confidence_threshold` (:50-162) with its magic `min_threshold=0.5129` and
  `max_top3_usage=0.3` is **DEAD CODE** — nothing calls it; the live path uses
  `find_optimal_threshold`. Provenance of 0.5129: unknown, presumed leftover experiment.
- A legacy `ambiguous_threshold.json` lingers in trained folders; `CommandRouter.__init__`
  (:298-311) reads only `threshold.json` + the `tiny_`/`large_` variants.

**What F1 / NDCG@3 mean here**: F1 (weighted) scores hard top-1 classification on the
25% held-out utterances. NDCG@3 credits partially-correct ranking — it is the quality
measure for the *top-k clarification list* the user sees when the model is unsure
(implementation :393-414 and :719-730). High F1 + low NDCG@3 would mean confident answers
are fine but clarification lists are bad.

Artifacts land in `<workflow>/___command_info/<context>/`: `tinymodel.pth/`,
`largemodel.pth/` (HF `save_pretrained` dirs), `label_encoder.pkl`, `threshold.json`,
`tiny_ambiguous_threshold.json`, `large_ambiguous_threshold.json`. Missing
`threshold.json` = untrained context; `is_workflow_trained` (:624-674) fail-fast checks
exactly this before chat starts. **NEVER let tests/experiments write into
`fastworkflow/examples/*/___command_info` — train into temp copies (fix-0hb incident,
commit fa97b48).**

Full training walkthrough + metrics math: [references/intent-model-training.md](references/intent-model-training.md).

## 3. Synthetic utterance generation (train-time)

`fastworkflow/train/generate_synthetic.py::generate_diverse_utterances`. Called from each
command file's generated `generate_utterances` staticmethod (template:
`build/command_file_template.py:112-115`). Per command:

- Samples `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` (template default 4) personas from
  HuggingFace `proj-persona/PersonaHub` `persona.jsonl` (:40), batches
  `PERSONAS_PER_BATCH` (1) × `UTTERANCES_PER_PERSONA` (5) through `litellm.completion` on
  `LLM_SYNDATA_GEN` (max_tokens=1000, temperature=1.0, top_p=0.9; :112-119).
- Returns `[command_name] + seeds + generated` → default economics ≈ name + seeds + 4×5=20
  synthetic utterances per command. These env dials are your cost/quality knobs.

**Traps (verified):**

| Trap | Location | Consequence |
|---|---|---|
| `RateLimitError` → returns `[]` — silently dropping even the SEED utterances | :120-122 | That command is underrepresented/absent in the intent model; training "succeeds" |
| LLM reply parsed by splitting on `[` / `]` persona headers | :128-143 | Format drift in the model's reply silently loses utterances |
| The 3 `SYNTHETIC_UTTERANCE_GEN_*` env vars are read at **module import** with `int` coercion, no defaults | :14-16 | Importing before `fastworkflow.init()` or with a sparse env file yields `None` constants, failing later and obscurely |
| No `datasets` package → seeds-only, warning logged | :26-32 | Deliberate slim-image degradation (`pip install fastworkflow[training]` to fix) |

## 4. DSPy — three distinct uses, don't conflate them

| Phase | What DSPy does | Where |
|---|---|---|
| `fastworkflow refine` | 5 `dspy.Signature` classes each in `ChainOfThought` generate field metadata, utterances, docstrings, workflow description; applied as additive-only LibCST edits | `build/genai_postprocessor.py:37-108` (signatures), :123/:151/:180-181/:217 (modules), LM = `get_lm("LLM_COMMAND_METADATA_GEN", "LITELLM_API_KEY_COMMANDMETADATA_GEN", max_tokens=2000)` :239 |
| `fastworkflow train` | NOT dspy calls — `litellm.completion` directly generates `dspy.Example(...)` literals as few-shot corpus for parameter extraction | `utils/generate_param_examples.py:312`, called with `num_examples=15, validation_threshold=0.3` from `train/__main__.py:110-114`; output `___command_info/<cmd>_param_labeled.json` |
| Run time | Dynamic Signature from the command's Pydantic `Input` model + `ChainOfThought` + `LabeledFewShot(k=len(trainset))` + `JSONAdapter` + `BestOfN(N=3, threshold=1.0)` | `utils/signatures.py:239-313` |

Run-time parameter extraction detail (`utils/signatures.py`):

- The signature docstring is generated per command, embedding field descriptions, enum
  values, `examples=[...]`, Required/Optional status, and defaults (:156-237). **This is why
  improving `Field(description=, examples=, pattern=)` metadata is THE lever for extraction
  quality** — it changes the prompt directly.
- `BestOfN`'s reward (`basic_checks` :285-292) returns 0.0 if any extracted value equals one
  of that field's `examples` — anti-parroting. Same rejection exists on the agent-mode
  regex path (`parameter_extraction.py:330-337`).
- Results are built with `model_construct(**param_dict)` — **no Pydantic validation** —
  with `NOT_FOUND` sentinels for gaps (:309-313); validation happens separately in
  `validate_parameters` (:315-670) which coerces types, checks regex `pattern`s, and runs
  `db_lookup`/`validate_extracted_parameters` hooks.
- Agent mode tries **regex XML extraction first** (`<field>value</field>` per
  `parameter_extraction.py:71-80, 296-357`); the LLM is only called if regex fails. All
  fields must be present or it falls back.
- `signatures.py:255` instantiates `dspy.LM(LLM_PARAM_EXTRACTION, ...)` directly — a legacy
  exception; every OTHER call site routes through `dspy_utils.get_lm` (so litellm_proxy
  routing works everywhere EXCEPT it also works here only if the raw model string resolves;
  see §6).
- Agent-loop `AdapterParseError` (DSPy failed to parse the LLM's structured reply) is
  retried up to **2 attempts total** at `workflow_execution_context.py:696-707`.
- The intent-clarification agent is a tool-free `ChainOfThought` over
  `IntentClarificationAgentSignature` (`intent_clarification_agent.py:11-54`).

**Verified defect you must know**: `generate_dspy_examples` computes fuzzy validation
(normalized-Levenshtein vs the utterance) and logs/saves rejections, **but returns ALL
parsed examples anyway** — `utils/generate_param_examples.py:608` transforms `examples`,
not `validated_examples`; the filtered return at :612 is commented out. So
`valid_examples` in `<cmd>_param_labeled.json` may contain hallucinated parameter values,
and a `rejected_examples.json` is dropped into the **current working directory** (:606).
Also note `eval()` on LLM-derived text at :179 (examples-list parsing). Treat this whole
file as a known-weak point.

**DSPy caching**: DSPy memoizes LLM calls on disk+memory. If refine/agent outputs seem
frozen after a prompt change, clear it:
`python -c "import dspy; dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)"`
in-process, or `rm -rf ~/.dspy_cache/ ./.dspy_cache/` (dir verified via
`python -m fastworkflow.utils.dspy_cache_utils status`, which reports `~/.dspy_cache` —
there is no `~/.cache/dspy`). `docs/DSPY_CACHE_GUIDE.md` documents this but has
doc rot: it references `fastworkflow.run_agent.agent_module` (module no longer exists) and
a root-level `dspy_cache_utils.py` (actually at `fastworkflow/utils/dspy_cache_utils.py`).
Cached calls also don't appear in `dspy.inspect_history()`.

More detail on all three phases: [references/dspy-and-synthetic-data.md](references/dspy-and-synthetic-data.md).

## 5. Ambiguity clarification state machine

`NLUPipelineStage` enum (`fastworkflow/__init__.py:13-18`), stored in the CME workflow's
context dict:

```
INTENT_DETECTION ──ambiguous (top-k)──> INTENT_AMBIGUITY_CLARIFICATION
INTENT_DETECTION ──out-of-scope*──────> INTENT_MISUNDERSTANDING_CLARIFICATION
(any) ──command resolved──> PARAMETER_EXTRACTION ──valid──> execute, reset to INTENT_DETECTION
```

\* out-of-scope: `wildcard.py:99-124` first walks PARENT contexts re-running prediction
before declaring misunderstanding.

- Suggested commands for the constrained re-selection are persisted in the app workflow's
  `___convo_info/cache.db` Rdict (`intent_detection.py:189-217`).
- In agent mode, ambiguity is delegated to the tool-free intent-clarification agent; if it
  can't resolve, it executes `abort` to reset the stage and tells the outer agent to
  `ask_user` (`workflow_agent.py:105-119, 229-259`).
- Successful clarification feeds the layer-3 learning cache (§1).
- Stale-state warning: `'command'`, `'stored_parameters'`, `'NLU_Pipeline_Stage'` in the CME
  context are reset by `Workflow.end_command_processing` (`workflow.py:291-303`); a leaked
  flag forces the wrong mode next turn.

## 6. litellm routing (per-role model selection)

One LLM role = one model env var + one key env var, resolved through
`utils/dspy_utils.py::get_lm` (:8-69). Template values are all
`mistral/mistral-small-latest` (`fastworkflow/examples/fastworkflow.env:6-11`).

| Role var | Key var | Consumer (verified call site) |
|---|---|---|
| `LLM_AGENT` | `LITELLM_API_KEY_AGENT` | `workflow_agent.py:270,309`; `workflow_execution_context.py:691` |
| `LLM_PLANNER` | `LITELLM_API_KEY_PLANNER` | `workflow_agent.py:507`; `workflow_execution_context.py:1019` |
| `LLM_PARAM_EXTRACTION` | `LITELLM_API_KEY_PARAM_EXTRACTION` | `utils/signatures.py:252-255` (direct `dspy.LM`, legacy) |
| `LLM_SYNDATA_GEN` | `LITELLM_API_KEY_SYNDATA_GEN` | `train/generate_synthetic.py:35-36`; `utils/generate_param_examples.py:333-334` |
| `LLM_COMMAND_METADATA_GEN` | `LITELLM_API_KEY_COMMANDMETADATA_GEN` (no underscore in COMMANDMETADATA) | `build/genai_postprocessor.py:239` — used by `build`/`refine`, **absent from the packaged env template**, and build/refine load NO env files (must be in OS env) |
| `LLM_CONVERSATION_STORE` | `LITELLM_API_KEY_CONVERSATION_STORE` | `run_fastapi_mcp/conversation_store.py:331` |
| `LLM_RESPONSE_GEN` | `LITELLM_API_KEY_RESPONSE_GEN` | **DEAD CONFIG** — templated and documented, zero code consumers (verified by grep) |

- **Proxy routing**: model value prefixed `litellm_proxy/` → routed via
  `LITELLM_PROXY_API_BASE` (mandatory, ValueError if unset) with optional
  `LITELLM_PROXY_API_KEY`; the per-role key var is IGNORED for proxied models (:48-65).
- **Env precedence quirk** (`fastworkflow/__init__.py:211-219`): `get_env_var` returns a
  supplied code default WITHOUT consulting `os.environ`. So `INTENT_DETECTION_TINY_MODEL`
  and `INTENT_DETECTION_LARGE_MODEL` can only be overridden via the workflow's **env
  file**, never the shell.
- New LLM call sites MUST use `get_lm(model_var, key_var)`, never bare `dspy.LM`.

## 7. Magic numbers: provenance is UNDOCUMENTED

No design doc, commit message, or comment justifies any of the following. **Treat every
one as empirical — "provenance unknown" — until the tau2 program (E-cards) revisits them.**
Top offenders (full table with all ~25 entries:
[references/magic-numbers.md](references/magic-numbers.md)):

| Value | What it gates | file:line |
|---|---|---|
| 0.3 | Fuzzy command-name match (max normalized Levenshtein distance) | `_workflows/command_metadata_extraction/intent_detection.py:111` |
| 0.85 | Embedding-cache cosine threshold (call site; fn default 0.90 unused) | `intent_detection.py:118` / `cache_matching.py:131` |
| 0.65 | `ModelPipeline` default confidence threshold (overwritten by trained value at load) | `model_pipeline_training.py:346,360,1059` |
| 0.15 | alpha: DistilBERT-usage penalty in threshold scoring | `model_pipeline_training.py:255` |
| 20 | linspace points swept in threshold search | `model_pipeline_training.py:227` |
| 12 / 5 | tiny / distil fine-tuning epochs | `model_pipeline_training.py:957,1020` |
| 1e-4 / 5e-5 | tiny / distil learning rates | `model_pipeline_training.py:956,1019` |
| 10 | training batch size | `model_pipeline_training.py:936,942` |
| 0.25 / 42 | test split fraction / random seed | `model_pipeline_training.py:914` |
| 0.5129 | `min_threshold` in DEAD `find_optimal_confidence_threshold` | `model_pipeline_training.py:50` |
| 15 / 0.3 | DSPy examples per command / fuzzy validation threshold (validation currently non-filtering, §4) | `train/__main__.py:113-114` |
| 0.9 / 4000 | temperature / max_tokens for DSPy example generation | `utils/generate_param_examples.py:335,407` |
| 1.0 / 0.9 / 1000 | temperature / top_p / max_tokens for utterance generation | `train/generate_synthetic.py:112-119` |
| 3 / 1.0 | BestOfN attempts / reward threshold in param extraction | `utils/signatures.py:295-299` |
| 2 | agent-call retries on AdapterParseError | `workflow_execution_context.py:700` |
| 0.2 / 0.7 | `DatabaseValidator.fuzzy_match` difflib cutoff / Levenshtein threshold | `utils/signatures.py:70,86` |
| 256 | embedding lru_cache size | `cache_matching.py:18` |

## 8. Known doc rot (trust code, cite this when correcting docs)

| Doc claim | Reality | Evidence |
|---|---|---|
| CLAUDE.md: intent models "DistilBERT/BERT via scikit-learn" | torch/transformers fine-tuning; sklearn = LabelEncoder/split/f1 only | `model_pipeline_training.py:3-17,956-957` |
| (Surprise, not rot) train-time code implies training lives in `fastworkflow/train/` | The actual training loop `model_pipeline_training.py` sits at the `fastworkflow/` package root (it is also a run-time dependency: `CommandRouter`/`ModelPipeline`) | `ls fastworkflow/model_pipeline_training.py` |
| `docs/DSPY_CACHE_GUIDE.md`: `fastworkflow.run_agent.agent_module`, root `dspy_cache_utils.py` | module gone; utility at `fastworkflow/utils/dspy_cache_utils.py` | grep |
| env template documents `LLM_RESPONSE_GEN` | zero consumers | grep `--include=*.py` |
| pyproject.toml:41 comment references `model_pipeline_training._load_tokenizer` | function no longer exists | grep |

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verify volatile facts:

```bash
# Training hyperparameters (epochs/lr/batch/split) and threshold logic
grep -n "num_epochs\|lr=\|batch_size=10\|test_size" fastworkflow/model_pipeline_training.py
# Dead 0.5129 function still dead? (should only show the def, no call sites)
grep -rn "find_optimal_confidence_threshold" fastworkflow/
# Matching-ladder thresholds
grep -n "threshold=0.3\|0.85" fastworkflow/_workflows/command_metadata_extraction/intent_detection.py
grep -n "threshold=0.90" fastworkflow/cache_matching.py
# Model defaults
grep -n "INTENT_DETECTION" fastworkflow/model_pipeline_training.py
# DSPy example-gen numbers and the non-filtering return
grep -n "num_examples=15\|validation_threshold=0.3" fastworkflow/train/__main__.py
sed -n '604,613p' fastworkflow/utils/generate_param_examples.py
# Per-role LLM consumers
grep -rn "get_lm(" fastworkflow/ | grep -v "def get_lm"
# LLM_RESPONSE_GEN still dead?
grep -rn "LLM_RESPONSE_GEN" fastworkflow/ --include=*.py
# BestOfN / LabeledFewShot runtime extraction
grep -n "BestOfN\|LabeledFewShot\|JSONAdapter" fastworkflow/utils/signatures.py
# Live thresholds of any trained workflow
python .claude/skills/fastworkflow-nlu-pipeline-reference/scripts/show_intent_thresholds.py <workflow_dir>
```
