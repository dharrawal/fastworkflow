# Magic numbers in the NLU stack — full inventory

Companion to `../SKILL.md` §7. Every value below was read from source on 2026-07-09
(v2.22.2, commit c33b9a5). **Provenance of ALL of these is undocumented — no design doc,
tuning log, or commit message justifies them. Treat each as empirical; if a tau2
experiment depends on one, measure sensitivity before trusting it.** "Live" = on the
executing path today; "dead" = present but unreached.

## Intent detection (runtime)

| Value | Meaning | file:line | Status |
|---|---|---|---|
| 0.3 | Max normalized Levenshtein distance for fuzzy command-name match | `fastworkflow/_workflows/command_metadata_extraction/intent_detection.py:111` | live |
| 0.4 | `find_best_matches` default distance threshold (overridden at both NLU call sites) | `fastworkflow/utils/fuzzy_match.py:19` | default, mostly shadowed |
| 0.85 | Cosine-similarity threshold for the utterance/clarification embedding cache | `intent_detection.py:118` (positional arg) | live |
| 0.90 | `cache_match` default threshold — no caller uses it | `fastworkflow/cache_matching.py:131` | dead default |
| 256 | `lru_cache` size for DistilBERT embeddings | `cache_matching.py:18` | live |
| 128 | Tokenizer max_length for embeddings | `cache_matching.py:46` | live |
| 5 | `majority_vote_predictions` n_predictions | `intent_detection.py:304` | dead (call commented out :122) |
| 10 | max_workers cap in majority vote | `intent_detection.py:332` | dead |

## Intent-model training

| Value | Meaning | file:line | Status |
|---|---|---|---|
| 0.65 | `ModelPipeline` default confidence threshold (used during training eval before tuning; runtime loads the trained value) | `fastworkflow/model_pipeline_training.py:346,360,1059` | live (transitional) |
| 12 | TinyBERT fine-tuning epochs | `model_pipeline_training.py:957` | live |
| 5 | DistilBERT fine-tuning epochs | `model_pipeline_training.py:1020` (re-set :1026) | live |
| 1e-4 | TinyBERT AdamW learning rate ("slightly higher" per comment) | `model_pipeline_training.py:956` | live |
| 5e-5 | DistilBERT AdamW learning rate | `model_pipeline_training.py:1019` | live |
| 10 | DataLoader batch size (train and test) | `model_pipeline_training.py:936,942,1007,1014` | live |
| 128 | max_length in train/eval/inference tokenization | `model_pipeline_training.py:928,439,470` | live |
| 0.25 | test split fraction | `model_pipeline_training.py:914` | live |
| 42 | split random_state | `model_pipeline_training.py:914` | live |
| 20 | linspace points in threshold sweep | `model_pipeline_training.py:227` | live |
| 0.15 | alpha — DistilBERT-usage penalty in threshold score | `model_pipeline_training.py:255` | live |
| 3 (or 2) | top-k size; 2 when only 2 classes | `model_pipeline_training.py:389,581-582,887-888` | live |
| 0.5129 | `min_threshold` in `find_optimal_confidence_threshold` | `model_pipeline_training.py:50` | **dead** — no call sites in `fastworkflow/` (a test `@patch` at `tests/test_wildcard_inheritance.py:390` still references it); suspicious 4-decimal precision suggests a copied empirical value; origin unknown |
| 0.3 | `max_top3_usage` in same dead function | `model_pipeline_training.py:50` | dead |
| 0.01 | step_size in same dead function | `model_pipeline_training.py:50` | dead |
| 0.95 | end-threshold cap in same dead function | `model_pipeline_training.py:69` | dead |

## Synthetic data generation

| Value | Meaning | file:line | Status |
|---|---|---|---|
| 4 / 5 / 1 | personas / utterances-per-persona / personas-per-batch (env template values; read at module import) | `fastworkflow/examples/fastworkflow.env:37-39`, `train/generate_synthetic.py:14-16` | live (env-tunable) |
| 1.0 / 0.9 / 1000 | temperature / top_p / max_tokens for utterance gen | `train/generate_synthetic.py:112-119` | live |
| 3 | min utterance length filter (`len(u) > 3`) | `train/generate_synthetic.py:137` | live |
| 15 | DSPy examples requested per command | `fastworkflow/train/__main__.py:113` | live |
| 0.3 | fuzzy validation threshold for examples (validation currently non-filtering — see dspy-and-synthetic-data.md §B) | `train/__main__.py:114` | live-but-inert |
| 10 / 0.4 | `generate_dspy_examples` defaults (overridden by the call site above) | `utils/generate_param_examples.py:315-316` | shadowed |
| 0.9 | temperature for example gen (hardcoded, ignores any arg) | `utils/generate_param_examples.py:335` | live |
| 4000 | max_tokens for example gen | `utils/generate_param_examples.py:407` | live |
| 0.4 | `validate_parameters` default threshold | `utils/generate_param_examples.py:21` | shadowed |
| 5 | max phrase length (words) in fuzzy example validation | `utils/generate_param_examples.py:59` | live |

## Parameter extraction (runtime)

| Value | Meaning | file:line | Status |
|---|---|---|---|
| 3 | `BestOfN` attempts | `fastworkflow/utils/signatures.py:297` | live |
| 1.0 | `BestOfN` reward threshold (i.e., must pass anti-parroting) | `signatures.py:299` | live |
| 0.2 | `DatabaseValidator.fuzzy_match` difflib cutoff default | `signatures.py:70,96` | live |
| 0.7 | Levenshtein threshold inside `DatabaseValidator.fuzzy_match` (NOTE: passed to a DISTANCE-thresholded function — 0.7 is very permissive; smells like a similarity/distance confusion, unconfirmed intent) | `signatures.py:86` | live |
| 3 | difflib `get_close_matches` n | `signatures.py:95` | live |
| -sys.maxsize / -sys.float_info.max | int/float "invalid" sentinels | `signatures.py:31-32`, `parameter_extraction.py:16-17` | live |
| 2 | agent-call attempts on `AdapterParseError` | `fastworkflow/workflow_execution_context.py:700` | live |
| 2000 | max_tokens for refine LM | `fastworkflow/build/genai_postprocessor.py:239` | live |

## Adjacent (owned by other skills, listed for completeness)

| Value | Meaning | file:line | Owner skill |
|---|---|---|---|
| 25 | agent max_iters | `fastworkflow/workflow_agent.py:387` | architecture-contract |
| 5 | conversation-history entries used for query refinement | `workflow_execution_context.py:981-991` | architecture-contract |
| 3 | invalid-tool-selection coaching bail-out | `fastworkflow/utils/react.py:219-236` | architecture-contract |

## Re-verification one-liner

```bash
grep -n "0.5129\|alpha = 0.15\|linspace\|threshold=0.3\|0.85\|threshold=0.90\|N=3\|max_retries = 2" \
  fastworkflow/model_pipeline_training.py \
  fastworkflow/_workflows/command_metadata_extraction/intent_detection.py \
  fastworkflow/cache_matching.py fastworkflow/utils/signatures.py \
  fastworkflow/workflow_execution_context.py
```
