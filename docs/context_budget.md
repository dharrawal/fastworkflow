# Context budgets: one input, every byte budget derived from it

`fastworkflow/context_budget.py`.

fastWorkflow bounds several things by UTF-8 bytes: how much trajectory the ReAct
loop carries into the next step, how much evidence the answer-time extract call
is given, how large one `search_memory` answer may be, how
much text the process-local cache may hold, and how much an offload must save
to be worth doing. Every one of them answers the same question — *how much of
the model's context window may this occupy* — so there is one input and the rest
are fractions of it.

Before 3.4.0 they were seven independent environment variables with seven
unrelated constants, which meant that moving a workflow from a 131k-token model
to a 32k-token one was seven correlated edits nobody made.

## The one input

The model's context window, in tokens. Resolved in this order:

| # | Source | How |
|---|--------|-----|
| 1 | `FW_MODEL_CONTEXT_TOKENS` | the deployment states it outright; read from the workflow's `fastworkflow.env` first, then the process environment. Values below `MIN_WINDOW_TOKENS` (4,096) or unparseable are refused with a warning and the next source answers. |
| 2 | the main agent model's metadata | `litellm.get_model_info(<LLM_AGENT>)["max_input_tokens"]` (falling back to `max_tokens`). A model litellm does not know is not an error; the next source answers. Answered once per process per model id. |
| 3 | the documented fallback | `REFERENCE_WINDOW_TOKENS` = **131,072**, the window the reference configuration runs at. |

`context_budget.context_window_tokens()` returns `(tokens, source)`, where
`source` is `"setting"`, `"model_metadata:<model id>"` or `"fallback"`.

**Tokens to bytes** is converted exactly once, in this module:
`BYTES_PER_TOKEN = 4`. It is a presentation ratio, not a tokenizer — these
budgets bound prompt text in UTF-8 bytes and 4 bytes per token is the ratio the
measured runs sat at. At the reference window that is **524,288 bytes**.

## The budgets

Each budget is a fixed rational fraction of the window in bytes, floored at the
minimum its own module always refused to go under. The fractions are calibrated
so that `cerebras/gpt-oss-120b` — for which litellm reports
`max_input_tokens = 131072` — reproduces the reference configuration's pinned
values exactly.

| Budget | What it bounds | Fraction of the window | At 131,072 tokens | Tuning override |
|---|---|---|---|---|
| `trajectory_max_bytes` | packed-trajectory target and replan bound | 875/16384 (≈ 5.34 %) | **28,000** | `FW_TRAJECTORY_MAX_BYTES` |
| `answer_rehydration_max_bytes` | answer-time rehydration budget for the extract call | 15625/32768 (≈ 47.68 %) | **250,000** | `FW_ANSWER_REHYDRATION_MAX_BYTES` |
| `search_answer_max_bytes` | one `search_memory` answer observation, marking included | 3/512 (≈ 0.586 %) | **3,072** | `FW_SEARCH_ANSWER_MAX_BYTES` |
| `offload_hot_max_bytes` | process-local hot cache of offloaded observations | 1/2 | **262,144** | `FW_OFFLOAD_HOT_MAX_BYTES` |
| `offload_min_saving_bytes` | minimum UTF-8 bytes an offload must free | 1/512 (≈ 0.195 %) | **1,024** | `FW_OFFLOAD_MIN_SAVING_BYTES` |

The hot cache takes half the window because it is not prompt: it is
what the prompt can be rebuilt from, and the durable copy is SQLite, so an
eviction costs a re-read and never loses evidence. The rehydration budget is by
far the largest prompt share because the extract call is one call with no loop
after it.

**One budget is not derived from this window, and is not in the table.** The
bound on the observation handed to the search model is sized from the *search*
model's context window (`LLM_OBSERVATION_SEARCH`, or `LLM_AGENT` when that is
unset), because it exists to fit that model's prompt rather than the agent's.
It is resolved separately, it has its own tuning override
(`FW_SEARCH_OBSERVATION_MAX_BYTES`), and `budget_provenance()` does not report
it — a provenance record carrying a number derived from a different window
would be wrong more often than it was useful. See
[`docs/observation_search.md`](observation_search.md) for how it is resolved.

`tests/test_context_budget.py` asserts the identity above, that half and double
the window give half and double every budget, that an override wins, and that
the fallback path is taken when nothing is set.

### Overrides are tuning, not the interface

The `FW_*_MAX_BYTES` names remain so that one budget can be moved without moving
the others — an experiment, a provider with an unusual prompt accounting. They
are not how a deployment picks its budgets: it sets the window, or lets the
model's own metadata set it, and leaves these alone. An override below its
budget's floor, or one that does not parse, is refused with a warning and the
derived value stands.

Unlike the readers they replace, an override written into a workflow's own
`fastworkflow.env` now takes effect: the reader consults the env file before the
process environment.

## Provenance

```python
from fastworkflow import context_budget
context_budget.budget_provenance()
```

```json
{
  "context_window_tokens": 131072,
  "context_window_source": "model_metadata:cerebras/gpt-oss-120b",
  "bytes_per_token": 4,
  "context_window_bytes": 524288,
  "reference_window_tokens": 131072,
  "budgets": {
    "trajectory_max_bytes": 28000,
    "answer_rehydration_max_bytes": 250000,
    "search_answer_max_bytes": 3072,
    "offload_hot_max_bytes": 262144,
    "offload_min_saving_bytes": 1024
  },
  "overrides": {}
}
```

One call, one JSON-serialisable dict, so a runner records what the run was
bounded by without re-deriving it. `overrides` names only the budgets a tuning
override actually moved, so an empty `overrides` is itself the statement "these
are the derived budgets".

## What is not a context budget

`FW_OBS_MAX_ATTR_BYTES` used to look like one and was not: it caps a single span
attribute written to the **observability store**, which is a database row, not a
model prompt, and nothing about it scales with a model's window. It is now the
constant `fastworkflow.tracing.MAX_ATTR_BYTES` (16,384) and the observability
provenance record still reports it, because the value in effect is now fixed.
