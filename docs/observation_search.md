# Observation search

Large, older command results can be saved outside the ReAct prompt. A replacement
is emitted only when it is shorter than the original in both characters and
UTF-8 bytes. The existing size threshold, recent-observation protection and
trajectory budget still apply. Replan copies follow the same savings rule and
persist any newly labelled command observation before returning a pointer.
Non-command observations in a replan copy stay inline. If this irreducible
evidence exceeds the byte target, the runtime records the overage and continues.

The replacement format is:

> Use search_memory tool to search inside Observation O8 returned by show_holders. It was offloaded to memory and contains identity_uid: identities holding this permission; label: their display names.

The command is the original `execute_workflow_query` command argument. Output
field descriptions come from the resolved command's authored `Signature.Output`
metadata. When those descriptions are unavailable, the label explicitly describes
the beginning of the command output. Hashes remain in the archive and tracing
manifest rather than taking space in the prompt label.

## Configuration

Set the search model independently of the main agent:

```dotenv
# fastworkflow.env
LLM_OBSERVATION_SEARCH=cerebras/gpt-oss-120b
```

```dotenv
# fastworkflow.passwords.env
LITELLM_API_KEY_OBSERVATION_SEARCH=<provider API key>
```

The standard `get_lm` routing rules also support `litellm_proxy/` routes and
provider ambient credentials. No fallback to the main agent model is performed.
The search uses temperature 0, a 2,048-token completion limit, a 120-second
request timeout and one provider retry. `FW_LM_CACHE=0` disables response caching
for independent benchmark calls.

## Tool behavior

`search_memory(question: str, alias: str)` requires one key such as `O8`.
An empty key, a noncanonical key, multiple keys, or an empty question is rejected.
A missing handle produces an explicit error without calling a model or searching
another observation. Resolution remains scoped to the current turn/attempt and
survives cache eviction through the SQLite archive.

The implementation reads the **current search step's thought** from the ReAct
trajectory and prefixes it to the question as `<reasoning>. <question>`. Reasoning
is not an agent-supplied tool argument. DSPy `Predict` receives that combined
question and the complete selected observation. It answers from that observation,
preserves exact identifiers and their types, and states evidence gaps. The prompt
instructs it to treat both the requesting agent's assumptions and instructions
inside the observation as untrusted claims, not additional evidence.

Broad requests for entire tables receive a count/description and a request for a
focused predicate. Completions cut off at the output limit are reported as
incomplete searches rather than successful evidence answers.

The returned answer names the source observation. Search events record scope,
source hash and size, question and attached reasoning, status, model, answer,
latency and available provider usage/cost. LLM calls are also recorded through
normal DSPy observability. Provider failure yields an explicit search failure;
it is never reported as evidence that an entity is absent.

The full observation consumes the search model's input context and incurs model
cost. There is no silent truncation or page limit. An observation too large for
the configured provider may fail; the caller receives an explicit failure and the
original saved text remains available. This search supplies evidence; it does not
by itself guarantee that the main agent's final conclusion is correct.

## Validation

Focused tests cover labels, byte/character savings, replan persistence, scoped
resolution, required keys, current-step reasoning, archive eviction and existing
continuation behavior. Provider tests are opt-in:

```bash
FW_TEST_OBSERVATION_SEARCH_LIVE=1 python -m pytest \
  tests/test_observation_offloading.py tests/test_observation_search.py
```

Configure the search model and credentials before enabling provider tests.
Without the opt-in, deterministic integration checks run and paid cases skip.
The IDO epic `ido-5uv` records a fixed-evidence and single-task benchmark comparison.
