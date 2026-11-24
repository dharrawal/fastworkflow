---
name: train-and-publish-models
description: >-
  Run, interpret and recover a `fastworkflow train` run, given that training is cached,
  incremental and versioned: deterministic utterance and parameter-example caches, per-command
  fingerprints that decide which contexts retrain, and an atomically published artifact version
  with one previous recovery point. Covers what invalidates a cache, what forces a full retrain
  instead of a selective one, what a failed run leaves behind, how to roll back, and which files
  must never be hand-deleted. Use before or after running `fastworkflow train`, when a run
  regenerated more than expected, when it retrained everything unexpectedly, when it reports
  artifacts are already up to date, or when trained models need to be rolled back.
---

# Training and publishing models

## The command

```bash
fastworkflow train <workflow_folderpath> [env_file_path] [passwords_file_path] [--regenerate-utterances]
```

The two env-file arguments are positional and optional; they default to `fastworkflow.env` and
`fastworkflow.passwords.env` inside the workflow folder, and the command exits with an error when
either is missing. `--regenerate-utterances` is the only training-policy override, and it is
expensive (Rule 2).

Training recurses into `<workflow>/_workflows/*` first, so a child workflow is trained before its
parent, each with its own persona source and caches. When the framework's own workflows (command
metadata extraction, plan metadata extraction) are untrained or stale, they are trained first.

## The intent model layout

`INTENT_MODEL_LAYOUT` in the workflow env file chooses what a run writes. It is read by
`fastworkflow train` only; the runtime reads whatever layout the published version holds.

| Layout | What is trained | Time |
|---|---|---|
| `embedding` (default) | ONE logistic-regression head over every command of the workflow, on a frozen sentence encoder that is never trained | seconds |
| `workflow` | ONE fine-tuned large classifier for the workflow | minutes to hours |
| `per_context` | a fine-tuned tiny and large classifier for every context | minutes to hours |

Every layout trains on the same rows: your seed utterances plus LLM-generated utterances, so the
caches below matter in all three. The `embedding` settings are `INTENT_EMBEDDING_MODEL` (default
`intfloat/e5-base-v2`, downloaded once into the Hugging Face cache, not stored per workflow),
`INTENT_EMBEDDING_POOLING` (`cls` or `mean`; the default follows the encoder) and
`INTENT_COMMIT_THRESHOLD` (default 0.65: the head answers with one command only at or above it).
The framework's own workflows are always `embedding` with a pinned encoder, whatever your env says.

An invalid layout or setting value logs one warning and means the default.
`INTENT_WRONG_COMMIT_TARGET` and `INTENT_FALSE_ESCALATION_TARGET` were removed: a value left in the
env file is ignored with one warning. Delete the line.

## What one run does, in order

Everything before the first paid call is a gate, and everything after publication is cleanup.

```
0. Command files load?            load_problems                <- exits 1 before any work
1. Build routing artifacts        command_directory.json, routing_definition.json
2. Duplicate-capability scan      lexical, seeds only          <- free, reports
3. Benchmark preflight            leak + defect checks         <- free, CAN FAIL THE RUN
4. Migrate legacy artifacts       into versions/ if unversioned
5. DSPy parameter examples        param_example_cache          <- costs money on a miss
6. Compute the training plan      selective or full
   -> if the plan is empty: republish, apply retention, stop ("already up to date")
7. Generate utterances + train    utterance_cache              <- costs money on a miss
8. Carry forward untrained contexts from the previous version (per_context only)
9. Router-confusion scan          needs every context present
10. Write signature + manifest
11. Safety gate on training data  CAN REFUSE TO PUBLISH
12. Publish + retain current and previous   <- the commit point
13. Prune orphaned artifacts
```

Steps 2 and 3 sit before every paid call deliberately: a leaked benchmark used to abort a run
that had already spent money on every command with parameters. Step 2 also warns about any command
with fewer than eight seed utterances; training continues.

Planning commands are labels of the model like execution commands, with generated utterances of
their own. A version trained before that rule (or before the current "not here" construction)
refuses to load at run time and says to retrain.

## Rule 1: the seed is fixed, so a cache miss is the only source of drift

`get_training_seed()` returns **42**, always. It is not configurable, and a `TRAINING_SEED`
environment variable is **not read**.

Seeding alone does not make training reproducible: the utterance generator is a live LLM, and two
runs at the same seed were measured producing different utterances and different row counts for
every command. Reproducibility comes from the **caches**. With caches warm, two runs train on
identical data and produce identical models. The moment something misses, that guarantee is gone
for the commands that missed.

## Rule 2: know what invalidates a cache before you blame the cache

Two caches, both under `___command_info/`, both keyed on a fingerprint of every input that could
change what the LLM produces:

| Cache | Holds | Directory |
|---|---|---|
| utterance cache | the synthetic utterances one command was trained on | `utterance_cache/` |
| parameter-example cache | the DSPy few-shot examples for one command's parameters | `param_example_cache/` |

A miss is **reported at INFO, naming the input that moved** (up to three, with before and after
values). Read that line instead of hand-diffing JSON. The inputs include the command name, its seed
utterances, the persona counts, the generation model and API base, the persona source, and a digest
of the generation prompt's own source code. Full list in [reference.md](reference.md).

A lost cache entry produces a slow, non-reproducible run, never a wrong one: editing a command's
seeds always regenerates exactly that command.

**`--regenerate-utterances` ignores both caches.** It calls the LLM again for everything, costs
money, breaks reproducibility against previous runs, and forces a **full** retrain (Rule 3). Use it
when you suspect a cache is wrong, not as a habit.

## Rule 3: what retrains

A command is dirty when its **source file bytes** or its **seed utterances** changed.

- **`embedding` and `workflow` layouts:** one model serves every context, so any dirty command
  refits the whole model, and nothing dirty is a no-op. In `embedding` this costs seconds plus LLM
  calls only for the commands whose utterance cache missed.
- **`per_context` layout:** dirty commands are closed over to the contexts that must retrain; the
  rest are carried forward from the previous version. Any path that cannot prove a context
  unchanged retrains it.

A **full** retrain is forced when any of these is true:

- there is no published version to carry forward from, or no usable previous signature;
- `--regenerate-utterances` was passed;
- any **global input** of the training signature changed: the layout, the `embedding` settings
  (encoder, pooling, commit threshold), the artifact format version, the seed, the tiny or large
  model setting, the synthetic-data model, **the persona source**, or a source digest of the
  framework's generator or trainer modules (full list in [reference.md](reference.md));
- (`per_context`) commands shared between retrained and carried-forward contexts have no cache
  entries, which would mix cached and freshly generated utterances in one model.

Two consequences people trip over:

- **Adding or editing `personas.json` retrains everything and regenerates every utterance.** It
  is a global input of both keys, though switching back to a recently used source reuses its
  utterances from the cache (`MAX_VARIANTS_PER_COMMAND` = 3 per command) while the model still retrains. See the `supply-training-personas` skill.
- **Upgrading the framework can force a full retrain**, because the generator and trainer source
  digests are global inputs.

There are no `--changed-only` / `--only-contexts` CLI flags.

## Rule 4: only a successful run becomes current

Model artifacts live in `___command_info/versions/<version_id>/`. A pointer file `current.json`
says which version is live; a `current` symlink and one compatibility entry per context point at
it. `current.json` is written **last**, so it is the commit point:

- A run that **fails** (training, carry-forward, the training-data safety gate) never publishes.
  The previous version stays current and complete, with the partial version beside it.
- Pruning runs only **after** a successful publish.
- Retention keeps the **current version and one previous**, plus, after a refit (see "Learning
  from use"), the full train its lineage descends from. The previous version is your only recovery
  point: do not run training twice to "get back" to a known-good set; the second run discards it.

## Rule 5: never hand-delete `___command_info`

It is not a build directory. In one place it holds:

- the trained models (for `per_context` and `workflow`, gigabytes and hours to rebuild);
- the **only** rollback point;
- both caches, which are the only thing making two runs train on the same data, and whose loss
  means paying for every utterance again;
- the selective-training baseline, whose loss forces a full retrain.

If you need to force work to happen, use the narrow instrument:

| Want | Do | Not |
|---|---|---|
| regenerate one command's utterances | edit its seed utterances | delete the cache |
| regenerate everything | `--regenerate-utterances` | `rm -rf ___command_info` |
| retrain | change a command (in `per_context`, a command of that context) | delete a model folder |
| go back to the previous models | republish the previous version (Rule 6) | retrain and hope |

The trainer's own pruning removes an artifact only when its command or context no longer exists,
and never touches `versions/`, `current` or either cache.

## Rule 6: rollback is a Python call, not a flag

There is no rollback CLI:

```python
from fastworkflow.train import artifact_versioning

for version in artifact_versioning.list_versions(workflow_folderpath):   # newest first
    print(version.version_id, version.created_at, version.is_current)

with artifact_versioning.publication_lock(workflow_folderpath):
    artifact_versioning.publish_version(workflow_folderpath, "<older version id>")
```

Republishing rewires the compatibility entries and rewrites `current.json`. That version's
`training_signature.json` comes back with it, so the next run diffs against the models you
restored. `publish_version` refuses to advance `current` to a version whose manifest contradicts
itself or the version its carried-forward models came from.

A running process picks up a newly published `embedding` head (a retrain, a refit or a rollback)
on its next request, within about 3 seconds, without a restart; a turn already running keeps its
head. For `workflow` and `per_context`, restart the server after any publish.

## Learning from use (`embedding` layout)

At run time every settled request is recorded in `<workflow>/___convo_info/intent_learning.sqlite3`
(`INTENT_LEARNING_RECORD`, default on; `off` writes nothing). Only requests a person or an agent
confirmed teach: a pick at a which-command question (by the user or the agent), a review that was
accepted or switched; never a match by command name.

```bash
fastworkflow learn <workflow_folderpath> [env_file_path] [--dry-run]   # refit + calibrate, no model call
fastworkflow learned list|stats <workflow_folderpath>                   # review what was recorded
fastworkflow learned remove <workflow_folderpath> <id> ...              # drop wrong rows
```

`learn` refits the current head with the training rows plus the confirmed rows (at least 5
distinct texts) and publishes a new version only when **both gates** pass: the base rows' cross-fit
accuracy falls at most 1.5 points, and the learned rows, held out, are right outright no less often
and wrong confidently no more often than without them (one more wrong row is allowed when at least
three more are right). It then **calibrates** the commit threshold, which can only be **lowered**
(to 0.60, 0.55 or 0.50, with at least 200 confirmed rows and at most 2% wrong in the new band); an
`INTENT_COMMIT_THRESHOLD` set in the env file always wins. It prints why it did or did not publish.

The same refit runs in a background thread after every 25 new confirmed rows
(`INTENT_LEARNING_REFIT_EVERY`); set `INTENT_LEARNING_REFIT=off` to stop it. A head trained
without `rows.npz` cannot be refitted until its next train. A full `fastworkflow train` does not
include the learned rows: run `fastworkflow learn` after it to add them back.

## Reading the output

| Line | Means |
|---|---|
| `Intent model layout: <layout> ...` | the layout this run writes, and for `embedding` the encoder and commit threshold |
| `Training plan: full retrain of N context(s).` + reasons | something global moved, or (one-model layout) any command changed; the reasons say which |
| `Training plan (selective):` with changed commands and per-context reasons | the `per_context` selective path |
| `Carried forward N context(s) from version <id>: ...` | those contexts were **not** retrained this run |
| `Utterance cache (reuse) at <path>: X reused, Y generated, Z written` | `Y` is what this run paid for |
| `Intent model: layout embedding, encoder ..., The head took ...` | the head's size and fit time |
| `Training artifacts are already up to date.` | nothing changed; reader paths were repaired and retention applied |
| `MODEL CONFUSION (N)` | diagnostic only, printed after training and before publication (it never blocks it); see `detect-duplicate-capabilities` |
| `Training complete.` | published; `current.json` now names this version |
| `LOCAL INTENT MODEL` section of the final report (`embedding`) | cross-fit accuracy on the training rows and the most confused command pairs, both directions; add seeds that tell a pair apart, or merge the two commands |

`Training artifacts are already up to date.` is a success, which is what makes re-running training
after a partial failure safe.

## After training

`fastworkflow run` and the servers print one line saying what the intent detection chain is: the
local model when it is confident, then the remote decision model when `JEV_API_KEY` is set, then
the user or the agent. With `LLM_AGENT` unset (no-agent mode), `JEV_API_KEY` is mandatory: without
it `fastworkflow run` exits 1 and the server refuses to start. `fastworkflow train` does not check
this.

## Pre-flight checklist

```
- [ ] Seed-utterance and command-source edits are complete; each makes its command dirty, and a seed edit also regenerates it
- [ ] Persona and layout/encoder changes are batched; each forces a full retrain
- [ ] intent_benchmark.json is disjoint from seeds (the preflight will fail the run otherwise)
- [ ] The previous version is the one you want as a recovery point
- [ ] --regenerate-utterances is being passed for a stated reason, not by habit
```

## Additional resources

- Exact cache fingerprint inputs, the global input list, the on-disk layout under
  `___command_info`, and a failure-mode catalogue: [reference.md](reference.md).

## Related

- Interpreting the numbers a run reports at the end: the `evaluate-intent-routing` skill.
- The persona configuration that forces a full retrain: the `supply-training-personas` skill.
- The two scans a run prints: the `detect-duplicate-capabilities` skill.
