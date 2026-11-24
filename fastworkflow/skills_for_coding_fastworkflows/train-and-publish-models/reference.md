# Reference: cache keys, layout and failure modes

Companion to [SKILL.md](SKILL.md). Read when a specific input, path or symptom is needed.

## Utterance cache fingerprint inputs

Every input below is hashed into a 24-character `variant_key`. Change any one of them and the
command regenerates. The miss report names up to three that moved.

| Input | Moves when |
|---|---|
| `cache_format_version` | the cache format itself changes |
| `command_name` | the command is renamed or moved between contexts |
| `seed_utterances` | any seed is edited, added, removed or **reordered** |
| `num_personas` | `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` changes |
| `utterances_per_persona` | `SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA` changes |
| `personas_per_batch` | `SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH` changes |
| `model` | `LLM_SYNDATA_GEN` changes |
| `api_base_digest` | the same model is pointed at a different proxy (hashed, since it can name an internal host) |
| `persona_source` | a `personas.json` is added, edited or removed, **or** the framework's persona-pool code changes |
| `completion_backend` | a caller injects its own completion function |
| `generator_source_digest` | the generation prompt's source text changes, comments included |

The persona component is three parts, `<name>#<content fingerprint>#<pool-code digest>`: a
changed third part with the first two intact says the framework's persona code was edited, not
your persona file.

`generator_source_digest` records `source-unavailable` when `inspect.getsource` cannot read the
module (zipapp, frozen build). Reuse still works; it just stops noticing prompt edits.

**Retention:** three variants per command survive a write, so flipping a prompt edit back and
forth, or training under two model strings, costs regeneration only once.

**Files:** `___command_info/utterance_cache/<slug>.<variant_key>.json`, plus a `README.md`
warning. The variant key covers everything except the seed; the seed selects an entry inside the
file.

## Training-signature global inputs

Any difference here forces a **full** retrain. Recorded per version in
`versions/<id>/training_signature.json`.

| Input | Source |
|---|---|
| `intent_model_layout` | `INTENT_MODEL_LAYOUT` |
| `embedding_head` | (`embedding` only) encoder, pooling, prefix, commit threshold, and the head trainer's source digest |
| `workflow_model_source_digest` | (`embedding`, `workflow`) source of the one-model row assembly and trainer |
| `format_version` | the artifact format constant |
| `seed` | `get_training_seed()`, fixed at 42 |
| `tiny_model` | `INTENT_DETECTION_TINY_MODEL` (default `google/bert_uncased_L-4_H-128_A-2`) |
| `large_model` | `INTENT_DETECTION_LARGE_MODEL` (default `distilbert-base-uncased`) |
| `syndata_model` | `LLM_SYNDATA_GEN` |
| `persona_source` | the active persona source label |
| `generator_source_digest` | source of the synthetic-generation module |
| `trainer_source_digest` | source of the model-pipeline-training module |
| `parameter_value_placeholders_sha256` | the reserved parameter-value placeholder set |
| `not_here` | the "not here" class construction the framework uses |

`tiny_model` and `large_model` are recorded in every layout, so changing either setting forces a
full retrain even in `embedding`, where neither is used. Settings are read from the workflow's env
files first and the process environment second, so an env-file line wins over a shell `export`.

Per-command staleness is decided by two hashes: the command's **source file bytes** and its
**seed utterances**. A command whose metadata cannot be read is always dirty.

In `per_context`, each context's signature also covers its label space and the commands of its
"not here" class, so a context retrains when its label space moves even if none of its own
commands changed. A context whose artifacts in the previous version are incomplete is retrained
rather than carried forward.

## Layout under `___command_info`

```
<workflow>/___command_info/
    command_directory.json          # NOT versioned: rebuilt from the command files
    routing_definition.json         # NOT versioned
    workflow_vocabulary.json        # NOT versioned
    <command>_param_labeled.json    # NOT versioned: read at runtime from the top level
    training_provenance.json        # this run's provenance (a copy is stamped into the version)
    heldout_evaluation.json         # this run's evaluation report
    training_report.txt / .json     # the final report
    duplicate_capabilities.json     # this run's duplicate scan
    utterance_cache/                # shared across versions, never pruned as an orphan
    param_example_cache/            # same
    current.json                    # AUTHORITATIVE pointer to the live version
    current       -> versions/<id>/            # convenience symlink, best effort
    <Context>     -> versions/<id>/<Context>/  # compatibility entry per context
    global        -> versions/<id>/global/     # the "*" context's folder is named `global`
    versions/
        README.md
        <version_id>/                          # e.g. 20261118T123510Z-94bf9a
            manifest.json
            training_signature.json
            training_provenance.json
            # embedding layout:
            workflow/{head.npz, scopes.json, rows.npz}
            <Context>/{threshold.json, workflow_scope.json}
            # workflow layout:
            workflow/{largemodel.pth/, label_encoder.pkl, scopes.json}
            <Context>/{threshold.json, workflow_scope.json}
            # per_context layout:
            <Context>/{tinymodel.pth, largemodel.pth, threshold.json,
                       tiny_ambiguous_threshold.json, large_ambiguous_threshold.json,
                       label_encoder.pkl}
```

In the one-model layouts a context's `threshold.json` holds 2.0, which is not a tuned number; the
thresholds that decide anything are in `workflow/scopes.json`. That file also holds the encoder,
each context's mask, the commit threshold and the cross-fit facts the training report prints.
`rows.npz` is the stored training vectors a refit reads; a head without it cannot be refitted.
The encoder itself is never stored.

Only the model folders belong to a version. The JSON snapshots are rebuilt by merely loading a
workflow, so versioning them would create versions on import.

Compatibility entries point **directly** at `versions/<id>/<Context>`, not through `current`, so
losing the `current` symlink cannot break every context at once. Where symlinks are unavailable
the entries are hardlink farms carrying a `.fastworkflow_compat` marker.

`list_versions` orders by the manifest's `created_at`, with the id as a tiebreaker. A refit
version's manifest adds `refit_of`, `lineage`, `learned_rows` and `refit_gate`.

`versions`, `current`, `utterance_cache`, `param_example_cache` and `__pycache__` are reserved
top-level names, skipped by publication's stale-entry sweep, by legacy migration and by orphan
pruning.

## Failure-mode catalogue

### The run regenerated everything and charged for it

**Cause.** A global input moved. The commonest are a new or edited `personas.json`, a changed
`LLM_SYNDATA_GEN`, and a framework upgrade that changed the generator source digest. A full
retrain of the `embedding` head alone costs no LLM call: only cache misses do.

**Diagnose.** Read the plan output (a full retrain prints its reasons) and the per-command miss
lines, which name the fingerprint input that moved.

### One command regenerated and nothing else did

**Cause.** Its seed utterances changed (reordering counts: the list is hashed in order) or its
command name moved. **Not a bug.** In a one-model layout the whole head is still refitted.

### A context shows old evaluation numbers after a selective run (`per_context`)

**Cause.** It was carried forward, not retrained. Its entry in `heldout_evaluation.json` is
re-inserted from the previous report and tagged `carried_forward` with the version it came from.
Those numbers still describe the models this version ships.

**Diagnose.** Cross-check the `Carried forward N context(s)` line and the manifest's
`contexts_carried_forward`.

### The run said "already up to date" but something is visibly wrong at runtime

**Cause.** Model fingerprints do not cover a deleted or corrupt `<command>_param_labeled.json`,
which is why parameter-example refresh runs *before* the no-op check. If intent routing is right
but parameter extraction is wrong, that file is the suspect.

**Fix.** `--regenerate-utterances` also bypasses the parameter-example cache.

### Training failed partway; what state is the workflow in

The previous version is still current and complete. The partial version sits beside it under
`versions/` and is pruned by retention on the next successful publish. Re-running training is
safe and re-plans from the still-current version.

### Publication refused with a training-data or consistency error

**Cause.** Either the training-data safety gate found structurally incomplete data, or the
version's manifest contradicted itself or the version its carried-forward models came from.

**Meaning.** Deliberate. Models were trained but not published, so the live workflow is unchanged.
Fix the reported commands and re-run; do not force-publish the version by hand.

### `fastworkflow learn` did not publish

Its report says why: fewer than 5 distinct teaching texts, "nothing new" (the current version was
already refitted on exactly these rows), no `rows.npz` (retrain first), a layout other than
`embedding`, or a failed gate with its numbers. A failed gate usually means some confirmations
contradict the seeds: review them with `fastworkflow learned list --confirmed` and remove the
wrong ones.

### Rolling back did not change behaviour

`current.json` is authoritative; read it rather than the `current` symlink, which is best-effort.
Confirm the target version still exists: retention keeps only current plus one previous (and a
refit lineage's full train). For `workflow` and `per_context`, restart the running server.

## Cost and time notes

- `embedding`: the head fits in seconds (about 10 s for a bundled example) and is about 100 KB,
  plus about 1 MB of `rows.npz`; the encoder is a shared download of about 418 MB.
- `per_context`: about 276 MB per context, several gigabytes for a large workflow, and
  fine-tuning time per context.
- In every layout the expensive part to rebuild is the generated utterances: hours of LLM calls
  for a large workflow, not reproducible byte-for-byte once regenerated. With both caches warm,
  two independent full retrains produced identical models; see the `evaluate-intent-routing`
  skill.
