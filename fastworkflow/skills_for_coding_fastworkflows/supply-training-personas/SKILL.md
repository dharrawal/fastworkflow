---
name: supply-training-personas
description: >-
  Decide who writes a fastWorkflow's synthetic training utterances, by adding a `personas.json`
  that either supplies the personas outright or conditions the PersonaHub draw on the
  application's domain keywords. Covers the file schema and where it is discovered, choosing
  between the two modes, the persona id rules that keep held-out evaluation honest, sizing a
  domain-filtered pool, and the full retrain a persona change forces. Use when generated
  utterances read nothing like the workflow's real users, when a workflow must train without
  downloading PersonaHub, or when authoring or reviewing a `personas.json`.
---

# Supplying training personas

## The mechanism being changed

`fastworkflow train` asks an LLM (`LLM_SYNDATA_GEN`) to write extra utterances for each command
while role-playing a persona. By default the persona is drawn from the whole PersonaHub corpus
(about 200k rows) and conditioned on nothing, so a retail workflow and a library
workflow get the same population of astrophysicists, sommeliers and marine biologists.

`personas.json` states who the application's users are. It changes the *pool*; the draw from it
stays deterministic per training seed and command.

In the default `embedding` layout the intent head is fitted on those generated utterances plus
each command's seed utterances (each counted three times), its humanised name and the first
sentence of its `Signature` docstring. Personas shape only the generated rows.

## Two modes, one file

| `personas.json` contains | Source | Effect |
|---|---|---|
| a non-empty `personas` list | `app_supplied` | Your personas are used verbatim. **PersonaHub is never loaded or downloaded**, and the `datasets` package is not needed. |
| `domain_keywords` (or `domain`) and no `personas` | `domain_conditioned` | PersonaHub is filtered to rows mentioning those keywords and the usual sample is drawn from them. |
| neither | | an error, not a fallback |

A `personas` list wins over `domain_keywords`: honouring both would mix generic rows into a
curated set.

## The file

`<workflow>/personas.json`, next to `intent_benchmark.json`:

```json
{
  "schema_version": 1,
  "domain": "retail order management",
  "domain_keywords": ["retail", "e-commerce", "shopper", "customer service"],
  "personas": [
    {"id": "impatient-shopper", "persona": "A shopper who orders often, tracks every delivery, and types in short fragments."},
    {"id": "support-agent", "persona": "A support agent handling returns and refunds all day; uses internal jargon."}
  ]
}
```

- A **bare JSON array** of personas is accepted, so a quick file needs no wrapper.
- An entry may be a **bare string**; its id becomes its position.
- An object entry takes its text from `persona` or `text`, and its id from `id` or its position.
- `schema_version` may be omitted; any value other than `1` is an error.
- `domain` with no `domain_keywords` is enough on its own: its words longer than two characters,
  lowercased, become the keywords. That is the least-effort form of the feature.

`SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE` points at a persona file outside the workflow folder, the
one route that needs no change to the workflow. Set it in `fastworkflow.env` (a process
environment value is read when the env file does not set it). It applies to every workflow the
run trains that has no `personas.json` of its own, including child workflows under `_workflows/`.

Discovery order, per workflow: `<workflow>/personas.json`, then
`SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE`, then plain PersonaHub.

## Rule 1: a broken file fails the run, a missing env path does not

A `personas.json` that exists but cannot be used (bad JSON shape, wrong `schema_version`, an
entry with no text, a bad or duplicate id, no personas and no keywords) raises
`PersonaConfigError` and stops training. A file that states an intent about the training data
must not silently train on the generic draw instead. The message for an empty file says the fix:
delete the file to use the default draw.

A `SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE` naming a path that does not exist is treated as unset,
and the run uses PersonaHub **silently**. If you use the env route, confirm the file was read
from the `Persona source:` log line.

## Rule 2: ids must not break the persona holdout

Training holds out whole personas to score generalisation, and it finds the personas behind a
generated utterance by parsing persona ids. Your ids are prefixed `app:` automatically, so they
never collide with PersonaHub row numbers or the reserved ids. Two rules remain, both checked at
load time:

- **No `+` in an id.** `+` joins the personas of a composite id; an id containing it would be
  split into fragments and the holdout would leak.
- **Ids are unique.** Otherwise provenance cannot say which persona wrote what.

## Rule 3: a domain-filtered pool has a floor, and padding is reported

A keyword matches a whole word of the lowercased persona text; a keyword containing a space or a
hyphen matches as a substring. A row matching any keyword survives and keeps its PersonaHub id.

The pool should hold at least `4 x SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` rows (at least 4 when
the setting is absent), or every command draws nearly the same handful. Each outcome is logged
once per run as a `Persona source:` line:

| Matched rows | What happens | What to do |
|---|---|---|
| enough | the pool is exactly the matched rows | nothing |
| some, under the floor | topped up with unfiltered rows to the floor; the matched rows are drawn first, so a command asking for more personas than matched gets unconditioned ones for the rest | add keywords, or supply `personas` |
| none | the full corpus is used: "Domain conditioning had no effect on this run" | fix the keywords; they are almost certainly misspelt or too specific |

Padding instead of failing is deliberate (a typo must not stop a long run), so the log line is
the only sign the feature did nothing. Read it.

Supplying fewer than 4 personas logs a warning for the same reason: the utterances share a voice
and the holdout has little to hold out.

## Rule 4: adding, editing or removing this file retrains everything

The persona source (its kind, the persona text or keyword list, and the framework's pool code)
is an input of two keys:

1. **The utterance-cache key of every command.** Every command's utterances are generated again,
   which **costs LLM calls** and is most of the run's time. A cache miss is logged with the input
   that moved (`persona_source`). Returning to a source used recently (for example deleting the
   file to go back to PersonaHub) reuses that source's cached utterances while they are among
   the 3 most recent variants kept per command.
2. **The training signature's global header.** `fastworkflow train` chooses a full retrain. In the
   `embedding` layout that is one refit of the head, which takes seconds once the utterances
   exist; in the `per_context` and `workflow` layouts it fine-tunes every model again.

Do not add a `personas.json` in the same change as a seed-utterance edit you want to measure:
you will not be able to attribute the result.

## What this does not do

It is not a measured improvement to intent detection. Domain-conditioned personas are a
hypothesis, unmeasured in this repository, and not measurable from one paired run: the
utterances come from a live LLM, so two runs at the same seed produce different training data.
Treat a persona change as a change of training data, not a fix. To claim it helped, use the
paired-build discipline in the `evaluate-intent-routing` skill.

## Verifying it took effect

```
- [ ] The training log has a `Persona source:` line naming the source you configured
- [ ] For a domain-conditioned pool, that line reports a match count, not a fallback or a top-up
- [ ] `___command_info/training_provenance.json` records the source (`generator_config.persona_source`) for each command
- [ ] The cache summary line (`Utterance cache ... X reused, Y generated, Z written`) shows the commands generated, not reused
```

## Authoring checklist

```
- [ ] personas.json is at the workflow root
- [ ] Exactly one mode is meant: a personas list, or domain keywords (a list silently overrides keywords)
- [ ] No id contains '+'; ids are unique
- [ ] Each persona describes how someone TALKS, not just who they are
- [ ] At least 4 personas if supplying them outright
- [ ] Domain keywords are common words that appear in persona descriptions, not product names
- [ ] The regeneration this forces is budgeted for, and isolated from other changes being measured
```

## Related

- What a full retrain and the caches involve: the `train-and-publish-models` skill.
- Measuring whether the resulting model is any good: the `evaluate-intent-routing` skill.
- Two commands no persona could phrase apart: personas will not fix it; see the
  `detect-duplicate-capabilities` skill.
