# Anonymize PDF filenames in the per-session workdir

Status: draft
Date: 2026-05-26

## Problem

Each domain session runs the agent in a scratch workdir whose `./pdfs/`
directory is a symlink farm over the real PDF basenames. The basenames often
encode information that overlaps with the query (e.g.
`Acme_FreeCashFlow_2024.pdf` for a query about Acme's free cash flow). A
stock coding agent can `ls` or `grep` filenames and reach a plausible answer
without ever performing retrieval over PDF contents, which makes the
benchmark indistinguishable from a filename-keyword game for that subset of
queries.

We want a feature that hides the original basenames from the agent while
keeping every other part of the harness (manifest, scoring, trial JSON,
rescore, summarizer) working unchanged.

## Goals

- The agent sees only opaque filenames inside `./pdfs/`.
- Recall scoring and the LLM judge keep using real `doc_id`s and real
  reference answers, so today's metrics remain directly comparable.
- The feature is opt-in: existing configs and existing artifact directories
  behave exactly as before.
- Trial JSON files written under the new mode retain enough information to
  debug what the agent saw.

## Non-goals

- **Prompt scrubbing.** The paraphrased prompt may itself quote a real
  basename; that is a separate leakage surface and is not handled by this
  feature. Users who need that today must scrub their manifests by hand. A
  follow-up could add a `--scrub-prompts` mode.
- **PDF content anonymization.** A PDF's internal metadata, title, or
  first-page header may still contain the original filename. File *contents*
  are not modified.
- **Domain-label scrubbing.** `prompt_taxonomy.domain_label` is a
  deliberately user-facing part of the setup prompt and is left alone.

## Approach

A single per-session `DocIdMap` is built when the symlink farm is created
and threaded through every component that touches `doc_id`s.

```
config.anonymize_filenames=True
        |
        v
_build_pdf_symlinks(pdf_source, dest, doc_map)
        |
        |  for each real_basename (without .pdf):
        |    anon = "doc_" + sha1(real_basename)[:10]
        |    symlink real.pdf -> dest / f"{anon}.pdf"
        |
        +-->  pdfs/doc_<hash>.pdf     (what the agent sees)
        +-->  DocIdMap {real <-> anon}  (in-memory; persisted to session_dir)

agent -> output.json with anonymized doc_ids
        |
        v
_parse_output_json()  (unchanged; returns whatever the agent wrote)
        |
        v
_run_one_turn() post-process:
   for each ranked entry: deanonymize via DocIdMap;
     entry["doc_id"]            = real
     entry["anonymized_doc_id"] = anon   (sidecar)
        |
        v
score.recall_at_k()  (unchanged; sees real doc_ids)
TrialResult JSON     (real doc_ids + optional anonymized_doc_id field)
```

When the feature is off (`doc_map is None`), every new code path
short-circuits and the output is byte-identical to today.

### Naming scheme

`anon = "doc_" + hashlib.sha1(real_basename.encode()).hexdigest()[:10]`

- Stable across runs over the same source PDFs (helps with debugging and
  cross-run inspection).
- 40 bits of entropy in the prefix: collisions across a few thousand PDFs
  are astronomically unlikely. The `DocIdMap` constructor raises on
  collision rather than silently overwriting a symlink.

### Round-trip schema

Each entry in `TrialResult.ranked_retrieved` gains an optional sidecar
field when anonymization is on:

```json
{"doc_id": "Acme_10K_2024", "page_number": 47, "rank": 1,
 "anonymized_doc_id": "doc_3f9a1b2c0d"}
```

`doc_id` continues to hold the real id, so `score.py`, `rescore`, and any
external tool that reads trial JSON works without changes. The sidecar
field is omitted when the feature is off.

## Components & interfaces

### `runner.py` — new `DocIdMap`

```python
@dataclass(frozen=True)
class DocIdMap:
    real_to_anon: dict[str, str]   # "Acme_10K_2024" -> "doc_3f9a1b2c0d"
    anon_to_real: dict[str, str]

    @classmethod
    def from_pdf_dir(cls, pdf_source: Path) -> "DocIdMap":
        """Build a stable hash map for every *.pdf in pdf_source.

        Raises on collision in the 10-hex-char prefix.
        """

    def anonymize(self, real_stem: str) -> str: ...
    def deanonymize(self, anon_stem: str) -> str:
        """Return the real stem, or `anon_stem` unchanged if not in map."""
```

### `runner.py` — `_build_pdf_symlinks` gains an optional map

```python
def _build_pdf_symlinks(
    pdf_source: Path, dest: Path, doc_map: DocIdMap | None = None
) -> None:
```

When `doc_map` is `None` the existing behavior is preserved exactly. When
supplied, each symlink target name is `f"{doc_map.anonymize(stem)}.pdf"`.

### `runner.py` — `_build_session_workdir` returns `(workdir, doc_map)`

Returns `(workdir, None)` when anonymization is off. The map is built once
per session and reused across all turns so the agent sees stable filenames
inside the session.

### `runner.py` — `_run_one_turn` translates `ranked_retrieved`

A new helper, called only after `_parse_output_json`, walks the parsed
ranked list and rewrites each entry to
`{doc_id: real, page_number, rank, anonymized_doc_id: anon}`. When
`doc_map is None` the helper is a no-op.

### `runner.py` — `run_session` signature

Gains `doc_map: DocIdMap | None = None`; threads it into `_run_one_turn`.
The caller in `cli.py` builds the workdir, gets the map back, and passes
it through.

### `cli.py` / `dataset.py` — new config option

- YAML: top-level `anonymize_filenames: false` (default).
- CLI: `--anonymize-filenames / --no-anonymize-filenames` overrides the
  config value for a run.
- The chosen value is echoed in the run header alongside the other
  settings.

### Mapping persistence

When anonymization is on, write
`<session_dir>/trials/<agent>/<condition>/<domain>/doc_id_mapping.json`
once per domain session:

```json
{ "real_to_anon": { "Acme_10K_2024": "doc_3f9a1b2c0d", ... } }
```

Used for debugging only; nothing in the harness reads it back.

### Rescore

No changes. Trial JSON already contains real doc_ids and the real
ground-truth answer; `score.py` and the judge work as today.

## Error handling

- **Hash-prefix collision** in `DocIdMap.from_pdf_dir`: raise with both
  conflicting basenames named. Session aborts before any subprocess is
  launched.
- **Unknown anonymized id** reported by the agent (hallucination or attempt
  to outsmart the renaming): `deanonymize` returns the string unchanged;
  `doc_id` and `anonymized_doc_id` are both set to that string. Recall is
  naturally 0 for that entry, which is the correct outcome.
- **Anonymization off**: every new code path short-circuits on
  `doc_map is None`. Trial JSON, scoring, rescore, and external tooling are
  byte-identical to today.

## Edge cases (not solved here)

- Paraphrased prompt text that mentions the original basename. (See
  Non-goals.)
- PDF metadata / first-page headers containing the original filename.
- `prompt_taxonomy.domain_label` hints.

These are documented limitations of the "filenames-only" scope.

## Testing

- **Unit — `DocIdMap`**: real<->anon round-trip; collision raises;
  `deanonymize` of an unknown id returns input unchanged.
- **Unit — `_build_pdf_symlinks`**: with a `DocIdMap`, produces symlinks
  named `doc_<hash>.pdf` resolving to the real PDFs.
- **Unit — `_run_one_turn` post-process**: rewrites a sample
  `ranked_retrieved`, including the unknown-id pass-through case.
- **Integration (anonymize on)**: end-to-end on `skill_eval_batch_test.yaml`
  with `anonymize_filenames: true` against a fixture domain of <=3 PDFs
  under a stubbed agent. Verify trial JSON contains real `doc_id`s, an
  `anonymized_doc_id` sidecar field, and that `doc_id_mapping.json` is
  written under the trial dir.
- **Integration (anonymize off)**: same test with `anonymize_filenames:
  false`. The new fields are absent and no mapping file is written; trial
  JSON shape is byte-identical to today.

## Migration & backward compatibility

- Default value is `false`. Existing configs and existing artifact
  directories are unaffected.
- The new `anonymized_doc_id` field in `ranked_retrieved` entries is
  additive and optional, so existing consumers (scoring, rescore, the
  Markdown report) continue to work unchanged.
- `rescore` against an artifact directory produced with anonymization on
  works without changes because `doc_id` and the ground-truth answer are
  already real values on disk.
