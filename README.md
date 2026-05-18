# nemo-retriever-skill-eval

A standalone harness that benchmarks a [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
skill against an off-the-shelf baseline on a labelled QA manifest.

## Quickstart

### 1. Clone the eval ground-truth manifest

```bash
git clone -b steve/agent_eval_sdg \
    https://gitlab-master.nvidia.com/sthan/retriever-sdg-v3.git \
    ~/git/retriever-sdg-v3
```

### 2. Install

```bash
uv sync                       # core harness
uv sync --extra llm           # + LLM-as-judge support via litellm
```

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) must be on `PATH`
(`claude --version` should work).

### 3. Run the c1_base baseline

```bash
cp src/nr_skill_eval/configs/skill_eval.yaml ~/my_skill_eval.yaml
# Edit ~/my_skill_eval.yaml: set eval_manifest_path + pdf_dirs to paths under ~/git/retriever-sdg-v3

uv run nr-skill-eval run --config ~/my_skill_eval.yaml --conditions c1_base
```

`c1_base` is the off-the-shelf baseline — no skill loaded, so `skill_source_dir`
is not required.

## What it measures

For each `(condition, domain)` pair in your manifest, the harness spawns one
`claude --print` session: turn 1 builds an index (the **setup turn**), turns
2..N answer one labelled question each. Three conditions ship by default:

| condition | skill loaded | slash commands | notes |
|---|---|---|---|
| `c1_base` | no | disabled | stock Claude Code with no skill, full access to whatever is on the host — measures the off-the-shelf experience |
| `c2_retriever` | yes | yes | NL prompt, relies on the skill's description-based auto-discovery |
| `c3_retriever_skill` | yes | yes | explicit `/<skill> ...` slash invocation |

For every query turn the harness records:

- `recall@{1,5,10}` against the manifest's `relevant_pages`
- LLM-as-judge score on a 1–5 scale, against `answer` (optional, via `litellm`)
- Anthropic agent-session cost + token breakdown (in / out / cache_read / cache_create)
- Wall-time, success/failure status, whether the skill fired

Per-condition and per-(condition, domain) rollups are written to
`session_summary.json` and `session_summary.md` in the timestamped session dir.

## Manifest schema

The harness expects a JSON list of dataset entries with these fields (see
`src/nr_skill_eval/dataset.py:load_eval_manifest` for the exact loader):

```json
{
  "primary_eval_id": "<domain>:<n>:<variant>",
  "original_query": "What is …?",
  "sdg_prompt_candidates": {"candidates": [{"variant_id": 0, "prompt": "…paraphrased…"}]},
  "sdg_prompt_validation": {"selected_variant_id": 0},
  "relevant_pages": [{"doc_id": "Foo_Report", "page_number_in_doc": 12, "score": 1}],
  "answer": "<ground-truth answer text>",
  "domain": "my_domain_a",
  "prompt_taxonomy": {"domain_label": "annual reports"}
}
```

`doc_id` matches the source PDF filename without the `.pdf` extension;
`page_number_in_doc` is 0-indexed.

## What's in this repo

```
src/nr_skill_eval/
  cli.py        — typer entrypoint; resolves config + spawns sessions
  runner.py     — per-trial workdir builder + claude subprocess driver
  dataset.py    — manifest + config loaders
  report.py     — per-(condition, domain) aggregation + summary writers
  score.py      — recall@k
  judge.py      — LLM-as-judge wrapper (litellm-backed)
  artifacts.py  — timestamped session dirs + JSON writers
  configs/      — packaged example config
  prompts/      — setup + per-trial prompt templates (NL and slash variants)
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
