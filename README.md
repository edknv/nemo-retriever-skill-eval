# skill-eval

A harness that measures stock [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
on a labelled QA manifest. For each domain in your manifest, it spawns one
`claude --print` session: turn 1 builds an index over the domain's PDFs (the
**setup turn**), turns 2..N answer one labelled question each. The agent has
no skill loaded and `--disable-slash-commands` is set — it works only with
stock tools (`Read`, `Grep`, `Bash`, etc.) plus whatever else is installed
on the host.

Every run is tagged as the single condition `c1_base`. The CLI no longer
supports condition selection, skill loading, or slash-command variants.

For every query turn the harness records:

- `recall@{1,5,10}` against the manifest's `relevant_pages`
- LLM-as-judge score on a 1–5 scale, against `answer` (optional, via `litellm`)
- Anthropic agent-session cost + token breakdown (in / out / cache_read / cache_create)
- Wall-time, success/failure status

After each domain session, a separate Claude call summarizes the tool-use
trace from the session JSONL so you can see *what* the agent did.

Per-domain rollups land in `session_summary.json` and `session_summary.md`
in the timestamped session dir.

## Quickstart

### 1. Install

```bash
uv sync                       # core harness
uv sync --extra llm           # + LLM-as-judge support via litellm
```

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) must be on
`PATH` (`claude --version` should work).

### 2. Run

```bash
# Copy the packaged config and edit it
cp src/skill_eval/configs/skill_eval.yaml ~/my_skill_eval.yaml
# (set eval_manifest_path and pdf_dirs; optionally testdata_prefixes)

uv run skill-eval run --config ~/my_skill_eval.yaml
```

`uv run` activates the project's virtualenv for the wrapped command, so you
don't need to `source .venv/bin/activate` manually.

To run on a subset of domains:

```bash
uv run skill-eval run --config ~/my_skill_eval.yaml --domains my_domain_a
```

Supported `run` options are `--config`, `--eval-manifest`, `--domains`, and
`--artifacts-root`. Old `--conditions`, `skill_source_dir`, `c2_retriever`,
and `c3_retriever_skill` workflows are not part of this CLI.

## Manifest schema

The harness expects a JSON list of dataset entries with these fields (see
`src/skill_eval/dataset.py:load_eval_manifest` for the exact loader):

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
src/skill_eval/
  cli.py              — typer entrypoint; resolves config + spawns sessions
  runner.py           — per-session workdir builder + claude subprocess driver
  dataset.py          — manifest + config loaders
  report.py           — per-domain aggregation + summary writers
  score.py            — recall@k
  judge.py            — LLM-as-judge wrapper (litellm-backed)
  trace_summarizer.py — claude-CLI-backed tool-use narrator
  artifacts.py        — timestamped session dirs + JSON writers
  configs/            — packaged example config
  prompts/            — setup + per-trial prompt templates
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
