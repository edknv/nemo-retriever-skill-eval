# skill-eval - benchmarking stock Claude Code over PDFs

`skill-eval run` measures how well stock [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
answers labelled questions over a folder of PDFs. It does not load a skill,
does not enable slash commands, and does not run multiple benchmark conditions.

Each domain in the manifest runs as one Claude Code session:

- Turn 1 is a setup turn over `./pdfs/`.
- Turns 2..N answer one labelled question each.
- Every result is tagged with the single condition `c1_base`.
- Claude is launched with `--disable-slash-commands`.

For every query turn, the harness records:

- `recall@{1,5,10}` against the manifest's `relevant_pages`.
- Optional LLM-as-judge score on a 1-5 scale against the manifest's `answer`.
- Anthropic agent-session cost and token breakdown.
- Wall time, status, final answer, and the ranked pages the agent reported.

After each domain session, an optional Claude call summarizes the tool-use trace
from the Claude Code session JSONL. Per-domain and overall rollups are written
to `session_summary.json` and `session_summary.md` in a timestamped artifact
directory.

## Table Of Contents

- [Prerequisites](#prerequisites)
- [Inputs](#inputs)
- [1. Make the PDF tree reachable](#1-make-the-pdf-tree-reachable)
- [2. Supply an agent-eval manifest](#2-supply-an-agent-eval-manifest)
- [3. Author your config](#3-author-your-config)
- [4. Run the benchmark](#4-run-the-benchmark)
- [CLI reference](#cli-reference)
- [Output layout](#output-layout)
- [Interpreting the summary](#interpreting-the-summary)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

## Prerequisites

- `uv` for environment and dependency management.
- `claude` on `PATH`; `claude --version` should work before starting a run.
- A Claude account or API access configured for `claude --print`.
- Claude Code autorun / permission access. The runner launches non-interactive
  Claude Code subprocesses with `--permission-mode bypassPermissions` and
  `--allow-dangerously-skip-permissions`.
- Disk for per-domain scratch workdirs under `/tmp/skill_eval/` by default.
  Each workdir contains a `pdfs/` symlink farm, `.claude/`, and whatever
  search artifacts the agent creates. It is deleted after the domain session.
- Optional `NVIDIA_API_KEY` for LLM-as-judge scoring via `litellm`.

Install the core package:

```bash
uv sync
```

Install judge support:

```bash
uv sync --extra llm
```

## Inputs

`skill-eval` needs three caller-supplied inputs:

1. A directory of PDFs for each manifest domain.
2. An agent-eval manifest JSON list describing queries, prompts, ground-truth
   pages, and ground-truth answers.
3. A YAML config binding the manifest to the PDF directories.

The packaged config at `src/skill_eval/configs/skill_eval.yaml` provides
defaults for model, budget, timeout, judge, and summarizer settings. You must
fill in `eval_manifest_path` and `pdf_dirs`.

## 1. Make The PDF Tree Reachable

The runner does not copy PDFs. For each domain, it creates a scratch workdir
and symlinks every `*.pdf` from the configured source directory into
`<workdir>/pdfs/`.

For ViDoRe v3, a typical PDF root is split by domain:

```text
vidore_v3_computer_science
vidore_v3_energy
vidore_v3_finance_en
vidore_v3_finance_fr
vidore_v3_hr
vidore_v3_industrial
vidore_v3_pharmaceuticals
vidore_v3_physics
```

The `pdf_dirs` keys in your config must match the `domain` strings in the
manifest exactly. They do not need to match filesystem directory names.

## 2. Supply An Agent-Eval Manifest

The manifest is a JSON list. Each item describes one query. The loader is
dataset-agnostic and accepts these fields:

| Field | Type | Purpose |
|---|---|---|
| `original_query` | string | Raw user question, used for judging context. |
| `sdg_prompt_candidates.candidates` | list of `{variant_id, prompt}` | Paraphrased prompt variants. |
| `sdg_prompt_validation.selected_variant_id` | int, optional | Chosen prompt variant; falls back to the first candidate. |
| `relevant_pages` | list of `{doc_id, page_number_in_doc, score}` | Ground-truth pages for recall. |
| `answer` | string | Ground-truth answer for the optional judge. |
| `domain` | string | Joins the entry to a `pdf_dirs` key. |
| `prompt_taxonomy.domain_label` | string | Human-readable label injected into the setup prompt. |
| `primary_eval_id` | string, optional | Stable query id; falls back to `eval_base_id`, then list position. |

The newer `scenario_prompt_candidates` and `scenario_prompt_validation` keys
are accepted as aliases. Entries with `prompt_export_status` not in
`(None, "exported")` are skipped, as are entries with no usable paraphrased
prompt.

Example entry:

```json
{
  "primary_eval_id": "vidore_v3_finance_en:42:variant-1",
  "domain": "vidore_v3_finance_en",
  "prompt_taxonomy": {"domain_label": "English-language corporate finance filings"},
  "original_query": "What was Acme Corp's free cash flow in FY2024?",
  "sdg_prompt_candidates": {
    "candidates": [
      {
        "variant_id": 1,
        "prompt": "Look at the PDFs at ./pdfs/ and tell me Acme Corp's FY2024 free cash flow."
      }
    ]
  },
  "sdg_prompt_validation": {"selected_variant_id": 1},
  "relevant_pages": [
    {"doc_id": "Acme_10K_2024", "page_number_in_doc": 47, "score": 1}
  ],
  "answer": "$3.2B, per the FY2024 cash flow statement."
}
```

`doc_id` is the PDF basename without `.pdf`. `page_number_in_doc` is
0-indexed.

## 3. Author Your Config

Copy the packaged config and edit it:

```bash
cp src/skill_eval/configs/skill_eval.yaml ~/datasets/skill_eval.yaml
```

Example config:

```yaml
eval_manifest_path: ~/datasets/vidore_v3/agent_eval_manifest.json

pdf_dirs:
  vidore_v3_computer_science:  /datasets/vidore_v3/vidore_v3_computer_science
  vidore_v3_energy:            /datasets/vidore_v3/vidore_v3_energy
  vidore_v3_finance_en:        /datasets/vidore_v3/vidore_v3_finance_en
  vidore_v3_finance_fr:        /datasets/vidore_v3/vidore_v3_finance_fr
  vidore_v3_hr:                /datasets/vidore_v3/vidore_v3_hr
  vidore_v3_industrial:        /datasets/vidore_v3/vidore_v3_industrial
  vidore_v3_pharmaceuticals:   /datasets/vidore_v3/vidore_v3_pharmaceuticals
  vidore_v3_physics:           /datasets/vidore_v3/vidore_v3_physics

testdata_prefixes:
  - test-data/vidore_v3/

agent_model: claude-opus-4-7
per_trial_budget_usd: 5.0
per_trial_timeout_s: 600
per_trial_workdir_root: /tmp/skill_eval

judge:
  enabled: true
  model: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1
  api_base: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_API_KEY

summarizer:
  enabled: true
  model: claude-opus-4-7
```

Things to check:

- `pdf_dirs` keys must exactly match the manifest's `domain` values.
- Each `pdf_dirs` value must be a directory containing PDFs, not a glob.
- If paraphrased prompts contain source-tree paths, add those prefixes to
  `testdata_prefixes` so prompt text resolves to `./pdfs/...` in the workdir.
- The single-path key `pdf_dir` is still honored as a fallback for one-domain
  configs.

## 4. Run The Benchmark

Smoke-test one domain first:

```bash
uv run skill-eval run \
  --config ~/datasets/skill_eval.yaml \
  --domains vidore_v3_finance_en
```

Run all domains in the manifest:

```bash
uv run skill-eval run --config ~/datasets/skill_eval.yaml
```

Run with judge support installed:

```bash
uv run --extra llm skill-eval run --config ~/datasets/skill_eval.yaml
```

Domains execute sequentially. Each domain is one Claude session, with one setup
turn followed by one query turn per matching manifest entry.

Example console shape:

```text
Loaded 412 dataset entries.
Domains in this run: ['vidore_v3_finance_en'] (52 entries total)
Session dir: /repo/artifacts/skilleval_20260519_170000_UTC
Starting session for vidore_v3_finance_en - setup + 52 query turns (pdfs=/datasets/...)
  turn 1 [vidore_v3_finance_en] setup: status=ok tokens(in/out/cache_r)=... cost=$0.041 retrieved=0
  turn 2 [vidore_v3_finance_en] entry_id=1 query_id=vidore_v3_finance_en:1:variant-1: status=ok ... judge=4
Recall for vidore_v3_finance_en: recall@1=0.115  recall@5=0.327  recall@10=0.481
Cleaned up workdir for vidore_v3_finance_en
```

## CLI Reference

```text
skill-eval run [OPTIONS]
```

| Option | Default | Notes |
|---|---|---|
| `--config PATH` | packaged config | Strongly recommended; packaged config exits until dataset paths are filled in. |
| `--eval-manifest PATH` | `cfg.eval_manifest_path` | Overrides the config manifest path for this invocation. |
| `--domains LIST` | all domains in the manifest | Comma-separated subset. Unknown domains exit with code `2`. |
| `--artifacts-root PATH` | `./artifacts/` | Where the timestamped session directory is created. |

There is no condition selector. The CLI always runs the stock `c1_base` path.

Configuration errors exit with code `2`, including missing `claude`, missing
manifest path, malformed `testdata_prefixes`, unknown domain, or missing PDF
directory.

## Output Layout

Each run writes a timestamped session directory:

```text
<artifacts-root>/skilleval_<timestamp>/
|-- config.yaml
|-- session_summary.json
|-- session_summary.md
`-- trials/
    `-- c1_base/
        `-- vidore_v3_finance_en/
            |-- c1_base_vidore_v3_finance_en_setup_t1.json
            |-- c1_base_vidore_v3_finance_en_e1_t2.json
            `-- ...
```

Per-trial JSON files serialize the `TrialResult` dataclass: status, duration,
token usage, cost, `final_answer`, `ranked_retrieved`, judge score, errors,
domain, session id, and optional tool-use summary on the setup turn.

Scratch workdirs under `per_trial_workdir_root` are deleted after each domain
session finishes. The Claude Code transcript JSONL remains under
`~/.claude/projects/` when available, and the harness reads it before cleanup
for the optional tool-use summary.

## Interpreting The Summary

`session_summary.md` contains an overall row and one row per domain:

- `success_rate`: fraction of turns that exited cleanly.
- `recall@1`, `recall@5`, `recall@10`: macro-averaged recall over
  `(doc_id, page_number)` pairs in `ranked_retrieved`.
- `judge`: mean judge score with sample size, or `-` when judging is disabled
  or unavailable.
- `q_input`, `q_output`, `q_cache_read`, `q_cache_create`: mean per-query
  Claude Code session token usage.
- `q_cost`: mean per-query turn USD cost.

The setup table reports one-time setup-turn cost summed across domains. The
session totals table reports setup plus all query turns. If summarization is
enabled, a "Tool-use summaries" section describes the agent's tools and
strategy per domain.

## Troubleshooting

**`Error: \`claude\` CLI is not on PATH`** - install Claude Code and confirm
`which claude` resolves before running.

**`config 'pdf_dirs' is missing an entry for domain '<X>'`** - add a matching
key to `pdf_dirs`, or use `--domains` to skip that subset.

**`PDF directory '...' for domain '...' does not exist or is not a directory`**
- resolve the configured path manually and update the config.

**Judge is disabled** - if `$NVIDIA_API_KEY` is unset, the run still succeeds
and recall metrics are written; the judge column is empty. Install with
`uv sync --extra llm` and export the configured API key env var to enable it.

**Tool-use summary is skipped** - the Claude Code session JSONL was not found
or the summarizer call failed. Core trial results and metrics are unaffected.

**Agent failed to write `./output.json`** - the trial JSON will show
`status="extraction_failed"` and `extraction_method` as `missing` or
`invalid_json`. Re-run the affected domain with `--domains <domain>` to capture
a fresh trace.

**Runs take too long** - use `--domains` to run a smaller subset. Domains are
independent and artifact directories do not collide.

## Repository Layout

```text
src/skill_eval/
  cli.py              - Typer entrypoint; resolves config and spawns sessions
  runner.py           - workdir builder and Claude subprocess driver
  dataset.py          - manifest and config loaders
  report.py           - aggregation and summary writers
  score.py            - recall@k
  judge.py            - LLM-as-judge wrapper
  trace_summarizer.py - Claude-CLI-backed tool-use narrator
  artifacts.py        - timestamped session dirs and JSON writers
  configs/            - packaged example config
  prompts/            - setup and per-query prompt templates
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
