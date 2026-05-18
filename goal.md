# Goal

Make condition c2_retriever beat c1_base on LLM-as-judge score, recall@5,
AND dollar cost.

In scope for edits:
  - /home/edwardk/git/skills/.claude/skills/nemo-retriever/SKILL.md and its
    references/ files (primary lever; what the agent reads at runtime).
  - Retriever code anywhere under /home/edwardk/git/skills/nemo_retriever/src/
    (CLI, graph, params, skill_eval runner, etc.) — fair game when the skill
    surface can't express what's needed. Examples already landed: auto-skip
    PageElementDetectionActor; positional INPUT_PATH on `pdf stage
    page-elements`; scenario_prompt_* support in dataset.py.

Out of scope: changing the c1_base condition, the eval manifests, or the
test harness scripts (test_c1_only.sh / base.log are frozen baselines).

## Baseline — DO NOT re-run c1

c1_base artifact paths live in /home/edwardk/git/skills/base.log, grouped by
config:
  - skill_eval_batch_1.yaml  — 46 queries  — 5 c1 runs
  - skill_eval_batch_2.yaml  — 135 queries — 4 c1 runs

To get c1 numbers: parse session_summary.md under
nemo_retriever/artifacts/<artifact_dir>/ for each path in base.log. Aggregate
mean ± σ per config across runs.

## Acceptance

For each of batch_1 and batch_2 independently, all three must hold:
  - mean c2 judge     > mean c1 judge
  - mean c2 recall@5  > mean c1 recall@5
  - mean c2 q_cost    < mean c1 q_cost

## Workflow — rotation, not full sweeps

There are 6 (config × domain) pairs:
  - batch_{1,2} × {vidore_v3_hr, vidore_v3_finance_en, vidore_v3_pharmaceuticals}

Each iteration runs ONE pair (~30 min for a batch_1 domain, ~70–80 min for
batch_2). Rotate through pairs between SKILL.md changes — that's roughly 5×
faster feedback than waiting for a full parallel sweep.

1. Read runs.log; parse session_summary.md for the latest entry per (config,
   domain) pair. Compare against the c1 baseline from base.log.
2. Form ONE concrete hypothesis: "change X should move metric Y by ~Z because
   …". Avoid bundled changes — they're hard to attribute.
3. Apply the change — to SKILL.md, or directly to retriever code under
   nemo_retriever/src/ (both are in scope; see top of file). Pick whichever
   surface fits the change; don't contort SKILL.md to express something the
   CLI/graph could express more naturally.
4. Pick the next pair to test (round-robin, or focus on whatever regressed).
   Trigger as a background bash task:
     ./test_c2_one.sh batch_1 vidore_v3_hr
   Appends the artifact path to runs.log tagged with `# <config> × <domain>`.
5. While it runs, do other work; don't poll. You'll be notified on completion.
6. Parse the new artifact. Confirm or refute the hypothesis. If it regressed
   vs the prior best for this pair, roll back the SKILL.md change.
7. Append a new entry to iteration_log.md (hypothesis, change, artifact path,
   result, verdict). One concise block per iteration. This is the audit trail
   — future sessions read it to avoid re-trying things that already failed.
8. Repeat. After ~6 iterations you've covered every pair at least once.

## Final acceptance

n=1 per pair is noisy (σ_recall@5 ≈ 0.06 on HR). When all 6 pairs look good
under the latest SKILL.md, do ONE full validation sweep:
  ./test_c2_only.sh
which runs 5 iterations × 2 configs (~9h) for proper σ-bands. Acceptance
verdict comes from that sweep, not the rotation iterations.

Always trigger c2 runs as a background bash task; continue working while
they run. Single-pair iteration via ./test_c2_one.sh; full validation sweep
via ./test_c2_only.sh.

## Prior-session findings — read iteration_log.md for detail

The full audit trail is in iteration_log.md. Summary:

**The pattern**: every SKILL.md change that *adds a constraint* triggers an
"agent does more turns" rebound — cache_read inflates faster than per-turn
bytes can be cut, so q_cost rises. *Subtractive* changes are pair-dependent
(help HR, hurt pharma). *Physical hooks* that force compliance break trials
(null judges). The exceptions are changes the agent doesn't perceive —
silent retriever-side improvements. Two have stuck (a third was removed
per user direction — see below):

1. Compact-JSON page-elements sidecars (`pdf/stage.py`: drop duplicate
   `primitives` key, `indent=2`→compact). 67% byte reduction; modest cost
   win on HR.
2. **Deserialize `metadata` from JSON-string to dict in CLI query
   output** (`adapters/cli/main.py:query_command`). The SKILL.md jq
   recipe `\(.metadata.type // "?")` had been *erroring* the entire
   prior session because metadata was stored as a JSON-stringified blob;
   `.type` on a string is null. The agent recovered by extra bash
   calls, inflating q_cost. Fixing this dropped batch_1 cost from +17%
   to +7% — a major win.
   Side effect: now that the agent can see `type=chart|image|table`,
   the SKILL.md's "always prefer the text hit" rule actually fires;
   on b2 this demotes chart-type GT pages below rank 5 and drops
   recall@5 by ~0.03.

**Reverted per user direction**: hardcoded build.nvidia.com endpoint
auto-defaults in `sdk_workflow.py` (for reranker / page-elements / OCR
/ graphic-elements / table-structure). Endpoints must be passed via
CLI arguments — `--reranker-invoke-url`, `--embed-invoke-url`,
`--page-elements-invoke-url`, `--ocr-invoke-url`,
`--graphic-elements-invoke-url`, `--table-structure-invoke-url`. The
reranker is the highest-leverage one; without it batch_1 cost reverts
from +7% back toward +17% over c1. Future sessions should plumb the
URL through `test_c2_one.sh` / `test_c2_only.sh` (set env vars or
extra CLI args on the `retriever skill-eval run` invocation), or via
SKILL.md so the agent passes them in its `retriever query` calls.

**Don't re-try** (each cost 30-90 min of c2 wall-time to refute):

- 80-word `final_answer` cap → all 3 metrics regressed
- jq-only forced sidecar reads (SKILL.md text) → all 3 metrics regressed
- `text[:150]` snippet (vs current `[:200]`) → recall −0.12 on finance
- Strict anti-pattern (single rank-1 PDF only) → HR recall 0.377→0.279
- Rank-1/2/3 escalation → HR recall unchanged at 0.276
- Drop chart/image caution → b2 pharma recall regressed
- Drop multi-entity reminder → b2 pharma recall −0.10
- PreToolUse hook capping Bash at N (tried N=5 and N=8) → null-judge rate explodes
- `Read` deny on `/tmp/pdf_text/**` → pharma cost +16% (agent wastes turns recovering)
- Domain-conditional SKILL.md (HR-only trim) → **overfitting** to eval domains; see feedback memory

**Last formal verdict with reranker auto-enabled** (5-run sweep,
metadata-fix + compact JSON + reranker — the reranker has since been
reverted; if you re-enable it by passing `--reranker-invoke-url`, these
are roughly the numbers to expect):

| batch | recall@5 | judge | cost |
|---|---|---|---|
| batch_1 | 0.528 ± 0.032 vs c1 0.524 ✓ (t=+0.29) | 4.712 vs 4.634 ✓ (t=+1.95) | $0.336 vs $0.315 ✗ +7% (t=+3.01) |
| batch_2 | 0.348 ± 0.044 vs c1 0.378 ✗ (t=−1.52) | 4.578 vs 4.438 ✓ (t=+6.80) | $0.363 vs $0.361 ≈ tied (t=+0.19) |

**Without reranker** (current state — endpoints must be CLI-supplied):
batch_2 reverts toward "all 3 tied with c1". batch_1 cost gap returns
toward +17%. Neither batch passes strict acceptance.

**Untried, possibly fruitful**: (a) hybrid BM25+dense chunking — could
improve intrinsic top-10 recall further. (b) opt-in `--lean-json` for
page-elements that strips null record fields (blocked by downstream
consumers expecting full schema). (c) prune verbose retriever query
output fields the agent doesn't use (`_distance`, `_rerank_score`,
`source`, `pdf_page` blob → keep only `text`, `pdf_basename`,
`page_number`, `metadata`). Each silent retriever-side cut.

**Don't re-try** (added this session): the "prefer text hit → preserve
retriever order" edit. Tried (n=5 sweep both batches): recovered b2
recall (0.348 → 0.379) but regressed b1 recall worse (0.528 →
0.459). The ordering rule is a domain-aware lever — every change to
it helps one batch and hurts the other equally. No single ordering
rule satisfies both.

**Don't overfit to eval domains.** Never `if domain == "vidore_v3_*"` branches
in runner.py or SKILL.md selection. The Vidore-V3 datasets are probes; the
deliverable is a single generalized skill.

**Operational**:
- `test_c2_one.sh --gpu N` and `test_c2_only.sh --gpu N` pin a run to one GPU.
  Bottleneck is Anthropic API throughput, not GPU — parallelize across GPUs
  0/1 for ~2× wall-time.
- Both scripts invoke `uv sync --extra llm` then `uv run --extra llm` so the
  in-process judge has litellm. Removing these guards causes silent `judge=—`.
- Workdirs are uuid-suffixed (`/tmp/skill_eval/c2_retriever_<domain>_<hash>/`),
  safe to parallelize.
- The NVIDIA NIM judge service (Mixtral 8x22B) intermittently 400s with
  "DEGRADED function cannot be invoked" — check `judge_error` in trial JSONs
  before trusting a judge number.
- Anthropic API has hit rate limits during long parallel sessions — symptom
  is many trials with `non-zero exit 1 / output.json not written`. Pause and
  resume later.
