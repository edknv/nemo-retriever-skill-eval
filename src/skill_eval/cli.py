# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`skill-eval run` benchmark."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from skill_eval.artifacts import create_session_dir
from skill_eval.dataset import DatasetEntry, load_config, load_eval_manifest
from skill_eval.report import overall_recall, write_summary
from skill_eval.runner import (
    BASE_CONDITION,
    DEFAULT_AGENT_MODELS,
    SUPPORTED_AGENTS,
    UNSCORABLE_JUDGE_ERRORS,
    TrialResult,
    _apply_judge,
    archive_session_log,
    cleanup_session_workdir,
    extract_compact_trace,
    run_session,
    save_trial,
)

app = typer.Typer(help="Measure stock coding agents on a labelled QA manifest.")
logger = logging.getLogger(__name__)


@app.callback()
def _main() -> None:
    """Keep `run` as an explicit subcommand even though it's currently the only one."""


def _resolve_pdf_source(
    cfg: dict,
    domain: str,
) -> Path:
    pdf_dirs = cfg.get("pdf_dirs")
    if isinstance(pdf_dirs, dict):
        if domain not in pdf_dirs:
            raise typer.BadParameter(
                f"config 'pdf_dirs' is missing an entry for domain '{domain}'. "
                f"Known domains: {sorted(pdf_dirs.keys())}"
            )
        return Path(str(pdf_dirs[domain])).expanduser().resolve()
    if cfg.get("pdf_dir"):
        return Path(str(cfg["pdf_dir"])).expanduser().resolve()
    raise typer.BadParameter("config must define either 'pdf_dirs' (per-domain map) or 'pdf_dir'.")


def _build_judge(cfg: dict, *, manifest_path: Optional[Path] = None) -> Optional[Any]:
    """Construct a ``JudgeContext`` from ``cfg['judge']`` or return ``None``.

    Skips silently (with a console note) when the API key env var is unset,
    so runs work end-to-end without network access. The two judge-prompt
    paths default to the manifest's parent directory if not overridden in
    the config; only the path for the mode a given entry uses must actually
    exist on disk. When ``judge.legacy_enabled`` is true (the default), the
    hardcoded-prompt ``LLMJudge`` is also constructed and run in parallel
    so scores remain comparable to runs that pre-date the scenario-aware
    judge.
    """
    judge_cfg = cfg.get("judge") or {}
    if not judge_cfg.get("enabled", True):
        typer.echo("Judge disabled by config (judge.enabled=false).")
        return None
    api_key_env = str(judge_cfg.get("api_key_env", "NVIDIA_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        typer.echo(f"Judge disabled: ${api_key_env} is not set in the environment.")
        return None

    try:
        from skill_eval.judging import LiteLLMChatClient
    except ImportError as exc:
        typer.echo(f"Judge disabled: failed to import LiteLLMChatClient ({exc}). Install skill-eval[llm].")
        return None
    from skill_eval.runner import JudgeContext

    model = str(judge_cfg.get("model", "openai/nvidia/nvidia/llama-3.3-nemotron-super-49b-v1.5"))
    api_base = judge_cfg.get("api_base")
    client_kwargs: dict[str, Any] = dict(model=model, api_base=api_base, api_key=api_key)
    if "temperature" in judge_cfg:
        client_kwargs["temperature"] = float(judge_cfg["temperature"])
    if "max_tokens" in judge_cfg:
        client_kwargs["max_tokens"] = int(judge_cfg["max_tokens"])
    client = LiteLLMChatClient.from_kwargs(**client_kwargs)

    legacy_judge = None
    if judge_cfg.get("legacy_enabled", True):
        try:
            from skill_eval.judge import LLMJudge
        except ImportError as exc:
            typer.echo(f"Legacy judge disabled: failed to import LLMJudge ({exc}).")
        else:
            legacy_kwargs: dict[str, Any] = dict(model=model, api_base=api_base, api_key=api_key)
            if "temperature" in judge_cfg:
                legacy_kwargs["temperature"] = float(judge_cfg["temperature"])
            if "max_tokens" in judge_cfg:
                legacy_kwargs["max_tokens"] = int(judge_cfg["max_tokens"])
            legacy_judge = LLMJudge.from_kwargs(**legacy_kwargs)

    simple_path = judge_cfg.get("simple_prompt_path")
    scenario_path = judge_cfg.get("scenario_prompt_path")
    manifest_dir: Optional[Path] = None
    if manifest_path is not None:
        manifest_dir = Path(manifest_path).expanduser().resolve().parent
    simple_resolved = (
        Path(str(simple_path)).expanduser().resolve()
        if simple_path
        else (manifest_dir / "llm_scorer_prompt.md" if manifest_dir is not None else None)
    )
    scenario_resolved = (
        Path(str(scenario_path)).expanduser().resolve()
        if scenario_path
        else (manifest_dir / "llm_scenario_scorer_prompt.md" if manifest_dir is not None else None)
    )
    ctx = JudgeContext(
        client=client,
        simple_prompt_path=str(simple_resolved) if (simple_resolved and simple_resolved.is_file()) else None,
        scenario_prompt_path=str(scenario_resolved) if (scenario_resolved and scenario_resolved.is_file()) else None,
        legacy_judge=legacy_judge,
    )
    typer.echo(
        "Judge enabled: model={m}  simple_prompt={s}  scenario_prompt={c}  legacy={legacy}".format(
            m=client.model,
            s=ctx.simple_prompt_path or "(missing)",
            c=ctx.scenario_prompt_path or "(missing)",
            legacy="on" if ctx.legacy_judge is not None else "off",
        )
    )
    return ctx


def _build_trace_summarizer(cfg: dict) -> Optional[Any]:
    """Construct a ``TraceSummarizer`` from ``cfg['summarizer']`` or return ``None``.

    Shells out to the ``claude`` CLI, so it reuses Claude Code's auth — no
    extra API key required. Independent of the judged agent and judge model:
    judging stays on a deterministic cheap model for cross-run score
    consistency; summarization can use a stronger narrator.
    """
    sum_cfg = cfg.get("summarizer") or {}
    if not sum_cfg.get("enabled", True):
        typer.echo("Trace summarizer disabled by config (summarizer.enabled=false).")
        return None
    if shutil.which("claude") is None:
        typer.echo("Trace summarizer disabled: `claude` CLI is not on PATH.")
        return None
    from skill_eval.trace_summarizer import TraceSummarizer

    summarizer = TraceSummarizer.from_kwargs(
        model=str(sum_cfg.get("model", "claude-opus-4-7")),
    )
    typer.echo(f"Trace summarizer enabled: model={summarizer.model}")
    return summarizer


def _resolve_agent(value: str) -> str:
    agent = value.strip().lower()
    if agent not in SUPPORTED_AGENTS:
        raise typer.BadParameter(f"agent must be one of {', '.join(SUPPORTED_AGENTS)}")
    return agent


def _resolve_agent_model(cfg: dict, agent: str, override: Optional[str]) -> str:
    if override:
        return override
    models = cfg.get("agent_models")
    if isinstance(models, dict) and models.get(agent):
        return str(models[agent])
    if cfg.get("agent_model"):
        return str(cfg["agent_model"])
    return DEFAULT_AGENT_MODELS[agent]


def _resolve_domain_label(entries: list[DatasetEntry], cfg: dict, domain: str) -> str:
    """Pick a human-readable label for the setup prompt.

    Prefers the manifest-provided ``domain_label`` carried on the entry. Falls
    back to an optional ``domain_labels`` map in the config, then to ``"PDFs"``.
    """
    for e in entries:
        if e.domain == domain and e.domain_label:
            return e.domain_label
    labels = cfg.get("domain_labels")
    if isinstance(labels, dict) and domain in labels:
        return str(labels[domain])
    return "PDFs"


@app.command("run")
def run_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to a skill_eval.yaml; defaults to the packaged config (copy and edit it).",
    ),
    eval_manifest: Optional[Path] = typer.Option(
        None,
        "--eval-manifest",
        help="Path to an agent-eval manifest (JSON list). Overrides config.eval_manifest_path.",
    ),
    domains: Optional[str] = typer.Option(
        None,
        "--domains",
        help="Optional comma-separated list of domains to include. Defaults to all domains present in the dataset.",
    ),
    artifacts_root: Optional[Path] = typer.Option(
        None, "--artifacts-root", help="Override the artifact root; defaults to ./artifacts/"
    ),
    agent_name: Optional[str] = typer.Option(
        None,
        "--agent",
        help="Agent CLI to evaluate: claude or codex. Overrides config.agent.",
    ),
    model_override: Optional[str] = typer.Option(
        None,
        "--model",
        help="Agent model override for this run.",
    ),
) -> None:
    """Run the benchmark across the dataset's domains, sequentially."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    agent = _resolve_agent(str(agent_name or cfg.get("agent") or "claude"))
    if shutil.which(agent) is None:
        typer.echo(f"Error: `{agent}` CLI is not on PATH.", err=True)
        raise typer.Exit(code=2)

    manifest_path = eval_manifest or cfg.get("eval_manifest_path")
    if not manifest_path:
        typer.echo("Error: config is missing 'eval_manifest_path' and --eval-manifest was not provided.", err=True)
        raise typer.Exit(code=2)
    entries = load_eval_manifest(Path(str(manifest_path)).expanduser().resolve())
    typer.echo(f"Loaded {len(entries)} dataset entries.")

    by_domain: dict[str, list[DatasetEntry]] = defaultdict(list)
    for e in entries:
        by_domain[e.domain].append(e)

    if domains:
        wanted = {d.strip() for d in domains.split(",") if d.strip()}
        unknown = wanted - set(by_domain)
        if unknown:
            typer.echo(
                f"Error: --domains references unknown domains {sorted(unknown)}. " f"Available: {sorted(by_domain)}",
                err=True,
            )
            raise typer.Exit(code=2)
        by_domain = {d: by_domain[d] for d in wanted}

    domain_order = sorted(by_domain.keys())
    typer.echo(f"Domains in this run: {domain_order} ({sum(len(v) for v in by_domain.values())} entries total)")

    workdir_root = Path(str(cfg.get("per_trial_workdir_root", "/tmp/skill_eval"))).expanduser()
    workdir_root.mkdir(parents=True, exist_ok=True)
    model = _resolve_agent_model(cfg, agent, model_override)
    budget = float(cfg.get("per_trial_budget_usd", 5.0))
    timeout = int(cfg.get("per_trial_timeout_s", 600))
    testdata_prefixes_raw = cfg.get("testdata_prefixes") or []
    if not isinstance(testdata_prefixes_raw, list):
        typer.echo("Error: config 'testdata_prefixes' must be a list of strings.", err=True)
        raise typer.Exit(code=2)
    testdata_prefixes = tuple(str(p) for p in testdata_prefixes_raw)

    judge = _build_judge(cfg, manifest_path=Path(str(manifest_path)).expanduser().resolve())
    summarizer = _build_trace_summarizer(cfg)

    base_dir = str(artifacts_root) if artifacts_root else None
    session_dir = create_session_dir("skilleval", base_dir=base_dir)
    typer.echo(f"Session dir: {session_dir}")
    typer.echo(f"Agent: {agent}  model={model}  condition={BASE_CONDITION}")

    resolved_cfg = dict(cfg)
    resolved_cfg["agent"] = agent
    resolved_cfg["agent_model"] = model
    (session_dir / "config.yaml").write_text(yaml.safe_dump(resolved_cfg, default_flow_style=False), encoding="utf-8")

    # Results keyed (agent, condition, domain) so reports can compare agent runs.
    results_by_key: dict[tuple[str, str, str], list] = {}
    for domain in domain_order:
        domain_entries = by_domain[domain]
        pdf_source = _resolve_pdf_source(cfg, domain)
        if not pdf_source.is_dir():
            typer.echo(
                f"Error: PDF directory '{pdf_source}' for domain '{domain}' does not exist or is not a directory. "
                f"Check the 'pdf_dirs' (or 'pdf_dir') setting in your config.",
                err=True,
            )
            raise typer.Exit(code=2)
        domain_label = _resolve_domain_label(domain_entries, cfg, domain)
        typer.echo(
            f"Starting {agent} session for {domain} — setup + {len(domain_entries)} query turns "
            f"(pdfs={pdf_source})"
        )
        workdir, results = run_session(
            agent=agent,
            entries=domain_entries,
            workdir_root=workdir_root,
            pdf_source=pdf_source,
            model=model,
            budget_usd=budget,
            timeout_s=timeout,
            domain=domain,
            domain_label=domain_label,
            judge=judge,
            testdata_prefixes=testdata_prefixes,
        )
        if summarizer is not None and results:
            trace = extract_compact_trace(agent, workdir, results[0].session_id)
            if trace:
                narrative = summarizer.summarize(condition=f"{agent}/{BASE_CONDITION}", domain=domain, trace=trace)
                if narrative:
                    for r in results:
                        if r.is_setup:
                            r.tool_use_summary = narrative
                            break
                    typer.echo(f"  tool-use summary: {len(narrative)} chars")
                else:
                    typer.echo("  tool-use summary: (summarizer returned empty)")
            else:
                typer.echo("  tool-use summary skipped: session JSONL unavailable")
        for r in results:
            save_trial(r, session_dir)
            kind = "setup" if r.is_setup else f"entry_id={r.entry_id} query_id={r.query_id}"
            judge_parts: list[str] = []
            if not r.is_setup:
                if r.judge_score is not None:
                    judge_parts.append(f"judge={r.judge_score}")
                elif any(v is not None for v in r.judge_subscores.values()):
                    n = sum(1 for v in r.judge_subscores.values() if v is not None)
                    judge_parts.append(f"judge_mode={r.judge_mode}/sub_n={n}")
                if r.legacy_judge_score is not None:
                    judge_parts.append(f"legacy={r.legacy_judge_score}")
            judge_str = (" " + " ".join(judge_parts)) if judge_parts else ""
            cost_str = f"${r.total_cost_usd:.3f}" if r.cost_available else "n/a"
            typer.echo(
                f"  turn {r.num_turns} [{agent}/{domain}] {kind}: status={r.status} "
                f"tokens(in/out/cache_r)={r.input_tokens}/{r.output_tokens}/{r.cache_read_input_tokens} "
                f"cost={cost_str} retrieved={len(r.ranked_retrieved)}{judge_str}"
            )
        results_by_key[(agent, BASE_CONDITION, domain)] = results

        entries_by_id = {e.entry_id: e for e in domain_entries}
        scores = overall_recall(results, entries_by_id)
        typer.echo(
            f"\nRecall for {domain}: "
            f"recall@1={scores['recall_1']:.3f}  "
            f"recall@5={scores['recall_5']:.3f}  "
            f"recall@10={scores['recall_10']:.3f}"
        )

        if results:
            archived = archive_session_log(
                session_dir=session_dir,
                agent=agent,
                condition=BASE_CONDITION,
                domain=domain,
                session_uuid=results[0].session_id,
                workdir=workdir,
            )
            if archived is not None:
                typer.echo(f"  archived session log: {archived.relative_to(session_dir)}")
            else:
                typer.echo(f"  session log not found for archiving ({agent}/{domain})")

        cleanup_session_workdir(workdir)
        typer.echo(f"Cleaned up workdir for {domain}\n")

    if judge is not None:
        typer.echo("\nLLM-as-judge scores (mean over query turns, 0-5 scale):")
        simple_scored: list[int] = []
        scenario_scored = 0
        legacy_scored: list[int] = []
        errored = 0
        for domain in domain_order:
            for r in results_by_key.get((agent, BASE_CONDITION, domain), []):
                if r.is_setup:
                    continue
                has_subscores = any(v is not None for v in r.judge_subscores.values())
                if r.judge_score is not None:
                    simple_scored.append(int(r.judge_score))
                elif has_subscores:
                    scenario_scored += 1
                elif r.judge_error:
                    errored += 1
                if r.legacy_judge_score is not None:
                    legacy_scored.append(int(r.legacy_judge_score))
        parts: list[str] = []
        if simple_scored:
            parts.append(f"simple mean={sum(simple_scored) / len(simple_scored):.2f} n={len(simple_scored)}")
        if scenario_scored:
            parts.append(f"scenario_scored={scenario_scored}")
        if legacy_scored:
            parts.append(f"legacy mean={sum(legacy_scored) / len(legacy_scored):.2f} n={len(legacy_scored)}")
        parts.append(f"errors={errored}")
        if not simple_scored and not scenario_scored and not legacy_scored:
            parts.append("(check judge config / litellm install)")
        typer.echo("  " + "  ".join(parts))

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_key=results_by_key,
        entries=entries,
        config=resolved_cfg,
        agent=agent,
        model=model,
        config_path=str(config) if config else "<packaged default>",
    )
    typer.echo(f"\nWrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")


def _needs_rescore(trial: dict[str, Any]) -> bool:
    """A query-turn trial needs rescoring if it produced no usable judge output.

    "Usable" means either the legacy/simple flat ``judge_score`` (an int in
    1-5) OR at least one non-null entry in ``judge_subscores`` (the
    scenario-aware judge's per-dimension scores). A trial counts as scored
    even when ``judge_score`` is ``None`` provided sub-scores are present.

    Intrinsically unscorable trials (missing ground truth, empty candidate,
    ``scoring_mode="skip"``) are terminal and never retried.
    """
    if trial.get("is_setup"):
        return False
    judge_error = trial.get("judge_error") or ""
    if judge_error in UNSCORABLE_JUDGE_ERRORS:
        return False
    sub_scores = trial.get("judge_subscores") or {}
    scored = trial.get("judge_score") is not None or any(v is not None for v in sub_scores.values())
    if not scored:
        return True
    if judge_error:
        return True
    return False


def _load_trial(path: Path) -> tuple[dict[str, Any], TrialResult]:
    """Load a trial JSON and reconstruct a ``TrialResult``.

    Returns the raw dict alongside the dataclass so callers can write back
    fields the dataclass doesn't carry (none today, but future-proof against
    on-disk schema drift).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(TrialResult)}
    ctor_kwargs = {k: v for k, v in data.items() if k in known}
    return data, TrialResult(**ctor_kwargs)


def _iter_trial_files(session_dir: Path) -> list[Path]:
    return sorted((session_dir / "trials").rglob("*.json"))


@app.command("rescore")
def rescore_command(
    session_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Artifact session directory from a previous `skill-eval run` (e.g. artifacts/skilleval_*).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Judge/manifest config to use. Defaults to the session's own config.yaml.",
    ),
    eval_manifest: Optional[Path] = typer.Option(
        None,
        "--eval-manifest",
        help="Manifest path. Overrides eval_manifest_path from --config / session config.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rescore every query-turn trial, not just the empty/failed ones.",
    ),
) -> None:
    """Re-judge query-turn trials with missing or failed judge scores.

    Walks ``session_dir/trials/**/*.json`` and rewrites each matching trial
    in place with a fresh ``judge_score`` / ``judge_reasoning`` /
    ``judge_error``. After rescoring, ``session_summary.json`` and
    ``session_summary.md`` are regenerated so aggregated judge metrics reflect
    the new scores.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    session_dir = session_dir.resolve()
    trials_dir = session_dir / "trials"
    if not trials_dir.is_dir():
        typer.echo(f"Error: {trials_dir} does not exist — is this a skill_eval session dir?", err=True)
        raise typer.Exit(code=2)

    session_cfg_path = session_dir / "config.yaml"
    if config is not None:
        cfg = load_config(config)
        config_path_str = str(config)
    elif session_cfg_path.is_file():
        cfg = load_config(session_cfg_path)
        config_path_str = str(session_cfg_path)
    else:
        typer.echo(
            f"Error: no --config given and {session_cfg_path} is missing; cannot resolve judge settings.",
            err=True,
        )
        raise typer.Exit(code=2)

    manifest_path = eval_manifest or cfg.get("eval_manifest_path")
    if not manifest_path:
        typer.echo("Error: config is missing 'eval_manifest_path' and --eval-manifest was not provided.", err=True)
        raise typer.Exit(code=2)
    entries = load_eval_manifest(Path(str(manifest_path)).expanduser().resolve())
    entries_by_id = {e.entry_id: e for e in entries}

    judge = _build_judge(cfg, manifest_path=Path(str(manifest_path)).expanduser().resolve())
    if judge is None:
        typer.echo("Error: judge is not configured (see messages above). Cannot rescore.", err=True)
        raise typer.Exit(code=2)

    trial_files = _iter_trial_files(session_dir)
    candidates = []
    for path in trial_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("is_setup"):
            continue
        if force or _needs_rescore(data):
            candidates.append(path)

    typer.echo(
        f"Rescoring {len(candidates)} trial(s) out of {len(trial_files)} on disk "
        f"(force={'on' if force else 'off'})."
    )

    rescored = 0
    unscorable = 0
    still_failed = 0
    for path in candidates:
        raw, result = _load_trial(path)
        entry = entries_by_id.get(result.entry_id)
        if entry is None:
            typer.echo(f"  {path.name}: skip (entry_id={result.entry_id} not in manifest)")
            continue

        # Clear stale judge state so _apply_judge starts from a clean slate.
        result.judge_score = None
        result.judge_reasoning = ""
        result.judge_error = ""
        result.judge_mode = ""
        result.judge_subscores = {}
        result.judge_flags = {}
        result.judge_lists = {}
        result.legacy_judge_score = None
        result.legacy_judge_reasoning = ""
        result.legacy_judge_error = ""

        _apply_judge(judge, entry, result)

        raw.update(asdict(result))
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

        scored_ok = result.judge_score is not None or any(
            v is not None for v in result.judge_subscores.values()
        )
        if scored_ok:
            rescored += 1
            if result.judge_mode == "simple" and result.judge_score is not None:
                typer.echo(f"  {path.name}: entry_id={result.entry_id} judge={result.judge_score}")
            else:
                n_sub = sum(1 for v in result.judge_subscores.values() if v is not None)
                typer.echo(
                    f"  {path.name}: entry_id={result.entry_id} "
                    f"mode={result.judge_mode} sub_scores={n_sub}"
                )
        elif result.judge_error in UNSCORABLE_JUDGE_ERRORS:
            unscorable += 1
            typer.echo(
                f"  {path.name}: entry_id={result.entry_id} unscorable ({result.judge_error})"
            )
        else:
            still_failed += 1
            typer.echo(
                f"  {path.name}: entry_id={result.entry_id} still failed "
                f"(error={result.judge_error or 'unknown'})"
            )

    typer.echo(
        f"\nRescored {rescored}; unscorable {unscorable}; still failed {still_failed}."
    )

    # Rebuild results_by_key from disk so the regenerated summary matches the
    # files we just wrote (including any trials we left untouched).
    results_by_key: dict[tuple[str, str, str], list[TrialResult]] = defaultdict(list)
    for path in trial_files:
        _, result = _load_trial(path)
        results_by_key[(result.agent, result.condition, result.domain)].append(result)

    agent = str(cfg.get("agent") or "claude")
    model = str(cfg.get("agent_model") or _resolve_agent_model(cfg, agent, None))

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_key=dict(results_by_key),
        entries=entries,
        config=cfg,
        agent=agent,
        model=model,
        config_path=config_path_str,
    )
    typer.echo(f"Wrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")
