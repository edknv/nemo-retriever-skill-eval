# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session runner: build a sandboxed workdir, spawn an agent CLI, parse outputs."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from skill_eval.dataset import DatasetEntry

logger = logging.getLogger(__name__)

BASE_CONDITION = "c1_base"
SUPPORTED_AGENTS = ("claude", "codex")
DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.5",
}

# Scenario-aware judging routes each entry by its ``scoring_mode`` field.
# ``ANSWERABLE_SCORING_MODES`` are the modes for which a missing
# ``ground_truth_answer`` makes the trial unscorable; ``ANSWER_REQUIRED``
# adds modes where the AGENT must produce a non-empty answer for the judge
# to have anything to grade (refusal / capability_gap / dispatcher_prompt all
# expect the agent to emit explanatory text even when no answer is possible).
ANSWERABLE_SCORING_MODES: frozenset[str] = frozenset({"answerable_retrieval", "ingest_plus_answer"})
ANSWER_REQUIRED_SCORING_MODES: frozenset[str] = frozenset(
    {"answerable_retrieval", "ingest_plus_answer", "refusal", "capability_gap", "dispatcher_prompt"}
)


@dataclass
class JudgeContext:
    """Bag of judge state threaded from CLI into ``_apply_judge``.

    Holds the chat-completion transport for the new scenario-aware judge
    (``client``), the two on-disk prompt paths (``simple_prompt_path``,
    ``scenario_prompt_path``), and the optional legacy ``LLMJudge``.
    ``None`` is a valid path when no entries in the run need that mode.
    """

    client: Any
    simple_prompt_path: str | None
    scenario_prompt_path: str | None
    legacy_judge: Any = None


@functools.lru_cache(maxsize=8)
def _load_prompt_template(name: str) -> str:
    return Path(str(pkg_files("skill_eval").joinpath(f"prompts/{name}"))).read_text(encoding="utf-8")


@dataclass
class TrialResult:
    trial_id: str
    condition: str
    agent: str
    entry_id: int
    query_id: str
    status: str
    extraction_method: str
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    total_cost_usd: float
    model_id: str
    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0
    final_answer: str = ""
    ranked_retrieved: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_setup: bool = False
    domain: str = ""
    # Legacy single-score field; for simple-mode entries the scenario judge
    # also populates this to ``answer_correctness`` for backward compatibility
    # with the existing summary/rescore paths.
    judge_score: int | None = None
    judge_reasoning: str = ""
    judge_error: str = ""
    # Scenario-aware judge outputs. ``judge_mode`` is "simple" or "scenario";
    # the rest are populated from ``judging.ScenarioJudgeResult``.
    judge_mode: str = ""
    judge_subscores: dict[str, int | None] = field(default_factory=dict)
    judge_flags: dict[str, bool | None] = field(default_factory=dict)
    judge_lists: dict[str, list[str]] = field(default_factory=dict)
    # Hardcoded-prompt ``LLMJudge`` outputs, kept on every trial when
    # ``judge.legacy_enabled`` is true so scores stay comparable to old runs.
    legacy_judge_score: int | None = None
    legacy_judge_reasoning: str = ""
    legacy_judge_error: str = ""
    tool_use_summary: str = ""
    cost_available: bool = True
    # Parallelization-related provenance. ``execution_mode`` is
    # "linear_session" when all turns share one agent session, or
    # "parallel_isolated" when query turns run in cloned workdirs after the
    # setup turn. ``raw_log_path`` / ``compact_trace_path`` are relative to
    # the artifact session directory (or "" when not persisted).
    execution_mode: str = "linear_session"
    raw_log_path: str = ""
    compact_trace_path: str = ""


@dataclass
class SessionRun:
    """Scratch paths and trial results for one (agent, domain) session.

    A ``linear_session`` run uses one workdir end-to-end; a
    ``parallel_isolated`` run uses one setup workdir plus a per-query
    cloned workdir (recorded in ``workdirs`` for cleanup and in
    ``result_workdirs`` so the CLI can locate the raw transcript log).
    """

    workdir: Path
    workdirs: list[Path]
    results: list[TrialResult]
    result_workdirs: dict[str, Path]
    execution_mode: str = "linear_session"


_DOC_ID_HASH_LEN = 10


def _hash_stem(stem: str) -> str:
    return "doc_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:_DOC_ID_HASH_LEN]


@dataclass(frozen=True)
class DocIdMap:
    """Bidirectional map between real PDF basenames (without ``.pdf``) and
    opaque ``doc_<sha1[:10]>`` ids used as anonymized filenames inside the
    agent's workdir.

    The same real stem always produces the same anonymized id (stable across
    runs), so debugging output is reproducible. Collisions in the 10-hex-char
    prefix raise immediately rather than silently overwriting a symlink.
    """

    real_to_anon: dict[str, str]
    anon_to_real: dict[str, str]

    @classmethod
    def from_stems(cls, stems: list[str]) -> "DocIdMap":
        real_to_anon: dict[str, str] = {}
        anon_to_real: dict[str, str] = {}
        for stem in stems:
            anon = _hash_stem(stem)
            existing = anon_to_real.get(anon)
            if existing is not None and existing != stem:
                raise ValueError(
                    f"doc_id hash collision: {existing!r} and {stem!r} both map to {anon!r}; "
                    f"widen _DOC_ID_HASH_LEN or rename one of the source files"
                )
            real_to_anon[stem] = anon
            anon_to_real[anon] = stem
        return cls(real_to_anon=real_to_anon, anon_to_real=anon_to_real)

    @classmethod
    def from_pdf_dir(cls, pdf_source: Path) -> "DocIdMap":
        stems = sorted(p.stem for p in pdf_source.glob("*.pdf"))
        return cls.from_stems(stems)

    def anonymize(self, real_stem: str) -> str:
        return self.real_to_anon.get(real_stem, real_stem)

    def deanonymize(self, anon_stem: str) -> str:
        return self.anon_to_real.get(anon_stem, anon_stem)


def _remap_pdf_paths(text: str, prefixes: tuple[str, ...]) -> str:
    """Rewrite caller-supplied path prefixes in *text* to ``./pdfs/``.

    Some agent-eval manifests' paraphrased prompts hard-code paths from the
    dataset source tree. Each trial workdir symlinks the domain's PDFs to
    ``./pdfs/``, so the agent only needs the basename.
    """
    for prefix in prefixes:
        text = text.replace(prefix, "./pdfs")
    return text


def _render_prompt(entry: DatasetEntry, testdata_prefixes: tuple[str, ...] = ()) -> str:
    text = _load_prompt_template("trial_user.j2")
    return text.replace(
        "{{ paraphrased_prompt }}", _remap_pdf_paths(entry.paraphrased_prompt, testdata_prefixes)
    ).replace("{{ original_query }}", entry.original_query)


def _render_setup_prompt(domain_label: str = "PDFs") -> str:
    text = _load_prompt_template("setup.j2")
    return text.replace("{{ domain_label }}", domain_label)


_ISOLATED_QUERY_NOTE = (
    "The setup turn has already completed in this isolated copy of the setup workdir. "
    "Use the artifacts already present here. If helpful, read ./setup_context.md before answering."
)


def _render_isolated_query_prompt(
    entry: DatasetEntry,
    testdata_prefixes: tuple[str, ...] = (),
) -> str:
    """Render a query prompt for a fresh session branched from setup artifacts."""
    base = _render_prompt(entry, testdata_prefixes)
    return f"{_ISOLATED_QUERY_NOTE}\n\n{base}"


def _write_setup_context(workdir: Path, *, agent: str) -> Path:
    """Write a durable handoff file for isolated query sessions.

    Lists the top-level artifacts produced by the setup turn so each
    parallel query session can find them without re-running discovery.
    """
    artifact_names: list[str] = []
    ignored = {".claude", ".codex", ".bin", "pdfs", "setup_context.md"}
    for child in sorted(workdir.iterdir(), key=lambda p: p.name):
        name = child.name
        if name in ignored or name == "output.json" or name.startswith("output_e"):
            continue
        suffix = "/" if child.is_dir() else ""
        artifact_names.append(f"- `{name}{suffix}`")

    lines = [
        "# skill_eval setup context",
        "",
        "This file was generated by the skill_eval harness after the setup turn completed.",
        "Future query sessions start from an isolated copy of this workdir.",
        "",
        f"- Agent: `{agent}`",
        "- PDFs are available under `./pdfs/`.",
        "",
        "## Top-level setup artifacts",
    ]
    lines.extend(artifact_names or ["- No additional top-level artifacts were detected."])
    out = workdir / "setup_context.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _ignore_runtime_outputs(_directory: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ``ignore`` callback that skips per-turn outputs."""
    return {name for name in names if name == "output.json" or name.startswith("output_e")}


def _copy_setup_workdir_for_query(setup_workdir: Path, entry: DatasetEntry) -> Path:
    """Clone setup artifacts into a per-query scratch workdir.

    ``symlinks=True`` preserves the ``./pdfs/`` symlink farm so we don't
    duplicate every PDF on disk for every query.
    """
    dest = setup_workdir.parent / f"{setup_workdir.name}_query_e{entry.entry_id}_{uuid.uuid4().hex[:8]}"
    shutil.copytree(setup_workdir, dest, symlinks=True, ignore=_ignore_runtime_outputs)
    return dest


def save_compact_trace(
    result: TrialResult,
    session_dir: Path,
    trace: str,
    *,
    suffix: str = "",
) -> Path:
    """Persist a compact tool-use trace alongside the trial JSON.

    Returns the absolute path; callers typically translate to a
    session-relative path before stamping it onto ``result.compact_trace_path``.
    """
    parts = [session_dir, "trials", result.agent, result.condition]
    if result.domain:
        parts.append(result.domain)
    tag = f"_{suffix}" if suffix else ""
    out_dir = Path(*[str(p) for p in parts])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{result.trial_id}{tag}_trace.txt"
    out.write_text(trace + ("\n" if not trace.endswith("\n") else ""), encoding="utf-8")
    return out


def _build_pdf_symlinks(
    pdf_source: Path, dest: Path, doc_map: "DocIdMap | None" = None
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_source.glob("*.pdf")):
        if doc_map is not None:
            anon = doc_map.anonymize(pdf.stem)
            target_name = f"{anon}.pdf"
        else:
            target_name = pdf.name
        target = dest / target_name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(pdf.resolve())


def _build_session_workdir(
    agent: str,
    root: Path,
    pdf_source: Path,
    domain: str = "",
    doc_map: "DocIdMap | None" = None,
) -> Path:
    """Build the per-session workdir.

    Workdir contents:
      - pdfs/ symlink farm into the source PDF folder (renamed via ``doc_map``
        when supplied so the agent only sees opaque filenames)
      - .claude/ sandbox with an empty settings.json for Claude runs

    The agent itself creates any retrieval artifacts (e.g. ./lancedb/) inside
    the workdir on the setup turn.
    """
    domain_seg = f"_{domain}" if domain else ""
    workdir = root / f"{agent}_{BASE_CONDITION}{domain_seg}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_pdf_symlinks(pdf_source, workdir / "pdfs", doc_map)
    if agent == "claude":
        (workdir / ".claude").mkdir(parents=True, exist_ok=True)
        (workdir / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    return workdir


def cleanup_session_workdir(workdir: Path) -> None:
    """Remove a session's scratch workdir (PDF symlinks and agent-built
    artifacts like .venv/, lancedb/, scratch scripts). Called after a session
    completes and its results have been persisted to the artifact dir.
    """
    if not workdir.exists():
        return
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("cleaned up workdir %s", workdir)


def archive_session_log(
    *,
    session_dir: Path,
    agent: str,
    condition: str,
    domain: str,
    session_uuid: str,
    workdir: Path,
) -> Path | None:
    """Copy the agent's rollout log into the artifact dir so it survives ``cleanup_session_workdir``.

    Without this, the per-trial JSONs are the only persistent record of the run —
    you cannot retroactively recompute token deltas, tool-use signals, or anything
    else that requires the raw event stream.
    """
    if agent == "claude":
        src: Path | None = _claude_session_log_path(workdir, session_uuid)
    elif agent == "codex":
        src = _codex_session_log_path(session_uuid)
    else:
        return None
    if src is None or not src.exists():
        return None
    parts = [session_dir, "trials", agent, condition]
    if domain:
        parts.append(domain)
    logs_dir = Path(*[str(p) for p in parts]) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dest = logs_dir / src.name
    shutil.copy2(src, dest)
    return dest


def _build_claude_command(
    model: str,
    budget_usd: float,
    session_uuid: str,
    workdir: Path,
    *,
    resume: bool = False,
) -> list[str]:
    """Build the `claude -p` command. First turn uses --session-id; subsequent turns use --resume.

    We deliberately do NOT pass --no-session-persistence because multi-turn requires
    the session to persist between subprocess invocations.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--add-dir",
        str(workdir),
        "--permission-mode",
        "bypassPermissions",
        "--allow-dangerously-skip-permissions",
        "--max-budget-usd",
        str(budget_usd),
        "--setting-sources",
        "project",
        "--disable-slash-commands",
    ]
    if resume:
        cmd.extend(["--resume", session_uuid])
    else:
        cmd.extend(["--session-id", session_uuid])
    return cmd


def _build_codex_command(
    model: str,
    session_uuid: str,
    workdir: Path,
    *,
    resume: bool = False,
) -> list[str]:
    """Build a non-interactive Codex command.

    Codex assigns the first session id itself; subsequent turns resume the id
    parsed from the setup turn's JSONL events.
    """
    common = [
        "--json",
        "--model",
        model,
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if resume:
        return ["codex", "exec", "resume", *common, session_uuid, "-"]
    return [
        "codex",
        "exec",
        *common,
        "--cd",
        str(workdir),
        "--add-dir",
        str(workdir),
        "-",
    ]


def _build_command(
    *,
    agent: str,
    model: str,
    budget_usd: float,
    session_uuid: str,
    workdir: Path,
    resume: bool = False,
) -> list[str]:
    if agent == "claude":
        return _build_claude_command(model, budget_usd, session_uuid, workdir, resume=resume)
    if agent == "codex":
        return _build_codex_command(model, session_uuid, workdir, resume=resume)
    raise ValueError(f"unsupported agent: {agent}")


def _parse_envelope(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        logger.warning("could not parse claude envelope (first 200 chars): %r", raw[:200])
        return {}


def _parse_jsonl_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _codex_session_id(events: list[dict[str, Any]], fallback: str) -> str:
    for ev in events:
        if ev.get("type") != "session_meta":
            continue
        payload = ev.get("payload") or {}
        if isinstance(payload, dict) and payload.get("id"):
            return str(payload["id"])
    return fallback


def _codex_has_error(events: list[dict[str, Any]]) -> bool:
    for ev in events:
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("type") in {"error", "task_failed", "turn_aborted"}:
            return True
    return False


def _populate_claude_tokens(result: TrialResult, envelope: dict[str, Any]) -> None:
    usage = envelope.get("usage") or {}
    result.input_tokens = int(usage.get("input_tokens") or 0)
    result.output_tokens = int(usage.get("output_tokens") or 0)
    result.cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)
    result.cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_detail = usage.get("cache_creation") or {}
    result.ephemeral_5m_input_tokens = int(cache_detail.get("ephemeral_5m_input_tokens") or 0)
    result.ephemeral_1h_input_tokens = int(cache_detail.get("ephemeral_1h_input_tokens") or 0)


_CODEX_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
)


def _extract_codex_total_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return the most recent cumulative ``total_token_usage`` from codex events.

    Each ``token_count`` event carries running session-wide counters; we want the
    last one so deltas between two snapshots equal one turn's true work.
    """
    for ev in reversed(events):
        if ev.get("type") != "event_msg":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        if not isinstance(info, dict):
            continue
        usage = info.get("total_token_usage") or {}
        if not isinstance(usage, dict):
            continue
        return {k: int(usage.get(k) or 0) for k in _CODEX_USAGE_FIELDS}
    return {k: 0 for k in _CODEX_USAGE_FIELDS}


def _populate_codex_tokens(
    result: TrialResult,
    current_totals: dict[str, int],
    prior_totals: dict[str, int],
) -> None:
    """Set per-turn token fields as the delta of cumulative ``total_token_usage``.

    Codex's resumed-session log is append-only across all turns, and each
    ``token_count`` event reports cumulative counters, so per-turn cost is the
    difference between snapshots taken before and after the subprocess call.
    ``output_tokens`` here folds in ``reasoning_output_tokens`` so the column
    reflects everything the model emitted, matching Claude's accounting.
    """
    def d(key: str) -> int:
        return max(0, current_totals.get(key, 0) - prior_totals.get(key, 0))

    result.input_tokens = d("input_tokens")
    result.output_tokens = d("output_tokens") + d("reasoning_output_tokens")
    result.cache_read_input_tokens = d("cached_input_tokens")
    result.cache_creation_input_tokens = 0


def _parse_output_json(workdir: Path) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    out_path = workdir / "output.json"
    errors: list[str] = []
    if not out_path.exists():
        return "", [], "missing", ["output.json not written"]
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "", [], "invalid_json", [f"invalid JSON: {e}"]
    if not isinstance(payload, dict):
        return "", [], "invalid_json", [f"top-level must be an object, got {type(payload).__name__}"]
    for required in ("final_answer", "ranked_retrieved"):
        if required not in payload:
            errors.append(f"missing required key '{required}'")
    ranked = payload.get("ranked_retrieved") or []
    if not isinstance(ranked, list):
        errors.append(f"ranked_retrieved must be a list, got {type(ranked).__name__}")
        ranked = []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(ranked, start=1):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        page = item.get("page_number")
        if doc_id is None or page is None:
            continue
        cleaned.append({"doc_id": str(doc_id), "page_number": int(page), "rank": int(item.get("rank") or i)})
    return str(payload.get("final_answer") or ""), cleaned, ("ok" if not errors else "schema_warning"), errors


def _apply_doc_map_to_ranked(
    ranked: list[dict[str, Any]],
    doc_map: "DocIdMap | None",
) -> list[dict[str, Any]]:
    """Rewrite each ranked entry's ``doc_id`` from anonymized -> real and
    attach the anonymized value on a sidecar ``anonymized_doc_id`` field.

    No-op when ``doc_map`` is ``None``: returns ``ranked`` unchanged so the
    on-disk schema is byte-identical to the pre-feature shape.

    An anonymized id the map doesn't recognize (agent hallucination, or it
    tried to outsmart the renaming) passes through unchanged in ``doc_id``;
    ``anonymized_doc_id`` is set to the same value. Recall scoring then
    naturally returns 0 for that entry.
    """
    if doc_map is None:
        return ranked
    out: list[dict[str, Any]] = []
    for entry in ranked:
        anon = str(entry.get("doc_id") or "")
        real = doc_map.deanonymize(anon)
        new_entry = dict(entry)
        new_entry["doc_id"] = real
        new_entry["anonymized_doc_id"] = anon
        out.append(new_entry)
    return out


def _extract_model_id(envelope: dict[str, Any], fallback: str) -> str:
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        return next(iter(model_usage.keys()))
    return str(envelope.get("model") or fallback)


def _extract_claude_error_detail(envelope: dict[str, Any]) -> str:
    for key in ("error", "message", "result"):
        value = envelope.get(key)
        if value:
            return str(value)

    content = envelope.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return " ".join(parts)
    return ""


def _claude_session_log_path(workdir: Path, session_uuid: str) -> Path:
    """Claude Code persists per-session transcripts at
    ``~/.claude/projects/<slug>/<session_id>.jsonl`` where ``<slug>`` is the
    project dir with ``/`` and ``_`` both replaced by ``-`` (and a leading ``-``
    preserved for the filesystem root).
    """
    slug = str(workdir).replace("/", "-").replace("_", "-")
    if not slug.startswith("-"):
        slug = "-" + slug
    return Path.home() / ".claude" / "projects" / slug / f"{session_uuid}.jsonl"


def _codex_session_log_path(session_uuid: str) -> Path | None:
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    matches = sorted(
        sessions_root.glob(f"**/*{session_uuid}.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return matches[0] if matches else None


def _codex_session_meta_from_log(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    ev = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "session_meta":
                    continue
                payload = ev.get("payload") or {}
                return payload if isinstance(payload, dict) else {}
    except OSError:
        return {}
    return {}


def _codex_session_log_for_workdir(workdir: Path) -> Path | None:
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    workdir_str = str(workdir)
    matches = sorted(
        sessions_root.glob("**/rollout-*.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for path in matches:
        meta = _codex_session_meta_from_log(path)
        if str(meta.get("cwd") or "") == workdir_str:
            return path
    return None


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    try:
        return _parse_jsonl_events(path.read_text(encoding="utf-8"))
    except OSError:
        return []


_TRACE_TOOL_INPUT_CAP = 200
_TRACE_FINAL_TEXT_CAP = 400


def _truncate(s: str, cap: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= cap else s[: cap - 1] + "…"


def _format_tool_input(name: str, inp: dict[str, Any]) -> str:
    """Render a tool_use input dict to a single short line."""
    if name == "Bash":
        cmd = str(inp.get("command", ""))
        return f"Bash: {_truncate(cmd, _TRACE_TOOL_INPUT_CAP)}"
    if name == "Read":
        path = str(inp.get("file_path", ""))
        offset = inp.get("offset")
        limit = inp.get("limit")
        tail = f" offset={offset} limit={limit}" if offset is not None or limit is not None else ""
        return f"Read: {path}{tail}"
    if name == "Grep":
        pat = str(inp.get("pattern", ""))
        path = str(inp.get("path", ""))
        return f"Grep: pattern={_truncate(pat, 80)} path={path}"
    if name == "Glob":
        return f"Glob: {inp.get('pattern', '')}"
    if name in ("Edit", "Write"):
        return f"{name}: {inp.get('file_path', '')}"
    parts = [f"{k}={_truncate(str(v), 80)}" for k, v in inp.items()]
    return f"{name}: " + " ".join(parts) if parts else name


def _extract_claude_compact_trace(
    workdir: Path,
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    """Walk the Claude Code session JSONL and emit a turn-organized text trace.

    Lists per turn: the user prompt, every ``tool_use`` invocation with
    truncated inputs, and the agent's final assistant text. ``tool_result``
    content is omitted — the summarizer needs actions, not tool-output noise.
    Returns ``None`` if the JSONL is missing or unreadable.
    """
    log_path = _claude_session_log_path(workdir, session_uuid)
    if not log_path.exists():
        return None

    turn_idx = 0
    lines_out: list[str] = []
    current_assistant_text: list[str] = []
    try:
        with log_path.open(encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    ev = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                msg = ev.get("message") or {}
                role = msg.get("role") or ev.get("type")
                content = msg.get("content")

                if role == "user":
                    if current_assistant_text:
                        joined = " ".join(current_assistant_text).strip()
                        if joined:
                            lines_out.append(f"  assistant: {_truncate(joined, _TRACE_FINAL_TEXT_CAP)}")
                        current_assistant_text = []
                    turn_idx += 1
                    user_text = ""
                    if isinstance(content, str):
                        user_text = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_text = str(item.get("text", ""))
                                break
                    label = (
                        first_turn_label
                        if turn_idx == 1 and first_turn_label
                        else ("setup" if turn_idx == 1 else f"query {turn_idx - 1}")
                    )
                    lines_out.append("")
                    lines_out.append(f"[Turn {turn_idx} — {label}]")
                    if user_text:
                        lines_out.append(f"  user: {_truncate(user_text, _TRACE_FINAL_TEXT_CAP)}")
                elif role == "assistant" and isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        itype = item.get("type")
                        if itype == "tool_use":
                            name = str(item.get("name", "?"))
                            inp = item.get("input") or {}
                            if isinstance(inp, dict):
                                lines_out.append(f"  tool_use {_format_tool_input(name, inp)}")
                            else:
                                lines_out.append(f"  tool_use {name}")
                        elif itype == "text":
                            text = str(item.get("text", "")).strip()
                            if text:
                                current_assistant_text.append(text)
    except OSError:
        return None

    if current_assistant_text:
        joined = " ".join(current_assistant_text).strip()
        if joined:
            lines_out.append(f"  assistant: {_truncate(joined, _TRACE_FINAL_TEXT_CAP)}")

    trace = "\n".join(lines_out).strip()
    return trace or None


def _string_from_content_items(content: Any, *, input_text: bool = True) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    wanted = "input_text" if input_text else "output_text"
    fallback = "text"
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {wanted, fallback}:
            out.append(str(item.get("text") or ""))
    return " ".join(x for x in out if x).strip()


def _format_codex_tool_input(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "?")
    args = payload.get("arguments") or ""
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=False)
    return f"{name}: {_truncate(args, _TRACE_TOOL_INPUT_CAP)}"


def _extract_codex_compact_trace(
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    log_path = _codex_session_log_path(session_uuid)
    if log_path is None or not log_path.exists():
        return None

    turn_idx = 0
    lines_out: list[str] = []
    for ev in _read_jsonl_events(log_path):
        etype = ev.get("type")
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        if etype == "event_msg" and payload.get("type") == "user_message":
            turn_idx += 1
            label = (
                first_turn_label
                if turn_idx == 1 and first_turn_label
                else ("setup" if turn_idx == 1 else f"query {turn_idx - 1}")
            )
            lines_out.append("")
            lines_out.append(f"[Turn {turn_idx} — {label}]")
            text = str(payload.get("message") or "")
            if text:
                lines_out.append(f"  user: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")
        elif etype == "event_msg" and payload.get("type") == "agent_message":
            text = str(payload.get("message") or "")
            if text:
                lines_out.append(f"  assistant: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")
        elif etype == "response_item":
            ptype = payload.get("type")
            if ptype == "function_call":
                lines_out.append(f"  tool_use {_format_codex_tool_input(payload)}")
            elif ptype == "message" and payload.get("role") == "assistant":
                text = _string_from_content_items(payload.get("content"), input_text=False)
                if text:
                    lines_out.append(f"  assistant: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")

    trace = "\n".join(lines_out).strip()
    return trace or None


def extract_compact_trace(
    agent: str,
    workdir: Path,
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    if agent == "claude":
        return _extract_claude_compact_trace(workdir, session_uuid, first_turn_label=first_turn_label)
    if agent == "codex":
        return _extract_codex_compact_trace(session_uuid, first_turn_label=first_turn_label)
    return None


def _run_one_turn(
    *,
    agent: str,
    prompt: str,
    trial_id: str,
    entry_id: int,
    query_id: str,
    domain: str,
    is_setup: bool,
    turn_idx: int,
    workdir: Path,
    session_uuid: str,
    cmd: list[str],
    env: dict[str, str],
    timeout_s: int,
    model: str,
    doc_map: "DocIdMap | None" = None,
) -> TrialResult:
    """Execute one turn. Query turns (is_setup=False) expect the agent to write
    ./output.json; the setup turn does not."""
    out_path = workdir / "output.json"
    if out_path.exists():
        out_path.unlink()

    domain_tag = f"[{domain}] " if domain else ""
    label = "setup" if is_setup else f"entry_id={entry_id}, query_id={query_id}"
    logger.info("turn %d %s(%s)", turn_idx + 1, domain_tag, label)

    prior_codex_usage: dict[str, int] = {k: 0 for k in _CODEX_USAGE_FIELDS}
    if agent == "codex":
        prior_log = _codex_session_log_path(session_uuid)
        if prior_log is not None:
            prior_codex_usage = _extract_codex_total_usage(_read_jsonl_events(prior_log))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(
            trial_id=trial_id,
            condition=BASE_CONDITION,
            agent=agent,
            entry_id=entry_id,
            query_id=query_id,
            status="timeout",
            extraction_method="none",
            duration_ms=int((time.monotonic() - t0) * 1000),
            duration_api_ms=0,
            num_turns=turn_idx + 1,
            total_cost_usd=0.0,
            model_id=model,
            session_id=session_uuid,
            errors=[f"turn exceeded {timeout_s}s wall timeout"],
            is_setup=is_setup,
            domain=domain,
            cost_available=(agent == "claude"),
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    envelope: dict[str, Any] = {}
    codex_events: list[dict[str, Any]] = []
    token_events: list[dict[str, Any]] = []
    if agent == "claude":
        envelope = _parse_envelope(proc.stdout)
        agent_error = bool(envelope.get("is_error", False))
        duration_ms = int(envelope.get("duration_ms") or elapsed_ms)
        duration_api_ms = int(envelope.get("duration_api_ms") or 0)
        total_cost_usd = float(envelope.get("total_cost_usd") or 0.0)
        model_id = _extract_model_id(envelope, fallback=model)
        actual_session_id = str(envelope.get("session_id") or session_uuid)
    else:
        codex_events = _parse_jsonl_events(proc.stdout)
        agent_error = _codex_has_error(codex_events)
        duration_ms = elapsed_ms
        duration_api_ms = 0
        total_cost_usd = 0.0
        model_id = model
        log_path = _codex_session_log_path(session_uuid)
        if log_path is None:
            log_path = _codex_session_log_for_workdir(workdir)
        token_events = _read_jsonl_events(log_path) if log_path is not None else codex_events
        actual_session_id = _codex_session_id(
            token_events,
            fallback=_codex_session_id(codex_events, fallback=session_uuid),
        )

    stderr = proc.stderr.strip()
    result = TrialResult(
        trial_id=trial_id,
        condition=BASE_CONDITION,
        agent=agent,
        entry_id=entry_id,
        query_id=query_id,
        status="ok" if proc.returncode == 0 and not agent_error else "error",
        extraction_method="n/a" if is_setup else "output_json",
        duration_ms=duration_ms,
        duration_api_ms=duration_api_ms,
        num_turns=turn_idx + 1,
        total_cost_usd=total_cost_usd,
        model_id=model_id,
        session_id=actual_session_id,
        is_setup=is_setup,
        domain=domain,
        cost_available=(agent == "claude"),
    )
    if agent == "claude":
        _populate_claude_tokens(result, envelope)
    else:
        current_codex_usage = _extract_codex_total_usage(token_events or codex_events)
        _populate_codex_tokens(result, current_codex_usage, prior_codex_usage)
    if proc.returncode != 0:
        result.errors.append(f"non-zero exit {proc.returncode}")
    if agent == "claude" and envelope.get("is_error"):
        result.errors.append(f"envelope is_error: {envelope.get('subtype') or '?'}")
        detail = _extract_claude_error_detail(envelope)
        if detail:
            result.errors.append(f"claude error: {detail[:500]}")
    if agent == "codex" and agent_error:
        result.errors.append("codex event stream reported an error")
    if stderr:
        result.errors.append(f"stderr: {stderr[:500]}")

    if not is_setup:
        answer, ranked, extract_status, extract_errors = _parse_output_json(workdir)
        ranked = _apply_doc_map_to_ranked(ranked, doc_map)
        if extract_status in ("missing", "invalid_json"):
            result.extraction_method = extract_status
            if result.status == "ok":
                result.status = "extraction_failed"
        elif extract_status == "schema_warning":
            result.extraction_method = "schema_warning"
        result.final_answer = answer
        result.ranked_retrieved = ranked
        result.errors.extend(extract_errors)
        if out_path.exists():
            out_path.rename(workdir / f"output_e{entry_id}.json")

    return result


UNSCORABLE_JUDGE_ERRORS: frozenset[str] = frozenset(
    {"no_ground_truth", "empty_candidate", "scoring_mode_skip"}
)


def _apply_judge(ctx: Any, entry: DatasetEntry, result: TrialResult) -> None:
    """Score ``result`` against ``entry`` via the dispatched judge.

    Behaviour summary:
      - ``ctx is None``               -> no-op (judge disabled).
      - ``scoring_mode == "skip"``    -> terminal ``judge_error="scoring_mode_skip"``.
      - Answerable-mode entry with
        no ground truth               -> terminal ``judge_error="no_ground_truth"``.
      - Empty ``final_answer`` for a
        mode that requires an answer  -> terminal ``judge_error="empty_candidate"``.
      - Otherwise: dispatch via
        ``judging.evaluate_entry`` and stamp sub-scores onto ``result``.

    For simple-mode entries (and only for them) the ``answer_correctness``
    sub-score is also written to ``result.judge_score`` so existing summary /
    rescore code paths that read the flat 1-5 score keep working.

    The legacy ``LLMJudge`` runs in parallel when ``ctx.legacy_judge`` is set
    and the trial has both a ground-truth answer and a non-empty candidate.
    A failure in one judge does not block the other -- they use different
    prompts and may be sensitive to different inputs (e.g. context-window
    limits handled by ``truncate_for_judge``).
    """
    if ctx is None:
        return
    if entry.scoring_mode == "skip":
        result.judge_error = "scoring_mode_skip"
        return

    needs_answer_mode = entry.scoring_mode in ANSWER_REQUIRED_SCORING_MODES or entry.scoring_mode == ""
    answerable_mode = entry.scoring_mode in ANSWERABLE_SCORING_MODES or entry.scoring_mode == ""
    if answerable_mode and not entry.ground_truth_answer:
        result.judge_error = "no_ground_truth"
        return
    if needs_answer_mode and not result.final_answer:
        result.judge_error = "empty_candidate"
        return

    from skill_eval.judging import evaluate_entry, truncate_for_judge

    verdict = None
    try:
        verdict = evaluate_entry(
            client=ctx.client,
            entry=entry,
            result=result,
            simple_prompt_path=ctx.simple_prompt_path,
            scenario_prompt_path=ctx.scenario_prompt_path,
        )
    except Exception as exc:
        result.judge_error = f"judge_invocation_error: {exc}"
        logger.warning("evaluate_entry raised for entry_id=%s: %s", result.entry_id, exc, exc_info=True)

    if verdict is not None:
        result.judge_mode = verdict.mode
        result.judge_subscores = dict(verdict.sub_scores)
        result.judge_flags = dict(verdict.flags)
        result.judge_lists = dict(verdict.lists)
        result.judge_reasoning = verdict.rationale or ""
        if verdict.mode == "simple":
            result.judge_score = verdict.sub_scores.get("answer_correctness")
        if verdict.error:
            result.judge_error = verdict.error

    # Also run the legacy LLMJudge for cross-run comparability when possible.
    # Independent of the new judge: a failure in one does not block the other.
    if ctx.legacy_judge is not None and entry.ground_truth_answer and result.final_answer:
        try:
            legacy_verdict = ctx.legacy_judge.judge(
                query=entry.original_query,
                reference=entry.ground_truth_answer,
                candidate=truncate_for_judge(result.final_answer),
            )
        except Exception as exc:
            result.legacy_judge_error = f"legacy_judge_invocation_error: {exc}"
            logger.warning(
                "legacy LLMJudge raised for entry_id=%s: %s",
                result.entry_id,
                exc,
                exc_info=True,
            )
        else:
            result.legacy_judge_score = legacy_verdict.score
            result.legacy_judge_reasoning = legacy_verdict.reasoning or ""
            if legacy_verdict.error:
                result.legacy_judge_error = legacy_verdict.error


def run_session(
    *,
    agent: str,
    entries: list[DatasetEntry],
    workdir_root: Path,
    pdf_source: Path,
    model: str,
    budget_usd: float,
    timeout_s: int,
    domain: str = "",
    domain_label: str = "PDFs",
    judge: Any = None,
    testdata_prefixes: tuple[str, ...] = (),
    doc_map: "DocIdMap | None" = None,
    query_parallelism: int = 1,
) -> SessionRun:
    """Run one agent session covering setup + all `entries`.

    When ``query_parallelism == 1`` (default), turn 1 creates the agent
    session and subsequent turns resume it linearly.

    When ``query_parallelism > 1``, turn 1 still runs in the shared
    workdir, but each query turn afterwards runs in an isolated clone of
    that workdir (created via ``_copy_setup_workdir_for_query``) and
    starts a fresh agent session whose first turn is the query. A
    thread-pool fans these out up to ``query_parallelism`` at a time;
    results are reordered to manifest order before judging.

    All ``entries`` are expected to share the same ``domain`` (the caller
    groups by domain so each session sees a single PDF corpus).
    """
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"unsupported agent: {agent}")

    query_parallelism = max(1, int(query_parallelism or 1))
    execution_mode = "parallel_isolated" if query_parallelism > 1 else "linear_session"

    workdir = _build_session_workdir(agent, workdir_root, pdf_source, domain=domain, doc_map=doc_map)
    session_uuid = str(uuid.uuid4())
    env = os.environ.copy()
    logger.info(
        "starting %s session for %s: workdir=%s session_id=%s mode=%s query_parallelism=%d",
        agent,
        domain or "default",
        workdir,
        session_uuid,
        execution_mode,
        query_parallelism,
    )

    results: list[TrialResult] = []
    workdirs: list[Path] = [workdir]
    result_workdirs: dict[str, Path] = {}

    setup_trial_id = f"{agent}_{BASE_CONDITION}_{domain or 'default'}_setup_t1"
    setup_cmd = _build_command(
        agent=agent,
        model=model,
        budget_usd=budget_usd,
        session_uuid=session_uuid,
        workdir=workdir,
        resume=False,
    )
    setup_result = _run_one_turn(
        agent=agent,
        prompt=_render_setup_prompt(domain_label),
        trial_id=setup_trial_id,
        entry_id=0,
        query_id="",
        domain=domain,
        is_setup=True,
        turn_idx=0,
        workdir=workdir,
        session_uuid=session_uuid,
        cmd=setup_cmd,
        env=env,
        timeout_s=timeout_s,
        model=model,
        doc_map=doc_map,
    )
    setup_result.execution_mode = execution_mode
    results.append(setup_result)
    result_workdirs[setup_result.trial_id] = workdir

    run = SessionRun(
        workdir=workdir,
        workdirs=workdirs,
        results=results,
        result_workdirs=result_workdirs,
        execution_mode=execution_mode,
    )

    if setup_result.status != "ok":
        logger.warning(
            "setup turn failed for %s/%s; skipping %d query turns",
            agent,
            domain or "default",
            len(entries),
        )
        return run

    session_uuid = setup_result.session_id or session_uuid
    entries_by_id = {e.entry_id: e for e in entries}

    if query_parallelism <= 1:
        resume_cmd = _build_command(
            agent=agent,
            model=model,
            budget_usd=budget_usd,
            session_uuid=session_uuid,
            workdir=workdir,
            resume=True,
        )
        for i, entry in enumerate(entries):
            turn_idx = i + 1
            result = _run_one_turn(
                agent=agent,
                prompt=_render_prompt(entry, testdata_prefixes),
                trial_id=f"{agent}_{BASE_CONDITION}_{domain or 'default'}_e{entry.entry_id}_t{turn_idx + 1}",
                entry_id=entry.entry_id,
                query_id=entry.query_id,
                domain=domain,
                is_setup=False,
                turn_idx=turn_idx,
                workdir=workdir,
                session_uuid=session_uuid,
                cmd=resume_cmd,
                env=env,
                timeout_s=timeout_s,
                model=model,
                doc_map=doc_map,
            )
            result.execution_mode = execution_mode
            _apply_judge(judge, entry, result)
            results.append(result)
            result_workdirs[result.trial_id] = workdir
        return run

    # Parallel-isolated path: clone the post-setup workdir per query and
    # start a fresh agent session for each. Each clone keeps the ./pdfs/
    # symlink farm (symlinks=True in copytree) so we don't duplicate PDFs
    # on disk, but per-turn output.json files are excluded so each query
    # starts clean.
    _write_setup_context(workdir, agent=agent)

    query_contexts: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        turn_idx = i + 1
        query_workdir = _copy_setup_workdir_for_query(workdir, entry)
        workdirs.append(query_workdir)
        query_session_uuid = str(uuid.uuid4())
        query_contexts.append(
            {
                "entry": entry,
                "turn_idx": turn_idx,
                "workdir": query_workdir,
                "session_uuid": query_session_uuid,
                "cmd": _build_command(
                    agent=agent,
                    model=model,
                    budget_usd=budget_usd,
                    session_uuid=query_session_uuid,
                    workdir=query_workdir,
                    resume=False,
                ),
                "env": os.environ.copy(),
                "prompt": _render_isolated_query_prompt(entry, testdata_prefixes),
                "trial_id": f"{agent}_{BASE_CONDITION}_{domain or 'default'}_e{entry.entry_id}_t{turn_idx + 1}",
            }
        )

    def execute_query(ctx: dict[str, Any]) -> TrialResult:
        entry = ctx["entry"]
        result = _run_one_turn(
            agent=agent,
            prompt=ctx["prompt"],
            trial_id=ctx["trial_id"],
            entry_id=entry.entry_id,
            query_id=entry.query_id,
            domain=domain,
            is_setup=False,
            turn_idx=ctx["turn_idx"],
            workdir=ctx["workdir"],
            session_uuid=ctx["session_uuid"],
            cmd=ctx["cmd"],
            env=ctx["env"],
            timeout_s=timeout_s,
            model=model,
            doc_map=doc_map,
        )
        result.execution_mode = execution_mode
        return result

    query_results: list[TrialResult] = []
    if query_contexts:
        workers = min(query_parallelism, len(query_contexts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ctx = {executor.submit(execute_query, ctx): ctx for ctx in query_contexts}
            for future in as_completed(future_to_ctx):
                ctx = future_to_ctx[future]
                entry = ctx["entry"]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: subprocess timeouts are handled in _run_one_turn
                    result = TrialResult(
                        trial_id=ctx["trial_id"],
                        condition=BASE_CONDITION,
                        agent=agent,
                        entry_id=entry.entry_id,
                        query_id=entry.query_id,
                        status="error",
                        extraction_method="none",
                        duration_ms=0,
                        duration_api_ms=0,
                        num_turns=int(ctx["turn_idx"]) + 1,
                        total_cost_usd=0.0,
                        model_id=model,
                        session_id=str(ctx["session_uuid"]),
                        errors=[f"isolated query worker failed: {exc}"],
                        is_setup=False,
                        domain=domain,
                        cost_available=(agent == "claude"),
                        execution_mode=execution_mode,
                    )
                query_results.append(result)
                result_workdirs[result.trial_id] = ctx["workdir"]

    # Judging must be applied in deterministic (manifest) order — not in
    # whichever order the futures completed — so cross-run comparisons of
    # judge_score_n / mean line up.
    entry_order = {entry.entry_id: i for i, entry in enumerate(entries)}
    query_results.sort(key=lambda r: entry_order.get(r.entry_id, len(entry_order)))
    for result in query_results:
        entry = entries_by_id.get(result.entry_id)
        if entry is not None:
            _apply_judge(judge, entry, result)
    results.extend(query_results)
    return run


def save_trial(result: TrialResult, session_dir: Path) -> Path:
    parts = [session_dir, "trials", result.agent, result.condition]
    if result.domain:
        parts.append(result.domain)
    out = Path(*[str(p) for p in parts]) / f"{result.trial_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return out
