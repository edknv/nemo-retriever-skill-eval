# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trial runner: build sandboxed workdir, spawn `claude -p`, parse outputs."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Optional

from nr_skill_eval.dataset import DatasetEntry

logger = logging.getLogger(__name__)

CONDITIONS = ("c1_base", "c2_retriever", "c3_retriever_skill")


@functools.lru_cache(maxsize=8)
def _load_prompt_template(name: str) -> str:
    return Path(str(pkg_files("nr_skill_eval").joinpath(f"prompts/{name}"))).read_text(encoding="utf-8")


@dataclass
class TrialResult:
    trial_id: str
    condition: str
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
    retriever_first_use_turn: int | None = None
    retriever_used_ever: bool = False
    skill_fired: bool | None = None
    is_setup: bool = False
    domain: str = ""
    judge_score: int | None = None
    judge_reasoning: str = ""
    judge_error: str = ""
    tool_use_summary: str = ""


def _remap_pdf_paths(text: str, prefixes: tuple[str, ...]) -> str:
    """Rewrite caller-supplied path prefixes in *text* to ``./pdfs/``.

    Some agent-eval manifests' paraphrased prompts hard-code paths from the
    dataset source tree. Each trial workdir symlinks the domain's PDFs to
    ``./pdfs/``, so the agent only needs the basename — rewriting the prefix
    lets the natural-language reference resolve to a real file.

    Prefixes are configured per-run via the ``testdata_prefixes`` config key
    (no dataset paths are hardcoded in this module).
    """
    for prefix in prefixes:
        text = text.replace(prefix, "./pdfs")
    return text


def _render_prompt(entry: DatasetEntry, condition: str, testdata_prefixes: tuple[str, ...] = ()) -> str:
    tpl_name = "trial_user_slash.j2" if condition == "c3_retriever_skill" else "trial_user_nl.j2"
    text = _load_prompt_template(tpl_name)
    return text.replace(
        "{{ paraphrased_prompt }}", _remap_pdf_paths(entry.paraphrased_prompt, testdata_prefixes)
    ).replace("{{ original_query }}", entry.original_query)


def _render_setup_prompt(condition: str, domain_label: str = "PDFs") -> str:
    tpl_name = "setup_slash.j2" if condition == "c3_retriever_skill" else "setup_nl.j2"
    text = _load_prompt_template(tpl_name)
    return text.replace("{{ domain_label }}", domain_label)


def _build_pdf_symlinks(pdf_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_source.glob("*.pdf")):
        target = dest / pdf.name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(pdf.resolve())


def _copy_skill(skill_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "SKILL.md").exists():
        return
    shutil.copy2(skill_source / "SKILL.md", dest / "SKILL.md")
    ref_src = skill_source / "references"
    if ref_src.is_dir():
        shutil.copytree(ref_src, dest / "references", dirs_exist_ok=True)


def _build_condition_workdir(
    condition: str,
    root: Path,
    pdf_source: Path,
    skill_source: Optional[Path],
    domain: str = "",
) -> Path:
    """Build one workdir per condition. Shared across all turns in the session.

    Workdir contents:
      - pdfs/ symlink farm into the source PDF folder
      - .claude/ sandbox (empty settings + per-condition skill copy)

    c1_base is stock Claude Code with no skill loaded and slash commands
    disabled, but otherwise full access to whatever is on the host. The agent
    itself creates any retrieval artifacts (e.g. ./lancedb/) inside the workdir
    on the setup turn.
    """
    domain_seg = f"_{domain}" if domain else ""
    workdir = root / f"{condition}{domain_seg}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_pdf_symlinks(pdf_source, workdir / "pdfs")
    (workdir / ".claude").mkdir(parents=True, exist_ok=True)
    (workdir / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    # c2 and c3 both have the nemo-retriever skill loaded.
    # The c2/c3 distinction is purely the prompt style (NL vs explicit slash command).
    if condition in ("c2_retriever", "c3_retriever_skill"):
        if skill_source is None:
            raise ValueError(f"condition '{condition}' requires skill_source_dir to be set in the config")
        _copy_skill(skill_source, workdir / ".claude" / "skills" / "nemo-retriever")
    return workdir


def cleanup_condition_workdir(workdir: Path) -> None:
    """Remove a condition's scratch workdir (PDFs symlinks, .claude/, agent-built
    artifacts like .venv/, lancedb/, scratch scripts). Called after a session
    completes and its results have been persisted to the artifact dir.
    """
    if not workdir.exists():
        return
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("cleaned up workdir %s", workdir)


def _build_command(
    condition: str,
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
    ]
    if resume:
        cmd.extend(["--resume", session_uuid])
    else:
        cmd.extend(["--session-id", session_uuid])
    # c1 disables slash commands so it cannot invoke the nemo-retriever skill;
    # c2 has the skill loaded but uses NL prompt (description-based auto-
    # discovery); c3 explicitly invokes via slash command.
    if condition == "c1_base":
        cmd.append("--disable-slash-commands")
    return cmd


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


def _populate_tokens(result: TrialResult, envelope: dict[str, Any]) -> None:
    usage = envelope.get("usage") or {}
    result.input_tokens = int(usage.get("input_tokens") or 0)
    result.output_tokens = int(usage.get("output_tokens") or 0)
    result.cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)
    result.cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_detail = usage.get("cache_creation") or {}
    result.ephemeral_5m_input_tokens = int(cache_detail.get("ephemeral_5m_input_tokens") or 0)
    result.ephemeral_1h_input_tokens = int(cache_detail.get("ephemeral_1h_input_tokens") or 0)


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


def _extract_model_id(envelope: dict[str, Any], fallback: str) -> str:
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        return next(iter(model_usage.keys()))
    return str(envelope.get("model") or fallback)


_PIPELINE_SEP = re.compile(r"(?:;|&&|\|\||\||\n|\$\(|`)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPER_CMDS = {"sudo", "time", "nice", "nohup", "exec", "env", "command", "builtin"}


def _retriever_in_command(cmd: str) -> bool:
    """Does this shell command line invoke the retriever CLI as a command?

    Matches when the **executable** in any pipeline segment is the retriever
    CLI — ``retriever``, ``./retriever``, ``/abs/path/retriever``, ``uv run
    retriever``, or ``python -m nemo_retriever``. Deliberately does *not*
    match cases where ``retriever`` appears only as a path argument (e.g.
    ``cat .bin/retriever``, ``ls /path/retriever/``, ``echo "use retriever"``).
    """
    if not cmd:
        return False

    for segment in _PIPELINE_SEP.split(cmd):
        seg = segment.strip()
        # Strip leading env-var assignments and command wrappers (sudo, time, ...).
        while seg:
            first = seg.split(None, 1)
            if not first:
                break
            head = first[0]
            rest = first[1] if len(first) > 1 else ""
            if _ENV_ASSIGN.match(head):
                seg = rest
                continue
            if head in _WRAPPER_CMDS:
                seg = rest
                continue
            break
        if not seg:
            continue
        head = seg.split(None, 1)[0]
        if head == "retriever" or head == "./retriever":
            return True
        if head.endswith("/retriever") and "/" in head[: -len("/retriever") + 1]:
            # An absolute or relative path whose final component is `retriever`,
            # e.g. /home/.../venv/bin/retriever. Reject pure ``/retriever`` which
            # is implausible as a real binary path.
            return True
        # ``uv run retriever ...`` and ``python -m nemo_retriever ...`` —
        # check the first two tokens of the segment.
        tokens = seg.split()
        if len(tokens) >= 3 and tokens[0] == "uv" and tokens[1] == "run" and tokens[2] == "retriever":
            return True
        if len(tokens) >= 3 and tokens[0].startswith("python") and tokens[1] == "-m" and tokens[2].startswith("nemo_retriever"):
            return True
    return False


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
    # Generic fallback: dump key=value pairs, truncated.
    parts = [f"{k}={_truncate(str(v), 80)}" for k, v in inp.items()]
    return f"{name}: " + " ".join(parts) if parts else name


def _extract_compact_trace(workdir: Path, session_uuid: str) -> str | None:
    """Walk the Claude Code session JSONL and emit a turn-organized text trace.

    The trace lists, per turn: the user prompt, every ``tool_use`` invocation
    (Bash/Read/Grep/Glob/Edit/Write/…) with truncated inputs, and the agent's
    final assistant text. ``tool_result`` content is omitted on purpose — the
    summarizer needs the *actions*, not the noise of tool outputs.

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
                    # Flush any pending assistant text from the prior turn.
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
                    label = "setup" if turn_idx == 1 else f"query {turn_idx - 1}"
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


def _scan_transcript_for_signals(
    envelope: dict[str, Any],
    workdir: Path | None = None,
    session_uuid: str | None = None,
) -> tuple[int | None, bool]:
    """Detect whether the agent invoked the ``retriever`` CLI.

    Primary signal: scan the Claude Code session jsonl for tool-use entries that
    spawn a shell command containing ``retriever``. This catches every actual
    invocation, regardless of whether the agent quoted it in its final reply.

    Fallback signal: if the session log isn't accessible (older runs, missing
    file), look for ``retriever`` in the envelope's ``result`` text — the legacy
    proxy. This undercounts but never overcounts.
    """
    # Primary: tool-call trace.
    if workdir is not None and session_uuid:
        log_path = _claude_session_log_path(workdir, session_uuid)
        if log_path.exists():
            try:
                with log_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = ev.get("message") or {}
                        content = msg.get("content")
                        if not isinstance(content, list):
                            continue
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") != "tool_use":
                                continue
                            if item.get("name") != "Bash":
                                continue
                            cmd = (item.get("input") or {}).get("command") or ""
                            if _retriever_in_command(cmd):
                                return 1, True
                return None, False
            except OSError:
                pass  # fall through to fallback

    # Fallback: scan the assistant's final text.
    text = str(envelope.get("result") or "")
    used = "retriever " in text or "\nretriever\n" in text
    return (1 if used else None), used


def _run_one_turn(
    *,
    condition: str,
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
) -> TrialResult:
    """Execute one turn. Query turns (is_setup=False) expect the agent to write
    ./output.json; the setup turn does not."""
    out_path = workdir / "output.json"
    if out_path.exists():
        out_path.unlink()

    domain_tag = f"[{domain}] " if domain else ""
    label = "setup" if is_setup else f"entry_id={entry_id}, query_id={query_id}"
    logger.info("turn %d for %s %s(%s)", turn_idx + 1, condition, domain_tag, label)
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
            condition=condition,
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
        )

    envelope = _parse_envelope(proc.stdout)
    stderr = proc.stderr.strip()
    result = TrialResult(
        trial_id=trial_id,
        condition=condition,
        entry_id=entry_id,
        query_id=query_id,
        status="ok" if proc.returncode == 0 and not envelope.get("is_error", False) else "error",
        extraction_method="n/a" if is_setup else "output_json",
        duration_ms=int(envelope.get("duration_ms") or (time.monotonic() - t0) * 1000),
        duration_api_ms=int(envelope.get("duration_api_ms") or 0),
        num_turns=turn_idx + 1,
        total_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        model_id=_extract_model_id(envelope, fallback=model),
        session_id=str(envelope.get("session_id") or session_uuid),
        is_setup=is_setup,
        domain=domain,
    )
    _populate_tokens(result, envelope)
    if proc.returncode != 0:
        result.errors.append(f"non-zero exit {proc.returncode}")
    if envelope.get("is_error"):
        result.errors.append(f"envelope is_error: {envelope.get('subtype') or '?'}")
    if stderr:
        result.errors.append(f"stderr: {stderr[:500]}")

    if not is_setup:
        answer, ranked, extract_status, extract_errors = _parse_output_json(workdir)
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

    first_use, used = _scan_transcript_for_signals(envelope, workdir=workdir, session_uuid=session_uuid)
    result.retriever_first_use_turn = first_use
    result.retriever_used_ever = used
    # c1 has the skill unavailable; leave skill_fired=None to distinguish from "loaded but didn't fire".
    if condition in ("c2_retriever", "c3_retriever_skill"):
        result.skill_fired = used and (first_use is not None) and first_use <= 2
    return result


def _apply_judge(judge: Any, entry: DatasetEntry, result: TrialResult) -> None:
    """Score ``result.final_answer`` against ``entry.ground_truth_answer``.

    Mutates the result in place. Skips silently when the judge is unset, the
    ground-truth answer is empty, or the trial didn't produce a final answer.
    Errors are recorded on the result rather than raised so a flaky judge
    endpoint never breaks an in-flight session.
    """
    if judge is None or not entry.ground_truth_answer or not result.final_answer:
        return
    try:
        verdict = judge.judge(
            query=entry.original_query,
            reference=entry.ground_truth_answer,
            candidate=result.final_answer,
        )
    except Exception as exc:  # defensive — LLMJudge already catches, but be safe.
        result.judge_error = f"judge_invocation_error: {exc}"
        logger.warning("LLMJudge raised for entry_id=%s: %s", result.entry_id, exc, exc_info=True)
        return
    result.judge_score = verdict.score
    result.judge_reasoning = verdict.reasoning or ""
    if verdict.error:
        result.judge_error = verdict.error


def run_condition(
    *,
    condition: str,
    entries: list[DatasetEntry],
    workdir_root: Path,
    pdf_source: Path,
    skill_source: Optional[Path],
    model: str,
    budget_usd: float,
    timeout_s: int,
    domain: str = "",
    domain_label: str = "PDFs",
    judge: Any = None,
    testdata_prefixes: tuple[str, ...] = (),
) -> tuple[Path, list[TrialResult]]:
    """Run one Claude Code session covering setup + all `entries` for `condition`.

    Turn 1 creates the session via --session-id; subsequent turns resume it. The
    first TrialResult has is_setup=True; the rest are query results, one per entry.
    All ``entries`` are expected to share the same ``domain`` (the caller groups
    by domain so each session sees a single PDF corpus).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    workdir = _build_condition_workdir(condition, workdir_root, pdf_source, skill_source, domain=domain)
    session_uuid = str(uuid.uuid4())
    env = os.environ.copy()
    logger.info(
        "starting session for %s/%s: workdir=%s session_id=%s",
        condition,
        domain or "default",
        workdir,
        session_uuid,
    )

    results: list[TrialResult] = []

    setup_trial_id = f"{condition}_{domain or 'default'}_setup_t1"
    setup_cmd = _build_command(condition, model, budget_usd, session_uuid, workdir, resume=False)
    setup_result = _run_one_turn(
        condition=condition,
        prompt=_render_setup_prompt(condition, domain_label),
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
    )
    results.append(setup_result)

    resume_cmd = _build_command(condition, model, budget_usd, session_uuid, workdir, resume=True)
    for i, entry in enumerate(entries):
        turn_idx = i + 1
        result = _run_one_turn(
            condition=condition,
            prompt=_render_prompt(entry, condition, testdata_prefixes),
            trial_id=f"{condition}_{domain or 'default'}_e{entry.entry_id}_t{turn_idx + 1}",
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
        )
        _apply_judge(judge, entry, result)
        results.append(result)
    return workdir, results


def save_trial(result: TrialResult, session_dir: Path) -> Path:
    parts = [session_dir, "trials", result.condition]
    if result.domain:
        parts.append(result.domain)
    out = Path(*[str(p) for p in parts]) / f"{result.trial_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return out
