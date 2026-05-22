#!/usr/bin/env python3
"""Backfill per-turn token counts in codex skill_eval trial JSONs.

Earlier runs recorded ``last_token_usage`` (a single API call's count) into each
trial's ``input_tokens``/``output_tokens``/``cache_read_input_tokens``. The
correct per-turn cost is the delta of cumulative ``total_token_usage`` snapshots
taken before and after each ``codex exec resume`` invocation, which we can
reconstruct from the on-disk codex session log.

Usage::

    python backfill_codex_tokens.py <artifact_dir> [<artifact_dir> ...]
    python backfill_codex_tokens.py --dry-run <artifact_dir>

Each codex trial JSON is rewritten in place; pass ``--dry-run`` to print
proposed deltas without writing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TURN_IDX_RE = re.compile(r"_t(\d+)\.json$")
_USAGE_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")
_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"


def _turn_idx_from_name(name: str) -> int | None:
    m = _TURN_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _load_jsonl(path: Path) -> list[dict]:
    events = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _find_session_log(session_id: str, artifact_dir: Path) -> Path | None:
    # Prefer an archived copy under the artifact dir if present; otherwise fall
    # back to the per-user codex root. (This repo doesn't currently archive
    # codex logs into artifacts, but the archive-first lookup keeps this script
    # compatible if/when it does.)
    for matches in (
        list(artifact_dir.glob(f"trials/**/logs/rollout-*{session_id}*.jsonl")),
        list(_SESSIONS_ROOT.glob(f"**/rollout-*{session_id}*.jsonl")) if _SESSIONS_ROOT.exists() else [],
    ):
        if matches:
            return matches[0]
    return None


def _partition_events_by_task(events: list[dict]) -> list[list[dict]]:
    """Split events into per-task chunks bounded by ``task_started``.

    Each ``codex exec resume`` invocation emits exactly one ``task_started``,
    so chunks map 1:1 to skill_eval turns. Events before the first
    ``task_started`` (e.g. ``session_meta``) go into chunk 0.
    """
    chunks: list[list[dict]] = [[]]
    for ev in events:
        if (
            ev.get("type") == "event_msg"
            and isinstance(ev.get("payload"), dict)
            and ev["payload"].get("type") == "task_started"
        ):
            chunks.append([])
        chunks[-1].append(ev)
    if not chunks[0]:
        chunks = chunks[1:]  # no pre-task preamble; drop empty leading chunk
    elif len(chunks) > 1:
        # merge any pre-task events into the first turn's chunk
        chunks = [chunks[0] + chunks[1]] + chunks[2:]
    return chunks


def _last_total_usage(events: list[dict]) -> dict[str, int]:
    for ev in reversed(events):
        if ev.get("type") != "event_msg":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        usage = info.get("total_token_usage") or {}
        if isinstance(usage, dict):
            return {k: int(usage.get(k) or 0) for k in _USAGE_FIELDS}
    return {k: 0 for k in _USAGE_FIELDS}


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: max(0, after.get(k, 0) - before.get(k, 0)) for k in _USAGE_FIELDS}


def _backfill_artifact(artifact_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    trials_root = artifact_dir / "trials"
    if not trials_root.is_dir():
        print(f"[skip] {artifact_dir}: no trials/ dir", file=sys.stderr)
        return 0, 0

    # Group trial JSONs by session_id
    by_session: dict[str, list[Path]] = {}
    skipped_non_codex = 0
    for path in trials_root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("agent") != "codex":
            skipped_non_codex += 1
            continue
        sid = data.get("session_id")
        if not sid:
            continue
        by_session.setdefault(sid, []).append(path)

    if skipped_non_codex:
        print(f"  (skipped {skipped_non_codex} non-codex trial(s))")

    rewritten = 0
    sessions_missing_log = 0
    boundary_mismatches = 0
    for session_id, files in sorted(by_session.items()):
        log_path = _find_session_log(session_id, artifact_dir)
        if log_path is None:
            print(f"  [warn] no codex log for session {session_id}", file=sys.stderr)
            sessions_missing_log += 1
            continue

        # Sort trial files by turn index from filename
        files_sorted = sorted(files, key=lambda p: _turn_idx_from_name(p.name) or 0)
        n_trials = len(files_sorted)

        events = _load_jsonl(log_path)
        chunks = _partition_events_by_task(events)
        if len(chunks) != n_trials:
            print(
                f"  [warn] session {session_id}: {n_trials} trial(s) but "
                f"{len(chunks)} task_started chunk(s) in {log_path.name} — skipping",
                file=sys.stderr,
            )
            boundary_mismatches += 1
            continue

        running: dict[str, int] = {k: 0 for k in _USAGE_FIELDS}
        for trial_path, chunk in zip(files_sorted, chunks):
            after = _last_total_usage(chunk)
            # Some chunks have no token_count event (e.g. setup turns where codex
            # exits before any model call). Carry forward the prior cumulative so
            # the delta is zero.
            if all(after[k] == 0 for k in _USAGE_FIELDS):
                after = running
            delta = _delta(after, running)
            running = after

            data = json.loads(trial_path.read_text())
            old_out = data.get("output_tokens")
            old_in = data.get("input_tokens")
            data["input_tokens"] = delta["input_tokens"]
            data["output_tokens"] = delta["output_tokens"] + delta["reasoning_output_tokens"]
            data["cache_read_input_tokens"] = delta["cached_input_tokens"]
            data["cache_creation_input_tokens"] = 0

            if dry_run:
                print(
                    f"  {trial_path.relative_to(artifact_dir)}: "
                    f"in {old_in}->{data['input_tokens']}, "
                    f"out {old_out}->{data['output_tokens']}"
                )
            else:
                trial_path.write_text(json.dumps(data, indent=2))
                rewritten += 1

    if not dry_run:
        print(
            f"  rewrote {rewritten} trial(s) across {len(by_session)} session(s) "
            f"({sessions_missing_log} missing log, {boundary_mismatches} boundary mismatch)"
        )
    return rewritten, sessions_missing_log + boundary_mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifact_dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_rewritten = 0
    total_problems = 0
    for d in args.artifact_dirs:
        d = d.resolve()
        print(f"== {d}")
        if not d.is_dir():
            print(f"  [skip] not a directory", file=sys.stderr)
            continue
        rewritten, problems = _backfill_artifact(d, dry_run=args.dry_run)
        total_rewritten += rewritten
        total_problems += problems

    if args.dry_run:
        print("(dry run — no files written)")
    else:
        print(f"\nDONE: {total_rewritten} trial(s) rewritten, {total_problems} session issue(s).")
    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
