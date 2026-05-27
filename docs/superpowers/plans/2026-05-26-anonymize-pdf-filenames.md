# PDF Filename Anonymization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in feature that renames the per-session `./pdfs/` symlinks to opaque sha1-prefix names so the agent can't grep query terms in PDF basenames, while keeping `doc_id`s in trial JSON real so scoring, rescore, and reporting are unchanged.

**Architecture:** A per-session `DocIdMap` (built from the PDF source dir) drives symlink renaming in `_build_pdf_symlinks`. After the agent writes `output.json`, a small helper deanonymizes each `ranked_retrieved` entry's `doc_id` back to the real basename and stores the anonymized form on a sidecar field. The CLI controls activation via a config flag and `--anonymize-filenames` option, and persists a per-domain `doc_id_mapping.json` for debugging.

**Tech Stack:** Python 3.10+, stdlib `hashlib`, existing pydantic/typer/yaml stack. Tests use `pytest` (new dev extra).

**Spec:** `docs/superpowers/specs/2026-05-26-anonymize-pdf-filenames-design.md`

---

## File Structure

Files this plan touches:

- **Modify** `src/skill_eval/runner.py` — adds `DocIdMap` and `_apply_doc_map_to_ranked`, threads `doc_map` through `_build_pdf_symlinks`, `_build_session_workdir`, `_run_one_turn`, and `run_session`.
- **Modify** `src/skill_eval/cli.py` — resolves `anonymize_filenames` from config + CLI, builds `DocIdMap` per domain, passes it into `run_session`, writes per-domain `doc_id_mapping.json`.
- **Modify** `src/skill_eval/configs/skill_eval.yaml` — adds `anonymize_filenames: false` with explanatory comment.
- **Modify** `pyproject.toml` — adds a `dev` optional extra with `pytest` and a `[tool.pytest.ini_options]` block.
- **Modify** `README.md` — short paragraph in the CLI reference and a row in the config example.
- **Create** `tests/__init__.py` (empty).
- **Create** `tests/test_doc_id_map.py` — unit tests for the `DocIdMap` class.
- **Create** `tests/test_pdf_symlinks.py` — unit tests for `_build_pdf_symlinks` with and without `doc_map`.
- **Create** `tests/test_apply_doc_map.py` — unit tests for the ranked-retrieved rewrite helper.

The runner stays a single file because every new piece (the map, the symlink change, the ranked rewrite) is small and used in the same call path; splitting would obscure that. Tests live under `tests/` at repo root because there is no existing test tree.

---

## Task 1: Set up pytest

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add a `dev` extra and pytest config to `pyproject.toml`**

Append after the existing `[project.optional-dependencies]` block (inside the same table), then add the pytest config table at end of file:

```toml
[project.optional-dependencies]
# (existing keys preserved)
dev = [
  "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 2: Create empty test package**

Create `tests/__init__.py` with a single trailing newline (empty file). This is enough to make pytest treat the `tests/` directory as a package without further configuration.

- [ ] **Step 3: Sync the dev extra**

Run: `uv sync --extra dev`
Expected: pytest is installed; no other errors.

- [ ] **Step 4: Confirm pytest collects zero tests**

Run: `uv run --extra dev pytest`
Expected: exit code 5 ("no tests collected"). This proves the wiring works before any tests exist.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py
git commit -m "test: scaffold pytest dev extra"
```

---

## Task 2: Add `DocIdMap` (TDD)

**Files:**
- Modify: `src/skill_eval/runner.py`
- Create: `tests/test_doc_id_map.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doc_id_map.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from skill_eval.runner import DocIdMap


def _expected_anon(stem: str) -> str:
    return "doc_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]


def test_anonymize_returns_stable_doc_prefix():
    m = DocIdMap.from_stems(["Acme_10K_2024", "Other_File"])
    assert m.anonymize("Acme_10K_2024") == _expected_anon("Acme_10K_2024")


def test_round_trip_real_anon_real():
    m = DocIdMap.from_stems(["Acme_10K_2024", "Other_File"])
    anon = m.anonymize("Acme_10K_2024")
    assert m.deanonymize(anon) == "Acme_10K_2024"


def test_deanonymize_unknown_returns_input_unchanged():
    m = DocIdMap.from_stems(["Acme_10K_2024"])
    assert m.deanonymize("doc_deadbeef99") == "doc_deadbeef99"
    assert m.deanonymize("Acme_10K_2024") == "Acme_10K_2024"


def test_from_pdf_dir_uses_stems(tmp_path):
    (tmp_path / "Foo.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "Bar.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "not_a_pdf.txt").write_text("ignored")
    m = DocIdMap.from_pdf_dir(tmp_path)
    assert set(m.real_to_anon.keys()) == {"Foo", "Bar"}


def test_collision_raises(monkeypatch):
    # Force every hash to the same prefix so any two stems collide.
    def fake_hex(_data: bytes) -> str:
        class _H:
            def hexdigest(self_) -> str:
                return "ff" * 20
        return _H()

    monkeypatch.setattr(
        "skill_eval.runner.hashlib.sha1",
        lambda data: fake_hex(data),
    )
    with pytest.raises(ValueError, match="collision"):
        DocIdMap.from_stems(["A", "B"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_doc_id_map.py -v`
Expected: collection error or ImportError because `DocIdMap` doesn't exist yet.

- [ ] **Step 3: Implement `DocIdMap` in `runner.py`**

In `src/skill_eval/runner.py`, add `hashlib` to the imports near the top (after `functools`):

```python
import hashlib
```

Then add the `DocIdMap` class immediately after the existing `TrialResult` dataclass (just above the `_remap_pdf_paths` function):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_doc_id_map.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_eval/runner.py tests/test_doc_id_map.py
git commit -m "feat(runner): add DocIdMap for opaque PDF basenames"
```

---

## Task 3: Wire `doc_map` into `_build_pdf_symlinks` (TDD)

**Files:**
- Modify: `src/skill_eval/runner.py` (the `_build_pdf_symlinks` function near line 96)
- Create: `tests/test_pdf_symlinks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pdf_symlinks.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from skill_eval.runner import DocIdMap, _build_pdf_symlinks


def _make_source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "Acme_10K_2024.pdf").write_bytes(b"%PDF-1.4\n")
    (src / "Globex_Report.pdf").write_bytes(b"%PDF-1.4\n")
    return src


def test_no_doc_map_preserves_real_basenames(tmp_path):
    src = _make_source_dir(tmp_path)
    dest = tmp_path / "dest"
    _build_pdf_symlinks(src, dest)
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["Acme_10K_2024.pdf", "Globex_Report.pdf"]


def test_doc_map_renames_symlinks_to_anonymized_ids(tmp_path):
    src = _make_source_dir(tmp_path)
    dest = tmp_path / "dest"
    doc_map = DocIdMap.from_pdf_dir(src)
    _build_pdf_symlinks(src, dest, doc_map)
    names = sorted(p.name for p in dest.iterdir())
    for name in names:
        assert name.startswith("doc_") and name.endswith(".pdf")
    # Every symlink resolves to one of the real PDFs.
    real_paths = {(src / f"{stem}.pdf").resolve() for stem in doc_map.real_to_anon}
    for link in dest.iterdir():
        assert link.is_symlink()
        assert link.resolve() in real_paths


def test_doc_map_round_trip_via_symlink_name(tmp_path):
    src = _make_source_dir(tmp_path)
    dest = tmp_path / "dest"
    doc_map = DocIdMap.from_pdf_dir(src)
    _build_pdf_symlinks(src, dest, doc_map)
    for stem, anon in doc_map.real_to_anon.items():
        link = dest / f"{anon}.pdf"
        assert link.resolve().name == f"{stem}.pdf"
```

- [ ] **Step 2: Run the tests to verify the first one passes and the rest fail**

Run: `uv run --extra dev pytest tests/test_pdf_symlinks.py -v`
Expected: `test_no_doc_map_preserves_real_basenames` passes (existing behavior is already correct); the other two fail with a `TypeError` from `_build_pdf_symlinks` not accepting `doc_map`.

- [ ] **Step 3: Update `_build_pdf_symlinks` to accept an optional `DocIdMap`**

In `src/skill_eval/runner.py`, replace the existing `_build_pdf_symlinks` function with:

```python
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
```

- [ ] **Step 4: Run tests to verify they all pass**

Run: `uv run --extra dev pytest tests/test_pdf_symlinks.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_eval/runner.py tests/test_pdf_symlinks.py
git commit -m "feat(runner): rename pdf symlinks via DocIdMap when supplied"
```

---

## Task 4: Thread `doc_map` through `_build_session_workdir`

**Files:**
- Modify: `src/skill_eval/runner.py` (the `_build_session_workdir` function around line 105)

- [ ] **Step 1: Update the function signature and body**

Replace the existing `_build_session_workdir` with:

```python
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
```

- [ ] **Step 2: Verify the previous tests still pass**

Run: `uv run --extra dev pytest -v`
Expected: 8 tests PASS (no regressions; `_build_session_workdir` isn't covered by its own test but is exercised indirectly when run_session is invoked end-to-end — not needed yet).

- [ ] **Step 3: Commit**

```bash
git add src/skill_eval/runner.py
git commit -m "refactor(runner): thread doc_map through _build_session_workdir"
```

---

## Task 5: Add `_apply_doc_map_to_ranked` (TDD)

**Files:**
- Modify: `src/skill_eval/runner.py`
- Create: `tests/test_apply_doc_map.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_apply_doc_map.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skill_eval.runner import DocIdMap, _apply_doc_map_to_ranked


def test_no_op_when_doc_map_is_none():
    ranked = [{"doc_id": "Foo", "page_number": 0, "rank": 1}]
    out = _apply_doc_map_to_ranked(ranked, None)
    assert out == ranked
    # The function may return the same list reference, but content must be unchanged.


def test_translates_known_anonymized_ids():
    doc_map = DocIdMap.from_stems(["Acme_10K_2024"])
    anon = doc_map.anonymize("Acme_10K_2024")
    ranked = [{"doc_id": anon, "page_number": 5, "rank": 1}]
    out = _apply_doc_map_to_ranked(ranked, doc_map)
    assert out == [
        {
            "doc_id": "Acme_10K_2024",
            "page_number": 5,
            "rank": 1,
            "anonymized_doc_id": anon,
        }
    ]


def test_unknown_anonymized_id_passes_through_with_sidecar():
    doc_map = DocIdMap.from_stems(["Acme_10K_2024"])
    ranked = [{"doc_id": "not_a_real_id", "page_number": 0, "rank": 1}]
    out = _apply_doc_map_to_ranked(ranked, doc_map)
    assert out == [
        {
            "doc_id": "not_a_real_id",
            "page_number": 0,
            "rank": 1,
            "anonymized_doc_id": "not_a_real_id",
        }
    ]


def test_preserves_unrelated_keys_and_order():
    doc_map = DocIdMap.from_stems(["A", "B"])
    a, b = doc_map.anonymize("A"), doc_map.anonymize("B")
    ranked = [
        {"doc_id": b, "page_number": 1, "rank": 1, "score": 0.9},
        {"doc_id": a, "page_number": 2, "rank": 2, "score": 0.4},
    ]
    out = _apply_doc_map_to_ranked(ranked, doc_map)
    assert [e["doc_id"] for e in out] == ["B", "A"]
    assert out[0]["score"] == 0.9
    assert out[0]["anonymized_doc_id"] == b
    assert out[1]["anonymized_doc_id"] == a
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_apply_doc_map.py -v`
Expected: ImportError or collection error because `_apply_doc_map_to_ranked` doesn't exist yet.

- [ ] **Step 3: Implement `_apply_doc_map_to_ranked` and wire it into `_run_one_turn`**

In `src/skill_eval/runner.py`, add this function right after `_parse_output_json` (around line 402):

```python
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
```

Then wire it into `_run_one_turn`. In `src/skill_eval/runner.py`, find the block near line 808 that reads:

```python
    if not is_setup:
        answer, ranked, extract_status, extract_errors = _parse_output_json(workdir)
```

and replace the entire `if not is_setup:` block with:

```python
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
```

Also add a `doc_map: "DocIdMap | None" = None` parameter to `_run_one_turn`. Find the signature block:

```python
def _run_one_turn(
    *,
    agent: str,
    prompt: str,
    ...
    model: str,
) -> TrialResult:
```

and add `doc_map: "DocIdMap | None" = None` as the last parameter (just before the closing `) -> TrialResult:`).

- [ ] **Step 4: Run all tests to verify**

Run: `uv run --extra dev pytest -v`
Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_eval/runner.py tests/test_apply_doc_map.py
git commit -m "feat(runner): translate ranked_retrieved through DocIdMap"
```

---

## Task 6: Thread `doc_map` through `run_session`

**Files:**
- Modify: `src/skill_eval/runner.py` (the `run_session` function near line 861)

- [ ] **Step 1: Update `run_session` signature**

Add `doc_map: "DocIdMap | None" = None` as the last keyword-only parameter to `run_session`. Existing signature ends:

```python
    judge: Any = None,
    testdata_prefixes: tuple[str, ...] = (),
) -> tuple[Path, list[TrialResult]]:
```

Replace with:

```python
    judge: Any = None,
    testdata_prefixes: tuple[str, ...] = (),
    doc_map: "DocIdMap | None" = None,
) -> tuple[Path, list[TrialResult]]:
```

- [ ] **Step 2: Pass `doc_map` to `_build_session_workdir`**

Find:

```python
    workdir = _build_session_workdir(agent, workdir_root, pdf_source, domain=domain)
```

Replace with:

```python
    workdir = _build_session_workdir(agent, workdir_root, pdf_source, domain=domain, doc_map=doc_map)
```

- [ ] **Step 3: Pass `doc_map` into both `_run_one_turn` calls**

Find the setup-turn call (`setup_result = _run_one_turn(`) and add `doc_map=doc_map,` as a new keyword argument before the closing `)`. Do the same for the query-turn call inside the `for i, entry in enumerate(entries):` loop.

- [ ] **Step 4: Run all tests**

Run: `uv run --extra dev pytest -v`
Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skill_eval/runner.py
git commit -m "refactor(runner): pipe doc_map through run_session"
```

---

## Task 7: Add `anonymize_filenames` config + CLI option + per-domain mapping file

**Files:**
- Modify: `src/skill_eval/cli.py`
- Modify: `src/skill_eval/configs/skill_eval.yaml`

- [ ] **Step 1: Add config default to the packaged YAML**

In `src/skill_eval/configs/skill_eval.yaml`, insert after the `testdata_prefixes:` block (around line 52) and before the `# Agent selection ...` section header:

```yaml
# ---------------------------------------------------------------------------
# OPTIONAL — anonymize PDF filenames in the agent's workdir
# ---------------------------------------------------------------------------
# When true, the per-domain ``./pdfs/`` symlinks are renamed to opaque
# ``doc_<sha1[:10]>.pdf`` so the agent cannot grep query terms out of the
# basenames. Real ``doc_id`` values still appear in ``trials/.../*.json``,
# so scoring, ``rescore``, and external tooling are unaffected. A
# per-domain ``doc_id_mapping.json`` is written under the trial dir for
# debugging.
anonymize_filenames: false
```

- [ ] **Step 2: Add import and CLI option to `cli.py`**

In `src/skill_eval/cli.py`, extend the import from `skill_eval.runner` to include `DocIdMap`. Find the existing import block:

```python
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
```

Add `DocIdMap,` alphabetically (after `BASE_CONDITION,`).

Then add a CLI option to `run_command`. Find the existing `model_override` option and insert this right after it:

```python
    anonymize_filenames: Optional[bool] = typer.Option(
        None,
        "--anonymize-filenames/--no-anonymize-filenames",
        help="Rename ./pdfs/ symlinks to opaque doc_<sha1[:10]>.pdf so the agent "
             "cannot grep query terms from filenames. Overrides config.anonymize_filenames.",
    ),
```

- [ ] **Step 3: Resolve the flag in `run_command`**

In `run_command`, find the line:

```python
    testdata_prefixes = tuple(str(p) for p in testdata_prefixes_raw)
```

Immediately after it, add:

```python
    anonymize = (
        anonymize_filenames
        if anonymize_filenames is not None
        else bool(cfg.get("anonymize_filenames", False))
    )
```

Then find the line that echoes the agent header:

```python
    typer.echo(f"Agent: {agent}  model={model}  condition={BASE_CONDITION}")
```

Immediately after it, add:

```python
    typer.echo(f"Anonymize filenames: {'on' if anonymize else 'off'}")
```

Also include the resolved value in the persisted config block. Find:

```python
    resolved_cfg = dict(cfg)
    resolved_cfg["agent"] = agent
    resolved_cfg["agent_model"] = model
```

and append:

```python
    resolved_cfg["anonymize_filenames"] = anonymize
```

- [ ] **Step 4: Build `DocIdMap` per domain and pass it to `run_session`; persist `doc_id_mapping.json`**

In the `for domain in domain_order:` loop in `run_command`, find the existing call to `run_session` (it starts with `workdir, results = run_session(`). Immediately before that call, add:

```python
        doc_map = DocIdMap.from_pdf_dir(pdf_source) if anonymize else None
```

Then in the `run_session(...)` call, add a new keyword argument right after `testdata_prefixes=testdata_prefixes,`:

```python
            doc_map=doc_map,
```

After the existing `for r in results:` loop (the one that calls `save_trial(r, session_dir)` and prints the per-turn line) but before the recall block that begins `entries_by_id = ...`, add:

```python
        if doc_map is not None:
            mapping_dir = session_dir / "trials" / agent / BASE_CONDITION / domain
            mapping_dir.mkdir(parents=True, exist_ok=True)
            (mapping_dir / "doc_id_mapping.json").write_text(
                json.dumps({"real_to_anon": doc_map.real_to_anon}, indent=2) + "\n",
                encoding="utf-8",
            )
```

- [ ] **Step 5: Run all tests, plus a manual smoke check of the CLI parser**

Run: `uv run --extra dev pytest -v`
Expected: 12 tests PASS.

Run: `uv run skill-eval run --help`
Expected: help text includes `--anonymize-filenames/--no-anonymize-filenames` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/skill_eval/cli.py src/skill_eval/configs/skill_eval.yaml
git commit -m "feat(cli): add --anonymize-filenames flag and per-domain mapping file"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the flag in the CLI reference and the config**

In `README.md`, find the `### \`skill-eval run\`` options table (around line 279). Add a new row to the table, just after the `--model` row:

```text
| `--anonymize-filenames / --no-anonymize-filenames` | `cfg.anonymize_filenames` or `false` | Rename `./pdfs/` symlinks to opaque `doc_<sha1[:10]>.pdf` so the agent can't grep query terms from filenames. Real `doc_id`s still appear in trial JSON. |
```

Then in the "Example config" YAML block (around line 174), insert after the `testdata_prefixes:` line:

```yaml
anonymize_filenames: false
```

And in the "Output Layout" section, extend the tree to mention the optional mapping file:

```text
                |-- claude_c1_base_vidore_v3_finance_en_setup_t1.json
                |-- claude_c1_base_vidore_v3_finance_en_e1_t2.json
                |-- doc_id_mapping.json   # only when --anonymize-filenames
                `-- ...
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document --anonymize-filenames in README"
```

---

## Self-Review Checklist

After completing all tasks, verify:

- [ ] `uv run --extra dev pytest -v` shows 12 tests passing.
- [ ] `uv run skill-eval run --help` lists `--anonymize-filenames/--no-anonymize-filenames`.
- [ ] `git log --oneline` shows ~7 commits, one per task.
- [ ] No `TODO` / `TBD` strings introduced (`git grep -nE 'TODO|TBD' src/skill_eval/`).
- [ ] `_apply_doc_map_to_ranked` is called from `_run_one_turn` exactly once (`git grep -n _apply_doc_map_to_ranked src/`).
- [ ] When `anonymize_filenames` is unset/false, the trial JSON schema is unchanged — no `anonymized_doc_id` keys appear and no `doc_id_mapping.json` is written. (Spot-check by inspecting a previous artifact directory or by running an off-mode smoke test if a real dataset is available.)
