# CLAUDE.md: Phantom Packages

DEV Hacktoberfest 2026 Open-Source AI Challenge, Week 4 ("Ghost in the Machine"), tag `#hf26challenge`.

**Deadline:** Oct 30 2026, 11:59 PM PDT. Internal target: Oct 30, 8 PM PDT.

The authoritative spec is `docs/03_BUILD_SPEC.md`. The blog needs are in `docs/05_BLOG_POST_BLUEPRINT.md`. Seeds: `docs/seed_tasks.md`. Verified API notes: `docs/api_notes/`. Spec references to `templates/…` mean this repo's `templates/` folder.

## Project rule
"The model can dream a package. Only the registry can make it real."
- LLMs propose.
- TabPFN gives calibrated risk.
- Deterministic code plus registry facts decide.

## Hard rules (never break)
1. NEVER `pip install`, `uv add`, download, or import any package name produced by a model, in code, tests or the shell. Metadata GETs only (PyPI JSON, pypistats). Cache under `data/cache/`. At most 5 requests per second, with the User-Agent from `.env`.
2. NEVER register packages anywhere.
3. Unregistered names that models repeat go ONLY to `private/` (gitignored). They must never appear in `results/`, `web/`, `README.md`, `POST_DRAFT.md`, commit messages or test fixtures. Use anonymised `sha256(name)[:8]` publicly.
4. NEVER invent numbers, results or user reactions. If something hasn't been run, write `{{TODO}}`.
5. Ask before anything that costs money, needs a new key, pushes to GitHub, or publishes anywhere.
6. Verify current library and API docs before use (TabPFN v9 / TabPFN-3.5, Ollama, Ultramarine, Claude Code hooks). Don't trust memory.
7. Features must not use registry data (leakage). Keep `tests/test_leakage.py` green.

## Environment
- Windows 11. Python via `uv`. Run stages with `uv run tasks.py <stage>`. Use `pathlib`; no bash-only scripts.
- Ollama at `http://localhost:11434`. Models are listed in `results/models.json`.
- Env vars: `TABPFN_TOKEN`, `TABPFN_NO_BROWSER=1`, optional `PAPERINSTRUMENTS_API_KEY`.

## Workflow
- Work phase by phase (spec §5, prompts in `docs/`).
- After each phase:
  1. Run `uv run pytest -q`.
  2. Commit.
  3. Print a 5-line status: done, key numbers, next, hours left, scope cuts.
- Record every scope cut in `results/run_config.json`.
- Long-running jobs (generation) run in the background and must be resumable.
- Every headline number goes into `results/summary.json`. The post reads from there.

## Style
- Small modules, type hints, docstrings on public functions, no dead code.
- Tests first for the extractor, guard and hook.
- Plain, honest language in outputs. Report losses (e.g. if TabPFN loses an arm).
