# 04: Claude Code Prompts

## How to use

1. Create the project folder:
   ```powershell
   mkdir C:\Users\Soumya\Desktop\DEV\phantom-packages
   cd C:\Users\Soumya\Desktop\DEV\phantom-packages
   ```
2. Copy the kit into the project:
   - `..\phantom-packages-kit\templates\CLAUDE.md` → `.\CLAUDE.md`
   - `..\phantom-packages-kit\03_BUILD_SPEC.md` → `.\docs\BUILD_SPEC.md`
   - `..\phantom-packages-kit\templates\seed_tasks.md` → `.\docs\seed_tasks.md`
   - `..\phantom-packages-kit\05_BLOG_POST_BLUEPRINT.md` → `.\docs\BLOG_BLUEPRINT.md`
3. Start `claude` in that folder and paste **Prompt 0 (Master)**.
4. After each phase finishes, paste the next phase prompt, or say "continue to Phase N".
5. If something breaks, use the **recovery prompts** at the bottom.

> Tip: plan mode (Shift+Tab) for Phase 0 lets you review the plan before any files are written.

---

## Prompt 0: Master (paste first)

```text
You are my build partner for "Phantom Packages", an entry for the DEV Hacktoberfest 2026 Open-Source AI Challenge Week 4: "Ghost in the Machine" (tag #hf26challenge).

HARD DEADLINE: publish by Oct 30 2026 11:59 PM PDT (challenge API ends_at = 2026-10-31T06:59Z). Internal target: post published by Oct 30 8 PM PDT.
Judging order: Writing Quality (heaviest) > Relevance to theme > Creativity > Technical Execution > Partner tech.

READ FIRST, fully, before writing any code:
- CLAUDE.md (project rules, always obey)
- docs/BUILD_SPEC.md (the authoritative spec: architecture, schemas, features, eval protocol, guard, hook, site, tests, acceptance criteria)
- docs/seed_tasks.md (prompt seeds)
- docs/BLOG_BLUEPRINT.md (what the post needs from the build: numbers, figures, demo moments)

THE IDEA: Local open-weight coding models tell people to `pip install` packages that don't exist (ghosts). Attackers register those names (slopsquatting). We (1) measure how often 3 local open models summon ghosts, (2) train TabPFN-3.5 to recognise a ghost name OFFLINE from generation-time features, benchmarked against rules / logistic regression / gradient boosting / LLM self-check with GroupKFold-by-prompt and leave-one-model-out, and (3) ship a guard: `phantom check` CLI (+ --offline), a Claude Code PreToolUse hook that blocks ghost installs, and a GitHub Action. Plus a static "Ghost Registry" site.
Rule: "The model can dream a package. Only the registry can make it real." Models propose, TabPFN gives calibrated risk, deterministic code + registry facts decide.

NON-NEGOTIABLE:
1. Never pip install / uv add / download / import any model-produced package name. Metadata GETs only (pypi.org/pypi/<name>/json, pypistats). Cache everything; ≤5 req/s; descriptive User-Agent.
2. Never register packages. Never publish unregistered, repeatedly hallucinated names; those go to private/ (gitignored).
3. Never invent numbers or reactions. Unrun = {{TODO}}.
4. Ask me before anything that costs money, needs a key, pushes to GitHub, or publishes anything.
5. Windows 11 host: pathlib, uv, `uv run tasks.py <stage>`; no bash-only scripts.
6. Verify current APIs before using them (TabPFN v9 / TabPFN-3.5 docs at docs.priorlabs.ai; Ollama /api/chat; Ultramarine docs.paperinstruments.com; Claude Code hooks docs). Versions changed in 2026; don't trust memory.

PHASES (spec section in brackets). After EACH phase: run pytest, git commit with a clear message, then print a 5-line status: what's done, key numbers, what's next, hours left to deadline, any scope cut.
 Phase 0  Scaffold repo, pyproject, tasks.py, config, cached http client, .gitignore, CI workflow, tests skeleton. Check env: ollama reachable + list models, TABPFN_TOKEN set, optional PAPERINSTRUMENTS_API_KEY. [§2–3]
 Phase 1  Prompts: ~200 tasks from seeds; niche mix 25/35/40. [§4.1, §5.1]
 Phase 2  Summon: resumable generation, ETA, budget rule, models.json; Ultramarine arm if key present (ask first). [§5.2]
 Phase 3  Extract + label with ≥25 extractor tests; subtypes (module_name_confusion, stdlib_confusion). [§5.3, §6]
 Phase 4  Features (no registry columns; leakage test). [§7]
 Phase 5  Eval: 6 arms × splits A/B/C, bootstrap CIs, calibration, threshold, final model + 5 fold models saved, summary.json. [§8]
 Phase 6  Squat check + private disclosure draft. [§5.6]
 Phase 7  Guard CLI, hook, Action, tests. Then help me record the hook demo in a real Claude Code session. [§9]
 Phase 8  Figures + Ghost Registry site + site data (public-safe). [§5.7, §5.8, §10]
 Phase 9  POST_DRAFT.md filled from results/summary.json following docs/BLOG_BLUEPRINT.md exactly. [Blueprint]

Time budget: Phases 0–3 today; 4–7 tomorrow morning; 8–9 tomorrow afternoon; publish by Oct 30 8 PM PDT. If a phase overruns, propose a scope cut (fewer prompts/samples/models), record it in results/run_config.json, and continue. Never fake.

Start with Phase 0. First, summarize back to me in ≤10 bullets what you understood from the spec and list anything ambiguous; then proceed.
```

---

## Phase prompts (paste when moving on, or if context was reset)

### Phase 1: Prompts
```text
Phase 1. Build data/prompts.jsonl from docs/seed_tasks.md per BUILD_SPEC §4.1 and §5.1.
- Target ~200 prompts across the 15 domains; niche_level mix ≈25% L1 / 35% L2 / 40% L3.
- You may expand seeds into variants yourself (deterministically, in code or as a committed list) but every prompt must be a realistic thing a developer would ask.
- Use the exact template and system prompt from §5.1.
- Print a table: domain × niche_level counts. Commit.
```

### Phase 2: Summon
```text
Phase 2. Implement experiments/02_summon.py per BUILD_SPEC §5.2.
- First call GET /api/tags and write results/models.json (tag, digest, size, quantization). Confirm with me which 3 models to use (gemma4:e4b + two different-family coder models).
- Run a 5-prompt smoke test per model; show me 1 raw output each and the measured latency; compute ETA for the full run (3 models × N prompts × 2 samples; sample 3 optional).
- If ETA > 4h, propose a cut and record it in results/run_config.json.
- Order loops samples → models → prompts so partial runs are balanced. Resumable, fsync after each line.
- Run it in the background and keep working on Phase 3 code while it runs (use cached partial data for tests).
- Ultramarine arm: ask me before spending; then sample 1 only, record usage + cost with rates from GET /v1/pricing.
```

### Phase 3: Extract and label
```text
Phase 3. Implement phantom/extract.py, phantom/registry.py and experiments/03_extract_label.py per BUILD_SPEC §5.3 and §6.
- Write the ≥25 extractor tests FIRST (TDD), including every case listed in §11.
- IMPORT_TO_DIST with ≥40 entries; stdlib via sys.stdlib_module_names.
- Registry lookups: cached, rate-limited, retry/backoff; metadata only.
- Output data/mentions.parquet and data/registry.parquet; print: total mentions, unique names, ghosts (404), ghost rate overall + per model, subtype counts, unmapped imports count, dropped tokens count.
- Show me the 15 most frequent ghost names in the terminal ONLY (not in any committed file).
```

### Phase 4: Features
```text
Phase 4. Implement phantom/topnames.py, phantom/charlm.py, phantom/features.py and experiments/04_features.py per BUILD_SPEC §7.
- Load the top-15k list from the URL in the spec (cache it).
- global_freq must be computable per-fold (provide a function that takes the training mask).
- Add the leakage test (no registry columns in feature_list.json).
- Output data/dataset.parquet + results/model/feature_list.json. Print class balance and feature summary.
```

### Phase 5: Eval (the heart of the post)
```text
Phase 5. Implement experiments/05_eval.py per BUILD_SPEC §8 exactly.
- Save fold assignment to results/folds.json; same folds for every arm.
- Arms: rule_top15k, rule_editdist, logreg, hgb, tabpfn, llm_selfcheck (cache answers).
- Splits: A GroupKFold(5) by prompt; B leave-one-model-out; C unseen-names subset of A.
- Metrics + bootstrap CIs by prompt group; calibration (10 bins); threshold τ for recall ≥ 0.90; time each arm.
- Check TabPFN docs for the current constructor/save-load; subsample to ≤5k train rows on CPU or set ignore_pretraining_limits=True and time it.
- Save final model + 5 fold models, threshold.json, eval.csv, loo.csv, summary.json (every headline number).
- Then write me a plain-English interpretation (≤12 bullets) including where TabPFN LOSES. No spin.
```

### Phase 6: Squat check
```text
Phase 6. Implement phantom/squat.py and experiments/06_squat_check.py per BUILD_SPEC §5.6.
- Re-check existence now for every ghost; record any that got registered during the study window (story beat!).
- Squat rules exactly as specified; never download artifacts.
- results/squat_public.csv (aggregates + registered names only), private/squat_full.csv, private/disclosure_draft.md.
- Add a CI test that fails if any name from private/unregistered_repeated.csv appears in web/ or results/ public files.
```

### Phase 7: Guard, hook, Action
```text
Phase 7. Implement phantom/guard.py, phantom/cli.py (check | why | scan-cmd | stats, --offline, --json, exit codes), hooks/claude_pretooluse_guard.py and .github/workflows/phantom-guard.yml per BUILD_SPEC §9.
- Before writing the hook, check the current Claude Code hooks documentation for the PreToolUse stdin schema and blocking exit code; cite what you verified in a code comment.
- Tests for guard, cli exit codes, hook stdin→exit code.
- Add README sections: install, CLI usage, hook setup (settings.json snippet), Action usage, offline mode, ethics.
- Then give me exact steps to record the demo: a fresh Claude Code session in a scratch folder with the hook enabled, a prompt that will make the agent try to install one of OUR real ghosts, and what to capture.
```

### Phase 8: Figures and site
```text
Phase 8. Implement experiments/07_figures.py, experiments/08_build_site_data.py and web/index.html per BUILD_SPEC §5.7, §5.8, §10.
- Dark Halloween theme with marigold accents, readable charts, contrast ≥ 4.5:1, mobile 360px, reduced-motion respected, alt text.
- Graveyard: tombstone cards; registered ghosts by name, unregistered ones anonymised (sha256[:8]).
- Everything public-safe; run the private-leak test.
- Prepare GitHub Pages deploy (and CNAME if I give you a .tech domain). Ask before pushing.
```

### Phase 9: Post draft
```text
Phase 9. Write POST_DRAFT.md following docs/BLOG_BLUEPRINT.md section by section.
- Every number must come from results/summary.json or a results CSV; link each table to the file that produced it.
- Keep ALL template headings in order. Include "What I Measured" and "What Went Wrong" (≥3 real failures from this build, with numbers).
- Fill the title bank with 5 options using our real numbers.
- Leave {{TODO}} only for: real-person reaction, demo GIF URL, live site URL, DevRelay session embed.
- Then run a self-review against docs/BLOG_BLUEPRINT.md "Judge's checklist" and list any gaps.
```

---

## Recovery and utility prompts

**Context reset / new session**
```text
Read CLAUDE.md, docs/BUILD_SPEC.md and results/run_config.json, then `git log --oneline -20`. Tell me which phase we're in, what's done, what's next, and hours left to Oct 30 11:59 PM PDT. Then continue.
```

**Generation is too slow**
```text
Generation ETA is too long. Propose 3 cut options with new ETAs (fewer prompts, fewer samples, smaller model tag), keeping balance across models/niche levels. Record the chosen cut in results/run_config.json and resume without losing completed rows.
```

**Too few ghosts**
```text
Class balance is too low for a stable classifier. Options: (a) add an extra sample at temperature 1.0 as a separate arm, (b) add more niche_level-3 prompts, (c) evaluate additionally on an external list (only as a separate eval, check license). Recommend one, explain the effect on the honesty of the results, and do it after I confirm.
```

**TabPFN install/auth trouble**
```text
TabPFN failed: <paste error>. Check docs.priorlabs.ai for current install/auth (TABPFN_TOKEN, TABPFN_NO_BROWSER, license acceptance URL). If local weights are blocked, switch the tabpfn arm to tabpfn-client with the same folds, note it in run_config.json, and continue.
```

**Numbers look too good**
```text
These metrics look suspiciously high. Audit for leakage: features derived from registry, global_freq computed on full data, same names across folds, in_top15k dominating. Report findings and re-run with fixes; show before/after.
```

**Final pre-publish review**
```text
Act as a strict DEV challenge judge. Review POST_DRAFT.md against: writing quality, theme relevance, creativity, technical execution, partner use. Score each 1–10 with one-line reasons, list the 5 highest-impact edits, and check every number against results/summary.json. Do not rewrite the whole post.
```
