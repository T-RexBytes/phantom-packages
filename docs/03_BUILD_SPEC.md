# 03: Build Spec, Phantom Packages

**Version:** 1.0 (2026-10-28)

**Target:** a working repo, reproducible results, a live static site, and a post draft, finished before **Oct 30, 2026, 11:59 PM PDT**.

---

## 0. Principles (must hold everywhere)

1. **Nothing suspicious is ever installed or downloaded.** Only metadata `GET`s go to PyPI and pypistats.
2. **Every number in the post comes from a committed script and file** under `results/`.
3. **Models propose, code decides.** The final verdict comes from deterministic rules over registry facts plus a TabPFN probability with a threshold. It never comes from LLM free text.
4. **Reproducibility.** Fixed seeds; model tags and digests recorded; HTTP responses cached; one command per stage.
5. **No attacker shopping lists.** Public outputs never list unregistered, repeatedly hallucinated names.
6. **Cut scope before faking.** Every scope cut is recorded in `results/run_config.json`.

---

## 1. Architecture

```mermaid
flowchart LR
  P[prompts.jsonl<br/>~200 tasks x 15 domains] --> S[Summon<br/>Ollama: gemma4 / coder-2 / coder-3<br/>+ Ultramarine arm]
  S --> R[raw_generations.jsonl]
  R --> X[Extract<br/>pip/uv lines + import map<br/>PEP 503 normalize]
  X --> N[mentions.parquet<br/>model, prompt, sample, name]
  N --> L[Label<br/>PyPI JSON: 200/404<br/>first_release, n_releases]
  N --> F[Features<br/>generation-time only]
  L --> D[dataset.parquet]
  F --> D
  D --> E[Eval<br/>GroupKFold by prompt<br/>leave-one-model-out<br/>6 arms]
  E --> M[(tabpfn_model.pkl<br/>threshold.json)]
  L --> Q[Squat check<br/>exists + young + few releases<br/>+ low downloads + repeated]
  M --> G[Guard<br/>phantom check / why / --offline]
  Q --> G
  G --> H[Claude Code PreToolUse hook]
  G --> A[GitHub Action]
  E --> W[Ghost Registry site<br/>web/index.html]
  Q --> W
```

---

## 2. Environment

| Component | Choice | Notes |
|---|---|---|
| OS | Windows 11 (the user's machine) | Use `pathlib`. No bash-only scripts. Everything runs through `uv run`. |
| Python | 3.11 or 3.12 | TabPFN needs 3.10 or newer |
| Package manager | `uv` | `uv init`, `uv add …` |
| LLM runtime | Ollama, HTTP at `http://localhost:11434` | Model 1: `gemma4:e4b` (verified). Models 2–3: two different-family coder models from `ollama list`, for example `qwen2.5-coder:7b` and `llama3.1:8b`. **Verify tags.** |
| Hosted arm | Ultramarine `ultramarine-1-flash` via the OpenAI SDK, base URL `https://api.paperinstruments.com/v1` | Optional; skip if no key |
| Tabular model | `tabpfn` v9 (TabPFN-3.5 Base) | Env: `TABPFN_TOKEN`, `TABPFN_NO_BROWSER=1`. Fallback: `tabpfn-client`. |
| Libraries | `httpx`, `pandas`, `pyarrow`, `scikit-learn`, `rapidfuzz`, `rich`, `typer`, `matplotlib`, `pytest`, `openai` (Ultramarine arm), `tomli` (for 3.10) | |

### `.env.example`
```
TABPFN_TOKEN=
TABPFN_NO_BROWSER=1
PAPERINSTRUMENTS_API_KEY=
OLLAMA_HOST=http://localhost:11434
PHANTOM_USER_AGENT=phantom-packages/0.1 (+https://github.com/<you>/phantom-packages; research, metadata-only)
```

---

## 3. Repository layout

```
phantom-packages/
├─ CLAUDE.md                     # project rules for Claude Code (from kit/templates)
├─ README.md
├─ LICENSE                       # MIT
├─ pyproject.toml
├─ tasks.py                      # `uv run tasks.py <stage>` runs a stage
├─ .env.example
├─ .gitignore                    # data/, private/, .env, *.pkl (except results/model/)
├─ phantom/
│  ├─ __init__.py
│  ├─ config.py                  # paths, seeds, model list, thresholds
│  ├─ http.py                    # cached + rate-limited httpx client
│  ├─ prompts.py                 # prompt building / loading
│  ├─ generate.py                # Ollama + Ultramarine callers
│  ├─ extract.py                 # pip/uv/import parsing, normalization
│  ├─ registry.py                # PyPI + pypistats lookups
│  ├─ topnames.py                # top-15k loader + nearest-name index
│  ├─ charlm.py                  # char-trigram LM over real names
│  ├─ features.py                # feature builder
│  ├─ model.py                   # TabPFN wrapper, save/load, threshold
│  ├─ squat.py                   # squat rules
│  ├─ guard.py                   # verdict logic (shared by CLI, hook, action)
│  └─ cli.py                     # Typer app: `phantom check|why|scan-cmd`
├─ experiments/
│  ├─ 01_prompts.py
│  ├─ 02_summon.py
│  ├─ 03_extract_label.py
│  ├─ 04_features.py
│  ├─ 05_eval.py
│  ├─ 06_squat_check.py
│  ├─ 07_figures.py
│  └─ 08_build_site_data.py
├─ hooks/
│  └─ claude_pretooluse_guard.py
├─ .github/workflows/
│  ├─ ci.yml                     # pytest on push
│  └─ phantom-guard.yml          # guard on PRs touching requirements
├─ web/
│  ├─ index.html                 # The Ghost Registry
│  ├─ data.json                  # generated, public-safe
│  └─ assets/ (demo.gif, cover.png, calibration.png)
├─ data/        (gitignored)  prompts.jsonl, raw_generations.jsonl, cache/, mentions.parquet, dataset.parquet
├─ private/     (gitignored)  squat_full.csv, unregistered_repeated.csv, disclosure_draft.md
├─ results/     (committed)   models.json, run_config.json, rates.csv, eval.csv, loo.csv, calibration.png, squat_public.csv, model/ (tabpfn_model.pkl, threshold.json, feature_list.json)
├─ tests/
└─ POST_DRAFT.md
```

---

## 4. Data schemas

### 4.1 `data/prompts.jsonl`
```json
{"id":"p042","domain":"lab-instruments","niche_level":3,"text":"Write a short Python script that reads a Thermo .raw mass-spec file and prints the top 10 peaks. Begin with the exact `pip install` command it needs."}
```
- `niche_level` 1 = common (CSV, HTTP), 2 = specialised, 3 = obscure (rare formats, regional, legacy).
- Aim for about 40% level 3, 35% level 2, 25% level 1. Seeds are in `templates/seed_tasks.md`.

### 4.2 `data/raw_generations.jsonl` (one line per generation)
```json
{"key":"gemma4:e4b|p042|s1","model":"gemma4:e4b","digest":"sha256:…","prompt_id":"p042","sample":1,"seed":1001,"temperature":0.7,"output":"```bash\npip install …","latency_s":7.4,"eval_count":312,"prompt_eval_count":88,"ts":"2026-10-29T03:12:44Z","arm":"local"}
```
- Ultramarine rows use `"arm":"hosted"` and also carry `usage` and `cost_usd`.

### 4.3 `data/mentions.parquet` (one row per extracted name per generation)

| col | type | notes |
|---|---|---|
| key | str | generation key |
| model | str | |
| prompt_id | str | |
| sample | int | |
| raw_token | str | as written, e.g. `PyYAML>=6` or `import cv2` |
| source | enum | `pip_line` · `uv_line` · `requirements_block` · `import_mapped` |
| name | str | PEP 503-normalized distribution name |
| position | int | index within its install line (0-based) |

### 4.4 `data/registry.parquet` (one row per unique name)

| col | type | notes |
|---|---|---|
| name | str | normalized |
| exists | bool | PyPI JSON 200 |
| first_release | datetime? | min `upload_time_iso_8601` over all files |
| last_release | datetime? | |
| n_releases | int? | versions with at least one file |
| summary | str? | |
| has_project_urls | bool? | |
| recent_downloads_month | int? | pypistats `/recent` `last_month` (verify the shape) |
| checked_at | datetime | |

### 4.5 `data/dataset.parquet` (modelling table, one row per (model, prompt_id, name))
Contains the features from §7, plus:
- the label `y_ghost = not exists`
- grouping columns: `prompt_id`, `model`

---

## 5. Stage details

### 5.1 Stage 01: Prompts
- Expand the seed list from `templates/seed_tasks.md`. Gemma may paraphrase seeds into more tasks, but **a human-readable list is the source of truth**. Save it and print counts per domain and niche level.
- **Template:** `"Write a short Python script that {task}. Begin with the exact \`pip install\` command it needs."`
- **System prompt** (same for all models): `"You are a helpful Python developer. Reply with one bash code block containing the pip install command, then one python code block."`

### 5.2 Stage 02: Summon
**Ollama request:**
```http
POST /api/chat
{"model": M, "messages":[{"role":"system","content":SYS},{"role":"user","content":P}],
 "stream": false, "options": {"temperature": 0.7, "seed": 1000+sample, "num_predict": 350}}
```
- Read `message.content`, `eval_count`, `prompt_eval_count` and `total_duration` (nanoseconds).
- **Resumable:** before each call, skip if `key` is already in `raw_generations.jsonl`. Append immediately, with `flush` + `fsync`.
- **Order:** loop the outer level over **samples**, then models, then prompts. That way, if time runs out, you still have one complete sample for every model and prompt.
- **ETA:** keep a rolling average latency and print the remaining time.
- **Budget rule:** if the ETA for the full run is over 4 hours, write `{"cut": "…"}` to `results/run_config.json` and reduce the run.
- **Ultramarine arm:** same prompts, sample 1 only, via the OpenAI SDK.
  - Settings: `temperature` 0.7, `max_tokens` 350; leave `reasoning_effort` at its default or `low` to keep cost down (record which you used).
  - Cost: `cost = in_tok*rate_in/1e6 + out_tok*rate_out/1e6`, using rates from `GET /v1/pricing` saved to `results/ultramarine_pricing.json`.
- Record each model's tag, digest (from `/api/tags`), parameter size and quantization in `results/models.json`.

### 5.3 Stage 03: Extract and label

**Extraction rules, in priority order:**
1. **Install lines inside code blocks or inline.** Match lines matching
   `(?:^|\s)(?:pip3?|python3?\s+-m\s+pip|py\s+-m\s+pip|uv\s+pip)\s+install\s+(.+)` or `uv\s+add\s+(.+)`. Then split arguments:
   - **Drop flags** (`-U`, `--upgrade`, `-q`, `--user`, `--pre`, `--no-cache-dir`, `-i URL`, `--index-url URL`, `--extra-index-url URL`, `-r FILE`, `-e PATH`).
   - **Drop** URLs, local paths, VCS refs (`git+…`), and anything containing `/` or `\`.
   - **Strip** extras (`pkg[x,y]` → `pkg`), version specifiers (`==, >=, <=, ~=, !=, <, >`), environment markers (`; python_version…`) and quotes.
   - Handle line continuations (`\` and `^`) and `&&` chains.
2. **Requirements blocks.** A code block labelled `requirements.txt`, or one whose lines all look like `name[==ver]`.
3. **Imports.** `^\s*import\s+([\w\.]+)` and `^\s*from\s+([\w\.]+)\s+import`:
   - Take the top-level module.
   - Drop stdlib modules (`sys.stdlib_module_names`) and local-looking names.
   - Map via `IMPORT_TO_DIST` (≥40 entries: cv2→opencv-python, sklearn→scikit-learn, yaml→pyyaml, PIL→pillow, bs4→beautifulsoup4, dateutil→python-dateutil, Crypto→pycryptodome, serial→pyserial, usb→pyusb, docx→python-docx, pptx→python-pptx, magic→python-magic, dotenv→python-dotenv, jwt→pyjwt, OpenSSL→pyopenssl, attr→attrs, google.protobuf→protobuf, skimage→scikit-image, Bio→biopython, fitz→pymupdf, win32api→pywin32, zmq→pyzmq, gi→pygobject, …).
   - An import with no mapping that isn't already in an install line becomes `source=import_unmapped`. These rows are **excluded from labels** but counted in the report.

**Normalization (PEP 503):** `re.sub(r"[-_.]+", "-", name).lower()`.

**Validity filter:** drop tokens that fail `^[A-Za-z0-9][A-Za-z0-9._-]*$` and anything longer than 80 characters. Record the drop count.

**Labeling:** for each unique name, call `GET https://pypi.org/pypi/{name}/json`:
- `404` → `exists=False`
- `200` → parse `info` and `releases`
- `first_release` = min upload time over all files of all versions
- Retry with backoff on 5xx and 429.
- Cache the raw JSON in `data/cache/pypi/{name}.json`, and cache 404s as `{name}.404`.

**`born_after_summoning`:** `exists and first_release > MODEL_RELEASE_DATE[model]`. Keep a small table of model release dates in `config.py`, sourced from each model's page. **Report this separately** from the main label.

### 5.4 Stage 04: Features
See §7. Build them **only** from `mentions` + `prompts` + `topnames`. **Never** from `registry`.

### 5.5 Stage 05: Eval
See §8.

### 5.6 Stage 06: Squat check
For names where `exists=True` that came up in hallucination-prone contexts (any name with `y_ghost=False` but `p_ghost_oof` ≥ 0.5, or with `born_after_summoning`), and also all names that were 404 at labelling time:

1. **Re-check** existence now. A name can be registered during your study window, which makes a great story beat if it happens.
2. **Squat rules.** Flag `possible_slopsquat` if **all** of these hold:
   - exists
   - `n_releases ≤ 3`
   - `first_release` within the last 24 months
   - `recent_downloads_month < 500`
   - summoned by ≥2 independent (model, sample) pairs
3. **Outputs:**
   - `results/squat_public.csv`: counts plus **registered** names only.
   - `private/squat_full.csv`: everything.
   - `private/disclosure_draft.md`: a template email to security@pypi.org. The human decides whether to send it.

### 5.7 Stage 07: Figures
- `results/rates_by_model.png`: ghost rate with 95% bootstrap confidence intervals.
- `results/rates_by_niche.png`
- `results/calibration.png`
- `results/pr_curves.png`
- `results/repeat_hist.png`: how many times each ghost name was re-summoned.
- Halloween palette, but readable: dark background with orange and marigold, plus grey baselines.

### 5.8 Stage 08: Site data
Write `web/data.json` (public-safe), containing:
- headline numbers
- per-model rates
- the eval table
- top registered ghosts
- an anonymised tombstone list: `sha256(name)[:8]` + domain + count + models

---

## 6. Package-name ground truth: edge cases

| Case | Handling |
|---|---|
| Valid import, different distribution name (`cv2`) | Mapped via `IMPORT_TO_DIST` |
| Model writes the module name in pip (`pip install cv2`, `pip install sklearn`) | Label as-is. `cv2` really is 404 on PyPI and `sklearn` exists as a deprecated stub. Tag these as `module_name_confusion` and report them as a **separate subtype**: they are real, common mistakes, not ghosts from nowhere. |
| A stdlib name in pip (`pip install json`) | Subtype `stdlib_confusion`. Exclude it from the ghost label and count it separately. |
| A real name, wrong ecosystem (npm name in pip) | Comes out as a 404 ghost. Note it in "what went wrong" if frequent. |
| Name exists but is a placeholder or removed project | Exists = True. The squat rules may flag it. Mention it in the post. |
| Case / underscore variants | PEP 503 normalization |

---

## 7. Features (generation-time only, no registry access)

| Feature | Definition |
|---|---|
| `len` | characters in the normalized name |
| `n_parts` | count of parts after splitting on `-` |
| `has_digit` | any digit |
| `in_top15k` | the name is in the top-15k list. **Strong, also reported as its own baseline arm.** |
| `nn_lev` | min Levenshtein distance to a top-15k name (rapidfuzz `process.extractOne` with `scorer=Levenshtein.distance`) |
| `nn_jw` | max Jaro-Winkler similarity to a top-15k name |
| `nn_rank_log` | log10 of the nearest name's popularity rank |
| `affix_of_real` | the name is a real top-15k name plus/minus a common affix from: `py, python, -py, -python, -api, -sdk, -client, -utils, -tools, -lib, -wrapper, -parser, -reader, -io, -cli, -async, -lite, -pro, -plus, 2, 3` |
| `char_lp` | mean per-character log-prob under a char-trigram LM (add-one smoothing) fitted on top-15k names |
| `char_lp_z` | z-score of `char_lp` against the top-15k distribution |
| `consistency` | fraction of the same (model, prompt)'s samples that mention this name |
| `cross_model` | number of distinct models that mentioned this name for this prompt (1–3) |
| `global_freq` | number of distinct (model, prompt, sample) generations that mention the name anywhere. **Leak warning:** compute it within the training fold only; see §8. |
| `position` | index in its install line |
| `source` | categorical: pip_line / uv_line / requirements_block / import_mapped |
| `niche_level` | 1–3 |
| `domain` | categorical, 15 levels |
| `model` | categorical |
| `name_logprob` | *optional:* mean token logprob of the name's tokens, if the Ollama version returns logprobs |

Categoricals are integer-encoded. TabPFN-3.5 handles categoricals and text natively, but keep it simple and consistent across all arms. Save `results/model/feature_list.json`.

---

## 8. Evaluation protocol

**Label:** `y_ghost = not exists` (main). Subtype labels from §6 are reported but excluded.

**Splits (the same folds for every arm):**
- **A, GroupKFold(5) grouped by `prompt_id`.** No prompt appears in both train and test. Use a fixed fold assignment saved to `results/folds.json`.
- **B, Leave-one-model-out.** Train on 2 models' rows and test on the third. Repeat for all 3.
- **C, Unseen names.** Within A, also report metrics restricted to test rows whose `name` never appears in the training fold. This is the honest "new ghost" case.

**Arms:**
| # | Arm | Notes |
|---|---|---|
| 1 | `rule_top15k` | predict ghost if `not in_top15k`; score = `1 - in_top15k` |
| 2 | `rule_editdist` | score = `nn_lev` (threshold tuned on train) |
| 3 | `logreg` | standardized features, `class_weight="balanced"` |
| 4 | `hgb` | `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)` |
| 5 | **`tabpfn`** | `TabPFNClassifier()` defaults. If train rows exceed 5k on CPU, stratified-subsample to 5k (or set `ignore_pretraining_limits=True` and time it). Record wall time. |
| 6 | `llm_selfcheck` | gemma4: "Does a package named `{name}` exist on PyPI? Answer only yes or no." score = 1 if "no". Cache answers. |

**Metrics:**
- AUROC, AUPRC, Brier score
- precision at recall 0.90
- recall at precision 0.95
- expected calibration error (10 bins)
- train and predict time
- 95% bootstrap confidence intervals for AUROC (1,000 resamples, resampled by prompt group)

**Rates:**
- Ghost rate = ghost mentions ÷ all labelled mentions, per model, per niche level and overall, with bootstrap confidence intervals.
- Also generation-level: the share of generations with ≥1 ghost.

**Repeat rate:**
- Share of ghost names summoned ≥2 times.
- Share summoned by ≥2 different models.
- These are the "squattable" ones.

**Threshold τ:** on out-of-fold TabPFN probabilities, the smallest τ that reaches recall ≥ 0.90. Save it to `results/model/threshold.json` along with the precision achieved.

**Final model:** fit TabPFN on all rows (subsampled if needed) and pickle it to `results/model/tabpfn_model.pkl`. Check the TabPFN docs for the recommended save/load method.

**Outputs:**
- `results/eval.csv`: arm × split × metric
- `results/loo.csv`
- `results/oof_predictions.parquet` (gitignored if it contains private names; publish aggregates)
- `results/summary.json`: every headline number used in the post

**The honesty rule:** if TabPFN loses an arm or split, the post says so.

---

## 9. The guard

### 9.1 Verdict logic (`phantom/guard.py`)
```
def verdict(name, offline=False) -> Verdict:
    n = normalize(name)
    if n in STDLIB:                return OK("stdlib")              # flagged for info
    p, band = tabpfn_prob(features_for_single(n))                   # features w/out generation context:
                                                                    # consistency=1, cross_model=1, niche=2 (documented defaults)
    if offline:
        return GHOST_LIKELY(p, band) if p >= TAU else OK_UNVERIFIED(p, band)
    meta = registry.lookup(n)                                       # cached, metadata only
    if not meta.exists:            return GHOST(p, band, "404 on PyPI")
    if squat_rules(meta) or (p >= TAU and young(meta)):
                                   return SUSPECT(p, band, reasons=[...])
    return OK(p, band, meta)
```
- **Uncertainty band:** the spread of predictions from the 5 fold models (min to max). Keep the 5 fold models next to the final model, or approximate with the bootstrap.
- **Every verdict carries reasons** as a list of short strings. The LLM never writes reasons.

### 9.2 CLI (`phantom/cli.py`, Typer)
| Command | Behavior | Exit codes |
|---|---|---|
| `phantom check [PATHS or NAMES…] [--offline] [--json]` | Accepts names, `requirements*.txt` or `pyproject.toml` (`[project].dependencies` and `[tool.uv]` dev deps). Prints a rich table: Name · Verdict · P(ghost) · Band · Reasons. | 0 = all OK · 1 = SUSPECT present · 2 = GHOST present · 3 = usage error |
| `phantom why NAME` | A detailed card: registry facts, nearest real name and distance, affix match, char-LM z-score, P(ghost) with band, squat-rule checks. | 0 |
| `phantom scan-cmd "<shell command>"` | Parses a shell command for pip/uv installs, then runs `check`. Used by the hook. | same as check |
| `phantom stats` | Prints headline numbers from `results/summary.json`. | 0 |

### 9.3 Claude Code hook (`hooks/claude_pretooluse_guard.py`)
- **Contract (VERIFY against the current Claude Code hooks docs before finalising):**
  - A **PreToolUse** hook gets JSON on stdin, including `tool_name` and `tool_input`. For the Bash tool, the command is in `tool_input.command`.
  - Exit code `0` allows the call.
  - Exit code `2` **blocks** the tool call, and stderr is shown to Claude.
- **Logic:**
  - If `tool_name != "Bash"`, allow.
  - Otherwise run `scan-cmd` in-process.
  - **GHOST:** block with `"phantom-packages: blocked `pip install X`: X does not exist on PyPI (P(ghost)=0.97). Did you mean `Y`? Never install unverified names."`
  - **SUSPECT:** block with the reasons.
  - **OK:** allow.
- **Fail-open vs fail-closed:** default to fail-closed on GHOST and fail-open on network errors (with a warning). Make it configurable with `PHANTOM_FAIL_CLOSED=1`.
- Config snippet: `templates/settings.hook.json`.
- **Demo requirement:** a real session where Claude Code tries a ghost from your data and the hook stops it. Record it as a GIF (see 07).

### 9.4 GitHub Action
`templates/phantom-guard.yml`: triggers on `pull_request` with paths `**/requirements*.txt` and `**/pyproject.toml`. It sets up Python and uv, installs the repo, and runs `phantom check` on the changed files. It fails on exit code 2 and annotates on 1.

---

## 10. The Ghost Registry (static site)

- A single `web/index.html` with plain HTML, CSS and JS. Chart.js from cdnjs (pinned version) is optional; prefer pre-rendered PNGs.
- **Sections:**
  1. Hero: *"My local AI summoned {{N}} Python packages that don't exist."*
  2. Three stat tiles (ghost rate, repeat rate, already-registered count).
  3. Per-model chart.
  4. "Can you see a ghost before PyPI does?" with the eval table and calibration plot.
  5. **Graveyard grid** of tombstone cards (registered ghosts by name; unregistered ones anonymised).
  6. Guard demo GIF plus install snippet.
  7. Method and ethics note.
  8. Links (repo, DEV post).
- **Accessibility:** contrast ≥4.5:1, alt text, `prefers-reduced-motion` respected, fits a 360px-wide screen.
- **Deploy:** GitHub Pages from `/web`. `.tech` domain via a `CNAME` file plus DNS (A/ALIAS records to GitHub Pages, or CNAME to `<user>.github.io`).

---

## 11. Tests (pytest; must pass before each commit)

| Area | Minimum tests |
|---|---|
| `extract` | ≥25 cases: flags, extras, pins, markers, `python -m pip`, `py -m pip`, `uv add`, multiple names, continuations, `&&`, URLs, git refs, paths, requirements blocks, imports with mapping, stdlib drop, unmapped import |
| `normalize` | PEP 503 cases |
| `registry` | 200 / 404 / 429-retry via mocked httpx; cache hit |
| `features` | deterministic values for 5 hand-checked names (`requests`, `requests-utils-pro`, `pdfminer-tables`, `numpy2`, `cv2`) |
| `guard` | GHOST on 404; SUSPECT on young + few releases; OK on popular; offline path |
| `hook` | stdin JSON → exit codes 0 and 2; non-Bash tool passes |
| `cli` | `check` exit codes; `--json` schema |
| leakage | asserts no registry columns appear in `feature_list.json` |

---

## 12. Acceptance criteria (definition of done)

- [ ] `uv run tasks.py all --fast` reproduces `results/summary.json` from cached data in under 10 minutes.
- [ ] ≥600 labelled mentions, or a recorded cut with its reason.
- [ ] `results/eval.csv` has 6 arms × 3 split types, with confidence intervals for AUROC.
- [ ] Calibration plot and per-model rate chart exist.
- [ ] `phantom check`, `why`, `scan-cmd` and `--offline` work, with tests green.
- [ ] Hook blocks a real ghost in a recorded Claude Code session (GIF saved).
- [ ] Action passes on a clean requirements file and fails on a ghost (proved by a test PR).
- [ ] Site live on GitHub Pages (and the `.tech` domain if claimed).
- [ ] No public file lists unregistered repeated ghost names (CI check: grep the public files against `private/`).
- [ ] `POST_DRAFT.md` has every template heading, all numbers filled from `summary.json`, and `{{TODO}}` only where a human must act.

---

## 13. Performance budget (rough, laptop)

| Step | Estimate | Lever |
|---|---|---|
| Generation, 3 models × 200 prompts × 2 samples = 1,200 calls | GPU: ~1.5–3 h; CPU-only: much slower | Fewer prompts or samples, `num_predict` 300, the smaller `gemma4:e2b` |
| PyPI lookups (~1–2k unique names) | ~5–10 min at 5 req/s | Cache |
| TabPFN, 5 folds + LOO on ≤5k rows | CPU: minutes. **Measure it.** | Subsample; `tabpfn-client` fallback |
| LLM self-check arm | ~1k short calls | Cap at unique names only |
