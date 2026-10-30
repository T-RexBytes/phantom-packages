# Phantom Packages

> The model can dream a package. Only the registry can make it real.

Local open-weight coding models sometimes tell you to `pip install` packages that
don't exist. Attackers can register those names ("slopsquatting"). This repo:

1. **Measures** how often local models summon these ghost packages.
2. **Trains TabPFN** to flag a ghost name offline, from generation-time features only.
3. **Ships a guard**: `phantom check`, a Claude Code PreToolUse hook and a GitHub Action.

Work in progress for the DEV Hacktoberfest 2026 Open-Source AI Challenge (Week 4).
Headline numbers live in [`results/summary.json`](results/summary.json). For example, 213 of the 1,410
install suggestions from three local models (15.1%) named packages that don't exist on PyPI.

## Install

```powershell
git clone https://github.com/T-RexBytes/phantom-packages
cd phantom-packages
uv sync                  # the guard: PyPI checks, CLI, hook
uv sync --extra model    # + TabPFN, for P(ghost) and --offline
```

- The `model` extra installs `tabpfn` (and PyTorch).
- The first prediction downloads the TabPFN-3.5 weights. That needs a Prior Labs API key in
  `.env` (`TABPFN_TOKEN=...`, `TABPFN_NO_BROWSER=1`) and the licence accepted once at
  https://ux.priorlabs.ai.
- After that, TabPFN runs fully offline.

## CLI usage

```powershell
uv run phantom check requests                         # names...
uv run phantom check requirements.txt pyproject.toml  # ...or dependency files
uv run phantom check requirements.txt --json          # machine-readable
uv run phantom why requests                           # one detailed card
uv run phantom scan-cmd "pip install -U requests"     # what the hook runs
uv run phantom stats                                  # headline study numbers
```

`check` prints one row per name: Name · Verdict · P(ghost) · Band · Reasons.

| Verdict | Meaning |
|---|---|
| `OK` | Exists on PyPI and matches no squat rule (or is in the standard library). |
| `SUSPECT` | Exists, but looks like a possible slopsquat: few releases, young, rarely downloaded, or ghost-like and new. |
| `GHOST` | PyPI returns 404: nobody ever published it. |
| `GHOST_LIKELY` / `OK_UNVERIFIED` | Offline estimate only: PyPI was not asked (or could not be reached). |

Exit codes for `check` and `scan-cmd`: `0` all OK · `1` a SUSPECT · `2` a GHOST (or GHOST_LIKELY) · `3` usage error.

Every reason is a short string written by deterministic code from registry facts. No LLM writes verdicts or
reasons. The TabPFN probability is advice; the registry decides.

## Claude Code hook setup

[`hooks/claude_pretooluse_guard.py`](hooks/claude_pretooluse_guard.py) is a PreToolUse hook. Before the
Bash or PowerShell tool runs, the hook does the following:
- **Finds installs:** it pulls the `pip install` / `uv add` / `uv pip install` names out of the command.
- **Blocks GHOST and SUSPECT names** with exit code 2. Claude sees the reason as stderr, for example:
  `phantom-packages: blocked `pip install zzq-ghostlib`: zzq-ghostlib does not exist on PyPI (P(ghost)=…). Did you mean `…`? Never install unverified names.`
- **Lets everything else through.** Commands without installs return immediately and never load the model.

Add this to `<your project>/.claude/settings.json` (also in
[`templates/settings.hook.json`](templates/settings.hook.json)), with the absolute path of your clone:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --project PATH/TO/phantom-packages python PATH/TO/phantom-packages/hooks/claude_pretooluse_guard.py",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

| Environment variable | Effect |
|---|---|
| *(default)* | Block GHOST and SUSPECT. If PyPI can't be reached or the guard fails, allow the call and print a warning (fail-open). |
| `PHANTOM_FAIL_CLOSED=1` | Also block when PyPI can't be reached, the guard fails, or the hook input is malformed. |
| `PHANTOM_OFFLINE=1` | Never ask PyPI; block names the TabPFN estimate calls ghost-likely. |
| `PHANTOM_HOOK_PROB=1` | Also print the model's P(ghost) in every block message (adds about 10–15 s on a CPU laptop). |

**Speed.** By default the hook asks PyPI first and loads the TabPFN model only when the model can
change the outcome: offline (`PHANTOM_OFFLINE=1`), when PyPI can't be reached, or for an existing
package that is young. A name PyPI returns 404 for is blocked in about 2 seconds without loading torch.

The hook contract (stdin JSON, `tool_input.command`, exit 2 = block) was checked against the
[Claude Code hooks reference](https://code.claude.com/docs/en/hooks) on 2026-10-30.

## GitHub Action usage

[`.github/workflows/phantom-guard.yml`](.github/workflows/phantom-guard.yml) runs on pull requests that
change a `requirements*.txt` or `pyproject.toml` file:
- **GHOST:** the job fails.
- **SUSPECT:** the job passes with a warning annotation.

It runs `phantom check` on the changed files only and never installs them.

To use it in another repository, copy the workflow and check this repo out next to yours:

```yaml
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
      - uses: actions/checkout@v7
        with:
          repository: T-RexBytes/phantom-packages
          path: .phantom
      - uses: astral-sh/setup-uv@v10.3.3
      - run: uv sync --locked --project .phantom
      - run: uv run --project .phantom phantom check requirements.txt
```

CI installs only the base dependencies, so the verdicts there come from PyPI facts. P(ghost) needs
the `model` extra.

## Offline mode

```powershell
uv run phantom check requirements.txt --offline
```

- **How it decides:** with `--offline` (or `PHANTOM_OFFLINE=1` for the hook), PyPI is never asked. Each
  name gets a TabPFN probability from features of the name alone: distance to the nearest of the top
  15,000 PyPI names, affixes, a character-level language model and similar. It is flagged when
  P(ghost) ≥ τ.
- **The threshold:** τ comes from out-of-fold predictions in the study (`results/model/guard_threshold.json`).
  At τ = 0.506 it caught 90.2% of ghosts, and 75.0% of the names it flagged were ghosts. **Expect about
  one false alarm in four flags.**
- **Online is still the source of truth.** The offline estimate is for planes, firewalls and private
  code, when you can't ask the registry. With the network on, the registry decides.

## Ethics

- **Nothing a model suggests is ever installed or downloaded,** in the study or in the guard. The
  only network calls are PyPI JSON and pypistats metadata `GET`s, cached and rate-limited
  (≤5 req/s).
- **The hook sends nothing anywhere else:** only the package names it finds, to PyPI's JSON API.
- **No packages are registered.**
- **Repeated ghost names are never published.** Unregistered names that models repeat stay in a
  private, gitignored folder; public outputs use a keyed hash (HMAC-SHA256 with the secret
  `PHANTOM_ANON_KEY`), so a guessed name cannot be checked against them. A CI test fails if a public file
  contains one of those names.
- **A SUSPECT verdict is not an accusation.** It means "young, rarely downloaded and suggested by a
  model": check by hand.

## Reproduce the study

```powershell
copy .env.example .env   # fill TABPFN_TOKEN and PHANTOM_ANON_KEY (any long random string); Ultramarine key optional
uv run tasks.py env      # check Ollama, models, keys
uv run tasks.py test     # pytest
uv run tasks.py <stage>  # prompts | summon | extract | features | eval | guard-model | squat | replay | ghost-types | error-analysis | figures | halloween | site | all
uv run tasks.py all --fast   # every stage from the cache, no model calls (about 6 minutes on a laptop CPU)
```

`--fast` needs the gitignored `data/` cache from a real run (generations, PyPI lookups, self-check answers).
Without it, the committed `results/` are the record.

## The Ghost Registry (local site)

`web/` is a static page that reads `web/data.json`; it is not hosted. To view it:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory web   # then open http://127.0.0.1:8000
```

Spec: [docs/03_BUILD_SPEC.md](docs/03_BUILD_SPEC.md).
