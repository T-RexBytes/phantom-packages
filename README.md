# Phantom Packages

> The model can dream a package. Only the registry can make it real.

Local open-weight coding models sometimes tell you to `pip install` packages that
don't exist. Attackers can register those names ("slopsquatting"). This repo:

1. **Measures** how often local models summon these ghost packages.
2. **Trains TabPFN** to flag a ghost name offline, from generation-time features only.
3. **Ships a guard**: `phantom check`, a Claude Code PreToolUse hook and a GitHub Action.

Work in progress for the DEV Hacktoberfest 2026 Open-Source AI Challenge (Week 4).

## Ethics

- Nothing a model suggests is ever installed or downloaded. Only PyPI/pypistats
  metadata `GET`s, cached and rate-limited (≤5 req/s).
- No packages are registered.
- Unregistered names that models repeat are never published; public outputs use
  `sha256(name)[:8]`.

## Quick start

```powershell
uv sync
copy .env.example .env   # fill TABPFN_TOKEN; Ultramarine key optional
uv run tasks.py env      # check Ollama, models, keys
uv run tasks.py test     # pytest
uv run tasks.py <stage>  # prompts | summon | extract | features | eval | squat | figures | site | all
```

Spec: [docs/03_BUILD_SPEC.md](docs/03_BUILD_SPEC.md).
