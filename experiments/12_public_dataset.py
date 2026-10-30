"""Stage 12: a public, name-free copy of the evaluation dataset -> results/public/.

Outsiders can rerun the detector evaluation (experiments/12b_eval_public.py) without the
gitignored data/ cache. Each row is one (model, prompt_id, name) of data/dataset.parquet with
the name replaced by ``name_id`` = ``phantom.privacy.anon_id(name)`` (keyed HMAC, 8 hex): stable
enough to recompute the per-fold ``global_freq`` and the unseen-name split, but it can't be
reversed or confirmed without the secret key. No string column can reveal a name, and no
registry column is included (CLAUDE.md rules 3 and 7).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from phantom import config, features, privacy

OUT_DIR = config.RESULTS / "public"
DATASET_CSV = OUT_DIR / "dataset.csv"
README = OUT_DIR / "README.md"
# FEATURES minus global_freq (recomputed per fold from name_id and n_gens); the "model" feature
# is stored as model_code (its integer code), next to the model tag.
FEATURE_COLUMNS = [f for f in features.FEATURES if f not in ("global_freq", "model")] + ["model_code"]
ID_COLUMNS = ["name_id", "model", "prompt_id", "fold", "y_ghost", "n_gens"]
COLUMNS = ID_COLUMNS + FEATURE_COLUMNS

README_TEXT = """# Public evaluation dataset (no package names)

`dataset.csv` is the modelling table of the study (one row per model × prompt × suggested
name, {n_rows} rows, {n_ghost} ghosts) with every package name replaced by a keyed id.
`uv run python experiments/12b_eval_public.py` reruns split A (by prompt) and split C
(unseen names) for TabPFN-3.5 and gradient boosting from this file alone.

| column | meaning |
|---|---|
| `name_id` | `HMAC-SHA256(PHANTOM_ANON_KEY, normalized name)[:8]`. Same name, same id; without the secret key a guessed name cannot be checked against it. |
| `model` | the local model that suggested the name (Ollama tag) |
| `prompt_id` | the coding task (`data/prompts.jsonl` ids, see `results/prompt_counts.csv`) |
| `fold` | split-A fold (GroupKFold by prompt, `results/folds.json`) |
| `y_ghost` | 1 if PyPI returned 404 at labelling time (the label; never a feature) |
| `n_gens` | generations of this (model, prompt) that mentioned the name; needed for `global_freq` |
| `len`, `n_parts`, `has_digit` | length, number of `-` parts, contains a digit |
| `in_top15k`, `nn_lev`, `nn_jw`, `nn_rank_log`, `affix_of_real` | relation to the top-15k PyPI names: member, edit distance and Jaro-Winkler to the nearest, log10 rank of the nearest, real name ± a common affix |
| `char_lp`, `char_lp_z` | character-trigram language model score (mean log-prob, z-score) |
| `consistency` | share of this model's samples for the prompt that suggested the name |
| `cross_model` | how many models suggested the name for the same prompt |
| `position` | first position of the name in the reply's install line |
| `source` | integer code: where the name was extracted from (`results/model/categories.json`) |
| `niche_level` | 1 everyday, 2 specialised, 3 obscure task |
| `domain` | integer code of the task domain (`results/model/categories.json`) |
| `name_logprob` | the model's mean token log-prob while writing the name |
| `model_code` | integer code of `model` (the `model` feature) |

`global_freq` is not stored: it must be computed per fold from training rows only
(`phantom.features.with_global_freq`, grouping by `name_id`), so test-fold names never inflate it.
Split C (unseen names) also needs `name_id`: a test row counts as unseen if its id never appears
in the training fold. No column comes from registry data (CLAUDE.md rule 7).
"""


def public_frame(ds: pd.DataFrame, folds: dict[str, int], key: bytes | None = None) -> pd.DataFrame:
    """The name-free public table in ``COLUMNS`` order (dataset row order kept)."""
    out = ds.assign(name_id=[privacy.anon_id(n, key) for n in ds["name"]],
                    fold=ds["prompt_id"].map(folds), y_ghost=ds["y_ghost"].astype(int))
    return out[COLUMNS].reset_index(drop=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    ds = pd.read_parquet(config.DATASET_PATH)
    folds = json.loads((config.RESULTS / "folds.json").read_text(encoding="utf-8"))["folds"]
    pub = public_frame(ds, folds)
    assert not set(pub.columns) & config.REGISTRY_COLUMNS - {"y_ghost"}  # label only, no registry facts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pub.to_csv(DATASET_CSV, index=False)  # pandas writes floats with round-trip precision
    README.write_text(README_TEXT.format(n_rows=len(pub), n_ghost=int(pub["y_ghost"].sum())), encoding="utf-8")
    print(f"wrote {DATASET_CSV.relative_to(config.ROOT)}: {len(pub)} rows x {len(pub.columns)} columns, "
          f"{int(pub['y_ghost'].sum())} ghosts, {pub['name_id'].nunique()} name ids; "
          f"{README.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
