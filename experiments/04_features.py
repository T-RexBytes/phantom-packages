"""Stage 04: build data/dataset.parquet and results/model/feature_list.json (BUILD_SPEC §7).

Inputs are stage 03's outputs (labelled mentions + the generation grid it analysed),
the prompts, generation logprobs and the top-15k list. Registry data is never read.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from phantom import config, features, generate, hosted, prompts, runconfig, topnames
from phantom.charlm import CharTrigramLM

LABELLED_PATH = config.DATA / "labelled.parquet"
GENERATIONS_PATH = config.DATA / "generations.parquet"
FEATURE_LIST_PATH = config.MODEL_DIR / "feature_list.json"
CATEGORIES_PATH = config.MODEL_DIR / "categories.json"
console = Console(highlight=False)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    labelled, gens = hosted.local_only(pd.read_parquet(LABELLED_PATH), pd.read_parquet(GENERATIONS_PATH))
    wanted = set(gens["key"])
    records = {r["key"]: r for r in generate.read_records(config.RAW_GENERATIONS_PATH) if r["key"] in wanted}

    doc = topnames.fetch_top()
    index = topnames.TopIndex.from_document(doc)
    lm = CharTrigramLM(index.names)
    cats = features.categories(sorted(gens["model"].unique()))

    ds = features.build(labelled, gens, prompts.load_prompts(), records, index, lm, cats)
    ds.to_parquet(config.DATASET_PATH, index=False)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_LIST_PATH.write_text(json.dumps(features.FEATURES, indent=2) + "\n", encoding="utf-8")
    CATEGORIES_PATH.write_text(json.dumps(
        {**cats, "categorical": features.CATEGORICAL, "per_fold": features.PER_FOLD}, indent=2) + "\n",
        encoding="utf-8")

    n_ghost = int(ds["y_ghost"].sum())
    console.print(f"dataset: {len(ds)} rows (model, prompt, name) from {len(gens)} generations | "
                  f"ghost {n_ghost} ({n_ghost / len(ds):.1%}) / real {len(ds) - n_ghost} | "
                  f"{ds['prompt_id'].nunique()} prompts, {ds['name'].nunique()} names")
    console.print("rows per model: " + ", ".join(f"{m} {n} ({g} ghost)" for m, n, g in
                  ds.groupby("model")["y_ghost"].agg(["size", "sum"]).itertuples()))

    # Display only: global_freq over all rows (the eval computes it per fold).
    show = features.feature_matrix(features.with_global_freq(ds, pd.Series(True, index=ds.index)))
    show["y_ghost"] = ds["y_ghost"].values
    table = Table(title="Feature summary: mean by class (global_freq on all rows, display only)")
    for col in ("feature", "real", "ghost", "missing"):
        table.add_column(col, justify="left" if col == "feature" else "right")
    means = show.groupby("y_ghost")[features.FEATURES].mean()
    for f in features.FEATURES:
        table.add_row(f, f"{means.loc[False, f]:.3f}", f"{means.loc[True, f]:.3f}",
                      f"{show[f].isna().mean():.0%}")
    console.print(table)

    runconfig.update("features", {
        "built_at": runconfig.utc_now(),
        "n_rows": len(ds), "n_ghost": n_ghost, "n_generations": len(gens),
        "top_list": {"url": config.TOP_PACKAGES_URL, "last_update": doc.get("last_update"),
                     "n_names": len(index.names)},
        "name_logprob_missing_share": float(ds["name_logprob"].isna().mean()),
    })
    console.print(f"Wrote {config.DATASET_PATH.relative_to(config.ROOT)}, "
                  f"{FEATURE_LIST_PATH.relative_to(config.ROOT)}, {CATEGORIES_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
