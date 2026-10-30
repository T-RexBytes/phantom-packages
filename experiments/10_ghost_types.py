"""Stage 10: what kind of invention is each ghost? (deterministic taxonomy, phantom.ghosttypes)

Inputs: data/labelled.parquet (ghost mentions), data/registry.parquet (names that exist),
data/prompts.jsonl (task text) and the cached top-15k list. No models are called.

Outputs:
  private/ghost_types.csv            one row per ghost name with its type and evidence (never public)
  results/ghost_types.csv            per type: names, mentions, share, repeats (no names)
  results/ghost_types_breakdown.csv  per type x model and per type x niche level (no names)
  results/summary.json               "ghost_types" section
"""

from __future__ import annotations

import argparse

import pandas as pd

from phantom import config, ghosttypes, prompts, runconfig, topnames

RULES = {
    "affix_of_real": "a real package name plus or minus one common affix (python-, py, -py, -api, -sdk, -parser ...)",
    "import_or_brand": "the import name or bare brand of a real package that is installed under another name",
    "task_words": "every non-generic part of the name is a word of the task (or an abbreviation of one)",
    "near_real_name": "one or two edits from a real package name (a typo of it, or a coincidence)",
    "invented": "none of the rules above",
}
PRIVATE_PATH = config.PRIVATE / "ghost_types.csv"
PUBLIC_PATH = config.RESULTS / "ghost_types.csv"
BREAKDOWN_PATH = config.RESULTS / "ghost_types_breakdown.csv"


def classify_all(labelled: pd.DataFrame, registry: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """One row per ghost name: type, evidence, mentions, models, niche, prompt."""
    meta = {p["id"]: p for p in rows}
    real = set(topnames.load_index().names) | set(registry.loc[registry["exists"] == True, "name"])  # noqa: E712
    clf = ghosttypes.Classifier(real)
    g = labelled[labelled["y_ghost"]]
    out = []
    for name, d in g.groupby("name"):
        pid = d["prompt_id"].iloc[0]
        t = clf.classify(name, meta[pid]["text"])
        out.append({"name": name, "type": t.type, "evidence": t.evidence, "mentions": len(d),
                    "n_models": d["model"].nunique(), "models": ",".join(sorted(d["model"].unique())),
                    "prompt_id": pid, "niche_level": meta[pid]["niche_level"]})
    return pd.DataFrame(out)


def per_type(df: pd.DataFrame) -> pd.DataFrame:
    """Public aggregate per type, in rule order."""
    agg = df.groupby("type").agg(n_names=("name", "size"), n_mentions=("mentions", "sum"),
                                 repeated_2plus=("mentions", lambda m: int((m >= 2).sum())),
                                 by_2plus_models=("n_models", lambda m: int((m >= 2).sum())))
    agg = agg.reindex(ghosttypes.TYPES, fill_value=0)
    agg["share_of_names"] = (agg["n_names"] / agg["n_names"].sum()).round(4)
    agg["share_of_mentions"] = (agg["n_mentions"] / agg["n_mentions"].sum()).round(4)
    agg["rule"] = [RULES[t] for t in agg.index]
    return agg.reset_index()


def breakdown(df: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Type x model (mentions and names from each model) and type x niche level."""
    types = df.set_index("name")["type"]
    g = labelled[labelled["y_ghost"]].assign(type=lambda d: d["name"].map(types))
    by_model = g.groupby(["model", "type"]).agg(n_mentions=("key", "size"), n_names=("name", "nunique"))
    by_niche = df.groupby(["niche_level", "type"]).agg(n_mentions=("mentions", "sum"), n_names=("name", "size"))
    parts = []
    for scope, frame, key in (("model", by_model, "model"), ("niche_level", by_niche, "niche_level")):
        f = frame.reset_index().rename(columns={key: "group"})
        f["group"] = f["group"].map(lambda v: f"L{v}" if scope == "niche_level" else v)
        parts.append(f.assign(scope=scope))
    out = pd.concat(parts)[["scope", "group", "type", "n_names", "n_mentions"]]
    out["type"] = pd.Categorical(out["type"], ghosttypes.TYPES, ordered=True)
    return out.sort_values(["scope", "group", "type"]).reset_index(drop=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    labelled = pd.read_parquet(config.DATA / "labelled.parquet")
    registry = pd.read_parquet(config.REGISTRY_PATH)
    df = classify_all(labelled, registry, prompts.load_prompts())
    config.PRIVATE.mkdir(parents=True, exist_ok=True)
    df.sort_values(["type", "mentions"], ascending=[True, False]).to_csv(PRIVATE_PATH, index=False)
    table = per_type(df)
    table.to_csv(PUBLIC_PATH, index=False)
    breakdown(df, labelled).to_csv(BREAKDOWN_PATH, index=False)

    summary = runconfig.load(config.SUMMARY_PATH)  # reload right before saving: other stages write too
    summary["ghost_types"] = {
        "n_ghost_names": int(len(df)), "n_ghost_mentions": int(df["mentions"].sum()),
        "rules_in_order": RULES,
        "by_type": {r["type"]: {k: r[k] for k in ("n_names", "n_mentions", "share_of_names", "share_of_mentions",
                                                   "repeated_2plus", "by_2plus_models")}
                    for r in table.to_dict("records")},
    }
    runconfig.save(summary, config.SUMMARY_PATH)
    print(table[["type", "n_names", "n_mentions", "share_of_names", "repeated_2plus", "by_2plus_models"]]
          .to_string(index=False))
    print(f"wrote {PUBLIC_PATH.relative_to(config.ROOT)}, {BREAKDOWN_PATH.relative_to(config.ROOT)}, "
          f"summary.json[ghost_types]; per-name types in {PRIVATE_PATH.relative_to(config.ROOT)} (private)")


if __name__ == "__main__":
    main()
