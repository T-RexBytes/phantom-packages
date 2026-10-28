"""Stage 03: extract mentions, look names up on PyPI, label them (BUILD_SPEC §5.3, §6).

Outputs (data/ and private/ are gitignored):
  data/mentions.parquet, data/registry.parquet, data/labelled.parquet
  private/ghost_counts.csv           per-name ghost counts (never public)
  results/label_summary.json         aggregate counts only, no names
Ghost names are printed to the terminal only.
"""

from __future__ import annotations

import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from phantom import config, generate, label, registry, runconfig
from phantom.http import MetadataError

LABELLED_PATH = config.DATA / "labelled.parquet"
GHOST_COUNTS_PATH = config.PRIVATE / "ghost_counts.csv"
SUMMARY_PATH = config.RESULTS / "label_summary.json"
console = Console(highlight=False)


def lookup_all(names: list[str], refresh: bool) -> pd.DataFrame:
    """Registry rows for every name (failed lookups get ``exists=None``)."""
    pypi, stats = registry.clients()
    rows = []
    with pypi, stats:
        for i, name in enumerate(names, 1):
            try:
                rows.append(registry.lookup(name, pypi, stats, refresh=refresh))
            except MetadataError as exc:
                console.print(f"[yellow]lookup failed ({exc}); will retry next run")
                rows.append({**registry.parse_pypi(name, None), "exists": None})
            if i % 100 == 0 or i == len(names):
                console.print(f"  looked up {i}/{len(names)} (network calls: pypi {pypi.network_calls}, "
                              f"pypistats {stats.network_calls})")
    df = pd.DataFrame(rows)
    for col in ("first_release", "last_release"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="(cache is always used; kept for tasks.py)")
    parser.add_argument("--refresh", action="store_true", help="re-check every name on PyPI")
    args = parser.parse_args()

    records = list(generate.read_records(config.RAW_GENERATIONS_PATH))
    gens = pd.DataFrame(records)[["key", "model"]]
    mentions, dropped = label.build_mentions(records)
    config.DATA.mkdir(parents=True, exist_ok=True)
    mentions.to_parquet(config.MENTIONS_PATH, index=False)

    names = sorted(mentions.loc[mentions["source"].isin(label.LABEL_SOURCES), "name"].unique())
    console.print(f"{len(records)} generations -> {len(mentions)} mentions, {len(names)} unique names to label")
    reg = lookup_all(names, args.refresh)
    reg.to_parquet(config.REGISTRY_PATH, index=False)

    lab = label.label(mentions, reg)
    lab.to_parquet(LABELLED_PATH, index=False)
    r = label.rates(lab, gens)

    table = Table(title="Ghost rate (subtypes excluded)")
    for col in ("model", "mentions", "ghosts", "ghost rate", "generations", "gens with >=1 ghost"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for model, row in r.iterrows():
        table.add_row(str(model), f"{int(row.mentions)}", f"{int(row.ghosts)}", f"{row.ghost_rate:.1%}",
                      f"{int(row.generations)}", f"{row.gen_ghost_share:.1%}")
    console.print(table)

    ghosts = lab[lab["y_ghost"]]
    unique_ghosts = ghosts["name"].nunique()
    subtypes = lab.loc[lab["subtype"].notna()].groupby("subtype").agg(
        mentions=("name", "size"), unique=("name", "nunique"))
    unmapped = int((mentions["source"] == "import_unmapped").sum())
    failed = int(reg["exists"].isna().sum())
    console.print(f"labelled mentions: {len(lab)} | unique labelled names: {lab['name'].nunique()} | "
                  f"unique ghost names (404): {unique_ghosts}")
    console.print(f"subtypes: {subtypes.to_dict('index')}")
    console.print(f"unmapped imports (counted, not labelled): {unmapped} | dropped tokens: {dropped} | "
                  f"failed lookups: {failed} | born_after_summoning mentions: {int(lab['born_after_summoning'].sum())}")

    per_ghost = ghosts.groupby("name").agg(
        mentions=("key", "size"), models=("model", "nunique"), prompts=("prompt_id", "nunique"),
        model_list=("model", lambda s: ",".join(sorted(set(s)))),
    ).sort_values(["mentions", "models"], ascending=False)
    config.PRIVATE.mkdir(parents=True, exist_ok=True)
    per_ghost.to_csv(GHOST_COUNTS_PATH)

    console.rule("15 most frequent ghost names (terminal only, never committed)")
    for name, row in per_ghost.head(15).iterrows():
        console.print(f"  {name:<32} {row.mentions:>3} mentions  {row.models} model(s)  {row.prompts} prompt(s)")

    planned = runconfig.load().get("summon", {})
    expected = len(planned.get("models", [])) * len(planned.get("samples", [])) * planned.get("n_prompts", 0)
    summary = {
        "checked_at": runconfig.utc_now(),
        "n_generations": len(records),
        "n_generations_planned": expected,
        "complete": expected > 0 and len(records) >= expected,
        "n_mentions_extracted": len(mentions),
        "n_labelled_mentions": len(lab),
        "n_unique_labelled_names": int(lab["name"].nunique()),
        "n_unique_ghost_names": int(unique_ghosts),
        "n_ghost_mentions": int(lab["y_ghost"].sum()),
        "ghost_rate_overall": float(r.loc["ALL", "ghost_rate"]),
        "gen_ghost_share_overall": float(r.loc["ALL", "gen_ghost_share"]),
        "per_model": {m: {k: float(v) for k, v in row.items()} for m, row in r.drop(index="ALL").iterrows()},
        "subtypes": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in subtypes.to_dict("index").items()},
        "unmapped_imports": unmapped,
        "dropped_tokens": dropped,
        "failed_lookups": failed,
        "born_after_summoning_mentions": int(lab["born_after_summoning"].sum()),
        "ghosts_repeated_2plus": int((per_ghost["mentions"] >= 2).sum()),
        "ghosts_multi_model": int((per_ghost["models"] >= 2).sum()),
    }
    runconfig.save(summary, SUMMARY_PATH)
    console.print(f"Wrote {SUMMARY_PATH.relative_to(config.ROOT)} (aggregates only); "
                  f"per-name counts in {GHOST_COUNTS_PATH.relative_to(config.ROOT)} (private)")


if __name__ == "__main__":
    main()
