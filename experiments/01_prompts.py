"""Stage 01: build data/prompts.jsonl and print domain x niche-level counts."""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd
from rich.console import Console
from rich.table import Table

from phantom import config, prompts, runconfig

COUNTS_PATH = config.RESULTS / "prompt_counts.csv"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    rows = prompts.default_prompts()
    prompts.write_jsonl(rows)

    df = pd.DataFrame(rows)
    counts = pd.crosstab(df["domain"], df["niche_level"]).reindex(prompts.DOMAINS.values())
    counts.columns = [f"L{c}" for c in counts.columns]
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(COUNTS_PATH, index_label="domain")

    by_level = df["niche_level"].value_counts().sort_index()
    table = Table(title=f"{len(df)} prompts: domain x niche level")
    for col in ("domain", "L1", "L2", "L3", "total"):
        table.add_column(col, justify="left" if col == "domain" else "right")
    for domain, r in counts.iterrows():
        table.add_row(domain, *(str(int(r[c])) for c in ("L1", "L2", "L3", "total")))
    table.add_section()
    table.add_row(
        "total",
        *(f"{int(by_level[lvl])} ({by_level[lvl] / len(df):.0%})" for lvl in (1, 2, 3)),
        str(len(df)),
    )
    console = Console()
    console.print(table)

    digest = hashlib.sha256(config.PROMPTS_PATH.read_bytes()).hexdigest()
    runconfig.update(
        "prompts",
        {
            "n_prompts": len(df),
            "n_domains": int(df["domain"].nunique()),
            "by_level": {f"L{k}": int(v) for k, v in by_level.items()},
            "by_origin": {k: int(v) for k, v in df["origin"].value_counts().items()},
            "sha256": digest,
            "template": config.PROMPT_TEMPLATE,
            "system_prompt": config.SYSTEM_PROMPT,
        },
    )
    console.print(f"Wrote {config.PROMPTS_PATH.relative_to(config.ROOT)} (sha256 {digest[:12]})")
    console.print(f"Wrote {COUNTS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
