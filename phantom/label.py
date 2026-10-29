"""Turn generations into mentions and label them against registry facts (BUILD_SPEC §5.3, §6, §8).

Categories of a labelled mention:
  real                  name exists on PyPI
  ghost                 404 on PyPI and no subtype (the main label, ``y_ghost``)
  module_name_confusion import name used as a pip name (``pip install cv2``)
  stdlib_confusion      standard-library module used as a pip name
Subtypes are reported separately and excluded from ``y_ghost`` rates.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from phantom import config
from phantom.extract import extract, subtype

LABEL_SOURCES = ("pip_line", "uv_line", "requirements_block", "import_mapped")
MENTION_COLUMNS = ["key", "model", "prompt_id", "sample", "arm", "raw_token", "source", "name",
                   "position", "subtype"]


def complete_text(record: dict[str, Any]) -> str:
    """Output without its unfinished last line when generation hit the token cap,
    so a name cut mid-word (``opencv-python-head``) is never labelled a ghost."""
    text = record["output"]
    if record.get("done_reason") == "length" and "\n" in text and not text.endswith("\n"):
        return text.rsplit("\n", 1)[0]
    return text


def build_mentions(records: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    """Extract mentions from generation records; returns (mentions, dropped token count)."""
    rows: list[dict[str, Any]] = []
    dropped = 0
    for r in records:
        ex = extract(complete_text(r))
        dropped += len(ex.dropped)
        for m in ex.mentions:
            rows.append({
                "key": r["key"], "model": r["model"], "prompt_id": r["prompt_id"],
                "sample": r["sample"], "arm": r.get("arm", "local"), "raw_token": m.raw_token,
                "source": m.source, "name": m.name, "position": m.position,
                # Subtypes only make sense for names the model typed as pip names.
                "subtype": subtype(m.name, m.raw_token) if m.source != "import_mapped" else None,
            })
    return pd.DataFrame(rows, columns=MENTION_COLUMNS), dropped


def label(mentions: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Labelled mentions: labellable sources with a successful lookup, plus category flags."""
    df = mentions[mentions["source"].isin(LABEL_SOURCES)].merge(
        registry[["name", "exists", "first_release"]], on="name", how="left"
    )
    df = df[df["exists"].notna()].copy()
    df["exists"] = df["exists"].astype(bool)
    df["category"] = df["subtype"].where(df["subtype"].notna(), df["exists"].map({True: "real", False: "ghost"}))
    df["y_ghost"] = df["category"] == "ghost"
    release = df["model"].map(
        {m: datetime.fromisoformat(d).replace(tzinfo=timezone.utc) for m, d in config.MODEL_RELEASE_DATE.items()}
    )
    df["born_after_summoning"] = df["exists"] & df["first_release"].notna() & release.notna() & (
        pd.to_datetime(df["first_release"], utc=True) > pd.to_datetime(release, utc=True)
    )
    return df.drop(columns=["first_release"])


def rates(labelled: pd.DataFrame, generations: pd.DataFrame) -> pd.DataFrame:
    """Per-model ghost counts and rates (subtypes excluded from the denominator)."""
    main = labelled[labelled["category"].isin(["real", "ghost"])]
    out = main.groupby("model").agg(mentions=("y_ghost", "size"), ghosts=("y_ghost", "sum"))
    out["ghost_rate"] = out["ghosts"] / out["mentions"]
    ghost_keys = set(main.loc[main["y_ghost"], "key"])
    gens = generations.groupby("model")["key"].agg(
        generations="size", with_ghost=lambda k: sum(x in ghost_keys for x in k)
    )
    out = out.join(gens, how="outer").fillna(0)
    out["gen_ghost_share"] = out["with_ghost"] / out["generations"]
    total = out[["mentions", "ghosts", "generations", "with_ghost"]].sum()
    out.loc["ALL"] = {**total, "ghost_rate": total["ghosts"] / total["mentions"],
                      "gen_ghost_share": total["with_ghost"] / total["generations"]}
    return out
