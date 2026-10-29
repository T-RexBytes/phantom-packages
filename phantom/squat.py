"""Squat rules and candidate selection (BUILD_SPEC §5.6).

A name is a ``possible_slopsquat`` when **all** checks hold: it exists, has at most
``SQUAT_MAX_RELEASES`` releases, a first release within ``SQUAT_MAX_AGE_MONTHS`` months,
fewer than ``SQUAT_MAX_MONTHLY_DOWNLOADS`` downloads last month, and was summoned by at
least ``SQUAT_MIN_SUMMONS`` independent (model, sample) pairs. Decisions come from
registry facts and counts only; no model writes a verdict or a reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from phantom import config

CHECKS = ["exists", "few_releases", "young", "low_downloads", "repeated"]
DAYS_PER_MONTH = 30.44
CANDIDATE_REASONS = ("ghost_at_label", "born_after_summoning", "p_ghost>=0.5")
P_GHOST_CANDIDATE = 0.5


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def checks(row: Mapping[str, Any], summons: int, now: datetime | None = None) -> dict[str, bool]:
    """Each §5.6 check for one registry row; registry checks are False for missing names.

    Missing download data (pypistats has no record) counts as low downloads.
    """
    now_ts = _utc(now or datetime.now(timezone.utc))
    exists = not _missing(row.get("exists")) and bool(row.get("exists"))
    n_releases, first, downloads = row.get("n_releases"), row.get("first_release"), row.get("recent_downloads_month")
    return {
        "exists": exists,
        "few_releases": exists and not _missing(n_releases) and int(n_releases) <= config.SQUAT_MAX_RELEASES,
        "young": exists and not _missing(first) and (
            now_ts - _utc(first) <= timedelta(days=DAYS_PER_MONTH * config.SQUAT_MAX_AGE_MONTHS)),
        "low_downloads": exists and (_missing(downloads) or int(downloads) < config.SQUAT_MAX_MONTHLY_DOWNLOADS),
        "repeated": int(summons) >= config.SQUAT_MIN_SUMMONS,
    }


def possible_slopsquat(c: Mapping[str, bool]) -> bool:
    """True only if every check in ``CHECKS`` holds."""
    return all(c[k] for k in CHECKS)


def reasons(row: Mapping[str, Any], c: Mapping[str, bool], summons: int,
            now: datetime | None = None) -> list[str]:
    """Short, factual reason strings built from registry facts and summon counts."""
    pairs = f"summoned by {int(summons)} (model, sample) pair{'s' if int(summons) != 1 else ''}"
    if not c["exists"]:
        return ["404 on PyPI", pairs]
    out = []
    n_releases = row.get("n_releases")
    if not _missing(n_releases):
        out.append(f"{int(n_releases)} release{'s' if int(n_releases) != 1 else ''}")
    first = row.get("first_release")
    if not _missing(first):
        first_ts = _utc(first)
        months = int((_utc(now or datetime.now(timezone.utc)) - first_ts).days / DAYS_PER_MONTH)
        out.append(f"first release {first_ts.date().isoformat()} ({months} month{'s' if months != 1 else ''} ago)")
    downloads = row.get("recent_downloads_month")
    out.append("no download data" if _missing(downloads) else f"~{int(downloads):,} downloads/month")
    out.append(pairs)
    return out


def _main(labelled: pd.DataFrame) -> pd.DataFrame:
    return labelled[labelled["category"].isin(["real", "ghost"])]


def summon_counts(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per name: distinct (model, sample) pairs (``summons``), mentions, models and prompts."""
    m = _main(labelled).assign(sample=lambda d: d["key"].str.rsplit("|", n=1).str[-1])
    m = m.assign(pair=m["model"] + "|" + m["sample"])
    return m.groupby("name").agg(
        summons=("pair", "nunique"), mentions=("key", "size"), n_models=("model", "nunique"),
        n_prompts=("prompt_id", "nunique"), models=("model", lambda s: ",".join(sorted(set(s)))),
    ).reset_index()


def select_candidates(labelled: pd.DataFrame, registry: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """§5.6 candidates, one row per name: 404 at labelling, born after summoning, or OOF p >= 0.5.

    ``oof`` holds out-of-fold TabPFN probabilities (column ``A_tabpfn``); the max per name is used.
    """
    m = _main(labelled)
    reg = registry[["name", "exists", "first_release"]].rename(
        columns={"exists": "exists_at_label", "first_release": "first_release_at_label"})
    born = m.groupby("name")["born_after_summoning"].any().rename("born_after_summoning")
    p = oof.groupby("name")["A_tabpfn"].max().rename("p_ghost_oof")
    df = summon_counts(labelled).merge(reg, on="name", how="left").join(born, on="name").join(p, on="name")
    exists = df["exists_at_label"].eq(True)
    flags = pd.DataFrame({
        "ghost_at_label": df["exists_at_label"].eq(False),
        "born_after_summoning": exists & df["born_after_summoning"].fillna(False).astype(bool),
        "p_ghost>=0.5": exists & df["p_ghost_oof"].ge(P_GHOST_CANDIDATE),
    })
    df["why_candidate"] = flags.apply(lambda r: ";".join(k for k in CANDIDATE_REASONS if r[k]), axis=1)
    cols = ["name", "exists_at_label", "first_release_at_label", "p_ghost_oof", "born_after_summoning",
            "why_candidate", "summons", "mentions", "n_models", "n_prompts", "models"]
    return df.loc[df["why_candidate"] != "", cols].sort_values("name").reset_index(drop=True)
