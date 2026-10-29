"""Generation-time features (BUILD_SPEC §7). Built only from mentions, prompts,
generation logprobs and the top-15k list, never from registry data (rule 7).

One dataset row per (model, prompt_id, name). ``global_freq`` is not stored:
it must be computed per fold with :func:`with_global_freq` and a training mask.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd

from phantom.charlm import CharTrigramLM
from phantom.extract import requirement_name
from phantom.label import LABEL_SOURCES
from phantom.prompts import DOMAINS
from phantom.topnames import TopIndex

NAME_FEATURES = ["len", "n_parts", "has_digit", "in_top15k", "nn_lev", "nn_jw", "nn_rank_log",
                 "affix_of_real", "char_lp", "char_lp_z"]
CONTEXT_FEATURES = ["consistency", "cross_model", "global_freq", "position", "source",
                    "niche_level", "domain", "model", "name_logprob"]
FEATURES = NAME_FEATURES + CONTEXT_FEATURES
CATEGORICAL = ["source", "domain", "model"]
PER_FOLD = ["global_freq"]


def name_features(name: str, index: TopIndex, lm: CharTrigramLM) -> dict[str, Any]:
    """Features of the name string alone (no registry, no generation context)."""
    _, dist, jw, rank_log = index.nearest(name)
    return {
        "len": len(name),
        "n_parts": len(name.split("-")),
        "has_digit": int(any(c.isdigit() for c in name)),
        "in_top15k": int(name in index),
        "nn_lev": dist,
        "nn_jw": jw,
        "nn_rank_log": rank_log,
        "affix_of_real": int(index.affix_match(name) is not None),
        "char_lp": lm.mean_logprob(name),
        "char_lp_z": lm.zscore(name),
    }


def name_logprob(record: dict[str, Any] | None, raw_token: str) -> float:
    """Mean generation logprob of the tokens that spell the name (NaN if unavailable)."""
    lp = (record or {}).get("logprobs")
    if not lp:
        return math.nan
    text = "".join(tok or "" for tok, _ in lp)
    if raw_token.startswith("import "):
        target = raw_token[len("import "):]
    elif raw_token.startswith("from "):
        target = raw_token[len("from "):].split(" import")[0]
    else:
        target = requirement_name(raw_token)
    at = text.find(raw_token)
    if at < 0 or not target:
        return math.nan
    start = at + raw_token.find(target)
    end = start + len(target)
    vals, pos = [], 0
    for tok, val in lp:
        nxt = pos + len(tok or "")
        if nxt > start and pos < end and val is not None:
            vals.append(val)
        pos = nxt
        if pos >= end:
            break
    return statistics.fmean(vals) if vals else math.nan


def categories(models: list[str]) -> dict[str, list[str]]:
    """Fixed category vocabularies; integer code = list index."""
    return {"source": list(LABEL_SOURCES), "domain": list(DOMAINS.values()), "model": sorted(models)}


def build(
    labelled: pd.DataFrame,
    generations: pd.DataFrame,
    prompt_rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    index: TopIndex,
    lm: CharTrigramLM,
    cats: dict[str, list[str]],
) -> pd.DataFrame:
    """Dataset rows for real/ghost mentions (subtypes excluded, §8).

    ``generations`` has one row per generation (``key, model, prompt_id``) and
    supplies the sample count per (model, prompt) for ``consistency``.
    """
    m = labelled[labelled["category"].isin(["real", "ghost"])].copy()
    m["name_logprob"] = [name_logprob(records.get(k), raw) for k, raw in zip(m["key"], m["raw_token"])]
    rows = m.groupby(["model", "prompt_id", "name"], sort=True).agg(
        n_gens=("key", "nunique"),
        position=("position", "min"),
        source=("source", "first"),
        name_logprob=("name_logprob", "mean"),
        y_ghost=("y_ghost", "first"),
    ).reset_index()

    samples = generations.groupby(["model", "prompt_id"])["key"].nunique().rename("n_samples")
    rows = rows.join(samples, on=["model", "prompt_id"])
    rows["consistency"] = rows["n_gens"] / rows["n_samples"]
    rows["cross_model"] = rows.groupby(["prompt_id", "name"])["model"].transform("nunique")

    meta = {p["id"]: p for p in prompt_rows}
    rows["niche_level"] = rows["prompt_id"].map(lambda p: meta[p]["niche_level"])
    rows["domain_name"] = rows["prompt_id"].map(lambda p: meta[p]["domain"])

    name_cols = pd.DataFrame([{"name": n, **name_features(n, index, lm)} for n in rows["name"].unique()])
    rows = rows.merge(name_cols, on="name", how="left")

    rows["source"] = rows["source"].map(cats["source"].index)
    rows["domain"] = rows["domain_name"].map(cats["domain"].index)
    rows["model_code"] = rows["model"].map(cats["model"].index)
    return rows.drop(columns=["domain_name"])


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs in ``FEATURES`` order (``model`` as its integer code)."""
    return df.rename(columns={"model": "model_tag", "model_code": "model"})[FEATURES]


def with_global_freq(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Add ``global_freq`` using training rows only.

    Count = generations mentioning the name in training rows, minus the row's own
    (model, prompt) contribution, plus that contribution. Training rows thus see the
    training total; test rows see training counts plus their own occurrence, which is
    exactly what the guard sees at inference time. Test-fold names never inflate it.
    """
    train_total = df.loc[train_mask].groupby("name")["n_gens"].sum()
    base = df["name"].map(train_total).fillna(0)
    own = df["n_gens"]
    out = df.copy()
    out["global_freq"] = base - own.where(train_mask, 0) + own
    return out
