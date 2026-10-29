"""Stage 06: re-check candidate names on PyPI and apply the squat rules (BUILD_SPEC §5.6).

Candidates (phantom.squat.select_candidates): every name that was 404 at labelling time,
every real name with out-of-fold P(ghost) >= 0.5, and every born-after-summoning name.
Each is re-checked now with metadata GETs only (nothing is downloaded or installed);
re-checks are cached under data/cache/.../recheck/ so the labelling-time cache stays intact.

Outputs:
  private/squat_full.csv               every candidate, all columns (never public)
  private/unregistered_repeated.csv    still-unregistered names summoned >= 2 times (never public)
  private/disclosure_draft.md          DRAFT email to security@pypi.org (not sent; the human decides)
  results/squat_public.csv             registered-now names only
  results/anon_ghost_ids.txt           sha256(name)[:8] of still-unregistered ghosts (CI leak check)
  results/squat_summary.json           counts; also merged into results/summary.json["squat"]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from rich.console import Console

from phantom import config, privacy, registry, runconfig, squat
from phantom.http import MetadataError

LABELLED_PATH = config.DATA / "labelled.parquet"
OOF_PATH = config.DATA / "oof_predictions.parquet"
FULL_PATH = config.PRIVATE / "squat_full.csv"
UNREGISTERED_REPEATED_PATH = config.PRIVATE / "unregistered_repeated.csv"
DISCLOSURE_PATH = config.PRIVATE / "disclosure_draft.md"
PUBLIC_PATH = config.RESULTS / "squat_public.csv"
SQUAT_SUMMARY_PATH = config.RESULTS / "squat_summary.json"
RECHECK_NAMESPACE = "recheck/"
NOW_FIELDS = ["first_release", "last_release", "n_releases", "recent_downloads_month", "summary",
              "has_project_urls"]
RULE_TEXT = [
    "exists on PyPI now",
    f"n_releases <= {config.SQUAT_MAX_RELEASES}",
    f"first_release within the last {config.SQUAT_MAX_AGE_MONTHS} months",
    f"recent_downloads_month < {config.SQUAT_MAX_MONTHLY_DOWNLOADS}",
    f"summoned by >= {config.SQUAT_MIN_SUMMONS} independent (model, sample) pairs",
]
console = Console(highlight=False)


def recheck(names: list[str], refresh: bool) -> tuple[dict[str, dict[str, Any] | None], int]:
    """Registry rows now for each name (None on a failed lookup) and the failure count."""
    pypi, stats = registry.clients()
    rows: dict[str, dict[str, Any] | None] = {}
    failed = 0
    with pypi, stats:
        for i, name in enumerate(names, 1):
            try:
                rows[name] = registry.lookup(name, pypi, stats, refresh=refresh, namespace=RECHECK_NAMESPACE)
            except MetadataError as exc:
                console.print(f"[yellow]re-check failed ({exc}); will retry next run")
                rows[name] = None
                failed += 1
            if i % 50 == 0 or i == len(names):
                console.print(f"  re-checked {i}/{len(names)} (network calls: pypi {pypi.network_calls}, "
                              f"pypistats {stats.network_calls})")
    return rows, failed


def assess(candidates: pd.DataFrame, now_rows: dict[str, dict[str, Any] | None],
           now: datetime | None = None) -> pd.DataFrame:
    """Candidates plus their registry state now, the squat checks, verdict, reasons and anon id."""
    now = now or datetime.now(timezone.utc)
    out = []
    for cand in candidates.to_dict("records"):
        row_now = now_rows.get(cand["name"])
        rec = {**cand, "anon_id": privacy.anon_id(cand["name"])}
        if row_now is None:  # failed lookup: unknown now, no verdict
            rec.update({"exists_now": None, "registered_during_study": None, "possible_slopsquat": None,
                        "reasons": "", "checked_at": None, **{f: None for f in NOW_FIELDS},
                        **{c: None for c in squat.CHECKS}})
        else:
            c = squat.checks(row_now, cand["summons"], now=now)
            rec.update({
                "exists_now": bool(row_now["exists"]),
                "registered_during_study": (not bool(cand["exists_at_label"])) and bool(row_now["exists"]),
                **{f: row_now.get(f) for f in NOW_FIELDS},
                **c,
                "possible_slopsquat": squat.possible_slopsquat(c),
                "reasons": "; ".join(squat.reasons(row_now, c, cand["summons"])),
                "checked_at": row_now.get("checked_at"),
            })
        out.append(rec)
    return pd.DataFrame(out)


def public_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Registered-now names only (never a currently unregistered name)."""
    reg = df[df["exists_now"].eq(True)].copy()
    reg["first_release"] = pd.to_datetime(reg["first_release"], utc=True).dt.strftime("%Y-%m-%d")
    reg["p_ghost_oof"] = reg["p_ghost_oof"].round(3)
    cols = ["name", "why_candidate", "registered_during_study", "born_after_summoning", "first_release",
            "n_releases", "recent_downloads_month", "summons", "n_models", *squat.CHECKS,
            "possible_slopsquat", "p_ghost_oof"]
    return reg[cols].sort_values(["possible_slopsquat", "summons", "name"], ascending=[False, False, True])


def still_unregistered_ghosts(df: pd.DataFrame) -> pd.DataFrame:
    """Label-time ghosts that are still 404 now."""
    return df[df["exists_at_label"].eq(False) & df["exists_now"].eq(False)]


def unregistered_repeated(df: pd.DataFrame) -> pd.DataFrame:
    """Still-unregistered ghosts summoned at least ``SQUAT_MIN_SUMMONS`` times (private)."""
    ghosts = still_unregistered_ghosts(df)
    rep = ghosts[ghosts["summons"] >= config.SQUAT_MIN_SUMMONS]
    cols = ["name", "anon_id", "summons", "mentions", "n_models", "n_prompts", "models"]
    return rep[cols].sort_values(["summons", "name"], ascending=[False, True])


def summarize(df: pd.DataFrame, failed: int, checked_at: str) -> dict[str, Any]:
    """Aggregate counts for results/squat_summary.json (no names)."""
    registered = df[df["exists_now"].eq(True)]
    return {
        "checked_at": checked_at,
        "n_candidates": int(len(df)),
        "by_why_candidate": {k: int(v) for k, v in df["why_candidate"].value_counts().items()},
        "n_ghosts_at_label": int(df["exists_at_label"].eq(False).sum()),
        "n_still_unregistered": int(len(still_unregistered_ghosts(df))),
        "n_registered_during_study": int(df["registered_during_study"].eq(True).sum()),
        "n_unregistered_repeated": int(len(unregistered_repeated(df))),
        "n_registered_candidates": int(len(registered)),
        "n_possible_slopsquat": int(df["possible_slopsquat"].eq(True).sum()),
        "n_born_after_summoning_names": int(df["born_after_summoning"].eq(True).sum()),
        "failed_lookups": int(failed),
        "rule": "possible_slopsquat if ALL hold: " + "; ".join(RULE_TEXT),
    }


def _count(value) -> str:
    """Whole-number display for counts that pandas stored as floats ("?" if missing)."""
    return "?" if pd.isna(value) else str(int(value))


def disclosure_markdown(df: pd.DataFrame, summary: dict[str, Any]) -> str:
    """DRAFT email to security@pypi.org. Never sent automatically."""
    flagged = df[df["possible_slopsquat"].eq(True) | df["registered_during_study"].eq(True)]
    lines = [
        "# DRAFT: NOT SENT. The human decides whether to send this.",
        "",
        "**To:** security@pypi.org",
        "**Subject:** Research note: package names hallucinated by local LLMs that are now registered",
        "",
        "Hello PyPI security team,",
        "",
        "I'm {{YOUR NAME}} ({{CONTACT}}). For a research project ({{REPO URL}}) I asked three small "
        "open-weight language models to write short Python scripts and recorded every `pip install` "
        "name they suggested. I only made metadata requests (the PyPI JSON API and pypistats); "
        "nothing was installed or downloaded, and I have not registered any names.",
        "",
        "**Method, in brief:**",
        f"1. {summary['n_candidates']} candidate names: names that were 404 at labelling time, real names "
        "my detector scored as ghost-like, and names first released after the model that suggested them.",
        f"2. Each was re-checked on {summary['checked_at']} via `https://pypi.org/pypi/<name>/json`.",
        "3. Flag rule (all must hold): " + "; ".join(RULE_TEXT) + ".",
        "",
        "**Names that match the rule or were registered during the study window:**",
        "",
        "| name | PyPI | registered during study | first release | releases | downloads/month | "
        "times suggested | models | rule match |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in flagged.to_dict("records"):
        first = pd.to_datetime(r["first_release"], utc=True)
        lines.append(
            f"| `{r['name']}` | https://pypi.org/project/{r['name']}/ | {bool(r['registered_during_study'])} | "
            f"{first.date() if pd.notna(first) else '?'} | {_count(r['n_releases'])} | {_count(r['recent_downloads_month'])} | "
            f"{r['summons']} | {r['n_models']} | {bool(r['possible_slopsquat'])} |")
    if flagged.empty:
        lines.append("| _none_ | | | | | | | | |")
        lines += ["", "_No name matched the rule or was registered during the study. There may be "
                      "nothing to send._"]
    lines += [
        "",
        "**Caveat:** a rule match is not evidence of malice. These are young, rarely downloaded packages "
        "whose names small models suggest repeatedly; many may be legitimate. I have not inspected or "
        "executed their contents. I'm sharing them in case they are useful for your own review.",
        "",
        "Thank you,",
        "{{YOUR NAME}}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="(cache is used unless --recheck; kept for tasks.py)")
    parser.add_argument("--recheck", action="store_true", help="fetch every candidate again (ignore the recheck cache)")
    args = parser.parse_args()

    labelled = pd.read_parquet(LABELLED_PATH)
    reg = pd.read_parquet(config.REGISTRY_PATH)
    oof = pd.read_parquet(OOF_PATH)
    candidates = squat.select_candidates(labelled, reg, oof)
    console.print(f"{len(candidates)} candidates: "
                  + ", ".join(f"{k} {v}" for k, v in candidates["why_candidate"].value_counts().items()))

    checked_at = runconfig.utc_now()
    now_rows, failed = recheck(sorted(candidates["name"]), refresh=args.recheck)
    df = assess(candidates, now_rows)

    config.PRIVATE.mkdir(parents=True, exist_ok=True)
    df.to_csv(FULL_PATH, index=False)
    unregistered_repeated(df).to_csv(UNREGISTERED_REPEATED_PATH, index=False)
    privacy.write_anon_ids(still_unregistered_ghosts(df)["name"])
    public_frame(df).to_csv(PUBLIC_PATH, index=False)

    summary = summarize(df, failed, checked_at)
    runconfig.save(summary, SQUAT_SUMMARY_PATH)
    full_summary = runconfig.load(config.SUMMARY_PATH) if config.SUMMARY_PATH.exists() else {}
    full_summary["squat"] = summary
    runconfig.save(full_summary, config.SUMMARY_PATH)
    DISCLOSURE_PATH.write_text(disclosure_markdown(df, summary), encoding="utf-8")
    runconfig.update("squat", {"run_at": checked_at, "recheck": args.recheck, "namespace": RECHECK_NAMESPACE,
                               "rules": RULE_TEXT, "counts": summary})

    console.print(" | ".join(f"{k}: {v}" for k, v in summary.items() if k.startswith("n_") or k == "failed_lookups"))
    console.rule("Registered during the study (terminal only)")
    for r in df[df["registered_during_study"].eq(True)].to_dict("records"):
        console.print(f"  {r['name']:<32} first release {r['first_release']}  {r['reasons']}")
    console.rule("possible_slopsquat (terminal only)")
    for r in df[df["possible_slopsquat"].eq(True)].to_dict("records"):
        console.print(f"  {r['name']:<32} {r['reasons']}")
    console.print(f"Wrote {PUBLIC_PATH.relative_to(config.ROOT)}, {SQUAT_SUMMARY_PATH.relative_to(config.ROOT)}, "
                  f"{privacy.ANON_IDS_PATH.relative_to(config.ROOT)}; private: squat_full.csv, "
                  "unregistered_repeated.csv, disclosure_draft.md")


if __name__ == "__main__":
    main()
