"""Error-analysis categoriser and aggregates on synthetic rows (no study data, no names)."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

import pandas as pd

from phantom import config

spec = importlib.util.spec_from_file_location("error_analysis", config.ROOT / "experiments" / "11_error_analysis.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def reg_row(first: str | None, n_releases, downloads):
    return pd.Series({"first_release": pd.Timestamp(first, tz="UTC") if first else pd.NaT,
                      "n_releases": n_releases, "recent_downloads_month": downloads})


def test_fp_categories():
    assert ea.fp_category(reg_row("2026-01-01", 1, 10), NOW) == "young"
    assert ea.fp_category(reg_row("2016-01-01", 2, 50_000), NOW) == "registry_quiet"  # few releases
    assert ea.fp_category(reg_row("2016-01-01", 40, 120), NOW) == "registry_quiet"    # few downloads
    assert ea.fp_category(reg_row("2016-01-01", 40, None), NOW) == "registry_quiet"   # no download data
    assert ea.fp_category(reg_row("2016-01-01", 40, 20_000), NOW) == "established"
    assert ea.fp_category(reg_row(None, 0, None), NOW) == "no_files"


def test_fn_categories():
    assert ea.fn_category("pyzzqfmt", 1) == "py_affixed"
    assert ea.fn_category("python-zzq", 2) == "py_affixed"
    assert ea.fn_category("zzqfmt-py", 2) == "py_affixed"
    assert ea.fn_category("vendor-circuit-zzq", 3) == "vendor_style"
    assert ea.fn_category("zzqfmt", 1) == "short_word"
    assert ea.fn_category("zzq-format-reader", 2) == "other"


def test_downloads_bucket():
    assert [ea.downloads_bucket(v) for v in (None, float("nan"), 5, 300, 900, 50_000)] == [
        "unknown", "unknown", "<100", "100-499", "500-9,999", ">=10,000"]


def test_analyse_counts_and_online_rescue():
    ds = pd.DataFrame({
        "name": ["zz-real-old", "zz-real-young", "zz-ghost-hit", "pyzzq", "zz-real-ok"],
        "model": "m", "prompt_id": "p001", "y_ghost": [False, False, True, True, False],
        "n_parts": [3, 3, 3, 1, 3], **{c: 0.0 for c in ea.DESCRIBE if c != "p"}})
    prob = pd.Series([0.9, 0.8, 0.95, 0.2, 0.1])
    registry = pd.DataFrame({"first_release": pd.to_datetime(["2015-01-01", "2026-06-01", None], utc=True),
                             "n_releases": [30, 1, None], "recent_downloads_month": [20_000, 5, None]},
                            index=["zz-real-old", "zz-real-young", "zz-real-ok"])
    agg, table = ea.analyse(ds, prob, 0.5, registry, NOW)
    c = agg["confusion"]
    assert (c["tp"]["rows"], c["fp"]["rows"], c["fn"]["rows"], c["tn"]["rows"]) == (1, 2, 1, 1)
    fp = agg["false_positives"]
    assert set(fp["by_category"]) == {"established", "young"}
    assert fp["rescued_online_rows"] == 1  # the young one stays blocked (SUSPECT) online
    assert agg["false_negatives"]["by_category"]["py_affixed"]["rows"] == 1
    json.dumps(agg, allow_nan=False)  # aggregates are valid JSON (no NaN)
    assert set(table["error"]) == {"FP", "FN"}
