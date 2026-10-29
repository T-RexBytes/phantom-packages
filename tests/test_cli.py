"""CLI exit codes and --json schema (BUILD_SPEC §9.2). guard.check is faked: no network, no model."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from phantom import cli, guard

runner = CliRunner()

CANNED = {
    "requests": "OK",
    "zzq-freshlib": "SUSPECT",
    "zzq-ghostlib": "GHOST",
}


@pytest.fixture
def fake_check(monkeypatch):
    calls = []

    def fake(names, offline=False, band=False, prob="auto", **kw):
        calls.append({"names": list(names), "offline": offline, "band": band, "prob": prob})
        out = []
        for n in dict.fromkeys(guard.normalize(x) for x in names):
            kind = CANNED.get(n, "OK")
            if offline:
                kind = "GHOST_LIKELY" if kind == "GHOST" else "OK_UNVERIFIED"
            out.append(guard.Verdict(n, n, kind, 0.9 if "GHOST" in kind else 0.1, (0.05, 0.95) if band else None,
                                     ["404 on PyPI"] if kind == "GHOST" else ["exists on PyPI"],
                                     "requests" if n != "requests" else None, 3 if n != "requests" else None,
                                     None, not offline))
        return out

    monkeypatch.setattr(guard, "check", fake)
    return calls


@pytest.mark.parametrize("args,code", [
    (["requests"], 0),
    (["requests", "zzq-freshlib"], 1),
    (["requests", "zzq-freshlib", "zzq-ghostlib"], 2),
])
def test_check_exit_codes(fake_check, args, code):
    result = runner.invoke(cli.app, ["check", *args])
    assert result.exit_code == code, result.output
    assert "requests" in result.output


def test_check_reads_files(fake_check, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("requests==2.32\nzzq-ghostlib\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["check", str(req)])
    assert result.exit_code == 2 and fake_check[0]["names"] == ["requests", "zzq-ghostlib"]


def test_check_json_schema(fake_check):
    result = runner.invoke(cli.app, ["check", "zzq-ghostlib", "requests", "--json"])
    assert result.exit_code == 2
    data = json.loads(result.output)
    assert data["exit_code"] == 2 and [v["verdict"] for v in data["verdicts"]] == ["GHOST", "OK"]
    assert set(data["verdicts"][0]) == {"name", "normalized", "verdict", "p_ghost", "band", "reasons",
                                        "nearest", "nearest_lev", "registry", "checked_online"}


def test_offline_and_band_flags_are_passed(fake_check):
    result = runner.invoke(cli.app, ["check", "zzq-ghostlib", "--offline", "--band"])
    assert result.exit_code == 2 and "GHOST_LIKELY" in result.output
    assert fake_check[0]["offline"] is True and fake_check[0]["band"] is True


@pytest.mark.parametrize("args", [
    ["check"],  # nothing to check
    ["check", "missing-requirements.txt"],  # looks like a file, doesn't exist
    ["check", "not a name!"],
    ["check", "--no-such-flag", "requests"],  # Click usage error: 3, not Click's 2
    ["why"],
])
def test_usage_errors_exit_3(fake_check, args):
    assert runner.invoke(cli.app, args).exit_code == 3


def test_unreadable_pyproject_is_usage_error(fake_check, tmp_path):
    bad = tmp_path / "pyproject.toml"
    bad.write_text("[project\n", encoding="utf-8")
    assert runner.invoke(cli.app, ["check", str(bad)]).exit_code == 3


def test_scan_cmd(fake_check):
    result = runner.invoke(cli.app, ["scan-cmd", "cd x && pip install zzq-ghostlib requests"])
    assert result.exit_code == 2 and fake_check[0]["names"] == ["zzq-ghostlib", "requests"]
    result = runner.invoke(cli.app, ["scan-cmd", "ls -la", "--json"])
    assert result.exit_code == 0 and json.loads(result.output) == {"verdicts": [], "exit_code": 0}
    result = runner.invoke(cli.app, ["scan-cmd", "echo hi"])
    assert result.exit_code == 0 and "no installs found" in result.output


def test_why_prints_a_card_and_exits_0(fake_check, monkeypatch):
    monkeypatch.setattr(guard, "describe", lambda v: {"registry": "404 on PyPI", "P(ghost)": "0.900"})
    result = runner.invoke(cli.app, ["why", "zzq-ghostlib"])
    assert result.exit_code == 0 and "404 on PyPI" in result.output
    assert fake_check[0] == {"names": ["zzq-ghostlib"], "offline": False, "band": True, "prob": "always"}


def test_stats_reads_summary(monkeypatch, tmp_path):
    summary = {
        "rates": {"overall": {"ghost_rate": 0.15, "rate_lo": 0.12, "rate_hi": 0.18, "mentions": 100,
                              "gen_share": 0.2},
                  "per_model": {"m1": {"ghost_rate": 0.1, "ghosts": 1, "mentions": 10}}},
        "eval": {"A": {"tabpfn": {"auroc": 0.97, "auroc_lo": 0.96, "auroc_hi": 0.98}}},
        "threshold": {"tau": 0.5, "recall": 0.9, "precision": 0.75},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(cli.config, "SUMMARY_PATH", path)
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0 and "15.0%" in result.output and "0.970" in result.output
    monkeypatch.setattr(cli.config, "SUMMARY_PATH", tmp_path / "missing.json")
    assert runner.invoke(cli.app, ["stats"]).exit_code == 3
