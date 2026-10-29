"""Guard verdicts (BUILD_SPEC §9.1, §11): GHOST on 404, SUSPECT on young + few releases,
OK on popular, offline path, plus command and file parsing. Mocked PyPI, fake model."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from phantom import guard
from phantom.http import MetadataClient
from phantom.topnames import TopIndex

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
INDEX = TopIndex(["requests", "numpy", "pandas", "pyyaml"])


def doc(*times, summary="demo"):
    return {"info": {"summary": summary, "project_urls": {"Home": "x"}, "version": "1"},
            "releases": {f"{i}.0": [{"upload_time_iso_8601": t}] for i, t in enumerate(times)}}


PYPI = {
    "requests": doc(*[f"20{y:02d}-01-01T00:00:00Z" for y in range(11, 26)], summary="HTTP for Humans"),
    "zzq-freshlib": doc("2026-09-01T00:00:00Z"),  # 1 release, 2 months old
    "zzq-busylib": doc(*[f"2026-0{m}-01T00:00:00Z" for m in range(1, 7)]),  # young, 6 releases
}
DOWNLOADS = {"requests": 900_000_000, "zzq-freshlib": 12, "zzq-busylib": 40_000}


class FakeModel:
    """Same interface as phantom.model.GuardModel."""

    tau = 0.5

    def __init__(self, probs=None):
        self.probs = probs or {}
        self.calls: list[list[str]] = []

    def predict(self, names):
        self.calls.append(list(names))
        return {n: (self.probs.get(n, 0.1), 0.05, 0.2) for n in names}


def fake_clients(tmp_path, fail=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("offline", request=request)
        parts = request.url.path.strip("/").split("/")
        if request.url.host == "pypi.org":
            name = parts[1]
            return httpx.Response(200, json=PYPI[name]) if name in PYPI else httpx.Response(404)
        name = parts[2]  # /api/packages/<name>/recent
        return httpx.Response(200, json={"data": {"last_month": DOWNLOADS[name]}})

    def make():
        return MetadataClient(cache_dir=tmp_path, user_agent="t", transport=httpx.MockTransport(handler),
                              sleep=lambda s: None, max_retries=0)
    return make(), make()


def run(tmp_path, names, fail=False, **kw):
    kw.setdefault("model", FakeModel())
    return guard.check(names, clients=fake_clients(tmp_path, fail), index=INDEX, now=NOW, **kw)


def test_ghost_on_404_with_nearest_real_name(tmp_path):
    model = FakeModel({"zzq-ghostlib": 0.97})
    [v] = run(tmp_path, ["zzq-ghostlib"], model=model)
    assert v.verdict == "GHOST" and v.checked_online and "404 on PyPI" in v.reasons
    assert v.p_ghost == pytest.approx(0.97) and v.nearest is not None and v.nearest_lev is not None
    assert guard.exit_code([v]) == 2


def test_suspect_on_young_few_releases_low_downloads(tmp_path):
    [v] = run(tmp_path, ["zzq-freshlib"])
    assert v.verdict == "SUSPECT"
    assert any("1 release" in r for r in v.reasons) and any("downloads/month" in r for r in v.reasons)
    assert not any("summoned" in r for r in v.reasons)  # study-only rule is not used by the guard
    assert guard.exit_code([v]) == 1


def test_suspect_when_young_and_model_says_ghost(tmp_path):
    [v] = run(tmp_path, ["zzq-busylib"], model=FakeModel({"zzq-busylib": 0.8}))
    assert v.verdict == "SUSPECT" and any("P(ghost)" in r for r in v.reasons)
    [v] = run(tmp_path, ["zzq-busylib"], model=FakeModel({"zzq-busylib": 0.2}))
    assert v.verdict == "OK"


def test_ok_on_popular_without_calling_the_model(tmp_path):
    model = FakeModel()
    [v] = run(tmp_path, ["Requests"], model=model)
    assert v.verdict == "OK" and v.normalized == "requests" and v.nearest is None
    assert v.registry["exists"] is True and v.registry["n_releases"] == 15
    assert model.calls == []  # prob="auto": mature existing names need no model
    assert guard.exit_code([v]) == 0


def test_prob_always_and_never(tmp_path):
    model = FakeModel()
    [v] = run(tmp_path, ["requests"], model=model, prob="always")
    assert model.calls == [["requests"]] and v.p_ghost == pytest.approx(0.1)
    model = FakeModel()
    [v] = run(tmp_path, ["zzq-ghostlib"], model=model, prob="never")
    assert v.verdict == "GHOST" and v.p_ghost is None and model.calls == []


def test_offline_path_never_touches_the_network(tmp_path):
    model = FakeModel({"zzq-ghostlib": 0.9, "requests": 0.01})
    verdicts = run(tmp_path, ["zzq-ghostlib", "requests"], fail=True, model=model, offline=True)
    assert [v.verdict for v in verdicts] == ["GHOST_LIKELY", "OK_UNVERIFIED"]
    assert not any(v.checked_online for v in verdicts) and model.calls == [["zzq-ghostlib", "requests"]]
    assert guard.exit_code(verdicts) == 2


def test_unreachable_pypi_falls_back_to_offline_estimate(tmp_path):
    [v] = run(tmp_path, ["zzq-ghostlib"], fail=True, model=FakeModel({"zzq-ghostlib": 0.9}))
    assert v.verdict == "GHOST_LIKELY" and not v.checked_online
    assert "PyPI unreachable: offline estimate" in v.reasons


def test_suggestion_prefers_affix_match_over_edit_distance(tmp_path):
    [v] = run(tmp_path, ["python-pyyaml"], model=FakeModel())  # 404; "pyyaml" minus nothing but a prefix
    assert v.verdict == "GHOST" and v.nearest == "pyyaml"
    assert "nearest real name: pyyaml (edit distance 7)" in v.reasons


def test_missing_model_still_gives_registry_verdict(tmp_path):
    class BrokenModel:  # e.g. the Action / CI: base install without the `model` extra
        tau = 0.5

        def predict(self, names):
            raise ModuleNotFoundError("No module named 'tabpfn'")

    [ghost, ok] = run(tmp_path, ["zzq-ghostlib", "requests"], model=BrokenModel(), prob="always")
    assert ghost.verdict == "GHOST" and ghost.p_ghost is None and ok.verdict == "OK"
    assert any("P(ghost) unavailable" in r for r in ghost.reasons)


def test_stdlib_is_ok_and_skips_registry_and_model(tmp_path):
    model = FakeModel()
    [v] = run(tmp_path, ["json"], fail=True, model=model)
    assert v.verdict == "OK" and v.reasons == ["standard library: no install needed"] and model.calls == []


def test_dedup_order_band_and_json(tmp_path):
    verdicts = run(tmp_path, ["zzq-ghostlib", "requests", "Zzq_Ghostlib"], band=True)
    assert [v.normalized for v in verdicts] == ["zzq-ghostlib", "requests"]
    d = verdicts[0].to_dict()
    json.dumps(d)  # JSON-safe
    assert d["verdict"] == "GHOST" and d["band"] == [0.05, 0.2]
    assert set(d) >= {"name", "normalized", "verdict", "p_ghost", "band", "reasons", "nearest",
                      "nearest_lev", "registry", "checked_online"}


def test_exit_code_precedence():
    def v(kind):
        return guard.Verdict(kind, kind, kind, None, None, [], None, None, None, False)
    assert guard.exit_code([]) == 0
    assert guard.exit_code([v("OK"), v("OK_UNVERIFIED")]) == 0
    assert guard.exit_code([v("OK"), v("SUSPECT")]) == 1
    assert guard.exit_code([v("SUSPECT"), v("GHOST_LIKELY")]) == 2


@pytest.mark.parametrize("cmd,names", [
    ("pip install requests", ["requests"]),
    ("python -m pip install -U 'requests[socks]>=2.31' numpy==2.1", ["requests", "numpy"]),
    ("py -m pip install --index-url https://example.org/simple zzq-ghostlib", ["zzq-ghostlib"]),
    ("uv pip install Zzq_Ghostlib", ["zzq-ghostlib"]),
    ("uv add pandas --dev", ["pandas"]),
    ("cd proj && pip3 install requests; pip install numpy", ["requests", "numpy"]),
    ("pip install git+https://github.com/x/y.git ./local requests", ["requests"]),
    ("pip install -e . requests", ["requests"]),
    ("ls -la && echo done", []),
    ("pip install \\\n  requests \\\n  numpy", ["requests", "numpy"]),
    ("pip install requests requests", ["requests"]),
])
def test_names_from_command(cmd, names):
    assert guard.names_from_command(cmd) == names


def test_names_from_command_reads_requirement_files(tmp_path):
    (tmp_path / "req.txt").write_text("requests\nzzq-ghostlib==1.0\n", encoding="utf-8")
    assert guard.names_from_command("pip install -r req.txt numpy", cwd=tmp_path) == [
        "numpy", "requests", "zzq-ghostlib"]
    assert guard.names_from_command("pip install -r missing.txt", cwd=tmp_path) == []


def test_names_from_requirements_file(tmp_path):
    (tmp_path / "base.txt").write_text("numpy>=2  # pinned\n", encoding="utf-8")
    req = tmp_path / "requirements-dev.txt"
    req.write_text(
        "# comment\n-r base.txt\n--index-url https://example.org/simple\nrequests[socks]==2.31 ; python_version>'3.8'\n"
        "-e .\nzzq-ghostlib \\\n  --hash=sha256:abc\nother @ git+https://github.com/x/other.git\n"
        "git+https://github.com/x/y.git\n\n", encoding="utf-8")
    assert guard.names_from_file(req) == ["numpy", "requests", "zzq-ghostlib"]


def test_names_from_pyproject(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[project]\nname = "demo"\ndependencies = ["requests>=2", "Zzq_Ghostlib"]\n'
        '[project.optional-dependencies]\nextra = ["numpy"]\n'
        '[dependency-groups]\ndev = ["pytest>=8", {include-group = "lint"}]\nlint = ["ruff"]\n'
        '[tool.uv]\ndev-dependencies = ["pandas"]\n', encoding="utf-8")
    assert guard.names_from_file(p) == ["requests", "zzq-ghostlib", "numpy", "pytest", "ruff", "pandas"]


def test_names_from_file_errors(tmp_path):
    with pytest.raises(OSError):
        guard.names_from_file(tmp_path / "nope.txt")
    bad = tmp_path / "pyproject.toml"
    bad.write_text("[project\n", encoding="utf-8")
    with pytest.raises(ValueError):
        guard.names_from_file(bad)


def test_import_guard_does_not_load_torch():
    import subprocess
    import sys
    code = "import sys, phantom.guard, phantom.cli; print('torch' in sys.modules, 'tabpfn' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
