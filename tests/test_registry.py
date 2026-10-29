"""Registry tests with mocked httpx: 200 / 404 / 429-retry / cache hit / downloads."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from phantom import registry
from phantom.http import MetadataClient


def pypi_doc(releases):
    return {"info": {"name": "Demo", "summary": "demo pkg", "project_urls": {"Home": "x"}, "version": "2.0"},
            "releases": releases}


def file(ts):
    return {"upload_time_iso_8601": ts, "filename": "x"}


OLD = pypi_doc({
    "1.0": [file("2015-01-02T03:04:05.000000Z"), file("2015-01-01T00:00:00.000000Z")],
    "1.1": [],  # version with no files is not counted
    "2.0": [file("2020-06-01T00:00:00Z")],
})


def client(tmp_path, handler, **kw):
    return MetadataClient(cache_dir=tmp_path, user_agent="t", transport=httpx.MockTransport(handler),
                          sleep=lambda s: None, **kw)


def test_parse_200_computes_release_facts():
    row = registry.parse_pypi("demo", OLD)
    assert row["exists"] is True
    assert row["first_release"] == datetime(2015, 1, 1, tzinfo=timezone.utc)
    assert row["last_release"] == datetime(2020, 6, 1, tzinfo=timezone.utc)
    assert row["n_releases"] == 2
    assert row["summary"] == "demo pkg" and row["has_project_urls"] is True


def test_parse_without_deprecated_releases_field_uses_urls():
    doc = {"info": {"version": "0.1"}, "urls": [file("2026-01-01T00:00:00Z")]}
    row = registry.parse_pypi("demo", doc)
    assert row["n_releases"] == 1 and row["first_release"].year == 2026


def test_lookup_404_is_ghost_and_skips_pypistats(tmp_path):
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return httpx.Response(404)

    row = registry.lookup("Not_A_Real.Pkg", client(tmp_path, handler), client(tmp_path, handler))
    assert row["name"] == "not-a-real-pkg" and row["exists"] is False
    assert seen == ["pypi.org"]


def test_lookup_retries_429_then_succeeds(tmp_path):
    responses = iter([httpx.Response(429), httpx.Response(200, json=OLD)])
    pypi = client(tmp_path, lambda r: next(responses))
    assert registry.lookup("demo", pypi)["exists"] is True
    assert pypi.network_calls == 2


def test_cache_hit_avoids_network(tmp_path):
    pypi = client(tmp_path, lambda r: httpx.Response(200, json=OLD))
    registry.lookup("demo", pypi)
    registry.lookup("DEMO", pypi)
    assert pypi.network_calls == 1


def test_downloads_fetched_for_existing_names(tmp_path):
    stats_calls = []

    def stats_handler(req):
        stats_calls.append(req.url.path)
        return httpx.Response(200, json={"data": {"last_day": 1, "last_week": 5, "last_month": 42}})

    pypi = client(tmp_path, lambda r: httpx.Response(200, json=OLD))
    stats = client(tmp_path / "s", stats_handler)
    assert registry.lookup("demo", pypi, stats)["recent_downloads_month"] == 42
    assert stats_calls == ["/api/packages/demo/recent"]


def test_recheck_namespace_uses_separate_nested_cache(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.host)
        if req.url.host == "pypi.org":
            return httpx.Response(200, json=OLD)
        return httpx.Response(200, json={"data": {"last_month": 7}})

    pypi, stats = client(tmp_path, handler), client(tmp_path, handler)
    registry.lookup("demo", pypi, stats)
    registry.lookup("demo", pypi, stats, namespace="recheck/")
    assert (tmp_path / "pypi" / "demo.json").exists() and (tmp_path / "pypistats" / "demo.json").exists()
    assert (tmp_path / "recheck" / "pypi" / "demo.json").exists()
    assert (tmp_path / "recheck" / "pypistats" / "demo.json").exists()
    assert len(calls) == 4  # the re-check never reads the labelling-time cache


def test_pypistats_404_gives_none(tmp_path):
    stats = client(tmp_path, lambda r: httpx.Response(404))
    assert registry.recent_downloads("young", stats) is None
