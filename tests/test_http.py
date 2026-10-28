"""Tests for the cached, rate-limited metadata client (no real network)."""

from __future__ import annotations

import httpx
import pytest

from phantom.http import MetadataClient, MetadataError, RateLimiter, cache_key, check_url

URL = "https://pypi.org/pypi/{}/json"


def make_client(tmp_path, handler, max_retries=3):
    sleeps: list[float] = []
    client = MetadataClient(
        cache_dir=tmp_path,
        user_agent="phantom-tests/0",
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        sleep=sleeps.append,
    )
    return client, sleeps


def test_200_is_parsed_and_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"info": {"name": "requests"}})

    client, _ = make_client(tmp_path, handler)
    first = client.get_json(URL.format("requests"), "pypi", "requests")
    second = client.get_json(URL.format("requests"), "pypi", "requests")
    assert first.status == 200 and not first.from_cache
    assert second.from_cache and second.data == {"info": {"name": "requests"}}
    assert len(calls) == 1
    assert (tmp_path / "pypi" / "requests.json").exists()
    assert calls[0].headers["User-Agent"] == "phantom-tests/0"


def test_404_is_cached_as_marker(tmp_path):
    client, _ = make_client(tmp_path, lambda r: httpx.Response(404))
    res = client.get_json(URL.format("nope"), "pypi", "nope")
    assert res.status == 404 and res.data is None
    assert (tmp_path / "pypi" / "nope.404").exists()
    again = client.get_json(URL.format("nope"), "pypi", "nope")
    assert again.from_cache and client.network_calls == 1


def test_429_then_200_retries_with_retry_after(tmp_path):
    responses = iter([httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json={})])
    client, sleeps = make_client(tmp_path, lambda r: next(responses))
    res = client.get_json(URL.format("x"), "pypi", "x")
    assert res.status == 200
    assert 3.0 in sleeps
    assert client.network_calls == 2


def test_persistent_5xx_raises_and_is_not_cached(tmp_path):
    client, _ = make_client(tmp_path, lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(MetadataError):
        client.get_json(URL.format("x"), "pypi", "x")
    assert client.network_calls == 3
    assert not list(tmp_path.rglob("x.*"))


def test_refresh_overwrites_stale_404(tmp_path):
    state = {"code": 404}
    client, _ = make_client(tmp_path, lambda r: httpx.Response(state["code"], json={}))
    client.get_json(URL.format("late"), "pypi", "late")
    state["code"] = 200
    res = client.get_json(URL.format("late"), "pypi", "late", refresh=True)
    assert res.status == 200 and not res.from_cache
    assert not (tmp_path / "pypi" / "late.404").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://files.pythonhosted.org/packages/x/requests.whl",
        "http://pypi.org/pypi/requests/json",
        "https://evil.example/pypi/requests/json",
    ],
)
def test_non_metadata_urls_are_refused(url):
    with pytest.raises(ValueError):
        check_url(url)


def test_cache_key_hashes_unsafe_strings():
    assert cache_key("requests") == "requests"
    assert cache_key("../etc/passwd").startswith("sha256-")
    assert cache_key("..").startswith("sha256-")


def test_rate_limiter_spaces_calls():
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    limiter = RateLimiter(5.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert sleeps == pytest.approx([0.2, 0.2])
