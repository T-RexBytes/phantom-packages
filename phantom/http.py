"""Cached, rate-limited, metadata-only HTTP client for PyPI and pypistats.

Only JSON ``GET`` requests to hosts in ``config.ALLOWED_HOSTS`` are possible,
so package artifacts (``files.pythonhosted.org``) can never be downloaded.
Successful responses are cached as ``{namespace}/{key}.json``; 404s are cached
as an empty ``{namespace}/{key}.404`` marker.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from phantom import config

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class MetadataError(RuntimeError):
    """Raised when a metadata request fails after all retries."""


@dataclass(frozen=True)
class CachedResponse:
    """Outcome of a metadata lookup: HTTP status, parsed JSON (or None), cache hit."""

    status: int
    data: Any | None
    from_cache: bool


def check_url(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` is HTTPS on an allowed metadata host."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in config.ALLOWED_HOSTS:
        raise ValueError(f"refusing non-metadata URL: {url}")


def cache_key(key: str) -> str:
    """Return a filesystem-safe cache key (sha256 prefix for unusual strings)."""
    if _SAFE_KEY.match(key) and key not in {".", ".."}:
        return key
    return "sha256-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class RateLimiter:
    """Thread-safe limiter enforcing a minimum interval between calls."""

    def __init__(
        self,
        per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 1.0 / per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        """Block until the next request is allowed."""
        with self._lock:
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self._interval


class MetadataClient:
    """Cached JSON GET client with rate limiting and retry/backoff."""

    def __init__(
        self,
        cache_dir: Path = config.CACHE,
        user_agent: str | None = None,
        per_second: float = config.MAX_REQUESTS_PER_SECOND,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(per_second, sleep=sleep)
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent or config.get_settings().user_agent,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(20.0),
            follow_redirects=False,
            transport=transport,
        )
        self.network_calls = 0

    def __enter__(self) -> MetadataClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _paths(self, namespace: str, key: str) -> tuple[Path, Path]:
        parts = [cache_key(p) for p in namespace.split("/") if p]  # "recheck/pypi" -> recheck/pypi/
        base = self.cache_dir.joinpath(*parts) / cache_key(key)
        return base.with_name(base.name + ".json"), base.with_name(base.name + ".404")

    def get_json(self, url: str, namespace: str, key: str, refresh: bool = False) -> CachedResponse:
        """GET ``url`` as JSON, serving from and writing to the cache.

        Returns status 200 with parsed data, or 404 with ``data=None``.
        Retries 429/5xx and transport errors with exponential backoff
        (honouring ``Retry-After``); raises ``MetadataError`` if they persist.
        """
        check_url(url)
        hit, miss = self._paths(namespace, key)
        if not refresh:
            if hit.exists():
                return CachedResponse(200, json.loads(hit.read_text(encoding="utf-8")), True)
            if miss.exists():
                return CachedResponse(404, None, True)

        response = self._get_with_retry(url)
        hit.parent.mkdir(parents=True, exist_ok=True)
        if response.status_code == 404:
            hit.unlink(missing_ok=True)
            miss.touch()
            return CachedResponse(404, None, False)
        if response.status_code != 200:
            raise MetadataError(f"GET {url} -> HTTP {response.status_code}")
        data = response.json()
        miss.unlink(missing_ok=True)
        hit.write_text(json.dumps(data), encoding="utf-8")
        return CachedResponse(200, data, False)

    def _get_with_retry(self, url: str) -> httpx.Response:
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            self.network_calls += 1
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = _backoff(attempt, None)
            else:
                if response.status_code not in RETRY_STATUSES:
                    return response
                last_error = f"HTTP {response.status_code}"
                delay = _backoff(attempt, response.headers.get("Retry-After"))
            if attempt < self.max_retries:
                self._sleep(delay)
        raise MetadataError(f"GET {url} failed after {self.max_retries + 1} attempts ({last_error})")


def _backoff(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before retry ``attempt + 1``."""
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after), 60.0)
    return min(0.5 * 2**attempt, 30.0)
