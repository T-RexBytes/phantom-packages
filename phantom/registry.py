"""PyPI and pypistats metadata lookups (BUILD_SPEC §4.4, §5.3). Metadata GETs only.

``first_release`` is the earliest ``upload_time_iso_8601`` over every file of
every version; ``n_releases`` counts versions that have at least one file.
Monthly downloads (pypistats ``last_month``, ~1 req/s) are fetched for every
existing name, as in the §4.4 schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from phantom import config
from phantom.extract import normalize
from phantom.http import MetadataClient, MetadataError
from phantom.runconfig import utc_now


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_pypi(name: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a PyPI JSON document (or None for a 404) into a registry row."""
    row: dict[str, Any] = {
        "name": name, "exists": data is not None, "first_release": None, "last_release": None,
        "n_releases": None, "summary": None, "has_project_urls": None,
        "recent_downloads_month": None, "checked_at": utc_now(),
    }
    if data is None:
        return row
    info = data.get("info") or {}
    releases = data.get("releases")
    if releases is None:  # deprecated field; fall back to the current version's files
        releases = {info.get("version", "?"): data.get("urls") or []}
    times = [_ts(f["upload_time_iso_8601"]) for files in releases.values() for f in files
             if f.get("upload_time_iso_8601")]
    row.update(
        first_release=min(times) if times else None,
        last_release=max(times) if times else None,
        n_releases=sum(1 for files in releases.values() if files),
        summary=info.get("summary") or None,
        has_project_urls=bool(info.get("project_urls")),
    )
    return row


def recent_downloads(name: str, client: MetadataClient, namespace: str = "") -> int | None:
    """``last_month`` downloads from pypistats ``/recent``; None if pypistats has no data.

    ``namespace`` prefixes the cache namespace (e.g. ``"recheck/"`` keeps a re-check apart).
    """
    res = client.get_json(config.PYPISTATS_RECENT_URL.format(name=name), f"{namespace}pypistats", name)
    if res.status == 404:
        return None
    return int(res.data["data"]["last_month"])


def lookup(
    name: str, pypi: MetadataClient, stats: MetadataClient | None = None, refresh: bool = False,
    namespace: str = "",
) -> dict[str, Any]:
    """Registry row for one name; monthly downloads added for existing names if ``stats`` given.

    ``namespace`` prefixes the cache namespaces (``"recheck/"`` -> ``recheck/pypi``,
    ``recheck/pypistats``) so a later re-check never overwrites the labelling-time cache.
    """
    n = normalize(name)
    res = pypi.get_json(config.PYPI_JSON_URL.format(name=n), f"{namespace}pypi", n, refresh=refresh)
    row = parse_pypi(n, res.data if res.status == 200 else None)
    if stats is not None and row["exists"]:
        try:
            row["recent_downloads_month"] = recent_downloads(n, stats, namespace)
        except MetadataError:
            row["recent_downloads_month"] = None
    return row


def clients() -> tuple[MetadataClient, MetadataClient]:
    """PyPI client (5 req/s) and pypistats client (1 req/s) sharing the cache dir."""
    return MetadataClient(), MetadataClient(per_second=config.PYPISTATS_REQUESTS_PER_SECOND)
