"""PyPI and pypistats metadata lookups (BUILD_SPEC §4.4, §5.3). Metadata GETs only.

``first_release`` is the earliest ``upload_time_iso_8601`` over every file of
every version; ``n_releases`` counts versions that have at least one file.
Monthly downloads (pypistats, ~1 req/s) are fetched only for squat candidates:
existing names that are young or have few releases (the only place they are used).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def is_squat_candidate(row: dict[str, Any], now: datetime | None = None) -> bool:
    """Existing name with few releases or a first release in the squat age window."""
    if not row["exists"]:
        return False
    now = now or datetime.now(timezone.utc)
    young = row["first_release"] is not None and (
        now - row["first_release"] <= timedelta(days=30.44 * config.SQUAT_MAX_AGE_MONTHS)
    )
    few = row["n_releases"] is not None and row["n_releases"] <= config.SQUAT_MAX_RELEASES
    return young or few


def recent_downloads(name: str, client: MetadataClient) -> int | None:
    """``last_month`` downloads from pypistats ``/recent``; None if pypistats has no data."""
    res = client.get_json(config.PYPISTATS_RECENT_URL.format(name=name), "pypistats", name)
    if res.status == 404:
        return None
    return int(res.data["data"]["last_month"])


def lookup(
    name: str, pypi: MetadataClient, stats: MetadataClient | None = None, refresh: bool = False
) -> dict[str, Any]:
    """Registry row for one name; downloads added for squat candidates if ``stats`` given."""
    n = normalize(name)
    res = pypi.get_json(config.PYPI_JSON_URL.format(name=n), "pypi", n, refresh=refresh)
    row = parse_pypi(n, res.data if res.status == 200 else None)
    if stats is not None and is_squat_candidate(row):
        try:
            row["recent_downloads_month"] = recent_downloads(n, stats)
        except MetadataError:
            row["recent_downloads_month"] = None
    return row


def clients() -> tuple[MetadataClient, MetadataClient]:
    """PyPI client (5 req/s) and pypistats client (1 req/s) sharing the cache dir."""
    return MetadataClient(), MetadataClient(per_second=config.PYPISTATS_REQUESTS_PER_SECOND)
