"""Top-15k PyPI names (hugovk/top-pypi-packages) and a nearest-real-name index (BUILD_SPEC §7).

The list is fetched once through the metadata client and cached under
data/cache/top/; it is never redistributed (licence unverified, see docs/api_notes).
Rank 1 is the most-downloaded project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from phantom import config
from phantom.extract import normalize
from phantom.http import MetadataClient

TOP_N = 15_000

# BUILD_SPEC §7 affix list; "py"/"python" are prefixes, the rest suffixes.
PREFIXES = ("py", "python", "py-", "python-")
SUFFIXES = ("-py", "-python", "-api", "-sdk", "-client", "-utils", "-tools", "-lib", "-wrapper",
            "-parser", "-reader", "-io", "-cli", "-async", "-lite", "-pro", "-plus", "2", "3")


def fetch_top(client: MetadataClient | None = None) -> dict[str, Any]:
    """Return the cached top-packages JSON document (fetching it on first use)."""
    owns = client is None
    client = client or MetadataClient()
    try:
        return client.get_json(config.TOP_PACKAGES_URL, "top", "top-pypi-packages").data
    finally:
        if owns:
            client.close()


@dataclass
class TopIndex:
    """Normalized top names in popularity order with lookup helpers."""

    names: list[str]

    def __post_init__(self) -> None:
        self.rank = {n: i + 1 for i, n in enumerate(self.names)}

    @classmethod
    def from_document(cls, doc: dict[str, Any], n: int = TOP_N) -> TopIndex:
        """Build from the hugovk JSON (rows already sorted by downloads)."""
        seen: dict[str, None] = {}
        for row in doc["rows"]:
            seen.setdefault(normalize(row["project"]), None)
            if len(seen) == n:
                break
        return cls(list(seen))

    def __contains__(self, name: str) -> bool:
        return name in self.rank

    def nearest(self, name: str) -> tuple[str, int, float, float]:
        """(nearest name, Levenshtein distance, max Jaro-Winkler similarity,
        log10 rank of the nearest name). Ties go to the more popular name."""
        best, dist, _ = process.extractOne(name, self.names, scorer=Levenshtein.distance)
        _, jw, _ = process.extractOne(name, self.names, scorer=JaroWinkler.similarity)
        return best, int(dist), float(jw), math.log10(self.rank[best])

    def affix_match(self, name: str) -> str | None:
        """A top name that ``name`` equals plus or minus one common affix, else None."""
        candidates: list[str] = []
        for p in PREFIXES:
            if name.startswith(p):
                candidates.append(name[len(p):].lstrip("-"))
            candidates.append(p + name)
        for s in SUFFIXES:
            if name.endswith(s):
                candidates.append(name[: -len(s)].rstrip("-"))
            candidates.append(name + s)
        return next((c for c in candidates if c and c != name and c in self.rank), None)


def load_index(client: MetadataClient | None = None) -> TopIndex:
    """Top-15k index from the cached list."""
    return TopIndex.from_document(fetch_top(client))
