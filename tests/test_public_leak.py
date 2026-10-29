"""No private ghost name may appear in a public file (CLAUDE.md rule 3, BUILD_SPEC §12).

Runs only where private/ exists (the author's machine); CI has no private data.
"""

from __future__ import annotations

import csv
import re

import pytest

from phantom import config

PRIVATE_LISTS = [config.PRIVATE / "ghost_counts.csv", config.PRIVATE / "unregistered_repeated.csv"]
PUBLIC_GLOBS = ["results/**/*.json", "results/**/*.csv", "web/**/*.html", "web/**/*.json",
                "README.md", "POST_DRAFT.md", "docs/blog/**/*.md"]


def private_names() -> set[str]:
    names: set[str] = set()
    for path in PRIVATE_LISTS:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                names |= {row["name"] for row in csv.DictReader(f) if row.get("name")}
    return names


def test_no_private_ghost_names_in_public_files():
    names = private_names()
    if not names:
        pytest.skip("no private ghost lists on this machine")
    pattern = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, sorted(names))) + r")(?![\w-])", re.I)
    leaks = []
    for glob in PUBLIC_GLOBS:
        for path in config.ROOT.glob(glob):
            for m in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                leaks.append(f"{path.relative_to(config.ROOT)}: {m.group(0)}")
    assert not leaks, f"private ghost names in public files: {leaks[:10]}"
