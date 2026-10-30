"""No unregistered ghost name may appear in a public file (CLAUDE.md rule 3, BUILD_SPEC §12).

Two checks over the same public globs (``phantom.privacy.PUBLIC_GLOBS``):
- CI-safe: hash every name-like token with the secret ``PHANTOM_ANON_KEY`` and compare with
  ``results/anon_ghost_ids.txt`` (written by stage 06, committed). Failures name file, line and
  id only. Skips (never passes vacuously) when the key is absent, e.g. CI without the secret.
- Plaintext: only where ``private/`` exists (the author's machine).
"""

from __future__ import annotations

import csv

import pytest

from phantom import config, privacy

PRIVATE_LISTS = [config.PRIVATE / "ghost_counts.csv", config.PRIVATE / "unregistered_repeated.csv",
                 config.PRIVATE / "ghost_counts_hosted.csv"]
SQUAT_FULL = config.PRIVATE / "squat_full.csv"


def test_no_anonymised_ghost_in_public_files():
    if not privacy.ANON_IDS_PATH.exists():
        pytest.skip("results/anon_ghost_ids.txt not written yet (stage 06 creates it)")
    if privacy.anon_key() is None:
        pytest.skip("PHANTOM_ANON_KEY not set (add it as a repository secret for CI)")
    ids = privacy.load_anon_ids()
    leaks = privacy.find_leaks(config.ROOT, ids)
    assert not leaks, "unregistered ghost names in public files (path:line id): " + ", ".join(
        f"{loc} {i}" for loc, i in leaks[:20])


def _falsy(value: str | None) -> bool:
    return str(value).strip().lower() in {"", "0", "false", "no", "none", "nan"}


def private_names() -> set[str]:
    """Plaintext names from the private lists (squat_full: only rows not registered now)."""
    names: set[str] = set()
    for path in PRIVATE_LISTS:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                names |= {row["name"] for row in csv.DictReader(f) if row.get("name")}
    if SQUAT_FULL.exists():
        with SQUAT_FULL.open(encoding="utf-8") as f:
            names |= {row["name"] for row in csv.DictReader(f)
                      if row.get("name") and _falsy(row.get("exists_now"))}
    return names


def test_no_private_ghost_names_in_public_files():
    names = private_names()
    if not names:
        pytest.skip("no private ghost lists on this machine")
    if privacy.anon_key() is None:
        pytest.skip("PHANTOM_ANON_KEY not set")
    leaks = privacy.find_leaks(config.ROOT, {privacy.anon_id(n) for n in names})
    assert not leaks, "private ghost names in public files (path:line id): " + ", ".join(
        f"{loc} {i}" for loc, i in leaks[:20])
