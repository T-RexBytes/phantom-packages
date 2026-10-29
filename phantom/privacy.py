"""Anonymised ghost ids and the public-file leak scan (CLAUDE.md rule 3, BUILD_SPEC §12).

Unregistered ghost names never leave ``private/``. Public files may carry only
``anon_id(name)`` = ``HMAC-SHA256(key, normalize(name))[:8]`` with the secret
``PHANTOM_ANON_KEY`` (``.env`` locally, a GitHub secret in CI). A plain hash would let anyone
who guesses a name, or reruns the same open models, confirm it is on the list; a keyed
hash cannot be checked without the key. Stage 06 writes the ids of every currently
unregistered ghost to ``results/anon_ghost_ids.txt`` (committed), so CI can check public
files without ever seeing a name: every name-like token is hashed with the key and compared
with that set. Reports carry file, line and id, never the token itself.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterable
from pathlib import Path

from phantom import config
from phantom.extract import normalize

ANON_IDS_PATH = config.RESULTS / "anon_ghost_ids.txt"
ANON_KEY_ENV = "PHANTOM_ANON_KEY"
PUBLIC_GLOBS = [
    "results/**/*.json", "results/**/*.csv", "results/**/*.txt", "results/**/*.md",
    "web/**/*.html", "web/**/*.json", "web/**/*.js", "web/**/*.md",
    "README.md", "POST_DRAFT.md", "docs/blog/**/*.md",
]
_HEADER = (
    "# Anonymised ids of ghost names that were unregistered on PyPI at the last check.\n"
    "# Each line is HMAC-SHA256(PHANTOM_ANON_KEY, normalized name)[:8]: keyed ids, not names\n"
    "# (CLAUDE.md rule 3). Without the key a guessed name cannot be checked against this list.\n"
    "# tests/test_public_leak.py fails if any public file contains a name with one of these ids.\n"
)
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


class MissingAnonKey(RuntimeError):
    """``PHANTOM_ANON_KEY`` is not set, so no public id can be computed."""


def anon_key() -> bytes | None:
    """The secret key from the environment (``.env`` loaded first), or None."""
    config.load_dotenv()
    value = os.environ.get(ANON_KEY_ENV, "").strip()
    return value.encode("utf-8") if value else None


def anon_id(name: str, key: bytes | None = None) -> str:
    """Public id of a name: first 8 hex chars of HMAC-SHA256(key, PEP 503 form)."""
    key = key or anon_key()
    if not key:
        raise MissingAnonKey(f"set {ANON_KEY_ENV} in .env (any long random string) to compute public ids")
    return hmac.new(key, normalize(name).encode("utf-8"), hashlib.sha256).hexdigest()[:8]


def write_anon_ids(names: Iterable[str], path: Path = ANON_IDS_PATH, key: bytes | None = None) -> int:
    """Write the sorted unique ids of ``names`` with an explanatory header; return the count."""
    ids = sorted({anon_id(n, key) for n in names})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "".join(f"{i}\n" for i in ids), encoding="utf-8")
    return len(ids)


def load_anon_ids(path: Path = ANON_IDS_PATH) -> set[str]:
    """Ids from ``path``, ignoring blank and ``#`` comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {s for s in (line.strip() for line in lines) if s and not s.startswith("#")}


def candidate_tokens(text: str) -> set[str]:
    """Normalised name-like tokens in ``text``.

    A token is a run of letters, digits and inner ``._-`` (so backticks, quotes, spaces
    and URL slashes all split). Dotted tokens also contribute their parts, so
    ``name.tar.gz`` or ``name.json`` still yields ``name``.
    """
    out: set[str] = set()
    for tok in _TOKEN.findall(text):
        out.add(normalize(tok))
        if "." in tok:
            out.update(normalize(p) for p in tok.split(".") if p)
    return out


def public_files(root: Path, globs: Iterable[str] = PUBLIC_GLOBS) -> list[Path]:
    """Public text files under ``root`` (the id list itself excluded)."""
    skip = (root / ANON_IDS_PATH.relative_to(config.ROOT)).resolve()
    files = {p for g in globs for p in root.glob(g) if p.is_file()}
    return sorted(p for p in files if p.resolve() != skip)


def find_leaks(root: Path, ids: set[str], globs: Iterable[str] = PUBLIC_GLOBS,
               key: bytes | None = None) -> list[tuple[str, str]]:
    """``("relative/path:line", anon_id)`` for every name-like token whose id is in ``ids``.

    The matching token is never returned, so the result is safe to print in a public CI log.
    """
    key = key or anon_key()
    leaks: list[tuple[str, str]] = []
    for path in public_files(root, globs):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            for i in sorted({anon_id(t, key) for t in candidate_tokens(line)} & ids):
                leaks.append((f"{path.relative_to(root).as_posix()}:{lineno}", i))
    return leaks
