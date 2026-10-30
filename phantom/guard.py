"""The guard's verdict logic (BUILD_SPEC §9.1), shared by the CLI, the hook and the Action.

"The model can dream a package. Only the registry can make it real."
- Online: PyPI decides. 404 -> GHOST. An existing name is SUSPECT if it matches the squat rule
  (§5.6 without the study-only "summoned >= 2 times" check) or if TabPFN scores it >= tau and it
  is young. Otherwise OK.
- Offline, or PyPI unreachable: the TabPFN guard model's P(ghost) against its threshold gives
  GHOST_LIKELY or OK_UNVERIFIED.
- Every reason is a short string built by this module from registry facts and model scores.

Nothing here imports torch or TabPFN at import time: the model is loaded only when a verdict
needs P(ghost), so commands without installs stay fast in the hook.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from phantom import config, extract, registry, squat
from phantom.extract import STDLIB, normalize, requirement_name
from phantom.http import MetadataClient, MetadataError

VERDICTS = ("OK", "OK_UNVERIFIED", "SUSPECT", "GHOST_LIKELY", "GHOST")
PROB_MODES = ("auto", "always", "never", "registry-first")
REGISTRY_NAMESPACE = "guard/"
GUARD_SQUAT_CHECKS = [c for c in squat.CHECKS if c != "repeated"]  # "repeated" needs study data
STDLIB_REASON = "standard library: no install needed"
UNREACHABLE_REASON = "PyPI unreachable: offline estimate"
OFFLINE_REASON = "not checked on PyPI (offline)"

_STDLIB = frozenset(normalize(m) for m in STDLIB)
_REQUIREMENT_FLAG = re.compile(r"(?:^|\s)(?:-r|--requirement)(?:\s+|=)(\"[^\"]+\"|'[^']+'|\S+)")


@dataclass
class Verdict:
    """One name's verdict with the facts behind it."""

    name: str
    normalized: str
    verdict: str
    p_ghost: float | None
    band: tuple[float, float] | None
    reasons: list[str]
    nearest: str | None
    nearest_lev: int | None
    registry: dict[str, Any] | None
    checked_online: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (band as a list, datetimes already ISO strings)."""
        return {
            "name": self.name, "normalized": self.normalized, "verdict": self.verdict,
            "p_ghost": self.p_ghost, "band": list(self.band) if self.band else None,
            "reasons": list(self.reasons), "nearest": self.nearest, "nearest_lev": self.nearest_lev,
            "registry": self.registry, "checked_online": self.checked_online,
        }


# ---------------------------------------------------------------- names from commands and files

def names_from_command(cmd: str, cwd: Path | None = None) -> list[str]:
    """Normalized names a shell command would install from PyPI (pip, python -m pip, py -m pip,
    uv pip install, uv add). Locations (URLs, git refs, paths) are skipped; ``-r FILE`` is read
    relative to ``cwd`` when the file exists."""
    names: list[str] = []
    files: list[str] = []
    for line in extract._join_continuations(cmd.splitlines()):
        for _source, _cmd, kept, _dropped in extract._command_names(line):
            names += [normalize(name) for _raw, name in kept]
        if extract._COMMAND.search(line):
            files += [m.group(1).strip("'\"") for m in _REQUIREMENT_FLAG.finditer(line)]
    base = Path(cwd) if cwd else Path.cwd()
    for f in files:
        path = base / f
        if path.is_file():
            names += names_from_file(path)
    return list(dict.fromkeys(names))


def _requirement(spec: str) -> str | None:
    """Normalized name of a PEP 508 requirement, or None for direct references and junk."""
    spec = spec.strip()
    if not spec or spec.startswith(("-", "#")) or extract._is_location(spec.split()[0]):
        return None
    if re.search(r"\s@\s|^[^@\s]+@", spec):  # "name @ url": not installed from PyPI
        return None
    name = requirement_name(spec)
    return normalize(name) if name and extract._VALID.match(name) else None


def _requirements_txt(path: Path, seen: set[Path]) -> list[str]:
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    names: list[str] = []
    lines = extract._join_continuations(path.read_text(encoding="utf-8").splitlines())
    for line in lines:
        s = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].strip()
        if m := re.match(r"^(?:-r|--requirement)(?:\s+|=)(\S+)", s):
            names += _requirements_txt(path.parent / m.group(1), seen)
            continue
        if n := _requirement(s):
            names.append(n)
    return names


def _pyproject(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: list[Any] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs += group
    for group in data.get("dependency-groups", {}).values():
        specs += group
    specs += data.get("tool", {}).get("uv", {}).get("dev-dependencies", [])
    return [n for s in specs if isinstance(s, str) and (n := _requirement(s))]


def names_from_file(path: Path) -> list[str]:
    """Normalized names in a ``requirements*.txt`` (following ``-r`` includes) or a
    ``pyproject.toml`` (project, optional, dependency-groups and ``[tool.uv]`` dev deps).
    Raises ``OSError`` if unreadable and ``ValueError`` if a TOML file is malformed."""
    path = Path(path)
    names = _pyproject(path) if path.name == "pyproject.toml" else _requirements_txt(path, set())
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------- verdicts

@lru_cache(maxsize=1)
def _default_index():
    from phantom import topnames

    return topnames.load_index()


def _nearest(n: str, index) -> tuple[str | None, int | None]:
    if index is None or n in index:
        return None, None
    affix = index.affix_match(n)  # "python-foo" -> "foo" beats a closer but unrelated edit-distance match
    if affix is not None:
        return affix, int(Levenshtein.distance(n, affix))
    near, lev, _jw, _rank = index.nearest(n)
    return near, int(lev)


def _guard_clients() -> tuple[MetadataClient, MetadataClient]:
    """Clients that give up fast (one retry) so an offline hook doesn't hang."""
    return (MetadataClient(max_retries=1),
            MetadataClient(per_second=config.PYPISTATS_REQUESTS_PER_SECOND, max_retries=1))


def _registry_dict(row: dict[str, Any]) -> dict[str, Any]:
    def iso(v):
        return v.isoformat() if isinstance(v, datetime) else v
    return {"exists": row["exists"], "n_releases": row["n_releases"], "first_release": iso(row["first_release"]),
            "last_release": iso(row["last_release"]), "recent_downloads_month": row["recent_downloads_month"],
            "summary": row["summary"]}


def _p_reason(p: float, tau: float) -> str:
    return f"P(ghost)={p:.2f} {'>=' if p >= tau else '<'} tau {tau:.2f}"


def _near_reason(near: str | None, lev: int | None) -> list[str]:
    return [f"nearest real name: {near} (edit distance {lev})"] if near else []


def check(names: Iterable[str], offline: bool = False, band: bool = False, prob: str = "auto",
          model: Any = None, clients: tuple[MetadataClient, MetadataClient] | None = None,
          index: Any = None, now: datetime | None = None) -> list[Verdict]:
    """Verdicts for unique normalized ``names`` in the order given.

    ``prob``: "auto" computes P(ghost) only where it matters (offline, PyPI unreachable, 404s,
    young existing names); "registry-first" is "auto" minus 404s, i.e. only where P(ghost) can change
    the verdict (the hook's default: a ghost is blocked without loading torch/TabPFN); "always" for
    every non-stdlib name; "never" skips the model.
    ``model`` (default ``phantom.model.GuardModel``), ``clients``, ``index`` and ``now`` are
    injectable for tests.
    """
    if prob not in PROB_MODES:
        raise ValueError(f"prob must be one of {PROB_MODES}")
    given: dict[str, str] = {}
    for name in names:
        given.setdefault(normalize(name), name)
    if index is None:
        try:
            index = _default_index()
        except MetadataError:
            index = None  # top list not cached and no network: no nearest-name hints

    rows: dict[str, dict[str, Any] | None] = {}  # None = not checked (offline or unreachable)
    candidates = [n for n in given if n not in _STDLIB]
    if not offline and candidates:
        own = clients is None
        pypi, stats = clients or _guard_clients()
        try:
            for n in candidates:
                try:
                    row = registry.lookup(n, pypi, None, refresh=True, namespace=REGISTRY_NAMESPACE)
                except MetadataError:
                    rows[n] = None
                    continue
                c = squat.checks(row, 0, now)
                row["downloads_checked"] = bool(c["few_releases"] and c["young"])
                if row["downloads_checked"]:  # downloads only matter for the squat rule
                    try:
                        row["recent_downloads_month"] = registry.recent_downloads(n, stats, REGISTRY_NAMESPACE)
                    except MetadataError:
                        pass  # unknown downloads count as low (conservative)
                rows[n] = row
        finally:
            if own:
                pypi.close()
                stats.close()

    def needs_p(n: str) -> bool:
        if prob == "never" or n in _STDLIB:
            return False
        if prob == "always":
            return True
        row = rows.get(n)
        if prob == "registry-first":  # model only where it can change the verdict (the hook's default)
            return row is None or (row["exists"] and squat.checks(row, 0, now)["young"])
        return row is None or not row["exists"] or squat.checks(row, 0, now)["young"]

    wanted = [n for n in given if needs_p(n)]
    scores: dict[str, tuple[float, float | None, float | None]] = {}
    tau = None
    model_note: list[str] = []
    if wanted:
        try:
            if model is None:
                from phantom.model import GuardModel  # lazy: torch/TabPFN only when needed

                model = GuardModel(band=band)
            scores = model.predict(wanted)
            tau = model.tau
        except Exception as exc:  # no `model` extra (CI, the Action) or weights unavailable: registry still decides
            scores, model_note = {}, [f"P(ghost) unavailable ({type(exc).__name__}): install the `model` extra"]

    out = []
    for n, original in given.items():
        if n in _STDLIB:
            out.append(Verdict(original, n, "OK", None, None, [STDLIB_REASON], None, None, None, False))
            continue
        near, lev = _nearest(n, index)
        p, lo, hi = scores.get(n, (None, None, None))
        b = (lo, hi) if lo is not None and hi is not None else None
        p_reasons = [_p_reason(p, tau)] if p is not None else (model_note if n in wanted else [])
        row = rows.get(n)
        if row is None:
            kind = "GHOST_LIKELY" if p is not None and p >= tau else "OK_UNVERIFIED"
            why = [OFFLINE_REASON if offline else UNREACHABLE_REASON, *p_reasons, *_near_reason(near, lev)]
            out.append(Verdict(original, n, kind, p, b, why, near, lev, None, False))
            continue
        if not row["exists"]:
            why = ["404 on PyPI", *p_reasons, *_near_reason(near, lev)]
            out.append(Verdict(original, n, "GHOST", p, b, why, near, lev, _registry_dict(row), True))
            continue
        c = squat.checks(row, 0, now)
        squat_hit = all(c[k] for k in GUARD_SQUAT_CHECKS)
        model_hit = p is not None and p >= tau and c["young"]
        facts = squat.reasons(row, c, 0, now)[:-1]  # drop the study-only "summoned by" line
        if not row.get("downloads_checked", True):
            facts = [f for f in facts if f != "no download data"]  # not looked up: it can't change the verdict
        if squat_hit or model_hit:
            why = (["matches the squat rule (few releases, young, low downloads)"] if squat_hit else []) + \
                  (["young and ghost-like name"] if model_hit else []) + facts + p_reasons + _near_reason(near, lev)
            out.append(Verdict(original, n, "SUSPECT", p, b, why, near, lev, _registry_dict(row), True))
        else:
            why = ["exists on PyPI", *(["in the top-15k list"] if index is not None and n in index else []),
                   *facts, *p_reasons]
            out.append(Verdict(original, n, "OK", p, b, why, near, lev, _registry_dict(row), True))
    return out


@lru_cache(maxsize=1)
def _default_lm():
    from phantom.charlm import CharTrigramLM

    return CharTrigramLM(_default_index().names)


def describe(v: Verdict, index: Any = None, lm: Any = None, now: datetime | None = None) -> dict[str, str]:
    """Labelled facts for ``phantom why``: registry, nearest name, affix, char-LM z-score,
    P(ghost) with band and the squat checks. Strings only, built from code, never from an LLM."""
    try:
        index = index if index is not None else _default_index()
        lm = lm if lm is not None else _default_lm()
    except MetadataError:
        index = lm = None
    card: dict[str, str] = {}
    if v.registry is None:
        card["registry"] = "not checked (offline)" if not v.checked_online else "unknown"
    elif not v.registry["exists"]:
        card["registry"] = "404 on PyPI"
    else:
        r = v.registry
        card["registry"] = f"exists: {r['n_releases']} release(s), first {str(r['first_release'])[:10]}, " \
                           f"last {str(r['last_release'])[:10]}"
        card["downloads/month"] = "-" if r["recent_downloads_month"] is None else f"{r['recent_downloads_month']:,}"
        card["summary"] = r["summary"] or "-"
    if index is not None:
        card["in top-15k"] = "yes" if v.normalized in index else "no"
        card["nearest real name"] = f"{v.nearest} (edit distance {v.nearest_lev})" if v.nearest else "-"
        card["affix of a real name"] = index.affix_match(v.normalized) or "no"
    if lm is not None:
        card["char-LM z-score"] = f"{lm.zscore(v.normalized):+.2f}"
    if v.p_ghost is not None:
        band = f" (fold band {v.band[0]:.2f}-{v.band[1]:.2f})" if v.band else ""
        card["P(ghost)"] = f"{v.p_ghost:.3f}{band}"
    if v.registry is not None and v.registry["exists"]:
        c = squat.checks(v.registry, 0, now)
        card["squat checks"] = ", ".join(f"{k}={'yes' if c[k] else 'no'}" for k in GUARD_SQUAT_CHECKS)
    return card


def exit_code(verdicts: Iterable[Verdict]) -> int:
    """2 if any GHOST or GHOST_LIKELY, 1 if any SUSPECT, else 0 (§9.2)."""
    kinds = {v.verdict for v in verdicts}
    if kinds & {"GHOST", "GHOST_LIKELY"}:
        return 2
    return 1 if "SUSPECT" in kinds else 0
