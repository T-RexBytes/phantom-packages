"""Deterministic taxonomy of ghost names: what kind of invention is each one?

Every ghost (a name PyPI returned 404 for) gets exactly one type; rules run in this order and
the first match wins. "Real" means a top-15k PyPI name or a name that existed on PyPI in the
study's own registry lookups.

1. ``affix_of_real``    a real name plus or minus one common affix (``python-``, ``py``, ``-py``,
                        ``-api``, ``-sdk``, ``-parser``, ``2`` ...; ``TopIndex.affix_match``).
2. ``import_or_brand``  the bare brand of a real package family: an import name in
                        ``extract.IMPORT_TO_DIST`` or the first ``-`` segment of a real
                        multi-part name (``brand`` for a real ``brand-python-lib``).
3. ``task_words``       built from the task: after dropping generic words (``python``, ``py``,
                        ``lib``, ``parser``, ``tools`` ...), every remaining part of the name is a
                        word of the prompt, two adjacent words glued (``as400`` from "as/400"), or
                        an abbreviation of one (a part of 3+ chars that starts a prompt word).
4. ``near_real_name``   one or two edits from a real name: distance 1 for names of 5-6 chars,
                        up to 2 from 7 chars on (shorter names collide by chance). This can be a
                        typo of that package or a coincidence; the rule cannot tell which.
5. ``invented``         none of the above.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from phantom.extract import IMPORT_TO_DIST, normalize
from phantom.topnames import TopIndex

TYPES = ["affix_of_real", "import_or_brand", "task_words", "near_real_name", "invented"]
GENERIC = frozenset({
    "python", "py", "lib", "library", "parser", "parse", "reader", "read", "tools", "tool", "toolkit",
    "utils", "api", "sdk", "client", "decoder", "codec", "converter", "validator", "checker", "file",
    "data", "loader", "editor", "text", "txt", "to", "2",
})
_SPLIT = re.compile(r"[-_.]+")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GhostType:
    """The type of one ghost name and the deterministic evidence for it."""

    name: str
    type: str
    evidence: str  # short, rule-specific: the real name or the matched task words


def _core_parts(name: str) -> list[str]:
    """Name parts with generic words removed; ``py`` prefixes and generic suffixes peeled off
    unseparated parts (``djvuparser`` -> ``djvu``, ``pyjalali`` -> ``jalali``)."""
    out = []
    for part in _SPLIT.split(name):
        if not part or part in GENERIC:
            continue
        for g in sorted(GENERIC, key=len, reverse=True):
            if len(g) > 1 and part.endswith(g) and len(part) - len(g) >= 2:
                part = part[: -len(g)]
                break
        if part.startswith("py") and len(part) >= 5:
            part = part[2:]
        out.append(part)
    return out


def task_vocabulary(text: str) -> set[str]:
    """Lower-case words of the task text plus every pair of adjacent words glued together."""
    words = _WORD.findall(text.lower())
    return set(words) | {a + b for a, b in zip(words, words[1:])}


def from_task(part: str, vocab: set[str]) -> bool:
    """A name part is a task word, two glued task words, or an abbreviation (3+ chars) of one."""
    return part in vocab or (len(part) >= 3 and any(w.startswith(part) for w in vocab))


class Classifier:
    """Assigns ``TYPES`` to ghost names given the set of real names."""

    def __init__(self, real_names: Iterable[str]):
        self.real = sorted({normalize(n) for n in real_names})
        self.index = TopIndex(self.real)
        brands: dict[str, str] = {}
        for n in self.real:
            head = n.split("-")[0]
            if "-" in n and len(head) >= 3:
                brands.setdefault(head, n)
        self.brands = brands
        self.imports = {normalize(k): v for k, v in IMPORT_TO_DIST.items()}

    def classify(self, name: str, task_text: str) -> GhostType:
        """The first matching type, with its evidence."""
        n = normalize(name)
        real = self.index.affix_match(n)
        if real:
            return GhostType(n, "affix_of_real", real)
        if n in self.imports:
            return GhostType(n, "import_or_brand", self.imports[n])
        if n in self.brands:
            return GhostType(n, "import_or_brand", self.brands[n])
        core, vocab = _core_parts(n), task_vocabulary(task_text)
        if core and all(from_task(p, vocab) for p in core):
            return GhostType(n, "task_words", "+".join(core))
        near, dist, _ = process.extractOne(n, self.real, scorer=Levenshtein.distance)
        if (len(n) >= 7 and dist <= 2) or (len(n) >= 5 and dist <= 1):
            return GhostType(n, "near_real_name", f"{near} (edit distance {dist})")
        return GhostType(n, "invented", "")
