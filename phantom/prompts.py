"""Build and load the prompt set (BUILD_SPEC §4.1, §5.1).

The human-readable task lists in ``docs/seed_tasks.md`` and
``docs/seed_variants.md`` are the source of truth; ``data/prompts.jsonl`` is
derived from them deterministically.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from phantom import config

SEED_PATH = config.DOCS / "seed_tasks.md"
VARIANT_PATH = config.DOCS / "seed_variants.md"

# Domain number in the task lists -> stable slug used in data and figures.
DOMAINS: dict[int, str] = {
    1: "data-files",
    2: "web-scraping",
    3: "images-media",
    4: "pdfs-documents",
    5: "lab-instruments",
    6: "bioinformatics",
    7: "geospatial",
    8: "hardware-iot",
    9: "finance-regional",
    10: "calendars-locale",
    11: "legacy-formats",
    12: "saas-apis",
    13: "audio-music",
    14: "games-retro",
    15: "devops-security",
}

_HEADING = re.compile(r"^##\s+(\d+)\.\s+\S")
_TASK = re.compile(r"^-\s+\[L([123])\]\s+(.+?)\s*$")


@dataclass(frozen=True)
class Task:
    """One task line from a task list."""

    domain: str
    niche_level: int
    task: str
    origin: str  # "seed" or "variant"


def parse_tasks(path: Path, origin: str) -> list[Task]:
    """Parse ``## N. Domain`` headings and ``- [L#] task`` lines from a markdown list."""
    tasks: list[Task] = []
    domain: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if m := _HEADING.match(line):
            domain = DOMAINS[int(m.group(1))]
        elif m := _TASK.match(line):
            if domain is None:
                raise ValueError(f"{path.name}:{lineno}: task before any domain heading")
            tasks.append(Task(domain, int(m.group(1)), m.group(2).rstrip("."), origin))
    return tasks


def render(task: str) -> str:
    """Fill the fixed user-prompt template with one task."""
    return config.PROMPT_TEMPLATE.format(task=task)


def build_prompts(tasks: list[Task]) -> list[dict]:
    """Order tasks by domain (seeds before variants) and assign ids ``p001``…

    Raises ``ValueError`` on duplicate task text.
    """
    dupes = [t for t, n in Counter(t.task.lower() for t in tasks).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate tasks: {dupes}")
    order = {slug: i for i, slug in enumerate(DOMAINS.values())}
    ordered = sorted(
        enumerate(tasks), key=lambda it: (order[it[1].domain], it[1].origin != "seed", it[0])
    )
    width = max(3, len(str(len(tasks))))
    return [
        {
            "id": f"p{i:0{width}d}",
            "domain": t.domain,
            "niche_level": t.niche_level,
            "text": render(t.task),
            "task": t.task,
            "origin": t.origin,
        }
        for i, (_, t) in enumerate(ordered, 1)
    ]


def default_prompts() -> list[dict]:
    """Build the full prompt set from the committed seed and variant lists."""
    return build_prompts(parse_tasks(SEED_PATH, "seed") + parse_tasks(VARIANT_PATH, "variant"))


def write_jsonl(prompts: list[dict], path: Path = config.PROMPTS_PATH) -> None:
    """Write prompts as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_prompts(path: Path = config.PROMPTS_PATH) -> list[dict]:
    """Read prompts from JSON Lines."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
