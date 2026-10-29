"""Tests for prompt parsing and the committed 200-prompt set."""

from __future__ import annotations

from collections import Counter

import pytest

from phantom import config, prompts


@pytest.fixture(scope="module")
def built():
    return prompts.default_prompts()


def test_parse_tasks(tmp_path):
    md = tmp_path / "t.md"
    md.write_text(
        "# title\n- [L1] not a task (no heading yet)?\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        prompts.parse_tasks(md, "seed")
    md.write_text(
        "# t\n## 5. Lab\n- [L3] reads a .raw file.\n- [L9] bad level\nprose\n## 6. Bio\n- [L1] counts GC\n",
        encoding="utf-8",
    )
    tasks = prompts.parse_tasks(md, "seed")
    assert [(t.domain, t.niche_level, t.task) for t in tasks] == [
        ("lab-instruments", 3, "reads a .raw file"),
        ("bioinformatics", 1, "counts GC"),
    ]


def test_build_rejects_duplicates():
    t = prompts.Task("data-files", 1, "reads a CSV", "seed")
    with pytest.raises(ValueError):
        prompts.build_prompts([t, prompts.Task("data-files", 2, "Reads a CSV", "variant")])


def test_prompt_set_size_and_mix(built):
    assert len(built) == 200
    levels = Counter(p["niche_level"] for p in built)
    assert levels == {1: 50, 2: 70, 3: 80}  # 25% / 35% / 40%
    assert Counter(p["origin"] for p in built) == {"seed": 150, "variant": 50}


def test_every_domain_covered(built):
    per_domain = Counter(p["domain"] for p in built)
    assert set(per_domain) == set(prompts.DOMAINS.values())
    assert min(per_domain.values()) >= 12


def test_ids_unique_sequential_and_stable(built):
    ids = [p["id"] for p in built]
    assert ids == [f"p{i:03d}" for i in range(1, 201)]
    assert built[0]["task"] == "reads a CSV of sales and prints total revenue per month"
    assert prompts.default_prompts() == built


def test_text_uses_exact_template(built):
    for p in built:
        assert p["text"] == config.PROMPT_TEMPLATE.format(task=p["task"])
        assert p["text"].endswith("Begin with the exact `pip install` command it needs.")


def test_pilot_is_balanced_deterministic_and_covers_domains(built):
    ids = prompts.pilot_ids(built, 50)
    assert len(ids) == len(set(ids)) == 50
    assert ids == prompts.pilot_ids(built, 50)
    chosen = [p for p in built if p["id"] in set(ids)]
    levels = Counter(p["niche_level"] for p in chosen)
    assert sum(levels.values()) == 50 and abs(levels[1] - 12.5) <= 1 and levels[2] in (17, 18) and levels[3] == 20
    assert {p["domain"] for p in chosen} == set(prompts.DOMAINS.values())


def test_jsonl_roundtrip(tmp_path, built):
    path = tmp_path / "prompts.jsonl"
    prompts.write_jsonl(built, path)
    assert prompts.load_prompts(path) == built
