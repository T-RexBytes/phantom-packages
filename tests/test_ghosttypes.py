"""Ghost taxonomy rules on synthetic names (no study names, no network)."""

from __future__ import annotations

import pytest

from phantom import ghosttypes

REAL = ["zzqlib", "zzqtool-python-lib", "zzqcodec", "requests", "numpy"]
TASK = "Write a short Python script that reads a Zorblax AS/400 save file and lists its frames."
CLF = ghosttypes.Classifier(REAL)


@pytest.mark.parametrize("name,expected,evidence", [
    ("python-zzqlib", "affix_of_real", "zzqlib"),        # real + prefix
    ("zzqlib-parser", "affix_of_real", "zzqlib"),        # real + suffix
    ("zzqtool", "import_or_brand", "zzqtool-python-lib"),  # bare brand of a real family
    ("zorblax-as400-save", "task_words", "zorblax+as400+save"),  # glued "as/400" counts
    ("pyzorblaxreader", "task_words", "zorblax"),        # py prefix and generic suffix peeled
    ("zorb-frames", "task_words", "zorb+frames"),        # abbreviation of a task word
    ("zzqcodex", "near_real_name", "zzqcodec (edit distance 1)"),
    ("qwvbn-klm", "invented", ""),
])
def test_rules(name, expected, evidence):
    t = CLF.classify(name, TASK)
    assert (t.type, t.evidence) == (expected, evidence)


def test_first_match_wins_and_types_are_exhaustive():
    # "python-zzqlib" also has task-free parts, but the affix rule comes first
    assert CLF.classify("python-zzqlib", "zzqlib python").type == "affix_of_real"
    assert ghosttypes.TYPES[-1] == "invented" and len(set(ghosttypes.TYPES)) == 5


def test_short_names_need_an_exact_single_edit():
    # 4-char names never count as near a real name: too many chance collisions
    assert CLF.classify("nuxy", "unrelated task").type == "invented"
    assert CLF.classify("numpz", "unrelated task").type == "near_real_name"  # 5 chars, 1 edit


def test_task_vocabulary_glues_adjacent_words():
    v = ghosttypes.task_vocabulary("an AS/400 save-file")
    assert {"as", "400", "as400", "save", "file", "savefile"} <= v
