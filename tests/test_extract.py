"""Extractor tests (BUILD_SPEC §5.3, §6, §11), written before the extractor.

Fixture names are well-known real packages or obviously synthetic examples from
the spec; no model-produced names appear here (CLAUDE.md rule 3).
"""

from __future__ import annotations

import pytest

from phantom.extract import IMPORT_TO_DIST, extract, normalize, subtype


def names(text: str, source: str | None = None) -> list[str]:
    return [m.name for m in extract(text).mentions if source is None or m.source == source]


def fence(body: str, lang: str = "bash") -> str:
    return f"```{lang}\n{body}\n```"


# --- normalization (PEP 503) -------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PyYAML", "pyyaml"),
        ("python_dateutil", "python-dateutil"),
        ("zope.interface", "zope-interface"),
        ("Foo__Bar--baz..qux", "foo-bar-baz-qux"),
        ("requests", "requests"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


# --- install lines -----------------------------------------------------------

def test_simple_pip_install():
    assert names(fence("pip install requests")) == ["requests"]


def test_multiple_names_and_positions():
    ms = extract(fence("pip install numpy pandas matplotlib")).mentions
    assert [(m.name, m.position) for m in ms] == [("numpy", 0), ("pandas", 1), ("matplotlib", 2)]
    assert {m.source for m in ms} == {"pip_line"}


def test_flags_are_dropped():
    assert names(fence("pip install -U --user -q --pre --no-cache-dir --upgrade requests")) == ["requests"]


def test_flags_with_values_are_dropped():
    text = fence(
        "pip install -i https://example.org/simple --index-url https://x.org/s "
        "--extra-index-url https://y.org/s -r requirements.txt -e ./local flask"
    )
    assert names(text) == ["flask"]


def test_extras_and_version_pins_are_stripped():
    ms = extract(fence("pip install 'uvicorn[standard]>=0.30' PyYAML==6.0.1 numpy~=2.0 rich!=13.0 attrs<24")).mentions
    assert [m.name for m in ms] == ["uvicorn", "pyyaml", "numpy", "rich", "attrs"]
    assert ms[1].raw_token == "PyYAML==6.0.1"


def test_environment_markers_and_quotes_are_stripped():
    assert names(fence("pip install \"tomli; python_version < '3.11'\" \"colorama>=0.4\"")) == [
        "tomli", "colorama",
    ]


def test_python_m_pip_and_py_m_pip_and_pip3():
    text = fence("python -m pip install httpx\npython3 -m pip install rich\npy -m pip install typer\npip3 install click")
    assert names(text) == ["httpx", "rich", "typer", "click"]


def test_uv_add_and_uv_pip_install_are_uv_lines():
    ms = extract(fence("uv add polars\nuv pip install duckdb")).mentions
    assert [(m.name, m.source) for m in ms] == [("polars", "uv_line"), ("duckdb", "uv_line")]


def test_backslash_continuation():
    assert names(fence("pip install numpy \\\n    scipy \\\n    sympy")) == ["numpy", "scipy", "sympy"]


def test_caret_continuation_windows_cmd():
    assert names(fence("pip install numpy ^\n  scipy", lang="bat")) == ["numpy", "scipy"]


def test_and_chain_stops_arguments():
    assert names(fence("pip install requests && python main.py")) == ["requests"]


def test_and_chain_with_second_install():
    assert names(fence("pip install -U pip && pip install lxml; echo done")) == ["pip", "lxml"]


def test_urls_git_refs_and_paths_are_dropped():
    text = fence(
        "pip install git+https://github.com/psf/requests.git ./vendor/pkg C:\\pkgs\\x.whl "
        "https://example.org/a.tar.gz . flask"
    )
    ex = extract(text)
    assert [m.name for m in ex.mentions] == ["flask"]


def test_trailing_comment_is_ignored():
    assert names(fence("pip install requests  # HTTP client")) == ["requests"]


def test_install_line_in_python_comment():
    assert names(fence("# pip install pyserial\nimport serial", lang="python")) == ["pyserial"]


def test_inline_code_in_prose():
    text = "First run `pip install beautifulsoup4 lxml` and then execute the script."
    assert names(text) == ["beautifulsoup4", "lxml"]


def test_plain_prose_mention_is_not_parsed_beyond_command():
    # Words after the package in a sentence must not become names.
    text = "You can pip install requests to make HTTP calls."
    assert "to" not in names(text) and "make" not in names(text)


def test_jupyter_magic():
    assert names(fence("!pip install seaborn\n%pip install plotly", lang="python")) == ["seaborn", "plotly"]


def test_invalid_and_placeholder_tokens_are_dropped_and_counted():
    ex = extract(fence("pip install <package-name> requests " + "a" * 81))
    assert [m.name for m in ex.mentions] == ["requests"]
    assert len(ex.dropped) == 2


def test_unclosed_fence_from_truncated_output_is_still_code():
    text = "```bash\npip install pillow\n```\n```python\nimport cv2\nimg = cv2.imread("
    assert names(text) == ["pillow", "opencv-python"]


def test_duplicate_mentions_are_deduplicated_first_wins():
    text = fence("pip install requests") + "\nRun `pip install requests` first."
    assert names(text) == ["requests"]


# --- requirements blocks -----------------------------------------------------

def test_requirements_txt_block():
    text = fence("# deps\nrequests==2.32.3\nPyYAML>=6\n\nrich", lang="requirements.txt")
    ms = extract(text).mentions
    assert [(m.name, m.source) for m in ms] == [
        ("requests", "requirements_block"), ("pyyaml", "requirements_block"), ("rich", "requirements_block"),
    ]


def test_unlabelled_block_of_requirement_lines():
    assert names(fence("numpy==2.1\npandas", lang=""), "requirements_block") == ["numpy", "pandas"]


def test_python_block_is_not_requirements():
    assert names(fence("print\nx", lang="python")) == []


# --- imports -----------------------------------------------------------------

def test_mapped_imports():
    text = fence("import cv2\nfrom sklearn.cluster import KMeans\nimport yaml, PIL.Image\nfrom bs4 import BeautifulSoup", "python")
    ms = extract(text).mentions
    assert [(m.name, m.source) for m in ms] == [
        ("opencv-python", "import_mapped"), ("scikit-learn", "import_mapped"),
        ("pyyaml", "import_mapped"), ("pillow", "import_mapped"), ("beautifulsoup4", "import_mapped"),
    ]
    assert ms[0].raw_token == "import cv2"


def test_dotted_mapping_google_protobuf():
    assert names(fence("from google.protobuf import json_format", "python")) == ["protobuf"]


def test_stdlib_imports_are_dropped():
    assert names(fence("import os, sys, json\nfrom pathlib import Path\nimport xml.etree.ElementTree as ET", "python")) == []


def test_local_and_relative_imports_are_dropped():
    assert names(fence("from . import helpers\nfrom .models import User\nimport utils\nimport config", "python")) == []


def test_import_already_covered_by_install_line_is_not_duplicated():
    text = fence("pip install opencv-python requests") + fence("import cv2\nimport requests", "python")
    ms = extract(text).mentions
    assert [(m.name, m.source) for m in ms] == [("opencv-python", "pip_line"), ("requests", "pip_line")]


def test_unmapped_import_is_flagged_not_labelled():
    ms = extract(fence("import numpy\nimport frobnicate_tools", "python")).mentions
    assert [(m.name, m.source) for m in ms] == [
        ("numpy", "import_unmapped"), ("frobnicate-tools", "import_unmapped"),
    ]


# --- mapping table and subtypes (§6) -------------------------------------------

def test_import_to_dist_has_at_least_40_entries_and_spec_examples():
    assert len(IMPORT_TO_DIST) >= 40
    for mod, dist in {
        "cv2": "opencv-python", "sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow",
        "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "Crypto": "pycryptodome",
        "serial": "pyserial", "usb": "pyusb", "docx": "python-docx", "fitz": "pymupdf",
        "win32api": "pywin32", "zmq": "pyzmq", "gi": "pygobject", "Bio": "biopython",
    }.items():
        assert IMPORT_TO_DIST[mod] == dist


@pytest.mark.parametrize(
    "name,expected",
    [
        ("cv2", "module_name_confusion"),
        ("sklearn", "module_name_confusion"),
        ("json", "stdlib_confusion"),
        ("tkinter", "stdlib_confusion"),
        ("requests", None),
        ("opencv-python", None),
    ],
)
def test_subtype(name, expected):
    assert subtype(name) == expected


def test_module_name_in_pip_line_is_labelled_as_is():
    # `pip install cv2` keeps the name cv2 (it is a real mistake, tagged later).
    assert names(fence("pip install cv2 sklearn")) == ["cv2", "sklearn"]
