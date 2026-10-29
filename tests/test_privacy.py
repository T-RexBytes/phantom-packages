"""Anonymised ids and the leak scan. All names here are made up and hashed at test time."""

from __future__ import annotations

import pytest

from phantom import config, privacy

FAKE = "zzq-ghostlib"  # invented; not a study name
TEST_KEY = "test-key-not-a-secret"


@pytest.fixture(autouse=True)
def test_key(monkeypatch):
    monkeypatch.setenv(privacy.ANON_KEY_ENV, TEST_KEY)


def test_ids_are_keyed_and_need_a_key(monkeypatch):
    assert privacy.anon_id(FAKE, b"one") != privacy.anon_id(FAKE, b"two")
    assert privacy.anon_id(FAKE) == privacy.anon_id(FAKE, TEST_KEY.encode())
    monkeypatch.delenv(privacy.ANON_KEY_ENV)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(privacy.MissingAnonKey):
        privacy.anon_id(FAKE)


def test_anon_id_is_stable_and_normalisation_insensitive():
    assert privacy.anon_id("Foo_Bar") == privacy.anon_id("foo-bar") == privacy.anon_id("foo.bar")
    assert len(privacy.anon_id(FAKE)) == 8 and privacy.anon_id(FAKE) == privacy.anon_id(FAKE)
    assert privacy.anon_id("foo-bar") != privacy.anon_id("foo-baz")


def test_write_load_roundtrip_ignores_comments(tmp_path):
    path = tmp_path / "ids.txt"
    assert privacy.write_anon_ids([FAKE, "ZZQ_Ghostlib", "other-fake"], path) == 2
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#") and FAKE not in text
    assert privacy.load_anon_ids(path) == {privacy.anon_id(FAKE), privacy.anon_id("other-fake")}


def test_candidate_tokens_split_on_markup_and_urls():
    text = ('Run `pip install Zzq_Ghostlib requests==2.0` or see https://pypi.org/project/zzq.ghostlib/ '
            'and "numpy", then zzq-ghostlib.tar.gz.')
    toks = privacy.candidate_tokens(text)
    assert {"zzq-ghostlib", "requests", "numpy", "pypi", "project"} <= toks
    assert "zzq-ghostlib-tar-gz" in toks and "zzq" in toks  # dotted tokens add their parts


def test_find_leaks_reports_path_line_and_id_only(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "results" / "table.csv").write_text("name,n\nrequests,3\nZZQ_GhostLib,2\n", encoding="utf-8")
    (tmp_path / "web" / "index.html").write_text("<p>All clear.</p>", encoding="utf-8")
    (tmp_path / "results" / "blob.bin").write_bytes(b"zzq-ghostlib")  # not a public text glob
    ids = {privacy.anon_id(FAKE)}
    leaks = privacy.find_leaks(tmp_path, ids)
    assert leaks == [("results/table.csv:3", privacy.anon_id(FAKE))]
    assert all(FAKE not in loc.lower() for loc, _ in leaks)


def test_find_leaks_skips_the_id_list_and_ignores_prose(tmp_path):
    ids_path = tmp_path / privacy.ANON_IDS_PATH.relative_to(privacy.config.ROOT)
    ids_path.parent.mkdir(parents=True)
    privacy.write_anon_ids([FAKE], ids_path)
    (tmp_path / "README.md").write_text("Models invent package names. We hash them.\n", encoding="utf-8")
    assert privacy.find_leaks(tmp_path, privacy.load_anon_ids(ids_path)) == []
