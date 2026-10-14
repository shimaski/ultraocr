from __future__ import annotations

from manga_rt_translator.glossary import (
    Glossary,
    combine_glossaries,
    work_glossary_identity,
)


def test_glossary_toml_round_trip_with_unicode_and_quoted_keys(tmp_path):
    glossary = Glossary(
        names={"白銀御行": "Miyuki Shirogane", "hero's party": "grupo do herói"},
        terms={"senpai": "senpai"},
        instructions=["Use português brasileiro.", "Preserve \"pausas\" e emoção."],
    )
    path = tmp_path / "custom-glossary.toml"

    glossary.save(path)
    loaded = Glossary.load(path)

    assert loaded == glossary


def test_save_replaces_existing_glossary_without_leaving_temp_files(tmp_path):
    path = tmp_path / "glossary.toml"
    path.write_text("[terms]\nold = \"old\"\n", encoding="utf-8")

    Glossary(names={"New Name": "New Name"}).save(path)

    assert Glossary.load(path).names == {"New Name": "New Name"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_without_path_uses_local_glossary_file_when_present(tmp_path, monkeypatch):
    path = tmp_path / "glossary.toml"
    Glossary(names={"Local Name": "Local Name"}).save(path)
    monkeypatch.chdir(tmp_path)

    assert Glossary.load(None).names == {"Local Name": "Local Name"}


def test_work_identity_is_stable_across_chapters_and_separated_by_site():
    first = work_glossary_identity(
        "https://demonicscans.org/title/The-Great-Mage/chapter/153/1"
    )
    next_chapter = work_glossary_identity(
        "https://demonicscans.org/title/The-Great-Mage/chapter/154/1"
    )
    other_site = work_glossary_identity(
        "https://reader.example/title/The-Great-Mage/chapter/153"
    )
    webtoon = work_glossary_identity(
        "https://www.webtoons.com/en/fantasy/list/viewer?titleNo=12345&episodeNo=8"
    )

    assert first is not None and first == next_chapter
    assert first[1] == "The-Great-Mage"
    assert other_site is not None and other_site[0] != first[0]
    assert webtoon is not None and webtoon[1] == "12345"
    assert work_glossary_identity("https://example.org/chapter/153/1") is None


def test_work_glossary_overrides_base_terms_and_instructions():
    combined = combine_glossaries(
        Glossary(names={"Aki": "Aki"}, terms={"guild": "guilda"}, instructions=["base"]),
        Glossary(terms={"guild": "associação"}, instructions=["work"]),
        "work:123",
    )

    assert combined.names == {"Aki": "Aki"}
    assert combined.terms == {"guild": "associação"}
    assert combined.instructions == ["base", "work"]
    assert combined.cache_scope == "work:123"
