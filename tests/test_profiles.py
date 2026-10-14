import json

from manga_rt_translator.models import Rect
from manga_rt_translator.profiles import (
    SiteProfile,
    SiteProfileStore,
    normalized_region,
    profile_key,
    scaled_region,
)


def test_profile_key_normalizes_url_but_keeps_subdomains():
    assert profile_key("https://www.example.com:8443/reader/1") == "example.com"
    assert profile_key("https://reader.example.com/chapter") == "reader.example.com"
    assert profile_key("file:///chapter") is None


def test_region_round_trips_across_viewport_sizes():
    region = normalized_region(Rect(100, 50, 400, 300), 1000, 800)
    assert region == (0.1, 0.0625, 0.4, 0.375)
    assert scaled_region(region, 2000, 1000) == Rect(200, 62, 800, 375)


def test_region_is_clamped_to_capture_viewport():
    assert normalized_region(Rect(-10, 10, 80, 100), 100, 100) == (0.0, 0.1, 0.7, 0.9)
    assert scaled_region((0.9, 0.9, 0.5, 0.5), 100, 100) == Rect(90, 90, 10, 10)
    assert scaled_region((0.1, 0.2, 1.2, 0.2), 100, 100) is None


def test_profile_store_round_trip_and_never_persists_api_keys(tmp_path):
    path = tmp_path / "profiles.json"
    store = SiteProfileStore(path)
    profile = SiteProfile(
        source_lang="japan",
        provider="DeepL",
        analysis_mode="Manual",
        fps=20,
        overlay_enabled=False,
        ocr_only=True,
        region=(0.1, 0.2, 0.6, 0.7),
        overlay_font_family="Arial",
        overlay_font_size=22,
        overlay_opacity=0.75,
    )

    assert store.save("https://www.manga.example/series", profile)
    loaded = store.load("https://manga.example/chapter/2")
    assert loaded == SiteProfile(
        source_lang="japan",
        provider="DeepL",
        analysis_mode="Manual",
        fps=12,
        overlay_enabled=False,
        ocr_only=True,
        region=(0.1, 0.2, 0.6, 0.7),
        overlay_font_family="Arial",
        overlay_font_size=22,
        overlay_opacity=0.75,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["sites"]) == {"manga.example"}
    assert "api_key" not in path.read_text(encoding="utf-8").lower()


def test_profile_overlay_appearance_is_clamped_and_legacy_profiles_remain_valid(tmp_path):
    path = tmp_path / "profiles.json"
    store = SiteProfileStore(path)
    assert store.save(
        "https://manga.example",
        SiteProfile(overlay_font_family="  Noto Sans  ", overlay_font_size=80, overlay_opacity=0.1),
    )
    assert store.load("https://manga.example") == SiteProfile(
        overlay_font_family="Noto Sans",
        overlay_font_size=36,
        overlay_opacity=0.35,
    )

    path.write_text('{"sites":{"old.example":{"fps":8}}}', encoding="utf-8")
    assert store.load("https://old.example") == SiteProfile()


def test_invalid_or_corrupt_profile_data_falls_back_without_crashing(tmp_path):
    path = tmp_path / "profiles.json"
    store = SiteProfileStore(path)
    path.write_text('{"sites":{"manga.example":{"fps":"bad"}}}', encoding="utf-8")
    assert store.load("https://manga.example") is None
    path.write_text("not json", encoding="utf-8")
    assert store.load("https://manga.example") is None
    assert not store.save("not a URL", SiteProfile())
