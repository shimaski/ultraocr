from __future__ import annotations

from manga_rt_translator.config import load_config


def test_defaults_are_realtime_friendly():
    config = load_config(None)
    assert config.translation.provider == "google"
    assert config.ocr.det_model == "mobile"
    assert config.ocr.source_lang == "en"
    assert config.diff.scroll_settle_ms == 550


def test_config_reads_det_model_and_korean(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[ocr]",
                'source_lang = "korean"',
                'extra_langs = ["japan"]',
                'det_model = "server"',
                "",
                "[translation]",
                'provider = "deepl"',
                'deepl_api_key = "abc123"',
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.ocr.source_lang == "korean"
    assert config.ocr.det_model == "server"
    assert config.translation.provider == "deepl"
    assert config.translation.deepl_api_key == "abc123"


def test_environment_secret_overrides_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[translation]\nprovider = "gemini"\ngemini_api_key = "from-file"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MANGA_RT_GEMINI_API_KEY", "from-environment")

    config = load_config(config_path)

    assert config.translation.gemini_api_key == "from-environment"
