from manga_rt_translator.cli import apply_cli_overrides, build_parser
from manga_rt_translator.config import AppConfig


def test_libre_api_key_cli_override_reaches_translation_config():
    args = build_parser().parse_args(
        ["run", "--provider", "libre", "--libre-api-key", "local-test-key"]
    )
    config = AppConfig()

    apply_cli_overrides(config, args)

    assert config.translation.provider == "libre"
    assert config.translation.libre_api_key == "local-test-key"
