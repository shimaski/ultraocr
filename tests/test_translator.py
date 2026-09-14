from __future__ import annotations

import pytest

from manga_rt_translator.config import load_config
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.translator import (
    EchoTranslator,
    FallbackChain,
    GeminiTranslator,
    GoogleWebTranslator,
    MyMemoryTranslator,
    RateLimitError,
    ResilientTranslator,
    Translator,
    _deepl_source,
    _deepl_target,
    _google_source,
    _google_target,
    _parse_gtx_segments,
    _validate_mymemory_response,
    create_translator,
)


class _FailingTranslator(EchoTranslator):
    def translate(self, text, source_lang, target_lang, glossary, context=""):
        raise RuntimeError("provider down")


def test_echo_translator_returns_original_text():
    result = EchoTranslator().translate(
        "Hello world", "en", "pt-BR", Glossary.default()
    )
    assert result.text == "Hello world"
    assert result.provider == "echo"


def test_google_language_mapping_handles_paddle_names():
    assert _google_source("japan") == "ja"
    assert _google_source("korean") == "ko"
    assert _google_source("en") == "en"
    assert _google_source("Auto") == "auto"
    assert _google_target("pt-BR") == "pt"


def test_deepl_language_mapping_handles_paddle_names():
    assert _deepl_source("japan") == "JA"
    assert _deepl_source("korean") == "KO"
    assert _deepl_source("Auto") == ""
    assert _deepl_target("pt-BR") == "PT-BR"


def test_resilient_translator_falls_back_to_echo():
    resilient = ResilientTranslator(_FailingTranslator(), EchoTranslator())
    result = resilient.translate("Still there", "en", "pt-BR", Glossary.default())
    assert result.text == "Still there"
    assert result.provider == "echo"


def test_create_translator_respects_echo_flag():
    config = load_config(None)
    translator = create_translator(config, echo=True)
    assert isinstance(translator, EchoTranslator)


def test_create_translator_defaults_to_google_chain_with_echo_last():
    config = load_config(None)
    translator = create_translator(config)
    assert isinstance(translator, FallbackChain)
    assert isinstance(translator.providers[0], GoogleWebTranslator)
    assert isinstance(translator.providers[1], MyMemoryTranslator)
    assert isinstance(translator.providers[2], EchoTranslator)


def test_fallback_chain_returns_first_working_provider():
    chain = FallbackChain([_FailingTranslator(), _FailingTranslator(), EchoTranslator()])
    result = chain.translate("Still there", "en", "pt-BR", Glossary.default())
    assert result.text == "Still there"
    assert result.provider == "echo"


def test_fallback_chain_requires_providers():
    try:
        FallbackChain([])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty provider list")


class _RateLimitedTranslator(Translator):
    def __init__(self) -> None:
        self.calls = 0

    def translate(self, text, source_lang, target_lang, glossary, context=""):
        self.calls += 1
        raise RateLimitError("throttled", retry_after=60.0)


def test_fallback_chain_cools_down_rate_limited_provider():
    throttled = _RateLimitedTranslator()
    chain = FallbackChain([throttled, EchoTranslator()])
    first = chain.translate("Hello", "en", "pt-BR", Glossary.default())
    assert first.provider == "echo"
    second = chain.translate("Hello again", "en", "pt-BR", Glossary.default())
    assert second.provider == "echo"
    # The throttled provider must not be contacted again during cooldown.
    assert throttled.calls == 1


class _GtxStub(GoogleWebTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []

    def _request(self, source, target, text):
        self.requests.append(text)
        if "\n" in text:
            return "linha um\nlinha dois"
        return text.upper()


def test_google_batch_sends_one_request_and_maps_lines_back():
    stub = _GtxStub()
    results = stub.translate_batch(["one", "two"], "en", "pt-BR", Glossary.default())
    assert [r.text for r in results] == ["linha um", "linha dois"]
    assert len(stub.requests) == 1
    assert stub.requests[0] == "one\ntwo"


def test_google_batch_falls_back_to_singles_on_line_mismatch():
    class _MergeStub(_GtxStub):
        def _request(self, source, target, text):
            self.requests.append(text)
            if "\n" in text:
                return "merged into one"
            return text.upper()

    stub = _MergeStub()
    results = stub.translate_batch(["one", "two"], "en", "pt-BR", Glossary.default())
    assert [r.text for r in results] == ["ONE", "TWO"]
    assert len(stub.requests) == 3  # 1 batch attempt + 2 single retries


class _GeminiBatchStub(GeminiTranslator):
    def __init__(self) -> None:
        super().__init__(api_key="test", min_interval=0)
        self.requests: list[str] = []

    def _request(self, prompt: str) -> str:
        self.requests.append(prompt)
        return "1. Olá\n2. Como você está?"


def test_gemini_batch_maps_one_based_response_to_original_indices():
    stub = _GeminiBatchStub()
    results = stub.translate_batch(
        ["Hello", "How are you?"], "en", "pt-BR", Glossary.default()
    )

    assert [result.text for result in results] == ["Olá", "Como você está?"]
    assert len(stub.requests) == 1


def test_base_translate_batch_loops_translate():
    translator = EchoTranslator()
    results = translator.translate_batch(["a", "b"], "en", "pt-BR", Glossary.default())
    assert [r.text for r in results] == ["a", "b"]


def test_gtx_segment_parsing():
    payload = [
        [["Olá mundo", "Hello world", None, None], ["!", "!", None, None]],
        None,
        "en",
    ]
    assert _parse_gtx_segments(payload) == "Olá mundo!"


def test_mymemory_response_validation():
    assert _validate_mymemory_response("Olá mundo", 200) == "Olá mundo"
    try:
        _validate_mymemory_response(
            "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY", 200
        )
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError for quota warning")


def test_create_translator_rejects_unknown_provider():
    config = load_config(None)
    config.translation.provider = "babelfish"
    with pytest.raises(RuntimeError, match="Unknown translation provider"):
        create_translator(config)


def test_config_loader_ignores_removed_translation_keys(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[translation]",
                'target_lang = "pt-BR"',
                'provider = "google"',
                'model = "legacy-model"',
                'ollama_url = "http://127.0.0.1:11434"',
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.translation.provider == "google"
    assert config.translation.target_lang == "pt-BR"
