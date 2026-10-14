from __future__ import annotations

from types import SimpleNamespace

from manga_rt_translator.argos_models import (
    install_model_route,
    installed_model_pairs,
    resolve_model_route,
)


def test_japanese_to_brazilian_portuguese_resolves_english_pivot():
    route = resolve_model_route(
        "japan", "pt-BR", [("ja", "en"), ("en", "pb")]
    )
    assert route == [("ja", "en"), ("en", "pb")]


def test_direct_route_is_preferred_to_pivot():
    route = resolve_model_route(
        "ja", "pt-BR", [("ja", "pb"), ("ja", "en"), ("en", "pb")]
    )
    assert route == [("ja", "pb")]


def test_unavailable_route_returns_empty():
    assert resolve_model_route("ko", "pt-BR", [("en", "pb")]) == []


def test_installed_models_excludes_non_translation_packages():
    packages = [
        SimpleNamespace(type="translate", from_code="ja", to_code="en"),
        SimpleNamespace(type="sbd", from_code="en", to_code=None),
    ]
    module = SimpleNamespace(get_installed_packages=lambda: packages)
    assert installed_model_pairs(module) == [("ja", "en")]


def test_install_route_requires_confirmation_and_installs_only_missing(monkeypatch):
    installed = []
    model_ja_en = SimpleNamespace(
        type="translate", from_code="ja", to_code="en", code="translate-ja_en",
        links=["https://models.example/ja-en"], install=lambda: installed.append("ja-en"),
    )
    model_en_pb = SimpleNamespace(
        type="translate", from_code="en", to_code="pb", code="translate-en_pb",
        links=["https://models.example/en-pb"], install=lambda: installed.append("en-pb"),
    )
    package_module = SimpleNamespace(
        update_package_index=lambda: None,
        get_available_packages=lambda: [model_ja_en, model_en_pb],
        get_installed_packages=lambda: [model_ja_en],
    )
    monkeypatch.setattr(
        "manga_rt_translator.argos_models.argos_modules",
        lambda: (package_module, object()),
    )
    messages = []

    result = install_model_route(
        "ja", "pt-BR", confirm=lambda _prompt: True, output=messages.append
    )

    assert result == 0
    assert installed == ["en-pb"]
    assert any("licença individual" in message for message in messages)


def test_install_cancellation_does_not_download_models(monkeypatch):
    installed = []
    model = SimpleNamespace(
        type="translate", from_code="en", to_code="pb", code="translate-en_pb",
        links=[], install=lambda: installed.append(True),
    )
    package_module = SimpleNamespace(
        update_package_index=lambda: None,
        get_available_packages=lambda: [model],
        get_installed_packages=lambda: [],
    )
    monkeypatch.setattr(
        "manga_rt_translator.argos_models.argos_modules",
        lambda: (package_module, object()),
    )
    messages = []

    result = install_model_route(
        "en", "pt-BR", confirm=lambda _prompt: False, output=messages.append
    )

    assert result == 0
    assert not installed
    assert messages[-1].startswith("Instalação cancelada")


def test_argos_install_confirmation_defaults_to_cancel_without_console(monkeypatch):
    from manga_rt_translator.cli import _confirm_argos_install

    def no_console(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_console)
    assert not _confirm_argos_install("Baixar? [s/N]")
