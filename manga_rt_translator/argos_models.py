from __future__ import annotations

from collections.abc import Callable, Iterable


_LANGUAGE_CODES = {
    "english": "en",
    "japan": "ja",
    "japanese": "ja",
    "korea": "ko",
    "korean": "ko",
    "chinese": "zh",
    "portuguese": "pt",
    "pt-br": "pb",
    "portuguese (brazil)": "pb",
}


def language_code(value: str) -> str:
    normalized = (value or "").strip().lower()
    return _LANGUAGE_CODES.get(normalized, normalized.split("-")[0])


def resolve_model_route(
    source: str,
    target: str,
    available_pairs: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    source_code, target_code = language_code(source), language_code(target)
    pairs = set(available_pairs)
    if (source_code, target_code) in pairs:
        return [(source_code, target_code)]
    for pivot in ("en",):
        if pivot in {source_code, target_code}:
            continue
        first, second = (source_code, pivot), (pivot, target_code)
        if first in pairs and second in pairs:
            return [first, second]
    return []


def argos_modules():
    try:
        from argostranslate import package, translate
    except ImportError as exc:
        raise RuntimeError(
            'Argos ausente. Instale o backend com: '
            'python -m pip install -e ".[offline]"'
        ) from exc
    return package, translate


def installed_model_pairs(package_module: object) -> list[tuple[str, str]]:
    pairs = {
        (item.from_code, item.to_code)
        for item in package_module.get_installed_packages()
        if getattr(item, "type", "translate") == "translate"
        and getattr(item, "from_code", None)
        and getattr(item, "to_code", None)
    }
    return sorted(pairs)


def list_installed_models(output: Callable[[str], None] = print) -> int:
    try:
        package_module, _translate_module = argos_modules()
    except RuntimeError as exc:
        output(str(exc))
        return 1
    pairs = installed_model_pairs(package_module)
    if not pairs:
        output("Nenhum modelo Argos de tradução está instalado.")
        output("Use `argos-models install --source en --target pt-BR` para inglês.")
        output("Para japonês, instale a rota ja -> en -> pb.")
        return 0
    output("Modelos Argos instalados:")
    for source, target in pairs:
        output(f"  {source} -> {target}")
    return 0


def install_model_route(
    source: str,
    target: str,
    *,
    confirm: Callable[[str], bool],
    output: Callable[[str], None] = print,
) -> int:
    try:
        package_module, _translate_module = argos_modules()
        package_module.update_package_index()
        available = [
            item
            for item in package_module.get_available_packages()
            if getattr(item, "type", "translate") == "translate"
            and getattr(item, "from_code", None)
            and getattr(item, "to_code", None)
        ]
    except Exception as exc:
        output(f"Não foi possível consultar o índice Argos: {exc}")
        return 1

    pairs = resolve_model_route(
        source,
        target,
        ((item.from_code, item.to_code) for item in available),
    )
    if not pairs:
        output(
            f"O índice Argos não oferece rota para {language_code(source)} -> "
            f"{language_code(target)}."
        )
        return 1

    installed = set(installed_model_pairs(package_module))
    missing = [pair for pair in pairs if pair not in installed]
    if not missing:
        output("Todos os modelos necessários já estão instalados.")
        return 0

    by_pair = {(item.from_code, item.to_code): item for item in available}
    output("Modelos que serão baixados e instalados:")
    for pair in missing:
        model = by_pair[pair]
        output(f"  {pair[0]} -> {pair[1]} ({getattr(model, 'code', 'Argos')})")
        for url in getattr(model, "links", []):
            output(f"    {url}")
    output(
        "O índice não informa a licença individual dos modelos. Verifique os "
        "termos do modelo antes de prosseguir."
    )
    if not confirm("Baixar e instalar esses modelos agora? [s/N] "):
        output("Instalação cancelada; nenhum modelo foi baixado.")
        return 0

    for pair in missing:
        output(f"Instalando {pair[0]} -> {pair[1]}...")
        try:
            by_pair[pair].install()
        except Exception as exc:
            output(f"Falha ao instalar {pair[0]} -> {pair[1]}: {exc}")
            return 1
    output("Instalação Argos concluída.")
    return 0
