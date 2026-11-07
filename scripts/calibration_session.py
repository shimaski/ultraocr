from __future__ import annotations

import argparse
import logging
import os
import random
import sqlite3
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manga_rt_translator.config import AppConfig, configure_logging, load_config
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.translator import (
    DeepLTranslator,
    EchoTranslator,
    FallbackChain,
    GeminiTranslator,
    GoogleWebTranslator,
    LibreTranslateProvider,
    MyMemoryTranslator,
    create_translator,
    get_rate_tracker,
)

PHRASES = 5000
BATCH_SIZE = 10
PROVIDERS = ("google", "gemini", "deepl", "libre", "mymemory")
LOGGER = logging.getLogger("calibration")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate available UltraOCR providers.")
    parser.add_argument("--ignore-cooldown", action="store_true",
                        help="Probe providers even when persisted breaker state is open.")
    parser.add_argument("--providers", default=None,
                        help="Comma-separated subset: google,gemini,deepl,libre,mymemory.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check provider availability and write the report without translating.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow a healthy subset of selected providers; at least two remain required.")
    parser.add_argument("--health-timeout-s", type=int, default=5,
                        help="Timeout per provider health check (default: 5 seconds).")
    parser.add_argument("--report-path", type=Path, default=ROOT / "logs" / "calibration_report.md",
                        help="Markdown report output path.")
    return parser.parse_args(argv)


def _explicit_libre_url(config_path: Path) -> str | None:
    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        LOGGER.warning("[calibration] config read failed for LibreTranslate detection: %s", exc)
        return None
    value = document.get("translation", {}).get("libre_url")
    return str(value).strip() if value else None


def _report_configuration_warnings(config: AppConfig, libre_url: str | None) -> None:
    if not config.translation.gemini_api_key:
        print("[calibration] WARNING: gemini is unconfigured. To enable: set "
              "MANGA_RT_GEMINI_API_KEY (obtain at https://aistudio.google.com/apikey)")
    if not config.translation.deepl_api_key:
        print("[calibration] WARNING: deepl is unconfigured. To enable: set "
              "MANGA_RT_DEEPL_API_KEY (obtain at https://www.deepl.com/pro-api)")
    if not libre_url:
        print("[calibration] WARNING: libre is unconfigured. To enable: set libre_url in "
              "config.toml (recommended: local Docker libretranslate on http://localhost:5000)")
    print("[calibration] provider google: configured (Google web; no API key required)")
    print("[calibration] provider mymemory: configured (built-in endpoint)")


def load_persisted_state(config: AppConfig | None = None) -> dict[str, dict[str, Any]]:
    """Read provider state without initializing or mutating any database."""
    config = config or load_config(ROOT / "config.toml")
    configured_cache = Path(config.cache.path)
    if not configured_cache.is_absolute():
        configured_cache = ROOT / configured_cache
    candidates = [
        Path(os.path.expandvars(r"%LOCALAPPDATA%\UltraOCR\translations.db")),
        configured_cache,
        ROOT / "logs" / "calibration_state.sqlite3",
    ]
    state: dict[str, dict[str, Any]] = {}
    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        try:
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=1.0)
            try:
                cursor = connection.execute("SELECT * FROM provider_state")
                columns = [item[0] for item in cursor.description or ()]
                for raw in cursor.fetchall():
                    row = dict(zip(columns, raw))
                    if str(row.get("provider", "")) not in PROVIDERS:
                        continue
                    row["_source"] = str(path.resolve())
                    state.setdefault(str(row["provider"]), row)
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            LOGGER.warning("[calibration] state read failed path=%s reason=%s", path, exc)
    return state


def _make_providers(config: AppConfig, names: list[str], timeout_s: int) -> dict[str, object]:
    translation = config.translation
    result: dict[str, object] = {}
    for name in names:
        if name == "google":
            result[name] = GoogleWebTranslator(timeout_seconds=timeout_s, retries=0)
        elif name == "gemini" and translation.gemini_api_key:
            result[name] = GeminiTranslator(
                api_key=translation.gemini_api_key, timeout_seconds=timeout_s,
                model=translation.gemini_model, min_interval=0.0,
            )
        elif name == "deepl" and translation.deepl_api_key:
            result[name] = DeepLTranslator(
                api_key=translation.deepl_api_key, timeout_seconds=timeout_s,
                free=translation.deepl_free,
            )
        elif name == "libre":
            result[name] = LibreTranslateProvider(
                libre_url=translation.libre_url,
                api_key=translation.libre_api_key,
                timeout_seconds=timeout_s,
            )
        elif name == "mymemory":
            result[name] = MyMemoryTranslator(timeout_seconds=timeout_s, retries=0)
    return result


def _cooldown_remaining(name: str, state: dict[str, Any], config: AppConfig) -> float:
    if state.get("circuit_state") != "open":
        return 0.0
    opened_at = float(state.get("opened_at", 0.0) or 0.0)
    if opened_at <= 0:
        return 0.0
    duration = float(getattr(config.rate_limits, name).open_duration_s)
    return max(0.0, opened_at + duration - time.time())


def _clear_expired_breaker(name: str, state: dict[str, Any], config: AppConfig) -> None:
    if state.get("circuit_state") != "open" or _cooldown_remaining(name, state, config) > 0:
        return
    source = state.get("_source")
    if not source:
        return
    opened_at = float(state.get("opened_at", 0.0) or 0.0)
    LOGGER.info("[calibration] clearing expired breaker provider=%s source=%s opened_at=%.3f",
                name, source, opened_at)
    connection = sqlite3.connect(source, timeout=2.0)
    try:
        connection.execute(
            "UPDATE provider_state SET circuit_state='closed', consecutive_failures=0, "
            "consecutive_successes=0, opened_at=0.0 "
            "WHERE provider=? AND circuit_state='open' AND opened_at=?",
            (name, opened_at),
        )
        connection.commit()
    finally:
        connection.close()


def health_check_all(providers: dict[str, object], timeout_s: int = 5) -> dict[str, dict[str, Any]]:
    """Run exactly one bounded health check per supplied provider."""
    outcomes: dict[str, dict[str, Any]] = {}
    for name, provider in providers.items():
        for attr in ("timeout_seconds",):
            if hasattr(provider, attr):
                setattr(provider, attr, timeout_s)
        if hasattr(provider, "retries"):
            setattr(provider, "retries", 0)
        started = time.monotonic()
        try:
            if isinstance(provider, LibreTranslateProvider):
                detail = provider._request("hello", "en", "pt")
                ok = bool(str(detail).strip())
                detail = "LibreTranslate ready" if ok else "empty translation response"
            else:
                checker = getattr(provider, "check_health", None)
                if not callable(checker):
                    outcomes[name] = {"status": "skipped", "reason": "no health check available",
                                      "latency_ms": 0.0}
                    continue
                ok, detail = checker()
            outcomes[name] = {
                "status": "ok" if ok else "failed",
                "reason": str(detail),
                "latency_ms": (time.monotonic() - started) * 1000,
            }
        except Exception as exc:
            outcomes[name] = {
                "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.monotonic() - started) * 1000,
            }
    return outcomes


def _format_loaded_state(state: dict[str, dict[str, Any]], names: list[str],
                         config: AppConfig) -> str:
    fields = []
    for name in names:
        row = state.get(name)
        if not row:
            fields.append(f"{name}=unknown(default closed)")
            continue
        circuit = str(row.get("circuit_state", "unknown"))
        cooldown = _cooldown_remaining(name, row, config)
        fields.append(f"{name}={circuit}(cooldown {cooldown:.1f}s)")
    return " ".join(fields)


def _report(path: Path, config: AppConfig, state: dict[str, dict[str, Any]],
            outcomes: dict[str, dict[str, Any]], status: str, detail: str,
            metrics: dict[str, Any] | None = None) -> None:
    rows = ["# UltraOCR Rate-Limit Calibration", "", f"## Resultado: {status}", "", detail, "",
            "## Estado persistido antes do preflight", "",
            "Banco primario: `%LOCALAPPDATA%\\UltraOCR\\translations.db` (ausente nesta maquina). "
            "O leitor alternativo foi o `cache.path` configurado e, por fim, o banco isolado de calibracao. "
            "O schema persiste `opened_at` e a duracao do breaker vem de `rate_limits`; nao ha coluna `cooldown_until`.", ""]
    if not state:
        rows.append("Nenhuma linha `provider_state` foi encontrada nos bancos existentes.")
    else:
        for name, value in state.items():
            rows.append(f"- `{name}`: circuit=`{value.get('circuit_state')}`, "
                        f"failures=`{value.get('consecutive_failures')}`, "
                        f"opened_at=`{value.get('opened_at')}`, "
                        f"fonte=`{value.get('_source')}`")
    rows.extend(["", "## Preflight por provedor", "", "| Provedor | Estado | Latência (ms) | Motivo |",
                 "|---|---|---:|---|"])
    for name in PROVIDERS:
        item = outcomes.get(name, {"status": "skipped", "reason": "fora do subset"})
        rows.append(f"| {name} | {item.get('status')} | "
                    f"{item.get('latency_ms', 0):.1f} | {item.get('reason', '')} |")
    rows.extend(["", "## Diagnóstico e ação", "", detail, ""])
    if metrics:
        rows.extend(["## Métricas da sessão", "", "```text"])
        rows.extend(str(item) for item in metrics.get("log_lines", []))
        rows.extend(["```", ""])
    rows.extend([
        "Calibração de provedor só é aplicada quando a sessão de tradução termina com pelo menos 5 minutos e dados utilizáveis.",
        "O resumo de saúde não é uma amostra de throughput; não use os números de health check para elevar `rps`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(ROOT / "config.toml")
    config.logging.level = "INFO"
    config.logging.rate_limiter_periodic_s = 15
    config.logging.rate_limiter_cache_s = 30
    config.logging.rate_limiter_emit_per_request = False
    configure_logging(config.logging)

    names = list(config.rate_limits.__dataclass_fields__)
    libre_url = _explicit_libre_url(ROOT / "config.toml")
    _report_configuration_warnings(config, libre_url)
    if args.providers is not None:
        selected = [item.strip().lower() for item in args.providers.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(names))
        if unknown:
            print(f"[calibration] abort: providers not configured: {', '.join(unknown)}")
            return 2
        names = list(dict.fromkeys(selected))
    if not names:
        print("[calibration] abort: provider subset is empty")
        return 2

    state = load_persisted_state(config)
    print("[calibration] loaded provider_state: " + _format_loaded_state(state, names, config))
    if args.ignore_cooldown:
        print("[calibration] WARNING: --ignore-cooldown set, persisted cooldowns will be ignored")

    eligible: list[str] = []
    outcomes: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for name in names:
        if name in state and not args.ignore_cooldown:
            _clear_expired_breaker(name, state[name], config)
        if name == "gemini" and not config.translation.gemini_api_key:
            outcomes[name] = {"status": "skipped", "reason": "Gemini API key absent from config/environment"}
        elif name == "deepl" and not config.translation.deepl_api_key:
            outcomes[name] = {"status": "skipped", "reason": "DeepL API key absent from config/environment"}
        elif name == "libre" and not libre_url:
            outcomes[name] = {"status": "skipped", "reason": "libre_url is not explicitly configured in config.toml"}
        else:
            remaining = 0.0 if args.ignore_cooldown or name not in state else _cooldown_remaining(name, state[name], config)
            if remaining > 0:
                outcomes[name] = {"status": "skipped", "reason": f"cooldown {remaining:.1f}s (from previous session)"}
            else:
                pending.append(name)
                eligible.append(name)

    providers = _make_providers(config, pending, timeout_s=5)
    if args.health_timeout_s < 1:
        print("[calibration] abort: --health-timeout-s must be at least 1")
        return 2
    health = health_check_all(providers, timeout_s=args.health_timeout_s)
    outcomes.update(health)
    for name in names:
        item = outcomes.get(name, {"status": "skipped", "reason": "not selected"})
        print(f"[calibration] health {name}={item['status']}: {item.get('reason', '')}")

    healthy = [name for name in names if outcomes.get(name, {}).get("status") == "ok"]
    if args.preflight_only:
        detail = (f"Preflight-only: {len(healthy)} provedor(es) saudavel(is). "
                  "Nenhuma traducao de calibracao foi enviada.")
        status = "PREFLIGHT OK" if len(healthy) >= 2 else "ABORTADA NO PREFLIGHT"
        _report(args.report_path, config, state, outcomes, status, detail)
        print(f"[calibration] preflight-only complete healthy={len(healthy)} report={args.report_path}")
        return 0 if len(healthy) >= 2 else 2

    if len(healthy) < 2:
        detail = (f"Preflight encontrou {len(healthy)} provedor(es) saudavel(is); "
                  "sao necessarios pelo menos dois. Nenhuma traducao de calibracao foi enviada. "
                  "Configure pelo menos um provedor adicional: Gemini com MANGA_RT_GEMINI_API_KEY, "
                  "DeepL com MANGA_RT_DEEPL_API_KEY ou LibreTranslate com translation.libre_url; "
                  "depois rode novamente. Verifique tambem o cooldown/limite do Google; "
                  "`--ignore-cooldown` ignora apenas o estado persistido, "
                  "nao remove limites do provedor. `--allow-partial` nao reduz o minimo de dois.")
        _report(args.report_path, config, state, outcomes,
                "ABORTADA NO PREFLIGHT", detail)
        print(f"[calibration] abort: {detail}")
        return 2

    if args.allow_partial:
        print(f"[calibration] partial provider set accepted: {','.join(healthy)} (minimum=2)")

    # Use the application's tracker construction, but keep its persisted state isolated
    # from the reader's translation-memory database.
    config.cache.path = str(ROOT / "logs" / "calibration_state.sqlite3")
    create_translator(config)
    tracker = get_rate_tracker()
    chain_providers = _make_providers(config, healthy, timeout_s=config.translation.timeout_seconds)
    chain = FallbackChain([chain_providers[name] for name in healthy] + [EchoTranslator()],
                          tracker=tracker)

    bucket_waits = {name: 0 for name in healthy}
    original_before_request = tracker.before_request

    def count_bucket_wait(provider: str, text_length: int = 0) -> None:
        bucket = tracker._buckets.get(provider)
        if bucket is not None and bucket.wait_time() > 0:
            bucket_waits[provider] = bucket_waits.get(provider, 0) + 1
        original_before_request(provider, text_length)

    tracker.before_request = count_bucket_wait
    glossary = Glossary.default()
    started = time.monotonic()
    print(f"[calibration] start phrases={PHRASES} batches={PHRASES // BATCH_SIZE} "
          f"batch_size={BATCH_SIZE} source=en target=pt-BR providers={','.join(healthy)}")
    try:
        for batch_start in range(0, PHRASES, BATCH_SIZE):
            texts = [f"Test sentence number {index + 1} for calibration purposes"
                     for index in range(batch_start, min(batch_start + BATCH_SIZE, PHRASES))]
            tracker.record_cache_activity(0, len(texts))
            translated = chain.translate_batch(texts, "en", "pt-BR", glossary,
                                               context="Synthetic rate-limit calibration batch.")
            if len(translated) != len(texts):
                LOGGER.error("[calibration] batch result count mismatch expected=%s actual=%s",
                             len(texts), len(translated))
            batch_number = batch_start // BATCH_SIZE + 1
            if batch_number % 20 == 0:
                print(f"[calibration] progress batches={batch_number}/{PHRASES // BATCH_SIZE} "
                      f"elapsed_min={(time.monotonic() - started) / 60:.1f}")
            if batch_start + BATCH_SIZE < PHRASES:
                time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        detail = "Sessao interrompida manualmente; resumo parcial abaixo."
        summary = tracker.session_summary()
        tracker.close()
        _report(args.report_path, config, state, outcomes,
                "INTERROMPIDA", detail, metrics=summary)
        return 130
    except Exception as exc:
        detail = f"Sessao terminou com {type(exc).__name__}: {exc}; resumo parcial abaixo."
        LOGGER.exception("[calibration] session failed")
        summary = tracker.session_summary()
        tracker.close()
        _report(args.report_path, config, state, outcomes,
                "FALHA DURANTE SESSAO", detail, metrics=summary)
        return 1

    elapsed = time.monotonic() - started
    for provider, count in bucket_waits.items():
        print(f"[calibration] bucket_wait_count {provider}={count}")
    print(f"[calibration] completed phrases={PHRASES} elapsed_min={elapsed / 60:.2f}")
    summary = tracker.session_summary()
    tracker.close()
    detail = (f"Sessao concluida em {elapsed / 60:.2f} min com {PHRASES} frases. "
              "A configuracao so deve ser calibrada a partir de limites exercitados; "
              "provedores que nao receberam requests permanecem sem recomendacao.")
    _report(args.report_path, config, state, outcomes,
            "CONCLUIDA" if elapsed >= 300 else "CURTA; SEM APLICACAO DE CONFIG",
            detail, metrics=summary)
    if elapsed < 300:
        print("[calibration] config unchanged: session shorter than 5 minutes")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
