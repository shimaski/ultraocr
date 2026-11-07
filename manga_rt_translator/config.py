from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass
class CaptureConfig:
    backend: str = "auto"
    fps: int = 30
    monitor: int = 0
    region: tuple[int, int, int, int] | None = None
    window_title: str | None = None
    window_handle: int | None = None


@dataclass
class DiffConfig:
    tile_size: int = 96
    pixel_threshold: int = 22
    min_changed_ratio: float = 0.015
    roi_padding: int = 20
    stable_ms: int = 220
    max_rois_per_tick: int = 12
    large_change_tile_ratio: float = 0.55
    scroll_cooldown_ms: int = 300
    # Debounce used by the event-driven selected-region reader.
    scroll_settle_ms: int = 550
    # Suppress runaway re-OCR of a spot that keeps changing in place — animated
    # ads, auto-rotating carousels, GIF banners. After a location fires this
    # many times inside the window it is skipped until the older fires age out,
    # so it re-checks occasionally instead of hammering OCR (the main cause of
    # "it freezes out of nowhere"). Real reading text appears once and never
    # trips this. Set animation_suppression = false to turn it off.
    animation_suppression: bool = True
    animation_window_ms: int = 9000
    animation_max_fires: int = 3


@dataclass
class OcrConfig:
    enabled: bool = True
    source_lang: str = "en"
    extra_langs: list[str] = field(default_factory=list)
    min_confidence: float = 0.55
    ignore_top_pixels: int = 120
    # PaddleOCR detection model family: "mobile" (fast, real-time scroll) or
    # "server" (slower, slightly better boxes on stylized text).
    det_model: str = "mobile"
    # PaddleOCR recognition model family: "mobile" (fast) or "server"
    # (better on stylized comic fonts, slower).
    rec_model: str = "mobile"
    # Text detector: paddle (default), comic, or hybrid (experimental English
    # batched recognition + Paddle fallback; requires an existing ONNX model).
    detector: str = "paddle"
    # A 2x retry helps small stylized text, but repeating a full viewport costs
    # several seconds. Only tight crops below this area may be retried.
    upscale_retry_max_pixels: int = 250_000
    # Cap large OCR crops before inference; geometry is mapped to the source frame.
    max_input_side: int = 640
    # CPU threads for PaddleOCR inference. 0 = auto (min(8, cpu_count)); the
    # recognizer is CPU-bound, so more threads shorten a dense page on a
    # many-core machine. OCR runs below-normal priority, so it yields to the UI.
    cpu_threads: int = 0


@dataclass
class TranslationConfig:
    target_lang: str = "pt-BR"
    provider: str = "google"
    aihub_app_package: str = "com.ultraocr"
    aihub_engine: str = "gemini"
    deepl_api_key: str = ""
    deepl_free: bool = True
    # Free Gemini API key (aistudio.google.com/apikey) — context-aware LLM
    # translation, much more natural than machine translation.
    gemini_api_key: str = ""
    # Gemini model (empty = app default, currently gemini-3.6-flash).
    gemini_model: str = ""
    # Minimum spacing between Gemini requests (seconds). The free tier allows
    # ~15 RPM (4s/request); lower values translate faster at the cost of more
    # 429s, which the chain absorbs by falling back to Google instantly.
    gemini_min_interval: float = 1.0
    # LibreTranslate: public instance (needs free key) or self-hosted Docker
    # (unlimited/offline): libre_url = "http://localhost:5000"
    libre_url: str = "https://libretranslate.org"
    libre_api_key: str = ""
    timeout_seconds: int = 20
    fuzzy_threshold: int = 92


@dataclass
class OverlayConfig:
    enabled: bool = True
    font_family: str = "Segoe UI"
    font_size: int = 18
    # High opacity so the translated box fully covers the original text.
    opacity: float = 0.94
    ttl_seconds: int = 900


@dataclass
class CacheConfig:
    path: str = "manga_translations.sqlite3"


@dataclass
class ProviderRateLimitConfig:
    failure_threshold: int = 3
    open_duration_s: float = 120.0
    rps: float = 1.0
    burst: int = 3
    quality: int = 5
    daily_limit: int = 0
    reset_at_utc_hour: int = 0
    chars: bool = False


@dataclass
class RateLimitsConfig:
    google: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(3, 60, 4.0, 10, 6, 0))
    gemini: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(5, 180, 0.5, 5, 9, 1500))
    aihub: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(3, 60, 1.0, 3, 8, 0))
    deepl: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(3, 120, 1.0, 3, 9, 500000, 1, True))
    libre: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(3, 60, 10.0, 20, 7, 0))
    mymemory: ProviderRateLimitConfig = field(default_factory=lambda: ProviderRateLimitConfig(3, 120, 0.5, 3, 5, 50000))


@dataclass
class LoggingConfig:
    level: str = "INFO"
    rate_limiter_periodic_s: float = 30.0
    rate_limiter_cache_s: float = 60.0
    rate_limiter_emit_per_request: bool = False


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    diff: DiffConfig = field(default_factory=DiffConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limits: RateLimitsConfig = field(default_factory=RateLimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path | None) -> AppConfig:
    config = AppConfig()
    if not path:
        _apply_env_secrets(config.translation)
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if tomllib is None:
        raise RuntimeError("Python 3.11+ is required to read TOML config files.")

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    _apply_section(config.capture, data.get("capture", {}))
    _apply_section(config.diff, data.get("diff", {}))
    _apply_section(config.ocr, data.get("ocr", {}))
    _apply_section(config.translation, data.get("translation", {}))
    _apply_section(config.overlay, data.get("overlay", {}))
    _apply_section(config.cache, data.get("cache", {}))
    _apply_section(config.logging, data.get("logging", {}))
    rate_limits = data.get("rate_limits", {})
    for provider_name in config.rate_limits.__dataclass_fields__:
        values = rate_limits.get(provider_name, {}) if isinstance(rate_limits, dict) else {}
        _apply_section(getattr(config.rate_limits, provider_name), values)

    if isinstance(config.capture.region, list):
        config.capture.region = tuple(config.capture.region)  # type: ignore[assignment]

    # Secrets belong in the process environment, not in a checked-in or
    # shareable TOML file. Environment values intentionally win over TOML.
    _apply_env_secrets(config.translation)
    return config


def configure_logging(config: LoggingConfig) -> None:
    level = getattr(logging, str(config.level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger().setLevel(level)


def _apply_env_secrets(translation: TranslationConfig) -> None:
    translation.gemini_api_key = os.getenv(
        "MANGA_RT_GEMINI_API_KEY", translation.gemini_api_key
    ).strip()
    translation.deepl_api_key = os.getenv(
        "MANGA_RT_DEEPL_API_KEY", translation.deepl_api_key
    ).strip()
    translation.libre_api_key = os.getenv(
        "MANGA_RT_LIBRE_API_KEY", translation.libre_api_key
    ).strip()


def _apply_section(target: object, values: dict[str, Any]) -> None:
    allowed = set(getattr(target, "__dataclass_fields__", {}).keys())
    for key, value in values.items():
        if key in allowed:
            setattr(target, key, value)
