from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from manga_rt_translator.models import Rect


@dataclass(frozen=True, slots=True)
class SiteProfile:
    source_lang: str = "Auto"
    provider: str = "Google"
    analysis_mode: str = "Contínuo"
    fps: int = 8
    overlay_enabled: bool = True
    ocr_only: bool = False
    region: tuple[float, float, float, float] | None = None
    overlay_font_family: str | None = None
    overlay_font_size: int | None = None
    overlay_opacity: float | None = None


def profile_key(url: str) -> str | None:
    try:
        host = (urlsplit(url).hostname or "").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None
    host = host.removeprefix("www.").rstrip(".")
    return host or None


def normalized_region(region: Rect | None, viewport_width: int, viewport_height: int) -> tuple[float, float, float, float] | None:
    if region is None or viewport_width <= 0 or viewport_height <= 0:
        return None
    x1 = max(0, min(viewport_width, region.x))
    y1 = max(0, min(viewport_height, region.y))
    x2 = max(x1, min(viewport_width, region.x + region.width))
    y2 = max(y1, min(viewport_height, region.y + region.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 / viewport_width, y1 / viewport_height,
            (x2 - x1) / viewport_width, (y2 - y1) / viewport_height)


def scaled_region(region: object, viewport_width: int, viewport_height: int) -> Rect | None:
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    if viewport_width <= 0 or viewport_height <= 0:
        return None
    try:
        x, y, width, height = (float(value) for value in region)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
        return None
    left = max(0, min(viewport_width - 1, round(x * viewport_width)))
    top = max(0, min(viewport_height - 1, round(y * viewport_height)))
    scaled_width = max(1, min(viewport_width - left, round(width * viewport_width)))
    scaled_height = max(1, min(viewport_height - top, round(height * viewport_height)))
    return Rect(left, top, scaled_width, scaled_height)


class SiteProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, url: str) -> SiteProfile | None:
        key = profile_key(url)
        if key is None:
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            values = payload.get("sites", {}).get(key)
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(values, dict):
            return None
        try:
            fps = max(1, min(12, int(values.get("fps", 8))))
            region_value = values.get("region")
            region = tuple(float(value) for value in region_value) if isinstance(region_value, list) and len(region_value) == 4 else None
            if region is not None and not all(0.0 <= value <= 1.0 for value in region):
                region = None
            source = str(values.get("source_lang", "Auto"))
            provider = str(values.get("provider", "Google"))
            analysis = str(values.get("analysis_mode", "Contínuo"))
            font_family_value = values.get("overlay_font_family")
            font_family = (
                str(font_family_value).strip()[:80]
                if isinstance(font_family_value, str) and font_family_value.strip()
                else None
            )
            font_size_value = values.get("overlay_font_size")
            font_size = (
                max(8, min(36, int(font_size_value)))
                if font_size_value is not None
                else None
            )
            opacity_value = values.get("overlay_opacity")
            opacity = (
                max(0.35, min(1.0, float(opacity_value)))
                if opacity_value is not None
                else None
            )
            if source not in {"Auto", "en", "japan", "korean"}:
                source = "Auto"
            if provider not in {"Google", "DeepL", "Gemini", "Libre", "Argos offline", "Echo"}:
                provider = "Google"
            if analysis not in {"Manual", "Contínuo"}:
                analysis = "Contínuo"
            return SiteProfile(
                source_lang=source,
                provider=provider,
                analysis_mode=analysis,
                fps=fps,
                overlay_enabled=bool(values.get("overlay_enabled", True)),
                ocr_only=bool(values.get("ocr_only", False)),
                region=region,
                overlay_font_family=font_family,
                overlay_font_size=font_size,
                overlay_opacity=opacity,
            )
        except (TypeError, ValueError):
            return None

    def save(self, url: str, profile: SiteProfile) -> bool:
        key = profile_key(url)
        if key is None:
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        sites = payload.get("sites")
        if not isinstance(sites, dict):
            sites = {}
        values: dict[str, object] = {
            "source_lang": profile.source_lang,
            "provider": profile.provider,
            "analysis_mode": profile.analysis_mode,
            "fps": max(1, min(12, int(profile.fps))),
            "overlay_enabled": bool(profile.overlay_enabled),
            "ocr_only": bool(profile.ocr_only),
            "region": list(profile.region) if profile.region is not None else None,
            "overlay_font_family": profile.overlay_font_family,
            "overlay_font_size": (
                max(8, min(36, int(profile.overlay_font_size)))
                if profile.overlay_font_size is not None
                else None
            ),
            "overlay_opacity": (
                max(0.35, min(1.0, float(profile.overlay_opacity)))
                if profile.overlay_opacity is not None
                else None
            ),
        }
        sites[key] = values
        payload["version"] = 1
        payload["sites"] = sites
        temp_name: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            return True
        except OSError:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            return False


def default_profile_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return root / "MangaRealTimeTranslator" / "reader_profiles.json"
