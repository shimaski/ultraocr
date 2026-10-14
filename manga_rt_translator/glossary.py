from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass
class Glossary:
    names: dict[str, str] = field(default_factory=dict)
    terms: dict[str, str] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    cache_scope: str = ""

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        path = resolve_glossary_path(path)
        if path is None:
            return cls.default()
        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")
        if tomllib is None:
            raise RuntimeError("Python 3.11+ is required to read TOML glossary files.")
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        style = data.get("style", {})
        return cls(
            names=dict(data.get("names", {})),
            terms=dict(data.get("terms", {})),
            instructions=list(style.get("instructions", [])),
        )

    @classmethod
    def default(cls) -> "Glossary":
        return cls(
            instructions=[
                "Use portugues brasileiro natural.",
                "Nao traduza nomes proprios.",
                "Mantenha o tom emocional do dialogo.",
            ]
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sections = (
            ("names", self.names),
            ("terms", self.terms),
        )
        lines: list[str] = []
        for section, values in sections:
            lines.append(f"[{section}]")
            for source, target in values.items():
                key = source if re.fullmatch(r"[A-Za-z0-9_-]+", source) else _toml_string(source)
                lines.append(f"{key} = {_toml_string(target)}")
            lines.append("")
        lines.extend(("[style]", "instructions = ["))
        lines.extend(f"  {_toml_string(item)}," for item in self.instructions)
        lines.extend(("]", ""))
        content = "\n".join(lines)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def prompt_block(self) -> str:
        lines: list[str] = []
        if self.instructions:
            lines.append("Regras:")
            lines.extend(f"- {item}" for item in self.instructions)
        if self.names:
            lines.append("Nomes protegidos:")
            lines.extend(f"- {source} => {target}" for source, target in self.names.items())
        if self.terms:
            lines.append("Glossario:")
            lines.extend(f"- {source} => {target}" for source, target in self.terms.items())
        return "\n".join(lines)

    def repair_protected_names(self, translated_text: str) -> str:
        repaired = translated_text
        for source, target in self.names.items():
            if target and source not in repaired and target.lower() in repaired.lower():
                repaired = _replace_case_insensitive(repaired, target, source)
        return repaired


def work_glossary_identity(url: str) -> tuple[str, str] | None:
    """Return a stable per-series key and readable label inferred from a chapter URL."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None
    if not host:
        return None
    host = host.removeprefix("www.").rstrip(".")
    segments = [unquote(part).strip() for part in parts.path.split("/") if part.strip()]
    markers = {"chapter", "chapters", "episode", "episodes", "viewer", "read"}
    query = {
        key.casefold().replace("_", "").replace("-", ""): value
        for key, value in parse_qsl(parts.query)
    }
    label = next(
        (query[key] for key in ("titleno", "seriesid", "mangaid", "webtoonid") if key in query),
        "",
    )
    if not label:
        for index, segment in enumerate(segments):
            if segment.casefold() in markers:
                label = next(
                    (candidate for candidate in reversed(segments[:index])
                     if candidate.casefold() not in {"title", "series", "manga"}),
                    "",
                )
                break
    if not label:
        generic = {
            "title", "series", "manga", "comic", "webtoon", "en", "pt", "pt-br",
            "ja", "jp", "ko", "kr", *markers,
        }
        label = next(
            (segment for segment in segments
             if segment.casefold() not in generic
             and not segment.isdecimal()),
            "",
        )
    label = label.strip()
    if not label:
        return None
    digest = hashlib.sha256(f"{host}\0{label.casefold()}".encode("utf-8")).hexdigest()[:20]
    return digest, label


def combine_glossaries(base: Glossary, specific: Glossary, scope: str) -> Glossary:
    return Glossary(
        names={**base.names, **specific.names},
        terms={**base.terms, **specific.terms},
        instructions=list(dict.fromkeys([*base.instructions, *specific.instructions])),
        cache_scope=scope,
    )


def _replace_case_insensitive(text: str, needle: str, replacement: str) -> str:
    lower = text.lower()
    needle_lower = needle.lower()
    start = lower.find(needle_lower)
    while start >= 0:
        end = start + len(needle)
        text = text[:start] + replacement + text[end:]
        lower = text.lower()
        start = lower.find(needle_lower, start + len(replacement))
    return text


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def resolve_glossary_path(path: str | Path | None) -> Path | None:
    if path:
        return Path(path)
    default_path = Path.cwd() / "glossary.toml"
    return default_path if default_path.is_file() else None
