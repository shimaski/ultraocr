from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass
class Glossary:
    names: dict[str, str] = field(default_factory=dict)
    terms: dict[str, str] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        if not path:
            return cls.default()
        path = Path(path)
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
