from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from manga_rt_translator.glossary import Glossary
from manga_rt_translator.work_queue import WorkCancelled


@dataclass
class TranslationResult:
    text: str
    provider: str


class RateLimitError(RuntimeError):
    """Provider refused the request (quota, throttle). Tells the chain how
    long to keep the provider on the sidelines."""

    def __init__(self, message: str, retry_after: float = 60.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitTracker:
    """Adaptive per-provider rate limit tracker with exponential backoff.

    Tracks consecutive failures and computes retry delays using exponential
    backoff with jitter. Parses Retry-After and X-RateLimit-* headers when
    available. Resets on success.
    """

    def __init__(self) -> None:
        self._consecutive_failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._last_success: dict[str, float] = {}
        self._request_times: dict[str, list[float]] = {}

    def record_success(self, provider: str) -> None:
        self._consecutive_failures[provider] = 0
        self._last_success[provider] = time.time()
        # Reset cooldown — provider responded successfully, it's available now
        self._cooldown_until.pop(provider, None)
        self._request_times.setdefault(provider, []).append(time.time())
        # Keep only the last 60s of request timestamps
        cutoff = time.time() - 60.0
        self._request_times[provider] = [
            t for t in self._request_times[provider] if t > cutoff
        ]

    def record_rate_limit(self, provider: str, retry_after: float = 0.0,
                          headers: dict[str, str] | None = None) -> None:
        """Record a rate limit hit and compute cooldown with backoff."""
        failures = self._consecutive_failures.get(provider, 0) + 1
        self._consecutive_failures[provider] = failures

        # Parse Retry-After header (seconds or HTTP-date)
        if retry_after <= 0 and headers:
            retry_after = self._parse_retry_after(headers)

        # Exponential backoff: base * 2^failures + jitter
        base = max(retry_after, 5.0)
        backoff = min(base * (2 ** min(failures - 1, 6)), 300.0)
        jitter = random.uniform(0.5, 1.5)
        delay = backoff * jitter

        self._cooldown_until[provider] = time.time() + delay
        print(
            f"[ratelimit] {provider}: {failures} consecutive failures, "
            f"cooldown {delay:.1f}s (base={base:.1f}, jitter={jitter:.2f})",
            flush=True,
        )

    def record_transient_error(self, provider: str) -> None:
        failures = self._consecutive_failures.get(provider, 0) + 1
        self._consecutive_failures[provider] = failures
        # Shorter cooldown for transient errors (network, timeout)
        delay = min(5.0 * (2 ** min(failures - 1, 4)), 60.0)
        self._cooldown_until[provider] = time.time() + delay

    def is_available(self, provider: str) -> bool:
        return time.time() >= self._cooldown_until.get(provider, 0.0)

    def cooldown_remaining(self, provider: str) -> float:
        return max(0.0, self._cooldown_until.get(provider, 0.0) - time.time())

    def requests_per_minute(self, provider: str) -> int:
        cutoff = time.time() - 60.0
        times = self._request_times.get(provider, [])
        return sum(1 for t in times if t > cutoff)

    def _parse_retry_after(self, headers: dict[str, str]) -> float:
        """Parse Retry-After header (seconds or HTTP-date)."""
        for key in ("Retry-After", "retry-after", "X-Retry-After"):
            val = headers.get(key, "")
            if not val:
                continue
            try:
                return float(val)
            except ValueError:
                pass
        # Parse X-RateLimit-Reset (unix timestamp)
        for key in ("X-RateLimit-Reset", "x-ratelimit-reset"):
            val = headers.get(key, "")
            if not val:
                continue
            try:
                reset_at = float(val)
                return max(0.0, reset_at - time.time())
            except ValueError:
                pass
        return 0.0

    def get_stats(self) -> dict[str, dict[str, object]]:
        """Return stats for all tracked providers."""
        now = time.time()
        stats: dict[str, dict[str, object]] = {}
        for provider in set(list(self._consecutive_failures.keys()) +
                           list(self._last_success.keys())):
            stats[provider] = {
                "failures": self._consecutive_failures.get(provider, 0),
                "cooldown_remaining": max(0.0, self._cooldown_until.get(provider, 0.0) - now),
                "rpm": self.requests_per_minute(provider),
                "last_success_ago": now - self._last_success.get(provider, 0.0),
            }
        return stats


# Global rate limit tracker shared across all providers
_rate_tracker = RateLimitTracker()


class Translator:
    def translate_batch_cancellable(self, *, check: Callable[[], None], texts: list[str],
                                     source_lang: str, target_lang: str, glossary: Glossary,
                                     context: str = '') -> list[TranslationResult]:
        check()
        if type(self).translate_batch is Translator.translate_batch:
            results = []
            for text in texts:
                check()
                results.append(self.translate(text, source_lang, target_lang, glossary, context))
            check()
            return results
        result = self.translate_batch(texts, source_lang, target_lang, glossary, context)
        check()
        return result

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        raise NotImplementedError

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> list[TranslationResult]:
        return [
            self.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary=glossary,
                context=context,
            )
            for text in texts
        ]

    def check_health(self) -> tuple[bool, str]:
        return True, "Translator ready"


def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _parse_gtx_segments(data: object) -> str:
    segments = data[0]  # type: ignore[index]
    return "".join(
        str(seg[0]) for seg in segments if seg and seg[0] is not None  # type: ignore[union-attr,index]
    )


def _validate_mymemory_response(translated: str, status: object) -> str:
    if status is not None and str(status) not in {"200", "200.0"} and status != 200:
        if str(status) in {"429"} or status == 429:
            raise RateLimitError(f"MyMemory HTTP {status}", retry_after=60.0)
        raise RuntimeError(f"MyMemory error ({status}): {translated[:120]}")
    if "MYMEMORY WARNING" in translated.upper() or "QUERY LENGTH LIMIT" in translated.upper():
        raise RateLimitError(
            "MyMemory daily quota exceeded", retry_after=3600.0
        )
    return translated


class EchoTranslator(Translator):
    """Returns the OCR text unchanged. Useful for OCR and overlay testing."""

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        return TranslationResult(text=text, provider="echo")


class GoogleWebTranslator(Translator):
    """Keyless Google Translate through the public `gtx` endpoint.

    This is the same free endpoint long used by browser extensions. It does
    not need an API key, but it is unofficial: when Google throttles it, the
    translation fails and the fallback chain takes over.
    """

    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    def __init__(self, timeout_seconds: int = 20, retries: int = 1) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        # Keep-alive: reusing one TLS connection saves a handshake on every
        # batched request (the pipeline sends few, larger requests).
        self._connection: http.client.HTTPSConnection | None = None
        self._tracker = _rate_tracker

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return TranslationResult(text="", provider="google")

        translated = self._request(
            _google_source(source_lang),
            _google_target(target_lang),
            cleaned,
        )
        translated = glossary.repair_protected_names(translated)
        return TranslationResult(text=translated, provider="google")

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> list[TranslationResult]:
        return self.translate_batch_cancellable(
            check=lambda: None, texts=texts, source_lang=source_lang,
            target_lang=target_lang, glossary=glossary, context=context,
        )

    def translate_batch_cancellable(self, *, check: Callable[[], None], texts: list[str],
                                     source_lang: str, target_lang: str, glossary: Glossary,
                                     context: str = '') -> list[TranslationResult]:
        check()

        def individually() -> list[TranslationResult]:
            results = []
            for text in texts:
                check()
                results.append(self.translate(text, source_lang, target_lang, glossary, context))
            check()
            return results

        items = [(idx, _clean_text(text)) for idx, text in enumerate(texts)]
        items = [(idx, text) for idx, text in items if text]
        if len(items) < 2:
            return individually()

        joined = "\n".join(text for _, text in items)
        translated = self._request(
            _google_source(source_lang),
            _google_target(target_lang),
            joined,
        )
        parts = translated.split("\n")
        check()
        if len(parts) != len(items):
            # Google merged or split lines; fall back to one request per text.
            return individually()

        results: list[TranslationResult | None] = [None] * len(texts)
        for (idx, _), part in zip(items, parts):
            results[idx] = TranslationResult(
                text=glossary.repair_protected_names(part.strip()),
                provider="google",
            )
        for idx, text in enumerate(texts):
            if results[idx] is None:
                results[idx] = TranslationResult(text="", provider="google")
        return results  # type: ignore[return-value]

    def check_health(self) -> tuple[bool, str]:
        try:
            result = self._request("en", "pt", "hello world")
        except RuntimeError as exc:
            return False, str(exc)
        if not result:
            return False, "Google Translate returned an empty response"
        return True, "Google Translate ready"

    def _request(self, source: str, target: str, text: str) -> str:
        last_error: RuntimeError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._request_once(source, target, text)
            except RateLimitError:
                raise  # throttled: retrying in sub-seconds only makes it worse
            except RuntimeError as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.6 * (attempt + 1))
        raise last_error or RuntimeError("Google Translate request failed")

    def _request_once(self, source: str, target: str, text: str) -> str:
        params = urllib.parse.urlencode(
            {"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text}
        )
        path = f"/translate_a/single?{params}"
        last_error: RuntimeError | None = None
        for attempt in range(2):
            try:
                if self._connection is None:
                    self._connection = http.client.HTTPSConnection(
                        "translate.googleapis.com", timeout=self.timeout_seconds
                    )
                self._connection.request(
                    "GET", path, headers={"User-Agent": self._USER_AGENT}
                )
                response = self._connection.getresponse()
                status = response.status
                # Read headers for rate limit parsing
                headers = dict(response.getheaders())
                body = response.read().decode("utf-8")
            except (http.client.HTTPException, OSError, TimeoutError) as exc:
                # Stale keep-alive connections surface here; drop it and
                # rebuild once before giving up.
                self._connection = None
                self._tracker.record_transient_error("google")
                last_error = RuntimeError(f"Google Translate unreachable: {exc}")
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                raise last_error from exc

            if status == 429 or status >= 500:
                self._connection = None
                # Parse Retry-After header for actual cooldown
                retry_after = self._tracker._parse_retry_after(headers)
                retry_after = max(retry_after, 60.0 if status == 429 else 30.0)
                self._tracker.record_rate_limit("google", retry_after, headers)
                raise RateLimitError(
                    f"Google Translate HTTP {status}", retry_after=retry_after
                )
            if status != 200:
                self._connection = None
                raise RuntimeError(f"Google Translate HTTP {status}")

            try:
                data = json.loads(body)
                translated = _parse_gtx_segments(data)
            except (json.JSONDecodeError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Google Translate returned an unexpected response: {body[:120]}"
                ) from exc
            if not translated.strip():
                raise RuntimeError("Google Translate returned no translation")
            self._tracker.record_success("google")
            return translated.strip()
        raise last_error or RuntimeError("Google Translate request failed")


class MyMemoryTranslator(Translator):
    """Keyless MyMemory translation (api.mymemory.translated.net).

    Free anonymous tier with a per-day character quota. It needs an explicit
    source language, so it only participates when the source is not `auto`.
    """

    _USER_AGENT = "MangaRTTranslator/0.1 (desktop reader)"

    def __init__(self, timeout_seconds: int = 20, retries: int = 1) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._tracker = _rate_tracker

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return TranslationResult(text="", provider="mymemory")

        source = _google_source(source_lang)
        if source in {"auto", ""}:
            raise RuntimeError(
                "MyMemory requires an explicit source language, not auto"
            )
        target = _google_target(target_lang)

        translated = self._request(source, target, cleaned)
        translated = glossary.repair_protected_names(translated)
        return TranslationResult(text=translated, provider="mymemory")

    def check_health(self) -> tuple[bool, str]:
        try:
            result = self._request("en", "pt", "hello")
        except RuntimeError as exc:
            return False, str(exc)
        if not result:
            return False, "MyMemory returned an empty response"
        return True, "MyMemory ready"

    def _request(self, source: str, target: str, text: str) -> str:
        last_error: RuntimeError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._request_once(source, target, text)
            except RateLimitError:
                raise
            except RuntimeError as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.6 * (attempt + 1))
        raise last_error or RuntimeError("MyMemory request failed")

    def _request_once(self, source: str, target: str, text: str) -> str:
        params = urllib.parse.urlencode({"q": text, "langpair": f"{source}|{target}"})
        url = f"https://api.mymemory.translated.net/get?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                headers = dict(response.getheaders())
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            headers = dict(exc.headers) if hasattr(exc, "headers") else {}
            if exc.code == 429:
                retry_after = self._tracker._parse_retry_after(headers)
                retry_after = max(retry_after, 60.0)
                self._tracker.record_rate_limit("mymemory", retry_after, headers)
                raise RateLimitError('MyMemory HTTP 429', retry_after=retry_after) from exc
            raise RuntimeError(f"MyMemory HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            self._tracker.record_transient_error("mymemory")
            raise RuntimeError(f"MyMemory unreachable: {exc}") from exc

        translated = str(data.get("responseData", {}).get("translatedText", "") or "")
        status = data.get("responseStatus")
        translated = _validate_mymemory_response(translated, status)
        if not translated.strip():
            raise RuntimeError("MyMemory returned no translation")
        self._tracker.record_success("mymemory")
        return translated.replace("\\'", "'").strip()


class GeminiTranslator(Translator):
    """Google Gemini free-tier translation. Context-aware LLM translation:
    understands scene context, protects glossary names and produces natural
    pt-BR. Requires a free API key (aistudio.google.com/apikey)."""

    API_HOST = "generativelanguage.googleapis.com"
    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 25,
        model: str = "",
        min_interval: float = 1.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.model = model.strip() or self.DEFAULT_MODEL
        self.min_interval = float(min_interval)
        self._last_request_at = 0.0
        self._tracker = _rate_tracker

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_request_at)
        if wait > 1.0:
            # Don't block the pipeline on the pacing — yield to the next
            # provider in the chain and come back when our turn arrives.
            self._last_request_at = time.time()
            raise RateLimitError(
                f"Gemini pacing ({wait:.1f}s)", retry_after=wait
            )
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.time()

    def _prompt(self, text: str, source_lang: str, target_lang: str, glossary: Glossary) -> str:
        lines = [
            f"Voce e um tradutor profissional de mangas/webtoons para {target_lang}.",
            f"Traduza o trecho abaixo de {source_lang} para {target_lang}.",
            "Respeite concordancia, tempo verbal e tom emocional do balao.",
            "Nomes proprios e expressoes recorrentes devem ficar naturais.",
        ]
        block = glossary.prompt_block()
        if block:
            lines.append(block)
        lines.append("Responda SOMENTE com a traducao, sem comentarios.")
        lines.append(f"Trecho: {text}")
        return "\n".join(lines)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return TranslationResult(text="", provider="gemini")
        source = "the original language" if source_lang in {"", "auto"} else source_lang
        translated = self._request(
            self._prompt(cleaned, source, target_lang, glossary)
        )
        translated = glossary.repair_protected_names(translated)
        return TranslationResult(text=translated, provider="gemini")

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> list[TranslationResult]:
        cleaned = [(idx, _clean_text(t)) for idx, t in enumerate(texts)]
        cleaned = [(idx, t) for idx, t in cleaned if t]
        if len(cleaned) < 2:
            return super().translate_batch(texts, source_lang, target_lang, glossary, context)

        # One request with the whole page: the model translates every balloon
        # with full scene context (better pronouns, names and flow).
        numbered = "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(cleaned))
        prompt = "\n".join(
            [
                "Voce e um tradutor profissional de mangas/webtoons para "
                f"{target_lang}.",
                "Os itens abaixo sao baloes na ordem de leitura. Traduza cada "
                "item mantendo concordancia, tempo verbal, tom emocional e "
                "coerencia entre os baloes (cena unica).",
                "Mantenha EXATAMENTE a mesma quantidade de itens, um por linha, "
                "no formato 'numero. traducao'.",
            ]
        )
        block = glossary.prompt_block()
        if block:
            prompt += "\n" + block
        prompt += "\n\n" + numbered

        translated = self._request(prompt)
        parts = translated.split("\n")
        results: list[TranslationResult | None] = [None] * len(texts)
        parsed: dict[int, str] = {}
        for part in parts:
            part = part.strip()
            if not part:
                continue
            marker = part.split(".", 1)[0].strip()
            if marker.isdigit() and int(marker) >= 1:
                response_index = int(marker) - 1
                body = part.split(".", 1)[1].strip() if "." in part else part
                if 0 <= response_index < len(cleaned):
                    original_index = cleaned[response_index][0]
                    parsed[original_index] = body
        if len(parsed) != len(cleaned):
            return super().translate_batch(texts, source_lang, target_lang, glossary, context)

        for idx, body in parsed.items():
            results[idx] = TranslationResult(
                text=glossary.repair_protected_names(body),
                provider="gemini",
            )
        for idx, _ in enumerate(texts):
            if results[idx] is None:
                results[idx] = TranslationResult(text="", provider="gemini")
        return results  # type: ignore[return-value]

    def check_health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini provider selected, but no API key is configured"
        try:
            result = self._request("Responda apenas: ok")
        except RuntimeError as exc:
            return False, str(exc)
        if not result:
            return False, "Gemini returned an empty response"
        return True, f"Gemini ready ({self.model})"

    def _request(self, prompt: str) -> str:
        self._throttle()
        path = f"/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 700},
        }
        request = urllib.request.Request(
            f"https://{self.API_HOST}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:160]
            headers = dict(exc.headers) if hasattr(exc, "headers") else {}
            if exc.code == 429:
                retry_after = self._tracker._parse_retry_after(headers)
                retry_after = max(retry_after, 30.0)
                self._tracker.record_rate_limit("gemini", retry_after, headers)
                raise RateLimitError(
                    "Gemini free-tier quota hit", retry_after=retry_after
                ) from exc
            if exc.code == 503:
                retry_after = max(self._tracker._parse_retry_after(headers), 12.0)
                self._tracker.record_rate_limit("gemini", retry_after, headers)
                raise RateLimitError(
                    "Gemini under high demand", retry_after=retry_after
                ) from exc
            if exc.code == 400 and "API key" in detail:
                raise RuntimeError(f"Gemini API key invalid: {detail}") from exc
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            self._tracker.record_transient_error("gemini")
            raise RuntimeError(f"Gemini unreachable: {exc}") from exc

        try:
            data = json.loads(body)
            translated = str(
                data["candidates"][0]["content"]["parts"][0]["text"]
            ).strip()
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Gemini returned an unexpected response: {body[:160]}"
            ) from exc
        if not translated:
            raise RuntimeError("Gemini returned no translation")
        self._tracker.record_success("gemini")
        return translated


class LibreTranslateProvider(Translator):
    """LibreTranslate (open source, Argos models). Works with the public
    instance (free API key at portal.libretranslate.com) or with your own
    Docker container (unlimited and offline):
        docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
    then set libre_url = "http://localhost:5000" in config.toml."""

    def __init__(
        self,
        libre_url: str = "https://libretranslate.org",
        api_key: str = "",
        timeout_seconds: int = 20,
    ) -> None:
        self.libre_url = (libre_url or "https://libretranslate.org").strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self._tracker = _rate_tracker

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return TranslationResult(text="", provider="libre")
        translated = self._request(cleaned, _libre_lang(source_lang), _libre_lang(target_lang))
        translated = glossary.repair_protected_names(translated)
        return TranslationResult(text=translated, provider="libre")

    def check_health(self) -> tuple[bool, str]:
        try:
            result = self._request("hello")
        except RuntimeError as exc:
            return False, str(exc)
        return True, f"LibreTranslate ready ({self.libre_url})"

    def _request(self, text: str, source: str, target: str) -> str:
        payload: dict[str, object] = {"q": text, "source": source, "target": target, "format": "text"}
        if self.api_key:
            payload["api_key"] = self.api_key
        request = urllib.request.Request(
            f"{self.libre_url}/translate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:160]
            headers = dict(exc.headers) if hasattr(exc, "headers") else {}
            if exc.code == 429:
                retry_after = self._tracker._parse_retry_after(headers)
                retry_after = max(retry_after, 30.0)
                self._tracker.record_rate_limit("libre", retry_after, headers)
                raise RateLimitError(
                    f"LibreTranslate rate limited: {detail}", retry_after=retry_after
                ) from exc
            raise RuntimeError(f"LibreTranslate HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            self._tracker.record_transient_error("libre")
            raise RuntimeError(f"LibreTranslate unreachable: {exc}") from exc
        translated = str(data.get("translatedText", "") or "").strip()
        if not translated:
            raise RuntimeError("LibreTranslate returned no translation")
        if "API KEY" in translated.upper() and "INVALID" in translated.upper():
            raise RuntimeError(f"LibreTranslate: {translated}")
        self._tracker.record_success("libre")
        return translated


def _libre_lang(lang: str) -> str:
    key = (lang or "").strip().lower()
    if key in {"", "auto"}:
        return "auto"
    aliases = {"en": "en", "english": "en", "japan": "ja", "ja": "ja", "japanese": "ja", "ko": "ko", "korean": "ko", "pt": "pt", "pt-br": "pt", "ptbr": "pt"}
    return aliases.get(key, key)


class FallbackChain(Translator):
    """Tries each provider in order and returns the first success.

    After a provider fails it cools down for a while (longer for rate limits
    and quotas), so a throttled Google does not eat a wasted round trip on
    every single bubble. Uses the global RateLimitTracker for exponential
    backoff with jitter.
    """

    _TRANSIENT_COOLDOWN = 10.0

    def __init__(self, providers: list[Translator]) -> None:
        if not providers:
            raise ValueError("FallbackChain needs at least one provider")
        self.providers = providers
        self._cooldown_until = [0.0] * len(providers)
        self._tracker = _rate_tracker

    def _available(self) -> list[tuple[int, Translator]]:
        now = time.time()
        return [
            (index, provider)
            for index, provider in enumerate(self.providers)
            if self._cooldown_until[index] <= now
        ]

    def _register_failure(self, index: int, exc: Exception) -> None:
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(exc, RateLimitError) and retry_after:
            self._cooldown_until[index] = time.time() + float(retry_after)
        else:
            self._cooldown_until[index] = time.time() + self._TRANSIENT_COOLDOWN

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        last_error: Exception | None = None
        for index, provider in self._available():
            try:
                result = provider.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary,
                    context=context,
                )
                return result
            except Exception as exc:
                self._register_failure(index, exc)
                provider_name = type(provider).__name__
                print(
                    f"[translator] {provider_name} failed: {exc}",
                    flush=True,
                )
                last_error = exc
        raise last_error or RuntimeError("No translation provider available")

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> list[TranslationResult]:
        return self.translate_batch_cancellable(
            check=lambda: None, texts=texts, source_lang=source_lang,
            target_lang=target_lang, glossary=glossary, context=context,
        )

    def translate_batch_cancellable(self, *, check: Callable[[], None], texts: list[str],
                                     source_lang: str, target_lang: str, glossary: Glossary,
                                     context: str = '') -> list[TranslationResult]:
        last_error: Exception | None = None
        for index, provider in self._available():
            check()
            try:
                return provider.translate_batch_cancellable(
                    check=check,
                    texts=texts,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary,
                    context=context,
                )
            except WorkCancelled:
                raise
            except Exception as exc:
                self._register_failure(index, exc)
                provider_name = type(provider).__name__
                print(
                    f"[translator] {provider_name} batch failed: {exc}",
                    flush=True,
                )
                last_error = exc
        raise last_error or RuntimeError("No translation provider available")

    def check_health(self) -> tuple[bool, str]:
        messages: list[str] = []
        stats = self._tracker.get_stats()
        for provider in self.providers:
            checker = getattr(provider, "check_health", None)
            if not callable(checker):
                return True, "Translator ready"
            ok, message = checker()
            if ok:
                # Log tracker stats for debugging
                pname = type(provider).__name__
                if pname in stats:
                    s = stats[pname]
                    print(
                        f"[translator] {pname}: {message} "
                        f"(failures={s['failures']}, rpm={s['rpm']})",
                        flush=True,
                    )
                return True, message
            messages.append(message)
        return False, " | ".join(messages)


class DeepLTranslator(Translator):
    """DeepL REST translation. Requires an API key (free tier supported)."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        free: bool = True,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.free = free
        self._tracker = _rate_tracker

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return TranslationResult(text="", provider="deepl")

        payload: dict[str, object] = {
            "text": [cleaned],
            "target_lang": _deepl_target(target_lang),
        }
        source = _deepl_source(source_lang)
        if source:
            payload["source_lang"] = source

        data = self._request("POST", "/v2/translate", payload)
        try:
            translated = str(data["translations"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"DeepL returned an unexpected response: {data}") from exc

        translated = glossary.repair_protected_names(translated)
        return TranslationResult(text=translated, provider="deepl")

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> list[TranslationResult]:
        cleaned = [_clean_text(text) for text in texts]
        if not cleaned:
            return []
        if len(cleaned) == 1:
            return super().translate_batch(texts, source_lang, target_lang, glossary, context)

        payload: dict[str, object] = {
            "text": cleaned,
            "target_lang": _deepl_target(target_lang),
        }
        source = _deepl_source(source_lang)
        if source:
            payload["source_lang"] = source

        data = self._request("POST", "/v2/translate", payload)
        try:
            translated_list = [str(item["text"]) for item in data["translations"]]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"DeepL returned an unexpected response: {data}") from exc
        if len(translated_list) != len(texts):
            return super().translate_batch(texts, source_lang, target_lang, glossary, context)
        return [
            TranslationResult(
                text=glossary.repair_protected_names(part.strip()),
                provider="deepl",
            )
            for part in translated_list
        ]

    def check_health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "DeepL provider selected, but no API key is configured"
        try:
            self._request("GET", "/v2/usage", None)
        except RuntimeError as exc:
            return False, str(exc)
        return True, "DeepL ready"

    def _request(self, method: str, path: str, payload: dict[str, object] | None) -> dict:
        host = "https://api-free.deepl.com" if self.free else "https://api.deepl.com"
        request = urllib.request.Request(
            f"{host}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={
                "Authorization": f"DeepL-Auth-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                headers = dict(response.getheaders())
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            headers = dict(exc.headers) if hasattr(exc, "headers") else {}
            if exc.code == 429:
                retry_after = self._tracker._parse_retry_after(headers)
                retry_after = max(retry_after, 60.0)
                self._tracker.record_rate_limit("deepl", retry_after, headers)
                raise RateLimitError(
                    f"DeepL throttled (HTTP 429)", retry_after=retry_after
                ) from exc
            raise RuntimeError(
                f"DeepL request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            self._tracker.record_transient_error("deepl")
            raise RuntimeError(f"DeepL unreachable: {exc}") from exc

        if not body:
            return {}
        self._tracker.record_success("deepl")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepL returned invalid JSON: {body[:200]}") from exc


class ResilientTranslator(Translator):
    """Wraps a primary provider and falls back to echo when it fails, so a
    network hiccup never kills the capture/OCR/overlay loop."""

    def __init__(self, primary: Translator, fallback: Translator) -> None:
        self.primary = primary
        self.fallback = fallback

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Glossary,
        context: str = "",
    ) -> TranslationResult:
        try:
            return self.primary.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary=glossary,
                context=context,
            )
        except Exception as exc:
            print(f"[translator] primary failed ({exc}); using echo fallback", flush=True)
            return self.fallback.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary=glossary,
                context=context,
            )

    def check_health(self) -> tuple[bool, str]:
        checker = getattr(self.primary, "check_health", None)
        if callable(checker):
            return checker()
        return True, "Translator ready"


def create_translator(config: object, echo: bool = False) -> Translator:
    """Build the configured translator. `config` is an AppConfig whose
    `translation` section selects the provider."""
    if echo:
        return EchoTranslator()

    translation = getattr(config, "translation")
    provider = str(getattr(translation, "provider", "google") or "google").strip().lower()
    timeout = translation.timeout_seconds

    if provider == "google":
        # Keyless chain: Google first, MyMemory when Google throttles us,
        # and echo as the last resort so the pipeline never stalls.
        return FallbackChain(
            [
                GoogleWebTranslator(timeout_seconds=max(1, min(timeout, 6)), retries=0),
                MyMemoryTranslator(timeout_seconds=max(1, min(timeout, 6)), retries=0),
                EchoTranslator(),
            ]
        )
    if provider == "gemini":
        key = str(getattr(translation, "gemini_api_key", "") or "").strip()
        if not key:
            raise RuntimeError(
                "Gemini selecionado: crie sua chave grátis em "
                "aistudio.google.com/apikey e cole no campo Chave de API"
            )
        # Quality first (Gemini LLM), but never let a congested LLM stall the
        # reading flow: the Gemini timeout is capped well below the global
        # one, so when the free tier is slow it falls to Google MT (~0.5-1s)
        # instead of holding the async translation thread for 15-20s. Echo is
        # the last resort. No waiting, no blanks.
        return FallbackChain(
            [
                GeminiTranslator(
                    api_key=key,
                    timeout_seconds=max(3, min(int(timeout), 4)),
                    model=str(getattr(translation, "gemini_model", "") or ""),
                    min_interval=float(
                        getattr(translation, "gemini_min_interval", 1.0) or 1.0
                    ),
                ),
                GoogleWebTranslator(timeout_seconds=timeout),
                EchoTranslator(),
            ]
        )
    if provider == "libre":
        return ResilientTranslator(
            LibreTranslateProvider(
                libre_url=str(getattr(translation, "libre_url", "") or "https://libretranslate.org"),
                api_key=str(getattr(translation, "libre_api_key", "") or ""),
                timeout_seconds=max(timeout, 25),
            ),
            EchoTranslator(),
        )
    if provider == "deepl":
        if not str(getattr(translation, "deepl_api_key", "") or "").strip():
            raise RuntimeError("DeepL provider selected, but no API key is configured")
        return ResilientTranslator(
            DeepLTranslator(
                api_key=str(translation.deepl_api_key),
                timeout_seconds=timeout,
                free=bool(getattr(translation, "deepl_free", True)),
            ),
            EchoTranslator(),
        )
    if provider == "echo":
        return EchoTranslator()
    raise RuntimeError(
        f"Unknown translation provider {provider!r}. "
        "Use one of: google, deepl, echo."
    )


def get_rate_stats() -> dict[str, dict[str, object]]:
    """Return rate limit stats for all providers (for diagnostics)."""
    return _rate_tracker.get_stats()


def _google_source(source_lang: str) -> str:
    key = (source_lang or "").strip().lower()
    if key in {"", "auto", "autodetect"}:
        return "auto"
    aliases = {
        "en": "en",
        "english": "en",
        "japan": "ja",  # PaddleOCR language name
        "ja": "ja",
        "japanese": "ja",
        "ko": "ko",
        "korean": "ko",
        "zh": "zh-CN",
        "chinese_cht": "zh-TW",
        "ch_sim": "zh-CN",
    }
    return aliases.get(key, key)


def _google_target(target_lang: str) -> str:
    key = (target_lang or "").strip().lower()
    aliases = {
        "pt": "pt",
        "pt-br": "pt",
        "ptbr": "pt",
        "brazilian portuguese": "pt",
        "en": "en",
        "ja": "ja",
        "es": "es",
    }
    return aliases.get(key, key)


def _deepl_source(source_lang: str) -> str:
    key = (source_lang or "").strip().lower()
    if key in {"", "auto", "autodetect"}:
        return ""
    aliases = {
        "en": "EN",
        "english": "EN",
        "japan": "JA",  # PaddleOCR language name
        "ja": "JA",
        "japanese": "JA",
        "ko": "KO",
        "korean": "KO",
        "zh": "ZH",
        "pt": "PT-PT",
    }
    return aliases.get(key, key.upper())


def _deepl_target(target_lang: str) -> str:
    key = (target_lang or "").strip().lower()
    aliases = {
        "pt": "PT-BR",
        "pt-br": "PT-BR",
        "ptbr": "PT-BR",
        "en": "EN-US",
        "en-us": "EN-US",
        "en-gb": "EN-GB",
        "ja": "JA",
        "es": "ES",
    }
    return aliases.get(key, key.upper())
