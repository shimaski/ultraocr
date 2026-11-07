from __future__ import annotations

import logging
import time

from manga_rt_translator.glossary import Glossary
from manga_rt_translator.translator import (
    CircuitBreaker,
    CircuitState,
    DailyQuota,
    EchoTranslator,
    FallbackChain,
    RateLimitTracker,
    TokenBucket,
    TranslationResult,
    Translator,
    ProviderMetrics,
)


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3, open_duration_s=20)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.can_attempt()


def test_circuit_breaker_half_open_after_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(failure_threshold=1, open_duration_s=5)
    breaker.record_failure()
    now[0] += 5
    assert breaker.can_attempt()
    assert breaker.state is CircuitState.HALF_OPEN


def test_circuit_breaker_closes_after_success_probes():
    breaker = CircuitBreaker(failure_threshold=1, open_duration_s=0, success_threshold=2)
    breaker.record_failure()
    assert breaker.begin_attempt()
    breaker.record_success()
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.begin_attempt()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_circuit_breaker_reopens_on_failed_probe(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(failure_threshold=1, open_duration_s=0)
    breaker.record_failure()
    assert breaker.begin_attempt()
    opened_before = breaker.opened_at
    now[0] += 1
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.opened_at > opened_before


def test_token_bucket_refills_over_time(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    bucket = TokenBucket(rate=2, burst=2)
    assert bucket.try_consume(2)
    assert not bucket.try_consume()
    now[0] += 0.5
    assert bucket.try_consume()


def test_token_bucket_wait_time(monkeypatch):
    now = [1.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    bucket = TokenBucket(rate=2, burst=1)
    assert bucket.try_consume()
    assert bucket.wait_time() == 0.5


def test_score_prefers_high_quality_when_all_available():
    tracker = RateLimitTracker()
    assert tracker.provider_score("gemini") > tracker.provider_score("google")
    assert tracker.provider_score("deepl") > tracker.provider_score("google")
    assert tracker.provider_score("google") > tracker.provider_score("libre")


def test_score_excludes_open_circuit():
    tracker = RateLimitTracker()
    tracker._breakers["google"].state = CircuitState.OPEN
    tracker._breakers["google"].opened_at = time.monotonic()
    assert tracker.provider_score("google") == -1


def test_score_penalizes_low_quota():
    tracker = RateLimitTracker()
    quota = tracker._quotas["gemini"]
    full_score = tracker.provider_score("gemini")
    quota.count = int(quota.daily_limit * 0.9)
    low_quota_score = tracker.provider_score("gemini")
    assert low_quota_score < full_score


def test_daily_quota_resets_at_utc_hour(monkeypatch):
    keys = iter(("2026-10-14", "2026-10-15"))
    quota = DailyQuota(daily_limit=10, period_key="2026-10-14")
    monkeypatch.setattr(quota, "_period_key", lambda: next(keys))
    quota.count = 9
    assert quota.remaining() == 1
    assert quota.remaining() == 10
    assert quota.count == 0


def test_daily_quota_chars_mode():
    quota = DailyQuota(daily_limit=100, chars=True)
    assert quota.consume(80)
    assert quota.remaining() == 20
    assert not quota.consume(21)


def test_persist_provider_state_roundtrip(tmp_path):
    db_path = tmp_path / "cache.sqlite3"
    first = RateLimitTracker(db_path=db_path, persist_interval_s=0)
    first.before_request("google", text_length=12)
    first.after_success("google", latency_ms=123, text_length=12)
    expected_latency = first._avg_latency_ms["google"]
    first.close()
    second = RateLimitTracker(db_path=db_path)
    assert second._avg_latency_ms["google"] == expected_latency
    assert second._quotas["google"].count == 1
    second.close()


class _NamedTranslator(Translator):
    def __init__(self, name: str, output: str) -> None:
        self.rate_limit_provider = name
        self.output = output

    def translate(self, text, source_lang, target_lang, glossary, context=""):
        return TranslationResult(self.output, self.rate_limit_provider)


def test_fallback_chain_uses_score():
    tracker = RateLimitTracker()
    tracker._breakers["google"].state = CircuitState.OPEN
    tracker._breakers["google"].opened_at = time.monotonic()
    chain = FallbackChain(
        [_NamedTranslator("google", "Google"), _NamedTranslator("gemini", "Gemini"),
         EchoTranslator()], tracker=tracker
    )
    result = chain.translate("Hello", "en", "pt-BR", Glossary.default())
    assert result.text == "Gemini"


def test_fallback_chain_drops_to_echo_after_cycles(monkeypatch):
    tracker = RateLimitTracker()
    for name in ("google", "gemini"):
        tracker._breakers[name].state = CircuitState.OPEN
        tracker._breakers[name].opened_at = time.monotonic()
    chain = FallbackChain(
        [_NamedTranslator("google", "Google"), _NamedTranslator("gemini", "Gemini"),
         EchoTranslator()], tracker=tracker
    )
    monkeypatch.setattr(chain, "_wait_for_provider", lambda: None)
    result = chain.translate("Hello", "en", "pt-BR", Glossary.default())
    assert result.provider == "echo"
    assert result.text == "Hello"


def test_metrics_count_requests_by_status():
    metrics = ProviderMetrics("google")
    for _ in range(5):
        metrics.record_request(10, "ok")
    for _ in range(3):
        metrics.record_request(20, "429")
    for _ in range(2):
        metrics.record_request(30, "error")
    assert (metrics.requests_ok, metrics.requests_429, metrics.requests_error) == (5, 3, 2)
    assert metrics.total_requests == 10


def test_rpm_current_window(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    metrics = ProviderMetrics("google")
    metrics.request_timestamps.extend([10.0] * 10 + [70.0] * 5)
    assert metrics.rpm_current() == 5
    metrics.request_timestamps.extend([90.0] * 10)
    assert metrics.rpm_current() == 15


def test_rpm_peak_sampling(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("manga_rt_translator.translator.time.monotonic", lambda: now[0])
    metrics = ProviderMetrics("google")
    metrics.request_timestamps.extend([100.0] * 8)
    metrics.rpm_peak = max(metrics.rpm_peak, metrics.rpm_current())
    now[0] += 1
    metrics.request_timestamps.extend([101.0] * 3)
    metrics.rpm_peak = max(metrics.rpm_peak, metrics.rpm_current())
    assert metrics.rpm_peak == 11


def test_avg_latency_ema():
    tracker = RateLimitTracker()
    tracker.after_success("google", 100)
    tracker.after_success("google", 200)
    assert tracker._avg_latency_ms["google"] == 184.0


def test_p95_latency():
    metrics = ProviderMetrics("google")
    for latency in range(100):
        metrics.record_request(latency, "ok")
    assert metrics.p95_latency_ms() == 95


def test_session_summary_format(caplog):
    tracker = RateLimitTracker()
    tracker.after_success("google", 125)
    tracker.record_cache_activity(3, 1)
    with caplog.at_level(logging.INFO, logger="manga_rt_translator.translator"):
        summary = tracker.session_summary()
    output = caplog.text
    assert "session_summary duration_min=" in output
    assert "requests=1 ok=1 429=0 error=0" in output
    assert "rpm_peak=" in output and "latency_p95_ms=" in output
    assert "cache_hit_ratio=75.0% falls_to_echo=0 total_requests=1" in output
    assert summary["total_requests"] == 1


def test_breaker_transition_logged(caplog):
    tracker = RateLimitTracker(provider_configs={"google": {
        "failure_threshold": 1, "open_duration_s": 60,
    }})
    with caplog.at_level(logging.INFO, logger="manga_rt_translator.translator"):
        tracker._breakers["google"].record_failure()
    assert "breaker closed -> open reason=failures" in caplog.text


def test_pick_provider_logs_candidates(caplog):
    tracker = RateLimitTracker()
    with caplog.at_level(logging.INFO, logger="manga_rt_translator.translator"):
        picked = tracker.pick_provider(["google", "gemini"])
    assert picked == "gemini"
    assert "picked gemini score=" in caplog.text
    assert "candidates: google=" in caplog.text and "gemini=" in caplog.text


def test_echo_fallback_warning(monkeypatch, caplog):
    tracker = RateLimitTracker()
    for name in ("google", "gemini"):
        tracker._breakers[name].state = CircuitState.OPEN
        tracker._breakers[name].opened_at = time.monotonic()
    chain = FallbackChain(
        [_NamedTranslator("google", "Google"), _NamedTranslator("gemini", "Gemini"),
         EchoTranslator()], tracker=tracker, max_cycles=3
    )
    monkeypatch.setattr(chain, "_wait_for_provider", lambda: None)
    with caplog.at_level(logging.WARNING, logger="manga_rt_translator.translator"):
        chain.translate("Hello", "en", "pt-BR", Glossary.default())
    assert "all providers unavailable after 3 cycles, falling back to echo" in caplog.text
