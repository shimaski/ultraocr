from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from manga_rt_translator.config import AppConfig
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import Frame, Rect, TextRegion
from manga_rt_translator.reader_runtime import ReaderRuntime
from manga_rt_translator.translator import (
    EchoTranslator, FallbackChain, GoogleWebTranslator, TranslationResult, Translator,
)
from manga_rt_translator.work_queue import LatestFrameQueue, ViewState, WorkCancelled
from tests.test_pipeline_integration import _build_pipeline


def frame(index):
    return Frame(np.full((180, 240, 3), index, np.uint8), time.time(), index)


def test_latest_mailbox_keeps_frame_and_request_together():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(1), force=True)
    first = mailbox.get()
    mailbox.put(frame(2), force=True)
    mailbox.put(frame(3), force=True)
    latest = mailbox.get()
    assert first.token.current() is False
    assert latest.frame.index == 3 and latest.force and latest.token.current()
    assert mailbox.dropped == 1
    assert not first.token.commit(lambda: pytest.fail('stale publication'))


def test_unforced_latest_frame_keeps_pending_full_read():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(1), force=True)
    mailbox.put(frame(2))

    request = mailbox.get()

    assert request.frame.index == 2
    assert request.force
    assert not request.incremental
    assert request.token.current()


def test_unforced_latest_frame_keeps_pending_inspection():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(1), inspect=True)
    mailbox.put(frame(2))

    request = mailbox.get()

    assert request.frame.index == 2
    assert request.inspect
    assert not request.force


def test_coalescing_incremental_frame_promotes_to_full_read():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(1), incremental=True)
    mailbox.put(frame(2))

    request = mailbox.get()

    assert request.frame.index == 2
    assert request.force
    assert not request.incremental


def test_in_flight_request_remains_visible_to_scroll_invalidation():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(1), force=True)
    request = mailbox.get()

    assert mailbox.has_work()
    mailbox.invalidate()
    assert mailbox.has_work()

    mailbox.complete(request)
    assert not mailbox.has_work()


def test_incremental_frame_is_prioritized_without_changing_its_processing_mode():
    state = ViewState()
    mailbox = LatestFrameQueue(state)
    mailbox.put(frame(8), force=True, incremental=True)

    request = mailbox.get()

    assert request.force
    assert request.incremental
    assert request.token.current()


def test_scroll_invalidation_records_when_queued_work_was_interrupted(tmp_path):
    config = AppConfig()
    config.cache.path = str(tmp_path / 'cache.sqlite')
    overlay = RecordingOverlay()
    runtime = ReaderRuntime(config, Glossary.default(), overlay, lambda text: None, False)
    runtime.frame_queue.put(frame(1), force=True)

    runtime.invalidate_view(clear_overlay=False, scrolling=True)

    assert runtime.needs_full_refresh_after_scroll
    assert not runtime.frame_queue.has_work()


class RecordingOverlay:
    def __init__(self):
        self.view = ViewState()
        self.translated = threading.Event()
        self.history = []

    def update(self, items):
        self.history.extend((item.source_text, item.text) for item in items)
        if any(item.source_text != item.text for item in items):
            self.translated.set()

    def stop(self):
        pass


class RecordingTranslator(Translator):
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def translate_batch(self, texts, source_lang, target_lang, glossary, context=''):
        self.calls.append(list(texts))
        assert target_lang == 'pt-BR'
        self.started.set()
        if self.block and len(self.calls) == 1:
            assert self.release.wait(5)
        return [TranslationResult(text=f'Traduzido: {t}', provider='test') for t in texts]


class BlockingOcr:
    detects_text = True

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize_cancellable(self, image, regions, check):
        index = int(image[0, 0, 0])
        self.calls.append(index)
        if len(self.calls) == 1:
            self.started.set()
            assert self.release.wait(5)
        check()
        return [TextRegion(Rect(20, 30, 100, 30), f'Frame {index}', .99,
                           'en', f'hash-{index}')]


def test_scroll_during_ocr_drops_old_and_pending_frames(tmp_path, monkeypatch):
    import manga_rt_translator.reader_runtime as module
    ocr, translator, overlay = BlockingOcr(), RecordingTranslator(), RecordingOverlay()
    monkeypatch.setattr(module, 'create_ocr_engine', lambda config: ocr)
    monkeypatch.setattr(module, 'create_translator', lambda config, echo: translator)
    config = AppConfig()
    config.cache.path = str(tmp_path / 'cache.sqlite')
    runtime = ReaderRuntime(config, Glossary.default(), overlay, lambda text: None, False)
    runtime.start()
    try:
        runtime.push_frame(frame(1).image, force_analyze=True)
        assert ocr.started.wait(5)
        runtime.invalidate_view(scrolling=True)
        assert runtime.needs_full_refresh_after_scroll
        runtime.push_frame(frame(2).image, force_analyze=True)
        runtime.push_frame(frame(3).image, force_analyze=True)
        runtime.mark_scroll_refresh_submitted()
        assert not runtime.needs_full_refresh_after_scroll
        ocr.release.set()
        assert overlay.translated.wait(5)
        assert ocr.calls == [1, 3]
        assert translator.calls == [['Frame 3']]
        assert all(source == 'Frame 3' for source, _ in overlay.history)
    finally:
        ocr.release.set()
        runtime.stop(5)
    assert not runtime.worker.is_alive()
    assert not runtime.last_error


def test_scroll_during_translation_requests_full_view_retry(tmp_path, monkeypatch):
    import manga_rt_translator.reader_runtime as module

    class ImmediateOcr:
        detects_text = True

        def recognize_cancellable(self, image, regions, check):
            check()
            index = int(image[0, 0, 0])
            return [TextRegion(Rect(20, 30, 100, 30), f'Frame {index}', .99,
                               'en', f'hash-{index}')]

    translator, overlay = RecordingTranslator(), RecordingOverlay()
    translator.block = True
    monkeypatch.setattr(module, 'create_ocr_engine', lambda config: ImmediateOcr())
    monkeypatch.setattr(module, 'create_translator', lambda config, echo: translator)
    config = AppConfig()
    config.cache.path = str(tmp_path / 'cache.sqlite')
    runtime = ReaderRuntime(config, Glossary.default(), overlay, lambda text: None, False)
    runtime.start()
    try:
        runtime.push_frame(frame(1).image, force_analyze=True)
        assert translator.started.wait(5)

        runtime.invalidate_view(clear_overlay=False, scrolling=True)
        assert runtime.needs_full_refresh_after_scroll
        runtime.push_frame(frame(2).image, force_analyze=True)
        runtime.mark_scroll_refresh_submitted()
        translator.release.set()

        assert overlay.translated.wait(5)
        assert translator.calls == [['Frame 1'], ['Frame 2']]
        assert overlay.history == [('Frame 2', 'Traduzido: Frame 2')]
    finally:
        translator.release.set()
        runtime.stop(5)
    assert not runtime.worker.is_alive()
    assert not runtime.last_error


def test_switching_provider_does_not_wait_for_or_publish_old_request(tmp_path, monkeypatch):
    import copy
    import manga_rt_translator.reader_runtime as module

    class ImmediateOcr:
        detects_text = True

        def recognize_cancellable(self, image, regions, check):
            check()
            index = int(image[0, 0, 0])
            return [TextRegion(Rect(20, 30, 100, 30), f'Frame {index}', .99,
                               'en', f'hash-{index}')]

    old_translator, new_translator = RecordingTranslator(), RecordingTranslator()
    old_translator.block = True
    translators = {'google': old_translator, 'gemini': new_translator}
    overlay = RecordingOverlay()
    monkeypatch.setattr(module, 'create_ocr_engine', lambda config: ImmediateOcr())
    monkeypatch.setattr(
        module, 'create_translator',
        lambda config, echo: translators[config.translation.provider],
    )
    config = AppConfig()
    config.cache.path = str(tmp_path / 'cache.sqlite')
    runtime = ReaderRuntime(config, Glossary.default(), overlay, lambda text: None, False)
    runtime.start()
    try:
        runtime.push_frame(frame(1).image, force_analyze=True)
        assert old_translator.started.wait(5)

        new_config = copy.deepcopy(config)
        new_config.translation.provider = 'gemini'
        runtime.replace_translation_config(new_config, echo_translator=False)
        runtime.push_frame(frame(2).image, force_analyze=True)

        assert overlay.translated.wait(5)
        assert new_translator.calls == [['Frame 2']]
        assert overlay.history == [('Frame 2', 'Traduzido: Frame 2')]
        old_translator.release.set()
        time.sleep(.05)
        assert overlay.history == [('Frame 2', 'Traduzido: Frame 2')]
    finally:
        old_translator.release.set()
        runtime.stop(5)
    assert not runtime.worker.is_alive()
    assert not runtime.last_error


def test_slow_translation_cannot_paint_or_queue_obsolete_pages(tmp_path):
    translator = RecordingTranslator()
    translator.block = True
    _, pipeline, _, memory = _build_pipeline(tmp_path, translator=translator)
    overlay = RecordingOverlay()
    pipeline.overlay = overlay
    pipeline.async_translate = True
    try:
        for index in (1, 2, 3):
            token = pipeline.view.advance()
            pipeline.discard_obsolete()
            region = TextRegion(Rect(20, 30, 100, 30), f'Page {index}', .99,
                                'en', f'unique-{index}')
            pipeline._translate_regions([region], token)
            if index == 1:
                assert translator.started.wait(5)
        translator.release.set()
        assert overlay.translated.wait(5)
        assert translator.calls == [['Page 1'], ['Page 3']]
        assert all(source == 'Page 3' for source, _ in overlay.history)
        assert memory.lookup('Page 1', 'en->pt-BR') is None
    finally:
        translator.release.set()
        pipeline.close(wait=True)
        memory.close()


def test_close_invalidates_inference_and_does_not_publish_late_translation(tmp_path):
    translator = RecordingTranslator()
    translator.block = True
    _, pipeline, overlay, memory = _build_pipeline(tmp_path, translator=translator)
    pipeline.async_translate = True
    region = TextRegion(Rect(20, 30, 100, 30), 'Old view', .99, 'en', 'old')
    token = pipeline.view.token()
    try:
        pipeline._translate_regions([region], token)
        assert translator.started.wait(5)
        # Keep a reference to await worker cleanup after nonblocking close.
        executor = pipeline._translate_executor
        pipeline.close()
        translator.release.set()
        executor.shutdown(wait=True)
        assert not token.current()
        assert overlay.items == []
        assert memory.lookup('Old view', 'en->pt-BR') is None
        with pytest.raises(WorkCancelled):
            token.check()
    finally:
        translator.release.set()
        memory.close()


def test_exact_ocr_cache_relocates_bboxes_and_distinguishes_sizes(tmp_path):
    class PositionedOcr:
        calls = 0

        def recognize(self, image, regions):
            self.calls += 1
            return [TextRegion(Rect(r.x+2, r.y+3, 20, 10), 'Hello', .9, 'en', 'visual')
                    for r in regions]

    ocr = PositionedOcr()
    _, pipeline, _, memory = _build_pipeline(tmp_path, ocr=ocr)
    image = np.full((300, 400, 3), 255, np.uint8)
    try:
        first = pipeline._ocr_cached(image, [Rect(10, 20, 60, 40)])
        moved = pipeline._ocr_cached(image, [Rect(200, 220, 60, 40)])
        assert ocr.calls == 1
        assert moved[0].bbox == Rect(202, 223, 20, 10)
        assert first[0].bbox == Rect(12, 23, 20, 10)
        pipeline._ocr_cached(image, [Rect(10, 20, 61, 40)])
        assert ocr.calls == 2
        image[25, 15] = 0
        pipeline._ocr_cached(image, [Rect(10, 20, 60, 40)])
        assert ocr.calls == 3
    finally:
        memory.close()


def test_cancelled_single_request_batch_does_not_send_remaining_texts():
    view = ViewState()
    token = view.token()
    calls = []
    class SlowSingle(Translator):
        def translate(self, text, *args):
            calls.append(text)
            view.advance()
            return TranslationResult('traduzido', 'test')
    with pytest.raises(WorkCancelled):
        SlowSingle().translate_batch_cancellable(
            check=token.check, texts=['first', 'second'], source_lang='en',
            target_lang='pt-BR', glossary=Glossary.default(),
        )
    assert calls == ['first']


def test_cancellation_does_not_trigger_fallback_or_cooldown():
    view = ViewState()
    token = view.token()
    class CancelledProvider(Translator):
        def translate_batch(self, *args):
            view.advance()
            token.check()
    chain = FallbackChain([CancelledProvider(), EchoTranslator()])
    with pytest.raises(WorkCancelled):
        chain.translate_batch_cancellable(
            check=token.check, texts=['first'], source_lang='en',
            target_lang='pt-BR', glossary=Glossary.default(),
        )
    assert chain._cooldown_until == [0, 0]


def test_cancelled_google_batch_does_not_retry_as_individual_requests(monkeypatch):
    view = ViewState()
    token = view.token()
    calls = []
    def request(*args):
        calls.append(args)
        view.advance()
        return 'merged lines'
    translator = GoogleWebTranslator()
    monkeypatch.setattr(translator, '_request', request)
    with pytest.raises(WorkCancelled):
        translator.translate_batch_cancellable(
            check=token.check, texts=['first', 'second'], source_lang='en',
            target_lang='pt-BR', glossary=Glossary.default(),
        )
    assert len(calls) == 1


def test_translation_mode_does_not_show_english_preview(tmp_path):
    translator = RecordingTranslator()
    translator.block = True
    _, pipeline, overlay, memory = _build_pipeline(tmp_path, translator=translator)
    pipeline.async_translate = True
    try:
        pipeline.process_frame(frame(1), force=True)
        assert translator.started.wait(5)
        assert overlay.items == []
    finally:
        translator.release.set()
        pipeline.close(wait=True)
        memory.close()
