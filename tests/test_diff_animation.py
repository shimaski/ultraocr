from __future__ import annotations

import numpy as np

from manga_rt_translator.config import DiffConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.models import Frame


def frame(image, t):
    return Frame(image=image, timestamp=t, index=int(t * 100))


def _base():
    return np.full((300, 300, 3), 128, np.uint8)


def _run_animated(cfg, cycles, dt=1.0, region=(0, 0, 100, 100)):
    """Drive the differ with a spot that keeps changing in place (an ad), one
    change + one stable frame per cycle, and count how many OCR fires result."""
    differ = FrameDiffer(cfg)
    differ.prime(frame(_base(), 0.0))
    rx, ry, rw, rh = region
    fires = 0
    t = 0.0
    for k in range(cycles):
        pattern = _base()
        rng = np.random.default_rng(k + 1)
        pattern[ry : ry + rh, rx : rx + rw] = rng.integers(
            0, 255, (rh, rw, 3), dtype=np.uint8
        )
        t += dt
        differ.update(frame(pattern, t))
        fires += len(differ.update(frame(pattern.copy(), t + 0.3)))
    return fires


def test_animated_region_is_throttled():
    cfg = DiffConfig()  # suppression on, max_fires=3, window=9s
    # 10 changes in ~10s would be 10 OCR runs without suppression.
    assert _run_animated(cfg, cycles=10) <= cfg.animation_max_fires


def test_suppression_can_be_disabled():
    cfg = DiffConfig()
    cfg.animation_suppression = False
    assert _run_animated(cfg, cycles=10) == 10


def test_one_time_reading_content_still_fires():
    cfg = DiffConfig()
    differ = FrameDiffer(cfg)
    differ.prime(frame(_base(), 0.0))
    pattern = _base()
    pattern[50:120, 150:260] = 0  # a balloon appears once
    differ.update(frame(pattern, 1.0))
    out = differ.update(frame(pattern.copy(), 1.3))
    assert len(out) == 1
