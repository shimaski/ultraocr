"""Compare cold/warm OCR on the same local image, without translation/network."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.models import Rect
from manga_rt_translator.ocr import create_ocr_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reader_debug_output/ocr_benchmark.json'))
    parser.add_argument('--detector', action='append', choices=['paddle', 'hybrid'],
                        help='Detector to benchmark; may be repeated (default: both).')
    args = parser.parse_args()
    raw = cv2.imread(str(args.image))
    if raw is None:
        raise SystemExit(f'Cannot read {args.image}')
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    full = Rect(0, 0, image.shape[1], image.shape[0])
    report = {'image': str(args.image.resolve()), 'runs': []}
    for detector in (args.detector or ['paddle', 'hybrid']):
        started = time.perf_counter()
        engine = create_ocr_engine(OcrConfig(detector=detector))
        for stage in ('cold', 'warm'):
            if stage == 'warm':
                started = time.perf_counter()
            regions = engine.recognize(image, [full])
            row = dict(detector=detector, stage=stage,
                       elapsed_ms=round((time.perf_counter()-started)*1000),
                       fallback_reason=getattr(engine, 'last_fallback_reason', ''),
                       blocks=[dict(text=r.text, confidence=round(r.confidence, 3),
                                    bbox=r.bbox.as_tuple()) for r in regions])
            report['runs'].append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
