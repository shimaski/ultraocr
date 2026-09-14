# Architecture

This project treats the desktop as a video stream, not as repeated screenshots.

## Runtime Pipeline

```text
ScreenStream
  -> FrameDiffer
  -> TextRegionDetector
  -> OCR
  -> Visual Region Cache
  -> Translation Memory
  -> Ollama Translator
  -> Overlay
```

## Stream Capture

The capture module exposes a long-lived `ScreenStream` protocol:

```python
stream.start()
for frame in stream.frames():
    ...
```

On Windows, `WindowsDxgiStream` uses `dxcam` with `video_mode=True`, which keeps a
continuous capture loop alive and lets the pipeline read the latest frame.

Future backends should follow the same contract:

- macOS: ScreenCaptureKit.
- Linux: PipeWire through xdg-desktop-portal ScreenCast.

## Frame Differencing

`FrameDiffer` avoids OCR on unchanged frames:

1. First frame creates an initial ROI.
2. Native dirty rects are used when available.
3. Otherwise the frame is divided into tiles.
4. Tiles are marked changed only when the pixel-change ratio crosses a threshold.
5. Nearby tiles are merged.
6. Regions must remain visually stable for `stable_ms` before OCR.

This keeps scrolls, loading animations, and duplicate frames from hammering OCR
or the LLM.

## OCR Targeting

`TextRegionDetector` is a cheap prefilter. It rejects changed regions with low
contrast or low edge density. PaddleOCR then performs real text detection only
inside those candidates.

The first production upgrade should be a tiny ONNX bubble/text detector. The
pipeline already has a separate detector layer for that.

## Cache Layers

There are two local cache layers:

- Visual cache: keyed by perceptual hash of a detected text region.
- Translation Memory: keyed by normalized text and language pair.

Lookup order:

1. Visual exact hit.
2. Text exact hit.
3. Fuzzy text hit.
4. Local LLM.

Every LLM result is saved back into both caches.

## Overlay

The overlay is independent from the browser. It receives absolute screen-space
rectangles and draws translated text in a transparent always-on-top Qt window.

The active overlay set is updated incrementally. When a changed ROI appears, any
old overlay item intersecting that ROI is removed before new OCR results are
inserted.
