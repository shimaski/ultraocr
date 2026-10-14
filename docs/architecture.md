# Architecture

This project treats the desktop as a video stream, not as repeated screenshots.

## Runtime Pipeline

```text
ScreenStream
  -> FrameDiffer
  -> TextRegionDetector
  -> OCR
  -> Exact ROI OCR Cache
  -> Visual Cache / Translation Memory
  -> Translation Provider Chain
  -> Translation History
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

`TextRegionDetector` is a cheap prefilter for OCR engines that do not provide
their own detector. The default PaddleOCR path detects text directly in each
changed ROI; the optional `comic` mode first runs an ONNX manga-text detector,
then sends its line boxes to PaddleOCR for recognition.

The standard mobile mode keeps detection inside PaddleOCR for the simpler,
lower-setup path.

The Reader defaults to automatic English/Japanese OCR. It tries English first
and invokes the Japanese recognizer when the English result is weak or contains
CJK script. Clear English results skip the second recognition pass. When both
models return kana, the Japanese-specific result wins over a raw confidence-only
comparison. The mobile Japanese path uses PaddleOCR's dedicated
`japan_PP-OCRv3_mobile_rec` model; it is warmed up in the background so model
initialization does not block the reader window. Selecting `en` or `japan`
explicitly limits OCR to that language.

## Cache Layers

There are two local cache layers:

- Visual cache: keyed by perceptual hash of a detected text region.
- Translation Memory: keyed by normalized text and language pair.

Lookup order:

1. Visual exact hit.
2. Text exact hit.
3. Fuzzy text hit.
4. Configured provider chain (Google, DeepL, Gemini, LibreTranslate, or
   optional Argos offline), using the first available successful provider.

Argos translation is local after model installation. It prefers a direct
language pair and can pivot Japanese through English (`ja -> en -> pb`) when
both models are installed; package licensing and downloads remain explicit
user choices.

Successful translations are saved to both reusable caches and appended to the
local history. History is searchable by original or translated text and records
the language pair and provider. Echo fallbacks are never persisted as a
translation.

The history is deliberately separate from Translation Memory: memory keeps one
current translation per normalized text/language pair, while history retains
each successful translation event in time order.

Text memory can be exported/imported as versioned JSON from the reader toolbar.
Exports preserve language pair, source/target language, context, frequency, and
timestamps; they deliberately omit visual-region hashes and translation history.
Imports validate the whole file first and apply it in one SQLite transaction.
Matching entries are replaced by default or can be kept; unmatched local entries
are never deleted. Updating the FTS index is part of that transaction.

An entry can be corrected from the reader history. The edit updates the selected
history row, matching text-memory entry, and matching visual-cache entries in one
SQLite transaction. The reader reanalyzes its selected region after saving, so
the corrected text can appear immediately. Other language pairs are untouched.

## Glossary

The reader and control panel edit protected names, preferred terms, and general
translation instructions in the local TOML glossary. A changed source term
invalidates only matching reusable cache entries; changing general instructions
requires confirmation before clearing reusable translation caches. History is
retained. Reader edits update the active glossary and reanalyze the selected
region; panel edits are available while its capture session is stopped.

The reader also supports a work glossary. It infers a stable series identity
from the host and the path segment before a chapter/episode route (or common
series query parameters), stores the TOML under the user's local application
data directory, and layers its entries over the global glossary. Series-scoped
text and visual cache keys prevent one work's terminology from leaking into
another. If a site uses an unusual URL shape, the global glossary remains
available as the reliable fallback.

## Windows Hotkeys

The reader registers `Ctrl+Shift+T` to translate the selected region and
`Ctrl+Shift+R` to clear it. Thread-wide `WM_HOTKEY` messages are consumed by a
Qt native event filter on the GUI thread. If another program owns a combination,
that one remains available as a local shortcut while the reader is focused.
Registered IDs are unregistered when the reader closes.

## Overlay

The overlay is independent from the browser. It receives absolute screen-space
rectangles and draws translated text in a transparent always-on-top Qt window.

When the OCR block sits inside a closed, uniform comic balloon, a bounded
connected-color segmentation estimates the balloon rectangle and uses it as the
caption anchor. The original OCR rectangle remains the identity/cache geometry.
If the candidate region reaches the local crop edge, is too large, or fails the
containment checks, the overlay falls back to the OCR rectangle; open or textured
art therefore does not get aggressively expanded.

The active overlay set is updated incrementally. When a changed ROI appears, any
old overlay item intersecting that ROI is removed before new OCR results are
inserted.
