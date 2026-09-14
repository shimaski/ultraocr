# MVP Plan

## Version 0.1

Goal: translate currently visible manga text on Windows with a local model.

Scope:

- Windows capture through DXGI/Desktop Duplication via `dxcam`.
- Full-screen or region capture.
- Tile diff with stable-region debounce.
- PaddleOCR on changed text candidates.
- SQLite Translation Memory with fuzzy matching.
- Ollama local translation.
- Qt transparent overlay.

Run target:

```powershell
manga-rt-translator run --config config.toml --glossary glossary.toml
```

## Tuning Checklist

Start with:

- `fps = 30`
- `tile_size = 96`
- `stable_ms = 220`
- `min_changed_ratio = 0.015`

If OCR triggers too often:

- Increase `stable_ms`.
- Increase `min_changed_ratio`.
- Increase `tile_size`.

If OCR misses small balloons:

- Lower `tile_size`.
- Lower `min_changed_ratio`.
- Increase `roi_padding`.

## Version 0.2

- Add window picker.
- Add browser chrome exclusion area.
- Add cache editor.
- Add per-title glossary files.
- Batch adjacent OCR lines before translation.

## Version 0.3

- Add YOLO/ONNX bubble detector.
- Add better Japanese vertical text handling.
- Add region tracking during scroll.
- Add DirectML/ONNX acceleration where useful.

## Version 1.0

- Native capture backends for Windows/macOS/Linux.
- Native overlay where Qt is too limited.
- Project UI for profiles, glossaries, cache, and model settings.
