# Manga Real-Time Translator

Local-first manga and webtoon translator for browser-style reading.

## Atualizacao: Regiao Fixa e Rolagem

Abra `START_READER.bat`, mantenha **Traduzir PT-BR**, escolha a origem `en`
para ingles e selecione a area de leitura uma vez. A selecao ativa o modo
continuo. Ao rolar, as caixas antigas somem; cerca de 1,1 segundo depois de
parar, o leitor captura a area novamente e processa seu novo conteudo.
Limpar a regiao desativa a analise automatica. `Ctrl+Shift+T` refaz a leitura.

- Fila de um frame: uma nova vista substitui o trabalho pendente. OCR e
  respostas de traducao antigos nao podem redesenhar caixas apos a rolagem.
- Cancelamento cooperativo entre etapas, lotes e provedores. Uma inferencia
  nativa ou requisicao HTTP ja iniciada precisa retornar antes de liberar o
  worker; nao ha encerramento forcado de threads.
- Cache OCR exato por pixels e dimensoes, com coordenadas relativas. Voltar
  ao mesmo conteudo nao exige novo OCR; a memoria SQLite evita retraducao.
- Linhas agrupadas antes da traducao, com ordem de leitura e verificacao de
  bordas entre baloes. Ainda e uma heuristica, nao um segmentador de baloes.
- O overlay calcula todas as caixas em conjunto: expande frases longas na
  horizontal, limita colunas muito altas, reduz a fonte em paginas densas e
  reposiciona pelas bordas livres. Caixas impossiveis de acomodar sao omitidas
  em vez de serem desenhadas sobre outra traducao.
- No modo de traducao, o texto original nao aparece como falsa traducao.
  A opcao **Somente OCR** continua disponivel para diagnostico.
- Paddle mobile continua como padrao. `ocr.detector = "hybrid"` e opcional:
  ONNX detecta linhas, Paddle reconhece lotes de ate 8 e rele blocos incertos.
  Exige o ONNX ja presente em `.models`; sem ele, usa Paddle, sem download
  automatico. Ingles somente no caminho rapido; outros idiomas usam Paddle.
- A captura de uma nova posição começa 550 ms após a rolagem parar. O retry
  OCR em 2x fica restrito a recortes abaixo de 250 mil pixels, evitando repetir
  uma região grande inteira quando apenas uma linha teve confiança baixa.

**Limite importante:** OCR e cache sao locais, mas a configuracao atual
traduz pela internet. No teste real desta revisao, Google respondeu HTTP 429
e MyMemory HTTP 504. O programa informa falha do tradutor; nao ha garantia
de disponibilidade de endpoints gratuitos. Ollama nao e iniciado pelo reader.

### Validacao Reproduzivel

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
.\.venv311\Scripts\python.exe scripts\validate_reader_scroll.py
.\.venv311\Scripts\python.exe scripts\validate_reader_scroll.py --scale-factor 1.25
.\.venv311\Scripts\python.exe scripts\benchmark_ocr.py reader_debug_output\last_capture.png
.\.venv311\Scripts\python.exe scripts\validate_overlay_layout.py caminho-da-imagem.png
```

O teste de janela usa Qt, captura e OCR reais, uma pagina local com a imagem
salva no projeto e **traducoes simuladas** para nao depender da rede. Ele
seleciona uma vez, rola, retorna e faz uma ida/volta antes do debounce.
Esta revisao passou nos 67 testes automatizados e nesse roteiro de janela,
inclusive com escala 125%.
`--live-translation` usa os provedores reais. Resultados, capturas e um banco
temporario separado ficam em `reader_debug_output/qa-*/`.

Na imagem local de 977x861, apos aquecimento, uma medicao foi 3317 ms com
Paddle e 2555 ms com hybrid. O hybrid ainda trocou uma palavra, portanto nao
foi promovido a padrao. No teste da regiao de 740x430, voltar ao mesmo quadro
levou 3-4 ms de processamento pelo cache, sem contar debounce e pintura.
Essas medicoes nao representam latencia de traducao online nem garantias.

O modo **reader com regiao selecionada e orientado a eventos**, nao um video
de 30-60 FPS: captura na selecao e depois da rolagem estabilizar. O stream
DXGI continua sendo o caminho separado de captura externa descrito abaixo.

The goal is to translate visible manga dialogue without modifying the original
site:

1. Capture the screen as a continuous frame stream.
2. Drop identical frames.
3. Build changed regions only.
4. Run OCR only on stable text candidates.
5. Check local Translation Memory before any translation call.
6. Translate misses with an online provider: Google Translate by default
   (free, no key), with optional DeepL support.
7. Draw translations in a transparent overlay.

This repository started as a Windows-first screen-stream MVP and is now
shifting toward a desktop reader with an embedded browser and in-app overlay.
The code is still organized so macOS and Linux backends can be added later
without rewriting the OCR and translation pipeline.

## MVP Status

Implemented:

- DXGI/Desktop Duplication style stream adapter through `dxcam`.
- Tile-based frame differencing with stable-region debounce.
- Speech-bubble text grouping with manga reading order: OCR lines are merged
  into blocks, CJK pages read right-to-left, other text top-to-bottom.
- Dedicated Korean (manhwa) OCR through PaddleOCR's `korean` model, selectable
  in the reader and the control panel; `Auto` mode falls back between English,
  Japanese, and Korean engines per region.
- Lightweight changed-region text candidate detection.
- PaddleOCR adapter with optional dependency loading.
- SQLite Translation Memory with exact and fuzzy matching.
- Glossary loader.
- Translation providers: Google Translate (default), DeepL, and Echo for
  testing, with automatic echo fallback when the provider fails.
- Transparent Qt overlay adapter.
- CLI entrypoint.
- Embedded desktop reader based on `PySide6` WebEngine.
- Reader-side diagnostics for captured frames, OCR events, and overlay state.

Still in progress:

- Native macOS ScreenCaptureKit backend.
- Native Linux PipeWire backend.
- Dedicated speech bubble detector.
- Vertical Japanese tuning beyond PaddleOCR defaults.
- Stable production-grade in-reader translation flow for every manga site.
- A native Windows stack comparable to mature commercial apps.

## Product Direction

The long-term product direction is now:

1. Open the manga site inside our own desktop reader.
2. Capture the selected reader region after selection and settled scrolling.
3. Run OCR and cache lookup locally; translate through the configured provider.
4. Render translated text inside the reader itself.

This is a better fit than a desktop-wide overlay because it keeps viewport
coordinates, scroll state, and overlay rendering inside one controlled app.

## Reference Findings

We inspected a commercial reference app to understand why it feels more stable.
The app was not copied; it was used as a product benchmark.

Key findings from the installed files:

- packaged with `Inno Setup`
- built as a Windows desktop `.NET` app
- UI stack appears to be `WPF`
- embedded browser stack appears to be `WebView2`
- local inference components include `ONNX Runtime`
- image and video processing include `OpenCvSharp` and `FFmpeg`
- local language tooling includes `CLD3`, `FastText`, and `OpenCC`
- local persistence includes `SQLite` and `SQLCipher`
- optional audio and speech tooling includes `Whisper`

That reference explains why a commercial app can feel sturdier than a Python
MVP: it uses a more native Windows-oriented stack, heavier integration, and
more mature runtime packaging.

Our project is intentionally moving in the same product direction while keeping
the implementation original and local-first.

## Install

Create a virtual environment, then install the package with the Windows MVP
extras:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[windows,ocr,overlay]"
```

Important: use Python 3.11 for the OCR stack. `paddleocr` currently installs on
newer Python versions more easily than `paddlepaddle`, but the runtime backend
needed for actual OCR does not reliably ship wheels for Python 3.14 yet.

Translation uses online providers. Google Translate needs no key or extra
install. DeepL needs a free API key from https://www.deepl.com/pro-api.

API keys should not be stored in `config.toml`. Set the matching environment
variable before starting the reader:

```powershell
$env:MANGA_RT_GEMINI_API_KEY = "your-key"
$env:MANGA_RT_DEEPL_API_KEY = "your-key"
$env:MANGA_RT_LIBRE_API_KEY = "your-key"
```

## Translation Providers

Choose the provider in `config.toml`:

```toml
[translation]
target_lang = "pt-BR"
provider = "google"   # google | deepl | echo
# deepl_api_key = ""  # required only for DeepL
```

- `google`: keyless chain — Google's public `gtx` endpoint first, MyMemory as
  fallback, echo as the last resort. Needs internet, no API key.
- `deepl`: DeepL REST API. Higher quality, requires `deepl_api_key`.
- `echo`: returns the OCR text unchanged. Useful to test OCR and overlay only.

If every provider fails (for example, no internet), the app echoes that string
instead of stopping the pipeline, and the original text stays visible in the
overlay. MyMemory has a free daily character quota; the translation memory
cache keeps repeated text from consuming it.

The pipeline is also quota-friendly by design: all uncached bubbles of a frame
are grouped by language and sent in one batched request, cache hits never
reach the network, and a provider that answers with a rate limit or an empty
quota is benched for a cooldown window while the others take over.

## Run

Copy the examples if you want to customize settings:

```powershell
Copy-Item config.example.toml config.toml
Copy-Item glossary.example.toml glossary.toml
```

Start the translator:

```powershell
manga-rt-translator run --config config.toml --glossary glossary.toml
```

List visible windows on Windows and choose the browser window by title:

```powershell
manga-rt-translator list-windows
manga-rt-translator run --config config.toml --glossary glossary.toml --window-title "Chrome"
```

Or open the lightweight control panel:

```powershell
manga-rt-translator ui --config config.toml --glossary glossary.toml
```

Or open the new desktop reader shell, which is the product direction closest to
apps such as Gaminik or FrankYomik:

```powershell
manga-rt-translator reader --config config.toml --glossary glossary.toml --url "https://mangadex.org"
```

The reader opens the manga site inside the program itself. This is the most
stable direction for a desktop product because the translation overlay can live
inside the app instead of floating over an arbitrary external browser window.

## Native Windows Reader

There is now a separate native Windows reader shell under:

```text
UltraMangaReader.Wpf/UltraMangaReader
```

This shell is the new Windows-first direction for a more stable desktop app.
It is built with:

- `.NET 8`
- `WPF`
- embedded browser shell

Local SDK bootstrap used in this repo:

```powershell
.\.dotnet\dotnet.exe build .\UltraMangaReader.Wpf\UltraMangaReader\UltraMangaReader.csproj
```

Run it with:

```powershell
.\UltraMangaReader.Wpf\UltraMangaReader\bin\Debug\net8.0-windows\UltraMangaReader.exe
```

Current native shell status:

- opens and stays running
- includes URL bar, navigation buttons, and reader controls
- is the new base for wiring OCR, region selection, and overlay in a more native stack

The control panel lets you:

- choose a visible window
- choose the translation provider: `Google`, `DeepL`, or `Echo`
- set a DeepL API key when using DeepL
- verify readiness before starting
- enable or disable overlay
- monitor basic runtime stats
- remember the last selected window and options

The reader currently gives you:

- a desktop app window with embedded browser
- back, forward, reload, and URL input
- source language, provider, FPS, and overlay controls in one place
- `Manual Analyze` and `Continuous` analysis modes
- selectable reading region inside the viewport
- stream capture from the embedded viewport
- basic in-app overlay rendering
- runtime diagnostics in the status bar
- a debug log in `.reader_debug.log`

Reader analysis modes:

- `Manual Analyze`: no continuous capture; translate the visible region only
  when you click `Translate Region` or press `Ctrl+Shift+T`
- `Continuous`: capture the viewport at the chosen FPS and let the diff
  pipeline decide when a frame deserves OCR

Selecting a reading region with `Select Region` turns on automatic
translation: the runtime starts (if needed), the mode switches to
`Continuous`, and everything that scrolls into the region is OCR'd and
translated without any further clicks. Clearing the region stops it.

Useful tuning knobs in `config.toml`:

- `ocr.ignore_top_pixels`: ignore the top browser UI strip
- `diff.large_change_tile_ratio`: suppress OCR during large scroll-like changes
- `diff.scroll_cooldown_ms`: wait briefly after a large visual jump

For a first smoke test without the overlay:

```powershell
manga-rt-translator run --no-overlay
```

## External Screen Stream

The capture layer owns a long-lived stream object. On Windows the MVP uses
`dxcam.start(..., video_mode=True)` and reads the latest frame from that running
stream. This describes the external capture backend, not the selected-region
reader workflow, which uses event-driven Qt viewport capture.

The differencer then decides whether a frame deserves work. If there is no
visual change and no pending stable region, the frame is dropped immediately.

## Project Layout

```text
manga_rt_translator/
  capture/            stream backends
  cache.py            SQLite Translation Memory
  cli.py              command line interface
  config.py           TOML config loader
  diff.py             tile diff and stable ROI debounce
  glossary.py         glossary and protected names
  models.py           shared data models
  ocr.py              PaddleOCR adapter
  hybrid_ocr.py       optional batched English recognition + Paddle repair
  overlay.py          Qt transparent overlay
  pipeline.py         end-to-end orchestration
  reader_runtime.py   cancellable reader worker and diagnostics
  work_queue.py       latest-frame mailbox and viewport generation tokens
  text_regions.py     lightweight text ROI filtering
  translator.py       translation providers (Google, DeepL, Echo)
```

## Current Reader Workflow

For the most reliable current test path:

```powershell
.\START_READER.bat
```

Then:

1. Open a page with visible text.
2. Leave `Mode = Traduzir PT-BR` and `Provider = Google`.
3. Use `Origem = en` for English pages. Use `Auto` only when the page mixes
   English, Japanese, and Korean because it loads additional OCR engines.
4. Use `Selecionar região` and drag over the reading area once.
5. Scroll normally. Selection starts the runtime and continuous mode.
6. For an explicit retry, use `Traduzir região` or `Ctrl+Shift+T`.

If something looks off, inspect `.reader_debug.log` in the project root.

If an older run showed untranslated English, repair only invalid Translation
Memory entries (a timestamped database backup is created automatically):

```powershell
.\.venv311\Scripts\python.exe scripts\repair_translation_cache.py --apply
```

## Next Build Steps

1. Improve OCR targeting for vertical Japanese beyond PaddleOCR defaults
   (a dedicated manga OCR model such as `manga-ocr` is a candidate).
2. Tune the overlay layout inside the reader viewport.
3. Consider moving performance-critical Windows pieces toward a more native stack.
