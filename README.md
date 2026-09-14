# UltraOCR — Manga Real-Time Translator

Tradutor em tempo real de mangás/webtoons para desktop Windows. Captura a tela, faz OCR local, traduz pela internet e sobrepõe a tradução em português (pt-BR) sobre o texto original.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Capturas de Tela](#capturas-de-tela)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Provedores de Tradução](#provedores-de-tradução)
- [Rate Limiter Adaptativo](#rate-limiter-adaptativo)
- [OCR](#ocr)
- [Detector de Texto (Comic)](#detector-de-texto-comic)
- [Overlay](#overlay)
- [Arquitetura](#arquitetura)
- [Comandos CLI](#comandos-cli)
- [Scripts de Validação](#scripts-de-validação)
- [Testes](#testes)
- [Problemas Conhecidos](#problemas-conhecidos)
- [Roadmap](#roadmap)
- [Licença](#licença)

---

## Visão Geral

O UltraOCR é um leitor de mangá/desktop que:

1. **Captura** a região de leitura da tela (via DXGI/dxcam ou viewport Qt)
2. **Detecta** balões de texto com detecção ONNX (GPU) ou PaddleOCR
3. **Reconhece** o texto via PaddleOCR (local, sem internet)
4. **Traduz** o texto via Google Translate, Gemini, DeepL ou LibreTranslate
5. **Renderiza** a tradução em português sobre o texto original em um overlay transparente

O app não modifica o site original — tudo acontece em uma janela overlay por cima do navegador.

---

## Funcionalidades

| Recurso | Status |
|---------|--------|
| Captura de tela via DXGI (GPU) | ✅ |
| Captura viewport via Qt | ✅ |
| OCR local (PaddleOCR) | ✅ |
| Detecção de texto ONNX (GPU, DirectML) | ✅ |
| Google Translate (grátis, sem chave) | ✅ |
| Gemini LLM (grátis, com chave) | ✅ |
| DeepL (grátis, com chave) | ✅ |
| LibreTranslate (Docker local ilimitado) | ✅ |
| Cache de traduções (SQLite) | ✅ |
| Cache de OCR (por hash de pixels) | ✅ |
| Tradução em lote (1 request por frame) | ✅ |
| Tradução assíncrona (não bloqueia UI) | ✅ |
| Rate limiter adaptativo (backoff + jitter) | ✅ |
| Fallback automático entre provedores | ✅ |
| Glossário de nomes próprios | ✅ |
| Overlay transparente (janela top-level) | ✅ |
| Modo contínuo (auto-traduz ao rolar) | ✅ |
| Modo manual (Ctrl+Shift+T) | ✅ |
| Interface 100% em português | ✅ |
| 67 testes automatizados | ✅ |

---

## Capturas de Tela

<!-- Adicionar screenshots aqui -->

```
Leitor com overlay traduzindo mangá em tempo real:
┌─────────────────────────────────┐
│  [Navegador com mangá]          │
│  ┌───────────────────┐          │
│  │  "Olá, mundo!"    │ ← overlay│
│  └───────────────────┘          │
│  ┌───────────────────┐          │
│  │  "Como vai você?" │ ← overlay│
│  └───────────────────┘          │
└─────────────────────────────────┘
```

---

## Requisitos

### Software

- **Windows 10/11** (64-bit)
- **Python 3.11** (SEMPRE usar `.venv311` — o `.venv` padrão usa 3.14 que quebra PaddleOCR)
- **GPU NVIDIA/AMD/Intel** (recomendado para detecção ONNX via DirectML)
- **Git** (para clonar)

### Hardware Recomendado

- GPU com suporte DirectML (RTX 3070 ou superior recomendado)
- 8GB+ de RAM
- SSD para carregamento rápido de modelos

### Python Packages

Dependências principais (instaladas automaticamente):

- `paddlepaddle` 3.3.1 (backend OCR)
- `paddleocr` 3.3.1 (reconhecimento de texto)
- `paddlex` 3.x (pipeline de modelos)
- `onnxruntime-directml` 1.24.4 (inferência GPU)
- `pyside6` 6.11 (UI + navegador embutido)
- `opencv-python` (processamento de imagem)

---

## Instalação

### 1. Clonar o repositório

```powershell
git clone https://github.com/shimaski/ultraocr.git
cd ultraocr
```

### 2. Criar ambiente virtual (Python 3.11)

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

### 3. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 4. Copiar configurações de exemplo

```powershell
Copy-Item config.example.toml config.toml
Copy-Item glossary.example.toml glossary.toml
```

### 5. Configurar chaves de API (opcional)

```powershell
# Gemini (grátis em aistudio.google.com/apikey)
$env:MANGA_RT_GEMINI_API_KEY = "sua-chave"

# DeepL (grátis em deepl.com/pro-api)
$env:MANGA_RT_DEEPL_API_KEY = "sua-chave"

# LibreTranslate (opcional, para instância pública)
$env:MANGA_RT_LIBRE_API_KEY = "sua-chave"
```

### 6. Iniciar

```powershell
START_READER.bat
```

---

## Configuração

### config.toml

```toml
[ocr]
# Detector de texto: "paddle" (CPU) ou "comic" (GPU ONNX, mais rápido)
detector = "comic"

# Modelo de reconhecimento: "mobile" (rápido) ou "server" (mais preciso)
rec_model = "mobile"

# Modelo de detecção: "mobile" (rápido) ou "server" (mais preciso)
det_model = "mobile"

# Idiomas extras para OCR além do principal
extra_langs = ["japan", "korean"]

[translation]
# Provedor padrão: "google", "gemini", "deepl", "libre", "echo"
provider = "gemini"

# Idioma de destino
target_lang = "pt-BR"

# Timeout para requests de tradução (segundos)
timeout_seconds = 6

# Intervalo mínimo entre requests do Gemini (segundos)
gemini_min_interval = 1.0

# Chaves de API (configure via variáveis de ambiente, NÃO no TOML)
# gemini_api_key = ""
# deepl_api_key = ""
# libre_api_key = ""

# URL do LibreTranslate (para Docker local)
# libre_url = "http://localhost:5000"

[overlay]
# Opacidade do overlay (0.0 a 1.0)
opacity = 0.94

# Fonte do overlay
font_family = "Segoe UI"
font_size = 14
```

### glossary.toml

```toml
# Nomes próprios que devem ser preservados na tradução
[names]
"Son Goku" = "Son Goku"
"Naruto" = "Naruto"

# Termos técnicos que devem ser traduzidos de forma específica
[terms]
"Saiyajin" = "Saiyajin"
"Quincy" = "Quincy"
```

---

## Uso

### Modo Principal (Recomendado)

```powershell
START_READER.bat
```

1. Navegue para um capítulo de mangá no navegador embutido
2. Feche popups/anúncios do site
3. Clique em **Selecionar região** e arraste sobre a coluna de quadrinhos
4. O modo Contínuo ativa automaticamente
5. Role a página — as traduções aparecem em ~1-3 segundos

### Atalhos

| Tecla | Ação |
|-------|------|
| `Ctrl+Shift+T` | Traduzir região selecionada |
| `Ctrl+Shift+R` | Limpar região e parar tradução automática |

### Modos de Análise

- **Contínuo** (padrão): captura automaticamente após rolagem parar
- **Manual**: traduz apenas quando você clica em "Traduzir região"

### CLI Directo

```powershell
# Leitor com URL específica
.\.venv311\Scripts\python.exe -m manga_rt_translator reader --config config.toml --glossary glossary.toml --url "https://mangadex.org"

# Listar janelas visíveis
.\.venv311\Scripts\python.exe -m manga_rt_translator list-windows

# Modo overlay externo (qualquer janela)
.\.venv311\Scripts\python.exe -m manga_rt_translator run --config config.toml --glossary glossary.toml --window-title "Chrome"
```

---

## Provedores de Tradução

### Google Translate (Padrão)

- **Chave necessária:** Não
- **Qualidade:** Boa
- **Velocidade:** ~700ms por request
- **Limite:** Pode bloquear IP após muitos requests (rate limiter adaptativo)
- **Setup:** Nenhum — funciona imediatamente

### Gemini (LLM)

- **Chave necessária:** Sim (grátis em aistudio.google.com/apikey)
- **Qualidade:** Excelente (entende contexto de cena)
- **Velocidade:** ~1-4s por request
- **Limite:** Quota diária gratuita
- **Setup:** Configurar `MANGA_RT_GEMINI_API_KEY`

### DeepL

- **Chave necessária:** Sim (grátis em deepl.com/pro-api)
- **Qualidade:** Muito boa
- **Velocidade:** ~500ms por request
- **Limite:** 500.000 caracteres/mês (grátis)
- **Setup:** Configurar `MANGA_RT_DEEPL_API_KEY`

### LibreTranslate

- **Chave necessária:** Não (pública) ou Sim (Docker local)
- **Qualidade:** Boa
- **Velocidade:** ~1s por request
- **Limite:** Instância pública tem rate limit; Docker local é ilimitado
- **Setup Docker local:**
  ```powershell
  docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
  ```
  Then config `libre_url = "http://localhost:5000"`

### Echo (Teste)

- **Chave necessária:** Não
- **Qualidade:** N/A (retorna texto original)
- **Uso:** Testar OCR e overlay sem tradução real

### Fallback Chain

O sistema usa uma cadeia de fallback automática:

1. Tenta o provedor configurado
2. Se falhar (429, 503, timeout), cooldown adaptativo
3. Tenta o próximo provedor disponível
4. Se todos falharem, usa Echo (texto original)

---

## Rate Limiter Adaptativo

O UltraOCR implementa um rate limiter inteligente para evitar bloqueios:

### Características

- **Backoff exponencial com jitter:** cooldown cresce exponencialmente com aleatoriedade
- **Parse de Retry-After:** respeita headers `Retry-After` e `X-RateLimit-Reset` dos servidores
- **Tracking por provedor:** cada provedor tem seu cooldown independente
- **Reset em sucesso:** provedor disponível imediatamente após resposta válida
- **Estatísticas:** logs de failures, RPM e cooldown restante

### Comportamento

```
[rate limit hit] → cooldown 60s × jitter
[2º hit]         → cooldown 120s × jitter
[3º hit]         → cooldown 240s × jitter
[sucesso]        → cooldown resetado, provider disponível
```

### Logs

```
[ratelimit] google: 2 consecutive failures, cooldown 45.3s (base=30.0, jitter=1.51)
[translator] GoogleWebTranslator batch failed: Google Translate HTTP 429
[translator] GeminiTranslator succeeded
```

---

## OCR

### PaddleOCR (Padrão)

- **Velocidade:** ~59ms por caixa (com doc-preprocessor desligado)
- **Idiomas:** English, Japanese, Korean, Chinese, e mais
- **GPU:** CPU por padrão (DirectML para detecção ONNX)
- **Cache:** hashes de pixels evitam reprocessamento

### Configuração

```toml
[ocr]
detector = "paddle"      # ou "comic"
rec_model = "mobile"     # ou "server"
det_model = "mobile"     # ou "server"
extra_langs = ["japan", "korean"]
```

### Otimizações

- **Doc-preprocessor desligado:** 1643ms → 59ms por caixa (28x mais rápido)
- **Modelos mobile:** `PP-OCRv5_mobile_rec` para en/korean; `PP-OCRv5_mobile_rec` genérico para japan
- **Retry 2x:** quando confiança < 0.85, tenta com upscaling
- **Dedup por IoU:** 0.55 para remover detecções duplicadas

---

## Detector de Texto (Comic)

### ComicTextDetector (ONNX)

- **Velocidade:** ~80ms na RTX 3070 (GPU via DirectML)
- **Precisão:** Boa para fonte impressa; calibrada para manuscrita
- **Threshold:** `LINE_THRESHOLD = 0.25` (ajustado de 0.35)
- **Merge:** `_merge_close_boxes` junta fragmentos com gap ≤14px

### Ativar

```toml
[ocr]
detector = "comic"
```

### Comparação

| Detector | Velocidade | Precisão | GPU |
|----------|-----------|----------|-----|
| Paddle | ~1643ms/crop | Boa | CPU |
| Comic (ONNX) | ~80ms/crop | Boa | DirectML |

---

## Overlay

### Características

- **Janela top-level transparente** (não filho do navegador)
- **Opacidade configurável** (padrão 0.94)
- **Caixas abraçam o bloco de texto** (sem cap de 420px)
- **Estilo provisório:** texto original em cinza itálico até a tradução chegar
- **Repaint inteligente:** só quando a assinatura dos itens muda
- **Imune a bugs do PySide6 6.11** (widgets filhos do QWebEngineView morrem)

### Renderização

```
┌─────────────────────────────────┐
│  Texto original (invisível)     │
│  ┌───────────────────┐          │
│  │  "Olá, mundo!"    │ ← overlay│
│  └───────────────────┘          │
└─────────────────────────────────┘
```

---

## Arquitetura

```
reader.py (app principal, PySide6)
  ├─ QWebEngineView (navegador embutido, ANGLE/GPU)
  ├─ Escuta de scroll: poll JS assíncrono (8/s)
  │    ├─ movimento → fade imediato das caixas velhas (0.4s)
  │    └─ parada (2 sondas ~250ms) → Traduzir região direto
  ├─ sweep_timer 1.0s → análise de frame fresco
  ├─ capture_timer 1.0s → heartbeat (suprimido durante scroll)
  ├─ Overlay: JANELA TOP-LEVEL transparente
  └─ ReaderRuntime (thread worker, prioridade abaixo do normal)
       └─ MangaTranslationPipeline
            ├─ FrameDiffer (diff por tiles + estabilidade 150ms)
            ├─ TextRegionDetector (Paddle) / detect_lines (comic ONNX)
            ├─ _ocr_cached: cache de OCR por hash de conteúdo
            ├─ PaddleOcrEngine (det+rec por recorte; retry 2x se conf<0.85)
            │    └─ OU ComicTextOcr (detector ONNX via DirectML)
            ├─ _translate_regions: cache → lote por idioma → 1 request
            └─ Overlay via bridge (sinal entre threads)

translator.py — cadeia de provedores (FallbackChain com cooldown)
  ├─ google  : endpoint gratuito gtx (conexão HTTPS persistente)
  ├─ gemini  : LLM grátis (contexto de cena, corrige erros de OCR)
  ├─ deepl   : REST oficial (chave grátis)
  ├─ libre   : Docker local ilimitado
  └─ echo    : último recurso (nunca trava)

RateLimitTracker (global)
  ├─ Backoff exponencial com jitter
  ├─ Parse Retry-After / X-RateLimit-Reset
  ├─ Tracking por provedor (failures, RPM, cooldown)
  └─ Reset em sucesso
```

### Estrutura de Arquivos

```
manga_rt_translator/
  __init__.py           pacote
  __main__.py           entry point
  app_runtime.py        runtime da aplicação
  cache.py              SQLite Translation Memory
  capture/              backends de captura
    base.py
    factory.py
    windows_dxgi.py     dxcam (Desktop Duplication)
    windows_windows.py  PrintWindow/SendKeys
  cli.py                command line interface
  config.py             TOML config loader
  diff.py               tile diff e stable ROI debounce
  glossary.py           glossário e nomes protegidos
  hybrid_ocr.py         OCR híbrido (ONNX + Paddle)
  image_hash.py         hashing de imagens
  models.py             data models compartilhados
  ocr.py                PaddleOCR adapter + ComicTextOcr
  overlay.py            Qt transparent overlay
  pipeline.py           orquestração end-to-end
  reader.py             app principal (PySide6)
  reader_runtime.py     worker cancelável
  text_regions.py       filtragem de ROI de texto
  translator.py         provedores de tradução + RateLimitTracker
  ui.py                 painel de controle
  work_queue.py         mailbox de último frame

tests/                  67 testes automatizados
scripts/                scripts de validação e benchmark
docs/                   documentação do projeto
```

---

## Comandos CLI

```powershell
# Iniciar leitor
manga-rt-translator reader --config config.toml --glossary glossary.toml --url "https://mangadex.org"

# Listar janelas visíveis
manga-rt-translator list-windows

# Modo overlay externo
manga-rt-translator run --config config.toml --glossary glossary.toml --window-title "Chrome"

# Painel de controle
manga-rt-translator ui --config config.toml --glossary glossary.toml
```

---

## Scripts de Validação

```powershell
# Testes automatizados
.\.venv311\Scripts\python.exe -m pytest tests -q

# Validação de scroll
.\.venv311\Scripts\python.exe scripts\validate_reader_scroll.py
.\.venv311\Scripts\python.exe scripts\validate_reader_scroll.py --scale-factor 1.25

# Benchmark de OCR
.\.venv311\Scripts\python.exe scripts\benchmark_ocr.py reader_debug_output\last_capture.png

# Validação de overlay
.\.venv311\Scripts\python.exe scripts\validate_overlay_layout.py caminho-da-imagem.png

# Reparo de cache
.\.venv311\Scripts\python.exe scripts\repair_translation_cache.py --apply
```

---

## Testes

```powershell
# Rodar todos os testes
.\.venv311\Scripts\python.exe -m pytest tests -q

# Resultado esperado: 67 passed
```

### Cobertura

- Config loader
- Cache de tradução (thread-safe)
- OCR grouping e qualidade
- Overlay layout
- Pipeline de integração
- Tradutores (Google, Gemini, DeepL, Libre)
- Rate limiter

---

## Problemas Conhecidos

1. **PySide6 6.11.1 — "Slot not found" intermitente:** widgets filhos do QWebEngineView perdem slots. Mitigado com overlay top-level.

2. **Lazy-load do site desloca o layout:** caixas ficam ~1 ciclo dessincronizadas. Corrigido por invalidação por content-hash.

3. **Comic ONNX fragmenta manuscrita:** calibrado em 06/09 (LINE_THRESHOLD 0.25, merge gap 14px). Validar em capturas reais.

4. **MyMemory responde pt-PT** ("Estás a pensar") — qualidade inferior; só entra quando Google falha.

5. **Gemini 503 em pico:** chain cai para Google MT automaticamente.

6. **Anúncios do site:** popups/banners são traduzidos (rotacionam → caixa destoa). Atenuar com região bem ajustada.

---

## Roadmap

### Curto Prazo

- [ ] Empacotamento: instalador Inno Setup + ícone + atalho
- [ ] Perfis por site/obra (região + provedor + FPS + idioma)
- [ ] Calibração do detector comic para fonte manuscrita

### Médio Prazo

- [ ] Legendas ao vivo: áudio → Whisper → tradução
- [ ] manga-ocr para japonês vertical
- [ ] Leitura em ONNX (rec ~5-10x mais rápido)

### Longo Prazo

- [ ] Shell nativo WPF (WebView2) — UltraMangaReader.Wpf
- [ ] Suporte macOS (ScreenCaptureKit) e Linux (PipeWire)
- [ ] Detector de balões dedicado

---

## Licença

Projeto privado. Todos os direitos reservados.

---

## Contato

- **GitHub:** [shimaski/ultraocr](https://github.com/shimaski/ultraocr)
- **Issues:** [GitHub Issues](https://github.com/shimaski/ultraocr/issues)
