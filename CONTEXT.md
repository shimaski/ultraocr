# CONTEXT.md — UltraOCR (Manga Real-Time Translator)

> Documento de contexto acumulado das sessões de desenvolvimento.
> Última atualização: 2026-09-07. As notas abaixo são históricas;
> esta primeira seção descreve o comportamento mais recente.

## Atualização 07/09/2026: Fila, OCR e Validação

- Otimização de latência em 09/09: debounce após rolagem reduzido de 1100 para
  550 ms (`diff.scroll_settle_ms`). Retry Paddle em 2x agora só roda abaixo de
  250.000 pixels (`ocr.upscale_retry_max_pixels`); regiões grandes com uma
  linha fraca não repetem mais toda a inferência. O registro anterior mostrou
  7,4 s e 15,5 s em páginas densas quando esse retry global era acionado.
  Após a mudança, 72 testes passaram. Um benchmark sobre a captura enviada
  pelo usuário marcou 17,6 s aquecido, mas essa imagem já continha os overlays
  renderizados e não representa a captura limpa (o leitor oculta o overlay
  antes do OCR). Os registros limpos recentes ficaram em 2,2–5,4 s por área
  nova; retornos atendidos pelo cache permanecem em poucos milissegundos.

- Correção posterior do overlay: layout global anti-colisão, largura adaptada
  ao texto PT-BR, altura máxima inicial de 140 px e fonte de 10 pt quando há
  oito ou mais caixas (12 pt a partir de cinco). A imagem reportada pelo
  usuário foi renderizada com oito caixas e zero interseções em
  `reader_debug_output/overlay-layout-validation.png`. Validação total após a
  mudança: 71 testes passando.

- `ReaderRuntime` extraído para `reader_runtime.py`; `work_queue.py` vincula
  pixels, pedido forçado e geração da vista. Uma rolagem invalida o trabalho
  anterior, limpa as caixas e cancela traduções ainda na fila.
- Sinais Qt carregam o token da vista. A validade é conferida novamente
  ao receber e pintar, impedindo que uma resposta antiga reapareça.
- Retornar à posição anterior antes do debounce também recaptura. Limpar
  seleção interrompe o automático; navegação invalida resultados antigos.
- Seleção usa captura Qt orientada a eventos, após 1100 ms sem rolagem,
  sem varreduras periódicas da mesma região. Não chamar esse modo de vídeo.
- Buffer Qt respeita escala DPI; removidas esperas `sleep` do caminho de
  captura da GUI. Falhas de inicialização não são ocultadas por "parada".
- Cache OCR agora usa BLAKE2b dos pixels completos + dimensões/tipo, não
  amostragem 96x96. As caixas são armazenadas relativamente ao recorte.
- Agrupamento não altera linhas originais, corrige ordem com pequenas
  diferenças de Y e considera bordas entre balões. Arrays NumPy no parser
  Paddle não usam mais avaliação booleana ambígua.
- Novo OCR `hybrid`: ONNX + reconhecimento em lotes de 8 + reparo Paddle.
  Experimental, inglês; Paddle permanece padrão por melhor qualidade no
  teste real. Inferência Paddle compartilhada protegida por lock.
- Cancelamento é cooperativo; não interrompe uma chamada nativa/HTTP já em
  execução. Checagens entre lotes/provedores evitam continuar uma página velha.
- Google/MyMemory no modo padrão têm timeout de socket limitado a 6 s e
  não repetem a requisição por retry externo. Uma reconexão de transporte
  Google ainda pode ocorrer. Isso não é um prazo total da operação.
- Balões em inglês são mostrados apenas no modo explicitamente Somente OCR;
  falhas de provedores não são promovidas a traduções concluídas.

Validação: 67 testes automatizados e compilação Python passaram. A janela
real, captura Qt e OCR Paddle passaram no roteiro local de seleção única,
rolagem, retorno e ida/volta rápida. Nessa execução a tradução foi simulada.
Registro: `reader_debug_output/qa-1788825414/result.json`; capturas no mesmo
diretório. O mesmo roteiro passou em escala 125%, com coordenadas lógicas
740x430 preservadas: `qa-1788825669/result.json` e capturas inspecionadas.
O teste com tradução real falhou por Google HTTP 429 e MyMemory
HTTP 504, registrado em `qa-1788825241/result.json`. Não alegar tradução
online ponta a ponta validada nem teste no site ao vivo nesta sessão.

Benchmark local após aquecimento, 977x861: Paddle 3317 ms; híbrido 2555 ms,
mas o híbrido leu HUMAN como HLIMAN. Por isso não alteramos o detector padrão.
Retorno da região 740x430 pelo cache: 3-4 ms no pipeline, fora debounce/rede.
Scripts de reprodução: `scripts/benchmark_ocr.py` e
`scripts/validate_reader_scroll.py` (opcional `--live-translation`).

O Python 3.11 estava instalado: a falha "No Python at..." era de acesso
no ambiente restrito. Testes e QA executados com permissão ampliada; não foi
necessário reinstalar pacotes ou remover modelos/cache do usuário.

## Atualização 07/09/2026 — correção estrutural do fluxo

- Ambiente Python 3.11 restaurado; 48 testes passando.
- Corrigido o parser de lote do Gemini: numeração da resposta 1-based agora
  é mapeada para os índices originais corretamente.
- Resultados `echo` deixaram de ser persistidos como tradução PT-BR e não são
  mais promovidos a overlay final quando representam falha dos provedores.
- Preview de OCR em idioma original agora expira em até 12 segundos.
- Removidos sweeps de quadro completo duplicados no worker; a interface é o
  único coordenador da captura automática.
- Encerramento assíncrono protegido contra envio de tarefas após shutdown.
- Chaves de API removidas do TOML; usar `MANGA_RT_GEMINI_API_KEY`,
  `MANGA_RT_DEEPL_API_KEY` ou `MANGA_RT_LIBRE_API_KEY`.
- Cache reparado: 148 traduções e 196 regiões visuais inválidas removidas;
  backup criado antes da limpeza.
- Configuração prática padrão: Google, origem inglesa e sem OCRs extras.

## 1. O que é o projeto

App desktop Windows de **tradução em tempo real de mangás/webtoons**:
leitor com navegador embutido (PySide6/QtWebEngine) → captura da região de
leitura → OCR local → tradução via provedores web → overlay com a tradução
pintada em cima do texto original.

- **Raiz**: `C:\Users\adrie\ultraocr`
- **Python**: usar SEMPRE `.venv311` (Python 3.11.9). O `.venv` (3.14) não
  roda PaddleOCR.
- **Rodar**: `START_READER.bat` (usa `QT_OPENGL=angle` — GPU via Direct3D;
  NÃO voltar para `software`, que causa travas, nem `hardware`/OpenGL
  desktop, que dispara bug de slots do PySide6 6.11).
- **Testes**: `.venv311\Scripts\python.exe -m pytest tests -q` (45 testes).
- **Modo overlay externo** (qualquer janela): `manga-rt-translator run ...`
  com dxcam (Desktop Duplication).
- **Tradução assíncrona**: desde 2026-09-06 o reader roda com
  `pipeline.async_translate=True` — o OCR/captura nunca esperam a rede; a
  tradução chega via executor em thread separada e troca o texto provisório.

## 2. Arquitetura atual

```
reader.py (app principal, PySide6)
  ├─ QWebEngineView (navegador embutido, ANGLE/GPU)
  ├─ Escuta de scroll: poll JS assíncrono (FPS controla, padrão 8/s)
  │    ├─ movimento → fade imediato das caixas velhas (0,4s)
  │    └─ parada (2 sondas ~250ms) → Traduzir região direto
  ├─ sweep_timer 1,5s → análise de frame fresco (pega balões no scroll lento)
  ├─ capture_timer 2s  → heartbeat (suprimido durante scroll/minimizado)
  ├─ Overlay: JANELA TOP-LEVEL transparente (não é filho do webview!)
  │    └─ pintada sobre o navegador; some sem foco/minimizado; fecha com a janela
  └─ ReaderRuntime (thread worker, prioridade abaixo do normal)
       └─ MangaTranslationPipeline
            ├─ FrameDiffer (diff por tiles + estabilidade 150ms)
            ├─ TextRegionDetector (Paddle) / detect_lines (comic ONNX)
            ├─ _ocr_cached: cache de OCR por hash de conteúdo (96x96, LRU 64)
            ├─ PaddleOcrEngine (det+rec por recorte; retry 2x se conf<0.85)
            │    └─ OU ComicTextOcr (detector ONNX via DirectML, ~80ms na RTX 3070)
            ├─ _translate_regions: cache → lote por idioma → 1 request
            └─ Overlay via bridge (sinal entre threads)

translator.py — cadeia de provedores (FallbackChain com cooldown)
  ├─ google  : endpoint gratuito gtx (conexão HTTPS persistente)
  ├─ deepl   : REST oficial (chave grátis)
  ├─ gemini  : LLM grátis (aistudio.google.com/apikey) — CONTEXTO DE CENA,
  │            corrige erros de OCR, usa o glossário; modelo gemini-3.6-flash
  ├─ libre   : LibreTranslate (pública c/ chave OU Docker local ilimitado)
  └─ echo    : último recurso (nunca trava)
  Pacedência não-bloqueante: se o provedor pedir espera >1,5s, sai da frente
  e o próximo traduz (ex.: gemini → google mt → echo).
  RateLimitError(retry_after) alimenta o cooldown da cadeia (429/503/cota).

pipeline.py — _translate_regions: cache visual/texto PRIMEIRO, misses em
  lote por idioma (1 request), save no SQLite (TranslationMemory).
```

Config (config.toml): `[ocr] detector="paddle"|det_model|rec_model`,
`[translation] provider="gemini"|google|deepl|libre|echo`,
`gemini_api_key` (chave do usuário configurada), `[overlay] opacity=0.94`.

## 3. O que foi feito (todas as sessões até aqui)

### Tradução (sem Ollama, por decisão do usuário)
- Removido Ollama; provedores: Google gtx (implementação própria, sem
  deep-translator — o endpoint antigo do deep-translator estava sendo
  bloqueado), DeepL REST, Gemini (LLM grátis), LibreTranslate, Echo.
- FallbackChain: cooldown por provedor (429/503 → 12-60s; cota MyMemory → 1h),
  pacedência não-bloqueante do Gemini (>1,5s de espera = sai da frente).
- **Tradução em lote**: balões do frame agrupados por idioma → 1 request
  (join com \n, verificação de contagem, fallback para individual).
- Cache-first: visual (hash do balão) → texto → só então rede.
- Keep-alive HTTPS para o Google (sem handshake por lote).
- Chain do Gemini inclui Google MT como fallback instantâneo (nunca echo
  direto em pico de demanda).

### OCR
- **Agrupamento por balão**: cada candidato (balão) é reconhecido e agrupado
  independentemente — linhas de balões diferentes nunca se misturam.
- **Ordem de leitura de mangá**: CJK → direita para esquerda; ocidental →
  topo-baixo, esquerda-direita. Hangul NÃO é CJK-RTL (coreano é LTR).
- **Dedup por IoU** (0.55) de detecções duplicadas; **retry 2x** quando
  confiança média < 0.85 (fonte estilizada).
- **Koreano** dedicado (PaddleOCR `korean`) + Hangul na troca de engines.
- **Detector comic ONNX** (comic-text-detector via DirectML, ~80ms na RTX
  3070): implementado e funcional, mas REVERTIDO para paddle por fragmentar
  letra manuscrita (thresholds não calibrados). Ativar com
  `detector = "comic"`. Modelo: `.models/comictextdetector.pt.onnx`.
  Detalhe crítico: usar CANAL 0 do tensor `det` (canal 1 é o threshold map
  do DBNet — fica alto em 95% da imagem).
- OCR em prioridade de thread abaixo do normal (não compete com a UI).

### Leitor (UX/UX técnico)
- UI 100% pt-BR (botões, rótulos, mensagens).
- Overlay: **janela top-level** transparente (Gaminik-style) — imune aos bugs
  de ciclo de vida do PySide6 6.11 (child widget morria com Chromium);
  segue foco/visibilidade; fecha com o leitor.
- Caixa abraça o bloco de texto (sem cap de 420px); opacidade 0.94.
- Estilo provisório (original em cinza itálico) → troca pela tradução.
- Repaint só quando a assinatura dos itens muda.
- **Escuta de scroll** (JS assíncrono, 8/s): invalida caixas no 1º movimento
  (fade 0.4s); parada → tradução direta; velocidade lenta → varredura 1,5s;
  scroll rápido → suprimido (frame cortado = lixo).
- **dxcam (Desktop Duplication)**: captura GPU da região, só com leitor em
  primeiro plano/tela primária; "nochange" pula o ciclo; fallback grab.
- Modo Contínuo é o padrão; Manual continua via botão/atalho.
- Memória de chave API por provedor (não sobrescreve config — bug corrigido).

### Diagnóstico
- `.reader_debug.log`: eventos (runtime, sweeps, resultados, poll).
- `reader_stdout.log` / `reader_stderr.log`: prints do app + tracebacks.
- `Inspecionar OCR` → `reader_debug_output/last_capture.png` + `last_ocr.txt`.
- Instrumentação: scroll.poll (calls/results), continuous.sweep,
  "[ocr] comic det Xms".

## 4. Estado atual (verificado ao vivo no capítulo Killer Peter 147)

- Provedor: **gemini** com chave do usuário (config.toml `gemini_api_key`).
- Detector: **paddle** (comic revertido por fragmentar manuscrita).
- Pipeline traduzindo (regions/items nos logs), overlay top-level pintando
  em cima do texto, escuta de scroll ativa, zero erros de renderização na
  última verificação.
- 45 testes passando.

### Sessão 06/09/2026 (3) — causa raiz dos engasgos: OCR Paddle lento (aplicada)

Medição isolada (captura real `reader_debug_output/last_capture.png`):
- `engine.ocr(crop)` com doc-preprocessor: **1643ms por caixa** (e primeiro call
  construía 5 modelos, ~42s)
- `engine.ocr(crop)` sem doc-preprocessor: **59ms por caixa** (28x mais rápido)
- engine japan (via fallback) usava doc-preprocessor e ~3s por caixa

Causas raiz corrigidas:
1. **Doc-preprocessor do PaddleX 3.x** (`use_doc_orientation_classify`,
   `use_doc_unwarping`, `use_textline_orientation`) era default ligado e
   carregava UVDoc + PP-LCNet (~1.5s/crop). Agora `_build_engine` desliga
   todos.
2. **`japan_PP-OCRv5_mobile_rec` não existe** no zoo do PaddleX 3.x (só
   `japan_PP-OCRv3_mobile_rec`). O nome inválido fazia o construtor cair no
   fallback lento (doc-preproc ligado + rec default). Agora há um mapa
   `v5_langs` (en/korean/latin/eslav/th/el/arabic/te/ta/devanagari/cyrillic)
   que usa o v5 por idioma; japan e demais usam `PP-OCRv5_mobile_rec` genérico.
3. **Crash oneDNN na 2ª inferência** (`ConvertPirAttribute2RuntimeAttribute
   not supported`) — `FLAGS_use_mkldnn`/`FLAGS_use_onednn`/`PADDLE_PDX_...`
   agora setados no topo de ocr.py ANTES de qualquer import do paddle (o
   `setdefault` no `__init__` era tardio).
4. **Fallback do comic rodava Paddle no frame inteiro** quando `detect_lines`
   acha 0 caixas — segundos de freeze + lixo. Agora `continue` (confia no
   detector).

Resultado: OCR de 3 caixas caiu de **50s → 1.85s** (27x), com cache de OCR
mantendo frames repetidos em ~0ms. Combinado com async translate + Gemini
timeout capado em 4s, a página traduz em ~2-3s sem travar o UI.

Problemas encontrados ao vivo e corrigidos:

- **dxcam NameError**: `_grab_dxcam` usava `screen` fora do escopo e
  `_time_module` (inexistente) — todo capture falhava e o app ficava preso
  re-analisando `last_frame` velho. Corrigido: `screen` obtido dentro do
  método e `time.time()`.
- **Comic fallback congelava**: quando `detect_lines` não achava caixas, o
  `ComicTextOcr.recognize` caía num fallback que rodava PaddleOCR no FRAME
  INTEIRO (segundos de freeze + texto lixo). Agora confia no detector: sem
  caixas → nada reconhecido.
- **rec_model "mobile" usava server_rec**: `_build_engine` só setava o nome
  do modelo de reconhecimento quando `rec_model == "server"`, então o PaddleX
  usava o `PP-OCRv5_server_rec` lento mesmo com mobile no config. Agora
  explicita `{lang}_PP-OCRv5_mobile_rec` (modelos baixados no warmup).
- **Warmup do comic não pré-carregava o Paddle**: `ComicTextOcr.warmup` só
  inicializava o ONNX; os 5 modelos Paddle (~42s) eram construídos no primeiro
  frame. Agora `warmup()` também pré-constroi os engines Paddle.
- **Executor shutdown race**: `cannot schedule new futures after shutdown`
  (sweep da GUI após stop). `_dispatch_translation` agora reconstrói o
  executor e tenta de novo.
- **Gemini congestionado segurava a fila**: medido — Google 720ms vs Gemini
  timeout 15s+. A cadeia do Gemini agora usa `timeout_seconds` capado em
  ~4s para o LLM; quando congestionado cai pro Google MT em ~1s.
  Config: `detector="comic"`, `gemini_min_interval=1.0`.

### Sessão 07/09/2026 — reprodução no Demonicscans e flicker corrigido

- Link testado: `https://demonicscans.org/title/The-Great-Mage-of-the-Hero%2527s-Party-Reincarnates/chapter/153/1`.
- **Alternância inglês/PT-BR**: condição de corrida confirmada. O OCR/sweep
  detectava pequenas mudanças de frame, removia a tradução válida e colocava o
  texto provisório em inglês enquanto a próxima tradução ainda estava em voo.
- Correção em `pipeline.py`: `_upsert_overlay_items` já não deixa provisório
  sobrescrever tradução real; agora `_remove_items_overlapping` também mantém
  traduções reais durante alterações pequenas. Mudanças grandes de scroll
  continuam usando fade, e uma nova tradução substitui a antiga quando chega.
- 45 testes passando após a correção; app reiniciado com o link acima.
- Diagnóstico de performance da captura real: `engine.ocr(crop)` com
  doc-preprocessor desligado ~59ms; OCR comic completo caiu de ~50s para
  ~1,85s. A primeira inicialização ainda pode carregar modelos japan/korean
  uma vez, mas o warmup agora faz isso antes da leitura.

### Sessão 06/09/2026 — otimizações de velocidade (todas aplicadas)

- **Pacing do Gemini**: `MIN_INTERVAL` 4,5s → configurável `gemini_min_interval`
  (default 1,0s, em config.toml). `_throttle` agora cede ao Google quando a
  espera passa de 1s. Traduções novas de página não pagam mais o piso de 4,5s.
- **Comic detector recalibrado e reativado** (`detector = "comic"` em
  config.toml): `LINE_THRESHOLD` 0,35 → 0,25 + `_merge_close_boxes` (junta
  fragmentos de linha manuscrita por banda vertical/gap ~14px). Detecção ONNX
  via DirectML (~80ms GPU) em vez de Paddle CPU no frame inteiro.
- **Pipeline paralelo**: `MangaTranslationPipeline.async_translate=True` no
  reader. `_translate_regions` resolve cache de forma síncrona e despacha os
  misses para um `ThreadPoolExecutor(max_workers=1)`; o overlay é atualizado da
  thread assíncrona (sinal Qt é thread-safe). Guarda `_in_flight` por
  visual_hash evita re-despachar o mesmo balão em sweeps repetidos.
  `TranslationMemory` virou thread-safe (RLock + check_same_thread=False).
- **Guards reduzidos**: settle do scroll 1,2s → 0,6s; sweep GUI 2,0s → 1,0s;
  sweep runtime 4,0s → 1,5s; timers sweep/capture 1,5s/2,0s → 1,0s/1,0s.
- **Bug corrigido**: `ReaderWindow._last_settle_capture` nunca era inicializado
  (AttributeError repetido no stderr matava o sweep da GUI; só sobrava o loop
  lento do runtime). Agora init = 0.0.
- Referência Gaminik (D:\Downloads\Gaminik-x64-5.5-portable): app .NET/WPF com
  DeepL + OCR local ONNX + WebView2 — pipeline curto e paralelo, sem LLM com
  pacing. Inspiração para as mudanças acima, mas sem re-engenharia reversa.

## 5. Problemas abertos (conhecidos, com status)

1. **PySide6 6.11.1 — "Slot not found" / paint SystemError intermitentes**:
   morriam overlays/timers no meio da sessão (2 ocorrências). Mitigações
   aplicadas: `_ACTIVE_WINDOW` global forte + overlay top-level. NÃO está
   100% provado que sumiu — monitorar stderr; se recorrer, considerar pin
   do PySide6 (6.7/6.8) ou migrar os timers para conexões com lambdas+
   referências fortes.
2. **Lazy-load do site desloca o layout sem scroll** → caixas ficam ~1 ciclo
   dessincronizadas (a varredura de 1,5s re-analisa e corrige). Fechar de
   vez: invalidação por content-hash do frame (independente de scroll).
3. **Comic ONNX fragmenta manuscrita**: calibrado em 06/09 (LINE_THRESHOLD
   0.35 → 0.25 + merge de caixas com gap 14px). VALIDAR em capturas reais se a
   fusão de fragmentos recombina as linhas; se ainda fragmentar, ajustar
   `MERGE_GAP` (↑) ou `LINE_THRESHOLD` (↓). Ativo por padrão em config.toml;
   reverter para `paddle` se a qualidade cair.
4. **MyMemory (fallback) responde pt-PT** ("Estás a pensar") — qualidade
   inferior; só entra quando Google falha.
5. **paddle2onnx quebrado com paddle 3.3.1 (DLL)** → leitura em ONNX
   bloqueada. Alternativas: modelos ONNX pré-convertidos do RapidOCR
   (ch_PP-OCRv5) ou env separado com paddle 2.6 só para exportar.
6. **Gemini 503 em pico** (noite): chain cai para Google MT automaticamente.
   Possível melhora: fallback de modelo (gemini-3.6-flash → outro gratuito).
   Com `gemini_min_interval=1.0` os 429s ficam mais frequentes em leitura
   rápida — a cadeia absorve caindo pro Google (rápido, qualidade inferior).
7. **Anúncios do site**: popups/banners são traduzidos (rotacionam → caixa
   destoa). Atenuar: região bem ajustada; não é bug.

## 6. Roteiro futuro (prioridade sugerida)

1. **Empacotamento**: instalador Inno Setup + ícone + atalho (mesma
   ferramenta do app de referência). Transforma o projeto em "app de gente".
2. **Perfis por site/obra**: salvar região + provedor + FPS + idioma e
   restaurar com 1 clique.
3. **Comic detector calibrado** (problema 3) — maior precisão em fonte
   manuscrita, mantendo a velocidade da GPU.
4. **Legendas ao vivo**: áudio do sistema → Whisper (faster-whisper) →
   tradução (o app de referência tem esse pilar).
5. **manga-ocr** para japonês vertical (modelo dedicado, melhor que Paddle
   para balões JP).
6. **Leitura em ONNX** (problema 5) — rec ~5-10x mais rápido.
7. **Shell nativo WPF** (`UltraMangaReader.Wpf`, WebView2) — direção de
   produto de longo prazo, já esboçada.

## 7. Lições/armadilhas deste projeto (para futuras sessões)

- PySide6 6.11: widgets filhos do QWebEngineView morrem/perdem slots
  (Chromium recomposite) → sempre overlay top-level + referência global
  forte (`_ACTIVE_WINDOW`).
- `webview.grab(QRect())` explícito retorna null no 6.11 — usar `grab()`
  puro quando não houver região.
- Endpoints gratuitos de tradução mudam/modelos são aposentados (Gemini
  2.0 → 3.6): tratar 404 de modelo com mensagem clara.
- deep-translator: endpoint quebrado; nunca reintroduzir.
- Nunca deixar o campo de UI sobrescrever config com valor vazio (bug da
  chave do Gemini).
- Testar SEMPRE com PrintWindow/SendKeys via PowerShell quando o módulo
  CUA estiver indisponível (ver histórico de comandos nesta sessão).
- Primeira inferência DML de um modelo compila shaders (~2s): pré-aquecer
  no arranque (`warmup_ocr` já faz).

## 8. Como testar manualmente (fluxo validado)

1. `START_READER.bat` → abre no capítulo
2. Fechar popups do site
3. `Selecionar região` sobre a coluna dos quadrados (liga modo Contínuo +
   runtime automaticamente)
4. Rolar e ler: caixas velhas somem no movimento; balões novos acendem
   (original cinza) e viram pt-BR em ~1-2,5s
5. `Ctrl+Shift+T` / `Traduzir região` = one-shot; `Inspecionar OCR` = dump
   de diagnóstico
