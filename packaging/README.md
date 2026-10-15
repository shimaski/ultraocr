# Executavel local do Windows

Este build cria um EXE de inicializacao em `dist/UltraOCR-local/UltraOCR.exe`.
Ele abre o leitor usando o Python, OCR e dependencias ja instalados em
`.venv311`. Portanto, e pratico para testar neste computador, mas nao e um
pacote portatil para copiar para outro PC.

```powershell
.\.venv311\Scripts\python.exe -m pip install -e ".[packaging]"
.\packaging\build_windows.ps1
```

Ao iniciar, o EXE cria/preserva `config.toml`, `glossary.toml`, banco SQLite e
`startup.log` em `%LOCALAPPDATA%\UltraOCR`. O leitor e iniciado sem janela de
terminal; seus diagnosticos ficam no arquivo de log.

Os modelos OCR do PaddleX nao sao embutidos. Se ainda nao estiverem no cache
local, o PaddleOCR os baixa na primeira inicializacao. Para abrir o leitor,
execute `dist\UltraOCR-local\UltraOCR.exe`. Argumentos opcionais, como
`--url "https://..."`, sao encaminhados ao leitor.
