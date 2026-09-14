@echo off
cd /d "%~dp0"
echo Iniciando UltraOCR Reader...
rem ANGLE = renderizacao via Direct3D usando a GPU (RTX 3070).
rem "software" usava apenas a CPU e causava travadas no scroll.
set QT_OPENGL=angle
.\.venv311\Scripts\python.exe -m manga_rt_translator reader --config config.toml --glossary glossary.toml --url "https://www.google.com"
pause
