@echo off
cd /d "%~dp0"
echo Iniciando UltraOCR Reader...
rem ANGLE = renderizacao via Direct3D usando a GPU.
set QT_OPENGL=angle
if exist ".\dist\UltraOCR-local\UltraOCR.exe" (
    start "" ".\dist\UltraOCR-local\UltraOCR.exe"
    exit /b 0
)
.\.venv311\Scripts\python.exe -m manga_rt_translator reader --config config.toml --glossary glossary.toml
pause
