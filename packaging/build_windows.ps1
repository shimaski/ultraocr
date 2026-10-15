param(
    [string]$Python = ".\.venv311\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $root $Python
$outputPath = Join-Path $root "dist\UltraOCR-local"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python nao encontrado: $pythonPath"
}
Push-Location $root
try {
    & $pythonPath -m PyInstaller --noconfirm --onefile --windowed `
        --name UltraOCR `
        --distpath $outputPath `
        --workpath (Join-Path $root "build\launcher") `
        --specpath (Join-Path $root "build\launcher") `
        (Join-Path $root "packaging\local_launcher.py")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller terminou com codigo $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Executavel local gerado em: $outputPath\UltraOCR.exe"
