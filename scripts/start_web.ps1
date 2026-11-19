# =============================================================================
# Start the web system (FastAPI + agent) and open the page in the browser.
#
#     powershell -ExecutionPolicy Bypass -File scripts\start_web.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\start_web.ps1 -Port 8080
# =============================================================================
param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not $env:TORCH_HOME) {
    $env:TORCH_HOME = Join-Path $root ".cache\torch"
}

$candidates = @(
    (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent (Split-Path -Parent $root)) ".venv\Scripts\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
)
$python = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        & $candidate -c "import torch, fastapi" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
    }
}
if (-not $python) {
    throw "No Python with the project dependencies was found. Run scripts\setup_env.ps1 first."
}

Write-Host "using python: $python"
Write-Host "starting web system on http://127.0.0.1:$Port  (Ctrl+C to stop)"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port" | Out-Null
}
& $python -m uvicorn src.web.app:app --host 127.0.0.1 --port $Port
