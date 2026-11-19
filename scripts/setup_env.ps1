# =============================================================================
# One-click environment setup for the car-recognition agent project.
# Run this ONCE in your own PowerShell (it needs internet access):
#
#     powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Optional switches:
#     -TorchIndexUrl https://download.pytorch.org/whl/cu124   # NVIDIA GPU build
#     -SkipTorch                                              # keep existing torch
# =============================================================================
param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
Write-Host "[1/5] project root: $root"

$python = $null
foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    $parts = $candidate.Split(" ")
    try {
        $version = & $parts[0] $parts[1..($parts.Length - 1)] -c "import sys;print(sys.version.split()[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) {
            $python = $candidate
            Write-Host "[2/5] using interpreter: $candidate ($version)"
            break
        }
    } catch { }
}
if (-not $python) {
    throw "No usable Python found. Install Python 3.12 (64-bit) from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[3/5] creating virtual environment .venv"
    $parts = $python.Split(" ")
    & $parts[0] $parts[1..($parts.Length - 1)] -m venv .venv
} else {
    Write-Host "[3/5] .venv already exists, reusing it"
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

Write-Host "[4/5] installing dependencies"
if ($SkipTorch) {
    & $venvPython -m pip install -r requirements.txt
} else {
    & $venvPython -m pip install torch torchvision --index-url $TorchIndexUrl
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "[5/5] downloading ImageNet pretrained weights and building the knowledge base"
& $venvPython scripts\download_weights.py
& $venvPython scripts\build_knowledge.py

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  $venvPython scripts\build_dataset.py"
Write-Host "  $venvPython src\train.py --preset baseline"
Write-Host "  $venvPython src\train.py --preset improved"
