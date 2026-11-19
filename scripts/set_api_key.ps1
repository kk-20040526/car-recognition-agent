# =============================================================================
# Save your DeepSeek API key into .env and verify it.
# Run it from the project folder:
#
#     powershell -ExecutionPolicy Bypass -File scripts\set_api_key.ps1
#
# The key is typed with hidden input and is written only to .env (never echoed).
# =============================================================================
param(
    [string]$Key
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not $Key) {
    $secure = Read-Host -Prompt "Paste your DeepSeek API key here (it will not be displayed)" -AsSecureString
    $Key = [System.Net.NetworkCredential]::new("", $secure).Password
}
$Key = ([string]$Key).Trim()
if (-not $Key) { throw "No key entered." }
if (-not $Key.StartsWith("sk-")) {
    Write-Host "Warning: DeepSeek keys normally start with 'sk-' - saving anyway."
}

$envPath = Join-Path $root ".env"
$lines = @()
if (Test-Path $envPath) { $lines = @(Get-Content -LiteralPath $envPath -Encoding UTF8) }

$found = $false
$out = foreach ($line in $lines) {
    if ($line -match '^\s*DEEPSEEK_API_KEY\s*=') { $found = $true; "DEEPSEEK_API_KEY=$Key" }
    else { $line }
}
if (-not $found) { $out = @($out) + "DEEPSEEK_API_KEY=$Key" }
Set-Content -LiteralPath $envPath -Value $out -Encoding UTF8

Write-Host "[1/2] key saved to $envPath"

# find a python that already has the project dependencies (openai)
$candidates = @(
    (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent (Split-Path -Parent $root)) ".venv\Scripts\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
)
$python = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        & $candidate -c "import openai" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
    }
}
if (-not $python) {
    foreach ($name in @("py -3.12", "py -3.10", "python")) {
        $parts = $name.Split(" ")
        & $parts[0] $parts[1..($parts.Length - 1)] -c "import openai" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $name; break }
    }
}
if (-not $python) {
    Write-Host "Key saved, but no Python with the project dependencies was found."
    Write-Host "Run one of these from the project folder to verify manually:"
    Write-Host "  .venv\Scripts\python scripts\check_deepseek.py"
    exit 0
}

Write-Host "[2/2] verifying the key against the DeepSeek API ..."
if ($python -match " ") {
    $parts = $python.Split(" ")
    & $parts[0] $parts[1..($parts.Length - 1)] "scripts\check_deepseek.py"
} else {
    & $python "scripts\check_deepseek.py"
}
exit $LASTEXITCODE
