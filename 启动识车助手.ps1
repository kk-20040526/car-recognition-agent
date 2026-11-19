$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "http://127.0.0.1:8000"

function Test-App {
    try {
        $r = Invoke-WebRequest -Uri "$url/api/status" -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

if (-not (Test-App)) {
    schtasks /run /tn "识车助手服务" | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 1
        if (Test-App) { $ready = $true; break }
    }
    if (-not $ready) {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$root\run_server.bat`"" -WindowStyle Hidden
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Seconds 1
            if (Test-App) { $ready = $true; break }
        }
    }
    if (-not $ready) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show("识车助手启动失败，请检查 storage\server.err.log", "识车助手", 0, 16) | Out-Null
        exit 1
    }
}
Start-Process $url