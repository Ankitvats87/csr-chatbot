# Start the full bot locally: FastAPI app + Telegram polling bridge.
# Usage (from the telegram-rag-bot folder):
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
#
# Requires a DEV bot token in .env (see local_poll.py guard) — it will
# refuse to hijack the production bot's webhook.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = "$env:USERPROFILE\venvs\csrbot\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Project: $root"
Write-Host "Python:  $py"
Write-Host ""
Write-Host "Starting FastAPI app on http://localhost:8000 (admin at /admin)..."
$app = Start-Process -PassThru -NoNewWindow $py -ArgumentList "-X", "utf8", "-m", "uvicorn", "app.main:app", "--port", "8000"

Start-Sleep -Seconds 5
Write-Host "Starting Telegram polling bridge (dev bot)..."
try {
    & $py -X utf8 scripts\local_poll.py
} finally {
    if ($app -and -not $app.HasExited) {
        Write-Host "Stopping FastAPI app (pid $($app.Id))..."
        Stop-Process -Id $app.Id -Force -Confirm:$false
    }
}
