# Starts the app from the repo root with the virtual environment in .venv.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\.venv\Scripts\Activate.ps1

$port = if ($env:LOCI_PORT) { $env:LOCI_PORT } else { "8770" }
Write-Host "Open http://127.0.0.1:$port in Chrome or Edge."
python -m uvicorn app.main:app --host 127.0.0.1 --port $port
