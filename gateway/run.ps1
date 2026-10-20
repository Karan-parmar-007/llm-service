# Run the gateway on Windows (speech needs an NVIDIA GPU).
#   Copy-Item .env.example .env    # LLM_ENABLED=1, SPEECH_ENABLED=1, BIND=0.0.0.0:8003, FLOW_TOKEN
#   .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim())
  }
}
if (-not (Test-Path venv)) { python -m venv venv }
$req = if ($env:SPEECH_ENABLED -eq "1") { "requirements-speech.txt" } else { "requirements.txt" }
.\venv\Scripts\pip install -q -r $req waitress
$env:TARA_DIR = ($env:TARA_DIR -replace '^~', $HOME)
$bind = if ($env:BIND) { $env:BIND } else { "0.0.0.0:8003" }
$hostPart, $portPart = $bind -split ':', 2
# gunicorn does not run on Windows; waitress does the same job here.
.\venv\Scripts\python -m waitress --host $hostPart --port $portPart --threads 8 app:app
