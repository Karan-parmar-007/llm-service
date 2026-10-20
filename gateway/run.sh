#!/usr/bin/env bash
# Run the gateway on a PC (Linux/WSL). With SPEECH_ENABLED=1 it serves both speech and
# the local LLM from one process and one port.
#   cp .env.example .env    # LLM_ENABLED=1, SPEECH_ENABLED=1, BIND=0.0.0.0:8003, FLOW_TOKEN
#   ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
[ -d venv ] || python3 -m venv venv
if [ "${SPEECH_ENABLED:-0}" = "1" ]; then
  ./venv/bin/pip install -q -r requirements-speech.txt
else
  ./venv/bin/pip install -q -r requirements.txt
fi
export TARA_DIR="${TARA_DIR/#\~/$HOME}"
exec ./venv/bin/gunicorn -w 1 --threads 8 --timeout 300 -b "${BIND:-0.0.0.0:8003}" app:app
