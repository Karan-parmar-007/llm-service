# Studio services

One Flask service, `gateway/`, sits in front of every model Studio calls. The Studio
backend never runs a model; it POSTs to a gateway URL from its `.env` with the
`X-Flow-Token` header and relays the answer (or a `202` queue ticket).

| Half | Routes | Switch |
|---|---|---|
| LLM (Ollama) | `/format`, `/chat`, pass-through `/v1/*` and `/api/*` | `LLM_ENABLED=1` |
| Speech (Tara, faster-whisper on CUDA) | `/transcribe` | `SPEECH_ENABLED=1` |

Shared by both: `/health`, `/queue`, `/jobs/{id}`. The LLM has three lanes (voice, chat,
rag), one request each; speech has one slot. Extra requests are queued and the caller
gets a job id to poll. Speech libraries are imported only when speech is on, so the LLM
VPS installs `requirements.txt` alone.

| Machine | `.env` | Port |
|---|---|---|
| LLM VPS (12 GB RAM) | `LLM_ENABLED=1`, `SPEECH_ENABLED=0` | `BIND=0.0.0.0:8011` |
| Karan's PC (GPU) | `LLM_ENABLED=1`, `SPEECH_ENABLED=1` | `BIND=0.0.0.0:8003` |

Keep `FLOW_TOKEN` identical here and in `studio-backend`.

## LLM VPS (Docker, arm64)

The LLM half ships as an image, `ghcr.io/karan-parmar-007/llm-service`, built for
arm64 by CI. On the server it runs from `/srv/docker/apps/studio-gateway/` using the
compose file and `.env.example` in `new_deployment/studio-gateway`. The container uses
host networking, so it reaches Ollama on `127.0.0.1:11434` and its port 8011 stays
under ufw (`sudo ufw allow from <web-vps-ip> to any port 8011 proto tcp`).

## Keep-alive (LLM VPS)

With `KEEPALIVE_ENABLED=1` the gateway keeps Ollama generating in a duty cycle while
nobody uses it, so Oracle never sees the instance as idle. Every user LLM request
cancels the running keep-alive generation immediately (the stream is closed and
Ollama stops at the next token), and keep-alive waits `KEEPALIVE_QUIET_SECONDS` after
the last user request before starting again. Its state is in `/health` under
`keepalive`. This replaces any separate keep-alive script on the server.

## PC

```bash
cd gateway && cp .env.example .env    # LLM_ENABLED=1, SPEECH_ENABLED=1, BIND=0.0.0.0:8003
./run.sh                               # run.ps1 on Windows
```

Expose port 8003 to the web VPS (port forward or a tunnel) and paste that one URL into
both "speech · local" and "llm · local" on the WhisperFlow page.

## Tests

```bash
cd gateway && pip install -r requirements-dev.txt && pytest
```

## Deploy

Push to `master`: tests run, CI builds the arm64 image and pushes it to GHCR, then on the
LLM VPS it pulls the image, recreates only the `studio-gateway` container (waiting for
its health check), removes the old image, and trims old GHCR versions. Secrets:
`LLM_HOST`, `LLM_USER`, `LLM_SSH_KEY`.
