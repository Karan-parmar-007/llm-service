"""Studio gateway: one Flask service for every model Studio calls.

LLM (Ollama): /format, /chat, and a pass-through of /v1/* and /api/* for RAG.
Three lanes run at once (voice, chat, rag); a second request on a busy lane is
queued and the caller gets a job id to poll.

Speech (Tara, faster-whisper on CUDA): /transcribe, one at a time, queued the same way.

Each half is switched on by env, so one codebase runs everywhere:
  LLM VPS: LLM_ENABLED=1 SPEECH_ENABLED=0  (speech code and its libraries never load)
  PC:      LLM_ENABLED=1 SPEECH_ENABLED=1  (one process, one port, one link)
"""

from __future__ import annotations

import base64
import functools
import importlib.util
import json
import logging
import os

import httpx
from flask import Flask, Response, request, stream_with_context

from lanes import Gate, QueueFull, gate
from points import shape_points
from roman import romanize

OLLAMA = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q5_K_M")
TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
MODEL_DIR = os.environ.get("TARA_DIR", os.path.expanduser("~/models/tara-ct2"))

speech_gate = Gate(("speech",))
_model = None

_LLM_PATHS = ("/format", "/chat", "/v1/", "/api/")


def llm_enabled() -> bool:
    return os.environ.get("LLM_ENABLED", "1").strip() != "0"


log = logging.getLogger("gateway")


def speech_requested() -> bool:
    return os.environ.get("SPEECH_ENABLED", "0").strip() == "1"


@functools.lru_cache(maxsize=4)
def _speech_libs_present(stub: bool) -> bool:
    """numpy always; faster_whisper too unless the model is stubbed (tests)."""
    needed = ("numpy",) if stub else ("numpy", "faster_whisper")
    return all(importlib.util.find_spec(name) is not None for name in needed)


def speech_libs_present() -> bool:
    return _speech_libs_present(os.environ.get("TARA_STUB") == "1")


def speech_enabled() -> bool:
    """Speech runs only when asked for AND its libraries are installed. The LLM-only
    image has no speech libraries, so a stray SPEECH_ENABLED=1 there turns speech off
    instead of crashing the whole gateway."""
    return speech_requested() and speech_libs_present()

FORMAT_SYSTEM = """You clean dictated speech into the text the speaker meant to send.

Rules:
- Do not translate. English stays English. Hindi stays Hindi.
- script=roman: write Hindi in Latin letters. "kal milte hain" is right. "कल मिलते हैं" is wrong. "see you tomorrow" is wrong because that is a translation.
- script=devanagari: write Hindi in Devanagari and English in Latin letters.
- Drop fillers only when they add nothing. Keep "matlab" and "toh" when they change the meaning.
- Apply self-corrections. "5 pm, actually 6" becomes "6 pm".
- When they number points ("point 1", "point 2", "pehla", "doosra"), output a numbered list and nothing else.
- Do not add facts, greetings, labels, or quotes around the result.
- Reply with the final text only.
"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.before_request
def require_flow_token():
    expected = os.environ.get("FLOW_TOKEN", "").strip()
    if not expected or request.path in {"/health", "/queue"}:
        return None
    sent = request.headers.get("X-Flow-Token", "")
    if sent == expected:
        return None
    return {"detail": "Send the Studio Flow token in X-Flow-Token."}, 401


@app.before_request
def only_enabled_parts():
    path = request.path
    if path.startswith(_LLM_PATHS) and not llm_enabled():
        return {"detail": "This gateway does not serve the LLM."}, 404
    if path == "/transcribe" and not speech_enabled():
        return {"detail": "This gateway does not serve speech."}, 404
    return None


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, base_url=OLLAMA)


def _json_error(message: str, status: int):
    return {"detail": message}, status


def lane_for(path: str) -> str:
    header = (request.headers.get("X-Lane") or "").strip().lower()
    if header in ("voice", "chat", "rag"):
        return header
    trimmed = path.rstrip("/")
    if trimmed.endswith(("/embeddings", "/embed")):
        return "rag"
    return "chat"


def queued_body(job, owner: Gate | None = None) -> dict:
    return {
        "id": job.id,
        "status": "queued",
        "lane": job.lane,
        "position": job.position,
        "detail": "Your request is queued.",
        "lanes": (owner or gate).snapshot(),
    }


def call_upstream(method: str, url: str, body: dict | None):
    try:
        with _client() as client:
            upstream = client.request(method, url, json=body)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Ollama did not answer: {exc}") from exc
    headers = getattr(upstream, "headers", None) or {}
    content_type = headers.get("content-type", "application/json")
    if hasattr(upstream, "json"):
        try:
            payload = upstream.json()
        except ValueError:
            payload = {"body": getattr(upstream, "text", "")}
    else:
        raw = upstream.content
        payload = json.loads(raw) if "json" in content_type else {"body": raw.decode()}
    if upstream.status_code >= 400:
        raise RuntimeError(str(payload)[:500])
    return payload


def dispatch(lane: str, fn):
    try:
        job = gate.admit(lane, fn)
    except QueueFull as exc:
        return {
            "detail": f"The {exc.lane} line is full. Your request was not queued.",
            "lane": exc.lane,
            "lanes": gate.snapshot(),
        }, 429
    if job.status == "queued":
        return queued_body(job), 202
    try:
        return gate.run_inline(job)
    except RuntimeError as exc:
        return _json_error(str(exc), 503)


def job_view(job, owner: Gate | None = None) -> tuple[dict, int]:
    if job.status == "queued":
        return queued_body(job, owner), 202
    if job.status == "running":
        return {
            "id": job.id,
            "status": "running",
            "lane": job.lane,
            "position": 0,
            "detail": "Your request is running.",
            "lanes": (owner or gate).snapshot(),
        }, 202
    if job.status == "error":
        return {
            "id": job.id,
            "status": "error",
            "lane": job.lane,
            "detail": job.error or "The request failed.",
        }, 502
    return {
        "id": job.id,
        "status": "done",
        "lane": job.lane,
        "position": 0,
        "result": job.result,
    }, 200


@app.get("/health")
def health():
    body: dict = {"status": "ok"}
    status = 200
    if llm_enabled():
        try:
            with _client() as client:
                upstream = client.get("/api/tags")
                upstream.raise_for_status()
                names = [item.get("name") for item in upstream.json().get("models", [])]
            body["llm"] = {"default_model": DEFAULT_MODEL, "models": names, **gate.snapshot()}
        except httpx.HTTPError as exc:
            body["llm"] = {"status": "down", "detail": f"Ollama did not answer: {exc}"}
            body["status"] = "degraded"
            status = 503
    else:
        body["llm"] = "off"
    if speech_enabled():
        ready = _model is not None or os.environ.get("TARA_STUB") == "1"
        body["speech"] = {"status": "ok" if ready else "loading", "model": MODEL_DIR, **speech_gate.snapshot()}
    elif speech_requested():
        body["speech"] = "off: SPEECH_ENABLED=1 but the speech libraries are not installed"
    else:
        body["speech"] = "off"
    return body, status


@app.get("/queue")
def queue():
    return {"llm": gate.snapshot(), "speech": speech_gate.snapshot()}


@app.get("/jobs/<job_id>")
def job(job_id: str):
    found = gate.get(job_id)
    if found is not None:
        return job_view(found)
    found = speech_gate.get(job_id)
    if found is not None:
        return job_view(found, speech_gate)
    return _json_error("Unknown request", 404)


@app.route("/v1/<path:path>", methods=["GET", "POST", "DELETE"])
@app.route("/api/<path:path>", methods=["GET", "POST", "DELETE"])
def proxy(path: str):
    if request.method == "GET":
        return forward_now(request.path)
    body = request.get_json(silent=True)
    if isinstance(body, dict) and request.path.startswith("/v1") and "model" not in body:
        body = dict(body)
        body["model"] = DEFAULT_MODEL
    url = request.path
    if request.query_string:
        url = f"{url}?{request.query_string.decode()}"
    stream = bool(isinstance(body, dict) and body.get("stream"))
    work = dict(body) if isinstance(body, dict) else body
    if isinstance(work, dict):
        work["stream"] = False

    method = request.method

    def run():
        return call_upstream(method, url, work if isinstance(work, dict) else None)

    try:
        job = gate.admit(lane_for(request.path), run)
    except QueueFull as exc:
        return {
            "detail": f"The {exc.lane} line is full. Your request was not queued.",
            "lane": exc.lane,
            "lanes": gate.snapshot(),
        }, 429
    if job.status == "queued":
        return queued_body(job), 202
    if stream:
        return stream_now(request.method, url, body, job)
    try:
        return gate.run_inline(job)
    except RuntimeError as exc:
        return _json_error(str(exc), 503)


def forward_now(url: str):
    try:
        with _client() as client:
            upstream = client.request("GET", url)
    except httpx.HTTPError as exc:
        return _json_error(f"Ollama did not answer: {exc}", 503)
    return Response(
        upstream.content,
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type", "application/json"),
    )


def stream_now(method: str, url: str, body, job):
    def generate():
        client = _client()
        try:
            with client.stream(method, url, json=body) as upstream:
                if upstream.status_code >= 400:
                    yield upstream.read()
                    return
                for chunk in upstream.iter_bytes():
                    yield chunk
        finally:
            client.close()
            gate.finish(job)

    return Response(stream_with_context(generate()), content_type="text/event-stream")


@app.post("/format")
def format_text():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text") or "").strip()
    if not text:
        return _json_error("Text is required", 400)
    script = "devanagari" if str(body.get("script") or "").lower() == "devanagari" else "roman"
    tone = str(body.get("tone") or "plain").strip() or "plain"
    model = str(body.get("model") or DEFAULT_MODEL)
    source = romanize(text) if script == "roman" else text
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
        "messages": [
            {"role": "system", "content": FORMAT_SYSTEM},
            {"role": "user", "content": f"script={script}\ntone={tone}\n\n{source}"},
        ],
    }

    def run():
        raw = call_upstream("POST", "/v1/chat/completions", payload)
        try:
            content = raw["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Ollama returned an unexpected body") from None
        if script == "roman":
            content = romanize(content)
        content = shape_points(content)
        return {"text": content, "model": model}

    return dispatch("voice", run)


@app.post("/chat")
def chat_text():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text") or "").strip()
    if not text:
        return _json_error("Text is required", 400)
    model = str(body.get("model") or DEFAULT_MODEL)
    payload = {
        "model": model,
        "temperature": 0.7,
        "stream": False,
        "messages": [{"role": "user", "content": text}],
    }

    def run():
        raw = call_upstream("POST", "/v1/chat/completions", payload)
        try:
            content = raw["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Ollama returned an unexpected body") from None
        return {"text": content, "model": model}

    return dispatch("chat", run)


# ---------------------------------------------------------------- speech
# Everything below only runs with SPEECH_ENABLED=1. numpy, av and faster_whisper
# are imported lazily so the LLM VPS never needs them installed.


def get_model():
    global _model
    if os.environ.get("TARA_STUB") == "1":
        return None
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
    return _model


def transcribe(data: bytes) -> dict:
    from audio import pcm_from_bytes

    pcm = pcm_from_bytes(data)
    if os.environ.get("TARA_STUB") == "1":
        heard = os.environ.get("TARA_STUB_TEXT", "point 1 home. point 2 keyboard.")
        return {"text": heard, "language": "en"}
    model = get_model()
    audio = pcm.astype("float32") / 32768.0
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = "".join(segment.text for segment in segments).strip()
    return {"text": text, "language": getattr(info, "language", "") or ""}


def speech_dispatch(fn):
    from audio import BadAudio

    try:
        job = speech_gate.admit("speech", fn)
    except QueueFull:
        return {
            "detail": "The speech line is full. Your request was not queued.",
            "lane": "speech",
            "lanes": speech_gate.snapshot(),
        }, 429
    if job.status == "queued":
        return queued_body(job, speech_gate), 202
    try:
        return speech_gate.run_inline(job)
    except BadAudio as exc:
        return _json_error(str(exc), 400)
    except RuntimeError as exc:
        return _json_error(str(exc), 503)


@app.post("/transcribe")
def transcribe_route():
    body = request.get_json(silent=True) or {}
    raw = body.get("audio_base64")
    if not isinstance(raw, str) or not raw.strip():
        return _json_error("audio_base64 is required", 400)
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception:
        return _json_error("audio_base64 is required", 400)
    return speech_dispatch(lambda: transcribe(data))


if speech_requested() and not speech_libs_present():
    log.warning(
        "SPEECH_ENABLED=1 but numpy/faster_whisper are not installed; speech is off. "
        "Set SPEECH_ENABLED=0 on LLM-only machines."
    )

if speech_enabled() and os.environ.get("TARA_EAGER", "1") == "1" and os.environ.get("TARA_STUB") != "1":
    # Load and warm the model at start so the first dictation is not slow.
    import numpy as np

    get_model().transcribe(np.zeros(16000, dtype="float32"), beam_size=1, vad_filter=True)
