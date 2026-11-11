"""Studio gateway: one Flask service for every model Studio calls.

LLM (Ollama): /format, /chat, /rerank, and a pass-through of /v1/* and /api/* for RAG.
Five lanes run at once: voice, chat and rag answer people; embed carries embeddings
and reranking so they never wait behind a long answer; ingest carries bulk document
work and steps aside while anyone is waiting. Format, chat, and transcribe answer
immediately with a job id and finish in the background, so a slow model does not
hold the HTTP connection open. A second request on a busy lane waits behind the first;
a caller that sends X-Wait holds its connection instead and still gets its stream.

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
import re
import threading

import httpx
from flask import Flask, Response, request, stream_with_context

import rerank
from keepalive import KeepAlive, ollama_burn
from lanes import LANES, USER_LANES, Gate, QueueFull, gate
from points import shape_points
from roman import romanize

OLLAMA = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _chat_path(path: str) -> bool:
    return path.startswith("/v1/chat") or path.startswith("/api/chat") or path.startswith("/api/generate")
TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
MODEL_DIR = os.environ.get("TARA_DIR", os.path.expanduser("~/models/tara-ct2"))

speech_gate = Gate(("speech",))
_model = None

_LLM_PATHS = ("/format", "/chat", "/v1/", "/api/", "/rerank")
# Longest a caller may hold its connection waiting for a lane (X-Wait header).
MAX_WAIT_SECONDS = 300.0
# Longest bulk ingest work steps aside for user requests before it runs anyway.
INGEST_YIELD_SECONDS = float(os.environ.get("INGEST_YIELD_SECONDS", "600"))


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
- Mixed Hindi and English stays mixed exactly as spoken. Never translate one part into the other language, and keep every word the speaker meant (a self-correction below is the one exception).
  "आज लंच पे चलते हैं, I will book the table" becomes "Aaj lunch pe chalte hain, I will book the table."
  "please call him, वो कब से वेट कर रहा है" becomes "Please call him, woh kab se wait kar raha hai."
  "मेरा लैपटॉप हैंग हो गया है" becomes "Mera laptop hang ho gaya hai."
- Drop fillers only when they add nothing. Keep "matlab" and "toh" when they change the meaning.
- Self-corrections override everything else: when the speaker changes their mind, keep only their final choice. Delete the earlier version and the correction words. Check for this before you answer. Correction words include: actually, I mean, sorry, no, wait no, scratch that, make that, or let it be, let it be, nahi, nahi nahi, rehne do, ek minute, matlab nahi.
  "call at 5 pm, actually 6" becomes "Call at 6 pm."
  "let's connect at 6 or 7, or let it be, let's connect at 8" becomes "Let's connect at 8."
  "send it to Rahul, I mean send it to Priya" becomes "Send it to Priya."
  "buy two tickets, make that four" becomes "Buy four tickets."
  "kal 4 baje, nahi nahi 5 baje" becomes "kal 5 baje"
  "mujhe chai chahiye, rehne do, coffee chahiye" becomes "mujhe coffee chahiye"
  "main Pune ja raha hoon, ek minute, main Mumbai ja raha hoon" becomes "main Mumbai ja raha hoon"
- It is only a correction when a later part replaces an earlier one. Keep everything when the speaker offers choices or uses these words normally:
  "we can meet on Friday or Saturday, whichever suits you" keeps both days, because both are still on offer.
  "I mean it, this is important" becomes "I mean it, this is important." Here "I mean it" is emphasis, not a filler and not a correction, so never drop it.
  "wait for me at the gate" and "sorry for the delay" stay as they are.
- When they number points ("point 1", "point 2", "pehla", "doosra"), output a numbered list and nothing else.
- Do not add facts, greetings, labels, or quotes around the result.
- Never write the words script= or tone=.
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
    if keeper is not None and request.method != "GET" and path.startswith(_LLM_PATHS):
        keeper.user_activity()  # user first: cancel any keep-alive generation now
    if path == "/transcribe" and not speech_enabled():
        return {"detail": "This gateway does not serve speech."}, 404
    return None


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, base_url=OLLAMA)


def _json_error(message: str, status: int):
    return {"detail": message}, status


def lane_for(path: str) -> str:
    header = (request.headers.get("X-Lane") or "").strip().lower()
    if header in LANES:
        return header
    trimmed = path.rstrip("/")
    if trimmed.endswith(("/embeddings", "/embed")):
        return "embed"
    return "chat"


def wait_seconds() -> float:
    """X-Wait: hold this connection up to N seconds for a lane instead of a 202."""
    raw = (request.headers.get("X-Wait") or "").strip()
    try:
        return min(MAX_WAIT_SECONDS, max(0.0, float(raw))) if raw else 0.0
    except ValueError:
        return 0.0


def yield_to_users() -> None:
    """Bulk ingest work starts only when no person is waiting on an answer."""
    waited = 0.0
    while gate.busy(USER_LANES) and waited < INGEST_YIELD_SECONDS:
        threading.Event().wait(0.5)
        waited += 0.5


def busy_body(lane: str, waited: bool) -> tuple[dict, int]:
    detail = (
        f"The {lane} line stayed busy. Try again in a minute."
        if waited
        else f"The {lane} line is full. Your request was not queued."
    )
    return {"detail": detail, "code": "busy", "lane": lane, "lanes": gate.snapshot()}, 429


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


def _answer(job, owner: Gate | None = None):
    """Hand back a finished result, or a job id the caller can poll.

    The GPU often needs longer than a quiet reverse tunnel will stay open. The
    HTTP call returns immediately and the model keeps running on this machine.
    """
    if job.status == "running":
        threading.Thread(target=(owner or gate)._run, args=(job,), daemon=True).start()
    if job.status == "done":
        return job.result
    if job.status == "error":
        return _json_error(job.error or "The request failed.", 503)
    return job_view(job, owner)


def dispatch(lane: str, fn):
    try:
        job = gate.admit(lane, fn)
    except QueueFull as exc:
        return busy_body(exc.lane, waited=False)
    return _answer(job)


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


def _llm_busy() -> bool:
    board = gate.snapshot()
    return board["running"] > 0 or any(lane["queued"] for lane in board["lanes"].values())


def _env_seconds(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


keeper: KeepAlive | None = None
if llm_enabled() and os.environ.get("KEEPALIVE_ENABLED", "0").strip() == "1":
    keeper = KeepAlive(
        _llm_busy,
        ollama_burn(OLLAMA, DEFAULT_MODEL),
        busy_seconds=_env_seconds("KEEPALIVE_BUSY_SECONDS", 20),
        idle_seconds=_env_seconds("KEEPALIVE_IDLE_SECONDS", 40),
        quiet_seconds=_env_seconds("KEEPALIVE_QUIET_SECONDS", 30),
    )
    keeper.start()


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
    body["keepalive"] = keeper.snapshot() if keeper is not None else "off"
    if not rerank.wanted():
        body["rerank"] = "off"
    else:
        body["rerank"] = "ok" if rerank.files_present() else "off: run fetch_reranker.py"
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
    if isinstance(body, dict) and (request.path.startswith("/v1") or _chat_path(request.path)):
        body = dict(body)
        if request.path.startswith("/v1") and "model" not in body:
            body["model"] = DEFAULT_MODEL
        if _chat_path(request.path):
            body["think"] = False
    url = request.path
    if request.query_string:
        url = f"{url}?{request.query_string.decode()}"
    stream = bool(isinstance(body, dict) and body.get("stream"))
    work = dict(body) if isinstance(body, dict) else body
    if isinstance(work, dict):
        work["stream"] = False

    method = request.method
    lane = lane_for(request.path)

    def run():
        if lane == "ingest":
            yield_to_users()
        return call_upstream(method, url, work if isinstance(work, dict) else None)

    hold = wait_seconds()
    try:
        job = gate.admit(lane, run, wait=hold > 0)
    except QueueFull as exc:
        return busy_body(exc.lane, waited=False)
    if job.status == "queued":
        if not hold:
            return queued_body(job), 202
        if not gate.wait_turn(job, hold):
            return busy_body(lane, waited=True)
    if stream:
        if lane == "ingest":
            yield_to_users()
        return stream_now(request.method, url, body, job)
    try:
        return gate.run_inline(job)
    except RuntimeError as exc:
        return _json_error(str(exc), 503)


@app.post("/rerank")
def rerank_route():
    """Score passages against a query with the cross-encoder. Body:
    {"query": str, "documents": [str], "top_n": int?} -> {"results": [{"index", "relevance_score"}]}
    """
    if not rerank.enabled():
        return _json_error("This gateway has no reranker.", 404)
    body = request.get_json(silent=True) or {}
    query = str(body.get("query") or "").strip()
    documents = body.get("documents")
    if not query or not isinstance(documents, list) or not documents:
        return _json_error("query and documents are required", 400)
    if len(documents) > 64:
        return _json_error("At most 64 documents per call", 400)
    docs = [str(item)[:6000] for item in documents]
    top_n = body.get("top_n")
    top_n = int(top_n) if isinstance(top_n, int) and top_n > 0 else None
    try:
        job = gate.admit("embed", lambda: rerank.rerank(query[:2000], docs, top_n), wait=True)
    except QueueFull as exc:
        return busy_body(exc.lane, waited=False)
    if not gate.wait_turn(job, wait_seconds() or 60):
        return busy_body("embed", waited=True)
    try:
        return {"results": gate.run_inline(job)}
    except Exception as exc:  # noqa: BLE001
        log.exception("rerank failed")
        return _json_error(f"Reranking failed: {exc}", 503)


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
    model = str(body.get("model") or DEFAULT_MODEL)
    # The model gets the transcript as spoken (Devanagari included) and writes the Hinglish
    # itself; that spells words properly (office, tabiyat, karenge). romanize() only runs
    # afterwards as a safety net so no Devanagari can reach a script=roman result.
    source = text
    script_rule = (
        "Write Hindi in Devanagari and English in Latin letters."
        if script == "devanagari"
        else "Write Hindi words in Latin letters the way people type Hinglish (kal, nahi, karenge, tabiyat). English words that arrive in Devanagari get their normal English spelling (मीटिंग is meeting, ऑफिस is office)."
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
        # Thinking models (qwen3.x) otherwise spend the whole token budget reasoning and
        # return empty content. Ollama's /v1 endpoint ignores "think" and honours this.
        "think": False,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": FORMAT_SYSTEM + "\n- " + script_rule},
            {"role": "user", "content": source},
        ],
    }

    def run():
        raw = call_upstream("POST", "/v1/chat/completions", payload)
        try:
            content = raw["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Ollama returned an unexpected body") from None
        lines = content.splitlines()
        while lines and lines[0].strip().lower().startswith(("script=", "tone=")):
            lines.pop(0)
        content = strip_thinking("\n".join(lines).strip())
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
        # Thinking models (qwen3.x) otherwise spend the whole token budget reasoning and
        # return empty content. Ollama's /v1 endpoint ignores "think" and honours this.
        "think": False,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": text}],
    }

    def run():
        raw = call_upstream("POST", "/v1/chat/completions", payload)
        try:
            content = strip_thinking(raw["choices"][0]["message"]["content"].strip())
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
    try:
        job = speech_gate.admit("speech", fn)
    except QueueFull:
        return {
            "detail": "The speech line is full. Your request was not queued.",
            "lane": "speech",
            "lanes": speech_gate.snapshot(),
        }, 429
    return _answer(job, speech_gate)


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
