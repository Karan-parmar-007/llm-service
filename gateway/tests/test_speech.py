import base64
import io
import threading
import wave

import pytest

import app as speech
from conftest import finished_body
from lanes import Gate


def wav_bytes(seconds: float = 0.2) -> bytes:
    frames = b"\x00\x00" * int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(speech, "speech_gate", Gate(("speech",)))
    monkeypatch.setenv("FLOW_TOKEN", "")
    return speech.app.test_client()


def test_a_wav_clip_is_transcribed(client):
    payload = {"audio_base64": base64.b64encode(wav_bytes()).decode()}
    response = client.post("/transcribe", json=payload)
    assert finished_body(client, response)["text"].startswith("point 1")


def test_empty_audio_is_refused(client):
    response = client.post("/transcribe", json={"audio_base64": ""})
    assert response.status_code == 400


def test_a_second_clip_is_queued(client, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def blocking(_data):
        started.set()
        release.wait(2)
        return {"text": "done", "language": "en"}

    monkeypatch.setattr(speech, "transcribe", blocking)
    first: dict = {}

    def fire():
        first["response"] = client.post(
            "/transcribe",
            json={"audio_base64": base64.b64encode(wav_bytes()).decode()},
        )

    threading.Thread(target=fire, daemon=True).start()
    assert started.wait(1)
    queued = client.post(
        "/transcribe",
        json={"audio_base64": base64.b64encode(wav_bytes()).decode()},
    )
    assert queued.status_code == 202
    body = queued.get_json()
    assert body["detail"] == "Your request is queued."
    assert body["lane"] == "speech"
    assert body["position"] == 1
    release.set()


def test_token_is_required_when_set(client, monkeypatch):
    monkeypatch.setenv("FLOW_TOKEN", "secret-token")
    response = client.post(
        "/transcribe",
        json={"audio_base64": base64.b64encode(wav_bytes()).decode()},
    )
    assert response.status_code == 401
    allowed = client.post(
        "/transcribe",
        json={"audio_base64": base64.b64encode(wav_bytes()).decode()},
        headers={"X-Flow-Token": "secret-token"},
    )
    assert finished_body(client, allowed)["text"].startswith("point 1")


def test_speech_is_off_unless_enabled(client, monkeypatch):
    monkeypatch.setenv("SPEECH_ENABLED", "0")
    response = client.post("/transcribe", json={"audio_base64": "UklGRg=="})
    assert response.status_code == 404


def test_llm_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "0")
    assert client.post("/format", json={"text": "hi"}).status_code == 404
    assert client.post("/chat", json={"text": "hi"}).status_code == 404


def test_one_jobs_route_serves_both_halves(client, monkeypatch):
    release = __import__("threading").Event()
    started = __import__("threading").Event()

    def blocking(_data):
        started.set()
        release.wait(2)
        return {"text": "done"}

    monkeypatch.setattr(speech, "transcribe", blocking)
    clip = {"audio_base64": base64.b64encode(wav_bytes()).decode()}
    __import__("threading").Thread(target=lambda: client.post("/transcribe", json=clip), daemon=True).start()
    assert started.wait(1)
    queued = client.post("/transcribe", json=clip).get_json()
    polled = client.get(f"/jobs/{queued['id']}")
    assert polled.status_code == 202 and polled.get_json()["lane"] == "speech"
    release.set()


def test_health_reports_each_half(client, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "0")
    body = client.get("/health").get_json()
    assert body["llm"] == "off"
    assert body["speech"]["status"] == "ok"


def test_speech_turns_itself_off_when_libraries_are_missing(client, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)  # as in the LLM-only image
    speech._speech_libs_present.cache_clear()
    try:
        assert speech.speech_enabled() is False
        assert client.post("/transcribe", json={"audio_base64": "UklGRg=="}).status_code == 404
        assert "not installed" in client.get("/health").get_json()["speech"]
    finally:
        speech._speech_libs_present.cache_clear()


def test_llm_image_with_speech_switched_on_still_boots(tmp_path):
    """Reproduces the CI crash: SPEECH_ENABLED=1 in an image without numpy."""
    import os
    import subprocess
    import sys

    here = os.path.dirname(os.path.dirname(__file__))
    code = (
        "import sys; sys.modules['numpy'] = None; sys.modules['faster_whisper'] = None; "
        "sys.path.insert(0, %r); import app; "
        "print(app.app.test_client().post('/format', json={}).status_code)"
    ) % here
    env = dict(os.environ, SPEECH_ENABLED="1", TARA_EAGER="1", FLOW_TOKEN="")
    env.pop("TARA_STUB", None)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "400"  # app imported and serves the LLM ("Text is required")
    assert "speech is off" in result.stderr
