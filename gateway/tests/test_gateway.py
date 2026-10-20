from __future__ import annotations

import json
import threading

import httpx
import pytest

import app as gateway
from lanes import Gate


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(gateway, "DEFAULT_MODEL", "qwen2.5:7b-instruct-q5_K_M")
    monkeypatch.setattr(gateway, "gate", Gate())
    return gateway.app.test_client()


def test_format_sends_the_dictation_task(client, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b""

        def json(self):
            return {"choices": [{"message": {"content": "1. Home\n2. Keyboard"}}]}

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, json):
            captured["path"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(gateway.httpx, "Client", FakeClient)
    response = client.post(
        "/format",
        json={"text": "point 1 home. point 2 keyboard.", "script": "roman", "tone": "whatsapp"},
    )
    assert response.status_code == 200
    assert response.get_json()["text"].startswith("1. Home")
    assert captured["path"] == "/v1/chat/completions"
    assert captured["json"]["model"] == "qwen2.5:7b-instruct-q5_K_M"
    assert "point 1" in captured["json"]["messages"][1]["content"]
    assert "script=roman" in captured["json"]["messages"][1]["content"]


def test_format_rejects_empty_text(client):
    response = client.post("/format", json={"text": "   "})
    assert response.status_code == 400


def test_chat_completions_are_forwarded_with_the_callers_model(client, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"id":"ok"}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"id": "ok"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(gateway.httpx, "Client", FakeClient)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "some-other-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    assert captured["url"] == "/v1/chat/completions"
    assert captured["json"]["model"] == "some-other-model"
    assert json.loads(response.data)["id"] == "ok"


def test_embeddings_path_is_forwarded(client, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"data":[]}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"data": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(gateway.httpx, "Client", FakeClient)
    response = client.post("/v1/embeddings", json={"input": "kara"})
    assert response.status_code == 200
    assert captured["url"] == "/v1/embeddings"
    assert captured["json"]["model"] == "qwen2.5:7b-instruct-q5_K_M"
    assert captured["json"]["input"] == "kara"


def test_a_second_voice_request_is_queued(client, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    class BlockingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, json):
            started.set()
            release.wait(2)

            class Done:
                status_code = 200
                headers = {"content-type": "application/json"}

                def json(self):
                    return {"choices": [{"message": {"content": "done"}}]}

            return Done()

    monkeypatch.setattr(gateway.httpx, "Client", BlockingClient)
    first: dict = {}

    def fire():
        first["response"] = client.post("/format", json={"text": "point 1 home"})

    threading.Thread(target=fire, daemon=True).start()
    assert started.wait(1)

    queued = client.post("/format", json={"text": "point 2 keyboard"})
    assert queued.status_code == 202
    body = queued.get_json()
    assert body["detail"] == "Your request is queued."
    assert body["lane"] == "voice"
    assert body["position"] == 1
    release.set()


def test_ollama_down_is_503(client, monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, *args, **kwargs):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(gateway.httpx, "Client", FakeClient)
    response = client.post("/api/chat", json={"model": "qwen2.5:7b-instruct-q5_K_M"})
    assert response.status_code == 503
