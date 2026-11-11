from __future__ import annotations

import threading
import time

import pytest

import app as gateway
import lanes
from lanes import Gate


def test_waiting_caller_runs_its_own_job_when_its_turn_comes():
    gate = Gate()
    release = threading.Event()
    first = gate.admit("rag", lambda: release.wait(2) or "first")
    threading.Thread(target=gate.run_inline, args=(first,), daemon=True).start()

    second = gate.admit("rag", lambda: "second", wait=True)
    assert second.status == "queued" and second.position == 1
    threading.Timer(0.05, release.set).start()
    assert gate.wait_turn(second, 2)
    assert second.status == "running"
    assert gate.run_inline(second) == "second"
    assert gate.snapshot()["lanes"]["rag"] == {"running": False, "queued": 0}


def test_a_waiting_caller_that_times_out_leaves_the_line():
    gate = Gate()
    gate.admit("chat", lambda: None)  # never finishes
    waiter = gate.admit("chat", lambda: None, wait=True)
    behind = gate.admit("chat", lambda: None)
    assert behind.position == 2
    assert gate.wait_turn(waiter, 0.05) is False
    assert waiter.status == "error"
    assert behind.position == 1


def test_finished_jobs_are_pruned(monkeypatch):
    gate = Gate()
    job = gate.admit("embed", lambda: "x")
    gate.run_inline(job)
    assert gate.get(job.id) is not None
    monkeypatch.setattr(lanes, "JOB_TTL_SECONDS", 0)
    job.finished_at = time.monotonic() - 1
    gate.admit("embed", lambda: "y")
    assert gate.get(job.id) is None


def test_busy_reports_user_lanes_only():
    gate = Gate()
    gate.admit("ingest", lambda: None)
    assert not gate.busy(lanes.USER_LANES)
    gate.admit("rag", lambda: None)
    assert gate.busy(lanes.USER_LANES)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(gateway, "gate", Gate())
    return gateway.app.test_client()


def test_embeddings_use_the_embed_lane(client):
    with gateway.app.test_request_context("/api/embed", method="POST"):
        assert gateway.lane_for("/api/embed") == "embed"
        assert gateway.lane_for("/v1/embeddings") == "embed"
        assert gateway.lane_for("/api/chat") == "chat"
    with gateway.app.test_request_context("/api/chat", method="POST", headers={"X-Lane": "ingest"}):
        assert gateway.lane_for("/api/chat") == "ingest"


def test_rerank_is_404_without_model_files(client, monkeypatch):
    monkeypatch.setattr(gateway.rerank, "enabled", lambda: False)
    response = client.post("/rerank", json={"query": "q", "documents": ["a"]})
    assert response.status_code == 404


def test_rerank_orders_documents(client, monkeypatch):
    monkeypatch.setattr(gateway.rerank, "enabled", lambda: True)

    class Fake:
        def scores(self, query, documents):
            return [0.1, 0.9, 0.5][: len(documents)]

    monkeypatch.setattr(gateway.rerank, "get", lambda: Fake())
    response = client.post("/rerank", json={"query": "q", "documents": ["a", "b", "c"], "top_n": 2})
    assert response.status_code == 200
    results = response.get_json()["results"]
    assert [item["index"] for item in results] == [1, 2]


def test_rerank_validates_input(client, monkeypatch):
    monkeypatch.setattr(gateway.rerank, "enabled", lambda: True)
    assert client.post("/rerank", json={"query": "", "documents": ["a"]}).status_code == 400
    assert client.post("/rerank", json={"query": "q", "documents": ["a"] * 65}).status_code == 400


def test_proxy_with_wait_header_returns_429_when_the_line_never_frees(client, monkeypatch):
    gateway.gate.admit("chat", lambda: None)  # occupy the lane forever
    response = client.post(
        "/api/chat", json={"model": "m", "messages": []}, headers={"X-Wait": "0.05"}
    )
    assert response.status_code == 429
    assert response.get_json()["code"] == "busy"
