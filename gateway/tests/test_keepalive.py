"""The keep-alive must burn CPU only while nobody is using the LLM, and yield at once."""

import threading
import time

import app as gateway
from keepalive import KeepAlive


class FakeBurn:
    """Stands in for an Ollama stream: 'generates' until told to stop or time runs out."""

    def __init__(self):
        self.started = threading.Event()
        self.stopped_early = threading.Event()
        self.calls = 0

    def __call__(self, should_stop, seconds):
        self.calls += 1
        self.started.set()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if should_stop():
                self.stopped_early.set()
                return 0
            time.sleep(0.01)
        return 1


def _keeper(burn, busy=lambda: False, busy_s=5, idle_s=0.05, quiet_s=0.05):
    return KeepAlive(busy, burn, busy_seconds=busy_s, idle_seconds=idle_s, quiet_seconds=quiet_s, poll_seconds=0.02)


def test_runs_when_nobody_is_using_the_llm():
    burn = FakeBurn()
    keeper = _keeper(burn, busy_s=0.05)
    keeper.start()
    try:
        assert burn.started.wait(1)
        time.sleep(0.3)
        assert burn.calls >= 2  # keeps cycling busy -> rest -> busy
    finally:
        keeper.stop()


def test_a_user_request_cancels_the_running_generation_at_once():
    burn = FakeBurn()
    keeper = _keeper(burn, busy_s=30, quiet_s=5)
    keeper.start()
    try:
        assert burn.started.wait(1)
        t0 = time.monotonic()
        keeper.user_activity()
        assert burn.stopped_early.wait(1)
        assert time.monotonic() - t0 < 0.2
        time.sleep(0.2)
        assert keeper.cancelled == 1
        assert keeper.state == "paused"  # waits out the quiet period before burning again
        assert burn.calls == 1
    finally:
        keeper.stop()


def test_waits_while_user_work_is_running_or_queued():
    burn = FakeBurn()
    busy = threading.Event()
    busy.set()
    keeper = _keeper(burn, busy=busy.is_set, busy_s=0.05)
    keeper.start()
    try:
        time.sleep(0.2)
        assert burn.calls == 0 and keeper.state == "paused"
        busy.clear()
        assert burn.started.wait(1)
    finally:
        keeper.stop()


def test_ollama_errors_do_not_kill_the_loop():
    calls = {"n": 0}

    def flaky(should_stop, seconds):
        calls["n"] += 1
        raise RuntimeError("ollama restarting")

    keeper = _keeper(flaky)
    keeper.start()
    try:
        time.sleep(0.3)
        assert calls["n"] >= 2 and keeper.errors >= 2
    finally:
        keeper.stop()


def test_user_llm_requests_notify_the_keeper(monkeypatch):
    seen = []

    class Spy:
        def user_activity(self):
            seen.append(1)

        def snapshot(self):
            return {"state": "spy"}

    monkeypatch.setattr(gateway, "keeper", Spy())
    client = gateway.app.test_client()
    client.post("/format", json={})          # LLM write -> notify
    client.get("/health")                    # read -> no notify
    assert seen == [1]
    assert client.get("/health").get_json()["keepalive"] == {"state": "spy"}


def test_keepalive_is_off_unless_enabled():
    assert gateway.keeper is None
    assert gateway.app.test_client().get("/health").get_json()["keepalive"] == "off"
