from __future__ import annotations

import threading

import pytest

from lanes import Gate, QueueFull


def _hold(started: threading.Event, release: threading.Event):
    started.set()
    release.wait(2)
    return "done"


def test_one_slot_per_lane_and_three_lanes_at_once():
    gate = Gate()
    started = {name: threading.Event() for name in ("voice", "chat", "rag")}
    release = threading.Event()

    jobs = []
    for name in ("voice", "chat", "rag"):
        job = gate.admit(name, lambda n=name: _hold(started[n], release))
        assert job.status == "running"
        threading.Thread(target=gate.run_inline, args=(job,), daemon=True).start()
        jobs.append(job)

    assert all(started[name].wait(1) for name in started)
    board = gate.snapshot()
    assert board["running"] == 3
    assert board["parallel_max"] == 3

    queued = gate.admit("voice", lambda: "later")
    assert queued.status == "queued"
    assert queued.position == 1
    chat = gate.admit("chat", lambda: "later-chat")
    assert chat.status == "queued"
    assert chat.position == 1

    release.set()
    for job in jobs:
        for _ in range(50):
            if job.status == "done":
                break
            threading.Event().wait(0.02)
        assert job.status == "done"


def test_finishing_a_lane_starts_the_queued_request():
    gate = Gate()
    release = threading.Event()
    started = threading.Event()
    first = gate.admit("rag", lambda: _hold(started, release))
    worker = threading.Thread(target=gate.run_inline, args=(first,), daemon=True)
    worker.start()
    assert started.wait(1)

    second = gate.admit("rag", lambda: "second")
    assert second.position == 1
    release.set()
    for _ in range(50):
        if second.status == "done" and second.result == "second":
            break
        threading.Event().wait(0.02)
    assert second.status == "done"
    assert second.result == "second"


def test_a_full_lane_is_refused():
    gate = Gate()
    gate.admit("chat", lambda: None)
    for _ in range(20):
        queued = gate.admit("chat", lambda: None)
        assert queued.status == "queued"
    with pytest.raises(QueueFull):
        gate.admit("chat", lambda: None)
