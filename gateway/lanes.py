"""Lanes, one running request each.

Voice, chat, and RAG can run at the same time. A second request for a busy
lane waits, and the caller is told it is queued.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

LANES = ("voice", "chat", "rag")
MAX_WAITING = 20

Work = Callable[[], object]


class QueueFull(Exception):
    def __init__(self, lane: str) -> None:
        self.lane = lane
        super().__init__(f"The {lane} line is full.")


@dataclass
class Job:
    id: str
    lane: str
    fn: Work
    status: str = "queued"
    position: int = 0
    result: object = None
    error: str = ""


@dataclass
class Lane:
    name: str
    running: Job | None = None
    waiting: deque[Job] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)


class Gate:
    def __init__(self, lanes: tuple[str, ...] = LANES) -> None:
        self.lanes = {name: Lane(name) for name in lanes}
        self.jobs: dict[str, Job] = {}

    def snapshot(self) -> dict:
        lanes = {}
        running = 0
        for name, lane in self.lanes.items():
            with lane.lock:
                is_running = lane.running is not None
                queued = len(lane.waiting)
            if is_running:
                running += 1
            lanes[name] = {"running": is_running, "queued": queued}
        return {"parallel_max": len(self.lanes), "running": running, "lanes": lanes}

    def admit(self, lane_name: str, fn: Work) -> Job:
        if lane_name not in self.lanes:
            raise KeyError(lane_name)
        lane = self.lanes[lane_name]
        job = Job(id=uuid.uuid4().hex, lane=lane_name, fn=fn)
        with lane.lock:
            self.jobs[job.id] = job
            if lane.running is None:
                lane.running = job
                job.status = "running"
                job.position = 0
                return job
            if len(lane.waiting) >= MAX_WAITING:
                del self.jobs[job.id]
                raise QueueFull(lane_name)
            lane.waiting.append(job)
            job.position = len(lane.waiting)
            job.status = "queued"
            return job

    def finish(self, job: Job) -> None:
        lane = self.lanes[job.lane]
        nxt: Job | None = None
        with lane.lock:
            if lane.running is job:
                lane.running = None
            if lane.waiting:
                nxt = lane.waiting.popleft()
                lane.running = nxt
                nxt.status = "running"
                nxt.position = 0
                for index, waiting in enumerate(lane.waiting, start=1):
                    waiting.position = index
        if nxt is not None:
            threading.Thread(target=self._run, args=(nxt,), daemon=True).start()

    def _run(self, job: Job) -> None:
        try:
            job.result = job.fn()
            job.status = "done"
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
        finally:
            self.finish(job)

    def run_inline(self, job: Job) -> object:
        try:
            job.result = job.fn()
            job.status = "done"
            return job.result
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            raise
        finally:
            self.finish(job)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


gate = Gate()
