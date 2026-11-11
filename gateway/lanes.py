"""Lanes, one running request each.

Voice, chat, and RAG answers can run at the same time. `embed` carries short
non-generation work (query embeddings, reranking) so it never waits behind a long
answer, and `ingest` carries bulk document work that yields to users. A second
request for a busy lane waits.

A waiting request is either handed back as a job id to poll (the default), or,
when the caller asks to wait (X-Wait), held on its own HTTP connection until its
turn comes and then run there, so a streamed answer still streams after queueing.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

LANES = ("voice", "chat", "rag", "embed", "ingest")
# Lanes that generate text for a person waiting on the other end.
USER_LANES = ("voice", "chat", "rag")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


MAX_WAITING = _env_int("LANE_MAX_WAITING", 20)
# Finished jobs are kept this long so a poller can still read the result.
JOB_TTL_SECONDS = _env_int("JOB_TTL_SECONDS", 600)

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
    # Set when the caller holds the connection and runs the job itself.
    turn: threading.Event | None = None
    finished_at: float = 0.0


@dataclass
class Lane:
    name: str
    running: Job | None = None
    waiting: deque[Job] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)


class Gate:
    def __init__(self, lanes: tuple[str, ...] = LANES, max_waiting: int | None = None) -> None:
        self.lanes = {name: Lane(name) for name in lanes}
        self.jobs: dict[str, Job] = {}
        self.max_waiting = max_waiting or MAX_WAITING
        self._jobs_lock = threading.Lock()

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
        return {
            "parallel_max": len(self.lanes),
            "running": running,
            "max_waiting": self.max_waiting,
            "lanes": lanes,
        }

    def busy(self, names: tuple[str, ...]) -> bool:
        for name in names:
            lane = self.lanes.get(name)
            if lane is None:
                continue
            with lane.lock:
                if lane.running is not None or lane.waiting:
                    return True
        return False

    def _prune(self) -> None:
        cutoff = time.monotonic() - JOB_TTL_SECONDS
        with self._jobs_lock:
            stale = [key for key, job in self.jobs.items() if job.finished_at and job.finished_at < cutoff]
            for key in stale:
                del self.jobs[key]

    def admit(self, lane_name: str, fn: Work, wait: bool = False) -> Job:
        if lane_name not in self.lanes:
            raise KeyError(lane_name)
        self._prune()
        lane = self.lanes[lane_name]
        job = Job(id=uuid.uuid4().hex, lane=lane_name, fn=fn)
        if wait:
            job.turn = threading.Event()
        with lane.lock:
            if lane.running is None:
                lane.running = job
                job.status = "running"
                job.position = 0
                if job.turn is not None:
                    job.turn.set()
            elif len(lane.waiting) >= self.max_waiting:
                raise QueueFull(lane_name)
            else:
                lane.waiting.append(job)
                job.position = len(lane.waiting)
                job.status = "queued"
        with self._jobs_lock:
            self.jobs[job.id] = job
        return job

    def wait_turn(self, job: Job, timeout: float) -> bool:
        """Block until a waiting job reaches the front. False (and dequeued) on timeout."""
        if job.turn is None:
            return job.status == "running"
        if job.turn.wait(timeout):
            return True
        if self.cancel(job):
            return False
        # Lost the race: it was promoted while timing out. It is ours to run now.
        job.turn.wait()
        return True

    def cancel(self, job: Job) -> bool:
        lane = self.lanes[job.lane]
        with lane.lock:
            if job not in lane.waiting:
                return False
            lane.waiting.remove(job)
            for index, waiting in enumerate(lane.waiting, start=1):
                waiting.position = index
        job.status = "error"
        job.error = "Gave up waiting in line."
        job.finished_at = time.monotonic()
        return True

    def finish(self, job: Job) -> None:
        lane = self.lanes[job.lane]
        nxt: Job | None = None
        if not job.finished_at:
            job.finished_at = time.monotonic()
        with lane.lock:
            if lane.running is job:
                lane.running = None
            if lane.running is None and lane.waiting:
                nxt = lane.waiting.popleft()
                lane.running = nxt
                nxt.status = "running"
                nxt.position = 0
                for index, waiting in enumerate(lane.waiting, start=1):
                    waiting.position = index
        if nxt is None:
            return
        if nxt.turn is not None:
            nxt.turn.set()  # its caller is holding the connection and runs it
        else:
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
        with self._jobs_lock:
            return self.jobs.get(job_id)


gate = Gate()
