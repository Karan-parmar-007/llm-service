"""Keep the LLM busy while nobody uses it, and yield instantly when someone does.

Oracle can reclaim an Always Free instance that looks idle. This loop runs a
generation for `busy_seconds`, rests for `idle_seconds`, and repeats, but only
while no user request is running or queued and none arrived in the last
`quiet_seconds`. When a user request arrives, the running generation is cancelled
by closing its HTTP stream (Ollama stops at the next token), so users never wait
behind it.

Switched on with KEEPALIVE_ENABLED=1 (LLM VPS only). Tuned with
KEEPALIVE_BUSY_SECONDS, KEEPALIVE_IDLE_SECONDS, KEEPALIVE_QUIET_SECONDS.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import httpx

log = logging.getLogger("gateway.keepalive")

PROMPT = (
    "Write a long, detailed essay on the history of mathematics, from counting "
    "in ancient Mesopotamia to modern computing. Keep going section by section."
)

ShouldStop = Callable[[], bool]
Burn = Callable[[ShouldStop, float], int]


def ollama_burn(base: str, model: str) -> Burn:
    """Stream a long generation from Ollama until told to stop or time runs out.
    Leaving the `with` block closes the connection, which cancels the generation."""

    def burn(should_stop: ShouldStop, seconds: float) -> int:
        deadline = time.monotonic() + seconds
        payload = {
            "model": model,
            "prompt": PROMPT,
            "stream": True,
            "keep_alive": -1,
            "options": {"num_predict": 4096},
        }
        chunks = 0
        timeout = httpx.Timeout(10.0, read=120.0)
        with httpx.Client(base_url=base, timeout=timeout) as client:
            with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for _line in response.iter_lines():
                    chunks += 1
                    if should_stop() or time.monotonic() >= deadline:
                        break
        return chunks

    return burn


class KeepAlive:
    def __init__(
        self,
        is_busy: Callable[[], bool],
        burn: Burn,
        busy_seconds: float = 20,
        idle_seconds: float = 40,
        quiet_seconds: float = 30,
        poll_seconds: float = 1,
    ) -> None:
        self._is_busy = is_busy
        self._burn = burn
        self.busy_seconds = busy_seconds
        self.idle_seconds = idle_seconds
        self.quiet_seconds = quiet_seconds
        self._poll = poll_seconds
        self._user = threading.Event()
        self._stop = threading.Event()
        self._last_user = 0.0
        self._thread: threading.Thread | None = None
        self.state = "starting"
        self.runs = 0
        self.cancelled = 0
        self.errors = 0

    # ------------------------------------------------------------ control
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="keepalive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._user.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def user_activity(self) -> None:
        """Call on every user LLM request. Cancels a running burn at once."""
        self._last_user = time.monotonic()
        self._user.set()

    # ------------------------------------------------------------ loop
    def _quiet(self) -> bool:
        return not self._is_busy() and time.monotonic() - self._last_user >= self.quiet_seconds

    def _should_stop(self) -> bool:
        return self._user.is_set() or self._stop.is_set() or self._is_busy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._quiet():
                self.state = "paused"
                self._stop.wait(self._poll)
                continue
            self._user.clear()
            self.state = "running"
            self.runs += 1
            try:
                self._burn(self._should_stop, self.busy_seconds)
            except Exception as exc:  # Ollama down or restarting: try again later
                self.errors += 1
                log.warning("keep-alive generation failed: %s", exc)
            if self._user.is_set() and not self._stop.is_set():
                self.cancelled += 1
            self.state = "resting"
            # Rest, but wake early on shutdown. User traffic is handled by _quiet().
            self._stop.wait(self.idle_seconds)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "runs": self.runs,
            "cancelled_for_users": self.cancelled,
            "errors": self.errors,
            "busy_seconds": self.busy_seconds,
            "idle_seconds": self.idle_seconds,
            "quiet_seconds": self.quiet_seconds,
        }
