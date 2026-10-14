from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from manga_rt_translator.models import Frame


class WorkCancelled(Exception):
    """The reader has moved on to a different viewport."""


class ViewState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.generation = 0
        self.closed = False

    def token(self) -> WorkToken:
        with self.lock:
            return WorkToken(self, self.generation)

    def advance(self) -> WorkToken:
        with self.lock:
            self.generation += 1
            return self.token()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.generation += 1


@dataclass(frozen=True)
class WorkToken:
    view: ViewState
    generation: int

    def current(self) -> bool:
        with self.view.lock:
            return not self.view.closed and self.generation == self.view.generation

    def check(self) -> None:
        if not self.current():
            raise WorkCancelled()

    def commit(self, action: Callable[[], None]) -> bool:
        # Invalidating a view and publishing its result must be atomic.
        with self.view.lock:
            if not self.current():
                return False
            action()
            return True


@dataclass(frozen=True)
class FrameRequest:
    frame: Frame
    token: WorkToken
    force: bool = False
    inspect: bool = False
    incremental: bool = False


class LatestFrameQueue:
    def __init__(self, view: ViewState) -> None:
        self.view = view
        self.queue: queue.Queue[FrameRequest] = queue.Queue(maxsize=1)
        self.dropped = 0
        self._available = threading.Event()
        self._in_flight: FrameRequest | None = None

    def invalidate(self) -> WorkToken:
        with self.view.lock:
            token = self.view.advance()
            self._discard_pending()
            return token

    def _discard_pending(self) -> None:
        self._pop_pending()

    def _pop_pending(self) -> FrameRequest | None:
        try:
            request = self.queue.get_nowait()
            self.dropped += 1
        except queue.Empty:
            request = None
        if self.queue.empty():
            self._available.clear()
        return request

    def put(self, frame: Frame, *, force: bool = False, inspect: bool = False,
            incremental: bool = False) -> None:
        with self.view.lock:
            if self.view.closed:
                return
            token = self.view.advance() if force else self.view.token()
            pending = self._pop_pending()
            coalesced_full_read = False
            if pending is not None and not force:
                # Keep the newest pixels, but do not let continuous capture
                # erase a pending full read or explicit inspection request.
                coalesced_full_read = pending.force or pending.incremental
                force = pending.force or pending.incremental
                inspect = not force and (inspect or pending.inspect)
            if coalesced_full_read:
                # Dirty regions describe the replaced frame, not this one.
                # A coalesced request must therefore analyze the full viewport.
                incremental = False
                inspect = False
            self.queue.put_nowait(
                FrameRequest(frame, token, force, inspect, incremental)
            )
            self._available.set()

    def get(self, timeout: float = 0.2) -> FrameRequest:
        deadline = time.monotonic() + timeout
        while True:
            with self.view.lock:
                try:
                    request = self.queue.get_nowait()
                except queue.Empty:
                    self._available.clear()
                else:
                    self._in_flight = request
                    if self.queue.empty():
                        self._available.clear()
                    return request
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._available.wait(remaining):
                raise queue.Empty

    def complete(self, request: FrameRequest) -> None:
        with self.view.lock:
            if self._in_flight is request:
                self._in_flight = None

    def has_work(self) -> bool:
        with self.view.lock:
            return self._in_flight is not None or not self.queue.empty()
