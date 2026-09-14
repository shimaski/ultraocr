from __future__ import annotations

import queue
import threading
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


class LatestFrameQueue:
    def __init__(self, view: ViewState) -> None:
        self.view = view
        self.queue: queue.Queue[FrameRequest] = queue.Queue(maxsize=1)
        self.dropped = 0

    def invalidate(self) -> WorkToken:
        with self.view.lock:
            token = self.view.advance()
            self._discard_pending()
            return token

    def _discard_pending(self) -> None:
        try:
            self.queue.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass

    def put(self, frame: Frame, *, force: bool = False, inspect: bool = False) -> None:
        with self.view.lock:
            if self.view.closed:
                return
            token = self.view.advance() if force else self.view.token()
            self._discard_pending()
            self.queue.put_nowait(FrameRequest(frame, token, force, inspect))

    def get(self, timeout: float = 0.2) -> FrameRequest:
        return self.queue.get(timeout=timeout)
