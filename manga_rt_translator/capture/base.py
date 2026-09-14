from __future__ import annotations

from typing import Iterator, Protocol

from manga_rt_translator.models import Frame


class ScreenStream(Protocol):
    """Long-lived source of video frames from the desktop or a window."""

    def start(self) -> None:
        ...

    def frames(self) -> Iterator[Frame]:
        ...

    def stop(self) -> None:
        ...
