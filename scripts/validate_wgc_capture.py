from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView

from manga_rt_translator.capture.windows_wgc import WindowsGraphicsCapture


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle("WGC capture validation")
    window.resize(760, 520)
    view = QWebEngineView(window)
    window.setCentralWidget(view)
    view.setHtml(
        "<html><body style='margin:0;background:#e63246;width:100vw;height:100vh'>"
        "<div style='height:100vh'></div></body></html>"
    )
    window.show()

    capture: WindowsGraphicsCapture | None = None
    stage = "red"
    result = 1
    deadline = QTimer()
    deadline.setSingleShot(True)

    def finish(code: int, message: str) -> None:
        nonlocal result
        result = code
        print(message, flush=True)
        if capture is not None:
            capture.stop()
        window.close()
        app.quit()

    def timeout() -> None:
        state = capture.state if capture is not None else "not-started"
        error = capture.last_error if capture is not None else ""
        phase = capture.startup_phase if capture is not None else ""
        finish(1, f"WGC_TIMEOUT state={state} phase={phase} error={error}")

    def begin_capture(_ok: bool) -> None:
        nonlocal capture
        capture = WindowsGraphicsCapture(int(window.winId()))
        capture.start()
        deadline.timeout.connect(timeout)
        deadline.start(8000)
        poll.start(40)

    def check_frame() -> None:
        nonlocal stage
        if capture is None:
            return
        state = capture.state
        if state == "failed":
            finish(1, f"WGC_FAILED {capture.last_error}")
            return
        status, frame = capture.read_latest()
        if status != "ok" or frame is None:
            return
        image = frame.image
        pixel = image[image.shape[0] // 2, image.shape[1] // 2]
        red, green, blue = (int(value) for value in pixel)
        print(
            f"WGC_FRAME stage={stage} size={image.shape[1]}x{image.shape[0]} "
            f"rgb={red},{green},{blue}",
            flush=True,
        )
        if stage == "red" and red > 150 and green < 100 and blue < 130:
            stage = "green"
            view.page().runJavaScript(
                "document.body.style.backgroundColor='#26c66a'"
            )
        elif stage == "green" and green > 130 and red < 100 and blue < 150:
            poll.stop()
            deadline.stop()
            finish(0, "WGC_OK event-driven frame and visual update captured")

    poll = QTimer()
    poll.timeout.connect(check_frame)
    view.loadFinished.connect(begin_capture)
    QTimer.singleShot(12000, lambda: app.quit() if result == 1 else None)
    app.exec()
    if result:
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
