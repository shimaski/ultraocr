from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "UltraOCR"


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(os.path.abspath(sys.executable)).parents[2]
    return Path(os.path.abspath(__file__)).parents[1]


def data_root() -> Path:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / APP_NAME


def ensure_user_files(root: Path, data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    for name, example in (
        ("config.toml", "config.example.toml"),
        ("glossary.toml", "glossary.example.toml"),
    ):
        target = data / name
        if target.exists():
            continue
        source = root / name
        if not source.exists():
            source = root / example
        if source.exists():
            shutil.copy2(source, target)


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)


def main() -> int:
    root = project_root()
    data = data_root()
    log_path = data / "startup.log"

    try:
        data.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[launcher] root={root}\n")
            log.flush()
        python = root / ".venv311" / "Scripts" / "python.exe"
        if not python.exists():
            raise FileNotFoundError(f"Ambiente Python nao encontrado: {python}")
        ensure_user_files(root, data)

        command = [
            str(python),
            "-m",
            "manga_rt_translator.cli",
            "reader",
            "--config",
            str(data / "config.toml"),
            "--glossary",
            str(data / "glossary.toml"),
            *sys.argv[1:],
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[launcher] starting reader with {python}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=data,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            log.write(f"[launcher] child pid={process.pid}\n")
            log.flush()
            try:
                return_code = process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.write("[launcher] child still running after 3s\n")
                return 0

            log.write(f"[launcher] child exited with {return_code}\n")
        if return_code:
            show_error(f"O tradutor nao iniciou. Consulte o diagnostico em:\n{log_path}")
        return return_code
    except Exception as error:
        data.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[launcher] {type(error).__name__}: {error}\n")
        show_error(f"Nao foi possivel iniciar o tradutor:\n{error}\n\nDiagnostico: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
