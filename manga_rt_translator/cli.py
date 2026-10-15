from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

from manga_rt_translator.app_runtime import TranslationSession
from manga_rt_translator.argos_models import install_model_route, list_installed_models
from manga_rt_translator.capture.windows_windows import find_window, list_visible_windows
from manga_rt_translator.config import AppConfig, load_config
from manga_rt_translator.glossary import Glossary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-windows":
        return list_windows_command()
    if args.command == "ui":
        return ui_command(args)
    if args.command == "reader":
        return reader_command(args)
    if args.command == "run":
        return run_command(args)
    if args.command == "argos-models":
        if args.action == "status":
            return list_installed_models()
        return install_model_route(
            args.source,
            args.target,
            confirm=_confirm_argos_install,
        )
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manga-rt-translator",
        description="Local-first stream-based manga translator.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run the real-time translator.")
    run.add_argument("--config", default=None, help="Path to config TOML.")
    run.add_argument("--glossary", default=None, help="Path to glossary TOML.")
    run.add_argument("--db", default=None, help="SQLite Translation Memory path.")
    run.add_argument(
        "--provider",
        default=None,
        choices=["google", "deepl", "gemini", "libre", "argos", "echo"],
        help="Translation provider (default comes from config).",
    )
    run.add_argument(
        "--deepl-api-key",
        default=None,
        help="DeepL API key (required when --provider deepl).",
    )
    run.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (required when --provider gemini).",
    )
    run.add_argument(
        "--libre-api-key",
        default=None,
        help="LibreTranslate API key (optional when --provider libre).",
    )
    run.add_argument("--source-lang", default=None, help="OCR source language.")
    run.add_argument(
        "--extra-langs",
        default=None,
        help="Comma-separated extra OCR languages, e.g. japan,korean.",
    )
    run.add_argument("--target-lang", default=None, help="Translation target language.")
    run.add_argument("--fps", type=int, default=None, help="Capture FPS.")
    run.add_argument("--window-title", default=None, help="Capture a visible window by title match.")
    run.add_argument("--window-handle", type=int, default=None, help="Capture a visible window by native handle.")
    run.add_argument("--stats-file", default=None, help=argparse.SUPPRESS)
    run.add_argument("--no-overlay", action="store_true", help="Print overlay items.")
    run.add_argument(
        "--echo-translator",
        action="store_true",
        help="Do not call translation services; echo OCR text. Useful for OCR/overlay testing.",
    )

    subparsers.add_parser("list-windows", help="List visible windows on Windows.")

    argos = subparsers.add_parser(
        "argos-models", help="List or install local Argos translation models."
    )
    argos_actions = argos.add_subparsers(dest="action", required=True)
    argos_actions.add_parser("status", help="List installed translation pairs.")
    install = argos_actions.add_parser("install", help="Install a model route after confirmation.")
    install.add_argument("--source", required=True, help="Source language, e.g. en or japan.")
    install.add_argument("--target", default="pt-BR", help="Target language (default: pt-BR).")

    ui = subparsers.add_parser("ui", help="Open a small control panel.")
    ui.add_argument("--config", default=None, help="Path to config TOML.")
    ui.add_argument("--glossary", default=None, help="Path to glossary TOML.")

    reader = subparsers.add_parser("reader", help="Open the desktop manga reader.")
    reader.add_argument("--config", default=None, help="Path to config TOML.")
    reader.add_argument("--glossary", default=None, help="Path to glossary TOML.")
    reader.add_argument("--url", default=None, help="Initial URL to open.")
    return parser


def run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    if config.capture.window_title or config.capture.window_handle is not None:
        window = find_window(
            title_contains=config.capture.window_title,
            handle=config.capture.window_handle,
        )
        if window is None:
            selector = (
                f"handle={config.capture.window_handle}"
                if config.capture.window_handle is not None
                else f"title contains {config.capture.window_title!r}"
            )
            raise RuntimeError(
                f"Unable to resolve capture window for {selector}. "
                "Use `manga-rt-translator list-windows` to inspect visible candidates."
            )
        print(
            "[capture] "
            f"window=0x{window.handle:08X} "
            f"title={window.title!r} "
            f"region={window.rect.crop_tuple()}",
            flush=True,
        )

    glossary = Glossary.load(args.glossary)
    session = TranslationSession(
        config=config,
        glossary=glossary,
        echo_translator=args.echo_translator,
        overlay_enabled=not args.no_overlay,
    )
    session.start()
    stats_stop = threading.Event()
    stats_worker = _start_stats_writer(session, args.stats_file, stats_stop)

    try:
        if args.no_overlay:
            while session.worker and session.worker.is_alive():
                time.sleep(2)
                print(
                    "[stats] "
                    f"frames={session.stats.frames_seen} "
                    f"roi_frames={session.stats.frames_with_rois} "
                    f"ocr={session.stats.ocr_regions} "
                    f"cache={session.stats.cache_hits} "
                    f"api={session.stats.translate_calls}",
                    flush=True,
                )
            return 0
        return session.overlay.run_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        stats_stop.set()
        session.stop()
        if stats_worker is not None:
            stats_worker.join(timeout=1)


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.db:
        config.cache.path = args.db
    if args.provider:
        config.translation.provider = args.provider
    if args.deepl_api_key:
        config.translation.deepl_api_key = args.deepl_api_key
    if getattr(args, "gemini_api_key", None):
        config.translation.gemini_api_key = args.gemini_api_key
    if getattr(args, "libre_api_key", None):
        config.translation.libre_api_key = args.libre_api_key
    if args.source_lang:
        config.ocr.source_lang = args.source_lang
    if getattr(args, "extra_langs", None) is not None:
        config.ocr.extra_langs = [
            item.strip() for item in args.extra_langs.split(",") if item.strip()
        ]
    if args.target_lang:
        config.translation.target_lang = args.target_lang
    if args.fps:
        config.capture.fps = args.fps
    if getattr(args, "window_title", None):
        config.capture.window_title = args.window_title
        config.capture.window_handle = None
        config.capture.region = None
    if getattr(args, "window_handle", None) is not None:
        config.capture.window_handle = args.window_handle
        config.capture.window_title = None
        config.capture.region = None


def list_windows_command() -> int:
    if platform.system().lower() != "windows":
        raise RuntimeError("Window listing is currently implemented for Windows only.")

    windows = list_visible_windows()
    for window in windows:
        line = (
            f"0x{window.handle:08X} | "
            f"{window.rect.crop_tuple()} | "
            f"{window.class_name} | {window.title}"
        )
        print(_safe_console_text(line))
    return 0


def ui_command(args: argparse.Namespace) -> int:
    from manga_rt_translator.ui import run_control_panel

    return run_control_panel(args.config, args.glossary)


def reader_command(args: argparse.Namespace) -> int:
    from manga_rt_translator.reader import run_reader_app

    return run_reader_app(
        config_path=args.config,
        glossary_path=args.glossary,
        initial_url=args.url,
    )


def _safe_console_text(text: str) -> str:
    encoding = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _confirm_argos_install(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"s", "sim", "y", "yes"}
    except EOFError:
        return False


def _start_stats_writer(
    session: TranslationSession,
    stats_file: str | None,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if not stats_file:
        return None

    path = Path(stats_file)

    def _writer() -> None:
        while not stop_event.is_set():
            payload = {
                "frames_seen": session.stats.frames_seen,
                "frames_with_rois": session.stats.frames_with_rois,
                "ocr_regions": session.stats.ocr_regions,
                "cache_hits": session.stats.cache_hits,
                "translate_calls": session.stats.translate_calls,
                "last_error": session.last_error or session.stats.last_error,
                "running": bool(session.worker and session.worker.is_alive()),
            }
            try:
                path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass
            time.sleep(1)

    worker = threading.Thread(target=_writer, daemon=True)
    worker.start()
    return worker


if __name__ == "__main__":
    raise SystemExit(main())
