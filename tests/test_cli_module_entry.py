import subprocess
import sys


def test_cli_module_invocation_displays_help():
    result = subprocess.run(
        [sys.executable, "-m", "manga_rt_translator.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0
    assert "Local-first stream-based manga translator" in result.stdout
    assert "reader" in result.stdout
