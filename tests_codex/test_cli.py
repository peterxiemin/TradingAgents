import json
import os
import subprocess
import sys

from tradingagents_codex.cli import main


def test_help_needs_no_auth(capsys):
    assert main([]) == 0
    assert "Codex-only" in capsys.readouterr().out


def test_demo_saves_marked_report_and_snapshot(tmp_path, capsys):
    assert main(["--demo", "--output-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "demo"
    assert result["orders_executed"] is False
    folders = list(tmp_path.iterdir())
    assert len(folders) == 1
    text = (folders[0] / "report.md").read_text()
    assert "SYNTHETIC DEMO" in text
    report = json.loads((folders[0] / "report.json").read_text())
    assert len(report["outputs"]) == 12
    assert (folders[0] / "snapshot.json").exists()


def test_fresh_process_never_imports_frameworks():
    script = '''
import importlib.abc, sys
class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("langchain", "langgraph", "openai_codex")):
            raise AssertionError("unexpected runtime import: " + fullname)
sys.meta_path.insert(0, RejectFrameworks())
from tradingagents_codex.cli import main
import tempfile
assert main(["--demo", "--output-dir", tempfile.mkdtemp()]) == 0
'''
    outcome = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                             env={"PATH": os.environ.get("PATH", ""), "PYTHON_DOTENV_DISABLED": "1"},
                             timeout=30)
    assert outcome.returncode == 0, outcome.stderr


def test_live_requires_explicit_auth_home():
    outcome = subprocess.run([sys.executable, "-m", "tradingagents_codex", "--live", "--date", "2026-01-09"],
                             capture_output=True, text=True, timeout=15)
    assert outcome.returncode == 2
    assert "requires --codex-home" in outcome.stderr
