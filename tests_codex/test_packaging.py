from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docker_context_excludes_runtime_credentials_and_environments():
    patterns = set((ROOT / ".dockerignore").read_text().splitlines())
    for pattern in (".env*", ".codex*", "**/auth.json", ".venv*", "venv*", "codex-results"):
        assert pattern in patterns


def test_default_entrypoint_is_framework_free():
    metadata = (ROOT / "pyproject.toml").read_text()
    assert 'tradingagents = "tradingagents_codex.cli:main"' in metadata
    default_dependencies = metadata.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "langchain" not in default_dependencies
    assert "langgraph" not in default_dependencies
    assert '"openai-codex==0.159.2"' in default_dependencies


def test_container_report_mountpoint_is_owned_by_runtime_user():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "install -d -m 0755 -o appuser -g appuser" in dockerfile
    assert "/home/appuser/app/codex-results" in dockerfile
