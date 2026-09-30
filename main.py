"""Default entry point for the Codex-only fork. Legacy: python -m cli.main."""

from tradingagents_codex.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
