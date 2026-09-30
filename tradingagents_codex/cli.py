"""Conservative CLI: no-argument help, explicit demo/live runs, no trading actions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from .contracts import ResearchError, RunConfig, RunReport
from .fake_backend import FakeBackend
from .orchestrator import Orchestrator
from .snapshot import Snapshot, collect_snapshot, demo_snapshot


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="tradingagents",
        description="Codex-only multi-role financial research. Never executes orders.",
    )
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic data + fake inference; offline and free")
    mode.add_argument("--live", action="store_true", help="Fetch real data and invoke a configured Codex account")
    result.add_argument("--ticker", default="SPY")
    result.add_argument("--date", type=date.fromisoformat, help="As-of date YYYY-MM-DD; demo defaults to 2025-01-15")
    result.add_argument("--model", help="Codex model ID; omit to use the account's default")
    result.add_argument("--codex-home", type=Path, help="Explicit dedicated Codex home, already authenticated by you")
    result.add_argument("--output-dir", type=Path, default=Path("codex-results"))
    result.add_argument("--language", default="Chinese")
    result.add_argument("--debate-rounds", type=int, default=1)
    result.add_argument("--risk-rounds", type=int, default=1)
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--attempts", type=int, default=2)
    result.add_argument("--role-timeout", type=float, default=120)
    result.add_argument("--run-timeout", type=float, default=1200)
    result.add_argument("--data-timeout", type=float, default=60)
    return result


def write_report(report: RunReport, snapshot: Snapshot, directory: Path) -> Path:
    destination = directory / report.run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "snapshot.json").write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    (destination / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    lines = [
        f"# TradingAgents Codex research: {report.ticker}", "",
        f"Mode: **{report.mode.upper()}** | As of: {report.as_of}", "",
        "Research only. No orders were placed. Not personalized investment advice.", "",
        f"Snapshot SHA-256: `{report.snapshot_hash}`", "",
    ]
    if report.mode == "demo":
        lines.extend(["**SYNTHETIC DEMO: fake inference and sample data, not market analysis.**", ""])
    for item in report.outputs:
        lines.extend([f"## {item.role}", "", item.summary, ""])
        if item.recommendation:
            lines.extend([f"Research recommendation: {item.recommendation}", ""])
        if item.action:
            lines.extend([f"Hypothetical action: {item.action}", ""])
        lines.extend([f"Evidence: {', '.join(item.evidence_ids) or 'none'}", ""])
        lines.extend(f"- {risk}" for risk in item.risks)
        lines.append("")
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


async def _execute(args, config: RunConfig, snapshot: Snapshot) -> RunReport:
    def progress(event):
        if event.event in ("role_completed", "role_retry", "role_failed"):
            print(f"{event.event}: {event.role} (attempt {event.attempt})", file=sys.stderr)

    if args.demo:
        return await Orchestrator(FakeBackend(), config, progress).run(snapshot)
    from .codex_backend import CodexBackend

    async with CodexBackend(
        model=args.model, codex_home=args.codex_home.resolve(),
        working_dir=args.output_dir.resolve() / ".sessions",
    ) as backend:
        return await Orchestrator(backend, config, progress).run(snapshot)


def main(argv: list[str] | None = None) -> int:
    command = parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        command.print_help()
        return 0
    args = command.parse_args(argv)
    if args.live and args.codex_home is None:
        command.error("--live requires --codex-home; this program does not find, copy or create credentials")
    if args.live and args.date is None:
        command.error("--live requires an explicit --date YYYY-MM-DD")
    if not 1 <= args.data_timeout <= 300:
        command.error("--data-timeout must be between 1 and 300 seconds")
    # Never auto-discover dotenv files through reused upstream vendor code.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    try:
        config = RunConfig(
            debate_rounds=args.debate_rounds, risk_rounds=args.risk_rounds,
            max_concurrency=args.concurrency, max_attempts=args.attempts,
            role_timeout_seconds=args.role_timeout, run_timeout_seconds=args.run_timeout,
            language=args.language,
        )
        snapshot = (
            demo_snapshot(args.ticker, args.date) if args.demo else
            collect_snapshot(args.ticker, args.date, timeout_seconds=args.data_timeout)
        )
        report = asyncio.run(_execute(args, config, snapshot))
        saved = write_report(report, snapshot, args.output_dir)
        print(json.dumps({
            "mode": report.mode, "report_dir": str(saved.resolve()),
            "recommendation": report.final_decision.recommendation,
            "orders_executed": False,
        }, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("Cancelled; no completed decision or order was produced.", file=sys.stderr)
        return 130
    except (ResearchError, ValidationError, ValueError, OSError, RuntimeError) as exc:
        # Avoid exposing provider/transport error payloads or credentials.
        print(f"Research stopped ({type(exc).__name__}); no order was placed. "
              "Check input, data availability, and your dedicated Codex setup.", file=sys.stderr)
        return 1
