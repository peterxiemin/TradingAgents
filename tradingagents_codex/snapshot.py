"""Frozen, bounded evidence shared by every Codex role.

The host imports only the standard library and Pydantic. Live collection runs a
fixed allowlist of keyless Yahoo/SEC readers in a disposable subprocess: no
legacy router, LLM, dotenv discovery, inherited credentials, or shared cache.
The subprocess can be killed at the overall deadline (threads cannot provide
that guarantee for synchronous vendor clients).

``as_of`` is an inclusive UTC calendar-day cutoff, not an intraday backtest.
Yahoo history is retrieved now and may contain vendor revisions, including
split adjustments. This is not a certified point-in-time historical database.
No later bars, articles, or SEC filing vintages are used. Live profiles are
withheld for historical dates and undated news is always omitted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Category = Literal["market", "fundamentals", "news", "sentiment"]
Status = Literal["available", "unavailable", "withheld"]
MAX_CONTENT_CHARS = 8_000
MAX_EVIDENCE_ITEMS = 16
MAX_SNAPSHOT_CHARS = 56_000
MAX_WORKER_BYTES = 400_000  # JSON escaping can expand bounded Unicode content.
DEMO_DATE = date(2025, 1, 15)
_TICKER_RE = re.compile(r"\^?[A-Z0-9][A-Z0-9.=_+\-]{0,31}\Z")

# The ordering, identities, sources and tools are host-owned, never model input.
_SPECS: tuple[tuple[str, Category, str], ...] = (
    ("market_snapshot", "market", "Yahoo Finance / stockstats"),
    ("company_profile", "fundamentals", "Yahoo Finance"),
    ("balance_sheet", "fundamentals", "SEC EDGAR"),
    ("income_statement", "fundamentals", "SEC EDGAR"),
    ("cashflow", "fundamentals", "SEC EDGAR"),
    ("company_news", "news", "Yahoo Finance"),
    ("social_sentiment", "sentiment", "Not collected"),
)


def _ticker(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("ticker must be a string")
    value = value.strip().upper()
    if len(value) > 32 or not _TICKER_RE.fullmatch(value) or ".." in value:
        raise ValueError("ticker must be a valid symbol of at most 32 characters")
    return value


def _as_of(value: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError("as_of must be a date, not a datetime")
    if value > datetime.now(UTC).date():
        raise ValueError("as_of cannot be in the future")
    if value < date(1900, 1, 1):
        raise ValueError("as_of must be on or after 1900-01-01")
    return value


class Evidence(BaseModel):
    """One immutable piece of external data, never an instruction to an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-]+$")
    category: Category
    source: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    status: Status


class Snapshot(BaseModel):
    """The single immutable input to all roles in one orchestration run."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    ticker: str
    as_of: date
    captured_at: datetime
    evidence: tuple[Evidence, ...] = Field(min_length=1, max_length=MAX_EVIDENCE_ITEMS)
    mode: Literal["demo", "live"]

    @field_validator("ticker", mode="before")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return _ticker(value)

    @field_validator("as_of")
    @classmethod
    def valid_date(cls, value: date) -> date:
        return _as_of(value)

    @field_validator("captured_at")
    @classmethod
    def aware_capture(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def bounded_unique_evidence(self) -> Snapshot:
        if len({item.id for item in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        if sum(len(item.content) for item in self.evidence) > MAX_SNAPSHOT_CHARS:
            raise ValueError("snapshot evidence exceeds total content limit")
        return self

    def fingerprint(self) -> str:
        """SHA256 of canonical JSON, including source status and capture time."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bounded(content: str) -> str:
    suffix = "\n[TRUNCATED: evidence exceeded the host content limit]"
    return (
        content
        if len(content) <= MAX_CONTENT_CHARS
        else content[: MAX_CONTENT_CHARS - len(suffix)] + suffix
    )


def _item(spec: tuple[str, Category, str], content: str, status: Status) -> Evidence:
    return Evidence(
        id=spec[0], category=spec[1], source=spec[2], content=_bounded(content), status=status
    )


def demo_snapshot(ticker: str = "SPY", as_of: date | None = None) -> Snapshot:
    """A repeatable synthetic scenario; no value claims to be a real quote.

    The default date and capture timestamp are intentionally fixed. ``ticker``
    labels the exercise only; the generated prices are synthetic units.
    """
    ticker, as_of = _ticker(ticker), _as_of(as_of if as_of is not None else DEMO_DATE)
    days: list[date] = []
    cursor = as_of
    while len(days) < 60:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    offset = int(hashlib.sha256(ticker.encode()).hexdigest()[:4], 16) % 20
    closes = [round(100 + offset + i * 0.12 + math.sin(i / 3) * 1.4, 2) for i in range(60)]
    rows = [
        {
            "date": day.isoformat(),
            "open": round(close - 0.2, 2),
            "high": round(close + 0.65, 2),
            "low": round(close - 0.8, 2),
            "close": close,
            "volume": 1_000_000 + (i % 7) * 25_000,
        }
        for i, (day, close) in enumerate(zip(days, closes, strict=True))
    ]
    content = json.dumps(
        {
            "warning": "SYNTHETIC DEMO ONLY: invented scenario units, not historical prices or actual quotes for this ticker",
            "capture_note": "captured_at is a deterministic scenario timestamp, not a real market-data retrieval time",
            "latest_ohlcv": rows[-1],
            "recent_ohlcv": rows[-15:],
            "indicators": {
                "close_10_sma": round(sum(closes[-10:]) / 10, 4),
                "close_50_sma": round(sum(closes[-50:]) / 50, 4),
            },
        },
        sort_keys=True,
    )
    evidence = [_item((_SPECS[0][0], "market", "Synthetic demo fixture"), content, "available")]
    evidence.extend(
        _item(
            spec,
            "Unavailable in the synthetic demo; no real company data, news, or sentiment was fetched.",
            "unavailable",
        )
        for spec in _SPECS[1:]
    )
    return Snapshot(
        ticker=ticker,
        as_of=as_of,
        captured_at=datetime.combine(as_of, datetime.min.time(), UTC),
        evidence=tuple(evidence),
        mode="demo",
    )


def _worker_env(run_dir: str) -> dict[str, str]:
    """A small allowlist, not a copy of host API keys, proxies, or credentials."""
    env = {
        "PATH": os.defpath,
        "HOME": run_dir,
        "USERPROFILE": run_dir,
        "XDG_CACHE_HOME": str(Path(run_dir) / "cache"),
        "TMPDIR": run_dir,
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
        "PYTHONIOENCODING": "utf-8",
        "TRADINGAGENTS_CODEX_DATA_WORKER": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }
    for key in (
        "SYSTEMROOT",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "SEC_EDGAR_USER_AGENT",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _decode_worker(output: str | bytes | None) -> dict[str, Evidence]:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not output or len(output.encode("utf-8")) > MAX_WORKER_BYTES:
        return {}
    specs = {spec[0]: spec for spec in _SPECS}
    found: dict[str, Evidence] = {}
    for line in output.splitlines()[:MAX_EVIDENCE_ITEMS]:
        try:
            item = Evidence.model_validate_json(line)
        except ValueError:
            continue
        spec = specs.get(item.id)
        if spec and (item.id, item.category, item.source) == spec and item.id not in found:
            found[item.id] = item
    return found


def collect_snapshot(ticker: str, as_of: date, *, timeout_seconds: float = 60) -> Snapshot:
    """Collect the allowlisted sources once; failures become explicit evidence.

    This synchronous call has a hard subprocess deadline. Complete results
    emitted before a timeout are retained. Temp files and vendor cookies are
    isolated per call and deleted after the worker exits. No paid tools run.
    """
    ticker, as_of = _ticker(ticker), _as_of(as_of)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 600
    ):
        raise ValueError("timeout_seconds must be finite and in (0, 600]")
    captured_at = datetime.now(UTC)
    missing_reason = "Collection worker did not return this source; no financial fact is inferred."
    with tempfile.TemporaryDirectory(prefix="tradingagents-codex-data-") as run_dir:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from tradingagents_codex.snapshot import _worker_main; _worker_main()",
                ],
                input=json.dumps(
                    {"ticker": ticker, "as_of": as_of.isoformat(), "run_dir": run_dir}
                ),
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
                cwd=run_dir,
                env=_worker_env(run_dir),
            )
            found = _decode_worker(result.stdout)
        except subprocess.TimeoutExpired as exc:
            found = _decode_worker(exc.output)
            missing_reason = "Collection deadline exceeded; this source was not retrieved. No absence of market activity is implied."
        except OSError:
            found = {}
            missing_reason = "The data worker could not start; this source is unavailable."
    evidence = []
    for spec in _SPECS:
        if spec[0] == "social_sentiment":
            evidence.append(
                _item(
                    spec,
                    "Social sentiment is not collected: no bounded, point-in-time keyless source is enabled.",
                    "withheld",
                )
            )
        elif spec[0] == "company_profile" and as_of < captured_at.date():
            evidence.append(
                _item(
                    spec,
                    "Historical profile withheld: Yahoo serves present-day values without a historical vintage.",
                    "withheld",
                )
            )
        else:
            evidence.append(found.get(spec[0]) or _item(spec, missing_reason, "unavailable"))
    return Snapshot(
        ticker=ticker, as_of=as_of, captured_at=captured_at, evidence=tuple(evidence), mode="live"
    )


def _market_content(ticker: str, as_of: date) -> str:
    """Small as-of adapter; upstream retry, normalization and stockstats retained."""
    import pandas as pd
    import yfinance as yf
    from stockstats import wrap

    from tradingagents.dataflows.symbols import normalize_symbol
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry
    from tradingagents.dataflows.vendors.yahoo.ohlcv import (
        _assert_ohlcv_not_stale,
        _clean_dataframe,
    )

    canonical = normalize_symbol(ticker)
    frame = yf_retry(
        lambda: yf.Ticker(canonical).history(
            start=(as_of - timedelta(days=550)).isoformat(),
            end=(as_of + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            back_adjust=False,
            actions=False,
            timeout=12,
        ),
        max_retries=1,
        base_delay=0.5,
    )
    if frame is None or frame.empty:
        raise ValueError("No price rows")
    frame = _clean_dataframe(frame.reset_index())
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if any(field not in frame for field in fields):
        raise ValueError("Incomplete OHLCV columns")
    # Cut BEFORE calculations. No backwards fill, invented rows, or future bars.
    frame = (
        frame[frame["Date"].dt.date <= as_of]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    frame = frame.dropna(subset=["Close"]).tail(400).copy()
    if frame.empty:
        raise ValueError("No dated price rows at or before cutoff")
    _assert_ohlcv_not_stale(frame, as_of.isoformat(), ticker, canonical)
    stock = wrap(frame.copy())
    indicators: dict[str, float | None] = {}
    for name, minimum in (
        ("close_10_ema", 10),
        ("close_50_sma", 50),
        ("close_200_sma", 200),
        ("rsi", 15),
        ("macd", 35),
        ("macds", 35),
        ("boll", 20),
        ("boll_ub", 20),
        ("boll_lb", 20),
        ("atr", 15),
    ):
        try:
            value = float(stock[name].iloc[-1]) if len(frame) >= minimum else math.nan
            indicators[name] = round(value, 6) if math.isfinite(value) else None
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            indicators[name] = None
    rows = []
    for _, row in frame.tail(20).iterrows():
        record = {"date": row["Date"].date().isoformat()}
        for field in fields:
            value = float(row[field])
            record[field.lower()] = (
                round(value, 6) if pd.notna(value) and math.isfinite(value) else None
            )
        rows.append(record)
    return json.dumps(
        {
            "symbol": canonical,
            "as_of": as_of.isoformat(),
            "limitation": "Retrieved now, auto_adjust=False. Yahoo may revise historical data or split-adjust prices; not certified point-in-time history. Same-day bar may be incomplete.",
            "latest_ohlcv": rows[-1],
            "indicators": indicators,
            "recent_ohlcv": rows,
            "indicator_note": "Null means unavailable or insufficient observations; price gaps are not filled.",
        },
        allow_nan=False,
        sort_keys=True,
    )


def _news_content(ticker: str, as_of: date) -> str:
    import yfinance as yf

    from tradingagents.dataflows.symbols import normalize_symbol
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry
    from tradingagents.dataflows.vendors.yahoo.news import _extract_article_data

    articles = (
        yf_retry(
            lambda: yf.Ticker(normalize_symbol(ticker)).get_news(count=12),
            max_retries=1,
            base_delay=0.5,
        )
        or []
    )
    start = datetime.combine(as_of - timedelta(days=7), datetime.min.time(), UTC)
    end = min(
        datetime.combine(as_of + timedelta(days=1), datetime.min.time(), UTC), datetime.now(UTC)
    )
    selected, seen = [], set()
    for article in articles[:24]:
        data = _extract_article_data(article)
        published = data["pub_date"]
        if published is None:
            continue
        published = (
            published.replace(tzinfo=UTC) if published.tzinfo is None else published.astimezone(UTC)
        )
        title = str(data["title"])[:300]
        if not start <= published < end or not title or title in seen:
            continue
        seen.add(title)
        selected.append(
            {
                "published_at": published.isoformat(),
                "title": title,
                "publisher": str(data["publisher"])[:120],
                "summary": str(data["summary"])[:500],
                "url": str(data["link"])[:500],
            }
        )
        if len(selected) == 6:
            break
    if not selected:
        raise ValueError("No dated news retrieved within requested window")
    return json.dumps(
        {
            "coverage": "Recent Yahoo feed only; incomplete historical coverage, not evidence that other news did not exist.",
            "articles": selected,
        },
        sort_keys=True,
    )


def _legacy_tools(
    ticker: str, as_of: date
) -> tuple[tuple[tuple[str, Category, str], Callable[[], str]], ...]:
    # Only the disposable worker may touch legacy imports. In addition to the
    # environment flag, replace dotenv entrypoints here for older installations
    # whose load_dotenv does not yet implement PYTHON_DOTENV_DISABLED.
    if os.environ.get("TRADINGAGENTS_CODEX_DATA_WORKER") != "1":
        raise RuntimeError("Legacy data imports require the isolated worker")
    import dotenv

    dotenv.find_dotenv = lambda *args, **kwargs: ""
    dotenv.load_dotenv = lambda *args, **kwargs: False
    from tradingagents.dataflows.vendors import sec_edgar
    from tradingagents.dataflows.vendors.yahoo.fundamentals import get_fundamentals

    iso = as_of.isoformat()
    tools = [(_SPECS[0], lambda: _market_content(ticker, as_of))]
    if as_of == datetime.now(UTC).date():
        tools.append((_SPECS[1], lambda: get_fundamentals(ticker, as_of_date=iso)))
    tools.extend(
        [
            (_SPECS[2], lambda: sec_edgar.get_balance_sheet(ticker, as_of_date=iso)),
            (_SPECS[3], lambda: sec_edgar.get_income_statement(ticker, as_of_date=iso)),
            (_SPECS[4], lambda: sec_edgar.get_cashflow(ticker, as_of_date=iso)),
            (_SPECS[5], lambda: _news_content(ticker, as_of)),
        ]
    )
    return tuple(tools)


def _worker_main() -> None:
    """Private bounded JSONL protocol; not an agent-visible tool endpoint."""
    request = json.loads(sys.stdin.read(4096))
    ticker, as_of = _ticker(request["ticker"]), _as_of(date.fromisoformat(request["as_of"]))
    run_dir = Path(request["run_dir"]).resolve()
    # Save stdout for protocol messages; vendor prints/logs are never evidence.
    output = sys.stdout
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        tools = _legacy_tools(ticker, as_of)
        import yfinance as yf

        from tradingagents.dataflows.config import run_config

        yf.set_tz_cache_location(str(run_dir / "yfinance"))
        config = {
            "data_cache_dir": str(run_dir / "cache"),
            "results_dir": str(run_dir / "results"),
            "memory_log_path": str(run_dir / "unused-memory.md"),
            "news_article_limit": 12,
        }
        with run_config(config):
            for spec, reader in tools:
                try:
                    content = reader()
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty vendor result")
                    evidence = _item(spec, content, "available")
                except Exception as exc:  # Each source may fail independently.
                    evidence = _item(
                        spec,
                        f"Source unavailable ({type(exc).__name__}); no usable evidence was retrieved. This does not mean the underlying activity was absent.",
                        "unavailable",
                    )
                output.write(evidence.model_dump_json() + "\n")
                output.flush()
